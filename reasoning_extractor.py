"""
Chain-of-thought reasoning extractor for ARIA evaluation.

For each tutoring episode and each model, this module:
1. Prompts the model to output structured JSON reasoning BEFORE the tutor response
2. Parses the reasoning trace separately from the final response
3. Scores the turn on all 8 rubric dimensions
4. Saves everything to data/reasoning_traces/{model}_{episode_id}.jsonl

Models evaluated (6 + distilled):
  Local:    llama3.1:8b, mistral:7b, gemma2:9b
  API:      gpt-4o, gemini-2.5-flash
  Distilled: aria_distilled
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.api_clients import query_model, is_available, ALL_MODELS
from eval.rubric import score_turn, WEIGHTS

DATA_DIR = Path(__file__).parent.parent / "data"
TRACES_DIR = DATA_DIR / "reasoning_traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

ALL_EVAL_MODELS = ["llama3.1:8b", "mistral:7b", "gemma2:9b", "gpt-4o", "gemini-2.5-flash", "aria_distilled"]


# ------------------------------------------------------------------
# Reasoning extraction prompt
# ------------------------------------------------------------------

_REASONING_SYSTEM = """You are an expert AI tutor evaluating a student who has ADHD.

Before responding to the student, output your reasoning in this exact JSON format (no other text before it):
{{
  "student_state": "what misconception or confusion state is the student in right now",
  "planned_approach": "what tutoring strategy will you use and exactly why",
  "adhd_considerations": "what specific ADHD accommodations are you making for this student",
  "predicted_response": "how do you expect the student to respond to your tutor message",
  "confidence": 0.85
}}

Then, on a new line after the JSON, output ONLY your actual tutor response.

CRITICAL RULES for tutor response:
- Maximum 3 sentences per paragraph
- Number ALL multi-step tasks as Step 1, Step 2, ...
- Never reveal the complete answer before diagnosing the student's confusion
- If the student shows frustration, explicitly validate it before redirecting
- If you have context about prior sessions, reference it specifically

Student profile context:
{profile_context}

Prior conversation context:
{memory_context}"""


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

_JSON_PATTERN = re.compile(r'\{[^{}]*"student_state"[^{}]*\}', re.DOTALL)
_REQUIRED_KEYS = {"student_state", "planned_approach", "adhd_considerations", "predicted_response", "confidence"}


def _parse_reasoning_and_response(raw: str) -> Tuple[Optional[dict], str]:
    """
    Split model output into (reasoning_dict, tutor_response_text).
    The model is instructed to output JSON first, then the response.
    """
    # Try to find JSON block
    match = _JSON_PATTERN.search(raw)
    if match:
        json_str = match.group(0)
        try:
            reasoning = json.loads(json_str)
            # Validate required keys
            for k in _REQUIRED_KEYS:
                reasoning.setdefault(k, "")
            reasoning["confidence"] = float(reasoning.get("confidence", 0.5))
            # Response is everything after the JSON block
            response_text = raw[match.end():].strip()
            if not response_text:
                response_text = raw[:match.start()].strip()
            return reasoning, response_text
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract JSON with json.loads on first {...} block
    brace_start = raw.find("{")
    brace_end = raw.rfind("}") + 1
    if brace_start != -1 and brace_end > brace_start:
        try:
            reasoning = json.loads(raw[brace_start:brace_end])
            if all(k in reasoning for k in {"student_state", "planned_approach"}):
                response_text = (raw[:brace_start] + raw[brace_end:]).strip()
                reasoning.setdefault("confidence", 0.5)
                return reasoning, response_text or raw
        except (json.JSONDecodeError, ValueError):
            pass

    # Could not parse reasoning — return None reasoning, full text as response
    return None, raw.strip()


def _build_profile_context(profile: dict) -> str:
    name = profile.get("name", "the student")
    diag = profile.get("diagnosis", "ADHD")
    style = profile.get("learning_style", "step_by_step")
    subjs = ", ".join(profile.get("subjects", []))
    triggers = profile.get("frustration_triggers", {})
    trigger_text = "; ".join(f"{s}: {', '.join(t)}" for s, t in triggers.items())
    return (
        f"Student: {name}, {diag}\n"
        f"Learning style preference: {style}\n"
        f"Studying: {subjs}\n"
        f"Frustration triggers: {trigger_text}"
    )


# ------------------------------------------------------------------
# Single-turn extraction
# ------------------------------------------------------------------

def extract_turn(
    model: str,
    student_msg: str,
    conversation_history: List[Dict[str, str]],
    profile: dict,
    memory_context: str,
    subject: str,
    episode_id: str,
    turn_idx: int,
    persona: str,
) -> dict:
    """
    Run one tutoring turn with reasoning extraction.
    Returns a full trace record.
    """
    profile_context = _build_profile_context(profile)

    system_prompt = _REASONING_SYSTEM.format(
        profile_context=profile_context,
        memory_context=memory_context or "No prior session context.",
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-8:])
    messages.append({"role": "user", "content": student_msg})

    # Query with generous token budget since we need JSON + response
    raw = query_model(model, messages, temperature=0.7, max_tokens=700)
    reasoning, tutor_response = _parse_reasoning_and_response(raw)

    # Score the tutor response
    scores = score_turn(
        student_msg=student_msg,
        tutor_msg=tutor_response,
        conversation_history=conversation_history,
        subject=subject,
        model="llama3.1:8b",  # always use local judge
    )

    record = {
        "episode_id": episode_id,
        "model": model,
        "turn": turn_idx,
        "student_message": student_msg,
        "reasoning": reasoning or {
            "student_state": "unknown",
            "planned_approach": "unknown",
            "adhd_considerations": "unknown",
            "predicted_response": "unknown",
            "confidence": 0.0,
        },
        "tutor_response": tutor_response,
        "rubric_scores": {k: scores[k] for k in WEIGHTS},
        "weighted_score": scores["weighted_score"],
        "solution_dump": scores.get("solution_dump", False),
        "reasoning_extracted": reasoning is not None,
        "profile_id": profile.get("profile_id", "unknown"),
        "persona": persona,
        "subject": subject,
        "timestamp": datetime.now().isoformat(),
    }
    return record


# ------------------------------------------------------------------
# Episode runner with reasoning extraction
# ------------------------------------------------------------------

def run_episode_with_reasoning(
    model: str,
    persona_name: str,
    subject_name: str,
    profile: dict,
    memory_context: str = "",
    seed_idx: int = 0,
    output_file: Optional[Path] = None,
) -> List[dict]:
    """
    Run one full episode extracting reasoning at every turn.
    Returns list of trace records.
    """
    from eval.simulator import SUBJECTS, SimulatedStudent

    episode_id = str(uuid.uuid4())
    student = SimulatedStudent(persona_name, subject_name, seed_idx)
    conversation: List[Dict[str, str]] = []
    records = []

    # Opening turn
    student_msg = student.opening_message()

    f = open(output_file, "a") if output_file else None

    turn_idx = 0
    while True:
        record = extract_turn(
            model=model,
            student_msg=student_msg,
            conversation_history=conversation,
            profile=profile,
            memory_context=memory_context,
            subject=subject_name,
            episode_id=episode_id,
            turn_idx=turn_idx,
            persona=persona_name,
        )
        records.append(record)
        if f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        tutor_response = record["tutor_response"]
        conversation.append({"role": "user", "content": student_msg})
        conversation.append({"role": "assistant", "content": tutor_response})

        if student.is_done():
            break

        student_msg = student.next_message(conversation)
        conversation.append({"role": "user", "content": student_msg})
        # remove the extra entry we just added (next_message expects conv without this turn)
        conversation.pop()
        turn_idx += 1

    if f:
        f.close()
    return records


# ------------------------------------------------------------------
# Multi-model episode runner
# ------------------------------------------------------------------

def run_all_models_on_episode(
    persona_name: str,
    subject_name: str,
    profile: dict,
    memory_context: str = "",
    seed_idx: int = 0,
    models: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """
    Run one episode across all available models.
    Returns {model_name: [records]}.
    """
    if models is None:
        models = [m for m in ALL_EVAL_MODELS if is_available(m)]

    results = {}
    for model in models:
        if verbose:
            print(f"    [{model}]", end=" ", flush=True)
        output_file = TRACES_DIR / f"{model.replace(':', '_').replace('/', '_')}_{persona_name}_{subject_name}.jsonl"
        try:
            records = run_episode_with_reasoning(
                model=model,
                persona_name=persona_name,
                subject_name=subject_name,
                profile=profile,
                memory_context=memory_context,
                seed_idx=seed_idx,
                output_file=output_file,
            )
            results[model] = records
            if verbose:
                ws = sum(r["weighted_score"] for r in records) / max(len(records), 1)
                print(f"ws={ws:.3f} turns={len(records)}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results[model] = []
    return results


# ------------------------------------------------------------------
# Load traces
# ------------------------------------------------------------------

def load_traces(model: Optional[str] = None) -> List[dict]:
    """Load all reasoning trace records, optionally filtered by model."""
    records = []
    pattern = f"{model.replace(':', '_')}*.jsonl" if model else "*.jsonl"
    for path in TRACES_DIR.glob(pattern):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records
