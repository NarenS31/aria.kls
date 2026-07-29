"""
PEBBLE-extended rubric for ARIA evaluation.

Original PEBBLE dimensions (S, D, R, M, A) extended with 3 neurodivergent-specific dimensions:
  TC — Task Chunking
  SC — Session Consistency
  FA — Frustration Adaptation

Weights: S=0.20, D=0.20, R=0.20, M=0.10, A=0.05, TC=0.10, SC=0.10, FA=0.05
Solution dump penalty: γ=0.40 applied to weighted score when detected.

LLM-as-judge: llama3.1:8b, JSON output only.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import ollama


# ------------------------------------------------------------------
# Dimension definitions
# ------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "S": 0.20,  # Scaffolding
    "D": 0.20,  # Depth
    "R": 0.20,  # Responsiveness
    "M": 0.10,  # Metacognition
    "A": 0.05,  # Affect
    "TC": 0.10, # Task Chunking
    "SC": 0.10, # Session Consistency
    "FA": 0.05, # Frustration Adaptation
}

DIMENSIONS: Dict[str, Dict] = {
    "S": {
        "name": "Scaffolding",
        "description": "Does the tutor use Socratic questioning and hints rather than giving away answers?",
        "exemplars": {
            0: "Tutor immediately states the full answer or solution procedure with no questions.",
            1: "Tutor provides some hints but also gives away key steps unprompted.",
            2: "Tutor guides entirely through questions and targeted hints; student must reach conclusions.",
        },
    },
    "D": {
        "name": "Depth",
        "description": "Is the conceptual depth appropriate? Does it build genuine understanding, not just memorization?",
        "exemplars": {
            0: "Response is surface-level, procedural only, or factually incomplete.",
            1: "Response covers the concept but misses key nuance or doesn't connect ideas.",
            2: "Response builds conceptual understanding, connects to prior knowledge, appropriate depth.",
        },
    },
    "R": {
        "name": "Responsiveness",
        "description": "Does the tutor directly address the student's specific confusion or question?",
        "exemplars": {
            0: "Response is generic, ignores the student's specific confusion, or answers a different question.",
            1: "Response is mostly relevant but partially misses the student's actual confusion.",
            2: "Response precisely targets the student's expressed confusion with a tailored explanation.",
        },
    },
    "M": {
        "name": "Metacognition",
        "description": "Does the tutor teach the student how to think, monitor their own understanding, or self-check?",
        "exemplars": {
            0: "No metacognitive prompting; tutor just delivers content.",
            1: "One metacognitive prompt but not integrated into the explanation.",
            2: "Tutor explicitly teaches thinking strategy, self-check method, or error-detection approach.",
        },
    },
    "A": {
        "name": "Affect",
        "description": "Is the tutor emotionally attuned? Encouraging, warm, appropriately calibrated to student state?",
        "exemplars": {
            0: "Tutor is cold, dismissive, or ignores emotional content of student message.",
            1: "Tutor is neutral; technically correct but no emotional attunement.",
            2: "Tutor is warm, validating, and calibrates encouragement to the student's current state.",
        },
    },
    "TC": {
        "name": "Task Chunking",
        "description": "For ADHD: Does the tutor break tasks into micro-steps (≤3 sentences each), avoid walls of text?",
        "exemplars": {
            0: "Response is a wall of text (>5 sentences) or presents a multi-step task without chunking.",
            1: "Some chunking but steps are uneven, or one large block among smaller ones.",
            2: "Task is broken into clearly numbered micro-steps, each ≤3 sentences, whitespace between steps.",
        },
    },
    "SC": {
        "name": "Session Consistency",
        "description": "Does the tutor demonstrate memory of and correctly build on previous sessions?",
        "exemplars": {
            0: "Tutor treats student as a stranger with no history; ignores or contradicts prior session content.",
            1: "Tutor references past session but inaccurately or only superficially.",
            2: "Tutor correctly recalls prior session specifics and explicitly builds the current explanation on them.",
        },
    },
    "FA": {
        "name": "Frustration Adaptation",
        "description": "When student shows frustration, does the tutor detect it and appropriately shift tone/approach?",
        "exemplars": {
            0: "Tutor ignores clear frustration signals, continues same approach, or increases complexity.",
            1: "Tutor acknowledges frustration verbally but doesn't change the actual approach.",
            2: "Tutor detects frustration, validates it, and shifts to a simpler, more supportive approach.",
        },
    },
}

SOLUTION_DUMP_GAMMA = 0.40  # penalty multiplier when solution dump detected

# ------------------------------------------------------------------
# Solution dump detection (heuristic + LLM confirmation)
# ------------------------------------------------------------------

def _heuristic_solution_dump(tutor_msg: str) -> bool:
    """Fast pre-filter before calling LLM."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', tutor_msg) if s.strip()]
    if len(sentences) > 8:
        return True
    # Has numbered steps covering ≥4 steps without any question marks
    steps = re.findall(r'(?:step\s*\d+|^\d+\.)', tutor_msg, re.IGNORECASE | re.MULTILINE)
    if len(steps) >= 4 and '?' not in tutor_msg:
        return True
    return False


# ------------------------------------------------------------------
# LLM-as-judge
# ------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an expert evaluator of AI tutoring responses for neurodivergent students.
Score the tutor response on 8 dimensions using only 0, 1, or 2.
Return ONLY valid JSON. No explanation. No markdown. Just the JSON object.

Dimension definitions and exemplars:
""" + "\n".join(
    f'{dim} ({meta["name"]}): {meta["description"]}\n'
    f'  0={meta["exemplars"][0]}\n'
    f'  1={meta["exemplars"][1]}\n'
    f'  2={meta["exemplars"][2]}'
    for dim, meta in DIMENSIONS.items()
)

_JUDGE_SYSTEM += """

Output format (exactly this, with integer values 0, 1, or 2):
{"S": <int>, "D": <int>, "R": <int>, "M": <int>, "A": <int>, "TC": <int>, "SC": <int>, "FA": <int>, "solution_dump": <bool>}"""


def score_turn(
    student_msg: str,
    tutor_msg: str,
    conversation_history: List[Dict[str, str]],
    subject: str,
    model: str = "llama3.1:8b",
    max_retries: int = 3,
) -> Dict:
    """
    Score a single tutor turn on all 8 dimensions.
    Returns dict with raw scores, weighted_score, solution_dump flag.
    """
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in conversation_history[-6:]
    )

    user_prompt = f"""Subject: {subject}

Conversation history:
{history_text}

STUDENT: {student_msg}

TUTOR: {tutor_msg}

Score the TUTOR response on all 8 dimensions."""

    for attempt in range(max_retries):
        try:
            result = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.0, "num_predict": 150},
            )
            raw = result.message.content.strip()
            scores = json.loads(raw)

            # Validate all dimensions present
            for dim in DIMENSIONS:
                if dim not in scores:
                    scores[dim] = 1
                scores[dim] = max(0, min(2, int(scores[dim])))

            solution_dump = bool(scores.pop("solution_dump", False)) or _heuristic_solution_dump(tutor_msg)
            scores["solution_dump"] = solution_dump

            weighted = sum(
                scores[dim] * WEIGHTS[dim] for dim in WEIGHTS
            )
            if solution_dump:
                weighted = weighted * (1.0 - SOLUTION_DUMP_GAMMA)

            scores["weighted_score"] = round(weighted, 4)
            return scores

        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue

    # Fallback: return neutral scores
    fallback = {dim: 1 for dim in DIMENSIONS}
    fallback["solution_dump"] = _heuristic_solution_dump(tutor_msg)
    fallback["weighted_score"] = sum(WEIGHTS[d] * 1 for d in WEIGHTS)
    return fallback


def compute_weighted_score(scores: Dict[str, int]) -> float:
    """Compute weighted score from a raw scores dict (without solution_dump logic)."""
    return round(sum(scores.get(dim, 1) * w for dim, w in WEIGHTS.items()), 4)
