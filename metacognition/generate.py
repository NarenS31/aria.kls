#!/usr/bin/env python3.11
"""
Generate a large synthetic dataset of student think-aloud responses across
subjects and cognitive states, for training and evaluating metacognition models.

Each sample pairs a student profile + problem with a three-part think-aloud
(pre / during / post attempt) that reflects one of seven cognitive states,
plus ground-truth metacognitive labels.

Generation is done with a local Ollama model (default: llama3.1:8b), run in
parallel batches. Progress is saved incrementally so a crash never loses work,
and `--resume` continues from wherever the last run stopped.

Usage:
    python3.11 metacognition/generate.py --samples 50
    python3.11 metacognition/generate.py --samples 200
    python3.11 metacognition/generate.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, replace
from typing import Any

import ollama
from tqdm import tqdm


# ------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic_thinkaloud")

DATASET_PATH = os.path.join(OUT_DIR, "dataset.jsonl")
STATS_PATH = os.path.join(OUT_DIR, "dataset_stats.json")
TRAIN_PATH = os.path.join(OUT_DIR, "train.jsonl")
VAL_PATH = os.path.join(OUT_DIR, "val.jsonl")
TEST_PATH = os.path.join(OUT_DIR, "test.jsonl")
FAILURE_LOG_PATH = os.path.join(OUT_DIR, "failures.log")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "checkpoint.json")

DEFAULT_MODEL = os.environ.get("ARIA_GEN_MODEL", "llama3.1:8b")
# Concurrency + robustness. BATCH_SIZE and the per-request timeout are env-tunable
# so a run can be made a politer GPU citizen when another job shares the machine.
# The timeout is CRITICAL: without it a saturated/stuck Ollama makes ollama.chat
# block forever, hanging the whole run (observed: a 10-hour zero-progress hang
# while a second process held the single-slot GPU). With it, a stalled request
# fails, is retried MAX_RETRIES times, then skipped and logged.
BATCH_SIZE = int(os.environ.get("ARIA_GEN_BATCH", "10"))
MAX_RETRIES = 3
CHECKPOINT_EVERY = 50
GEN_TIMEOUT = float(os.environ.get("ARIA_GEN_TIMEOUT", "300"))

_client: "ollama.Client | None" = None
_client_lock = threading.Lock()


def _gen_client() -> "ollama.Client":
    """A shared Ollama client with a finite request timeout (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = ollama.Client(timeout=GEN_TIMEOUT)
    return _client

# ------------------------------------------------------------------
# Multi-generator augmentation (reduce single-generator bias)
# ------------------------------------------------------------------
# The alternative generators used to diversify the corpus, and a short filesystem
# tag for each (used in output filenames + as an id suffix so merged records stay
# unique across generators).
GENERATOR_TAGS = {
    "llama3.1:8b": "llama",
    "mistral:7b": "mistral",
    "gemma2:9b": "gemma2",
    "phi3:medium": "phi3",
}
AUGMENT_MODELS = ["mistral:7b", "gemma2:9b", "phi3:medium"]
# All four generators participate in the FLOW-specific augmentation (Part 5).
FLOW_AUGMENT_MODELS = ["llama3.1:8b", "mistral:7b", "gemma2:9b", "phi3:medium"]

# The FLOW hard-negative decoy states: think-alouds written to look like FLOW but
# truly PLANNING (deliberate setup) or INSIGHT (sudden realization).
FLOW_DECOY_STATES = ["PLANNING", "INSIGHT"]


def _tag(model: str) -> str:
    return GENERATOR_TAGS.get(model, re.sub(r"[^a-z0-9]+", "-", model.lower()))


def augment_path(model: str) -> str:
    return os.path.join(OUT_DIR, f"augment_{_tag(model)}.jsonl")


MIXED_DATASET_PATH = os.path.join(OUT_DIR, "dataset_mixed.jsonl")
MIXED_TRAIN_PATH = os.path.join(OUT_DIR, "mixed_train.jsonl")
MIXED_VAL_PATH = os.path.join(OUT_DIR, "mixed_val.jsonl")
MIXED_TEST_PATH = os.path.join(OUT_DIR, "mixed_test.jsonl")
FLOW_HARD_NEG_PATH = os.path.join(OUT_DIR, "flow_hard_negatives.jsonl")
MIXED_FEWSHOT_PATH = os.path.join(REPO_ROOT, "data", "eval", "mixed_fewshot_examples.json")

COGNITIVE_STATES = [
    "PLANNING", "FLOW", "CONFUSED", "RUSHING",
    "FRUSTRATED", "STUCK", "INSIGHT",
]

DIFFICULTIES = ["easy", "medium", "hard"]

ADHD_PROFILES = ["inattentive", "hyperactive", "combined"]

# subject -> list of (topic, problem_text) tuples, spread across difficulty.
# The generator picks a topic per sample; difficulty is applied independently.
SUBJECTS: dict[str, list[tuple[str, str]]] = {
    "ACT Math": [
        ("quadratic equations", "If x² + 5x + 6 = 0, what are the values of x?"),
        ("linear systems", "Solve for x and y: 2x + y = 7 and x - y = 2."),
        ("percentages", "A shirt costs $40 after a 20% discount. What was the original price?"),
        ("geometry basics", "A right triangle has legs of length 6 and 8. What is the hypotenuse?"),
        ("exponents", "Simplify (3² · 3⁴) / 3³."),
    ],
    "ACT English": [
        ("comma usage", "Choose the best punctuation: 'After the game we went home tired but happy.'"),
        ("subject-verb agreement", "Pick the correct verb: 'Each of the students (is/are) responsible.'"),
        ("sentence structure", "Which revision best fixes the run-on: 'It was raining we stayed inside.'"),
        ("word choice", "Choose the correct word: 'Their/There/They're going to the store.'"),
        ("redundancy", "Trim the redundancy: 'She returned back to the same identical spot.'"),
    ],
    "ACT Reading": [
        ("main idea", "A passage describes a town's decline after a factory closes. What is the main idea?"),
        ("inference", "The author says the sky 'wept all afternoon.' What does this imply about the weather?"),
        ("author's tone", "The narrator calls the reunion 'a gauntlet of forced smiles.' What is the tone?"),
        ("vocabulary in context", "In context, 'The plan was untenable' most nearly means the plan was ___?"),
        ("supporting detail", "Which detail best supports the claim that the character felt isolated?"),
    ],
    "ACT Science": [
        ("data interpretation", "A graph shows plant growth doubling each week. Predict week 4 if week 1 is 2 cm."),
        ("experimental design", "A study changes both temperature and light. Why is the result hard to interpret?"),
        ("hypothesis testing", "Results contradict a hypothesis. What is the most reasonable next step?"),
        ("reading tables", "A table lists reaction rates at 3 temperatures. Which temperature is fastest?"),
        ("scientific reasoning", "Two scientists disagree on a cause. What evidence would resolve it?"),
    ],
    "Algebra": [
        ("solving for x", "Solve: 3(x - 4) = 2x + 5."),
        ("factoring", "Factor completely: x² - 9."),
        ("inequalities", "Solve and graph: 2x - 3 < 7."),
        ("functions", "If f(x) = 2x + 1, what is f(3)?"),
        ("slope", "Find the slope of the line through (1, 2) and (4, 11)."),
    ],
    "Geometry": [
        ("area", "Find the area of a circle with radius 5."),
        ("angles", "Two angles are complementary. One is 35°. What is the other?"),
        ("triangles", "A triangle has angles 40° and 75°. What is the third angle?"),
        ("volume", "Find the volume of a cube with side length 4."),
        ("pythagorean theorem", "A ladder leans on a wall, base 3 ft out, top 4 ft up. How long is the ladder?"),
    ],
    "Biology": [
        ("cell structure", "What organelle is responsible for producing energy in a cell?"),
        ("photosynthesis", "What are the reactants and products of photosynthesis?"),
        ("genetics", "If both parents are Bb, what fraction of offspring are expected to be bb?"),
        ("evolution", "How does natural selection lead to changes in a population over time?"),
        ("ecosystems", "What happens to a food web if the primary producer population collapses?"),
    ],
    "Chemistry": [
        ("balancing equations", "Balance: H₂ + O₂ → H₂O."),
        ("stoichiometry", "How many moles of water form from 4 moles of H₂ (excess O₂)?"),
        ("periodic trends", "Which is larger, a sodium atom or a chlorine atom? Why?"),
        ("acids and bases", "What is the pH of a neutral solution at 25°C?"),
        ("molar mass", "What is the molar mass of CO₂?"),
    ],
    "US History": [
        ("causes of the revolution", "Why did 'taxation without representation' anger the colonists?"),
        ("the constitution", "What problem did the system of checks and balances aim to solve?"),
        ("civil war", "What was a central economic difference between the North and South before the war?"),
        ("civil rights movement", "What was the significance of Brown v. Board of Education?"),
        ("industrialization", "How did railroads change the American economy in the late 1800s?"),
    ],
    "Python Programming": [
        ("loops", "Write a loop that prints the numbers 1 through 5."),
        ("functions", "Write a function that returns the square of a number."),
        ("lists", "How do you find the largest number in a list called nums?"),
        ("conditionals", "Write code that prints 'even' or 'odd' for a variable n."),
        ("strings", "How do you reverse the string s in Python?"),
    ],
}

# State-appropriate marker guidance injected into the prompt.
STATE_MARKERS = {
    "CONFUSED": "wait, huh, I don't get it, circular reasoning, second-guessing",
    "RUSHING": "fast, skips steps, 'probably', 'whatever', careless jumps",
    "PLANNING": "'first I need to', 'let me think', deliberate step-by-step setup",
    "FRUSTRATED": "ugh, 'this makes no sense', 'forget it', irritation",
    "STUCK": "'I don't know' repeated, [pause], mental breakdown of progress",
    "FLOW": "smooth, step by step, confident transitions, momentum",
    "INSIGHT": "'OH', 'wait', 'ohhhh', sudden clarity, a click moment",
}

# Ground-truth labels are derived deterministically from the cognitive state so
# they are consistent regardless of model output noise.
STATE_GROUND_TRUTH = {
    "PLANNING":   {"planning_detected": True,  "self_correction": False, "insight_moment": False, "gave_up": False},
    "FLOW":       {"planning_detected": False, "self_correction": False, "insight_moment": False, "gave_up": False},
    "CONFUSED":   {"planning_detected": False, "self_correction": True,  "insight_moment": False, "gave_up": False},
    "RUSHING":    {"planning_detected": False, "self_correction": False, "insight_moment": False, "gave_up": False},
    "FRUSTRATED": {"planning_detected": False, "self_correction": False, "insight_moment": False, "gave_up": False},
    "STUCK":      {"planning_detected": False, "self_correction": False, "insight_moment": False, "gave_up": True},
    "INSIGHT":    {"planning_detected": False, "self_correction": True,  "insight_moment": True,  "gave_up": False},
}

# Ground-truth self-initiated metacognition, derived deterministically from the
# cognitive state. In a standalone think-aloud (no ARIA prompt) these are the
# states whose voice contains DISTINCTIVE, separable self-initiated metacognitive
# language:
#   PLANNING -> deliberate setup ("first I need to…", "let me think…", "my plan")
#   INSIGHT  -> a spontaneous realization ("ohhh", "it clicks", "the reason…")
# The other states (FLOW narration; CONFUSED/RUSHING/FRUSTRATED/STUCK) share the
# same messy "no wait / or is it" doubt across the board, so generic monitoring
# language is NOT a reliable self-initiation signal in this corpus — labelling
# them positive would only add noise. These labels train + evaluate the
# TransferDetector (eval Metric 4).
STATE_SELF_INITIATED = {
    "PLANNING":   (True,  "planning"),
    "FLOW":       (False, "none"),
    "CONFUSED":   (False, "none"),
    "RUSHING":    (False, "none"),
    "FRUSTRATED": (False, "none"),
    "STUCK":      (False, "none"),
    "INSIGHT":    (True,  "reflection"),
}


def confidence_from_subject(subject_confidence: float) -> int:
    """Map a 0-1 subject confidence onto a 1-5 pre-attempt confidence rating."""
    return max(1, min(5, int(round(1 + float(subject_confidence) * 4))))


def metacognition_labels(state: str, subject_confidence: float, correct: bool) -> dict[str, Any]:
    """Deterministic per-sample metacognition ground-truth fields (spec §7)."""
    si, mtype = STATE_SELF_INITIATED.get(state, (False, "none"))
    return {
        "self_initiated_metacognition": si,
        "metacognitive_type": mtype,
        "confidence_before": confidence_from_subject(subject_confidence),
        "correct": bool(correct),
    }


def ensure_meta_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Add (or refresh) the metacognition ground-truth fields on a sample.

    Derived purely from existing content (cognitive_state, subject_confidence,
    correct_answer), so it backfills older samples without any model calls. The
    fields are recomputed deterministically every time, so the corpus always
    reflects the current STATE_SELF_INITIATED mapping.
    """
    state = record.get("cognitive_state", "FLOW")
    conf = record.get("student_profile", {}).get("subject_confidence", 0.5)
    correct = record.get("correct_answer", record.get("correct", False))
    record.update(metacognition_labels(state, conf, correct))
    # Backfill the generator provenance on older samples (pre-multi-generator).
    record.setdefault("generator_model", DEFAULT_MODEL)
    return record


# ------------------------------------------------------------------
# Sample specification
# ------------------------------------------------------------------

@dataclass
class SampleSpec:
    """Everything needed to generate one sample, plus a stable identity key."""
    cognitive_state: str
    subject: str
    difficulty: str
    adhd_type: str
    topic: str
    problem_text: str
    grade: int
    subject_confidence: float
    dup_index: int = 0

    @property
    def key(self) -> str:
        # Stable identity used for de-duplication / resume. When the target
        # exceeds the unique combinatorial grid, combos are cycled and each
        # repeat carries a dup_index so its key stays unique. dup_index == 0
        # produces the original key, keeping older records backward-compatible.
        parts = [
            self.cognitive_state, self.subject, self.difficulty,
            self.adhd_type, self.topic,
            f"{self.subject_confidence:.2f}",
        ]
        if self.dup_index:
            parts.append(f"#{self.dup_index}")
        return "|".join(parts)


def build_specs(samples_per_state: int, seed: int = 1234) -> list[SampleSpec]:
    """Build the full list of sample specs, evenly distributed across states.

    We iterate the full combinatorial grid (state x subject x difficulty x
    adhd profile x topic) deterministically and take `samples_per_state`
    specs for each cognitive state.
    """
    rng = random.Random(seed)
    specs: list[SampleSpec] = []

    for state in COGNITIVE_STATES:
        combos: list[SampleSpec] = []
        for subject, topics in SUBJECTS.items():
            for difficulty in DIFFICULTIES:
                for adhd_type in ADHD_PROFILES:
                    for topic, problem_text in topics:
                        # Confidence correlates loosely with difficulty.
                        base = {"easy": 0.65, "medium": 0.45, "hard": 0.3}[difficulty]
                        conf = round(min(0.95, max(0.05, base + rng.uniform(-0.15, 0.15))), 2)
                        combos.append(SampleSpec(
                            cognitive_state=state,
                            subject=subject,
                            difficulty=difficulty,
                            adhd_type=adhd_type,
                            topic=topic,
                            problem_text=problem_text,
                            grade=11,
                            subject_confidence=conf,
                        ))
        rng.shuffle(combos)
        # If the caller asks for more samples than unique combos, we allow
        # repeats by cycling — but each still gets a unique key via suffix.
        if samples_per_state <= len(combos):
            chosen = combos[:samples_per_state]
        else:
            chosen = []
            i = 0
            while len(chosen) < samples_per_state:
                base_spec = combos[i % len(combos)]
                cycle = i // len(combos)
                chosen.append(base_spec if cycle == 0
                              else replace(base_spec, dup_index=cycle))
                i += 1
        specs.extend(chosen)

    rng.shuffle(specs)
    return specs


# ------------------------------------------------------------------
# Prompt construction & generation
# ------------------------------------------------------------------

def build_prompt(spec: SampleSpec) -> str:
    confidence_pct = int(round(spec.subject_confidence * 100))
    markers = STATE_MARKERS[spec.cognitive_state]
    return f"""Generate a realistic think-aloud response from a grade {spec.grade} \
student with ADHD-{spec.adhd_type} who is {confidence_pct}% confident in \
{spec.subject}, attempting this {spec.difficulty} problem:

{spec.problem_text}

The student is in a {spec.cognitive_state} state.

Generate three parts:
1. PRE_ATTEMPT: what they say/think before starting (1-3 sentences, authentic ADHD voice)
2. DURING_ATTEMPT: their reasoning while solving (2-5 sentences, include false starts, corrections, uncertainty markers appropriate to state)
3. POST_ATTEMPT: their reflection after (1-2 sentences)

Make it sound like a real student, not a textbook.
Include state-appropriate markers for {spec.cognitive_state}: {markers}

Output JSON only, with exactly these keys and no extra text:
{{
  "pre_attempt": "...",
  "during_attempt": "...",
  "post_attempt": "...",
  "correct": true
}}"""


def build_hard_negative_prompt(spec: SampleSpec, decoy_state: str) -> str:
    """Prompt for a FLOW hard negative: reads like FLOW but is truly PLANNING/INSIGHT.

    Uses the same three-part think-aloud contract as build_prompt so the samples
    are schema-identical to the rest of the corpus."""
    confidence_pct = int(round(spec.subject_confidence * 100))
    behavior = {
        "PLANNING": "deliberate setup before starting — laying out a plan, listing "
                    "what they need, or choosing an approach — rather than smooth "
                    "continuous progress",
        "INSIGHT": "a sudden realization after confusion — a click moment where it "
                   "finally makes sense — rather than smooth continuous progress",
    }[decoy_state]
    return f"""Generate a realistic think-aloud response from a grade {spec.grade} \
student with ADHD-{spec.adhd_type} who is {confidence_pct}% confident in \
{spec.subject}, attempting this {spec.difficulty} problem:

{spec.problem_text}

Generate a student think-aloud that could be MISTAKEN for FLOW (engaged, \
progressing) but is actually {decoy_state}. The student should show {behavior}. \
Keep the surface engaged and forward-moving (so it superficially resembles flow), \
but the true underlying cognitive state is {decoy_state}. Label: {decoy_state}.

Generate three parts:
1. PRE_ATTEMPT: what they say/think before starting (1-3 sentences, authentic ADHD voice)
2. DURING_ATTEMPT: their reasoning while solving (2-5 sentences)
3. POST_ATTEMPT: their reflection after (1-2 sentences)

Make it sound like a real student, not a textbook.

Output JSON only, with exactly these keys and no extra text:
{{
  "pre_attempt": "...",
  "during_attempt": "...",
  "post_attempt": "...",
  "correct": true
}}"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("no JSON object found in model output")
    return json.loads(m.group(0))


def generate_one(spec: SampleSpec, model: str, *, prompt: str | None = None,
                 labelled_state: str | None = None) -> dict[str, Any]:
    """Generate a single sample dict. Raises on unrecoverable failure.

    `prompt` overrides the default per-state prompt (used for hard negatives);
    `labelled_state` overrides the ground-truth state written into the record."""
    prompt = prompt if prompt is not None else build_prompt(spec)
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            result = _gen_client().chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are simulating a real grade-11 student's inner voice while they work a problem. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.9, "num_predict": 400},
            )
            raw = result.message.content
            parsed = _extract_json(raw)

            pre = str(parsed.get("pre_attempt", "")).strip()
            during = str(parsed.get("during_attempt", "")).strip()
            post = str(parsed.get("post_attempt", "")).strip()
            if not (pre and during and post):
                raise ValueError("missing one or more think-aloud parts")

            correct = parsed.get("correct", None)
            if isinstance(correct, str):
                correct = correct.strip().lower() in ("true", "yes", "1", "correct")
            if not isinstance(correct, bool):
                # If the model omitted it, infer plausibly from state.
                correct = spec.cognitive_state in ("FLOW", "INSIGHT", "PLANNING")

            return _assemble(spec, pre, during, post, correct, model,
                             labelled_state=labelled_state)
        except Exception as e:  # noqa: BLE001 — retry any generation error
            last_err = e
            continue

    raise RuntimeError(f"generation failed after {MAX_RETRIES} attempts: {last_err}")


def _assemble(spec: SampleSpec, pre: str, during: str, post: str, correct: bool,
              model: str = DEFAULT_MODEL, *, labelled_state: str | None = None) -> dict[str, Any]:
    """Assemble a dataset record.

    `labelled_state` overrides the ground-truth cognitive_state independently of
    the state whose *voice* the prompt asked for — used for FLOW hard negatives,
    where the think-aloud is written to look like FLOW but is truly PLANNING or
    INSIGHT. When omitted the labelled state equals spec.cognitive_state.
    `model` is recorded as `generator_model` for generator-bias analysis.
    """
    state = labelled_state or spec.cognitive_state
    record = {
        "id": spec.key,
        "student_profile": {
            "adhd_type": spec.adhd_type,
            "grade": spec.grade,
            "subject_confidence": spec.subject_confidence,
        },
        "problem": {
            "subject": spec.subject,
            "topic": spec.topic,
            "difficulty": spec.difficulty,
            "text": spec.problem_text,
        },
        "think_aloud": {
            "pre_attempt": pre,
            "during_attempt": during,
            "post_attempt": post,
        },
        "cognitive_state": state,
        "ground_truth": dict(STATE_GROUND_TRUTH[state]),
        "correct_answer": bool(correct),
        "generator_model": model,
    }
    # Metacognition ground-truth fields for the three measurement systems (§7).
    record.update(metacognition_labels(state, spec.subject_confidence, correct))
    return record


# ------------------------------------------------------------------
# Incremental persistence
# ------------------------------------------------------------------

class IncrementalWriter:
    """Append-only JSONL writer, flushed after every sample, thread-safe."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def load_existing_keys(path: str) -> set[str]:
    keys: set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("id")
            if k:
                keys.add(k)
    return keys


def load_all(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def log_failure(spec: SampleSpec, err: Exception) -> None:
    os.makedirs(os.path.dirname(FAILURE_LOG_PATH), exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{spec.key}\t{type(err).__name__}: {err}\n")


def save_checkpoint(
    completed_keys: set[str],
    *,
    model: str,
    seed: int,
    samples_per_state: int,
    generated_this_run: int,
    failed_this_run: int,
) -> None:
    """Atomically snapshot progress so --resume knows exactly what is done.

    dataset.jsonl remains the authoritative per-sample record (it is fsync'd
    after every write); this file is a compact, human-readable progress
    summary written every CHECKPOINT_EVERY samples and at run end.
    """
    payload = {
        "model": model,
        "seed": seed,
        "samples_per_state": samples_per_state,
        "total_completed": len(completed_keys),
        "generated_this_run": generated_this_run,
        "failed_this_run": failed_this_run,
        "completed_keys": sorted(completed_keys),
    }
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, CHECKPOINT_PATH)  # atomic; never leaves a half-written file


def load_checkpoint_keys() -> set[str]:
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return set()
    return set(data.get("completed_keys", []))


# ------------------------------------------------------------------
# Downstream artifacts: stats + stratified split
# ------------------------------------------------------------------

def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_state = Counter(r["cognitive_state"] for r in records)
    by_subject = Counter(r["problem"]["subject"] for r in records)
    by_difficulty = Counter(r["problem"]["difficulty"] for r in records)
    by_adhd = Counter(r["student_profile"]["adhd_type"] for r in records)
    by_metacog_type = Counter(r.get("metacognitive_type", "none") for r in records)
    self_initiated = sum(1 for r in records if r.get("self_initiated_metacognition"))
    correct = sum(1 for r in records if r["correct_answer"])

    def avg_len(field_name: str) -> float:
        if not records:
            return 0.0
        total = sum(len(r["think_aloud"][field_name].split()) for r in records)
        return round(total / len(records), 2)

    return {
        "total_samples": len(records),
        "by_cognitive_state": dict(sorted(by_state.items())),
        "by_subject": dict(sorted(by_subject.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_adhd_type": dict(sorted(by_adhd.items())),
        "by_metacognitive_type": dict(sorted(by_metacog_type.items())),
        "self_initiated_rate": round(self_initiated / len(records), 3) if records else 0.0,
        "correct_answer_rate": round(correct / len(records), 3) if records else 0.0,
        "avg_words": {
            "pre_attempt": avg_len("pre_attempt"),
            "during_attempt": avg_len("during_attempt"),
            "post_attempt": avg_len("post_attempt"),
        },
    }


def stratified_split(
    records: list[dict[str, Any]], seed: int = 7
) -> tuple[list, list, list]:
    """Split 80/10/10 with equal cognitive-state distribution in each split."""
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[r["cognitive_state"]].append(r)

    train, val, test = [], [], []
    for state, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        # Guarantee at least one in val/test when there is enough data.
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def finalize(model: str) -> None:
    """Recompute stats + splits from the full dataset on disk.

    Backfills the deterministic metacognition ground-truth fields onto every
    record (older samples predate them) and rewrites dataset.jsonl so the whole
    corpus — and the train/val/test splits derived from it — carries the fields
    the three measurement systems need.
    """
    records = load_all(DATASET_PATH)
    if not records:
        print("No records to finalize.")
        return

    missing = sum(1 for r in records if "self_initiated_metacognition" not in r)
    records = [ensure_meta_fields(r) for r in records]
    # Rewrite the corpus so every sample carries the (deterministic) metacognition
    # fields under the current STATE_SELF_INITIATED mapping.
    write_jsonl(DATASET_PATH, records)
    if missing:
        print(f"Added metacognition fields to {missing} samples missing them.")
    print(f"Refreshed metacognition ground-truth fields on {len(records)} samples.")

    stats = compute_stats(records)
    stats["model"] = model
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    train, val, test = stratified_split(records)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_jsonl(TEST_PATH, test)

    print(f"\nFinalized {len(records)} samples:")
    print(f"  stats  -> {STATS_PATH}")
    print(f"  train  -> {TRAIN_PATH} ({len(train)})")
    print(f"  val    -> {VAL_PATH} ({len(val)})")
    print(f"  test   -> {TEST_PATH} ({len(test)})")
    print("  state distribution:", stats["by_cognitive_state"])


# ------------------------------------------------------------------
# Multi-generator augmentation orchestration
# ------------------------------------------------------------------

def _atomic_write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    """Write JSONL via temp + rename so concurrent readers never see a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _generate_jobs(jobs: list[tuple[str, Any]], out_path: str, desc: str) -> int:
    """Run (final_id, callable()->record) jobs in parallel batches, appending to
    out_path. Skips ids already present in out_path (resume-safe). The callable's
    returned record has its id overwritten with final_id. Returns new count."""
    existing = load_existing_keys(out_path)
    todo = [(fid, fn) for (fid, fn) in jobs if fid not in existing]
    if not todo:
        print(f"{desc}: nothing to do ({len(existing)} already present in "
              f"{os.path.basename(out_path)}).")
        return 0

    writer = IncrementalWriter(out_path)
    generated = 0
    failed = 0
    try:
        with tqdm(total=len(todo), desc=desc, unit="sample") as bar:
            for start in range(0, len(todo), BATCH_SIZE):
                batch = todo[start:start + BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(fn): fid for (fid, fn) in batch}
                    for fut in as_completed(futures):
                        fid = futures[fut]
                        try:
                            record = fut.result()
                            record["id"] = fid
                            writer.write(record)
                            generated += 1
                        except Exception as e:  # noqa: BLE001
                            failed += 1
                            with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as fh:
                                fh.write(f"{fid}\t{type(e).__name__}: {e}\n")
                        finally:
                            bar.update(1)
                            bar.set_postfix(ok=generated, fail=failed)
    finally:
        writer.close()
    if failed:
        print(f"\n{failed} generation(s) failed and were skipped "
              f"(logged to {FAILURE_LOG_PATH}).")
    return generated


def run_augmentation(model: str, samples_per_state: int, seed: int = 1234) -> int:
    """Generate `samples_per_state` per cognitive state with `model`, appended to
    that model's augment file. Uses the EXACT same per-state prompts as the base
    corpus; only the generator changes. Resume-safe."""
    out_path = augment_path(model)
    tag = _tag(model)
    check_model(model)
    specs = build_specs(samples_per_state, seed=seed)  # same grid → controlled comparison
    jobs = [(f"{s.key}|{tag}", (lambda sp=s: generate_one(sp, model))) for s in specs]
    print(f"\n=== Augmenting with {model} "
          f"({samples_per_state}/state x {len(COGNITIVE_STATES)} states "
          f"= {len(jobs)} target) -> {os.path.basename(out_path)} ===")
    done = _generate_jobs(jobs, out_path, desc=f"augment:{tag}")
    total = len(load_existing_keys(out_path))
    print(f"{model}: +{done} this run, {total} total in {os.path.basename(out_path)}")
    return done


def _flow_specs(n: int, seed: int) -> list[SampleSpec]:
    """n distinct FLOW specs (problems/profiles), for FLOW-targeted augmentation."""
    return [s for s in build_specs(n, seed=seed) if s.cognitive_state == "FLOW"][:n]


def run_flow_augmentation(model: str, n_flow: int = 500, seed: int = 4242) -> int:
    """Part 5 — generate n_flow FLOW-targeted samples for `model`, HALF of them
    hard negatives (look like FLOW but are truly PLANNING or INSIGHT). All appended
    to flow_hard_negatives.jsonl. Resume-safe."""
    tag = _tag(model)
    check_model(model)
    pool = _flow_specs(n_flow, seed)
    half = n_flow // 2
    genuine = pool[:half]
    decoys = pool[half:n_flow]

    def genuine_job(sp):
        def run():
            rec = generate_one(sp, model)
            rec["flow_augment"] = True
            rec["flow_hard_negative"] = False
            return rec
        return run

    def decoy_job(sp, decoy):
        def run():
            rec = generate_one(sp, model,
                               prompt=build_hard_negative_prompt(sp, decoy),
                               labelled_state=decoy)
            rec["flow_augment"] = True
            rec["flow_hard_negative"] = True
            return rec
        return run

    jobs: list[tuple[str, Any]] = []
    for s in genuine:
        jobs.append((f"{s.key}|{tag}|flow", genuine_job(s)))
    for i, s in enumerate(decoys):
        decoy = FLOW_DECOY_STATES[i % len(FLOW_DECOY_STATES)]
        jobs.append((f"{s.key}|{tag}|hn-{decoy.lower()}", decoy_job(s, decoy)))

    print(f"\n=== FLOW augmentation with {model} "
          f"({len(genuine)} genuine FLOW + {len(decoys)} hard negatives) "
          f"-> {os.path.basename(FLOW_HARD_NEG_PATH)} ===")
    done = _generate_jobs(jobs, FLOW_HARD_NEG_PATH, desc=f"flow-aug:{tag}")
    print(f"{model}: +{done} FLOW-aug samples this run")
    return done


# ------------------------------------------------------------------
# Mixed-corpus merge, split, stats, few-shot extraction
# ------------------------------------------------------------------

def stratified_split_by_pair(
    records: list[dict[str, Any]], seed: int = 7
) -> tuple[list, list, list]:
    """80/10/10 split stratified by BOTH cognitive_state AND generator_model, so
    every (state, generator) cell appears in train/val/test when it has >= 3."""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (r.get("cognitive_state", ""), r.get("generator_model", DEFAULT_MODEL))
        buckets[key].append(r)

    train, val, test = [], [], []
    for _key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_mixed_corpus() -> dict[str, Any]:
    """Merge base + all augment files into dataset_mixed.jsonl, resplit stratified
    by state x generator, refresh stats with by_generator, and fold the FLOW
    augmentation into the training split. Returns a summary dict.

    dataset_mixed.jsonl = base(llama) + the three general augment files (Part 2).
    flow_hard_negatives.jsonl is kept separate (Part 5) and appended to the
    TRAINING split only, so mixed_test measures the held-out general distribution.
    """
    # Base corpus: backfill generator_model + metacognition fields, persist atomically.
    base = [ensure_meta_fields(r) for r in load_all(DATASET_PATH)]
    if base:
        _atomic_write_jsonl(DATASET_PATH, base)

    mixed = list(base)
    for model in AUGMENT_MODELS:
        recs = [ensure_meta_fields(r) for r in load_all(augment_path(model))]
        mixed.extend(recs)

    write_jsonl(MIXED_DATASET_PATH, mixed)

    train, val, test = stratified_split_by_pair(mixed)

    # Fold FLOW augmentation into TRAINING only (Part 5: "include for final training").
    flow_aug = [ensure_meta_fields(r) for r in load_all(FLOW_HARD_NEG_PATH)]
    if flow_aug:
        rng = random.Random(11)
        rng.shuffle(flow_aug)
        train = train + flow_aug
        random.Random(13).shuffle(train)

    write_jsonl(MIXED_TRAIN_PATH, train)
    write_jsonl(MIXED_VAL_PATH, val)
    write_jsonl(MIXED_TEST_PATH, test)

    by_generator = dict(sorted(
        Counter(r.get("generator_model", DEFAULT_MODEL) for r in mixed).items()))

    stats = compute_stats(mixed)
    stats["source"] = os.path.basename(MIXED_DATASET_PATH)
    stats["by_generator"] = by_generator
    stats["flow_augmentation"] = {
        "file": os.path.basename(FLOW_HARD_NEG_PATH),
        "total": len(flow_aug),
        "hard_negatives": sum(1 for r in flow_aug if r.get("flow_hard_negative")),
        "genuine_flow": sum(1 for r in flow_aug if not r.get("flow_hard_negative")),
        "folded_into": "mixed_train.jsonl",
    }
    stats["splits"] = {
        "mixed_train": len(train),
        "mixed_val": len(val),
        "mixed_test": len(test),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"\nMixed corpus built:")
    print(f"  dataset_mixed -> {MIXED_DATASET_PATH} ({len(mixed)})")
    print(f"  by_generator  : {by_generator}")
    print(f"  mixed_train   -> {MIXED_TRAIN_PATH} ({len(train)}  incl. {len(flow_aug)} FLOW-aug)")
    print(f"  mixed_val     -> {MIXED_VAL_PATH} ({len(val)})")
    print(f"  mixed_test    -> {MIXED_TEST_PATH} ({len(test)})")
    # Every generator must appear in each split (stratification guarantee).
    for name, split in (("train", train), ("val", val), ("test", test)):
        gens = dict(sorted(Counter(r.get("generator_model", DEFAULT_MODEL)
                                   for r in split).items()))
        print(f"    {name} by generator: {gens}")
    return {"total": len(mixed), "by_generator": by_generator,
            "train": len(train), "val": len(val), "test": len(test)}


def build_fewshot_examples(per_generator: int = 2) -> dict[str, Any]:
    """Part 3 — pull `per_generator` highest-confidence examples per (state,
    generator) from mixed_train.jsonl, giving a multi-generator few-shot set
    (e.g. 2/generator x 4 generators = 8 per state). Saved to
    data/eval/mixed_fewshot_examples.json for versioning/citation.

    NOTE: wiring these into the analyzer's LLM prompt would edit
    metacognition/analyzer.py, which is off-limits (concurrent process). The JSON
    is produced and ready to load; the wiring is intentionally deferred.
    """
    records = load_all(MIXED_TRAIN_PATH)
    if not records:
        raise FileNotFoundError(
            f"{MIXED_TRAIN_PATH} not found — run --merge first.")

    # group by (state, generator), sort by confidence_before desc (highest first).
    grouped: dict[tuple, list] = defaultdict(list)
    for r in records:
        st = r.get("cognitive_state", "")
        gen = r.get("generator_model", DEFAULT_MODEL)
        grouped[(st, gen)].append(r)

    examples: dict[str, list] = {s: [] for s in COGNITIVE_STATES}
    generators_seen: set[str] = set()
    for state in COGNITIVE_STATES:
        for gen in GENERATOR_TAGS:  # deterministic generator order
            items = grouped.get((state, gen), [])
            items.sort(key=lambda r: (r.get("confidence_before", 0),
                                      r.get("id", "")), reverse=True)
            for r in items[:per_generator]:
                generators_seen.add(gen)
                ta = r["think_aloud"]
                examples[state].append({
                    "generator_model": gen,
                    "confidence_before": r.get("confidence_before"),
                    "subject": r.get("problem", {}).get("subject"),
                    "think_aloud": (f"{ta.get('pre_attempt','')} "
                                    f"{ta.get('during_attempt','')} "
                                    f"{ta.get('post_attempt','')}").strip(),
                    "cognitive_state": state,
                    "source_id": r.get("id"),
                })

    payload = {
        "description": "Multi-generator few-shot examples for the LLM-as-classifier "
                       "(Part 3). Highest-confidence examples per (state, generator).",
        "per_generator": per_generator,
        "generators": sorted(generators_seen),
        "n_states": len(COGNITIVE_STATES),
        "counts_per_state": {s: len(examples[s]) for s in COGNITIVE_STATES},
        "wiring_note": ("Not yet wired into metacognition/analyzer.py (off-limits: "
                        "concurrent process). Load this JSON into the LLM few-shot "
                        "prompt to apply."),
        "examples": examples,
    }
    os.makedirs(os.path.dirname(MIXED_FEWSHOT_PATH), exist_ok=True)
    with open(MIXED_FEWSHOT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nFew-shot examples -> {MIXED_FEWSHOT_PATH}")
    print(f"  generators: {sorted(generators_seen)}")
    print(f"  per state : {payload['counts_per_state']}")
    return payload


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run_generation(
    specs: list[SampleSpec],
    model: str,
    *,
    existing_keys: set[str] | None = None,
    seed: int = 1234,
    samples_per_state: int = 0,
) -> int:
    """Generate all specs in parallel batches, saving incrementally.

    Returns the number of newly generated samples.
    """
    writer = IncrementalWriter(DATASET_PATH)
    generated = 0
    failed = 0
    # Cumulative set of everything on disk, so the checkpoint reflects the
    # true total (prior runs + this run), not just this run.
    completed_keys: set[str] = set(existing_keys or ())

    def checkpoint() -> None:
        save_checkpoint(
            completed_keys, model=model, seed=seed,
            samples_per_state=samples_per_state,
            generated_this_run=generated, failed_this_run=failed,
        )

    try:
        with tqdm(total=len(specs), desc="generating", unit="sample") as bar:
            for start in range(0, len(specs), BATCH_SIZE):
                batch = specs[start:start + BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(generate_one, s, model): s for s in batch}
                    for fut in as_completed(futures):
                        spec = futures[fut]
                        try:
                            record = fut.result()
                            writer.write(record)
                            generated += 1
                            completed_keys.add(spec.key)
                            if generated % CHECKPOINT_EVERY == 0:
                                checkpoint()
                        except Exception as e:  # noqa: BLE001
                            failed += 1
                            log_failure(spec, e)
                        finally:
                            bar.update(1)
                            bar.set_postfix(ok=generated, fail=failed)
    finally:
        writer.close()
        checkpoint()  # always leave an up-to-date checkpoint, even on crash

    if failed:
        print(f"\n{failed} generation(s) failed and were skipped "
              f"(logged to {FAILURE_LOG_PATH}).")
    return generated


def print_state_samples() -> None:
    """Print one sample per cognitive state for quality verification."""
    records = load_all(DATASET_PATH)
    by_state: dict[str, dict] = {}
    for r in records:
        st = r["cognitive_state"]
        if st not in by_state:
            by_state[st] = r

    print("\n" + "=" * 70)
    print("SAMPLE PER COGNITIVE STATE (quality check)")
    print("=" * 70)
    for state in COGNITIVE_STATES:
        r = by_state.get(state)
        print(f"\n### {state}")
        if not r:
            print("  (no sample generated for this state)")
            continue
        p = r["problem"]
        prof = r["student_profile"]
        ta = r["think_aloud"]
        print(f"  [{p['subject']} / {p['topic']} / {p['difficulty']}] "
              f"ADHD-{prof['adhd_type']}, conf={prof['subject_confidence']}")
        print(f"  problem: {p['text']}")
        print(f"  PRE   : {ta['pre_attempt']}")
        print(f"  DURING: {ta['during_attempt']}")
        print(f"  POST  : {ta['post_attempt']}")
        print(f"  correct={r['correct_answer']}  ground_truth={r['ground_truth']}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic student think-aloud dataset via Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--samples", type=int, default=50,
                   help="Samples to generate PER cognitive state (min 50 recommended).")
    p.add_argument("--resume", action="store_true",
                   help="Skip specs already present in dataset.jsonl and continue.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="Ollama model to use for generation.")
    p.add_argument("--seed", type=int, default=1234, help="Spec-generation seed.")
    p.add_argument("--backfill", action="store_true",
                   help="Add metacognition fields to the existing dataset and "
                        "rewrite stats + splits — no new generation.")
    # Multi-generator augmentation (Parts 1, 2, 3, 5).
    p.add_argument("--augment", type=str, metavar="MODEL",
                   help="Augment with one model (writes augment_<tag>.jsonl); "
                        "uses --samples per state.")
    p.add_argument("--augment-all", action="store_true",
                   help=f"Augment with every alternative generator "
                        f"({', '.join(AUGMENT_MODELS)}); uses --samples per state.")
    p.add_argument("--flow-augment", type=str, metavar="MODEL",
                   help="FLOW-targeted augmentation for one model (Part 5).")
    p.add_argument("--flow-augment-all", action="store_true",
                   help=f"FLOW-targeted augmentation for every generator "
                        f"({', '.join(FLOW_AUGMENT_MODELS)}).")
    p.add_argument("--flow-samples", type=int, default=500,
                   help="FLOW-targeted samples per generator (half hard negatives).")
    p.add_argument("--merge", action="store_true",
                   help="Merge base + augment files into dataset_mixed.jsonl and "
                        "resplit stratified by state x generator.")
    p.add_argument("--fewshot", action="store_true",
                   help="Build data/eval/mixed_fewshot_examples.json from mixed_train.")
    return p.parse_args(argv)


def check_model(model: str) -> None:
    try:
        available = {m.model for m in ollama.list().models}
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not query Ollama ({e}). "
              f"Ensure `ollama serve` is running.", file=sys.stderr)
        return
    # Ollama tags may or may not include ':latest'; match loosely.
    names = {n.split(":")[0] for n in available} | available
    if model not in available and model.split(":")[0] not in names:
        print(f"WARNING: model '{model}' not found in `ollama list`. "
              f"Available: {sorted(available)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)

    # Backfill-only: add the new metacognition fields to the existing corpus and
    # rewrite stats + splits, without any (slow) model generation.
    if args.backfill:
        print("Backfill mode: adding metacognition fields to the existing dataset.")
        finalize(args.model)
        print_state_samples()
        return 0

    # --- Multi-generator augmentation modes (no default generation) ---
    if args.augment or args.augment_all:
        models = AUGMENT_MODELS if args.augment_all else [args.augment]
        for m in models:
            run_augmentation(m, args.samples, seed=args.seed)
        return 0

    if args.flow_augment or args.flow_augment_all:
        models = FLOW_AUGMENT_MODELS if args.flow_augment_all else [args.flow_augment]
        for m in models:
            run_flow_augmentation(m, n_flow=args.flow_samples, seed=args.seed)
        return 0

    if args.merge:
        build_mixed_corpus()
        return 0

    if args.fewshot:
        build_fewshot_examples()
        return 0

    check_model(args.model)

    specs = build_specs(args.samples, seed=args.seed)

    # dataset.jsonl is authoritative (fsync'd per sample); the checkpoint is a
    # cross-check so resume is exact even if the two ever diverge.
    existing = load_existing_keys(DATASET_PATH) | load_checkpoint_keys()

    if args.resume:
        before = len(specs)
        specs = [s for s in specs if s.key not in existing]
        print(f"Resume: {len(existing)} already on disk, "
              f"{before - len(specs)} of the target already done, "
              f"{len(specs)} remaining to generate.")
    else:
        if os.path.exists(DATASET_PATH):
            print(f"Note: appending to existing {DATASET_PATH}. "
                  f"Use --resume to skip duplicates.")

    total_target = len(COGNITIVE_STATES) * args.samples
    print(f"Model: {args.model}")
    print(f"Target: {args.samples} samples/state x {len(COGNITIVE_STATES)} states "
          f"= {total_target} samples")
    print(f"To generate this run: {len(specs)}\n")

    if specs:
        generated = run_generation(
            specs, args.model,
            existing_keys=existing,
            seed=args.seed,
            samples_per_state=args.samples,
        )
        print(f"\nGenerated {generated} new samples this run.")
    else:
        print("Nothing to generate (all specs already present).")

    finalize(args.model)
    print_state_samples()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
