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
BATCH_SIZE = 10
MAX_RETRIES = 3
CHECKPOINT_EVERY = 50

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


def generate_one(spec: SampleSpec, model: str) -> dict[str, Any]:
    """Generate a single sample dict. Raises on unrecoverable failure."""
    prompt = build_prompt(spec)
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            result = ollama.chat(
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

            return _assemble(spec, pre, during, post, correct)
        except Exception as e:  # noqa: BLE001 — retry any generation error
            last_err = e
            continue

    raise RuntimeError(f"generation failed after {MAX_RETRIES} attempts: {last_err}")


def _assemble(spec: SampleSpec, pre: str, during: str, post: str, correct: bool) -> dict[str, Any]:
    return {
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
        "cognitive_state": spec.cognitive_state,
        "ground_truth": dict(STATE_GROUND_TRUTH[spec.cognitive_state]),
        "correct_answer": bool(correct),
    }


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
    """Recompute stats + splits from the full dataset on disk."""
    records = load_all(DATASET_PATH)
    if not records:
        print("No records to finalize.")
        return

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
