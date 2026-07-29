"""
LoRA training data formatter for ARIA.

Produces two datasets:

Dataset A — ARIA personalization (per-profile):
  Source: data/synthetic_history.jsonl + live generation to reach 100 sessions/profile
  Format: ChatML <|system|>...<|user|>...<|assistant|>
  Filter: rubric score > 1.5 OR no scores present (include all synthetic turns)
  Output: lora/data/{profile_id}/train.jsonl + valid.jsonl

Dataset B — Failure-aware distillation (shared):
  Source: data/reasoning_traces/*.jsonl
  Positive: score > 1.5 → keep response as-is
  Negative mining: score < 0.8 → rewrite with llama3.1:8b and use fixed response
  Output: lora/data/distilled/train.jsonl + valid.jsonl
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
LORA_DIR    = ROOT / "lora"
LORA_DATA   = LORA_DIR / "data"

SYNTHETIC_HISTORY = DATA_DIR / "synthetic_history.jsonl"
TRACES_DIR        = DATA_DIR / "reasoning_traces"

# Target sessions per profile before training
TARGET_SESSIONS_PER_PROFILE = 100

CHATML_TEMPLATE = "{system}<|user|>\n{user}\n<|assistant|>\n{assistant}"

PROFILE_IDS = [
    "alex_chen", "jordan_rivera", "sam_okonkwo", "maya_patel", "eli_washington"
]


# ------------------------------------------------------------------
# ChatML formatting
# ------------------------------------------------------------------

def _format_chatml(system: str, user: str, assistant: str) -> str:
    return f"<|system|>\n{system.strip()}\n<|user|>\n{user.strip()}\n<|assistant|>\n{assistant.strip()}"


def _build_system_prompt(
    name: str,
    diagnosis: str,
    learning_style: str,
    topic: str,
    confidence: float,
    struggles: List[str],
    best_style: str,
) -> str:
    struggle_str = ", ".join(struggles) if struggles else "unknown areas"
    return (
        f"You are ARIA tutoring {name}, a {diagnosis} student who learns best through "
        f"{learning_style}. Their confidence in {topic} is {confidence:.0%}. "
        f"They struggle with {struggle_str}. "
        f"Their best explanation style for this topic is {best_style}. "
        f"Keep responses to 3 sentences max per block. Always use numbered micro-steps."
    )


def _distilled_system_prompt(success_snippets: str, failure_snippets: str) -> str:
    return (
        "You are a neurodivergent-aware AI tutor. "
        f"Use these proven approaches: {success_snippets}. "
        f"Never do: {failure_snippets}. "
        "Max 3 sentences per paragraph. Always number steps."
    )


# ------------------------------------------------------------------
# Load and group synthetic history
# ------------------------------------------------------------------

def _load_history_by_profile() -> Dict[str, List[dict]]:
    by_profile: Dict[str, List[dict]] = defaultdict(list)
    if not SYNTHETIC_HISTORY.exists():
        return by_profile
    with open(SYNTHETIC_HISTORY) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                session = json.loads(line)
                pid = session.get("subject_key", session.get("subject", "unknown")).replace(" ", "_")
                # Map subject to profile
                subject = session.get("subject", "")
                for pid_candidate in PROFILE_IDS:
                    # rough match: alex=algebra/biology/python, jordan=geometry/chemistry/js, etc.
                    by_profile[pid_candidate].append(session)
                    break  # assign to first — will be refined below
            except json.JSONDecodeError:
                continue

    # Actually re-read with profile-aware logic if profile_id stored
    by_profile_refined: Dict[str, List[dict]] = defaultdict(list)
    if SYNTHETIC_HISTORY.exists():
        with open(SYNTHETIC_HISTORY) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    session = json.loads(line)
                    # If profile was embedded, use it; otherwise distribute round-robin
                    pid = session.get("profile_id", None)
                    if pid and pid in PROFILE_IDS:
                        by_profile_refined[pid].append(session)
                    else:
                        # Distribute based on subject
                        subject = session.get("subject", "")
                        pid = _subject_to_profile(subject)
                        by_profile_refined[pid].append(session)
                except json.JSONDecodeError:
                    continue
    return by_profile_refined


def _subject_to_profile(subject: str) -> str:
    subject_lower = subject.lower()
    if any(s in subject_lower for s in ["algebra", "biology", "python"]):
        return "alex_chen"
    if any(s in subject_lower for s in ["geometry", "chemistry", "javascript", "js"]):
        return "jordan_rivera"
    if any(s in subject_lower for s in ["statistics", "physics", "java"]):
        return "sam_okonkwo"
    if any(s in subject_lower for s in ["earth", "scratch", "basics"]):
        return "maya_patel"
    if any(s in subject_lower for s in ["calculus", "cs theory", "theory"]):
        return "eli_washington"
    return "alex_chen"


# ------------------------------------------------------------------
# Generate more sessions for a profile if needed
# ------------------------------------------------------------------

def _generate_more_sessions(profile_id: str, need: int, verbose: bool = True) -> List[dict]:
    """Generate `need` more synthetic sessions for a profile."""
    from eval.profiles import PROFILE_MAP, ProfileVectorStore, ProfileLearningGraph
    from eval.synthetic_history import seed_history
    from eval.profiles import _make_vs_adaptor, _make_lg_adaptor

    profile = PROFILE_MAP.get(profile_id)
    if profile is None:
        return []

    if verbose:
        print(f"  [data_formatter] Generating {need} additional sessions for {profile_id}...")

    pvs = ProfileVectorStore(profile_id)
    plg = ProfileLearningGraph(profile_id)

    profile_dict = {
        "name": profile.name,
        "age": profile.age,
        "diagnosis": profile.diagnosis,
        "subjects": profile.subjects,
        "preferred_styles": profile.preferred_styles,
        "frustration_triggers": getattr(profile, "frustration_triggers", {}),
    }

    sessions = seed_history(
        vector_store=_make_vs_adaptor(pvs),
        learning_graph=_make_lg_adaptor(plg),
        profile=profile_dict,
        n_sessions=need,
        verbose=False,
    )
    # Tag with profile_id and append to file
    with open(SYNTHETIC_HISTORY, "a") as f:
        for s in sessions:
            s["profile_id"] = profile_id
            f.write(json.dumps(s) + "\n")
    return sessions


# ------------------------------------------------------------------
# Dataset A: profile-specific
# ------------------------------------------------------------------

def _session_to_examples(
    session: dict,
    profile_id: str,
    rubric_threshold: float,
) -> List[str]:
    """Convert one session dict to a list of ChatML-formatted training examples."""
    from eval.profiles import PROFILE_MAP
    profile = PROFILE_MAP.get(profile_id)
    if not profile:
        return []

    turns = session.get("turns", [])
    subject = session.get("subject", profile.subjects[0] if profile.subjects else "general")
    confidence = profile.baseline_confidence.get(subject, 0.5)
    struggles = list(profile.misconceptions.get(subject, ["unknown"]))[:2]
    best_style = profile.preferred_styles.get(subject, profile.learning_style)

    system = _build_system_prompt(
        name=profile.name,
        diagnosis=profile.diagnosis,
        learning_style=profile.learning_style,
        topic=subject,
        confidence=confidence,
        struggles=struggles,
        best_style=best_style,
    )

    examples = []
    for i, turn in enumerate(turns):
        if turn.get("role") == "user" and i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
            user_msg = turn["content"].strip()
            asst_msg = turns[i + 1]["content"].strip()
            if not user_msg or not asst_msg:
                continue
            # Check rubric score if present
            score = turn.get("weighted_score") or turn.get("score")
            if score is not None and float(score) < rubric_threshold:
                continue
            text = _format_chatml(system, user_msg, asst_msg)
            examples.append(text)
    return examples


def build_dataset_a(
    rubric_threshold: float = 1.5,
    target_per_profile: int = TARGET_SESSIONS_PER_PROFILE,
    val_fraction: float = 0.2,
    verbose: bool = True,
    reuse_existing: bool = True,
) -> dict:
    """Build per-profile LoRA training datasets.

    If reuse_existing is True (default) and a profile already has a non-trivial
    train.jsonl on disk, that profile's data is reused instead of regenerating
    ~100 LLM sessions. This makes the function idempotent so that a second
    caller (e.g. `main.py --train-lora`, which calls build_dataset_a before
    training) does not repeat the expensive generation performed by an earlier
    `data_formatter.py --dataset A/both` run.
    """
    (LORA_DATA).mkdir(parents=True, exist_ok=True)

    by_profile = _load_history_by_profile()
    stats: dict = {}

    for pid in PROFILE_IDS:
        profile_dir = LORA_DATA / pid
        existing_train = profile_dir / "train.jsonl"
        if reuse_existing and existing_train.exists():
            with open(existing_train) as _f:
                n_tr = sum(1 for _ in _f if _.strip())
            existing_val = profile_dir / "valid.jsonl"
            n_va = 0
            if existing_val.exists():
                with open(existing_val) as _f:
                    n_va = sum(1 for _ in _f if _.strip())
            if n_tr >= 5:
                stats[pid] = {
                    "sessions": 0,          # not regenerated this run
                    "total_examples": n_tr + n_va,
                    "train": n_tr,
                    "val": n_va,
                    "mean_length": 0,
                    "reused": True,
                }
                if verbose:
                    print(f"  {pid}: reusing existing dataset "
                          f"({n_tr} train / {n_va} val) — skipping regeneration")
                continue

        sessions = by_profile.get(pid, [])
        have = len(sessions)

        if have < target_per_profile:
            extra = _generate_more_sessions(pid, target_per_profile - have, verbose=verbose)
            sessions = sessions + extra

        all_examples: List[str] = []
        for session in sessions:
            examples = _session_to_examples(session, pid, rubric_threshold)
            all_examples.extend(examples)

        if not all_examples:
            if verbose:
                print(f"  [data_formatter] WARNING: No examples for {pid}")
            continue

        random.shuffle(all_examples)
        n_val = max(1, int(len(all_examples) * val_fraction))
        train_ex = all_examples[n_val:]
        val_ex   = all_examples[:n_val]

        profile_dir = LORA_DATA / pid
        profile_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(profile_dir / "train.jsonl", [{"text": t} for t in train_ex])
        _write_jsonl(profile_dir / "valid.jsonl", [{"text": t} for t in val_ex])

        stats[pid] = {
            "sessions": len(sessions),
            "total_examples": len(all_examples),
            "train": len(train_ex),
            "val": len(val_ex),
            "mean_length": int(sum(len(e) for e in all_examples) / max(len(all_examples), 1)),
        }

        if verbose:
            print(f"  {pid}: {len(sessions)} sessions → {len(train_ex)} train / {len(val_ex)} val examples")

    return stats


# ------------------------------------------------------------------
# Dataset B: failure-aware distillation
# ------------------------------------------------------------------

def _rewrite_failed_response(
    student_msg: str,
    bad_response: str,
    subject: str,
    failure_dim: str,
) -> str:
    """Use llama3.1:8b to fix a bad tutor response on a specific dimension."""
    import ollama
    prompt = (
        f"A student studying {subject} said: '{student_msg}'\n\n"
        f"An AI tutor gave this poor response (failed on {failure_dim}):\n'{bad_response}'\n\n"
        f"Rewrite the tutor response to score 2/2 on {failure_dim}:\n"
        f"- TC (Task Chunking): break into numbered micro-steps, max 3 sentences each\n"
        f"- SC (Session Consistency): reference past context and build on it explicitly\n"
        f"- FA (Frustration Adaptation): validate frustration, then gently redirect\n\n"
        f"Write ONLY the improved tutor response (no preamble):"
    )
    try:
        result = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6, "num_predict": 350},
        )
        return result.message.content.strip()
    except Exception:
        return bad_response  # fallback: keep original if rewrite fails


def build_dataset_b(
    positive_threshold: float = 1.5,
    negative_threshold: float = 0.8,
    val_fraction: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Build distillation dataset from reasoning traces."""
    (LORA_DATA / "distilled").mkdir(parents=True, exist_ok=True)

    # Load failure patterns for system prompt context
    fp_path = DATA_DIR / "failure_patterns.json"
    failure_patterns = {}
    if fp_path.exists():
        with open(fp_path) as f:
            failure_patterns = json.load(f)

    success_snippets = "; ".join(
        (fp.get("success_patterns", [{}])[0].get("pattern", "")[:80]
         if fp.get("success_patterns") else "")
        for dim, fp in list(failure_patterns.items())[:2]
    ) or "Break tasks into numbered steps. Reference prior sessions. Validate frustration first."

    failure_snippets = "; ".join(
        (fp.get("failure_patterns", [{}])[0].get("pattern", "")[:80]
         if fp.get("failure_patterns") else "")
        for dim, fp in list(failure_patterns.items())[:2]
    ) or "Do not give walls of text. Do not ignore student confusion. Do not skip micro-steps."

    system = _distilled_system_prompt(success_snippets, failure_snippets)

    positives: List[str] = []
    negatives_fixed: List[str] = []
    n_fixed = 0

    if not TRACES_DIR.exists():
        if verbose:
            print("  [data_formatter] No reasoning traces found — Dataset B will be empty.")
        _write_jsonl(LORA_DATA / "distilled" / "train.jsonl", [])
        _write_jsonl(LORA_DATA / "distilled" / "valid.jsonl", [])
        return {"positives": 0, "negatives_fixed": 0, "train": 0, "val": 0}

    failed_dim_to_records: Dict[str, List[dict]] = {"TC": [], "SC": [], "FA": []}

    for trace_file in TRACES_DIR.glob("*.jsonl"):
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                student_msg   = record.get("student_message", record.get("student_msg", ""))
                tutor_response = record.get("tutor_response", record.get("tutor_msg", ""))
                scores = record.get("rubric_scores", record.get("scores", {}))
                ws = float(record.get("weighted_score", 1.0))
                subject = record.get("subject", "general")

                if not student_msg or not tutor_response:
                    continue

                if ws >= positive_threshold:
                    text = _format_chatml(system, student_msg, tutor_response)
                    positives.append(text)

                elif ws < negative_threshold:
                    # Find which ND dimension failed worst
                    worst_dim = min(
                        ["TC", "SC", "FA"],
                        key=lambda d: float(scores.get(d, 1)),
                    )
                    failed_dim_to_records[worst_dim].append({
                        "student_msg": student_msg,
                        "tutor_response": tutor_response,
                        "subject": subject,
                        "dim": worst_dim,
                    })

    # Hard negative mining: rewrite up to 50 failed responses per dimension
    for dim, records in failed_dim_to_records.items():
        sample = random.sample(records, min(50, len(records)))
        for rec in sample:
            if verbose and n_fixed % 10 == 0:
                print(f"  [data_formatter] Rewriting {n_fixed+1} negative example ({dim})...", end="\r")
            fixed = _rewrite_failed_response(
                rec["student_msg"], rec["tutor_response"], rec["subject"], dim
            )
            text = _format_chatml(system, rec["student_msg"], fixed)
            negatives_fixed.append(text)
            n_fixed += 1

    if verbose:
        print()

    all_examples = positives + negatives_fixed
    random.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * val_fraction))
    train_ex = all_examples[n_val:]
    val_ex   = all_examples[:n_val]

    _write_jsonl(LORA_DATA / "distilled" / "train.jsonl", [{"text": t} for t in train_ex])
    _write_jsonl(LORA_DATA / "distilled" / "valid.jsonl", [{"text": t} for t in val_ex])

    stats = {
        "positives": len(positives),
        "negatives_fixed": n_fixed,
        "total": len(all_examples),
        "train": len(train_ex),
        "val": len(val_ex),
        "positive_fraction": round(len(positives) / max(len(all_examples), 1), 3),
    }
    if verbose:
        print(f"  Dataset B: {len(positives)} positives + {n_fixed} fixed-negatives")
        print(f"  → {len(train_ex)} train / {len(val_ex)} val")
    return stats


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------

def _write_jsonl(path: Path, records: List[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(stats_a: dict, stats_b: dict) -> None:
    print("\n" + "=" * 60)
    print("  DATASET A (profile-specific personalization)")
    print("=" * 60)
    total_a = 0
    for pid, s in stats_a.items():
        print(f"  {pid:20s}  sessions={s['sessions']:3d}  train={s['train']:4d}  val={s['val']:3d}")
        total_a += s.get("train", 0) + s.get("val", 0)
    print(f"  Total Dataset A examples: {total_a}")
    print()
    print("  DATASET B (failure-aware distillation)")
    print("=" * 60)
    print(f"  Positives (score>1.5):      {stats_b.get('positives', 0)}")
    print(f"  Fixed negatives (score<0.8): {stats_b.get('negatives_fixed', 0)}")
    print(f"  Train: {stats_b.get('train', 0)}   Val: {stats_b.get('val', 0)}")
    print("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build LoRA training datasets")
    parser.add_argument("--dataset", choices=["A", "B", "both"], default="both")
    parser.add_argument("--target-sessions", type=int, default=TARGET_SESSIONS_PER_PROFILE)
    parser.add_argument("--rubric-threshold", type=float, default=1.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stats_a, stats_b = {}, {}
    if args.dataset in ("A", "both"):
        print("Building Dataset A (profile personalization)...")
        stats_a = build_dataset_a(
            rubric_threshold=args.rubric_threshold,
            target_per_profile=args.target_sessions,
            verbose=not args.quiet,
        )
    if args.dataset in ("B", "both"):
        print("Building Dataset B (failure-aware distillation)...")
        stats_b = build_dataset_b(verbose=not args.quiet)

    report(stats_a, stats_b)
