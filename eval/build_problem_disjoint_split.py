#!/usr/bin/env python3.11
"""
Build a problem-disjoint partition of the synthetic think-aloud corpus.

WHY THIS EXISTS
---------------
The corpus in data/synthetic_thinkaloud/ is 50 mathematics/science/verbal
problems, each rendered many times over: seven cognitive states, several ADHD
personas, several subject-confidence levels, and four generator models. The
shipped train/val/test split (train.jsonl / val.jsonl / test.jsonl) splits those
renderings at the *utterance* level, so all 50 problems appear in all three
files. Any evaluation on test.jsonl is therefore a seen-problem evaluation: the
classifier is scored on new utterances about problems whose wording, topic and
answer were all visible while the classifier was being built.

This script partitions at the *problem* level instead. A problem and every
utterance derived from it — every state, persona, confidence level, paraphrase
and generator — lands in exactly one split. That makes the test estimate a
statement about unseen problems.

WHAT IT DOES
------------
1. Collects every record in the evaluation pool (see POOL below) and derives a
   stable problem identity for each: the problem text is Unicode-normalized,
   lowercased and whitespace-collapsed, then SHA-256 hashed. Problems are
   numbered P01..P50 in ascending hash order, so the IDs do not depend on file
   order, on the shuffle seed, or on which files happen to be present.
2. Shuffles the 50 problem IDs with a recorded seed and cuts 30 / 10 / 10.
3. Checks that every cognitive state survives in validation and test, for every
   generator. If a plain shuffle drops a class, it falls back to a deterministic
   group-stratified assignment (see `_stratified_partition`) and records which
   procedure was used.
4. Asserts no problem ID and no normalized-problem hash appears in more than one
   split, and that no utterance text crosses splits.
5. Writes a machine-readable manifest (JSON + CSV) with, per problem: the ID, the
   normalized-problem hash, the assigned split, and the per-generator and
   per-cognitive-state example counts.

    python3.11 eval/build_problem_disjoint_split.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from metacognition.analyzer import COGNITIVE_STATES  # noqa: E402

DATA_DIR = os.path.join(EVAL_ROOT, "data", "synthetic_thinkaloud")

# The evaluation pool, keyed by generator. These seven files are mutually
# disjoint at the utterance level (verified in `load_pool`), and together they
# are every generated think-aloud in the corpus.
#
#   dataset.jsonl        = train.jsonl + val.jsonl + test.jsonl (llama3.1:8b)
#   holdout_<gen>.jsonl  = the cross-generator holdout, 50 per state
#   augment_<gen>.jsonl  = the larger non-llama augmentation pool
#
# Deliberately NOT in the pool:
#   dataset_mixed.jsonl, mixed_{train,val,test}.jsonl — resamples of the files
#     above, so including them would double-count utterances.
#   flow_hard_negatives.jsonl — 160 adversarially written FLOW look-alikes over
#     28 problems. It is a targeted stress set, not a sample from the corpus
#     distribution, so folding it into the headline pool would bias accuracy
#     downward by construction. It is scored separately and labelled as such.
POOL: dict[str, list[str]] = {
    "llama3.1:8b": ["dataset.jsonl"],
    "mistral:7b": ["holdout_mistral.jsonl", "augment_mistral.jsonl"],
    "gemma2:9b": ["holdout_gemma2.jsonl", "augment_gemma2.jsonl"],
    "phi3:medium": ["holdout_phi3.jsonl", "augment_phi3.jsonl"],
}

GENERATORS = list(POOL)

SPLIT_SIZES = {"train": 30, "val": 10, "test": 10}
DEFAULT_SEED = 20260826


# ------------------------------------------------------------------
# Problem identity
# ------------------------------------------------------------------

def normalize_problem(text: str) -> str:
    """Canonical form of a problem statement.

    NFKC folds the corpus's typographic variants (H₂ subscripts, curly quotes,
    non-breaking spaces) onto one code point each, so two renderings of the same
    problem cannot hash apart on punctuation alone.
    """
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.lower().split()).strip()


def problem_hash(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def join_think_aloud(ta) -> str:
    """Flatten a think-aloud record the same way the analyzer's loader does."""
    if isinstance(ta, dict):
        parts = [ta.get(k, "") for k in ("pre_attempt", "during_attempt", "post_attempt")]
        return " ".join(p for p in parts if p).strip()
    return str(ta or "").strip()


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_pool(data_dir: str = DATA_DIR) -> list[dict]:
    """Load every pooled record, tagged with generator, source file and problem ID.

    Raises if the pool files overlap at the utterance level, which would make
    every count downstream a double count.
    """
    records: list[dict] = []
    for generator, files in POOL.items():
        for fname in files:
            path = os.path.join(data_dir, fname)
            with open(path, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = join_think_aloud(rec.get("think_aloud"))
                    records.append({
                        "generator": generator,
                        "source_file": os.path.join("eval", "data",
                                                    "synthetic_thinkaloud", fname),
                        "source_line": line_no,
                        "record_id": rec.get("id", ""),
                        "problem_text": rec["problem"]["text"],
                        "problem_hash": problem_hash(rec["problem"]["text"]),
                        "subject": rec["problem"].get("subject", ""),
                        "topic": rec["problem"].get("topic", ""),
                        "difficulty": rec["problem"].get("difficulty", ""),
                        "persona": (rec.get("student_profile") or {}).get("adhd_type", ""),
                        "subject_confidence": (rec.get("student_profile") or {})
                                              .get("subject_confidence"),
                        "cognitive_state": rec["cognitive_state"],
                        "text": text,
                    })

    seen: dict[str, str] = {}
    for r in records:
        key = r["text"]
        if key in seen:
            raise AssertionError(
                f"Pool files overlap: utterance from {r['source_file']} also in {seen[key]}"
            )
        seen[key] = r["source_file"]

    # Assign P01..PNN by ascending normalized hash: independent of file order,
    # of the shuffle seed, and of which pool files are present.
    hashes = sorted({r["problem_hash"] for r in records})
    width = max(2, len(str(len(hashes))))
    hash_to_id = {h: f"P{i:0{width}d}" for i, h in enumerate(hashes, 1)}
    for r in records:
        r["problem_id"] = hash_to_id[r["problem_hash"]]
    return records


def problem_table(records: list[dict]) -> dict[str, dict]:
    """One row per problem: identity plus its per-generator / per-state counts."""
    table: dict[str, dict] = {}
    for r in records:
        row = table.setdefault(r["problem_id"], {
            "problem_id": r["problem_id"],
            "problem_hash": r["problem_hash"],
            "problem_text": r["problem_text"],
            "normalized_problem": normalize_problem(r["problem_text"]),
            "subject": r["subject"],
            "topic": r["topic"],
            "difficulty": r["difficulty"],
            "n_examples": 0,
            "by_generator": Counter(),
            "by_state": Counter(),
            "personas": set(),
        })
        row["n_examples"] += 1
        row["by_generator"][r["generator"]] += 1
        row["by_state"][r["cognitive_state"]] += 1
        row["personas"].add(r["persona"])
    return table


# ------------------------------------------------------------------
# Partitioning
# ------------------------------------------------------------------

def _shuffle_partition(problem_ids: list[str], seed: int) -> dict[str, str]:
    """Plain deterministic shuffle, then cut 30 / 10 / 10."""
    order = sorted(problem_ids)
    random.Random(seed).shuffle(order)
    assignment: dict[str, str] = {}
    cursor = 0
    for split, size in SPLIT_SIZES.items():
        for pid in order[cursor:cursor + size]:
            assignment[pid] = split
        cursor += size
    return assignment


def _stratified_partition(problem_ids: list[str], table: dict[str, dict],
                          seed: int) -> dict[str, str]:
    """Deterministic group-stratified fallback.

    Used only if the plain shuffle leaves a cognitive state with zero examples in
    validation or test. Procedure, in full:

      1. Order problems by subject, then by ascending problem ID, so the ordering
         is a function of the data alone.
      2. Shuffle within each subject with `seed`, so the choice inside a subject
         is random but reproducible.
      3. Deal the subjects' problems round-robin into train/val/test in the
         repeating pattern train,train,train,val,test — which yields exactly
         30/10/10 over 50 problems and spreads every subject across all three
         splits.

    Problems are still assigned whole: this changes *which* problems land where,
    never whether a problem's utterances are kept together.
    """
    rng = random.Random(seed)
    by_subject: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(problem_ids):
        by_subject[table[pid]["subject"]].append(pid)

    ordered: list[str] = []
    for subject in sorted(by_subject):
        group = sorted(by_subject[subject])
        rng.shuffle(group)
        ordered.extend(group)

    pattern = ["train", "train", "train", "val", "test"]
    assignment = {pid: pattern[i % len(pattern)] for i, pid in enumerate(ordered)}

    # The pattern only guarantees the 30/10/10 shape when the total is a multiple
    # of 5. Fail loudly rather than silently returning a lopsided split.
    counts = Counter(assignment.values())
    if any(counts[s] != n for s, n in SPLIT_SIZES.items()):
        raise AssertionError(f"stratified partition produced {dict(counts)}, "
                             f"expected {SPLIT_SIZES}")
    return assignment


def _missing_classes(records: list[dict], assignment: dict[str, str]) -> dict:
    """States absent from val/test, overall and per generator."""
    missing: dict[str, list[str]] = {}
    for split in ("val", "test"):
        rows = [r for r in records if assignment[r["problem_id"]] == split]
        present = {r["cognitive_state"] for r in rows}
        gone = [s for s in COGNITIVE_STATES if s not in present]
        if gone:
            missing[split] = gone
        for gen in GENERATORS:
            gp = {r["cognitive_state"] for r in rows if r["generator"] == gen}
            gone_g = [s for s in COGNITIVE_STATES if s not in gp]
            if gone_g:
                missing[f"{split}:{gen}"] = gone_g
    return missing


def build_partition(records: list[dict], table: dict[str, dict],
                    seed: int) -> tuple[dict[str, str], str, dict]:
    """Return (assignment, procedure_used, class_coverage_report)."""
    problem_ids = sorted(table)
    assignment = _shuffle_partition(problem_ids, seed)
    missing = _missing_classes(records, assignment)
    if not missing:
        return assignment, "seeded_shuffle_30_10_10", {"seeded_shuffle_missing": {}}

    stratified = _stratified_partition(problem_ids, table, seed)
    missing_after = _missing_classes(records, stratified)
    return stratified, "deterministic_group_stratified_by_subject", {
        "seeded_shuffle_missing": missing,
        "stratified_missing": missing_after,
    }


# ------------------------------------------------------------------
# Leakage assertions
# ------------------------------------------------------------------

def assert_no_leakage(records: list[dict], assignment: dict[str, str]) -> dict:
    """Fail if any problem identity or utterance crosses a split boundary.

    Returns the pairwise overlap report on success (all zeros by construction of
    the checks, but the report is written to the manifest so a reader does not
    have to take the assertion's word for it).
    """
    ids: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    normalized: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = defaultdict(set)

    for r in records:
        split = assignment[r["problem_id"]]
        ids[split].add(r["problem_id"])
        hashes[split].add(r["problem_hash"])
        normalized[split].add(normalize_problem(r["problem_text"]))
        texts[split].add(r["text"])

    report: dict[str, int] = {}
    splits = ["train", "val", "test"]
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            for name, d in (("problem_id", ids), ("problem_hash", hashes),
                            ("normalized_problem", normalized), ("utterance", texts)):
                overlap = d[a] & d[b]
                report[f"{name}:{a}|{b}"] = len(overlap)
                if overlap:
                    raise AssertionError(
                        f"LEAKAGE — {len(overlap)} {name} value(s) in both "
                        f"{a} and {b}; first: {sorted(overlap)[:3]}"
                    )

    # A problem may not be assigned two ways, and every problem must be assigned.
    per_problem = defaultdict(set)
    for r in records:
        per_problem[r["problem_id"]].add(assignment[r["problem_id"]])
    multi = {p: s for p, s in per_problem.items() if len(s) != 1}
    if multi:
        raise AssertionError(f"LEAKAGE — problems with multiple splits: {multi}")

    unassigned = sorted({r["problem_id"] for r in records} - set(assignment))
    if unassigned:
        raise AssertionError(f"unassigned problems: {unassigned}")

    report["n_problems"] = len(per_problem)
    report["n_utterances"] = len(records)
    return report


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

def build_manifest(records: list[dict], table: dict[str, dict],
                   assignment: dict[str, str], seed: int, procedure: str,
                   coverage: dict, leakage: dict) -> dict:
    problems = []
    for pid in sorted(table):
        row = table[pid]
        problems.append({
            "problem_id": pid,
            "normalized_problem_sha256": row["problem_hash"],
            "split": assignment[pid],
            "subject": row["subject"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
            "problem_text": row["problem_text"],
            "n_examples": row["n_examples"],
            "n_personas": len(row["personas"]),
            "examples_by_generator": {g: row["by_generator"].get(g, 0) for g in GENERATORS},
            "examples_by_cognitive_state": {s: row["by_state"].get(s, 0)
                                            for s in COGNITIVE_STATES},
        })

    distributions = {}
    for split in ("train", "val", "test"):
        rows = [r for r in records if assignment[r["problem_id"]] == split]
        by_gen_state = {
            g: {s: sum(1 for r in rows if r["generator"] == g and r["cognitive_state"] == s)
                for s in COGNITIVE_STATES}
            for g in GENERATORS
        }
        distributions[split] = {
            "n_problems": sum(1 for p in assignment.values() if p == split),
            "n_examples": len(rows),
            "by_cognitive_state": {s: sum(1 for r in rows if r["cognitive_state"] == s)
                                   for s in COGNITIVE_STATES},
            "by_generator": {g: sum(1 for r in rows if r["generator"] == g)
                             for g in GENERATORS},
            "by_generator_and_state": by_gen_state,
            "by_subject": dict(sorted(Counter(r["subject"] for r in rows).items())),
            "problem_ids": sorted(p for p, s in assignment.items() if s == split),
        }

    return {
        "schema": "aria.problem_disjoint_split/1",
        "seed": seed,
        "partition_procedure": procedure,
        "split_sizes": SPLIT_SIZES,
        "problem_id_scheme": "P## in ascending SHA-256 order of the normalized problem text",
        "normalization": "unicodedata.NFKC -> lowercase -> collapse whitespace -> strip",
        "pool": {g: [os.path.join("eval", "data", "synthetic_thinkaloud", f) for f in files]
                 for g, files in POOL.items()},
        "excluded_from_pool": {
            "eval/data/synthetic_thinkaloud/dataset_mixed.jsonl":
                "resample of dataset.jsonl + augment_*.jsonl; would double-count",
            "eval/data/synthetic_thinkaloud/mixed_train.jsonl":
                "resample; would double-count",
            "eval/data/synthetic_thinkaloud/mixed_val.jsonl":
                "resample; would double-count",
            "eval/data/synthetic_thinkaloud/mixed_test.jsonl":
                "resample; would double-count (all 413 rows are in the pool already)",
            "eval/data/synthetic_thinkaloud/train.jsonl":
                "subset of dataset.jsonl; would double-count",
            "eval/data/synthetic_thinkaloud/val.jsonl":
                "subset of dataset.jsonl; would double-count",
            "eval/data/synthetic_thinkaloud/test.jsonl":
                "subset of dataset.jsonl; would double-count. Scored separately "
                "as the seen-problem/utterance-disjoint reference condition.",
            "eval/data/synthetic_thinkaloud/flow_hard_negatives.jsonl":
                "160 adversarial FLOW look-alikes over 28 problems; a targeted "
                "stress set rather than a sample of the corpus distribution. "
                "Scored separately and labelled as supplementary.",
        },
        "totals": {
            "n_problems": len(table),
            "n_examples": len(records),
            "by_generator": dict(Counter(r["generator"] for r in records)),
            "by_cognitive_state": dict(Counter(r["cognitive_state"] for r in records)),
        },
        "class_coverage": coverage,
        "leakage_check": leakage,
        "distributions": distributions,
        "problems": problems,
    }


def write_manifest(manifest: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "split_manifest.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    csv_path = os.path.join(out_dir, "split_manifest.csv")
    gens = GENERATORS
    states = COGNITIVE_STATES
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem_id", "normalized_problem_sha256", "split", "subject",
                    "topic", "difficulty", "n_examples", "n_personas"]
                   + [f"n_{g}" for g in gens] + [f"n_{s}" for s in states]
                   + ["problem_text"])
        for p in manifest["problems"]:
            w.writerow([p["problem_id"], p["normalized_problem_sha256"], p["split"],
                        p["subject"], p["topic"], p["difficulty"], p["n_examples"],
                        p["n_personas"]]
                       + [p["examples_by_generator"][g] for g in gens]
                       + [p["examples_by_cognitive_state"][s] for s in states]
                       + [p["problem_text"]])
    return json_path, csv_path


def build(seed: int = DEFAULT_SEED, data_dir: str = DATA_DIR) -> tuple[list[dict], dict]:
    """Load the pool, partition it, assert no leakage, return (records, manifest)."""
    records = load_pool(data_dir)
    table = problem_table(records)
    assignment, procedure, coverage = build_partition(records, table, seed)
    leakage = assert_no_leakage(records, assignment)
    manifest = build_manifest(records, table, assignment, seed, procedure,
                              coverage, leakage)
    for r in records:
        r["split"] = assignment[r["problem_id"]]
    return records, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=os.path.join(EVAL_ROOT, "data", "eval",
                                                      "problem_disjoint_20260826"))
    args = ap.parse_args()

    records, manifest = build(args.seed)
    json_path, csv_path = write_manifest(manifest, args.out_dir)

    print(f"seed                 {manifest['seed']}")
    print(f"partition procedure  {manifest['partition_procedure']}")
    print(f"problems / examples  {manifest['totals']['n_problems']} / "
          f"{manifest['totals']['n_examples']}")
    print("\nleakage check (all pairwise overlaps must be 0):")
    for k, v in manifest["leakage_check"].items():
        if "|" in k:
            print(f"  {k:44s} {v}")
    print("\nper-split distributions:")
    for split, d in manifest["distributions"].items():
        print(f"  {split:6s} problems={d['n_problems']:3d} examples={d['n_examples']:5d}")
        print(f"         states={d['by_cognitive_state']}")
        print(f"         generators={d['by_generator']}")
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
