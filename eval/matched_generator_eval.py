#!/usr/bin/env python3.11
"""
Matched cross-generator robustness check for the offline seven-state classifier.

WHY
---
`problem_disjoint_eval.py` scores each generator on all of its utterances for the
10 held-out test problems. Those arms are equal in problems but not in support:
llama3.1:8b contributes 695 utterances, the other three about 340 each, and the
per-(problem, state) mix differs between them. A generator gap measured that way
is partly a gap in *what was sampled* rather than in what the classifier can do.

This script removes that confound. For every (held-out problem x cognitive state)
cell it takes k = the smallest count any generator has in that cell, then draws
exactly k utterances from each generator with a recorded seed. Every generator
then has an identical support vector: the same number of examples for the same
problem and the same label, cell by cell. Any remaining difference is
attributable to generator style alone.

The classifier is the same frozen artefact. Nothing is retuned; this module never
imports anything that could mutate it.

    python3.11 eval/matched_generator_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from metacognition.analyzer import (  # noqa: E402
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
)
from build_problem_disjoint_split import (  # noqa: E402
    DEFAULT_SEED,
    GENERATORS,
    build,
)
from problem_disjoint_eval import (  # noqa: E402
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    FROZEN_SOURCES,
    _accuracy,
    _sha256_file,
    classify_all,
    freeze_record,
    paired_difference,
    score,
)

MATCH_SEED = 20260826
LLAMA = "llama3.1:8b"


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------

def match_cells(test_rows: list[dict], seed: int = MATCH_SEED) -> tuple[list[dict], dict]:
    """Draw an equal, cell-matched sample from every generator.

    The matching unit is the (problem_id, cognitive_state) cell. Within a cell,
    k = min over generators of that generator's count, and exactly k rows are
    drawn from each. Cells where some generator has nothing would contribute
    k = 0 and drop out for everyone, which keeps the match exact rather than
    silently unbalancing it; the audit records how many cells that cost.

    Each cell/generator draw is seeded from (seed, problem_id, state, generator),
    so adding or removing a cell does not perturb the draws in any other cell.
    """
    cells: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for r in test_rows:
        cells[(r["problem_id"], r["cognitive_state"])][r["generator"]].append(r)

    matched: list[dict] = []
    dropped_cells: list[dict] = []
    audit: list[dict] = []

    for key in sorted(cells):
        problem_id, state = key
        available = {g: cells[key].get(g, []) for g in GENERATORS}
        k = min(len(available[g]) for g in GENERATORS)
        audit.append({
            "problem_id": problem_id,
            "cognitive_state": state,
            "available": {g: len(available[g]) for g in GENERATORS},
            "matched_k": k,
        })
        if k == 0:
            dropped_cells.append({
                "problem_id": problem_id, "cognitive_state": state,
                "available": {g: len(available[g]) for g in GENERATORS},
            })
            continue
        for g in GENERATORS:
            # Sort first so the pool order cannot depend on file iteration order.
            pool = sorted(available[g], key=lambda r: (r["source_file"], r["source_line"]))
            rng = random.Random(f"{seed}|{problem_id}|{state}|{g}")
            for r in rng.sample(pool, k):
                matched.append({**r, "matched_cell": f"{problem_id}|{state}"})

    return matched, {
        "matching_unit": "(problem_id, cognitive_state) cell",
        "rule": "k = min over the four generators of that cell's count; exactly k "
                "drawn from each generator",
        "seed": seed,
        "seed_derivation": "random.Random(f'{seed}|{problem_id}|{state}|{generator}') "
                           "per cell and generator, over a (source_file, source_line)-"
                           "sorted pool",
        "n_cells_total": len(cells),
        "n_cells_used": len(cells) - len(dropped_cells),
        "n_cells_dropped": len(dropped_cells),
        "dropped_cells": dropped_cells,
        "n_per_generator": len(matched) // len(GENERATORS) if matched else 0,
        "n_total": len(matched),
        "cell_audit": audit,
    }


# ------------------------------------------------------------------
# Artefacts
# ------------------------------------------------------------------

def write_manifest(matched: list[dict], match_info: dict, out_dir: str) -> list[str]:
    """Record exactly which examples were drawn.

    Identifiers, labels and source coordinates only — deliberately no utterance
    text, so the manifest stays a reproducibility record rather than a second
    copy of the generated corpus.
    """
    rows = sorted(matched, key=lambda r: (r["problem_id"], r["cognitive_state"],
                                          r["generator"], r["source_file"],
                                          r["source_line"]))
    csv_path = os.path.join(out_dir, "matched_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["matched_cell", "problem_id", "cognitive_state", "generator",
                    "record_id", "source_file", "source_line", "persona",
                    "subject", "topic", "difficulty"])
        for r in rows:
            w.writerow([r["matched_cell"], r["problem_id"], r["cognitive_state"],
                        r["generator"], r["record_id"], r["source_file"],
                        r["source_line"], r["persona"], r["subject"], r["topic"],
                        r["difficulty"]])

    json_path = os.path.join(out_dir, "matched_manifest.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "schema": "aria.matched_generator_manifest/1",
            "matching": {k: v for k, v in match_info.items() if k != "cell_audit"},
            "cell_audit": match_info["cell_audit"],
            "examples": [{
                "matched_cell": r["matched_cell"],
                "problem_id": r["problem_id"],
                "cognitive_state": r["cognitive_state"],
                "generator": r["generator"],
                "record_id": r["record_id"],
                "source_file": r["source_file"],
                "source_line": r["source_line"],
            } for r in rows],
        }, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return [json_path, csv_path]


def write_support_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator"] + list(COGNITIVE_STATES) + ["total"])
        for gen in GENERATORS:
            counts = results["support"][gen]
            w.writerow([gen] + [counts[s] for s in COGNITIVE_STATES]
                       + [sum(counts.values())])


def write_per_class_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "state", "precision", "recall", "f1", "support"])
        for gen in GENERATORS:
            per = results["conditions"][gen]["per_state"]
            for s in COGNITIVE_STATES:
                w.writerow([gen, s, per[s]["precision"], per[s]["recall"],
                            per[s]["f1"], per[s]["support"]])


def write_comparison_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "n_examples", "n_unique_problems", "accuracy",
                    "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1",
                    "macro_f1_ci95_low", "macro_f1_ci95_high",
                    "delta_accuracy_vs_llama", "delta_acc_ci95_low",
                    "delta_acc_ci95_high", "delta_acc_excludes_zero",
                    "delta_macro_f1_vs_llama", "delta_f1_ci95_low",
                    "delta_f1_ci95_high", "delta_f1_excludes_zero",
                    "unmatched_accuracy", "matched_minus_unmatched_accuracy"])
        for gen in GENERATORS:
            c = results["conditions"][gen]
            bs = c["bootstrap"]
            d = results["vs_llama"][gen]
            dac = d.get("delta_accuracy_ci95", [None, None])
            dfc = d.get("delta_macro_f1_ci95", [None, None])
            u = results["matched_vs_unmatched"][gen]
            w.writerow([gen, c["n_examples"], c["n_unique_problems"], c["accuracy"],
                        bs["accuracy_ci95"][0], bs["accuracy_ci95"][1], c["macro_f1"],
                        bs["macro_f1_ci95"][0], bs["macro_f1_ci95"][1],
                        d.get("delta_accuracy"), dac[0], dac[1],
                        d.get("delta_accuracy_excludes_zero"),
                        d.get("delta_macro_f1"), dfc[0], dfc[1],
                        d.get("delta_macro_f1_excludes_zero"),
                        u["unmatched_accuracy"], u["delta"]])


def write_confusion_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "true_state"] + [f"pred_{s}" for s in COGNITIVE_STATES])
        for gen in GENERATORS:
            cm = results["conditions"][gen]["confusion_matrix"]
            for i, s in enumerate(COGNITIVE_STATES):
                w.writerow([gen, s] + cm[i])


def write_figure(results: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        return None

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
    abbr = [s[:4].title() for s in COGNITIVE_STATES]
    for ax, gen in zip(axes, GENERATORS):
        c = results["conditions"][gen]
        cm = np.array(c["confusion_matrix"], dtype=float)
        rows = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(abbr)), abbr, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(abbr)), abbr, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        if gen == GENERATORS[0]:
            ax.set_ylabel("true state", fontsize=9)
        delta = results["vs_llama"][gen].get("delta_accuracy")
        subtitle = ("baseline" if gen == LLAMA else f"Δacc {delta:+.3f} vs llama")
        ax.set_title(f"{gen}\nacc={c['accuracy']:.3f}  macro-F1={c['macro_f1']:.3f}\n"
                     f"{subtitle}", fontsize=9.5)
        for i in range(len(abbr)):
            for j in range(len(abbr)):
                if cm[i, j] == 0:
                    continue
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.55 else "#22303c")
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.015)
    cbar.set_label("row-normalised rate (recall on the diagonal)", fontsize=8.5)
    n = results["matching"]["n_per_generator"]
    fig.suptitle("Matched cross-generator evaluation — identical support "
                 f"({n} examples per generator, matched by held-out problem x state)",
                 fontsize=13, y=1.02)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="problem-partition seed")
    ap.add_argument("--match-seed", type=int, default=MATCH_SEED)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--out-dir", default=os.path.join(
        EVAL_ROOT, "data", "eval", "problem_disjoint_20260826"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frozen = freeze_record()
    print(f"frozen at commit {frozen['git_commit'][:12]} "
          f"(analyzer.py sha256 "
          f"{frozen['source_sha256']['eval/metacognition/analyzer.py'][:12]})")

    records, manifest = build(args.seed)
    test_rows = [r for r in records if r["split"] == "test"]

    matched, match_info = match_cells(test_rows, args.match_seed)
    print(f"matched: {match_info['n_per_generator']} examples per generator "
          f"({match_info['n_total']} total) from "
          f"{match_info['n_cells_used']}/{match_info['n_cells_total']} cells; "
          f"{match_info['n_cells_dropped']} cell(s) dropped")

    analyzer = CognitiveStateAnalyzer(use_llm=False)
    scored = classify_all(matched, analyzer)
    unmatched_scored = classify_all(test_rows, analyzer)

    per_gen = {g: [r for r in scored if r["generator"] == g] for g in GENERATORS}
    unmatched_per_gen = {g: [r for r in unmatched_scored if r["generator"] == g]
                         for g in GENERATORS}

    # Equal support is the whole point — verify it rather than assume it.
    support = {g: {s: sum(1 for r in per_gen[g] if r["cognitive_state"] == s)
                   for s in COGNITIVE_STATES} for g in GENERATORS}
    reference = support[LLAMA]
    for g in GENERATORS:
        assert support[g] == reference, f"support mismatch for {g}: {support[g]}"
        assert len(per_gen[g]) == match_info["n_per_generator"]
    cells_by_gen = {g: Counter(r["matched_cell"] for r in per_gen[g]) for g in GENERATORS}
    for g in GENERATORS:
        assert cells_by_gen[g] == cells_by_gen[LLAMA], f"cell mismatch for {g}"
    print(f"support check: all four generators have identical per-state and "
          f"per-cell support ({sum(reference.values())} examples each)")

    conditions = {
        g: score(g, f"Matched cross-generator arm. {g} on the same "
                    f"{match_info['n_per_generator']} (problem x state) matched "
                    f"cells as every other generator.",
                 per_gen[g], reps=args.bootstrap_reps)
        for g in GENERATORS
    }

    vs_llama: dict[str, dict] = {}
    for g in GENERATORS:
        if g == LLAMA:
            vs_llama[g] = {"baseline": True, "delta_accuracy": 0.0,
                           "delta_macro_f1": 0.0,
                           "delta_accuracy_ci95": [0.0, 0.0],
                           "delta_macro_f1_ci95": [0.0, 0.0]}
            continue
        d = paired_difference(per_gen[LLAMA], per_gen[g], reps=args.bootstrap_reps)
        d["delta_macro_f1"] = round(conditions[g]["macro_f1"]
                                    - conditions[LLAMA]["macro_f1"], 4)
        lo, hi = d["delta_macro_f1_ci95"]
        d["delta_accuracy_excludes_zero"] = d.pop("excludes_zero_accuracy")
        d["delta_macro_f1_excludes_zero"] = not (lo <= 0.0 <= hi)
        vs_llama[g] = d

    matched_vs_unmatched = {
        g: {
            "matched_accuracy": conditions[g]["accuracy"],
            "unmatched_accuracy": round(_accuracy(unmatched_per_gen[g]), 4),
            "delta": round(conditions[g]["accuracy"]
                           - _accuracy(unmatched_per_gen[g]), 4),
            "matched_n": len(per_gen[g]),
            "unmatched_n": len(unmatched_per_gen[g]),
        } for g in GENERATORS
    }

    results = {
        "schema": "aria.matched_generator_eval/1",
        "purpose": "Robustness check: equalise support across generators so a "
                   "generator gap cannot be a sampling artefact.",
        "frozen": frozen,
        "split": {
            "seed": manifest["seed"],
            "procedure": manifest["partition_procedure"],
            "test_problem_ids": manifest["distributions"]["test"]["problem_ids"],
        },
        "matching": {k: v for k, v in match_info.items() if k != "cell_audit"},
        "support": support,
        "support_identical_across_generators": True,
        "conditions": conditions,
        "vs_llama": vs_llama,
        "matched_vs_unmatched": matched_vs_unmatched,
    }

    paths = list(write_manifest(matched, match_info, args.out_dir))
    p = os.path.join(args.out_dir, "matched_metrics.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    paths.append(p)

    for name, fn in (("matched_support.csv", write_support_csv),
                     ("matched_per_class_metrics.csv", write_per_class_csv),
                     ("matched_generator_comparison.csv", write_comparison_csv),
                     ("matched_confusion_matrices.csv", write_confusion_csv)):
        p = os.path.join(args.out_dir, name)
        fn(results, p)
        paths.append(p)

    fig = write_figure(results, os.path.join(args.out_dir, "matched_confusion_matrix.png"))
    if fig:
        paths.append(fig)

    # Regenerate the directory-wide checksum file so it covers these artefacts too.
    checks = {}
    for fname in sorted(os.listdir(args.out_dir)):
        if fname == "CHECKSUMS.sha256":
            continue
        checks[fname] = _sha256_file(os.path.join(args.out_dir, fname))
    for src in FROZEN_SOURCES + [
            "eval/build_problem_disjoint_split.py",
            "eval/problem_disjoint_eval.py",
            "eval/matched_generator_eval.py"]:
        checks[src] = _sha256_file(os.path.join(REPO_ROOT, src))
    p = os.path.join(args.out_dir, "CHECKSUMS.sha256")
    with open(p, "w", encoding="utf-8") as fh:
        for k in sorted(checks):
            fh.write(f"{checks[k]}  {k}\n")
    paths.append(p)

    print("\n" + "=" * 82)
    print(f"{'generator':16s}{'n':>6}{'acc':>9}{'acc CI':>20}"
          f"{'macroF1':>10}{'dAcc':>9}{'dAcc CI':>20}")
    for g in GENERATORS:
        c = conditions[g]
        d = vs_llama[g]
        print(f"{g:16s}{c['n_examples']:>6}{c['accuracy']:>9.4f}"
              f"{str(c['bootstrap']['accuracy_ci95']):>20}"
              f"{c['macro_f1']:>10.4f}{d['delta_accuracy']:>+9.4f}"
              f"{str(d['delta_accuracy_ci95']):>20}")
    print("=" * 82)
    for k in paths:
        print(f"  {os.path.relpath(k, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
