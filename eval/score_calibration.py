#!/usr/bin/env python3.11
"""Score the calibration round — the pre-check run before the main 150.

    python3.11 eval/score_calibration.py \\
        --rater1 data/annotation/rater1_calibration.csv \\
        --rater2 data/annotation/rater2_calibration.csv

UNTESTED AGAINST RETURNED SHEETS. No rater has coded anything yet, so this has
only run against synthetic fills (see tests/test_calibration_scoring.py). The
logic is complete; what has never been seen is a sheet a human filled in.

WHAT THIS IS

Both raters code the 14 passages the rubric itself quotes as canonical examples,
before touching the main sheet. The question is whether the rubric and the
corpus labels are coherent enough that a careful reader lands on the intended
category. It is a floor check on the instrument.

WHAT THIS IS NOT

It is not rater accuracy, and it is not kappa. Raters read these same passages
minutes earlier under a labelled heading, so recognition inflates the score.
That asymmetry is the whole point of the design:

    a HIGH score is uninformative — it may be memory
    a LOW score is the signal — it means the definition, the example, or the
      original corpus label is wrong, and it says which category

Nothing here is folded into the main analysis. The main kappa is computed over
the 150-item sheet alone, and this script refuses to emit a kappa so the two
numbers cannot be confused for one another in a write-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

ANNOTATION_DIR = os.path.join(HERE, "data", "annotation")
DEFAULT_KEY = os.path.join(ANNOTATION_DIR, "annotation_private_key.json")
DEFAULT_RATER1 = os.path.join(ANNOTATION_DIR, "rater1_calibration.csv")
DEFAULT_RATER2 = os.path.join(ANNOTATION_DIR, "rater2_calibration.csv")
DEFAULT_OUT = os.path.join(HERE, "data", "eval", "calibration_precheck.json")

#: Below this, re-read the rubric before starting the main sheet. Not a
#: statistical threshold — a working rule for an instrument check on 14 items,
#: where missing more than about a fifth means something is unclear.
REVIEW_THRESHOLD = 0.80


def load_key(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run eval/build_annotation_study.py.")
    with open(path, encoding="utf-8") as fh:
        key = json.load(fh)
    calibration = key.get("calibration_round")
    if not calibration:
        raise SystemExit(f"{path} has no calibration_round block.")
    return calibration


def read_sheet(path: str, label: str) -> dict[str, dict]:
    if not os.path.exists(path):
        raise SystemExit(f"{label}: {path} not found.")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{label}: {path} has no rows.")
    coded = {}
    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            continue
        coded[item_id] = {
            "category": (row.get("predicted_category") or "").strip().upper(),
            "confidence": (row.get("confidence") or "").strip(),
        }
    return coded


def score(key: dict, sheets: dict[str, dict[str, dict]]) -> dict:
    items = key["items"]
    per_rater: dict[str, dict] = {}
    per_item: list[dict] = []

    for label, coded in sheets.items():
        correct = blank = 0
        missed_by_category: Counter = Counter()
        unblank = 0
        for item_id, meta in items.items():
            answer = coded.get(item_id, {}).get("category", "")
            if not answer:
                blank += 1
                continue
            unblank += 1
            if answer == meta["true_category"]:
                correct += 1
            else:
                missed_by_category[meta["true_category"]] += 1
        per_rater[label] = {
            "n_items": len(items),
            "n_coded": unblank,
            "n_blank": blank,
            "n_recovered": correct,
            "recovery_rate": round(correct / unblank, 4) if unblank else None,
            "missed_by_true_category": dict(missed_by_category.most_common()),
            "needs_review_before_main_sheet": bool(
                unblank and (correct / unblank) < REVIEW_THRESHOLD),
        }

    labels = list(sheets)
    for item_id, meta in sorted(items.items()):
        answers = {label: sheets[label].get(item_id, {}).get("category", "")
                   for label in labels}
        recovered = {label: answers[label] == meta["true_category"]
                     for label in labels}
        per_item.append({
            "item_id": item_id,
            "true_category": meta["true_category"],
            "answers": answers,
            "recovered": recovered,
            "n_recovered": sum(recovered.values()),
            "both_raters_agree": len(set(answers.values())) == 1,
            "both_wrong_same_way": (
                len(set(answers.values())) == 1
                and all(not v for v in recovered.values())
                and all(answers.values())),
        })

    # Where both raters miss the same item in the same direction, the corpus
    # label is the thing under suspicion, not the raters.
    suspect = [row for row in per_item if row["both_wrong_same_way"]]
    by_category: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "recovered_by_both": 0})
    for row in per_item:
        bucket = by_category[row["true_category"]]
        bucket["n"] += 1
        bucket["recovered_by_both"] += int(row["n_recovered"] == len(labels))

    return {
        "generated_by": "eval/score_calibration.py",
        "what_this_is": "Pre-check on the rubric and the corpus labels, run "
                        "before the main sheet. Reported separately and never "
                        "folded into the main kappa.",
        "what_this_is_not": "Rater accuracy. Raters saw these passages in the "
                            "rubric under labelled headings, so a high score "
                            "may be recognition. A low score is the signal.",
        "no_kappa_reported": "Deliberate. Agreement on 14 recognition-assisted "
                             "items is not an inter-rater reliability estimate; "
                             "computing one here would invite it being quoted "
                             "as the study's kappa.",
        "review_threshold": REVIEW_THRESHOLD,
        "per_rater": per_rater,
        "per_category": {state: dict(counts)
                         for state, counts in sorted(by_category.items())},
        "per_item": per_item,
        "items_both_raters_missed_identically": [
            {"item_id": row["item_id"], "true_category": row["true_category"],
             "both_said": next(iter(set(row["answers"].values())))}
            for row in suspect
        ],
        "verdict": verdict(per_rater, suspect),
    }


def verdict(per_rater: dict, suspect: list) -> str:
    needs = [label for label, stats in per_rater.items()
             if stats["needs_review_before_main_sheet"]]
    parts = []
    if needs:
        parts.append(
            f"{', '.join(needs)} recovered fewer than "
            f"{REVIEW_THRESHOLD:.0%} of the calibration items. Re-read the "
            f"rubric and discuss the misses BEFORE starting the main sheet.")
    else:
        parts.append("Both raters cleared the calibration threshold; proceed to "
                     "the main sheet.")
    if suspect:
        ids = ", ".join(row["item_id"] for row in suspect)
        parts.append(
            f"{len(suspect)} item(s) ({ids}) were missed the same way by both "
            f"raters. When two independent readers agree with each other and "
            f"disagree with the corpus, the corpus label is the thing to doubt. "
            f"Record these — they are evidence about the synthetic labels, not "
            f"about the raters.")
    return " ".join(parts)


def print_report(report: dict) -> None:
    rule = "=" * 78
    print(rule)
    print("CALIBRATION PRE-CHECK — not the study kappa")
    print(rule)
    print(report["what_this_is"])
    print()
    print(f"NOT rater accuracy: {report['what_this_is_not']}")
    print()
    print("-" * 78)
    print(f"{'rater':<12}{'coded':>7}{'blank':>7}{'recovered':>11}{'rate':>8}"
          f"   review needed?")
    print("-" * 78)
    for label, stats in report["per_rater"].items():
        rate = stats["recovery_rate"]
        print(f"{label:<12}{stats['n_coded']:>7}{stats['n_blank']:>7}"
              f"{stats['n_recovered']:>11}"
              f"{'—' if rate is None else f'{rate:>7.0%}'}"
              f"   {'YES' if stats['needs_review_before_main_sheet'] else 'no'}")

    print("\n" + "-" * 78)
    print("PER CATEGORY (recovered by both raters)")
    print("-" * 78)
    for state, counts in report["per_category"].items():
        print(f"  {state:<12}{counts['recovered_by_both']}/{counts['n']}")

    print("\n" + "-" * 78)
    print("PER ITEM")
    print("-" * 78)
    labels = list(report["per_rater"])
    header = "".join(f"{label:>12}" for label in labels)
    print(f"{'item':<9}{'true':<12}{header}   flag")
    for row in report["per_item"]:
        answers = "".join(f"{row['answers'][label] or '(blank)':>12}"
                          for label in labels)
        flag = ("BOTH MISSED — SAME WAY" if row["both_wrong_same_way"]
                else "" if row["n_recovered"] == len(labels) else "miss")
        print(f"{row['item_id']:<9}{row['true_category']:<12}{answers}   {flag}")

    print("\n" + rule)
    print("VERDICT")
    print(rule)
    import textwrap
    print(textwrap.fill(report["verdict"], 78))
    print(rule)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rater1", default=DEFAULT_RATER1)
    ap.add_argument("--rater2", default=DEFAULT_RATER2)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    key = load_key(args.key)
    sheets = {
        "rater1": read_sheet(args.rater1, "rater1"),
        "rater2": read_sheet(args.rater2, "rater2"),
    }
    report = score(key, sheets)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if not args.quiet:
        print_report(report)
    print(f"\nWrote {os.path.relpath(os.path.abspath(args.out), REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
