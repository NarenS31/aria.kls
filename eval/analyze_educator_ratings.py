#!/usr/bin/env python3.11
"""Analyze locked, condition-blinded educator ratings at episode level."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .research_stats import compare_conditions
except ImportError:  # Direct script execution: python eval/analyze_educator_ratings.py
    from research_stats import compare_conditions


ORDINAL = (
    "problem_grounding_1_5",
    "student_grounding_1_5",
    "diagnostic_usefulness_1_5",
    "actionability_1_5",
    "learner_ownership_1_5",
    "learner_context_fit_1_5",
)
BINARY = (
    "reveals_answer_yes_no",
    "invents_student_action_yes_no",
    "generic_across_problems_yes_no",
    "safety_fairness_concern_yes_no",
)
PRIMARY = (
    "student_grounding_1_5",
    "diagnostic_usefulness_1_5",
    "actionability_1_5",
)


def weighted_kappa(a: list[int], b: list[int], levels: list[int]) -> float:
    if not a:
        return 0.0
    if a == b:
        return 1.0
    size = max(1, len(levels) - 1)
    counts_a, counts_b = Counter(a), Counter(b)
    observed = sum(
        1 - ((x - y) / size) ** 2 for x, y in zip(a, b)
    ) / len(a)
    expected = sum(
        (counts_a[x] / len(a))
        * (counts_b[y] / len(b))
        * (1 - ((x - y) / size) ** 2)
        for x in levels
        for y in levels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def read_rater(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


def binary_value(value: str) -> int:
    normalized = value.strip().lower()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"Expected yes/no, found {value!r}")
    return int(normalized == "yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ratings) < 2:
        raise SystemExit("At least two independently completed rating files are required.")

    raters = {path.stem: read_rater(path) for path in args.ratings}
    mapping = {
        row["item_id"]: row
        for row in json.loads(args.key.read_text())["mapping"]
    }
    common = set(mapping)
    for rows in raters.values():
        common &= set(rows)
    if not common:
        raise SystemExit("No commonly rated item IDs matched the private key.")

    agreement = {}
    for left, right in itertools.combinations(sorted(raters), 2):
        pair = {}
        for metric in ORDINAL:
            a = [int(raters[left][item][metric]) for item in common]
            b = [int(raters[right][item][metric]) for item in common]
            pair[metric] = {
                "n": len(a),
                "quadratic_weighted_kappa": round(
                    weighted_kappa(a, b, [1, 2, 3, 4, 5]), 4
                ),
                "raw_agreement": round(
                    sum(x == y for x, y in zip(a, b)) / len(a), 4
                ),
            }
        for metric in BINARY:
            a = [binary_value(raters[left][item][metric]) for item in common]
            b = [binary_value(raters[right][item][metric]) for item in common]
            pair[metric] = {
                "n": len(a),
                "cohens_kappa": round(weighted_kappa(a, b, [0, 1]), 4),
                "raw_agreement": round(
                    sum(x == y for x, y in zip(a, b)) / len(a), 4
                ),
            }
        agreement[f"{left}__{right}"] = pair

    rows = []
    for item in sorted(common):
        key = mapping[item]
        row = {
            "item_id": item,
            "episode_id": key["episode_id"],
            "condition": key["condition"],
        }
        for metric in ORDINAL:
            row[metric] = sum(
                int(rater[item][metric]) for rater in raters.values()
            ) / len(raters)
        for metric in BINARY:
            # Safety event rate, so yes=1 remains adverse.
            row[metric] = sum(
                binary_value(rater[item][metric]) for rater in raters.values()
            ) / len(raters)
        rows.append(row)

    comparisons = {}
    for control in ("generic", "problem_only", "turn_grounded"):
        comparisons[f"full_closed_loop_vs_{control}"] = compare_conditions(
            rows,
            treatment="full_closed_loop",
            control=control,
            metrics=list(PRIMARY),
            seed=20260729,
        )
    adverse_rates = {
        condition: {
            metric: round(
                sum(row[metric] for row in condition_rows) / len(condition_rows),
                4,
            )
            for metric in BINARY
        }
        for condition in sorted({row["condition"] for row in rows})
        for condition_rows in [[row for row in rows if row["condition"] == condition]]
    }
    output = {
        "status": "completed human ratings",
        "n_items": len(rows),
        "raters": sorted(raters),
        "agreement": agreement,
        "primary_comparisons": comparisons,
        "adverse_event_rates": adverse_rates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
