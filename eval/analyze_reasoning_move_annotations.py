#!/usr/bin/env python3.11
"""Measure agreement for independently coded observable reasoning moves."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reasoning_moves import MOVE_DEFINITIONS  # noqa: E402


def read(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


def binary(value: str) -> int:
    value = value.strip().lower()
    if value not in {"yes", "no"}:
        raise ValueError(f"Incomplete yes/no rating: {value!r}")
    return int(value == "yes")


def kappa(a: list[int], b: list[int]) -> float:
    if not a:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    pa = sum(a) / len(a)
    pb = sum(b) / len(b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.ratings) < 2:
        raise SystemExit("At least two independent rating files are required.")
    raters = {path.stem: read(path) for path in args.ratings}
    common = set.intersection(*(set(rows) for rows in raters.values()))
    result = {"n_common_items": len(common), "rater_pairs": {}}
    for left, right in itertools.combinations(sorted(raters), 2):
        codes = {}
        for code in MOVE_DEFINITIONS:
            column = f"{code}_yes_no"
            a = [binary(raters[left][item][column]) for item in common]
            b = [binary(raters[right][item][column]) for item in common]
            both_positive = sum(x == y == 1 for x, y in zip(a, b))
            positive_total = sum(a) + sum(b)
            codes[code] = {
                "prevalence_rater_1": round(sum(a) / len(a), 4),
                "prevalence_rater_2": round(sum(b) / len(b), 4),
                "raw_agreement": round(
                    sum(x == y for x, y in zip(a, b)) / len(a), 4
                ),
                "positive_agreement": round(
                    2 * both_positive / positive_total, 4
                ) if positive_total else 1.0,
                "cohens_kappa": round(kappa(a, b), 4),
            }
        result["rater_pairs"][f"{left}__{right}"] = codes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
