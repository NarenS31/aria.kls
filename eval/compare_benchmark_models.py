#!/usr/bin/env python3.11
"""Compare two completed closed-loop benchmark artifacts episode by episode."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from research_stats import compare_conditions


METRICS = [
    "student_grounding",
    "problem_grounding",
    "question_contract",
    "no_answer_leakage",
    "misconception_targeting",
    "conciseness",
    "composite",
]


def means(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    return {
        condition: {
            metric: round(
                sum(float(row[metric]) for row in items) / len(items), 4
            )
            for metric in METRICS + ["latency_seconds"]
        }
        for condition, items in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    treatment = json.loads(args.treatment.read_text())
    base_name = base["model"]
    treatment_name = treatment["model"]

    paired_rows = []
    for artifact in (base, treatment):
        for row in artifact["rows"]:
            if row["condition"] != "full_closed_loop":
                continue
            paired_rows.append({
                **row,
                "condition": f"{artifact['model']}::full_closed_loop",
            })

    result = {
        "evidence_tier": "SIMULATED_PROXY",
        "base_model": base_name,
        "treatment_model": treatment_name,
        "n_paired_episodes": base["n_episodes"],
        "within_model_condition_means": {
            base_name: means(base["rows"]),
            treatment_name: means(treatment["rows"]),
        },
        "full_pipeline_treatment_minus_base": compare_conditions(
            paired_rows,
            treatment=f"{treatment_name}::full_closed_loop",
            control=f"{base_name}::full_closed_loop",
            metrics=METRICS,
        ),
        "latency_note": (
            "Runs were executed sequentially, but latency still depends on local "
            "model loading and is descriptive rather than a hardware-independent benchmark."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
