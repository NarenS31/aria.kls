#!/usr/bin/env python3.11
"""Export randomized, condition-blinded educator annotation packets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text())
    task_by_id = {
        task["id"]: task
        for task in json.loads((ROOT / "data" / "question_bank.json").read_text())
    }
    episode_utterances = {}
    for row in artifact["rows"]:
        audit = row.get("selector_audit", {}) or {}
        signature = audit.get("signature", {}) or {}
        anchor = str(signature.get("student_anchor", "")).strip()
        if anchor:
            episode_utterances[row["episode_id"]] = anchor

    items = []
    key = []
    for index, row in enumerate(artifact["rows"]):
        item_id = f"ARIA-RATE-{index + 1:04d}"
        task = task_by_id[row["task_id"]]
        selector_audit = row.get("selector_audit", {}) or {}
        signature = selector_audit.get("signature", {}) or {}
        items.append({
            "item_id": item_id,
            "subject": row["subject"],
            "topic": row["topic"],
            "problem": task["problem"],
            "student_utterance": (
                signature.get("student_anchor")
                or episode_utterances.get(row["episode_id"], "")
            ),
            "tutor_response": row["response"],
            "problem_grounding_1_5": "",
            "student_grounding_1_5": "",
            "diagnostic_usefulness_1_5": "",
            "actionability_1_5": "",
            "learner_ownership_1_5": "",
            "learner_context_fit_1_5": "",
            "reveals_answer_yes_no": "",
            "invents_student_action_yes_no": "",
            "generic_across_problems_yes_no": "",
            "safety_fairness_concern_yes_no": "",
            "rater_notes": "",
        })
        key.append({
            "item_id": item_id,
            "episode_id": row["episode_id"],
            "condition": row["condition"],
            "model": row["model"],
            "task_id": row["task_id"],
        })

    rng = random.Random(args.seed)
    rng.shuffle(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    args.key.write_text(json.dumps({
        "seed": args.seed,
        "artifact": str(args.artifact),
        "mapping": key,
    }, indent=2) + "\n")
    print(f"Wrote {len(items)} blinded items to {args.output}")
    print(f"Wrote private condition key to {args.key}")


if __name__ == "__main__":
    main()
