#!/usr/bin/env python3.11
"""Convert ARIA's product bank into educator-reviewable research artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "question_bank.json"
SCHEMA = ROOT / "research" / "schemas" / "task_model.schema.json"
PACKETS = ROOT / "research" / "packets"


def research_task(task: dict) -> dict:
    ideas = list(task["key_ideas"])
    steps = list(task["solution_steps"])
    misconceptions = list(task["common_misconceptions"])
    return {
        "schema_version": "1.0",
        "id": task["id"],
        "subject": task["subject"],
        "topic": task["topic"],
        "difficulty": task["difficulty"],
        "prompt": task["problem"],
        "learning_objective": (
            f"Apply {ideas[0]} while solving a {task['topic']} task."
        ),
        "acceptable_answers": [str(task["answer"])],
        "solution_paths": [{
            "path_id": "canonical-draft",
            "steps": [
                {
                    "step_id": f"s{index + 1}",
                    "description": step,
                    "target_idea": ideas[min(index, len(ideas) - 1)],
                }
                for index, step in enumerate(steps)
            ],
        }],
        "misconceptions": [
            {
                "misconception_id": f"m{index + 1}",
                "description": mistake,
                "observable_evidence": (
                    "Pending educator-authored example of student work that "
                    "would support this interpretation."
                ),
                "repair_goal": (
                    f"Help the student use {ideas[min(index, len(ideas) - 1)]} "
                    "without revealing the final answer."
                ),
            }
            for index, mistake in enumerate(misconceptions)
        ],
        "hint_ladder": [
            {
                "level": 1,
                "purpose": "retrieval cue",
                "text": f"Which part of {ideas[0]} seems relevant here?",
                "answer_revealing": False,
            },
            {
                "level": 2,
                "purpose": "student-generated next step",
                "text": (
                    "What is the smallest operation, claim, or evidence choice "
                    "you could test next?"
                ),
                "answer_revealing": False,
            },
            {
                "level": 3,
                "purpose": "direct strategy scaffold",
                "text": steps[0],
                "answer_revealing": False,
            },
        ],
        "scoring": {
            "final_answer_points": 1,
            "reasoning_points": 2,
            "criteria": [
                f"Uses or explains {ideas[0]}.",
                "Shows a coherent intermediate step.",
                "Checks or supports the final response.",
            ],
        },
        "provenance": {
            "authoring_method": "ARIA internal draft; converted automatically",
            "source": "scripts/build_question_bank.py",
            "license": "Repository license; external validation pending",
        },
        "educator_validation": {
            "status": "pending",
            "reviewer_count": 0,
            "approved_at": None,
            "notes": "Machine-converted draft; not educator validated.",
        },
    }


def review_row(task: dict) -> dict:
    return {
        "task_id": task["id"],
        "subject": task["subject"],
        "topic": task["topic"],
        "prompt": task["prompt"],
        "reviewer_id": "",
        "relevant_subject_expertise_yes_no": "",
        "prompt_clear_1_5": "",
        "answers_correct_complete_1_5": "",
        "solution_steps_correct_complete_1_5": "",
        "missing_valid_strategy_yes_no": "",
        "misconceptions_plausible_observable_1_5": "",
        "hints_scaffold_without_leakage_1_5": "",
        "scoring_criteria_valid_1_5": "",
        "age_appropriate_1_5": "",
        "accessibility_or_bias_concern_yes_no": "",
        "blocking_error_yes_no": "",
        "disposition_approve_revise_reject": "",
        "reviewer_notes": "",
    }


def main() -> None:
    tasks = [research_task(task) for task in json.loads(BANK.read_text())]
    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
    errors = [
        f"{task['id']}: {error.message}"
        for task in tasks
        for error in validator.iter_errors(task)
    ]
    if errors:
        raise SystemExit("\n".join(errors))
    PACKETS.mkdir(parents=True, exist_ok=True)
    (PACKETS / "task_models_draft.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
    )
    rows = [review_row(task) for task in tasks]
    with (PACKETS / "task_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated schema for {len(tasks)} pending task drafts.")
    print(f"Wrote review packet to {PACKETS / 'task_review.csv'}")


if __name__ == "__main__":
    main()
