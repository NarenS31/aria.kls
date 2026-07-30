#!/usr/bin/env python3.11
"""Machine-check the ARIA research package without implying human approval."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REQUIRED = (
    "README.md",
    "EVIDENCE_MAP.md",
    "OBSERVABLE_REASONING_TAXONOMY.md",
    "TASK_VALIDATION_PROTOCOL.md",
    "SYSTEM_EVALUATION_PREREGISTRATION.md",
    "LANGUAGE_VALIDATION_PROTOCOL.md",
    "FEASIBILITY_PROTOCOL.md",
    "STUDENT_STUDY_PREREGISTRATION.md",
    "OUTCOME_MEASUREMENT_PLAN.md",
    "ETHICS_AND_PRIVACY.md",
    "DATA_MANAGEMENT_PLAN.md",
    "ADHD_POSITION.md",
    "MODEL_CARD.md",
    "DATA_CARD.md",
    "EXTERNAL_ACTIONS.md",
    "templates/PARENT_PERMISSION_DRAFT.md",
    "templates/STUDENT_ASSENT_DRAFT.md",
    "templates/EDUCATOR_REVIEWER_CONSENT_DRAFT.md",
)


def main() -> None:
    checks = {}
    for relative in REQUIRED:
        checks[f"exists:{relative}"] = (RESEARCH / relative).is_file()

    schema = json.loads(
        (RESEARCH / "schemas" / "task_model.schema.json").read_text()
    )
    tasks = json.loads(
        (RESEARCH / "packets" / "task_models_draft.json").read_text()
    )
    validator = Draft202012Validator(schema)
    schema_errors = [
        f"{task.get('id')}: {error.message}"
        for task in tasks
        for error in validator.iter_errors(task)
    ]
    checks["task_count_is_100"] = len(tasks) == 100
    checks["task_schema_valid"] = not schema_errors
    checks["all_tasks_explicitly_pending"] = all(
        task["educator_validation"]["status"] == "pending"
        and task["educator_validation"]["reviewer_count"] == 0
        for task in tasks
    )
    checks["no_private_language_packet_tracked"] = not (
        RESEARCH / "packets" / "moves_rater.csv"
    ).exists()
    output = {
        "machine_readiness": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "schema_errors": schema_errors,
        "external_gates": {
            "institutional_determination": "pending",
            "educator_task_reviews": "pending",
            "blinded_intervention_ratings": "pending",
            "independent_real_language_labels": "pending",
            "student_feasibility_data": "pending",
            "controlled_learning_outcomes": "pending",
        },
        "warning": (
            "Machine readiness means files are present and internally valid. "
            "It is not IRB approval, educator validation, or evidence of effect."
        ),
    }
    path = RESEARCH / "READINESS_STATUS.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if output["machine_readiness"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
