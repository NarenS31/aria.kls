# ARIA research program

This directory separates what ARIA is designed to do from what evidence has
actually demonstrated.

## Current evidence status

- **Supported implementation properties:** ARIA conditions a response on a
  keyed task model, the current utterance, learner context, and recent history;
  deterministic gates check answer leakage, repetition, and grounding.
- **Development evidence only:** synthetic and weak-label benchmarks test
  software behavior, not student learning.
- **Pending independent evidence:** educator ratings, human-labeled student
  language, feasibility with real students, and controlled learning outcomes.

No document in this directory is IRB approval, legal advice, a completed
preregistration, or evidence of effectiveness. Bracketed fields must be
completed with a qualified research mentor and reviewing institution before
recruitment.

## Research sequence

1. `EVIDENCE_MAP.md`: source-to-design traceability.
2. `OBSERVABLE_REASONING_TAXONOMY.md`: human annotation target.
3. `TASK_VALIDATION_PROTOCOL.md`: educator review of the 100-task bank.
4. `SYSTEM_EVALUATION_PREREGISTRATION.md`: blinded response-quality study.
5. `FEASIBILITY_PROTOCOL.md`: usability and safety pilot.
6. `STUDENT_STUDY_PREREGISTRATION.md`: controlled learning study.
7. `OUTCOME_MEASUREMENT_PLAN.md`: independent learning, transfer, calibration,
   and retention measurement.
8. `ETHICS_AND_PRIVACY.md` and `DATA_MANAGEMENT_PLAN.md`: participant
   protections.
9. `ADHD_POSITION.md`, `MODEL_CARD.md`, and `DATA_CARD.md`: scope, intended
   use, data provenance, and prohibited claims.
10. `templates/`: drafts for institutional review, not ready-to-use consent
   forms.
11. `EXTERNAL_ACTIONS.md`: work that code cannot perform.

## Claim gates

| Claim | Minimum gate |
|---|---|
| “ARIA enforces answer-safety and grounding checks” | Reproducible software tests |
| “Educators rate ARIA responses above a baseline” | Blinded independent ratings, agreement, paired analysis |
| “ARIA recognizes observable reasoning moves” | Student-separated real-text test set with independent human labels |
| “ARIA is usable and acceptable” | Prospective feasibility pilot |
| “ARIA improves learning” | Preregistered controlled study with an independent outcome |
| “ARIA improves transfer” | Unprompted performance on new tasks with ARIA absent |
| “ARIA benefits students with ADHD” | Appropriately powered, reviewed population-specific study |
