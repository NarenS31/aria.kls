# Task-model validation protocol

## Purpose

ARIA can only give problem-specific guidance when the problem model is correct.
The existing 100 tasks are internally authored drafts. Passing a JSON schema
does not make them educator validated.

## Reviewers

- At least two reviewers with relevant teaching or curriculum expertise review
  every task independently.
- Mathematics and English/science tasks are assigned only to reviewers with
  relevant subject competence.
- Reviewers disclose prior involvement with ARIA. At least one reviewer per
  task should be independent of the development team.

## Review dimensions

Each reviewer evaluates:

1. Prompt clarity and age appropriateness.
2. Correctness and completeness of acceptable answers.
3. Correctness of every solution-path step.
4. Whether alternative valid strategies are missing.
5. Whether each misconception is plausible and observable from student work.
6. Whether misconception repair goals are correct.
7. Whether hints increase gradually and avoid premature answer disclosure.
8. Whether the scoring criteria distinguish answer from reasoning.
9. Accessibility, cultural loading, ambiguity, and safety concerns.
10. Overall disposition: approve, revise, or reject.

## Decision rule

- A task becomes `approved` only after two reviewers approve every correctness
  dimension and all blocking concerns are resolved.
- Disagreement is adjudicated by a third qualified reviewer.
- Original reviews are retained. Adjudication never overwrites them.
- Any content change after approval creates a new task version and requires
  re-review of affected fields.

## Leakage control

The research task bank is split by underlying template or concept family, not
only by item ID. Paraphrases and numeric variants of a task must remain in the
same split. Student-study post-tests are authored or reviewed independently and
are never exposed during tutoring.

## Artifacts

Run:

```bash
python3.11 eval/prepare_task_review.py
```

This creates:

- `research/packets/task_models_draft.json`
- `research/packets/task_review.csv`

Generated drafts remain `pending` until real reviews are entered and locked.

