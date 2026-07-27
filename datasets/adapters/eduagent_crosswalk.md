# EduAgent → ARIA cognitive-state crosswalk

This document is the authoritative mapping from EduAgent's cognitive taxonomy to
ARIA's seven cognitive states. It is kept in sync with `EDUAGENT_TO_ARIA` in
[`eduagent.py`](eduagent.py). **A state with no clean mapping maps to `None`, not
to a guess.**

## Background: what EduAgent actually records

The public EduAgent release (`github.com/EduAgent/EduAgent`) does **not** ship a
column of discrete cognitive-state labels. Instead:

- `dataset/student_demo.csv` — demographics only (no cognitive states).
- `dataset/student_answer_item_revised.csv` — per student-question rows with
  **real, gaze-derived cognitive measures**: `confusion_dur` (seconds of
  measured confusion) and `inattention_dur` (seconds of measured inattention),
  plus `accuracy`.
- `dataset/during_behavior_slide.csv` — raw gaze/mouse time series.

The adapter therefore derives a **discrete real label** from the continuous gaze
measures (see `_derive_from_measures`), thresholding:

| Gaze measure                              | Derived label   | ARIA state | Confidence |
|-------------------------------------------|-----------------|------------|------------|
| `confusion_dur > 0` (and ≥ inattention)   | `confusion`     | CONFUSED   | high       |
| `inattention_dur > 0` (no confusion)      | `inattention`   | **None**   | —          |
| `confusion_dur == 0 && inattention == 0`  | `focused`       | FLOW       | medium     |
| otherwise                                 | —               | None       | —          |

The derived label uses **only gaze measures**, never `accuracy`, so that an
accuracy-based behavioural proxy can be compared against it without circularity
(Experiment D).

## Taxonomy crosswalk (for any discrete EduAgent labels)

If a future EduAgent release exposes discrete cognitive-state labels, they are
mapped as follows. Only defensible correspondences map to a state.

| EduAgent label (normalized) | ARIA state | Confidence | Justification |
|-----------------------------|------------|------------|---------------|
| planning / plan / goal_setting | PLANNING | high | explicit strategy/goal-setting == ARIA PLANNING |
| focused / focus / engaged / concentration | FLOW | medium | sustained on-task focus ~ ARIA FLOW |
| flow | FLOW | high | direct match |
| confused / confusion | CONFUSED | high | confusion == ARIA CONFUSED |
| uncertain | CONFUSED | medium | expressed uncertainty ~ ARIA CONFUSED |
| rushing | RUSHING | high | direct match |
| careless / impulsive | RUSHING | medium | fast/impulsive responding ~ ARIA RUSHING |
| frustrated / frustration | FRUSTRATED | high | frustration == ARIA FRUSTRATED |
| stuck | STUCK | high | direct match |
| gave_up | STUCK | medium | giving up ~ ARIA STUCK |
| insight / aha / eureka | INSIGHT | high | realization == ARIA INSIGHT |
| **distracted** | **None** | — | ARIA has no "distracted" state |
| **mind_wandering** | **None** | — | ARIA has no "mind-wandering" state |
| **bored / boredom** | **None** | — | ARIA has no "bored" state |
| **inattention** | **None** | — | ARIA has no "inattention" state |
| **neutral** | **None** | — | no cognitive commitment to map |

## Consequences for validation

- Only `confusion` (→ CONFUSED) and `focused` (→ FLOW) are recoverable from the
  public EduAgent310 data. PLANNING, RUSHING, FRUSTRATED, STUCK and INSIGHT have
  no gaze-derived signal here and appear as `None`.
- `inattention` is a genuine measured state with **no ARIA equivalent**; mapping
  it to any ARIA state would be a fabrication, so it is `None`. This is itself a
  finding: ARIA's taxonomy does not cover pure inattention.
