> Edited by hand on branch `summit-text` (2026-10-21). This slide uses **2009–10 numbers only**. The 2012 and cross-dataset numbers that an earlier version showed were removed because the data terms are not confirmed. `aria_pace/make_slide.py` still generates that earlier version: do not re-run it until it is updated. Every number is listed with its source in `docs/summit/NUMBERS.md`.

# ARIA-Pace: knowing when *not* to step in

**Question.** When should an AI math tutor stay quiet, and when should it speak?

A symbolic engine checks each step. A step that is right gets no reply. After a verified wrong step, a risk estimate decides whether to offer a hint. The three settings below are **cost weights** (how much an unnecessary interruption costs compared with a delayed hint), **not types of learner**.

| Setting (interruption : delay cost) | Fewer interventions than helping after every wrong step, *under the 3-correct-in-a-row mastery definition* (stricter definitions shrink it, see Sensitivity) | Students who went on to get stuck and were never helped |
|---|---|---|
| Quiet (3 : 1) | **52%** (1.22 vs 2.54 per student) | **3.7%** (95% CI 2.2% to 5.3%) |
| Balanced (1 : 1) | 15% (2.15 vs 2.54) | 0.2% |
| Supportive (1 : 3) | 3% (2.46 vs 2.54; difference includes zero) | 0% |

**Headline.** *Under the 3-correct-in-a-row mastery definition* (stuck = no three correct answers in a row within ten attempts), the quiet setting intervened 52% less often than helping after every wrong step, on a held-out test of 597 students evaluated once. Thresholds were frozen on validation first.

**Sensitivity.** Stricter definitions shrink the saving: on validation, 4 correct in a row gives 8.6% and a moving-average rule gives 9.0%, against 49.8% for the original definition (exploratory, validation only, BKT only; no alternative has a test number).

**Limitation.** This is a replay of logged data, not a classroom: nothing here shows what a student would have done if interrupted or helped, and we make no claim that it improves learning.

**What it is built on.** The risk model is Bayesian Knowledge Tracing, a 1995 model, fitted on ASSISTments skill-builder math logs (grade level not stated), not calculus. The contribution is the decision policy, the protocol (written down first, tested once) and the negative results (neural models did not improve the decision).

---
**Speaker notes**

- Say the definition of "stuck" every time you say 52%. If asked about other definitions, use `docs/summit/QA.md` question 4.
- The closed-loop simulation is an assumption-dependent sensitivity analysis, not evidence of effect: the policy pays off only if leaving a stuck student alone is cheap. Use it only if asked (`QA.md` question 5).
- Do not describe any of this as improving learning, and do not describe the three settings as kinds of student.
- Do not show or discuss results from other datasets.

---
**Traceability.** Source files and lines for each number are in `docs/summit/NUMBERS.md`.
