# "Would you step in?" — analysis plan (committed before the activity UI)

Branch `summit-activity`, started from the tag `summit-2026-10-24`. This file fixes how the teacher responses will be analysed **before** any are collected, so the analysis cannot
be chosen after seeing the data.

## 1. What this is, and what it is not

Visitors (mostly teachers) see 8 short moments from a student's work and decide, for each, whether a tutor should **step in** or **stay quiet**. Afterwards they see what ARIA decided and
say whether they agree. The goal is a descriptive picture: do teachers agree with each other about timing, and where does ARIA sit relative to them?

It is **not** evidence that ARIA's timing is correct or that it helps students. Agreement with a convenience sample of visitors on hypothetical moments is not validity. There is no learning
outcome, no student data, no clinical or demographic claim about students, and ARIA's profiles are cost settings, not categories of learners.

## 2. The moments

- Exactly 8 moments, regenerated from **this frozen build** by running the unmodified `DemoTutor` (`agent/demo_tutor.py`, tag `summit-2026-10-24`) over turns supplied by the owner
  (`scripts/make_teacher_moments.py`). Output: `data/teacher_moments.json`.
- For each moment the **ground truth is ARIA's logged decision** on that turn (`intervened` and `intervention_type` from the turn's metadata), not a human label.
- Two properties are computed from the build, never typed by hand:
  - `words_only`: `check_answer(message).status == "none"`, i.e. the message has no math for SymPy to check.
  - `verified_wrong`: `check_answer(message).status == "wrong"`.
- **ARIA never stays quiet on a verified wrong step.** `derive_state` maps `status == "wrong"` to STUCK (`agent/demo_tutor.py:111-112`), and the test suite asserts that every wrong answer is met with an
  intervention (`tests/test_demo_tutor.py:34-36`). So on `verified_wrong` moments ARIA's decision is fixed by design; agreement there says nothing about ARIA's timing judgement and those moments
  are reported separately (section 5d).
- Type mapping used to compare what kind of help (ARIA's five intervention types, `eval/metacognition/jitai.py:419-448`, to the activity's five answer choices). This mapping is a judgement call and is
  stated here so it can be challenged before data arrive:

  | ARIA intervention type | Activity category |
  |---|---|
  | `hint` | hint |
  | `task_chunking` | hint (a smaller step to try; amended, see Amendment 1) |
  | `reflection` | prompt (a question about what made it click) |
  | `encouragement` | encouragement |
  | `break_suggestion` | break |
  | (teacher chooses "other") | other, never matches ARIA |

## 3. What is recorded

One row per moment answered, appended to `data/demo_logs/teacher_taps.csv` (git-ignored directory), columns exactly:
`session_id, moment_id, role, years_band, grade_band, teacher_decision, teacher_type, aria_decision, aria_type, agree, timestamp`.

- `session_id`: random per visitor, not linked to anything. `role`, `years_band`, `grade_band`: optional dropdowns, default "prefer not to say". **No free text, no names, no network.**
- `teacher_decision` is `step_in` or `stay_quiet`; `teacher_type` is one of the five categories and empty when `stay_quiet`; `aria_decision`/`aria_type` are copied from `teacher_moments.json`;
  `agree` is the visitor's own Agree/Disagree **after** the reveal (`agree` or `disagree`).
- Skip on the consent screen records nothing. A visitor who leaves midway leaves only the rows already answered; analysis uses completed moments only.
- Ethics/consent: this is a demo activity. Whether the venue or school requires review is for the owner to decide; nothing in this repository states that a review took place.

## 4. Minimum sample before anything is reported

A statistic is reported only if **every moment in the set it summarises has at least 15 distinct sessions** (n = distinct `session_id` per `moment_id`). Otherwise the analysis prints
`not reported (n = ...)` for that set. For any subgroup (role, years band, grade band) a table cell with fewer than 15 sessions is suppressed too, and counts below 5 are shown as `<5`.
No hypothesis tests are run on subgroups; they are descriptive.

## 5. Analyses (all descriptive)

**a. Teacher agreement with each other.** Fleiss' kappa on `step_in` vs `stay_quiet`, one rating per teacher per moment. Fleiss' kappa needs the same number of raters per moment; if counts differ, each
moment is randomly subsampled (seed 20261024) to the smallest count, and that count is printed. Reported for (i) all moments, (ii) moments with math, (iii) words-only moments, each separately.
Kappa is reported with a percentile bootstrap 95% interval (resampling teachers, 1,000 draws, seed 20261025). Labels such as "moderate" are conventions, not thresholds, and are not used to decide anything.

**b. ARIA versus the teacher majority.** For each moment: the share of teachers choosing `step_in`, the majority decision (`split` if the share is between 40% and 60%), and ARIA's decision.
Then a 2x2 confusion table, rows = teacher majority (`step_in`, `stay_quiet`), columns = ARIA, counted over **moments**, with `split` moments listed separately. Also the pooled response-level
agreement rate (a teacher's decision equals ARIA's), with a bootstrap 95% interval over teachers.

**c. Words-only moments are analysed separately and never pooled** with moments that contain math. There, ARIA's decision comes from language cues only (the affect flags need an explicit feeling word,
`agent/demo_tutor.py` `_flags`), so it is the weaker half of the system and the more interesting comparison.

**d. Verified-wrong moments** are reported on their own: the teacher `stay_quiet` rate on a verified wrong step, with ARIA fixed at `step_in` by design (section 2). A high `stay_quiet` rate there is a finding
about teachers' willingness to wait, not an ARIA error, and it is not counted in the ARIA-vs-majority table of 5b.

**e. Kind of help.** Among responses where both teacher and ARIA say `step_in`: the proportion whose categories match under the mapping in section 2, and the full teacher-category distribution per moment.

**f. After the reveal.** The proportion who press Agree, overall and split by whether their own decision matched ARIA's. The reveal comes after the decision is recorded, so the decision is not contaminated;
the Agree/Disagree answer may be (people tend to agree with what they have just been shown), so it is reported as descriptive only.

**g. Order and fatigue.** Moments are shown in the file's fixed order. The analysis reports the `step_in` share by position for what it is worth; order is not randomised, so position and moment are confounded.

## 6. Limits, stated in advance

Convenience sample of visitors at one event; hypothetical, scripted moments; text-only context (no tone, no history beyond what is shown); teachers' views differ by school and subject; "step in" is a binary
simplification of a continuum. ARIA's decision is a rule-based policy over verified correctness, not a ground truth about what is pedagogically best.

## 7. Reproduce

```
/usr/bin/python3 scripts/analyze_teacher_taps.py data/demo_logs/teacher_taps.csv --moments data/teacher_moments.json
```

Every number in any write-up must come from that command's output.

## Amendment 1 (2026-10-16, before any teacher response was collected)

The owner fixed the type mapping: `hint` -> hint, **`task_chunking` -> hint** (was: prompt), `reflection` -> prompt, `encouragement` -> encouragement, `break_suggestion` -> break.
Only the `task_chunking` row changes. The table in section 2 above shows the amended mapping; nothing else in this plan changes. No data existed when this was decided, so it cannot depend on responses.
