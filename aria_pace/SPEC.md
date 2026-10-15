# ARIA-Pace — Specification (frozen before any modelling)

Status: **frozen 2026-10-15**, committed before any model is fit. Any later change is an *amendment* appended in Section 12 with a date and a reason; the original text is not edited. Everything here is for offline replay on logged data. It does not measure learning, does not test the SymPy answer-check layer, and says nothing about any clinical group.

## 1. Data and unit of analysis

- **Dataset:** ASSISTments 2009–10 Skill Builder, corrected collapsed file `skill_builder_data_corrected_collapsed.csv` (64,412,812 bytes, sha256 `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`). The loader refuses to run on a file with a different checksum. See `docs/aria_pace/DATA.md`.
- **Cleaning, in this order, each step logged with counts in `results/data_report.json`:**
  1. Drop rows with null `skill_id`. The count and percentage are reported.
  2. Assert `(user_id, order_id)` is unique and `correct` is in {0, 1}; stop otherwise.
  3. Sort by `(user_id, order_id)`.
- **Unit ("sequence"):** one student's rows within one problem-set assignment, i.e. `(user_id, assignment_id)`, ordered by `order_id`. Skill-builder mastery is defined per problem set, so this matches the Wang et al. / Kai et al. "student–problem-set pair". (In this file 636 of 3,415 assignments, 18.6%, contain more than one skill string, so skill is not the unit; 64,818 (user, assignment) sequences exist after dropping null-skill rows, before any further filtering.)
- **Opportunity index** `i = 1, 2, ...` is **recomputed** as the rank of the row by `order_id` within the sequence. The file's own `opportunity` column is not used (it agrees with this rank in only 83.7% of rows on the skill-level grouping).
- **Knowledge-component (KC) key of a sequence** = the `skill_id` string of its **first** opportunity (so it is known at prediction time). It is used only by the PFA and BKT baselines.
- **`correct`** = the log's own first-attempt correctness: 1 only if the first action was a correct answer with no hint (verified: all 223,818 `correct`=1 rows have `first_action`=0 and 223,805 have `hint_count`=0). Replay uses this value as ground truth; the SymPy verification layer from the demo is **not** evaluated here.
- **Not used for features or labels:** `first_action` (its code meanings are inferred, not documented), `attempt_count` (outlier up to 3,824), `ms_first_response` (7 negative values), `hint_*`, affect columns. `first_action` appears only in descriptive counts in `data_report.json`. Stage 3 baselines use `correct` only.

## 2. Mastery and the wheel-spinning label

- **Mastery** of a sequence occurs at the first opportunity `m` such that `correct` is 1 at opportunities `m-2, m-1, m` (three consecutive correct). If there is no such opportunity, `m = infinity`. `n` = number of opportunities in the sequence.
- **Labelled in this study as:** "following Wang et al. / Kai et al., attributed to Beck & Gong". The primary text of Beck & Gong 2013 was not read; this study never claims an exact Beck & Gong definition.
- **Variant A (primary, ">= 10"):**
  - `y = 0` if `m <= 10` (mastered within the first 10 opportunities).
  - `y = 1` if `m > 10` and `n >= 10`.
  - **Unlabelled leaver** if `m > 10` and `n < 10`.
- **Variant B (sensitivity, "> 10"):**
  - `y = 0` if `m <= 10`.
  - `y = 1` if `m > 10` and `n >= 11`.
  - **Unlabelled leaver** if `m > 10` and `n <= 10`.
- A and B differ only for sequences with `m > 10` and `n = 10` exactly (positive in A, unlabelled in B). Both are reported everywhere; **A is primary and fixed now**.
- Wheel-spinning "determination point" `T = 10` opportunities.

## 3. Prediction scope at k in {3, 5, 8}

For each `k` the question is: after the student's first `k` opportunities, will this sequence be wheel-spinning (`y = 1`)?

- **Eligible:** sequences with `n >= k` **and** `m > k` (not mastered at or before opportunity `k`). Sequences with `n < k`, or `m <= k`, are excluded, and both counts are reported separately.
- **Scored:** eligible sequences that are labelled. Eligible **unlabelled leavers are excluded and their count is reported** at each `k`, for each variant.
- **Inputs:** only opportunities `1..k` of the sequence (their `correct` values and the KC key). Nothing after `k` may influence a score. Fitting data for PFA/BKT dynamics is *all rows of training-split students' sequences* (the dynamics model may see rows after `k` of training students; scoring rows never do).
- **Reported per k, per variant, per split used:** total sequences, excluded `n < k`, excluded `m <= k`, eligible, eligible leavers, scored N (sequences and students), positives, label rate.
- **Known selection bias, to be stated in RESULTS.md:** removing leavers uses information after `k`; mastered sequences are labelled 0 by construction; neither is corrected.

## 4. Split

- **Level:** student (`user_id`), after step 1 of cleaning. No sequence, row, or student may appear in more than one split; asserted in code and in a test.
- **Proportions:** 60% train / 20% validation / 20% test of distinct students.
- **Seed:** `20261015`, NumPy `RandomState`, applied to the sorted array of unique `user_id`. The split is written to `results/split.json` (id lists + sha256).
- **Test split is frozen.** Nothing may be evaluated on it until all model and threshold choices are final. Stage 3 reports **validation only**. The test split is evaluated **once**, in Stage 5, for all methods together; the run records a lock file `results/TEST_LOCK` when it happens and refuses to run again.

## 5. Baselines (Stage 3)

All produce a risk score in [0,1] for `y = 1` from the first `k` opportunities.

1. **Always-help:** score = 1 for every sequence.
2. **Three-wrong-in-a-row rule:** score = 1 if `correct = 0` at three consecutive opportunities within `1..k`, else 0.
3. **PFA** (Pavlik, Cen & Koedinger 2009, Eq. 3, verified against the paper PDF): logit P(correct at i) = sum over KC of (beta + gamma * s + rho * f), with `s`, `f` = counts of earlier correct/incorrect responses **within the sequence**. Deviations, stated: L2-regularised logistic regression; per-KC `beta`, `gamma`, `rho` plus shared global terms so unseen KCs fall back; counts restricted to the sequence. Regularisation strength `C` is chosen from {0.01, 0.1, 1, 10} by **validation log-loss of next-response correctness** (not on the wheel-spinning label). The wheel-spinning risk is `P(no mastery in opportunities k+1..10 | state after k)`, computed **exactly** by dynamic programming over (successes, current correct run) using the fitted PFA, updating the counts after each simulated response.
4. **BKT** (Corbett & Anderson 1995; four-parameter form, standard definition): per-KC prior, learn, guess, slip fitted by EM on training sequences, 30 iterations, fixed initial values, bounds guess, slip in [0.01, 0.30], learn in [0.01, 0.60], prior in [0.01, 0.99] (bounds are fixed a priori, not tuned). KCs with fewer than `min_seq` training sequences use pooled parameters; `min_seq` chosen from {10, 30} by validation log-loss of next-response correctness. Risk = `P(no mastery in k+1..10 | first k responses)` by exact DP over (latent knowledge, current correct run) after forward filtering.
   - The citation for the four-parameter formulation is background knowledge: the Corbett & Anderson abstract was not opened (see LITERATURE.md 1.4). RESULTS.md will say so.

The PFA and BKT "mastery" they simulate is the same three-consecutive-correct rule as the label, so the label and the risk target the same event up to the leaver selection.

## 6. Metrics for Stage 3

For each baseline, `k`, and variant (validation split):
- **AUC** (ROC). Constant scores give 0.5; the binary rule gives the AUC of its single threshold.
- **Brier score** and **log-loss** (scores clipped to [1e-6, 1-1e-6]).
- **Calibration:** expected calibration error with 10 equal-mass bins (ECE) and the reliability table (mean score, observed rate, count per bin); calibration intercept and slope from a logistic regression of `y` on logit(score), reported for continuous scores only (PFA, BKT). For binary or constant scores, the observed rate per score value is reported instead.
- **Base rate** (label rate) and **N**, always alongside.
- **95% confidence intervals** for AUC by bootstrap over **students** (1,000 resamples, seed `20261016`).

## 7. Policy objects (defined now, implemented in Stage 4)

All quantities are in **opportunities**; the file has no wall-clock time.

- **Decision point:** after observing opportunity `i` for `i = 3, ..., 9` of a sequence not yet mastered (`m > i`). Help given at decision point `i` takes effect from opportunity `i + 1`.
- **Verified error:** `correct = 0` at opportunity `i`, i.e. the log's own first-attempt-wrong. This includes problems where the first action was a hint request (the log marks those `correct = 0`); the count of such problems is reported descriptively.
- **Intervention:** the policy fires at decision point `i` iff a verified error occurs at `i` **and** risk at `i` >= `tau`. (Risk at `i` uses opportunities `1..i` only.) Always-help fires at every verified error; the three-wrong rule fires when its condition first holds.
- **Unnecessary interruption:** an intervention in a sequence with `y = 0` (it mastered within 10 opportunities without help, in the log). Reported as count per student and as share of interventions.
- **Missed help:** a sequence with `y = 1` in which the policy never fires at a decision point `i <= 9`. Reported as count per student and as share of `y = 1` sequences.
- **Lead time:** for `y = 1` sequences where the policy fires, `T - i_first` with `T = 10` and `i_first` the first firing point, in opportunities.
- **Help delay:** for `y = 1` sequences, `i_first - i_err`, where `i_err` is the first verified error at a decision point (what always-help would act on). If the policy never fires, delay is **censored** at `T - i_err` and counted as missed. Mean delay is reported both including and excluding censored sequences. A sequence with no verified error at any decision point has no help opportunity and is excluded from delay (count reported).
- **Profiles (quiet / balanced / supportive):** explicit cost weights `(c_interrupt, c_delay)` on unnecessary interruption versus help delay and missed help, defined in Stage 4 *before* looking at validation results, and `tau` chosen per profile on validation only.
- **Limitation, part of the definition:** replay cannot say what would have happened had the student been interrupted; `y = 0` sequences are assumed not to need help, and `y = 1` sequences are assumed to benefit from earlier help. These are assumptions, not findings.

## 8. Seeds and reproducibility

- Split seed `20261015`; bootstrap seed `20261016`; all model fitting is deterministic (fixed initial values, fixed `random_state`). No other randomness.
- One command from the repository root: `python -m aria_pace.run_baselines` (after `pip install -r aria_pace/requirements.txt` and the download step in `docs/aria_pace/DATA.md`). It writes `results/data_report.json`, `results/split.json`, `results/label_counts.json`, `results/baselines_val.json`.
- Every number in a later `RESULTS.md` must trace to a file under `aria_pace/results/`.

## 9. Required statements for RESULTS.md

1. Replay uses the logs' own correctness; the SymPy verification layer is not evaluated.
2. Grade level of the 2009–10 data is not stated by the sources; no grade claim is made.
3. `first_action` code meanings are inferred, not documented; they are not used in the models.
4. No learning-gain claim; no claim about ADHD, math anxiety, or any clinical or demographic group; profiles are cost settings.
5. The wheel-spinning label is "following Wang et al. / Kai et al., attributed to Beck & Gong".
6. The test split was evaluated once (Stage 5).

## 10. Out of scope

Sequence neural models (Stage 4 only if they clearly beat logistic regression on validation), the demo path (`agent/demo_*.py`, `ui/demo_app.py`, `scripts/demo_harness.py`, `data/demo_problems.json`), and anything under `eval/`.

## 11. Files

`aria_pace/{loader,split,labels,baselines,metrics,run_baselines}.py`, `aria_pace/tests/`, `aria_pace/requirements.txt`, `aria_pace/results/*.json`. Raw data lives only in the gitignored `aria_pace/data/raw/`.

## 12. Amendments

(none)
