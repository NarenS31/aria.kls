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

### Amendment 1 — Stage 4 policy definitions (2026-10-15, before any Stage 4 modelling)

Reason: SPEC Section 7 left profile weights, scope, and several operational details open. This amendment fixes them before the risk head is trained or any policy is evaluated. Nothing in Sections 1–11 is changed. The test split is not touched in Stage 4.

**A1.1 Scope.** Policy replay uses sequences with `m > 3` and `n >= 3` (at least one decision point). Primary: **labelled sequences only (variant A), unlabelled leavers dropped**. Sensitivity ("leavers as non-wheel-spinning"): the unlabelled variant-A leavers (`m > 10`, `n < 10`) are added with `y = 0`. A leaver's decision points stop at `min(9, n)`. Policy replay uses variant A only; variant B is not run for policies.

**A1.2 Decision points and firing rule.** Decision point `i` in `3..min(9, n)` exists iff `m > i`. The policy **fires at `i`** iff (a) opportunity `i` is a *verified error* and (b) risk at `i` >= `tau`, where risk at `i` is computed from opportunities `1..i` only. **At most one intervention per sequence**: the first firing; later decision points of that sequence are ignored. Always-help fires at the first verified error at a decision point. The three-wrong rule fires at the first `i` where opportunities `i-2, i-1, i` are all verified errors (verified-error definition below).

**A1.3 Verified error.** *Inferred, not documented:* `first_action` code 0 = first action was an answer attempt, 1 = hint request, 2 = unknown "other". These meanings are inferred from cross-tabulation with `correct` and `hint_count` (DATA.md 3.1); no source documents them.
- **Primary:** `correct = 0` **and** `first_action = 0` (a wrong first answer; hint-first rows, code 1, are excluded, and so are code-2 rows, whose meaning is unknown).
- **Sensitivity ("including hint-first"):** `correct = 0` (all first-attempt-wrong rows, i.e. codes 0, 1 and 2).
- Risk models use `correct` only, never `first_action`.

**A1.4 Unnecessary interruption is a proxy.** The logs contain no tutor interventions. "Unnecessary" is defined as: an intervention in a sequence with `y = 0`, i.e. a sequence that **later masters within 10 opportunities without any logged help**. This is a proxy: the student may have mastered because of help the log does not show, an interruption might have helped or harmed, and a `y = 1` sequence might not have benefited. It is not a measurement of harm.

**A1.5 Quantities (all in opportunities, per SPEC 7).** `T = 10`. For a `y = 1` sequence, `i_err` = its first verified error at a decision point; the sequence has **no help opportunity** (excluded from delay, count reported) if none exists. Help delay `d = i_first - i_err` if fired, else censored `d = T - i_err` (counted as missed help). Lead time = `T - i_first` for fired `y = 1` sequences. Missed help = `y = 1` sequence with a help opportunity and no firing. Reported: interventions per student; unnecessary interventions per student and share of interventions; missed-help count and share of `y = 1` sequences with a help opportunity; mean lead time; mean delay including and excluding censored; N of sequences without a help opportunity.

**A1.6 Profiles (explicit cost weights, interruption : delayed-help).**

| Profile | `w_I` (per unnecessary interruption) | `w_D` (per opportunity of help delay) | Ratio |
|---|---|---|---|
| quiet | 3 | 1 | 3:1 |
| balanced | 1 | 1 | 1:1 |
| supportive | 1 | 3 | 1:3 |

Cost `J(tau) = w_I * U + w_D * D`, where `U` = number of unnecessary interruptions and `D` = total help delay in opportunities over `y = 1` sequences (missed help contributes its censored delay). These are cost settings, not clinical or demographic categories.

**A1.7 Choosing tau.** On the **validation split only**, after recalibration (A1.9), over the grid `tau in {0.00, 0.01, ..., 1.00}` plus a never-fire value (`tau = 1.01`); the minimiser of `J` per profile; ties go to the larger `tau`. Methods without `tau` (always-help, three-wrong) have none.

**A1.8 Risk head.** Logistic regression (L2) on windowed features of the first `i` responses only: `i`, number correct, fraction correct, last three responses (`c_i`, `c_{i-1}`, `c_{i-2}`), number wrong in the last 3 and last 5, current correct run, current wrong run, longest wrong run. Features are standardised with training statistics. One model across decision points `i = 3..9`, trained on training-split labelled variant-A rows `(sequence, i)` with `m > i`, `n >= i`. `C` chosen from {0.01, 0.1, 1, 10} by validation log-loss on the same row type. **Beat rule (fixed now):** the head replaces PFA/BKT as the risk score only if its mean validation AUC over `k in {3, 5, 8}` (variant A, the Stage 3 scored sets) exceeds that of **both** BKT and PFA, and the margin over the better of them is **>= 0.005**. Otherwise the BKT or PFA with the higher mean AUC is the primary risk score. Either way, all three risk scores are reported. PFA uses `C = 0.1` and BKT `min_seq = 10`, as chosen in Stage 3 (not reselected). A paired student-bootstrap interval for the AUC difference is reported but does not change the rule.

**A1.9 Recalibration.** Each risk score is recalibrated by Platt scaling (logistic regression of `y` on the logit of the raw score, two parameters) fitted on **validation** decision rows (all `(sequence, i)` rows in scope, `i = 3..9`, `m > i`), then used for tau selection. Calibrator and `tau` therefore share the validation split, so validation costs are optimistic; that is stated wherever they are reported. Only the Platt line is fitted; no other validation-based tuning.

**A1.10 Random baseline with matched intervention rate.** For each (risk method, profile) operating point, a random policy: i.i.d. uniform scores per `(sequence, i)`, fires at verified errors with score >= `t`, where `t` is found by bisection so the number of intervened sequences equals the policy's number (nearest achievable); 200 draws, seed `20261017`; mean and standard deviation of every metric are reported.

**A1.11 Leaver sensitivity reporting.** Primary tau is chosen with leavers dropped. The sensitivity table reports (a) the same tau applied with leavers added, and (b) tau re-chosen with leavers added. The hint-first sensitivity (A1.3) is reported the same way for the primary-scope sample, with tau re-chosen under that error definition and also the primary tau carried over.

**A1.12 Reporting rule.** Calibration metrics (ECE, calibration intercept, slope) are shown as `n/a` for 0/1 methods (three-wrong rule, always-help); AUC for them is shown as the AUC of the single binary score.

**A1.13 Output.** `results/policy_val.json`, `results/riskhead_val.json`. No `RESULTS.md` until Stage 5. No test-split file is read.

### Amendment 2 — Stage 5 test protocol (2026-10-15, before freezing values or reading any test label)

Reason: the owner set conditions for the one-time test evaluation. Sections 1–11 and Amendment 1 are unchanged except where stated.

**A2.1 Fair three-wrong baseline.** The three-wrong rule is generalised to *n consecutive verified errors ending at decision point `i`* (needs `i >= n`), with `n` in {1, 2, 3, 4, 5, 6}; `n = 1` equals always-help. `n` is chosen **per profile** to minimise validation cost `J` (ties go to the larger `n`), separately for each configuration in A2.2. The untuned `n = 3` rule from Stage 4 is also reported.

**A2.2 Configurations.** *main* (leavers dropped, primary verified error); *leavers* (leavers added as `y = 0`, primary error); *hintfirst* (leavers dropped, error includes hint-first). For *leavers* and *hintfirst* two variants are reported: **carried** (the *main* taus and `n`) and **rechosen** (re-chosen on validation under that configuration). Everything in this list is decided on validation and then frozen.

**A2.3 Frozen artifacts.** Before any test label is read, `results/frozen_val.json` is written and **committed**. It holds: model hyperparameters (PFA `C`, BKT `min_seq`, risk-head `C`), the Platt parameters `(intercept, slope)` per risk method, and for every configuration and variant: `tau` per (risk method, profile) and tuned `n` per profile. Its sha256 is recorded; the test script refuses to run if the hash differs from the recorded one. PFA, BKT and the risk head are refitted on the **training split only** (deterministic), never on validation or test.

**A2.4 Test protocol.** `python -m aria_pace.run_test` evaluates *main* and both sensitivities once, with the frozen values, and writes `results/test_results.json` and `results/TEST_LOCK`; it refuses to run if the lock exists. No `tau`, `n`, or calibrator is chosen on test. In the same single pass it also computes the full `tau` sweep on test **for diagnosis and plotting only** (including the test-optimal `tau`); those test-optimal values feed nothing and are labelled as such. Test results are reported as they are, including where they contradict validation. A `--verify` mode, intended for third parties, recomputes and compares without writing.

**A2.5 Bootstrap.** Students are resampled with replacement (multinomial counts over students; 1,000 draws; seed `20261018`), identically for all methods (paired). Reported: 95% percentile intervals of the cost difference `J(BKT) - J(baseline)` per student, negative meaning BKT is cheaper, for baselines always-help, tuned three-wrong, untuned three-wrong, PFA, risk head, and the matched-rate random policy. The random policy enters as its expected per-sequence contribution (200 draws, seed `20261017`, threshold matched to BKT's intervened-sequence count on the full sample and held fixed across resamples). Intervals for BKT's own metrics (interventions, unnecessary, missed share, delay, lead time) use the same draws. Validation intervals are computed the same way and labelled optimistic.

**A2.6 Tau stability.** On validation, 1,000 student-bootstrap resamples (seed `20261019`) re-choose `tau` for BKT in each configuration and profile; reported: median and 10th–90th percentile of the chosen `tau`, the fraction of resamples within 0.10 of the frozen `tau`, and the range of `tau` whose validation `J` is within 2% of the minimum. This is reported as a finding for the quiet profile.

**A2.7 Figures.** (1) Frontier: BKT, PFA and risk-head `tau` sweeps, expected matched-random, tuned and untuned three-wrong over `n`, and always-help, with the three profiles' operating points marked on the BKT curve; x = interventions per student with y = mean delay (including censored), and x = unnecessary interventions per student with y = missed-help share. (2) `J` versus `tau` for each profile. Each is drawn for validation (marked optimistic) and test.

**A2.8 Tables.** Calibration columns are `n/a` for 0/1 methods. Every validation figure is labelled optimistic because calibrator and `tau` were fitted on the same split.

**A2.9 Headline.** `RESULTS.md` carries the owner's headline: "On ASSISTments 2009 logs, a BKT-threshold abstention rule roughly halves interventions when interruptions are costly, with ~5% missed help; when delayed help is costly, abstention saves little." It is accompanied by the test numbers; if the test numbers do not support any part of it, `RESULTS.md` says which part and by how much.

