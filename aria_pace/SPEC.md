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


### Amendment 3 — Stage 6: cross-dataset test, DKT, affect (2026-10-16, before any Stage 6 code)

Reason: the owner set the Stage 6 design. Nothing in Sections 1-11 or Amendments 1-2 changes; the 2009 test split stays locked (`results/TEST_LOCK`) and is **never touched again**. Every result in Stage 6 is labelled **(a)** or **(b)** (A3.2). Nothing here may be published until ASSISTments confirms the data licence (A3.11).

**A3.1 Data roles.**
- **2009-10** (`skill_builder_data_corrected_collapsed.csv`): train and validation only, with the existing student split (`results/split.json`). Its test portion is not used.
- **2012-13** (`2012-2013-data-with-predictions-4-final.csv`, see `docs/aria_pace/DATA_v2.md`): the held-out data. Pool = students with at least one `type == "MasterySection"` row, **minus the user IDs that also occur in the 2009 file** (an ID collision may be the same student; removing them means no 2012 test student can have been in 2009 training). The pool size and the overlap count were recorded in the split file, which is held back pending dataset terms.
- **2012 split, student level, fixed seed `20261016`:** `np.random.RandomState(seed).permutation` of the sorted pool IDs; **first 20% = locked test**, of the remaining 80% the first 75% = train and the last 25% = validation (so 60 / 20 / 20 of the pool). The split is computed **once**, written to `results/split_2012.json` (ids and sha256) and committed **before any model touches 2012 data**. The 2012 test students are dropped from memory right after the split, in every script, until the single evaluation (A3.7).
- **Own lock file `results/TEST_LOCK_2012`**, written at the single 2012 test evaluation; the test script refuses to run if it exists (same mechanism as A2.4).

**A3.2 Two separate claims; every number is labelled.**
- **(a) Cross-dataset:** models trained and selected **only on 2009** (BKT, PFA, the windowed logistic risk head, DKT, and the frozen policy: Platt calibrators, `tau`, tuned `n`) are evaluated on the **2012 test portion**. No 2012 data are used for fitting, selection, calibration or `tau`. The unit is the same: sequence = student x assignment, label variant A (primary) and B (sensitivity).
- **(b) Within-dataset affect question:** a with-affect and a without-affect risk model (A3.4) are trained on **2012-train**, selected on **2012-val**, and evaluated **once** on the same locked 2012 test portion.
- Results of (a) and (b) are never pooled or compared as if they were one experiment.

**A3.3 Preparing the 2012 data (same rules as 2009 unless stated).**
- First-attempt correct := `correct >= 1` (the file has fractional values; fractions count as not correct). Order within a sequence: `start_time`, then `problem_log_id`. Mastery, label, eligibility, scope and decision points: SPEC sections 2, 3 and 7 and Amendment 1, unchanged.
- **KC key** of a sequence = `skill_id` of its first opportunity, formatted as an integer string (`"277.0"` -> `"277"`); missing -> `"UNK"`. A 2012 KC that is not in a 2009 model's vocabulary uses that model's pooled / unknown parameters; the **share of test sequences that do** is reported with (a). (2009 composite IDs such as `10_104` never match a 2012 ID, so they stay unused.)
- **Verified error** (policy): primary = `correct < 1` **and** `first_action == 0`; sensitivity = `correct < 1`. The `first_action` meanings are inferred, not documented, for 2012 as for 2009 (UNVERIFIED for 2012).

**A3.4 Affect features (b).**
- Base risk model = the A1.8 windowed logistic regression (same features, one model over decision points i = 3..9, L2, `C` in {0.01, 0.1, 1, 10} chosen by validation log-loss on decision rows), fitted separately for **without affect** and **with affect** on 2012-train.
- Affect inputs are the four `Average_confidence(...)` columns, which are **outputs of detectors of unverified version**, not probabilities and not human observations (`DATA_v2.md` section 4).
- **Transform (fixed now): per-column rank.** For each column, the mid-rank empirical CDF computed on the **2012-train** skill-builder rows (ties share the mid-rank), applied unchanged to validation and test rows, giving values in [0, 1]. Chosen over a z-score because the columns are saturated and non-Gaussian (medians at the minimum or maximum).
- **At prediction step k** (decision point i = k): for each transformed column, three features, **last value, mean, maximum**, over rows `<= k` of the sequence (12 features) added to the A1.8 features. **Lagged sensitivity:** the same 12 features over rows `<= k-1`. Both are reported; the primary is rows `<= k`. (Detector outputs on row k may use that row's own interaction features, which the response features already contain; the lagged version reduces that overlap.)
- **Primary endpoint (b):** mean over k in {3, 5, 8} of AUC(with) minus AUC(without) on the 2012 test portion, variant A, with a paired bootstrap over students (1,000 draws, seed `20261026`). **"Affect adds information" is claimed only if the 95% interval's lower bound is > 0 and the point estimate is >= 0.005**; otherwise the report says no evidence that detector-predicted affect adds information beyond response history. If the primary and lagged versions disagree on this criterion, the result is reported as **not robust**. Per-k differences, Brier and ECE are reported as descriptive.
- **Secondary (b), descriptive only, no claim:** policy cost `J` per profile (A1.6) for both models, with `tau` re-chosen on 2012-val for each model separately after Platt recalibration on 2012-val, and the paired-bootstrap interval of the `J` difference on test.

**A3.5 DKT (and SAKT only if close).**
- **Model:** a standard LSTM deep knowledge tracer. Vocabulary: KC strings with >= 10 training sequences get ids 1..K, all others share id 0 (UNK). The step input is the token `kc * 2 + correct`; an embedding (dim 32) feeds one LSTM layer (hidden size chosen from {32, 64, 128}), dropout 0.2, a linear layer over K+1 outputs, and the prediction for the next step reads the output of that step's own KC (every step of a sequence uses the sequence's KC, exactly as BKT and PFA do). Loss: binary cross-entropy on next-response correctness, all steps. Training uses the **first 100 opportunities** of each training sequence. Adam, learning rate 1e-3, batch 64, at most 40 epochs, early stopping with patience 4 on validation next-response log-loss. Hidden size and the stopping epoch are chosen by **validation next-response log-loss** (never the wheel-spinning label). CPU, 4 threads, `torch.manual_seed(20261023)`, deterministic.
- **Risk score:** `P(no mastery in opportunities k+1..10 | first k responses)`, computed **exactly** by enumerating the future outcome tree (at most 2^(10-k) paths, pruned when a 3-run completes) with the LSTM state carried along each path: the same estimand as BKT and PFA. The score is defined for any k from 3 to 9.
- **Promotion rule (validation only, variant A, the 2009 validation sets used for BKT):** DKT replaces BKT as the primary risk score only if, **at both k = 3 and k = 5**, (i) AUC(DKT) - AUC(BKT) >= **0.01** and (ii) the **95% paired student-bootstrap interval of that difference lies entirely above 0** (1,000 draws, seed `20261024`; "non-overlapping" is read as "the paired interval excludes 0"). **k = 8 is reported but does not decide.** Otherwise **BKT stays** and the loss is reported with the same numbers.
- **SAKT** (a self-attentive tracer) is trained only if DKT is **not** promoted but is close: mean over k in {3, 5} of AUC(DKT) - AUC(BKT) >= -0.01. It is then selected and judged by the same rules against BKT. At most one model is promoted; if both qualify, the higher mean validation AUC at k in {3, 5} wins.
- If a model is promoted, it also gets its own Platt calibrator and its own `tau` per profile chosen on 2009 validation (A1.7, A1.9), frozen before 2012 test. BKT results are still reported.
- Calibration, Brier, log-loss and AUC with student-bootstrap intervals are reported for DKT and BKT on 2009 validation, as in Stage 3. Citations for DKT and SAKT are verified in `LITERATURE.md` before they appear in any results document.

**A3.6 Freeze before 2012.** After C3 and before any 2012 model fitting: write `results/frozen_v2.json` (+ `.sha256`) holding the promotion decision, hyperparameters, Platt parameters, `tau` and `n` per profile and configuration for every 2009-trained model, and save DKT weights; commit it. The 2012 evaluation script verifies the hash and refuses to run on a mismatch.

**A3.7 The single 2012 evaluation.** One run evaluates, on the locked test portion: **(a)** (i) AUC with student-bootstrap 95% intervals, ECE, calibration intercept and slope, Brier, at k = 3, 5, 8, for BKT, PFA, the risk head, and DKT (promoted or not), for variants A and B, with the unknown-KC share; (ii) the policy with frozen 2009 values in the configurations main / leavers / hintfirst (carried and rechosen-on-2009-validation, as in Stage 5): interventions per student, unnecessary, missed help, lead time, delay and `J` per profile, for the primary risk score, PFA, the head, always-help, the tuned n-rule, the untuned three-wrong rule and the matched-rate random policy, with paired-bootstrap intervals of `J` differences (seed `20261025`); **(b)** the with/without-affect comparison of A3.4, primary and lagged. **Pre-registered primary endpoints:** (a-P1) AUC of the primary risk score at k = 3, 5, 8; (a-P2) in the quiet profile, interventions per student relative to always-help and the missed-help share; (a-P3) the `J` difference to always-help in the three profiles; (b-P1) the A3.4 primary endpoint. Everything else is **exploratory** and labelled so. The 2009 test results (`test_results.json`) are shown beside the 2012 numbers as context, not recomputed. The run writes `results/TEST_LOCK_2012`.

**A3.8 Profiles and policy are unchanged** (A1.1-A1.13, A2.x): profile weights 3:1, 1:1, 1:3, at most one intervention per sequence, decision points 3..9, tuned `n` grid, same `J` and metrics. No new policy variant is introduced in Stage 6.

**A3.9 Order of work and what may read what.** C3 reads 2009 only. C4 reads 2012 train and validation only. C5 is the only step that reads 2012 test rows. No training runs while a demo rehearsal is running on this machine (checked before each run). Work ends before Oct 22.

**A3.10 Statements required in `RESULTS_v2.md` and the slide.** Replay on logs (no counterfactual); "unnecessary" is a proxy; affect is **detector-predicted, of unverified version, not human observation** (no human labels were found); grade level of the 2012 file is unknown; the 2012 data come from the same platform as 2009 (transfer, not independence); no learning-gain claim; no clinical claim about any student or group; the `first_action` meanings are inferred; every number traces to a file in `aria_pace/results/`.

**A3.11 Licence hold.** Confirmation of the data licence from ASSISTments is **pending** (owner, 2026-10-16). `RESULTS_v2.md`, the slide, and any figure from the 2012 file are marked **"INTERNAL: not for publication until ASSISTments confirms the licence"** and are not to be published or shared outside the team until it arrives.

**A3.12 Seeds.** 2012 split `20261016`; DKT training `20261023`; DKT-vs-BKT bootstrap `20261024`; (a) bootstrap `20261025`; (b) bootstrap `20261026`; random-policy draws `20261017` (200 draws); all else as in earlier amendments.

### Amendment 3a — SAKT architecture (2026-10-16, after the DKT result, before any SAKT code or number)

**Why this exists.** A3.5 says SAKT is trained if DKT is not promoted but is close. That condition is met (commit `a05a695`: DKT not promoted; mean validation AUC difference to BKT over k = 3 and 5 is -0.0027, which is >= -0.01). A3.5 did not fix SAKT's architecture; it is fixed **here**, after DKT's result is known but before any SAKT number exists, so it is a design choice made with that knowledge, and it is recorded as such. **The selection criterion, the promotion rule, the data, the seeds and the scoring estimand do not change.**

- **Tokens and vocabulary:** as DKT (A3.5): step token `kc * 2 + correct`, KC vocabulary from training sequences with >= 10 sequences, others share id 0, first 100 opportunities, the first step is never predicted.
- **Model (one self-attention block):** interaction embedding of each earlier step plus a learned position embedding (positions 0..99) are the **keys and values**; the **query** for predicting step s is the embedding of that step's KC plus the position embedding of s. **Causal multi-head attention** (2 heads, a step attends to all earlier steps, no window), residual connection from the query, LayerNorm, a feed-forward layer (d -> d, ReLU, dropout 0.2, d -> d) with residual and LayerNorm, then a linear layer to one logit. Model width `d` is chosen from {32, 64, 128}; attention dropout 0.2.
- **Training and selection:** same loss (binary cross-entropy on next-response correctness, all steps), optimiser (Adam, lr 1e-3), batch 64, at most 40 epochs, patience 4, seed `20261023`, CPU 4 threads, deterministic. `d` and the stopping epoch are chosen by **validation next-response log-loss** (never the wheel-spinning label).
- **Risk score:** the same exact outcome-tree enumeration as DKT; because attention has no recurrent state, each path re-reads its full token prefix.
- **Promotion rule:** identical to DKT's against BKT: AUC difference >= 0.01 and the paired 95% interval lower bound > 0, at **both** k = 3 and k = 5 (variant A, seed `20261024`); k = 8 reported, not deciding. At most one model is promoted overall; BKT stays unless one qualifies.
- **Multiplicity, stated:** two candidates (DKT, SAKT) are compared with BKT. The +0.01 margin and the interval requirement are the only guards; no further correction is applied, and both comparisons are reported whatever the outcome.

### Amendment 4 — Mastery-definition sensitivity (2026-10-20, before any A4 code runs or any A4 number exists)

Reason: the owner asked how much the policy depends on the choice of mastery rule (SPEC section 2). Nothing in sections 1-11 or Amendments 1-3a changes. `frozen_val.json`, `frozen_v2.json` and their `.sha256` files are not edited, and no file under `results/` that produced a published number is changed; A4 writes only new files. **No locked split is read:** the 2009 test students are dropped right after the split exactly as in `freeze_val.py:22-24`, and the 2012 file is not loaded. Every A4 output is **exploratory sensitivity on validation, not pre-registered**, except the control check in A4.3, which is pre-registered.

**A4.1 Definitions.** A student-by-assignment sequence has mastery at opportunity `m` (1-based) under each definition; the label, scope and decision points then follow A1 with the determination point `T` of that definition.
- **D3 (control):** three correct in a row, T = 10 (the frozen definition, `labels.py:7-14`).
- **D4:** four correct in a row, T = 10.
- **D5T15, D5T20:** five correct in a row, T = 15 and T = 20.
- **DEMA:** exponential moving average of correctness, alpha = 0.3, initialised at the first response (`ema_1 = x_1`, `ema_i = 0.3 x_i + 0.7 ema_{i-1}`); mastery at the first opportunity `i >= 3` such that `ema_{i-2}, ema_{i-1}, ema_i` are all >= 0.8; T = 10. (alpha, the initialisation and the 0.8 bar are fixed here, before any result.)
- Label (variant A only): `y = 0` if `m <= T`; `y = 1` if `m > T` and `n >= T`; otherwise an unlabelled leaver (dropped, as in A1.1). Policy scope: `n >= 3` and `m > 3`. **Decision points stay i = 3..9 for every definition** (A1.2), a decision point exists iff `m > i`, verified error = primary (A1.3), at most one intervention per sequence.

**A4.2 Pipeline (BKT only).** Per definition, on the 2009 train / validation split (`results/split.json`), the same steps as `freeze_val.py:19-68` for the primary risk score only (`frozen_v2.json` `primary_risk_score = "bkt"`): per-KC BKT fitted on train once (its parameters do not depend on the mastery definition; `bkt_min_seq = 10` from `baselines_val.json`); raw risk = `P(no mastery in opportunities k+1..T | first k responses)` computed **exactly** under the definition (forward filter, then an exact dynamic programme over (latent state, run) for run rules, and over (latent state, EMA, run of EMA >= 0.8) for DEMA; the future KC is the sequence KC, as in `baselines.py:BKT.score`); Platt calibration fitted on validation primary (non-leaver) valid cells; `tau` per profile chosen on validation by minimising `J = w_I * unnecessary + w_D * delay` over `TAU_GRID`, ties to the larger value (`stage5.py:153-171`), weights 3:1, 1:1, 1:3, delay censored at `T - i_err` with the definition's own `T`. PFA and the risk head are **not** re-run (the head is trained on the label and is not the primary score; PFA is not primary). All code is new (`aria_pace/mastery_sens.py`); no existing module is edited.

**A4.3 Control check (pre-registered).** Under D3 the pipeline must reproduce `frozen_val.json`: `platt.bkt.intercept` and `.slope`, and `configs.main.frozen.taus.bkt` for the three profiles, equal to within 1e-9 (taus exactly). If it does not, the work stops and the pipeline is fixed before any other definition is run or looked at.

**A4.4 Outputs per definition (validation only; all optimistic because Platt and `tau` are chosen on the same split).** Number of in-scope sequences and students; label rate (share of `y = 1` among labelled in-scope sequences); BKT AUC at k = 3, 5, 8 on the eligible labelled sequences (`n >= k`, `m > k`) with a 95% student-bootstrap interval (1,000 draws, seed `20261016`); the chosen `tau` per profile, with the student-bootstrap distribution of the chosen `tau` (1,000 draws, seed `20261019`; median and 10th-90th percentile); and, at the chosen `tau`, the intervention reduction relative to always-help and the missed-help share per profile with 95% student-bootstrap intervals (1,000 draws, seed `20261018`, one resample shared by the policies). One table and one figure of how `tau` moves.

**A4.5 Reading rule (fixed now).** A `tau` is called **materially shifted** from the control if it differs by more than 0.10 (the window used in `stage5.py:193`). Nothing else is declared a pass or a fail; the table is descriptive. **No result of A4 changes the frozen thresholds, the demo, or any published number.** Every A4 result file carries `"status": "exploratory sensitivity (validation only; not pre-registered)"`.

**A4.6 Seeds.** As stated in A4.4; EM and all fitting are deterministic as in A1.

### Amendment 5 — Multi-skill DKT and SAKT (2026-10-20, before any A5 code runs or any A5 number exists)

Reason: A3.5 and A3a fed each model **one assignment's sequence at a time** (every step carries the sequence's own KC). The owner asked whether a model that sees the student's **earlier attempts on other skills** predicts wheel-spinning better. Nothing in sections 1-11 or Amendments 1-4 changes; frozen files and published results files are untouched; A5 writes only new files. **Only 2009 train and validation are read** (the 2009 test students are dropped right after the split, `freeze_val.py:22-24`; the 2012 file is not loaded). Labels: variant A, three-correct-in-a-row, T = 10 (the frozen definition). All A5 outputs are validation-only and say "pre-registered promotion rule" or "exploratory" as marked below.

**A5.1 Histories.** For each student, all cleaned rows in `order_id` order form one stream; a row is the token `kc * 2 + correct`, where `kc` is that row's own `skill_id` mapped through the vocabulary (A5.2). A scored decision is a sequence (student x assignment) at decision point k in {3, 5, 8}. The model sees **every row of that student with `order_id` less than or equal to that of the sequence's k-th opportunity**, whichever assignment or skill it belongs to (rows of other assignments that fall between the sequence's opportunities included; nothing later is used), truncated to the **most recent 90 rows**. Future steps of the sequence in the outcome tree use the **sequence's own KC** (its first opportunity's `skill_id`), exactly as in A3.5. The target and the risk definition are unchanged: `P(no mastery in opportunities k+1..10 | the sequence's first k responses and the student's earlier history)`, computed exactly by enumerating the future outcome tree (at most 2^(10-k) paths, a path is dropped when three correct in a row complete).

**A5.2 Vocabulary.** Skill strings that occur in at least 10 distinct training (student, assignment) groups (any row of the group) get ids 1..K; everything else shares id 0. (This differs from A3.5, which counted only each sequence's first-opportunity KC; the multi-skill histories contain other skills' rows, which must be representable.)

**A5.3 Models.** The same two architectures and training settings as A3.5 / A3a, applied to the multi-skill stream: **MS-DKT** (embedding dim 32 -> one LSTM layer of hidden size h -> dropout 0.2 -> linear over all KCs; the prediction for a step reads the output of that step's own KC) and **MS-SAKT** (interaction embedding + learned position as keys/values, query = KC embedding + position, causal 2-head attention, residual, LayerNorm, feed-forward d -> d, ReLU, dropout 0.2, d -> d, residual, LayerNorm, linear to a logit; width d). Width/hidden in {32, 64, 128}. Training windows: each training student's stream is cut into windows of **100 rows with stride 50** (state is reset at each window); loss = binary cross-entropy on next-response correctness for every step that has at least one earlier step in its window. Adam, lr 1e-3, batch 64, at most 40 epochs, early stopping with patience 4, `torch.manual_seed(20261023)`, CPU, 4 threads, deterministic. **Selection** of the width and the stopping epoch uses the **validation next-response log-loss over all non-first rows of the validation streams** (validation streams cut into non-overlapping windows of 100, a row's context = the earlier rows of its own window), **never the wheel-spinning label**. The inference history cap (90 rows, A5.1) plus at most 7 future path steps stays below the 100 positions the models were trained on.

**A5.4 Promotion rule (pre-registered; unchanged from A3.5).** On the 2009 **validation** sets, variant A, a multi-skill model replaces BKT as the primary risk score only if, **at both k = 3 and k = 5**, (i) AUC(model) - AUC(BKT) >= **0.01** and (ii) the **95% paired student-bootstrap interval of that difference lies entirely above 0** (1,000 draws, seed `20261024`). k = 8 is reported and does not decide. At most one model is promoted; if both qualify the higher mean validation AUC at k in {3, 5} wins. **Multiplicity, stated:** two further candidates are compared with BKT (after DKT and SAKT in A3.5 / A3a); the margin and the interval requirement are the only guards.

**A5.5 If a multi-skill model is promoted: STOP.** Nothing further is computed (no calibrator, no `tau`, no test number); the owner is told, and **a fresh holdout must be declared in a new amendment before any test number is produced** (the 2009 test split is locked and spent, `TEST_LOCK`; the 2012 split is locked and spent, `TEST_LOCK_2012`). If neither is promoted, BKT stays and the numbers are reported as a negative result.

**A5.6 Reported.** Per model and for BKT, validation, variant A, k = 3, 5, 8: eligible sequences and students, AUC with a 95% student-bootstrap interval (1,000 draws, seed `20261016`), and the paired AUC difference to BKT with its interval (seed `20261024`); next-response log-loss on the matched set (every non-first opportunity of every validation sequence), with a paired student-bootstrap interval of the log-loss difference to BKT (seed `20261024`), where BKT's prediction for a step is `prior * (1 - slip) + (1 - prior) * guess` with the filtered prior, as in `baselines.py:BKT.next_response_logloss`. The single-skill DKT and SAKT numbers from `results/dkt_val.json` and `results/sakt_val.json` are quoted for reference, not recomputed. Status field of every A5 file: "validation only; promotion rule pre-registered (A5.4), everything else exploratory".

**A5.7 Known approximations, stated now.** The tree assumes the rest of the sequence follows immediately, with no interleaved rows from other assignments; models see token order only, never timestamps; windows reset state, so a row late in a long stream has at most 99 earlier rows of context in training and evaluation and at most 90 at scoring.

### Amendment 6 — Parametric closed-loop sensitivity analysis (2026-10-20, before any A6 code runs or any A6 number exists)

Reason: replay (SPEC section 7) cannot say what would happen if students were actually helped or interrupted; its limitation is part of the definition. The owner asked for a small **parametric** simulation to see **how the policy's advantage over always-help depends on assumptions** about the effect of help, the cost of an interruption, and the cost of being left stuck. **This is an assumption-dependent sensitivity analysis. It is not evidence of any effect on students.** It uses **no LLM agents, no real data and no locked split**; the only inputs from the project are the frozen pooled BKT parameters (`baselines_val.json` `selection.bkt_pooled_params`), the frozen Platt calibrator and the frozen `tau` per profile (`frozen_val.json` `platt.bkt`, `configs.main.frozen.taus.bkt`), all read, never edited. **No `tau` is re-tuned.** Frozen files and published results files are untouched; A6 writes only new files. Every A6 output carries `"status": "assumption-dependent sensitivity analysis, not evidence of effect"`.

**A6.1 Student model.** One sequence per simulated student, opportunities t = 1..T with T = 10. Latent state K in {unknown, known}; `K_1 ~ Bernoulli(prior)`. At opportunity t the response is correct with probability `1 - slip_t` if known and `guess` if unknown (`slip_t = slip` unless the student was interrupted at t-1, A6.2); after the response an unknown student becomes known with probability `learn_t` (`learn` unless help has been given, A6.2). `prior, learn, guess, slip` = the frozen pooled BKT parameters. Mastery = three correct in a row (the frozen definition); a mastered sequence stops. All draws use **common random numbers**: one fixed array of uniforms per (student, step, role) is shared by every policy and every grid point, so policies differ only through their decisions.

**A6.2 Policies and the three parameters.** Decision points t = 3..9 (A1.2). At a decision point where the response was wrong (a verified error; every wrong response is one, first-action codes do not exist here) a policy may intervene, at most once per sequence. Policies: **always-help** (intervene at the first such point), and the three **frozen profiles** (intervene if the Platt-calibrated BKT risk of the observed prefix is >= the profile's frozen `tau`, 0.57 / 0.19 / 0.09; the risk is the exact quantity of `baselines.py:BKT.score` for the pooled parameters, so the policy's model equals the generator: a deliberately favourable assumption, stated), plus a **no-help world** used only to define "unnecessary". The parameters, each in [0, 1]:
- `p_help_effect`: after an intervention at t, every later learning transition of that student uses `learn' = 1 - (1 - learn)(1 - p_help_effect)`.
- `p_interrupt_cost`: the response at t+1 after an intervention uses `slip' = slip + p_interrupt_cost (1 - slip)` (a one-response distraction; it applies to every intervened student, whatever their state).
- `p_stuck_cost`: at a decision point t in 3..9 where the response was wrong and the student is **not helped at t and has not been helped before**, the student abandons with probability `p_stuck_cost` (the sequence ends and counts as not mastered).

**A6.3 Grid.** `p_help_effect` in {0, 0.1, 0.2, 0.3, 0.5}; `p_interrupt_cost` in {0, 0.1, 0.2, 0.3}; `p_stuck_cost` in {0, 0.05, 0.1, 0.2, 0.3}: 100 points. 100,000 simulated students per point, seed `20261101` (arrays drawn once, reused at every point).

**A6.4 Outcomes (per in-scope sequence; scope = not mastered by opportunity 3 and not ended before 3, identical for every policy).** Realised outcome `y = 1` if not mastered within T (abandonment counts). `i_err` = the first wrong response at t in 3..9 (a prefix event, identical across policies). **Unnecessary** = intervened and mastered within T in the no-help world. **Delay** (for `y = 1` sequences with an `i_err`) = `i_first - i_err` if intervened, else `T - i_err` (A1 definitions). **Missed help** = `y = 1`, has an `i_err`, never intervened. Cost `J = w_I * unnecessary + w_D * delay` per sequence with the profile weights (3:1, 1:1, 1:3), evaluated on the **realised** trajectories. Also reported: mastery rate within T and interventions per student. Headline per profile and grid point: `Delta J = J(profile policy) - J(always-help)` under that profile's weights (negative = the policy is cheaper), with a normal-approximation 95% interval of the paired per-sequence difference (this interval reflects **simulation noise only**, not uncertainty about the parameters).

**A6.5 Regions (fixed now).** At a grid point and profile the policy **beats** always-help if the interval of `Delta J` lies below 0, **flips** (is worse) if it lies above 0, and is **indistinguishable** otherwise. The report shows these regions as maps over (`p_help_effect`, `p_stuck_cost`) for each `p_interrupt_cost`, and states in words where the sign changes. Nothing in A6 is used to choose a `tau`, a profile, or a claim about students.

**A6.6 Checks (pre-registered).** (i) At `p_help_effect = p_interrupt_cost = p_stuck_cost = 0` the simulated trajectories contain no effect, so the simulator's `J` per sequence must equal the replay machinery's (`mastery_sens.SensTable` / `seq_vec_T` on the same simulated trajectories with the same frozen `tau` and calibrator) exactly; (ii) the first-response correctness rate must match `prior * (1 - slip) + (1 - prior) * guess` within Monte Carlo error; (iii) the risk table must equal `BKT.score`.

### Amendment 7 — Cross-validated trade-off curves per mastery definition (2026-10-22, before any A7 code runs or any A7 number exists)

Reason: A4 chose `tau` and the Platt calibrator on the same validation students it then evaluated, so its savings are optimistic (A4.4), and it reports one operating point per definition. The owner asked how the saving-versus-missed-help trade-off moves with the definition, without that optimism. **Exploratory, validation only.** Nothing in sections 1-11 or Amendments 1-6 changes; frozen files and published results files are not edited; A7 writes only new files. **No locked split is read:** the 2009 test students are dropped right after the split exactly as in `run_mastery_sens.py` (the 2009 test split and the 2012 data are never loaded), and **no new test evaluation is made.**

**A7.1 Pool and folds.** Pool = 2009 train + validation students (the same sequences A4 uses: every student not in the test split). Five student-level folds: a seeded permutation of the sorted pool student ids (`numpy.random.RandomState(20261022)`), folds = consecutive fifths. Fold `f` is the held-out fold `H_f`; sequences of one student are never split.

**A7.2 Per fold and definition.** The other four folds are split by student into FIT (3/4) and CAL (1/4) with `RandomState(20261022 + f)`. BKT is fitted on FIT with `bkt_min_seq` from `baselines_val.json` (as A4.2; it does not depend on the definition). For each of the five A4 definitions (D3, D4, D5T15, D5T20, DEMA; A4.1): relabel CAL and `H_f` under the definition; raw risk by `mastery_sens.raw_matrix`; the Platt calibrator is fitted on CAL (primary, variant A, as `run_definition`); `tau` per profile is chosen on CAL by the same rule (argmin J over `TAU_GRID`, ties to the larger `tau`, `choose_taus`). The chosen `tau` and calibrator are then applied to `H_f` **only**; nothing about `H_f` is used for any choice.

**A7.3 Curves and operating points.** For each definition, the held-out predictions of the five folds are pooled (each fold with its own FIT/CAL calibration). For every `tau` in `TAU_GRID`: intervention saving = `1 - interventions(tau) / interventions(always-help)` and missed-help share = missed / sequences with a help opportunity (the definitions of `mastery_sens.policy_ci`, with that definition's `T`). One figure per profile (quiet, balanced, supportive); x = missed-help share, y = intervention saving; one curve per definition; bands = 95% student-bootstrap intervals (1,000 draws over pooled held-out students, seed `20261023`) of the saving and the missed-help share at each `tau`; markers = the pooled held-out result at the per-fold chosen `tau` of that profile, with its bootstrap interval.

**A7.4 How much `tau` moves.** Per definition and profile: the five fold `tau` values, their range and standard deviation, the shift against the frozen `tau` (0.57 / 0.19 / 0.09, `frozen_val.json`), and the shift against D3 **in the same fold**; a shift above 0.10 is "material" as in A4.5.

**A7.5 Risk scores.** BKT only. PFA and the risk head were not re-run in A4 because they need a definition-specific rewrite of their scoring; **they are not included here either** and this is a stated limitation (cost: not cheap; not attempted).

**A7.6 Reading rule and status.** Descriptive; no pass or fail. Every result file carries `"status": "exploratory, validation only (A7)"`. Seeds as stated; fitting is deterministic. No result changes the frozen thresholds, the demo or any published number.

### Amendment 8 — Parameter drift across skills (2026-10-22, before any A8 code runs or any A8 number exists)

Reason: the demo applies one set of pooled BKT parameters to every problem, while the research pipeline uses per-skill parameters where a skill has enough data. The owner asked how much the parameters differ across skills and how much the Quiet / Balanced / Supportive decisions would change if a skill's own parameters replaced the pooled ones. This replaces the invalid "synthetic calculus" idea. **It estimates transfer risk across ASSISTments skills; it is NOT a calculus calibration, and no statement about calculus follows from it.** Exploratory, validation only; the 2009 test split and the 2012 data are not read; no new test evaluation.

**A8.1 Fit.** On the 2009 **train** split only (`make_split`, test students dropped as in `run_mastery_sens.py`), the BKT of `baselines.py` is fitted per skill (prior, learn, guess, slip) for every skill with at least `bkt_min_seq` train sequences (the value in `baselines_val.json`); the pooled parameters are the fallback model's (`BKT.params` for an unseen skill).

**A8.2 Spread.** Per parameter: median, interquartile range (25th and 75th percentile), minimum and maximum across the fitted skills, the number of skills, and the share of sequences those skills cover. Also each skill's distance from the pooled value; skills at a parameter bound are counted.

**A8.3 Grouping (fixed now, from skill names only).** If the skill name contains one of the keywords below the skill gets that group, tested in this order: **algebra-type** ("Equation", "Solving for a variable", "Linear", "Slope", "Inequalit", "Exponent", "Order of Operations"); **geometry / measurement** ("Area", "Volume", "Circumference", "Surface", "Angle", "Pythagorean", "Similar", "Perimeter"); **data / probability** ("Probability", "Venn", "Box and Whisker", "Scatter", "Histogram", "Median", "Mean", "Stem", "Table", "Circle Graph", "Counting", "Pattern"); **number / arithmetic** ("Addition", "Subtraction", "Multiplication", "Division", "Fraction", "Decimal", "Percent", "Integer", "Ordering", "Estimation", "Proportion", "Unit Rate", "Multiple", "Divisib", "Square Root", "Scientific", "Absolute", "Number Line", "Rounding"); otherwise "other". Names are a rough label, not an ontology; group sizes are reported and a group with fewer than 5 skills is reported as such. Parameters are summarised per group.

**A8.4 Decisions on VALIDATION.** For the frozen Platt calibrator and the frozen `tau` per profile (`frozen_val.json`; frozen files are only read), compare two ways of scoring each validation decision cell (D3, `T = 10`, decision points 3-9): (i) **own-skill** parameters (skills with enough data; others pooled), which is the frozen research pipeline; (ii) **pooled** parameters for every skill, which is what the demo does. Report per profile: the number of decision cells, the share where the speak / flag decision differs, flips in each direction, and the reduction versus always-help and the missed-help share (student-bootstrap 95% intervals, 1,000 draws, seed `20261024`) under (i) and under (ii) using the same frozen Platt and `tau`; and the share of flipped cells per skill group. Cells of skills that have no own parameters are identical under (i) and (ii) and are reported separately.

**A8.5 Status.** Every file carries `"status": "exploratory, validation only (A8); transfer risk across ASSISTments skills, not a calculus calibration"`. No result changes a frozen file, the demo or any published number.

### Amendment 9 — Per-turn hazard model with response time (2026-10-22, before any A9 code runs or any A9 number exists)

Reason: the frozen risk models score only decision points k = 3, 5, 8. The owner asked for a per-turn model for k = 1..9 that also uses response latency. **Exploratory, validation only; the 2009 test split and the 2012 data are not read; no new test evaluation.**

**A9.0 Feasibility check (done before this text was written; no result was computed).** The raw 2009 file has `ms_first_response` (time to first response in milliseconds), `attempt_count` and `hint_count`. `aria_pace/loader.py:12` (`USECOLS`) does **not** load `ms_first_response` (it loads `user_id, assignment_id, order_id, skill_id, correct, first_action, hint_count`); loading it is a change to a new loader path only. In the raw file `ms_first_response` has no missing values, 0.09% of values are zero or negative (a clock artefact; treated as missing), and a very long upper tail (maximum about 84 million ms), so it is used only through a rank transform. It is therefore usable.

**A9.1 Data and labels.** Train / validation as in A4; variant A, definition D3 (three in a row, `T = 10`); eligible sequences for decision point `k` in 1..9: `n >= k`, `m > k`, labelled. **k = 1 and 2 are outside the frozen policy window (A1.2, decision points 3-9)**: they are scored and reported here but cannot change the policy.

**A9.2 Features at `k`** (computed from opportunities 1..k only): (1) BKT risk (the raw score of `mastery_sens.bkt_risk` under D3 at prefix length `k`, per-skill parameters from train); (2) number of wrong responses among the last 3 opportunities; (3) cumulative `hint_count` over opportunities 1..k; (4) log response time of opportunity `k` (`log(ms_first_response)`, values <= 0 treated as missing and set to the train median of the log) **after a rank transform fitted on TRAIN only** (mid-rank ECDF as `freeze_v2.py` uses; applied to validation). Model: an L2 logistic regression per `k` (C = 1.0, fixed now), fitted on train sequences, evaluated on validation.

**A9.3 Comparison.** AUC of the model versus the BKT raw risk on validation per `k` = 1..9, paired student-bootstrap 95% interval of the difference (1,000 draws, seed `20261025`). Promotion rule (unchanged from A3.5 / A5.4): the model is **promoted only if** the AUC difference is >= +0.01 with the interval lower bound above zero at **both** k = 3 and k = 5; otherwise BKT stays. A promotion would not change any frozen file or the demo by itself.

**A9.4 Ablations (descriptive).** The same model without the response-time feature, and with response time only, so any gain is attributed.

**A9.5 Status.** Every file carries `"status": "exploratory, validation only (A9)"`. The live demo already rescored the BKT risk on every attempt 3-9 (`agent/demo_pace.py`, decision points follow the frozen window); this analysis does not touch it.
