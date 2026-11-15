# CALE-AI additional analyses: plan (written before any analysis code ran)

Date: 2026-11-12. Branch `cale-experiments` from `main` (548bf48).

**Every analysis below is exploratory and was added after the locked evaluation.** None of them changes a frozen file (`aria_pace/results/frozen_val.json`, `frozen_v2.json`), a threshold, the test lock (`aria_pace/results/TEST_LOCK`), a published number, or anything under `eval/`, `aria_pace/results/`, `RESULTS*.md` or `SPEC.md`. Nothing is tuned here: every threshold, calibrator, hyperparameter and fold assignment is either the frozen value or the existing A7 cross-validation procedure, used as it is.

## Data and what may be read

- ASSISTments 2009-10 skill builder only (`skill_builder_data_corrected_collapsed.csv`, sha256 checked by `aria_pace/loader.py`). The 2012-13 file is never opened.
- The student split is `aria_pace/split.py:make_split` (seed 20261015). **Test students are dropped right after the split**, exactly as in `aria_pace/freeze_val.py:22-24`, in every script. An assertion and a unit test check that no test student id reaches any computation or output.
- **Per-sequence outputs of the locked test run do not exist.** `aria_pace/run_test.py` wrote only aggregates (`test_results.json`); no per-sequence file was saved. So the post hoc test breakdown asked for in E3 cannot be produced, and E4 runs on the cross-validation folds. The test aggregates already on disk (e.g. BKT Quiet 19 missed of 518) may be quoted as context, never recomputed.

## Reused pipeline (nothing reimplemented)

| Piece | Where |
|---|---|
| Loading, cleaning, sequences | `aria_pace/loader.py:load_raw`, `build_sequences` |
| Student split | `aria_pace/split.py:make_split` |
| Label, scope, decision points 3-9, verified error (primary) | `aria_pace/policy.py:Table`, `aria_pace/labels.py` |
| BKT (per-KC EM, `min_seq` = 10), PFA (`C` = 0.1), risk head (`C` = 10) | `aria_pace/baselines.py`, `aria_pace/riskhead.py`, `aria_pace/stage5.py:fit_models` |
| Stuck risk = P(no mastery in k+1..10 given first k) | `BKT.score`, `PFA.score`, `RiskHead.score` via `stage5.raw_scores` |
| Platt calibration | `policy.platt_fit`, `policy.platt_apply` |
| Threshold selection (argmin J over `TAU_GRID`, ties to larger) | `stage5.choose_values`, `stage5.pick` |
| Policy metrics per sequence and student aggregation | `stage5.seq_vec`, `stage5.Agg`, `stage5.metric_arrays` |
| n-wrong-in-a-row rule, matched-rate random policy | `policy.run_fire`, `stage5.random_matched_vec` |
| Cross-validation folds (A7) | `aria_pace/cv_tradeoff.py:make_folds`, `fit_cal_split` (5 folds over train+validation students, seed 20261022; per fold FIT 3/4 and CAL 1/4, seed 20261022 + f) |

New code lives in `scripts/cale/`; results in `aria_pace/results_cale/`; documents in `docs/cale/`.

## Metric definitions (SPEC section 7 and Amendment 1, unchanged)

- Scope: sequences with `n >= 3`, `m > 3`, labelled under variant A (leavers dropped). `y = 1` ("stuck") = not mastered (three correct in a row) within 10 opportunities and `n >= 10`.
- Decision point `i` in 3..min(9, n) exists iff `m > i`. Verified error (primary) = `correct = 0` and `first_action = 0`.
- A policy fires at the first decision point with a verified error and calibrated risk >= tau; at most one intervention ("first hint") per sequence.
- **First hints per student** = intervened sequences / students in scope. **Reduction vs always-help** = 1 - interventions / interventions of always-help (always-help fires at the first verified error at a decision point).
- **Missed-help rate** = stuck sequences with a help opportunity (a verified error at some decision point) and no firing before opportunity 10, divided by stuck sequences with a help opportunity.

## Step 1. Reproduction check (gate)

Recompute, in memory and on validation students only, exactly the steps of `aria_pace/freeze_val.py` (fit PFA, BKT and the risk head on train; Platt on validation decision rows; `stage5.choose_values`), then `stage5.evaluate_split` on validation. Compare with the saved files without writing to them:
- `frozen_val.json`: Platt intercepts and slopes within 1e-9; every tau and n exactly.
- `val_results.json`: every number in the file (JSON equality after a round trip; any numeric difference reported with its path and size).

**If anything differs, stop and report before running E1-E6.** Output: `aria_pace/results_cale/step1_reproduction.json`.

## E1. Trade-off curve (main figure); cross-validation folds

- Data: the A7 folds (train + validation students; no test student). Per fold: BKT, PFA and the risk head fitted on FIT with the frozen hyperparameters; one Platt calibrator per method fitted on CAL decision rows (as `cv_tradeoff.calibrate_and_choose` does for BKT); risk computed and calibrated on the held-out fold only. Held-out folds are pooled.
- Threshold sweep: `policy.TAU_GRID` (0.00 to 1.00 in steps of 0.01, 101 values, plus never-fire 1.01).
- Per tau and method: first hints per student, % reduction vs always-help, missed-help rate.
- Bands: 95% percentile intervals from 1,000 student-level bootstrap resamples (multinomial counts over pooled held-out students, seed 20261112, the same resamples for every curve and point).
- Baselines on the same pooled held-out students: always-help; the n-wrong-in-a-row rule for n = 1..5 (n = 1 is always-help); the untuned three-wrong rule (n = 3, labelled separately); matched-rate random policy (`stage5.random_matched_vec`, 200 draws, seed 20261017) matched to the BKT intervention count at tau = 0.09, 0.19, 0.30, 0.45, 0.57, 0.70; PFA and risk-head tau curves.
- Markers on the BKT curve: the frozen thresholds 0.57 (Quiet), 0.19 (Balanced), 0.09 (Supportive) applied to the per-fold-calibrated risk. Also reported (not plotted as the main marker): the result at the tau chosen on each fold's CAL part (the A7 procedure); this must equal the A7 D3 numbers in `cv_tradeoff_val.json` (cross-check).
- Figure: x = first hints per student, y = missed-help rate. Output `results_cale/e1_tradeoff.json`, `results_cale/figures/e1_tradeoff.{pdf,png}`.

## E2. Calibration; validation students

- Rows: validation primary-scope decision cells, tries 3-9 (the rows the frozen Platt calibrator was fitted on). Score: BKT risk before (raw) and after (frozen Platt from `frozen_val.json`) calibration; BKT fitted on train (as in the frozen pipeline).
- Reliability diagram: 10 equal-count bins of the score (`metrics.reliability`); per bin mean predicted risk and observed stuck rate with 95% student-bootstrap intervals (bins fixed on the full sample, 1,000 resamples, seed 20261113).
- ECE (10 equal-count bins, `metrics.reliability`) and Brier score (`metrics.brier`) before and after, each with a student-bootstrap 95% interval.
- Per try 3, 5, 7: the same quantities on the cells of that try only, if the try has at least 200 cells and at least 20 stuck cells (otherwise reported as too few).
- Stated caveat: after-Platt validation calibration is **in-sample** (the calibrator was fitted on these rows). As a non-optimistic secondary, the same ECE/Brier are reported on the pooled CV held-out cells with each fold's CAL-fitted Platt (from E1).
- Output `results_cale/e2_calibration.json`, `results_cale/figures/e2_calibration.{pdf,png}`.

## E3. Error analysis of missed help; cross-validation folds

- Operating point: BKT at the frozen Quiet tau 0.57 on the per-fold-calibrated held-out risk (the E1 marker).
- Population: pooled held-out stuck sequences (`y = 1`) with a help opportunity; split into **missed** (never fired) and **caught** (fired).
- Per group, descriptive only (counts, mean, median, IQR or shares): sequence length `n`; number of wrong answers at tries 1-9; number of primary verified errors at decision points; number of hint-first or other-code wrong answers (wrong answers that are not verified errors); longest wrong run in tries 1-9; number of right/wrong switches in tries 1-9 (alternating vs consecutive); a pattern class (has 2+ consecutive wrongs at tries 1-9 vs none); correct answers before the first error; maximum calibrated risk reached over the sequence's decision points (all and verified-error cells only); skill (KC id and name; top skills among misses with their miss share).
- The 5 most common correctness patterns at tries 1-9 among misses (string such as `110101011`, 1 = correct), with counts and how often the same pattern is caught.
- Test misses: not possible (no per-sequence test outputs on disk); stated as not done.
- Output `results_cale/e3_errors.json`, `results_cale/e3_summary.md`.

## E4. Paired comparison with the tuned two-wrong rule; cross-validation folds

- Because per-sequence test outputs do not exist, E4 runs on the pooled CV held-out students and says so.
- Policies on the same sequences: BKT at Quiet tau 0.57 (per-fold calibration) and the n-rule with n = 2 (the frozen tuned Quiet n).
- Statistics: difference BKT minus rule in missed-help rate and in first hints per student; paired student-level bootstrap (the same 1,000 multinomial resamples for both policies, seed 20261114), 95% percentile interval. Rates are ratio-of-sums within each resample.
- Sensitivity (reported, not primary): BKT at each fold's CAL-chosen Quiet tau.
- Context only: the locked test aggregates already in `test_results.json` (BKT Quiet and tuned n-rule missed help and interventions per student), quoted, not recomputed, no interval (no per-sequence data).
- Output `results_cale/e4_paired.json`.

## E5. Logged-hint check; training and validation

- Columns: `hint_count` (number of hints the student requested on that problem in ASSISTments) is the indicator of a logged hint (`hint_count > 0`). Descriptive cross-checks: `first_action = 1` (inferred: hint as first action), `bottom_hint`, `hint_total` (hints available, not received; not an indicator). Rows are aligned to the pipeline's sequences by `(user_id, order_id)` after the loader's cleaning, and the alignment is asserted against `correct`.
- Counts on train and on validation (separately): in-scope sequences with any logged hint at tries 1-9; decision cells (tries 3-9) and verified-error cells with a logged hint at that try, and those at or after the first logged hint.
- Re-run of the validation evaluation with **frozen** BKT (fit on train), frozen Platt and frozen taus (all three profiles), nothing re-chosen:
  - (a) exclude sequences with any logged hint at tries 1-9;
  - (b) keep all sequences but drop decision points at or after the first logged hint (a decision point `i` is dropped if `hint_count > 0` at any try `t <= i`; the hint at try `i` is received before decision point `i`).
  - Metrics per profile: first hints per student, missed-help rate, reduction vs always-help (always-help recomputed on the same subset), each with a 95% student-bootstrap interval (1,000 resamples, seed 20261115); compared with the main validation numbers from Step 1.
- Stated caveat: excluding sequences on a post-decision variable (hints) is a selection; this is a robustness check of the label's dependence on in-log help, not a causal estimate.
- Output `results_cale/e5_logged_hints.json`.

## E6. Heterogeneity across skills; cross-validation folds (only after E1-E5)

- Operating point as E3 (BKT, Quiet tau 0.57, per-fold calibration), pooled held-out sequences.
- Skill = the sequence's KC (first-opportunity `skill_id`). Skills with at least 30 stuck sequences with a help opportunity in the pooled held-out data.
- Per skill: Quiet reduction vs always-help and missed-help rate. Across skills: median, IQR, min, max, number of skills, and the share of all stuck sequences they cover. Plot of the per-skill distribution. Descriptive only; no per-skill significance tests.
- Output `results_cale/e6_skills.json`, `results_cale/figures/e6_skills.{pdf,png}`.

## Figures

Vector PDF 3.3 in wide (single AAAI column), fonts >= 8 pt at that size, TrueType fonts embedded; colorblind-safe palette (Okabe-Ito for baselines); setting colors Quiet #2457A6, Balanced #7A4AA8, Supportive #1F7A58. A PNG of each figure for preview.

## Reproducibility

`scripts/cale/run_all.sh` runs Step 1, then E1-E6, stopping if Step 1 fails. Fixed seeds as listed; each output records the sha256 of the raw file and of every input file under `aria_pace/results/` it reads, package versions, and runtime. `docs/cale/REPRODUCE.md` and `docs/cale/RESULTS_CALE.md` are written after the runs.

## What none of this shows

Replay on logs cannot say what would have happened had a student been helped or not. No analysis here measures learning, causation, or real students' reactions. "Stuck" is the logged label, "missed help" and "unnecessary" are defined on that label.

---

## Addendum 1 (2026-11-12, written and committed before E7-E9 ran)

Same rules as above: exploratory and post hoc; CV folds only (the A7 folds, pooled held-out, no test student); no test-set run; nothing re-tuned; frozen files, `eval/`, `aria_pace/results/`, `SPEC.md`, the test lock and the tags untouched. The original definitions (verified error = `correct = 0` and `first_action = 0`) are unchanged; E8 is a labelled variant. All intervals: 95% percentile, 1,000 student-level bootstrap resamples (multinomial counts over pooled held-out students).

### E7. Missed help split by in-system help

- Population: the Quiet misses of E3 (BKT, frozen tau 0.57, per-fold CAL Platt; 142 stuck sequences with a help opportunity and no firing).
- In-system help before try 10 = at any try 1-9, `hint_count > 0` **or** a hint-first attempt (`first_action = 1`). Same row alignment as E5 (asserted against `correct`).
- Report: misses with and without in-system help; the same split for caught and for all stuck sequences with a help opportunity.
- **"No help from anyone" missed-help rate** (primary) = misses with no in-system help at tries 1-9 / all stuck sequences with a help opportunity (2,279). Secondary: the same numerator over stuck sequences with a help opportunity and no in-system help (the miss rate within that group). Student bootstrap, seed 20261116.
- Output `results_cale/e7_misses_by_help.json`.

### E8. Hint-first attempts as decision points (labelled variant)

- Variant gate error at decision point `i` (tries 3-9, decision point exists iff not mastered at `i`): `correct = 0` and `first_action` in {0, 1} (wrong first answer **or** hint-first attempt). Code-2 rows stay excluded. Everything else as before: first firing only; always-help fires at the first variant error; help opportunity and missed help use the variant error; stuck label unchanged.
- Same pooled CV held-out risk (per-fold CAL Platt) and the **frozen** thresholds 0.57 / 0.19 / 0.09 (not re-chosen); the n-wrong rule with n = 2 under the variant (two consecutive variant errors ending at `i`).
- Report for Quiet, Balanced, Supportive and the n = 2 rule: first hints per student, reduction vs always-help (variant always-help), missed-help rate, with CIs (seed 20261117); the primary-definition numbers from E1 alongside.
- Output `results_cale/e8_hintfirst_variant.json`.

### E9. Paired comparison of risk scores at a matched hint rate

- Target: the BKT Quiet hint rate on the pooled CV held-out data (tau 0.57; 3,087 first hints, 1.31 per student).
- For PFA and the logistic risk head (per-fold CAL Platt, as in E1), one threshold each on the pooled held-out calibrated scores is found by bisection so the number of intervened sequences is as close as possible to BKT's. This uses only intervention counts, never labels. The thresholds and the achieved hint rates are reported.
- Paired student bootstrap (seed 20261118, the same resamples for all three): difference in missed-help rate BKT minus PFA and BKT minus risk head, and the difference in hints per student (to show the match).
- Output `results_cale/e9_riskscore_paired.json`.

---

## Addendum 2 (2026-11-15, written and committed before E10 ran)

Same rules as above. The analysis is exploratory and post hoc. It uses training and validation students only (test students are dropped right after the split, as in every script). There is no test-set run and nothing is re-tuned. Frozen files, `eval/`, `aria_pace/results/`, `SPEC.md`, the test lock and the tags stay untouched. No risk model is used: E10 is a planning calculation on the logs.

### E10. Micro-randomized trial (MRT) planning: proximal effect on next-answer correctness

**Data.** All train + validation students (3,331) and all of their sequences, with no policy-scope filter. Leavers and short sequences are kept, because a trial cannot select on future labels. Students with no decision point still count in N.

**Session.** The 2009-10 file has no session id or timestamp. One session = one sequence (one student's rows in one assignment, `loader.build_sequences`). This is stated as an assumption.

**Decision point (DP).** Try `i` in 3..9 with `i <= n`, not yet mastered through try `i` (`m > i`), and a verified error at try `i` (`correct = 0`, `first_action = 0`, the primary definition). These are exactly the cells where `policy.Table` has `valid & err_primary`, applied to all sequences. **Proximal outcome** Y = `correct` at try `i + 1`, the first answer on the next problem. A DP with no try `i + 1` (the student left the assignment) has no outcome. These DPs are counted and reported, but they are excluded from the availability used for power.

**Estimates (student bootstrap: 1,000 resamples, seed 20261119).**
- DPs per session: mean, distribution, and share of sessions with at least one DP. DPs per student, total over the observed record and per session (the mean of each student's per-session mean). Each is reported with and without outcome, and among all students and among students with at least one DP.
- Base rate p0 = pooled mean of Y over DPs with outcome, with a CI.
- Variability:
  - p0 by try `i`;
  - the SD of student-level rates (students with at least 5 DPs);
  - the ANOVA (method-of-moments) ICC of Y within student, for unbalanced clusters;
  - the spread across skills (first-opportunity KC, skills with at least 100 DPs: median, IQR, min, max);
  - the correlation of Y between consecutive DPs of the same student.

**Liao et al. (2016) sample size.**
- Working model: Y = α0 + (A − ρ)β0 + ε, with a constant effect (p = 1) and an intercept-only control (q = 1).
- ρ = 0.5, two-sided α = 0.05, power 0.80. Standardized effect d = Δ / sqrt(p0(1 − p0)), for Δ = 0.03, 0.05, 0.08.
- Noncentrality λ(N) = N ρ(1 − ρ) d² τ̄, where τ̄ is the expected number of DPs with outcome per student.
- Power(N) = 1 − F_{1, N−q−p; λ}(F⁻¹_{1, N−q−p}(0.95)) (noncentral F, `scipy.stats.ncf`). N is the smallest integer with power ≥ 0.80.

**Scenarios.**
- S_obs: each student contributes their observed record (all 2009-10 sessions). τ̄ = mean over students.
- S_K: K sessions per student, K = 1, 5 and 10. τ̄ = K × the mean of students' per-session mean DPs.

**Simulation-based power (nonparametric, resampling real students).**
- A simulated student is a train + validation student drawn uniformly with replacement. Under S_obs the student keeps all of their sequences. Under S_K, K of that student's sequences are drawn with replacement.
- Each sequence is replayed try by try. At each DP (recomputed on the sequence as modified so far: mastery and verified error), A ~ Bernoulli(0.5).
- If A = 0, the next answer is as logged. If A = 1 and the logged next answer is wrong, it is turned correct (and marked as an answer attempt) with probability δ. The rest of the sequence stays as logged. Treatment can therefore remove later DPs, by creating mastery or by removing an error.
- This keeps the real within-student correlation, the uneven DP counts and the dependence of availability on earlier outcomes. It implies an effect that is heterogeneous on the risk-difference scale (students who often answer the next item correctly gain less).
- δ is calibrated per Δ and scenario so that the average proximal effect over the trial's DPs, E[Y(1) − Y(0)] = δ · E[1 − Y(0)], equals Δ: 3 fixed-point iterations of 200 large simulated trials (N = 2,000 students each). The achieved Δ is reported.
- Analysis of each simulated trial: weighted-centred least squares with working independence. OLS of Y on (1, A − 0.5) over DPs with outcome, a student-clustered sandwich variance × N/(N − 2), and rejection if |t| > t_{N−2, 0.975} (the F(1, N − q − p) test).
- Power curve: N = round(N_Liao × {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6}), 1,000 trials each. A probit fit of rejection on log N is solved for 0.80 (rounded up) and then confirmed with 2,000 trials at that N, reporting power and MC SE. Seed 20261120.
- Type I error check: Δ = 0 (δ = 0) under S_obs, at the simulated N for Δ = 0.05, with 2,000 trials.

**Decision rule (fixed now).** For each Δ × scenario, the reported answer is N_Liao if the simulated power at N_Liao is within 0.77 to 0.83. Otherwise the simulation N is reported as the answer, with the reason. Both numbers are always shown. The Liao working assumptions are assessed in the write-up:
- the continuous outcome and constant residual variance, versus a binary Y;
- a constant (homogeneous) effect, which a binary outcome with heterogeneous base rates cannot have on the risk-difference scale;
- known, fixed expected availability, versus availability that depends on earlier outcomes and treatment, and that varies widely between students;
- the large-N approximation.

**Stated caveats.**
- The logged control condition already includes ASSISTments' own hints (E5), so p0 is the "usual ASSISTments" next-answer rate, not a no-help rate.
- A trial would randomize at every DP, unlike ARIA's at-most-one-hint policy.
- The flip model is an assumption about how a hint works, not an estimate.
- Nothing here estimates an effect: Δ values are inputs.

Output: `results_cale/e10_mrt_power.json`.
