# E11: late-hint decomposition and fact check (CV, post hoc, exploratory)

Branch `cale-experiments`. Read-only on `eval/`, `aria_pace/results/`, the test set and `TEST_LOCK` (sha256 of `TEST_LOCK` checked before and after: unchanged). No test student is used. New files: `scripts/cale/e11_late_hint.py`, `aria_pace/results_cale/e11_late_hint.json` (all Part A numbers), this report.

## Flags (read first)

1. **Step 0 reproduction: no mismatch.** Recomputed on the pooled CV held-out folds (same 5 folds, seed 20261022): Quiet reduction 50.7%, Quiet missed help 6.2% (142 / 2279), two-errors (tuned, n = 2) missed help 27.3% (622 / 2279). These are identical to `e1_tradeoff.json` (`frozen_tau_markers/quiet`, `n_rule/2`). Caveat: this used the cached held-out predictions (`results_cale/_cache/cv_heldout_dbd403fdbff0e13e.pkl`, key = hash of the raw CSV, `baselines_val.json`, `riskhead_val.json`), not a from-scratch refit of the five folds.
2. **`r_k` is not defined anywhere in the repo** (grep of code, docs, paper). I used your gloss: a first hint at decision point i is *late* (r_k = 1.00) when mastery by opportunity 10 is already impossible. A hint fires on an error, so the streak is 0 after it and three more correct answers are needed in opportunities i+1..10. That is possible only for i <= 7, so late means i >= 8. If your r_k reaches 1.00 already at i = 7 (zero slack), use the sensitivity row in A1 (`i >= 7`).
3. **"Logistic risk (matched)" and "PFA" are matched to Quiet's hint count** (E9 rule: one threshold per model, bisection on the intervened-sequence count, labels unused), not run at their own frozen tau (PFA 0.46, risk head 0.55, which give 1.56 and 1.42 hints per student). All three then give 1.314 hints per student.
4. **The CV "frozen" thresholds were chosen on the validation students, who are also in the CV pool** (`cv_heldout.py` pools train + validation). The Quiet CV figures are therefore optimistic for the thresholds, though BKT, Platt and calibration are refit per fold (B4).
5. CIs in A2 use a new bootstrap seed (20261123), so they differ in the last digit from the paper's (e.g. two-errors missed 25.4–29.1 here vs 25.4–29.3 in `e1_tradeoff.json`). Point estimates are identical.
6. The headline of A2: **counting only hints given while mastery was still possible, Quiet's missed help is 30.9%, not 6.2%.** The Quiet vs two-errors gap shrinks from 21.1 to 5.8 points (30.9% vs 36.7%). The risk score matched to Quiet's rate does better (23.2%). No CI is computed for these differences.

## Part A

Data: pooled CV held-out folds, 2,349 students, 14,640 in-scope sequences, 2,526 stuck, of which 2,279 have a verified-error decision point (the paper's denominator). Bootstrap: 1,000 student resamples. Quiet / Balanced / Supportive = BKT at frozen tau 0.57 / 0.19 / 0.09 on per-fold calibrated risk.

### A1. Share of first policy hints at r_k = 1.00 (i >= 8), among stuck sequences with a decision point

| Policy | Stuck seqs with a hint | Late (i >= 8) | Share late | By opportunity i = 3,4,5,6,7,8,9 | Sensitivity i >= 7 |
|---|---|---|---|---|---|
| Quiet | 2,137 | 563 | **26.3%** | 314, 328, 334, 314, 284, 385, 178 | 847 (39.6%) |
| Balanced | 2,273 | 114 | **5.0%** | 997, 612, 337, 143, 70, 73, 41 | 184 (8.1%) |
| Supportive | 2,278 | 108 | **4.7%** | 1125, 603, 293, 104, 45, 69, 39 | 153 (6.7%) |
| two-errors (extra) | 1,657 | 214 | 12.9% | | 360 (21.7%) |
| three-errors (extra) | 967 | 150 | 15.5% | | 253 (26.2%) |

### A2. Missed help counting only hints given while mastery was still possible (i <= 7)

Denominator 2,279 stuck sequences with a decision point. "Missed (paper)" = no hint at all; "missed (early only)" = no hint at i <= 7 (a late hint counts as missed). Hints per student = first hints per student over all in-scope sequences.

| Policy | Hints/student (all) | Hints/student at i <= 7 | Missed (paper), % [95% CI], n | Missed (early only), % [95% CI], n |
|---|---|---|---|---|
| Quiet | 1.314 | 1.074 | 6.2 [5.3, 7.3], 142 | **30.9 [29.0, 32.8], 705** |
| Balanced | 2.324 | 2.276 | 0.26 [0.08, 0.47], 6 | 5.3 [4.3, 6.3], 120 |
| Supportive | 2.590 | 2.544 | 0.04 [0.00, 0.14], 1 | 4.8 [3.9, 5.8], 109 |
| two-errors (tuned, n = 2) | 1.379 | 1.288 | 27.3 [25.4, 29.1], 622 | **36.7 [34.6, 38.8], 836** |
| three-errors | 0.748 | 0.684 | 57.6 [55.4, 59.9], 1,312 | 64.2 [62.1, 66.2], 1,462 |
| random, matched to Quiet | 1.314 | 1.193 | 33.2 [32.5, 34.1], 757.2* | 45.7 [44.8, 46.6], 1,041.7* |
| PFA, matched to Quiet (thr 0.545) | 1.314 | 1.090 | 5.7 [4.7, 6.7], 130 | 28.7 [26.7, 30.7], 655 |
| Logistic risk head, matched to Quiet (thr 0.613) | 1.314 | 1.134 | 4.7 [3.8, 5.6], 106 | 23.2 [21.4, 25.0], 529 |

\* expected counts, average of 200 random draws (seed 20261017, same as E1). The random-matched CI reflects student resampling of the draw-averaged vector only.

Secondary denominator (stuck sequences with a verified error at some i <= 7, n = 2,173, so help could have been early): early-only missed is Quiet 27.6%, Balanced 0.6%, Supportive 0.1%, two-errors 33.6%, three-errors 62.4%, PFA matched 25.3%, risk head matched 19.5% (JSON key `A2_secondary_denominator_stuck_with_error_at_i_le_7`, with CIs).

### A3. Opportunity of the first policy hint in stuck sequences (those that got a hint)

| Policy | n | Median | IQR (Q1–Q3) | Mean | Share of the 2,279 that got any hint |
|---|---|---|---|---|---|
| Quiet | 2,137 | **6** | 4–8 | 5.84 | 93.8% |
| two-errors | 1,657 | **4** | 3–6 | 4.88 | 72.7% |

### A4. I(tau) and delay D on CV (definitions of the J objective: `policy.py:evaluate_fire`, `stage5.py:seq_vec`)

I = hints given on sequences that go on to master by opportunity 10 (unnecessary interruptions), per student. D = delay on stuck sequences with a help opportunity: i_first − i_err if hinted, else 10 − i_err (censored). Mean and median are over the 2,279 stuck sequences with a decision point, in opportunities.

| Policy | I per student (count) | D per student (total) | Mean delay | Median delay | IQR | Mean delay, hinted only |
|---|---|---|---|---|---|---|
| Quiet | 0.404 (950) | 2.071 (4,864) | 2.134 | 2 | 0–4 | 1.869 |
| Balanced | 1.357 (3,187) | 0.192 (451) | 0.198 | 0 | 0–0 | 0.181 |
| Supportive | 1.620 (3,805) | 0.030 (70) | 0.031 | 0 | 0–0 | 0.028 |
| always-help | 1.695 (3,982) | 0 (0) | 0 | 0 | 0–0 | 0 |
| two-errors | 0.674 (1,583) | 2.241 (5,265) | 2.310 | 1 | 0–4 | 1.228 |

Each policy's own J per student: Quiet 3.284 (w_I 3, w_D 1), Balanced 1.549, Supportive 1.709. With Quiet weights, two-errors scores 4.263 and always-help 5.086.

## Part B

**B1. Sequence.** One `Sequence` per (user_id, assignment_id), i.e. one student's skill-builder problem set, with opportunities re-ranked by `order_id` (`aria_pace/loader.py:64-72`; dataclass at `loader.py:51-60`). `kc` is the skill string of its first row; multi-skill problems are already collapsed in the file. 52,363 sequences / 3,331 students in the CV pool (train + validation).

**B2. Unmastered sequences with fewer than 10 opportunities.** Label `None`, an "unlabelled leaver", excluded from the stuck/not-stuck evaluation (`aria_pace/labels.py:17-22`: mastered by 10 -> 0; else n >= 10 -> 1; else None). In `Table` they are dropped unless `include_leavers=True` (`policy.py:21-26`), and CV uses `include_leavers=False` (`cv_heldout.py:54,62`). Sequences with n < 3 or mastered by opportunity 3 are out of scope before labelling (`policy.py:21`). Counts in the CV pool (train + validation), computed by loading the pool with test students removed: 31,618 sequences with m > 10 and n < 10, of which 26,664 have n < 3 (out of scope) and **4,954 are in-scope unlabelled leavers (excluded)**. Labelled: 14,640 in scope, 2,526 stuck (n >= 10, m > 10; 327 of these have n = 10 exactly). Test: only the in-scope count exists in an output file, **1,226** (`aria_pace/results/test_results.json`, `leavers/scope/leavers_included`); the test n < 3 count is not in any output file and I did not compute it.

**B3. Scaffolding rows and hint-first actions.** Rows with `original == 0` are **not removed**: `original` is not even loaded (`loader.py:11`, `USECOLS`); the only row filter is a null `skill_id` (`loader.py:30-33`). Among non-test students, 19,535 of 229,697 cleaned rows (8.5%) have `original == 0` (computed from the raw CSV restricted to non-test students). Mastery is `correct` only (`labels.py:7-14`, 3 consecutive 1s). In the data, every row with `first_action` 1 (hint request) or 2 has `correct == 0` (non-test students: 17,525 and 4,977 rows; zero rows with correct = 1 and first_action != 0), so a hint-request first action counts as an incorrect response and **resets the streak**. It is a decision point only under the alternative definition (B6).

**B4. CV refits.** Per fold, BKT, PFA and the risk head are refit on that fold's FIT students (`cv_heldout.py:53`, `S5.fit_models`) and Platt is refit on the fold's CAL students (`cv_heldout.py:57-60`). Only the hyperparameters (PFA C 0.1, BKT min_seq 10, head C 10) are carried from validation (`cv_heldout.py:37-38`). The Quiet/Balanced/Supportive CV points use the **frozen validation thresholds** 0.57 / 0.19 / 0.09 (`common.py:33`, used in `e1_tradeoff.py:60-63`) on those per-fold calibrated scores. Thresholds chosen per fold on CAL are computed separately (`cv_heldout.py:61`, `chosen_fire`; E1 `cal_chosen_tau_points`, A7 `cv_tradeoff_val.json`) and are not used in the 50.7% / 6.2% numbers. The tuned two-errors rule is the fixed n = 2 from validation (`e1_tradeoff.py:67-68`).

**B5. ECE.** ECE 0.012 -> 0.005 (0.0118 -> 0.0051): BKT risk on the **validation** primary-scope decision cells, tries 3–9 (11,867 cells, 10 equal-count bins), BKT fit on train, frozen Platt fitted on those same rows, so the after-Platt value is **in-sample** (`results_cale/e2_calibration.json`, `validation_all_tries`; `e2_calibration.py:83-86`). The out-of-sample CV version is 0.0116 -> 0.0047 (46,335 cells, `cv_heldout`). The per-opportunity 0.03–0.06 matches the same validation cells split by try (3, 5, 7; `validation_per_try`): raw 0.041 / 0.029 / 0.050, after Platt 0.032 / 0.032 / 0.059 (0.03–0.06 is the after-Platt range; the raw range is 0.03–0.05). Platt does not lower per-try ECE at tries 5 and 7.

**B6. Hint requests as decision points, two-errors hints per student.** **1.764** [1.624, 1.903] (4,143 interventions; reduction 38.8%, missed help 11.7%, 291 of 2,484 stuck with a help opportunity), vs Quiet 1.499 and always-help 2.881 under the same definition (`results_cale/e8_hintfirst_variant.json`, `variant/two_wrong_rule`). n = 2 is held fixed, not re-tuned. Under the primary definition it is 1.379.

**B7. Closed-loop simulation (A6).** J per sequence = w_I · unnecessary + w_D · delay (`aria_pace/closed_loop.py:outcomes`; weights per profile from `policy.PROFILES`, `summarise_point`). Unnecessary = a hint on a student who would have mastered in the no-help world; delay = i_first − i_err if hinted, else 10 − i_err, over stuck sequences with an error. There is no separate dropout or learning term in J. Both act through the simulated trajectories: `p_help_effect` raises the learn probability after help, `p_interrupt_cost` raises slip on the next response, and `p_stuck_cost` is the chance of abandoning after an unaided error, which removes the student from play (`closed_loop.py:54-110`). The grid is p_help_effect {0, .1, .2, .3, .5} × p_interrupt_cost {0, .1, .2, .3} × p_stuck_cost {0, .05, .1, .2, .3} (`closed_loop.py:21`; 100 points, 100,000 students each, seed 20261101). The {0, 0.05, 0.1, 0.2, 0.3} grid you quote is the p_stuck_cost axis. The interval 0.087–0.100 is **not a grid value**: for each of the 20 (p_help_effect, p_interrupt_cost) pairs, delta J (Quiet minus always-help) is negative at p_stuck_cost 0.05 and >= 0 at 0.1, and the crossing is linearly interpolated between those two grid points (`scripts/cale/paper_numbers.py:238-249`). The 20 crossings run from 0.087 (help 0, interrupt 0.3) to 0.100 (help 0.5, interrupt 0). It uses point estimates of delta J, not the CI-based "flips" region, and it assumes linearity between 0.05 and 0.1.

**B8. Random-matched.** Averaged, not single-seed: 200 draws from one RNG stream (seed 20261017, `stage5.py:19-20,123-143`), and the reported vector is the draw mean. It is matched on the **same table being evaluated** by bisecting a threshold on uniform draws until the number of hinted sequences equals BKT's count there (labels unused), not a probability tuned on validation. In CV it is matched to BKT at tau 0.57 on the pooled held-out folds (`e1_tradeoff.py:70-73`), giving 1.314 hints per student.

**B9. Validation AUCs (pre-specified comparison, variant A, validation; `dkt_val.json`, `sakt_val.json`).**

| k | BKT | DKT [95% CI] | DKT − BKT (paired) | SAKT [95% CI] | SAKT − BKT (paired) |
|---|---|---|---|---|---|
| 3 | 0.8125 | 0.8098 [0.791, 0.830] | −0.0027 | 0.8047 [0.785, 0.825] | −0.0078 |
| 5 | 0.8055 | 0.8027 [0.780, 0.826] | −0.0027 | 0.7921 [0.770, 0.815] | −0.0134 |
| 8 (reported, not deciding) | 0.9105 | 0.9141 | +0.0036 | 0.9112 | +0.0006 |

Neither is promoted (rule: AUC gain >= 0.01 with paired CI lower bound > 0 at both k = 3 and 5).

**B10. fig2_tradeoff: n = 4 and 5 vs the BKT curve at the same hint rate** (`fig2_tradeoff.py` plots `e1_tradeoff.json`; BKT value linearly interpolated along its sweep at each rule's hint rate; point estimates):

| n | Hints/student | Rule missed help | BKT missed help at same rate | Gap |
|---|---|---|---|---|
| 4 | 0.301 | 78.7% [76.9, 80.5] | 68.9% (band 68.2–69.7) | +9.8 pts |
| 5 | 0.136 | 89.2% [88.0, 90.5] | 86.0% (band 85.6–86.3) | +3.2 pts |

Both sit above (worse than) the BKT curve, and each rule's interval lies outside the BKT band at that rate. The gap narrows toward the high-silence end (n = 2: 27.3% vs 5.7%; n = 3: 57.6% vs 22.9%).
