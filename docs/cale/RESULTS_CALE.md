# CALE-AI additional analyses: results

**Every result below is exploratory and post hoc.** These analyses were added after the locked evaluation (plan committed first: `docs/cale/ANALYSIS_PLAN.md`). No threshold, calibrator or model was re-chosen. The test set was not re-run. The test lock, frozen files and tags are unchanged. Data: ASSISTments 2009-10 skill builder only. All CV numbers use the A7 five-fold cross-validation over train + validation students (2,349 in-scope held-out students, 14,640 sequences, 2,279 stuck sequences with a help opportunity); no test student is included. Intervals are 95% student-level bootstrap percentile intervals (1,000 resamples). Replay on logs: nothing here measures learning, causal effects, or how real students would react. "Stuck", "missed help" and "unnecessary" are defined on the logged label (SPEC section 7). Every number traces to a file in `aria_pace/results_cale/`.

## Step 1. Reproduction check: passed

Recomputing the frozen pipeline in memory reproduces `frozen_val.json` (Platt within 1e-9; every tau and n exactly) and every value in `val_results.json` (0 differences; `step1_reproduction.json`). Validation main numbers, BKT: Quiet 1.35 first hints per student, a 49.8% reduction vs always-help (2.70), and 4.8% missed help (28 of 580). Balanced: 13.0% reduction, 0 missed. Supportive: 2.7% reduction, 0 missed. Reminder: these validation numbers are optimistic, because Platt and tau were chosen on the same students.

## E1. Trade-off curve (main figure)

Figure: `aria_pace/results_cale/figures/e1_tradeoff.pdf`. Data: `e1_tradeoff.json`.

| Policy (CV held-out) | First hints / student | Reduction vs always-help | Missed help |
|---|---|---|---|
| Always-help (n = 1) | 2.67 [2.49, 2.87] | 0 | 0 |
| BKT Quiet, tau 0.57 | 1.31 [1.22, 1.42] | 50.7% [49.0, 52.2] | 6.2% [5.3, 7.2] (142) |
| BKT Balanced, tau 0.19 | 2.32 [2.16, 2.50] | 12.8% [11.8, 13.8] | 0.3% [0.1, 0.5] (6) |
| BKT Supportive, tau 0.09 | 2.59 [2.41, 2.79] | 2.8% [2.4, 3.3] | 0.04% [0, 0.14] (1) |
| n-wrong rule n = 2 (tuned Quiet rule) | 1.38 [1.27, 1.50] | 48.3% [46.8, 49.7] | 27.3% [25.4, 29.3] |
| n = 3 (untuned three-wrong) | 0.75 [0.68, 0.81] | 72.0% [70.6, 73.3] | 57.6% [55.4, 59.7] |
| n = 4 / n = 5 | 0.30 / 0.14 | 88.7% / 94.9% | 78.7% / 89.3% |
| Random, matched to Quiet rate | 1.31 | 50.7% | 33.2% [32.5, 34.0] |

The thresholds used for the markers are the frozen values (0.57, 0.19, 0.09), applied to risk calibrated per fold. Tau chosen on each fold's calibration part instead reproduces A7 exactly (Quiet 50.1% reduction, 6.2% missed).

- **What it shows:** at the same number of hints, the BKT threshold rule misses far less help than the n-wrong rules or a random policy. Near the Quiet setting it misses about 21 points less than the two-wrong rule while giving slightly fewer hints.
- **What weakens it:** at the Quiet hint rate, the logistic risk head (4.6%) and PFA (5.7%) miss less help than BKT (6.2%) (point estimates interpolated along the sweeps, no interval). BKT was made primary by the pre-registered AUC rule, not by this curve, so the curve does not show BKT is the best risk score. The CV Quiet missed-help rate (6.2%) is higher than on the locked test (3.7%).
- **What it does not show:** whether fewer hints, or a missed hint, changes what a student learns.

## E2. Calibration (validation, decision tries 3-9)

Figure: `figures/e2_calibration.pdf`. Data: `e2_calibration.json`. 11,867 decision cells; 4,508 of them are stuck cells, from 596 students.

| | ECE (10 equal-count bins) | Brier |
|---|---|---|
| Raw BKT | 0.012 | 0.137 [0.130, 0.145] |
| After Platt (in-sample) | 0.005 | 0.137 [0.130, 0.144] |
| CV held-out, raw | 0.012 | 0.135 [0.132, 0.139] |
| CV held-out, Platt fitted on each fold's CAL part | 0.005 | 0.135 [0.132, 0.139] |

| Try | Cells | ECE raw / Platt | Calibration intercept, slope (raw) |
|---|---|---|---|
| 3 | 3,742 | 0.041 / 0.032 | -0.40, 0.94 (over-predicts) |
| 5 | 1,769 | 0.029 / 0.032 | -0.03, 0.95 |
| 7 | 1,037 | 0.050 / 0.059 | +0.22, 0.94 (under-predicts) |

- **What it shows:** pooled over tries 3-9, raw BKT risk is already close to calibrated (calibration slope 0.998, intercept -0.086). Platt scaling only removes that small offset, and the result holds out of sample on CV.
- **Weaker than expected:** within a single try, calibration is worse (ECE 0.03-0.06). Raw risk over-predicts at try 3 and under-predicts at try 7, and the one Platt line does not fix this.
- **Reporting note:** student-bootstrap intervals for ECE are not centred on the estimate, because ECE is biased upward under resampling (e.g. Platt 0.005, resampled range 0.006-0.023). They are in the JSON but should not be quoted as confidence intervals.
- **What it does not show:** calibration on other platforms or for other students.

## E3. Error analysis of missed help (CV held-out, BKT Quiet tau 0.57)

Data: `e3_errors.json`, `e3_summary.md`. 142 missed and 2,137 caught out of 2,279 stuck sequences with a help opportunity.

| Median (IQR) | Missed | Caught |
|---|---|---|
| Wrong answers, tries 1-9 | 5 (4-7) | 6 (5-8) |
| Verified errors at decision points 3-9 | 2 (1-2) | 4 (3-5) |
| Wrong answers that are not verified errors (hint-first or other code) | 3 (1-5) | 1 (0-3) |
| Right/wrong switches, tries 1-9 | 5 (3-6) | 4 (2-5) |
| Has two or more consecutive wrong answers | 82% | 94% |
| Correct answers before first error (mean) | 0.46 | 0.22 |
| Max risk at a verified error | 0.43 (0.35-0.49) | 1.00 (1.00-1.00) |
| Sequence length n | 13 (11-19) | 14 (12-20) |

Most common patterns among misses (tries 1-9, 1 = correct): `000000000` 15 (11%; 239 caught with the same pattern), `011011011` 6, `101010000` 3, `010101101` 3, `010101011` 2. Misses are spread over 51 skills.

- **What it shows:** misses are mostly a property of the verified-error gate, not of the risk score. In **118 of 142 misses (83%)** the risk reached 0.57 at some decision point, but no verified error came at or after it: the later wrong answers were hint-first or "other" `first_action` codes, which the frozen definition does not count as verified errors. The remaining misses have alternating right/wrong answers, which keep the risk below the threshold.
- **What it does not show:** anything about the 19 locked-test misses. **Not done:** the locked test run saved only aggregates, so no per-sequence test outputs exist, and the test set was not re-run.

## E4. Paired comparison with the two-wrong rule (CV held-out, not test)

Data: `e4_paired.json`. Per-sequence test outputs do not exist, so this ran on the CV folds.

| | BKT Quiet (tau 0.57) | n-wrong rule, n = 2 | Difference (BKT minus rule), paired bootstrap |
|---|---|---|---|
| Missed help | 6.2% [5.3, 7.4] (142/2,279) | 27.3% [25.2, 29.2] (622/2,279) | **-21.1 points [-23.1, -19.0]** |
| First hints / student | 1.31 [1.21, 1.41] | 1.38 [1.27, 1.49] | **-0.065 [-0.101, -0.026]** |

With each fold's calibration-part tau instead of 0.57: missed help -21.1 points [-23.2, -19.1], hints -0.048 [-0.083, -0.009]. For context only, the locked-test aggregates already on disk (quoted, not recomputed, no interval possible) are BKT 3.7% missed (19/518) at 1.22 hints per student, and the n = 2 rule 25.9% (134/518) at 1.28.

- **What it shows:** on held-out students, the BKT rule misses about a fifth fewer stuck sequences than the tuned two-wrong rule and gives slightly fewer hints. Both intervals exclude zero.
- **What it does not show:** a test-set interval, or any effect on students.

## E5. Logged-hint check

Data: `e5_logged_hints.json`. **Column used:** `hint_count > 0`, the number of hints the student requested from ASSISTments on that problem. `first_action`, `bottom_hint` and `hint_total` were used only descriptively; `hint_total` counts hints available, not hints received. In validation, 4,112 of 24,786 rows at tries 1-9 have a logged hint. 1,844 of those are hint-first and 2,268 are a wrong answer followed by a hint; only 1 is a correct row. 2,267 of the 7,758 verified-error rows have a logged hint.

| Logged hint at tries 1-9 | Train | Validation |
|---|---|---|
| In-scope sequences with any logged hint | 5,269 / 10,898 (48.3%) | 1,785 / 3,742 (47.7%) |
| Stuck sequences with a logged hint | 70.4% | 70.0% |
| Non-stuck sequences with a logged hint | 43.8% | 43.1% |
| Decision points with a hint at that try | 6,450 / 34,468 | 2,134 / 11,867 |
| Decision points at or after the first hint | 18,602 / 34,468 | 6,483 / 11,867 |
| Verified-error decision points at or after the first hint | 5,645 / 10,399 | 2,057 / 3,631 |

Validation re-evaluation with the frozen BKT, Platt and taus:

| | Main | (a) without sequences with a logged hint | (b) without decision points at or after the first logged hint |
|---|---|---|---|
| Sequences / stuck / stuck with opportunity | 3,742 / 644 / 580 | 1,957 / 193 / 186 | 3,742 / 644 / 224 |
| Always-help hints / student | 2.70 | 1.61 | 1.29 |
| Quiet: hints / student | 1.35 [1.16, 1.56] | 0.64 [0.52, 0.77] | 0.49 [0.40, 0.59] |
| Quiet: reduction | 49.8% [46.6, 53.1] | 60.4% [55.3, 65.0] | 62.0% [57.3, 66.3] |
| Quiet: missed help | 4.8% [3.0, 6.9] | 3.2% [0.8, 6.4] | **14.7% [10.0, 19.9]** |
| Balanced: reduction / missed | 13.0% / 0.0% | 19.9% / 0.0% | 19.3% / 0.9% [0, 2.2] |
| Supportive: reduction / missed | 2.7% / 0.0% | 4.7% / 0.0% | 4.4% / 0.0% |

- **What it shows: a limitation for the paper.** The stuck label is entangled with help that ASSISTments already gave: 70% of stuck sequences received in-system hints, compared with 43% of non-stuck ones. If decision points after the first logged hint are removed (b), the Quiet missed-help rate roughly triples, from 4.8% to 14.7%. That evaluation also keeps only 224 of 580 stuck sequences with a help opportunity, because many verified errors happen at or after a hint. Removing hinted sequences entirely (a) leaves 193 stuck sequences: the reduction holds (60%) and missed help is 3.2%, but the interval is wide.
- **What it does not show:** what the policy would do on students who received no help at all. Both exclusions select on something that happens after the decision, so neither is a causal estimate.

## E6. Heterogeneity across skills (CV held-out, BKT Quiet tau 0.57)

Figure: `figures/e6_skills.pdf`. Data: `e6_skills.json`. 23 of 139 skills have at least 30 stuck sequences with a help opportunity; together they cover 65% of those sequences.

| Across the 23 skills | Median | IQR | Min | Max | Pooled over all skills |
|---|---|---|---|---|---|
| Quiet reduction vs always-help | 34.7% | 25.6-58.3% | 0.0% | 72.0% | 50.7% |
| Quiet missed help | 3.2% | 0.0-12.4% | 0.0% | 26.2% | 6.2% |

- **What it shows:** the effect varies a lot by skill. The median skill saves 35%, well below the pooled 51%. Skills with a lower stuck rate save more (Spearman -0.78), and the skills with the largest savings (58-72%) also have the most missed help (11-26%; e.g. Probability of a Single Event 26%, Subtraction Whole Numbers 22%).
- **What it does not show:** no per-skill significance tests were run. Small skills give noisy rates. Skill names are ASSISTments labels, not a curriculum ontology.

## Addendum 1 (E7-E9; plan committed before running, `ANALYSIS_PLAN.md` Addendum 1)

### E7. Quiet misses split by in-system help (CV held-out)

Data: `e7_misses_by_help.json`. "In-system help" means ASSISTments gave a hint at some try 1-9: `hint_count > 0`, or the first action was a hint request. In this file, every hint-first attempt also has `hint_count > 0`.

| | n | With in-system help before try 10 | With none |
|---|---|---|---|
| Quiet misses | 142 | 114 (80.3%) | **28** |
| Caught | 2,137 | 1,456 (68.1%) | 681 |
| All stuck with a help opportunity | 2,279 | 1,570 (68.9%) | 709 |

- **"No help from anyone" missed-help rate:** 28 / 2,279 = **1.2% [0.8, 1.7]**. These are stuck sequences that got neither a policy hint nor an ASSISTments hint. Among the 709 stuck sequences without in-system help, 3.9% [2.5, 5.5] were missed. Overall missed help: 6.2% [5.2, 7.3].
- **What it shows:** four in five Quiet misses received ASSISTments hints in the log. Most "missed" students were therefore not left with no help at all in the logged data.
- **What it does not show:** whether those logged hints were enough. In the replay, a sequence counts as stuck because it did not master within 10 tries despite those hints. This split is not a claim that the misses were harmless.

### E8. Hint-first attempts counted as decision points (labelled variant, CV held-out)

Data: `e8_hintfirst_variant.json`. In this variant, a gate error is a wrong first answer **or** a hint-first attempt at tries 3-9; "other" codes are still excluded. The thresholds are the frozen ones (not re-chosen), and always-help and help opportunities follow the variant definition. Under the variant there are 18,394 error cells (14,030 under the primary rule), always-help gives 2.88 hints per student (2.67 primary), and 2,484 stuck sequences have a help opportunity (2,279 primary).

| Variant (primary in brackets below each) | Hints / student | Reduction vs always-help | Missed help |
|---|---|---|---|
| Quiet (0.57) | 1.50 [1.38, 1.62] | 48.0% [46.4, 49.6] | **1.8% [1.3, 2.3]** (44) |
| - primary | 1.31 | 50.7% | 6.2% (142) |
| Balanced (0.19) | 2.52 [2.34, 2.73] | 12.4% [11.5, 13.3] | 0.2% [0.04, 0.3] (4) |
| - primary | 2.32 | 12.8% | 0.3% (6) |
| Supportive (0.09) | 2.80 [2.60, 3.02] | 2.8% [2.4, 3.2] | 0.04% [0, 0.13] (1) |
| - primary | 2.59 | 2.8% | 0.04% (1) |
| Two-wrong rule (n = 2) | 1.76 [1.62, 1.90] | 38.8% [37.3, 40.3] | 11.7% [10.4, 13.1] (291) |
| - primary | 1.38 | 48.3% | 27.3% (622) |

- **What it shows:** if hint requests open a decision point, Quiet missed help falls from 6.2% to 1.8%, and the reduction relative to (the larger) always-help stays at 48%. This confirms E3: most misses come from the error definition, not from the risk threshold. The gap to the two-wrong rule remains large (1.8% vs 11.7% missed, at fewer hints).
- **What it does not show:** that the variant is better. It gives 0.19 more hints per student in absolute terms. A hint-first attempt may mean the student is already being helped, so whether to treat it as an error is a design choice the logs cannot settle.

### E9. BKT vs PFA vs risk head at BKT's Quiet hint rate (CV held-out, paired)

Data: `e9_riskscore_paired.json`. Target: 3,087 first hints (1.314 per student). Each model's threshold on its per-fold-calibrated risk was found by bisection on the intervention count only; labels were not used.

| Model | Threshold | First hints / student | Missed help |
|---|---|---|---|
| BKT | 0.570 (frozen) | 1.3142 | 6.2% [5.3, 7.3] (142) |
| PFA | 0.545 | 1.3138 | 5.7% [4.8, 6.7] (130) |
| Risk head | 0.613 | 1.3142 | 4.7% [3.8, 5.6] (106) |

| Paired difference (1,000 student resamples) | Missed help | Hints / student |
|---|---|---|
| BKT minus PFA | +0.5 points [-0.2, +1.3] | +0.000 [-0.022, +0.020] |
| BKT minus risk head | **+1.6 points [+0.7, +2.5]** | 0.000 [-0.026, +0.024] |

- **What it shows:** at the Quiet hint rate, the logistic risk head misses less help than BKT (about a quarter fewer misses: 106 vs 142), and the interval excludes zero. PFA and BKT are not distinguishable.
- **What it does not show:** that the head should replace BKT. The pre-registered rule (SPEC A1.8) chose BKT by validation AUC; this comparison is post hoc, at one operating point, on CV data. It is a reason not to claim BKT is the best risk score for this decision.

## What weakens or contradicts the main result

1. **E5:** in the logs, the stuck label overlaps with help ASSISTments already gave. With decision points after the first logged hint removed, Quiet missed help is 14.7% instead of 4.8%.
2. **E3:** 83% of CV misses come from the verified-error definition (hint-first wrong answers do not count), not from the risk threshold.
3. **E1 / E9:** at the Quiet hint rate, the logistic risk head misses less help than BKT: 4.7% vs 6.2%, paired difference 1.6 points [0.7, 2.5]. PFA is not distinguishable from BKT. BKT is the primary score because of the pre-registered AUC rule, not because it is best at this decision.
4. **E6:** the pooled 51% Quiet saving hides wide skill variation (median 35%), and the high-saving skills carry the highest missed help.
5. **E2:** within a single try, calibration is noticeably worse than pooled, and Platt does not fix it.
6. The CV Quiet missed-help rate (6.2%) is higher than the locked-test rate (3.7%), as A7 already showed.
7. **E7 / E8 (mitigating, but they qualify the "missed help" wording):** 80% of Quiet misses received ASSISTments hints in the log; only 1.2% [0.8, 1.7] of stuck sequences got no help from anyone. Counting hint requests as decision points lowers Quiet missed help to 1.8%, at 0.19 more hints per student.

## Not done

- Test-set breakdown of the 19 locked-test misses (E3) and a test-set paired interval (E4): no per-sequence test outputs exist on disk, and the test set was not re-run, by design.
