# ARIA-Pace RESULTS v2: the locked ASSISTments 2012-13 test portion

> **INTERNAL: not for publication.** Licence confirmation from ASSISTments is PENDING (docs/aria_pace/DATA_v2.md). Nothing here is to be published, posted or shown outside the team until it arrives.

Evaluated once on the locked 2012 test portion (4,651 students, seed 20261016; `results/split_2012.json`). Frozen values: `results/frozen_v2.json` (sha256 c455ed708157..., committed in c458650 before any test row was read). Lock file `results/TEST_LOCK_2012` (started 2026-10-18T15:45:40.783906, completed 2026-10-18T16:06:07.985976). Raw numbers: `results/test_results_2012.json`; this file is generated from it by `aria_pace/make_results_v2.py`.

## Caveats (they apply to every number below)

1. **"DKT-style" and "SAKT-style"** are simplified models: one skill per sequence (the first opportunity's KC), a small vocabulary, no multi-skill input. They are not reproductions of Piech et al. or Pandey & Karypis. They are reported, not promoted (BKT stays the primary risk score; promotion decided on 2009 validation).
2. **Affect** = outputs of ASSISTments' published affect detectors, **of unverified version**, not human-observed labels and not probabilities. They are rank-transformed (per-column mid-rank ECDF fitted on 2012-train rows). No human BROMP labels exist in this file.
3. **"Unnecessary" is a proxy**: an intervention on a sequence that later reaches mastery within 10 opportunities. It is not a measure of harm or of learner experience.
4. **Grade level is unverified** for both datasets.
5. **This is a replay of logged data, not a classroom.** No intervention was delivered to anyone; "delay" and "unnecessary" are counted against logged outcomes.
6. **Licence confirmation is pending** (see banner).
7. **No learning-gain claim and no clinical claim** is made or implied anywhere in this document.
8. The first_action codes (0/1/2) are inferred, not documented; "verified error" uses first_action == 0. 2012 KCs missing or unseen in 2009 use pooled parameters (BKT/PFA) or the UNK id (DKT/SAKT-style); the share of scored sequences with a KC outside the 2009 vocabulary is in table A1.
9. 2012 sequences are ordered by (start_time, problem_log_id); mastery in 2012 skill builders ends the sequence far more often than in 2009 (DATA_v2.md), so the two datasets differ in structure as well as in population.

Claims are never pooled: **(a)** cross-dataset (2009-trained models, no 2012 data in training) and **(b)** within-dataset affect question (2012-train, chosen on 2012-val).

## Pre-registered endpoints

### (a-P1) AUC of the primary risk score (BKT, 2009-trained), variant A, 95% CI over students

| k | N | positives | AUC [95% CI] | ECE | calibration intercept / slope | Brier |
|---|---|---|---|---|---|---|
| 3 | 31,552 | 5,027 | 0.7450 [0.7378, 0.7523] | 0.051 | -0.65 / 0.73 | 0.125 |
| 5 | 17,448 | 5,027 | 0.7677 [0.7603, 0.7751] | 0.034 | -0.27 / 0.80 | 0.170 |
| 8 | 7,580 | 5,027 | 0.8995 [0.8923, 0.9062] | 0.019 | -0.03 / 0.82 | 0.117 |

Scores are the raw (uncalibrated) 2009-trained BKT risk. A negative intercept means the 2009-trained scores over-estimate risk on 2012 on average, and a slope below 1 means they are too spread out; the frozen 2009 Platt calibrator is applied unchanged in the policy replay.

### (a-P2) and (a-P3) Policy replay on the 2012 test, BKT with the frozen 2009 taus (main configuration)

Scope: 3,815 students, 31,552 sequences, 5,027 wheel-spinning, leavers excluded. Frozen taus (BKT): balanced 0.19, quiet 0.57, supportive 0.09. Bootstrap over students, 1,000 draws, seed 20261025.

| profile | interventions/student BKT [CI] | always-help | reduction vs always-help | missed-help share BKT [CI] | J/student BKT | J/student always-help | J(BKT) - J(always-help) [CI] |
|---|---|---|---|---|---|---|---|
| quiet | 1.873 [1.772, 1.986] | 4.339 | 56.8% | 5.94% [5.23%, 6.63%] | 5.324 | 9.202 | -3.878 [-4.144, -3.616] |
| balanced | 3.604 [3.420, 3.795] | 4.339 | 16.9% | 0.23% [0.10%, 0.38%] | 2.776 | 3.067 | -0.291 [-0.342, -0.245] |
| supportive | 4.162 [3.964, 4.379] | 4.339 | 4.1% | 0.02% [0.00%, 0.06%] | 3.124 | 3.067 | +0.056 [+0.011, +0.103] |

Negative J difference = BKT policy cheaper than always-help under that profile's weights (quiet 3:1, balanced 1:1, supportive 1:3 as w_I:w_D).

### (b-P1) Does affect add information? 2012 test, variant A, mean over k = 3, 5, 8 of AUC(with) - AUC(without)

| variant of the question | mean difference [95% CI] |
|---|---|
| rows <= k (primary) | +0.0022 [+0.0013, +0.0031] |
| rows <= k-1 (lagged sensitivity) | +0.0024 [+0.0015, +0.0034] |

Pre-registered rule: "affect adds information" only if the CI lower bound is above 0 AND the point estimate is at least 0.005. Primary passes: False; lagged passes: False. **Verdict: no evidence affect adds information at the pre-registered threshold.** The lower bounds are above 0, so a small positive difference is detectable, but it is well below the pre-registered 0.005 threshold.

| k | N | AUC without | AUC with (rows <= k) | difference [CI] | AUC with (rows <= k-1) | difference [CI] |
|---|---|---|---|---|---|---|
| 3 | 31,552 | 0.7474 | 0.7528 | +0.0054 [+0.0036, +0.0074] | 0.7536 | +0.0061 [+0.0043, +0.0083] |
| 5 | 17,448 | 0.7806 | 0.7817 | +0.0011 [+0.0001, +0.0019] | 0.7815 | +0.0009 [+0.0000, +0.0017] |
| 8 | 7,580 | 0.8933 | 0.8935 | +0.0001 [-0.0006, +0.0009] | 0.8936 | +0.0003 [-0.0004, +0.0010] |

The gain is concentrated at k = 3 (about +0.005 to +0.006) and is near zero at k = 8. Head logistic model, C = 10 for all (the edge of the pre-registered grid), selected on 2012 validation.

## Exploratory (everything below was not a pre-registered endpoint)

### A1. All risk models on the 2012 test (claim a), variant A: AUC [95% CI] and paired difference to BKT

| k | BKT | PFA | windowed head | DKT-style | SAKT-style | scored with KC outside the 2009 vocabulary |
|---|---|---|---|---|---|---|
| 3 | 0.7450 [0.7378, 0.7523] | 0.7287 (-0.0163 [-0.0201, -0.0125]) | 0.7529 (+0.0079 [+0.0031, +0.0126]) | 0.7393 (-0.0057 [-0.0093, -0.0022]) | 0.7307 (-0.0143 [-0.0182, -0.0103]) | 14.6% |
| 5 | 0.7677 [0.7603, 0.7751] | 0.7564 (-0.0113 [-0.0151, -0.0075]) | 0.7804 (+0.0127 [+0.0081, +0.0171]) | 0.7650 (-0.0027 [-0.0060, +0.0008]) | 0.7440 (-0.0238 [-0.0282, -0.0191]) | 15.5% |
| 8 | 0.8995 [0.8923, 0.9062] | 0.8940 (-0.0055 [-0.0088, -0.0023]) | 0.8954 (-0.0041 [-0.0078, -0.0000]) | 0.8985 (-0.0009 [-0.0035, +0.0017]) | 0.8957 (-0.0038 [-0.0070, -0.0007]) | 17.2% |

DKT-style and SAKT-style are reported, not promoted. Neither beats BKT on 2012, consistent with the 2009 validation decision. **The windowed logistic head scores higher than BKT at k = 3 and 5 on 2012**; BKT was the pre-registered primary on 2009 validation evidence, so this is reported as an exploratory observation, not a promotion. The decision of which score is primary is not revisited.

### A2. Variant B (n > 10) AUC, same models

| k | BKT | PFA | windowed head | DKT-style | SAKT-style |
|---|---|---|---|---|---|
| 3 | 0.7439 | 0.7269 | 0.7499 | 0.7356 | 0.7283 |
| 5 | 0.7644 | 0.7529 | 0.7771 | 0.7598 | 0.7395 |
| 8 | 0.8971 | 0.8906 | 0.8923 | 0.8949 | 0.8924 |

### A3. Policy comparators, main configuration (frozen 2009 values)

| profile | policy | interventions/student | missed-help share | J/student | J(BKT) - J(policy) [CI] (negative = BKT cheaper) |
|---|---|---|---|---|---|
| quiet | n-errors rule (n tuned on 2009 val) | 2.133 | 26.55% | 6.664 | -1.340 [-1.480, -1.206] |
| quiet | three-wrong rule (untuned) | 1.081 | 56.92% | 6.879 | -1.554 [-1.712, -1.412] |
| quiet | PFA policy | 2.176 | 3.32% | 5.430 | -0.106 [-0.191, -0.022] |
| quiet | head policy | 2.178 | 2.18% | 5.486 | -0.161 [-0.254, -0.073] |
| quiet | random, matched to BKT | 1.873 | 36.88% | 7.435 | -2.111 [-2.250, -1.983] |
| balanced | n-errors rule (n tuned on 2009 val) | 4.339 | 0.00% | 3.067 | -0.291 [-0.342, -0.245] |
| balanced | three-wrong rule (untuned) | 1.081 | 56.92% | 5.812 | -3.036 [-3.297, -2.803] |
| balanced | PFA policy | 3.336 | 0.91% | 2.908 | -0.132 [-0.177, -0.088] |
| balanced | head policy | 3.786 | 0.08% | 2.756 | +0.019 [-0.019, +0.056] |
| balanced | random, matched to BKT | 3.604 | 4.24% | 3.459 | -0.683 [-0.742, -0.631] |
| supportive | n-errors rule (n tuned on 2009 val) | 4.339 | 0.00% | 3.067 | +0.056 [+0.011, +0.103] |
| supportive | three-wrong rule (untuned) | 1.081 | 56.92% | 16.371 | -13.247 [-14.188, -12.421] |
| supportive | PFA policy | 3.921 | 0.27% | 3.445 | -0.321 [-0.394, -0.246] |
| supportive | head policy | 3.786 | 0.08% | 3.240 | -0.116 [-0.195, -0.033] |
| supportive | random, matched to BKT | 4.162 | 0.61% | 3.605 | -0.481 [-0.540, -0.428] |

### A4. Other configurations (BKT, carried 2009 taus vs taus re-chosen on 2009 validation for that configuration)

| configuration | scope | quiet: interventions/student | quiet: missed-help | quiet: J(BKT) - J(always-help) [CI] |
|---|---|---|---|---|
| leavers / carried (tau quiet 0.57) | 36,845 sequences | 2.089 | 5.94% | -4.923 [-5.207, -4.630] |
| leavers / rechosen (tau quiet 0.85) | 36,845 sequences | 1.172 | 14.39% | -5.163 [-5.508, -4.819] |
| hintfirst / carried (tau quiet 0.57) | 31,552 sequences | 2.039 | 1.81% | -4.153 [-4.433, -3.869] |
| hintfirst / rechosen (tau quiet 0.49) | 31,552 sequences | 2.324 | 1.01% | -4.013 [-4.268, -3.749] |

### B2. Affect, variant B (n > 10), mean over k of the difference

- rows <= k: +0.0009 [-0.0001, +0.0018]
- rows <= k-1: +0.0010 [+0.0001, +0.0020]

### B3. Affect and the policy (descriptive; tau re-chosen on 2012 validation per model, main configuration)

| profile | model | tau | interventions/student | J/student | J(with) - J(without) [CI] |
|---|---|---|---|---|---|
| quiet | without | 0.58 | 1.850 | 5.2611 | (reference) |
| quiet | with_primary | 0.57 | 1.857 | 5.2619 | +0.0008 [-0.0249, +0.0275] |
| quiet | with_lagged | 0.56 | 1.895 | 5.2815 | +0.0204 [-0.0079, +0.0490] |
| balanced | without | 0.25 | 3.786 | 2.7565 | (reference) |
| balanced | with_primary | 0.17 | 3.788 | 2.7565 | +0.0000 [-0.0031, +0.0029] |
| balanced | with_lagged | 0.22 | 3.786 | 2.7565 | +0.0000 [+0.0000, +0.0000] |
| supportive | without | 0.14 | 4.339 | 3.0674 | (reference) |
| supportive | with_primary | 0.12 | 4.339 | 3.0674 | +0.0000 [+0.0000, +0.0000] |
| supportive | with_lagged | 0.12 | 4.339 | 3.0674 | +0.0000 [+0.0000, +0.0000] |

Every interval for the policy cost difference spans zero or is exactly zero (identical decisions), so affect changes no policy outcome measurably here.

## Reproduction

`python -m aria_pace.run_test2012` refuses to run again (TEST_LOCK_2012). Figures for the summit slide are produced in C6 from `results/test_results_2012.json`.
