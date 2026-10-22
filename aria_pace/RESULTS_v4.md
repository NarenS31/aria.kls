# RESULTS v4 (branch `aria-pace-v4`)

**Nothing here changes a published number, a frozen file, the tagged demo (`summit-2026-10-24-pace`) or `eval/`. No locked test split was read (2009 test students are dropped right after the split; the 2012 file is never loaded) and no new test evaluation was made.** A7, A8 and A9 are numbered amendments in `aria_pace/SPEC.md`, written and committed before their code and results (SPEC commit `1179a31`, code `0951ab7`). All three are **exploratory, validation only**. The 2009 data are ASSISTments skill-builder math logs, grade level not stated.

## Step 1. Stress test of the demo build (read-only): targets met
50,000 simulated turns through the real `DemoSession`, 30 problems x 3 profiles, fuzzed input (`docs/aria_pace/STRESS_50K.md`, `scripts/stress_50k.py`, `results/stress_50k_summit_pace_mac.json`; demo commit `ce62604` untouched): **0 leaks, 0 math errors, 0 false claims, 0 crashes, 0 silent wrong steps (of 10,746 verified-wrong turns), 0 SymPy fallbacks**; latency p50 13.1 / p95 56.8 / p99 119.3 ms, max 2015 ms, none over 5 s. Caveat: the extreme-number inputs were turned away quickly, so the SymPy time-budget path was not exercised by this run (covered by the existing robustness tests).

## Step 2. A7: cross-validated trade-off per mastery definition (validation only, BKT only)
Five student-level folds over train + validation; per fold, tau and the Platt calibrator are chosen on a separate calibration part and applied to the held-out fold only (so, unlike A4, the numbers below are **not optimistic**). Figures: `docs/aria_pace/figures_v4/cv_tradeoff_{quiet,balanced,supportive}.png`. Pooled held-out result at the per-fold chosen tau (95% student bootstrap):

| Definition | Profile | Saving vs always-help | Missed-help share | Fold taus (range, SD) | Mean shift vs frozen tau | Folds with a material (>0.10) shift vs D3 |
|---|---|---|---|---|---|---|
| D3 | quiet | 50.1% [48.5%, 51.6%] | 6.2% [5.2%, 7.2%] | 0.56, 0.58, 0.56, 0.60, 0.51 (0.09, 0.033) | -0.008 | - |
| D3 | balanced | 15.5% [14.5%, 16.4%] | 0.4% [0.2%, 0.7%] | 0.18, 0.24, 0.23, 0.19, 0.22 (0.06, 0.026) | +0.022 | - |
| D3 | supportive | 1.7% [1.4%, 2.1%] | 0.0% [0.0%, 0.1%] | 0.11, 0.05, 0.04, 0.02, 0.11 (0.09, 0.042) | -0.024 | - |
| D4 | quiet | 8.3% [7.5%, 9.2%] | 0.2% [0.1%, 0.3%] | 0.44, 0.52, 0.43, 0.46, 0.47 (0.09, 0.035) | -0.106 | 3 |
| D4 | balanced | 2.3% [1.9%, 2.8%] | 0.0% [0.0%, 0.1%] | 0.13, 0.41, 0.15, 0.30, 0.33 (0.28, 0.120) | +0.074 | 3 |
| D4 | supportive | 0.2% [0.1%, 0.4%] | 0.0% [0.0%, 0.0%] | 0.10, 0.16, 0.07, 0.11, 0.14 (0.09, 0.035) | +0.026 | 1 |
| D5T15 | quiet | 26.4% [24.7%, 28.2%] | 4.3% [3.4%, 5.3%] | 0.31, 0.41, 0.43, 0.44, 0.43 (0.13, 0.054) | -0.166 | 4 |
| D5T15 | balanced | 4.5% [3.7%, 5.3%] | 0.5% [0.1%, 0.8%] | 0.18, 0.16, 0.07, 0.10, 0.08 (0.11, 0.049) | -0.072 | 2 |
| D5T15 | supportive | 2.1% [1.6%, 2.5%] | 0.2% [0.0%, 0.5%] | 0.10, 0.04, 0.00, 0.07, 0.08 (0.10, 0.039) | -0.032 | 0 |
| D5T20 | quiet | 41.6% [39.2%, 43.8%] | 10.4% [8.1%, 12.7%] | 0.29, 0.25, 0.24, 0.40, 0.31 (0.16, 0.064) | -0.272 | 5 |
| D5T20 | balanced | 13.0% [11.5%, 14.6%] | 1.4% [0.7%, 2.4%] | 0.08, 0.17, 0.00, 0.07, 0.15 (0.17, 0.068) | -0.096 | 2 |
| D5T20 | supportive | 2.1% [1.5%, 2.7%] | 0.2% [0.0%, 0.6%] | 0.04, 0.01, 0.00, 0.01, 0.01 (0.04, 0.015) | -0.076 | 0 |
| DEMA | quiet | 7.1% [6.2%, 8.0%] | 0.3% [0.2%, 0.5%] | 0.63, 0.75, 0.52, 0.77, 0.69 (0.25, 0.101) | +0.102 | 3 |
| DEMA | balanced | 2.7% [2.2%, 3.2%] | 0.0% [0.0%, 0.0%] | 0.53, 0.50, 0.22, 0.44, 0.42 (0.31, 0.121) | +0.232 | 4 |
| DEMA | supportive | 0.2% [0.1%, 0.4%] | 0.0% [0.0%, 0.0%] | 0.10, 0.16, 0.12, 0.21, 0.16 (0.11, 0.042) | +0.060 | 2 |

- **Under the frozen definition (D3) the cross-validated Quiet result is a saving of 50.1% with 6.2% missed help** (A4's optimistic single-split figure was 49.8% and 4.8%): the saving holds, the missed-help share is higher once tau is not chosen on the evaluated students.
- **The saving is fragile to the definition**, as in A4, now without the selection optimism: Quiet saves D3 50.1%, D4 8.3%, D5T15 26.4%, D5T20 41.6%, DEMA 7.1%.
- **How much tau moves:** within D3 the five fold taus stay within 0.09 for Quiet (0.51-0.60) and shift little from the frozen 0.57; under other definitions the chosen tau moves by more than 0.10 in several folds (table, last column), most for D5T20 (Quiet mean shift -0.27) and the EMA rule. So a threshold tuned for 3-in-a-row would not carry over to another definition.
- **Limitation:** BKT only. PFA and the risk head were not re-run (they need a definition-specific rewrite of their scoring; not cheap; A7.5).

## Step 3. A8: how much do BKT parameters drift across skills? (validation only)
**This estimates transfer risk across ASSISTments skills. It is NOT a calculus calibration.** Per-skill BKT fitted on 2009 train only: 125 skills have enough sequences (covering 99.9% of train sequences). Pooled: guess 0.21, learn 0.09, prior 0.82, slip 0.20.

| Parameter | Median | IQR (25th-75th) | Min - max | Pooled | Skills at lower / upper bound |
|---|---|---|---|---|---|
| guess | 0.25 | 0.13-0.30 | 0.01-0.30 | 0.21 | 4 / 45 |
| learn | 0.15 | 0.08-0.27 | 0.01-0.60 | 0.09 | 2 / 10 |
| prior | 0.77 | 0.61-0.86 | 0.01-0.99 | 0.82 | 3 / 5 |
| slip | 0.19 | 0.11-0.26 | 0.01-0.30 | 0.20 | 3 / 15 |

Groups by skill name (rule fixed in A8.3; names are a rough label): algebra-type: 15 skills; data/probability: 32 skills; geometry/measurement: 21 skills; number/arithmetic: 30 skills; other: 27 skills. Group medians differ (e.g. learning rate algebra-type 0.20, data/probability 0.13, geometry/measurement 0.17, number/arithmetic 0.14, other 0.17).

Decisions on validation (3,631 cells with a verified wrong step), frozen Platt and tau, **own-skill parameters (the frozen research pipeline) versus pooled parameters (what the demo does)**:

| Profile | Cells that flip | Reduction: own-skill | Reduction: pooled | Missed help: own-skill | Missed help: pooled |
|---|---|---|---|---|---|
| balanced | 5.9% (215) | 13.0% [11.3%, 14.8%] | 6.9% [5.6%, 8.4%] | 0.0% | 0.0% |
| quiet | 14.4% (523) | 49.8% [46.5%, 53.0%] | 43.9% [40.6%, 47.0%] | 4.8% | 3.4% |
| supportive | 1.6% (58) | 2.7% [2.0%, 3.4%] | 0.0% [0.0%, 0.0%] | 0.0% | 0.0% |

- Pooled parameters make the policy **speak more**: in the Quiet profile 369 cells switch from flag to speak and 154 the other way; in the Supportive profile the pooled version is exactly always-help (0.0% saving). Quiet flips by group: algebra-type 7.1%, data/probability 17.9%, geometry/measurement 13.0%, number/arithmetic 15.0%, other 21.2%.
- Reading: parameters differ widely between skills (many sit at the fitting bounds), and applying one pooled set costs about 6 points of Quiet saving on these skills while lowering measured missed help, so the demo's numbers are not the research pipeline's. How this transfers to calculus is **not estimated** here.

## Step 4. Calculus data availability (research, nothing downloaded)
`docs/aria_pace/CALCULUS_DATA.md`. **Public student-level step logs for calculus: not found.** DataShop's catalog could not be browsed by the fetcher (HTTP 404), so a calculus dataset there is not ruled out; account requirements for public data are described inconsistently by two DataShop pages; several other sources appeared only in search results and are marked unverified. Verified: the MATH benchmark (12,500 competition problems with step-by-step solutions) is a problem/solution dataset with no student attempts or logs.

## Step 5. A9: per-turn model with response time (validation only)
`ms_first_response` exists in the raw file but `aria_pace/loader.py:12` does not load it; usable after a rank transform fitted on train. Per-turn logistic model (BKT risk, wrong in last 3, cumulative hints, rank-transformed log response time), k = 1..9:

| k | In frozen window | n val | AUC BKT | AUC model | Diff vs BKT [95% CI] | Diff without response time | Response time only (AUC) |
|---|---|---|---|---|---|---|---|
| 1 | no | 5266 | 0.734 | 0.745 | +0.0103 [+0.0003, +0.0201] | +0.0094 [+0.0031, +0.0157] | 0.533 |
| 2 | no | 5266 | 0.813 | 0.820 | +0.0071 [+0.0010, +0.0128] | +0.0080 [+0.0026, +0.0134] | 0.513 |
| 3 | yes | 3742 | 0.813 | 0.820 | +0.0072 [+0.0019, +0.0124] | +0.0062 [+0.0023, +0.0102] | 0.545 |
| 4 | yes | 2414 | 0.798 | 0.804 | +0.0065 [+0.0019, +0.0114] | +0.0039 [-0.0009, +0.0085] | 0.554 |
| 5 | yes | 1769 | 0.805 | 0.809 | +0.0034 [-0.0012, +0.0081] | +0.0030 [-0.0002, +0.0066] | 0.550 |
| 6 | yes | 1312 | 0.813 | 0.820 | +0.0070 [+0.0034, +0.0104] | +0.0013 [-0.0010, +0.0033] | 0.590 |
| 7 | yes | 1037 | 0.798 | 0.802 | +0.0040 [-0.0015, +0.0092] | +0.0003 [-0.0022, +0.0028] | 0.589 |
| 8 | yes | 856 | 0.911 | 0.912 | +0.0015 [-0.0002, +0.0030] | +0.0005 [-0.0009, +0.0020] | 0.589 |
| 9 | yes | 737 | 0.978 | 0.979 | +0.0014 [-0.0009, +0.0038] | +0.0007 [-0.0004, +0.0017] | 0.547 |

- **Not promoted** (rule: >= +0.01 with the interval above zero at both k = 3 and k = 5): the gain is +0.0072 at k = 3 and +0.0034 at k = 5. BKT stays. This is a negative result.
- Response time adds little: removing it changes the difference by at most a few thousandths at k = 3-5 (the gain at k = 3 is mostly the error-count and hint features), and response time alone barely beats chance (AUC about 0.5-0.6; negative sign of its coefficient: slower first responses go with lower risk). The only k whose point estimate reaches +0.01 is k = 1 (+0.0103, interval +0.0003 to +0.0201), which is **outside the frozen policy window** (decision points 3-9, A1.2) and so cannot change the policy.
- The live demo already rescored the BKT risk after every verified attempt and applies the gate at attempts 3-9 (`agent/demo_pace.py:204-205`, window from `aria_pace/policy.py` `DEC`, `agent/demo_pace.py:67`); k = 1 and 2 use the old always-speak rule (`wrong_before_window`, `:208`). The demo was not touched.

## Step 6. Teacher activity v2 (design + branch `activity-v2`; not in the demo)
Branch `activity-v2` from `summit-activity`: optional categorical reason when stepping in (off by default; the free-text note is kept in memory only and never written), per-profile and reason analyses, draft amendment in `docs/TEACHER_ACTIVITY.md` and `docs/teacher_timing_analysis.md`. Tests: `tests/test_teacher_activity.py` 33 passed. **Finding (negative):** ARIA's decision on all 8 current moments is identical under Quiet, Balanced and Supportive (the two wrong-step moments are before attempt 3; the rest have no wrong step), so per-profile alignment cannot differ with these moments. Not run with real teachers; any collection needs organizer sign-off and adult-reviewed consent wording first. Not merged into the demo build.

## Step 7. Pilot design fix (docs only)
`docs/aria_pace/PILOT_PROTOCOL.md` section 10 and `docs/aria_pace/LIMITS.md`: the cost of delayed help (w_D) and the cost of interruption (w_I) need separate comparisons; recovery latency can inform but cannot derive the weights, which stay a stated value judgment. Draft; needs ethics review, a school sponsor, parental consent and an adult responsible for the data.

## Step 8. Annotation kappa readiness
`docs/aria_pace/KAPPA_READINESS.md`. **No script computing the main 150-item (seven-category) kappa was found**; `scripts/annotation_kappa.py` (new, outside `eval/`) does it and passes 4 tests on synthetic annotators labelled "code test only" (generated in a temp directory, not saved). The real run needs two independent annotators and the private key (gitignored, local only). Humans are needed; no annotation was created.

## What could not be done or was not verified
- A7 covers BKT only (PFA and the risk head not re-run). A8 and A9 use validation students only and are exploratory; none is a calibration for calculus.
- The stress test's oracle is the demo's own SymPy checker, and the time-budget fallback was not triggered by any input (0 fallbacks).
- DataShop's catalog was not browsed (no account, no download); other dataset claims are unverified search leads.
- `activity-v2` was tested through its handlers and a Gradio build, not in a browser; no Safari check.
- Several earlier items remain open: the website's production deployment of PR #10 was not visible, and `docs/summit/QA.md` / `BRIEF.md` are on other branches.
