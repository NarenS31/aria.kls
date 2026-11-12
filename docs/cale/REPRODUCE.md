# Reproducing the CALE-AI exploratory analyses

Everything here is exploratory and was added after the locked evaluation (`docs/cale/ANALYSIS_PLAN.md`, commit `4b530e5`, written before any analysis ran). Only the ASSISTments 2009-10 skill-builder file is used. The 2009 test students are dropped right after the split in every script, and the test set is never re-run. Nothing under `aria_pace/results/`, `eval/`, `SPEC.md` or `RESULTS*.md` is written.

## One command

```bash
# from the repository root, Python 3.11 with aria_pace/requirements.txt installed
PYTHON=aria_pace/.venv/bin/python scripts/cale/run_all.sh
```

This runs Step 1 (reproduction gate; the script stops if it fails), builds the cross-validation held-out predictions, then E1-E9. Outputs go to `aria_pace/results_cale/` (JSON, `e3_summary.md`, `figures/*.pdf|png`, `run_all.log`). The full run (E1-E9) took **278 s** on an Apple-silicon Mac (macOS 26.4, Python 3.11.18; numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2, the versions pinned in `aria_pace/requirements.txt`). Every JSON records the sha256 of the raw file and of each `aria_pace/results/` file it read, the package versions, the seeds and its runtime (`_provenance`).

Re-run one analysis (after `run_all.sh` or `build_cv.py` has made the cache in `aria_pace/results_cale/_cache/`, which is gitignored):

```bash
cd scripts/cale && ../../aria_pace/.venv/bin/python e3_errors.py     # or step1_reproduce, build_cv, e1_tradeoff, e2_calibration, e4_paired, e5_logged_hints, e6_skills
../../aria_pace/.venv/bin/python -m pytest tests                     # 20 checks on the outputs
```

## Data and preprocessing

- **Download:** `skill_builder_data_corrected_collapsed.csv`, ASSISTments 2009-2010 Skill Builder (corrected, collapsed), from the public link on the ASSISTments dataset page; see `docs/aria_pace/DATA.md` section 5. Save it as `aria_pace/data/raw/skill_builder_data_corrected_collapsed.csv`. The loader refuses any file that is not 64,412,812 bytes with sha256 `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`.
- **Cleaning** (`aria_pace/loader.py:load_raw`, SPEC section 1), counts from `aria_pace/results/data_report.json`, recomputed in `step1_reproduction.json`:
  - 346,860 rows, 4,217 students in the file.
  - **63,755 rows (18.38%) dropped because `skill_id` is missing.** No skill means no knowledge component for BKT, so these rows cannot be scored. 54 students had only such rows, leaving 283,105 rows and 4,163 students.
  - Asserted: `(user_id, order_id)` unique, `correct` in {0, 1}. Sorted by `(user_id, order_id)`.
- **Sequence** = one student's rows in one assignment, ordered by `order_id`; opportunity index recomputed as rank (64,818 sequences). KC = `skill_id` of the first opportunity.
- **Policy scope** (SPEC Amendment 1): `n >= 3`, not mastered by try 3, labelled variant A (unlabelled leavers, `m > 10` and `n < 10`, are dropped). Validation: 3,742 sequences, 596 students, 644 stuck.

## Split

`aria_pace/split.py:make_split`: sorted unique `user_id`, `numpy.random.RandomState(20261015).permutation`, 60/20/20 by student: train 2,498 students (39,201 sequences), validation 833 (13,162), test 832 (12,455). Test students are removed immediately after the split (`scripts/cale/common.py:load_pool`); a unit test checks that no test id is in the CV held-out data.

**Cross-validation folds** (the A7 folds, `aria_pace/cv_tradeoff.py`): the 3,331 train + validation students, `RandomState(20261022)` permutation, five consecutive fifths. For held-out fold `f`, the other four folds are split by student into FIT (3/4) and CAL (1/4) with `RandomState(20261022 + f)`. Pooled in-scope held-out data: 14,640 sequences, 2,349 students, 2,526 stuck (2,279 with a help opportunity).

## BKT and the stuck risk

- Four-parameter BKT per KC (`aria_pace/baselines.py:BKT`, `fit_bkt_em`): EM, 30 iterations, fixed initial values (prior 0.30, learn 0.20, guess 0.15, slip 0.10), bounds prior [0.01, 0.99], learn [0.01, 0.60], guess and slip [0.01, 0.30]. KCs with fewer than `min_seq` = 10 training sequences (chosen in Stage 3 by validation next-response log-loss) use pooled parameters.
- **Pooled parameters fitted on train** (`aria_pace/results/baselines_val.json`): prior 0.8212, learn 0.0894, guess 0.2115, slip 0.2046. 125 of 141 training KCs have their own fit. (In CV each fold refits on its FIT part; fold-0 pooled: prior 0.8205, learn 0.0950, guess 0.2120, slip 0.2075; all folds in `e1_tradeoff.json` `fold_info`.)
- **Stuck risk** at try `k` = P(no three-in-a-row mastery in tries k+1..10 | first k answers), exact dynamic programme over (latent state, current correct run) after forward filtering (`BKT.score`).
- **Platt scaling** (`policy.platt_fit`): logistic regression of the label on logit(risk) over all in-scope decision rows (tries 3-9). Frozen validation calibrator: intercept -0.0860, slope 0.9983 (`frozen_val.json` `platt.bkt`). In CV, one calibrator per fold, fitted on CAL.
- PFA (C = 0.1) and the windowed logistic risk head (C = 10) use the frozen hyperparameters and the same Platt step (`stage5.fit_models`).

## Threshold grid and selection rule

`policy.TAU_GRID` = 0.00, 0.01, ..., 1.00 plus never-fire 1.01 (102 values). Cost `J = w_I * unnecessary + w_D * total delay`, weights (w_I, w_D) Quiet 3:1, Balanced 1:1, Supportive 1:3; tau = argmin J, ties to the larger tau (`stage5.pick`). Frozen BKT thresholds (chosen on validation): Quiet 0.57, Balanced 0.19, Supportive 0.09; tuned n-wrong rule n = 2 (Quiet), 1 (Balanced, Supportive). **Nothing is re-chosen in these analyses**; E1 also reports the tau chosen on each fold's CAL part by this rule (the A7 procedure) and reproduces A7 exactly.

## Metric definitions (SPEC section 7, Amendment 1)

- Decision point `i` in 3..min(9, n), exists iff not yet mastered at `i`. **Verified error** = `correct = 0` and `first_action = 0` (wrong first answer; hint-first rows excluded; `first_action` codes are inferred, not documented).
- A policy fires at the first decision point with a verified error and calibrated risk >= tau; one intervention ("first hint") per sequence at most. Always-help fires at the first verified error at a decision point. The n-wrong rule fires when tries i-n+1..i are all verified errors.
- **Stuck** (`y = 1`): no mastery within 10 tries and `n >= 10`.
- **First hints per student** = intervened sequences / in-scope students. **Reduction vs always-help** = 1 - interventions / always-help interventions.
- **Missed-help rate** = stuck sequences with a help opportunity (a verified error at some decision point) and no firing, / stuck sequences with a help opportunity.
- Student bootstrap: multinomial counts over students, 1,000 draws, percentile 95% intervals; rates are ratios of resampled totals.

## Seeds

Split 20261015; CV folds 20261022 (+f for FIT/CAL); random policy 20261017 (200 draws); bootstraps E1 20261112, E2 20261113, E4 20261114, E5 20261115, E7 20261116, E8 20261117, E9 20261118. All model fitting is deterministic.

## Per analysis

| Analysis | Script | Output | Data |
|---|---|---|---|
| Step 1 | `step1_reproduce.py` | `step1_reproduction.json` | train fit, validation compare |
| CV build | `build_cv.py`, `cv_heldout.py` | `_cache/` (not committed) | A7 folds |
| E1 | `e1_tradeoff.py` | `e1_tradeoff.json`, `figures/e1_tradeoff.*` | CV held-out |
| E2 | `e2_calibration.py` | `e2_calibration.json`, `figures/e2_calibration.*` | validation (+ CV held-out) |
| E3 | `e3_errors.py` | `e3_errors.json`, `e3_summary.md` | CV held-out |
| E4 | `e4_paired.py` | `e4_paired.json` | CV held-out |
| E5 | `e5_logged_hints.py` | `e5_logged_hints.json` | train counts, validation |
| E6 | `e6_skills.py` | `e6_skills.json`, `figures/e6_skills.*` | CV held-out |
| E7 (Addendum 1) | `e7_misses_by_help.py` | `e7_misses_by_help.json` | CV held-out |
| E8 (Addendum 1) | `e8_hintfirst_variant.py` | `e8_hintfirst_variant.json` | CV held-out |
| E9 (Addendum 1) | `e9_riskscore_paired.py` | `e9_riskscore_paired.json` | CV held-out |

Figures: matplotlib, 3.3 in wide (237.6 pt), all text 8 pt, TrueType fonts embedded (`pdf.fonttype 42`), setting colours Quiet #2457A6, Balanced #7A4AA8, Supportive #1F7A58, baselines in Okabe-Ito colours. Quiet and Balanced are hard to tell apart under protan colour vision, so each setting also has its own marker shape and text label.
