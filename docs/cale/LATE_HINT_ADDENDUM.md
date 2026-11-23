# E12 addendum to the late-hint report (CV, post hoc, exploratory)

Script `scripts/cale/e12_late_hint_cis.py`, output `aria_pace/results_cale/e12_late_hint_cis.json`. Same pooled CV held-out data as E11 (2,349 students; 2,279 stuck sequences with a decision point). Read-only on `eval/`, `aria_pace/results/`, the test set, `TEST_LOCK`.

## Flags vs e11

- **All point estimates and counts match e11.** Only the CI end points differ in the last digit (e.g. Quiet early-only missed 29.0–32.7 here vs 29.0–32.8 in e11; PFA 26.6–30.8 vs 26.7–30.7), because e11 used its own seed (20261123) and this addendum uses the paper's seeds.
- **Original paired difference reproduces exactly:** −21.06 pp [−23.05, −18.98] (paper −21.1 [−23.1, −19.0]), identical to `e4_paired.json` (script stops if not).
- Random-matched counts are draw-averaged expectations (200 draws), hence fractional.
- No CI for the late shares in the table beyond what is in the JSON (`late_share_ci95`).

## Seeds (all: 1,000 student multinomial resamples, 95% percentile)

Single-policy CIs: 20261112 (E1, the main CV table). Quiet vs two-errors: 20261114 (E4, the paper's −21.1). BKT vs PFA / logistic: 20261118 (E9, the paper's matched-rate comparison).

## Table (items 1, 2, 3, 6)

Early-only missed = stuck sequences with no first hint at opportunities 3–7 (a hint at 8–9 counts as missed), over N = 2,279. Late share = first hints at opportunity ≥ 8 among stuck sequences that got a hint.

| Policy | Early-only missed / N | Rate [95% CI] | First hints in stuck | Late (i ≥ 8) | Late share |
|---|---|---|---|---|---|
| Quiet | 705 / 2,279 | 30.9% [29.0, 32.7] | 2,137 | 563 | 26.3% |
| Balanced | 120 / 2,279 | 5.3% [4.3, 6.3] | 2,273 | 114 | 5.0% |
| Supportive | 109 / 2,279 | 4.8% [3.8, 5.8] | 2,278 | 108 | 4.7% |
| two-errors (tuned) | 836 / 2,279 | 36.7% [34.6, 38.8] | 1,657 | 214 | 12.9% |
| three-errors | 1,462 / 2,279 | 64.2% [62.2, 66.3] | 967 | 150 | 15.5% |
| random, matched to Quiet | 1,041.7 / 2,279 | 45.7% [44.8, 46.7] | 1,521.8 | 284.5 | 18.7% |
| PFA, matched | 655 / 2,279 | 28.7% [26.6, 30.8] | 2,149 | 525 | 24.4% |
| Logistic risk, matched | 529 / 2,279 | 23.2% [21.3, 25.0] | 2,173 | 423 | 19.5% |

| Paired difference (percentage points of missed help) | Difference | 95% paired CI |
|---|---|---|
| Original (any hint counts): Quiet − two-errors | −21.1 | [−23.1, −19.0] (reproduces paper) |
| **Early-only: Quiet − two-errors** | **−5.7** | [−7.7, −3.7] |
| Early-only: Quiet (BKT) − logistic risk (matched) | **+7.7** | [+6.0, +9.6] (logistic is better) |
| Early-only: Quiet (BKT) − PFA (matched) | **+2.2** | [+0.8, +3.6] (PFA is better) |

Quiet is still better than two-errors early-only, by about a quarter of the original margin. At Quiet's hint rate the logistic score and PFA both beat BKT once late hints stop counting. (In the original definition the BKT − logistic and BKT − PFA gaps are in E9, `e9_riskscore_paired.json`.)

## Item 4: delay D

- **Definition** (`aria_pace/stage5.py:53-62`, same in `aria_pace/policy.py:100-110`): `i_err` = first opportunity in 3..9 with a verified error (wrong first attempt, `first_action == 0`) on a sequence not yet mastered (`stage5.py:56-57`); `i_first` = first opportunity where the policy fires on a verified error (`stage5.py:53-54`). Delay = `i_first − i_err` if a hint was given, else `T_DETERMINE − i_err = 10 − i_err` (`stage5.py:62`). So it is measured **from the first verified error to the first policy hint**, in opportunities, and **from the first error to opportunity 10 if no hint**.
- **Included:** only stuck sequences with a decision point (`y1opp = stuck & has an error at i = 3..9`, `stage5.py:59`); everything else contributes 0. Sequences that master by 10 are not in D.
- **No hint:** not excluded; **capped (censored) at opportunity 10**. These are the missed-help sequences. The "mean delay incl. censored" in e11 is over all 2,279; "hinted only" excludes them. D in the objective J is the sum over the 2,279.
- **Why Quiet has a later median first hint (6 vs 4) but a smaller mean delay (2.13 vs 2.31):** three reasons, all from `delay_decomposition` in the JSON.
  1. The median 6 vs 4 is the **absolute opportunity** of the first hint among hinted sequences only. Delay is relative to the first error (mean first-error opportunity is 3.96 for both). Among hinted sequences Quiet's mean delay is 1.87 vs 1.23 for two-errors, so Quiet is slower where it does hint, consistent with the later median.
  2. Two-errors leaves 622 stuck sequences unhinted (27.3%) against 142 for Quiet (6.2%), and each unhinted sequence is charged `10 − i_err` (mean 5.2 for two-errors, 6.1 for Quiet).
  3. The totals: two-errors D = 2,035 (hinted) + 3,230 (censored) = 5,265; Quiet D = 3,994 + 870 = 4,864. The censored penalty on two-errors outweighs Quiet's slower hints. The median delay (two-errors 1, Quiet 2) is a different statistic: cumulative share of stuck sequences with delay ≤ 0 / 1 / 2 / 3 is Quiet 35.1 / 48.1 / 59.7 / 71.3% and two-errors 25.4 / 59.4 / 61.2 / 69.9% (recomputed from the same vectors, not stored in the JSON). Two-errors puts much of its mass at delay exactly 1 (it fires at the second consecutive error), so its median is 1; Quiet crosses 50% only at delay 2. The long censored tail (delay up to 7 for the 27% never hinted) raises two-errors' mean, not its median.
  Note the medians (6 vs 4) are conditional on being hinted and use a different population from the mean (all stuck), so the two statistics need not agree in ordering.

## Item 5: I(τ)

`unnec = fired & ~y1` (`stage5.py:64`; `policy.py:106`): a sequence counts once if the policy gave its (single, first) hint at some opportunity 3–9 **and the sequence is labelled not stuck, i.e. the student masters it by opportunity 10** (`y == 0`; `labels.py:17-22`). Only labelled in-scope sequences are present (unlabelled leavers excluded in CV). A hint on a stuck sequence is never unnecessary. I(τ) is the count divided by the number of in-scope held-out students (2,349), summed over sequences per student; it is not a rate per hint. Note a hint can fire only at a verified error, so an unnecessary interruption is a hint on an error in a sequence that would have mastered anyway, including hints after which mastery was already imminent.
