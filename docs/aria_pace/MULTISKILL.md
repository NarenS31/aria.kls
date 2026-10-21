# Multi-skill DKT and SAKT (Amendment A5)

**Status: validation only; promotion rule pre-registered (A5.4), everything else exploratory.** Source: `aria_pace/results/multiskill_val.json`, produced by `python -m aria_pace.run_multiskill`. 2009 train and validation only; no locked split was read. Variant A (three correct in a row, T = 10). The models see the student's earlier attempts on **other skills** (most recent 90 rows up to the sequence's k-th opportunity); the target is still the current sequence's wheel-spinning label.

**Pre-registered rule (A5.4):** AUC(model) - AUC(BKT) >= 0.01 AND paired 95% CI lower bound > 0, at BOTH k=3 and k=5 (variant A); k=8 reported, not deciding. **Outcome: no multi-skill model is promoted; BKT stays the primary risk score.**

## Wheel-spinning AUC on validation vs BKT

| Model | k | N | AUC BKT [95% CI] | AUC model [95% CI] | Paired difference [95% CI] | Single-skill version (A3.5 / A3a), same k |
|---|---|---|---|---|---|---|
| multi-skill DKT | 3 | 3742 | 0.8125 [0.7935, 0.8329] | 0.8084 [0.7880, 0.8287] | -0.0042 [-0.0170, +0.0080] | -0.0027 [-0.0097, +0.0042] |
| multi-skill DKT | 5 | 1769 | 0.8055 [0.7836, 0.8278] | 0.7958 [0.7724, 0.8184] | -0.0097 [-0.0218, +0.0026] | -0.0027 [-0.0099, +0.0039] |
| multi-skill DKT | 8 (does not decide) | 856 | 0.9105 [0.8877, 0.9313] | 0.9077 [0.8850, 0.9287] | -0.0029 [-0.0159, +0.0098] | +0.0036 [-0.0030, +0.0104] |
| multi-skill SAKT | 3 | 3742 | 0.8125 [0.7935, 0.8329] | 0.8014 [0.7796, 0.8213] | -0.0111 [-0.0265, +0.0043] | -0.0078 [-0.0152, -0.0001] |
| multi-skill SAKT | 5 | 1769 | 0.8055 [0.7836, 0.8278] | 0.7977 [0.7720, 0.8219] | -0.0077 [-0.0228, +0.0083] | -0.0134 [-0.0224, -0.0047] |
| multi-skill SAKT | 8 (does not decide) | 856 | 0.9105 [0.8877, 0.9313] | 0.9147 [0.8928, 0.9339] | +0.0042 [-0.0078, +0.0157] | +0.0006 [-0.0065, +0.0072] |

Decision (mechanical): ms_dkt: k=3 diff -0.0042 (>= +0.01: False, CI lower bound > 0: False), k=5 diff -0.0097 (>= +0.01: False, CI lower bound > 0: False) -> promote: False; ms_sakt: k=3 diff -0.0111 (>= +0.01: False, CI lower bound > 0: False), k=5 diff -0.0077 (>= +0.01: False, CI lower bound > 0: False) -> promote: False

## Next-response log-loss (matched set: every non-first opportunity of every validation sequence)

| Model | Chosen size | Log-loss model | Log-loss BKT | Difference [95% paired student-bootstrap CI] | Steps / students |
|---|---|---|---|---|---|
| multi-skill DKT | 128 | 0.5470 | 0.5644 | -0.0174 [-0.0218, -0.0133] | 44608 / 784 |
| multi-skill SAKT | 32 | 0.5629 | 0.5644 | -0.0015 [-0.0068, +0.0035] | 44608 / 784 |

Reading: the multi-skill LSTM predicts the **next response** better than BKT (the interval excludes 0), but that does not carry over to the wheel-spinning risk, where every k = 3 and k = 5 difference is below the +0.01 margin and every interval includes 0. Next-response accuracy and wheel-spinning discrimination are different questions.

Approximations (A5.7): the outcome tree assumes the rest of the sequence follows immediately; the models see token order, not timestamps; windows reset state. Selection of the size used validation next-response log-loss only; the label was never used to choose anything.
