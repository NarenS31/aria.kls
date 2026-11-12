# E3 summary: missed vs caught stuck sequences (exploratory, CV held-out)

Operating point: BKT, frozen Quiet tau 0.57, per-fold CAL Platt (pooled CV held-out). 142 missed and 2137 caught of 2279 stuck sequences with a help opportunity (missed-help rate 6.2%).

| Feature | Missed | Caught |
|---|---|---|
| Sequence length n | 13.00 (IQR 11.00-19.00; mean 17.29) | 14.00 (IQR 12.00-20.00; mean 17.71) |
| Wrong answers, tries 1-9 | 5.00 (IQR 4.00-7.00; mean 5.55) | 6.00 (IQR 5.00-8.00; mean 6.35) |
| Verified errors at decision points 3-9 | 2.00 (IQR 1.00-2.00; mean 1.75) | 4.00 (IQR 3.00-5.00; mean 3.61) |
| Wrong but not a verified error (hint-first/other code), tries 1-9 | 3.00 (IQR 1.00-5.00; mean 3.04) | 1.00 (IQR 0.00-3.00; mean 1.62) |
| Longest wrong run, tries 1-9 | 3.00 (IQR 2.00-4.75; mean 3.52) | 3.00 (IQR 2.00-6.00; mean 4.18) |
| Right/wrong switches, tries 1-9 | 5.00 (IQR 3.00-6.00; mean 4.20) | 4.00 (IQR 2.00-5.00; mean 3.63) |
| Has 2+ consecutive wrongs | 82.4% | 93.9% |
| Correct answers before first error | 0.00 (IQR 0.00-1.00; mean 0.46) | 0.00 (IQR 0.00-0.00; mean 0.22) |
| Max calibrated risk at a verified error | 0.43 (IQR 0.35-0.49; mean 0.41) | 1.00 (IQR 1.00-1.00; mean 0.97) |

Five most common patterns among misses (tries 1-9, 1 = correct):

| Pattern | Missed | Caught with same pattern |
|---|---|---|
| `000000000` | 15 (10.6%) | 239 |
| `011011011` | 6 (4.2%) | 2 |
| `101010000` | 3 (2.1%) | 3 |
| `010101101` | 3 (2.1%) | 10 |
| `010101011` | 2 (1.4%) | 1 |

Test misses: NOT DONE: the locked test run saved only aggregates (aria_pace/results/test_results.json); no per-sequence test outputs exist on disk, and the test set was not re-run.

Descriptive only; replay on logs; no causal reading.
