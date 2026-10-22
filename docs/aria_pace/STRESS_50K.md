# Stress test of the summit demo build: 50,000 simulated turns (read-only)

Date: 2026-10-22. Build tested: branch `summit-pace-mac`, commit `ce62604` (checked out in a separate git worktree and imported; **nothing in it was changed**, no bytecode or log was written there, and `git status` of that worktree was clean afterwards). The tagged demo build `summit-2026-10-24-pace` was not touched. Script: `scripts/stress_50k.py`. Raw output: `results/stress_50k_summit_pace_mac.json`.

## Result: targets met (0 leaks, 0 math errors, 0 crashes, p95 < 5 s)

| Measure | Result |
|---|---|
| Turns run through the real `DemoSession` | **50,000** (30 problems x 3 profiles; 16,667 / 16,667 / 16,666 per profile) |
| Crashes (exception escaping `turn()`) | **0** |
| Answer leaks (`reveals_answer` on every reply) | **0** |
| False math claims in a reply (`false_claims`) | **0** |
| Math errors (reply approves a turn SymPy does not rate correct, or rejects a correct one) | **0** |
| Silent wrong steps (verified wrong, empty reply), of 10,746 verified-wrong turns | **0** |
| SymPy fallback or timeout (`meta["fallback"]` set) | **0** of 50,000; abandoned math threads at the end of each worker: [0, 0, 0, 0] |
| Latency p50 / p95 / p99 / max (ms) | **13.1 / 56.8 / 119.3 / 2015** (mean 20.0) |
| Turns slower than 5 s | **0** |

## What was run

Sessions of 3-12 turns (45% from scripted sequences such as stuck -> wrong x k -> step -> correct, same wrong answer repeated up to 9 times, the rest random mixes), one `DemoSession` per session so the BKT tracker and the pace gate see several verdicts, each with one of the three profiles. Turns by category: ask 2103, control 1667, correct 3784, dup 5730, empty 1671, extreme 1620, frustrated 1926, idk 2619, latex 2135, latexwrap 1631, long 1640, malformed 1638, parens 1638, question 1633, restate 1890, step 3444, unicode 1645, unsimplified 1608, whitespace 1620, wrong 8358.

- **Fuzzed input:** malformed algebra (random deletions, inserted operators, swapped characters), unmatched parentheses, extreme numbers and exponents (`10**1000`, `x**99999999`, `2**(2**64)`, `factorial(100000)`, 300-5,000 digit numbers, `e**(e**(e**100))`, ...), unicode look-alikes (minus sign, multiplication sign, Cyrillic x, full-width and Arabic-Indic digits, emoji, right-to-left marks), LaTeX wrappers, empty and whitespace-only input, control characters, a byte-order mark, very long input (up to 100,000 characters), and rapid duplicate submissions (the same text three times in a row).
- **Checks:** the same definitions as `scripts/demo_harness.py` (leak: `agent.math_check.reveals_answer`; false claim: `false_claims`; verdict check: the harness's `verdict_in_text` rule). The SymPy oracle for "is this turn correct" is `check_answer` on the **sanitized** text the session itself sees (`agent/demo_tutor.py:177`, `agent/demo_input.py:32`); for the extreme and very long inputs the session's own verdict stands in, so the oracle cannot hang the run.

## Latency by input category (ms)

| Category | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| ask | 2103 | 13.2 | 36.6 | 59.1 | 143 |
| control | 1667 | 0.4 | 55.5 | 102.8 | 278 |
| correct | 3784 | 17.4 | 53.4 | 99.4 | 515 |
| dup | 5730 | 14.3 | 57.6 | 128.1 | 1149 |
| empty | 1671 | 0.1 | 0.2 | 0.6 | 5 |
| extreme | 1620 | 0.5 | 58.3 | 83.6 | 172 |
| frustrated | 1926 | 12.8 | 34.1 | 59.1 | 385 |
| idk | 2619 | 13.3 | 38.0 | 69.4 | 963 |
| latex | 2135 | 17.7 | 55.0 | 93.1 | 363 |
| latexwrap | 1631 | 12.6 | 45.7 | 80.2 | 187 |
| long | 1640 | 8.7 | 610.3 | 918.3 | 1385 |
| malformed | 1638 | 6.7 | 55.2 | 110.0 | 257 |
| parens | 1638 | 10.0 | 43.5 | 86.8 | 464 |
| question | 1633 | 13.4 | 35.2 | 60.6 | 196 |
| restate | 1890 | 1.8 | 4.9 | 10.0 | 39 |
| step | 3444 | 4.1 | 21.7 | 37.0 | 847 |
| unicode | 1645 | 14.8 | 51.7 | 87.1 | 179 |
| unsimplified | 1608 | 22.4 | 59.9 | 108.5 | 1626 |
| whitespace | 1620 | 0.1 | 0.2 | 0.6 | 4 |
| wrong | 8358 | 20.0 | 81.1 | 133.1 | 2015 |

The slow tail is the very long inputs (p95 about 610 ms) and a few wrong or unsimplified answers (max about 2.0 s); nothing came near 5 s.

## Caveats, stated plainly

- **Same oracle as the tutor.** The correctness oracle is the demo's own SymPy checker, so this test shows the replies are consistent with the verdicts and that nothing crashes or leaks; it cannot show that SymPy itself is right. The bank's truths are separately re-derived by SymPy (`docs/benchmark/README.md`).
- **The extreme-number inputs did not exercise the time budget.** They were turned away quickly (median 0.5 ms, p95 58 ms, with an empty reply), so the SymPy time-budget path (`MATH_BUDGET_S = 3.0`, `agent/demo_tutor.py:49`) and the "math busy" lock-out were **not** triggered: 0 fallbacks in 50,000 turns. That path is covered by the existing robustness tests, not by this run. Those inputs get **no reply at all** (a student who types `10**1000` hears nothing), which the stress test counts as acceptable because the input has no verdict, not as a wrong step.
- **Timing was measured on a busy laptop** (4 worker processes; I was running other analyses at the same time), so the latencies are, if anything, pessimistic. They are not Wi-Fi-off or Safari numbers.
- **Templates only.** The optional LLM path was off, as in the demo.
- **Simulated students, not real ones.** Sequences are scripted and random mixes; they say nothing about how real students type. The inputs are synthetic; no student text was used.
- An earlier two attempts of the full run were discarded: they overlapped with orphaned worker processes from a killed run (which inflates latency); only the final clean run is reported. A 3,000-turn smoke run first found two issues in my own oracle (it checked raw rather than sanitized text), fixed before the full run.

## Reproduce

```
git worktree add --detach /tmp/summit-pace-mac origin/summit-pace-mac
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/stress_50k.py --repo /tmp/summit-pace-mac --turns 50000 --workers 4 --out stress_50k.json
```
