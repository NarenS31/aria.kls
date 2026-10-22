# Benchmark: a symbolically verified bank of 30 problems

**30 problems, algebra / trigonometry / calculus** (10 each), each with an answer that SymPy re-derives when the bank loads, plus a harness that plays scripted student messages against a tutor and measures what a tutor must never do wrong. It lives in this repository (`data/demo_problems.json`, `agent/demo_bank.py`, `scripts/demo_harness.py`); there is no separate repository or organisation.

**What it is for:** checking a math tutor's replies (answer leaks, false math claims, crashes, latency, silence on wrong steps) on typed input with known correct answers. **What it is not:** a measure of learning, of student behaviour, or of how helpful a reply is. The inputs are scripted, not students.

## The bank

`data/demo_problems.json` is a JSON list. Per problem: the prompt, the authored answer (or equation), terms and steps for partial credit, authored hint moves, scripted student messages for the harness, and (added on branch `learning-graph`) hand-authored concept tags and prerequisite edges used only by an exploratory Teacher-summary view. 20 problems are expressions (differentiate, integrate, expand, simplify, evaluate) and 10 are equations to solve. Levels: 8 intro, 17 core, 5 advanced (counted from the file). 8 problems are marked `featured` and are the ones shown in the live demo.

| id | topic | level | featured | kind | prompt |
|---|---|---|---|---|---|
| `alg-01` | algebra | intro | yes | solve | Solve for x: 3(x - 4) = 2x + 5 |
| `alg-02` | algebra | intro |  | solve | Solve for x: 2x + 7 = 15 |
| `alg-03` | algebra | intro |  | solve | Solve for x: 5x - 3 = 2x + 9 |
| `alg-04` | algebra | core |  | solve | Solve for x: (x + 1)/2 = 7 |
| `alg-05` | algebra | core | yes | solve | Solve for x: x^2 - 5x + 6 = 0 |
| `alg-06` | algebra | core |  | solve | Solve for x: x^2 - 9 = 0 |
| `alg-07` | algebra | advanced |  | solve | Solve for x: 2x^2 + 3x - 2 = 0 |
| `alg-08` | algebra | core |  | expr | Expand (x + 3)^2 |
| `alg-09` | algebra | core |  | expr | Expand (2x - 1)(x + 4) |
| `alg-10` | algebra | core |  | expr | Simplify (x^2 - 4)/(x - 2) |
| `trig-01` | trig | intro | yes | expr | Evaluate sin(pi/6) + cos(pi/3) |
| `trig-02` | trig | intro |  | expr | Evaluate tan(pi/4) * cos(pi/3) |
| `trig-03` | trig | core |  | expr | Evaluate cos(pi/4)^2 |
| `trig-04` | trig | core | yes | expr | Simplify sin(x)cos(x)/(1 - sin(x)^2) |
| `trig-05` | trig | core |  | expr | Simplify (1 - cos(x)^2)/sin(x) |
| `trig-06` | trig | core |  | expr | Simplify sin(2x)/(2 sin(x)) |
| `trig-07` | trig | intro |  | expr | Simplify tan(x) cos(x) |
| `trig-08` | trig | core |  | solve | Solve sin(x) = 1/2 for 0 <= x <= pi/2 |
| `trig-09` | trig | core |  | solve | Solve 2cos(x) = 1 for 0 <= x <= pi/2 |
| `trig-10` | trig | core |  | solve | Solve tan(x) = 1 for 0 <= x <= pi/2 |
| `calc-01` | calculus | core | yes | expr | Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9 |
| `calc-02` | calculus | advanced | yes | expr | Differentiate f(x) = x^2 cos(2x) |
| `calc-03` | calculus | advanced |  | expr | Differentiate f(x) = x^2 sin(3x) |
| `calc-04` | calculus | core |  | expr | Differentiate f(x) = e^(2x) |
| `calc-05` | calculus | core |  | expr | Differentiate f(x) = ln(x^2 + 1) |
| `calc-06` | calculus | advanced | yes | expr | Differentiate f(x) = (x^2 + 1)^3 |
| `calc-07` | calculus | core | yes | expr | Find the integral of 6x^2 - 4x + 1 dx |
| `calc-08` | calculus | intro |  | expr | Find the integral of cos(x) dx |
| `calc-09` | calculus | intro |  | expr | Find the integral of x^3 dx |
| `calc-10` | calculus | advanced |  | expr | Differentiate f(x) = sin(x) cos(x) |

## How answers are re-derived with SymPy

`load_problems()` builds a SymPy ground truth for every problem and checks it **independently of the authored answer**, so a typo in the JSON stops the program at load time (`agent/demo_bank.py:1-12`, `:190-203`).

- **Expression problems (20):** each carries a `check` with an operation and the source expression; SymPy computes `diff`, `integrate`, `expand`, `simplify` or a numeric `value` of it and the authored answer must be equivalent to that result (`agent/demo_bank.py:74-87`, comparison `:102-111`). Antiderivative answers are equivalent up to a constant (`mod_const`).
- **Equation problems (10):** the roots are taken from `sp.solveset` over the reals and must be a finite non-empty set; for the restricted-domain trig problems the authored roots are accepted only if they satisfy the equation to 1e-12 (`agent/math_check.py:419-430`).
- **Every problem:** the scripted correct, unsimplified, wrong, restated and intermediate-step messages must be classified as expected by the same checker, and each authored hint move must be consistent (`agent/demo_bank.py:152-187`).
- **The 8 featured problems** have a stricter rule: an expression problem must have a `check`, a solve problem must get its roots from `solveset` (authored roots are not accepted), the expression in the prompt must be the one that was checked, and hint moves are required (`agent/demo_bank.py:118-150`).
- **Tests:** `tests/test_demo_bank.py:46-69` and `:82-90` re-derive the 8 featured problems' answers a second time with SymPy; `tests/test_math_check.py:153` loads all 30 and fails if any authored answer disagrees with SymPy. A second, independent re-derivation of the other 22 beyond that load-time check: not found.

## What the harness measures

`scripts/demo_harness.py` plays 13 single-turn inputs per problem (correct answer, common wrong answer, "I don't know", gibberish, "just tell me the answer", unsimplified answer, restated problem, correct intermediate step, a free-form question, a frustrated message, a LaTeX-wrapped answer, empty input, a 2000-character paste; `scripts/demo_harness.py:66-80`). For tutors with the pace gate it also plays a 7-message "same student" flow in one session (`:83`). Per backend it reports:

| Measure | How |
|---|---|
| **Answer leaks** | the symbolic leak check on every reply, using the problem and the student's own text as exemptions (`agent/math_check.py` `reveals_answer`, called at `scripts/demo_harness.py:331`) |
| **Math errors** | a claim in the reply that SymPy refutes (`false_claims`, `:333`), or a verdict that contradicts SymPy (praising a wrong or empty turn, rejecting a correct one) |
| **Crashes** | an exception, or no reply where one is required |
| **Latency** | wall-clock p50, p95 and max per turn; timeouts count at the 10 s cap (`:61`) |
| **Silent wrong steps** | a verified wrong step answered with no reply at all (`:392`); required to be 0 for the ARIA-Pace backends (`:408-412`) |

Targets, checked with `--strict` (exit code 1 if missed): 0 leaks, 0 math errors, 0 crashes, p95 under 5,000 ms (`:60`), and 0 silent wrong steps for the pace backends. The harness also has self-test fakes (a leaker, a liar, a crasher, a mute tutor) so that it can be shown to *see* each failure; they are tested in `tests/test_demo_harness.py`.

## Results

The committed results file is **`results/demo_harness_summit_pace.json`**. It covers the **8 featured problems** (`--featured`), scripted inputs, two builds:

| Build | Commit | Turns | Leaks | Math errors | Crashes | p95 latency | Silent wrong steps |
|---|---|---|---|---|---|---|---|
| Original summit build (`summit-2026-10-24`) | `e09218c` | 104 | 0 | 0 | 0 | 50.8 ms | not measured |
| Pace build, Quiet profile (`summit-2026-10-24-pace`) | `498882c` | 160 | 0 | 0 | 0 | 47.2 ms | 0 |
| Pace build, Balanced profile | `498882c` | 160 | 0 | 0 | 0 | 45.0 ms | 0 |
| Pace build, Supportive profile | `498882c` | 160 | 0 | 0 | 0 | 52.6 ms | 0 |

(`results/demo_harness_summit_pace.json` lines 22-29, 51-58, 79-86, 107-114 and the silent-wrong lines 48, 76, 104.)

**Not found:** a committed results file for a template-only run over **all 30** problems. The only full-bank numbers in the repository are older ones for the optional language-model path (1,155 turns, 0 leaks, 0 math errors, 0 crashes, on the 2026-10-15 build, before the pace gate): `docs/DEMO_RUNBOOK.md` lines 66-76, flagged there as not re-run. To produce the 30-problem numbers, run the command below without `--featured`.

## Run it

```bash
cd <repo root>
/usr/bin/python3 scripts/demo_harness.py --backend demo --profiles all --featured --strict --json out.json   # the 8 featured problems
/usr/bin/python3 scripts/demo_harness.py --backend demo --profiles all --strict --json out30.json            # all 30
/usr/bin/python3 -m pytest tests/test_demo_bank.py -q                                                        # the bank's own tests
```

`/usr/bin/python3` is the interpreter that has SymPy; other interpreters on the build machine do not (`docs/DEMO_RUNBOOK.md` section 1). Run each test file separately: importing `eval` first changes how `ui` resolves.

## Limits

Scripted inputs are not students. The bank is small, hand-authored and typed-math only (no handwriting or photos). The checker decides correctness of the final expression or equation, not the quality of reasoning. Passing the harness shows the tutor did not leak, mis-state or crash on these inputs; it says nothing about whether the replies help anyone.
