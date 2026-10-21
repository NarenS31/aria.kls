# Synthetic calculus trajectories (behavioural stress test)

**Behavioural stress test, not transfer evidence.** Rule-based synthetic students (no LLM) run through the demo tutor with the frozen ARIA-Pace gate, to check that it behaves sensibly. It says nothing about real students, learning, or transfer. Docs only: no tutor, UI or frozen file was changed.

- **Code under test:** the pace build of the demo tutor, branch `summit-pace-mac` (commit `945516e`, `agent/demo_tutor.py` with `agent/demo_pace.py`), used read-only from a separate checkout. The generator and runner are `scripts/synthetic_calculus_stress.py` (this branch); raw output is `docs/aria_pace/synthetic_calculus_stress.json`.
- **Students:** 40 per (type, problem), 10 calculus problems from the demo bank (calc-01, calc-02, calc-03, calc-04, calc-05, calc-06, calc-07, calc-08, calc-09, calc-10), 3 types, 3 profiles = 3,600 trajectories, 20,199 tutor turns. Seed 20261102.
- **Types (all rule-based):** fluent (P(wrong) 0.10, P(recover after a wrong step) 0.80), intermittent (0.35 / 0.50), stuck-prone (0.60 / 0.20). After two wrong steps in a row a student asks for help with probability 0.3. The first two right attempts type the problem's authored correct intermediate step when it has one; the next right attempt types the correct answer and ends the trajectory; 14 turns at most. The students do **not** react to what the tutor says (open loop), and the bank offers few distinct messages per problem, so the trajectories are coarse.
- **Acceptance criteria, fixed before the run:** 0 crashes, 0 answer leaks, 0 math errors, 0 silent verified-wrong steps, p95 turn latency < 5 s, and mean interventions per trajectory non-decreasing from Quiet to Balanced to Supportive and from fluent to intermittent to stuck-prone.

## Result

| Profile | Student type | Trajectories | Turns / trajectory | Tutor interventions / trajectory | Verified wrong steps | of which flagged without a hint | Silent wrong steps | Leaks | Math errors | Crashes | p95 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quiet | fluent | 400 | 2.96 | 1.20 | 133 | 61 | 0 | 0 | 0 | 0 | 37 ms |
| quiet | intermittent | 400 | 4.42 | 2.30 | 642 | 212 | 0 | 0 | 0 | 0 | 39 ms |
| quiet | stuck-prone | 400 | 9.45 | 7.40 | 2268 | 285 | 0 | 0 | 0 | 0 | 43 ms |
| balanced | fluent | 400 | 2.96 | 1.28 | 133 | 30 | 0 | 0 | 0 | 0 | 37 ms |
| balanced | intermittent | 400 | 4.42 | 2.71 | 642 | 47 | 0 | 0 | 0 | 0 | 44 ms |
| balanced | stuck-prone | 400 | 9.45 | 8.06 | 2268 | 20 | 0 | 0 | 0 | 0 | 49 ms |
| supportive | fluent | 400 | 2.96 | 1.35 | 133 | 0 | 0 | 0 | 0 | 0 | 40 ms |
| supportive | intermittent | 400 | 4.42 | 2.83 | 642 | 0 | 0 | 0 | 0 | 0 | 51 ms |
| supportive | stuck-prone | 400 | 9.45 | 8.12 | 2268 | 0 | 0 | 0 | 0 | 0 | 45 ms |

Criteria: `crashes==0` = True; `degraded_turns` = 0; `interventions_nondecreasing_fluent_intermittent_stuck` = True; `interventions_nondecreasing_quiet_balanced_supportive` = True; `leaks==0` = True; `math_errors==0` = True; `p95<5s` = True; `silent_wrong==0` = True.

What this shows, and only this: across about 20,199 tutor turns there were no crashes, no answer leaks, no false math claims and no verified wrong step left without a reply; the number of interventions rises from the fluent to the stuck-prone student and from the Quiet to the Supportive profile; the flagged-without-a-hint replies appear only where a profile's bar is not met (the Supportive profile never flags). Interventions here include the existing rules for help requests and for the reflection after a correct answer, not only the pace gate.

What it does not show: that the trajectories resemble real calculus students (they are scripts), that hints help, or anything about learning. The Quiet and Balanced profiles flag a wrong step instead of hinting only at attempts 3 to 9, as designed (`docs/DEMO_RUNBOOK.md` section 6 on `summit-pace-mac`).
