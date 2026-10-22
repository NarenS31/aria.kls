# Pilot measures the current system can really compute

**Status: draft, not approved. Nothing here has been collected from students.** These are definitions of four measures that can be computed from the demo's existing per-turn log, and the script that computes them: `scripts/pilot_measures.py`, tested on a synthetic log (`tests/test_pilot_measures.py`). No number from real students exists.

## What the log contains

One line per tutor turn in `data/demo_logs/turns.jsonl` (written unless the app is started with `--no-log`). Only whitelisted fields are written, numbers and short identifiers only; **no student text and no tutor text** (`agent/demo_log.py:19-27`, `_clean` at `:30-42`). The fields used here:

| Field | Meaning | Where it is set |
|---|---|---|
| `session_id`, `attempt`, `problem_id`, `turn` | one random session id per problem session, the attempt number (Reset increments it), the problem, the turn number | `agent/demo_tutor.py` `DemoSession`, `agent/demo_log.py:21` |
| `correctness` | SymPy verdict: `correct`, `partial`, `step`, `restate`, `wrong`, `none` | `agent/math_check.py:9` |
| `intervened`, `intervention_type` | whether ARIA replied with a hint-type reply, and its type (`hint`, `task_chunking`, `encouragement`, `break_suggestion`, `reflection`) | `agent/demo_log.py:23` |
| `hint_level` | value of a counter after the turn: starts at 0 per attempt (`agent/demo_tutor.py:174`), is incremented by every reply that is not a reflection (`:286`), and selects the next authored step the student has not produced (`:299`) | `agent/demo_tutor.py:330` |
| `t_s` | seconds since the attempt started, at the end of the turn, one decimal | `agent/demo_tutor.py:171`, `:332` |

An **attempt** below means one `(session_id, attempt, problem_id)` group of turns. **Scored** turn = a turn whose verdict is right (`correct`, `partial`, `step`) or `wrong`; `restate` and `none` are not scored (the same rule the pace tracker uses, `agent/demo_pace.py:38-39`). A **hint turn** = `intervened` with `intervention_type` in (`hint`, `task_chunking`) (`agent/demo_summary.py:34`).

The log has **no learner identifier**: `session_id` changes whenever the visitor picks another problem, so measures cannot follow one learner across problems from the log alone.

## The four measures

**(a) Correction after a hint.** For each hint turn, find the next scored turn in the same attempt. Report: number of hints, how many have a later scored turn, how many of those are right, and the share right. A second row uses only the first hint of each attempt. *What it is:* a descriptive count of what the student's next scored answer was. *What it is not:* evidence that the hint caused the correction; there is no comparison group, and a student may have been about to correct anyway.

**(b) Hints per problem, and `hint_level` reached.** Per problem: attempts, hints, hints per attempt, and the mean over attempts of the largest `hint_level` logged. **`hint_level` is not a three-level engine.** It is a pointer into the problem's authored steps that counts every non-reflection reply (including encouragement and a break suggestion); it does not mean a prompt, then a hint, then an explanation. No such escalation exists in the code (`docs/aria_pace/SCAFFOLD_LEVELS.md`, section 1). It is reported because it is logged, not because it measures level.

**(c) Time before the first hint (`t_s`).** `t_s` of the first hint turn of each attempt (median and quartiles), plus how many scored attempts came before it. **This is a proxy** for how long a student worked before being helped, i.e. at best a rough stand-in for productive struggle. It includes reading, typing and thinking time and the tutor's own processing; it cannot tell productive struggle from being lost; and it is not validated against anything. Do not describe it as a measurement of struggle.

**(d) Exploratory friction index.** The share of attempts flagged *high-friction* by the rule of the learning graph (`agent/demo_graph.py`): the attempt is **not secure** (the last three scored attempts are not all right) **and** has at least **2 wrong attempts** and at least **1 hint turn**. Optionally broken down by the concepts the bank tags on the problem (`data/demo_problems.json`, `concepts`). **Exploratory, not validated:** the thresholds (2 and 1) are conventions chosen for the demo, not fitted or checked against any outcome.

## What the script prints

`/usr/bin/python3 scripts/pilot_measures.py` prints a table with these four measures (example on the synthetic test log, 3 attempts and 10 turns, invented data):

```
(a) hints 5 | with a later scored attempt 4 | next attempt right 2 | share right 0.50
(b) attempts with a hint 2 | hints per attempt 1.67 | max hint_level mean 1.67, max 3
(c) attempts with a hint 2 | median 3.0 s | quartiles 2.5 to 3.5 s | scored attempts before it, mean 0
(d) attempts flagged high-friction 2 of 3 (share 0.67)
```

It reads only the log (and optionally the bank for concept labels), writes nothing unless `--json` is given, and exits with a message if the log is missing or empty.

## What these measures cannot show

They describe what happened in a session; they say nothing about learning, retention or transfer, and none was validated against an outcome. A study that wants those needs an unassisted outcome measure, a comparison condition and approval (see `docs/aria_pace/PILOT_PROTOCOL.md`, draft).
