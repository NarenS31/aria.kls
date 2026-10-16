# "Would you step in?" teacher activity: how to run it

Branch `summit-activity` (from the tag `summit-2026-10-24`; the tagged demo build is untouched). Analysis plan, written and committed first: `docs/teacher_timing_analysis.md`.

## What is NOT here yet: the 8 real moments

`data/teacher_moments.json` does not exist. It must be generated from **your** turns; nothing in this repository was invented to stand in for them. Until it exists the tab says
"The teacher activity is not set up on this laptop" and the rest of the demo is unaffected.

1. Write `data/teacher_moments_turns.json`, a list of exactly 8 objects:
   ```json
   [{"id": "m1", "problem_id": "calc-01", "history": ["I don't know", "12x^3"], "message": "12x^3 - 10x - 9"}, ...]
   ```
   `problem_id` is one of the bank ids (the 8 featured ones are listed in `docs/DEMO_RUNBOOK.md` section 6); `history` is the student's earlier messages (may be empty); `message` is the moment.
2. `cd ~/aria && /usr/bin/python3 scripts/make_teacher_moments.py --turns data/teacher_moments_turns.json`
   It replays each moment through the unmodified `DemoTutor` (LLM path off) and records ARIA's decision on `message` as the ground truth, plus `words_only` and `verified_wrong`
   computed from SymPy. It prints one line per moment: **read it before the event**, because it shows exactly what ARIA decides on each one.
3. Check the mix. The analysis plan reports words-only moments separately and treats verified-wrong moments specially (ARIA never stays quiet on a verified wrong step), so a set made only
   of one kind will leave parts of the analysis empty.

## Launch

Same as the demo (`docs/DEMO_RUNBOOK.md` section 1): `cd ~/aria && GRADIO_ANALYTICS_ENABLED=False /usr/bin/python3 -m ui.demo_app`. The third tab, **Would you step in?**, is the activity.
Options: `--moments PATH` and `--taps-csv PATH`. `--no-log` also stops the activity from writing its CSV (the visitor is then told 0 answers were saved).

## Per visitor

Consent screen (Continue / Skip; three optional dropdowns, default "Prefer not to say") → for each of the 8 moments: **Step in** or **Stay quiet** → if Step in, the kind of help
(prompt / hint / encouragement / break / other) → ARIA's decision is revealed with its reason and what it would say → **Agree** / **Disagree** → next moment. At the end: **Start for the next visitor**
gives a fresh random session id. **Skip saves nothing.**

## Data

`data/demo_logs/teacher_taps.csv` (git-ignored). Columns: `session_id, moment_id, role, years_band, grade_band, teacher_decision, teacher_type, aria_decision, aria_type, agree, timestamp`.
Every value comes from a fixed list and the writer rejects anything else, so free text cannot reach the file (tested). **Wipe:** `rm -f data/demo_logs/teacher_taps.csv`.

## Analyse

```
/usr/bin/python3 scripts/analyze_teacher_taps.py data/demo_logs/teacher_taps.csv --moments data/teacher_moments.json --json /tmp/teacher_analysis.json
```

Nothing is reported for a set of moments until every moment in it has at least 15 sessions (it prints `not reported (n < 15 ...)`); subgroups below 15 sessions are suppressed. See the plan for the
exact statistics.

## Tests

`/usr/bin/python3 -m unittest tests.test_teacher_activity` (18 tests: state machine, CSV validation, moment generation against the real tutor, Fleiss' kappa against a hand-computed value,
n >= 15 gating, UI group visibility, HTML escaping). The moments in that test file are **fixtures**, not the real set.

## Checked in a real browser

A fixture set was driven through the tab in Chrome (consent → 8 moments with both decisions and all four help types → Agree/Disagree → done → next visitor → Skip): 8 rows with the exact
columns, no moment text or ARIA reply in the CSV, Skip added no rows. **Not checked:** Safari; the real moments; a real Wi-Fi-off run; more than one visitor at a time.
