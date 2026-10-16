# "Would you step in?" teacher activity: how to run it

Branch `summit-activity` (from the tag `summit-2026-10-24`; the tagged demo build is untouched). Analysis plan, written and committed first: `docs/teacher_timing_analysis.md`.

## The 8 moments

`data/teacher_moments_turns.json` holds the owner's 8 turns verbatim (M1 to M8) and `data/teacher_moments.json` is generated from them by running the unmodified tutor of the tag
`summit-2026-10-24` (e09218c): `/usr/bin/python3 scripts/make_teacher_moments.py --turns data/teacher_moments_turns.json`. Each moment carries a `build` stamp (tag, commit, and whether every file the tutor's
decisions depend on is byte-identical to the tag; a test fails if it is not). ARIA's recorded decision is the ground truth for each moment:

| Moment | Problem | Student's new message | Verdict | ARIA |
|---|---|---|---|---|
| M1 | calc-01 | `12x^3 - 10x` (after "first I'll differentiate term by term") | partial | stays quiet |
| M2 | alg-03 | `3x - 3 = 9` (after "move the 2x over") | correct step | stays quiet |
| M3 | calc-02 | `2x cos(2x) + 2x^2 sin(2x)` (after "product rule") | wrong | steps in: hint |
| M4 | alg-05 | "I think I'll try factoring first" | words only | stays quiet (state UNKNOWN) |
| M5 | trig-01 | "so I need sin of pi over 6 plus cos of pi over 3" | words only | stays quiet |
| M6 | calc-07 | "ugh I hate integrals" | words only | steps in: encouragement |
| M7 | alg-05 | "can you just tell me the answer" | words only | steps in: hint |
| M8 | alg-02 | `x = 11` | wrong | steps in: hint |

Words-only: M4, M5, M6, M7. Verified-wrong (ARIA never stays quiet there): M3, M8. In the ARIA-vs-majority comparison: M1, M2, M4, M5, M6, M7 (six). Only `hint` and `encouragement` occur, so the
kind-of-help question is descriptive this round (see Amendment 2 in the analysis plan).

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

`/usr/bin/python3 -m unittest tests.test_teacher_activity` (23 tests: state machine, CSV validation, moment generation, a golden snapshot of the 8 real decisions, Fleiss' kappa against a hand-computed value,
n >= 15 gating, UI group visibility, HTML escaping). The moments used by most of those tests are **fixtures**; the `RealMoments` class checks the real set.

## Checked in a real browser

The real moments were driven through the tab in Chrome (consent → all 8 moments, answering the opposite of ARIA each time → Disagree → done): every moment displayed correctly, nothing about ARIA appeared before
the answer, every reveal matched the recorded decision, the CSV had 8 rows with the exact columns and ARIA's decision/type, and no moment text or ARIA reply was in the file. A fixture run also checked Skip (adds no rows).
**Not checked:** Safari; a real Wi-Fi-off run; more than one visitor at a time.
