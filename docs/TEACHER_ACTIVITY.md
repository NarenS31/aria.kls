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
| M4 | alg-05 | "I think I'll try factoring first" | words only | stayed quiet (state UNKNOWN; planning not recognised) |
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

## Version 2 (DRAFT amendment, branch `activity-v2`; not in the demo build)

**Status: draft. Not merged into the demo build or `summit-activity`; not part of the tagged demo (`summit-2026-10-24-pace`).** Nothing below is stored or sent by default.

### What is added

- **An optional categorical reason when a teacher chooses "Step in".** After the kind of help, a second optional question, "Why?", offers `Slowing pace`, `Conceptual error`, `Emotional distress`, `Other`, or **Skip**. It exists only when the app is started with `--activity-reasons` (default **off**; the activity then behaves exactly as in v1, same screens and same CSV columns).
- **An optional free-text note** ("stays on this screen only; never saved"). It is kept in memory for the current moment only (`Activity.set_note`, `agent/teacher_activity.py`), truncated to 280 printable characters, cleared when the moment ends, and **never written to a file or sent anywhere** (tested).
- **The CSV** gets one extra column, `teacher_reason`, **only** when the log was opened with `collect_reasons=True` (`--activity-reasons`). Values come from the fixed list; the writer rejects a reason with "Stay quiet", an unknown value, or any reason when collection is off. Without the flag the file is byte-for-byte what v1 writes.

### Before anything is collected with reasons on

1. **The event organizer must sign off** on collecting the reason column (the writer is off by default for this reason).
2. **Adult-reviewed consent wording** must replace the current consent screen text. The screen's text changes automatically to mention the optional reason and that the note is not saved, but that wording is a draft, not reviewed.
3. Whether the venue or a school needs an ethics review is for the owner to decide; nothing in this repository states that a review took place.

### Analysis (descriptive; plan in `docs/teacher_timing_analysis.md`, Amendment 3 DRAFT)

`python scripts/analyze_teacher_taps.py taps.csv --moments data/teacher_moments.json [--profiles data/teacher_moments_profiles.json]` adds, **only when every moment has at least 15 sessions** (the v1 gate):

- **Alignment with ARIA per cost profile.** ARIA's decision on each moment under Quiet / Balanced / Supportive, generated by `scripts/make_moment_profiles.py` from the pace build (branch `summit-pace-mac`, read only), because v1's moments come from the pre-pace tag where profiles do not exist. Per profile: the confusion table of teacher majority against ARIA, response-level agreement with a bootstrap interval. The kappa **between teachers** is the v1 Fleiss' kappa (unchanged), reported when n allows.
- **Reasons against the state ARIA derived.** A table of the reasons teachers gave against ARIA's derived state for the moment (`stuck` / `flow` / `planning` / ...), counts under 5 shown as `<5`.
- **Verified-wrong moments reported separately.** They are left out of the ARIA-versus-majority tables (ARIA's decision there is fixed by design) and have their own reason table.
- **n is small and every result is descriptive only.** Reasons are teachers' own labels on hypothetical, scripted moments from a convenience sample; they show how teachers describe their choice, not why a student needs help, and nothing here validates ARIA.

### Tests

`tests/test_teacher_activity.py` (v1 tests unchanged except the handler tuple now has a sixth element): `ReasonsV2` (default unchanged, fixed values only, free text never in the file, UI phases) and `AnalysisV2` (reasons and per-profile tables at n = 15, nothing reported below it, v1 files still analyse).

### Finding while building it: the current 8 moments cannot tell the profiles apart

`data/teacher_moments_profiles.json` (generated from the pace build, `scripts/make_moment_profiles.py`) shows ARIA's decision on each of the 8 moments is **identical under Quiet, Balanced and Supportive**: stays quiet on M1, M2, M4, M5; steps in on M3, M6, M7, M8. The per-profile alignment table therefore cannot differ between profiles with these moments. The reason is by design: the profiles only act at attempts 3-9 on a verified wrong step (`agent/demo_pace.py`, ARIA-Pace gate), and these moments are early or contain no new verified wrong step at that stage. To compare profiles, a v2 set would need moments with a verified wrong step at attempts 3-9 (with history), chosen by the owner and regenerated with `make_moment_profiles.py`. Not done here.
