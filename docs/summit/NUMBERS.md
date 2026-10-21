# Numbers used in the summit text, with sources

Files covered: `PROGRAM_DESCRIPTION.md`, `docs/aria_pace/SUMMIT_SLIDE.md` (the slide, updated in place), `TALK_90S.md`, `QA.md`. Paths are in this repository (NarenS31/aria.kls) unless a branch or another repository is named. Line numbers are on branch `summit-text` (= `aria-pace` at `eb4222a` plus these files) unless stated. **No number from the 2012 file or from any cross-dataset comparison is used anywhere.** Nothing was guessed: anything not found is listed at the end.

## Wording decision (sourced): how the 2009 data is described

Wherever the 2009–10 data is described, the text says **"ASSISTments skill-builder math logs (grade level not stated), not calculus"**. It never says "middle school" for this data.

| Reason | Source |
|---|---|
| Required statement: the grade level of the 2009–10 data is not stated by the sources; no grade claim is made | `aria_pace/SPEC.md:94` |
| Limitation 5: the sources for the 2009-10 file do not state the grade level | `docs/aria_pace/RESULTS.md:205` |
| Risk 1: the results must say the grade level is not stated rather than assume middle school | `docs/aria_pace/DATA.md:140` |
| The only "middle school" in the data note describes a different, related analysis, not this file | `docs/aria_pace/DATA.md:38` |
| "Not calculus": checked on the raw file, not on a committed file (see "Not found" below) | `aria_pace/data/raw/skill_builder_data_corrected_collapsed.csv`, column `skill_name` (the file is not committed) |

## Numbers

| Number as written | Where used | Source (file:line) |
|---|---|---|
| 52% (quiet setting, fewer interventions than always helping) | Program description, slide, 90 s, 3 min, QA 4 | `docs/aria_pace/RESULTS.md:11` |
| 1.22 vs 2.54 interventions per student | Slide, QA 6 | `docs/aria_pace/RESULTS.md:11` |
| 3.7% missed (95% CI 2.2% to 5.3%) | Program description, slide, 90 s, 3 min, QA 12 | `docs/aria_pace/RESULTS.md:11` |
| 597 test students, evaluated once | Slide, 3 min | `docs/aria_pace/RESULTS.md:9`, `:20` (597 students, 3,468 sequences, 570 wheel-spinning) |
| 15% fewer (2.15 vs 2.54), 0.2% missed (balanced) | Slide, 3 min | `docs/aria_pace/RESULTS.md:12` |
| 3% fewer (2.46 vs 2.54), difference includes zero; 0% missed (supportive) | Slide, 3 min, QA 6 | `docs/aria_pace/RESULTS.md:13`, `:158` |
| Weights 3 : 1, 1 : 1, 1 : 3 | Slide, QA 10 | `docs/aria_pace/RESULTS.md:22` |
| Stuck = no three correct in a row within ten attempts | All files | `docs/aria_pace/RESULTS.md:19` |
| Decision points are attempts 3 to 9 | QA 7 | `aria_pace/SPEC.md:75`; `docs/aria_pace/RESULTS.md:211` |
| Wrong step at attempts 1-2 and from attempt 10 on always gets a hint (live pace build); an attempt = a right/wrong answer | QA 7 | `agent/demo_pace.py:10-11` and `docs/DEMO_RUNBOOK.md:98`, both on branch `summit-pace-mac` (commit `945516e`) |
| BKT is from 1995 (Corbett & Anderson) | Program description, slide, 90 s, 3 min, QA 1 | `docs/aria_pace/LITERATURE.md:35-36` |
| Validation numbers are optimistic | QA 4 | `docs/aria_pace/RESULTS.md:210` |
| Limitations: replay has no tutor interventions; no learning-gain claim; grade level unknown | Slide, QA 3 | `docs/aria_pace/RESULTS.md:201`, `:203`, `:205` |
| Replay does not evaluate the math checker | QA 11 | `docs/aria_pace/RESULTS.md:204` |
| Profiles are cost settings; no claim about any clinical or other group | QA 10 | `docs/aria_pace/RESULTS.md:207`; `aria_pace/SPEC.md:96` |
| Why 2009 data (four reasons; terms allow research use with citation; checksummed download) | QA 2 | `docs/aria_pace/DATA.md:13-17`; weaknesses `:19` |
| Mastery rules: 49.8% (original, validation) | Slide, QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:17` |
| 4 correct in a row: 8.6% | Slide, 3 min ("about 9%"), QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:20` |
| 5 correct in a row within 15 attempts: 32.2% | QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:23` |
| 5 correct in a row within 20 attempts: 45.4% | QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:26` |
| Moving-average rule: 9.0% | Slide, QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:29` |
| Share counted as stuck: 10.7% to 75.9% | QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:7-11` (10.7% at line 10, 75.9% at line 11) |
| Best quiet threshold 0.27 to 0.69 | QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:26` (0.27), `:29` (0.69); original 0.57 at `:17` |
| Exploratory, validation only, BKT only, no test number for alternatives | Slide, 3 min, QA 4 | `docs/aria_pace/MASTERY_SENSITIVITY.md:3`; BKT only: `docs/aria_pace/RESULTS_v3.md` row 1 (column "What could not be done") |
| Closed loop: 100 settings; quiet beat 40, worse 55, indistinguishable 5 | QA 5 | `docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:17` |
| Closed loop: balanced beat 20, worse 80; supportive same as always helping | QA 5 | `docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:18-19`, `:78` |
| Flip at roughly 9 to 10% (quiet) and roughly 2% (balanced) | QA 5 | `docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:76` (0.087 to 0.100; 0.018 to 0.025), table `:65-72` |
| Assumption-dependent sensitivity analysis, not evidence of effect | Slide, 3 min, QA 5 | `docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:3` |
| Margin +0.01 in AUC at 3 and 5 attempts, interval above zero (promotion rule) | QA 1 | `docs/aria_pace/MULTISKILL.md:18`; rule fixed in `aria_pace/SPEC.md` Amendment 5, A5.4 |
| Single-skill LSTM-style -0.0027, -0.0027; attention-style -0.0078, -0.0134 | QA 1 | `docs/aria_pace/MULTISKILL.md:11-12`, `:14-15` (last column) |
| Multi-skill LSTM-style -0.0042, -0.0097; attention-style -0.0111, -0.0077 | QA 1 | `docs/aria_pace/MULTISKILL.md:11-12`, `:14-15` (paired difference column) |
| Multi-skill LSTM log-loss lower by 0.0174 (interval 0.0133 to 0.0218) | QA 1 | `docs/aria_pace/MULTISKILL.md:24` |
| Counting early leavers as not stuck: quiet threshold 0.57 to 0.85; missed help 3.7% to 12.4% (95% CI 9.4% to 15.5%) | QA 12 | `docs/aria_pace/RESULTS.md:122-123` |
| 8 moments in the activity | Program description (activity), 90 s, 3 min, QA 8 | `data/teacher_moments.json` lines 3, 26, 49, 72, 93, 114, 135, 156 (ids M1-M8), branch `summit-activity` |
| About two minutes; answers stay in the browser tab; nothing uploaded; draft consent not yet reviewed by an adult | QA 8, 90 s, 3 min | `src/TeacherActivity.tsx:42`, `:43`, `:44`, `:63` in NarenS31/ariawebsite, `origin/main` |
| No analytics; forms prepare an email; tutor storage and network behaviour not yet verified or promised | QA 8 | `app/privacy/page.tsx:10-11` in NarenS31/ariawebsite, `origin/main` |
| Pilot proposal: 2 to 3 sessions, 30 to 45 minutes; pilot enrollment not open; offline checks, consent review, school approval first | QA 9 | `src/HomeSections.tsx:29`, `:30`, `:112` in NarenS31/ariawebsite, `origin/main` |
| 8 featured problems, 104 turns, 0 leaks, 0 math errors, 0 crashes (scripted, original summit build) | QA 11 | `results/demo_harness_summit_pace.json` lines 3, 22-25, branch `summit-pace-mac` |
| 1995, "ten tries", "three", "four", "five", "one", "two" used as plain words or labels | All files | not data: definitions and labels (see rows above for the values they refer to) |

## Not found (nothing guessed)

- **"Not calculus" has no committed source.** No committed file lists the skill names of the 2009 file. I checked the raw file (not committed): its `skill_name` column holds only arithmetic and pre-algebra topics (fractions, decimals, equation solving, proportions, probability, a Pythagorean theorem skill) and none about calculus. If you want this sourced from a committed file, the check should be written into `docs/aria_pace/DATA.md`.
- **"Middle school" for the 2009 data: not found, and not used** (see the wording decision above).
- **No source in this repository describes the 3-minute "talk" timings**; they are speaking estimates (about 155 words a minute), not data.
