# Numbers used in the summit text, with sources

Files covered: `PROGRAM_DESCRIPTION.md`, `docs/aria_pace/SUMMIT_SLIDE.md` (the slide, updated in place), `TALK_90S.md`, `QA.md`. Paths are in this repository (NarenS31/aria.kls) unless a branch or another repository is named. Line numbers are on branch `summit-text` (= `aria-pace` at `eb4222a` plus these files) unless stated. **Rows marked "aria-pace d596b26" cite `aria_pace/RESULTS_v4.md` and `aria_pace/results/*_val.json`; both are now on `main` and the line numbers are unchanged. `docs/summit/BRIEF.md` uses the same numbers as QA 4 and the slide (sensitivity, missed help, pooled-parameter note); its rows are marked "BRIEF".** **No number from the 2012 file or from any cross-dataset comparison is used anywhere.** Nothing was guessed: anything not found is listed at the end.

## Wording decision (sourced): how the 2009 data is described

Wherever the 2009–10 data is described, the text says **"ASSISTments skill-builder math logs (grade level not stated), not calculus"**. It never names a school level for this data.

| Reason | Source |
|---|---|
| Required statement: the grade level of the 2009–10 data is not stated by the sources; no grade claim is made | `aria_pace/SPEC.md:94` |
| Limitation 5: the sources for the 2009-10 file do not state the grade level | `docs/aria_pace/RESULTS.md:205` |
| Risk 1: the results must say the grade level is not stated rather than assume a school level | `docs/aria_pace/DATA.md:140` |
| The only school-level label in the data note describes a different, related analysis, not this file | `docs/aria_pace/DATA.md:38` |
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
| Cross-validated Quiet saving under 3 in a row: 50.1% (95% 48.5% to 51.6%); missed help 6.2% | QA 4, QA limits b, slide, BRIEF | aria-pace d596b26: `aria_pace/RESULTS_v4.md:13` (and `:29`); data `aria_pace/results/cv_tradeoff_val.json` (D3, quiet) |
| Cross-validated Quiet saving, other rules: 8.3% (4 in a row), 26.4% (5 in a row, 15 attempts), 41.6% (5 in a row, 20 attempts), 7.1% (moving-average rule); range 7.1% to 41.6% | QA 4, QA limits b, slide, BRIEF, 3 min ("about 8%", "about 7% to 42%") | aria-pace d596b26: `aria_pace/RESULTS_v4.md:16`, `:19`, `:22`, `:25` (summary `:30`) |
| Threshold moves by more than 0.10 in several folds under other definitions | QA 4 | aria-pace d596b26: `aria_pace/RESULTS_v4.md:16-25` (last column: 3, 4, 5, 3 folds for the Quiet rows of D4, D5T15, D5T20, the moving-average rule), `:31` |
| Missed help "about 4 to 6%": 3.7% on the locked test; 4.8% in validation; 6.2% cross-validated | QA 4, QA limits b, slide, BRIEF, 90 s | 3.7%: `docs/aria_pace/RESULTS.md:11`; 4.8%: `docs/aria_pace/MASTERY_SENSITIVITY.md:17`; 6.2%: aria-pace d596b26 `aria_pace/RESULTS_v4.md:13` |
| Original 49.8% (validation, own-skill parameters) versus 43.9% with one pooled parameter set (Quiet); 14.4% of Quiet wrong-step decisions flip | QA 2, QA limits c, slide, BRIEF, 3 min ("about 44% instead of 50%") | aria-pace d596b26: `aria_pace/RESULTS_v4.md:51` (also `:54-55`); data `aria_pace/results/param_drift_val.json` (`decisions_validation.profiles.quiet`) |
| The demo uses one pooled BKT set; the evaluated policy used per-skill parameters where a skill has enough data | QA 2, QA limits c, slide, BRIEF, 3 min | demo: `agent/demo_pace.py:99` (`bkt_pooled_params` from `baselines_val.json`: one pooled parameter set), branch `summit-pace-mac`; evaluated policy: `aria_pace/baselines.py` `BKT.params` (per-skill with pooled fallback), `aria_pace/SPEC.md` Amendment 8 (aria-pace d596b26) |
| Adding response time and a per-turn model (attempts 1 to 9): +0.0072 AUC at 3 attempts, +0.0034 at 5, against the +0.01 bar | QA 1, BRIEF, 3 min ("did not clear our bar") | aria-pace d596b26: `aria_pace/RESULTS_v4.md:67`, `:69`, `:75`; bar: `aria_pace/SPEC.md` Amendment 9, A9.3 |
| The 8 "Would you step in?" moments get the same ARIA decision under quiet, balanced and supportive | QA 10 | aria-pace d596b26: `aria_pace/RESULTS_v4.md:80`; data: `data/teacher_moments_profiles.json` and `docs/TEACHER_ACTIVITY.md` ("Finding while building it"), branch `activity-v2` (pace reasons `wrong_before_window` for M3 and M8) |
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
- **A school level for the 2009 data: not found, and not used** (see the wording decision above).
- **No source in this repository describes the 3-minute "talk" timings**; they are speaking estimates (about 155 words a minute), not data.
