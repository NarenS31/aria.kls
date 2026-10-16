# ARIA-Pace — Data check v2: ASSISTments 2012-13 (C1)

Branch `aria-pace`. Checked 2026-10-16. Docs only: no model code, no training. Numbers come from running descriptive code on the downloaded file
(results: `docs/aria_pace/data_profile_2012.json`; the two profiling scripts are in the appendix).

> **LICENCE HOLD (owner, 2026-10-16): confirmation of the data licence from ASSISTments is PENDING.** Until it arrives, nothing derived from the 2012-13 file (results, figures, the summit slide)
> is to be published or shared outside the team; `RESULTS_v2.md` and the slide carry the same notice (SPEC Amendment 3, A3.11). The dataset page states no licence (section 2); this note records the
> owner's statement, not something verified from a source.

**Labels.** **VERIFIED** = read from the primary source or from the file itself in this session. **UNVERIFIED** = not opened, or only seen as a search snippet, or inferred.
(Companion to `DATA.md`, which covers the 2009-10 file and the ASSISTments terms.)

## 1. Verdict

| Question | Answer |
|---|---|
| Usable as the held-out test set? | **Yes.** 46,674 students, a full school year, 23,280 students with skill-builder work. |
| Affect columns present? | **Yes: four, with no missing values.** So **C4 continues**. They are *detector-predicted*, not human labels. |
| Human BROMP observations obtainable? | **Not from this file.** No label column exists, and no public source for them was found (section 5). Availability on request: **UNVERIFIED**. |
| Grade level | **UNVERIFIED for this file.** The detector-training population was middle school (section 4). |
| Licence confirmation | **PENDING** from ASSISTments (owner, 2026-10-16). **No publication of 2012-13 results until it arrives.** |

## 2. License, terms, access  (VERIFIED unless marked)

- **License:** the dataset page (opened 2026-10-16) states **no license**. It links a "TermsOfUseForUsingData" page whose content I could not open (UNVERIFIED). The binding terms I could read are the E-TRIALS Terms of Use
  and the public data-sets page conditions, recorded in `DATA.md` section 3.1 (opened 2026-10-15): no commercial use, no de-anonymisation, cite ASSISTments and the dataset, acknowledge the TestBed, FERPA acknowledgement.
  Whether the E-TRIALS terms formally cover this public file is unclear; the same "do not share the data / make your data public" tension noted in `DATA.md` applies (publish code and the official link, do not redistribute the CSV).
- **Citation requests (page):** if the affect columns are used, cite **Wang, Heffernan & Heffernan (2015)**; otherwise cite Feng, Heffernan & Koedinger (2009). Problem text and answers are request-only (etrials@assistments.org).
- **Access:** public Google Drive link (no login). The sandbox can fetch it with the `drive.usercontent.google.com/download?...&confirm=t` form (the plain link returns a virus-scan page).
  File `2012-2013-data-with-predictions-4-final.zip`, **576,365,408 bytes**, sha256 `7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445`, zip CRCs OK, one CSV inside (3,009,494,391 bytes).
  Stored only in `aria_pace/data/raw/a2012/`, which is git-ignored (`aria_pace/.gitignore`). Never committed.
- **Reproduce the download:** `curl -L -o aria_pace/data/raw/a2012/2012-2013-data-with-predictions-4-final.zip "https://drive.usercontent.google.com/download?id=1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY&export=download&confirm=t"`, then check the sha256.

## 3. The file  (VERIFIED from the file)

- **6,123,270 rows; 46,674 students (`user_id`); 189,760 assignments; 803,980 (student, assignment) sequences; start times 2012-09-01 to 2013-08-31.** `problem_log_id` equals `problemlogid` in every row;
  within a sequence `problem_log_id` order equals time order in 99.98% of sequences.
- **35 columns:** `problem_log_id, skill, problem_id, user_id, assignment_id, assistment_id, start_time, end_time, problem_type, original, correct, bottom_hint, hint_count, actions, attempt_count,
  ms_first_response, tutor_mode, sequence_id, student_class_id, position, type, base_sequence_id, skill_id, teacher_id, school_id, overlap_time, template_id, answer_id, answer_text, first_action,
  problemlogid, Average_confidence(FRUSTRATED), Average_confidence(CONFUSED), Average_confidence(CONCENTRATING), Average_confidence(BORED)`. `actions` holds multi-line text, so the file needs a quote-aware parser.
- **Problem-set types (`type`, rows):** LinearSection 3,349,671; **MasterySection (skill builder) 1,868,746**; RandomChildOrderSection 875,949; RandomIterateSection 21,452; PlacementsSection 3,947; ChooseConditionSection 3,489; NumericLimitSection 16.
- **Skill builder subset:** 23,280 students, 368,752 sequences, 299,242 of which reach three correct in a row. Only **8.3%** of mastered sequences continue after the first 3-run (the 2009 file: about 47%), median sequence length 3, 90th percentile 9.
- **`correct` is not binary:** 4,141,564 ones, 1,976,383 zeros, and ~5,000 fractional values (0.25, 0.5, 0.75, ...). The page (2026-10-15) recommends treating 1 as correct and below 1 as incorrect; the same rule as the 2009 loader.
- **`first_action`** codes 0 / 1 / 2 = 5,782,418 / 305,009 / 35,843. As in 2009 the meanings are **inferred, not documented**.
- **Hints, attempts, time:** `hint_count` (mean 0.34, max 14), `attempt_count` (mean 1.34, max 29), `ms_first_response` (13 negative values, median 22,331 ms); real timestamps (`start_time`, `end_time`) exist, unlike the 2009 file.
- **Skill IDs:** null in 7.15% of skill-builder rows and 4.72% of skill-builder sequences have no first skill. 2012 IDs are floats (`277.0`); the **ID space is shared with 2009** (ID `13` is "Median" in both). 76.4% of 2012 skill-builder sequences start on
  a skill that appears in the 2009 file (80.1% of those with a skill); 56% of the distinct 2012 IDs appear in 2009. The 2009 file uses composite IDs (`10_104`), 2012 does not. Unseen skills fall back to pooled parameters, as in Stage 3.

## 4. Affect columns  (VERIFIED from the file; origin VERIFIED where noted)

| Column | min | max | mean | median | distinct values | nulls |
|---|---:|---:|---:|---:|---:|---:|
| `Average_confidence(FRUSTRATED)` | 0.361 | 0.867 | 0.390 | 0.361 | 831 | 0 |
| `Average_confidence(CONFUSED)` | 0.000 | 1.000 | 0.045 | 0.000 | 2,476 | 0 |
| `Average_confidence(CONCENTRATING)` | 0.171 | 0.767 | 0.682 | 0.767 | 842 | 0 |
| `Average_confidence(BORED)` | 0.000 | 1.000 | 0.257 | 0.222 | 11,213 | 0 |

- They are **predicted per row by detectors** (page: "predicted ... of student for the problem"; scores 0 to 1). They are **not observations**. In the first five opportunities of scored sequences there are no missing values.
- **The columns do not look like calibrated probabilities on a common scale**: FRUSTRATED and CONCENTRATING have narrow ranges and sit at their extreme for half the rows (median equals min or max) with only about 830 distinct values;
  CONFUSED is 0 for most rows. Pairwise correlations are small (largest: BORED vs CONCENTRATING, -0.225).
- **Which detector version produced them is UNVERIFIED.** The page says the detectors were refit later (first by Pardos, then by Botelho with deep learning); the file name says only "predictions-4-final".
- **Leakage risk to pre-register:** the detectors take interaction features (errors, correct answers, hints, timing; Ocumpaugh et al. list 69 features) that overlap with what the wheel-spinning models already see, and whether a row's value
  uses only that row's past is UNVERIFIED. An affect model may mostly re-encode response history, or may contain information from later in the sequence. The affect comparison must be run causally (only opportunities up to the decision point) and read with this in mind.

## 5. Source papers and human observations

| Paper | Record | Content opened? |
|---|---|---|
| San Pedro, Baker, Gowda & Heffernan (2013). Towards an Understanding of Affect and Knowledge from Student Interaction with an ITS. AIED, LNCS, pp. 41-50. doi:10.1007/978-3-642-39112-5_5 | **VERIFIED** (Crossref) | No (UNVERIFIED content) |
| Pardos, Baker, San Pedro, Gowda & Gowda (2014). Affective States and State Tests. *J. Learning Analytics* 1, 107-128. doi:10.18608/jla.2014.11.6 (LAK 2013 version doi:10.1145/2460296.2460320) | **VERIFIED** (Crossref) | Landing page only: detectors run on tutor log data; it does not mention BROMP or sample size |
| Ocumpaugh, Baker, Gowda, Heffernan & Heffernan (2014). Population validity for EDM models: a case study in affect detection. *BJET* 45(3), 487-501. doi:10.1111/bjet.12156 | **VERIFIED** (Crossref) | **Yes (pre-publication PDF)**: labels are quantitative field observations coded with **BROMP** (Ocumpaugh, Baker & Rodrigo 2012) on the HART Android app, synchronised to the logs; models built on **urban, suburban and rural middle-school** students using ASSISTments in the **Northeastern US**; 69 features. States no data release. |
| Wang, Heffernan & Heffernan (2015). Towards better affect detectors: effect of missing skills, class features and common wrong answers. LAK, pp. 31-35. doi:10.1145/2723576.2723618 | **VERIFIED** (Crossref) | Abstract only, via search snippet (three attempts to improve boredom / frustration / confusion / engagement detectors). Full text UNVERIFIED. |
| Botelho, Baker & Heffernan (2017). Improving Sensor-Free Affect Detection Using Deep Learning. AIED, LNCS, pp. 40-51. doi:10.1007/978-3-319-61425-0_4 | **VERIFIED** (Crossref) | Abstract only, via search snippet. Full text UNVERIFIED. |
| Botelho, Baker, Ocumpaugh & Heffernan. Studying Affect Dynamics and Chronometry Using Sensor-Free Detectors (the "EDM2018" paper) | Title and authors **VERIFIED** from the PDF's first page; **venue, year and pages UNVERIFIED** (Crossref returned no record; "EDM2018" appears only in its data URL) | **Yes (full text)** |

What the 2018-type paper says (VERIFIED from its text): the detectors "were originally developed using data collected by conducting **field observations** of student affect as **838 students** used ASSISTments", **3,127** 20-second observations;
and, about the data it released: "**The ground-truth labels used in model training are removed from this dataset and instead are replaced with the estimates produced by the sensor-free detectors.**" Its data link, `http://tiny.cc/EDM2018affectdata`,
now **redirects to a 404** (checked today). It does not use the word BROMP; the BROMP link comes from the Ocumpaugh et al. paper above.

**Human BROMP labels: not in this file (VERIFIED: no label or observation column among the 35), not in the public release described by that paper (labels removed, VERIFIED), no working public link found.** Whether the Baker/Heffernan groups
would share the original 3,127 observations on request is **UNVERIFIED**; I did not contact anyone and have no verified contact for this. Note also that the 838 observed students were the *training* population for the detectors; those students may or may not be in this file (UNVERIFIED).
The "affect" in all of C4 is therefore **detector-predicted affect**. The same page also lists Slater et al. (2017, ACII) as a detector reference; I did not look it up.

## 6. What the file can support  (counts only, not results)

Skill-builder sequences, same wheel-spinning label as before (fewer than 3 consecutive correct within the first 10 opportunities; first-attempt correct = `correct` >= 1; `type == MasterySection`; sequence = student x assignment).
"Scored" = eligible at k and labelled; "leavers" are eligible sequences with no label.

| Variant, k | Eligible | Leavers | Scored sequences | Students | Positives | Label rate |
|---|---:|---:|---:|---:|---:|---:|
| A (n >= 10), k=3 | 183,081 | 26,080 | 157,001 | 19,220 | 24,648 | 15.7% |
| A, k=5 | 98,766 | 12,401 | 86,365 | 16,374 | 24,648 | 28.5% |
| A, k=8 | 40,592 | 3,303 | 37,289 | 11,607 | 24,648 | 66.1% |
| B (n > 10), k=3 | 183,081 | 30,677 | 152,404 | 18,788 | 20,051 | 13.2% |
| B, k=5 | 98,766 | 16,998 | 81,768 | 15,701 | 20,051 | 24.5% |
| B, k=8 | 40,592 | 7,900 | 32,692 | 10,342 | 20,051 | 61.3% |

For scale: 2009 had 3,742 / 1,769 / 856 scored validation sequences at k = 3 / 5 / 8. A 20% student-level test portion of the 2012 skill-builder students would hold about 4,700 students and, at variant A, about 31,000 / 17,000 / 7,500
sequences at k = 3 / 5 / 8 (arithmetic from the totals above, not a measurement). The sequences are skewed: 15.7% positive at k=3 but 66.1% at k=8, so AUC and calibration are not comparable across k (as in 2009).
Skill-builder is 30.5% of rows; the linear and random sets have no mastery rule and are not used.

## 7. Caveats carried into the plan

- Replay on logs; "unnecessary" stays a proxy; no learning-gain claim; no clinical claim about anyone; affect here is **model output**, trained on a different population, with undocumented semantics (section 4).
- **Grade level is not in the file**; the only grade information is about the detector-training population (middle school, Northeastern US), so no grade claim is made about this file.
- A held-out test from a *different school year and population* tests transfer, but the 2009 and 2012-13 data come from the same platform, so it is not independent evidence about other tutoring systems.
- 2012 has real time stamps and affect; 2009 has neither. A model trained on 2009 cannot use affect.

## Appendix: profiling scripts

Scripts (not committed as code; run with `aria_pace/.venv/bin/python -I`, reading the CSV as untrusted data) produced `data_profile_2012.json`: (1) chunked read of 20 columns, descriptive statistics, skill-builder structure,
and label feasibility using `aria_pace.labels` (`first_mastery`, `label`, `eligibility`); (2) skill-builder `skill_id` null rates and overlap with the 2009 `skill_id` set. They are deterministic and read-only.
