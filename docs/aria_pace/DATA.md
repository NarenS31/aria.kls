# ARIA-Pace — Data Check (Stage 2)

Branch `aria-pace`. Written 2026-10-15. Docs only: no training, no model code. One descriptive profile of the recommended file was computed (counts only) to check feasibility.

Labels: **VERIFIED** = read from the primary page or from the file itself in this session. **SECONDARY** = from a search-result snippet or third-party description, page not opened. **NOT STATED** = the sources I opened do not say. Nothing below is legal advice; where a term is ambiguous I say so.

---

## 1. Recommendation

**Primary dataset: ASSISTments 2009–2010 Skill Builder data, the corrected "collapsed" file (`skill_builder_data_corrected_collapsed.csv`).**

Why this one:
1. It is the only candidate I verified to contain all four signals the task needs per problem: first-attempt correctness (`correct`), `attempt_count`, `hint_count`, and `ms_first_response`, plus `user_id` and `skill_id`.
2. Every row is from a mastery ("Skill Builder") problem set (`type` = `MasterySection` in all 346,860 rows), so the three-correct-in-a-row mastery rule behind the Wang/Kai wheel-spinning label is the system's own design, not something I impose.
3. The file is directly reachable from the sandbox, so every number is reproducible from a single scripted download (Section 5), checksummed.
4. Terms permit research use with citation and no commercial use (Section 4).

Known weaknesses, stated up front (details in 3.1 and 6): grade level **not stated** by the source pages; no real timestamps (only `order_id` ordering), so "help delay" must be measured in problems/opportunities, not seconds; one row per problem, not per attempt, so "a verified error" has to be defined from `correct` = 0 and `first_action`, not from individual attempt events; 18.4% of rows have no skill and must be dropped.

Fallback if per-attempt timestamps become essential: PSLC DataShop (Section 3.3). It needs a manual account/dataset choice and cannot be fetched from the sandbox.

---

## 2. Comparison table

| | ASSISTments 2009–10 Skill Builder (corrected) | ASSISTments 2012–13 (school data, with affect) | ASSISTments 2014–15 "100 skill builders" | PSLC DataShop | EdNet (KT1–KT4) | Eedi (NeurIPS 2020 Education Challenge) |
|---|---|---|---|---|---|---|
| Access | Public Google Drive link on dataset page; **downloaded here** | Public Drive link; Drive returned a virus-scan interstitial to curl, **not downloaded** | Public Drive link; **header peeked only** | Account login; per-dataset | bit.ly → Google Drive zips | CodaLab competition page (per guide) |
| Student-level ID | `user_id` (VERIFIED in file) | `user_id` (page) | `user_id` (VERIFIED header) | `Anon Student Id` (help page) | one `{user_id}.csv` per student (page) | `UserId` (SECONDARY) |
| Correct | `correct` 0/1 (VERIFIED) | `correct` (page) | `correct` (VERIFIED header) | `Outcome` / `First Attempt` | derive from `user_answer` vs content | `IsCorrect` (SECONDARY) |
| Hints | `hint_count`, `hint_total`, `bottom_hint` (VERIFIED) | `hint_count`, `bottom_hint` (page) | **none** | `Hints` / `Total Num Hints` (help page) | **none** in KT1 (page) | **none** (single MC response) |
| Attempts | `attempt_count` (VERIFIED) | `attempt_count` (page) | **none** | `Attempt at Step` (transaction export) | no | no |
| Response time | `ms_first_response`, `overlap_time` (VERIFIED) | `ms_first_response`, `start_time`, `end_time` (page) | **none** | `Duration`, step times (help page) | `elapsed_time` (KT1) | NOT VERIFIED |
| Wall-clock timestamps | **none** (order only) | yes (`start_time`, `end_time`) | none (`log_id` order) | yes | yes | NOT VERIFIED |
| Skill / KC | `skill_id`, `skill_name` | `skill`, `skill_id` | `sequence_id` only | KC models | question tags (Contents) | question metadata |
| Mastery-set design | all rows (`MasterySection`) | `type` has Linear/Random/Mastery | all skill builders | depends on dataset | no | no |
| Grade level | **NOT STATED** | **NOT STATED** | middle school (298 middle schools, 23,896 students in the *related* JEDM analysis; see 3.2) | depends on dataset | **NOT STATED** (Korean test-prep app) | ages ~7–18 (SECONDARY) |
| Subject | math | math | math | varies | English (TOEIC) test prep (SECONDARY) | math diagnostic MCQs |
| Size | 346,860 rows; 4,217 students (VERIFIED) | ~10M problems in the underlying DB (page); file size not stated | 18.3 MB file | varies | 784,309 students; 131M interactions (page) | 118,971 students; 15.9M answers (SECONDARY) |
| Licence / terms | ASSISTments data terms (3.1) | same + affect-column citation request | same | non-commercial research; no redistribution (VERIFIED) | CC BY-NC 4.0 (VERIFIED on GitHub README) | CC BY-NC-ND 4.0 (SECONDARY; page now 404) |
| Fit for ARIA-Pace | **Best** | Possible 2nd dataset (has timestamps) | Replication of Wang et al. only; **no hints** | Best structure, not retrievable here | Poor (non-math, no hints) | Poor (no hints, no mastery rule) |

---

## 3. Per-dataset notes

### 3.1 ASSISTments 2009–10 Skill Builder (recommended)
- **Page:** sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010 (opened). The original file contains duplicated records; the page points to a **corrected version** with one row per student-problem, in which multi-skill problems are collapsed into one row with a combined skill string (e.g. `2_37_48`). Citation request on the page: cite the exact dataset URL; cite Feng, Heffernan & Koedinger (2009) for the system.
- **File fetched:** `skill_builder_data_corrected_collapsed.csv`, 64,412,812 bytes, sha256 `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`. Columns (VERIFIED from header): `order_id, assignment_id, user_id, assistment_id, problem_id, original, correct, attempt_count, ms_first_response, tutor_mode, answer_type, sequence_id, student_class_id, position, type, base_sequence_id, skill_id, skill_name, teacher_id, school_id, hint_count, hint_total, overlap_time, template_id, answer_id, answer_text, first_action, bottom_hint, opportunity, opportunity_original` (+ a leading unnamed row index).
- **Descriptive profile (VERIFIED by running code on the file; counts only):** 346,860 rows; 4,217 students; 149 distinct skill strings; 250 classes; 75 schools; 153 teachers; `type` is `MasterySection` for every row; `order_id` unique.
- **Column semantics checked against the data:**
  - `correct` is **first-attempt** correctness: `correct`=1 rows all have `first_action`=0, and `hint_count`=0 in all but 13 of 223,818. `correct`=0 includes both wrong-first-attempt and hint-requested-first.
  - `first_action` ∈ {0 attempt, 1 hint, 2 other} (distribution 309,271 / 31,273 / 6,316). I could not find a documented meaning for the codes; the interpretation 0 = answer attempt, 1 = hint request is inferred from the cross-tabulation with `correct` and `hint_count`, **not documented**.
  - `attempt_count` has an extreme outlier (max 3,824) and 7,213 zeros; needs clipping or a documented rule in Stage 3. `ms_first_response` has 7 negative values.
  - `opportunity` matches the within-student-skill rank by `order_id` in only 83.7% of rows, so I would **recompute opportunity from `order_id`** rather than trust the column.
  - 63,755 rows (18.4%) have no `skill_id`; 16.6% of rows with a skill are multi-skill combined strings.
- **Feasibility counts (descriptive, VERIFIED by code; student-skill pairs, ordered by `order_id`):**

  | Sequence length | pairs with at least this many opportunities | students with at least one such pair |
  |---|---|---|
  | >= 3 | 25,500 | 3,717 |
  | >= 5 | 18,635 | 3,316 |
  | >= 8 | 10,988 | 2,599 |
  | >= 10 | 8,396 | 2,278 |

  Of 42,995 student-skill pairs, 19,050 reach 3 correct in a row at some point.
- **Wheel-spinning sensitivity (label = no 3-consecutive-correct within the first 10 opportunities, following Wang et al. / Kai et al., attributed to Beck & Gong):**

  | Variant | Pairs with enough opportunities | Wheel-spinning pairs | Rate | Students with at least one |
  |---|---|---|---|---|
  | n >= 10 | 8,396 | 2,487 | 29.6% | 1,191 |
  | n > 10 | 7,077 | 2,241 | 31.7% | (not computed) |

  Both variants are feasible and differ by 2.1 points in prevalence. These are counts for feasibility only; they are not results. Selection caveats for Stage 3: (a) a skill builder stops at mastery, so sequences that master quickly are short by design; (b) pairs where the student leaves before 10 opportunities without mastering have no label under this definition and are excluded; the prediction at k must be conditioned on "not yet mastered by k", or the label leaks.
- **Terms.** Two ASSISTments sources, neither giving one clean licence for the public files:
  1. E-TRIALS Terms of Use (www.etrialstestbed.org/resources/terms-of-use, opened): written for studies run through E-TRIALS; the page does not say whether it covers the public data-set page. Clauses: no commercial use ("You will not use or reference ASSISTments data for any commercial purpose."), no de-anonymization, cite ASSISTments and the specific dataset, FERPA acknowledgement, no guarantee of quality.
  2. Data-sets page (www.etrialstestbed.org/data-sets, opened): public sets are those WPI judged to contain no PII; users agree not to identify students, not to share the data with others, to use it for the stated purpose, to acknowledge the ASSISTments TestBed, and to make **data and algorithms** public in papers.
  - **Ambiguity to resolve:** "do not share the data" and "make your data public" conflict on the face of it. Safe reading: publish code and the exact public download link plus checksum; do **not** redistribute the CSV in the repository. I recommend confirming with etrials@assistments.org before submission. Also, the dataset pages ask that papers citing the data be emailed to etrials@assistments.org.
- **Can it be used in a research paper?** Yes, with the conditions above. I did not find a statement forbidding publication, and the pages encourage it. Confirm with the maintainers for the final paper.

### 3.2 ASSISTments 2012–13 and 2014–15
- **2012–13** (page opened): adds `start_time`, `end_time`, `overlap_time`, `actions`, and four predicted-affect columns (frustrated/confused/concentrating/bored, 0–1 scores from detectors). The page asks you to cite Wang, Heffernan & Heffernan (2015) if you use affect, otherwise Feng, Heffernan & Koedinger (2009); problem text/answers are request-only. Row/student counts are **NOT STATED**; the page says about 3 GB of RAM is needed. **Not downloaded:** Drive served a virus-scan page to curl for this large file. Manual steps are in Section 5. Using the affect columns would run into the "no clinical categories" rule; they are model-predicted labels, not ground truth, and I would not use them.
- **2014–15 "100 skill builders"** (page opened, header peeked): columns are only `user_id, log_id, sequence_id, correct` (VERIFIED from the first lines of `2015_100_skill_builders_main_problems.csv`, 18,334,729 bytes). No hints, attempts, or response time. This is the file family Wang, Kai & Baker used (26,522 students, 1,088 problem sets, per their PDF). It is useful only for reproducing the "first 3/5/8 opportunities" result and cannot support help-related features. Grade: the related JEDM analysis (Kai et al.) reports 23,896 students from 298 middle schools in 2014–15 (PDF, line 307), but that is a different extract; the dataset page itself does not state it.

### 3.3 PSLC DataShop
- **Terms (opened, learnlab.org/datashop/Terms and the pslcdatashop /Terms page):** data are for research, not commercial purposes; no re-identification; **no redistribution** (collaborators need their own account); no guarantees; user indemnifies CMU. The page does not state citation requirements or which datasets are public.
- **Access:** account login ("Log in" on the About page). Whether any dataset can be downloaded anonymously is **NOT VERIFIED**. Not retrievable from the sandbox.
- **Columns (opened, help?page=export):** transaction export includes `Anon Student Id`, `Time`, `Duration (sec)`, `Attempt at Step`, `Outcome`, `Help Level`, `Total Num Hints`, `KC`, `School`, `Class`; student-step export includes `First Attempt`, `Incorrects`, `Hints`, `Corrects`, `Step Duration (sec)`, step start/end times. This is the richest schema (true per-attempt timestamps), but it varies by dataset and I have not chosen or inspected one. Grade level depends on the dataset.
- **Paper use:** permitted for research under these terms; citation guidance must come from each dataset's page.

### 3.4 EdNet
- **Licence (VERIFIED, github.com/riiid/ednet README):** CC BY-NC 4.0 for research. Download via bit.ly links to Google Drive: KT1 1.2 GB compressed (5.6 GB raw), KT2 0.6 GB, KT3 0.8 GB, KT4 1.2 GB. Contact research@riiid.co.
- **Columns (VERIFIED, README):** KT1 `timestamp, solving_id, question_id, user_answer, elapsed_time`; KT2–KT4 add action types (e.g. explanation reading, lecture watching, cursor time). Student ID = per-student file name.
- **Population:** 784,309 students, ~131M interactions (arXiv:1912.03072 abstract, opened). Subject and market: the app targets TOEIC English test preparation, mainly in Korea (SECONDARY: search results summarising arXiv/Riiid material). Age/grade: **NOT STATED**.
- **Verdict:** not suitable. English-language test prep for (presumably) adults is not the K-12 math tutor setting, KT1 has no hint or attempt fields, and results would not speak to the ARIA problem domain. Not downloaded (several GB).

### 3.5 Eedi
- **Licence (SECONDARY):** a search result states the Eedi project page says CC BY-NC-ND 4.0. The page (eedi.com/projects/neurips-education-challenge) now returns **404**, so this is unconfirmed. NoDerivatives deserves a read if any derived dataset were ever released.
- **Access (SECONDARY):** the competition guide (arXiv:2007.12061) says data is downloaded from the CodaLab competition page ("Participate" → "Get Data"). Not retrievable here; CodaLab may require registration. NOT VERIFIED.
- **Content (SECONDARY):** multiple-choice diagnostic questions, Sept 2018–May 2020; example columns `QuestionId, UserId, AnswerId, AnswerValue, CorrectAnswer, IsCorrect`; ages roughly 7–18; 118,971 students and 15.9M answers (tasks 1–2). One response per question: no hint, no attempts, no mastery rule.
- **Verdict:** not suitable for a hint-timing study.

---

## 4. Use in a research paper — summary

| Dataset | Research paper use | Redistribute data | Commercial | Citation |
|---|---|---|---|---|
| ASSISTments public sets | Allowed per the pages opened; confirm with etrials@assistments.org | Do not redistribute; publish code + official link + checksum | No | Dataset URL + Feng et al. 2009; email the citation to the maintainers |
| DataShop | Allowed (research) | No | No | Per dataset page (not checked) |
| EdNet | Allowed (CC BY-NC 4.0) | Per CC BY-NC (attribution) | No | Cite the AIED 2020 paper |
| Eedi | Allowed (SECONDARY; CC BY-NC-ND) | Not verified | No | Cite the challenge papers |

---

## 5. Exact steps

Automated (works from the sandbox; reproduced here):

```bash
mkdir -p aria_pace/data/raw && cd aria_pace/data/raw      # must be gitignored (Stage 3)
curl -L -o skill_builder_data_corrected_collapsed.csv \
  "https://drive.google.com/uc?export=download&id=1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
shasum -a 256 skill_builder_data_corrected_collapsed.csv
# expected: 162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da   (64,412,812 bytes)
```

Drive IDs can change; if the checksum or size differs, stop and re-open the dataset page.

Manual (if the automated download fails): open the 2009–10 Skill Builder page, follow the link labelled as the corrected version, use "Download", save as above, verify the checksum.

2012–13 (optional second dataset): open the 2012–13 school-data page, download the full file via the first Drive link; Drive will show a "can't scan for viruses" warning, choose "Download anyway"; allow ~3 GB RAM; email etrials@assistments.org only if you also need problem text.

DataShop (fallback): create an account at pslcdatashop.web.cmu.edu, search Dataset Search, open the chosen dataset's page, confirm the terms and citation text there, export the student-step or transaction table.

---

## 6. Risks to carry into Stage 3
1. **Grade level not stated** for the recommended file; the results section must say so rather than assume middle school.
2. **No wall-clock time**: "help delay" and "lead time" are measured in opportunities (problems), not seconds.
3. **Row = problem, not attempt**: a "verified error" is `correct` = 0 on a problem (first-attempt error), not each wrong attempt.
4. **Selection / censoring** from the mastery stopping rule and from students who leave before 10 opportunities.
5. **Label prevalence is high (~30%)** among labelled pairs; AUC and calibration need to be reported with the base rate.
6. **Duplicate and outlier handling** (`attempt_count` max 3,824; negative `ms_first_response`; null skills) must be written into the loader and logged, not done silently.
7. **Terms**: do not commit the CSV; confirm publication terms with the maintainers.
