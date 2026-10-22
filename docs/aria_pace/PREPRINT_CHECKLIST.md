# Preprint checklist (draft; nothing has been posted)

Status: a checklist for the author to work through. Nothing here has been submitted, posted or uploaded, and no venue policy below has been checked by the assistant that wrote this. Every policy item is marked **to verify** against the venue's own current page.

## 1. Where to post: OSF Preprints vs arXiv

| | OSF Preprints | arXiv |
|---|---|---|
| Account | Free account; no endorsement step that this checklist knows of (to verify) | A new submitter in a subject area usually needs an **endorsement** from an established author (to verify the current rule for the category you pick, probably cs.LG or cs.CY) |
| Review before it appears | Moderation depends on the chosen preprint service (to verify) | arXiv moderators screen submissions; reclassification or hold is possible (to verify) |
| Versioning | New versions allowed (to verify) | New versions allowed; every version stays public (to verify) |
| Withdrawal | A posted preprint generally cannot be fully deleted (to verify) | A withdrawn paper stays listed as withdrawn (to verify) |
| Fit for this work | Education / social-science readership (to verify) | CS / ML readership (to verify) |

Decision for the author. The assistant has no recommendation that depends on facts it has not checked. If no endorser is available, OSF avoids that dependency.

Either way: **posting is hard to undo.** Treat every item in section 2 as a gate before upload, and have a second person read the final PDF.

## 2. What must stay out

Keep out of the PDF, the supplement and any linked repository until the stated condition is met.

1. **All 2012-13 numbers and figures.** The 2012 results are a held-out test that the project treated as locked. Conditions to include any of them: the author decides, in writing, whether the 2012 evaluation is part of the paper. Until then, none of: `aria_pace/results/test_results_2012.json`, `docs/aria_pace/RESULTS_v2.md` 2012 tables, `docs/aria_pace/figures_v2/fig2_quiet_2009_vs_2012.png`, and the affect result (+0.0022, a 2012 test number).
2. **Any ASSISTments data, derived files or student identifiers**, until the terms of use for the dataset are confirmed. This includes the student-id split files `aria_pace/results/split.json` and `aria_pace/results/split_2012.json` (they hold lists of student ids: 2,498 / 833 / 832 for 2009 and 13,953 / 4,651 / 4,651 for 2012, plus 25 overlap ids), and any raw or row-level CSV. The terms, citation and redistribution rules are **not found in this repository**; confirm them with the dataset owner's page.
3. **Trained weights** `aria_pace/results/dkt_2009.pt` and `sakt_2009.pt`: derived from the data; same condition as item 2.
4. **Claims that are not in the frozen results** (anything about learning gains, clinical groups, "three-level scaffolding", "100 tasks", or classroom use).
5. **Student text, names or records** from any pilot (none exist; the pilot protocol is a draft and not approved).

### Finding to act on before posting anything: the public repository already serves some of these files

Checked read-only on 2026-10-21 (HTTP 200 on raw.githubusercontent.com; repository visibility reported as PUBLIC):

- branch `aria-pace`: `aria_pace/results/test_results_2012.json`, `aria_pace/results/split_2012.json`, `aria_pace/results/split.json`, `docs/aria_pace/RESULTS_v2.md`, `docs/aria_pace/figures_v2/fig2_quiet_2009_vs_2012.png`
- branch `summit-pace-mac`: `test_results_2012.json`, `split_2012.json`, `split.json` (RESULTS_v2.md and the figure returned 404 there)
- branch `main`: `test_results_2012.json` and `RESULTS_v2.md` returned 404 (the id-split files were not checked on main)

So the 2012 numbers and the id lists are reachable by anyone who knows the branch name, and a preprint that links the repository would point at them. Nothing was deleted or changed; removing files from a public branch does not remove them from git history or caches, and is a decision for the author (options: make the repository private until terms are confirmed; or rewrite history, which is outside this task). This is listed here so it is not discovered by a reviewer first.

## 3. Venue policies the author must check

None of these was looked up. Do not rely on any entry as stated; each is a question, not an answer.

| Venue | Question to settle | Status |
|---|---|---|
| LAK27 | Does it allow a preprint of a submitted paper? Is review double-blind (would a public repository or preprint break anonymity)? Is there a rule on prior posting? | to verify |
| NeurIPS workshop(s) the author submitted to (name each: ______) | Archival vs non-archival? Preprint allowed during review? Anonymity rules? | to verify |
| Any other venue (______) | Same three questions | to verify |
| ASSISTments data terms | Redistribution, derived data, required citation or acknowledgement | to verify |
| Institution / school | Does any school affiliation or sponsor need to approve public posting? | to verify |

## 4. Wording

- Say **"preprint"** or "technical report". Never "published paper", "peer-reviewed", "accepted".
- The data are "ASSISTments skill-builder math logs (grade level not stated), not calculus". Do not say "middle school" for them.
- The profiles are **cost weights** (3:1, 1:1, 1:3), not learner types. No clinical, ADHD or "learning gain" language.
- The model is Bayesian knowledge tracing (Corbett & Anderson, 1995). Multi-skill DKT / SAKT were not promoted.
- Describe the tutor result as a **simulated replay on logged data**, not as an effect on students.

## 5. Claims that need a qualifier, with the source of the qualifier

| Claim | Required qualifier | Source |
|---|---|---|
| "52% fewer interventions" | Quiet profile; test students (597); "mastery" means 3 correct in a row; missed-intervention rate 3.7% (CI 2.2-5.3). Not a claim about learning. | RESULTS.md:11; MASTERY_SENSITIVITY.md:3, 17-29 (both on branch `aria-pace`; not on this branch) |
| Sensitivity to the mastery definition | 49.8% / 8.6% / 32.2% / 45.4% / 9.0% across definitions; validation split only; BKT only; exploratory | MASTERY_SENSITIVITY.md (branch `aria-pace`) |
| "84.6%" | Measured on 350 held-out **synthetic** messages from the **same generator**, with no human ground truth; not a human-labelled accuracy | `eval/data/eval/eval_heuristic_350.json` lines 5, 6, 125; `cross_generator_heuristic.json:542` |
| "0 answer leaks" (demo harness) | Scripted inputs on a 30-problem SymPy-verified bank; says nothing about free-form students | `results/demo_harness_summit_pace.json`; `docs/benchmark/README.md` |
| Closed-loop simulation | Depends on stated assumptions about how students respond; a sensitivity analysis, not a measurement | RESULTS_v3 / A6 notes (branch `aria-pace`) |
| Neural models (DKT/SAKT) | Negative result on the validation split only; do not write "zero utility" | MASTERY_SENSITIVITY / RESULTS_v3 (branch `aria-pace`) |
| Friction index, concept map | Exploratory, not validated; in-session only | `docs/aria_pace/PILOT_MEASURES.md`; `agent/demo_graph.py:1-16` |
| Population | Grade level not stated for the logs | SPEC.md:94, RESULTS.md:205, DATA.md:140 (branch `aria-pace`) |

Line numbers for files on branch `aria-pace` were taken earlier in this task and were not re-opened after the branch switch; re-check each before quoting.

## 6. Before upload (tick when done)

- [ ] Section 2 items resolved or confirmed absent from the PDF and the supplement.
- [ ] Public-repository finding (section 2) decided.
- [ ] Section 3 policies checked and written down with the date.
- [ ] Every number in the PDF traced to a file and line (see `docs/summit/NUMBERS.md` on branch `summit-text`).
- [ ] Section 4 wording searched for in the full text.
- [ ] Section 5 qualifiers present next to each claim, not only in a limitations section.
- [ ] A second reader has checked the PDF.
- [ ] The author, not an assistant, presses submit.
