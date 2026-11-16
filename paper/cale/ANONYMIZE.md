# What to remove for an anonymous code link (prepared, not done)

Use this only if CALE-AI review turns out to be double-blind. Nothing has been removed, rewritten or published. The list comes from `git grep` over the `cale-experiments` branch at the commit that adds this file, plus `git log`, `git remote` and `git tag`.

## Recommended approach

Do not anonymize the repository in place. Build a **fresh snapshot with no git history** (for example `git archive` of a chosen subset), check it, and upload it to an anonymizing host (for example anonymous.4open.science; TODO: choose one).

The history cannot be cleaned cheaply: every commit carries a name, an email or a hostname (section 1).

**Minimal subset that reproduces the paper:**
- `aria_pace/` (pipeline, `SPEC.md`, `results/`, `results_cale/`)
- `scripts/cale/`
- `docs/cale/`
- `docs/aria_pace/` (only the files the paper cites)
- `paper/cale/` (without `ANONYMIZE.md`)
- `README` (TODO: a short anonymous one)

Everything else in the repository (the tutoring app, `eval/`, `agent/`, `ui/`, `docs/summit/`, the audits) is not needed for the paper and is the main source of identifying text.

## 1. Git metadata (dropped entirely by a history-free snapshot)

| Item | What it reveals |
|---|---|
| Commit authors and committers | `naren <narensara@narens-MacBook-Pro.local>` (113 commits), `Naren Saravanan <narens831709@gmail.com>`, `NarenS31 <sar762009@gmail.com>`, `Naren Saravanan <...+NarenS31@users.noreply.github.com>`, `Karthik Malireddy <...+kdmboss@users.noreply.github.com>` |
| Hostname in author emails | `narens-MacBook-Pro.local` |
| Remote | `https://github.com/NarenS31/aria.kls.git` (account name and repository name) |
| Tags | `summit-2026-10-24`, `summit-2026-10-24-pace` (name an event; possibly identifying) |

## 2. Names and account handles inside files

**In the minimal subset:**

| File | Line(s) | Content |
|---|---|---|
| `aria_pace/RESULTS_v3.md` | 30 | `NarenS31/ariawebsite` (GitHub handle) |
| `aria_pace/make_results_v3.py` | 46 | the same string, written into `RESULTS_v3.md` |

**Outside the minimal subset** (exclude these files, or edit them if they are shipped):
- `.gitignore:58`: `transfer_naren.jsonl`.
- `docs/AUDIT_20260826.md`, `docs/AUDIT_20260901.md`, `docs/SUMMIT_AUDIT.md`, `docs/summit/NUMBERS.md`: name, handle, repository name, and local paths.
- `docs/DEMO_RUNBOOK.md`, `docs/summit/`: no name, but 14+ mentions of the summit event (possibly identifying).
- `tests/test_learner_model.py`, `tests/test_intervention_pipeline.py`, `tests/test_product_reliability.py`, `tests/test_llm_fallback.py`: first name used as a test student; `narensara` in a path.
- `eval/data/metacognition/longitudinal_naren.json` and `interventions_naren.json`: **a real person's own learning data, named after them.** Exclude rather than rename.
- `eval/eval/evidence_report.py`, `eval/data/eval/EVIDENCE.md`, `eval/data/eval/LIMITATIONS.md`: refer to those files by name.
- `data/real_sessions.jsonl`: real usage sessions that contain the first name. Exclude.
- `docs/WORKSPACE_QA.md:33`: `/Users/karthikmalireddy/...` (a collaborator's name in a path).

## 3. Personal paths

| File | Content | Fix |
|---|---|---|
| `aria_pace/results_cale/run_all.log` | lines 33-73: `wrote /Users/narensara/aria-polish/...` | Regenerate after making `common.write` log paths relative to the repository root, or strip the prefix in the snapshot |
| future `run_all.log` / console output | `scripts/cale/common.py` and `e10_mrt_power.py` log absolute output paths | Same as above |
| `docs/AUDIT_*.md`, `docs/WORKSPACE_QA.md` | `/Users/...` paths | Excluded with those files |

## 4. Emails inside files (third parties, not authors)

`etrials@assistments.org` (`docs/aria_pace/DATA.md`, `eval/datasets/REQUEST_EMAIL.md`), plus `research@riiid.co`, `jiayizhg@upenn.edu` and `cborcher@cs.cmu.edu` (`eval/datasets/REQUEST_EMAIL.md`).

These are public contact addresses of dataset owners, not author identity. Keeping `DATA.md` is fine. `REQUEST_EMAIL.md` is a draft from the authors, so exclude it.

## 5. Not identifying (checked)

- **PDF metadata:** `main.pdf` shows only TeX and pdfTeX; the figure PDFs show only Matplotlib. No author field.
- **`_provenance` in result JSONs:** OS, Python and package versions, with no user name or path.
- **Raw dataset ids:** the ASSISTments `user_id` values are the dataset's own pseudonyms. The paper materials store only a hash for the Figure 1 sequence.

## Checklist before uploading the snapshot

- [ ] `git grep -i -E "naren|saravanan|narensara|NarenS31|karthik|kdmboss|aria\.kls|/Users/"` returns nothing in the snapshot
- [ ] No `.git/` directory in the snapshot
- [ ] `README` and `paper/cale/main.tex` contain no names, affiliations or acknowledgements
- [ ] Data files are excluded; the README points to the public ASSISTments download (DATA.md)
- [ ] TODO: confirm the CALE-AI review model (double-blind or not) before doing any of this
