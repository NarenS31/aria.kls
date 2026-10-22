# Annotation kappa: readiness check (draft; humans are needed)

Date: 2026-10-22. Scope: the blind 150-item study for the language analyzer (seven cognitive-state categories). Docs plus a code check; **no real annotation exists and none was created.** Nothing under `eval/` was edited or moved.

## What exists

- **Study builder:** `eval/build_annotation_study.py` samples 150 items (`N_ITEMS = 150`, `:68`; seed `20260823`) from the synthetic think-aloud test split, evenly across the seven categories (`PLANNING, FLOW, CONFUSED, RUSHING, FRUSTRATED, STUCK, INSIGHT`: `eval/metacognition/generate.py:115-118`), and writes two blind sheets, a rubric, and a private key (`:1-26`, `write_key` at `:536`).
- **Blind sheets:** `eval/data/annotation/rater1_blind.csv` and `rater2_blind.csv` (header `item_id, problem_text, think_aloud, predicted_category, confidence`; 150 items; the `predicted_category` and `confidence` cells are empty; the two files are byte-identical, as designed). Rubric: `eval/data/annotation/rubric.md`.
- **Private key:** `annotation_private_key.json` is gitignored (`.gitignore:57`), so it is **not in the repository**. It exists only in the author's local checkout (existence checked, contents not opened for this note).
- **Calibration check (14 items):** `eval/score_calibration.py`, with tests `tests/test_calibration_scoring.py`. This script deliberately **refuses to emit a kappa** (`:21-30`, `:155`). Run in the local checkout where the key exists: **14 passed, 5 subtests passed**. Run on a clean clone (no key): 7 failed with `FileNotFoundError` for the key and 7 passed, so those tests depend on a local private file.
- **Kappa code that exists elsewhere:** binary Cohen's kappa for reasoning-move codes (`eval/analyze_reasoning_move_annotations.py:32`) and quadratic-weighted kappa for educator ratings (`eval/analyze_educator_ratings.py:40`); `tests/test_educator_ratings.py` passes (3). Neither scores the 150-item **seven-category** sheet. `tests/test_reasoning_moves.py` errors on import here (`No module named 'ollama'`), an environment gap, not a kappa problem.

## Gap found

`eval/score_calibration.py:29-30` says "the main kappa is computed over the 150-item sheet alone", but **a script that does that was not found** (searched `eval/`, `tests/`, `scripts/`, docs for the key file name and kappa functions).

## What was added (outside `eval/`)

- `scripts/annotation_kappa.py`: unweighted multi-class Cohen's kappa between two returned blind sheets, with a 95% bootstrap interval over items, raw agreement, per-category agreement, each rater against the private key, and agreement with and without the items whose text contains their own category word (the lexical confound recorded in the key, `eval/build_annotation_study.py:553`). It refuses incomplete sheets, unknown categories, wrong item counts, mismatched item ids, and two identical sheets.
- `aria_pace/tests/test_annotation_kappa.py`: 4 tests (known kappa values against a hand computation and scikit-learn when installed, chance-level near zero, an end-to-end run, the refusals). **Result: 4 passed.**
- **The test annotators are synthetic and labelled "CODE TEST ONLY":** they are generated in a temporary directory from a synthetic key (cell text `CODE TEST ONLY`, file names `CODE_TEST_ONLY_*`), are deleted when the test ends, and are not saved in the repository, so they cannot be mistaken for real annotations. The script also accepts `--label "CODE TEST ONLY ..."`, which is written into its output.

## What the real run needs

1. **Two independent annotators**, each given their own copy of the blind sheet and the rubric, working separately and without the key; neither sees the other's sheet. They fill `predicted_category` (one of the seven) and, if asked, `confidence` (1-5).
2. **Private key handling:** the key stays with the study owner, not in the repository and not in any folder shared with raters (`eval/build_annotation_study.py:540-541`); it is opened only after both sheets are returned and locked.
3. **Calibration round first** (`eval/score_calibration.py`): it checks the instrument, not the raters, and never yields the study kappa.
4. Run `python scripts/annotation_kappa.py --rater1 ... --rater2 ... --key <private key> --out <file>`; report kappa with its interval, raw agreement, per-category agreement and the with/without own-label-word split. No claim about real student language follows from synthetic items alone; the sheet text is synthetic (`eval/build_annotation_study.py:1-5`).
5. Raters' ethics and consent arrangements are the owner's to settle; not found in the repository.
