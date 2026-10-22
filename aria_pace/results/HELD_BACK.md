# Held back

Held back pending dataset terms confirmation.

Removed from the public branch: `test_results_2012.json`, `split_2012.json` (2012-13 student-id lists and results) and `split.json` (2009-10 student-id lists). Code that reads them (`run_test2012.py`, `run_affect.py`, `split_2012.py`, `run_baselines.py`) is unchanged; tests that need them are skipped when the file is missing. The frozen policy `frozen_v2.json` and its `.sha256` are kept.
