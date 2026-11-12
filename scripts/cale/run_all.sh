#!/usr/bin/env bash
# CALE-AI exploratory analyses: one command, from the repository root.
#   PYTHON=aria_pace/.venv/bin/python scripts/cale/run_all.sh
# Needs the 2009-10 file at aria_pace/data/raw/skill_builder_data_corrected_collapsed.csv (docs/aria_pace/DATA.md).
# Writes only aria_pace/results_cale/. Never reads the test split; never touches aria_pace/results/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
OUT=aria_pace/results_cale
LOG="$OUT/run_all.log"
mkdir -p "$OUT"
rm -rf "$OUT/_cache"                       # full recomputation every time
: > "$LOG"
start=$(date +%s)
{
  echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)  git $(git rev-parse --short HEAD)"
  echo "python: $("$PY" -c 'import sys; print(sys.version.split()[0])')"
  "$PY" -m pip freeze 2>/dev/null | grep -iE '^(numpy|pandas|scipy|scikit-learn|matplotlib)==' || true
  echo "raw sha256: $(shasum -a 256 aria_pace/data/raw/skill_builder_data_corrected_collapsed.csv | cut -d' ' -f1)"
  echo "TEST_LOCK sha256 (must be unchanged): $(shasum -a 256 aria_pace/results/TEST_LOCK | cut -d' ' -f1)"
} | tee -a "$LOG"

run() {
  local t0=$(date +%s)
  echo "== $1" | tee -a "$LOG"
  (cd scripts/cale && "$PY" "$1.py") 2>&1 | tee -a "$LOG"
  echo "   $1 took $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
}

run step1_reproduce          # exits non-zero (and stops this script) if the saved validation numbers do not reproduce
run build_cv
run e1_tradeoff
run e2_calibration
run e3_errors
run e4_paired
run e5_logged_hints
run e6_skills
run e7_misses_by_help        # Addendum 1
run e8_hintfirst_variant
run e9_riskscore_paired
echo "TEST_LOCK sha256 after: $(shasum -a 256 aria_pace/results/TEST_LOCK | cut -d' ' -f1)" | tee -a "$LOG"
echo "total runtime $(( $(date +%s) - start ))s" | tee -a "$LOG"
