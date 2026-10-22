"""Frozen constants. Values come from SPEC.md; do not change without an amendment there."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv"
RESULTS = ROOT / "results"
EXPECTED_SHA256 = "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
EXPECTED_BYTES = 64_412_812

SPLIT_SEED = 20261015
BOOTSTRAP_SEED = 20261016
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test, by student
KS = (3, 5, 8)
T_DETERMINE = 10  # wheel-spinning determination point (opportunities)
MASTERY_RUN = 3
VARIANTS = ("A", "B")  # A: n >= 10 (primary), B: n > 10 (sensitivity)
