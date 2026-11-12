"""Checks for the CALE-AI exploratory analyses (scripts/cale). Run: python -m pytest scripts/cale/tests"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUTPUTS = ["step1_reproduction.json", "e1_tradeoff.json", "e2_calibration.json", "e3_errors.json", "e4_paired.json",
           "e5_logged_hints.json", "e6_skills.json"]
HAVE_RAW = C.RAW_CSV.exists()


def _load(name):
    p = C.OUT / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet (run scripts/cale/run_all.sh)")
    return json.loads(p.read_text())


@pytest.mark.parametrize("name", OUTPUTS)
def test_every_output_is_labelled_exploratory_and_has_provenance(name):
    o = _load(name)
    assert "exploratory" in o["status"]
    prov = o["_provenance"]
    assert prov["inputs_sha256"] and prov["env"]["numpy"] and prov["runtime_seconds"] >= 0


def test_step1_reproduced():
    o = _load("step1_reproduction.json")
    assert o["reproduces"] and o["n_frozen_val_differences"] == 0 and o["n_val_results_differences"] == 0


def test_frozen_files_and_test_lock_unchanged():
    text = (C.RESULTS / "frozen_val.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (C.RESULTS / "frozen_val.sha256").read_text().strip()
    lock = json.loads((C.RESULTS / "TEST_LOCK").read_text())
    assert lock["status"] == "completed" and lock["frozen_sha256"].startswith("057f4389adbc")


def test_e1_matches_a7_at_cal_chosen_tau():
    o = _load("e1_tradeoff.json")
    for prof, r in o["a7_crosscheck"].items():
        assert r["a7_fold_taus"] == r["e1_fold_taus"]
        assert abs(r["a7_saving"] - r["e1_saving"]) < 1e-12 and abs(r["a7_missed"] - r["e1_missed"]) < 1e-12
    assert len(o["tau_grid"]) >= 50


def test_e5_main_equals_saved_validation():
    o = _load("e5_logged_hints.json")
    assert all(o["validation_main_matches_val_results_json"].values())


@pytest.mark.skipif(not HAVE_RAW, reason="raw 2009 file not present")
def test_no_test_student_in_cv_heldout():
    from aria_pace.loader import load_raw
    from aria_pace.split import make_split
    caches = sorted(C.CACHE.glob("cv_heldout_*.pkl"))
    if not caches:
        pytest.skip("no CV cache")
    d = pickle.loads(caches[-1].read_bytes())
    df, _ = load_raw(C.RAW_CSV)
    test_ids = set(make_split(df.user_id.unique())["test"])
    assert not set(d["H"].sid.tolist()) & test_ids


def test_e5_rule_b_drops_points_at_or_after_first_hint():
    HM = np.array([[False, False, False, True, False, False, False, False, False],
                   [True, False, False, False, False, False, False, False, False],
                   [False] * 9])
    upto = np.cumsum(HM, 1)[:, 2:9] > 0          # decision points i = 3..9
    assert upto[0].tolist() == [False, True, True, True, True, True, True]   # hint at try 4 -> drop i >= 4
    assert upto[1].all() and not upto[2].any()


def test_jsonable_handles_nan_and_numpy():
    assert C.jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.0, 2.0])}) == {"a": None, "b": 3, "c": [1.0, 2.0]}
