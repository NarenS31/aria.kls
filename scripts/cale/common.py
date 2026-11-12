"""Shared helpers for the CALE-AI exploratory analyses (docs/cale/ANALYSIS_PLAN.md).

Only the 2009-10 file is loaded. The 2009 test students are dropped right after the split, exactly as aria_pace/freeze_val.py:22-24.
Nothing here writes outside aria_pace/results_cale/.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from aria_pace.constants import RAW_CSV, RESULTS  # noqa: E402
from aria_pace.loader import build_sequences, load_raw, sha256_file  # noqa: E402
from aria_pace.split import make_split  # noqa: E402

OUT = REPO / "aria_pace" / "results_cale"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
STATUS = "exploratory, post hoc addition to the locked evaluation (CALE-AI); no test student used"

SEEDS = {"split": 20261015, "cv_folds": 20261022, "random_policy": 20261017,
         "e1_bootstrap": 20261112, "e2_bootstrap": 20261113, "e4_bootstrap": 20261114, "e5_bootstrap": 20261115}
N_BOOT = 1000
FROZEN_TAUS = {"quiet": 0.57, "balanced": 0.19, "supportive": 0.09}


def log_fn():
    t0 = time.time()
    return lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)


def read_result(name):
    """Read-only access to a file under aria_pace/results/ (records its hash)."""
    p = RESULTS / name
    _INPUTS[str(p.relative_to(REPO))] = sha256_file(p)
    return json.loads(p.read_text())


_INPUTS: dict = {}


def load_pool(log=print):
    """Returns (df_clean, parts, train_seqs, val_seqs). Test students are removed before sequences are kept."""
    df, rep = load_raw(RAW_CSV)
    _INPUTS[str(RAW_CSV.relative_to(REPO))] = sha256_file(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    assert not {q.user_id for q in seqs} & test_ids
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    log(f"loaded: train {len(train)} seqs, val {len(val)} seqs; test students dropped: {len(test_ids)}")
    return df, parts, train, val, rep


def assert_no_test(ids, parts):
    bad = set(int(u) for u in ids) & set(parts["test"])
    assert not bad, f"{len(bad)} test students reached an output"


def env():
    import matplotlib, pandas, scipy, sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(name, obj, t_start):
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj.setdefault("status", STATUS)
    obj["_provenance"] = {"inputs_sha256": dict(sorted(_INPUTS.items())), "env": env(), "seeds": SEEDS,
                          "runtime_seconds": round(time.time() - t_start, 1),
                          "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = OUT / name
    p.write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True))
    return p


def ci(a, axis=0):
    lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=axis)
    return [lo, hi]
