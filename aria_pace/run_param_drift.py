"""Amendment A8: parameter drift across skills (exploratory, validation only; SPEC.md A8).

    python -m aria_pace.run_param_drift          (from the repo root)

Fits BKT per skill on the 2009 TRAIN split only, reads the frozen calibrator and thresholds, scores VALIDATION. Writes results/param_drift_val.json only.
The 2009 test students are dropped right after the split; the 2012 file is never loaded. This is NOT a calculus calibration.
"""
import json
import time

import pandas as pd

from . import baselines as B
from .constants import RAW_CSV, RESULTS
from .loader import build_sequences, load_raw
from .param_drift import STATUS, decisions, per_skill_distance, spread
from .split import make_split


def skill_name_map(path=RAW_CSV):
    df = pd.read_csv(path, encoding="latin-1", usecols=["skill_id", "skill_name"], low_memory=False).dropna(subset=["skill_id"])
    df["skill_id"] = df.skill_id.astype(str)
    return {k: str(g.skill_name.mode().iloc[0]) for k, g in df.groupby("skill_id")}


def main():
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids, train_ids = set(parts["test"]), set(parts["train"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    assert not {q.user_id for q in seqs} & test_ids
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    min_seq = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]["bkt_min_seq"]
    frozen = json.load(open(RESULTS / "frozen_val.json"))
    names = skill_name_map()
    bkt = B.BKT(min_seq).fit(train)
    log(f"train sequences {len(train)}, validation {len(val)}; per-skill BKT fitted ({len(bkt.kc_fits) - 1} skills seen)")
    out = {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 8", "bkt_min_seq": min_seq,
           "spread": spread(bkt.kc_fits, min_seq, names, len(train)), "skill_distance": per_skill_distance(bkt.kc_fits, min_seq, names),
           "decisions_validation": decisions(val, bkt, names, frozen["platt"]["bkt"], frozen["configs"]["main"]["frozen"]["taus"]["bkt"])}
    out["env"] = {"seconds": round(time.time() - t0)}
    with open(RESULTS / "param_drift_val.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("wrote results/param_drift_val.json")


if __name__ == "__main__":
    main()
