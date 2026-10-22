"""Amendment A7: run the cross-validated trade-off analysis on 2009 train + validation only (exploratory; SPEC.md A7).

    python -m aria_pace.run_cv_tradeoff          (from the repo root)

Writes results/cv_tradeoff_val.json only. The 2009 test students are dropped right after the split (as run_mastery_sens.py); the 2012 file is never loaded.
"""
import json
import time

from .constants import RAW_CSV, RESULTS
from .cv_tradeoff import run
from .loader import build_sequences, load_raw
from .split import make_split


def main():
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids = set(parts["test"])
    seqs = [q for q in build_sequences(df) if q.user_id not in test_ids]
    assert not {q.user_id for q in seqs} & test_ids
    min_seq = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]["bkt_min_seq"]
    frozen_tau = json.load(open(RESULTS / "frozen_val.json"))["configs"]["main"]["frozen"]["taus"]["bkt"]
    log(f"pool sequences {len(seqs)} (train+validation; test students dropped: {len(test_ids)})")
    out = run(seqs, min_seq, frozen_tau, log=log)
    out["env"] = {"seconds": round(time.time() - t0), "pool_students": len({q.user_id for q in seqs}), "test_students_dropped": len(test_ids)}
    with open(RESULTS / "cv_tradeoff_val.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("wrote results/cv_tradeoff_val.json")


if __name__ == "__main__":
    main()
