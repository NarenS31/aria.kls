"""Amendment A9: per-turn model with response time (exploratory, validation only; SPEC.md A9).

    python -m aria_pace.run_hazard          (from the repo root)

Reads the 2009 file (train / validation only; the test students are dropped right after the split; the 2012 file is never loaded). Writes results/hazard_val.json only.
"""
import json
import time

from .constants import RAW_CSV, RESULTS
from .hazard import load_sequences_with_time, run
from .loader import load_raw
from .split import make_split


def main():
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    test_ids, train_ids = set(parts["test"]), set(parts["train"])
    df = df[~df.user_id.isin(test_ids)]
    seqs = load_sequences_with_time(df)
    assert not {q.user_id for q in seqs} & test_ids
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    min_seq = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]["bkt_min_seq"]
    log(f"train sequences {len(train)}, validation {len(val)}")
    out = run(train, val, min_seq, log=log)
    out["env"] = {"seconds": round(time.time() - t0)}
    with open(RESULTS / "hazard_val.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("wrote results/hazard_val.json")


if __name__ == "__main__":
    main()
