"""The 2012-13 student split (SPEC Amendment 3, A3.1). Computed ONCE and committed before any model touches 2012 data.

    python -m aria_pace.split_2012

Pool = students with at least one MasterySection (skill-builder) row, minus the user IDs that also occur in the 2009 file (25 of them).
Student-level, seed 20261016: one permutation of the sorted pool; the first 20% are the LOCKED TEST portion, of the remaining 80% the first 75% are
train and the last 25% validation (60 / 20 / 20 of the pool). Only user_id and type are read here; no outcomes are looked at.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from .constants import RESULTS, ROOT

CSV_2012 = ROOT / "data" / "raw" / "a2012" / "2012-2013-data-with-predictions-4-final.csv"
CSV_2009 = ROOT / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv"
SEED_2012 = 20261016
TEST_FRACTION = 0.20
TRAIN_OF_REST = 0.75


def pool_and_removed():
    sb = set()
    for c in pd.read_csv(CSV_2012, usecols=["user_id", "type"], chunksize=1_000_000):
        sb |= set(c.loc[c.type == "MasterySection", "user_id"].unique().tolist())
    u09 = set(pd.read_csv(CSV_2009, encoding="latin-1", usecols=["user_id"]).user_id.unique().tolist())
    removed = sorted(sb & u09)
    return sorted(sb - u09), removed, len(sb), len(u09)


def split_ids(pool, seed=SEED_2012):
    ids = np.array(sorted(pool))
    perm = np.random.RandomState(seed).permutation(len(ids))
    n_test = int(round(TEST_FRACTION * len(ids)))
    rest = perm[n_test:]
    n_train = int(round(TRAIN_OF_REST * len(rest)))
    parts = {"test": sorted(int(x) for x in ids[perm[:n_test]]), "train": sorted(int(x) for x in ids[rest[:n_train]]),
             "val": sorted(int(x) for x in ids[rest[n_train:]])}
    sets = {k: set(v) for k, v in parts.items()}
    assert not (sets["test"] & sets["train"]) and not (sets["test"] & sets["val"]) and not (sets["train"] & sets["val"])
    assert sum(len(v) for v in parts.values()) == len(ids)
    return parts


def main():
    pool, removed, n_sb, n_09 = pool_and_removed()
    parts = split_ids(pool)
    rec = {"seed": SEED_2012, "rule": "sorted pool -> RandomState(seed).permutation; first 20% test; of the remaining 80% first 75% train, last 25% val",
           "fractions_of_pool": {k: round(len(v) / len(pool), 4) for k, v in parts.items()},
           "n_skill_builder_students_2012": n_sb, "n_students_2009_file": n_09, "n_removed_overlap_with_2009": len(removed), "removed_overlap_ids": removed,
           "n_pool": len(pool), "n_students": {k: len(v) for k, v in parts.items()},
           "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in parts.items()}, "students": parts}
    (RESULTS / "split_2012.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({k: rec[k] for k in ("seed", "n_skill_builder_students_2012", "n_removed_overlap_with_2009", "n_pool", "n_students", "fractions_of_pool")}, indent=1))


if __name__ == "__main__":
    main()
