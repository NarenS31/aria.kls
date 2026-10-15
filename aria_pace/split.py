"""Student-level split with a fixed seed (SPEC 4)."""
import hashlib
import json

import numpy as np

from .constants import SPLIT_FRACTIONS, SPLIT_SEED


def make_split(user_ids, seed=SPLIT_SEED, fractions=SPLIT_FRACTIONS):
    ids = np.array(sorted(set(int(u) for u in user_ids)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n_tr = int(round(fractions[0] * len(ids)))
    n_va = int(round(fractions[1] * len(ids)))
    parts = {
        "train": sorted(ids[perm[:n_tr]].tolist()),
        "val": sorted(ids[perm[n_tr:n_tr + n_va]].tolist()),
        "test": sorted(ids[perm[n_tr + n_va:]].tolist()),
    }
    assert_disjoint(parts)
    assert sum(len(v) for v in parts.values()) == len(ids)
    return parts


def assert_disjoint(parts):
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(parts[a]) & set(parts[b])
            assert not overlap, f"students in both {a} and {b}: {sorted(overlap)[:5]}"


def split_record(parts, seed=SPLIT_SEED):
    return {
        "seed": seed,
        "fractions": list(SPLIT_FRACTIONS),
        "n_students": {k: len(v) for k, v in parts.items()},
        "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in parts.items()},
        "students": parts,
    }
