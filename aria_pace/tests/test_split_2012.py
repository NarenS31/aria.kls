import hashlib
import json

import pytest

from aria_pace.constants import RESULTS
from aria_pace.split_2012 import SEED_2012, split_ids

if not (RESULTS / "split_2012.json").exists() or not (RESULTS / "split.json").exists():
    pytest.skip("split_2012.json / split.json are held back pending dataset terms confirmation (see results/HELD_BACK.md)",
                allow_module_level=True)

REC = json.loads((RESULTS / "split_2012.json").read_text())
P = REC["students"]


def test_disjoint_complete_and_fractions():
    s = {k: set(v) for k, v in P.items()}
    assert not (s["test"] & s["train"]) and not (s["test"] & s["val"]) and not (s["train"] & s["val"])
    n = sum(len(v) for v in s.values())
    assert n == REC["n_pool"] == 23255
    assert [len(P[k]) for k in ("test", "train", "val")] == [4651, 13953, 4651]
    assert abs(len(P["test"]) / n - 0.20) < 0.001 and abs(len(P["train"]) / n - 0.60) < 0.001


def test_overlap_with_2009_removed_and_hashes_match():
    removed = set(REC["removed_overlap_ids"])
    assert len(removed) == 25 == REC["n_removed_overlap_with_2009"]
    assert all(removed.isdisjoint(set(v)) for v in P.values())
    for k, v in P.items():
        assert hashlib.sha256(json.dumps(v).encode()).hexdigest() == REC["sha256"][k]


def test_recomputing_from_the_pool_gives_the_identical_split():
    pool = sorted(set().union(*[set(v) for v in P.values()]))
    again = split_ids(pool, SEED_2012)
    assert {k: sorted(v) for k, v in again.items()} == {k: sorted(v) for k, v in P.items()}
    assert REC["seed"] == SEED_2012 == 20261016


def test_no_2009_test_or_train_student_is_in_the_2012_test():
    s09 = json.loads((RESULTS / "split.json").read_text())["students"]
    ids09 = set().union(*[set(v) for v in s09.values()])
    assert ids09.isdisjoint(set(P["test"]))
