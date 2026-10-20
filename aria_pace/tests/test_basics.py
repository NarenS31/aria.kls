import numpy as np

from aria_pace.baselines import AlwaysHelp, BKT, PFA, ThreeWrongInARow, fit_bkt_em
from aria_pace.labels import eligibility, first_mastery, label
from aria_pace.loader import Sequence
from aria_pace.split import make_split


def seq(c, kc="1"):
    c = np.array(c, dtype=np.int8)
    return Sequence(1, 1, kc, c, len(c), first_mastery(c))


def test_first_mastery():
    assert first_mastery([1, 1, 1]) == 3
    assert first_mastery([1, 1, 0, 1, 1, 1]) == 6
    assert first_mastery([0, 1, 1]) == float("inf")


def test_label_variants_differ_only_at_n_equals_10():
    never = [0] * 10
    q10, q11, q9 = seq(never), seq(never + [0]), seq([0] * 9)
    assert label(q10.n, q10.m, "A") == 1 and label(q10.n, q10.m, "B") is None
    assert label(q11.n, q11.m, "A") == 1 and label(q11.n, q11.m, "B") == 1
    assert label(q9.n, q9.m, "A") is None and label(q9.n, q9.m, "B") is None
    mastered = seq([1, 1, 1])
    assert label(mastered.n, mastered.m, "A") == 0 and label(mastered.n, mastered.m, "B") == 0
    late = seq([0] * 10 + [1, 1, 1])  # mastery after opportunity 10 counts as wheel-spinning
    assert label(late.n, late.m, "A") == 1 and label(late.n, late.m, "B") == 1


def test_eligibility():
    assert eligibility(2, np.inf, 3) == "short"
    assert eligibility(5, 3, 3) == "mastered"   # mastered at k is excluded
    assert eligibility(5, 4, 3) == "eligible"


def test_split_is_student_disjoint_and_deterministic():
    ids = list(range(1000))
    a, b = make_split(ids), make_split(ids)
    assert a == b
    assert not (set(a["train"]) & set(a["val"])) and not (set(a["train"]) & set(a["test"])) and not (set(a["val"]) & set(a["test"]))
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 1000


def test_rules_use_only_first_k():
    q = seq([1, 0, 1, 0, 0, 0])
    assert ThreeWrongInARow().score(q, 3) == 0.0 and ThreeWrongInARow().score(q, 6) == 1.0
    assert AlwaysHelp().score(q, 3) == 1.0


def test_bkt_and_pfa_risk_is_probability_and_ignores_future():
    rng = np.random.RandomState(0)
    train = [seq((rng.rand(12) < 0.6).astype(int)) for _ in range(200)]
    train = [q for q in train]
    kc_fits = {None: (len(train), fit_bkt_em([q.correct for q in train]))}
    bkt = BKT(10).fit(train, kc_fits)
    pfa = PFA(1.0).fit(train)
    a = seq([0, 1, 0, 0, 1, 1, 1, 1])
    b = seq([0, 1, 0, 0, 1, 0, 0, 0])  # same first 5 responses, different future
    for m in (bkt, pfa):
        ra, rb = m.score(a, 5), m.score(b, 5)
        assert 0.0 <= ra <= 1.0 and abs(ra - rb) < 1e-12
