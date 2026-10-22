"""A4 helpers: the control definition must equal the frozen code, and the EMA risk must equal brute-force enumeration."""
import itertools

import numpy as np

from aria_pace import baselines as B
from aria_pace import mastery_sens as MS
from aria_pace import policy as P
from aria_pace.labels import first_mastery, label
from aria_pace.loader import Sequence

PAR = {"prior": 0.8211508544355045, "learn": 0.08935045476163825, "guess": 0.21145065486693454, "slip": 0.20462863992921038}
D = {d.name: d for d in MS.DEFS}


def seq(bits, uid=1, fa=None):
    c = np.array(bits, dtype=np.int8)
    return Sequence(uid, 1, "k", c, len(c), first_mastery(c), np.zeros(len(c), dtype=np.int8) if fa is None else np.array(fa, dtype=np.int8))


def test_control_mastery_and_label_equal_frozen_code():
    rng = np.random.RandomState(0)
    for _ in range(300):
        bits = rng.randint(0, 2, rng.randint(1, 25))
        assert MS.first_mastery_def(bits, D["D3"]) == first_mastery(bits)
        m = first_mastery(bits)
        assert MS.label_def(len(bits), m, 10) == label(len(bits), m, "A")


def test_control_risk_equals_bkt_score_for_every_prefix():
    bkt = B.BKT(10)
    bkt.kc_fits = {None: (10 ** 6, dict(PAR))}
    for L in range(1, 10):
        for bits in itertools.product((0, 1), repeat=L):
            if len(bits) >= 3 and all(b == 1 for b in bits[-3:]):
                continue
            q = seq(list(bits) + [0] * 2)
            q.kc = None
            ref = bkt.score(q, L)
            got = MS.bkt_risk(np.array(bits, dtype=np.int8), PAR, D["D3"])
            assert abs(ref - got) < 1e-12, bits


def brute_force(prefix, par, d):
    """P(no mastery in k+1..T): enumerate latent paths and outcome paths explicitly."""
    post, _ = B.BKT._filter(np.array(prefix, dtype=np.int8), par)
    k = len(prefix)
    total = 0.0

    def rec(t, ls, hist, mass):
        nonlocal total
        if t > d.T:
            total += mass
            return
        for ls2, pt in ((1, 1.0 if ls == 1 else par["learn"]), (0, 0.0 if ls == 1 else 1 - par["learn"])):
            if pt == 0:
                continue
            for x, px in ((1, (1 - par["slip"]) if ls2 else par["guess"]), (0, par["slip"] if ls2 else 1 - par["guess"])):
                h = hist + [x]
                if MS.first_mastery_def(h, d) <= len(h) and MS.first_mastery_def(h, d) > k:
                    continue                            # mastered inside the window
                rec(t + 1, ls2, h, mass * pt * px)

    for ls, m in ((1, post[-1]), (0, 1 - post[-1])):
        rec(k + 1, ls, list(prefix), m)
    return total


def test_ema_and_run_rules_equal_brute_force():
    for name in ("DEMA", "D4", "D5T15"):
        d = D[name]
        d_small = MS.Def(d.name, d.kind, min(d.T, 8), run=d.run, alpha=d.alpha, thr=d.thr, consec=d.consec)
        for prefix in ([1, 0, 1], [0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0]):
            if MS.first_mastery_def(prefix, d_small) <= len(prefix):
                continue
            assert abs(MS.bkt_risk(np.array(prefix, dtype=np.int8), PAR, d_small) - brute_force(prefix, PAR, d_small)) < 1e-10, (name, prefix)


def test_ema_definition_examples():
    d = D["DEMA"]
    assert MS.first_mastery_def([1, 1, 1], d) == 3.0                    # ema 1,1,1
    assert MS.first_mastery_def([1, 0, 1, 1, 1, 1], d) == 6.0           # 1, .7, .79, .853, .897, .928 -> three >= .8 at 4,5,6
    assert MS.first_mastery_def([0, 1, 1, 1, 1], d) == float("inf")     # ema 0, .3, .51, .657, .76


def test_sens_table_equals_policy_table_under_control():
    rng = np.random.RandomState(1)
    seqs = [seq(rng.randint(0, 2, rng.randint(2, 20)).tolist(), uid=i // 3) for i in range(80)]
    a = P.Table(seqs, include_leavers=True)
    b = MS.SensTable(MS.relabel(seqs, D["D3"]), True, 10)
    for attr in ("y", "leaver", "sid", "valid", "err_primary", "err_all"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    from aria_pace import stage5 as S5
    fire = a.valid & a.err_primary
    assert np.array_equal(S5.seq_vec(a, a.err_primary, fire), MS.seq_vec_T(b, b.err_primary, fire, 10))
