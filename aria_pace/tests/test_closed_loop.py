"""A6 pre-registered checks: with all three assumed effects at 0 the simulator equals the replay machinery; first-response rate; risk table; effect directions."""
import numpy as np

from aria_pace import baselines as B
from aria_pace import closed_loop as CL
from aria_pace import mastery_sens as MS
from aria_pace import policy as P
from aria_pace.labels import first_mastery
from aria_pace.loader import Sequence

FZ = CL.load_frozen()
TABS = CL.risk_tables(FZ["par"], FZ["platt"])
N = 60000
U = CL.draw(N, seed=11)


def test_risk_table_equals_bkt_score():
    bkt = B.BKT(10)
    bkt.kc_fits = {None: (10 ** 6, dict(FZ["par"]))}
    rng = np.random.RandomState(0)
    for _ in range(200):
        t = int(rng.randint(3, 10))
        bits = rng.randint(0, 2, t).astype(np.int8)
        if any(bits[j:j + 3].all() for j in range(t - 2)):
            continue
        q = Sequence(1, 1, None, np.concatenate([bits, np.zeros(2, np.int8)]), t + 2, first_mastery(bits), np.zeros(t + 2, np.int8))
        code = int("".join(map(str, bits)), 2)
        raw = bkt.score(q, t)
        assert abs(P.platt_apply(np.array([raw]), *FZ["platt"])[0] - TABS[t][code]) < 1e-12


def test_first_response_rate_matches_bkt():
    run = CL.simulate(FZ, TABS, U, None, 0, 0, 0)
    par = FZ["par"]
    expect = par["prior"] * (1 - par["slip"]) + (1 - par["prior"]) * par["guess"]
    got = float(run["resp"][:, 0].mean())
    assert abs(got - expect) < 4 * np.sqrt(expect * (1 - expect) / N)


def test_zero_effects_equal_replay_machinery():
    nohelp = CL.simulate(FZ, TABS, U, None, 0, 0, 0)
    seqs, keep = [], []
    for i in range(0, 4000):
        c = nohelp["resp"][i].astype(np.int8)
        seqs.append(Sequence(i, i, None, c, T := len(c), first_mastery(c), np.zeros(T, np.int8)))
    bkt = B.BKT(10)
    bkt.kc_fits = {None: (10 ** 6, dict(FZ["par"]))}
    for prof, (wI, wD) in P.PROFILES.items():
        tab = MS.SensTable(MS.relabel(seqs, MS.DEFS[0]), include_leavers=False, T=10)
        raw = MS.raw_matrix(tab, bkt, MS.DEFS[0])
        cal = P.platt_apply(raw, *FZ["platt"])
        fire = np.nan_to_num(cal, nan=-1.0) >= FZ["taus"][prof]
        replay = MS.seq_vec_T(tab, tab.err["primary"], fire, 10)                  # columns: fired, unnec, y1opp, missed, delay, ...
        run = CL.simulate(FZ, TABS, U, FZ["taus"][prof], 0, 0, 0)
        o = CL.outcomes(run, nohelp, wI, wD)
        idx = np.array([q.user_id for q in tab.seqs])
        np.testing.assert_array_equal(o["unnec"][idx], replay[:, 1].astype(bool))
        np.testing.assert_array_equal(o["delay"][idx], replay[:, 4])
        np.testing.assert_array_equal(o["missed"][idx], replay[:, 3].astype(bool))
        np.testing.assert_array_equal(o["fired"][idx], replay[:, 0].astype(bool))
        assert len(idx) > 500


def test_help_effect_improves_mastery_and_interrupt_cost_hurts():
    base = CL.simulate(FZ, TABS, U, -np.inf, 0.0, 0.0, 0.0)
    helped = CL.simulate(FZ, TABS, U, -np.inf, 0.5, 0.0, 0.0)
    hurt = CL.simulate(FZ, TABS, U, -np.inf, 0.0, 0.3, 0.0)
    scope = ~((base["mastered"] >= 1) & (base["mastered"] <= 3))
    m = lambda r: float((r["mastered"] > 0)[scope].mean())  # noqa: E731
    assert m(helped) > m(base) + 0.02 and m(hurt) < m(base) - 0.002
    stuck = CL.simulate(FZ, TABS, U, None, 0.0, 0.0, 0.3)
    nostuck = CL.simulate(FZ, TABS, U, None, 0.0, 0.0, 0.0)
    assert stuck["abandoned"].mean() > 0.05 and nostuck["abandoned"].sum() == 0


def test_common_random_numbers_prefixes_agree_until_intervention():
    a = CL.simulate(FZ, TABS, U, None, 0.3, 0.1, 0.0)
    b = CL.simulate(FZ, TABS, U, FZ["taus"]["supportive"], 0.3, 0.1, 0.0)
    first = np.where(b["i_first"] > 0, b["i_first"], 11)
    for t in range(1, 11):
        same = a["resp"][:, t - 1] == b["resp"][:, t - 1]
        assert same[first >= t].all(), t                                         # identical up to and including the intervention step
