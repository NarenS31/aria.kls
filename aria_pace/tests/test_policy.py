import numpy as np

from aria_pace import policy as P
from aria_pace.labels import first_mastery
from aria_pace.loader import Sequence


def mk(c, fa=None, uid=1):
    c = np.array(c, dtype=np.int8)
    fa = np.zeros(len(c), dtype=np.int8) if fa is None else np.array(fa, dtype=np.int8)
    return Sequence(uid, uid, "k", c, len(c), first_mastery(c), fa)


def test_always_help_and_cost_by_hand():
    ws = mk([0] * 12, uid=1)                        # y=1; first verified error at i=3 (decision pt 3)
    ok = mk([1, 0, 0, 0, 1, 1, 1], uid=2)           # y=0, masters at 7; errors at i=3,4 (before m=7)
    gone = mk([0, 0, 0, 0], uid=3)                  # leaver (n=4<10, never masters): dropped in primary
    t = P.Table([ws, ok, gone], include_leavers=False)
    assert len(t.y) == 2 and t.y.tolist() == [1, 0]
    err = t.err["primary"]
    m = P.evaluate_fire(t, err, t.valid & err, 1.0, 1.0)
    assert m["interventions"] == 2 and m["unnecessary"] == 1       # ok-sequence fired although it later masters
    assert m["total_delay_D"] == 0.0 and m["missed_help"] == 0     # always-help has zero delay
    assert m["mean_lead_time"] == 10 - 3
    never = P.evaluate_fire(t, err, np.zeros_like(err), 1.0, 1.0)
    assert never["interventions"] == 0 and never["missed_help"] == 1
    assert never["total_delay_D"] == 10 - 3                         # censored delay T - i_err
    full = P.Table([ws, ok, gone], include_leavers=True)
    assert len(full.y) == 3 and full.leaver.tolist() == [False, False, True] and full.y[2] == 0


def test_hint_first_is_excluded_from_primary_error():
    q = mk([0, 0, 0, 1], fa=[0, 1, 1, 0])           # opps 2,3 are wrong but hint-first (code 1)
    t = P.Table([q, mk([0] * 10, uid=2)], include_leavers=True)
    assert not t.err_primary[0, 0] and t.err_all[0, 0]


def test_future_responses_do_not_change_firing():
    a = mk([0, 0, 0, 0] + [1, 1, 1], fa=[0] * 7)
    b = mk([0, 0, 0, 0] + [0, 0, 0], fa=[0] * 7)
    S = lambda q, i: float(1 - q.correct[:i].mean())
    ta, tb = P.Table([a], True), P.Table([b], True)
    # decision points 3 and 4 only depend on first i responses
    Sa, Sb = P.score_matrix(ta, S), P.score_matrix(tb, S)
    assert np.allclose(Sa[0, :2], Sb[0, :2])


def test_matched_random_hits_target():
    seqs = [mk(([0, 1] * 8)[:12], uid=u) for u in range(1, 200)]
    t = P.Table(seqs, False)
    r = P.random_matched(t, t.err["primary"], 50, 1.0, 1.0, 20, 1)
    assert abs(r["interventions"]["mean"] - 50) <= 2


def test_choose_tau_prefers_larger_on_ties():
    sw = {0.0: {"unnecessary": 1, "total_delay_D": 1.0}, 0.5: {"unnecessary": 1, "total_delay_D": 1.0}}
    assert P.choose_tau(sw, 1.0, 1.0)[0] == 0.5


def test_run_fire_n1_is_always_help_and_vec_matches_evaluate_fire():
    from aria_pace import stage5 as S5
    seqs = [mk([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], uid=1), mk([1, 0, 0, 0, 1, 1, 1], uid=2), mk([0] * 12, uid=3)]
    t = P.Table(seqs, True)
    err = t.err["primary"]
    assert (P.run_fire(t, "primary", 1) & t.valid == (t.valid & err)).all()
    fire = P.run_fire(t, "primary", 2)
    old = P.evaluate_fire(t, err, fire, 3.0, 1.0)
    new = S5.point_metrics(S5.seq_vec(t, err, fire), len(set(t.sid.tolist())), 3.0, 1.0)
    for k in ("interventions", "unnecessary", "missed_help", "total_delay_D", "cost_J_per_student", "mean_lead_time",
              "mean_delay_incl_censored"):
        assert abs((old[k] or 0) - (new[k] or 0)) < 1e-9, k
