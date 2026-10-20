import numpy as np
import pandas as pd

from aria_pace.affect import RankTransform, affect_features, cumulative
from aria_pace.loader2012 import AFF, USECOLS, load_2012, norm_kc
from aria_pace.paired import auc_diff, mean_auc_diff


def test_rank_transform_is_monotone_midrank_bounded_and_train_only():
    train = np.array([[0.1, 5], [0.1, 6], [0.1, 7], [0.9, 8]], dtype=float)
    rt = RankTransform().fit(train)
    out = rt.transform(np.array([[0.1, 5], [0.9, 8], [0.0, 4], [1.0, 99]]))
    assert np.allclose(out[0], [(0 + 3) / 2 / 4, (0 + 1) / 2 / 4])        # ties share the mid-rank (3 equal values -> (0+3)/2)
    assert out[1, 0] == (3 + 4) / 2 / 4 and out[2, 0] == 0.0 and out[3, 0] == 1.0     # values beyond the training range clip to 0 and 1
    assert (out >= 0).all() and (out <= 1).all()
    x = np.linspace(0, 1, 50)
    assert np.all(np.diff(RankTransform().fit(np.c_[x, x]).transform(np.c_[x, x])[:, 0]) >= 0)


def test_affect_features_use_only_rows_up_to_k_and_lagged_only_up_to_k_minus_1():
    rng = np.random.RandomState(0)
    a = rng.rand(10, 4).astype(np.float32)
    k = 5
    base = affect_features(a, *cumulative(a), k)
    lag = affect_features(a, *cumulative(a), k, lag=1)
    b = a.copy(); b[k:] = rng.rand(5, 4)                                  # change everything after row k
    assert np.allclose(base, affect_features(b, *cumulative(b), k)) and np.allclose(lag, affect_features(b, *cumulative(b), k, lag=1))
    c = a.copy(); c[k - 1] = rng.rand(4)                                  # change row k itself
    assert not np.allclose(base, affect_features(c, *cumulative(c), k))   # primary sees row k
    assert np.allclose(lag, affect_features(c, *cumulative(c), k, lag=1)) # lagged does not
    assert len(base) == 12 and np.allclose(base[:4], a[k - 1]) and np.allclose(base[4:8], a[:k].mean(0)) and np.allclose(base[8:], a[:k].max(0))


def test_loader_keeps_only_requested_students_orders_by_time_and_normalises(tmp_path):
    rows = []
    def r(plid, uid, aid, t, correct, typ="MasterySection", skill=277.0, fa=0):
        rows.append({**{c: 0.5 for c in AFF}, "problem_log_id": plid, "user_id": uid, "assignment_id": aid, "start_time": t, "correct": correct,
                     "first_action": fa, "type": typ, "skill_id": skill})
    r(5, 1, 10, "2012-09-02 10:00:03", 1.0); r(4, 1, 10, "2012-09-02 10:00:01", 0.5); r(6, 1, 10, "2012-09-02 10:00:05", 1.0)   # given out of order
    r(7, 1, 11, "2012-09-03 09:00:00", 1.0, typ="LinearSection")                                                                # not skill builder
    r(8, 2, 20, "2012-09-02 11:00:00", 1.0, skill=np.nan)                                                                         # student 2: not requested
    r(9, 3, 30, "2012-09-02 12:00:00", 0.0, skill=np.nan)
    p = tmp_path / "x.csv"
    pd.DataFrame(rows)[USECOLS].to_csv(p, index=False)
    seqs = load_2012(p, keep_ids={1, 3})
    assert {q.user_id for q in seqs} == {1, 3}                           # the unrequested student (a stand-in for the locked test) never loaded
    q1 = next(q for q in seqs if q.user_id == 1)
    assert q1.n == 3 and list(q1.correct) == [0, 1, 1]                   # time order, and 0.5 counts as not correct
    assert q1.kc == "277" and next(q for q in seqs if q.user_id == 3).kc == "UNK"
    assert q1.affect.shape == (3, 4) and norm_kc(12.0) == "12" and norm_kc(float("nan")) == "UNK"


def test_paired_bootstrap_detects_a_real_gain_and_not_a_null():
    rng = np.random.RandomState(0)
    n = 4000
    sid = rng.randint(0, 800, n); y = rng.randint(0, 2, n)
    good = y + rng.randn(n) * 0.8; bad = rng.randn(n)
    r = auc_diff(y, sid, good, bad, n_boot=200, seed=1)
    assert r["ci95"][0] > 0
    r0 = auc_diff(y, sid, bad, bad + rng.randn(n) * 1e-9, n_boot=200, seed=1)
    assert r0["ci95"][0] <= 0 <= r0["ci95"][1] + 1e-9
    sets = {3: (y, sid, good, bad), 5: (y[:3000], sid[:3000], good[:3000], bad[:3000])}
    m = mean_auc_diff(sets, n_boot=100, seed=2)
    assert m["ci95"][0] > 0 and m["draws"] > 50
