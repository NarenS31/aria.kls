"""A5 multi-skill models: histories are built without leakage, and the exact risk equals the single-skill code (no other rows) and brute force (other rows)."""
import numpy as np
import pandas as pd
import torch

from aria_pace import dkt as D
from aria_pace import multiskill as MS
from aria_pace import sakt as SK
from aria_pace.constants import MASTERY_RUN
from aria_pace.loader import build_sequences

VOCAB = {"a": 1, "b": 2}


def frame(rows):
    df = pd.DataFrame(rows, columns=["user_id", "assignment_id", "order_id", "skill_id", "correct"])
    df["first_action"] = 0
    df["hint_count"] = 0
    return df.sort_values(["user_id", "order_id"], kind="stable").reset_index(drop=True)


def single_student_df():
    rng = np.random.RandomState(3)
    c = rng.randint(0, 2, 14)
    c[:3] = [0, 1, 0]
    return frame([(1, 1, i, "a", int(x)) for i, x in enumerate(c)])


def interleaved_df():
    rows, oid = [], 0
    rng = np.random.RandomState(5)
    for i in range(6):                                           # rows of another assignment/skill before and between
        rows.append((7, 99, oid, "b", int(rng.randint(0, 2)))); oid += 1
    for i in range(12):
        rows.append((7, 1, oid, "a", int([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0][i]))); oid += 1
        if i in (1, 4):
            rows.append((7, 98, oid, "b", 1)); oid += 1          # interleaved rows from a third assignment
    return frame(rows)


def models(seed=0):
    torch.manual_seed(seed)
    dkt = D.DKT(3, 16).eval()
    sk = SK.SAKT(3, 16).eval()
    return dkt, sk


def test_prefix_uses_only_rows_up_to_the_kth_opportunity():
    df = interleaved_df()
    st = MS.Streams(df, VOCAB)
    q = [s for s in build_sequences(df) if s.assignment_id == 1][0]
    toks, n = MS._prefix(st, q, 3)
    p = st.pos[(7, 1)]
    assert n == p[2] + 1 and len(toks) == n                       # everything up to and including opportunity 3, nothing later
    assert (toks == st.tokens(7)[:n]).all()
    assert any(t // 2 == 2 for t in toks)                         # contains the other-skill rows (kc id 2)
    assert n > 3                                                  # more than the sequence's own three rows (other rows are before it)


def test_history_cap():
    rows = [(1, 1, i, "a", i % 2) for i in range(300)]
    df = frame(rows)
    st = MS.Streams(df, VOCAB)
    q = build_sequences(df)[0]
    toks, n = MS._prefix(st, q, 5)
    assert n == 5                                                 # k-th opportunity is row 5; history is rows 0..4 only
    df2 = frame([(1, 1, i, "a", 0) for i in range(5)] + [(1, 2, 100 + i, "a", i % 2) for i in range(12)])
    st2 = MS.Streams(df2, VOCAB)
    q2 = [x for x in build_sequences(df2) if x.assignment_id == 2][0]
    assert MS._prefix(st2, q2, 3)[1] == 5 + 3                      # the earlier assignment's 5 rows plus the 3 rows of this sequence
    big = frame([(1, 1, i, "a", i % 2) for i in range(300)] + [(1, 2, 1000 + i, "a", 1) for i in range(12)])
    stb = MS.Streams(big, VOCAB)
    qb = [x for x in build_sequences(big) if x.assignment_id == 2][0]
    assert MS._prefix(stb, qb, 4)[1] == MS.HIST                    # capped at the most recent 90 rows


def test_no_other_rows_equals_single_skill_code():
    df = single_student_df()
    st = MS.Streams(df, VOCAB)
    seqs = [q for q in build_sequences(df)]
    dkt, sk = models()
    ms_sak = MS.MSSAKT(3, 16).eval()
    ms_sak.load_state_dict(sk.state_dict())
    ms_dkt = MS.MSDKT(3, 16).eval()
    ms_dkt.load_state_dict(dkt.state_dict())
    for k in (3, 5, 8):
        qs = [q for q in seqs if q.n >= k and q.m > k]
        if not qs:
            continue
        a = D.risk_scores(dkt, VOCAB, qs, k)
        b = MS.risk_scores("dkt", ms_dkt, st, VOCAB, qs, k)
        np.testing.assert_allclose(a, b, atol=1e-6)
        c = SK.risk_scores(sk, VOCAB, qs, k)
        d = MS.risk_scores("sakt", ms_sak, st, VOCAB, qs, k)
        np.testing.assert_allclose(c, d, atol=1e-6)


def brute(kind, model, toks, kc, run0, k, T=10):
    """P(no mastery in k+1..T) by explicit recursion over the outcome tree, re-running the whole model on each path."""
    def p_next(tl):
        t = torch.tensor(tl, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            if kind == "dkt":
                o, _ = model.lstm(model.emb(t))
                return float(torch.sigmoid(model.out(o[:, -1])[0, kc]))
            return float(torch.sigmoid(model.predict_next(torch.nn.functional.pad(t, (0, 1)), torch.tensor([len(tl)]), torch.tensor([kc]))[0]))

    def rec(step, tl, run, mass):
        if step > T:
            return mass
        p = p_next(tl)
        tot = 0.0
        run_c = run + 1
        if run_c < MASTERY_RUN:
            tot += rec(step + 1, tl + [kc * 2 + 1], run_c, mass * p)
        if step == T and run_c >= MASTERY_RUN:
            pass
        tot += rec(step + 1, tl + [kc * 2], 0, mass * (1 - p))
        return tot
    return rec(k + 1, list(toks), run0, 1.0)


def test_interleaved_history_equals_brute_force():
    df = interleaved_df()
    st = MS.Streams(df, VOCAB)
    q = [s for s in build_sequences(df) if s.assignment_id == 1][0]
    dkt, sk = models(1)
    ms_dkt = MS.MSDKT(3, 16).eval(); ms_dkt.load_state_dict(dkt.state_dict())
    ms_sak = MS.MSSAKT(3, 16).eval(); ms_sak.load_state_dict(sk.state_dict())
    for k in (3, 5, 8):
        if not (q.n >= k and q.m > k):
            continue
        toks, _ = MS._prefix(st, q, k)
        ref_d = brute("dkt", ms_dkt, toks, VOCAB[q.kc], MS._run0(q, k), k)
        ref_s = brute("sakt", ms_sak, toks, VOCAB[q.kc], MS._run0(q, k), k)
        np.testing.assert_allclose(MS.risk_scores("dkt", ms_dkt, st, VOCAB, [q], k)[0], ref_d, atol=1e-6)
        np.testing.assert_allclose(MS.risk_scores("sakt", ms_sak, st, VOCAB, [q], k)[0], ref_s, atol=1e-6)


def test_windows_cover_every_row_and_val_windows_are_disjoint():
    df = frame([(1, 1, i, "a", i % 2) for i in range(230)])
    st = MS.Streams(df, VOCAB)
    tw = MS.train_windows(st, [1])
    assert tw[0] == (1, 0, 100) and tw[1] == (1, 50, 150) and tw[-1][2] == 230
    covered = set()
    for _, s, e in tw:
        covered |= set(range(s, e))
    assert covered == set(range(230))
    vw = MS.val_windows(st, [1])
    assert vw == [(1, 0, 100), (1, 100, 200), (1, 200, 230)]
