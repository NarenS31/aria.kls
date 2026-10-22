"""Amendment A5: multi-skill DKT (LSTM) and SAKT (one attention block). See SPEC.md A5 for every design choice.

Each student's cleaned rows in order_id order form ONE stream of tokens kc*2+correct (kc = that row's own skill, A5.2). Models are trained on windows of
100 rows (stride 50) and predict the next response of every step from the earlier steps of the window. The wheel-spinning risk of a sequence at decision
point k is P(no mastery in opportunities k+1..10 | its first k responses and the student's earlier rows), computed EXACTLY by enumerating the future
outcome tree with the sequence's own KC for the future steps (as dkt.py / sakt.py do), from the most recent <= 90 rows up to and including the k-th
opportunity (A5.1). Nothing here edits dkt.py, sakt.py or any existing module.
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from . import dkt as D
from .constants import MASTERY_RUN, T_DETERMINE

WIN = 100            # training / validation window length (A5.3)
STRIDE = 50          # training stride
HIST = 90            # inference history cap (A5.1)
HEADS = 2
MIN_GROUPS = 10      # vocabulary: skill strings in >= 10 training (student, assignment) groups (A5.2)


# ----------------------------------------------------------------------------- streams
class Streams:
    """Per-student token streams + the stream positions of every (student, assignment) sequence."""

    def __init__(self, df, vocab):
        self.kc, self.cor, self.nonfirst, self.pos = {}, {}, {}, {}
        for uid, g in df.groupby("user_id", sort=False):
            self.kc[uid] = np.array([vocab.get(s, 0) for s in g.skill_id.to_numpy()], dtype=np.int64)
            self.cor[uid] = g.correct.to_numpy(dtype=np.int64)
        p = df.assign(_p=df.groupby("user_id").cumcount())
        for (u, a), g in p.groupby(["user_id", "assignment_id"], sort=False):
            self.pos[(int(u), int(a))] = g._p.to_numpy()

    def tokens(self, uid):
        return self.kc[uid] * 2 + self.cor[uid]


def build_vocab(df, train_ids, min_groups=MIN_GROUPS) -> dict:
    """Skill strings occurring in >= min_groups distinct training (student, assignment) groups (any row of the group)."""
    sub = df[df.user_id.isin(train_ids)]
    n = sub.drop_duplicates(["user_id", "assignment_id", "skill_id"]).groupby("skill_id").size()
    return {s: i + 1 for i, s in enumerate(sorted(n[n >= min_groups].index))}


def train_windows(streams, uids):
    """(uid, start, end) windows of WIN rows with stride STRIDE (last one may be shorter); only windows with >= 2 rows."""
    out = []
    for u in uids:
        n = len(streams.kc[u])
        s = 0
        while s < n - 1:
            out.append((u, s, min(s + WIN, n)))
            if s + WIN >= n:
                break
            s += STRIDE
    return out


def val_windows(streams, uids):
    """Non-overlapping windows of WIN rows (a row's context = the earlier rows of its own window)."""
    out = []
    for u in uids:
        n = len(streams.kc[u])
        for s in range(0, n, WIN):
            if min(s + WIN, n) - s >= 2:
                out.append((u, s, min(s + WIN, n)))
    return out


def _batch(streams, wins, idx):
    L = max(wins[i][2] - wins[i][1] for i in idx)
    B = len(idx)
    tok = np.zeros((B, L), dtype=np.int64)
    kc = np.zeros((B, L), dtype=np.int64)
    cor = np.zeros((B, L), dtype=np.float32)
    msk = np.zeros((B, L), dtype=bool)
    for b, i in enumerate(idx):
        u, s, e = wins[i]
        n = e - s
        tok[b, :n] = streams.tokens(u)[s:e]
        kc[b, :n] = streams.kc[u][s:e]
        cor[b, :n] = streams.cor[u][s:e]
        msk[b, :n] = True
    return torch.from_numpy(tok), torch.from_numpy(kc), torch.from_numpy(cor), torch.from_numpy(msk)


def _chunks(wins, shuffle_seed=None, batch=D.BATCH):
    order = sorted(range(len(wins)), key=lambda i: wins[i][2] - wins[i][1])
    ch = [order[i:i + batch] for i in range(0, len(order), batch)]
    if shuffle_seed is not None:
        np.random.RandomState(shuffle_seed).shuffle(ch)
    return ch


# ----------------------------------------------------------------------------- models
class MSDKT(D.DKT):
    """dkt.DKT; used with per-step KCs: the prediction for step t+1 reads the output of THAT step's own KC."""

    def step_logits(self, tok, kc):                                   # tok, kc (B, L) -> logits (B, L-1) predicting steps 2..L
        out = self.forward(tok[:, :-1])
        return out.gather(2, kc[:, 1:].unsqueeze(2)).squeeze(2)


class MSSAKT(nn.Module):
    """sakt.SAKT with a per-step query KC and WIN+1 positions (A5.3)."""

    def __init__(self, n_kc: int, d: int):
        super().__init__()
        self.n_kc, self.d = n_kc, d
        self.e_int = nn.Embedding(2 * n_kc, d)
        self.e_q = nn.Embedding(n_kc, d)
        self.pos = nn.Embedding(WIN + 1, d)
        self.attn = nn.MultiheadAttention(d, HEADS, dropout=D.DROPOUT, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(D.DROPOUT), nn.Linear(d, d))
        self.ln2 = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def _head(self, q, a):
        x = self.ln1(q + a)
        return self.out(self.ln2(x + self.ff(x))).squeeze(-1)

    def step_logits(self, tok, kc, key_pad=None):                    # (B, L) -> (B, L-1)
        L = tok.shape[1]
        pos = self.pos(torch.arange(L, device=tok.device))
        keys = self.e_int(tok[:, :L - 1]) + pos[:L - 1]
        q = self.e_q(kc[:, 1:]) + pos[1:L]
        causal = torch.triu(torch.ones(L - 1, L - 1, dtype=torch.bool, device=tok.device), diagonal=1)
        a, _ = self.attn(q, keys, keys, attn_mask=causal, key_padding_mask=key_pad, need_weights=False)
        return self._head(q, a)

    def predict_next(self, toks, lens, kc):
        """toks (P, W) right-padded prefixes with true lengths lens (P,), kc (P,): logit that step lens+1 is correct."""
        W = int(lens.max())
        keys = self.e_int(toks[:, :W]) + self.pos(torch.arange(W))
        q = (self.e_q(kc) + self.pos(lens)).unsqueeze(1)
        key_pad = torch.arange(W).unsqueeze(0) >= lens.unsqueeze(1)
        a, _ = self.attn(q, keys, keys, key_padding_mask=key_pad, need_weights=False)
        return self._head(q, a)[:, 0]


def make_model(kind, n_kc, size):
    return MSDKT(n_kc, size) if kind == "dkt" else MSSAKT(n_kc, size)


def _loss(kind, model, streams, wins, idx):
    tok, kc, cor, msk = _batch(streams, wins, idx)
    if tok.shape[1] < 2:
        return None, 0
    if kind == "dkt":
        logits = model.step_logits(tok, kc)
    else:
        logits = model.step_logits(tok, kc, ~msk[:, :tok.shape[1] - 1])
    m = msk[:, 1:]
    loss = nn.functional.binary_cross_entropy_with_logits(logits[m], cor[:, 1:][m], reduction="sum")
    return loss, int(m.sum())


@torch.no_grad()
def predict_windows(kind, model, streams, wins, batch=256):
    """Dict (uid, row position in stream) -> P(correct) for every row with >= 1 earlier row in its window."""
    model.eval()
    out = {}
    for idx in _chunks(wins, None, batch):
        tok, kc, cor, msk = _batch(streams, wins, idx)
        logits = model.step_logits(tok, kc) if kind == "dkt" else model.step_logits(tok, kc, ~msk[:, :tok.shape[1] - 1])
        p = torch.sigmoid(logits).numpy()
        for b, i in enumerate(idx):
            u, s, e = wins[i]
            for t in range(1, e - s):
                out[(u, s + t)] = float(p[b, t - 1])
    return out


@torch.no_grad()
def val_logloss(kind, model, streams, wins) -> float:
    model.eval()
    tot, n = 0.0, 0
    for idx in _chunks(wins, None, 256):
        loss, cnt = _loss(kind, model, streams, wins, idx)
        if loss is not None:
            tot += float(loss); n += cnt
    return tot / max(n, 1)


def train_one(kind, train_streams, train_wins, val_streams, val_wins, vocab, size, seed=D.SEED, max_epochs=D.MAX_EPOCHS, patience=D.PATIENCE, log=None):
    D.setup_torch(seed)
    model = make_model(kind, len(vocab) + 1, size)
    opt = torch.optim.Adam(model.parameters(), lr=D.LR)
    best, best_state, best_epoch, bad, hist = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in _chunks(train_wins, shuffle_seed=seed + epoch):
            loss, cnt = _loss(kind, model, train_streams, train_wins, idx)
            if loss is None:
                continue
            opt.zero_grad()
            (loss / cnt).backward()
            opt.step()
        v = val_logloss(kind, model, val_streams, val_wins)
        hist.append(round(v, 6))
        if log:
            log(f"  {kind} size={size} epoch={epoch} val_next_response_logloss={v:.5f}")
        if v < best - 1e-5:
            best, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"size": size, "best_epoch": best_epoch, "val_next_response_logloss": best, "epochs_run": len(hist), "val_curve": hist}


# ----------------------------------------------------------------------------- risk by exact enumeration
def _prefix(streams, q, k):
    """Token / kc arrays of the student's rows up to and including the sequence's k-th opportunity, most recent HIST rows."""
    p = streams.pos[(q.user_id, q.assignment_id)][k - 1]
    s = max(0, p + 1 - HIST)
    u = q.user_id
    return streams.tokens(u)[s:p + 1], p + 1 - s


def _run0(q, k):
    r = 0
    for x in q.correct[:k][::-1]:
        if x == 1:
            r += 1
        else:
            break
    return r


@torch.no_grad()
def _dkt_chunk(model, kc_ids, tok_list, lens, run0, k, T=T_DETERMINE):
    N = len(kc_ids)
    W = int(max(lens))
    pad = torch.zeros((N, W), dtype=torch.long)
    for i, t in enumerate(tok_list):
        pad[i, :len(t)] = torch.from_numpy(t)
    packed = nn.utils.rnn.pack_padded_sequence(model.emb(pad), torch.tensor(lens), batch_first=True, enforce_sorted=False)
    _, (h, c) = model.lstm(packed)
    h, c = h[0], c[0]
    o_last = h
    idx = torch.arange(N)
    kc_path = kc_ids.clone()
    mass = torch.ones(N, dtype=torch.float64)
    run = run0.clone()
    alive = torch.zeros(N, dtype=torch.float64)
    for t in range(k + 1, T + 1):
        p = torch.sigmoid(model.out(o_last).gather(1, kc_path.view(-1, 1)).squeeze(1)).double()
        m_c, m_w = mass * p, mass * (1 - p)
        run_c = run + 1
        done = run_c >= MASTERY_RUN
        if t == T:
            alive.index_add_(0, idx, m_w)
            alive.index_add_(0, idx[~done], m_c[~done])
            break
        keep = ~done
        tokens = torch.cat([kc_path[keep] * 2 + 1, kc_path * 2])
        h_in, c_in = torch.cat([h[keep], h]), torch.cat([c[keep], c])
        o_new, (h_new, c_new) = model.lstm(model.emb(tokens).unsqueeze(1), (h_in.unsqueeze(0).contiguous(), c_in.unsqueeze(0).contiguous()))
        o_last, h, c = o_new[:, 0, :], h_new[0], c_new[0]
        idx = torch.cat([idx[keep], idx])
        mass = torch.cat([m_c[keep], m_w])
        run = torch.cat([run_c[keep], torch.zeros_like(run)])
        kc_path = kc_ids[idx]
    return alive


@torch.no_grad()
def _sakt_chunk(model, kc_ids, tok_list, lens, run0, k, T=T_DETERMINE):
    N = len(kc_ids)
    width = int(max(lens)) + (T - k)
    toks = torch.zeros((N, width), dtype=torch.long)
    for i, t in enumerate(tok_list):
        toks[i, :len(t)] = torch.from_numpy(t)
    lens = torch.tensor(lens, dtype=torch.long)
    idx = torch.arange(N)
    kc_path = kc_ids.clone()
    mass = torch.ones(N, dtype=torch.float64)
    run = run0.clone()
    alive = torch.zeros(N, dtype=torch.float64)
    for t in range(k + 1, T + 1):
        p = torch.sigmoid(model.predict_next(toks, lens, kc_path)).double()
        m_c, m_w = mass * p, mass * (1 - p)
        run_c = run + 1
        done = run_c >= MASTERY_RUN
        if t == T:
            alive.index_add_(0, idx, m_w)
            alive.index_add_(0, idx[~done], m_c[~done])
            break
        keep = ~done
        tc, lc = toks[keep].clone(), lens[keep].clone()
        tc[torch.arange(len(lc)), lc] = kc_path[keep] * 2 + 1
        tw, lw = toks.clone(), lens.clone()
        tw[torch.arange(len(lw)), lw] = kc_path * 2
        toks, lens = torch.cat([tc, tw]), torch.cat([lc + 1, lw + 1])
        idx = torch.cat([idx[keep], idx])
        mass = torch.cat([m_c[keep], m_w])
        run = torch.cat([run_c[keep], torch.zeros_like(run)])
        kc_path = kc_ids[idx]
    return alive


def risk_scores(kind, model, streams, vocab, seqs, k, chunk=None):
    """P(no mastery in opportunities k+1..10) per sequence (all need n >= k and m > k), from the student's history up to the k-th opportunity."""
    model.eval()
    chunk = chunk or (400 if kind == "dkt" else 24)
    out = []
    for s in range(0, len(seqs), chunk):
        part = seqs[s:s + chunk]
        pre = [_prefix(streams, q, k) for q in part]
        kc = torch.tensor([vocab.get(q.kc, 0) for q in part], dtype=torch.long)
        run0 = torch.tensor([_run0(q, k) for q in part], dtype=torch.long)
        fn = _dkt_chunk if kind == "dkt" else _sakt_chunk
        out.append(fn(model, kc, [p[0] for p in pre], [p[1] for p in pre], run0, k).numpy())
    return np.concatenate(out)
