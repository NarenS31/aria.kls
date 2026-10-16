"""Self-attentive knowledge tracing for ARIA-Pace, as fixed in SPEC Amendment 3a. Shares vocabulary, batching, loss and training settings with dkt.py.

One causal multi-head attention block: keys/values = embeddings of the earlier interaction tokens (kc*2 + correct) plus learned positions; the query for
predicting step s = embedding of that step's KC plus the position of s. The risk score is the same exact outcome-tree enumeration as DKT; attention keeps
no recurrent state, so each path re-reads its full token prefix.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from . import dkt as D
from .constants import MASTERY_RUN, T_DETERMINE

HEADS = 2
HIDDEN_GRID = D.HIDDEN_GRID           # model width d in {32, 64, 128}


class SAKT(nn.Module):
    def __init__(self, n_kc: int, d: int):
        super().__init__()
        self.n_kc, self.d = n_kc, d
        self.e_int = nn.Embedding(2 * n_kc, d)
        self.e_q = nn.Embedding(n_kc, d)
        self.pos = nn.Embedding(D.MAX_LEN + 1, d)
        self.attn = nn.MultiheadAttention(d, HEADS, dropout=D.DROPOUT, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(D.DROPOUT), nn.Linear(d, d))
        self.ln2 = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def _head(self, q, a):
        x = self.ln1(q + a)
        return self.out(self.ln2(x + self.ff(x))).squeeze(-1)

    def forward(self, tokens, kc, key_pad=None):
        """tokens (B, L), kc (B,). Returns logits (B, L-1): entry j predicts step j+2 from steps 1..j+1 (causal)."""
        L = tokens.shape[1]
        pos = self.pos(torch.arange(L, device=tokens.device))
        keys = self.e_int(tokens[:, :L - 1]) + pos[:L - 1]
        q = self.e_q(kc).unsqueeze(1).expand(-1, L - 1, -1) + pos[1:L]
        causal = torch.triu(torch.ones(L - 1, L - 1, dtype=torch.bool, device=tokens.device), diagonal=1)
        a, _ = self.attn(q, keys, keys, attn_mask=causal, key_padding_mask=key_pad, need_weights=False)
        return self._head(q, a)

    def predict_next(self, prefix, kc):
        """prefix (P, t) interaction tokens of steps 1..t, kc (P,). Logit that step t+1 is correct."""
        t = prefix.shape[1]
        keys = self.e_int(prefix) + self.pos(torch.arange(t))
        q = (self.e_q(kc) + self.pos(torch.tensor(t))).unsqueeze(1)
        a, _ = self.attn(q, keys, keys, need_weights=False)
        return self._head(q, a)[:, 0]


def _loss(model, seqs, vocab, idx):
    tok, cor, msk, kc = D._arrays([seqs[i] for i in idx], vocab)
    if tok.shape[1] < 2:
        return None, 0
    n = msk.sum(1)
    key_pad = ~msk[:, :tok.shape[1] - 1]                          # keys beyond the sequence's length are ignored
    logits = model(tok, kc, key_pad)
    m = msk[:, 1:]
    loss = nn.functional.binary_cross_entropy_with_logits(logits[m], cor[:, 1:][m], reduction="sum")
    return loss, int(m.sum())


@torch.no_grad()
def next_response_logloss(model, seqs, vocab) -> float:
    model.eval()
    tot, n = 0.0, 0
    for idx in D._batches(seqs, None, 256):
        loss, cnt = _loss(model, seqs, vocab, idx)
        if loss is not None:
            tot += float(loss); n += cnt
    return tot / max(n, 1)


def train_one(train_seqs, val_seqs, vocab, d, seed=D.SEED, max_epochs=D.MAX_EPOCHS, patience=D.PATIENCE, log=None):
    D.setup_torch(seed)
    model = SAKT(len(vocab) + 1, d)
    opt = torch.optim.Adam(model.parameters(), lr=D.LR)
    best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in D._batches(train_seqs, shuffle_seed=seed + epoch):
            loss, cnt = _loss(model, train_seqs, vocab, idx)
            if loss is None:
                continue
            opt.zero_grad()
            (loss / cnt).backward()
            opt.step()
        v = next_response_logloss(model, val_seqs, vocab)
        history.append(round(v, 6))
        if log:
            log(f"  d={d} epoch={epoch} val_next_response_logloss={v:.5f}")
        if v < best - 1e-5:
            best, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"hidden": d, "best_epoch": best_epoch, "val_next_response_logloss": best, "epochs_run": len(history), "val_curve": history}


@torch.no_grad()
def _risk_chunk(model, kc_ids, prefix_tokens, run0, k, T=T_DETERMINE):
    N = len(kc_ids)
    idx = torch.arange(N)
    kc_path = kc_ids.clone()
    toks = prefix_tokens.clone()                                    # (P, t) tokens of steps 1..t along each path
    mass = torch.ones(N, dtype=torch.float64)
    run = run0.clone()
    alive = torch.zeros(N, dtype=torch.float64)
    mastered = torch.zeros(N, dtype=torch.float64)
    for t in range(k + 1, T + 1):
        p = torch.sigmoid(model.predict_next(toks, kc_path)).double()
        m_c, m_w = mass * p, mass * (1 - p)
        run_c = run + 1
        done = run_c >= MASTERY_RUN
        mastered.index_add_(0, idx[done], m_c[done])
        if t == T:
            alive.index_add_(0, idx, m_w)
            alive.index_add_(0, idx[~done], m_c[~done])
            break
        keep = ~done
        new_c = torch.cat([toks[keep], (kc_path[keep] * 2 + 1).unsqueeze(1)], dim=1)
        new_w = torch.cat([toks, (kc_path * 2).unsqueeze(1)], dim=1)
        toks = torch.cat([new_c, new_w])
        idx = torch.cat([idx[keep], idx])
        mass = torch.cat([m_c[keep], m_w])
        run = torch.cat([run_c[keep], torch.zeros_like(run)])
        kc_path = kc_ids[idx]
    return alive, mastered


def risk_scores(model, vocab, seqs, k, chunk=400, return_mastered=False):
    model.eval()
    out_a, out_m = [], []
    for s in range(0, len(seqs), chunk):
        part = seqs[s:s + chunk]
        kc = torch.tensor([vocab.get(q.kc, 0) for q in part], dtype=torch.long)
        pre = torch.tensor(np.stack([q.correct[:k].astype(np.int64) for q in part]), dtype=torch.long)
        tokens = kc.view(-1, 1) * 2 + pre
        run0 = []
        for q in part:
            r = 0
            for x in q.correct[:k][::-1]:
                if x == 1: r += 1
                else: break
            run0.append(r)
        a, m = _risk_chunk(model, kc, tokens, torch.tensor(run0, dtype=torch.long), k)
        out_a.append(a.numpy()); out_m.append(m.numpy())
    a, m = np.concatenate(out_a), np.concatenate(out_m)
    return (a, m) if return_mastered else a
