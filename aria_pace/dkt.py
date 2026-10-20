"""Deep knowledge tracing (LSTM) for ARIA-Pace, as specified in SPEC Amendment 3 (A3.5).

* Vocabulary: KC strings with >= MIN_KC training sequences get ids 1..K; everything else shares id 0 (UNK).
* Step input token = kc * 2 + correct; embedding -> one LSTM layer -> dropout -> linear over all KCs. The prediction for the next step
  reads the output of that step's own KC; every step of a sequence uses the sequence's KC (as BKT and PFA do).
* Training uses the first MAX_LEN opportunities. Hidden size and stopping epoch are chosen by VALIDATION next-response log-loss only.
* Risk score = P(no mastery in opportunities k+1..10 | first k responses), computed EXACTLY by enumerating the tree of future outcomes
  (at most 2**(10-k) paths, a path is dropped when three correct in a row complete) with the LSTM state carried along each path.
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .constants import MASTERY_RUN, T_DETERMINE

MAX_LEN = 100
MIN_KC = 10
EMBED = 32
DROPOUT = 0.2
LR = 1e-3
BATCH = 64
MAX_EPOCHS = 40
PATIENCE = 4
HIDDEN_GRID = (32, 64, 128)
SEED = 20261023
THREADS = 4


def setup_torch(seed: int = SEED):
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def build_vocab(train_seqs, min_kc: int = MIN_KC) -> dict:
    """kc string -> id (1..K). Anything absent (rare in training, or unseen) is id 0 via vocab.get(kc, 0)."""
    counts = Counter(q.kc for q in train_seqs)
    return {kc: i + 1 for i, kc in enumerate(sorted(k for k, n in counts.items() if n >= min_kc))}


class DKT(nn.Module):
    def __init__(self, n_kc: int, hidden: int):
        super().__init__()
        self.n_kc, self.hidden = n_kc, hidden               # n_kc includes UNK (id 0)
        self.emb = nn.Embedding(2 * n_kc, EMBED)
        self.lstm = nn.LSTM(EMBED, hidden, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(hidden, n_kc)

    def forward(self, tokens):                              # (B, L) -> logits (B, L, n_kc)
        o, _ = self.lstm(self.emb(tokens))
        return self.out(self.drop(o))


# ----------------------------------------------------------------------------- data
def _arrays(seqs, vocab, max_len=MAX_LEN):
    """Padded tokens (B, L), correctness (B, L), mask (B, L), kc ids (B,)."""
    L = min(max(q.n for q in seqs), max_len)
    B = len(seqs)
    tok = np.zeros((B, L), dtype=np.int64)
    cor = np.zeros((B, L), dtype=np.float32)
    msk = np.zeros((B, L), dtype=bool)
    kc = np.zeros(B, dtype=np.int64)
    for b, q in enumerate(seqs):
        n = min(q.n, max_len)
        kc[b] = vocab.get(q.kc, 0)
        c = q.correct[:n].astype(np.int64)
        tok[b, :n] = kc[b] * 2 + c
        cor[b, :n] = c
        msk[b, :n] = True
    return torch.from_numpy(tok), torch.from_numpy(cor), torch.from_numpy(msk), torch.from_numpy(kc)


def _batches(seqs, shuffle_seed=None, batch=BATCH):
    """Batches of similar length (less padding). Order shuffled with a seeded generator when training."""
    order = sorted(range(len(seqs)), key=lambda i: min(seqs[i].n, MAX_LEN))
    chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(chunks)
    return chunks


def _loss(model, seqs, vocab, idx):
    tok, cor, msk, kc = _arrays([seqs[i] for i in idx], vocab)
    if tok.shape[1] < 2:
        return None, 0
    logits = model(tok[:, :-1])                                   # output at t predicts step t+1
    pred = logits.gather(2, kc.view(-1, 1, 1).expand(-1, logits.shape[1], 1)).squeeze(2)
    m = msk[:, 1:]
    loss = nn.functional.binary_cross_entropy_with_logits(pred[m], cor[:, 1:][m], reduction="sum")
    return loss, int(m.sum())


@torch.no_grad()
def next_response_logloss(model, seqs, vocab) -> float:
    model.eval()
    tot, n = 0.0, 0
    for idx in _batches(seqs, None, 256):
        loss, cnt = _loss(model, seqs, vocab, idx)
        if loss is not None:
            tot += float(loss); n += cnt
    return tot / max(n, 1)


def train_one(train_seqs, val_seqs, vocab, hidden, seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE, log=None):
    """Train one hidden size. Returns (best model, info). Early stopping on validation next-response log-loss."""
    setup_torch(seed)
    model = DKT(len(vocab) + 1, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in _batches(train_seqs, shuffle_seed=seed + epoch):
            loss, cnt = _loss(model, train_seqs, vocab, idx)
            if loss is None:
                continue
            opt.zero_grad()
            (loss / cnt).backward()
            opt.step()
        v = next_response_logloss(model, val_seqs, vocab)
        history.append(round(v, 6))
        if log:
            log(f"  hidden={hidden} epoch={epoch} val_next_response_logloss={v:.5f}")
        if v < best - 1e-5:
            best, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"hidden": hidden, "best_epoch": best_epoch, "val_next_response_logloss": best, "epochs_run": len(history), "val_curve": history}


# ----------------------------------------------------------------------------- risk by exact enumeration
@torch.no_grad()
def _risk_chunk(model, kc_ids, prefix, run0, k, T=T_DETERMINE):
    """Returns (alive_mass, mastered_mass) per sequence: P(no mastery in k+1..T) and P(mastery completed in k+1..T); they sum to 1."""
    N = len(kc_ids)
    o, (h, c) = model.lstm(model.emb(prefix))
    o_last, h, c = o[:, -1, :], h[0], c[0]
    idx = torch.arange(N)
    kc_path = kc_ids.clone()
    mass = torch.ones(N, dtype=torch.float64)
    run = run0.clone()
    alive = torch.zeros(N, dtype=torch.float64)
    mastered = torch.zeros(N, dtype=torch.float64)
    for t in range(k + 1, T + 1):
        logits = model.out(o_last)
        p = torch.sigmoid(logits.gather(1, kc_path.view(-1, 1)).squeeze(1)).double()
        m_c, m_w = mass * p, mass * (1 - p)
        run_c = run + 1
        done = run_c >= MASTERY_RUN
        mastered.index_add_(0, idx[done], m_c[done])
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
    return alive, mastered


def risk_scores(model, vocab, seqs, k, chunk=400, return_mastered=False):
    """P(no mastery in opportunities k+1..10) for each sequence, from its first k responses only. All sequences need n >= k and m > k."""
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
