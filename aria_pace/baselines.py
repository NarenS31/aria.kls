"""Baselines (SPEC 5): always-help, three-wrong-in-a-row, PFA, BKT.

Every scorer returns P(wheel-spinning)-style risk in [0, 1] from the first k opportunities only.
PFA and BKT risk = P(no mastery in opportunities k+1..T | first k responses), computed exactly by DP.
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .constants import MASTERY_RUN, T_DETERMINE


# ----------------------------------------------------------------------------- simple rules
class AlwaysHelp:
    name = "always_help"

    def fit(self, seqs):
        return self

    def score(self, seq, k):
        return 1.0


class ThreeWrongInARow:
    name = "three_wrong_rule"

    def fit(self, seqs):
        return self

    def score(self, seq, k):
        run = 0
        for x in seq.correct[:k]:
            run = run + 1 if x == 0 else 0
            if run >= 3:
                return 1.0
        return 0.0


# ----------------------------------------------------------------------------- PFA
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class PFA:
    """Pavlik, Cen & Koedinger (2009) Eq. 3 with the deviations listed in SPEC 5."""
    name = "pfa"

    def __init__(self, C=1.0):
        self.C = C

    def _design(self, kc_idx, s, f):
        n, K = len(s), self.K
        rows = np.arange(n)
        glob = sp.csr_matrix(np.column_stack([np.ones(n), s, f]))
        known = kc_idx >= 0
        r = rows[known]
        j = kc_idx[known]
        data_i = np.ones(len(r))
        ind = sp.csr_matrix((data_i, (r, j)), shape=(n, K))
        ind_s = sp.csr_matrix((s[known], (r, j)), shape=(n, K))
        ind_f = sp.csr_matrix((f[known], (r, j)), shape=(n, K))
        return sp.hstack([glob, ind, ind_s, ind_f], format="csr")

    def _rows(self, seqs):
        kc, s_, f_, y = [], [], [], []
        for q in seqs:
            c = q.correct
            cs = np.concatenate([[0], np.cumsum(c)[:-1]])
            idx = self.kc_map.get(q.kc, -1)
            kc.append(np.full(len(c), idx))
            s_.append(cs)
            f_.append(np.arange(len(c)) - cs)
            y.append(c)
        return (np.concatenate(kc), np.concatenate(s_).astype(float),
                np.concatenate(f_).astype(float), np.concatenate(y))

    def fit(self, seqs):
        self.kc_map = {k: i for i, k in enumerate(sorted({q.kc for q in seqs}))}
        self.K = len(self.kc_map)
        kc, s, f, y = self._rows(seqs)
        X = self._design(kc, s, f)
        self.clf = LogisticRegression(C=self.C, fit_intercept=False, solver="lbfgs", max_iter=3000)
        self.clf.fit(X, y)
        w = self.clf.coef_[0]
        K = self.K
        self.glob = w[:3]
        self.kc_w = np.stack([w[3:3 + K], w[3 + K:3 + 2 * K], w[3 + 2 * K:3 + 3 * K]], axis=1)  # beta, gamma, rho
        return self

    def params(self, kc):
        j = self.kc_map.get(kc)
        return self.glob if j is None else self.glob + self.kc_w[j]

    def next_response_logloss(self, seqs):
        kc, s, f, y = self._rows(seqs)
        p = np.clip(self.clf.predict_proba(self._design(kc, s, f))[:, 1], 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def score(self, seq, k):
        beta, gamma, rho = self.params(seq.kc)
        c = seq.correct[:k]
        s0 = int(c.sum())
        r0 = 0
        for x in c[::-1]:  # current correct run at k
            if x == 1:
                r0 += 1
            else:
                break

        @lru_cache(maxsize=None)
        def no_master(t, s, r):  # t responses seen, s correct among them, current run r (< MASTERY_RUN)
            if t >= T_DETERMINE:
                return 1.0
            p = _sigmoid(beta + gamma * s + rho * (t - s))
            win = 0.0 if r + 1 >= MASTERY_RUN else no_master(t + 1, s + 1, r + 1)
            return p * win + (1 - p) * no_master(t + 1, s, 0)

        return float(no_master(k, s0, r0))


# ----------------------------------------------------------------------------- BKT
BKT_BOUNDS = {"prior": (0.01, 0.99), "learn": (0.01, 0.60), "guess": (0.01, 0.30), "slip": (0.01, 0.30)}
BKT_INIT = {"prior": 0.30, "learn": 0.20, "guess": 0.15, "slip": 0.10}
BKT_ITERS = 30


def _clip(name, v):
    lo, hi = BKT_BOUNDS[name]
    return float(min(max(v, lo), hi))


def _em_stats(C, par):
    """Scaled forward-backward on an equal-length group C (g x L, 0/1). Returns sufficient statistics."""
    g, L = C.shape
    pi, ln, gu, sl = par["prior"], par["learn"], par["guess"], par["slip"]
    e0 = np.where(C == 1, gu, 1 - gu)
    e1 = np.where(C == 1, 1 - sl, sl)
    a0 = np.zeros((L, g)); a1 = np.zeros((L, g)); cs = np.zeros((L, g))
    x0, x1 = (1 - pi) * e0[:, 0], pi * e1[:, 0]
    cs[0] = x0 + x1; a0[0], a1[0] = x0 / cs[0], x1 / cs[0]
    for t in range(1, L):
        p0 = a0[t - 1] * (1 - ln)
        p1 = a1[t - 1] + a0[t - 1] * ln
        x0, x1 = p0 * e0[:, t], p1 * e1[:, t]
        cs[t] = x0 + x1; a0[t], a1[t] = x0 / cs[t], x1 / cs[t]
    b0 = np.ones((L, g)); b1 = np.ones((L, g))
    for t in range(L - 2, -1, -1):
        n1 = e1[:, t + 1] * b1[t + 1]
        b0[t] = ((1 - ln) * e0[:, t + 1] * b0[t + 1] + ln * n1) / cs[t + 1]
        b1[t] = n1 / cs[t + 1]
    g0 = a0 * b0; g1 = a1 * b1
    Ct = C.T
    st = {"ll": float(np.log(cs).sum()), "pi_num": float(g1[0].sum()), "n": g}
    if L > 1:
        xi = a0[:-1] * ln * e1[:, 1:].T * b1[1:] / cs[1:]
        st["ln_num"] = float(xi.sum()); st["ln_den"] = float(g0[:-1].sum())
    else:
        st["ln_num"] = st["ln_den"] = 0.0
    st["gu_num"] = float((g0 * Ct).sum()); st["gu_den"] = float(g0.sum())
    st["sl_num"] = float((g1 * (1 - Ct)).sum()); st["sl_den"] = float(g1.sum())
    return st


def fit_bkt_em(arrays, iters=BKT_ITERS):
    """EM for one KC. arrays: list of 1-D 0/1 arrays. Fixed init, fixed bounds, deterministic."""
    by_len = {}
    for a in arrays:
        by_len.setdefault(len(a), []).append(a)
    groups = [np.stack(v).astype(float) for v in by_len.values()]
    par = dict(BKT_INIT)
    for _ in range(iters):
        tot = {"pi_num": 0.0, "n": 0, "ln_num": 0.0, "ln_den": 0.0, "gu_num": 0.0, "gu_den": 0.0,
               "sl_num": 0.0, "sl_den": 0.0}
        for C in groups:
            st = _em_stats(C, par)
            for key in tot:
                tot[key] += st[key]
        par = {
            "prior": _clip("prior", tot["pi_num"] / tot["n"]),
            "learn": _clip("learn", tot["ln_num"] / max(tot["ln_den"], 1e-12)),
            "guess": _clip("guess", tot["gu_num"] / max(tot["gu_den"], 1e-12)),
            "slip": _clip("slip", tot["sl_num"] / max(tot["sl_den"], 1e-12)),
        }
    return par


class BKT:
    name = "bkt"

    def __init__(self, min_seq=10):
        self.min_seq = min_seq

    def fit(self, seqs, kc_fits=None):
        """kc_fits: optional precomputed {kc: (n_train_sequences, params)} + key None for pooled params."""
        if kc_fits is None:
            kc_fits = fit_all_kcs(seqs)
        self.kc_fits = kc_fits
        return self

    def params(self, kc):
        n, par = self.kc_fits.get(kc, (0, None))
        return par if (par is not None and n >= self.min_seq) else self.kc_fits[None][1]

    @staticmethod
    def _filter(c, par):
        """Return (posterior P(known) after each observation, prior P(known) before each observation)."""
        pk = par["prior"]
        post, prior = [], []
        for x in c:
            prior.append(pk)
            e1 = (1 - par["slip"]) if x == 1 else par["slip"]
            e0 = par["guess"] if x == 1 else (1 - par["guess"])
            pk = pk * e1 / (pk * e1 + (1 - pk) * e0)
            post.append(pk)
            pk = pk + (1 - pk) * par["learn"]
        return np.array(post), np.array(prior)

    def next_response_logloss(self, seqs):
        ll, n = 0.0, 0
        for q in seqs:
            par = self.params(q.kc)
            _, prior = self._filter(q.correct, par)
            p = np.clip(prior * (1 - par["slip"]) + (1 - prior) * par["guess"], 1e-6, 1 - 1e-6)
            y = q.correct
            ll += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()); n += len(y)
        return ll / n

    def score(self, seq, k):
        par = self.params(seq.kc)
        c = seq.correct[:k]
        post, _ = self._filter(c, par)
        r0 = 0
        for x in c[::-1]:
            if x == 1:
                r0 += 1
            else:
                break
        pk = post[-1]
        d = np.zeros((2, MASTERY_RUN))   # [latent known?, current run] mass, not yet mastered
        d[1, r0] = pk; d[0, r0] = 1 - pk
        for _ in range(k + 1, T_DETERMINE + 1):
            nd = np.zeros_like(d)
            for ls in (0, 1):
                for r in range(MASTERY_RUN):
                    m = d[ls, r]
                    if m == 0:
                        continue
                    for ls2, pt in ((1, 1.0 if ls == 1 else par["learn"]), (0, 0.0 if ls == 1 else 1 - par["learn"])):
                        if pt == 0:
                            continue
                        pc = (1 - par["slip"]) if ls2 == 1 else par["guess"]
                        if r + 1 < MASTERY_RUN:
                            nd[ls2, r + 1] += m * pt * pc
                        nd[ls2, 0] += m * pt * (1 - pc)
            d = nd
        return float(d.sum())


def fit_all_kcs(seqs, min_fit=10):
    """Fit BKT per KC with >= min_fit training sequences (min_seq only selects among these) and a pooled fit."""
    by_kc = {}
    for q in seqs:
        by_kc.setdefault(q.kc, []).append(q.correct)
    fits = {None: (len(seqs), fit_bkt_em([q.correct for q in seqs]))}
    for kc, arrs in sorted(by_kc.items()):
        if len(arrs) >= min_fit:
            fits[kc] = (len(arrs), fit_bkt_em(arrs))
        else:
            fits[kc] = (len(arrs), None)
    return fits
