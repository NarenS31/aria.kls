"""Amendment A9: per-turn model (k = 1..9) with response time versus the BKT risk (exploratory, validation only; SPEC.md A9)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import baselines as B
from . import labels as L
from . import paired
from .constants import RAW_CSV

KS = tuple(range(1, 10))
STATUS = "exploratory, validation only (A9)"
BOOT_DRAWS, BOOT_SEED, C_FIXED = 1000, 20261025, 1.0
FEATURE_SETS = {"full": ("bkt", "wrong_last3", "hints", "rt_rank"), "no_rt": ("bkt", "wrong_last3", "hints"), "rt_only": ("rt_rank",)}


@dataclass
class XSeq:
    user_id: int
    kc: str
    correct: np.ndarray
    hints: np.ndarray
    rt: np.ndarray            # raw ms_first_response per opportunity (may be <= 0)
    n: int
    m: float


def load_sequences_with_time(df_clean, path=RAW_CSV):
    """df_clean: the cleaned frame from loader.load_raw (checksum already verified). Adds ms_first_response by (user_id, order_id)."""
    t = pd.read_csv(path, encoding="latin-1", usecols=["user_id", "order_id", "ms_first_response"], low_memory=False)
    if t.duplicated(["user_id", "order_id"]).any():
        raise RuntimeError("(user_id, order_id) not unique in the response-time columns")
    df = df_clean.merge(t, on=["user_id", "order_id"], how="left", validate="one_to_one")
    if df.ms_first_response.isna().any():
        raise RuntimeError("response time missing after merge")
    out = []
    for (u, a), g in df.groupby(["user_id", "assignment_id"], sort=False):
        g = g.sort_values("order_id", kind="stable")
        c = g.correct.to_numpy(dtype=np.int8)
        out.append(XSeq(int(u), str(g.skill_id.iloc[0]), c, g.hint_count.to_numpy(dtype=float), g.ms_first_response.to_numpy(dtype=float),
                        len(c), L.first_mastery(c)))
    return out


class RankLogRT:
    """Mid-rank ECDF of log response time fitted on TRAIN only (values <= 0 are missing: set to the train median of the log first)."""

    def fit(self, train_seqs):
        v = np.concatenate([q.rt for q in train_seqs])
        pos = v[v > 0]
        self.median = float(np.median(np.log(pos)))
        self.sorted = np.sort(np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), self.median))
        return self

    def transform(self, rt):
        lg = np.where(rt > 0, np.log(np.where(rt > 0, rt, 1.0)), self.median)
        lo = np.searchsorted(self.sorted, lg, side="left")
        hi = np.searchsorted(self.sorted, lg, side="right")
        return (lo + hi) / 2.0 / len(self.sorted)


def eligible(seqs, k):
    """Variant A, D3, T = 10 (A9.1): n >= k, not mastered by k, labelled."""
    rows = []
    for q in seqs:
        if q.n >= k and q.m > k:
            y = L.label(q.n, q.m, "A")
            if y is not None:
                rows.append((q, y))
    return rows


def feature_matrix(rows, k, bkt, rank):
    X = np.zeros((len(rows), 4))
    for r, (q, _) in enumerate(rows):
        x = q.correct[:k]
        X[r, 0] = _bkt_score(bkt, q, k)
        X[r, 1] = float((1 - x[-3:]).sum())
        X[r, 2] = float(q.hints[:k].sum())
        X[r, 3] = float(rank.transform(np.array([q.rt[k - 1]]))[0])
    return X


def _bkt_score(bkt, q, k):
    class _S:                                   # B.BKT.score reads .kc and .correct only
        pass
    s = _S()
    s.kc, s.correct = q.kc, q.correct
    return bkt.score(s, k)


def run(train, val, min_seq, log=print):
    bkt = B.BKT(min_seq).fit(train)
    rank = RankLogRT().fit(train)
    out = {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 9", "ks": list(KS), "C": C_FIXED, "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED},
           "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()}, "rt_rank_train_median_log_ms": rank.median, "per_k": {}}
    cols = {"bkt": 0, "wrong_last3": 1, "hints": 2, "rt_rank": 3}
    for k in KS:
        tr, va = eligible(train, k), eligible(val, k)
        Xt, yt = feature_matrix(tr, k, bkt, rank), np.array([y for _, y in tr])
        Xv, yv = feature_matrix(va, k, bkt, rank), np.array([y for _, y in va])
        sid = np.array([q.user_id for q, _ in va])
        row = {"k": k, "in_frozen_policy_window": k >= 3, "n_train": int(len(yt)), "n_val": int(len(yv)), "val_students": int(len(set(sid.tolist()))),
               "val_base_rate": float(yv.mean()) if len(yv) else None, "auc_bkt_raw": None, "models": {}}
        if len(set(yv.tolist())) < 2 or len(set(yt.tolist())) < 2:
            row["note"] = "one class only: not scored"
            out["per_k"][str(k)] = row
            continue
        s_bkt = Xv[:, 0]
        row["auc_bkt_raw"] = float(roc_auc_score(yv, s_bkt))
        for name, feats in FEATURE_SETS.items():
            idx = [cols[f] for f in feats]
            sc = StandardScaler().fit(Xt[:, idx])
            clf = LogisticRegression(C=C_FIXED, max_iter=2000).fit(sc.transform(Xt[:, idx]), yt)
            s_new = clf.predict_proba(sc.transform(Xv[:, idx]))[:, 1]
            d = paired.auc_diff(yv, sid, s_new, s_bkt, BOOT_DRAWS, BOOT_SEED)
            row["models"][name] = {"auc": float(roc_auc_score(yv, s_new)), "auc_minus_bkt": d["diff"], "ci95": d["ci95"], "draws": d["draws"],
                                   "coefficients_standardised": {f: float(c) for f, c in zip(feats, clf.coef_[0])}}
        out["per_k"][str(k)] = row
        log(f"k={k}: n_val={len(yv)} auc bkt {row['auc_bkt_raw']:.3f}; full {row['models']['full']['auc']:.3f} "
            f"(diff {row['models']['full']['auc_minus_bkt']:+.4f} [{row['models']['full']['ci95'][0]:+.4f}, {row['models']['full']['ci95'][1]:+.4f}])")
    f3, f5 = out["per_k"]["3"]["models"].get("full"), out["per_k"]["5"]["models"].get("full")
    ok = bool(f3 and f5 and all(m["auc_minus_bkt"] >= 0.01 and m["ci95"][0] > 0 for m in (f3, f5)))
    out["promotion_rule"] = {"rule": ">= +0.01 AUC with 95% interval lower bound > 0 at both k = 3 and k = 5 (A3.5 / A5.4 / A9.3)", "full_model_promoted": ok}
    return out
