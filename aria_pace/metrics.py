"""Metrics (SPEC 6)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def brier(y, s):
    return float(np.mean((np.asarray(s) - np.asarray(y)) ** 2))


def logloss(y, s):
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS); y = np.asarray(y)
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))


def reliability(y, s, bins=10):
    """Equal-mass bins when the score has >= `bins` distinct values; otherwise one bin per distinct value."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(np.unique(s)) < bins:
        groups = [np.where(s == v)[0] for v in np.unique(s)]
    else:
        order = np.argsort(s, kind="stable")
        groups = [g for g in np.array_split(order, bins) if len(g)]
    rows = [{"mean_score": float(s[g].mean()), "obs_rate": float(y[g].mean()), "n": int(len(g))} for g in groups]
    ece = float(sum(r["n"] * abs(r["mean_score"] - r["obs_rate"]) for r in rows) / len(y))
    return ece, rows


def calibration_line(y, s):
    """Logistic recalibration y ~ a + b*logit(s). Ideal: intercept 0, slope 1. None for degenerate scores."""
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    if len(np.unique(s)) < 10 or len(set(np.asarray(y).tolist())) < 2:
        return None
    z = np.log(s / (1 - s)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0][0])}


def bootstrap_auc_ci(y, s, student_ids, n_boot, seed):
    """95% percentile CI of AUC, resampling students (not sequences) with replacement."""
    y = np.asarray(y); s = np.asarray(s); sid = np.asarray(student_ids)
    uniq, inv = np.unique(sid, return_inverse=True)
    idx_by_student = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_student[i] for i in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def summarize(y, s, student_ids, n_boot, seed, constant_ok=True):
    y = np.asarray(y); s = np.asarray(s, float)
    ece, rel = reliability(y, s)
    out = {
        "n": int(len(y)), "n_students": int(len(set(np.asarray(student_ids).tolist()))),
        "positives": int(y.sum()), "base_rate": float(y.mean()),
        "auc": auc(y, s), "brier": brier(y, s), "logloss": logloss(y, s),
        "ece_10bin": ece, "reliability": rel, "calibration_line": calibration_line(y, s),
        "score_mean": float(s.mean()), "n_distinct_scores": int(len(np.unique(s))),
    }
    if len(np.unique(s)) > 1 and len(set(y.tolist())) > 1:
        ci, used = bootstrap_auc_ci(y, s, student_ids, n_boot, seed)
        out["auc_ci95_student_bootstrap"] = ci
        out["bootstrap_resamples_used"] = used
    else:
        out["auc_ci95_student_bootstrap"] = None
    return out
