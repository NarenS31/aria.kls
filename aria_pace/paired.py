"""Paired student-level bootstrap for differences of AUC (SPEC A3.4, A3.5)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _by_student(sid):
    uniq, inv = np.unique(np.asarray(sid), return_inverse=True)
    return uniq, [np.where(inv == i)[0] for i in range(len(uniq))]


def auc_diff(y, sid, s_new, s_old, n_boot=1000, seed=0):
    y = np.asarray(y)
    uniq, by = _by_student(sid)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([by[i] for i in rng.randint(0, len(uniq), len(uniq))])
        if len(set(y[idx].tolist())) < 2:
            continue
        d.append(roc_auc_score(y[idx], s_new[idx]) - roc_auc_score(y[idx], s_old[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, s_new) - roc_auc_score(y, s_old)), "ci95": [float(lo), float(hi)], "draws": len(d)}


def mean_auc_diff(sets, n_boot=1000, seed=0):
    """sets: {k: (y, sid, s_new, s_old)}. The SAME student resample is used for every k; the statistic is the mean over k of AUC(new)-AUC(old)."""
    all_students = np.unique(np.concatenate([np.asarray(v[1]) for v in sets.values()]))
    per = {}
    for k, (y, sid, a, b) in sets.items():
        sid = np.asarray(sid)
        order = np.argsort(sid, kind="stable")
        per[k] = (np.asarray(y)[order], sid[order], np.asarray(a)[order], np.asarray(b)[order])
    rng = np.random.RandomState(seed)
    point = float(np.mean([roc_auc_score(v[0], v[2]) - roc_auc_score(v[0], v[3]) for v in per.values()]))
    cuts = {k: {s: (np.searchsorted(v[1], s, "left"), np.searchsorted(v[1], s, "right")) for s in all_students} for k, v in per.items()}
    out = []
    for _ in range(n_boot):
        draw = all_students[rng.randint(0, len(all_students), len(all_students))]
        ds = []
        for k, v in per.items():
            idx = np.concatenate([np.arange(*cuts[k][s]) for s in draw])
            if len(idx) == 0 or len(set(v[0][idx].tolist())) < 2:
                break
            ds.append(roc_auc_score(v[0][idx], v[2][idx]) - roc_auc_score(v[0][idx], v[3][idx]))
        else:
            out.append(float(np.mean(ds)))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"mean_diff": point, "ci95": [float(lo), float(hi)], "draws": len(out)}
