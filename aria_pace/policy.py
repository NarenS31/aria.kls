"""Abstention policy replay (SPEC Section 7 and Amendment 1). All time units are opportunities."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import T_DETERMINE
from .labels import label

DEC = np.arange(3, 10)          # decision points i = 3..9
NJ = len(DEC)
PROFILES = {"quiet": (3.0, 1.0), "balanced": (1.0, 1.0), "supportive": (1.0, 3.0)}  # (w_I, w_D)
TAU_GRID = np.round(np.concatenate([np.arange(0.0, 1.0001, 0.01), [1.01]]), 2)
EPS = 1e-6


class Table:
    """Decision table for a list of sequences in scope. Arrays are (N, 7) over i = 3..9."""

    def __init__(self, seqs, include_leavers):
        items = []
        for q in seqs:
            if q.n >= 3 and q.m > 3:
                y = label(q.n, q.m, "A")
                if y is None:
                    if include_leavers:
                        items.append((q, 0, True))
                else:
                    items.append((q, y, False))
        self.items = items
        self.seqs = [t[0] for t in items]
        self.y = np.array([t[1] for t in items])
        self.leaver = np.array([t[2] for t in items])
        self.sid = np.array([t[0].user_id for t in items])
        N = len(items)
        self.valid = np.zeros((N, NJ), bool)
        self.err_primary = np.zeros((N, NJ), bool)
        self.err_all = np.zeros((N, NJ), bool)
        for r, (q, _, _) in enumerate(items):
            for j, i in enumerate(DEC):
                if i <= min(9, q.n) and q.m > i:
                    self.valid[r, j] = True
                    wrong = q.correct[i - 1] == 0
                    self.err_all[r, j] = wrong
                    self.err_primary[r, j] = wrong and q.first_action[i - 1] == 0
        self.err = {"primary": self.err_primary, "incl_hint_first": self.err_all}

    def subset(self, mask):
        t = Table.__new__(Table)
        t.items = [x for x, m in zip(self.items, mask) if m]
        t.seqs = [x[0] for x in t.items]
        for a in ("y", "leaver", "sid", "valid", "err_primary", "err_all"):
            setattr(t, a, getattr(self, a)[mask])
        t.err = {"primary": t.err_primary, "incl_hint_first": t.err_all}
        return t


def score_matrix(table, fn):
    """fn(seq, i) -> raw risk, evaluated only at valid cells."""
    S = np.full(table.valid.shape, np.nan)
    for r, q in enumerate(table.seqs):
        for j, i in enumerate(DEC):
            if table.valid[r, j]:
                S[r, j] = fn(q, int(i))
    return S


def three_wrong_fire(table, kind):
    """Three consecutive verified errors ending at decision point i (kind: 'primary' or 'incl_hint_first')."""
    F = np.zeros(table.valid.shape, bool)
    for r, q in enumerate(table.seqs):
        for j, i in enumerate(DEC):
            if not table.valid[r, j]:
                continue
            idx = [i - 3, i - 2, i - 1]  # 0-based opportunity indices
            if kind == "primary":
                F[r, j] = all(q.correct[t] == 0 and q.first_action[t] == 0 for t in idx)
            else:
                F[r, j] = all(q.correct[t] == 0 for t in idx)
    return F


def evaluate_fire(table, err, fire, w_I, w_D):
    """Metrics for a boolean fire matrix; at most one intervention per sequence (the first firing)."""
    fire = fire & table.valid & err
    any_fire = fire.any(1)
    i_first = fire.argmax(1) + 3
    opp = (table.valid & err)
    has_opp = opp.any(1)
    i_err = opp.argmax(1) + 3
    y1 = table.y == 1
    n_students = len(set(table.sid.tolist()))
    unnecessary = any_fire & ~y1
    y1_opp = y1 & has_opp
    fired1 = y1_opp & any_fire
    missed = y1_opp & ~any_fire
    delay = np.where(fired1, i_first - i_err, T_DETERMINE - i_err)
    D = float(delay[y1_opp].sum())
    U = int(unnecessary.sum())
    n_int = int(any_fire.sum())
    out = {
        "n_sequences": int(len(table.y)), "n_students": n_students,
        "n_y1": int(y1.sum()), "n_y1_with_help_opportunity": int(y1_opp.sum()),
        "n_y1_no_help_opportunity": int((y1 & ~has_opp).sum()),
        "interventions": n_int, "interventions_per_student": n_int / n_students,
        "unnecessary": U, "unnecessary_per_student": U / n_students,
        "share_unnecessary": (U / n_int) if n_int else None,
        "missed_help": int(missed.sum()),
        "missed_help_share": float(missed.sum() / y1_opp.sum()) if y1_opp.sum() else None,
        "mean_lead_time": float((T_DETERMINE - i_first[fired1]).mean()) if fired1.sum() else None,
        "mean_delay_incl_censored": float(delay[y1_opp].mean()) if y1_opp.sum() else None,
        "mean_delay_excl_censored": float(delay[fired1].mean()) if fired1.sum() else None,
        "total_delay_D": D,
    }
    if w_I is not None:
        out["cost_J"] = w_I * U + w_D * D
        out["cost_J_per_student"] = out["cost_J"] / n_students
    return out


def sweep_tau(table, err, S):
    """Metrics (without cost) for every tau in the grid, using calibrated scores S."""
    res = {}
    for tau in TAU_GRID:
        fire = np.nan_to_num(S, nan=-1.0) >= tau
        res[float(tau)] = evaluate_fire(table, err, fire, None, None)
    return res


def choose_tau(sweep, w_I, w_D):
    best = None
    for tau, m in sweep.items():
        J = w_I * m["unnecessary"] + w_D * m["total_delay_D"]
        if best is None or J < best[1] - 1e-12 or (abs(J - best[1]) <= 1e-12 and tau > best[0]):
            best = (tau, J)
    return best


def with_cost(m, w_I, w_D):
    m = dict(m)
    m["cost_J"] = w_I * m["unnecessary"] + w_D * m["total_delay_D"]
    m["cost_J_per_student"] = m["cost_J"] / m["n_students"]
    return m


def random_matched(table, err, target, w_I, w_D, draws, seed):
    """Random policy with the same number of intervened sequences as `target` (bisection on a score threshold)."""
    rng = np.random.RandomState(seed)
    base = table.valid & err
    rows = []
    for _ in range(draws):
        R = rng.rand(*table.valid.shape)
        lo, hi, best = 0.0, 1.0, (None, 10**9)
        for _ in range(40):
            t = (lo + hi) / 2
            cnt = int((base & (R >= t)).any(1).sum())
            if abs(cnt - target) < best[1]:
                best = (t, abs(cnt - target))
            if cnt > target: lo = t
            elif cnt < target: hi = t
            else: break
        rows.append(evaluate_fire(table, err, base & (R >= best[0]), w_I, w_D))
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and v is not None]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] is not None], float)
        out[k] = {"mean": float(vals.mean()), "sd": float(vals.std())} if len(vals) else None
    out["draws"] = draws
    return out


def platt_fit(y, s):
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    z = np.log(s / (1 - s)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def platt_apply(S, a, b):
    s = np.clip(S, EPS, 1 - EPS)
    z = a + b * np.log(s / (1 - s))
    return 1.0 / (1.0 + np.exp(-z))
