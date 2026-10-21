"""Amendment A4: mastery-definition sensitivity (validation only, BKT only). Exploratory; see SPEC.md A4.

The frozen pipeline binds MASTERY_RUN = 3 and T_DETERMINE = 10 at import (labels.py:4, policy.py:5, stage5.py:10), so this module re-implements the
same steps with the mastery rule and T as parameters and leaves every existing module untouched. Under D3 (three in a row, T = 10) it must reproduce
frozen_val.json exactly (A4.3); `python -m aria_pace.run_mastery_sens` checks that before it looks at any other definition.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import baselines as B
from . import policy as P
from . import stage5 as S5


@dataclass(frozen=True)
class Def:
    name: str
    kind: str            # "run" | "ema"
    T: int
    run: int = 3         # run rules: consecutive correct needed
    alpha: float = 0.3   # ema rule
    thr: float = 0.8
    consec: int = 3
    text: str = ""


DEFS = (
    Def("D3", "run", 10, run=3, text="3 correct in a row, T=10 (control = frozen definition)"),
    Def("D4", "run", 10, run=4, text="4 correct in a row, T=10"),
    Def("D5T15", "run", 15, run=5, text="5 correct in a row, T=15"),
    Def("D5T20", "run", 20, run=5, text="5 correct in a row, T=20"),
    Def("DEMA", "ema", 10, text="EMA(alpha=0.3, ema_1=x_1) >= 0.8 for 3 consecutive opportunities, T=10"),
)


# ----------------------------------------------------------------------------- mastery and labels
def first_mastery_def(correct, d: Def) -> float:
    """First 1-based opportunity at which the definition's mastery condition is met, else inf."""
    if d.kind == "run":
        run = 0
        for i, x in enumerate(correct, start=1):
            run = run + 1 if x == 1 else 0
            if run >= d.run:
                return float(i)
        return float("inf")
    ema, cnt = None, 0
    for i, x in enumerate(correct, start=1):
        ema = float(x) if ema is None else d.alpha * float(x) + (1 - d.alpha) * ema
        cnt = cnt + 1 if ema >= d.thr - 1e-12 else 0
        if cnt >= d.consec:
            return float(i)
    return float("inf")


def label_def(n: int, m: float, T: int):
    """Variant A: 0 if mastered within T, 1 if not mastered and n >= T, else None (unlabelled leaver)."""
    if m <= T:
        return 0
    return 1 if n >= T else None


def relabel(seqs, d: Def):
    """Copies of the sequences with `m` recomputed under the definition (the Sequence dataclass is not modified)."""
    return [dataclasses.replace(q, m=first_mastery_def(q.correct, d)) for q in seqs]


# ----------------------------------------------------------------------------- decision table (copy of policy.Table with parametrised label)
class SensTable(P.Table):
    def __init__(self, seqs, include_leavers, T):
        self.T = T
        items = []
        for q in seqs:
            if q.n >= 3 and q.m > 3:
                y = label_def(q.n, q.m, T)
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
        self.valid = np.zeros((N, P.NJ), bool)
        self.err_primary = np.zeros((N, P.NJ), bool)
        self.err_all = np.zeros((N, P.NJ), bool)
        for r, (q, _, _) in enumerate(items):
            for j, i in enumerate(P.DEC):
                if i <= min(9, q.n) and q.m > i:
                    self.valid[r, j] = True
                    wrong = q.correct[i - 1] == 0
                    self.err_all[r, j] = wrong
                    self.err_primary[r, j] = wrong and q.first_action[i - 1] == 0
        self.err = {"primary": self.err_primary, "incl_hint_first": self.err_all}

    def subset(self, mask):
        t = super().subset(mask)
        t.T = self.T
        return t


def seq_vec_T(table, err, fire, T):
    """S5.seq_vec with the determination point T as a parameter (stage5.py:50-65 binds T_DETERMINE = 10)."""
    fire = fire & table.valid & err
    fired = fire.any(1)
    i_first = fire.argmax(1) + 3
    opp = table.valid & err
    has_opp = opp.any(1)
    i_err = opp.argmax(1) + 3
    y1 = table.y == 1
    y1opp = y1 & has_opp
    fired1 = y1opp & fired
    missed = y1opp & ~fired
    delay = np.where(y1opp, np.where(fired, i_first - i_err, T - i_err), 0)
    lead = np.where(fired1, T - i_first, 0)
    unnec = fired & ~y1
    return np.column_stack([fired, unnec, y1opp, missed, delay, fired1, lead, np.where(fired1, delay, 0)]).astype(float)


# ----------------------------------------------------------------------------- BKT risk under a definition (exact)
def bkt_risk(correct_prefix, par, d: Def) -> float:
    """P(no mastery in opportunities k+1..T | first k responses), k = len(correct_prefix), under definition d.
    Same forward filter and latent-state dynamics as baselines.BKT.score (baselines.py:BKT._filter / score)."""
    post, _ = B.BKT._filter(correct_prefix, par)
    pk = float(post[-1])
    k = len(correct_prefix)
    learn, slip, guess = par["learn"], par["slip"], par["guess"]
    if d.kind == "run":
        R = d.run
        r0 = 0
        for x in correct_prefix[::-1]:
            if x == 1:
                r0 += 1
            else:
                break
        if r0 >= R:
            raise ValueError("sequence already mastered at the decision point")
        dd = [[0.0] * R for _ in range(2)]
        dd[1][r0], dd[0][r0] = pk, 1 - pk
        for _ in range(k + 1, d.T + 1):
            nd = [[0.0] * R for _ in range(2)]
            for ls in (0, 1):
                for r in range(R):
                    m = dd[ls][r]
                    if m == 0.0:
                        continue
                    for ls2, pt in ((1, 1.0 if ls == 1 else learn), (0, 0.0 if ls == 1 else 1 - learn)):
                        if pt == 0.0:
                            continue
                        pc = (1 - slip) if ls2 == 1 else guess
                        if r + 1 < R:
                            nd[ls2][r + 1] += m * pt * pc
                        nd[ls2][0] += m * pt * (1 - pc)
            dd = nd
        return float(sum(map(sum, dd)))
    # EMA rule: state = (latent, ema, consecutive count of ema >= thr)
    ema, cnt = None, 0
    for x in correct_prefix:
        ema = float(x) if ema is None else d.alpha * float(x) + (1 - d.alpha) * ema
        cnt = cnt + 1 if ema >= d.thr - 1e-12 else 0
    if cnt >= d.consec:
        raise ValueError("sequence already mastered at the decision point")
    states = {(1, round(ema, 12), cnt): pk, (0, round(ema, 12), cnt): 1 - pk}
    for _ in range(k + 1, d.T + 1):
        nxt = {}
        for (ls, e, c), m in states.items():
            for ls2, pt in ((1, 1.0 if ls == 1 else learn), (0, 0.0 if ls == 1 else 1 - learn)):
                if pt == 0.0:
                    continue
                for x, px in ((1, (1 - slip) if ls2 == 1 else guess), (0, slip if ls2 == 1 else 1 - guess)):
                    e2 = d.alpha * x + (1 - d.alpha) * e
                    c2 = c + 1 if e2 >= d.thr - 1e-12 else 0
                    if c2 >= d.consec:
                        continue                         # mastered: leaves the "no mastery" mass
                    key = (ls2, round(e2, 12), c2)
                    nxt[key] = nxt.get(key, 0.0) + m * pt * px
        states = nxt
    return float(sum(states.values()))


def raw_matrix(table, bkt, d: Def):
    S = np.full(table.valid.shape, np.nan)
    for r, q in enumerate(table.seqs):
        par = bkt.params(q.kc)
        for j, i in enumerate(P.DEC):
            if table.valid[r, j]:
                S[r, j] = bkt_risk(q.correct[:int(i)], par, d)
    return S


# ----------------------------------------------------------------------------- selection and evaluation (validation only)
def choose_taus(table, err, S_cal, T):
    """tau per profile: argmin J over the grid, ties to the larger tau (stage5.py:153-171), with the definition's T."""
    Sx = np.nan_to_num(S_cal, nan=-1.0)
    tot = np.stack([seq_vec_T(table, err, Sx >= tau, T).sum(0) for tau in P.TAU_GRID])
    return {prof: S5.pick(wI * tot[:, 1] + wD * tot[:, 4], P.TAU_GRID) for prof, (wI, wD) in P.PROFILES.items()}, tot


def tau_bootstrap(table, err, S_cal, T, frozen_tau, B_=S5.BOOT_DRAWS, seed=S5.TAU_SEED):
    agg = S5.Agg(table.sid)
    Sx = np.nan_to_num(S_cal, nan=-1.0)
    A = np.stack([agg.by_student(seq_vec_T(table, err, Sx >= tau, T)) for tau in P.TAU_GRID])
    counts = agg.draws(B_, seed)
    tot = np.einsum("bs,tsf->btf", counts, A)
    out = {}
    for prof, (wI, wD) in P.PROFILES.items():
        J = wI * tot[:, :, 1] + wD * tot[:, :, 4]
        chosen = np.array([S5.pick(j, P.TAU_GRID) for j in J])
        out[prof] = {"point_tau": frozen_tau[prof], "boot_median": float(np.median(chosen)),
                     "boot_p10_p90": [float(np.percentile(chosen, 10)), float(np.percentile(chosen, 90))]}
    return out


def policy_ci(table, err, S_cal, taus, T, B_=S5.BOOT_DRAWS, seed=S5.BOOT_SEED):
    """Reduction vs always-help and missed-help share per profile at the chosen taus, 95% student-bootstrap intervals (one resample shared)."""
    agg = S5.Agg(table.sid)
    counts = agg.draws(B_, seed)
    Sx = np.nan_to_num(S_cal, nan=-1.0)
    always = agg.by_student(seq_vec_T(table, err, table.valid & err, T))
    out = {}
    for prof in P.PROFILES:
        vec = seq_vec_T(table, err, Sx >= taus[prof], T)
        A = agg.by_student(vec)
        tb, ta = counts @ A, counts @ always
        with np.errstate(divide="ignore", invalid="ignore"):
            red_b = 1.0 - tb[:, 0] / ta[:, 0]
        miss_b = S5.metric_arrays(tb, agg.S)["missed_help_share"]
        tp, ap = vec.sum(0), always.sum(0)
        out[prof] = {
            "tau": taus[prof],
            "interventions_per_student": float(tp[0] / agg.S), "always_help_interventions_per_student": float(ap[0] / agg.S),
            "reduction_vs_always_help": float(1 - tp[0] / ap[0]), "reduction_ci95": S5.ci(red_b),
            "missed_help_share": float(tp[3] / tp[2]) if tp[2] else None, "missed_help_ci95": S5.ci(miss_b),
            "n_y1_with_help_opportunity": int(tp[2]), "n_students": int(agg.S),
        }
    return out


def auc_by_k(prim, S_cal_or_raw, ks=(3, 5, 8), n_boot=1000, seed=20261016):
    from . import metrics as M
    out = {}
    for k in ks:
        j = k - 3
        m = prim.valid[:, j]
        y, s, sid = prim.y[m], S_cal_or_raw[m, j], prim.sid[m]
        if len(set(y.tolist())) < 2:
            out[f"k{k}"] = {"n": int(len(y)), "auc": None}
            continue
        ci, used = M.bootstrap_auc_ci(y, s, sid, n_boot, seed)
        out[f"k{k}"] = {"n": int(len(y)), "n_students": int(len(set(sid.tolist()))), "positives": int(y.sum()), "base_rate": float(y.mean()),
                        "auc": M.auc(y, s), "auc_ci95_student_bootstrap": ci, "resamples": used}
    return out


def run_definition(val_seqs, bkt, d: Def, frozen_tau_check=None):
    """Everything A4.4 asks for, one definition. val_seqs: validation sequences (original `m`)."""
    seqs = relabel(val_seqs, d)
    full = SensTable(seqs, include_leavers=True, T=d.T)
    prim_mask = ~full.leaver
    prim = full.subset(prim_mask)
    raw_full = raw_matrix(full, bkt, d)
    raw_prim = raw_full[prim_mask]
    mask = prim.valid
    rows_y = np.repeat(prim.y[:, None], P.NJ, axis=1)[mask]
    a, b = P.platt_fit(rows_y, raw_prim[mask])
    cal_prim = P.platt_apply(raw_prim, a, b)
    err = prim.err["primary"]
    taus, _ = choose_taus(prim, err, cal_prim, d.T)
    labelled = [q for q in seqs if q.n >= 3 and q.m > 3 and label_def(q.n, q.m, d.T) is not None]
    return {
        "definition": d.text, "T": d.T, "platt": {"intercept": a, "slope": b}, "taus": taus,
        "scope": {"in_scope_labelled_sequences": int(len(prim.y)), "students": int(len(set(prim.sid.tolist()))), "y1": int(prim.y.sum()),
                  "label_rate": float(prim.y.mean()), "unlabelled_leavers_dropped": int(full.leaver.sum()),
                  "decision_cells": int(prim.valid.sum()), "verified_error_cells": int((prim.valid & err).sum())},
        "bkt_auc_val": auc_by_k(prim, raw_prim),
        "tau_bootstrap": tau_bootstrap(prim, err, cal_prim, d.T, taus),
        "policy_at_chosen_tau": policy_ci(prim, err, cal_prim, taus, d.T),
        "_n_labelled_in_scope_check": len(labelled),
    }
