"""Amendment A8: how much do BKT parameters drift across ASSISTments skills, and how much would the speak / flag decisions change if a skill's own
parameters replaced the pooled ones? Exploratory, validation only (SPEC.md A8). This is transfer risk across skills, NOT a calculus calibration."""
from __future__ import annotations

import numpy as np

from . import baselines as B
from . import mastery_sens as MS
from . import policy as P
from . import stage5 as S5

STATUS = "exploratory, validation only (A8); transfer risk across ASSISTments skills, not a calculus calibration"
BOOT_DRAWS = 1000
BOOT_SEED = 20261024
PARAMS = ("prior", "learn", "guess", "slip")
GROUPS = (   # A8.3: first match, in this order; fixed before any result
    ("algebra-type", ("Equation", "Solving for a variable", "Linear", "Slope", "Inequalit", "Exponent", "Order of Operations")),
    ("geometry/measurement", ("Area", "Volume", "Circumference", "Surface", "Angle", "Pythagorean", "Similar", "Perimeter")),
    ("data/probability", ("Probability", "Venn", "Box and Whisker", "Scatter", "Histogram", "Median", "Mean", "Stem", "Table", "Circle Graph",
                          "Counting", "Pattern")),
    ("number/arithmetic", ("Addition", "Subtraction", "Multiplication", "Division", "Fraction", "Decimal", "Percent", "Integer", "Ordering",
                          "Estimation", "Proportion", "Unit Rate", "Multiple", "Divisib", "Square Root", "Scientific", "Absolute", "Number Line",
                          "Rounding")),
)


def group_of(skill_name: str) -> str:
    for g, keys in GROUPS:
        if any(k in skill_name for k in keys):
            return g
    return "other"


def _q(x):
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "min": float(np.min(x)), "max": float(np.max(x))}


def spread(kc_fits, min_seq, skill_names, n_total_train):
    """A8.2: per parameter, median / IQR / min / max over skills with at least min_seq train sequences."""
    pooled = kc_fits[None][1]
    own = {kc: (n, par) for kc, (n, par) in kc_fits.items() if kc is not None and par is not None and n >= min_seq}
    out = {"n_skills_fitted": len(own), "n_skills_total": len([k for k in kc_fits if k is not None]), "pooled": pooled,
           "share_of_train_sequences_covered": float(sum(n for n, _ in own.values()) / n_total_train), "parameters": {}}
    for p in PARAMS:
        vals = np.array([par[p] for _, par in own.values()])
        lo, hi = B.BKT_BOUNDS[p]
        out["parameters"][p] = {**_q(vals), "pooled": pooled[p],
                                "abs_distance_from_pooled": _q(np.abs(vals - pooled[p])),
                                "n_at_lower_bound": int((vals <= lo + 1e-9).sum()), "n_at_upper_bound": int((vals >= hi - 1e-9).sum())}
    groups = {}
    for kc, (n, par) in own.items():
        groups.setdefault(group_of(skill_names.get(kc, "")), []).append((kc, n, par))
    out["by_group"] = {}
    for g, items in sorted(groups.items()):
        row = {"n_skills": len(items), "skills": sorted(skill_names.get(kc, kc) for kc, _, _ in items)[:40]}
        if len(items) >= 5:
            row["parameters"] = {p: _q(np.array([par[p] for _, _, par in items])) for p in PARAMS}
        else:
            row["parameters"] = "fewer than 5 skills: not summarised (A8.3)"
        out["by_group"][g] = row
    return out


class _Pooled(B.BKT):
    """The demo's situation: the pooled parameters for every skill."""

    def __init__(self, bkt):
        super().__init__(bkt.min_seq)
        self.kc_fits = {None: bkt.kc_fits[None]}

    def params(self, kc):
        return self.kc_fits[None][1]


def decisions(val_seqs, bkt, skill_names, frozen_platt, frozen_tau):
    """A8.4. D3, T = 10, decision points 3-9, frozen calibrator and thresholds; own-skill versus pooled parameters."""
    d3 = MS.DEFS[0]
    full = MS.SensTable(val_seqs, include_leavers=True, T=d3.T)
    prim = full.subset(~full.leaver)
    a, b = frozen_platt["intercept"], frozen_platt["slope"]
    raw_own = MS.raw_matrix(prim, bkt, d3)
    raw_pool = MS.raw_matrix(prim, _Pooled(bkt), d3)
    cal_own, cal_pool = P.platt_apply(raw_own, a, b), P.platt_apply(raw_pool, a, b)
    err = prim.err["primary"]
    cell = prim.valid & err                                                   # decision cells with a verified wrong step
    own_param_skill = np.array([bkt.kc_fits.get(q.kc, (0, None))[1] is not None and bkt.kc_fits[q.kc][0] >= bkt.min_seq for q in prim.seqs])
    group = np.array([group_of(skill_names.get(q.kc, "")) for q in prim.seqs])
    agg = S5.Agg(prim.sid)
    counts = agg.draws(BOOT_DRAWS, BOOT_SEED)
    always = agg.by_student(MS.seq_vec_T(prim, err, prim.valid & err, d3.T))
    out = {"decision_cells_with_verified_error": int(cell.sum()), "sequences": int(len(prim.y)), "students": int(agg.S),
           "cells_of_skills_with_own_params": int(cell[own_param_skill].sum()), "cells_of_skills_without_own_params_identical": int(cell[~own_param_skill].sum()),
           "profiles": {}}
    for prof in P.PROFILES:
        tau = frozen_tau[prof]
        f_own, f_pool = np.nan_to_num(cal_own, nan=-1.0) >= tau, np.nan_to_num(cal_pool, nan=-1.0) >= tau
        flip = (f_own != f_pool) & cell
        row = {"tau": tau, "flips": int(flip.sum()), "flip_share_of_cells": float(flip.sum() / cell.sum()),
               "flip_share_among_cells_of_skills_with_own_params": float(flip[own_param_skill].sum() / max(cell[own_param_skill].sum(), 1)),
               "speak_to_flag(own speaks, pooled flags)": int(((f_own & ~f_pool) & cell).sum()),
               "flag_to_speak(own flags, pooled speaks)": int(((~f_own & f_pool) & cell).sum()),
               "flip_share_by_group": {}}
        for g in sorted(set(group.tolist())):
            m = np.repeat((group == g)[:, None], P.NJ, axis=1) & cell
            row["flip_share_by_group"][g] = {"cells": int(m.sum()), "flips": int((flip & m).sum()),
                                             "share": float((flip & m).sum() / m.sum()) if m.sum() else None}
        for name, f in (("own_skill_params", f_own), ("pooled_params", f_pool)):
            vec = MS.seq_vec_T(prim, err, f & prim.valid, d3.T)
            A = agg.by_student(vec)
            tb, ta = counts @ A, counts @ always
            with np.errstate(divide="ignore", invalid="ignore"):
                red_b = 1.0 - tb[:, 0] / ta[:, 0]
            miss_b = S5.metric_arrays(tb, agg.S)["missed_help_share"]
            tp, ap = vec.sum(0), always.sum(0)
            row[name] = {"reduction_vs_always_help": float(1 - tp[0] / ap[0]), "reduction_ci95": S5.ci(red_b),
                         "missed_help_share": float(tp[3] / tp[2]) if tp[2] else None, "missed_help_ci95": S5.ci(miss_b),
                         "interventions_per_student": float(tp[0] / agg.S)}
        out["profiles"][prof] = row
    return out


def per_skill_distance(kc_fits, min_seq, skill_names, top=10):
    """Largest absolute parameter distances from the pooled fit (descriptive)."""
    pooled = kc_fits[None][1]
    rows = []
    for kc, (n, par) in kc_fits.items():
        if kc is None or par is None or n < min_seq:
            continue
        rows.append({"skill_id": kc, "skill_name": skill_names.get(kc, ""), "train_sequences": int(n), **{p: par[p] for p in PARAMS},
                     "max_abs_dist": float(max(abs(par[p] - pooled[p]) for p in PARAMS))})
    rows.sort(key=lambda r: -r["max_abs_dist"])
    return {"largest": rows[:top], "smallest": rows[-top:]}
