"""Stage 5 shared machinery: per-sequence cost vectors, student aggregation, paired bootstrap, sweeps, freezing.

Cost J = w_I * unnecessary + w_D * delay is additive over sequences, so student-level bootstrap is a
multinomial re-weighting of per-student totals (SPEC A2.5).
"""
import numpy as np

from . import baselines as B
from . import policy as P
from .constants import T_DETERMINE
from .riskhead import RiskHead, features

COLS = ["fired", "unnec", "y1opp", "missed", "delay", "fired1", "lead", "delay_fired1"]
RISK_METHODS = ("bkt", "pfa", "risk_head")
N_GRID = (1, 2, 3, 4, 5, 6)
BOOT_DRAWS = 1000
BOOT_SEED = 20261018
TAU_SEED = 20261019
RANDOM_SEED = 20261017
RANDOM_DRAWS = 200


# ----------------------------------------------------------------------------- models and scores
def head_rows(seqs):
    from .labels import label
    X, y = [], []
    for q in seqs:
        yy = label(q.n, q.m, "A")
        if yy is None:
            continue
        for i in range(3, 10):
            if q.n >= i and q.m > i:
                X.append(features(q.correct, i)); y.append(yy)
    return np.array(X), np.array(y)


def fit_models(train, pfa_C, bkt_min_seq, head_C):
    pfa = B.PFA(pfa_C).fit(train)
    bkt = B.BKT(bkt_min_seq).fit(train)
    X, y = head_rows(train)
    head = RiskHead(head_C).fit(X, y)
    return {"pfa": pfa, "bkt": bkt, "risk_head": head}


def raw_scores(full, models):
    return {m: P.score_matrix(full, lambda q, i, mm=models[m]: mm.score(q, i)) for m in RISK_METHODS}


# ----------------------------------------------------------------------------- vectors and aggregation
def seq_vec(table, err, fire):
    """(N, 8) per-sequence contributions for a boolean fire matrix; first firing only (SPEC A1.2)."""
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
    delay = np.where(y1opp, np.where(fired, i_first - i_err, T_DETERMINE - i_err), 0)
    lead = np.where(fired1, T_DETERMINE - i_first, 0)
    unnec = fired & ~y1
    return np.column_stack([fired, unnec, y1opp, missed, delay, fired1, lead, np.where(fired1, delay, 0)]).astype(float)


class Agg:
    def __init__(self, sid):
        self.uniq, self.codes = np.unique(sid, return_inverse=True)
        self.S = len(self.uniq)

    def by_student(self, vec):
        A = np.zeros((self.S, vec.shape[1]))
        np.add.at(A, self.codes, vec)
        return A

    def draws(self, B_, seed):
        rng = np.random.RandomState(seed)
        return rng.multinomial(self.S, np.ones(self.S) / self.S, size=B_).astype(float)  # (B, S)


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def metric_arrays(tot, S, wI=None, wD=None):
    """tot: (..., 8) totals. Returns dict of arrays, NaN where a denominator is zero."""
    fired, unnec, y1opp, missed, delay, fired1, lead, dfired1 = [tot[..., k] for k in range(8)]
    out = {
        "interventions_per_student": fired / S, "unnecessary_per_student": unnec / S,
        "share_unnecessary": _div(unnec, fired), "missed_help_share": _div(missed, y1opp),
        "mean_delay_incl_censored": _div(delay, y1opp), "mean_delay_excl_censored": _div(dfired1, fired1),
        "mean_lead_time": _div(lead, fired1),
    }
    if wI is not None:
        out["cost_J_per_student"] = (wI * unnec + wD * delay) / S
    return out


def _f(x):
    x = float(x)
    return None if np.isnan(x) else x


def point_metrics(vec, S, wI=None, wD=None):
    tot = vec.sum(0)
    out = {k: _f(v) for k, v in metric_arrays(tot, S, wI, wD).items()}
    out.update({"interventions": int(tot[0]), "unnecessary": int(tot[1]), "n_y1_with_help_opportunity": int(tot[2]),
                "missed_help": int(tot[3]), "total_delay_D": float(tot[4]), "n_students": int(S)})
    return out


def ci(x, lo=2.5, hi=97.5):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi)), int(len(x))]


# ----------------------------------------------------------------------------- random baseline
def random_matched_vec(table, err, target, draws, seed):
    """Expected per-sequence vector of a random policy matched to `target` intervened sequences."""
    rng = np.random.RandomState(seed)
    base = table.valid & err
    acc = np.zeros((len(table.y), len(COLS)))
    per_draw = []
    for _ in range(draws):
        R = rng.rand(*table.valid.shape)
        lo, hi, best = 0.0, 1.0, (0.5, 10 ** 9)
        for _ in range(40):
            t = (lo + hi) / 2
            cnt = int((base & (R >= t)).any(1).sum())
            if abs(cnt - target) < best[1]:
                best = (t, abs(cnt - target))
            if cnt > target: lo = t
            elif cnt < target: hi = t
            else: break
        v = seq_vec(table, err, base & (R >= best[0]))
        acc += v
        per_draw.append(v.sum(0))
    return acc / draws, np.array(per_draw)


# ----------------------------------------------------------------------------- selection (validation only)
def sweep_totals(table, err, S, agg=None):
    """Totals (T, 8) over the tau grid for one calibrated score matrix."""
    Sx = np.nan_to_num(S, nan=-1.0)
    return np.stack([seq_vec(table, err, Sx >= tau).sum(0) for tau in P.TAU_GRID])


def pick(J, grid):
    """argmin of J with ties going to the larger grid value."""
    J = np.asarray(J, float)
    best = np.nanmin(J)
    idx = np.where(np.abs(J - best) <= 1e-9)[0]
    return float(np.asarray(grid)[idx[-1]])


def choose_values(table, kind, S_cal):
    err = table.err[kind]
    taus = {}
    for m in RISK_METHODS:
        tot = sweep_totals(table, err, S_cal[m])
        taus[m] = {}
        for prof, (wI, wD) in P.PROFILES.items():
            taus[m][prof] = pick(wI * tot[:, 1] + wD * tot[:, 4], P.TAU_GRID)
    ntot = np.stack([seq_vec(table, err, P.run_fire(table, kind, n)).sum(0) for n in N_GRID])
    ns = {prof: int(pick(wI * ntot[:, 1] + wD * ntot[:, 4], N_GRID)) for prof, (wI, wD) in P.PROFILES.items()}
    return {"taus": taus, "n": ns}


def tau_stability(table, kind, S_cal_m, frozen_tau, B_=BOOT_DRAWS, seed=TAU_SEED):
    """Student-bootstrap distribution of the validation-chosen tau (SPEC A2.6)."""
    err = table.err[kind]
    agg = Agg(table.sid)
    Sx = np.nan_to_num(S_cal_m, nan=-1.0)
    A = np.stack([agg.by_student(seq_vec(table, err, Sx >= tau)) for tau in P.TAU_GRID])  # (T, S, 8)
    counts = agg.draws(B_, seed)
    tot = np.einsum("bs,tsf->btf", counts, A)  # (B, T, 8)
    tot_full = A.sum(1)                          # (T, 8)
    out = {}
    for prof, (wI, wD) in P.PROFILES.items():
        J = wI * tot[:, :, 1] + wD * tot[:, :, 4]
        chosen = np.array([pick(j, P.TAU_GRID) for j in J])
        Jfull = wI * tot_full[:, 1] + wD * tot_full[:, 4]
        near = np.where(Jfull <= Jfull.min() * 1.02)[0]
        out[prof] = {
            "frozen_tau": frozen_tau[prof],
            "boot_tau_median": float(np.median(chosen)),
            "boot_tau_p10_p90": [float(np.percentile(chosen, 10)), float(np.percentile(chosen, 90))],
            "frac_within_0.10_of_frozen": float(np.mean(np.abs(chosen - frozen_tau[prof]) <= 0.10 + 1e-9)),
            "tau_range_within_2pct_of_min_J": [float(P.TAU_GRID[near].min()), float(P.TAU_GRID[near].max())],
            "J_per_student_at_frozen": float(Jfull[int(np.argmin(np.abs(P.TAU_GRID - frozen_tau[prof])))] / agg.S),
            "J_per_student_min": float(Jfull.min() / agg.S),
        }
    return out


# ----------------------------------------------------------------------------- evaluation of frozen values
def sweep_summary(table, kind, S_cal, with_random=True):
    """Descriptive sweeps (per-student metrics) for plots: risk-method taus, n for the n-error rule, random frontier."""
    err = table.err[kind]
    agg = Agg(table.sid)
    out = {"tau_grid": [float(t) for t in P.TAU_GRID], "methods": {}, "n_rule": {}, "n_students": agg.S}
    for m in RISK_METHODS:
        tot = sweep_totals(table, err, S_cal[m])
        rows = []
        for t_idx in range(len(P.TAU_GRID)):
            met = {k: _f(v) for k, v in metric_arrays(tot[t_idx], agg.S).items()}
            met["J"] = {prof: float((wI * tot[t_idx, 1] + wD * tot[t_idx, 4]) / agg.S) for prof, (wI, wD) in P.PROFILES.items()}
            rows.append(met)
        out["methods"][m] = rows
    for n in N_GRID:
        tot = seq_vec(table, err, P.run_fire(table, kind, n)).sum(0)
        met = {k: _f(v) for k, v in metric_arrays(tot, agg.S).items()}
        met["J"] = {prof: float((wI * tot[1] + wD * tot[4]) / agg.S) for prof, (wI, wD) in P.PROFILES.items()}
        out["n_rule"][str(n)] = met
    if with_random:
        fr = []
        bkt_tot = sweep_totals(table, err, S_cal["bkt"])
        for t_idx in range(0, len(P.TAU_GRID) - 1, 5):
            target = int(round(bkt_tot[t_idx, 0]))
            if target == 0:
                continue
            vec, _ = random_matched_vec(table, err, target, 40, RANDOM_SEED)
            met = {k: _f(v) for k, v in metric_arrays(vec.sum(0), agg.S).items()}
            fr.append(met)
        out["random_frontier"] = fr
    return out


def evaluate_variant(table, kind, S_cal, taus, ns):
    """Evaluate frozen taus/ns. Nothing is chosen here. Paired student bootstrap for J differences."""
    err = table.err[kind]
    agg = Agg(table.sid)
    counts = agg.draws(BOOT_DRAWS, BOOT_SEED)
    vecs, A = {}, {}

    def add(key, vec):
        vecs[key] = vec
        A[key] = agg.by_student(vec)

    for m in RISK_METHODS:
        Sx = np.nan_to_num(S_cal[m], nan=-1.0)
        for prof in P.PROFILES:
            add((m, prof), seq_vec(table, err, Sx >= taus[m][prof]))
    for prof in P.PROFILES:
        add(("always_help", prof), seq_vec(table, err, table.valid & err))
        add(("n_rule_tuned", prof), seq_vec(table, err, P.run_fire(table, kind, ns[prof])))
        add(("three_wrong_untuned", prof), seq_vec(table, err, P.run_fire(table, kind, 3)))
    rand_sd = {}
    for m in RISK_METHODS:
        for prof in P.PROFILES:
            target = int(vecs[(m, prof)][:, 0].sum())
            vec, per_draw = random_matched_vec(table, err, target, RANDOM_DRAWS, RANDOM_SEED)
            add((f"random_matched_to_{m}", prof), vec)
            wI, wD = P.PROFILES[prof]
            rand_sd[(m, prof)] = float(((wI * per_draw[:, 1] + wD * per_draw[:, 4]) / agg.S).std())

    res = {"n_students": agg.S, "n_sequences": int(len(table.y)), "n_y1": int(table.y.sum()),
           "taus": taus, "n_rule_n": ns, "methods": {}, "j_diff_vs_bkt": {}}
    for (key, prof), vec in vecs.items():
        wI, wD = P.PROFILES[prof]
        pm = point_metrics(vec, agg.S, wI, wD)
        bs = metric_arrays(counts @ A[(key, prof)], agg.S, wI, wD)   # (B,) per metric
        pm["ci95"] = {k: ci(v) for k, v in bs.items()}
        if key.startswith("random_matched_to_"):
            pm["J_sd_over_random_draws"] = rand_sd[(key.replace("random_matched_to_", ""), prof)]
        res["methods"].setdefault(key, {})[prof] = pm
    for prof, (wI, wD) in P.PROFILES.items():
        Jb = (wI * (counts @ A[("bkt", prof)][:, 1]) + wD * (counts @ A[("bkt", prof)][:, 4])) / agg.S
        res["j_diff_vs_bkt"][prof] = {}
        for key in ("always_help", "n_rule_tuned", "three_wrong_untuned", "pfa", "risk_head", "random_matched_to_bkt"):
            Jo = (wI * (counts @ A[(key, prof)][:, 1]) + wD * (counts @ A[(key, prof)][:, 4])) / agg.S
            d = Jb - Jo
            point = float((wI * (vecs[("bkt", prof)][:, 1].sum() - vecs[(key, prof)][:, 1].sum())
                           + wD * (vecs[("bkt", prof)][:, 4].sum() - vecs[(key, prof)][:, 4].sum())) / agg.S)
            lo, hi, _ = ci(d)
            res["j_diff_vs_bkt"][prof][key] = {"diff_bkt_minus_other": point, "ci95": [lo, hi],
                                               "bkt_cheaper_significant": bool(hi < 0), "bkt_costlier_significant": bool(lo > 0)}
    return res


# ----------------------------------------------------------------------------- split-level driver (used for val and test alike)
def split_configs(full, cal):
    prim = full.subset(~full.leaver)
    calp = {m: S[~full.leaver] for m, S in cal.items()}
    return {"main": (prim, "primary", calp), "leavers": (full, "primary", cal), "hintfirst": (prim, "incl_hint_first", calp)}


def evaluate_split(seqs, frozen, models, extra_val_stability=False):
    """Apply frozen calibrators/taus/ns to `seqs`. Nothing is chosen on `seqs`."""
    full = P.Table(seqs, include_leavers=True)
    raw = raw_scores(full, models)
    cal = {m: P.platt_apply(raw[m], frozen["platt"][m]["intercept"], frozen["platt"][m]["slope"]) for m in RISK_METHODS}
    out = {}
    for name, (tab, kind, S) in split_configs(full, cal).items():
        out[name] = {
            "scope": {"sequences": int(len(tab.y)), "students": int(len(set(tab.sid.tolist()))), "y1": int(tab.y.sum()),
                      "leavers_included": int(tab.leaver.sum()),
                      "verified_error_cells": int((tab.valid & tab.err[kind]).sum()), "decision_cells": int(tab.valid.sum())},
            "variants": {v: evaluate_variant(tab, kind, S, vals["taus"], vals["n"])
                         for v, vals in frozen["configs"][name].items()},
            "sweep": sweep_summary(tab, kind, S),
        }
        if extra_val_stability:
            first = next(iter(frozen["configs"][name].values()))
            out[name]["tau_stability_bkt"] = tau_stability(tab, kind, S["bkt"], first["taus"]["bkt"])
    return out
