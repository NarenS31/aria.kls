"""Amendment A6: parametric closed-loop sensitivity analysis. ASSUMPTION-DEPENDENT, NOT EVIDENCE OF EFFECT. No LLM, no real data, no locked split.

Student = BKT-style latent known/unknown state with the frozen pooled parameters; policies = always-help and the three frozen profiles (frozen Platt
calibrator and frozen tau, never re-tuned); three assumed parameters (p_help_effect, p_interrupt_cost, p_stuck_cost). Common random numbers are shared by
every policy and grid point. See SPEC.md A6 for every definition.
"""
from __future__ import annotations

import json

import numpy as np

from . import mastery_sens as MS
from . import policy as P
from .constants import RESULTS

T = 10
RUN = 3
SEED = 20261101
N_STUDENTS = 100_000
GRID = {"p_help_effect": (0.0, 0.1, 0.2, 0.3, 0.5), "p_interrupt_cost": (0.0, 0.1, 0.2, 0.3), "p_stuck_cost": (0.0, 0.05, 0.1, 0.2, 0.3)}
STATUS = "assumption-dependent sensitivity analysis, not evidence of effect"
POLICIES = ("always_help", "quiet", "balanced", "supportive")


def load_frozen():
    val = json.load(open(RESULTS / "frozen_val.json"))
    par = json.load(open(RESULTS / "baselines_val.json"))["selection"]["bkt_pooled_params"]
    pl = val["platt"]["bkt"]
    return {"par": {k: float(par[k]) for k in ("prior", "learn", "guess", "slip")}, "platt": (float(pl["intercept"]), float(pl["slope"])),
            "taus": {k: float(v) for k, v in val["configs"]["main"]["frozen"]["taus"]["bkt"].items()}}


def risk_tables(par, platt):
    """tab[t][code] = Platt-calibrated BKT risk of the observed prefix (bits c_1..c_t packed MSB-first), NaN for prefixes that already contain a 3-run."""
    d3 = MS.DEFS[0]
    tabs = {}
    for t in range(3, 10):
        tab = np.full(2 ** t, np.nan)
        for code in range(2 ** t):
            bits = np.array([(code >> (t - 1 - j)) & 1 for j in range(t)], dtype=np.int8)
            if any(bits[j:j + RUN].all() for j in range(t - RUN + 1)):
                continue
            tab[code] = float(P.platt_apply(np.array([MS.bkt_risk(bits, par, d3)]), *platt)[0])
        tabs[t] = tab
    return tabs


def draw(n=N_STUDENTS, seed=SEED):
    rng = np.random.RandomState(seed)
    return {"k0": rng.rand(n), "resp": rng.rand(n, T), "learn": rng.rand(n, T), "ab": rng.rand(n, T)}


def simulate(fz, tabs, U, tau, p_help, p_int, p_stuck):
    """One policy on the shared random numbers. tau: None = never intervene (no-help world), -inf = always-help, else risk threshold."""
    par = fz["par"]
    N = len(U["k0"])
    K = U["k0"] < par["prior"]
    run = np.zeros(N, int)
    mastered = np.zeros(N, int)
    abandoned = np.zeros(N, bool)
    in_play = np.ones(N, bool)
    code = np.zeros(N, np.int64)
    help_on = np.zeros(N, bool)
    interrupt = np.zeros(N, bool)
    i_first = np.zeros(N, int)
    i_err = np.zeros(N, int)
    resp = np.zeros((N, T), np.int8)
    learn_h = 1 - (1 - par["learn"]) * (1 - p_help)
    for t in range(1, T + 1):
        cont = ~abandoned
        slip_t = np.where(interrupt, par["slip"] + p_int * (1 - par["slip"]), par["slip"])
        c = (U["resp"][:, t - 1] < np.where(K, 1 - slip_t, par["guess"])).astype(np.int8)
        interrupt = np.zeros(N, bool)
        resp[:, t - 1] = np.where(cont, c, -1)
        run = np.where(c == 1, run + 1, 0)
        code = code * 2 + c
        new_m = in_play & cont & (run >= RUN)
        mastered[new_m & (mastered == 0)] = t
        in_play = in_play & ~new_m
        if 3 <= t <= 9:
            wrong = in_play & cont & (c == 0)
            i_err[wrong & (i_err == 0)] = t
            cand = wrong & (i_first == 0)
            if tau is None:
                fire = np.zeros(N, bool)
            elif tau == -np.inf:
                fire = cand
            else:
                risk = np.full(N, -1.0)
                risk[cand] = tabs[t][code[cand]]
                fire = cand & (risk >= tau)
            i_first[fire] = t
            help_on |= fire
            interrupt |= fire
            unaided = wrong & ~fire & (i_first == 0)
            ab = unaided & (U["ab"][:, t - 1] < p_stuck)
            abandoned |= ab
            in_play = in_play & ~ab
        p_learn = np.where(help_on, learn_h, par["learn"])
        K = K | (~K & (U["learn"][:, t - 1] < p_learn))
    return {"mastered": mastered, "abandoned": abandoned, "i_first": i_first, "i_err": i_err, "resp": resp}


def outcomes(run, nohelp, w_I, w_D):
    """Per-sequence outcome vectors over the in-scope set (not mastered by opportunity 3; identical for every policy)."""
    scope = ~((nohelp["mastered"] >= 1) & (nohelp["mastered"] <= 3))
    y1 = run["mastered"] == 0
    has_opp = run["i_err"] > 0
    fired = run["i_first"] > 0
    unnec = fired & (nohelp["mastered"] > 0)
    y1opp = y1 & has_opp
    delay = np.where(y1opp, np.where(fired, run["i_first"] - run["i_err"], T - run["i_err"]), 0)
    missed = y1opp & ~fired
    J = w_I * unnec + w_D * delay
    return {"scope": scope, "J": J, "unnec": unnec, "delay": delay, "fired": fired, "y1": y1, "y1opp": y1opp, "missed": missed}


def summarise_point(fz, tabs, U, p_help, p_int, p_stuck):
    nohelp = simulate(fz, tabs, U, None, p_help, p_int, p_stuck)
    runs = {"always_help": simulate(fz, tabs, U, -np.inf, p_help, p_int, p_stuck)}
    for prof in ("quiet", "balanced", "supportive"):
        runs[prof] = simulate(fz, tabs, U, fz["taus"][prof], p_help, p_int, p_stuck)
    out = {"p_help_effect": p_help, "p_interrupt_cost": p_int, "p_stuck_cost": p_stuck, "policies": {}, "delta_J_policy_minus_always_help": {}}
    scope = None
    for name, run in runs.items():
        o = outcomes(run, nohelp, 1.0, 1.0)
        scope = o["scope"]
        n = int(scope.sum())
        out["policies"][name] = {"n_in_scope": n, "interventions_per_scope_sequence": float(o["fired"][scope].mean()),
                                 "mastery_rate_within_T": float((~o["y1"])[scope].mean()), "abandon_rate": float(run["abandoned"][scope].mean()),
                                 "missed_help_share": float(o["missed"][scope].sum() / max(o["y1opp"][scope].sum(), 1)),
                                 "unnecessary_per_sequence": float(o["unnec"][scope].mean()), "delay_per_sequence": float(o["delay"][scope].mean())}
    for prof, (wI, wD) in P.PROFILES.items():
        a = outcomes(runs["always_help"], nohelp, wI, wD)
        b = outcomes(runs[prof], nohelp, wI, wD)
        d = (b["J"] - a["J"])[scope].astype(float)
        m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        out["delta_J_policy_minus_always_help"][prof] = {
            "J_policy": float(b["J"][scope].mean()), "J_always_help": float(a["J"][scope].mean()), "delta": m, "ci95_simulation_noise_only": [lo, hi],
            "region": "beats" if hi < 0 else ("flips" if lo > 0 else "indistinguishable")}
    return out


def run_grid(n=N_STUDENTS, log=None):
    fz = load_frozen()
    tabs = risk_tables(fz["par"], fz["platt"])
    U = draw(n)
    pts = []
    for ph in GRID["p_help_effect"]:
        for pi in GRID["p_interrupt_cost"]:
            for ps in GRID["p_stuck_cost"]:
                pts.append(summarise_point(fz, tabs, U, ph, pi, ps))
        if log:
            log(f"p_help_effect={ph} done")
    return {"status": STATUS, "spec": "aria_pace/SPEC.md Amendment 6", "frozen_inputs": {"pooled_bkt": fz["par"], "platt_bkt": list(fz["platt"]), "taus_bkt": fz["taus"]},
            "grid": {k: list(v) for k, v in GRID.items()}, "n_students_per_point": n, "seed": SEED, "points": pts}
