"""Stage 4 entry point: python -m aria_pace.run_policy  (from the repo root). VALIDATION ONLY.

Test-split students are removed from memory right after the split is made; nothing here reads their labels.
Writes results/riskhead_val.json and results/policy_val.json.
"""
import json
import time

import numpy as np

from . import baselines as B
from . import policy as P
from . import metrics as M
from .constants import BOOTSTRAP_SEED, KS, RAW_CSV, RESULTS
from .labels import eligibility, label
from .loader import build_sequences, load_raw
from .riskhead import RiskHead, features
from .split import make_split

RANDOM_SEED = 20261017
RANDOM_DRAWS = 200
C_GRID = (0.01, 0.1, 1.0, 10.0)
BEAT_MARGIN = 0.005


def dump(name, obj):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def head_rows(seqs):
    X, y, sid = [], [], []
    for q in seqs:
        yy = label(q.n, q.m, "A")
        if yy is None:
            continue
        for i in range(3, 10):
            if q.n >= i and q.m > i:
                X.append(features(q.correct, i)); y.append(yy); sid.append(q.user_id)
    return np.array(X), np.array(y), np.array(sid)


def scored(seqs, k):
    out = []
    for q in seqs:
        if eligibility(q.n, q.m, k) == "eligible":
            yy = label(q.n, q.m, "A")
            if yy is not None:
                out.append((q, yy))
    return out


def main():
    t0 = time.time()
    df, _ = load_raw(RAW_CSV)
    parts = make_split(df.user_id.unique())
    seqs = build_sequences(df)
    test_ids = set(parts["test"])
    seqs = [q for q in seqs if q.user_id not in test_ids]       # test students are never used below
    train_ids = set(parts["train"])
    train = [q for q in seqs if q.user_id in train_ids]
    val = [q for q in seqs if q.user_id not in train_ids]
    assert not ({q.user_id for q in train} & {q.user_id for q in val})
    assert not ({q.user_id for q in seqs} & test_ids)

    base_sel = json.load(open(RESULTS / "baselines_val.json"))["selection"]["chosen"]
    pfa = B.PFA(base_sel["pfa_C"]).fit(train)
    bkt = B.BKT(base_sel["bkt_min_seq"]).fit(train)
    print(f"fitted PFA/BKT t={time.time() - t0:.0f}s", flush=True)

    # ---------------- risk head: C chosen on validation log-loss; beat rule from SPEC A1.8
    Xtr, ytr, _ = head_rows(train); Xva, yva, _ = head_rows(val)
    heads = {C: RiskHead(C).fit(Xtr, ytr) for C in C_GRID}
    ll = {str(C): M.logloss(yva, h.predict_rows(Xva)) for C, h in heads.items()}
    bestC = min(C_GRID, key=lambda C: ll[str(C)])
    head = heads[bestC]
    auc_rows = {}
    ci_sets = {}
    for k in KS:
        pairs = scored(val, k)
        y = np.array([p[1] for p in pairs]); sid = np.array([p[0].user_id for p in pairs])
        sc = {"risk_head": head.predict_rows(np.array([features(p[0].correct, k) for p in pairs])),
              "pfa": np.array([pfa.score(p[0], k) for p in pairs]),
              "bkt": np.array([bkt.score(p[0], k) for p in pairs])}
        auc_rows[k] = {m: M.auc(y, s) for m, s in sc.items()} | {"n": len(y)}
        ci_sets[k] = (y, sid, sc)
    mean_auc = {m: float(np.mean([auc_rows[k][m] for k in KS])) for m in ("risk_head", "pfa", "bkt")}
    best_base = max(("pfa", "bkt"), key=lambda m: mean_auc[m])
    margin = mean_auc["risk_head"] - mean_auc[best_base]
    head_wins = (mean_auc["risk_head"] > mean_auc["pfa"] and mean_auc["risk_head"] > mean_auc["bkt"] and margin >= BEAT_MARGIN)
    primary_method = "risk_head" if head_wins else best_base
    # paired student bootstrap of (head - best baseline) mean AUC over k
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    students = sorted({s for k in KS for s in ci_sets[k][1]})
    idx = {k: {s: np.where(ci_sets[k][1] == s)[0] for s in students} for k in KS}
    diffs = []
    for _ in range(500):
        pick = rng.choice(students, len(students), replace=True)
        d = []
        for k in KS:
            ii = np.concatenate([idx[k][s] for s in pick if len(idx[k][s])])
            y, _, sc = ci_sets[k]
            if len(set(y[ii].tolist())) < 2: continue
            d.append(M.auc(y[ii], sc["risk_head"][ii]) - M.auc(y[ii], sc[best_base][ii]))
        if len(d) == len(KS): diffs.append(np.mean(d))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    dump("riskhead_val.json", {
        "chosen_C": bestC, "val_logloss_by_C": ll, "coefficients_standardised": head.coefficients(),
        "n_train_rows": int(len(ytr)), "n_val_rows": int(len(yva)),
        "val_auc_by_k": {str(k): v for k, v in auc_rows.items()}, "mean_val_auc": mean_auc,
        "best_baseline": best_base, "margin_over_best_baseline": margin, "beat_margin_required": BEAT_MARGIN,
        "head_beats_rule": bool(head_wins), "primary_risk_method": primary_method,
        "paired_bootstrap_margin_ci95": [float(lo), float(hi)], "bootstrap_draws_used": len(diffs),
    })
    print(f"risk head done t={time.time() - t0:.0f}s  mean AUC {mean_auc}  primary={primary_method}", flush=True)

    # ---------------- decision tables on validation (superset with leavers)
    full = P.Table(val, include_leavers=True)
    raw = {
        "risk_head": P.score_matrix(full, lambda q, i: head.score(q, i)),
        "pfa": P.score_matrix(full, lambda q, i: pfa.score(q, i)),
        "bkt": P.score_matrix(full, lambda q, i: bkt.score(q, i)),
    }
    prim = full.subset(~full.leaver)
    prim_raw = {m: S[~full.leaver] for m, S in raw.items()}
    # Platt recalibration on validation decision rows of the primary scope (A1.9)
    cal, calib = {}, {}
    for m, S in prim_raw.items():
        mask = prim.valid
        rows_y = np.repeat(prim.y[:, None], P.NJ, axis=1)[mask]
        a, b = P.platt_fit(rows_y, S[mask])
        calib[m] = {"intercept": a, "slope": b, "n_rows": int(mask.sum()),
                    "auc_rows": M.auc(rows_y, S[mask])}
        cal[m] = P.platt_apply(raw[m], a, b)
    print(f"tables + calibration t={time.time() - t0:.0f}s", flush=True)

    def run_config(table, kind, calS, fixed_tau=None):
        """Per risk method: sweep, choose tau per profile (or apply fixed_tau); plus 0/1 methods."""
        err = table.err[kind]
        out = {"methods": {}, "sizes": P.evaluate_fire(table, err, np.zeros_like(err), None, None)["n_sequences"]}
        for m, S in calS.items():
            sw = P.sweep_tau(table, err, S)
            out["methods"][m] = {}
            for prof, (wi, wd) in P.PROFILES.items():
                tau_c, J = P.choose_tau(sw, wi, wd)
                tau = tau_c if fixed_tau is None else fixed_tau[m][prof]
                met = P.with_cost(sw[float(tau)], wi, wd)
                out["methods"][m][prof] = {"tau": float(tau), "tau_rechosen_here": float(tau_c), **met}
        zero_one = {"always_help": table.valid & err, "three_wrong_rule": P.three_wrong_fire(table, kind)}
        out["zero_one"] = {}
        for name, F in zero_one.items():
            out["zero_one"][name] = {prof: P.with_cost(P.evaluate_fire(table, err, F, None, None), wi, wd)
                                     for prof, (wi, wd) in P.PROFILES.items()}
        return out

    results = {"profiles": {k: {"w_interrupt": v[0], "w_delay": v[1]} for k, v in P.PROFILES.items()},
               "primary_risk_method": primary_method, "recalibration_platt_on_val": calib,
               "note": "Validation split only; Platt calibrator and tau share the validation split, so costs are optimistic."}
    cal_prim = {m: S[~full.leaver] for m, S in cal.items()}
    main = run_config(prim, "primary", cal_prim)
    # random baseline with matched number of intervened sequences
    for m in main["methods"]:
        for prof, (wi, wd) in P.PROFILES.items():
            tgt = main["methods"][m][prof]["interventions"]
            main["methods"][m][prof]["random_matched"] = P.random_matched(
                prim, prim.err["primary"], tgt, wi, wd, RANDOM_DRAWS, RANDOM_SEED)
    results["main_leavers_dropped_primary_error"] = main
    tau_carry = {m: {prof: main["methods"][m][prof]["tau"] for prof in P.PROFILES} for m in cal_prim}
    # sensitivity 1: leavers counted as non-wheel-spinning
    results["sens_leavers_as_non_ws__tau_carried_over"] = run_config(full, "primary", cal, tau_carry)
    results["sens_leavers_as_non_ws__tau_rechosen"] = run_config(full, "primary", cal)
    # sensitivity 2: verified error includes hint-first (primary scope)
    results["sens_incl_hint_first__tau_carried_over"] = run_config(prim, "incl_hint_first", cal_prim, tau_carry)
    results["sens_incl_hint_first__tau_rechosen"] = run_config(prim, "incl_hint_first", cal_prim)
    results["scope_counts"] = {
        "primary_scope_sequences": int(len(prim.y)), "primary_scope_students": int(len(set(prim.sid.tolist()))),
        "primary_y1": int(prim.y.sum()), "leavers_added": int(full.leaver.sum()),
        "decision_cells_valid": int(prim.valid.sum()),
        "verified_error_cells_primary": int((prim.valid & prim.err_primary).sum()),
        "verified_error_cells_incl_hint_first": int((prim.valid & prim.err_all).sum()),
    }
    dump("policy_val.json", results)
    print(f"wrote policy_val.json t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
