"""E12: CIs, paired differences and late-hint shares for the E11 late-hint decomposition (CV held-out, exploratory, read-only).

Seeds: single-policy CIs use the E1 seed (paper's main CV table, 20261112); the Quiet-vs-two-errors pair uses the E4 seed (20261114, the paper's -21.1 [-23.1,-19.0]);
BKT vs PFA/logistic pairs use the E9 seed (20261118). All: 1000 student multinomial resamples, percentile 95%.
"""
import json, sys, time
import numpy as np
import common as C
import cv_heldout as CVH
import e9_riskscore_paired as E9
import e11_late_hint as E11
from aria_pace import policy as P
from aria_pace import stage5 as S5


def counts_for(seed, S):
    return np.random.RandomState(seed).multinomial(S, np.ones(S) / S, size=C.N_BOOT).astype(float)


def random_avg(H, err, target, draws, seed):
    """Same draws as E11/E1; returns draw-averaged (N,3): early-missed, stuck-hinted, stuck-hinted-late."""
    rng = np.random.RandomState(seed); base = H.valid & err
    acc = np.zeros((len(H.y), 3))
    for _ in range(draws):
        R = rng.rand(*H.valid.shape); lo, hi, best = 0.0, 1.0, (0.5, 10 ** 9)
        for _ in range(40):
            t = (lo + hi) / 2; cnt = int((base & (R >= t)).any(1).sum())
            if abs(cnt - target) < best[1]: best = (t, abs(cnt - target))
            if cnt > target: lo = t
            elif cnt < target: hi = t
            else: break
        inf = E11.first_info(H, err, base & (R >= best[0]))
        v = E11.vec(inf); sh = inf["y1opp"] & inf["fired"]
        acc += np.column_stack([v[:, 3], sh, sh & (inf["i_first"] >= 8)])
    return acc / draws


def main():
    t0 = time.time(); log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log); H = d["H"]; C.assert_no_test(H.sid, parts)
    err = H.err["primary"]; agg = S5.Agg(H.sid); S = agg.S
    Sx = {m: np.nan_to_num(d["cal"][m], nan=-1.0) for m in S5.RISK_METHODS}
    fires = {p: Sx["bkt"] >= C.FROZEN_TAUS[p] for p in E11.PROFILES}
    fires["two_errors_tuned"] = P.run_fire(H, "primary", 2); fires["three_errors"] = P.run_fire(H, "primary", 3)
    target = int(E11.first_info(H, err, fires["quiet"])["fired"].sum())
    thr = {m: E9.match_threshold(H, err, d["cal"][m], target) for m in ("pfa", "risk_head")}
    fires["pfa_matched"] = Sx["pfa"] >= thr["pfa"]; fires["risk_head_matched"] = Sx["risk_head"] >= thr["risk_head"]
    # per-student columns: [y1opp, missed_early, stuck_hinted, stuck_hinted_late]
    cols = {}
    for k, f in fires.items():
        inf = E11.first_info(H, err, f); v = E11.vec(inf); sh = inf["y1opp"] & inf["fired"]
        cols[k] = np.column_stack([v[:, 1], v[:, 3], sh, sh & (inf["i_first"] >= 8)])
    ra = random_avg(H, err, target, S5.RANDOM_DRAWS, C.SEEDS["random_policy"])
    base1 = cols["quiet"][:, 0]
    cols["random_matched_to_quiet"] = np.column_stack([base1, ra[:, 0], ra[:, 1], ra[:, 2]])
    A = {k: agg.by_student(v) for k, v in cols.items()}
    c1 = counts_for(C.SEEDS["e1_bootstrap"], S)
    rows = {}
    for k, a in A.items():
        t = a.sum(0); tb = c1 @ a
        rows[k] = {"N_stuck_with_decision_point": int(t[0]), "missed_early_only_n": round(float(t[1]), 1), "missed_early_only_rate": float(t[1] / t[0]),
                   "ci95": C.ci(tb[:, 1] / tb[:, 0]), "first_hints_in_stuck": round(float(t[2]), 1), "late_i_ge_8": round(float(t[3]), 1),
                   "late_share": float(t[3] / t[2]), "late_share_ci95": C.ci(tb[:, 3] / tb[:, 2])}

    def pair(a, b, seed):
        cn = counts_for(seed, S); ta, tb_ = A[a].sum(0), A[b].sum(0)
        ra_, rb_ = cn @ A[a], cn @ A[b]
        dif = ra_[:, 1] / ra_[:, 0] - rb_[:, 1] / rb_[:, 0]
        return {"diff_pp": float(100 * (ta[1] / ta[0] - tb_[1] / tb_[0])), "ci95_pp": [float(100 * x) for x in C.ci(dif)], "seed": seed}
    # original paired difference (all hints counted), via stage5 vectors
    cn4 = counts_for(C.SEEDS["e4_bootstrap"], S)
    aq = agg.by_student(S5.seq_vec(H, err, fires["quiet"])); a2 = agg.by_student(S5.seq_vec(H, err, fires["two_errors_tuned"]))
    rq, r2 = cn4 @ aq, cn4 @ a2
    dd = rq[:, 3] / rq[:, 2] - r2[:, 3] / r2[:, 2]
    orig = {"diff_pp": float(100 * (aq.sum(0)[3] / aq.sum(0)[2] - a2.sum(0)[3] / a2.sum(0)[2])), "ci95_pp": [float(100 * x) for x in C.ci(dd)], "seed": C.SEEDS["e4_bootstrap"]}
    e4 = json.loads((C.OUT / "e4_paired.json").read_text())["primary_frozen_tau_0.57"]
    ok = abs(orig["diff_pp"] - 100 * e4["diff_missed_help_rate_bkt_minus_rule"]) < 1e-9 and abs(orig["ci95_pp"][0] - 100 * e4["diff_missed_ci95"][0]) < 1e-9 and abs(orig["ci95_pp"][1] - 100 * e4["diff_missed_ci95"][1]) < 1e-9
    log(f"original paired diff {orig}; matches e4_paired.json: {ok}")
    if not ok: sys.exit("STOP: original paired difference does not reproduce")
    pairs = {"original_all_hints_quiet_minus_two_errors": {**orig, "matches_e4_paired_json": ok},
             "early_only_quiet_minus_two_errors": pair("quiet", "two_errors_tuned", C.SEEDS["e4_bootstrap"]),
             "early_only_quiet_minus_risk_head_matched": pair("quiet", "risk_head_matched", 20261118),
             "early_only_quiet_minus_pfa_matched": pair("quiet", "pfa_matched", 20261118)}
    # delay decomposition
    dec = {}
    for k in ("quiet", "two_errors_tuned"):
        inf = E11.first_info(H, err, fires[k]); s = inf["y1opp"]; h = s & inf["fired"]; c = s & ~inf["fired"]
        dl = np.where(inf["fired"], inf["i_first"] - inf["i_err"], 10 - inf["i_err"])
        dec[k] = {"n_stuck": int(s.sum()), "hinted": int(h.sum()), "unhinted_censored": int(c.sum()),
                  "mean_i_err_all": float(inf["i_err"][s].mean()), "median_first_hint_opp": float(np.median(inf["i_first"][h])),
                  "mean_delay_hinted": float(dl[h].mean()), "mean_delay_censored": float(dl[c].mean()),
                  "D_from_hinted": int(dl[h].sum()), "D_from_censored": int(dl[c].sum()), "mean_delay_all": float(dl[s].mean()),
                  "median_delay_all": float(np.median(dl[s])), "share_delay_zero": float((dl[s] == 0).mean())}
    out = {"analysis": "E12 CIs, paired differences, late shares", "n_students": S, "matched_thresholds": thr, "per_policy": rows, "paired": pairs, "delay_decomposition": dec,
           "seeds": {"single_policy_ci": C.SEEDS["e1_bootstrap"], "quiet_vs_two_errors": C.SEEDS["e4_bootstrap"], "bkt_vs_pfa_risk_head": 20261118}}
    log(f"wrote {C.write('e12_late_hint_cis.json', out, t0)}")


if __name__ == "__main__":
    main()
