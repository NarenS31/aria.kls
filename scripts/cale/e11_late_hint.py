"""E11: late-hint decomposition on the pooled CV held-out folds (exploratory, post hoc; read-only on existing artefacts).

Step 0 reproduces the paper's CV numbers from the cached E1 predictions and STOPS if they differ.
A "late" first hint is one given at decision point i >= 8 on a stuck sequence: after the error at i the streak is 0
and only 10 - i < 3 opportunities remain, so 3-in-a-row mastery by opportunity 10 (T_DETERMINE) is already impossible
(the task's r_k = 1.00). "Early" = i <= 7 (mastery still possible).
No test student, no file under aria_pace/results/ is written. Output: aria_pace/results_cale/e11_late_hint.json
"""
import sys
import time

import numpy as np

import common as C
import cv_heldout as CVH
import e9_riskscore_paired as E9
from aria_pace import policy as P
from aria_pace import stage5 as S5
from aria_pace.constants import T_DETERMINE

SEED = 20261123
LAST_POSSIBLE = 7          # i <= 7: three more correct fit in opportunities i+1..10
PROFILES = ("quiet", "balanced", "supportive")


def first_info(H, err, fire):
    """Per-sequence arrays for a boolean fire matrix (first firing only)."""
    f = fire & H.valid & err
    fired = f.any(1)
    i_first = np.where(fired, f.argmax(1) + 3, 0)
    opp = H.valid & err
    has = opp.any(1)
    i_err = np.where(has, opp.argmax(1) + 3, 0)
    y1opp = (H.y == 1) & has
    return dict(fired=fired, i_first=i_first, i_err=i_err, y1opp=y1opp, unnec=fired & (H.y == 0))


def vec(info, last_possible=LAST_POSSIBLE):
    """(N, 6): fired, y1opp, missed_all, missed_early, early_hint, delay."""
    fired, y1opp, i_first, i_err = info["fired"], info["y1opp"], info["i_first"], info["i_err"]
    early = fired & (i_first <= last_possible)
    delay = np.where(y1opp, np.where(fired, i_first - i_err, T_DETERMINE - i_err), 0)
    return np.column_stack([fired, y1opp, y1opp & ~fired, y1opp & ~early, early, delay]).astype(float)


def rate_ci(A, counts, num, den):
    tot = A.sum(0)
    tb = counts @ A
    r = tot[num] / tot[den]
    return float(r), C.ci(tb[:, num] / tb[:, den])


def random_infos(H, err, target, draws, seed):
    """Same draws/bisection as stage5.random_matched_vec; returns the draw-averaged (N,6) vec."""
    rng = np.random.RandomState(seed)
    base = H.valid & err
    acc = np.zeros((len(H.y), 6))
    for _ in range(draws):
        R = rng.rand(*H.valid.shape)
        lo, hi, best = 0.0, 1.0, (0.5, 10 ** 9)
        for _ in range(40):
            t = (lo + hi) / 2
            cnt = int((base & (R >= t)).any(1).sum())
            if abs(cnt - target) < best[1]:
                best = (t, abs(cnt - target))
            if cnt > target: lo = t
            elif cnt < target: hi = t
            else: break
        acc += vec(first_info(H, err, base & (R >= best[0])))
    return acc / draws


def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    err = H.err["primary"]
    agg = S5.Agg(H.sid)
    S = agg.S
    Sx = {m: np.nan_to_num(d["cal"][m], nan=-1.0) for m in S5.RISK_METHODS}
    fires = {p: Sx["bkt"] >= C.FROZEN_TAUS[p] for p in PROFILES}
    fires["always_help"] = H.valid & err
    fires["two_errors_tuned"] = P.run_fire(H, "primary", 2)
    fires["three_errors"] = P.run_fire(H, "primary", 3)
    infos = {k: first_info(H, err, f) for k, f in fires.items()}

    # ---------------- step 0: reproduce the paper's CV numbers (E1 file + recomputation)
    e1 = C.read_result  # noqa  (E1 lives in results_cale, read directly below)
    import json
    E1 = json.loads((C.OUT / "e1_tradeoff.json").read_text())
    rep = {}
    for k, ref in (("quiet", E1["frozen_tau_markers"]["quiet"]), ("two_errors_tuned", E1["n_rule"]["2"]), ("three_errors", E1["n_rule"]["3"])):
        v = S5.seq_vec(H, err, fires[k])
        tot = v.sum(0)
        always = S5.seq_vec(H, err, fires["always_help"]).sum(0)
        rep[k] = {"recomputed_reduction": float(1 - tot[0] / always[0]), "recomputed_missed": float(tot[3] / tot[2]), "recomputed_missed_n": int(tot[3]),
                  "recomputed_hints_per_student": float(tot[0] / S), "e1_reduction": ref["reduction_vs_always_help"], "e1_missed": ref["missed_help_rate"],
                  "e1_missed_n": ref["missed_help"], "e1_hints_per_student": ref["hints_per_student"]}
    paper = {"quiet_reduction_pct": 50.7, "quiet_missed_pct": 6.2, "two_errors_missed_pct": 27.3}
    got = {"quiet_reduction_pct": round(100 * rep["quiet"]["recomputed_reduction"], 1), "quiet_missed_pct": round(100 * rep["quiet"]["recomputed_missed"], 1),
           "two_errors_missed_pct": round(100 * rep["two_errors_tuned"]["recomputed_missed"], 1)}
    ok = got == paper and all(rep[k]["recomputed_missed_n"] == rep[k]["e1_missed_n"] for k in rep)
    log(f"step 0: paper {paper} recomputed {got} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print(json.dumps({"paper": paper, "recomputed": got, "detail": rep}, indent=1))
        sys.exit("STOP: CV numbers do not reproduce; no analysis written")

    counts = np.random.RandomState(SEED).multinomial(S, np.ones(S) / S, size=C.N_BOOT).astype(float)
    out = {"analysis": "E11 late-hint decomposition", "data": "pooled held-out folds of the A7 5-fold CV (no test student)",
           "definition": {"late_first_hint": f"first policy hint at decision point i >= {LAST_POSSIBLE + 1} (r_k = 1.00: mastery by opportunity {T_DETERMINE} impossible)",
                          "early_first_hint": f"i <= {LAST_POSSIBLE}", "basis": "after the error at i the streak is 0; 3 correct needed within i+1..10, so i <= 7",
                          "note": "r_k is not defined anywhere in the repo; this is the task's gloss. Sensitivity with the zero-slack boundary i >= 7 is reported."},
           "n_students": S, "n_sequences": int(len(H.y)), "n_stuck": int(H.y.sum()), "step0_reproduction": {"paper": paper, "recomputed": got, "match": ok, "detail": rep},
           "bootstrap": {"draws": C.N_BOOT, "seed": SEED, "unit": "student"}}

    # ---------------- A1: share of first policy hints at r_k = 1.00 (stuck sequences with a decision point)
    a1 = {}
    for k in list(PROFILES) + ["two_errors_tuned", "three_errors"]:
        inf = infos[k]
        sel = inf["y1opp"] & inf["fired"]
        i = inf["i_first"][sel]
        late = int((i >= LAST_POSSIBLE + 1).sum())
        a1[k] = {"stuck_with_decision_point": int(inf["y1opp"].sum()), "stuck_with_first_hint": int(sel.sum()), "late_hints_i_ge_8": late,
                 "late_share": late / int(sel.sum()), "by_opportunity": {str(j): int((i == j).sum()) for j in range(3, 10)},
                 "sensitivity_i_ge_7_zero_slack": {"count": int((i >= 7).sum()), "share": float((i >= 7).mean())}}
    out["A1_late_share_of_first_hints"] = a1

    # ---------------- A2: missed help counting only hints given while mastery was possible
    always_A = agg.by_student(vec(infos["always_help"]))
    target = int(infos["quiet"]["fired"].sum())
    methods = {k: infos[k] for k in ("quiet", "balanced", "supportive", "two_errors_tuned", "three_errors")}
    mats = {k: agg.by_student(vec(v)) for k, v in methods.items()}
    mats_i = dict(methods)
    mats["random_matched_to_quiet"] = agg.by_student(random_infos(H, err, target, S5.RANDOM_DRAWS, C.SEEDS["random_policy"]))
    thr = {"pfa": E9.match_threshold(H, err, d["cal"]["pfa"], target), "risk_head": E9.match_threshold(H, err, d["cal"]["risk_head"], target)}
    for m in ("pfa", "risk_head"):
        inf = first_info(H, err, Sx[m] >= thr[m])
        mats[f"{m}_matched_to_quiet"] = agg.by_student(vec(inf))
        infos[f"{m}_matched"] = inf
    a2 = {}
    for k, A in mats.items():
        tot = A.sum(0)
        m_all, ci_all = rate_ci(A, counts, 2, 1)
        m_early, ci_early = rate_ci(A, counts, 3, 1)
        a2[k] = {"stuck_with_help_opportunity": int(tot[1]), "hints_per_student_all": float(tot[0] / S), "early_hints_per_student": float(tot[4] / S),
                 "missed_all": {"rate": m_all, "ci95": ci_all, "count": round(float(tot[2]), 1)},
                 "missed_early_only": {"rate": m_early, "ci95": ci_early, "count": round(float(tot[3]), 1)},
                 "late_hints_counted_as_missed": round(float(tot[3] - tot[2]), 1)}
    a2["note"] = ("random is the draw-average of 200 draws (counts are expected values); PFA and risk head use one threshold matched to Quiet's intervened-sequence count (E9 rule), "
                  "thresholds: " + str({k: round(v, 4) for k, v in thr.items()}))
    a2["thresholds_matched"] = thr
    out["A2_missed_help_early_hints_only"] = a2
    # secondary denominator: stuck sequences that had an error at some decision point i <= 7 (help could have been early)
    opp_e = (H.valid & err)[:, : LAST_POSSIBLE - 2].any(1) & (H.y == 1)
    sec = {}
    for k in ("quiet", "balanced", "supportive", "two_errors_tuned", "three_errors", "pfa_matched", "risk_head_matched"):
        inf = infos[k]
        hit = inf["fired"] & (inf["i_first"] <= LAST_POSSIBLE)
        v = np.column_stack([opp_e, opp_e & ~hit]).astype(float)
        A = agg.by_student(v)
        r, c = rate_ci(A, counts, 1, 0)
        sec[k] = {"denominator": int(opp_e.sum()), "missed_early_only": {"rate": r, "ci95": c, "count": int(A.sum(0)[1])}}
    out["A2_secondary_denominator_stuck_with_error_at_i_le_7"] = sec

    # ---------------- A3: opportunity of first policy hint in stuck sequences
    a3 = {}
    for k in ("quiet", "two_errors_tuned", "balanced", "supportive", "three_errors"):
        inf = infos[k]
        i = inf["i_first"][inf["y1opp"] & inf["fired"]]
        q = np.percentile(i, [25, 50, 75])
        a3[k] = {"n": int(len(i)), "median": float(q[1]), "q1": float(q[0]), "q3": float(q[2]), "mean": float(i.mean()),
                 "share_of_stuck_with_decision_point_that_got_a_hint": float(len(i) / inf["y1opp"].sum())}
    out["A3_first_hint_opportunity_in_stuck"] = a3

    # ---------------- A4: I(tau) and delay D, same definitions as the J objective (stage5.seq_vec)
    a4 = {}
    for k in list(PROFILES) + ["always_help", "two_errors_tuned"]:
        inf = infos[k]
        v = S5.seq_vec(H, err, fires[k])
        tot = v.sum(0)
        dl = v[:, 4][inf["y1opp"]]
        a4[k] = {"I_unnecessary_per_student": float(tot[1] / S), "unnecessary_count": int(tot[1]), "D_total_per_student": float(tot[4] / S),
                 "D_total": float(tot[4]), "mean_delay_incl_censored": float(dl.mean()), "median_delay_incl_censored": float(np.median(dl)),
                 "iqr_delay": [float(np.percentile(dl, 25)), float(np.percentile(dl, 75))], "n_stuck_with_decision_point": int(len(dl)),
                 "mean_delay_fired_only": float(v[:, 4][v[:, 5] > 0].mean()), "censored_n": int(tot[3]),
                 "J_per_student_quiet_weights": float((3 * tot[1] + 1 * tot[4]) / S),
                 "J_per_student_own_profile": float((P.PROFILES[k][0] * tot[1] + P.PROFILES[k][1] * tot[4]) / S) if k in P.PROFILES else None}
    out["A4_I_and_D"] = a4
    out["A4_definitions"] = ("policy.evaluate_fire / stage5.seq_vec: unnecessary = first hint fired on a sequence with y=0 (mastered by opp 10); "
                             "delay on stuck sequences with a help opportunity = i_first - i_err if hinted, else T_DETERMINE(10) - i_err (censored); i_err = first verified error at i in 3..9")
    p = C.write("e11_late_hint.json", out, t0)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
