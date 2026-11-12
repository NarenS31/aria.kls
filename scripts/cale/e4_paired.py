"""E4: paired student-bootstrap comparison of BKT (Quiet) with the tuned two-wrong rule on the CV held-out folds (exploratory)."""
import time

import numpy as np

import common as C
import cv_heldout as CVH
from aria_pace import policy as P
from aria_pace import stage5 as S5


def rates(tot, S):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tot[..., 0] / S, tot[..., 3] / tot[..., 2]


def compare(A_bkt, A_rule, counts, S):
    hb, mb = rates(counts @ A_bkt, S)
    hr, mr = rates(counts @ A_rule, S)
    h0b, m0b = rates(A_bkt.sum(0), S)
    h0r, m0r = rates(A_rule.sum(0), S)
    dm, dh = mb - mr, hb - hr
    return {"bkt": {"missed_help_rate": m0b, "missed_ci95": C.ci(mb), "hints_per_student": h0b, "hints_ci95": C.ci(hb),
                    "missed": int(A_bkt.sum(0)[3]), "interventions": int(A_bkt.sum(0)[0])},
            "two_wrong_rule": {"missed_help_rate": m0r, "missed_ci95": C.ci(mr), "hints_per_student": h0r, "hints_ci95": C.ci(hr),
                               "missed": int(A_rule.sum(0)[3]), "interventions": int(A_rule.sum(0)[0])},
            "stuck_with_help_opportunity": int(A_bkt.sum(0)[2]),
            "diff_missed_help_rate_bkt_minus_rule": m0b - m0r, "diff_missed_ci95": C.ci(dm),
            "diff_hints_per_student_bkt_minus_rule": h0b - h0r, "diff_hints_ci95": C.ci(dh),
            "share_resamples_bkt_fewer_missed": float(np.mean(dm < 0)), "share_resamples_bkt_fewer_hints": float(np.mean(dh < 0))}


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
    counts = np.random.RandomState(C.SEEDS["e4_bootstrap"]).multinomial(S, np.ones(S) / S, size=C.N_BOOT).astype(float)
    Sx = np.nan_to_num(d["cal"]["bkt"], nan=-1.0)
    A_rule = agg.by_student(S5.seq_vec(H, err, P.run_fire(H, "primary", 2)))
    A_frozen = agg.by_student(S5.seq_vec(H, err, Sx >= C.FROZEN_TAUS["quiet"]))
    A_chosen = agg.by_student(S5.seq_vec(H, err, CVH.chosen_fire(d, "bkt", "quiet")))
    test = C.read_result("test_results.json")["main"]["variants"]["frozen"]["methods"]
    ctx = {k: {f: test[k]["quiet"][f] for f in ("interventions_per_student", "missed_help", "missed_help_share", "n_y1_with_help_opportunity")}
           for k in ("bkt", "n_rule_tuned")}
    out = {"analysis": "E4 paired comparison, BKT (Quiet) vs tuned n-wrong rule (n = 2)",
           "data": "pooled CV held-out folds (A7 folds; train+validation students). Per-sequence outputs of the locked test run do not exist, so E4 is NOT a test-set result.",
           "bootstrap": {"draws": C.N_BOOT, "seed": C.SEEDS["e4_bootstrap"], "unit": "student", "paired": True, "interval": "95% percentile"},
           "n_students": S, "primary_frozen_tau_0.57": compare(A_frozen, A_rule, counts, S),
           "sensitivity_cal_chosen_tau": compare(A_chosen, A_rule, counts, S),
           "cal_chosen_quiet_taus": [fi["cal_chosen"]["taus"]["bkt"]["quiet"] for fi in d["fold_info"]],
           "cal_chosen_quiet_n": [fi["cal_chosen"]["n"]["quiet"] for fi in d["fold_info"]],
           "locked_test_context_quoted_not_recomputed": {"source": "aria_pace/results/test_results.json main/frozen quiet", "values": ctx,
                                                         "note": "aggregates only; no paired interval possible without per-sequence outputs"}}
    log(f"wrote {C.write('e4_paired.json', out, t0)}")


if __name__ == "__main__":
    main()
