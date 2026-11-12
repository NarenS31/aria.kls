"""E8: labelled variant where a hint-first attempt at tries 3-9 also counts as a gate error (CV held-out; exploratory).

The original definitions are unchanged; this builds a separate error matrix and passes it to the same stage5.seq_vec.
"""
import time

import numpy as np

import common as C
import cv_heldout as CVH
from aria_pace import policy as P
from aria_pace import stage5 as S5

SEED = 20261117


def opp_error(q, t):
    """Variant error at 0-based opportunity t: wrong first answer (code 0) or hint-first attempt (code 1)."""
    return q.correct[t] == 0 and q.first_action[t] in (0, 1)


def variant_err(H):
    E = np.zeros(H.valid.shape, bool)
    for r, q in enumerate(H.seqs):
        for j, i in enumerate(P.DEC):
            if H.valid[r, j]:
                E[r, j] = opp_error(q, i - 1)
    return E


def variant_run_fire(H, n):
    """n consecutive variant errors ending at decision point i (same structure as policy.run_fire)."""
    F = np.zeros(H.valid.shape, bool)
    for r, q in enumerate(H.seqs):
        for j, i in enumerate(P.DEC):
            if H.valid[r, j] and i >= n:
                F[r, j] = all(opp_error(q, t) for t in range(i - n, i))
    return F


def summarise(A, al, cnt, S):
    t, a = A.sum(0), al.sum(0)
    tb, ab = cnt @ A, cnt @ al
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"hints_per_student": float(t[0] / S), "hints_ci95": C.ci(tb[:, 0] / S),
                "reduction_vs_always_help": float(1 - t[0] / a[0]), "reduction_ci95": C.ci(1 - tb[:, 0] / ab[:, 0]),
                "missed_help_rate": float(t[3] / t[2]), "missed_ci95": C.ci(tb[:, 3] / tb[:, 2]),
                "missed": int(t[3]), "stuck_with_help_opportunity": int(t[2]), "interventions": int(t[0])}


def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    agg = S5.Agg(H.sid)
    cnt = np.random.RandomState(SEED).multinomial(agg.S, np.ones(agg.S) / agg.S, size=C.N_BOOT).astype(float)
    Sx = np.nan_to_num(d["cal"]["bkt"], nan=-1.0)
    out = {"analysis": "E8 hint-first attempts as decision points (labelled variant)",
           "variant_error": "correct == 0 and first_action in {0, 1} at decision point i (tries 3-9); code-2 rows excluded",
           "primary_error": "correct == 0 and first_action == 0 (unchanged; reported alongside)",
           "thresholds": "frozen BKT taus 0.57 / 0.19 / 0.09 on per-fold CAL-calibrated risk (not re-chosen)",
           "bootstrap": {"draws": C.N_BOOT, "seed": SEED, "unit": "student"}, "n_students": agg.S}
    for name, err, fire2 in (("variant", variant_err(H), variant_run_fire(H, 2)),
                             ("primary", H.err["primary"], P.run_fire(H, "primary", 2))):
        al = agg.by_student(S5.seq_vec(H, err, H.valid & err))
        res = {"error_cells": int((H.valid & err).sum()), "always_help_hints_per_student": float(al.sum(0)[0] / agg.S),
               "stuck_with_help_opportunity": int(al.sum(0)[2])}
        for prof, tau in C.FROZEN_TAUS.items():
            res[prof] = {"tau": tau, **summarise(agg.by_student(S5.seq_vec(H, err, Sx >= tau)), al, cnt, agg.S)}
        res["two_wrong_rule"] = summarise(agg.by_student(S5.seq_vec(H, err, fire2)), al, cnt, agg.S)
        out[name] = res
    log(f"wrote {C.write('e8_hintfirst_variant.json', out, t0)}")


if __name__ == "__main__":
    main()
