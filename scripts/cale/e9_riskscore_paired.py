"""E9: paired comparison of BKT, PFA and the risk head at BKT's Quiet hint rate (CV held-out; exploratory)."""
import time

import numpy as np

import common as C
import cv_heldout as CVH
from aria_pace import stage5 as S5

SEED = 20261118


def match_threshold(H, err, S, target):
    """Bisection on one threshold so the number of intervened sequences is as close as possible to target (labels unused)."""
    base = H.valid & err
    Sx = np.nan_to_num(S, nan=-1.0)
    lo, hi, best = 0.0, 1.0, (None, 10 ** 9)
    for _ in range(60):
        t = (lo + hi) / 2
        c = int((base & (Sx >= t)).any(1).sum())
        if abs(c - target) < best[1] or (abs(c - target) == best[1] and best[0] is not None and t > best[0]):
            best = (t, abs(c - target))
        if c > target: lo = t
        elif c < target: hi = t
        else: break
    return best[0]


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
    cnt = np.random.RandomState(SEED).multinomial(S, np.ones(S) / S, size=C.N_BOOT).astype(float)
    tau_b = C.FROZEN_TAUS["quiet"]
    vecs = {"bkt": S5.seq_vec(H, err, np.nan_to_num(d["cal"]["bkt"], nan=-1.0) >= tau_b)}
    target = int(vecs["bkt"][:, 0].sum())
    taus = {"bkt": tau_b}
    for m in ("pfa", "risk_head"):
        taus[m] = match_threshold(H, err, d["cal"][m], target)
        vecs[m] = S5.seq_vec(H, err, np.nan_to_num(d["cal"][m], nan=-1.0) >= taus[m])
    A = {m: agg.by_student(v) for m, v in vecs.items()}
    tb = {m: cnt @ a for m, a in A.items()}
    tot = {m: a.sum(0) for m, a in A.items()}
    per = {m: {"threshold": taus[m], "interventions": int(tot[m][0]), "hints_per_student": float(tot[m][0] / S),
               "missed": int(tot[m][3]), "missed_help_rate": float(tot[m][3] / tot[m][2]),
               "missed_ci95": C.ci(tb[m][:, 3] / tb[m][:, 2])} for m in A}
    comp = {}
    for m in ("pfa", "risk_head"):
        dm = tb["bkt"][:, 3] / tb["bkt"][:, 2] - tb[m][:, 3] / tb[m][:, 2]
        dh = (tb["bkt"][:, 0] - tb[m][:, 0]) / S
        comp[f"bkt_minus_{m}"] = {"diff_missed_help_rate": per["bkt"]["missed_help_rate"] - per[m]["missed_help_rate"], "ci95": C.ci(dm),
                                  "diff_hints_per_student": per["bkt"]["hints_per_student"] - per[m]["hints_per_student"], "diff_hints_ci95": C.ci(dh),
                                  "share_resamples_bkt_misses_more": float(np.mean(dm > 0))}
    out = {"analysis": "E9 paired comparison of risk scores at a matched hint rate",
           "target": {"interventions": target, "hints_per_student": target / S, "source": "BKT, frozen Quiet tau 0.57, per-fold CAL Platt"},
           "threshold_matching": "one threshold per model on pooled held-out per-fold-calibrated scores, bisection on the intervened-sequence count only (labels unused)",
           "bootstrap": {"draws": C.N_BOOT, "seed": SEED, "unit": "student", "paired": True}, "n_students": S,
           "per_model": per, "differences": comp}
    log(f"wrote {C.write('e9_riskscore_paired.json', out, t0)}")


if __name__ == "__main__":
    main()
