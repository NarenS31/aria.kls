"""E2: calibration of the BKT stuck risk before and after Platt scaling (exploratory; ANALYSIS_PLAN.md E2).

Primary: validation primary-scope decision cells (tries 3-9), BKT fitted on train, frozen Platt (in-sample on these rows).
Secondary: pooled CV held-out cells with each fold's CAL-fitted Platt (not in-sample).
"""
import time

import numpy as np

import common as C
import cv_heldout as CVH
import figstyle as FS
from aria_pace import baselines as B
from aria_pace import metrics as M
from aria_pace import policy as P
from aria_pace import stage5 as S5

BINS = 10


def cell_arrays(tab, score):
    r, j = np.where(tab.valid)
    return {"y": tab.y[r].astype(float), "s": score[r, j], "sid": tab.sid[r], "try": P.DEC[j]}


def calib(y, s, sid, counts_seed):
    """ECE (10 equal-count bins, metrics.reliability), Brier, per-bin rates; 95% student-bootstrap CIs with bins fixed on the full sample."""
    ece, rel = M.reliability(y, s, BINS)
    order = np.argsort(s, kind="stable")
    bin_of = np.empty(len(s), int)
    for b, g in enumerate(np.array_split(order, BINS)):
        bin_of[g] = b
    uniq, codes = np.unique(sid, return_inverse=True)
    nS = len(uniq)
    # per-student per-bin sums: count, sum y, sum s ; per-student brier sums
    cnt = np.zeros((nS, BINS)); sy = np.zeros((nS, BINS)); ss = np.zeros((nS, BINS))
    np.add.at(cnt, (codes, bin_of), 1); np.add.at(sy, (codes, bin_of), y); np.add.at(ss, (codes, bin_of), s)
    br = np.zeros(nS); np.add.at(br, codes, (s - y) ** 2)
    counts = np.random.RandomState(counts_seed).multinomial(nS, np.ones(nS) / nS, size=C.N_BOOT).astype(float)
    cb, yb, sb = counts @ cnt, counts @ sy, counts @ ss
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_b, pred_b = yb / cb, sb / cb
        ece_b = np.nansum(cb * np.abs(pred_b - obs_b), 1) / cb.sum(1)
        brier_b = (counts @ br) / cb.sum(1)
    obs_ci, pred_ci = C.ci(obs_b), C.ci(pred_b)
    bins = [{"mean_predicted": rel[b]["mean_score"], "observed_stuck_rate": rel[b]["obs_rate"], "n_cells": rel[b]["n"],
             "observed_ci95": [obs_ci[0][b], obs_ci[1][b]], "predicted_ci95": [pred_ci[0][b], pred_ci[1][b]]} for b in range(len(rel))]
    return {"n_cells": int(len(y)), "n_students": int(nS), "n_stuck_cells": int(y.sum()), "base_rate": float(y.mean()),
            "ece_10bin": ece, "ece_ci95": C.ci(ece_b), "brier": M.brier(y, s), "brier_ci95": C.ci(brier_b), "bins": bins,
            "calibration_line": M.calibration_line(y, s)}


def both(cells_raw, cells_cal, seed, min_cells=200, min_pos=20):
    if len(cells_raw["y"]) < min_cells or cells_raw["y"].sum() < min_pos:
        return {"too_few": True, "n_cells": int(len(cells_raw["y"])), "n_stuck_cells": int(cells_raw["y"].sum())}
    return {"before_platt": calib(cells_raw["y"], cells_raw["s"], cells_raw["sid"], seed),
            "after_platt": calib(cells_cal["y"], cells_cal["s"], cells_cal["sid"], seed)}


def sub(c, m):
    return {k: v[m] for k, v in c.items()}


def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    frozen = C.read_result("frozen_val.json")
    min_seq = frozen["models"]["bkt_min_seq"]
    a, b = frozen["platt"]["bkt"]["intercept"], frozen["platt"]["bkt"]["slope"]
    bkt = B.BKT(min_seq).fit(train)
    tab = P.Table(val, include_leavers=False)
    C.assert_no_test(tab.sid, parts)
    raw = P.score_matrix(tab, bkt.score)
    cal = P.platt_apply(raw, a, b)
    cr, cc = cell_arrays(tab, raw), cell_arrays(tab, cal)
    seed = C.SEEDS["e2_bootstrap"]
    out = {"analysis": "E2 calibration", "score": "BKT stuck risk P(no mastery in k+1..10 | first k), BKT fit on train (min_seq %d)" % min_seq,
           "platt_frozen": {"intercept": a, "slope": b},
           "rows": "validation primary-scope decision cells, tries 3-9 (the rows the frozen Platt was fitted on); cells of one sequence share its label",
           "caveat": "After-Platt validation calibration is in-sample: the calibrator was fitted on these same rows. See cv_heldout for an out-of-sample check.",
           "bootstrap": {"draws": C.N_BOOT, "seed": seed, "unit": "student", "bins": "10 equal-count bins fixed on the full sample"},
           "validation_all_tries": both(cr, cc, seed), "validation_per_try": {}}
    for t in (3, 5, 7):
        m = cr["try"] == t
        out["validation_per_try"][str(t)] = both(sub(cr, m), sub(cc, m), seed)
    # Verified-error cells only (where a decision is actually taken), descriptive.
    me = tab.err["primary"][tab.valid]
    out["validation_verified_error_cells"] = both(sub(cr, me), sub(cc, me), seed)
    log("validation calibration done")
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    out["cv_heldout"] = both(cell_arrays(H, d["raw"]["bkt"]), cell_arrays(H, d["cal"]["bkt"]), seed)
    out["cv_heldout"]["note"] = "Pooled CV held-out cells; each fold calibrated with Platt fitted on its own CAL part (not in-sample)."
    out["figure"] = plot(out)
    log(f"wrote {C.write('e2_calibration.json', out, t0)}")


def plot(o):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FS.WIDTH, 2.7), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#9A9A9A", lw=0.8, ls="--", zorder=1)
    v = o["validation_all_tries"]
    for key, col, mk, lab in (("before_platt", FS.OI["vermillion"], "s", "Raw BKT"),
                              ("after_platt", "#2457A6", "o", "After Platt")):
        r = v[key]
        xs = np.array([bb["mean_predicted"] for bb in r["bins"]]); ys = np.array([bb["observed_stuck_rate"] for bb in r["bins"]])
        lo = np.array([bb["observed_ci95"][0] for bb in r["bins"]]); hi = np.array([bb["observed_ci95"][1] for bb in r["bins"]])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=col, marker=mk, ms=4, lw=1.0, capsize=1.5, elinewidth=0.8,
                    label=f"{lab}: ECE {r['ece_10bin']:.3f}", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted stuck risk (bin mean)")
    ax.set_ylabel("Observed stuck rate")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.4, borderaxespad=0.1)
    return FS.save(fig, "e2_calibration")


if __name__ == "__main__":
    main()
