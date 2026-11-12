"""E6: distribution across skills of the Quiet reduction and missed-help rate (CV held-out; descriptive only)."""
import time

import numpy as np

import common as C
import cv_heldout as CVH
import figstyle as FS
from aria_pace import stage5 as S5
from e3_errors import skill_names

MIN_STUCK = 30
TAU = C.FROZEN_TAUS["quiet"]


def dist(v):
    v = np.asarray(v, float)
    return {"n_skills": int(len(v)), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def main():
    t0 = time.time()
    log = C.log_fn()
    _, parts, train, val, _ = C.load_pool(log)
    d = CVH.get(train, val, log)
    H = d["H"]
    C.assert_no_test(H.sid, parts)
    err = H.err["primary"]
    Sx = np.nan_to_num(d["cal"]["bkt"], nan=-1.0)
    pol = S5.seq_vec(H, err, Sx >= TAU)
    alw = S5.seq_vec(H, err, H.valid & err)
    kc = np.array([q.kc for q in H.seqs])
    names = skill_names()
    rows = []
    for k in sorted(set(kc.tolist())):
        m = kc == k
        stuck_opp = pol[m, 2].sum()
        if stuck_opp < MIN_STUCK:
            continue
        rows.append({"skill_id": k, "skill_name": names.get(k, ""), "sequences": int(m.sum()), "students": int(len(set(H.sid[m].tolist()))),
                     "stuck_with_help_opportunity": int(stuck_opp), "always_help_interventions": int(alw[m, 0].sum()),
                     "quiet_interventions": int(pol[m, 0].sum()), "reduction_vs_always_help": float(1 - pol[m, 0].sum() / alw[m, 0].sum()),
                     "missed_help_rate": float(pol[m, 3].sum() / stuck_opp), "missed": int(pol[m, 3].sum()),
                     "stuck_rate": float(H.y[m].mean())})
    tot_stuck = pol[:, 2].sum()
    out = {"analysis": "E6 heterogeneity across skills", "operating_point": f"BKT, frozen Quiet tau {TAU}, per-fold CAL Platt (pooled CV held-out)",
           "skill": "sequence KC = skill_id of the first opportunity", "inclusion": f">= {MIN_STUCK} stuck sequences with a help opportunity",
           "n_skills_total": int(len(set(kc.tolist()))), "n_skills_included": len(rows),
           "share_of_stuck_with_help_opportunity_covered": float(sum(r["stuck_with_help_opportunity"] for r in rows) / tot_stuck),
           "pooled_reduction": float(1 - pol[:, 0].sum() / alw[:, 0].sum()), "pooled_missed_help_rate": float(pol[:, 3].sum() / tot_stuck),
           "reduction_distribution": dist([r["reduction_vs_always_help"] for r in rows]),
           "missed_help_distribution": dist([r["missed_help_rate"] for r in rows]),
           "correlation_stuck_rate_vs_reduction_spearman": spearman([r["stuck_rate"] for r in rows], [r["reduction_vs_always_help"] for r in rows]),
           "per_skill": sorted(rows, key=lambda r: r["reduction_vs_always_help"]),
           "note": "Descriptive only; no per-skill significance testing. Small skills have noisy rates."}
    out["figure"] = plot(rows, out)
    log(f"wrote {C.write('e6_skills.json', out, t0)}")


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic)


def plot(rows, o):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FS.WIDTH, 2.5), layout="constrained")
    x = 100 * np.array([r["reduction_vs_always_help"] for r in rows])
    y = 100 * np.array([r["missed_help_rate"] for r in rows])
    s = np.array([r["stuck_with_help_opportunity"] for r in rows])
    ax.scatter(x, y, s=8 + 40 * s / s.max(), color=FS.SETTING["quiet"], alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
    ax.axvline(100 * o["pooled_reduction"], color=FS.OI["gray"], lw=0.8, ls="--", zorder=1)
    ax.axhline(100 * o["pooled_missed_help_rate"], color=FS.OI["gray"], lw=0.8, ls="--", zorder=1)
    ax.text(100 * o["pooled_reduction"], 0.02, " pooled", transform=ax.get_xaxis_transform(), fontsize=8, color=FS.OI["gray"], va="bottom")
    ax.set_xlabel("Quiet reduction vs always-help (%)")
    ax.set_ylabel("Missed help (% of stuck)")
    ax.set_title(f"{len(rows)} skills, >= {MIN_STUCK} stuck each; size = stuck n", fontsize=8, loc="left")
    ax.grid(True)
    return FS.save(fig, "e6_skills")


if __name__ == "__main__":
    main()
