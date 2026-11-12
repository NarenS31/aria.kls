"""E1: missed help versus first hints per student on the pooled CV held-out folds (exploratory; ANALYSIS_PLAN.md E1)."""
import time

import numpy as np

import common as C
import cv_heldout as CVH
import figstyle as FS
from aria_pace import policy as P
from aria_pace import stage5 as S5

RANDOM_AT_TAU = (0.09, 0.19, 0.30, 0.45, 0.57, 0.70)


def metrics(tot, always_fired, S):
    """tot (..., 8). hints/student, reduction vs always-help, missed-help rate."""
    with np.errstate(divide="ignore", invalid="ignore"):
        hints = tot[..., 0] / S
        red = 1.0 - tot[..., 0] / always_fired
        miss = np.where(tot[..., 2] > 0, tot[..., 3] / np.where(tot[..., 2] > 0, tot[..., 2], 1), np.nan)
    return hints, red, miss


def summarise(A, always_A, counts, S):
    """A: (S, 8) per-student totals of one policy. Point and 95% CI of the three metrics (shared resamples)."""
    tot, al = A.sum(0), always_A.sum(0)
    tb, ab = counts @ A, counts @ always_A
    h, r, m = metrics(tot, al[0], S)
    hb, rb, mb = metrics(tb, ab[:, 0], S)
    return {"hints_per_student": h, "hints_ci95": C.ci(hb), "reduction_vs_always_help": r, "reduction_ci95": C.ci(rb),
            "missed_help_rate": m, "missed_ci95": C.ci(mb), "interventions": int(tot[0]), "missed_help": int(tot[3]),
            "stuck_with_help_opportunity": int(tot[2])}


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
    counts = np.random.RandomState(C.SEEDS["e1_bootstrap"]).multinomial(S, np.ones(S) / S, size=C.N_BOOT).astype(float)
    always_A = agg.by_student(S5.seq_vec(H, err, H.valid & err))
    out = {"analysis": "E1 trade-off curve", "data": "pooled held-out folds of the A7 5-fold CV (train+validation students; no test student)",
           "n_students": S, "n_sequences": int(len(H.y)), "n_stuck": int(H.y.sum()), "fold_info": d["fold_info"], "hyperparameters": d["hyper"],
           "tau_grid": P.TAU_GRID.tolist(), "bootstrap": {"draws": C.N_BOOT, "seed": C.SEEDS["e1_bootstrap"], "unit": "student"},
           "always_help": summarise(always_A, always_A, counts, S), "curves": {}, "n_rule": {}, "random_matched": {}, "frozen_tau_markers": {},
           "cal_chosen_tau_points": {}}
    grid_tot = {}
    for m in S5.RISK_METHODS:
        Sx = np.nan_to_num(d["cal"][m], nan=-1.0)
        As = [agg.by_student(S5.seq_vec(H, err, Sx >= tau)) for tau in P.TAU_GRID]
        grid_tot[m] = np.array([a.sum(0) for a in As])
        rows = [summarise(a, always_A, counts, S) for a in As]
        out["curves"][m] = {k: [r[k] for r in rows] for k in rows[0]}
        log(f"curve {m} done")
        if m == "bkt":
            for prof, tau in C.FROZEN_TAUS.items():
                j = int(np.argmin(np.abs(P.TAU_GRID - tau)))
                out["frozen_tau_markers"][prof] = {"tau": tau, **rows[j]}
        for prof in P.PROFILES:
            out["cal_chosen_tau_points"].setdefault(m, {})[prof] = summarise(
                agg.by_student(S5.seq_vec(H, err, CVH.chosen_fire(d, m, prof))), always_A, counts, S)
    for n in (1, 2, 3, 4, 5):
        out["n_rule"][str(n)] = summarise(agg.by_student(S5.seq_vec(H, err, P.run_fire(H, "primary", n))), always_A, counts, S)
    out["n_rule"]["note"] = "n = 1 is always-help; n = 3 is the untuned three-wrong rule (SPEC 5); the frozen tuned Quiet rule is n = 2"
    for tau in RANDOM_AT_TAU:
        j = int(np.argmin(np.abs(P.TAU_GRID - tau)))
        target = int(round(grid_tot["bkt"][j, 0]))
        vec, _ = S5.random_matched_vec(H, err, target, S5.RANDOM_DRAWS, C.SEEDS["random_policy"])
        out["random_matched"][f"{tau:.2f}"] = {"matched_to_bkt_tau": tau, "target_interventions": target,
                                               **summarise(agg.by_student(vec), always_A, counts, S)}
    log("baselines done")
    # Cross-check against A7 (cv_tradeoff_val.json, D3): the CAL-chosen-tau BKT points must match.
    a7 = C.read_result("cv_tradeoff_val.json")["definitions"]["D3"]
    out["a7_crosscheck"] = {prof: {"a7_saving": a7["chosen_tau_pooled_result"][prof]["saving"],
                                   "e1_saving": out["cal_chosen_tau_points"]["bkt"][prof]["reduction_vs_always_help"],
                                   "a7_missed": a7["chosen_tau_pooled_result"][prof]["missed"],
                                   "e1_missed": out["cal_chosen_tau_points"]["bkt"][prof]["missed_help_rate"],
                                   "a7_fold_taus": a7["tau_movement"][prof]["fold_taus"],
                                   "e1_fold_taus": [fi["cal_chosen"]["taus"]["bkt"][prof] for fi in d["fold_info"]]} for prof in P.PROFILES}
    # Descriptive: missed help of each risk score at the hint rate of each frozen BKT setting (linear interpolation along the sweep; no interval).
    out["missed_at_matched_hint_rate"] = {}
    for prof, mk in out["frozen_tau_markers"].items():
        row = {"hints_per_student": mk["hints_per_student"]}
        for m in S5.RISK_METHODS:
            h = np.array(out["curves"][m]["hints_per_student"]); ms = np.array(out["curves"][m]["missed_help_rate"], float)
            o = np.argsort(h)
            row[m] = float(np.interp(mk["hints_per_student"], h[o], ms[o]))
        out["missed_at_matched_hint_rate"][prof] = row
    out["missed_at_matched_hint_rate"]["note"] = "point estimates only, interpolated along each pooled held-out sweep"
    out["figure"] = plot(out)
    p = C.write("e1_tradeoff.json", out, t0)
    log(f"wrote {p}")


def plot(o):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FS.WIDTH, 2.55), layout="constrained")
    b = o["curves"]["bkt"]
    x = np.array(b["hints_per_student"]); y = 100 * np.array(b["missed_help_rate"], float)
    ci_ = 100 * np.array(b["missed_ci95"], float); lo, hi = ci_[:, 0], ci_[:, 1]
    ax.fill_between(x, lo, hi, color="#2457A6", alpha=0.18, lw=0)
    ax.plot(x, y, color="#2B2B2B", lw=1.5, label="BKT (threshold sweep)", zorder=3)
    for m, col, ls, lab in (("pfa", FS.OI["orange"], "--", "PFA"), ("risk_head", FS.OI["sky"], ":", "Risk head")):
        c = o["curves"][m]
        ax.plot(c["hints_per_student"], 100 * np.array(c["missed_help_rate"], float), color=col, ls=ls, lw=1.2, label=lab, zorder=2)
    nx = [o["n_rule"][str(n)]["hints_per_student"] for n in range(1, 6)]
    ny = [100 * o["n_rule"][str(n)]["missed_help_rate"] for n in range(1, 6)]
    ax.plot(nx, ny, color=FS.OI["vermillion"], lw=0.8, marker="D", ms=4, label="n-wrong rule, n=1-5", zorder=4)
    arrow = dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=0, shrinkB=2)
    for n, xx, yy in zip(range(1, 6), nx, ny):
        if n == 1:
            ax.annotate("always-\nhelp", (xx, yy), xytext=(0, 6), textcoords="offset points", fontsize=8, color="#2B2B2B",
                        ha="center", va="bottom", linespacing=0.9)
        else:
            ax.annotate(f"n={n} (untuned)" if n == 3 else f"n={n}", (xx, yy), xytext=(4, -2), textcoords="offset points", fontsize=8, color="#2B2B2B")
    rx = [v["hints_per_student"] for v in o["random_matched"].values()]
    ry = [100 * v["missed_help_rate"] for v in o["random_matched"].values()]
    ax.scatter(rx, ry, marker="x", s=16, color=FS.OI["gray"], lw=0.9, label="Random, matched rate", zorder=4)
    handles = []
    for prof, mk in o["frozen_tau_markers"].items():
        handles.append(ax.scatter([mk["hints_per_student"]], [100 * mk["missed_help_rate"]], s=34, marker=FS.SETTING_MARKER[prof],
                                  color=FS.SETTING[prof], edgecolor="white", lw=0.8, zorder=6, label=f"{prof.capitalize()} (\u03c4={mk['tau']:.2f})"))
    leg2 = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.60), frameon=False, title="Frozen thresholds (BKT)",
                     title_fontsize=8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    for t, prof in zip(leg2.get_texts(), o["frozen_tau_markers"]):
        t.set_color(FS.SETTING[prof])
    ax.add_artist(leg2)
    ax.set_xlabel("First hints per student")
    ax.set_ylabel("Missed help (% of stuck)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, max(nx) * 1.12)
    ax.grid(True, axis="y")
    ax.legend(handles=[h for h in ax.get_legend_handles_labels()[0] if h not in handles], loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    return FS.save(fig, "e1_tradeoff")


if __name__ == "__main__":
    main()
