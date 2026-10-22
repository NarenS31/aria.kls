"""Stage 5c: figures and markdown tables from results/*.json. python -m aria_pace.report (after run_test)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import RESULTS

# Categorical slots 1-3 from the dataviz reference palette (validated all-pairs); neutral gray / ink for the rest.
COL = {"bkt": "#2a78d6", "pfa": "#eb6834", "risk_head": "#1baf7a", "random": "#8a8985", "n_rule": "#e87ba4", "always": "#0b0b0b"}
LAB = {"bkt": "BKT threshold", "pfa": "PFA threshold", "risk_head": "Risk head threshold", "random": "Random, matched rate",
       "n_rule": "n consecutive errors (n=1..6)", "always": "Always-help"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PROFILE_MARK = {"quiet": "o", "balanced": "s", "supportive": "^"}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors=MUTED, labelsize=9)


def frontier(res, cfg, title, path, optimistic):
    sw = res[cfg]["sweep"]
    var = next(iter(res[cfg]["variants"].values())) if cfg == "main" else res[cfg]["variants"]["rechosen"]
    panels = [("interventions_per_student", "mean_delay_incl_censored", "Interventions per student", "Mean help delay (opportunities)"),
              ("unnecessary_per_student", "missed_help_share", "Unnecessary interventions per student", "Missed-help share")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="#fcfcfb")
    for ax, (xk, yk, xl, yl) in zip(axes, panels):
        style(ax)
        for m in ("bkt", "pfa", "risk_head"):
            pts = [(r[xk], r[yk]) for r in sw["methods"][m] if r[xk] is not None and r[yk] is not None]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COL[m], lw=2, label=LAB[m], zorder=3, ls="-" if m == "bkt" else "--" if m == "pfa" else ":")
        fr = [(r[xk], r[yk]) for r in sw.get("random_frontier", []) if r[yk] is not None]
        if fr:
            xs, ys = zip(*sorted(fr))
            ax.plot(xs, ys, color=COL["random"], lw=2, label=LAB["random"], zorder=2, ls="-.")
        pts = [(r[xk], r[yk]) for n, r in sw["n_rule"].items() if r[yk] is not None]
        ax.plot(*zip(*pts), color=COL["n_rule"], lw=1.5, marker="D", ms=5, label=LAB["n_rule"], zorder=3)
        a = sw["n_rule"]["1"]
        ax.scatter([a[xk]], [a[yk]], s=70, color=COL["always"], zorder=5, label=LAB["always"])
        for prof, mk in PROFILE_MARK.items():
            v = var["methods"]["bkt"][prof]
            xv = v["interventions_per_student"] if xk.startswith("interv") else v["unnecessary_per_student"]
            yv = v["mean_delay_incl_censored"] if yk.startswith("mean") else v["missed_help_share"]
            ax.scatter([xv], [yv], s=110, marker=mk, facecolor="white", edgecolor=INK, lw=2, zorder=6)
            ax.annotate(prof, (xv, yv), textcoords="offset points", xytext=(7, 7), fontsize=9, color=INK)
        ax.set_xlabel(xl, color=INK, fontsize=10); ax.set_ylabel(yl, color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    sub = "VALIDATION (optimistic: calibrator and tau were chosen on this split)" if optimistic else "TEST (frozen validation values; curves are descriptive)"
    fig.suptitle(f"{title}\n{sub}. Open markers = frozen BKT operating points.", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor="#fcfcfb"); plt.close(fig)


def j_vs_tau(val, test, cfg, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), facecolor="#fcfcfb")
    grid = np.array(val[cfg]["sweep"]["tau_grid"])
    fv = val[cfg]["variants"]
    for ax, prof in zip(axes, ("quiet", "balanced", "supportive")):
        style(ax)
        for res, ls, lab in ((val, "-", "validation (optimistic)"), (test, "--", "test (diagnostic)")):
            if res is None:
                continue
            for m in ("bkt", "pfa", "risk_head"):
                J = [r["J"][prof] for r in res[cfg]["sweep"]["methods"][m]]
                ax.plot(grid, J, color=COL[m], ls=ls, lw=2 if m == "bkt" else 1.3, alpha=1 if m == "bkt" else 0.8,
                        label=f"{LAB[m]}, {lab}" if prof == "quiet" else None)
        for vname, v in fv.items():
            ax.axvline(v["taus"]["bkt"][prof], color=INK, lw=1, ls=":" if vname != "frozen" and vname != "rechosen" else "-", alpha=0.7)
        ax.set_title(prof, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("tau (calibrated risk)", color=INK, fontsize=10)
        if prof == "quiet":
            ax.set_ylabel("Cost J per student", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"Cost versus tau ({cfg}). Vertical lines: frozen BKT tau values.", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140, facecolor="#fcfcfb"); plt.close(fig)


def main():
    val = json.load(open(RESULTS / "val_results.json"))
    test = json.load(open(RESULTS / "test_results.json"))
    out = RESULTS.parent / "figures"
    out.mkdir(exist_ok=True)
    frontier(val, "main", "Interruption vs help-delay frontier", out / "frontier_val.png", True)
    frontier(test, "main", "Interruption vs help-delay frontier", out / "frontier_test.png", False)
    for cfg in ("main", "leavers", "hintfirst"):
        j_vs_tau(val, test, cfg, out / f"j_vs_tau_{cfg}.png")
    print("wrote figures", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
