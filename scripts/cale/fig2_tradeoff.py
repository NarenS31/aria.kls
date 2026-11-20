"""Paper figure (trade-off): missed help vs first hints per student on the pooled CV held-out folds (CALE E1, post hoc).

Plot only. Reads aria_pace/results_cale/e1_tradeoff.json and recomputes nothing; no test data.
Bands: 95% student bootstrap (1000 draws, from the E1 file). Baselines and the three frozen BKT settings are points with
95% student-bootstrap error bars. Writes paper/cale/figures/fig2_tradeoff.{pdf,png}.
"""
import json

import numpy as np

import common as C
import figstyle as FS

import matplotlib.pyplot as plt  # noqa: E402  (figstyle sets the backend and rcParams)

OUTDIR = C.REPO / "paper" / "cale" / "figures"


def pct(v):
    return 100 * np.array(v, float)


def point(ax, r, **kw):
    x, y = r["hints_per_student"], 100 * r["missed_help_rate"]
    (ylo, yhi), (xlo, xhi) = pct(r["missed_ci95"]), np.array(r["hints_ci95"], float)
    ax.errorbar([x], [y], xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt="none",
                ecolor=kw.get("color", "black"), elinewidth=0.6, capsize=0, zorder=kw.get("zorder", 4) - 0.5)
    return ax.scatter([x], [y], **kw)


def main():
    o = json.loads((C.OUT / "e1_tradeoff.json").read_text())
    fig, ax = plt.subplots(figsize=(FS.WIDTH, 2.75), layout="constrained")

    # Score sweeps with 95% student-bootstrap bands.
    sweeps = (("bkt", "#2B2B2B", "-", "BKT"), ("pfa", FS.OI["orange"], "--", "PFA"), ("risk_head", FS.OI["sky"], ":", "Logistic risk"))
    for m, col, ls, lab in sweeps:
        c = o["curves"][m]
        x = np.array(c["hints_per_student"], float)
        y, ci = pct(c["missed_help_rate"]), pct(c["missed_ci95"])
        k = np.argsort(x)
        ax.fill_between(x[k], ci[k, 0], ci[k, 1], color=col if m != "bkt" else FS.SETTING["quiet"], alpha=0.18 if m == "bkt" else 0.12, lw=0, zorder=1)
        ax.plot(x[k], y[k], color=col, ls=ls, lw=1.5 if m == "bkt" else 1.2, label=lab, zorder=3 if m == "bkt" else 2)

    # Baselines as points.
    rule = [o["n_rule"][str(n)] for n in range(1, 6)]
    for n, r in zip(range(1, 6), rule):
        h = point(ax, r, marker="D", s=18, color=FS.OI["vermillion"], zorder=5, label="k-in-a-row rule" if n == 2 else None)
        if n == 1:
            ax.annotate("Always-\nhelp", (r["hints_per_student"], 100 * r["missed_help_rate"]), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", linespacing=0.9, color="#2B2B2B")
        else:
            ax.annotate(f"k={n}", (r["hints_per_student"], 100 * r["missed_help_rate"]), xytext=(-6, 0) if n == 2 else (5, -1),
                        textcoords="offset points", va="center", ha="right" if n == 2 else "left", color="#2B2B2B")
    for i, r in enumerate(o["random_matched"].values()):
        point(ax, r, marker="x", s=18, color=FS.OI["gray"], linewidths=0.9, zorder=4, label="Random (matched)" if i == 0 else None)

    # The three frozen settings: distinct marker and colour each.
    settings = []
    for prof in ("quiet", "balanced", "supportive"):
        mk = o["frozen_tau_markers"][prof]
        settings.append(point(ax, mk, s=40, marker=FS.SETTING_MARKER[prof], color=FS.SETTING[prof], edgecolors="white", linewidths=0.7,
                              zorder=7, label=f"{prof.capitalize()} ($\\tau$={mk['tau']:.2f})"))

    ax.set_xlabel("First hints per student")
    ax.set_ylabel("Missed help (% of stuck)")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-2, 102)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.grid(True, axis="y")
    handles = [h for h in ax.get_legend_handles_labels()[0] if h not in settings] + settings
    leg = ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2,
                    handletextpad=0.5)
    for t in leg.get_texts()[-3:]:
        t.set_color(FS.SETTING[t.get_text().split()[0].lower()])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "fig2_tradeoff.pdf", metadata=FS.PDF_META)
    fig.savefig(OUTDIR / "fig2_tradeoff.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
