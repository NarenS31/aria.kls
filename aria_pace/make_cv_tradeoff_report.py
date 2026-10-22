"""Figures and tables for Amendment A7 from results/cv_tradeoff_val.json (no recomputation). One figure per profile."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import RESULTS, ROOT

PROFS = ("quiet", "balanced", "supportive")
DOCS = ROOT.parent / "docs" / "aria_pace"
COLORS = {"D3": "#1b6ca8", "D4": "#d95f02", "D5T15": "#1b9e77", "D5T20": "#7570b3", "DEMA": "#a6761d"}


def main():
    d = json.load(open(RESULTS / "cv_tradeoff_val.json"))
    out_dir = DOCS / "figures_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    for prof in PROFS:
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        for name, r in d["definitions"].items():
            x = np.array([np.nan if v is None else v for v in r["missed"]], float) * 100
            y = np.array(r["saving"], float) * 100
            lo, hi = np.array(r["saving_ci95"][0], float) * 100, np.array(r["saving_ci95"][1], float) * 100
            ok = ~np.isnan(x) & ~np.isnan(y)
            order = np.argsort(x[ok])
            ax.plot(x[ok][order], y[ok][order], color=COLORS[name], lw=1.6, label=name)
            ax.fill_between(x[ok][order], lo[ok][order], hi[ok][order], color=COLORS[name], alpha=0.15, lw=0)
            pt = r["chosen_tau_pooled_result"][prof]
            ax.errorbar([100 * pt["missed"]], [100 * pt["saving"]], xerr=[[100 * (pt["missed"] - pt["missed_ci95"][0])], [100 * (pt["missed_ci95"][1] - pt["missed"])]],
                        yerr=[[100 * (pt["saving"] - pt["saving_ci95"][0])], [100 * (pt["saving_ci95"][1] - pt["saving"])]], fmt="o", color=COLORS[name],
                        mec="black", ms=7, capsize=2, zorder=5)
        ax.set_xlabel("missed-help share, % (held-out folds, pooled)")
        ax.set_ylabel("intervention saving vs always-help, %")
        ax.set_title(f"{prof.capitalize()} profile: cross-validated trade-off per mastery definition\n"
                     "line = tau sweep; band = 95% student bootstrap of the saving; marker = per-fold chosen tau", fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.legend(title="definition", fontsize=8)
        fig.text(0.01, 0.005, "exploratory, validation only (A7); BKT only; tau and Platt chosen on a separate calibration part of the training folds", fontsize=6.5)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(out_dir / f"cv_tradeoff_{prof}.png", dpi=150)
        plt.close(fig)
    print("wrote", [f"cv_tradeoff_{p}.png" for p in PROFS])


if __name__ == "__main__":
    main()
