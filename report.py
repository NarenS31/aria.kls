"""
Report generator for ARIA evaluation.

Reads eval_results_aria_*.jsonl and eval_results_baseline_*.jsonl from data/.
Produces:
  - Leaderboard table (stdout + data/figures/leaderboard.png)
  - Per-dimension bar chart (PEBBLE Figure 1 style)
  - Learning curve: SC/FA score vs episodes (ARIA improves, baseline flatlines)
  - Frustration adaptation accuracy over time
  - All figures saved to data/figures/ at 300 DPI
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rubric import WEIGHTS, DIMENSIONS
from eval.scoring import summarize, load_results, bootstrap_ci, learning_curve_summary

DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ARIA_COLOR = "#4C72B0"
BASE_COLOR = "#DD8452"
HIGHLIGHT_COLOR = "#C44E52"


# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------

def _find_latest(model: str) -> Optional[Path]:
    files = sorted(DATA_DIR.glob(f"eval_results_{model}_*.jsonl"), reverse=True)
    return files[0] if files else None


def _load_both(
    aria_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
) -> tuple:
    aria_path = aria_path or _find_latest("aria")
    baseline_path = baseline_path or _find_latest("baseline")

    if not aria_path or not aria_path.exists():
        raise FileNotFoundError("No aria eval results found. Run: python eval/runner.py --model aria")
    if not baseline_path or not baseline_path.exists():
        raise FileNotFoundError("No baseline eval results found. Run: python eval/runner.py --model baseline")

    return load_results(aria_path), load_results(baseline_path), aria_path, baseline_path


# ------------------------------------------------------------------
# Leaderboard table (stdout)
# ------------------------------------------------------------------

def print_leaderboard(summary: dict) -> None:
    dims = list(WEIGHTS.keys()) + ["weighted"]
    dim_names = {d: DIMENSIONS[d]["name"] if d in DIMENSIONS else "Weighted Score" for d in dims}

    header = f"{'Dimension':<22} {'ARIA mean':>10} {'95% CI':>14} {'Base mean':>10} {'95% CI':>14} {'Cohen d':>8} {'p':>8} {'sig':>4}"
    print("\n" + "=" * len(header))
    print("  ARIA vs Baseline — Evaluation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for dim in dims:
        r = summary["per_dimension"][dim]
        a = r["aria"]
        b = r["baseline"]
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        print(
            f"{dim_names[dim]:<22} "
            f"{a['mean']:>10.3f} "
            f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]{'':<1} "
            f"{b['mean']:>10.3f} "
            f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}]{'':<1} "
            f"{r['cohens_d']:>8.3f} "
            f"{r['p_value']:>8.4f} "
            f"{sig:>4}"
        )

    print("-" * len(header))
    print(f"  Solution dump rate — ARIA: {summary['solution_dump_rate']['aria']:.1%}  Baseline: {summary['solution_dump_rate']['baseline']:.1%}")
    print(f"  N records — ARIA: {summary['n_records']['aria']}  Baseline: {summary['n_records']['baseline']}")
    print("  * p<0.05  ** p<0.01  *** p<0.001")
    print("=" * len(header) + "\n")


# ------------------------------------------------------------------
# Figure 1: Leaderboard bar chart (PEBBLE style)
# ------------------------------------------------------------------

def plot_leaderboard(summary: dict, path: Path) -> None:
    dims = list(WEIGHTS.keys())
    dim_labels = [f"{d}\n({DIMENSIONS[d]['name'][:12]})" for d in dims]

    aria_means = [summary["per_dimension"][d]["aria"]["mean"] for d in dims]
    base_means = [summary["per_dimension"][d]["baseline"]["mean"] for d in dims]
    aria_err_lo = [summary["per_dimension"][d]["aria"]["mean"] - summary["per_dimension"][d]["aria"]["ci_lo"] for d in dims]
    aria_err_hi = [summary["per_dimension"][d]["aria"]["ci_hi"] - summary["per_dimension"][d]["aria"]["mean"] for d in dims]
    base_err_lo = [summary["per_dimension"][d]["baseline"]["mean"] - summary["per_dimension"][d]["baseline"]["ci_lo"] for d in dims]
    base_err_hi = [summary["per_dimension"][d]["baseline"]["ci_hi"] - summary["per_dimension"][d]["baseline"]["mean"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_aria = ax.bar(
        x - width / 2, aria_means, width,
        label="ARIA", color=ARIA_COLOR, alpha=0.88,
        yerr=[aria_err_lo, aria_err_hi], capsize=3, error_kw={"linewidth": 1.2},
    )
    bars_base = ax.bar(
        x + width / 2, base_means, width,
        label="Baseline (vanilla LLM)", color=BASE_COLOR, alpha=0.88,
        yerr=[base_err_lo, base_err_hi], capsize=3, error_kw={"linewidth": 1.2},
    )

    # Highlight ADHD-specific dimensions
    for i, dim in enumerate(dims):
        if dim in ("TC", "SC", "FA"):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=HIGHLIGHT_COLOR, zorder=0)

    ax.set_xlabel("Rubric Dimension")
    ax.set_ylabel("Mean Score (0–2)")
    ax.set_title("ARIA vs Baseline: Per-Dimension Scores with 95% Bootstrap CIs\n"
                 "(shaded = ADHD-specific dimensions)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 2.3)
    ax.legend(loc="upper left")

    # Significance annotations
    for i, dim in enumerate(dims):
        p = summary["per_dimension"][dim]["p_value"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        if sig:
            ymax = max(aria_means[i] + aria_err_hi[i], base_means[i] + base_err_hi[i])
            ax.text(i, ymax + 0.08, sig, ha="center", va="bottom", fontsize=11, color="#333")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [report] Saved: {path}")


# ------------------------------------------------------------------
# Figure 2: Learning curve (SC and FA across episode order)
# ------------------------------------------------------------------

def plot_learning_curve(records_aria: List[dict], records_baseline: List[dict], path: Path) -> None:
    """SC and FA dimension scores across time. ARIA should improve, baseline flatlines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, dim in zip(axes, ["SC", "FA"]):
        # Group into windows of 10 episodes
        def window_scores(records: List[dict], window: int = 10) -> tuple:
            by_ep = defaultdict(list)
            ep_order = []
            seen = {}
            for r in records:
                eid = r.get("episode_id", "")
                if eid not in seen:
                    seen[eid] = len(seen)
                    ep_order.append(eid)
                by_ep[eid].append(float(r["scores"].get(dim, 1)))

            ep_means = [np.mean(by_ep[eid]) for eid in ep_order]
            if not ep_means:
                return [], []

            xs, ys = [], []
            for i in range(0, len(ep_means), window):
                chunk = ep_means[i: i + window]
                xs.append(i + window // 2)
                ys.append(np.mean(chunk))
            return xs, ys

        xs_a, ys_a = window_scores(records_aria)
        xs_b, ys_b = window_scores(records_baseline)

        if xs_a:
            ax.plot(xs_a, ys_a, "o-", color=ARIA_COLOR, label="ARIA", linewidth=2, markersize=5)
        if xs_b:
            ax.plot(xs_b, ys_b, "s--", color=BASE_COLOR, label="Baseline", linewidth=2, markersize=5)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel(f"Mean {dim} Score (0–2)")
        ax.set_title(f"{DIMENSIONS[dim]['name']} over Time\n({dim})")
        ax.set_ylim(0, 2.2)
        ax.legend()

    fig.suptitle("Learning Curve: ARIA Adapts Over Time, Baseline Stays Flat", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [report] Saved: {path}")


# ------------------------------------------------------------------
# Figure 3: FA accuracy over time (frustration adaptation)
# ------------------------------------------------------------------

def plot_frustration_adaptation(records_aria: List[dict], records_baseline: List[dict], path: Path) -> None:
    """FA score trend over episodes for frustrated persona only."""
    fig, ax = plt.subplots(figsize=(9, 4))

    def fa_trend(records: List[dict], window: int = 5) -> tuple:
        frustrated = [r for r in records if r.get("persona") == "frustrated"]
        ep_ids, seen = [], {}
        by_ep = defaultdict(list)
        for r in frustrated:
            eid = r.get("episode_id", "")
            if eid not in seen:
                seen[eid] = len(seen)
                ep_ids.append(eid)
            by_ep[eid].append(float(r["scores"].get("FA", 1)))

        ep_means = [np.mean(by_ep[eid]) for eid in ep_ids]
        xs, ys = [], []
        for i in range(0, len(ep_means), window):
            chunk = ep_means[i: i + window]
            xs.append(i + window // 2)
            ys.append(np.mean(chunk))
        return xs, ys

    xs_a, ys_a = fa_trend(records_aria)
    xs_b, ys_b = fa_trend(records_baseline)

    if xs_a:
        ax.plot(xs_a, ys_a, "o-", color=ARIA_COLOR, label="ARIA", linewidth=2, markersize=5)
    if xs_b:
        ax.plot(xs_b, ys_b, "s--", color=BASE_COLOR, label="Baseline", linewidth=2, markersize=5)

    ax.set_xlabel("Episode Number (frustrated persona only)")
    ax.set_ylabel("Mean FA Score (0–2)")
    ax.set_title("Frustration Adaptation Accuracy Over Time\n(higher = detects & appropriately shifts tone)")
    ax.set_ylim(0, 2.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [report] Saved: {path}")


# ------------------------------------------------------------------
# Full report entry point
# ------------------------------------------------------------------

def generate_report(
    aria_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
) -> None:
    print("[report] Loading results...")
    records_aria, records_baseline, aria_path, baseline_path = _load_both(aria_path, baseline_path)
    print(f"  ARIA: {len(records_aria)} turns from {aria_path.name}")
    print(f"  Baseline: {len(records_baseline)} turns from {baseline_path.name}")

    print("[report] Computing statistics (10 000 bootstrap iterations)...")
    summary = summarize(records_aria=records_aria, records_baseline=records_baseline, n_bootstrap=10_000)

    print_leaderboard(summary)

    print("[report] Generating figures...")
    plot_leaderboard(summary, FIGURES_DIR / "leaderboard.png")
    plot_learning_curve(records_aria, records_baseline, FIGURES_DIR / "learning_curve.png")
    plot_frustration_adaptation(records_aria, records_baseline, FIGURES_DIR / "frustration_adaptation.png")

    print(f"\n[report] Figures saved to {FIGURES_DIR}/")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ARIA evaluation report")
    parser.add_argument("--aria", type=str, default=None, help="Path to aria eval JSONL")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline eval JSONL")
    args = parser.parse_args()

    generate_report(
        aria_path=Path(args.aria) if args.aria else None,
        baseline_path=Path(args.baseline) if args.baseline else None,
    )
