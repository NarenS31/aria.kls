"""
Full controlled experiment: ARIA personalization vs vanilla baseline.

Condition A: vanilla llama3.1:8b — no memory, no graph, no profile context
Condition B: llama3.1:8b + ARIA personalization — memory from ChromaDB, graph context,
             behavioral parameters from synthetic profile

Independent variable: personalization context (present vs absent)
Dependent variables: scores on all 8 rubric dimensions (S, D, R, M, A, TC, SC, FA)

Learning curve experiment: run Condition B with N = [0, 5, 10, 20] pre-seeded sessions
to show how scores improve as prior history grows.

CLI:
  python eval/experiment.py --n-sessions 10 --episodes 180
  python eval/experiment.py --learning-curve
  python eval/experiment.py --quick   (20 episodes for testing)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.rubric import WEIGHTS, DIMENSIONS
from eval.simulator import PERSONAS, SUBJECTS, run_episode
from eval.baseline import BaselineAgent
from eval.scoring import (
    bootstrap_ci, paired_ttest, cohens_d, summarize, learning_curve_summary
)
from eval.synthetic_profile import ALEX_CHEN, profile_as_aria_profile, generate_profile_json
from eval.synthetic_history import seed_history_n

DATA_DIR = ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

PERSONA_NAMES = list(PERSONAS.keys())
SUBJECT_NAMES = list(SUBJECTS.keys())
N_VALUES = [0, 5, 10, 20]

# Publication plot style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ARIA_COLOR = "#4C72B0"
BASE_COLOR = "#DD8452"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ------------------------------------------------------------------
# Condition B agent builder (ARIA with pre-seeded context)
# ------------------------------------------------------------------

def _build_aria_agent(n_sessions: int, fresh_stores: bool = True):
    """
    Build an ARIAAgent pre-warmed with N synthetic history sessions.
    fresh_stores=True creates isolated in-memory stores per run to avoid cross-contamination.
    """
    from eval.profiles import ProfileVectorStore, ProfileLearningGraph
    from agent.reasoning import ARIAAgent

    vs = ProfileVectorStore("eval_aria")
    lg = ProfileLearningGraph("eval_aria")

    if n_sessions > 0:
        seed_history_n(vs, lg, ALEX_CHEN, n=n_sessions, verbose=False)

    aria_profile = profile_as_aria_profile(ALEX_CHEN)
    return ARIAAgent(vector_store=vs, learning_graph=lg, user_profile=aria_profile)


# ------------------------------------------------------------------
# Episode plan: balanced across personas and subjects
# ------------------------------------------------------------------

def _episode_plan(n_episodes: int) -> List[dict]:
    per_persona = n_episodes // len(PERSONA_NAMES)
    plan = []
    for persona in PERSONA_NAMES:
        for i in range(per_persona):
            plan.append({
                "persona": persona,
                "subject": SUBJECT_NAMES[i % len(SUBJECT_NAMES)],
                "seed_idx": i % 2,
            })
    while len(plan) < n_episodes:
        i = len(plan)
        plan.append({
            "persona": PERSONA_NAMES[i % len(PERSONA_NAMES)],
            "subject": SUBJECT_NAMES[i % len(SUBJECT_NAMES)],
            "seed_idx": 0,
        })
    return plan


# ------------------------------------------------------------------
# Single-condition runner (returns records list)
# ------------------------------------------------------------------

def _run_condition(
    condition: str,
    n_episodes: int,
    n_sessions: int = 0,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> List[dict]:
    from eval.rubric import score_turn

    plan = _episode_plan(n_episodes)
    records = []

    if condition == "B":
        agent = _build_aria_agent(n_sessions)
        tutor_fn = agent.chat
    else:
        agent = BaselineAgent()
        tutor_fn = agent.chat

    label = f"Condition {'B (ARIA)' if condition == 'B' else 'A (Baseline)'}  n_sessions={n_sessions}"
    if verbose:
        print(f"\n[experiment] {label}  episodes={n_episodes}")

    if output_path:
        out_f = open(output_path, "a")
    else:
        out_f = None

    for ep_idx, spec in enumerate(plan):
        persona, subject, seed_idx = spec["persona"], spec["subject"], spec["seed_idx"]

        if verbose:
            print(f"  ep {ep_idx+1:3d}/{n_episodes}  {persona:10s}  {subject}", end="  ", flush=True)

        if condition == "A" and hasattr(agent, "reset"):
            agent.reset()

        try:
            ep = run_episode(tutor_fn, persona, subject, seed_idx)
            conv: List[dict] = []
            for turn in ep.turns:
                scores = score_turn(
                    turn["student_msg"], turn["tutor_msg"],
                    conv, subject,
                )
                record = {
                    "episode_id": ep.episode_id,
                    "condition": condition,
                    "n_sessions": n_sessions,
                    "model": "aria" if condition == "B" else "baseline",
                    "persona": persona,
                    "subject": subject,
                    "turn": turn["turn_idx"],
                    "student_msg": turn["student_msg"],
                    "tutor_msg": turn["tutor_msg"],
                    "scores": {k: scores[k] for k in WEIGHTS},
                    "weighted_score": scores["weighted_score"],
                    "solution_dump": scores.get("solution_dump", False),
                    "timestamp": datetime.now().isoformat(),
                    "episode_turn_count": len(ep.turns),
                }
                records.append(record)
                if out_f:
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                conv.append({"role": "user", "content": turn["student_msg"]})
                conv.append({"role": "assistant", "content": turn["tutor_msg"]})

            if verbose:
                ws = np.mean([r["weighted_score"] for r in records[-len(ep.turns):]])
                print(f"turns={len(ep.turns)}  ws={ws:.3f}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    if out_f:
        out_f.close()
    return records


# ------------------------------------------------------------------
# Figure generation
# ------------------------------------------------------------------

def _plot_leaderboard(summary: dict, path: Path) -> None:
    dims = list(WEIGHTS.keys())
    labels = [f"{d}\n{DIMENSIONS[d]['name'][:10]}" for d in dims]
    x = np.arange(len(dims))
    w = 0.35

    aria_m = [summary["per_dimension"][d]["aria"]["mean"] for d in dims]
    base_m = [summary["per_dimension"][d]["baseline"]["mean"] for d in dims]
    aria_lo = [summary["per_dimension"][d]["aria"]["mean"] - summary["per_dimension"][d]["aria"]["ci_lo"] for d in dims]
    aria_hi = [summary["per_dimension"][d]["aria"]["ci_hi"] - summary["per_dimension"][d]["aria"]["mean"] for d in dims]
    base_lo = [summary["per_dimension"][d]["baseline"]["mean"] - summary["per_dimension"][d]["baseline"]["ci_lo"] for d in dims]
    base_hi = [summary["per_dimension"][d]["baseline"]["ci_hi"] - summary["per_dimension"][d]["baseline"]["mean"] for d in dims]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, aria_m, w, label="ARIA (Condition B)", color=ARIA_COLOR,
           alpha=0.88, yerr=[aria_lo, aria_hi], capsize=3)
    ax.bar(x + w/2, base_m, w, label="Baseline (Condition A)", color=BASE_COLOR,
           alpha=0.88, yerr=[base_lo, base_hi], capsize=3)

    # Shade ADHD-specific dims
    for i, dim in enumerate(dims):
        if dim in ("TC", "SC", "FA"):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.07, color="#C44E52", zorder=0)

    for i, dim in enumerate(dims):
        p = summary["per_dimension"][dim]["p_value"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if sig:
            ymax = max(aria_m[i] + aria_hi[i], base_m[i] + base_hi[i])
            ax.text(i, ymax + 0.07, sig, ha="center", fontsize=11)

    adhd_patch = mpatches.Patch(color="#C44E52", alpha=0.2, label="ADHD-specific dimensions")
    ax.legend(handles=[
        mpatches.Patch(color=ARIA_COLOR, alpha=0.88, label="ARIA (Condition B)"),
        mpatches.Patch(color=BASE_COLOR, alpha=0.88, label="Baseline (Condition A)"),
        adhd_patch,
    ])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 2.35)
    ax.set_ylabel("Mean Score (0–2)")
    ax.set_title("ARIA vs Baseline: All 8 Rubric Dimensions with 95% Bootstrap CIs\n(* p<0.05  ** p<0.01  *** p<0.001  |  shaded = ADHD-specific)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_learning_curve(lc_data: Dict[int, Dict[str, dict]], path: Path) -> None:
    """
    lc_data: {n_sessions: {dim: {mean, ci_lo, ci_hi}}}
    Show SC and FA improving with N; baseline flatline added as horizontal band.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, dim in zip(axes, ["SC", "FA"]):
        ns = sorted(lc_data.keys())
        means = [lc_data[n][dim]["mean"] for n in ns]
        lo = [lc_data[n][dim]["ci_lo"] for n in ns]
        hi = [lc_data[n][dim]["ci_hi"] for n in ns]

        ax.plot(ns, means, "o-", color=ARIA_COLOR, linewidth=2.2, markersize=6, label="ARIA (Condition B)")
        ax.fill_between(ns, lo, hi, alpha=0.2, color=ARIA_COLOR)

        # Baseline flat line
        base_mean = lc_data.get("baseline_mean_" + dim, {}).get("mean", None)
        if base_mean is not None:
            ax.axhline(base_mean, color=BASE_COLOR, linestyle="--", linewidth=1.8, label="Baseline (Condition A)")
            ax.axhspan(
                lc_data.get("baseline_mean_" + dim, {}).get("ci_lo", base_mean - 0.05),
                lc_data.get("baseline_mean_" + dim, {}).get("ci_hi", base_mean + 0.05),
                alpha=0.1, color=BASE_COLOR,
            )

        ax.set_xlabel("N Pre-seeded History Sessions")
        ax.set_ylabel(f"Mean {dim} Score (0–2)")
        ax.set_title(f"{DIMENSIONS[dim]['name']}: Score vs Prior History\n"
                     f"(hypothesis: {dim} improves most as N increases)")
        ax.set_xticks(ns)
        ax.set_ylim(0, 2.2)
        ax.legend()

    fig.suptitle("Learning Curve: How Prior Session History Boosts ARIA's Adaptive Dimensions", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_persona_breakdown(records_a: List[dict], records_b: List[dict], path: Path) -> None:
    """3×3 heatmap: persona × subject, comparing ARIA vs baseline weighted score."""
    personas = PERSONA_NAMES
    subjects = SUBJECT_NAMES

    def mean_ws(records: List[dict], persona: str, subject: str) -> float:
        sub = [r["weighted_score"] for r in records
               if r.get("persona") == persona and r.get("subject") == subject]
        return float(np.mean(sub)) if sub else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, records, title in zip(axes, [records_b, records_a], ["ARIA (Condition B)", "Baseline (Condition A)"]):
        matrix = np.array([[mean_ws(records, p, s) for s in subjects] for p in personas])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([s.replace("_", "\n") for s in subjects], fontsize=8)
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas, fontsize=9)
        ax.set_title(title)
        for i in range(len(personas)):
            for j in range(len(subjects)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
        plt.colorbar(im, ax=ax, label="Mean Weighted Score")
    fig.suptitle("Per-Persona × Per-Subject Weighted Score Heatmap", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_correlation_matrix(records: List[dict], path: Path) -> None:
    dims = list(WEIGHTS.keys())
    data = np.array([[float(r["scores"].get(d, 1)) for d in dims] for r in records if r.get("scores")])
    if len(data) < 5:
        print("  [skip] Not enough data for correlation matrix")
        return
    corr = np.corrcoef(data.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims)
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims)
    for i in range(len(dims)):
        for j in range(len(dims)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Rubric Dimension Correlation Matrix (ARIA scores)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------
# Print full results table
# ------------------------------------------------------------------

def _print_results_table(summary: dict) -> None:
    dims = list(WEIGHTS.keys()) + ["weighted"]
    dim_labels = {d: DIMENSIONS[d]["name"] if d in DIMENSIONS else "Weighted" for d in dims}
    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS: ARIA (Condition B) vs Baseline (Condition A)")
    print("=" * 90)
    fmt = "{:<20} {:>8} {:>14} {:>8} {:>14} {:>8} {:>8} {:>4}"
    print(fmt.format("Dimension", "B mean", "95% CI", "A mean", "95% CI", "d", "p", "sig"))
    print("-" * 90)
    for dim in dims:
        r = summary["per_dimension"][dim]
        a = r["aria"]      # condition B (ARIA)
        b = r["baseline"]  # condition A (baseline)
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns"))
        print(fmt.format(
            dim_labels[dim][:20],
            f"{a['mean']:.3f}", f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]",
            f"{b['mean']:.3f}", f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}]",
            f"{r['cohens_d']:.3f}", f"{r['p_value']:.4f}", sig,
        ))
    print("=" * 90)
    print("  * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
    print(f"  N (ARIA): {summary['n_records']['aria']}  N (Baseline): {summary['n_records']['baseline']}\n")


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run_main_experiment(n_sessions: int = 10, n_episodes: int = 180, verbose: bool = True) -> dict:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    result_path_a = DATA_DIR / f"eval_results_baseline_{ts}.jsonl"
    result_path_b = DATA_DIR / f"eval_results_aria_{ts}.jsonl"

    generate_profile_json()

    print("\n[experiment] Running Condition A: Baseline (no personalization)")
    records_a = _run_condition("A", n_episodes, 0, result_path_a, verbose)

    print("\n[experiment] Running Condition B: ARIA (with personalization)")
    records_b = _run_condition("B", n_episodes, n_sessions, result_path_b, verbose)

    print("\n[experiment] Computing statistics...")
    summary = summarize(records_aria=records_b, records_baseline=records_a, n_bootstrap=10_000)
    _print_results_table(summary)

    print("[experiment] Generating figures...")
    _plot_leaderboard(summary, FIGURES_DIR / "leaderboard.png")
    _plot_persona_breakdown(records_a, records_b, FIGURES_DIR / "persona_breakdown.png")
    _plot_correlation_matrix(records_b, FIGURES_DIR / "correlation_matrix.png")

    full_results = {
        "summary": summary,
        "metadata": {
            "n_sessions": n_sessions,
            "n_episodes": n_episodes,
            "timestamp": ts,
            "aria_results_file": str(result_path_b),
            "baseline_results_file": str(result_path_a),
        },
    }
    results_json = DATA_DIR / "experiment_results.json"
    with open(results_json, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"[experiment] Full results saved to {results_json}")

    return full_results


# ------------------------------------------------------------------
# Learning curve experiment: N = [0, 5, 10, 20]
# ------------------------------------------------------------------

def run_learning_curve_experiment(episodes_per_n: int = 60, verbose: bool = True) -> None:
    print("\n[experiment] Learning curve: N = [0, 5, 10, 20] pre-seeded sessions")

    lc_data: Dict = {}

    # Baseline: run once
    print("\n  Baseline (Condition A, N=0):")
    records_baseline = _run_condition("A", episodes_per_n, 0, None, verbose)
    # Store baseline dim means for comparison
    for dim in ["SC", "FA"]:
        vals = [float(r["scores"][dim]) for r in records_baseline if dim in r.get("scores", {})]
        mean, lo, hi = bootstrap_ci(vals, n_iter=1000)
        lc_data[f"baseline_mean_{dim}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    for n in N_VALUES:
        print(f"\n  Condition B: N={n} pre-seeded sessions")
        records = _run_condition("B", episodes_per_n, n, None, verbose)

        by_dim: Dict[str, dict] = {}
        for dim in ["SC", "FA"]:
            vals = [float(r["scores"][dim]) for r in records if dim in r.get("scores", {})]
            mean, lo, hi = bootstrap_ci(vals, n_iter=1000)
            by_dim[dim] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
        lc_data[n] = by_dim

    print("\n[experiment] Learning curve results:")
    for dim in ["SC", "FA"]:
        print(f"\n  {DIMENSIONS[dim]['name']} ({dim}):")
        baseline_m = lc_data.get(f"baseline_mean_{dim}", {}).get("mean", 0)
        print(f"    Baseline A:  {baseline_m:.3f}")
        for n in N_VALUES:
            m = lc_data[n][dim]["mean"]
            lo = lc_data[n][dim]["ci_lo"]
            hi = lc_data[n][dim]["ci_hi"]
            print(f"    N={n:2d}:  {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    _plot_learning_curve(lc_data, FIGURES_DIR / "learning_curve.png")

    with open(DATA_DIR / "learning_curve_results.json", "w") as f:
        json.dump(lc_data, f, indent=2, default=str)
    print(f"\n[experiment] Learning curve saved to {DATA_DIR / 'learning_curve_results.json'}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA controlled experiment")
    parser.add_argument("--n-sessions", type=int, default=10, help="Pre-seeded history sessions for Condition B")
    parser.add_argument("--episodes", type=int, default=180, help="Episodes per condition")
    parser.add_argument("--learning-curve", action="store_true", help="Run N=[0,5,10,20] learning curve")
    parser.add_argument("--quick", action="store_true", help="20 episodes for smoke test")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 20
        args.n_sessions = 2

    if args.learning_curve:
        episodes_per_n = 20 if args.quick else 60
        run_learning_curve_experiment(episodes_per_n=episodes_per_n, verbose=not args.quiet)
    else:
        run_main_experiment(
            n_sessions=args.n_sessions,
            n_episodes=args.episodes,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
