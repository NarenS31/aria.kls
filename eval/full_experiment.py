"""
Full end-to-end experiment pipeline for ARIA.

10-step pipeline:
  1  Generate and seed all 5 profiles
  2  Run all 6 base models against all 5 profiles x 3 personas x 3 subjects
  3  Score all episodes on 8-dimension rubric
  4  Extract and save all reasoning traces
  5  Run failure analysis
  6  Generate distilled model (ARIA-Distilled)
  7  Run distilled model for same episode set
  8  Score distilled model
  9  Statistical analyses
  10 Generate all figures and paper draft

CLI:
  python eval/full_experiment.py --quick       (20 episodes per model, smoke test)
  python eval/full_experiment.py --full        (270 episodes per model)
  python eval/full_experiment.py --distill-only
  python eval/full_experiment.py --figures-only
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
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api_clients import query_model, is_available, ALL_EVAL_MODELS, LOCAL_ONLY_MODELS, session_cost_summary
from profiles import ALL_PROFILES, PROFILE_MAP, ProfileVectorStore, ProfileLearningGraph, seed_all_profiles
from rubric import WEIGHTS, DIMENSIONS
from simulator import PERSONAS, SUBJECTS, run_episode
from reasoning_extractor import run_episode_with_reasoning, load_traces, TRACES_DIR
from failure_analyzer import run_failure_analysis, plot_failure_heatmap
from distillation import run_distillation_pipeline, run_distilled_inference
from scoring import bootstrap_ci, paired_ttest, cohens_d, summarize

DATA_DIR = ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PERSONA_NAMES = list(PERSONAS.keys())
SUBJECT_NAMES = list(SUBJECTS.keys())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

MODEL_COLORS = {
    "llama3.1:8b":       "#4C72B0",
    "mistral:7b":        "#DD8452",
    "gemma2:9b":         "#55A868",
    "gpt-4o":            "#C44E52",
    "gemini-2.5-flash":  "#8172B2",
    "aria_distilled":    "#CCB974",
}
MODEL_LABELS = {
    "llama3.1:8b":      "LLaMA 3.1 8B",
    "mistral:7b":       "Mistral 7B",
    "gemma2:9b":        "Gemma2 9B",
    "gpt-4o":           "GPT-4o",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "aria_distilled":   "ARIA-Distilled",
}


# ------------------------------------------------------------------
# Episode plan
# ------------------------------------------------------------------

def _episode_plan(n_per_model: int) -> List[dict]:
    """Balanced: n_per_model / (5 profiles × 3 personas × 3 subjects)."""
    per_cell = max(1, n_per_model // (len(ALL_PROFILES) * len(PERSONA_NAMES) * len(SUBJECT_NAMES)))
    plan = []
    for profile in ALL_PROFILES:
        for persona in PERSONA_NAMES:
            for subject in SUBJECT_NAMES:
                for seed_idx in range(per_cell):
                    plan.append({
                        "profile": profile,
                        "persona": persona,
                        "subject": subject,
                        "seed_idx": seed_idx % 2,
                    })
                    if len(plan) >= n_per_model:
                        return plan
    # Fill remainder
    idx = 0
    while len(plan) < n_per_model:
        profile = ALL_PROFILES[idx % len(ALL_PROFILES)]
        plan.append({
            "profile": profile,
            "persona": PERSONA_NAMES[idx % len(PERSONA_NAMES)],
            "subject": SUBJECT_NAMES[idx % len(SUBJECT_NAMES)],
            "seed_idx": 0,
        })
        idx += 1
    return plan


# ------------------------------------------------------------------
# Step 1: Profile seeding
# ------------------------------------------------------------------

def step1_seed_profiles(n_sessions: int = 10, verbose: bool = True) -> None:
    print("\n[Step 1] Seeding all 5 student profiles...")
    seed_all_profiles(n_history_sessions=n_sessions, verbose=verbose)


# ------------------------------------------------------------------
# Step 2-4: Run all models with reasoning extraction
# ------------------------------------------------------------------

def step2_run_models(
    n_episodes: int,
    models: List[str],
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    print(f"\n[Step 2-4] Running {n_episodes} episodes per model across {len(models)} models")

    plan = _episode_plan(n_episodes)
    all_results: Dict[str, List[dict]] = {m: [] for m in models}

    for ep_idx, spec in enumerate(plan):
        profile = spec["profile"]
        persona = spec["persona"]
        subject = spec["subject"]
        seed_idx = spec["seed_idx"]

        # Get memory context from profile's vector store
        pvs = ProfileVectorStore(profile.profile_id)
        ctx_results = pvs.retrieve_context(
            f"{subject} {persona}", n=3
        )
        memory_context = "\n---\n".join(r["text"][:200] for r in ctx_results)

        aria_profile = profile.as_aria_profile()
        aria_profile["profile_id"] = profile.profile_id

        if verbose:
            print(f"\n  ep {ep_idx+1:3d}/{len(plan)}  {profile.profile_id[:10]:10s}  {persona:10s}  {subject}")

        for model in models:
            if not is_available(model):
                if verbose and ep_idx == 0:
                    print(f"    [{model}] SKIP (not available)")
                continue

            output_file = TRACES_DIR / f"{model.replace(':', '_').replace('/', '_')}_{profile.profile_id}.jsonl"

            try:
                if verbose:
                    print(f"    [{model:<20}]", end=" ", flush=True)

                records = run_episode_with_reasoning(
                    model=model,
                    persona_name=persona,
                    subject_name=subject,
                    profile=aria_profile,
                    memory_context=memory_context,
                    seed_idx=seed_idx,
                    output_file=output_file,
                )
                all_results[model].extend(records)

                if verbose:
                    ws = sum(r["weighted_score"] for r in records) / max(len(records), 1)
                    print(f"ws={ws:.3f} turns={len(records)}")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")

    return all_results


# ------------------------------------------------------------------
# Step 5: Failure analysis
# ------------------------------------------------------------------

def step5_failure_analysis(all_records: Dict[str, List[dict]], verbose: bool = True) -> dict:
    print("\n[Step 5] Running failure analysis...")
    flat_records = []
    for model, recs in all_records.items():
        for r in recs:
            r["model"] = model
        flat_records.extend(recs)
    return run_failure_analysis(records=flat_records, models=list(all_records.keys()), verbose=verbose)


# ------------------------------------------------------------------
# Step 6: Distillation
# ------------------------------------------------------------------

def step6_distillation(failure_patterns: dict, verbose: bool = True) -> bool:
    print("\n[Step 6] Running distillation pipeline...")
    return run_distillation_pipeline(failure_patterns=failure_patterns, verbose=verbose)


# ------------------------------------------------------------------
# Step 7-8: Run distilled model
# ------------------------------------------------------------------

def step7_run_distilled(n_episodes: int, verbose: bool = True) -> List[dict]:
    print("\n[Step 7-8] Running ARIA-Distilled...")
    if not is_available("aria_distilled"):
        print("  aria_distilled not available — skipping")
        return []

    plan = _episode_plan(n_episodes)
    records = []
    output_file = TRACES_DIR / "aria_distilled_all.jsonl"

    for ep_idx, spec in enumerate(plan):
        profile = spec["profile"]
        aria_profile = profile.as_aria_profile()
        aria_profile["profile_id"] = profile.profile_id

        pvs = ProfileVectorStore(profile.profile_id)
        ctx = pvs.retrieve_context(spec["subject"], n=3)
        memory_context = "\n---\n".join(r["text"][:200] for r in ctx)

        if verbose:
            print(f"  ep {ep_idx+1}/{len(plan)}", end=" ", flush=True)

        try:
            ep_records = run_episode_with_reasoning(
                model="aria_distilled",
                persona_name=spec["persona"],
                subject_name=spec["subject"],
                profile=aria_profile,
                memory_context=memory_context,
                seed_idx=spec["seed_idx"],
                output_file=output_file,
            )
            records.extend(ep_records)
            if verbose:
                ws = sum(r["weighted_score"] for r in ep_records) / max(len(ep_records), 1)
                print(f"ws={ws:.3f}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return records


# ------------------------------------------------------------------
# Step 9: Statistical analyses
# ------------------------------------------------------------------

def step9_statistics(
    all_results: Dict[str, List[dict]],
    baseline_model: str = "llama3.1:8b",
) -> dict:
    print("\n[Step 9] Computing statistics...")
    stats: dict = {"per_model": {}, "pairwise": {}}

    for model, records in all_results.items():
        if not records:
            continue
        # Per-dimension mean + CI
        dim_stats: dict = {}
        for dim in list(WEIGHTS.keys()) + ["weighted"]:
            key = "weighted_score" if dim == "weighted" else None
            vals = (
                [r["weighted_score"] for r in records]
                if dim == "weighted"
                else [float(r.get("rubric_scores", r.get("scores", {})).get(dim, 1)) for r in records]
            )
            mean, lo, hi = bootstrap_ci(vals, n_iter=2000)
            dim_stats[dim] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
        stats["per_model"][model] = dim_stats

    # Pairwise vs baseline
    base_recs = all_results.get(baseline_model, [])
    for model, records in all_results.items():
        if model == baseline_model or not records or not base_recs:
            continue
        pair_stats: dict = {}
        n = min(len(records), len(base_recs))
        for dim in WEIGHTS:
            model_vals = [float(r.get("rubric_scores", r.get("scores", {})).get(dim, 1)) for r in records[:n]]
            base_vals  = [float(r.get("rubric_scores", r.get("scores", {})).get(dim, 1)) for r in base_recs[:n]]
            t, p = paired_ttest(model_vals, base_vals)
            d = cohens_d(model_vals, base_vals)
            pair_stats[dim] = {"t": t, "p": p, "cohens_d": d, "significant": p < 0.05}
        stats["pairwise"][f"{model}_vs_{baseline_model}"] = pair_stats

    out_path = DATA_DIR / "experiment_results_full.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    return stats


# ------------------------------------------------------------------
# Step 10: Figures
# ------------------------------------------------------------------

def _model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def fig1_leaderboard(stats: dict, models: List[str], path: Path) -> None:
    dims = list(WEIGHTS.keys())
    n_models = len([m for m in models if m in stats.get("per_model", {})])
    if n_models == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(dims))
    width = 0.8 / max(n_models, 1)

    for i, model in enumerate([m for m in models if m in stats.get("per_model", {})]):
        dim_stats = stats["per_model"][model]
        means = [dim_stats.get(d, {}).get("mean", 0) for d in dims]
        err_lo = [dim_stats.get(d, {}).get("mean", 0) - dim_stats.get(d, {}).get("ci_lo", 0) for d in dims]
        err_hi = [dim_stats.get(d, {}).get("ci_hi", 0) - dim_stats.get(d, {}).get("mean", 0) for d in dims]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width * 0.9,
            label=_model_label(model),
            color=MODEL_COLORS.get(model, "#999"),
            alpha=0.85,
            yerr=[err_lo, err_hi], capsize=2,
        )

    for i, dim in enumerate(dims):
        if dim in ("TC", "SC", "FA"):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#C44E52", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIMENSIONS[d]['name'][:8]}" for d in dims])
    ax.set_ylim(0, 2.4)
    ax.set_ylabel("Mean Score (0–2)")
    ax.set_title("Fig 1: All Models × All Dimensions with 95% Bootstrap CIs\n(shaded = ADHD-specific: TC, SC, FA)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_learning_curve(
    lc_data: Dict[int, Dict[str, dict]],
    path: Path,
    n_values: List[int] = None,
) -> None:
    if not lc_data:
        return
    n_values = n_values or sorted(lc_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, dim in zip(axes, ["SC", "FA"]):
        aria_means = [lc_data.get(n, {}).get(dim, {}).get("mean", 0) for n in n_values]
        aria_lo    = [lc_data.get(n, {}).get(dim, {}).get("ci_lo", 0) for n in n_values]
        aria_hi    = [lc_data.get(n, {}).get(dim, {}).get("ci_hi", 0) for n in n_values]

        ax.plot(n_values, aria_means, "o-", color=MODEL_COLORS["aria_distilled"],
                linewidth=2.2, markersize=6, label="ARIA-Distilled")
        ax.fill_between(n_values, aria_lo, aria_hi, alpha=0.2, color=MODEL_COLORS["aria_distilled"])

        # Baseline flat line from lc_data
        base_mean = lc_data.get(f"baseline_{dim}", {}).get("mean")
        if base_mean is not None:
            ax.axhline(base_mean, color=MODEL_COLORS["llama3.1:8b"], linestyle="--",
                       linewidth=1.8, label="LLaMA 3.1 8B (baseline)")

        ax.set_xlabel("N Pre-seeded History Sessions")
        ax.set_ylabel(f"Mean {dim} Score (0–2)")
        ax.set_title(f"{DIMENSIONS[dim]['name']}: Score vs Prior History")
        ax.set_xticks(n_values)
        ax.set_ylim(0, 2.2)
        ax.legend()

    fig.suptitle("Fig 2: ARIA-Distilled Learning Curve — SC and FA Improve with N, Baseline Stays Flat")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_profile_breakdown(all_results: Dict[str, List[dict]], models: List[str], path: Path) -> None:
    profiles = [p.profile_id for p in ALL_PROFILES]
    avail_models = [m for m in models if m in all_results and all_results[m]]
    if not avail_models:
        return

    n_m = len(avail_models)
    fig, axes = plt.subplots(1, n_m, figsize=(3 * n_m, 4))
    if n_m == 1:
        axes = [axes]

    for ax, model in zip(axes, avail_models):
        matrix = np.zeros((len(profiles), 1))
        for i, pid in enumerate(profiles):
            vals = [r["weighted_score"] for r in all_results[model] if r.get("profile_id") == pid]
            matrix[i, 0] = np.mean(vals) if vals else 0.0

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
        ax.set_yticks(range(len(profiles)))
        ax.set_yticklabels([p.replace("_", "\n") for p in profiles], fontsize=7)
        ax.set_xticks([])
        ax.set_title(_model_label(model), fontsize=9)
        for i in range(len(profiles)):
            ax.text(0, i, f"{matrix[i,0]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=axes[-1], label="Mean Weighted Score")
    fig.suptitle("Fig 4: Per-Profile Weighted Score by Model", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_persona_breakdown(all_results: Dict[str, List[dict]], models: List[str], path: Path) -> None:
    avail_models = [m for m in models if m in all_results and all_results[m]]
    if not avail_models:
        return

    matrix = np.zeros((len(PERSONA_NAMES), len(avail_models)))
    for j, model in enumerate(avail_models):
        for i, persona in enumerate(PERSONA_NAMES):
            vals = [r["weighted_score"] for r in all_results[model] if r.get("persona") == persona]
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(avail_models)))
    ax.set_xticklabels([_model_label(m) for m in avail_models], fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(PERSONA_NAMES)))
    ax.set_yticklabels(PERSONA_NAMES)
    for i in range(len(PERSONA_NAMES)):
        for j in range(len(avail_models)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean Weighted Score")
    ax.set_title("Fig 5: Per-Persona × Per-Model Weighted Score")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_reasoning_correlation(all_results: Dict[str, List[dict]], path: Path) -> None:
    """Scatter: reasoning confidence vs actual weighted score."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    all_records = [r for recs in all_results.values() for r in recs]

    conf_vals = [r.get("reasoning", {}).get("confidence", None) for r in all_records]
    ws_vals   = [r.get("weighted_score", None) for r in all_records]
    pairs = [(c, w) for c, w in zip(conf_vals, ws_vals) if c is not None and w is not None]

    if pairs:
        xs, ys = zip(*pairs)
        axes[0].scatter(xs, ys, alpha=0.3, s=10, color="#4C72B0")
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(0, 1, 50)
        axes[0].plot(x_line, m * x_line + b, "r-", linewidth=1.5, label=f"y={m:.2f}x+{b:.2f}")
        r = np.corrcoef(xs, ys)[0, 1]
        axes[0].set_title(f"Fig 6a: Reasoning Confidence vs Weighted Score\n(Pearson r = {r:.3f})")
        axes[0].set_xlabel("Model's Self-Reported Confidence")
        axes[0].set_ylabel("Actual Weighted Rubric Score")
        axes[0].legend()

    # Reasoning extraction rate per model
    models_avail = [m for m in MODEL_LABELS if m in all_results and all_results[m]]
    extract_rates = []
    for model in models_avail:
        recs = all_results[model]
        rate = sum(1 for r in recs if r.get("reasoning_extracted", False)) / max(len(recs), 1)
        extract_rates.append(rate)

    if models_avail:
        colors = [MODEL_COLORS.get(m, "#999") for m in models_avail]
        axes[1].barh(range(len(models_avail)), extract_rates, color=colors, alpha=0.85)
        axes[1].set_yticks(range(len(models_avail)))
        axes[1].set_yticklabels([_model_label(m) for m in models_avail], fontsize=9)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Reasoning Extraction Success Rate")
        axes[1].set_title("Fig 6b: Reasoning JSON Parse Success Rate by Model")
        for i, rate in enumerate(extract_rates):
            axes[1].text(rate + 0.01, i, f"{rate:.0%}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig7_effect_sizes(stats: dict, models: List[str], baseline: str, path: Path) -> None:
    dims = list(WEIGHTS.keys())
    target_models = [m for m in models if m != baseline and f"{m}_vs_{baseline}" in stats.get("pairwise", {})]
    if not target_models:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(dims))
    width = 0.8 / max(len(target_models), 1)

    for i, model in enumerate(target_models):
        key = f"{model}_vs_{baseline}"
        pair = stats["pairwise"].get(key, {})
        ds = [pair.get(d, {}).get("cohens_d", 0.0) for d in dims]
        sigs = [pair.get(d, {}).get("significant", False) for d in dims]
        offset = (i - len(target_models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, ds, width * 0.9,
                      label=_model_label(model),
                      color=MODEL_COLORS.get(model, "#999"), alpha=0.85)
        for j, (bar, sig) in enumerate(zip(bars, sigs)):
            if sig:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        "*", ha="center", va="bottom", fontsize=11)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.8, label="Small effect (d=0.2)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Medium effect (d=0.5)")
    ax.axhline(0.8, color="gray", linestyle="-", linewidth=0.8, label="Large effect (d=0.8)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIMENSIONS[d]['name'][:8]}" for d in dims])
    ax.set_ylabel("Cohen's d (vs LLaMA 3.1 8B baseline)")
    ax.set_title(f"Fig 7: Effect Sizes vs {_model_label(baseline)} (positive = better than baseline)\n(* p<0.05)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_figures(
    all_results: Dict[str, List[dict]],
    stats: dict,
    failure_patterns: dict,
    lc_data: Optional[dict] = None,
) -> None:
    print("\n[Step 10] Generating figures...")
    models = [m for m in MODEL_LABELS if m in all_results]

    flat_all = [r for recs in all_results.values() for r in recs]

    fig1_leaderboard(stats, models, FIGURES_DIR / "fig1_leaderboard.png")
    if lc_data:
        fig2_learning_curve(lc_data, FIGURES_DIR / "fig2_learning_curve.png")
    if failure_patterns:
        plot_failure_heatmap(failure_patterns, [m for m in models if m in all_results and all_results[m]],
                             FIGURES_DIR / "fig3_failure_heatmap.png")
    fig4_profile_breakdown(all_results, models, FIGURES_DIR / "fig4_profile_breakdown.png")
    fig5_persona_breakdown(all_results, models, FIGURES_DIR / "fig5_persona_breakdown.png")
    fig6_reasoning_correlation(all_results, FIGURES_DIR / "fig6_reasoning_correlation.png")
    fig7_effect_sizes(stats, models, "llama3.1:8b", FIGURES_DIR / "fig7_effect_sizes.png")
    print(f"[Step 10] All figures saved to {FIGURES_DIR}/")


# ------------------------------------------------------------------
# Learning curve experiment
# ------------------------------------------------------------------

def run_learning_curve(episodes_per_n: int = 60, n_values: Optional[List[int]] = None, verbose: bool = True) -> dict:
    from synthetic_history import seed_history_n
    if n_values is None:
        n_values = [0, 5, 10, 20]

    lc_data: dict = {}
    plan = _episode_plan(episodes_per_n)

    # Baseline (N=0, no personalization, llama3.1:8b)
    base_records = []
    for spec in plan[:min(episodes_per_n, len(plan))]:
        profile = spec["profile"]
        try:
            recs = run_episode_with_reasoning(
                model="llama3.1:8b",
                persona_name=spec["persona"],
                subject_name=spec["subject"],
                profile=profile.as_aria_profile(),
                memory_context="",
                seed_idx=spec["seed_idx"],
            )
            base_records.extend(recs)
        except Exception:
            pass

    for dim in ["SC", "FA"]:
        vals = [float(r.get("rubric_scores", {}).get(dim, 1)) for r in base_records]
        mean, lo, hi = bootstrap_ci(vals, n_iter=1000)
        lc_data[f"baseline_{dim}"] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    for n in n_values:
        if verbose:
            print(f"  Learning curve: N={n}")
        n_records = []
        for spec in plan[:min(episodes_per_n, len(plan))]:
            profile = spec["profile"]
            pvs = ProfileVectorStore(profile.profile_id)
            plg = ProfileLearningGraph(profile.profile_id)

            # Temporarily seed extra history
            from synthetic_history import seed_history_n as shn
            from profiles import _make_vs_adaptor, _make_lg_adaptor
            if n > 0:
                shn(_make_vs_adaptor(pvs), _make_lg_adaptor(plg), profile.as_aria_profile(), n=n, verbose=False)

            ctx = pvs.retrieve_context(spec["subject"], n=3)
            memory_context = "\n---\n".join(r["text"][:200] for r in ctx)

            aria_profile = profile.as_aria_profile()
            aria_profile["profile_id"] = profile.profile_id
            try:
                recs = run_episode_with_reasoning(
                    model="aria_distilled" if is_available("aria_distilled") else "llama3.1:8b",
                    persona_name=spec["persona"],
                    subject_name=spec["subject"],
                    profile=aria_profile,
                    memory_context=memory_context,
                    seed_idx=spec["seed_idx"],
                )
                n_records.extend(recs)
            except Exception:
                pass

        by_dim: dict = {}
        for dim in ["SC", "FA"]:
            vals = [float(r.get("rubric_scores", {}).get(dim, 1)) for r in n_records]
            mean, lo, hi = bootstrap_ci(vals, n_iter=1000)
            by_dim[dim] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
        lc_data[n] = by_dim

    with open(DATA_DIR / "learning_curve_full.json", "w") as f:
        json.dump(lc_data, f, indent=2, default=str)
    return lc_data


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_full_pipeline(n_episodes: int = 270, n_history: int = 10, verbose: bool = True, local_only: bool = False) -> None:
    model_pool = LOCAL_ONLY_MODELS if local_only else ALL_EVAL_MODELS
    avail = [m for m in model_pool if is_available(m)]
    print(f"[full_experiment] Available models: {avail}")

    if not avail:
        print("No models available. Ensure Ollama is running and at least llama3.1:8b is pulled.")
        return

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    step1_seed_profiles(n_sessions=n_history, verbose=verbose)

    all_results = step2_run_models(n_episodes, avail, verbose=verbose)

    failure_patterns = step5_failure_analysis(all_results, verbose=verbose)

    distill_ok = step6_distillation(failure_patterns, verbose=verbose)

    distilled_records: List[dict] = []
    if distill_ok:
        distilled_records = step7_run_distilled(n_episodes, verbose=verbose)
        if distilled_records:
            all_results["aria_distilled"] = distilled_records

    stats = step9_statistics(all_results)

    lc_data = {}
    if "aria_distilled" in all_results:
        lc_data = run_learning_curve(episodes_per_n=max(20, n_episodes // 10), verbose=verbose)
        fig2_learning_curve(lc_data, FIGURES_DIR / "fig2_learning_curve.png")

    generate_all_figures(all_results, stats, failure_patterns, lc_data)

    # Generate paper
    from papers.aria_paper import generate_paper
    generate_paper(stats=stats, failure_patterns=failure_patterns, all_results=all_results)

    # Print cost summary
    costs = session_cost_summary()
    if costs:
        print(f"\n[API costs] {costs}")
        print(f"  Total: ${sum(costs.values()):.4f}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def run_lora_experiment(n_episodes: int = 180, verbose: bool = True) -> None:
    """
    Step 11: 4-condition LoRA A/B/C/D experiment.
    Requires: lora adapters trained (python main.py --train-lora / --train-distilled)
    """
    from lora.evaluate_conditions import run_all_conditions, compute_stats, generate_figures, save_summary
    print(f"\n[Step 11] 4-Condition LoRA Experiment ({n_episodes} episodes per condition)...")
    all_results = run_all_conditions(n_episodes=n_episodes, verbose=verbose)
    stats = compute_stats(all_results)
    figures = generate_figures(all_results, stats)
    summary_path = save_summary(stats, figures, n_episodes)
    print(f"  LoRA experiment complete. Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA full experiment pipeline")
    parser.add_argument("--quick", action="store_true", help="20 episodes per model (smoke test)")
    parser.add_argument("--full", action="store_true", help="270 episodes per model")
    parser.add_argument("--distill-only", action="store_true", help="Only run distillation step")
    parser.add_argument("--figures-only", action="store_true", help="Regenerate figures from saved results")
    parser.add_argument("--learning-curve", action="store_true", help="Run learning curve experiment only")
    parser.add_argument("--stats-only", action="store_true", help="Skip steps 1-8; run step9+step10 on existing JSONL files in data/reasoning_traces/")
    parser.add_argument("--lora-experiment", action="store_true", help="Run 4-condition LoRA A/B/C/D experiment")
    parser.add_argument("--n-history", type=int, default=10, help="History sessions per profile")
    parser.add_argument("--local-only", action="store_true", help="Skip API models; only run llama3.1:8b, mistral:7b, gemma2:9b")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.stats_only:
        traces_dir = ROOT / "data" / "reasoning_traces"
        all_results: Dict[str, List[dict]] = defaultdict(list)
        for jsonl_path in sorted(traces_dir.glob("*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    model = rec.get("model", jsonl_path.stem)
                    all_results[model].append(rec)
        if not all_results:
            print("No JSONL records found in data/reasoning_traces/. Run the full pipeline first.")
            return
        print(f"[--stats-only] Loaded {sum(len(v) for v in all_results.values())} records across {len(all_results)} models.")
        stats = step9_statistics(dict(all_results))
        fp_path = DATA_DIR / "failure_patterns.json"
        failure_patterns = json.loads(fp_path.read_text()) if fp_path.exists() else {}
        lc_path = DATA_DIR / "learning_curve_full.json"
        lc_data = json.loads(lc_path.read_text()) if lc_path.exists() else {}
        generate_all_figures(dict(all_results), stats, failure_patterns, lc_data)

    elif args.distill_only:
        failure_patterns = {}
        fp_path = DATA_DIR / "failure_patterns.json"
        if fp_path.exists():
            with open(fp_path) as f:
                failure_patterns = json.load(f)
        run_distillation_pipeline(failure_patterns=failure_patterns, verbose=verbose)

    elif args.figures_only:
        stats_path = DATA_DIR / "experiment_results_full.json"
        if not stats_path.exists():
            print("No experiment_results_full.json found. Run --full first.")
            return
        with open(stats_path) as f:
            stats = json.load(f)
        all_results = {m: load_traces(m) for m in ALL_EVAL_MODELS}
        fp_path = DATA_DIR / "failure_patterns.json"
        failure_patterns = json.loads(fp_path.read_text()) if fp_path.exists() else {}
        lc_path = DATA_DIR / "learning_curve_full.json"
        lc_data = json.loads(lc_path.read_text()) if lc_path.exists() else {}
        generate_all_figures(all_results, stats, failure_patterns, lc_data)

    elif args.learning_curve:
        run_learning_curve(episodes_per_n=30, verbose=verbose)

    elif args.lora_experiment:
        n_ep = 60 if args.quick else (720 if args.full else 180)
        run_lora_experiment(n_episodes=n_ep, verbose=verbose)

    else:
        n_ep = 20 if args.quick else 270
        run_full_pipeline(n_episodes=n_ep, n_history=args.n_history, verbose=verbose, local_only=args.local_only)


if __name__ == "__main__":
    main()
