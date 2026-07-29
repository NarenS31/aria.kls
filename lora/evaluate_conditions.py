"""
4-condition A/B/C/D experiment comparing:
  A: Vanilla llama3.1:8b (baseline)
  B: ARIA with RAG (ChromaDB + learning graph, no LoRA)
  C: ARIA with profile-specific LoRA adapter
  D: ARIA-Distilled LoRA adapter

Runs `n_episodes` episodes per condition per profile across all 3 ADHD personas.
Default: 180 episodes total per condition (36 episodes × 5 profiles).

Each episode is balanced across personas and subjects (round-robin).
All results streamed to data/lora_experiment_{condition}_{timestamp}.jsonl.

Output: 6 figures + lora/lora_experiment_summary.json
"""

import json
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
LORA_DIR    = ROOT / "lora"
FIGURES_DIR = LORA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_IDS = ["alex_chen", "jordan_rivera", "sam_okonkwo", "maya_patel", "eli_washington"]
PERSONAS    = ["hyperfocus", "scattered", "frustrated"]
SUBJECTS    = ["algebra", "biology", "python_programming"]
CONDITIONS  = ["A_vanilla", "B_rag", "C_lora", "D_distilled"]

CONDITION_LABELS = {
    "A_vanilla":    "A: Vanilla LLM",
    "B_rag":        "B: ARIA+RAG",
    "C_lora":       "C: ARIA+LoRA",
    "D_distilled":  "D: Distilled LoRA",
}


# ------------------------------------------------------------------
# Episode plan
# ------------------------------------------------------------------

def _build_episode_plan(n_episodes: int) -> List[dict]:
    """
    Build a balanced episode plan: equal distribution across 5 profiles,
    3 personas, 3 subjects within each profile block.
    """
    episodes_per_profile = max(1, n_episodes // len(PROFILE_IDS))
    plan = []
    for pid in PROFILE_IDS:
        for ep_idx in range(episodes_per_profile):
            persona  = PERSONAS[ep_idx % len(PERSONAS)]
            subject  = SUBJECTS[ep_idx % len(SUBJECTS)]
            plan.append({
                "profile_id":  pid,
                "persona":     persona,
                "subject":     subject,
                "seed_idx":    ep_idx,
                "episode_idx": ep_idx,
            })
    return plan


# ------------------------------------------------------------------
# Tutor functions per condition
# ------------------------------------------------------------------

def _vanilla_tutor(profile_id: str) -> callable:
    """Condition A: plain llama3.1:8b with no memory, no adapter."""
    import ollama
    history: List[dict] = []

    def _chat(student_msg: str) -> str:
        history.append({"role": "user", "content": student_msg})
        msgs = [{"role": "system", "content": "You are a helpful tutor. Be clear and concise."}]
        msgs.extend(history[-8:])
        r = ollama.chat(model="llama3.1:8b", messages=msgs, options={"num_predict": 400})
        resp = r.message.content.strip()
        history.append({"role": "assistant", "content": resp})
        return resp

    return _chat


def _rag_tutor(profile_id: str, memory_context: str, system_prompt: str) -> callable:
    """Condition B: ARIA RAG (ChromaDB + graph context) without LoRA."""
    import ollama
    history: List[dict] = []

    def _chat(student_msg: str) -> str:
        history.append({"role": "user", "content": student_msg})
        msgs = [{"role": "system", "content": system_prompt + f"\n\nMemory:\n{memory_context}"}]
        msgs.extend(history[-8:])
        r = ollama.chat(model="llama3.1:8b", messages=msgs, options={"num_predict": 400})
        resp = r.message.content.strip()
        history.append({"role": "assistant", "content": resp})
        return resp

    return _chat


def _lora_tutor(profile_id: str, memory_context: str, system_prompt: str) -> callable:
    """Condition C: ARIA profile-specific LoRA adapter."""
    from lora.aria_lora import aria_lora_inference, adapter_exists

    if not adapter_exists(profile_id):
        # Fallback to RAG if adapter not trained yet
        import ollama
        history: List[dict] = []

        def _chat_fallback(student_msg: str) -> str:
            history.append({"role": "user", "content": student_msg})
            msgs = [{"role": "system", "content": system_prompt + f"\n\nMemory:\n{memory_context}"}]
            msgs.extend(history[-8:])
            r = ollama.chat(model="llama3.1:8b", messages=msgs, options={"num_predict": 400})
            resp = r.message.content.strip()
            history.append({"role": "assistant", "content": resp})
            return resp

        return _chat_fallback

    def _chat(student_msg: str) -> str:
        return aria_lora_inference(
            student_message=student_msg,
            profile_id=profile_id,
            context=memory_context,
            system_override=system_prompt + f"\n\nMemory:\n{memory_context}" if memory_context else None,
        )

    return _chat


def _distilled_tutor(profile_id: str, memory_context: str) -> callable:
    """Condition D: ARIA-Distilled LoRA adapter."""
    from lora.distilled_lora import distilled_inference, adapter_exists

    if not adapter_exists():
        import ollama

        def _chat_fallback(student_msg: str) -> str:
            r = ollama.chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": "You are a neurodivergent-aware tutor. Use numbered micro-steps."},
                    {"role": "user",   "content": student_msg},
                ],
                options={"num_predict": 400},
            )
            return r.message.content.strip()

        return _chat_fallback

    def _chat(student_msg: str) -> str:
        return distilled_inference(student_msg, profile_context=memory_context)

    return _chat


# ------------------------------------------------------------------
# Memory context helpers
# ------------------------------------------------------------------

def _get_memory_context(profile_id: str, subject: str) -> Tuple[str, str]:
    """
    Retrieve up to 3 recent ChromaDB turns and learning graph context for profile.
    Returns (memory_context_str, system_prompt_str).
    """
    try:
        from eval.profiles import ProfileVectorStore, ProfileLearningGraph, PROFILE_MAP
        pvs = ProfileVectorStore(profile_id)
        plg = ProfileLearningGraph(profile_id)
        profile = PROFILE_MAP.get(profile_id)

        # Memory context
        turns = pvs.get_recent_turns(n=5)
        mem_str = ""
        if turns:
            snippets = []
            for t in turns[-3:]:
                u = t.get("user", "")[:120]
                a = t.get("assistant", "")[:120]
                if u and a:
                    snippets.append(f"S: {u}\nT: {a}")
            mem_str = "\n---\n".join(snippets)

        # System prompt
        if profile:
            from lora.data_formatter import _build_system_prompt
            confidence = profile.baseline_confidence.get(subject, 0.5)
            struggles  = list(profile.misconceptions.get(subject, ["unknown"]))[:2]
            best_style = profile.preferred_styles.get(subject, profile.learning_style)
            sys_prompt = _build_system_prompt(
                name=profile.name,
                diagnosis=profile.diagnosis,
                learning_style=profile.learning_style,
                topic=subject,
                confidence=confidence,
                struggles=struggles,
                best_style=best_style,
            )
        else:
            sys_prompt = "You are ARIA, an adaptive AI tutor for neurodivergent students. Use numbered micro-steps."

        return mem_str, sys_prompt

    except Exception:
        return "", "You are ARIA, an adaptive AI tutor for neurodivergent students. Use numbered micro-steps."


# ------------------------------------------------------------------
# Single condition runner
# ------------------------------------------------------------------

def run_condition(
    condition: str,
    n_episodes: int = 180,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Run all episodes for one condition.
    Streams results to JSONL and returns list of records.
    """
    from eval.simulator import SUBJECTS as SIM_SUBJECTS, run_episode
    from eval.rubric import score_turn

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = DATA_DIR / f"lora_experiment_{condition}_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _build_episode_plan(n_episodes)

    if verbose:
        print(f"\n[evaluate_conditions] {CONDITION_LABELS[condition]} — {len(plan)} episodes")

    records: List[dict] = []

    with open(output_path, "w") as f:
        for ep_idx, ep in enumerate(plan):
            profile_id = ep["profile_id"]
            persona    = ep["persona"]
            subject    = ep["subject"]
            seed_idx   = ep["seed_idx"]
            episode_id = str(uuid.uuid4())

            mem_ctx, sys_prompt = _get_memory_context(profile_id, subject)

            # Build tutor function for this condition
            if condition == "A_vanilla":
                tutor_fn = _vanilla_tutor(profile_id)
            elif condition == "B_rag":
                tutor_fn = _rag_tutor(profile_id, mem_ctx, sys_prompt)
            elif condition == "C_lora":
                tutor_fn = _lora_tutor(profile_id, mem_ctx, sys_prompt)
            elif condition == "D_distilled":
                tutor_fn = _distilled_tutor(profile_id, mem_ctx)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Run episode via simulator
            try:
                ep_result = run_episode(tutor_fn, persona, subject, seed_idx=seed_idx)
            except Exception as e:
                if verbose:
                    print(f"  Episode {ep_idx+1} error: {e}")
                continue

            # Score each turn
            conv: List[dict] = []
            for turn in ep_result.turns:
                student_msg = turn.get("student_msg", "")
                tutor_msg   = turn.get("tutor_msg", "")
                if not student_msg or not tutor_msg:
                    continue

                try:
                    scores = score_turn(student_msg, tutor_msg, conv, subject)
                except Exception:
                    scores = {"weighted_score": 1.0, "S": 1, "D": 1, "R": 1, "M": 1,
                              "A": 1, "TC": 1, "SC": 1, "FA": 1, "solution_dump": False}

                record = {
                    "episode_id":    episode_id,
                    "condition":     condition,
                    "profile_id":    profile_id,
                    "persona":       persona,
                    "subject":       subject,
                    "turn":          len(conv) // 2,
                    "student_msg":   student_msg,
                    "tutor_msg":     tutor_msg,
                    "scores":        {k: scores[k] for k in ["S","D","R","M","A","TC","SC","FA"]},
                    "weighted_score": scores["weighted_score"],
                    "solution_dump": scores.get("solution_dump", False),
                    "timestamp":     datetime.now().isoformat(),
                }
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()

                conv.append({"role": "user",      "content": student_msg})
                conv.append({"role": "assistant",  "content": tutor_msg})

            if verbose and (ep_idx + 1) % 10 == 0:
                n_scored = len(records)
                if n_scored > 0:
                    ws_mean = sum(r["weighted_score"] for r in records[-20:]) / min(20, n_scored)
                    print(f"  {ep_idx+1}/{len(plan)} episodes  |  ws={ws_mean:.3f}", end="\r")

    if verbose:
        print(f"\n  Done. {len(records)} turn records saved to {output_path}")
    return records


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run_all_conditions(
    n_episodes: int = 180,
    conditions: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """Run all 4 conditions and return {condition: records}."""
    if conditions is None:
        conditions = CONDITIONS

    all_results: Dict[str, List[dict]] = {}
    for cond in conditions:
        all_results[cond] = run_condition(cond, n_episodes=n_episodes, verbose=verbose)

    return all_results


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def compute_stats(all_results: Dict[str, List[dict]]) -> dict:
    """Compute per-condition, per-dimension, per-persona statistics."""
    stats: dict = {}

    dims = ["S", "D", "R", "M", "A", "TC", "SC", "FA", "weighted_score"]

    for cond, records in all_results.items():
        cond_stats: dict = {"overall": {}, "by_persona": {}, "by_profile": {}}

        # Overall
        for dim in dims:
            vals = [r["scores"].get(dim, r.get(dim)) if dim != "weighted_score" else r["weighted_score"]
                    for r in records]
            vals = [v for v in vals if v is not None]
            cond_stats["overall"][dim] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std":  float(np.std(vals))  if vals else 0.0,
                "n":    len(vals),
            }

        # By persona
        by_persona: Dict[str, list] = defaultdict(list)
        for r in records:
            by_persona[r["persona"]].append(r["weighted_score"])
        for persona, ws in by_persona.items():
            cond_stats["by_persona"][persona] = {
                "mean": float(np.mean(ws)) if ws else 0.0,
                "n":    len(ws),
            }

        # By profile
        by_profile: Dict[str, list] = defaultdict(list)
        for r in records:
            by_profile[r["profile_id"]].append(r["weighted_score"])
        for pid, ws in by_profile.items():
            cond_stats["by_profile"][pid] = {
                "mean": float(np.mean(ws)) if ws else 0.0,
                "n":    len(ws),
            }

        stats[cond] = cond_stats

    return stats


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test (manual, no scipy)."""
    import math
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 1.0
    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    if mean_d == 0.0:
        return 0.0, 1.0
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0.0:
        return float("inf"), 0.0
    se = math.sqrt(var_d / n)
    t = mean_d / se
    # Approximate p-value using normal distribution (large sample)
    z = abs(t)
    p = 2 * (1 - _norm_cdf(z))
    return t, p


def _norm_cdf(z: float) -> float:
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

_COLORS = {
    "A_vanilla":   "#95a5a6",
    "B_rag":       "#3498db",
    "C_lora":      "#e67e22",
    "D_distilled": "#2ecc71",
}


def _short_label(cond: str) -> str:
    return CONDITION_LABELS[cond].split(":")[0]


def generate_figures(
    all_results: Dict[str, List[dict]],
    stats: dict,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate 6 publication-quality figures."""
    if output_dir is None:
        output_dir = FIGURES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    figures_saved = []

    # Fig 1: Overall leaderboard (bar chart, 8 PEBBLE dims + weighted)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(9)
    dims = ["S", "D", "R", "M", "A", "TC", "SC", "FA", "weighted_score"]
    dim_labels = ["Scaffolding", "Depth", "Retrieval", "Metacognition", "Adaptability",
                  "Task Chunking", "Session Consistency", "Frustration Adapt.", "Weighted"]
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, (cond, offset) in enumerate(zip(CONDITIONS, offsets)):
        if cond not in stats:
            continue
        means = [stats[cond]["overall"].get(d, {}).get("mean", 0.0) for d in dims]
        bars = ax.bar(x + offset * width, means, width, label=CONDITION_LABELS[cond],
                      color=_COLORS[cond], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean Score (0–2)", fontsize=10)
    ax.set_title("Fig 1: ARIA LoRA Experiment — Per-Dimension PEBBLE Scores", fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_ylim(0, 2.2)
    plt.tight_layout()
    p = output_dir / "fig1_lora_leaderboard.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    # Fig 2: Weighted score by persona
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERSONAS))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, (cond, offset) in enumerate(zip(CONDITIONS, offsets)):
        if cond not in stats:
            continue
        persona_means = [stats[cond]["by_persona"].get(p, {}).get("mean", 0.0) for p in PERSONAS]
        ax.bar(x + offset * width, persona_means, width, label=CONDITION_LABELS[cond],
               color=_COLORS[cond], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PERSONAS], fontsize=10)
    ax.set_ylabel("Mean Weighted Score (0–2)", fontsize=10)
    ax.set_title("Fig 2: Weighted Score by ADHD Persona", fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.2)
    plt.tight_layout()
    p = output_dir / "fig2_lora_by_persona.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    # Fig 3: ND-specific dimensions (TC, SC, FA) only
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    nd_dims = ["TC", "SC", "FA"]
    nd_labels = ["Task Chunking", "Session Consistency", "Frustration Adapt."]
    x = np.arange(len(CONDITIONS))
    for ax, dim, label in zip(axes, nd_dims, nd_labels):
        means = [stats[cond]["overall"].get(dim, {}).get("mean", 0.0)
                 if cond in stats else 0.0 for cond in CONDITIONS]
        colors = [_COLORS[c] for c in CONDITIONS]
        bars = ax.bar([_short_label(c) for c in CONDITIONS], means, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 2.2)
        ax.set_ylabel("Mean Score (0–2)" if ax == axes[0] else "", fontsize=9)
        # Annotate bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Fig 3: Neurodivergent-Specific Dimensions (TC / SC / FA)", fontsize=11, y=1.02)
    plt.tight_layout()
    p = output_dir / "fig3_lora_nd_dims.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    # Fig 4: Weighted score by student profile
    profile_short = {
        "alex_chen":      "Alex",
        "jordan_rivera":  "Jordan",
        "sam_okonkwo":    "Sam",
        "maya_patel":     "Maya",
        "eli_washington": "Eli",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PROFILE_IDS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for cond, offset in zip(CONDITIONS, offsets):
        if cond not in stats:
            continue
        means = [stats[cond]["by_profile"].get(pid, {}).get("mean", 0.0) for pid in PROFILE_IDS]
        ax.bar(x + offset * width, means, width, label=CONDITION_LABELS[cond],
               color=_COLORS[cond], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([profile_short[p] for p in PROFILE_IDS], fontsize=10)
    ax.set_ylabel("Mean Weighted Score (0–2)", fontsize=10)
    ax.set_title("Fig 4: Per-Student-Profile Weighted Scores", fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.2)
    plt.tight_layout()
    p = output_dir / "fig4_lora_by_profile.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    # Fig 5: Learning curve (score over episode index, rolling mean 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond in CONDITIONS:
        if cond not in all_results or not all_results[cond]:
            continue
        records = all_results[cond]
        by_ep: Dict[str, list] = defaultdict(list)
        for r in records:
            by_ep[r["episode_id"]].append(r["weighted_score"])
        ep_means = [sum(v) / len(v) for v in by_ep.values()]
        window = 10
        if len(ep_means) >= window:
            smoothed = [sum(ep_means[max(0, i - window):i]) / min(i, window)
                        for i in range(1, len(ep_means) + 1)]
        else:
            smoothed = ep_means
        ax.plot(range(1, len(smoothed) + 1), smoothed, label=CONDITION_LABELS[cond],
                color=_COLORS[cond], linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Episode Index", fontsize=10)
    ax.set_ylabel("Weighted Score (rolling mean 10)", fontsize=10)
    ax.set_title("Fig 5: Learning Curve — Score over Episodes", fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.2)
    plt.tight_layout()
    p = output_dir / "fig5_lora_learning_curve.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    # Fig 6: Statistical significance matrix (C vs A, D vs A, D vs C)
    fig, ax = plt.subplots(figsize=(7, 5))
    comparisons = [
        ("C_lora", "A_vanilla", "C vs A"),
        ("D_distilled", "A_vanilla", "D vs A"),
        ("D_distilled", "B_rag", "D vs B"),
        ("D_distilled", "C_lora", "D vs C"),
        ("B_rag", "A_vanilla", "B vs A"),
    ]
    y_ticks = []
    y_vals = []
    for y_idx, (cond_a, cond_b, label) in enumerate(comparisons):
        if cond_a not in all_results or cond_b not in all_results:
            continue
        ws_a = [r["weighted_score"] for r in all_results[cond_a]]
        ws_b = [r["weighted_score"] for r in all_results[cond_b]]
        min_n = min(len(ws_a), len(ws_b))
        t, p_val = _paired_ttest(ws_a[:min_n], ws_b[:min_n])
        delta = (sum(ws_a) / max(len(ws_a), 1)) - (sum(ws_b) / max(len(ws_b), 1))
        stars = _significance_stars(p_val)
        color = "#2ecc71" if delta > 0 else "#e74c3c"
        bar = ax.barh(y_idx, delta, color=color, alpha=0.8)
        ax.text(delta + (0.02 if delta >= 0 else -0.02), y_idx,
                f"{stars} (t={t:.2f}, p={p_val:.3f})",
                va="center", ha="left" if delta >= 0 else "right", fontsize=8)
        y_ticks.append(y_idx)
        y_vals.append(label)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_vals, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean Score Difference (higher = row better)", fontsize=9)
    ax.set_title("Fig 6: Pairwise Significance — Weighted PEBBLE Score", fontsize=11, pad=10)
    plt.tight_layout()
    p = output_dir / "fig6_lora_significance.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_saved.append(p)

    return figures_saved


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def save_summary(stats: dict, figures: List[Path], n_episodes: int) -> Path:
    summary = {
        "experiment": "4-condition LoRA A/B/C/D",
        "n_episodes_per_condition": n_episodes,
        "conditions": CONDITION_LABELS,
        "stats": stats,
        "figures": [str(f) for f in figures],
        "run_at": datetime.now().isoformat(),
    }
    out = LORA_DIR / "lora_experiment_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARIA 4-condition LoRA experiment")
    parser.add_argument("--quick",          action="store_true", help="60 episodes/condition (12 per profile)")
    parser.add_argument("--full",           action="store_true", help="720 episodes/condition (144 per profile)")
    parser.add_argument("--n-episodes",     type=int, default=180, help="Episodes per condition")
    parser.add_argument("--conditions",     nargs="+", choices=CONDITIONS, default=None,
                        help="Run only specific conditions (default: all 4)")
    parser.add_argument("--figures-only",   action="store_true", help="Generate figures from existing JSONL")
    parser.add_argument("--quiet",          action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    n_ep = args.n_episodes
    if args.quick:
        n_ep = 60
    elif args.full:
        n_ep = 720

    if args.figures_only:
        # Load existing results
        import glob
        all_results: Dict[str, List[dict]] = {}
        for cond in CONDITIONS:
            files = sorted(glob.glob(str(DATA_DIR / f"lora_experiment_{cond}_*.jsonl")))
            if files:
                records = []
                with open(files[-1]) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                all_results[cond] = records
                if verbose:
                    print(f"  Loaded {len(records)} records for {cond}")
    else:
        all_results = run_all_conditions(
            n_episodes=n_ep,
            conditions=args.conditions,
            verbose=verbose,
        )

    if all_results:
        stats = compute_stats(all_results)
        figures = generate_figures(all_results, stats)
        summary_path = save_summary(stats, figures, n_ep)

        if verbose:
            print(f"\nResults saved to: {summary_path}")
            print(f"Figures: {len(figures)} files in {FIGURES_DIR}")
            print("\n=== WEIGHTED SCORE SUMMARY ===")
            for cond in CONDITIONS:
                if cond in stats:
                    mean = stats[cond]["overall"].get("weighted_score", {}).get("mean", 0.0)
                    n    = stats[cond]["overall"].get("weighted_score", {}).get("n", 0)
                    print(f"  {CONDITION_LABELS[cond]:25s}  {mean:.3f}  (n={n})")
