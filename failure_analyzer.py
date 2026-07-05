"""
Failure pattern analyzer for ARIA evaluation.

A failure is defined as: any turn where TC < 1 OR SC < 1 OR FA < 1
(the three neurodivergent-critical dimensions).

Analysis pipeline:
1. Load all reasoning traces
2. Identify failed turns per ND dimension
3. Cluster by dimension x profile x persona x subject
4. Synthesize failure/success patterns via llama3.1:8b
5. Statistical tests: chi-square per dimension across models
6. Output failure_patterns.json + heatmap figure
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.api_clients import query_model
from eval.reasoning_extractor import load_traces, TRACES_DIR

DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ND_DIMENSIONS = ["TC", "SC", "FA"]   # neurodivergent-critical
FAILURE_THRESHOLD = 1.0              # score < 1.0 = failure


# ------------------------------------------------------------------
# Failure identification
# ------------------------------------------------------------------

def identify_failures(records: List[dict]) -> Dict[str, List[dict]]:
    """
    Returns {dimension: [failed_records]}.
    A record fails on dimension D if scores[D] < FAILURE_THRESHOLD.
    """
    by_dim: Dict[str, List[dict]] = {d: [] for d in ND_DIMENSIONS}
    for r in records:
        scores = r.get("rubric_scores", r.get("scores", {}))
        for dim in ND_DIMENSIONS:
            val = scores.get(dim)
            if val is not None and float(val) < FAILURE_THRESHOLD:
                by_dim[dim].append(r)
    return by_dim


def identify_successes(records: List[dict]) -> Dict[str, List[dict]]:
    by_dim: Dict[str, List[dict]] = {d: [] for d in ND_DIMENSIONS}
    for r in records:
        scores = r.get("rubric_scores", r.get("scores", {}))
        for dim in ND_DIMENSIONS:
            val = scores.get(dim)
            if val is not None and float(val) >= FAILURE_THRESHOLD:
                by_dim[dim].append(r)
    return by_dim


# ------------------------------------------------------------------
# Pattern synthesis via LLM
# ------------------------------------------------------------------

def _format_traces_for_synthesis(records: List[dict], n: int = 6) -> str:
    """Format up to N trace records as readable text for the LLM."""
    sampled = random.sample(records, min(n, len(records)))
    parts = []
    for i, r in enumerate(sampled):
        reasoning = r.get("reasoning", {})
        parts.append(
            f"Trace {i+1}:\n"
            f"  Student: {r.get('student_message', '')[:150]}\n"
            f"  Tutor planned: {reasoning.get('planned_approach', 'unknown')[:150]}\n"
            f"  ADHD considerations: {reasoning.get('adhd_considerations', 'none')[:120]}\n"
            f"  Tutor response: {r.get('tutor_response', '')[:200]}\n"
        )
    return "\n".join(parts)


def synthesize_pattern(records: List[dict], dimension: str, is_failure: bool) -> str:
    """Use llama3.1:8b to synthesize a failure or success pattern from traces."""
    if not records:
        return f"No {'failure' if is_failure else 'success'} traces available for {dimension}."

    pattern_type = "failure" if is_failure else "success"
    trace_text = _format_traces_for_synthesis(records, n=5)

    prompt = (
        f"Here are {min(5, len(records))} tutoring reasoning traces that all "
        f"{'FAILED' if is_failure else 'SUCCEEDED'} on the {dimension} dimension "
        f"({'Task Chunking' if dimension == 'TC' else 'Session Consistency' if dimension == 'SC' else 'Frustration Adaptation'}) "
        f"for students with ADHD.\n\n"
        f"{trace_text}\n\n"
        f"Identify the common reasoning {pattern_type} pattern in exactly 2-3 sentences. "
        f"Be specific about what the tutor {'did wrong' if is_failure else 'did right'}. "
        f"Start with 'Tutors that {'fail' if is_failure else 'succeed'} on {dimension}...'"
    )
    try:
        return query_model(
            "llama3.1:8b",
            [{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
    except Exception as e:
        return f"Pattern synthesis failed: {e}"


# ------------------------------------------------------------------
# Chi-square test (manual implementation)
# ------------------------------------------------------------------

def chi_square_failure_rates(
    records: List[dict],
    dimension: str,
    models: List[str],
) -> Tuple[float, float, dict]:
    """
    Chi-square test: is failure rate significantly different across models for this dimension?
    Returns (chi2_stat, p_value, failure_counts_by_model).
    """
    counts: Dict[str, Dict[str, int]] = {}
    for model in models:
        model_records = [r for r in records if r.get("model") == model]
        fail = sum(
            1 for r in model_records
            if float(r.get("rubric_scores", r.get("scores", {})).get(dimension, 2)) < FAILURE_THRESHOLD
        )
        total = len(model_records)
        counts[model] = {"fail": fail, "success": max(0, total - fail), "total": total}

    if not counts:
        return 0.0, 1.0, counts

    # Build observed matrix [models x {fail, success}]
    observed = np.array([[counts[m]["fail"], counts[m]["success"]] for m in models if counts[m]["total"] > 0])
    if observed.shape[0] < 2:
        return 0.0, 1.0, counts

    # Expected frequencies under H0 (independence)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    if grand_total == 0:
        return 0.0, 1.0, counts
    expected = row_totals * col_totals / grand_total

    # Chi-square statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0)))

    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = _chi2_pvalue(chi2, df)

    return round(chi2, 4), round(p_value, 6), counts


def _chi2_pvalue(chi2: float, df: int) -> float:
    """Approximate p-value for chi-square distribution using regularized gamma."""
    if chi2 <= 0 or df <= 0:
        return 1.0
    # Use the regularized upper incomplete gamma function approximation
    # For large values, use normal approximation
    try:
        return _regularized_upper_gamma(df / 2, chi2 / 2)
    except Exception:
        return 1.0


def _regularized_upper_gamma(a: float, x: float) -> float:
    """Q(a, x) = 1 - P(a, x): regularized upper incomplete gamma."""
    # Use series expansion for x < a + 1, continued fraction otherwise
    if x < a + 1:
        return 1.0 - _gamma_series(a, x)
    else:
        return _gamma_cf(a, x)


def _gamma_series(a: float, x: float, max_iter: int = 200) -> float:
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-7:
            break
    return total * math.exp(-x + a * math.log(x) - _log_gamma(a))


def _gamma_cf(a: float, x: float, max_iter: int = 200) -> float:
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-7:
            break
    return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _log_gamma(x: float) -> float:
    """Lanczos approximation of log(Γ(x))."""
    coeffs = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.001208650973866179, -0.000005395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in coeffs:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


# ------------------------------------------------------------------
# Fisher's exact test (for small cell counts n < 5)
# ------------------------------------------------------------------

def fishers_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher's exact test p-value for 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def _hypergeom_prob(a: int, b: int, c: int, d: int, n: int) -> float:
        try:
            log_p = (
                _log_comb(a + b, a) + _log_comb(c + d, c)
                - _log_comb(n, a + c)
            )
            return math.exp(log_p)
        except Exception:
            return 0.0

    def _log_comb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return sum(math.log(i) for i in range(k + 1, n + 1)) - sum(math.log(i) for i in range(1, n - k + 1))

    observed_p = _hypergeom_prob(a, b, c, d, n)
    p_val = 0.0
    r1, r2, c1 = a + b, c + d, a + c
    for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1):
        p = _hypergeom_prob(x, r1 - x, c1 - x, n - r1 - c1 + x, n)
        if p <= observed_p + 1e-10:
            p_val += p
    return min(1.0, p_val)


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------

def run_failure_analysis(
    records: Optional[List[dict]] = None,
    models: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Full failure analysis pipeline.
    Returns failure_patterns dict, saves to data/failure_patterns.json.
    """
    if records is None:
        records = load_traces()

    if not records:
        print("[failure_analyzer] No traces found. Run reasoning_extractor first.")
        return {}

    if models is None:
        models = list(set(r.get("model", "unknown") for r in records))

    if verbose:
        print(f"[failure_analyzer] Analyzing {len(records)} traces across {len(models)} models")

    failures_by_dim = identify_failures(records)
    successes_by_dim = identify_successes(records)

    output: dict = {}

    for dim in ND_DIMENSIONS:
        failed = failures_by_dim[dim]
        succeeded = successes_by_dim[dim]

        if verbose:
            print(f"\n  {dim}: {len(failed)} failures, {len(succeeded)} successes")

        # Synthesize patterns (sample up to 5 from diverse sources)
        fail_patterns = []
        success_patterns = []

        if failed:
            # Cluster by profile for richer patterns
            by_profile = defaultdict(list)
            for r in failed:
                by_profile[r.get("profile_id", "unknown")].append(r)
            for prof, prof_fails in list(by_profile.items())[:2]:
                pat = synthesize_pattern(prof_fails, dim, is_failure=True)
                fail_patterns.append({"pattern": pat, "profile": prof, "n": len(prof_fails)})

            # Overall pattern
            overall_fail = synthesize_pattern(failed, dim, is_failure=True)
            fail_patterns.insert(0, {"pattern": overall_fail, "profile": "all", "n": len(failed)})

        if succeeded:
            overall_success = synthesize_pattern(succeeded, dim, is_failure=False)
            success_patterns.append({"pattern": overall_success, "profile": "all", "n": len(succeeded)})

        # Most-failed profile and persona
        profile_fail_counts = defaultdict(int)
        persona_fail_counts = defaultdict(int)
        for r in failed:
            profile_fail_counts[r.get("profile_id", "?")] += 1
            persona_fail_counts[r.get("persona", "?")] += 1

        most_failed_profile = max(profile_fail_counts, key=profile_fail_counts.get) if profile_fail_counts else "none"
        most_failed_persona = max(persona_fail_counts, key=persona_fail_counts.get) if persona_fail_counts else "none"

        # Failure rate by model
        fail_rate_by_model: Dict[str, float] = {}
        for model in models:
            model_recs = [r for r in records if r.get("model") == model]
            if model_recs:
                model_fails = [r for r in model_recs
                               if float(r.get("rubric_scores", r.get("scores", {})).get(dim, 2)) < FAILURE_THRESHOLD]
                fail_rate_by_model[model] = round(len(model_fails) / len(model_recs), 4)
            else:
                fail_rate_by_model[model] = 0.0

        # Statistical test
        chi2, pval, counts = chi_square_failure_rates(records, dim, models)

        # Fisher's exact for any 2-model pair with small n
        fisher_results = {}
        model_list = [m for m in models if counts.get(m, {}).get("total", 0) > 0]
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                m1, m2 = model_list[i], model_list[j]
                c1, c2 = counts[m1], counts[m2]
                if min(c1["fail"], c1["success"], c2["fail"], c2["success"]) < 5:
                    fp = fishers_exact(c1["fail"], c1["success"], c2["fail"], c2["success"])
                    fisher_results[f"{m1}_vs_{m2}"] = round(fp, 6)

        output[dim] = {
            "failure_patterns": fail_patterns,
            "success_patterns": success_patterns,
            "most_failed_profile": most_failed_profile,
            "most_failed_persona": most_failed_persona,
            "failure_rate_by_model": fail_rate_by_model,
            "chi_square": {"stat": chi2, "p_value": pval, "significant": pval < 0.05},
            "fisher_exact": fisher_results,
            "n_failures": len(failed),
            "n_successes": len(succeeded),
        }

        if verbose:
            print(f"    Chi²={chi2:.2f}, p={pval:.4f} ({'*' if pval < 0.05 else 'ns'})")
            print(f"    Most-failed profile: {most_failed_profile}, persona: {most_failed_persona}")

    # Save
    out_path = DATA_DIR / "failure_patterns.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    if verbose:
        print(f"\n[failure_analyzer] Saved to {out_path}")

    return output


# ------------------------------------------------------------------
# Failure rate heatmap figure
# ------------------------------------------------------------------

def plot_failure_heatmap(
    failure_patterns: dict,
    models: List[str],
    path: Optional[Path] = None,
) -> None:
    if path is None:
        path = FIGURES_DIR / "fig3_failure_heatmap.png"

    dims = ND_DIMENSIONS
    matrix = np.zeros((len(models), len(dims)))

    for j, dim in enumerate(dims):
        rates = failure_patterns.get(dim, {}).get("failure_rate_by_model", {})
        for i, model in enumerate(models):
            matrix[i, j] = rates.get(model, 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"{d}\n({_dim_name(d)})" for d in dims])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("llama3.1:", "llama3.1\n") for m in models], fontsize=8)

    for i in range(len(models)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Failure Rate")
    ax.set_title("Failure Rate Heatmap: Model × ND Dimension\n(red = high failure rate, green = low)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _dim_name(d: str) -> str:
    return {"TC": "Task Chunk", "SC": "Session Cons.", "FA": "Frustration Adapt."}[d]
