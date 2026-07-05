"""
Statistical scoring for ARIA evaluation results.

- Weighted score computation (PEBBLE-compatible)
- Nonparametric bootstrap 95% CIs (10 000 iterations)
- Paired t-test ARIA vs baseline per dimension
- Cohen's d effect size per dimension
- Reads from JSONL logs, outputs summary dict
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.rubric import WEIGHTS


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_results(path: Path) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ------------------------------------------------------------------
# Bootstrap CI
# ------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_iter: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Returns (mean, lower_CI, upper_CI)."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_iter * (alpha / 2))]
    hi = means[int(n_iter * (1 - alpha / 2))]
    return (sum(values) / n, lo, hi)


# ------------------------------------------------------------------
# Paired t-test (two-tailed, assumes equal-length lists)
# ------------------------------------------------------------------

def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Returns (t_stat, p_value). Trims to min length."""
    n = min(len(a), len(b))
    if n < 2:
        return (0.0, 1.0)
    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    if mean_d == 0.0:
        return (0.0, 1.0)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0.0:
        # All diffs identical and non-zero: perfect separation, p→0
        return (float("inf"), 0.0)
    se = math.sqrt(var_d / n)
    if se == 0:
        return (0.0, 1.0)
    t = mean_d / se
    # Approximate p-value using t-distribution (large n → normal approx)
    # Using Abramowitz & Stegun approximation for CDF
    p = _approx_t_pvalue(t, df=n - 1)
    return (round(t, 4), round(p, 6))


def _approx_t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value approximation via normal distribution (adequate for df > 30)."""
    import math
    if df >= 30:
        # Use standard normal approximation
        z = abs(t)
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
        return min(1.0, 2 * p_one_tail)
    # For small df, use a rough correction factor
    z = abs(t) * math.sqrt(df / (df + t ** 2 + 1e-9))
    p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return min(1.0, 2 * p_one_tail)


# ------------------------------------------------------------------
# Cohen's d
# ------------------------------------------------------------------

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d effect size (positive = group1 > group2)."""
    if not group1 or not group2:
        return 0.0
    m1 = sum(group1) / len(group1)
    m2 = sum(group2) / len(group2)
    n1, n2 = len(group1), len(group2)
    var1 = sum((x - m1) ** 2 for x in group1) / max(n1 - 1, 1)
    var2 = sum((x - m2) ** 2 for x in group2) / max(n2 - 1, 1)
    pooled_sd = math.sqrt((var1 * (n1 - 1) + var2 * (n2 - 1)) / max(n1 + n2 - 2, 1))
    if pooled_sd == 0:
        return float("inf") if m1 != m2 else 0.0
    return round((m1 - m2) / pooled_sd, 4)


# ------------------------------------------------------------------
# Full summarization
# ------------------------------------------------------------------

def summarize(
    aria_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
    records_aria: Optional[List[dict]] = None,
    records_baseline: Optional[List[dict]] = None,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Compute full statistical summary comparing ARIA vs baseline.
    Accepts either file paths or pre-loaded record lists.
    """
    if records_aria is None and aria_path:
        records_aria = load_results(aria_path)
    if records_baseline is None and baseline_path:
        records_baseline = load_results(baseline_path)

    records_aria = records_aria or []
    records_baseline = records_baseline or []

    def extract_dim_scores(records: List[dict]) -> Dict[str, List[float]]:
        by_dim: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            scores = r.get("scores", {})
            for dim in WEIGHTS:
                if dim in scores:
                    by_dim[dim].append(float(scores[dim]))
            ws = r.get("weighted_score")
            if ws is not None:
                by_dim["weighted"].append(float(ws))
        return dict(by_dim)

    aria_dims = extract_dim_scores(records_aria)
    base_dims = extract_dim_scores(records_baseline)

    dimensions = list(WEIGHTS.keys()) + ["weighted"]
    results = {}

    for dim in dimensions:
        a_vals = aria_dims.get(dim, [])
        b_vals = base_dims.get(dim, [])

        a_mean, a_lo, a_hi = bootstrap_ci(a_vals, n_iter=n_bootstrap)
        b_mean, b_lo, b_hi = bootstrap_ci(b_vals, n_iter=n_bootstrap)
        t_stat, p_val = paired_ttest(a_vals, b_vals)
        d = cohens_d(a_vals, b_vals)

        results[dim] = {
            "aria": {"mean": round(a_mean, 4), "ci_lo": round(a_lo, 4), "ci_hi": round(a_hi, 4), "n": len(a_vals)},
            "baseline": {"mean": round(b_mean, 4), "ci_lo": round(b_lo, 4), "ci_hi": round(b_hi, 4), "n": len(b_vals)},
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": d,
            "significant": p_val < 0.05,
        }

    # Per-persona breakdown
    persona_results = {}
    for persona in ["hyperfocus", "scattered", "frustrated"]:
        a_sub = [r for r in records_aria if r.get("persona") == persona]
        b_sub = [r for r in records_baseline if r.get("persona") == persona]
        a_ws = [r.get("weighted_score", 0.0) for r in a_sub]
        b_ws = [r.get("weighted_score", 0.0) for r in b_sub]
        a_mean, a_lo, a_hi = bootstrap_ci(a_ws, n_iter=min(n_bootstrap, 1000))
        b_mean, b_lo, b_hi = bootstrap_ci(b_ws, n_iter=min(n_bootstrap, 1000))
        persona_results[persona] = {
            "aria": {"mean": round(a_mean, 4), "ci_lo": round(a_lo, 4), "ci_hi": round(a_hi, 4)},
            "baseline": {"mean": round(b_mean, 4), "ci_lo": round(b_lo, 4), "ci_hi": round(b_hi, 4)},
        }

    # Per-subject breakdown
    subject_results = {}
    for subject in ["algebra", "biology", "python_programming"]:
        a_sub = [r for r in records_aria if r.get("subject") == subject]
        b_sub = [r for r in records_baseline if r.get("subject") == subject]
        a_ws = [r.get("weighted_score", 0.0) for r in a_sub]
        b_ws = [r.get("weighted_score", 0.0) for r in b_sub]
        a_mean, a_lo, a_hi = bootstrap_ci(a_ws, n_iter=min(n_bootstrap, 1000))
        b_mean, b_lo, b_hi = bootstrap_ci(b_ws, n_iter=min(n_bootstrap, 1000))
        subject_results[subject] = {
            "aria": {"mean": round(a_mean, 4), "ci_lo": round(a_lo, 4), "ci_hi": round(a_hi, 4)},
            "baseline": {"mean": round(b_mean, 4), "ci_lo": round(b_lo, 4), "ci_hi": round(b_hi, 4)},
        }

    # Solution dump rate
    aria_dump_rate = sum(1 for r in records_aria if r.get("solution_dump")) / max(len(records_aria), 1)
    base_dump_rate = sum(1 for r in records_baseline if r.get("solution_dump")) / max(len(records_baseline), 1)

    return {
        "per_dimension": results,
        "per_persona": persona_results,
        "per_subject": subject_results,
        "solution_dump_rate": {
            "aria": round(aria_dump_rate, 4),
            "baseline": round(base_dump_rate, 4),
        },
        "n_records": {"aria": len(records_aria), "baseline": len(records_baseline)},
    }


# ------------------------------------------------------------------
# Learning curve: SC scores vs N pre-seeded sessions
# ------------------------------------------------------------------

def learning_curve_summary(
    results_by_n: Dict[int, List[dict]],
    dim: str = "SC",
    n_bootstrap: int = 1000,
) -> Dict[int, dict]:
    """
    results_by_n: {n_sessions: [records]}
    Returns {n: {"mean": ..., "ci_lo": ..., "ci_hi": ...}} for the given dim.
    """
    output = {}
    for n_sessions, records in sorted(results_by_n.items()):
        vals = [float(r["scores"][dim]) for r in records if dim in r.get("scores", {})]
        mean, lo, hi = bootstrap_ci(vals, n_iter=n_bootstrap)
        output[n_sessions] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
    return output
