"""Research-grade paired statistics for ARIA experiments.

The unit of inference is an episode, not an individual turn.  This avoids
pseudoreplication when several responses come from the same simulated or real
student episode.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_episode_differences(
    rows: Iterable[dict],
    treatment: str,
    control: str,
    metric: str,
) -> dict[str, float]:
    """Average repeated rows inside an episode, then compute paired differences."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in {treatment, control} or metric not in row:
            continue
        grouped[(str(row["episode_id"]), condition)].append(float(row[metric]))

    episode_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for (episode_id, condition), values in grouped.items():
        episode_scores[episode_id][condition] = _mean(values)
    return {
        episode_id: scores[treatment] - scores[control]
        for episode_id, scores in episode_scores.items()
        if treatment in scores and control in scores
    }


def paired_randomization_test(
    differences: Iterable[float],
    *,
    iterations: int = 20_000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip randomization test."""
    values = list(differences)
    if not values:
        return 1.0
    observed = abs(_mean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = _mean([
            value if rng.random() < 0.5 else -value for value in values
        ])
        extreme += abs(permuted) >= observed - 1e-12
    return (extreme + 1) / (iterations + 1)


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    iterations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Episode-level percentile bootstrap CI for a paired mean difference."""
    values = list(differences)
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    draws = sorted(
        _mean([rng.choice(values) for _ in values])
        for _ in range(iterations)
    )
    lower = draws[int(iterations * alpha / 2)]
    upper = draws[min(iterations - 1, int(iterations * (1 - alpha / 2)))]
    return _mean(values), lower, upper


def paired_standardized_effect(differences: Iterable[float]) -> float:
    """Cohen's dz for paired observations."""
    values = list(differences)
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return math.inf if mean else 0.0
    return mean / sd


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise error correction."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, (total - index) * value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def compare_conditions(
    rows: list[dict],
    *,
    treatment: str,
    control: str,
    metrics: list[str],
    seed: int = 42,
) -> dict:
    """Return paired effects, CIs, randomization p-values, and Holm p-values."""
    results = {}
    raw_p = {}
    for offset, metric in enumerate(metrics):
        diffs = paired_episode_differences(rows, treatment, control, metric)
        mean, lower, upper = paired_bootstrap_ci(
            diffs.values(), seed=seed + offset
        )
        p_value = paired_randomization_test(
            diffs.values(), seed=seed + offset
        )
        raw_p[metric] = p_value
        results[metric] = {
            "n_paired_episodes": len(diffs),
            "mean_difference": round(mean, 4),
            "ci_95": [round(lower, 4), round(upper, 4)],
            "cohens_dz": round(paired_standardized_effect(diffs.values()), 4),
            "p_randomization": round(p_value, 6),
        }
    adjusted = holm_adjust(raw_p)
    for metric in metrics:
        results[metric]["p_holm"] = round(adjusted[metric], 6)
    return results
