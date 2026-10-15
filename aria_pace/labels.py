"""Mastery, wheel-spinning labels and eligibility (SPEC 2-3)."""
import numpy as np

from .constants import MASTERY_RUN, T_DETERMINE


def first_mastery(correct) -> float:
    """First 1-based opportunity at which MASTERY_RUN consecutive correct responses are completed, else inf."""
    run = 0
    for i, x in enumerate(correct, start=1):
        run = run + 1 if x == 1 else 0
        if run >= MASTERY_RUN:
            return float(i)
    return float(np.inf)


def label(n: int, m: float, variant: str):
    """Return 1 (wheel-spinning), 0 (mastered within T), or None (unlabelled leaver)."""
    if m <= T_DETERMINE:
        return 0
    need = T_DETERMINE if variant == "A" else T_DETERMINE + 1
    return 1 if n >= need else None


def eligibility(n: int, m: float, k: int) -> str:
    """'short' (n<k), 'mastered' (m<=k) or 'eligible'. Exclusion reasons are counted separately."""
    if n < k:
        return "short"
    if m <= k:
        return "mastered"
    return "eligible"
