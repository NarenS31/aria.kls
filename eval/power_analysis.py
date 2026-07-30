#!/usr/bin/env python3.11
"""Transparent planning calculations for a two-arm education study.

This is a screening calculation, not a substitute for a statistician or
simulation tailored to the final outcome model.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist


def required_sample(
    *,
    standardized_effect: float,
    power: float = 0.80,
    alpha: float = 0.05,
    pretest_r_squared: float = 0.0,
    cluster_size: float = 1.0,
    icc: float = 0.0,
    attrition: float = 0.0,
) -> dict:
    if not 0 < standardized_effect:
        raise ValueError("standardized_effect must be positive")
    for name, value in {
        "power": power,
        "alpha": alpha,
        "pretest_r_squared": pretest_r_squared,
        "icc": icc,
        "attrition": attrition,
    }.items():
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if not power > 0.5:
        raise ValueError("power must be greater than 0.5")
    if cluster_size < 1:
        raise ValueError("cluster_size must be at least 1")

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    base_per_arm = (
        2
        * (z_alpha + z_power) ** 2
        * (1 - pretest_r_squared)
        / standardized_effect ** 2
    )
    design_effect = 1 + (cluster_size - 1) * icc
    adjusted_per_arm = base_per_arm * design_effect / (1 - attrition)
    per_arm = math.ceil(adjusted_per_arm)
    return {
        "standardized_effect": standardized_effect,
        "power": power,
        "two_sided_alpha": alpha,
        "pretest_r_squared": pretest_r_squared,
        "cluster_size": cluster_size,
        "icc": icc,
        "attrition": attrition,
        "design_effect": round(design_effect, 4),
        "required_per_arm": per_arm,
        "required_total": per_arm * 2,
        "method": (
            "normal approximation for two independent means, multiplied by "
            "ANCOVA variance factor, cluster design effect, and attrition factor"
        ),
        "warning": (
            "Replace with a simulation using pilot variance, the final outcome "
            "model, number of clusters, and institution-approved assumptions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=float, required=True)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pretest-r2", type=float, default=0.0)
    parser.add_argument("--cluster-size", type=float, default=1.0)
    parser.add_argument("--icc", type=float, default=0.0)
    parser.add_argument("--attrition", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(required_sample(
        standardized_effect=args.effect,
        power=args.power,
        alpha=args.alpha,
        pretest_r_squared=args.pretest_r2,
        cluster_size=args.cluster_size,
        icc=args.icc,
        attrition=args.attrition,
    ), indent=2))


if __name__ == "__main__":
    main()
