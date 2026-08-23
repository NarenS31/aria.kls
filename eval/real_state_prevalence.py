#!/usr/bin/env python3.11
"""Real state prevalence vs the synthetic corpus's uniform assumption.

    python3.11 eval/real_state_prevalence.py
    python3.11 eval/real_state_prevalence.py --quiet

ARIA's synthetic think-aloud corpus is balanced by construction: exactly 50
items for each of the seven states. Every downstream comparison that treats that
corpus as "what ARIA expects" therefore carries an implicit claim that students
are in each state about a seventh of the time. This script measures what the
real classroom data actually shows and reports the size of the gap.

The measurement is already computed — the external validation run derives a
behavioral proxy state for every ASSISTments row — so this only aggregates it,
reads the synthetic corpus for the expected side, and does the arithmetic.

TWO COMPARISONS, BECAUSE ONE OF THEM IS UNFAIR

The proxy rules can never emit INSIGHT: a moment of realization is not
recoverable from response time, attempts and hints, and the rule set says so
explicitly. Comparing a six-state-capable proxy against a seven-state uniform
expectation therefore charges the proxy for a state it was never able to
produce, and about a seventh of the expected mass is unreachable by
construction. Both comparisons are reported:

    vs uniform 1/7   what the synthetic corpus literally assumes
    vs uniform 1/6   the same assumption restricted to the states behaviour can
                     actually express — the fair test, and the headline number

POOLING

The two ASSISTments years are combined, and nothing else is. They are the same
platform, the same instrument and adjacent cohorts, which is the case where
combining describes a real population. Aleven Fractions Lab is reported beside
them as an independent contrast and is never merged in: it is a different
platform at a different interaction grain, and averaging it with ASSISTments
would describe nobody.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RESULTS_DIR = os.path.join(HERE, "data", "eval")
MULTI_PATH = os.path.join(RESULTS_DIR, "external_validation_multi_dataset.json")
SYNTHETIC_PATH = os.path.join(HERE, "data", "synthetic_thinkaloud", "test.jsonl")
OUT_PATH = os.path.join(RESULTS_DIR, "real_vs_synthetic_prevalence.json")

#: The two runs that are the same instrument and may be combined.
ASSISTMENTS_KEYS = ("assistments_2004", "assistments_2005")
#: Reported alongside, never merged in.
CONTRAST_KEYS = ("aleven_fractions",)

ALL_STATES = ("PLANNING", "FLOW", "CONFUSED", "RUSHING", "FRUSTRATED",
              "STUCK", "INSIGHT")
#: Not derivable from behaviour by design; see the shared proxy rules.
NOT_BEHAVIORALLY_DERIVABLE = ("INSIGHT",)
DERIVABLE_STATES = tuple(s for s in ALL_STATES
                         if s not in NOT_BEHAVIORALLY_DERIVABLE)


def load_json(path: str):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\nRun eval/eval/external_validation_multi.py first — "
            "this script only aggregates what that run already computed.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def synthetic_counts(path: str) -> Counter:
    """State counts in the synthetic corpus, read rather than assumed."""
    counts: Counter = Counter()
    if not os.path.exists(path):
        return counts
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                state = json.loads(line).get("cognitive_state")
            except json.JSONDecodeError:
                continue
            if state in ALL_STATES:
                counts[state] += 1
    return counts


def chi_square(observed: dict[str, int], expected: dict[str, float]) -> dict:
    """Goodness-of-fit over the states the expectation covers.

    scipy underflows to exactly 0 for large statistics, so a p of 0 is reported
    as an underflow rather than printed as if it were a measured zero.
    """
    states = [s for s in expected if expected[s] > 0]
    total = sum(observed.get(s, 0) for s in states)
    if not total:
        return {"computable": False, "reason": "no observations"}
    stat = 0.0
    for state in states:
        exp = expected[state] * total
        obs = observed.get(state, 0)
        if exp > 0:
            stat += (obs - exp) ** 2 / exp
    dof = max(1, len(states) - 1)
    p_value = None
    try:
        from scipy.stats import chi2
        p_value = float(chi2.sf(stat, dof))
    except Exception:                                   # noqa: BLE001
        pass
    # Cramer's V for a goodness-of-fit test: 0 = perfect fit, 1 = maximal.
    cramers_v = (stat / (total * dof)) ** 0.5 if total else None
    return {
        "computable": True,
        "chi2": round(stat, 3),
        "dof": dof,
        "n": total,
        "p_value": p_value,
        "p_value_display": ("< 1e-300" if p_value == 0.0 else
                            (f"{p_value:.4g}" if p_value is not None else "n/a")),
        "p_value_underflow": p_value == 0.0,
        "cramers_v": round(cramers_v, 4) if cramers_v is not None else None,
    }


def compare(observed: dict[str, int], states: tuple[str, ...]) -> dict:
    """Observed prevalence against a uniform expectation over `states`."""
    total = sum(observed.get(s, 0) for s in states)
    expected_share = 1.0 / len(states)
    expected = {s: expected_share for s in states}
    rows = []
    for state in states:
        count = observed.get(state, 0)
        share = count / total if total else 0.0
        rows.append({
            "state": state,
            "n": count,
            "observed_share": round(share, 4),
            "expected_share": round(expected_share, 4),
            "difference": round(share - expected_share, 4),
            "ratio_to_expected": (round(share / expected_share, 2)
                                  if expected_share else None),
        })
    rows.sort(key=lambda r: -r["observed_share"])
    total_variation = round(
        0.5 * sum(abs(r["observed_share"] - r["expected_share"]) for r in rows), 4)
    return {
        "states_compared": list(states),
        "n": total,
        "expected_share_each": round(expected_share, 4),
        "per_state": rows,
        "total_variation_distance": total_variation,
        "max_over_represented": rows[0] if rows else None,
        "max_under_represented": rows[-1] if rows else None,
        "chi_square": chi_square(observed, expected),
    }


def dataset_entry(multi: dict, key: str) -> dict | None:
    for entry in multi.get("datasets", []):
        if entry.get("dataset") == key:
            return entry
    return None


def build(multi: dict) -> dict:
    counts_by_dataset: dict[str, Counter] = {}
    missing = []
    for key in ASSISTMENTS_KEYS + CONTRAST_KEYS:
        entry = dataset_entry(multi, key)
        if not entry or entry.get("status") != "ran":
            missing.append(key)
            continue
        counts_by_dataset[key] = Counter(entry.get("proxy_state_counts", {}))
    if not any(k in counts_by_dataset for k in ASSISTMENTS_KEYS):
        raise SystemExit(
            "no ASSISTments proxy-state counts in "
            f"{os.path.relpath(MULTI_PATH, REPO_ROOT)}; nothing to aggregate.")

    combined: Counter = Counter()
    for key in ASSISTMENTS_KEYS:
        combined.update(counts_by_dataset.get(key, Counter()))

    synth = synthetic_counts(SYNTHETIC_PATH)
    synth_total = sum(synth.values())
    synth_shares = ({s: round(synth.get(s, 0) / synth_total, 4) for s in ALL_STATES}
                    if synth_total else {})
    synth_uniform = (len({round(v, 4) for v in synth_shares.values()}) == 1
                     if synth_shares else None)

    report = {
        "generated_by": "eval/real_state_prevalence.py",
        "question": "ARIA's synthetic corpus assumes every cognitive state is "
                    "equally likely. Do real classroom logs agree?",
        "source": os.path.relpath(MULTI_PATH, REPO_ROOT),
        "synthetic_corpus": {
            "path": os.path.relpath(SYNTHETIC_PATH, REPO_ROOT),
            "n": synth_total,
            "counts": dict(synth),
            "shares": synth_shares,
            "is_uniform_by_construction": synth_uniform,
            "note": "Balanced deliberately for classifier training. Uniformity "
                    "is a construction choice, not a claim about students — "
                    "which is exactly why it should not be cited as an "
                    "expected prevalence.",
        },
        "insight_caveat": {
            "states_not_behaviorally_derivable": list(NOT_BEHAVIORALLY_DERIVABLE),
            "detail": "The behavioral proxy rules cannot emit INSIGHT; it needs "
                      "the student's words. Roughly 1/7 of the synthetic "
                      "corpus's expected mass is therefore unreachable by any "
                      "behavioral method, so the 1/6 comparison over derivable "
                      "states is the fair one and is reported as the headline.",
        },
        "pooling": "The two ASSISTments years are combined (same platform, same "
                   "instrument, adjacent cohorts). Nothing else is combined.",
        "primary": {
            "label": "ASSISTments 2004 + 2005 (combined)",
            "datasets": list(ASSISTMENTS_KEYS),
            "counts": dict(combined),
            "vs_uniform_seven_states": compare(combined, ALL_STATES),
            "vs_uniform_six_derivable_states": compare(combined, DERIVABLE_STATES),
        },
        "per_dataset": {},
        "contrast_not_pooled": {},
    }

    for key in ASSISTMENTS_KEYS:
        if key in counts_by_dataset:
            report["per_dataset"][key] = {
                "counts": dict(counts_by_dataset[key]),
                "vs_uniform_six_derivable_states": compare(
                    counts_by_dataset[key], DERIVABLE_STATES),
            }
    for key in CONTRAST_KEYS:
        if key in counts_by_dataset:
            report["contrast_not_pooled"][key] = {
                "counts": dict(counts_by_dataset[key]),
                "vs_uniform_six_derivable_states": compare(
                    counts_by_dataset[key], DERIVABLE_STATES),
                "note": "Different platform and interaction grain. Reported for "
                        "contrast; deliberately not pooled with ASSISTments.",
            }
    if missing:
        report["missing_datasets"] = missing

    report["finding"] = finding_sentence(report)
    return report


def finding_sentence(report: dict) -> str:
    """The one sentence this belongs in the paper as."""
    fair = report["primary"]["vs_uniform_six_derivable_states"]
    top = fair["max_over_represented"]
    bottom = fair["max_under_represented"]
    chi = fair["chi_square"]
    return (
        f"Across {fair['n']:,} proxy-labelled interactions from two independent "
        f"ASSISTments cohorts, behavioral state prevalence is far from the "
        f"uniform distribution ARIA's synthetic corpus assumes: "
        f"{top['state']} accounts for {top['observed_share']:.1%} of "
        f"interactions against an expected {fair['expected_share_each']:.1%} "
        f"({top['ratio_to_expected']}x), while {bottom['state']} accounts for "
        f"{bottom['observed_share']:.1%} ({bottom['ratio_to_expected']}x). The "
        f"total variation distance from uniform is "
        f"{fair['total_variation_distance']:.3f} "
        f"(chi-square {chi['chi2']:,.0f}, df {chi['dof']}, "
        f"p {chi['p_value_display']}, Cramer's V {chi['cramers_v']}). A "
        f"synthetic corpus balanced across states is therefore a reasonable "
        f"training set but not a model of classroom prevalence, and should not "
        f"be cited as one."
    )


def print_comparison(title: str, block: dict) -> None:
    print(f"\n  {title}   (n = {block['n']:,}, expected "
          f"{block['expected_share_each']:.1%} each)")
    print(f"    {'state':<12}{'n':>9}{'observed':>10}{'expected':>10}"
          f"{'diff':>9}{'ratio':>8}")
    for row in block["per_state"]:
        print(f"    {row['state']:<12}{row['n']:>9,}"
              f"{row['observed_share']:>9.1%}{row['expected_share']:>10.1%}"
              f"{row['difference']:>+9.1%}{row['ratio_to_expected']:>7.2f}x")
    chi = block["chi_square"]
    if chi.get("computable"):
        print(f"    total variation distance {block['total_variation_distance']:.3f}"
              f"   chi2={chi['chi2']:,.0f} df={chi['dof']} "
              f"p={chi['p_value_display']} Cramer's V={chi['cramers_v']}")


def print_report(report: dict) -> None:
    rule = "=" * 78
    print(rule)
    print("REAL STATE PREVALENCE vs THE SYNTHETIC CORPUS'S UNIFORM ASSUMPTION")
    print(rule)
    synth = report["synthetic_corpus"]
    print(f"synthetic corpus  {synth['path']}  n={synth['n']:,}  "
          f"uniform by construction: {synth['is_uniform_by_construction']}")
    print(f"real source       {report['source']}")
    print(f"pooling           {report['pooling']}")
    print(f"\nINSIGHT caveat    {report['insight_caveat']['detail']}")

    print("\n" + "-" * 78)
    print(f"PRIMARY — {report['primary']['label']}")
    print("-" * 78)
    print_comparison("vs uniform 1/7 (what the corpus literally assumes)",
                     report["primary"]["vs_uniform_seven_states"])
    print_comparison("vs uniform 1/6 (derivable states only — the fair test)",
                     report["primary"]["vs_uniform_six_derivable_states"])

    if report["per_dataset"]:
        print("\n" + "-" * 78)
        print("EACH COHORT SEPARATELY (does the pattern replicate?)")
        print("-" * 78)
        for key, block in report["per_dataset"].items():
            print_comparison(key, block["vs_uniform_six_derivable_states"])

    if report["contrast_not_pooled"]:
        print("\n" + "-" * 78)
        print("CONTRAST — different platform, NOT pooled")
        print("-" * 78)
        for key, block in report["contrast_not_pooled"].items():
            print_comparison(key, block["vs_uniform_six_derivable_states"])
            print(f"    {block['note']}")

    print("\n" + rule)
    print("FINDING")
    print(rule)
    import textwrap
    print(textwrap.fill(report["finding"], 78))
    print(rule)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true",
                    help="Write the JSON without printing the report.")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    report = build(load_json(MULTI_PATH))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if not args.quiet:
        print_report(report)
    print(f"\nWrote {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
