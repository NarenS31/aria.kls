#!/usr/bin/env python3.11
"""Multi-dataset external validation — seven real datasets, reported separately.

    python3.11 eval/eval/external_validation_multi.py
    python3.11 eval/eval/external_validation_multi.py --quiet   # table only

Runs the SAME behavioral proxy rules and the SAME two checks over seven
independent classroom/platform datasets spanning 2004-2020, and reports each one
on its own. Nothing is pooled, averaged, or combined into a headline number: the
datasets differ in platform, grain, and which fields they even have, so a mean
across them would describe no real population. Every metric in the output JSON
belongs to exactly one dataset.

THE TWO CHECKS

  distribution match — chi-square goodness-of-fit between the proxy states this
      dataset produces and the state prevalence ARIA assumes (taken from the
      synthetic think-aloud corpus). Answers: does real student behaviour
      distribute the way ARIA's model expects?

  timing agreement — the same rows labelled two independent ways: the proxy
      rules (Method 1) versus ARIA's own BehavioralFeatureExtractor driven by
      response time (Method 2), restricted to rows Method 1 calls a
      timing-driven state (RUSHING / PLANNING / STUCK). Answers: do ARIA's
      absolute keystroke thresholds track percentile-based real behaviour? This
      is the same metric and the same restriction as
      eval/eval/assistments_behavioral.py, so the numbers are comparable.

WHY SOME CELLS ARE EMPTY

A check is computable only if the dataset carries the fields its rules read.
Three of the seven do not, and each says which field is missing and why:
ASSISTments 2015 ships four columns with no behavioral data at all; OULAD is a
distance-learning VLE with no per-problem interaction to log; EdNet and Eedi
record one answer per question with no hints, and their timing cannot be
attributed per problem (see each adapter). An uncomputable check is reported as
uncomputable. It is never filled in with a substitute signal, because a number
that is not the same measurement as its neighbours is worse than a blank.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EVAL_ROOT)
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from datasets.adapters.aleven_fractions import AlevenFractionsAdapter  # noqa: E402
from datasets.adapters.assistments_2004 import Assistments2004Adapter  # noqa: E402
from datasets.adapters.assistments_2005 import Assistments2005Adapter  # noqa: E402
from datasets.adapters.assistments_2015 import Assistments2015Adapter  # noqa: E402
from datasets.adapters.base import DatasetNotAvailableError  # noqa: E402
from datasets.adapters.behavioral_base import SHARED_PROXY_RULES  # noqa: E402
from datasets.adapters.ednet_riiid import EdnetRiiidAdapter  # noqa: E402
from datasets.adapters.eedi_neurips2020 import EediNeurips2020Adapter  # noqa: E402
from datasets.adapters.oulad import OuladAdapter  # noqa: E402

# Reused verbatim so this validation measures the same things the existing
# single-dataset experiments do.
from eval.assistments_behavioral import (  # noqa: E402
    aria_keystroke_from_row,
)
from eval.external_validation import (  # noqa: E402
    chi_square_gof,
    synthetic_state_distribution,
)
from metacognition.behavioral import (  # noqa: E402
    SIGNAL_TO_STATE,
    BehavioralFeatureExtractor,
    dominant_signal,
)

EXTERNAL = os.path.join(EVAL_ROOT, "data", "external")
RESULTS_PATH = os.path.join(
    EVAL_ROOT, "data", "eval", "external_validation_multi_dataset.json")

#: Applied to every dataset so "rows used" means the same thing in each row of
#: the summary table. Sources smaller than this are used whole.
MAX_ROWS = 500_000

#: The chi-square expectation comes from the synthetic think-aloud corpus, which
#: was generated deliberately balanced (exactly 50 items per state, 350 total).
#: That makes "ARIA's assumed prevalence" uniform as a training-set construction
#: choice, not as a claim about how often real students are in each state. It
#: matters for reading the result: rejecting uniformity is a weak finding, since
#: nobody expects real behaviour to be uniform. What the test actually shows is
#: the SHAPE of the departure, per dataset, which is in `observed` vs `expected`.
EXPECTATION_CAVEAT = (
    "NOTE: this expectation is uniform because the synthetic corpus is balanced\n"
    "by construction (50 items per state), not because uniform prevalence is\n"
    "claimed for real students. Read the per-state observed/expected counts,\n"
    "not just the p-value: rejecting uniformity is close to a foregone result."
)

#: The states ARIA's extractor can express from timing alone. Agreement is
#: restricted to these for a like-for-like comparison, exactly as in
#: eval/eval/assistments_behavioral.py.
TIMING_STATES = {"RUSHING", "PLANNING", "STUCK"}

DATASETS = [
    (Assistments2004Adapter, os.path.join(EXTERNAL, "assistments_2004", "raw")),
    (Assistments2005Adapter, os.path.join(EXTERNAL, "assistments_2005", "raw")),
    (Assistments2015Adapter, os.path.join(EXTERNAL, "assistments_2015", "raw")),
    (AlevenFractionsAdapter, os.path.join(EXTERNAL, "aleven_fractions", "raw")),
    (OuladAdapter, os.path.join(EXTERNAL, "oulad", "raw")),
    (EdnetRiiidAdapter, os.path.join(EXTERNAL, "ednet_riiid", "raw")),
    (EediNeurips2020Adapter, os.path.join(EXTERNAL, "eedi_neurips2020", "raw")),
]


def aria_extractor_state(extractor, response_time_ms):
    """Method 2 label: ARIA's own behavioral extractor, driven by response time.

    Goes through aria_keystroke_from_row so the response-time-to-keystroke
    mapping is the one the existing experiment uses, not a second copy of it.
    """
    payload = aria_keystroke_from_row({"ms_first_response": response_time_ms})
    features = extractor.extract("", payload, [])
    signal, _score = dominant_signal(features["behavioral_state_signals"])
    return SIGNAL_TO_STATE.get(signal)


def _no_rules_reason(computability: dict, matched_nothing: bool = False) -> str:
    """Why this dataset produced no proxy states, in the dataset's own terms."""
    blocked = computability.get("rules_blocked", {})
    vacuous = computability.get("rules_vacuous", {})
    parts = []
    if blocked:
        missing = sorted({f for fields in blocked.values() for f in fields})
        parts.append(
            f"{len(blocked)} of {len(blocked) + len(vacuous) + len(computability.get('rules_computable', []))} "
            f"rules read a field this dataset lacks ({', '.join(missing)})")
    if vacuous:
        parts.append("; ".join(
            f"{state} cannot fire because {why}" for state, why in vacuous.items()))
    if not parts:
        parts.append("no rule matched any row")
    tail = (" No rule can therefore produce a state, so there is no "
            "distribution to test." if not matched_nothing else
            " No state was produced, so there is no distribution to test.")
    return ". ".join(parts) + "." + tail


def run_dataset(adapter_class, raw_dir: str, expected: dict[str, float]) -> dict:
    adapter = adapter_class(raw_dir=raw_dir, max_rows=MAX_ROWS)
    entry = adapter.describe()
    entry["raw_dir"] = os.path.relpath(raw_dir, REPO_ROOT)

    computability = entry["computability"]
    timing_usable = adapter.FIELDS.get("response_time")
    timing_usable = bool(timing_usable and timing_usable.usable)

    extractor = BehavioralFeatureExtractor()
    proxy_counts: Counter = Counter()
    method2_counts: Counter = Counter()
    confusion: dict[str, Counter] = {}
    timing_total = timing_agree = 0
    correct_true = correct_known = 0
    n_records = 0

    try:
        for record in adapter.to_aria_schema():
            n_records += 1
            features = record["behavioral_features"]
            state1 = record["aria_state_proxy"]
            if state1 is not None:
                proxy_counts[state1] += 1
            if features["correct"] is not None:
                correct_known += 1
                correct_true += bool(features["correct"])

            if timing_usable and features["response_time_ms"]:
                state2 = aria_extractor_state(
                    extractor, features["response_time_ms"])
                if state2 is not None:
                    method2_counts[state2] += 1
                if state1 in TIMING_STATES:
                    timing_total += 1
                    confusion.setdefault(state1, Counter())[state2 or "none"] += 1
                    if state1 == state2:
                        timing_agree += 1
    except DatasetNotAvailableError as exc:
        entry.update({"status": "excluded", "reason": str(exc)})
        return entry
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        entry.update({"status": "excluded",
                      "reason": f"{type(exc).__name__}: {exc}"})
        return entry

    stats = adapter._last_stats
    entry.update({
        "status": "ran",
        "n_rows_used": stats.get("n_rows_used", n_records),
        "n_students": stats.get("n_students"),
        "n_problems": stats.get("n_problems"),
        "n_problems_with_response_time_percentiles":
            stats.get("n_problems_with_percentiles"),
        "correct_rate": (round(correct_true / correct_known, 4)
                         if correct_known else None),
        "n_with_known_correctness": correct_known,
        "proxy_state_counts": dict(proxy_counts),
        "proxy_coverage": (round(sum(proxy_counts.values()) / n_records, 4)
                           if n_records else 0.0),
    })

    # -- check 1: distribution match -----------------------------------
    if not computability["distribution_check_computable"]:
        entry["distribution_check"] = {
            "computable": False,
            "reason": _no_rules_reason(computability),
        }
    elif not proxy_counts:
        # Every rule's fields are present, but nothing matched. If a rule is
        # unfirable because a platform constant contradicts its condition, say
        # that rather than reporting an empty result as a surprise.
        entry["distribution_check"] = {
            "computable": False,
            "reason": _no_rules_reason(computability, matched_nothing=True),
        }
    else:
        result = chi_square_gof(proxy_counts, expected)
        result["computable"] = True
        result["expected_source"] = ("ARIA assumed state prevalence "
                                     "(synthetic think-aloud corpus)")
        # scipy's survival function underflows to exactly 0 for large chi2, and
        # a printed "p = 0" reads as a missing value rather than a tiny one.
        if result.get("p_value") == 0.0:
            result["p_value_underflow"] = True
            result["p_value_display"] = "< 1e-300"
        else:
            result["p_value_display"] = (
                f"{result['p_value']:.4g}" if result.get("p_value") is not None
                else "n/a")
        entry["distribution_check"] = result

    # -- check 2: timing agreement -------------------------------------
    if not timing_usable:
        status = adapter.FIELDS.get("response_time")
        entry["timing_check"] = {
            "computable": False,
            "reason": status.reason if status else "no response time available",
            "field_status": status.status if status else "absent",
        }
    elif not timing_total:
        entry["timing_check"] = {
            "computable": False,
            "reason": "response times exist but no row was labelled a "
                      "timing-driven state (RUSHING/PLANNING/STUCK), so there "
                      "is nothing to compare.",
        }
    else:
        entry["timing_check"] = {
            "computable": True,
            "agreement_rate": round(timing_agree / timing_total, 4),
            "n_compared": timing_total,
            "n_agreed": timing_agree,
            "restricted_to": sorted(TIMING_STATES),
            "method_1": "shared proxy rules (per-problem RT percentiles + "
                        "attempts + hints + correctness)",
            "method_2": "ARIA BehavioralFeatureExtractor on response time",
            "method_2_label_distribution": dict(method2_counts),
            "confusion_method1_vs_method2": {
                state: dict(counts) for state, counts in confusion.items()},
        }

    entry["checks_computable"] = [
        name for name, computed in (
            ("distribution", entry["distribution_check"].get("computable")),
            ("timing", entry["timing_check"].get("computable")))
        if computed
    ]
    return entry


def print_dataset(entry: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{entry['dataset']}  —  {entry['platform']} ({entry['year']})")
    print("=" * 78)
    if entry["status"] != "ran":
        print(f"  EXCLUDED: {entry.get('reason')}")
        return
    if not entry["commercial_use_allowed"]:
        print("  LICENCE: non-commercial use only")
    print(f"  rows used         {entry['n_rows_used']:,}"
          f"   students {entry['n_students']:,}"
          f"   problems {entry['n_problems']:,}")
    if entry["correct_rate"] is not None:
        print(f"  correctness rate  {entry['correct_rate']:.1%} "
              f"over {entry['n_with_known_correctness']:,} graded rows")

    print("  fields:")
    for name, status in entry["fields"].items():
        mark = "+" if status["usable"] else "-"
        detail = status["source"] or status["reason"]
        print(f"    {mark} {name:<15}{status['status']:<10}{detail[:44]}")

    counts = entry["proxy_state_counts"]
    print(f"  proxy states      {counts if counts else 'none — no rule could fire'}"
          f"   (coverage {entry['proxy_coverage']:.1%})")

    dist = entry["distribution_check"]
    if dist.get("computable"):
        print(f"  distribution      chi2={dist['chi2']} dof={dist['dof']} "
              f"p={dist.get('p_value_display', dist['p_value'])}")
    else:
        print(f"  distribution      NOT COMPUTABLE — {dist['reason']}")

    timing = entry["timing_check"]
    if timing.get("computable"):
        print(f"  timing agreement  {timing['agreement_rate']:.1%} "
              f"over {timing['n_compared']:,} rows "
              f"(restricted to {', '.join(timing['restricted_to'])})")
    else:
        print(f"  timing agreement  NOT COMPUTABLE — {timing['reason'][:70]}")


def print_summary(entries: list[dict]) -> None:
    print("\n" + "=" * 124)
    print("EXTERNAL VALIDATION — PER DATASET (nothing pooled)")
    print("=" * 124)
    header = (f"{'Dataset':<19}{'Source/Platform':<26}{'Year':<6}"
              f"{'N Students':>11}  {'N Transactions':>15}  "
              f"{'Dist. match (p)':>16}  {'Timing agree':>13}  Checks")
    print(header)
    print("-" * 124)
    for entry in entries:
        if entry["status"] != "ran":
            print(f"{entry['dataset']:<19}{entry['platform']:<26}"
                  f"{entry['year']:<6}{'—':>11}  {'—':>15}  {'excluded':>16}  "
                  f"{'—':>13}  {entry.get('reason', '')[:24]}")
            continue
        dist = entry["distribution_check"]
        timing = entry["timing_check"]
        p_text = (dist.get("p_value_display", "n/a") if dist.get("computable")
                  else "not computable")
        t_text = (f"{timing['agreement_rate']:.1%}" if timing.get("computable")
                  else "not computable")
        checks = ", ".join(entry["checks_computable"]) or "none"
        print(f"{entry['dataset']:<19}{entry['platform']:<26}{entry['year']:<6}"
              f"{entry['n_students']:>11,}  {entry['n_rows_used']:>15,}  "
              f"{p_text:>16}  {t_text:>13}  {checks}")
    print("-" * 124)
    ran = [e for e in entries if e["status"] == "ran"]
    with_dist = [e for e in ran if "distribution" in e["checks_computable"]]
    with_timing = [e for e in ran if "timing" in e["checks_computable"]]
    print(f"{len(ran)}/{len(entries)} datasets mapped without error · "
          f"{len(with_dist)} support the distribution check · "
          f"{len(with_timing)} support the timing check")
    print("Per-dataset only. These columns are not averaged: the datasets differ "
          "in platform, grain and available fields.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true",
                    help="Print the summary table only.")
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    expected = synthetic_state_distribution()
    uniform = len(set(round(p, 6) for p in expected.values())) == 1
    if not args.quiet:
        print("ARIA assumed state prevalence (chi-square expectation):")
        for state, proportion in sorted(expected.items(),
                                        key=lambda kv: -kv[1]):
            print(f"    {state:<12}{proportion:.3f}")
        if uniform:
            print("    " + EXPECTATION_CAVEAT.replace("\n", "\n    "))

    entries = []
    for adapter_class, raw_dir in DATASETS:
        entry = run_dataset(adapter_class, raw_dir, expected)
        entries.append(entry)
        if not args.quiet:
            print_dataset(entry)

    print_summary(entries)

    report = {
        "generated_on": date.today().isoformat(),
        "generator": "eval/eval/external_validation_multi.py",
        "tier": "PROXY",
        "pooling": "none — every dataset is reported independently and no "
                   "metric is averaged across datasets",
        "max_rows_per_dataset": MAX_ROWS,
        "shared_proxy_rules": SHARED_PROXY_RULES,
        "aria_expected_state_distribution": expected,
        "aria_expected_state_distribution_source":
            "eval/data/synthetic_thinkaloud/test.jsonl (350 items, 50 per state)",
        "aria_expected_state_distribution_caveat":
            EXPECTATION_CAVEAT.replace("\n", " "),
        "timing_agreement_definition": {
            "method_1": "shared behavioral proxy rules",
            "method_2": "ARIA BehavioralFeatureExtractor on response time",
            "restricted_to": sorted(TIMING_STATES),
            "same_as": "eval/eval/assistments_behavioral.py",
        },
        "n_datasets": len(entries),
        "n_datasets_mapped": sum(1 for e in entries if e["status"] == "ran"),
        "datasets": entries,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nWrote {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
