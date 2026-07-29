#!/usr/bin/env python3.11
"""
Cross-check ARIA's behavioral state-signal thresholds against real ITS behaviour
using the ASSISTments dataset.

ASSISTments logs per-problem behavioural columns (time to first response, attempt
count, hint usage, correctness) but no keystroke stream and no think-aloud text.
Two independent proxy methods are compared on the same rows:

  METHOD 1 — ASSISTments-native proxy (mirrors datasets/adapters/assistments.py)
    ``assistments_to_behavioral_features(row)`` scores each of the five states
    directly from the ITS columns (percentile of ms_first_response, attempt
    count, hint count, correctness) and takes the arg-max as the proxy label.

  METHOD 2 — ARIA behavioral extractor
    The same row's timing is mapped to a keystroke payload
    (time-to-first-response -> pre-typing pause, the only behavioural signal
    ASSISTments carries) and passed through the *real*
    ``BehavioralFeatureExtractor``; the arg-max of its behavioral_state_signals
    is the proxy label.

We report the agreement rate between the two labellings. High agreement means
ARIA's absolute keystroke thresholds already track the percentile-based patterns
of real student behaviour — i.e. they are calibrated before any real think-aloud
data is collected.

If ``data/external/assistments/raw/`` contains ASSISTments CSVs they are used;
otherwise a deterministic synthetic ASSISTments-shaped table is generated so the
experiment still runs end-to-end (flagged as ``data_source: synthetic_fallback``
in the output).

Output: ``data/eval/assistments_behavioral_agreement.json``
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metacognition.behavioral import (  # noqa: E402
    BehavioralFeatureExtractor,
    SIGNAL_TO_STATE,
    dominant_signal,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "external", "assistments", "raw")
RESULTS_PATH = os.path.join(DATA_DIR, "eval", "assistments_behavioral_agreement.json")

# Candidate ASSISTments column names (datasets vary by release year).
COLUMN_ALIASES = {
    "ms_first_response": ["ms_first_response", "first_response_time",
                          "ms_first_response_time", "firstActionTime"],
    "attempt_count": ["attempt_count", "attemptCount", "attempts", "num_attempts"],
    "correct": ["correct", "is_correct", "correctness"],
    "hint_count": ["hint_count", "hintCount", "hints", "num_hints", "hint_total"],
}


# ------------------------------------------------------------------
# Row access + percentile helpers
# ------------------------------------------------------------------

def _get(row: dict, canonical: str):
    for name in COLUMN_ALIASES[canonical]:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_fields(row: dict) -> tuple:
    """Extract (correct: bool|None, attempt: int|None, hint: int|None, rt: int|None)."""
    def _i(canonical):
        v = _get(row, canonical)
        try:
            return None if v in (None, "") else int(float(v))
        except (TypeError, ValueError):
            return None
    c = _i("correct")
    return (None if c is None else bool(c), _i("attempt_count"),
            _i("hint_count"), _i("ms_first_response"))


def _load_reference_classifier():
    """Prefer the canonical proxy mapping from datasets/adapters/assistments.py.

    Returns (callable(correct, attempt, hint, rt, pct)->state|None, source_name).
    Falls back to the inline score functions (which mirror the same PROXY_RULES,
    per this task's spec) if the adapter is unavailable."""
    try:
        from datasets.adapters.assistments import AssistmentsAdapter
        return (lambda correct, attempt, hint, rt, pct:
                AssistmentsAdapter.classify(correct, attempt, hint, rt, pct)[0]), "adapter"
    except Exception:
        return None, "inline_scores"


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def compute_timing_stats(rows: list[dict]) -> dict:
    times = sorted(_num(_get(r, "ms_first_response")) for r in rows
                   if _get(r, "ms_first_response") is not None)
    return {
        "p10": _percentile(times, 10),
        "p25": _percentile(times, 25),
        "p75": _percentile(times, 75),
        "p90": _percentile(times, 90),
        "n": len(times),
    }


# ------------------------------------------------------------------
# METHOD 1 — ASSISTments-native state scores
# ------------------------------------------------------------------

def rushing_score(row: dict, stats: dict) -> float:
    ms = _num(_get(row, "ms_first_response"))
    attempts = _num(_get(row, "attempt_count"), 1)
    correct = _num(_get(row, "correct"))
    s = 0.0
    if ms < stats["p10"]:
        s += 0.5
    if attempts == 1:
        s += 0.25
    if correct == 0:
        s += 0.25
    return s


def stuck_score(row: dict, stats: dict) -> float:
    hints = _num(_get(row, "hint_count"))
    correct = _num(_get(row, "correct"))
    s = 0.0
    if hints >= 3:
        s += 0.6
    if correct == 0:
        s += 0.4
    return s


def flow_score(row: dict, stats: dict) -> float:
    ms = _num(_get(row, "ms_first_response"))
    attempts = _num(_get(row, "attempt_count"), 1)
    correct = _num(_get(row, "correct"))
    s = 0.0
    if attempts == 1:
        s += 0.34
    if correct == 1:
        s += 0.33
    if stats["p25"] <= ms <= stats["p75"]:
        s += 0.33
    return s


def planning_score(row: dict, stats: dict) -> float:
    ms = _num(_get(row, "ms_first_response"))
    attempts = _num(_get(row, "attempt_count"), 1)
    s = 0.0
    if ms > stats["p75"]:
        s += 0.6
    if attempts == 1:
        s += 0.4
    return s


def frustrated_score(row: dict, stats: dict) -> float:
    attempts = _num(_get(row, "attempt_count"), 1)
    correct = _num(_get(row, "correct"))
    hints = _num(_get(row, "hint_count"))
    s = 0.0
    if attempts >= 4:
        s += 0.5
    if correct == 0:
        s += 0.3
    if hints >= 2:
        s += 0.2
    return s


def assistments_to_behavioral_features(row: dict, stats: dict) -> dict:
    """Map one ASSISTments row to the BehavioralFeatureExtractor schema, with the
    behavioral_state_signals filled by the ASSISTments-native score functions."""
    return {
        "pause_before_first_key_ms": None,
        "total_typing_time_ms": _num(_get(row, "ms_first_response")) or None,
        "backspace_rate": None,
        "typing_speed_wpm": None,
        "response_length_words": None,
        "response_length_chars": None,
        "length_trend": None,
        "behavioral_state_signals": {
            "rushing": rushing_score(row, stats),
            "stuck": stuck_score(row, stats),
            "flow": flow_score(row, stats),
            "planning": planning_score(row, stats),
            "frustrated": frustrated_score(row, stats),
        },
    }


# ------------------------------------------------------------------
# METHOD 2 — ARIA behavioral extractor on the same row
# ------------------------------------------------------------------

def aria_keystroke_from_row(row: dict) -> dict:
    """Map an ASSISTments row to a keystroke payload for BehavioralFeatureExtractor.

    ASSISTments has no keystroke capture; time-to-first-response is the closest
    analog of the pre-typing pause (thinking time before the first action), so it
    drives ARIA's absolute timing thresholds (pause>5000 -> planning, >10000 ->
    stuck, <500 -> rushing)."""
    ms = _num(_get(row, "ms_first_response"))
    return {
        "pause_before_first_key_ms": int(ms) if ms > 0 else None,
        "total_typing_time_ms": int(ms) if ms > 0 else None,
        "typing_speed_wpm": None,
        "backspace_rate": None,
    }


# ------------------------------------------------------------------
# Data loading (real CSVs, else synthetic fallback)
# ------------------------------------------------------------------

def load_real_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            rows.extend(dict(r) for r in csv.DictReader(fh))
    return rows


def synthetic_rows(n: int = 1200) -> list[dict]:
    """Deterministic ASSISTments-shaped rows spanning the behaviour space.

    A linear-congruential generator (no wall-clock / randomness) produces a mix
    of fast-guessers, deliberate solvers, hint-heavy strugglers and multi-attempt
    frustration so both proxy methods see a full label distribution."""
    seed = 987654321

    def rnd() -> float:
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF

    archetypes = [
        # (ms_lo, ms_hi, attempts, correct, hint_lo, hint_hi)
        (200, 1500, 1, 0, 0, 0),      # rushing: fast, single attempt, wrong
        (6000, 20000, 1, 1, 0, 1),    # flow/planning: deliberate, correct
        (2000, 9000, 1, 1, 0, 0),     # flow: mid-time, single attempt, correct
        (15000, 45000, 1, 0, 3, 6),   # stuck: long, hint-heavy, wrong
        (3000, 12000, 5, 0, 2, 4),    # frustrated: many attempts, wrong
        (1000, 6000, 2, 1, 0, 1),     # ordinary: couple of attempts, correct
    ]
    rows: list[dict] = []
    for i in range(n):
        a = archetypes[i % len(archetypes)]
        ms = int(a[0] + (a[1] - a[0]) * rnd())
        attempts = a[2] + (1 if rnd() > 0.75 else 0)
        correct = a[3] if rnd() > 0.15 else (1 - a[3])
        hints = int(a[4] + (a[5] - a[4]) * rnd())
        rows.append({
            "ms_first_response": ms,
            "attempt_count": attempts,
            "correct": correct,
            "hint_count": hints,
        })
    return rows


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

def run(rows: list[dict], data_source: str) -> dict:
    stats = compute_timing_stats(rows)
    extractor = BehavioralFeatureExtractor()
    reference, method1_source = _load_reference_classifier()
    pct_tuple = (stats["p10"], stats["p25"], stats["p90"])  # for adapter.classify

    agree = 0
    labeled = 0
    unlabeled = 0
    method1_labels: dict[str, int] = {}
    method2_labels: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}
    # ASSISTments carries only timing (no text / keystroke stream), so ARIA's
    # extractor can only express the timing-driven states. Restrict a second
    # metric to those states for a fair like-for-like threshold comparison.
    timing_states = {"RUSHING", "PLANNING", "STUCK"}
    timing_total = 0
    timing_agree = 0

    for row in rows:
        # METHOD 1 (reference proxy label).
        if reference is not None:
            correct, attempt, hint, rt = _row_fields(row)
            state1 = reference(correct, attempt, hint, rt, pct_tuple)
        else:
            native = assistments_to_behavioral_features(row, stats)
            state1 = SIGNAL_TO_STATE[dominant_signal(native["behavioral_state_signals"])[0]]

        # METHOD 2 (ARIA behavioral extractor label).
        aria_feats = extractor.extract("", aria_keystroke_from_row(row), [])
        state2 = SIGNAL_TO_STATE[dominant_signal(aria_feats["behavioral_state_signals"])[0]]
        method2_labels[state2] = method2_labels.get(state2, 0) + 1

        if state1 is None:  # adapter matched no rule (incl. INSIGHT) — not comparable
            unlabeled += 1
            continue

        labeled += 1
        method1_labels[state1] = method1_labels.get(state1, 0) + 1
        confusion.setdefault(state1, {}).setdefault(state2, 0)
        confusion[state1][state2] += 1
        if state1 == state2:
            agree += 1
        if state1 in timing_states:
            timing_total += 1
            if state1 == state2:
                timing_agree += 1

    agreement_rate = round(agree / labeled, 4) if labeled else 0.0
    timing_rate = round(timing_agree / timing_total, 4) if timing_total else 0.0
    return {
        "data_source": data_source,
        "n": len(rows),
        "method1_source": method1_source,
        "labeled_rows": labeled,
        "unlabeled_rows": unlabeled,
        "timing_percentiles_ms": {k: round(v, 1) for k, v in stats.items() if k != "n"},
        "agreement_rate": agreement_rate,
        "agreements": agree,
        "timing_dimension_agreement_rate": timing_rate,
        "timing_dimension_n": timing_total,
        "method1_reference_label_distribution": method1_labels,
        "method2_aria_extractor_label_distribution": method2_labels,
        "confusion_reference_vs_aria": confusion,
        "note": (
            "ASSISTments has no keystroke/text stream; ARIA's extractor can only "
            "express timing-driven states (RUSHING/PLANNING/STUCK) from "
            "time-to-first-response. Method 1 is the canonical proxy mapping from "
            "datasets/adapters/assistments.py when importable (else the inline "
            "score functions). `agreement_rate` is over labeled rows across all "
            "states; `timing_dimension_agreement_rate` restricts to rows Method 1 "
            "labels as a timing-driven state, for a fair threshold comparison."
        ),
    }


def print_report(results: dict) -> None:
    print("\n" + "=" * 74)
    print("ASSISTMENTS BEHAVIORAL AGREEMENT")
    print("=" * 74)
    print(f"data source : {results['data_source']}   n={results['n']}")
    print(f"method 1 ref: {results['method1_source']}   "
          f"labeled={results['labeled_rows']}  unlabeled={results['unlabeled_rows']}")
    print(f"timing pct  : {results['timing_percentiles_ms']}")
    print(f"\nMethod 1 (reference proxy)   labels: {results['method1_reference_label_distribution']}")
    print(f"Method 2 (ARIA extractor)    labels: {results['method2_aria_extractor_label_distribution']}")
    print(f"\nAGREEMENT RATE (all 5 states): {results['agreement_rate']:.1%}  "
          f"({results['agreements']}/{results['labeled_rows']} labeled rows)")
    print(f"AGREEMENT RATE (timing dimension, RUSHING/PLANNING/STUCK): "
          f"{results['timing_dimension_agreement_rate']:.1%}  "
          f"({results['timing_dimension_n']} rows)")
    rate = results["timing_dimension_agreement_rate"]
    interp = ("high — ARIA's absolute timing thresholds track the ITS "
              "percentile patterns" if rate >= 0.6 else
              "moderate — ARIA's absolute keystroke thresholds and the ITS "
              "percentile labels diverge on the middle of the distribution")
    print(f"interpretation: {interp}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Agreement between ASSISTments-native and ARIA behavioral proxies.")
    ap.add_argument("--synthetic-n", type=int, default=1200,
                    help="Rows to synthesise when no ASSISTments CSVs are present.")
    ap.add_argument("--output", default=RESULTS_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    real = load_real_rows()
    if real:
        print(f"Loaded {len(real)} ASSISTments rows from {RAW_DIR}")
        rows, source = real, "assistments_raw"
    else:
        print(f"No ASSISTments CSVs in {RAW_DIR} — using deterministic synthetic "
              f"fallback ({args.synthetic_n} rows).")
        rows, source = synthetic_rows(args.synthetic_n), "synthetic_fallback"

    results = run(rows, source)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print_report(results)
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
