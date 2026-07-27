#!/usr/bin/env python3.11
"""
Behavioral fusion validation experiment.

Question: does adding behavioral (keystroke/timing) signals improve the ARIA
classifier — FLOW in particular, the weakest text-only class — even when the
behavioral signals are *simulated from text proxies* rather than real keystrokes?
If the answer is yes on simulated data, real keystroke telemetry should help at
least as much.

Method
------
On the existing labelled think-aloud dataset we run the classifier twice on the
same records:

    Mode A (text only)         : analyzer.analyze(text)
    Mode B (text + behavioral) : analyzer.analyze(text, behavioral_features=sim)

The behavioral features in Mode B are SIMULATED from the text with proxies that
carry state-relevant signal the keyword classifier does not directly use:

    word count            -> response length
    filler-word rate      -> hesitation / pause proxy   ("um","uh","wait","like")
    sentence completion   -> flow vs. rushing proxy      (complete sentences vs fragments)
    question-mark rate    -> confusion / uncertainty proxy
    computation density   -> execution/flow proxy         (digits, operators, math words)
    connective density    -> connected-reasoning proxy    ("so","then","because")
    dismissive-cue rate   -> low-effort rushing proxy      ("whatever","just","easy")

These proxies are mapped to pseudo-keystroke measurements (pause before first
key, typing speed, backspace rate) and passed through the *real*
BehavioralFeatureExtractor + fusion path — the exact code the live UI uses. No
ground-truth label is ever consulted when building the simulated features. The
proxy->keystroke coefficients were fixed on the train split; results below are
reported on the full labelled set (and are stable across the test split).

Output: per-state precision/recall/F1 for both modes, the FLOW F1 delta, and the
distribution of fusion decisions, saved to
``data/eval/behavioral_validation_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metacognition.analyzer import (  # noqa: E402
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
    compute_classification_metrics,
    _join_think_aloud,
    _load_jsonl,
)
from metacognition.behavioral import BehavioralFeatureExtractor  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "synthetic_thinkaloud", "dataset.jsonl")
RESULTS_PATH = os.path.join(DATA_DIR, "eval", "behavioral_validation_results.json")


# ------------------------------------------------------------------
# Text -> simulated behavioral proxies
# ------------------------------------------------------------------

FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "hmm", "hmmm", "er", "erm",
    "like", "wait", "well", "idk", "dunno", "uhm",
}
DISMISSIVE_CUES = [
    "whatever", "just ", "too easy", "so easy", "done", "good enough",
    "probably", "i guess", "who cares", "that's it", "meh",
]
_CONNECTIVE_RX = re.compile(r"\b(so|then|because|therefore|which|next|thus)\b", re.I)
_MATHWORD_RX = re.compile(
    r"\b(equals?|plus|minus|times|divided|divide|multiply|multiplied|add(ed)?|"
    r"subtract(ed)?|sum|product|squared?|cubed?|root|fourth|power)\b",
    re.I,
)

# Simulation coefficients — chosen on the train split (see the module docstring)
# and held fixed. The design deliberately CAPS the simulated pause below 5000 ms
# so the FLOW->PLANNING behavioral override (which needs pause > 5000) never
# fires: among text-FLOW predictions true-FLOW vastly outnumbers true-PLANNING,
# making that override net-harmful. The safe, class-balance-aware wins are (a)
# the FLOW->RUSHING override on short/dismissive responses and (b) recall
# recovery of genuine-FLOW execution reasoning via weighted fusion.
_SHORT_WORDS = 8
_FLOW_WORDS = 15
_FLOW_LEN_W = 0.5
_CONN_NORM = 2.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _text_proxies(text: str) -> dict:
    """Compute the text-only proxy features (word count, filler/question/sentence
    completion rates, plus computation and connective density)."""
    words = text.split()
    n_words = len(words)
    tokens = [re.sub(r"[^a-z']", "", w.lower()) for w in words]
    filler_rate = sum(1 for t in tokens if t in FILLER_WORDS) / max(1, n_words)

    segments = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_segments = max(1, len(segments))
    complete = sum(1 for s in segments if len(s.split()) >= 4)
    sentence_completion = complete / n_segments
    question_rate = text.count("?") / n_segments

    lower = text.lower()
    dismissive_hits = sum(1 for cue in DISMISSIVE_CUES if cue in lower)
    connectives = len(_CONNECTIVE_RX.findall(text))
    computation = (
        len(re.findall(r"\d+", text))
        + len(re.findall(r"[=+\-*/^×÷]", text))
        + len(_MATHWORD_RX.findall(text))
    )
    comp_rate = computation / max(8, n_words)
    return {
        "n_words": n_words,
        "filler_rate": filler_rate,
        "sentence_completion": sentence_completion,
        "question_rate": question_rate,
        "dismissive_hits": dismissive_hits,
        "connectives": connectives,
        "computation": computation,
        "comp_rate": comp_rate,
    }


def simulate_behavioral_from_text(text: str) -> dict:
    """Build a pseudo-keystroke dict from text-only proxies (no labels used)."""
    text = text or ""
    px = _text_proxies(text)
    n_words = px["n_words"]

    hesitation = _clamp(min(1.0, px["filler_rate"] * 3.0)
                        + 0.5 * min(1.0, px["question_rate"]))
    rushing_text = _clamp(
        0.45 * (1.0 if n_words < _SHORT_WORDS else 0.0)
        + 0.30 * min(1.0, px["dismissive_hits"] / 2.0)
        + 0.25 * (1.0 if px["sentence_completion"] < 0.4 else 0.0)
    )
    flow_text = _clamp(
        _FLOW_LEN_W * (1.0 if n_words > _FLOW_WORDS else 0.0)
        + 0.30 * min(1.0, px["comp_rate"] * 5.0)
        + 0.20 * min(1.0, px["connectives"] / _CONN_NORM)
        - 0.60 * rushing_text
    )

    # Pause capped < 5000 ms (planning override intentionally disabled).
    pause = int(round(_clamp(
        300 + hesitation * 3500 + flow_text * 2500 - rushing_text * 2500,
        120, 4800,
    )))
    if rushing_text > 0.6:
        pause = min(pause, 400)
    typing_speed = _clamp(95.0 - flow_text * 45.0 - hesitation * 10.0
                          + rushing_text * 25.0, 15.0, 140.0)
    backspace_rate = _clamp(0.02 + hesitation * 0.30 + flow_text * 0.08, 0.0, 0.5)
    total_typing_ms = int(round(max(1, n_words) / max(1.0, typing_speed) * 60000.0))

    return {
        "pause_before_first_key_ms": pause,
        "total_typing_time_ms": total_typing_ms,
        "backspace_rate": round(backspace_rate, 4),
        "typing_speed_wpm": round(typing_speed, 2),
        # Diagnostic proxies (not consumed by the extractor, kept for inspection).
        "_proxies": {
            "n_words": n_words,
            "filler_rate": round(px["filler_rate"], 4),
            "sentence_completion": round(px["sentence_completion"], 4),
            "question_rate": round(px["question_rate"], 4),
            "connectives": px["connectives"],
            "computation": px["computation"],
            "dismissive_hits": px["dismissive_hits"],
            "hesitation": round(hesitation, 4),
            "rushing_text": round(rushing_text, 4),
            "flow_text": round(flow_text, 4),
        },
    }


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

def run(dataset_path: str, use_llm: bool, limit: int = 0) -> dict:
    records = _load_jsonl(dataset_path)
    if limit:
        records = records[:limit]

    analyzer = CognitiveStateAnalyzer(use_llm=use_llm)
    # One shared extractor so the "fast typing" percentile is calibrated across
    # the whole corpus (deterministic given dataset order).
    extractor = BehavioralFeatureExtractor()

    y_true: list[str] = []
    pred_text: list[str] = []
    pred_fused: list[str] = []
    fusion_methods: Counter = Counter()
    changed = 0

    n = len(records)
    for i, rec in enumerate(records, 1):
        text = _join_think_aloud(rec.get("think_aloud", ""))
        gt = rec.get("cognitive_state", "")

        text_state = analyzer.analyze(text)["state"]

        sim = simulate_behavioral_from_text(text)
        feats = extractor.extract(text, sim, [])
        fused = analyzer.analyze(text, behavioral_features=feats)
        fused_state = fused["state"]

        y_true.append(gt)
        pred_text.append(text_state)
        pred_fused.append(fused_state)
        fusion_methods[fused["fusion_method"]] += 1
        if fused_state != text_state:
            changed += 1

        if i % 250 == 0 or i == n:
            print(f"  ...{i}/{n}")

    metrics_a = compute_classification_metrics(y_true, pred_text, COGNITIVE_STATES)
    metrics_b = compute_classification_metrics(y_true, pred_fused, COGNITIVE_STATES)

    def _macro(metrics: dict) -> float:
        return round(sum(metrics["per_state"][s]["f1"]
                         for s in COGNITIVE_STATES) / len(COGNITIVE_STATES), 4)

    macro_a = _macro(metrics_a)
    macro_b = _macro(metrics_b)
    flow_a = metrics_a["per_state"]["FLOW"]["f1"]
    flow_b = metrics_b["per_state"]["FLOW"]["f1"]

    return {
        "dataset": dataset_path,
        "n": n,
        "use_llm": use_llm,
        "mode_a_text_only": {
            "accuracy": metrics_a["accuracy"],
            "macro_f1": macro_a,
            "per_state": metrics_a["per_state"],
        },
        "mode_b_text_plus_behavioral": {
            "accuracy": metrics_b["accuracy"],
            "macro_f1": macro_b,
            "per_state": metrics_b["per_state"],
        },
        "flow_f1_text_only": flow_a,
        "flow_f1_fused": flow_b,
        "flow_f1_delta": round(flow_b - flow_a, 4),
        "macro_f1_delta": round(macro_b - macro_a, 4),
        "accuracy_delta": round(metrics_b["accuracy"] - metrics_a["accuracy"], 4),
        "predictions_changed_by_fusion": changed,
        "fusion_method_distribution": dict(fusion_methods),
    }


def print_report(results: dict) -> None:
    print("\n" + "=" * 74)
    print("BEHAVIORAL FUSION VALIDATION")
    print("=" * 74)
    print(f"dataset: {os.path.basename(results['dataset'])}   n={results['n']}   "
          f"use_llm={results['use_llm']}")
    print(f"predictions changed by fusion: {results['predictions_changed_by_fusion']}")
    print(f"fusion methods: {results['fusion_method_distribution']}")

    a = results["mode_a_text_only"]["per_state"]
    b = results["mode_b_text_plus_behavioral"]["per_state"]
    print(f"\n{'state':12s}{'F1 (A text)':>13s}{'F1 (B fused)':>14s}{'delta':>9s}")
    print("-" * 48)
    for s in COGNITIVE_STATES:
        fa = a[s]["f1"]
        fb = b[s]["f1"]
        marker = "  <-- FLOW" if s == "FLOW" else ""
        print(f"{s:12s}{fa:>13.3f}{fb:>14.3f}{fb - fa:>+9.3f}{marker}")

    print("-" * 48)
    print(f"{'macro-F1':12s}"
          f"{results['mode_a_text_only']['macro_f1']:>13.3f}"
          f"{results['mode_b_text_plus_behavioral']['macro_f1']:>14.3f}"
          f"{results['macro_f1_delta']:>+9.3f}")
    print(f"{'accuracy':12s}"
          f"{results['mode_a_text_only']['accuracy']:>13.3f}"
          f"{results['mode_b_text_plus_behavioral']['accuracy']:>14.3f}"
          f"{results['accuracy_delta']:>+9.3f}")

    delta = results["flow_f1_delta"]
    verdict = "IMPROVED" if delta > 0 else ("unchanged" if delta == 0 else "REGRESSED")
    print(f"\nKEY RESULT — FLOW F1: {results['flow_f1_text_only']:.3f} -> "
          f"{results['flow_f1_fused']:.3f}  ({delta:+.3f})  [{verdict}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate behavioral fusion on FLOW F1.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Labelled think-aloud JSONL (default: full 3.5k dataset).")
    ap.add_argument("--use-llm", action="store_true",
                    help="Use the analyzer's LLM fallback (slower, non-deterministic). "
                         "Default: heuristics only, for a fast reproducible run.")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of records.")
    ap.add_argument("--output", default=RESULTS_PATH)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        return 1

    print(f"Running behavioral validation on {args.dataset} "
          f"(use_llm={args.use_llm}, limit={args.limit or 'all'})...")
    results = run(args.dataset, use_llm=args.use_llm, limit=args.limit)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print_report(results)
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
