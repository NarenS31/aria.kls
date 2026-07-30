#!/usr/bin/env python3.11
"""Evaluate the trained intent model on a human-written challenge set."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.student_intent import classify_student_intent  # noqa: E402
from agent.student_understanding import understand_student_turn  # noqa: E402


DATA = ROOT / "eval" / "data" / "student_language_challenge.json"
OUTPUT = ROOT / "eval" / "data" / "eval" / "student_intent_challenge_results.json"


def evaluate(mode: str) -> dict:
    cases = json.loads(DATA.read_text())
    expected, predicted, confidences, rows = [], [], [], []
    started = time.perf_counter()
    for case in cases:
        if mode == "hybrid":
            understanding = understand_student_turn(case["text"], allow_deep=False)
            label = understanding.intent
            confidence = understanding.confidence
            source = understanding.source
        else:
            result = classify_student_intent(case["text"])
            label = result.label
            confidence = result.confidence
            source = "fast_intent_model"
        expected.append(case["label"])
        predicted.append(label)
        confidences.append(confidence)
        rows.append({
            **case,
            "prediction": label,
            "confidence": round(confidence, 4),
            "source": source,
            "correct": label == case["label"],
        })
    elapsed = time.perf_counter() - started
    labels = sorted(set(expected) | set(predicted))
    report = classification_report(
        expected, predicted, labels=labels, output_dict=True, zero_division=0
    )
    bins = defaultdict(lambda: {"n": 0, "confidence": 0.0, "correct": 0})
    for confidence, is_correct in zip(
        confidences, [a == b for a, b in zip(expected, predicted)]
    ):
        lower = min(9, int(confidence * 10))
        bucket = bins[f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}"]
        bucket["n"] += 1
        bucket["confidence"] += confidence
        bucket["correct"] += int(is_correct)
    calibration = {}
    ece = 0.0
    for name, bucket in bins.items():
        mean_conf = bucket["confidence"] / bucket["n"]
        accuracy = bucket["correct"] / bucket["n"]
        calibration[name] = {
            "n": bucket["n"],
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(accuracy, 4),
        }
        ece += bucket["n"] / len(cases) * abs(mean_conf - accuracy)
    return {
        "benchmark": "human-written adversarial student-language challenge",
        "mode": mode,
        "n": len(cases),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "mean_latency_ms": elapsed / len(cases) * 1000,
        "expected_calibration_error": ece,
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=labels
        ).tolist(),
        "calibration": calibration,
        "failures": [row for row in rows if not row["correct"]],
        "rows": rows,
    }


def main() -> None:
    fast_output = evaluate("fast")
    hybrid_output = evaluate("hybrid")
    output = {
        "benchmark": fast_output["benchmark"],
        "fast": fast_output,
        "hybrid": hybrid_output,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "fast": {
            "n": fast_output["n"],
            "accuracy": fast_output["accuracy"],
            "macro_f1": fast_output["macro_f1"],
            "mean_latency_ms": fast_output["mean_latency_ms"],
        },
        "hybrid": {
            "n": hybrid_output["n"],
            "accuracy": hybrid_output["accuracy"],
            "macro_f1": hybrid_output["macro_f1"],
            "mean_latency_ms": hybrid_output["mean_latency_ms"],
            "failures": hybrid_output["failures"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
