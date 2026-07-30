#!/usr/bin/env python3.11
"""Train ARIA's low-latency student-intent classifier on real dialogue.

Source: Eedi Question-Anchored-Tutoring-Dialogues-2k, CC BY-NC-SA 4.0.
Labels are high-precision weak supervision and must not be reported as human
ground truth. Character n-grams help the model generalize across spelling,
punctuation, abbreviations, and short conversational forms.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "eval" / "data" / "external" / "eedi_dialogues" / "raw"
MODEL = ROOT / "models" / "student_intent.joblib"
METADATA = ROOT / "models" / "student_intent_metadata.json"

PATTERNS = {
    "HELP_REQUEST": (
        r"\bhelp\b", r"\bstuck\b", r"\bno clue\b", r"\bdon'?t know\b",
        r"\bdont know\b", r"\bidk\b", r"\bwhere do i start\b",
        r"\bhow do i start\b", r"\bconfused\b",
    ),
    "ATTEMPT_META": (
        r"\bfirst (?:attempt|try|time)\b", r"\bjust started\b",
        r"\bhaven'?t tried\b", r"\bnot tried yet\b", r"\bmy attempt\b",
    ),
    "SELF_CORRECTION": (
        r"\bsorry i meant\b", r"\bi meant\b", r"\bwait\b", r"\bactually\b",
        r"\binstead\b", r"\bi see\b", r"\bmy mistake\b",
    ),
    "FRUSTRATION": (
        r"\bi hate (?:this|math)\b", r"\bthis is (?:stupid|annoying|impossible)\b",
        r"\bi can'?t do this\b", r"\bi give up\b", r"\bso frustrating\b",
        r"\bthis makes no sense\b", r"\bim done\b",
    ),
    "CLARIFICATION_REQUEST": (
        r"\bwhat does .+ mean\b", r"\bwhat is (?:a|an|the) .+\b",
        r"\bcan you explain\b", r"\bexplain that\b", r"\bsay that again\b",
        r"\bwhat do you mean\b", r"\bwhy (?:do|does|is|are|would)\b",
    ),
    "CONFIRMATION_REQUEST": (
        r"\bis (?:that|this|it) (?:right|correct)\b", r"\bam i right\b",
        r"\bdid i (?:do|get) .+ right\b", r"\bcorrect\??$",
        r"\bdoes that look right\b",
    ),
    "CONTROL_REQUEST": (
        r"\bnew problem\b", r"\banother (?:problem|one|question)\b",
        r"\bmake it (?:easier|harder)\b", r"\bskip (?:this|it)\b",
        r"\bchange (?:the )?subject\b", r"\bgo back\b",
    ),
    "UNCERTAINTY": (
        r"\bnot sure\b", r"\bi think\b", r"\bi guess\b", r"\bmaybe\b",
        r"\bprobably\b",
    ),
}
SOCIAL = re.compile(
    r"^(?:hi|hello|hey|thanks?|thank you|thx|ok(?:ay)?|bye|goodbye|fine|"
    r"yes|no|yep|nope|cool|great)[!?.\s😀👍]*$",
    re.IGNORECASE,
)
SHORT_ANSWER = re.compile(
    r"^(?:[a-d]|true|false|yes|no|[-+]?\d+(?:[.,/]\d+)?(?:x)?)[!?.\s]*$",
    re.IGNORECASE,
)
REASONING_MARKERS = (
    "because", "so ", "then", "equals", "=", "subtract", "add", "multiply",
    "divide", "times", "distribut", "factor", "claim", "evidence", "therefore",
    "answer is", "i got", "would be",
)


def weak_label(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return None
    # Conversation management takes precedence over words like "first" that
    # can otherwise be mistaken for mathematical planning.
    for label in (
        "CONTROL_REQUEST", "ATTEMPT_META", "FRUSTRATION",
        "CLARIFICATION_REQUEST", "CONFIRMATION_REQUEST", "HELP_REQUEST",
        "SELF_CORRECTION", "UNCERTAINTY",
    ):
        if any(re.search(pattern, normalized) for pattern in PATTERNS[label]):
            return label
    if SOCIAL.fullmatch(normalized):
        return "SOCIAL"
    if SHORT_ANSWER.fullmatch(normalized):
        return "SHORT_ANSWER"
    if any(marker in normalized for marker in REASONING_MARKERS):
        return "REASONING"
    if 4 <= len(normalized.split()) <= 60:
        return "OTHER"
    return None


def load_student_turns(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            row["MessageString"].strip()
            for row in rows
            if str(row.get("IsTutor", "")).strip() in {"0", "False", "false"}
            and row.get("MessageString", "").strip()
        ]


def balanced_examples(texts: list[str], cap: int, seed: int) -> tuple[list[str], list[str]]:
    groups = defaultdict(list)
    for text in texts:
        label = weak_label(text)
        if label:
            groups[label].append(text)
    # Add explicit conversational forms that matter in the product but are rare
    # in the source sample. They are evaluation-critical, not synthetic claims.
    seeds = {
        "ATTEMPT_META": [
            "this is my first attempt", "this is my first attemot",
            "i havent even tried yet", "i just opened the problem",
            "this is the first time im trying it",
        ],
        "HELP_REQUEST": [
            "i need help", "help idk where to start", "can u help me",
            "bro i dont get this", "what am i supposed to do",
        ],
        "FRUSTRATION": [
            "bro this makes no sense", "this is so annoying",
            "i cant do ts", "im done with this", "nah this is impossible",
        ],
        "CLARIFICATION_REQUEST": [
            "what does coefficient mean", "wait what do you mean by inverse",
            "can you explain that part", "why would i distribute",
            "say that in normal words",
        ],
        "CONFIRMATION_REQUEST": [
            "is this right", "did i get that right", "so thats correct?",
            "does that look right", "am i doing it right",
        ],
        "CONTROL_REQUEST": [
            "give me another problem", "make this easier", "skip this one",
            "new question please", "can we do english instead",
        ],
    }
    for label, items in seeds.items():
        groups[label].extend(items * 30)
    rng = random.Random(seed)
    selected_x, selected_y = [], []
    for label, items in sorted(groups.items()):
        rng.shuffle(items)
        for text in items[:cap]:
            selected_x.append(text)
            selected_y.append(label)
    order = list(range(len(selected_x)))
    rng.shuffle(order)
    return [selected_x[i] for i in order], [selected_y[i] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    train_raw = load_student_turns(RAW / "train.csv")
    test_raw = load_student_turns(RAW / "test.csv")
    train_x, train_y = balanced_examples(train_raw, args.cap_per_class, args.seed)
    test_x, test_y = balanced_examples(test_raw, 700, args.seed + 1)

    model = Pipeline([
        ("features", FeatureUnion([
            ("chars", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                max_features=65_000, sublinear_tf=True,
            )),
            ("words", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=35_000, sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            max_iter=700,
            class_weight="balanced",
            C=4.0,
            random_state=args.seed,
        )),
    ])
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - started
    predictions = model.predict(test_x)
    report = classification_report(
        test_y, predictions, output_dict=True, zero_division=0
    )
    labels = sorted(set(train_y))
    latency_inputs = test_x[:1000]
    latency_started = time.perf_counter()
    model.predict_proba(latency_inputs)
    latency_ms = (
        (time.perf_counter() - latency_started)
        / max(len(latency_inputs), 1)
        * 1000
    )

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    metadata = {
        "schema_version": 1,
        "source": "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
        "source_url": "https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k",
        "license": "CC BY-NC-SA 4.0",
        "commercial_use": False,
        "label_provenance": "weak supervision; not human ground truth",
        "train_student_turns_available": len(train_raw),
        "test_student_turns_available": len(test_raw),
        "train_examples_used": len(train_x),
        "test_examples_used": len(test_x),
        "train_class_counts": dict(Counter(train_y)),
        "test_class_counts": dict(Counter(test_y)),
        "weak_label_test_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(
            test_y, predictions, labels=labels
        ).tolist(),
        "training_seconds": round(training_seconds, 3),
        "mean_inference_ms_per_utterance": round(latency_ms, 4),
        "seed": args.seed,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "model": str(MODEL),
        "train_examples": len(train_x),
        "test_examples": len(test_x),
        "macro_f1_weak_labels": report["macro avg"]["f1-score"],
        "latency_ms": latency_ms,
        "class_counts": dict(Counter(train_y)),
    }, indent=2))


if __name__ == "__main__":
    main()
