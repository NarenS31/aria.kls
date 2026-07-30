"""Fast conversational-intent layer for student learning turns.

The intent layer separates dialogue management from mathematical reasoning.
It is intentionally deterministic and inspectable; problem correctness remains
the responsibility of the answer-keyed problem model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_MODEL = None
_MODEL_ATTEMPTED = False
_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "student_intent.joblib"


@dataclass(frozen=True)
class StudentIntent:
    label: str
    confidence: float
    evidence: str
    contains_reasoning: bool


_HELP_PATTERNS = (
    r"\bi need help\b",
    r"\bhelp me\b",
    r"\bcan you help\b",
    r"\bwhere do i (?:start|begin)\b",
    r"\bhow do i (?:start|begin)\b",
    r"\bno clue\b",
    r"\bi(?:'m| am) stuck\b",
    r"\bidk\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bdont know\b",
)
_META_PATTERNS = (
    r"\bthis is my first (?:attempt|try|time)\b",
    r"\bi just started\b",
    r"\bi haven't tried yet\b",
    r"\bi havent tried yet\b",
    r"\bfirst attempt\b",
)
_UNCERTAINTY_PATTERNS = (
    r"\bi(?:'m| am) not sure\b",
    r"\bnot sure\b",
    r"\bmaybe\b",
    r"\bi guess\b",
    r"\bdoes that make sense\b",
)
_CORRECTION_PATTERNS = (
    r"\bwait\b",
    r"\bactually\b",
    r"\bi see\b",
    r"\binstead\b",
    r"\bthat means\b",
)
_FRUSTRATION_PATTERNS = (
    r"\bi hate (?:this|math)\b", r"\bthis is (?:stupid|annoying|impossible)\b",
    r"\bi can'?t do this\b", r"\bi give up\b", r"\bthis makes no sense\b",
)
_CLARIFICATION_PATTERNS = (
    r"\bwhat does .+ mean\b", r"\bcan you explain\b",
    r"\bwhat do you mean\b", r"\bsay that again\b",
)
_CONFIRMATION_PATTERNS = (
    r"\bis (?:that|this|it) (?:right|correct)\b", r"\bam i right\b",
    r"\bdid i (?:do|get) .+ right\b", r"\bdoes that look right\b",
)
_CONTROL_PATTERNS = (
    r"\bnew problem\b", r"\banother (?:problem|one|question)\b",
    r"\bmake it (?:easier|harder)\b", r"\bskip (?:this|it)\b",
    r"\bchange (?:the )?subject\b",
)
_REASONING_MARKERS = (
    "because", "so ", "then", "equals", "=", "subtract", "add", "multiply",
    "divide", "distribute", "claim", "evidence", "quote", "therefore",
)
_SOCIAL_ONLY = re.compile(
    r"^(?:hi|hello|hey|thanks?|thank you|thx|ok(?:ay)?|bye|goodbye|fine|"
    r"yes|no|yep|nope|cool|great)[!?.\s😀👍]*$",
    re.IGNORECASE,
)


def _load_model():
    global _MODEL, _MODEL_ATTEMPTED
    if _MODEL_ATTEMPTED:
        return _MODEL
    _MODEL_ATTEMPTED = True
    try:
        import joblib
        if _MODEL_PATH.exists():
            _MODEL = joblib.load(_MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def warm_student_intent_model() -> bool:
    """Load the intent artifact during app startup instead of on the first turn."""
    return _load_model() is not None


def _rule_fallback(text: str) -> StudentIntent:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    contains_reasoning = any(marker in normalized for marker in _REASONING_MARKERS)

    if any(re.search(pattern, normalized) for pattern in _HELP_PATTERNS):
        return StudentIntent(
            "HELP_REQUEST",
            0.98,
            "the student explicitly requested help or said they could not begin",
            contains_reasoning,
        )
    for label, patterns, evidence in (
        ("CONTROL_REQUEST", _CONTROL_PATTERNS, "the student requested a change to the activity"),
        ("FRUSTRATION", _FRUSTRATION_PATTERNS, "the student expressed frustration"),
        ("CLARIFICATION_REQUEST", _CLARIFICATION_PATTERNS, "the student requested an explanation or definition"),
        ("CONFIRMATION_REQUEST", _CONFIRMATION_PATTERNS, "the student asked whether their work was correct"),
    ):
        if any(re.search(pattern, normalized) for pattern in patterns):
            return StudentIntent(label, 0.9, evidence, contains_reasoning)
    if any(re.search(pattern, normalized) for pattern in _META_PATTERNS):
        return StudentIntent(
            "ATTEMPT_META",
            0.98,
            "the student described the attempt itself, not a solution plan",
            contains_reasoning,
        )
    if any(re.search(pattern, normalized) for pattern in _CORRECTION_PATTERNS):
        return StudentIntent(
            "SELF_CORRECTION",
            0.85,
            "the student signaled a revision to their reasoning",
            contains_reasoning,
        )
    if any(re.search(pattern, normalized) for pattern in _UNCERTAINTY_PATTERNS):
        return StudentIntent(
            "UNCERTAINTY",
            0.85,
            "the student expressed uncertainty",
            contains_reasoning,
        )
    if _SOCIAL_ONLY.fullmatch(normalized):
        return StudentIntent(
            "SOCIAL",
            0.9,
            "the turn is conversational acknowledgement rather than problem reasoning",
            False,
        )
    if contains_reasoning:
        return StudentIntent(
            "REASONING",
            0.75,
            "the turn contains an operation, justification, or conclusion",
            True,
        )
    return StudentIntent(
        "OTHER",
        0.5,
        "the turn does not contain enough evidence for a specific intent",
        False,
    )


def classify_student_intent(text: str) -> StudentIntent:
    """Classify natural student talk with the trained Eedi-derived model.

    The transparent rule classifier is retained only when the model artifact is
    unavailable or its posterior is too uncertain.
    """
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    contains_reasoning = any(marker in normalized for marker in _REASONING_MARKERS)
    model = _load_model()
    if model is not None and normalized:
        try:
            probabilities = model.predict_proba([normalized])[0]
            classes = list(model.classes_)
            best_index = max(range(len(classes)), key=lambda index: probabilities[index])
            label = str(classes[best_index])
            confidence = float(probabilities[best_index])
            if confidence >= 0.52:
                return StudentIntent(
                    label,
                    confidence,
                    (
                        f"the dialogue model classified this as "
                        f"{label.lower().replace('_', ' ')}"
                    ),
                    contains_reasoning,
                )
        except Exception:
            pass
    return _rule_fallback(text)
