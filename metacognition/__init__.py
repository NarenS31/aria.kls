"""
ARIA metacognition engine.

Text + audio think-aloud analysis, metacognitive intervention generation, and
longitudinal cognitive-state tracking.

Public API:
    CognitiveStateAnalyzer          — detect cognitive state from think-aloud
    evaluate                        — score the analyzer on a labelled dataset
    MetacognitiveInterventionGenerator — Socratic questions per state
    MetacognitionTracker            — per-session + longitudinal tracking
    ThinkAloudListener              — optional mic recording + transcription
    COGNITIVE_STATES                — the seven canonical states

Metacognitive development measurement (are students learning the skill, not
just responding to prompts?):
    TransferDetector                — self-initiated vs prompted metacognition
    CalibrationTracker              — does the student know what they know?
    InterventionTimer               — the optimal moment to intervene per state
"""

from .analyzer import (
    CognitiveStateAnalyzer,
    COGNITIVE_STATES,
    evaluate,
    compute_classification_metrics,
    print_confusion_matrix,
)
from .interventions import (
    MetacognitiveInterventionGenerator,
    INTERVENTION_BANK,
    STATE_RANK,
)
from .tracker import MetacognitionTracker
from .transfer import (
    TransferDetector,
    detect_self_initiation,
    classify_metacognition,
    is_metacognitive_prompt,
    TRANSFER_THRESHOLD,
)
from .calibration import (
    CalibrationTracker,
    normalise_confidence,
    CONFIDENCE_PROMPT,
)
from .timing import (
    InterventionTimer,
    NEGATIVE_STATES,
    DEFAULT_OPTIMAL,
    adhd_subtype,
)

# ThinkAloudListener depends on optional audio libraries; import defensively so
# that the text pipeline works even when sounddevice/whisper are absent.
try:
    from .listener import ThinkAloudListener, AudioFeatures
except Exception:  # pragma: no cover - only if module itself is broken
    ThinkAloudListener = None  # type: ignore
    AudioFeatures = None       # type: ignore

__all__ = [
    "CognitiveStateAnalyzer",
    "COGNITIVE_STATES",
    "evaluate",
    "compute_classification_metrics",
    "print_confusion_matrix",
    "MetacognitiveInterventionGenerator",
    "INTERVENTION_BANK",
    "STATE_RANK",
    "MetacognitionTracker",
    "ThinkAloudListener",
    "AudioFeatures",
    # Metacognitive development measurement
    "TransferDetector",
    "detect_self_initiation",
    "classify_metacognition",
    "is_metacognitive_prompt",
    "TRANSFER_THRESHOLD",
    "CalibrationTracker",
    "normalise_confidence",
    "CONFIDENCE_PROMPT",
    "InterventionTimer",
    "NEGATIVE_STATES",
    "DEFAULT_OPTIMAL",
    "adhd_subtype",
]
