"""
ARIA metacognition engine.

Text + audio think-aloud analysis, metacognitive intervention generation, and
longitudinal cognitive-state tracking.

Public API:
    CognitiveStateAnalyzer          - detect cognitive state from think-aloud
    evaluate                        - score the analyzer on a labelled dataset
    MetacognitiveInterventionGenerator - Socratic questions per state
    MetacognitionTracker            - per-session + longitudinal tracking
    InteractionLogger               - SQLite logs for interactions/outcomes
    RuleBasedJITAIPolicy            - interpretable JITAI decision policy
    ThinkAloudListener              - optional mic recording + transcription
    COGNITIVE_STATES                - the seven canonical states

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
from .behavioral import (
    BehavioralFeatureExtractor,
    BEHAVIORAL_SIGNALS,
    SIGNAL_TO_STATE,
    dominant_signal,
)
from .interventions import (
    MetacognitiveInterventionGenerator,
    INTERVENTION_BANK,
    INTERVENTION_CITATIONS,
    STATE_RANK,
    extract_concept,
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
    TIMING_RULES,
    adhd_subtype,
)
from .interaction_logger import (
    InteractionLogger,
    InteractionEvent,
    InterventionEvent,
    InterventionOutcome,
    tone_proxy,
    summarize_pauses,
)
from .jitai import (
    JITAIContext,
    JITAIDecision,
    JITAIComponents,
    RuleBasedJITAIPolicy,
    build_jitai_intervention,
    context_from_signals,
    topic_stats_from_learning_graph,
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
    "BehavioralFeatureExtractor",
    "BEHAVIORAL_SIGNALS",
    "SIGNAL_TO_STATE",
    "dominant_signal",
    "MetacognitiveInterventionGenerator",
    "INTERVENTION_BANK",
    "INTERVENTION_CITATIONS",
    "STATE_RANK",
    "extract_concept",
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
    "TIMING_RULES",
    "adhd_subtype",
    "InteractionLogger",
    "InteractionEvent",
    "InterventionEvent",
    "InterventionOutcome",
    "tone_proxy",
    "summarize_pauses",
    "JITAIContext",
    "JITAIDecision",
    "JITAIComponents",
    "RuleBasedJITAIPolicy",
    "build_jitai_intervention",
    "context_from_signals",
    "topic_stats_from_learning_graph",
]
