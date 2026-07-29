"""
Rule-based JITAI policy for ARIA metacognitive interventions.

This is intentionally an interpretable rule engine, not a trained model. The
policy can be replaced later by a calibrated learner after real local logs
exist, but the default path must remain transparent and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


INTERVENTION_TYPES = {
    "hint",
    "encouragement",
    "break_suggestion",
    "task_chunking",
    "reflection",
    "none",
}

NEGATIVE_STATES = {"CONFUSED", "STUCK", "FRUSTRATED", "RUSHING"}
POSITIVE_STATES = {"FLOW", "INSIGHT"}

RULE_CITATIONS = {
    "jitai.insight_transfer": ["zimmerman_2002", "hattie_timperley_2007"],
    "jitai.no_interrupt_positive": ["zimmerman_2002"],
    "jitai.escalate_after_ignored": ["barkley_2015"],
    "jitai.frustration_reset": ["shaw_2014", "sweller_1988"],
    "jitai.repeated_confusion": ["sweller_1988"],
    "jitai.rushing_slowdown": ["barkley_2015", "sweller_1988"],
    "jitai.inactivity_after_struggle": ["barkley_2015"],
    "jitai.high_struggle_topic": ["black_wiliam_1998", "sweller_1988"],
    "jitai.initial_negative_support": ["hattie_timperley_2007"],
    "jitai.no_action": ["zimmerman_2002"],
}


@dataclass
class JITAIContext:
    profile_id: str
    topic: str
    cognitive_state: str
    consecutive_state_count: int = 1
    response_latency_ms: int | None = None
    time_since_previous_ms: int | None = None
    pause_summary: dict[str, Any] = field(default_factory=dict)
    tone_proxy_label: str = "neutral"
    tone_proxy_score: float = 0.0
    ignored_intervention_count: int = 0
    topic_stats: dict[str, Any] = field(default_factory=dict)
    allow_positive_reflection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JITAIComponents:
    tailoring_variable: dict[str, Any]
    decision_point: str
    decision_rule: str
    intervention_option: str
    proximal_outcome: str
    distal_outcome: str


@dataclass
class JITAIDecision:
    should_intervene: bool
    intervention_type: str
    trigger_condition: str
    rule_id: str
    rationale: str
    citation_key: str = ""
    citation_keys: list[str] = field(default_factory=list)
    components: JITAIComponents | None = None
    extension_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.components is not None:
            data["components"] = asdict(self.components)
        return data


class PolicyProvider(Protocol):
    """Extension point for future policies after real logged data exists."""

    def decide(self, context: JITAIContext) -> JITAIDecision:
        ...


class RuleBasedJITAIPolicy:
    """Deterministic JITAI policy using transparent thresholds."""

    VERY_SHORT_LATENCY_MS = 4_000
    INACTIVITY_THRESHOLD_MS = 180_000
    LONG_PAUSE_MS = 12_000
    HIGH_STRUGGLE_RATE = 0.5
    LOW_CONFIDENCE = 0.4

    def decide(self, context: JITAIContext) -> JITAIDecision:
        state = (context.cognitive_state or "CONFUSED").upper()
        context.cognitive_state = state

        if state == "INSIGHT":
            return self._decision(
                context,
                should=True,
                intervention_type="reflection",
                trigger="insight_transfer_opportunity",
                rule_id="jitai.insight_transfer",
                rationale=(
                    "An insight is a useful transfer moment, so the policy asks "
                    "the student to explain or apply what just clicked."
                ),
            )

        if state in POSITIVE_STATES and not context.allow_positive_reflection:
            return self._decision(
                context,
                should=False,
                intervention_type="none",
                trigger="positive_state_no_interrupt",
                rule_id="jitai.no_interrupt_positive",
                rationale=(
                    f"Student is in {state}; avoid interrupting productive flow "
                    "unless reflection was explicitly requested."
                ),
            )

        if context.ignored_intervention_count >= 1 and state in NEGATIVE_STATES:
            return self._decision(
                context,
                should=True,
                intervention_type="break_suggestion",
                trigger="ignored_prior_intervention",
                rule_id="jitai.escalate_after_ignored",
                rationale=(
                    "A previous intervention was ignored in this session, so the "
                    "policy escalates once to a short reset instead of repeating "
                    "the same nudge."
                ),
            )

        if state == "FRUSTRATED" and (
            context.consecutive_state_count >= 2 or context.tone_proxy_score <= -0.55
        ):
            intervention_type = "break_suggestion" if context.tone_proxy_score <= -0.55 else "encouragement"
            return self._decision(
                context,
                should=True,
                intervention_type=intervention_type,
                trigger="persistent_frustration_or_negative_tone",
                rule_id="jitai.frustration_reset",
                rationale=(
                    "Frustration is persistent or strongly negative by proxy, so "
                    "the intervention should reduce pressure before adding more "
                    "cognitive load."
                ),
            )

        if state in {"CONFUSED", "STUCK"} and context.consecutive_state_count >= 2:
            high_struggle = _high_struggle(context.topic_stats)
            intervention_type = "task_chunking" if high_struggle else "hint"
            return self._decision(
                context,
                should=True,
                intervention_type=intervention_type,
                trigger="repeated_confused_or_stuck",
                rule_id="jitai.repeated_confusion",
                rationale=(
                    f"Student has remained {state} for "
                    f"{context.consecutive_state_count} turns. "
                    + (
                        "Topic history shows high struggle, so the policy prefers "
                        "task chunking over a generic hint."
                        if high_struggle
                        else "No high topic-struggle signal was found, so the "
                        "policy starts with a small hint."
                    )
                ),
            )

        if state == "RUSHING" and _is_very_short_latency(context):
            return self._decision(
                context,
                should=True,
                intervention_type="task_chunking",
                trigger="rushing_with_short_latency",
                rule_id="jitai.rushing_slowdown",
                rationale=(
                    "The student appears to be rushing and responded very quickly, "
                    "so the policy asks for a smaller step before continuing."
                ),
            )

        if state in NEGATIVE_STATES and _inactive_or_long_pause(context):
            return self._decision(
                context,
                should=True,
                intervention_type="task_chunking",
                trigger="negative_state_with_inactivity",
                rule_id="jitai.inactivity_after_struggle",
                rationale=(
                    "A negative cognitive state combined with inactivity or a long "
                    "pause is treated as a decision point for a chunking nudge."
                ),
            )

        if state in NEGATIVE_STATES and _high_struggle(context.topic_stats):
            return self._decision(
                context,
                should=True,
                intervention_type="task_chunking",
                trigger="topic_high_struggle_history",
                rule_id="jitai.high_struggle_topic",
                rationale=(
                    "The learning graph indicates high prior struggle on this "
                    "topic, so the policy chooses task chunking."
                ),
            )

        if state in NEGATIVE_STATES:
            intervention_type = {
                "CONFUSED": "hint",
                "STUCK": "hint",
                "FRUSTRATED": "encouragement",
                "RUSHING": "task_chunking",
            }[state]
            return self._decision(
                context,
                should=True,
                intervention_type=intervention_type,
                trigger="initial_negative_state_support",
                rule_id="jitai.initial_negative_support",
                rationale=(
                    f"The student is {state}; provide one low-intensity "
                    "metacognitive prompt without giving away the answer."
                ),
            )

        return self._decision(
            context,
            should=False,
            intervention_type="none",
            trigger="no_threshold_met",
            rule_id="jitai.no_action",
            rationale="No JITAI decision threshold was met.",
        )

    def _decision(
        self,
        context: JITAIContext,
        should: bool,
        intervention_type: str,
        trigger: str,
        rule_id: str,
        rationale: str,
    ) -> JITAIDecision:
        if intervention_type not in INTERVENTION_TYPES:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        components = JITAIComponents(
            tailoring_variable={
                "profile_id": context.profile_id,
                "topic": context.topic,
                "cognitive_state": context.cognitive_state,
                "consecutive_state_count": context.consecutive_state_count,
                "latency_ms": context.response_latency_ms,
                "time_since_previous_ms": context.time_since_previous_ms,
                "pause_summary": context.pause_summary,
                "tone_proxy_label": context.tone_proxy_label,
                "tone_proxy_score": context.tone_proxy_score,
                "topic_stats": context.topic_stats,
            },
            decision_point=trigger,
            decision_rule=rule_id,
            intervention_option=intervention_type,
            proximal_outcome="responded|re_engaged|ignored|ended_session|unknown",
            distal_outcome=(
                "longitudinal planning ratio, recovery speed, flow ratio, "
                "self-correction rate, and frustration frequency"
            ),
        )
        citation_keys = list(RULE_CITATIONS.get(rule_id, []))
        return JITAIDecision(
            should_intervene=should,
            intervention_type=intervention_type,
            trigger_condition=trigger,
            rule_id=rule_id,
            rationale=rationale,
            citation_key=citation_keys[0] if citation_keys else "",
            citation_keys=citation_keys,
            components=components,
            extension_metadata={
                "policy_family": "rule_based_jitai",
                "future_policy_hook": (
                    "Replace or augment RuleBasedJITAIPolicy after enough real "
                    "local interaction logs exist for calibration."
                ),
            },
        )


JITAI_INTERVENTION_BANK = {
    "hint": {
        "CONFUSED": "What is one small clue in the problem you can use first?",
        "STUCK": "What is the first thing the problem is asking you to find?",
        "RUSHING": "Before moving on, what step did you just take?",
        "FRUSTRATED": "What is one part of this that still makes some sense?",
        "default": "What is the smallest useful next step?",
    },
    "encouragement": {
        "FRUSTRATED": "This is a hard moment. What is one part you do understand?",
        "CONFUSED": "You do not need the whole answer yet. What is one thing you know?",
        "default": "Stay with the process. What is one thing you can check?",
    },
    "break_suggestion": {
        "FRUSTRATED": "Take a short reset, then come back and tell me the first thing you notice.",
        "STUCK": "Take a short reset, then restart with just the first sentence.",
        "default": "Take a short reset, then come back to one small step.",
    },
    "task_chunking": {
        "CONFUSED": "Break it down: what is given, and what are you trying to find?",
        "STUCK": "Break this into one tiny step: read the first sentence and name the goal.",
        "RUSHING": "Slow down and split this into steps. What is step one?",
        "FRUSTRATED": "Make it smaller: what is the easiest piece to handle first?",
        "default": "Break this into one tiny step. What comes first?",
    },
    "reflection": {
        "INSIGHT": "What made that click, and where else could you use it?",
        "default": "What changed in your thinking just now?",
    },
}


def build_jitai_intervention(
    decision: JITAIDecision,
    cognitive_state: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map a JITAI decision into the existing intervention-event shape."""

    state = (cognitive_state or "default").upper()
    intervention_type = decision.intervention_type
    if not decision.should_intervene or intervention_type == "none":
        return {
            "state": state,
            "type": "none",
            "text": "",
            "escalated": False,
            "trigger_condition": decision.trigger_condition,
            "rule_id": decision.rule_id,
            "rationale": decision.rationale,
            "citation_keys": list(decision.citation_keys),
            "metadata": metadata or {},
        }

    bank = JITAI_INTERVENTION_BANK.get(intervention_type, {})
    text = bank.get(state) or bank.get("default") or "What is the smallest next step?"
    return {
        "state": state,
        "type": intervention_type,
        "text": text,
        "escalated": intervention_type == "break_suggestion",
        "trigger_condition": decision.trigger_condition,
        "rule_id": decision.rule_id,
        "rationale": decision.rationale,
        "citation_keys": list(decision.citation_keys),
        "metadata": metadata or {},
    }


def context_from_signals(
    profile_id: str,
    topic: str,
    cognitive_state: str,
    consecutive_state_count: int = 1,
    recent_signals: list[dict[str, Any]] | None = None,
    ignored_intervention_count: int = 0,
    topic_stats: dict[str, Any] | None = None,
    allow_positive_reflection: bool = False,
) -> JITAIContext:
    """Build a JITAIContext from logger query rows and graph topic stats."""

    latest = (recent_signals or [{}])[0] if recent_signals else {}
    pause_summary = latest.get("pause_summary") or {}
    return JITAIContext(
        profile_id=profile_id,
        topic=topic,
        cognitive_state=cognitive_state,
        consecutive_state_count=consecutive_state_count,
        response_latency_ms=latest.get("response_latency_ms"),
        time_since_previous_ms=latest.get("time_since_previous_ms"),
        pause_summary=pause_summary if isinstance(pause_summary, dict) else {},
        tone_proxy_label=latest.get("tone_proxy_label") or "neutral",
        tone_proxy_score=float(latest.get("tone_proxy_score") or 0.0),
        ignored_intervention_count=ignored_intervention_count,
        topic_stats=topic_stats or {},
        allow_positive_reflection=allow_positive_reflection,
    )


def topic_stats_from_learning_graph(learning_graph: Any, topic: str) -> dict[str, Any]:
    """Read topic struggle stats from ProfileLearningGraph or a compatible graph."""

    if learning_graph is None:
        return {}
    graph = getattr(learning_graph, "graph", learning_graph)
    candidates = _topic_candidates(topic)
    node_key = None
    for key in candidates:
        try:
            if graph.has_node(key):
                node_key = key
                break
        except AttributeError:
            nodes = getattr(graph, "nodes", {})
            if key in nodes:
                node_key = key
                break
    if node_key is None:
        return {}

    nodes = getattr(graph, "nodes", {})
    raw = nodes[node_key] if node_key in nodes else {}
    study_count = int(raw.get("study_count", 0) or 0)
    struggle_count = int(raw.get("struggle_count", 0) or 0)
    confidence = raw.get("confidence")
    struggle_rate = struggle_count / study_count if study_count else 0.0
    return {
        "topic_key": node_key,
        "confidence": confidence,
        "study_count": study_count,
        "struggle_count": struggle_count,
        "struggle_rate": round(struggle_rate, 3),
        "study_hours": list(raw.get("study_hours", []) or []),
        "explanation_styles": dict(raw.get("explanation_styles", {}) or {}),
        "high_struggle": _high_struggle(
            {
                "confidence": confidence,
                "study_count": study_count,
                "struggle_count": struggle_count,
                "struggle_rate": struggle_rate,
            }
        ),
    }


def _topic_candidates(topic: str) -> list[str]:
    raw = topic or ""
    lowered = raw.strip().lower()
    underscored = lowered.replace(" ", "_")
    return [raw, lowered, underscored]


def _high_struggle(topic_stats: dict[str, Any]) -> bool:
    if not topic_stats:
        return False
    study_count = int(topic_stats.get("study_count", 0) or 0)
    struggle_count = int(topic_stats.get("struggle_count", 0) or 0)
    rate = topic_stats.get("struggle_rate")
    if rate is None and study_count:
        rate = struggle_count / study_count
    confidence = topic_stats.get("confidence")
    low_confidence = confidence is not None and float(confidence) <= RuleBasedJITAIPolicy.LOW_CONFIDENCE
    high_rate = study_count >= 2 and float(rate or 0.0) >= RuleBasedJITAIPolicy.HIGH_STRUGGLE_RATE
    repeated_struggle = struggle_count >= 2
    return bool(low_confidence or high_rate or repeated_struggle or topic_stats.get("high_struggle"))


def _is_very_short_latency(context: JITAIContext) -> bool:
    value = context.response_latency_ms
    return value is not None and int(value) <= RuleBasedJITAIPolicy.VERY_SHORT_LATENCY_MS


def _inactive_or_long_pause(context: JITAIContext) -> bool:
    inactive = (
        context.time_since_previous_ms is not None
        and int(context.time_since_previous_ms) >= RuleBasedJITAIPolicy.INACTIVITY_THRESHOLD_MS
    )
    max_pause = int((context.pause_summary or {}).get("max_ms", 0) or 0)
    long_pause = max_pause >= RuleBasedJITAIPolicy.LONG_PAUSE_MS
    return inactive or long_pause
