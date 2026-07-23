"""Gradio interface for the local ARIA metacognition pipeline."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr

from metacognition.analyzer import CognitiveStateAnalyzer
from metacognition.interaction_logger import (
    InteractionEvent,
    InteractionLogger,
    InterventionEvent,
    InterventionOutcome,
    tone_proxy,
)
from metacognition.interventions import (
    STATE_RANK,
    MetacognitiveInterventionGenerator,
)
from metacognition.jitai import JITAIContext, RuleBasedJITAIPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]
CITATIONS_PATH = REPO_ROOT / "data" / "research" / "citations.json"
SESSION_COLUMNS = ["Time", "State", "Confidence", "Student message", "ARIA response"]

STATE_COLORS = {
    "PLANNING": "#2563eb",
    "FLOW": "#059669",
    "CONFUSED": "#d97706",
    "RUSHING": "#dc2626",
    "FRUSTRATED": "#be123c",
    "STUCK": "#7c3aed",
    "INSIGHT": "#0891b2",
}

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.aria-header { border-bottom: 1px solid var(--border-color-primary); padding: 8px 0 16px; }
.aria-header h1 { font-size: 28px !important; letter-spacing: 0 !important; margin: 0 !important; }
.aria-header p { color: var(--body-text-color-subdued); margin: 4px 0 0 !important; }
.aria-panel { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 14px; }
.state-readout { min-height: 76px; }
.state-readout h2 { font-size: 20px !important; letter-spacing: 0 !important; margin: 0 0 4px !important; }
.state-readout p { margin: 0 !important; color: var(--body-text-color-subdued); }
"""


@dataclass
class SessionRuntime:
    session_id: str
    profile_id: str
    analyzer: CognitiveStateAnalyzer
    generator: MetacognitiveInterventionGenerator
    logger: InteractionLogger
    policy: RuleBasedJITAIPolicy = field(default_factory=RuleBasedJITAIPolicy)
    last_state: str = ""
    consecutive_state_count: int = 0
    last_turn_at: float | None = None
    pending_intervention_id: int | None = None
    pending_state: str = ""


_RUNTIMES: dict[str, SessionRuntime] = {}


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _runtime(
    session_id: str,
    profile_id: str,
    use_llm: bool,
    logger: InteractionLogger | None = None,
) -> SessionRuntime:
    runtime = _RUNTIMES.get(session_id)
    if runtime is None:
        runtime = SessionRuntime(
            session_id=session_id,
            profile_id=profile_id,
            analyzer=CognitiveStateAnalyzer(use_llm=use_llm),
            generator=MetacognitiveInterventionGenerator(
                student_name=f"demo_{session_id[:8]}"
            ),
            logger=logger or InteractionLogger(),
        )
        _RUNTIMES[session_id] = runtime
    runtime.profile_id = profile_id
    runtime.analyzer.use_llm = use_llm
    return runtime


def _citation_catalog() -> dict[str, dict[str, Any]]:
    try:
        with CITATIONS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("citations", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _citation_text(keys: list[str]) -> str:
    catalog = _citation_catalog()
    labels = []
    for key in keys:
        item = catalog.get(key, {})
        title = item.get("title", key)
        year = item.get("year", "")
        url = item.get("url", "")
        label = f"{title} ({year})" if year else title
        labels.append(f"[{label}]({url})" if url else label)
    return " · ".join(labels) if labels else "No citation attached"


def _state_readout(state: str, confidence: float, method: str) -> str:
    color = STATE_COLORS.get(state, "#475569")
    return (
        "<div class='state-readout'>"
        f"<h2 style='color:{color}'>{state}</h2>"
        f"<p>{confidence:.0%} confidence | {method}</p>"
        "</div>"
    )


def _session_rows(logger: InteractionLogger, session_id: str) -> list[list[Any]]:
    rows = logger.get_recent_interaction_signals(session_id, limit=20)
    output: list[list[Any]] = []
    for row in reversed(rows):
        output.append(
            [
                str(row.get("timestamp", ""))[11:19],
                row.get("cognitive_state", ""),
                row.get("analyzer_confidence", ""),
                row.get("student_message", ""),
                row.get("aria_response", ""),
            ]
        )
    return output


def _close_pending_outcome(
    runtime: SessionRuntime,
    outcome: str,
    next_state: str | None = None,
) -> None:
    if runtime.pending_intervention_id is None:
        return
    reengaged = None
    if next_state and runtime.pending_state:
        reengaged = STATE_RANK.get(next_state, 0) > STATE_RANK.get(
            runtime.pending_state, 0
        )
    runtime.logger.log_outcome(
        InterventionOutcome(
            intervention_id=runtime.pending_intervention_id,
            outcome=outcome,
            next_cognitive_state=next_state,
            reengaged=reengaged,
        )
    )
    if next_state and runtime.pending_state:
        runtime.generator.record_outcome(runtime.pending_state, next_state)
    runtime.pending_intervention_id = None
    runtime.pending_state = ""


def handle_message(
    message: str,
    history: list[dict[str, Any]] | None,
    session_id: str,
    profile_id: str,
    topic: str,
    use_llm: bool,
) -> tuple[
    str,
    list[dict[str, Any]],
    str,
    str,
    str,
    str,
    str,
    list[list[Any]],
]:
    """Run one complete analyzer -> policy -> intervention -> logger turn."""
    clean_message = (message or "").strip()
    messages = list(history or [])
    if not clean_message:
        return (
            "",
            messages,
            _state_readout("WAITING", 0.0, "idle"),
            "",
            "",
            "",
            "No outcome to record",
            [],
        )

    runtime = _runtime(session_id, profile_id.strip() or "demo_student", use_llm)
    now = time.monotonic()
    elapsed_ms = (
        int((now - runtime.last_turn_at) * 1000)
        if runtime.last_turn_at is not None
        else None
    )
    analysis = runtime.analyzer.analyze(clean_message)
    state = analysis["state"]

    _close_pending_outcome(runtime, "responded", next_state=state)

    if state == runtime.last_state:
        runtime.consecutive_state_count += 1
    else:
        runtime.consecutive_state_count = 1
    runtime.last_state = state
    runtime.last_turn_at = now

    tone = tone_proxy(clean_message)
    decision = runtime.policy.decide(
        JITAIContext(
            profile_id=runtime.profile_id,
            topic=topic.strip() or "general",
            cognitive_state=state,
            consecutive_state_count=runtime.consecutive_state_count,
            time_since_previous_ms=elapsed_ms,
            tone_proxy_label=tone.label,
            tone_proxy_score=tone.score,
            ignored_intervention_count=runtime.logger.get_ignored_intervention_count(
                session_id
            ),
        )
    )

    citation_keys = list(decision.citation_keys)
    if decision.should_intervene:
        intervention = runtime.generator.generate(
            state,
            consecutive_count=runtime.consecutive_state_count,
            student_input=clean_message,
        )
        response = intervention["text"]
        citation_keys = list(
            dict.fromkeys(citation_keys + intervention.get("citation_keys", []))
        )
        intervention_id = runtime.logger.log_intervention(
            InterventionEvent(
                session_id=session_id,
                profile_id=runtime.profile_id,
                topic=topic.strip() or "general",
                trigger_condition=decision.trigger_condition,
                intervention_type=decision.intervention_type,
                intervention_text=response,
                rule_id=decision.rule_id,
                rationale=decision.rationale,
                citation_keys=citation_keys,
                metadata={"analysis_method": analysis["method"]},
            )
        )
        runtime.pending_intervention_id = intervention_id
        runtime.pending_state = state
        outcome_status = "Awaiting outcome"
    else:
        response = "Keep going. I'm listening to your process."
        outcome_status = "No intervention triggered"

    runtime.logger.log_interaction(
        InteractionEvent(
            session_id=session_id,
            profile_id=runtime.profile_id,
            topic=topic.strip() or "general",
            student_message=clean_message,
            aria_response=response,
            time_since_previous_ms=elapsed_ms,
            tone_proxy_label=tone.label,
            tone_proxy_score=tone.score,
            tone_proxy_evidence=tone.evidence,
            cognitive_state=state,
            analyzer_confidence=float(analysis["confidence"]),
            metadata={
                "evidence": analysis["evidence"],
                "method": analysis["method"],
                "rule_id": decision.rule_id,
            },
        )
    )

    messages.append({"role": "user", "content": clean_message})
    messages.append({"role": "assistant", "content": response})
    return (
        "",
        messages,
        _state_readout(state, float(analysis["confidence"]), analysis["method"]),
        analysis["evidence"],
        f"`{decision.rule_id}`  \n{decision.rationale}",
        _citation_text(citation_keys),
        outcome_status,
        _session_rows(runtime.logger, session_id),
    )


def mark_outcome(
    session_id: str, profile_id: str, use_llm: bool, outcome: str
) -> tuple[str, list[list[Any]]]:
    runtime = _runtime(session_id, profile_id.strip() or "demo_student", use_llm)
    if runtime.pending_intervention_id is None:
        return "No pending intervention", _session_rows(runtime.logger, session_id)
    _close_pending_outcome(runtime, outcome)
    labels = {
        "re_engaged": "Recorded: re-engaged",
        "responded": "Recorded: responded",
        "ignored": "Recorded: ignored",
    }
    return labels[outcome], _session_rows(runtime.logger, session_id)


def reset_session() -> tuple[
    str,
    list[dict[str, Any]],
    str,
    str,
    str,
    str,
    str,
    list[list[Any]],
]:
    session_id = _new_session_id()
    return (
        session_id,
        [],
        _state_readout("WAITING", 0.0, "idle"),
        "",
        "",
        "",
        "New session",
        [],
    )


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="ARIA") as demo:
        session_id = gr.State(_new_session_id())

        gr.HTML(
            "<div class='aria-header'><h1>ARIA</h1>"
            "<p>Metacognitive tutor console</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    value=[],
                    height=470,
                    label="Learning session",
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Think out loud...",
                        show_label=False,
                        lines=2,
                        scale=8,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    reengaged = gr.Button("Re-engaged")
                    responded = gr.Button("Responded")
                    ignored = gr.Button("Ignored")
                    reset = gr.Button("New session")
                outcome_status = gr.Markdown("New session")

            with gr.Column(scale=4, elem_classes=["aria-panel"]):
                state_readout = gr.HTML(
                    _state_readout("WAITING", 0.0, "idle"), label="State"
                )
                evidence = gr.Textbox(label="Detection evidence", interactive=False)
                rule = gr.Markdown(label="Decision rule")
                citations = gr.Markdown(label="Research basis")
                profile_id = gr.Textbox(value="demo_student", label="Student")
                topic = gr.Textbox(value="algebra", label="Topic")
                use_llm = gr.Checkbox(value=True, label="Use local Ollama fallback")

        session_table = gr.Dataframe(
            headers=SESSION_COLUMNS,
            datatype=["str", "str", "number", "str", "str"],
            value=[],
            interactive=False,
            label="Session history",
            wrap=True,
        )

        turn_inputs = [message, chatbot, session_id, profile_id, topic, use_llm]
        turn_outputs = [
            message,
            chatbot,
            state_readout,
            evidence,
            rule,
            citations,
            outcome_status,
            session_table,
        ]
        send.click(handle_message, inputs=turn_inputs, outputs=turn_outputs)
        message.submit(handle_message, inputs=turn_inputs, outputs=turn_outputs)

        for button, outcome in (
            (reengaged, "re_engaged"),
            (responded, "responded"),
            (ignored, "ignored"),
        ):
            button.click(
                lambda sid, pid, llm, selected=outcome: mark_outcome(
                    sid, pid, llm, selected
                ),
                inputs=[session_id, profile_id, use_llm],
                outputs=[outcome_status, session_table],
            )

        reset.click(
            reset_session,
            outputs=[
                session_id,
                chatbot,
                state_readout,
                evidence,
                rule,
                citations,
                outcome_status,
                session_table,
            ],
        )

    return demo
