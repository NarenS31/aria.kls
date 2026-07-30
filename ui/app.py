"""
ARIA — single Gradio app (localhost:7860).

A metacognitive learning tool for neurodivergent students, plus the research
system that measures whether it works.

Four tabs:
  1. LEARN       — Chat mode + Think-Aloud mode (toggle)
  2. MY PROGRESS — personal metacognitive growth dashboard
  3. RESEARCH    — experiment controls, results, figures, metacognition eval
  4. SETTINGS    — profile, ARIA behaviour, data

Exports used by main.py:  build_ui(), set_agent(agent, scheduler),
CUSTOM_CSS, ARIA_THEME.
"""

import io
import json
import os
import shutil
import subprocess
import threading
import zipfile
import html
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import gradio as gr

from agent.nudge import nudge_queue

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ARIA_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ARIA_ROOT / "data"
EVAL_DIR = ARIA_ROOT / "eval"
FIGURES_DIR = DATA_DIR / "figures"
RESULTS_JSON = DATA_DIR / "experiment_results_full.json"
FAILURE_JSON = DATA_DIR / "failure_patterns.json"
METACOG_EVAL_JSON = EVAL_DIR / "data" / "eval" / "metacognition_eval_results.json"
EXPORT_DIR = DATA_DIR / "exports"


# ------------------------------------------------------------------
# Module-level singletons (injected by main.py)
# ------------------------------------------------------------------

_aria_agent = None
_nudge_scheduler = None


def set_agent(agent, scheduler) -> None:
    global _aria_agent, _nudge_scheduler
    _aria_agent = agent
    _nudge_scheduler = scheduler


def _profile() -> dict:
    if _aria_agent is not None:
        return _aria_agent.profile
    try:
        from memory.graph import load_profile
        return load_profile() or {}
    except Exception:
        return {}


# ==================================================================
#  LEARN — shared: emotion tracker + streaming chat
# ==================================================================

_session_message_count = 0
_emotion_tracker = None


def _get_emotion_tracker():
    global _emotion_tracker
    if _emotion_tracker is None:
        try:
            from agent.emotion import EmotionTracker
            _emotion_tracker = EmotionTracker()
        except Exception:
            pass
    return _emotion_tracker


def respond_stream(user_message: str, history: Optional[List[dict]]) -> Generator[tuple, None, None]:
    """Streaming, emotion-aware chat used by the Learn → Chat mode."""
    global _session_message_count
    user_message = (user_message or "").strip()
    history = list(history or [])
    if not user_message:
        yield "", history
        return

    tracker = _get_emotion_tracker()
    emotion_result = tracker.analyze(user_message) if tracker else None

    if emotion_result and emotion_result.get("intervention") == "mandatory_break":
        break_msg = (
            "⏸️ **Take a 2-minute break.** You've been pushing through something tough. "
            "Step away, grab water, come back fresh. I'll be right here."
        )
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": break_msg},
        ]
        yield "", history
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "▌"},
    ]
    yield "", history

    accumulated = ""
    try:
        if _aria_agent is None:
            raise RuntimeError("ARIA is still starting. Try again in a moment.")
        for token in _aria_agent.chat_stream(user_message):
            accumulated += token
            history[-1]["content"] = accumulated + "▌"
            yield "", history
    except Exception as e:
        history[-1]["content"] = f"(ARIA hit an error: {e})"
        yield "", history
        return

    if emotion_result and emotion_result.get("auto_response") and emotion_result.get("state") != "FLOW":
        history[-1]["content"] = f"_{emotion_result['auto_response']}_\n\n{accumulated}"
    else:
        history[-1]["content"] = accumulated

    _session_message_count += 1

    if _session_message_count >= 5 and _nudge_scheduler and hasattr(_nudge_scheduler, "initiation"):
        try:
            _nudge_scheduler.initiation.mark_session_started()
        except Exception:
            pass

    if _session_message_count % 10 == 0:
        try:
            from memory.checkpoints import save_checkpoint
            topic = _aria_agent.get_last_topic() if _aria_agent else "session"
            user_msgs = [m["content"] for m in history if m.get("role") == "user"]
            save_checkpoint(
                topic=topic or "session",
                concept=user_msgs[-1][:80] if user_msgs else "in progress",
                step=_session_message_count // 5,
                last_question=user_msgs[-1] if user_msgs else "",
                last_answer=accumulated[:500],
                confidence=0.5,
            )
        except Exception:
            pass

    yield "", history


def respond(user_message: str, history: Optional[List[dict]]) -> tuple:
    user_message = (user_message or "").strip()
    history = list(history or [])
    if not user_message:
        return "", history
    try:
        if _aria_agent is None:
            raise RuntimeError("ARIA is still starting. Try again in a moment.")
        response = _aria_agent.chat(user_message)
    except Exception as e:
        response = f"(ARIA hit an error: {e})"
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return "", history


def inject_stuck(history):
    return respond("I'm stuck. Break this into the smallest possible step for me.", history)


def inject_example(history):
    return respond("Give me one concrete real-world example of what we just covered. Nothing else.", history)


def inject_quiz(history):
    return respond("Quiz me. Ask 3 short questions on what we just covered, one at a time.", history)


# ------------------------------------------------------------------
# Focus timer
# ------------------------------------------------------------------

_focus_start: Optional[datetime] = None
_focus_lock = threading.Lock()
_POMODORO_SECONDS = 25 * 60


def start_focus_session():
    global _focus_start
    with _focus_lock:
        _focus_start = datetime.now()
    if _aria_agent is not None:
        _aria_agent.reset_break_timer()
    return get_focus_status()


def stop_focus_session():
    global _focus_start
    with _focus_lock:
        if _focus_start is None:
            return _focus_html("No active session", False)
        mins = int((datetime.now() - _focus_start).total_seconds()) // 60
        _focus_start = None
    return _focus_html(f"Nice — you focused for {mins} min. Take a breath.", False)


def get_focus_status():
    with _focus_lock:
        if _focus_start is None:
            return _focus_html("No active session", False)
        elapsed = (datetime.now() - _focus_start).total_seconds()
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    over = elapsed >= _POMODORO_SECONDS
    label = f"⏱️ {mins:02d}:{secs:02d}"
    if over:
        label += " — 25 min hit, time for a break"
    return _focus_html(label, over)


def _focus_html(text: str, alert: bool) -> str:
    color = "#b42318" if alert else "#205e55"
    return (
        f"<div class='aria-focus-display' style='color:{color};'>{text}</div>"
    )


# ------------------------------------------------------------------
# Chat right-panel: context + SRS
# ------------------------------------------------------------------

def _conf_bar(conf: float, label: str = "") -> str:
    pct = max(0.0, min(1.0, conf))
    color = "#e88375" if pct < 0.4 else "#f4d01f" if pct < 0.7 else "#8dcfbd"
    return (
        f"<div class='aria-confidence-bar'>"
        f"<div class='aria-confidence-label'>{label} {pct:.0%}</div>"
        f"<div class='aria-confidence-track'>"
        f"<div style='width:{pct*100:.0f}%;height:100%;background:{color};'></div></div></div>"
    )


def get_context_panel() -> str:
    if _aria_agent is None:
        return "_Loading…_"
    topic = _aria_agent.get_last_topic()
    parts = []
    if topic:
        info = _aria_agent.lg.get_topic_info(topic) or {}
        conf = float(info.get("confidence", 0.5))
        parts.append(f"### Current topic\n**{topic}**")
        parts.append(_conf_bar(conf, "Your confidence:"))
    else:
        parts.append("### Current topic\n_Nothing yet — start a conversation._")

    recap = topic or ""
    if recap:
        parts.append(f"**Last session:** you worked on _{recap}_.")
    return "\n\n".join(parts)


def get_srs_status() -> str:
    try:
        from memory.srs import get_due_cards, get_all_cards
        due = get_due_cards()
        all_cards = get_all_cards()
        if not all_cards:
            return "_No concepts tracked yet. Chat with ARIA to build your review queue._"
        lines = [f"**{len(due)} concept(s) due for review today**"]
        for card in due[:4]:
            lines.append(f"🔴 **{card['concept']}** · _{card['topic']}_")
        upcoming = [c for c in all_cards if c not in due]
        upcoming.sort(key=lambda c: c.get("next_review_date", "9999"))
        for card in upcoming[:2]:
            lines.append(f"🟡 **{card['concept']}** · due {card.get('next_review_date', '?')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"_SRS unavailable: {e}_"


def start_srs_quiz(history):
    try:
        from memory.srs import get_due_cards, format_srs_prompt
        due = get_due_cards()
        if not due:
            return "", history + [
                {"role": "assistant", "content": "Nothing due for review right now. Come back later!"}
            ]
        return respond(format_srs_prompt(due[0]), history)
    except Exception as e:
        return "", history + [{"role": "assistant", "content": f"SRS error: {e}"}]


def poll_nudge() -> str:
    try:
        msg = nudge_queue.get_nowait()
        return f"💡 **ARIA:** {msg['message']}"
    except Exception:
        return ""


# ==================================================================
#  LEARN — Think-Aloud mode
# ==================================================================

_think_states: List[dict] = []      # session history of detected states (last N)
_current_problem: dict = {"problem": "", "topic": ""}

_INTERVENTION_LABEL = {
    "PLANNING": "Strategy Check",
    "FLOW": "Deepen Understanding",
    "CONFUSED": "Decompose",
    "RUSHING": "Force Planning",
    "FRUSTRATED": "De-escalate",
    "STUCK": "Smallest First Step",
    "INSIGHT": "Consolidate",
}


def new_think_problem(topic: Optional[str] = None):
    """Generate a fresh practice problem and reset the think-aloud panels."""
    global _current_problem, _think_states
    if _aria_agent is None:
        return "_Loading…_", "", _state_panel_html(), ""
    try:
        if topic:
            _current_problem = _aria_agent.generate_think_aloud_problem(
                topic_override=topic
            )
        else:
            _current_problem = _aria_agent.generate_think_aloud_problem()
    except Exception:
        _current_problem = {
            "problem": "Solve for x: 3(x - 4) = 2x + 5",
            "topic": "Algebra",
        }
    _think_states = []
    topic = html.escape(str(_current_problem.get("topic", "Practice")))
    problem = html.escape(str(_current_problem.get("problem", "")))
    prob_md = (
        f"<div class='aria-problem-label'>Practice problem · {topic}</div>"
        f"<div class='aria-problem-card'>{problem}</div>"
        f"<p class='aria-problem-prompt'>{_aria_agent.THINK_ALOUD_PROMPT}</p>"
    )
    return prob_md, "", _state_panel_html(), "_Start when you're ready._"


def start_topic_practice(topic):
    """Open a topic-specific practice problem in the Learn tab."""
    problem, text, state, response = new_think_problem(topic)
    return problem, text, state, response, gr.Tabs(selected="tab_learn")


def _state_panel_html() -> str:
    if not _think_states:
        return (
            "<div class='aria-state-empty'>"
            "<strong>ARIA is listening</strong>"
            "<span>Work through the problem in your own words.</span></div>"
        )
    def display_label(turn: dict) -> str:
        intent = turn.get("intent")
        if intent == "HELP_REQUEST":
            return "Needs a starting point"
        if intent == "ATTEMPT_META":
            return "Getting started"
        if turn["state"] == "UNKNOWN":
            return "Not enough evidence"
        return turn["state"].title()

    latest = _think_states[-1]
    state_label = display_label(latest)
    big = (
        f"<div class='aria-state-result'>"
        f"<span>Thinking pattern</span>"
        f"<strong>{state_label}</strong>"
        f"<p>ARIA noticed {latest.get('evidence') or 'signals in your wording'}.</p>"
        f"</div>"
    )
    hist = "<div class='aria-state-history'><b>This session</b><div>"
    for s in _think_states[-5:][::-1]:
        hist += (
            f"<span>{display_label(s)}</span>"
        )
    hist += "</div></div>"
    return big + hist


def submit_think_aloud(text: str):
    """Run one think-aloud turn: detect state, return a metacognitive question."""
    global _think_states
    if _aria_agent is None:
        return _state_panel_html(), "_Loading…_", ""
    if not text or not text.strip():
        return _state_panel_html(), "_Type (or record) your reasoning first, then submit._", ""

    try:
        result = _aria_agent.think_aloud_turn(text.strip())
    except Exception:
        return (
            _state_panel_html(),
            "_ARIA could not analyze that turn. Your text is still here, so you can try again._",
            text,
        )
    _think_states.append(result)

    if result.get("acknowledged"):
        # The student self-initiated metacognition — ARIA acknowledges the habit
        # instead of prompting, so we don't override what they already did.
        mtype = result.get("metacognitive_type", "metacognition")
        banner = f"SELF-INITIATED {mtype.upper()} · NICE"
    elif not result.get("intervened", True):
        # Timer says it's not the moment yet — a light, non-prompting nudge.
        banner = "GIVING YOU A MOMENT"
    else:
        label = _INTERVENTION_LABEL.get(result.get("intervention_state", result["state"]), "Reflect")
        if result.get("escalated"):
            label += f" · escalated ({result.get('escalation_kind', 'help')})"
        banner = f"INTERVENTION · {label.upper()}"

    response_md = (
        f"<div class='aria-response-card'>"
        f"<div class='aria-response-label'>{html.escape(str(banner))}</div>"
        f"<div class='aria-response-question'>{html.escape(str(result['question']))}</div>"
        f"<div class='aria-response-note'>One question. No answer revealed.</div></div>"
    )
    return _state_panel_html(), response_md, ""


def submit_confidence(rating):
    """Store the student's pre-attempt confidence (1-5) for calibration."""
    if _aria_agent is None:
        return "_Loading…_"
    if rating is None:
        return "_Pick a confidence (1–5) before you start._"
    try:
        _aria_agent.set_confidence(int(rating))
    except Exception:
        return "_Could not store confidence._"
    return (f"<div class='aria-inline-status'>Confidence <b>{int(rating)}/5</b> "
            f"recorded. Now think it through out loud, then tell ARIA how it went.</div>")


def resolve_correctness(correct: bool):
    """Finalise the calibration record with the attempt's outcome."""
    if _aria_agent is None:
        return "_Loading…_"
    try:
        res = _aria_agent.resolve_confidence(bool(correct))
    except Exception:
        return "_Could not record outcome._"
    if not res.get("ok"):
        return "_Set a confidence rating first, then work the problem, then mark the outcome._"
    rec = res["record"]
    verdict = "correct ✓" if rec["correct"] else "not quite ✗"
    return (f"<div class='aria-inline-status'>Logged: confidence "
            f"<b>{rec['confidence_before']}/5</b> → <b>{verdict}</b>. "
            f"Your calibration just updated.</div>")


def reset_confidence_ui():
    """Clear the confidence widgets when a new problem starts."""
    return gr.update(value=None), "", ""


def transcribe_audio(filepath):
    """Best-effort mic transcription (Whisper). Falls back to a hint if unavailable."""
    if not filepath:
        return ""
    try:
        import whisper  # openai-whisper
        model = _get_whisper()
        if model is None:
            model = whisper.load_model("base")
            _set_whisper(model)
        out = model.transcribe(filepath, fp16=False)
        return (out.get("text") or "").strip()
    except Exception:
        return "(Mic transcription needs Whisper — please type your reasoning instead.)"


_whisper_model = None
def _get_whisper():
    return _whisper_model
def _set_whisper(m):
    global _whisper_model
    _whisper_model = m


# ==================================================================
#  MY PROGRESS
# ==================================================================

def _tracker():
    """The metacognition tracker for the current student (may be empty)."""
    if _aria_agent is not None:
        try:
            return _aria_agent.metacognition_tracker()
        except Exception:
            pass
    try:
        import sys
        if str(EVAL_DIR) not in sys.path:
            sys.path.insert(0, str(EVAL_DIR))
        from metacognition.tracker import MetacognitionTracker
        return MetacognitionTracker(_profile().get("name", "default"))
    except Exception:
        return None


def _delta_arrow(cur, prev, lower_is_better=False):
    if cur is None or prev is None:
        return ""
    if abs(cur - prev) < 1e-9:
        return " <span style='color:#68788d;'>→</span>"
    improved = (cur < prev) if lower_is_better else (cur > prev)
    if improved:
        return " <span style='color:#205e55;'>▲</span>"
    return " <span style='color:#b42318;'>▼</span>"


def _metric_card(title, value, arrow, good):
    border = "#205e55" if good else "#b42318" if good is False else "#c8d1dc"
    return (
        f"<div class='aria-metric-card' style='border-color:{border};'>"
        f"<div>{title}</div><strong>{value}{arrow}</strong></div>"
    )


def get_week_cards() -> str:
    tr = _tracker()
    if tr is None:
        return "_Progress tracking unavailable._"
    try:
        this_week, last_week = tr._split_by_week()
        cur = tr._aggregate(this_week)
        prev = tr._aggregate(last_week)
    except Exception:
        cur, prev = {}, {}

    def pct(x):
        return f"{x*100:.0f}%" if x is not None else "—"

    def turns(x):
        return f"{x:.1f}" if x is not None else "—"

    if not cur:
        return (
            "<div class='aria-inline-status'>No think-aloud sessions yet this week. "
            "Head to <b>Learn → Think Aloud</b> and reason through a few problems — your "
            "metacognition metrics will appear here.</div>"
        )

    cards = [
        _metric_card("Planning Ratio", pct(cur.get("planning_ratio")),
                     _delta_arrow(cur.get("planning_ratio"), prev.get("planning_ratio")),
                     _is_good(cur.get("planning_ratio"), prev.get("planning_ratio"))),
        _metric_card("Self-Correction Rate", pct(cur.get("self_correction_rate")),
                     _delta_arrow(cur.get("self_correction_rate"), prev.get("self_correction_rate")),
                     _is_good(cur.get("self_correction_rate"), prev.get("self_correction_rate"))),
        _metric_card("Recovery Speed", turns(cur.get("recovery_speed_turns")),
                     _delta_arrow(cur.get("recovery_speed_turns"), prev.get("recovery_speed_turns"), lower_is_better=True),
                     _is_good(cur.get("recovery_speed_turns"), prev.get("recovery_speed_turns"), lower_is_better=True)),
        _metric_card("Flow Ratio", pct(cur.get("flow_ratio")),
                     _delta_arrow(cur.get("flow_ratio"), prev.get("flow_ratio")),
                     _is_good(cur.get("flow_ratio"), prev.get("flow_ratio"))),
    ]
    return "<div style='display:flex;gap:12px;flex-wrap:wrap;'>" + "".join(cards) + "</div>"


def _is_good(cur, prev, lower_is_better=False):
    if cur is None or prev is None:
        return None
    if abs(cur - prev) < 1e-9:
        return None
    return (cur < prev) if lower_is_better else (cur > prev)


# --- charts -------------------------------------------------------

def _new_ax(title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=110)
    fig.patch.set_facecolor("#fffdf8")
    ax.set_facecolor("#fffdf8")
    for spine in ax.spines.values():
        spine.set_color("#c8d1dc")
    ax.tick_params(colors="#5d6b7e", labelsize=8)
    ax.title.set_color("#10213b")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig, ax


def _fig_to_pil(fig):
    from PIL import Image
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return img


def _placeholder_img(msg):
    fig, ax = _new_ax("")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#5d6b7e",
            fontsize=10, wrap=True, transform=ax.transAxes)
    ax.axis("off")
    return _fig_to_pil(fig)


def chart_planning_ratio():
    tr = _tracker()
    series = tr.weekly_series("planning_ratio") if tr else []
    pts = [(i, s["value"]) for i, s in enumerate(series) if s["value"] is not None]
    if len(pts) < 1:
        return _placeholder_img("Planning ratio over sessions\n(no data yet)")
    fig, ax = _new_ax("Planning ratio over last 30 sessions")
    xs, ys = zip(*pts[-30:])
    ax.plot(range(len(ys)), [y * 100 for y in ys], "-o", color="#2f78bd", ms=4)
    ax.set_ylabel("%", color="#5d6b7e")
    ax.set_ylim(0, 100)
    return _fig_to_pil(fig)


def chart_state_distribution():
    tr = _tracker()
    dist = tr.state_distribution(last_n=15) if tr else []
    dist = [d for d in dist if d["distribution"]]
    if not dist:
        return _placeholder_img("State distribution per session\n(no data yet)")
    states = ["PLANNING", "FLOW", "INSIGHT", "RUSHING", "CONFUSED", "FRUSTRATED", "STUCK"]
    colors = {"PLANNING": "#22c55e", "FLOW": "#10b981", "INSIGHT": "#a78bfa",
              "RUSHING": "#f59e0b", "CONFUSED": "#eab308", "FRUSTRATED": "#f97316", "STUCK": "#ef4444"}
    fig, ax = _new_ax("State distribution per session")
    import numpy as np
    x = list(range(len(dist)))
    bottom = [0] * len(dist)
    for st in states:
        vals = [d["distribution"].get(st, 0) for d in dist]
        if sum(vals) == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=st, color=colors[st], width=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("turns", color="#5d6b7e")
    ax.legend(fontsize=6, ncol=2, facecolor="#fffdf8", edgecolor="#c8d1dc", labelcolor="#10213b")
    return _fig_to_pil(fig)


def chart_frustration_trend():
    tr = _tracker()
    if tr is None or not tr.sessions:
        return _placeholder_img("Frustration frequency trend\n(no data yet)")
    ys = []
    for r in tr.sessions[-30:]:
        m = r.get("metrics") or {}
        ys.append(1 if m.get("had_frustration") else 0)
    fig, ax = _new_ax("Frustration frequency (should decline)")
    # rolling frequency
    window = []
    freq = []
    for v in ys:
        window.append(v)
        if len(window) > 5:
            window.pop(0)
        freq.append(sum(window) / len(window) * 100)
    ax.plot(range(len(freq)), freq, "-o", color="#f97316", ms=4)
    ax.set_ylabel("% sessions", color="#5d6b7e")
    ax.set_ylim(0, 100)
    return _fig_to_pil(fig)


def chart_recovery_trend():
    tr = _tracker()
    series = tr.weekly_series("recovery_speed_turns") if tr else []
    pts = [s["value"] for s in series if s["value"] is not None]
    if not pts:
        return _placeholder_img("Recovery speed trend\n(no data yet)")
    fig, ax = _new_ax("Recovery speed — turns to get unstuck (should decline)")
    ax.plot(range(len(pts[-30:])), pts[-30:], "-o", color="#22d3ee", ms=4)
    ax.set_ylabel("turns", color="#5d6b7e")
    return _fig_to_pil(fig)


# --- topic map ----------------------------------------------------

def _srs_status_for(topic: str) -> str:
    try:
        from memory.srs import get_all_cards, get_due_cards
        due_topics = {c["topic"] for c in get_due_cards()}
        if topic in due_topics:
            return "Due today"
        for c in get_all_cards():
            if c["topic"] == topic:
                return f"Due {c.get('next_review_date', '?')[:10]}"
    except Exception:
        pass
    return "—"


def get_topic_map() -> str:
    if _aria_agent is None:
        return "_Loading…_"
    topics = _aria_agent.lg.all_topics_summary()
    if not topics:
        return "_No topics studied yet. Start a session and they'll appear here._"
    cards = []
    for t in topics[:24]:
        conf = float(t["confidence"])
        color = "#e88375" if conf < 0.4 else "#f4d01f" if conf < 0.7 else "#8dcfbd"
        last = (t.get("last_studied") or "")[:10] or "never"
        srs = _srs_status_for(t["topic"])
        cards.append(
            f"<div class='aria-topic-card'>"
            f"<div style='font-weight:700;font-size:1.02em;'>{t['topic']}</div>"
            f"<div class='aria-topic-track'>"
            f"<div style='width:{conf*100:.0f}%;height:100%;background:{color};'></div></div>"
            f"<div class='aria-topic-meta'>"
            f"Confidence {conf:.0%} · studied {t['study_count']}×<br>"
            f"Last: {last} · SRS: {srs}</div></div>"
        )
    return "<div style='display:flex;gap:12px;flex-wrap:wrap;'>" + "".join(cards) + "</div>"


def get_topic_choices():
    if _aria_agent is None:
        return gr.update(choices=[])
    topics = [t["topic"] for t in _aria_agent.lg.all_topics_summary()]
    return gr.update(choices=topics, value=(topics[0] if topics else None))


def start_focused_session(topic, history):
    """Jump into a focused chat session on a topic (switches to Learn tab)."""
    if not topic:
        return history, gr.update(), gr.update(), gr.update(), gr.update()
    _, new_hist = respond(
        f"Let's do a focused session on {topic}. Start me with one small warm-up step.",
        history,
    )
    return (
        new_hist,
        gr.Tabs(selected="tab_learn"),
        "Chat",
        gr.update(visible=True),
        gr.update(visible=False),
    )


# --- insights -----------------------------------------------------

def get_insights() -> str:
    tr = _tracker()
    if tr is None or not getattr(tr, "sessions", None):
        return (
            "_Insights appear after a few Think-Aloud sessions. They're generated weekly "
            "from your own metacognition data._"
        )
    # facts from the tracker, then have llama3.1:8b (fallback 3b) narrate them.
    try:
        facts = tr._summary_facts(
            tr._aggregate(tr._split_by_week()[0]),
            tr._aggregate(tr._split_by_week()[1]),
        )
    except Exception:
        facts = []
    text = _narrate_insights(facts)
    return text or "_Not enough data yet — keep thinking out loud!_"


def _narrate_insights(facts: List[str]) -> str:
    if not facts:
        return ""
    prompt = (
        "You are ARIA, a warm learning coach for a neurodivergent student. Turn these "
        "facts into 2-3 short, specific, encouraging insight sentences. Be concrete with "
        "numbers, never condescending. Facts:\n" + "\n".join(f"- {f}" for f in facts)
    )
    for model in ("llama3.1:8b", "llama3.2:3b"):
        try:
            import ollama
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You write brief, warm, specific progress insights as prose."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.6, "num_predict": 200},
            )
            out = resp.message.content.strip()
            if out:
                return out
        except Exception:
            continue
    return "\n\n".join(f"- {f}" for f in facts)


# ------------------------------------------------------------------
#  Metacognitive development — transfer, calibration, timing
# ------------------------------------------------------------------

def _eval_pkg(name: str):
    """Import a class from the sibling eval metacognition package on demand."""
    import sys
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import importlib
    mod = importlib.import_module(f"metacognition.{name}")
    return mod


def _transfer():
    if _aria_agent is not None:
        try:
            return _aria_agent.transfer_detector()
        except Exception:
            pass
    try:
        return _eval_pkg("transfer").TransferDetector(_profile().get("name", "default"))
    except Exception:
        return None


def _calibration():
    if _aria_agent is not None:
        try:
            return _aria_agent.calibration_tracker()
        except Exception:
            pass
    try:
        return _eval_pkg("calibration").CalibrationTracker(_profile().get("name", "default"))
    except Exception:
        return None


def _timer():
    if _aria_agent is not None:
        try:
            return _aria_agent.intervention_timer()
        except Exception:
            pass
    try:
        return _eval_pkg("timing").InterventionTimer(_profile().get("name", "default"))
    except Exception:
        return None


def _recent_earlier(vals):
    """Split a chronological value series into (earlier_mean, recent_mean)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return None, vals[0]
    mid = len(vals) // 2
    earlier, recent = vals[:mid], vals[mid:]
    e = sum(earlier) / len(earlier) if earlier else None
    r = sum(recent) / len(recent) if recent else None
    return e, r


def _dev_card(title, value, arrow, good, subtitle):
    border = "#205e55" if good else "#b42318" if good is False else "#c8d1dc"
    return (
        f"<div class='aria-metric-card aria-metric-card--wide' style='border-color:{border};'>"
        f"<div>{title}</div><strong>{value}{arrow}</strong><small>{subtitle}</small></div>"
    )


def get_metacog_dev_cards() -> str:
    """The three metacognitive-development metric cards (spec §4)."""
    tr, cal, trk = _transfer(), _calibration(), _tracker()

    # Self-Initiation Rate (higher = more transfer).
    si_val = si_prev = None
    if tr is not None:
        try:
            series = [s["self_initiation_rate"] for s in tr.rate_by_session()]
            si_prev, si_val = _recent_earlier(series)
        except Exception:
            pass

    # Calibration Error (lower = knows what they know).
    cal_val = cal_prev = None
    if cal is not None:
        try:
            series = [s["calibration_error"] for s in cal.per_session()]
            cal_prev, cal_val = _recent_earlier(series)
        except Exception:
            pass

    # Recovery Speed (lower = faster to get unstuck).
    rec_val = rec_prev = None
    if trk is not None:
        try:
            series = [s["value"] for s in trk.weekly_series("recovery_speed_turns")]
            rec_prev, rec_val = _recent_earlier(series)
        except Exception:
            pass

    if si_val is None and cal_val is None and rec_val is None:
        return ("<div class='aria-inline-status'>No metacognitive-development "
                "data yet. Reason through a few problems in <b>Learn → Think Aloud</b> "
                "(and give a confidence rating) — transfer, calibration and recovery "
                "metrics will appear here.</div>")

    def pct(x):
        return f"{x*100:.0f}%" if x is not None else "—"

    cards = [
        _dev_card("Self-Initiation Rate", pct(si_val),
                  _delta_arrow(si_val, si_prev),
                  _is_good(si_val, si_prev),
                  "You're starting to plan before ARIA asks"),
        _dev_card("Calibration Error", f"{cal_val:.2f}" if cal_val is not None else "—",
                  _delta_arrow(cal_val, cal_prev, lower_is_better=True),
                  _is_good(cal_val, cal_prev, lower_is_better=True),
                  "You know what you know more accurately"),
        _dev_card("Recovery Speed", f"{rec_val:.1f} turns" if rec_val is not None else "—",
                  _delta_arrow(rec_val, rec_prev, lower_is_better=True),
                  _is_good(rec_val, rec_prev, lower_is_better=True),
                  "You're getting unstuck faster"),
    ]
    return "<div style='display:flex;gap:12px;flex-wrap:wrap;'>" + "".join(cards) + "</div>"


def chart_transfer():
    """Self-initiation rate per session, with the 40% transfer threshold."""
    tr = _transfer()
    series = tr.transfer_series() if tr else []
    pts = [(s["session_ordinal"], s["value"]) for s in series if s.get("value") is not None]
    if not pts:
        return _placeholder_img("Self-initiation rate per session\n(no data yet)")
    fig, ax = _new_ax("Self-initiation rate per session (transfer)")
    xs = [p[0] for p in pts]
    ys = [p[1] * 100 for p in pts]
    ax.plot(xs, ys, "-o", color="#a78bfa", ms=4)
    threshold = getattr(tr, "TRANSFER_THRESHOLD", 0.40) * 100
    ax.axhline(threshold, ls="--", color="#22c55e", lw=1.2)
    ax.text(xs[0], min(threshold + 3, 96), f"Transfer threshold ({threshold:.0f}%)",
            color="#22c55e", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of turns", color="#5d6b7e")
    ax.set_xlabel("session", color="#5d6b7e")
    return _fig_to_pil(fig)


def chart_calibration():
    """Calibration error per session, with overconfidence / underconfidence bands."""
    cal = _calibration()
    series = cal.error_series() if cal else []
    pts = [s for s in series if s.get("calibration_error") is not None]
    if not pts:
        return _placeholder_img("Calibration error per session\n(no data yet)")
    fig, ax = _new_ax("Calibration error per session (should decline)")
    xs = [s["session_ordinal"] for s in pts]
    err = [s["calibration_error"] for s in pts]
    over = [s.get("overconfidence_rate") or 0 for s in pts]
    under = [s.get("underconfidence_rate") or 0 for s in pts]
    ax.fill_between(xs, 0, over, color="#ef4444", alpha=0.16, label="overconfidence")
    ax.fill_between(xs, 0, under, color="#3b82f6", alpha=0.16, label="underconfidence")
    ax.plot(xs, err, "-o", color="#f59e0b", ms=4, label="calibration error")
    ax.set_ylim(0, 1)
    ax.set_ylabel("error / rate", color="#5d6b7e")
    ax.set_xlabel("session", color="#5d6b7e")
    ax.legend(fontsize=6, facecolor="#fffdf8", edgecolor="#c8d1dc", labelcolor="#10213b")
    return _fig_to_pil(fig)


def chart_timing_heatmap():
    """state x intervention-turn -> recovery-rate heatmap (optimal moments)."""
    tm = _timer()
    hm = tm.heatmap() if tm else None
    if not hm:
        return _placeholder_img("Intervention-timing heatmap\n(no data yet)")
    import numpy as np
    states, buckets = hm["states"], hm["buckets"]
    grid, ncount = hm["recovery_rate"], hm["n"]
    mat = np.full((len(states), len(buckets)), np.nan)
    for i, s in enumerate(states):
        for j, b in enumerate(buckets):
            v = grid.get(s, {}).get(b)
            if v is not None:
                mat[i, j] = v
    if np.all(np.isnan(mat)):
        return _placeholder_img("Intervention-timing heatmap\n(no data yet)")
    fig, ax = _new_ax("Recovery rate by intervention turn (find the bright cells)")
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f"turn {b}" if b < 4 else "turn 4+" for b in buckets], fontsize=7)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=7)
    for i in range(len(states)):
        for j in range(len(buckets)):
            n = ncount.get(states[i], {}).get(buckets[j], 0)
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]*100:.0f}%\nn={n}", ha="center", va="center",
                        fontsize=6, color="#111827")
    return _fig_to_pil(fig)


def get_metacog_insight() -> str:
    """A weekly, LLM-narrated insight over the three development systems (§4)."""
    tr, cal, tm = _transfer(), _calibration(), _timer()
    facts: List[str] = []
    if tr is not None:
        try:
            summ = tr.summary()
            rate = summ.get("self_initiation_rate")
            phases = summ.get("phases", {})
            early, late = phases.get("early_1_5"), phases.get("late_16_plus")
            if early is not None and late is not None:
                facts.append(f"Your self-initiation rate went from {early*100:.0f}% in your "
                             f"early sessions to {late*100:.0f}% recently.")
            elif rate is not None:
                facts.append(f"You self-initiated metacognition on {rate*100:.0f}% of your "
                             f"think-aloud turns without ARIA prompting.")
            order = (summ.get("by_type") or {}).get("transfer_order") or []
            if order:
                facts.append(f"The metacognitive habit you picked up first was {order[0]}.")
            if summ.get("transferred"):
                facts.append("You're now above the 40% transfer threshold — the habit is sticking.")
        except Exception:
            pass
    if cal is not None:
        try:
            c = cal.summary()
            ce = c.get("calibration_error")
            if ce is not None:
                facts.append(f"Your calibration error is {ce:.2f} (lower means your confidence "
                             f"matches how you actually do).")
            over = c.get("overconfidence_rate")
            if over:
                facts.append(f"You were overconfident on {over*100:.0f}% of problems this period.")
        except Exception:
            pass
    if tm is not None:
        try:
            opt = tm.optimal_by_state()
            if opt:
                pairs = ", ".join(f"{s.lower()} at turn {t}" for s, t in opt.items())
                facts.append(f"ARIA's best moments to step in for you: {pairs}.")
        except Exception:
            pass
    if not facts:
        return ("_Insights appear after a few Think-Aloud sessions with confidence ratings. "
                "They're generated from your own transfer, calibration and timing data._")
    return _narrate_insights(facts) or "\n\n".join(f"- {f}" for f in facts)


# ==================================================================
#  RESEARCH
# ==================================================================

def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ARIA_ROOT}:{EVAL_DIR}:" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_subprocess(cmd: List[str]) -> Generator[str, None, None]:
    """Run a command from the ARIA root, streaming combined output to the log box."""
    header = f"$ {' '.join(cmd)}\n{'-'*50}\n"
    yield header
    try:
        proc = subprocess.Popen(
            ["python3.11", *cmd],
            cwd=str(ARIA_ROOT),
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        yield header + f"\n[failed to start: {e}]"
        return
    acc = header
    assert proc.stdout is not None
    for line in proc.stdout:
        acc += line
        yield acc[-8000:]        # keep the log box bounded
    proc.wait()
    acc += f"\n{'-'*50}\n[exit code {proc.returncode}]"
    yield acc[-8000:]


def run_quick_experiment():
    yield from _stream_subprocess(["eval/full_experiment.py", "--quick", "--local-only"])

def run_full_experiment():
    yield from _stream_subprocess(["eval/full_experiment.py", "--full", "--local-only"])

def run_learning_curve():
    yield from _stream_subprocess(["eval/full_experiment.py", "--learning-curve"])

def run_lora_training():
    yield from _stream_subprocess(["main.py", "--train-lora"])

def run_generate_paper():
    yield from _stream_subprocess(["papers/aria_paper.py"])

def run_metacognition_eval():
    yield from _stream_subprocess(["eval/metacognition_eval.py"])

def run_generate_synthetic(n):
    try:
        n = max(50, int(n))
    except Exception:
        n = 50
    yield from _stream_subprocess(["eval/metacognition/generate.py", "--samples", str(n), "--resume"])


def get_leaderboard() -> str:
    if not RESULTS_JSON.exists():
        return "_No results yet. Run an experiment to generate `experiment_results_full.json`._"
    try:
        data = json.loads(RESULTS_JSON.read_text())
        per_model = data.get("per_model", {})
        rows = []
        for model, dims in per_model.items():
            w = dims.get("weighted", {})
            rows.append((model, w.get("mean"), w.get("ci_lo"), w.get("ci_hi"), w.get("n")))
        rows.sort(key=lambda r: (r[1] is not None, r[1] or 0), reverse=True)
        md = ["| Rank | Model | Weighted score | 95% CI | n |", "|---|---|---|---|---|"]
        for i, (model, mean, lo, hi, n) in enumerate(rows, 1):
            mean_s = f"{mean:.3f}" if mean is not None else "—"
            ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None and hi is not None else "—"
            md.append(f"| {i} | `{model}` | **{mean_s}** | {ci} | {n or '—'} |")
        return "\n".join(md)
    except Exception as e:
        return f"_Could not read results: {e}_"


def get_personalization_audit() -> str:
    """Explain the most recent intervention decision for research review."""
    if _aria_agent is None:
        return "_ARIA is still loading._"
    meta = getattr(_aria_agent, "_last_coaching_meta", {}) or {}
    if not meta:
        return "_Complete one learning turn to see the decision audit._"
    signature = meta.get("signature", {}) or {}
    learner = meta.get("learner_model", {}) or {}
    mastery_mean = learner.get("mastery_mean")
    skill_states = learner.get("skills", {}) or {}
    skill_lines = []
    for skill, state in list(skill_states.items())[:3]:
        interval = state.get("ci_95", ["?", "?"])
        skill_lines.append(
            f"{skill}: {state.get('mean', 0):.0%} "
            f"(95% interval {interval[0]}–{interval[1]}, "
            f"{state.get('observations', 0)} observations)"
        )
    selected_evidence = {}
    for candidate in meta.get("ranked_candidates", []) or []:
        if (
            candidate.get("strategy") == meta.get("selected_strategy")
            and candidate.get("source") == meta.get("selected_source")
            and candidate.get("valid")
        ):
            selected_evidence = candidate.get("policy_evidence", {}) or {}
            break
    return (
        f"**Selected strategy:** {meta.get('selected_strategy') or 'verified fallback'}  \n"
        f"**Response source:** {meta.get('selected_source') or 'verified'}  \n"
        f"**Valid candidates:** {meta.get('valid_candidate_count', 0)} "
        f"of {meta.get('candidate_count', 0)}  \n"
        f"**Semantic repeats blocked:** {meta.get('semantic_repeats_blocked', 0)}  \n"
        f"**Student-history turns used:** {meta.get('history_turns_used', 0)}  \n"
        f"**Previous intervention outcome:** {meta.get('prior_outcome', 'none')}  \n"
        f"**Student intent:** {meta.get('student_intent', 'not classified')} "
        f"({meta.get('intent_confidence', 0):.0%}; "
        f"{meta.get('intent_model', 'fallback')})  \n"
        f"**Estimated skill mastery:** "
        f"{f'{mastery_mean:.0%}' if isinstance(mastery_mean, (int, float)) else 'not enough evidence'} "
        f"({learner.get('mastery_band', 'unknown')}; uncertainty retained)  \n"
        f"**Skill evidence:** {'; '.join(skill_lines) if skill_lines else 'no observations yet'}  \n"
        f"**Policy evidence:** {selected_evidence.get('observations', 0)} comparable outcomes; "
        f"posterior mean {selected_evidence.get('posterior_mean', 0.5):.0%}; "
        f"95% interval {selected_evidence.get('ci_95', [0.06, 0.94])}  \n"
        f"**State status:** hypothesis, not a diagnosis  \n"
        f"**Decision fingerprint:** `{signature.get('fingerprint', 'not recorded')}`"
    )


def get_figures():
    if not FIGURES_DIR.exists():
        return []
    imgs = sorted(str(p) for p in FIGURES_DIR.glob("*.png"))
    return imgs


def get_metacog_eval() -> str:
    if not METACOG_EVAL_JSON.exists():
        return "_No metacognition eval results yet. Click **Run metacognition eval**._"
    try:
        d = json.loads(METACOG_EVAL_JSON.read_text())
    except Exception as e:
        return f"_Could not read results: {e}_"
    lines = []
    sd = d.get("state_detection", {})
    if sd:
        lines.append(f"### State detection\n**Accuracy: {sd.get('accuracy', 0):.1%}** (n={sd.get('n', '?')})\n")
        per = sd.get("per_state", {})
        if per:
            lines.append("| State | Precision | Recall | F1 | n |")
            lines.append("|---|---|---|---|---|")
            for st, m in per.items():
                lines.append(
                    f"| {st} | {m.get('precision', 0):.2f} | {m.get('recall', 0):.2f} "
                    f"| {m.get('f1', 0):.2f} | {m.get('support', '?')} |"
                )
    ia = d.get("intervention_appropriateness", {})
    if ia:
        score = ia.get("mean_score", ia.get("appropriateness", ia.get("accuracy")))
        lines.append("\n### Intervention appropriateness")
        lines.append(f"Mean appropriateness: **{score:.2f}**" if isinstance(score, (int, float))
                     else f"```\n{json.dumps(ia, indent=1)[:400]}\n```")
    te = d.get("transition_effectiveness", {})
    if te:
        lines.append("\n### Transition effectiveness")
        rate = te.get("improvement_rate", te.get("rate"))
        lines.append(f"Improved-state rate: **{rate:.1%}**" if isinstance(rate, (int, float))
                     else f"```\n{json.dumps(te, indent=1)[:400]}\n```")

    # Metacognitive-development metrics (4-6).
    td = d.get("transfer_detection", {})
    if td and not td.get("error"):
        lines.append("\n### Transfer detection (self-initiated metacognition)")
        lines.append(f"Precision **{td.get('precision', 0):.2f}** · Recall **{td.get('recall', 0):.2f}** "
                     f"· F1 **{td.get('f1', 0):.2f}** · type-accuracy "
                     f"{td.get('metacognitive_type_accuracy', 0):.2f} (n={td.get('n', '?')})")
    cv = d.get("calibration_validity", {})
    if cv:
        ce = cv.get("mean_calibration_error")
        lines.append("\n### Calibration validity")
        lines.append(f"Mean calibration error **{ce:.3f}** · overconf {cv.get('overconfidence_rate', 0):.2f} "
                     f"· underconf {cv.get('underconfidence_rate', 0):.2f} · "
                     f"computation **{'valid' if cv.get('valid') else 'INVALID'}**"
                     if isinstance(ce, (int, float)) else f"```\n{json.dumps(cv, indent=1)[:300]}\n```")
    tv = d.get("timing_validity", {})
    if tv:
        lines.append("\n### Intervention-timing validity")
        mr = tv.get("match_rate")
        lines.append(f"Optimal-timing match rate **{mr:.0%}** over {tv.get('n_scenarios', '?')} scenarios · "
                     f"detection **{'valid' if tv.get('valid') else 'INVALID'}**"
                     if isinstance(mr, (int, float)) else f"```\n{json.dumps(tv, indent=1)[:300]}\n```")
    return "\n".join(lines) or "_Results present but empty._"


def _find_paper() -> Optional[Path]:
    candidates = []
    for base in (ARIA_ROOT / "papers", ARIA_ROOT / "reports" / "output",
                 ARIA_ROOT / "reports", DATA_DIR):
        if base.exists():
            candidates += list(base.glob("*.md"))
    candidates = [c for c in candidates if "readme" not in c.name.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_paper_status() -> str:
    p = _find_paper()
    if p is None:
        return "_No paper draft found yet. Click **Regenerate paper**._"
    text = p.read_text(errors="ignore")
    words = len(text.split())
    when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    first_para = ""
    for block in text.split("\n\n"):
        block = block.strip()
        if block and not block.startswith("#") and len(block) > 60:
            first_para = block
            break
    return (
        f"**File:** `{p.name}`  \n"
        f"**Last generated:** {when}  \n"
        f"**Word count:** {words:,}\n\n"
        f"**Opening:**\n\n> {first_para[:600]}"
    )


# ==================================================================
#  SETTINGS
# ==================================================================

DIAGNOSIS_OPTIONS = ["ADHD-Combined", "ADHD-Inattentive", "ADHD-Hyperactive",
                     "ADHD+Dyslexia", "ADHD+ASD", "Other"]
STYLE_OPTIONS = ["visual", "analogy", "step-by-step", "kinesthetic", "mixed"]


def _load_settings_values():
    p = _profile()
    diag = [d for d in p.get("diagnosis", [])]
    # map stored (lowercased) diagnosis back to option labels where possible
    diag_labels = []
    for d in diag:
        match = next((o for o in DIAGNOSIS_OPTIONS if o.lower() == d.lower()), None)
        diag_labels.append(match or d)
    style = p.get("learning_style", "mixed").replace("_", "-")
    return (
        p.get("name", ""),
        diag_labels,
        style if style in STYLE_OPTIONS else "mixed",
        ", ".join(p.get("subjects", [])),
        ", ".join(p.get("goals", [])),
        ", ".join(str(h) for h in p.get("study_hours", [])),
        p.get("response_length", "medium"),
        bool(p.get("think_aloud_default", False)),
        p.get("nudge_frequency", "medium"),
        bool(p.get("nightly_loop", True)),
    )


def save_settings(name, diagnosis, style, subjects, goals, peak_hours,
                  resp_len, think_default, nudge_freq, nightly_on):
    p = _profile()
    p["name"] = name.strip() or p.get("name", "friend")
    p["diagnosis"] = [d.strip().lower() for d in (diagnosis or [])]
    p["learning_style"] = style.replace("-", "_")
    p["subjects"] = [s.strip() for s in subjects.split(",") if s.strip()] or ["general learning"]
    p["goals"] = [g.strip() for g in goals.split(",") if g.strip()]
    hours = []
    for h in peak_hours.replace(",", " ").split():
        try:
            hh = int(h)
            if 0 <= hh <= 23:
                hours.append(hh)
        except ValueError:
            pass
    p["study_hours"] = hours
    p["response_length"] = resp_len
    p["think_aloud_default"] = bool(think_default)
    p["nudge_frequency"] = nudge_freq
    p["nightly_loop"] = bool(nightly_on)

    try:
        from memory.graph import save_profile
        save_profile(p)
    except Exception as e:
        return f"⚠️ Saved in memory but file write failed: {e}"

    # apply scheduler toggles live
    _apply_scheduler_toggles(nudge_freq, nightly_on)
    return f"✅ Saved. ARIA will use these from your next message on."


def _apply_scheduler_toggles(nudge_freq, nightly_on):
    if _nudge_scheduler is None:
        return
    try:
        sched = _nudge_scheduler.scheduler
        if nudge_freq == "off":
            sched.pause_job("nudge_check")
        else:
            sched.resume_job("nudge_check")
        if nightly_on:
            sched.resume_job("nightly_loop")
        else:
            sched.pause_job("nightly_loop")
    except Exception:
        pass


def run_nightly_now():
    if _nudge_scheduler is None:
        return "Scheduler not running."
    try:
        _nudge_scheduler.run_nightly_now()
        return "🌙 Nightly loop triggered — results in ~30s (check tomorrow's recap)."
    except Exception as e:
        return f"Error: {e}"


def get_data_stats() -> str:
    p = _profile()
    convos = topics = 0
    try:
        convos = _aria_agent.vs.conversations.count() if _aria_agent else 0
    except Exception:
        pass
    try:
        topics = len(_aria_agent.lg.all_topics_summary()) if _aria_agent else 0
    except Exception:
        pass
    days = 0
    if p.get("created_at"):
        try:
            days = max(0, (datetime.now() - datetime.fromisoformat(p["created_at"])).days)
        except Exception:
            pass
    return (
        f"**Total conversations:** {convos}  \n"
        f"**Total topics tracked:** {topics}  \n"
        f"**Days active:** {days}"
    )


def get_graph_stats() -> str:
    if _aria_agent is None:
        return "_Graph unavailable._"
    g = _aria_agent.lg.graph
    lines = [
        f"**Nodes (topics):** {g.number_of_nodes()}",
        f"**Edges (links):** {g.number_of_edges()}",
    ]
    try:
        struggling = _aria_agent.lg.struggling_topics()
        if struggling:
            lines.append("**Struggling:** " + ", ".join(f"{t['topic']} ({t['confidence']:.0%})" for t in struggling[:5]))
        peak = _aria_agent.lg.peak_focus_hours()
        if peak:
            lines.append("**Peak focus hours:** " + ", ".join(f"{h}:00" for h in peak))
    except Exception:
        pass
    return "\n\n".join(lines)


def export_data():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / "aria_export.zip"
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for pattern in ("*.json", "*.jsonl", "*.pkl"):
                for f in DATA_DIR.glob(pattern):
                    zf.write(f, f.name)
            meta_dir = DATA_DIR / "metacognition"
            if meta_dir.exists():
                for f in meta_dir.glob("*.json"):
                    zf.write(f, f"metacognition/{f.name}")
        return str(out)
    except Exception:
        return None


def reset_everything(confirm: bool):
    if not confirm:
        return "☑️ Tick the confirmation box first — this cannot be undone."
    removed = []
    try:
        for name in ("learning_graph.pkl", "srs_state.json", "real_sessions.jsonl",
                     "nightly_log.jsonl", "initiation_triggers.jsonl"):
            f = DATA_DIR / name
            if f.exists():
                f.unlink()
                removed.append(name)
        for sub in ("chroma", "metacognition", "checkpoints", "emotion_logs"):
            d = DATA_DIR / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(exist_ok=True)
                removed.append(sub + "/")
    except Exception as e:
        return f"Partial reset — error: {e}"
    return (
        "🧹 Cleared: " + ", ".join(removed) +
        ".\n\n**Restart ARIA** (quit and relaunch) for a completely fresh store."
    )


# ==================================================================
#  THEME + CSS
# ==================================================================

ARIA_THEME = gr.themes.Base(
    primary_hue="yellow",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#f6f7f3",
    body_background_fill_dark="#f6f7f3",
    body_text_color="#10213b",
    body_text_color_dark="#10213b",
    background_fill_primary="#fffdf8",
    background_fill_primary_dark="#fffdf8",
    background_fill_secondary="#eef6fb",
    background_fill_secondary_dark="#eef6fb",
    block_background_fill="#fffdf8",
    block_background_fill_dark="#fffdf8",
    block_border_color="#cbd7e3",
    block_label_background_fill="#fffdf8",
    block_title_text_color="#10213b",
    border_color_primary="#cbd7e3",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#b9c8d8",
    button_primary_background_fill="#f4d01f",
    button_primary_background_fill_hover="#ffdf3a",
    button_primary_text_color="#10213b",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color="#10213b",
    color_accent_soft="#fff5ad",
)

CUSTOM_CSS = """
:root {
  --aria-ink: #10213b;
  --aria-ink-2: #34455d;
  --aria-muted: #68788d;
  --aria-paper: #fffdf8;
  --aria-canvas: #f6f7f3;
  --aria-yellow: #f4d01f;
  --aria-cyan: #dceff9;
  --aria-mint: #dff3e9;
  --aria-coral: #f7dfd8;
  --aria-rule: #cbd7e3;
  --aria-shadow: 0 18px 44px rgba(16, 33, 59, 0.08);
}

html, body {
  background: var(--aria-canvas) !important;
}

body, .gradio-container {
  color: var(--aria-ink) !important;
  font-family: "Avenir Next", "Segoe UI", system-ui, sans-serif !important;
}

.gradio-container {
  --body-text-color: var(--aria-ink);
  --body-text-color-subdued: var(--aria-ink-2);
  --block-info-text-color: var(--aria-ink-2);
  --block-label-text-color: var(--aria-ink);
  --input-placeholder-color: #52657b;
  max-width: none !important;
  min-height: 100vh;
  padding: 0 0 64px !important;
  background:
    radial-gradient(circle at 88% 5%, rgba(220, 239, 249, 0.82), transparent 24rem),
    var(--aria-canvas) !important;
}

.gradio-container > .main,
.gradio-container > div {
  max-width: 1240px;
  margin-inline: auto;
}

.aria-product-header {
  display: flex;
  min-height: 64px;
  align-items: center;
  gap: 24px;
  margin: 0 0 16px;
  padding: 0 4px;
  border-bottom: 1px solid var(--aria-rule);
}

.aria-product-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.aria-product-brand-copy {
  display: grid;
  gap: 2px;
}

.aria-product-brand-copy strong {
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.aria-product-brand-copy span,
.aria-eyebrow,
.aria-problem-label,
.aria-response-label,
.aria-state-result > span,
.aria-state-history > b {
  color: var(--aria-muted);
  font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.aria-face {
  position: relative;
  width: 48px;
  height: 48px;
  flex: 0 0 auto;
}

.aria-face__shape {
  position: absolute;
  inset: 5px;
  background: var(--aria-yellow);
  border: 3px solid var(--aria-ink);
  border-radius: 48% 52% 45% 55%;
  box-shadow: 2px 3px 0 var(--aria-ink);
  transform: rotate(-7deg);
}

.aria-face__shape::before,
.aria-face__shape::after {
  position: absolute;
  top: 34%;
  width: 5px;
  height: 8px;
  background: var(--aria-ink);
  border-radius: 999px;
  content: "";
}

.aria-face__shape::before { left: 27%; }
.aria-face__shape::after { right: 27%; }

.aria-face__smile {
  position: absolute;
  inset: 57% 30% auto;
  height: 8px;
  border-bottom: 3px solid var(--aria-ink);
  border-radius: 0 0 999px 999px;
}

#main-navigation {
  background: transparent !important;
  border: 0 !important;
}

#main-navigation > .tab-nav {
  gap: 6px;
  margin-bottom: 18px;
  padding: 5px;
  background: #e8edf0;
  border: 1px solid var(--aria-rule);
  border-radius: 14px;
}

#main-navigation > .tab-nav button {
  min-height: 42px;
  color: var(--aria-ink-2);
  background: transparent;
  border: 0;
  border-radius: 10px;
  font-weight: 700;
}

#main-navigation > .tab-nav button.selected {
  color: var(--aria-ink);
  background: var(--aria-paper);
  box-shadow: 0 2px 8px rgba(16, 33, 59, 0.08);
}

.gradio-container fieldset label {
  color: var(--aria-ink) !important;
  background: var(--aria-paper) !important;
  border-color: var(--aria-rule) !important;
}

.gradio-container fieldset label:has(input:checked) {
  background: var(--aria-yellow) !important;
  border-color: var(--aria-ink) !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  border-color: var(--aria-rule) !important;
  color: var(--aria-ink) !important;
}

.gradio-container .form {
  background: transparent !important;
}

.gradio-container label,
.gradio-container span[data-testid="block-info"] {
  color: var(--aria-ink-2) !important;
  opacity: 1 !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #52657b !important;
  opacity: 1 !important;
}

.gradio-container .group,
.aria-surface {
  background: transparent !important;
  border: 0 !important;
}

.aria-surface > .styler {
  color: var(--aria-ink) !important;
  background: transparent !important;
}

.aria-panel {
  padding: 20px !important;
  background: var(--aria-paper) !important;
  border: 1px solid var(--aria-rule) !important;
  border-radius: 20px !important;
  box-shadow: var(--aria-shadow);
}

.aria-learning-column {
  gap: 12px !important;
}

.aria-secondary-action {
  align-self: flex-start;
  width: auto !important;
  min-width: 0 !important;
}

.aria-secondary-action button,
button.aria-secondary-action {
  min-height: 36px !important;
  padding-inline: 14px !important;
  font-size: 0.8rem !important;
}

#think-submit {
  margin-top: 2px;
}

#learn-chatbot {
  height: 460px;
  background: #fbfcfa !important;
  border: 1px solid var(--aria-rule) !important;
  border-radius: 18px !important;
}

#learn-chatbot .message {
  border-radius: 16px !important;
  box-shadow: none !important;
}

#learn-chatbot .message.user {
  color: var(--aria-ink) !important;
  background: var(--aria-cyan) !important;
}

#learn-chatbot .message.bot {
  color: var(--aria-ink) !important;
  background: #fff5ad !important;
}

.gradio-container textarea,
.gradio-container input {
  color: var(--aria-ink) !important;
  background: #ffffff !important;
  border-color: #b9c8d8 !important;
  border-radius: 14px !important;
  caret-color: var(--aria-ink) !important;
  cursor: text !important;
  pointer-events: auto !important;
  position: relative;
  z-index: 2;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: #2f78bd !important;
  box-shadow: 0 0 0 3px rgba(47, 120, 189, 0.18) !important;
  outline: none !important;
}

.gradio-container button {
  min-height: 42px;
  border-radius: 12px !important;
  font-weight: 700 !important;
}

.gradio-container button.primary {
  color: var(--aria-ink) !important;
  background: var(--aria-yellow) !important;
  border: 1px solid var(--aria-ink) !important;
  box-shadow: 3px 3px 0 var(--aria-ink);
}

.gradio-container button.primary:hover {
  background: #ffdf3a !important;
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 var(--aria-ink);
}

.quick-btn button, .quick-btn {
  min-height: 38px !important;
  font-size: 0.82rem !important;
}

#nudge-banner {
  min-height: 20px;
  padding: 10px 14px;
  color: var(--aria-ink-2);
  background: var(--aria-cyan);
  border: 1px solid #b8d9eb;
  border-radius: 12px;
  font-size: 0.88rem;
}

.aria-focus-display {
  padding: 8px 0;
  font-size: 1.25rem;
  font-weight: 800;
  text-align: center;
}

.aria-confidence-bar { margin: 8px 0; }
.aria-confidence-label {
  margin-bottom: 4px;
  color: var(--aria-muted);
  font-size: 0.78rem;
}
.aria-confidence-track {
  height: 10px;
  overflow: hidden;
  background: #e7edf2;
  border-radius: 999px;
}

#think-problem {
  padding: 22px !important;
  background: var(--aria-paper) !important;
  border: 1px solid var(--aria-rule) !important;
  border-radius: 20px !important;
  box-shadow: var(--aria-shadow);
}

.aria-problem-label {
  margin-bottom: 12px;
  color: #2f78bd;
}

.aria-problem-card {
  padding: 20px;
  background: var(--aria-cyan);
  border: 1px solid #b8d9eb;
  border-radius: 16px;
  font-size: 1.2rem;
  font-weight: 700;
  line-height: 1.55;
}

.aria-problem-prompt {
  margin: 14px 0 0;
  color: var(--aria-ink-2);
  font-size: 0.9rem;
}

#think-input textarea {
  min-height: 148px !important;
  font-size: 1rem;
  line-height: 1.55;
}

.aria-inline-status {
  color: var(--aria-muted);
  font-size: 0.86rem;
}

#thinking-state-card,
#aria-response-panel {
  padding: 20px !important;
  background: var(--aria-paper) !important;
  border: 1px solid var(--aria-rule) !important;
  border-radius: 20px !important;
}

#thinking-state-card {
  background: var(--aria-cyan) !important;
}

.aria-state-empty {
  display: grid;
  min-height: 72px;
  place-content: center;
  gap: 5px;
  color: var(--aria-muted);
  text-align: center;
}

.aria-state-empty strong {
  color: var(--aria-ink);
  font-size: 1.05rem;
}

.aria-state-result {
  display: grid;
  gap: 8px;
  padding-bottom: 16px;
}

.aria-state-result strong {
  font-size: 2.4rem;
  letter-spacing: -0.05em;
}

.aria-state-result p {
  margin: 0;
  color: var(--aria-ink-2);
}

.aria-state-history {
  padding-top: 14px;
  border-top: 1px solid #b8d9eb;
}

.aria-state-history > div {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 9px;
}

.aria-state-history span {
  padding: 5px 9px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid #b8d9eb;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
}

.aria-response-heading {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.aria-response-heading strong {
  font-size: 1rem;
}

.aria-response-heading span {
  display: block;
  color: var(--aria-muted);
  font-size: 0.76rem;
}

.aria-response-card {
  padding: 18px;
  background: #fff8c7;
  border: 1px solid #e2c54b;
  border-radius: 16px;
}

.aria-response-label {
  margin-bottom: 8px;
  color: #6c5700;
}

.aria-response-question {
  font-size: 1.15rem;
  font-weight: 700;
  line-height: 1.45;
}

.aria-response-note {
  margin-top: 14px;
  color: var(--aria-muted);
  font-size: 0.76rem;
}

.aria-metric-card {
  flex: 1;
  min-width: 150px;
  padding: 16px;
  background: var(--aria-paper);
  border: 1px solid var(--aria-rule);
  border-radius: 16px;
}

.aria-metric-card > div {
  color: var(--aria-muted);
  font-size: 0.78rem;
}

.aria-metric-card strong {
  display: block;
  margin-top: 6px;
  color: var(--aria-ink);
  font-size: 1.8rem;
}

.aria-metric-card small {
  display: block;
  margin-top: 8px;
  color: var(--aria-muted);
  font-size: 0.72rem;
}

.aria-topic-card {
  width: 210px;
  padding: 14px;
  background: var(--aria-paper);
  border: 1px solid var(--aria-rule);
  border-radius: 16px;
}

.aria-topic-track {
  height: 9px;
  margin: 9px 0;
  overflow: hidden;
  background: #e7edf2;
  border-radius: 999px;
}

.aria-topic-meta {
  color: var(--aria-muted);
  font-size: 0.74rem;
  line-height: 1.5;
}

#research-log textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
  font-size: 0.8rem !important;
  color: #d9f5e8 !important;
  background: var(--aria-ink) !important;
}

.gradio-container table {
  overflow: hidden;
  border: 1px solid var(--aria-rule);
  border-radius: 12px;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--aria-ink) !important;
  letter-spacing: -0.035em;
}

.gradio-container p,
.gradio-container li {
  line-height: 1.6;
}

footer { display:none !important; }

@media (max-width: 760px) {
  .gradio-container { padding-inline: 12px !important; }
  .aria-product-header { min-height: 68px; }
  #main-navigation > .tab-nav { overflow-x: auto; }
}
"""


# ==================================================================
#  BUILD UI
# ==================================================================

def build_ui() -> gr.Blocks:
    initial_problem, initial_text, initial_state, initial_response = new_think_problem()

    with gr.Blocks(title="ARIA — Metacognitive Learning") as demo:
        gr.HTML(
            """
            <header class="aria-product-header">
              <div class="aria-product-brand">
                <div class="aria-face" aria-hidden="true">
                  <span class="aria-face__shape"><i class="aria-face__smile"></i></span>
                </div>
                <div class="aria-product-brand-copy">
                  <strong>ARIA</strong>
                  <span>Learning tool</span>
                </div>
              </div>
            </header>
            """
        )

        with gr.Tabs(elem_id="main-navigation") as main_tabs:

            # ========================================================
            # TAB 1 — LEARN
            # ========================================================
            with gr.Tab("Learn", id="tab_learn"):
                with gr.Group(
                    elem_classes=["aria-surface"],
                ):
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["aria-learning-column"]):
                            with gr.Row():
                                practice_subject = gr.Dropdown(
                                    choices=["Math", "English"],
                                    value="Math",
                                    label="Subject",
                                    scale=2,
                                    allow_custom_value=False,
                                )
                                new_problem_btn = gr.Button(
                                    "Try another problem",
                                    size="sm",
                                    variant="secondary",
                                    scale=1,
                                    elem_classes=["aria-secondary-action"],
                                )
                            think_problem = gr.Markdown(
                                initial_problem,
                                elem_id="think-problem",
                            )
                            think_input = gr.Textbox(
                                label="Work it out here",
                                value=initial_text,
                                placeholder="What would you try first?",
                                lines=6,
                                elem_id="think-input",
                                interactive=True,
                                autofocus=True,
                                html_attributes={
                                    "autocomplete": "off",
                                    "spellcheck": "true",
                                },
                            )
                            think_submit = gr.Button(
                                "Continue",
                                variant="primary",
                                elem_id="think-submit",
                            )
                        with gr.Column(scale=4):
                            gr.HTML(
                                """
                                <div class="aria-response-heading">
                                  <div class="aria-face" aria-hidden="true">
                                    <span class="aria-face__shape"><i class="aria-face__smile"></i></span>
                                  </div>
                                  <div><strong>ARIA</strong><span>Your next move</span></div>
                                </div>
                                """
                            )
                            think_response = gr.Markdown(
                                initial_response,
                                elem_id="aria-response-panel",
                            )
                            think_state = gr.HTML(
                                initial_state,
                                elem_id="thinking-state-card",
                            )

            # ========================================================
            # TAB 2 — MY PROGRESS
            # ========================================================
            with gr.Tab("My Progress", id="tab_progress"):
                gr.Markdown("## This week")
                week_cards = gr.HTML("_Loading…_")
                refresh_progress_btn = gr.Button("Refresh progress", size="sm")

                gr.Markdown("## Metacognitive development\n"
                            "_Are you building the skills — not just answering prompts?_")
                dev_cards = gr.HTML("_Loading…_")
                with gr.Row():
                    img_transfer = gr.Image(label="Transfer", show_label=False, height=250)
                    img_calibration = gr.Image(label="Calibration", show_label=False, height=250)
                img_timing = gr.Image(label="Intervention timing", show_label=False, height=300)
                gr.Markdown("### This week's insight")
                metacog_insight = gr.Markdown("_Loading…_")
                dev_insight_btn = gr.Button("Generate metacognition insight", size="sm")

                gr.Markdown("## Your mind over time")
                with gr.Row():
                    img_planning = gr.Image(label="Planning ratio", show_label=False, height=250)
                    img_states = gr.Image(label="State distribution", show_label=False, height=250)
                with gr.Row():
                    img_frustration = gr.Image(label="Frustration", show_label=False, height=250)
                    img_recovery = gr.Image(label="Recovery speed", show_label=False, height=250)

                gr.Markdown("## Topic map")
                topic_map = gr.HTML("_Loading…_")
                with gr.Row():
                    topic_dropdown = gr.Dropdown(label="Jump into a focused session", choices=[], scale=3)
                    focus_topic_btn = gr.Button("Start focused session", variant="primary", scale=1)

                gr.Markdown("## Insights")
                insights_md = gr.Markdown("_Loading…_")
                gen_insights_btn = gr.Button("Generate insights", size="sm")

            # ========================================================
            # TAB 3 — RESEARCH
            # ========================================================
            with gr.Tab("Research", id="tab_research"):
                gr.Markdown(
                    "## Research tools\n"
                    "These controls evaluate ARIA itself. Students do not need them while learning.\n\n"
                    "- **Quick experiment:** checks a small sample for obvious failures.\n"
                    "- **Full experiment:** runs the complete comparison used in the study.\n"
                    "- **Learning curve:** tests how performance changes as training data grows.\n"
                    "- **LoRA training:** fine-tunes the research model on ARIA examples.\n"
                    "- **Paper draft:** turns the latest saved results into a draft report."
                )
                with gr.Row():
                    quick_btn = gr.Button("Quick experiment (20)")
                    full_btn = gr.Button("Full experiment (270)")
                    lc_btn = gr.Button("Learning-curve sweep")
                    lora_btn = gr.Button("LoRA training")
                    paper_btn = gr.Button("Generate paper draft", variant="primary")
                research_log = gr.Textbox(label="Experiment output", lines=14, elem_id="research-log",
                                          interactive=False)

                gr.Markdown("## Personalization audit")
                gr.Markdown(
                    "Shows why ARIA selected its most recent intervention. "
                    "This is for research review, not part of the student lesson."
                )
                personalization_audit = gr.Markdown(
                    "_Complete one learning turn to see the decision audit._"
                )
                refresh_audit_btn = gr.Button("Refresh decision audit", size="sm")

                gr.Markdown("## Current results")
                leaderboard = gr.Markdown("_Loading…_")
                figures_gallery = gr.Gallery(label="Figures", columns=3, height=380, show_label=True)
                refresh_results_btn = gr.Button("Reload results and figures", size="sm")

                gr.Markdown("## Metacognition eval")
                with gr.Row():
                    metacog_btn = gr.Button("Run metacognition eval")
                    syn_count = gr.Number(value=50, label="Synthetic samples / state", precision=0, scale=1)
                    syn_btn = gr.Button("Generate synthetic data")
                metacog_results = gr.Markdown("_Loading…_")

                gr.Markdown("## Paper status")
                paper_status = gr.Markdown("_Loading…_")
                regen_paper_btn = gr.Button("Refresh paper status", size="sm")

            # ========================================================
            # TAB 4 — SETTINGS
            # ========================================================
            with gr.Tab("Settings", id="tab_settings"):
                gr.Markdown("## Your profile")
                with gr.Row():
                    set_name = gr.Textbox(label="Name")
                    set_style = gr.Dropdown(STYLE_OPTIONS, label="Learning style")
                set_diagnosis = gr.Dropdown(DIAGNOSIS_OPTIONS, label="Diagnosis", multiselect=True)
                set_subjects = gr.Textbox(label="Subjects (comma-separated)")
                set_goals = gr.Textbox(label="Goals (comma-separated)")
                set_peak = gr.Textbox(label="Peak study hours (0–23, comma or space separated)")

                gr.Markdown("## ARIA behaviour")
                set_resp = gr.Radio(
                    [("Short (2 sentences)", "short"), ("Medium (3)", "medium"), ("Detailed (5)", "detailed")],
                    label="Response length", value="medium",
                )
                set_think_default = gr.Checkbox(label="Open in Think-Aloud mode by default")
                set_nudge = gr.Radio(
                    [("Off", "off"), ("Low (1/day)", "low"), ("Medium (3/day)", "medium")],
                    label="Nudge frequency", value="medium",
                )
                set_nightly = gr.Checkbox(label="Nightly learning loop", value=True)
                with gr.Row():
                    save_settings_btn = gr.Button("Save settings", variant="primary")
                    nightly_now_btn = gr.Button("Run nightly loop now")
                settings_status = gr.Markdown("")

                gr.Markdown("## Data")
                data_stats = gr.Markdown("_Loading…_")
                with gr.Row():
                    export_btn = gr.Button("Export my data")
                    graph_btn = gr.Button("View raw learning graph")
                export_file = gr.File(label="Your export", visible=True)
                graph_stats = gr.Markdown("")
                gr.Markdown("### Danger zone")
                reset_confirm = gr.Checkbox(label="I understand this permanently deletes all my ARIA data")
                reset_btn = gr.Button("Reset everything", variant="stop")
                reset_status = gr.Markdown("")

            # ========================================================
            # EVENT WIRING
            # ========================================================

            focus_visible_input_js = """
            () => {
              window.setTimeout(() => {
                const visible = document.querySelector('#think-input textarea');
                if (visible) visible.focus();
              }, 80);
            }
            """

            # --- think aloud ---
            new_problem_btn.click(
                new_think_problem,
                inputs=[practice_subject],
                outputs=[think_problem, think_input, think_state, think_response],
            ).then(None, js=focus_visible_input_js)
            think_submit.click(
                submit_think_aloud,
                [think_input],
                [think_state, think_response, think_input],
                show_progress="minimal",
            ).then(None, js=focus_visible_input_js)

            # --- progress ---
            def _refresh_progress():
                return (get_week_cards(), chart_planning_ratio(), chart_state_distribution(),
                        chart_frustration_trend(), chart_recovery_trend(),
                        get_topic_map(), get_topic_choices())
            refresh_progress_btn.click(
                _refresh_progress,
                outputs=[week_cards, img_planning, img_states, img_frustration,
                         img_recovery, topic_map, topic_dropdown],
            )

            # --- metacognitive-development section ---
            def _refresh_dev():
                return (get_metacog_dev_cards(), chart_transfer(),
                        chart_calibration(), chart_timing_heatmap())
            refresh_progress_btn.click(
                _refresh_dev, outputs=[dev_cards, img_transfer, img_calibration, img_timing])
            dev_insight_btn.click(get_metacog_insight, outputs=metacog_insight)

            gen_insights_btn.click(get_insights, outputs=insights_md)
            focus_topic_btn.click(
                start_topic_practice,
                [topic_dropdown],
                [think_problem, think_input, think_state, think_response, main_tabs],
            )

            # --- research ---
            quick_btn.click(run_quick_experiment, outputs=research_log)
            full_btn.click(run_full_experiment, outputs=research_log)
            lc_btn.click(run_learning_curve, outputs=research_log)
            lora_btn.click(run_lora_training, outputs=research_log)
            paper_btn.click(run_generate_paper, outputs=research_log)
            paper_btn.click(get_paper_status, outputs=paper_status)
            metacog_btn.click(run_metacognition_eval, outputs=research_log)
            metacog_btn.click(get_metacog_eval, outputs=metacog_results)
            syn_btn.click(run_generate_synthetic, [syn_count], research_log)

            def _reload_results():
                return get_leaderboard(), get_figures()
            refresh_results_btn.click(_reload_results, outputs=[leaderboard, figures_gallery])
            refresh_audit_btn.click(
                get_personalization_audit,
                outputs=personalization_audit,
            )
            regen_paper_btn.click(get_paper_status, outputs=paper_status)

            # --- settings ---
            save_settings_btn.click(
                save_settings,
                [set_name, set_diagnosis, set_style, set_subjects, set_goals, set_peak,
                 set_resp, set_think_default, set_nudge, set_nightly],
                settings_status,
            )
            nightly_now_btn.click(run_nightly_now, outputs=settings_status)
            export_btn.click(export_data, outputs=export_file)
            graph_btn.click(get_graph_stats, outputs=graph_stats)
            reset_btn.click(reset_everything, [reset_confirm], reset_status)

            # --- initial focus ---
            demo.load(None, js=focus_visible_input_js)
            demo.load(_refresh_dev,
                      outputs=[dev_cards, img_transfer, img_calibration, img_timing])
            demo.load(get_leaderboard, outputs=leaderboard)
            demo.load(get_personalization_audit, outputs=personalization_audit)
            demo.load(get_figures, outputs=figures_gallery)
            demo.load(get_metacog_eval, outputs=metacog_results)
            demo.load(get_paper_status, outputs=paper_status)
            demo.load(get_data_stats, outputs=data_stats)
            demo.load(lambda: _load_settings_values(),
                      outputs=[set_name, set_diagnosis, set_style, set_subjects, set_goals,
                               set_peak, set_resp, set_think_default, set_nudge, set_nightly])

    return demo
