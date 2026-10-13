"""Minimal Gradio page for the summit demo: chat + a *visible* ARIA state.

    python -m ui.demo_app                 # templates only (default), http://127.0.0.1:7861
    python -m ui.demo_app --llm-path      # also let the gated LLM path answer free-form questions

The point of the demo is timing, so the state badge is the main thing on screen:
    WATCHING        ARIA read the student's work and chose to say nothing
    STEPPING IN     ARIA is speaking, with the policy rule that decided it
Everything runs locally; there are no network calls (the LLM path is off unless asked for).
"""
from __future__ import annotations

import argparse
import html

import gradio as gr

from agent.demo_bank import load_problems
from agent.demo_tutor import DemoSession, DemoTutor

STATE_LABELS = {
    "FLOW": "in flow", "PLANNING": "planning", "INSIGHT": "got it", "STUCK": "stuck", "FRUSTRATED": "frustrated",
    "CONFUSED": "confused", "UNKNOWN": "unclear", "RUSHING": "rushing",
}
RULE_TEXT = {
    "jitai.no_interrupt_positive": "productive work: stay quiet",
    "jitai.no_action": "no reason to interrupt: stay quiet",
    "jitai.insight_transfer": "a good moment to reflect",
    "jitai.initial_negative_support": "student needs support: one small prompt",
    "jitai.repeated_confusion": "stuck for several turns: a smaller step",
    "jitai.frustration_reset": "frustration is building: lower the pressure",
    "jitai.escalate_after_ignored": "earlier nudge was ignored: suggest a reset",
}

CSS = """
.aria-badge{display:flex;align-items:center;gap:14px;padding:16px 20px;border-radius:14px;font-family:system-ui,sans-serif;
  border:2px solid var(--b,#94a3b8);background:var(--bg,#f1f5f9);color:#0f172a}
.aria-badge .dot{width:16px;height:16px;border-radius:50%;background:var(--b,#94a3b8)}
.aria-badge.watching{--b:#2563eb;--bg:#eff6ff}.aria-badge.watching .dot{animation:pulse 1.6s ease-in-out infinite}
.aria-badge.stepping{--b:#d97706;--bg:#fffbeb}
.aria-badge.idle{--b:#94a3b8;--bg:#f1f5f9}
.aria-badge .big{font-weight:700;letter-spacing:.06em;font-size:1.15rem}
.aria-badge .small{font-size:.9rem;opacity:.85}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.25)}}
.aria-chat{display:flex;flex-direction:column;gap:8px;min-height:300px;max-height:380px;overflow:auto;padding:8px;font-family:system-ui,sans-serif}
.aria-chat .msg{padding:10px 14px;border-radius:12px;max-width:80%;line-height:1.35}
.aria-chat .user{align-self:flex-end;background:#e2e8f0}
.aria-chat .aria{align-self:flex-start;background:#fef3c7;border:1px solid #f59e0b}
.aria-chat .quiet{align-self:flex-start;color:#475569;font-style:italic;font-size:.9rem;padding:4px 8px}
"""


def chat_html(history: list) -> str:
    rows = []
    for m in history:
        cls = {"user": "user", "aria": "aria", "quiet": "quiet"}[m["role"]]
        rows.append(f'<div class="msg {cls}">{html.escape(m["content"])}</div>')
    return '<div class="aria-chat">' + "".join(rows) + "</div>"


def badge_html(kind: str, headline: str = "", detail: str = "") -> str:
    labels = {"watching": "WATCHING", "stepping": "STEPPING IN", "idle": "READY"}
    return (f'<div class="aria-badge {kind}"><span class="dot"></span><div>'
            f'<div class="big">{labels[kind]}</div><div class="small">{html.escape(headline)}'
            f'{"<br>" + html.escape(detail) if detail else ""}</div></div></div>')


def make_handlers(tutor: DemoTutor, problems: list):
    ids = {f"{p.prompt}": p.id for p in problems}

    def start(prompt):
        s = tutor.new_session(ids[prompt])
        return s, [], chat_html([]), badge_html("idle", "Type a step or an answer. ARIA will decide whether to speak."), ""

    def send(text, history, session: DemoSession):
        history = list(history or [])
        if session is None:
            return history, None, chat_html(history), badge_html("idle", "Pick a problem first."), ""
        r = session.turn(text)
        if (text or "").strip():
            history.append({"role": "user", "content": text[:500]})
        state = STATE_LABELS.get(r.state, r.state.lower())
        why = RULE_TEXT.get(r.meta["rule_id"], r.meta["rule_id"])
        if r.intervened:
            history.append({"role": "aria", "content": r.reply})
            badge = badge_html("stepping", f"Student seems {state}.", why)
        else:
            history.append({"role": "quiet", "content": r.watch_note})
            badge = badge_html("watching", f"Student seems {state}.", why)
        return history, session, chat_html(history), badge, ""

    def reset(session):
        if session is not None:
            session.reset()
        return [], session, chat_html([]), badge_html("idle", "Reset. Type a step or an answer."), ""

    return start, send, reset


def build_demo_ui(tutor: DemoTutor | None = None) -> gr.Blocks:
    problems = (tutor.problems.values() if tutor else load_problems())
    tutor = tutor or DemoTutor()
    problems = list(tutor.problems.values())
    start, send, reset = make_handlers(tutor, problems)
    with gr.Blocks(title="ARIA — when to help") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown("## ARIA\nOther tutors speak every turn. ARIA decides **whether** to.")
        session, history = gr.State(None), gr.State([])
        problem = gr.Dropdown([p.prompt for p in problems], value=problems[0].prompt, label="Problem")
        badge = gr.HTML(badge_html("idle", "Type a step or an answer. ARIA will decide whether to speak."))
        chat = gr.HTML(chat_html([]))
        box = gr.Textbox(placeholder="Type your next step or answer, then press Enter", show_label=False)
        reset_btn = gr.Button("Reset (new attempt)")
        reset_btn.click(reset, [session], [history, session, chat, badge, box])
        problem.change(start, [problem], [session, history, chat, badge, box])
        box.submit(send, [box, history, session], [history, session, chat, badge, box])
        demo.load(start, [problem], [session, history, chat, badge, box])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-path", action="store_true", help="enable the gated LLM path for free-form questions (default off)")
    ap.add_argument("--llm-model", default="llama3.2:3b")
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    tutor = DemoTutor(llm_path=a.llm_path, llm_model=a.llm_model)
    build_demo_ui(tutor).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
