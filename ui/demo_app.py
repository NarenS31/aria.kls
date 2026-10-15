"""Minimal Gradio page for the summit demo: chat + a *visible* ARIA state.

    python -m ui.demo_app                 # templates only (default), the 8 featured problems, http://127.0.0.1:7861
    python -m ui.demo_app --all-problems  # the full 30-problem bank
    python -m ui.demo_app --llm-path      # also let the gated LLM path answer free-form questions
                                          # (model is loaded once at startup and kept warm; falls back to
                                          #  templates on any timeout/error)
The per-turn log (no student text) goes to data/demo_logs/turns.jsonl unless --no-log.

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
from agent.demo_input import sanitize_input
from agent.demo_summary import RULE_TEXT, STATE_LABELS, headline, summarize
from agent.demo_log import DEFAULT_PATH as LOG_PATH
from agent.demo_tutor import DemoSession, DemoTutor

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
.aria-sum{font-family:system-ui,sans-serif;color:#0f172a;line-height:1.4}
.aria-sum h3{margin:14px 0 6px;font-size:1rem}
.aria-sum .head{font-size:1.05rem;font-weight:600;margin-bottom:4px}
.aria-sum .where{padding:10px 14px;border-radius:10px;background:#fffbeb;border:1px solid #f59e0b;margin:8px 0}
.aria-sum .tiles{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.aria-sum .tile{border:1px solid #cbd5e1;border-radius:10px;padding:8px 14px;min-width:90px;background:#f8fafc}
.aria-sum .tile b{display:block;font-size:1.3rem}
.aria-sum table{border-collapse:collapse;width:100%;font-size:.92rem}
.aria-sum th,.aria-sum td{border-bottom:1px solid #e2e8f0;padding:5px 8px;text-align:left;vertical-align:top}
.aria-sum th{background:#f1f5f9;font-weight:600}
.aria-sum .pill{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid #94a3b8;background:#f1f5f9;font-size:.85rem}
.aria-sum .pill.bad{border-color:#d97706;background:#fffbeb}.aria-sum .pill.good{border-color:#16a34a;background:#f0fdf4}
.aria-sum .note{margin-top:14px;font-size:.82rem;color:#475569}
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


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pill(label: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{_esc(label)}</span>'


def summary_html(summary: dict | None) -> str:
    """Teacher view of one attempt. Everything shown is derived metadata or authored text; no chat transcript."""
    if not summary or not summary["timeline"]:
        return '<div class="aria-sum">No attempt to summarise yet. Pick a problem and try a few steps in the Student tab.</div>'
    pr, c = summary["problem"], summary["counts"]
    outcome = {"solved": f"Solved at turn {summary['solved_at_turn']}", "in_progress": "Not solved yet"}[summary["outcome"]]
    secs = summary.get("elapsed_s")
    when = f", {secs:.0f} s into the attempt at the last turn" if secs is not None else ""
    tiles = "".join(f'<div class="tile"><b>{v}</b>{_esc(k)}</div>' for k, v in (
        ("turns", c["turns"]), ("quiet", c["stayed_quiet"]), ("stepped in", c["stepped_in"]), ("hints given", c["hints_given"])))
    rows = []
    for t in summary["timeline"]:
        bad = t["state"] in ("STUCK", "FRUSTRATED", "CONFUSED")
        act = _pill(t["action_label"], "bad" if t["action"] != "none" else "")
        if t.get("fallback"):
            act += " " + _pill("unreadable input")
        clock = "" if t["t_s"] is None else "%d s" % round(t["t_s"])
        rows.append(f'<tr><td>{t["turn"]}</td><td>{clock}</td>'
                    f'<td>{_pill(t["state_label"], "bad" if bad else "good")}</td><td>{_esc(t["verdict_label"])}</td>'
                    f'<td>{act}</td><td>{_esc(t["why"])}</td></tr>')
    timeline = ('<table><tr><th>Turn</th><th>Time</th><th>ARIA read the student as</th><th>Student\'s message was</th>'
                '<th>ARIA</th><th>Why</th></tr>' + "".join(rows) + "</table>")
    if summary["hints"]:
        def target(h):
            if h["step"]:
                return h["step"]
            return "after the correct answer" if h["type"] == "reflection" else "a free-form question"
        hrows = "".join(f'<tr><td>{h["turn"]}</td><td>{_esc(h["type_label"])}</td><td>{_esc(target(h))}</td></tr>'
                        for h in summary["hints"])
        hints = ('<div>Every time ARIA spoke. Hints and smaller steps count as hints; the rest are encouragement, a pause, or reflection.</div>'
                 f'<table><tr><th>Turn</th><th>What ARIA did</th><th>Step it pointed to</th></tr>{hrows}</table>')
    else:
        hints = "<div>ARIA did not step in during this attempt.</div>"
    srows = []
    for st in summary["steps"]:
        mark = {True: "done", False: "not yet", None: "not tracked"}[st["done"]]
        hinted = f"hinted at turn {', '.join(map(str, st['hinted_at_turns']))}" if st["hinted_at_turns"] else "no hint needed"
        srows.append(f'<tr><td>{st["index"] + 1}</td><td>{_esc(st["step"])}</td><td>{_pill(mark, "good" if st["done"] else "")}</td><td>{_esc(hinted)}</td></tr>')
    steps = ('<table><tr><th>#</th><th>Authored step</th><th>Student produced it</th><th>Hints</th></tr>' + "".join(srows) + "</table>"
             if srows else "<div>This problem has no authored steps.</div>")
    return (f'<div class="aria-sum"><div class="head">{_esc(pr["prompt"])}</div>'
            f'<div>{_esc(pr["level"])} {_esc(pr["topic"])}, attempt {summary["attempt"]}. <b>{_esc(outcome)}</b>{_esc(when)}.</div>'
            f'<div class="where"><b>Where the student got stuck.</b> {_esc(headline(summary))}</div>'
            f'<div class="tiles">{tiles}</div><h3>State timeline</h3>{timeline}<h3>Hints given</h3>{hints}'
            f'<h3>Authored steps</h3>{steps}<div class="note">{_esc(summary["note"])}</div></div>')


def make_summary_handler():
    def show(session):
        return summary_html(summarize(session) if session is not None else None)
    return show


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
        shown = sanitize_input(text)[0]
        if shown:
            history.append({"role": "user", "content": shown[:500]})
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
    show_summary = make_summary_handler()
    with gr.Blocks(title="ARIA — when to help") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown("## ARIA\nOther tutors speak every turn. ARIA decides **whether** to.")
        session, history = gr.State(None), gr.State([])
        problem = gr.Dropdown([p.prompt for p in problems], value=problems[0].prompt, label="Problem")
        with gr.Tabs():
            with gr.Tab("Student"):
                badge = gr.HTML(badge_html("idle", "Type a step or an answer. ARIA will decide whether to speak."))
                chat = gr.HTML(chat_html([]))
                box = gr.Textbox(placeholder="Type your next step or answer, then press Enter", show_label=False)
                reset_btn = gr.Button("Reset (new attempt)")
            with gr.Tab("Teacher summary") as teacher_tab:
                refresh = gr.Button("Refresh summary")
                summary = gr.HTML(summary_html(None))
        reset_btn.click(reset, [session], [history, session, chat, badge, box]).then(show_summary, [session], [summary])
        problem.change(start, [problem], [session, history, chat, badge, box]).then(show_summary, [session], [summary])
        box.submit(send, [box, history, session], [history, session, chat, badge, box]).then(show_summary, [session], [summary])
        demo.load(start, [problem], [session, history, chat, badge, box])
        teacher_tab.select(show_summary, [session], [summary])
        refresh.click(show_summary, [session], [summary])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-path", action="store_true", help="enable the gated LLM path for free-form questions (default off)")
    ap.add_argument("--llm-model", default="llama3.2:3b")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--all-problems", action="store_true", help="list all 30 bank problems instead of the 8 featured ones")
    ap.add_argument("--no-log", action="store_true", help="do not write the local per-turn log (default: data/demo_logs/turns.jsonl)")
    a = ap.parse_args()
    tutor = DemoTutor(problems=load_problems(featured_only=not a.all_problems), llm_path=a.llm_path, llm_model=a.llm_model,
                      log_path=None if a.no_log else LOG_PATH)
    if a.llm_path:
        ok = tutor.warm_up()          # load the model once, keep it resident
        print(f"LLM path: {'warm (' + a.llm_model + ')' if ok else 'Ollama not reachable, using templates only'}")
    build_demo_ui(tutor).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
