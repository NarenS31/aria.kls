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
from agent.teacher_activity import (BAND_LABELS, GRADES, LABELS, MOMENTS_PATH, PREFER_NOT, ROLE_LABELS, ROLES, TAPS_PATH, TYPES, YEARS, Activity,
                                    TapLog, load_moments)

CSS = """
/* Every card sets its own background AND its own text colour (with !important): Gradio's dark theme otherwise paints
   near-white text onto these light cards. */
.aria-badge{display:flex;align-items:center;gap:14px;padding:16px 20px;border-radius:14px;font-family:system-ui,sans-serif;
  border:2px solid var(--b,#94a3b8);background:var(--bg,#f1f5f9)}
.aria-badge, .aria-badge *{color:#0f172a !important}
.aria-badge .dot{width:16px;height:16px;border-radius:50%;background:var(--b,#94a3b8);flex:none}
.aria-badge.watching{--b:#2563eb;--bg:#eff6ff}.aria-badge.watching .dot{animation:pulse 1.6s ease-in-out infinite}
.aria-badge.stepping{--b:#d97706;--bg:#fffbeb}
.aria-badge.idle{--b:#94a3b8;--bg:#f1f5f9}
.aria-badge .big{font-weight:700;letter-spacing:.06em;font-size:1.15rem}
.aria-badge .small{font-size:.9rem}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.25)}}
/* column-reverse keeps the newest message in view without any script */
.aria-chat{display:flex;flex-direction:column-reverse;min-height:300px;max-height:380px;overflow:auto;padding:8px;font-family:system-ui,sans-serif}
.aria-chat-in{display:flex;flex-direction:column;gap:8px}
.aria-chat .msg{padding:10px 14px;border-radius:12px;max-width:80%;line-height:1.35}
.aria-chat .msg, .aria-chat .msg *{color:#0f172a !important}
.aria-chat .user{align-self:flex-end;background:#e2e8f0}
.aria-chat .aria{align-self:flex-start;background:#fef3c7;border:1px solid #f59e0b}
.aria-chat .quiet{align-self:flex-start;background:#f1f5f9;border:1px solid #cbd5e1;font-style:italic;font-size:.9rem;padding:6px 12px;border-radius:999px;max-width:80%}
.aria-chat .quiet, .aria-chat .quiet *{color:#334155 !important}
.aria-sum{font-family:system-ui,sans-serif;line-height:1.4;background:#fcfcfb;border:1px solid #cbd5e1;border-radius:12px;padding:16px}
.aria-sum, .aria-sum *{color:#0f172a !important}
.aria-sum h3{margin:14px 0 6px;font-size:1rem}
.aria-sum .head{font-size:1.05rem;font-weight:600;margin-bottom:4px}
.aria-sum .where{padding:10px 14px;border-radius:10px;background:#fffbeb;border:1px solid #f59e0b;margin:8px 0}
.aria-sum .tiles{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.aria-sum .tile{border:1px solid #cbd5e1;border-radius:10px;padding:8px 14px;min-width:90px;background:#f8fafc}
.aria-sum .tile b{display:block;font-size:1.3rem}
.aria-sum table{border-collapse:collapse;width:100%;font-size:.92rem;background:#fff}
.aria-sum th,.aria-sum td{border-bottom:1px solid #e2e8f0;padding:5px 8px;text-align:left;vertical-align:top;background:#fff}
.aria-sum th{background:#f1f5f9;font-weight:600}
.aria-sum .pill{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid #94a3b8;background:#f1f5f9;font-size:.85rem}
.aria-sum .pill.bad{border-color:#d97706;background:#fffbeb}.aria-sum .pill.good{border-color:#16a34a;background:#f0fdf4}
.aria-sum .note{margin-top:14px;font-size:.82rem}
.aria-sum .note{color:#475569 !important}
.aria-act{font-family:system-ui,sans-serif;line-height:1.45;background:#fcfcfb;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin:6px 0}
.aria-act, .aria-act *{color:#0f172a !important}
.aria-act .step{font-size:.85rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.aria-act .prob{font-size:1.1rem;font-weight:600;margin:6px 0 10px}
.aria-act .work{margin:4px 0;padding:6px 10px;border-radius:8px;background:#f1f5f9}
.aria-act .work.now{background:#fffbeb;border:1px solid #f59e0b;font-weight:600}
.aria-act .q{margin-top:12px;font-weight:600}
.aria-act .you{margin:2px 0}.aria-act .aria-says{margin-top:8px;padding:8px 12px;border-radius:10px;background:#fef3c7;border:1px solid #f59e0b}
.aria-act .quietbox{margin-top:8px;padding:8px 12px;border-radius:10px;background:#eff6ff;border:1px solid #2563eb}
"""


def chat_html(history: list) -> str:
    rows = []
    for m in history:
        cls = {"user": "user", "aria": "aria", "quiet": "quiet"}[m["role"]]
        rows.append(f'<div class="msg {cls}">{html.escape(m["content"])}</div>')
    return '<div class="aria-chat"><div class="aria-chat-in">' + "".join(rows) + "</div></div>"


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
            if h["type"] == "reflection":
                return "after the correct answer"
            return "a free-form question" if h["source"] == "llm" else "a general prompt (every authored step was already done)"
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


def activity_card(act: Activity) -> str:
    m = act.moment
    if m is None:
        return ""
    work = "".join(f'<div class="work">{_esc(h)}</div>' for h in m["history"]) + f'<div class="work now">{_esc(m["message"])}</div>'
    return (f'<div class="aria-act"><div class="step">Moment {act.i + 1} of {len(act.moments)}</div><div class="prob">{_esc(m["problem_prompt"])}</div>'
            f'<div>What the student has written, newest last:</div>{work}<div class="q">{"Would you step in here?" if act.phase == "decision" else "What kind of help?"}</div></div>')


def activity_reveal(act: Activity) -> str:
    r = act.reveal()
    if r is None:
        return ""
    mine = LABELS[r["teacher_decision"]] + (f" ({LABELS[r['teacher_type']].lower()})" if r["teacher_type"] else "")
    if r["aria_decision"] == "step_in":
        theirs = (f'<div class="aria-says"><b>ARIA would step in</b> with {LABELS[r["aria_type"]].lower()}. Why: {_esc(r["reason"])}.<br>'
                  f'It would say: "{_esc(r["reply"])}"</div>')
    else:
        theirs = f'<div class="quietbox"><b>ARIA would stay quiet.</b> Why: {_esc(r["reason"])}.</div>'
    return f'<div class="aria-act"><div class="you">You chose: <b>{_esc(mine)}</b></div>{theirs}<div class="q">Do you agree with ARIA?</div></div>'


def activity_render(act: Activity | None):
    """-> updates for [state, consent, decide, type, reveal, done, card, reveal_html, done_md]."""
    vis = lambda *phases: gr.update(visible=bool(act and act.phase in phases))  # noqa: E731
    done_md = ""
    if act and act.phase == "skipped":
        done_md = "No problem. Nothing was recorded."
    elif act and act.phase == "done":
        done_md = f"Thank you. {act.rows_written} answers were saved on this laptop only. Pass the laptop to the next visitor."
    return (act, vis("consent"), vis("decision"), vis("type"), vis("reveal"), vis("done", "skipped"),
            gr.update(value=activity_card(act) if act else "", visible=bool(act and act.phase in ("decision", "type"))),
            gr.update(value=activity_reveal(act) if act else ""), gr.update(value=done_md))


def make_activity_handlers(moments: list, log: TapLog):
    def new():
        return activity_render(Activity(moments, log))

    def proceed(act, role, years, grade, accept):
        act = act or Activity(moments, log)
        act.set_profile(role, years, grade)
        act.consent(accept)
        return activity_render(act)

    def act_on(method):
        def fn(act, value):
            if act is not None:
                getattr(act, method)(value)
            return activity_render(act)
        return fn

    return new, proceed, act_on("decide"), act_on("choose_type"), act_on("agree")


def build_activity_tab(moments_path=MOMENTS_PATH, taps_path=TAPS_PATH):
    """The 'Would you step in?' tab. Needs data/teacher_moments.json (scripts/make_teacher_moments.py); without it the tab says so."""
    try:
        moments = load_moments(moments_path)
    except (OSError, ValueError) as e:
        gr.Markdown(f"The teacher activity is not set up on this laptop ({type(e).__name__}). Run `scripts/make_teacher_moments.py` first.")
        return None
    new, proceed, decide, choose_type, agree = make_activity_handlers(moments, TapLog(taps_path))
    act = gr.State(None)
    with gr.Group(visible=True) as consent:
        gr.Markdown("### Would you step in?\nYou will see 8 short moments from a student's work. For each one, decide whether a tutor should **step in** or **stay quiet**, "
                    "then see what ARIA decided and say whether you agree.\n\n"
                    "**What is saved:** your choices only (step in or stay quiet, the kind of help, agree or disagree), plus the three optional answers below. "
                    "No names, no typed text, no network; the file stays on this laptop. **Skip** saves nothing.")
        role = gr.Dropdown([(ROLE_LABELS[r], r) for r in ROLES], value=PREFER_NOT, label="Your role (optional)")
        years = gr.Dropdown([(("Prefer not to say" if y == PREFER_NOT else y), y) for y in YEARS], value=PREFER_NOT, label="Years of experience (optional)")
        grade = gr.Dropdown([(BAND_LABELS[g], g) for g in GRADES], value=PREFER_NOT, label="Grade band you work with (optional)")
        with gr.Row():
            go = gr.Button("Continue", variant="primary")
            skip = gr.Button("Skip")
    card = gr.HTML("", visible=False)
    with gr.Row(visible=False) as decide_row:
        step_in, stay = gr.Button("Step in", variant="primary"), gr.Button("Stay quiet")
    with gr.Row(visible=False) as type_row:
        type_btns = [(gr.Button(LABELS[t]), t) for t in TYPES]
    with gr.Group(visible=False) as reveal:
        reveal_html = gr.HTML("")
        with gr.Row():
            yes, no = gr.Button("Agree", variant="primary"), gr.Button("Disagree")
    with gr.Group(visible=False) as done:
        done_md = gr.Markdown("")
        again = gr.Button("Start for the next visitor")
    outs = [act, consent, decide_row, type_row, reveal, done, card, reveal_html, done_md]
    go.click(lambda a, r, y, g: proceed(a, r, y, g, True), [act, role, years, grade], outs)
    skip.click(lambda a, r, y, g: proceed(a, r, y, g, False), [act, role, years, grade], outs)
    step_in.click(lambda a: decide(a, "step_in"), [act], outs)
    stay.click(lambda a: decide(a, "stay_quiet"), [act], outs)
    for btn, t in type_btns:
        btn.click(lambda a, _t=t: choose_type(a, _t), [act], outs)
    yes.click(lambda a: agree(a, "agree"), [act], outs)
    no.click(lambda a: agree(a, "disagree"), [act], outs)
    again.click(new, None, outs)
    return new, outs


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


def build_demo_ui(tutor: DemoTutor | None = None, moments_path=MOMENTS_PATH, taps_path=TAPS_PATH) -> gr.Blocks:
    problems = (tutor.problems.values() if tutor else load_problems())
    tutor = tutor or DemoTutor()
    problems = list(tutor.problems.values())
    start, send, reset = make_handlers(tutor, problems)
    show_summary = make_summary_handler()
    with gr.Blocks(title="ARIA — when to help") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown("## ARIA\nOther tutors speak every turn. ARIA decides **whether** to.")
        session, history = gr.State(None), gr.State([])
        with gr.Tabs():
            with gr.Tab("Student"):
                problem = gr.Dropdown([p.prompt for p in problems], value=problems[0].prompt, label="Problem")
                badge = gr.HTML(badge_html("idle", "Type a step or an answer. ARIA will decide whether to speak."))
                chat = gr.HTML(chat_html([]))
                box = gr.Textbox(placeholder="Type your next step or answer, then press Enter", show_label=False)
                reset_btn = gr.Button("Reset (new attempt)")
            with gr.Tab("Teacher summary") as teacher_tab:
                refresh = gr.Button("Refresh summary")
                summary = gr.HTML(summary_html(None))
            with gr.Tab("Would you step in?"):
                activity = build_activity_tab(moments_path, taps_path)
        reset_btn.click(reset, [session], [history, session, chat, badge, box]).then(show_summary, [session], [summary])
        problem.change(start, [problem], [session, history, chat, badge, box]).then(show_summary, [session], [summary])
        box.submit(send, [box, history, session], [history, session, chat, badge, box]).then(show_summary, [session], [summary])
        demo.load(start, [problem], [session, history, chat, badge, box])
        if activity:
            demo.load(activity[0], None, activity[1])
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
    ap.add_argument("--moments", default=str(MOMENTS_PATH), help="teacher-activity moments (scripts/make_teacher_moments.py)")
    ap.add_argument("--taps-csv", default=str(TAPS_PATH), help="where the activity answers go (data/demo_logs/teacher_taps.csv)")
    a = ap.parse_args()
    tutor = DemoTutor(problems=load_problems(featured_only=not a.all_problems), llm_path=a.llm_path, llm_model=a.llm_model,
                      log_path=None if a.no_log else LOG_PATH)
    if a.llm_path:
        ok = tutor.warm_up()          # load the model once, keep it resident
        print(f"LLM path: {'warm (' + a.llm_model + ')' if ok else 'Ollama not reachable, using templates only'}")
    build_demo_ui(tutor, a.moments, None if a.no_log else a.taps_csv).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
