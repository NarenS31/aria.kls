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
import math

import gradio as gr

from agent.demo_bank import load_problems
from agent.demo_input import sanitize_input
from agent.demo_summary import RULE_TEXT, STATE_LABELS, headline, summarize
from agent.demo_log import DEFAULT_PATH as LOG_PATH
from agent.demo_pace import PROFILE_LABELS
from agent.demo_tutor import DemoSession, DemoTutor

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
.aria-pace{font-family:ui-monospace,Menlo,monospace;font-size:.88rem;padding:8px 14px;border-radius:10px;margin-top:8px;
  border:1px solid var(--b,#94a3b8);background:var(--bg,#f8fafc)}
.aria-pace, .aria-pace *{color:#0f172a !important}
.aria-pace.watching{--b:#2563eb;--bg:#eff6ff}.aria-pace.stepping{--b:#d97706;--bg:#fffbeb}.aria-pace.idle{--b:#94a3b8;--bg:#f1f5f9}
.aria-pace .st{font-weight:700;letter-spacing:.05em}
.aria-profiles{font-family:system-ui,sans-serif;font-size:.85rem;line-height:1.35;padding:10px 14px;border-radius:10px;
  border:1px solid #cbd5e1;background:#f1f5f9}
.aria-profiles, .aria-profiles *{color:#0f172a !important}
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


def pace_reason(meta: dict, intervened: bool, cfg=None) -> str:
    """Why ARIA is watching / stepping in, in words. Only derived metadata; never the student's text."""
    if meta.get("fallback"):
        return "could not read that step in time: stayed quiet, tracker not updated"
    r = meta.get("pace_reason", "")
    if r == "wrong_risk_ge_tau":
        return "wrong step and the risk is high enough: hint given"
    if r == "wrong_risk_lt_tau":
        return "wrong step, but the risk is still below the bar: flagged, no hint yet"
    if r == "wrong_before_window":
        return (f"wrong step on an early attempt (1-{cfg.k_min - 1}): ARIA always steps in; the risk bar starts at attempt {cfg.k_min}"
                if cfg else "wrong step on an early attempt: ARIA always steps in")
    if r == "wrong_after_window":
        return f"wrong step after attempt {cfg.k_max}: ARIA always steps in" if cfg else "wrong step on a late attempt: ARIA always steps in"
    if r == "right_step":
        return "correct answer: asking for reflection (existing rule); tracker updated" if intervened else "step checks out: tracker updated, nothing to fix"
    if r == "restate":
        return "restated the problem: tracker not updated"
    if r == "no_verdict":
        return ("help request or question: existing rule answered; tracker not updated" if intervened
                else "no math to check: tracker not updated")
    return RULE_TEXT.get(meta.get("rule_id", ""), "")


def pct(x: float) -> int:
    """Whole percent, rounded down, so 'risk X% needs Y%' never reads equal-but-not-stepping-in (taus are whole percents)."""
    return int(math.floor(x * 100 + 1e-9))


def pace_line_html(session, cfg) -> str:
    """The live ARIA-Pace line: risk of getting stuck, the risk needed to step in, WATCHING / STEPPING IN and the reason."""
    if session is None or cfg is None or getattr(session, "pace", None) is None:
        return ""
    tau, label = cfg.taus[session.profile], PROFILE_LABELS[session.profile]
    last = session.log[-1] if session.log else None
    risk = session.pace.risk()
    shown = "-" if risk is None else f"{pct(risk)}%"
    attempts = len(session.pace.obs)
    bar = f"needs <b>{pct(tau)}%</b> to step in" + (f" from attempt {cfg.k_min}" if attempts < cfg.k_min else "")
    if last is None:
        kind, status, reason = "idle", "READY", "no verified step yet"
    else:
        kind = "stepping" if last["intervened"] else "watching"
        status = "STEPPING IN" if last["intervened"] else "WATCHING"
        reason = pace_reason(last, last["intervened"], cfg)
        if last.get("pace_profile") not in (None, session.profile):
            reason += f" (profile changed to {label}: applies from the next step)"
    return (f'<div class="aria-pace {kind}">risk of getting stuck <b>{shown}</b> &nbsp;|&nbsp; {_esc(label)} {bar} '
            f'&nbsp;|&nbsp; <span class="st">{status}</span>' + (f" &mdash; {_esc(reason)}" if reason else "") + "</div>")


def profiles_note_html(cfg) -> str:
    """What the three profiles are: cost weights, set offline on logged data. Loaded from the frozen files, not hard-coded."""
    rows = "".join(f'<b>{_esc(p["label"])}</b>: interrupting costs {p["w_I"]:g}, a late hint costs {p["w_D"]:g}; '
                   f'steps in at {pct(p["tau"])}% risk<br>' for p in cfg.profiles())
    return ('<div class="aria-profiles">Profiles are <b>cost weights</b> (how much an unnecessary interruption costs relative to a late hint), '
            'not learner types. A higher bar means ARIA waits longer before giving a hint after a wrong step, on '
            f'attempts {cfg.k_min}-{cfg.k_max}; on attempts 1-{cfg.k_min - 1} a wrong step always gets a hint. "Risk of getting stuck" is a '
            'BKT estimate that the student will not master this within 10 steps; the bars were chosen offline on logged ASSISTments data. '
            f'This demo does not measure learning.<br>{rows}</div>')


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pill(label: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{_esc(label)}</span>'


STATE_FILL = {"unseen": "#e2e8f0", "working": "#dbeafe", "secure": "#dcfce7", "high_friction": "#fed7aa"}
STATE_STROKE = {"unseen": "#94a3b8", "working": "#3b82f6", "secure": "#16a34a", "high_friction": "#d97706"}
STATE_WORD = {"unseen": "unseen", "working": "working", "secure": "secure", "high_friction": "high friction"}


def concept_map_html(cm: dict | None) -> str:
    """Small concept map for the Teacher summary: nodes (authored concept labels), prerequisite edges, one state per node. No student text."""
    if not cm or not cm.get("nodes"):
        return ""
    nodes, edges = cm["nodes"], cm["edges"]
    ids = [n["id"] for n in nodes]
    prereq: dict = {i: [] for i in ids}
    for a, b in edges:
        prereq[b].append(a)
    depth: dict = {}

    def d(i):
        if i not in depth:
            depth[i] = 0 if not prereq[i] else 1 + max(d(a) for a in prereq[i])
        return depth[i]
    for i in ids:
        d(i)
    cols: dict = {}
    for n in nodes:
        cols.setdefault(depth[n["id"]], []).append(n)
    W, H, GX, GY = 214, 46, 268, 62
    pos = {n["id"]: (10 + c * GX, 10 + r * GY) for c, col in sorted(cols.items()) for r, n in enumerate(col)}
    width = 20 + (max(cols) + 1) * GX - (GX - W)
    height = 20 + max(len(c) for c in cols.values()) * GY - (GY - H)
    parts = [f'<svg role="img" aria-label="Concept map: prerequisite edges and one state per concept" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
             f'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#64748b"/></marker></defs>']
    for a, b in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        parts.append(f'<line x1="{xa + W}" y1="{ya + H / 2}" x2="{xb - 2}" y2="{yb + H / 2}" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)"/>')
    for n in nodes:
        x, y = pos[n["id"]]
        st = n["state"]
        second = STATE_WORD[st] + (f" · p {n['p_known']:.2f}" if n["p_known"] is not None else "")
        dash = ' stroke-dasharray="4 3"' if st == "unseen" else ""
        parts.append(f'<g><rect x="{x}" y="{y}" width="{W}" height="{H}" rx="8" fill="{STATE_FILL[st]}" stroke="{STATE_STROKE[st]}" stroke-width="2"'
                     f'{dash}/>'
                     f'<text x="{x + 8}" y="{y + 19}" font-size="12" font-weight="600" fill="#0f172a" font-family="system-ui,sans-serif">{_esc(n["label"][:36])}</text>'
                     f'<text x="{x + 8}" y="{y + 36}" font-size="11" fill="#334155" font-family="system-ui,sans-serif">{_esc(second)}</text></g>')
    parts.append("</svg>")
    legend = " ".join(_pill(STATE_WORD[s], "") for s in ("unseen", "working", "secure", "high_friction"))
    return (f'<h3>Concept map</h3><div style="overflow-x:auto;max-width:100%">{"".join(parts)}</div><div>{legend}</div>'
            f'<div class="note"><b>{_esc(cm["note"])}</b> {_esc(cm["rule"])}. Arrows point from a prerequisite to the concept that needs it.</div>')


def summary_html(summary: dict | None) -> str:
    """Teacher view of one attempt. Everything shown is derived metadata or authored text; no chat transcript."""
    if not summary or not summary["timeline"]:
        return ('<div class="aria-sum">No attempt to summarise yet. Pick a problem and try a few steps in the Student tab.'
                + (concept_map_html(summary.get("concept_map")) if summary else "") + "</div>")
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
            f'<h3>Authored steps</h3>{steps}{concept_map_html(summary.get("concept_map"))}<div class="note">{_esc(summary["note"])}</div></div>')


def make_summary_handler():
    def show(session):
        return summary_html(summarize(session) if session is not None else None)
    return show


def make_handlers(tutor: DemoTutor, problems: list):
    ids = {f"{p.prompt}": p.id for p in problems}

    def start(prompt, profile=None, session=None):
        # switching problem keeps the learning graph (in memory, this browser session only); Reset clears it
        s = tutor.new_session(ids[prompt], getattr(session, "graph", None))
        if profile in PROFILE_LABELS and tutor.pace_cfg:
            s.set_profile(profile)
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


def make_pace_handlers(tutor: DemoTutor):
    def show(session):
        return pace_line_html(session, tutor.pace_cfg)

    def set_profile(profile, session):
        if session is not None and profile in PROFILE_LABELS and tutor.pace_cfg:
            session.set_profile(profile)
        return show(session)
    return show, set_profile


def build_demo_ui(tutor: DemoTutor | None = None) -> gr.Blocks:
    problems = (tutor.problems.values() if tutor else load_problems())
    tutor = tutor or DemoTutor()
    problems = list(tutor.problems.values())
    start, send, reset = make_handlers(tutor, problems)
    show_summary = make_summary_handler()
    show_pace, set_profile = make_pace_handlers(tutor)
    cfg = tutor.pace_cfg
    with gr.Blocks(title="ARIA — when to help") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown("## ARIA\nOther tutors speak every turn. ARIA decides **whether** to.")
        session, history = gr.State(None), gr.State([])
        problem = gr.Dropdown([p.prompt for p in problems], value=problems[0].prompt, label="Problem")
        if cfg:
            profile = gr.Radio([(f"{p['label']} (steps in at {pct(p['tau'])}% risk)", p["key"]) for p in cfg.profiles()], value=tutor.pace_profile,
                               label="Pace profile (cost weights, not learner types)")
            gr.HTML(profiles_note_html(cfg))
        else:
            profile = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Student"):
                badge = gr.HTML(badge_html("idle", "Type a step or an answer. ARIA will decide whether to speak."))
                pace_box = gr.HTML("")
                chat = gr.HTML(chat_html([]))
                box = gr.Textbox(placeholder="Type your next step or answer, then press Enter", show_label=False)
                reset_btn = gr.Button("Reset (new attempt)")
            with gr.Tab("Teacher summary") as teacher_tab:
                refresh = gr.Button("Refresh summary")
                summary = gr.HTML(summary_html(None))
        reset_btn.click(reset, [session], [history, session, chat, badge, box]).then(show_summary, [session], [summary]).then(show_pace, [session], [pace_box])
        problem.change(start, [problem, profile, session], [session, history, chat, badge, box]).then(show_summary, [session], [summary]).then(show_pace, [session], [pace_box])
        box.submit(send, [box, history, session], [history, session, chat, badge, box]).then(show_summary, [session], [summary]).then(show_pace, [session], [pace_box])
        demo.load(start, [problem, profile], [session, history, chat, badge, box]).then(show_pace, [session], [pace_box])
        if cfg:
            profile.change(set_profile, [profile, session], [pace_box])
        teacher_tab.select(show_summary, [session], [summary])
        refresh.click(show_summary, [session], [summary])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-path", action="store_true", help="enable the gated LLM path for free-form questions (default off)")
    ap.add_argument("--llm-model", default="llama3.2:3b")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--all-problems", action="store_true", help="list all 30 bank problems instead of the 8 featured ones")
    ap.add_argument("--pace-profile", default="balanced", choices=list(PROFILE_LABELS), help="starting ARIA-Pace profile (changeable in the page)")
    ap.add_argument("--no-log", action="store_true", help="do not write the local per-turn log (default: data/demo_logs/turns.jsonl)")
    a = ap.parse_args()
    tutor = DemoTutor(problems=load_problems(featured_only=not a.all_problems), llm_path=a.llm_path, llm_model=a.llm_model,
                      log_path=None if a.no_log else LOG_PATH, pace_profile=a.pace_profile)
    if a.llm_path:
        ok = tutor.warm_up()          # load the model once, keep it resident
        print(f"LLM path: {'warm (' + a.llm_model + ')' if ok else 'Ollama not reachable, using templates only'}")
    build_demo_ui(tutor).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
