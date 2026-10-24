"""Student-first Gradio workspace over the existing Python tutor."""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import gradio as gr
import sympy as sp

from agent.demo_workspace import Workspace
from agent.math_check import _LOCALS, _TRANSFORMS, extract_candidates, normalize, parse_expr, to_expr

ROOT = Path(__file__).resolve().parent.parent
MATH = [{"left": "$$", "right": "$$", "display": True}]

# Inline icons: every verdict and every reply carries a shape as well as a colour.
ICON_OK = '<svg class="ui-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/><path d="m5.8 10.4 2.7 2.7 5.7-6" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
ICON_ISSUE = '<svg class="ui-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8.2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M10 5.6v5.2" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="10" cy="14" r="1.25" fill="currentColor"/></svg>'
ICON_NEUTRAL = '<svg class="ui-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8.2" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="10" cy="10" r="2.6" fill="currentColor"/></svg>'
ICON_HOLD = '<svg class="ui-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/><path d="M7.6 6.2v7.6M12.4 6.2v7.6" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/></svg>'
ICON_HINT = '<svg class="ui-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/><path d="M10 5a3.6 3.6 0 0 0-2 6.6V13h4v-1.4A3.6 3.6 0 0 0 10 5Zm-1.4 9.4h2.8" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>'
VERDICT_ICONS = {"ok": ICON_OK, "issue": ICON_ISSUE, "neutral": ICON_NEUTRAL}


def font_css():
    """Figtree from ui/assets/fonts, embedded so the page works with the network off."""
    faces = []
    for weight in (400, 600, 700, 800):
        data = base64.b64encode((ROOT / f"ui/assets/fonts/figtree-latin-{weight}-normal.woff2").read_bytes()).decode()
        faces.append(f'@font-face{{font-family:"Figtree";font-style:normal;font-weight:{weight};font-display:swap;'
                     f'src:url(data:font/woff2;base64,{data}) format("woff2");}}')
    return "".join(faces)


def problem_label(prompt):
    """Use readable Unicode in plain-text choices without changing problem data."""
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    label = re.sub(r"\^\(?(-?\d+)\)?", lambda m: m[1].translate(superscripts), prompt)
    label = re.sub(r"\bpi\b", "π", label)
    return label.replace("*", "×").replace(" - ", " − ")


def latex(text):
    """Keep written grouping for display after passing the existing guarded parser."""
    s = normalize(text)
    if to_expr(s) is None:
        return None
    if "/" in s:
        return sp.latex(to_expr(s))
    try:
        return sp.latex(parse_expr(s, local_dict=dict(_LOCALS), transformations=_TRANSFORMS, evaluate=False))
    except Exception:
        return sp.latex(to_expr(s))


def math_text(text):
    if "=" in text:
        sides = text.split("=")
        if len(sides) == 2:
            parts = [latex(s) for s in sides]
            if all(parts):
                return "$$" + " = ".join(parts) + "$$"
    expr = latex(text)
    if expr:
        return "$$" + expr + "$$"
    candidates = extract_candidates(text)
    if candidates:
        c = candidates[0]
        if c.kind == "expr":
            return "$$" + sp.latex(c.expr) + "$$"
        if c.kind == "eq":
            return "$$" + sp.latex(c.lhs) + " = " + sp.latex(c.rhs) + "$$"
    return '<p class="literal">' + html.escape(text) + '</p>'


def problem_md(problem):
    spec = problem.spec
    if problem.truth.kind == "solve":
        label, expression = "Solve for x", spec["equation"]
    else:
        label = {"diff": "Differentiate", "integrate": "Find the indefinite integral", "simplify": "Simplify",
                 "expand": "Expand", "value": "Evaluate"}.get(spec.get("check", {}).get("op"), "Work through this problem")
        expression = spec.get("check", {}).get("of", spec.get("given", [problem.prompt])[0])
    topic = html.escape(problem.topic.title())
    return f'<p class="problem-topic">{topic}</p>\n<p class="problem-type">{html.escape(label)}</p>\n\n{math_text(expression)}'


def work_md(events):
    if not events:
        return ""
    rows, n = [], 0
    labels = {"correct": ("ok", "Problem solved"), "partial": ("ok", "Valid progress"), "step": ("ok", "Valid step"),
              "wrong": ("issue", "Check this step"), "restate": ("neutral", "Starting equation"), "none": ("issue", "No checkable math found")}
    for e in events:
        r = e.result
        if e.kind == "step":
            n += 1
            tone, label = labels.get(r.status, ("issue", "Could not check this step"))
            rows.append(f'<div class="step {tone}">\n<p class="step-label"><span class="step-num">{n}</span><span class="step-name">Step {n}</span>'
                        f'<span class="verdict {tone}">{VERDICT_ICONS[tone]}{label}</span></p>\n\n{math_text(e.text)}\n\n</div>')
        if r.intervened and r.meta.get("intervention_type") != "reflection":
            source = "Requested hint" if r.meta.get("hint_source") == "requested" else "ARIA hint"
            rows.append(f'<details class="hint-history"><summary>{ICON_HINT}{source}</summary><p>{html.escape(r.reply)}</p></details>')
    return '\n\n'.join(rows)


def support_md(w):
    if w.solved:
        return ""
    for event in reversed(w.events):
        if event.result.intervened and event.result.meta.get("intervention_type") != "reflection":
            reply = html.escape(event.result.reply).replace("\n", "<br>")
            return (f'<div class="aria-reply given"><p class="reply-tag">{ICON_HINT}Here\u2019s a hint</p>'
                    f'<h3>A small next step</h3><p>{reply}</p></div>')
        if event.kind == "step":
            if event.result.status == "wrong":
                return (f'<div class="aria-reply held"><p class="reply-tag">{ICON_HOLD}Holding the hint</p>'
                        '<h3>Take another look</h3><p>Revise this step or request a hint. Your work stays in place.</p></div>')
            if event.result.status == "none":
                return ('<div class="aria-reply note"><h3>Try a math expression</h3>'
                        '<p>Use an equation or expression, such as <code>3x - 12 = 2x + 5</code>.</p></div>')
            break
    return ""


def teacher_html(w):
    s, esc = w.summary(), html.escape
    math_events = [e for e in w.events if e.kind == "step"]
    wrong = [i + 1 for i, e in enumerate(math_events) if e.result.status == "wrong"]
    verified = sum(e.result.status in ("step", "partial", "correct") for e in math_events)
    follow = next((x["step"] for x in s["steps"] if not x["done"]), "Ask the student to explain why their last transformation preserves the result.")
    outcome = "Complete" if w.solved else "In progress" if math_events else "Ready to start"
    revision_note = (f'<p class="revision-note">This work has {len(w.branches)} retained earlier branch(es). Counts below describe the current branch; earlier errors and hints remain in Previous work.</p>') if w.branches else ""
    breakdown = ("Incorrect work appeared at " + ("step " if len(wrong) == 1 else "steps ") + ", ".join(map(str, wrong)) + ".") if wrong else "No incorrect mathematical steps have been recorded in this attempt."
    hint_rows = []
    for i, e in enumerate(w.events):
        if not e.result.intervened or e.result.meta.get("intervention_type") not in ("hint", "task_chunking"):
            continue
        nxt = next((v for v in w.events[i + 1:] if v.kind == "step"), None)
        after = "No subsequent step yet." if nxt is None else ("The next step was verified." if nxt.result.status in ("step", "partial", "correct") else "The next step was not verified as correct.")
        source = "Requested" if e.result.meta.get("hint_source") == "requested" else "Automatic"
        move = e.result.meta.get("move_index", -1)
        focus = w.session.problem.moves[move].get("do", "General prompt") if move >= 0 else "General prompt"
        hint_rows.append(f'<li><strong>{source}:</strong> {esc(focus)}<br><span>{after}</span></li>')
    timeline = ''.join(f'<tr><td>{i + 1}</td><td>{esc(e.result.status)}</td><td>{esc(e.result.meta.get("hint_source", "none"))}</td><td>{esc(e.profile.title())}</td></tr>' for i, e in enumerate(w.events))
    return f'''<section class="teacher-report"><p class="eyebrow">CURRENT ATTEMPT</p><h2>{outcome}</h2>
      <p>{esc(w.session.problem.prompt)}</p>{revision_note}<div class="report-counts"><div><strong>{len(math_events)}</strong><span>math steps</span></div><div><strong>{verified}</strong><span>verified steps</span></div><div><strong>{s['counts']['requested_hints']}</strong><span>requested hints</span></div><div><strong>{s['counts']['automatic_hints']}</strong><span>automatic hints</span></div></div>
      <h3>Where to follow up</h3><p>{breakdown}</p><p><strong>Suggested conversation:</strong> {esc(follow)}</p>
      <h3>Support and the next step</h3><ul>{''.join(hint_rows) or '<li>No mathematical hints have been given yet.</li>'}</ul>
      <p class="muted">{verified} verified {'step was' if verified == 1 else 'steps were'} recorded. This does not establish independent mastery or show that hints caused improvement.</p>
      <details><summary>Attempt timeline and revisions</summary><div class="table-scroll"><table><thead><tr><th>Event</th><th>Math check</th><th>Hint source</th><th>Profile at event</th></tr></thead><tbody>{timeline}</tbody></table></div><p>{len(w.branches)} revised branches; {len(w.archive)} previous attempts remain in this session.</p></details>
      </section>'''


def risk_meter(w, snapshot):
    """Stuck risk as a bar with one threshold tick per profile, in the profile colours."""
    cfg = w.tutor.pace_cfg
    ticks = sorted(((p["key"], p["label"], p["tau"]) for p in cfg.profiles()), key=lambda t: t[2]) if cfg else []
    risk = None if snapshot.risk is None else min(max(snapshot.risk, 0), 1)
    said = "no estimate yet" if risk is None else f"{risk:.0%}"
    desc = "; ".join(f"{label} steps in at {tau:.0%}" for _, label, tau in ticks)
    fill = '' if risk is None else f'<span class="risk-fill" style="width:{risk * 100:.1f}%"></span>'
    marks = ''.join(f'<span class="risk-tick {key}" style="left:{tau * 100:.1f}%"></span>' for key, _, tau in ticks)
    labels = ''.join(f'<span class="risk-label {key}" style="left:{tau * 100:.1f}%">{tau:.0%}</span>' for key, _, tau in ticks)
    legend = ''.join(f'<span class="risk-key {key}"><i></i>{html.escape(label)}</span>' for key, label, _ in ticks)
    return (f'<div class="risk-meter" role="img" aria-label="Estimated non-mastery risk: {said}. Thresholds: {html.escape(desc)}.">'
            f'<div class="risk-track">{fill}{marks}</div><div class="risk-labels">{labels}</div><div class="risk-legend">{legend}</div></div>\n\n')


def decision_md(w):
    snapshot = w.session.pace.snapshot(w.session.profile) if w.session.pace else None
    if not snapshot:
        return "Timing estimates are disabled in this session."
    risk = "No estimate yet" if snapshot.risk is None else f"{snapshot.risk:.1%}"
    last = w.events[-1].result.meta if w.events else {}
    return (risk_meter(w, snapshot) +
            f"**Current setting:** {w.session.profile.title()} · **Threshold:** {snapshot.tau:.0%}\n\n"
            f"**Verified observations:** {snapshot.n_obs} · **Estimated non-mastery risk:** {risk}\n\n"
            f"**Last decision:** `{last.get('rule_id', 'No decision yet')}`\n\n"
            "The demo uses pooled BKT parameters, not the per-skill evaluated model. The threshold applies on verified attempts 3–9. "
            "Outside that window, a wrong step receives support. Requested hints bypass automatic waiting and do not add a math observation. "
            "Changing the setting affects future decisions only.")


def archive_md(w):
    rows = []
    for i, a in enumerate(w.archive):
        rows.append(f"### Attempt {i + 1} · {html.escape(w.tutor.problems[a['problem']].prompt)}\n\n" + work_md(a['events']))
        if a.get("reflection"):
            rows.append('<p><strong>Reflection:</strong> ' + html.escape(a["reflection"]) + '</p>')
        for j, branch in enumerate(a.get("branches", [])):
            rows.append(f"### Attempt {i + 1}, replaced branch {j + 1}\n\n" + work_md(branch["events"]))
    for i, b in enumerate(w.branches):
        rows.append(f"### Earlier branch {i + 1} (replaced)\n\n" + work_md(b['events']))
    return '\n\n'.join(rows) or "No previous attempts or revised branches in this session."


def build_workspace(tutor):
    problems = list(tutor.problems.values())
    topics = ["Featured", "All topics"] + sorted({p.topic.title() for p in problems})
    logo = (ROOT / "ui/assets/aria-wordmark.svg").read_text()
    theme = gr.themes.Base(font=["Figtree", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"], font_mono=["monospace"])
    colors = {
        "body_background_fill": "var(--color-page)", "body_text_color": "var(--color-ink)",
        "body_text_color_subdued": "var(--color-muted)", "background_fill_primary": "var(--color-page)",
        "background_fill_secondary": "var(--color-paper-2)", "block_background_fill": "var(--color-paper)",
        "block_label_text_color": "var(--color-ink-2)", "block_title_text_color": "var(--color-ink)",
        "block_label_background_fill": "var(--color-paper)", "input_background_fill": "var(--color-paper)",
        "input_placeholder_color": "var(--color-muted)", "input_border_color": "var(--color-rule)",
        "border_color_primary": "var(--color-rule)", "panel_background_fill": "var(--color-paper)",
        "accordion_text_color": "var(--color-ink)", "checkbox_label_text_color": "var(--color-ink)",
        "checkbox_label_background_fill": "var(--color-paper)", "checkbox_label_background_fill_selected": "var(--color-paper-2)",
        "button_secondary_text_color": "var(--color-ink)", "button_secondary_background_fill": "var(--color-paper)",
        "table_text_color": "var(--color-ink)", "link_text_color": "var(--color-ink)",
    }
    theme.set(color_accent="var(--aria-accent)", **{key + suffix: value for key, value in colors.items() for suffix in ("", "_dark")})
    focus_js = """(...args) => {setTimeout(() => {const x=document.querySelector('#step-input textarea');if(x&&!x.disabled)x.focus();},100);return args;}"""
    init_js = """() => {document.documentElement.classList.remove('dark'); const bind=()=>{const x=document.querySelector('#step-input textarea');if(x&&!x.dataset.ariaBound){x.dataset.ariaBound='1';x.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.isComposing||e.keyCode===229))e.stopImmediatePropagation();},true);}};new MutationObserver(bind).observe(document.body,{childList:true,subtree:true});bind();}"""
    with gr.Blocks(title="ARIA | Math workspace", theme=theme, analytics_enabled=False, fill_width=True,
                   head="<style>" + font_css() + (ROOT / "tokens.css").read_text() + (ROOT / "ui/workspace.css").read_text() + "</style>", js=init_js) as app:
        state = gr.State(None, time_to_live=14400)
        revision = gr.Number(0, visible=False, precision=0)
        with gr.Row(elem_id="app-bar"):
            gr.HTML(f'<header class="app-header"><div class="brand">{logo}<span>Math workspace</span></div></header>', elem_id="brand-block")
            profile = gr.Radio([("Quiet", "quiet"), ("Balanced", "balanced"), ("Supportive", "supportive")], value=tutor.pace_profile, label="Hint timing", show_label=False, container=False, elem_id="timing-control")
        with gr.Row(elem_id="problem-nav"):
            topic = gr.Dropdown(topics, value="Featured", label="Topic", show_label=False, container=False, scale=1, min_width=130)
            picker = gr.Dropdown([(problem_label(p.prompt), p.id) for p in problems if p.featured], value=problems[0].id, label="Problem", show_label=False, container=False, scale=4, min_width=180)
        with gr.Tabs(elem_id="workspace-tabs"):
            with gr.Tab("Your work", id="student"):
                with gr.Row(elem_id="work-layout"):
                    with gr.Column(scale=7, min_width=260, elem_id="work-column"):
                        problem = gr.Markdown(problem_md(problems[0]), latex_delimiters=MATH, elem_id="active-problem")
                        work = gr.Markdown(work_md([]), latex_delimiters=MATH, elem_id="work-steps")
                        notice = gr.Markdown("", elem_id="workspace-notice")
                        with gr.Group(elem_id="composer") as composer:
                            box = gr.Textbox(label="Next step", placeholder="Write your next equation or expression…", lines=1, max_lines=5, max_length=1000, container=False, elem_id="step-input")
                            with gr.Row(elem_id="math-tools"):
                                square = gr.Button("x²", size="sm", min_width=42, elem_id="symbol-square")
                                root = gr.Button("√", size="sm", min_width=42, elem_id="symbol-root")
                                fraction = gr.Button("a/b", size="sm", min_width=42, elem_id="symbol-fraction")
                                pi = gr.Button("π", size="sm", min_width=42, elem_id="symbol-pi")
                            preview = gr.Markdown("", latex_delimiters=MATH, elem_id="input-preview")
                            with gr.Row(elem_id="submit-actions"):
                                check = gr.Button("Check step", variant="primary", elem_id="check-step")
                                mobile_hint = gr.Button("Get a hint", elem_id="get-hint-mobile")
                        with gr.Group(visible=False, elem_id="completion") as completion:
                            gr.Markdown("### Problem complete\nYour final answer has been verified.")
                            next_problem = gr.Button("Next problem", variant="primary")
                            with gr.Accordion("Add a reflection (optional)", open=False):
                                reflection = gr.Textbox(label="Reflection", placeholder="Which step made the difference?", lines=2)
                                reflect = gr.Button("Save reflection", size="sm")
                                reflection_status = gr.Markdown("")
                    with gr.Column(scale=3, min_width=240, elem_id="support-column"):
                        activity_status = gr.HTML('', visible=False)
                        hint = gr.Button("Get a hint", elem_id="get-hint")
                        support = gr.Markdown("", elem_id="support-copy")
                        with gr.Accordion("Session options", open=False, elem_id="session-options"):
                            editor = gr.Dropdown([("New step", "new")], value="new", label="Working on", elem_id="edit-step")
                            retry = gr.Button("Retry problem", elem_id="retry-problem")
                            with gr.Accordion("Previous work", open=False):
                                archive = gr.Markdown("No earlier work yet.", latex_delimiters=MATH)
                            with gr.Accordion("Clear session", open=False):
                                confirm = gr.Checkbox(label="Clear all work and learning estimates in this session")
                                clear = gr.Button("Clear session", variant="stop")
                        with gr.Accordion("Decision details", open=False, elem_id="decision-details"):
                            decisions = gr.Markdown("")
            with gr.Tab("Teacher view", id="teacher"):
                report = gr.HTML("")
                with gr.Accordion("Model estimates", open=False):
                    graph = gr.JSON(label="Exploratory concept estimates")
        gr.HTML('<footer class="app-footer"><span>ARIA</span><span>Work is held in this server session, not saved as a transcript. Sessions expire after four hours.</span></footer>')

        outputs = [state, revision, problem, work, support, decisions, report, graph, editor, box, notice, completion, check, hint, archive, preview, reflection, reflection_status, composer, activity_status, mobile_hint]

        def render(w, draft="", editing="new"):
            choices = [("New step", "new")]
            n = 0
            for i, e in enumerate(w.events):
                if e.kind == "step":
                    n += 1
                    choices.append((f"Revise step {n}", str(i)))
            if editing not in [v for _, v in choices]:
                editing = "new"
            steps = sum(e.kind == "step" for e in w.events)
            hints = sum(e.result.intervened and e.result.meta.get("intervention_type") in ("hint", "task_chunking") for e in w.events)
            status = "Problem complete" if w.solved else (f"{steps} {'step' if steps == 1 else 'steps'} checked · {hints} {'hint' if hints == 1 else 'hints'} given" if steps else "Ready for your first step")
            return (w, w.revision, problem_md(w.session.problem), work_md(w.events), support_md(w), decision_md(w), teacher_html(w),
                    w.summary().get("concept_map"), gr.update(choices=choices, value=editing),
                    gr.update(value=draft, interactive=not w.solved or editing != "new", label="Revise step" if editing != "new" else "Next step"),
                    html.escape(w.notice), gr.update(visible=w.solved), gr.update(value="Check step", interactive=not w.solved or editing != "new"),
                    gr.update(interactive=not w.solved), archive_md(w), math_text(draft) if draft.strip() else "", w.reflection,
                    "Reflection saved for this session." if w.reflection else "", gr.update(visible=not w.solved or editing != "new"),
                    '<p class="activity-status' + (' is-complete' if w.solved else '') + '">' + status + '</p>', gr.update(interactive=not w.solved))

        def start():
            return render(Workspace(tutor))

        def send(text, w, rev, edit):
            w = w or Workspace(tutor)
            ok = w.submit(text, int(rev), None if edit == "new" else int(edit))
            return render(w, "" if ok else text, "new" if ok else edit)

        def send_ui(text, w, rev, edit, pid=None):
            busy = [gr.skip() for _ in outputs]
            busy[9] = gr.update(interactive=False)
            busy[12] = gr.update(value="Checking…", interactive=False)
            busy[13] = gr.update(interactive=False)
            busy[20] = gr.update(interactive=False)
            yield tuple(busy)
            if w is None:
                w = Workspace(tutor, pid)
                w.notice = "Your session restarted. Review the problem and check your step again."
                yield render(w, text)
                return
            try:
                yield send(text, w, rev, edit)
            except Exception:
                w.notice = "The check could not finish. Your input is still here; please try again."
                yield render(w, text, edit)

        def ask(w, rev, draft, edit, pid=None):
            if w is None:
                w = Workspace(tutor, pid)
                w.notice = "Your session restarted. Your draft is still here; request a hint again when ready."
                return render(w, draft)
            w.hint(int(rev))
            return render(w, draft, edit)

        def switch(pid, w):
            w = w or Workspace(tutor)
            if pid and pid != w.session.problem.id:
                w.change_problem(pid)
            return render(w)

        def restart(w, pid=None):
            if w is None:
                return render(Workspace(tutor, pid))
            w.change_problem(w.session.problem.id, retry=True)
            return render(w)

        def erase(confirmed, w, draft, edit, pid=None):
            w = w or Workspace(tutor, pid)
            if confirmed:
                w.clear()
            else:
                w.notice = "Confirm the session-wide reset before clearing."
            return render(w, draft if not confirmed else "", edit if not confirmed else "new") + (False,)

        def set_profile(key, w, draft, edit, pid=None):
            restarted = w is None
            w = w or Workspace(tutor, pid)
            w.session.set_profile(key)
            w.notice = f"{key.title()} timing applies to subsequent decisions. Earlier decisions are unchanged."
            if restarted:
                w.notice = "Your session restarted. " + w.notice
                return render(w, draft)
            updates = [gr.skip() for _ in outputs]
            updates[0] = w
            updates[5] = decision_md(w)
            updates[10] = html.escape(w.notice)
            return tuple(updates)

        def pick_edit(value, w, pid=None):
            if w is None:
                w = Workspace(tutor, pid)
                w.notice = "Your session restarted. Earlier work is no longer available in this session."
                return render(w)
            if value != "new" and (not value.isdigit() or int(value) >= len(w.events)):
                return render(w)
            return render(w, "" if value == "new" else w.events[int(value)].text, value)

        common = dict(api_name=False, show_progress="minimal", concurrency_id="workspace", concurrency_limit=1)
        app.load(start, outputs=outputs, **common)
        for event in (check.click, box.submit):
            event(send_ui, [box, state, revision, editor, picker], outputs, trigger_mode="once", **common).then(fn=None, js=focus_js, inputs=[], outputs=[], queue=False)
        hint.click(ask, [state, revision, box, editor, picker], outputs, **common)
        mobile_hint.click(ask, [state, revision, box, editor, picker], outputs, **common)
        picker.input(switch, [picker, state], outputs, **common)
        topic.input(lambda t: gr.update(choices=[(problem_label(p.prompt), p.id) for p in problems if t == "All topics" or (t == "Featured" and p.featured) or p.topic.title() == t]), topic, picker, api_name=False)
        editor.input(pick_edit, [editor, state, picker], outputs, **common)
        retry.click(restart, [state, picker], outputs, **common)
        clear.click(erase, [confirm, state, box, editor, picker], outputs + [confirm], **common)
        profile.input(set_profile, [profile, state, box, editor, picker], outputs, **common)
        def advance(w, pid=None):
            w = w or Workspace(tutor, pid)
            ids = [p.id for p in problems if p.featured] or [p.id for p in problems]
            current = ids.index(w.session.problem.id) if w.session.problem.id in ids else -1
            pid = ids[(current + 1) % len(ids)]
            w.change_problem(pid)
            return render(w) + (gr.update(value=pid),)
        next_problem.click(advance, [state, picker], outputs + [picker], **common)
        def save_reflection(text, w):
            if w is None:
                return "Your session restarted. This reflection was not saved."
            if w.solved:
                w.reflection = sanitize_reflection(text)
            return "Reflection saved for this session." if w.reflection else "Reflection skipped."
        reflect.click(save_reflection, [reflection, state], reflection_status, api_name=False)
        box.input(lambda s: math_text(s) if s.strip() else "", box, preview, api_name=False, show_progress="hidden", trigger_mode="always_last")
        for button, token, label in [(square, "^2", "Insert exponent 2"), (root, "sqrt()", "Insert square root"), (fraction, "()/()", "Insert fraction"), (pi, "pi", "Insert pi")]:
            button.click(fn=None, inputs=box, outputs=box, js=f"(text) => {{const x=document.querySelector('#step-input textarea');if(x?.disabled)return text;const a=x?.selectionStart??text.length,b=x?.selectionEnd??a;const v=text.slice(0,a)+{token!r}+text.slice(b);setTimeout(()=>x?.focus(),50);return v;}}", api_name=False, queue=False)
        app.load(fn=None, js="""() => {
            const labels={'symbol-square':'Insert exponent 2','symbol-root':'Insert square root','symbol-fraction':'Insert fraction','symbol-pi':'Insert pi'};
            for(const[id,label]of Object.entries(labels)){const b=document.getElementById(id);if(b){b.setAttribute('aria-label',label);b.title=label;}}
            const profiles={quiet:'Quiet',balanced:'Balanced',supportive:'Supportive'};
            const radios=[...document.querySelectorAll('#timing-control input')];
            const group=radios[0]?.closest('label')?.parentElement;
            if(group){group.setAttribute('role','radiogroup');group.setAttribute('aria-label','Hint timing');}
            // Gradio gives these inputs distinct native names; supply conventional radio-group navigation.
            radios.forEach((r,i)=>{
                r.setAttribute('aria-label',profiles[r.value]);r.tabIndex=r.checked?0:-1;
                r.addEventListener('change',()=>radios.forEach(x=>x.tabIndex=x===r?0:-1));
                r.addEventListener('keydown',e=>{
                    let j=i;
                    if(['ArrowRight','ArrowDown'].includes(e.key))j=(i+1)%radios.length;
                    else if(['ArrowLeft','ArrowUp'].includes(e.key))j=(i+radios.length-1)%radios.length;
                    else if(e.key==='Home')j=0;
                    else if(e.key==='End')j=radios.length-1;
                    else return;
                    e.preventDefault();radios[j].click();radios[j].focus();
                });
            });
        }""", api_name=False)
    return app


def sanitize_reflection(text):
    from agent.demo_input import sanitize_input
    return sanitize_input(text)[0][:1000]
