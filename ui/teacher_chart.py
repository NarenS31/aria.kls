"""Teacher view graphics: stuck risk over the attempt and the "other settings" comparison, as server-rendered inline SVG.

Read-only. The risk, bar, decision and reason for every try are the values the tutor already stored on each event
(`pace_*` metadata, the numbers Decision details shows); nothing is recomputed. The comparison replays this attempt's
own answers on throw-away sessions and never touches the live one. No script, font or image is fetched.
"""
from __future__ import annotations

import copy
import html

from agent.demo_summary import RULE_TEXT

ORDER = ("quiet", "balanced", "supportive")
HINT_TYPES = ("hint", "task_chunking")
RIGHT = ("correct", "partial", "step")
W, L, R, TOP, PH = 960, 64, 190, 20, 260
PW = W - L - R
ICON_BULB = '<path d="M10 5a3.6 3.6 0 0 0-2 6.6V13h4v-1.4A3.6 3.6 0 0 0 10 5Zm-1.4 9.4h2.8" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'


def _asked(e):
    m = e.result.meta
    return bool(e.result.intervened and m.get("intervention_type") in HINT_TYPES and m.get("hint_source") == "requested")


def risk_series(w):
    """-> (tries, requested): one dict per verified try, and for each requested hint the number of tries before it."""
    tries, requested = [], []
    for e in w.events:
        m, r = e.result.meta, e.result
        if e.kind == "hint" or _asked(e):
            requested.append(len(tries))
            continue
        if e.kind != "step" or not m.get("pace_updated") or m.get("pace_risk") is None:
            continue
        gave = bool(r.intervened and m.get("intervention_type") in HINT_TYPES)
        tries.append({"n": m.get("pace_n_obs", len(tries) + 1), "risk": m["pace_risk"], "tau": m.get("pace_tau"),
                      "profile": m.get("pace_profile") or e.profile, "right": r.status in RIGHT, "wrong": r.status == "wrong",
                      "hint": gave, "held": r.status == "wrong" and not r.intervened, "reason": m.get("pace_reason"), "rule": m.get("rule_id")})
    return tries, requested


def bars(w):
    """[(key, label, tau)] from the pace config, lowest bar first."""
    cfg = w.tutor.pace_cfg
    return sorted(((p["key"], p["label"], p["tau"]) for p in cfg.profiles()), key=lambda t: t[2]) if cfg else []


def other_settings(w):
    """{setting: [{"n", "hint"}]}: what each setting would have done with this attempt's own answers (None if it can't be replayed).
    Throw-away sessions run the same events through the existing tutor; the live session is not touched."""
    try:
        out = {}
        for key in ORDER:
            s = w.tutor.new_session(w.session.problem.id, graph=copy.deepcopy(w.base_graph))
            s.replaying = True
            s.set_profile(key)
            row = []
            for e in w.events:
                res = s.request_hint() if e.kind == "hint" else s.turn(e.text)
                if res.meta.get("fallback"):
                    return None
                m = res.meta
                if e.kind == "step" and m.get("pace_updated") and m.get("pace_risk") is not None and not _asked(e):
                    row.append({"n": m.get("pace_n_obs", len(row) + 1), "hint": bool(res.intervened and m.get("intervention_type") in HINT_TYPES)})
            out[key] = row
        return out
    except Exception:  # noqa: BLE001
        return None


def first_hint(row):
    return next((t["n"] for t in row if t["hint"]), None)


def _pct(x):
    return f"{x:.0%}"


def _tries_text(ns):
    ns = list(ns)
    if len(ns) == 1:
        return f"try {ns[0]}"
    return "tries " + (", ".join(map(str, ns[:-1])) + f" and {ns[-1]}")


def caption(w, tries, requested):
    if not tries:
        return "No checked steps yet. The chart fills in as the student works."
    parts = []
    hint = next((t for t in tries if t["hint"]), None)
    if hint:
        who = str(hint["profile"]).title()
        if hint["reason"] == "wrong_risk_ge_tau":
            parts.append(f"Risk crossed the {who} bar at try {hint['n']}, so ARIA stepped in there.")
        elif hint["reason"] == "wrong_before_window":
            parts.append(f"ARIA stepped in at try {hint['n']}: the bar only applies from the third try, and earlier wrong steps always get support.")
        elif hint["reason"] == "wrong_after_window":
            parts.append(f"ARIA stepped in at try {hint['n']}: after the ninth try every wrong step gets support.")
        else:
            parts.append(f"ARIA stepped in at try {hint['n']} ({RULE_TEXT.get(hint['rule'], 'a hint was due')}).")
    held = [t for t in tries if t["held"]]
    if held:
        parts.append(f"It held the hint at {_tries_text(t['n'] for t in held)} (risk {', '.join(_pct(t['risk']) for t in held)}).")
    if requested:
        parts.append("The student asked for a hint after " + _tries_text(max(k, 1) for k in requested) + ".")
    if not parts:
        parts.append("Every checked step so far was correct, so ARIA stayed quiet.")
    return " ".join(parts)


def _x(i, n):
    return L + (i - 0.5) * PW / n


def _y(p):
    return TOP + PH * (1 - min(max(p, 0), 1))


def _mark_right(x, y):
    return f'<g class="rc-mk-right"><circle cx="{x:.1f}" cy="{y}" r="12" class="rc-fill-ok"/><path d="m{x - 5.5:.1f} {y + .5}l3.8 3.8 7.2-7.8" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></g>'


def _mark_wrong(x, y):
    return f'<g class="rc-mk-wrong"><rect x="{x - 12:.1f}" y="{y - 12}" width="24" height="24" rx="4" class="rc-fill-issue"/><path d="m{x - 5:.1f} {y - 5}l10 10m0-10-10 10" stroke="#fff" stroke-width="2.8" stroke-linecap="round"/></g>'


def _mark_hint(x, y, key):
    return (f'<g class="rc-mk-hint"><circle cx="{x:.1f}" cy="{y}" r="13" class="rc-fill-{key}"/>'
            f'<g transform="translate({x - 10:.1f} {y - 10})">{ICON_BULB}</g></g>')


def _mark_held(x, y, key):
    return (f'<g class="rc-mk-held"><circle cx="{x:.1f}" cy="{y}" r="12" fill="#fff" class="rc-stroke-{key}" stroke-width="2.5" stroke-dasharray="4 3"/>'
            f'<path d="M{x - 3.5:.1f} {y - 5}v10M{x + 3.5:.1f} {y - 5}v10" class="rc-stroke-{key}" stroke-width="3" stroke-linecap="round"/></g>')


def _mark_requested(x, y):
    return (f'<g class="rc-mk-req"><path d="M{x:.1f} {y - 14}l14 14-14 14-14-14z" class="rc-fill-accent"/>'
            f'<text x="{x:.1f}" y="{y + 5.5}" text-anchor="middle" class="rc-glyph">?</text></g>')


def chart_svg(w, tries, requested):
    active = w.session.profile
    brs = bars(w)
    n = max(len(tries), 4)
    tau_active = next((t for k, _, t in brs if k == active), None)
    y_a, y_b = TOP + PH + 70, TOP + PH + 112
    H = y_b + 30
    o = [f'<svg class="rc-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="rc-title rc-desc" xmlns="http://www.w3.org/2000/svg">',
         '<title id="rc-title">Stuck risk over this attempt</title>',
         f'<desc id="rc-desc">{html.escape(caption(w, tries, requested))}</desc>']
    if tau_active is not None:
        o.append(f'<rect x="{L}" y="{TOP}" width="{PW}" height="{_y(tau_active) - TOP:.1f}" class="rc-above rc-fill-{active}"/>')
        o.append(f'<text x="{L + 12}" y="{TOP + 22}" class="rc-note rc-text-{active}">Above the {html.escape(active.title())} bar, ARIA steps in on a wrong step</text>')
    for g in (0, .25, .5, .75, 1):
        o.append(f'<line x1="{L}" x2="{L + PW}" y1="{_y(g):.1f}" y2="{_y(g):.1f}" class="rc-grid"/>'
                 f'<text x="{L - 10}" y="{_y(g) + 5:.1f}" text-anchor="end" class="rc-axis">{g:.0%}</text>')
    o.append(f'<text transform="translate(16 {TOP + PH / 2}) rotate(-90)" text-anchor="middle" class="rc-axis-title">Stuck risk</text>')
    for key, label, tau in brs:
        cls = "rc-bar rc-bar-on" if key == active else "rc-bar rc-bar-off"
        o.append(f'<g class="{cls} rc-stroke-{key}"><line x1="{L}" x2="{L + PW}" y1="{_y(tau):.1f}" y2="{_y(tau):.1f}"/></g>'
                 f'<text x="{L + PW + 10}" y="{_y(tau) + 5:.1f}" class="rc-bar-label rc-text-{key}{" rc-on" if key == active else ""}">{html.escape(label)} {_pct(tau)}</text>')
    for i in range(1, n + 1):
        o.append(f'<g id="try-{i}" class="rc-col"><rect x="{_x(i, n) - PW / n / 2:.1f}" y="{TOP}" width="{PW / n:.1f}" height="{y_b + 24 - TOP}" class="rc-hl"/>'
                 f'<text x="{_x(i, n):.1f}" y="{TOP + PH + 26}" text-anchor="middle" class="rc-axis">{i}</text></g>')
    o.append(f'<text x="{L + PW / 2}" y="{TOP + PH + 46}" text-anchor="middle" class="rc-axis-title">Try (each checked step)</text>')
    o.append(f'<text x="{L - 10}" y="{y_a + 5}" text-anchor="end" class="rc-axis">Answer</text><text x="{L - 10}" y="{y_b + 5}" text-anchor="end" class="rc-axis">ARIA</text>')
    pts = " ".join(f"{_x(i, n):.1f},{_y(t['risk']):.1f}" for i, t in enumerate(tries, 1))
    if len(tries) > 1:
        o.append(f'<polyline points="{pts}" class="rc-line"/>')
    cross = next((i for i, t in enumerate(tries, 1) if t["tau"] is not None and t["risk"] >= t["tau"] and t["profile"] == active), None)
    for i, t in enumerate(tries, 1):
        x, y = _x(i, n), _y(t["risk"])
        if i == cross:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" class="rc-ring rc-stroke-{active}"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" class="rc-dot" data-try="{t["n"]}" data-risk="{t["risk"]:.4f}"/>'
                 f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" class="rc-val">{_pct(t["risk"])}</text>')
        o.append(_mark_right(x, y_a) if t["right"] else _mark_wrong(x, y_a) if t["wrong"] else "")
        key = str(t["profile"]) if t["profile"] in ORDER else active
        if t["hint"]:
            o.append(_mark_hint(x, y_b, key))
        elif t["held"]:
            o.append(_mark_held(x, y_b, key))
    for k in requested:
        o.append(_mark_requested(_x(max(k, 1), n) + (18 if k else -4), y_b))
    if not tries:
        o.append(f'<text x="{L + PW / 2}" y="{TOP + PH / 2}" text-anchor="middle" class="rc-empty">No checked steps yet</text>')
    o.append("</svg>")
    return "".join(o)


def legend_html():
    s = lambda body: f'<svg class="rc-key-ico" viewBox="0 0 28 28" aria-hidden="true">{body}</svg>'
    items = [(s(_mark_right(14, 14)), "correct step"), (s(_mark_wrong(14, 14)), "incorrect step"),
             (s(_mark_hint(14, 14, "ink")), "ARIA gave a hint"), (s(_mark_held(14, 14, "ink")), "ARIA held the hint"),
             (s(_mark_requested(14, 14)), "student asked for a hint")]
    return '<ul class="rc-legend">' + "".join(f"<li>{i}<span>{t}</span></li>" for i, t in items) + "</ul>"


def alt_list(w, tries, requested):
    items = []
    for t in tries:
        said = "ARIA gave a hint" if t["hint"] else "ARIA held the hint" if t["held"] else "no hint needed"
        items.append(f"<li>Try {t['n']}: {'correct' if t['right'] else 'incorrect'}, stuck risk {_pct(t['risk'])}, bar {_pct(t['tau'])}; {said}.</li>")
    for k in requested:
        items.append(f"<li>The student asked for a hint after try {k}.</li>" if k else "<li>The student asked for a hint before the first checked step.</li>")
    return '<ul class="sr-only">' + "".join(items) + "</ul>"


def strip_svg(w, rows):
    active = w.session.profile
    n = max(max((len(r) for r in rows.values()), default=0), 4)
    rh = 46
    H = rh * 3 + 8
    o = [f'<svg class="rc-svg rc-strip" viewBox="0 0 {W} {H}" role="img" aria-labelledby="st-title" xmlns="http://www.w3.org/2000/svg">'
         '<title id="st-title">What each setting would have done with these same answers</title>']
    for j, key in enumerate(ORDER):
        y, row, fh = rh * j + rh / 2 + 4, rows[key], first_hint(rows[key])
        o.append(f'<line x1="{L}" x2="{L + PW}" y1="{y}" y2="{y}" class="rc-grid"/>')
        seen = False
        for i, t in enumerate(row, 1):
            x = _x(i, n)
            if t["hint"] and not seen:
                seen = True
                o.append(f'<circle cx="{x:.1f}" cy="{y}" r="15" class="rc-ring rc-stroke-{key}"/><circle cx="{x:.1f}" cy="{y}" r="9" class="rc-fill-{key} rc-first"/>')
            elif t["hint"]:
                o.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" class="rc-fill-{key} rc-later"/>')
            else:
                o.append(f'<circle cx="{x:.1f}" cy="{y}" r="9" fill="#fff" class="rc-stroke-{key} rc-wait" stroke-width="2.5"/>')
        label = "no hint yet" if fh is None else f"first hint: try {fh}"
        o.append(f'<text x="{L + PW + 10}" y="{y - 2}" class="rc-strip-name rc-text-{key}">{key.title()}{" (now)" if key == active else ""}</text>'
                 f'<text x="{L + PW + 10}" y="{y + 17}" class="rc-strip-sub">{label}</text>')
    o.append("</svg>")
    return "".join(o)


def strip_legend():
    s = lambda body: f'<svg class="rc-key-ico" viewBox="0 0 28 28" aria-hidden="true">{body}</svg>'
    items = [(s('<circle cx="14" cy="14" r="9" fill="#fff" stroke="#20211D" stroke-width="2.5"/>'), "waited"),
             (s('<circle cx="14" cy="14" r="12" fill="none" stroke="#20211D" stroke-width="2.5"/><circle cx="14" cy="14" r="7" fill="#20211D"/>'), "first hint"),
             (s('<circle cx="14" cy="14" r="5" fill="#20211D"/>'), "later hints")]
    return '<ul class="rc-legend">' + "".join(f"<li>{i}<span>{t}</span></li>" for i, t in items) + "</ul>"


def strip_alt(rows):
    return '<ul class="sr-only">' + "".join(
        f"<li>{k.title()}: " + ("no hint yet" if first_hint(rows[k]) is None else f"first hint at try {first_hint(rows[k])}") + ".</li>" for k in ORDER) + "</ul>"


def teacher_graphics(w):
    """-> (chart_block_html, strip_block_html, first_unsettled_try_or_None)"""
    tries, requested = risk_series(w)
    cap = html.escape(caption(w, tries, requested))
    chart = ('<div class="risk-chart"><h3 id="risk-chart-title">Stuck risk over this attempt</h3>'
             f'{chart_svg(w, tries, requested)}{legend_html()}<p class="rc-caption">{cap}</p>{alt_list(w, tries, requested)}</div>')
    rows = other_settings(w)
    strip = ""
    if rows:
        strip = ('<div class="risk-strip"><h3>Same answers, other settings</h3>'
                 '<p class="rc-sub">What each setting would have done with these same answers.</p>'
                 f'{strip_svg(w, rows)}{strip_legend()}{strip_alt(rows)}</div>')
    focus = next((t["n"] for t in tries if t["wrong"]), None)
    return chart, strip, focus
