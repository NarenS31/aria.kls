"""Teacher summary of one demo attempt: state timeline, hints given, where the student got stuck.

Built ONLY from the per-turn metadata the tutor already keeps (`DemoSession.log`) and the authored problem steps.
It contains no student text and no tutor text, so it cannot become a chat transcript. See agent/demo_tutor.py `_finish`.
"""
from __future__ import annotations

from agent.math_check import equivalent, normalize, to_expr
from agent.demo_progress import completed_moves

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
    "pace.wrong_below_tau": "wrong step, risk below this profile's bar: flagged, no hint yet",
    "input.empty": "empty message: nothing to read",
    "system.fallback": "could not read that message: stayed quiet",
}
VERDICT_LABELS = {
    "correct": "correct answer", "partial": "part of the answer", "step": "a correct step", "restate": "restated the problem",
    "wrong": "incorrect", "none": "no math to check",
}
ACTION_LABELS = {
    "hint": "gave a hint", "task_chunking": "split it into a smaller step", "encouragement": "encouraged",
    "break_suggestion": "suggested a short pause", "reflection": "asked for reflection", "none": "stayed quiet",
}
HINT_TYPES = ("hint", "task_chunking")
STUCK_STATES = ("STUCK", "FRUSTRATED", "CONFUSED")
NOTE = ("State is read from verified correctness (SymPy) plus language cues in one attempt. It describes this attempt; "
        "it is not a diagnosis or a grade, and no chat transcript is kept here.")


def _step_text(problem, idx: int):
    if 0 <= idx < len(problem.moves):
        return problem.moves[idx].get("do")
    return None


def _steps_done(problem, achieved) -> list:
    """Per authored step: True / False, or None when the step is an equation we do not track."""
    out = []
    for m in problem.moves:
        res = m.get("result")
        if not res or "=" in res:
            out.append(None)
            continue
        e = to_expr(normalize(res))
        out.append(bool(e is not None and any(equivalent(e, a) for a in achieved)))
    return out


def summarize(session) -> dict:
    p, log = session.problem, list(session.log)
    timeline, hints = [], []
    for row in log:
        itype = row.get("intervention_type", "none") if row.get("intervened") else "none"
        acted = bool(row.get("intervened"))
        step = _step_text(p, row.get("move_index", -1)) if acted else None
        timeline.append({
            "turn": row["turn"], "t_s": row.get("t_s"), "state": row["derived_state"],
            "state_label": STATE_LABELS.get(row["derived_state"], row["derived_state"].lower()),
            "verdict": row["correctness"], "verdict_label": VERDICT_LABELS.get(row["correctness"], row["correctness"]),
            "action": itype, "action_label": ACTION_LABELS.get(itype, itype),
            "why": RULE_TEXT.get(row["rule_id"], row["rule_id"]), "step": step, "fallback": row.get("fallback", ""),
        })
        if acted:
            hints.append({"turn": row["turn"], "type": itype, "type_label": ACTION_LABELS.get(itype, itype),
                          "step_index": row.get("move_index", -1), "step": step, "hint_level_after": row.get("hint_level"),
                          "source": "llm" if row.get("llm_used") else "template"})
    stretches, cur = [], None
    for t in timeline:
        if t["state"] in STUCK_STATES:
            if cur is None:
                cur = {"from_turn": t["turn"], "to_turn": t["turn"], "states": [], "step": None, "resolved": False}
            cur["to_turn"] = t["turn"]
            if t["state"] not in cur["states"]:
                cur["states"].append(t["state"])
            if cur["step"] is None and t["step"]:
                cur["step"] = t["step"]
        elif cur is not None:
            cur["resolved"] = True
            stretches.append(cur)
            cur = None
    if cur is not None:
        stretches.append(cur)
    for s in stretches:
        s["length"] = s["to_turn"] - s["from_turn"] + 1
        s["state_labels"] = [STATE_LABELS.get(x, x.lower()) for x in s["states"]]
    longest = max(stretches, key=lambda s: (s["length"], -s["from_turn"]), default=None)
    solved_at = next((t["turn"] for t in timeline if t["verdict"] == "correct"), None)
    done = completed_moves(session)
    hinted = {}
    for h in hints:
        if h["step_index"] >= 0:
            hinted.setdefault(h["step_index"], []).append(h["turn"])
    steps = [{"index": i, "step": m.get("do"), "done": done[i], "hinted_at_turns": hinted.get(i, [])} for i, m in enumerate(p.moves)]
    by_type = {}
    for h in hints:
        by_type[h["type"]] = by_type.get(h["type"], 0) + 1
    return {
        "problem": {"id": p.id, "level": p.level, "topic": p.topic, "prompt": p.prompt},
        "attempt": getattr(session, "attempt", 1),
        "outcome": "not_started" if not log else ("solved" if solved_at else "in_progress"),
        "solved_at_turn": solved_at,
        "elapsed_s": log[-1].get("t_s") if log else None,
        "counts": {"turns": len(log), "stayed_quiet": sum(t["action"] == "none" for t in timeline),
                   "requested_hints": sum(r.get("hint_source") == "requested" and r.get("intervened") for r in log),
                   "automatic_hints": sum(r.get("hint_source") == "automatic" and r.get("intervened") and r.get("intervention_type") in HINT_TYPES for r in log),
                   "stepped_in": len(hints), "hints_given": sum(by_type.get(k, 0) for k in HINT_TYPES), "by_type": by_type},
        "timeline": timeline, "hints": hints, "stuck": {"stretches": stretches, "longest": longest}, "steps": steps, "note": NOTE,
        # learning graph v0 (agent/demo_graph.py): ids, authored labels, states, counts only; None until something was scored
        "concept_map": (session.graph.snapshot() if getattr(session, "graph", None) is not None and session.graph.has_data() else None),
    }


def headline(summary: dict) -> str:
    """One sentence: where did this student get stuck."""
    lg = summary["stuck"]["longest"]
    if not summary["timeline"]:
        return "No attempt yet."
    if lg is None:
        return "No stuck turns in this attempt."
    span = f"turn {lg['from_turn']}" if lg["length"] == 1 else f"turns {lg['from_turn']} to {lg['to_turn']} ({lg['length']} turns)"
    on = f", at the step \"{lg['step']}\"" if lg["step"] else ""
    end = ", then moved on" if lg["resolved"] else ", and was still stuck at the end"
    return f"Longest stuck stretch: {span}{on}{end}."
