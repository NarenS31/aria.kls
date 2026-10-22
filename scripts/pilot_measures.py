#!/usr/bin/env python
"""Pilot measures computed from the demo's per-turn log only (data/demo_logs/turns.jsonl; fields whitelisted in agent/demo_log.py:20-27).

    /usr/bin/python3 scripts/pilot_measures.py [--log data/demo_logs/turns.jsonl] [--problems data/demo_problems.json] [--json out.json]

Definitions (docs/aria_pace/PILOT_MEASURES.md): an "attempt" = one (session_id, attempt, problem_id) group of turns.
  (a) correction after a hint   next SCORED turn after a hint turn is right (correct / partial / step); scored = right or wrong
  (b) hints per problem; hint_level reached   hint_level is a pointer into authored steps that counts every non-reflection reply. It is NOT a three-level engine.
  (c) time before the first hint   t_s at the first hint turn (seconds since the attempt started). A PROXY for productive struggle, not a measurement of it.
  (d) friction index   share of attempts flagged high-friction by the exploratory rule of agent/demo_graph.py (not secure, >= 2 wrong, >= 1 hint). Exploratory.
Only counts, verdict strings and times are read; the log holds no student or tutor text. Nothing is written unless --json is given.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HINT_TYPES = ("hint", "task_chunking")            # agent/demo_summary.py HINT_TYPES
RIGHT = ("correct", "partial", "step")
WRONG = ("wrong",)
FRICTION_WRONG, FRICTION_HINTS, SECURE_RUN = 2, 1, 3   # agent/demo_graph.py
DEFAULT_LOG = ROOT / "data" / "demo_logs" / "turns.jsonl"
DEFAULT_BANK = ROOT / "data" / "demo_problems.json"


def load_rows(path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "session_id" in r and "problem_id" in r and "turn" in r:
                rows.append(r)
    return rows


def attempts(rows) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[(r["session_id"], r.get("attempt", 1), r["problem_id"])].append(r)
    return {k: sorted(v, key=lambda r: r["turn"]) for k, v in g.items()}


def is_hint(r) -> bool:
    return bool(r.get("intervened")) and r.get("intervention_type") in HINT_TYPES


def scored(r) -> bool:
    return r.get("correctness") in RIGHT or r.get("correctness") in WRONG


def _q(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def friction_flag(turns) -> bool:
    sc = [t["correctness"] for t in turns if scored(t)]
    if len(sc) >= SECURE_RUN and all(c in RIGHT for c in sc[-SECURE_RUN:]):
        return False
    return sum(c in WRONG for c in sc) >= FRICTION_WRONG and sum(is_hint(t) for t in turns) >= FRICTION_HINTS


def measures(rows, bank: dict = None) -> dict:
    att = attempts(rows)
    out = {"turns": len(rows), "attempts": len(att)}
    # (a) correction after a hint
    n_h = n_next = n_right = 0
    f_h = f_next = f_right = 0
    for turns in att.values():
        first = True
        for i, t in enumerate(turns):
            if not is_hint(t):
                continue
            nxt = next((u for u in turns[i + 1:] if scored(u)), None)
            n_h += 1
            n_next += nxt is not None
            n_right += nxt is not None and nxt["correctness"] in RIGHT
            if first:
                f_h += 1
                f_next += nxt is not None
                f_right += nxt is not None and nxt["correctness"] in RIGHT
                first = False
    out["a_correction_after_hint"] = {"hints": n_h, "with_a_later_scored_attempt": n_next, "next_attempt_right": n_right,
                                      "share_right": (n_right / n_next) if n_next else None,
                                      "first_hint_only": {"hints": f_h, "with_a_later_scored_attempt": f_next, "next_attempt_right": f_right,
                                                          "share_right": (f_right / f_next) if f_next else None}}
    # (b) hints per problem and hint_level reached
    per = defaultdict(lambda: {"attempts": 0, "hints": 0, "max_hint_level": []})
    for (sid, a, pid), turns in att.items():
        r = per[pid]
        r["attempts"] += 1
        r["hints"] += sum(is_hint(t) for t in turns)
        r["max_hint_level"].append(max([t.get("hint_level") or 0 for t in turns] + [0]))
    allh = [sum(is_hint(t) for t in turns) for turns in att.values()]
    allm = [max([t.get("hint_level") or 0 for t in turns] + [0]) for turns in att.values()]
    out["b_hints"] = {"attempts_with_a_hint": sum(h > 0 for h in allh),
                      "hints_per_attempt_mean": statistics.mean(allh) if allh else None,
                      "max_hint_level_mean": statistics.mean(allm) if allm else None, "max_hint_level_max": max(allm) if allm else None,
                      "note": "hint_level counts every non-reflection reply and points at authored steps; it is not a three-level engine",
                      "per_problem": {pid: {"attempts": r["attempts"], "hints": r["hints"], "hints_per_attempt": r["hints"] / r["attempts"],
                                            "max_hint_level_mean": statistics.mean(r["max_hint_level"])} for pid, r in sorted(per.items())}}
    # (c) time before the first hint (proxy)
    ts, before = [], []
    for turns in att.values():
        i = next((k for k, t in enumerate(turns) if is_hint(t)), None)
        if i is None or turns[i].get("t_s") is None:
            continue
        ts.append(float(turns[i]["t_s"]))
        before.append(sum(scored(u) for u in turns[:i]))
    out["c_time_before_first_hint_s"] = {"attempts_with_a_hint": len(ts), "median": _q(ts, 0.5), "q1": _q(ts, 0.25), "q3": _q(ts, 0.75),
                                         "scored_attempts_before_first_hint_mean": statistics.mean(before) if before else None,
                                         "label": "proxy for productive struggle, not validated"}
    # (d) exploratory friction index
    flags = {k: friction_flag(t) for k, t in att.items()}
    out["d_friction_index"] = {"attempts": len(att), "flagged_high_friction": sum(flags.values()),
                               "share": (sum(flags.values()) / len(att)) if att else None,
                               "label": "exploratory, not validated (agent/demo_graph.py rule)"}
    if bank:
        by_c = defaultdict(lambda: [0, 0])
        for (sid, a, pid), fl in flags.items():
            for c in bank.get(pid, {}).get("concepts", []):
                by_c[c][0] += 1
                by_c[c][1] += int(fl)
        out["d_friction_index"]["per_concept"] = {c: {"attempts": n, "flagged": f, "share": f / n} for c, (n, f) in sorted(by_c.items())}
    return out


def fmt(x, nd=2):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_table(m: dict, bank: dict = None):
    a, b, c, d = m["a_correction_after_hint"], m["b_hints"], m["c_time_before_first_hint_s"], m["d_friction_index"]
    print(f"Turns: {m['turns']}   attempts (session x problem): {m['attempts']}")
    print("\n(a) Correction after a hint")
    print(f"    hints {a['hints']} | with a later scored attempt {a['with_a_later_scored_attempt']} | next attempt right {a['next_attempt_right']} | share right {fmt(a['share_right'])}")
    f = a["first_hint_only"]
    print(f"    first hint of each attempt only: {f['hints']} hints | later scored {f['with_a_later_scored_attempt']} | right {f['next_attempt_right']} | share {fmt(f['share_right'])}")
    print("    (descriptive; no comparison group, so it does not show the hint caused the correction)")
    print("\n(b) Hints per problem; hint_level reached (a pointer into authored steps, NOT a three-level engine)")
    print(f"    attempts with a hint {b['attempts_with_a_hint']} | hints per attempt {fmt(b['hints_per_attempt_mean'])} | max hint_level mean {fmt(b['max_hint_level_mean'])}, max {fmt(b['max_hint_level_max'])}")
    print(f"    {'problem':10s} {'attempts':>8s} {'hints':>6s} {'hints/att':>9s} {'max hint_level (mean)':>22s}")
    for pid, r in b["per_problem"].items():
        print(f"    {pid:10s} {r['attempts']:8d} {r['hints']:6d} {r['hints_per_attempt']:9.2f} {r['max_hint_level_mean']:22.2f}")
    print("\n(c) Time before the first hint, t_s (PROXY for productive struggle, not validated)")
    print(f"    attempts with a hint {c['attempts_with_a_hint']} | median {fmt(c['median'], 1)} s | quartiles {fmt(c['q1'], 1)} to {fmt(c['q3'], 1)} s | scored attempts before it, mean {fmt(c['scored_attempts_before_first_hint_mean'])}")
    print("\n(d) Friction index (EXPLORATORY, not validated)")
    print(f"    attempts flagged high-friction {d['flagged_high_friction']} of {d['attempts']} (share {fmt(d['share'])})")
    for cid, r in (d.get("per_concept") or {}).items():
        label = (bank or {}).get("_labels", {}).get(cid, cid)
        print(f"    {label:40s} {r['flagged']}/{r['attempts']}")


def load_bank(path) -> dict:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    bank = {p["id"]: {"concepts": p.get("concepts", [])} for p in d if isinstance(p, dict) and "id" in p}
    bank["_labels"] = {cid: lab for p in d if isinstance(p, dict) for cid, lab in (p.get("concept_labels") or {}).items()}
    return bank


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--problems", default=str(DEFAULT_BANK))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if not Path(a.log).exists():
        print(f"no log at {a.log} (the demo writes it unless started with --no-log)")
        return 1
    rows = load_rows(a.log)
    if not rows:
        print("the log has no turns")
        return 1
    bank = load_bank(a.problems)
    m = measures(rows, {k: v for k, v in bank.items() if k != "_labels"} or None)
    print_table(m, bank)
    if a.json:
        Path(a.json).write_text(json.dumps(m, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
