#!/usr/bin/env python
"""Regenerate the 'Would you step in?' moments from THIS build: run the unmodified DemoTutor over turns supplied by the owner.

    /usr/bin/python3 scripts/make_teacher_moments.py --turns data/teacher_moments_turns.json --out data/teacher_moments.json

turns file: a JSON list of exactly 8 objects
    {"id": "m1", "problem_id": "calc-01", "history": ["I don't know", "12x^3"], "message": "12x^3 - 10x - 9"}
`history` are the student's earlier messages (may be empty); `message` is the moment itself. The tutor replays history then message in a fresh
session; the GROUND TRUTH for the moment is ARIA's logged decision on `message` (intervened / intervention_type / rule). `words_only` and
`verified_wrong` are computed from SymPy's verdict on `message`, never typed. LLM path off, no log written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.demo_bank import load_problems  # noqa: E402
from agent.demo_input import sanitize_input  # noqa: E402
from agent.demo_summary import RULE_TEXT  # noqa: E402
from agent.demo_tutor import DemoTutor  # noqa: E402
from agent.math_check import check_answer  # noqa: E402
from agent.teacher_activity import N_MOMENTS, aria_category  # noqa: E402


def build(turns: list, expect: int | None = N_MOMENTS) -> list:
    if expect is not None and len(turns) != expect:
        raise SystemExit(f"need exactly {expect} moments, got {len(turns)}")
    problems = {p.id: p for p in load_problems()}
    tutor = DemoTutor(problems=list(problems.values()))
    out, seen = [], set()
    for t in turns:
        pid, mid = t["problem_id"], t["id"]
        if pid not in problems:
            raise SystemExit(f"{mid}: unknown problem_id {pid!r}")
        if mid in seen:
            raise SystemExit(f"duplicate id {mid!r}")
        seen.add(mid)
        s = tutor.new_session(pid)
        for h in t.get("history", []):
            s.turn(h)
        r = s.turn(t["message"])
        verdict = check_answer(sanitize_input(t["message"])[0], problems[pid].truth).status
        stepped = bool(r.intervened)
        out.append({
            "id": mid, "problem_id": pid, "problem_prompt": problems[pid].prompt, "history": [sanitize_input(h)[0] for h in t.get("history", [])],
            "message": sanitize_input(t["message"])[0],
            "aria_decision": "step_in" if stepped else "stay_quiet",
            "aria_type": aria_category(r.meta["intervention_type"]) if stepped else "",
            "aria_intervention_type": r.meta["intervention_type"] if stepped else "",
            "aria_reason": RULE_TEXT.get(r.meta["rule_id"], r.meta["rule_id"]),
            "aria_reply": r.reply if stepped else "",
            "aria_state": r.state, "verdict": verdict,
            "words_only": verdict == "none", "verified_wrong": verdict == "wrong",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "teacher_moments.json"))
    a = ap.parse_args()
    moments = build(json.loads(Path(a.turns).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(moments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for m in moments:
        print(f"{m['id']:6s} {m['problem_id']:8s} ARIA {m['aria_decision']:10s} {m['aria_type']:13s} words_only={m['words_only']!s:5s} verified_wrong={m['verified_wrong']!s:5s} | {m['message'][:50]!r}")
    print(f"wrote {a.out} ({len(moments)} moments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
