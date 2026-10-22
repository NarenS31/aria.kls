#!/usr/bin/env python
"""v2 DRAFT: ARIA's decision on each activity moment under each ARIA-Pace cost profile (quiet / balanced / supportive).

    /usr/bin/python3 scripts/make_moment_profiles.py --repo /path/to/checkout-of-summit-pace-mac --turns data/teacher_moments_turns.json \\
        --out data/teacher_moments_profiles.json

The moments of v1 were generated from the pre-pace tag (summit-2026-10-24), where cost profiles do not exist. This script replays the same owner-supplied
turns through the pace build (branch summit-pace-mac, read only: imported from --repo, nothing is written there) once per profile, in a fresh session each time
(history replayed first, so the BKT tracker sees the same verdicts), and records ARIA's decision for the moment: `step_in` if it spoke (a hint, prompt, ...),
`stay_quiet` if it did not (a flagged-for-teacher notice under ARIA-Pace counts as quiet and is marked `flagged`). Template-only; LLM path off; no log written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROFILES = ("quiet", "balanced", "supportive")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from agent.demo_bank import load_problems
    from agent.demo_tutor import DemoTutor
    problems = {p.id: p for p in load_problems()}
    tutor = DemoTutor(problems=list(problems.values()), log_path=None)
    out = {}
    for t in json.loads(Path(a.turns).read_text(encoding="utf-8")):
        out[t["id"]] = {}
        for prof in PROFILES:
            s = tutor.new_session(t["problem_id"])
            s.set_profile(prof)
            for h in t.get("history", []):
                s.turn(h)
            r = s.turn(t["message"])
            out[t["id"]][prof] = {"decision": "step_in" if r.intervened else "stay_quiet",
                                  "intervention_type": r.meta["intervention_type"] if r.intervened else "",
                                  "state": r.state, "verdict": r.status,
                                  "flagged": (r.meta.get("pace_reason") or "") == "wrong_risk_lt_tau",     # agent/demo_pace.py:207
                                  "pace_reason": r.meta.get("pace_reason") or ""}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for mid, row in out.items():
        print(mid, {p: v["decision"] + ("*" if v["flagged"] else "") for p, v in row.items()}, "state", row["balanced"]["state"])
    print("(* = flagged for the teacher, not spoken)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
