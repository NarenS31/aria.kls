#!/usr/bin/env python
"""Behavioural stress test (NOT transfer evidence): rule-based synthetic calculus students run through the demo tutor with the frozen ARIA-Pace gate.

    cd <a checkout of branch summit-pace-mac> && /usr/bin/python3 <path>/synthetic_calculus_stress.py --repo . --json out.json

Acceptance criteria, fixed before the run: 0 crashes, 0 answer leaks, 0 math errors, 0 silent verified-wrong steps, p95 turn latency < 5 s, and the
mean number of tutor interventions per trajectory is non-decreasing from Quiet to Balanced to Supportive and from the fluent to the intermittent to the
stuck-prone student type. Nothing else is claimed: the students are scripts (their choices do not react to the tutor), the bank has few distinct messages
per problem, and no LLM is used. Students (all rule-based):
    fluent        P(wrong at the start of a problem or after a right step) = 0.10, P(right after a wrong step) = 0.80
    intermittent  0.35 / 0.50
    stuck-prone   0.60 / 0.20
After two wrong steps in a row a student asks for help ("I don't know") with probability 0.3 instead of attempting. The first two right attempts type the problem's
authored correct intermediate step (when it has one); the next right attempt types the correct answer and ends the trajectory; 14 turns at most.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

TYPES = {"fluent": (0.10, 0.80), "intermittent": (0.35, 0.50), "stuck_prone": (0.60, 0.20)}
PROFILES = ("quiet", "balanced", "supportive")
STUDENTS_PER_CELL = 40
MAX_TURNS = 14
SEED = 20261102


def trajectory(p, ptype, rng):
    """List of student messages (rule-based; independent of the tutor's replies)."""
    p_wrong, p_recover = TYPES[ptype]
    msgs, rights, last_wrong, wrongs_in_row = [], 0, False, 0
    while len(msgs) < MAX_TURNS:
        if wrongs_in_row >= 2 and rng.rand() < 0.3:
            msgs.append("I don't know")
            wrongs_in_row = 0 if False else wrongs_in_row
            continue
        right = (rng.rand() < p_recover) if last_wrong else (rng.rand() >= p_wrong)
        if right:
            rights += 1
            step = p.msg_step()
            if step and rights <= 2:
                msgs.append(step)
                last_wrong, wrongs_in_row = False, 0
            else:
                msgs.append(p.msg_correct())
                break
        else:
            msgs.append(p.msg_wrong())
            last_wrong, wrongs_in_row = True, wrongs_in_row + 1
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout containing agent/demo_tutor.py with the ARIA-Pace gate (branch summit-pace-mac)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from agent.demo_bank import load_problems
    from agent.demo_tutor import DemoTutor
    from agent.math_check import check_answer, false_claims, reveals_answer

    problems = [p for p in load_problems() if p.id.startswith("calc")]
    out = {"status": "behavioural stress test, not transfer evidence", "seed": SEED, "students_per_type_problem": STUDENTS_PER_CELL,
           "problems": [p.id for p in problems], "cells": {}}
    t0 = time.time()
    for prof in PROFILES:
        tutor = DemoTutor(pace_profile=prof, problems=problems)
        for ti, ptype in enumerate(TYPES):
            agg = {"trajectories": 0, "turns": 0, "interventions": 0, "flagged_wrong": 0, "verified_wrong": 0, "silent_wrong": 0, "leaks": 0, "math_errors": 0,
                   "crashes": 0, "degraded": 0, "solved": 0, "lat": []}
            for pi, p in enumerate(problems):
                rng = np.random.RandomState(SEED + 1000 * ti + pi)
                for _ in range(STUDENTS_PER_CELL):
                    msgs = trajectory(p, ptype, rng)
                    s = tutor.new_session(p.id)
                    agg["trajectories"] += 1
                    for m in msgs:
                        t1 = time.perf_counter()
                        try:
                            r = s.turn(m)
                        except Exception:  # noqa: BLE001
                            agg["crashes"] += 1
                            continue
                        agg["lat"].append((time.perf_counter() - t1) * 1000)
                        agg["turns"] += 1
                        agg["interventions"] += int(bool(r.intervened))
                        if r.meta.get("fallback"):
                            agg["degraded"] += 1
                        ctx = [p.prompt, m]
                        if r.reply:
                            agg["leaks"] += int(reveals_answer(r.reply, p.truth, ctx).leaked)
                            agg["math_errors"] += len(false_claims(r.reply, p.truth, ctx))
                        if check_answer(m, p.truth).status == "wrong":
                            agg["verified_wrong"] += 1
                            agg["silent_wrong"] += int(not r.reply.strip())
                            agg["flagged_wrong"] += int(not r.intervened and bool(r.reply.strip()))
                    agg["solved"] += int(any(check_answer(m, p.truth).status == "correct" for m in msgs))
            lat = sorted(agg.pop("lat"))
            agg["p95_ms"] = lat[int(0.95 * (len(lat) - 1))] if lat else None
            agg["interventions_per_trajectory"] = agg["interventions"] / agg["trajectories"]
            agg["turns_per_trajectory"] = agg["turns"] / agg["trajectories"]
            out["cells"][f"{prof}|{ptype}"] = agg
            print(f"[{time.time() - t0:4.0f}s] {prof:10s} {ptype:12s} traj={agg['trajectories']} turns/traj={agg['turns_per_trajectory']:.1f} interventions/traj={agg['interventions_per_trajectory']:.2f} "
                  f"flagged_wrong={agg['flagged_wrong']} silent_wrong={agg['silent_wrong']} leaks={agg['leaks']} math_err={agg['math_errors']} crashes={agg['crashes']} p95={agg['p95_ms']:.0f}ms", flush=True)
    c = out["cells"]
    tot = lambda k: sum(v[k] for v in c.values())  # noqa: E731
    mono_profile = all(c[f"quiet|{t}"]["interventions_per_trajectory"] <= c[f"balanced|{t}"]["interventions_per_trajectory"] <= c[f"supportive|{t}"]["interventions_per_trajectory"] for t in TYPES)
    mono_type = all(c[f"{pf}|fluent"]["interventions_per_trajectory"] <= c[f"{pf}|intermittent"]["interventions_per_trajectory"] <= c[f"{pf}|stuck_prone"]["interventions_per_trajectory"] for pf in PROFILES)
    out["criteria"] = {"crashes==0": tot("crashes") == 0, "leaks==0": tot("leaks") == 0, "math_errors==0": tot("math_errors") == 0, "silent_wrong==0": tot("silent_wrong") == 0,
                       "p95<5s": max(v["p95_ms"] for v in c.values()) < 5000, "interventions_nondecreasing_quiet_balanced_supportive": bool(mono_profile),
                       "interventions_nondecreasing_fluent_intermittent_stuck": bool(mono_type), "degraded_turns": tot("degraded")}
    print(json.dumps(out["criteria"], indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0 if all(v for k, v in out["criteria"].items() if k != "degraded_turns") else 1


if __name__ == "__main__":
    sys.exit(main())
