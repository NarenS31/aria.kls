#!/usr/bin/env python
"""Read-only stress test of the summit demo build: 50,000 simulated student turns through the real DemoSession.

    python scripts/stress_50k.py --repo /path/to/checkout-of-summit-pace-mac [--turns 50000] [--workers 6] [--seed 20261022]

The demo code is imported from --repo (a git worktree of branch summit-pace-mac) and is NEVER modified: no bytecode is
written, the tutor is built with log_path=None (no turn log), and nothing is written into --repo. Output goes to --out
(JSON, default: stress_50k.json next to this script's cwd).

Coverage: all 30 problems x the 3 ARIA-Pace profiles (90 cells, equal turn budget), sessions of 3-12 turns that mix
well-formed and fuzzed student input:
  well-formed   correct / unsimplified / LaTeX-wrapped correct / common wrong answer / correct intermediate step / restatement /
                "I don't know" / "just tell me the answer" / free-form question / frustrated message
  fuzzed        malformed algebra syntax, unmatched parentheses, extreme numbers and exponents, unicode look-alikes and LaTeX
                wrappers, empty and whitespace-only input, control characters, very long input, rapid duplicate submissions
  sequences     stuck -> wrong x k -> step -> correct, same wrong answer repeated, etc. (each session is ONE DemoSession, so the
                BKT tracker and the pace gate see several verdicts)

Measured per turn (definitions follow scripts/demo_harness.py on the same branch):
  latency            wall clock of session.turn() only
  crash              an exception escaping turn()
  fallback           meta["fallback"] non-empty (the tutor's own SymPy time-budget / error fallback, agent/demo_tutor.py:236-239)
  leak               agent.math_check.reveals_answer(reply, truth, [problem prompt, student text]).leaked
  false claim        agent.math_check.false_claims(reply, truth, ...) non-empty
  math error         the reply approves a turn that is not correct, or rejects a correct one (harness verdict_in_text rule)
  silent wrong step  a verified wrong step (SymPy verdict "wrong", from the oracle for well-formed input and from the session's own
                     verdict for fuzzed input) that got an empty reply
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

PROFILES = ("quiet", "balanced", "supportive")
WELL_FORMED = ("correct", "unsimplified", "latex", "wrong", "step", "restate", "idk", "ask", "question", "frustrated")
FUZZ = ("malformed", "parens", "extreme", "unicode", "latexwrap", "empty", "whitespace", "control", "long", "dup")
EXTREME = ["10**1000", "x**99999999", "9" * 400, "2**(2**64)", "factorial(100000)", "(10**1000)*x + 1", "sqrt(10**1000)",
           "1e999999", "x = 10^1000", "x = " + "9" * 300, "(10**1000)!", "x^(10^10)", "10^10^10^10", "1/0", "0**0 = x",
           "x = 1" + "0" * 5000, "e**(e**(e**100))", "sin(10**1000)"]
QUESTIONS = {"algebra": "Why do we have to do the same thing to both sides?",
             "trig": "What is the unit circle actually for?",
             "calculus": "Why does the power go down by one when we differentiate?"}
SEQUENCES = [
    ["idk", "wrong", "wrong", "wrong", "step", "correct"], ["wrong", "wrong", "wrong", "wrong", "wrong", "correct"],
    ["step", "step", "correct"], ["restate", "step", "unsimplified"], ["ask", "wrong", "wrong", "idk", "correct"],
    ["frustrated", "wrong", "wrong", "wrong", "frustrated", "step", "correct"], ["question", "restate", "step", "correct", "correct"],
    ["wrong", "dup", "wrong", "wrong", "latex"], ["latex", "latex", "wrong", "correct"],
    ["wrong"] * 9 + ["step", "correct"], ["idk", "idk", "idk", "ask", "ask", "step"],
]


def fuzz_text(cat: str, p, rng: random.Random, last: str) -> str:
    base = p.msg_correct() or "x"
    if cat == "malformed":
        s = list(base)
        for _ in range(rng.randint(1, 4)):
            op = rng.choice("del ins dup swap".split())
            i = rng.randrange(max(1, len(s)))
            if op == "del" and s:
                s.pop(i % len(s))
            elif op == "ins":
                s.insert(i, rng.choice("+-*/^=(),.!_{}[]<>|\\"))
            elif op == "dup" and s:
                s.insert(i, s[i % len(s)] * rng.randint(2, 4))
            elif op == "swap" and len(s) > 1:
                j = rng.randrange(len(s))
                s[i % len(s)], s[j] = s[j], s[i % len(s)]
        return "".join(s)
    if cat == "parens":
        k = rng.randint(1, 60)
        return rng.choice(["(" * k + base, base + ")" * k, "(" * k, ")" * k, base.replace("(", "", 1), "[" * k + base + "}" * k])
    if cat == "extreme":
        return rng.choice(EXTREME)
    if cat == "unicode":
        t = base
        for a, b in (("-", "−"), ("*", "×"), ("x", "х"), ("0", "０"), ("1", "١"), ("=", "＝"), ("^", "²")):
            if rng.random() < 0.5:
                t = t.replace(a, b)
        return t + rng.choice(["", " \U0001f600", " ‮", " \u0000", " ​", " éè你好"])
    if cat == "latexwrap":
        return rng.choice(["$$ " + base + " $$", "\\[ " + base + " \\]", "\\( " + base + " \\)", "\\text{" + base + "}",
                           "\\begin{align} " + base + " \\end{align}", "\\frac{" + base + "}{1}", "$" + base + "$",
                           "\\left(" + base + "\\right)", "\\sqrt{" + base + "}"])
    if cat == "empty":
        return ""
    if cat == "whitespace":
        return rng.choice([" ", "   ", "\n", "\t\t", " \n \t \r\n"])
    if cat == "control":
        return rng.choice(["\x00", "\x07\x08", "x\x00=\x001\x007", "\x1b[31mx = 5\x1b[0m", "﻿x = 5", "x = 5  "])
    if cat == "long":
        return rng.choice([base * rng.randint(50, 400), ("word " * rng.randint(2000, 20000)), "x+" * rng.randint(500, 20000) + "1",
                           "1" * rng.randint(5000, 100000)])
    if cat == "dup":
        return last or base
    raise ValueError(cat)


def text_for(cat: str, p, rng: random.Random, last: str) -> str:
    if cat == "correct":
        return p.msg_correct()
    if cat == "unsimplified":
        return p.msg_unsimplified()
    if cat == "latex":
        return p.msg_latex()
    if cat == "wrong":
        return p.msg_wrong()
    if cat == "step":
        return p.msg_step() or p.msg_correct()
    if cat == "restate":
        return p.msg_restate()
    if cat == "idk":
        return rng.choice(["I don't know", "idk", "i have no idea what to do", "I'm stuck"])
    if cat == "ask":
        return rng.choice(["just tell me the answer", "what is the answer?", "give me the answer please"])
    if cat == "question":
        return QUESTIONS[p.topic]
    if cat == "frustrated":
        return rng.choice(["ugh this makes no sense, I keep getting it wrong and I hate this", "I'm so frustrated, this is confusing"])
    return fuzz_text(cat, p, rng, last)


def build_session(rng: random.Random, remaining: int) -> list:
    if rng.random() < 0.45:
        cats = list(rng.choice(SEQUENCES))
        # sprinkle fuzz into a sequence
        for _ in range(rng.randint(0, 2)):
            cats.insert(rng.randrange(len(cats) + 1), rng.choice(FUZZ))
    else:
        cats = [rng.choice(WELL_FORMED + FUZZ) for _ in range(rng.randint(3, 12))]
    return cats[:remaining]


def worker(args):
    repo, cells, seed, out_path = args
    sys.path.insert(0, repo)
    res = {"turns": [], "failures": [], "worker_error": None}
    try:
        from agent.demo_bank import load_problems
        from agent.demo_tutor import DemoTutor
        from agent.demo_input import sanitize_input
        from agent.math_check import check_answer, false_claims, reveals_answer
        sys.path.insert(0, str(Path(repo) / "scripts"))
        from demo_harness import verdict_in_text
        problems = load_problems()
        by_id = {p.id: p for p in problems}
        tutor = DemoTutor(problems=problems, log_path=None)
        rng = random.Random(seed)

        def fail(kind, p, profile, cat, msg, detail=""):
            res["failures"].append({"kind": kind, "problem": p.id, "profile": profile, "cat": cat, "input_len": len(msg),
                                    "input": msg[:300], "detail": str(detail)[:300]})

        for pid, profile, budget in cells:
            p = by_id[pid]
            done = 0
            while done < budget:
                cats = build_session(rng, budget - done)
                s = tutor.new_session(pid)
                s.set_profile(profile)
                last = ""
                for cat in cats:
                    msg = text_for(cat, p, rng, last)
                    if cat != "dup":
                        last = msg
                    reps = 3 if cat == "dup" else 1                       # rapid duplicate submissions
                    for _ in range(reps):
                        if done >= budget:
                            break
                        done += 1
                        rec = {"cat": cat, "p": pid, "prof": profile, "lat": 0.0, "crash": 0, "fb": "", "leak": 0, "false": 0,
                               "mathe": 0, "silent": 0, "empty": int(msg.strip() == ""), "status": "", "reply_empty": 0}
                        t0 = time.perf_counter()
                        try:
                            r = s.turn(msg)
                        except Exception as e:  # noqa: BLE001
                            rec["lat"] = (time.perf_counter() - t0) * 1000
                            rec["crash"] = 1
                            res["turns"].append(rec)
                            fail("crash", p, profile, cat, msg, f"{type(e).__name__}: {e}")
                            continue
                        rec["lat"] = (time.perf_counter() - t0) * 1000
                        reply = r.reply or ""
                        rec["status"] = r.status
                        rec["reply_empty"] = int(not reply.strip())
                        rec["fb"] = (r.meta or {}).get("fallback") or ""
                        ctx = [p.prompt, msg]
                        try:
                            if reveals_answer(reply, p.truth, ctx).leaked:
                                rec["leak"] = 1
                                fail("leak", p, profile, cat, msg, reply)
                            fc = false_claims(reply, p.truth, ctx)
                            if fc:
                                rec["false"] = 1
                                fail("false_claim", p, profile, cat, msg, fc)
                        except Exception as e:  # noqa: BLE001
                            rec["crash"] = 1
                            fail("checker_crash", p, profile, cat, msg, f"{type(e).__name__}: {e}")
                        # SymPy oracle (agent.math_check.check_answer on the sanitized text, as the session sees it: agent/demo_tutor.py:177,
                        # agent/demo_input.py:32). Skipped for the
                        # extreme / very long inputs, where the session's own verdict stands in so the oracle cannot hang the run.
                        oracle = r.status
                        if cat not in ("extreme", "long"):
                            try:
                                oracle = check_answer(sanitize_input(msg)[0], p.truth).status
                            except Exception:  # noqa: BLE001
                                pass
                        said = r.accepted if r.accepted is not None else verdict_in_text(reply)
                        if said is True and oracle not in ("correct", "partial", "step"):
                            rec["mathe"] = 1
                            fail("math_error", p, profile, cat, msg, f"approves a turn SymPy rates '{oracle}': {reply}")
                        if said is False and oracle == "correct":
                            rec["mathe"] = 1
                            fail("math_error", p, profile, cat, msg, f"rejects a correct answer: {reply}")
                        verified_wrong = r.status == "wrong" or oracle == "wrong"
                        if verified_wrong and not reply.strip():
                            rec["silent"] = 1
                            fail("silent_wrong_step", p, profile, cat, msg, f"status {r.status}")
                        if rec["fb"]:
                            fail("fallback", p, profile, cat, msg, rec["fb"])
                        res["turns"].append(rec)
        res["math_stuck_at_end"] = tutor._math_stuck
    except Exception:  # noqa: BLE001
        res["worker_error"] = traceback.format_exc()
    Path(out_path).write_text(json.dumps(res))
    os._exit(0)                                              # abandoned SymPy threads must not block the exit


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="checkout of the demo build (branch summit-pace-mac); never modified")
    ap.add_argument("--turns", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default="stress_50k.json")
    ap.add_argument("--timeout-s", type=int, default=7200, help="stop waiting for a worker after this long")
    a = ap.parse_args(argv)
    repo = str(Path(a.repo).resolve())
    sys.path.insert(0, repo)
    from agent.demo_bank import load_problems
    ids = [p.id for p in load_problems()]
    cells = [(pid, prof) for pid in ids for prof in PROFILES]
    base, extra = divmod(a.turns, len(cells))
    cells = [(pid, prof, base + (1 if i < extra else 0)) for i, (pid, prof) in enumerate(cells)]
    shards = [cells[i::a.workers] for i in range(a.workers)]
    tmp = Path(a.out).with_suffix(".parts")
    tmp.mkdir(exist_ok=True)
    procs = []
    t_start = time.time()
    for i, sh in enumerate(shards):
        outp = tmp / f"w{i}.json"
        if outp.exists():
            outp.unlink()
        pr = mp.Process(target=worker, args=((repo, sh, a.seed + i, str(outp)),), daemon=True)
        pr.start()
        procs.append((pr, outp))

    def _stop(*_):                                       # a killed parent must not leave orphaned workers behind
        for pr, _o in procs:
            if pr.is_alive():
                pr.kill()
        os._exit(130)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    parts = []
    for pr, outp in procs:
        pr.join(max(1, a.timeout_s - (time.time() - t_start)))
        if pr.is_alive():
            pr.kill()
        parts.append(json.loads(outp.read_text()) if outp.exists() else {"turns": [], "failures": [], "worker_error": "no output (killed)"})
    turns = [t for pt in parts for t in pt["turns"]]
    fails = [f for pt in parts for f in pt["failures"]]
    errs = [pt["worker_error"] for pt in parts if pt.get("worker_error")]
    lat = [t["lat"] for t in turns]
    kinds = {}
    for f in fails:
        kinds.setdefault(f["kind"], []).append(f)
    by_cat = {}
    for t in turns:
        by_cat.setdefault(t["cat"], []).append(t["lat"])
    summary = {
        "repo_head": os.popen(f"git -C {repo} rev-parse --short HEAD").read().strip(),
        "turns": len(turns), "seed": a.seed, "workers": a.workers, "wall_s": round(time.time() - t_start, 1),
        "problems": len(ids), "profiles": list(PROFILES), "worker_errors": errs,
        "latency_ms": {"p50": pct(lat, .5), "p95": pct(lat, .95), "p99": pct(lat, .99), "max": max(lat) if lat else None,
                       "mean": statistics.fmean(lat) if lat else None},
        "crashes": sum(t["crash"] for t in turns), "leaks": sum(t["leak"] for t in turns),
        "false_claims": sum(t["false"] for t in turns), "math_errors": sum(t["mathe"] for t in turns),
        "silent_wrong_steps": sum(t["silent"] for t in turns),
        "fallbacks": sum(1 for t in turns if t["fb"]),
        "fallback_reasons": {k: sum(1 for t in turns if t["fb"] == k) for k in sorted({t["fb"] for t in turns if t["fb"]})},
        "verified_wrong_turns": sum(1 for t in turns if t["status"] == "wrong"),
        "empty_input_turns": sum(t["empty"] for t in turns),
        "latency_by_category_ms": {c: {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "max": max(v)} for c, v in sorted(by_cat.items())},
        "turns_by_category": {c: len(v) for c, v in sorted(by_cat.items())},
        "turns_by_profile": {pr: sum(1 for t in turns if t["prof"] == pr) for pr in PROFILES},
        "problems_covered": len({t["p"] for t in turns}),
        "slow_turns_over_5s": sum(1 for x in lat if x > 5000),
        "failure_examples": {k: v[:25] for k, v in kinds.items()},
        "failure_counts": {k: len(v) for k, v in kinds.items()},
        "math_stuck_at_end": [pt.get("math_stuck_at_end") for pt in parts],
    }
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    ok = (summary["leaks"] == 0 and summary["math_errors"] == 0 and summary["crashes"] == 0 and summary["latency_ms"]["p95"] < 5000
          and not errs and summary["turns"] == a.turns)
    print(json.dumps({k: summary[k] for k in ("repo_head", "turns", "wall_s", "latency_ms", "crashes", "leaks", "false_claims", "math_errors",
                                              "silent_wrong_steps", "fallbacks", "fallback_reasons", "slow_turns_over_5s", "failure_counts")}, indent=1))
    print("TARGETS MET" if ok else "TARGETS NOT MET (see failure_examples in " + a.out + ")")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
