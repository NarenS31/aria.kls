#!/usr/bin/env python
"""Summit-demo reliability harness.

For each of the 30 demo problems (data/demo_problems.json) it plays scripted student turns
against a tutor backend, each in a FRESH session:

    T1 correct answer            T2 common wrong answer     T3 "I don't know"
    T4 gibberish                 T5 "just tell me the answer"
    T6 correct but unsimplified (e.g. x = 34/2, "7 - 10x + 12x^3")
    P1 restates the problem      P2 a correct intermediate step (where authored)

and reports, per backend:
    answer-leak rate     symbolic check (agent/math_check.reveals_answer), exemptions = problem + student text
    math-error rate      a math claim in the reply that SymPy refutes, OR a verdict that contradicts SymPy
                         (praising a wrong/empty turn, rejecting a correct one)
    crashes / empties    exceptions, or no reply at all
    latency p50 / p95    wall clock per turn (timeouts count at the cap)
    timing behaviour     quiet on correct intermediate steps (P1,P2); intervened on wrong / stuck (T2,T3,T5)
    false rejects        correct answers (T1,T6) the tutor did not accept

Targets (from the brief): 0 leaks, 0 math errors, 0 crashes, p95 < 5 s.  `--strict` exits 1 if missed.

Backends
    legacy            the shipping think_aloud_turn product path (no LLM needed; baseline "before")
    ollama:<tag>      raw LLM tutor, e.g. ollama:llama3.2:3b / ollama:llama3.1:8b (needs a local Ollama)
    demo              the summit DemoTutor (agent/demo_tutor.py; step C)
    selftest-*        deliberately flawed fakes that prove this harness can see leaks / lies / crashes

Examples
    python scripts/demo_harness.py --backend legacy
    python scripts/demo_harness.py --backend ollama:llama3.2:3b --backend ollama:llama3.1:8b
    python scripts/demo_harness.py --backend demo --strict --log /tmp/turns.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.demo_bank import DemoProblem, load_problems  # noqa: E402
from agent.math_check import check_answer, false_claims, reveals_answer  # noqa: E402

TARGET_P95_MS = 5000.0
HARD_TIMEOUT_S = 10.0

# (turn id, expectation, description)
#   accept   tutor should accept the answer           quiet     tutor should not intervene
#   intervene tutor should step in                    any       no expectation (robustness only)
TURNS = [
    ("T1", "accept", "correct answer"),
    ("T2", "intervene", "common wrong answer"),
    ("T3", "intervene", "I don't know"),
    ("T4", "any", "gibberish"),
    ("T5", "intervene", "just tell me the answer"),
    ("T6", "accept", "correct, unsimplified"),
    ("P1", "quiet", "restates the problem"),
    ("P2", "quiet", "correct intermediate step"),
    ("Q1", "intervene", "free-form question"),
    ("F1", "intervene", "frustrated"),
    ("U1", "accept", "correct, LaTeX-wrapped"),
    ("E1", "any", "empty input"),
    ("L1", "any", "2000-char paste"),
]
QUESTIONS = {"algebra": "Why do we have to do the same thing to both sides?",
             "trig": "What is the unit circle actually for?",
             "calculus": "Why does the power go down by one when we differentiate?"}
GIBBERISH = "asdf qwer zxcv jkl;"


def student_message(p: DemoProblem, tid: str) -> Optional[str]:
    return {
        "T1": p.msg_correct(), "T2": p.msg_wrong(), "T3": "I don't know", "T4": GIBBERISH,
        "T5": "just tell me the answer", "T6": p.msg_unsimplified(),
        "P1": p.msg_restate(), "P2": p.msg_step(),
        "Q1": QUESTIONS[p.topic], "F1": "ugh this makes no sense, I keep getting it wrong and I hate this",
        "U1": p.msg_latex(), "E1": "", "L1": ("I think I should try something with the first term and then maybe the next one. " * 40)[:2000],
    }[tid]


def sympy_status(p: DemoProblem, tid: str, msg: str) -> str:
    """What SymPy says about the scripted message (the oracle for verdict checks)."""
    return check_answer(msg, p.truth).status


# ----------------------------------------------------------------------------- backends
@dataclass
class Reply:
    text: str
    intervened: Optional[bool] = None      # None = backend does not say; inferred from text
    accepted: Optional[bool] = None        # structured verdict if the backend has one
    meta: dict = field(default_factory=dict)


class Backend:
    name = "base"

    def turn(self, p: DemoProblem, text: str) -> Reply:          # fresh session each call
        raise NotImplementedError


class LegacyBackend(Backend):
    """Shipping product path: ARIAAgent.think_aloud_turn (templates, no LLM). Built the same
    way the Phase-1 audit probe built it, with outcome files redirected to a temp dir."""
    name = "legacy"

    def __init__(self):
        import agent.reasoning as R
        self.R = R
        R._INTERVENTION_OUTCOMES = Path(tempfile.mkdtemp(prefix="aria_harness_")) / "outcomes.json"

    def turn(self, p: DemoProblem, text: str) -> Reply:
        R = self.R
        a = object.__new__(R.ARIAAgent)
        a.profile = {}
        a.model = "harness-no-llm"
        a._coaching_trace = deque(maxlen=8)
        a._recent_coaching_responses = deque(maxlen=20)
        a._coaching_turn_index = 0
        a._meta_last_prompt = ""
        a._meta_transfer = None
        a._meta_tracker = None
        a._intervention_pipeline = None
        a._latest_intervention_outcome = None
        a._last_coaching_meta = {}
        a._current_problem_ctx = {
            "task_id": p.id, "subject": "Math", "topic": p.topic.title(), "difficulty": "medium",
            "task_type": "answer_key", "problem": p.prompt, "answer": p.truth.answer_text(),
            "solution_steps": [m["do"] + "." for m in p.moves] or ["Work one step at a time."],
            "key_ideas": ["one step at a time"], "common_misconceptions": ["skipping a step"],
        }
        r = a.think_aloud_turn(text)
        sm = r.get("situation_model") or {}
        # legacy's own internal verdict (reasoning.py: _build_situation_model): True = "step is correct"
        verdict = sm.get("proposed_step_is_correct")
        return Reply(str(r.get("question", "")), bool(r.get("intervened")), verdict if verdict is True else None,
                     {"action": r.get("policy_action"), "internal_correct": verdict})


class OllamaBackend(Backend):
    """Raw LLM tutor with the same 80-token / 10 s limits the demo will use. This is the
    'scaffolds every turn' baseline AND the measurement for the gated LLM path."""

    def __init__(self, tag: str, num_predict: int = 80, timeout: float = HARD_TIMEOUT_S):
        import ollama
        self.name = f"ollama:{tag}"
        self.tag, self.num_predict = tag, num_predict
        self.client = ollama.Client(timeout=timeout)
        self.client.generate(model=tag, prompt="hi", options={"num_predict": 1})        # warm-up, not timed

    def turn(self, p: DemoProblem, text: str) -> Reply:
        system = ("You are a math tutor for a high-school student. Never state the final answer. "
                  "Reply in at most two short sentences and end with one question.")
        user = f"Problem: {p.prompt}\nStudent says: {text}"
        r = self.client.chat(model=self.tag, messages=[{"role": "system", "content": system},
                                                       {"role": "user", "content": user}],
                             options={"num_predict": self.num_predict, "temperature": 0.2})
        return Reply(r["message"]["content"].strip(), True, None)


class DemoBackend(Backend):
    """The summit DemoTutor (Step C). Expected duck-typed API:
        DemoTutor.new_session(problem_id) -> session ; session.turn(text) -> object with
        .reply (str), .intervened (bool), .accepted (bool|None)"""
    name = "demo"

    def __init__(self, **kw):
        try:
            from agent.demo_tutor import DemoTutor
        except ImportError as e:                                   # pragma: no cover
            raise SystemExit(f"--backend demo needs agent/demo_tutor.py (Step C): {e}")
        self.tutor = DemoTutor(**kw)

    def turn(self, p: DemoProblem, text: str) -> Reply:
        s = self.tutor.new_session(p.id)
        r = s.turn(text)
        return Reply(r.reply, bool(r.intervened), getattr(r, "accepted", None), getattr(r, "meta", {}) or {})


# --- self-test fakes: prove the harness can see each failure class -------------------------
class _SelfTest(Backend):
    def __init__(self, kind: str):
        self.kind, self.name = kind, f"selftest-{kind}"

    def turn(self, p: DemoProblem, text: str) -> Reply:
        st = check_answer(text, p.truth).status
        stuck = st in ("wrong", "none") and text.strip() != ""
        if self.kind == "oracle":      # perfect-by-construction: SymPy decides, never leaks
            if st == "correct":
                return Reply("That's right. What made that step click?", True, True)
            if st in ("partial", "step", "restate"):
                return Reply("", False, None)
            return Reply("Not quite yet. What is the smallest useful next step?", True, False)
        if self.kind == "leaker":
            return Reply(f"The answer is {p.truth.answer_text()}.", True, None)
        if self.kind == "liar":
            return Reply("That's right, great job! And 2 + 2 = 5.", True, True)
        if self.kind == "chatty":
            return Reply("What is one small clue you can use first?", True, None)   # scaffolds every turn
        if self.kind == "crasher":
            raise RuntimeError("boom")
        if self.kind == "slow":
            time.sleep(0.05)
            return Reply("Take a breath. What is the first thing the problem asks?", True, None)
        raise ValueError(self.kind)


def make_backend(spec: str) -> Backend:
    if spec == "legacy":
        return LegacyBackend()
    if spec == "demo":
        return DemoBackend()
    if spec.startswith("ollama:"):
        return OllamaBackend(spec.split(":", 1)[1])
    if spec.startswith("selftest-"):
        return _SelfTest(spec.split("-", 1)[1])
    raise SystemExit(f"unknown backend {spec!r}")


# ----------------------------------------------------------------------------- judging
_APPROVE = re.compile(
    r"^\s*(?:yes[,!. ]*)?(?:that'?s|that is|you'?re|it'?s)\s+(?:correct|right|it)\b|^\s*correct\b"
    r"|\b(?:exactly|well done|great (?:job|work)|nice (?:work|job)|spot on|perfect|you got it)\b", re.I)
_DISAPPROVE = re.compile(
    r"\b(?:not quite|incorrect|isn'?t (?:quite )?(?:right|correct)|not (?:right|correct)|wrong|try again|almost there)\b", re.I)


def verdict_in_text(text: str) -> Optional[bool]:
    if _DISAPPROVE.search(text):
        return False
    if _APPROVE.search(text):
        return True
    return None


@dataclass
class TurnResult:
    problem: str
    turn: str
    expect: str
    student: str
    sympy_status: str
    reply: str
    intervened: Optional[bool]
    accepted: Optional[bool]
    latency_ms: float
    leak: bool = False
    leak_detail: list = field(default_factory=list)
    math_error: bool = False
    math_detail: list = field(default_factory=list)
    crash: Optional[str] = None
    empty: bool = False
    timeout: bool = False


def run_turn(backend: Backend, p: DemoProblem, tid: str, expect: str, msg: str) -> TurnResult:
    status = sympy_status(p, tid, msg)
    t0 = time.perf_counter()
    res = TurnResult(p.id, tid, expect, msg, status, "", None, None, 0.0)
    try:
        r = backend.turn(p, msg)
        res.reply, res.intervened, res.accepted = r.text or "", r.intervened, r.accepted
    except Exception as e:  # noqa: BLE001
        res.crash = f"{type(e).__name__}: {e}"[:200]
        res.timeout = "timeout" in res.crash.lower() or "timed out" in res.crash.lower()
    res.latency_ms = (time.perf_counter() - t0) * 1000
    if res.timeout:
        res.latency_ms = max(res.latency_ms, HARD_TIMEOUT_S * 1000)
    if res.crash:
        return res
    # backends that do not report 'intervened' are judged by whether they said anything
    if res.intervened is None:
        res.intervened = bool(res.reply.strip())
    res.empty = (expect in ("accept", "intervene")) and not res.reply.strip()
    ctx = [p.prompt, msg]
    lk = reveals_answer(res.reply, p.truth, ctx)
    res.leak, res.leak_detail = lk.leaked, lk.matches
    errs = [f"false claim: {c}" for c in false_claims(res.reply, p.truth, ctx)]
    said = res.accepted if res.accepted is not None else verdict_in_text(res.reply)
    res.accepted = said
    if said is True and (status == "wrong" or tid in ("T3", "T4", "T5")):      # praising wrong/empty/gibberish turns
        errs.append(f"approves a turn SymPy says is '{status}'")
    if said is False and status == "correct":
        errs.append("rejects a correct answer")
    res.math_error, res.math_detail = bool(errs), errs
    return res


def run(backend: Backend, problems: list, log=None, progress=True) -> list:
    out = []
    for i, p in enumerate(problems):
        for tid, expect, _ in TURNS:
            msg = student_message(p, tid)
            if msg is None:
                continue
            r = run_turn(backend, p, tid, expect, msg)
            out.append(r)
            if log:
                log.write(json.dumps({"backend": backend.name, **r.__dict__}, ensure_ascii=False) + "\n")
        if progress:
            print(f"\r  {backend.name}: {i + 1}/{len(problems)} problems", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return out


# ----------------------------------------------------------------------------- reporting
def pct(vals: list, q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list) -> dict:
    n = len(results)
    ok = [r for r in results if not r.crash]
    lat = [r.latency_ms for r in results]
    by = lambda ids: [r for r in ok if r.turn in ids]          # noqa: E731
    quiet, stuck, acc = by({"P1", "P2"}), by({"T2", "T3", "T5"}), by({"T1", "T6"})
    wrong = by({"T2"})
    return {
        "turns": n,
        "leaks": sum(r.leak for r in results),
        "math_errors": sum(r.math_error for r in results),
        "crashes": sum(bool(r.crash) for r in results),
        "empty_replies": sum(r.empty for r in results),
        "timeouts": sum(r.timeout for r in results),
        "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95), "max_ms": max(lat) if lat else float("nan"),
        "quiet_ok": sum(not r.intervened for r in quiet), "quiet_n": len(quiet),
        "intervened_ok": sum(bool(r.intervened) for r in stuck), "intervened_n": len(stuck),
        "wrong_not_praised": sum(r.accepted is not True for r in wrong), "wrong_n": len(wrong),
    }


def targets_met(s: dict) -> dict:
    return {"leaks==0": s["leaks"] == 0, "math_errors==0": s["math_errors"] == 0,
            "crashes==0": s["crashes"] == 0, "p95<5s": s["p95_ms"] < TARGET_P95_MS}


def print_report(name: str, results: list):
    s = summarize(results)
    t = targets_met(s)
    print(f"\n=== {name}   ({s['turns']} turns, {len(set(r.problem for r in results))} problems) ===")
    rows = [
        ("answer leaks", f"{s['leaks']}/{s['turns']}  ({100 * s['leaks'] / s['turns']:.1f}%)", t["leaks==0"]),
        ("math errors", f"{s['math_errors']}/{s['turns']}  ({100 * s['math_errors'] / s['turns']:.1f}%)", t["math_errors==0"]),
        ("crashes", f"{s['crashes']}  (timeouts {s['timeouts']}, empty replies {s['empty_replies']})", t["crashes==0"]),
        ("latency p50 / p95 / max", f"{s['p50_ms']:.0f} / {s['p95_ms']:.0f} / {s['max_ms']:.0f} ms", t["p95<5s"]),
    ]
    for k, v, ok in rows:
        print(f"  {k:26s} {v:42s} {'OK ' if ok else 'MISS'}")
    print("  -- timing behaviour (reported, not a pass/fail target here) --")
    print(f"  {'quiet on correct steps':26s} {s['quiet_ok']}/{s['quiet_n']}  (P1 restate + P2 correct intermediate step)")
    print(f"  {'intervened when stuck':26s} {s['intervened_ok']}/{s['intervened_n']}  (T2 wrong, T3 'I don't know', T5 'tell me')")
    print(f"  {'wrong answer not praised':26s} {s['wrong_not_praised']}/{s['wrong_n']}")
    acc_known = [r for r in results if r.turn in ("T1", "T6") and not r.crash]
    accepted = sum(r.accepted is True for r in acc_known)
    rejected = sum(r.accepted is False for r in acc_known)
    print(f"  {'correct answers accepted':26s} {accepted}/{len(acc_known)} said so, {rejected} rejected, "
          f"{len(acc_known) - accepted - rejected} gave no verdict  (T1 + T6)")
    print("  per turn type:  turn  n   intervened  leaks  math_err  crash   p95 ms")
    for tid, _, desc in TURNS:
        rs = [r for r in results if r.turn == tid]
        if not rs:
            continue
        print(f"    {tid} {desc:27s} {len(rs):2d}  {sum(bool(r.intervened) for r in rs):6d}  {sum(r.leak for r in rs):6d}  "
              f"{sum(r.math_error for r in rs):6d}  {sum(bool(r.crash) for r in rs):5d}  {pct([r.latency_ms for r in rs], .95):7.0f}")
    bad = [r for r in results if r.leak or r.math_error or r.crash]
    for r in bad[:8]:
        why = r.crash or "; ".join(r.leak_detail and [f"LEAK {r.leak_detail}"] or []) + " ".join(r.math_detail)
        print(f"    ! {r.problem} {r.turn}: {why[:150]}  | reply: {r.reply[:90]!r}")
    if len(bad) > 8:
        print(f"    ... and {len(bad) - 8} more flagged turns")
    return s, t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="append", default=[], help="legacy | demo | ollama:<tag> | selftest-<kind> (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems")
    ap.add_argument("--problems", default=None, help="path to a problems json (default data/demo_problems.json)")
    ap.add_argument("--log", default=None, help="write one json line per turn here")
    ap.add_argument("--json", default=None, help="write the summary json here")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless all four targets are met")
    a = ap.parse_args(argv)
    problems = load_problems(a.problems)
    if a.limit:
        problems = problems[: a.limit]
    log = open(a.log, "w", encoding="utf-8") if a.log else None
    summary, missed = {}, False
    for spec in a.backend or ["legacy"]:
        be = make_backend(spec)
        res = run(be, problems, log)
        s, t = print_report(be.name, res)
        summary[be.name] = {**s, "targets": t}
        missed |= not all(t.values())
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))
    return 1 if (a.strict and missed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
