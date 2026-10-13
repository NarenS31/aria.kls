"""ARIA summit-demo tutor: *when* to speak, decided from verified math + the paper's policy.

Per student turn
    1. SymPy decides correctness            (agent/math_check.check_answer)   -> status
    2. analyzer marker scoring adds flags   (eval/metacognition/analyzer.py)  -> frustrated / confused / help-seeking
       (flags can ADD Frustrated/Confused/help-seeking; they can never override correctness:
        a correct step is never Stuck, a wrong step is never Flow)
    3. the resulting state goes, unchanged, through RuleBasedJITAIPolicy      -> intervene or stay quiet
    4. words come from templates with SymPy-verified slots; every reply is re-checked with the
       symbolic leak / false-claim checkers before it is shown.
    5. an optional LLM path (flag, default OFF) answers free-form questions only, behind the same checks.

This is NOT the paper's classifier: the paper's analyzer decides state from language; here the
state comes from verified correctness, with the analyzer used only for the affect/help flags.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent import demo_templates as T
from agent.demo_bank import DemoProblem, load_problems
from agent.math_check import check_answer, equivalent, false_claims, matched_terms, normalize, prose_of, reveals_answer, student_exprs, to_expr
from eval.metacognition.analyzer import CognitiveStateAnalyzer
from eval.metacognition.jitai import (
    JITAI_INTERVENTION_BANK,
    JITAIContext,
    RuleBasedJITAIPolicy,
    build_jitai_intervention,
)

MAX_INPUT_CHARS = 1000
LLM_TIMEOUT_S = 10.0
LLM_NUM_PREDICT = 80

_HELP = re.compile(r"\b(i\s*don'?t\s*know|no\s*idea|i\s*give\s*up|i'?m\s*stuck|stuck|help(?:\s*me)?|i\s*can'?t)\b", re.I)
_ASK_ANSWER = re.compile(r"\b(?:just\s+)?(?:tell|give|show)\s+me\s+(?:the\s+)?(?:answer|solution)\b|what'?s\s+the\s+answer|\bjust\s+tell\s+me\b", re.I)
_QUESTION = re.compile(r"\?|^\s*(?:why|how|what|when|where|which|can|could|does|do|is|are)\b", re.I)


@dataclass
class TurnResult:
    reply: str
    intervened: bool
    accepted: Optional[bool]
    watching: bool
    watch_note: str
    state: str
    status: str
    meta: dict = field(default_factory=dict)


def _flags(analysis: dict, text: str) -> dict:
    """Affect / help flags read off the analyzer's marker scores (never used for correctness)."""
    scores = analysis.get("state_scores") or {}
    st, conf = analysis.get("state"), float(analysis.get("confidence") or 0)
    frustrated = (st == "FRUSTRATED" and conf >= 0.6) or scores.get("FRUSTRATED", 0) >= 2.0
    confused = (st == "CONFUSED" and conf >= 0.6) or scores.get("CONFUSED", 0) >= 2.0
    ask_answer = bool(_ASK_ANSWER.search(text))
    # analyzer STUCK / gave_up are its help-seeking evidence; the regex only adds explicit requests
    help_seeking = st == "STUCK" or bool(analysis.get("gave_up")) or bool(_HELP.search(text)) or ask_answer
    return {"frustrated": frustrated, "confused": confused, "help_seeking": help_seeking,
            "ask_answer": ask_answer, "planning": bool(analysis.get("planning_detected"))}


def derive_state(status: str, flags: dict, text: str) -> tuple:
    """-> (state, confidence, kind).  Correctness is the sole source of Flow vs Stuck."""
    if status == "wrong":
        base, kind = "STUCK", "wrong"
    elif status == "correct":
        base, kind = "INSIGHT", "correct"
    elif status in ("partial", "step"):
        base, kind = "FLOW", "progress"
    elif status == "restate":
        base, kind = "PLANNING", "planning"
    else:                                                         # no math content
        if flags["ask_answer"]:
            base, kind = "STUCK", "ask_answer"
        elif flags["help_seeking"]:
            base, kind = "STUCK", "help"
        elif _QUESTION.search(text):
            base, kind = "CONFUSED", "question"
        elif flags["planning"]:
            base, kind = "PLANNING", "planning"
        else:
            base, kind = "UNKNOWN", "unclear"
    conf = 1.0 if status != "none" else 0.8
    # flags add affect on top; they never move wrong->Flow or correct->Stuck
    if flags["frustrated"] and base not in ("INSIGHT",):
        return "FRUSTRATED", conf, "frustrated" if status == "none" else kind
    if flags["confused"] and base in ("FLOW", "PLANNING", "UNKNOWN"):
        return "CONFUSED", conf, "confused" if status == "none" else kind
    return base, conf, kind


class DemoSession:
    def __init__(self, tutor: "DemoTutor", problem: DemoProblem):
        self.tutor, self.problem = tutor, problem
        self.reset()

    def reset(self):
        self.rotor = T.Rotor()
        self.log: list = []
        self.hint_level = 0
        self.prev_state: Optional[str] = None
        self.consecutive = 0
        self.t_last: Optional[float] = None
        self.achieved: list = []          # sympy exprs the student has already got right (terms / steps)

    # ----------------------------------------------------------------------------------
    def turn(self, text) -> TurnResult:
        t0 = time.perf_counter()
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        text = text[:MAX_INPUT_CHARS]
        p, tut = self.problem, self.tutor
        if not text.strip():
            return self._finish(t0, text, "none", "UNKNOWN", {}, None, None, "empty", "", False, None)

        verdict = check_answer(text, p.truth)
        words = prose_of(text)                      # the analyzer scores language, not symbols
        analysis = tut.analyzer.analyze(words) if words.strip() else {}
        flags = _flags(analysis, text)
        state, conf, kind = derive_state(verdict.status, flags, text)
        self.consecutive = self.consecutive + 1 if state == self.prev_state else 1
        self.prev_state = state

        ctx = JITAIContext(
            profile_id="demo-student", topic=p.topic, cognitive_state=state, state_confidence=conf,
            consecutive_state_count=self.consecutive, allow_positive_reflection=(verdict.status == "correct"),
        )
        decision = tut.policy.decide(ctx)
        accepted = True if verdict.status == "correct" else None
        if verdict.status in ("partial", "step", "correct"):
            self.achieved += matched_terms(text, p.truth) + student_exprs(text)

        reply, tpl, llm_used = "", "", False
        if decision.should_intervene:
            intervention = build_jitai_intervention(decision, state)
            reply, tpl, llm_used = self._speak(decision.intervention_type, intervention, state, kind, verdict.status, text)
        return self._finish(t0, text, verdict.status, state, analysis, flags, decision, kind, reply, bool(reply), accepted,
                            tpl=tpl, llm_used=llm_used)

    # ----------------------------------------------------------------------------------
    def _speak(self, itype: str, intervention: dict, state: str, kind: str, status: str, text: str) -> tuple:
        p = self.problem
        move = self._next_move()
        do, result = (move or {}).get("do", ""), (move or {}).get("result") if move and "claim" not in move else None
        bank = intervention["text"]

        if kind == "question" and self.tutor.llm_path:
            llm = self.tutor.llm_answer(p, text)
            if llm:
                return llm, "llm", True

        def render(tpl: str, with_result=True) -> str:
            return T.fill(tpl, bank=bank, do=do, result=result if with_result else None)

        options = T.PHRASINGS[itype] if move is not None else ["{bank}"]      # every authored move already done
        if itype == "reflection":
            lead = self.rotor.pick("ack", T.CORRECT_ACKS, lambda s: s)[0]
        else:
            lead_opts = T.LEADS.get(kind) or [""]
            lead = self.rotor.pick(f"lead-{kind}", lead_opts, lambda s: s)[0]

        body, idx = self.rotor.pick(itype, options, lambda tpl: render(tpl))
        reply = f"{lead} {body}".strip()
        # last line of defence: nothing that states the answer or a false identity leaves this function
        ctx = [p.prompt, text]
        if reveals_answer(reply, p.truth, ctx).leaked or false_claims(reply, p.truth, ctx):
            body = bank
            reply = f"{lead} {body}".strip()
            if reveals_answer(reply, p.truth, ctx).leaked:
                reply = bank
            idx = -1
        if itype != "reflection":
            self.hint_level += 1
        return reply, f"{itype}:{idx}", False

    def _next_move(self) -> Optional[dict]:
        """First authored move whose verified result the student has not produced yet."""
        moves = self.problem.moves
        for m in moves[self.hint_level:] or moves[-1:]:
            res = m.get("result")
            e = to_expr(normalize(res)) if res and "=" not in res else None
            if e is not None and any(equivalent(e, a) for a in self.achieved):
                continue
            return m
        return None if moves else {}

    def _finish(self, t0, text, status, state, analysis, flags, decision, kind, reply, intervened, accepted, tpl="", llm_used=False):
        watching = not intervened
        note = ""
        if watching:
            key = {"progress": "progress", "planning": "planning", "empty": "empty"}.get(kind, "unclear")
            if status in ("partial", "step"):
                key = "progress"
            note = self.rotor.pick(f"watch-{key}", T.WATCHING[key], lambda s: s)[0]
        ms = (time.perf_counter() - t0) * 1000
        meta = {
            "turn": len(self.log) + 1, "latency_ms": round(ms, 2), "correctness": status, "derived_state": state,
            "kind": kind, "analyzer_state": (analysis or {}).get("state"),
            "analyzer_confidence": (analysis or {}).get("confidence"), "flags": flags or {},
            "rule_id": getattr(decision, "rule_id", "input.empty" if kind == "empty" else "none"),
            "intervention_type": getattr(decision, "intervention_type", "none"), "template": tpl,
            "llm_used": llm_used, "hint_level": self.hint_level, "student_chars": len(text),
        }
        self.log.append(meta)
        return TurnResult(reply, intervened, accepted, watching, note, state, status, meta)


class DemoTutor:
    def __init__(self, problems=None, llm_path: bool = False, llm_fn: Optional[Callable[[str, str], str]] = None,
                 llm_model: str = "llama3.2:3b"):
        self.problems = {p.id: p for p in (problems or load_problems())}
        self.analyzer = CognitiveStateAnalyzer(use_llm=False)          # heuristic-only: no LLM in the state path
        self.policy = RuleBasedJITAIPolicy()
        self.llm_path = bool(llm_path)
        self.llm_model = llm_model
        self._llm_fn = llm_fn
        self._pool = ThreadPoolExecutor(max_workers=1)

    def new_session(self, problem_id: str) -> DemoSession:
        return DemoSession(self, self.problems[problem_id])

    # -- gated LLM path: free-form questions only --------------------------------------------
    def llm_answer(self, p: DemoProblem, question: str) -> Optional[str]:
        fn = self._llm_fn or self._ollama_fn
        try:
            fut = self._pool.submit(fn, p.prompt, question)
            out = (fut.result(timeout=LLM_TIMEOUT_S) or "").strip()
        except (FutureTimeout, Exception):                              # noqa: BLE001 - any failure -> template
            return None
        out = re.sub(r"\s+", " ", out)[:300]
        ctx = [p.prompt, question]
        if not out or reveals_answer(out, p.truth, ctx).leaked or false_claims(out, p.truth, ctx):
            return None
        return out

    def _ollama_fn(self, problem: str, question: str) -> str:
        import ollama
        client = ollama.Client(timeout=LLM_TIMEOUT_S)
        r = client.chat(model=self.llm_model, messages=[
            {"role": "system", "content": "You are a math tutor. Never state the final answer. Answer the student's question in at most two short sentences."},
            {"role": "user", "content": f"Problem: {problem}\nStudent asks: {question}"}],
            options={"num_predict": LLM_NUM_PREDICT, "temperature": 0.2})
        return r["message"]["content"]
