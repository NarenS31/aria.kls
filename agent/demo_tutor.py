"""ARIA summit-demo tutor: *when* to speak, decided from verified math + the paper's policy.

Per student turn
    1. SymPy decides correctness            (agent/math_check.check_answer)   -> status
    2. analyzer marker scoring adds flags   (eval/metacognition/analyzer.py)  -> frustrated / confused / help-seeking
       (flags can ADD Frustrated/Confused/help-seeking; they can never override correctness:
        a correct step is never Stuck, a wrong step is never Flow)
    3. the resulting state goes, unchanged, through RuleBasedJITAIPolicy      -> intervene or stay quiet
       ARIA-Pace gate (agent/demo_pace.py): a verified WRONG step updates a per-session BKT tracker. At attempts 3-9 (the frozen
       policy's decision points) ARIA gives a hint only when risk >= tau of the chosen profile; at attempts 1-2 (and 10+) the old
       rule holds: a verified wrong step always gets a hint. Below tau the wrong step is still never ignored: a short "not quite
       yet, no hint yet" notice is shown (no hint, no answer). Help requests, restatements and correct answers are unchanged.
    4. words come from templates with SymPy-verified slots; every reply is re-checked with the
       symbolic leak / false-claim checkers before it is shown.
    5. an optional LLM path (flag, default OFF) answers free-form questions only, behind the same checks.

This is NOT the paper's classifier: the paper's analyzer decides state from language; here the
state comes from verified correctness, with the analyzer used only for the affect/help flags.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent import demo_graph as GRAPH
from agent import demo_pace as PACE
from agent import demo_templates as T
from agent.demo_bank import DemoProblem, load_problems
from agent.demo_input import MAX_INPUT_CHARS, sanitize_input
from agent.demo_log import TurnLog
from agent.math_check import check_answer, equivalent, false_claims, matched_terms, normalize, prose_of, reveals_answer, student_exprs, to_expr
from eval.metacognition.analyzer import _MARKERS, CognitiveStateAnalyzer
from eval.metacognition.jitai import (
    JITAI_INTERVENTION_BANK,
    JITAIContext,
    RuleBasedJITAIPolicy,
    build_jitai_intervention,
)

LLM_TIMEOUT_S = 10.0              # hard cap on one LLM answer; on timeout the template reply is used
LLM_NUM_PREDICT = 80              # generation cap (tokens)
LLM_KEEP_ALIVE = "30m"            # keep the model resident in Ollama between turns
LLM_WARMUP_TIMEOUT_S = 60.0       # first load of the model may take a while; only at startup
LLM_COOLDOWN_S = 20.0             # after a timeout/error, skip the LLM path for this long (no repeated 10 s stalls)
MATH_BUDGET_S = 3.0               # wall-clock budget for the symbolic check of one student message
MAX_STUCK_MATH = 2                # abandoned (over-budget) symbolic checks allowed to run in the background

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


def _trim_to_sentence(text: str) -> str:
    """The 80-token cap can end mid-sentence: keep up to the last full sentence when there is one."""
    if not text or text[-1] in ".?!":
        return text
    cut = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
    return text[:cut + 1] if cut >= 20 else text


_QUESTION_OPENER = re.compile(r"\s*(?:why|what|how)\b", re.I)


def _explicit_affect(state: str, words: str) -> float:
    """Weight of analyzer markers for `state` that the student actually wrote as feeling words.

    Not counted: the analyzer's structural scores (e.g. "circular repetition", which misfires once prose_of() has
    removed the math from a restated problem) and question openers ("why do", "what is", "how does ...?"), which the
    analyzer lists as FRUSTRATED/CONFUSED markers (eval/metacognition/analyzer.py:231,263) but which are just questions.
    """
    total = 0.0
    for pat, weight in _MARKERS.get(state, ()):
        m = pat.search(words)
        if m and not _QUESTION_OPENER.match(m.group(0)):
            total += weight
    return total


def _flags(analysis: dict, text: str, status: str = "none") -> dict:
    """Affect / help flags read off the analyzer's marker scores (never used for correctness).

    A frustrated/confused flag needs the analyzer's verdict AND at least one explicit feeling word in the student's own
    words; a restatement of the problem never raises an affect flag."""
    scores = analysis.get("state_scores") or {}
    st, conf = analysis.get("state"), float(analysis.get("confidence") or 0)
    words = prose_of(text)
    frustrated = ((st == "FRUSTRATED" and conf >= 0.6) or scores.get("FRUSTRATED", 0) >= 2.0) and _explicit_affect("FRUSTRATED", words) > 0
    confused = ((st == "CONFUSED" and conf >= 0.6) or scores.get("CONFUSED", 0) >= 2.0) and _explicit_affect("CONFUSED", words) > 0
    if status == "restate":
        frustrated = confused = False
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
    def __init__(self, tutor: "DemoTutor", problem: DemoProblem, graph=None):
        self.tutor, self.problem = tutor, problem
        self.session_id = uuid.uuid4().hex[:12]
        self.attempt = 0
        self.profile = tutor.pace_profile            # a cost-weight setting, kept across Reset; changed with set_profile()
        # learning graph v0 (agent/demo_graph.py): read-only observer, in memory, exploratory. A graph may be handed over when the
        # visitor switches problem in the UI; Reset clears it.
        self.graph = graph if graph is not None else tutor.new_graph()
        self._reset_attempt()

    def set_profile(self, key: str) -> None:
        if key not in PACE.PROFILE_ORDER:
            raise ValueError(f"unknown profile {key!r}")
        self.profile = key

    def reset(self):
        """One-click new attempt: all per-attempt state, and the learning graph, are cleared; the per-turn file log keeps going."""
        self._reset_attempt()
        if self.graph is not None:
            self.graph.clear()

    def _reset_attempt(self):
        self.attempt += 1
        self._llm_outcome = ""
        self._move_idx = -1               # which authored step the last hint pointed at (-1: none)
        self.t_start = time.time()
        self.rotor = T.Rotor()
        self.log: list = []
        self.hint_level = 0
        self.prev_state: Optional[str] = None
        self.consecutive = 0
        self.t_last: Optional[float] = None
        self.achieved: list = []          # sympy exprs the student has already got right (terms / steps)
        self.pace = PACE.PaceTracker(self.tutor.pace_cfg) if self.tutor.pace_cfg else None
        self._pace_dec = None

    # ----------------------------------------------------------------------------------
    def turn(self, text) -> TurnResult:
        """One student message -> one TurnResult. Never raises: any failure becomes a quiet 'watching' reply."""
        t0 = time.perf_counter()
        clean, in_flags = sanitize_input(text)
        try:
            return self._turn(t0, clean, in_flags)
        except Exception as e:  # noqa: BLE001 - last line of defence; logged by class name only
            return self._fallback(t0, clean, in_flags, f"error:{type(e).__name__}"[:48])

    def _analyze(self, text: str) -> tuple:
        """The symbolic part of a turn (runs under the time budget): verdict, analyzer scores, flags, terms achieved."""
        p, tut = self.problem, self.tutor
        verdict = check_answer(text, p.truth)
        words = prose_of(text)                      # the analyzer scores language, not symbols
        analysis = tut.analyzer.analyze(words) if words.strip() else {}
        flags = _flags(analysis, text, verdict.status)
        terms = (matched_terms(text, p.truth) + student_exprs(text)) if verdict.status in ("partial", "step", "correct") else []
        return verdict, analysis, flags, terms

    def _turn(self, t0: float, text: str, in_flags: list) -> TurnResult:
        p, tut = self.problem, self.tutor
        self._llm_outcome = ""
        self._move_idx = -1
        self._pace_dec = None
        if not text.strip():
            return self._finish(t0, text, "none", "UNKNOWN", {}, None, None, "empty", "", False, None, in_flags=in_flags)

        reason, phase = tut.run_budgeted(self._analyze, text)
        if reason:
            return self._fallback(t0, text, in_flags, reason)
        verdict, analysis, flags, terms = phase
        state, conf, kind = derive_state(verdict.status, flags, text)
        self.consecutive = self.consecutive + 1 if state == self.prev_state else 1
        self.prev_state = state

        ctx = JITAIContext(
            profile_id="demo-student", topic=p.topic, cognitive_state=state, state_confidence=conf,
            consecutive_state_count=self.consecutive, allow_positive_reflection=(verdict.status == "correct"),
        )
        decision = tut.policy.decide(ctx)
        accepted = True if verdict.status == "correct" else None
        self.achieved += terms

        # ARIA-Pace: the verdict updates the BKT tracker; a verified wrong step speaks only if risk >= tau (never silently ignored)
        pace = self.pace.decide(verdict.status, self.profile) if self.pace else None
        self._pace_dec = pace
        flagged = bool(pace and pace.verified_wrong and not pace.speak)
        if pace and pace.speak and not decision.should_intervene:
            # a verified wrong step above tau always gets a hint, whatever the language-based rules said
            decision = tut.policy.decide(JITAIContext(
                profile_id="demo-student", topic=p.topic, cognitive_state="STUCK", state_confidence=conf,
                consecutive_state_count=max(self.consecutive, 2), allow_positive_reflection=False))

        reply, tpl, llm_used = "", "", False
        if flagged:
            decision = None
        elif decision.should_intervene:
            intervention = build_jitai_intervention(decision, state)
            reply, tpl, llm_used = self._speak(decision.intervention_type, intervention, state, kind, verdict.status, text)
        return self._finish(t0, text, verdict.status, state, analysis, flags, decision, kind, reply, bool(reply), accepted,
                            tpl=tpl, llm_used=llm_used, in_flags=in_flags, flagged=flagged)

    def _fallback(self, t0: float, text: str, in_flags: list, reason: str) -> TurnResult:
        """Could not read / check the message in time: stay quiet and say so, never crash, never guess a verdict."""
        self.prev_state, self.consecutive = "UNKNOWN", 1
        return self._finish(t0, text, "none", "UNKNOWN", {}, {}, None, "error", "", False, None, in_flags=in_flags, fallback=reason)

    # ----------------------------------------------------------------------------------
    def _speak(self, itype: str, intervention: dict, state: str, kind: str, status: str, text: str) -> tuple:
        p = self.problem
        move = self._next_move()
        do, result = (move or {}).get("do", ""), (move or {}).get("result") if move and "claim" not in move else None
        bank = intervention["text"]

        if kind == "question" and self.tutor.llm_path:
            llm = self.tutor.llm_answer(p, text)
            self._llm_outcome = self.tutor.last_llm_outcome
            if llm:
                return llm, "llm", True
        # a template hint points at an authored step; a reflection after a correct answer does not
        self._move_idx = -1 if itype == "reflection" else next((i for i, m in enumerate(p.moves) if m is move), -1)

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

    def _screened(self, note: str, text: str) -> str:
        """The same symbolic leak / false-claim checks as every spoken reply (the notice has no math, so this should never fire)."""
        ctx = [self.problem.prompt, text]
        if reveals_answer(note, self.problem.truth, ctx).leaked or false_claims(note, self.problem.truth, ctx):
            return "Watching."
        return note

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

    def _finish(self, t0, text, status, state, analysis, flags, decision, kind, reply, intervened, accepted, tpl="",
                llm_used=False, in_flags=None, fallback="", flagged=False):
        watching = not intervened
        note = ""
        if watching:
            key = {"progress": "progress", "planning": "planning", "empty": "empty", "error": "error"}.get(kind, "unclear")
            if status in ("partial", "step"):
                key = "progress"
            if flagged:
                key = "flagged"
            note = self.rotor.pick(f"watch-{key}", T.WATCHING[key], lambda s: s)[0]
            if flagged:
                reply = note = self._screened(note, text)       # the wrong step is acknowledged; the leak check still runs
        ms = (time.perf_counter() - t0) * 1000
        rule = getattr(decision, "rule_id", {"empty": "input.empty", "error": "system.fallback"}.get(kind, "none"))
        if flagged:
            rule = "pace.wrong_below_tau"
        meta = {
            "turn": len(self.log) + 1, "latency_ms": round(ms, 2), "correctness": status, "derived_state": state,
            "kind": kind, "analyzer_state": (analysis or {}).get("state"),
            "analyzer_confidence": (analysis or {}).get("confidence"), "flags": flags or {},
            "rule_id": rule,
            "intervention_type": getattr(decision, "intervention_type", "none"), "template": tpl,
            "llm_used": llm_used, "llm_outcome": self._llm_outcome, "hint_level": self.hint_level,
            "student_chars": len(text), "reply_chars": len(reply), "input_flags": list(in_flags or []), "fallback": fallback,
            "intervened": bool(intervened), "move_index": self._move_idx if intervened else -1, "t_s": round(time.time() - self.t_start, 1),
        }
        pace = self._pace_dec or (self.pace.snapshot(self.profile) if self.pace else None)
        if pace:
            meta.update({"pace_profile": pace.profile, "pace_tau": pace.tau, "pace_risk": None if pace.risk is None else round(pace.risk, 4),
                         "pace_n_obs": pace.n_obs, "pace_updated": pace.updated, "pace_reason": pace.reason})
        self.log.append(meta)
        if self.graph is not None:                                     # read-only for the policy: runs after the decision, can never raise into a turn
            try:
                self.graph.observe(self.problem, status, bool(intervened) and meta["intervention_type"] in GRAPH.HINT_TYPES)
            except Exception:  # noqa: BLE001
                self.graph.errors += 1
        self.tutor.turn_log.write({**meta, "session_id": self.session_id, "attempt": self.attempt, "problem_id": self.problem.id,
                                   "intervened": intervened, "analyzer_state": meta["analyzer_state"] or ""})
        return TurnResult(reply, intervened, accepted, watching, note, state, status, meta)


class DemoTutor:
    def __init__(self, problems=None, llm_path: bool = False, llm_fn: Optional[Callable[[str, str], str]] = None,
                 llm_model: str = "llama3.2:3b", log_path=None, pace=True, pace_profile: str = PACE.DEFAULT_PROFILE):
        self.problems = {p.id: p for p in (problems or load_problems())}
        self.analyzer = CognitiveStateAnalyzer(use_llm=False)          # heuristic-only: no LLM in the state path
        self.policy = RuleBasedJITAIPolicy()
        self.llm_path = bool(llm_path)
        self.llm_model = llm_model
        self._llm_fn = llm_fn
        self._pool = ThreadPoolExecutor(max_workers=1)                 # LLM calls (one at a time)
        self._math_pool = ThreadPoolExecutor(max_workers=4)            # symbolic checks under a time budget
        self._lock = threading.Lock()
        self._math_stuck = 0
        self._client = None                                            # one persistent Ollama client (model stays loaded)
        self._llm_future = None
        self._block_until = 0.0
        self.llm_ready = False
        self.last_llm_outcome = ""
        self.turn_log = TurnLog(log_path)                              # path None = no file log
        self.pace_cfg = (PACE.load_config() if pace is True else pace) or None     # ARIA-Pace; False/None = old behaviour
        if pace_profile not in PACE.PROFILE_ORDER:
            raise ValueError(f"unknown profile {pace_profile!r}")
        self.pace_profile = pace_profile
        self.concepts = GRAPH.ConceptCatalog(self.problems.values())      # hand-authored concept tags/edges from the bank (validated here)

    def new_session(self, problem_id: str, graph=None) -> DemoSession:
        return DemoSession(self, self.problems[problem_id], graph)

    def new_graph(self):
        """An empty in-memory learning graph (frozen pooled BKT parameters); None if the pace parameters are switched off."""
        return GRAPH.LearningGraph(self.concepts, self.pace_cfg.params) if self.pace_cfg else None

    # -- bounded symbolic work -------------------------------------------------------------------
    def run_budgeted(self, fn, *args) -> tuple:
        """-> ('', value) or (reason, None). SymPy cannot be interrupted, so an over-budget check is abandoned in its
        thread; at most MAX_STUCK_MATH of them may be outstanding, after which new checks are refused immediately."""
        with self._lock:
            if self._math_stuck >= MAX_STUCK_MATH:
                return "math_busy", None
        fut = self._math_pool.submit(fn, *args)
        try:
            return "", fut.result(timeout=MATH_BUDGET_S)
        except FutureTimeout:
            with self._lock:
                self._math_stuck += 1
            fut.add_done_callback(self._release_math)
            return "math_timeout", None
        except Exception as e:  # noqa: BLE001
            return f"error:{type(e).__name__}"[:48], None

    def _release_math(self, _fut):
        with self._lock:
            self._math_stuck = max(0, self._math_stuck - 1)

    # -- gated LLM path: free-form questions only --------------------------------------------
    def warm_up(self) -> bool:
        """Load the model once, at startup, and keep it resident. Returns False (and the LLM path stays on cooldown)
        if Ollama is not reachable; the demo then runs on templates only."""
        if not self.llm_path:
            return False
        if self._llm_fn is not None:
            self.llm_ready = True
            return True
        try:
            import ollama
            ollama.Client(timeout=LLM_WARMUP_TIMEOUT_S).generate(
                model=self.llm_model, prompt="hi", options={"num_predict": 1}, keep_alive=LLM_KEEP_ALIVE)
            self._client = ollama.Client(timeout=LLM_TIMEOUT_S)
            self.llm_ready = True
        except Exception:  # noqa: BLE001
            self.llm_ready = False
            self._block_until = time.monotonic() + LLM_COOLDOWN_S
        return self.llm_ready

    def _fail_llm(self, outcome: str):
        self.last_llm_outcome = outcome
        self._block_until = time.monotonic() + LLM_COOLDOWN_S
        return None

    def llm_answer(self, p: DemoProblem, question: str) -> Optional[str]:
        """Template fallback (None) on: cooldown, a still-running earlier call, timeout, any error, empty output,
        or output that the symbolic leak / false-claim checks reject. Never waits longer than LLM_TIMEOUT_S."""
        self.last_llm_outcome = ""
        if time.monotonic() < self._block_until:
            self.last_llm_outcome = "cooldown"
            return None
        if self._llm_future is not None and not self._llm_future.done():
            self.last_llm_outcome = "busy"
            return None
        fn = self._llm_fn or self._ollama_fn
        try:
            fut = self._pool.submit(fn, p.prompt, question)
            self._llm_future = fut
            out = (fut.result(timeout=LLM_TIMEOUT_S) or "").strip()
        except FutureTimeout:
            return self._fail_llm("timeout")
        except Exception:  # noqa: BLE001 - any failure -> template
            return self._fail_llm("error")
        out = _trim_to_sentence(re.sub(r"\s+", " ", out)[:300])
        ctx = [p.prompt, question]
        if not out:
            self.last_llm_outcome = "empty"
            return None
        if reveals_answer(out, p.truth, ctx).leaked or false_claims(out, p.truth, ctx):
            self.last_llm_outcome = "rejected"
            return None
        self.last_llm_outcome = "ok"
        return out

    def _ollama_fn(self, problem: str, question: str) -> str:
        if self._client is None:
            import ollama
            self._client = ollama.Client(timeout=LLM_TIMEOUT_S)
        r = self._client.chat(model=self.llm_model, messages=[
            {"role": "system", "content": "You are a math tutor. Never state the final answer. Answer the student's question in at most two short sentences."},
            {"role": "user", "content": f"Problem: {problem}\nStudent asks: {question}"}],
            options={"num_predict": LLM_NUM_PREDICT, "temperature": 0.2}, keep_alive=LLM_KEEP_ALIVE)
        return r["message"]["content"]
