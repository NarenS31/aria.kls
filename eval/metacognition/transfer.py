"""
Metacognitive transfer detection for ARIA.

The core research question this module answers is *not* "did the student
respond to a metacognitive prompt?" but "did the student start monitoring their
own thinking WITHOUT ARIA prompting them?" — self-initiated metacognition is
the actual learning outcome. When a student begins planning, checking, and
reflecting on their own, ARIA's scaffolding has been internalised. That is
transfer.

Transfer is intentionally stricter than detecting planning language. A transfer
event is confirmed only when an unprompted PLAN or MONITORING move appears on a
different task, refers to that task's content, and is subsequently executed.
Prompted planning is logged separately and never counted as transfer.

Per-turn records are appended to
    data/metacognition/transfer_{student}.jsonl
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

from .interventions import INTERVENTION_BANK

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
META_DIR = os.path.join(REPO_ROOT, "data", "metacognition")

# Self-initiation is considered "internalised" once the student self-initiates
# metacognition on more than this fraction of turns.
TRANSFER_THRESHOLD = 0.40

# Session-phase boundaries (1-indexed session ordinals).
EARLY_SESSIONS = (1, 5)     # baseline
MID_SESSIONS = (6, 15)      # active intervention
LATE_SESSIONS = (16, None)  # transfer

METACOGNITIVE_TYPES = ["planning", "monitoring", "reflection", "none"]


# ------------------------------------------------------------------
# Marker banks — the LANGUAGE a student uses when they self-monitor
# ------------------------------------------------------------------
# Each entry is (compiled_regex, weight). Distinctive, unambiguous phrases carry
# more weight than generic ones. These describe the *student's own* voice, not
# ARIA's questions.

def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_SELF_INITIATED_MARKERS: dict[str, list[tuple[re.Pattern, float]]] = {
    # Setting up an approach before diving in. Kept broad enough to catch real
    # ADHD think-aloud planning ("first I need to…", "let me think…") but
    # excluding generic solving ("I need to find/use") that every state shows.
    "planning": [
        (_rx(r"\blet me think\b"), 2.5),
        (_rx(r"\bfirst,?\s+i(?:'ll| will)?\s+(?:need|have|want|should|gotta|had|gonna|am\s+gonna)\b"), 3.0),
        (_rx(r"\bfirst,?\s+i\s+(?:need|gotta|have|should|want|had|gonna)\b"), 3.0),
        (_rx(r"\b(?:okay|ok|alright|so),?\s+(?:so\s+)?(?:first|my (?:plan|approach|strategy))\b"), 2.0),
        (_rx(r"\bmy (?:plan|strategy|approach)\b"), 3.0),
        (_rx(r"\bbefore i (?:start|begin|dive|jump|do|even)\b"), 3.0),
        (_rx(r"\bi need to (?:figure|think about|remember|make sure|work out|plan|understand|start)\b"), 2.5),
        (_rx(r"\blet me (?:just )?(?:take|have|grab) a (?:sec|second|moment|minute)\b"), 2.5),
        (_rx(r"\blet me (?:plan|map|break|start by|begin by|set|figure)\b"), 2.5),
        (_rx(r"\bi'?m going to (?:start|first|begin)\b"), 2.0),
        (_rx(r"\bstep (?:one|1)\b"), 1.5),
        (_rx(r"\bwhat i need to do (?:is|first|here)\b"), 2.5),
        (_rx(r"\bi (?:should|gotta) (?:plan|start by|figure|think)\b"), 2.5),
    ],
    # Checking their own work. Kept SPECIFIC on purpose — generic doubt ("no
    # wait", "or is it") appears in every state, so we only count explicit,
    # deliberate self-checking to avoid firing on ordinary confusion.
    "monitoring": [
        (_rx(r"\bwait,? let me (?:check|re-?read|re-?look|double.?check|verify|make sure)\b"), 3.0),
        (_rx(r"\blet me (?:double.?check|check my (?:work|answer)|re-?read|verify|make sure)\b"), 3.0),
        (_rx(r"\bdoes (?:this|that) (?:even )?make sense\b"), 3.0),
        (_rx(r"\bi (?:think|might have|may have) made (?:a|an) (?:error|mistake)\b"), 3.0),
        (_rx(r"\bam i (?:doing this right|on the right track|sure about|missing)\b"), 2.5),
        (_rx(r"\b(?:did|do) i (?:do|get) (?:that|this) right\b"), 2.5),
        (_rx(r"\bthat (?:doesn'?t|does not) (?:seem|look) right\b"), 2.5),
        (_rx(r"\blet me make sure\b"), 2.0),
        (_rx(r"\bi (?:should|need to) (?:check|double.?check|re-?read)\b"), 2.5),
    ],
    # Making sense of understanding — the spontaneous "OH, it clicks, now I see
    # WHY" moment, plus after-the-fact reasoning about what worked / went wrong.
    # Deliberately excludes bare "that makes sense" / "I get it now", which
    # fluent (non-reflective) solving also produces.
    "reflection": [
        (_rx(r"\bohh+\b"), 3.0),                     # a drawn-out "ohhh" — the click
        (_rx(r"\bit click(?:ed|s)\b"), 3.0),
        (_rx(r"\boh+!"), 2.5),                       # "OH!"
        (_rx(r"\boh+[,.]?\s+(?:i (?:get|see|got)|now i|it'?s|that'?s|of course|wait,? (?:it'?s|that'?s))"), 2.0),
        (_rx(r"\bof course\b"), 2.0),
        (_rx(r"\bthe reason (?:this|that|it) works\b"), 3.0),
        (_rx(r"\bi got (?:this|that|it) wrong because\b"), 3.0),
        (_rx(r"\bnext time i(?:'ll| will)\b"), 3.0),
        (_rx(r"\bi see the pattern\b"), 3.0),
        (_rx(r"\bi (?:finally )?realize[ds]?\b"), 2.5),
        (_rx(r"\bnow i (?:get|see|understand) why\b"), 2.5),
    ],
}


# ------------------------------------------------------------------
# ARIA-prompt detection — was the *previous* message a metacog question?
# ------------------------------------------------------------------
# If ARIA's previous message asked a metacognitive question, then any
# metacognition in the student's reply is prompted, not self-initiated.

# Every canned intervention (flattened, normalised) counts as a metacog prompt.
_INTERVENTION_PHRASES = {
    re.sub(r"\s+", " ", txt).strip().lower()
    for bank in INTERVENTION_BANK.values()
    for txt in bank
}

# Generic shapes ARIA uses to prompt reflection (covers escalations + LLM
# rewordings that aren't verbatim bank entries).
_METACOG_PROMPT_PATTERNS = [
    _rx(r"\bwhat('?s| is) your (plan|strategy|approach)\b"),
    _rx(r"\bwhat do you (already )?know\b"),
    _rx(r"\bwhat would you (try|guess|do first|tell)\b"),
    _rx(r"\bwalk me through your reasoning\b"),
    _rx(r"\bwhy (that|this|does that) (step )?works?\b"),
    _rx(r"\bexplain (it|why|that|your)\b"),
    _rx(r"\bwhat did the problem (actually )?ask\b"),
    _rx(r"\bwhat do you predict\b"),
    _rx(r"\bwhat('?s| is) the (very )?first (word|step|thing)\b"),
    _rx(r"\bwhat('?s| is) your plan before\b"),
    _rx(r"\btell me what you (do|don'?t) understand\b"),
    _rx(r"\bwhere else could you use\b"),
    _rx(r"\bwhat was the (moment|one part)\b"),
    _rx(r"\bwhat('?s| is) (one|the one) (word|part|thing) you\b"),
    _rx(r"\bread (the|it|this) (first sentence|question|problem) (out loud|again)\b"),
    _rx(r"\bwould (this|the same) (approach|thing) work\b"),
    _rx(r"\bhow confident are you\b"),
]


def is_metacognitive_prompt(text: str) -> bool:
    """True if `text` (ARIA's previous message) is a metacognitive question."""
    if not text:
        return False
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if not norm:
        return False
    # Verbatim (or near-verbatim) bank interventions.
    if norm in _INTERVENTION_PHRASES:
        return True
    for phrase in _INTERVENTION_PHRASES:
        if len(phrase) > 12 and phrase in norm:
            return True
    # Generic reflective-question shapes.
    return any(p.search(text) for p in _METACOG_PROMPT_PATTERNS)


# ------------------------------------------------------------------
# Core (stateless) detection
# ------------------------------------------------------------------

def classify_metacognition(text: str) -> tuple[str, str]:
    """Classify the metacognitive TYPE of a student utterance.

    Returns (type, evidence) where type is one of
    planning / monitoring / reflection / none.
    """
    text = (text or "").strip()
    if not text:
        return "none", ""

    scores: dict[str, float] = {t: 0.0 for t in ("planning", "monitoring", "reflection")}
    hits: dict[str, list[str]] = {t: [] for t in scores}
    for mtype, markers in _SELF_INITIATED_MARKERS.items():
        for pat, weight in markers:
            m = pat.search(text)
            if m:
                scores[mtype] += weight
                hits[mtype].append(m.group(0).strip())

    top = max(scores, key=lambda t: scores[t])
    if scores[top] <= 0:
        return "none", ""

    # Tie-break toward the more specific behaviours when weights are equal.
    priority = {"reflection": 3, "monitoring": 2, "planning": 1}
    best_score = scores[top]
    contenders = [t for t, s in scores.items() if abs(s - best_score) < 1e-9]
    top = max(contenders, key=lambda t: priority[t])

    evidence = "; ".join(hits[top][:3])
    return top, evidence


def detect_self_initiation(student_input: str, aria_previous_prompt: str = "") -> dict[str, Any]:
    """Decide whether metacognition in `student_input` was self-initiated.

    `aria_previous_prompt` is ARIA's message immediately before this turn. If it
    was a metacognitive question, metacognition here is prompted, not
    self-initiated. Purely stateless — no persistence.
    """
    mtype, evidence = classify_metacognition(student_input)
    prompted = is_metacognitive_prompt(aria_previous_prompt)
    has_meta = mtype != "none"
    self_initiated = has_meta and not prompted

    if has_meta and prompted:
        evidence = (evidence + " (after ARIA prompt → prompted)").strip("; ")
    elif not has_meta:
        evidence = evidence or "no self-initiated metacognitive language"

    return {
        "self_initiated_metacognition": self_initiated,
        "metacognitive_type": mtype,
        "prompted_by_aria": prompted,
        "evidence": evidence,
    }


# ------------------------------------------------------------------
# Per-turn record + persistence
# ------------------------------------------------------------------

def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()) or "default"


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class TransferTurn:
    turn: int
    session: str
    student_input: str
    aria_previous_prompt: str
    self_initiated_metacognition: bool
    metacognitive_type: str
    prompted_by_aria: bool
    evidence: str
    student_profile: str = ""
    subject: str = ""
    task_id: str = ""
    moves_detected: list[str] | None = None
    no_aria_prompt_previous_turn: bool = False
    new_or_different_task: bool = False
    task_content_referenced: bool = False
    strategy_executed: bool = False
    transfer_candidate: bool = False
    transfer_confirmed: bool = False
    prompted_planning: bool = False
    confirms_candidate_turn: Optional[int] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["student_input"] = d["student_input"][:500]
        d["aria_previous_prompt"] = d["aria_previous_prompt"][:500]
        return d


class TransferDetector:
    """Detect and track self-initiated (vs prompted) metacognition over time."""

    TRANSFER_THRESHOLD = TRANSFER_THRESHOLD

    def __init__(self, student_name: str = "default"):
        self.student_name = student_name
        self.path = os.path.join(META_DIR, f"transfer_{_slug(student_name)}.jsonl")
        self._last_task_id = ""
        self._pending_candidate: Optional[dict] = None

    # -- detection + logging ----------------------------------------

    def detect(
        self,
        student_input: str,
        aria_previous_prompt: str = "",
        *,
        turn: int = 0,
        session: str = "",
        student_profile: str = "",
        subject: str = "",
        task_id: str = "",
        moves_detected: Optional[list[str]] = None,
        task_content_referenced: bool = False,
        strategy_executed: bool = False,
        persist: bool = True,
    ) -> dict:
        """Run strict, prospective transfer measurement for one turn.

        A qualifying plan becomes a candidate on the task's first relevant
        turn. It is confirmed only by a later execution turn on that same task.
        """
        core = detect_self_initiation(student_input, aria_previous_prompt)
        moves = list(moves_detected or [])
        no_previous_prompt = not bool((aria_previous_prompt or "").strip())
        different_task = bool(
            task_id and self._last_task_id and task_id != self._last_task_id
        )
        plan_or_monitoring = bool({"PLAN", "MONITORING"} & set(moves))
        prompted_planning = plan_or_monitoring and not no_previous_prompt

        confirms_turn = None
        transfer_confirmed = False
        if self._pending_candidate is not None:
            pending = self._pending_candidate
            if (
                pending["task_id"] == task_id
                and turn != pending["turn"]
                and strategy_executed
                and task_content_referenced
            ):
                transfer_confirmed = True
                confirms_turn = pending["turn"]
                self._pending_candidate = None
            elif task_id and pending["task_id"] != task_id:
                self._pending_candidate = None

        transfer_candidate = all((
            no_previous_prompt,
            different_task,
            plan_or_monitoring,
            task_content_referenced,
        ))
        if transfer_candidate:
            self._pending_candidate = {"task_id": task_id, "turn": turn}

        rec = TransferTurn(
            turn=turn,
            session=session,
            student_input=student_input or "",
            aria_previous_prompt=aria_previous_prompt or "",
            self_initiated_metacognition=core["self_initiated_metacognition"],
            metacognitive_type=core["metacognitive_type"],
            prompted_by_aria=core["prompted_by_aria"],
            evidence=core["evidence"],
            student_profile=student_profile,
            subject=subject,
            task_id=task_id,
            moves_detected=moves,
            no_aria_prompt_previous_turn=no_previous_prompt,
            new_or_different_task=different_task,
            task_content_referenced=bool(task_content_referenced),
            strategy_executed=bool(strategy_executed),
            transfer_candidate=transfer_candidate,
            transfer_confirmed=transfer_confirmed,
            prompted_planning=prompted_planning,
            confirms_candidate_turn=confirms_turn,
            timestamp=_now_iso(),
        ).to_dict()
        if task_id:
            self._last_task_id = task_id
        if persist:
            self._append(rec)
        return rec

    def _append(self, record: dict) -> None:
        os.makedirs(META_DIR, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        out: list[dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # -- ordering ----------------------------------------------------

    @staticmethod
    def _session_order(records: list[dict]) -> list[str]:
        """Distinct sessions in first-appearance (chronological) order."""
        order: list[str] = []
        seen: set[str] = set()
        for r in records:
            s = r.get("session") or ""
            if s not in seen:
                seen.add(s)
                order.append(s)
        return order

    # -- metrics -----------------------------------------------------

    def self_initiation_rate(self, records: Optional[list[dict]] = None) -> Optional[float]:
        """Self-Initiation Rate = self_initiated_turns / total_turns."""
        records = self.load() if records is None else records
        if not records:
            return None
        si = sum(1 for r in records if r.get("self_initiated_metacognition"))
        return round(si / len(records), 3)

    def task_grounded_transfer_rate(
        self, records: Optional[list[dict]] = None
    ) -> Optional[float]:
        """Confirmed transfers divided by eligible cross-task candidates."""
        records = self.load() if records is None else records
        candidates = sum(1 for row in records if row.get("transfer_candidate"))
        if not candidates:
            return None
        confirmed = sum(1 for row in records if row.get("transfer_confirmed"))
        return round(confirmed / candidates, 3)

    def rate_by_session(self, records: Optional[list[dict]] = None) -> list[dict]:
        """Per-session self-initiation rate, in chronological order.

        Also reports how often ARIA prompted metacognition that session, so a
        rising self-initiation rate can be read against falling prompting.
        """
        records = self.load() if records is None else records
        order = self._session_order(records)
        by_session: dict[str, list[dict]] = {s: [] for s in order}
        for r in records:
            by_session[r.get("session") or ""].append(r)

        out = []
        for i, s in enumerate(order, 1):
            rows = by_session[s]
            n = len(rows)
            si = sum(1 for r in rows if r.get("self_initiated_metacognition"))
            prompted = sum(1 for r in rows if r.get("prompted_by_aria"))
            out.append({
                "session": s,
                "session_ordinal": i,
                "total_turns": n,
                "self_initiated_turns": si,
                "self_initiation_rate": round(si / n, 3) if n else 0.0,
                "aria_prompt_rate": round(prompted / n, 3) if n else 0.0,
                "transferred": (si / n) > self.TRANSFER_THRESHOLD if n else False,
            })
        return out

    def transfer_series(self, records: Optional[list[dict]] = None) -> list[dict]:
        """[{session_ordinal, value}] of self-initiation rate — for a line chart."""
        return [
            {"session_ordinal": s["session_ordinal"], "value": s["self_initiation_rate"]}
            for s in self.rate_by_session(records)
        ]

    def rate_by_phase(self, records: Optional[list[dict]] = None) -> dict:
        """Baseline (sessions 1-5) vs active (6-15) vs transfer (16+) rates."""
        per = self.rate_by_session(records)

        def _agg(lo: int, hi: Optional[int]) -> Optional[float]:
            rows = [s for s in per if s["session_ordinal"] >= lo
                    and (hi is None or s["session_ordinal"] <= hi)]
            turns = sum(s["total_turns"] for s in rows)
            si = sum(s["self_initiated_turns"] for s in rows)
            return round(si / turns, 3) if turns else None

        return {
            "early_1_5": _agg(*EARLY_SESSIONS),
            "mid_6_15": _agg(*MID_SESSIONS),
            "late_16_plus": _agg(*LATE_SESSIONS),
        }

    def type_transfer_order(self, records: Optional[list[dict]] = None) -> dict:
        """First session ordinal at which each metacognitive type self-initiated.

        Planning typically transfers before reflection; this surfaces the order.
        """
        records = self.load() if records is None else records
        order = self._session_order(records)
        ordinal = {s: i for i, s in enumerate(order, 1)}
        first: dict[str, Optional[int]] = {"planning": None, "monitoring": None, "reflection": None}
        for r in records:
            if not r.get("self_initiated_metacognition"):
                continue
            t = r.get("metacognitive_type")
            if t in first and first[t] is None:
                first[t] = ordinal.get(r.get("session") or "", None)
        ranked = sorted(
            (t for t in first if first[t] is not None),
            key=lambda t: first[t],
        )
        return {"first_self_initiated_session": first, "transfer_order": ranked}

    def by_profile(self, records: Optional[list[dict]] = None) -> dict:
        """Self-initiation rate per ADHD profile (which profile transfers fastest)."""
        return self._group_rate("student_profile", records)

    def by_subject(self, records: Optional[list[dict]] = None) -> dict:
        """Self-initiation rate per subject (which subject shows most transfer)."""
        return self._group_rate("subject", records)

    def _group_rate(self, key: str, records: Optional[list[dict]]) -> dict:
        records = self.load() if records is None else records
        groups: dict[str, list[dict]] = {}
        for r in records:
            g = r.get(key) or "unknown"
            groups.setdefault(g, []).append(r)
        out = {}
        for g, rows in groups.items():
            n = len(rows)
            si = sum(1 for r in rows if r.get("self_initiated_metacognition"))
            out[g] = {"self_initiation_rate": round(si / n, 3) if n else 0.0, "n": n}
        return dict(sorted(out.items(), key=lambda kv: kv[1]["self_initiation_rate"], reverse=True))

    def summary(self, records: Optional[list[dict]] = None) -> dict:
        """Everything the dashboard needs, in one call."""
        records = self.load() if records is None else records
        per = self.rate_by_session(records)
        phases = self.rate_by_phase(records)
        overall = self.self_initiation_rate(records)
        # Trend: late minus early self-initiation rate (positive => transfer).
        delta = None
        if phases["late_16_plus"] is not None and phases["early_1_5"] is not None:
            delta = round(phases["late_16_plus"] - phases["early_1_5"], 3)
        elif len(per) >= 2 and per[0]["total_turns"] and per[-1]["total_turns"]:
            delta = round(per[-1]["self_initiation_rate"] - per[0]["self_initiation_rate"], 3)
        return {
            "student": self.student_name,
            "total_turns": len(records),
            "total_sessions": len(per),
            "self_initiation_rate": overall,
            "task_grounded_transfer_rate": self.task_grounded_transfer_rate(records),
            "confirmed_transfer_events": sum(
                1 for row in records if row.get("transfer_confirmed")
            ),
            "prompted_planning_turns": sum(
                1 for row in records if row.get("prompted_planning")
            ),
            "transferred": (overall or 0) > self.TRANSFER_THRESHOLD,
            "transfer_threshold": self.TRANSFER_THRESHOLD,
            "phases": phases,
            "phase_delta": delta,
            "by_type": self.type_transfer_order(records),
            "by_profile": self.by_profile(records),
            "by_subject": self.by_subject(records),
            "per_session": per,
        }


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    det = TransferDetector("demo_transfer")
    examples = [
        ("Okay so my approach is to isolate x first, then substitute.", ""),
        ("Wait, let me check — does this even make sense? 3 times 4 is 12, not 14.", ""),
        ("Um, the answer is 12.", "What's your plan before going further?"),
        ("I got this wrong because I forgot to distribute. Next time I'll expand first.", ""),
        ("idk, 7?", "Just read the first sentence out loud."),
    ]
    for i, (inp, prev) in enumerate(examples, 1):
        r = detect_self_initiation(inp, prev)
        tag = "SELF-INITIATED" if r["self_initiated_metacognition"] else (
            "prompted" if r["prompted_by_aria"] and r["metacognitive_type"] != "none" else "—")
        print(f"{i}. [{r['metacognitive_type']:10s}] {tag:15s} :: {inp}")
