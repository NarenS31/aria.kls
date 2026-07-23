"""
Longitudinal metacognition tracking for ARIA.

MetacognitionTracker records, per study session:
  * the full cognitive-state sequence with timestamps
  * time (turns / seconds) spent in each state
  * every intervention given, with the state before and after
  * self-corrections, planning events, and insight events

Across sessions it computes longitudinal metrics:
  1. Planning Ratio        = planning_turns / total_turns
  2. Self-Correction Rate  = self_caught / total_errors
  3. Recovery Speed        = mean turns to exit CONFUSED / STUCK
  4. Flow Ratio            = flow_turns / total_turns
  5. Frustration Frequency = frustrated_sessions / total_sessions

State is persisted to data/metacognition/longitudinal_{name}.json and a
personalised weekly summary can be generated with llama3.2:3b.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import ollama
except ImportError:  # pragma: no cover - depends on local optional install
    ollama = None  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
META_DIR = os.path.join(REPO_ROOT, "data", "metacognition")

SUMMARY_MODEL = os.environ.get("ARIA_META_MODEL", "llama3.2:3b")

PLANNING_STATES = {"PLANNING"}
FLOW_STATES = {"FLOW", "INSIGHT"}
NEGATIVE_STATES = {"CONFUSED", "STUCK"}
FRUSTRATION_STATES = {"FRUSTRATED"}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()) or "default"


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class StateEvent:
    state: str
    timestamp: str
    text: str = ""
    confidence: float = 0.0
    turn: int = 0
    self_correction: bool = False
    planning_detected: bool = False
    insight_moment: bool = False
    gave_up: bool = False
    was_error: bool = False  # student made an error this turn
    # Transfer: did the student self-initiate metacognition this turn (unprompted)?
    self_initiated: bool = False
    metacognitive_type: str = "none"


@dataclass
class InterventionEvent:
    intervention: str
    state_before: str
    state_after: Optional[str]
    escalated: bool
    escalation_kind: Optional[str]
    timestamp: str
    turn: int


@dataclass
class Session:
    session_id: str
    started_at: str
    student_name: str
    state_events: list = field(default_factory=list)
    intervention_events: list = field(default_factory=list)
    ended_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "student_name": self.student_name,
            "state_events": [asdict(e) if isinstance(e, StateEvent) else e
                             for e in self.state_events],
            "intervention_events": [asdict(e) if isinstance(e, InterventionEvent) else e
                                    for e in self.intervention_events],
        }


class MetacognitionTracker:
    """Track cognitive states and interventions per session and over time."""

    def __init__(self, student_name: str = "default"):
        self.student_name = student_name
        self.path = os.path.join(META_DIR, f"longitudinal_{_slug(student_name)}.json")
        self.sessions: list[dict] = []
        self._load()
        self.current: Optional[Session] = None
        self._turn = 0

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.sessions = data.get("sessions", [])
            except (json.JSONDecodeError, OSError):
                self.sessions = []

    def _save(self) -> None:
        os.makedirs(META_DIR, exist_ok=True)
        payload = {
            "student_name": self.student_name,
            "sessions": self.sessions,
            "updated_at": _now_iso(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # -- session lifecycle ------------------------------------------

    def start_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or f"{_slug(self.student_name)}_{_now_iso()}"
        self.current = Session(
            session_id=sid,
            started_at=_now_iso(),
            student_name=self.student_name,
        )
        self._turn = 0
        return sid

    def _ensure_session(self) -> Session:
        if self.current is None:
            self.start_session()
        return self.current  # type: ignore[return-value]

    def end_session(self) -> dict:
        sess = self._ensure_session()
        sess.ended_at = _now_iso()
        record = sess.to_dict()
        record["metrics"] = self._session_metrics(record)
        self.sessions.append(record)
        self._save()
        self.current = None
        return record

    # -- recording ---------------------------------------------------

    def record_state(self, analysis: dict, text: str = "", was_error: bool = False) -> None:
        """Record one analysed think-aloud turn (dict from CognitiveStateAnalyzer)."""
        sess = self._ensure_session()
        self._turn += 1
        ev = StateEvent(
            state=analysis.get("state", "CONFUSED"),
            timestamp=_now_iso(),
            text=text[:500],
            confidence=float(analysis.get("confidence", 0.0)),
            turn=self._turn,
            self_correction=bool(analysis.get("self_correction", False)),
            planning_detected=bool(analysis.get("planning_detected", False)),
            insight_moment=bool(analysis.get("insight_moment", False)),
            gave_up=bool(analysis.get("gave_up", False)),
            was_error=was_error,
            self_initiated=bool(analysis.get("self_initiated_metacognition", False)),
            metacognitive_type=str(analysis.get("metacognitive_type", "none")),
        )
        sess.state_events.append(ev)
        self._save()

    def record_intervention(
        self,
        intervention: dict,
        state_before: str,
        state_after: Optional[str] = None,
    ) -> None:
        sess = self._ensure_session()
        ev = InterventionEvent(
            intervention=intervention.get("text", ""),
            state_before=state_before,
            state_after=state_after,
            escalated=bool(intervention.get("escalated", False)),
            escalation_kind=intervention.get("escalation_kind"),
            timestamp=_now_iso(),
            turn=self._turn,
        )
        sess.intervention_events.append(ev)
        self._save()

    def set_intervention_outcome(self, state_after: str) -> None:
        """Fill in state_after for the most recent intervention event."""
        sess = self._ensure_session()
        if sess.intervention_events:
            last = sess.intervention_events[-1]
            if isinstance(last, InterventionEvent):
                last.state_after = state_after
            else:
                last["state_after"] = state_after
            self._save()

    def consecutive_state(self, state: str) -> int:
        """How many consecutive most-recent turns have been `state`."""
        sess = self._ensure_session()
        count = 0
        for ev in reversed(sess.state_events):
            s = ev.state if isinstance(ev, StateEvent) else ev["state"]
            if s == state:
                count += 1
            else:
                break
        return count

    # -- per-session metrics ----------------------------------------

    @staticmethod
    def _events(record: dict) -> list[dict]:
        out = []
        for e in record.get("state_events", []):
            out.append(e if isinstance(e, dict) else asdict(e))
        return out

    def _session_metrics(self, record: dict) -> dict:
        events = self._events(record)
        total = len(events)
        if total == 0:
            return {"total_turns": 0}

        planning = sum(1 for e in events if e["state"] in PLANNING_STATES
                       or e.get("planning_detected"))
        flow = sum(1 for e in events if e["state"] in FLOW_STATES)
        errors = sum(1 for e in events if e.get("was_error"))
        self_caught = sum(1 for e in events if e.get("self_correction"))
        insights = sum(1 for e in events if e.get("insight_moment"))

        # State durations in turns and seconds.
        dur_turns: dict[str, int] = {}
        dur_secs: dict[str, float] = {}
        for i, e in enumerate(events):
            dur_turns[e["state"]] = dur_turns.get(e["state"], 0) + 1
            if i + 1 < len(events):
                t0 = _parse(e["timestamp"])
                t1 = _parse(events[i + 1]["timestamp"])
                if t0 and t1:
                    dur_secs[e["state"]] = dur_secs.get(e["state"], 0.0) + (t1 - t0).total_seconds()

        recovery = _recovery_speed(events)

        return {
            "total_turns": total,
            "planning_turns": planning,
            "flow_turns": flow,
            "errors": errors,
            "self_corrections": self_caught,
            "insight_events": insights,
            "planning_ratio": round(planning / total, 3),
            "flow_ratio": round(flow / total, 3),
            "self_correction_rate": round(self_caught / errors, 3) if errors else None,
            "recovery_speed_turns": recovery,
            "state_duration_turns": dur_turns,
            "state_duration_seconds": {k: round(v, 1) for k, v in dur_secs.items()},
            "had_frustration": any(e["state"] in FRUSTRATION_STATES for e in events),
        }

    # -- longitudinal metrics ---------------------------------------

    def longitudinal_metrics(self, include_current: bool = True) -> dict:
        records = list(self.sessions)
        if include_current and self.current and self.current.state_events:
            cur = self.current.to_dict()
            cur["metrics"] = self._session_metrics(cur)
            records = records + [cur]

        if not records:
            return {"total_sessions": 0}

        total_turns = 0
        planning_turns = 0
        flow_turns = 0
        total_errors = 0
        self_caught = 0
        recoveries: list[float] = []
        frustrated_sessions = 0

        for r in records:
            m = r.get("metrics") or self._session_metrics(r)
            total_turns += m.get("total_turns", 0)
            planning_turns += m.get("planning_turns", 0)
            flow_turns += m.get("flow_turns", 0)
            total_errors += m.get("errors", 0)
            self_caught += m.get("self_corrections", 0)
            if m.get("recovery_speed_turns") is not None:
                recoveries.append(m["recovery_speed_turns"])
            if m.get("had_frustration"):
                frustrated_sessions += 1

        n_sessions = len(records)
        return {
            "total_sessions": n_sessions,
            "total_turns": total_turns,
            "planning_ratio": round(planning_turns / total_turns, 3) if total_turns else 0.0,
            "self_correction_rate": round(self_caught / total_errors, 3) if total_errors else None,
            "recovery_speed_turns": round(sum(recoveries) / len(recoveries), 2) if recoveries else None,
            "flow_ratio": round(flow_turns / total_turns, 3) if total_turns else 0.0,
            "frustration_frequency": round(frustrated_sessions / n_sessions, 3),
        }

    def weekly_series(self, metric: str = "planning_ratio") -> list[dict]:
        """Return the given per-session metric over time (for charts)."""
        out = []
        for r in self.sessions:
            m = r.get("metrics") or self._session_metrics(r)
            out.append({
                "session_id": r.get("session_id"),
                "started_at": r.get("started_at"),
                "value": m.get(metric),
            })
        return out

    def state_distribution(self, last_n: Optional[int] = None) -> list[dict]:
        """Per-session state-count distribution (for stacked bar charts)."""
        records = self.sessions[-last_n:] if last_n else self.sessions
        out = []
        for r in records:
            m = r.get("metrics") or self._session_metrics(r)
            out.append({
                "session_id": r.get("session_id"),
                "started_at": r.get("started_at"),
                "distribution": m.get("state_duration_turns", {}),
            })
        return out

    # -- weekly summary ---------------------------------------------

    def weekly_summary(self, use_llm: bool = True) -> str:
        """Generate a personalised weekly summary comparing this week to last."""
        this_week, last_week = self._split_by_week()
        cur = self._aggregate(this_week)
        prev = self._aggregate(last_week)

        facts = self._summary_facts(cur, prev)
        if not use_llm or ollama is None:
            return "\n".join(facts)

        prompt = (
            "You are ARIA, a warm learning coach for a neurodivergent student. "
            "Write a short (3-4 sentence) encouraging weekly summary of the "
            "student's metacognition, using ONLY these facts. Be specific with "
            "the numbers, warm, and never condescending.\n\nFacts:\n"
            + "\n".join(f"- {f}" for f in facts)
        )
        try:
            resp = ollama.chat(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content":
                        "You write brief, warm, specific progress summaries. No lists, just prose."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.6, "num_predict": 200},
            )
            return resp.message.content.strip()
        except Exception:
            return "\n".join(facts)

    def _summary_facts(self, cur: dict, prev: dict) -> list[str]:
        facts: list[str] = []

        def pct(x):
            return f"{round(x * 100)}%" if x is not None else "n/a"

        cp, pp = cur.get("planning_ratio"), prev.get("planning_ratio")
        if cp is not None and pp is not None:
            facts.append(f"Your planning ratio went from {pct(pp)} to {pct(cp)} this week.")
        elif cp is not None:
            facts.append(f"Your planning ratio this week was {pct(cp)}.")

        cr, pr = cur.get("recovery_speed_turns"), prev.get("recovery_speed_turns")
        if cr is not None and pr not in (None, 0):
            faster = round((1 - cr / pr) * 100)
            if faster > 0:
                facts.append(f"You recovered from confusion {faster}% faster than last week.")
            else:
                facts.append(f"Recovery took a bit longer this week ({cr} turns on average).")
        elif cr is not None:
            facts.append(f"On average it took {cr} turns to get unstuck this week.")

        cf = cur.get("flow_ratio")
        if cf is not None:
            facts.append(f"You spent {pct(cf)} of your turns in a flow/insight state.")

        csc = cur.get("self_correction_rate")
        if csc is not None:
            facts.append(f"You caught {pct(csc)} of your own mistakes without being told.")

        if not facts:
            facts.append("Not enough data yet this week — keep thinking out loud!")
        return facts

    def _split_by_week(self) -> tuple[list[dict], list[dict]]:
        """Split sessions into (this 7 days, previous 7 days) by started_at."""
        this_week, last_week = [], []
        if not self.sessions:
            return this_week, last_week
        # Anchor on the most recent session's date to avoid Date.now() issues
        # when this runs in offline/replay contexts.
        latest = max((_parse(r.get("started_at", "")) for r in self.sessions
                      if _parse(r.get("started_at", ""))), default=None)
        if latest is None:
            return self.sessions, []
        for r in self.sessions:
            t = _parse(r.get("started_at", ""))
            if t is None:
                continue
            age = (latest - t).days
            if age < 7:
                this_week.append(r)
            elif age < 14:
                last_week.append(r)
        return this_week, last_week

    def _aggregate(self, records: list[dict]) -> dict:
        if not records:
            return {}
        total_turns = sum((r.get("metrics") or self._session_metrics(r)).get("total_turns", 0)
                          for r in records)
        planning = sum((r.get("metrics") or self._session_metrics(r)).get("planning_turns", 0)
                       for r in records)
        flow = sum((r.get("metrics") or self._session_metrics(r)).get("flow_turns", 0)
                   for r in records)
        errors = sum((r.get("metrics") or self._session_metrics(r)).get("errors", 0)
                     for r in records)
        self_caught = sum((r.get("metrics") or self._session_metrics(r)).get("self_corrections", 0)
                          for r in records)
        recoveries = [m for m in ((r.get("metrics") or self._session_metrics(r)).get("recovery_speed_turns")
                                  for r in records) if m is not None]
        return {
            "planning_ratio": planning / total_turns if total_turns else None,
            "flow_ratio": flow / total_turns if total_turns else None,
            "self_correction_rate": self_caught / errors if errors else None,
            "recovery_speed_turns": sum(recoveries) / len(recoveries) if recoveries else None,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _recovery_speed(events: list[dict]) -> Optional[float]:
    """Mean number of turns to exit a CONFUSED/STUCK run into a better state."""
    runs: list[int] = []
    i = 0
    n = len(events)
    from_rank = {"CONFUSED", "STUCK"}
    better = {"PLANNING", "FLOW", "INSIGHT", "RUSHING"}
    while i < n:
        if events[i]["state"] in from_rank:
            length = 1
            j = i + 1
            while j < n and events[j]["state"] in from_rank:
                length += 1
                j += 1
            # Recovered only if the run is followed by a better state.
            if j < n and events[j]["state"] in better:
                runs.append(length)
            i = j
        else:
            i += 1
    if not runs:
        return None
    return round(sum(runs) / len(runs), 2)


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    t = MetacognitionTracker("demo")
    t.start_session()
    seq = ["CONFUSED", "STUCK", "CONFUSED", "PLANNING", "FLOW", "INSIGHT"]
    for s in seq:
        t.record_state({"state": s, "confidence": 0.9,
                        "self_correction": s == "INSIGHT",
                        "insight_moment": s == "INSIGHT",
                        "planning_detected": s == "PLANNING"},
                       text=f"({s})", was_error=(s in ("CONFUSED", "STUCK")))
    rec = t.end_session()
    print("Session metrics:", json.dumps(rec["metrics"], indent=2))
    print("\nLongitudinal:", json.dumps(t.longitudinal_metrics(), indent=2))
    print("\nWeekly summary:\n", t.weekly_summary(use_llm=False))
