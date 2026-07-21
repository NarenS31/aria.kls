"""
Intervention-timing optimisation for ARIA.

The core question: when a student is in a negative cognitive state (CONFUSED,
RUSHING, FRUSTRATED, STUCK), what is the optimal moment to intervene? Too early
and ARIA robs them of productive struggle; too late and they spiral. The right
moment differs by state and by ADHD subtype.

A *negative-state episode* is a run of consecutive turns in the same negative
state. For each episode we record, when ARIA intervenes:

    {episode_id, state, turns_in_state_before_intervention, intervention_text,
     turns_to_recovery, recovered, student_profile, subject, session}

Grouping episodes by `turns_in_state_before_intervention` (1, 2, 3, 4+) yields a
recovery rate and mean recovery speed per timing bucket, and hence the optimal
intervention point per state — overall, and per ADHD subtype.

After a student has enough sessions, the optimal timing is persisted to
    data/metacognition/timing_{student}.json
and read by agent/reasoning.py, which delays or accelerates its interventions
accordingly (adaptive timing). Per-episode events are appended to
    data/metacognition/timing_events_{student}.jsonl
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
META_DIR = os.path.join(REPO_ROOT, "data", "metacognition")

# States a student can be "stuck in" and that an intervention should rescue.
NEGATIVE_STATES = ["CONFUSED", "RUSHING", "FRUSTRATED", "STUCK"]

# Same rank table as interventions.py — an episode "recovers" when the state
# moves to a strictly higher rank than the negative state it started in.
STATE_RANK = {
    "STUCK": 0, "FRUSTRATED": 1, "CONFUSED": 2, "RUSHING": 3,
    "PLANNING": 4, "FLOW": 5, "INSIGHT": 6,
}

# Timing buckets: how many turns the student was in the state before ARIA acted.
TIMING_BUCKETS = [1, 2, 3, 4]      # 4 means "4+"
MAX_BUCKET = 4

# Theory-driven defaults, used before a student has enough data to adapt.
#   RUSHING / FRUSTRATED: intervene immediately (turn 1)
#   CONFUSED / STUCK:     let them try once or twice first (turn 2)
DEFAULT_OPTIMAL = {"CONFUSED": 2, "RUSHING": 1, "FRUSTRATED": 1, "STUCK": 2}

# ADHD-subtype shift applied to the default wait, in turns.
#   hyperactive: earlier (they spiral fast)   -> shift earlier
#   inattentive: later (give them space)      -> shift later
PROFILE_SHIFT = {"hyperactive": -1, "inattentive": +1, "combined": 0}

# Sessions of history required before ARIA trusts learned (adaptive) timing.
MIN_SESSIONS_FOR_ADAPTIVE = 10
# A timing bucket needs at least this many episodes to be trusted.
MIN_EPISODES_PER_BUCKET = 3


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()) or "default"


def _now_iso() -> str:
    return datetime.now().isoformat()


def bucket_of(turns_before: int) -> int:
    """Map a raw turns-before-intervention count to its bucket (1,2,3,4+)."""
    return min(MAX_BUCKET, max(1, int(turns_before)))


def adhd_subtype(profile: Any) -> str:
    """Normalise a profile / diagnosis into hyperactive|inattentive|combined|unknown."""
    if isinstance(profile, dict):
        raw = " ".join(str(v) for v in (
            profile.get("adhd_type"), profile.get("diagnosis"), profile.get("subtype")) if v)
    elif isinstance(profile, (list, tuple)):
        raw = " ".join(str(v) for v in profile)
    else:
        raw = str(profile or "")
    raw = raw.lower()
    has_hyper = "hyperactive" in raw or "impulsive" in raw
    has_inatt = "inattentive" in raw
    if "combined" in raw or (has_hyper and has_inatt):
        return "combined"
    if has_hyper:
        return "hyperactive"
    if has_inatt:
        return "inattentive"
    return "unknown"


@dataclass
class TimingEpisode:
    episode_id: str
    state: str
    turns_in_state_before_intervention: int
    intervention_text: str
    turns_to_recovery: int
    recovered: bool
    student_profile: str
    subject: str
    session: str
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["intervention_text"] = d["intervention_text"][:300]
        return d


class InterventionTimer:
    """Record negative-state episodes, learn optimal timing, and gate interventions."""

    MIN_SESSIONS_FOR_ADAPTIVE = MIN_SESSIONS_FOR_ADAPTIVE
    NEGATIVE_STATES = NEGATIVE_STATES

    def __init__(self, student_name: str = "default"):
        self.student_name = student_name
        self.events_path = os.path.join(META_DIR, f"timing_events_{_slug(student_name)}.jsonl")
        self.config_path = os.path.join(META_DIR, f"timing_{_slug(student_name)}.json")
        self.config: dict = {}
        self._load_config()

    # -- persistence -------------------------------------------------

    def _load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    self.config = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self.config = {}

    def record_episode(
        self,
        episode_id: str,
        state: str,
        turns_in_state_before_intervention: int,
        intervention_text: str,
        turns_to_recovery: int,
        recovered: bool,
        *,
        student_profile: str = "",
        subject: str = "",
        session: str = "",
        persist: bool = True,
    ) -> dict:
        """Record one intervention episode."""
        rec = TimingEpisode(
            episode_id=episode_id or "",
            state=(state or "").upper(),
            turns_in_state_before_intervention=int(turns_in_state_before_intervention),
            intervention_text=intervention_text or "",
            turns_to_recovery=int(turns_to_recovery),
            recovered=bool(recovered),
            student_profile=student_profile or "",
            subject=subject or "",
            session=session or "",
            timestamp=_now_iso(),
        ).to_dict()
        if persist:
            os.makedirs(META_DIR, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def load_events(self) -> list[dict]:
        out: list[dict] = []
        if not os.path.exists(self.events_path):
            return out
        with open(self.events_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # -- analysis ----------------------------------------------------

    def analyze_by_turn(
        self, state: Optional[str] = None, events: Optional[list[dict]] = None
    ) -> dict:
        """Recovery rate + speed grouped by turns-before-intervention bucket.

        If `state` is given, restrict to that state; otherwise pool all states.
        Returns {bucket: {n, recovered, recovery_rate, mean_turns_to_recovery}}.
        """
        events = self.load_events() if events is None else events
        if state:
            events = [e for e in events if e.get("state") == state.upper()]
        buckets: dict[int, list[dict]] = {b: [] for b in TIMING_BUCKETS}
        for e in events:
            b = bucket_of(e.get("turns_in_state_before_intervention", 1))
            buckets[b].append(e)

        out: dict[int, dict] = {}
        for b, rows in buckets.items():
            n = len(rows)
            recovered = [e for e in rows if e.get("recovered")]
            n_rec = len(recovered)
            speeds = [e["turns_to_recovery"] for e in recovered
                      if e.get("turns_to_recovery") is not None]
            out[b] = {
                "n": n,
                "recovered": n_rec,
                "recovery_rate": round(n_rec / n, 3) if n else None,
                "mean_turns_to_recovery": round(sum(speeds) / len(speeds), 2) if speeds else None,
            }
        return out

    def optimal_timing(
        self, state: str, events: Optional[list[dict]] = None
    ) -> Optional[int]:
        """Best turns-before-intervention bucket for `state` from recorded data.

        "Best" = highest recovery rate among buckets with enough evidence,
        breaking ties toward faster recovery, then toward earlier intervention.
        Returns None when there isn't enough data to decide.
        """
        by_turn = self.analyze_by_turn(state, events)
        candidates = [
            (b, m) for b, m in by_turn.items()
            if m["n"] >= MIN_EPISODES_PER_BUCKET and m["recovery_rate"] is not None
        ]
        if not candidates:
            return None
        # Maximise recovery rate; tie-break: faster recovery, then earlier turn.
        candidates.sort(key=lambda bm: (
            -bm[1]["recovery_rate"],
            bm[1]["mean_turns_to_recovery"] if bm[1]["mean_turns_to_recovery"] is not None else 99,
            bm[0],
        ))
        return candidates[0][0]

    def optimal_by_state(self, events: Optional[list[dict]] = None) -> dict:
        events = self.load_events() if events is None else events
        out = {}
        for s in NEGATIVE_STATES:
            learned = self.optimal_timing(s, events)
            out[s] = learned if learned is not None else DEFAULT_OPTIMAL[s]
        return out

    def optimal_by_profile(self, events: Optional[list[dict]] = None) -> dict:
        """Optimal timing per ADHD subtype (does timing differ by subtype?)."""
        events = self.load_events() if events is None else events
        out: dict[str, dict] = {}
        by_profile: dict[str, list[dict]] = {}
        for e in events:
            by_profile.setdefault(adhd_subtype(e.get("student_profile")), []).append(e)
        for profile, rows in by_profile.items():
            per_state = {}
            for s in NEGATIVE_STATES:
                learned = self.optimal_timing(s, rows)
                per_state[s] = learned if learned is not None else DEFAULT_OPTIMAL[s]
            out[profile] = per_state
        return out

    def heatmap(self, events: Optional[list[dict]] = None) -> dict:
        """state x bucket -> recovery_rate grid, for the timing heatmap chart."""
        events = self.load_events() if events is None else events
        grid: dict[str, dict[int, Optional[float]]] = {}
        counts: dict[str, dict[int, int]] = {}
        for s in NEGATIVE_STATES:
            by_turn = self.analyze_by_turn(s, events)
            grid[s] = {b: by_turn[b]["recovery_rate"] for b in TIMING_BUCKETS}
            counts[s] = {b: by_turn[b]["n"] for b in TIMING_BUCKETS}
        return {"states": NEGATIVE_STATES, "buckets": TIMING_BUCKETS,
                "recovery_rate": grid, "n": counts}

    def n_sessions(self, events: Optional[list[dict]] = None) -> int:
        events = self.load_events() if events is None else events
        return len({e.get("session") for e in events if e.get("session")})

    # -- adaptive gate (used live by reasoning.py) -------------------

    def recommended_wait(self, state: str, profile: Any = "") -> int:
        """How many turns the student should be in `state` before ARIA intervenes.

        Uses learned per-student timing once there is enough history; otherwise
        theory defaults, shifted by ADHD subtype.
        """
        state = (state or "").upper()
        if state not in NEGATIVE_STATES:
            return 1  # positive/neutral states: never gated

        subtype = adhd_subtype(profile)

        # Prefer learned config if the student has enough sessions.
        cfg = self.config or {}
        if cfg.get("adaptive"):
            # Profile-specific learned timing takes precedence, then per-state.
            prof_cfg = (cfg.get("optimal_by_profile") or {}).get(subtype, {})
            if state in prof_cfg:
                return max(1, min(MAX_BUCKET, int(prof_cfg[state])))
            state_cfg = cfg.get("optimal_by_state") or {}
            if state in state_cfg:
                return max(1, min(MAX_BUCKET, int(state_cfg[state])))

        base = DEFAULT_OPTIMAL[state]
        shift = PROFILE_SHIFT.get(subtype, 0)
        return max(1, min(MAX_BUCKET, base + shift))

    def should_intervene(self, state: str, turns_in_state: int, profile: Any = "") -> bool:
        """True if now is the right moment to intervene given how long they've struggled."""
        state = (state or "").upper()
        if state not in NEGATIVE_STATES:
            return True  # positive/neutral: intervention is not time-gated
        return int(turns_in_state) >= self.recommended_wait(state, profile)

    # -- config computation + persistence ----------------------------

    def save_config(self, n_sessions: Optional[int] = None) -> dict:
        """Compute optimal timing from history and persist it to timing_{student}.json.

        `n_sessions` (e.g. the tracker's session count) decides whether the
        learned timing is trusted; falls back to distinct sessions in the events.
        """
        events = self.load_events()
        sessions = n_sessions if n_sessions is not None else self.n_sessions(events)
        adaptive = sessions >= MIN_SESSIONS_FOR_ADAPTIVE and len(events) > 0
        config = {
            "student": self.student_name,
            "n_sessions": sessions,
            "n_episodes": len(events),
            "adaptive": adaptive,
            "optimal_by_state": self.optimal_by_state(events),
            "optimal_by_profile": self.optimal_by_profile(events),
            "analysis": {s: self.analyze_by_turn(s, events) for s in NEGATIVE_STATES},
            "defaults": DEFAULT_OPTIMAL,
            "updated_at": _now_iso(),
        }
        os.makedirs(META_DIR, exist_ok=True)
        tmp = self.config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.config_path)
        self.config = config
        return config

    def summary(self, events: Optional[list[dict]] = None) -> dict:
        events = self.load_events() if events is None else events
        return {
            "student": self.student_name,
            "n_episodes": len(events),
            "n_sessions": self.n_sessions(events),
            "adaptive": bool(self.config.get("adaptive")),
            "optimal_by_state": self.optimal_by_state(events),
            "optimal_by_profile": self.optimal_by_profile(events),
            "by_turn": {s: self.analyze_by_turn(s, events) for s in NEGATIVE_STATES},
            "heatmap": self.heatmap(events),
        }

    # -- derive episodes from tracker session records ----------------

    @staticmethod
    def derive_episodes_from_sessions(sessions: list[dict]) -> list[dict]:
        """Reconstruct timing episodes from MetacognitionTracker session records.

        Walks each session's state_events, finds runs of a negative state, marks
        the turn ARIA intervened (from intervention_events on that state), and
        measures turns to the first strictly-better state. Lets the dashboard and
        eval read timing straight from longitudinal tracker data.
        """
        episodes: list[dict] = []
        for sess in sessions or []:
            sid = sess.get("session_id", "")
            events = sess.get("state_events", []) or []
            # Turns at which an intervention was given.
            interventions = sess.get("intervention_events", []) or []
            iv_by_turn = {iv.get("turn"): iv for iv in interventions}

            n = len(events)
            i = 0
            while i < n:
                st = events[i].get("state")
                if st not in NEGATIVE_STATES:
                    i += 1
                    continue
                # Extent of the consecutive same-state run.
                j = i
                while j < n and events[j].get("state") == st:
                    j += 1
                run = events[i:j]
                # First turn in the run where ARIA intervened.
                iv_pos = None
                for k, ev in enumerate(run):
                    if ev.get("turn") in iv_by_turn:
                        iv_pos = k
                        break
                if iv_pos is not None:
                    turns_before = iv_pos + 1  # 1-indexed within the run
                    iv = iv_by_turn[run[iv_pos].get("turn")]
                    # Recovery: first strictly-better state after the run ends.
                    recovered = j < n and STATE_RANK.get(events[j].get("state"), 0) > STATE_RANK.get(st, 0)
                    turns_to_recovery = (len(run) - iv_pos) if recovered else (len(run) - iv_pos)
                    episodes.append(TimingEpisode(
                        episode_id=f"{sid}:{i}",
                        state=st,
                        turns_in_state_before_intervention=turns_before,
                        intervention_text=iv.get("intervention", ""),
                        turns_to_recovery=turns_to_recovery,
                        recovered=recovered,
                        student_profile=sess.get("student_name", ""),
                        subject="",
                        session=sid,
                    ).to_dict())
                i = j
        return episodes


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    tm = InterventionTimer("demo_timing")
    # Synthetic episodes: for CONFUSED, waiting to turn 2 recovers best.
    demo = [
        ("CONFUSED", 1, False, 3), ("CONFUSED", 1, False, 2), ("CONFUSED", 1, True, 3),
        ("CONFUSED", 2, True, 1), ("CONFUSED", 2, True, 1), ("CONFUSED", 2, True, 2),
        ("RUSHING", 1, True, 1), ("RUSHING", 1, True, 1), ("RUSHING", 1, True, 1),
        ("RUSHING", 3, False, 2), ("RUSHING", 3, False, 3), ("RUSHING", 3, True, 2),
    ]
    recs = [tm.record_episode(f"e{i}", s, t, "q?", ttr, rec, session="s", persist=False)
            for i, (s, t, rec, ttr) in enumerate(demo)]
    print("CONFUSED by turn:", json.dumps(tm.analyze_by_turn("CONFUSED", recs), indent=1))
    print("optimal CONFUSED:", tm.optimal_timing("CONFUSED", recs), "(expect 2)")
    print("optimal RUSHING :", tm.optimal_timing("RUSHING", recs), "(expect 1)")
    print("recommended wait CONFUSED / hyperactive:",
          tm.recommended_wait("CONFUSED", "hyperactive"))
