"""
Metacognitive calibration tracking for ARIA.

The core question: does the student know what they know? ADHD students are
notoriously miscalibrated — confidently wrong on weak topics, needlessly unsure
on strong ones. Good calibration (confidence that tracks actual accuracy) is a
metacognitive skill that ARIA aims to build.

Before every problem attempt ARIA elicits a confidence rating:

    "Before you try this — how confident are you? 1 (totally lost) to 5 (got this)"

After the attempt, correctness is resolved (from the student's self-report or by
checking their answer against the known correct answer). Each problem yields:

    {problem_id, topic, difficulty, confidence_before (1-5), correct (bool),
     cognitive_state_during, session, timestamp}

From these we compute:

  1. Calibration Error   = mean(|confidence_norm - accuracy|)   (lower is better)
  2. Overconfidence Rate = % problems with confidence > 3 but wrong
  3. Underconfidence Rate= % problems with confidence < 3 but right
  4. Calibration by topic
  5. Calibration by cognitive state (does FLOW track better calibration? does
     RUSHING track overconfidence?)

and, longitudinally, calibration error per session over time — the hypothesis
being that it falls as the student learns to read their own understanding.

Per-problem records are appended to
    data/metacognition/calibration_{student}.jsonl
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

CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5
CONFIDENCE_MID = 3  # the neutral point; > MID = confident, < MID = unsure

CONFIDENCE_PROMPT = (
    "Before you try this — how confident are you? "
    "1 (totally lost) to 5 (got this)"
)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()) or "default"


def _now_iso() -> str:
    return datetime.now().isoformat()


def normalise_confidence(confidence: int | float) -> float:
    """Map a 1-5 confidence rating onto the 0-1 accuracy scale.

    1 -> 0.0, 5 -> 1.0. This puts confidence and accuracy in the same range so
    their absolute difference is a meaningful calibration error.
    """
    c = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence)))
    return (c - CONFIDENCE_MIN) / (CONFIDENCE_MAX - CONFIDENCE_MIN)


def clamp_confidence(confidence: Any) -> int:
    """Coerce arbitrary input to an integer 1-5 confidence rating."""
    try:
        c = int(round(float(confidence)))
    except (TypeError, ValueError):
        c = CONFIDENCE_MID
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, c))


@dataclass
class CalibrationRecord:
    problem_id: str
    topic: str
    difficulty: str
    confidence_before: int
    correct: bool
    cognitive_state_during: str
    session: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class CalibrationTracker:
    """Track confidence-vs-accuracy calibration per student over time."""

    def __init__(self, student_name: str = "default"):
        self.student_name = student_name
        self.path = os.path.join(META_DIR, f"calibration_{_slug(student_name)}.jsonl")

    CONFIDENCE_PROMPT = CONFIDENCE_PROMPT

    # -- recording ---------------------------------------------------

    def record(
        self,
        problem_id: str,
        topic: str,
        difficulty: str,
        confidence_before: int,
        correct: bool,
        *,
        cognitive_state_during: str = "",
        session: str = "",
        persist: bool = True,
    ) -> dict:
        """Record one problem's confidence + outcome."""
        rec = CalibrationRecord(
            problem_id=problem_id or "",
            topic=topic or "unknown",
            difficulty=difficulty or "unknown",
            confidence_before=clamp_confidence(confidence_before),
            correct=bool(correct),
            cognitive_state_during=cognitive_state_during or "",
            session=session or "",
            timestamp=_now_iso(),
        ).to_dict()
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

    # -- core metrics (pure functions over a record list) ------------

    @staticmethod
    def calibration_error(records: list[dict]) -> Optional[float]:
        """Mean(|confidence_norm - accuracy|) over records. Lower = better."""
        if not records:
            return None
        total = 0.0
        for r in records:
            conf = normalise_confidence(r.get("confidence_before", CONFIDENCE_MID))
            acc = 1.0 if r.get("correct") else 0.0
            total += abs(conf - acc)
        return round(total / len(records), 4)

    @staticmethod
    def overconfidence_rate(records: list[dict]) -> Optional[float]:
        """Fraction of problems with confidence > 3 that were wrong."""
        if not records:
            return None
        n = sum(1 for r in records if r.get("confidence_before", 0) > CONFIDENCE_MID
                and not r.get("correct"))
        return round(n / len(records), 3)

    @staticmethod
    def underconfidence_rate(records: list[dict]) -> Optional[float]:
        """Fraction of problems with confidence < 3 that were right."""
        if not records:
            return None
        n = sum(1 for r in records if r.get("confidence_before", 0) < CONFIDENCE_MID
                and r.get("correct"))
        return round(n / len(records), 3)

    @staticmethod
    def accuracy(records: list[dict]) -> Optional[float]:
        if not records:
            return None
        return round(sum(1 for r in records if r.get("correct")) / len(records), 3)

    @staticmethod
    def mean_confidence(records: list[dict]) -> Optional[float]:
        if not records:
            return None
        return round(sum(r.get("confidence_before", CONFIDENCE_MID) for r in records) / len(records), 2)

    def _block(self, records: list[dict]) -> dict:
        """The standard metric block for any slice of records."""
        return {
            "n": len(records),
            "calibration_error": self.calibration_error(records),
            "overconfidence_rate": self.overconfidence_rate(records),
            "underconfidence_rate": self.underconfidence_rate(records),
            "accuracy": self.accuracy(records),
            "mean_confidence": self.mean_confidence(records),
        }

    # -- grouped views -----------------------------------------------

    def by_topic(self, records: Optional[list[dict]] = None) -> dict:
        records = self.load() if records is None else records
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r.get("topic", "unknown"), []).append(r)
        out = {t: self._block(rows) for t, rows in groups.items()}
        # Worst-calibrated topics first (ADHD students often miscalibrated on weak topics).
        return dict(sorted(out.items(),
                           key=lambda kv: (kv[1]["calibration_error"] is None,
                                           -(kv[1]["calibration_error"] or 0))))

    def by_difficulty(self, records: Optional[list[dict]] = None) -> dict:
        records = self.load() if records is None else records
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r.get("difficulty", "unknown"), []).append(r)
        return {d: self._block(rows) for d, rows in groups.items()}

    def by_cognitive_state(self, records: Optional[list[dict]] = None) -> dict:
        """Calibration split by the cognitive state the student was in.

        Tests whether FLOW correlates with better calibration and RUSHING with
        overconfidence.
        """
        records = self.load() if records is None else records
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r.get("cognitive_state_during") or "UNKNOWN", []).append(r)
        return {s: self._block(rows) for s, rows in groups.items()}

    # -- longitudinal ------------------------------------------------

    @staticmethod
    def _session_order(records: list[dict]) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()
        for r in records:
            s = r.get("session") or ""
            if s not in seen:
                seen.add(s)
                order.append(s)
        return order

    def per_session(self, records: Optional[list[dict]] = None) -> list[dict]:
        """Calibration metrics per session, in chronological order."""
        records = self.load() if records is None else records
        order = self._session_order(records)
        by_session: dict[str, list[dict]] = {s: [] for s in order}
        for r in records:
            by_session[r.get("session") or ""].append(r)
        out = []
        for i, s in enumerate(order, 1):
            block = self._block(by_session[s])
            block["session"] = s
            block["session_ordinal"] = i
            out.append(block)
        return out

    def error_series(self, records: Optional[list[dict]] = None) -> list[dict]:
        """[{session_ordinal, calibration_error, overconfidence_rate,
        underconfidence_rate}] for the calibration chart."""
        return [
            {
                "session_ordinal": s["session_ordinal"],
                "calibration_error": s["calibration_error"],
                "overconfidence_rate": s["overconfidence_rate"],
                "underconfidence_rate": s["underconfidence_rate"],
            }
            for s in self.per_session(records)
        ]

    def summary(self, records: Optional[list[dict]] = None) -> dict:
        records = self.load() if records is None else records
        overall = self._block(records)
        per = self.per_session(records)
        # Trend: earlier vs later calibration error (falling => improving).
        delta = None
        errs = [s["calibration_error"] for s in per if s["calibration_error"] is not None]
        if len(errs) >= 2:
            delta = round(errs[-1] - errs[0], 4)
        overall.update({
            "student": self.student_name,
            "total_sessions": len(per),
            "calibration_error_delta": delta,
            "by_topic": self.by_topic(records),
            "by_difficulty": self.by_difficulty(records),
            "by_cognitive_state": self.by_cognitive_state(records),
            "per_session": per,
        })
        return overall


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    tr = CalibrationTracker("demo_calibration")
    # A miscalibrated student: confident on wrong answers, unsure on right ones.
    demo = [
        ("p1", "quadratics", "hard", 5, False, "RUSHING"),
        ("p2", "quadratics", "medium", 4, False, "RUSHING"),
        ("p3", "fractions", "easy", 2, True, "FLOW"),
        ("p4", "fractions", "easy", 5, True, "FLOW"),
        ("p5", "geometry", "medium", 1, True, "CONFUSED"),
    ]
    recs = [tr.record(*d[:5], cognitive_state_during=d[5], session="s1", persist=False)
            for d in demo]
    print("Calibration error :", tr.calibration_error(recs))
    print("Overconfidence    :", tr.overconfidence_rate(recs))
    print("Underconfidence   :", tr.underconfidence_rate(recs))
    print("By state RUSHING  :", tr.by_cognitive_state(recs)["RUSHING"])
    print("By state FLOW     :", tr.by_cognitive_state(recs)["FLOW"])
