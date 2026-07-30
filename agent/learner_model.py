"""Persistent, uncertainty-aware learner model for ARIA.

This is deliberately smaller and more auditable than a neural knowledge
tracing model.  Each student-skill pair has a Beta posterior over demonstrated
mastery, and each student-topic-misconception tuple has a Beta posterior over
whether the misconception persists.  Updates use observable evidence; a
cognitive-state estimate alone never changes mastery.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional


def _posterior(bucket: dict) -> dict:
    alpha = float(bucket.get("alpha", 1.0))
    beta = float(bucket.get("beta", 1.0))
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total ** 2) * (total + 1))
    radius = 1.96 * math.sqrt(variance)
    return {
        "mean": round(mean, 4),
        "ci_95": [
            round(max(0.0, mean - radius), 4),
            round(min(1.0, mean + radius), 4),
        ],
        "observations": int(bucket.get("observations", 0)),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
    }


class LearnerModelStore:
    """JSON-backed mastery and misconception posteriors."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {
            "schema_version": 1,
            "students": {},
            "updated_at": None,
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict) and loaded.get("schema_version") == 1:
                self.data = loaded
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2) + "\n")
            temporary.replace(self.path)
        except OSError:
            pass

    def _student(self, student: str) -> dict:
        key = student.strip() or "student"
        return self.data["students"].setdefault(
            key,
            {"skills": {}, "misconceptions": {}},
        )

    def state(
        self,
        *,
        student: str,
        topic: str,
        skills: list[str],
        misconception: str = "",
    ) -> dict:
        record = self._student(student)
        skill_names = [item.strip() for item in skills if item.strip()] or [topic]
        skill_states = {}
        for skill in skill_names:
            bucket = record["skills"].setdefault(
                f"{topic}::{skill}",
                {"alpha": 1.0, "beta": 1.0, "observations": 0},
            )
            skill_states[skill] = _posterior(bucket)
        mastery = sum(item["mean"] for item in skill_states.values()) / len(skill_states)
        if mastery < 0.4:
            mastery_band = "developing"
        elif mastery < 0.75:
            mastery_band = "emerging"
        else:
            mastery_band = "secure"

        misconception_state = None
        if misconception:
            bucket = record["misconceptions"].setdefault(
                f"{topic}::{misconception}",
                {"alpha": 1.0, "beta": 1.0, "observations": 0},
            )
            misconception_state = _posterior(bucket)
        return {
            "mastery_mean": round(mastery, 4),
            "mastery_band": mastery_band,
            "skills": skill_states,
            "misconception_persistence": misconception_state,
        }

    def observe(
        self,
        *,
        student: str,
        topic: str,
        skills: list[str],
        misconception: str,
        correct: Optional[bool],
        grounded_progress: bool,
        self_correction: bool,
        misconception_persisted: Optional[bool],
        evidence_strength: float,
    ) -> dict:
        """Update only from observable performance evidence.

        Evidence strength is capped so one chat turn cannot produce a large
        mastery jump. Correctness has the highest weight, while grounded
        progress and self-correction provide weaker partial evidence.
        """
        record = self._student(student)
        strength = max(0.0, min(1.0, float(evidence_strength)))
        success_weight = 0.0
        failure_weight = 0.0
        if correct is True:
            success_weight += 1.0
        elif correct is False:
            failure_weight += 0.8
        if grounded_progress:
            success_weight += 0.25 * strength
        if self_correction:
            success_weight += 0.2 * strength
        if correct is None and not grounded_progress:
            # An uninformative turn should barely update the posterior.
            failure_weight += 0.05

        for skill in ([item for item in skills if item] or [topic]):
            bucket = record["skills"].setdefault(
                f"{topic}::{skill}",
                {"alpha": 1.0, "beta": 1.0, "observations": 0},
            )
            bucket["alpha"] += success_weight
            bucket["beta"] += failure_weight
            bucket["observations"] += 1

        if misconception:
            bucket = record["misconceptions"].setdefault(
                f"{topic}::{misconception}",
                {"alpha": 1.0, "beta": 1.0, "observations": 0},
            )
            if misconception_persisted is True:
                bucket["alpha"] += max(0.25, strength)
            elif misconception_persisted is False:
                bucket["beta"] += max(0.25, strength)
            bucket["observations"] += int(misconception_persisted is not None)

        self._save()
        return self.state(
            student=student,
            topic=topic,
            skills=skills,
            misconception=misconception,
        )
