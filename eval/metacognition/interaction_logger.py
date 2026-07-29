"""
Structured local logging for ARIA interactions and interventions.

This module stores event data in SQLite, separate from the ChromaDB memory
store. The goal is later AIED/JITAI analysis: every tutor interaction,
intervention trigger, and proximal outcome should be queryable without relying
on vector memory.

The sentiment/tone fields here are heuristic proxy signals only. They are not
diagnostic labels and should not be described as clinical assessment.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "logs" / "aria_interactions.sqlite3"

OUTCOMES = {"responded", "re_engaged", "ignored", "ended_session", "unknown"}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _json_dumps(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any) -> Any:
    if value in (None, ""):
        return {}
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@dataclass
class ToneProxy:
    """A lightweight tone proxy, not a diagnostic or clinical signal."""

    label: str
    score: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(float(self.score), 3),
            "evidence": list(self.evidence),
        }


@dataclass
class InteractionEvent:
    session_id: str
    profile_id: str
    topic: str
    student_message: str
    aria_response: str = ""
    timestamp: str = field(default_factory=_now_iso)
    response_latency_ms: Optional[int] = None
    time_since_previous_ms: Optional[int] = None
    pause_summary: dict[str, Any] = field(default_factory=dict)
    tone_proxy_label: Optional[str] = None
    tone_proxy_score: Optional[float] = None
    tone_proxy_evidence: list[str] = field(default_factory=list)
    cognitive_state: Optional[str] = None
    analyzer_confidence: Optional[float] = None
    # Multimodal behavioral layer. `cognitive_state` is the fused (final) state;
    # `text_state`/`text_confidence` record what the text classifier alone said,
    # and `fusion_method` how the two were combined. All optional -> a turn with
    # no keystroke telemetry logs an empty behavioral_features and text_only.
    behavioral_features: dict[str, Any] = field(default_factory=dict)
    text_state: Optional[str] = None
    text_confidence: Optional[float] = None
    fusion_method: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionEvent:
    session_id: str
    profile_id: str
    topic: str
    trigger_condition: str
    intervention_type: str
    intervention_text: str
    timestamp: str = field(default_factory=_now_iso)
    rule_id: str = ""
    rationale: str = ""
    citation_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionOutcome:
    intervention_id: int
    outcome: str = "unknown"
    timestamp: str = field(default_factory=_now_iso)
    outcome_latency_ms: Optional[int] = None
    next_cognitive_state: Optional[str] = None
    reengaged: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionLogger:
    """SQLite-backed local interaction log for ARIA."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS interaction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    student_message TEXT NOT NULL,
                    aria_response TEXT NOT NULL DEFAULT '',
                    response_latency_ms INTEGER,
                    time_since_previous_ms INTEGER,
                    pause_summary_json TEXT NOT NULL DEFAULT '{}',
                    tone_proxy_label TEXT,
                    tone_proxy_score REAL,
                    tone_proxy_evidence_json TEXT NOT NULL DEFAULT '[]',
                    cognitive_state TEXT,
                    analyzer_confidence REAL,
                    behavioral_features_json TEXT NOT NULL DEFAULT '{}',
                    text_state TEXT,
                    text_confidence REAL,
                    fusion_method TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_interaction_session_time
                    ON interaction_events(session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_interaction_profile_topic
                    ON interaction_events(profile_id, topic);

                CREATE TABLE IF NOT EXISTS intervention_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    trigger_condition TEXT NOT NULL,
                    intervention_type TEXT NOT NULL,
                    intervention_text TEXT NOT NULL,
                    rule_id TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    citation_keys_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_intervention_session_time
                    ON intervention_events(session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_intervention_profile_topic
                    ON intervention_events(profile_id, topic);

                CREATE TABLE IF NOT EXISTS intervention_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intervention_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    outcome_latency_ms INTEGER,
                    next_cognitive_state TEXT,
                    reengaged INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(intervention_id) REFERENCES intervention_events(id)
                );

                CREATE INDEX IF NOT EXISTS idx_outcome_intervention
                    ON intervention_outcomes(intervention_id);

                CREATE TABLE IF NOT EXISTS typing_pause_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    pause_ms INTEGER NOT NULL,
                    position INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_typing_pause_session
                    ON typing_pause_events(session_id, timestamp);
                """
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additively bring an existing interaction_events table up to date.

        CREATE TABLE IF NOT EXISTS never adds columns to a pre-existing table, so
        databases created before the behavioral layer need the new columns added
        via ALTER TABLE. This is idempotent and preserves all existing rows.
        """
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(interaction_events)")
        }
        additions = [
            ("behavioral_features_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("text_state", "TEXT"),
            ("text_confidence", "REAL"),
            ("fusion_method", "TEXT"),
        ]
        for name, decl in additions:
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE interaction_events ADD COLUMN {name} {decl}"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_interaction(self, event: Optional[InteractionEvent] = None, **kwargs: Any) -> int:
        if event is None:
            event = InteractionEvent(**kwargs)
        if event.tone_proxy_label is None or event.tone_proxy_score is None:
            tone = tone_proxy(event.student_message)
            event.tone_proxy_label = tone.label
            event.tone_proxy_score = tone.score
            event.tone_proxy_evidence = tone.evidence

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO interaction_events (
                    timestamp, session_id, profile_id, topic, student_message,
                    aria_response, response_latency_ms, time_since_previous_ms,
                    pause_summary_json, tone_proxy_label, tone_proxy_score,
                    tone_proxy_evidence_json, cognitive_state, analyzer_confidence,
                    behavioral_features_json, text_state, text_confidence,
                    fusion_method, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    event.session_id,
                    event.profile_id,
                    event.topic,
                    event.student_message,
                    event.aria_response,
                    event.response_latency_ms,
                    event.time_since_previous_ms,
                    _json_dumps(event.pause_summary),
                    event.tone_proxy_label,
                    event.tone_proxy_score,
                    _json_dumps(event.tone_proxy_evidence),
                    event.cognitive_state,
                    event.analyzer_confidence,
                    _json_dumps(event.behavioral_features),
                    event.text_state,
                    event.text_confidence,
                    event.fusion_method,
                    _json_dumps(event.metadata),
                ),
            )
            return int(cur.lastrowid)

    def log_intervention(self, event: Optional[InterventionEvent] = None, **kwargs: Any) -> int:
        if event is None:
            event = InterventionEvent(**kwargs)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO intervention_events (
                    timestamp, session_id, profile_id, topic, trigger_condition,
                    intervention_type, intervention_text, rule_id, rationale,
                    citation_keys_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    event.session_id,
                    event.profile_id,
                    event.topic,
                    event.trigger_condition,
                    event.intervention_type,
                    event.intervention_text,
                    event.rule_id,
                    event.rationale,
                    _json_dumps(event.citation_keys),
                    _json_dumps(event.metadata),
                ),
            )
            return int(cur.lastrowid)

    def log_outcome(self, record: Optional[InterventionOutcome] = None, **kwargs: Any) -> int:
        if record is None:
            record = InterventionOutcome(**kwargs)
        if record.outcome not in OUTCOMES:
            raise ValueError(f"Unknown intervention outcome: {record.outcome}")

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO intervention_outcomes (
                    intervention_id, timestamp, outcome, outcome_latency_ms,
                    next_cognitive_state, reengaged, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.intervention_id,
                    record.timestamp,
                    record.outcome,
                    record.outcome_latency_ms,
                    record.next_cognitive_state,
                    None if record.reengaged is None else int(record.reengaged),
                    _json_dumps(record.metadata),
                ),
            )
            return int(cur.lastrowid)

    def log_typing_pause(
        self,
        session_id: str,
        profile_id: str,
        topic: str,
        pause_ms: int,
        position: Optional[int] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO typing_pause_events (
                    timestamp, session_id, profile_id, topic, pause_ms, position,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp or _now_iso(),
                    session_id,
                    profile_id,
                    topic,
                    int(pause_ms),
                    position,
                    _json_dumps(metadata or {}),
                ),
            )
            return int(cur.lastrowid)

    def get_recent_interventions(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM intervention_events
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()
        return [_decode_row(r) for r in rows]

    def get_recent_interaction_signals(
        self, session_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM interaction_events
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()
        return [_decode_row(r) for r in rows]

    def get_ignored_intervention_count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM intervention_events ie
                JOIN intervention_outcomes io ON io.intervention_id = ie.id
                WHERE ie.session_id = ? AND io.outcome = 'ignored'
                """,
                (session_id,),
            ).fetchone()
        return int(row["n"] if row else 0)

    def get_topic_intervention_stats(self, profile_id: str, topic: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ie.intervention_type,
                    COALESCE(io.outcome, 'unknown') AS outcome,
                    COUNT(*) AS n
                FROM intervention_events ie
                LEFT JOIN intervention_outcomes io ON io.intervention_id = ie.id
                WHERE ie.profile_id = ? AND ie.topic = ?
                GROUP BY ie.intervention_type, COALESCE(io.outcome, 'unknown')
                """,
                (profile_id, topic),
            ).fetchall()

        by_type: dict[str, dict[str, int]] = {}
        total = 0
        for row in rows:
            intervention_type = row["intervention_type"]
            outcome = row["outcome"]
            n = int(row["n"])
            by_type.setdefault(intervention_type, {})[outcome] = n
            total += n
        return {"profile_id": profile_id, "topic": topic, "total": total, "by_type": by_type}

    def tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [str(r["name"]) for r in rows]


def tone_proxy(text: str) -> ToneProxy:
    """Return a small keyword-based affect proxy for logging and rules."""

    lowered = (text or "").lower()
    evidence: list[str] = []
    score = 0.0

    negative_markers = {
        "ugh": -0.35,
        "hate": -0.55,
        "stupid": -0.45,
        "annoying": -0.35,
        "frustrating": -0.45,
        "give up": -0.65,
        "can't do": -0.45,
        "dont get": -0.3,
        "don't get": -0.3,
        "no idea": -0.4,
    }
    positive_markers = {
        "got it": 0.45,
        "makes sense": 0.4,
        "i see": 0.35,
        "ohhh": 0.35,
        "clicked": 0.45,
    }

    for marker, weight in negative_markers.items():
        if marker in lowered:
            score += weight
            evidence.append(marker)
    for marker, weight in positive_markers.items():
        if marker in lowered:
            score += weight
            evidence.append(marker)

    if re.search(r"!{2,}", text or ""):
        score -= 0.15
        evidence.append("repeated exclamation")
    if len((text or "").split()) <= 2 and lowered.strip() in {"idk", "what", "huh"}:
        score -= 0.25
        evidence.append("minimal stuck response")

    score = max(-1.0, min(1.0, score))
    if score <= -0.55:
        label = "strong_negative"
    elif score < -0.15:
        label = "negative"
    elif score >= 0.25:
        label = "positive"
    else:
        label = "neutral"
    return ToneProxy(label=label, score=score, evidence=evidence[:6])


def summarize_pauses(pauses_ms: list[int]) -> dict[str, Any]:
    if not pauses_ms:
        return {"count": 0, "max_ms": 0, "avg_ms": 0}
    values = [int(p) for p in pauses_ms]
    return {
        "count": len(values),
        "max_ms": max(values),
        "avg_ms": round(sum(values) / len(values)),
    }


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in list(data):
        if key.endswith("_json"):
            decoded_key = key[:-5]
            data[decoded_key] = _json_loads(data.pop(key))
    if "reengaged" in data and data["reengaged"] is not None:
        data["reengaged"] = bool(data["reengaged"])
    return data


def event_to_dict(event: Any) -> dict[str, Any]:
    if hasattr(event, "__dataclass_fields__"):
        return asdict(event)
    return dict(event)
