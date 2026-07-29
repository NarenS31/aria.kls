"""
Student profile management for ARIA.
Handles profile CRUD, SRS state, learning graphs, emotion logs, and checkpoints.
"""

import json
import os
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import networkx as nx

STUDENTS_DATA_DIR = Path("data/students")


class StudentProfile:
    def __init__(
        self,
        name: str,
        grade: str,
        diagnosis: str,
        learning_style: str,
        goals: list,
        subjects: list,
        attention_span_minutes: int = 20,
        teacher_flags: list = None,
        flagged_for_attention: bool = False,
        created_at: str = None,
        last_active: str = None,
    ):
        self.name = name
        self.grade = grade
        self.diagnosis = diagnosis
        self.learning_style = learning_style
        self.goals = goals if goals is not None else []
        self.subjects = subjects if subjects is not None else []
        self.attention_span_minutes = attention_span_minutes
        self.teacher_flags = teacher_flags if teacher_flags is not None else []
        self.flagged_for_attention = flagged_for_attention
        self.created_at = created_at or datetime.now().isoformat()
        self.last_active = last_active

    @classmethod
    def from_dict(cls, d: dict) -> "StudentProfile":
        return cls(
            name=d.get("name", ""),
            grade=d.get("grade", ""),
            diagnosis=d.get("diagnosis", ""),
            learning_style=d.get("learning_style", "mixed"),
            goals=d.get("goals", []),
            subjects=d.get("subjects", []),
            attention_span_minutes=d.get("attention_span_minutes", 20),
            teacher_flags=d.get("teacher_flags", []),
            flagged_for_attention=d.get("flagged_for_attention", False),
            created_at=d.get("created_at"),
            last_active=d.get("last_active"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grade": self.grade,
            "diagnosis": self.diagnosis,
            "learning_style": self.learning_style,
            "goals": self.goals,
            "subjects": self.subjects,
            "attention_span_minutes": self.attention_span_minutes,
            "teacher_flags": self.teacher_flags,
            "flagged_for_attention": self.flagged_for_attention,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


class StudentProfileManager:
    def __init__(self, data_dir: str = "data/students"):
        # Resolve relative to project root
        path = Path(data_dir)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / data_dir
        self.data_dir = path
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, name: str) -> Path:
        return self.data_dir / name / "profile.json"

    def list_students(self) -> list:
        profiles = []
        for entry in self.data_dir.iterdir():
            if entry.is_dir():
                profile_file = entry / "profile.json"
                if profile_file.exists():
                    try:
                        with open(profile_file) as f:
                            profiles.append(StudentProfile.from_dict(json.load(f)))
                    except Exception:
                        pass
        return profiles

    def get_student(self, name: str) -> Optional[StudentProfile]:
        profile_file = self._profile_path(name)
        if not profile_file.exists():
            return None
        try:
            with open(profile_file) as f:
                return StudentProfile.from_dict(json.load(f))
        except Exception:
            return None

    def save_student(self, profile: StudentProfile):
        student_dir = self.data_dir / profile.name
        student_dir.mkdir(parents=True, exist_ok=True)
        (student_dir / "checkpoints").mkdir(exist_ok=True)
        (student_dir / "emotion_logs").mkdir(exist_ok=True)
        with open(student_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def create_student(
        self,
        name: str,
        grade: str,
        diagnosis: str,
        learning_style: str,
        goals: list,
        subjects: list,
        attention_span_minutes: int = 20,
    ) -> StudentProfile:
        profile = StudentProfile(
            name=name,
            grade=grade,
            diagnosis=diagnosis,
            learning_style=learning_style,
            goals=goals,
            subjects=subjects,
            attention_span_minutes=attention_span_minutes,
        )
        self.save_student(profile)
        return profile

    def update_last_active(self, name: str):
        profile = self.get_student(name)
        if profile:
            profile.last_active = datetime.now().isoformat()
            self.save_student(profile)

    def flag_student(self, name: str, note: str):
        profile = self.get_student(name)
        if profile:
            profile.teacher_flags.append(note)
            profile.flagged_for_attention = True
            self.save_student(profile)

    def unflag_student(self, name: str):
        profile = self.get_student(name)
        if profile:
            profile.teacher_flags = []
            profile.flagged_for_attention = False
            self.save_student(profile)

    def get_student_data_path(self, name: str) -> Path:
        return self.data_dir / name

    # ------------------------------------------------------------------
    # SRS state
    # ------------------------------------------------------------------

    def get_srs_state(self, name: str) -> dict:
        srs_path = self.data_dir / name / "srs_state.json"
        if not srs_path.exists():
            return {}
        try:
            with open(srs_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def save_srs_state(self, name: str, state: dict):
        student_dir = self.data_dir / name
        student_dir.mkdir(parents=True, exist_ok=True)
        with open(student_dir / "srs_state.json", "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Learning graph
    # ------------------------------------------------------------------

    def get_learning_graph(self, name: str):
        graph_path = self.data_dir / name / "learning_graph.pkl"
        if not graph_path.exists():
            return nx.DiGraph(), {}
        try:
            with open(graph_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                graph = data.get("graph", nx.DiGraph())
                confidence = data.get("confidence", {})
            elif isinstance(data, nx.DiGraph):
                graph = data
                confidence = {
                    n: graph.nodes[n].get("confidence", 0.0)
                    for n in graph.nodes
                }
            else:
                graph = nx.DiGraph()
                confidence = {}
            return graph, confidence
        except Exception:
            return nx.DiGraph(), {}

    def save_learning_graph(self, name: str, graph):
        student_dir = self.data_dir / name
        student_dir.mkdir(parents=True, exist_ok=True)
        # Accept either a raw DiGraph or a dict with "graph" key
        if isinstance(graph, dict):
            data = graph
        else:
            confidence = {
                n: graph.nodes[n].get("confidence", 0.0)
                for n in graph.nodes
            }
            data = {"graph": graph, "confidence": confidence}
        with open(student_dir / "learning_graph.pkl", "wb") as f:
            pickle.dump(data, f)

    def get_confidence_summary(self, name: str) -> dict:
        graph, confidence = self.get_learning_graph(name)
        # Prefer node attribute data over the stored confidence dict
        result = {}
        for node in graph.nodes:
            result[node] = graph.nodes[node].get("confidence", confidence.get(node, 0.0))
        # Also include confidence keys not in graph
        for topic, conf in confidence.items():
            if topic not in result:
                result[topic] = conf
        return result

    # ------------------------------------------------------------------
    # Emotion logs
    # ------------------------------------------------------------------

    def save_emotion_log(self, name: str, session_date: str, entries: list):
        emotion_dir = self.data_dir / name / "emotion_logs"
        emotion_dir.mkdir(parents=True, exist_ok=True)
        log_path = emotion_dir / f"{session_date}.jsonl"
        with open(log_path, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def get_emotion_trend(self, name: str, days: int = 7) -> list:
        emotion_dir = self.data_dir / name / "emotion_logs"
        if not emotion_dir.exists():
            return []
        cutoff = (datetime.now() - timedelta(days=days)).date()
        entries = []
        for path in sorted(emotion_dir.glob("*.jsonl")):
            date_str = path.stem
            try:
                log_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if log_date < cutoff:
                continue
            try:
                with open(path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            ts = rec.get("timestamp", date_str)
                            emotion = rec.get("emotion_state", rec.get("state", "FLOW"))
                            entries.append((ts, emotion))
                        except Exception:
                            pass
            except Exception:
                pass
        return entries

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str, topic: str, checkpoint_data: dict):
        checkpoint_dir = self.data_dir / name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        today_str = date.today().isoformat()
        safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
        filename = f"{safe_topic}_{today_str}.json"
        checkpoint_data["saved_at"] = datetime.now().isoformat()
        checkpoint_data["topic"] = topic
        with open(checkpoint_dir / filename, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def get_latest_checkpoint(self, name: str) -> Optional[dict]:
        checkpoint_dir = self.data_dir / name / "checkpoints"
        if not checkpoint_dir.exists():
            return None
        files = sorted(checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        try:
            with open(files[0]) as f:
                return json.load(f)
        except Exception:
            return None
