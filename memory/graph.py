"""
NetworkX-based personal learning graph for ARIA.
Nodes = topics. Edges = topic relationships.
Node attributes track confidence, struggle patterns, time-of-day focus,
explanation style success rates, and study frequency.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import networkx as nx

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
GRAPH_PATH = DATA_DIR / "learning_graph.pkl"
PROFILE_PATH = DATA_DIR / "user_profile.json"


class LearningGraph:
    def __init__(self):
        self.graph: nx.DiGraph = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> nx.DiGraph:
        if GRAPH_PATH.exists():
            with open(GRAPH_PATH, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def save(self) -> None:
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)

    # ------------------------------------------------------------------
    # Topic nodes
    # ------------------------------------------------------------------

    def _default_node(self, topic: str) -> dict:
        return {
            "topic": topic,
            "confidence": 0.5,
            "study_count": 0,
            "struggle_count": 0,
            "study_hours": [],          # list of hours (0-23) when studied
            "explanation_styles": {},   # style -> success_count
            "last_studied": None,
            "goal_topic": False,
            "notes": "",
        }

    def ensure_topic(self, topic: str) -> None:
        if not self.graph.has_node(topic):
            self.graph.add_node(topic, **self._default_node(topic))

    def record_study(
        self,
        topic: str,
        struggled: bool,
        explanation_style: str,
        was_helpful: bool,
        hour: Optional[int] = None,
    ) -> None:
        self.ensure_topic(topic)
        node = self.graph.nodes[topic]
        node["study_count"] += 1
        node["last_studied"] = datetime.now().isoformat()

        if struggled:
            node["struggle_count"] += 1
            node["confidence"] = max(0.05, node["confidence"] - 0.05)
        else:
            node["confidence"] = min(1.0, node["confidence"] + 0.03)

        h = hour if hour is not None else datetime.now().hour
        node["study_hours"].append(h)
        if len(node["study_hours"]) > 100:
            node["study_hours"] = node["study_hours"][-100:]

        styles = node["explanation_styles"]
        styles.setdefault(explanation_style, 0)
        if was_helpful:
            styles[explanation_style] += 1

        self.save()

    def update_confidence(self, topic: str, delta: float) -> None:
        self.ensure_topic(topic)
        node = self.graph.nodes[topic]
        node["confidence"] = max(0.0, min(1.0, node["confidence"] + delta))
        self.save()

    def get_topic_info(self, topic: str) -> Optional[dict]:
        if not self.graph.has_node(topic):
            return None
        return dict(self.graph.nodes[topic])

    def best_explanation_style(self, topic: str) -> str:
        if not self.graph.has_node(topic):
            return "clear and simple"
        styles = self.graph.nodes[topic].get("explanation_styles", {})
        if not styles:
            return "clear and simple"
        return max(styles, key=styles.get)

    def peak_focus_hours(self) -> List[int]:
        """Returns hours of day with most successful study sessions across all topics."""
        hour_counts: dict[int, int] = {}
        for _, data in self.graph.nodes(data=True):
            for h in data.get("study_hours", []):
                hour_counts[h] = hour_counts.get(h, 0) + 1
        if not hour_counts:
            return []
        top = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]
        return sorted(top)

    def struggling_topics(self) -> List[dict]:
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get("struggle_count", 0) >= 2 or data.get("confidence", 1) < 0.4:
                results.append({"topic": node, "confidence": data.get("confidence", 0.5)})
        return sorted(results, key=lambda x: x["confidence"])

    def all_topics_summary(self) -> List[dict]:
        rows = []
        for node, data in self.graph.nodes(data=True):
            rows.append(
                {
                    "topic": node,
                    "confidence": round(data.get("confidence", 0.5), 2),
                    "study_count": data.get("study_count", 0),
                    "struggle_count": data.get("struggle_count", 0),
                    "last_studied": data.get("last_studied"),
                    "best_style": self.best_explanation_style(node),
                }
            )
        return sorted(rows, key=lambda x: x["study_count"], reverse=True)

    # ------------------------------------------------------------------
    # Topic relationships
    # ------------------------------------------------------------------

    def link_topics(self, from_topic: str, to_topic: str, relation: str = "related") -> None:
        self.ensure_topic(from_topic)
        self.ensure_topic(to_topic)
        self.graph.add_edge(from_topic, to_topic, relation=relation)
        self.save()

    def related_topics(self, topic: str) -> List[str]:
        if not self.graph.has_node(topic):
            return []
        neighbors = list(self.graph.successors(topic)) + list(self.graph.predecessors(topic))
        return list(set(neighbors))

    # ------------------------------------------------------------------
    # Seeding from intake
    # ------------------------------------------------------------------

    def seed_from_profile(self, profile: dict) -> None:
        subjects = profile.get("subjects", [])
        goals = profile.get("goals", [])
        learning_style = profile.get("learning_style", "mixed")

        for subject in subjects:
            self.ensure_topic(subject)
            self.graph.nodes[subject]["goal_topic"] = True
            self.graph.nodes[subject]["explanation_styles"][learning_style] = 1

        for i, s1 in enumerate(subjects):
            for s2 in subjects[i + 1:]:
                self.link_topics(s1, s2, "co-study")

        for goal in goals:
            self.ensure_topic(goal)
            self.graph.nodes[goal]["goal_topic"] = True
            self.graph.nodes[goal]["notes"] = "User stated goal"

        self.save()


# ------------------------------------------------------------------
# User profile persistence (flat JSON, not graph)
# ------------------------------------------------------------------

def load_profile() -> Optional[dict]:
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH) as f:
            return json.load(f)
    return None


def save_profile(profile: dict) -> None:
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
