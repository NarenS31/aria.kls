"""
Synthetic ADHD student profile for controlled experiments.

Alex Chen — 16-year-old, Grade 11, ADHD-Combined.
Exported as both a Python dataclass and data/synthetic_profile.json.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class SyntheticProfile:
    # Identity
    name: str
    age: int
    grade: int
    diagnosis: str
    learning_style: str
    subjects: List[str]
    baseline_confidence: Dict[str, float]

    # Behavioral parameters (mathematical heart of the experiment)
    peak_focus_hours: List[int]
    attention_dropoff_minutes: int
    frustration_threshold: float       # fraction of failed attempts that triggers frustration
    frustration_triggers: Dict[str, List[str]]
    preferred_styles: Dict[str, str]
    misconceptions: Dict[str, List[str]]
    response_patterns: Dict[str, List[str]]

    # Meta
    support_people: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    study_hours: List[int] = field(default_factory=list)


ALEX_CHEN = SyntheticProfile(
    name="Alex Chen",
    age=16,
    grade=11,
    diagnosis="ADHD-Combined",
    learning_style="step_by_step",
    subjects=["algebra", "biology", "python programming"],
    baseline_confidence={
        "algebra": 0.35,
        "biology": 0.50,
        "python programming": 0.45,
    },

    # Behavioral parameters
    peak_focus_hours=[22, 23, 0],       # 10 PM – midnight
    attention_dropoff_minutes=20,
    frustration_threshold=0.40,

    frustration_triggers={
        "algebra": ["fractions", "negative numbers", "word problems"],
        "biology": ["meiosis", "protein synthesis"],
        "python programming": ["recursion", "scope"],
    },

    preferred_styles={
        "algebra": "analogy",
        "biology": "visual",
        "python programming": "step_by_step",
    },

    misconceptions={
        "algebra": [
            "distributes addition before multiplication (treats 2(x+3) as 2x+3 instead of 2x+6)",
            "drops negative signs when moving terms across the equals sign",
        ],
        "biology": [
            "confuses mitosis products (2 identical) with meiosis products (4 unique cells)",
            "thinks all cells divide at the same rate regardless of cell type",
        ],
        "python programming": [
            "off-by-one indexing in range() — uses range(1, n) when range(n) needed",
            "confuses local and global scope — modifies global variable inside function without global keyword",
        ],
    },

    response_patterns={
        "when_confused": ["wait what", "i dont get it", "can you repeat that", "huh"],
        "when_frustrated": ["idk", "this makes no sense", "forget it", "ugh", "why doesnt this work"],
        "when_successful": ["oh ok", "that makes sense", "wait so then", "ohhhh"],
    },

    support_people=["Mom", "tutor Sam"],
    goals=["pass algebra final", "understand cell division for bio test", "finish python project"],
    study_hours=[21, 22, 23, 0],
)


def generate_profile_json(path: Path = DATA_DIR / "synthetic_profile.json") -> Path:
    """Serialize the canonical profile to JSON and return the path."""
    data = asdict(ALEX_CHEN)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_profile_json(path: Path = DATA_DIR / "synthetic_profile.json") -> SyntheticProfile:
    with open(path) as f:
        data = json.load(f)
    return SyntheticProfile(**data)


def profile_as_aria_profile(profile: SyntheticProfile = ALEX_CHEN) -> dict:
    """Convert to the dict format ARIA's ARIAAgent expects."""
    return {
        "name": profile.name,
        "learning_style": profile.learning_style,
        "subjects": profile.subjects,
        "goals": profile.goals,
        "support_people": profile.support_people,
        "study_hours": profile.peak_focus_hours,
        "diagnosis": profile.diagnosis,
        "age": profile.age,
        "grade": profile.grade,
        "baseline_confidence": profile.baseline_confidence,
        "preferred_styles": profile.preferred_styles,
        "frustration_triggers": profile.frustration_triggers,
    }


if __name__ == "__main__":
    path = generate_profile_json()
    print(f"Profile saved to {path}")
    print(json.dumps(asdict(ALEX_CHEN), indent=2))
