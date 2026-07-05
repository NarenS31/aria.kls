"""
Five synthetic ADHD student profiles for ARIA evaluation.

Each profile:
  - Defined as a StudentProfile dataclass
  - Serialized to data/profiles/{name}.json
  - Seeded into a profile-namespaced ChromaDB collection
  - Seeded into a separate learning graph at data/graphs/{name}_graph.pkl
  - Pre-warmed with N=10 synthetic history sessions

Profile list:
  1. Alex Chen      — ADHD-Combined
  2. Jordan Rivera  — ADHD-Inattentive
  3. Sam Okonkwo    — ADHD-Combined + Dyslexia
  4. Maya Patel     — ADHD-Hyperactive
  5. Eli Washington — ADHD-Combined + Autism Spectrum
"""

import json
import pickle
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).parent.parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"
GRAPHS_DIR = DATA_DIR / "graphs"
CHROMA_DIR = DATA_DIR / "chroma"

for _d in (PROFILES_DIR, GRAPHS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Profile dataclass
# ------------------------------------------------------------------

@dataclass
class StudentProfile:
    # Identity
    name: str
    profile_id: str
    age: int
    grade: int
    diagnosis: str
    learning_style: str         # visual | step_by_step | analogy | kinesthetic
    subjects: List[str]
    baseline_confidence: Dict[str, float]

    # Behavioral parameters
    peak_focus_hours: List[int]
    attention_dropoff_minutes: int
    frustration_threshold: float
    frustration_triggers: Dict[str, List[str]]
    preferred_styles: Dict[str, str]
    misconceptions: Dict[str, List[str]]
    response_patterns: Dict[str, List[str]]

    # Research metadata
    theoretical_basis: str

    # Optional fields
    support_people: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)

    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")

    def as_aria_profile(self) -> dict:
        return {
            "name": self.name,
            "learning_style": self.learning_style,
            "subjects": self.subjects,
            "goals": self.goals,
            "support_people": self.support_people,
            "study_hours": self.peak_focus_hours,
            "diagnosis": self.diagnosis,
            "age": self.age,
            "grade": self.grade,
            "baseline_confidence": self.baseline_confidence,
            "preferred_styles": self.preferred_styles,
            "frustration_triggers": self.frustration_triggers,
        }


# ------------------------------------------------------------------
# Profile definitions
# ------------------------------------------------------------------

ALEX_CHEN = StudentProfile(
    name="Alex Chen",
    profile_id="alex_chen",
    age=16, grade=11,
    diagnosis="ADHD-Combined",
    learning_style="step_by_step",
    subjects=["algebra", "biology", "python programming"],
    baseline_confidence={"algebra": 0.35, "biology": 0.50, "python programming": 0.45},
    peak_focus_hours=[22, 23, 0],
    attention_dropoff_minutes=20,
    frustration_threshold=0.40,
    frustration_triggers={
        "algebra": ["fractions", "negative numbers"],
        "biology": ["meiosis", "protein synthesis"],
        "python programming": ["recursion", "scope"],
    },
    preferred_styles={"algebra": "analogy", "biology": "visual", "python programming": "step_by_step"},
    misconceptions={
        "algebra": [
            "distributes addition before multiplication (treats 2(x+3) as 2x+3)",
            "drops negative signs when moving terms across equals sign",
        ],
        "biology": [
            "confuses mitosis products (2 identical) with meiosis products (4 unique)",
            "thinks all cells divide at same rate regardless of type",
        ],
        "python programming": [
            "off-by-one in range() — uses range(1,n) when range(n) needed",
            "confuses local and global scope",
        ],
    },
    response_patterns={
        "confused": ["wait what", "i dont get it", "huh"],
        "frustrated": ["idk", "this makes no sense", "forget it", "ugh"],
        "successful": ["oh ok", "that makes sense", "ohhhh"],
    },
    theoretical_basis=(
        "ADHD-Combined with working memory deficits (Barkley 2015) "
        "and emotional dysregulation"
    ),
    goals=["pass algebra final", "understand cell division", "finish python project"],
)

JORDAN_RIVERA = StudentProfile(
    name="Jordan Rivera",
    profile_id="jordan_rivera",
    age=15, grade=10,
    diagnosis="ADHD-Inattentive",
    learning_style="visual",
    subjects=["geometry", "chemistry", "javascript"],
    baseline_confidence={"geometry": 0.60, "chemistry": 0.30, "javascript": 0.40},
    peak_focus_hours=[15, 16],
    attention_dropoff_minutes=15,
    frustration_threshold=0.60,
    frustration_triggers={
        "chemistry": ["stoichiometry", "balancing equations"],
        "geometry": ["proofs"],
        "javascript": ["callbacks", "async"],
    },
    preferred_styles={"geometry": "visual", "chemistry": "step_by_step", "javascript": "analogy"},
    misconceptions={
        "chemistry": [
            "confuses moles with molecules",
            "adds coefficients instead of balancing by trial",
        ],
        "geometry": [
            "assumes all quadrilaterals are parallelograms",
        ],
        "javascript": [
            "thinks var is block-scoped like let",
            "confuses == and === (coercion vs strict equality)",
        ],
    },
    response_patterns={
        "confused": ["wait so", "can you explain again", "i thought you said"],
        "frustrated": ["this is taking forever", "i give up", "whatever"],
        "successful": ["got it", "ok so basically", "wait thats actually easy"],
    },
    theoretical_basis=(
        "ADHD-Inattentive with sluggish cognitive tempo (Barkley 2015) "
        "and visual-spatial strength"
    ),
    goals=["pass geometry", "understand chemistry for SAT", "build a website"],
)

SAM_OKONKWO = StudentProfile(
    name="Sam Okonkwo",
    profile_id="sam_okonkwo",
    age=17, grade=12,
    diagnosis="ADHD-Combined + Dyslexia",
    learning_style="analogy",
    subjects=["statistics", "physics", "java"],
    baseline_confidence={"statistics": 0.40, "physics": 0.45, "java": 0.55},
    peak_focus_hours=[20, 21],
    attention_dropoff_minutes=25,
    frustration_threshold=0.30,
    frustration_triggers={
        "statistics": ["hypothesis testing", "p-values"],
        "physics": ["vector components", "torque"],
        "java": ["inheritance", "interfaces"],
    },
    preferred_styles={"statistics": "analogy", "physics": "analogy", "java": "step_by_step"},
    misconceptions={
        "statistics": [
            "confuses correlation with causation",
            "misinterprets p-value as probability that H0 is true",
        ],
        "physics": [
            "thinks heavier objects fall faster (pre-Galilean intuition)",
            "confuses speed (scalar) and velocity (vector)",
        ],
        "java": [
            "thinks overriding a method changes the parent class",
            "confuses abstract class and interface purposes",
        ],
    },
    response_patterns={
        "confused": ["i dont understand any of this", "can you use an example", "what does that even mean"],
        "frustrated": ["forget it", "this is stupid", "im not smart enough for this"],
        "successful": ["OH thats like", "wait so its basically", "that actually makes sense now"],
    },
    theoretical_basis=(
        "ADHD-Combined with comorbid dyslexia and anxiety, emotional dysregulation (Shaw 2014), "
        "benefits from analogical reasoning (Sweller 1988)"
    ),
    goals=["AP stats exam", "physics college prep", "java OOP project"],
    support_people=["tutor Sam", "older sister"],
)

MAYA_PATEL = StudentProfile(
    name="Maya Patel",
    profile_id="maya_patel",
    age=14, grade=9,
    diagnosis="ADHD-Hyperactive",
    learning_style="kinesthetic",
    subjects=["algebra", "earth science", "python basics"],
    baseline_confidence={"algebra": 0.55, "earth science": 0.65, "python basics": 0.50},
    peak_focus_hours=[19, 20],
    attention_dropoff_minutes=10,
    frustration_threshold=0.50,
    frustration_triggers={
        "algebra": ["multi-step equations", "inequalities"],
        "earth science": ["rock cycle sequences"],
        "python basics": ["functions", "parameters"],
    },
    preferred_styles={"algebra": "kinesthetic", "earth science": "visual", "python basics": "step_by_step"},
    misconceptions={
        "algebra": [
            "thinks you can add unlike terms (2x + 3 = 5x)",
            "confuses expression and equation",
        ],
        "earth science": [
            "thinks continents stopped moving",
            "confuses weather (short-term) and climate (long-term)",
        ],
        "python basics": [
            "thinks print() saves a value to a variable",
            "confuses parameter (definition) and argument (call)",
        ],
    },
    response_patterns={
        "confused": ["huh", "what", "i dont get it"],
        "frustrated": ["this is boring", "can we do something else", "idk idc"],
        "successful": ["oh cool", "wait thats actually fun", "can i try"],
    },
    theoretical_basis=(
        "ADHD-Hyperactive with shortest attention_dropoff (10 min), "
        "requires frequent reorientation (Barkley 2015), "
        "kinesthetic engagement (Sweller 1988)"
    ),
    goals=["pass algebra", "understand earth science", "make a simple game"],
)

ELI_WASHINGTON = StudentProfile(
    name="Eli Washington",
    profile_id="eli_washington",
    age=16, grade=11,
    diagnosis="ADHD-Combined + Autism Spectrum",
    learning_style="step_by_step",
    subjects=["calculus", "cs theory", "physics"],
    baseline_confidence={"calculus": 0.70, "cs theory": 0.75, "physics": 0.60},
    peak_focus_hours=[21, 22, 23],
    attention_dropoff_minutes=45,
    frustration_threshold=0.20,
    frustration_triggers={
        "calculus": ["epsilon-delta proofs", "implicit differentiation"],
        "cs theory": ["NP-completeness", "undecidability"],
        "physics": ["quantum uncertainty", "wave-particle duality"],
    },
    preferred_styles={"calculus": "step_by_step", "cs theory": "step_by_step", "physics": "analogy"},
    misconceptions={
        "calculus": [
            "thinks derivative is just slope of secant line (not limit of secant)",
            "confuses limit of f at point vs value of f at point",
        ],
        "cs theory": [
            "thinks P=NP is obviously false because of practical experience",
            "confuses decidable (yes/no answer) and tractable (efficient answer)",
        ],
        "physics": [
            "thinks particles have definite position at all times (pre-quantum)",
            "confuses rest mass and relativistic mass",
        ],
    },
    response_patterns={
        "confused": [
            "that contradicts what you said before",
            "thats not what the textbook says",
            "i need the exact definition",
        ],
        "frustrated": [
            "this is inconsistent",
            "you keep changing the explanation",
            "i need a precise answer",
        ],
        "successful": [
            "ok that is logically consistent",
            "i understand the formal definition now",
            "that follows from what you said",
        ],
    },
    theoretical_basis=(
        "ADHD-Combined with ASD, rigid thinking patterns (APA DSM-5), "
        "hyperfocus capability (attention_dropoff=45 min), "
        "requires logical consistency and precision"
    ),
    goals=["AP Calculus BC", "cs theory olympiad", "physics research project"],
    support_people=["school counselor", "math club advisor"],
)

ALL_PROFILES: List[StudentProfile] = [
    ALEX_CHEN, JORDAN_RIVERA, SAM_OKONKWO, MAYA_PATEL, ELI_WASHINGTON
]

PROFILE_MAP: Dict[str, StudentProfile] = {p.profile_id: p for p in ALL_PROFILES}


# ------------------------------------------------------------------
# Profile-namespaced stores
# ------------------------------------------------------------------

class ProfileVectorStore:
    """ChromaDB store namespaced to a specific student profile."""

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        coll_name = f"{profile_id}_conversations"
        # ChromaDB collection names must match [a-zA-Z0-9_-]
        coll_name = coll_name.replace(" ", "_")[:63]
        self.conversations = self.client.get_or_create_collection(
            name=coll_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def store_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        topics: List[str],
        frustration: bool,
        explanation_style: str,
        timestamp=None,
    ) -> None:
        from datetime import datetime
        ts = (timestamp or datetime.now()).isoformat()
        doc_id = str(uuid.uuid4())
        combined = f"User: {user_msg}\nARIA: {assistant_msg}"
        self.conversations.add(
            ids=[doc_id],
            documents=[combined],
            metadatas=[{
                "user_msg": user_msg[:500],
                "assistant_msg": assistant_msg[:500],
                "topics": json.dumps(topics),
                "frustration": str(frustration),
                "explanation_style": explanation_style,
                "timestamp": ts,
            }],
        )

    def retrieve_context(self, query: str, n: int = 5) -> List[dict]:
        count = self.conversations.count()
        if count == 0:
            return []
        results = self.conversations.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        return [
            {"text": doc, "meta": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def count(self) -> int:
        return self.conversations.count()


class ProfileLearningGraph:
    """NetworkX learning graph namespaced to a specific student profile."""

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.path = GRAPHS_DIR / f"{profile_id}_graph.pkl"
        self.graph = self._load()

    def _load(self):
        import networkx as nx
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self.graph, f)

    def ensure_topic(self, topic: str) -> None:
        if not self.graph.has_node(topic):
            self.graph.add_node(topic, confidence=0.5, study_count=0,
                                struggle_count=0, study_hours=[], explanation_styles={})

    def record_study(self, topic: str, struggled: bool, style: str, helpful: bool, hour: int = 0) -> None:
        self.ensure_topic(topic)
        n = self.graph.nodes[topic]
        n["study_count"] += 1
        if struggled:
            n["struggle_count"] += 1
            n["confidence"] = max(0.05, n["confidence"] - 0.05)
        else:
            n["confidence"] = min(1.0, n["confidence"] + 0.03)
        n["study_hours"].append(hour)
        styles = n["explanation_styles"]
        styles.setdefault(style, 0)
        if helpful:
            styles[style] += 1
        self.save()

    def seed_from_profile(self, profile: StudentProfile) -> None:
        for subj in profile.subjects:
            key = subj.replace(" ", "_").lower()
            self.ensure_topic(key)
            conf = profile.baseline_confidence.get(subj, 0.5)
            self.graph.nodes[key]["confidence"] = conf
            style = profile.preferred_styles.get(subj, profile.learning_style)
            self.graph.nodes[key]["explanation_styles"][style] = 1
        self.save()


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------

def save_profile_json(profile: StudentProfile) -> Path:
    path = PROFILES_DIR / f"{profile.slug()}.json"
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return path


def load_profile_json(profile_id: str) -> Optional[StudentProfile]:
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return StudentProfile(**data)


# ------------------------------------------------------------------
# Full seeding pipeline
# ------------------------------------------------------------------

def seed_all_profiles(n_history_sessions: int = 10, verbose: bool = True) -> None:
    """
    For each of the 5 profiles:
      1. Save JSON
      2. Seed learning graph with baseline confidence
      3. Generate and seed N synthetic history sessions
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.synthetic_history import seed_history

    for profile in ALL_PROFILES:
        if verbose:
            print(f"\n[profiles] Seeding {profile.name} ({profile.diagnosis})")

        save_profile_json(profile)

        vs = ProfileVectorStore(profile.profile_id)
        lg = ProfileLearningGraph(profile.profile_id)
        lg.seed_from_profile(profile)

        if n_history_sessions > 0:
            # Build a profile dict compatible with synthetic_history.seed_history
            profile_dict = {
                "name": profile.name,
                "age": profile.age,
                "diagnosis": profile.diagnosis,
                "subjects": profile.subjects,
                "preferred_styles": profile.preferred_styles,
                "frustration_triggers": profile.frustration_triggers,
            }

            # Adaptor: synthetic_history uses VectorStore/LearningGraph interfaces
            # so we wrap our profile stores
            _VSAdaptor = _make_vs_adaptor(vs)
            _LGAdaptor = _make_lg_adaptor(lg)

            seed_history(
                vector_store=_VSAdaptor,
                learning_graph=_LGAdaptor,
                profile=profile_dict,
                n_sessions=n_history_sessions,
                verbose=verbose,
            )

        if verbose:
            print(f"  -> {vs.count()} turns in ChromaDB, {lg.graph.number_of_nodes()} topics in graph")


# ------------------------------------------------------------------
# Adaptors so seed_history works with ProfileVectorStore/Graph
# ------------------------------------------------------------------

def _make_vs_adaptor(pvs: ProfileVectorStore):
    """Duck-type adaptor for synthetic_history.seed_history."""
    class _A:
        def store_turn(self, user_msg, assistant_msg, topics, frustration, explanation_style, timestamp=None):
            pvs.store_turn(user_msg, assistant_msg, topics, frustration, explanation_style, timestamp)
    return _A()


def _make_lg_adaptor(plg: ProfileLearningGraph):
    class _A:
        def ensure_topic(self, t): plg.ensure_topic(t)
        def record_study(self, topic, struggled, explanation_style, was_helpful, hour=0):
            plg.record_study(topic, struggled, explanation_style, was_helpful, hour)
        def update_confidence(self, topic, delta):
            plg.ensure_topic(topic)
            n = plg.graph.nodes[topic]
            n["confidence"] = max(0.0, min(1.0, n["confidence"] + delta))
            plg.save()
    return _A()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Seeding {len(ALL_PROFILES)} profiles with {n} history sessions each...")
    seed_all_profiles(n_history_sessions=n, verbose=True)
    print("\nDone.")
