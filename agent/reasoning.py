"""
LangGraph ReAct-style reasoning loop for ARIA.

Flow per turn:
  retrieve_context → detect_state → generate_response → update_stores

Before every response the agent:
  1. Queries ChromaDB for semantically similar past exchanges
  2. Pulls topic data from the learning graph
  3. Detects frustration (smart multi-signal detection)
  4. Normalises topics via LLM before storing
  5. Checks whether current topic connects to a stated goal
  6. Enforces ADHD-friendly formatting rules
  7. Saves the exchange back to ChromaDB and updates the graph
"""

import json
import re
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
import operator

_SESSION_LOG = Path(__file__).parent.parent / "data" / "real_sessions.jsonl"

import ollama
from langgraph.graph import StateGraph, END

from memory.vector_store import VectorStore
from memory.graph import LearningGraph


# ------------------------------------------------------------------
# Metacognition engine bootstrap
# ------------------------------------------------------------------
# The think-aloud metacognition engine lives in the sibling `eval/` project
# (eval/metacognition). Add it to the path lazily so importing reasoning.py
# never hard-fails if that subproject is missing.

def _load_metacognition():
    import sys
    eval_dir = Path(__file__).parent.parent / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from metacognition.analyzer import CognitiveStateAnalyzer
    from metacognition.interventions import MetacognitiveInterventionGenerator
    from metacognition.tracker import MetacognitionTracker
    from metacognition.transfer import TransferDetector
    from metacognition.calibration import CalibrationTracker
    from metacognition.timing import InterventionTimer
    return (CognitiveStateAnalyzer,
            MetacognitiveInterventionGenerator,
            MetacognitionTracker,
            TransferDetector,
            CalibrationTracker,
            InterventionTimer)


# Subtle UI indicator per cognitive state.
STATE_INDICATOR = {
    "FLOW": "🟢", "PLANNING": "🟢",
    "CONFUSED": "🟡", "RUSHING": "🟡",
    "FRUSTRATED": "🔴", "STUCK": "🔴",
    "INSIGHT": "✨",
}


# ------------------------------------------------------------------
# State definition
# ------------------------------------------------------------------

class ARIAState(TypedDict):
    user_input: str
    messages: Annotated[List[dict], operator.add]
    memory_context: str
    graph_context: str
    topics_mentioned: List[str]
    frustration_detected: bool
    frustration_level: int          # 0=none, 1=mild, 2=strong (spiral)
    explanation_style: str
    response: str
    user_profile: dict
    goal_connection: str            # one-sentence goal reminder or ""
    session_goal_reminded: bool     # only remind once per session


# ------------------------------------------------------------------
# Frustration detection — multi-signal
# ------------------------------------------------------------------

FRUSTRATION_PATTERNS = [
    r"\bi\s*(don'?t|dont|can'?t|cannot)\s*(get|understand|follow|do|see)\b",
    r"\b(ugh+|argh+|wtf|omg|ffs|help\s*me)\b",
    r"\bstupid\b",
    r"\bthis\s+(makes\s+no\s+sense|is\s+impossible|is\s+too\s+hard|sucks)\b",
    r"\bi'?m?\s*(lost|confused|overwhelmed|stuck|frustrated|giving\s+up)\b",
    r"\bwhy\s+(won'?t|doesn'?t|can'?t)\s+(this|it)\b",
    r"!!!+",
    r"\bgive\s+up\b",
    r"\bi\s+hate\b",
    r"\bidk\b",
    r"\bidek\b",
    r"\bidc\b",
    r"\bwhatever\b",
    r"\bforget\s+it\b",
    r"\bnevermind\b",
    r"\bnvm\b",
]

def detect_frustration(text: str) -> bool:
    lower = text.lower()
    for pat in FRUSTRATION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def _short_response(text: str) -> bool:
    """True if the response is very short — a signal of disengagement."""
    return len(text.strip().split()) < 5


# ------------------------------------------------------------------
# Topic normalisation — canonical forms
# ------------------------------------------------------------------

TOPIC_ALIASES = {
    # Math
    r"\b(trig|trigonometry|sin\s*cos\s*tan|sohcahtoa)\b": "Trigonometry",
    r"\b(alg(ebra)?|linear\s+eq(uation)?s?)\b": "Algebra",
    r"\b(calc(ulus)?|derivatives?|integrals?|limits?)\b": "Calculus",
    r"\b(geo(metry)?|shapes?|angles?|triangles?)\b": "Geometry",
    r"\b(stats?|statistics?|probability)\b": "Statistics",
    r"\b(arith(metic)?|fractions?|decimals?|percentages?)\b": "Arithmetic",
    # Science
    r"\b(bio(logy)?|cells?|genetics?|evolution|organisms?)\b": "Biology",
    r"\b(chem(istry)?|elements?|compounds?|reactions?|periodic\s+table)\b": "Chemistry",
    r"\b(phys(ics)?|forces?|motion|energy|waves?|circuits?)\b": "Physics",
    r"\b(earth\s+sci(ence)?|geology|meteorology)\b": "Earth Science",
    # English / Writing
    r"\b(essay|writing|5[\s-]*para(graph)?|thesis|paragraphs?)\b": "Essay Writing",
    r"\b(grammar|punctuation|sentences?|syntax)\b": "Grammar",
    r"\b(reading\s+comp(rehension)?|passages?|inference)\b": "Reading Comprehension",
    r"\b(vocab(ulary)?|words?|definitions?)\b": "Vocabulary",
    r"\b(lit(erature)?|novels?|poetry|shakespeare)\b": "Literature",
    # Standardised tests
    r"\b(act\b|act\s+prep|act\s+math|act\s+english|act\s+reading|act\s+science)\b": "ACT Prep",
    r"\b(sat\b|sat\s+prep)\b": "SAT Prep",
    r"\b(ap\s+bio(logy)?)\b": "AP Biology",
    r"\b(ap\s+(calc|calculus))\b": "AP Calculus",
    # Programming
    r"\b(python|py\b)\b": "Python",
    r"\b(java(script)?|js\b)\b": "JavaScript",
    r"\b(coding|programming)\b": "Programming",
    # History / Social Studies
    r"\b(hist(ory)?|wwi+|world\s+war)\b": "History",
    r"\b(econ(omics)?|supply|demand|markets?)\b": "Economics",
    r"\b(gov(ernment)?|civics?|politics?)\b": "Government",
    # Languages
    r"\b(spanish|español)\b": "Spanish",
    r"\b(french|français)\b": "French",
}

def normalise_topic(raw: str) -> str:
    """Map a raw topic string to its canonical form."""
    lower = raw.lower().strip()
    for pattern, canonical in TOPIC_ALIASES.items():
        if re.search(pattern, lower):
            return canonical
    # Title-case the raw string as a fallback
    return raw.strip().title()


def extract_and_normalise_topics(text: str, known_topics: List[str]) -> List[str]:
    found = set()
    lower = text.lower()

    # Match known graph topics directly
    for topic in known_topics:
        if topic.lower() in lower:
            found.add(topic)

    # Match alias patterns
    for pattern, canonical in TOPIC_ALIASES.items():
        if re.search(pattern, lower):
            found.add(canonical)

    return sorted(found)


# ------------------------------------------------------------------
# Sycophantic opener filter
# ------------------------------------------------------------------

SYCOPHANTIC_OPENERS = re.compile(
    r"^(great\s+question[!.]?|certainly[!.]?|of\s+course[!.]?|absolutely[!.]?|"
    r"sure[!.]?|definitely[!.]?|that'?s?\s+a\s+(great|good|wonderful|excellent|fantastic)"
    r")[,\s]",
    re.IGNORECASE,
)

def strip_sycophantic_opener(text: str) -> str:
    return SYCOPHANTIC_OPENERS.sub("", text).lstrip()


# ------------------------------------------------------------------
# Adaptive style instructions per profile
# ------------------------------------------------------------------

def build_style_instructions(profile: dict) -> str:
    style = profile.get("learning_style", "mixed")
    diagnosis = [d.lower() for d in profile.get("diagnosis", [])]
    attention = profile.get("attention_span_minutes", 20)
    answer_style = profile.get("answer_style", "brief")

    parts = []

    # Core learning style
    style_map = {
        "visual": (
            "Describe everything spatially. Use ASCII diagrams when they help. "
            "Say 'picture this' or 'imagine a line with...' before explaining anything abstract."
        ),
        "analogy": (
            "Every concept gets a real-world comparison FIRST, before any technical explanation. "
            "Lead with 'It's like...' or 'Think of it like...'"
        ),
        "step_by_step": (
            "Always number every step. Never compress multiple steps into one. "
            "Do Step 1, then stop and ask if they're ready before continuing."
        ),
        "kinesthetic": (
            "Frame everything as something they're physically doing. "
            "'Imagine you are holding...', 'Now move this piece to...', hands-on framing always."
        ),
        "mixed": (
            "Blend a real-world analogy with numbered micro-steps. "
            "Use brief spatial descriptions when it helps."
        ),
    }
    parts.append(f"LEARNING STYLE — {style.upper()}: {style_map.get(style, style_map['mixed'])}")

    # Attention span
    if attention < 15:
        parts.append(
            "SHORT ATTENTION SPAN (<15 min): Maximum 2 sentences per response. "
            "Always use micro-steps. After every step, ask 'Got it?' before moving on. "
            "Never give more than one idea at a time."
        )
    elif attention < 25:
        parts.append(
            "MODERATE ATTENTION SPAN: Keep responses under 4 sentences. "
            "Break tasks into small numbered steps."
        )

    # Answer length preference
    response_length = profile.get("response_length")
    if response_length == "short":
        parts.append("PREFERRED ANSWER LENGTH: Max 2 sentences. Get to the point fast.")
    elif response_length == "detailed":
        parts.append("PREFERRED ANSWER LENGTH: Up to 5 sentences. A fuller walk-through is welcome.")
    elif response_length == "medium":
        parts.append("PREFERRED ANSWER LENGTH: Max 3 sentences.")
    elif answer_style == "brief":
        parts.append("PREFERRED ANSWER LENGTH: Brief and sharp. Get to the point fast.")
    else:
        parts.append("PREFERRED ANSWER LENGTH: Detailed walk-through preferred.")

    # Diagnosis-specific rules
    if any("anxiety" in d for d in diagnosis):
        parts.append(
            "ANXIETY: Extra validating tone throughout. "
            "Never imply they should already know this. "
            "Normalise struggle explicitly: 'This trips most people up.'"
        )
    if any("dyslexia" in d for d in diagnosis):
        parts.append(
            "DYSLEXIA: Short words over long ones always. Simple sentence structure. "
            "Never ask them to read a long passage. Break text into tiny chunks."
        )
    if any(d in ("asd", "autism", "autistic") for d in diagnosis):
        parts.append(
            "ASD: Be precise and literal. No sarcasm, no idioms without explanation. "
            "Always give exact definitions. Avoid ambiguous phrasing."
        )
    if any("adhd" in d for d in diagnosis):
        parts.append(
            "ADHD: One idea per message. Use bold for the key point. "
            "Check in frequently. Never give a wall of text."
        )

    return "\n".join(parts)


# ------------------------------------------------------------------
# System prompt builder
# ------------------------------------------------------------------

def _days_active(profile: dict) -> int:
    """Whole days since the profile was created (0 if unknown)."""
    created = profile.get("created_at")
    if not created:
        return 0
    try:
        delta = datetime.now() - datetime.fromisoformat(created)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


def _max_sentences(profile: dict) -> int:
    return {"short": 2, "medium": 3, "detailed": 5}.get(
        profile.get("response_length"), 3
    )


def build_system_prompt(
    user_profile: dict,
    memory_context: str,
    graph_context: str,
    frustration: bool,
    frustration_level: int,
    explanation_style: str,
    goal_connection: str,
) -> str:
    name = user_profile.get("name", "friend")
    grade = user_profile.get("grade", "")
    diagnosis = user_profile.get("diagnosis", [])
    diagnosis_str = ", ".join(diagnosis) if diagnosis else "none stated"
    learning_style = user_profile.get("learning_style", "mixed").replace("_", "-")
    subjects = user_profile.get("subjects", [])
    subjects_str = ", ".join(subjects) if subjects else "general learning"
    goals = user_profile.get("goals", [])
    focus_minutes = user_profile.get("attention_span_minutes", 20)
    peak_hours = user_profile.get("study_hours", [])
    peak_str = ", ".join(f"{h}:00" for h in peak_hours) if peak_hours else "not set"
    biggest_struggle = user_profile.get("biggest_struggle", "")
    what_helped = user_profile.get("what_helped", "")
    what_failed = user_profile.get("what_failed", "")
    days_active = _days_active(user_profile)
    max_sent = _max_sentences(user_profile)
    style_instructions = build_style_instructions(user_profile)

    if frustration_level >= 2:
        tone = (
            "The student is spiraling — multiple frustration signals in a row. "
            "Acknowledge in exactly ONE sentence. Then drop to the ABSOLUTE SMALLEST micro-step possible. "
            "Optionally suggest a 2-minute break. Do NOT add more information."
        )
    elif frustration:
        tone = (
            "The student seems frustrated. Acknowledge in one sentence, then help them take ONE tiny step. "
            "Be warm. Never pile on more content."
        )
    else:
        tone = "Warm, direct, like a smart friend — not a corporate tutor."

    goals_str = "\n".join(f"  - {g}" for g in goals) if goals else "  (none stated)"

    system = f"""You are ARIA — a metacognitive learning assistant built specifically
for neurodivergent students. You talk like a smart, patient friend — never a corporate tutor.

STUDENT PROFILE:
Name: {name}
{f"Grade: {grade}" if grade else ""}
Diagnosis: {diagnosis_str}
Learning style: {learning_style}
Subjects: {subjects_str}
GOALS:
{goals_str}
Focus window: {focus_minutes} minutes
Peak hours: {peak_str}
{f"Biggest struggle: {biggest_struggle}" if biggest_struggle else ""}
{f"What has helped them: {what_helped}" if what_helped else ""}
{f"What has never worked: {what_failed}" if what_failed else ""}

YOUR JOB:
You are not just a tutor — you are a metacognitive coach. Your goal is to make
{name} a better learner, not only to answer their questions.

TONE: {tone}

{style_instructions}

ABSOLUTE RULES — NEVER BREAK THESE:
1. Max {max_sent} sentences per response block. Always.
2. Number every step. Never skip a step.
3. Never re-explain something the same way twice. If they didn't get it, use a completely different angle.
4. Never open with: "Great question!", "Certainly!", "Of course!", "Absolutely!", "I'd be happy to", or any sycophantic phrase.
5. Talk like a smart friend, not a corporate tutor.
6. One question at a time. Never ask multiple questions.
7. If they're stuck: ask the SMALLEST possible question to get them unstuck.
8. If they got it right: one sentence acknowledgment, then immediately move forward.
9. Never give walls of text.
10. Always respond to what they actually said, not what you think they meant.

{f"GOAL CONNECTION THIS TURN: {goal_connection}" if goal_connection else ""}

RELEVANT PAST CONTEXT:
{memory_context or "No past context yet."}

LEARNING GRAPH:
{graph_context or "No topic history yet."}

Remember: you've been working with {name} for {days_active} day(s). Use that history."""
    return system


# ------------------------------------------------------------------
# Goal connection check
# ------------------------------------------------------------------

def check_goal_connection(topics: List[str], profile: dict, already_reminded: bool) -> str:
    if already_reminded:
        return ""
    goals = profile.get("goals", [])
    if not goals or not topics:
        return ""
    goals_lower = " ".join(goals).lower()
    for topic in topics:
        if any(word in goals_lower for word in topic.lower().split()):
            return f"This topic connects to your goal: {goals[0]}."
    return ""


# ------------------------------------------------------------------
# Agent class
# ------------------------------------------------------------------

class ARIAAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        learning_graph: LearningGraph,
        user_profile: dict,
        lora_adapter_path: Optional[str] = None,
        think_aloud_mode: bool = False,
    ):
        self.vs = vector_store
        self.lg = learning_graph
        self.profile = user_profile
        self.model = "llama3.2:3b"
        self.think_aloud_mode = think_aloud_mode
        self._lora_adapter_path = lora_adapter_path
        self._lora_model = None
        self._lora_tokenizer = None
        if lora_adapter_path:
            self._load_lora(lora_adapter_path)
        self._graph = self._build_graph()
        self._chat_history: List[dict] = []
        self._current_episode_id: str = str(uuid.uuid4())
        self._session_goal_reminded: bool = False
        self._session_start: datetime = datetime.now()
        self._break_suggested: bool = False

        # Spiral detection: track last N response lengths
        self._recent_inputs: deque = deque(maxlen=4)

        # --- Metacognition engine (lazy) ---
        self._meta_analyzer = None
        self._meta_interventions = None
        self._meta_tracker = None
        self._meta_transfer = None       # self-initiated metacognition detection
        self._meta_calibration = None    # confidence-vs-accuracy calibration
        self._meta_timer = None          # intervention-timing optimisation
        self._meta_last_state: Optional[str] = None
        self._meta_last_prompt: str = ""      # ARIA's previous think-aloud message
        self._meta_awaiting_intervention: bool = False
        self._meta_episode: Optional[dict] = None   # active negative-state episode
        self._calib_pending: Optional[dict] = None  # confidence awaiting an outcome
        self._current_problem_ctx: dict = {}        # most recent think-aloud problem
        if think_aloud_mode:
            self._init_metacognition()

    # ------------------------------------------------------------------
    # Think-aloud metacognition
    # ------------------------------------------------------------------

    def _init_metacognition(self) -> bool:
        """Load the metacognition engine on demand. Returns True on success."""
        if self._meta_analyzer is not None:
            return True
        try:
            (Analyzer, Interventions, Tracker,
             Transfer, Calibration, Timer) = _load_metacognition()
        except Exception as e:
            print(f"[reasoning] Metacognition engine unavailable: {e}")
            return False
        name = self.profile.get("name", "default")
        self._meta_analyzer = Analyzer()
        self._meta_interventions = Interventions(student_name=name)
        self._meta_tracker = Tracker(student_name=name)
        self._meta_transfer = Transfer(student_name=name)
        self._meta_calibration = Calibration(student_name=name)
        self._meta_timer = Timer(student_name=name)
        self._meta_tracker.start_session()
        self.think_aloud_mode = True
        return True

    def _adhd_profile_tag(self) -> str:
        """Normalise this student's diagnosis into an ADHD subtype tag."""
        diag = self.profile.get("diagnosis") or self.profile.get("adhd_type") or ""
        if isinstance(diag, (list, tuple)):
            diag = " ".join(str(d) for d in diag)
        d = str(diag).lower()
        has_hyper = "hyperactive" in d or "impulsive" in d
        has_inatt = "inattentive" in d
        if "combined" in d or (has_hyper and has_inatt):
            return "combined"
        if has_hyper:
            return "hyperactive"
        if has_inatt:
            return "inattentive"
        return "unknown"

    THINK_ALOUD_PROMPT = (
        "Think through this out loud before answering. "
        "Type your reasoning OR use the microphone."
    )

    # When the student self-initiates metacognition, ARIA acknowledges the
    # habit instead of prompting for something they already did.
    _TRANSFER_ACK = {
        "planning": "I noticed you planned before jumping in — that's exactly the habit. Keep going.",
        "monitoring": "Nice — you stopped to check your own thinking without me asking. That's the skill. Keep going.",
        "reflection": "That's real reflection — you're seeing *why* it works, not just that it does. Keep going.",
    }

    # When the timer says it's not yet the moment to intervene, ARIA stays out
    # of the way with a light nudge (NOT a metacognitive question — so a student
    # who then self-monitors still counts as self-initiated).
    _CONTINUE_NUDGE = "Keep going — talk me through the next bit."

    _FALLBACK_PROBLEMS = {
        "algebra": "Solve for x:  3(x - 4) = 2x + 5",
        "act": "A line passes through (2, 3) and (6, 11). What is its slope?",
        "sat": "If 2x + 3y = 12 and x = 3, what is y?",
        "geometry": "A right triangle has legs of length 6 and 8. How long is the hypotenuse?",
        "calculus": "Find the derivative of f(x) = 3x^2 - 5x + 2.",
        "trigonometry": "In a right triangle, the angle is 30° and the hypotenuse is 10. How long is the opposite side?",
        "statistics": "A dataset is 4, 8, 15, 16, 23, 42. What is the median?",
        "physics": "A car accelerates from 0 to 20 m/s in 4 seconds. What is its acceleration?",
        "chemistry": "How many moles are in 36 grams of water (H2O)?",
        "biology": "In a cross between two heterozygous (Aa) parents, what fraction of offspring are recessive (aa)?",
    }

    def generate_think_aloud_problem(self) -> dict:
        """Generate one practice problem tuned to the student's subjects/goals.

        Prioritises a concept that is due for SRS review. Returns
        {"problem": str, "topic": str}. Falls back to a template bank if the
        LLM is unavailable so this never hard-fails.
        """
        subjects = self.profile.get("subjects", []) or ["general learning"]
        goals = self.profile.get("goals", [])
        topic = subjects[0]

        # Prefer an SRS-due concept if one exists.
        due_concept = ""
        try:
            from memory.srs import get_due_cards
            due = get_due_cards()
            if due:
                due_concept = due[0].get("concept", "")
                topic = due[0].get("topic", topic)
        except Exception:
            pass

        focus = due_concept or topic
        grade = self.profile.get("grade", "")
        goal_str = f" (their goal: {goals[0]})" if goals else ""
        prompt = (
            f"Generate ONE short practice problem for a {grade or 'high-school'} student "
            f"studying {topic}{goal_str}, focused on '{focus}'. "
            "Return ONLY the problem itself — no answer, no solution, no preamble, "
            "no 'Here is'. One or two sentences maximum. Make it concrete and solvable."
        )
        try:
            result = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                        "You write single, concrete practice problems. Output only the problem."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.8, "num_predict": 90},
                keep_alive=3600,
            )
            problem = result.message.content.strip().strip('"')
            problem = strip_sycophantic_opener(problem)
            if not problem:
                raise ValueError("empty problem")
        except Exception:
            key = topic.lower().strip()
            problem = None
            for k, v in self._FALLBACK_PROBLEMS.items():
                if k in key:
                    problem = v
                    break
            problem = problem or self._FALLBACK_PROBLEMS["algebra"]

        # A fresh problem starts a new calibration cycle: any prior confidence
        # that was never resolved is dropped, and the negative-state episode is
        # closed out so timing episodes don't bleed across problems.
        if self._meta_tracker is not None:
            self._close_episode(recovered=False)
        self._calib_pending = None
        difficulty = self.profile.get("default_difficulty", "medium")
        ctx = {
            "problem": problem,
            "topic": topic,
            "difficulty": difficulty,
            "problem_id": uuid.uuid4().hex[:12],
        }
        self._current_problem_ctx = ctx
        return ctx

    # ---- calibration: confidence elicitation ----

    def confidence_prompt(self) -> str:
        """The question ARIA asks before a first attempt."""
        if not self._init_metacognition():
            return "Before you try this — how confident are you? 1 (totally lost) to 5 (got this)"
        return self._meta_calibration.CONFIDENCE_PROMPT

    def set_confidence(self, rating, problem: Optional[dict] = None) -> dict:
        """Store the student's 1-5 confidence for the current problem.

        Called once, before the first attempt. Correctness is resolved later via
        resolve_confidence(); the record is only written once we know the outcome.
        """
        if not self._init_metacognition():
            return {"ok": False}
        prob = problem or getattr(self, "_current_problem_ctx", {}) or {}
        from metacognition.calibration import clamp_confidence
        conf = clamp_confidence(rating)
        topic = prob.get("topic")
        if not topic:
            try:
                topic = self.get_last_topic() or "unknown"
            except Exception:
                topic = "unknown"
        self._calib_pending = {
            "problem_id": prob.get("problem_id") or uuid.uuid4().hex[:12],
            "topic": topic,
            "difficulty": prob.get("difficulty") or "medium",
            "confidence": conf,
            "states": [],
        }
        return {"ok": True, "confidence": conf}

    def resolve_confidence(self, correct: bool) -> dict:
        """Finalise the calibration record for the current problem.

        `correct` comes from the student's self-report or from checking their
        answer against the known correct answer. The cognitive state recorded is
        the modal state observed while they worked the problem.
        """
        if not self._init_metacognition() or not self._calib_pending:
            return {"ok": False}
        p = self._calib_pending
        states = p.get("states") or ([self._meta_last_state] if self._meta_last_state else [])
        modal = max(set(states), key=states.count) if states else (self._meta_last_state or "")
        session = getattr(self._meta_tracker.current, "session_id", "") if self._meta_tracker.current else ""
        rec = self._meta_calibration.record(
            problem_id=p["problem_id"],
            topic=p["topic"],
            difficulty=p["difficulty"],
            confidence_before=p["confidence"],
            correct=bool(correct),
            cognitive_state_during=modal,
            session=session,
        )
        self._calib_pending = None
        return {"ok": True, "record": rec}

    # States that count as "negative" for intervention-timing purposes.
    _NEGATIVE_STATES = {"CONFUSED", "RUSHING", "FRUSTRATED", "STUCK"}

    def think_aloud_turn(
        self,
        student_input: str,
        audio_features: Optional[dict] = None,
    ) -> dict:
        """Process one think-aloud utterance WITHOUT answering the problem.

        The turn runs the full measurement pipeline (spec §5):
          1. TransferDetector — did the student self-initiate metacognition?
          2. CognitiveStateAnalyzer — detect the cognitive state.
          3. InterventionTimer — is now the right moment to intervene?
          4. Generate an intervention ONLY if the timer says yes (or acknowledge
             self-initiation instead of overriding a habit they already showed).
          5. Log everything to the tracker + all three measurement systems.
        """
        if not self._init_metacognition():
            return {
                "state": "UNKNOWN",
                "question": "Tell me what you're thinking so far.",
                "indicator": "🟡",
                "flags": {}, "escalated": False, "intervened": False,
            }

        prev_prompt = self._meta_last_prompt
        profile_tag = self._adhd_profile_tag()
        subject = (getattr(self, "_current_problem_ctx", {}) or {}).get("topic", "") \
            or (self.get_last_topic() or "")
        session_id = getattr(self._meta_tracker.current, "session_id", "") \
            if self._meta_tracker.current else ""

        # 1. Cognitive state (with transfer detection folded in via the previous
        #    ARIA prompt) + a persisted transfer record for longitudinal tracking.
        analysis = self._meta_analyzer.analyze(
            student_input, audio_features=audio_features, aria_previous_prompt=prev_prompt)
        state = analysis["state"]
        self_initiated = bool(analysis["self_initiated_metacognition"])
        mtype = analysis["metacognitive_type"]

        # Attribute the outcome of the *previous* intervention (if we gave one)
        # to the change we just observed.
        if self._meta_awaiting_intervention and self._meta_last_state:
            self._meta_interventions.record_outcome(self._meta_last_state, state)
            self._meta_tracker.set_intervention_outcome(state)

        self._meta_tracker.record_state(analysis, text=student_input)
        turn_no = self._meta_tracker._turn

        transfer_rec = self._meta_transfer.detect(
            student_input, aria_previous_prompt=prev_prompt,
            turn=turn_no, session=session_id,
            student_profile=profile_tag, subject=subject)

        # Feed the running calibration cycle the state observed this attempt.
        if self._calib_pending is not None:
            self._calib_pending["states"].append(state)

        consecutive = self._meta_tracker.consecutive_state(state)
        is_negative = state in self._NEGATIVE_STATES

        # If a negative-state episode was open and the student has now recovered,
        # close it out as recovered (records a timing episode if we'd intervened).
        if self._meta_episode and not is_negative:
            self._close_episode(recovered=True)

        intervention = None
        intervened = False
        acknowledged = False

        if self_initiated and not is_negative:
            # 4a. They planned/checked/reflected on their own — acknowledge the
            #     habit, do NOT override with a (planning) intervention.
            question = self._TRANSFER_ACK.get(mtype, "That's exactly the habit. Keep going.")
            label_state = state
            acknowledged = True
        elif is_negative:
            # 4b. Negative state — the InterventionTimer decides the moment.
            if not self._meta_episode or self._meta_episode.get("state") != state:
                # A different negative state ends the prior episode (not a recovery).
                if self._meta_episode:
                    self._close_episode(recovered=False)
                self._start_episode(state)

            ready = self._meta_timer.should_intervene(state, consecutive, profile_tag)
            if ready and not self._meta_episode.get("intervened"):
                intervention = self._meta_interventions.generate(state, consecutive_count=consecutive)
                self._meta_tracker.record_intervention(intervention, state_before=state)
                self._meta_episode.update({
                    "intervened": True,
                    "turns_before": consecutive,
                    "intervention_turn": turn_no,
                    "intervention_text": intervention["text"],
                })
                intervened = True
                question = intervention["text"]
                label_state = intervention.get("state", state)
            else:
                # Not the moment yet (or already intervened this episode): stay
                # out of the way with a light, non-metacognitive nudge.
                question = self._CONTINUE_NUDGE
                label_state = state
        else:
            # 4c. Positive/neutral, unprompted metacognition absent — a normal
            #     Socratic intervention (FLOW/PLANNING/INSIGHT deepeners).
            intervention = self._meta_interventions.generate(state, consecutive_count=consecutive)
            self._meta_tracker.record_intervention(intervention, state_before=state)
            intervened = True
            question = intervention["text"]
            label_state = intervention.get("state", state)

        self._meta_last_state = state
        self._meta_last_prompt = question
        self._meta_awaiting_intervention = intervened

        return {
            "state": state,
            "confidence": analysis["confidence"],
            "evidence": analysis.get("evidence", ""),
            "question": question,
            "indicator": STATE_INDICATOR.get(state, "🟡"),
            "escalated": intervention.get("escalated", False) if intervention else False,
            "escalation_kind": intervention.get("escalation_kind") if intervention else None,
            "intervention_state": label_state,
            "intervened": intervened,
            "acknowledged": acknowledged,
            "self_initiated_metacognition": self_initiated,
            "metacognitive_type": mtype,
            "prompted_by_aria": transfer_rec["prompted_by_aria"],
            "turns_in_state": consecutive,
            "recommended_wait": self._meta_timer.recommended_wait(state, profile_tag)
                if is_negative else 0,
            "flags": {
                "planning_detected": analysis["planning_detected"],
                "self_correction": analysis["self_correction"],
                "insight_moment": analysis["insight_moment"],
                "gave_up": analysis["gave_up"],
            },
            "method": analysis["method"],
        }

    # ---- intervention-timing episode bookkeeping ----

    def _start_episode(self, state: str) -> None:
        self._meta_episode = {
            "episode_id": uuid.uuid4().hex[:12],
            "state": state,
            "intervened": False,
            "turns_before": None,
            "intervention_turn": None,
            "intervention_text": "",
        }

    def _close_episode(self, recovered: bool) -> None:
        """Finalise the active negative-state episode, recording a timing event.

        Only episodes where ARIA actually intervened produce a timing data point
        (there's no timing to learn from an episode ARIA never acted on).
        """
        ep = self._meta_episode
        self._meta_episode = None
        if not ep or not ep.get("intervened") or self._meta_timer is None:
            return
        cur_turn = self._meta_tracker._turn if self._meta_tracker else 0
        iv_turn = ep.get("intervention_turn") or cur_turn
        turns_to_recovery = max(0, cur_turn - iv_turn)
        subject = (getattr(self, "_current_problem_ctx", {}) or {}).get("topic", "")
        session_id = getattr(self._meta_tracker.current, "session_id", "") \
            if self._meta_tracker and self._meta_tracker.current else ""
        self._meta_timer.record_episode(
            episode_id=ep["episode_id"],
            state=ep["state"],
            turns_in_state_before_intervention=ep.get("turns_before") or 1,
            intervention_text=ep.get("intervention_text", ""),
            turns_to_recovery=turns_to_recovery,
            recovered=recovered,
            student_profile=self._adhd_profile_tag(),
            subject=subject,
            session=session_id,
        )

    def end_think_aloud_session(self) -> dict:
        """Close the current think-aloud session and return its metrics."""
        if self._meta_tracker is None:
            return {}
        # Any open negative episode ends unresolved.
        self._close_episode(recovered=False)
        record = self._meta_tracker.end_session()
        # Re-learn optimal intervention timing from history (adaptive after
        # enough sessions); reasoning reads this config on the next session.
        if self._meta_timer is not None:
            try:
                self._meta_timer.save_config(n_sessions=len(self._meta_tracker.sessions))
            except Exception as e:
                print(f"[reasoning] timing config update skipped: {e}")
        return record.get("metrics", {})

    def metacognition_tracker(self):
        """Expose the tracker (for dashboards); initialises if needed."""
        if self._meta_tracker is None:
            self._init_metacognition()
        return self._meta_tracker

    def transfer_detector(self):
        """Expose the TransferDetector (for dashboards); initialises if needed."""
        self._init_metacognition()
        return self._meta_transfer

    def calibration_tracker(self):
        """Expose the CalibrationTracker (for dashboards); initialises if needed."""
        self._init_metacognition()
        return self._meta_calibration

    def intervention_timer(self):
        """Expose the InterventionTimer (for dashboards); initialises if needed."""
        self._init_metacognition()
        return self._meta_timer

    def _load_lora(self, adapter_path: str) -> None:
        try:
            from mlx_lm import load
            self._lora_model, self._lora_tokenizer = load(
                "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                adapter_path=adapter_path,
            )
        except Exception as e:
            print(f"[reasoning] WARNING: Could not load LoRA adapter: {e}")
            print("[reasoning] Falling back to Ollama llama3.2:3b")

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _retrieve_context(self, state: ARIAState) -> ARIAState:
        query = state["user_input"]

        past = self.vs.retrieve_context(query, n_results=5)
        memory_parts = []
        for item in past:
            ts = item["meta"].get("timestamp", "")[:10]
            memory_parts.append(f"[{ts}] {item['text'][:300]}")
        memory_context = "\n---\n".join(memory_parts)

        # Pull relevant knowledge base chunks
        try:
            from knowledge.ingestion import query_knowledge
            kb_hits = query_knowledge(query, n_results=3)
            if kb_hits:
                kb_parts = []
                for h in kb_hits:
                    src = h.get("source_file", "notes")
                    pg = h.get("page_number", 1)
                    kb_parts.append(f"[From {src}, p.{pg}] {h['text'][:250]}")
                memory_context += "\n\nKNOWLEDGE BASE:\n" + "\n---\n".join(kb_parts)
        except Exception:
            pass

        summaries = self.vs.retrieve_summaries(query, n=2)
        if summaries:
            memory_context += "\n\nPAST LEARNING SUMMARIES:\n" + "\n".join(summaries[:2])

        all_topics = list(self.lg.graph.nodes())
        mentioned = extract_and_normalise_topics(query, all_topics)

        topic_infos = []
        for t in mentioned:
            info = self.lg.get_topic_info(t)
            if info:
                style = self.lg.best_explanation_style(t)
                conf = info.get("confidence", 0.5)
                struggles = info.get("struggle_count", 0)
                topic_infos.append(
                    f"'{t}': confidence={conf:.0%}, struggles={struggles}, best_style='{style}'"
                )

        struggling = self.lg.struggling_topics()
        if struggling:
            topic_infos.append(
                "Needs attention: " + ", ".join(t["topic"] for t in struggling[:3])
            )

        peak_hours = self.lg.peak_focus_hours()
        if peak_hours:
            hour_strs = [f"{h}:00" for h in peak_hours]
            topic_infos.append(f"Peak focus: {', '.join(hour_strs)}")

        graph_context = "\n".join(topic_infos)

        best_style = self.profile.get("learning_style", "mixed")
        if mentioned:
            styles = [self.lg.best_explanation_style(t) for t in mentioned]
            valid = [s for s in styles if s]
            if valid:
                best_style = max(set(valid), key=valid.count)

        state["memory_context"] = memory_context
        state["graph_context"] = graph_context
        state["topics_mentioned"] = mentioned
        state["explanation_style"] = best_style
        return state

    def _detect_state(self, state: ARIAState) -> ARIAState:
        text = state["user_input"]
        frustrated = detect_frustration(text)
        short = _short_response(text)

        self._recent_inputs.append({
            "frustrated": frustrated,
            "short": short,
        })

        # Spiral = 3+ of last 4 inputs were frustrated or very short
        spiral_count = sum(
            1 for r in self._recent_inputs if r["frustrated"] or r["short"]
        )
        frustration_level = 0
        if spiral_count >= 3:
            frustration_level = 2
        elif frustrated:
            frustration_level = 1

        state["frustration_detected"] = frustrated or short
        state["frustration_level"] = frustration_level

        # Goal connection (once per session)
        goal_conn = check_goal_connection(
            state["topics_mentioned"],
            self.profile,
            self._session_goal_reminded,
        )
        if goal_conn:
            self._session_goal_reminded = True
        state["goal_connection"] = goal_conn
        state["session_goal_reminded"] = self._session_goal_reminded

        return state

    def _generate_response(self, state: ARIAState) -> ARIAState:
        system_prompt = build_system_prompt(
            user_profile=self.profile,
            memory_context=state["memory_context"],
            graph_context=state["graph_context"],
            frustration=state["frustration_detected"],
            frustration_level=state["frustration_level"],
            explanation_style=state["explanation_style"],
            goal_connection=state["goal_connection"],
        )

        msgs = [{"role": "system", "content": system_prompt}]
        for m in self._chat_history[-12:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": state["user_input"]})

        # --- Session break suggestion after 20 minutes ---
        elapsed_min = (datetime.now() - self._session_start).total_seconds() / 60
        break_injection = ""
        if elapsed_min >= 20 and not self._break_suggested:
            self._break_suggested = True
            break_injection = (
                "\n\n[ARIA NOTE: Student has been studying for 20 minutes. "
                "After your response, suggest a 5-minute break in one sentence. "
                "Then say you'll pick up exactly where you left off.]"
            )
        if break_injection:
            msgs[-1]["content"] += break_injection

        if self._lora_model is not None:
            from mlx_lm import generate as mlx_generate
            prompt = (
                f"<|system|>\n{system_prompt}\n"
                + "".join(
                    f"<|{'user' if m['role']=='user' else 'assistant'}|>\n{m['content']}\n"
                    for m in msgs[1:]
                )
                + "<|assistant|>\n"
            )
            response = mlx_generate(
                self._lora_model, self._lora_tokenizer,
                prompt=prompt, max_tokens=256, verbose=False,
            )
            response = response.split("<|user|>")[0].split("<|system|>")[0].strip()
        else:
            result = ollama.chat(
                model=self.model,
                messages=msgs,
                options={"temperature": 0.7, "num_predict": 256},
                keep_alive=3600,
            )
            response = result.message.content.strip()

        response = strip_sycophantic_opener(response)
        state["response"] = response
        return state

    def _update_stores(self, state: ARIAState) -> ARIAState:
        user_msg = state["user_input"]
        response = state["response"]
        topics = state["topics_mentioned"]
        frustration = state["frustration_detected"]
        style = state["explanation_style"]

        self.vs.store_turn(
            user_msg=user_msg,
            assistant_msg=response,
            topics=topics,
            frustration=frustration,
            explanation_style=style,
        )

        for topic in topics:
            self.lg.record_study(
                topic=topic,
                struggled=frustration,
                explanation_style=style,
                was_helpful=True,
                hour=datetime.now().hour,
            )
            for other in topics:
                if other != topic:
                    self.lg.link_topics(topic, other)

            # Register topic as SRS card if it's new
            try:
                from memory.srs import ensure_card
                ensure_card(topic, topic)
            except Exception:
                pass

        self._chat_history.append({"role": "user", "content": user_msg})
        self._chat_history.append({"role": "assistant", "content": response})
        if len(self._chat_history) > 40:
            self._chat_history = self._chat_history[-40:]

        _SESSION_LOG.parent.mkdir(exist_ok=True)
        record = {
            "episode_id": self._current_episode_id,
            "model": "aria",
            "persona": "real_user",
            "subject": topics[0] if topics else "general",
            "turn": len(self._chat_history) // 2 - 1,
            "student_msg": user_msg,
            "tutor_msg": response,
            "scores": {},
            "weighted_score": None,
            "solution_dump": False,
            "frustration_detected": frustration,
            "frustration_level": state["frustration_level"],
            "topics": topics,
            "explanation_style": style,
            "timestamp": datetime.now().isoformat(),
        }
        with open(_SESSION_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

        return state

    # ------------------------------------------------------------------
    # LangGraph build
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(ARIAState)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("detect_state", self._detect_state)
        builder.add_node("generate_response", self._generate_response)
        builder.add_node("update_stores", self._update_stores)

        builder.set_entry_point("retrieve_context")
        builder.add_edge("retrieve_context", "detect_state")
        builder.add_edge("detect_state", "generate_response")
        builder.add_edge("generate_response", "update_stores")
        builder.add_edge("update_stores", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        initial_state: ARIAState = {
            "user_input": user_input,
            "messages": [],
            "memory_context": "",
            "graph_context": "",
            "topics_mentioned": [],
            "frustration_detected": False,
            "frustration_level": 0,
            "explanation_style": self.profile.get("learning_style", "mixed"),
            "response": "",
            "user_profile": self.profile,
            "goal_connection": "",
            "session_goal_reminded": self._session_goal_reminded,
        }
        final_state = self._graph.invoke(initial_state)
        return final_state["response"]

    def chat_stream(self, user_input: str):
        """Yield response tokens one by one while still running the full pipeline."""
        initial_state: ARIAState = {
            "user_input": user_input,
            "messages": [],
            "memory_context": "",
            "graph_context": "",
            "topics_mentioned": [],
            "frustration_detected": False,
            "frustration_level": 0,
            "explanation_style": self.profile.get("learning_style", "mixed"),
            "response": "",
            "user_profile": self.profile,
            "goal_connection": "",
            "session_goal_reminded": self._session_goal_reminded,
        }

        # Run context retrieval and state detection
        state = self._retrieve_context(initial_state)
        state = self._detect_state(state)

        # Build messages the same way as _generate_response
        system_prompt = build_system_prompt(
            user_profile=self.profile,
            memory_context=state["memory_context"],
            graph_context=state["graph_context"],
            frustration=state["frustration_detected"],
            frustration_level=state["frustration_level"],
            explanation_style=state["explanation_style"],
            goal_connection=state["goal_connection"],
        )
        msgs = [{"role": "system", "content": system_prompt}]
        for m in self._chat_history[-12:]:
            msgs.append(m)

        elapsed_min = (datetime.now() - self._session_start).total_seconds() / 60
        user_content = user_input
        if elapsed_min >= 20 and not self._break_suggested:
            self._break_suggested = True
            user_content += (
                "\n\n[ARIA NOTE: Student has been studying for 20 minutes. "
                "After your response, suggest a 5-minute break in one sentence. "
                "Then say you'll pick up exactly where you left off.]"
            )
        msgs.append({"role": "user", "content": user_content})

        # Stream tokens from ollama
        full_response = ""
        for chunk in ollama.chat(
            model=self.model,
            messages=msgs,
            options={"temperature": 0.7, "num_predict": 256},
            keep_alive=3600,
            stream=True,
        ):
            token = chunk.message.content
            full_response += token
            yield token

        full_response = strip_sycophantic_opener(full_response)
        state["response"] = full_response

        # Persist to stores
        self._update_stores(state)

    def get_last_topic(self) -> str:
        """Most recently studied topic, for session recap."""
        recent = self.vs.get_recent_turns(n=1)
        if recent:
            try:
                topics = json.loads(recent[0]["meta"].get("topics", "[]"))
                if topics:
                    return topics[0]
            except Exception:
                pass
        return ""

    def get_stats(self) -> dict:
        topics = self.lg.all_topics_summary()
        struggling = self.lg.struggling_topics()
        peak_hours = self.lg.peak_focus_hours()
        total_turns = self.vs.conversations.count()
        return {
            "total_conversations": total_turns,
            "topics_tracked": len(topics),
            "struggling_topics": struggling,
            "peak_focus_hours": [f"{h}:00" for h in peak_hours],
            "topic_details": topics[:10],
        }

    def reset_break_timer(self) -> None:
        """Call after user returns from a break."""
        self._session_start = datetime.now()
        self._break_suggested = False
