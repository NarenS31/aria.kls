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
import random
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
from agent.intervention_pipeline import ClosedLoopInterventionPipeline
from agent.student_intent import warm_student_intent_model
from agent.student_understanding import understand_student_turn

_INTERVENTION_OUTCOMES = (
    Path(__file__).parent.parent / "data" / "intervention_outcomes.jsonl"
)

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


def offline_coaching_response(user_input: str) -> str:
    """Return a useful metacognitive prompt when the local model is unavailable."""
    text = (user_input or "").strip()
    lower = text.lower()
    if detect_frustration(text) or any(word in lower for word in ("stuck", "lost", "confused")):
        return (
            "**Let’s shrink the problem.** What is one fact, value, or step "
            "you know is definitely correct?"
        )
    if any(word in lower for word in ("plan", "start", "begin")):
        return (
            "**Start with the target.** What does a finished answer need to "
            "show, and what is the smallest first step toward it?"
        )
    if any(word in lower for word in ("check", "verify", "right", "correct")):
        return (
            "**Check one assumption.** Which step could you test using a "
            "different method or a simple example?"
        )
    return (
        "**Keep ownership of the next step.** What do you already know, and "
        "what is the first point where your reasoning becomes uncertain?"
    )


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
    _QUESTION_BANK_CACHE: Optional[List[dict]] = None

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
        # Student practice should use the live, context-grounded coach by
        # default. This is separate from optional research LoRA adapters.
        # Verified problem-aware responses are the default interactive path.
        # Local model candidate generation remains opt-in because a 5–12 second
        # wait is unacceptable for ordinary student help-seeking turns.
        self.profile.setdefault("dynamic_problem_coaching", False)
        warm_student_intent_model()
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
        self._last_problem_task_id: str = ""
        self._coaching_trace: deque = deque(maxlen=8)
        self._recent_coaching_responses: deque = deque(maxlen=20)
        self._coaching_turn_index: int = 0
        self._last_coaching_meta: dict = {}
        self._intervention_pipeline = ClosedLoopInterventionPipeline(
            _INTERVENTION_OUTCOMES
        )
        self._latest_intervention_outcome: Optional[dict] = None
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
        "Work through it in your own words before answering."
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
        "algebra": {
            "problem": "Solve for x: 3(x - 4) = 2x + 5",
            "topic": "Algebra",
            "answer": "x = 17",
            "solution_steps": [
                "Distribute 3 across x - 4 to get 3x - 12.",
                "Subtract 2x from both sides to get x - 12 = 5.",
                "Add 12 to both sides to get x = 17.",
            ],
            "key_ideas": ["distribution", "keeping equations balanced", "inverse operations"],
            "common_misconceptions": [
                "Forgetting to multiply both terms inside the parentheses by 3.",
                "Changing a sign when moving terms across the equation.",
                "Combining unlike terms before simplifying both sides.",
            ],
        },
        "act": {
            "problem": "A line passes through (2, 3) and (6, 11). What is its slope?",
            "topic": "ACT Math",
            "answer": "2",
            "solution_steps": [
                "Use slope = change in y divided by change in x.",
                "Compute 11 - 3 = 8 and 6 - 2 = 4.",
                "Divide 8 by 4 to get 2.",
            ],
            "key_ideas": ["slope as rate of change", "coordinate pairs"],
            "common_misconceptions": [
                "Reversing x and y changes.",
                "Subtracting coordinates in a different order for numerator and denominator.",
            ],
        },
        "sat": {
            "problem": "If 2x + 3y = 12 and x = 3, what is y?",
            "topic": "SAT Math",
            "answer": "y = 2",
            "solution_steps": [
                "Substitute x = 3 into 2x + 3y = 12.",
                "Simplify 2(3) + 3y = 12 to 6 + 3y = 12.",
                "Subtract 6, then divide by 3 to get y = 2.",
            ],
            "key_ideas": ["substitution", "inverse operations"],
            "common_misconceptions": [
                "Solving for x again even though x is already given.",
                "Forgetting to multiply 2 by the substituted value.",
            ],
        },
        "geometry": {
            "problem": "A right triangle has legs of length 6 and 8. How long is the hypotenuse?",
            "topic": "Geometry",
            "answer": "10",
            "solution_steps": [
                "Use the Pythagorean theorem: a^2 + b^2 = c^2.",
                "Compute 6^2 + 8^2 = 36 + 64 = 100.",
                "Take the square root of 100 to get 10.",
            ],
            "key_ideas": ["Pythagorean theorem", "square root"],
            "common_misconceptions": [
                "Adding 6 and 8 directly.",
                "Forgetting that the final step is a square root.",
            ],
        },
        "calculus": {
            "problem": "Find the derivative of f(x) = 3x^2 - 5x + 2.",
            "topic": "Calculus",
            "answer": "f'(x) = 6x - 5",
            "solution_steps": [
                "Apply the power rule to 3x^2 to get 6x.",
                "The derivative of -5x is -5.",
                "The derivative of the constant 2 is 0.",
            ],
            "key_ideas": ["power rule", "constant rule"],
            "common_misconceptions": [
                "Leaving the constant in the derivative.",
                "Multiplying the coefficient but forgetting to lower the exponent.",
            ],
        },
        "trigonometry": {
            "problem": "In a right triangle, the angle is 30 degrees and the hypotenuse is 10. How long is the opposite side?",
            "topic": "Trigonometry",
            "answer": "5",
            "solution_steps": [
                "Use sine because opposite and hypotenuse are involved.",
                "sin(30 degrees) = opposite / 10.",
                "Since sin(30 degrees) = 1/2, the opposite side is 5.",
            ],
            "key_ideas": ["sine ratio", "special right triangles"],
            "common_misconceptions": [
                "Using cosine instead of sine.",
                "Putting the hypotenuse over the opposite side.",
            ],
        },
        "statistics": {
            "problem": "A dataset is 4, 8, 15, 16, 23, 42. What is the median?",
            "topic": "Statistics",
            "answer": "15.5",
            "solution_steps": [
                "Notice the data is already ordered.",
                "With six values, average the 3rd and 4th values.",
                "Compute (15 + 16) / 2 = 15.5.",
            ],
            "key_ideas": ["median", "even number of data points"],
            "common_misconceptions": [
                "Choosing one middle number even though there are two.",
                "Calculating the mean instead of the median.",
            ],
        },
        "physics": {
            "problem": "A car accelerates from 0 to 20 m/s in 4 seconds. What is its acceleration?",
            "topic": "Physics",
            "answer": "5 m/s^2",
            "solution_steps": [
                "Use acceleration = change in velocity divided by time.",
                "Compute 20 - 0 = 20 m/s.",
                "Divide 20 by 4 to get 5 m/s^2.",
            ],
            "key_ideas": ["acceleration", "change over time"],
            "common_misconceptions": [
                "Using final velocity as acceleration.",
                "Forgetting to divide by the time interval.",
            ],
        },
        "chemistry": {
            "problem": "How many moles are in 36 grams of water (H2O)?",
            "topic": "Chemistry",
            "answer": "2 moles",
            "solution_steps": [
                "Find the molar mass of water: 18 grams per mole.",
                "Divide mass by molar mass.",
                "36 / 18 = 2 moles.",
            ],
            "key_ideas": ["molar mass", "grams to moles"],
            "common_misconceptions": [
                "Multiplying grams by molar mass instead of dividing.",
                "Using the atomic mass of oxygen alone.",
            ],
        },
        "biology": {
            "problem": "In a cross between two heterozygous (Aa) parents, what fraction of offspring are recessive (aa)?",
            "topic": "Biology",
            "answer": "1/4",
            "solution_steps": [
                "Set up Aa x Aa.",
                "The possible genotypes are AA, Aa, Aa, and aa.",
                "One of the four outcomes is aa, so the fraction is 1/4.",
            ],
            "key_ideas": ["Punnett square", "recessive genotype"],
            "common_misconceptions": [
                "Counting heterozygous Aa as recessive.",
                "Using phenotype ratios when the question asks for aa genotype.",
            ],
        },
        "english": {
            "problem": (
                "Write a short analytical paragraph answering this prompt: How does the author "
                "show that the narrator feels out of place? Use one piece of evidence and explain it."
            ),
            "topic": "English Writing",
            "task_type": "writing",
            "answer": "A strong paragraph makes a clear claim, uses relevant evidence, and explains how the evidence supports the claim.",
            "solution_steps": [
                "Name the claim in one sentence.",
                "Choose one quote or detail that directly supports the claim.",
                "Explain what the evidence shows about the narrator's feeling.",
                "Connect the explanation back to the prompt.",
            ],
            "key_ideas": ["claim", "evidence", "analysis", "connection to prompt"],
            "common_misconceptions": [
                "Dropping in a quote without explaining what it proves.",
                "Summarizing the story instead of analyzing the author's choices.",
                "Writing a broad claim that does not answer the exact prompt.",
            ],
        },
        "writing": {
            "problem": (
                "Revise this thesis so it is specific and arguable: Social media affects teenagers "
                "in many ways."
            ),
            "topic": "English Writing",
            "task_type": "writing",
            "answer": "A strong thesis takes a specific position that could be argued with evidence.",
            "solution_steps": [
                "Identify the vague words in the thesis.",
                "Choose one clear effect to focus on.",
                "Turn that effect into a claim someone could agree or disagree with.",
                "Make sure the thesis previews the reason or direction of the argument.",
            ],
            "key_ideas": ["specific thesis", "arguable claim", "focus"],
            "common_misconceptions": [
                "Keeping the thesis too broad to prove in one essay.",
                "Listing a topic instead of making an argument.",
                "Using dramatic wording without a precise claim.",
            ],
        },
    }

    _PROBLEM_SCHEMA_KEYS = {
        "problem", "topic", "answer", "solution_steps", "key_ideas",
        "common_misconceptions", "task_type", "subject", "id", "coach_hint",
    }

    def _static_fallback_for_topic(self, topic: str) -> dict:
        key = (topic or "").lower().strip()
        for k, v in self._FALLBACK_PROBLEMS.items():
            if k in key:
                return dict(v)
        return dict(self._FALLBACK_PROBLEMS["algebra"])

    @classmethod
    def _load_question_bank(cls) -> List[dict]:
        """Load the committed task bank once, with safe validation."""
        if cls._QUESTION_BANK_CACHE is not None:
            return cls._QUESTION_BANK_CACHE
        path = Path(__file__).parent.parent / "data" / "question_bank.json"
        try:
            raw = json.loads(path.read_text())
            required = {
                "id", "subject", "topic", "problem", "answer",
                "solution_steps", "key_ideas", "common_misconceptions",
            }
            bank = [
                item for item in raw
                if isinstance(item, dict) and required.issubset(item)
            ]
        except Exception as exc:
            print(f"[reasoning] Question bank unavailable: {exc}")
            bank = []
        cls._QUESTION_BANK_CACHE = bank
        return bank

    def _fallback_problem_for_topic(self, topic: str) -> dict:
        """Return a non-repeating, answer-keyed task for a subject or topic."""
        key = (topic or "math").lower().strip()
        bank = self._load_question_bank()

        if key in {"math", "mathematics"}:
            candidates = [item for item in bank if item.get("subject") == "Math"]
        elif any(word in key for word in ("english", "writing", "reading", "language")):
            candidates = [item for item in bank if item.get("subject") == "English"]
        elif "coding" in key or "programming" in key:
            candidates = [item for item in bank if item.get("topic") == "Coding Reasoning"]
        elif "science" in key:
            candidates = [item for item in bank if item.get("topic") == "Science Reasoning"]
        else:
            candidates = [
                item for item in bank
                if key in str(item.get("topic", "")).lower()
                or key in str(item.get("subject", "")).lower()
            ]

        last_id = getattr(self, "_last_problem_task_id", "")
        available = [item for item in candidates if item.get("id") != last_id]
        if available:
            chosen = dict(random.choice(available))
            self._last_problem_task_id = str(chosen.get("id", ""))
            return chosen
        return self._static_fallback_for_topic(topic)

    def _normalise_problem_ctx(self, raw: dict, topic: str, difficulty: str) -> dict:
        base = self._static_fallback_for_topic(topic)
        if not isinstance(raw, dict):
            raw = {}
        ctx = {}
        for key in self._PROBLEM_SCHEMA_KEYS:
            ctx[key] = raw.get(key) or base.get(key)
        for list_key in ("solution_steps", "key_ideas", "common_misconceptions"):
            if not isinstance(ctx.get(list_key), list) or not ctx[list_key]:
                ctx[list_key] = base[list_key]
            ctx[list_key] = [str(x).strip() for x in ctx[list_key] if str(x).strip()]
        ctx["problem"] = str(ctx.get("problem") or base["problem"]).strip()
        ctx["topic"] = str(ctx.get("topic") or topic or base["topic"]).strip()
        ctx["answer"] = str(ctx.get("answer") or base["answer"]).strip()
        ctx["task_type"] = str(ctx.get("task_type") or base.get("task_type") or "answer_key").strip()
        ctx["subject"] = str(ctx.get("subject") or base.get("subject") or "Math").strip()
        ctx["difficulty"] = str(raw.get("difficulty") or difficulty)
        ctx["task_id"] = str(ctx.pop("id", "") or "")
        ctx["problem_id"] = uuid.uuid4().hex[:12]
        return ctx

    def _is_writing_task(self, ctx: Optional[dict] = None) -> bool:
        ctx = ctx or getattr(self, "_current_problem_ctx", {}) or {}
        topic = str(ctx.get("topic", "")).lower()
        problem = str(ctx.get("problem", "")).lower()
        return (
            ctx.get("task_type") == "writing"
            or any(w in topic for w in ("english", "writing", "essay", "reading"))
            or any(w in problem for w in ("paragraph", "thesis", "quote", "author", "essay"))
        )

    def _problem_context_block(self, ctx: Optional[dict] = None) -> str:
        ctx = ctx or getattr(self, "_current_problem_ctx", {}) or {}
        if not ctx:
            return "No active practice problem."
        steps = "\n".join(f"- {s}" for s in ctx.get("solution_steps", [])[:5])
        ideas = ", ".join(ctx.get("key_ideas", [])[:5])
        mistakes = "\n".join(f"- {m}" for m in ctx.get("common_misconceptions", [])[:5])
        return (
            f"Problem: {ctx.get('problem', '')}\n"
            f"{'Rubric target' if self._is_writing_task(ctx) else 'Correct answer'}: {ctx.get('answer', '')}\n"
            f"Key ideas: {ideas}\n"
            f"Expected solution steps:\n{steps}\n"
            f"Common mistakes:\n{mistakes}"
        )

    @staticmethod
    def _student_quote_anchor(student_input: str, max_words: int = 8) -> str:
        """Return a short, safe fragment that grounds feedback in this turn."""
        cleaned = re.sub(
            r"^(?:okay|ok|so|well|maybe|i think|i guess|wait,?\s+i see|now i see)\b[\s,.:;-]*",
            "",
            (student_input or "").strip(),
            flags=re.IGNORECASE,
        )
        words = cleaned.split()
        if not words:
            return "what you wrote"
        selected = []
        for word in words:
            if selected and word.lower().strip(",;:") in {"because", "since", "then"}:
                break
            selected.append(word)
            if len(selected) >= max_words:
                break
        fragment = " ".join(selected)
        return fragment.rstrip(".,;:")

    def _learner_context_block(self, student_input: str, state: str) -> str:
        """Build the student-model side of ARIA's dual-grounded response."""
        profile = getattr(self, "profile", {}) or {}
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        topic = str(ctx.get("topic", "")).lower()
        subject = str(ctx.get("subject", ""))

        misconceptions = profile.get("misconceptions", {}) or {}
        known_misconceptions = []
        if isinstance(misconceptions, dict):
            for key, values in misconceptions.items():
                if str(key).lower() in topic or topic in str(key).lower():
                    if isinstance(values, list):
                        known_misconceptions.extend(str(v) for v in values[:3])

        preferred_styles = profile.get("preferred_styles", {}) or {}
        preferred_style = profile.get("learning_style", "mixed")
        if isinstance(preferred_styles, dict):
            for key, value in preferred_styles.items():
                if str(key).lower() in topic or topic in str(key).lower():
                    preferred_style = value
                    break

        prior_trace = list(getattr(self, "_coaching_trace", []))[-4:]
        trace_text = "\n".join(
            f"- Student: {turn['student']}\n  ARIA: {turn['aria']}"
            for turn in prior_trace
        ) or "- No earlier turn on this problem."

        past_learning = []
        vs = getattr(self, "vs", None)
        if vs is not None and hasattr(vs, "retrieve_context"):
            try:
                for item in vs.retrieve_context(student_input, n_results=3):
                    text = str(item.get("text", "")).strip()
                    if text:
                        past_learning.append(text[:240])
            except Exception:
                pass

        graph_summary = ""
        lg = getattr(self, "lg", None)
        if lg is not None and hasattr(lg, "get_topic_info"):
            try:
                info = lg.get_topic_info(ctx.get("topic", ""))
                if info:
                    graph_summary = (
                        f"confidence={info.get('confidence', 'unknown')}; "
                        f"struggles={info.get('struggle_count', 0)}"
                    )
            except Exception:
                pass

        return (
            f"Student: {profile.get('name', 'this student')}\n"
            f"Grade: {profile.get('grade', 'unknown')}\n"
            f"Current subject: {subject or topic or 'unknown'}\n"
            f"Detected thinking state: {state}\n"
            f"Preferred explanation style: {preferred_style}\n"
            f"Response length preference: {profile.get('response_length', profile.get('answer_style', 'brief'))}\n"
            f"Attention span: {profile.get('attention_span_minutes', profile.get('attention_dropoff_minutes', 'unknown'))} minutes\n"
            f"Current goals: {profile.get('goals', [])}\n"
            f"What has helped: {profile.get('what_helped', '')}\n"
            f"What has failed: {profile.get('what_failed', '')}\n"
            f"Known misconceptions for this topic: {known_misconceptions}\n"
            f"Learning graph summary: {graph_summary or 'none yet'}\n"
            f"Relevant past learning: {past_learning or ['none yet']}\n"
            f"Trace on this exact problem:\n{trace_text}"
        )

    @staticmethod
    def _looks_generic_coaching(response: str) -> bool:
        lower = (response or "").lower().strip()
        generic_phrases = (
            "what is the smallest next step",
            "what do you know so far",
            "tell me what you are thinking",
            "try breaking it down",
            "check your work",
        )
        return not lower or any(phrase in lower for phrase in generic_phrases)

    def _get_intervention_pipeline(self) -> ClosedLoopInterventionPipeline:
        pipeline = getattr(self, "_intervention_pipeline", None)
        if pipeline is None:
            pipeline = ClosedLoopInterventionPipeline(_INTERVENTION_OUTCOMES)
            self._intervention_pipeline = pipeline
        return pipeline

    def _matched_problem_misconception(self, student_input: str) -> str:
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        student_terms = set(re.findall(r"[a-z0-9]+", student_input.lower()))
        best = ""
        best_overlap = 0
        for misconception in ctx.get("common_misconceptions", []):
            terms = {
                term for term in re.findall(r"[a-z0-9]+", str(misconception).lower())
                if len(term) >= 4
            }
            overlap = len(student_terms & terms)
            if overlap > best_overlap:
                best = str(misconception)
                best_overlap = overlap
        return best

    def _expected_step_conflict(self, student_input: str) -> Optional[dict]:
        """Detect a narrow, verifiable conflict with the keyed first step.

        This never attempts open-ended diagnosis. It only fires when the
        student's stated operation is the explicit inverse of the operation in
        the answer-keyed first step and both mention the same term.
        """
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        first_step = str((ctx.get("solution_steps") or [""])[0]).lower()
        student = (student_input or "").lower()
        opposites = {
            "add": "subtract",
            "subtract": "add",
            "multiply": "divide",
            "divide": "multiply",
        }
        expected = next(
            (operation for operation in opposites if first_step.startswith(operation)),
            "",
        )
        if not expected:
            return None
        observed = opposites[expected]
        if not re.search(rf"\b{observed}\b", student):
            return None
        target_match = re.search(
            rf"^{expected}\s+(.+?)(?:\s+from|\s+to|\s+by|\s+on)\b",
            first_step,
        )
        target = target_match.group(1).strip() if target_match else ""
        if target:
            target_tokens = {
                token for token in re.findall(r"[a-z0-9]+", target)
                if len(token) >= 1
            }
            student_tokens = set(re.findall(r"[a-z0-9]+", student))
            if not target_tokens.issubset(student_tokens):
                return None
        return {
            "observed_operation": observed,
            "expected_operation": expected,
            "target": target,
            "evidence": (
                f"the proposed {observed} operation conflicts with the keyed "
                f"inverse-operation step for {target or 'the same term'}"
            ),
        }

    def _matched_solution_progress(self, student_input: str) -> Optional[dict]:
        """Return the furthest keyed intermediate result visible in the turn."""
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        normalized_student = re.sub(r"\s+", "", (student_input or "").lower())
        matched = None
        for index, step in enumerate(ctx.get("solution_steps", [])):
            result_match = re.search(
                r"\bto get\s+(.+?)(?:\.$|;|$)",
                str(step),
                flags=re.IGNORECASE,
            )
            if not result_match:
                continue
            result = result_match.group(1).strip()
            if re.sub(r"\s+", "", result.lower()) in normalized_student:
                matched = {
                    "step_index": index,
                    "result": result,
                    "step": str(step),
                }
        return matched

    @staticmethod
    def _response_passes_grounding(response: str, student_input: str) -> bool:
        """Reject polished-sounding drafts that are not tied to this turn."""
        response = (response or "").strip()
        lower = response.lower()
        student_lower = (student_input or "").lower()
        if not response or len(response) > 520:
            return False
        if response.count("?") > 1:
            return False
        if len([line for line in response.splitlines() if line.strip()]) > 3:
            return False
        if any(phrase in lower for phrase in (
            "good start",
            "great job",
            "you've successfully",
            "you have successfully",
            "you already solved",
            "reasoning move:",
        )):
            return False

        # The coach is required to visibly anchor itself in the current turn.
        has_quote = any(mark in response for mark in ('"', "“", "”"))
        student_words = [
            word for word in re.findall(r"[a-z0-9.]+", student_lower)
            if len(word) >= 3
        ]
        meaningful_overlap = sum(
            1 for word in set(student_words)
            if word in re.findall(r"[a-z0-9.]+", lower)
        )
        return has_quote and meaningful_overlap >= 2

    def _remember_coaching_response(
        self,
        student_input: str,
        state: str,
        response: str,
        fallback_response: str,
        force_anchor: bool = False,
        repeat_blocked_override: bool = False,
        selection_meta: Optional[dict] = None,
    ) -> str:
        """Guarantee a turn-grounded response and prevent exact reuse."""
        recent = list(getattr(self, "_recent_coaching_responses", []))
        anchor = self._student_quote_anchor(student_input)
        final = (response or fallback_response).strip()
        repeated = final in recent or repeat_blocked_override
        generic = self._looks_generic_coaching(final)

        if repeated or generic or force_anchor:
            lead_ins = (
                "You wrote",
                "I am using this part of your reasoning",
                "The key phrase in your attempt is",
                "Your current move is",
            )
            turn_index = getattr(self, "_coaching_turn_index", 0)
            lead = lead_ins[turn_index % len(lead_ins)]
            final = f'{lead} “{anchor}” {fallback_response}'.strip()

        # A repeated student message can still produce a repeated fallback.
        # Reference the evolving reasoning path without adding meaningless IDs.
        if final in recent:
            prior_count = sum(
                1 for item in getattr(self, "_coaching_trace", [])
                if item.get("student") == student_input.strip()
            )
            final = (
                f'You returned to “{anchor}” after {prior_count + 1} pass'
                f'{"es" if prior_count else ""}. {fallback_response}'
            )

        if not hasattr(self, "_recent_coaching_responses"):
            self._recent_coaching_responses = deque(maxlen=20)
        if not hasattr(self, "_coaching_trace"):
            self._coaching_trace = deque(maxlen=8)
        self._recent_coaching_responses.append(final)
        self._coaching_trace.append({
            "student": student_input.strip(),
            "state": state,
            "aria": final,
        })
        self._coaching_turn_index = getattr(self, "_coaching_turn_index", 0) + 1
        self._last_coaching_meta = {
            "problem_grounded": True,
            "student_words_grounded": True,
            "profile_grounded": bool(getattr(self, "profile", {})),
            "history_turns_used": max(0, len(self._coaching_trace) - 1),
            "repeat_blocked": repeated,
            "generic_blocked": generic,
            "model_draft_rejected": force_anchor,
            **(selection_meta or {}),
        }
        return final

    def _problem_aware_coaching_response(self, student_input: str, state: str, base_question: str) -> str:
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        if not ctx.get("answer"):
            return self._remember_coaching_response(
                student_input, state, base_question, base_question
            )

        answer = str(ctx.get("answer", "")).lower()
        text = student_input.lower()
        understanding = (
            getattr(self, "_current_student_understanding", None)
            if getattr(self, "_current_student_understanding_text", None)
            == student_input
            else None
        )
        if understanding is None:
            understanding = understand_student_turn(
                student_input,
                problem=str(ctx.get("problem", "")),
                recent_turns=getattr(self, "_coaching_trace", []),
                allow_deep=bool(
                    getattr(self, "profile", {}).get(
                        "deep_language_understanding", False
                    )
                ),
            )
        intent_label = understanding.intent
        orientation_response = ""
        if intent_label in {"HELP_REQUEST", "ATTEMPT_META", "FRUSTRATION"}:
            problem_text = str(ctx.get("problem", ""))
            grouped = re.search(
                r"([+-]?\d+(?:\.\d+)?)\s*\(([^)]+)\)",
                problem_text,
            )
            if grouped:
                expression = grouped.group(0)
                coefficient = grouped.group(1)
                orientation_response = (
                    f"Start with the grouped part {expression}. "
                    f"Before moving anything, which terms inside the parentheses must "
                    f"{coefficient} multiply?"
                )
            elif self._is_writing_task(ctx):
                orientation_response = str(
                    ctx.get("coach_hint")
                    or (
                        f"Start with one decision: "
                        f"{(ctx.get('solution_steps') or ['Identify the main claim.'])[0]} "
                        "What rough version would you try?"
                    )
                )
            else:
                first_step = str(
                    (ctx.get("solution_steps") or ["Identify the first operation."])[0]
                )
                target_match = re.search(
                    r"^(?:add|subtract|multiply|divide)\s+(.+?)"
                    r"(?:\s+from|\s+to|\s+by|\s+on)\b",
                    first_step,
                    flags=re.IGNORECASE,
                )
                target = target_match.group(1) if target_match else "one term"
                orientation_response = (
                    f"Start by locating {target} in the original problem. "
                    f"Which inverse operation would remove it while keeping the two sides equivalent?"
                )
            if intent_label == "ATTEMPT_META":
                orientation_response = (
                    "You have not made a reasoning attempt yet. "
                    + orientation_response
                )
            elif intent_label == "FRUSTRATION":
                orientation_response = (
                    "Let’s make this smaller. " + orientation_response
                )

        if orientation_response:
            fallback_response = orientation_response
        elif intent_label == "CLARIFICATION_REQUEST":
            term_match = re.search(
                r"(?:what does|what is|define)\s+(?:the\s+)?"
                r"([a-z][a-z -]{1,30}?)(?:\s+mean|\?|$)",
                text,
            )
            term = term_match.group(1).strip() if term_match else ""
            definitions = {
                "coefficient": (
                    "A coefficient is the number multiplying a variable. "
                    "Which number is attached directly to the variable here?"
                ),
                "inverse operation": (
                    "An inverse operation undoes another operation, like subtraction "
                    "undoing addition. Which operation is attached to the variable here?"
                ),
                "distribute": (
                    "To distribute means multiplying the outside factor by every term "
                    "inside the parentheses. Which inside term would you multiply first?"
                ),
                "theme": (
                    "A theme is a broader idea a text develops, not a one-word topic. "
                    "What repeated message do the character’s choices suggest?"
                ),
            }
            fallback_response = definitions.get(
                term,
                (
                    "I can explain the exact part you mean without solving it for you. "
                    "Which word, symbol, or previous sentence should I unpack?"
                ),
            )
        elif intent_label == "CONFIRMATION_REQUEST" and not (
            self._matched_solution_progress(student_input)
            or (answer and answer in text)
        ):
            fallback_response = (
                "I can check your reasoning, but I need the step itself. "
                "What equation, sentence, or choice are you asking me to verify?"
            )
        elif intent_label == "CONTROL_REQUEST":
            fallback_response = (
                "Use “Try another problem” to switch tasks. Before you leave this one, "
                "do you want to save your current attempt?"
            )
        elif intent_label in {"SOCIAL", "OTHER"} and not understanding.contains_reasoning:
            fallback_response = (
                "I do not have enough reasoning to choose a useful intervention yet. "
                "Do you want a starting hint, a term explained, or a step checked?"
            )
        elif self._is_writing_task(ctx):
            normalized_answer = re.sub(r"[^a-z0-9]+", " ", answer).strip()
            normalized_text = re.sub(r"[^a-z0-9]+", " ", text).strip()
            exact_revision = (
                str(ctx.get("topic", "")).lower() == "language and revision"
                and normalized_answer
                and normalized_answer in normalized_text
            )
            if exact_revision:
                fallback_response = (
                    "That revision matches the target. What exact grammar or clarity problem did you fix?"
                )
            elif any(w in text for w in ("quote", "evidence", "citation")) and not any(
                w in text for w in ("shows", "because", "suggests", "reveals", "proves")
            ):
                fallback_response = (
                    "You have evidence, but the missing move is analysis. "
                    "What does that quote reveal about the narrator feeling out of place?"
                )
            elif any(w in text for w in ("thesis", "claim")) and any(
                w in text for w in ("many ways", "things", "stuff", "good", "bad")
            ):
                fallback_response = (
                    "The claim is still broad. What is one specific effect or idea you can argue instead?"
                )
            elif any(w in text for w in ("i don't know", "dont know", "stuck", "idk")):
                first_step = (ctx.get("solution_steps") or ["Name the claim first."])[0]
                fallback_response = f"Start with the thinking move, not perfect wording: {first_step} What is your rough version?"
            elif ctx.get("coach_hint"):
                fallback_response = str(ctx["coach_hint"])
            else:
                first_step = (ctx.get("solution_steps") or ["Connect your evidence to the prompt."])[0]
                fallback_response = f"Use the rubric target here: {first_step} Which sentence in your draft is doing that job?"
        elif answer and answer in text:
            fallback_response = (
                "That matches the target answer. Can you name the step that made the equation simpler?"
            )
        elif progress := self._matched_solution_progress(student_input):
            result = progress["result"]
            compact = re.sub(r"\s+", "", result.lower())
            coefficient_match = re.search(r"([+-]?\d+(?:\.\d+)?)x=", compact)
            if coefficient_match:
                coefficient = coefficient_match.group(1)
                fallback_response = (
                    f"You reached “{result}.” What inverse operation undoes multiplying x "
                    f"by {coefficient} while keeping both sides balanced?"
                )
            else:
                next_index = progress["step_index"] + 1
                next_steps = list(ctx.get("solution_steps", []))
                next_step = (
                    next_steps[next_index]
                    if next_index < len(next_steps)
                    else "Check the result against the original problem."
                )
                fallback_response = (
                    f"You reached “{result}.” Without carrying it out yet, "
                    f"what operation does this next goal require: {next_step}"
                )
        else:
            fallback_response = ""
            problem = str(ctx.get("problem", "")).lower()
            step_conflict = self._expected_step_conflict(student_input)
            if step_conflict:
                target = step_conflict.get("target") or "that term"
                expected = step_conflict["expected_operation"]
                if {expected, step_conflict["observed_operation"]} == {
                    "add", "subtract"
                }:
                    fallback_response = (
                        f"You want the variable terms together, and the term to cancel is {target}. "
                        f"Which inverse operation cancels {target} while keeping both sides balanced?"
                    )
                else:
                    fallback_response = (
                        f"Your move uses {step_conflict['observed_operation']} on {target}. "
                        f"Which inverse operation would undo {target} while preserving equivalence?"
                    )
            if "3(x - 4)" in problem and ("3x - 4" in text or "subtract 4" in text):
                fallback_response = (
                    "The part to check is distribution: the 3 applies to both x and -4. "
                    "What does 3 times -4 become before you move any terms?"
                )
            grouped = re.search(
                r"([+-]?\d+(?:\.\d+)?)\s*\(([^)]+)\)",
                str(ctx.get("problem", "")),
            )
            if (
                not fallback_response
                and grouped
                and any(word in text for word in ("subtract", "add ", "divide", "move"))
                and "distribut" not in text
            ):
                expression = grouped.group(0)
                coefficient = grouped.group(1)
                fallback_response = (
                    f"The grouped expression to check is {expression}: "
                    f"{coefficient} sits outside the parentheses. "
                    "Which terms inside must it multiply before anything is moved?"
                )
            if not fallback_response:
                for mistake in ctx.get("common_misconceptions", [])[:2]:
                    terms = [w for w in re.findall(r"[a-z0-9]+", mistake.lower()) if len(w) > 4]
                    if any(w in text for w in terms):
                        fallback_response = (
                            f"Check this part carefully: {mistake} "
                            "What is the smallest line you can rewrite before moving on?"
                        )
                        break
            if not fallback_response:
                first_step = (ctx.get("solution_steps") or ["Identify the first operation."])[0]
                action = re.split(r"\s+to get\b|:\s*", first_step, maxsplit=1)[0].rstrip(".")
                fallback_response = (
                    f"The operation to check before your move is: {action}. "
                    "Which part of the original expression requires that operation first?"
                )

        pipeline = self._get_intervention_pipeline()
        recent_responses = list(getattr(self, "_recent_coaching_responses", []))[-8:]
        if intent_label in {"HELP_REQUEST", "ATTEMPT_META", "FRUSTRATION"}:
            # Coherence beats novelty for dialogue-management turns. Repeating
            # the same problem-specific starting point is better than inventing
            # a less relevant alternative.
            recent_responses = []
        anchor = self._student_quote_anchor(student_input)
        first_step = (ctx.get("solution_steps") or ["Identify the first operation."])[0]
        misconception = self._matched_problem_misconception(student_input)
        if not misconception and any(
            marker in fallback_response.lower()
            for marker in (
                "part to check",
                "grouped expression to check",
                "missing move",
                "claim is still broad",
                "operation to check",
            )
        ):
            misconception = f"Verified issue: {fallback_response}"
        profile = getattr(self, "profile", {}) or {}
        latest_outcome = getattr(self, "_latest_intervention_outcome", None)
        prior_outcome = "none"
        if latest_outcome:
            if latest_outcome.get("recovered"):
                prior_outcome = "the previous intervention led to recovery"
            elif latest_outcome.get("self_correction"):
                prior_outcome = "the student self-corrected after the previous intervention"
            elif latest_outcome.get("remained_stuck"):
                prior_outcome = "the student remained stuck after the previous intervention"
            else:
                prior_outcome = "the previous intervention did not show a clear outcome"

        student_name = str(profile.get("name", "student"))
        key_ideas = list(ctx.get("key_ideas") or [])
        learner_state = pipeline.learner_state(
            student=student_name,
            topic=str(ctx.get("topic", "")),
            key_ideas=key_ideas,
            misconception=misconception,
        )
        signature = pipeline.build_signature(
            student=student_name,
            task_id=str(ctx.get("task_id") or ctx.get("problem_id", "")),
            topic=str(ctx.get("topic", "")),
            problem_step=str(first_step),
            student_anchor=anchor,
            misconception=misconception,
            state=state,
            style=str(profile.get("learning_style", "mixed")),
            prior_outcome=prior_outcome,
            key_ideas=key_ideas,
            mastery_mean=float(learner_state["mastery_mean"]),
            mastery_band=str(learner_state["mastery_band"]),
        )
        safe_candidates = pipeline.safe_candidates(
            anchor=anchor,
            fallback=fallback_response,
            key_idea=str((ctx.get("key_ideas") or ["the first required operation"])[0]),
            state=state,
            has_prior_turn=bool(getattr(self, "_coaching_trace", [])),
            repeat_anchor=intent_label not in {
                "HELP_REQUEST", "ATTEMPT_META", "FRUSTRATION"
            },
        )
        model_candidates = []

        system = (
            "You generate candidate interventions for ARIA's closed-loop student coach. "
            "Return valid JSON only in this form: "
            '{"candidates":[{"text":"...","strategy":"..."},{"text":"...","strategy":"..."},{"text":"...","strategy":"..."}]}. '
            "Use three different strategies chosen from error_localization, smallest_step, "
            "contrast_case, self_explanation, retrieval_cue, verification, planning, reflection.\n"
            "Every candidate must:\n"
            "- quote the supplied student anchor exactly;\n"
            "- respond only to reasoning literally present in the current message;\n"
            "- use only facts in the problem model or verified fallback;\n"
            "- avoid claiming the student completed an unstated step;\n"
            "- contain exactly one targeted question;\n"
            "- avoid the final answer unless the student already supplied it;\n"
            "- stay under 55 words and never use generic praise;\n"
            "- differ meaningfully in coaching strategy, not merely wording."
        )
        user = (
            f"PROBLEM MODEL\n{self._problem_context_block(ctx)}\n\n"
            f"STUDENT MODEL\n{self._learner_context_block(student_input, state)}\n\n"
            f"CURRENT STUDENT MESSAGE\n{student_input}\n\n"
            f"EXACT ANCHOR TO QUOTE\n{anchor}\n\n"
            f"INTERVENTION SIGNATURE\n{json.dumps(signature, ensure_ascii=False)}\n\n"
            f"VERIFIED FALLBACK FACTS\n{fallback_response}\n\n"
            f"Recent responses that must be semantically different: {recent_responses}\n"
            f"Default metacognitive move: {base_question}"
        )
        if getattr(self, "profile", {}).get("dynamic_problem_coaching", False):
            try:
                result = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options={"temperature": 0.6, "num_predict": 240},
                    keep_alive=3600,
                )
                model_candidates = pipeline.parse_model_candidates(
                    result.message.content.strip()
                )
            except Exception:
                model_candidates = []

        all_candidates = model_candidates + safe_candidates
        response_pref = str(
            profile.get("response_length", profile.get("answer_style", "brief"))
        ).lower()
        preferred_words = {
            "short": 38,
            "brief": 38,
            "medium": 55,
            "detailed": 75,
        }.get(response_pref, 55)
        selected, selection_meta = pipeline.select(
            candidates=all_candidates,
            student_input=student_input,
            recent_responses=recent_responses,
            key_ideas=list(ctx.get("key_ideas") or []),
            correct_answer=str(ctx.get("answer", "")),
            validator=lambda response, current_input: (
                (
                    intent_label in {
                        "HELP_REQUEST", "ATTEMPT_META", "FRUSTRATION"
                    }
                    or self._response_passes_grounding(response, current_input)
                )
                and not self._looks_generic_coaching(response)
            ),
            signature=signature,
            state=state,
            preferred_words=preferred_words,
        )
        selection_meta.update({
            "problem_grounded": True,
            "student_words_grounded": True,
            "profile_grounded": bool(profile),
            "history_turns_used": len(getattr(self, "_coaching_trace", [])),
            "model_candidates_generated": len(model_candidates),
            "verified_candidates_generated": len(safe_candidates),
            "prior_outcome": prior_outcome,
            "learner_model": learner_state,
            "state_is_hypothesis": True,
            "student_intent": intent_label,
            "intent_confidence": round(understanding.confidence, 4),
            "intent_model": understanding.source,
            "semantic_understanding": understanding.to_dict(),
        })

        if selected:
            return self._remember_coaching_response(
                student_input,
                state,
                selected,
                fallback_response,
                selection_meta=selection_meta,
            )
        return self._remember_coaching_response(
            student_input,
            state,
            fallback_response,
            fallback_response,
            force_anchor=True,
            selection_meta=selection_meta,
        )

    def generate_think_aloud_problem(self, topic_override: Optional[str] = None) -> dict:
        """Generate one practice problem tuned to the student's subjects/goals.

        Prioritises a concept that is due for SRS review. Returns
        {"problem": str, "topic": str}. Falls back to a template bank if the
        LLM is unavailable so this never hard-fails.
        """
        subjects = self.profile.get("subjects", []) or ["general learning"]
        goals = self.profile.get("goals", [])
        topic = topic_override or subjects[0]

        # Prefer an SRS-due concept if one exists.
        due_concept = ""
        try:
            from memory.srs import get_due_cards
            due = get_due_cards()
            if due and not topic_override:
                due_concept = due[0].get("concept", "")
                topic = due[0].get("topic", topic)
        except Exception:
            pass

        focus = due_concept or topic
        grade = self.profile.get("grade", "")
        goal_str = f" (their goal: {goals[0]})" if goals else ""
        difficulty = self.profile.get("default_difficulty", "medium")
        problem_data = self._fallback_problem_for_topic(topic)
        if not self.profile.get("dynamic_problem_generation", False):
            if self._meta_tracker is not None:
                self._close_episode(recovered=False)
            self._calib_pending = None
            getattr(self, "_coaching_trace", deque()).clear()
            self._get_intervention_pipeline().pending = None
            self._latest_intervention_outcome = None
            ctx = self._normalise_problem_ctx(problem_data, topic, difficulty)
            self._current_problem_ctx = ctx
            return ctx

        prompt = (
            f"Generate ONE short practice problem for a {grade or 'high-school'} student "
            f"studying {topic}{goal_str}, focused on '{focus}'. "
            "Return valid JSON only with these keys: problem, topic, answer, "
            "solution_steps, key_ideas, common_misconceptions. The problem must be "
            "concrete and solvable. solution_steps, key_ideas, and "
            "common_misconceptions must be arrays of short strings."
        )
        try:
            result = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                        "You write answer-keyed practice problems. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.45, "num_predict": 260},
                keep_alive=3600,
            )
            raw = result.message.content.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
            generated_problem = json.loads(raw)
            if not isinstance(generated_problem, dict) or not generated_problem.get("problem"):
                raise ValueError("empty problem")
            problem_data = generated_problem
        except Exception:
            problem_data = self._fallback_problem_for_topic(topic)

        # A fresh problem starts a new calibration cycle: any prior confidence
        # that was never resolved is dropped, and the negative-state episode is
        # closed out so timing episodes don't bleed across problems.
        if self._meta_tracker is not None:
            self._close_episode(recovered=False)
        self._calib_pending = None
        getattr(self, "_coaching_trace", deque()).clear()
        self._get_intervention_pipeline().pending = None
        self._latest_intervention_outcome = None
        ctx = self._normalise_problem_ctx(problem_data, topic, difficulty)
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

    def _quick_thinking_state(self, student_input: str) -> dict:
        text = (student_input or "").strip()
        lower = text.lower()
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        understanding = understand_student_turn(
            text,
            problem=str(ctx.get("problem", "")),
            recent_turns=getattr(self, "_coaching_trace", []),
            allow_deep=bool(
                getattr(self, "profile", {}).get(
                    "deep_language_understanding", False
                )
            ),
        )
        self._current_student_understanding = understanding
        self._current_student_understanding_text = student_input
        intent_label = understanding.intent
        problem = str(
            (getattr(self, "_current_problem_ctx", {}) or {}).get("problem", "")
        )
        grouped_expression = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*\(([^)]+)\)",
            problem,
        )
        skipped_grouping = (
            grouped_expression is not None
            and any(word in lower for word in ("subtract", "add ", "divide", "move"))
            and "distribut" not in lower
        )
        self_correction = any(
            phrase in lower
            for phrase in (
                "wait, i see",
                "wait i see",
                "oh, i see",
                "oh i see",
                "now i see",
                "actually",
                "that means",
            )
        )
        step_conflict = self._expected_step_conflict(student_input)
        if intent_label == "HELP_REQUEST":
            state = "STUCK"
            evidence = "the student asked for help or a starting point"
        elif intent_label == "ATTEMPT_META":
            state = "STUCK"
            evidence = "the student described beginning, not a solution step"
        elif intent_label == "FRUSTRATION":
            state = "FRUSTRATED"
            evidence = "the student expressed frustration with the task"
        elif intent_label == "UNCERTAINTY":
            state = "CONFUSED"
            evidence = "the student proposed an idea while expressing uncertainty"
        elif intent_label == "SELF_CORRECTION":
            state = "INSIGHT"
            evidence = "the student revised part of their reasoning"
        elif intent_label in {
            "SOCIAL", "OTHER", "CONTROL_REQUEST", "CLARIFICATION_REQUEST",
            "CONFIRMATION_REQUEST", "SHORT_ANSWER",
        } and not understanding.contains_reasoning:
            state = "UNKNOWN"
            evidence = "there is not enough problem reasoning in this turn to estimate a thinking pattern"
        elif any(w in lower for w in ("stuck", "lost", "idk", "don't know", "dont know", "confused")):
            state = "STUCK"
            evidence = "the student said they were stuck or unsure"
        elif self_correction:
            state = "INSIGHT"
            evidence = "the student revised or connected their reasoning"
        elif skipped_grouping:
            state = "CONFUSED"
            evidence = "the proposed move skips a grouped expression"
        elif step_conflict:
            state = "CONFUSED"
            evidence = step_conflict["evidence"]
        elif any(w in lower for w in ("wait", "not sure", "can't tell", "cannot tell", "wrong")):
            state = "CONFUSED"
            evidence = "the student flagged uncertainty in the next step"
        elif any(w in lower for w in ("first", "plan", "start", "then", "next")):
            state = "PLANNING"
            evidence = "the student described a planned step"
        elif any(w in lower for w in ("makes sense", "got it")):
            state = "INSIGHT"
            evidence = "the student noticed a connection"
        else:
            state = "FLOW"
            evidence = "the student is continuing their reasoning"
        return {
            "state": state,
            "confidence": 0.7,
            "evidence": evidence,
            "method": "fast_problem_loop",
            "intent": intent_label,
            "intent_confidence": understanding.confidence,
            "understanding_source": understanding.source,
            "flags": {
                "planning_detected": state == "PLANNING",
                "self_correction": "wait" in lower,
                "insight_moment": state == "INSIGHT",
                "gave_up": state == "STUCK",
            },
        }

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
        if not getattr(self, "profile", {}).get("full_research_metacognition", False):
            analysis = self._quick_thinking_state(student_input)
            progress = self._matched_solution_progress(student_input)
            ctx = getattr(self, "_current_problem_ctx", {}) or {}
            answer = str(ctx.get("answer", "")).lower().strip()
            normalized_input = re.sub(
                r"\s+", "", (student_input or "").lower()
            )
            answer_observed = bool(
                answer
                and re.sub(r"\s+", "", answer) in normalized_input
            )
            step_conflict = self._expected_step_conflict(student_input)
            self._latest_intervention_outcome = (
                self._get_intervention_pipeline().observe_next_turn(
                    state=analysis["state"],
                    student_input=student_input,
                    state_confidence=float(analysis.get("confidence", 0.0)),
                    correct=True if (progress or answer_observed) else None,
                    misconception_persisted=(
                        True if step_conflict
                        else False if (progress or answer_observed)
                        else None
                    ),
                )
            )
            question = self._problem_aware_coaching_response(
                student_input,
                analysis["state"],
                "What is the smallest next step you can check against the problem?",
            )
            return {
                "state": analysis["state"],
                "confidence": analysis["confidence"],
                "evidence": analysis["evidence"],
                "question": question,
                "indicator": STATE_INDICATOR.get(analysis["state"], "🟡"),
                "escalated": False,
                "escalation_kind": None,
                "intervention_state": analysis["state"],
                "intervened": True,
                "acknowledged": False,
                "self_initiated_metacognition": False,
                "metacognitive_type": "",
                "prompted_by_aria": bool(self._meta_last_prompt),
                "turns_in_state": 1,
                "recommended_wait": 0,
                "flags": analysis["flags"],
                "method": analysis["method"],
                "intent": analysis.get("intent", "OTHER"),
                "intent_confidence": analysis.get("intent_confidence", 0.0),
                "personalization": dict(getattr(self, "_last_coaching_meta", {})),
            }

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
        progress = self._matched_solution_progress(student_input)
        ctx = getattr(self, "_current_problem_ctx", {}) or {}
        answer = str(ctx.get("answer", "")).lower().strip()
        answer_observed = bool(
            answer
            and re.sub(r"\s+", "", answer)
            in re.sub(r"\s+", "", student_input.lower())
        )
        step_conflict = self._expected_step_conflict(student_input)
        self._latest_intervention_outcome = (
            self._get_intervention_pipeline().observe_next_turn(
                state=state,
                student_input=student_input,
                state_confidence=float(analysis.get("confidence", 0.0)),
                correct=True if (progress or answer_observed) else None,
                misconception_persisted=(
                    True if step_conflict
                    else False if (progress or answer_observed)
                    else None
                ),
            )
        )
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

        question = self._problem_aware_coaching_response(student_input, state, question)
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
            "personalization": dict(getattr(self, "_last_coaching_meta", {})),
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
            try:
                result = ollama.chat(
                    model=self.model,
                    messages=msgs,
                    options={"temperature": 0.7, "num_predict": 256},
                    keep_alive=3600,
                )
                response = result.message.content.strip()
            except Exception:
                response = offline_coaching_response(state["user_input"])

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

        # Stream tokens from Ollama. If it is stopped or still waking up, keep
        # the student flow usable with a short local coaching prompt.
        full_response = ""
        try:
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
        except Exception:
            full_response = offline_coaching_response(user_input)
            yield full_response

        full_response = strip_sycophantic_opener(full_response)
        state["response"] = full_response

        # Persist to stores
        try:
            self._update_stores(state)
        except Exception as exc:
            print(f"[reasoning] Could not persist chat turn: {exc}")

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
