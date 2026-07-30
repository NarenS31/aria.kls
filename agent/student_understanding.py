"""Context-aware semantic parsing for natural student language.

The fast intent model handles obvious turns in milliseconds. Ambiguous,
out-of-distribution, mixed-intent, and referential turns are sent to a small
local language model. The language model may interpret what the student means,
but it is never trusted to decide whether mathematical work is correct.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import ollama

from agent.student_intent import StudentIntent, classify_student_intent
from agent.reasoning_moves import observe_reasoning_moves


INTENT_LABELS = (
    "HELP_REQUEST",
    "ATTEMPT_META",
    "FRUSTRATION",
    "CLARIFICATION_REQUEST",
    "CONFIRMATION_REQUEST",
    "CONTROL_REQUEST",
    "SELF_CORRECTION",
    "UNCERTAINTY",
    "REASONING",
    "SHORT_ANSWER",
    "SOCIAL",
    "OTHER",
)

_DEEP_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENT_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "contains_reasoning": {"type": "boolean"},
        "reasoning_summary": {"type": "string"},
        "requested_support": {"type": "string"},
        "referent": {"type": "string"},
        "affect": {"type": "string"},
        "ambiguous": {"type": "boolean"},
    },
    "required": [
        "intent",
        "confidence",
        "contains_reasoning",
        "reasoning_summary",
        "requested_support",
        "referent",
        "affect",
        "ambiguous",
    ],
}

_UNCERTAINTY_LANGUAGE = re.compile(
    r"\b(?:maybe|probably|prolly|possibly|i think|i guess|could it|might|"
    r"not sure|unsure|leaning|seems like)\b",
    re.IGNORECASE,
)
_REFERENTIAL_LANGUAGE = re.compile(
    r"\b(?:it|that|this|there|the thing|that part|this part|what you said)\b",
    re.IGNORECASE,
)
_QUESTION_LANGUAGE = re.compile(
    r"(?:\?|^(?:what|why|how|where|when|which|can|could|would|did|does|"
    r"is|are|am)\b)",
    re.IGNORECASE,
)
_CORRECTION_LANGUAGE = re.compile(
    r"\b(?:nvm|never ?mind|scratch that|hold up|my bad|i meant|"
    r"read it wrong|forgot the|not .+ but)\b",
    re.IGNORECASE,
)
_SLANG_OR_FRAGMENT = re.compile(
    r"\b(?:bro|bruh|lowkey|rn|ts|idk|idek|ngl|cuz|bc|u|ur|wym)\b",
    re.IGNORECASE,
)

_SEMANTIC_PATTERNS = (
    (
        "CONTROL_REQUEST",
        re.compile(
            r"\b(?:new|different|another|next)\s+(?:problem|question|one)\b|"
            r"\b(?:skip|switch|change)\b|"
            r"\bmake (?:it|this) (?:a little )?(?:easier|harder)\b|"
            r"\btry (?:english|math|reading|writing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ATTEMPT_META",
        re.compile(
            r"\b(?:haven'?t|havent|didn'?t|didnt|not)\s+"
            r"(?:even\s+)?(?:start|started|begin|begun|try|tried|attempt|"
            r"attempted|touch|touched|do|done)\b|"
            r"\b(?:just|only)\s+(?:started|beginning)\b|"
            r"\b(?:first|1st)\s+(?:attempt|try|go|time)\b|"
            r"\bwe (?:literally )?just started\b|"
            r"\bnot (?:a|an) (?:attempt|answer|step)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "FRUSTRATION",
        re.compile(
            r"\b(?:hate|annoying|frustrat(?:ed|ing)|making me mad|"
            r"over this|done with this|give up|impossible|stupid)\b|"
            r"\b(?:can'?t|cant|cannot) do (?:this|it|ts)\b|"
            r"\bthis (?:sucks|is awful|makes no sense)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CLARIFICATION_REQUEST",
        re.compile(
            r"\bwhat (?:does|do|is|are|did) .+ mean\b|"
            r"\b.+ means? what\b|"
            r"\b(?:explain|define|clarify|rephrase)\b|"
            r"\b(?:say|put) (?:that|it) (?:again|differently|in normal words)\b|"
            r"\bwhy (?:do|does|did|is|are|would|should|can)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CONFIRMATION_REQUEST",
        re.compile(
            r"\b(?:is|was) (?:that|this|it|my answer) "
            r"(?:right|correct|okay|ok)\b|"
            r"\b(?:am i|did i|have i) .*(?:right|correct|properly|mess(?:ed)? .*up)\b|"
            r"\bdoes (?:that|this|it|my (?:step|work|answer)) "
            r"(?:look right|check out|work)\b|"
            r"\bthat(?:'s|s| is) (?:the answer|right|correct),? right\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SELF_CORRECTION",
        re.compile(
            r"\b(?:nvm|never ?mind|scratch that|hold up|my bad|"
            r"i meant|i read it wrong|forgot (?:the|a)|"
            r"not .+ (?:but|actually)|correction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "HELP_REQUEST",
        re.compile(
            r"\b(?:help|lost|stuck|no clue|clueless|need a hint|"
            r"point me in the right direction|where do i (?:go|start|begin)|"
            r"how do i (?:start|begin)|what am i supposed to do|"
            r"what do i do|don'?t know where|dont know where)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "UNCERTAINTY",
        _UNCERTAINTY_LANGUAGE,
    ),
)

_SHORT_ANSWER = re.compile(
    r"^\s*(?:[a-d]|true|false|yes|no|[-+]?\d+(?:\.\d+)?(?:/\d+)?"
    r"|[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_SOCIAL_ONLY = re.compile(
    r"^\s*(?:hi|hello|hey|thanks?|thank you|thx|alright|aight|"
    r"ok(?:ay)?|cool|gotcha|bye|goodbye)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StudentUnderstanding:
    intent: str
    confidence: float
    contains_reasoning: bool
    reasoning_summary: str
    requested_support: str
    referent: str
    affect: str
    ambiguous: bool
    source: str
    fast_intent: str
    fast_confidence: float
    observable_moves: tuple[str, ...] = ()
    move_evidence: tuple[dict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _needs_deep_parse(text: str, fast: StudentIntent) -> bool:
    """Prefer recall: a confident fast-model mistake is worse than a fallback."""
    normalized = (text or "").strip()
    words = normalized.split()
    if fast.label == "OTHER":
        return True
    if fast.confidence < 0.91:
        return True
    if _UNCERTAINTY_LANGUAGE.search(normalized):
        return True
    if _CORRECTION_LANGUAGE.search(normalized):
        return True
    if _QUESTION_LANGUAGE.search(normalized):
        return True
    if _SLANG_OR_FRAGMENT.search(normalized):
        return True
    if _REFERENTIAL_LANGUAGE.search(normalized) and len(words) <= 14:
        return True
    return False


def _semantic_override(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    for label, pattern in _SEMANTIC_PATTERNS:
        if pattern.search(normalized):
            return label
    if _SOCIAL_ONLY.fullmatch(normalized):
        return "SOCIAL"
    if _SHORT_ANSWER.fullmatch(normalized):
        return "SHORT_ANSWER"
    if re.search(
        r"\b(?:because|since|therefore|so then|"
        r"i (?:would|will|could|plan to|chose|picked|got)|"
        r"i(?:'ll|'d) |my (?:plan|next step) is|"
        r"add(?:ed|ing)?|subtract(?:ed|ing)?|multipl(?:y|ied|ying)|"
        r"divid(?:e|ed|ing)|distribut(?:e|ed|ing)|factor(?:ed|ing)?)\b|=",
        normalized,
        re.IGNORECASE,
    ):
        return "REASONING"
    return None


def _recent_context(recent_turns: Iterable[dict] | None) -> str:
    rows = []
    for turn in list(recent_turns or [])[-3:]:
        student = str(turn.get("student", "")).strip()
        aria = str(turn.get("aria", "")).strip()
        if student:
            rows.append(f"STUDENT: {student}")
        if aria:
            rows.append(f"ARIA: {aria}")
    return "\n".join(rows) or "(none)"


def _deep_parse(
    text: str,
    problem: str,
    recent_turns: Iterable[dict] | None,
    model: str,
) -> dict[str, Any] | None:
    system = """You are the language-understanding layer of a tutoring tool.
Interpret informal student speech: slang, typos, fragments, indirect requests,
self-corrections, and references to the active problem or recent dialogue.

Choose the student's PRIMARY communicative intent:
- HELP_REQUEST: asks for help or a starting point, including indirect distress
- ATTEMPT_META: talks about beginning/not beginning, without proposing a step
- FRUSTRATION: affect or disengagement is the main message
- CLARIFICATION_REQUEST: asks what a term, instruction, or explanation means
- CONFIRMATION_REQUEST: asks whether their work/answer is correct
- CONTROL_REQUEST: asks to skip, switch, change difficulty, or get a new task
- SELF_CORRECTION: revises or retracts earlier reasoning
- UNCERTAINTY: proposes/considers content while explicitly unsure
- REASONING: gives a plan, operation, claim, evidence, or explanation
- SHORT_ANSWER: supplies only an answer or choice
- SOCIAL: greeting, thanks, or acknowledgement without academic content
- OTHER: genuinely off-topic or unintelligible

Do not infer a plan from words such as 'first attempt' or 'just started'.
Do not mark distress-only text as reasoning. A correction can contain reasoning.
Resolve words like 'it' or 'that part' from context when possible. If not
possible, set ambiguous true. Summarize only content actually stated. Do not
solve the problem and do not judge correctness."""
    user = (
        f"ACTIVE PROBLEM:\n{problem or '(not provided)'}\n\n"
        f"RECENT DIALOGUE:\n{_recent_context(recent_turns)}\n\n"
        f"CURRENT STUDENT TURN:\n{text}"
    )
    try:
        response = ollama.Client(timeout=2.5).chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=_DEEP_SCHEMA,
            options={"temperature": 0, "num_predict": 150},
            keep_alive=3600,
        )
        parsed = json.loads(response.message.content)
        if parsed.get("intent") not in INTENT_LABELS:
            return None
        return parsed
    except Exception:
        return None


def understand_student_turn(
    text: str,
    *,
    problem: str = "",
    recent_turns: Iterable[dict] | None = None,
    allow_deep: bool = False,
    model: str = "llama3.2:3b",
) -> StudentUnderstanding:
    """Parse a turn using a fast path plus a contextual local-model fallback."""
    fast = classify_student_intent(text)
    move_evidence = observe_reasoning_moves(text)
    observable_moves = tuple(move.code for move in move_evidence)
    serialized_moves = tuple(move.to_dict() for move in move_evidence)
    semantic_label = _semantic_override(text)
    if semantic_label is not None:
        contains_reasoning = fast.contains_reasoning or (
            semantic_label in {"REASONING", "SELF_CORRECTION", "UNCERTAINTY"}
            and bool(re.search(
                r"\b(?:add|subtract|multiply|divide|distribut|because|"
                r"equals?|=|claim|evidence|quote|answer)\b",
                text,
                re.IGNORECASE,
            ))
        )
        return StudentUnderstanding(
            intent=semantic_label,
            confidence=0.94,
            contains_reasoning=contains_reasoning,
            reasoning_summary="",
            requested_support="",
            referent="",
            affect="frustrated" if semantic_label == "FRUSTRATION" else "",
            ambiguous=False,
            source="semantic_router",
            fast_intent=fast.label,
            fast_confidence=fast.confidence,
            observable_moves=observable_moves,
            move_evidence=serialized_moves,
        )
    parsed = (
        _deep_parse(text, problem, recent_turns, model)
        if allow_deep and _needs_deep_parse(text, fast)
        else None
    )
    if parsed is not None:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        return StudentUnderstanding(
            intent=str(parsed["intent"]),
            confidence=confidence,
            contains_reasoning=bool(parsed.get("contains_reasoning", False)),
            reasoning_summary=str(parsed.get("reasoning_summary", "")).strip(),
            requested_support=str(parsed.get("requested_support", "")).strip(),
            referent=str(parsed.get("referent", "")).strip(),
            affect=str(parsed.get("affect", "")).strip(),
            ambiguous=bool(parsed.get("ambiguous", False)),
            source="contextual_local_model",
            fast_intent=fast.label,
            fast_confidence=fast.confidence,
            observable_moves=observable_moves,
            move_evidence=serialized_moves,
        )
    return StudentUnderstanding(
        intent=fast.label,
        confidence=fast.confidence,
        contains_reasoning=fast.contains_reasoning,
        reasoning_summary="",
        requested_support="",
        referent="",
        affect="",
        ambiguous=fast.label == "OTHER",
        source="fast_intent_model",
        fast_intent=fast.label,
        fast_confidence=fast.confidence,
        observable_moves=observable_moves,
        move_evidence=serialized_moves,
    )
