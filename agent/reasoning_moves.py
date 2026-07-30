"""Observable reasoning-move extraction for ARIA research logs.

These codes describe evidence present in a student's utterance. They do not
claim to reveal a hidden cognitive or clinical state. The extractor is a
transparent baseline that must be validated against independent human labels
before its outputs are used as research outcomes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


MOVE_DEFINITIONS = {
    "TASK_ORIENTATION": "Restates or analyzes what the task requires.",
    "TASK_META": "Comments on beginning, attempting, or participating without proposing an academic step.",
    "PLAN": "Names an intended strategy or future step.",
    "STRATEGY_STEP": "States a concrete operation, claim, or evidence choice.",
    "JUSTIFICATION": "Gives a reason connecting a step to the task.",
    "MONITORING": "Checks current understanding, progress, or correctness.",
    "EVALUATION": "Judges a completed result or strategy against a criterion.",
    "SELF_CORRECTION": "Retracts or revises an earlier claim or step.",
    "UNCERTAINTY": "Explicitly marks uncertainty about academic content.",
    "HELP_SEEKING": "Requests a hint, explanation, starting point, or check.",
    "ANSWER_ONLY": "Provides an answer without visible reasoning.",
    "AFFECT": "Expresses task-related emotion or disengagement.",
    "OFF_TASK": "Contains no codable task reasoning or regulation move.",
}


@dataclass(frozen=True)
class MoveEvidence:
    code: str
    evidence: str
    start: int
    end: int
    detector: str = "transparent_pattern_baseline"

    def to_dict(self) -> dict:
        return asdict(self)


_PATTERNS = {
    "TASK_ORIENTATION": (
        r"\b(?:i have to|the question (?:asks|wants)|i need to find|"
        r"the goal is|it says to|we are supposed to)\b",
    ),
    "TASK_META": (
        r"\b(?:this is my first (?:attempt|try)|i(?:'m| am) (?:just )?"
        r"(?:starting|beginning)|i haven'?t (?:started|begun)|"
        r"i did not (?:start|begin)|i(?:'m| am) trying)\b",
    ),
    "PLAN": (
        r"\b(?:(?:first|next|then)\s+(?:i(?:'ll| will| would| could)|"
        r"we(?:'ll| will)|add|subtract|multiply|divide|distribute|"
        r"compare|write|read|check)|my plan|i(?:'ll| will| would| could)|"
        r"i plan to|i(?:'m| am) going to|before i)\b",
    ),
    "STRATEGY_STEP": (
        r"\b(?:add|subtract|multiply|divide|distribute|factor|substitute|"
        r"combine|isolate|compare|quote|claim|evidence|revise|rewrite|"
        r"calculate|graph|solve)\w*\b",
        r"(?:[a-z]\s*=\s*[-+]?\d|\d+\s*[+\-*/=]\s*\d+)",
    ),
    "JUSTIFICATION": (
        r"\b(?:because|since|therefore|so that|which means|this shows|"
        r"the reason is|that matters because)\b",
    ),
    "MONITORING": (
        r"\b(?:does (?:this|that|it) make sense|am i (?:right|on track)|"
        r"is (?:this|that|it) (?:right|correct)|check my|"
        r"i understand|i don'?t understand|where did i go wrong)\b",
    ),
    "EVALUATION": (
        r"\b(?:my answer (?:works|doesn'?t work)|i checked|"
        r"it satisfies|it does not satisfy|the result is reasonable|"
        r"that strategy (?:worked|failed)|i completed)\b",
    ),
    "SELF_CORRECTION": (
        r"\b(?:wait|actually|i meant|my bad|scratch that|nvm|never ?mind|"
        r"i read it wrong|i forgot|correction)\b",
    ),
    "UNCERTAINTY": (
        r"\b(?:maybe|probably|prolly|possibly|i think|i guess|might|"
        r"not sure|unsure|could it|leaning toward|seems like)\b",
    ),
    "HELP_SEEKING": (
        r"\b(?:help|hint|explain|clarify|what does .+ mean|"
        r"where do i (?:start|begin|go)|how do i (?:start|begin)|"
        r"what am i supposed to do|can you check|did i mess .+ up|"
        r"i(?:'m| am) lost|i(?:'m| am) stuck|no clue)\b",
    ),
    "AFFECT": (
        r"\b(?:frustrat\w*|annoy\w*|hate|mad|overwhelmed|"
        r"this sucks|give up|i can'?t do this|boring|excited)\b",
    ),
}

_ANSWER_ONLY = re.compile(
    r"^\s*(?:[a-d]|true|false|[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
    r"[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def observe_reasoning_moves(text: str) -> list[MoveEvidence]:
    """Return multi-label observable moves and exact supporting spans."""
    utterance = text or ""
    found: list[MoveEvidence] = []
    for code, patterns in _PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, utterance, re.IGNORECASE)
            if match:
                found.append(MoveEvidence(
                    code=code,
                    evidence=match.group(0),
                    start=match.start(),
                    end=match.end(),
                ))
                break
    if _ANSWER_ONLY.fullmatch(utterance):
        found.append(MoveEvidence(
            code="ANSWER_ONLY",
            evidence=utterance.strip(),
            start=max(0, len(utterance) - len(utterance.lstrip())),
            end=len(utterance.rstrip()),
        ))
    if not found:
        found.append(MoveEvidence(
            code="OFF_TASK",
            evidence=utterance.strip(),
            start=0,
            end=len(utterance),
        ))
    return found
