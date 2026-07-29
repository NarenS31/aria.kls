"""
ADHD student persona simulator for ARIA evaluation.

Three personas matching PEBBLE's schema extended with ADHD-specific behavioral parameters:
  - hyperfocus: rapid-fire questions, resists topic transitions, 5+ questions per concept
  - scattered: jumps topics mid-conversation, forgets prior context, needs reorientation
  - frustrated: short disengaged responses, increasing withdrawal signals

Episodes: 4-8 turns each. Subjects: algebra, biology, python_programming.
Misconception library: 2 canonical misconceptions per subject seed.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ollama


# ------------------------------------------------------------------
# Persona dataclass (PEBBLE-compatible schema)
# ------------------------------------------------------------------

@dataclass
class Persona:
    name: str
    # PEBBLE schema fields
    affect: str               # "eager" | "neutral" | "avoidant"
    persistence: float        # 0–1: how long they stay on a concept before giving up
    hedging: float            # 0–1: probability of hedging language ("I think...", "maybe")
    # ADHD-specific extensions
    attention_span: int       # max turns before attention drift
    topic_jump_prob: float    # probability of jumping topic mid-episode
    frustration_decay: float  # how fast frustration builds if not addressed (per turn)
    question_burst: int       # how many follow-up questions per concept (hyperfocus param)
    response_length: str      # "short" | "medium" | "verbose"
    disengagement_phrases: List[str]
    success_phrases: List[str]
    confusion_phrases: List[str]


PERSONAS: Dict[str, Persona] = {
    "hyperfocus": Persona(
        name="hyperfocus",
        affect="eager",
        persistence=0.95,
        hedging=0.15,
        attention_span=8,
        topic_jump_prob=0.05,
        frustration_decay=0.15,
        question_burst=5,
        response_length="medium",
        disengagement_phrases=["wait but also", "ok but what about", "and another thing"],
        success_phrases=["OH WAIT", "so that means", "wait so then if", "ok but then"],
        confusion_phrases=["but why though", "I still don't get why", "wait that doesn't make sense"],
    ),
    "scattered": Persona(
        name="scattered",
        affect="neutral",
        persistence=0.35,
        hedging=0.60,
        attention_span=3,
        topic_jump_prob=0.45,
        frustration_decay=0.20,
        question_burst=1,
        response_length="short",
        disengagement_phrases=["anyway", "oh wait I was thinking about something else", "never mind"],
        success_phrases=["oh ok", "wait yeah", "I think I get it?"],
        confusion_phrases=["wait what were we doing", "I forgot what we were talking about", "huh"],
    ),
    "frustrated": Persona(
        name="frustrated",
        affect="avoidant",
        persistence=0.20,
        hedging=0.30,
        attention_span=4,
        topic_jump_prob=0.10,
        frustration_decay=0.45,
        question_burst=1,
        response_length="short",
        disengagement_phrases=["idk", "forget it", "this doesn't even matter", "whatever"],
        success_phrases=["ok fine", "I guess that makes sense", "whatever ok"],
        confusion_phrases=["ugh", "this makes no sense", "why doesn't this work", "I hate this"],
    ),
}


# ------------------------------------------------------------------
# Subject seeds with misconceptions
# ------------------------------------------------------------------

@dataclass
class SubjectSeed:
    name: str
    display: str
    opening_questions: List[str]
    misconceptions: List[str]          # canonical misconceptions (2 per PEBBLE spec)
    topic_pool: List[str]              # for scattered persona topic jumps


SUBJECTS: Dict[str, SubjectSeed] = {
    "algebra": SubjectSeed(
        name="algebra",
        display="Algebra",
        opening_questions=[
            "I don't understand how to solve for x when it's on both sides",
            "Can you help me with this equation: 2x + 3 = x - 5",
            "I keep getting the wrong answer when I do these linear equations",
        ],
        misconceptions=[
            "student distributes addition before multiplication (e.g., treats 2(x+3) as 2x+3 instead of 2x+6)",
            "student drops negative signs when moving terms across the equals sign",
        ],
        topic_pool=["fractions", "inequalities", "graphing lines", "systems of equations", "polynomials"],
    ),
    "biology": SubjectSeed(
        name="biology",
        display="Biology",
        opening_questions=[
            "I'm so confused about the difference between mitosis and meiosis",
            "Can you explain what happens during cell division",
            "I don't understand why cells divide in the first place",
        ],
        misconceptions=[
            "student confuses the products of mitosis (2 identical cells) with meiosis (4 genetically unique cells)",
            "student thinks all cells divide at the same rate regardless of cell type",
        ],
        topic_pool=["DNA replication", "protein synthesis", "cell membrane", "photosynthesis", "genetics"],
    ),
    "python_programming": SubjectSeed(
        name="python_programming",
        display="Python Programming",
        opening_questions=[
            "I don't understand why my for loop isn't working right",
            "Can you help me understand how if/else statements work",
            "My code keeps printing the wrong thing and I don't know why",
        ],
        misconceptions=[
            "student uses off-by-one indexing in range() (e.g., range(1, n) when range(n) needed)",
            "student confuses local and global variable scope (modifying global inside function without global keyword)",
        ],
        topic_pool=["lists", "functions", "recursion", "dictionaries", "classes", "file I/O"],
    ),
}


# ------------------------------------------------------------------
# Simulated student message generator
# ------------------------------------------------------------------

_STUDENT_SYSTEM = """You are simulating a {persona_name} ADHD student in a tutoring session.
Student behavioral parameters:
- Affect: {affect}
- Persistence: {persistence} (0=gives up fast, 1=keeps trying)
- Hedging probability: {hedging} (add phrases like "I think" or "maybe" this fraction of the time)
- Response length style: {response_length}
- Subject: {subject}
- Turn number: {turn_num} of {max_turns}
- Frustration level: {frustration_level:.1f}/1.0

Current misconception the student holds: {misconception}

Persona-specific behavior:
{persona_instructions}

Generate ONLY the student's next message (1-3 sentences max for short, 2-4 for medium).
Do NOT generate the tutor's response. Be realistic — students don't use perfect grammar.
Show the misconception naturally if it's relevant."""


def _persona_instructions(persona: Persona, frustration_level: float, turn_num: int) -> str:
    if persona.name == "hyperfocus":
        return (
            f"Ask a follow-up question about the same concept. "
            f"Add 'but what about' or 'ok but then' style extensions. "
            f"You're intensely curious and won't let go of this topic."
        )
    elif persona.name == "scattered":
        if random.random() < persona.topic_jump_prob:
            return (
                "Suddenly switch to a different topic or mention you forgot what you were doing. "
                "Then partially come back. Be a bit scattered."
            )
        return (
            "Give a short, slightly confused response. "
            "You may have lost the thread a bit. Ask for clarification."
        )
    elif persona.name == "frustrated":
        if frustration_level > 0.6:
            return (
                f"Express growing frustration. Use one of these phrases: "
                f"{', '.join(random.sample(persona.disengagement_phrases, min(2, len(persona.disengagement_phrases))))}. "
                f"Short response. Minimal engagement."
            )
        return (
            "Give a short, slightly defeated response. "
            "Not fully engaged but still asking something."
        )
    return "Respond naturally to the tutor's last message."


class SimulatedStudent:
    def __init__(self, persona_name: str, subject_name: str, seed_idx: int = 0):
        self.persona = PERSONAS[persona_name]
        self.subject = SUBJECTS[subject_name]
        self.misconception = self.subject.misconceptions[seed_idx % len(self.subject.misconceptions)]
        self.frustration_level: float = 0.0
        self.turn_num: int = 0
        self.max_turns: int = random.randint(4, 8)
        self.model = "llama3.1:8b"

    def opening_message(self) -> str:
        return random.choice(self.subject.opening_questions)

    def next_message(self, conversation_history: List[Dict[str, str]]) -> str:
        self.turn_num += 1

        # Update frustration: decays if tutor addressed it, grows otherwise
        last_tutor = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"),
            "",
        )
        if any(w in last_tutor.lower() for w in ["that's okay", "totally fine", "makes sense", "great job", "don't worry"]):
            self.frustration_level = max(0.0, self.frustration_level - 0.2)
        else:
            self.frustration_level = min(1.0, self.frustration_level + self.persona.frustration_decay)

        system_prompt = _STUDENT_SYSTEM.format(
            persona_name=self.persona.name,
            affect=self.persona.affect,
            persistence=self.persona.persistence,
            hedging=self.persona.hedging,
            response_length=self.persona.response_length,
            subject=self.subject.display,
            turn_num=self.turn_num,
            max_turns=self.max_turns,
            frustration_level=self.frustration_level,
            misconception=self.misconception,
            persona_instructions=_persona_instructions(self.persona, self.frustration_level, self.turn_num),
        )

        history_text = conversation_history[-6:]
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(history_text)
        msgs.append({"role": "user", "content": "Generate the student's next message:"})

        try:
            result = ollama.chat(
                model=self.model,
                messages=msgs,
                options={"temperature": 0.85, "num_predict": 120},
            )
            response = result.message.content.strip()
            # Strip any "Student:" prefix the model might add
            response = re.sub(r'^(student|user)\s*:\s*', '', response, flags=re.IGNORECASE)
            return response
        except Exception:
            return random.choice(self.persona.confusion_phrases)

    def is_done(self) -> bool:
        return (
            self.turn_num >= self.max_turns
            or (self.frustration_level >= 0.9 and self.persona.name == "frustrated")
        )


import re


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode_id: str
    persona: str
    subject: str
    turns: List[Dict]


def run_episode(
    tutor_fn,
    persona_name: str,
    subject_name: str,
    seed_idx: int = 0,
    extra_meta: Optional[Dict] = None,
) -> EpisodeResult:
    """
    Run one evaluation episode.
    tutor_fn: callable(user_msg: str) -> str
    """
    episode_id = str(uuid.uuid4())
    student = SimulatedStudent(persona_name, subject_name, seed_idx)
    conversation: List[Dict[str, str]] = []
    turns_data = []

    # Opening
    opening = student.opening_message()
    conversation.append({"role": "user", "content": opening})

    turn_idx = 0
    while True:
        tutor_response = tutor_fn(opening if turn_idx == 0 else student_msg)
        conversation.append({"role": "assistant", "content": tutor_response})

        turns_data.append({
            "turn_idx": turn_idx,
            "student_msg": conversation[-2]["content"],
            "tutor_msg": tutor_response,
        })

        if student.is_done():
            break

        student_msg = student.next_message(conversation)
        conversation.append({"role": "user", "content": student_msg})
        turn_idx += 1

        if turn_idx >= student.max_turns:
            break

    return EpisodeResult(
        episode_id=episode_id,
        persona=persona_name,
        subject=subject_name,
        turns=turns_data,
    )
