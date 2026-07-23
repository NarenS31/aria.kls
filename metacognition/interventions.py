"""
Metacognitive intervention generation for ARIA.

Given a detected cognitive state, this module returns exactly ONE Socratic
question that pushes the student to reflect — it NEVER gives the answer and
NEVER explains the concept. The intent is to respond to *what the student
said* (their state), not to solve the problem for them.

Design rules (hard constraints):
  * NEVER give the answer directly.
  * NEVER explain the concept unprompted.
  * ALWAYS respond to the state, not the underlying question.
  * ONE question at a time, always.

Each state has a fixed bank of interventions. Selection is random within the
state, but weighted by per-student effectiveness: interventions that have
historically moved this student to a better state are favoured, and ones that
never work for them are deprioritised over time.

Escalation: if a student stays in the same hard state for 3+ consecutive
turns, the generator escalates (a tiny hint, a break suggestion, or the first
step) instead of asking yet another question.

Effectiveness is persisted per student to
    data/metacognition/interventions_{name}.json
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
META_DIR = os.path.join(REPO_ROOT, "data", "metacognition")

# States ranked from worst (0) to best (higher). Used to decide whether an
# intervention "improved" the student's state.
STATE_RANK = {
    "STUCK": 0,
    "FRUSTRATED": 1,
    "CONFUSED": 2,
    "RUSHING": 3,
    "PLANNING": 4,
    "FLOW": 5,
    "INSIGHT": 6,
}


INTERVENTION_BANK: dict[str, list[str]] = {
    "PLANNING": [
        "Good — what's your overall strategy?",
        "Before you start, what do you predict will be hardest?",
        "What do you already know that applies here?",
    ],
    "FLOW": [
        "Can you explain why that step works, not just what you did?",
        "Would this same approach work on a different problem?",
        "What would happen if you changed one part of your approach?",
    ],
    "CONFUSED": [
        "You mentioned {concept}. What do you already know about that specifically?",
        "You mentioned {concept}. Which part feels unclear right now?",
        "You mentioned {concept}. Can you point to the exact step that loses you?",
        "Where does {concept} stop making sense to you?",
        "Which part of {concept} would be most useful to clarify first?",
        "You mentioned {concept}. What do you understand so far, even a little?",
    ],
    "RUSHING": [
        "Stop. What's your plan before going further?",
        "Walk me through your reasoning step by step.",
        "What did the problem actually ask for?",
        "Read the question out loud again.",
    ],
    "FRUSTRATED": [
        "That's genuinely hard. What's the one part you DO understand?",
        "Take 30 seconds. Then tell me just the first thing you'd try.",
        "You're not missing something obvious — what have you tried?",
    ],
    "STUCK": [
        "What is the smallest action you can take without solving the whole problem?",
        "What is one tiny first step you can take without finding the answer?",
        "You're stuck, so let's make this tiny: what is one thing you know for certain?",
        "What is the question asking you to find, in your own words?",
        "Without solving it, what is one detail you can write down?",
        "What is one useful action you can take in the next ten seconds?",
    ],
    "INSIGHT": [
        "What changed in your thinking when it clicked?",
        "Where else could you apply what you just realized?",
        "How would you explain what just clicked to another student?",
        "What clue could help you recognize this pattern next time?",
        "How is this insight different from what you thought before?",
        "What made it click for you just now?",
    ],
}

INTERVENTION_CITATIONS = {
    "PLANNING": "zimmerman_2002",
    "FLOW": "hattie_timperley_2007",
    "CONFUSED": "sweller_1988",
    "RUSHING": "barkley_2015",
    "FRUSTRATED": "shaw_2014",
    "STUCK": "sweller_1988",
    "INSIGHT": "zimmerman_2002",
}

# States for which repeated occurrence triggers escalation.
ESCALATION_STATES = {"CONFUSED", "FRUSTRATED", "STUCK"}
ESCALATION_THRESHOLD = 3


@dataclass
class Intervention:
    state: str
    text: str
    template: str = ""
    escalated: bool = False
    escalation_kind: Optional[str] = None  # "hint" | "break" | "first_step"
    consecutive: int = 1
    citation_key: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "text": self.text,
            "escalated": self.escalated,
            "escalation_kind": self.escalation_kind,
            "consecutive": self.consecutive,
            "citation_key": self.citation_key,
            "citation_keys": [self.citation_key] if self.citation_key else [],
        }


class MetacognitiveInterventionGenerator:
    """Select and adapt metacognitive interventions per student."""

    def __init__(self, student_name: str = "default", seed: int = 0):
        self.student_name = student_name
        self._rng = random.Random(seed)
        # Per-student effectiveness stats:
        #   stats[state][text] = {"tries": n, "improved": m}
        self.stats: dict[str, dict[str, dict]] = {}
        # Track consecutive-state history for escalation.
        self._last_state: Optional[str] = None
        self._consecutive = 0
        # Track (state, text) of the last intervention so record_outcome() can
        # attribute an improvement to it.
        self._last_intervention: Optional[tuple[str, str]] = None
        self._last_template: Optional[str] = None
        self._rotation_queues: dict[str, list[str]] = {}
        self.interventions_used_this_session: set[str] = set()
        self._path = os.path.join(META_DIR, f"interventions_{_slug(student_name)}.json")
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self.stats = json.load(fh).get("stats", {})
            except (json.JSONDecodeError, OSError):
                self.stats = {}

    def _save(self) -> None:
        os.makedirs(META_DIR, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"student": self.student_name, "stats": self.stats}, fh, indent=2)
        os.replace(tmp, self._path)

    # -- selection ---------------------------------------------------

    def generate(
        self,
        state: str,
        consecutive_count: Optional[int] = None,
        student_input: str = "",
    ) -> dict:
        """Return one intervention for `state`.

        `consecutive_count` may be supplied by the caller (e.g. the tracker);
        otherwise it is inferred from calls to this generator.
        """
        state = state.upper()
        if state not in INTERVENTION_BANK:
            state = "CONFUSED"

        # Update consecutive-state counter.
        if consecutive_count is not None:
            self._consecutive = consecutive_count
        elif state == self._last_state:
            self._consecutive += 1
        else:
            self._consecutive = 1
        self._last_state = state

        # Escalate if stuck in a hard state too long.
        if state in ESCALATION_STATES and self._consecutive >= ESCALATION_THRESHOLD:
            interv = self._escalate(state, self._consecutive, student_input)
        else:
            template = self._select_template(state)
            text = _render_template(template, student_input)
            interv = Intervention(
                state=state,
                text=text,
                template=template,
                consecutive=self._consecutive,
                citation_key=INTERVENTION_CITATIONS[state],
            )

        tracked_text = interv.template or interv.text
        self._last_intervention = (interv.state, tracked_text)
        self._register_try(interv.state, tracked_text)
        return interv.to_dict()

    def reset_session(self) -> None:
        """Reset prompt rotation and consecutive-state tracking."""
        self._last_state = None
        self._consecutive = 0
        self._last_intervention = None
        self._last_template = None
        self._rotation_queues.clear()
        self.interventions_used_this_session.clear()

    def _escalate(
        self, state: str, consecutive: int, student_input: str
    ) -> Intervention:
        if state == "CONFUSED":
            template = self._select_template(state)
            base = _render_template(template, student_input)
            text = ("Here's one tiny nudge: focus on just the numbers or key "
                    f"terms first. Now — {base.lower()}")
            return Intervention(
                state, text, template=template, escalated=True,
                escalation_kind="hint", consecutive=consecutive,
                citation_key=INTERVENTION_CITATIONS[state],
            )
        if state == "FRUSTRATED":
            text = "Take a real break. Come back in 5 minutes."
            return Intervention(
                state, text, template=text, escalated=True,
                escalation_kind="break", consecutive=consecutive,
                citation_key=INTERVENTION_CITATIONS[state],
            )
        # STUCK -> give the first step only, then ask what's next
        text = ("Let's do the very first step together: read the problem and "
                "write down the one thing it's asking for. What's the next step "
                "after that?")
        return Intervention(
            state, text, template=text, escalated=True,
            escalation_kind="first_step", consecutive=consecutive,
            citation_key=INTERVENTION_CITATIONS[state],
        )

    def _select_template(self, state: str) -> str:
        """Rotate a shuffled bank, preferring prompts that helped this student."""
        options = INTERVENTION_BANK[state]
        queue = self._rotation_queues.get(state, [])
        if not queue:
            for template in options:
                self.interventions_used_this_session.discard(template)

            queue = list(options)
            self._rng.shuffle(queue)
            state_stats = self.stats.get(state, {})

            def effectiveness(template: str) -> float:
                item = state_stats.get(template, {"tries": 0, "improved": 0})
                return (item["improved"] + 1) / (item["tries"] + 2)

            queue.sort(key=effectiveness, reverse=True)
            if (
                len(queue) > 1
                and self._last_template is not None
                and queue[0] == self._last_template
            ):
                queue[0], queue[1] = queue[1], queue[0]
            self._rotation_queues[state] = queue

        template = queue.pop(0)
        self.interventions_used_this_session.add(template)
        self._last_template = template
        return template

    # -- effectiveness tracking -------------------------------------

    def _register_try(self, state: str, text: str) -> None:
        self.stats.setdefault(state, {}).setdefault(text, {"tries": 0, "improved": 0})
        self.stats[state][text]["tries"] += 1
        self._save()

    def record_outcome(self, state_before: str, state_after: str) -> bool:
        """Record whether the last intervention improved the student's state.

        Returns True if it counted as an improvement. Call this once you have
        analysed the student's response to the intervention.
        """
        if not self._last_intervention:
            return False
        st, text = self._last_intervention
        improved = STATE_RANK.get(state_after, 0) > STATE_RANK.get(state_before, 0)
        entry = self.stats.setdefault(st, {}).setdefault(text, {"tries": 0, "improved": 0})
        if improved:
            entry["improved"] += 1
        self._save()
        return improved

    # -- introspection ----------------------------------------------

    def effectiveness_report(self) -> dict:
        """Per-state ranking of interventions by success rate for this student."""
        report: dict[str, list[dict]] = {}
        for state, texts in self.stats.items():
            rows = []
            for text, s in texts.items():
                tries, improved = s["tries"], s["improved"]
                rate = improved / tries if tries else None
                rows.append({
                    "text": text, "tries": tries, "improved": improved,
                    "success_rate": round(rate, 3) if rate is not None else None,
                })
            rows.sort(key=lambda r: (r["success_rate"] is None, -(r["success_rate"] or 0)))
            report[state] = rows
        return report

    def ineffective_for_student(self, min_tries: int = 3) -> list[dict]:
        """Interventions that never work for this student (0 improvements)."""
        out = []
        for state, texts in self.stats.items():
            for text, s in texts.items():
                if s["tries"] >= min_tries and s["improved"] == 0:
                    out.append({"state": state, "text": text, "tries": s["tries"]})
        return out


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()) or "default"


def extract_concept(student_input: str) -> str:
    """Extract a short problem concept without requiring another model call."""
    text = re.sub(r"\s+", " ", (student_input or "").strip())
    if not text:
        return "the problem"

    patterns = (
        r"(?:don'?t|do not|can'?t|cannot)\s+(?:understand|get|figure out)\s+(.+)",
        r"(?:confused|stuck)\s+(?:about|on|with)\s+(.+)",
        r"(?:question|problem|part)\s+(?:about|on|with)\s+(.+)",
        r"(?:how|why|what)\s+(?:do|does|is|are|can)\s+(.+)",
    )
    candidate = ""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            break

    if not candidate:
        quoted = re.search(r"[\"']([^\"']{2,80})[\"']", text)
        candidate = quoted.group(1) if quoted else text

    candidate = re.split(
        r"\b(?:because|but|and then|even though|so i|please help)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.sub(r"^[\s,.:;!?-]+|[\s,.:;!?-]+$", "", candidate)
    candidate = re.sub(
        r"^(?:the\s+)?(?:whole\s+)?(?:thing|it|this|that|anything|everything)$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    words = candidate.split()
    if not words:
        return "the problem"
    return " ".join(words[:8])


def _render_template(template: str, student_input: str) -> str:
    return template.format(concept=extract_concept(student_input))


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    gen = MetacognitiveInterventionGenerator("demo")
    print("One intervention per state:\n")
    for st in INTERVENTION_BANK:
        print(f"{st:11s} -> {gen.generate(st)['text']}")

    print("\nEscalation (STUCK x3):")
    for i in range(1, 4):
        iv = gen.generate("STUCK", consecutive_count=i)
        flag = " [ESCALATED]" if iv["escalated"] else ""
        print(f"  turn {i}: {iv['text']}{flag}")
