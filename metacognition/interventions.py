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
from dataclasses import dataclass, field
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
        "Forget the whole problem. What's the very first word in the question?",
        "What do you know for certain, even if it's tiny?",
        "What would you try if you had to guess?",
        "Tell me what you DO understand so far.",
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
        "Just read the first sentence out loud.",
        "What's one word in this problem you recognize?",
        "If you had to make a guess, what would it be?",
        "What would you tell a friend to do first?",
    ],
    "INSIGHT": [
        "Explain it back to me like I've never seen this.",
        "Where else could you use what you just figured out?",
        "What was the moment it clicked?",
        "Could you solve a harder version of this now?",
    ],
}

# States for which repeated occurrence triggers escalation.
ESCALATION_STATES = {"CONFUSED", "FRUSTRATED", "STUCK"}
ESCALATION_THRESHOLD = 3


@dataclass
class Intervention:
    state: str
    text: str
    escalated: bool = False
    escalation_kind: Optional[str] = None  # "hint" | "break" | "first_step"
    consecutive: int = 1

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "text": self.text,
            "escalated": self.escalated,
            "escalation_kind": self.escalation_kind,
            "consecutive": self.consecutive,
        }


class MetacognitiveInterventionGenerator:
    """Select and adapt metacognitive interventions per student."""

    def __init__(self, student_name: str = "default", seed: int = 0):
        self.student_name = student_name
        # Deterministic-yet-varied index counter (no Math.random needed).
        self._counter = seed
        # Per-student effectiveness stats:
        #   stats[state][text] = {"tries": n, "improved": m}
        self.stats: dict[str, dict[str, dict]] = {}
        # Track consecutive-state history for escalation.
        self._last_state: Optional[str] = None
        self._consecutive = 0
        # Track (state, text) of the last intervention so record_outcome() can
        # attribute an improvement to it.
        self._last_intervention: Optional[tuple[str, str]] = None
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
            interv = self._escalate(state, self._consecutive)
        else:
            text = self._select(state)
            interv = Intervention(state=state, text=text, consecutive=self._consecutive)

        self._last_intervention = (interv.state, interv.text)
        self._register_try(interv.state, interv.text)
        return interv.to_dict()

    def _escalate(self, state: str, consecutive: int) -> Intervention:
        if state == "CONFUSED":
            # add one tiny hint then ask again
            base = self._select(state)
            text = ("Here's one tiny nudge: focus on just the numbers or key "
                    f"terms first. Now — {base.lower()}")
            return Intervention(state, text, escalated=True,
                                escalation_kind="hint", consecutive=consecutive)
        if state == "FRUSTRATED":
            text = "Take a real break. Come back in 5 minutes."
            return Intervention(state, text, escalated=True,
                                escalation_kind="break", consecutive=consecutive)
        # STUCK -> give the first step only, then ask what's next
        text = ("Let's do the very first step together: read the problem and "
                "write down the one thing it's asking for. What's the next step "
                "after that?")
        return Intervention(state, text, escalated=True,
                            escalation_kind="first_step", consecutive=consecutive)

    def _select(self, state: str) -> str:
        """Pick an intervention, weighted by per-student effectiveness."""
        options = INTERVENTION_BANK[state]
        state_stats = self.stats.get(state, {})

        # Score each option: prefer higher success rate; unused options get a
        # neutral optimistic prior so they still get tried.
        scored: list[tuple[float, int, str]] = []
        for i, text in enumerate(options):
            s = state_stats.get(text, {"tries": 0, "improved": 0})
            tries, improved = s["tries"], s["improved"]
            # Laplace-smoothed success rate.
            rate = (improved + 1) / (tries + 2)
            # Rotate the tie-breaker so equal-rate options cycle deterministically.
            rotation = (i + self._counter) % len(options)
            scored.append((rate, -rotation, text))

        self._counter += 1
        # Deprioritise options that consistently never work (rate ~0 with
        # enough evidence) by sorting them last.
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return scored[0][2]

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
