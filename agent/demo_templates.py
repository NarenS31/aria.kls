"""Template tutor language for the summit demo (no LLM).

Every intervention type has 3-4 phrasings.  Slots (`{do}`, `{result}`) come from the
SymPy-verified `moves` of the problem (agent/demo_bank.py), and one phrasing per type is the
EXACT prompt from eval/metacognition/jitai.py's JITAI_INTERVENTION_BANK (`{bank}`), so the
text that the paper's policy chose is still what the student can see.  A `Rotor` makes
consecutive replies differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- leads: what the tutor acknowledges before the move ------------------------------------
LEADS = {
    "wrong": ["Not quite yet.", "That doesn't check out yet.", "Let's look at that again.", "Not there yet."],
    "help": ["No problem, let's start small.", "That's okay. Let's find a way in.", "Fair enough, one step at a time."],
    "ask_answer": ["I won't hand over the answer, but I'll help you get there.",
                   "Giving it away wouldn't help you, so here's a first move.",
                   "You can get this, so here's a way in."],
    "question": ["Good question, let's work toward it.", "Let's figure that out together.", "Worth asking. Here's a way in."],
    "stuck": [""],
    "frustrated": [""],
    "confused": [""],
}

# --- phrasings per intervention type ({bank} == exact JITAI bank prompt) ------------------
PHRASINGS = {
    "hint": [
        "Try this: {do}. What do you get?",
        "{do_result}",                         # "{do}, you should land on {result}. What comes next?" or small-step fallback
        "Start here: {do}. Tell me what you find.",
        "{bank}",
    ],
    "task_chunking": [
        "Let's split it up. Step one: {do}. What do you get?",
        "Smaller piece first: {do}. What do you see?",
        "One thing at a time. {Do}, then tell me what changes.",
        "{bank}",
    ],
    "encouragement": [
        "{bank}",
        "Tricky ones feel like this. Let's shrink it: {do}. What do you notice?",
        "Hang in there, this one takes a couple of tries. {do}. Where does that take you?",
        "You don't need the whole thing yet. {do}. What do you get?",
    ],
    "break_suggestion": [
        "{bank}",
        "Let's pause for a moment. When you're ready, start with just this: {do}.",
        "No rush. Take a breath, then try one small thing: {do}.",
    ],
    "reflection": [
        "{bank}",
        "That checks out. What was the key step for you?",
        "That checks out. Where else could you use this idea?",
        "That checks out. What would you do differently if the numbers changed?",
    ],
}

CORRECT_ACKS = ["Yes, that's correct.", "That's right.", "Yes, that checks out.", "Correct."]

WATCHING = {
    "progress": ["Watching. You're making progress, so I'll stay quiet.", "Still watching. Nothing needed from me right now.",
                 "Watching. Keep going, you're on track.", "Quiet mode: you're working it out."],
    "planning": ["Watching. Take your time to plan; I'll step in only if you need me.", "Watching while you think this through.",
                 "Quiet mode: planning looks productive."],
    "unclear": ["Watching. Take your time. Write a step or ask me something when you're ready.",
                "Watching. Try writing your next step, or ask me a question."],
    "empty": ["Watching. I'm here when you're ready."],
    "error": ["Watching. I couldn't read that one. Try writing your next step as plain math."],
    # verified wrong step, risk below this profile's threshold: say so, give no hint, no answer
    "flagged": ["Not quite yet. I'm holding the hint for now. Ask me if you want one.",
                "That step doesn't check out yet. Watching for now; ask for a hint any time.",
                "Not there yet. I'll give you room to try again, and I'm here if you want a hint."],
}


@dataclass
class Rotor:
    """Round-robin over phrasings, skipping anything equal to the last two replies."""
    counters: dict = field(default_factory=dict)
    recent: list = field(default_factory=list)

    def pick(self, key: str, options: list, render) -> tuple:
        n = len(options)
        start = self.counters.get(key, 0)
        for k in range(n):
            idx = (start + k) % n
            text = render(options[idx])
            if text and text not in self.recent[-2:]:
                self.counters[key] = idx + 1
                self._remember(text)
                return text, idx
        idx = start % n
        text = render(options[idx])
        self.counters[key] = idx + 1
        self._remember(text)
        return text, idx

    def _remember(self, text: str):
        self.recent.append(text)
        self.recent[:] = self.recent[-6:]


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if len(s) > 1 and s[0].isupper() and not s[1].isupper() else s


def fill(template: str, *, bank: str, do: str, result: Optional[str]) -> str:
    do = (do or "Look at the first piece of the problem").rstrip(".")
    Do, do = do[0].upper() + do[1:], _lower_first(do)
    if template == "{do_result}":
        if result:
            return f"{Do}, and you should land on {result}. What comes next?"
        return f"Let's take one small step: {do}. What does that give you?"
    return template.format(bank=bank, do=do, Do=Do, result=result or "")
