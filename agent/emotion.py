"""
Emotion-aware pacing for ARIA.

EmotionTracker monitors per-session signals:
  - Response length trend
  - Response time trend
  - Frustration keywords
  - Stuck-loop detection (repeated same question)
  - Emoji sentiment
  - Profanity rate

Emotional states: FLOW, CONFUSED, FRUSTRATED, DISENGAGED, OVERWHELMED

State transitions trigger automatic ARIA responses.
Session emotion log saved to data/emotion_logs/session_{date}.jsonl
"""

import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
EMOTION_LOGS_DIR = DATA_DIR / "emotion_logs"
EMOTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

FRUSTRATION_KEYWORDS = re.compile(
    r"\b(idk|i\s*don'?t\s*know|whatever|ugh+|forget\s+it|i\s+give\s+up|"
    r"this\s+is\s+stupid|i\s+don'?t\s+get\s+it|can'?t\s+do\s+this|"
    r"this\s+sucks|i\s+hate\s+this|nvm|nevermind|this\s+makes\s+no\s+sense)\b",
    re.IGNORECASE,
)

NEGATIVE_EMOJIS = re.compile(r"[😭🥲😤😩😫😰😱😵💀🤯😿]")
PROFANITY = re.compile(r"\b(wtf|shit|fuck|crap|damn|hell)\b", re.IGNORECASE)

CONFUSED_PHRASES = re.compile(
    r"\b(what\?|huh\?|i\s+don'?t\s+understand|can\s+you\s+repeat|say\s+that\s+again|"
    r"i'?m\s+lost|wait\s+what|repeat\s+that|one\s+more\s+time)\b",
    re.IGNORECASE,
)

STATES = ("FLOW", "CONFUSED", "FRUSTRATED", "DISENGAGED", "OVERWHELMED")

# ---------------------------------------------------------------------------
# State → automatic ARIA response snippets
# ---------------------------------------------------------------------------

STATE_RESPONSES = {
    "CONFUSED": (
        "No worries — let me make this even smaller. "
        "Forget everything I just said. Here's the ONE thing you need to know right now:"
    ),
    "FRUSTRATED": (
        "Okay, let's stop and breathe for a second. This is genuinely hard stuff. "
        "Want to try a completely different angle, or take a 2-minute break first?"
    ),
    "DISENGAGED": "Hey, still with me? Just say Yes or No — no wrong answer.",
    "OVERWHELMED": "Stop. Let's do just one tiny thing. Ready?",
}


# ---------------------------------------------------------------------------
# EmotionTracker
# ---------------------------------------------------------------------------

class EmotionTracker:
    def __init__(self):
        self.state: str = "FLOW"
        self._history: deque = deque(maxlen=10)  # recent turn signals
        self._consecutive_negative: int = 0
        self._last_questions: deque = deque(maxlen=5)  # detect stuck loops
        self._last_input_time: Optional[float] = None
        self._session_date = datetime.now().date().isoformat()
        self._log_path = EMOTION_LOGS_DIR / f"session_{self._session_date}.jsonl"

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, user_text: str) -> dict:
        """
        Analyze a user message and return:
          {state, signals, intervention, auto_response}
        auto_response is a string for ARIA to prepend if state is negative, else "".
        """
        now = time.time()
        word_count = len(user_text.split())
        elapsed = (now - self._last_input_time) if self._last_input_time else None
        self._last_input_time = now

        signals = {
            "word_count": word_count,
            "frustration_keywords": bool(FRUSTRATION_KEYWORDS.search(user_text)),
            "negative_emojis": bool(NEGATIVE_EMOJIS.search(user_text)),
            "profanity": bool(PROFANITY.search(user_text)),
            "confused_phrase": bool(CONFUSED_PHRASES.search(user_text)),
            "very_short": word_count < 5,
            "question_marks": user_text.count("?"),
            "elapsed_seconds": round(elapsed, 1) if elapsed else None,
        }

        # Stuck loop: same or very similar short message repeated
        stub = user_text.strip().lower()[:60]
        stuck = self._last_questions.count(stub) >= 2
        signals["stuck_loop"] = stuck
        self._last_questions.append(stub)

        self._history.append(signals)
        new_state = self._compute_state(signals)

        if new_state in ("FRUSTRATED", "OVERWHELMED", "CONFUSED", "DISENGAGED"):
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0

        self.state = new_state

        intervention = None
        if self._consecutive_negative >= 3:
            intervention = "mandatory_break"

        auto_response = ""
        if new_state in STATE_RESPONSES:
            auto_response = STATE_RESPONSES[new_state]

        result = {
            "state": new_state,
            "signals": signals,
            "intervention": intervention,
            "auto_response": auto_response,
            "consecutive_negative": self._consecutive_negative,
        }

        self._log(user_text, result)
        return result

    def _compute_state(self, signals: dict) -> str:
        frustrated_signals = (
            signals["frustration_keywords"]
            or signals["negative_emojis"]
            or signals["profanity"]
        )
        confused_signals = signals["confused_phrase"] or signals["stuck_loop"]
        short = signals["very_short"]

        recent = list(self._history)[-4:]

        # Count recent frustrated/short turns
        recent_frustrated = sum(
            1 for s in recent
            if s.get("frustration_keywords") or s.get("negative_emojis") or s.get("profanity")
        )
        recent_short = sum(1 for s in recent if s.get("very_short"))

        # OVERWHELMED: rapid frustrated responses
        if frustrated_signals and recent_frustrated >= 3:
            return "OVERWHELMED"

        # FRUSTRATED: clear frustration signal
        if frustrated_signals:
            return "FRUSTRATED"

        # CONFUSED: stuck or asking to repeat
        if confused_signals:
            return "CONFUSED"

        # DISENGAGED: consistently very short, no frustration
        if short and recent_short >= 3:
            return "DISENGAGED"

        return "FLOW"

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def can_introduce_new_concept(self) -> bool:
        return self.state == "FLOW"

    def should_simplify(self) -> bool:
        return self.state in ("CONFUSED", "FRUSTRATED", "OVERWHELMED")

    def needs_break(self) -> bool:
        return self._consecutive_negative >= 3

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, user_text: str, result: dict) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "state": result["state"],
            "consecutive_negative": result["consecutive_negative"],
            "signals": result["signals"],
            "user_preview": user_text[:80],
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def session_arc(self) -> list:
        """Return list of state labels from the emotion log for this session."""
        if not self._log_path.exists():
            return []
        states = []
        try:
            with open(self._log_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        states.append(rec.get("state", "FLOW"))
                    except Exception:
                        continue
        except Exception:
            pass
        return states
