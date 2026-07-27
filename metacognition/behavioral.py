"""
Behavioral (keystroke + timing) feature extraction for ARIA's multimodal
cognitive-state classifier.

This layer runs ALONGSIDE the text classifier (``metacognition.analyzer``); it
does not replace it. It converts raw keystroke / timing telemetry captured in
the browser — pause before the first key, total typing time, backspace rate,
typing speed — into five soft "behavioral state signals":

    rushing, stuck, flow, planning, frustrated

Each signal is a 0-1 score (a graded probability-like proxy, NOT a hard label).
``CognitiveStateAnalyzer`` fuses these with the text-derived state.

Everything is OPTIONAL. With no keystroke telemetry the extractor still returns
a well-formed result (signals derived from response length + the session's
length trend only), and the analyzer falls back to text-only classification, so
there is zero regression when behavioral data is absent.

The signals are heuristic proxy scores for learning-science instrumentation, not
diagnostic or clinical labels.
"""

from __future__ import annotations

import math
from typing import Any, Optional


# The five behavioral signals and the cognitive state each one maps onto.
BEHAVIORAL_SIGNALS = ["rushing", "stuck", "flow", "planning", "frustrated"]
SIGNAL_TO_STATE = {
    "rushing": "RUSHING",
    "stuck": "STUCK",
    "flow": "FLOW",
    "planning": "PLANNING",
    "frustrated": "FRUSTRATED",
}

# Fallback "fast typing" threshold (wpm) used before enough per-session samples
# have accumulated to compute an empirical 80th percentile.
DEFAULT_FAST_WPM = 80.0
# Minimum rolling samples before the empirical percentile is trusted.
MIN_PERCENTILE_SAMPLES = 5
# Word/message slope below which the length trend counts as "strongly negative".
STRONG_NEGATIVE_TREND = -3.0
# Rolling-window cap for session statistics used in percentile computation.
MAX_ROLLING = 50


# ------------------------------------------------------------------
# Small numeric helpers (defensive against messy JS-provided values)
# ------------------------------------------------------------------

def _as_opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _as_opt_int(value: Any) -> Optional[int]:
    f = _as_opt_float(value)
    return None if f is None else int(round(f))


def _as_float(value: Any, default: float) -> float:
    f = _as_opt_float(value)
    return default if f is None else f


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (pct in 0-100)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _linreg_slope(ys: list[float]) -> float:
    """Least-squares slope of ys against x = 0, 1, 2, ... (words per message)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _history_word_len(entry: Any) -> Optional[int]:
    """Best-effort word count for a prior turn (dict or raw string)."""
    if isinstance(entry, str):
        return len(entry.split())
    if isinstance(entry, dict):
        for key in ("response_length_words",):
            val = entry.get(key)
            if isinstance(val, (int, float)):
                return int(val)
        bf = entry.get("behavioral_features")
        if isinstance(bf, dict) and isinstance(bf.get("response_length_words"), (int, float)):
            return int(bf["response_length_words"])
        for key in ("student_input", "student_message", "text", "content", "message"):
            val = entry.get(key)
            if isinstance(val, str):
                return len(val.split())
    return None


def _score(rules: list[tuple[bool, float]]) -> float:
    """Sum the weights of satisfied rules and normalise by total weight -> [0,1]."""
    total = sum(w for _, w in rules)
    if total <= 0:
        return 0.0
    got = sum(w for cond, w in rules if cond)
    return round(max(0.0, min(1.0, got / total)), 4)


# ------------------------------------------------------------------
# Extractor
# ------------------------------------------------------------------

class BehavioralFeatureExtractor:
    """Turn keystroke/timing telemetry + text into behavioral state signals.

    The extractor is stateful: it keeps rolling per-session statistics (typing
    speeds and response times, capped at ``max_rolling`` entries) so that the
    "faster than usual" rushing/frustration rules can use an empirical 80th
    percentile once enough samples exist.
    """

    def __init__(self, max_rolling: int = MAX_ROLLING):
        self.max_rolling = max_rolling
        self._typing_speeds: list[float] = []
        self._response_times: list[float] = []

    # -- public API --------------------------------------------------

    def extract(
        self,
        text: str,
        keystroke_data: Optional[dict] = None,
        session_history: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Extract behavioral features for one message.

        Args:
            text: the student's typed think-aloud for this turn.
            keystroke_data: raw browser telemetry, any of
                ``pause_before_first_key_ms``, ``total_typing_time_ms``,
                ``backspace_rate``, ``typing_speed_wpm``,
                ``time_since_last_message_ms`` (all optional).
            session_history: prior turns (dicts or strings) — used for the
                response-length trend, message count and inter-message timing.

        Returns a dict with the fields described in the module docstring plus a
        ``behavioral_state_signals`` sub-dict of five 0-1 scores.
        """
        text = text or ""
        keystroke_data = keystroke_data or {}
        session_history = session_history or []

        words = text.split()
        response_length_words = len(words)
        response_length_chars = len(text)

        pause = _as_opt_int(keystroke_data.get("pause_before_first_key_ms"))
        total_typing = _as_opt_int(keystroke_data.get("total_typing_time_ms"))
        backspace_rate = max(0.0, min(1.0, _as_float(keystroke_data.get("backspace_rate"), 0.0)))

        typing_speed = _as_opt_float(keystroke_data.get("typing_speed_wpm"))
        if typing_speed is None or typing_speed <= 0:
            typing_speed = self._compute_wpm(response_length_words, total_typing)

        length_trend = self._length_trend(session_history, response_length_words)
        time_since_last = _as_opt_int(keystroke_data.get("time_since_last_message_ms"))
        session_message_count = len(session_history) + 1

        # Compute the "fast typing" threshold from history recorded on prior
        # calls (before this call's speed is folded in).
        fast_threshold = self._fast_typing_threshold()

        signals = self._state_signals(
            pause=pause,
            backspace_rate=backspace_rate,
            typing_speed=typing_speed,
            response_length_words=response_length_words,
            length_trend=length_trend,
            fast_threshold=fast_threshold,
        )

        # Fold this turn's measurements into the rolling session statistics.
        if typing_speed is not None and typing_speed > 0:
            self._record(self._typing_speeds, float(typing_speed))
        if total_typing is not None and total_typing > 0:
            self._record(self._response_times, float(total_typing))

        return {
            "pause_before_first_key_ms": pause,
            "total_typing_time_ms": total_typing,
            "backspace_rate": round(backspace_rate, 4),
            "typing_speed_wpm": round(typing_speed, 2) if typing_speed is not None else None,
            "response_length_chars": response_length_chars,
            "response_length_words": response_length_words,
            "length_trend": round(length_trend, 4),
            "time_since_last_message_ms": time_since_last,
            "session_message_count": session_message_count,
            "behavioral_state_signals": signals,
        }

    # -- internals ---------------------------------------------------

    @staticmethod
    def _compute_wpm(n_words: int, total_typing_ms: Optional[int]) -> Optional[float]:
        if not n_words or not total_typing_ms or total_typing_ms <= 0:
            return None
        minutes = total_typing_ms / 60000.0
        if minutes <= 0:
            return None
        return n_words / minutes

    def _length_trend(self, session_history: list, current_words: int) -> float:
        lengths: list[float] = []
        for entry in session_history:
            wl = _history_word_len(entry)
            if wl is not None:
                lengths.append(float(wl))
        lengths.append(float(current_words))
        return _linreg_slope(lengths[-5:])

    def _fast_typing_threshold(self) -> float:
        if len(self._typing_speeds) >= MIN_PERCENTILE_SAMPLES:
            p80 = _percentile(self._typing_speeds, 80.0)
            if p80 is not None:
                return p80
        return DEFAULT_FAST_WPM

    def _record(self, store: list[float], value: float) -> None:
        store.append(value)
        if len(store) > self.max_rolling:
            del store[: len(store) - self.max_rolling]

    def _state_signals(
        self,
        *,
        pause: Optional[int],
        backspace_rate: float,
        typing_speed: Optional[float],
        response_length_words: int,
        length_trend: float,
        fast_threshold: float,
    ) -> dict[str, float]:
        fast = (
            typing_speed is not None
            and fast_threshold is not None
            and typing_speed > fast_threshold
        )

        rushing_rules = [
            (fast, 0.4),
            (backspace_rate < 0.05, 0.2),
            (response_length_words < 8, 0.2),
            (pause is not None and pause < 500, 0.2),
        ]
        stuck_rules = [
            (pause is not None and pause > 10000, 0.4),
            (response_length_words < 5, 0.3),
            (length_trend < 0, 0.3),
        ]
        flow_rules = [
            (typing_speed is not None and 30 <= typing_speed <= 70, 0.3),
            (0.05 <= backspace_rate <= 0.15, 0.3),
            (pause is not None and 1000 <= pause <= 5000, 0.2),
            (response_length_words > 20, 0.2),
        ]
        planning_rules = [
            (pause is not None and pause > 5000, 0.5),
            (typing_speed is not None and typing_speed < 40, 0.3),
            (response_length_words > 15, 0.2),
        ]
        frustrated_rules = [
            (backspace_rate > 0.25, 0.3),
            (length_trend < STRONG_NEGATIVE_TREND, 0.3),
            (fast, 0.2),
            (response_length_words < 10, 0.2),
        ]

        return {
            "rushing": _score(rushing_rules),
            "stuck": _score(stuck_rules),
            "flow": _score(flow_rules),
            "planning": _score(planning_rules),
            "frustrated": _score(frustrated_rules),
        }


def dominant_signal(signals: dict[str, float]) -> tuple[str, float]:
    """Return the (signal_name, score) with the highest score, ties broken by
    the fixed ``BEHAVIORAL_SIGNALS`` order for determinism."""
    best_name = BEHAVIORAL_SIGNALS[0]
    best_score = -1.0
    for name in BEHAVIORAL_SIGNALS:
        score = float(signals.get(name, 0.0) or 0.0)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, max(0.0, best_score)


if __name__ == "__main__":
    import json

    extractor = BehavioralFeatureExtractor()
    demo = extractor.extract(
        "I dont get it",
        {
            "pause_before_first_key_ms": 12000,
            "backspace_rate": 0.3,
            "typing_speed_wpm": 95,
        },
        [],
    )
    print(json.dumps(demo, indent=2))
