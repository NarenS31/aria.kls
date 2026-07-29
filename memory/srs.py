"""
Spaced Repetition System (SRS) for ARIA — ADHD-modified SM-2 algorithm.

Each concept card:
  ease_factor: float (default 2.5)
  interval: int (days until next review)
  repetitions: int
  next_review_date: ISO date string
  last_emotion_state: str (FLOW, CONFUSED, FRUSTRATED, DISENGAGED, OVERWHELMED)
  last_score: int (0-5, SM-2 quality)
  topic: str
  concept: str
  created_at: str

ADHD modification:
  - If last emotion was FRUSTRATED or OVERWHELMED, halve the interval.
  - If last emotion was FLOW, standard SM-2 applies.

State saved to data/srs_state.json.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SRS_PATH = DATA_DIR / "srs_state.json"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if SRS_PATH.exists():
        try:
            with open(SRS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"cards": {}}


def _save_state(state: dict) -> None:
    with open(SRS_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# SM-2 core
# ---------------------------------------------------------------------------

def _sm2_update(card: dict, quality: int, emotion: str) -> dict:
    """
    quality: 0-5 (5=perfect, 3=correct with effort, <3=failed)
    Returns updated card dict.
    """
    ease = card.get("ease_factor", 2.5)
    interval = card.get("interval", 1)
    reps = card.get("repetitions", 0)

    if quality >= 3:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ease)
        reps += 1
    else:
        # Failed — reset
        reps = 0
        interval = 1

    # Update ease factor
    ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(1.3, ease)

    # ADHD modification: if frustrated/overwhelmed, review sooner
    if emotion in ("FRUSTRATED", "OVERWHELMED") and quality >= 3:
        interval = max(1, interval // 2)

    next_date = (datetime.now() + timedelta(days=interval)).date().isoformat()

    card["ease_factor"] = round(ease, 3)
    card["interval"] = interval
    card["repetitions"] = reps
    card["next_review_date"] = next_date
    card["last_emotion_state"] = emotion
    card["last_score"] = quality
    card["last_reviewed"] = datetime.now().isoformat()
    return card


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_card(topic: str, concept: str) -> dict:
    """Create card if it doesn't exist; return it."""
    state = _load_state()
    card_id = f"{topic}::{concept}"
    if card_id not in state["cards"]:
        state["cards"][card_id] = {
            "card_id": card_id,
            "topic": topic,
            "concept": concept,
            "ease_factor": 2.5,
            "interval": 1,
            "repetitions": 0,
            "next_review_date": datetime.now().date().isoformat(),
            "last_emotion_state": "FLOW",
            "last_score": None,
            "last_reviewed": None,
            "created_at": datetime.now().isoformat(),
        }
        _save_state(state)
    return state["cards"][card_id]


def record_review(topic: str, concept: str, quality: int, emotion: str = "FLOW") -> dict:
    """Record a review result (quality 0-5) and return updated card."""
    state = _load_state()
    card_id = f"{topic}::{concept}"
    card = state["cards"].get(card_id) or ensure_card(topic, concept)
    card = _sm2_update(card, quality, emotion)
    state["cards"][card_id] = card
    _save_state(state)
    return card


def get_due_cards(as_of: Optional[str] = None) -> List[dict]:
    """Return all cards due for review today or overdue."""
    today = as_of or datetime.now().date().isoformat()
    state = _load_state()
    due = [
        card for card in state["cards"].values()
        if card.get("next_review_date", "9999-12-31") <= today
    ]
    due.sort(key=lambda c: c.get("next_review_date", "9999-12-31"))
    return due


def get_all_cards() -> List[dict]:
    return list(_load_state()["cards"].values())


def skip_card(topic: str, concept: str) -> None:
    """Reschedule a card for tomorrow (e.g. user was frustrated during review)."""
    state = _load_state()
    card_id = f"{topic}::{concept}"
    if card_id in state["cards"]:
        tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
        state["cards"][card_id]["next_review_date"] = tomorrow
        _save_state(state)


def due_count() -> int:
    return len(get_due_cards())


def format_srs_prompt(card: dict) -> str:
    """Generate a natural-language retrieval practice question for ARIA to weave in."""
    topic = card["topic"]
    concept = card["concept"]
    days_since = 0
    if card.get("last_reviewed"):
        try:
            dt = datetime.fromisoformat(card["last_reviewed"])
            days_since = (datetime.now() - dt).days
        except Exception:
            pass

    if days_since == 0:
        when = "just now"
    elif days_since == 1:
        when = "yesterday"
    else:
        when = f"{days_since} days ago"

    return (
        f"Quick check — you learned **{concept}** (under {topic}) {when}. "
        f"Can you explain it or give an example? Take your time."
    )
