"""
Checkpoint system for ARIA — working memory offload across sessions.

Every session tracks:
  - Current topic, concept, step within that concept
  - Last question asked and answer given
  - Confidence level at pause
  - Timestamp

Checkpoints saved to data/checkpoints/{topic}_{date}.json
Loaded on startup; recent ones (< 7 days) are offered to the user.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(topic: str, date_str: str) -> str:
    safe_topic = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)
    safe_topic = safe_topic[:40].strip().replace(" ", "_")
    return f"{safe_topic}_{date_str}.json"


def save_checkpoint(
    topic: str,
    concept: str,
    step: int,
    last_question: str,
    last_answer: str,
    confidence: float,
    source_file: Optional[str] = None,
    source_page: Optional[int] = None,
) -> str:
    """Save a checkpoint and return its filename."""
    date_str = datetime.now().date().isoformat()
    filename = _safe_filename(topic, date_str)
    path = CHECKPOINTS_DIR / filename

    data = {
        "topic": topic,
        "concept": concept,
        "step": step,
        "last_question": last_question,
        "last_answer": last_answer,
        "confidence": round(confidence, 3),
        "timestamp": datetime.now().isoformat(),
        "source_file": source_file,
        "source_page": source_page,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return filename


def get_recent_checkpoints(max_age_days: int = 7) -> list:
    """Return all checkpoints newer than max_age_days, sorted newest first."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    results = []

    for path in CHECKPOINTS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data["timestamp"])
            if ts >= cutoff:
                data["_filename"] = path.name
                results.append(data)
        except Exception:
            continue

    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


def get_latest_checkpoint() -> Optional[dict]:
    """Return the single most recent checkpoint within 7 days, or None."""
    recent = get_recent_checkpoints(max_age_days=7)
    return recent[0] if recent else None


def load_checkpoint(filename: str) -> Optional[dict]:
    path = CHECKPOINTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def delete_checkpoint(filename: str) -> None:
    path = CHECKPOINTS_DIR / filename
    if path.exists():
        path.unlink()


def format_checkpoint_resume_message(checkpoint: dict) -> str:
    """Return the greeting ARIA shows when offering to resume."""
    topic = checkpoint.get("topic", "that topic")
    concept = checkpoint.get("concept", "that concept")
    step = checkpoint.get("step", 1)
    conf = checkpoint.get("confidence", 0.5)
    ts = checkpoint.get("timestamp", "")[:10]
    source = checkpoint.get("source_file")
    page = checkpoint.get("source_page")

    location = f"step {step} of **{concept}**"
    if source and page:
        location += f" (from *{source}*, page {page})"

    conf_label = "struggling a bit" if conf < 0.4 else "making progress" if conf < 0.7 else "doing well"

    msg = (
        f"Last time ({ts}) we were working on **{topic}** — {location}. "
        f"You were {conf_label} (confidence {conf:.0%}). "
        f"Want to pick up there or start fresh?"
    )
    return msg
