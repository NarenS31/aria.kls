"""
Proactive nudge system and nightly learning loop for ARIA.

APScheduler runs two background jobs:
  1. Nudge checker (every 30 min) — goal-aware, tracks which types the user responds to
  2. Nightly loop (3:00 AM) — summarises the day, updates graph confidence scores

Rules:
  - Max 3 nudges per day total
  - Track which nudge types the user responds to; prioritise those
  - Nudges are surfaced through a shared queue that the Gradio UI reads
"""

import json
import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path

import ollama
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

nudge_queue: queue.Queue = queue.Queue()
_nightly_lock = threading.Lock()

NUDGE_MODEL = "llama3.2:3b"


# ------------------------------------------------------------------
# Daily nudge counter
# ------------------------------------------------------------------

_daily_nudge_count: dict[str, int] = {}  # date_str -> count

def _nudges_today() -> int:
    today = datetime.now().date().isoformat()
    return _daily_nudge_count.get(today, 0)

def _record_nudge() -> None:
    today = datetime.now().date().isoformat()
    _daily_nudge_count[today] = _daily_nudge_count.get(today, 0) + 1


# ------------------------------------------------------------------
# Nudge generation
# ------------------------------------------------------------------

def _hour_label(h: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    display = h if h <= 12 else h - 12
    if display == 0:
        display = 12
    return f"{display}{suffix}"


def _generate_nudge(prompt_text: str) -> str:
    try:
        result = ollama.chat(
            model=NUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ARIA, a warm study buddy for neurodivergent students. "
                        "Write ONE short nudge (max 2 sentences). No emojis. "
                        "Be specific, warm, and never sycophantic. "
                        "Do not start with 'Great', 'Certainly', 'Of course', or 'Absolutely'."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            options={"temperature": 0.8, "num_predict": 80},
            keep_alive=3600,
        )
        return result.message.content.strip()
    except Exception:
        return "Time to take a small step on something you've been working on."


# ------------------------------------------------------------------
# Nudge preference tracking
# ------------------------------------------------------------------

def record_nudge_response(nudge_type: str, profile: dict) -> None:
    """Call when user sends a message after a nudge — signals the nudge worked."""
    stats = profile.setdefault("nudge_stats", {})
    stats[nudge_type] = stats.get(nudge_type, 0) + 1


def _preferred_nudge_types(profile: dict) -> list[str]:
    stats = profile.get("nudge_stats", {})
    return sorted(stats, key=lambda k: stats[k], reverse=True)


# ------------------------------------------------------------------
# Exam date awareness
# ------------------------------------------------------------------

def _days_until_exam(exam_dates: list[str]) -> list[tuple[str, int]]:
    """Return list of (exam_label, days_remaining) for upcoming exams."""
    today = datetime.now().date()
    results = []
    for entry in exam_dates:
        # Try to parse a date from each entry string
        for fmt in ("%B %d", "%b %d", "%m/%d", "%m/%d/%Y"):
            try:
                parts = entry.strip()
                # strip exam label prefix (e.g. "ACT June 14" -> try "June 14")
                words = parts.split()
                for start in range(len(words)):
                    candidate = " ".join(words[start:])
                    try:
                        dt = datetime.strptime(candidate, fmt).replace(year=today.year).date()
                        if dt >= today:
                            days = (dt - today).days
                            results.append((parts, days))
                            break
                    except ValueError:
                        continue
                break
            except Exception:
                continue
    return results


# ------------------------------------------------------------------
# Main nudge logic
# ------------------------------------------------------------------

def check_and_nudge(vector_store, learning_graph, user_profile: dict) -> None:
    """Runs every 30 minutes. Max 3 nudges per day."""
    if _nudges_today() >= 3:
        return

    now = datetime.now()
    hour = now.hour
    name = user_profile.get("name", "friend")
    subjects = user_profile.get("subjects", [])
    goals = user_profile.get("goals", [])
    exam_dates = user_profile.get("exam_dates", [])
    preferred = _preferred_nudge_types(user_profile)

    # --- Priority 1: Exam coming up in ≤7 days (ramp up frequency)
    if exam_dates:
        upcoming = _days_until_exam(exam_dates)
        for label, days in upcoming:
            if days <= 7:
                # Only send during study hours or peak hours
                peak = learning_graph.peak_focus_hours()
                if hour in peak or (14 <= hour <= 22):
                    goal_str = f"Goal: {goals[0]}" if goals else ""
                    msg = _generate_nudge(
                        f"{name} has '{label}' in {days} day{'s' if days != 1 else ''}. "
                        f"{goal_str}. Nudge them to do a focused practice session on the most relevant topic."
                    )
                    nudge_queue.put({"type": "exam_countdown", "message": msg, "timestamp": now.isoformat()})
                    _record_nudge()
                    return

    # --- Priority 2: Topic unvisited for 3+ days and in goals
    goals_lower = " ".join(goals).lower()
    struggling = learning_graph.struggling_topics()
    for t in struggling[:2]:
        info = learning_graph.get_topic_info(t["topic"])
        if not info or not info.get("last_studied"):
            continue
        try:
            last = datetime.fromisoformat(info["last_studied"])
            days_away = (now - last).days
            if days_away >= 3:
                in_goals = any(word in goals_lower for word in t["topic"].lower().split())
                if info.get("goal_topic") or in_goals:
                    msg = _generate_nudge(
                        f"{name} hasn't touched '{t['topic']}' in {days_away} days "
                        f"and it's in their goals. Confidence is {t['confidence']:.0%}. "
                        f"Suggest one concrete micro-step to revisit it."
                    )
                    nudge_queue.put({"type": "goal_topic_gap", "topic": t["topic"], "message": msg, "timestamp": now.isoformat()})
                    _record_nudge()
                    return
        except (ValueError, TypeError):
            pass

    # --- Priority 3: Peak hour reminder (personalised to study history)
    peak_hours = learning_graph.peak_focus_hours()
    if peak_hours and hour in peak_hours:
        msg = _generate_nudge(
            f"{name} historically focuses best at {_hour_label(hour)}. "
            f"They're studying {', '.join(subjects[:2]) if subjects else 'various topics'}. "
            f"Motivate them to use this peak window."
        )
        nudge_queue.put({"type": "peak_hour", "message": msg, "timestamp": now.isoformat()})
        _record_nudge()
        return

    # --- Priority 4: Morning tip based on yesterday's struggles (9 AM)
    if hour == 9 and now.minute < 30:
        recent = vector_store.get_recent_turns(n=10)
        yesterday = (now - timedelta(days=1)).date().isoformat()
        struggle_topics = []
        for turn in recent:
            ts = turn["meta"].get("timestamp", "")[:10]
            if ts == yesterday and turn["meta"].get("frustration") == "True":
                try:
                    struggle_topics.extend(json.loads(turn["meta"].get("topics", "[]")))
                except Exception:
                    pass
        struggle_topics = list(set(struggle_topics))[:2]
        topic_str = f"specifically '{struggle_topics[0]}'" if struggle_topics else f"in {subjects[0]}" if subjects else "on your studies"
        msg = _generate_nudge(
            f"Morning tip for {name} who struggled {topic_str} yesterday. "
            f"One specific, actionable tip to approach it differently today."
        )
        nudge_queue.put({"type": "morning_tip", "message": msg, "timestamp": now.isoformat()})
        _record_nudge()
        return

    # --- Priority 5: Evening recap (8 PM)
    if hour == 20 and now.minute < 30:
        recent = vector_store.get_recent_turns(n=5)
        today_str = now.date().isoformat()
        topics_today = []
        for turn in recent:
            if turn["meta"].get("timestamp", "").startswith(today_str):
                try:
                    topics_today.extend(json.loads(turn["meta"].get("topics", "[]")))
                except Exception:
                    pass
        topics_today = list(set(topics_today))[:2]
        if topics_today:
            msg = _generate_nudge(
                f"Evening recap for {name}. They studied {', '.join(topics_today)} today. "
                f"Suggest a gentle 5-minute wind-down review activity and a topic to try tomorrow."
            )
            nudge_queue.put({"type": "evening_recap", "message": msg, "timestamp": now.isoformat()})
            _record_nudge()


# ------------------------------------------------------------------
# Nightly learning loop
# ------------------------------------------------------------------

def nightly_loop(vector_store, learning_graph, user_profile: dict) -> None:
    if not _nightly_lock.acquire(blocking=False):
        return
    try:
        _run_nightly(vector_store, learning_graph, user_profile)
    finally:
        _nightly_lock.release()


def _run_nightly(vector_store, learning_graph, user_profile: dict) -> None:
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    date_str = yesterday.date().isoformat()

    turns = vector_store.get_turns_since(yesterday.replace(hour=0, minute=0, second=0))
    if not turns:
        return

    digest_parts = []
    topic_counts: dict[str, int] = {}
    frustration_turns = 0

    for turn in turns:
        digest_parts.append(turn["text"][:300])
        try:
            for t in json.loads(turn["meta"].get("topics", "[]")):
                topic_counts[t] = topic_counts.get(t, 0) + 1
        except Exception:
            pass
        if turn["meta"].get("frustration") == "True":
            frustration_turns += 1

    digest = "\n---\n".join(digest_parts[:20])
    name = user_profile.get("name", "the student")
    goals = user_profile.get("goals", [])
    goals_str = ", ".join(goals[:3]) if goals else "none stated"

    summary_prompt = f"""Analyze today's tutoring session for {name} (goals: {goals_str}).

CONVERSATIONS:
{digest}

Write a concise summary (max 120 words):
1. Topics studied today
2. What clicked vs what caused confusion
3. One specific recommendation for tomorrow

Be warm and specific."""

    try:
        result = ollama.chat(
            model=NUDGE_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            options={"temperature": 0.4, "num_predict": 250},
            keep_alive=3600,
        )
        summary = result.message.content.strip()
    except Exception as e:
        summary = f"Could not generate summary: {e}"

    vector_store.store_summary(summary, date_str)

    if turns:
        frustration_ratio = frustration_turns / len(turns)
        for topic, count in topic_counts.items():
            if count >= 2:
                delta = -0.05 if frustration_ratio > 0.4 else 0.02
                learning_graph.update_confidence(topic, delta)

    learning_graph.save()

    log_path = DATA_DIR / "nightly_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "date": date_str,
            "turns_processed": len(turns),
            "topics": topic_counts,
            "frustration_ratio": round(frustration_turns / max(len(turns), 1), 2),
            "summary_length": len(summary),
        }) + "\n")


# ------------------------------------------------------------------
# Initiation Trigger — summon system
# ------------------------------------------------------------------

class InitiationTrigger:
    """
    At peak focus hours, sends a system notification AND queues a nudge with
    ONE specific actionable task derived from:
      - Most overdue SRS card
      - Most struggled topic in last 3 days
      - Most overdue checkpoint

    Tracks which triggers resulted in a real study session (5+ min of chat).
    Optimises timing based on response history.
    """

    TRIGGER_LOG = DATA_DIR / "initiation_triggers.jsonl"

    def __init__(self, learning_graph, user_profile: dict):
        self.lg = learning_graph
        self.profile = user_profile
        self._follow_up_sent: dict[str, bool] = {}  # date -> bool

    def _load_log(self) -> list:
        if not self.TRIGGER_LOG.exists():
            return []
        records = []
        try:
            with open(self.TRIGGER_LOG) as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
        return records

    def _log_trigger(self, task: str, trigger_type: str) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "trigger_type": trigger_type,
            "resulted_in_session": False,
        }
        with open(self.TRIGGER_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def mark_session_started(self) -> None:
        """Call when user has been active for 5+ minutes — marks last trigger as successful."""
        records = self._load_log()
        if not records:
            return
        last = records[-1]
        if not last.get("resulted_in_session"):
            last["resulted_in_session"] = True
            try:
                with open(self.TRIGGER_LOG, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
            except Exception:
                pass

    def _best_task(self) -> tuple[str, str]:
        """Return (task_sentence, trigger_type)."""
        from memory.srs import get_due_cards
        from memory.checkpoints import get_latest_checkpoint

        # Priority 1: Overdue SRS card
        due = get_due_cards()
        if due:
            card = due[0]
            return (
                f"Review one concept: **{card['concept']}** (under {card['topic']}).",
                "srs_due",
            )

        # Priority 2: Most struggled topic in last 3 days
        struggling = self.lg.struggling_topics()
        if struggling:
            topic = struggling[0]["topic"]
            return (
                f"Work through one practice problem on **{topic}**.",
                "struggling_topic",
            )

        # Priority 3: Resume latest checkpoint
        ckpt = get_latest_checkpoint()
        if ckpt:
            return (
                f"Pick up where you left off: **{ckpt['concept']}**, step {ckpt['step']}.",
                "checkpoint_resume",
            )

        subjects = self.profile.get("subjects", ["your studies"])
        return (f"Spend 10 minutes on **{subjects[0]}**.", "general")

    def _already_triggered_today(self) -> bool:
        today = datetime.now().date().isoformat()
        records = self._load_log()
        return any(r["timestamp"][:10] == today for r in records)

    def _already_followed_up_today(self) -> bool:
        today = datetime.now().date().isoformat()
        return self._follow_up_sent.get(today, False)

    def check_and_trigger(self) -> None:
        """Call every 30 minutes. Sends initiation nudge during peak hours."""
        now = datetime.now()
        hour = now.hour
        peak_hours = self.lg.peak_focus_hours() if hasattr(self.lg, "peak_focus_hours") else []
        study_hours = self.profile.get("study_hours", [])
        active_hours = set(peak_hours) | set(study_hours)

        if hour not in active_hours:
            return

        if self._already_triggered_today():
            # Send one follow-up 10 minutes after first trigger if no session started
            today = datetime.now().date().isoformat()
            if not self._already_followed_up_today():
                records = self._load_log()
                today_records = [r for r in records if r["timestamp"][:10] == today]
                if today_records and not today_records[-1].get("resulted_in_session"):
                    first_ts = datetime.fromisoformat(today_records[-1]["timestamp"])
                    if (now - first_ts).total_seconds() >= 600:
                        self._follow_up_sent[today] = True
                        task, _ = self._best_task()
                        name = self.profile.get("name", "")
                        msg = _generate_nudge(
                            f"Send one gentle follow-up to {name} who hasn't responded "
                            f"to the earlier study nudge. Specific task: {task}"
                        )
                        nudge_queue.put({
                            "type": "initiation_followup",
                            "message": msg,
                            "timestamp": now.isoformat(),
                        })
                        _record_nudge()
            return

        task, trigger_type = self._best_task()
        name = self.profile.get("name", "")
        msg = _generate_nudge(
            f"Summon {name} to study right now. One sentence. "
            f"Tell them specifically: {task} Do not say anything except the task."
        )
        nudge_queue.put({
            "type": "initiation_trigger",
            "message": f"Time to study → {task}",
            "timestamp": now.isoformat(),
        })
        self._log_trigger(task, trigger_type)
        _record_nudge()

        # macOS notification (best-effort)
        try:
            import subprocess
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{task}" with title "ARIA — Time to Study"'],
                timeout=3,
                capture_output=True,
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# Scheduler factory
# ------------------------------------------------------------------

class NudgeScheduler:
    def __init__(self, vector_store, learning_graph, user_profile: dict):
        self.vs = vector_store
        self.lg = learning_graph
        self.profile = user_profile
        self.scheduler = BackgroundScheduler(timezone="America/Los_Angeles")
        self.initiation = InitiationTrigger(learning_graph, user_profile)
        self._setup()

    def _setup(self) -> None:
        self.scheduler.add_job(
            func=self._nudge_job,
            trigger=IntervalTrigger(minutes=30),
            id="nudge_check",
            replace_existing=True,
            misfire_grace_time=120,
        )
        self.scheduler.add_job(
            func=self._nightly_job,
            trigger=CronTrigger(hour=3, minute=0),
            id="nightly_loop",
            replace_existing=True,
            misfire_grace_time=3600,
        )

    def _nudge_job(self) -> None:
        try:
            check_and_nudge(self.vs, self.lg, self.profile)
        except Exception:
            pass
        try:
            self.initiation.check_and_trigger()
        except Exception:
            pass

    def _nightly_job(self) -> None:
        try:
            nightly_loop(self.vs, self.lg, self.profile)
        except Exception:
            pass

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def run_nightly_now(self) -> None:
        threading.Thread(
            target=nightly_loop,
            args=(self.vs, self.lg, self.profile),
            daemon=True,
        ).start()
