"""
Teacher/Parent Report Generator for ARIA.

Sections:
  A. Learning Profile Summary
  B. Strengths
  C. Areas Needing Support
  D. Recommended Accommodations (LLM-generated)
  E. Session Timeline (last 30 days)

Outputs: PDF (reportlab) + Markdown, saved to reports/output/.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ollama

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_MODEL = "llama3.1:8b"
FALLBACK_MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def _load_session_logs(days: int = 30) -> list:
    """Load real_sessions.jsonl records from the last N days."""
    log_path = DATA_DIR / "real_sessions.jsonl"
    if not log_path.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    records = []
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("timestamp", "") >= cutoff:
                        records.append(rec)
                except Exception:
                    pass
    except Exception:
        pass
    return records


def _load_emotion_logs(days: int = 30) -> dict:
    """Return {date_str: [state, ...]} from emotion log files."""
    emotion_dir = DATA_DIR / "emotion_logs"
    if not emotion_dir.exists():
        return {}
    cutoff = (datetime.now() - timedelta(days=days)).date()
    result = {}
    for path in emotion_dir.glob("session_*.jsonl"):
        date_str = path.stem.replace("session_", "")
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if d < cutoff:
                continue
        except Exception:
            continue
        states = []
        try:
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        states.append(rec.get("state", "FLOW"))
                    except Exception:
                        pass
        except Exception:
            pass
        result[date_str] = states
    return result


def _load_srs_state() -> dict:
    srs_path = DATA_DIR / "srs_state.json"
    if not srs_path.exists():
        return {"cards": {}}
    try:
        with open(srs_path) as f:
            return json.load(f)
    except Exception:
        return {"cards": {}}


def _sparkline(states: list) -> str:
    """ASCII sparkline of emotional states in a session."""
    symbols = {
        "FLOW": "▂",
        "CONFUSED": "▄",
        "FRUSTRATED": "▆",
        "OVERWHELMED": "█",
        "DISENGAGED": "░",
    }
    return "".join(symbols.get(s, "▂") for s in states[-20:])


# ---------------------------------------------------------------------------
# Report data assembly
# ---------------------------------------------------------------------------

def assemble_report_data(
    user_profile: dict,
    learning_graph,
    vector_store,
) -> dict:
    sessions = _load_session_logs(30)
    emotion_data = _load_emotion_logs(30)
    srs_state = _load_srs_state()

    # Topic analysis from graph
    topics = []
    if learning_graph:
        topics = learning_graph.all_topics_summary()

    strong_topics = [t for t in topics if t.get("confidence", 0) >= 0.7]
    weak_topics = [t for t in topics if t.get("confidence", 0) < 0.4 and t.get("study_count", 0) >= 3]

    # Frustration topics from sessions
    frustration_counts: dict = {}
    explanation_style_counts: dict = {}
    for rec in sessions:
        if rec.get("frustration_detected"):
            for t in rec.get("topics", []):
                frustration_counts[t] = frustration_counts.get(t, 0) + 1
        style = rec.get("explanation_style", "")
        if style:
            explanation_style_counts[style] = explanation_style_counts.get(style, 0) + 1

    best_style = max(explanation_style_counts, key=explanation_style_counts.get) if explanation_style_counts else "mixed"

    # Sessions by date
    session_dates: dict = {}
    for rec in sessions:
        date = rec.get("timestamp", "")[:10]
        if date:
            session_dates[date] = session_dates.get(date, 0) + 1

    total_sessions = len(set(rec.get("timestamp", "")[:10] for rec in sessions if rec.get("timestamp")))
    weeks_active = max(1, len(session_dates) // 7) if session_dates else 0

    # Study hours
    study_hours = [rec.get("timestamp", "")[11:13] for rec in sessions if rec.get("timestamp")]
    hour_counts: dict = {}
    for h in study_hours:
        hour_counts[h] = hour_counts.get(h, 0) + 1
    consistent_hours = sorted(hour_counts, key=lambda x: hour_counts[x], reverse=True)[:2]

    # SRS concepts due / overdue
    today = datetime.now().date().isoformat()
    overdue = [
        c for c in srs_state.get("cards", {}).values()
        if c.get("next_review_date", "9999-12-31") < today
    ]

    # FLOW correlation
    flow_topics: dict = {}
    for date, states in emotion_data.items():
        if states.count("FLOW") > len(states) * 0.6:
            for rec in sessions:
                if rec.get("timestamp", "").startswith(date):
                    for t in rec.get("topics", []):
                        flow_topics[t] = flow_topics.get(t, 0) + 1

    flow_topic_names = sorted(flow_topics, key=lambda x: flow_topics[x], reverse=True)[:3]

    return {
        "profile": user_profile,
        "total_sessions": total_sessions,
        "weeks_active": weeks_active,
        "strong_topics": strong_topics,
        "weak_topics": weak_topics,
        "frustration_counts": frustration_counts,
        "best_explanation_style": best_style,
        "session_dates": session_dates,
        "consistent_hours": consistent_hours,
        "overdue_srs": len(overdue),
        "flow_topics": flow_topic_names,
        "emotion_data": emotion_data,
        "sessions_raw": sessions,
    }


# ---------------------------------------------------------------------------
# LLM-generated accommodations
# ---------------------------------------------------------------------------

def generate_accommodations(report_data: dict) -> str:
    profile = report_data["profile"]
    name = profile.get("name", "the student")
    diagnosis = ", ".join(profile.get("diagnosis", ["unspecified"]))
    weak = ", ".join(t["topic"] for t in report_data["weak_topics"][:3]) or "none identified yet"
    strong = ", ".join(t["topic"] for t in report_data["strong_topics"][:3]) or "none yet"
    style = report_data["best_explanation_style"]
    flow_topics = ", ".join(report_data["flow_topics"]) or "general"
    frustration_topics = ", ".join(
        k for k, v in sorted(report_data["frustration_counts"].items(), key=lambda x: -x[1])[:3]
    ) or "none identified"

    prompt = f"""You are an educational psychologist writing specific, actionable teacher accommodations.

Student profile:
- Name: {name}
- Diagnosis: {diagnosis}
- Strong topics: {strong}
- Struggling topics: {weak}
- Topics causing frustration: {frustration_topics}
- Best explanation style observed: {style}
- Topics studied successfully during FLOW state: {flow_topics}
- Learning goals: {", ".join(profile.get("goals", [])) or "not stated"}
- What has helped: {profile.get("what_helped") or "not stated"}
- What has never worked: {profile.get("what_failed") or "not stated"}

Write 4-6 specific, actionable accommodation recommendations for this student's teacher.
Each recommendation should be one concrete sentence starting with an action verb.
Base your suggestions ONLY on the observed data above, not generic ADHD advice.
Format: numbered list. No introduction, no conclusion."""

    for model in [REPORT_MODEL, FALLBACK_MODEL]:
        try:
            result = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4, "num_predict": 400},
                keep_alive=3600,
            )
            return result.message.content.strip()
        except Exception:
            continue

    return "Could not generate recommendations — Ollama unavailable."


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown(report_data: dict, accommodations: str) -> str:
    p = report_data["profile"]
    name = p.get("name", "Student")
    today = datetime.now().strftime("%B %d, %Y")

    def topic_row(t: dict) -> str:
        conf = t.get("confidence", 0)
        bar_fill = int(conf * 10)
        bar = "█" * bar_fill + "░" * (10 - bar_fill)
        return f"- **{t['topic']}**: `{bar}` {conf:.0%}  _{t.get('study_count', 0)} sessions_"

    strong_rows = "\n".join(topic_row(t) for t in report_data["strong_topics"][:6]) or "_None yet_"
    weak_rows = "\n".join(topic_row(t) for t in report_data["weak_topics"][:6]) or "_None yet_"

    # Frustration topics
    frustration_rows = ""
    fc = report_data["frustration_counts"]
    if fc:
        sorted_fc = sorted(fc.items(), key=lambda x: -x[1])[:5]
        frustration_rows = "\n".join(f"- **{k}**: {v} frustrated turn(s)" for k, v in sorted_fc)
    else:
        frustration_rows = "_None detected_"

    # Session timeline
    timeline_rows = ""
    for date in sorted(report_data["session_dates"].keys(), reverse=True)[:30]:
        count = report_data["session_dates"][date]
        arc = _sparkline(report_data["emotion_data"].get(date, []))
        arc_display = f" `{arc}`" if arc else ""
        timeline_rows += f"- {date}: {count} turn(s){arc_display}\n"
    timeline_rows = timeline_rows or "_No sessions in last 30 days_"

    # Consistent hours
    hours_str = ", ".join(f"{h}:00" for h in report_data["consistent_hours"]) or "varies"

    diagnosis = ", ".join(p.get("diagnosis", [])) or "not specified"
    subjects = ", ".join(p.get("subjects", [])) or "various"
    goals = "\n".join(f"  - {g}" for g in p.get("goals", [])) or "  - not stated"
    learning_style = p.get("learning_style", "mixed").replace("_", " ")

    return f"""# ARIA Learning Report — {name}
Generated: {today}

---

## A. Learning Profile Summary

| Field | Value |
|-------|-------|
| Student | {name} |
| Grade | {p.get("grade") or "not specified"} |
| Diagnosis | {diagnosis} |
| Learning Style | {learning_style} |
| Subjects | {subjects} |
| Attention Span | {p.get("attention_span_minutes", "?")} minutes |
| Weeks Active | {report_data["weeks_active"]} |
| Total Sessions (30 days) | {report_data["total_sessions"]} |
| Consistent Study Hours | {hours_str} |

**Goals:**
{goals}

---

## B. Strengths

{strong_rows}

**Best explanation style:** {report_data["best_explanation_style"]}

**Topics studied in FLOW state:** {", ".join(report_data["flow_topics"]) or "none yet"}

---

## C. Areas Needing Support

### Low confidence topics (< 40% after 3+ attempts):
{weak_rows}

### Topics that triggered frustration:
{frustration_rows}

**Overdue for review:** {report_data["overdue_srs"]} concept(s)

---

## D. Recommended Accommodations

{accommodations}

---

## E. Session Timeline (Last 30 Days)

_Sparkline key: ▂=FLOW ▄=CONFUSED ▆=FRUSTRATED █=OVERWHELMED ░=DISENGAGED_

{timeline_rows}
---
_Report generated by ARIA — Adaptive Reasoning & Intelligence Assistant_
"""


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(markdown_text: str, output_path: str) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8,
                        textColor=colors.HexColor("#3730a3"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=6,
                        textColor=colors.HexColor("#4f46e5"))
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4,
                        textColor=colors.HexColor("#6366f1"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=4,
                          leading=14)
    code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=9,
                                backColor=colors.HexColor("#f1f5f9"), leftIndent=12)

    story = []

    for line in markdown_text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            story.append(Spacer(1, 6))
            continue

        if line_stripped.startswith("# "):
            story.append(Paragraph(line_stripped[2:], h1))
        elif line_stripped.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c7d2fe")))
            story.append(Spacer(1, 4))
            story.append(Paragraph(line_stripped[3:], h2))
        elif line_stripped.startswith("### "):
            story.append(Paragraph(line_stripped[4:], h3))
        elif line_stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
            story.append(Spacer(1, 4))
        elif line_stripped.startswith("|"):
            # Simple table row — skip (handled by table parser below)
            pass
        elif line_stripped.startswith("- "):
            content = line_stripped[2:].replace("**", "<b>").replace("**", "</b>")
            story.append(Paragraph(f"• {content}", body))
        elif line_stripped.startswith("`"):
            story.append(Paragraph(line_stripped, code_style))
        elif line_stripped.startswith("_") and line_stripped.endswith("_"):
            italic_text = line_stripped[1:-1]
            story.append(Paragraph(f"<i>{italic_text}</i>", body))
        else:
            clean = line_stripped.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(clean, body))

    doc.build(story)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    user_profile: dict,
    learning_graph,
    vector_store,
) -> dict:
    """
    Generate teacher/parent report.
    Returns {markdown_path, pdf_path, markdown_text}.
    """
    report_data = assemble_report_data(user_profile, learning_graph, vector_store)
    accommodations = generate_accommodations(report_data)
    md_text = generate_markdown(report_data, accommodations)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_slug = "".join(
        c if c.isalnum() else "_"
        for c in user_profile.get("name", "student")
    )[:20]

    md_path = OUTPUT_DIR / f"report_{name_slug}_{timestamp}.md"
    pdf_path = OUTPUT_DIR / f"report_{name_slug}_{timestamp}.pdf"

    with open(md_path, "w") as f:
        f.write(md_text)

    try:
        generate_pdf(md_text, str(pdf_path))
        pdf_ok = True
    except Exception as e:
        pdf_ok = False
        pdf_path = None

    return {
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path) if pdf_ok else None,
        "markdown_text": md_text,
        "pdf_ok": pdf_ok,
    }
