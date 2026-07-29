"""
PDF report generation for the ARIA teacher dashboard.
Generates per-student and class-wide PDF reports, and CSV exports.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


class ReportGenerator:
    def __init__(self, output_dir: str = "data/reports/output"):
        path = Path(output_dir)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / output_dir
        self.output_dir = path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    def _get_styles(self):
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle(
            "H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8,
            textColor=colors.HexColor("#3730a3"),
        )
        h2 = ParagraphStyle(
            "H2", parent=styles["Heading2"], fontSize=14, spaceAfter=6,
            textColor=colors.HexColor("#4f46e5"),
        )
        h3 = ParagraphStyle(
            "H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4,
            textColor=colors.HexColor("#6366f1"),
        )
        body = ParagraphStyle(
            "Body", parent=styles["Normal"], fontSize=10, spaceAfter=4, leading=14,
        )
        small = ParagraphStyle(
            "Small", parent=styles["Normal"], fontSize=9, spaceAfter=2, leading=12,
            textColor=colors.HexColor("#475569"),
        )
        return {"h1": h1, "h2": h2, "h3": h3, "body": body, "small": small}

    def _confidence_to_color(self, confidence: float) -> colors.Color:
        if confidence < 0.4:
            return colors.HexColor("#ef4444")   # red
        elif confidence < 0.7:
            return colors.HexColor("#f97316")   # orange
        else:
            return colors.HexColor("#22c55e")   # green

    def _confidence_bar(self, confidence: float) -> str:
        filled = int(confidence * 10)
        return "█" * filled + "░" * (10 - filled)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_slug(self, name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name)[:30]

    # ------------------------------------------------------------------
    # Student report
    # ------------------------------------------------------------------

    def generate_student_report(
        self,
        student_name: str,
        profile: dict,
        confidence_summary: dict,
        emotion_trend: list,
        session_stats: dict,
    ) -> str:
        filename = f"student_{self._safe_slug(student_name)}_{self._timestamp()}.pdf"
        filepath = str(self.output_dir / filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        styles = self._get_styles()
        story = []
        today = datetime.now().strftime("%B %d, %Y")

        # Title
        story.append(Paragraph(f"ARIA Student Report — {student_name}", styles["h1"]))
        story.append(Paragraph(f"Generated: {today}", styles["small"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c7d2fe")))
        story.append(Spacer(1, 10))

        # --- Section 1: Student Profile ---
        story.append(Paragraph("1. Student Profile", styles["h2"]))
        goals_str = ", ".join(profile.get("goals", [])) or "not stated"
        profile_data = [
            ["Field", "Value"],
            ["Name", profile.get("name", student_name)],
            ["Grade", profile.get("grade", "—")],
            ["Diagnosis", profile.get("diagnosis", "—")],
            ["Learning Style", profile.get("learning_style", "—")],
            ["Attention Span", f"{profile.get('attention_span_minutes', '—')} minutes"],
            ["Goals", goals_str],
        ]
        profile_table = Table(profile_data, colWidths=[2 * inch, 4.5 * inch])
        profile_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(profile_table)
        story.append(Spacer(1, 14))

        # --- Section 2: Subject Confidence Heatmap ---
        story.append(Paragraph("2. Subject Confidence Heatmap", styles["h2"]))
        if confidence_summary:
            conf_data = [["Topic", "Confidence", "Level"]]
            for topic, conf in sorted(confidence_summary.items(), key=lambda x: x[1]):
                bar = self._confidence_bar(conf)
                conf_data.append([topic, f"{conf:.0%}", bar])
            conf_table = Table(conf_data, colWidths=[2.5 * inch, 1 * inch, 3 * inch])
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
            for row_idx, (topic, conf) in enumerate(sorted(confidence_summary.items(), key=lambda x: x[1]), start=1):
                cell_color = self._confidence_to_color(conf)
                style_cmds.append(("BACKGROUND", (2, row_idx), (2, row_idx), cell_color))
                style_cmds.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), colors.white))
            conf_table.setStyle(TableStyle(style_cmds))
            story.append(conf_table)
        else:
            story.append(Paragraph("No confidence data available yet.", styles["body"]))
        story.append(Spacer(1, 14))

        # --- Section 3: Emotional Trend Summary ---
        story.append(Paragraph("3. Emotional Trend Summary (Last 7 Days)", styles["h2"]))
        emotion_counts: dict = {}
        for _, emotion in emotion_trend:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        if emotion_counts:
            em_data = [["Emotion State", "Count"]]
            for state, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
                em_data.append([state, str(count)])
            em_table = Table(em_data, colWidths=[3 * inch, 1.5 * inch])
            em_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(em_table)
        else:
            story.append(Paragraph("No emotional trend data available for this period.", styles["body"]))
        story.append(Spacer(1, 14))

        # --- Section 4: Session Statistics ---
        story.append(Paragraph("4. Session Statistics", styles["h2"]))
        total_sessions = session_stats.get("total_sessions", 0)
        total_hours = session_stats.get("total_hours", 0.0)
        avg_duration = session_stats.get("avg_session_duration_minutes", 0.0)
        stats_data = [
            ["Metric", "Value"],
            ["Total Sessions", str(total_sessions)],
            ["Total Hours", f"{total_hours:.1f} hrs"],
            ["Avg Session Duration", f"{avg_duration:.0f} min"],
        ]
        stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 14))

        # --- Section 5: Recommendations ---
        story.append(Paragraph("5. Recommendations", styles["h2"]))
        recommendations = self._generate_recommendations(confidence_summary, emotion_counts, profile)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles["body"]))
        story.append(Spacer(1, 10))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
        story.append(Paragraph("Report generated by ARIA — Adaptive Reasoning & Intelligence Assistant", styles["small"]))

        doc.build(story)
        return filepath

    def _generate_recommendations(
        self,
        confidence_summary: dict,
        emotion_counts: dict,
        profile: dict,
    ) -> list:
        recs = []
        if confidence_summary:
            weak = [t for t, c in confidence_summary.items() if c < 0.4]
            strong = [t for t, c in confidence_summary.items() if c >= 0.7]
            if weak:
                recs.append(
                    f"Prioritize additional review sessions for: {', '.join(weak[:3])} "
                    f"where confidence remains below 40%."
                )
            if strong:
                recs.append(
                    f"Build on demonstrated strength in {', '.join(strong[:2])} "
                    f"by introducing more advanced material in those areas."
                )
        frustrated_count = emotion_counts.get("FRUSTRATED", 0) + emotion_counts.get("OVERWHELMED", 0)
        flow_count = emotion_counts.get("FLOW", 0)
        if frustrated_count > flow_count:
            recs.append(
                "Reduce task complexity and increase scaffolding — frustration/overwhelm states "
                "outweigh FLOW states in recent sessions."
            )
        elif flow_count > 0:
            recs.append(
                "Maintain current pacing; FLOW states indicate effective task-difficulty matching."
            )
        attn = profile.get("attention_span_minutes", 20)
        if isinstance(attn, int) and attn <= 10:
            recs.append(
                f"Keep individual task blocks under {attn} minutes and use frequent "
                f"micro-breaks to accommodate the {attn}-minute attention span."
            )
        while len(recs) < 3:
            recs.append(
                "Continue monitoring session patterns to refine personalized learning goals."
            )
        return recs[:3]

    # ------------------------------------------------------------------
    # Class report
    # ------------------------------------------------------------------

    def generate_class_report(self, students_data: list) -> str:
        filename = f"class_report_{self._timestamp()}.pdf"
        filepath = str(self.output_dir / filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        styles = self._get_styles()
        story = []
        today = datetime.now().strftime("%B %d, %Y")

        story.append(Paragraph("ARIA Class Report", styles["h1"]))
        story.append(Paragraph(f"Generated: {today}  |  Students: {len(students_data)}", styles["small"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c7d2fe")))
        story.append(Spacer(1, 10))

        # --- Section 1: Class Overview table ---
        story.append(Paragraph("1. Class Overview — Confidence by Topic", styles["h2"]))

        # Collect all topics across students
        all_topics: set = set()
        for sd in students_data:
            all_topics.update(sd.get("confidence_summary", {}).keys())
        all_topics_list = sorted(all_topics)

        if all_topics_list and students_data:
            header = ["Student"] + all_topics_list
            table_data = [header]
            for sd in students_data:
                name = sd.get("name", "Unknown")
                row = [name]
                for topic in all_topics_list:
                    conf = sd.get("confidence_summary", {}).get(topic)
                    if conf is None:
                        row.append("—")
                    else:
                        row.append(f"{conf:.0%}")
                table_data.append(row)

            col_w = min(1.2 * inch, 6.5 * inch / max(len(all_topics_list) + 1, 1))
            col_widths = [1.5 * inch] + [col_w] * len(all_topics_list)
            overview_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ]
            for row_idx, sd in enumerate(students_data, start=1):
                for col_idx, topic in enumerate(all_topics_list, start=1):
                    conf = sd.get("confidence_summary", {}).get(topic)
                    if conf is not None:
                        cell_color = self._confidence_to_color(conf)
                        style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), cell_color))
                        style_cmds.append(("TEXTCOLOR", (col_idx, row_idx), (col_idx, row_idx), colors.white))
            overview_table.setStyle(TableStyle(style_cmds))
            story.append(overview_table)
        else:
            story.append(Paragraph("No topic confidence data available.", styles["body"]))
        story.append(Spacer(1, 14))

        # --- Section 2: Struggling Topics ---
        story.append(Paragraph("2. Struggling Topics (Class Average Confidence < 40%)", styles["h2"]))
        topic_avg: dict = {}
        for topic in all_topics_list:
            confs = [
                sd["confidence_summary"][topic]
                for sd in students_data
                if topic in sd.get("confidence_summary", {})
            ]
            if confs:
                topic_avg[topic] = sum(confs) / len(confs)
        struggling = {t: avg for t, avg in topic_avg.items() if avg < 0.4}
        if struggling:
            str_data = [["Topic", "Class Avg Confidence"]]
            for topic, avg in sorted(struggling.items(), key=lambda x: x[1]):
                str_data.append([topic, f"{avg:.0%}"])
            str_table = Table(str_data, colWidths=[3.5 * inch, 2 * inch])
            str_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ef4444")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fef2f2"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(str_table)
        else:
            story.append(Paragraph("No topics with class average below 40%.", styles["body"]))
        story.append(Spacer(1, 14))

        # --- Section 3: Individual Summaries ---
        story.append(Paragraph("3. Individual Student Summaries", styles["h2"]))
        for sd in students_data:
            name = sd.get("name", "Unknown")
            conf = sd.get("confidence_summary", {})
            stats = sd.get("session_stats", {})
            emotion_trend = sd.get("emotion_trend", [])
            emotion_counts: dict = {}
            for _, state in emotion_trend:
                emotion_counts[state] = emotion_counts.get(state, 0) + 1

            top_topic = max(conf, key=lambda t: conf[t]) if conf else "N/A"
            weak_topic = min(conf, key=lambda t: conf[t]) if conf else "N/A"
            top_conf = f"{conf[top_topic]:.0%}" if conf else "—"
            weak_conf = f"{conf[weak_topic]:.0%}" if conf else "—"
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "N/A"
            sessions = stats.get("total_sessions", 0)

            summary = (
                f"<b>{name}</b>: {sessions} session(s) logged. "
                f"Strongest topic: {top_topic} ({top_conf}). "
                f"Needs support: {weak_topic} ({weak_conf}). "
                f"Dominant emotional state: {dominant_emotion}."
            )
            story.append(Paragraph(summary, styles["body"]))
            story.append(Spacer(1, 4))

        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
        story.append(Paragraph("Report generated by ARIA — Adaptive Reasoning & Intelligence Assistant", styles["small"]))

        doc.build(story)
        return filepath

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self, students_data: list) -> str:
        filename = f"class_data_{self._timestamp()}.csv"
        filepath = str(self.output_dir / filename)

        rows = []
        for sd in students_data:
            name = sd.get("name", "Unknown")
            conf_summary = sd.get("confidence_summary", {})
            stats = sd.get("session_stats", {})
            last_active = stats.get("last_active", "")
            total_sessions = stats.get("total_sessions", 0)

            if conf_summary:
                for topic, confidence in conf_summary.items():
                    # Infer subject as first subject from profile if available, else blank
                    subject = sd.get("subjects", [""])[0] if sd.get("subjects") else ""
                    rows.append({
                        "student_name": name,
                        "subject": subject,
                        "topic": topic,
                        "confidence": f"{confidence:.4f}",
                        "total_sessions": total_sessions,
                        "last_active": last_active,
                    })
            else:
                rows.append({
                    "student_name": name,
                    "subject": "",
                    "topic": "",
                    "confidence": "",
                    "total_sessions": total_sessions,
                    "last_active": last_active,
                })

        fieldnames = ["student_name", "subject", "topic", "confidence", "total_sessions", "last_active"]
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return filepath
