"""Open University Learning Analytics Dataset (OULAD) — behavioral proxy adapter.

Source: seven related CSVs from the 2013-14 OU presentations. The tables that
matter here:

    studentAssessment.csv  173,912 rows  id_assessment, id_student,
                                         date_submitted, is_banked, score
    assessments.csv           206 rows   assessment due date, type, weight
    studentInfo.csv        32,593 rows   demographics + final_result
    studentVle.csv     10,655,280 rows   daily click counts per resource

OULAD is a distance-learning VLE, not a tutor. Nothing in it records a student
working a problem: there is no per-problem interaction anywhere in these tables,
so three of the four proxy inputs have no counterpart and cannot be recovered by
joining. Specifically, and each checked against the real columns rather than
assumed:

  * attempts — `num_of_prev_attempts` in studentInfo counts how many times the
    student has taken the whole MODULE before. It is not a count of tries at a
    question, and using it as one would silently change what FRUSTRATED means
    for this dataset only.
  * hints — the platform has no hint mechanism to log.
  * response time — `date_submitted` is an integer day index relative to module
    start, paired with the assessment's due `date`. The difference is "days
    early or late", a deadline behaviour measured in days. It is not the time
    spent producing the answer, and feeding it to percentile rules built for
    per-problem response times would compare quantities that are not the same
    kind of thing.

What OULAD does give is a real, graded outcome per submission, so correctness is
mapped and reported. The distribution and timing checks are not computable here,
and the adapter says so rather than substituting a proxy that would be
comparable to the other datasets in name only.

The pass threshold is the Open University's standard 40-mark pass for a scored
assessment. It is a constant on the class so the choice is visible and
changeable, not buried in an expression.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Iterator, Optional

from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus

#: Open University pass mark for a scored assessment.
PASS_MARK = 40

NO_PROBLEM_LEVEL_INTERACTION = (
    "OULAD is a distance-learning VLE with no per-problem interaction log. "
    "The only graded events are assessment submissions."
)


def _to_int(value: str) -> Optional[int]:
    try:
        if value is None or value.strip() in ("", "?"):
            return None
        return int(float(value))
    except (ValueError, AttributeError):
        return None


class OuladAdapter(BehavioralProxyAdapter):
    source_dataset = "oulad"
    platform = "OULAD (Open University)"
    year = "2013"
    modality = "behavioral"
    citation_key = "kuzilek_2017"
    commercial_use_allowed = True

    ASSESSMENT_FILE = "studentAssessment.csv"
    ASSESSMENTS_FILE = "assessments.csv"
    STUDENT_FILE = "studentInfo.csv"

    FIELDS = {
        "correct": FieldStatus(
            "derived", "studentAssessment.score",
            reason=f"score >= {PASS_MARK} (Open University pass mark)"),
        "attempt_count": FieldStatus(
            "absent",
            reason=NO_PROBLEM_LEVEL_INTERACTION + " `num_of_prev_attempts` "
                   "counts module retakes, not attempts at a question."),
        "hint_count": FieldStatus(
            "absent", reason="the platform has no hint mechanism to log."),
        "response_time": FieldStatus(
            "absent",
            reason="`date_submitted` is a day index against an assessment "
                   "deadline, i.e. days early or late. It is not time spent "
                   "answering, so it cannot feed response-time percentiles."),
    }

    def _path(self, name: str) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"raw directory {self.raw_dir!r} does not exist.")
        path = os.path.join(self.raw_dir, name)
        if not os.path.exists(path):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"{name} not found in {self.raw_dir}. Unzip the OULAD archive "
                f"there first.")
        return path

    def assessment_types(self) -> dict[str, str]:
        """id_assessment -> assessment_type (TMA / CMA / Exam), for reporting."""
        types: dict[str, str] = {}
        with open(self._path(self.ASSESSMENTS_FILE), encoding="utf-8",
                  newline="") as fh:
            for row in csv.DictReader(fh):
                types[str(row["id_assessment"])] = row.get("assessment_type", "")
        return types

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        types = self.assessment_types()
        with open(self._path(self.ASSESSMENT_FILE), encoding="utf-8",
                  newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if self.max_rows and i >= self.max_rows:
                    break
                score = _to_int(row.get("score"))
                assessment_id = str(row.get("id_assessment"))
                yield {
                    "record_id": f"{row.get('id_student')}:{assessment_id}",
                    "student_id": row.get("id_student"),
                    "problem_id": assessment_id,
                    "correct": (score >= PASS_MARK) if score is not None else None,
                    "attempt_count": None,
                    "hint_count": None,
                    "response_time_ms": None,
                    "skill": types.get(assessment_id) or None,
                }
