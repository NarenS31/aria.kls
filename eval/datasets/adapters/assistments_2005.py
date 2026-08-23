"""ASSISTments 2005-06 per-problem tutor log — behavioral proxy adapter.

Source: dataframe3_aggr_06.txt.output.txt, tab-separated, 162,898 rows, 3,213
students. Same generation of the platform as the 2004 file but a renamed and
slightly wider schema:

    student_id  question_id  response  attempt_count  hint_count
    start_time  finish_time  total_SECONDS  curriculum_id  is_original
    is_done  original_question_id  bottom_out_hint_count

`response` is the 0/1 outcome (88,463 zeros / 74,435 ones), `total_SECONDS` the
elapsed time, and `curriculum_id` stands in for a skill tag. All four proxy
inputs are real columns.

12,172 rows carry attempt_count = 0 — a problem the student opened and left
without submitting an answer. They are kept with attempt_count 0 rather than
being coerced to 1: no rule matches a zero-attempt row, which is the correct
outcome for a problem that was never actually attempted.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator

from .assistments import _to_bool, _to_int
from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus


class Assistments2005Adapter(BehavioralProxyAdapter):
    source_dataset = "assistments_2005"
    platform = "ASSISTments"
    year = "2005"
    modality = "behavioral"
    citation_key = "feng_2009"
    commercial_use_allowed = True

    COL_STUDENT = "student_id"
    COL_PROBLEM = "question_id"
    COL_CORRECT = "response"
    COL_ATTEMPT = "attempt_count"
    COL_HINT = "hint_count"
    COL_SECONDS = "total_SECONDS"
    COL_SKILL = "curriculum_id"

    FIELDS = {
        "correct": FieldStatus("present", "response (0/1)"),
        "attempt_count": FieldStatus("present", "attempt_count"),
        "hint_count": FieldStatus("present", "hint_count"),
        "response_time": FieldStatus(
            "derived", "total_SECONDS",
            reason="whole seconds in the source, multiplied to milliseconds"),
    }

    FILE_GLOB = "dataframe3_aggr_06.txt.output.txt"

    def _find_file(self) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"raw directory {self.raw_dir!r} does not exist.")
        matches = sorted(glob.glob(os.path.join(self.raw_dir, self.FILE_GLOB)))
        if not matches:
            matches = sorted(glob.glob(os.path.join(self.raw_dir, "*.txt")))
        if not matches:
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"no {self.FILE_GLOB} in {self.raw_dir}.")
        return matches[0]

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        path = self._find_file()
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for i, row in enumerate(reader):
                if self.max_rows and i >= self.max_rows:
                    break
                seconds = _to_int(row.get(self.COL_SECONDS))
                yield {
                    "record_id": f"{row.get(self.COL_STUDENT)}:"
                                 f"{row.get(self.COL_PROBLEM)}:{i}",
                    "student_id": row.get(self.COL_STUDENT),
                    "problem_id": row.get(self.COL_PROBLEM),
                    "correct": _to_bool(row.get(self.COL_CORRECT)),
                    "attempt_count": _to_int(row.get(self.COL_ATTEMPT)),
                    "hint_count": _to_int(row.get(self.COL_HINT)),
                    "response_time_ms": (seconds * 1000
                                         if seconds is not None else None),
                    "skill": row.get(self.COL_SKILL) or None,
                }
