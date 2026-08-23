"""ASSISTments 2004-05 per-problem tutor log — behavioral proxy adapter.

Source: dataframe3_aggr.txt.output.txt, tab-separated, 226,099 rows, 912
students. One row per (student, item) with the attempt set already aggregated:

    studentID  itemid  correctonfirstattempt  attempts  hints  seconds  ...

This is the same shape the original ASSISTments adapter was written for — a
problem-level row carrying outcome, attempt count, hint count and elapsed time —
so all four proxy inputs are real columns and every rule can fire.

`correctonfirstattempt` is the outcome field. That is stricter than "eventually
correct", which is the right reading here: the proxy rules are about what the
student did on this problem, and a first-attempt failure followed by hints is
exactly the STUCK/CONFUSED signal they are looking for.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator

from .assistments import _to_bool, _to_int
from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus


class Assistments2004Adapter(BehavioralProxyAdapter):
    source_dataset = "assistments_2004"
    platform = "ASSISTments"
    year = "2004"
    modality = "behavioral"
    citation_key = "feng_2009"
    commercial_use_allowed = True

    COL_STUDENT = "studentID"
    COL_PROBLEM = "itemid"
    COL_CORRECT = "correctonfirstattempt"
    COL_ATTEMPT = "attempts"
    COL_HINT = "hints"
    COL_SECONDS = "seconds"

    FIELDS = {
        "correct": FieldStatus("present", "correctonfirstattempt (0/1)"),
        "attempt_count": FieldStatus("present", "attempts"),
        "hint_count": FieldStatus("present", "hints"),
        "response_time": FieldStatus(
            "derived", "seconds",
            reason="whole seconds in the source, multiplied to milliseconds"),
    }

    FILE_GLOB = "dataframe3_aggr.txt.output.txt"

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
                    "skill": None,
                }
