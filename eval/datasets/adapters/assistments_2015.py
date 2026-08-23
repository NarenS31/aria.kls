"""ASSISTments 2015 skill builders — behavioral proxy adapter.

Source: 2015_100_skill_builders_main_problems.csv, 708,631 rows. Despite the
shared platform name this file has nothing in common with the 2004/2005 logs.
It has four columns:

    user_id, log_id, sequence_id, correct

and that is the entire schema. There is no attempt count, no hint count, and no
timing of any kind — not a null-heavy column, not a differently-named column,
nothing. This was verified by inspection before the adapter was written; see
eval/data/eval/external_schema_inspection.json.

Consequence, stated plainly rather than worked around: five of the six proxy
rules read a field this dataset does not have, and the sixth (PLANNING) needs
both attempts and timing. So NO rule can fire, every row maps to
aria_state_proxy = None, and the distribution and timing checks are not
computable here. The adapter still runs and still reports N rows, N students and
the correctness base rate, because "we looked and the data cannot support this
check" is a result worth recording — but no proxy is manufactured to fill the
gap.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator

from .assistments import _to_bool
from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus

NO_BEHAVIORAL_COLUMNS = (
    "The 2015 export carries only user_id, log_id, sequence_id and correct. "
    "Attempt counts, hint counts and response times are absent from the source."
)


class Assistments2015Adapter(BehavioralProxyAdapter):
    source_dataset = "assistments_2015"
    platform = "ASSISTments"
    year = "2015"
    modality = "behavioral"
    citation_key = "heffernan_2014"
    commercial_use_allowed = True

    COL_STUDENT = "user_id"
    COL_PROBLEM = "sequence_id"
    COL_CORRECT = "correct"
    COL_LOG = "log_id"

    FIELDS = {
        "correct": FieldStatus("present", "correct (0/1)"),
        "attempt_count": FieldStatus("absent", reason=NO_BEHAVIORAL_COLUMNS),
        "hint_count": FieldStatus("absent", reason=NO_BEHAVIORAL_COLUMNS),
        "response_time": FieldStatus("absent", reason=NO_BEHAVIORAL_COLUMNS),
    }

    FILE_GLOB = "*skill_builders*.csv"

    def _find_file(self) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"raw directory {self.raw_dir!r} does not exist.")
        matches = sorted(glob.glob(os.path.join(self.raw_dir, self.FILE_GLOB)))
        if not matches:
            matches = sorted(glob.glob(os.path.join(self.raw_dir, "*.csv")))
        if not matches:
            raise DatasetNotAvailableError(
                self.source_dataset, f"no CSV in {self.raw_dir}.")
        return matches[0]

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        path = self._find_file()
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if self.max_rows and i >= self.max_rows:
                    break
                yield {
                    "record_id": row.get(self.COL_LOG) or str(i),
                    "student_id": row.get(self.COL_STUDENT),
                    "problem_id": row.get(self.COL_PROBLEM),
                    "correct": _to_bool(row.get(self.COL_CORRECT)),
                    "attempt_count": None,
                    "hint_count": None,
                    "response_time_ms": None,
                    "skill": row.get(self.COL_PROBLEM) or None,
                }
