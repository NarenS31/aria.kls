"""Aleven Fractions Lab (DataShop dataset 1826) — behavioral proxy adapter.

Source: All_Data_1826_2013_0408_075715_forReimport.txt, tab-separated, 46
columns, 37,889 transactions, 74 students, CMU lab study "Fraction Study 2012".

DataShop logs one row per TRANSACTION, not per problem, which is the one real
structural difference from the ASSISTments files. A student working a step
produces an ATTEMPT row per try and a separate HINT_REQUEST row per hint asked:

    Student Response Type   ATTEMPT 37,454 · HINT_REQUEST 435
    Outcome                 Correct 18,344 · InCorrect 5,471 · Hint 435 · (blank) 13,639
    Total Num Hints         only ever set on HINT_REQUEST rows

So a hint count does exist, but it lives in the row count of a different record
type rather than in a column on the attempt. Feeding the raw transactions to the
proxy rules would ask "how many hints are on this attempt row?" and always read
zero. The adapter therefore folds transactions into one record per step —
matching the per-problem grain the ASSISTments rows already have — and counts
the hints that belong to that step.

Aggregation per (student, session, problem set, problem, step), in file order:

    hint_count        number of HINT_REQUEST transactions on the step
    attempt_count     highest "Attempt At Step" reached
    correct           outcome of the last graded attempt is Correct
    response_time_ms  summed Duration (sec) across the step's transactions

Column positions are fixed rather than looked up by name: the header genuinely
contains "Step Name" twice plus one unnamed column, so a name-keyed reader would
silently collapse them. The header is verified on load, and a mismatch raises
instead of quietly reading the wrong column.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator, Optional

from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus

# Column positions in the DataShop export, with the header text each must carry.
COL_STUDENT = (2, "Anon Student Id")
COL_SESSION = (4, "Session Id")
COL_DURATION = (7, "Duration (sec)")
COL_RESPONSE_TYPE = (8, "Student Response Type")
COL_PROBLEM_SET = (13, "Level(ProblemSet)")
COL_PROBLEM = (14, "Problem Name")
COL_STEP = (19, "Step Name")
COL_ATTEMPT_AT_STEP = (20, "Attempt At Step")
COL_OUTCOME = (21, "Outcome")
EXPECTED_COLUMNS = (COL_STUDENT, COL_SESSION, COL_DURATION, COL_RESPONSE_TYPE,
                    COL_PROBLEM_SET, COL_PROBLEM, COL_STEP,
                    COL_ATTEMPT_AT_STEP, COL_OUTCOME)

ATTEMPT = "ATTEMPT"
HINT_REQUEST = "HINT_REQUEST"
CORRECT_OUTCOMES = {"correct"}
GRADED_OUTCOMES = {"correct", "incorrect"}


def _to_float(value: str) -> Optional[float]:
    try:
        if value is None or value.strip() in ("", "."):
            return None
        return float(value)
    except (ValueError, AttributeError):
        return None


def _to_int(value: str) -> Optional[int]:
    number = _to_float(value)
    return int(number) if number is not None else None


class AlevenFractionsAdapter(BehavioralProxyAdapter):
    source_dataset = "aleven_fractions"
    platform = "Fractions Lab (DataShop)"
    year = "2012"
    modality = "behavioral"
    citation_key = "rau_2013"
    commercial_use_allowed = True

    FIELDS = {
        "correct": FieldStatus(
            "derived", "Outcome",
            reason="outcome of the last graded attempt on the step "
                   "(Correct vs InCorrect)"),
        "attempt_count": FieldStatus(
            "derived", "Attempt At Step",
            reason="highest attempt index reached on the step"),
        "hint_count": FieldStatus(
            "derived", "Student Response Type == HINT_REQUEST",
            reason="hints are separate transaction rows in DataShop, so they "
                   "are counted per step rather than read from a column"),
        "response_time": FieldStatus(
            "derived", "Duration (sec)",
            reason="summed across the step's transactions, converted to "
                   "milliseconds"),
    }

    FILE_GLOB = "All_Data_*.txt"

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
                f"no DataShop export in {self.raw_dir}.")
        return matches[0]

    @staticmethod
    def _check_header(header: list[str]) -> None:
        for index, expected in EXPECTED_COLUMNS:
            actual = header[index] if index < len(header) else "<missing>"
            if actual.strip() != expected:
                raise DatasetNotAvailableError(
                    "aleven_fractions",
                    f"DataShop column layout changed: expected {expected!r} at "
                    f"position {index}, found {actual!r}. Positions are fixed "
                    f"because the header repeats 'Step Name'; re-inspect the "
                    f"file before mapping it.")

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        path = self._find_file()
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader, [])
            self._check_header(header)

            # Fold transactions into steps, keeping first-seen order.
            steps: dict[tuple, dict[str, Any]] = {}
            for row in reader:
                if len(row) <= COL_OUTCOME[0]:
                    continue
                student = row[COL_STUDENT[0]]
                step_name = row[COL_STEP[0]]
                if not student or not step_name:
                    continue
                key = (student, row[COL_SESSION[0]], row[COL_PROBLEM_SET[0]],
                       row[COL_PROBLEM[0]], step_name)
                step = steps.get(key)
                if step is None:
                    if self.max_rows and len(steps) >= self.max_rows:
                        continue
                    step = steps[key] = {
                        "student": student,
                        "problem_id": f"{row[COL_PROBLEM_SET[0]]}|"
                                      f"{row[COL_PROBLEM[0]]}|{step_name}",
                        "hints": 0,
                        "max_attempt": None,
                        "last_graded_correct": None,
                        "duration_s": 0.0,
                        "index": len(steps),
                    }

                duration = _to_float(row[COL_DURATION[0]])
                if duration is not None and duration > 0:
                    step["duration_s"] += duration

                response_type = row[COL_RESPONSE_TYPE[0]].strip().upper()
                outcome = row[COL_OUTCOME[0]].strip().lower()

                if response_type == HINT_REQUEST:
                    step["hints"] += 1
                elif response_type == ATTEMPT:
                    attempt = _to_int(row[COL_ATTEMPT_AT_STEP[0]])
                    if attempt is not None:
                        step["max_attempt"] = max(
                            step["max_attempt"] or 0, attempt)
                    if outcome in GRADED_OUTCOMES:
                        step["last_graded_correct"] = outcome in CORRECT_OUTCOMES

        for key, step in steps.items():
            duration_ms = int(round(step["duration_s"] * 1000))
            yield {
                "record_id": f"{step['student']}:{step['index']}",
                "student_id": step["student"],
                "problem_id": step["problem_id"],
                "correct": step["last_graded_correct"],
                "attempt_count": step["max_attempt"],
                "hint_count": step["hints"],
                "response_time_ms": duration_ms if duration_ms > 0 else None,
                "skill": key[2] or None,
            }
