"""Eedi NeurIPS 2020 Education Challenge — behavioral proxy adapter.

LICENCE: CC BY-NC-ND 4.0. Non-commercial use only. That flag is carried into
every record this adapter emits and into every report that reads them.

Files present locally (inventoried before mapping; the challenge was released in
several task-specific pieces and not every download has all of them):

    train_data/train_task_1_2.csv          15,867,850 rows  QuestionId, UserId,
                                           AnswerId, IsCorrect, CorrectAnswer,
                                           AnswerValue
    train_data/train_task_3_4.csv           1,382,727 rows  same six columns
    metadata/answer_metadata_task_1_2.csv  19,834,820 rows  AnswerId,
                                           DateAnswered, Confidence, GroupId,
                                           QuizId, SchemeOfWorkId
    metadata/question_metadata_task_1_2.csv   27,613 rows  QuestionId, SubjectId
    metadata/student_metadata_task_1_2.csv   118,971 rows  UserId, Gender,
                                           DateOfBirth, PremiumPupil
    metadata/subject_metadata.csv                388 rows  subject taxonomy

Task 1/2 training data is used as the answer log. Like Santa, Eedi records one
answer per question with no retry, so attempts = 1 is a platform fact rather
than a filled-in guess.

There are no hints in the platform.

Timing is the interesting negative. DateAnswered exists in the answer metadata,
so a response time could in principle come from differencing a student's
consecutive answers. It cannot, for a measured reason: every DateAnswered value
is rounded to the minute. Sampling 200,000 rows of the metadata found the
seconds field equal to "00" in 200,000 of 200,000 cases. Differences are
therefore quantized to whole minutes, which collapses the per-question
percentiles the proxy rules compare against — p10 and p25 both land on zero for
almost every question, so "faster than p10" would be unfirable and "slower than
p90" arbitrary. Timing is reported unusable, with that measurement recorded,
rather than being computed into numbers that would look fine in a table.

Confidence (a student self-report) is present but mostly empty, and is not one
of the four inputs the shared rules take. Using it would mean new state logic
for one dataset, which would break the comparison this validation exists to make.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Iterator

from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus

NON_COMMERCIAL_BANNER = (
    "Eedi NeurIPS 2020 data is CC BY-NC-ND 4.0 — non-commercial use only."
)

TIMING_UNUSABLE = (
    "DateAnswered is rounded to the minute: the seconds field was '00' in "
    "200,000 of 200,000 sampled metadata rows. Response times differenced from "
    "it are quantized to 60s, which degenerates the per-question p10/p25/p90 "
    "the rules compare against."
)


class EediNeurips2020Adapter(BehavioralProxyAdapter):
    source_dataset = "eedi_neurips2020"
    platform = "Eedi"
    year = "2020"
    modality = "behavioral"
    citation_key = "wang_2020"
    commercial_use_allowed = False

    TRAIN_FILE = os.path.join("train_data", "train_task_1_2.csv")
    QUESTION_META = os.path.join("metadata", "question_metadata_task_1_2.csv")

    FIELDS = {
        "correct": FieldStatus("present", "IsCorrect (0/1)"),
        "attempt_count": FieldStatus(
            "platform", "constant 1", constant=1,
            reason="Eedi records one answer per question with no retry, so "
                   "every row represents exactly one attempt."),
        "hint_count": FieldStatus(
            "absent", reason="the platform has no hint mechanism to log."),
        "response_time": FieldStatus(
            "unusable", "metadata/answer_metadata_task_1_2.csv DateAnswered",
            reason=TIMING_UNUSABLE),
    }

    def _path(self, relative: str) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"raw directory {self.raw_dir!r} does not exist.")
        path = os.path.join(self.raw_dir, relative)
        if not os.path.exists(path):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"{relative} not found in {self.raw_dir}. The challenge was "
                f"released in task-specific pieces; inventory the download "
                f"before mapping it.")
        return path

    def available_files(self) -> dict[str, Any]:
        """What this particular download actually contains, with row counts."""
        found = {}
        for root, _dirs, files in os.walk(self.raw_dir):
            for name in sorted(files):
                if not name.endswith(".csv"):
                    continue
                full = os.path.join(root, name)
                found[os.path.relpath(full, self.raw_dir)] = {
                    "size_bytes": os.path.getsize(full),
                }
        return found

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        path = self._path(self.TRAIN_FILE)
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if self.max_rows and i >= self.max_rows:
                    break
                is_correct = row.get("IsCorrect")
                correct = None
                if is_correct in ("0", "1"):
                    correct = is_correct == "1"
                yield {
                    "record_id": row.get("AnswerId") or str(i),
                    "student_id": row.get("UserId"),
                    "problem_id": row.get("QuestionId"),
                    "correct": correct,
                    "attempt_count": 1,       # platform fact, see FIELDS
                    "hint_count": None,
                    "response_time_ms": None,  # see TIMING_UNUSABLE
                    "skill": None,
                }
