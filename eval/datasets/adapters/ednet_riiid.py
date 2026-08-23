"""EdNet / Riiid AIEd Challenge (Santa) — behavioral proxy adapter.

Source: the 500k uniform random sample at
eval/data/external/ednet_riiid/sample_500k.csv, drawn once by
eval/eval/sample_ednet.py from a 5.85 GB, 101,230,332-row train.csv. The full
file is never opened here.

    row_id, timestamp, user_id, content_id, content_type_id,
    task_container_id, user_answer, answered_correctly,
    prior_question_elapsed_time, prior_question_had_explanation

Lecture rows (content_type_id == 1, 9,768 of the 500k) carry
answered_correctly = -1 because they are not questions. They are dropped, which
leaves 490,232 answered questions from 138,033 users.

attempts = 1 is set as a PLATFORM fact, not an inference. Santa presents a
question, takes one answer, and moves on; there is no retry to log, so exactly
one attempt happened on every one of these rows. That is different in kind from
guessing at a missing value, and it is tagged "platform" in FIELDS to keep the
distinction visible.

Two fields look usable and are not:

  * prior_question_had_explanation is whether the student was shown an
    explanation for the PREVIOUS bundle after answering it. It is not a hint the
    student asked for mid-problem, and counting it as one would make Santa's
    STUCK rate a measure of a UI default.

  * prior_question_elapsed_time is a real response time, but for the previous
    question, so recovering this row's time means shifting the column one place
    within a user's ordered stream. That works on the full log and does NOT work
    on a uniform random sample: two rows adjacent in a 0.49% sample are about
    200 interactions apart in reality, so the shifted value belongs to some
    unidentified question, not the one on this row. Per-problem percentiles need
    the time attached to the right problem, so timing is reported unusable here
    rather than attached to the wrong content_id.

With hints absent and timing unusable, the rules that could still fire all
require timing, so this dataset produces no proxy states. The adapter reports
that, with the reason, instead of relaxing a rule to manufacture one.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Iterator

from .base import DatasetNotAvailableError
from .behavioral_base import BehavioralProxyAdapter, FieldStatus

#: content_type_id for a lecture rather than a question.
LECTURE_CONTENT_TYPE = "1"

TIMING_UNUSABLE = (
    "prior_question_elapsed_time times the PREVIOUS question, so it must be "
    "shifted one row within a user's ordered stream. The 500k sample is drawn "
    "uniformly at random over 101,230,332 rows, so consecutive sample rows for "
    "a user are ~200 interactions apart in the real log and the shift would "
    "attach a response time to a question that did not produce it. Recoverable "
    "only from a user-contiguous sample or the full file."
)


class EdnetRiiidAdapter(BehavioralProxyAdapter):
    source_dataset = "ednet_riiid"
    platform = "EdNet / Santa (Riiid)"
    year = "2020"
    modality = "behavioral"
    citation_key = "choi_2020"
    commercial_use_allowed = True

    SAMPLE_FILE = "sample_500k.csv"

    FIELDS = {
        "correct": FieldStatus("present", "answered_correctly (0/1)"),
        "attempt_count": FieldStatus(
            "platform", "constant 1", constant=1,
            reason="Santa takes one answer per question and offers no retry, "
                   "so every row represents exactly one attempt."),
        "hint_count": FieldStatus(
            "absent",
            reason="no hint mechanism in the log. "
                   "prior_question_had_explanation records an explanation shown "
                   "after the previous bundle, not a hint requested during this "
                   "question."),
        "response_time": FieldStatus("unusable", "prior_question_elapsed_time",
                                     reason=TIMING_UNUSABLE),
    }

    def _sample_path(self) -> str:
        # raw_dir points at the dataset directory; the sample sits beside raw/.
        base = self.raw_dir
        if os.path.basename(os.path.normpath(base)) == "raw":
            base = os.path.dirname(os.path.normpath(base))
        path = os.path.join(base, self.SAMPLE_FILE)
        if not os.path.exists(path):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"{self.SAMPLE_FILE} not found in {base}. Run "
                f"'python3.11 eval/eval/sample_ednet.py' first — the full "
                f"train.csv must not be read by this pipeline.")
        return path

    def sampling_metadata(self) -> dict[str, Any]:
        path = os.path.splitext(self._sample_path())[0] + "_meta.json"
        if not os.path.exists(path):
            return {}
        import json
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        path = self._sample_path()
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            emitted = 0
            for row in reader:
                if row.get("content_type_id") == LECTURE_CONTENT_TYPE:
                    continue          # lecture view, not an answered question
                if self.max_rows and emitted >= self.max_rows:
                    break
                answered = row.get("answered_correctly")
                correct = None
                if answered in ("0", "1"):
                    correct = answered == "1"
                emitted += 1
                yield {
                    "record_id": row.get("row_id") or str(emitted),
                    "student_id": row.get("user_id"),
                    "problem_id": row.get("content_id"),
                    "correct": correct,
                    "attempt_count": 1,      # platform fact, see FIELDS
                    "hint_count": None,
                    "response_time_ms": None,   # see TIMING_UNUSABLE
                    "skill": None,
                }
