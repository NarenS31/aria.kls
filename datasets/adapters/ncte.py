"""
NCTE Transcripts adapter — real 4th/5th-grade math classroom dialogue.

This is the best available test of whether ARIA's classifier survives contact
with genuine, non-LLM-generated human speech (Experiment B). There are no
ground-truth ARIA labels here, so:

  * ``modality`` = "dialogue", ``text`` = the student's utterance.
  * ``original_label`` = the NCTE discourse-move annotation(s) on that turn
    (e.g. "student_reasoning"), preserved verbatim.
  * ``aria_state_proxy`` = None — we never invent a cognitive label for
    classroom talk; the classifier's *predictions* on this text are the
    experiment, not a stored proxy.

We keep only STUDENT turns, and within those, turns that contain reasoning
language (either the NCTE reasoning annotation is set, or the utterance carries
explicit reasoning markers). Filtering to reasoning turns targets the utterances
where a cognitive state could plausibly be read at all.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from typing import Any, Iterator, Optional

from .base import DatasetAdapter, DatasetNotAvailableError, aria_record

SPEAKER_COLUMNS = ("speaker", "role", "turn_speaker", "speaker_role")
TEXT_COLUMNS = ("text", "utterance", "turn_text", "content", "transcript")
ID_COLUMNS = ("turn_id", "id", "order", "turn", "obsid", "transcript_id")

# NCTE turn-level discourse-move annotation columns (superset; whichever exist).
DISCOURSE_MOVE_COLUMNS = (
    "student_on_task", "student_reasoning", "uptake", "high_uptake",
    "focusing_question", "launch", "teacher_on_task", "student_talk",
    "questioning", "explanation", "math_terms",
)
STUDENT_REASONING_COL = "student_reasoning"

REASONING_MARKERS = re.compile(
    r"\b(because|since|so that|therefore|thus|reason|if\b|then\b|why|"
    r"means that|in order to|i think|i believe|that'?s why|depends|"
    r"could be|might be|however|for example|figured|makes sense|"
    r"the answer is|equals|equal to|add|subtract|multiply|divide)\b",
    re.IGNORECASE,
)

PROXY_METHOD = ("ncte_dialogue_v1: student turns with reasoning language; "
                "original_label = NCTE discourse move; no ARIA proxy assigned.")


def has_reasoning_language(text: str) -> bool:
    return bool(text) and bool(REASONING_MARKERS.search(text))


class NCTEAdapter(DatasetAdapter):
    modality = "dialogue"
    citation_key = "demszky_2023"
    commercial_use_allowed = False

    def _find_file(self) -> tuple[str, str]:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset or "ncte",
                f"raw directory {self.raw_dir!r} does not exist. The full "
                "transcripts require the NCTE data-request form.")
        for ext in ("*.csv", "*.jsonl", "*.json"):
            hits = sorted(glob.glob(os.path.join(self.raw_dir, "**", ext),
                                    recursive=True))
            if hits:
                return hits[0], ext.lstrip("*.")
        raise DatasetNotAvailableError(
            self.source_dataset or "ncte",
            f"no transcript file (csv/json/jsonl) found in {self.raw_dir}.")

    def load_raw(self) -> Iterator[dict[str, Any]]:
        path, kind = self._find_file()
        if kind == "csv":
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self._fieldnames = reader.fieldnames or []
                for row in reader:
                    yield row
        elif kind == "jsonl":
            with open(path, "r", encoding="utf-8") as fh:
                first = None
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if first is None:
                        first = list(rec.keys())
                    yield rec
                self._fieldnames = first or []
        else:  # json
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows = data if isinstance(data, list) else data.get("turns", [])
            self._fieldnames = list(rows[0].keys()) if rows else []
            for rec in rows:
                yield rec

    @staticmethod
    def _match(fieldnames: list[str], candidates) -> Optional[str]:
        lowered = {f.lower(): f for f in fieldnames}
        for c in candidates:
            if c in lowered:
                return lowered[c]
        for f in fieldnames:
            if any(c in f.lower() for c in candidates):
                return f
        return None

    @staticmethod
    def _is_student(value: str) -> bool:
        v = (value or "").strip().lower()
        return v in ("student", "s", "stu", "pupil", "learner", "1", "true")

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.load_raw())
        fields = getattr(self, "_fieldnames", []) or (list(rows[0]) if rows else [])
        speaker_col = self._match(fields, SPEAKER_COLUMNS)
        text_col = self._match(fields, TEXT_COLUMNS)
        id_col = self._match(fields, ID_COLUMNS)
        move_cols = [f for f in fields
                     if f.lower() in DISCOURSE_MOVE_COLUMNS]

        if text_col is None:
            raise DatasetNotAvailableError(
                self.source_dataset or "ncte",
                f"could not find a text/utterance column in {fields}.")

        for i, r in enumerate(rows):
            # student turns only (if a speaker column exists)
            if speaker_col is not None and not self._is_student(r.get(speaker_col)):
                continue
            text = (r.get(text_col) or "").strip()
            if not text:
                continue
            reasoning_flag = False
            active_moves = []
            for mc in move_cols:
                val = str(r.get(mc, "")).strip().lower()
                if val in ("1", "true", "yes"):
                    active_moves.append(mc)
                    if mc == STUDENT_REASONING_COL:
                        reasoning_flag = True
            if not (reasoning_flag or has_reasoning_language(text)):
                continue
            original_label = ",".join(active_moves) if active_moves else None
            rid = (r.get(id_col) if id_col else None) or f"turn{i}"
            yield aria_record(
                source_dataset=self.source_dataset or "ncte",
                source_record_id=rid,
                modality="dialogue",
                citation_key=self.citation_key,
                commercial_use_allowed=self.commercial_use_allowed,
                proxy_method=PROXY_METHOD,
                text=text,
                original_label=original_label,
                aria_state_proxy=None,     # no invented cognitive label
                proxy_confidence=None,
            )
