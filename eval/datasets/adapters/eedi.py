"""
Eedi / NeurIPS 2020 Education Challenge adapter — behavioral proxies.

HARD LICENCE CONSTRAINT
-----------------------
Eedi is released under CC BY-NC-ND 4.0. It is NON-COMMERCIAL. Every record this
adapter emits carries ``commercial_use_allowed = False`` unconditionally, and
``NON_COMMERCIAL_BANNER`` is exported so any report that touches Eedi data can
print the warning. This is not configurable.

Behaviorally, Eedi is one row per answer: it reliably has correctness and (in
the answer metadata) a self-reported confidence, but — unlike ASSISTments — it
has NO hint counts and NO multi-attempt counts. So the ASSISTments hint/attempt
rules cannot fire here; the proxy set is correspondingly reduced and mostly
low-confidence, and FRUSTRATED / STUCK / PLANNING / INSIGHT are not derivable
from a single Eedi answer (mapped to None).
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator, Optional

import numpy as np

from .base import DatasetAdapter, DatasetNotAvailableError, aria_record

NON_COMMERCIAL_BANNER = (
    "================================================================\n"
    " NON-COMMERCIAL DATA (Eedi / NeurIPS 2020, CC BY-NC-ND 4.0)\n"
    " Results below include Eedi-derived records. Research use only;\n"
    " no commercial use, no derivatives redistribution.\n"
    "================================================================"
)

MIN_RESP_FOR_PCT = 5

EEDI_PROXY_RULES = [
    {"state": "CONFUSED", "confidence": "medium",
     "rule": "correct == 0 AND confidence <= 25 (low confidence + wrong)"},
    {"state": "RUSHING", "confidence": "medium",
     "rule": "correct == 0 AND response_time < p10(question) (fast + wrong)"},
    {"state": "RUSHING", "confidence": "low",
     "rule": "correct == 0 AND confidence >= 75 (overconfident + wrong)"},
    {"state": "FLOW", "confidence": "low",
     "rule": "correct == 1 AND (response_time < p25(question) OR confidence >= 75)"},
    {"state": None, "confidence": None,
     "rule": "no rule matched. FRUSTRATED/STUCK/PLANNING/INSIGHT not derivable "
             "from a single Eedi answer (no hints, no attempts, no text)."},
]

PROXY_METHOD = ("eedi_behavioral_rules_v1: correctness + self-reported "
                "confidence + optional per-question response-time percentiles; "
                "reduced ruleset (no hints/attempts in Eedi).")


def _to_int(v) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _to_bool(v) -> Optional[bool]:
    i = _to_int(v)
    return None if i is None else bool(i)


class EediAdapter(DatasetAdapter):
    modality = "behavioral"
    citation_key = "wang_2020"
    commercial_use_allowed = False  # enforced regardless of spec

    CORRECT_COLS = ("iscorrect", "is_correct", "correct")
    CONF_COLS = ("confidence",)
    RT_COLS = ("response_time", "ms_first_response", "elapsedtime", "time_taken")
    QID_COLS = ("questionid", "question_id", "problem_id")
    ID_COLS = ("answerid", "answer_id", "id")

    def _find_csv(self) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset or "eedi",
                f"raw directory {self.raw_dir!r} does not exist.")
        for pat in ("train_task_1_2*.csv", "*.csv"):
            hits = sorted(glob.glob(os.path.join(self.raw_dir, "**", pat),
                                    recursive=True))
            if hits:
                return hits[0]
        raise DatasetNotAvailableError(
            self.source_dataset or "eedi",
            f"no CSV found in {self.raw_dir}. Expected train_task_1_2.csv.")

    def load_raw(self) -> Iterator[dict[str, Any]]:
        path = self._find_csv()
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            self._fieldnames = reader.fieldnames or []
            for row in reader:
                yield row

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
    def classify(correct, confidence, rt, pct) -> tuple[Optional[str], Optional[str]]:
        p10 = p25 = None
        if pct is not None:
            p10, p25, _p90 = pct
        if correct is False and confidence is not None and confidence <= 25:
            return "CONFUSED", "medium"
        if (correct is False and rt is not None and p10 is not None and rt < p10):
            return "RUSHING", "medium"
        if correct is False and confidence is not None and confidence >= 75:
            return "RUSHING", "low"
        if correct is True and (
                (rt is not None and p25 is not None and rt < p25)
                or (confidence is not None and confidence >= 75)):
            return "FLOW", "low"
        return None, None

    def _percentiles(self, rows, qcol, rtcol) -> dict[str, tuple]:
        if rtcol is None:
            return {}
        by_q: dict[str, list[int]] = {}
        for r in rows:
            q = r.get(qcol) if qcol else None
            rt = _to_int(r.get(rtcol))
            if q is None or rt is None or rt <= 0:
                continue
            by_q.setdefault(q, []).append(rt)
        out = {}
        for q, vals in by_q.items():
            if len(vals) < MIN_RESP_FOR_PCT:
                continue
            a = np.array(vals, dtype=float)
            out[q] = (float(np.percentile(a, 10)),
                      float(np.percentile(a, 25)),
                      float(np.percentile(a, 90)))
        return out

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.load_raw())
        fields = getattr(self, "_fieldnames", []) or (list(rows[0]) if rows else [])
        correct_col = self._match(fields, self.CORRECT_COLS)
        conf_col = self._match(fields, self.CONF_COLS)
        rt_col = self._match(fields, self.RT_COLS)
        qid_col = self._match(fields, self.QID_COLS)
        id_col = self._match(fields, self.ID_COLS)
        pct = self._percentiles(rows, qid_col, rt_col)

        for i, r in enumerate(rows):
            correct = _to_bool(r.get(correct_col)) if correct_col else None
            confidence = _to_int(r.get(conf_col)) if conf_col else None
            rt = _to_int(r.get(rt_col)) if rt_col else None
            state, conf = self.classify(
                correct, confidence, rt, pct.get(r.get(qid_col)) if qid_col else None)
            rid = (r.get(id_col) if id_col else None) or f"row{i}"
            yield aria_record(
                source_dataset=self.source_dataset or "eedi",
                source_record_id=rid,
                modality="behavioral",
                citation_key=self.citation_key,
                commercial_use_allowed=False,   # HARD constraint
                proxy_method=PROXY_METHOD,
                text=None,
                response_time_ms=rt,
                attempt_count=None,             # Eedi has no attempt counts
                hint_count=None,                # Eedi has no hint counts
                correct=correct,
                original_label=None,
                aria_state_proxy=state,
                proxy_confidence=conf,
            )
