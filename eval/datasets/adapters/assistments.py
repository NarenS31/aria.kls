"""
ASSISTments 2009-2010 Skill Builder adapter — BEHAVIORAL PROXY mapping.

ASSISTments logs carry NO cognitive-state labels. This adapter derives an ARIA
state PROXY from observable behavior (response time, attempts, hints, outcome).
Every rule is documented with a confidence tier, and every emitted record marks
the label as a proxy. These are NOT ground truth and must never be reported as
"validated against real labels".

INSIGHT is intentionally never assigned: a moment of realization is not
derivable from behavioral logs alone (it needs the student's words). Rows that
match no rule get ``aria_state_proxy = None``.

Percentile rules ("faster than the 25th percentile for that problem") are
computed per problem_id from the response-time distribution, using only
problems with enough responses for the percentile to mean something.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator, Optional

import numpy as np

from .base import DatasetAdapter, DatasetNotAvailableError, aria_record

# Minimum responses on a problem before its response-time percentiles are trusted.
MIN_RESP_FOR_PCT = 5

# The proxy rules, in priority order (first match wins). Documented here so
# reports and the paper can render the exact derivation with its confidence.
PROXY_RULES = [
    {"state": "STUCK", "confidence": "high",
     "rule": "correct == 0 AND hint_count >= 3"},
    {"state": "RUSHING", "confidence": "high",
     "rule": "correct == 0 AND attempt_count == 1 AND response_time < p10(problem)"},
    {"state": "FLOW", "confidence": "medium",
     "rule": "correct == 1 AND attempt_count == 1 AND response_time < p25(problem)"},
    {"state": "CONFUSED", "confidence": "medium",
     "rule": "correct == 0 AND hint_count in (1,2)"},
    {"state": "FRUSTRATED", "confidence": "low",
     "rule": "correct == 0 AND attempt_count >= 4 "
             "(increasing inter-attempt gaps approximated; flat log lacks per-attempt timestamps)"},
    {"state": "PLANNING", "confidence": "low",
     "rule": "attempt_count == 1 AND response_time > p90(problem)"},
    {"state": None, "confidence": None,
     "rule": "no rule matched (includes INSIGHT, which is not derivable from behavior)"},
]

PROXY_METHOD = ("assistments_behavioral_rules_v1: per-problem response-time "
                "percentiles + attempt_count + hint_count + correctness; "
                "priority-ordered, first match wins; INSIGHT not derivable.")


def _to_int(v) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _to_bool(v) -> Optional[bool]:
    i = _to_int(v)
    if i is None:
        return None
    return bool(i)


class AssistmentsAdapter(DatasetAdapter):
    source_dataset = "assistments2009"
    modality = "behavioral"
    citation_key = "feng_2009"
    commercial_use_allowed = True

    # canonical ASSISTments skill_builder columns
    COL_ID = "order_id"
    COL_PROBLEM = "problem_id"
    COL_CORRECT = "correct"
    COL_ATTEMPT = "attempt_count"
    COL_HINT = "hint_count"
    COL_RT = "ms_first_response"
    COL_SKILL = "skill_name"

    def _find_csv(self) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"raw directory {self.raw_dir!r} does not exist.")
        candidates = []
        for pat in ("skill_builder_data*.csv", "*.csv"):
            candidates.extend(sorted(glob.glob(os.path.join(self.raw_dir, pat))))
        # de-dup preserving order
        seen, files = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                files.append(c)
        if not files:
            raise DatasetNotAvailableError(
                self.source_dataset,
                f"no CSV found in {self.raw_dir}. Expected skill_builder_data.csv.")
        return files[0]

    def load_raw(self) -> Iterator[dict[str, Any]]:
        path = self._find_csv()
        # ASSISTments CSVs are frequently latin-1 encoded.
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=enc, newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        yield row
                return
            except UnicodeDecodeError:
                continue
        raise DatasetNotAvailableError(
            self.source_dataset, f"could not decode {path} as utf-8 or latin-1.")

    def _percentiles(self, rows: list[dict]) -> dict[str, tuple]:
        """Per-problem (p10, p25, p90) of response time, for well-sampled problems."""
        by_problem: dict[str, list[int]] = {}
        for r in rows:
            pid = r.get(self.COL_PROBLEM)
            rt = _to_int(r.get(self.COL_RT))
            if pid is None or rt is None or rt <= 0:
                continue
            by_problem.setdefault(pid, []).append(rt)
        pct: dict[str, tuple] = {}
        for pid, vals in by_problem.items():
            if len(vals) < MIN_RESP_FOR_PCT:
                continue
            arr = np.array(vals, dtype=float)
            pct[pid] = (float(np.percentile(arr, 10)),
                        float(np.percentile(arr, 25)),
                        float(np.percentile(arr, 90)))
        return pct

    @staticmethod
    def classify(correct: Optional[bool], attempt: Optional[int],
                 hint: Optional[int], rt: Optional[int],
                 pct: Optional[tuple]) -> tuple[Optional[str], Optional[str]]:
        """Apply PROXY_RULES; return (state, confidence) or (None, None)."""
        p10 = p25 = p90 = None
        if pct is not None:
            p10, p25, p90 = pct
        # priority-ordered
        if correct is False and (hint is not None and hint >= 3):
            return "STUCK", "high"
        if (correct is False and attempt == 1 and rt is not None
                and p10 is not None and rt < p10):
            return "RUSHING", "high"
        if (correct is True and attempt == 1 and rt is not None
                and p25 is not None and rt < p25):
            return "FLOW", "medium"
        if correct is False and hint in (1, 2):
            return "CONFUSED", "medium"
        if correct is False and (attempt is not None and attempt >= 4):
            return "FRUSTRATED", "low"
        if (attempt == 1 and rt is not None and p90 is not None and rt > p90):
            return "PLANNING", "low"
        return None, None

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.load_raw())  # materialize for percentile pass
        pct = self._percentiles(rows)
        for r in rows:
            rid = r.get(self.COL_ID) or r.get(self.COL_PROBLEM) or ""
            correct = _to_bool(r.get(self.COL_CORRECT))
            attempt = _to_int(r.get(self.COL_ATTEMPT))
            hint = _to_int(r.get(self.COL_HINT))
            rt = _to_int(r.get(self.COL_RT))
            state, conf = self.classify(
                correct, attempt, hint, rt, pct.get(r.get(self.COL_PROBLEM)))
            yield aria_record(
                source_dataset=self.source_dataset,
                source_record_id=rid,
                modality=self.modality,
                citation_key=self.citation_key,
                commercial_use_allowed=self.commercial_use_allowed,
                proxy_method=PROXY_METHOD,
                text=None,
                response_time_ms=rt,
                attempt_count=attempt,
                hint_count=hint,
                correct=correct,
                original_label=None,           # no cognitive label in ASSISTments
                aria_state_proxy=state,
                proxy_confidence=conf,
            )
