"""
EduAgent adapter (EduAgent310 real students / EduAgent705 synthetic agents).

EduAgent310 carries *inferred cognitive states* for 310 real students — the
highest-value external signal available without the EDM think-aloud data. This
adapter maps EduAgent's cognitive taxonomy onto ARIA's seven states through an
explicit, justified crosswalk (see datasets/adapters/eduagent_crosswalk.md).

Design commitments:
  * The raw EduAgent label is preserved verbatim in ``original_label``.
  * ``aria_state_proxy`` holds the crosswalked ARIA state, or None when EduAgent's
    label has no clean ARIA equivalent (e.g. "distracted", "bored" — ARIA has no
    such state). A missing mapping is None, never a guess.
  * The EduAgent schema is auto-detected at load time (the public release has
    shifted between paper and code), so the adapter degrades gracefully: if no
    per-record cognitive-state column is present it still emits records (with
    aria_state_proxy None) and ``validate()`` will show zero proxy coverage.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Any, Iterator, Optional

from .base import DatasetAdapter, DatasetNotAvailableError, aria_record

# ------------------------------------------------------------------
# Crosswalk: normalized EduAgent cognitive label -> (ARIA state, confidence, why)
# Only clean, defensible correspondences map to a state; everything else -> None.
# Keep this in sync with eduagent_crosswalk.md.
# ------------------------------------------------------------------
EDUAGENT_TO_ARIA: dict[str, tuple[Optional[str], Optional[str], str]] = {
    "planning":        ("PLANNING",   "high",   "explicit strategy/goal-setting == ARIA PLANNING"),
    "plan":            ("PLANNING",   "high",   "strategy/goal-setting == ARIA PLANNING"),
    "focused":         ("FLOW",       "medium", "sustained on-task focus ~ ARIA FLOW"),
    "focus":           ("FLOW",       "medium", "sustained on-task focus ~ ARIA FLOW"),
    "engaged":         ("FLOW",       "medium", "high engagement ~ ARIA FLOW"),
    "concentration":   ("FLOW",       "medium", "high concentration ~ ARIA FLOW"),
    "flow":            ("FLOW",       "high",   "direct match"),
    "confused":        ("CONFUSED",   "high",   "confusion == ARIA CONFUSED"),
    "confusion":       ("CONFUSED",   "high",   "confusion == ARIA CONFUSED"),
    "uncertain":       ("CONFUSED",   "medium", "expressed uncertainty ~ ARIA CONFUSED"),
    "rushing":         ("RUSHING",    "high",   "direct match"),
    "careless":        ("RUSHING",    "medium", "careless/fast responding ~ ARIA RUSHING"),
    "impulsive":       ("RUSHING",    "medium", "impulsive answering ~ ARIA RUSHING"),
    "frustrated":      ("FRUSTRATED", "high",   "frustration == ARIA FRUSTRATED"),
    "frustration":     ("FRUSTRATED", "high",   "frustration == ARIA FRUSTRATED"),
    "stuck":           ("STUCK",      "high",   "direct match"),
    "gave_up":         ("STUCK",      "medium", "giving up ~ ARIA STUCK"),
    "insight":         ("INSIGHT",    "high",   "direct match"),
    "aha":             ("INSIGHT",    "high",   "aha moment == ARIA INSIGHT"),
    "eureka":          ("INSIGHT",    "high",   "realization == ARIA INSIGHT"),
    # Deliberately unmapped (ARIA has no equivalent state) -> None:
    "distracted":      (None, None, "ARIA has no 'distracted' state"),
    "mind_wandering":  (None, None, "ARIA has no 'mind-wandering' state"),
    "bored":           (None, None, "ARIA has no 'bored' state"),
    "boredom":         (None, None, "ARIA has no 'bored' state"),
    "neutral":         (None, None, "no cognitive commitment to map"),
}

STATE_COLUMN_CANDIDATES = (
    "cognitive_state", "cog_state", "cognitive_label", "mental_state",
    "state", "affect", "emotion", "label",
)
ID_COLUMN_CANDIDATES = ("student_id", "agent_id", "user_id", "id", "student")
TEXT_COLUMN_CANDIDATES = ("text", "utterance", "think_aloud", "response", "transcript")
# EduAgent310 answer-item schema: real gaze-derived cognitive measures.
CONFUSION_DUR_COL = "confusion_dur"
INATTENTION_DUR_COL = "inattention_dur"
ACCURACY_CANDIDATES = ("accuracy", "correct", "is_correct")
QUESTION_CANDIDATES = ("question_id", "item_id", "qid")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

PROXY_METHOD = ("eduagent_taxonomy_crosswalk_v1: map EduAgent inferred "
                "cognitive labels to ARIA states; unmapped labels -> None; "
                "see eduagent_crosswalk.md")


def _normalize(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_").replace("-", "_")


def crosswalk(raw_label: str) -> tuple[Optional[str], Optional[str]]:
    """Return (aria_state, confidence) for an EduAgent label; (None, None) if unmapped."""
    key = _normalize(raw_label)
    if key in EDUAGENT_TO_ARIA:
        state, conf, _why = EDUAGENT_TO_ARIA[key]
        return state, conf
    return None, None


class EduAgentAdapter(DatasetAdapter):
    modality = "behavioral"
    citation_key = "xu_2024_eduagent"

    def __init__(self, spec=None, raw_dir: Optional[str] = None):
        super().__init__(spec, raw_dir)
        cfg = getattr(spec, "download_config", {}) or {}
        self.variant = cfg.get("variant", "real")
        self.priority_files = cfg.get("priority_files", [])

    def _find_csv(self) -> str:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise DatasetNotAvailableError(
                self.source_dataset or "eduagent",
                f"raw directory {self.raw_dir!r} does not exist.")
        # prefer the variant-specific demographics file
        for pf in self.priority_files:
            hits = glob.glob(os.path.join(self.raw_dir, "**", pf), recursive=True)
            if hits:
                return sorted(hits)[0]
        anycsv = sorted(glob.glob(os.path.join(self.raw_dir, "**", "*.csv"),
                                  recursive=True))
        if not anycsv:
            raise DatasetNotAvailableError(
                self.source_dataset or "eduagent",
                f"no CSV found in {self.raw_dir}.")
        return anycsv[0]

    def load_raw(self) -> Iterator[dict[str, Any]]:
        path = self._find_csv()
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            self._fieldnames = reader.fieldnames or []
            for row in reader:
                yield row

    @staticmethod
    def _match_column(fieldnames: list[str], candidates) -> Optional[str]:
        lowered = {f.lower(): f for f in fieldnames}
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
        # substring fallback
        for f in fieldnames:
            if any(cand in f.lower() for cand in candidates):
                return f
        return None

    @staticmethod
    def _derive_from_measures(row, acc_col):
        """Derive a REAL cognitive label from EduAgent's gaze measures.

        confusion_dur / inattention_dur are real, gaze-derived durations. We
        threshold them into a discrete label: measured confusion -> "confusion"
        (ARIA CONFUSED); measured inattention with no confusion -> "inattention"
        (no ARIA equivalent -> None); otherwise a correct answer with neither ->
        "focused" (ARIA FLOW). Returns (raw_label, correct_bool).
        """
        conf = _to_float(row.get(CONFUSION_DUR_COL))
        inatt = _to_float(row.get(INATTENTION_DUR_COL))
        acc = row.get(acc_col) if acc_col else None
        try:
            correct = bool(int(float(acc))) if acc not in (None, "") else None
        except (TypeError, ValueError):
            correct = None
        # Label from GAZE MEASURES ALONE (never from accuracy) so a downstream
        # accuracy-based proxy prediction stays independent of the ground truth.
        if conf is not None and conf > 0 and (inatt is None or conf >= inatt):
            return "confusion", correct
        if inatt is not None and inatt > 0:
            return "inattention", correct   # -> None in ARIA (no such state)
        if conf is not None and inatt is not None and conf == 0 and inatt == 0:
            return "focused", correct        # gaze on-task -> FLOW
        return None, correct

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.load_raw())
        fields = getattr(self, "_fieldnames", []) or (list(rows[0]) if rows else [])
        state_col = self._match_column(fields, STATE_COLUMN_CANDIDATES)
        id_col = self._match_column(fields, ID_COLUMN_CANDIDATES)
        text_col = self._match_column(fields, TEXT_COLUMN_CANDIDATES)
        qid_col = self._match_column(fields, QUESTION_CANDIDATES)
        acc_col = self._match_column(fields, ACCURACY_CANDIDATES)
        has_measures = (CONFUSION_DUR_COL in [f.lower() for f in fields]
                        or INATTENTION_DUR_COL in [f.lower() for f in fields])
        modality = "think_aloud" if text_col else "behavioral"

        for i, r in enumerate(rows):
            correct = None
            if state_col and (r.get(state_col) not in (None, "")):
                raw_label = r.get(state_col)
                method = PROXY_METHOD
            elif has_measures:
                raw_label, correct = self._derive_from_measures(r, acc_col)
                method = ("eduagent_gaze_measure_v1: discrete label thresholded "
                          "from real gaze-derived confusion_dur/inattention_dur; "
                          "crosswalked to ARIA (confusion->CONFUSED, inattention->None).")
            else:
                raw_label, method = None, PROXY_METHOD
            aria_state, conf = crosswalk(raw_label) if raw_label else (None, None)
            base_id = (r.get(id_col) if id_col else None) or f"row{i}"
            rid = f"{base_id}:{r.get(qid_col)}" if qid_col and r.get(qid_col) else str(base_id)
            text = (r.get(text_col) or None) if text_col else None
            yield aria_record(
                source_dataset=self.source_dataset or "eduagent",
                source_record_id=rid,
                modality=modality,
                citation_key=self.citation_key,
                commercial_use_allowed=self.commercial_use_allowed,
                proxy_method=method,
                text=text,
                correct=correct,
                original_label=(str(raw_label) if raw_label else None),
                aria_state_proxy=aria_state,
                proxy_confidence=conf,
            )
