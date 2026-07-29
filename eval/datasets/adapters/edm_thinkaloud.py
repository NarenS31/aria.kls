"""
EDM 2024 Think-Aloud adapter (Zhang, Borchers, Aleven, Baker).

This is the ONLY external source that directly matches ARIA's input modality:
real student think-aloud text with self-regulated-learning (SRL) labels. It is
access-controlled, so:

  * ``load_raw()`` raises ``DatasetNotAvailableError`` naming the ready-to-send
    request email (datasets/REQUEST_EMAIL.md) when the files are absent.
  * The adapter is nonetheless fully implemented against the published schema, so
    it works the moment access is granted — no code changes required.

The proposed SRL -> ARIA crosswalk is a documented constant below and is
described in the request email so the authors can sanity-check it. Ambiguous SRL
codes (e.g. bare "monitoring", which can accompany either smooth progress or
confusion) map to None rather than a guess.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Iterator, Optional

from .base import DatasetAdapter, DatasetNotAvailableError, aria_record

# ------------------------------------------------------------------
# Proposed SRL -> ARIA crosswalk (documented constant; see REQUEST_EMAIL.md).
# key: normalized SRL code -> (ARIA state or None, confidence, justification)
# ------------------------------------------------------------------
SRL_TO_ARIA: dict[str, tuple[Optional[str], Optional[str], str]] = {
    # Planning / forethought phase
    "planning":        ("PLANNING",   "high",   "SRL planning == ARIA PLANNING"),
    "orientation":     ("PLANNING",   "medium", "task orientation ~ ARIA PLANNING"),
    "goal_setting":    ("PLANNING",   "high",   "goal setting == ARIA PLANNING"),
    "prior_knowledge_activation": ("PLANNING", "medium",
                                   "activating prior knowledge ~ forethought/PLANNING"),
    # Productive performance
    "elaboration":     ("FLOW",       "medium", "productive elaboration ~ ARIA FLOW"),
    "organization":    ("FLOW",       "low",    "organizing content ~ on-task FLOW"),
    "strategy_use":    ("FLOW",       "low",    "smooth strategy execution ~ FLOW"),
    # Difficulty / breakdown
    "confusion":       ("CONFUSED",   "high",   "confusion == ARIA CONFUSED"),
    "struggle":        ("CONFUSED",   "medium", "struggle ~ ARIA CONFUSED"),
    "help_seeking":    ("STUCK",      "medium", "seeking help ~ ARIA STUCK"),
    "negative_affect": ("FRUSTRATED", "high",   "negative affect == ARIA FRUSTRATED"),
    "frustration":     ("FRUSTRATED", "high",   "frustration == ARIA FRUSTRATED"),
    # Realization
    "insight":         ("INSIGHT",    "high",   "realization == ARIA INSIGHT"),
    "evaluation_positive": ("INSIGHT", "low",   "positive re-evaluation ~ realization"),
    # Deliberately unmapped (ambiguous or no ARIA equivalent) -> None
    "monitoring":      (None, None, "bare monitoring is state-ambiguous"),
    "evaluation":      (None, None, "bare evaluation is state-ambiguous"),
    "reflection":      (None, None, "reflection spans multiple ARIA states"),
    "off_task":        (None, None, "ARIA has no off-task state"),
    "motivation":      (None, None, "motivation is not a cognitive state"),
}

TEXT_COLUMNS = ("text", "utterance", "segment", "think_aloud", "transcript", "content")
LABEL_COLUMNS = ("srl_label", "srl", "code", "label", "srl_code", "category")
ID_COLUMNS = ("segment_id", "id", "utterance_id", "participant_id", "row_id")

PROXY_METHOD = ("edm_srl_crosswalk_v1: map published SRL codes to ARIA states; "
                "ambiguous codes -> None; see SRL_TO_ARIA / REQUEST_EMAIL.md")

DEFAULT_REQUEST_EMAIL = "datasets/REQUEST_EMAIL.md"


def _normalize(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_").replace("-", "_")


def srl_crosswalk(raw_label: str) -> tuple[Optional[str], Optional[str]]:
    key = _normalize(raw_label)
    if key in SRL_TO_ARIA:
        state, conf, _ = SRL_TO_ARIA[key]
        return state, conf
    return None, None


class EDMThinkAloudAdapter(DatasetAdapter):
    modality = "think_aloud"
    citation_key = "zhang_2024"
    commercial_use_allowed = False

    def __init__(self, spec=None, raw_dir: Optional[str] = None):
        super().__init__(spec, raw_dir)
        cfg = getattr(spec, "download_config", {}) or {}
        self.request_email_path = cfg.get("request_email_path", DEFAULT_REQUEST_EMAIL)

    def _not_available(self, detail: str) -> DatasetNotAvailableError:
        return DatasetNotAvailableError(
            self.source_dataset or "edm_thinkaloud",
            detail, request_email_path=self.request_email_path)

    def _find_file(self) -> tuple[str, str]:
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            raise self._not_available(
                f"raw directory {self.raw_dir!r} does not exist.")
        for ext in ("*.csv", "*.jsonl", "*.json"):
            hits = sorted(glob.glob(os.path.join(self.raw_dir, "**", ext),
                                    recursive=True))
            if hits:
                return hits[0], ext.lstrip("*.")
        raise self._not_available(
            f"no think-aloud file (csv/json/jsonl) found in {self.raw_dir}.")

    def load_raw(self) -> Iterator[dict[str, Any]]:
        path, kind = self._find_file()
        if kind == "csv":
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self._fieldnames = reader.fieldnames or []
                for row in reader:
                    yield row
        elif kind == "jsonl":
            first = None
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    first = first or list(rec.keys())
                    yield rec
            self._fieldnames = first or []
        else:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows = data if isinstance(data, list) else data.get("segments", [])
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

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.load_raw())  # raises DatasetNotAvailableError if absent
        fields = getattr(self, "_fieldnames", []) or (list(rows[0]) if rows else [])
        text_col = self._match(fields, TEXT_COLUMNS)
        label_col = self._match(fields, LABEL_COLUMNS)
        id_col = self._match(fields, ID_COLUMNS)
        if text_col is None:
            raise self._not_available(
                f"could not find a text/utterance column in {fields}.")
        for i, r in enumerate(rows):
            text = (r.get(text_col) or "").strip() or None
            raw_label = r.get(label_col) if label_col else None
            aria_state, conf = srl_crosswalk(raw_label) if raw_label else (None, None)
            rid = (r.get(id_col) if id_col else None) or f"seg{i}"
            yield aria_record(
                source_dataset=self.source_dataset or "edm_thinkaloud",
                source_record_id=rid,
                modality="think_aloud",
                citation_key=self.citation_key,
                commercial_use_allowed=self.commercial_use_allowed,
                proxy_method=PROXY_METHOD,
                text=text,
                original_label=(str(raw_label) if raw_label else None),
                aria_state_proxy=aria_state,
                proxy_confidence=conf,
            )
