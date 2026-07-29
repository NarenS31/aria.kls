"""
Abstract adapter contract: map any external dataset into ARIA's unified schema.

Every adapter turns a raw source into an iterator of ARIA records with EXACTLY
the fields below. The two label fields are kept rigorously separate:

  * ``original_label``   — whatever label the source itself carries (a discourse
                           move, an SRL code, None for pure behavioral logs).
  * ``aria_state_proxy`` — an ARIA cognitive state. For sources without real
                           cognitive labels this is a DERIVED PROXY, and
                           ``proxy_confidence`` / ``proxy_method`` record exactly
                           how it was derived and how much to trust it.

Downstream code must never treat a proxy label as ground truth. ``validate()``
surfaces coverage, null rates and class balance so a proxy corpus can't quietly
masquerade as validated data.
"""

from __future__ import annotations

import abc
from collections import Counter
from typing import Any, Iterator, Optional

# ARIA's seven cognitive states (single source of truth for proxy validation).
ARIA_STATES = (
    "PLANNING", "FLOW", "CONFUSED", "RUSHING", "FRUSTRATED", "STUCK", "INSIGHT",
)
MODALITIES = ("think_aloud", "behavioral", "dialogue")
CONFIDENCE_TIERS = ("high", "medium", "low")


class DatasetNotAvailableError(RuntimeError):
    """Raised when a dataset's raw files are not present locally.

    Carries the concrete next step (a download instruction or the path to the
    ready-to-send request email) so the failure is always actionable.
    """

    def __init__(self, dataset: str, detail: str = "", request_email_path: str = ""):
        self.dataset = dataset
        self.request_email_path = request_email_path
        msg = f"dataset {dataset!r} is not available locally."
        if detail:
            msg += f" {detail}"
        if request_email_path:
            msg += (f" To request access, send the email at "
                    f"'{request_email_path}', then place the released files in "
                    f"the dataset's raw/ directory.")
        super().__init__(msg)


def aria_record(
    *,
    source_dataset: str,
    source_record_id: str,
    modality: str,
    citation_key: str,
    commercial_use_allowed: bool,
    proxy_method: str,
    text: Optional[str] = None,
    response_time_ms: Optional[int] = None,
    attempt_count: Optional[int] = None,
    hint_count: Optional[int] = None,
    correct: Optional[bool] = None,
    original_label: Optional[str] = None,
    aria_state_proxy: Optional[str] = None,
    proxy_confidence: Optional[str] = None,
) -> dict[str, Any]:
    """Construct one schema-valid ARIA record, validating the enum fields."""
    if modality not in MODALITIES:
        raise ValueError(f"bad modality {modality!r}")
    if aria_state_proxy is not None and aria_state_proxy not in ARIA_STATES:
        raise ValueError(f"bad aria_state_proxy {aria_state_proxy!r}")
    if proxy_confidence is not None and proxy_confidence not in CONFIDENCE_TIERS:
        raise ValueError(f"bad proxy_confidence {proxy_confidence!r}")
    return {
        "source_dataset": source_dataset,
        "source_record_id": str(source_record_id),
        "modality": modality,
        "text": text,
        "behavioral_features": {
            "response_time_ms": response_time_ms,
            "attempt_count": attempt_count,
            "hint_count": hint_count,
            "correct": correct,
        },
        "original_label": original_label,
        "aria_state_proxy": aria_state_proxy,
        "proxy_confidence": proxy_confidence,
        "proxy_method": proxy_method,
        "commercial_use_allowed": bool(commercial_use_allowed),
        "citation_key": citation_key,
    }


class DatasetAdapter(abc.ABC):
    """Base class: load a raw source and map it into ARIA's schema."""

    #: filled in by concrete adapters, or resolved from a DatasetSpec
    source_dataset: str = ""
    modality: str = "behavioral"
    citation_key: str = ""
    commercial_use_allowed: bool = True

    def __init__(self, spec=None, raw_dir: Optional[str] = None):
        """`spec` is a datasets.registry.DatasetSpec; `raw_dir` overrides its path."""
        self.spec = spec
        if spec is not None:
            self.source_dataset = spec.name
            self.modality = spec.modality
            self.citation_key = spec.citation_key
            self.commercial_use_allowed = spec.commercial_use_allowed
            self.raw_dir = raw_dir or spec.raw_dir
        else:
            self.raw_dir = raw_dir or ""

    # -- required interface ------------------------------------------
    @abc.abstractmethod
    def load_raw(self) -> Iterator[Any]:
        """Yield raw rows/records from the source files.

        Must raise DatasetNotAvailableError if the raw files are absent.
        """

    @abc.abstractmethod
    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        """Yield ARIA-schema records (see ``aria_record``)."""

    # -- shared validation -------------------------------------------
    def validate(self, limit: int = 0) -> dict[str, Any]:
        """Report coverage, null rates and class balance over the mapped records.

        Does not raise on an empty/absent dataset — it returns a report with
        ``available: False`` so callers can decide what to do.
        """
        report: dict[str, Any] = {
            "source_dataset": self.source_dataset,
            "modality": self.modality,
            "commercial_use_allowed": self.commercial_use_allowed,
            "available": True,
            "n_records": 0,
        }
        try:
            records = self.to_aria_schema()
            rows = []
            for i, r in enumerate(records):
                rows.append(r)
                if limit and i + 1 >= limit:
                    break
        except DatasetNotAvailableError as e:
            report["available"] = False
            report["error"] = str(e)
            return report

        n = len(rows)
        report["n_records"] = n
        if n == 0:
            report["available"] = False
            report["error"] = "no records produced (empty source)"
            return report

        # null coverage per field
        def null_rate(pred) -> float:
            return round(sum(1 for r in rows if pred(r)) / n, 4)

        report["null_rates"] = {
            "text": null_rate(lambda r: r["text"] is None),
            "response_time_ms": null_rate(
                lambda r: r["behavioral_features"]["response_time_ms"] is None),
            "attempt_count": null_rate(
                lambda r: r["behavioral_features"]["attempt_count"] is None),
            "hint_count": null_rate(
                lambda r: r["behavioral_features"]["hint_count"] is None),
            "correct": null_rate(
                lambda r: r["behavioral_features"]["correct"] is None),
            "original_label": null_rate(lambda r: r["original_label"] is None),
            "aria_state_proxy": null_rate(lambda r: r["aria_state_proxy"] is None),
        }
        # proxy class balance + confidence mix
        proxy_counts = Counter(r["aria_state_proxy"] for r in rows
                               if r["aria_state_proxy"] is not None)
        conf_counts = Counter(r["proxy_confidence"] for r in rows
                              if r["proxy_confidence"] is not None)
        orig_counts = Counter(r["original_label"] for r in rows
                              if r["original_label"] is not None)
        report["aria_state_proxy_balance"] = dict(proxy_counts)
        report["proxy_confidence_balance"] = dict(conf_counts)
        report["original_label_balance"] = dict(orig_counts.most_common(20))
        report["proxy_coverage"] = round(
            sum(proxy_counts.values()) / n, 4)
        report["has_real_cognitive_labels"] = bool(
            self.spec.has_cognitive_labels) if self.spec else False
        return report
