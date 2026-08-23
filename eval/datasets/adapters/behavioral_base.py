"""Shared machinery for behavioral-proxy adapters over external datasets.

Every dataset in the multi-dataset external validation runs the SAME proxy
rules. That is the whole point of the comparison: if each adapter reimplemented
"rushing" or "stuck" against its own schema, a difference between datasets could
just as easily be a difference between adapters. So the classification here is
literally ``AssistmentsAdapter.classify`` — imported, not copied — and the only
thing an adapter is allowed to vary is which of its own columns feed it.

A subclass provides ``normalized_rows()``, yielding this shape:

    {"record_id", "student_id", "problem_id", "correct", "attempt_count",
     "hint_count", "response_time_ms", "skill"}

with ``None`` for anything the source genuinely does not carry. Nothing is
invented to fill a gap. A dataset with no hint column reports None for
hint_count, and every rule that reads hints then simply does not fire — which
shows up honestly as an uncomputable check rather than as an invented state.

FIELDS declares, per adapter, what the source has and why. The statuses are:

    present   — a real column in the source, read directly
    derived   — computed from other columns (the derivation is stated)
    platform  — a constant that is a property of how the platform logs, not an
                inference (e.g. a system with no retries logs one attempt)
    absent    — the field does not exist in the source at all
    unusable  — the data exists but cannot support the check, with the measured
                reason (e.g. timestamps quantized to the minute)

Only ``present``, ``derived`` and ``platform`` feed the rules. ``absent`` and
``unusable`` are reported so a reader can see exactly which of a dataset's
checks were computable and which were not.
"""

from __future__ import annotations

import abc
from typing import Any, Iterator, Optional

import numpy as np

from .assistments import MIN_RESP_FOR_PCT, PROXY_RULES, AssistmentsAdapter
from .base import DatasetAdapter, aria_record

# Statuses a field can have in an adapter's FIELDS declaration.
USABLE_STATUSES = ("present", "derived", "platform")
ALL_STATUSES = USABLE_STATUSES + ("absent", "unusable")

# Re-exported so the runner and the report can render the exact rule set that
# produced every state, without importing from a specific dataset's module.
SHARED_PROXY_RULES = PROXY_RULES


class FieldStatus:
    """What one proxy input is, in one dataset, and why.

    ``constant`` is set when the field is a platform fact with a fixed value
    (a system with no retries always logs one attempt). It matters because a
    field can be perfectly available and still make a rule unfirable: a rule
    testing ``attempt_count >= 4`` can never match a log where every row has
    exactly one attempt. That is a different situation from a missing field and
    is reported differently.
    """

    def __init__(self, status: str, source: str = "", reason: str = "",
                 constant: Optional[float] = None):
        if status not in ALL_STATUSES:
            raise ValueError(f"bad field status {status!r}")
        self.status = status
        self.source = source
        self.reason = reason
        self.constant = constant

    @property
    def usable(self) -> bool:
        return self.status in USABLE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "source": self.source,
                "reason": self.reason, "usable": self.usable,
                "constant": self.constant}


# What each proxy rule reads, and the numeric condition it puts on that field.
# Used to tell "the field is missing" apart from "the field is present but
# pinned to a value the rule can never match".
RULE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "STUCK": {"fields": ("correct", "hint_count"),
              "conditions": {"hint_count": (lambda h: h >= 3, ">= 3")}},
    "RUSHING": {"fields": ("correct", "attempt_count", "response_time"),
                "conditions": {"attempt_count": (lambda a: a == 1, "== 1")}},
    "FLOW": {"fields": ("correct", "attempt_count", "response_time"),
             "conditions": {"attempt_count": (lambda a: a == 1, "== 1")}},
    "CONFUSED": {"fields": ("correct", "hint_count"),
                 "conditions": {"hint_count": (lambda h: h in (1, 2), "in (1,2)")}},
    "FRUSTRATED": {"fields": ("correct", "attempt_count"),
                   "conditions": {"attempt_count": (lambda a: a >= 4, ">= 4")}},
    "PLANNING": {"fields": ("attempt_count", "response_time"),
                 "conditions": {"attempt_count": (lambda a: a == 1, "== 1")}},
}


class BehavioralProxyAdapter(DatasetAdapter):
    """Maps one external dataset onto the shared behavioral proxy rules."""

    #: {"correct"|"attempt_count"|"hint_count"|"response_time": FieldStatus}
    FIELDS: dict[str, FieldStatus] = {}
    #: Human-readable platform label for the summary table.
    platform: str = ""
    year: str = ""
    #: Cap applied to very large sources, so every dataset reports what it used.
    max_rows: int = 500_000

    def __init__(self, raw_dir: str = "", max_rows: Optional[int] = None):
        super().__init__(spec=None, raw_dir=raw_dir)
        if max_rows is not None:
            self.max_rows = max_rows
        self._last_stats: dict[str, Any] = {}

    # -- subclass interface -------------------------------------------
    @abc.abstractmethod
    def normalized_rows(self) -> Iterator[dict[str, Any]]:
        """Yield normalized rows (see module docstring)."""

    def load_raw(self) -> Iterator[Any]:
        """Satisfies DatasetAdapter; the normalized form is the useful one."""
        return self.normalized_rows()

    # -- shared behaviour ---------------------------------------------
    @property
    def proxy_method(self) -> str:
        usable = ", ".join(
            f"{name}={status.status}" for name, status in self.FIELDS.items())
        return (f"shared_behavioral_rules_v1 (identical to "
                f"assistments_behavioral_rules_v1): per-problem response-time "
                f"percentiles + attempt_count + hint_count + correctness; "
                f"priority-ordered, first match wins; INSIGHT not derivable. "
                f"Field availability for this dataset: {usable}.")

    def field_report(self) -> dict[str, Any]:
        return {name: status.to_dict() for name, status in self.FIELDS.items()}

    def _percentiles(self, rows: list[dict]) -> dict[str, tuple]:
        """Per-problem (p10, p25, p90) of response time.

        Same computation and same minimum sample size as the ASSISTments
        adapter, so a percentile means the same thing in every dataset.
        """
        by_problem: dict[str, list[int]] = {}
        for row in rows:
            pid = row.get("problem_id")
            rt = row.get("response_time_ms")
            if pid is None or rt is None or rt <= 0:
                continue
            by_problem.setdefault(str(pid), []).append(rt)
        percentiles: dict[str, tuple] = {}
        for pid, values in by_problem.items():
            if len(values) < MIN_RESP_FOR_PCT:
                continue
            arr = np.array(values, dtype=float)
            percentiles[pid] = (float(np.percentile(arr, 10)),
                                float(np.percentile(arr, 25)),
                                float(np.percentile(arr, 90)))
        return percentiles

    def to_aria_schema(self) -> Iterator[dict[str, Any]]:
        rows = list(self.normalized_rows())
        percentiles = self._percentiles(rows)
        self._last_stats = {
            "n_rows_used": len(rows),
            "n_students": len({r["student_id"] for r in rows
                               if r.get("student_id") is not None}),
            "n_problems": len({r["problem_id"] for r in rows
                               if r.get("problem_id") is not None}),
            "n_problems_with_percentiles": len(percentiles),
        }
        for row in rows:
            state, confidence = AssistmentsAdapter.classify(
                row.get("correct"),
                row.get("attempt_count"),
                row.get("hint_count"),
                row.get("response_time_ms"),
                percentiles.get(str(row.get("problem_id"))),
            )
            yield aria_record(
                source_dataset=self.source_dataset,
                source_record_id=row.get("record_id", ""),
                modality=self.modality,
                citation_key=self.citation_key,
                commercial_use_allowed=self.commercial_use_allowed,
                proxy_method=self.proxy_method,
                text=None,
                response_time_ms=row.get("response_time_ms"),
                attempt_count=row.get("attempt_count"),
                hint_count=row.get("hint_count"),
                correct=row.get("correct"),
                original_label=None,
                aria_state_proxy=state,
                proxy_confidence=confidence,
            )

    # -- computability -------------------------------------------------
    def rule_computability(self) -> dict[str, Any]:
        """Which proxy rules can fire at all, given this dataset's fields.

        Derived from the declared field availability, not from the output, so a
        dataset that produces no states says why in advance rather than looking
        like an empty result.
        """
        names = ("correct", "attempt_count", "hint_count", "response_time")
        statuses = {n: self.FIELDS.get(n, FieldStatus("absent")) for n in names}
        has = {n: s.usable for n, s in statuses.items()}

        computable, blocked, vacuous = [], {}, {}
        for state, spec in RULE_REQUIREMENTS.items():
            missing = [f for f in spec["fields"] if not has.get(f)]
            if missing:
                blocked[state] = missing
                continue
            # Present, but is the rule's condition reachable?
            unreachable = None
            for field, (test, description) in spec["conditions"].items():
                status = statuses[field]
                if status.constant is not None and not test(status.constant):
                    unreachable = (
                        f"{field} is fixed at {status.constant:g} by the "
                        f"platform, and this rule needs {field} {description}")
                    break
            if unreachable:
                vacuous[state] = unreachable
            else:
                computable.append(state)

        return {
            "fields_usable": has,
            "rules_computable": computable,
            "rules_blocked": blocked,
            "rules_vacuous": vacuous,
            "distribution_check_computable": bool(computable),
            "timing_check_computable": has["response_time"] and bool(computable),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.source_dataset,
            "platform": self.platform,
            "year": self.year,
            "citation_key": self.citation_key,
            "commercial_use_allowed": self.commercial_use_allowed,
            "max_rows": self.max_rows,
            "fields": self.field_report(),
            "computability": self.rule_computability(),
        }
