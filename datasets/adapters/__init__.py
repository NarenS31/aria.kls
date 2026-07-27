"""ARIA dataset adapters: map external sources into ARIA's unified schema."""

from .base import (
    DatasetAdapter,
    DatasetNotAvailableError,
    aria_record,
    ARIA_STATES,
    MODALITIES,
    CONFIDENCE_TIERS,
)
from .assistments import AssistmentsAdapter
from .eduagent import EduAgentAdapter
from .ncte import NCTEAdapter
from .eedi import EediAdapter, NON_COMMERCIAL_BANNER
from .edm_thinkaloud import EDMThinkAloudAdapter

__all__ = [
    "DatasetAdapter",
    "DatasetNotAvailableError",
    "aria_record",
    "ARIA_STATES",
    "MODALITIES",
    "CONFIDENCE_TIERS",
    "AssistmentsAdapter",
    "EduAgentAdapter",
    "NCTEAdapter",
    "EediAdapter",
    "NON_COMMERCIAL_BANNER",
    "EDMThinkAloudAdapter",
]
