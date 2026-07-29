"""
Dataset registry for ARIA external validation.

Each external source ARIA can validate against is described by a single
``DatasetSpec``. The registry is the one place that knows a dataset's licence,
modality, whether it can be fetched automatically, and which adapter maps it
into ARIA's schema. ``download.py`` and the adapters both read from here so
there is exactly one source of truth.

IMPORTANT honesty rules encoded here:
  * ``commercial_use_allowed`` is carried verbatim into every derived record and
    every report. Eedi in particular is CC BY-NC-ND 4.0 — non-commercial only.
  * ``has_cognitive_labels`` marks whether a source has *real* (human-derived or
    directly-measured) cognitive-state labels. Where it is False, any ARIA state
    attached downstream is a behavioural PROXY, never ground truth.
  * ``citation_key`` matches a key in data/research/citations.json.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EXTERNAL_ROOT = os.path.join(REPO_ROOT, "data", "external")

MODALITIES = ("think_aloud", "behavioral", "dialogue")
DOWNLOAD_METHODS = ("github", "google_drive", "dataverse", "manual", "email")


@dataclass
class DatasetSpec:
    """Everything needed to acquire, licence-check, and adapt one dataset."""

    name: str
    url: str
    license: str
    requires_manual_download: bool
    requires_email_request: bool
    modality: str                      # one of MODALITIES
    has_cognitive_labels: bool
    commercial_use_allowed: bool
    citation_key: str                  # key in data/research/citations.json
    adapter_class: Optional[str]       # dotted import path, or None if no adapter
    short_description: str = ""
    approx_size: str = ""
    # How download.py should try to fetch it, and the parameters it needs.
    download_method: str = "manual"    # one of DOWNLOAD_METHODS
    download_config: dict = field(default_factory=dict)
    notes: str = ""
    local_path: Optional[str] = None   # data/external/<name>/raw ; auto-filled

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"{self.name}: bad modality {self.modality!r}")
        if self.download_method not in DOWNLOAD_METHODS:
            raise ValueError(f"{self.name}: bad download_method {self.download_method!r}")
        if self.local_path is None:
            self.local_path = os.path.join(EXTERNAL_ROOT, self.name, "raw")

    # -- derived paths ------------------------------------------------
    @property
    def root_dir(self) -> str:
        return os.path.join(EXTERNAL_ROOT, self.name)

    @property
    def raw_dir(self) -> str:
        return self.local_path  # type: ignore[return-value]

    @property
    def license_path(self) -> str:
        return os.path.join(self.root_dir, "LICENSE.txt")

    # -- status -------------------------------------------------------
    def is_present(self) -> bool:
        """True if raw_dir exists and contains at least one non-hidden file."""
        d = self.raw_dir
        if not d or not os.path.isdir(d):
            return False
        for _root, _dirs, files in os.walk(d):
            for f in files:
                if not f.startswith("."):
                    return True
        return False

    def resolve_adapter(self):
        """Import and return the adapter class, or None if unset."""
        if not self.adapter_class:
            return None
        module_path, _, cls_name = self.adapter_class.rpartition(".")
        mod = __import__(module_path, fromlist=[cls_name])
        return getattr(mod, cls_name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["local_path"] = self.local_path
        return d


# ==================================================================
# The eight registered sources
# ==================================================================

REGISTRY: dict[str, DatasetSpec] = {}


def _register(spec: DatasetSpec) -> None:
    REGISTRY[spec.name] = spec


_register(DatasetSpec(
    name="assistments2009",
    url="https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data",
    license="Open for research use (ASSISTments); attribution required",
    requires_manual_download=True,     # Google Sites, no stable direct link
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=False,        # cognitive states derived as proxies
    commercial_use_allowed=True,
    citation_key="feng_2009",
    adapter_class="datasets.adapters.assistments.AssistmentsAdapter",
    short_description="ASSISTments 2009-2010 Skill Builder response logs.",
    approx_size="~300k response logs, ~4k students",
    download_method="manual",
    download_config={
        "expected_files": ["skill_builder_data.csv"],
        "instructions": (
            "1. Open the ASSISTments data page (URL above).\n"
            "2. Under '2009-2010 ASSISTment Data', download the "
            "'Skill Builder Data' CSV (skill_builder_data.csv, sometimes "
            "distributed as skill_builder_data_corrected.csv).\n"
            "3. Place the CSV in the target path shown below."
        ),
    },
    notes="Behavioral only. ARIA states are derived as documented proxies.",
))

_register(DatasetSpec(
    name="eduagent310",
    url="https://github.com/EduAgent/EduAgent",
    license="See repository (research use); verify before redistribution",
    requires_manual_download=False,
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=True,         # inferred cognitive states for real students
    commercial_use_allowed=False,      # unverified; treat as research-only
    citation_key="xu_2024_eduagent",
    adapter_class="datasets.adapters.eduagent.EduAgentAdapter",
    short_description="EduAgent310: 310 real students with inferred cognitive "
                      "states, eye-gaze and mouse-movement behavior.",
    approx_size="310 real students",
    download_method="github",
    download_config={
        "owner": "EduAgent", "repo": "EduAgent", "branch": "main",
        "file_patterns": [".csv", ".json", ".jsonl"],
        # student_answer_item_revised.csv carries the real gaze-derived cognitive
        # measures (confusion_dur, inattention_dur) per student-question;
        # student_demo.csv is demographics.
        "priority_files": ["student_answer_item_revised.csv", "student_demo.csv"],
        "variant": "real",
    },
    notes="HIGHEST-VALUE target: real inferred cognitive labels enable a DIRECT "
          "validation (Experiment D). Attempt HuggingFace/GitHub auto-download.",
))

_register(DatasetSpec(
    name="eduagent705",
    url="https://github.com/EduAgent/EduAgent",
    license="See repository (research use); verify before redistribution",
    requires_manual_download=False,
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=True,         # synthetic agents carry attribute labels
    commercial_use_allowed=False,
    citation_key="xu_2024_eduagent",
    adapter_class="datasets.adapters.eduagent.EduAgentAdapter",
    short_description="EduAgent705: 705 synthetic agents with attitude / "
                      "concentration / prior-knowledge attributes.",
    approx_size="705 synthetic agents",
    download_method="github",
    download_config={
        "owner": "EduAgent", "repo": "EduAgent", "branch": "main",
        "file_patterns": [".csv", ".json", ".jsonl"],
        "priority_files": ["student_demo_generated.csv"],
        "variant": "synthetic",
    },
    notes="Second synthetic corpus for cross-corpus comparison (Experiment C).",
))

_register(DatasetSpec(
    name="ncte",
    url="https://github.com/ddemszky/classroom-transcript-analysis",
    license="Research use; access via NCTE data-request form",
    requires_manual_download=True,     # transcripts gated behind a Google Form
    requires_email_request=False,
    modality="dialogue",
    has_cognitive_labels=False,        # discourse-move labels, not ARIA states
    commercial_use_allowed=False,
    citation_key="demszky_2023",
    adapter_class="datasets.adapters.ncte.NCTEAdapter",
    short_description="NCTE Transcripts: real 4th/5th-grade math classroom "
                      "transcripts with turn-level discourse-move annotations.",
    approx_size="1,660 lessons, 317 teachers",
    download_method="github",          # code repo public; full data needs form
    download_config={
        "owner": "ddemszky", "repo": "classroom-transcript-analysis",
        "branch": "main",
        "file_patterns": [".csv", ".json", ".jsonl"],
        "form_url": "https://forms.gle/1yWybvsjciqL8Y9p8",
        "instructions": (
            "The full NCTE transcripts are released only after filling the "
            "data-request form (form_url). This fetch retrieves the public "
            "GitHub repo (annotation schema + any sample data); place the "
            "form-provided transcript CSVs in the target path when granted."
        ),
    },
    notes="Best non-LLM human-text robustness test (Experiment B). Real "
          "classroom speech, so it stress-tests synthetic-to-real transfer.",
))

_register(DatasetSpec(
    name="eedi",
    url="https://eedi.com/projects/neurips-education-challenge",
    license="CC BY-NC-ND 4.0",
    requires_manual_download=True,
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=False,
    commercial_use_allowed=False,      # HARD constraint — non-commercial only
    citation_key="wang_2020",
    adapter_class="datasets.adapters.eedi.EediAdapter",
    short_description="Eedi / NeurIPS 2020 Education Challenge answer records.",
    approx_size="~17M answer records, ages 7-18",
    download_method="manual",
    download_config={
        "expected_files": ["train_task_1_2.csv", "answer_metadata_task_1_2.csv"],
        "instructions": (
            "1. Register / accept the CC BY-NC-ND 4.0 terms on the Eedi / "
            "NeurIPS 2020 Education Challenge page (URL above).\n"
            "2. Download the Task 1&2 answer records + metadata CSVs.\n"
            "3. Place them in the target path.\n"
            "NOTE: CC BY-NC-ND 4.0 — NON-COMMERCIAL, no derivatives. Every "
            "report that touches Eedi prints a non-commercial banner."
        ),
    },
    notes="commercial_use_allowed is FALSE and is propagated to every record.",
))

_register(DatasetSpec(
    name="moocradar",
    url="https://github.com/THU-KEG/MOOC-Radar",
    license="Research use (THU-KEG); verify before redistribution",
    requires_manual_download=False,
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=False,
    commercial_use_allowed=False,
    citation_key="yu_2023",
    adapter_class=None,                # registered for acquisition; no adapter yet
    short_description="MOOCRadar: exercise + concept + behavioral MOOC records.",
    approx_size="~2.5k exercises, ~14k students, 12M behaviors",
    download_method="github",
    download_config={
        "owner": "THU-KEG", "repo": "MOOC-Radar", "branch": "main",
        "file_patterns": [".json", ".jsonl", ".csv"],
        "instructions": (
            "Large files are distributed via download links in the repo "
            "README (cloud drive). Auto-fetch pulls small repo data files; "
            "follow the README links for the full behavioral logs."
        ),
    },
    notes="No ARIA adapter yet — registered for acquisition and completeness.",
))

_register(DatasetSpec(
    name="xes3g5m",
    url="https://github.com/ai4ed/XES3G5M",
    license="MIT",
    requires_manual_download=False,
    requires_email_request=False,
    modality="behavioral",
    has_cognitive_labels=False,
    commercial_use_allowed=True,       # MIT
    citation_key="liu_2023",
    adapter_class=None,                # registered for acquisition; no adapter yet
    short_description="XES3G5M: K-12 knowledge-tracing benchmark with question text.",
    approx_size="~5M interactions, ~18k students, ~8k questions",
    download_method="google_drive",
    download_config={
        "drive_file_id": "1eFiIYyh5O2V90RA0brammGH6EpHvPDQe",
        "owner": "ai4ed", "repo": "XES3G5M", "branch": "main",
        "instructions": (
            "The full dataset is hosted on Google Drive (drive_file_id above). "
            "Auto-fetch attempts the Drive download; if Drive blocks scripted "
            "access, download manually from the repo's README link and place "
            "the files in the target path."
        ),
    },
    notes="MIT-licensed; behavioral + question text.",
))

_register(DatasetSpec(
    name="edm_thinkaloud",
    url="https://doi.org/10.5281/zenodo.12729790",
    license="Author-controlled; access by request",
    requires_manual_download=False,
    requires_email_request=True,       # the only email-gated source
    modality="think_aloud",
    has_cognitive_labels=True,         # real SRL labels on real think-aloud text
    commercial_use_allowed=False,
    citation_key="zhang_2024",
    adapter_class="datasets.adapters.edm_thinkaloud.EDMThinkAloudAdapter",
    short_description="EDM 2024 Think-Aloud (Zhang, Borchers, Aleven, Baker): "
                      "real think-aloud text with SRL labels.",
    approx_size="Human think-aloud protocols with SRL annotations",
    download_method="email",
    download_config={
        "request_email_path": "datasets/REQUEST_EMAIL.md",
        "contacts": ["Conrad Borchers (CMU)", "Jiayi Zhang (Penn)"],
        "instructions": (
            "This dataset is not openly downloadable. Send the prepared request "
            "email (see request_email_path) to the authors. When access is "
            "granted, place the released files in the target path; the adapter "
            "is already implemented against the published schema."
        ),
    },
    notes="THE only source that directly matches ARIA's input modality (real "
          "think-aloud + SRL labels). Enables the strongest possible DIRECT "
          "validation once access is granted.",
))


# ==================================================================
# Convenience accessors
# ==================================================================

def all_specs() -> list[DatasetSpec]:
    return list(REGISTRY.values())


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}. Registered: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY.keys())
