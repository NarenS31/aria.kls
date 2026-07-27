# `datasets/` — external data acquisition + adaptation for ARIA

This package turns ARIA from a synthetic-only system into a multi-source,
externally validated one. It knows how to (1) *acquire* eight external education
datasets and (2) *adapt* them into one unified ARIA schema, keeping real labels
and derived proxies rigorously separate.

```
datasets/
├── registry.py            # DatasetSpec for each source (single source of truth)
├── download.py            # CLI: --list / --fetch <name> / --fetch-all
├── adapters/
│   ├── base.py            # DatasetAdapter ABC + aria_record() + validate()
│   ├── assistments.py     # behavioral proxy mapping (documented, confidence-tiered)
│   ├── eduagent.py        # real gaze-measure → ARIA crosswalk
│   ├── ncte.py            # real classroom dialogue (reasoning turns)
│   ├── eedi.py            # behavioral proxy; NON-COMMERCIAL enforced
│   ├── edm_thinkaloud.py  # real think-aloud + SRL labels (access-controlled)
│   └── eduagent_crosswalk.md
├── REQUEST_EMAIL.md       # ready-to-send email for the EDM think-aloud data
└── README.md
```

## Quick start

```bash
# from the repo root (/Users/narensara/aria/eval)
python3.11 datasets/download.py --list          # what exists, licences, actions
python3.11 datasets/download.py --fetch eduagent310
python3.11 datasets/download.py --fetch-all      # auto where possible; instructions for the rest
```

```python
from datasets.registry import get_spec
spec = get_spec("assistments2009")
adapter = spec.resolve_adapter()(spec)
report = adapter.validate()          # coverage, null rates, class balance
for rec in adapter.to_aria_schema(): # unified ARIA records
    ...
```

## The eight registered sources

| name | modality | real cognitive labels | commercial | acquisition |
|------|----------|-----------------------|------------|-------------|
| `assistments2009` | behavioral | no (proxies) | yes | manual (Google Sites) |
| `eduagent310` | behavioral | **yes** (gaze measures) | no | auto (GitHub) |
| `eduagent705` | behavioral | attributes | no | auto (GitHub) |
| `ncte` | dialogue | no (discourse moves) | no | manual (data form) |
| `eedi` | behavioral | no (proxies) | **NO — CC BY-NC-ND** | manual |
| `moocradar` | behavioral | no | no | auto (GitHub) |
| `xes3g5m` | behavioral | no | yes (MIT) | auto (Google Drive) |
| `edm_thinkaloud` | think_aloud | **yes** (SRL labels) | no | **email request** |

## The unified ARIA schema

Every adapter yields records shaped by `aria_record()`:

```python
{
  "source_dataset": str, "source_record_id": str,
  "modality": "think_aloud"|"behavioral"|"dialogue",
  "text": str | None,
  "behavioral_features": {"response_time_ms","attempt_count","hint_count","correct"},
  "original_label": str | None,        # the source's OWN label, verbatim
  "aria_state_proxy": str | None,      # ARIA state; PROXY unless source has real labels
  "proxy_confidence": "high"|"medium"|"low"|None,
  "proxy_method": str,                 # exactly how the state was derived
  "commercial_use_allowed": bool,      # propagated from the licence
  "citation_key": str                  # matches data/research/citations.json
}
```

## Honesty rules (enforced, not aspirational)

- **Proxies are never ground truth.** For behavioral sources, `aria_state_proxy`
  is derived by documented rules with a confidence tier; downstream reports label
  these PROXY, never "validated against real labels".
- **Missing mappings are `None`, never a guess** (e.g. EduAgent `inattention`,
  ASSISTments `INSIGHT`).
- **Licences propagate.** `commercial_use_allowed=False` (Eedi, EduAgent, NCTE,
  EDM) flows into every record; Eedi additionally prints a non-commercial banner.
- **No raw data is committed.** `download.py` adds `data/external/` to
  `.gitignore` (keeping only `LICENSE.txt` files) and writes a provenance
  `LICENSE.txt` per source from its spec.

See `../eval/external_validation.py` for the experiments that consume these
adapters, and `../eval/evidence_report.py` for the evidence-tier accounting.
