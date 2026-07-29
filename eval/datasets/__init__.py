"""
ARIA dataset acquisition + adaptation layer.

    from datasets.registry import REGISTRY, get_spec, all_specs
    spec = get_spec("assistments2009")
    adapter = spec.resolve_adapter()(spec)
    for rec in adapter.to_aria_schema():
        ...

Sub-modules:
  registry   — DatasetSpec for each external source (single source of truth)
  download   — CLI to list / fetch datasets (auto where possible, else instructions)
  adapters   — per-source schema adapters into ARIA's unified record format
"""

from .registry import REGISTRY, DatasetSpec, get_spec, all_specs, names

__all__ = ["REGISTRY", "DatasetSpec", "get_spec", "all_specs", "names"]
