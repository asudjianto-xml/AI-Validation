"""AI for Validation (aiv): general-purpose model and GenAI validation tooling.

The package covers predictive-model validation (weak-region discovery over the
residual field and over an error model) and GenAI validation (KG-grounded RAG,
adversarial prompting, claim verification). A credit-default dataset is bundled
as a worked example (see `default_data_path`) so the core workflow runs with no
external files; any tabular dataset with a binary target works the same way. The
`kg_rag` and `report` modules additionally need an on-disk GMS store and the
`claude` CLI at run time (their `knowlytix` dependency installs from PyPI).

Prompt optimization for the reporting workflow uses a vendored subset of
Frontier-Discovery under `aiv._vendor.prompt_factorization`, so the package depends
on no other repository.
"""
from __future__ import annotations

from aiv.datasets import default_data_path
from aiv.report import (
    REPORT_SCHEMA,
    EvidencePacket,
    Fact,
    assemble_packets,
    load_provenance,
    optimize_instruction,
    score_section,
)
from aiv.predictive import (
    FanovaModel,
    PredictiveAdapter,
    PredictiveSystem,
    SegmentedSystem,
    apply_stress,
    slice_mask,
)
from aiv.frontier_selection import (
    Ownership,
    coverage_shares,
    frontier_owners,
    selection_probabilities,
)
from aiv.weak_region import (
    CentroidRouter,
    RadiusRouter,
    WeakRegion,
    discover_weak_regions,
    weak_membership,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "default_data_path",
    "FanovaModel",
    "PredictiveSystem",
    "SegmentedSystem",
    "PredictiveAdapter",
    "slice_mask",
    "apply_stress",
    "Ownership",
    "frontier_owners",
    "coverage_shares",
    "selection_probabilities",
    "WeakRegion",
    "CentroidRouter",
    "RadiusRouter",
    "discover_weak_regions",
    "weak_membership",
]
