"""Shared route I/O contract for external retrosynthesis baselines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RouteSearchConfig:
    """Backend-neutral retrosynthesis search request."""

    target_smiles: str
    stock_names: list[str] = field(default_factory=list)
    max_iterations: int = 10
    max_depth: int = 6
    expansion_topk: int = 50
    one_step_models: list[str] = field(default_factory=list)
    search_flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendFailure:
    """Structured failure returned instead of raising from benchmark loops."""

    category: str
    message: str
    target_smiles: str = ""
    retryable: bool = False
    raw_backend_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteStepCandidate:
    """One retrosynthetic step in a normalized route candidate."""

    product_smiles: str
    reactant_smiles: list[str]
    rxn_smiles: str
    source_model: str = ""
    score: float | None = None
    stock_status: dict[str, bool | None] = field(default_factory=dict)
    condition_predictions: list[dict[str, Any]] = field(default_factory=list)
    enzyme_ec_annotations: list[dict[str, Any]] = field(default_factory=list)
    catalyst_annotations: list[dict[str, Any]] = field(default_factory=list)
    raw_backend_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_enzyme_annotation(self) -> bool:
        if self.enzyme_ec_annotations:
            return True
        for item in self.catalyst_annotations:
            if item.get("ec_number") or item.get("uniprot_id"):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteCandidate:
    """Backend-neutral complete route candidate."""

    target_smiles: str
    steps: list[RouteStepCandidate] = field(default_factory=list)
    backend: str = ""
    score: float | None = None
    solved: bool = False
    stock_status: dict[str, bool | None] = field(default_factory=dict)
    search_time_s: float | None = None
    route_rank: int = 0
    raw_backend_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def enzymatic_step_present(self) -> bool:
        return any(step.has_enzyme_annotation for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["enzymatic_step_present"] = self.enzymatic_step_present
        return data


@dataclass
class BaselineRunResult:
    """Per-target result for an external backend run."""

    target_smiles: str
    backend: str
    routes: list[RouteCandidate] = field(default_factory=list)
    failures: list[BackendFailure] = field(default_factory=list)
    raw_backend_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return any(route.solved for route in self.routes)

    @property
    def route_count(self) -> int:
        return len(self.routes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "backend": self.backend,
            "solved": self.solved,
            "route_count": self.route_count,
            "routes": [route.to_dict() for route in self.routes],
            "failures": [failure.to_dict() for failure in self.failures],
            "raw_backend_metadata": self.raw_backend_metadata,
        }

