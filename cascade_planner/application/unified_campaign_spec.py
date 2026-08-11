"""Immutable, target-blind input contract for one campaign.

The contract deliberately excludes target names, dataset identifiers and
acceptance modes.  Those values may exist in adapters or quality projections,
but they are not inputs to candidate generation, scheduling or budgeting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import math
from typing import Any, Mapping

from cascade_planner.application.campaign_contract_json import (
    bound_row as _bound_row,
    digest as _digest,
    freeze_json as _freeze_json,
    is_sha256 as _is_sha256,
    normalized_strings as _normalized_strings,
    plain_json as _plain_json,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)


UNIFIED_CAMPAIGN_SPEC_SCHEMA = "unified_campaign_spec.v1"
STOCK_ORACLE_REFERENCE_SCHEMA = "stock_oracle_reference.v1"
TARGET_CONSTRAINTS_SCHEMA = "target_constraints.v1"
CAMPAIGN_RESOURCE_BUDGET_SCHEMA = "campaign_resource_budget.v1"
_BOUNDARIES = frozenset({"benchmark_search", "procurement", "in_house"})
_EXECUTION_DOMAINS = frozenset(
    {"chemical", "biocatalytic", "whole_cell", "hybrid", "mechanistic"}
)
_CONTROL_TOKENS = (
    "dataset",
    "objective",
    "benchmark",
    "retrostar",
    "paroutes",
)


@dataclass(frozen=True, slots=True)
class CampaignResourceBudget:
    """Run-wide budget vector, independent of any target or dataset label."""

    model: RetrosynthesisRunBudget = field(default_factory=RetrosynthesisRunBudget)
    max_total_tasks: int = 256
    max_evidence_tasks: int = 64
    max_stock_tasks: int = 128
    max_validation_tasks: int = 128
    max_program_tasks: int = 64
    max_experiment_tasks: int = 32
    max_run_wall_time_s: float = 7_200.0
    schema_version: str = CAMPAIGN_RESOURCE_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        integers = (
            self.max_total_tasks,
            self.max_evidence_tasks,
            self.max_stock_tasks,
            self.max_validation_tasks,
            self.max_program_tasks,
            self.max_experiment_tasks,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in integers):
            raise ValueError("campaign resource limits cannot be negative")
        if not math.isfinite(float(self.max_run_wall_time_s)) or (
            self.max_run_wall_time_s < 0
        ):
            raise ValueError("max_run_wall_time_s must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "max_total_tasks": self.max_total_tasks,
            "max_evidence_tasks": self.max_evidence_tasks,
            "max_stock_tasks": self.max_stock_tasks,
            "max_validation_tasks": self.max_validation_tasks,
            "max_program_tasks": self.max_program_tasks,
            "max_experiment_tasks": self.max_experiment_tasks,
            "max_run_wall_time_s": self.max_run_wall_time_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignResourceBudget":
        row = dict(value)
        model_row = dict(row.get("model") or {})
        model_row.pop("schema_version", None)
        model_row.pop("content_sha256", None)
        return cls(
            model=RetrosynthesisRunBudget(**model_row),
            max_total_tasks=int(row.get("max_total_tasks", 256)),
            max_evidence_tasks=int(row.get("max_evidence_tasks", 64)),
            max_stock_tasks=int(row.get("max_stock_tasks", 128)),
            max_validation_tasks=int(row.get("max_validation_tasks", 128)),
            max_program_tasks=int(row.get("max_program_tasks", 64)),
            max_experiment_tasks=int(row.get("max_experiment_tasks", 32)),
            max_run_wall_time_s=float(row.get("max_run_wall_time_s", 7_200.0)),
        )


@dataclass(frozen=True, slots=True)
class TargetConstraints:
    """Chemistry/execution constraints that cannot encode benchmark identity."""

    forbidden_reagents: tuple[str, ...] = ()
    max_route_steps: int | None = None
    allowed_execution_domains: tuple[str, ...] = tuple(sorted(_EXECUTION_DOMAINS))
    safety_limits: Mapping[str, Any] = field(default_factory=dict)
    stock_source_ids: tuple[str, ...] = ()
    schema_version: str = TARGET_CONSTRAINTS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forbidden_reagents",
            _normalized_strings(self.forbidden_reagents),
        )
        object.__setattr__(
            self,
            "allowed_execution_domains",
            _normalized_strings(self.allowed_execution_domains),
        )
        object.__setattr__(
            self,
            "stock_source_ids",
            _normalized_strings(self.stock_source_ids),
        )
        domains = set(self.allowed_execution_domains)
        if not domains or not domains.issubset(_EXECUTION_DOMAINS):
            raise ValueError("target execution domains are invalid")
        if self.max_route_steps is not None and (
            isinstance(self.max_route_steps, bool)
            or not isinstance(self.max_route_steps, int)
            or self.max_route_steps < 1
        ):
            raise ValueError("max_route_steps must be a positive integer or null")
        safety = _freeze_json(dict(self.safety_limits))
        for key in safety:
            normalized = str(key).casefold()
            if any(token in normalized for token in _CONTROL_TOKENS):
                raise ValueError("target constraints cannot encode dataset controls")
        object.__setattr__(self, "safety_limits", safety)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "forbidden_reagents": list(self.forbidden_reagents),
            "max_route_steps": self.max_route_steps,
            "allowed_execution_domains": list(self.allowed_execution_domains),
            "safety_limits": _plain_json(self.safety_limits),
            "stock_source_ids": list(self.stock_source_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetConstraints":
        row = dict(value)
        allowed = {
            "schema_version",
            "forbidden_reagents",
            "max_route_steps",
            "allowed_execution_domains",
            "safety_limits",
            "stock_source_ids",
        }
        if set(row) - allowed:
            raise ValueError("target constraints contain unsupported fields")
        if row.get("schema_version", TARGET_CONSTRAINTS_SCHEMA) != (
            TARGET_CONSTRAINTS_SCHEMA
        ):
            raise ValueError("target constraints schema is invalid")
        return cls(
            forbidden_reagents=tuple(row.get("forbidden_reagents") or ()),
            max_route_steps=row.get("max_route_steps"),
            allowed_execution_domains=tuple(
                row.get("allowed_execution_domains") or sorted(_EXECUTION_DOMAINS)
            ),
            safety_limits=dict(row.get("safety_limits") or {}),
            stock_source_ids=tuple(row.get("stock_source_ids") or ()),
        )


@dataclass(frozen=True, slots=True)
class StockOracleReference:
    """Content-bound reference to stock authority or a snapshot resolver."""

    oracle_id: str
    boundary: str
    binding: Mapping[str, Any]
    schema_version: str = STOCK_ORACLE_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if not str(self.oracle_id or "").strip():
            raise ValueError("stock oracle id is required")
        if self.boundary not in _BOUNDARIES:
            raise ValueError("stock oracle boundary is invalid")
        binding = dict(self.binding)
        supplied = str(binding.pop("content_sha256", "")).lower()
        if not _is_sha256(supplied) or supplied != _digest(binding):
            raise ValueError("stock oracle binding digest is invalid")
        binding["content_sha256"] = supplied
        object.__setattr__(self, "binding", _freeze_json(binding))

    @property
    def binding_sha256(self) -> str:
        return str(self.binding.get("content_sha256") or "")

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "oracle_id": self.oracle_id,
            "boundary": self.boundary,
            "binding": _plain_json(self.binding),
            "semantics": {
                "reference_is_immutable": True,
                "positive_stock_requires_content_bound_observation": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StockOracleReference":
        row = dict(value)
        supplied = str(row.pop("content_sha256", "")).lower()
        if row.get("schema_version") != STOCK_ORACLE_REFERENCE_SCHEMA:
            raise ValueError("stock oracle reference schema is invalid")
        if supplied != _digest(row):
            raise ValueError("stock oracle reference digest is invalid")
        return cls(
            oracle_id=str(row.get("oracle_id") or ""),
            boundary=str(row.get("boundary") or ""),
            binding=dict(row.get("binding") or {}),
        )

    @classmethod
    def from_binding(
        cls,
        *,
        oracle_id: str,
        boundary: str,
        binding: Mapping[str, Any],
    ) -> "StockOracleReference":
        return cls(oracle_id=oracle_id, boundary=boundary, binding=dict(binding))

    @classmethod
    def compatibility_unbound(cls, *, boundary: str) -> "StockOracleReference":
        binding = _bound_row(
            {
                "schema_version": "stock_oracle_binding.v1",
                "kind": "compatibility_unbound",
                "positive_authority": False,
            }
        )
        return cls(
            oracle_id="compatibility-unbound-stock-oracle",
            boundary=boundary,
            binding=binding,
        )


@dataclass(frozen=True, slots=True)
class UnifiedCampaignSpec:
    """The only algorithm-facing input for a new campaign."""

    target_smiles: str
    stock_oracle: StockOracleReference
    constraints: TargetConstraints = field(default_factory=TargetConstraints)
    resource_budget: CampaignResourceBudget = field(
        default_factory=CampaignResourceBudget
    )
    schema_version: str = UNIFIED_CAMPAIGN_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if not str(self.target_smiles or "").strip():
            raise ValueError("campaign target SMILES is required")

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "target": {"canonical_smiles": self.target_smiles},
            "stock_oracle": self.stock_oracle.to_dict(),
            "constraints": self.constraints.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "semantics": {
                "display_metadata_is_not_an_input": True,
                "dataset_identity_is_not_an_input": True,
                "acceptance_is_a_quality_projection": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnifiedCampaignSpec":
        row = dict(value)
        supplied = str(row.pop("content_sha256", "")).lower()
        if row.get("schema_version") != UNIFIED_CAMPAIGN_SPEC_SCHEMA:
            raise ValueError("unified campaign spec schema is invalid")
        if supplied != _digest(row):
            raise ValueError("unified campaign spec digest is invalid")
        target = dict(row.get("target") or {})
        if set(target) != {"canonical_smiles"}:
            raise ValueError("unified campaign target fields are invalid")
        return cls(
            target_smiles=str(target.get("canonical_smiles") or ""),
            stock_oracle=StockOracleReference.from_dict(
                dict(row.get("stock_oracle") or {})
            ),
            constraints=TargetConstraints.from_dict(
                dict(row.get("constraints") or {})
            ),
            resource_budget=CampaignResourceBudget.from_dict(
                dict(row.get("resource_budget") or {})
            ),
        )


def stock_oracle_reference_from_builder(
    builder: Any,
    *,
    boundary: str,
) -> StockOracleReference:
    """Bind a frozen index/snapshot builder or its immutable resolver contract."""

    explicit = getattr(builder, "stock_oracle_binding", None)
    if isinstance(explicit, Mapping):
        binding = dict(explicit)
        oracle_id = str(binding.get("oracle_id") or "host-stock-oracle")
    elif _is_sha256(str(getattr(builder, "index_sha256", ""))):
        binding = _bound_row(
            {
                "schema_version": "stock_oracle_binding.v1",
                "kind": "frozen_benchmark_index",
                "index_sha256": str(builder.index_sha256),
                "source_sha256": str(getattr(builder, "source_sha256", "")),
                "catalog_name": str(getattr(builder, "catalog_name", "")),
                "member_count": int(getattr(builder, "member_count", 0)),
            }
        )
        oracle_id = f"frozen-index:{str(builder.index_sha256)[:24]}"
    else:
        subject = builder if inspect.isfunction(builder) else type(builder)
        module = str(getattr(subject, "__module__", ""))
        qualname = str(getattr(subject, "__qualname__", ""))
        try:
            source = inspect.getsource(subject)
        except (OSError, TypeError):
            source = f"{module}:{qualname}"
        binding = _bound_row(
            {
                "schema_version": "stock_oracle_binding.v1",
                "kind": "snapshot_resolver_contract",
                "callable_module": module,
                "callable_qualname": qualname,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "outputs_require_content_addressing": True,
            }
        )
        oracle_id = f"snapshot-resolver:{_digest(binding)[:24]}"
    return StockOracleReference(
        oracle_id=oracle_id,
        boundary=boundary,
        binding=binding,
    )


__all__ = [
    "CAMPAIGN_RESOURCE_BUDGET_SCHEMA",
    "CampaignResourceBudget",
    "STOCK_ORACLE_REFERENCE_SCHEMA",
    "StockOracleReference",
    "TARGET_CONSTRAINTS_SCHEMA",
    "TargetConstraints",
    "UNIFIED_CAMPAIGN_SPEC_SCHEMA",
    "UnifiedCampaignSpec",
    "stock_oracle_reference_from_builder",
]
