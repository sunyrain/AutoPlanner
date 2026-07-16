"""Stable provider interfaces for the AutoPlanner application layer.

Providers produce typed envelopes; they never mutate the blackboard or grant a
solved claim.  Correlation policy belongs to the trusted registry so multiple
LLM roles cannot declare themselves independent scientific sources.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable


class ProviderKind(str, Enum):
    PROPOSAL = "proposal"
    EVIDENCE = "evidence"
    STOCK = "stock"
    VERIFIER = "verifier"
    AGENT_BACKEND = "agent_backend"
    ARTIFACT_STORE = "artifact_store"
    RENDERER = "renderer"
    EXPERIMENT_EXECUTOR = "experiment_executor"


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    kind: ProviderKind
    version: str
    input_schemas: tuple[str, ...]
    output_schemas: tuple[str, ...]
    correlation_group: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    deterministic: bool = False
    network_access: bool = False
    estimated_cost_units: float = 0.0
    schema_version: ClassVar[str] = "provider_descriptor.v1"

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id is required")
        if not self.version.strip():
            raise ValueError("provider version is required")
        if not self.input_schemas or not self.output_schemas:
            raise ValueError("provider schemas are required")
        if not self.correlation_group.strip():
            raise ValueError("provider correlation_group is required")
        if self.estimated_cost_units < 0:
            raise ValueError("estimated_cost_units must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["schema_version"] = self.schema_version
        return row


@dataclass(frozen=True)
class ProviderContext:
    run_id: str
    case_id: str
    target_smiles: str
    artifact_revision_id: str = ""
    budget_remaining: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: ClassVar[str] = "provider_context.v1"

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class ProviderResultEnvelope:
    provider_id: str
    provider_version: str
    provider_kind: ProviderKind
    correlation_group: str
    output_schema: str
    accepted: bool
    payload: Mapping[str, Any]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    no_solved_claim: bool = True
    schema_version: ClassVar[str] = "provider_result_envelope.v1"

    def __post_init__(self) -> None:
        if not self.provider_id or not self.provider_version or not self.output_schema:
            raise ValueError("provider result identity is incomplete")
        if self.no_solved_claim is not True:
            raise ValueError("provider results cannot carry solved authority")

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        row = asdict(self)
        row["provider_kind"] = self.provider_kind.value
        row["schema_version"] = self.schema_version
        if include_hash:
            row["content_hash"] = _content_hash(row)
        return row


@runtime_checkable
class Provider(Protocol):
    @property
    def descriptor(self) -> ProviderDescriptor: ...

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope: ...


@runtime_checkable
class ProposalProvider(Provider, Protocol):
    """Produces reaction candidate envelopes only; never proof."""


@runtime_checkable
class EvidenceProvider(Provider, Protocol):
    """Produces source/document/evidence claims with provenance."""


@runtime_checkable
class StockProvider(Provider, Protocol):
    """Produces immutable stock boundaries or timestamped supplier offers."""


@runtime_checkable
class VerifierProvider(Provider, Protocol):
    """Produces deterministic verification overlays."""


@runtime_checkable
class AgentBackendProvider(Provider, Protocol):
    """Runs agent work and emits observed runtime events plus draft results."""


@runtime_checkable
class ArtifactStoreProvider(Provider, Protocol):
    """Persists content-addressed artifacts and manifests."""


@runtime_checkable
class RendererProvider(Provider, Protocol):
    """Projects canonical graph records into a read-only view artifact."""


@runtime_checkable
class ExperimentExecutorProvider(Provider, Protocol):
    """Accepts a bounded experiment request; never grants validation."""


def validate_provider_result(
    value: Any,
    *,
    descriptor: ProviderDescriptor | None = None,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["provider_result_not_object"]
    row = dict(value)
    reasons: list[str] = []
    if row.get("schema_version") != ProviderResultEnvelope.schema_version:
        reasons.append("invalid_provider_result_schema")
    if row.get("no_solved_claim") is not True:
        reasons.append("provider_result_missing_no_solved_claim")
    if row.get("accepted") not in {True, False}:
        reasons.append("provider_result_accepted_not_boolean")
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        reasons.append("provider_result_payload_not_object")
    if descriptor is not None:
        if row.get("provider_id") != descriptor.provider_id:
            reasons.append("provider_result_id_mismatch")
        if row.get("provider_version") != descriptor.version:
            reasons.append("provider_result_version_mismatch")
        if row.get("provider_kind") != descriptor.kind.value:
            reasons.append("provider_result_kind_mismatch")
        if row.get("correlation_group") != descriptor.correlation_group:
            reasons.append("provider_result_correlation_group_mismatch")
        if row.get("output_schema") not in descriptor.output_schemas:
            reasons.append("provider_result_output_schema_not_declared")
    supplied_hash = str(row.pop("content_hash", ""))
    if not supplied_hash or supplied_hash != _content_hash(row):
        reasons.append("provider_result_content_hash_mismatch")
    return sorted(set(reasons))


def _content_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
