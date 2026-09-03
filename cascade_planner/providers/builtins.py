"""Built-in adapters that expose mainline services through the provider SPI."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
)
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.providers.stock import SnapshotStockProvider
from cascade_planner.providers.experiment import ManualExperimentExecutorProvider
from cascade_planner.providers.http_experiment import HttpExperimentExecutorProvider


_HOST_BUILTIN_TRUST_POLICY: dict[str, dict[str, Any]] = {
    "autoplanner.snapshot_stock": {
        "kind": ProviderKind.STOCK,
        "correlation_group": "stock_snapshot",
        "deterministic": True,
    },
    "autoplanner.benchmark_catalog_stock": {
        "kind": ProviderKind.STOCK,
        "correlation_group": "benchmark_catalog_artifact",
        "deterministic": True,
    },
    "autoplanner.reaction_route_verifier": {
        "kind": ProviderKind.VERIFIER,
        "correlation_group": "deterministic_reaction_verifier",
        "deterministic": True,
    },
    "autoplanner.chemenzy_proposals": {
        "kind": ProviderKind.PROPOSAL,
        "correlation_group": "computational:chem_enzy",
        "deterministic": False,
    },
    "autoplanner.literature_evidence": {
        "kind": ProviderKind.EVIDENCE,
        "correlation_group": "literature_evidence_pipeline",
        "deterministic": False,
    },
    "autoplanner.manual_experiment_executor": {
        "kind": ProviderKind.EXPERIMENT_EXECUTOR,
        "correlation_group": "manual_experiment_handoff",
        "deterministic": True,
    },
}


def _host_trusted_builtin_descriptor(provider: Any) -> ProviderDescriptor:
    claimed = provider.descriptor
    policy = _HOST_BUILTIN_TRUST_POLICY.get(claimed.provider_id)
    if policy is None:
        raise ValueError(f"missing host trust policy for builtin provider {claimed.provider_id}")
    return replace(
        claimed,
        kind=policy["kind"],
        correlation_group=str(policy["correlation_group"]),
        deterministic=bool(policy["deterministic"]),
    )


class ReactionRouteVerifierProvider:
    descriptor = ProviderDescriptor(
        provider_id="autoplanner.reaction_route_verifier",
        kind=ProviderKind.VERIFIER,
        version="1.2.0",
        input_schemas=("reaction_route_verification_request.v1",),
        output_schemas=("reaction_route_validation.v1",),
        correlation_group="deterministic_reaction_verifier",
        capabilities=("atom_mapping_audit", "weakest_link_route_proof"),
        deterministic=True,
    )

    def __init__(
        self,
        *,
        trusted_precedent_bindings: Mapping[str, Mapping[str, Any]] | None = None,
        verified_procurement_bindings: Mapping[str, Mapping[str, Any]] | None = None,
        trusted_stock_providers: Mapping[str, Any] | None = None,
    ) -> None:
        # Privileged overlays are construction-time dependencies.  They cannot
        # be supplied through an invoke request controlled by a model/client.
        self._trusted_precedent_bindings = {
            str(key): dict(value)
            for key, value in (trusted_precedent_bindings or {}).items()
            if isinstance(value, Mapping)
        }
        self._verified_procurement_bindings = {
            str(key): dict(value)
            for key, value in (verified_procurement_bindings or {}).items()
            if isinstance(value, Mapping)
        }
        self._trusted_stock_providers = dict(trusted_stock_providers or {})

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        del context
        from cascade_planner.harness.reaction_step_verifier import verify_reaction_route

        proof = verify_reaction_route(
            [dict(row) for row in request.get("steps") or [] if isinstance(row, Mapping)],
            graph_and_stock_closed=bool(request.get("graph_and_stock_closed")),
            trusted_precedent_bindings=self._trusted_precedent_bindings,
            procurement_bindings=self._verified_procurement_bindings,
            trusted_stock_providers=self._trusted_stock_providers,
        )
        privileged_request_fields = sorted(
            field
            for field in ("trusted_precedent_bindings", "procurement_bindings")
            if field in request
        )
        provider_reasons = {
            str(reason)
            for step in proof.get("step_proofs") or []
            for reason in step.get("reasons") or []
        }
        provider_reasons.update(
            f"privileged_request_field_ignored:{field}"
            for field in privileged_request_fields
        )
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema="reaction_route_validation.v1",
            accepted=proof.get("accepted") is True,
            payload=proof,
            reasons=tuple(sorted(provider_reasons)),
        )


class ChemEnzyProposalProvider:
    """Typed advisory envelope around an injected ChemEnzy proposal runner."""

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.chemenzy_proposals",
        kind=ProviderKind.PROPOSAL,
        version="1.1.0",
        input_schemas=(
            "chemenzy_proposal_request.v1",
            "chemenzy_proposal_request.v2",
        ),
        output_schemas=("retrosynthesis_candidate_batch.v1",),
        correlation_group="computational:chem_enzy",
        capabilities=(
            "one_step_retrosynthesis",
            "multi_step_route_proposals",
            "guided_frontier_request_contract",
        ),
    )

    def __init__(self, runner: Any) -> None:
        if not callable(runner):
            raise TypeError("ChemEnzy proposal runner must be callable")
        self.runner = runner

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        payload = self.runner(dict(request), context=context)
        row = dict(payload) if isinstance(payload, Mapping) else {}
        solved_claim = _payload_has_solved_claim(row)
        reasons = [str(value) for value in row.get("reasons") or []]
        if solved_claim:
            reasons.append("proposal_provider_attempted_solved_claim")
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema="retrosynthesis_candidate_batch.v1",
            accepted=bool(row and not solved_claim and row.get("accepted", True)),
            payload=row,
            reasons=tuple(sorted(set(reasons))),
            source_refs=tuple(str(value) for value in row.get("source_refs") or []),
            evidence_refs=tuple(str(value) for value in row.get("evidence_refs") or []),
        )


class LiteratureEvidenceProvider:
    """Typed advisory envelope around an injected literature evidence runner."""

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.literature_evidence",
        kind=ProviderKind.EVIDENCE,
        version="1.0.0",
        input_schemas=("literature_evidence_request.v1",),
        output_schemas=("literature_evidence_batch.v1",),
        correlation_group="literature_evidence_pipeline",
        capabilities=("source_resolution", "source_detail_step_extraction"),
        network_access=True,
    )

    def __init__(self, runner: Any) -> None:
        if not callable(runner):
            raise TypeError("literature evidence runner must be callable")
        self.runner = runner

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        payload = self.runner(dict(request), context=context)
        row = dict(payload) if isinstance(payload, Mapping) else {}
        solved_claim = _payload_has_solved_claim(row)
        reasons = [str(value) for value in row.get("reasons") or []]
        if solved_claim:
            reasons.append("evidence_provider_attempted_solved_claim")
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema="literature_evidence_batch.v1",
            accepted=bool(row and not solved_claim and row.get("accepted", True)),
            payload=row,
            reasons=tuple(sorted(set(reasons))),
            source_refs=tuple(str(value) for value in row.get("source_refs") or []),
            evidence_refs=tuple(str(value) for value in row.get("evidence_refs") or []),
        )


def _payload_has_solved_claim(value: Mapping[str, Any]) -> bool:
    stack: list[Any] = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if id(current) in seen:
                continue
            seen.add(id(current))
            for key, raw in current.items():
                normalized = str(key).strip().lower()
                if normalized in {"solved", "route_solved", "parent_route_solved"} and raw is True:
                    return True
                if normalized in {"route_status", "verdict", "status"} and str(raw).lower() == "solved":
                    return True
                stack.append(raw)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return False


def build_default_provider_registry(
    *,
    include_chemenzy: ChemEnzyProposalProvider | None = None,
    include_literature: LiteratureEvidenceProvider | None = None,
    include_manual_experiment_executor: bool = False,
    include_http_experiment_executor: HttpExperimentExecutorProvider | None = None,
) -> ProviderRegistry:
    registry = ProviderRegistry()
    stock = SnapshotStockProvider()
    verifier = ReactionRouteVerifierProvider()
    registry.register(
        stock,
        trusted_descriptor=_host_trusted_builtin_descriptor(stock),
        authority="autoplanner_host_builtin_allowlist.v1",
    )
    registry.register(
        verifier,
        trusted_descriptor=_host_trusted_builtin_descriptor(verifier),
        authority="autoplanner_host_builtin_allowlist.v1",
    )
    for provider in (include_chemenzy, include_literature):
        if provider is None:
            continue
        registry.register(
            provider,
            trusted_descriptor=_host_trusted_builtin_descriptor(provider),
            authority="autoplanner_host_builtin_allowlist.v1",
        )
    if include_manual_experiment_executor:
        manual = ManualExperimentExecutorProvider()
        registry.register(
            manual,
            trusted_descriptor=_host_trusted_builtin_descriptor(manual),
            authority="autoplanner_host_builtin_allowlist.v1",
        )
    if include_http_experiment_executor is not None:
        registry.register(
            include_http_experiment_executor,
            trusted_descriptor=include_http_experiment_executor.descriptor,
            authority="autoplanner_host_environment_allowlist.v1",
        )
    return registry
