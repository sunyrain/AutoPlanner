from __future__ import annotations

from dataclasses import dataclass

import pytest

from cascade_planner.providers import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderRegistry,
    ProviderRegistryError,
    ProviderResultEnvelope,
    validate_provider_result,
)


@dataclass
class _Provider:
    descriptor: ProviderDescriptor

    def invoke(self, request, *, context):
        del context
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema=self.descriptor.output_schemas[0],
            accepted=True,
            payload={"request": dict(request)},
        )


def _descriptor(provider_id: str, *, group: str = "codex_model") -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id=provider_id,
        kind=ProviderKind.PROPOSAL,
        version="1.0.0",
        input_schemas=("proposal_request.v1",),
        output_schemas=("retrosynthesis_candidate_batch.v1",),
        correlation_group=group,
        capabilities=("retrosynthesis",),
    )


def test_registry_invokes_typed_provider_and_validates_hash() -> None:
    registry = ProviderRegistry()
    registry.register(_Provider(_descriptor("codex-strategy")))
    context = ProviderContext(run_id="run", case_id="case", target_smiles="CCO")

    result = registry.invoke(
        "codex-strategy",
        {"schema_version": "proposal_request.v1", "target_smiles": "CCO"},
        context=context,
    )

    assert result.accepted is True
    assert result.correlation_group == "untrusted_provider"
    assert validate_provider_result(
        result.to_dict(),
        descriptor=registry.descriptor("codex-strategy"),
    ) == []


def test_codex_roles_remain_one_correlation_group() -> None:
    registry = ProviderRegistry()
    strategy = _Provider(_descriptor("codex-strategy"))
    literature = _Provider(_descriptor("codex-literature"))
    registry.register(
        strategy,
        trusted_descriptor=strategy.descriptor,
        authority="test_host_policy",
    )
    registry.register(
        literature,
        trusted_descriptor=literature.descriptor,
        authority="test_host_policy",
    )

    assert registry.correlation_groups(["codex-strategy", "codex-literature"]) == {
        "codex_model": ["codex-literature", "codex-strategy"]
    }


def test_duplicate_id_and_undeclared_schema_fail_closed() -> None:
    registry = ProviderRegistry()
    provider = _Provider(_descriptor("provider"))
    registry.register(provider)

    with pytest.raises(ProviderRegistryError, match="duplicate provider_id"):
        registry.register(provider)
    with pytest.raises(ProviderRegistryError, match="does not accept schema"):
        registry.invoke(
            "provider",
            {"schema_version": "wrong.v1"},
            context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
        )


def test_tampered_result_hash_is_rejected() -> None:
    descriptor = _descriptor("provider")
    result = _Provider(descriptor).invoke(
        {"schema_version": "proposal_request.v1"},
        context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
    ).to_dict()
    result["payload"] = {"tampered": True}

    assert "provider_result_content_hash_mismatch" in validate_provider_result(
        result,
        descriptor=descriptor,
    )


def test_third_party_cannot_self_declare_independence_or_deterministic_authority() -> None:
    claimed = ProviderDescriptor(
        provider_id="third-party",
        kind=ProviderKind.PROPOSAL,
        version="1.0.0",
        input_schemas=("proposal_request.v1",),
        output_schemas=("retrosynthesis_candidate_batch.v1",),
        correlation_group="independent_literature",
        capabilities=("trusted_exact_evidence",),
        deterministic=True,
        network_access=False,
    )
    registry = ProviderRegistry()
    registry.register(_Provider(claimed))

    effective = registry.descriptor("third-party")
    trust = registry.trust_record("third-party")
    assert effective.correlation_group == "untrusted_provider"
    assert effective.deterministic is False
    assert effective.capabilities == ()
    assert effective.network_access is True
    assert trust["trusted"] is False
    assert trust["claimed_correlation_group"] == "independent_literature"


def test_privileged_provider_kind_requires_host_trust_record() -> None:
    privileged = ProviderDescriptor(
        provider_id="third-party-verifier",
        kind=ProviderKind.VERIFIER,
        version="1.0.0",
        input_schemas=("verify_request.v1",),
        output_schemas=("verify_result.v1",),
        correlation_group="independent_verifier",
        deterministic=True,
    )

    with pytest.raises(ProviderRegistryError, match="untrusted provider cannot claim privileged kind"):
        ProviderRegistry().register(_Provider(privileged))
