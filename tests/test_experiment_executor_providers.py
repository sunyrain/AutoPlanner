from __future__ import annotations

from copy import deepcopy

import pytest

from cascade_planner.application.experiment_execution_contracts import (
    build_experiment_execution_request,
)
from cascade_planner.providers import (
    ExperimentExecutorPolicyError,
    ManualExperimentExecutorProvider,
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderRegistry,
    ProviderRegistryError,
    select_experiment_executor,
    validate_experiment_dispatch_handoff,
)
from cascade_planner.providers.builtins import build_default_provider_registry
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _request(*, resource_hints: dict | None = None) -> dict:
    plan = {
        "schema_version": "fixture_experiment_plan.v1",
        "plan_id": "plan:fixture-experiment",
        "program_id": "program:fixture-experiment",
        "exact_boundary": {
            "input_states": [
                {
                    "state_id": "state:input",
                    "molecule_id": "m:input",
                    "canonical_smiles": "CC=O",
                }
            ],
            "output_states": [
                {
                    "state_id": "state:output",
                    "molecule_id": "m:output",
                    "canonical_smiles": "CCO",
                }
            ],
        },
        "required_checks": [
            {"check_id": "conversion", "objective": "measure conversion", "required": True}
        ],
        "required_output_contract": {
            "schema_version": "biocatalysis_program_validation.v1"
        },
    }
    plan["content_sha256"] = strict_canonical_json_sha256(plan)
    return build_experiment_execution_request(
        run_id="run:fixture",
        route_id="route:fixture",
        work_item_id="work:fixture",
        domain="biocatalytic",
        plan=plan,
        canonical_frontier_sha256="a" * 64,
        resource_hints=resource_hints,
    )


def _policy() -> dict:
    return {
        "schema_version": "experiment_executor_policy.v1",
        "enabled": True,
        "allowed_provider_ids": ["autoplanner.manual_experiment_executor"],
        "preferred_provider_ids": ["autoplanner.manual_experiment_executor"],
        "allowed_domains": ["biocatalytic"],
        "allow_network_access": False,
        "max_estimated_cost_units": 0,
    }


def test_manual_executor_selection_and_handoff_are_bound_but_non_authoritative() -> None:
    request = _request()
    registry = build_default_provider_registry(include_manual_experiment_executor=True)
    selection = select_experiment_executor(registry, request, _policy())
    result = registry.invoke(
        selection["selected"]["provider_id"],
        request,
        context=ProviderContext(
            run_id=request["run_id"],
            case_id=request["run_id"],
            target_smiles="CCO",
            config={"dispatch_id": "experiment-dispatch:" + "b" * 32},
        ),
    )

    validate_experiment_dispatch_handoff(result.payload, request=request)
    assert selection["semantics"]["client_policy_cannot_elevate_provider_trust"] is True
    assert result.payload["state"] == "awaiting_external_result"
    assert result.payload["semantics"][
        "handoff_grants_no_validation_claim_or_route_authority"
    ] is True


def test_executor_neutral_cost_hint_is_normalized_and_fail_closed() -> None:
    request = _request(resource_hints={"estimated_cost_units": 2.5})

    assert request["resource_hints"]["estimated_cost_units"] == 2.5
    with pytest.raises(ValueError, match="resource_hints_invalid"):
        _request(resource_hints={"estimated_cost_units": -1.0})
    with pytest.raises(ValueError, match="resource_hints_invalid"):
        _request(resource_hints={"estimated_cost_units": True})


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", 1),
        ("allowed_provider_ids", "autoplanner.manual_experiment_executor"),
        ("preferred_provider_ids", "autoplanner.manual_experiment_executor"),
        ("allowed_domains", "biocatalytic"),
        ("allow_network_access", 0),
        ("max_estimated_cost_units", True),
    ],
)
def test_executor_policy_rejects_json_type_confusion(field: str, value: object) -> None:
    policy = _policy()
    policy[field] = value
    registry = build_default_provider_registry(include_manual_experiment_executor=True)
    with pytest.raises(ExperimentExecutorPolicyError, match="policy_values_invalid"):
        select_experiment_executor(registry, _request(), policy)


def test_tampered_handoff_semantics_are_rejected_even_with_fresh_digest() -> None:
    request = _request()
    provider = ManualExperimentExecutorProvider()
    result = provider.invoke(
        request,
        context=ProviderContext(
            run_id=request["run_id"], case_id=request["run_id"],
            target_smiles="CCO",
            config={"dispatch_id": "experiment-dispatch:" + "c" * 32},
        ),
    )
    tampered = deepcopy(result.payload)
    tampered["semantics"]["handoff_is_not_experiment_completion"] = False
    tampered.pop("content_sha256")
    tampered["content_sha256"] = strict_canonical_json_sha256(tampered)

    with pytest.raises(ExperimentExecutorPolicyError, match="handoff_invalid"):
        validate_experiment_dispatch_handoff(tampered, request=request)


def test_untrusted_provider_cannot_self_declare_experiment_executor_kind() -> None:
    class Provider:
        descriptor = ProviderDescriptor(
            provider_id="third-party-experiment",
            kind=ProviderKind.EXPERIMENT_EXECUTOR,
            version="1",
            input_schemas=("experiment_execution_request.v1",),
            output_schemas=("experiment_dispatch_handoff.v1",),
            correlation_group="self-claimed-independent-lab",
        )

        def invoke(self, request, *, context):  # pragma: no cover - registration fails
            del request, context

    with pytest.raises(ProviderRegistryError, match="untrusted provider cannot claim"):
        ProviderRegistry().register(Provider())
