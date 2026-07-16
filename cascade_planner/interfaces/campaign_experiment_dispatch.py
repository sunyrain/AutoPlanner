"""Gateway adapter for bounded experiment dispatch, recovery, and settlement."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.experiment_dispatch_runtime import (
    ExperimentDispatchError,
    dispatch_current_route_experiment,
    recover_current_route_experiment_dispatch,
    settle_current_route_experiment_dispatch,
)
from cascade_planner.providers.experiment import ExperimentExecutorPolicyError
from cascade_planner.providers.registry import ProviderRegistry, ProviderRegistryError
from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def dispatch_route_experiment_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    return _result(
        service,
        "route-experiment-dispatch",
        dispatch_current_route_experiment,
        registry=registry,
        **kwargs,
    )


def recover_route_experiment_dispatch_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    result = _result(
        service,
        "route-experiment-dispatch-recovery",
        recover_current_route_experiment_dispatch,
        registry=registry,
        **kwargs,
    )
    result["recovered"] = True
    return result


def settle_route_experiment_dispatch_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    return _result(
        service,
        "route-experiment-dispatch-settlement",
        settle_current_route_experiment_dispatch,
        registry=registry,
        **kwargs,
    )


def stage_experiment_json_artifact_result(
    service: Any,
    *,
    artifact: Mapping[str, Any],
    logical_name: str,
    enable_experiment_artifact_staging: bool = False,
) -> dict[str, Any]:
    if enable_experiment_artifact_staging is not True:
        raise CampaignGatewayError("experiment_artifact_staging_explicit_enable_required")
    value = dict(artifact) if isinstance(artifact, Mapping) else {}
    name = str(logical_name or "").strip()
    if not value or not name:
        raise CampaignGatewayError("experiment_artifact_payload_and_name_required")
    ref = service.kernel.artifacts.put_json(
        value,
        logical_name=name,
        producer="autoplanner.experiment_artifact_staging",
    )
    artifact_id = "experiment-raw:" + strict_canonical_json_sha256(
        {"logical_name": name, "sha256": ref.sha256}
    )[:32]
    service.kernel.index.index_artifact(
        run_id=service.kernel.spec.run_id,
        artifact_id=artifact_id,
        ref=ref,
        revision=service.kernel.state.graph_revision,
        authority_scope="untrusted_experiment_raw_artifact_only",
    )
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "stage-experiment-json-artifact",
        "run_id": service.kernel.spec.run_id,
        "artifact": ref.to_dict(),
        "semantics": {
            "content_addressed_bytes_only": True,
            "artifact_grants_no_validation_claim_or_route_authority": True,
        },
    }


def _result(
    service: Any,
    operation: str,
    handler: Any,
    *,
    registry: ProviderRegistry,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        value = handler(
            service.kernel,
            service.graph_store.load(),
            acceptance_spec=service.kernel.spec.acceptance,
            registry=registry,
            **kwargs,
        )
    except (
        ExperimentDispatchError,
        ExperimentExecutorPolicyError,
        ProviderRegistryError,
        RunKernelError,
        ValueError,
    ) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        "dispatch": value,
    }


__all__ = [
    "dispatch_route_experiment_result",
    "recover_route_experiment_dispatch_result",
    "settle_route_experiment_dispatch_result",
    "stage_experiment_json_artifact_result",
]
