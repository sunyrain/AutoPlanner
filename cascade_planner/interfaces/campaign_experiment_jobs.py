"""Gateway results for external experiment job receipts and cancellation."""

from __future__ import annotations

from typing import Any

from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.experiment_dispatch_support import (
    ExperimentDispatchError,
)
from cascade_planner.orchestration.experiment_external_job_runtime import (
    record_current_route_experiment_job_receipt,
    request_current_route_experiment_cancellation,
)
from cascade_planner.providers.registry import ProviderRegistry, ProviderRegistryError


def record_route_experiment_job_receipt_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    return _result(
        service, "route-experiment-job-receipt",
        record_current_route_experiment_job_receipt, registry=registry, **kwargs,
    )


def request_route_experiment_cancellation_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    return _result(
        service, "route-experiment-cancellation-request",
        request_current_route_experiment_cancellation, registry=registry, **kwargs,
    )


def _result(
    service: Any, operation: str, handler: Any,
    *, registry: ProviderRegistry, **kwargs: Any,
) -> dict[str, Any]:
    try:
        value = handler(
            service.kernel, service.graph_store.load(),
            acceptance_spec=service.kernel.spec.acceptance,
            registry=registry, **kwargs,
        )
    except (
        ExperimentDispatchError, ProviderRegistryError, RunKernelError, ValueError
    ) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        "dispatch": value,
    }


__all__ = [
    "record_route_experiment_job_receipt_result",
    "request_route_experiment_cancellation_result",
]
