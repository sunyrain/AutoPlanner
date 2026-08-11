"""Gateway result adapter for explicit experiment provider transport calls."""

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
from cascade_planner.orchestration.experiment_job_transport_runtime import (
    execute_current_route_experiment_transport,
)
from cascade_planner.providers.registry import ProviderRegistry, ProviderRegistryError


def execute_route_experiment_transport_result(
    service: Any, *, registry: ProviderRegistry, **kwargs: Any
) -> dict[str, Any]:
    operation = str(kwargs.get("operation") or "")
    try:
        value = execute_current_route_experiment_transport(
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
        "operation": f"route-experiment-transport-{operation}",
        "run_id": service.kernel.spec.run_id,
        "dispatch": value,
    }


__all__ = ["execute_route_experiment_transport_result"]
