"""Gateway responses for durable biocatalytic Program shadow admissions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_program_store import (
    BiocatalyticProgramStoreError,
)
from cascade_planner.application.biocatalytic_programs import BiocatalyticProgramError
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.biocatalytic_program_admission_runtime import (
    admit_route_biocatalytic_programs,
    biocatalytic_program_store_read,
)


def admit_route_program_innovations_result(
    service: Any,
    *,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_biocatalytic_program_admission: bool = False,
    experience_library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        value = admit_route_biocatalytic_programs(
            service.kernel,
            service.graph_store,
            acceptance_spec=service.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_biocatalytic_program_admission=(
                enable_biocatalytic_program_admission
            ),
            experience_library=experience_library,
        )
    except (BiocatalyticProgramError, BiocatalyticProgramStoreError, ValueError) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "admit-route-program-innovations", value)


def biocatalytic_program_store_result(service: Any) -> dict[str, Any]:
    try:
        value = biocatalytic_program_store_read(service.kernel)
    except BiocatalyticProgramStoreError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "biocatalytic-program-store", value)


def _result(service: Any, operation: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        **value,
    }


__all__ = [
    "admit_route_program_innovations_result",
    "biocatalytic_program_store_result",
]
