"""Gateway responses for durable mechanism Program shadow admissions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.mechanism_program_store import MechanismProgramStoreError
from cascade_planner.application.mechanism_programs import MechanismProgramError
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.mechanism_program_admission_runtime import (
    admit_route_mechanism_programs,
    mechanism_program_store_read,
)


def admit_route_mechanism_programs_result(
    service: Any,
    *,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    experience_library: Mapping[str, Any] | None = None,
    enable_mechanism_program_admission: bool = False,
) -> dict[str, Any]:
    try:
        value = admit_route_mechanism_programs(
            service.kernel,
            service.graph_store,
            acceptance_spec=service.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            experience_library=experience_library,
            enable_mechanism_program_admission=enable_mechanism_program_admission,
        )
    except (MechanismProgramError, MechanismProgramStoreError, ValueError) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "admit-route-mechanism-programs", value)


def mechanism_program_store_result(service: Any) -> dict[str, Any]:
    try:
        value = mechanism_program_store_read(service.kernel)
    except MechanismProgramStoreError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "mechanism-program-store", value)


def _result(service: Any, operation: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        **value,
    }


__all__ = ["admit_route_mechanism_programs_result", "mechanism_program_store_result"]
