"""CampaignGateway responses for non-authoritative Program migration APIs."""

from __future__ import annotations

from typing import Any

from cascade_planner.application.route_program_dual_read import (
    RouteProgramDualReadError,
    project_workbench_routes_to_programs,
    route_program_dual_read_oracle,
)
from cascade_planner.application.transformation_program_store import (
    TransformationProgramStoreError,
)
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)


def program_projection_result(service: Any) -> dict[str, Any]:
    return _result(service, "program-projection", service.program_projection())


def program_store_result(service: Any) -> dict[str, Any]:
    try:
        value = service.program_store()
    except TransformationProgramStoreError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "program-store", value)


def route_program_dual_read_result(service: Any) -> dict[str, Any]:
    try:
        workbench = service.workbench()["snapshot"]
        program_projection = service.program_projection()["projection"]
        overlay = project_workbench_routes_to_programs(workbench, program_projection)
        oracle = route_program_dual_read_oracle(workbench, program_projection, overlay)
    except RouteProgramDualReadError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(
        service,
        "route-program-dual-read",
        {"overlay": overlay, "oracle": oracle},
    )


def admit_programs_result(
    service: Any, *, enable_program_admission: bool = False
) -> dict[str, Any]:
    try:
        value = service.admit_programs(enable_program_admission=enable_program_admission)
    except TransformationProgramStoreError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "admit-programs", value)


def _result(service: Any, operation: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        **value,
    }


__all__ = [
    "admit_programs_result",
    "program_projection_result",
    "program_store_result",
    "route_program_dual_read_result",
]
