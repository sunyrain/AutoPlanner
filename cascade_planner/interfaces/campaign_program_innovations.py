"""Gateway adapter for read-only biocatalytic Program proposal reviews."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_programs import BiocatalyticProgramError
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.program_innovation_runtime import (
    review_route_program_innovations,
)
from cascade_planner.orchestration.experiment_execution_runtime import (
    audit_current_route_experiment_result,
)


def route_program_innovations_result(
    service: Any,
    *,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
    experience_library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        value = review_route_program_innovations(
            service.graph_store.load(),
            acceptance_spec=service.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            reported_candidate_packs=reported_candidate_packs,
            experience_library=experience_library,
        )
    except (BiocatalyticProgramError, ValueError) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "route-program-innovations",
        "run_id": service.kernel.spec.run_id,
        **value,
    }


def route_experiment_result_audit_result(
    service: Any,
    *,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    result: Mapping[str, Any],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        value = audit_current_route_experiment_result(
            service.graph_store.load(),
            acceptance_spec=service.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            result=result,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
        )
    except ValueError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "route-experiment-result-audit",
        **value,
    }


__all__ = [
    "route_experiment_result_audit_result",
    "route_program_innovations_result",
]
