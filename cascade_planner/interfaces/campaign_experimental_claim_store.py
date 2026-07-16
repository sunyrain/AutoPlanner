"""Gateway responses for durable exact-boundary experimental Claims."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.experimental_claim_store import (
    ExperimentalClaimStoreError,
)
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.experimental_claim_admission_runtime import (
    admit_route_experimental_claims,
    experimental_claim_store_read,
)


def admit_route_experimental_claims_result(
    service: Any,
    *,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experimental_claim_admission: bool = False,
) -> dict[str, Any]:
    try:
        value = admit_route_experimental_claims(
            service.kernel,
            service.graph_store,
            acceptance_spec=service.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experimental_claim_admission=enable_experimental_claim_admission,
        )
    except (ExperimentalClaimStoreError, ValueError) as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "admit-route-experimental-claims", value)


def experimental_claim_store_result(service: Any) -> dict[str, Any]:
    try:
        value = experimental_claim_store_read(service.kernel)
    except ExperimentalClaimStoreError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "experimental-claim-store", value)


def _result(service: Any, operation: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        **value,
    }


__all__ = [
    "admit_route_experimental_claims_result",
    "experimental_claim_store_result",
]
