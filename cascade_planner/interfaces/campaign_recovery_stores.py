"""Replay shadow Program stores for campaign validation and recovery."""

from __future__ import annotations

from typing import Any

from cascade_planner.application.biocatalytic_program_store import (
    BiocatalyticProgramStoreError,
)
from cascade_planner.application.experimental_claim_store import (
    ExperimentalClaimStoreError,
)
from cascade_planner.application.transformation_program_store import (
    TransformationProgramStoreError,
)
from cascade_planner.interfaces.campaign_gateway_contract import CampaignGatewayError
from cascade_planner.orchestration.biocatalytic_program_admission_runtime import (
    biocatalytic_program_store_read,
)
from cascade_planner.orchestration.experimental_claim_admission_runtime import (
    experimental_claim_store_read,
)


def recovery_program_stores(service: Any) -> dict[str, Any]:
    try:
        return {
            "baseline": service.program_store(),
            "biocatalytic": biocatalytic_program_store_read(service.kernel),
            "experimental_claims": experimental_claim_store_read(service.kernel),
        }
    except (
        BiocatalyticProgramStoreError,
        ExperimentalClaimStoreError,
        TransformationProgramStoreError,
    ) as exc:
        raise CampaignGatewayError(str(exc)) from exc


__all__ = ["recovery_program_stores"]
