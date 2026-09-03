"""Pure result and digest projections for the shared campaign gateway."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def campaign_gateway_result(
    service: RetrosynthesisCampaignService,
    *,
    operation: str,
    operations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        "run_dir": str(service.kernel.run_dir),
        "campaign_spec": service.kernel.spec.campaign_spec.to_dict(),
        "status": service.status(),
        "operations": dict(operations or {}),
    }


def campaign_payload_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["campaign_gateway_result", "campaign_payload_digest"]
