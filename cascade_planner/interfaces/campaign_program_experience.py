"""Gateway responses for replay-gated Program self-evolution memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.orchestration.program_experience_runtime import (
    program_experience_library_read,
    synchronize_program_experience,
)


def learn_program_experience_result(
    service: Any,
    *,
    library_path: str | Path,
    enable_program_experience_learning: bool = False,
) -> dict[str, Any]:
    try:
        value = synchronize_program_experience(
            service.kernel,
            library_path=library_path,
            enable_program_experience_learning=enable_program_experience_learning,
        )
    except ValueError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "learn-program-experience", value)


def program_experience_result(service: Any, *, library_path: str | Path) -> dict[str, Any]:
    try:
        value = program_experience_library_read(library_path)
    except ValueError as exc:
        raise CampaignGatewayError(str(exc)) from exc
    return _result(service, "program-experience", value)


def _result(service: Any, operation: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": operation,
        "run_id": service.kernel.spec.run_id,
        **value,
    }


__all__ = ["learn_program_experience_result", "program_experience_result"]
