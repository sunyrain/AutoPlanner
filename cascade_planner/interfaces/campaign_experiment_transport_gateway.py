"""Explicit network transport methods for CampaignGateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.interfaces.campaign_experiment_transport import (
    execute_route_experiment_transport_result,
)


class CampaignExperimentTransportGatewayMixin:
    def submit_route_experiment_job(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        timeout_s: float = 0.0,
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_transport: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return self._experiment_transport(
            run_id, operation="submit", route_id=route_id,
            capabilities=capabilities, dispatch_id=dispatch_id, timeout_s=timeout_s,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_transport=enable_experiment_transport, run_dir=run_dir,
        )

    def poll_route_experiment_job(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        timeout_s: float = 0.0,
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_transport: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return self._experiment_transport(
            run_id, operation="poll", route_id=route_id,
            capabilities=capabilities, dispatch_id=dispatch_id, timeout_s=timeout_s,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_transport=enable_experiment_transport, run_dir=run_dir,
        )

    def transmit_route_experiment_cancellation(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        timeout_s: float = 0.0,
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_transport: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return self._experiment_transport(
            run_id, operation="cancel", route_id=route_id,
            capabilities=capabilities, dispatch_id=dispatch_id, timeout_s=timeout_s,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_transport=enable_experiment_transport, run_dir=run_dir,
        )

    def _experiment_transport(
        self, run_id: str, *, operation: str, route_id: str, capabilities: Any,
        dispatch_id: str, timeout_s: float,
        mechanism_proposals: Iterable[Mapping[str, Any]],
        validations: Iterable[Mapping[str, Any]],
        enable_experiment_transport: bool, run_dir: str | Path | None,
    ) -> dict[str, Any]:
        return execute_route_experiment_transport_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            operation=operation, route_id=route_id, capabilities=capabilities,
            dispatch_id=dispatch_id, timeout_s=timeout_s,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_transport=enable_experiment_transport,
        )


__all__ = ["CampaignExperimentTransportGatewayMixin"]
