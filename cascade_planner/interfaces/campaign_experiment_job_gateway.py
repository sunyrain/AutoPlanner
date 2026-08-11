"""External experiment job lifecycle methods for CampaignGateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.interfaces.campaign_experiment_jobs import (
    record_route_experiment_job_receipt_result,
    request_route_experiment_cancellation_result,
)


class CampaignExperimentJobGatewayMixin:
    def record_route_experiment_job_receipt(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        job_receipt: Mapping[str, Any],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_job_receipt: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return record_route_experiment_job_receipt_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            route_id=route_id, capabilities=capabilities, dispatch_id=dispatch_id,
            job_receipt=job_receipt, mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experiment_job_receipt=enable_experiment_job_receipt,
        )

    def request_route_experiment_cancellation(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        cancellation_request: Mapping[str, Any],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_cancellation: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return request_route_experiment_cancellation_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            route_id=route_id, capabilities=capabilities, dispatch_id=dispatch_id,
            cancellation_request=cancellation_request,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_cancellation=enable_experiment_cancellation,
        )


__all__ = ["CampaignExperimentJobGatewayMixin"]
