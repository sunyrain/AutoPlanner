"""Experiment review and operational handoff methods for CampaignGateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.interfaces.campaign_experiment_dispatch import (
    dispatch_route_experiment_result,
    recover_route_experiment_dispatch_result,
    settle_route_experiment_dispatch_result,
    stage_experiment_json_artifact_result,
)
from cascade_planner.interfaces.campaign_experiment_job_gateway import (
    CampaignExperimentJobGatewayMixin,
)
from cascade_planner.interfaces.campaign_program_innovations import (
    route_experiment_result_audit_result,
)


class CampaignExperimentGatewayMixin(CampaignExperimentJobGatewayMixin):
    def audit_route_experiment_result(
        self, run_id: str, *, route_id: str, capabilities: Any,
        result: Mapping[str, Any], mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return route_experiment_result_audit_result(
            self._open(run_id, run_dir=run_dir), route_id=route_id,
            capabilities=capabilities, result=result,
            mechanism_proposals=mechanism_proposals, validations=validations,
        )

    def dispatch_route_experiment(
        self, run_id: str, *, route_id: str, capabilities: Any,
        request_id: str, policy: Mapping[str, Any],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_dispatch: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return dispatch_route_experiment_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            route_id=route_id, capabilities=capabilities, request_id=request_id,
            policy=policy, mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experiment_dispatch=enable_experiment_dispatch,
        )

    def recover_route_experiment_dispatch(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_dispatch_recovery: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return recover_route_experiment_dispatch_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            route_id=route_id, capabilities=capabilities, dispatch_id=dispatch_id,
            mechanism_proposals=mechanism_proposals, validations=validations,
            enable_experiment_dispatch_recovery=enable_experiment_dispatch_recovery,
        )

    def settle_route_experiment_dispatch(
        self, run_id: str, *, route_id: str, capabilities: Any, dispatch_id: str,
        result: Mapping[str, Any],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experiment_settlement: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return settle_route_experiment_dispatch_result(
            self._open(run_id, run_dir=run_dir), registry=self.providers,
            route_id=route_id, capabilities=capabilities, dispatch_id=dispatch_id,
            result=result, mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experiment_settlement=enable_experiment_settlement,
        )

    def stage_experiment_json_artifact(
        self, run_id: str, *, artifact: Mapping[str, Any], logical_name: str,
        enable_experiment_artifact_staging: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return stage_experiment_json_artifact_result(
            self._open(run_id, run_dir=run_dir), artifact=artifact,
            logical_name=logical_name,
            enable_experiment_artifact_staging=enable_experiment_artifact_staging,
        )


__all__ = ["CampaignExperimentGatewayMixin"]
