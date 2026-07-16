"""Program projection, review, and shadow-store methods for CampaignGateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.interfaces.campaign_experimental_claim_store import (
    admit_route_experimental_claims_result,
    experimental_claim_store_result,
)
from cascade_planner.interfaces.campaign_program_innovation_store import (
    admit_route_program_innovations_result,
    biocatalytic_program_store_result,
)
from cascade_planner.interfaces.campaign_program_innovations import (
    route_program_innovations_result,
)
from cascade_planner.interfaces.campaign_experiment_gateway import (
    CampaignExperimentGatewayMixin,
)
from cascade_planner.interfaces.campaign_programs import (
    admit_programs_result,
    program_projection_result,
    program_store_result,
    route_program_dual_read_result,
)


class CampaignProgramGatewayMixin(CampaignExperimentGatewayMixin):
    """Keep Program migration concerns out of the primary gateway façade."""

    def program_projection(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return program_projection_result(self._open(run_id, run_dir=run_dir))

    def program_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return program_store_result(self._open(run_id, run_dir=run_dir))

    def route_program_dual_read(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return route_program_dual_read_result(self._open(run_id, run_dir=run_dir))

    def route_program_innovations(
        self,
        run_id: str,
        *,
        route_id: str,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return route_program_innovations_result(
            self._open(run_id, run_dir=run_dir),
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            reported_candidate_packs=reported_candidate_packs,
        )

    def admit_programs(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        enable_program_admission: bool = False,
    ) -> dict[str, Any]:
        return admit_programs_result(
            self._open(run_id, run_dir=run_dir),
            enable_program_admission=enable_program_admission,
        )

    def admit_route_program_innovations(
        self,
        run_id: str,
        *,
        route_id: str,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_biocatalytic_program_admission: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return admit_route_program_innovations_result(
            self._open(run_id, run_dir=run_dir),
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_biocatalytic_program_admission=(
                enable_biocatalytic_program_admission
            ),
        )

    def biocatalytic_program_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return biocatalytic_program_store_result(self._open(run_id, run_dir=run_dir))

    def admit_route_experimental_claims(
        self,
        run_id: str,
        *,
        route_id: str,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experimental_claim_admission: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return admit_route_experimental_claims_result(
            self._open(run_id, run_dir=run_dir),
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experimental_claim_admission=enable_experimental_claim_admission,
        )

    def experimental_claim_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return experimental_claim_store_result(self._open(run_id, run_dir=run_dir))

    def audit_programs(
        self, *, run_ids: tuple[str, ...] = (), limit: int = 100
    ) -> dict[str, Any]:
        from cascade_planner.interfaces.program_migration import (
            audit_program_migration,
        )

        return audit_program_migration(self, run_ids=run_ids, limit=limit)


__all__ = ["CampaignProgramGatewayMixin"]
