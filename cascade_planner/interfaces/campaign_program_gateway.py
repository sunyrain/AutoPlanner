"""Program projection and migration methods for CampaignGateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cascade_planner.interfaces.campaign_program_innovation_gateway import (
    CampaignProgramInnovationGatewayMixin,
)
from cascade_planner.interfaces.campaign_programs import (
    admit_programs_result,
    program_projection_result,
    program_store_result,
    route_program_dual_read_result,
)


class CampaignProgramGatewayMixin(CampaignProgramInnovationGatewayMixin):
    """Keep Program migration concerns out of the primary gateway facade."""

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

    def audit_programs(
        self, *, run_ids: tuple[str, ...] = (), limit: int = 100
    ) -> dict[str, Any]:
        from cascade_planner.interfaces.program_migration import (
            audit_program_migration,
        )

        return audit_program_migration(self, run_ids=run_ids, limit=limit)


__all__ = ["CampaignProgramGatewayMixin"]
