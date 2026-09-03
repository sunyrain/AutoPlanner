"""Program innovation, experiment Claim, and experience-memory gateway methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.application.program_experience_store import (
    DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME,
    read_program_experience_library,
)
from cascade_planner.interfaces.campaign_experiment_gateway import (
    CampaignExperimentGatewayMixin,
)
from cascade_planner.interfaces.campaign_experimental_claim_store import (
    admit_route_experimental_claims_result,
    experimental_claim_store_result,
)
from cascade_planner.interfaces.campaign_gateway_contract import CampaignGatewayError
from cascade_planner.interfaces.campaign_mechanism_program_store import (
    admit_route_mechanism_programs_result,
    mechanism_program_store_result,
)
from cascade_planner.interfaces.campaign_program_experience import (
    learn_program_experience_result,
    program_experience_result,
)
from cascade_planner.interfaces.campaign_program_innovation_store import (
    admit_route_program_innovations_result,
    biocatalytic_program_store_result,
)
from cascade_planner.interfaces.campaign_program_innovations import (
    route_program_innovations_result,
)


class CampaignProgramInnovationGatewayMixin(CampaignExperimentGatewayMixin):
    """Expose one Program innovation boundary across all execution domains."""

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
            experience_library=self._program_experience_library(),
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
            enable_biocatalytic_program_admission=enable_biocatalytic_program_admission,
            experience_library=self._program_experience_library(),
        )

    def biocatalytic_program_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return biocatalytic_program_store_result(self._open(run_id, run_dir=run_dir))

    def admit_route_mechanism_programs(
        self,
        run_id: str,
        *,
        route_id: str,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_mechanism_program_admission: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return admit_route_mechanism_programs_result(
            self._open(run_id, run_dir=run_dir),
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            experience_library=self._program_experience_library(),
            enable_mechanism_program_admission=enable_mechanism_program_admission,
        )

    def mechanism_program_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return mechanism_program_store_result(self._open(run_id, run_dir=run_dir))

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
            experience_library=self._program_experience_library(),
        )

    def experimental_claim_store(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return experimental_claim_store_result(self._open(run_id, run_dir=run_dir))

    def program_experience(
        self, run_id: str, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return program_experience_result(
            self._open(run_id, run_dir=run_dir),
            library_path=self._program_experience_path(),
        )

    def learn_program_experience(
        self,
        run_id: str,
        *,
        enable_program_experience_learning: bool = False,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return learn_program_experience_result(
            self._open(run_id, run_dir=run_dir),
            library_path=self._program_experience_path(),
            enable_program_experience_learning=enable_program_experience_learning,
        )

    def _program_experience_path(self) -> Path:
        return self.paths.external_data_root / "self-evo" / DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME

    def _program_experience_library(self) -> dict[str, Any]:
        library, error = read_program_experience_library(self._program_experience_path())
        if error:
            raise CampaignGatewayError(error)
        return library


__all__ = ["CampaignProgramInnovationGatewayMixin"]
