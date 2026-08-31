"""Shared CLI/Web gateway delegating all scientific state to the V4 service."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.application.unified_campaign_spec import (
    CampaignResourceBudget,
    StockOracleReference,
    TargetConstraints,
    UnifiedCampaignSpec,
)
from cascade_planner.interfaces.campaign_operations import (
    benchmark_campaign,
    export_campaign,
    plan_artifact_gc,
)
from cascade_planner.interfaces.campaign_gateway_identity import (
    new_run_id,
    run_segment,
    utc_now,
)
from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
    CampaignGatewayError,
)
from cascade_planner.interfaces.campaign_gateway_projection import (
    campaign_gateway_result,
    campaign_payload_digest,
)
from cascade_planner.interfaces.campaign_gateway_stock_oracle import (
    default_stock_oracle_reference,
)
from cascade_planner.interfaces.campaign_program_gateway import (
    CampaignProgramGatewayMixin,
)
from cascade_planner.interfaces.campaign_milestone_gateway import (
    CampaignMilestoneGatewayMixin,
)
from cascade_planner.interfaces.campaign_recovery import (
    replay_campaign,
    validate_campaign,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex
from cascade_planner.providers.builtins import build_default_provider_registry
from cascade_planner.providers.http_experiment import (
    configured_http_experiment_executor,
)
from cascade_planner.providers.registry import ProviderRegistry


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class CampaignGateway(CampaignMilestoneGatewayMixin, CampaignProgramGatewayMixin):
    """Expose one bounded interface shared by CLI and HTTP adapters."""

    def __init__(
        self,
        paths: RuntimePaths | None = None,
        *,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self.paths = paths or RuntimePaths.discover()
        self.paths.ensure_runtime_directories()
        self.index = RunIndex(self.paths.run_index_path)
        self.providers = provider_registry or build_default_provider_registry(
            include_manual_experiment_executor=True,
            include_http_experiment_executor=configured_http_experiment_executor(),
        )

    def create_run(
        self,
        *,
        target_name: str,
        target_smiles: str,
        run_id: str | None = None,
        run_dir: str | Path | None = None,
        acceptance: RetrosynthesisAcceptanceSpec | None = None,
        budget: RetrosynthesisRunBudget | None = None,
        campaign_spec: UnifiedCampaignSpec | None = None,
        constraints: TargetConstraints | None = None,
        stock_oracle_reference: StockOracleReference | None = None,
        global_plan: Mapping[str, Any] | None = None,
        materialize: bool = False,
        closeout: bool = False,
    ) -> dict[str, Any]:
        target_name = str(target_name or "").strip()
        target_smiles = str(target_smiles or "").strip()
        if not target_name or not target_smiles:
            raise CampaignGatewayError("target_name_and_smiles_required")
        resolved_acceptance = acceptance or RetrosynthesisAcceptanceSpec()
        resolved_limits = (
            RunLimits.from_dict(campaign_spec.resource_budget.to_dict())
            if campaign_spec is not None
            else RunLimits(
                model=budget or RetrosynthesisRunBudget(max_model_invocations=0)
            )
        )
        if budget is not None and resolved_limits.model != budget:
            raise CampaignGatewayError("campaign_spec_model_budget_conflict")
        resolved_campaign_spec = campaign_spec or UnifiedCampaignSpec(
            target_smiles=target_smiles,
            stock_oracle=(
                stock_oracle_reference
                or default_stock_oracle_reference(
                    self.providers,
                    boundary=resolved_acceptance.stock_boundary
                )
            ),
            constraints=constraints or TargetConstraints(),
            resource_budget=CampaignResourceBudget.from_dict(
                resolved_limits.to_dict()
            ),
        )
        if resolved_campaign_spec.target_smiles != target_smiles:
            raise CampaignGatewayError("campaign_spec_target_conflict")
        if resolved_campaign_spec.stock_oracle.boundary != (
            resolved_acceptance.stock_boundary
        ):
            raise CampaignGatewayError("campaign_spec_stock_boundary_conflict")
        legacy_budget = CampaignResourceBudget.from_dict(resolved_limits.to_dict())
        resource_budget = resolved_campaign_spec.resource_budget
        if any(
            getattr(resource_budget, name) != getattr(legacy_budget, name)
            for name in (
                "model",
                "max_total_tasks",
                "max_evidence_tasks",
                "max_stock_tasks",
                "max_validation_tasks",
                "max_run_wall_time_s",
            )
        ):
            raise CampaignGatewayError("campaign_spec_budget_conflict")
        identity = self._normalize_run_id(
            run_id or new_run_id(target_name, target_smiles)
        )
        directory = self._run_dir(identity, explicit=run_dir, require=False)
        spec_path = directory / ".autoplanner" / "kernel" / "run_spec.json"
        if spec_path.is_file():
            service = self._open(identity, run_dir=directory)
            if (
                service.kernel.spec.target_name != target_name
                or service.kernel.spec.target_smiles != target_smiles
            ):
                raise CampaignGatewayError("existing_run_target_conflict")
        else:
            service = RetrosynthesisCampaignService.create(
                self.paths.runtime_root,
                directory,
                spec=RunSpec(
                    run_id=identity,
                    target_name=target_name,
                    target_smiles=target_smiles,
                    acceptance=resolved_acceptance,
                    limits=resolved_limits,
                    campaign_spec=resolved_campaign_spec,
                    created_at=utc_now(),
                ),
                artifact_store_root=self.paths.artifact_store_root,
                run_index_path=self.paths.run_index_path,
            )
        operations: dict[str, Any] = {}
        if global_plan is not None:
            operations["global_plan"] = service.apply_global_plan(
                global_plan,
                idempotency_key=(
                    f"gateway:plan:{campaign_payload_digest(global_plan)[:24]}"
                ),
            )
        if materialize:
            revision = service.kernel.state.graph_revision
            operations["materialization"] = service.execute_frontier_materialization(
                idempotency_key=f"gateway:frontier-materialization:{revision}"
            )
        if closeout:
            revision = service.kernel.state.graph_revision
            operations["closeout"] = service.closeout(
                idempotency_key=f"gateway:closeout:{revision}"
            )
        return campaign_gateway_result(service, operation="run", operations=operations)

    def solve_target(self, **kwargs: Any) -> dict[str, Any]:
        from cascade_planner.interfaces.target_solver import solve_target

        return solve_target(self, **kwargs)

    def fork_target_validation(self, **kwargs: Any) -> dict[str, Any]:
        from cascade_planner.interfaces.validation_fork import fork_target_validation

        return fork_target_validation(self, **kwargs)

    def import_evidence(self, **kwargs: Any) -> dict[str, Any]:
        from cascade_planner.interfaces.evidence_import import import_structured_evidence

        return import_structured_evidence(self, **kwargs)

    def resume(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        materialize: bool = False,
        closeout: bool = False,
    ) -> dict[str, Any]:
        service = self._open(run_id, run_dir=run_dir)
        operations: dict[str, Any] = {}
        if service.kernel.state.status == "paused":
            operations["transition"] = service.kernel.resume(
                idempotency_key=f"gateway:resume:{service.kernel.state.revision}"
            ).to_dict()
        if materialize:
            operations["materialization"] = service.execute_frontier_materialization(
                idempotency_key=(
                    f"gateway:resume-materialization:{service.kernel.state.graph_revision}"
                )
            )
        if closeout:
            operations["closeout"] = service.closeout(
                idempotency_key=f"gateway:resume-closeout:{service.kernel.state.revision}"
            )
        return campaign_gateway_result(
            service, operation="resume", operations=operations
        )

    def apply_plan(
        self,
        run_id: str,
        plan: Mapping[str, Any],
        *,
        run_dir: str | Path | None = None,
        materialize: bool = False,
    ) -> dict[str, Any]:
        service = self._open(run_id, run_dir=run_dir)
        operations: dict[str, Any] = {
            "global_plan": service.apply_global_plan(
                plan,
                idempotency_key=f"gateway:plan:{campaign_payload_digest(plan)[:24]}",
            )
        }
        if materialize:
            revision = service.kernel.state.graph_revision
            operations["materialization"] = service.execute_frontier_materialization(
                idempotency_key=f"gateway:frontier-materialization:{revision}"
            )
        return campaign_gateway_result(
            service, operation="apply-plan", operations=operations
        )

    def status(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return campaign_gateway_result(
            self._open(run_id, run_dir=run_dir), operation="status"
        )

    def cancel(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        reasons: Iterable[str] = ("user_requested",),
        idempotency_key: str = "gateway:cancel",
    ) -> dict[str, Any]:
        """Cancel one campaign through its canonical Kernel state machine."""

        service = self._open(run_id, run_dir=run_dir)
        event = service.kernel.cancel(
            idempotency_key=idempotency_key,
            reasons=reasons,
        )
        return campaign_gateway_result(
            service,
            operation="cancel",
            operations={"cancellation": event.to_dict()},
        )

    def workbench(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        service = self._open(run_id, run_dir=run_dir)
        return {
            "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
            "operation": "workbench",
            "run_id": service.kernel.spec.run_id,
            **service.workbench(),
        }

    def validate(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return validate_campaign(self._open(run_id, run_dir=run_dir))

    def replay(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return replay_campaign(self._open(run_id, run_dir=run_dir))

    def benchmark(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        iterations: int = 3,
    ) -> dict[str, Any]:
        service = self._open(run_id, run_dir=run_dir)
        return benchmark_campaign(service, iterations=iterations)

    def export(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        service = self._open(run_id, run_dir=run_dir)
        return export_campaign(service, output_dir=output_dir)

    def gc_plan(self, *, minimum_age_s: float = 86_400.0) -> dict[str, Any]:
        return plan_artifact_gc(
            self.paths,
            self.index,
            minimum_age_s=minimum_age_s,
        )

    def list_runs(self, *, limit: int = 100) -> dict[str, Any]:
        rows = self.index.list_runs(limit=max(1, min(1_000, int(limit))))
        return {
            "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
            "operation": "list",
            "run_count": len(rows),
            "runs": rows,
        }

    def remove_run_from_history(self, run_id: str) -> dict[str, Any]:
        """Hide a finished run from the task queue without deleting evidence."""

        identity = self._normalize_run_id(run_id)
        if self.index.get_run(identity) is None:
            raise CampaignGatewayError(f"run_not_found:{identity}")
        return self.index.remove_run_projection(identity)

    def _open(
        self,
        run_id: str,
        *,
        run_dir: str | Path | None = None,
        director_runner: Any = None,
        director_config: Any = None,
    ) -> RetrosynthesisCampaignService:
        identity = self._normalize_run_id(run_id)
        directory = self._run_dir(identity, explicit=run_dir, require=True)
        service = RetrosynthesisCampaignService.open(
            self.paths.runtime_root,
            directory,
            artifact_store_root=self.paths.artifact_store_root,
            run_index_path=self.paths.run_index_path,
            director_runner=director_runner,
            director_config=director_config,
        )
        if service.kernel.spec.run_id != identity:
            raise CampaignGatewayError("run_directory_identity_mismatch")
        return service

    def _run_dir(
        self,
        run_id: str,
        *,
        explicit: str | Path | None,
        require: bool,
    ) -> Path:
        if explicit is not None:
            directory = Path(explicit).expanduser().resolve()
        else:
            manifest = self.index.get_run(run_id)
            if manifest and manifest.get("run_dir"):
                directory = Path(str(manifest["run_dir"])).expanduser().resolve()
            else:
                directory = self.paths.runs_root / run_segment(run_id)
        if require and not (directory / ".autoplanner" / "kernel" / "run_spec.json").is_file():
            raise CampaignGatewayError(f"run_not_found:{run_id}")
        return directory

    @staticmethod
    def _normalize_run_id(value: str) -> str:
        identity = str(value or "").strip()
        if not _RUN_ID.fullmatch(identity):
            raise CampaignGatewayError("run_id_invalid")
        return identity

    def _default_stock_oracle_reference(self, *, boundary: str) -> StockOracleReference:
        return default_stock_oracle_reference(self.providers, boundary=boundary)
__all__ = [
    "CAMPAIGN_GATEWAY_RESULT_SCHEMA",
    "CampaignGateway",
    "CampaignGatewayError",
]
