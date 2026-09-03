"""Coordinate global reasoning and deterministic work on one V4 RunKernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.campaign_context import CampaignContext, CampaignContextCompiler
from cascade_planner.application.campaign_quality_state import (
    compile_campaign_quality_state,
)
from cascade_planner.application.campaign_action_status import (
    compile_active_campaign_actions,
)
from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
)
from cascade_planner.application.frontier_runtime import publish_frontier_items
from cascade_planner.application.proof_portfolio import (
    PortfolioConfig,
    compile_proof_portfolio,
    publish_proof_portfolio,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
)
from cascade_planner.application.run_kernel import RunKernel, RunSpec
from cascade_planner.application.route_workbench import (
    compile_route_workbench,
    compile_route_workbench_delta,
)
from cascade_planner.application.worker_runtime import WorkerRuntime
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    DirectorRunner,
    GlobalCampaignDirector,
)
from cascade_planner.orchestration.program_admission_runtime import (
    admit_program_projection,
    program_projection_read,
    program_store_read,
)
from cascade_planner.orchestration.retrosynthesis_service_execution import (
    _RetrosynthesisServiceExecutionMixin,
)
from cascade_planner.orchestration.retrosynthesis_service_planning import (
    _RetrosynthesisServicePlanningMixin,
)
from cascade_planner.orchestration.workbench_publication import publish_workbench_snapshot, published_workbench_campaign_summary


CAMPAIGN_SERVICE_STATUS_SCHEMA = "retrosynthesis_campaign_service_status.v1"


class RetrosynthesisCampaignService(
    _RetrosynthesisServicePlanningMixin,
    _RetrosynthesisServiceExecutionMixin,
):
    """Coordinate one resumable campaign through canonical boundaries only."""

    def __init__(
        self,
        kernel: RunKernel,
        *,
        artifact_authorities: Mapping[str, str] | None = None,
        director_runner: DirectorRunner | None = None,
        director_config: DirectorConfig | None = None,
    ) -> None:
        self.kernel = kernel
        self.graph_store = CanonicalHypergraphStore(kernel)
        self.workers = WorkerRuntime(
            kernel,
            build_retrosynthesis_worker_handlers(),
            artifact_authorities=artifact_authorities,
        )
        self.context_compiler = CampaignContextCompiler()
        self.director = GlobalCampaignDirector(
            kernel,
            runner=director_runner,
            config=director_config,
        )
        self._previous_context: CampaignContext | None = None

    @classmethod
    def create(
        cls,
        runtime_root: str | Path,
        run_dir: str | Path,
        *,
        spec: RunSpec,
        artifact_store_root: str | Path | None = None,
        run_index_path: str | Path | None = None,
        artifact_authorities: Mapping[str, str] | None = None,
        director_runner: DirectorRunner | None = None,
        director_config: DirectorConfig | None = None,
    ) -> "RetrosynthesisCampaignService":
        kernel = RunKernel(
            runtime_root,
            run_dir,
            spec=spec,
            artifact_store_root=artifact_store_root,
            run_index_path=run_index_path,
        )
        if kernel.state.status == "created":
            kernel.start()
        return cls(
            kernel,
            artifact_authorities=artifact_authorities,
            director_runner=director_runner,
            director_config=director_config,
        )

    @classmethod
    def open(
        cls,
        runtime_root: str | Path,
        run_dir: str | Path,
        *,
        artifact_store_root: str | Path | None = None,
        run_index_path: str | Path | None = None,
        artifact_authorities: Mapping[str, str] | None = None,
        director_runner: DirectorRunner | None = None,
        director_config: DirectorConfig | None = None,
    ) -> "RetrosynthesisCampaignService":
        return cls(
            RunKernel(
                runtime_root,
                run_dir,
                artifact_store_root=artifact_store_root,
                run_index_path=run_index_path,
            ),
            artifact_authorities=artifact_authorities,
            director_runner=director_runner,
            director_config=director_config,
        )

    def register_artifact_authorities(self, values: Mapping[str, str]) -> None:
        self.workers.artifact_authorities.update(
            {
                str(digest).lower(): str(scope)
                for digest, scope in values.items()
                if str(digest).strip() and str(scope).strip()
            }
        )

    def closeout(
        self,
        *,
        idempotency_key: str,
        config: PortfolioConfig | None = None,
        budget_exhausted: bool = False,
    ) -> dict[str, Any]:
        return publish_proof_portfolio(
            self.kernel,
            self.graph_store.load(),
            idempotency_key=idempotency_key,
            config=config,
            budget_exhausted=budget_exhausted,
        )

    def status(self) -> dict[str, Any]:
        graph = self.graph_store.load()
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        workbench = compile_route_workbench(graph, portfolio)
        quality_state = compile_campaign_quality_state(workbench=workbench)
        state = self.kernel.state
        return {
            "schema_version": CAMPAIGN_SERVICE_STATUS_SCHEMA,
            "run_id": self.kernel.spec.run_id,
            "status": state.status,
            "graph_revision": state.graph_revision,
            "evidence_revision": state.evidence_revision,
            "attempt_count": state.attempt_count,
            "accepted_expansion_count": state.accepted_expansion_count,
            "model_totals": dict(state.model_totals),
            "native_search": self.kernel.native_search_budget(),
            "task_budget": self.kernel.task_budget(),
            "active_actions": compile_active_campaign_actions(state),
            "frontier": list(state.deficits),
            "portfolio": portfolio,
            "campaign_spec": self.kernel.spec.campaign_spec.to_dict(),
            "quality_state": quality_state,
            "stop_decision": self.kernel.decide_stop().to_dict(),
            "semantics": {
                "single_kernel": True,
                "single_graph": True,
                "single_frontier": True,
                "blackboard_is_not_authority": True,
            },
        }

    def workbench(
        self,
        *,
        previous: Mapping[str, Any] | None = None,
        campaign_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        graph = self.graph_store.load()
        if campaign_summary is None:
            campaign_summary = published_workbench_campaign_summary(self.kernel, graph)
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        snapshot = compile_route_workbench(graph, portfolio, campaign_summary=campaign_summary)
        quality_state = dict((campaign_summary or {}).get("quality_state") or {})
        if not quality_state:
            quality_state = compile_campaign_quality_state(workbench=snapshot)
        return {
            "snapshot": snapshot,
            "delta": compile_route_workbench_delta(previous, snapshot),
            "quality_state": quality_state,
        }

    def program_projection(self) -> dict[str, Any]:
        return program_projection_read(self.graph_store)

    def program_store(self) -> dict[str, Any]:
        return program_store_read(self.kernel, self.graph_store)

    def admit_programs(self, *, enable_program_admission: bool = False) -> dict[str, Any]:
        return admit_program_projection(
            self.kernel,
            self.graph_store,
            enable_program_admission=enable_program_admission,
        )

    def publish_workbench(
        self,
        *,
        previous: Mapping[str, Any] | None = None,
        campaign_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish the display projection without changing scientific state."""

        result = self.workbench(previous=previous, campaign_summary=campaign_summary)
        return {
            **result,
            "snapshot_ref": publish_workbench_snapshot(self.kernel, result["snapshot"]),
        }

    def _publish_graph_frontier(
        self,
        graph: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> None:
        frontier = dict(graph.get("deficit_frontier") or {})
        publish_frontier_items(
            self.kernel,
            frontier.get("items") or [],
            source_revision=int(graph.get("revision") or 0),
            idempotency_key=idempotency_key,
            projection_sha256=str(frontier.get("content_sha256") or ""),
        )


__all__ = [
    "CAMPAIGN_SERVICE_STATUS_SCHEMA",
    "RetrosynthesisCampaignService",
]
