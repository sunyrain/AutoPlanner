"""Small V4 campaign service coordinating the canonical application modules.

This is the only orchestration owner for new runs.  It keeps global Codex
reasoning, deterministic workers, graph ingestion, the work frontier, and
proof closeout on one ``RunKernel`` without importing the legacy blackboard
controller or its private queues.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any, Iterable, Mapping

from cascade_planner.application.campaign_context import (
    CampaignContext,
    CampaignContextCompiler,
)
from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
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
from cascade_planner.application.run_kernel import RunKernel, RunKernelError, RunSpec
from cascade_planner.application.route_workbench import (
    compile_route_workbench,
    compile_route_workbench_delta,
)
from cascade_planner.application.worker_runtime import (
    WorkerCommand,
    WorkerResult,
    WorkerRuntime,
)
from cascade_planner.orchestration.global_campaign_director import (
    DirectorConfig,
    DirectorOutcome,
    DirectorRunner,
    GlobalCampaignDirector,
)


CAMPAIGN_SERVICE_STATUS_SCHEMA = "retrosynthesis_campaign_service_status.v1"


class RetrosynthesisCampaignService:
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

    def register_artifact_authorities(
        self,
        values: Mapping[str, str],
    ) -> None:
        self.workers.artifact_authorities.update(
            {
                str(digest).lower(): str(scope)
                for digest, scope in values.items()
                if str(digest).strip() and str(scope).strip()
            }
        )

    def apply_global_plan(
        self,
        plan: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.apply_batch(
            CanonicalIngestionBatch(global_plans=(dict(plan),)),
            idempotency_key=idempotency_key,
        )

    def run_global_director(
        self,
        *,
        mode: str,
        material_events: Iterable[str] = (),
        force: bool = False,
        idempotency_key: str,
    ) -> DirectorOutcome:
        context = self.compile_global_context(material_events=material_events)
        outcome = self.director.run(context, mode=mode, force=force)
        self._previous_context = context
        if outcome.plan is not None and outcome.status == "accepted":
            admitted_ids = sorted(
                str(row.get("proposal_id") or "")
                for row in outcome.proposal_audits
                if row.get("accepted") is True and str(row.get("proposal_id") or "")
            )
            self.apply_global_plan(
                {**outcome.plan.to_dict(), "_host_admitted_proposal_ids": admitted_ids},
                idempotency_key=f"{idempotency_key}:plan",
            )
        return outcome

    def compile_global_context(
        self, *, material_events: Iterable[str] = ()
    ) -> CampaignContext:
        graph = self.graph_store.load()
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        return self.context_compiler.compile(
            kernel=self.kernel,
            hypergraph=graph,
            route_portfolio=portfolio,
            material_events=material_events,
            previous=self._previous_context,
        )

    def execute_frontier_materialization(
        self,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        commands = self.graph_store.frontier_materialization_commands()
        results: list[WorkerResult] = []
        stopped_reasons: list[str] = []
        for command in commands:
            try:
                results.append(self.workers.execute(command))
            except RunKernelError as exc:
                reason = str(exc)
                if "budget_exhausted" not in reason:
                    raise
                stopped_reasons.append(reason)
                break
        if not results:
            return {
                "changed": False,
                "executed_command_count": 0,
                "skipped_command_count": len(commands),
                "stopped_reasons": sorted(set(stopped_reasons)),
                "graph": self.graph_store.load(),
            }
        applied = self.apply_worker_results(
            tuple(results),
            idempotency_key=idempotency_key,
        )
        return {
            **applied,
            "executed_command_count": len(results),
            "skipped_command_count": max(0, len(commands) - len(results)),
            "stopped_reasons": sorted(set(stopped_reasons)),
        }

    def execute_commands(
        self,
        commands: Iterable[WorkerCommand],
        *,
        idempotency_key: str,
        include_scheduled: bool = True,
    ) -> dict[str, Any]:
        results: list[WorkerResult] = []
        material_events: set[str] = set()
        for command in commands:
            if include_scheduled:
                batch = self.workers.execute_pipeline(command)
                results.extend(batch.results)
                material_events.update(batch.material_events)
            else:
                result = self.workers.execute(command)
                results.append(result)
                material_events.update(result.material_events)
        applied = self.apply_worker_results(
            results,
            idempotency_key=idempotency_key,
        )
        return {
            **applied,
            "executed_command_count": len(results),
            "material_events": sorted(material_events),
        }

    def apply_worker_results(
        self,
        results: Iterable[WorkerResult | Mapping[str, Any]],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.apply_batch(
            CanonicalIngestionBatch(worker_results=tuple(results)),
            idempotency_key=idempotency_key,
        )

    def apply_batch(
        self,
        batch: CanonicalIngestionBatch,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        result = self.graph_store.apply(
            batch,
            worker_runtime=self.workers if batch.worker_results else None,
            idempotency_key=idempotency_key,
        )
        if result.get("changed") is True:
            self._publish_graph_frontier(
                result["graph"],
                idempotency_key=f"graph-frontier:{idempotency_key}",
            )
        return result

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
            "frontier": list(state.deficits),
            "portfolio": portfolio,
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
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        snapshot = compile_route_workbench(
            graph, portfolio, campaign_summary=campaign_summary
        )
        return {
            "snapshot": snapshot,
            "delta": compile_route_workbench_delta(previous, snapshot),
        }

    def publish_workbench(
        self,
        *,
        previous: Mapping[str, Any] | None = None,
        campaign_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish the display projection without changing scientific state."""

        result = self.workbench(previous=previous, campaign_summary=campaign_summary)
        snapshot = result["snapshot"]
        ref = self.kernel.artifacts.put_json(
            snapshot,
            logical_name="retrosynthesis_route_workbench.json",
            producer="autoplanner.route_workbench",
        )
        run_digest = hashlib.sha256(self.kernel.spec.run_id.encode("utf-8")).hexdigest()
        self.kernel.artifacts.write_pointer(
            f"u/{run_digest[:24]}/latest",
            ref,
            metadata={
                "run_id": self.kernel.spec.run_id,
                "graph_revision": self.kernel.state.graph_revision,
                "portfolio_route_count": snapshot["portfolio"]["route_count"],
            },
        )
        self.kernel.index.index_artifact(
            run_id=self.kernel.spec.run_id,
            artifact_id="retrosynthesis_route_workbench",
            ref=ref,
            revision=self.kernel.state.graph_revision,
            authority_scope="display_projection_only",
        )
        return {**result, "snapshot_ref": ref.to_dict()}

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
