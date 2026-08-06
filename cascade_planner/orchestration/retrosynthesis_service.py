"""Coordinate global reasoning and deterministic work on one V4 RunKernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.application.campaign_context import CampaignContext, CampaignContextCompiler
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
from cascade_planner.orchestration import route_innovation_runtime
from cascade_planner.orchestration import program_innovation_runtime
from cascade_planner.orchestration.experimental_claim_admission_runtime import (
    admit_route_experimental_claims,
)
from cascade_planner.orchestration.program_admission_runtime import (
    admit_program_projection,
    program_projection_read,
    program_store_read,
)
from cascade_planner.orchestration.workbench_publication import publish_workbench_snapshot, published_workbench_campaign_summary


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

    def register_artifact_authorities(self, values: Mapping[str, str]) -> None:
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
        proposal_origin_kind: str = "manual",
        proposal_origin_ref: str = "",
    ) -> dict[str, Any]:
        trusted_plan = {
            key: value
            for key, value in dict(plan).items()
            if key not in {"_proposal_origin_kind", "_proposal_origin_ref"}
        }
        trusted_plan["_proposal_origin_kind"] = str(proposal_origin_kind).lower()
        trusted_plan["_proposal_origin_ref"] = str(proposal_origin_ref)
        return self.apply_batch(
            CanonicalIngestionBatch(global_plans=(trusted_plan,)),
            idempotency_key=idempotency_key,
        )

    def run_global_director(
        self,
        *,
        mode: str,
        material_events: Iterable[str] = (),
        evidence_observations: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        force: bool = False,
        idempotency_key: str,
    ) -> DirectorOutcome:
        context = self.compile_global_context(
            material_events=material_events,
            evidence_observations=evidence_observations,
        )
        return self.run_global_director_with_context(
            context,
            mode=mode,
            force=force,
            idempotency_key=idempotency_key,
        )

    def run_global_director_with_context(
        self,
        context: CampaignContext,
        *,
        mode: str,
        force: bool = False,
        idempotency_key: str,
    ) -> DirectorOutcome:
        """Run against an already frozen canonical context and merge by union."""

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
                proposal_origin_kind="codex_global_director",
                proposal_origin_ref=(
                    f"director_task:{outcome.task_id}"
                    if outcome.task_id
                    else f"director_context:{outcome.context_sha256}"
                ),
            )
        if mode == "initial_architecture" and outcome.status in {
            "accepted",
            "budget_exhausted",
        }:
            self._record_settled_initial_architecture(
                outcome,
                idempotency_key=f"{idempotency_key}:settled",
            )
        return outcome

    def _record_settled_initial_architecture(
        self,
        outcome: DirectorOutcome,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Persist a terminal initial-architecture attempt as operational state.

        The record is deliberately separate from accepted chemistry hypotheses:
        an accepted Director response may still have every proposal rejected by
        host admission.  That is a settled attempt, not a reason to schedule the
        same initial pass after every unrelated graph revision.
        """

        graph = self.graph_store.load()
        target_id = str(graph.get("target_molecule_id") or "")
        audits = [dict(row) for row in outcome.proposal_audits]
        accepted_count = sum(row.get("accepted") is True for row in audits)
        signal_id = f"director-attempt:initial_architecture:{target_id}"
        return self.publish_action_signals(
            (
                {
                    "signal_id": signal_id,
                    "deficit_id": signal_id,
                    "kind": "architecture",
                    "status": "resolved",
                    "object_id": target_id,
                    "entity_ids": [target_id],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": True,
                    "reason": "initial_global_architecture_attempt_settled",
                    "metadata": {
                        "director_mode": "initial_architecture",
                        "director_status": outcome.status,
                        "context_sha256": outcome.context_sha256,
                        "task_id": outcome.task_id,
                        "invoked": outcome.invoked,
                        "cache_hit": outcome.cache_hit,
                        "proposal_audit_count": len(audits),
                        "host_admitted_proposal_count": accepted_count,
                        "host_rejected_proposal_count": len(audits) - accepted_count,
                    },
                    "resolution": {
                        "status": outcome.status,
                        "artifact_sha256": outcome.artifact_sha256,
                        "reasons": list(outcome.reasons),
                    },
                },
            ),
            idempotency_key=idempotency_key,
        )

    def compile_global_context(
        self,
        *,
        material_events: Iterable[str] = (),
        evidence_observations: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
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
            evidence_ledger=evidence_observations,
            material_events=material_events,
            previous=self._previous_context,
            # The audit view does not spend the director's separately
            # reserved prompt budget.
            enforce_limit=False,
        )

    def review_route_innovations(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return route_innovation_runtime.review_route_innovations(
            self.graph_store.load(),
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
        )

    def review_route_program_innovations(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return program_innovation_runtime.review_route_program_innovations(
            self.graph_store.load(),
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
        )

    def admit_route_experimental_claims(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experimental_claim_admission: bool = False,
    ) -> dict[str, Any]:
        return admit_route_experimental_claims(
            self.kernel,
            self.graph_store,
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experimental_claim_admission=(
                enable_experimental_claim_admission
            ),
        )

    def execute_frontier_materialization(
        self,
        *,
        idempotency_key: str,
        hypothesis_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        commands = self.graph_store.frontier_materialization_commands(hypothesis_ids)
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
        command_rows = tuple(commands)
        state = self.kernel.state
        limits = self.kernel.spec.limits
        global_budget_reasons = []
        if state.settled_task_count >= limits.max_total_tasks:
            global_budget_reasons.append("run_total_task_budget_exhausted")
        if state.task_wall_time_s >= limits.max_run_wall_time_s:
            global_budget_reasons.append("run_wall_time_budget_exhausted")
        if (
            state.status == "running"
            and global_budget_reasons
        ):
            self.kernel.transition(
                "budget_exhausted",
                idempotency_key=(
                    "campaign-service:global-budget-terminal:"
                    f"{state.revision}"
                ),
                reasons=global_budget_reasons,
            )
            state = self.kernel.state
        if state.status == "budget_exhausted":
            return {
                "status": "budget_exhausted",
                "changed": False,
                "reused": False,
                "graph": self.graph_store.load(),
                "graph_ref": {},
                "rejected": [],
                "executed_command_count": 0,
                "skipped_command_count": len(command_rows),
                "stopped_reasons": list(state.failure_reasons),
                "material_events": [],
                "semantics": {
                    "terminal_kernel_reserves_no_new_tasks": True,
                    "reached_global_cap_is_terminalized_before_worker_dispatch": True,
                    "closeout_projection_remains_available": True,
                    "command_idempotency_keys_are_not_consumed": True,
                },
            }
        results: list[WorkerResult] = []
        material_events: set[str] = set()
        for command in command_rows:
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
            "skipped_command_count": 0,
            "stopped_reasons": [],
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

    def publish_action_signals(
        self,
        signals: Iterable[Mapping[str, Any]],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Publish operational events into the one canonical deficit frontier."""

        return self.apply_batch(
            CanonicalIngestionBatch(
                action_signals=tuple(
                    dict(value) for value in signals if isinstance(value, Mapping)
                )
            ),
            idempotency_key=idempotency_key,
        )

    def resolve_action_signals(
        self,
        signal_ids: Iterable[str],
        *,
        resolution: Mapping[str, Any] | None = None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        graph = self.graph_store.load()
        rows = []
        for signal_id in sorted({str(value) for value in signal_ids if str(value)}):
            existing = dict(
                dict(graph.get("action_signals") or {}).get(signal_id) or {}
            )
            if not existing or existing.get("status") == "resolved":
                continue
            existing.pop("content_sha256", None)
            rows.append(
                {
                    **existing,
                    "status": "resolved",
                    "resolution": dict(resolution or {}),
                }
            )
        return self.publish_action_signals(rows, idempotency_key=idempotency_key)

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
            "native_search": self.kernel.native_search_budget(),
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
        if campaign_summary is None:
            campaign_summary = published_workbench_campaign_summary(self.kernel, graph)
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        snapshot = compile_route_workbench(graph, portfolio, campaign_summary=campaign_summary)
        return {
            "snapshot": snapshot,
            "delta": compile_route_workbench_delta(previous, snapshot),
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
