"""Worker dispatch and canonical mutation operations for the V4 service."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.canonical_hypergraph import CanonicalIngestionBatch
from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.application.worker_runtime import WorkerCommand, WorkerResult


class _RetrosynthesisServiceExecutionMixin:
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
        self.terminalize_global_budget_if_reached(
            idempotency_key=(
                "campaign-service:global-budget-terminal:"
                f"{state.revision}"
            )
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

    def terminalize_global_budget_if_reached(
        self,
        *,
        idempotency_key: str,
    ) -> bool:
        """Persist global task/wall-time exhaustion before later dispositions."""

        state = self.kernel.state
        if state.status == "budget_exhausted":
            return True
        if state.status != "running":
            return False
        limits = self.kernel.spec.limits
        reasons = []
        if state.settled_task_count >= limits.max_total_tasks:
            reasons.append("run_total_task_budget_exhausted")
        if state.task_wall_time_s >= limits.max_run_wall_time_s:
            reasons.append("run_wall_time_budget_exhausted")
        if not reasons:
            return False
        self.kernel.transition(
            "budget_exhausted",
            idempotency_key=idempotency_key,
            reasons=reasons,
        )
        return True

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



__all__: list[str] = []
