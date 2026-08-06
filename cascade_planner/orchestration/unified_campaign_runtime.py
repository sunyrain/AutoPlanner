"""Run revision-bound campaign actions through the single RunKernel ledger."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import time
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.campaign_actions import (
    CampaignAction,
    CampaignActionKind,
    bind_scheduled_action,
)
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.runtime.artifact_store import ArtifactReferenceError


CAMPAIGN_ACTION_OUTCOME_SCHEMA = "campaign_action_outcome.v1"
CAMPAIGN_ACTION_EXECUTION_SCHEMA = "campaign_action_execution.v1"
CAMPAIGN_ACTION_COHORT_SCHEMA = "campaign_action_concurrent_cohort.v1"
CAMPAIGN_ANYTIME_LOOP_SCHEMA = "campaign_anytime_action_loop.v1"

CampaignActionHandler = Callable[[CampaignAction], Mapping[str, Any]]
CampaignActionStateProvider = Callable[[], Mapping[str, Any]]
CampaignActionExecutionObserver = Callable[[int, Mapping[str, Any]], None]


class CampaignActionRuntimeError(RuntimeError):
    """Raised when a persisted action receipt cannot be trusted or replayed."""


class CampaignActionRuntime:
    """Schedule and execute registered actions without a second work queue."""

    def __init__(
        self,
        kernel: RunKernel,
        handlers: Mapping[CampaignActionKind | str, CampaignActionHandler],
        *,
        scheduler_policy: str = "adaptive",
    ) -> None:
        self.kernel = kernel
        self.scheduler_policy = str(scheduler_policy or "adaptive")
        self.handlers = {
            (
                kind
                if isinstance(kind, CampaignActionKind)
                else CampaignActionKind(str(kind))
            ): handler
            for kind, handler in handlers.items()
        }
        if any(not callable(handler) for handler in self.handlers.values()):
            raise TypeError("campaign action handlers must be callable")

    def schedule_and_execute(
        self,
        opportunity_set: Mapping[str, Any],
        *,
        milestones: Mapping[str, Any],
        resource_availability: Mapping[str, Any],
        excluded_action_ids: tuple[str, ...] = (),
        round_robin_cursor: int = 0,
    ) -> dict[str, Any]:
        decision = schedule_next_action(
            opportunity_set,
            milestones=milestones,
            resource_availability=resource_availability,
            in_flight_action_ids=excluded_action_ids,
            available_action_kinds=tuple(
                sorted(kind.value for kind in self.handlers)
            ),
            policy=self.scheduler_policy,
            round_robin_cursor=round_robin_cursor,
        )
        if not decision.get("selected_action_id"):
            return {
                "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
                "status": "no_action",
                "decision": decision,
                "cache_hit": False,
                "semantics": {
                    "no_registered_eligible_action": True,
                    "no_second_queue_created": True,
                },
            }
        action = bind_scheduled_action(
            decision,
            input_revision=self.kernel.state.graph_revision,
        )
        return self.execute(action, decision=decision)

    def execute_slice(
        self,
        *,
        opportunity_provider: CampaignActionStateProvider,
        milestones_provider: CampaignActionStateProvider,
        resource_availability_provider: CampaignActionStateProvider,
        max_actions: int,
        excluded_action_ids: tuple[str, ...] = (),
        on_execution: CampaignActionExecutionObserver | None = None,
    ) -> list[dict[str, Any]]:
        """Run one bounded scheduler slice without creating another queue."""

        result = self.run_anytime(
            opportunity_provider=opportunity_provider,
            milestones_provider=milestones_provider,
            resource_availability_provider=resource_availability_provider,
            max_actions=max_actions,
            max_consecutive_no_gain=max(1, int(max_actions)) + 1,
            excluded_action_ids=excluded_action_ids,
            on_execution=on_execution,
        )
        return [dict(value) for value in result.get("executions") or []]

    def run_anytime(
        self,
        *,
        opportunity_provider: CampaignActionStateProvider,
        milestones_provider: CampaignActionStateProvider,
        resource_availability_provider: CampaignActionStateProvider,
        max_actions: int,
        max_consecutive_no_gain: int = 3,
        excluded_action_ids: tuple[str, ...] = (),
        concurrent_start_kinds: tuple[CampaignActionKind | str, ...] = (),
        on_execution: CampaignActionExecutionObserver | None = None,
    ) -> dict[str, Any]:
        """Own one bounded anytime loop over the latest canonical revision."""

        executions: list[dict[str, Any]] = []
        globally_excluded = {
            str(value) for value in excluded_action_ids if str(value)
        }
        attempted_by_revision: dict[int, set[str]] = {}
        no_gain_bindings: dict[str, str] = {}
        consecutive_no_gain = 0
        termination = "action_limit"
        action_limit = max(1, int(max_actions))
        no_gain_limit = max(1, int(max_consecutive_no_gain))
        normalized_start_kinds = tuple(
            (
                kind
                if isinstance(kind, CampaignActionKind)
                else CampaignActionKind(str(kind))
            )
            for kind in concurrent_start_kinds
        )
        start_cohort: dict[str, Any] = {}
        if len(normalized_start_kinds) >= 2 and action_limit >= 2:
            initial_revision = self.kernel.state.graph_revision
            start_cohort = self.execute_concurrent_cohort(
                opportunity_provider(),
                action_kinds=normalized_start_kinds,
                milestones=milestones_provider(),
                resource_availability=resource_availability_provider(),
                excluded_action_ids=tuple(sorted(globally_excluded)),
            )
            cohort_executions = [
                dict(value) for value in start_cohort.get("executions") or []
            ][:action_limit]
            for execution in cohort_executions:
                executions.append(execution)
                action_row = dict(execution.get("action") or {})
                action_id = str(action_row.get("action_id") or "")
                action_revision = int(
                    action_row.get("input_revision")
                    if action_row.get("input_revision") is not None
                    else initial_revision
                )
                if action_id:
                    attempted_by_revision.setdefault(action_revision, set()).add(
                        action_id
                    )
                if on_execution is not None:
                    on_execution(len(executions), execution)
                gained = _execution_gained(execution)
                if action_id and not gained:
                    no_gain_bindings[action_id] = str(
                        action_row.get("opportunity_sha256") or ""
                    )
                consecutive_no_gain = 0 if gained else consecutive_no_gain + 1

        for index in range(len(executions) + 1, action_limit + 1):
            input_revision = self.kernel.state.graph_revision
            attempted = attempted_by_revision.setdefault(input_revision, set())
            opportunity_set = opportunity_provider()
            current_opportunity_sha256 = {
                str(row.get("action_id") or ""): str(
                    row.get("content_sha256") or ""
                )
                for row in opportunity_set.get("actions") or []
                if isinstance(row, Mapping) and str(row.get("action_id") or "")
            }
            no_gain_excluded = {
                action_id
                for action_id, opportunity_sha256 in no_gain_bindings.items()
                if current_opportunity_sha256.get(action_id) == opportunity_sha256
            }
            execution = self.schedule_and_execute(
                opportunity_set,
                milestones=milestones_provider(),
                resource_availability=resource_availability_provider(),
                excluded_action_ids=tuple(
                    sorted(globally_excluded | attempted | no_gain_excluded)
                ),
                round_robin_cursor=index - 1,
            )
            if execution.get("status") == "no_action":
                termination = "no_action"
                break
            executions.append(execution)
            action_id = str(
                dict(execution.get("action") or {}).get("action_id") or ""
            )
            if action_id:
                attempted.add(action_id)
            if on_execution is not None:
                on_execution(index, execution)
            gained = _execution_gained(execution)
            if action_id and not gained:
                no_gain_bindings[action_id] = str(
                    dict(execution.get("action") or {}).get(
                        "opportunity_sha256"
                    )
                    or ""
                )
            consecutive_no_gain = 0 if gained else consecutive_no_gain + 1
            if consecutive_no_gain >= no_gain_limit:
                termination = "converged_low_marginal_gain"
                break
        result = {
            "schema_version": CAMPAIGN_ANYTIME_LOOP_SCHEMA,
            "termination": termination,
            "execution_count": len(executions),
            "consecutive_no_gain": consecutive_no_gain,
            "no_gain_binding_count": len(no_gain_bindings),
            "start_cohort": start_cohort,
            "executions": executions,
            "final_graph_revision": self.kernel.state.graph_revision,
            "semantics": {
                "single_scheduler_loop": True,
                "scheduler_policy": self.scheduler_policy,
                "latest_revision_recompiled_each_iteration": True,
                "same_revision_start_cohort_is_non_blocking": (
                    start_cohort.get("status") == "completed"
                ),
                "cohort_failures_do_not_cancel_peers": True,
                "cohort_observation_order_is_stable": True,
                "B4_and_B5_do_not_stop_the_loop": True,
                "no_action_and_low_gain_converge_finitely": True,
            },
        }
        result["content_sha256"] = _digest(result)
        return result

    def execute_concurrent_cohort(
        self,
        opportunity_set: Mapping[str, Any],
        *,
        action_kinds: Iterable[CampaignActionKind | str],
        milestones: Mapping[str, Any],
        resource_availability: Mapping[str, Any],
        excluded_action_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Reserve same-revision actions first, then execute without peer cancellation."""

        input_revision = self.kernel.state.graph_revision
        normalized_kinds = tuple(
            (
                kind
                if isinstance(kind, CampaignActionKind)
                else CampaignActionKind(str(kind))
            )
            for kind in action_kinds
        )
        decisions: list[dict[str, Any]] = []
        actions: list[CampaignAction] = []
        selected_action_ids = {
            str(value) for value in excluded_action_ids if str(value)
        }
        for kind in normalized_kinds:
            decision = schedule_next_action(
                opportunity_set,
                milestones=milestones,
                resource_availability=resource_availability,
                in_flight_action_ids=tuple(sorted(selected_action_ids)),
                available_action_kinds=(kind.value,),
                policy=self.scheduler_policy,
            )
            if not decision.get("selected_action_id"):
                continue
            action = bind_scheduled_action(decision, input_revision=input_revision)
            decisions.append(decision)
            actions.append(action)
            selected_action_ids.add(action.action_id)
        if len(actions) < 2:
            result = {
                "schema_version": CAMPAIGN_ACTION_COHORT_SCHEMA,
                "status": "not_launched",
                "input_revision": input_revision,
                "requested_action_kinds": [kind.value for kind in normalized_kinds],
                "selected_action_ids": [action.action_id for action in actions],
                "decisions": decisions,
                "executions": [],
                "semantics": {
                    "cohort_requires_multiple_eligible_actions": True,
                    "fallback_to_single_scheduler_loop": True,
                },
            }
            result["content_sha256"] = _digest(result)
            return result

        prepared: list[
            tuple[CampaignAction, Mapping[str, Any], Mapping[str, Any]]
        ] = []
        cached_by_execution_id: dict[str, dict[str, Any]] = {}
        for action, decision in zip(actions, decisions, strict=True):
            cached = self._load_cached(action)
            if cached is not None:
                cached_by_execution_id[action.execution_id] = {
                    "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
                    "status": str(cached.get("status") or "completed"),
                    "action": action.to_dict(),
                    "decision": dict(decision),
                    "outcome": cached,
                    "cache_hit": True,
                }
                continue
            if self.handlers.get(action.kind) is None:
                raise CampaignActionRuntimeError(
                    f"campaign_action_handler_missing:{action.kind.value}"
                )
            prepared.append((action, decision, self._reserve_action(action)))

        futures = {}
        if prepared:
            with ThreadPoolExecutor(
                max_workers=len(prepared),
                thread_name_prefix="campaign-start",
            ) as executor:
                futures = {
                    action.execution_id: executor.submit(
                        self._execute_reserved,
                        action,
                        decision=decision,
                        resource_reservation=resource_reservation,
                    )
                    for action, decision, resource_reservation in prepared
                }
                wait(tuple(futures.values()))

        executions: list[dict[str, Any]] = []
        action_execution_ids = [action.execution_id for action in actions]
        cohort_id = "campaign-cohort:" + _digest(
            {
                "input_revision": input_revision,
                "action_execution_ids": action_execution_ids,
            }
        )
        for observation_index, action in enumerate(actions, start=1):
            if action.execution_id in cached_by_execution_id:
                execution = dict(cached_by_execution_id[action.execution_id])
            else:
                execution = dict(futures[action.execution_id].result())
            execution["cohort"] = {
                "cohort_id": cohort_id,
                "input_revision": input_revision,
                "action_execution_ids": action_execution_ids,
                "observation_index": observation_index,
            }
            executions.append(execution)
        result = {
            "schema_version": CAMPAIGN_ACTION_COHORT_SCHEMA,
            "status": "completed",
            "cohort_id": cohort_id,
            "input_revision": input_revision,
            "requested_action_kinds": [kind.value for kind in normalized_kinds],
            "selected_action_ids": [action.action_id for action in actions],
            "action_execution_ids": action_execution_ids,
            "decisions": decisions,
            "max_in_flight_action_count": len(prepared),
            "executions": executions,
            "semantics": {
                "all_actions_bound_to_one_input_revision": True,
                "reservations_precede_handler_start": True,
                "handler_failure_does_not_cancel_peer": True,
                "observation_order_follows_requested_kind_order": True,
                "canonical_handlers_retain_union_merge_authority": True,
            },
        }
        result["content_sha256"] = _digest(result)
        return result

    def execute(
        self,
        action: CampaignAction,
        *,
        decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action.input_revision != self.kernel.state.graph_revision:
            return {
                "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
                "status": "stale",
                "action": action.to_dict(),
                "decision": dict(decision or {}),
                "cache_hit": False,
                "reasons": ["campaign_action_input_revision_stale"],
            }
        cached = self._load_cached(action)
        if cached is not None:
            return {
                "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
                "status": str(cached.get("status") or "completed"),
                "action": action.to_dict(),
                "decision": dict(decision or {}),
                "outcome": cached,
                "cache_hit": True,
            }
        handler = self.handlers.get(action.kind)
        if handler is None:
            raise CampaignActionRuntimeError(
                f"campaign_action_handler_missing:{action.kind.value}"
            )
        resource_reservation = self._reserve_action(action)
        return self._execute_reserved(
            action,
            decision=decision,
            resource_reservation=resource_reservation,
        )

    def _reserve_action(self, action: CampaignAction) -> dict[str, Any]:
        action_row = action.to_dict()
        native_resource_units = (
            1
            if action.resource_class
            in {"native_search_target", "native_search_frontier"}
            else 0
        )
        self.kernel.reserve_task(
            task_id=action.task_id,
            kind="other",
            idempotency_key=f"{action.idempotency_key}:reserve",
            input_revision=action.input_revision,
            uses_model=False,
            resource_class=action.resource_class,
            resource_units=native_resource_units,
            metadata={
                "campaign_action_id": action.action_id,
                "campaign_action_execution_id": action.execution_id,
                "campaign_action_sha256": action_row["content_sha256"],
                "delegated_resource_class": action.resource_class,
                "producer": action.producer,
            },
        )
        resource_reservation = dict(
            dict(
                self.kernel.state.in_flight_tasks.get(action.task_id) or {}
            ).get("resource_reservation")
            or {}
        )
        return resource_reservation

    def _execute_reserved(
        self,
        action: CampaignAction,
        *,
        decision: Mapping[str, Any] | None,
        resource_reservation: Mapping[str, Any],
    ) -> dict[str, Any]:
        handler = self.handlers.get(action.kind)
        if handler is None:
            raise CampaignActionRuntimeError(
                f"campaign_action_handler_missing:{action.kind.value}"
            )
        action_row = action.to_dict()
        started = time.perf_counter()
        failure_reasons: list[str] = []
        try:
            raw_result = dict(handler(action) or {})
            status = str(raw_result.get("status") or "completed")
            if status in {"failed", "error", "timed_out", "timeout"}:
                failure_reasons.extend(
                    str(value)
                    for value in raw_result.get("reasons")
                    or raw_result.get("failure_reasons")
                    or [f"handler_status:{status}"]
                    if str(value)
                )
        except Exception as exc:  # action failures remain replayable
            status = "failed"
            raw_result = {}
            failure_reasons.append(
                f"campaign_action_handler_error:{type(exc).__name__}:{str(exc)[:500]}"
            )
        elapsed_s = round(max(0.0, time.perf_counter() - started), 6)
        outcome = {
            "schema_version": CAMPAIGN_ACTION_OUTCOME_SCHEMA,
            "status": status,
            "action_execution_id": action.execution_id,
            "action_sha256": action_row["content_sha256"],
            "input_revision": action.input_revision,
            "output_revision": self.kernel.state.graph_revision,
            "handler_result": _json_result(raw_result),
            "failure_reasons": sorted(set(failure_reasons)),
            "elapsed_s": elapsed_s,
            "resource_reservation": dict(resource_reservation),
            "semantics": {
                "handler_child_tasks_own_resource_accounting": True,
                "outcome_grants_no_scientific_authority": True,
                "canonical_ingestion_remains_the_only_chemistry_write_path": True,
            },
        }
        outcome["content_sha256"] = _digest(outcome)
        ref = self.kernel.artifacts.put_json(
            outcome,
            logical_name=f"{action.task_id}.json",
            producer="autoplanner.unified_campaign_runtime",
        )
        self.kernel.settle_task(
            task_id=action.task_id,
            idempotency_key=f"{action.idempotency_key}:settle",
            status=status,
            output_sha256=ref.sha256,
            failure_reasons=failure_reasons,
            elapsed_s=elapsed_s,
        )
        pointer_name = self._pointer_name(action)
        self.kernel.artifacts.write_pointer(
            pointer_name,
            ref,
            metadata={
                "action_execution_id": action.execution_id,
                "action_sha256": action_row["content_sha256"],
                "input_revision": action.input_revision,
                "output_revision": self.kernel.state.graph_revision,
            },
        )
        self.kernel.index.index_artifact(
            run_id=self.kernel.spec.run_id,
            artifact_id=action.task_id,
            ref=ref,
            revision=self.kernel.state.graph_revision,
            authority_scope="campaign_action_execution_receipt",
        )
        return {
            "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
            "status": status,
            "action": action_row,
            "decision": dict(decision or {}),
            "outcome": outcome,
            "outcome_ref": ref.to_dict(),
            "cache_hit": False,
        }

    def _load_cached(self, action: CampaignAction) -> dict[str, Any] | None:
        try:
            ref, pointer = self.kernel.artifacts.load_pointer(
                self._pointer_name(action)
            )
        except ArtifactReferenceError:
            return None
        metadata = dict(pointer.get("metadata") or {})
        action_sha256 = action.to_dict()["content_sha256"]
        if (
            metadata.get("action_execution_id") != action.execution_id
            or metadata.get("action_sha256") != action_sha256
        ):
            raise CampaignActionRuntimeError("campaign_action_pointer_binding_invalid")
        value = self.kernel.artifacts.read_json(ref)
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != CAMPAIGN_ACTION_OUTCOME_SCHEMA
            or value.get("action_execution_id") != action.execution_id
            or value.get("action_sha256") != action_sha256
        ):
            raise CampaignActionRuntimeError("campaign_action_outcome_binding_invalid")
        expected = _digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
        if value.get("content_sha256") != expected:
            raise CampaignActionRuntimeError("campaign_action_outcome_digest_invalid")
        return dict(value)

    def _pointer_name(self, action: CampaignAction) -> str:
        binding_digest = hashlib.sha256(
            (
                self.kernel.spec.run_id
                + "\0"
                + action.execution_id
            ).encode("utf-8")
        ).hexdigest()
        return f"ca/{binding_digest[:32]}"


def _json_result(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _execution_gained(execution: Mapping[str, Any]) -> bool:
    outcome = dict(execution.get("outcome") or {})
    handler_result = dict(outcome.get("handler_result") or {})
    explicit_gain = bool(
        handler_result.get("changed") is True
        or handler_result.get("material_events")
        or handler_result.get("plan")
        or int(handler_result.get("proposal_count") or 0) > 0
        or int(handler_result.get("candidate_count") or 0) > 0
    )
    if explicit_gain:
        return True
    if execution.get("cohort"):
        return False
    return int(outcome.get("output_revision") or 0) != int(
        outcome.get("input_revision") or 0
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CampaignActionHandler",
    "CampaignActionExecutionObserver",
    "CampaignActionRuntime",
    "CampaignActionRuntimeError",
    "CampaignActionStateProvider",
    "CAMPAIGN_ACTION_COHORT_SCHEMA",
    "CAMPAIGN_ANYTIME_LOOP_SCHEMA",
]
