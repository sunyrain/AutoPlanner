"""Run revision-bound campaign actions through the single RunKernel ledger."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import time
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.application.action_convergence import (
    compile_action_convergence_ledger,
    no_gain_binding_map,
)
from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.action_service_policy import action_class_for_kind
from cascade_planner.application.campaign_actions import (
    ACTION_RESULT_SCHEMA,
    ActionResult,
    CampaignAction,
    CampaignActionKind,
    action_task_kind,
    bind_scheduled_action,
    legacy_campaign_action_sha256,
)
from cascade_planner.application.run_kernel import RunKernel, RunKernelBudgetError
from cascade_planner.runtime.artifact_store import ArtifactReferenceError


LEGACY_CAMPAIGN_ACTION_OUTCOME_SCHEMA = "campaign_action_outcome.v1"
CAMPAIGN_ACTION_OUTCOME_SCHEMA = ACTION_RESULT_SCHEMA
CAMPAIGN_ACTION_RESOURCE_ACCOUNTING_SCHEMA = (
    "campaign_action_resource_accounting.v1"
)
CAMPAIGN_ACTION_EXECUTION_SCHEMA = "campaign_action_execution.v1"
CAMPAIGN_ACTION_COHORT_SCHEMA = "campaign_action_concurrent_cohort.v1"
CAMPAIGN_ANYTIME_LOOP_SCHEMA = "campaign_anytime_action_loop.v1"
CAMPAIGN_UNEXECUTED_ACTION_SET_SCHEMA = "campaign_unexecuted_action_set.v1"

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

    def action_service_history(self) -> tuple[str, ...]:
        """Rebuild Action-class service order from RunKernel authority."""

        history: list[str] = []
        for event in self.kernel.task_reservation_history():
            payload = dict(event.get("payload") or {})
            metadata = dict(payload.get("metadata") or {})
            kind = str(metadata.get("campaign_action_kind") or "")
            if (
                kind
                and metadata.get("campaign_action_id")
                and metadata.get("campaign_action_execution_id")
            ):
                history.append(kind)
        return tuple(history)

    def action_execution_history(self) -> tuple[dict[str, Any], ...]:
        """Rebuild settled Action outcomes and in-flight attempts in event order."""

        history: list[dict[str, Any]] = []
        for event in self.kernel.task_reservation_history():
            payload = dict(event.get("payload") or {})
            metadata = dict(payload.get("metadata") or {})
            task_id = str(payload.get("task_id") or "")
            action_id = str(metadata.get("campaign_action_id") or "")
            execution_id = str(
                metadata.get("campaign_action_execution_id") or ""
            )
            action_sha256 = str(metadata.get("campaign_action_sha256") or "")
            if not task_id or not action_id or not execution_id or not action_sha256:
                continue
            row: dict[str, Any] = {
                "reservation_sequence": int(event.get("sequence") or 0),
                "task_id": task_id,
                "action_id": action_id,
                "action_execution_id": execution_id,
                "action_kind": str(
                    metadata.get("campaign_action_kind") or ""
                ),
                "input_revision": int(payload.get("input_revision") or 0),
                "opportunity_sha256": str(
                    metadata.get("campaign_action_opportunity_sha256") or ""
                ),
                "same_revision_cohort": metadata.get(
                    "campaign_action_same_revision_cohort"
                )
                is True,
                "settled": False,
            }
            try:
                outcome = self._load_bound_outcome(
                    execution_id=execution_id,
                    accepted_action_sha256={action_sha256},
                )
            except ArtifactReferenceError:
                lifecycle = self.kernel.task_lifecycle(task_id)
                if lifecycle.get("status") == "settled":
                    raise CampaignActionRuntimeError(
                        "campaign_action_history_outcome_pointer_missing"
                    ) from None
            else:
                row.update(
                    {
                        "settled": True,
                        "status": str(outcome.get("status") or "completed"),
                        "output_revision": int(
                            outcome.get("output_revision") or 0
                        ),
                        "gained": _outcome_gained(
                            outcome,
                            concurrent_cohort=row["same_revision_cohort"],
                        ),
                        "outcome_sha256": str(
                            outcome.get("content_sha256") or ""
                        ),
                    }
                )
            history.append(row)
        return tuple(history)

    def action_convergence_ledger(
        self,
        *,
        current_graph_revision: int | None = None,
    ) -> dict[str, Any]:
        """Project cross-slice convergence from durable Action history."""

        revision = (
            self.kernel.state.graph_revision
            if current_graph_revision is None
            else int(current_graph_revision)
        )
        return compile_action_convergence_ledger(
            self.action_execution_history(),
            current_graph_revision=revision,
        )

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
            prior_action_kinds=self.action_service_history(),
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
        try:
            return self.execute(action, decision=decision)
        except RunKernelBudgetError as exc:
            reasons = _terminal_budget_reasons(exc)
            if not reasons:
                raise
            return {
                "schema_version": CAMPAIGN_ACTION_EXECUTION_SCHEMA,
                "status": "budget_exhausted",
                "decision": decision,
                "cache_hit": False,
                "reasons": list(reasons),
                "semantics": {
                    "operator_budget_terminal": True,
                    "no_action_reserved_after_budget_exhaustion": True,
                    "no_second_queue_created": True,
                },
            }

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
        termination = "action_limit"
        termination_reasons: list[str] = []
        action_limit = max(1, int(max_actions))
        no_gain_limit = max(1, int(max_consecutive_no_gain))
        initial_revision = self.kernel.state.graph_revision
        initial_convergence = self.action_convergence_ledger(
            current_graph_revision=initial_revision
        )
        attempted_by_revision: dict[int, set[str]] = {
            initial_revision: {
                str(value)
                for value in initial_convergence.get(
                    "attempted_action_ids_at_current_revision"
                )
                or []
                if str(value)
            }
        }
        no_gain_bindings = no_gain_binding_map(initial_convergence)
        consecutive_no_gain = int(
            initial_convergence.get("consecutive_no_gain") or 0
        )
        convergence_resumed_from_history = consecutive_no_gain > 0
        normalized_start_kinds = tuple(
            (
                kind
                if isinstance(kind, CampaignActionKind)
                else CampaignActionKind(str(kind))
            )
            for kind in concurrent_start_kinds
        )
        start_cohort: dict[str, Any] = {}
        initial_kernel_termination = _kernel_loop_termination(self.kernel.state)
        if initial_kernel_termination is not None:
            termination, termination_reasons = initial_kernel_termination
            action_limit = 0
        elif consecutive_no_gain >= no_gain_limit:
            termination = "converged_low_marginal_gain"
            action_limit = 0
        if len(normalized_start_kinds) >= 2 and action_limit >= 2:
            start_opportunity_set = opportunity_provider()
            start_attempted = attempted_by_revision.setdefault(
                initial_revision,
                set(),
            )
            start_no_gain_excluded = _matching_no_gain_action_ids(
                start_opportunity_set,
                no_gain_bindings,
            )
            try:
                start_cohort = self.execute_concurrent_cohort(
                    start_opportunity_set,
                    action_kinds=normalized_start_kinds,
                    milestones=milestones_provider(),
                    resource_availability=resource_availability_provider(),
                    excluded_action_ids=tuple(
                        sorted(
                            globally_excluded
                            | start_attempted
                            | start_no_gain_excluded
                        )
                    ),
                )
            except RunKernelBudgetError as exc:
                reasons = _terminal_budget_reasons(exc)
                if not reasons:
                    raise
                termination = "budget_exhausted"
                termination_reasons = list(reasons)
                action_limit = 0
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
                elif action_id:
                    no_gain_bindings.pop(action_id, None)
                consecutive_no_gain = 0 if gained else consecutive_no_gain + 1

        for index in range(len(executions) + 1, action_limit + 1):
            kernel_termination = _kernel_loop_termination(self.kernel.state)
            if kernel_termination is not None:
                termination, termination_reasons = kernel_termination
                break
            input_revision = self.kernel.state.graph_revision
            attempted = attempted_by_revision.setdefault(input_revision, set())
            opportunity_set = opportunity_provider()
            no_gain_excluded = _matching_no_gain_action_ids(
                opportunity_set,
                no_gain_bindings,
            )
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
            if execution.get("status") == "budget_exhausted":
                termination = "budget_exhausted"
                termination_reasons = [
                    str(value)
                    for value in execution.get("reasons") or []
                    if str(value)
                ]
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
            elif action_id:
                no_gain_bindings.pop(action_id, None)
            consecutive_no_gain = 0 if gained else consecutive_no_gain + 1
            if consecutive_no_gain >= no_gain_limit:
                termination = "converged_low_marginal_gain"
                break
        kernel_stop_decision: dict[str, Any] = {}
        if termination == "budget_exhausted":
            terminal_revision = self.kernel.state.revision
            if not self.kernel.state.terminal:
                self.kernel.transition(
                    "budget_exhausted",
                    idempotency_key=(
                        "campaign:anytime:global-budget-terminal:"
                        f"{terminal_revision}"
                    ),
                    reasons=termination_reasons,
                )
            stop_decision = self.kernel.decide_stop()
            if stop_decision.decision != "budget_exhausted":
                raise CampaignActionRuntimeError(
                    "campaign_anytime_budget_terminal_state_mismatch"
                )
            kernel_stop_decision = stop_decision.to_dict()
        elif self.kernel.state.status != "running":
            kernel_stop_decision = self.kernel.decide_stop().to_dict()
        final_revision = self.kernel.state.graph_revision
        final_convergence = self.action_convergence_ledger(
            current_graph_revision=final_revision
        )
        durable_no_gain_bindings = no_gain_binding_map(final_convergence)
        durable_attempted = {
            str(value)
            for value in final_convergence.get(
                "attempted_action_ids_at_current_revision"
            )
            or []
            if str(value)
        }
        final_opportunity_set = opportunity_provider()
        final_decision = schedule_next_action(
            final_opportunity_set,
            milestones=milestones_provider(),
            resource_availability=resource_availability_provider(),
            available_action_kinds=tuple(
                sorted(kind.value for kind in self.handlers)
            ),
            prior_action_kinds=self.action_service_history(),
            policy=self.scheduler_policy,
        )
        unexecuted_actions = _unexecuted_action_set(
            decision=final_decision,
            executions=executions,
            final_revision=final_revision,
            attempted_action_ids=durable_attempted,
            no_gain_bindings=durable_no_gain_bindings,
            globally_excluded=globally_excluded,
            termination=termination,
            termination_reasons=termination_reasons,
        )
        result = {
            "schema_version": CAMPAIGN_ANYTIME_LOOP_SCHEMA,
            "termination": termination,
            "termination_reasons": termination_reasons,
            "execution_count": len(executions),
            "consecutive_no_gain": int(
                final_convergence.get("consecutive_no_gain") or 0
            ),
            "no_gain_binding_count": len(durable_no_gain_bindings),
            "initial_convergence_ledger_sha256": str(
                initial_convergence.get("content_sha256") or ""
            ),
            "convergence_ledger": final_convergence,
            "start_cohort": start_cohort,
            "executions": executions,
            "kernel_stop_decision": kernel_stop_decision,
            "unexecuted_actions": unexecuted_actions,
            "action_class_service": dict(
                final_decision.get("action_class_service") or {}
            ),
            "final_graph_revision": final_revision,
            "semantics": {
                "single_scheduler_loop": True,
                "scheduler_policy": self.scheduler_policy,
                "latest_revision_recompiled_each_iteration": True,
                "same_revision_start_cohort_is_non_blocking": (
                    start_cohort.get("status") == "completed"
                ),
                "cohort_failures_do_not_cancel_peers": True,
                "cohort_observation_order_is_stable": True,
                "unexecuted_actions_have_explicit_reasons": True,
                "B4_and_B5_do_not_stop_the_loop": True,
                "no_action_and_low_gain_converge_finitely": True,
                "cross_slice_no_gain_state_replays_from_action_outcomes": True,
                "convergence_resumed_from_history": (
                    convergence_resumed_from_history
                ),
                "global_budget_exhaustion_is_a_normal_terminal": (
                    termination == "budget_exhausted"
                ),
                "global_budget_terminal_is_persisted_before_return": (
                    termination != "budget_exhausted"
                    or self.kernel.state.status == "budget_exhausted"
                ),
                "action_class_service_replays_from_run_kernel_events": True,
                "blocked_class_capacity_is_borrowed_without_a_second_queue": True,
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
        service_history = list(self.action_service_history())
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
                prior_action_kinds=tuple(service_history),
                policy=self.scheduler_policy,
            )
            if not decision.get("selected_action_id"):
                continue
            action = bind_scheduled_action(decision, input_revision=input_revision)
            decisions.append(decision)
            actions.append(action)
            selected_action_ids.add(action.action_id)
            service_history.append(action.kind.value)
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
            prepared.append(
                (
                    action,
                    decision,
                    self._reserve_action(
                        action,
                        decision=decision,
                        same_revision_cohort=True,
                    ),
                )
            )

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
        resource_reservation = self._reserve_action(action, decision=decision)
        return self._execute_reserved(
            action,
            decision=decision,
            resource_reservation=resource_reservation,
        )

    def _reserve_action(
        self,
        action: CampaignAction,
        *,
        decision: Mapping[str, Any] | None,
        same_revision_cohort: bool = False,
    ) -> dict[str, Any]:
        action_row = action.to_dict()
        lifecycle = self.kernel.task_lifecycle(action.task_id)
        if lifecycle["status"] == "settled":
            raise CampaignActionRuntimeError(
                "campaign_action_settled_without_outcome_pointer"
            )
        if lifecycle["status"] == "in_flight":
            reservation = dict(
                dict(lifecycle.get("reservation") or {}).get("payload") or {}
            )
            metadata = dict(reservation.get("metadata") or {})
            accepted_sha256 = {
                action_row["content_sha256"],
                legacy_campaign_action_sha256(action),
            }
            if (
                metadata.get("campaign_action_id") != action.action_id
                or metadata.get("campaign_action_execution_id")
                != action.execution_id
                or metadata.get("campaign_action_sha256") not in accepted_sha256
            ):
                raise CampaignActionRuntimeError(
                    "campaign_action_in_flight_binding_invalid"
                )
            return dict(reservation.get("resource_reservation") or {})
        native_resource_units = (
            1
            if action.resource_class
            in {"native_search_target", "native_search_frontier"}
            else 0
        )
        action_class_service = dict(
            dict(decision or {}).get("action_class_service") or {}
        )
        service_metadata = (
            {
                "action_class_service_ordinal": int(
                    action_class_service.get("next_action_ordinal") or 0
                ),
                "action_class_service_sha256": str(
                    action_class_service.get("content_sha256") or ""
                ),
                "minimum_service_guarantee_applied": (
                    action_class_service.get(
                        "minimum_service_guarantee_applied"
                    )
                    is True
                ),
            }
            if action_class_service
            else {}
        )
        self.kernel.reserve_task(
            task_id=action.task_id,
            kind=action_task_kind(action.resource_class),
            idempotency_key=f"{action.idempotency_key}:reserve",
            input_revision=action.input_revision,
            uses_model=False,
            resource_class=action.resource_class,
            resource_units=native_resource_units,
            metadata={
                "campaign_action_id": action.action_id,
                "campaign_action_execution_id": action.execution_id,
                "campaign_action_kind": action.kind.value,
                "campaign_action_class": action_class_for_kind(
                    action.kind.value
                ),
                "campaign_action_sha256": action_row["content_sha256"],
                "campaign_action_opportunity_sha256": (
                    action.opportunity_sha256
                ),
                "campaign_action_opportunity_set_sha256": (
                    action.opportunity_set_sha256
                ),
                "campaign_action_same_revision_cohort": bool(
                    same_revision_cohort
                ),
                "delegated_resource_class": action.resource_class,
                "expected_resources": dict(action.expected_resources),
                "expected_resources_sha256": str(
                    action.expected_resources.get("content_sha256") or ""
                ),
                "producer": action.producer,
                **service_metadata,
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
        with self.kernel.action_resource_scope(
            action_execution_id=action.execution_id,
            expected_resources_sha256=str(
                action.expected_resources.get("content_sha256") or ""
            ),
        ):
            try:
                raw_result = dict(handler(action) or {})
                status = str(raw_result.get("status") or "completed")
                if _is_failure_settlement(status):
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
        self._settle_owned_children(
            action,
            status=status,
            failure_reasons=failure_reasons,
        )
        elapsed_s = round(max(0.0, time.perf_counter() - started), 6)
        actual_resources = self.kernel.action_resource_usage(
            action.execution_id,
            pending_task_id=action.task_id,
            pending_status=status,
            pending_elapsed_s=elapsed_s,
        )
        resource_accounting = _resource_accounting(
            expected=action.expected_resources,
            actual=actual_resources,
        )
        output_revision = self.kernel.state.graph_revision
        graph_revision_delta = max(0, output_revision - action.input_revision)
        outcome = ActionResult(
            action_execution_id=action.execution_id,
            action_sha256=action_row["content_sha256"],
            status=status,
            input_revision=action.input_revision,
            output_revision=output_revision,
            immutable_artifact_refs=_immutable_artifact_refs(raw_result),
            actual_resources=actual_resources,
            resource_accounting=resource_accounting,
            resource_reservation=dict(resource_reservation),
            material_events=_material_events(raw_result),
            candidate_delta=_candidate_delta(raw_result),
            fact_delta={
                "graph_revision_delta": graph_revision_delta,
                "changed": graph_revision_delta > 0,
                "handler_reported_changed": raw_result.get("changed") is True,
                "authority": "run_kernel_canonical_graph_revision",
            },
            failure_type=_failure_type(status, failure_reasons),
            failure_reasons=tuple(sorted(set(failure_reasons))),
            elapsed_s=elapsed_s,
            handler_result=_json_result(raw_result),
        ).to_dict()
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
            resource_usage=actual_resources,
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

    def _settle_owned_children(
        self,
        action: CampaignAction,
        *,
        status: str,
        failure_reasons: Iterable[str],
    ) -> None:
        """Release leaked child reservations after a terminal failed Action."""

        if not _is_failure_settlement(status):
            return
        normalized = str(status or "failed").casefold()
        child_status = {
            "canceled": "cancelled",
            "cancelled": "cancelled",
            "timed_out": "timeout",
        }.get(normalized, normalized)
        reason_rows = tuple(failure_reasons) or (
            f"campaign_action_parent_status:{normalized}",
        )
        for task_id, reservation in sorted(self.kernel.state.in_flight_tasks.items()):
            if task_id == action.task_id:
                continue
            metadata = dict(reservation.get("metadata") or {})
            if metadata.get("campaign_action_execution_id") != action.execution_id:
                continue
            task_digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
            self.kernel.settle_task(
                task_id=task_id,
                idempotency_key=(
                    f"{action.idempotency_key}:release-child:{task_digest}"
                ),
                status=child_status,
                failure_reasons=reason_rows,
            )

    def _load_cached(self, action: CampaignAction) -> dict[str, Any] | None:
        action_sha256 = action.to_dict()["content_sha256"]
        try:
            return self._load_bound_outcome(
                execution_id=action.execution_id,
                accepted_action_sha256={
                    action_sha256,
                    legacy_campaign_action_sha256(action),
                },
            )
        except ArtifactReferenceError:
            return None

    def _load_bound_outcome(
        self,
        *,
        execution_id: str,
        accepted_action_sha256: set[str],
    ) -> dict[str, Any]:
        ref, pointer = self.kernel.artifacts.load_pointer(
            self._pointer_name_for_execution(execution_id)
        )
        metadata = dict(pointer.get("metadata") or {})
        if (
            metadata.get("action_execution_id") != execution_id
            or metadata.get("action_sha256") not in accepted_action_sha256
        ):
            raise CampaignActionRuntimeError("campaign_action_pointer_binding_invalid")
        cached_action_sha256 = str(metadata.get("action_sha256") or "")
        value = self.kernel.artifacts.read_json(ref)
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version")
            not in {
                CAMPAIGN_ACTION_OUTCOME_SCHEMA,
                LEGACY_CAMPAIGN_ACTION_OUTCOME_SCHEMA,
            }
            or value.get("action_execution_id") != execution_id
            or value.get("action_sha256") != cached_action_sha256
        ):
            raise CampaignActionRuntimeError("campaign_action_outcome_binding_invalid")
        expected = _digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
        if value.get("content_sha256") != expected:
            raise CampaignActionRuntimeError("campaign_action_outcome_digest_invalid")
        return dict(value)

    def _pointer_name(self, action: CampaignAction) -> str:
        return self._pointer_name_for_execution(action.execution_id)

    def _pointer_name_for_execution(self, execution_id: str) -> str:
        binding_digest = hashlib.sha256(
            (
                self.kernel.spec.run_id
                + "\0"
                + str(execution_id)
            ).encode("utf-8")
        ).hexdigest()
        return f"ca/{binding_digest[:32]}"


def _resource_accounting(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    estimated = dict(expected.get("estimated") or {})
    model_usage = dict(actual.get("model_usage") or {})
    native_search = dict(actual.get("native_search_units") or {})
    observed = {
        "task_counts": dict(actual.get("task_counts") or {}),
        "total_tasks": int(actual.get("settled_task_count") or 0),
        "native_search_units": int(native_search.get("total") or 0),
        "model_invocations": int(model_usage.get("model_invocations") or 0),
        "visual_invocations": int(model_usage.get("visual_invocations") or 0),
    }
    dimensions = (
        "total_tasks",
        "native_search_units",
        "model_invocations",
        "visual_invocations",
    )
    result = {
        "schema_version": CAMPAIGN_ACTION_RESOURCE_ACCOUNTING_SCHEMA,
        "expected": dict(expected),
        "actual": dict(actual),
        "variance": {
            **{
                key: int(observed[key]) - int(estimated.get(key) or 0)
                for key in dimensions
            },
            "task_counts": {
                key: int(observed["task_counts"].get(key) or 0)
                - int(dict(estimated.get("task_counts") or {}).get(key) or 0)
                for key in sorted(
                    set(observed["task_counts"])
                    | set(dict(estimated.get("task_counts") or {}))
                )
            },
        },
        "semantics": {
            "expected_is_declared_before_reservation": True,
            "actual_is_derived_from_action_bound_run_events": True,
            "variance_does_not_change_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _candidate_delta(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_count": _reported_count(
            result,
            count_key="proposal_count",
            collection_keys=("proposals",),
        ),
        "candidate_count": _reported_count(
            result,
            count_key="candidate_count",
            collection_keys=("candidates",),
        ),
        "accepted_count": int(
            result.get("accepted_count")
            or result.get("accepted_expansion_count")
            or 0
        ),
    }


def _reported_count(
    result: Mapping[str, Any],
    *,
    count_key: str,
    collection_keys: Iterable[str],
) -> int:
    if result.get(count_key) is not None:
        return max(0, int(result.get(count_key) or 0))
    return sum(
        len(value)
        for key in collection_keys
        for value in (result.get(key),)
        if isinstance(value, (list, tuple))
    )


def _material_events(result: Mapping[str, Any]) -> tuple[Any, ...]:
    value = result.get("material_events")
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(_json_result(item) for item in values)


def _failure_type(status: str, reasons: Iterable[str]) -> str:
    normalized = str(status or "").casefold()
    reason_text = " ".join(str(value).casefold() for value in reasons)
    if normalized in {"timeout", "timed_out"} or "timeout" in reason_text:
        return "timeout"
    if normalized in {"cancelled", "canceled"} or "cancel" in reason_text:
        return "cancelled"
    if normalized in {
        "partial",
        "partially_completed",
        "completed_with_failures",
    } or "partial" in reason_text:
        return "partial_failure"
    if "budget" in reason_text or "resource" in reason_text:
        return "resource_exhausted"
    if normalized in {"awaiting_external_result", "pending"}:
        return "pending_external"
    if normalized in {"failed", "error"}:
        return "handler_failure"
    return ""


def _is_failure_settlement(status: str) -> bool:
    return str(status or "").casefold() in {
        "failed",
        "error",
        "timed_out",
        "timeout",
        "cancelled",
        "canceled",
        "partial",
        "partially_completed",
        "completed_with_failures",
    }


def _kernel_loop_termination(state: Any) -> tuple[str, list[str]] | None:
    status = str(getattr(state, "status", "") or "")
    reasons = [
        str(value)
        for value in getattr(state, "failure_reasons", ()) or ()
        if str(value)
    ]
    if status == "cancelled":
        return "user_cancelled", reasons or ["explicit_user_cancelled"]
    if status == "failed":
        return "unrecoverable_error", reasons or ["run_failed"]
    if status == "paused":
        return "paused", reasons or ["operator_paused"]
    if status in {"completed", "unresolved"}:
        return "kernel_terminal", reasons or [f"run_already_{status}"]
    return None


def _immutable_artifact_refs(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    refs: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            row = dict(value)
            sha256 = str(row.get("sha256") or "")
            if (
                len(sha256) == 64
                and "size_bytes" in row
                and "media_type" in row
            ):
                refs.setdefault(sha256, row)
                return
            for child in row.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(result)
    return tuple(refs[key] for key in sorted(refs))


def _unexecuted_action_set(
    *,
    decision: Mapping[str, Any],
    executions: Iterable[Mapping[str, Any]],
    final_revision: int,
    attempted_action_ids: Iterable[str],
    no_gain_bindings: Mapping[str, str],
    globally_excluded: Iterable[str],
    termination: str,
    termination_reasons: Iterable[str],
) -> dict[str, Any]:
    executed_at_final_revision = {
        str(action.get("action_id") or "")
        for execution in executions
        for action in (dict(execution.get("action") or {}),)
        if int(action.get("input_revision") or 0) == int(final_revision)
        and str(action.get("action_id") or "")
    }
    attempted = {str(value) for value in attempted_action_ids if str(value)}
    excluded = {str(value) for value in globally_excluded if str(value)}
    terminal_reasons = [str(value) for value in termination_reasons if str(value)]
    termination_reason = {
        "action_limit": "action_limit_reached",
        "converged_low_marginal_gain": "low_marginal_gain_convergence",
        "no_action": "scheduler_no_action",
        "budget_exhausted": "budget_exhausted",
        "user_cancelled": "explicit_user_cancelled",
        "unrecoverable_error": "unrecoverable_run_error",
        "paused": "operator_paused",
        "kernel_terminal": "kernel_already_terminal",
    }.get(str(termination), f"loop_terminated:{termination}")
    actions = []
    for raw in decision.get("candidates") or []:
        candidate = dict(raw)
        action_id = str(candidate.get("action_id") or "")
        if not action_id or action_id in executed_at_final_revision:
            continue
        reasons = [
            str(value)
            for value in candidate.get("blocked_reasons") or []
            if str(value)
        ]
        if action_id in excluded:
            reasons.append("excluded_by_caller")
        if action_id in attempted:
            reasons.append("already_attempted_at_current_revision")
        if no_gain_bindings.get(action_id) == str(
            candidate.get("content_sha256") or ""
        ):
            reasons.append("unchanged_after_no_gain")
        if not reasons:
            reasons.extend(terminal_reasons or [termination_reason])
            if termination in {
                "user_cancelled",
                "unrecoverable_error",
                "paused",
                "kernel_terminal",
            }:
                reasons.append(termination_reason)
        actions.append(
            {
                "action_id": action_id,
                "kind": str(candidate.get("kind") or ""),
                "resource_class": str(candidate.get("resource_class") or ""),
                "subject_ids": list(candidate.get("subject_ids") or []),
                "route_family_ids": list(candidate.get("route_family_ids") or []),
                "opportunity_sha256": str(
                    candidate.get("content_sha256") or ""
                ),
                "reasons": sorted(set(reasons)),
            }
        )
    result = {
        "schema_version": CAMPAIGN_UNEXECUTED_ACTION_SET_SCHEMA,
        "final_revision": int(final_revision),
        "termination": str(termination),
        "action_count": len(actions),
        "actions": actions,
        "semantics": {
            "all_rows_are_non_executed_at_final_revision": True,
            "reasons_are_scheduler_or_loop_terminal_facts": True,
            "grants_no_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


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


def _matching_no_gain_action_ids(
    opportunity_set: Mapping[str, Any],
    no_gain_bindings: Mapping[str, str],
) -> set[str]:
    current_opportunity_sha256 = {
        str(row.get("action_id") or ""): str(
            row.get("content_sha256") or ""
        )
        for raw in opportunity_set.get("actions") or []
        if isinstance(raw, Mapping)
        for row in (dict(raw),)
        if str(row.get("action_id") or "")
    }
    return {
        action_id
        for action_id, opportunity_sha256 in no_gain_bindings.items()
        if current_opportunity_sha256.get(action_id) == opportunity_sha256
    }


def _execution_gained(execution: Mapping[str, Any]) -> bool:
    outcome = dict(execution.get("outcome") or {})
    return _outcome_gained(
        outcome,
        concurrent_cohort=bool(execution.get("cohort")),
    )


def _outcome_gained(
    outcome: Mapping[str, Any],
    *,
    concurrent_cohort: bool,
) -> bool:
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
    if concurrent_cohort:
        return False
    return int(outcome.get("output_revision") or 0) != int(
        outcome.get("input_revision") or 0
    )


def _terminal_budget_reasons(exc: RunKernelBudgetError) -> tuple[str, ...]:
    reasons = tuple(
        sorted(
            {
                value.strip()
                for value in str(exc).split(";")
                if value.strip()
            }
        )
    )
    global_terminal_reasons = {
        "run_total_task_budget_exhausted",
        "run_wall_time_budget_exhausted",
    }
    if not global_terminal_reasons.intersection(reasons):
        return ()
    return reasons


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
