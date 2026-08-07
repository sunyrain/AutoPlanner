"""Deterministic target-blind scheduling over campaign action opportunities."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ACTION_SCHEDULE_DECISION_SCHEMA = "campaign_action_schedule_decision.v1"
ACTION_SCHEDULER_POLICIES = frozenset({"adaptive", "round_robin"})

_ROUTE_ACTIONS = frozenset(
    {
        "host_materialize",
        "stock_audit",
        "chemenzy_target_expand",
        "chemenzy_frontier_expand",
        "codex_global_architecture",
        "codex_global_replan",
        "program_discover",
        "program_review",
        "program_admit",
        "recompute_route_closure",
    }
)
_PROOF_ACTIONS = frozenset(
    {
        "reaction_validate",
        "acquire_exact_evidence",
        "bind_exact_evidence",
        "condition_enrich",
        "resolve_conflict",
        "program_validate",
        "experiment_feedback_ingest",
    }
)
_KIND_ORDER = {
    "resolve_conflict": 0,
    "host_materialize": 1,
    "reaction_validate": 2,
    "stock_audit": 3,
    "acquire_exact_evidence": 4,
    "bind_exact_evidence": 5,
    "condition_enrich": 6,
    "chemenzy_target_expand": 7,
    "codex_global_architecture": 8,
    "chemenzy_frontier_expand": 9,
    "codex_global_replan": 10,
    "program_discover": 11,
    "program_review": 12,
    "program_admit": 13,
    "program_validate": 14,
    "experiment_feedback_ingest": 15,
    "recompute_route_closure": 16,
}


def schedule_next_action(
    opportunity_set: Mapping[str, Any],
    *,
    milestones: Mapping[str, Any] | None = None,
    resource_availability: Mapping[str, Any] | None = None,
    in_flight_action_ids: tuple[str, ...] = (),
    available_action_kinds: tuple[str, ...] | None = None,
    policy: str = "adaptive",
    round_robin_cursor: int = 0,
) -> dict[str, Any]:
    """Rank one shared action set using only current state and resources."""

    scheduler_policy = str(policy or "adaptive")
    if scheduler_policy not in ACTION_SCHEDULER_POLICIES:
        raise ValueError(f"unsupported campaign action scheduler policy: {policy}")
    cursor = max(0, int(round_robin_cursor))

    gates = {str(key): value is True for key, value in dict(milestones or {}).items()}
    resources = {
        str(key): value is not False
        for key, value in dict(resource_availability or {}).items()
    }
    in_flight = {str(value) for value in in_flight_action_ids if str(value)}
    handler_filter_applied = available_action_kinds is not None
    available_kinds = {
        str(value) for value in (available_action_kinds or ()) if str(value)
    }
    raw_actions = [
        dict(raw)
        for raw in opportunity_set.get("actions") or []
        if isinstance(raw, Mapping)
    ]
    pending_materialization = any(
        str(row.get("kind") or "") == "host_materialize"
        for row in raw_actions
    )
    pending_materialization_routes = {
        str(route_id)
        for row in raw_actions
        if str(row.get("kind") or "") == "host_materialize"
        for route_id in row.get("route_family_ids") or []
        if str(route_id)
    }
    candidates = []
    for row in raw_actions:
        action_id = str(row.get("action_id") or "")
        kind = str(row.get("kind") or "")
        resource_class = str(row.get("resource_class") or "")
        blocked_reasons: list[str] = []
        if not action_id or not kind:
            blocked_reasons.append("action_identity_missing")
        if action_id in in_flight:
            blocked_reasons.append("action_already_in_flight")
        if handler_filter_applied and kind not in available_kinds:
            blocked_reasons.append(f"handler_unavailable:{kind}")
        if resources.get(resource_class, True) is False:
            blocked_reasons.append(f"resource_unavailable:{resource_class}")
        # A regular expansion may advertise Codex as a provider preference,
        # but that is not a global replan contract.  The target runtime
        # intentionally rejects such actions because only the bounded event
        # replan signal carries the required ``global_replan`` scope.  Keep
        # the opportunity visible for diagnostics, but never dispatch an
        # action that the handler is guaranteed to reject.
        if kind == "codex_global_replan":
            metadata = dict(row.get("metadata") or {})
            if (
                row.get("global_replan") is not True
                and metadata.get("global_replan") is not True
            ):
                blocked_reasons.append("global_replan_scope_missing")
        if pending_materialization and kind in {
            "acquire_exact_evidence",
            "bind_exact_evidence",
        }:
            blocked_reasons.append("pending_materialization_precedes_evidence")
        candidate_routes = {
            str(route_id)
            for route_id in row.get("route_family_ids") or []
            if str(route_id)
        }
        if candidate_routes & pending_materialization_routes:
            if kind == "reaction_validate":
                blocked_reasons.append(
                    "route_materialization_precedes_reaction_validation"
                )
            if kind == "stock_audit":
                blocked_reasons.append("route_materialization_precedes_stock_audit")
        base = float(row.get("base_priority") or 0.0)
        state_bonus = _state_bonus(kind, gates)
        deterministic_bonus = 30.0 if row.get("deterministic") is True else 0.0
        dependency_penalty = min(120.0, 30.0 * len(row.get("dependency_ids") or []))
        marginal_value = (
            90.0 * float(row.get("expected_route_gain") or 0.0)
            + 80.0 * float(row.get("expected_proof_gain") or 0.0)
            + 70.0 * float(row.get("expected_diversity_gain") or 0.0)
            - 65.0 * float(row.get("cost_penalty") or 0.0)
            - 85.0 * float(row.get("failure_risk_penalty") or 0.0)
        )
        total = round(
            base
            + state_bonus
            + deterministic_bonus
            + marginal_value
            - dependency_penalty,
            6,
        )
        candidates.append(
            {
                **row,
                "eligible": not blocked_reasons,
                "blocked_reasons": blocked_reasons,
                "schedule_score": total,
                "schedule_components": {
                    "base_priority": base,
                    "state_bonus": state_bonus,
                    "deterministic_bonus": deterministic_bonus,
                    "marginal_value": round(marginal_value, 6),
                    "dependency_penalty": dependency_penalty,
                },
            }
        )
    if scheduler_policy == "round_robin":
        kind_count = max(1, len(_KIND_ORDER))
        offset = cursor % kind_count

        def round_robin_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            kind_rank = _KIND_ORDER.get(str(row.get("kind") or ""), kind_count)
            cyclic_rank = (kind_rank - offset) % kind_count
            return (
                row.get("eligible") is not True,
                cyclic_rank,
                str(row.get("action_id") or ""),
            )

        ranked = sorted(candidates, key=round_robin_key)
    else:
        ranked = sorted(
            candidates,
            key=lambda row: (
                row.get("eligible") is not True,
                -float(row.get("schedule_score") or 0.0),
                _KIND_ORDER.get(str(row.get("kind") or ""), 99),
                str(row.get("action_id") or ""),
            ),
        )
    selected = next((dict(row) for row in ranked if row.get("eligible") is True), {})
    result = {
        "schema_version": ACTION_SCHEDULE_DECISION_SCHEMA,
        "opportunity_set_sha256": str(opportunity_set.get("content_sha256") or ""),
        "selected_action_id": str(selected.get("action_id") or ""),
        "selected_action": selected,
        "candidate_count": len(ranked),
        "eligible_candidate_count": sum(
            1 for row in ranked if row.get("eligible") is True
        ),
        "candidates": ranked,
        "milestones": gates,
        "resource_availability": resources,
        "available_action_kinds": sorted(available_kinds),
        "handler_filter_applied": handler_filter_applied,
        "scheduler_policy": scheduler_policy,
        "round_robin_cursor": cursor,
        "semantics": {
            "task_labels_are_not_inputs": True,
            "same_state_and_resources_produce_same_order": True,
            "B4_is_state_not_task_membership": True,
            "B5_is_milestone_not_scheduler_stop": True,
            "route_dependencies_precede_validation_and_stock": True,
            "selection_grants_no_scientific_authority": True,
            "round_robin_ignores_adaptive_value_score_for_ordering": (
                scheduler_policy == "round_robin"
            ),
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _state_bonus(kind: str, gates: Mapping[str, bool]) -> float:
    has_routes = gates.get("B1_global_multi_route") is True
    validated = gates.get("B2_host_validated_routes") is True
    evidence_closed = gates.get("B3_exact_multi_source") is True
    stock_closed = gates.get("B4_stock_boundary") is True
    bonus = 0.0
    if not has_routes and kind in _ROUTE_ACTIONS:
        bonus += 120.0
    if has_routes and not stock_closed and kind in _ROUTE_ACTIONS:
        bonus += 75.0
    if not validated and kind == "reaction_validate":
        bonus += 90.0
    if not evidence_closed and kind in {
        "acquire_exact_evidence",
        "bind_exact_evidence",
        "resolve_conflict",
    }:
        bonus += 80.0
    if stock_closed and (not validated or not evidence_closed) and kind in _PROOF_ACTIONS:
        bonus += 55.0
    return bonus


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


__all__ = ["ACTION_SCHEDULER_POLICIES", "schedule_next_action"]
