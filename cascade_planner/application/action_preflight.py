"""Read-only structural preflight over canonical Action opportunities."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


ACTION_PREFLIGHT_SCHEMA = "campaign_action_preflight.v1"
STRUCTURAL_PREFLIGHT_BLOCK_REASON = (
    "processable_structural_preflight_precedes_downstream_action"
)

_MATERIALIZATION_KIND = "host_materialize"
_DEPENDENT_ACTION_KINDS = frozenset(
    {
        "resolve_conflict",
        "reaction_validate",
        "stock_audit",
        "acquire_exact_evidence",
        "bind_exact_evidence",
        "condition_enrich",
        "chemenzy_target_expand",
        "native_short_tail_expand",
        "codex_global_architecture",
        "codex_global_replan",
        "program_discover",
        "program_review",
        "program_admit",
        "program_validate",
        "experiment_feedback_ingest",
        "recompute_route_closure",
    }
)
_COVERED_CHECKS = (
    "canonical_reaction_identity",
    "valid_material_structure",
    "valid_reagent_structure",
    "element_inventory",
    "large_atom_jump",
    "self_loop",
    "ancestor_cycle",
    "canonical_graph_cycle",
    "duplicate_reaction_edge",
)


def compile_action_preflight(
    opportunity_set: Mapping[str, Any],
    *,
    resource_availability: Mapping[str, Any] | None = None,
    in_flight_action_ids: Iterable[str] = (),
    available_action_kinds: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Project whether canonical cheap checks must precede downstream work.

    The projector deliberately does not rerun chemistry validators.  A
    canonical materialization deficit exists only after the shared admission
    path created an accepted hypothesis, and its host worker owns the final
    identity, structure, cycle, duplicate, and element checks.
    """

    resources = {
        str(key): value is not False
        for key, value in dict(resource_availability or {}).items()
    }
    in_flight = {str(value) for value in in_flight_action_ids if str(value)}
    handler_filter_applied = available_action_kinds is not None
    available_kinds = {
        str(value) for value in (available_action_kinds or ()) if str(value)
    }
    actions = [
        dict(value)
        for value in opportunity_set.get("actions") or []
        if isinstance(value, Mapping)
    ]
    check_candidates: list[dict[str, Any]] = []
    for row in actions:
        if not _is_canonical_materialization(row):
            continue
        action_id = str(row.get("action_id") or "")
        kind = str(row.get("kind") or "")
        resource_class = str(row.get("resource_class") or "")
        reasons: list[str] = []
        if not action_id:
            reasons.append("action_identity_missing")
        if row.get("deterministic") is not True or resource_class != "deterministic":
            reasons.append("canonical_materialization_contract_invalid")
        if action_id in in_flight:
            reasons.append("action_already_in_flight")
        if handler_filter_applied and kind not in available_kinds:
            reasons.append(f"handler_unavailable:{kind}")
        if resources.get(resource_class, True) is False:
            reasons.append(f"resource_unavailable:{resource_class}")
        check_candidates.append(
            {
                "action_id": action_id,
                "route_family_ids": sorted(
                    str(value)
                    for value in row.get("route_family_ids") or []
                    if str(value)
                ),
                "processable": not reasons,
                "blocked_reasons": reasons,
            }
        )

    processable = [row for row in check_candidates if row["processable"]]
    processable_ids = {
        str(row["action_id"]) for row in processable if str(row["action_id"])
    }
    processable_routes = {
        str(route_id)
        for row in processable
        for route_id in row.get("route_family_ids") or []
        if str(route_id)
    }
    action_block_reasons: dict[str, list[str]] = {}
    if processable:
        for row in actions:
            action_id = str(row.get("action_id") or "")
            kind = str(row.get("kind") or "")
            if not action_id or kind not in _DEPENDENT_ACTION_KINDS:
                continue
            routes = {
                str(value)
                for value in row.get("route_family_ids") or []
                if str(value)
            }
            shares_pending_route = bool(routes & processable_routes)
            if kind == "reaction_validate":
                if not shares_pending_route:
                    continue
                reason = "route_materialization_precedes_reaction_validation"
            elif kind == "stock_audit":
                if not shares_pending_route:
                    continue
                reason = "route_materialization_precedes_stock_audit"
            elif kind == "recompute_route_closure":
                if not shares_pending_route:
                    continue
                reason = "route_materialization_precedes_route_closure"
            elif kind in {"acquire_exact_evidence", "bind_exact_evidence"}:
                reason = "pending_materialization_precedes_evidence"
            else:
                reason = STRUCTURAL_PREFLIGHT_BLOCK_REASON
            action_block_reasons[action_id] = [reason]

    check_contract_reasons = {
        str(row["action_id"]): ["canonical_materialization_contract_invalid"]
        for row in check_candidates
        if row.get("action_id")
        and "canonical_materialization_contract_invalid"
        in row.get("blocked_reasons", ())
    }
    result = {
        "schema_version": ACTION_PREFLIGHT_SCHEMA,
        "gate_active": bool(processable),
        "pending_check_action_ids": sorted(
            str(row["action_id"])
            for row in check_candidates
            if str(row["action_id"])
        ),
        "processable_check_action_ids": sorted(processable_ids),
        "processable_route_family_ids": sorted(processable_routes),
        "check_candidates": sorted(
            check_candidates,
            key=lambda row: str(row.get("action_id") or ""),
        ),
        "check_contract_block_reasons": dict(sorted(check_contract_reasons.items())),
        "action_block_reasons": dict(sorted(action_block_reasons.items())),
        "blocked_action_count": len(action_block_reasons),
        "covered_checks": list(_COVERED_CHECKS),
        "initial_discovery_exempt": not check_candidates,
        "semantics": {
            "read_only_projection": True,
            "canonical_admission_and_materialization_are_reused": True,
            "chemistry_validators_are_not_duplicated": True,
            "only_processable_checks_block_downstream_actions": True,
            "target_dataset_and_objective_labels_are_not_inputs": True,
            "initial_discovery_without_canonical_candidates_is_not_blocked": True,
            "round_robin_orders_only_the_remaining_eligible_actions": True,
            "preflight_creates_no_queue_budget_or_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _is_canonical_materialization(row: Mapping[str, Any]) -> bool:
    metadata = dict(row.get("metadata") or {})
    return (
        str(row.get("kind") or "") == _MATERIALIZATION_KIND
        and str(metadata.get("frontier_kind") or "") == "materialization"
        and any(str(value) for value in row.get("subject_ids") or [])
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
    "ACTION_PREFLIGHT_SCHEMA",
    "STRUCTURAL_PREFLIGHT_BLOCK_REASON",
    "compile_action_preflight",
]
