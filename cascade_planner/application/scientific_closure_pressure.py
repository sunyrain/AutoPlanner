"""Target-blind scientific-closure pressure from one Action opportunity set."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCIENTIFIC_CLOSURE_PRESSURE_SCHEMA = "campaign_scientific_closure_pressure.v1"

_ACTION_AXES = {
    "reaction_validate": "reaction_validation",
    "acquire_exact_evidence": "exact_evidence",
    "bind_exact_evidence": "exact_evidence",
    "resolve_conflict": "exact_evidence",
    "condition_enrich": "conditions",
}
_AXIS_BASE_BONUS = {
    "reaction_validation": 90.0,
    "exact_evidence": 80.0,
    "conditions": 0.0,
}


def compile_scientific_closure_pressure(
    opportunity_set: Mapping[str, Any],
    *,
    milestones: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Increase proof work value as canonical routes mature toward closure."""

    gates = {
        str(key): value is True
        for key, value in dict(milestones or {}).items()
    }
    actions = [
        dict(value)
        for value in opportunity_set.get("actions") or []
        if isinstance(value, Mapping)
    ]
    scientific_actions: list[dict[str, Any]] = []
    axis_action_counts = {axis: 0 for axis in _AXIS_BASE_BONUS}
    axis_route_families = {axis: set() for axis in _AXIS_BASE_BONUS}
    for action in actions:
        axis = _ACTION_AXES.get(str(action.get("kind") or ""))
        if axis is None:
            continue
        route_family_ids = sorted(
            str(value)
            for value in action.get("route_family_ids") or []
            if str(value)
        )
        scientific_actions.append(
            {
                "action_id": str(action.get("action_id") or ""),
                "kind": str(action.get("kind") or ""),
                "route_family_ids": route_family_ids,
            }
        )
        axis_action_counts[axis] += 1
        axis_route_families[axis].update(route_family_ids)
    open_axes = sorted(
        axis for axis, count in axis_action_counts.items() if count > 0
    )
    stock_closed = gates.get("B4_stock_boundary") is True
    has_routes = gates.get("B1_global_multi_route") is True or stock_closed
    if stock_closed:
        maturity = "stock_closed_route_portfolio"
        maturity_bonus = 55.0
    elif has_routes:
        maturity = "route_portfolio_available"
        maturity_bonus = 25.0
    else:
        maturity = "pre_route_portfolio"
        maturity_bonus = 0.0
    last_open_axis_bonus = 20.0 if has_routes and len(open_axes) == 1 else 0.0
    axis_bonuses = {
        axis: round(
            _AXIS_BASE_BONUS[axis]
            + maturity_bonus
            + last_open_axis_bonus,
            6,
        )
        for axis in open_axes
    }
    action_kind_bonuses = {
        kind: axis_bonuses[axis]
        for kind, axis in sorted(_ACTION_AXES.items())
        if axis in axis_bonuses
    }
    result = {
        "schema_version": SCIENTIFIC_CLOSURE_PRESSURE_SCHEMA,
        "scientific_action_projection_sha256": _digest(
            sorted(
                scientific_actions,
                key=lambda row: (
                    row["kind"],
                    row["action_id"],
                    row["route_family_ids"],
                ),
            )
        ),
        "route_maturity": maturity,
        "open_axes": open_axes,
        "open_axis_count": len(open_axes),
        "axis_action_counts": dict(sorted(axis_action_counts.items())),
        "axis_route_family_counts": {
            axis: len(axis_route_families[axis])
            for axis in sorted(axis_route_families)
        },
        "progression": {
            "route_portfolio_bonus": 25.0 if has_routes else 0.0,
            "stock_closed_increment": 30.0 if stock_closed else 0.0,
            "last_open_axis_bonus": last_open_axis_bonus,
        },
        "axis_bonuses": axis_bonuses,
        "action_kind_bonuses": action_kind_bonuses,
        "semantics": {
            "single_action_opportunity_set_is_the_only_work_source": True,
            "target_labels_and_dataset_metadata_are_not_inputs": True,
            "conditions_remain_independent_of_B2_and_B3": True,
            "pressure_changes_priority_not_budget_or_authority": True,
            "route_topology_is_not_mutated_or_removed": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


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
    "SCIENTIFIC_CLOSURE_PRESSURE_SCHEMA",
    "compile_scientific_closure_pressure",
]
