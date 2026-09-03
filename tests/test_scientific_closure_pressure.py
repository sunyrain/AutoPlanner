from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.scientific_closure_pressure import (
    compile_scientific_closure_pressure,
)


def _opportunities(*, include_all_axes: bool = True) -> dict:
    actions = [
        {
            "action_id": "action:route:architecture",
            "kind": "codex_global_architecture",
            "resource_class": "model",
            "deterministic": False,
            "base_priority": 150.0,
            "route_family_ids": [],
            "metadata": {"global_architecture": True},
        },
        {
            "action_id": "action:proof:condition",
            "kind": "condition_enrich",
            "resource_class": "condition",
            "deterministic": False,
            "base_priority": 100.0,
            "route_family_ids": ["family:1"],
            "metadata": {},
        },
    ]
    if include_all_axes:
        actions.extend(
            [
                {
                    "action_id": "action:proof:validation",
                    "kind": "reaction_validate",
                    "resource_class": "validation",
                    "deterministic": True,
                    "base_priority": 0.0,
                    "route_family_ids": ["family:1"],
                    "metadata": {},
                },
                {
                    "action_id": "action:proof:evidence",
                    "kind": "bind_exact_evidence",
                    "resource_class": "evidence",
                    "deterministic": True,
                    "base_priority": 0.0,
                    "route_family_ids": ["family:1"],
                    "metadata": {},
                },
            ]
        )
    return {
        "content_sha256": "scientific-closure-opportunities",
        "actions": actions,
    }


def test_scientific_closure_pressure_progresses_with_route_maturity() -> None:
    opportunities = _opportunities()
    before_routes = compile_scientific_closure_pressure(opportunities)
    with_routes = compile_scientific_closure_pressure(
        opportunities,
        milestones={"B1_global_multi_route": True},
    )
    stock_closed = compile_scientific_closure_pressure(
        opportunities,
        milestones={
            "B1_global_multi_route": True,
            "B4_stock_boundary": True,
        },
    )

    assert before_routes["axis_bonuses"] == {
        "conditions": 0.0,
        "exact_evidence": 80.0,
        "reaction_validation": 90.0,
    }
    assert with_routes["axis_bonuses"] == {
        "conditions": 25.0,
        "exact_evidence": 105.0,
        "reaction_validation": 115.0,
    }
    assert stock_closed["axis_bonuses"] == {
        "conditions": 55.0,
        "exact_evidence": 135.0,
        "reaction_validation": 145.0,
    }


def test_condition_axis_receives_last_mile_value_independently() -> None:
    pressure = compile_scientific_closure_pressure(
        _opportunities(include_all_axes=False),
        milestones={
            "B1_global_multi_route": True,
            "B2_host_validated_routes": True,
            "B3_exact_multi_source": True,
            "B4_stock_boundary": True,
        },
    )

    assert pressure["open_axes"] == ["conditions"]
    assert pressure["progression"] == {
        "route_portfolio_bonus": 25.0,
        "stock_closed_increment": 30.0,
        "last_open_axis_bonus": 20.0,
    }
    assert pressure["action_kind_bonuses"] == {"condition_enrich": 75.0}


def test_adaptive_scheduler_promotes_science_after_route_closure() -> None:
    opportunities = _opportunities(include_all_axes=False)
    before_routes = schedule_next_action(
        opportunities,
        resource_availability={"model": True, "condition": True},
    )
    after_closure = schedule_next_action(
        opportunities,
        milestones={
            "B1_global_multi_route": True,
            "B2_host_validated_routes": True,
            "B3_exact_multi_source": True,
            "B4_stock_boundary": True,
        },
        resource_availability={"model": True, "condition": True},
    )

    assert before_routes["selected_action"]["kind"] == (
        "codex_global_architecture"
    )
    assert after_closure["selected_action"]["kind"] == "condition_enrich"
    assert after_closure["selected_action"]["schedule_components"][
        "scientific_closure_pressure_bonus"
    ] == 75.0


def test_pressure_is_label_order_invariant_and_read_only() -> None:
    opportunities = _opportunities()
    frozen = deepcopy(opportunities)
    labelled = deepcopy(opportunities)
    for action in labelled["actions"]:
        action["metadata"] = {
            **dict(action.get("metadata") or {}),
            "target_name": "opaque",
            "dataset_id": "held-out",
            "objective_mode": "compatibility-only",
        }
    labelled["actions"].reverse()

    baseline = compile_scientific_closure_pressure(
        opportunities,
        milestones={"B1_global_multi_route": True},
    )
    comparison = compile_scientific_closure_pressure(
        labelled,
        milestones={"B1_global_multi_route": True},
    )

    assert baseline == comparison
    assert opportunities == frozen
    assert baseline["semantics"]["route_topology_is_not_mutated_or_removed"]


def test_pressure_cannot_override_policy_or_resource_budget() -> None:
    opportunities = _opportunities(include_all_axes=False)
    milestones = {
        "B1_global_multi_route": True,
        "B4_stock_boundary": True,
    }
    round_robin = schedule_next_action(
        opportunities,
        milestones=milestones,
        resource_availability={"model": True, "condition": True},
        policy="round_robin",
        round_robin_cursor=8,
    )
    exhausted = schedule_next_action(
        {
            **opportunities,
            "actions": [opportunities["actions"][1]],
        },
        milestones=milestones,
        resource_availability={"condition": False},
    )

    assert round_robin["selected_action"]["kind"] == (
        "codex_global_architecture"
    )
    assert round_robin["semantics"][
        "round_robin_ignores_adaptive_value_score_for_ordering"
    ] is True
    assert exhausted["selected_action_id"] == ""
    assert exhausted["candidates"][0]["blocked_reasons"] == [
        "resource_unavailable:condition"
    ]
