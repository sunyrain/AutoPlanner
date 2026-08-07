from __future__ import annotations

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.campaign_actions import compile_action_opportunities
from cascade_planner.application.campaign_trajectory import (
    compile_campaign_snapshot,
    compile_campaign_trajectory,
)


def _frontier(*, metadata: dict | None = None) -> dict:
    return {
        "content_sha256": "frontier-sha",
        "items": [
            {
                "deficit_id": "deficit:validation:1",
                "kind": "validation",
                "object_id": "edge:1",
                "entity_ids": ["edge:1"],
                "route_family_ids": ["route-family:1"],
                "dependency_ids": [],
                "deterministic": True,
                "model_allowed": False,
                "reason": "materialized_edge_requires_reaction_validation",
                "priority": 500.0,
                "score": {
                    "expected_portfolio_gain": 0.9,
                    "distance_to_closure": 0.9,
                    "evidence_gain": 0.5,
                    "route_diversity_gain": 0.1,
                    "cost_penalty": 0.2,
                    "failure_risk_penalty": 0.1,
                },
                "metadata": dict(metadata or {}),
            },
            {
                "deficit_id": "deficit:expansion:1",
                "kind": "expansion",
                "object_id": "molecule:1",
                "entity_ids": ["molecule:1"],
                "route_family_ids": ["route-family:1"],
                "dependency_ids": [],
                "deterministic": False,
                "model_allowed": True,
                "reason": "stock_rejected_leaf_requires_upstream_expansion",
                "priority": 420.0,
                "score": {
                    "expected_portfolio_gain": 0.8,
                    "distance_to_closure": 0.8,
                    "evidence_gain": 0.1,
                    "route_diversity_gain": 0.6,
                    "cost_penalty": 0.4,
                    "failure_risk_penalty": 0.3,
                },
                "metadata": {
                    **dict(metadata or {}),
                    "provider_preferences": [
                        "chemenzy",
                        "codex_global_director",
                    ],
                },
            },
        ],
    }


def test_action_scheduler_is_invariant_to_legacy_view_metadata() -> None:
    benchmark = compile_action_opportunities(
        _frontier(metadata={"legacy_view": "benchmark_search"})
    )
    scientific = compile_action_opportunities(
        _frontier(metadata={"legacy_view": "scientific_proof"})
    )
    milestones = {
        "B1_global_multi_route": True,
        "B2_host_validated_routes": False,
        "B3_exact_multi_source": False,
        "B4_stock_boundary": True,
        "B5_configured_portfolio_acceptance": False,
    }

    first = schedule_next_action(benchmark, milestones=milestones)
    second = schedule_next_action(scientific, milestones=milestones)

    assert first["selected_action_id"] == second["selected_action_id"]
    assert [row["action_id"] for row in first["candidates"]] == [
        row["action_id"] for row in second["candidates"]
    ]
    assert first["selected_action"]["kind"] == "reaction_validate"


def test_action_scheduler_expands_one_deficit_into_provider_choices() -> None:
    opportunities = compile_action_opportunities(_frontier())

    assert {row["kind"] for row in opportunities["actions"]} == {
        "reaction_validate",
        "chemenzy_frontier_expand",
        "codex_global_replan",
    }
    resources = {
        row["kind"]: row["resource_class"]
        for row in opportunities["actions"]
    }
    assert resources["chemenzy_frontier_expand"] == "native_search_frontier"


def test_scheduler_does_not_dispatch_unscoped_codex_replan() -> None:
    opportunities = compile_action_opportunities(_frontier())

    decision = schedule_next_action(
        opportunities,
        resource_availability={
            "validation": True,
            "native_search_frontier": True,
            "model": True,
        },
    )

    codex = next(
        row
        for row in decision["candidates"]
        if row["kind"] == "codex_global_replan"
    )
    assert codex["eligible"] is False
    assert "global_replan_scope_missing" in codex["blocked_reasons"]
    assert decision["selected_action"]["kind"] != "codex_global_replan"


def test_round_robin_scheduler_uses_frozen_kind_cursor_not_adaptive_score() -> None:
    opportunities = compile_action_opportunities(_frontier())

    decision = schedule_next_action(
        opportunities,
        policy="round_robin",
        round_robin_cursor=9,
        resource_availability={
            "validation": True,
            "native_search_frontier": True,
            "model": True,
        },
    )

    assert decision["scheduler_policy"] == "round_robin"
    assert decision["selected_action"]["kind"] == "chemenzy_frontier_expand"
    assert decision["semantics"][
        "round_robin_ignores_adaptive_value_score_for_ordering"
    ] is True


def test_target_native_search_has_a_distinct_protected_resource_class() -> None:
    frontier = _frontier()
    frontier["items"] = [
        {
            **frontier["items"][1],
            "metadata": {
                "provider_preferences": ["chemenzy"],
                "target_level_native_search": True,
                "native_budget_reservation": "target_level",
            },
        }
    ]

    opportunities = compile_action_opportunities(frontier)

    assert opportunities["actions"][0]["kind"] == "chemenzy_target_expand"
    assert (
        opportunities["actions"][0]["resource_class"]
        == "native_search_target"
    )


def test_route_materialization_precedes_same_route_validation_and_stock() -> None:
    frontier = {
        "content_sha256": "frontier-sha",
        "items": [
            {
                "deficit_id": "deficit:materialization:1",
                "kind": "materialization",
                "object_id": "hypothesis:1",
                "entity_ids": ["hypothesis:1"],
                "route_family_ids": ["route-family:1"],
                "dependency_ids": [],
                "deterministic": True,
                "model_allowed": False,
                "reason": "accepted_hypothesis_requires_host_materialization",
                "priority": 10.0,
                "score": {
                    "expected_portfolio_gain": 0.1,
                    "distance_to_closure": 0.1,
                    "evidence_gain": 0.0,
                    "route_diversity_gain": 0.0,
                    "cost_penalty": 0.1,
                    "failure_risk_penalty": 0.1,
                },
                "metadata": {},
            },
            {
                "deficit_id": "deficit:validation:1",
                "kind": "validation",
                "object_id": "edge:1",
                "entity_ids": ["edge:1"],
                "route_family_ids": ["route-family:1"],
                "dependency_ids": [],
                "deterministic": True,
                "model_allowed": False,
                "reason": "materialized_edge_requires_reaction_validation",
                "priority": 900.0,
                "score": {
                    "expected_portfolio_gain": 1.0,
                    "distance_to_closure": 1.0,
                    "evidence_gain": 1.0,
                    "route_diversity_gain": 0.0,
                    "cost_penalty": 0.0,
                    "failure_risk_penalty": 0.0,
                },
                "metadata": {},
            },
            {
                "deficit_id": "deficit:stock:1",
                "kind": "stock",
                "object_id": "molecule:1",
                "entity_ids": ["molecule:1"],
                "route_family_ids": ["route-family:1"],
                "dependency_ids": [],
                "deterministic": True,
                "model_allowed": False,
                "reason": "selected_leaf_requires_trusted_stock_audit",
                "priority": 1_000.0,
                "score": {
                    "expected_portfolio_gain": 1.0,
                    "distance_to_closure": 1.0,
                    "evidence_gain": 1.0,
                    "route_diversity_gain": 0.0,
                    "cost_penalty": 0.0,
                    "failure_risk_penalty": 0.0,
                },
                "metadata": {},
            },
        ],
    }

    decision = schedule_next_action(compile_action_opportunities(frontier))

    assert decision["selected_action"]["kind"] == "host_materialize"
    blocked = {
        row["kind"]: row["blocked_reasons"] for row in decision["candidates"]
    }
    assert blocked["reaction_validate"] == [
        "route_materialization_precedes_reaction_validation"
    ]
    assert blocked["stock_audit"] == [
        "route_materialization_precedes_stock_audit"
    ]


def test_campaign_trajectory_records_first_milestones() -> None:
    first = compile_campaign_snapshot(
        phase="seed",
        observed_at="2026-08-06T00:00:00Z",
        graph_revision=2,
        gates={
            "gates": {
                "B1_global_multi_route": True,
                "B4_stock_boundary": False,
            },
            "counts": {"target_rooted_distinct_skeletons": 2},
        },
        resource_usage={"attempt_count": 2},
    )
    second = compile_campaign_snapshot(
        phase="closeout",
        observed_at="2026-08-06T00:01:00Z",
        graph_revision=4,
        gates={
            "gates": {
                "B1_global_multi_route": True,
                "B4_stock_boundary": True,
            },
            "counts": {"stock_closed_skeletons": 1},
        },
        resource_usage={"attempt_count": 4},
    )

    trajectory = compile_campaign_trajectory([second, first])

    assert [row["phase"] for row in trajectory["snapshots"]] == [
        "seed",
        "closeout",
    ]
    assert trajectory["first_achieved"]["B1_global_multi_route"][
        "graph_revision"
    ] == 2
    assert trajectory["first_achieved"]["B4_stock_boundary"][
        "graph_revision"
    ] == 4
