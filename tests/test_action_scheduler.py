from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.action_service_policy import (
    ACTION_CLASS_ORDER,
    action_class_for_kind,
)
from cascade_planner.application.campaign_actions import compile_action_opportunities
from cascade_planner.application.campaign_actions import CampaignActionKind
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


def _service_opportunities() -> dict:
    actions = [
        {
            "action_id": "action:route:model-flood",
            "kind": "codex_global_architecture",
            "resource_class": "model",
            "deterministic": False,
            "base_priority": 10_000.0,
            "metadata": {"global_architecture": True},
        },
        {
            "action_id": "action:closure:materialize",
            "kind": "host_materialize",
            "resource_class": "deterministic",
            "deterministic": True,
            "base_priority": 1.0,
            "metadata": {},
        },
        {
            "action_id": "action:proof:conditions",
            "kind": "condition_enrich",
            "resource_class": "model",
            "deterministic": False,
            "base_priority": 1.0,
            "metadata": {},
        },
        {
            "action_id": "action:program:discover",
            "kind": "program_discover",
            "resource_class": "program",
            "deterministic": False,
            "base_priority": 1.0,
            "metadata": {},
        },
    ]
    return {
        "content_sha256": "service-opportunities",
        "actions": actions,
    }


def _structural_preflight_frontier(*, metadata: dict | None = None) -> dict:
    shared_metadata = dict(metadata or {})

    def item(
        kind: str,
        suffix: str,
        *,
        route_id: str = "route-family:downstream",
        deterministic: bool = False,
        item_metadata: dict | None = None,
    ) -> dict:
        return {
            "deficit_id": f"deficit:{kind}:{suffix}",
            "kind": kind,
            "object_id": f"object:{suffix}",
            "entity_ids": [f"entity:{suffix}"],
            "route_family_ids": [route_id],
            "dependency_ids": [],
            "deterministic": deterministic,
            "model_allowed": not deterministic,
            "reason": f"preflight_fixture_{kind}",
            "priority": 1.0 if kind == "materialization" else 10_000.0,
            "score": {
                "expected_portfolio_gain": 1.0,
                "distance_to_closure": 1.0,
                "evidence_gain": 1.0,
                "route_diversity_gain": 1.0,
                "cost_penalty": 0.0,
                "failure_risk_penalty": 0.0,
            },
            "metadata": {
                **shared_metadata,
                **dict(item_metadata or {}),
            },
        }

    return {
        "content_sha256": "structural-preflight-frontier",
        "items": [
            item(
                "materialization",
                "candidate",
                route_id="route-family:pending",
                deterministic=True,
            ),
            item("condition", "condition"),
            item("evidence", "evidence"),
            item("architecture", "architecture"),
            item("program_discovery", "program"),
            item(
                "route_closure",
                "closure",
                route_id="route-family:pending",
                deterministic=True,
            ),
            item(
                "expansion",
                "native",
                item_metadata={
                    "provider_preferences": ["chemenzy"],
                    "target_level_native_search": True,
                },
            ),
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


def test_scheduler_is_a_read_only_projection_of_canonical_opportunities() -> None:
    opportunities = compile_action_opportunities(_frontier())
    frozen = deepcopy(opportunities)

    decision = schedule_next_action(
        opportunities,
        milestones={"B1_global_multi_route": True},
        resource_availability={"validation": True},
    )

    assert opportunities == frozen
    assert decision["semantics"]["scheduler_does_not_execute_or_mutate_actions"]
    assert decision["semantics"]["stock_oracle_names_are_not_scheduler_inputs"]


def test_processable_structural_preflight_blocks_all_downstream_actions() -> None:
    opportunities = compile_action_opportunities(_structural_preflight_frontier())
    frozen = deepcopy(opportunities)

    adaptive = schedule_next_action(opportunities)
    round_robin = schedule_next_action(
        opportunities,
        policy="round_robin",
        round_robin_cursor=8,
    )

    assert opportunities == frozen
    assert adaptive["selected_action"]["kind"] == "host_materialize"
    assert round_robin["selected_action"]["kind"] == "host_materialize"
    preflight = adaptive["action_preflight"]
    assert preflight["schema_version"] == "campaign_action_preflight.v1"
    assert preflight["gate_active"] is True
    assert preflight["initial_discovery_exempt"] is False
    assert preflight["blocked_action_count"] == 6
    assert {
        "canonical_reaction_identity",
        "valid_material_structure",
        "valid_reagent_structure",
        "element_inventory",
        "large_atom_jump",
        "self_loop",
        "ancestor_cycle",
        "canonical_graph_cycle",
        "duplicate_reaction_edge",
    } == set(preflight["covered_checks"])
    by_kind = {row["kind"]: row for row in adaptive["candidates"]}
    assert by_kind["condition_enrich"]["blocked_reasons"] == [
        "processable_structural_preflight_precedes_downstream_action"
    ]
    assert by_kind["codex_global_architecture"]["blocked_reasons"] == [
        "processable_structural_preflight_precedes_downstream_action"
    ]
    assert by_kind["program_discover"]["blocked_reasons"] == [
        "processable_structural_preflight_precedes_downstream_action"
    ]
    assert by_kind["recompute_route_closure"]["blocked_reasons"] == [
        "route_materialization_precedes_route_closure"
    ]
    assert by_kind["acquire_exact_evidence"]["blocked_reasons"] == [
        "pending_materialization_precedes_evidence"
    ]
    assert by_kind["chemenzy_target_expand"]["blocked_reasons"] == [
        "processable_structural_preflight_precedes_downstream_action"
    ]
    assert round_robin["action_preflight"] == preflight
    assert round_robin["semantics"][
        "round_robin_ignores_adaptive_value_score_for_ordering"
    ] is True


def test_initial_discovery_and_unprocessable_checks_remain_schedulable() -> None:
    frontier = _structural_preflight_frontier()
    frontier["items"] = [
        row for row in frontier["items"] if row["kind"] != "materialization"
    ]
    initial = schedule_next_action(compile_action_opportunities(frontier))
    initial_by_kind = {row["kind"]: row for row in initial["candidates"]}

    assert initial["action_preflight"]["gate_active"] is False
    assert initial["action_preflight"]["initial_discovery_exempt"] is True
    assert initial_by_kind["chemenzy_target_expand"]["eligible"] is True
    assert initial_by_kind["codex_global_architecture"]["eligible"] is True

    opportunities = compile_action_opportunities(_structural_preflight_frontier())
    unavailable = schedule_next_action(
        opportunities,
        resource_availability={"deterministic": False},
        policy="round_robin",
        round_robin_cursor=8,
    )
    unavailable_by_kind = {
        row["kind"]: row for row in unavailable["candidates"]
    }
    assert unavailable["action_preflight"]["gate_active"] is False
    assert unavailable["action_preflight"]["pending_check_action_ids"]
    assert unavailable["action_preflight"]["processable_check_action_ids"] == []
    assert unavailable_by_kind["host_materialize"]["blocked_reasons"] == [
        "resource_unavailable:deterministic"
    ]
    assert unavailable["selected_action"]["kind"] == "codex_global_architecture"

    handler_deferred = schedule_next_action(
        opportunities,
        available_action_kinds=("condition_enrich",),
    )
    handler_by_kind = {row["kind"]: row for row in handler_deferred["candidates"]}
    assert handler_deferred["action_preflight"]["gate_active"] is False
    assert handler_by_kind["host_materialize"]["blocked_reasons"] == [
        "handler_unavailable:host_materialize"
    ]
    assert handler_deferred["selected_action"]["kind"] == "condition_enrich"

    scoped_cohort = schedule_next_action(
        opportunities,
        available_action_kinds=("codex_global_architecture",),
        preflight_available_action_kinds=(
            "host_materialize",
            "codex_global_architecture",
        ),
    )
    scoped_by_kind = {row["kind"]: row for row in scoped_cohort["candidates"]}
    assert scoped_cohort["action_preflight"]["gate_active"] is True
    assert scoped_cohort["selected_action_id"] == ""
    assert scoped_by_kind["codex_global_architecture"]["blocked_reasons"] == [
        "processable_structural_preflight_precedes_downstream_action"
    ]


def test_structural_preflight_is_label_blind_and_replay_stable() -> None:
    first_opportunities = compile_action_opportunities(
        _structural_preflight_frontier(
            metadata={
                "target_label": "benchmark-a",
                "dataset": "hidden-a",
                "objective": "legacy-score-a",
            }
        )
    )
    second_opportunities = compile_action_opportunities(
        _structural_preflight_frontier(
            metadata={
                "target_label": "benchmark-b",
                "dataset": "hidden-b",
                "objective": "legacy-score-b",
            }
        )
    )

    first = schedule_next_action(first_opportunities)
    replay = schedule_next_action(first_opportunities)
    relabeled = schedule_next_action(second_opportunities)

    assert first["action_preflight"] == replay["action_preflight"]
    assert first["action_preflight"] == relabeled["action_preflight"]
    assert first["selected_action_id"] == replay["selected_action_id"]
    assert first["selected_action_id"] == relabeled["selected_action_id"]
    assert [
        (row["action_id"], row["eligible"], row["blocked_reasons"])
        for row in first["candidates"]
    ] == [
        (row["action_id"], row["eligible"], row["blocked_reasons"])
        for row in relabeled["candidates"]
    ]


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
        prior_action_kinds=("codex_global_architecture",) * 7,
    )

    assert decision["scheduler_policy"] == "round_robin"
    assert decision["selected_action"]["kind"] == "chemenzy_frontier_expand"
    assert decision["semantics"][
        "round_robin_ignores_adaptive_value_score_for_ordering"
    ] is True
    assert decision["action_class_service"]["required_action_class"] == ""
    assert decision["action_class_service"]["minimum_service_enforced"] is False


def test_every_campaign_action_kind_has_one_target_blind_service_class() -> None:
    observed = {
        action_class_for_kind(kind.value) for kind in CampaignActionKind
    }

    assert observed == set(ACTION_CLASS_ORDER)
    assert all(
        action_class_for_kind(kind.value) != "unclassified"
        for kind in CampaignActionKind
    )


def test_adaptive_minimum_service_prevents_model_flood_from_starving_closure() -> None:
    decision = schedule_next_action(
        _service_opportunities(),
        prior_action_kinds=("codex_global_architecture",) * 9,
    )

    assert decision["selected_action"]["kind"] == "host_materialize"
    assert decision["selected_action"]["selection_reasons"] == [
        "minimum_service_guarantee_due:deterministic_closure"
    ]
    service = decision["action_class_service"]
    assert service["next_action_ordinal"] == 10
    assert service["required_action_class"] == "deterministic_closure"
    assert service["minimum_service_guarantee_applied"] is True


def test_adaptive_service_window_reaches_every_continuously_eligible_class() -> None:
    history: list[str] = []
    trace: list[str] = []

    for _ in range(12):
        decision = schedule_next_action(
            _service_opportunities(),
            prior_action_kinds=tuple(history),
        )
        selected_kind = str(decision["selected_action"]["kind"])
        trace.append(str(decision["selected_action"]["action_class"]))
        history.append(selected_kind)

    assert trace[:9] == ["route_discovery"] * 9
    assert trace[9:] == [
        "deterministic_closure",
        "scientific_proof",
        "program_experiment",
    ]
    assert set(trace) == set(ACTION_CLASS_ORDER)


def test_blocked_action_classes_lend_capacity_without_creating_budget() -> None:
    decision = schedule_next_action(
        _service_opportunities(),
        resource_availability={
            "model": True,
            "deterministic": False,
            "program": False,
        },
        available_action_kinds=("codex_global_architecture",),
        prior_action_kinds=("codex_global_architecture",) * 7,
    )

    assert decision["selected_action"]["kind"] == "codex_global_architecture"
    service = decision["action_class_service"]
    assert service["required_action_class"] == ""
    assert service["borrowed_service_capacity"] is True
    assert service["borrowable_from_action_classes"] == [
        "deterministic_closure",
        "scientific_proof",
        "program_experiment",
    ]
    assert service["semantics"][
        "borrowed_service_capacity_creates_no_run_kernel_budget"
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


def test_adaptive_score_exposes_complete_value_and_cost_components() -> None:
    frontier = _frontier()
    frontier["items"] = [
        {
            **frontier["items"][0],
            "deficit_id": "deficit:validation:low",
            "object_id": "edge:low",
            "entity_ids": ["edge:low"],
            "route_family_ids": ["route-family:low"],
            "priority": 0.0,
            "score": {},
        },
        {
            **frontier["items"][0],
            "deficit_id": "deficit:validation:high",
            "object_id": "edge:high",
            "entity_ids": ["edge:high"],
            "route_family_ids": ["route-family:high"],
            "priority": 0.0,
            "score": {
                "expected_portfolio_gain": 0.4,
                "evidence_gain": 0.3,
                "route_diversity_gain": 0.2,
                "dependency_unblock_count": 2,
                "novelty_gain": 0.5,
                "success_probability_interval": [0.6, 0.9],
                "cost_penalty": 0.1,
                "failure_risk_penalty": 0.2,
            },
        },
    ]

    decision = schedule_next_action(compile_action_opportunities(frontier))

    assert decision["selected_action"]["deficit_id"] == (
        "deficit:validation:high"
    )
    components = decision["selected_action"]["schedule_components"]
    assert components["route_gain"] == 36.0
    assert components["proof_gain"] == 24.0
    assert components["diversity_gain"] == 14.0
    assert components["dependency_unblock_gain"] == 120.0
    assert components["novelty_gain"] == 22.5
    assert components["success_likelihood_gain"] == 30.0
    assert components["cost_penalty"] == 6.5
    assert components["risk_penalty"] == 17.0
    assert decision["selected_action"]["selection_reasons"] == [
        "highest_ranked_eligible_action"
    ]
    unselected = next(
        row
        for row in decision["candidates"]
        if row["deficit_id"] == "deficit:validation:low"
    )
    assert unselected["not_selected_reasons"] == ["lower_deterministic_rank"]


def test_equal_scores_use_stable_action_id_tie_break() -> None:
    frontier = _frontier()
    shared = {
        **frontier["items"][0],
        "priority": 0.0,
        "score": {},
    }
    first = {
        **shared,
        "deficit_id": "deficit:validation:a",
        "object_id": "edge:a",
        "entity_ids": ["edge:a"],
        "route_family_ids": ["route-family:a"],
    }
    second = {
        **shared,
        "deficit_id": "deficit:validation:b",
        "object_id": "edge:b",
        "entity_ids": ["edge:b"],
        "route_family_ids": ["route-family:b"],
    }
    forward = compile_action_opportunities(
        {"content_sha256": "tie", "items": [first, second]}
    )
    reversed_input = compile_action_opportunities(
        {"content_sha256": "tie", "items": [second, first]}
    )

    left = schedule_next_action(forward)
    right = schedule_next_action(reversed_input)

    expected = min(row["action_id"] for row in forward["actions"])
    assert left["selected_action_id"] == expected
    assert right["selected_action_id"] == expected
    assert [row["action_id"] for row in left["candidates"]] == [
        row["action_id"] for row in right["candidates"]
    ]


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
