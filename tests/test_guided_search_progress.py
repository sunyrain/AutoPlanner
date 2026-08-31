from __future__ import annotations

from cascade_planner.application.guided_search_progress import (
    compile_parent_route_stock_progress,
    evaluate_guided_stock_progress,
)


def _portfolio(*routes: dict) -> dict:
    return {"route_candidates": list(routes)}


def test_guided_progress_counts_only_target_rooted_parent_routes() -> None:
    progress = compile_parent_route_stock_progress(
        _portfolio(
            {
                "route_id": "route:root",
                "route_family_id": "family:root",
                "root_edge_ids": ["edge:target"],
                "open_leaf_molecule_ids": ["mol:a", "mol:b"],
                "all_leaves_stock_closed": False,
            },
            {
                "route_id": "route:subtarget",
                "route_family_id": "family:root",
                "root_edge_ids": [],
                "open_leaf_molecule_ids": [],
                "all_leaves_stock_closed": True,
            },
            {
                "route_id": "route:other",
                "route_family_id": "family:other",
                "root_edge_ids": ["edge:other"],
                "open_leaf_molecule_ids": [],
                "all_leaves_stock_closed": True,
            },
        ),
        parent_route_family_ids=("family:root",),
    )

    assert progress["target_rooted_route_count"] == 1
    assert progress["best_open_leaf_count"] == 2
    assert progress["root_stock_closed"] is False
    assert progress["semantics"][
        "subtarget_stock_closure_does_not_grant_root_progress"
    ] is True


def test_root_stock_closure_does_not_suppress_distinct_frontiers() -> None:
    before = {"best_open_leaf_count": 3}
    progressed = evaluate_guided_stock_progress(
        before,
        {"best_open_leaf_count": 2},
        root_b4_reached=False,
    )
    unchanged = evaluate_guided_stock_progress(
        before,
        {"best_open_leaf_count": 3},
        root_b4_reached=False,
    )
    closed = evaluate_guided_stock_progress(
        before,
        {"best_open_leaf_count": 0},
        root_b4_reached=True,
    )

    assert progressed["stock_open_leaf_decrease"] == 1
    assert progressed["continue_guided_search"] is True
    assert unchanged["continue_guided_search"] is True
    assert unchanged["retry_same_frontier"] is False
    assert unchanged["reason"] == (
        "parent_route_stock_open_leaf_count_not_decreased"
    )
    assert closed["continue_guided_search"] is True
    assert closed["reason"] == "root_b4_stock_boundary_reached"
    assert closed["semantics"][
        "root_b4_is_a_portfolio_milestone_not_a_frontier_queue_stop"
    ] is True


def test_guided_provider_success_without_parent_route_is_not_progress() -> None:
    audit = evaluate_guided_stock_progress(
        {"best_open_leaf_count": None},
        {"best_open_leaf_count": None, "root_stock_closed": True},
        root_b4_reached=False,
    )

    assert audit["progressed"] is False
    assert audit["continue_guided_search"] is True
    assert audit["retry_same_frontier"] is False
    assert audit["reason"] == "target_rooted_parent_route_progress_unavailable"
