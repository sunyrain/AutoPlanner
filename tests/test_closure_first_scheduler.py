from cascade_planner.application.action_scheduler import schedule_next_action


def _action(action_id: str, kind: str, resource_class: str) -> dict:
    return {
        "action_id": action_id,
        "kind": kind,
        "resource_class": resource_class,
        "base_priority": 1000 if "expand" in kind else 1,
        "deterministic": kind == "host_materialize",
        "dependency_ids": [],
        "metadata": {},
    }


def test_existing_route_closes_before_new_frontier_even_when_expansion_scores_higher() -> None:
    decision = schedule_next_action(
        {
            "actions": [
                _action("expand", "chemenzy_target_expand", "native_search_target"),
                _action("builder", "codex_frontier_expand", "model"),
                _action("stock", "stock_audit", "stock"),
                _action("validate", "reaction_validate", "validation"),
                _action("materialize", "host_materialize", "deterministic"),
            ]
        },
        milestones={
            "target_rooted_route_exists": True,
            "B1_global_multi_route": False,
            "B4_stock_boundary": False,
        },
        resource_availability={},
        available_action_kinds=(
            "chemenzy_target_expand",
            "codex_frontier_expand",
            "stock_audit",
            "reaction_validate",
            "host_materialize",
        ),
    )
    assert decision["pending_route_closure_stage"] == "host_materialize"
    assert decision["selected_action_id"] == "materialize"
    blocked = {row["action_id"]: row["blocked_reasons"] for row in decision["candidates"]}
    assert "route_closure_pipeline_pending:host_materialize" in blocked["expand"]
    assert "route_closure_pipeline_pending:host_materialize" in blocked["builder"]
    assert "earlier_route_closure_stage_pending:host_materialize" in blocked["validate"]
    assert "earlier_route_closure_stage_pending:host_materialize" in blocked["stock"]
