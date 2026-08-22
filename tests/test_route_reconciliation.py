from cascade_planner.application.route_reconciliation import (
    compile_route_reconciliation,
)


def _family(route_id: str, steps: int, open_leaves: int) -> dict:
    return {
        "route_family_id": route_id,
        "title": route_id,
        "aizynthfinder_strategy_search": {
            "path_route_step_count": steps,
            "path_route_projection_complete": True,
            "selected_open_leaves": open_leaves,
            "selected_solved": open_leaves == 0,
        },
    }


def _record(route_id: str, proposal_id: str, *, materialized: bool, reasons=()):
    return {
        "materialization": {"materialized": materialized},
        "admission": {"accepted": materialized, "reasons": list(reasons)},
        "validation": {"accepted": False, "reasons": []},
        "origin_records": [{"route_family_id": route_id, "proposal_id": proposal_id}],
    }


def test_reconciliation_distinguishes_admission_gap_from_stock_open() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [
                        _family("family:solved", 5, 0),
                        _family("family:gap", 4, 1),
                        _family("family:stock", 2, 1),
                    ]
                },
            }
        ],
        lifecycle={
            "records": [
                *[
                    _record("family:solved", f"p{i}", materialized=True)
                    for i in range(5)
                ],
                _record("family:gap", "p0", materialized=True),
                _record(
                    "family:gap",
                    "p1",
                    materialized=False,
                    reasons=("critic_reactionjson_product_binding_mismatch",),
                ),
                _record("family:stock", "p0", materialized=True),
                _record("family:stock", "p1", materialized=True),
            ]
        },
        paper_equivalent={
            "reached_routes": [
                {"route_family_id": "family:solved"},
                {"route_family_id": "family:gap"},
                {"route_family_id": "family:stock"},
            ],
            "solved_routes": [{"route_family_id": "family:solved"}],
        },
    )
    rows = {row["route_family_id"]: row for row in result["routes"]}
    assert rows["family:solved"]["classification"] == "paper_equivalent_solved"
    assert rows["family:gap"]["classification"] == "materialization_admission_gap"
    assert rows["family:gap"]["materialization_gap_step_count"] == 3
    assert rows["family:gap"]["materialization_gap_reasons"] == [
        "critic_reactionjson_product_binding_mismatch"
    ]
    assert rows["family:stock"]["classification"] == "stock_closure_open"
    assert result["materialization_gap_route_count"] == 1


def test_reconciliation_is_diagnostic_only() -> None:
    result = compile_route_reconciliation([], {"records": []}, {})
    assert result["routes"] == []
    assert result["semantics"]["does_not_change_paper_equivalent_metric"] is True
