from __future__ import annotations

from scripts.project_frozen_route_closure import _route_result_counts


def test_route_result_counts_use_canonical_closed_and_closure_rate_fields() -> None:
    counts = _route_result_counts(
        {
            "route_families": {
                "closed": {
                    "selected": True,
                    "closed": True,
                    "stock_closure_rate": 1.0,
                    "unmaterialized_hypothesis_ids": [],
                },
                "open": {
                    "selected": True,
                    "closed": False,
                    "stock_closure_rate": 0.75,
                    "unmaterialized_hypothesis_ids": ["hypothesis:1"],
                },
                "unselected": {
                    "selected": False,
                    "closed": True,
                    "stock_closure_rate": 1.0,
                    "unmaterialized_hypothesis_ids": ["hypothesis:2"],
                },
            }
        }
    )

    assert counts == {
        "route_count": 3,
        "selected_route_count": 2,
        "selected_closed_route_count": 1,
        "selected_unmaterialized_hypothesis_count": 1,
        "selected_stock_closure_rate_max": 1.0,
    }
