from cascade_planner.application.route_reconciliation import (
    compile_route_reconciliation,
)


def _family(
    route_id: str,
    steps: int,
    open_leaves: int,
    *,
    current_steps: int | None = None,
) -> dict:
    family = {
        "route_family_id": route_id,
        "title": route_id,
        "aizynthfinder_strategy_search": {
            "path_route_step_count": steps,
            "path_route_projection_complete": True,
            "selected_open_leaves": open_leaves,
            "selected_solved": open_leaves == 0,
        },
    }
    if current_steps is not None:
        family["steps"] = [
            {"step_id": f"{route_id}:current:{index}"} for index in range(current_steps)
        ]
    return family


def _record(route_id: str, proposal_id: str, *, materialized: bool, reasons=()):
    return {
        "materialization": {"materialized": materialized},
        "admission": {"accepted": materialized, "reasons": list(reasons)},
        "validation": {"accepted": False, "reasons": []},
        "origin_records": [{"route_family_id": route_id, "proposal_id": proposal_id}],
    }


def test_reconciliation_keeps_builder_quarantine_separate_from_final_stock_state() -> None:
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
                *[_record("family:solved", f"p{i}", materialized=True) for i in range(5)],
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
    assert rows["family:gap"]["classification"] == "stock_closure_open"
    assert rows["family:gap"]["builder_quarantined_candidate_count"] == 1
    assert rows["family:gap"]["builder_quarantine_reasons"] == [
        "critic_reactionjson_product_binding_mismatch"
    ]
    assert rows["family:stock"]["classification"] == "stock_closure_open"
    assert result["builder_quarantined_route_count"] == 1


def test_reconciliation_does_not_treat_editor_revision_as_materialization_gap() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [
                        _family(
                            "family:edited",
                            18,
                            2,
                            current_steps=10,
                        )
                    ]
                },
            }
        ],
        lifecycle={
            "records": [_record("family:edited", f"p{i}", materialized=True) for i in range(10)]
        },
        paper_equivalent={
            "reached_routes": [{"route_family_id": "family:edited"}],
            "solved_routes": [],
        },
    )

    row = result["routes"][0]
    assert row["strategy_search_projected_step_count"] == 18
    assert row["builder_projected_step_count"] == 10
    assert row["host_materialized_candidate_count"] == 10
    assert row["builder_quarantined_candidate_count"] == 0
    assert row["classification"] == "stock_closure_open"
    assert result["builder_quarantined_route_count"] == 0


def test_reconciliation_uses_host_assembled_skeleton_for_builder_depth() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [_family("family:assembled", 1, 0)],
                    "multi_step_skeletons": [
                        {
                            "skeleton_id": "skeleton:assembled",
                            "route_family_id": "family:assembled",
                            "steps": [
                                {"step_id": "step:1"},
                                {"step_id": "step:2"},
                                {"step_id": "step:3"},
                            ],
                        }
                    ],
                },
            }
        ],
        lifecycle={
            "records": [
                _record("family:assembled", f"p{i}", materialized=True)
                for i in range(3)
            ]
        },
        paper_equivalent={
            "reached_routes": [{"route_family_id": "family:assembled"}],
            "solved_routes": [{"route_family_id": "family:assembled"}],
        },
    )

    row = result["routes"][0]
    assert row["strategy_search_projected_step_count"] == 1
    assert row["builder_projected_step_count"] == 3


def test_reconciliation_is_diagnostic_only() -> None:
    result = compile_route_reconciliation([], {"records": []}, {})
    assert result["routes"] == []
    assert result["semantics"]["does_not_change_paper_equivalent_metric"] is True


def test_reconciliation_joins_director_alias_to_canonical_family_id() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {"route_families": [_family("codex:family:1", 1, 1)]},
            }
        ],
        lifecycle={
            "records": [
                {
                    "materialization": {"materialized": True},
                    "admission": {"accepted": True, "reasons": []},
                    "validation": {"accepted": False, "reasons": []},
                    "route_family_ids": ["route-family:canonical-1"],
                    "origin_records": [
                        {
                            "route_family_id": "codex:family:1",
                            "canonical_route_family_ids": ["route-family:canonical-1"],
                        }
                    ],
                }
            ]
        },
        paper_equivalent={
            "reached_routes": [{"route_family_id": "route-family:canonical-1"}],
            "solved_routes": [],
        },
    )

    row = result["routes"][0]
    assert row["canonical_route_family_ids"] == ["route-family:canonical-1"]
    assert row["paper_reached"] is True
    assert row["classification"] == "stock_closure_open"


def test_reconciliation_does_not_merge_shared_record_families_across_origins() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [
                        _family("codex:sequential:family:1", 2, 0),
                        _family("codex:sequential:family:2", 1, 0),
                    ]
                },
            }
        ],
        lifecycle={
            "records": [
                {
                    "materialization": {"materialized": True},
                    "admission": {"accepted": True, "reasons": []},
                    "validation": {"accepted": False, "reasons": []},
                    "route_family_ids": [
                        "route-family:canonical-1",
                        "route-family:canonical-2",
                    ],
                    "origin_records": [
                        {
                            "route_family_id": "codex:sequential:family:1",
                            "canonical_route_family_ids": ["route-family:canonical-1"],
                        },
                        {
                            "route_family_id": "codex:sequential:family:2",
                            "canonical_route_family_ids": ["route-family:canonical-2"],
                        },
                    ],
                }
            ]
        },
        paper_equivalent={
            "reached_routes": [
                {
                    "route_family_id": "route-family:canonical-1",
                    "edge_ids": ["edge:1", "edge:2"],
                },
                {
                    "route_family_id": "route-family:canonical-2",
                    "edge_ids": ["edge:3"],
                },
            ],
            "solved_routes": [
                {"route_family_id": "route-family:canonical-1"},
                {"route_family_id": "route-family:canonical-2"},
            ],
        },
    )

    rows = {row["route_family_id"]: row for row in result["routes"]}
    assert rows["codex:sequential:family:1"]["canonical_route_family_ids"] == [
        "route-family:canonical-1"
    ]
    assert rows["codex:sequential:family:1"]["final_canonical_edge_count"] == 2
    assert rows["codex:sequential:family:2"]["canonical_route_family_ids"] == [
        "route-family:canonical-2"
    ]
    assert rows["codex:sequential:family:2"]["final_canonical_edge_count"] == 1


def test_reconciliation_narrows_historical_origin_union_from_shared_edge() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [
                        _family("codex:sequential:family:1", 2, 0),
                        _family("codex:sequential:family:2", 1, 0),
                    ]
                },
            }
        ],
        lifecycle={
            "records": [
                {
                    "materialization": {"materialized": True},
                    "admission": {"accepted": True, "reasons": []},
                    "validation": {"accepted": False, "reasons": []},
                    "route_family_ids": [
                        "route-family:canonical-1",
                        "route-family:canonical-2",
                    ],
                    "origin_records": [
                        {
                            "route_family_id": "codex:sequential:family:1",
                            "canonical_route_family_ids": [
                                "route-family:canonical-1",
                                "route-family:canonical-2",
                            ],
                        },
                        {
                            "route_family_id": "codex:sequential:family:1",
                            "canonical_route_family_ids": [
                                "route-family:canonical-1"
                            ],
                        },
                        {
                            "route_family_id": "codex:sequential:family:2",
                            "canonical_route_family_ids": [
                                "route-family:canonical-1",
                                "route-family:canonical-2",
                            ],
                        },
                        {
                            "route_family_id": "codex:sequential:family:2",
                            "canonical_route_family_ids": [
                                "route-family:canonical-2"
                            ],
                        },
                    ],
                }
            ]
        },
        paper_equivalent={
            "reached_routes": [
                {
                    "route_family_id": "route-family:canonical-1",
                    "edge_ids": ["edge:1", "edge:2"],
                },
                {
                    "route_family_id": "route-family:canonical-2",
                    "edge_ids": ["edge:3"],
                },
            ],
            "solved_routes": [
                {"route_family_id": "route-family:canonical-1"},
                {"route_family_id": "route-family:canonical-2"},
            ],
        },
    )

    rows = {row["route_family_id"]: row for row in result["routes"]}
    assert rows["codex:sequential:family:1"]["canonical_route_family_ids"] == [
        "route-family:canonical-1"
    ]
    assert rows["codex:sequential:family:1"]["final_canonical_edge_count"] == 2
    assert rows["codex:sequential:family:2"]["canonical_route_family_ids"] == [
        "route-family:canonical-2"
    ]
    assert rows["codex:sequential:family:2"]["final_canonical_edge_count"] == 1


def test_reconciliation_reports_final_canonical_route_not_builder_history() -> None:
    graph = {
        "target_molecule_id": "m:target",
        "edges": {
            "e:1": {
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:a", "m:side"],
            },
            "e:2": {
                "product_molecule_id": "m:a",
                "precursor_molecule_ids": ["m:b"],
            },
            "e:3": {
                "product_molecule_id": "m:b",
                "precursor_molecule_ids": ["m:c"],
            },
        },
    }
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {"route_families": [_family("family:closed", 2, 1, current_steps=2)]},
            }
        ],
        lifecycle={
            "records": [
                _record("family:closed", "draft:ok", materialized=True),
                _record(
                    "family:closed",
                    "draft:rejected",
                    materialized=False,
                    reasons=("reactionjson_replay_failed",),
                ),
            ]
        },
        paper_equivalent={
            "reached_routes": [
                {
                    "route_family_id": "family:closed",
                    "edge_ids": ["e:1", "e:2", "e:3"],
                }
            ],
            "solved_routes": [
                {
                    "route_family_id": "family:closed",
                    "edge_ids": ["e:1", "e:2", "e:3"],
                }
            ],
        },
        graph=graph,
    )

    row = result["routes"][0]
    assert row["builder_projected_step_count"] == 2
    assert row["host_materialized_candidate_count"] == 1
    assert row["builder_quarantined_candidate_count"] == 1
    assert row["final_canonical_edge_count"] == 3
    assert row["final_longest_linear_sequence"] == 3
    assert row["final_stock_closed"] is True
    assert row["classification"] == "paper_equivalent_solved"


def test_reconciliation_follows_selected_final_repair_successor() -> None:
    result = compile_route_reconciliation(
        [
            {
                "status": "accepted",
                "plan": {
                    "route_families": [
                        _family("director:branch:1", 4, 0, current_steps=4)
                    ]
                },
            }
        ],
        lifecycle={
            "records": [
                {
                    **_record("director:branch:1", "builder:1", materialized=True),
                    "origin_records": [
                        {
                            "route_family_id": "director:branch:1",
                            "proposal_id": "builder:1",
                            "canonical_route_family_ids": ["route:old"],
                        }
                    ],
                }
            ]
        },
        paper_equivalent={
            "reached_routes": [
                {"route_family_id": "route:new", "edge_ids": ["edge:new"]}
            ],
            "solved_routes": [
                {"route_family_id": "route:new", "edge_ids": ["edge:new"]}
            ],
        },
        graph={
            "target_molecule_id": "m:target",
            "edges": {
                "edge:new": {
                    "product_molecule_id": "m:target",
                    "precursor_molecule_ids": ["m:leaf"],
                }
            },
            "route_families": {
                "route:old": {"selected": False, "edge_ids": ["edge:old"]},
                "route:new": {
                    "selected": True,
                    "supersedes_route_family_id": "route:old",
                    "edge_ids": ["edge:new"],
                    "chemical_critic": {
                        "status": "viable",
                        "review_state": "complete",
                        "route_overall_evaluation": "Coherent repaired route.",
                    },
                },
            },
        },
    )

    row = result["routes"][0]
    assert row["origin_canonical_route_family_ids"] == ["route:old"]
    assert row["canonical_route_family_ids"] == ["route:new"]
    assert row["repair_successor_lineages"] == [
        {
            "origin_route_family_id": "route:old",
            "current_route_family_id": "route:new",
            "route_family_ids": ["route:old", "route:new"],
            "ambiguous": False,
        }
    ]
    assert row["final_canonical_edge_count"] == 1
    assert row["final_stock_closed"] is True
    assert row["final_critic_status"] == "viable"
    assert row["final_route_overall_evaluation"] == "Coherent repaired route."
