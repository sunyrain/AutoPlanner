from __future__ import annotations

import copy

import pytest

from cascade_planner.harness.route_forest_layout import (
    BRANCH_LANE_SCHEMA_VERSION,
    LAYOUT_SCHEMA_VERSION,
    build_branch_lane_projection,
    build_dependency_layout_projection,
    count_layer_crossings,
)


def _edge(
    edge_id: str, source: str, target: str, *, step: str = "", branch: str = ""
) -> dict:
    return {
        "edge_id": edge_id,
        "source_graph_node_id": source,
        "target_graph_node_id": target,
        "reaction_step_id": step,
        "branch_id": branch,
    }


def _node(
    node_id: str,
    label: str | None = None,
    *,
    node_type: str = "molecule",
    step: str = "",
) -> dict:
    row = {
        "graph_node_id": node_id,
        "node_type": node_type,
        "label": label or node_id,
    }
    if step:
        row["reaction_step_id"] = step
    return row


def test_dependency_layout_is_permutation_invariant_and_digest_stable() -> None:
    graph = {
        "nodes": [
            _node("m:a", "same"),
            _node("r:1", "same", node_type="reaction"),
            _node("m:b", "same"),
            _node("m:x", "detached"),
        ],
        "edges": [
            _edge("e:1", "m:a", "r:1"),
            _edge("e:2", "r:1", "m:b"),
        ],
    }

    expected = build_dependency_layout_projection(graph)
    permuted = {
        "nodes": list(reversed(copy.deepcopy(graph["nodes"]))),
        "edges": list(reversed(copy.deepcopy(graph["edges"]))),
    }

    assert expected == build_dependency_layout_projection(permuted)
    assert expected["schema_version"] == LAYOUT_SCHEMA_VERSION
    assert expected["component_count"] == 2
    assert len(expected["layout_sha256"]) == 64


def test_dependency_layout_condenses_cycles_before_assigning_layers() -> None:
    graph = {
        "nodes": [_node(node_id) for node_id in ("a", "b", "c", "d", "e")],
        "edges": [
            _edge("ab", "a", "b"),
            _edge("ba", "b", "a"),
            _edge("bc", "b", "c"),
            _edge("de", "d", "e"),
        ],
    }

    projection = build_dependency_layout_projection(graph)
    by_id = {row["graph_node_id"]: row for row in projection["nodes"]}
    cyclic = [
        row for row in projection["strongly_connected_components"] if row["cyclic"]
    ]

    assert projection["cyclic_scc_count"] == 1
    assert cyclic[0]["node_ids"] == ["a", "b"]
    assert by_id["a"]["scc_id"] == by_id["b"]["scc_id"]
    assert by_id["a"]["layer"] == by_id["b"]["layer"]
    assert by_id["c"]["layer"] == by_id["b"]["layer"] + 1
    assert projection["component_count"] == 2


def test_fixed_barycentric_sweeps_reduce_a_crossing_fixture() -> None:
    graph = {
        "nodes": [_node(node_id, node_id) for node_id in ("a", "b", "c", "d")],
        "edges": [
            _edge("a-d", "a", "d"),
            _edge("b-c", "b", "c"),
            _edge("a-c", "a", "c"),
        ],
    }

    alphabetical = build_dependency_layout_projection(graph, barycentric_sweeps=0)
    optimised = build_dependency_layout_projection(graph, barycentric_sweeps=4)

    assert count_layer_crossings(alphabetical, graph["edges"]) == 1
    assert count_layer_crossings(optimised, graph["edges"]) == 0


def test_dependency_layout_places_late_feed_root_next_to_its_consumer() -> None:
    graph = {
        "nodes": [
            _node("m:main-start"),
            _node("r:1", node_type="reaction"),
            _node("m:intermediate"),
            _node("m:late-feed"),
            _node("r:2", node_type="reaction"),
            _node("m:product"),
        ],
        "edges": [
            _edge("e1", "m:main-start", "r:1"),
            _edge("e2", "r:1", "m:intermediate"),
            _edge("e3", "m:intermediate", "r:2"),
            _edge("e4", "m:late-feed", "r:2"),
            _edge("e5", "r:2", "m:product"),
        ],
    }

    projection = build_dependency_layout_projection(graph)
    by_id = {row["graph_node_id"]: row for row in projection["nodes"]}

    assert by_id["m:main-start"]["layer"] == 0
    assert by_id["m:intermediate"]["layer"] == 2
    assert by_id["m:late-feed"]["layer"] == 2
    assert by_id["r:2"]["layer"] == 3
    assert projection["algorithm"].startswith("scc_condensation_sink_aligned")


def test_branch_lanes_include_only_explicit_edges_and_have_local_layouts() -> None:
    branch_id = "branch:one"
    nodes = [
        _node("m:a"),
        _node("r:1", node_type="reaction", step="s1"),
        _node("m:b"),
        _node("m:c"),
        _node("r:2", node_type="reaction", step="s2"),
        _node("m:d"),
    ]
    edges = [
        _edge("e1", "m:a", "r:1", step="s1", branch=branch_id),
        _edge("e2", "r:1", "m:b", step="s1", branch=branch_id),
        _edge("e3", "m:c", "r:2", step="s2", branch=branch_id),
        _edge("e4", "r:2", "m:d", step="s2", branch=branch_id),
    ]
    graph = {
        "nodes": nodes,
        "edges": edges,
        "branch_views": [
            {
                "branch_id": branch_id,
                "step_ids": ["s1", "s2"],
                "topological_step_ids": ["s1", "s2"],
                "dependencies": [],
                "acyclic": True,
            }
        ],
    }
    branches = [
        {
            "branch_id": branch_id,
            "title": "Disconnected explicit observations",
            "kind": "route_consensus",
            "step_ids": ["s1", "s2"],
            "listed": True,
            "advisory_only": True,
        }
    ]
    steps = [
        {"step_id": "s1", "trust_vector": {"proof_tier": "L0_materialized"}},
        {"step_id": "s2", "trust_vector": {"proof_tier": "L0_advisory"}},
    ]

    projection = build_branch_lane_projection(branches, graph, steps)
    lane = projection["lanes"][0]

    assert projection["schema_version"] == BRANCH_LANE_SCHEMA_VERSION
    assert set(lane["edge_ids"]) == {"e1", "e2", "e3", "e4"}
    assert lane["dependency_count"] == 0
    assert lane["component_count"] == 2
    assert lane["proof_tier"] == "L0_advisory"
    assert lane["stage_memberships"] == ["suggestion"]
    assert lane["stage_evidence"]["authority_available"] is False
    assert "frontier_ledger_missing" in lane["stage_evidence"]["expanded"][
        "reasons"
    ]
    assert {row["graph_node_id"] for row in lane["node_layout"]} == {
        "m:a",
        "r:1",
        "m:b",
        "m:c",
        "r:2",
        "m:d",
    }
    assert projection["semantics"]["array_adjacency"] == "never_creates_an_edge"


def test_branch_lane_group_order_is_independent_of_input_order() -> None:
    graph = {"nodes": [], "edges": [], "branch_views": []}
    branches = [
        {"branch_id": "proposal", "title": "B", "kind": "retrosynthetic_proposal"},
        {"branch_id": "verified", "title": "A", "kind": "direct_verified_route"},
    ]

    forward = build_branch_lane_projection(branches, graph)
    reverse = build_branch_lane_projection(list(reversed(branches)), graph)

    assert forward == reverse
    assert [row["branch_id"] for row in forward["lanes"]] == ["verified", "proposal"]


def test_consensus_lane_label_requires_independent_multi_source_support() -> None:
    graph = {"nodes": [], "edges": [], "branch_views": []}
    branches = [
        {
            "branch_id": "multi",
            "title": "Independent support",
            "kind": "route_consensus",
            "consensus_scope": "multi_source",
            "multi_source": True,
        },
        {
            "branch_id": "correlated",
            "title": "Several Codex roles",
            "kind": "route_consensus",
            "consensus_scope": "correlated_single_source",
            "multi_source": False,
        },
    ]

    projection = build_branch_lane_projection(branches, graph)
    lanes = {row["branch_id"]: row for row in projection["lanes"]}

    assert lanes["multi"]["kind_label"] == "多信源共识"
    assert lanes["multi"]["multi_source"] is True
    assert lanes["correlated"]["kind_label"] == "相关源共识"
    assert lanes["correlated"]["multi_source"] is False
    assert projection["groups"][0]["label"] == "共识候选"


def test_dense_96_branch_projection_keeps_every_lane_node_and_edge() -> None:
    branches = []
    steps = []
    nodes = []
    edges = []
    views = []
    for index in range(96):
        branch_id = f"branch:{index:03d}"
        step_id = f"step:{index:03d}"
        source_id = f"molecule:source:{index:03d}"
        reaction_id = f"reaction:{index:03d}"
        target_id = "molecule:shared-target"
        branches.append(
            {
                "branch_id": branch_id,
                "title": f"Route {index:03d}",
                "kind": "route_consensus",
                "step_ids": [step_id],
                "listed": True,
            }
        )
        steps.append(
            {"step_id": step_id, "trust_vector": {"proof_tier": "L0_advisory"}}
        )
        nodes.extend(
            [
                _node(source_id),
                _node(reaction_id, node_type="reaction", step=step_id),
            ]
        )
        edges.extend(
            [
                _edge(
                    f"edge:{index:03d}:in",
                    source_id,
                    reaction_id,
                    step=step_id,
                    branch=branch_id,
                ),
                _edge(
                    f"edge:{index:03d}:out",
                    reaction_id,
                    target_id,
                    step=step_id,
                    branch=branch_id,
                ),
            ]
        )
        views.append(
            {
                "branch_id": branch_id,
                "step_ids": [step_id],
                "topological_step_ids": [step_id],
                "dependencies": [],
                "acyclic": True,
            }
        )
    nodes.append(_node("molecule:shared-target"))

    projection = build_branch_lane_projection(
        branches,
        {"nodes": nodes, "edges": edges, "branch_views": views},
        steps,
    )

    assert projection["branch_count"] == 96
    assert len(projection["lanes"]) == 96
    assert sum(len(row["node_layout"]) for row in projection["lanes"]) == 288
    assert sum(len(row["edge_ids"]) for row in projection["lanes"]) == 192
    assert all(row["max_layer"] == 2 for row in projection["lanes"])


def test_dependency_layout_fails_closed_on_duplicate_node_ids() -> None:
    graph = {"nodes": [_node("m:a"), _node("m:a")], "edges": []}

    with pytest.raises(ValueError, match="dependency_graph_node_id_duplicate"):
        build_dependency_layout_projection(graph)


def test_dependency_layout_fails_closed_on_missing_edge_endpoint() -> None:
    graph = {
        "nodes": [_node("m:a")],
        "edges": [_edge("edge:bad", "m:a", "m:missing")],
    }

    with pytest.raises(ValueError, match="dependency_graph_edge_endpoint_missing"):
        build_dependency_layout_projection(graph)


def test_dependency_layout_fails_closed_on_duplicate_edge_ids() -> None:
    graph = {
        "nodes": [_node("m:a"), _node("m:b")],
        "edges": [
            _edge("edge:duplicate", "m:a", "m:b"),
            _edge("edge:duplicate", "m:b", "m:a"),
        ],
    }

    with pytest.raises(ValueError, match="dependency_graph_edge_id_duplicate"):
        build_dependency_layout_projection(graph)


def test_branch_lane_projection_fails_closed_on_duplicate_branch_ids() -> None:
    branches = [
        {"branch_id": "branch:duplicate"},
        {"branch_id": "branch:duplicate"},
    ]

    with pytest.raises(ValueError, match="branch_lane_branch_branch_id_duplicate"):
        build_branch_lane_projection(
            branches, {"nodes": [], "edges": [], "branch_views": []}
        )


def test_branch_lane_projection_fails_closed_on_invalid_or_unknown_steps() -> None:
    graph = {
        "nodes": [],
        "edges": [],
        "branch_views": [{"branch_id": "branch:one", "step_ids": ["missing"]}],
    }

    with pytest.raises(ValueError, match="branch_lane_branch_step_id_unknown"):
        build_branch_lane_projection(
            [{"branch_id": "branch:one", "step_ids": ["missing"]}],
            graph,
            [],
        )

    with pytest.raises(ValueError, match="branch_lane_branch_step_ids_invalid"):
        build_branch_lane_projection(
            [{"branch_id": "branch:one", "step_ids": "step:not-an-array"}],
            {"nodes": [], "edges": [], "branch_views": []},
            [],
        )


def test_branch_lane_projection_validates_complete_graph_before_lane_filtering() -> (
    None
):
    graph = {
        "nodes": [_node("m:valid")],
        "edges": [_edge("edge:hidden-bad", "m:valid", "m:missing")],
        "branch_views": [],
    }

    with pytest.raises(ValueError, match="dependency_graph_edge_endpoint_missing"):
        build_branch_lane_projection([], graph, [])


def _stage_authority(
    *,
    edge_rows: list[dict] | None = None,
    molecule_rows: list[dict] | None = None,
    authoritative: bool = True,
) -> dict:
    return {
        "schema_version": "route_forest_frontier_ledger_view.v1",
        "authoritative": authoritative,
        "content_sha256": "a" * 64,
        "stage_authority": {
            "schema_version": "route_forest_stage_authority.v1",
            "authoritative": authoritative,
            "molecules": list(molecule_rows or []),
            "edges": list(edge_rows or []),
        },
    }


def _stage_matrix_projection(
    *,
    proof_tier: str = "L0_advisory",
    edge_proof_level: int = 1,
    edge_authority: str = "none",
    expansion_succeeded: bool = False,
    expansion_job_ids: list[str] | None = None,
    benchmark_closed: bool = False,
    procurement_closed: bool = False,
    current_observation_ids: list[str] | None = None,
    stock_job_ids: list[str] | None = None,
    host_replay_verified: bool = False,
    all_leaves_stock_bound: bool = False,
    authority: bool = True,
) -> dict:
    branch_id = "branch:matrix"
    step_id = "step:matrix"
    graph = {
        "nodes": [
            {
                **_node("graph:leaf"),
                "molecule_node_id": "mol:leaf",
                "canonical_isomeric_smiles": "C",
            },
            _node("graph:reaction", node_type="reaction", step=step_id),
            {
                **_node("graph:product"),
                "molecule_node_id": "mol:product",
                "canonical_isomeric_smiles": "CC",
            },
        ],
        "edges": [
            _edge(
                "display:in",
                "graph:leaf",
                "graph:reaction",
                step=step_id,
                branch=branch_id,
            ),
            _edge(
                "display:out",
                "graph:reaction",
                "graph:product",
                step=step_id,
                branch=branch_id,
            ),
        ],
        "branch_views": [
            {
                "branch_id": branch_id,
                "step_ids": [step_id],
                "topological_step_ids": [step_id],
                "root_molecule_node_ids": ["mol:leaf"],
                "all_leaves_stock_bound": all_leaves_stock_bound,
                "dependencies": [],
                "acyclic": True,
            }
        ],
    }
    frontier_ledger = _stage_authority(
        authoritative=authority,
        edge_rows=[
            {
                "exact_edge_signature": "edge:matrix",
                "step_ids": [step_id],
                "product_smiles": "CC",
                "precursor_smiles": ["C"],
                "reaction_proof": {
                    "achieved_proof_level": edge_proof_level,
                    "authority": edge_authority,
                    "current_host_reaction_validated": (
                        edge_proof_level >= 2
                        and edge_authority == "current_host_verifier_replay"
                    ),
                    "proof_request_ids": ["proof:matrix"],
                },
            }
        ],
        molecule_rows=[
            {
                "canonical_smiles": "C",
                "node_ids": ["mol:leaf"],
                "work": {
                    "proposal_expansion_succeeded": False,
                    "job_ids": [],
                },
                "stock": {
                    "benchmark_search_boundary_closed": benchmark_closed,
                    "procurement_boundary_closed": procurement_closed,
                    "host_replay_verified": host_replay_verified,
                    "current_observation_ids": list(current_observation_ids or []),
                    "closure_job_ids": list(stock_job_ids or []),
                },
            },
            {
                "canonical_smiles": "CC",
                "node_ids": ["mol:product"],
                "work": {
                    "proposal_expansion_succeeded": expansion_succeeded,
                    "job_ids": list(expansion_job_ids or []),
                },
                "stock": {
                    "benchmark_search_boundary_closed": False,
                    "procurement_boundary_closed": False,
                    "host_replay_verified": False,
                    "current_observation_ids": [],
                    "closure_job_ids": [],
                },
            },
        ],
    )
    return build_branch_lane_projection(
        [
            {
                "branch_id": branch_id,
                "kind": "route_consensus",
                "step_ids": [step_id],
                "weakest_proof_tier": proof_tier,
            }
        ],
        graph,
        [
            {
                "step_id": step_id,
                "branch_id": branch_id,
                "trust_vector": {"proof_tier": proof_tier},
            }
        ],
        frontier_ledger=frontier_ledger,
    )


@pytest.mark.parametrize(
    ("overrides", "memberships", "stock_scope"),
    [
        ({"proof_tier": "L0_rejected"}, [], "none"),
        ({"proof_tier": "L0_materialized"}, ["suggestion"], "none"),
        ({"authority": False}, ["suggestion"], "none"),
        (
            {
                "expansion_succeeded": True,
                "expansion_job_ids": ["expand:1"],
            },
            ["suggestion", "expanded"],
            "none",
        ),
        (
            {
                "proof_tier": "L4_procurement_ready",
                "edge_proof_level": 2,
                "edge_authority": "current_host_verifier_replay",
            },
            ["reaction"],
            "none",
        ),
        (
            {
                "benchmark_closed": True,
                "host_replay_verified": True,
                "current_observation_ids": ["observation:benchmark"],
                "stock_job_ids": ["stock:benchmark"],
            },
            ["suggestion", "stock"],
            "benchmark",
        ),
        (
            {
                "procurement_closed": True,
                "host_replay_verified": True,
                "current_observation_ids": ["observation:procurement"],
                "stock_job_ids": ["stock:procurement"],
            },
            ["suggestion", "stock"],
            "procurement",
        ),
        (
            {
                "proof_tier": "L4_procurement_ready",
                "edge_proof_level": 4,
                "edge_authority": "current_host_verifier_replay",
                "benchmark_closed": True,
                "all_leaves_stock_bound": True,
            },
            ["reaction"],
            "none",
        ),
        (
            {
                "proof_tier": "L4_procurement_ready",
                "edge_proof_level": 4,
                "edge_authority": "current_host_verifier_replay",
                "expansion_succeeded": True,
                "expansion_job_ids": ["expand:1"],
                "procurement_closed": True,
                "host_replay_verified": True,
                "current_observation_ids": ["observation:procurement"],
                "stock_job_ids": ["stock:procurement"],
            },
            ["expanded", "reaction", "stock"],
            "procurement",
        ),
    ],
)
def test_branch_stage_authority_matrix_fails_closed_without_exact_bindings(
    overrides: dict,
    memberships: list[str],
    stock_scope: str,
) -> None:
    projection = _stage_matrix_projection(**overrides)
    lane = projection["lanes"][0]

    assert projection["schema_version"] == "route_forest_branch_lanes.v2"
    assert (
        lane["stage_evidence"]["schema_version"]
        == "route_forest_branch_stage_evidence.v3"
    )
    assert lane["stage_memberships"] == memberships
    assert lane["stage_evidence"]["stock"]["closure_scope"] == stock_scope
    assert lane["stage_evidence"]["stock"]["semantics"][
        "benchmark_is_not_procurement"
    ] is True


def test_expanded_stage_requires_every_step_and_reports_partial_progress() -> None:
    branch_id = "branch:two-step"
    step_ids = ["step:one", "step:two"]
    branch = {
        "branch_id": branch_id,
        "kind": "route_consensus",
        "step_ids": step_ids,
        "weakest_proof_tier": "L0_materialized",
    }
    graph = {
        "nodes": [],
        "edges": [],
        "branch_views": [
            {
                "branch_id": branch_id,
                "step_ids": step_ids,
                "root_molecule_node_ids": [],
            }
        ],
    }
    steps = [
        {"step_id": step_id, "branch_id": branch_id} for step_id in step_ids
    ]
    ledger = _stage_authority(
        edge_rows=[
            {
                "exact_edge_signature": "edge:one",
                "step_ids": ["step:one"],
                "product_smiles": "CC",
            },
            {
                "exact_edge_signature": "edge:two",
                "step_ids": ["step:two"],
                "product_smiles": "CCC",
            },
        ],
        molecule_rows=[
            {
                "canonical_smiles": "CC",
                "work": {
                    "proposal_expansion_succeeded": True,
                    "job_ids": ["expand:one"],
                },
            },
            {
                "canonical_smiles": "CCC",
                "work": {
                    "proposal_expansion_succeeded": False,
                    "job_ids": [],
                },
            },
        ],
    )

    partial = build_branch_lane_projection(
        [branch], graph, steps, frontier_ledger=ledger
    )["lanes"][0]
    expanded = partial["stage_evidence"]["expanded"]

    assert partial["stage_memberships"] == ["suggestion"]
    assert expanded["member"] is False
    assert expanded["fully_expanded"] is False
    assert expanded["partial_expanded"] is True
    assert expanded["matched_step_count"] == 1
    assert expanded["required_step_count"] == 2
    assert expanded["matched_step_ids"] == ["step:one"]
    assert expanded["remaining_step_ids"] == ["step:two"]
    assert "route_only_partially_expanded:1/2" in expanded["reasons"]
    assert expanded["semantics"][
        "partial_expanded_is_non_authoritative_progress"
    ] is True

    complete_ledger = copy.deepcopy(ledger)
    second_work = complete_ledger["stage_authority"]["molecules"][1]["work"]
    second_work["proposal_expansion_succeeded"] = True
    second_work["job_ids"] = ["expand:two"]
    complete = build_branch_lane_projection(
        [branch], graph, steps, frontier_ledger=complete_ledger
    )["lanes"][0]
    complete_expanded = complete["stage_evidence"]["expanded"]

    assert complete["stage_memberships"] == ["suggestion", "expanded"]
    assert complete_expanded["member"] is True
    assert complete_expanded["fully_expanded"] is True
    assert complete_expanded["partial_expanded"] is False
    assert complete_expanded["matched_step_count"] == 2
    assert complete_expanded["required_step_count"] == 2
    assert complete_expanded["remaining_step_ids"] == []


def test_expanded_stage_fails_closed_for_branch_without_steps() -> None:
    branch_id = "branch:empty"
    lane = build_branch_lane_projection(
        [
            {
                "branch_id": branch_id,
                "kind": "route_consensus",
                "step_ids": [],
                "weakest_proof_tier": "L0_advisory",
            }
        ],
        {
            "nodes": [],
            "edges": [],
            "branch_views": [
                {
                    "branch_id": branch_id,
                    "step_ids": [],
                    "root_molecule_node_ids": [],
                }
            ],
        },
        [],
        frontier_ledger=_stage_authority(),
    )["lanes"][0]
    expanded = lane["stage_evidence"]["expanded"]

    assert lane["stage_memberships"] == ["suggestion"]
    assert expanded["member"] is False
    assert expanded["fully_expanded"] is False
    assert expanded["partial_expanded"] is False
    assert expanded["matched_step_count"] == 0
    assert expanded["required_step_count"] == 0
    assert "branch_steps_empty" in expanded["reasons"]


def test_reaction_stage_requires_every_step_to_have_current_host_l2_binding() -> None:
    projection = _stage_matrix_projection(
        proof_tier="L4_procurement_ready",
        edge_proof_level=4,
        edge_authority="current_host_verifier_replay",
    )
    lane = projection["lanes"][0]
    branch = {
        "branch_id": lane["branch_id"],
        "kind": "route_consensus",
        "step_ids": ["step:matrix", "step:unbound"],
        "weakest_proof_tier": "L4_procurement_ready",
    }
    graph = {
        "nodes": [],
        "edges": [],
        "branch_views": [
            {
                "branch_id": lane["branch_id"],
                "step_ids": ["step:matrix", "step:unbound"],
                "root_molecule_node_ids": [],
            }
        ],
    }
    ledger = _stage_authority(
        edge_rows=[
            {
                "exact_edge_signature": "edge:matrix",
                "step_ids": ["step:matrix"],
                "product_smiles": "CC",
                "reaction_proof": {
                    "achieved_proof_level": 4,
                    "authority": "current_host_verifier_replay",
                    "current_host_reaction_validated": True,
                    "proof_request_ids": ["proof:matrix"],
                },
            }
        ],
    )

    strict = build_branch_lane_projection(
        [branch],
        graph,
        [
            {"step_id": "step:matrix", "branch_id": lane["branch_id"]},
            {
                "step_id": "step:unbound",
                "branch_id": lane["branch_id"],
                "trust_vector": {"proof_tier": "L4_procurement_ready"},
            },
        ],
        frontier_ledger=ledger,
    )["lanes"][0]

    assert "reaction" not in strict["stage_memberships"]
    assert strict["stage_evidence"]["reaction"]["matched_step_ids"] == [
        "step:matrix"
    ]
    assert "reaction_edge_binding_missing:step:unbound" in strict[
        "stage_evidence"
    ]["reaction"]["reasons"]


def test_stage_authority_joins_regenerated_display_step_by_explicit_graph_step_id() -> (
    None
):
    branch_id = "branch:display"
    display_step_id = "display:step:1"
    source_step_id = "consensus:step:1"
    ledger = _stage_authority(
        edge_rows=[
            {
                "exact_edge_signature": "edge:source",
                "step_ids": [source_step_id],
                "product_smiles": "CC",
                "precursor_smiles": ["C"],
                "reaction_proof": {
                    "achieved_proof_level": 2,
                    "authority": "current_host_verifier_replay",
                    "current_host_reaction_validated": True,
                    "proof_request_ids": ["proof:source"],
                },
            }
        ],
        molecule_rows=[
            {
                "canonical_smiles": "CC",
                "node_ids": [],
                "work": {
                    "proposal_expansion_succeeded": True,
                    "job_ids": ["expand:source"],
                },
                "stock": {},
            }
        ],
    )

    lane = build_branch_lane_projection(
        [
            {
                "branch_id": branch_id,
                "kind": "route_consensus_graph",
                "step_ids": [display_step_id],
                "weakest_proof_tier": "L2_reaction_validated",
            }
        ],
        {
            "nodes": [],
            "edges": [],
            "branch_views": [
                {
                    "branch_id": branch_id,
                    "step_ids": [display_step_id],
                    "root_molecule_node_ids": [],
                }
            ],
        },
        [
            {
                "step_id": display_step_id,
                "branch_id": branch_id,
                "graph_step_id": source_step_id,
            }
        ],
        frontier_ledger=ledger,
    )["lanes"][0]

    assert lane["stage_memberships"] == ["expanded", "reaction"]
    assert lane["stage_evidence"]["expanded"]["matched_edge_signatures"] == [
        "edge:source"
    ]
    assert lane["stage_evidence"]["reaction"]["matched_step_ids"] == [
        display_step_id
    ]
