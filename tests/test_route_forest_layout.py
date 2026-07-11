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
