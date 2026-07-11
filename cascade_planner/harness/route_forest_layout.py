"""Deterministic logical layouts for route-forest dependency graphs.

The compiler owns chemical truth.  This module only projects that truth into
stable logical ranks which a delivery client can turn into pixels.  It never
infers an edge from array adjacency and it deliberately keeps branch-local
lanes separate from the shared canonical overlay.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any


LAYOUT_SCHEMA_VERSION = "route_forest_layout.v1"
BRANCH_LANE_SCHEMA_VERSION = "route_forest_branch_lanes.v1"

_PROOF_RANK = {
    "L0_rejected": 0,
    "L0_advisory": 1,
    "L0_materialized": 2,
    "L1_graph_stock_closed": 3,
    "L1_graph_and_stock_closed": 3,
    "L2_mapping_consistent": 4,
    "L2_reaction_validated": 5,
    "L3_precedent_supported": 6,
    "L4_procurement_ready": 7,
}

_BRANCH_KIND_RANK = {
    "stitched_verified_route": 0,
    "direct_verified_route": 1,
    "proof_eligible_portfolio_route": 2,
    "validated_replacement_route": 3,
    "subgoal_verified_route": 4,
    "exact_literature": 5,
    "process_evidence": 6,
    "visual_chain": 7,
    "route_consensus_graph": 8,
    "route_consensus": 9,
    "retrosynthetic_proposal": 10,
    "broad_template": 11,
    "diagnostic_failure": 12,
}

_BRANCH_KIND_LABEL = {
    "stitched_verified_route": "拼接验证路线",
    "direct_verified_route": "已验证路线",
    "proof_eligible_portfolio_route": "Proof-eligible portfolio",
    "validated_replacement_route": "后端重验替换路线",
    "subgoal_verified_route": "子目标闭合",
    "exact_literature": "精确文献路线",
    "process_evidence": "过程证据",
    "visual_chain": "图像文献链",
    "route_consensus_graph": "Codex 多步路线",
    "route_consensus": "多信源共识",
    "retrosynthetic_proposal": "逆合成提案",
    "broad_template": "通用模板",
    "diagnostic_failure": "诊断与未解决项",
}


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dependency_layout_projection(
    graph: Mapping[str, Any],
    *,
    barycentric_sweeps: int = 4,
) -> dict[str, Any]:
    """Build a permutation-invariant logical layout for an explicit graph.

    Strongly connected components are condensed before ranking.  Disconnected
    components remain explicit so a renderer can pack them without unrelated
    nodes being interleaved.  Fixed-count barycentric sweeps reduce crossings;
    every tie is resolved by stable text and graph-node identifiers.
    """

    node_records = list(graph.get("nodes") or [])
    if any(not isinstance(row, Mapping) for row in node_records):
        raise ValueError("dependency_graph_node_record_invalid")
    raw_nodes = [dict(row) for row in node_records]
    raw_node_ids = [str(row.get("graph_node_id") or "") for row in raw_nodes]
    if any(not node_id for node_id in raw_node_ids):
        raise ValueError("dependency_graph_node_id_missing")
    if len(set(raw_node_ids)) != len(raw_node_ids):
        raise ValueError("dependency_graph_node_id_duplicate")
    nodes_by_id = {str(row.get("graph_node_id") or ""): row for row in raw_nodes}
    node_ids = sorted(nodes_by_id)
    valid_ids = set(node_ids)
    edge_records = list(graph.get("edges") or [])
    if any(not isinstance(row, Mapping) for row in edge_records):
        raise ValueError("dependency_graph_edge_record_invalid")
    raw_edges = [dict(row) for row in edge_records]
    edge_ids = [str(row.get("edge_id") or "") for row in raw_edges]
    if any(not edge_id for edge_id in edge_ids):
        raise ValueError("dependency_graph_edge_id_missing")
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("dependency_graph_edge_id_duplicate")
    for edge in raw_edges:
        source = str(edge.get("source_graph_node_id") or "")
        target = str(edge.get("target_graph_node_id") or "")
        if source not in valid_ids or target not in valid_ids:
            raise ValueError("dependency_graph_edge_endpoint_missing")
    edges = sorted(raw_edges, key=_edge_sort_key)

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    reverse: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    incident_branches: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source_graph_node_id") or "")
        target = str(edge.get("target_graph_node_id") or "")
        adjacency[source].add(target)
        reverse[target].add(source)
        branch_id = str(edge.get("branch_id") or "")
        if branch_id:
            incident_branches[source].add(branch_id)
            incident_branches[target].add(branch_id)

    scc_node_groups = _strongly_connected_components(node_ids, adjacency, reverse)
    scc_id_by_node: dict[str, str] = {}
    scc_rows: list[dict[str, Any]] = []
    for member_ids in scc_node_groups:
        scc_id = _stable_identifier("scc", member_ids)
        cyclic = len(member_ids) > 1 or any(
            node_id in adjacency[node_id] for node_id in member_ids
        )
        for node_id in member_ids:
            scc_id_by_node[node_id] = scc_id
        scc_rows.append(
            {
                "scc_id": scc_id,
                "node_ids": member_ids,
                "cyclic": cyclic,
            }
        )
    scc_rows.sort(key=lambda row: str(row["scc_id"]))
    scc_by_id = {str(row["scc_id"]): row for row in scc_rows}

    scc_adjacency: dict[str, set[str]] = {str(row["scc_id"]): set() for row in scc_rows}
    scc_reverse: dict[str, set[str]] = {str(row["scc_id"]): set() for row in scc_rows}
    for source, targets in adjacency.items():
        source_scc = scc_id_by_node[source]
        for target in targets:
            target_scc = scc_id_by_node[target]
            if source_scc == target_scc:
                continue
            scc_adjacency[source_scc].add(target_scc)
            scc_reverse[target_scc].add(source_scc)

    scc_layers = _longest_path_layers(scc_adjacency, scc_reverse)
    components = _undirected_components(node_ids, adjacency, reverse)
    components.sort(key=lambda member_ids: (-len(member_ids), member_ids[0]))
    component_id_by_node: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for component_order, member_ids in enumerate(components):
        component_id = _stable_identifier("component", member_ids)
        for node_id in member_ids:
            component_id_by_node[node_id] = component_id
        component_rows.append(
            {
                "component_id": component_id,
                "component_order": component_order,
                "node_ids": member_ids,
                "node_count": len(member_ids),
                "cyclic": any(
                    scc_by_id[scc_id_by_node[node_id]]["cyclic"]
                    for node_id in member_ids
                ),
            }
        )

    node_layers = {node_id: scc_layers[scc_id_by_node[node_id]] for node_id in node_ids}
    component_orders = {
        str(row["component_id"]): int(row["component_order"]) for row in component_rows
    }
    component_local_orders: dict[str, int] = {}
    for component in component_rows:
        member_ids = list(component["node_ids"])
        ordered = _stable_barycentric_order(
            member_ids,
            nodes_by_id,
            adjacency,
            reverse,
            node_layers,
            sweeps=max(0, int(barycentric_sweeps)),
        )
        for layer_nodes in ordered.values():
            for local_order, node_id in enumerate(layer_nodes):
                component_local_orders[node_id] = local_order

    global_layer_buckets: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        global_layer_buckets[node_layers[node_id]].append(node_id)
    global_order: dict[str, int] = {}
    for layer, bucket in global_layer_buckets.items():
        bucket.sort(
            key=lambda node_id: (
                component_orders[component_id_by_node[node_id]],
                component_local_orders.get(node_id, 0),
                node_id,
            )
        )
        for order, node_id in enumerate(bucket):
            global_order[node_id] = order

    node_rows = []
    for node_id in node_ids:
        scc_id = scc_id_by_node[node_id]
        scc = scc_by_id[scc_id]
        component_id = component_id_by_node[node_id]
        node_rows.append(
            {
                "graph_node_id": node_id,
                "component_id": component_id,
                "component_order": component_orders[component_id],
                "scc_id": scc_id,
                "scc_cyclic": bool(scc["cyclic"]),
                "layer": node_layers[node_id],
                "order": global_order[node_id],
                "component_local_order": component_local_orders.get(node_id, 0),
                "incident_branch_ids": sorted(incident_branches[node_id]),
                "shared_across_branches": len(incident_branches[node_id]) > 1,
            }
        )

    projection: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "algorithm": "scc_condensation_longest_path_fixed_barycentric.v1",
        "deterministic_tie_break": "casefolded_label_then_graph_node_id",
        "barycentric_sweeps": max(0, int(barycentric_sweeps)),
        "node_count": len(node_rows),
        "edge_count": len(edges),
        "component_count": len(component_rows),
        "scc_count": len(scc_rows),
        "cyclic_scc_count": sum(bool(row["cyclic"]) for row in scc_rows),
        "max_layer": max(node_layers.values(), default=0),
        "nodes": node_rows,
        "components": component_rows,
        "strongly_connected_components": scc_rows,
        "semantics": {
            "edges": "explicit_source_and_target_ids_only",
            "array_adjacency": "never_creates_an_edge",
            "coordinates": "logical_ranks_only",
            "cycles": "condensed_before_layer_assignment",
        },
    }
    projection["layout_sha256"] = canonical_sha256(projection)
    return projection


def build_branch_lane_projection(
    branches: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project every listed branch into a stable lane using explicit edges."""

    if not isinstance(graph, Mapping):
        raise ValueError("branch_lane_dependency_graph_invalid")
    branch_rows = _validated_identified_rows(
        branches, field="branch_id", scope="branch_lane_branch"
    )
    step_rows = _validated_identified_rows(
        steps, field="step_id", scope="branch_lane_step"
    )
    view_rows = _validated_identified_rows(
        graph.get("branch_views") or (),
        field="branch_id",
        scope="branch_lane_view",
    )
    # Validate the complete shared graph before selecting any lane subset.  A
    # branch-only projection must never hide malformed unused nodes or edges.
    build_dependency_layout_projection(graph)

    step_by_id = {str(row["step_id"]): row for row in step_rows}
    branch_by_id = {str(row["branch_id"]): row for row in branch_rows}
    views = {str(row["branch_id"]): row for row in view_rows}
    for view in view_rows:
        branch_id = str(view["branch_id"])
        if branch_id not in branch_by_id:
            raise ValueError("branch_lane_view_branch_id_unknown")
    for branch in branch_rows:
        branch_id = str(branch["branch_id"])
        branch_step_ids = _validated_string_ids(
            branch.get("step_ids") or (), scope="branch_lane_branch_step"
        )
        for step_id in branch_step_ids:
            if step_id not in step_by_id:
                raise ValueError("branch_lane_branch_step_id_unknown")
            step_branch_id = str(step_by_id[step_id].get("branch_id") or "")
            if step_branch_id and step_branch_id != branch_id:
                raise ValueError("branch_lane_step_branch_id_mismatch")
    for view in view_rows:
        branch_id = str(view["branch_id"])
        view_step_ids = _validated_string_ids(
            view.get("topological_step_ids")
            or view.get("step_ids")
            or branch_by_id[branch_id].get("step_ids")
            or (),
            scope="branch_lane_view_step",
        )
        for step_id in view_step_ids:
            if step_id not in step_by_id:
                raise ValueError("branch_lane_view_step_id_unknown")
            step_branch_id = str(step_by_id[step_id].get("branch_id") or "")
            if step_branch_id and step_branch_id != branch_id:
                raise ValueError("branch_lane_view_step_branch_id_mismatch")
    graph_nodes = {
        str(row.get("graph_node_id") or ""): dict(row)
        for row in graph.get("nodes") or []
    }
    edges = [dict(row) for row in graph.get("edges") or []]

    lane_rows: list[dict[str, Any]] = []
    for branch in branch_rows:
        branch_id = str(branch.get("branch_id") or "")
        view = views.get(branch_id, {})
        step_ids = _validated_string_ids(
            view.get("topological_step_ids")
            or view.get("step_ids")
            or branch.get("step_ids")
            or (),
            scope="branch_lane_selected_step",
        )
        step_set = set(step_ids)
        branch_edges = sorted(
            (
                edge
                for edge in edges
                if str(edge.get("reaction_step_id") or "") in step_set
                and str(edge.get("branch_id") or "") in {"", branch_id}
            ),
            key=_edge_sort_key,
        )
        graph_node_ids = {
            str(edge.get("source_graph_node_id") or "") for edge in branch_edges
        } | {str(edge.get("target_graph_node_id") or "") for edge in branch_edges}
        for graph_node_id, node in graph_nodes.items():
            if (
                node.get("node_type") == "reaction"
                and str(node.get("reaction_step_id") or "") in step_set
            ):
                graph_node_ids.add(graph_node_id)
        lane_graph = {
            "nodes": [
                graph_nodes[node_id]
                for node_id in sorted(graph_node_ids)
                if node_id in graph_nodes
            ],
            "edges": branch_edges,
        }
        lane_layout = build_dependency_layout_projection(lane_graph)
        kind = str(branch.get("kind") or "unspecified")
        proof_tier = _branch_proof_tier(branch, step_ids, step_by_id)
        category = _branch_category(branch, kind)
        lane_rows.append(
            {
                "branch_id": branch_id,
                "title": str(branch.get("title") or branch_id),
                "kind": kind,
                "kind_label": _BRANCH_KIND_LABEL.get(
                    kind, kind.replace("_", " ").title()
                ),
                "category": category,
                "proof_tier": proof_tier,
                "proof_rank": _PROOF_RANK.get(proof_tier, -1),
                "is_primary": bool(branch.get("is_primary")),
                "listed": branch.get("listed") is not False,
                "solved": branch.get("solved") is True,
                "executable": branch.get("executable") is True,
                "advisory_only": branch.get("advisory_only") is not False,
                "synthesis_class": str(branch.get("synthesis_class") or "unspecified"),
                "source_refs": sorted(_dedupe_strings(branch.get("source_refs") or [])),
                "step_ids": step_ids,
                "edge_ids": [str(edge.get("edge_id") or "") for edge in branch_edges],
                "graph_node_ids": sorted(graph_node_ids),
                "node_layout": [
                    {
                        "graph_node_id": str(row.get("graph_node_id") or ""),
                        "component_order": int(row.get("component_order") or 0),
                        "layer": int(row.get("layer") or 0),
                        "order": int(row.get("order") or 0),
                        "scc_cyclic": bool(row.get("scc_cyclic")),
                    }
                    for row in lane_layout.get("nodes") or []
                ],
                "component_count": int(lane_layout.get("component_count") or 0),
                "max_layer": int(lane_layout.get("max_layer") or 0),
                "dependency_count": len(view.get("dependencies") or []),
                "acyclic": view.get("acyclic") is not False,
                "all_leaves_stock_bound": view.get("all_leaves_stock_bound") is True,
            }
        )

    lane_rows.sort(key=_branch_lane_sort_key)
    for lane_index, lane in enumerate(lane_rows):
        lane["lane_index"] = lane_index

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane in lane_rows:
        grouped[str(lane["kind"])].append(lane)
    group_rows = []
    for kind in sorted(
        grouped, key=lambda value: (_BRANCH_KIND_RANK.get(value, 999), value)
    ):
        lanes = grouped[kind]
        group_rows.append(
            {
                "group_id": f"kind:{kind}",
                "kind": kind,
                "label": _BRANCH_KIND_LABEL.get(kind, kind.replace("_", " ").title()),
                "branch_ids": [str(lane["branch_id"]) for lane in lanes],
                "branch_count": len(lanes),
                "step_count": sum(len(lane["step_ids"]) for lane in lanes),
                "categories": sorted({str(lane["category"]) for lane in lanes}),
                "proof_tiers": sorted(
                    {str(lane["proof_tier"]) for lane in lanes},
                    key=lambda value: (-_PROOF_RANK.get(value, -1), value),
                ),
            }
        )

    projection: dict[str, Any] = {
        "schema_version": BRANCH_LANE_SCHEMA_VERSION,
        "algorithm": "explicit_branch_dependency_lanes.v1",
        "branch_count": len(lane_rows),
        "listed_branch_count": sum(bool(row["listed"]) for row in lane_rows),
        "groups": group_rows,
        "lanes": lane_rows,
        "semantics": {
            "shared_molecules": "visual_aliases_keep_canonical_graph_node_id",
            "edges": "selected_only_by_explicit_reaction_step_binding",
            "array_adjacency": "never_creates_an_edge",
        },
    }
    projection["layout_sha256"] = canonical_sha256(projection)
    return projection


def count_layer_crossings(
    projection: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
) -> int:
    """Count pairwise order inversions between equal layer pairs.

    This metric is intentionally simple and deterministic; it is useful in
    layout regression tests, not a claim about rendered Bézier intersections.
    """

    positions = {
        str(row.get("graph_node_id") or ""): (
            int(row.get("layer") or 0),
            int(row.get("order") or 0),
        )
        for row in projection.get("nodes") or []
        if isinstance(row, Mapping)
    }
    grouped: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source_graph_node_id") or "")
        target = str(edge.get("target_graph_node_id") or "")
        if source not in positions or target not in positions:
            continue
        grouped[(positions[source][0], positions[target][0])].append((source, target))
    crossings = 0
    for pairs in grouped.values():
        for index, (source_a, target_a) in enumerate(pairs):
            for source_b, target_b in pairs[index + 1 :]:
                if source_a == source_b or target_a == target_b:
                    continue
                source_delta = positions[source_a][1] - positions[source_b][1]
                target_delta = positions[target_a][1] - positions[target_b][1]
                if source_delta * target_delta < 0:
                    crossings += 1
    return crossings


def _validated_identified_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    scope: str,
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{scope}_records_invalid")
    rows = list(records)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{scope}_record_invalid")
    copied = [dict(row) for row in rows]
    identifiers = [str(row.get(field) or "") for row in copied]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{scope}_{field}_missing")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{scope}_{field}_duplicate")
    return copied


def _validated_string_ids(values: Any, *, scope: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{scope}_ids_invalid")
    identifiers = [str(value or "") for value in values]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{scope}_id_missing")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{scope}_id_duplicate")
    return identifiers


def _strongly_connected_components(
    node_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
) -> list[list[str]]:
    finish_order: list[str] = []
    seen: set[str] = set()
    for root in node_ids:
        if root in seen:
            continue
        seen.add(root)
        stack: list[tuple[str, int, list[str]]] = [(root, 0, sorted(adjacency[root]))]
        while stack:
            node_id, index, neighbours = stack[-1]
            if index >= len(neighbours):
                finish_order.append(node_id)
                stack.pop()
                continue
            neighbour = neighbours[index]
            stack[-1] = (node_id, index + 1, neighbours)
            if neighbour in seen:
                continue
            seen.add(neighbour)
            stack.append((neighbour, 0, sorted(adjacency[neighbour])))

    components: list[list[str]] = []
    assigned: set[str] = set()
    for root in reversed(finish_order):
        if root in assigned:
            continue
        members: list[str] = []
        stack = [root]
        assigned.add(root)
        while stack:
            node_id = stack.pop()
            members.append(node_id)
            for neighbour in reversed(sorted(reverse[node_id])):
                if neighbour in assigned:
                    continue
                assigned.add(neighbour)
                stack.append(neighbour)
        components.append(sorted(members))
    components.sort(key=lambda members: members[0])
    return components


def _longest_path_layers(
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
) -> dict[str, int]:
    indegree = {node_id: len(reverse[node_id]) for node_id in adjacency}
    queue = sorted(node_id for node_id, count in indegree.items() if count == 0)
    layers = {node_id: 0 for node_id in adjacency}
    while queue:
        node_id = queue.pop(0)
        for target in sorted(adjacency[node_id]):
            layers[target] = max(layers[target], layers[node_id] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if any(
        count for count in indegree.values()
    ):  # pragma: no cover - SCC condensation is a DAG.
        raise ValueError("SCC condensation unexpectedly contains a cycle")
    return layers


def _undirected_components(
    node_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for root in node_ids:
        if root in seen:
            continue
        seen.add(root)
        queue: deque[str] = deque([root])
        members: list[str] = []
        while queue:
            node_id = queue.popleft()
            members.append(node_id)
            neighbours = adjacency[node_id] | reverse[node_id]
            for neighbour in sorted(neighbours):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                queue.append(neighbour)
        components.append(sorted(members))
    return components


def _stable_barycentric_order(
    member_ids: Sequence[str],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
    layers: Mapping[str, int],
    *,
    sweeps: int,
) -> dict[int, list[str]]:
    member_set = set(member_ids)
    buckets: dict[int, list[str]] = defaultdict(list)
    for node_id in member_ids:
        buckets[layers[node_id]].append(node_id)
    for bucket in buckets.values():
        bucket.sort(key=lambda node_id: _node_sort_key(nodes_by_id[node_id], node_id))

    for _ in range(sweeps):
        ranks = _bucket_ranks(buckets)
        for layer in sorted(buckets):
            buckets[layer].sort(
                key=lambda node_id: _barycentric_key(
                    node_id,
                    reverse[node_id] & member_set,
                    ranks,
                    nodes_by_id,
                )
            )
            ranks = _bucket_ranks(buckets)
        for layer in sorted(buckets, reverse=True):
            buckets[layer].sort(
                key=lambda node_id: _barycentric_key(
                    node_id,
                    adjacency[node_id] & member_set,
                    ranks,
                    nodes_by_id,
                )
            )
            ranks = _bucket_ranks(buckets)
    return dict(buckets)


def _bucket_ranks(buckets: Mapping[int, Sequence[str]]) -> dict[str, int]:
    return {
        node_id: index
        for bucket in buckets.values()
        for index, node_id in enumerate(bucket)
    }


def _barycentric_key(
    node_id: str,
    neighbours: set[str],
    ranks: Mapping[str, int],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    ranked = sorted(ranks[neighbour] for neighbour in neighbours if neighbour in ranks)
    if ranked:
        barycentre = sum(ranked) / len(ranked)
        return (
            0,
            barycentre,
            ranks.get(node_id, 0),
            *_node_sort_key(nodes_by_id[node_id], node_id),
        )
    return (1, ranks.get(node_id, 0), *_node_sort_key(nodes_by_id[node_id], node_id))


def _node_sort_key(node: Mapping[str, Any], node_id: str) -> tuple[Any, ...]:
    node_type = str(node.get("node_type") or "")
    type_rank = 0 if node_type == "molecule" else 1 if node_type == "reaction" else 2
    return (type_rank, str(node.get("label") or "").casefold(), node_id)


def _edge_sort_key(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source_graph_node_id") or ""),
        str(edge.get("target_graph_node_id") or ""),
        str(edge.get("edge_id") or ""),
    )


def _stable_identifier(prefix: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256("\0".join(sorted(values)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _branch_proof_tier(
    branch: Mapping[str, Any],
    step_ids: Sequence[str],
    step_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    explicit = str(
        branch.get("weakest_proof_tier")
        or (branch.get("trust_vector") or {}).get("proof_tier")
        or ""
    )
    if explicit:
        return explicit
    tiers = [
        str(
            (step_by_id.get(step_id, {}).get("trust_vector") or {}).get("proof_tier")
            or ""
        )
        for step_id in step_ids
    ]
    tiers = [tier for tier in tiers if tier]
    if not tiers:
        return "L0_advisory"
    return min(tiers, key=lambda tier: (_PROOF_RANK.get(tier, -1), tier))


def _branch_category(branch: Mapping[str, Any], kind: str) -> str:
    verified = (
        branch.get("solved") is True
        and branch.get("executable") is True
        and branch.get("advisory_only") is False
        and branch.get("not_parent_route_proof") is False
    )
    if verified:
        return "verified"
    if kind in {"exact_literature", "process_evidence", "visual_chain"}:
        return "evidence"
    if kind == "diagnostic_failure":
        return "diagnostic"
    return "advisory"


def _branch_lane_sort_key(branch: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _BRANCH_KIND_RANK.get(str(branch.get("kind") or ""), 999),
        0 if branch.get("is_primary") else 1,
        -int(branch.get("proof_rank") or -1),
        str(branch.get("title") or "").casefold(),
        str(branch.get("branch_id") or ""),
    )
