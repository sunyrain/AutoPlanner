"""Proof-aware, bounded read model for the retrosynthesis workbench.

The workbench never consumes the exploratory hypergraph directly.  This module
projects one canonical graph revision and one proof portfolio into a compact
read model, then computes entity deltas between revisions.  It owns no
scientific state and cannot promote a hypothesis or proof level.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ROUTE_WORKBENCH_SCHEMA = "retrosynthesis_route_workbench.v1"
ROUTE_WORKBENCH_DELTA_SCHEMA = "retrosynthesis_route_workbench_delta.v1"
MAX_VISIBLE_ROUTES = 5
MAX_VISIBLE_HYPOTHESES = 48

PROOF_VISUALS: dict[int, dict[str, str]] = {
    0: {"name": "L0_hypothesis", "color": "#e76f51", "tone": "proposal"},
    1: {"name": "L1_structural_materialized", "color": "#8b5cf6", "tone": "materialized"},
    2: {"name": "L2_reaction_validated", "color": "#3b82f6", "tone": "validated"},
    3: {"name": "L3_exact_source", "color": "#0f9f8f", "tone": "supported"},
    4: {"name": "L4_procurement_ready", "color": "#16a34a", "tone": "closed"},
}


class RouteWorkbenchProjectionError(ValueError):
    """The graph/portfolio pair cannot form one authoritative UI revision."""


def compile_route_workbench(
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    *,
    campaign_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a canonical graph and proof portfolio into a bounded read model."""

    _validate_bindings(graph, portfolio)
    graph_revision = int(graph.get("revision") or 0)
    molecules = dict(graph.get("molecules") or {})
    edges = dict(graph.get("edges") or {})
    edge_proofs = dict(portfolio.get("edge_proofs") or {})
    leaf_proofs = dict(portfolio.get("leaf_proofs") or {})
    selected = [
        dict(value)
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping)
    ][:MAX_VISIBLE_ROUTES]

    selected_edge_ids = {
        str(edge_id)
        for route in selected
        for edge_id in route.get("edge_ids") or []
        if str(edge_id) in edges
    }
    selected_molecule_ids = {
        str(molecule_id)
        for edge_id in selected_edge_ids
        for molecule_id in (
            edges[edge_id].get("product_molecule_id"),
            *(edges[edge_id].get("precursor_molecule_ids") or []),
        )
        if str(molecule_id) in molecules
    }
    target_id = str(graph.get("target_molecule_id") or "")
    if target_id in molecules:
        selected_molecule_ids.add(target_id)

    molecule_rows = {
        molecule_id: _molecule_row(
            molecule_id,
            molecules[molecule_id],
            graph=graph,
            leaf_proof=leaf_proofs.get(molecule_id),
            target_id=target_id,
        )
        for molecule_id in sorted(selected_molecule_ids)
    }
    edge_rows = {
        edge_id: _edge_row(
            edge_id,
            edges[edge_id],
            proof=dict(edge_proofs.get(edge_id) or {}),
            graph=graph,
        )
        for edge_id in sorted(selected_edge_ids)
    }
    deficit_rows = [
        _copy_json(value)
        for value in portfolio.get("deficits") or []
        if isinstance(value, Mapping)
    ]
    route_rows = {
        str(route["route_id"]): _route_row(
            route,
            edge_rows=edge_rows,
            deficits=deficit_rows,
        )
        for route in selected
        if str(route.get("route_id") or "")
    }
    hypotheses = _hypothesis_rows(graph)
    modules = _module_rows(portfolio, selected_route_ids=set(route_rows))
    shared_intermediates = _shared_intermediate_rows(
        portfolio,
        molecules=molecules,
    )
    views = _views(route_rows, hypotheses)
    layout = _stable_layout(
        target_id=target_id,
        route_rows=route_rows,
        edge_rows=edge_rows,
    )
    payload = {
        "schema_version": ROUTE_WORKBENCH_SCHEMA,
        "run_id": str(graph.get("run_id") or ""),
        "target": {
            "molecule_id": target_id,
            "name": str(graph.get("target_name") or ""),
            "canonical_smiles": str(
                dict(molecules.get(target_id) or {}).get("canonical_smiles") or ""
            ),
        },
        "revision": {
            "graph": graph_revision,
            "evidence": int(portfolio.get("evidence_revision") or graph_revision),
            "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
            "portfolio_sha256": str(portfolio.get("content_sha256") or ""),
        },
        "portfolio": {
            "route_ids": list(route_rows),
            "default_route_id": next(iter(route_rows), ""),
            "route_count": len(route_rows),
            "accepted": portfolio.get("accepted") is True,
            "closeout": _copy_json(portfolio.get("closeout") or {}),
            "metrics": _copy_json(portfolio.get("metrics") or {}),
            "display_limit": MAX_VISIBLE_ROUTES,
        },
        "campaign_summary": _campaign_summary(campaign_summary),
        "views": views,
        "routes": route_rows,
        "molecules": molecule_rows,
        "edges": edge_rows,
        "hypotheses": hypotheses,
        "modules": modules,
        "shared_intermediates": shared_intermediates,
        "layout": layout,
        "inspectors": {
            "routes": {
                route_id: _route_inspector(route, edge_rows=edge_rows)
                for route_id, route in route_rows.items()
            },
            "edges": {
                edge_id: _edge_inspector(edge_id, graph=graph, proof=edge_proofs.get(edge_id))
                for edge_id in edge_rows
            },
            "molecules": {
                molecule_id: _molecule_inspector(molecule_id, graph=graph)
                for molecule_id in molecule_rows
            },
            "rejections": _copy_json(dict(graph.get("delta") or {}).get("rejected") or []),
            "conflicts": {
                str(key): _copy_json(value)
                for key, value in sorted(dict(graph.get("conflicts") or {}).items())
            },
        },
        "proof_visuals": {str(key): value for key, value in PROOF_VISUALS.items()},
        "semantics": {
            "read_model_only": True,
            "canonical_graph_is_authority": True,
            "proof_color_never_grants_proof": True,
            "default_is_bounded_portfolio": True,
            "shared_intermediates_are_not_duplicated": True,
            "hypotheses_are_not_routes": True,
            "aggregate_counts_never_grant_completion": True,
            "campaign_gates_are_measurements_only": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def compile_route_workbench_delta(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a revision-bound delta; consumers need not replace the graph."""

    if current.get("schema_version") != ROUTE_WORKBENCH_SCHEMA:
        raise RouteWorkbenchProjectionError("route_workbench_current_schema_invalid")
    previous_row = dict(previous or {})
    if previous_row and previous_row.get("schema_version") != ROUTE_WORKBENCH_SCHEMA:
        raise RouteWorkbenchProjectionError("route_workbench_previous_schema_invalid")
    if previous_row and previous_row.get("run_id") != current.get("run_id"):
        raise RouteWorkbenchProjectionError("route_workbench_delta_run_mismatch")

    collections = ("routes", "molecules", "edges", "hypotheses", "modules")
    upserts: dict[str, dict[str, Any]] = {}
    removals: dict[str, list[str]] = {}
    for name in collections:
        before = dict(previous_row.get(name) or {})
        after = dict(current.get(name) or {})
        changed = {
            key: _copy_json(value)
            for key, value in after.items()
            if key not in before or _digest(before[key]) != _digest(value)
        }
        removed = sorted(set(before) - set(after))
        if changed:
            upserts[name] = changed
        if removed:
            removals[name] = removed
    metadata_changed = not previous_row or any(
        _digest(previous_row.get(name)) != _digest(current.get(name))
        for name in (
            "target",
            "revision",
            "portfolio",
            "campaign_summary",
            "views",
            "shared_intermediates",
            "layout",
            "inspectors",
        )
    )
    payload = {
        "schema_version": ROUTE_WORKBENCH_DELTA_SCHEMA,
        "run_id": str(current.get("run_id") or ""),
        "from_graph_revision": int(
            dict(previous_row.get("revision") or {}).get("graph") or 0
        ),
        "to_graph_revision": int(dict(current.get("revision") or {}).get("graph") or 0),
        "base_sha256": str(previous_row.get("content_sha256") or ""),
        "result_sha256": str(current.get("content_sha256") or ""),
        "upserts": upserts,
        "removals": removals,
        "metadata": (
            {
                name: _copy_json(current.get(name))
                for name in (
                    "target",
                    "revision",
                    "portfolio",
                    "campaign_summary",
                    "views",
                    "shared_intermediates",
                    "layout",
                    "inspectors",
                )
            }
            if metadata_changed
            else {}
        ),
        "empty": not upserts and not removals and not metadata_changed,
        "semantics": {
            "apply_only_to_matching_base_sha256": True,
            "entity_upserts_preserve_selection_when_ids_survive": True,
            "full_snapshot_is_recovery_fallback": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _campaign_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bound reporting metadata without letting the UI infer scientific state."""

    source = dict(value or {})
    gates = {
        str(key): item is True
        for key, item in dict(source.get("gates") or {}).items()
        if str(key).startswith("B")
    }
    return {
        "schema_version": "retrosynthesis_campaign_summary.v1",
        "available": bool(source),
        "gates": gates,
        "highest_contiguous_gate": str(
            source.get("highest_contiguous_gate") or "none"
        ),
        "counts": _copy_json(source.get("counts") or {}),
        "resource_envelope": _copy_json(source.get("resource_envelope") or {}),
        "model_cost": _copy_json(source.get("model_cost") or {}),
        "stop_decision": _copy_json(source.get("stop_decision") or {}),
        "claim": _copy_json(source.get("claim") or {}),
        "current_disposition": _copy_json(
            source.get("current_disposition") or {}
        ),
        "semantics": {
            "measurement_only": True,
            "independent_gates_may_pass_after_a_contiguous_gap": True,
            "branch_counts_never_grant_completion": True,
        },
    }


def _validate_bindings(graph: Mapping[str, Any], portfolio: Mapping[str, Any]) -> None:
    if graph.get("schema_version") != "canonical_retrosynthesis_hypergraph.v1":
        raise RouteWorkbenchProjectionError("route_workbench_graph_schema_invalid")
    if portfolio.get("schema_version") != "proof_stitched_route_portfolio.v1":
        raise RouteWorkbenchProjectionError("route_workbench_portfolio_schema_invalid")
    if int(portfolio.get("graph_revision") or 0) != int(graph.get("revision") or 0):
        raise RouteWorkbenchProjectionError("route_workbench_graph_revision_mismatch")
    bound = str(portfolio.get("graph_scientific_sha256") or "")
    actual = str(graph.get("scientific_sha256") or "")
    if bound and actual and bound != actual:
        raise RouteWorkbenchProjectionError("route_workbench_graph_digest_mismatch")


def _molecule_row(
    molecule_id: str,
    molecule: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    leaf_proof: Mapping[str, Any] | None,
    target_id: str,
) -> dict[str, Any]:
    proof = dict(leaf_proof or {})
    stock_id = str(molecule.get("active_stock_observation_id") or "")
    stock = dict(dict(graph.get("stock_observations") or {}).get(stock_id) or {})
    return {
        "molecule_id": molecule_id,
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "role": "target" if molecule_id == target_id else (
            "stock_leaf" if molecule.get("is_leaf") else "intermediate"
        ),
        "is_leaf": molecule.get("is_leaf") is True,
        "stock_closed": proof.get("accepted") is True,
        "stock_observation_id": stock_id,
        "stock_label": str(stock.get("catalog_number") or stock.get("supplier") or ""),
        "badges": ["stock-audited"] if stock_id else [],
    }


def _edge_row(
    edge_id: str,
    edge: Mapping[str, Any],
    *,
    proof: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    level = max(0, min(4, int(proof.get("achieved_level") or 1)))
    origins = sorted(
        {
            str(value.get("origin_kind") or "")
            for value in edge.get("origin_records") or []
            if isinstance(value, Mapping) and str(value.get("origin_kind") or "")
        }
    )
    source_kinds = sorted(
        {
            str(value.get("source_kind") or "")
            for source_id in proof.get("source_binding_ids") or []
            for value in [dict(dict(graph.get("source_bindings") or {}).get(source_id) or {})]
            if str(value.get("source_kind") or "")
        }
    )
    badges = [f"proposal:{value}" for value in origins]
    badges.extend(f"source:{value}" for value in source_kinds)
    if proof.get("reaction_validated") is True:
        badges.append("reaction-validated")
    if proof.get("exact_source_bound") is True:
        badges.append("exact-source")
    if proof.get("conflict_ids"):
        badges.append("conflict")
    return {
        "edge_id": edge_id,
        "product_molecule_id": str(edge.get("product_molecule_id") or ""),
        "precursor_molecule_ids": [
            str(value) for value in edge.get("precursor_molecule_ids") or []
        ],
        "proof_level": level,
        "proof_name": PROOF_VISUALS[level]["name"],
        "proof_color": PROOF_VISUALS[level]["color"],
        "accepted": proof.get("accepted") is True,
        "origin_kinds": origins,
        "source_kinds": source_kinds,
        "badges": sorted(set(badges)),
    }


def _route_row(
    route: Mapping[str, Any],
    *,
    edge_rows: Mapping[str, Mapping[str, Any]],
    deficits: list[dict[str, Any]],
) -> dict[str, Any]:
    route_id = str(route.get("route_id") or "")
    level = max(0, min(4, int(route.get("minimum_edge_proof_level") or 0)))
    if route.get("complete") is True:
        stage = "stock_closed"
    elif route.get("all_edges_proven") is True and level >= 2:
        stage = "reaction_validated"
    elif route.get("edge_ids"):
        stage = "expanded"
    else:
        stage = "hypothesis"
    route_deficits = [
        value
        for value in deficits
        if route_id in {
            str(value.get("route_id") or ""),
            *(str(item) for item in value.get("route_ids") or []),
        }
    ]
    badges = [stage.replace("_", "-")]
    source_kinds = sorted(
        {
            str(kind)
            for edge_id in route.get("edge_ids") or []
            for kind in dict(edge_rows.get(str(edge_id)) or {}).get("source_kinds") or []
        }
    )
    badges.extend(f"source:{value}" for value in source_kinds)
    if route.get("pareto_optimal") is True:
        badges.append("pareto")
    return {
        "route_id": route_id,
        "route_family_id": str(route.get("route_family_id") or ""),
        "strategy": str(route.get("strategy") or ""),
        "stage": stage,
        "proof_level": level,
        "proof_name": PROOF_VISUALS[level]["name"],
        "proof_color": PROOF_VISUALS[level]["color"],
        "edge_ids": [str(value) for value in route.get("edge_ids") or []],
        "leaf_molecule_ids": [
            str(value) for value in route.get("leaf_molecule_ids") or []
        ],
        "root_edge_ids": [str(value) for value in route.get("root_edge_ids") or []],
        "module_selections": dict(route.get("module_selections") or {}),
        "complete": route.get("complete") is True,
        "stock_closure_rate": float(route.get("stock_closure_rate") or 0.0),
        "independent_source_groups": list(route.get("independent_source_groups") or []),
        "risk_score": float(route.get("risk_score") or 0.0),
        "convergence_score": float(route.get("convergence_score") or 0.0),
        "deficit_count": len(route_deficits),
        "badges": sorted(set(badges)),
    }


def _hypothesis_rows(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = [
        (str(key), dict(value))
        for key, value in dict(graph.get("hypotheses") or {}).items()
        if isinstance(value, Mapping) and value.get("status") != "materialized"
    ]
    values.sort(
        key=lambda item: (
            -float(item[1].get("route_diversity_gain") or 0.0),
            item[0],
        )
    )
    return {
        hypothesis_id: {
            "hypothesis_id": hypothesis_id,
            "product_smiles": str(value.get("product_smiles") or ""),
            "precursor_smiles": list(value.get("precursor_smiles") or []),
            "route_family_ids": list(value.get("route_family_ids") or []),
            "status": "hypothesis",
            "proof_level": 0,
            "proof_color": PROOF_VISUALS[0]["color"],
            "origin_kinds": sorted(
                {
                    str(row.get("origin_kind") or "")
                    for row in value.get("origin_records") or []
                    if isinstance(row, Mapping) and str(row.get("origin_kind") or "")
                }
            ),
            "badges": ["proposal", "unmaterialized"],
        }
        for hypothesis_id, value in values[:MAX_VISIBLE_HYPOTHESES]
    }


def _module_rows(
    portfolio: Mapping[str, Any],
    *,
    selected_route_ids: set[str],
) -> dict[str, dict[str, Any]]:
    selected_families = {
        str(value.get("route_family_id") or "")
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping) and str(value.get("route_id") or "") in selected_route_ids
    }
    return {
        str(value["module_id"]): _copy_json(value)
        for value in portfolio.get("route_modules") or []
        if isinstance(value, Mapping)
        and str(value.get("module_id") or "")
        and str(value.get("route_family_id") or "") in selected_families
    }


def _shared_intermediate_rows(
    portfolio: Mapping[str, Any],
    *,
    molecules: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    shared = dict(dict(portfolio.get("metrics") or {}).get("shared_intermediates") or {})
    return {
        molecule_id: {
            "molecule_id": molecule_id,
            "canonical_smiles": str(
                dict(molecules.get(molecule_id) or {}).get("canonical_smiles") or ""
            ),
            "route_ids": sorted(str(value) for value in route_ids),
            "render_once": True,
        }
        for molecule_id, route_ids in sorted(shared.items())
        if molecule_id in molecules
    }


def _views(
    routes: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    stages = {
        "hypotheses": {"label": "Disconnection hypotheses", "route_ids": []},
        "expanded": {"label": "Expanded routes", "route_ids": []},
        "reaction_validated": {"label": "Reaction-validated routes", "route_ids": []},
        "stock_closed": {"label": "Stock-closed routes", "route_ids": []},
    }
    stages["hypotheses"]["hypothesis_ids"] = list(hypotheses)
    for route_id, route in routes.items():
        stage = str(route.get("stage") or "expanded")
        if stage == "stock_closed":
            stages["stock_closed"]["route_ids"].append(route_id)
        if stage in {"reaction_validated", "stock_closed"}:
            stages["reaction_validated"]["route_ids"].append(route_id)
        stages["expanded"]["route_ids"].append(route_id)
    for value in stages.values():
        value["count"] = len(value.get("route_ids") or value.get("hypothesis_ids") or [])
    return stages


def _stable_layout(
    *,
    target_id: str,
    route_rows: Mapping[str, Mapping[str, Any]],
    edge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    adjacency = {
        str(edge["product_molecule_id"]): [
            str(value) for value in edge.get("precursor_molecule_ids") or []
        ]
        for edge in edge_rows.values()
    }
    layers: dict[str, int] = {target_id: 0} if target_id else {}
    queue = [target_id] if target_id else []
    while queue:
        product = queue.pop(0)
        next_layer = layers[product] + 1
        for precursor in adjacency.get(product, []):
            if next_layer > layers.get(precursor, -1):
                layers[precursor] = next_layer
                queue.append(precursor)
    nodes = [
        {"molecule_id": molecule_id, "layer": layer, "order": order}
        for layer in sorted(set(layers.values()))
        for order, molecule_id in enumerate(
            sorted(value for value, current in layers.items() if current == layer)
        )
    ]
    return {
        "algorithm": "canonical_layer_order.v1",
        "orientation": "target_to_precursors",
        "nodes": nodes,
        "route_edge_sets": {
            route_id: list(route.get("edge_ids") or [])
            for route_id, route in route_rows.items()
        },
        "stable_ids": True,
        "heavy_layout_required": len(nodes) > 120,
    }


def _route_inspector(
    route: Mapping[str, Any],
    *,
    edge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": route.get("stage"),
        "proof_level": route.get("proof_level"),
        "stock_closure_rate": route.get("stock_closure_rate"),
        "risk_score": route.get("risk_score"),
        "independent_source_groups": list(route.get("independent_source_groups") or []),
        "edge_badges": {
            edge_id: list(dict(edge_rows.get(edge_id) or {}).get("badges") or [])
            for edge_id in route.get("edge_ids") or []
        },
    }


def _edge_inspector(
    edge_id: str,
    *,
    graph: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    proof_row = dict(proof or {})
    edge = dict(dict(graph.get("edges") or {}).get(edge_id) or {})
    source_ids = [str(value) for value in proof_row.get("source_binding_ids") or []]
    record_ids = [str(value) for value in proof_row.get("exact_record_ids") or []]
    conflict_ids = [str(value) for value in proof_row.get("conflict_ids") or []]
    return {
        "proof": _copy_json(proof_row),
        "reaction_proofs": _copy_json(edge.get("reaction_proofs") or []),
        "sources": [
            _copy_json(dict(graph.get("source_bindings") or {}).get(value) or {})
            for value in source_ids
        ],
        "exact_records": [
            _copy_json(dict(graph.get("exact_records") or {}).get(value) or {})
            for value in record_ids
        ],
        "conflicts": [
            _copy_json(dict(graph.get("conflicts") or {}).get(value) or {})
            for value in conflict_ids
        ],
        "provenance": _copy_json(edge.get("origin_records") or []),
        "rejection_reasons": list(proof_row.get("reasons") or []),
    }


def _molecule_inspector(molecule_id: str, *, graph: Mapping[str, Any]) -> dict[str, Any]:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    observation_ids = [str(value) for value in molecule.get("stock_observation_ids") or []]
    return {
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "stock_closed": molecule.get("stock_closed") is True,
        "stock_observations": [
            _copy_json(dict(graph.get("stock_observations") or {}).get(value) or {})
            for value in observation_ids
        ],
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "MAX_VISIBLE_ROUTES",
    "PROOF_VISUALS",
    "ROUTE_WORKBENCH_DELTA_SCHEMA",
    "ROUTE_WORKBENCH_SCHEMA",
    "RouteWorkbenchProjectionError",
    "compile_route_workbench",
    "compile_route_workbench_delta",
]
