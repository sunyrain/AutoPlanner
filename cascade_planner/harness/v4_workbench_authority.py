"""Honest proof, frontier, and cost projections for the V4 display adapter."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import canonical_json_sha256


def frontier_ledger(
    source: Mapping[str, Any],
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    authority_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    molecules_by_smiles: dict[str, dict[str, Any]] = {}
    for molecule_id, node in nodes_by_id.items():
        smiles = str(node.get("canonical_isomeric_smiles") or "")
        if not smiles:
            continue
        stock_closed = node.get("stock_closed") is True
        observation_id = str(node.get("stock_observation_id") or "")
        authority_scope = str(node.get("stock_authority_scope") or "")
        procurement_authority = authority_scope in {
            "procurement_offer_verified",
            "procurement_stock_observation",
            "in_house_stock_observation",
        }
        current = molecules_by_smiles.setdefault(
            smiles,
            {
                "canonical_smiles": smiles,
                "work": {"proposal_expansion_succeeded": True, "job_ids": []},
                "stock": {
                    "host_replay_verified": False,
                    "current_observation_ids": [],
                    "closure_job_ids": [],
                    "benchmark_search_boundary_closed": False,
                    "procurement_boundary_closed": False,
                },
            },
        )
        current["work"]["job_ids"] = sorted(
            {*current["work"]["job_ids"], f"graph:{molecule_id}"}
        )
        if stock_closed and observation_id:
            current["stock"].update(
                {
                    "host_replay_verified": (
                        node.get("stock_observation_accepted") is True
                    ),
                    "current_observation_ids": sorted(
                        {*current["stock"]["current_observation_ids"], observation_id}
                    ),
                    "closure_job_ids": sorted(
                        {*current["stock"]["closure_job_ids"], f"stock:{molecule_id}"}
                    ),
                    "benchmark_search_boundary_closed": True,
                    "procurement_boundary_closed": procurement_authority,
                }
            )
    edges_by_signature: dict[str, dict[str, Any]] = {}
    for edge in authority_edges:
        signature = str(edge.get("exact_edge_signature") or "")
        if not signature:
            continue
        current = edges_by_signature.setdefault(signature, dict(edge))
        current["step_ids"] = sorted(
            {
                *(str(value) for value in current.get("step_ids") or []),
                *(str(value) for value in edge.get("step_ids") or []),
            }
        )
        current_level = int(
            dict(current.get("reaction_proof") or {}).get("achieved_proof_level") or 0
        )
        next_level = int(
            dict(edge.get("reaction_proof") or {}).get("achieved_proof_level") or 0
        )
        if next_level > current_level:
            current["reaction_proof"] = dict(edge.get("reaction_proof") or {})
    routes = [
        dict(value)
        for value in dict(source.get("routes") or {}).values()
        if isinstance(value, Mapping)
    ]
    complete_routes = [route for route in routes if route.get("complete") is True]
    procurement_routes = [
        route for route in routes if route.get("procurement_closed") is True
    ]
    edges = [
        dict(value)
        for value in dict(source.get("edges") or {}).values()
        if isinstance(value, Mapping)
    ]
    reachable_leaves = {
        str(value)
        for route in routes
        for value in route.get("leaf_molecule_ids") or []
    }
    stock_closed_leaves = {
        molecule_id
        for molecule_id in reachable_leaves
        if dict(nodes_by_id.get(molecule_id) or {}).get("stock_closed") is True
    }
    procurement_closed_leaves = {
        molecule_id
        for molecule_id in stock_closed_leaves
        if str(
            dict(nodes_by_id.get(molecule_id) or {}).get(
                "stock_authority_scope"
            )
            or ""
        )
        in {
            "procurement_offer_verified",
            "procurement_stock_observation",
            "in_house_stock_observation",
        }
    }
    accepted = dict(source.get("portfolio") or {}).get("accepted") is True
    ledger = {
        "schema_version": "route_forest_frontier_ledger_authority.v1",
        "authoritative": True,
        "stage_authority": {
            "schema_version": "route_forest_stage_authority.v1",
            "authoritative": True,
            "reasons": [],
            "molecules": [molecules_by_smiles[key] for key in sorted(molecules_by_smiles)],
            "edges": [edges_by_signature[key] for key in sorted(edges_by_signature)],
        },
        "counts": {
            "selected_routes": len(routes),
            "complete_routes": len(complete_routes),
            "l0_break_suggestion_edges": sum(
                int(edge.get("proof_level") or 0) == 0 for edge in edges
            ),
            "l1_source_reported_edges": sum(
                int(edge.get("proof_level") or 0) == 1
                and "paper_si" in set(edge.get("source_kinds") or [])
                for edge in edges
            ),
            "l1_materialized_edges": sum(
                int(edge.get("proof_level") or 0) == 1 for edge in edges
            ),
            "expanded_work_molecules": len(nodes_by_id),
            "reachable_molecules": len(nodes_by_id),
            "l2_reaction_edges": sum(
                int(edge.get("proof_level") or 0) >= 2 for edge in edges
            ),
            "l3_precedent_edges": sum(
                int(edge.get("proof_level") or 0) >= 3 for edge in edges
            ),
            "stock_closed_leaves": len(stock_closed_leaves),
            "reachable_leaves": len(reachable_leaves),
            "benchmark_only_stock_leaves": len(
                stock_closed_leaves - procurement_closed_leaves
            ),
            "procurement_boundary_leaves": len(procurement_closed_leaves),
            "l4_procurement_edges": 0,
        },
        "closure": {
            "any_benchmark_route_closed": bool(complete_routes),
            "all_explored_benchmark_closed": bool(routes)
            and len(complete_routes) == len(routes),
            "any_procurement_route_closed": bool(procurement_routes),
            "all_explored_procurement_closed": bool(routes)
            and len(procurement_routes) == len(routes),
            "l3_parent_solved": accepted,
            "l4_procurement_ready": accepted,
        },
        "semantics": {
            "display_projection_only": True,
            "aggregate_counts_never_authorize_stage_membership": True,
        },
    }
    ledger["content_sha256"] = canonical_json_sha256(ledger)
    return ledger


def selected_route_proof(source: Mapping[str, Any]) -> dict[str, Any]:
    routes = [
        dict(value)
        for value in dict(source.get("routes") or {}).values()
        if isinstance(value, Mapping) and value.get("complete") is True
    ]
    accepted = dict(source.get("portfolio") or {}).get("accepted") is True
    route_rows = [
        {
            "route_id": str(route.get("route_id") or ""),
            "edge_set_sha256": canonical_json_sha256(
                sorted(str(value) for value in route.get("edge_ids") or [])
            ),
            "selected_hyperedge_ids": [
                str(value) for value in route.get("edge_ids") or []
            ],
            "weakest_edge_proof_level": int(route.get("proof_level") or 0),
            "benchmark_closed": True,
            "procurement_ready": route.get("process_ready") is True,
        }
        for route in routes
    ]
    return {
        "schema_version": "selected_route_parent_proof_display.v1",
        "available": True,
        "accepted": accepted,
        "benchmark_solved": accepted,
        "procurement_ready": accepted,
        "any_procurement_route_ready": any(
            row["procurement_ready"] is True for row in route_rows
        ),
        "minimum_complete_routes": max(1, len(routes)),
        "distinct_complete_route_count": len(routes),
        "benchmark_route_count": len(routes),
        "procurement_route_count": sum(
            row["procurement_ready"] is True for row in route_rows
        ),
        "routes": route_rows,
        "reasons": [] if accepted else ["canonical_v4_portfolio_not_accepted"],
        "semantics": {
            "derived_only_from_current_v4_proof_portfolio": True,
            "all_selected_edges_require_l3_or_better": True,
            "all_selected_leaves_require_exact_stock_bindings": True,
        },
    }


def retrosynthesis_control(
    source: Mapping[str, Any],
    *,
    selected_route_proof: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = dict(source.get("portfolio") or {}).get("accepted") is True
    route_count = len(dict(source.get("routes") or {}))
    return {
        "schema_version": "retrosynthesis_control_display.v1",
        "available": True,
        "authoritative": True,
        "acceptance": {
            "accepted": accepted,
            "selected_route_count": int(
                selected_route_proof.get("distinct_complete_route_count") or 0
            ),
            "acceptance_spec": {"minimum_complete_routes": max(1, route_count)},
        },
        "next_deficit": {},
        "cost_totals": {"model_invocations": 0, "input_tokens": 0, "output_tokens": 0},
        "cost_budget": {
            "max_model_invocations": 0,
            "max_total_input_tokens": 0,
            "max_total_output_tokens": 0,
        },
        "semantics": {
            "current_v4_closeout_is_authority": True,
            "display_adapter_cannot_change_acceptance": True,
        },
    }


__all__ = ["frontier_ledger", "retrosynthesis_control", "selected_route_proof"]
