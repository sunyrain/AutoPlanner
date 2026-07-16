"""Focused molecule and reaction-edge rows for the route Workbench."""

from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.route_workbench_inspectors import (
    edge_proof_vector,
)
from cascade_planner.application.route_workbench_route_rows import PROOF_VISUALS


def molecule_row(
    molecule_id: str,
    molecule: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    leaf_proof: Mapping[str, Any] | None,
    target_id: str,
) -> dict[str, Any]:
    proof = dict(leaf_proof or {})
    inactive_facts = [
        dict(value)
        for value in proof.get("inactive_facts") or []
        if isinstance(value, Mapping)
    ]
    stock_id = str(molecule.get("active_stock_observation_id") or "")
    stock = dict(dict(graph.get("stock_observations") or {}).get(stock_id) or {})
    return {
        "molecule_id": molecule_id,
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "label": str(molecule.get("label") or molecule.get("name") or ""),
        "role": "target" if molecule_id == target_id else (
            "stock_leaf" if molecule.get("is_leaf") else "intermediate"
        ),
        "is_leaf": molecule.get("is_leaf") is True,
        "stock_closed": proof.get("accepted") is True,
        "stock_observation_id": stock_id,
        "stock_label": str(stock.get("catalog_number") or stock.get("supplier") or ""),
        "stock_authority_scope": str(stock.get("authority_scope") or ""),
        "stock_observation_accepted": bool(stock) and stock.get("accepted") is not False,
        "inactive_fact_count": len(inactive_facts),
        "inactive_facts": inactive_facts,
        "badges": sorted(
            {
                *(["stock-audited"] if stock_id else []),
                *(
                    f"fact-{value.get('status') or 'inactive'}"
                    for value in inactive_facts
                ),
            }
        ),
    }


def edge_row(
    edge_id: str,
    edge: Mapping[str, Any],
    *,
    proof: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    raw_level = proof.get("achieved_level")
    level = max(0, min(4, int(1 if raw_level is None else raw_level)))
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
    inactive_facts = [
        dict(value)
        for value in proof.get("inactive_facts") or []
        if isinstance(value, Mapping)
    ]
    badges.extend(f"fact-{value.get('status') or 'inactive'}" for value in inactive_facts)
    route_innovations = [
        _copy_json(value)
        for value in edge.get("route_innovations") or []
        if isinstance(value, Mapping)
    ]
    innovation_kinds = sorted(
        {
            str(value.get("kind") or "")
            for value in route_innovations
            if str(value.get("kind") or "")
        }
    )
    badges.extend(f"innovation:{value}" for value in innovation_kinds)
    if any(
        value.get("kind") == "biocatalytic_superstep"
        and int(value.get("step_savings") or 0) > 0
        for value in route_innovations
    ):
        badges.append("multi-step-compression")
    proof_vector = edge_proof_vector(edge=edge, proof=proof, graph=graph)
    if proof_vector["conditions"] == "missing":
        badges.append("conditions-missing")
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
        "proof_vector": proof_vector,
        "condition_status": proof_vector["conditions"],
        "route_innovations": route_innovations,
        "innovation_kinds": innovation_kinds,
        "innovation_proof_gate": _copy_json(proof.get("innovation_proof_gate") or {}),
        "inactive_fact_count": len(inactive_facts),
        "inactive_facts": inactive_facts,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


__all__ = ["edge_row", "molecule_row"]
