"""Canonical route enumeration and weakest-link candidate construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import Any, Mapping

from cascade_planner.application.proof_policy import (
    ProofPolicy,
    stitch_leaf_stock_proof,
)


PROOF_ROUTE_SCHEMA = "proof_stitched_route.v1"
ROUTE_MODULE_SCHEMA = "route_replacement_module.v1"


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    minimum_routes_to_show: int = 2
    maximum_routes_to_show: int = 5
    maximum_variants_per_family: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_routes_to_show <= self.maximum_routes_to_show <= 12:
            raise ValueError("portfolio route display limits are invalid")
        if not 1 <= self.maximum_variants_per_family <= 2048:
            raise ValueError("portfolio family variant limit is invalid")


@dataclass(frozen=True, slots=True)
class RouteSubroute:
    edge_ids: frozenset[str]
    leaf_ids: frozenset[str]
    module_selections: tuple[tuple[str, str], ...]


def enumerate_family_variants(
    graph: Mapping[str, Any],
    *,
    family_id: str,
    family: Mapping[str, Any],
    policy: ProofPolicy,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    leaf_proof_cache: dict[str, dict[str, Any]],
    limit: int,
) -> tuple[list[RouteSubroute], list[dict[str, Any]]]:
    allowed = {
        str(value)
        for value in family.get("edge_ids") or []
        if str(value) in dict(graph.get("edges") or {})
    }
    outgoing: dict[str, list[str]] = {}
    for edge_id in sorted(allowed):
        edge = graph["edges"][edge_id]
        outgoing.setdefault(str(edge["product_molecule_id"]), []).append(edge_id)
    modules: list[dict[str, Any]] = []
    for molecule_id, edge_ids in sorted(outgoing.items()):
        if len(edge_ids) < 2:
            continue
        module = {
            "schema_version": ROUTE_MODULE_SCHEMA,
            "module_id": (
                f"module:{_digest({'family': family_id, 'product': molecule_id})}"
            ),
            "route_family_id": family_id,
            "product_molecule_id": molecule_id,
            "alternatives": [
                {
                    "edge_id": edge_id,
                    "proof_level": int(
                        edge_proofs.get(edge_id, {}).get("achieved_level") or 0
                    ),
                    "accepted": edge_proofs.get(edge_id, {}).get("accepted") is True,
                }
                for edge_id in sorted(edge_ids)
            ],
            "semantics": {
                "same_product_replacement_boundary": True,
                "shared_subgraph_not_duplicated": True,
            },
        }
        module["content_sha256"] = _digest(module)
        modules.append(module)
    module_by_product = {
        str(value["product_molecule_id"]): str(value["module_id"])
        for value in modules
    }

    def walk(molecule_id: str, ancestors: frozenset[str]) -> list[RouteSubroute]:
        if molecule_id in ancestors:
            return []
        options: list[RouteSubroute] = []
        stock = leaf_proof_cache.setdefault(
            molecule_id,
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy),
        )
        edges = outgoing.get(molecule_id, [])
        if stock["accepted"] is True or not edges:
            options.append(
                RouteSubroute(frozenset(), frozenset({molecule_id}), ())
            )
        for edge_id in edges:
            edge = graph["edges"][edge_id]
            precursor_variants = [
                walk(str(precursor_id), ancestors | {molecule_id})
                for precursor_id in edge.get("precursor_molecule_ids") or []
            ]
            if any(not values for values in precursor_variants):
                continue
            for combination in product(*precursor_variants):
                edges_used = {edge_id}
                leaves: set[str] = set()
                selections: dict[str, str] = {}
                for subroute in combination:
                    edges_used.update(subroute.edge_ids)
                    leaves.update(subroute.leaf_ids)
                    selections.update(dict(subroute.module_selections))
                if molecule_id in module_by_product:
                    selections[module_by_product[molecule_id]] = edge_id
                options.append(
                    RouteSubroute(
                        frozenset(edges_used),
                        frozenset(leaves),
                        tuple(sorted(selections.items())),
                    )
                )
                if len(options) >= limit:
                    break
            if len(options) >= limit:
                break
        deduped = {
            (value.edge_ids, value.leaf_ids, value.module_selections): value
            for value in options
        }
        return list(deduped.values())[:limit]

    root = str(graph.get("target_molecule_id") or "")
    variants = walk(root, frozenset()) if root else []
    return variants[:limit], modules


def build_route_candidate(
    graph: Mapping[str, Any],
    *,
    family_id: str,
    family: Mapping[str, Any],
    variant: RouteSubroute,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    leaf_proof_cache: dict[str, dict[str, Any]],
    policy: ProofPolicy,
) -> dict[str, Any]:
    edge_ids = sorted(variant.edge_ids)
    leaf_ids = sorted(variant.leaf_ids)
    proofs = [dict(edge_proofs[edge_id]) for edge_id in edge_ids]
    leaves = [
        leaf_proof_cache.setdefault(
            molecule_id,
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy),
        )
        for molecule_id in leaf_ids
    ]
    source_groups = sorted(
        {
            str(group)
            for proof in proofs
            for group in proof.get("independent_source_groups") or []
            if str(group)
        }
    )
    conflicts = sorted(
        {
            str(conflict_id)
            for proof in proofs
            for conflict_id in proof.get("conflict_ids") or []
            if str(conflict_id)
        }
    )
    min_proof = min((int(value["achieved_level"]) for value in proofs), default=0)
    unproven_edge_ids = sorted(
        str(value["edge_id"])
        for value in proofs
        if value.get("accepted") is not True
    )
    stock_rate = sum(value["accepted"] is True for value in leaves) / max(
        1, len(leaves)
    )
    open_leaf_molecule_ids = sorted(
        str(value["molecule_id"])
        for value in leaves
        if value.get("accepted") is not True
    )
    source_met = len(source_groups) >= policy.minimum_independent_source_groups
    complete = bool(edge_ids) and all(value["accepted"] is True for value in proofs)
    if policy.require_stock_for_every_selected_leaf:
        complete = complete and bool(leaves) and stock_rate == 1.0
    complete = complete and source_met and not conflicts
    root_edges = sorted(
        edge_id
        for edge_id in edge_ids
        if str(graph["edges"][edge_id]["product_molecule_id"])
        == str(graph.get("target_molecule_id") or "")
    )
    precursor_frequency: dict[str, int] = {}
    for edge_id in edge_ids:
        for molecule_id in graph["edges"][edge_id]["precursor_molecule_ids"]:
            key = str(molecule_id)
            precursor_frequency[key] = precursor_frequency.get(key, 0) + 1
    convergence = sum(value > 1 for value in precursor_frequency.values()) / max(
        1, len(precursor_frequency)
    )
    risk = (
        0.35 * (1.0 - min_proof / 4.0)
        + 0.25 * (1.0 - stock_rate)
        + 0.20 * (not source_met)
        + 0.15 * bool(conflicts)
        + 0.05 * min(1.0, len(edge_ids) / 12.0)
    )
    identity = {
        "route_family_id": family_id,
        "edge_ids": edge_ids,
        "leaf_molecule_ids": leaf_ids,
    }
    return with_content_digest(
        {
            "schema_version": PROOF_ROUTE_SCHEMA,
            "route_id": f"route:{_digest(identity)}",
            "route_family_id": family_id,
            "strategy": str(family.get("strategy") or ""),
            "edge_ids": edge_ids,
            "leaf_molecule_ids": leaf_ids,
            "root_edge_ids": root_edges,
            "module_selections": dict(variant.module_selections),
            "minimum_edge_proof_level": min_proof,
            "all_edges_proven": bool(proofs)
            and all(value["accepted"] for value in proofs),
            "unproven_edge_ids": unproven_edge_ids,
            "stock_closure_rate": round(stock_rate, 6),
            "all_leaves_stock_closed": bool(leaves) and stock_rate == 1.0,
            "open_leaf_molecule_ids": open_leaf_molecule_ids,
            "independent_source_groups": source_groups,
            "source_independence_met": source_met,
            "conflict_ids": conflicts,
            "length": len(edge_ids),
            "convergence_score": round(convergence, 6),
            "risk_score": round(float(risk), 6),
            "complete": complete,
            "selected": False,
            "semantics": {
                "weakest_edge_controls_route": True,
                "every_leaf_requires_stock_observation": True,
                "counts_do_not_override_boolean_proofs": True,
            },
        }
    )


def with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


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
    "PROOF_ROUTE_SCHEMA",
    "ROUTE_MODULE_SCHEMA",
    "PortfolioConfig",
    "RouteSubroute",
    "build_route_candidate",
    "enumerate_family_variants",
    "with_content_digest",
]
