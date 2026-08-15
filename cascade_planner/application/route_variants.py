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
from cascade_planner.application.route_candidate_builder import build_route_candidate


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
