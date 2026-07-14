"""Cross-family replacement modules for proof-stitched route portfolios."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.route_variants import (
    ROUTE_MODULE_SCHEMA,
    with_content_digest,
)


def compile_replacement_modules(
    graph: Mapping[str, Any],
    *,
    candidates: list[dict[str, Any]],
    edge_proofs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build same-product modules across every fully stitched route family."""
    edges = dict(graph.get("edges") or {})
    edge_route_ids: dict[str, set[str]] = {}
    edge_family_ids: dict[str, set[str]] = {}
    by_product: dict[str, set[str]] = {}
    for candidate in candidates:
        route_id = str(candidate.get("route_id") or "")
        family_id = str(candidate.get("route_family_id") or "")
        for raw_edge_id in candidate.get("edge_ids") or []:
            edge_id = str(raw_edge_id)
            edge = dict(edges.get(edge_id) or {})
            product_id = str(edge.get("product_molecule_id") or "")
            if not edge_id or not product_id:
                continue
            by_product.setdefault(product_id, set()).add(edge_id)
            if route_id:
                edge_route_ids.setdefault(edge_id, set()).add(route_id)
            if family_id:
                edge_family_ids.setdefault(edge_id, set()).add(family_id)

    modules: list[dict[str, Any]] = []
    module_alternatives: dict[str, set[str]] = {}
    for product_id, alternative_ids in sorted(by_product.items()):
        if len(alternative_ids) < 2:
            continue
        family_ids = sorted(
            {
                family_id
                for edge_id in alternative_ids
                for family_id in edge_family_ids.get(edge_id, set())
            }
        )
        module_id = f"module:{_digest({'product': product_id})}"
        module = {
            "schema_version": ROUTE_MODULE_SCHEMA,
            "module_id": module_id,
            "route_family_id": family_ids[0] if len(family_ids) == 1 else "",
            "route_family_ids": family_ids,
            "product_molecule_id": product_id,
            "alternatives": [
                {
                    "edge_id": edge_id,
                    "proof_level": int(
                        edge_proofs.get(edge_id, {}).get("achieved_level") or 0
                    ),
                    "accepted": edge_proofs.get(edge_id, {}).get("accepted") is True,
                    "route_family_ids": sorted(edge_family_ids.get(edge_id, set())),
                    "complete_route_ids": sorted(edge_route_ids.get(edge_id, set())),
                }
                for edge_id in sorted(alternative_ids)
            ],
            "semantics": {
                "same_product_replacement_boundary": True,
                "cross_family_replacement_supported": len(family_ids) > 1,
                "full_restitched_candidate_route_required": True,
                "shared_subgraph_not_duplicated": True,
            },
        }
        module["content_sha256"] = _digest(module)
        modules.append(module)
        module_alternatives[module_id] = set(alternative_ids)

    projected_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        edge_ids = {str(value) for value in candidate.get("edge_ids") or []}
        selections = {
            module_id: selected[0]
            for module_id, alternatives in module_alternatives.items()
            if len(selected := sorted(edge_ids & alternatives)) == 1
        }
        projected_candidates.append(
            with_content_digest({**candidate, "module_selections": selections})
        )
    return modules, projected_candidates


def validate_module_replacement(
    portfolio: Mapping[str, Any],
    *,
    route_id: str,
    module_id: str,
    replacement_edge_id: str,
) -> dict[str, Any]:
    module = next(
        (
            dict(value)
            for value in portfolio.get("route_modules") or []
            if value.get("module_id") == module_id
        ),
        {},
    )
    route = next(
        (
            dict(value)
            for value in portfolio.get("route_candidates") or []
            if value.get("route_id") == route_id
        ),
        {},
    )
    alternatives = {
        str(value.get("edge_id") or "") for value in module.get("alternatives") or []
    }
    current = str(dict(route.get("module_selections") or {}).get(module_id) or "")
    reasons: list[str] = []
    if not module:
        reasons.append("replacement_module_missing")
    if not route:
        reasons.append("replacement_route_missing")
    if replacement_edge_id not in alternatives:
        reasons.append("replacement_edge_not_in_module")
    if replacement_edge_id == current:
        reasons.append("replacement_edge_is_already_selected")
    if module and route and not current:
        reasons.append("replacement_module_not_used_by_route")
    module_family_ids = {
        str(value) for value in module.get("route_family_ids") or [] if str(value)
    }
    if not module_family_ids and str(module.get("route_family_id") or ""):
        module_family_ids.add(str(module["route_family_id"]))
    if (
        module
        and route
        and str(route.get("route_family_id") or "") not in module_family_ids
    ):
        reasons.append("replacement_module_route_family_mismatch")
    patch = {
        "schema_version": "route_module_replacement_patch.v1",
        "route_id": route_id,
        "module_id": module_id,
        "product_molecule_id": str(module.get("product_molecule_id") or ""),
        "remove_edge_id": current,
        "add_edge_id": replacement_edge_id,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "patch_reuses_canonical_subgraph": True,
            "patch_does_not_duplicate_entire_route": True,
            "replacement_requires_reproof": True,
        },
    }
    patch["content_sha256"] = _digest(patch)
    return patch


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


__all__ = ["compile_replacement_modules", "validate_module_replacement"]
