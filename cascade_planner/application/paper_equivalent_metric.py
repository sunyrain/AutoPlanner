"""Independent topology-and-stock metric used for paper comparison."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


PAPER_EQUIVALENT_METRIC_SCHEMA = "paper_equivalent_solved_metric.v1"
SYNTHEX_STOCK_MEMBER_COUNT = 39_684_411


def compile_paper_equivalent_metric(
    blind_acceptance: Mapping[str, Any],
    *,
    stock_oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Score any one connected, all-leaves-in-stock route as solved.

    ``blind_acceptance.routes`` has already been walked from the exact target,
    so disconnected skeletons are absent.  This projection intentionally does
    not read reaction validation, evidence, conditions, or the configured
    minimum route count.
    """

    routes = [
        dict(row)
        for row in blind_acceptance.get("routes") or []
        if isinstance(row, Mapping)
    ]
    solved_routes = [
        {
            "route_family_id": str(row.get("route_family_id") or ""),
            "skeleton_id": str(row.get("skeleton_id") or ""),
            "edge_ids": sorted(str(value) for value in row.get("edge_ids") or []),
            "leaf_molecule_ids": sorted(
                str(value) for value in row.get("leaf_molecule_ids") or []
            ),
        }
        for row in routes
        if row.get("stock_closed") is True
        and bool(row.get("edge_ids"))
        and bool(row.get("leaf_molecule_ids"))
    ]
    oracle = dict(stock_oracle or {})
    binding = dict(oracle.get("binding") or {})
    catalog_name = str(binding.get("catalog_name") or "")
    member_count = int(binding.get("member_count") or 0)
    identity_key = str(binding.get("identity_key") or "canonical_smiles")
    normalized_name = catalog_name.casefold().replace(" ", "")
    exact_paper_stock = (
        "zinc" in normalized_name
        and "emolecules" in normalized_name
        and member_count == SYNTHEX_STOCK_MEMBER_COUNT
        and identity_key == "full_inchikey"
    )
    result = {
        "schema_version": PAPER_EQUIVALENT_METRIC_SCHEMA,
        "paper_equivalent_solved": bool(solved_routes),
        "paper_equivalent_solved_route_count": len(solved_routes),
        "solved_routes": solved_routes,
        "stock_comparable_to_synthex": exact_paper_stock,
        "stock_catalog_name": catalog_name,
        "stock_member_count": member_count,
        "stock_identity_key": identity_key,
        "required_stock_catalog_name": "ZINC+eMolecules",
        "required_stock_member_count": SYNTHEX_STOCK_MEMBER_COUNT,
        "required_stock_identity_key": "full_inchikey",
        "comparison_disposition": (
            "paper_comparable"
            if exact_paper_stock
            else "metric_valid_for_bound_stock_but_not_paper_stock_comparable"
        ),
        "independent_axes": {
            "reaction_validation": "reported_elsewhere_not_required",
            "exact_evidence": "reported_elsewhere_not_required",
            "condition_completeness": "reported_elsewhere_not_required",
        },
        "semantics": {
            "existential_one_route_metric": True,
            "target_rooted_topology_required": True,
            "all_leaves_must_hit_one_bound_stock_oracle": True,
            "configured_minimum_route_count_not_used": True,
            "reaction_validation_not_used": True,
            "evidence_not_used": True,
            "conditions_not_used": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


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
    "PAPER_EQUIVALENT_METRIC_SCHEMA",
    "SYNTHEX_STOCK_MEMBER_COUNT",
    "compile_paper_equivalent_metric",
]
