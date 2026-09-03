"""Discover proposal-only whole-cell and hybrid execution windows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.route_execution_capabilities import (
    normalize_program_execution_catalog,
)
from cascade_planner.application.route_structure_matching import (
    match_structure_capability,
)
from cascade_planner.application.route_innovation_windows import (
    route_window_boundary,
)


def split_route_innovation_capabilities(
    value: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate enzymatic capabilities from execution-program capabilities."""

    rows = (
        list(value.get("capabilities") or [])
        if isinstance(value, Mapping)
        else list(value)
    )
    biocatalytic: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if str(row.get("execution_domain") or "enzymatic") in {
            "whole_cell",
            "hybrid",
        }:
            execution.append(row)
        else:
            biocatalytic.append(row)
    return biocatalytic, execution


def discover_program_execution_windows(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    paths: Iterable[Sequence[str]],
    capabilities: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match normalized execution capabilities to exact route boundaries."""

    normalized, rejected = normalize_program_execution_catalog(capabilities)
    candidates: list[dict[str, Any]] = []
    for path in paths:
        boundary = route_window_boundary(graph, path)
        if not boundary:
            continue
        for capability in normalized:
            audit = match_structure_capability(
                capability,
                boundary["precursor_smiles"],
                boundary["product_smiles"],
                window_steps=len(path),
            )
            if audit["accepted"] is True:
                candidates.append(
                    _execution_candidate(
                        route,
                        capability=capability,
                        boundary=boundary,
                        edge_ids=path,
                        match_audit=audit,
                    )
                )
    return candidates, rejected


def _execution_candidate(
    route: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    boundary: Mapping[str, Any],
    edge_ids: Sequence[str],
    match_audit: Mapping[str, Any],
) -> dict[str, Any]:
    domain = str(capability["execution_domain"])
    isolated_operations = sum(
        row.get("isolated_operation") is True
        for row in capability["operation_blueprints"]
    )
    identity = {
        "capability_id": capability["capability_id"],
        "execution_domain": domain,
        "edge_ids": list(edge_ids),
    }
    boundary_ready = int(boundary.get("minimum_boundary_proof_level") or 0) >= 1
    warnings = {
        "EXACT_SUBSTRATE_UNVALIDATED",
        "SPECIALIZED_EXECUTION_VALIDATION_REQUIRED",
    }
    if not boundary_ready:
        warnings.add("ROUTE_BOUNDARY_BELOW_L1")
    if domain == "whole_cell":
        warnings.add("WHOLE_CELL_PROCESS_UNVALIDATED")
    else:
        warnings.add("HYBRID_OPERATION_COMPATIBILITY_UNVALIDATED")
    return {
        "candidate_id": f"route-innovation:{_digest(identity)[:24]}",
        "candidate_kind": "program_execution_window",
        "execution_domain": domain,
        "review_status": (
            "ready_for_specialized_execution_screen"
            if boundary_ready
            else "requires_boundary_materialization"
        ),
        "priority_score": round(float(match_audit.get("match_score") or 0.0), 6),
        "estimated_net_operation_savings": len(edge_ids) - isolated_operations,
        "capability_id": str(capability["capability_id"]),
        "boundary": {**dict(boundary), "replaced_edge_ids": list(edge_ids)},
        "execution_capability": dict(capability),
        "match_audit": dict(match_audit),
        "not_program_yet": True,
        "warning_codes": sorted(warnings),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "discover_program_execution_windows",
    "split_route_innovation_capabilities",
]
