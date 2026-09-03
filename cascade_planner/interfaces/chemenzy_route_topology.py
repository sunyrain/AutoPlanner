"""Measure complete provider-route conservation through canonical ingestion."""
from __future__ import annotations

from typing import Any, Mapping


def compile_route_topology_lineage(
    route: Mapping[str, Any],
    *,
    alias: str,
    imported_proposal_ids: set[str],
    canonical_proposal_ids: set[str],
    applicable: bool,
) -> dict[str, Any]:
    step_proposal_ids = (
        [
            f"{alias}:step:{index}"
            for index, _step in enumerate(route.get("steps") or [], start=1)
        ]
        if alias
        else []
    )
    imported = [
        proposal_id
        for proposal_id in step_proposal_ids
        if proposal_id in imported_proposal_ids
    ]
    bound = [
        proposal_id
        for proposal_id in step_proposal_ids
        if proposal_id in canonical_proposal_ids
    ]
    provider_count = int(
        dict(route.get("normalization_audit") or {}).get("raw_step_count")
        or route.get("raw_step_count")
        or 0
    )
    normalized_count = len(route.get("steps") or [])
    accepted = bool(
        applicable
        and provider_count == normalized_count
        and normalized_count == len(imported)
        and len(imported) == len(bound)
    )
    return {
        "step_proposal_ids": step_proposal_ids,
        "provider_step_count": provider_count,
        "normalized_step_count": normalized_count,
        "imported_proposal_count": len(imported),
        "canonical_bound_step_count": len(bound),
        "missing_imported_proposal_ids": sorted(
            set(step_proposal_ids) - set(imported)
        ),
        "missing_canonical_proposal_ids": sorted(set(imported) - set(bound)),
        "topology_conservation_applicable": applicable,
        "topology_conservation_accepted": accepted if applicable else None,
    }


def topology_conservation_failures(
    route_lineage: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(lineage)
        for lineage in route_lineage
        if lineage.get("topology_conservation_applicable") is True
        and lineage.get("topology_conservation_accepted") is not True
    ]
