"""Normalize provider-quarantined ChemEnzy routes for L0 review."""
from __future__ import annotations

from typing import Any, Callable, Mapping


RouteNormalizer = Callable[..., dict[str, Any]]


def normalized_quarantined_routes(
    value: Mapping[str, Any],
    *,
    start_index: int,
    normalizer: RouteNormalizer,
) -> list[dict[str, Any]]:
    rows = value.get("quarantined_routes")
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for offset, route in enumerate(rows):
        if not isinstance(route, Mapping) or not isinstance(route.get("steps"), list):
            continue
        row = normalizer(dict(route), route_index=start_index + offset)
        row["proposal_eligible"] = False
        row["admission_reasons"] = sorted(
            {
                *row.get("admission_reasons", []),
                "provider_route_quarantined",
                *(str(value) for value in route.get("warning_codes") or []),
            }
        )
        normalized.append(row)
    return normalized


__all__ = ["normalized_quarantined_routes"]
