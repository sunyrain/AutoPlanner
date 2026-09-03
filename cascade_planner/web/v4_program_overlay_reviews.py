"""Read-only Program reviews used by the route-workbench shadow layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CAPABILITY_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "route_innovation_capabilities.v1.json"
)


def collect_program_overlay_reviews(
    gateway: Any,
    run_id: str,
    snapshot: Mapping[str, Any],
    *,
    capability_catalog_path: Path = DEFAULT_CAPABILITY_CATALOG,
) -> tuple[dict[str, Any], ...]:
    """Screen bounded routes without admitting or persisting any proposal."""

    try:
        capabilities = json.loads(capability_catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    review = getattr(gateway, "route_program_innovations", None)
    if not callable(review):
        return ()
    routes = [
        dict(value)
        for value in dict(snapshot.get("routes") or {}).values()
        if isinstance(value, Mapping) and len(value.get("edge_ids") or []) >= 2
    ]
    results: list[dict[str, Any]] = []
    for route in routes[:5]:
        route_id = str(route.get("route_id") or "")
        if not route_id:
            continue
        try:
            value = review(run_id, route_id=route_id, capabilities=capabilities)
        except Exception:  # A display enhancement must never hide the canonical route.
            continue
        if isinstance(value, Mapping):
            results.append(dict(value))
    return tuple(results)


__all__ = ["collect_program_overlay_reviews"]
