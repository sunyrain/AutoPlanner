"""Small dependency helpers for the V4 target runtime."""
from __future__ import annotations

from typing import Any, Mapping


CHEMENZY_PROFILE_DEFAULTS = {
    "fast": {"steps": 8, "iterations": 20, "topk": 50, "timeout": 180.0},
    "standard": {"steps": 14, "iterations": 60, "topk": 100, "timeout": 600.0},
    "proof": {"steps": 20, "iterations": 120, "topk": 180, "timeout": 1200.0},
}


def inventory_snapshot_builder(payload: Mapping[str, Any]) -> Any:
    path = str(payload.get("inventory_snapshot_path") or "").strip()
    if not path:
        return None
    from cascade_planner.interfaces.live_stock import load_versioned_inventory_snapshot

    frozen = load_versioned_inventory_snapshot(path)

    def builder(_smiles: Any, **_kwargs: Any) -> Any:
        return frozen

    return builder


__all__ = ["CHEMENZY_PROFILE_DEFAULTS", "inventory_snapshot_builder"]
