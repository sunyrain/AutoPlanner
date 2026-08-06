"""Small dependency helpers for the V4 target runtime."""
from __future__ import annotations

from typing import Any, Mapping


TARGET_PROFILE_DEFAULTS = {
    "fast": {
        "steps": 8,
        "iterations": 20,
        "topk": 50,
        "timeout": 180.0,
        "workers": 2,
        "max_input_tokens": 90_000,
        "max_output_tokens": 22_000,
        "max_model_wall_time_s": 900.0,
        "max_director_wall_time_s": 600.0,
    },
    "standard": {
        "steps": 14,
        "iterations": 60,
        "topk": 100,
        "timeout": 600.0,
        "workers": 4,
        "max_input_tokens": 90_000,
        "max_output_tokens": 22_000,
        "max_model_wall_time_s": 900.0,
        "max_director_wall_time_s": 600.0,
    },
    "proof": {
        "steps": 20,
        "iterations": 60,
        "topk": 120,
        "timeout": 3_600.0,
        "workers": 8,
        "max_input_tokens": 1_200_000,
        "max_output_tokens": 200_000,
        "max_model_wall_time_s": 1_800.0,
        "max_director_wall_time_s": 1_800.0,
    },
}

# Backward-compatible name for callers that only consume ChemEnzy controls.
CHEMENZY_PROFILE_DEFAULTS = TARGET_PROFILE_DEFAULTS


def inventory_snapshot_builder(payload: Mapping[str, Any]) -> Any:
    path = str(payload.get("inventory_snapshot_path") or "").strip()
    if not path:
        return None
    from cascade_planner.interfaces.live_stock import load_versioned_inventory_snapshot

    frozen = load_versioned_inventory_snapshot(path)

    def builder(_smiles: Any, **_kwargs: Any) -> Any:
        return frozen

    return builder


__all__ = [
    "CHEMENZY_PROFILE_DEFAULTS",
    "TARGET_PROFILE_DEFAULTS",
    "inventory_snapshot_builder",
]
