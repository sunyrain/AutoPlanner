"""Frozen optional route-tree scorers retained for checkpoint replay."""
from __future__ import annotations

import os
from typing import Any

from cascade_planner.legacy.guard import require_legacy_research_enabled


def ccts_runtime_from_env() -> Any | None:
    """Load the requested frozen CCTS scorer after explicit legacy opt-in."""

    ccts_v3_requested = bool(os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_MODEL"))
    ccts_v0_requested = bool(os.environ.get("AUTOPLANNER_CCTS_V0_MODEL"))
    if not ccts_v3_requested and not ccts_v0_requested:
        return None
    require_legacy_research_enabled("route-tree CCTS checkpoint runtime")
    if ccts_v3_requested:
        try:
            from cascade_planner.legacy.route_tree_runtime.ccts_v3_runtime import (
                ccts_v3_runtime_from_env,
            )

            return ccts_v3_runtime_from_env()
        except Exception:
            return None
    try:
        from cascade_planner.legacy.route_tree_runtime.ccts_v0 import ccts_v0_runtime_from_env

        return ccts_v0_runtime_from_env()
    except Exception:
        return None


def plan_with_legacy_ccts(**kwargs: Any) -> list[Any]:
    """Run the canonical planner with an explicitly requested frozen scorer."""

    from cascade_planner.route_tree.search import plan_with_route_tree

    require_legacy_research_enabled("route-tree CCTS checkpoint replay")
    if "ccts_scorer" in kwargs or "ccts_weight" in kwargs:
        raise TypeError("legacy CCTS wrapper owns ccts_scorer and ccts_weight")
    try:
        weight = float(os.environ.get("AUTOPLANNER_ROUTE_TREE_CCTS_WEIGHT") or 0.35)
    except (TypeError, ValueError):
        weight = 0.35
    return plan_with_route_tree(
        **kwargs,
        ccts_scorer=ccts_runtime_from_env(),
        ccts_weight=weight,
    )


__all__ = ["ccts_runtime_from_env", "plan_with_legacy_ccts"]
