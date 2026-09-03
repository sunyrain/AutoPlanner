"""Explicit AutoPlannerLLM worker for the active live benchmark shell."""
from __future__ import annotations

import os
from typing import Any

from cascade_planner.cascadeboard.live_benchmark import build_parser, run_from_args
from cascade_planner.research.autoplannrellm.controller import (
    DeepSeekSelectionController,
)
from cascade_planner.research.autoplannrellm.proposals import append_llm_candidate
from cascade_planner.route_tree.extensions import RouteTreeExtensions
from cascade_planner.route_tree.runtime import default_route_tree_runtime


def build_route_tree_extensions() -> RouteTreeExtensions:
    if not _env_truthy("AUTOPLANNRELLM_ENABLE"):
        return RouteTreeExtensions()

    controller_factory = None
    if _env_truthy_default("AUTOPLANNRELLM_LLM_SELECTION", True):
        controller: Any | None = None

        def controller_factory() -> Any:
            nonlocal controller
            if controller is None:
                controller = DeepSeekSelectionController(
                    fallback_runtime=default_route_tree_runtime()
                )
            return controller

    candidate_appender = (
        append_llm_candidate
        if _env_truthy_default("AUTOPLANNRELLM_ADD_LLM_CANDIDATE", True)
        else None
    )
    return RouteTreeExtensions(
        controller_factory=controller_factory,
        candidate_appender=candidate_appender,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_from_args(args, route_tree_extensions=build_route_tree_extensions())


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
