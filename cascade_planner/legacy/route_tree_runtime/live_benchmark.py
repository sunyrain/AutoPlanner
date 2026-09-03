"""Legacy live-benchmark worker for reservoir-distilled checkpoint replay."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from cascade_planner.cascadeboard.live_benchmark import build_parser, run_from_args
from cascade_planner.legacy.guard import require_legacy_research_enabled
from cascade_planner.route_tree.extensions import RouteTreeExtensions
from cascade_planner.route_tree.runtime import default_route_tree_runtime
from cascade_planner.route_tree.source_gate import default_source_gate


RESERVOIR_CONTROLLER_ENV = "AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER"
CASCADE_ORACLE_ENABLE_ENV = "AUTOPLANNER_ENABLE_CASCADE_ORACLE_VALUE"
CASCADE_ORACLE_PAYLOAD_ENV = "AUTOPLANNER_CASCADE_ORACLE_PAYLOAD"
CASCADE_ORACLE_WEIGHT_ENV = "AUTOPLANNER_CASCADE_ORACLE_ACTION_WEIGHT"
_UNSET = object()


def build_route_tree_extensions() -> RouteTreeExtensions:
    """Translate the frozen reservoir environment contract into explicit adapters."""

    checkpoint = str(os.environ.get(RESERVOIR_CONTROLLER_ENV) or "").strip()
    oracle_enabled = _env_truthy(CASCADE_ORACLE_ENABLE_ENV)
    oracle_payload = str(os.environ.get(CASCADE_ORACLE_PAYLOAD_ENV) or "").strip()
    if not checkpoint and not oracle_enabled:
        return RouteTreeExtensions()
    require_legacy_research_enabled("reservoir-distilled route-tree checkpoint replay")
    return RouteTreeExtensions(
        controller_factory=(
            _lazy_factory(lambda: _build_controller(checkpoint))
            if checkpoint
            else None
        ),
        source_gate_factory=(
            _lazy_factory(lambda: _build_source_gate(checkpoint))
            if checkpoint
            else None
        ),
        action_value_advisor_factory=(
            _lazy_factory(lambda: _build_action_value_advisor(oracle_payload))
            if oracle_enabled and oracle_payload
            else None
        ),
        action_value_advisor_weight=(
            _env_float(CASCADE_ORACLE_WEIGHT_ENV, 0.0)
            if oracle_enabled and oracle_payload
            else 0.0
        ),
    )


def _build_controller(checkpoint: str) -> Any:
    from cascade_planner.legacy.route_tree_runtime.reservoir_distilled import (
        ReservoirDistilledControllerRuntime,
        UnavailableReservoirRouteTreeRuntime,
    )

    fallback = default_route_tree_runtime()
    if not Path(checkpoint).exists():
        return UnavailableReservoirRouteTreeRuntime(
            "missing_checkpoint",
            fallback_runtime=fallback,
        )
    try:
        return ReservoirDistilledControllerRuntime(checkpoint, fallback_runtime=fallback)
    except Exception as exc:
        return UnavailableReservoirRouteTreeRuntime(
            f"{type(exc).__name__}:load_failed",
            fallback_runtime=fallback,
        )


def _build_source_gate(checkpoint: str) -> Any:
    from cascade_planner.legacy.route_tree_runtime.reservoir_distilled import (
        ReservoirDistilledControllerRuntime,
        UnavailableReservoirSourceGate,
    )

    fallback = default_source_gate()
    if not Path(checkpoint).exists():
        return UnavailableReservoirSourceGate(
            "missing_checkpoint",
            fallback_source_gate=fallback,
        )
    try:
        return ReservoirDistilledControllerRuntime(
            checkpoint,
            fallback_source_gate=fallback,
        )
    except Exception as exc:
        return UnavailableReservoirSourceGate(
            f"{type(exc).__name__}:load_failed",
            fallback_source_gate=fallback,
        )


def _build_action_value_advisor(payload_path: str) -> Any | None:
    if not Path(payload_path).exists():
        return None
    try:
        from cascade_planner.legacy.route_tree_runtime.cascade_oracle import (
            CascadeOracleRuntime,
        )

        return CascadeOracleRuntime(payload_path)
    except Exception:
        return None


def _lazy_factory(builder: Callable[[], Any]) -> Callable[[], Any]:
    value: Any = _UNSET

    def load() -> Any:
        nonlocal value
        if value is _UNSET:
            value = builder()
        return value

    return load


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    require_legacy_research_enabled("reservoir-distilled live benchmark worker")
    args = build_parser().parse_args()
    run_from_args(args, route_tree_extensions=build_route_tree_extensions())


if __name__ == "__main__":
    main()


__all__ = ["build_route_tree_extensions", "main"]
