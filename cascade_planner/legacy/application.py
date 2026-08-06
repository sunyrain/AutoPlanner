"""Frozen application-layer exports retained for historical callers only."""
from __future__ import annotations

from typing import Any

from cascade_planner.legacy._exports import load_legacy_export


_EXPORTS = {
    "FRONTIER_LEDGER_SCHEMA": (
        "cascade_planner.legacy.application_runtime.frontier_ledger",
        "FRONTIER_LEDGER_SCHEMA",
    ),
    "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA": (
        "cascade_planner.legacy.application_runtime.retrosynthesis_acceptance",
        "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA",
    ),
    "FrontierCompletenessReport": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "FrontierCompletenessReport",
    ),
    "FrontierExecutor": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "FrontierExecutor",
    ),
    "FrontierJob": ("cascade_planner.legacy.application_runtime.frontier_scheduler", "FrontierJob"),
    "FrontierJobState": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "FrontierJobState",
    ),
    "FrontierScheduler": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "FrontierScheduler",
    ),
    "PersistentFrontierQueue": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "PersistentFrontierQueue",
    ),
    "RouteDeficit": (
        "cascade_planner.legacy.application_runtime.route_deficit_queue",
        "RouteDeficit",
    ),
    "RouteDeficitKind": (
        "cascade_planner.legacy.application_runtime.route_deficit_queue",
        "RouteDeficitKind",
    ),
    "RoutePortfolioItem": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "RoutePortfolioItem",
    ),
    "RoutePortfolioReport": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "RoutePortfolioReport",
    ),
    "assess_frontier_completeness": (
        "cascade_planner.legacy.application_runtime.frontier_scheduler",
        "assess_frontier_completeness",
    ),
    "build_route_verifier_bundle": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "build_route_verifier_bundle",
    ),
    "compile_route_deficit_queue": (
        "cascade_planner.legacy.application_runtime.route_deficit_queue",
        "compile_route_deficit_queue",
    ),
    "derive_portfolio_bindings": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "derive_portfolio_bindings",
    ),
    "exact_edge_signature": (
        "cascade_planner.legacy.application_runtime.frontier_ledger",
        "exact_edge_signature",
    ),
    "evaluate_retrosynthesis_acceptance": (
        "cascade_planner.legacy.application_runtime.retrosynthesis_acceptance",
        "evaluate_retrosynthesis_acceptance",
    ),
    "next_route_deficit": (
        "cascade_planner.legacy.application_runtime.route_deficit_queue",
        "next_route_deficit",
    ),
    "project_frontier_ledger": (
        "cascade_planner.legacy.application_runtime.frontier_ledger",
        "project_frontier_ledger",
    ),
    "solve_diverse_routes": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "solve_diverse_routes",
    ),
    "validate_frontier_ledger": (
        "cascade_planner.legacy.application_runtime.frontier_ledger",
        "validate_frontier_ledger",
    ),
    "validate_portfolio_replacements": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "validate_portfolio_replacements",
    ),
    "validate_route_replacement": (
        "cascade_planner.legacy.application_runtime.route_portfolio",
        "validate_route_replacement",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    value = load_legacy_export(
        name,
        _EXPORTS,
        replacement="cascade_planner.application V4 contracts",
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
