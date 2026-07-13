"""Lazy application API with V4 contracts as the primary surface.

Legacy V3 symbols remain import-compatible but are loaded only on demand.
They are inventoried compatibility paths, not owners of new scientific state.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_V4_EXPORTS = {
    "CAMPAIGN_CONTEXT_DELTA_SCHEMA": (
        "cascade_planner.application.campaign_context",
        "CAMPAIGN_CONTEXT_DELTA_SCHEMA",
    ),
    "CAMPAIGN_CONTEXT_SCHEMA": (
        "cascade_planner.application.campaign_context",
        "CAMPAIGN_CONTEXT_SCHEMA",
    ),
    "CampaignContext": ("cascade_planner.application.campaign_context", "CampaignContext"),
    "CampaignContextCompiler": (
        "cascade_planner.application.campaign_context",
        "CampaignContextCompiler",
    ),
    "CampaignContextDelta": (
        "cascade_planner.application.campaign_context",
        "CampaignContextDelta",
    ),
    "CampaignContextError": (
        "cascade_planner.application.campaign_context",
        "CampaignContextError",
    ),
    "CampaignContextTooLargeError": (
        "cascade_planner.application.campaign_context",
        "CampaignContextTooLargeError",
    ),
    "CanonicalHypergraphStore": (
        "cascade_planner.application.canonical_hypergraph",
        "CanonicalHypergraphStore",
    ),
    "CanonicalIngestionBatch": (
        "cascade_planner.application.canonical_hypergraph",
        "CanonicalIngestionBatch",
    ),
    "Deficit": ("cascade_planner.application.run_kernel", "Deficit"),
    "DeficitFrontierItem": (
        "cascade_planner.application.deficit_frontier",
        "DeficitItem",
    ),
    "PortfolioConfig": (
        "cascade_planner.application.proof_portfolio",
        "PortfolioConfig",
    ),
    "ModelCostEvent": (
        "cascade_planner.application.retrosynthesis_run_contract",
        "ModelCostEvent",
    ),
    "ProofPolicy": ("cascade_planner.application.proof_policy", "ProofPolicy"),
    "RetrosynthesisAcceptanceSpec": (
        "cascade_planner.application.retrosynthesis_run_contract",
        "RetrosynthesisAcceptanceSpec",
    ),
    "RetrosynthesisRunBudget": (
        "cascade_planner.application.retrosynthesis_run_contract",
        "RetrosynthesisRunBudget",
    ),
    "RetrosynthesisCostLedger": (
        "cascade_planner.application.retrosynthesis_run_contract",
        "RetrosynthesisCostLedger",
    ),
    "RunEvent": ("cascade_planner.application.run_kernel", "RunEvent"),
    "RunKernel": ("cascade_planner.application.run_kernel", "RunKernel"),
    "RunKernelBudgetError": (
        "cascade_planner.application.run_kernel",
        "RunKernelBudgetError",
    ),
    "RunKernelCorruptionError": (
        "cascade_planner.application.run_kernel",
        "RunKernelCorruptionError",
    ),
    "RunKernelError": ("cascade_planner.application.run_kernel", "RunKernelError"),
    "RunKernelIdempotencyConflict": (
        "cascade_planner.application.run_kernel",
        "RunKernelIdempotencyConflict",
    ),
    "RunLimits": ("cascade_planner.application.run_kernel", "RunLimits"),
    "RunSpec": ("cascade_planner.application.run_kernel", "RunSpec"),
    "RunRevision": ("cascade_planner.application.run_kernel", "RunRevision"),
    "RunState": ("cascade_planner.application.run_kernel", "RunState"),
    "RouteWorkbenchProjectionError": (
        "cascade_planner.application.route_workbench",
        "RouteWorkbenchProjectionError",
    ),
    "StopDecision": ("cascade_planner.application.run_kernel", "StopDecision"),
    "WorkerCommand": (
        "cascade_planner.application.worker_runtime",
        "WorkerCommand",
    ),
    "WorkerRuntime": (
        "cascade_planner.application.worker_runtime",
        "WorkerRuntime",
    ),
    "compile_deficit_frontier": (
        "cascade_planner.application.deficit_frontier",
        "compile_deficit_frontier",
    ),
    "compile_proof_portfolio": (
        "cascade_planner.application.proof_portfolio",
        "compile_proof_portfolio",
    ),
    "compile_route_workbench": (
        "cascade_planner.application.route_workbench",
        "compile_route_workbench",
    ),
    "compile_route_workbench_delta": (
        "cascade_planner.application.route_workbench",
        "compile_route_workbench_delta",
    ),
    "publish_proof_portfolio": (
        "cascade_planner.application.proof_portfolio",
        "publish_proof_portfolio",
    ),
}

_LEGACY_EXPORTS = {
    "FRONTIER_LEDGER_SCHEMA": (
        "cascade_planner.application.frontier_ledger",
        "FRONTIER_LEDGER_SCHEMA",
    ),
    "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA": (
        "cascade_planner.application.retrosynthesis_acceptance",
        "RETROSYNTHESIS_ACCEPTANCE_REPORT_SCHEMA",
    ),
    "FrontierCompletenessReport": (
        "cascade_planner.application.frontier_scheduler",
        "FrontierCompletenessReport",
    ),
    "FrontierExecutor": (
        "cascade_planner.application.frontier_scheduler",
        "FrontierExecutor",
    ),
    "FrontierJob": ("cascade_planner.application.frontier_scheduler", "FrontierJob"),
    "FrontierJobState": (
        "cascade_planner.application.frontier_scheduler",
        "FrontierJobState",
    ),
    "FrontierScheduler": (
        "cascade_planner.application.frontier_scheduler",
        "FrontierScheduler",
    ),
    "PersistentFrontierQueue": (
        "cascade_planner.application.frontier_scheduler",
        "PersistentFrontierQueue",
    ),
    "RouteDeficit": (
        "cascade_planner.application.route_deficit_queue",
        "RouteDeficit",
    ),
    "RouteDeficitKind": (
        "cascade_planner.application.route_deficit_queue",
        "RouteDeficitKind",
    ),
    "RoutePortfolioItem": (
        "cascade_planner.application.route_portfolio",
        "RoutePortfolioItem",
    ),
    "RoutePortfolioReport": (
        "cascade_planner.application.route_portfolio",
        "RoutePortfolioReport",
    ),
    "assess_frontier_completeness": (
        "cascade_planner.application.frontier_scheduler",
        "assess_frontier_completeness",
    ),
    "build_route_verifier_bundle": (
        "cascade_planner.application.route_portfolio",
        "build_route_verifier_bundle",
    ),
    "compile_route_deficit_queue": (
        "cascade_planner.application.route_deficit_queue",
        "compile_route_deficit_queue",
    ),
    "derive_portfolio_bindings": (
        "cascade_planner.application.route_portfolio",
        "derive_portfolio_bindings",
    ),
    "exact_edge_signature": (
        "cascade_planner.application.frontier_ledger",
        "exact_edge_signature",
    ),
    "evaluate_retrosynthesis_acceptance": (
        "cascade_planner.application.retrosynthesis_acceptance",
        "evaluate_retrosynthesis_acceptance",
    ),
    "next_route_deficit": (
        "cascade_planner.application.route_deficit_queue",
        "next_route_deficit",
    ),
    "project_frontier_ledger": (
        "cascade_planner.application.frontier_ledger",
        "project_frontier_ledger",
    ),
    "solve_diverse_routes": (
        "cascade_planner.application.route_portfolio",
        "solve_diverse_routes",
    ),
    "validate_portfolio_replacements": (
        "cascade_planner.application.route_portfolio",
        "validate_portfolio_replacements",
    ),
    "validate_frontier_ledger": (
        "cascade_planner.application.frontier_ledger",
        "validate_frontier_ledger",
    ),
    "validate_route_replacement": (
        "cascade_planner.application.route_portfolio",
        "validate_route_replacement",
    ),
}

_EXPORTS = {**_V4_EXPORTS, **_LEGACY_EXPORTS}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
