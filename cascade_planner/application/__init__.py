"""Lazy public API for canonical V4 application contracts."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_V4_EXPORTS = {
    "CampaignAction": (
        "cascade_planner.application.campaign_actions",
        "CampaignAction",
    ),
    "CampaignActionKind": (
        "cascade_planner.application.campaign_actions",
        "CampaignActionKind",
    ),
    "CampaignActionOpportunity": (
        "cascade_planner.application.campaign_actions",
        "CampaignActionOpportunity",
    ),
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
    "compile_campaign_quality_state": (
        "cascade_planner.application.campaign_quality_state",
        "compile_campaign_quality_state",
    ),
    "CampaignResourceBudget": (
        "cascade_planner.application.unified_campaign_spec",
        "CampaignResourceBudget",
    ),
    "StockOracleReference": (
        "cascade_planner.application.unified_campaign_spec",
        "StockOracleReference",
    ),
    "TargetConstraints": (
        "cascade_planner.application.unified_campaign_spec",
        "TargetConstraints",
    ),
    "UnifiedCampaignSpec": (
        "cascade_planner.application.unified_campaign_spec",
        "UnifiedCampaignSpec",
    ),
    "unified_frontier_acceptance": (
        "cascade_planner.application.campaign_work_policy",
        "unified_frontier_acceptance",
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
    "compile_action_opportunities": (
        "cascade_planner.application.campaign_actions",
        "compile_action_opportunities",
    ),
    "compile_action_preflight": (
        "cascade_planner.application.action_preflight",
        "compile_action_preflight",
    ),
    "bind_scheduled_action": (
        "cascade_planner.application.campaign_actions",
        "bind_scheduled_action",
    ),
    "compile_campaign_snapshot": (
        "cascade_planner.application.campaign_trajectory",
        "compile_campaign_snapshot",
    ),
    "compile_campaign_trajectory": (
        "cascade_planner.application.campaign_trajectory",
        "compile_campaign_trajectory",
    ),
    "compile_action_counts": (
        "cascade_planner.application.campaign_trajectory",
        "compile_action_counts",
    ),
    "compile_route_snapshot": (
        "cascade_planner.application.campaign_trajectory",
        "compile_route_snapshot",
    ),
    "compile_trajectory_bindings": (
        "cascade_planner.application.campaign_trajectory",
        "compile_trajectory_bindings",
    ),
    "project_campaign_trajectory_at_cutoff": (
        "cascade_planner.application.campaign_trajectory",
        "project_campaign_trajectory_at_cutoff",
    ),
    "compile_campaign_review_bundle": (
        "cascade_planner.application.campaign_review_bundle",
        "compile_campaign_review_bundle",
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
    "schedule_next_action": (
        "cascade_planner.application.action_scheduler",
        "schedule_next_action",
    ),
}

__all__ = sorted(_V4_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _V4_EXPORTS.get(name)
    if target is not None:
        module_name, attribute_name = target
        value = getattr(import_module(module_name), attribute_name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_V4_EXPORTS})
