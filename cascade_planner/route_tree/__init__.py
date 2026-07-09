"""Route-tree planner components.

This package is the new route-level control surface. Single-step engines are
adapted as proposal tools; search control is centralized in a route-tree state
model with legacy-safe fallback.
"""
from __future__ import annotations

from cascade_planner.route_tree.proposal_rankers import SourceSpecificProposalRankers
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.reservoir_distilled import (
    ReservoirDistilledController,
    ReservoirDistilledControllerRuntime,
    UnavailableReservoirRouteTreeRuntime,
    UnavailableReservoirSourceGate,
    load_reservoir_controller_runtime,
    reservoir_controller_feature_dim,
    reservoir_controller_feature_vector,
)
from cascade_planner.route_tree.runtime import RouteTreeRuntime, default_route_tree_runtime
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState, RouteTreeStep
from cascade_planner.route_tree.search import NeuralGuidedAOSearch, plan_with_route_tree
from cascade_planner.route_tree.source_gate import CascadeSourcePolicyGate, LearnedSourceGate, SourceGate
from cascade_planner.route_tree.verifier import RouteVerifier

__all__ = [
    "CandidateAction",
    "LearnedSourceGate",
    "CascadeSourcePolicyGate",
    "NeuralGuidedAOSearch",
    "ReservoirDistilledController",
    "ReservoirDistilledControllerRuntime",
    "ProposalContext",
    "RetroEngineProposalTool",
    "UnavailableReservoirRouteTreeRuntime",
    "UnavailableReservoirSourceGate",
    "RouteTreeRuntime",
    "RouteVerifier",
    "RouteTreeState",
    "RouteTreeStep",
    "SourceGate",
    "SourceSpecificProposalRankers",
    "default_route_tree_runtime",
    "load_reservoir_controller_runtime",
    "plan_with_route_tree",
    "reservoir_controller_feature_dim",
    "reservoir_controller_feature_vector",
]
