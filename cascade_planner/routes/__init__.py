"""Canonical route-domain contracts and multi-source fusion."""

from cascade_planner.routes.consensus import (
    RETROSYNTHESIS_CANDIDATE_SCHEMA,
    RETROSYNTHESIS_PROPOSAL_REPORT_PAYLOAD_SCHEMA,
    ROUTE_CONSENSUS_SCHEMA,
    consensus_to_blackboard_proposals,
    fuse_route_candidates,
    normalize_route_candidate,
    validate_retrosynthesis_report_payload,
)
from cascade_planner.routes.graph import (
    GRAPH_SCHEMA,
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    select_route_consensus_frontier,
    validate_route_consensus_expansion,
)
from cascade_planner.routes.adapters import rebuild_consensus_graph_from_blackboard

__all__ = [
    "RETROSYNTHESIS_CANDIDATE_SCHEMA",
    "RETROSYNTHESIS_PROPOSAL_REPORT_PAYLOAD_SCHEMA",
    "ROUTE_CONSENSUS_SCHEMA",
    "consensus_to_blackboard_proposals",
    "fuse_route_candidates",
    "normalize_route_candidate",
    "validate_retrosynthesis_report_payload",
    "GRAPH_SCHEMA",
    "assemble_route_consensus_graph",
    "make_route_consensus_expansion",
    "select_route_consensus_frontier",
    "validate_route_consensus_expansion",
    "rebuild_consensus_graph_from_blackboard",
]
