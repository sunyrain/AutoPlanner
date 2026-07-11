"""Canonical route-domain contracts and multi-source fusion."""

from cascade_planner.routes.admission import (
    RETROSYNTHETIC_ADMISSION_SCHEMA,
    RetrosyntheticAdmissionPolicy,
    audit_retrosynthetic_candidate,
)
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
from cascade_planner.routes.domain import (
    ROUTE_HYPERGRAPH_OVERLAY_SCHEMA,
    ROUTE_NEIGHBORHOOD_SCHEMA,
    AlternativeSet,
    EvidenceClaim,
    MoleculeIdentity,
    ReactionCandidateEnvelope,
    ReactionHyperedge,
    RouteVariant,
)
from cascade_planner.routes.overlay import build_route_hypergraph_v2_overlay

__all__ = [
    "RETROSYNTHETIC_ADMISSION_SCHEMA",
    "RetrosyntheticAdmissionPolicy",
    "audit_retrosynthetic_candidate",
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
    "ROUTE_HYPERGRAPH_OVERLAY_SCHEMA",
    "ROUTE_NEIGHBORHOOD_SCHEMA",
    "MoleculeIdentity",
    "EvidenceClaim",
    "ReactionCandidateEnvelope",
    "ReactionHyperedge",
    "AlternativeSet",
    "RouteVariant",
    "build_route_hypergraph_v2_overlay",
]
