"""Lazy candidate expansion for CascadeBoard++.

When the initial candidate pool is insufficient (all particles fail constraints,
energy too high, no stock-reachable leaves), dynamically expand the search.
"""
from __future__ import annotations

from cascade_planner.cascadeboard import CascadeBoard, CompiledConstraints
from cascade_planner.cascadeboard.candidate_graph import (
    CandidateHypergraph, CandidateReaction,
)
from cascade_planner.cascadeboard.energy_api import EnergyAPI


def diagnose_failure(
    board: CascadeBoard,
    energy_api: EnergyAPI,
) -> str:
    """Diagnose why a board failed and suggest expansion strategy."""
    if not board.slots:
        return "empty_board"

    # Check stock
    stock_score = energy_api.score_stock(board)
    if stock_score < 0.3:
        return "no_stock_reachable"

    # Check compatibility
    if board.compatibility_scores and min(board.compatibility_scores) < 0.2:
        return "condition_conflict"

    # Check retro scores
    retro_scores = [s.e_retro or 0 for s in board.slots]
    if retro_scores and max(retro_scores) < 0.3:
        return "weak_candidates"

    # Check enzyme
    enz_slots = [s for s in board.slots if s.is_enzymatic()]
    if enz_slots and all((s.e_enzyme or 0) < 0.2 for s in enz_slots):
        return "enzyme_missing"

    return "unknown"


def lazy_expand(
    graph: CandidateHypergraph,
    board: CascadeBoard,
    compiled: CompiledConstraints | None,
    energy_api: EnergyAPI,
    reason: str = "",
) -> bool:
    """Attempt to expand the candidate graph based on failure diagnosis.

    Returns True if expansion was performed.
    """
    if not reason:
        reason = diagnose_failure(board, energy_api)

    expanded = False

    if reason == "no_stock_reachable":
        # Increase depth by 1
        if graph.max_depth < 5:
            graph.max_depth += 1
            if graph.root:
                graph._expand(graph.root, compiled)
            expanded = True

    elif reason == "weak_candidates":
        # Increase branch factor
        if graph.branch_factor < 20:
            graph.branch_factor += 5
            if graph.root:
                graph._expand(graph.root, compiled)
            expanded = True

    elif reason == "condition_conflict":
        # Try allowing purification (relax one_pot constraint)
        # This is handled at the constraint level, not graph level
        pass

    elif reason == "enzyme_missing":
        # Increase branch factor for enzymatic candidates
        if graph.branch_factor < 20:
            graph.branch_factor += 5
            if graph.root:
                graph._expand(graph.root, compiled)
            expanded = True

    return expanded


def should_expand(
    particles: list,
    compiled: CompiledConstraints | None = None,
    energy_threshold: float = -3.0,
) -> bool:
    """Check if expansion is needed based on particle quality."""
    if not particles:
        return True

    # All particles have very high energy (bad)
    energies = [getattr(p, "energy", float("inf")) for p in particles]
    if min(energies) > energy_threshold:
        return True

    # Very low diversity
    sigs = set()
    for p in particles:
        board = getattr(p, "board", None)
        if board:
            sig = tuple(s.reaction_type for s in board.slots)
            sigs.add(sig)
    if len(sigs) < max(2, len(particles) // 4):
        return True

    return False
