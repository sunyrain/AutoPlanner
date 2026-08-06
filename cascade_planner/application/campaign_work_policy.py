"""Fixed work policy used to compile one shared action frontier."""
from __future__ import annotations

from dataclasses import replace

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


def unified_frontier_acceptance(
    configured: RetrosynthesisAcceptanceSpec,
) -> RetrosynthesisAcceptanceSpec:
    """Keep the action frontier scientifically complete for every result view.

    The caller's stock boundary and any stricter numerical requirements remain
    meaningful inputs.  Lower proof/route requirements cannot disable work in
    the shared frontier, which prevents a route-search evaluation from silently
    turning off validation and evidence actions.
    """

    return replace(
        configured,
        minimum_complete_routes=max(2, int(configured.minimum_complete_routes)),
        minimum_edge_proof_level=max(3, int(configured.minimum_edge_proof_level)),
        minimum_independent_source_groups=max(
            2,
            int(configured.minimum_independent_source_groups),
        ),
        require_distinct_edge_sets=True,
    )


__all__ = ["unified_frontier_acceptance"]
