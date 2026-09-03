"""Read-only product profiles compiled from route proof vectors."""
from __future__ import annotations

from typing import Any, Mapping


PRODUCT_PROFILE_ORDER = (
    "exploration_closed",
    "reaction_validated",
    "literature_grounded",
    "condition_complete",
    "procurement_closed",
    "process_ready",
)


def product_profiles(
    proof_vector: Mapping[str, Any],
    *,
    closure_profile: str,
) -> dict[str, bool]:
    """Compile named claims without writing proof back to the graph."""

    vector = dict(proof_vector)
    boundary_closed = closure_profile in {
        "exploration_closed",
        "procurement_closed",
        "in_house_closed",
    }
    reaction_validated = vector.get("reaction") == "all_validated"
    literature_grounded = bool(
        reaction_validated
        and vector.get("identity") == "all_source_exact"
        and vector.get("sources") in {"single_group", "independent_2_plus"}
    )
    condition_complete = bool(
        reaction_validated
        and vector.get("conditions") == "source_exact"
        and vector.get("condition_completeness") == "complete"
    )
    procurement_closed = vector.get("stock") in {"offer_verified", "in_house"}
    process_ready = bool(
        literature_grounded
        and condition_complete
        and procurement_closed
        and vector.get("process") == "executable_candidate"
    )
    return {
        "exploration_closed": boundary_closed,
        "reaction_validated": reaction_validated,
        "literature_grounded": literature_grounded,
        "condition_complete": condition_complete,
        "procurement_closed": procurement_closed,
        "process_ready": process_ready,
    }


__all__ = ["PRODUCT_PROFILE_ORDER", "product_profiles"]
