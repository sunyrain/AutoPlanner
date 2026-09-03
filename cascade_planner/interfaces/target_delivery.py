"""Honest two-stage delivery projection for live target-only jobs."""
from __future__ import annotations

from typing import Any, Mapping


def delivery_projection(
    stages: list[Mapping[str, Any]],
    *,
    job_status: str,
) -> dict[str, Any]:
    """Separate usable route delivery from slower proof closure."""

    by_name = {
        str(row.get("stage") or ""): str(row.get("status") or "")
        for row in stages
    }
    initial_status = by_name.get("initial_workbench", "")
    route_candidates_available = initial_status in {
        "completed",
        "accepted",
        "reused_or_empty",
    }
    evidence_status = by_name.get("evidence_acquisition", "")
    evidence_complete = evidence_status in {
        "completed",
        "accepted",
        "partial",
        "reused_or_empty",
        "unresolved",
        "discovered_unbound",
        "structure_bound_unproven",
    }
    normalized_job = str(job_status or "").casefold()
    if normalized_job == "cancelled":
        state = "cancelled"
    elif normalized_job == "cancelling":
        state = "cancelling"
    elif normalized_job == "failed":
        state = "failed"
    elif normalized_job == "complete":
        state = "complete"
    elif normalized_job == "unresolved":
        state = "unresolved"
    elif evidence_status == "running":
        state = "route_candidates_ready_evidence_running"
    elif route_candidates_available and not evidence_complete:
        state = "route_candidates_ready_proof_pending"
    elif evidence_complete:
        state = "proof_review_ready"
    elif by_name.get("global_campaign") == "running":
        state = "planning_routes"
    else:
        state = "initializing"
    return {
        "state": state,
        "route_candidates_available": route_candidates_available,
        "proof_closure_complete": normalized_job == "complete",
        "proof_closure_known": normalized_job in {
            "complete",
            "unresolved",
            "failed",
        },
        "evidence_stage_complete": evidence_complete,
        "workbench_available": route_candidates_available,
        "semantics": {
            "route_candidates_do_not_imply_exact_evidence": True,
            "proof_closure_may_continue_after_first_route_delivery": True,
            "execution_finished_does_not_imply_scientific_acceptance": True,
        },
    }


__all__ = ["delivery_projection"]
