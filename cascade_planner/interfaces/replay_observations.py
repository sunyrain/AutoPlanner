"""Scientific observation metrics shared by replay acceptance reports."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.fact_lifecycle import summarize_fact_lifecycle


def replay_scientific_observations(
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    workbench: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle = summarize_fact_lifecycle(graph)
    edge_proofs = dict(portfolio.get("edge_proofs") or {})
    active_exact_ids = {
        str(record_id)
        for proof in edge_proofs.values()
        for record_id in proof.get("exact_record_ids") or []
    }
    active_procedure_ids = {
        str(record_id)
        for proof in edge_proofs.values()
        for record_id in proof.get("procedure_record_ids") or []
    }
    procedures = dict(graph.get("procedure_records") or {})
    active_procedures = [
        procedures[record_id]
        for record_id in sorted(active_procedure_ids)
        if record_id in procedures
    ]
    profile_counts = dict(
        dict(workbench.get("portfolio") or {}).get("acceptance_profile_counts") or {}
    )
    return {
        "accepted": portfolio.get("accepted") is True,
        "complete_route_count": int(
            dict(portfolio.get("closeout") or {}).get("complete_route_count") or 0
        ),
        "selected_route_count": len(portfolio.get("selected_routes") or []),
        "hyperedge_count": len(graph.get("edges") or {}),
        "validated_edge_count": sum(
            proof.get("reaction_validated") is True for proof in edge_proofs.values()
        ),
        "exact_record_count": len(graph.get("exact_records") or {}),
        "active_exact_record_count": len(active_exact_ids),
        "procedure_record_count": len(procedures),
        "active_procedure_record_count": len(active_procedure_ids),
        "condition_complete_procedure_count": sum(
            dict(record.get("condition_completeness") or {}).get("complete") is True
            for record in active_procedures
        ),
        "condition_partial_procedure_count": sum(
            record.get("procedure_status") == "condition_partial"
            for record in active_procedures
        ),
        "condition_unparsed_procedure_count": sum(
            record.get("procedure_status") == "procedure_located_condition_unparsed"
            for record in active_procedures
        ),
        "condition_complete_route_count": int(
            profile_counts.get("condition_complete") or 0
        ),
        "process_ready_route_count": int(profile_counts.get("process_ready") or 0),
        "stock_terminal_count": sum(
            dict(proof).get("accepted") is True
            for proof in dict(portfolio.get("leaf_proofs") or {}).values()
        ),
        "independent_source_groups": sorted(
            {
                str(group)
                for proof in edge_proofs.values()
                for group in proof.get("independent_source_groups") or []
                if str(group)
            }
        ),
        "fact_lifecycle_event_count": lifecycle["event_count"],
        "inactive_fact_count": lifecycle["inactive_fact_count"],
        "revoked_fact_count": lifecycle["revoked_fact_count"],
        "expired_fact_count": lifecycle["expired_fact_count"],
    }


__all__ = ["replay_scientific_observations"]
