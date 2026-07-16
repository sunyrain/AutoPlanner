"""Route-branch rows for the V4 RouteForest display adapter."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.harness.v4_route_evidence_projection import (
    PROOF_TIER,
    closure_label,
)


def route_branch(
    route: Mapping[str, Any],
    *,
    branch_id: str,
    step_ids: list[str],
    primary: bool,
) -> dict[str, Any]:
    level = max(0, min(4, int(route.get("proof_level") or 0)))
    complete = route.get("complete") is True
    closure_profile = str(route.get("closure_profile") or "unresolved")
    process_ready = route.get("process_ready") is True
    proof_vector = dict(route.get("proof_vector") or {})
    condition_status = str(proof_vector.get("conditions") or "missing")
    condition_complete = route.get("condition_complete") is True
    literature_grounded = route.get("literature_grounded") is True
    reaction_validated = route.get("reaction_validated") is True
    procurement_closed = route.get("procurement_closed") is True
    inactive_facts = list(route.get("inactive_facts") or [])
    reported_in_source = route.get("reported_in_source") is True
    paper_reported_step_count = int(route.get("reported_step_count") or 0)
    planner_hypothesis_step_count = int(
        route.get("planner_hypothesis_step_count") or 0
    )
    route_state_label = (
        "权威事实失效 · 路线已降级"
        if inactive_facts
        else "工艺候选可执行"
        if process_ready
        else "采购闭合 · 条件待补"
        if closure_profile in {"procurement_closed", "in_house_closed"}
        else "搜索边界闭合 · 非采购路线"
        if closure_profile == "exploration_closed"
        else "反应已验证 · 叶节点未闭合"
        if level >= 2
        else "路线骨架 · 待递归展开"
    )
    if reported_in_source and complete and not process_ready:
        route_state_label = (
            f"路线已闭合 · {paper_reported_step_count} 步文献报道 · "
            f"{planner_hypothesis_step_count} 步规划待补证"
        )
    elif reported_in_source and not complete:
        route_state_label = "文献报道路线 · 含低可信或未校准边"
    condition_label = {
        "source_exact": "来源条件",
        "source_recorded_unverified": "来源条件候选",
        "model_predicted": "预测条件",
        "missing": "条件缺失",
    }.get(condition_status, "条件状态未知")
    source_refs = [
        str(value)[7:]
        for value in route.get("badges") or []
        if str(value).startswith("source:")
    ]
    source_refs = sorted(
        set(source_refs)
        | {
            str(value)
            for value in route.get("reported_source_refs") or []
            if str(value)
        }
    )
    return {
        "branch_id": branch_id,
        "title": str(route.get("strategy") or f"Portfolio route {branch_id[-6:]}"),
        "kind": (
            "reported_candidate_route"
            if reported_in_source and not process_ready
            else "proof_eligible_portfolio_route"
        ),
        "listed": True,
        "is_primary": primary,
        "step_ids": step_ids,
        "solved": process_ready,
        "complete": complete,
        "configured_boundary_closed": complete,
        "closure_profile": closure_profile,
        "completion_label": closure_label(closure_profile),
        "route_state_label": route_state_label,
        "condition_status": condition_status,
        "condition_label": condition_label,
        "condition_complete": condition_complete,
        "reaction_validated": reaction_validated,
        "literature_grounded": literature_grounded,
        "procurement_closed": procurement_closed,
        "proof_vector": proof_vector,
        "inactive_fact_count": len(inactive_facts),
        "inactive_facts": inactive_facts,
        "acceptance_profiles": dict(route.get("acceptance_profiles") or {}),
        "achieved_profiles": list(route.get("achieved_profiles") or []),
        "full_synthesis_claim": process_ready,
        "executable": process_ready,
        "process_ready": process_ready,
        "advisory_only": not process_ready,
        "reported_in_source": reported_in_source,
        "reported_step_count": paper_reported_step_count,
        "planner_hypothesis_step_count": planner_hypothesis_step_count,
        "physical_step_count": int(route.get("physical_step_count") or len(step_ids)),
        "chemical_step_equivalent_count": int(
            route.get("chemical_step_equivalent_count") or len(step_ids)
        ),
        "net_step_savings": int(route.get("net_step_savings") or 0),
        "biocatalytic_superstep_count": int(
            route.get("biocatalytic_superstep_count") or 0
        ),
        "biocatalytic_step_count": int(route.get("biocatalytic_step_count") or 0),
        "mechanism_extrapolation_count": int(
            route.get("mechanism_extrapolation_count") or 0
        ),
        "unvalidated_biocatalytic_edge_ids": list(
            route.get("unvalidated_biocatalytic_edge_ids") or []
        ),
        "proof_level_counts": dict(route.get("proof_level_counts") or {}),
        "all_edges_proven": route.get("all_edges_proven") is True,
        "warning_codes": list(route.get("warning_codes") or []),
        "not_parent_route_proof": not complete,
        "proof_tier": PROOF_TIER[level],
        "confidence": "high" if level >= 3 else "medium" if level >= 2 else "low",
        "source_refs": source_refs,
        "multi_source": len(route.get("independent_source_groups") or []) >= 2,
        "synthesis_class": "canonical_portfolio",
        "trust_vector": {
            "min_trusted_source_group_count_across_steps": len(
                route.get("independent_source_groups") or []
            ),
            "corroborated_edge_count": len(step_ids) if level >= 3 else 0,
            "all_edges_corroborated": bool(step_ids) and level >= 3,
        },
    }


__all__ = ["route_branch"]
