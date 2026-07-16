"""Evidence, trust, and replacement projections for the V4 route adapter."""
from __future__ import annotations

from typing import Any, Mapping


PROOF_TIER = {
    0: "L0_advisory",
    1: "L0_materialized",
    2: "L2_reaction_validated",
    3: "L3_precedent_supported",
    4: "L4_procurement_ready",
}


def closure_label(value: str) -> str:
    return {
        "exploration_closed": "搜索边界闭合",
        "procurement_closed": "采购边界闭合",
        "in_house_closed": "厂内库存闭合",
        "configured_boundary_closed": "配置边界闭合",
    }.get(value, "尚未闭合")


def replacement_validation_projection(
    source: Mapping[str, Any],
    *,
    step_id_by_branch_edge: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    raw = dict(source.get("replacement_validation") or {})
    records: list[dict[str, Any]] = []
    for value in raw.get("records") or []:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        base_branch_id = str(row.get("base_route_id") or "")
        replacement_branch_id = str(row.get("replacement_route_id") or "")
        base_step_id = str(
            step_id_by_branch_edge.get(
                (base_branch_id, str(row.get("base_edge_id") or "")), ""
            )
        )
        candidate_step_id = str(
            step_id_by_branch_edge.get(
                (replacement_branch_id, str(row.get("replacement_edge_id") or "")),
                "",
            )
        )
        validated = bool(
            row.get("accepted") is True
            and base_step_id
            and replacement_branch_id
            and candidate_step_id
        )
        reasons = list(row.get("reasons") or [])
        if row.get("accepted") is True and not validated:
            reasons.append("replacement_projection_binding_missing")
        records.append(
            {
                "replacement_id": str(row.get("replacement_id") or ""),
                "base_step_id": base_step_id,
                "base_branch_id": base_branch_id,
                "candidate_step_id": candidate_step_id if validated else "",
                "candidate_branch_id": replacement_branch_id if validated else "",
                "revalidated_route_branch_id": replacement_branch_id if validated else "",
                "replacement_hyperedge_id": str(row.get("replacement_edge_id") or ""),
                "accepted": validated,
                "validated": validated,
                "status": "route_revalidated" if validated else "rejected",
                "preview_only": True,
                "connectivity_revalidated": validated,
                "stock_closure_revalidated": validated,
                "reaction_proof_revalidated": validated,
                "reasons": sorted(
                    set(str(reason) for reason in reasons if str(reason))
                ),
            }
        )
    return {
        "schema_version": "route_replacement_validation.v1",
        "candidate_count": len(records),
        "validated_count": sum(row["validated"] is True for row in records),
        "records": records,
        "semantics": dict(raw.get("semantics") or {}),
    }


def trust_vector(
    edge: Mapping[str, Any], sources: list[Mapping[str, Any]]
) -> dict[str, Any]:
    level = int(edge.get("proof_level") or 0)
    proof_vector = dict(edge.get("proof_vector") or {})
    source_groups = {
        str(value.get("independence_group") or value.get("source_ref") or "")
        for value in sources
        if str(value.get("independence_group") or value.get("source_ref") or "")
    }
    return {
        "proof_tier": PROOF_TIER[max(0, min(4, level))],
        "identity": 1.0 if proof_vector.get("identity") == "source_exact" else 0.75,
        "connectivity": (
            1.0
            if proof_vector.get("reaction") == "source_reaction_exact"
            else 0.8
            if proof_vector.get("reaction") == "host_validated"
            else 0.5
            if level >= 1
            else 0.0
        ),
        "source_independence": min(1.0, len(source_groups) / 2),
        "stock": 1.0 if level >= 4 else 0.0,
        "conditions": {
            "source_exact": (
                1.0
                if proof_vector.get("condition_completeness") == "complete"
                else 0.75
            ),
            "source_recorded_unverified": 0.65,
            "model_predicted": 0.3,
        }.get(str(proof_vector.get("conditions") or "missing"), 0.0),
        "forward_feasibility": 1.0 if level >= 2 else 0.0,
        "trusted_source_group_count": len(source_groups),
        "corroborated": len(source_groups) >= 2,
    }


def conditions(records: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    for record in records:
        raw = record.get("conditions")
        if not isinstance(raw, Mapping) or not raw:
            continue
        return [
            {"label": str(key).replace("_", " "), "value": str(value)}
            for key, value in sorted(raw.items())
            if value not in (None, "", [], {})
        ]
    return []


def condition_summary(status: str) -> str:
    return {
        "source_exact": "已绑定可重放的来源实验条件",
        "source_recorded_unverified": "来源记录了条件，但尚未升级为精确过程证明",
        "model_predicted": "仅有模型预测条件，不是文献事实",
        "missing": "未抽取可重放的反应条件；精确结构来源不等于实验条件",
    }.get(status, "反应条件状态未知")


def literature_counts(
    routes: Mapping[str, Any], edge_inspectors: Mapping[str, Any]
) -> dict[str, int]:
    return {
        "independent_source_group_count": len(
            {
                str(group)
                for route in routes.values()
                for group in dict(route).get("independent_source_groups") or []
            }
        ),
        "document_count": len(
            {
                str(source.get("source_ref") or "")
                for inspector in edge_inspectors.values()
                for source in dict(inspector).get("sources") or []
                if isinstance(source, Mapping)
            }
        ),
        "representation_count": sum(
            len(dict(inspector).get("exact_records") or [])
            for inspector in edge_inspectors.values()
        ),
        "real_source_candidate_records": sum(
            len(dict(inspector).get("exact_records") or [])
            for inspector in edge_inspectors.values()
        ),
        "source_procedure_records": sum(
            len(dict(inspector).get("procedure_records") or [])
            for inspector in edge_inspectors.values()
        ),
        "source_observation_records": sum(
            len(dict(inspector).get("source_observation_records") or [])
            for inspector in edge_inspectors.values()
        ),
    }


__all__ = [
    "PROOF_TIER",
    "closure_label",
    "condition_summary",
    "conditions",
    "literature_counts",
    "replacement_validation_projection",
    "trust_vector",
]
