"""Evidence, trust, and replacement projections for the V4 route adapter."""

from __future__ import annotations

from typing import Any, Mapping


PROOF_TIER = {
    0: "L0_advisory",
    1: "L1_structural_materialized",
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
            step_id_by_branch_edge.get((base_branch_id, str(row.get("base_edge_id") or "")), "")
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
                "reasons": sorted(set(str(reason) for reason in reasons if str(reason))),
            }
        )
    return {
        "schema_version": "route_replacement_validation.v1",
        "candidate_count": len(records),
        "validated_count": sum(row["validated"] is True for row in records),
        "records": records,
        "semantics": dict(raw.get("semantics") or {}),
    }


def trust_vector(edge: Mapping[str, Any], sources: list[Mapping[str, Any]]) -> dict[str, Any]:
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
                1.0 if proof_vector.get("condition_completeness") == "complete" else 0.75
            ),
            "source_recorded_unverified": 0.65,
            "model_predicted": 0.3,
        }.get(str(proof_vector.get("conditions") or "missing"), 0.0),
        "forward_feasibility": 1.0 if level >= 2 else 0.0,
        "trusted_source_group_count": len(source_groups),
        "corroborated": len(source_groups) >= 2,
    }


def condition_summary(status: str) -> str:
    return {
        "source_exact": "已绑定可重放的来源实验条件",
        "source_recorded_unverified": "来源记录了条件，但尚未升级为精确过程证明",
        "model_predicted": "仅有模型预测条件，不是文献事实",
        "missing": "未抽取可重放的反应条件；精确结构来源不等于实验条件",
    }.get(status, "反应条件状态未知")


def condition_resolution(
    status: str,
    inspector: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain a condition gap without implying that the user must fill it.

    Condition truth has several independent stages: a route hypothesis may
    exist before a source is found, a source may be bound before its procedure
    is located, and a procedure may be located before all fields are parsed.
    The workbench must expose that lifecycle instead of rendering every state
    as the same anonymous ``0 items`` group.
    """

    exact_records = [
        value
        for value in inspector.get("exact_records") or []
        if isinstance(value, Mapping)
    ]
    sources = [
        value
        for value in inspector.get("sources") or []
        if isinstance(value, Mapping)
    ]
    procedures = [
        value
        for value in inspector.get("procedure_records") or []
        if isinstance(value, Mapping)
    ]
    observations = [
        value
        for value in inspector.get("source_observation_records") or []
        if isinstance(value, Mapping)
    ]
    predictions = [
        value
        for value in inspector.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]
    prediction_attempts = [
        value
        for value in inspector.get("condition_prediction_attempts") or []
        if isinstance(value, Mapping)
    ]
    if status == "source_exact":
        stage = "replayable_source_conditions"
        label = "可重放来源条件"
        summary = "已绑定当前反应连接的原始实验过程和结构化条件。"
        next_action = "核对加料顺序、规模、后处理和纯化后进入实验评审。"
    elif status == "source_recorded_unverified":
        stage = "source_conditions_need_binding"
        label = "来源条件候选待核"
        summary = "来源中观察到条件，但尚未与当前反应连接闭合为精确过程证明。"
        next_action = "系统继续核对结构、反应连接、页码和原文片段。"
    elif status == "model_predicted":
        stage = "model_prediction_only"
        label = "仅有模型条件候选"
        summary = "已有实验设计候选，但它不是论文或专利中的实验事实。"
        next_action = "系统继续主动检索原始来源；候选只用于实验设计和排序。"
    elif prediction_attempts:
        stage = "condition_prediction_retry_pending"
        label = "条件预测待重试"
        failure_reasons = sorted(
            {
                str(reason)
                for attempt in prediction_attempts
                for reason in attempt.get("failure_reasons") or []
                if str(reason)
            }
        )
        summary = (
            "系统已主动尝试补全实验条件，但本轮没有返回可用候选；"
            "这不是要求用户手工填写条件。"
        )
        next_action = "系统继续重试条件模型，并并行检索、下载和解析原始论文或专利过程。"
    elif procedures or observations:
        stage = "source_procedure_parse_pending"
        label = "来源过程待解析"
        summary = "已定位来源过程，但尚未抽取出足够的可重放条件字段。"
        next_action = "系统继续解析正文、实施例、补充材料或 OCR 页面。"
    elif exact_records or sources:
        stage = "source_procedure_location_pending"
        label = "实验过程待定位"
        summary = "已找到结构或来源记录，但尚未定位到当前反应步骤的实验过程；精确结构来源不等于实验条件。"
        next_action = "系统继续下载原始论文/专利并定位 Experimental、Example 或制备段落。"
    else:
        stage = "exact_reaction_source_search_pending"
        label = "精确反应来源待发现"
        summary = "该步骤目前只是结构化反应假设，尚未发现与当前反应连接精确匹配的实验来源，因此没有可诚实展示的来源条件。"
        next_action = "系统继续主动检索并下载原始论文/专利；不要求用户手工补录，也不会编造条件。"
    return {
        "schema_version": "route_condition_resolution.v1",
        "stage": stage,
        "label": label,
        "summary": summary,
        "next_action": next_action,
        "source_record_count": len(exact_records) + len(sources),
        "procedure_record_count": len(procedures),
        "source_observation_count": len(observations),
        "prediction_count": len(predictions),
        "prediction_attempt_count": len(prediction_attempts),
        "prediction_failure_reasons": (
            failure_reasons if prediction_attempts and not predictions else []
        ),
        "acquisition_owner": "autoplanner",
        "user_input_required": False,
    }


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
    "condition_resolution",
    "condition_summary",
    "literature_counts",
    "replacement_validation_projection",
    "trust_vector",
]
