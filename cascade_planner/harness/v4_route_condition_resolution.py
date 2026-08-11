"""Explain route-condition evidence states without granting source authority."""

from __future__ import annotations

from typing import Any, Mapping


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


__all__ = ["condition_resolution", "condition_summary"]
