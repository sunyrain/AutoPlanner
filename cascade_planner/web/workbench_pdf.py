"""Project-level PDF reports for the retrosynthesis route workbench."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import html
import json
from typing import Any, Iterable, Mapping

from cascade_planner.harness.v4_route_evidence_projection import condition_resolution


class WorkbenchPdfError(RuntimeError):
    """Raised when the installed browser cannot render a route PDF."""


_CONDITION_LABELS = {
    "source_exact": "来源精确条件",
    "source_recorded_unverified": "来源候选 · 待核验",
    "model_predicted": "模型预测 · 非文献事实",
    "missing": "条件待取证",
}

_PROOF_LABELS = {
    "L0_materialized": "L0 已物化",
    "L1_structural_materialized": "L1 结构已物化",
    "L2_reaction_validated": "L2 反应已重验",
    "L3_precedent_supported": "L3 文献先例支持",
    "L4_procurement_ready": "L4 采购闭合",
}


def compile_workbench_report_html(snapshot: Mapping[str, Any]) -> str:
    """Compile a complete, self-contained project report from one snapshot."""

    value = dict(snapshot or {})
    target = dict(value.get("target") or {})
    target_name = str(target.get("name") or "未命名逆合成目标")
    target_smiles = str(target.get("canonical_smiles") or "")
    run_id = str(value.get("run_id") or "unknown-run")
    routes = _report_routes(value)
    canonical_count = sum(row["kind"] == "portfolio" for row in routes)
    planned_count = len(routes) - canonical_count
    portfolio = dict(value.get("portfolio") or {})
    closeout = dict(portfolio.get("closeout") or {})
    unique_steps = _unique_steps(routes)
    condition_counts = Counter(
        str(step.get("condition_status") or "missing") for step in unique_steps
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary = _project_summary(
        target_name=target_name,
        canonical_count=canonical_count,
        planned_count=planned_count,
        portfolio=portfolio,
        closeout=closeout,
        condition_counts=condition_counts,
    )
    route_inventory = "".join(
        _route_inventory_row(index, route) for index, route in enumerate(routes, 1)
    ) or '<tr><td colspan="7">当前快照尚无可导出的路线。</td></tr>'
    route_sections = "".join(
        _route_section(index, route) for index, route in enumerate(routes, 1)
    )

    accepted = portfolio.get("accepted") is True
    achieved = str(portfolio.get("achieved_profile") or "unresolved")
    closeout_reasons = "、".join(
        _closeout_reason_text(str(row)) for row in closeout.get("reasons") or []
    )
    target_svg = _molecule_svg(target_smiles, 640, 360)
    status_class = "ok" if accepted else "warn"
    status_label = "配置验收通过" if accepted else "未收敛"
    css = _report_css()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_esc(target_name)} · 逆合成项目报告</title>
  <style>{css}</style>
</head>
<body>
  <section class="cover report-page">
    <div class="eyebrow">AUTOPLANNER · RETROSYNTHESIS DOSSIER</div>
    <div class="cover-grid">
      <div class="target-hero">
        <div class="target-structure">{target_svg}</div>
        <div class="target-smiles">{_esc(target_smiles or "目标结构未记录")}</div>
      </div>
      <div class="cover-copy">
        <div class="status-pill {status_class}">{status_label}</div>
        <h1>{_esc(target_name)}</h1>
        <p class="lead">逆合成路线全景与逐步证据报告</p>
        <p>{_esc(summary)}</p>
        <dl class="cover-meta">
          <div><dt>Run ID</dt><dd>{_esc(run_id)}</dd></div>
          <div><dt>快照修订</dt><dd>{_esc(_revision_label(value.get("revision")))}</dd></div>
          <div><dt>实际达到档位</dt><dd>{_esc(achieved)}</dd></div>
          <div><dt>生成时间</dt><dd>{generated_at}</dd></div>
        </dl>
      </div>
    </div>
    <div class="cover-metrics">
      {_metric("证明组合路线", canonical_count)}
      {_metric("规划候选路线", planned_count)}
      {_metric("完整路线", closeout.get("complete_route_count") or 0)}
      {_metric("条件完整路线", _profile_count(portfolio, "condition_complete"))}
      {_metric("工艺就绪路线", _profile_count(portfolio, "process_ready"))}
      {_metric("待闭合缺口", closeout.get("deficit_count") or 0)}
    </div>
    <div class="scientific-note">
      <strong>科学状态说明</strong>
      <span>{_esc(closeout_reasons or "当前没有记录 closeout reason。")}</span>
    </div>
  </section>

  <section class="report-page project-overview">
    <header class="section-header">
      <div><span class="section-no">00</span><h2>项目概述与路线目录</h2></div>
      <p>{_esc(summary)}</p>
    </header>
    <div class="evidence-band">
      {_evidence_stat("来源精确", condition_counts["source_exact"], "exact")}
      {_evidence_stat("来源候选", condition_counts["source_recorded_unverified"], "candidate")}
      {_evidence_stat("模型预测", condition_counts["model_predicted"], "predicted")}
      {_evidence_stat("条件待取证", condition_counts["missing"], "missing")}
    </div>
    <table class="route-table">
      <thead><tr><th>#</th><th>路线类型</th><th>策略概述</th><th>步骤</th><th>证明</th><th>条件</th><th>缺口</th></tr></thead>
      <tbody>{route_inventory}</tbody>
    </table>
    <div class="legend-box">
      <strong>阅读口径</strong>
      <span>来源精确条件可追溯到结构绑定的实验记录；来源候选仍需核验；模型预测只用于实验设计，不作为文献事实；条件待取证表示尚无可重放记录。</span>
    </div>
  </section>
  {route_sections}
</body>
</html>"""


def render_workbench_pdf(
    snapshot: Mapping[str, Any],
    *,
    timeout_ms: int = 30_000,
) -> bytes:
    """Render a complete project dossier as a designed A3 landscape PDF."""

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - optional deployment boundary
        raise WorkbenchPdfError("workbench_pdf_playwright_unavailable") from exc

    body = compile_workbench_report_html(snapshot)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    viewport={"width": 1800, "height": 1200},
                    device_scale_factor=1,
                )
                page.set_content(body, wait_until="load", timeout=timeout_ms)
                page.evaluate("() => document.fonts ? document.fonts.ready : true")
                page.emulate_media(media="print")
                return bytes(
                    page.pdf(
                        landscape=True,
                        print_background=True,
                        prefer_css_page_size=True,
                        display_header_footer=True,
                        header_template="<span></span>",
                        footer_template=(
                            '<div style="width:100%;font-size:8px;color:#64748b;'
                            'padding:0 14mm;display:flex;justify-content:space-between;'
                            'font-family:Arial,sans-serif">'
                            "<span>AutoPlanner retrosynthesis dossier</span>"
                            '<span><span class="pageNumber"></span> / '
                            '<span class="totalPages"></span></span></div>'
                        ),
                    )
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise WorkbenchPdfError(
            f"workbench_pdf_render_failed:{type(exc).__name__}"
        ) from exc


def _report_routes(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    molecules = _mapping(snapshot.get("molecules"))
    edges = _mapping(snapshot.get("edges"))
    inspectors = _mapping(_mapping(snapshot.get("inspectors")).get("edges"))
    routes: list[dict[str, Any]] = []
    for route_id, raw_route in sorted(_mapping(snapshot.get("routes")).items()):
        route = dict(raw_route or {})
        steps = [
            _canonical_step(edge_id, edges, inspectors, molecules)
            for edge_id in route.get("edge_ids") or []
            if str(edge_id) in edges
        ]
        routes.append(
            {
                "route_id": str(route_id),
                "kind": "portfolio",
                "strategy": str(route.get("strategy") or "未记录路线策略"),
                "route": route,
                "steps": _topological_steps(steps),
            }
        )
    for route_id, raw_route in sorted(_mapping(snapshot.get("planned_routes")).items()):
        route = dict(raw_route or {})
        steps = [
            _planned_step(raw, edges, inspectors, molecules)
            for raw in route.get("steps") or []
            if isinstance(raw, Mapping)
        ]
        routes.append(
            {
                "route_id": str(route_id),
                "kind": "planned",
                "strategy": str(route.get("strategy") or "未记录规划策略"),
                "route": route,
                "steps": _topological_steps(steps),
            }
        )
    return routes


def _canonical_step(
    edge_id: Any,
    edges: Mapping[str, Any],
    inspectors: Mapping[str, Any],
    molecules: Mapping[str, Any],
) -> dict[str, Any]:
    identity = str(edge_id)
    edge = dict(edges.get(identity) or {})
    precursor_ids = [str(row) for row in edge.get("precursor_molecule_ids") or []]
    product_id = str(edge.get("product_molecule_id") or "")
    return {
        "step_id": identity,
        "edge": edge,
        "inspector": dict(inspectors.get(identity) or {}),
        "precursor_refs": precursor_ids,
        "product_ref": product_id,
        "precursors": [_molecule_value(molecules, row) for row in precursor_ids],
        "product": _molecule_value(molecules, product_id),
        "condition_status": str(edge.get("condition_status") or "missing"),
    }


def _planned_step(
    raw: Mapping[str, Any],
    edges: Mapping[str, Any],
    inspectors: Mapping[str, Any],
    molecules: Mapping[str, Any],
) -> dict[str, Any]:
    step = dict(raw)
    edge_id = str(step.get("edge_id") or step.get("step_id") or "")
    if edge_id in edges:
        value = _canonical_step(edge_id, edges, inspectors, molecules)
        value["planning_hypothesis"] = str(
            step.get("transformation_hypothesis") or ""
        )
        return value
    precursor_smiles = [str(row) for row in step.get("precursor_smiles") or []]
    product_smiles = str(step.get("product_smiles") or "")
    return {
        "step_id": edge_id or str(step.get("step_id") or "planned-step"),
        "edge": {"proof_name": "planner_hypothesis", "accepted": False},
        "inspector": {},
        "precursor_refs": precursor_smiles,
        "product_ref": product_smiles,
        "precursors": [
            {"canonical_smiles": smiles, "label": ""} for smiles in precursor_smiles
        ],
        "product": {"canonical_smiles": product_smiles, "label": ""},
        "condition_status": "missing",
        "planning_hypothesis": str(step.get("transformation_hypothesis") or ""),
    }


def _topological_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order route reactions from starting materials toward the target."""

    if len(steps) < 2:
        return steps
    dependencies: list[set[int]] = [set() for _ in steps]
    for current_index, current in enumerate(steps):
        precursors = set(current.get("precursor_refs") or [])
        for producer_index, producer in enumerate(steps):
            if producer_index != current_index and producer.get("product_ref") in precursors:
                dependencies[current_index].add(producer_index)
    ordered: list[dict[str, Any]] = []
    remaining = set(range(len(steps)))
    while remaining:
        ready = sorted(index for index in remaining if not dependencies[index] & remaining)
        if not ready:
            return steps
        for index in ready:
            ordered.append(steps[index])
            remaining.remove(index)
    return ordered


def _route_inventory_row(index: int, route: Mapping[str, Any]) -> str:
    raw = dict(route.get("route") or {})
    steps = list(route.get("steps") or [])
    statuses = Counter(str(row.get("condition_status") or "missing") for row in steps)
    condition_text = "；".join(
        f"{_CONDITION_LABELS.get(key, key)} {count}"
        for key, count in statuses.items()
    ) or "无步骤"
    kind = "证明组合路线" if route.get("kind") == "portfolio" else "规划候选"
    proof = _PROOF_LABELS.get(
        str(raw.get("proof_name") or ""), str(raw.get("proof_name") or "待验证")
    )
    return (
        "<tr>"
        f"<td>{index:02d}</td><td>{kind}</td>"
        f"<td>{_esc(route.get('strategy'))}</td><td>{len(steps)}</td>"
        f"<td>{_esc(proof)}</td><td>{_esc(condition_text)}</td>"
        f"<td>{_esc(raw.get('deficit_count') or 0)}</td>"
        "</tr>"
    )


def _route_section(index: int, route: Mapping[str, Any]) -> str:
    raw = dict(route.get("route") or {})
    steps = list(route.get("steps") or [])
    kind_label = "证明组合路线" if route.get("kind") == "portfolio" else "规划候选路线"
    route_id = str(route.get("route_id") or "")
    snake = _snake_diagram(steps)
    details = "".join(
        _step_detail(position, step) for position, step in enumerate(steps, 1)
    ) or '<div class="empty-report">这条路线尚未物化任何反应步骤。</div>'
    achieved = " · ".join(str(row) for row in raw.get("achieved_profiles") or [])
    route_facts = (
        _fact("路线类别", kind_label)
        + _fact("物理步骤", raw.get("physical_step_count") or len(steps))
        + _fact("证明级别", raw.get("proof_name") or "planner_hypothesis")
        + _fact("已达档位", achieved or "unresolved")
        + _fact("风险分数", raw.get("risk_score") or 0)
        + _fact("缺口数量", raw.get("deficit_count") or 0)
    )
    return f"""
  <section class="route-overview report-page">
    <header class="section-header route-heading">
      <div><span class="section-no">{index:02d}</span><h2>{kind_label}</h2></div>
      <p class="route-id">{_esc(route_id)}</p>
    </header>
    <h3 class="strategy">{_esc(route.get("strategy"))}</h3>
    <div class="route-facts">{route_facts}</div>
    <div class="direction-label">合成方向：起始物 → 目标产物</div>
    <div class="snake-diagram">{snake}</div>
  </section>
  <section class="route-details">
    <header class="detail-route-header">
      <span>{index:02d}</span>
      <div><h2>路线逐步详单</h2><p>{_esc(route.get("strategy"))}</p></div>
    </header>
    {details}
  </section>"""


def _snake_diagram(steps: list[dict[str, Any]], columns: int = 3) -> str:
    if not steps:
        return '<div class="empty-report">尚无路线步骤。</div>'
    rows: list[str] = []
    indexed = list(enumerate(steps, 1))
    for row_index in range(0, len(indexed), columns):
        chunk = indexed[row_index : row_index + columns]
        reverse = (row_index // columns) % 2 == 1
        visual = list(reversed(chunk)) if reverse else chunk
        arrow = "←" if reverse else "→"
        cards: list[str] = []
        if reverse and len(visual) < columns:
            cards.extend(
                '<span class="snake-spacer"></span>'
                for _ in range((columns - len(visual)) * 2)
            )
        for item_index, (step_index, step) in enumerate(visual):
            if item_index:
                cards.append(f'<span class="snake-arrow">{arrow}</span>')
            cards.append(_snake_card(step_index, step))
        rows.append(
            f'<div class="snake-row{" reverse" if reverse else ""}">'
            + "".join(cards)
            + "</div>"
        )
        if row_index + columns < len(indexed):
            turn = "left" if reverse else "right"
            rows.append(f'<div class="snake-turn {turn}">↓</div>')
    return "".join(rows)


def _snake_card(index: int, step: Mapping[str, Any]) -> str:
    product = dict(step.get("product") or {})
    precursor_smiles = [
        str(dict(row or {}).get("canonical_smiles") or "")
        for row in step.get("precursors") or []
    ]
    product_smiles = str(product.get("canonical_smiles") or "")
    inspector = dict(step.get("inspector") or {})
    hypothesis = _step_hypothesis(step, inspector)
    condition = str(step.get("condition_status") or "missing")
    proof = str(dict(step.get("edge") or {}).get("proof_name") or "待验证")
    precursors = '<span class="mini-plus">+</span>'.join(
        f'<span>{_molecule_svg(smiles, 150, 90)}</span>'
        for smiles in precursor_smiles
    ) or '<span class="structure-fallback">?</span>'
    return f"""<article class="snake-card">
      <div class="snake-card-top"><b>Step {index}</b>{_condition_badge(condition)}</div>
      <div class="mini-reaction"><div class="mini-side">{precursors}</div><i>→</i><div class="mini-side product">{_molecule_svg(product_smiles, 170, 96)}</div></div>
      <div class="snake-hypothesis">{_esc(hypothesis or "转化假设待补充")}</div>
      <div class="snake-proof">{_esc(_PROOF_LABELS.get(proof, proof))}</div>
    </article>"""


def _step_detail(index: int, step: Mapping[str, Any]) -> str:
    edge = dict(step.get("edge") or {})
    inspector = dict(step.get("inspector") or {})
    condition_status = str(step.get("condition_status") or "missing")
    hypothesis = _step_hypothesis(step, inspector)
    precursors = list(step.get("precursors") or [])
    product = dict(step.get("product") or {})
    precursor_html = '<span class="scheme-plus">+</span>'.join(
        _scheme_molecule(value) for value in precursors
    ) or '<div class="scheme-molecule missing-structure">起始物未记录</div>'
    proof = dict(inspector.get("proof") or {})
    conditions = _conditions_html(inspector, condition_status)
    validation = _validation_html(inspector)
    provenance = _provenance_html(inspector, step)
    source = _source_html(inspector)
    proof_name = str(edge.get("proof_name") or proof.get("achieved_level_name") or "待验证")
    reasons = list(proof.get("reasons") or inspector.get("rejection_reasons") or [])
    first_class = " first-step" if index == 1 else ""
    return f"""<article class="step-detail{first_class}">
      <header class="step-title">
        <div class="step-number">{index:02d}</div>
        <div><span>REACTION STEP</span><h3>{_esc(hypothesis or step.get("step_id"))}</h3></div>
        {_condition_badge(condition_status)}
      </header>
      <div class="reaction-scheme">
        <div class="scheme-side precursors">{precursor_html}</div>
        <div class="scheme-arrow"><span>⟶</span><small>{_esc(_CONDITION_LABELS.get(condition_status, condition_status))}</small></div>
        <div class="scheme-side product">{_scheme_molecule(product)}</div>
      </div>
      <div class="step-meta">
        {_fact("Edge ID", step.get("step_id"))}
        {_fact("证明级别", _PROOF_LABELS.get(proof_name, proof_name))}
        {_fact("反应验收", "通过" if edge.get("accepted") is True else "未通过")}
        {_fact("来源状态", dict(edge.get("proof_vector") or {}).get("sources") or "none")}
        {_fact("工艺状态", dict(edge.get("proof_vector") or {}).get("process") or "blocked")}
        {_fact("主要缺口", "、".join(str(row) for row in reasons) or "无已记录缺口")}
      </div>
      <div class="detail-columns">
        <div class="detail-row">
          <section class="detail-panel conditions-panel"><h4>反应条件与取证状态</h4>{conditions}</section>
          <section class="detail-panel validation-panel"><h4>反应验证</h4>{validation}</section>
        </div>
        <div class="detail-row">
          <section class="detail-panel"><h4>来源与实验记录</h4>{source}</section>
          <section class="detail-panel"><h4>提出过程与转化依据</h4>{provenance}</section>
        </div>
      </div>
    </article>"""


def _conditions_html(inspector: Mapping[str, Any], status: str) -> str:
    exact = list(inspector.get("exact_records") or [])
    procedures = list(inspector.get("procedure_records") or [])
    observations = list(inspector.get("source_observation_records") or [])
    predictions = list(inspector.get("condition_predictions") or [])
    resolution = condition_resolution(status, inspector)
    parts = [
        f'<div class="condition-callout {_esc(status)}"><strong>{_esc(_CONDITION_LABELS.get(status, status))}</strong>'
        f'<span>{_esc(resolution["summary"])}</span>'
        f'<small>下一步：{_esc(resolution["next_action"])}</small></div>'
    ]
    if exact:
        parts.append(_record_group("精确来源条件", exact, "exact"))
    if procedures:
        parts.append(_record_group("可重放实验过程", procedures, "procedure"))
    if observations:
        parts.append(_record_group("来源观察", observations, "observation"))
    if predictions:
        parts.append(_record_group("模型条件候选（非文献事实）", predictions, "prediction"))
    missing = [str(row) for row in inspector.get("condition_missing_required_groups") or []]
    if missing:
        parts.append(
            '<div class="missing-groups"><b>缺失字段</b><span>'
            + _esc("、".join(missing))
            + "</span></div>"
        )
    if len(parts) == 1:
        parts.append(
            '<p class="empty-copy">当前没有可诚实展示的来源条件；自动取证仍由系统负责。</p>'
        )
    return "".join(parts)


def _validation_html(inspector: Mapping[str, Any]) -> str:
    proofs = [dict(row) for row in inspector.get("reaction_proofs") or []]
    if not proofs:
        reasons = inspector.get("rejection_reasons") or []
        return (
            '<p class="empty-copy">尚无反应重验记录。</p>'
            + _tag_list(reasons, "reason")
        )
    rows: list[str] = []
    for index, proof in enumerate(proofs, 1):
        checks = dict(proof.get("checks") or {})
        failed = [str(key) for key, accepted in checks.items() if accepted is False]
        mapped = str(proof.get("mapped_reaction") or "")
        rows.append(
            f'<div class="validation-record"><div><b>验证记录 {index}</b>'
            f'<span>{"通过" if proof.get("accepted") is True else "未通过"} · '
            f'{_esc(proof.get("validator_version") or "validator unknown")}</span></div>'
            f'{_tag_list(proof.get("reasons") or [], "reason")}'
            f'{_tag_list(failed, "failed-check")}'
            f'<code>{_esc(mapped or "映射反应未记录")}</code></div>'
        )
    return "".join(rows)


def _source_html(inspector: Mapping[str, Any]) -> str:
    records: list[tuple[str, Iterable[Any]]] = [
        ("精确记录", inspector.get("exact_records") or []),
        ("来源", inspector.get("sources") or []),
        ("过程记录", inspector.get("procedure_records") or []),
        ("来源观察", inspector.get("source_observation_records") or []),
    ]
    rendered = [
        _record_group(label, list(values), "source")
        for label, values in records
        if list(values)
    ]
    return "".join(rendered) or '<p class="empty-copy">尚无精确来源或可重放过程绑定。</p>'


def _provenance_html(inspector: Mapping[str, Any], step: Mapping[str, Any]) -> str:
    records = [dict(row) for row in inspector.get("provenance") or []]
    planning = str(step.get("planning_hypothesis") or "")
    if planning:
        records.insert(
            0,
            {
                "origin_kind": "planner_hypothesis",
                "transformation_hypothesis": planning,
            },
        )
    if not records:
        return '<p class="empty-copy">提出过程尚未记录。</p>'
    rows: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        signature = (
            str(record.get("origin_kind") or ""),
            str(record.get("origin_ref") or ""),
            str(record.get("transformation_hypothesis") or ""),
        )
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(
            '<div class="provenance-record">'
            f'<b>{_esc(signature[0] or "unknown origin")}</b>'
            f'<span>{_esc(signature[2] or "未记录转化说明")}</span>'
            f'<small>{_esc(signature[1])}</small></div>'
        )
    return "".join(rows)


def _record_group(label: str, records: list[Any], class_name: str) -> str:
    rendered: list[str] = []
    for index, record in enumerate(records, 1):
        if isinstance(record, Mapping):
            fields = _record_fields(record)
        else:
            fields = [("记录", _value_text(record))]
        body = "".join(
            f'<div><dt title="{_esc(key)}">{_record_key_html(key)}</dt>'
            f'<dd>{_esc(value)}</dd></div>'
            for key, value in fields
            if value
        )
        rendered.append(
            f'<div class="record-card {class_name}"><b>{_esc(label)} {index}</b>'
            f'<dl>{body or "<div><dd>记录存在，但没有可展示字段。</dd></div>"}</dl></div>'
        )
    return "".join(rendered)


def _record_fields(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    ignored = {
        "content_sha256",
        "semantics",
        "not_reaction_proof",
        "authority_scope",
        "null1",
        "null2",
    }
    priority = (
        "title",
        "doi",
        "patent_id",
        "url",
        "citation",
        "procedure",
        "conditions",
        "Catalyst",
        "Reagent",
        "Solvent",
        "Temperature",
        "Time",
        "Pressure",
        "Yield",
        "Score",
    )
    ordered_keys = list(priority) + sorted(
        str(key) for key in record if str(key) not in priority
    )
    rows: list[tuple[str, str]] = []
    for key in ordered_keys:
        if key in ignored or key not in record:
            continue
        text = _value_text(record.get(key))
        if text:
            rows.append((key, text))
    return rows


def _value_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, Mapping):
        return "；".join(
            f"{key}: {_value_text(item)}"
            for key, item in value.items()
            if _value_text(item)
        )
    if isinstance(value, (list, tuple, set)):
        return "；".join(_value_text(item) for item in value if _value_text(item))
    return str(value)


def _record_key_html(value: Any) -> str:
    """Escape an audit key and add safe wrap opportunities at separators."""

    return _esc(value).replace("_", "_<wbr>").replace("/", "/<wbr>")


def _step_hypothesis(step: Mapping[str, Any], inspector: Mapping[str, Any]) -> str:
    planning = str(step.get("planning_hypothesis") or "")
    if planning:
        return planning
    for row in inspector.get("provenance") or []:
        if isinstance(row, Mapping) and row.get("transformation_hypothesis"):
            return str(row["transformation_hypothesis"])
    return ""


def _scheme_molecule(value: Mapping[str, Any]) -> str:
    molecule = dict(value or {})
    smiles = str(molecule.get("canonical_smiles") or "")
    label = str(molecule.get("label") or "")
    stock = "可采购/库存闭合" if molecule.get("stock_closed") is True else "库存待核验"
    return f"""<div class="scheme-molecule">
      <div class="scheme-structure">{_molecule_svg(smiles, 330, 190)}</div>
      <b>{_esc(label or smiles or "结构未记录")}</b>
      <small>{_esc(smiles)}</small><span>{stock}</span>
    </div>"""


def _molecule_value(molecules: Mapping[str, Any], molecule_id: str) -> dict[str, Any]:
    value = dict(molecules.get(molecule_id) or {})
    value.setdefault("molecule_id", molecule_id)
    return value


@lru_cache(maxsize=2_048)
def _molecule_svg(smiles: str, width: int, height: int) -> str:
    if not smiles:
        return '<div class="structure-fallback">结构未记录</div>'
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid_smiles")
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg
    except Exception:  # pragma: no cover - depiction fallback boundary
        return f'<div class="structure-fallback">{_esc(smiles)}</div>'


def _unique_steps(routes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for route in routes:
        for raw in route.get("steps") or []:
            step = dict(raw)
            identity = str(step.get("step_id") or json.dumps(step, sort_keys=True))
            if identity not in seen:
                seen.add(identity)
                rows.append(step)
    return rows


def _project_summary(
    *,
    target_name: str,
    canonical_count: int,
    planned_count: int,
    portfolio: Mapping[str, Any],
    closeout: Mapping[str, Any],
    condition_counts: Counter[str],
) -> str:
    accepted = portfolio.get("accepted") is True
    status = "已达到配置验收条件" if accepted else "尚未达到配置验收条件"
    return (
        f"本报告汇总 {target_name} 的 {canonical_count} 条证明组合路线和 "
        f"{planned_count} 条规划候选。当前{status}，完整路线 "
        f"{int(closeout.get('complete_route_count') or 0)} 条，仍有 "
        f"{int(closeout.get('deficit_count') or 0)} 个闭合缺口。唯一反应步骤中，"
        f"{condition_counts['source_exact']} 步具有精确来源条件，"
        f"{condition_counts['source_recorded_unverified']} 步为来源候选，"
        f"{condition_counts['model_predicted']} 步只有模型预测，"
        f"{condition_counts['missing']} 步仍待条件取证。"
    )


def _condition_explanation(status: str) -> str:
    return {
        "source_exact": "条件已与精确结构和来源实验过程绑定。",
        "source_recorded_unverified": "已发现来源条件，但结构或过程绑定尚未完成核验。",
        "model_predicted": "以下条件来自模型预测，可用于实验设计，但不能作为文献事实。",
        "missing": "尚无可重放的来源实验条件，需要继续检索、下载、视觉抽取和结构绑定。",
    }.get(status, "条件状态尚未解释。")


def _revision_label(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = [
            f"{key} {item}"
            for key, item in value.items()
            if key in {"graph", "evidence", "scientific_sha256"} and item not in {None, ""}
        ]
        return " · ".join(parts) or "revision unavailable"
    return f"revision {value or 0}"


def _closeout_reason_text(value: str) -> str:
    return {
        "distinct_complete_edge_sets_not_met": "独立完整路线集合数量不足",
        "minimum_complete_route_count_not_met": "完整路线数量未达到要求",
        "configured_scientific_acceptance_not_met": "未达到配置科学验收条件",
    }.get(value, value)


def _condition_badge(status: str) -> str:
    return (
        f'<span class="condition-badge {_esc(status)}">'
        f'{_esc(_CONDITION_LABELS.get(status, status))}</span>'
    )


def _profile_count(portfolio: Mapping[str, Any], key: str) -> int:
    return int(dict(portfolio.get("acceptance_profile_counts") or {}).get(key) or 0)


def _metric(label: str, value: Any) -> str:
    return f'<div class="cover-metric"><b>{_esc(value)}</b><span>{_esc(label)}</span></div>'


def _evidence_stat(label: str, value: Any, class_name: str) -> str:
    return (
        f'<div class="evidence-stat {class_name}"><b>{_esc(value)}</b>'
        f'<span>{_esc(label)}</span></div>'
    )


def _fact(label: str, value: Any) -> str:
    return f'<div class="fact"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'


def _tag_list(values: Iterable[Any], class_name: str) -> str:
    tags = "".join(
        f'<span class="detail-tag {class_name}">{_esc(value)}</span>'
        for value in values
        if str(value)
    )
    return f'<div class="tag-list">{tags}</div>' if tags else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _report_css() -> str:
    return r"""
@page { size: A3 landscape; margin: 12mm 14mm 17mm; }
* { box-sizing: border-box; }
html { color: #172033; font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; font-size: 10pt; }
body { margin: 0; background: #fff; }
.report-page { break-after: page; min-height: 250mm; }
.cover { display: flex; flex-direction: column; justify-content: space-between; padding: 4mm 2mm 2mm; }
.eyebrow { color: #4f46e5; font-size: 10pt; font-weight: 800; letter-spacing: .18em; }
.cover-grid { display: grid; grid-template-columns: 1.05fr .95fr; gap: 18mm; align-items: center; }
.target-hero { border: 1px solid #dfe5f1; border-radius: 8mm; padding: 8mm; background: linear-gradient(145deg,#f8faff,#eef2ff); }
.target-structure { height: 105mm; display: flex; align-items: center; justify-content: center; }
.target-structure svg { width: 100%; height: 100%; }
.target-smiles { padding: 3mm 4mm; border-radius: 3mm; background: #fff; color: #334155; font-family: Consolas, monospace; text-align: center; overflow-wrap: anywhere; }
.cover-copy h1 { margin: 5mm 0 2mm; font-size: 31pt; line-height: 1.12; color: #111827; }
.lead { margin: 0 0 5mm; color: #4f46e5; font-size: 17pt; font-weight: 700; }
.cover-copy > p:not(.lead) { color: #475569; font-size: 11pt; line-height: 1.8; }
.status-pill { display: inline-block; padding: 2mm 4mm; border-radius: 999px; font-weight: 800; }
.status-pill.ok { color: #047857; background: #d1fae5; }
.status-pill.warn { color: #b45309; background: #fef3c7; }
.cover-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 3mm; margin-top: 7mm; }
.cover-meta div { padding: 3mm; border-left: 1.2mm solid #c7d2fe; background: #f8fafc; }
.cover-meta dt { color: #64748b; font-size: 8pt; }
.cover-meta dd { margin: 1mm 0 0; font-weight: 700; overflow-wrap: anywhere; }
.cover-metrics { display: grid; grid-template-columns: repeat(6,1fr); gap: 3mm; }
.cover-metric { padding: 4mm; border: 1px solid #e2e8f0; border-radius: 4mm; background: #fff; }
.cover-metric b { display: block; font-size: 20pt; color: #312e81; }
.cover-metric span { color: #64748b; }
.scientific-note { display: flex; gap: 5mm; padding: 4mm; border-radius: 4mm; background: #fff7ed; color: #9a3412; }
.section-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 12mm; border-bottom: 1px solid #cbd5e1; padding: 3mm 0 4mm; }
.section-header > div { display: flex; align-items: center; gap: 4mm; }
.section-no { display: grid; place-items: center; width: 14mm; height: 14mm; border-radius: 4mm; background: #312e81; color: #fff; font-size: 15pt; font-weight: 800; }
.section-header h2 { margin: 0; font-size: 22pt; }
.section-header p { max-width: 52%; margin: 0; color: #64748b; line-height: 1.65; }
.evidence-band { display: grid; grid-template-columns: repeat(4,1fr); gap: 4mm; margin: 8mm 0; }
.evidence-stat { padding: 4mm; border-radius: 4mm; border: 1px solid #dbe4f0; }
.evidence-stat b { display: block; font-size: 18pt; }
.evidence-stat.exact { background:#ecfdf5;color:#047857; }.evidence-stat.candidate{background:#eff6ff;color:#0369a1;}.evidence-stat.predicted{background:#f5f3ff;color:#6d28d9;}.evidence-stat.missing{background:#fff7ed;color:#c2410c;}
.route-table { width: 100%; border-collapse: collapse; font-size: 9pt; }
.route-table th { padding: 3mm; color: #475569; background: #f1f5f9; text-align: left; }
.route-table td { padding: 3mm; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
.route-table td:nth-child(3) { width: 35%; }
.legend-box { display: flex; gap: 5mm; margin-top: 7mm; padding: 4mm; border-left: 1.5mm solid #4f46e5; background: #eef2ff; line-height: 1.7; }
.route-heading .route-id { max-width: 48%; font-family: Consolas,monospace; font-size: 8pt; overflow-wrap:anywhere; }
.route-overview { break-before:page; }
.strategy { margin: 6mm 0 4mm; font-size: 15pt; line-height: 1.45; color:#1e293b; }
.route-facts,.step-meta { display: grid; grid-template-columns: repeat(6,1fr); gap: 2.5mm; }
.fact { min-width:0; padding: 2.5mm 3mm; border-radius: 3mm; background:#f8fafc; border:1px solid #e2e8f0; }
.fact span { display:block; color:#64748b; font-size:7.5pt; }
.fact b { display:block; margin-top:1mm; font-size:9pt; overflow-wrap:anywhere; }
.direction-label { margin: 5mm 0 3mm; color:#4f46e5; font-weight:800; letter-spacing:.05em; }
.snake-diagram { padding: 4mm; border-radius: 5mm; background: #f8fafc; border: 1px solid #dbe4f0; }
.snake-row { display:grid; grid-template-columns: 1fr 10mm 1fr 10mm 1fr; align-items:center; min-height:47mm; }
.snake-card { height:45mm; padding:2.5mm; border-radius:4mm; background:#fff; border:1px solid #cbd5e1; box-shadow:0 1mm 3mm rgba(15,23,42,.06); display:grid; grid-template-columns:1fr 1fr; grid-template-rows:auto 23mm 1fr; gap:1mm 3mm; }
.snake-card-top { grid-column:1/-1; display:flex; align-items:center; justify-content:space-between; gap:2mm; }
.mini-reaction { grid-column:1/-1; display:grid; grid-template-columns:1fr 7mm 1fr; align-items:center; min-width:0; }
.mini-reaction > i { color:#4f46e5; font-size:14pt; font-style:normal; text-align:center; }
.mini-side { height:22mm; display:flex; align-items:center; justify-content:center; min-width:0; }
.mini-side > span { flex:1; height:21mm; min-width:0; display:flex; align-items:center; justify-content:center; }
.mini-side svg { width:100%; height:100%; }.mini-plus{flex:0 0 auto!important;color:#64748b;font-size:10pt;}
.snake-hypothesis { align-self:center; font-size:7.3pt; line-height:1.25; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.snake-proof { align-self:center; color:#475569; font-size:7.2pt; text-align:right; }
.snake-arrow { text-align:center; color:#4f46e5; font-size:20pt; font-weight:900; }
.snake-spacer { min-width:0; }
.snake-turn { height:8mm; color:#4f46e5; font-size:18pt; font-weight:900; line-height:8mm; }
.snake-turn.right { text-align:right; padding-right:15%; }.snake-turn.left { text-align:left; padding-left:15%; }
.condition-badge { display:inline-block; max-width:58mm; padding:1.2mm 2.2mm; border-radius:999px; font-size:7.5pt; font-weight:800; white-space:nowrap; }
.condition-badge.source_exact { background:#d1fae5;color:#047857; }.condition-badge.source_recorded_unverified{background:#dbeafe;color:#0369a1;}.condition-badge.model_predicted{background:#ede9fe;color:#6d28d9;}.condition-badge.missing{background:#ffedd5;color:#c2410c;}
.route-details { break-before: page; }
.detail-route-header { display:flex; gap:5mm; align-items:center; margin-bottom:6mm; padding-bottom:4mm; border-bottom:1px solid #cbd5e1; }
.detail-route-header > span { display:grid; place-items:center; width:14mm;height:14mm;border-radius:4mm;background:#312e81;color:#fff;font-size:15pt;font-weight:800; }
.detail-route-header h2 { margin:0; font-size:20pt; }.detail-route-header p{margin:1mm 0 0;color:#64748b;}
.step-detail { break-inside:auto; margin:0 0 8mm; padding:5mm; border:1px solid #d8e0ec; border-radius:5mm; background:#fff; }
.step-detail:not(.first-step) { break-before:page; }
.step-title { display:grid; grid-template-columns:13mm 1fr auto; gap:4mm; align-items:center; }
.step-number { display:grid;place-items:center;width:13mm;height:13mm;border-radius:50%;background:#eef2ff;color:#3730a3;font-size:13pt;font-weight:800; }
.step-title span:not(.condition-badge){color:#64748b;font-size:7pt;letter-spacing:.12em}.step-title h3{margin:1mm 0 0;font-size:13pt;line-height:1.4;}
.reaction-scheme { display:grid; grid-template-columns:1fr 34mm 1fr; align-items:center; gap:4mm; margin:5mm 0; padding:4mm; background:#f8fafc; border-radius:4mm; }
.scheme-side { display:flex; align-items:center; justify-content:center; gap:3mm; min-width:0; }
.scheme-plus { color:#64748b; font-size:18pt; font-weight:700; }
.scheme-arrow { text-align:center; color:#4f46e5; }.scheme-arrow span{display:block;font-size:28pt;line-height:1}.scheme-arrow small{display:block;margin-top:2mm;color:#6d28d9;font-size:7pt;}
.scheme-molecule { flex:1; min-width:0; text-align:center; }
.scheme-structure { height:34mm; display:flex; align-items:center; justify-content:center; }.scheme-structure svg{width:100%;height:100%;}
.scheme-molecule b { display:block; font-size:8pt; overflow-wrap:anywhere; }.scheme-molecule small{display:block;color:#64748b;font-family:Consolas,monospace;font-size:6.7pt;overflow-wrap:anywhere;}.scheme-molecule > span{display:inline-block;margin-top:1mm;color:#475569;font-size:6.5pt;}
.step-meta { margin-bottom:4mm; }
.detail-columns { display:block; }
.detail-row { display:contents; }
.detail-panel { break-inside:auto; margin-top:4mm; padding:3.5mm; border-radius:3mm; border:1px solid #e2e8f0; background:#fff; }
.detail-panel h4 { margin:0 0 3mm; color:#312e81; font-size:10pt; }
.condition-callout { display:flex; flex-direction:column; gap:1mm; padding:3mm; margin-bottom:3mm; border-radius:3mm; background:#fff7ed;color:#9a3412; }.condition-callout.model_predicted{background:#f5f3ff;color:#6d28d9}.condition-callout.source_exact{background:#ecfdf5;color:#047857}.condition-callout.source_recorded_unverified{background:#eff6ff;color:#0369a1}
.record-card,.validation-record,.provenance-record { margin-top:2.5mm; padding:2.5mm; border-left:1mm solid #cbd5e1; background:#f8fafc; }
.validation-record,.provenance-record { break-inside:avoid; page-break-inside:avoid; }
.record-card { break-inside:auto; page-break-inside:auto; overflow:visible; }
.record-card > b { display:block; break-after:avoid; page-break-after:avoid; }
.record-card.exact{border-color:#10b981}.record-card.prediction{border-color:#8b5cf6}.record-card.source,.record-card.procedure,.record-card.observation{border-color:#0ea5e9}
.record-card dl { min-width:0; margin:2mm 0 0; }
.record-card dl div { display:grid; grid-template-columns:42mm minmax(0,1fr); align-items:start; column-gap:3mm; min-width:0; margin-top:1.3mm; line-height:1.45; }
.record-card dt,.record-card dd { min-width:0; max-width:100%; overflow-wrap:anywhere; word-break:break-word; }
.record-card dt { color:#64748b; white-space:normal; }
.record-card dd { margin:0; white-space:pre-wrap; }
.validation-record > div:first-child { display:flex; flex-wrap:wrap; justify-content:space-between; gap:1mm 3mm; min-width:0; }
.validation-record > div:first-child > * { min-width:0; max-width:100%; overflow-wrap:anywhere; word-break:break-word; }
.validation-record code{display:block;margin-top:2mm;padding:2mm;background:#0f172a;color:#e2e8f0;border-radius:2mm;font-size:6.5pt;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;}
.tag-list{display:flex;flex-wrap:wrap;gap:1.5mm;margin-top:2mm}.detail-tag{padding:1mm 2mm;border-radius:999px;background:#e2e8f0;font-size:6.5pt}.detail-tag.reason,.detail-tag.failed-check{background:#fee2e2;color:#b91c1c}
.provenance-record { display:grid; grid-template-columns:42mm minmax(0,1fr); align-items:start; gap:2mm 3mm; min-width:0; }
.provenance-record > * { min-width:0; max-width:100%; overflow-wrap:anywhere; word-break:break-word; }
.provenance-record small{grid-column:2;color:#64748b;}
.missing-groups{display:flex;gap:3mm;margin-top:2mm;color:#b45309}.empty-copy,.empty-report{color:#64748b;font-style:italic}.structure-fallback{display:flex;align-items:center;justify-content:center;width:100%;height:100%;padding:2mm;border:1px dashed #cbd5e1;color:#64748b;font-family:Consolas,monospace;font-size:7pt;overflow-wrap:anywhere;}
"""


__all__ = [
    "WorkbenchPdfError",
    "compile_workbench_report_html",
    "render_workbench_pdf",
]
