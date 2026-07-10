"""Compile a finished blackboard run into a read-only explored route forest."""
from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from cascade_planner.harness.parent_route_proof import is_solved_parent_route_proof
from cascade_planner.harness.route_verifier import (
    is_accepted_route_verifier_report,
    verify_chemenzy_raw_routes,
)
from cascade_planner.harness.stitched_route import (
    compile_stitched_semisynthesis_route,
    is_validated_source_detail_literature_step,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception:  # pragma: no cover - route forest still renders without RDKit.
    Chem = None
    rdDepictor = None
    rdMolDescriptors = None
    rdMolDraw2D = None


SCHEMA_VERSION = "explored_route_forest.v1"

CONFIDENCE_RANK = {"failed": 0, "low": 1, "medium": 2, "medium_high": 3, "high": 4}
EXACTNESS_RANK = {
    "failed_or_unresolved": 0,
    "name_only": 1,
    "model_hypothesis": 2,
    "visual_inferred": 3,
    "named_literature": 4,
    "exact_literature_row": 5,
}

_STRUCTURE_CACHE: dict[str, dict[str, Any]] = {}

_SYNTHESIS_CLASS_FIELDS = {
    "synthesis_class",
    "route_claim",
    "route_class_hint",
    "route_objective_type",
    "objective_type",
    "process_type",
}
_SYNTHESIS_CLASSES = {"total_synthesis", "semisynthesis", "biosynthesis", "hybrid", "unspecified"}


def compile_explored_route_forest(
    blackboard: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    max_visual_branches: int = 8,
    max_proposal_branches: int = 10,
    max_template_branches: int = 8,
) -> dict[str, Any]:
    """Project a complex blackboard into user-facing explored route branches."""
    compiler = _RouteForestCompiler(blackboard, run_dir=run_dir)
    compiler.add_direct_verified_route_branch()
    compiler.add_stitched_verified_route_branch()
    compiler.add_subgoal_verified_route_branches()
    compiler.add_visual_branches(limit=max_visual_branches)
    compiler.add_process_evidence_branches()
    compiler.add_route_consensus_branches(limit=max_proposal_branches)
    compiler.add_route_consensus_graph_branches(limit=max_proposal_branches)
    compiler.add_proposal_branches(limit=max_proposal_branches)
    compiler.add_template_branches(limit=max_template_branches)
    compiler.add_exact_row_branch()
    compiler.add_diagnostic_failure_branch_if_empty()
    return compiler.finish()


def write_route_forest_artifacts(
    blackboard: dict[str, Any],
    *,
    run_dir: str | Path,
    forest_output: str | Path | None = None,
    html_output: str | Path | None = None,
    max_visual_branches: int = 8,
    max_proposal_branches: int = 10,
    max_template_branches: int = 8,
) -> dict[str, Any]:
    """Write the read-only route forest JSON and HTML display for a run."""
    run_path = Path(run_dir).resolve()
    forest_path = Path(forest_output).resolve() if forest_output is not None else run_path / "explored_route_forest.json"
    html_path = Path(html_output).resolve() if html_output is not None else run_path / "route_forest.html"
    forest = compile_explored_route_forest(
        blackboard,
        run_dir=run_path,
        max_visual_branches=max_visual_branches,
        max_proposal_branches=max_proposal_branches,
        max_template_branches=max_template_branches,
    )
    forest_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    forest_path.write_text(json.dumps(forest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_route_forest_html(forest), encoding="utf-8")
    return {
        "schema_version": "route_forest_outputs.v1",
        "forest_path": str(forest_path),
        "html_path": str(html_path),
        "forest": forest,
        "counts": dict(forest.get("counts") or {}),
        "target": dict(forest.get("target") or {}),
    }


def render_route_forest_html(forest: dict[str, Any]) -> str:
    return _render_route_forest_html_delivery(forest)


def _render_route_forest_html_delivery(forest: dict[str, Any]) -> str:
    data = json.dumps(forest, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")
    title = _escape(str(forest.get("target", {}).get("name") or forest.get("case_id") or "Route forest"))
    html_doc = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__ · 路线工作台</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #fff;
      --ink: #121826;
      --muted: #64748b;
      --line: #d7dee8;
      --soft: #eef3f8;
      --green: #2f855a;
      --blue: #2563eb;
      --amber: #b7791f;
      --orange: #d35f2d;
      --red: #be3b4a;
      --gray: #667085;
      --focus: #111827;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; overflow: hidden; }
    body {
      display: flex;
      flex-direction: column;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button { font: inherit; color: inherit; }
    .topbar {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: end;
      padding: 15px 22px 12px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    .eyebrow { color: var(--muted); font-size: 12px; font-weight: 780; text-transform: uppercase; }
    h1 { margin: 4px 0 8px; font-size: 24px; line-height: 1.2; }
    .summary, .legend, .toolbar { display: flex; flex-wrap: wrap; gap: 7px; }
    .toolbar { justify-content: flex-end; }
    .pill, .toggle-button, .tab-button, .clear-button {
      display: inline-flex;
      align-items: center;
      min-height: 27px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: #344054;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 740;
      white-space: nowrap;
      text-decoration: none;
      cursor: pointer;
    }
    .toggle-button:hover, .toggle-button.active, .tab-button:hover, .tab-button.active, .clear-button:hover {
      border-color: var(--focus);
      background: #f8fbff;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .workspace {
      flex: 1 1 auto;
      min-height: 0;
      display: grid;
      grid-template-columns: 286px minmax(620px, 1fr) 414px;
      gap: 14px;
      padding: 14px;
      overflow: hidden;
    }
    body.nav-collapsed .nav-panel { display: none; }
    body.inspector-collapsed .inspector-panel { display: none; }
    body.nav-collapsed:not(.inspector-collapsed) .workspace { grid-template-columns: minmax(620px, 1fr) 414px; }
    body.inspector-collapsed:not(.nav-collapsed) .workspace { grid-template-columns: 286px minmax(620px, 1fr); }
    body.nav-collapsed.inspector-collapsed .workspace { grid-template-columns: minmax(620px, 1fr); }
    body.embedded-route .topbar { display: none; }
    body.embedded-route .workspace {
      height: 100%;
      grid-template-columns: minmax(560px, 1fr) 380px;
      gap: 0;
      padding: 0;
    }
    body.embedded-route .nav-panel { display: none; }
    body.embedded-route .route-panel,
    body.embedded-route .inspector-panel {
      border-top: 0;
      border-bottom: 0;
      border-radius: 0;
    }
    body.embedded-route .route-canvas { padding: 12px; }
    .panel {
      min-height: 0;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .route-panel { display: flex; flex-direction: column; overflow: hidden; }
    .panel-head {
      position: sticky;
      top: 0;
      z-index: 3;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .panel-head h2 { margin: 0; font-size: 15px; }
    .panel-head p { margin: 5px 0 0; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .side-section { padding: 12px; border-bottom: 1px solid var(--line); }
    .side-section:last-child { border-bottom: 0; }
    .side-title { margin: 0 0 8px; color: var(--muted); font-size: 12px; font-weight: 780; text-transform: uppercase; }
    .view-list, .stage-list, .alt-list, .trace-list { display: grid; gap: 8px; }
    .view-button, .stage-button, .alt-button {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-left: 5px solid var(--gray);
      border-radius: 9px;
      background: #fff;
      padding: 9px 10px;
      cursor: pointer;
    }
    .view-button:hover, .view-button.active,
    .stage-button:hover, .stage-button.active,
    .alt-button:hover, .alt-button.active {
      border-color: #9aa7bb;
      background: #f8fbff;
    }
    .item-title { font-size: 13px; line-height: 1.35; font-weight: 780; overflow-wrap: anywhere; }
    .item-sub { margin-top: 5px; color: var(--muted); font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
    .hint, .empty { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .empty {
      border: 1px dashed #cbd5e1;
      border-radius: 9px;
      background: #fbfcfe;
      padding: 12px;
    }
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat { border: 1px solid var(--line); border-radius: 9px; padding: 9px; background: #fbfcfe; }
    .stat-value { font-size: 18px; font-weight: 800; }
    .stat-label { margin-top: 2px; color: var(--muted); font-size: 12px; }
    .route-head {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .route-title { margin: 0; font-size: 19px; line-height: 1.25; }
    .route-subtitle { margin-top: 5px; color: var(--muted); font-size: 13px; line-height: 1.45; overflow-wrap: anywhere; }
    .route-canvas { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 16px; }
    .flow-map-wrap {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
      padding: 8px;
    }
    .route-flow-svg { display: block; width: 100%; height: auto; }
    .flow-step-node { cursor: pointer; }
    .flow-step-node .flow-tile-bg {
      fill: #fff;
      stroke: #d8dee8;
      stroke-width: 1.2;
      filter: drop-shadow(0 1px 1px rgba(16, 24, 40, 0.08));
    }
    .flow-step-node:hover .flow-tile-bg { stroke: #98a4b8; }
    .flow-step-node.selected .flow-tile-bg { stroke: #111827; stroke-width: 3; }
    .flow-side-bar { opacity: .95; }
    .flow-connector { fill: none; stroke: #9aa7bb; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
    .flow-card-html {
      height: 100%;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 7px;
      padding: 10px 10px 9px 16px;
      color: #151b23;
      overflow: hidden;
    }
    .flow-card-head { display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 8px; align-items: start; }
    .flow-card-index {
      width: 24px;
      height: 24px;
      border: 1px solid #d8dee8;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #fff;
      color: #626f82;
      font-size: 11px;
      font-weight: 800;
    }
    .flow-card-title {
      font-size: 12px;
      font-weight: 800;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .flow-card-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 3px; max-height: 21px; overflow: hidden; }
    .flow-tag {
      border: 1px solid #d8dee8;
      border-radius: 999px;
      padding: 1px 6px;
      background: #fff;
      color: #344054;
      font-size: 9.5px;
      line-height: 15px;
      white-space: nowrap;
    }
    .flow-reaction {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 18px minmax(0, 1fr);
      gap: 6px;
      min-height: 0;
      align-items: stretch;
      overflow: hidden;
    }
    .flow-mini-list { display: grid; gap: 5px; align-content: start; min-width: 0; overflow: hidden; }
    .flow-mini-role { color: #626f82; font-size: 9px; font-weight: 800; text-transform: uppercase; }
    .flow-mini-mol {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 4px;
      border: 1px solid #edf0f5;
      border-radius: 8px;
      background: #fbfcfe;
      padding: 4px;
      min-height: 90px;
      overflow: hidden;
    }
    .flow-mini-structure {
      min-height: 66px;
      display: grid;
      place-items: center;
      overflow: hidden;
      color: #626f82;
      font-size: 9px;
      text-align: center;
    }
    .flow-mini-structure svg { width: 100%; height: auto; max-height: 72px; }
    .flow-mini-name {
      min-width: 0;
      color: #344054;
      font-size: 10px;
      line-height: 1.2;
      font-weight: 760;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .flow-mini-more { color: #626f82; font-size: 10px; line-height: 1.25; }
    .flow-mid-arrow { display: grid; place-items: center; color: #626f82; font-size: 21px; font-weight: 800; }
    .detail-tabs { display: flex; gap: 7px; margin-top: 10px; }
    .detail { padding: 14px; }
    .detail-title { margin: 0; font-size: 20px; line-height: 1.25; overflow-wrap: anywhere; }
    .detail-kind { margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .detail-section { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }
    .detail-section h3 { margin: 0 0 8px; font-size: 13px; color: #344054; }
    .kv { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 10px; padding: 6px 0; border-bottom: 1px solid #eef2f7; }
    .kv:last-child { border-bottom: 0; }
    .k { color: var(--muted); font-size: 12px; font-weight: 780; }
    .v { min-width: 0; font-size: 13px; line-height: 1.45; overflow-wrap: anywhere; }
    .molecule-pair { display: grid; grid-template-columns: minmax(0, 1fr) 28px minmax(0, 1fr); gap: 8px; align-items: stretch; }
    .mol-card { border: 1px solid var(--line); border-radius: 9px; background: #fbfcfe; padding: 8px; min-width: 0; }
    .mol-title { font-weight: 800; font-size: 13px; line-height: 1.3; overflow-wrap: anywhere; }
    .mol-meta { margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
    .mol-structure { min-height: 116px; margin-top: 6px; display: grid; place-items: center; overflow: hidden; }
    .mol-structure svg { max-width: 100%; max-height: 150px; }
    .detail-arrow { display: grid; place-items: center; color: var(--muted); font-size: 22px; font-weight: 800; }
    .condition-list { display: grid; gap: 7px; }
    .condition-line { display: grid; grid-template-columns: 82px minmax(0, 1fr); gap: 8px; }
    .condition-label { color: var(--muted); font-size: 12px; font-weight: 780; }
    .condition-value { font-size: 13px; overflow-wrap: anywhere; }
    .trace-row {
      border: 1px solid #edf0f5;
      border-radius: 9px;
      background: #fbfcfe;
      padding: 9px;
      font-size: 12px;
      line-height: 1.42;
    }
    .trace-row strong { display: block; font-size: 13px; overflow-wrap: anywhere; }
    .trace-row code { display: block; margin-top: 4px; color: #475569; white-space: pre-wrap; overflow-wrap: anywhere; }
    .notice {
      border: 1px solid #c8d7ee;
      background: #f6f9ff;
      border-radius: 9px;
      padding: 10px;
      color: #344054;
      font-size: 13px;
      line-height: 1.45;
    }
    .exact-exact_literature_row { border-left-color: var(--green); }
    .exact-named_literature { border-left-color: var(--blue); }
    .exact-visual_inferred { border-left-color: var(--amber); }
    .exact-model_hypothesis { border-left-color: var(--orange); }
    .exact-failed_or_unresolved { border-left-color: var(--red); }
    @media (max-width: 1120px) {
      html, body { overflow: auto; }
      .workspace { grid-template-columns: 1fr; overflow: visible; }
      .panel, .route-panel { min-height: 420px; }
      body.nav-collapsed .nav-panel,
      body.inspector-collapsed .inspector-panel { display: block; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <div class="eyebrow">Explored Route Forest</div>
      <h1 id="pageTitle"></h1>
      <div class="summary" id="summary"></div>
    </div>
    <div>
      <div class="toolbar">
        <button class="toggle-button active" type="button" data-toggle-panel="nav">导航栏</button>
        <button class="toggle-button active" type="button" data-toggle-panel="inspector">检查器</button>
        <button class="toggle-button" type="button" data-reset-route>恢复默认分支</button>
      </div>
      <div class="legend" id="legend"></div>
    </div>
  </header>
  <div class="workspace">
    <aside class="panel nav-panel">
      <div class="panel-head">
        <h2>路线导航</h2>
        <p>默认分支优先展示确定性重验路线；没有父路线证明时只展示明确标注的 advisory 分支。</p>
      </div>
      <div class="side-section">
        <h3 class="side-title">默认与候选分支</h3>
        <div class="view-list" id="viewPicker"></div>
      </div>
      <div class="side-section">
        <h3 class="side-title">步骤</h3>
        <div class="stage-list" id="stageIndex"></div>
      </div>
      <div class="side-section">
        <h3 class="side-title">证据概览</h3>
        <div class="stat-grid" id="evidenceStats"></div>
      </div>
    </aside>
    <main class="panel route-panel">
      <div class="route-head">
        <div>
          <h2 class="route-title" id="routeTitle"></h2>
          <div class="route-subtitle" id="routeSubtitle"></div>
        </div>
        <button class="clear-button" type="button" data-reset-route>默认分支</button>
      </div>
      <div class="route-canvas"><div id="mainRoute"></div></div>
    </main>
    <aside class="panel inspector-panel">
      <div class="panel-head">
        <h2>检查器</h2>
        <p>固定显示当前步骤、可替换备选和文献/工具过程。</p>
        <div class="detail-tabs">
          <button class="tab-button active" type="button" data-detail-tab="step">步骤</button>
          <button class="tab-button" type="button" data-detail-tab="alternatives">备选</button>
          <button class="tab-button" type="button" data-detail-tab="evidence">证据过程</button>
        </div>
      </div>
      <div class="detail" id="detail"></div>
    </aside>
  </div>
  <script id="forest-data" type="application/json">__DATA__</script>
  <script>
    const forest = JSON.parse(document.getElementById('forest-data').textContent);
    const nodes = new Map((forest.nodes || []).map(n => [n.node_id, n]));
    const steps = new Map((forest.steps || []).map(s => [s.step_id, s]));
    const branches = forest.branches || [];
    const modules = forest.modules || [];
    const relationships = forest.relationships || [];
    let selectedBranchId = defaultBranch().branch_id || '';
    let selectedStepId = firstStepId(defaultBranch());
    let detailTab = 'step';
    let activeReplacement = null;
    const panelClass = { nav: 'nav-collapsed', inspector: 'inspector-collapsed' };
    const exactLabel = {
      exact_literature_row: 'exact row',
      named_literature: '文献命名',
      visual_inferred: '图像推断',
      model_hypothesis: '模型假设',
      failed_or_unresolved: '未解决',
      name_only: '仅名称'
    };
    const confidenceLabel = { high: '高可信', medium_high: '中高可信', medium: '中可信', low: '低可信', failed: '失败' };
    const kindLabel = {
      stitched_verified_route: '拼接验证路线',
      direct_verified_route: '已验证路线',
      subgoal_verified_route: '子目标闭合',
      visual_chain: '图像链',
      process_evidence: '过程证据',
      route_consensus: '多信源共识（建议）',
      route_consensus_graph: 'Codex 多步路线假设（建议）',
      retrosynthetic_proposal: '模型提案',
      broad_template: '通用模板',
      exact_literature: 'exact row',
      diagnostic_failure: '诊断'
    };
    const synthesisClassLabel = {
      total_synthesis: '全合成',
      semisynthesis: '半合成',
      biosynthesis: '生物合成 / 生物转化',
      hybrid: '混合路线',
      unspecified: '未分类'
    };
    const moduleLabel = {
      sidechain_installation: '侧链安装',
      amide_or_sidechain_assembly: '酰胺/侧链连接',
      protection_state_adjustment: '保护基调整',
      semisynthesis_anchor: '半合成锚点',
      heterocycle_core_construction: '杂环母核构建',
      scaffold_core_construction: '骨架/母核构建',
      visual_literature_hint: '图像文献提示',
      other_route_module: '其他路线模块',
      ketal_deprotection: '缩酮脱保护',
      ester_hydrolysis: '酯水解',
      salt_formation: '成盐/分离',
      form_adjustment: '盐型/游离酸调整',
      subgoal_stock_closure: 'ChemEnzy 子目标闭合',
      diagnostic_failure: '诊断失败',
      visual_failed_or_empty: '图像链失败或为空'
    };
    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function isCorrupt(value) {
      return /[�]|锟|閹|鐠|濡|閸|閺|闁|娑|鎺|璺|妯|鍥|鏂|榛|姝|澶|鍙|鍒|浜|鏉|绾|缁|瀛|骞|浠|涓|侀|湪|棴|濈/.test(String(value || ''));
    }
    function clean(value, fallback='') {
      const text = String(value || '').trim();
      return text && !isCorrupt(text) ? text : fallback;
    }
    function basename(path) {
      const text = String(path || '');
      return text.split(/[\\\\/]/).filter(Boolean).pop() || text;
    }
    function chip(text, cls='') { return `<span class="pill ${cls}">${esc(text)}</span>`; }
    function cssColor(exactness) {
      return {
        exact_literature_row: 'var(--green)',
        named_literature: 'var(--blue)',
        visual_inferred: 'var(--amber)',
        model_hypothesis: 'var(--orange)',
        failed_or_unresolved: 'var(--red)',
        name_only: 'var(--gray)'
      }[exactness] || 'var(--gray)';
    }
    function flowHexColor(exactness) {
      return {
        exact_literature_row: '#2f855a',
        named_literature: '#2563eb',
        visual_inferred: '#b7791f',
        model_hypothesis: '#d35f2d',
        failed_or_unresolved: '#be3b4a',
        name_only: '#667085'
      }[exactness] || '#667085';
    }
    function branchRank(kind) {
      return {
        stitched_verified_route: 8,
        direct_verified_route: 6,
        subgoal_verified_route: 5.8,
        exact_literature: 5,
        process_evidence: 3.5,
        visual_chain: 3,
        route_consensus: 2.5,
        route_consensus_graph: 2.7,
        retrosynthetic_proposal: 2,
        broad_template: 1,
        diagnostic_failure: 0
      }[kind] || 0;
    }
    function defaultBranch() {
      return branches.find(b => b.branch_id === forest.primary_branch_id)
        || branches.find(b => b.kind === 'stitched_verified_route')
        || branches.find(b => b.kind === 'direct_verified_route')
        || branches.find(b => b.kind === 'exact_literature')
        || branches.find(b => b.kind === 'route_consensus_graph')
        || branches.find(b => b.kind === 'route_consensus')
        || branches[0]
        || {};
    }
    function currentBranch() { return branches.find(b => b.branch_id === selectedBranchId) || defaultBranch() || {}; }
    function branchById(branchId) { return branches.find(b => b.branch_id === branchId) || {}; }
    function firstStepId(branch) { return ((branch || {}).step_ids || [])[0] || ''; }
    function routeSteps(branch=currentBranch()) { return ((branch || {}).step_ids || []).map(id => steps.get(id)).filter(Boolean); }
    function branchForStep(stepId) { return branches.find(b => (b.step_ids || []).includes(stepId)) || {}; }
    function moduleForStep(step) { return modules.find(m => m.module_key === (step || {}).module_key) || {}; }
    function targetName() { return clean(forest.target?.name, clean(forest.case_id, 'target')); }
    function branchTitle(branch) {
      const kind = branch?.kind || '';
      if (kind === 'stitched_verified_route') return `拼接验证路线：${targetName()}`;
      if (kind === 'direct_verified_route') return `已验证路线：${targetName()}`;
      if (kind === 'subgoal_verified_route') return 'ChemEnzy 子目标闭合';
      return clean(branch?.title, kindLabel[kind] || '路线分支');
    }
    function branchSummary(branch) {
      if (branch?.kind === 'stitched_verified_route') {
        return '由已重验的库存闭合路线和严格文献链组成的统一合成 DAG；其他黑板分支仍是独立建议。';
      }
      return clean(branch?.summary, clean(branch?.recommendation, '路线分支'));
    }
    function branchProofText(branch) {
      const verified = branch?.solved === true
        && branch?.executable === true
        && branch?.advisory_only === false
        && branch?.not_parent_route_proof === false;
      if (verified) return '父路线：确定性验证通过';
      if (branch?.kind === 'subgoal_verified_route') return '父路线未闭合：这里只展示已验证的子目标片段';
      return '父路线未闭合：当前分支仅供探索，不是可执行路线';
    }
    function sourceRefsText(refs) {
      const rows = (refs || []).filter(Boolean).map(ref => basename(ref)).filter(Boolean);
      return rows.length ? rows.slice(0, 3).join(' · ') : '来源未记录';
    }
    function stepTitle(step, index=0) {
      return clean(step?.label, `${moduleLabel[step?.module_key] || step?.module_key || '步骤'} ${index || ''}`.trim());
    }
    function stepModule(step) { return moduleLabel[step?.module_key] || clean(step?.module_label, step?.module_key || '未分组'); }
    function nodeLabel(node) { return clean(node?.label, node?.smiles || node?.representation_kind || '未命名分子'); }
    function nodeNames(ids) {
      return (ids || []).map(id => nodeLabel(nodes.get(id))).filter(Boolean).join(' + ');
    }
    function routeEvidenceText(branch) {
      const rows = routeSteps(branch);
      const counts = rows.reduce((acc, step) => {
        const key = exactLabel[step.exactness] || step.exactness || '未知';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {});
      const exactText = Object.entries(counts).map(([key, value]) => `${key} ${value}`).join(' · ');
      const consensusText = branch.kind === 'route_consensus'
        ? ` · 独立支持组 ${branch.independent_source_count || 0} · 仅建议`
        : '';
      const synthesisClass = synthesisClassLabel[branch.synthesis_class] || synthesisClassLabel.unspecified;
      return `${rows.length} 步 · ${kindLabel[branch.kind] || branch.kind || '路线'} · ${synthesisClass} · ${exactText || '未分级'}${consensusText}`;
    }
    function principalBranches() {
      const core = defaultBranch();
      const rows = [core, ...branches.filter(branch => branch.kind === 'route_consensus')];
      const seen = new Set();
      return rows.filter(branch => {
        if (!branch?.branch_id || seen.has(branch.branch_id)) return false;
        seen.add(branch.branch_id);
        return true;
      }).slice(0, 16);
    }
    function alternativesForStep(step) {
      const key = step?.module_key || '';
      if (!key || key === 'other_route_module' || key === 'diagnostic_failure') return [];
      const ids = new Set([...(moduleForStep(step).step_ids || [])]);
      for (const candidate of steps.values()) {
        if (candidate.module_key === key) ids.add(candidate.step_id);
      }
      const currentBranchId = branchForStep(step.step_id).branch_id || '';
      const grouped = new Map();
      for (const id of ids) {
        if (!id || id === step.step_id) continue;
        const alt = steps.get(id);
        if (!alt) continue;
        const branch = branchForStep(id);
        if (branch.branch_id && branch.branch_id === currentBranchId) continue;
        const groupKey = [stepTitle(alt), branch.kind || '', alt.exactness || '', alt.module_key || ''].join('|');
        if (!grouped.has(groupKey)) grouped.set(groupKey, { step: alt, branch, count: 0 });
        grouped.get(groupKey).count += 1;
      }
      return Array.from(grouped.values()).sort((a, b) => {
        const rankDelta = branchRank(b.branch.kind) - branchRank(a.branch.kind);
        if (rankDelta) return rankDelta;
        return stepTitle(a.step).localeCompare(stepTitle(b.step));
      }).slice(0, 16);
    }
    function branchTailSteps(branch, fromStepId) {
      const rows = ((branch || {}).step_ids || []).map(id => steps.get(id)).filter(Boolean);
      const idx = rows.findIndex(step => step.step_id === fromStepId);
      return idx >= 0 ? rows.slice(idx + 1) : [];
    }
    function activeSteps() {
      const baseRows = routeSteps();
      if (!activeReplacement) return baseRows;
      const alt = steps.get(activeReplacement.altStepId);
      if (!alt) return baseRows;
      const baseIndex = baseRows.findIndex(step => step.step_id === activeReplacement.baseStepId);
      const prefix = baseIndex >= 0 ? baseRows.slice(0, baseIndex) : [];
      const altBranch = branchById(activeReplacement.altBranchId) || branchForStep(alt.step_id);
      const tail = branchTailSteps(altBranch, alt.step_id);
      const seen = new Set();
      return [...prefix, alt, ...tail].filter(step => {
        if (!step || seen.has(step.step_id)) return false;
        seen.add(step.step_id);
        return true;
      });
    }
    function conditionText(step) {
      const rows = (step?.conditions || []).filter(Boolean);
      if (step?.condition_summary && !isCorrupt(step.condition_summary)) return step.condition_summary;
      if (rows.length) return rows.map(row => `${clean(row.label, '条件')}: ${clean(row.value, '')}`).filter(Boolean).join(' · ');
      return '条件未记录';
    }
    function sourceText(step) {
      const refs = (step?.source_refs || []).filter(Boolean);
      return refs.length ? refs.slice(0, 3).map(ref => basename(ref)).join(' · ') : '来源未记录';
    }
    function moleculeMini(nodeId, roleLabel) {
      const node = nodes.get(nodeId);
      if (!node) return '';
      const structure = node.structure_svg
        ? node.structure_svg
        : `<div>${esc(node.smiles || node.representation_kind || '无结构')}</div>`;
      return `<div class="flow-mini-mol" data-node="${esc(node.node_id)}">
        <div class="flow-mini-structure">${structure}</div>
        <div class="flow-mini-name">${esc(roleLabel ? `${roleLabel}: ${nodeLabel(node)}` : nodeLabel(node))}</div>
      </div>`;
    }
    function moleculeMiniList(ids, roleLabel) {
      const shown = (ids || []).slice(0, 1).map(id => moleculeMini(id, roleLabel)).join('');
      const more = (ids || []).length > 1 ? `<div class="flow-mini-more">+ ${(ids || []).length - 1} 个，见详情</div>` : '';
      return shown || `<div class="empty">未记录</div>` + more;
    }
    function flowStepTile(step, index, eventStepId, x, y, w, h, replacement=false) {
      const selected = selectedStepId === eventStepId ? ' selected' : '';
      const tags = [
        confidenceLabel[step.confidence] || step.confidence || '未知',
        exactLabel[step.exactness] || step.exactness || '未知',
        replacement ? '备选预览' : ''
      ].filter(Boolean);
      return `<g class="flow-step-node exact-${esc(step.exactness || 'name_only')}${selected}" data-route-step="${esc(eventStepId)}">
        <rect class="flow-tile-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="14"></rect>
        <rect class="flow-side-bar" x="${x}" y="${y}" width="7" height="${h}" rx="4" fill="${flowHexColor(step.exactness)}"></rect>
        <foreignObject x="${x}" y="${y}" width="${w}" height="${h}">
          <div xmlns="http://www.w3.org/1999/xhtml" class="flow-card-html">
            <div class="flow-card-head">
              <div class="flow-card-index">${index}</div>
              <div>
                <div class="flow-card-title">${esc(stepTitle(step, index))}</div>
                <div class="flow-card-tags">${tags.map(tag => `<span class="flow-tag">${esc(tag)}</span>`).join('')}</div>
              </div>
            </div>
            <div class="flow-reaction">
              <div class="flow-mini-list"><div class="flow-mini-role">反应物</div>${moleculeMiniList(step.from_node_ids, '前体')}</div>
              <div class="flow-mid-arrow">→</div>
              <div class="flow-mini-list"><div class="flow-mini-role">产物</div>${moleculeMiniList(step.to_node_ids, '产物')}</div>
            </div>
          </div>
        </foreignObject>
      </g>`;
    }
    function routeFlowSvg(rows) {
      if (!rows.length) return '<div class="empty">没有可展示路线。</div>';
      const host = document.getElementById('mainRoute');
      const hostWidth = Math.max(720, host?.clientWidth || 1200);
      const margin = 18;
      const gapX = 24;
      const gapY = rows.length > 12 ? 30 : 34;
      const maxCols = hostWidth >= 1320 ? 5 : (hostWidth >= 900 ? 4 : 3);
      const cols = Math.max(2, Math.min(rows.length, maxCols));
      const tileW = Math.max(238, Math.min(320, Math.floor((hostWidth - margin * 2 - (cols - 1) * gapX) / cols)));
      const tileH = rows.length > 14 ? 178 : 198;
      const rowCount = Math.ceil(rows.length / cols);
      const svgW = margin * 2 + cols * tileW + Math.max(0, cols - 1) * gapX;
      const svgH = margin * 2 + rowCount * tileH + Math.max(0, rowCount - 1) * gapY;
      const positions = rows.map((step, idx) => {
        const band = Math.floor(idx / cols);
        const slot = idx % cols;
        const snake = band % 2 === 1;
        const col = snake ? cols - 1 - slot : slot;
        return { step, index: idx + 1, col, lane: band, x: margin + col * (tileW + gapX), y: margin + band * (tileH + gapY) };
      });
      const byId = new Map(positions.map(pos => [pos.step.step_id, pos]));
      const connectors = rows.slice(0, -1).map((step, index) => {
        const a = byId.get(step.step_id);
        const b = byId.get(rows[index + 1].step_id);
        if (!a || !b) return '';
        if (a.lane === b.lane) {
          const forward = b.x > a.x;
          const x1 = forward ? a.x + tileW + 5 : a.x - 5;
          const y1 = a.y + tileH / 2;
          const x2 = forward ? b.x - 5 : b.x + tileW + 5;
          const y2 = b.y + tileH / 2;
          const dx = Math.max(24, Math.abs(x2 - x1) * 0.36);
          const c1 = forward ? x1 + dx : x1 - dx;
          const c2 = forward ? x2 - dx : x2 + dx;
          return `<path class="flow-connector" marker-end="url(#arrowHead)" d="M ${x1} ${y1} C ${c1} ${y1}, ${c2} ${y2}, ${x2} ${y2}"></path>`;
        }
        const x1 = a.x + tileW / 2;
        const y1 = a.y + tileH + 5;
        const x2 = b.x + tileW / 2;
        const y2 = b.y - 5;
        const dy = Math.max(20, (y2 - y1) * 0.45);
        return `<path class="flow-connector" marker-end="url(#arrowHead)" d="M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}"></path>`;
      });
      const tiles = positions.map(pos => {
        const replacement = activeReplacement && pos.step.step_id === activeReplacement.altStepId;
        const eventStepId = replacement ? activeReplacement.baseStepId : pos.step.step_id;
        return flowStepTile(pos.step, pos.index, eventStepId, pos.x, pos.y, tileW, tileH, replacement);
      }).join('');
      return `<div class="flow-map-wrap">
        <svg id="routeFlowSvg" class="route-flow-svg" viewBox="0 0 ${svgW} ${svgH}" width="100%" height="${svgH}" role="img" aria-label="当前路线分支 SVG 流程图">
          <defs>
            <marker id="arrowHead" markerWidth="9" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
              <path d="M 0 0 L 9 4 L 0 8 z" fill="#9aa7bb"></path>
            </marker>
          </defs>
          ${connectors.join('')}
          ${tiles}
        </svg>
      </div>`;
    }
    function selectedBaseStep() {
      return steps.get(selectedStepId) || routeSteps()[0] || null;
    }
    function renderSummary() {
      const c = forest.counts || {};
      document.getElementById('pageTitle').textContent = `${targetName()} · 路线工作台`;
      document.getElementById('summary').innerHTML = [
        chip((forest.primary_selection || {}).status === 'deterministically_verified' ? '父路线：已验证' : '父路线：未闭合'),
        chip(`${c.branches || branches.length || 0} 条探索分支`),
        chip(`${c.steps || steps.size || 0} 个步骤`),
        chip(`${c.nodes || nodes.size || 0} 个分子节点`),
        chip(`${c.relationships || relationships.length || 0} 个路线关系`),
        chip('只读结果页')
      ].join('');
      document.getElementById('legend').innerHTML = [
        ['exact row', '#2f855a'],
        ['文献命名', '#2563eb'],
        ['图像推断', '#b7791f'],
        ['模型假设', '#d35f2d']
      ].map(([label, color]) => `<span class="pill"><span class="dot" style="background:${color}"></span>${esc(label)}</span>`).join('');
    }
    function renderLayoutControls() {
      document.querySelectorAll('[data-toggle-panel]').forEach(button => {
        const key = button.getAttribute('data-toggle-panel') || '';
        const cls = panelClass[key];
        const visible = !document.body.classList.contains(cls);
        button.classList.toggle('active', visible);
        button.setAttribute('aria-pressed', visible ? 'true' : 'false');
        button.onclick = () => {
          document.body.classList.toggle(cls);
          renderLayoutControls();
          renderRoute();
        };
      });
      document.querySelectorAll('[data-reset-route]').forEach(button => button.onclick = clearReplacement);
    }
    function renderViewPicker() {
      const html = principalBranches().map(branch => `<button class="view-button${branch.branch_id === currentBranch().branch_id ? ' active' : ''}" style="border-left-color:${cssColor((routeSteps(branch)[0] || {}).exactness || 'name_only')}" data-view-branch="${esc(branch.branch_id)}">
        <div class="item-title">${esc(branchTitle(branch))}</div>
        <div class="item-sub">${esc(routeEvidenceText(branch))}</div>
      </button>`).join('');
      document.getElementById('viewPicker').innerHTML = html || '<div class="empty">没有默认或候选分支。</div>';
      document.querySelectorAll('[data-view-branch]').forEach(el => el.addEventListener('click', () => {
        selectedBranchId = el.getAttribute('data-view-branch') || '';
        selectedStepId = firstStepId(currentBranch());
        activeReplacement = null;
        detailTab = 'step';
        renderAll();
      }));
    }
    function renderStepIndex() {
      const rows = activeSteps();
      document.getElementById('stageIndex').innerHTML = rows.map((step, index) => {
        const active = selectedStepId === step.step_id || (activeReplacement?.baseStepId === step.step_id);
        const altCount = alternativesForStep(step).length;
        const preview = activeReplacement && activeReplacement.baseStepId === step.step_id ? ' · 备选预览' : '';
        return `<button class="stage-button exact-${esc(step.exactness || 'name_only')}${active ? ' active' : ''}" data-stage-step="${esc(step.step_id)}">
          <div class="item-title">${index + 1}. ${esc(stepTitle(step, index + 1))}</div>
          <div class="item-sub">${esc(stepModule(step))} · ${altCount} 个备选${preview}</div>
        </button>`;
      }).join('') || '<div class="empty">当前分支没有步骤。</div>';
      document.querySelectorAll('[data-stage-step]').forEach(el => el.addEventListener('click', () => {
        selectedStepId = el.getAttribute('data-stage-step') || '';
        detailTab = 'step';
        renderAll();
      }));
    }
    function renderStats() {
      const c = forest.counts || {};
      const trace = forest.run_trace || {};
      const lit = trace.literature_counts || {};
      const stats = [
        ['文献候选', lit.source_candidates ?? 0],
        ['PDF/图像链', c.visual_chains || 0],
        ['过程锚点', c.process_evidence_rows || 0],
        ['exact rows', c.exact_rows || 0],
        ['ChemEnzy 子目标', (forest.evidence_index?.route_expansion_subgoals || []).length],
        ['共识候选', c.route_consensus_proposals || 0],
        ['备选提案', c.proposals || 0],
      ];
      document.getElementById('evidenceStats').innerHTML = stats.map(([label, value]) => `<div class="stat"><div class="stat-value">${esc(value)}</div><div class="stat-label">${esc(label)}</div></div>`).join('');
    }
    function renderRoute() {
      const branch = currentBranch();
      const rows = activeSteps();
      document.getElementById('routeTitle').textContent = activeReplacement ? `${branchTitle(branch)} · 备选预览` : branchTitle(branch);
      document.getElementById('routeSubtitle').textContent = activeReplacement
        ? `${branchProofText(branch)} · 画布已替换当前步骤；后续步骤按该备选所属分支中可接续的片段展开。`
        : `${branchProofText(branch)} · ${branchSummary(branch)}`;
      document.getElementById('mainRoute').innerHTML = routeFlowSvg(rows);
      document.querySelectorAll('[data-route-step]').forEach(el => el.addEventListener('click', () => {
        selectedStepId = el.getAttribute('data-route-step') || '';
        detailTab = 'step';
        document.body.classList.remove(panelClass.inspector);
        renderAll();
      }));
    }
    function renderDetailTabs() {
      document.querySelectorAll('[data-detail-tab]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-detail-tab') === detailTab);
        el.onclick = () => {
          detailTab = el.getAttribute('data-detail-tab') || 'step';
          renderAll();
        };
      });
    }
    function moleculeDetail(ids) {
      const cards = (ids || []).map(id => nodes.get(id)).filter(Boolean).map(node => {
        const structure = node.structure_svg ? node.structure_svg : `<div>${esc(node.smiles || node.representation_kind || '无结构')}</div>`;
        return `<div class="mol-card">
          <div class="mol-title">${esc(nodeLabel(node))}</div>
          <div class="mol-meta">${esc([node.formula ? `分子式 ${node.formula}` : '', node.smiles || node.representation_kind || ''].filter(Boolean).join(' · '))}</div>
          <div class="mol-structure">${structure}</div>
        </div>`;
      }).join('');
      return cards || '<div class="empty">未记录</div>';
    }
    function downstreamUsage(step, currentRows=activeSteps()) {
      const products = new Set((step || {}).to_node_ids || []);
      if (!products.size) return [];
      const currentIndex = currentRows.findIndex(row => row.step_id === step.step_id);
      const rows = [];
      currentRows.forEach((candidate, index) => {
        if (candidate.step_id === step.step_id) return;
        if (currentIndex >= 0 && index <= currentIndex) return;
        const used = (candidate.from_node_ids || []).filter(id => products.has(id));
        used.forEach(nodeId => rows.push({ step: candidate, node: nodes.get(nodeId), index: index + 1 }));
      });
      return rows;
    }
    function downstreamBlock(step, currentRows=activeSteps()) {
      const rows = downstreamUsage(step, currentRows);
      if (!rows.length) return '<div class="empty">当前展示路线的后续步骤没有继续使用该步骤产物；若这是备选预览，说明分支在此处截断。</div>';
      return `<div class="trace-list">${rows.map(row => `<div class="trace-row">
        <strong>step ${row.index}: ${esc(stepTitle(row.step, row.index))}</strong>
        <div>使用产物：${esc(nodeLabel(row.node))}</div>
      </div>`).join('')}</div>`;
    }
    function conditionBlock(step) {
      const rows = (step?.conditions || []).filter(Boolean);
      if (!rows.length) return `<div class="empty">${esc(conditionText(step))}</div>`;
      return `<div class="condition-list">${rows.map(row => `<div class="condition-line">
        <div class="condition-label">${esc(clean(row.label, '条件'))}</div>
        <div class="condition-value">${esc(clean(row.value, ''))}</div>
      </div>`).join('')}</div>`;
    }
    function consensusSupportBlock(step) {
      if (step?.origin !== 'route_consensus') return '';
      const supports = (step.support_records || []).map(row => {
        const refs = [...(row.source_refs || []), ...(row.evidence_refs || [])];
        return `<div class="trace-row">
          <strong>${esc(row.source_channel || 'other')}</strong>
          <div>${esc(row.evidence_level || 'model_only')} · ${esc(row.confidence || 'low')} · support group: ${esc(row.support_group || 'unbound')}</div>
          <code>${esc(sourceRefsText(refs))}</code>
        </div>`;
      }).join('');
      const conflicts = (step.conflicts || []).map(row => `<div class="trace-row">
        <strong>${esc(row.field || 'conflict')}</strong>
        <div>${esc(JSON.stringify(row.values || []))}</div>
      </div>`).join('');
      const groups = (step.independent_support_groups || []).join(' · ') || 'none';
      return `<div class="detail-section consensus-audit">
        <h3>Multi-source consensus audit</h3>
        <div class="notice">Advisory only — not solved or executable. Independent support groups: ${esc(step.independent_source_count || 0)} (${esc(groups)}). Codex roles are correlated.</div>
        <h3>Source channel support</h3>
        <div class="trace-list">${supports || '<div class="empty">No source records.</div>'}</div>
        <h3>Condition conflicts</h3>
        <div class="trace-list">${conflicts || '<div class="empty">No recorded conflicts.</div>'}</div>
      </div>`;
    }
    function renderStepDetail(baseStep) {
      const alt = activeReplacement && activeReplacement.baseStepId === baseStep.step_id ? steps.get(activeReplacement.altStepId) : null;
      const step = alt || baseStep;
      const rows = activeSteps();
      document.getElementById('detail').innerHTML = `<h3 class="detail-title">${esc(stepTitle(step))}</h3>
        <div class="detail-kind">${alt ? `备选预览：替换默认分支步骤「${stepTitle(baseStep)}」` : (currentBranch().advisory_only ? 'Advisory 分支步骤（不是父路线证明）' : '已验证父路线步骤')}</div>
        <div class="detail-section">
          <div class="molecule-pair">
            <div>${moleculeDetail(step.from_node_ids)}</div>
            <div class="detail-arrow">→</div>
            <div>${moleculeDetail(step.to_node_ids)}</div>
          </div>
        </div>
        <div class="detail-section">
          <div class="kv"><div class="k">模块</div><div class="v">${esc(stepModule(step))}</div></div>
          <div class="kv"><div class="k">可信度</div><div class="v">${esc(confidenceLabel[step.confidence] || step.confidence || '未知')}</div></div>
          <div class="kv"><div class="k">证据级别</div><div class="v">${esc(exactLabel[step.exactness] || step.exactness || '未知')}</div></div>
          <div class="kv"><div class="k">来源</div><div class="v">${esc(sourceText(step))}</div></div>
          <div class="kv"><div class="k">说明</div><div class="v">${esc(clean(step.summary, '未记录'))}</div></div>
        </div>
        ${consensusSupportBlock(step)}
        <div class="detail-section"><h3>条件：</h3>${conditionBlock(step)}</div>
        <div class="detail-section"><h3>后续使用</h3>${downstreamBlock(step, rows)}</div>
        <div class="detail-section"><h3>缺口 / 风险</h3>${(step.missing || step.risk_flags || []).length ? `<div class="trace-list">${(step.missing || step.risk_flags || []).map(x => `<div class="trace-row">${esc(clean(x, String(x || '')))}</div>`).join('')}</div>` : '<div class="empty">无记录</div>'}</div>`;
    }
    function renderAlternativesDetail(step) {
      const alts = alternativesForStep(step);
      const altHtml = alts.map(row => {
        const alt = row.step;
        const branch = row.branch || {};
        const active = activeReplacement?.altStepId === alt.step_id ? ' active' : '';
        const tail = branchTailSteps(branch, alt.step_id).length;
        const suffix = row.count > 1 ? ` · 合并 ${row.count} 条相似记录` : '';
        return `<button class="alt-button exact-${esc(alt.exactness || 'name_only')}${active}" data-alt-step="${esc(alt.step_id)}" data-alt-branch="${esc(branch.branch_id || '')}">
          <div class="item-title">${esc(stepTitle(alt))}</div>
          <div class="item-sub">${esc(branchTitle(branch))} · ${esc(confidenceLabel[alt.confidence] || alt.confidence || '')} · 后续 ${tail} 步${esc(suffix)}</div>
        </button>`;
      }).join('');
      document.getElementById('detail').innerHTML = `<h3 class="detail-title">${esc(stepTitle(step))}</h3>
        <div class="detail-kind">这些备选属于同一反应模块。点击后会替换主画布中的当前步骤，并沿该备选分支展开后续可接续片段。</div>
        <div class="detail-section"><div class="alt-list">${altHtml || '<div class="empty">没有同模块备选。</div>'}</div></div>
        ${activeReplacement ? '<div class="detail-section"><button class="clear-button" type="button" data-reset-route>恢复默认分支</button></div>' : ''}`;
      document.querySelectorAll('[data-alt-step]').forEach(el => el.addEventListener('click', () => {
        activeReplacement = {
          baseStepId: step.step_id,
          altStepId: el.getAttribute('data-alt-step') || '',
          altBranchId: el.getAttribute('data-alt-branch') || ''
        };
        selectedStepId = step.step_id;
        detailTab = 'step';
        renderAll();
      }));
    }
    function traceRows(rows, emptyText) {
      if (!rows || !rows.length) return `<div class="empty">${esc(emptyText || '无记录')}</div>`;
      return `<div class="trace-list">${rows.map(row => {
        const title = clean(row.title || row.source_title || row.source_ref || row.action_type || row.key, '记录');
        const meta = [
          row.round_index ? `round ${row.round_index}` : '',
          row.accepted === true ? 'accepted' : '',
          row.accepted === false ? 'not accepted' : '',
          row.step_count ? `${row.step_count} steps` : '',
          row.useful_artifact ? 'useful artifact' : ''
        ].filter(Boolean).join(' · ');
        const code = row.path || row.local_pdf || row.source_ref || (row.reasons || []).join('; ') || '';
        return `<div class="trace-row"><strong>${esc(title)}</strong>${meta ? `<div>${esc(meta)}</div>` : ''}${code ? `<code>${esc(basename(code))}</code>` : ''}</div>`;
      }).join('')}</div>`;
    }
    function relationshipRows() {
      if (!relationships.length) return '<div class="empty">没有编译出路线关系。</div>';
      return `<div class="trace-list">${relationships.slice(0, 30).map(rel => {
        const left = branchById(rel.from_branch_id);
        const right = branchById(rel.to_branch_id);
        const shared = (rel.shared_node_labels || []).slice(0, 3).join(' · ');
        return `<div class="trace-row"><strong>${esc(branchTitle(left))} → ${esc(branchTitle(right))}</strong>
          <div>${esc(clean(rel.kind, '路线关系'))}${shared ? ` · ${esc(shared)}` : ''}</div>
          <div>${esc(clean(rel.summary, ''))}</div>
          <code>${esc(sourceRefsText(rel.source_refs))}</code>
        </div>`;
      }).join('')}</div>`;
    }
    function consensusOverview() {
      const consensus = forest.route_consensus || {};
      if (!consensus.available) return '<div class="empty">没有 route_consensus.v1 记录。</div>';
      const rows = (consensus.proposals || []).map(row => `<div class="trace-row">
        <strong>#${esc(row.rank || '?')} ${esc(row.reaction_family || 'unspecified')}</strong>
        <div>${esc(row.evidence_level || 'model_only')} · ${esc(row.status || 'model_hypothesis')} · independent support groups ${esc(row.independent_source_count || 0)}</div>
        <div>channels: ${esc((row.source_channels || []).join(', ') || 'none')}</div>
        <div>conflicts: ${esc((row.conflicts || []).length)} · advisory only · not executable</div>
        <code>${esc(sourceRefsText(row.source_refs))}</code>
      </div>`).join('');
      return `<div class="notice">Canonical route_consensus.v1 is displayed as candidate disconnections. It never marks a route solved.</div>
        <div class="trace-list">${rows || '<div class="empty">没有有效共识候选。</div>'}</div>`;
    }
    function renderEvidenceDetail() {
      const evidence = forest.evidence_index || {};
      const trace = forest.run_trace || {};
      const lit = trace.literature_counts || {};
      const counts = [
        `文献候选 ${lit.source_candidates || 0}`,
        `PDF结构证据 ${lit.pdf_structure_evidence || 0}`,
        `图像链 ${lit.visual_chains || 0}`,
        `过程锚点 ${lit.process_evidence_rows || 0}`,
        `exact rows ${lit.exact_rows || 0}`,
        `scout ${lit.scout_attempts || 0}`
      ].join(' · ');
      document.getElementById('detail').innerHTML = `<h3 class="detail-title">文献与运行过程</h3>
        <div class="detail-kind">这些记录来自 blackboard、artifact_refs 和 action_history，是核心路线的证据库与审计轨迹。</div>
        <div class="detail-section"><div class="notice">${esc(counts)}</div></div>
        <div class="detail-section"><h3>Multi-source route consensus</h3>${consensusOverview()}</div>
        <div class="detail-section"><h3>文献候选 / PDF</h3>${traceRows(evidence.source_candidates || [], '没有文献候选记录。')}</div>
        <div class="detail-section"><h3>图像链</h3>${traceRows(evidence.visual_chains || [], '没有图像链记录。')}</div>
        <div class="detail-section"><h3>过程锚点</h3>${traceRows(evidence.process_evidence_rows || [], '没有过程证据记录。')}</div>
        <div class="detail-section"><h3>ChemEnzy 子目标闭合</h3>${traceRows(evidence.route_expansion_subgoals || [], '没有 ChemEnzy 子目标闭合记录。')}</div>
        <div class="detail-section"><h3>exact row 审计</h3>${traceRows(evidence.exact_chain_audits || [], '没有 exact row 审计记录。')}</div>
        <div class="detail-section"><h3>路线关系</h3>${relationshipRows()}</div>
        <div class="detail-section"><h3>工具动作</h3>${traceRows((trace.actions || []).slice(0, 40), '没有 action_history 记录。')}</div>
        <div class="detail-section"><h3>产物文件</h3>${traceRows((trace.artifact_refs || []).slice(0, 40), '没有 artifact_refs 记录。')}</div>`;
    }
    function renderDetail() {
      if (detailTab === 'evidence') return renderEvidenceDetail();
      const step = selectedBaseStep();
      if (!step) {
        document.getElementById('detail').innerHTML = '<div class="empty">没有选中的步骤。</div>';
        return;
      }
      if (detailTab === 'alternatives') return renderAlternativesDetail(step);
      renderStepDetail(step);
    }
    function clearReplacement() {
      activeReplacement = null;
      detailTab = 'step';
      renderAll();
    }
    function renderAll() {
      renderSummary();
      renderLayoutControls();
      renderViewPicker();
      renderStepIndex();
      renderStats();
      renderDetailTabs();
      renderRoute();
      renderDetail();
    }
    window.addEventListener('resize', () => renderRoute());
    renderAll();
  </script>
</body>
</html>
"""
    return html_doc.replace("__TITLE__", title).replace("__DATA__", data)


class _RouteForestCompiler:
    def __init__(self, blackboard: dict[str, Any], *, run_dir: str | Path | None = None) -> None:
        self.blackboard = dict(blackboard or {})
        self.run_dir = str(run_dir or "")
        self.evidence = dict(self.blackboard.get("literature_evidence") or {})
        self.nodes: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.branches: list[dict[str, Any]] = []
        self._branch_ids: set[str] = set()
        self._consensus_branch_ids: dict[str, str] = {}

    def finish(self) -> dict[str, Any]:
        modules = self._modules()
        relationships = self._branch_relationships()
        target = self._target()
        route_consensus = self._route_consensus_view()
        route_consensus_graph = self._route_consensus_graph_view()
        primary_selection = self._primary_selection()
        synthesis_class_counts: dict[str, int] = {}
        for branch in self.branches:
            synthesis_class = str(branch.get("synthesis_class") or "unspecified")
            synthesis_class_counts[synthesis_class] = synthesis_class_counts.get(synthesis_class, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": str(self.blackboard.get("case_id") or ""),
            "run_dir": self.run_dir,
            "target": target,
            "counts": {
                "branches": len(self.branches),
                "nodes": len(self.nodes),
                "steps": len(self.steps),
                "visual_chains": len(self.evidence.get("visual_chains") or []),
                "process_evidence_rows": len(self.evidence.get("process_evidence_rows") or []),
                "exact_rows": len(self.evidence.get("exact_rows") or []),
                "broad_templates": len(self.blackboard.get("broad_transform_templates") or []),
                "proposals": len(self.blackboard.get("retrosynthetic_proposals") or []),
                "route_consensus_proposals": len(route_consensus.get("proposals") or []),
                "route_consensus_rejected_candidates": int(
                    (route_consensus.get("source_summary") or {}).get("rejected_count") or 0
                ),
                "route_consensus_graph_routes": int(route_consensus_graph.get("route_count") or 0),
                "route_consensus_graph_steps": int(route_consensus_graph.get("step_count") or 0),
                "semisynthesis_anchors": len(self.blackboard.get("semisynthesis_anchors") or []),
                "scout_attempts": len(self.evidence.get("scout_attempts") or []),
                "relationships": len(relationships),
                "synthesis_classes": synthesis_class_counts,
            },
            "primary_branch_id": str(primary_selection.get("primary_branch_id") or ""),
            "primary_selection": primary_selection,
            "branches": self.branches,
            "nodes": sorted(self.nodes.values(), key=lambda row: str(row.get("node_id") or "")),
            "steps": sorted(self.steps.values(), key=lambda row: str(row.get("step_id") or "")),
            "modules": modules,
            "relationships": relationships,
            "route_consensus": route_consensus,
            "route_consensus_graph": route_consensus_graph,
            "evidence_index": self._evidence_index(),
            "run_trace": self._run_trace(),
            "design_notes": [
                "This is a read-only projection of explored blackboard branches.",
                "Clicking a module only switches or highlights explored alternatives; it does not run new planning.",
                "Named or visual-inferred nodes may intentionally omit SMILES when exact structure recovery was not reliable.",
                "Solved stitched branches are rebuilt only from revalidated proof inputs as stock-to-frontier-to-target DAGs.",
                "Visual, process, and consensus branches remain independent advisory alternatives.",
                "Route consensus branches are advisory disconnections, never solved or executable routes.",
                "Route consensus graph branches assemble frontier expansions but remain advisory and non-executable.",
                "Codex role channels are displayed separately but share one correlated support group.",
            ],
        }

    def add_direct_verified_route_branch(self) -> None:
        route_result = self._best_direct_route_result()
        if not route_result:
            return
        route = dict(route_result.get("route") or {})
        route_steps = [dict(row) for row in route.get("steps") or [] if isinstance(row, dict)]
        if not route_steps:
            return
        branch_id = "branch:direct_verified_chemenzy_route"
        target = self._target()
        target_smiles = str(target.get("smiles") or "")
        source_refs = _dedupe(
            [
                str(route_result.get("source_ref") or "ChemEnzy route verifier"),
                str(route_result.get("artifact_path") or ""),
            ]
        )
        rendered_steps = _forward_synthesis_step_order(
            route_steps,
            target_smiles=target_smiles,
        )
        if not rendered_steps:
            return
        step_ids: list[str] = []
        for index, row in enumerate(rendered_steps, start=1):
            product = _route_step_product(row)
            reactants = _route_step_reactants(row)
            if not product and not reactants:
                continue
            product_label = self._route_smiles_label(product, role="product", target_smiles=target_smiles)
            from_nodes = [
                self._add_node(
                    self._route_smiles_label(smiles, role=f"precursor {idx}", target_smiles=target_smiles),
                    role="verified_route_precursor",
                    smiles=smiles if _looks_like_smiles(smiles) else "",
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=source_refs,
                    missing=[],
                )
                for idx, smiles in enumerate(reactants, start=1)
                if str(smiles or "").strip()
            ]
            to_nodes = [
                self._add_node(
                    product_label,
                    role="target" if _same_molecule(product, target_smiles) else "verified_route_intermediate",
                    smiles=product if _looks_like_smiles(product) else "",
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=source_refs,
                    missing=[] if product else ["product missing from ChemEnzy step"],
                )
            ]
            label = _clean_label(
                row.get("reaction_type")
                or (row.get("reaction_interpretation") or {}).get("reaction_class")
                or f"ChemEnzy verified step {index}"
            )
            if label.lower() in {"template", "reaction", "step"}:
                label = f"ChemEnzy verified step {index}"
            step_ids.append(
                self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=from_nodes,
                    to_nodes=to_nodes,
                    module_key=_module_key_for_text(
                        " ".join(
                            [
                                label,
                                str((row.get("reaction_interpretation") or {}).get("forward_summary") or ""),
                                str(row.get("reaction_smiles") or ""),
                            ]
                        )
                    ),
                    module_label=_module_label_for_key(_module_key_for_text(label)),
                    confidence=_route_step_confidence(row),
                    exactness="model_hypothesis",
                    source_refs=source_refs,
                    origin="direct_verified_chemenzy_route",
                    summary=str((row.get("reaction_interpretation") or {}).get("forward_summary") or row.get("source") or "ChemEnzy route verifier accepted this step in a solved parent route."),
                    conditions=_conditions_from_row(row),
                    missing=_dedupe(
                        [
                            "computational/template route, not an exact literature row",
                            *[str(x) for x in ((row.get("reaction_interpretation") or {}).get("atom_change") or {}).get("notes") or []],
                        ]
                    )[:8],
                )
            )
        if not step_ids:
            return
        self._add_branch(
            branch_id=branch_id,
            title=f"Direct verified route: {target.get('name') or 'target'}",
            kind="direct_verified_route",
            recommendation="verified route",
            confidence="high",
            summary="Route verifier accepted this ChemEnzy route as a deterministic parent-route proof for the target.",
            step_ids=step_ids,
            source_refs=source_refs,
            missing=["Not a literature exact-row route", "Conditions may be template-level unless separately predicted"],
            classification_records=[
                *[dict(row) for row in route_result.get("classification_records") or [] if isinstance(row, dict)],
                route,
            ],
        )

    def add_stitched_verified_route_branch(self) -> None:
        """Render the revalidated stock-to-terminal-to-target proof DAG.

        The display is rebuilt exclusively from the stitched proof's immutable
        ``proof_inputs``.  Top-level route summaries, display bindings, and
        other blackboard branches are deliberately outside this trust boundary.
        """
        target = self._target()
        target_smiles = str(target.get("smiles") or "").strip()
        projection = _revalidated_stitched_proof_projection(
            self.blackboard.get("parent_route_proof"),
            expected_target_smiles=target_smiles,
        )
        if not projection:
            return

        branch_id = "branch:stitched_verified_parent_route"
        literature_frontiers = [
            str(smiles)
            for smiles in projection.get("literature_frontier_smiles") or []
            if str(smiles or "").strip()
        ]
        literature_frontier_keys = {
            _canonical_molecule_smiles(smiles) for smiles in literature_frontiers
        }
        stock_terminals = {
            _canonical_molecule_smiles(smiles)
            for smiles in projection.get("stock_terminal_smiles") or []
            if _canonical_molecule_smiles(smiles)
        }
        subgoal_source_refs = ["deterministic:chemenzy-route-verifier"]
        literature_source_refs = _dedupe(
            [
                str(projection.get("literature_source_ref") or ""),
                *[
                    str(ref)
                    for row in projection.get("literature_steps") or []
                    for ref in [
                        row.get("source_ref"),
                        *(row.get("evidence_refs") or []),
                    ]
                    if str(ref or "").strip()
                ],
            ]
        )
        step_ids: list[str] = []
        subgoal_step_ids: list[str] = []
        subgoal_segments: list[dict[str, Any]] = []
        literature_step_ids: list[str] = []
        stock_terminal_node_ids: list[str] = []
        literature_terminal_node_ids: list[str] = []

        for segment_index, segment in enumerate(projection.get("subgoal_segments") or [], start=1):
            segment_frontier = str(segment.get("frontier_smiles") or "")
            segment_step_ids: list[str] = []
            for route_step_index, row in enumerate(segment.get("steps") or [], start=1):
                product = _route_step_product(row)
                reactants = _route_step_reactants(row)
                from_nodes: list[str] = []
                for precursor_index, smiles in enumerate(reactants, start=1):
                    canonical = _canonical_molecule_smiles(smiles)
                    is_stock = canonical in stock_terminals
                    node_id = self._add_node(
                        self._route_smiles_label(
                            smiles,
                            role=f"stock precursor {precursor_index}" if is_stock else f"route precursor {precursor_index}",
                            target_smiles=target_smiles,
                        ),
                        role="stock_terminal" if is_stock else "stitched_route_intermediate",
                        smiles=smiles,
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=subgoal_source_refs,
                        missing=[],
                    )
                    from_nodes.append(node_id)
                    if is_stock:
                        stock_terminal_node_ids.append(node_id)
                product_is_terminal = _same_molecule(product, segment_frontier)
                product_node = self._add_node(
                    f"literature terminal {segment_index}"
                    if product_is_terminal
                    else self._route_smiles_label(product, role="route intermediate", target_smiles=target_smiles),
                    role="literature_terminal" if product_is_terminal else "stitched_route_intermediate",
                    smiles=product,
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=subgoal_source_refs,
                    missing=[],
                )
                if product_is_terminal:
                    literature_terminal_node_ids.append(product_node)
                label = _clean_label(
                    row.get("reaction_type")
                    or (row.get("reaction_interpretation") or {}).get("reaction_class")
                    or f"Verified stock closure {segment_index}.{route_step_index}"
                )
                step_id = self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=from_nodes,
                    to_nodes=[product_node],
                    module_key=f"stitched_stock_closure:{segment_index:02d}:{route_step_index:02d}",
                    module_label=f"Verified stock closure {segment_index}.{route_step_index}",
                    confidence=_route_step_confidence(row),
                    exactness="model_hypothesis",
                    source_refs=subgoal_source_refs,
                    origin="stitched_verified_subgoal_route",
                    summary=str(
                        (row.get("reaction_interpretation") or {}).get("forward_summary")
                        or "The deterministic route verifier accepted this stock-closure step."
                    ),
                    conditions=_conditions_from_row(row),
                    missing=["Computational route step; not an exact literature row"],
                )
                step_ids.append(step_id)
                subgoal_step_ids.append(step_id)
                segment_step_ids.append(step_id)
            subgoal_segments.append(
                {
                    "segment_id": f"verified_stock_closure_{segment_index}",
                    "frontier_smiles": segment_frontier,
                    "step_ids": segment_step_ids,
                    "status": "deterministically_verified",
                }
            )

        for index, row in enumerate(projection.get("literature_steps") or [], start=1):
            product = _route_step_product(row)
            reactants = _route_step_reactants(row)
            row_source_refs = _dedupe(
                [
                    str(row.get("source_ref") or ""),
                    *[str(ref) for ref in row.get("evidence_refs") or []],
                ]
            )
            from_nodes: list[str] = []
            for precursor_index, smiles in enumerate(reactants, start=1):
                is_terminal = _canonical_molecule_smiles(smiles) in literature_frontier_keys
                node_id = self._add_node(
                    "literature terminal"
                    if is_terminal
                    else self._route_smiles_label(
                        smiles,
                        role=f"literature precursor {precursor_index}",
                        target_smiles=target_smiles,
                    ),
                    role="literature_terminal" if is_terminal else "literature_intermediate",
                    smiles=smiles,
                    exactness="exact_literature_row",
                    confidence="high",
                    source_refs=row_source_refs,
                    missing=[],
                )
                from_nodes.append(node_id)
                if is_terminal:
                    literature_terminal_node_ids.append(node_id)
            product_is_target = _same_molecule(product, target_smiles)
            product_node = self._add_node(
                str(target.get("name") or "target")
                if product_is_target
                else self._route_smiles_label(product, role="literature intermediate", target_smiles=target_smiles),
                role="target" if product_is_target else "literature_intermediate",
                smiles=product,
                exactness="exact_literature_row",
                confidence="high",
                source_refs=row_source_refs,
                missing=[],
            )
            label = _clean_label(
                row.get("reaction_class")
                or row.get("step_label")
                or row.get("step_id")
                or f"Exact literature step {index}"
            )
            module_key = _module_key_for_text(
                " ".join([label, str(row.get("reaction_smiles") or "")])
            )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=from_nodes,
                to_nodes=[product_node],
                module_key=module_key,
                module_label=_module_label_for_key(module_key),
                confidence="high",
                exactness="exact_literature_row",
                source_refs=row_source_refs,
                origin="stitched_verified_literature_chain",
                summary="This exact literature edge was revalidated against its source-detail evidence binding.",
                conditions=_conditions_from_row(row),
                missing=[],
            )
            step_ids.append(step_id)
            literature_step_ids.append(step_id)

        if (
            not subgoal_step_ids
            or not literature_step_ids
            or len(set(literature_terminal_node_ids)) != len(literature_frontier_keys)
        ):
            return
        self._add_branch(
            branch_id=branch_id,
            title=f"Stitched verified route: {target.get('name') or 'target'}",
            kind="stitched_verified_route",
            recommendation="deterministically verified stitched route",
            confidence="high",
            summary=(
                "A single revalidated synthesis DAG connects every stock terminal through the "
                "verified subgoal closure and strict source-detail literature chain to the target."
            ),
            step_ids=step_ids,
            source_refs=_dedupe([*subgoal_source_refs, *literature_source_refs]),
            missing=[
                "Stock-closure steps are computational unless independently replaced by exact literature rows"
            ],
            classification_records=[{"synthesis_class": "semisynthesis"}],
        )
        branch = self.branches[-1]
        branch["route_direction"] = "stock_to_literature_terminal_to_target"
        branch["stock_terminal_node_ids"] = _dedupe(stock_terminal_node_ids)
        branch["literature_terminal_node_ids"] = _dedupe(literature_terminal_node_ids)
        branch["segments"] = [
            *subgoal_segments,
            {
                "segment_id": "strict_literature_chain",
                "step_ids": literature_step_ids,
                "status": "source_detail_exact",
            },
        ]

    def add_subgoal_verified_route_branches(self) -> None:
        for index, record in enumerate(self._subgoal_route_records(), start=1):
            route = dict(record.get("route") or {})
            route_steps = [dict(row) for row in route.get("steps") or [] if isinstance(row, dict)]
            subgoal_name = _clean_label(record.get("name") or f"subgoal {index}")
            target_smiles = str(record.get("target_smiles") or "").strip()
            step_source_refs = _dedupe(
                [
                    "route_expansion_subgoal_search_result",
                    str(record.get("search_path") or ""),
                    str(record.get("raw_path") or ""),
                ]
            )
            branch_source_refs = _dedupe(
                [
                    *step_source_refs,
                    *[str(x) for x in record.get("evidence_refs") or [] if str(x).strip()],
                ]
            )
            branch_id = f"branch:subgoal_verified_route:{_slug(subgoal_name)}:{index}"

            def route_label(value: str, *, role: str) -> str:
                text = str(value or "").strip()
                if target_smiles and _same_text(text, target_smiles):
                    return f"{subgoal_name} terminal"
                if not text:
                    return role
                if _looks_like_smiles(text):
                    return _compact_smiles_label(text)
                return _clean_label(text)

            step_ids: list[str] = []
            rendered_steps = list(reversed(route_steps))
            for step_index, row in enumerate(rendered_steps, start=1):
                product = str(row.get("product") or "").strip()
                reactants = _route_step_reactants(row)
                if not product and not reactants:
                    continue
                from_nodes = [
                    self._add_node(
                        route_label(smiles, role=f"subgoal precursor {idx}"),
                        role="subgoal_route_precursor",
                        smiles=smiles if _looks_like_smiles(smiles) else "",
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=step_source_refs,
                        missing=[],
                    )
                    for idx, smiles in enumerate(reactants, start=1)
                    if str(smiles or "").strip()
                ]
                to_nodes = [
                    self._add_node(
                        route_label(product, role="subgoal product"),
                        role="subgoal_literature_terminal" if target_smiles and _same_text(product, target_smiles) else "subgoal_route_intermediate",
                        smiles=product if _looks_like_smiles(product) else "",
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=step_source_refs,
                        missing=[] if product else ["product missing from ChemEnzy subgoal step"],
                    )
                ]
                label = _clean_label(
                    row.get("reaction_type")
                    or (row.get("reaction_interpretation") or {}).get("reaction_class")
                    or f"ChemEnzy 子目标闭合 step {step_index}"
                )
                if label.lower() in {"template", "reaction", "step", "chemenzyretroplanner"}:
                    label = f"ChemEnzy 子目标闭合 step {step_index}"
                step_ids.append(
                    self._add_step(
                        branch_id=branch_id,
                        label=label,
                        from_nodes=from_nodes,
                        to_nodes=to_nodes,
                        module_key=f"subgoal_stock_closure:{step_index:02d}",
                        module_label=f"ChemEnzy 子目标闭合 {step_index}",
                        confidence=_route_step_confidence(row),
                        exactness="model_hypothesis",
                        source_refs=step_source_refs,
                        origin="subgoal_verified_chemenzy_route",
                        summary=str(
                            (row.get("reaction_interpretation") or {}).get("forward_summary")
                            or row.get("source")
                            or "ChemEnzy route expansion closed the upstream stock-to-literature-terminal subgoal."
                        ),
                        conditions=_conditions_from_row(row),
                        missing=_dedupe(
                            [
                                "子目标闭合路线：支持 stitched final route，但本身不是最终目标路线",
                                "计算/模板路线，不是 exact literature row",
                                *[str(x) for x in ((row.get("reaction_interpretation") or {}).get("atom_change") or {}).get("notes") or []],
                            ]
                        )[:8],
                    )
                )
            if not step_ids and target_smiles:
                target_node = self._add_node(
                    f"{subgoal_name} terminal",
                    role="subgoal_literature_terminal",
                    smiles=target_smiles,
                    exactness="model_hypothesis",
                    confidence="high" if record.get("accepted") else "medium",
                    source_refs=branch_source_refs,
                    missing=[],
                )
                step_ids.append(
                    self._add_step(
                        branch_id=branch_id,
                        label=f"ChemEnzy 子目标闭合：{subgoal_name}",
                        from_nodes=[
                            self._add_node(
                                "ChemEnzy stock closure route pool",
                                role="subgoal_route_pool",
                                exactness="model_hypothesis",
                                confidence="medium",
                                source_refs=step_source_refs,
                                missing=["route forest 编译器未加载到 raw ChemEnzy steps"],
                            )
                        ],
                        to_nodes=[target_node],
                        module_key="subgoal_stock_closure",
                        module_label=_module_label_for_key("subgoal_stock_closure"),
                        confidence="high" if record.get("accepted") else "medium",
                        exactness="model_hypothesis",
                        source_refs=step_source_refs,
                        origin="subgoal_verified_chemenzy_route",
                        summary="ChemEnzy route expansion 接受了 stock-to-literature-terminal 子目标闭合。",
                        conditions=[],
                        missing=["只有路线级子目标证明；未加载逐步 raw route"],
                    )
                )
            if not step_ids:
                continue
            verifier_reasons = [str(x) for x in record.get("reasons") or [] if str(x).strip()]
            route_rank = record.get("route_rank")
            self._add_branch(
                branch_id=branch_id,
                title=f"ChemEnzy 子目标闭合：{subgoal_name}",
                kind="subgoal_verified_route",
                recommendation="子目标闭合审计",
                confidence="high" if record.get("accepted") else "medium",
                summary=(
                    "ChemEnzy 已闭合 stitched parent proof 使用的上游 stock-to-literature-terminal 片段。"
                    "它作为独立 advisory 分支展示；只有 proof inputs 重验通过时才会另建拼接验证路线。"
                ),
                step_ids=step_ids,
                source_refs=branch_source_refs,
                missing=_dedupe(
                    [
                        "只支持上游 child target / literature terminal 片段",
                        "不是 literature exact-row 路线",
                        f"best route rank: {route_rank}" if route_rank is not None else "",
                        *[f"verifier noted route-pool issue: {reason}" for reason in verifier_reasons[:4]],
                    ]
                ),
                classification_records=[record, route],
            )

    def add_diagnostic_failure_branch_if_empty(self) -> None:
        if self.branches:
            return
        failures = [dict(row) for row in self.blackboard.get("route_failures") or [] if isinstance(row, dict)]
        diagnostics = [row for row in self._guided_result_artifacts() if isinstance(row.get("chemenzy_runtime_diagnostic"), dict)]
        if not failures and not diagnostics:
            self._add_unclosed_exploration_branch_if_empty()
            return
        reasons = _dedupe(
            [
                *[str(row.get("reason") or row.get("failure_class") or "") for row in failures],
                *[
                    str(reason)
                    for row in diagnostics
                    for reason in (row.get("chemenzy_runtime_diagnostic") or {}).get("reasons") or []
                ],
            ]
        )
        branch_id = "branch:diagnostic_unresolved_route"
        step_id = self._add_step(
            branch_id=branch_id,
            label="No accepted route produced",
            from_nodes=[
                self._add_node(
                    "ChemEnzy / planner diagnostic",
                    role="diagnostic_source",
                    exactness="failed_or_unresolved",
                    confidence="failed",
                    source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
                    missing=reasons or ["route unresolved"],
                )
            ],
            to_nodes=[self._target_node()],
            module_key="diagnostic_failure",
            module_label="Diagnostic failure",
            confidence="failed",
            exactness="failed_or_unresolved",
            source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
            origin="route_failure_diagnostic",
            summary="The run reached a route-search diagnostic instead of a displayable synthesis route.",
            missing=reasons or ["no displayable route branch"],
        )
        self._add_branch(
            branch_id=branch_id,
            title=f"Unresolved diagnostic: {self._target().get('name') or 'target'}",
            kind="diagnostic_failure",
            recommendation="needs rerun or repair",
            confidence="failed",
            summary="No route branch was available; this panel preserves the failure reason instead of showing a blank route.",
            step_ids=[step_id],
            source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
            missing=reasons or ["no displayable route branch"],
        )

    def _primary_selection(self) -> dict[str, Any]:
        """Select a real compiled branch without manufacturing route chemistry."""
        if not self.branches:
            return {
                "schema_version": "route_forest_primary_selection.v1",
                "primary_branch_id": "",
                "status": "unavailable",
                "proof_level": "none",
                "advisory_only": True,
                "reasons": ["no_compiled_branch"],
            }
        priority = {
            "stitched_verified_route": 80,
            "direct_verified_route": 70,
            "exact_literature": 60,
            "subgoal_verified_route": 50,
            "route_consensus_graph": 45,
            "process_evidence": 40,
            "visual_chain": 35,
            "route_consensus": 30,
            "literature_candidate": 25,
            "retrosynthetic_proposal": 20,
            "broad_template": 10,
            "diagnostic_failure": 0,
        }
        selected = max(
            self.branches,
            key=lambda row: (
                priority.get(str(row.get("kind") or ""), -1),
                CONFIDENCE_RANK.get(str(row.get("confidence") or ""), 0),
                len(row.get("step_ids") or []),
                str(row.get("branch_id") or ""),
            ),
        )
        selected_id = str(selected.get("branch_id") or "")
        for branch in self.branches:
            branch["is_primary"] = bool(selected_id and branch.get("branch_id") == selected_id)
        kind = str(selected.get("kind") or "")
        if kind in {"stitched_verified_route", "direct_verified_route"}:
            status = "deterministically_verified"
            proof_level = "parent_route_proof"
            advisory_only = False
        elif kind == "exact_literature":
            status = "evidence_backed"
            proof_level = "literature_rows"
            advisory_only = True
        elif kind == "diagnostic_failure":
            status = "diagnostic"
            proof_level = "none"
            advisory_only = True
        else:
            status = "advisory"
            proof_level = "route_hint"
            advisory_only = True
        return {
            "schema_version": "route_forest_primary_selection.v1",
            "primary_branch_id": selected_id,
            "status": status,
            "proof_level": proof_level,
            "advisory_only": advisory_only,
            "synthesis_class": str(selected.get("synthesis_class") or "unspecified"),
            "reasons": [
                f"selected_from_compiled_branch_kind:{kind or 'unknown'}",
                "no_target_name_route_injection",
            ],
        }

    def _add_unclosed_exploration_branch_if_empty(self) -> None:
        if self.branches:
            return
        candidates = [dict(row) for row in self.evidence.get("source_candidates") or [] if isinstance(row, dict)]
        pdf_rows = [dict(row) for row in self.evidence.get("pdf_structure_evidence") or [] if isinstance(row, dict)]
        visual_rows = [dict(row) for row in self.evidence.get("visual_chains") or [] if isinstance(row, dict)]
        exact_rows = [dict(row) for row in self.evidence.get("exact_rows") or [] if isinstance(row, dict)]
        actions = [dict(row) for row in self.blackboard.get("action_history") or [] if isinstance(row, dict)]
        if not any((candidates, pdf_rows, visual_rows, exact_rows, actions, self.blackboard.get("current_belief"))):
            return
        source_refs = _dedupe(
            [
                str(row.get("source_ref") or row.get("doi") or row.get("title") or row.get("url") or "")
                for row in candidates[:8]
            ]
        )
        status_bits = [
            f"source candidates: {len(candidates)}" if candidates else "",
            f"PDF evidence rows: {len(pdf_rows)}" if pdf_rows else "",
            f"visual chains: {len(visual_rows)}" if visual_rows else "",
            f"exact rows: {len(exact_rows)}" if exact_rows else "",
            f"actions recorded: {len(actions)}" if actions else "",
        ]
        missing = _dedupe(
            [
                "no displayable route branch was compiled",
                "no deterministic parent-route proof",
                *[bit for bit in status_bits if bit],
            ]
        )
        branch_id = "branch:unclosed_exploration_state"
        step_id = self._add_step(
            branch_id=branch_id,
            label="Exploration recorded, no route branch",
            from_nodes=[
                self._add_node(
                    "Blackboard exploration state",
                    role="diagnostic_source",
                    exactness="failed_or_unresolved",
                    confidence="low",
                    source_refs=source_refs,
                    missing=missing,
                )
            ],
            to_nodes=[self._target_node()],
            module_key="diagnostic_failure",
            module_label="Diagnostic / incomplete exploration",
            confidence="low",
            exactness="failed_or_unresolved",
            source_refs=source_refs,
            origin="unclosed_blackboard_exploration",
            summary="The run recorded planning or evidence activity, but no exact, visual, process, proposal, template, or verified route branch was available.",
            missing=missing,
        )
        self._add_branch(
            branch_id=branch_id,
            title=f"Exploration incomplete: {self._target().get('name') or 'target'}",
            kind="diagnostic_failure",
            recommendation="needs more evidence or rerun",
            confidence="low",
            summary="The route forest preserves the non-empty blackboard state instead of showing a blank route.",
            step_ids=[step_id],
            source_refs=source_refs,
            missing=missing,
        )

    def add_visual_branches(self, *, limit: int) -> None:
        chains = [row for row in self.evidence.get("visual_chains") or [] if isinstance(row, dict)]
        for index, chain in enumerate(chains[: max(0, limit)], start=1):
            source_ref = str(chain.get("source_ref") or chain.get("source_title") or f"visual:{index}")
            title = str(chain.get("source_title") or source_ref or f"Visual chain {index}")
            branch_id = self._unique_branch_id(f"branch:visual:{_slug(source_ref or title)}:{index}")
            chain_steps = chain.get("steps") or chain.get("candidate_steps") or []
            step_ids: list[str] = []
            if chain_steps:
                for row_index, row in enumerate([x for x in chain_steps if isinstance(x, dict)], start=1):
                    step_ids.append(self._visual_step(branch_id, chain, row, row_index))
            else:
                step_ids.append(
                    self._add_step(
                        branch_id=branch_id,
                        label="Visual extraction produced no route step",
                        from_nodes=[
                            self._add_node(
                                title,
                                role="source_placeholder",
                                exactness="failed_or_unresolved",
                                confidence="failed",
                                source_refs=[source_ref],
                            )
                        ],
                        to_nodes=[],
                        module_key="visual_failed_or_empty",
                        module_label="视觉链失败或为空",
                        confidence="failed",
                        exactness="failed_or_unresolved",
                        source_refs=[source_ref],
                        origin="visual_chain",
                        summary=str(chain.get("rejection_reason") or "No candidate step was accepted from this source."),
                        missing=["No displayable route step"],
                    )
                )
            self._add_branch(
                branch_id=branch_id,
                title=f"文献图像分支：{title}",
                kind=(
                    "diagnostic_failure"
                    if chain.get("accepted") is False or not chain_steps
                    else "visual_chain"
                ),
                recommendation="支持/备选",
                confidence="low" if chain_steps else "failed",
                summary="Steps inferred from rendered literature figures. These are useful route hints, not exact proof.",
                step_ids=step_ids,
                source_refs=[source_ref],
                missing=["image-derived structures may be incomplete", "stereochemistry may be partial"],
                classification_records=[chain],
            )

    def add_process_evidence_branches(self, *, limit: int = 8) -> None:
        rows = [row for row in self.evidence.get("process_evidence_rows") or [] if isinstance(row, dict)]
        for index, row in enumerate(rows[: max(0, limit)], start=1):
            endpoints = _labels_from_any(row.get("endpoint_labels")) or ["process endpoint"]
            substrates = _labels_from_any(row.get("substrate_or_feedstock_labels")) or ["process substrate/feedstock"]
            process_labels = _labels_from_any(row.get("biocatalyst_or_process_labels")) or [
                str(row.get("process_type") or "process")
            ]
            source_refs = _dedupe(
                [
                    str(row.get("source_ref") or ""),
                    str(row.get("source_title") or ""),
                    *[str(item) for item in row.get("evidence_refs") or []],
                ]
            )
            endpoint_text = " / ".join(endpoints[:3])
            substrate_text = " / ".join(substrates[:4])
            process_text = " / ".join(process_labels[:4])
            label = f"{substrate_text} via {process_text} to {endpoint_text}"
            branch_id = self._unique_branch_id(f"branch:process:{_slug(str(row.get('row_id') or index))}")
            module_key = _module_key_for_text(" ".join([label, str(row.get("process_type") or "")]))
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=[
                    self._add_node(
                        label=substrate,
                        role="process_substrate_or_feedstock",
                        exactness="name_only",
                        confidence=str(row.get("confidence") or "medium"),
                        source_refs=source_refs,
                        missing=["structure may be class/name-only in process evidence"],
                    )
                    for substrate in substrates[:6]
                ],
                to_nodes=[
                    self._add_node(
                        label=endpoint,
                        role="process_endpoint",
                        exactness="named_literature",
                        confidence=str(row.get("confidence") or "medium"),
                        source_refs=source_refs,
                    )
                    for endpoint in endpoints[:4]
                ],
                module_key=module_key,
                module_label=_module_label_for_key(module_key),
                confidence=str(row.get("confidence") or "medium"),
                exactness="named_literature",
                source_refs=source_refs,
                origin="process_evidence",
                summary=str(row.get("summary") or label),
                conditions=_conditions_from_row(row),
                missing=_dedupe(
                    [
                        "process evidence is an advisory route anchor, not an exact reaction row",
                        *[str(item) for item in row.get("risk_flags") or []],
                    ]
                ),
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"文献过程锚点：{endpoint_text}",
                kind="process_evidence",
                recommendation="文献锚点",
                confidence=str(row.get("confidence") or "medium"),
                summary=str(row.get("summary") or label),
                step_ids=[step_id],
                source_refs=source_refs,
                missing=[
                    "not a deterministic stock-closed route",
                    "not exact reaction SMILES",
                    "requires objective-specific verification",
                ],
                classification_records=[row],
            )

    def add_route_consensus_branches(self, *, limit: int) -> None:
        """Project canonical consensus proposals without promoting them to routes.

        A consensus proposal is one advisory reaction edge.  Its direct
        literature/evidence references stay on that edge; the consensus JSON
        artifact itself is retained separately as route-level provenance.
        """
        consensus = self._route_consensus_payload()
        proposals = [
            row
            for row in consensus.get("proposals") or []
            if isinstance(row, dict)
            and str(row.get("schema_version") or "") == "route_consensus_proposal.v1"
        ]
        proposals.sort(
            key=lambda row: (
                int(row.get("rank") or 1_000_000),
                -float(row.get("rank_score") or 0.0),
                str(row.get("consensus_id") or ""),
            )
        )
        target = self._target()
        target_smiles = str(target.get("smiles") or "").strip()
        consensus_target_smiles = str(consensus.get("target_smiles") or "").strip()
        route_level_refs = self._route_consensus_route_refs()

        for index, proposal in enumerate(proposals[: max(0, limit)], start=1):
            product_smiles = str(proposal.get("product_smiles") or "").strip()
            precursor_smiles = _consensus_precursor_smiles(proposal)
            if not product_smiles or not precursor_smiles:
                continue
            target_matches = (
                not target_smiles
                or _same_molecule(product_smiles, target_smiles)
                or bool(consensus_target_smiles and _same_text(product_smiles, consensus_target_smiles))
            )
            if not target_matches:
                # A canonical route_consensus.v1 producer already quarantines
                # this case.  Do not reconnect a malformed payload to target.
                continue

            consensus_id = str(proposal.get("consensus_id") or f"proposal-{index}")
            branch_id = f"branch:route_consensus:{_slug(consensus_id)}"
            direct_refs = _consensus_direct_source_refs(proposal)
            support_records = _consensus_support_records(proposal)
            support_groups = _consensus_independent_support_groups(proposal, support_records)
            conflicts = _consensus_conflicts(proposal)
            evidence_level = str(proposal.get("evidence_level") or "model_only")
            confidence = str(proposal.get("confidence") or "low")
            reaction_family = _clean_label(proposal.get("reaction_family") or "unspecified transformation")
            exactness = "named_literature" if evidence_level == "literature_exact" else "model_hypothesis"
            missing = _dedupe(
                [
                    "advisory consensus only; not solved or executable",
                    "deterministic parent-route proof is still required",
                    *[str(item) for item in proposal.get("limitations") or []],
                    *[
                        f"required validation: {item}"
                        for item in proposal.get("required_validation") or []
                        if str(item).strip()
                    ],
                    "condition conflict requires review" if conflicts else "",
                ]
            )
            precursor_nodes = [
                self._add_node(
                    f"consensus precursor {precursor_index}",
                    role="consensus_precursor",
                    smiles=smiles,
                    exactness=exactness,
                    confidence=confidence,
                    source_refs=direct_refs,
                    missing=missing,
                )
                for precursor_index, smiles in enumerate(precursor_smiles, start=1)
            ]
            product_node = self._add_node(
                str(target.get("name") or "consensus product"),
                role="target" if target_smiles else "consensus_product",
                smiles=product_smiles,
                exactness=exactness,
                confidence=confidence,
                source_refs=direct_refs,
                missing=missing,
            )
            step_id = self._add_step(
                branch_id=branch_id,
                label=reaction_family,
                from_nodes=precursor_nodes,
                to_nodes=[product_node],
                module_key=_module_key_for_text(reaction_family),
                module_label=_module_label_for_key(_module_key_for_text(reaction_family)),
                confidence=confidence,
                exactness=exactness,
                source_refs=direct_refs,
                origin="route_consensus",
                summary="Advisory multi-source retrosynthetic disconnection; it is not a parent-route proof.",
                conditions=_conditions_from_row(proposal),
                missing=missing,
            )
            consensus_metadata = {
                "consensus_id": consensus_id,
                "consensus_status": str(proposal.get("status") or "model_hypothesis"),
                "evidence_level": evidence_level,
                "source_channels": _dedupe([str(item) for item in proposal.get("source_channels") or []]),
                "support_records": support_records,
                "support_count": len(support_records) or int(proposal.get("support_count") or 0),
                "independent_support_groups": support_groups,
                "independent_source_count": len(support_groups),
                "codex_roles_correlated": any(group == "codex_model" for group in support_groups),
                "condition_support": [
                    dict(row) for row in proposal.get("condition_support") or [] if isinstance(row, dict)
                ],
                "conflicts": conflicts,
                "rank": int(proposal.get("rank") or index),
                "rank_score": float(proposal.get("rank_score") or 0.0),
                "advisory_only": True,
                "solved": False,
                "executable": False,
                "not_parent_route_proof": True,
            }
            self.steps[step_id].update(consensus_metadata)
            if conflicts:
                self.steps[step_id]["condition_status"] = "conflicting"

            self._add_branch(
                branch_id=branch_id,
                title=f"Consensus #{int(proposal.get('rank') or index)}: {reaction_family}",
                kind="route_consensus",
                recommendation="advisory consensus",
                confidence=confidence,
                summary="Multi-source candidate edge. Independent support is counted by correlated support group, not by Codex role.",
                step_ids=[step_id],
                source_refs=direct_refs,
                missing=missing,
                classification_records=[proposal],
            )
            actual_branch = self.branches[-1]
            actual_branch.update(consensus_metadata)
            actual_branch["route_level_source_refs"] = route_level_refs
            self._consensus_branch_ids[consensus_id] = str(actual_branch.get("branch_id") or branch_id)

    def add_route_consensus_graph_branches(self, *, limit: int) -> None:
        graph = self._route_consensus_graph_payload()
        if not graph:
            return
        step_by_id = {
            str(row.get("step_id") or ""): dict(row)
            for row in graph.get("steps") or []
            if isinstance(row, dict) and str(row.get("step_id") or "")
        }
        conflict_by_id = {
            str(row.get("conflict_id") or ""): dict(row)
            for row in graph.get("conflicts") or []
            if isinstance(row, dict) and str(row.get("conflict_id") or "")
        }
        target_smiles = str(self._target().get("smiles") or "")
        routes = [dict(row) for row in graph.get("route_hypotheses") or [] if isinstance(row, dict)]
        routes.sort(key=lambda row: (-float(row.get("rank_score") or 0.0), str(row.get("route_id") or "")))
        for index, route in enumerate(routes[: max(0, int(limit))], start=1):
            graph_route_id = str(route.get("route_id") or f"route-{index}")
            branch_id = f"branch:route_consensus_graph:{_slug(graph_route_id)}"
            rendered_step_ids: list[str] = []
            graph_steps: list[dict[str, Any]] = []
            for graph_step_id in route.get("forward_step_ids") or []:
                graph_step = dict(step_by_id.get(str(graph_step_id)) or {})
                if not graph_step:
                    continue
                product = str(graph_step.get("product_smiles") or "")
                precursors = [str(value) for value in graph_step.get("precursor_smiles") or [] if str(value).strip()]
                if not product or not precursors:
                    continue
                direct_refs = _dedupe(
                    [
                        *[str(value) for value in graph_step.get("source_refs") or []],
                        *[str(value) for value in graph_step.get("evidence_refs") or []],
                    ]
                )
                from_nodes = [
                    self._add_node(
                        _compact_smiles_label(smiles),
                        role="consensus_graph_precursor",
                        smiles=smiles if _looks_like_smiles(smiles) else "",
                        exactness="model_hypothesis",
                        confidence=str(graph_step.get("confidence") or "low"),
                        source_refs=direct_refs,
                        missing=["advisory graph node; deterministic identity audit still required"],
                    )
                    for smiles in precursors
                ]
                to_nodes = [
                    self._add_node(
                        self._route_smiles_label(product, role="consensus graph product", target_smiles=target_smiles),
                        role="target" if target_smiles and _same_molecule(product, target_smiles) else "consensus_graph_intermediate",
                        smiles=product if _looks_like_smiles(product) else "",
                        exactness="model_hypothesis",
                        confidence=str(graph_step.get("confidence") or "low"),
                        source_refs=direct_refs,
                        missing=["advisory graph node; deterministic identity audit still required"],
                    )
                ]
                family = str(graph_step.get("reaction_family") or "multi-source disconnection")
                conditions = [
                    {"label": "candidate condition", "value": str(value)}
                    for value in graph_step.get("conditions") or []
                    if str(value).strip()
                ]
                display_step_id = self._add_step(
                    branch_id=branch_id,
                    label=family,
                    from_nodes=from_nodes,
                    to_nodes=to_nodes,
                    module_key=_module_key_for_text(family),
                    module_label=_module_label_for_key(_module_key_for_text(family)),
                    confidence=str(graph_step.get("confidence") or "low"),
                    exactness="model_hypothesis",
                    source_refs=direct_refs,
                    origin="route_consensus_graph",
                    summary="Advisory multi-step graph edge assembled from a frontier-specific consensus.",
                    conditions=conditions,
                    missing=_dedupe(
                        [
                            "advisory graph edge; not solved or executable",
                            "deterministic parent-route proof is required",
                            *[str(value) for value in graph_step.get("limitations") or []],
                            *[f"required validation: {value}" for value in graph_step.get("required_validation") or []],
                        ]
                    ),
                )
                self.steps[display_step_id].update(
                    {
                        "graph_step_id": str(graph_step_id),
                        "support_records": [
                            dict(row) for row in graph_step.get("source_records") or [] if isinstance(row, dict)
                        ],
                        "independent_support_groups": [
                            str(value) for value in graph_step.get("independent_support_groups") or []
                        ],
                        "conflicts": [
                            conflict_by_id[str(conflict_id)]
                            for conflict_id in graph_step.get("conflict_ids") or []
                            if str(conflict_id) in conflict_by_id
                        ],
                        "advisory_only": True,
                        "solved": False,
                        "executable": False,
                        "not_parent_route_proof": True,
                    }
                )
                rendered_step_ids.append(display_step_id)
                graph_steps.append(graph_step)
            if not rendered_step_ids:
                continue
            self._add_branch(
                branch_id=branch_id,
                title=f"Codex multi-step hypothesis #{index}",
                kind="route_consensus_graph",
                recommendation="advisory multi-step hypothesis",
                confidence=_confidence_from_score(float(route.get("rank_score") or 0.0)),
                summary="Frontier-specific Codex teams assembled these edges into one read-only forward route hypothesis.",
                step_ids=rendered_step_ids,
                source_refs=[],
                missing=[
                    "not a deterministic parent-route proof",
                    "frontier leaves may remain unexpanded",
                    "all reaction edges require forward and stock validation",
                ],
                classification_records=graph_steps,
            )
            branch = self.branches[-1]
            branch.update(
                {
                    "graph_route_id": graph_route_id,
                    "rank_score": float(route.get("rank_score") or 0.0),
                    "forward_dependencies": [
                        dict(row) for row in route.get("forward_dependencies") or [] if isinstance(row, dict)
                    ],
                    "frontier": [dict(row) for row in route.get("frontier") or [] if isinstance(row, dict)],
                    "conflict_ids": [str(value) for value in route.get("conflict_ids") or []],
                    "route_level_source_refs": self._route_consensus_graph_refs(),
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "not_parent_route_proof": True,
                }
            )

    def _route_consensus_graph_payload(self) -> dict[str, Any]:
        if self._codex_team_projection_reasons():
            return {}
        direct = self.blackboard.get("route_consensus_graph")
        if isinstance(direct, dict) and direct.get("schema_version") == "route_consensus_graph.v1":
            return dict(direct)
        team = dict(self.blackboard.get("codex_agent_team") or {})
        nested = team.get("route_consensus_graph")
        if (
            team.get("accepted") is True
            and isinstance(nested, dict)
            and nested.get("schema_version") == "route_consensus_graph.v1"
        ):
            return dict(nested)
        return {}

    def _route_consensus_graph_refs(self) -> list[str]:
        team = dict(self.blackboard.get("codex_agent_team") or {})
        refs = [str(team.get("route_consensus_graph_ref") or "")]
        refs.extend(
            str(value)
            for key, value in (self.blackboard.get("artifact_refs") or {}).items()
            if "route_consensus_graph" in str(key).lower() and str(value).strip()
        )
        return _dedupe(refs)

    def _route_consensus_graph_view(self) -> dict[str, Any]:
        graph = self._route_consensus_graph_payload()
        if not graph:
            return {
                "schema_version": "route_consensus_graph_view.v1",
                "available": False,
                "route_count": 0,
                "step_count": 0,
                "node_count": 0,
                "semantics": {"advisory_only": True, "solved": False, "executable": False},
            }
        return {
            "schema_version": "route_consensus_graph_view.v1",
            "source_schema_version": str(graph.get("schema_version") or ""),
            "available": True,
            "has_hypotheses": bool(graph.get("has_hypotheses")),
            "route_count": len(graph.get("route_hypotheses") or []),
            "step_count": len(graph.get("steps") or []),
            "node_count": len(graph.get("nodes") or []),
            "conflict_count": len(graph.get("conflicts") or []),
            "cycle_count": len(graph.get("cycles") or []),
            "truncation": dict(graph.get("truncation") or {}),
            "route_level_source_refs": self._route_consensus_graph_refs(),
            "semantics": {"advisory_only": True, "solved": False, "executable": False},
        }

    def _route_consensus_payload(self) -> dict[str, Any]:
        if self._codex_team_projection_reasons():
            return {}
        direct = self.blackboard.get("route_consensus")
        if isinstance(direct, dict) and direct.get("schema_version") == "route_consensus.v1":
            return dict(direct)
        team = dict(self.blackboard.get("codex_agent_team") or {})
        nested = team.get("route_consensus")
        if (
            team.get("accepted") is True
            and isinstance(nested, dict)
            and nested.get("schema_version") == "route_consensus.v1"
        ):
            return dict(nested)
        return {}

    def _codex_team_projection_reasons(self) -> list[str]:
        team = self.blackboard.get("codex_agent_team")
        if not isinstance(team, dict) or not team:
            return []
        reasons: list[str] = []
        if team.get("accepted") is not True:
            reasons.append("codex_agent_team_not_accepted")
        validation = team.get("artifact_validation")
        if isinstance(validation, dict) and validation.get("accepted") is not True:
            reasons.append("codex_agent_team_artifact_validation_failed")
        coordinator = team.get("coordinator")
        if isinstance(coordinator, dict):
            status = str(coordinator.get("status") or "").strip()
            if status and status != "accepted_draft":
                reasons.append(f"codex_agent_team_coordinator_status:{status}")
        runtime = team.get("runtime_summary")
        if isinstance(runtime, dict) and runtime.get("consistent") is not True:
            reasons.append("codex_agent_team_runtime_inconsistent")
        child_reports = team.get("child_reports")
        if isinstance(child_reports, list) and any(
            not isinstance(row, dict) or row.get("accepted") is not True for row in child_reports
        ):
            reasons.append("codex_agent_team_child_report_rejected")
        return _dedupe(reasons)

    def _route_consensus_route_refs(self) -> list[str]:
        team = dict(self.blackboard.get("codex_agent_team") or {})
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        refs = [str(team.get("route_consensus_ref") or "")]
        refs.extend(
            str(value)
            for key, value in artifact_refs.items()
            if "route_consensus" in str(key).lower() and str(value).strip()
        )
        return _dedupe(refs)

    def _route_consensus_view(self) -> dict[str, Any]:
        consensus = self._route_consensus_payload()
        if not consensus:
            quarantine_reasons = self._codex_team_projection_reasons()
            return {
                "schema_version": "route_consensus_view.v1",
                "available": False,
                "quarantined": bool(quarantine_reasons),
                "reasons": quarantine_reasons,
                "proposals": [],
                "source_summary": {},
                "semantics": {
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "deterministic_parent_proof_required": True,
                },
            }
        proposal_views = []
        for proposal in consensus.get("proposals") or []:
            if not isinstance(proposal, dict):
                continue
            consensus_id = str(proposal.get("consensus_id") or "")
            support_records = _consensus_support_records(proposal)
            support_groups = _consensus_independent_support_groups(proposal, support_records)
            proposal_views.append(
                {
                    "consensus_id": consensus_id,
                    "branch_id": self._consensus_branch_ids.get(consensus_id, ""),
                    "rank": int(proposal.get("rank") or 0),
                    "reaction_family": str(proposal.get("reaction_family") or "unspecified"),
                    "status": str(proposal.get("status") or "model_hypothesis"),
                    "evidence_level": str(proposal.get("evidence_level") or "model_only"),
                    "confidence": str(proposal.get("confidence") or "low"),
                    "rank_score": float(proposal.get("rank_score") or 0.0),
                    "source_channels": _dedupe([str(item) for item in proposal.get("source_channels") or []]),
                    "support_records": support_records,
                    "support_count": len(support_records) or int(proposal.get("support_count") or 0),
                    "independent_support_groups": support_groups,
                    "independent_source_count": len(support_groups),
                    "codex_roles_correlated": any(group == "codex_model" for group in support_groups),
                    "source_refs": _consensus_direct_source_refs(proposal),
                    "condition_support": [
                        dict(row) for row in proposal.get("condition_support") or [] if isinstance(row, dict)
                    ],
                    "conflicts": _consensus_conflicts(proposal),
                    "limitations": _dedupe([str(item) for item in proposal.get("limitations") or []]),
                    "required_validation": _dedupe(
                        [str(item) for item in proposal.get("required_validation") or []]
                    ),
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "not_parent_route_proof": True,
                }
            )
        return {
            "schema_version": "route_consensus_view.v1",
            "source_schema_version": "route_consensus.v1",
            "available": True,
            "has_candidates": bool(proposal_views),
            "accepted_as_route": False,
            "route_level_source_refs": self._route_consensus_route_refs(),
            "source_summary": dict(consensus.get("source_summary") or {}),
            "proposals": proposal_views,
            "rejected_candidates": [
                dict(row) for row in consensus.get("rejected_candidates") or [] if isinstance(row, dict)
            ],
            "semantics": {
                "advisory_only": True,
                "solved": False,
                "executable": False,
                "deterministic_parent_proof_required": True,
                "codex_roles_are_correlated": True,
            },
        }

    def add_proposal_branches(self, *, limit: int) -> None:
        proposals = [row for row in self.blackboard.get("retrosynthetic_proposals") or [] if isinstance(row, dict)]
        if self._codex_team_projection_reasons():
            proposals = [
                row
                for row in proposals
                if str(row.get("source_type") or "") != "multi_source_consensus"
                and not str(row.get("proposal_id") or "").startswith("consensus:")
            ]
        if self._route_consensus_payload():
            proposals = [
                row
                for row in proposals
                if str(row.get("source_type") or "") != "multi_source_consensus"
                and not str(row.get("proposal_id") or "").startswith("consensus:")
            ]
        proposals.sort(key=lambda row: (not bool(row.get("executable")), -float(row.get("score") or 0.0)))
        seen_labels: set[str] = set()
        count = 0
        for proposal in proposals:
            label = _clean_label(proposal.get("proposal_label") or proposal.get("proposal_type") or "proposal")
            dedupe_key = f"{label}:{proposal.get('precursor_smiles') or ''}"[:220]
            if dedupe_key in seen_labels:
                continue
            seen_labels.add(dedupe_key)
            count += 1
            if count > max(0, limit):
                break
            branch_id = self._unique_branch_id(f"branch:proposal:{_slug(str(proposal.get('proposal_id') or label))}")
            precursor_nodes = self._proposal_precursor_nodes(proposal)
            if not precursor_nodes:
                precursor_nodes = [
                    self._add_node(
                        f"Strategic precursor: {label}",
                        role="hypothesis_precursor",
                        exactness="model_hypothesis",
                        confidence=str(proposal.get("confidence") or "medium"),
                        source_refs=[str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()],
                        missing=["no machine-readable precursor structure"],
                    )
                ]
            source_refs = [str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()][:8]
            product_smiles = str(
                proposal.get("product_smiles") or proposal.get("target_smiles") or ""
            ).strip()
            requested_target_smiles = str(self._target().get("smiles") or "")
            if product_smiles and _same_molecule(product_smiles, requested_target_smiles):
                product_node = self._target_node()
            else:
                product_label = _clean_label(
                    proposal.get("product_label")
                    or proposal.get("target_label")
                    or (_compact_smiles_label(product_smiles) if product_smiles else "unbound proposal product")
                )
                product_node = self._add_node(
                    product_label,
                    role="hypothesis_product",
                    smiles=product_smiles if _looks_like_smiles(product_smiles) else "",
                    exactness="model_hypothesis",
                    confidence=str(proposal.get("confidence") or "medium"),
                    source_refs=source_refs,
                    missing=[
                        "proposal product is not the requested target"
                        if product_smiles
                        else "proposal product identity is not bound to the requested target"
                    ],
                )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=precursor_nodes,
                to_nodes=[product_node],
                module_key=_module_key_for_text(label),
                module_label=_module_label_for_key(_module_key_for_text(label)),
                confidence=str(proposal.get("confidence") or "medium"),
                exactness="model_hypothesis",
                source_refs=source_refs,
                origin="retrosynthetic_proposal",
                summary=str(proposal.get("proposal_type") or proposal.get("route_objective_type") or "Explored proposal"),
                conditions=_conditions_from_row(proposal),
                missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:8],
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"候选逆合成分支：{label}",
                kind="retrosynthetic_proposal",
                recommendation="探索触碰",
                confidence=str(proposal.get("confidence") or "medium"),
                summary="A model/planner proposal touched during exploration. It is not a parent-route proof.",
                step_ids=[step_id],
                source_refs=[str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()][:8],
                missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:8],
                classification_records=[proposal],
            )

    def add_template_branches(self, *, limit: int) -> None:
        templates = [row for row in self.blackboard.get("broad_transform_templates") or [] if isinstance(row, dict)]
        for index, template in enumerate(templates[: max(0, limit)], start=1):
            template_id = str(template.get("template_id") or f"template:{index}")
            branch_id = self._unique_branch_id(f"branch:template:{_slug(template_id)}")
            label = _clean_label(template.get("transform_logic") or template.get("objective_type") or template_id)
            source_refs = [str(x) for x in template.get("source_refs") or template.get("evidence_refs") or [] if str(x).strip()]
            from_label, to_label = _template_endpoint_labels(template)
            reactant_smiles = str(
                template.get("reactant_smiles") or template.get("from_smiles") or ""
            ).strip()
            product_smiles = str(
                template.get("product_smiles") or template.get("to_smiles") or ""
            ).strip()
            requested_target_smiles = str(self._target().get("smiles") or "")
            from_node = self._add_node(
                from_label,
                role="template_precursor",
                smiles=reactant_smiles if _looks_like_smiles(reactant_smiles) else "",
                exactness="model_hypothesis",
                confidence="medium",
                source_refs=source_refs,
                missing=["broad-template endpoint; exact structure is not established"],
            )
            if product_smiles and _same_molecule(product_smiles, requested_target_smiles):
                to_node = self._target_node()
            else:
                to_node = self._add_node(
                    to_label,
                    role="template_product",
                    smiles=product_smiles if _looks_like_smiles(product_smiles) else "",
                    exactness="model_hypothesis",
                    confidence="medium",
                    source_refs=source_refs,
                    missing=["broad-template endpoint is not bound to the requested target"],
                )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=[from_node],
                to_nodes=[to_node],
                module_key=_module_key_for_text(label + " " + str(template.get("reaction_center") or "")),
                module_label=_module_label_for_key(_module_key_for_text(label)),
                confidence="medium",
                exactness="model_hypothesis",
                source_refs=source_refs,
                origin="broad_transform_template",
                summary=str(template.get("reaction_center") or template.get("objective_type") or ""),
                conditions=_conditions_from_row(template),
                missing=_dedupe(
                    [
                        *[str(x) for x in template.get("risk_flags") or [] if str(x).strip()],
                        "broad template is advisory and does not imply a route to the requested target",
                    ]
                )[:8],
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"宽泛模板分支：{_clean_label(template.get('objective_type') or template_id)}",
                kind="broad_template",
                recommendation="模板提示",
                confidence="medium",
                summary="A broad transform touched by the planner after exact literature rows were unavailable.",
                step_ids=[step_id],
                source_refs=source_refs,
                missing=_dedupe(
                    [
                        *[str(x) for x in template.get("risk_flags") or [] if str(x).strip()],
                        "broad template is advisory and does not imply a route to the requested target",
                    ]
                )[:8],
                classification_records=[template],
            )

    def add_exact_row_branch(self) -> None:
        rows = self._source_detail_chain_rows()
        if not rows:
            rows = [row for row in self.evidence.get("exact_rows") or [] if isinstance(row, dict)]
        if not rows:
            return
        branch_id = "branch:exact_literature_rows"
        step_ids: list[str] = []
        verified_flags: list[bool] = []
        for index, row in enumerate(rows, start=1):
            source_ref = str(row.get("source_ref") or row.get("source_title") or "")
            source_refs = _dedupe(
                [
                    source_ref,
                    *[str(x) for x in row.get("evidence_refs") or [] if str(x).strip()],
                ]
            )
            label = _clean_label(
                row.get("reaction_label")
                or row.get("step_label")
                or row.get("step_id")
                or row.get("source_template_id")
                or row.get("row_id")
                or f"exact row {index}"
            )
            reactants = _labels_from_any(
                row.get("reactant_labels")
                or row.get("reactants")
                or row.get("reactant_smiles")
                or row.get("main_reactant_smiles")
            )
            products = _labels_from_any(row.get("product_labels") or row.get("products") or row.get("product_smiles"))
            module_key = self._exact_row_module_key(row=row, label=label, index=index)
            module_label = self._exact_row_module_label(row=row, label=label, fallback=module_key)
            row_verified = _exact_row_is_verified(row)
            verified_flags.append(row_verified)
            row_confidence = "high" if row_verified else "medium"
            row_exactness = "exact_literature_row" if row_verified else "named_literature"
            row_missing = [] if row_verified else ["exact row is not deterministically validated"]

            def exact_node(value: str, role: str) -> str:
                return self._add_node(
                    label=_exact_node_label(value, row=row, role=role),
                    role=role,
                    smiles=value if _looks_like_smiles(value) else "",
                    exactness=row_exactness,
                    confidence=row_confidence,
                    source_refs=source_refs,
                    missing=row_missing,
                )

            step_ids.append(
                self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=[exact_node(x, "exact_reactant") for x in reactants],
                    to_nodes=[exact_node(x, "exact_product") for x in products],
                    module_key=module_key,
                    module_label=module_label,
                    confidence=row_confidence,
                    exactness=row_exactness,
                    source_refs=source_refs,
                    origin="exact_literature_row",
                    summary="Exact row compiled from source details.",
                    conditions=_conditions_from_row(row),
                    missing=row_missing,
                )
            )
        all_verified = bool(verified_flags and all(verified_flags))
        self._add_branch(
            branch_id=branch_id,
            title="Exact literature rows",
            kind="exact_literature" if all_verified else "literature_candidate",
            recommendation="强证据",
            confidence="high" if all_verified else "medium",
            summary="Machine-readable literature rows; only deterministically validated rows are displayed as exact.",
            step_ids=step_ids,
            source_refs=_dedupe([str(row.get("source_ref") or "") for row in rows if str(row.get("source_ref") or "").strip()]),
            missing=[] if all_verified else ["contains unvalidated literature rows"],
            classification_records=rows,
        )

    def _source_detail_chain_rows(self) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        if self.run_dir:
            run = Path(self.run_dir)
            candidates.extend(
                [
                    run / "source_detail_chain_route_result.json",
                    run / "source_detail_chain_route" / "source_detail_route_chain_audit.json",
                ]
            )
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        for key, value in artifact_refs.items():
            if "source_detail" not in str(key).lower() and "source_detail" not in str(value).lower():
                continue
            candidates.append(Path(str(value)))
        seen: set[str] = set()
        for path in candidates:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                continue
            path_key = str(resolved).lower()
            if path_key in seen or not resolved.is_file():
                continue
            seen.add(path_key)
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = _chain_rows_from_source_detail_payload(data)
            if rows:
                return rows
        return []

    def _exact_row_module_key(self, *, row: dict[str, Any], label: str, index: int) -> str:
        source_id = str(row.get("source_template_id") or row.get("step_id") or row.get("row_id") or "").strip()
        if source_id:
            return f"source_detail_exact_step:{_slug(source_id)}"
        return f"source_detail_exact_step:{index:02d}_{_slug(label)}"

    def _exact_row_module_label(self, *, row: dict[str, Any], label: str, fallback: str) -> str:
        condition = dict(row.get("condition_candidate") or {})
        bits = [
            str(row.get("step_id") or row.get("row_id") or label or "").replace("_", " "),
            str(condition.get("reagent") or condition.get("reagents") or "").strip(),
            str(condition.get("reported_yield") or condition.get("yield") or "").strip(),
        ]
        text = " · ".join(bit for bit in bits if bit)
        return text[:80] or fallback

    def _visual_step(self, branch_id: str, chain: dict[str, Any], row: dict[str, Any], index: int) -> str:
        source_ref = str(chain.get("source_ref") or chain.get("source_title") or "")
        reactants = _labels_from_any(row.get("reactant_labels") or row.get("reactants") or row.get("main_reactant_smiles") or row.get("reactant_smiles"))
        products = _labels_from_any(row.get("product_label") or row.get("product_labels") or row.get("product_smiles"))
        if not products:
            products = [f"visual product {index}"]
        from_nodes = []
        reactant_smiles = _labels_from_any(row.get("reactant_smiles") or row.get("main_reactant_smiles"))
        for idx, label in enumerate(reactants or [f"visual precursor {index}"], start=0):
            smiles = reactant_smiles[idx] if idx < len(reactant_smiles) and _looks_like_smiles(reactant_smiles[idx]) else ""
            from_nodes.append(
                self._add_node(
                    label=label,
                    role="visual_precursor",
                    smiles=smiles,
                    exactness="visual_inferred",
                    confidence=str(row.get("confidence") or "low"),
                    source_refs=[source_ref] if source_ref else [],
                    missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:4],
                )
            )
        product_smiles = _labels_from_any(row.get("product_smiles"))
        to_nodes = []
        for idx, label in enumerate(products, start=0):
            smiles = product_smiles[idx] if idx < len(product_smiles) and _looks_like_smiles(product_smiles[idx]) else ""
            to_nodes.append(
                self._add_node(
                    label=label,
                    role="visual_product",
                    smiles=smiles,
                    exactness="visual_inferred",
                    confidence=str(row.get("confidence") or "low"),
                    source_refs=[source_ref] if source_ref else [],
                    missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:4],
                )
            )
        label = _clean_label(row.get("reaction_class") or row.get("step_id") or f"visual step {index}")
        return self._add_step(
            branch_id=branch_id,
            label=label,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            module_key=_module_key_for_text(" ".join([label, source_ref, str(row.get("source_locator") or "")])),
            module_label=_module_label_for_key(_module_key_for_text(label)),
            confidence=str(row.get("confidence") or "low"),
            exactness="visual_inferred",
            source_refs=[source_ref] if source_ref else [],
            origin="visual_chain",
            summary=str(row.get("source_locator") or row.get("allowed_use") or ""),
            conditions=_conditions_from_row(row),
            missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:8],
        )

    def _proposal_precursor_nodes(self, proposal: dict[str, Any]) -> list[str]:
        text = str(proposal.get("precursor_smiles") or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split(".") if part.strip()]
        out = []
        for idx, smiles in enumerate(parts[:5], start=1):
            label = f"proposal precursor {idx}"
            out.append(
                self._add_node(
                    label=label,
                    role="hypothesis_precursor",
                    smiles=smiles if _looks_like_smiles(smiles) else "",
                    exactness="model_hypothesis",
                    confidence=str(proposal.get("confidence") or "medium"),
                    source_refs=[str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()][:8],
                    missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:4],
                )
            )
        return out

    def _target_node(self) -> str:
        target = self._target()
        return self._add_node(
            target.get("name") or "target",
            role="target",
            smiles=str(target.get("smiles") or ""),
            exactness="name_only",
            confidence="high",
            source_refs=[],
        )

    def _target(self) -> dict[str, Any]:
        profile = dict(self.blackboard.get("target_profile") or {})
        raw_name = str(profile.get("target_name") or self.blackboard.get("case_id") or "")
        family_hint = str(profile.get("family_hint") or "")
        return {
            "name": _display_target_name(raw_name, family_hint, str(self.blackboard.get("case_id") or "")),
            "smiles": str(profile.get("target_smiles") or profile.get("canonical_smiles") or profile.get("isomeric_smiles") or ""),
            "family_hint": family_hint,
        }

    def _add_branch(
        self,
        *,
        branch_id: str,
        title: str,
        kind: str,
        recommendation: str,
        confidence: str,
        summary: str,
        step_ids: list[str],
        source_refs: list[str],
        missing: list[str],
        classification_records: list[dict[str, Any]] | None = None,
    ) -> None:
        branch_id = self._unique_branch_id(branch_id)
        title = _branch_title_for_display(branch_id=branch_id, title=title, kind=kind)
        recommendation = _recommendation_for_display(kind=kind, recommendation=recommendation)
        node_ids: list[str] = []
        for step_id in step_ids:
            step = self.steps.get(step_id) or {}
            node_ids.extend([str(x) for x in step.get("from_node_ids") or []])
            node_ids.extend([str(x) for x in step.get("to_node_ids") or []])
        row = {
            "branch_id": branch_id,
            "title": title,
            "kind": kind,
            "recommendation": recommendation,
            "confidence": _normalize_confidence(confidence),
            "summary": summary,
            "step_ids": [sid for sid in step_ids if sid in self.steps],
            "node_ids": _dedupe(node_ids),
            "source_refs": _dedupe(source_refs),
            "missing": _dedupe(missing),
        }
        verified_parent_route = kind in {"direct_verified_route", "stitched_verified_route"}
        row.update(
            {
                "solved": verified_parent_route,
                "executable": verified_parent_route,
                "advisory_only": not verified_parent_route,
                "not_parent_route_proof": not verified_parent_route,
            }
        )
        row.update(_classify_synthesis_records(classification_records or []))
        self.branches.append(row)

    def _add_step(
        self,
        *,
        branch_id: str,
        label: str,
        from_nodes: list[str],
        to_nodes: list[str],
        module_key: str,
        module_label: str,
        confidence: str,
        exactness: str,
        source_refs: list[str],
        origin: str,
        summary: str,
        missing: list[str],
        conditions: list[dict[str, str]] | None = None,
    ) -> str:
        if _display_text_is_corrupt(module_label):
            module_label = _module_label_for_key(module_key)
        step_id = f"step:{_slug(branch_id)}:{_slug(label)}:{len(self.steps) + 1}"
        condition_rows = _normalize_condition_rows(conditions or [])
        self.steps[step_id] = {
            "step_id": step_id,
            "branch_id": branch_id,
            "label": label,
            "from_node_ids": [x for x in from_nodes if x],
            "to_node_ids": [x for x in to_nodes if x],
            "module_key": module_key or "other",
            "module_label": module_label or "Other",
            "confidence": _normalize_confidence(confidence),
            "exactness": _normalize_exactness(exactness),
            "source_refs": _dedupe(source_refs),
            "origin": origin,
            "summary": summary,
            "conditions": condition_rows,
            "condition_summary": _condition_summary(condition_rows),
            "condition_status": _condition_status(condition_rows, missing),
            "missing": _dedupe(missing),
        }
        return step_id

    def _add_node(
        self,
        label: str,
        *,
        role: str,
        smiles: str = "",
        exactness: str,
        confidence: str,
        source_refs: list[str],
        missing: list[str] | None = None,
    ) -> str:
        label = _clean_label(label) or "unnamed node"
        smiles = str(smiles or "").strip()
        node_id, canonical_smiles = _molecule_node_identity(smiles=smiles, label=label)
        existing = self.nodes.get(node_id)
        row = {
            "node_id": node_id,
            "label": label,
            "role": role,
            "roles": [role],
            "smiles": canonical_smiles or smiles,
            "input_smiles": smiles,
            "canonical_isomeric_smiles": canonical_smiles,
            "representation_kind": "smiles" if smiles else "name_only",
            "exactness": _normalize_exactness(exactness),
            "confidence": _normalize_confidence(confidence),
            "source_refs": _dedupe(source_refs),
            "missing": _dedupe(missing or []),
            "assertions": [{
                "exactness": _normalize_exactness(exactness),
                "confidence": _normalize_confidence(confidence),
                "source_refs": _dedupe(source_refs),
            }],
        }
        if existing:
            row["label"] = _better_node_label(existing.get("label"), label)
            row["roles"] = _dedupe(
                [
                    *[str(item) for item in existing.get("roles") or [existing.get("role")] if str(item or "")],
                    role,
                ]
            )
            row["role"] = _preferred_node_role(row["roles"])
            row["source_refs"] = _dedupe([*(existing.get("source_refs") or []), *row["source_refs"]])
            row["missing"] = _dedupe([*(existing.get("missing") or []), *row["missing"]])
            row["assertions"] = [*(existing.get("assertions") or []), *row["assertions"]]
            row["exactness"] = _worst_ranked(existing.get("exactness"), row["exactness"], EXACTNESS_RANK)
            row["confidence"] = _worst_ranked(existing.get("confidence"), row["confidence"], CONFIDENCE_RANK)
            if not existing.get("smiles") and smiles:
                row["smiles"] = smiles
                row["representation_kind"] = "smiles"
            elif existing.get("smiles"):
                row["smiles"] = existing.get("smiles")
                row["representation_kind"] = "smiles"
        row.update(_structure_payload_for_smiles(row.get("smiles")))
        self.nodes[node_id] = row
        return node_id

    def _node_id_for_label(self, label: str) -> str:
        node_id, _ = _molecule_node_identity(smiles="", label=label)
        if node_id not in self.nodes:
            return self._add_node(label, role="intermediate", exactness="name_only", confidence="medium", source_refs=[])
        return node_id

    def _best_direct_route_result(self) -> dict[str, Any]:
        proof = dict(self.blackboard.get("parent_route_proof") or {})
        proof_accepted = is_solved_parent_route_proof(
            proof,
            expected_target_smiles=str(self._target().get("smiles") or ""),
        )
        for artifact in self._guided_result_artifacts():
            verifier = dict(artifact.get("raw_route_verifier") or {})
            # Never promote the backend's own ``solved`` claim (or an unrelated
            # parent proof) into verification of this artifact.  The route and
            # best rank below are meaningful only under this artifact's
            # deterministic verifier report.
            if not _deterministic_route_verifier_accepted(
                verifier,
                expected_target_smiles=str(self._target().get("smiles") or ""),
            ):
                continue
            result = dict(artifact.get("result") or {})
            routes = [dict(row) for row in result.get("routes") or artifact.get("routes") or [] if isinstance(row, dict)]
            if not routes:
                continue
            expected_target = str(self._target().get("smiles") or "")
            reverified = verify_chemenzy_raw_routes(
                {"result": {**result, "routes": routes}},
                target_smiles=expected_target,
            )
            if not is_accepted_route_verifier_report(
                reverified,
                expected_target_smiles=expected_target,
            ):
                continue
            if (
                verifier.get("best_route_rank") != reverified.get("best_route_rank")
                or int(verifier.get("best_route_step_count") or 0)
                != int(reverified.get("best_route_step_count") or 0)
            ):
                continue
            best_rank = reverified.get("best_route_rank")
            route = _route_by_verified_rank(routes, best_rank)
            if route.get("steps"):
                return {
                    "route": route,
                    "artifact_path": str(artifact.get("_artifact_path") or ""),
                    "source_ref": str(artifact.get("_artifact_key") or "guided_chemenzy_result"),
                    "classification_records": [artifact, result, route],
                }
        if proof_accepted:
            route = _route_from_parent_route_proof(proof)
            if route.get("steps"):
                return {
                    "route": route,
                    "artifact_path": "",
                    "source_ref": str(proof.get("source_ref") or "parent_route_proof"),
                    "classification_records": [proof, route],
                }
        return {}

    def _guided_result_artifacts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        candidate_paths: list[tuple[str, Path]] = []
        for key, value in artifact_refs.items():
            text = str(value or "").strip()
            if not text:
                continue
            if "chemenzy" not in str(key).lower() and "chemenzy" not in text.lower() and "guided" not in text.lower():
                continue
            candidate_paths.append((str(key), Path(text)))
        if self.run_dir:
            run = Path(self.run_dir)
            for name in ("guided_chemenzy_result.json", "guided_chemenzy_raw_result.json"):
                candidate_paths.append((name, run / name))
            for path in run.glob("*guided_chemenzy*_result*.json"):
                candidate_paths.append((path.name, path))
        for key, path in candidate_paths:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                continue
            path_key = str(resolved).lower()
            if path_key in seen or not resolved.is_file():
                continue
            seen.add(path_key)
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            data["_artifact_key"] = key
            data["_artifact_path"] = str(resolved)
            out.append(data)
        return out

    def _subgoal_route_records(self) -> list[dict[str, Any]]:
        candidate_paths: list[tuple[str, Path]] = []
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        for key, value in artifact_refs.items():
            text = f"{key} {value}".lower()
            if "route_expansion" in text or "subgoal" in text:
                candidate_paths.append((str(key), Path(str(value))))
        if self.run_dir:
            run = Path(self.run_dir)
            candidate_paths.append(("route_expansion_subgoal_search_result", run / "route_expansion_subgoal_search_result.json"))
            for path in run.glob("*route_expansion_subgoal_search_result*.json"):
                candidate_paths.append((path.name, path))

        out: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_records: set[str] = set()
        for key, path in candidate_paths:
            resolved = self._resolve_artifact_path(path)
            if resolved is None:
                continue
            path_key = str(resolved).lower()
            if path_key in seen_paths or not resolved.is_file():
                continue
            seen_paths.add(path_key)
            data = self._read_json_dict(resolved)
            if not data:
                continue
            if data.get("accepted") is False or data.get("solved") is False:
                continue
            for index, row in enumerate(data.get("subgoals") or [], start=1):
                if not isinstance(row, dict):
                    continue
                verifier = dict(row.get("verifier") or {})
                parent_relevance = dict(row.get("parent_relevance_gate") or {})
                accepted = bool(
                    row.get("accepted") is True
                    and row.get("solved") is True
                    and str(row.get("route_status") or "").strip().lower() == "solved"
                    and parent_relevance.get("accepted") is True
                    and _deterministic_route_verifier_accepted(verifier)
                )
                if not accepted:
                    continue
                subgoal = dict(row.get("subgoal") or {})
                policy = dict(subgoal.get("policy") or {})
                preferred = dict(policy.get("preferred_subgoal") or {})
                terminal = dict(preferred.get("terminal_candidate") or {})
                name = (
                    str(subgoal.get("name") or "").strip()
                    or str(terminal.get("name") or "").strip()
                    or f"subgoal {index}"
                )
                target_smiles = (
                    str(subgoal.get("smiles") or "").strip()
                    or str(terminal.get("canonical_smiles") or terminal.get("smiles") or "").strip()
                )
                raw_path = self._resolve_artifact_path(row.get("raw_result_path"))
                raw = self._read_json_dict(raw_path) if raw_path is not None else {}
                routes = [
                    dict(candidate)
                    for candidate in (raw.get("routes") or (raw.get("result") or {}).get("routes") or [])
                    if isinstance(candidate, dict)
                ]
                best_rank = verifier.get("best_route_rank")
                route = self._choose_route_by_rank(routes, best_rank)
                if not any(_materialized_route_step(step) for step in route.get("steps") or []):
                    continue
                route_rank = route.get("route_rank") if route else best_rank
                record_key = f"{name}|{target_smiles}|{raw_path or ''}|{route_rank}"
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)
                evidence_refs = _dedupe(
                    [
                        *[str(x) for x in policy.get("evidence_refs") or [] if str(x).strip()],
                        str(terminal.get("source_ref") or ""),
                    ]
                )
                out.append(
                    {
                        "name": name,
                        "target_smiles": target_smiles,
                        "accepted": accepted,
                        "route": route,
                        "route_objective_type": str(
                            subgoal.get("route_objective_type")
                            or policy.get("route_objective_type")
                            or ""
                        ),
                        "route_rank": route_rank,
                        "search_path": str(resolved),
                        "search_key": key,
                        "raw_path": str(raw_path or ""),
                        "evidence_refs": evidence_refs,
                        "reasons": [str(x) for x in verifier.get("reasons") or row.get("reasons") or []],
                        "accepted_route_count": verifier.get("accepted_route_count"),
                        "route_count": row.get("route_count") or raw.get("n_results"),
                    }
                )
        return out

    def _choose_route_by_rank(self, routes: list[dict[str, Any]], best_rank: Any) -> dict[str, Any]:
        return _route_by_verified_rank(routes, best_rank)

    def _resolve_artifact_path(self, path: Any) -> Path | None:
        text = str(path or "").strip()
        if not text:
            return None
        raw = Path(text)
        candidates = [raw]
        if self.run_dir:
            run = Path(self.run_dir)
            candidates.append(run / raw.name)
            candidates.append(run / "route_expansion_subgoals" / raw.name)
            parts = list(raw.parts)
            if run.name in parts:
                index = parts.index(run.name)
                tail = parts[index + 1 :]
                if tail:
                    candidates.append(run.joinpath(*tail))
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        try:
            return raw.expanduser().resolve()
        except OSError:
            return raw

    def _read_json_dict(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _route_smiles_label(self, value: str, *, role: str, target_smiles: str) -> str:
        text = str(value or "").strip()
        if target_smiles and _same_text(text, target_smiles):
            return self._target().get("name") or "target"
        if not text:
            return role
        if _looks_like_smiles(text):
            return _compact_smiles_label(text)
        return _clean_label(text)

    def _unique_branch_id(self, branch_id: str) -> str:
        base = branch_id
        idx = 2
        while branch_id in self._branch_ids:
            branch_id = f"{base}:{idx}"
            idx += 1
        self._branch_ids.add(branch_id)
        return branch_id

    def _target_node_ids(self) -> set[str]:
        target = self._target()
        target_name = str(target.get("name") or "").strip().lower()
        target_smiles = str(target.get("smiles") or "").strip()
        out: set[str] = set()
        for node_id, node in self.nodes.items():
            label = str(node.get("label") or "").strip().lower()
            smiles = str(node.get("smiles") or "").strip()
            role = str(node.get("role") or "").strip().lower()
            if role == "target":
                out.add(node_id)
                continue
            if target_smiles and smiles == target_smiles:
                out.add(node_id)
                continue
            if target_name and target_name in label and ("free acid" in label or "target" in role):
                out.add(node_id)
        return out

    def _branch_modules(self, branch: dict[str, Any]) -> set[str]:
        generic = {"", "other", "other_route_module", "diagnostic_failure", "visual_failed_or_empty"}
        out: set[str] = set()
        for step_id in branch.get("step_ids") or []:
            step = self.steps.get(str(step_id)) or {}
            key = str(step.get("module_key") or "")
            if key not in generic:
                out.add(key)
        return out

    def _branch_relationships(self) -> list[dict[str, Any]]:
        target_ids = self._target_node_ids()
        out: list[dict[str, Any]] = []
        for left_index, left in enumerate(self.branches):
            for right in self.branches[left_index + 1 :]:
                rel = self._branch_relationship(left, right, target_ids)
                if rel:
                    out.append(rel)
        return out

    def _branch_relationship(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        target_ids: set[str],
    ) -> dict[str, Any] | None:
        major_kinds = {
            "stitched_verified_route",
            "direct_verified_route",
            "subgoal_verified_route",
            "exact_literature",
            "process_evidence",
            "visual_chain",
        }
        left_kind = str(left.get("kind") or "")
        right_kind = str(right.get("kind") or "")
        left_major = left_kind in major_kinds
        right_major = right_kind in major_kinds
        left_nodes = set(str(x) for x in left.get("node_ids") or [])
        right_nodes = set(str(x) for x in right.get("node_ids") or [])
        shared_nodes = sorted(left_nodes & right_nodes)
        shared_target_nodes = [node_id for node_id in shared_nodes if node_id in target_ids]
        left_modules = self._branch_modules(left)
        right_modules = self._branch_modules(right)
        shared_modules = sorted(left_modules & right_modules)
        right_refs = set(str(x) for x in right.get("source_refs") or [])
        shared_refs = _dedupe(
            [
                ref
                for ref in left.get("source_refs") or []
                if ref and ref in right_refs and _external_source_ref(ref)
            ]
        )
        if shared_target_nodes and left_major and right_major:
            kind = "shared_target_endpoint"
        elif shared_modules and (left_major or right_major):
            kind = "same_reaction_module"
        elif shared_refs and (left_major or right_major):
            kind = "shared_literature_source"
        else:
            return None
        source_refs = shared_refs or _dedupe([*(left.get("source_refs") or []), *(right.get("source_refs") or [])])[:8]
        if "route_consensus" in {left_kind, right_kind}:
            summary = "共识候选与另一分支共享目标或反应模块；该关系仅供候选对照，不构成 solved 或 executable 证明。"
        elif kind == "shared_target_endpoint":
            summary = "这些路线共享目标或终点分子，应作为同一目标下的路线变体对照查看。"
        elif kind == "same_reaction_module":
            labels = [_module_label_for_key(key) for key in shared_modules[:3]]
            summary = f"这些路线触碰了相同反应模块：{'、'.join(labels)}。"
        else:
            summary = "这些路线引用了相同文献来源，可作为同一证据链下的分支查看。"
        shared_node_labels = [
            str((self.nodes.get(node_id) or {}).get("label") or node_id)
            for node_id in shared_nodes[:8]
        ]
        return {
            "relationship_id": f"rel:{_slug(left.get('branch_id'))}:{_slug(right.get('branch_id'))}:{kind}",
            "kind": kind,
            "from_branch_id": str(left.get("branch_id") or ""),
            "to_branch_id": str(right.get("branch_id") or ""),
            "summary": summary,
            "shared_node_ids": shared_nodes[:12],
            "shared_node_labels": shared_node_labels,
            "shared_module_keys": shared_modules[:12],
            "shared_module_labels": [_module_label_for_key(key) for key in shared_modules[:12]],
            "source_refs": source_refs,
        }

    def _modules(self) -> list[dict[str, Any]]:
        rows: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        for step in self.steps.values():
            key = str(step.get("module_key") or "other")
            rows.setdefault(key, []).append(str(step.get("step_id") or ""))
            labels.setdefault(key, str(step.get("module_label") or key))
        return [
            {
                "module_key": key,
                "module_label": labels.get(key, key),
                "step_ids": ids,
                "alternative_count": max(0, len(ids) - 1),
            }
            for key, ids in sorted(rows.items(), key=lambda item: (-len(item[1]), item[0]))
        ]

    def _evidence_index(self) -> dict[str, Any]:
        return {
            "source_candidates": [
                {
                    "source_ref": str(row.get("source_ref") or ""),
                    "title": str(row.get("title") or row.get("source_title") or ""),
                    "local_pdf": str(row.get("local_pdf") or row.get("pdf_path") or ""),
                }
                for row in self.evidence.get("source_candidates") or []
                if isinstance(row, dict)
            ][:20],
            "exact_chain_audits": [
                {
                    "accepted": bool(row.get("accepted")),
                    "source_ref": str(row.get("source_ref") or ""),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self.evidence.get("exact_chain_audits") or []
                if isinstance(row, dict)
            ][:20],
            "visual_chains": [
                {
                    "accepted": bool(row.get("accepted") or row.get("exploratory_accepted")),
                    "source_ref": str(row.get("source_ref") or row.get("source_title") or ""),
                    "step_count": int(row.get("step_count") or len(row.get("steps") or row.get("candidate_steps") or [])),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self.evidence.get("visual_chains") or []
                if isinstance(row, dict)
            ][:20],
            "process_evidence_rows": [
                {
                    "source_ref": str(row.get("source_ref") or row.get("source_title") or ""),
                    "endpoint_labels": [str(x) for x in row.get("endpoint_labels") or []],
                    "local_pdf": str(row.get("local_pdf") or row.get("source_pdf_path") or ""),
                }
                for row in self.evidence.get("process_evidence_rows") or []
                if isinstance(row, dict)
            ][:20],
            "route_expansion_subgoals": [
                {
                    "title": str(row.get("name") or "subgoal closure"),
                    "accepted": bool(row.get("accepted")),
                    "step_count": len((row.get("route") or {}).get("steps") or []),
                    "route_rank": row.get("route_rank"),
                    "source_ref": str(row.get("raw_path") or row.get("search_path") or ""),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self._subgoal_route_records()
            ][:20],
        }

    def _run_trace(self) -> dict[str, Any]:
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        actions = [
            {
                "round_index": int(row.get("round_index") or 0),
                "action_type": str(row.get("action_type") or ""),
                "useful_artifact": bool(row.get("useful_artifact")),
                "reasons": [str(item) for item in row.get("reasons") or []],
            }
            for row in self.blackboard.get("action_history") or []
            if isinstance(row, dict)
        ]
        return {
            "schema_version": "route_forest_run_trace.v1",
            "run_dir": self.run_dir,
            "actions": actions[:80],
            "artifact_refs": [
                {"key": str(key), "path": str(value)}
                for key, value in sorted(artifact_refs.items())
                if str(value or "").strip()
            ][:120],
            "literature_counts": {
                "source_candidates": len(self.evidence.get("source_candidates") or []),
                "source_refs": len(self.evidence.get("source_refs") or []),
                "visual_chains": len(self.evidence.get("visual_chains") or []),
                "process_evidence_rows": len(self.evidence.get("process_evidence_rows") or []),
                "exact_rows": len(self.evidence.get("exact_rows") or []),
                "pdf_structure_evidence": len(self.evidence.get("pdf_structure_evidence") or []),
                "scout_attempts": len(self.evidence.get("scout_attempts") or []),
            },
        }


def _classify_synthesis_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify display branches from structured metadata, never target names."""
    classes: set[str] = set()
    evidence: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for path, raw_value in _structured_synthesis_markers(record, path=f"record[{index}]"):
            synthesis_class = _synthesis_class_for_marker(path.rsplit(".", 1)[-1], raw_value)
            if not synthesis_class or synthesis_class == "unspecified":
                continue
            classes.add(synthesis_class)
            evidence.append(f"{path}={raw_value}->{synthesis_class}")
    if "hybrid" in classes or len(classes - {"hybrid"}) > 1:
        synthesis_class = "hybrid"
    elif classes:
        synthesis_class = next(iter(classes))
    else:
        synthesis_class = "unspecified"
    return {
        "synthesis_class": synthesis_class,
        "classification_evidence": _dedupe(evidence)[:24],
        "classification_policy": "structured_metadata_only",
    }


def _structured_synthesis_markers(
    value: Any,
    *,
    path: str,
    depth: int = 0,
) -> list[tuple[str, str]]:
    if depth > 6:
        return []
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if str(key) in _SYNTHESIS_CLASS_FIELDS and isinstance(nested, (str, int, float)):
                text = str(nested or "").strip()
                if text:
                    out.append((child_path, text))
            elif isinstance(nested, (dict, list, tuple)):
                out.extend(_structured_synthesis_markers(nested, path=child_path, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list, tuple)):
                out.extend(
                    _structured_synthesis_markers(nested, path=f"{path}[{index}]", depth=depth + 1)
                )
    return out


def _synthesis_class_for_marker(field: str, raw_value: Any) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(raw_value or "").strip().lower()).strip("_")
    if not value:
        return ""
    if field == "synthesis_class" and value in _SYNTHESIS_CLASSES:
        return value
    if "total_synth" in value:
        return "total_synthesis"
    if "semisynth" in value:
        return "semisynthesis"
    if any(token in value for token in ("biosynth", "biotransformation", "fermentation")):
        return "biosynthesis"
    return ""


def _module_key_for_text(text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("heterocycle formation", "heterocycle synthesis", "pyrrole formation")):
        return "heterocycle_core_construction"
    if "ketal" in lowered:
        return "ketal_deprotection"
    if any(token in lowered for token in ("hydrolysis", "saponification")):
        return "ester_hydrolysis"
    if any(token in lowered for token in ("salt formation", "salt isolation", "salt metathesis")):
        return "salt_formation"
    if any(token in lowered for token in ("free acid", "form adjustment", "free base")):
        return "form_adjustment"
    if any(token in lowered for token in ("sidechain installation", "side-chain installation")):
        return "sidechain_installation"
    if any(token in lowered for token in ("sidechain", "side-chain", "anthranilate", "ester_to", "acid chloride", "amide")):
        return "amide_or_sidechain_assembly"
    if any(token in lowered for token in ("protect", "tes", "silyl", "deprotection", "deprotect")):
        return "protection_state_adjustment"
    if any(token in lowered for token in ("semisynthesis", "same-scaffold", "same_core")):
        return "semisynthesis_anchor"
    if any(token in lowered for token in ("core", "b ring", "cage", "scaffold", "ring system")):
        return "scaffold_core_construction"
    if any(token in lowered for token in ("visual", "scheme", "image")):
        return "visual_literature_hint"
    return "other_route_module"

def _display_target_name(raw_name: str, family_hint: str = "", case_id: str = "") -> str:
    name = str(raw_name or case_id or "").strip()
    if "_advisory" in name:
        name = name.split("_advisory", 1)[0]
    if "_fullflow" in name:
        name = name.split("_fullflow", 1)[0]
    return name or "target"


def _labels_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_labels_from_any(item))
        return _dedupe(out)
    if isinstance(value, dict):
        for key in ("label", "name", "smiles", "canonical_smiles"):
            if str(value.get(key) or "").strip():
                return [str(value.get(key))]
        return []
    text = str(value or "").strip()
    if not text:
        return []
    if "." in text and _looks_like_smiles(text):
        return [part for part in text.split(".") if part]
    return [text]


def _looks_like_smiles(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or " " in raw:
        return False
    if Chem is None:
        # Without the chemistry parser, failing closed is safer than attaching
        # a molecule identity to a human-readable compound name.
        return False
    try:
        return Chem.MolFromSmiles(raw) is not None
    except Exception:
        return False


def _deterministic_route_verifier_accepted(
    verifier: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Use the same fail-closed verifier contract as final-verdict consumers."""
    return bool(
        is_accepted_route_verifier_report(
            verifier,
            expected_target_smiles=expected_target_smiles,
        )
        and verifier.get("stock_audit_passed") is not False
    )


def _route_by_verified_rank(routes: list[dict[str, Any]], best_rank: Any) -> dict[str, Any]:
    """Select only the route explicitly named by a verifier; never fall back."""
    if best_rank is None:
        return {}
    rank = str(best_rank).strip()
    for candidate in routes:
        if isinstance(candidate, dict) and str(candidate.get("route_rank")).strip() == rank:
            return dict(candidate)
    return {}


def _materialized_route_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    product = str(step.get("product") or step.get("product_smiles") or "").strip()
    return bool(product and _route_step_reactants(step))


def _chain_rows_from_source_detail_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    chain_audit = dict(data.get("chain_audit") or {}) if isinstance(data.get("chain_audit"), dict) else {}
    nested_audit = (
        dict(data.get("source_detail_route_chain_audit") or {})
        if isinstance(data.get("source_detail_route_chain_audit"), dict)
        else {}
    )
    candidates = [
        (data.get("chain"), data.get("accepted") is True),
        (
            chain_audit.get("chain"),
            chain_audit.get("accepted") is True and data.get("accepted") is not False,
        ),
        (
            nested_audit.get("chain"),
            nested_audit.get("accepted") is True and data.get("accepted") is not False,
        ),
    ]
    for value, audit_accepted in candidates:
        if not audit_accepted:
            continue
        rows = [dict(row) for row in value or [] if isinstance(row, dict)]
        rows = [
            row
            for row in rows
            if str(row.get("product_smiles") or "").strip()
            and (row.get("reactant_smiles") or row.get("main_reactant_smiles") or row.get("reactants"))
        ]
        if rows:
            # Source-detail chains are stored in retrosynthetic order
            # (target-proximal first). The route forest displays synthesis
            # direction, ending at the requested target.
            return list(reversed(sorted(rows, key=lambda row: int(row.get("step_index") or 0))))
    return []


def _exact_node_label(value: Any, *, row: dict[str, Any], role: str) -> str:
    text = str(value or "").strip()
    if role == "exact_product":
        step_id = str(row.get("step_id") or row.get("source_template_id") or "").strip()
        if step_id:
            parts = step_id.split("_")
            if parts:
                suffix = parts[-1]
                if suffix and not suffix.isdigit():
                    return suffix.replace("-", " ")
    if _looks_like_smiles(text):
        return _compact_smiles_label(text)
    return _clean_label(text) or role


def _external_source_ref(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text.startswith(("idea:", "route_objective:", "target_side_", "broad_template")):
        return False
    return any(token in text for token in ("doi", "10.", "http", ".pdf", "science", "pubmed", "acs", "elsevier", "springer"))


def _structure_payload_for_smiles(smiles: Any) -> dict[str, Any]:
    text = str(smiles or "").strip()
    if not text:
        return {
            "structure_svg": "",
            "structure_valid": False,
            "structure_status": "no_smiles",
            "formula": "",
            "heavy_atom_count": None,
        }
    cached = _STRUCTURE_CACHE.get(text)
    if cached is not None:
        return dict(cached)
    out: dict[str, Any] = {
        "structure_svg": "",
        "structure_valid": False,
        "structure_status": "rdkit_unavailable" if Chem is None else "invalid_smiles",
        "formula": "",
        "heavy_atom_count": None,
    }
    if Chem is not None:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            try:
                if rdDepictor is not None:
                    rdDepictor.Compute2DCoords(mol)
            except Exception:
                pass
            out.update(
                {
                    "structure_svg": _mol_svg(mol),
                    "structure_valid": True,
                    "structure_status": "rendered",
                    "formula": rdMolDescriptors.CalcMolFormula(mol) if rdMolDescriptors is not None else "",
                    "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                }
            )
    _STRUCTURE_CACHE[text] = dict(out)
    return out


def _mol_svg(mol: Any, *, width: int = 240, height: int = 170) -> str:
    if rdMolDraw2D is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg
    except Exception:
        return ""


def _conditions_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        text = _condition_value_text(value)
        if text:
            out.append({"label": label, "value": text})

    candidate = row.get("condition_candidate")
    if isinstance(candidate, dict):
        add("试剂", candidate.get("reagent") or candidate.get("reagents"))
        add("催化剂", candidate.get("catalyst") or candidate.get("catalysts"))
        add("碱", candidate.get("base"))
        add("氧化剂", candidate.get("oxidant"))
        add("溶剂", candidate.get("solvent") or candidate.get("solvents"))
        add("温度", candidate.get("temperature"))
        add("时间", candidate.get("duration") or candidate.get("time"))
        add("收率", candidate.get("reported_yield") or candidate.get("yield"))
        add("条件原文", candidate.get("condition_text_transcribed"))
        add("来源依据", candidate.get("source_grounding") or candidate.get("source_excerpt"))

    if not out:
        # Legacy artifacts used several aliases before the visual-agent contract
        # standardized on condition_candidate.
        legacy_fields = (
            ("条件", row.get("condition_text")),
            ("反应条件", row.get("reaction_conditions")),
            ("可见条件", row.get("visible_conditions")),
            ("文献条件", row.get("source_grounded_conditions")),
            ("条件", row.get("conditions")),
            ("试剂", row.get("reagents") or row.get("reagent")),
            ("催化剂", row.get("catalysts") or row.get("catalyst")),
            ("溶剂", row.get("solvents") or row.get("solvent")),
            ("温度", row.get("temperature")),
            ("时间", row.get("duration") or row.get("time")),
            ("收率", row.get("yield") or row.get("yield_percent")),
        )
        for label, value in legacy_fields:
            add(label, value)
    locator = str(row.get("source_locator") or "").strip()
    if locator and any(token in locator.lower() for token in ("condition", "reagent", "arrow", "scheme")):
        add("来源位置", locator)
    return _normalize_condition_rows(out)


def _condition_value_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _condition_value_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = [_condition_value_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def _normalize_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = _clean_label(row.get("label") or "条件")[:36] or "条件"
        value = str(row.get("value") or "").strip()
        if not value:
            continue
        value = re.sub(r"\s+", " ", value)[:500]
        key = (label.lower(), value.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "value": value})
    return out[:8]


def _condition_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    parts = [f"{row.get('label')}: {row.get('value')}" for row in rows[:2]]
    text = "；".join(parts)
    return text[:240]


def _condition_status(rows: list[dict[str, str]], missing: list[str]) -> str:
    if rows:
        return "available"
    missing_text = " | ".join(str(item or "") for item in missing).lower()
    if "not shown" in missing_text or "no reaction condition" in missing_text or "conditions_not_shown" in missing_text:
        return "not_shown"
    if "condition" in missing_text or "conditions" in missing_text:
        return "not_compiled"
    return "not_recorded"


def _route_from_parent_route_proof(proof: dict[str, Any]) -> dict[str, Any]:
    if str(proof.get("proof_mode") or "") != "direct_parent_route":
        return {}
    evidence = dict(proof.get("proof_evidence") or {})
    parent_verifier = dict(evidence.get("parent_verifier") or {})
    embedded = parent_verifier.get("accepted_route")
    if isinstance(embedded, dict) and isinstance(embedded.get("steps"), list):
        return dict(embedded)
    return {}


def _revalidated_stitched_proof_projection(
    value: Any,
    *,
    expected_target_smiles: str,
) -> dict[str, Any]:
    """Return only chemistry reconstructed from a valid stitched proof input.

    This function intentionally ignores every route-shaped field outside
    ``proof_evidence.stitched_route.proof_inputs``.  Both the subgoal route and
    the literature chain are revalidated before any display node is created.
    """
    if not isinstance(value, dict) or not str(expected_target_smiles or "").strip():
        return {}
    proof = dict(value)
    if proof.get("proof_mode") != "stitched_parent_route" or not is_solved_parent_route_proof(
        proof,
        expected_target_smiles=expected_target_smiles,
    ):
        return {}
    evidence = proof.get("proof_evidence")
    if not isinstance(evidence, dict):
        return {}
    stitched = evidence.get("stitched_route")
    if not isinstance(stitched, dict):
        return {}
    inputs = stitched.get("proof_inputs")
    if not isinstance(inputs, dict) or inputs.get("schema_version") != "stitched_semisynthesis_proof_inputs.v1":
        return {}
    chain = inputs.get("literature_chain_audit")
    selected_subgoal = inputs.get("selected_subgoal")
    provided_verifier = inputs.get("provided_subgoal_verifier")
    raw = inputs.get("subgoal_raw_result")
    stored_expansion = inputs.get("route_expansion_result")
    if not isinstance(stored_expansion, dict):
        stored_expansion = {"subgoals": [dict(selected_subgoal or {})]}
    if not all(
        isinstance(item, dict)
        for item in (chain, selected_subgoal, provided_verifier, raw, stored_expansion)
    ):
        return {}

    recomputed = compile_stitched_semisynthesis_route(
        literature_chain_audit=dict(chain),
        route_expansion_result=dict(stored_expansion),
        subgoal_verifier=dict(provided_verifier),
        subgoal_raw_result=dict(raw),
        target_smiles=expected_target_smiles,
        target_name=str(inputs.get("target_name") or ""),
        case_id=str(inputs.get("case_id") or ""),
    )
    if not (
        recomputed.get("accepted") is True
        and recomputed.get("solved") is True
        and str(recomputed.get("route_status") or "") == "solved"
    ):
        return {}

    if not _same_molecule(str(chain.get("target_smiles") or ""), expected_target_smiles):
        return {}

    raw_literature_steps = chain.get("chain") or chain.get("steps") or []
    if not isinstance(raw_literature_steps, list):
        return {}
    literature_rows = [dict(row) for row in raw_literature_steps if isinstance(row, dict)]
    if not literature_rows or not all(
        is_validated_source_detail_literature_step(row) for row in literature_rows
    ):
        return {}
    literature_steps = _forward_synthesis_step_order(
        literature_rows,
        target_smiles=expected_target_smiles,
    )
    if not literature_steps:
        return {}
    if len(literature_steps) != int((recomputed.get("literature_chain") or {}).get("step_count") or 0):
        return {}
    if not _same_molecule(_route_step_product(literature_steps[-1]), expected_target_smiles):
        return {}

    frontier_smiles = _dedupe(
        [
            str(smiles)
            for smiles in (recomputed.get("literature_chain") or {}).get("graph_terminal_frontier") or []
            if str(smiles or "").strip()
        ]
    )
    coverage = dict(recomputed.get("frontier_coverage_audit") or {})
    if not (
        frontier_smiles
        and coverage.get("accepted") is True
        and int(coverage.get("frontier_count") or 0) == len(frontier_smiles)
        and int(coverage.get("closed_frontier_count") or 0) == len(frontier_smiles)
    ):
        return {}
    expansion_rows = [
        dict(row)
        for row in stored_expansion.get("subgoals") or []
        if isinstance(row, dict)
    ]
    subgoal_segments: list[dict[str, Any]] = []
    stock_terminal_smiles: list[str] = []
    recomputed_closures = [
        dict(row)
        for row in recomputed.get("subgoal_closures") or []
        if isinstance(row, dict)
    ]
    for frontier in frontier_smiles:
        candidates = [
            row
            for row in expansion_rows
            if _same_molecule(_subgoal_projection_target_smiles(row), frontier)
        ]
        if len(candidates) != 1:
            return {}
        candidate = candidates[0]
        candidate_raw = candidate.get("raw_result") or candidate.get("result")
        candidate_verifier = candidate.get("verifier")
        if not isinstance(candidate_raw, dict) or not isinstance(candidate_verifier, dict):
            return {}
        if not is_accepted_route_verifier_report(
            candidate_verifier,
            expected_target_smiles=frontier,
        ):
            return {}
        reverified = verify_chemenzy_raw_routes(
            dict(candidate_raw),
            target_smiles=frontier,
        )
        if not is_accepted_route_verifier_report(
            reverified,
            expected_target_smiles=frontier,
        ):
            return {}
        if (
            candidate_verifier.get("best_route_rank") != reverified.get("best_route_rank")
            or int(candidate_verifier.get("best_route_step_count") or 0)
            != int(reverified.get("best_route_step_count") or 0)
        ):
            return {}
        accepted_route = reverified.get("accepted_route")
        if not isinstance(accepted_route, dict):
            return {}
        subgoal_steps = _forward_synthesis_step_order(
            [
                dict(row)
                for row in accepted_route.get("steps") or []
                if isinstance(row, dict)
            ],
            target_smiles=frontier,
        )
        closure = next(
            (
                row
                for row in recomputed_closures
                if _same_molecule(
                    str((row.get("frontier") or {}).get("input_smiles") or ""),
                    frontier,
                )
            ),
            {},
        )
        if (
            not subgoal_steps
            or len(subgoal_steps) != int(closure.get("best_route_step_count") or 0)
            or not _same_molecule(_route_step_product(subgoal_steps[-1]), frontier)
        ):
            return {}
        product_keys = {
            _canonical_molecule_smiles(_route_step_product(row))
            for row in subgoal_steps
        }
        segment_stock = _dedupe(
            [
                smiles
                for row in subgoal_steps
                for smiles in _route_step_reactants(row)
                if _canonical_molecule_smiles(smiles) not in product_keys
            ]
        )
        if not segment_stock:
            return {}
        stock_terminal_smiles.extend(segment_stock)
        subgoal_segments.append(
            {
                "frontier_smiles": frontier,
                "steps": subgoal_steps,
                "stock_terminal_smiles": segment_stock,
            }
        )
    if (
        sum(len(segment["steps"]) for segment in subgoal_segments) + len(literature_steps)
        != int((recomputed.get("combined_route") or {}).get("combined_step_count") or 0)
    ):
        return {}
    return {
        "subgoal_segments": subgoal_segments,
        "literature_steps": literature_steps,
        "stock_terminal_smiles": _dedupe(stock_terminal_smiles),
        "literature_frontier_smiles": frontier_smiles,
        "literature_source_ref": str(chain.get("source_ref") or ""),
    }


def _subgoal_projection_target_smiles(row: dict[str, Any]) -> str:
    raw = row.get("raw_result") or row.get("result")
    raw_result = dict((raw or {}).get("result") or raw or {}) if isinstance(raw, dict) else {}
    verifier = dict(row.get("verifier") or {})
    audit = dict(verifier.get("target_equivalence_audit") or {})
    selected = dict(row.get("subgoal") or {})
    return str(
        row.get("frontier_smiles")
        or audit.get("request_canonical_isomeric_smiles")
        or audit.get("request_target_smiles")
        or raw_result.get("target")
        or raw_result.get("target_smiles")
        or selected.get("smiles")
        or ""
    ).strip()


def _forward_synthesis_step_order(
    steps: list[dict[str, Any]],
    *,
    target_smiles: str,
) -> list[dict[str, Any]]:
    """Topologically order retrosynthetic edges in forward synthesis order."""
    if not steps:
        return []
    by_product: dict[str, dict[str, Any]] = {}
    reactants_by_product: dict[str, list[str]] = {}
    for row in steps:
        product = _route_step_product(row)
        reactants = _route_step_reactants(row)
        product_key = _canonical_molecule_smiles(product)
        reactant_keys = [_canonical_molecule_smiles(item) for item in reactants]
        if (
            not product_key
            or not reactants
            or any(not item for item in reactant_keys)
            or product_key in by_product
        ):
            return []
        by_product[product_key] = dict(row)
        reactants_by_product[product_key] = reactant_keys

    target_key = _canonical_molecule_smiles(target_smiles)
    if not target_key or target_key not in by_product:
        return []
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(product_key: str) -> bool:
        if product_key in visited:
            return True
        if product_key in visiting:
            return False
        visiting.add(product_key)
        for reactant_key in reactants_by_product[product_key]:
            if reactant_key in by_product and not visit(reactant_key):
                return False
        visiting.remove(product_key)
        visited.add(product_key)
        ordered.append(dict(by_product[product_key]))
        return True

    if not visit(target_key) or len(visited) != len(by_product):
        return []
    return ordered


def _route_step_product(step: dict[str, Any]) -> str:
    product = str(
        step.get("product")
        or step.get("product_smiles")
        or step.get("final_product_smiles")
        or ""
    ).strip()
    if product:
        return product
    reaction_smiles = str(step.get("reaction_smiles") or "").strip()
    return reaction_smiles.split(">>", 1)[1].strip() if ">>" in reaction_smiles else ""


def _canonical_molecule_smiles(smiles: str) -> str:
    _, canonical = _molecule_node_identity(smiles=str(smiles or ""), label=str(smiles or ""))
    return canonical


def _route_step_reactants(step: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in (
        "main_reactant",
        "main_reactant_smiles",
        "reactant",
        "reactant_smiles",
        "precursor_smiles",
    ):
        value = step.get(key)
        if value:
            out.extend(_labels_from_any(value))
    for item in step.get("aux_reactants") or step.get("reactants") or []:
        out.extend(_labels_from_any(item))
    reaction_smiles = str(step.get("reaction_smiles") or "").strip()
    if not out and ">>" in reaction_smiles:
        left = reaction_smiles.split(">>", 1)[0]
        out.extend(_labels_from_any(left))
    return _dedupe([item for item in out if item])


def _confidence_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.65:
        return "medium_high"
    if score >= 0.4:
        return "medium"
    return "low"


def _route_step_confidence(step: dict[str, Any]) -> str:
    scores = dict(step.get("scores") or {})
    raw = scores.get("confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is None:
        return "medium"
    if value >= 0.75:
        return "high"
    if value >= 0.35:
        return "medium"
    return "low"


def _consensus_precursor_smiles(proposal: dict[str, Any]) -> list[str]:
    raw = proposal.get("precursor_smiles")
    if isinstance(raw, str):
        values = raw.split(".")
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = []
    return _dedupe([str(value).strip() for value in values if str(value).strip()])[:12]


def _consensus_support_group(source_channel: str, support_group: str) -> str:
    channel = str(source_channel or "").strip().lower()
    group = str(support_group or "").strip()
    if channel.startswith("codex_") or group.lower().startswith("codex"):
        return "codex_model"
    return group or (f"source:{channel}" if channel else "source:unbound")


def _consensus_support_records(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in proposal.get("source_records") or []:
        if not isinstance(raw, dict):
            continue
        source_channel = str(raw.get("source_channel") or "other")
        records.append(
            {
                "candidate_id": str(raw.get("candidate_id") or ""),
                "source_channel": source_channel,
                "evidence_level": str(raw.get("evidence_level") or "model_only"),
                "confidence": str(raw.get("confidence") or "low"),
                "support_group": _consensus_support_group(
                    source_channel,
                    str(raw.get("support_group") or ""),
                ),
                "source_refs": _dedupe([str(item) for item in raw.get("source_refs") or []]),
                "evidence_refs": _dedupe([str(item) for item in raw.get("evidence_refs") or []]),
            }
        )
    return records[:32]


def _consensus_independent_support_groups(
    proposal: dict[str, Any],
    support_records: list[dict[str, Any]],
) -> list[str]:
    groups = _dedupe([str(row.get("support_group") or "") for row in support_records])
    if not groups:
        groups = _dedupe(
            [
                _consensus_support_group("", str(group))
                for group in proposal.get("independent_support_groups") or []
            ]
        )
    if not groups:
        groups = _dedupe(
            [
                _consensus_support_group(str(channel), "")
                for channel in proposal.get("source_channels") or []
            ]
        )
    # Collapse any malformed producer output that counted individual Codex
    # roles as independent evidence.
    return _dedupe(["codex_model" if str(group).lower().startswith("codex") else group for group in groups])


def _consensus_direct_source_refs(proposal: dict[str, Any]) -> list[str]:
    values = [
        *[str(item) for item in proposal.get("source_refs") or []],
        *[str(item) for item in proposal.get("evidence_refs") or []],
    ]
    for record in _consensus_support_records(proposal):
        values.extend(str(item) for item in record.get("source_refs") or [])
        values.extend(str(item) for item in record.get("evidence_refs") or [])
    return _dedupe(values)[:32]


def _consensus_conflicts(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts = [
        dict(row) for row in proposal.get("condition_conflicts") or [] if isinstance(row, dict)
    ]
    reaction_families = _dedupe([str(item) for item in proposal.get("reaction_families") or []])
    if len(reaction_families) > 1:
        conflicts.append(
            {
                "field": "reaction_family",
                "values": reaction_families,
                "requires_review": True,
            }
        )
    return conflicts[:16]


def _same_molecule(left: str, right: str) -> bool:
    left_id, left_canonical = _molecule_node_identity(smiles=left, label=left)
    right_id, right_canonical = _molecule_node_identity(smiles=right, label=right)
    if left_canonical and right_canonical:
        return left_id == right_id
    return _same_text(left, right)


def _same_text(left: str, right: str) -> bool:
    return str(left or "").strip() == str(right or "").strip()


def _compact_smiles_label(smiles: str, *, max_len: int = 58) -> str:
    text = str(smiles or "").strip()
    if len(text) <= max_len:
        return text
    keep = max(12, (max_len - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def _template_endpoint_labels(template: dict[str, Any]) -> tuple[str, str]:
    """Return advisory endpoint labels without implying a target connection."""
    logic = _clean_label(template.get("transform_logic") or "")
    for separator in ("->", "→", "=>"):
        if separator not in logic:
            continue
        left, right = (part.strip() for part in logic.split(separator, 1))
        if left and right:
            return left, right
    precursor = _clean_label(
        template.get("reactant_label")
        or template.get("from_label")
        or template.get("preserved_scaffold")
        or "unbound template precursor"
    )
    product = _clean_label(
        template.get("product_label")
        or template.get("to_label")
        or f"unbound template product: {template.get('objective_type') or 'unspecified'}"
    )
    return precursor, product


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:90] or "item"


def _display_text_is_corrupt(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    mojibake_markers = (
        "鎺",
        "璺",
        "妫",
        "鍥",
        "鏂",
        "閰",
        "姣嶆",
        "涓荤",
        "氱偣",
        "囬€",
        "惧儚",
    )
    return "�" in text or "\ufffd" in text or any(marker in text for marker in mojibake_markers)


def _branch_title_for_display(*, branch_id: str, title: str, kind: str) -> str:
    if not _display_text_is_corrupt(title):
        return str(title or "")
    return {
        "recommended_strategy": "推荐路线",
        "visual_chain": "图像证据分支",
        "stitched_verified_route": "拼接验证路线",
        "process_evidence": "文献工艺锚点",
        "route_consensus": "多信源共识候选",
        "retrosynthetic_proposal": "备选逆合成分支",
        "broad_template": "通用模板分支",
        "direct_verified_route": "已验证路线",
        "subgoal_verified_route": "子目标闭合路线",
        "exact_literature": "exact row 路线",
        "diagnostic_failure": "探索诊断",
    }.get(str(kind or ""), str(title or "route branch"))


def _recommendation_for_display(*, kind: str, recommendation: str) -> str:
    if not _display_text_is_corrupt(recommendation):
        return str(recommendation or "")
    return {
        "recommended_strategy": "主推荐",
        "stitched_verified_route": "拼接验证",
        "visual_chain": "支持/备选",
        "process_evidence": "工艺锚点",
        "route_consensus": "仅建议",
        "retrosynthetic_proposal": "探索备选",
        "broad_template": "模板提示",
        "direct_verified_route": "已验证",
        "subgoal_verified_route": "子目标闭合",
        "exact_literature": "强证据",
        "diagnostic_failure": "诊断",
    }.get(str(kind or ""), str(recommendation or ""))


def _module_label_for_key(key: str) -> str:
    return {
        "sidechain_installation": "侧链安装",
        "amide_or_sidechain_assembly": "酰胺 / 侧链连接",
        "protection_state_adjustment": "保护基 / 脱保护调整",
        "semisynthesis_anchor": "半合成锚点",
        "heterocycle_core_construction": "杂环母核构建",
        "scaffold_core_construction": "骨架构建 / 母核调整",
        "visual_literature_hint": "图像文献提示",
        "other_route_module": "其他路线模块",
        "ketal_deprotection": "缩酮脱保护",
        "ester_hydrolysis": "酯水解",
        "salt_formation": "成盐 / 分离",
        "form_adjustment": "盐型 / 游离酸调整",
        "subgoal_stock_closure": "ChemEnzy 子目标闭合",
        "diagnostic_failure": "诊断失败",
        "visual_failed_or_empty": "图像链失败或为空",
    }.get(str(key or ""), str(key or ""))


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text[:180]


def _molecule_node_identity(*, smiles: str, label: str) -> tuple[str, str]:
    """Return a collision-resistant ID while preserving stereochemistry."""
    text = str(smiles or "").strip()
    canonical = ""
    if text and Chem is not None:
        try:
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            canonical = ""
    identity = f"smiles:{canonical}" if canonical else f"name:{_clean_label(label).casefold()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"mol:{digest}", canonical


def _exact_row_is_verified(row: dict[str, Any]) -> bool:
    if row.get("accepted") is False:
        return False
    return is_validated_source_detail_literature_step(row)


def _normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    if text in CONFIDENCE_RANK:
        return text
    if "high" in text:
        return "high"
    if "low" in text:
        return "low"
    if "fail" in text or "reject" in text:
        return "failed"
    return "medium"


def _normalize_exactness(value: Any) -> str:
    text = str(value or "name_only").strip().lower()
    if text in EXACTNESS_RANK:
        return text
    if "exact" in text:
        return "exact_literature_row"
    if "visual" in text:
        return "visual_inferred"
    if "hypothesis" in text or "model" in text:
        return "model_hypothesis"
    if "fail" in text or "unresolved" in text:
        return "failed_or_unresolved"
    if "literature" in text or "named" in text:
        return "named_literature"
    return "name_only"


def _best_ranked(a: Any, b: Any, rank: dict[str, int]) -> str:
    av = str(a or "")
    bv = str(b or "")
    return av if rank.get(av, 0) >= rank.get(bv, 0) else bv


def _worst_ranked(a: Any, b: Any, rank: dict[str, int]) -> str:
    av = str(a or "")
    bv = str(b or "")
    return av if rank.get(av, 0) <= rank.get(bv, 0) else bv


def _better_node_label(existing: Any, new: Any) -> str:
    old = _clean_label(existing)
    fresh = _clean_label(new)
    if not old:
        return fresh or "unnamed node"
    if not fresh:
        return old
    generic_prefixes = ("proposal precursor", "visual precursor", "visual product", "target", "unnamed node")
    old_generic = old.lower().startswith(generic_prefixes)
    fresh_generic = fresh.lower().startswith(generic_prefixes)
    if old_generic and not fresh_generic:
        return fresh
    if fresh_generic and not old_generic:
        return old
    if "/" in old and "/" not in fresh:
        return old
    if "/" in fresh and "/" not in old:
        return fresh
    return old if len(old) >= len(fresh) else fresh


def _preferred_node_role(roles: list[str]) -> str:
    priority = {
        "target": 100,
        "literature_terminal": 90,
        "stock_terminal": 80,
        "literature_intermediate": 60,
        "stitched_route_intermediate": 50,
    }
    cleaned = [str(role) for role in roles if str(role or "").strip()]
    if not cleaned:
        return "intermediate"
    return max(cleaned, key=lambda role: priority.get(role, 0))


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _escape(text: str) -> str:
    return html.escape(text, quote=True)
