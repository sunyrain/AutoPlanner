(() => {
  'use strict';

  const forestDataText = document.getElementById('forest-data')?.textContent || '{}';
  const forest = JSON.parse(forestDataText);
  const STORAGE_KEY = `autoplanner.route-forest-ui.v3:${forest.case_id || forest.target?.name || 'route'}`;
  const LEGACY_STORAGE_KEY = 'autoplanner.route-forest-ui.v2';
  const PAN_DRAG_THRESHOLD_PX = 5;
  const BRANCH_LANE_SCHEMA_VERSION = 'route_forest_branch_lanes.v2';
  const BRANCH_STAGE_EVIDENCE_SCHEMA_VERSION = 'route_forest_branch_stage_evidence.v2';
  const COPY = Object.freeze({
    consensus: 'Consensus evidence audit',
    support: 'Independent support groups',
    conflicts: 'Condition conflicts',
    correlated: 'Codex roles are correlated',
    noReplacement: 'No backend AND/OR-revalidated replacement is available.',
    noSplice: 'Pairwise interface comparisons never enable a single-step splice.',
    explicitEdges: 'Array adjacency never creates an edge.',
    resolvedReplacement: 'full AND/OR route re-solved',
    noClosedRoute: 'no_stock_closed_reaction_validated_route'
  });
  const PROOF_ORDER = [
    'L4_procurement_ready', 'L3_precedent_supported', 'L2_reaction_validated',
    'L2_mapping_consistent', 'L1_graph_stock_closed', 'L1_graph_and_stock_closed',
    'L0_materialized', 'L0_advisory', 'L0_rejected'
  ];
  const PROOF_LABEL = {
    L4_procurement_ready: 'L4 可采购',
    L3_precedent_supported: 'L3 精确先例',
    L2_reaction_validated: 'L2 反应重验',
    L2_mapping_consistent: 'L2 映射一致',
    L1_graph_stock_closed: 'L1 图与库存闭合',
    L1_graph_and_stock_closed: 'L1 图与库存闭合',
    L0_materialized: 'L0 结构已具象',
    L0_advisory: 'L0 探索建议',
    L0_rejected: 'L0 已拒绝'
  };
  const TIER_CLASS = {
    L4_procurement_ready: 'tier-l4',
    L3_precedent_supported: 'tier-l3',
    L2_reaction_validated: 'tier-l2-validated',
    L2_mapping_consistent: 'tier-l2-mapping',
    L1_graph_stock_closed: 'tier-l1',
    L1_graph_and_stock_closed: 'tier-l1',
    L0_materialized: 'tier-l0-materialized',
    L0_advisory: 'tier-l0-advisory',
    L0_rejected: 'tier-l0-rejected'
  };
  const TIER_COLOR = {
    L4_procurement_ready: '#15803d', L3_precedent_supported: '#0f766e',
    L2_reaction_validated: '#2563eb', L2_mapping_consistent: '#64748b',
    L1_graph_stock_closed: '#a16207', L1_graph_and_stock_closed: '#a16207',
    L0_materialized: '#7c3aed', L0_advisory: '#ea580c', L0_rejected: '#be123c'
  };
  const graph = forest.dependency_graph || {};
  const layout = forest.dependency_layout || {};
  const lanesProjection = forest.branch_lanes || {};
  const frontierLedger = forest.frontier_ledger || forest.semantic_summary?.frontier_ledger || {};
  const graphNodes = new Map((graph.nodes || []).map(row => [row.graph_node_id, row]));
  const moleculeNodes = new Map((forest.nodes || []).map(row => [row.node_id, row]));
  const steps = new Map((forest.steps || []).map(row => [row.step_id, row]));
  const branches = new Map((forest.branches || []).map(row => [row.branch_id, row]));
  const laneByBranch = new Map((lanesProjection.lanes || []).map(row => [row.branch_id, row]));
  const edgeById = new Map((graph.edges || []).map(row => [row.edge_id, row]));
  const layoutByNode = new Map((layout.nodes || []).map(row => [row.graph_node_id, row]));
  const persisted = loadState();
  const legacyChrome = loadState(LEGACY_STORAGE_KEY);
  const allProofTiers = unique((lanesProjection.lanes || []).map(row => row.proof_tier).filter(Boolean));
  const allKinds = unique((lanesProjection.lanes || []).map(row => row.kind).filter(Boolean));
  const defaultBranchId = chooseDefaultBranchId();
  const initialBranchId = persisted.selectedBranchId && branches.has(persisted.selectedBranchId)
    ? persisted.selectedBranchId : defaultBranchId;

  const state = {
    mode: oneOf(persisted.mode, ['clusters', 'shared', 'current'], 'current'),
    selectedBranchId: initialBranchId,
    selectedGraphNodeId: '',
    selectedInstanceId: '',
    selectedStepId: '',
    detailTab: oneOf(persisted.detailTab, ['step', 'evidence', 'alternatives'], 'step'),
    query: '',
    stageFilter: oneOf(persisted.stageFilter, ['all', 'suggestion', 'expanded', 'reaction', 'stock'], 'all'),
    branchFilter: oneOf(persisted.branchFilter, ['all', 'verified', 'evidence', 'advisory', 'diagnostic'], 'all'),
    proofFilters: new Set(Array.isArray(persisted.proofFilters) ? persisted.proofFilters.filter(tier => allProofTiers.includes(tier)) : allProofTiers),
    kindFilters: new Set(Array.isArray(persisted.kindFilters) ? persisted.kindFilters.filter(kind => allKinds.includes(kind)) : allKinds),
    edgeFilter: oneOf(persisted.edgeFilter, ['all', 'selected'], 'all'),
    orientation: oneOf(persisted.orientation, ['horizontal', 'vertical'], 'horizontal'),
    density: oneOf(persisted.density, ['comfortable', 'compact', 'overview'], 'comfortable'),
    edgeStyle: oneOf(persisted.edgeStyle, ['trust', 'simple', 'contrast'], 'trust'),
    labelMode: oneOf(persisted.labelMode, ['semantic', 'full', 'minimal'], 'semantic'),
    layoutPreset: oneOf(persisted.layoutPreset, ['explore', 'focus', 'review'], 'explore'),
    theme: oneOf(persisted.theme || legacyChrome.theme, ['light', 'dark'], preferredTheme()),
    navOpen: Object.hasOwn(persisted, 'navOpen') ? persisted.navOpen !== false : !matchMedia('(max-width: 1023px)').matches,
    inspectorOpen: Object.hasOwn(persisted, 'inspectorOpen') ? persisted.inspectorOpen !== false : false,
    navWidth: clamp(Number(persisted.navWidth || legacyChrome.navWidth) || 280, 240, 460),
    inspectorWidth: clamp(Number(persisted.inspectorWidth || legacyChrome.inspectorWidth) || 380, 320, 560),
    zoom: 1,
    panX: 0,
    panY: 0,
    showAllOverview: false,
    expandedGroups: new Set(),
    activeReplacement: null
  };
  let renderModel = null;
  let panSession = null;
  let panAnimationFrame = 0;
  let suppressGraphClickPointerId = null;
  let suppressGraphClickTimer = 0;
  let resizeSession = null;
  let liveAnnouncementTimer = 0;
  let deliveryIntegrityStatus = 'pending';

  if (new URLSearchParams(location.search).get('embed') === '1') {
    document.body.classList.add('embedded-route');
  }

  function element(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
  }
  function clamp(value, minimum, maximum) { return Math.min(maximum, Math.max(minimum, value)); }
  function unique(values) { return [...new Set(values)]; }
  function oneOf(value, allowed, fallback) { return allowed.includes(value) ? value : fallback; }
  function stableTextCompare(left, right) {
    const a = String(left), b = String(right);
    return a < b ? -1 : a > b ? 1 : 0;
  }
  function branchDisplayScore(lane) {
    const branch = branches.get(lane?.branch_id) || {};
    if (branch.solved === true && branch.executable === true && branch.advisory_only === false) return 10000;
    const kindScore = {
      stitched_verified_route: 900, direct_verified_route: 860,
      proof_eligible_portfolio_route: 760, exact_literature: 700,
      subgoal_verified_route: 640, process_evidence: 520,
      route_consensus_graph: 420, visual_chain: 360,
      route_consensus: 300, literature_candidate: 280,
      retrosynthetic_proposal: 140, broad_template: 80,
      diagnostic_failure: -200
    }[lane?.kind] ?? 0;
    const confidenceScore = { high: 40, medium_high: 30, medium: 20, low: 10, failed: -40 }[
      branch.confidence || lane?.confidence
    ] || 0;
    const proofScore = Math.max(0, Number(lane?.proof_rank ?? -1)) * 160;
    const stockEvidence = lane?.stage_evidence?.stock || {};
    const stockScore = stockEvidence.member === true
      ? (stockEvidence.closure_scope === 'procurement' ? 700 : 500) : 0;
    const primaryScore = lane?.is_primary ? 120 : 0;
    const stepScore = Math.min(8, (lane?.step_ids || []).length) * 16;
    const sourceScore = Math.min(8, (lane?.source_refs || []).length) * 16;
    const structureScore = Math.min(12, (lane?.graph_node_ids || []).filter(graphNodeId => {
      const graphNode = graphNodes.get(graphNodeId) || {};
      return Boolean(moleculeNodes.get(graphNode.molecule_node_id)?.structure_svg);
    }).length) * 28;
    const failedPenalty = /failed|diagnostic|rejected/i.test(`${lane?.title || ''} ${branch.summary || ''}`) ? 80 : 0;
    return kindScore + proofScore + stockScore + primaryScore + confidenceScore
      + stepScore + sourceScore + structureScore - failedPenalty;
  }
  function chooseDefaultBranchId() {
    const primaryId = String(forest.primary_branch_id
      || (lanesProjection.lanes || []).find(row => row.is_primary)?.branch_id || '');
    const primary = branches.get(primaryId) || {};
    if (forest.primary_selection?.display_tiebreak_only && primaryId) return primaryId;
    if (primary.solved === true && primary.executable === true && primary.advisory_only === false) return primaryId;
    const featured = (lanesProjection.lanes || [])
      .filter(row => row.listed !== false)
      .slice()
      .sort((left, right) => branchDisplayScore(right) - branchDisplayScore(left)
        || stableTextCompare(left.branch_id, right.branch_id))[0];
    return featured?.branch_id || primaryId || (lanesProjection.lanes || [])[0]?.branch_id || '';
  }
  function safeStructureSvg(value) {
    const svg = String(value || '').trim();
    if (!svg.startsWith('<svg') || /<script\b|<foreignObject\b|\son\w+\s*=|javascript:/i.test(svg)) return '';
    return svg;
  }
  function looksLikeSmiles(value, node) {
    const text = String(value || '').trim();
    const canonical = String(node?.canonical_isomeric_smiles || '').trim();
    return Boolean(text && ((canonical && text === canonical)
      || (text.length > 24 && !/\s/.test(text) && /[()[\]@=#\\/]/.test(text))));
  }
  function moleculeCaption(node, fallback) {
    const label = String(fallback || '').trim();
    if (label && !looksLikeSmiles(label, node)) return middleEllipsis(label, 30);
    const role = ({
      target: '目标产物', visual_precursor: '文献前体', visual_intermediate: '文献中间体',
      visual_product: '文献产物', consensus_precursor: '共识前体',
      consensus_graph_precursor: '共识前体', consensus_graph_intermediate: '共识中间体',
      consensus_graph_product: '共识产物', stock: '库存原料',
      starting_material: '起始原料', intermediate: '路线中间体'
    })[node?.role] || String(node?.role || '路线分子').replaceAll('_', ' ');
    return node?.formula ? `${role} · ${node.formula}` : role;
  }
  function safeStorageGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function safeStorageSet(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* sandboxed embed */ } }
  function loadState(key = STORAGE_KEY) {
    try { return JSON.parse(safeStorageGet(key) || '{}'); } catch (_) { return {}; }
  }
  function persistState() {
    safeStorageSet(STORAGE_KEY, JSON.stringify({
      mode: state.mode, selectedBranchId: state.selectedBranchId, detailTab: state.detailTab,
      stageFilter: state.stageFilter, branchFilter: state.branchFilter,
      proofFilters: [...state.proofFilters],
      kindFilters: [...state.kindFilters], edgeFilter: state.edgeFilter,
      orientation: state.orientation, density: state.density, edgeStyle: state.edgeStyle,
      labelMode: state.labelMode, layoutPreset: state.layoutPreset, theme: state.theme,
      navOpen: state.navOpen, inspectorOpen: state.inspectorOpen,
      navWidth: state.navWidth, inspectorWidth: state.inspectorWidth
    }));
  }
  function preferredTheme() {
    return matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function tierOfStep(step) { return step?.trust_vector?.proof_tier || 'L0_advisory'; }
  function tierClass(tier) { return TIER_CLASS[tier] || 'tier-l0-advisory'; }
  function tierLabel(tier) { return PROOF_LABEL[tier] || tier || '未分级'; }
  function targetName() {
    const raw = String(forest.target?.name || forest.case_id || '目标分子');
    return raw.replace(/(?:[\s_-]+full)?[\s_-]+rerun(?:[\s_-].*)?$/i, '').replaceAll('_', ' ');
  }
  function basename(value) { return String(value || '').split(/[\\/]/).filter(Boolean).pop() || String(value || ''); }
  function synthesisLabel(value) {
    return ({
      biosynthesis: '生物合成 / 生物转化', semisynthesis: '半合成',
      total_synthesis: '全合成', hybrid: '混合路线', unspecified: '未分类'
    })[value] || value || '未分类';
  }
  function isTypingTarget(target) {
    return Boolean(target?.closest?.('input, textarea, select, [contenteditable="true"]'));
  }
  function middleEllipsis(value, limit = 30) {
    const text = String(value || '未命名');
    if (state.labelMode === 'full' || text.length <= limit) return text;
    const left = Math.ceil((limit - 1) / 2);
    return `${text.slice(0, left)}…${text.slice(-(limit - 1 - left))}`;
  }
  function wrapLabel(value, limit = 24) {
    const text = state.labelMode === 'minimal' ? '' : middleEllipsis(value, state.labelMode === 'full' ? 52 : limit);
    if (!text) return [];
    if (text.length <= limit) return [text];
    const split = Math.min(limit, Math.ceil(text.length / 2));
    return [text.slice(0, split), text.slice(split, split + limit)];
  }
  function svgTextLines(lines, x, y, lineHeight = 15) {
    return lines.map((line, index) => `<tspan x="${x}" dy="${index ? lineHeight : 0}">${esc(line)}</tspan>`).join('');
  }

  function filteredLanes({ includeReplacementPreview = false } = {}) {
    const query = state.query.trim().toLocaleLowerCase();
    return (lanesProjection.lanes || []).filter(lane => {
      const isReplacementPreview = includeReplacementPreview
        && lane.branch_id === state.activeReplacement?.replacementBranchId;
      if (lane.listed === false && !isReplacementPreview) return false;
      if (isReplacementPreview) return true;
      if (!laneMatchesStage(lane)) return false;
      if (state.branchFilter !== 'all' && lane.category !== state.branchFilter) return false;
      if (!state.proofFilters.has(lane.proof_tier)) return false;
      if (!state.kindFilters.has(lane.kind)) return false;
      if (!query) return true;
      const branch = branches.get(lane.branch_id) || {};
      const stepText = (lane.step_ids || []).flatMap(id => {
        const step = steps.get(id) || {};
        return [step.label, step.module_label, ...(step.source_refs || [])];
      });
      const nodeText = (lane.graph_node_ids || []).map(id => graphNodes.get(id)?.label || '');
      return [lane.title, lane.kind_label, lane.kind, lane.synthesis_class,
        ...(lane.source_refs || []), branch.summary, ...stepText, ...nodeText]
        .join(' ').toLocaleLowerCase().includes(query);
    });
  }

  function laneMatchesStage(lane) {
    if (state.stageFilter === 'all') return true;
    // Older lane payloads have no per-record authority.  They deliberately
    // fail closed: proof tiers, step counts, and stock aliases are styling or
    // aggregate hints and cannot reconstruct expanded/reaction/stock truth.
    return stageMembershipIsAuthoritative(lane, state.stageFilter);
  }

  function renderChrome({ refreshIntegrity = true } = {}) {
    const primary = forest.primary_selection || {};
    const primaryBranch = branches.get(primary.primary_branch_id || forest.primary_branch_id) || {};
    const verified = primary.status === 'deterministically_verified'
      && primary.proof_level === 'parent_route_proof'
      && primary.advisory_only === false
      && primaryBranch.solved === true
      && primaryBranch.executable === true
      && primaryBranch.advisory_only === false
      && primaryBranch.not_parent_route_proof === false;
    const deliveryBytesVerified = deliveryIntegrityStatus === 'verified';
    const ledgerAuthoritative = deliveryBytesVerified && frontierLedger.authoritative === true;
    const closure = frontierLedger.closure || {};
    const anyRouteClosed = ledgerAuthoritative && closure.any_benchmark_route_closed === true;
    const allGraphClosed = ledgerAuthoritative && closure.all_explored_benchmark_closed === true;
    const anyProcurementClosed = ledgerAuthoritative && closure.any_procurement_route_closed === true;
    const parentL3Solved = deliveryBytesVerified && verified && closure.l3_parent_solved === true;
    const procurementL4Ready = parentL3Solved
      && anyProcurementClosed
      && closure.l4_procurement_ready === true;
    element('pageTitle').textContent = `${targetName()} · 路线工作台`;
    const verdict = element('verdictBadge');
    verdict.textContent = procurementL4Ready ? 'L4 采购就绪'
      : parentL3Solved ? 'L3 父路线已解'
        : allGraphClosed ? 'Benchmark 全探索闭合'
          : anyRouteClosed ? '存在 Benchmark 闭合路线' : '父路线未闭合';
    const verdictState = procurementL4Ready || parentL3Solved
      ? 'verified' : anyRouteClosed || allGraphClosed ? 'partial' : 'unresolved';
    verdict.dataset.status = verdictState;
    verdict.classList.toggle('status-badge--verified', verdictState === 'verified');
    verdict.classList.toggle('status-badge--partial', verdictState === 'partial');
    verdict.classList.toggle('status-badge--unresolved', verdictState === 'unresolved');
    const agentTasks = forest.semantic_summary?.agent_tasks || {};
    const ledgerCounts = frontierLedger.counts || {};
    const ledgerValue = key => ledgerAuthoritative ? Number(ledgerCounts[key] || 0) : '—';
    const overviewRows = [
      ['Agent 完成', `${agentTasks.completed || 0}/${agentTasks.total || 0}`],
      ['L0 断键边', ledgerValue('l0_break_suggestion_edges')],
      ['已展开 work', ledgerValue('expanded_work_molecules')],
      ['L2 反应边', ledgerValue('l2_reaction_edges')],
      ['L3 先例边', ledgerValue('l3_precedent_edges')],
      ['库存边界叶', ledgerValue('stock_closed_leaves')]
    ];
    if (primary.display_tiebreak_only && Number(primary.tied_candidate_count || 0) > 1) {
      overviewRows.push(['同分候选', Number(primary.tied_candidate_count)]);
    }
    element('overviewMetrics').innerHTML = overviewRows
      .map(([label, value]) => `<span class="metric-chip"><strong>${esc(value)}</strong>${esc(label)}</span>`).join('');
    renderClosureStatus({
      verified,
      parentL3Solved,
      procurementL4Ready,
      ledgerAuthoritative,
      deliveryBytesVerified
    });
    if (refreshIntegrity) renderIntegrityStatus('pending');
    renderEvidenceStats();
    applyPersistentChromeState();
  }

  function renderClosureStatus({
    verified,
    parentL3Solved,
    procurementL4Ready,
    ledgerAuthoritative,
    deliveryBytesVerified
  }) {
    const authoritative = ledgerAuthoritative === true;
    const closure = frontierLedger.closure || {};
    const counts = frontierLedger.counts || {};
    const badge = element('ledgerAuthorityBadge');
    const digest = String(frontierLedger.content_sha256 || '');
    const reasons = (frontierLedger.validation_reasons || []).map(String);
    badge.dataset.authoritative = String(authoritative);
    badge.dataset.integrity = deliveryIntegrityStatus;
    badge.textContent = authoritative
      ? `账本当前复验通过 · 交付仅字节完整 ${digest.slice(0, 10)}`
      : !deliveryBytesVerified
        ? deliveryIntegrityStatus === 'pending'
          ? '正在校验交付字节 · 结论锁定'
          : '交付字节完整性未验证 · fail-closed'
        : '无有效账本 · 结论 fail-closed';
    badge.title = authoritative
      ? `frontier_ledger.v1 · ${frontierLedger.source_ref || 'embedded'}`
      : !deliveryBytesVerified
        ? `delivery_integrity_status:${deliveryIntegrityStatus}`
        : reasons.join(' · ') || 'frontier_ledger_artifact_missing';

    const countValue = key => authoritative ? Number(counts[key] || 0) : '—';
    const ratioValue = (numerator, denominator) => authoritative
      ? `${Number(counts[numerator] || 0)}/${Number(counts[denominator] || 0)}` : '—';
    const stockDetail = authoritative
      ? `搜索闭合叶；其中 benchmark ${Number(counts.benchmark_only_stock_leaves || 0)}，采购边界 ${Number(counts.procurement_boundary_leaves || 0)}。benchmark 不等于可采购。`
      : '缺少有效 frontier ledger；库存与采购均不作正向声明';
    const progress = [
      ['L0 断键建议', countValue('l0_break_suggestion_edges'), '未达到 L2 的精确候选反应边'],
      ['已展开 work', ratioValue('expanded_work_molecules', 'reachable_molecules'), '已成功完成 proposal expansion 的分子 frontier'],
      ['L2 反应验证', countValue('l2_reaction_edges'), '当前 host verifier 接受的 L2 反应边'],
      ['L3 精确先例', countValue('l3_precedent_edges'), '绑定精确文献先例的 L3 反应边'],
      ['搜索库存叶', ratioValue('stock_closed_leaves', 'reachable_leaves'), stockDetail]
    ];
    element('ledgerProgressMetrics').innerHTML = progress.map(([label, value, detail]) => `
      <span class="ledger-progress-chip" data-authoritative="${String(authoritative)}" title="${esc(detail)}">
        <span>${esc(label)}</span><strong>${esc(value)}</strong>
      </span>`).join('');

    const ledgerState = value => authoritative ? (value === true ? 'closed' : 'open') : 'unknown';
    const ledgerValue = (value, positive, negative) => authoritative
      ? (value === true ? positive : negative) : '账本缺失';
    const cards = [
      {
        label: 'ANY BENCHMARK ROUTE',
        value: ledgerValue(closure.any_benchmark_route_closed, '存在搜索闭合路线', '尚无搜索闭合路线'),
        state: ledgerState(closure.any_benchmark_route_closed),
        detail: '至少一条 AND/OR 路径达到 benchmark-search 闭合；不代表采购闭合'
      },
      {
        label: 'ALL BENCHMARK GRAPH',
        value: ledgerValue(closure.all_explored_benchmark_closed, 'Benchmark 全探索闭合', '仍有搜索开放分支'),
        state: ledgerState(closure.all_explored_benchmark_closed),
        detail: '全部可达反应边与叶节点通过 benchmark-search 固定点闭合'
      },
      {
        label: 'ANY PROCUREMENT ROUTE',
        value: ledgerValue(closure.any_procurement_route_closed, '存在采购闭合路线', '尚无采购闭合路线'),
        state: ledgerState(closure.any_procurement_route_closed),
        detail: '至少一条完整路径的所有叶节点具有商业、在库或通用品采购 authority'
      },
      {
        label: 'ALL PROCUREMENT GRAPH',
        value: ledgerValue(closure.all_explored_procurement_closed, '采购全探索闭合', '仍有非采购闭合分支'),
        state: ledgerState(closure.all_explored_procurement_closed),
        detail: '全部已探索路径通过独立采购固定点；benchmark membership 不参与此判定'
      },
      {
        label: 'L3 PARENT SOLVED',
        value: !deliveryBytesVerified ? '交付字节未验证'
          : parentL3Solved ? '父路线已解' : verified ? '证明层级不足' : '未证明',
        state: !deliveryBytesVerified ? 'unknown' : parentL3Solved ? 'closed' : 'open',
        detail: '完整父路线的最弱反应达到精确文献先例；独立于搜索账本闭合'
      },
      {
        label: 'L4 PROCUREMENT',
        value: !deliveryBytesVerified ? '交付字节未验证' : procurementL4Ready ? '采购就绪' : '未证明',
        state: !deliveryBytesVerified ? 'unknown' : procurementL4Ready ? 'closed' : 'open',
        detail: `${authoritative ? Number(counts.procurement_boundary_leaves || 0) : '—'} 个采购边界 · ${authoritative ? Number(counts.l4_procurement_edges || 0) : '—'} 条 L4 边；必须同时满足父路线证明与采购固定点`
      }
    ];
    element('closureStatusGrid').innerHTML = cards.map(card => `
      <article class="closure-status-card" data-state="${esc(card.state)}">
        <span class="closure-status-dot" aria-hidden="true"></span>
        <div class="closure-status-copy">
          <span class="closure-status-label">${esc(card.label)}</span>
          <strong class="closure-status-value">${esc(card.value)}</strong>
          <span class="closure-status-detail">${esc(card.detail)}</span>
        </div>
      </article>`).join('');
  }

  function renderIntegrityStatus(status) {
    const revision = forest.source_revision_context || forest.artifact_revision || {};
    const coverage = forest.projection_coverage || {};
    const integrity = element('integrityStatus');
    const deliveryDigest = String(forest.delivery_sha256 || '');
    const sourceDigest = String(forest.source_forest_sha256 || '');
    const embeddedDigest = String(forest.embedded_json_sha256 || '');
    integrity.dataset.status = status === 'verified' ? 'bound' : status === 'invalid' ? 'invalid' : 'pending';
    const digestText = status === 'verified'
      ? `Delivery bytes verified ${embeddedDigest.slice(0, 12)} · source forest digest ${sourceDigest.slice(0, 12)}`
      : status === 'invalid'
        ? 'Delivery integrity verification failed'
        : status === 'unavailable'
          ? `Digest metadata ${deliveryDigest.slice(0, 12)} · browser verification unavailable`
          : 'Verifying embedded delivery bytes…';
    const sourceContextText = revision.revision_id
      ? ` · source context ${String(revision.revision_id).slice(0, 19)}; current closeout requires external manifest`
      : ' · current closeout requires external manifest';
    const projectionText = coverage.complete === false
      ? ` · Projection truncated: ${forest.counts?.truncated_projection_rows || coverage.omitted_count || 0} omitted`
      : '';
    integrity.classList.toggle('projection-warning', coverage.complete === false);
    integrity.title = `${digestText}${sourceContextText}${projectionText}`;
    integrity.innerHTML = `<span class="integrity-dot" aria-hidden="true"></span><span>${esc(digestText + sourceContextText + projectionText)}</span>`;
  }

  async function verifyEmbeddedDeliveryDigest() {
    const expected = String(forest.embedded_json_sha256 || '');
    const marker = `"embedded_json_sha256":"${expected}",`;
    if (!/^[0-9a-f]{64}$/.test(expected) || !forestDataText.includes(marker)) {
      deliveryIntegrityStatus = 'invalid';
      renderIntegrityStatus(deliveryIntegrityStatus);
      renderChrome({ refreshIntegrity: false });
      return deliveryIntegrityStatus;
    }
    if (!globalThis.crypto?.subtle || typeof TextEncoder === 'undefined') {
      deliveryIntegrityStatus = 'unavailable';
      renderIntegrityStatus(deliveryIntegrityStatus);
      renderChrome({ refreshIntegrity: false });
      return deliveryIntegrityStatus;
    }
    try {
      const canonicalEmbeddedJson = forestDataText.replace(marker, '');
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonicalEmbeddedJson));
      const actual = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
      deliveryIntegrityStatus = actual === expected ? 'verified' : 'invalid';
    } catch (_) {
      deliveryIntegrityStatus = 'unavailable';
    }
    renderIntegrityStatus(deliveryIntegrityStatus);
    renderChrome({ refreshIntegrity: false });
    return deliveryIntegrityStatus;
  }

  function renderEvidenceStats() {
    const counts = forest.counts || {};
    const literature = forest.run_trace?.literature_counts || {};
    const rows = [
      ['独立信源组', literature.independent_source_group_count ?? 0],
      ['文献文档', literature.document_count ?? 0],
      ['来源表示', literature.representation_count ?? 0],
      ['候选记录', literature.real_source_candidate_records ?? literature.real_source_candidates ?? literature.source_candidates ?? 0],
      ['文献/图像链', counts.visual_chains ?? 0],
      ['共识候选', counts.route_consensus_proposals ?? 0], ['后端替换', forest.replacement_validation?.validated_count ?? 0]
    ];
    element('evidenceStats').innerHTML = rows.map(([label, value]) =>
      `<div class="evidence-stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join('');
  }

  function renderFilters() {
    const listedLanes = (lanesProjection.lanes || []).filter(lane => lane.listed !== false);
    const partialExpandedCount = listedLanes.filter(lane => partialExpansionProgress(lane)).length;
    document.querySelectorAll('[data-stage-filter]').forEach(button => {
      const active = button.dataset.stageFilter === state.stageFilter;
      const stage = button.dataset.stageFilter;
      const count = stage === 'all'
        ? listedLanes.length
        : listedLanes.filter(lane => stageMembershipIsAuthoritative(lane, stage)).length;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
      button.setAttribute('aria-label', `${button.textContent.trim()}，${count} 条路线`);
      button.dataset.stageCount = String(count);
      if (stage === 'expanded') {
        button.title = `${count} 条全路径完成展开；${partialExpandedCount} 条仅部分展开，不计入本阶段`;
      }
    });
    const partialSummary = element('partialExpandedSummary');
    if (partialSummary) {
      partialSummary.hidden = partialExpandedCount === 0;
      partialSummary.textContent = partialExpandedCount
        ? `另有 ${partialExpandedCount} 条路线仅部分展开；它们保留在探索视图，不计入“全路径已展开”。`
        : '';
    }
    document.querySelectorAll('[data-branch-filter]').forEach(button => {
      const active = button.dataset.branchFilter === state.branchFilter;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const host = element('graphFilterBar');
    if (host) {
      const proofButtons = PROOF_ORDER.filter(tier => allProofTiers.includes(tier)).map(tier => {
        const active = state.proofFilters.has(tier);
        return `<button class="filter-chip ${tierClass(tier)}${active ? ' is-active' : ''}" type="button" data-proof-filter="${esc(tier)}" aria-pressed="${active}">${esc(tierLabel(tier))}</button>`;
      }).join('');
      const listedKindCounts = new Map();
      for (const lane of lanesProjection.lanes || []) {
        if (lane.listed === false) continue;
        listedKindCounts.set(lane.kind, (listedKindCounts.get(lane.kind) || 0) + 1);
      }
      const kindButtons = (lanesProjection.groups || []).filter(group => listedKindCounts.has(group.kind)).map(group => {
        const active = state.kindFilters.has(group.kind);
        return `<button class="filter-chip${active ? ' is-active' : ''}" type="button" data-kind-filter="${esc(group.kind)}" aria-pressed="${active}">${esc(group.label)} <span>${esc(listedKindCounts.get(group.kind))}</span></button>`;
      }).join('');
      host.innerHTML = `<div class="filter-row" aria-label="证明层级筛选">${proofButtons}</div>
        <div class="filter-row" aria-label="路线类别筛选">${kindButtons}</div>
        <div class="filter-row" aria-label="边显示范围">
          <button class="filter-chip${state.edgeFilter === 'all' ? ' is-active' : ''}" type="button" data-edge-filter="all" aria-pressed="${state.edgeFilter === 'all'}">全部边</button>
          <button class="filter-chip${state.edgeFilter === 'selected' ? ' is-active' : ''}" type="button" data-edge-filter="selected" aria-pressed="${state.edgeFilter === 'selected'}">仅选中路线</button>
          <button class="filter-chip" type="button" data-filter-reset>重置筛选</button>
        </div>`;
    }
    const optionChanges = (allProofTiers.length - state.proofFilters.size)
      + (allKinds.length - state.kindFilters.size)
      + (state.stageFilter !== 'all' ? 1 : 0)
      + (state.edgeFilter === 'selected' ? 1 : 0)
      + (state.orientation !== 'horizontal' ? 1 : 0)
      + (state.density !== 'comfortable' ? 1 : 0)
      + (state.edgeStyle !== 'trust' ? 1 : 0)
      + (state.labelMode !== 'semantic' ? 1 : 0);
    const optionCount = element('graphOptionsCount');
    if (optionCount) {
      optionCount.textContent = optionChanges ? String(optionChanges) : '';
      optionCount.toggleAttribute('hidden', !optionChanges);
    }
    document.querySelectorAll('[data-graph-mode]').forEach(button => {
      const active = button.dataset.graphMode === state.mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  function stageMembershipIsAuthoritative(lane, stage) {
    if (lanesProjection.schema_version !== BRANCH_LANE_SCHEMA_VERSION) return false;
    const evidence = lane?.stage_evidence;
    if (evidence?.schema_version !== BRANCH_STAGE_EVIDENCE_SCHEMA_VERSION) return false;
    const stageEvidence = evidence?.[stage];
    const member = stageEvidence?.member === true
      && Array.isArray(lane?.stage_memberships)
      && lane.stage_memberships.includes(stage);
    if (!member || stage !== 'expanded') return member;
    const matched = Number(stageEvidence.matched_step_count);
    const required = Number(stageEvidence.required_step_count);
    return stageEvidence.fully_expanded === true
      && stageEvidence.partial_expanded === false
      && Number.isInteger(matched)
      && Number.isInteger(required)
      && required > 0
      && matched === required
      && Array.isArray(stageEvidence.matched_step_ids)
      && stageEvidence.matched_step_ids.length === required
      && Array.isArray(stageEvidence.remaining_step_ids)
      && stageEvidence.remaining_step_ids.length === 0;
  }

  function partialExpansionProgress(lane) {
    const evidence = lane?.stage_evidence;
    const expanded = evidence?.expanded;
    if (lanesProjection.schema_version !== BRANCH_LANE_SCHEMA_VERSION
      || evidence?.schema_version !== BRANCH_STAGE_EVIDENCE_SCHEMA_VERSION
      || expanded?.partial_expanded !== true
      || expanded?.fully_expanded === true
      || expanded?.member === true
      || lane?.stage_memberships?.includes('expanded')) return null;
    const matched = Number(expanded.matched_step_count);
    const required = Number(expanded.required_step_count);
    if (!Number.isInteger(matched) || !Number.isInteger(required)
      || matched <= 0 || required <= matched) return null;
    return {
      matched,
      required,
      remainingStepIds: Array.isArray(expanded.remaining_step_ids) ? expanded.remaining_step_ids : [],
      reasons: Array.isArray(expanded.reasons) ? expanded.reasons : []
    };
  }

  function renderBranchGroups({ restoreFocusId = '' } = {}) {
    const lanes = filteredLanes();
    const grouped = new Map();
    for (const lane of lanes) {
      if (!grouped.has(lane.kind)) grouped.set(lane.kind, []);
      grouped.get(lane.kind).push(lane);
    }
    const host = element('branchGroups');
    host.innerHTML = (lanesProjection.groups || []).filter(group => grouped.has(group.kind)).map(group => {
      const rows = grouped.get(group.kind) || [];
      const visibleLimit = Number(forest.display_policy?.default_group_visible_count || 5);
      const expanded = state.expandedGroups.has(group.kind) || rows.length <= visibleLimit;
      const visibleRows = expanded ? rows : rows.slice(0, visibleLimit);
      const omitted = rows.length - visibleRows.length;
      return `<section class="branch-group" data-branch-kind="${esc(group.kind)}">
        <div class="branch-group-heading"><strong>${esc(group.label)}</strong><span>${rows.length} 个探索视图</span></div>
        <div class="branch-group-list">${visibleRows.map(lane => branchCard(lane)).join('')}</div>
        ${omitted ? `<button class="group-expand-button" type="button" data-expand-group="${esc(group.kind)}">展开其余 ${omitted} 个</button>` : (rows.length > visibleLimit ? `<button class="group-expand-button" type="button" data-collapse-group="${esc(group.kind)}">收起</button>` : '')}
      </section>`;
    }).join('') || `<div class="empty-state branch-empty-state" role="status" aria-live="polite">
      <strong>${state.stageFilter === 'all' ? '没有匹配路线' : '当前阶段没有权威绑定路线'}</strong>
      <span>${state.stageFilter === 'all'
        ? '清除搜索或放宽筛选条件。'
        : '旧版数据、聚合 proof tier、步骤数量与库存别名都不会被推断为阶段完成。'}</span>
      <button class="detail-action" type="button" data-filter-reset>重置筛选</button>
    </div>`;
    const status = element('stageFilterStatus');
    if (status) {
      const label = document.querySelector(`[data-stage-filter="${state.stageFilter}"]`)?.textContent?.trim() || '当前阶段';
      status.textContent = `${label}：${lanes.length} 条匹配路线`;
    }
    updateBranchRovingTabindex();
    if (restoreFocusId) {
      [...host.querySelectorAll('.branch-card[data-branch-id]')].find(row => row.dataset.branchId === restoreFocusId)?.focus();
    }
  }

  function branchCard(lane) {
    const selected = lane.branch_id === state.selectedBranchId;
    const branch = branches.get(lane.branch_id) || {};
    const stageEvidence = lane.stage_evidence?.schema_version === BRANCH_STAGE_EVIDENCE_SCHEMA_VERSION
      ? lane.stage_evidence : {};
    const stockEvidence = stageEvidence.stock || {};
    const partialProgress = partialExpansionProgress(lane);
    const partialReason = partialProgress
      ? `仅 ${partialProgress.matched}/${partialProgress.required} 个合成步具有当前 canonical queue succeeded 绑定${partialProgress.remainingStepIds.length ? `；待展开：${partialProgress.remainingStepIds.join('、')}` : ''}${partialProgress.reasons.length ? `；原因：${partialProgress.reasons.join('；')}` : ''}`
      : '';
    const partialBadge = partialProgress
      ? `<span class="branch-badge branch-badge--partial-expanded" title="${esc(partialReason)}">部分展开 ${esc(partialProgress.matched)}/${esc(partialProgress.required)}</span>`
      : '';
    const stateLabel = lane.solved && lane.executable && !lane.advisory_only
      ? '完整父路线'
      : lane.kind === 'proof_eligible_portfolio_route' ? '完整 portfolio'
        : stageMembershipIsAuthoritative(lane, 'stock')
          ? (stockEvidence.closure_scope === 'procurement' ? '采购闭合' : 'Benchmark 闭合')
          : stageMembershipIsAuthoritative(lane, 'reaction') ? '反应已验证'
            : stageMembershipIsAuthoritative(lane, 'expanded') ? '全路径已展开'
              : stageEvidence.suggestion?.member === true ? '断键建议'
                : partialProgress ? '探索中' : '阶段证据未绑定';
    return `<button class="branch-card ${tierClass(lane.proof_tier)}${selected ? ' is-selected' : ''}" type="button"
      data-branch-id="${esc(lane.branch_id)}" aria-current="${selected ? 'true' : 'false'}" tabindex="-1">
      <span class="branch-card-title">${esc(lane.title || lane.branch_id)}</span>
      <span class="branch-card-meta">${esc((lane.step_ids || []).length)} 步 · ${esc(tierLabel(lane.proof_tier))}</span>
      <span class="branch-card-badges">${lane.is_primary ? `<span class="branch-badge">${forest.primary_selection?.display_tiebreak_only ? '展示锚点' : '主分支'}</span>` : ''}<span class="branch-badge">${esc(stateLabel)}</span>${partialBadge}<span class="branch-badge">${esc(synthesisLabel(branch.synthesis_class))}</span></span>
    </button>`;
  }

  function updateBranchRovingTabindex() {
    const rows = [...document.querySelectorAll('.branch-card[data-branch-id]')];
    const active = rows.find(row => row.dataset.branchId === state.selectedBranchId) || rows[0];
    rows.forEach(row => row.tabIndex = row === active ? 0 : -1);
  }

  function densityMetrics() {
    if (state.mode === 'current' && state.density === 'comfortable') {
      return matchMedia('(max-width: 639px)').matches
        ? { layerGap: 180, rowGap: 172, laneGap: 24, componentGap: 22, nodeScale: 1 }
        : { layerGap: 246, rowGap: 156, laneGap: 24, componentGap: 22, nodeScale: 1 };
    }
    if (state.density === 'compact') return { layerGap: 188, rowGap: 78, laneGap: 18, componentGap: 18, nodeScale: .88 };
    if (state.density === 'overview') return { layerGap: 150, rowGap: 58, laneGap: 12, componentGap: 12, nodeScale: .72 };
    return { layerGap: 238, rowGap: 108, laneGap: 28, componentGap: 26, nodeScale: 1 };
  }

  function effectiveOrientation() {
    return state.mode === 'current' && matchMedia('(max-width: 639px)').matches
      ? 'vertical'
      : state.orientation;
  }

  function buildGraphModel() {
    const lanes = filteredLanes({ includeReplacementPreview: true });
    if (state.mode === 'shared') return buildSharedModel(overviewLanes(lanes));
    const selected = lanes.find(lane => lane.branch_id === state.selectedBranchId);
    const activeLanes = state.mode === 'current' ? (selected ? [selected] : []) : overviewLanes(lanes);
    return buildLaneModel(activeLanes);
  }

  function overviewLanes(lanes) {
    if (state.showAllOverview) return lanes;
    const topK = Number(forest.display_policy?.default_overview_top_k || 12);
    return lanes.slice().sort((left, right) => branchDisplayScore(right) - branchDisplayScore(left)
      || stableTextCompare(left.branch_id, right.branch_id)).slice(0, topK);
  }

  function effectiveLaneRows(lanes) {
    if (state.edgeFilter !== 'selected' || !state.selectedBranchId) return lanes;
    return lanes.filter(lane => lane.branch_id === state.selectedBranchId);
  }

  function buildSharedModel(rawLanes) {
    const lanes = effectiveLaneRows(rawLanes);
    const branchIds = new Set(lanes.map(row => row.branch_id));
    let edges = (graph.edges || []).filter(edge => branchIds.has(edge.branch_id));
    edges = edges.filter(edge => state.proofFilters.has(tierOfStep(steps.get(edge.reaction_step_id))));
    const includedIds = new Set(edges.flatMap(edge => [edge.source_graph_node_id, edge.target_graph_node_id]));
    const nodes = [...includedIds].map(id => graphNodes.get(id)).filter(Boolean);
    const metrics = densityMetrics();
    const orientation = effectiveOrientation();
    const buckets = new Map();
    for (const node of nodes) {
      const logical = layoutByNode.get(node.graph_node_id) || { layer: node.layer || 0, order: 0, component_order: 0 };
      const key = Number(logical.layer || 0);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push({ node, logical });
    }
    const positions = new Map();
    const layerEntries = [...buckets.entries()].sort((a, b) => a[0] - b[0]);
    const largestLayer = Math.max(1, ...layerEntries.map(([, bucket]) => bucket.length));
    const sampleSize = nodeSize({ node_type: 'molecule' }, metrics.nodeScale);
    const cellWidth = sampleSize.w + Math.max(18, metrics.componentGap);
    const cellHeight = sampleSize.h + Math.max(22, metrics.rowGap - 50);
    const targetRatio = 1.65;
    const horizontalWrap = clamp(Math.round(Math.sqrt(
      (largestLayer * cellHeight * targetRatio) / Math.max(cellWidth * layerEntries.length, 1)
    )), 1, largestLayer);
    const verticalWrap = clamp(Math.round(Math.sqrt(
      (largestLayer * cellWidth) / Math.max(cellHeight * layerEntries.length * targetRatio, 1)
    )), 1, largestLayer);
    let layerCursor = 44;
    for (const [, bucket] of layerEntries) {
      bucket.sort((a, b) => Number(a.logical.order || 0) - Number(b.logical.order || 0)
        || stableTextCompare(a.node.graph_node_id, b.node.graph_node_id));
      const wrap = orientation === 'vertical'
        ? Math.min(verticalWrap, bucket.length)
        : Math.min(horizontalWrap, bucket.length);
      const crossCount = Math.max(1, Math.ceil(bucket.length / wrap));
      bucket.forEach((row, index) => {
        const size = nodeSize(row.node, metrics.nodeScale);
        const primaryIndex = Math.floor(index / crossCount);
        const crossIndex = index % crossCount;
        const position = orientation === 'vertical'
          ? { x: 44 + crossIndex * cellWidth, y: layerCursor + primaryIndex * cellHeight, ...size }
          : { x: layerCursor + primaryIndex * cellWidth, y: 44 + crossIndex * cellHeight, ...size };
        positions.set(row.node.graph_node_id, position);
      });
      const bandExtent = wrap * (orientation === 'vertical' ? cellHeight : cellWidth);
      layerCursor += bandExtent + Math.max(72, metrics.layerGap * .55);
    }
    const instances = nodes.map(node => ({ instanceId: node.graph_node_id, graphNodeId: node.graph_node_id, branchId: '', node }));
    return finaliseModel('shared', instances, edges.map(edge => ({ ...edge, sourceInstanceId: edge.source_graph_node_id, targetInstanceId: edge.target_graph_node_id })), positions, []);
  }

  function buildLaneModel(rawLanes) {
    const lanes = effectiveLaneRows(rawLanes);
    const metrics = densityMetrics();
    const orientation = effectiveOrientation();
    const positions = new Map();
    const instances = [];
    const renderedEdges = [];
    const decorations = [];
    const tiles = lanes.map(lane => {
      const localRows = (lane.node_layout || []).slice().sort((a, b) => Number(a.layer) - Number(b.layer)
        || Number(a.order) - Number(b.order) || stableTextCompare(a.graph_node_id, b.graph_node_id));
      const byLayer = new Map();
      for (const row of localRows) {
        if (!byLayer.has(Number(row.layer || 0))) byLayer.set(Number(row.layer || 0), []);
        byLayer.get(Number(row.layer || 0)).push(row);
      }
      const maxLayerRows = Math.max(1, ...[...byLayer.values()].map(rows => rows.length));
      const maximumNode = nodeSize({ node_type: 'molecule' }, metrics.nodeScale);
      const maximumLayer = Math.max(0, Number(lane.max_layer || 0));
      const tileWidth = orientation === 'vertical'
        ? 40 + (maxLayerRows - 1) * metrics.rowGap + maximumNode.w
        : 92 + maximumLayer * metrics.layerGap + maximumNode.w;
      const tileHeight = orientation === 'vertical'
        ? 76 + maximumLayer * metrics.layerGap + maximumNode.h
        : 76 + (maxLayerRows - 1) * metrics.rowGap + maximumNode.h;
      const relativePositions = new Map();
      for (const [layerIndex, rows] of byLayer) {
        rows.sort((a, b) => Number(a.order || 0) - Number(b.order || 0) || stableTextCompare(a.graph_node_id, b.graph_node_id));
        rows.forEach((logical, rowIndex) => {
          const node = graphNodes.get(logical.graph_node_id);
          if (!node) return;
          const instanceId = `${lane.branch_id}::${node.graph_node_id}`;
          const size = nodeSize(node, metrics.nodeScale);
          const position = orientation === 'vertical'
            ? { x: 20 + rowIndex * metrics.rowGap, y: 52 + layerIndex * metrics.layerGap, ...size }
            : { x: 52 + layerIndex * metrics.layerGap, y: 52 + rowIndex * metrics.rowGap, ...size };
          relativePositions.set(instanceId, position);
        });
      }
      return { lane, byLayer, relativePositions, w: tileWidth, h: tileHeight };
    });
    const averageWidth = tiles.reduce((sum, tile) => sum + tile.w, 0) / Math.max(1, tiles.length);
    const averageHeight = tiles.reduce((sum, tile) => sum + tile.h, 0) / Math.max(1, tiles.length);
    const targetRatio = orientation === 'vertical' ? 1.55 : 1.68;
    const columns = state.mode === 'current' ? 1 : clamp(
      Math.round(Math.sqrt(tiles.length * averageHeight / Math.max(1, averageWidth) * targetRatio)),
      1,
      Math.max(1, tiles.length)
    );
    let originY = 28;
    for (let rowStart = 0; rowStart < tiles.length; rowStart += columns) {
      const row = tiles.slice(rowStart, rowStart + columns);
      const rowHeight = Math.max(...row.map(tile => tile.h));
      let originX = 28;
      for (const tile of row) {
        const { lane } = tile;
        decorations.push({ x: originX, y: originY, w: tile.w, h: tile.h, label: `${lane.kind_label || lane.kind} · ${lane.title}`, branchId: lane.branch_id, kind: lane.kind });
        for (const [instanceId, relative] of tile.relativePositions) {
          const graphNodeId = instanceId.slice(instanceId.indexOf('::') + 2);
          const node = graphNodes.get(graphNodeId);
          if (!node) continue;
          positions.set(instanceId, { ...relative, x: relative.x + originX, y: relative.y + originY });
          instances.push({ instanceId, graphNodeId, branchId: lane.branch_id, node });
        }
        for (const edgeId of lane.edge_ids || []) {
          const edge = edgeById.get(edgeId);
          if (!edge) continue;
          renderedEdges.push({
            ...edge,
            sourceInstanceId: `${lane.branch_id}::${edge.source_graph_node_id}`,
            targetInstanceId: `${lane.branch_id}::${edge.target_graph_node_id}`
          });
        }
        originX += tile.w + metrics.laneGap;
      }
      originY += rowHeight + metrics.laneGap;
    }
    const model = finaliseModel(state.mode, instances, renderedEdges, positions, decorations);
    model.packing = { algorithm: 'deterministic_adaptive_shelf_grid.v1', columns, targetRatio };
    return model;
  }

  function nodeSize(node, scale) {
    const reaction = node.node_type === 'reaction';
    const mobileCurrent = state.mode === 'current' && matchMedia('(max-width: 639px)').matches;
    const base = state.mode === 'current'
      ? (mobileCurrent
        ? (reaction ? { w: 136, h: 62 } : { w: 158, h: 118 })
        : (reaction ? { w: 154, h: 68 } : { w: 220, h: 144 }))
      : (reaction ? { w: 166, h: 70 } : { w: 194, h: 78 });
    return { w: Math.round(base.w * scale), h: Math.round(base.h * scale) };
  }

  function finaliseModel(mode, instances, edges, positions, decorations) {
    const boxes = [...positions.values(), ...decorations];
    const maxX = Math.max(mode === 'current' ? 1 : 700, ...boxes.map(row => row.x + row.w + 44));
    const maxY = Math.max(mode === 'current' ? 1 : 420, ...boxes.map(row => row.y + row.h + 44));
    return { mode, instances, edges, positions, decorations, bounds: { x: 0, y: 0, w: maxX, h: maxY }, packing: null };
  }

  function renderGraph({ fit = true } = {}) {
    renderModel = buildGraphModel();
    const viewport = element('graphViewport');
    viewport.dataset.graphMode = state.mode;
    viewport.dataset.orientation = effectiveOrientation();
    const width = Math.max(1, viewport.clientWidth || 1100);
    const height = Math.max(1, viewport.clientHeight || 680);
    if (!renderModel.instances.length) {
      const matchingLaneCount = filteredLanes().length;
      element('mainRoute').innerHTML = `<div class="empty-state graph-empty-state" role="status" aria-live="polite">
        <strong>${matchingLaneCount ? '匹配路线没有可绘制的显式依赖' : '当前筛选没有权威绑定路线'}</strong>
        <span>${matchingLaneCount
          ? '该视图不会根据数组顺序补画分子或反应边。'
          : '请选择其他阶段，或重置筛选查看全部探索分支。'}</span>
        <button class="detail-action" type="button" data-filter-reset>重置筛选</button>
      </div>`;
      element('graphMinimap').innerHTML = '';
      element('graphMinimap').hidden = true;
      element('graphVisibleCount').textContent = `0/${matchingLaneCount} 探索视图 · 0 节点 · 0 边`;
      element('overviewToggle').hidden = true;
      element('graphTitle').textContent = matchingLaneCount
        ? '没有可绘制的显式依赖' : '没有权威绑定路线';
      element('graphSubtitle').textContent = matchingLaneCount
        ? COPY.explicitEdges : '阶段筛选只消费后端 v2 authority evidence；旧版聚合提示不会被猜测为完成。';
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      element('zoomReadout').textContent = '100%';
      return;
    }
    const markers = unique(PROOF_ORDER.map(tierClass)).map(cssClass => {
      const tier = PROOF_ORDER.find(value => tierClass(value) === cssClass);
      return `<marker id="arrow-${cssClass}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="${TIER_COLOR[tier] || '#64748b'}"></path></marker>`;
    }).join('')
      + '<marker id="arrow-neutral" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="#94a3b8"></path></marker>';
    const decorations = renderModel.decorations.map(row => `<g class="graph-lane-decoration${row.branchId === state.selectedBranchId ? ' is-selected' : ''}" data-lane-branch-id="${esc(row.branchId)}">
      <rect x="${row.x}" y="${row.y}" width="${row.w}" height="${row.h}" rx="16"></rect>
      <text x="${row.x + 16}" y="${row.y + 24}">${esc(middleEllipsis(row.label, 54))}</text></g>`).join('');
    const edges = renderModel.edges.map(edge => edgeSvg(edge, renderModel.positions)).join('');
    const nodes = renderModel.instances.map(instance => nodeSvg(instance, renderModel.positions.get(instance.instanceId))).join('');
    element('mainRoute').innerHTML = `<svg class="graph-svg dependency-svg" data-layout-packing="${esc(renderModel.packing?.algorithm || 'shared_component_layers.v1')}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="graphTitle graphSubtitle">
      <defs>${markers}</defs><g class="graph-world">${decorations}${edges}${nodes}</g></svg>`;
    const visibleBranches = new Set(renderModel.instances.map(row => row.branchId).filter(Boolean));
    element('graphVisibleCount').textContent = `${visibleBranches.size || (state.mode === 'shared' ? Math.min(filteredLanes().length, Number(forest.display_policy?.default_overview_top_k || 12)) : 0)}/${filteredLanes().length} 探索视图 · ${renderModel.instances.length} 节点 · ${renderModel.edges.length} 边`;
    const replacementPreview = state.activeReplacement && state.mode === 'current';
    const overviewToggle = element('overviewToggle');
    const filteredCount = filteredLanes().length;
    const topK = Number(forest.display_policy?.default_overview_top_k || 12);
    overviewToggle.hidden = !['clusters', 'shared'].includes(state.mode) || filteredCount <= topK;
    overviewToggle.textContent = state.showAllOverview
      ? `仅显示 Top ${topK}` : `显示全部 ${filteredCount} 个探索视图`;
    const currentLane = laneByBranch.get(state.selectedBranchId) || {};
    element('graphTitle').textContent = replacementPreview ? '完整替换路线预览 · 后端已重验'
      : state.mode === 'clusters' ? `路线全景 · ${visibleBranches.size}/${filteredCount} 个探索视图`
        : state.mode === 'shared' ? '共享骨架 · 规范分子–反应图'
          : `重点分支 · ${middleEllipsis(currentLane.title || state.selectedBranchId, 48)}`;
    element('graphSubtitle').textContent = replacementPreview
      ? '当前画布是完整的后端 AND/OR 重验分支；它不是单步拼接，也不建立父路线证明。'
      : state.mode === 'clusters'
        ? '默认按闭合度与可信度展示高价值 Top-K；其余探索视图可按需展开，数量不代表完整路线数。'
        : state.mode === 'shared'
          ? '相同规范分子合并显示；线宽表达支持，颜色仅表达反应证明层级。'
          : `${(currentLane.step_ids || []).length} 步 · ${tierLabel(currentLane.proof_tier)} · ${(currentLane.source_refs || []).length} 个来源引用；分子保持中性，证明颜色只用于反应和依赖边。`;
    renderMinimap();
    if (fit) fitGraph({ readable: state.mode === 'current' }); else applyViewportTransform();
    applyGraphSelection();
  }

  function edgeSvg(edge, positions) {
    const source = positions.get(edge.sourceInstanceId);
    const target = positions.get(edge.targetInstanceId);
    if (!source || !target) return '';
    const step = steps.get(edge.reaction_step_id);
    const tier = edge.trust_vector?.proof_tier || tierOfStep(step);
    const visual = edge.visual_encoding || step?.visual_encoding || step?.trust_vector?.visual_encoding || {};
    const simple = state.edgeStyle === 'simple';
    const contrast = state.edgeStyle === 'contrast';
    const color = simple ? '#94a3b8' : (visual.color || TIER_COLOR[tier] || '#64748b');
    const width = contrast ? Math.max(2.4, Number(visual.width || 1.5)) : (simple ? 1.25 : Number(visual.width || 1.5));
    const opacity = contrast ? .92 : (simple ? .48 : Number(visual.opacity || .62));
    const dash = simple ? '' : String(visual.dash_pattern || '');
    const path = edgePath(source, target, state.edgeStyle !== 'trust');
    const markerId = simple ? 'arrow-neutral' : `arrow-${tierClass(tier)}`;
    return `<path class="graph-edge dependency-edge trust-edge ${tierClass(tier)}" data-edge-id="${esc(edge.edge_id)}" data-branch-id="${esc(edge.branch_id)}" data-reaction-step-id="${esc(edge.reaction_step_id)}" d="${path}" stroke="${esc(color)}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-dasharray="${esc(dash)}" marker-end="url(#${markerId})"><title>${esc(`${tierLabel(tier)} · ${edge.edge_type || '显式依赖'} · ${edge.branch_id || ''}`)}</title></path>`;
  }

  function edgePath(source, target, orthogonal) {
    if (effectiveOrientation() === 'vertical') {
      const x1 = source.x + source.w / 2, y1 = source.y + source.h;
      const x2 = target.x + target.w / 2, y2 = target.y;
      const middle = (y1 + y2) / 2;
      return orthogonal ? `M ${x1} ${y1} V ${middle} H ${x2} V ${y2}`
        : `M ${x1} ${y1} C ${x1} ${middle}, ${x2} ${middle}, ${x2} ${y2}`;
    }
    const x1 = source.x + source.w, y1 = source.y + source.h / 2;
    const x2 = target.x, y2 = target.y + target.h / 2;
    const middle = (x1 + x2) / 2;
    return orthogonal ? `M ${x1} ${y1} H ${middle} V ${y2} H ${x2}`
      : `M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`;
  }

  function nodeSvg(instance, position) {
    if (!position) return '';
    const node = instance.node;
    const reaction = node.node_type === 'reaction';
    const molecule = reaction ? null : (moleculeNodes.get(node.molecule_node_id) || node);
    const step = reaction ? steps.get(node.reaction_step_id) : null;
    const tier = reaction ? (node.proof_tier || tierOfStep(step)) : '';
    const nodeTierClass = reaction ? tierClass(tier) : 'node-tier-neutral';
    const fullLabel = node.label || node.graph_node_id;
    const structureSvg = !reaction && state.mode === 'current' && state.labelMode !== 'minimal'
      ? safeStructureSvg(molecule?.structure_svg) : '';
    const displayLabel = structureSvg ? moleculeCaption(molecule, fullLabel) : fullLabel;
    const lines = wrapLabel(
      state.labelMode === 'minimal' ? (reaction ? tierLabel(tier) : '分子') : displayLabel,
      reaction ? 22 : (structureSvg ? 30 : 25)
    );
    const textX = position.x + 12;
    const textY = structureSvg ? position.y + position.h - 13 : position.y + (lines.length > 1 ? 27 : 34);
    const selected = node.graph_node_id === state.selectedGraphNodeId || (reaction && node.reaction_step_id === state.selectedStepId);
    const semanticLabel = reaction ? tierLabel(tier) : (node.role || '分子中间体');
    return `<g class="graph-node dependency-${reaction ? 'reaction graph-node--reaction' : 'molecule graph-node--molecule'} ${nodeTierClass}${selected ? ' is-selected' : ''}" data-graph-node-id="${esc(node.graph_node_id)}" data-node-type="${esc(node.node_type)}" data-node-role="${esc(node.role || '')}" data-branch-id="${esc(instance.branchId)}" data-instance-id="${esc(instance.instanceId)}" ${reaction ? `data-route-step="${esc(node.reaction_step_id)}"` : ''} tabindex="${selected ? '0' : '-1'}" role="button" aria-label="${esc(`${reaction ? '反应' : '分子'}：${fullLabel}，${semanticLabel}`)}">
      <title>${esc(fullLabel)}</title><rect class="node-surface" x="${position.x}" y="${position.y}" width="${position.w}" height="${position.h}" rx="${reaction ? 12 : 24}"></rect>
      ${structureSvg ? `<foreignObject class="node-depiction" x="${position.x + 7}" y="${position.y + 7}" width="${position.w - 14}" height="${position.h - 39}"><div xmlns="http://www.w3.org/1999/xhtml" class="node-depiction-frame">${structureSvg}</div></foreignObject>` : ''}
      <text class="node-label" x="${textX}" y="${textY}">${svgTextLines(lines, textX, textY)}</text>
      ${reaction && state.labelMode !== 'minimal' ? `<text class="graph-node-tier node-meta" x="${textX}" y="${position.y + position.h - 10}">${esc(tierLabel(tier))}</text>` : ''}</g>`;
  }

  function applyGraphSelection() {
    if (!renderModel) return;
    const selectedBranch = state.selectedBranchId;
    const selectedNode = state.selectedGraphNodeId;
    const selectedStep = state.selectedStepId;
    const hasSelection = Boolean(selectedNode || selectedStep || selectedBranch);
    let focusAssigned = false;
    document.querySelectorAll('.graph-node').forEach(row => {
      const selected = row.dataset.graphNodeId === selectedNode || row.dataset.routeStep === selectedStep;
      const related = selected || (selectedBranch && row.dataset.branchId === selectedBranch);
      const focusable = selected && (state.selectedInstanceId
        ? row.dataset.instanceId === state.selectedInstanceId
        : !focusAssigned);
      if (focusable) focusAssigned = true;
      row.classList.toggle('is-selected', selected);
      row.classList.toggle('is-dimmed', hasSelection && !related && state.edgeFilter === 'selected');
      row.tabIndex = focusable ? 0 : -1;
    });
    document.querySelectorAll('.graph-edge').forEach(row => {
      const related = (selectedStep && row.dataset.reactionStepId === selectedStep)
        || (selectedBranch && row.dataset.branchId === selectedBranch)
        || (!selectedStep && !selectedBranch);
      row.classList.toggle('is-selected', Boolean(selectedStep && row.dataset.reactionStepId === selectedStep));
      row.classList.toggle('is-dimmed', state.edgeFilter === 'selected' && hasSelection && !related);
    });
    document.querySelectorAll('[data-lane-branch-id]').forEach(row => {
      row.classList.toggle('is-selected', row.dataset.laneBranchId === selectedBranch);
    });
    document.querySelectorAll('.branch-card[data-branch-id]').forEach(row => {
      const selected = row.dataset.branchId === selectedBranch;
      row.classList.toggle('is-selected', selected);
      row.setAttribute('aria-current', String(selected));
    });
    if (!document.querySelector('.graph-node[tabindex="0"]')) {
      const first = document.querySelector('.graph-node');
      if (first) first.tabIndex = 0;
    }
    updateBranchRovingTabindex();
  }

  function currentTargetPosition() {
    if (!renderModel) return null;
    const orientation = effectiveOrientation();
    const candidates = renderModel.instances.map(instance => {
      const molecule = instance.node?.node_type === 'molecule'
        ? moleculeNodes.get(instance.node.molecule_node_id) : null;
      const roles = [instance.node?.role, molecule?.role, ...(molecule?.roles || [])];
      return {
        position: renderModel.positions.get(instance.instanceId),
        target: roles.includes('target')
      };
    }).filter(row => row.position);
    const explicit = candidates.filter(row => row.target);
    const ranked = explicit.length ? explicit : candidates;
    return ranked.sort((left, right) => orientation === 'vertical'
      ? right.position.y - left.position.y
      : right.position.x - left.position.x)[0]?.position || null;
  }

  function fitGraph({ readable = false } = {}) {
    if (!renderModel) return;
    const viewport = element('graphViewport');
    const width = Math.max(1, viewport.clientWidth || 1000);
    const height = Math.max(1, viewport.clientHeight || 620);
    const padding = 42;
    const naturalFit = Math.min(
      (width - padding * 2) / renderModel.bounds.w,
      (height - padding * 2) / renderModel.bounds.h
    );
    if (readable && state.mode === 'current') {
      const minimumReadableZoom = matchMedia('(max-width: 639px)').matches ? .85 : .82;
      state.zoom = clamp(Math.max(naturalFit, minimumReadableZoom), minimumReadableZoom, 1.15);
      const oversized = naturalFit < minimumReadableZoom;
      const vertical = effectiveOrientation() === 'vertical';
      const target = currentTargetPosition();
      state.panX = oversized && !vertical
        ? width - ((target?.x ?? renderModel.bounds.w) + (target?.w || 0)) * state.zoom - 48
        : oversized && vertical && target
          ? width / 2 - (target.x + target.w / 2) * state.zoom
        : (width - renderModel.bounds.w * state.zoom) / 2;
      state.panY = oversized && vertical
        ? height - ((target?.y ?? renderModel.bounds.h) + (target?.h || 0)) * state.zoom - 48
        : oversized && !vertical && target
          ? height / 2 - (target.y + target.h / 2) * state.zoom
        : (height - renderModel.bounds.h * state.zoom) / 2;
    } else {
      state.zoom = clamp(naturalFit, .015, 2.5);
      state.panX = (width - renderModel.bounds.w * state.zoom) / 2;
      state.panY = (height - renderModel.bounds.h * state.zoom) / 2;
    }
    applyViewportTransform();
  }

  function resetGraph() { state.zoom = 1; state.panX = 24; state.panY = 24; applyViewportTransform(); }
  function zoomGraph(factor, clientX = null, clientY = null) {
    const viewport = element('graphViewport');
    const rect = viewport.getBoundingClientRect();
    const anchorX = clientX === null ? rect.width / 2 : clientX - rect.left;
    const anchorY = clientY === null ? rect.height / 2 : clientY - rect.top;
    const previous = state.zoom;
    const next = clamp(previous * factor, .015, 3.5);
    const worldX = (anchorX - state.panX) / previous;
    const worldY = (anchorY - state.panY) / previous;
    state.zoom = next;
    state.panX = anchorX - worldX * next;
    state.panY = anchorY - worldY * next;
    applyViewportTransform();
  }
  function applyPanTransform() {
    const svg = document.querySelector('.graph-svg');
    if (!svg) return;
    const transform = `translate3d(${state.panX}px, ${state.panY}px, 0)`;
    if (svg.style.transform !== transform) svg.style.transform = transform;
  }
  function applyViewportTransform() {
    applyPanTransform();
    const world = document.querySelector('.graph-world');
    const scaleTransform = `scale(${state.zoom})`;
    if (world && world.getAttribute('transform') !== scaleTransform) world.setAttribute('transform', scaleTransform);
    const viewport = element('graphViewport');
    const zoomBand = state.zoom < .18 ? 'overview' : state.zoom < .5 ? 'medium' : 'detail';
    if (viewport.dataset.labelMode !== state.labelMode) viewport.dataset.labelMode = state.labelMode;
    if (viewport.dataset.zoomBand !== zoomBand) viewport.dataset.zoomBand = zoomBand;
    const zoomLabel = `${Math.round(state.zoom * 100)}%`;
    if (element('zoomReadout').textContent !== zoomLabel) element('zoomReadout').textContent = zoomLabel;
    updateMinimapViewport();
  }

  function renderMinimap() {
    const host = element('graphMinimap');
    if (!renderModel || !renderModel.instances.length) { host.innerHTML = ''; return; }
    const width = 180, height = 108;
    const scale = Math.min(width / renderModel.bounds.w, height / renderModel.bounds.h);
    const nodes = renderModel.instances.map(instance => {
      const pos = renderModel.positions.get(instance.instanceId);
      if (!pos) return '';
      return `<rect x="${pos.x * scale}" y="${pos.y * scale}" width="${Math.max(2, pos.w * scale)}" height="${Math.max(2, pos.h * scale)}" rx="1"></rect>`;
    }).join('');
    host.innerHTML = `<svg class="minimap-svg" viewBox="0 0 ${width} ${height}" data-minimap-scale="${scale}" role="img" aria-label="当前图全局位置">${nodes}<rect class="minimap-viewport" x="0" y="0" width="1" height="1"></rect></svg>`;
    updateMinimapViewport();
  }
  function updateMinimapViewport() {
    if (!renderModel) return;
    const host = element('graphMinimap');
    const svg = host?.querySelector('.minimap-svg');
    const rectNode = svg?.querySelector('.minimap-viewport');
    if (!svg || !rectNode) return;
    const scale = Number(svg.dataset.minimapScale || 1);
    const viewport = element('graphViewport');
    const viewportWidth = viewport.clientWidth;
    const viewportHeight = viewport.clientHeight;
    const visibleRatio = clamp(viewportWidth / (renderModel.bounds.w * state.zoom), 0, 1)
      * clamp(viewportHeight / (renderModel.bounds.h * state.zoom), 0, 1);
    host.toggleAttribute('hidden', matchMedia('(max-width: 639px)').matches || visibleRatio >= .7);
    const worldX = -state.panX / state.zoom;
    const worldY = -state.panY / state.zoom;
    rectNode.setAttribute('x', String(clamp(worldX * scale, 0, 180)));
    rectNode.setAttribute('y', String(clamp(worldY * scale, 0, 108)));
    rectNode.setAttribute('width', String(clamp((viewportWidth / state.zoom) * scale, 2, 180)));
    rectNode.setAttribute('height', String(clamp((viewportHeight / state.zoom) * scale, 2, 108)));
  }

  function selectBranch(branchId, { focusGraph = false } = {}) {
    if (!branches.has(branchId)) return;
    if (state.activeReplacement?.replacementBranchId !== branchId) {
      state.activeReplacement = null;
    }
    state.selectedBranchId = branchId;
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedStepId = '';
    persistState();
    if (state.mode === 'current' || state.edgeFilter === 'selected') renderGraph();
    else applyGraphSelection();
    renderDetail();
    announce(`已选择路线 ${branches.get(branchId)?.title || branchId}`);
    if (focusGraph) document.querySelector('.graph-node:not(.is-dimmed)')?.focus();
  }
  function selectGraphNode(graphNodeId, { focus = false, branchId = '', instanceId = '' } = {}) {
    const node = graphNodes.get(graphNodeId);
    if (!node) return;
    state.selectedGraphNodeId = graphNodeId;
    state.selectedInstanceId = instanceId;
    state.selectedStepId = node.node_type === 'reaction' ? node.reaction_step_id : '';
    const selectedInstance = renderModel?.instances.find(row => row.graphNodeId === graphNodeId && row.branchId);
    const selectedBranchId = branchId || selectedInstance?.branchId || '';
    if (selectedBranchId && branches.has(selectedBranchId)) state.selectedBranchId = selectedBranchId;
    state.inspectorOpen = true;
    if (isMobileDrawerLayout()) state.navOpen = false;
    applyPersistentChromeState();
    updateMobileNavigation('inspector');
    applyGraphSelection();
    renderDetail();
    persistState();
    if (isInspectorOverlayLayout()) {
      requestAnimationFrame(() => element('detailTabs')?.querySelector('[aria-selected="true"]')?.focus());
    }
    announce(`已选择${node.node_type === 'reaction' ? '反应' : '分子'} ${node.label || graphNodeId}`);
    if (focus && !isInspectorOverlayLayout()) [...document.querySelectorAll('[data-graph-node-id]')].find(row =>
      instanceId ? row.dataset.instanceId === instanceId : row.dataset.graphNodeId === graphNodeId)?.focus();
  }

  function selectedEntity() {
    if (state.selectedStepId && steps.has(state.selectedStepId)) return { type: 'reaction', value: steps.get(state.selectedStepId) };
    const graphNode = graphNodes.get(state.selectedGraphNodeId);
    if (graphNode?.node_type === 'molecule') return { type: 'molecule', value: moleculeNodes.get(graphNode.molecule_node_id) || graphNode };
    if (state.selectedBranchId && branches.has(state.selectedBranchId)) return { type: 'branch', value: branches.get(state.selectedBranchId) };
    return null;
  }

  function renderDetailTabs() {
    let activeTabId = '';
    document.querySelectorAll('[data-detail-tab]').forEach(button => {
      const active = button.dataset.detailTab === state.detailTab;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      if (active) activeTabId = button.id;
    });
    if (activeTabId) element('detail')?.setAttribute('aria-labelledby', activeTabId);
  }
  function renderDetail() {
    renderDetailTabs();
    const entity = selectedEntity();
    const host = element('detail');
    if (!entity) { host.innerHTML = '<div class="empty-state"><strong>选择一个反应或分子</strong><span>检查结构、条件、证明层级与来源。</span></div>'; return; }
    element('inspectorTitle').textContent = entity.type === 'reaction' ? '反应检查器' : entity.type === 'molecule' ? '分子检查器' : '路线检查器';
    if (state.detailTab === 'alternatives') { renderAlternatives(entity, host); return; }
    if (state.detailTab === 'evidence') { renderEvidence(entity, host); return; }
    host.innerHTML = entity.type === 'reaction' ? reactionOverview(entity.value)
      : entity.type === 'molecule' ? moleculeOverview(entity.value) : branchOverview(entity.value);
  }

  function reactionOverview(step) {
    const trust = step.trust_vector || {};
    const sources = (step.source_refs || []).map(ref => `<div class="trace-row">${esc(basename(ref))}</div>`).join('');
    const conditions = (step.conditions || []).map(row => `<div class="condition-line"><span class="condition-label">${esc(row.label || '条件')}</span><span class="condition-value">${esc(row.value || '')}</span></div>`).join('');
    return `<article><header><p class="detail-kind">反应步骤 · ${esc(tierLabel(tierOfStep(step)))}</p><h3 class="detail-title">${esc(step.label || step.step_id)}</h3></header>
      ${state.activeReplacement ? `<div class="notice replacement-preview-notice"><strong>完整替换路线预览</strong><span>该分支已由后端 AND/OR 对 connectivity、stock 与 reaction proof 整路重验；预览不等于父路线证明。</span><button class="detail-action" type="button" data-replacement-reset>恢复原路线</button></div>` : ''}
      <section class="detail-section"><h3>反应连接</h3><p class="v">${esc(nodeNames(step.from_node_ids))} → ${esc(nodeNames(step.to_node_ids))}</p></section>
      <section class="detail-section"><h3>条件</h3><div class="condition-list">${conditions || `<div class="empty">${esc(step.condition_summary || '条件未记录')}</div>`}</div></section>
      <section class="detail-section"><h3>Trust vector</h3><div class="trust-grid">${['identity','connectivity','source_independence','stock','conditions','forward_feasibility'].map(key => `<div class="trust-cell ${tierClass(tierOfStep(step))}" style="--trust-value:${clamp(Number(trust[key] || 0), 0, 1)}"><strong>${esc(key)}</strong><span>${Number(trust[key] || 0).toFixed(2)}</span></div>`).join('')}</div></section>
      <section class="detail-section"><h3>来源</h3><div class="trace-list">${sources || '<div class="empty">来源未记录</div>'}</div></section></article>`;
  }
  function moleculeOverview(node) {
    return `<article><header><p class="detail-kind">分子节点</p><h3 class="detail-title">${esc(node.label || node.node_id || '未命名分子')}</h3></header>
      ${node.structure_svg ? `<div class="mol-structure">${node.structure_svg}</div>` : ''}
      <section class="detail-section"><div class="kv"><span class="k">分子式</span><span class="v">${esc(node.formula || '未记录')}</span></div>
      <div class="kv"><span class="k">Canonical</span><span class="v"><code>${esc(node.canonical_isomeric_smiles || node.smiles || '未记录')}</code></span></div>
      <div class="kv"><span class="k">角色</span><span class="v">${esc(node.role || 'intermediate')}</span></div></section></article>`;
  }
  function branchOverview(branch) {
    const lane = laneByBranch.get(branch.branch_id) || {};
    return `<article><header><p class="detail-kind">路线分支 · ${esc(tierLabel(lane.proof_tier))}</p><h3 class="detail-title">${esc(branch.title || branch.branch_id)}</h3></header>
      <div class="notice">${esc(branch.summary || branch.recommendation || '没有路线摘要。')}</div>
      <section class="detail-section"><div class="kv"><span class="k">步骤</span><span class="v">${esc((lane.step_ids || []).length)}</span></div><div class="kv"><span class="k">DAG</span><span class="v">${lane.acyclic === false ? '检测到环路' : '无环'}</span></div><div class="kv"><span class="k">执行状态</span><span class="v">${branch.solved && branch.executable ? '已验证' : '探索建议'}</span></div></section></article>`;
  }
  function nodeNames(ids) {
    return (ids || []).map(id => moleculeNodes.get(id)?.label || moleculeNodes.get(id)?.smiles || id).join(' + ') || '未记录';
  }

  function renderEvidence(entity, host) {
    const value = entity.value || {};
    const refs = Array.isArray(value.source_refs) ? value.source_refs : [];
    const conflicts = Array.isArray(value.conflicts) ? value.conflicts : [];
    const support = Array.isArray(value.support_records) ? value.support_records : [];
    host.innerHTML = `<article><header><p class="detail-kind">${COPY.consensus}</p><h3 class="detail-title">证据与来源链</h3></header>
      <section class="detail-section"><h3>${COPY.support}</h3><div class="trace-list">${support.length ? support.map(rawRow => {
        const row = rawRow && typeof rawRow === 'object' ? rawRow : {};
        const recordRefs = [...new Set([
          ...(Array.isArray(row.source_refs) ? row.source_refs : []),
          ...(Array.isArray(row.evidence_refs) ? row.evidence_refs : [])
        ].map(String).filter(Boolean))];
        const details = [
          row.claim || row.condition_summary || '',
          row.evidence_level ? `证据 ${row.evidence_level}` : '',
          row.confidence ? `可信度 ${row.confidence}` : '',
          recordRefs.length ? recordRefs.join(' · ') : ''
        ].filter(Boolean).join('；');
        return `<div class="trace-row"><strong>${esc(row.support_group || row.source_channel || '来源')}</strong><span>${esc(details)}</span></div>`;
      }).join('') : '<div class="empty">当前节点没有独立支持组明细。</div>'}</div></section>
      <section class="detail-section"><h3>${COPY.conflicts}</h3><div class="trace-list">${conflicts.length ? conflicts.map(rawRow => {
        const row = rawRow && typeof rawRow === 'object' ? rawRow : {};
        const values = Array.isArray(row.values) ? row.values.map(String).join(' / ') : '';
        return `<div class="trace-row"><strong>${esc(row.field || '字段')}</strong><span>${esc(values || row.reason || '')}</span></div>`;
      }).join('') : '<div class="empty">没有记录到条件冲突。</div>'}</div></section>
      <section class="detail-section"><h3>来源引用</h3><div class="trace-list">${refs.length ? refs.map(ref => `<div class="trace-row">${esc(ref)}</div>`).join('') : '<div class="empty">来源未记录。</div>'}</div></section>
      <div class="notice">${COPY.correlated}; multiple role reports never count as independent literature sources by themselves.</div></article>`;
  }
  function renderAlternatives(entity, host) {
    if (entity.type !== 'reaction') {
      host.innerHTML = `<div class="empty-state"><strong>请选择反应步骤</strong><span>整路线替换只绑定到后端验证过的 base step。</span></div>`;
      return;
    }
    const baseStep = state.activeReplacement
      ? steps.get(state.activeReplacement.baseStepId) || entity.value
      : entity.value;
    const records = (forest.replacement_validation?.records || []).filter(row => row.base_step_id === baseStep.step_id);
    if (!records.length) {
      host.innerHTML = `<div class="empty-state"><strong>${COPY.noReplacement}</strong><span>${COPY.noSplice}</span></div>`;
      return;
    }
    host.innerHTML = `<article><header><p class="detail-kind">完整路线替换</p><h3 class="detail-title">${esc(baseStep.label || baseStep.step_id)}</h3></header>
      <div class="notice">只有预先通过后端 AND/OR connectivity、stock 与 reaction-proof 整路线重验的分支可预览。${COPY.noSplice}</div><div class="trace-list">${records.map(row => {
        const valid = row.validated === true && row.connectivity_revalidated === true && row.stock_closure_revalidated === true && row.reaction_proof_revalidated === true;
        const replacementBranchId = String(row.revalidated_route_branch_id || row.candidate_branch_id || '');
        const previewable = valid && branches.has(replacementBranchId) && laneByBranch.has(replacementBranchId);
        const active = state.activeReplacement?.replacementId === String(row.replacement_id || row.candidate_id || '');
        const reasons = (row.reasons || []).join(', ') || (previewable ? COPY.resolvedReplacement : valid ? 'revalidated_route_branch_missing' : COPY.noClosedRoute);
        const content = `<strong>${esc(row.replacement_hyperedge_id || row.candidate_id || '替换候选')} · ${previewable ? 'AND/OR ROUTE REVALIDATED' : 'REJECTED'}</strong><span>${esc(reasons)}</span>`;
        return previewable
          ? `<button class="trace-row replacement-option is-valid${active ? ' is-active' : ''}" type="button" data-replacement-preview data-replacement-id="${esc(row.replacement_id || row.candidate_id || '')}" data-replacement-branch-id="${esc(replacementBranchId)}" data-replacement-step-id="${esc(row.candidate_step_id || '')}" data-base-step-id="${esc(baseStep.step_id)}" aria-pressed="${active}">${content}</button>`
          : `<section class="trace-row projection-warning" data-replacement-id="${esc(row.replacement_id || row.candidate_id || '')}">${content}</section>`;
      }).join('')}</div>${state.activeReplacement ? '<div class="detail-section"><button class="detail-action" type="button" data-replacement-reset>恢复原路线</button></div>' : ''}</article>`;
  }

  function previewReplacement(target) {
    const replacementBranchId = String(target.dataset.replacementBranchId || '');
    const replacementId = String(target.dataset.replacementId || '');
    const baseStepId = String(target.dataset.baseStepId || state.selectedStepId || '');
    const lane = laneByBranch.get(replacementBranchId);
    if (!replacementId || !branches.has(replacementBranchId) || !lane) return;
    const previous = state.activeReplacement;
    const replacementStepId = String(target.dataset.replacementStepId || '') || lane.step_ids?.[0] || '';
    const reactionNode = [...graphNodes.values()].find(node =>
      node.node_type === 'reaction' && node.reaction_step_id === replacementStepId
    );
    state.activeReplacement = {
      replacementId,
      replacementBranchId,
      replacementStepId,
      baseBranchId: previous?.baseBranchId || state.selectedBranchId,
      baseStepId: previous?.baseStepId || baseStepId,
      previousMode: previous?.previousMode || state.mode
    };
    state.selectedBranchId = replacementBranchId;
    state.selectedStepId = replacementStepId;
    state.selectedGraphNodeId = reactionNode?.graph_node_id || '';
    state.selectedInstanceId = reactionNode ? `${replacementBranchId}::${reactionNode.graph_node_id}` : '';
    state.mode = 'current';
    state.detailTab = 'step';
    state.inspectorOpen = true;
    applyPersistentChromeState();
    rerenderForControls();
    announce('已打开完整的后端重验替换路线预览');
  }

  function restoreReplacementPreview({ render = true } = {}) {
    const preview = state.activeReplacement;
    if (!preview) return;
    state.activeReplacement = null;
    state.selectedBranchId = branches.has(preview.baseBranchId) ? preview.baseBranchId : defaultBranchId;
    state.selectedStepId = steps.has(preview.baseStepId)
      ? preview.baseStepId : laneByBranch.get(state.selectedBranchId)?.step_ids?.[0] || '';
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.mode = oneOf(preview.previousMode, ['clusters', 'shared', 'current'], 'current');
    state.detailTab = 'alternatives';
    if (render) rerenderForControls();
    persistState();
    announce('已恢复原路线');
  }

  function applyPersistentChromeState() {
    document.body.dataset.theme = state.theme;
    document.body.classList.remove('layout-explore', 'layout-focus', 'layout-review');
    document.body.classList.add(`layout-${state.layoutPreset}`);
    document.body.classList.toggle('nav-collapsed', !state.navOpen);
    document.body.classList.toggle('inspector-collapsed', !state.inspectorOpen);
    document.body.classList.toggle('is-nav-collapsed', !state.navOpen);
    document.body.classList.toggle('is-inspector-collapsed', !state.inspectorOpen);
    document.body.classList.toggle('nav-open', state.navOpen);
    document.body.classList.toggle('inspector-open', state.inspectorOpen);
    document.documentElement.style.setProperty('--nav-width', `${state.navWidth}px`);
    document.documentElement.style.setProperty('--inspector-width', `${state.inspectorWidth}px`);
    element('themeToggle').setAttribute('aria-pressed', String(state.theme === 'dark'));
    element('layoutPreset').value = state.layoutPreset;
    element('orientationSelect').value = state.orientation;
    element('densitySelect').value = state.density;
    element('edgeStyleSelect').value = state.edgeStyle;
    element('labelModeSelect').value = state.labelMode;
    element('navResizeHandle')?.setAttribute('aria-valuenow', String(state.navWidth));
    element('inspectorResizeHandle')?.setAttribute('aria-valuenow', String(state.inspectorWidth));
    const embedded = document.body.classList.contains('embedded-route');
    const narrowReview = state.layoutPreset === 'review' && !matchMedia('(max-width: 1023px)').matches;
    const navVisible = state.navOpen && state.layoutPreset === 'explore' && !embedded;
    const inspectorVisible = state.layoutPreset !== 'focus' && (state.inspectorOpen || narrowReview);
    element('navToggle').setAttribute('aria-expanded', String(navVisible));
    element('inspectorToggle').setAttribute('aria-expanded', String(inspectorVisible));
    setPanelExposure(element('navPanel'), navVisible);
    setPanelExposure(element('inspectorPanel'), inspectorVisible);
    setPanelExposure(element('navResizeHandle'), navVisible && !isMobileDrawerLayout());
    setPanelExposure(element('inspectorResizeHandle'), inspectorVisible && !isInspectorOverlayLayout());
  }

  function setPanelExposure(panel, visible) {
    if (!panel) return;
    panel.setAttribute('aria-hidden', String(!visible));
    panel.toggleAttribute('inert', !visible);
  }

  function focusPanelReturn(preferredId) {
    const preferred = element(preferredId);
    if (preferred && preferred.getClientRects().length && !preferred.closest('[inert]')) preferred.focus();
    else element('graphViewport')?.focus();
  }

  function announce(message) {
    clearTimeout(liveAnnouncementTimer);
    let live = document.getElementById('graphSelectionStatus');
    if (!live) {
      live = document.createElement('div');
      live.id = 'graphSelectionStatus';
      live.className = 'sr-only';
      live.setAttribute('aria-live', 'polite');
      document.body.append(live);
    }
    liveAnnouncementTimer = setTimeout(() => { live.textContent = message; }, 30);
  }

  function rerenderForControls({ restoreBranchFocus = '', restoreControl = null } = {}) {
    ensureSelectedBranchMatchesFilters({ includeReplacementPreview: Boolean(state.activeReplacement) });
    renderFilters();
    renderBranchGroups({ restoreFocusId: restoreBranchFocus });
    renderGraph();
    renderDetail();
    persistState();
    if (restoreControl) {
      [...document.querySelectorAll(`[data-${restoreControl.kind}-filter]`)].find(row =>
        row.dataset[`${restoreControl.kind}Filter`] === restoreControl.value)?.focus();
    }
  }

  function ensureSelectedBranchMatchesFilters({ includeReplacementPreview = false } = {}) {
    const visible = filteredLanes({ includeReplacementPreview });
    if (visible.some(lane => lane.branch_id === state.selectedBranchId)) return;
    state.activeReplacement = null;
    const next = visible.slice().sort((left, right) =>
      branchDisplayScore(right) - branchDisplayScore(left)
      || stableTextCompare(left.branch_id, right.branch_id))[0];
    state.selectedBranchId = next?.branch_id || '';
    state.selectedStepId = next?.step_ids?.[0] || '';
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
  }

  function clearReplacementPreviewForFilterChange() {
    if (!state.activeReplacement) return;
    state.activeReplacement = null;
    state.selectedStepId = '';
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
  }

  function bindEvents() {
    document.addEventListener('click', event => {
      const target = event.target.closest('button, [data-graph-node-id], .graph-minimap');
      if (!target) return;
      if (target.dataset.replacementPreview !== undefined) { previewReplacement(target); return; }
      if (target.dataset.replacementReset !== undefined) { restoreReplacementPreview(); return; }
      if (target.dataset.graphNodeId) { selectGraphNode(target.dataset.graphNodeId, { branchId: target.dataset.branchId || '', instanceId: target.dataset.instanceId || '' }); return; }
      if (target.dataset.branchId) { selectBranch(target.dataset.branchId); return; }
      if (target.dataset.graphMode) {
        const nextMode = target.dataset.graphMode;
        if (state.activeReplacement && nextMode !== 'current') restoreReplacementPreview({ render: false });
        state.mode = nextMode;
        rerenderForControls();
        return;
      }
      if (target.dataset.overviewToggle !== undefined) { state.showAllOverview = !state.showAllOverview; renderGraph(); return; }
      if (target.dataset.expandGroup) { state.expandedGroups.add(target.dataset.expandGroup); renderBranchGroups(); return; }
      if (target.dataset.collapseGroup) { state.expandedGroups.delete(target.dataset.collapseGroup); renderBranchGroups(); return; }
      if (target.dataset.branchFilter) { clearReplacementPreviewForFilterChange(); state.branchFilter = target.dataset.branchFilter; rerenderForControls(); return; }
      if (target.dataset.stageFilter) { clearReplacementPreviewForFilterChange(); state.stageFilter = target.dataset.stageFilter; rerenderForControls(); return; }
      if (target.dataset.proofFilter) {
        clearReplacementPreviewForFilterChange();
        const tier = target.dataset.proofFilter;
        state.proofFilters.has(tier) ? state.proofFilters.delete(tier) : state.proofFilters.add(tier);
        rerenderForControls({ restoreControl: { kind: 'proof', value: tier } }); return;
      }
      if (target.dataset.kindFilter) {
        clearReplacementPreviewForFilterChange();
        const kind = target.dataset.kindFilter;
        state.kindFilters.has(kind) ? state.kindFilters.delete(kind) : state.kindFilters.add(kind);
        rerenderForControls({ restoreControl: { kind: 'kind', value: kind } }); return;
      }
      if (target.dataset.edgeFilter) { state.edgeFilter = target.dataset.edgeFilter; rerenderForControls({ restoreControl: { kind: 'edge', value: target.dataset.edgeFilter } }); return; }
      if (target.dataset.filterReset !== undefined) {
        clearReplacementPreviewForFilterChange();
        state.stageFilter = 'all'; state.branchFilter = 'all'; state.proofFilters = new Set(allProofTiers);
        state.kindFilters = new Set(allKinds); state.edgeFilter = 'all'; state.query = '';
        element('branchSearch').value = ''; rerenderForControls(); return;
      }
      if (target.dataset.graphAction === 'fit') { fitGraph(); return; }
      if (target.dataset.graphAction === 'zoom-in') { zoomGraph(1.2); return; }
      if (target.dataset.graphAction === 'zoom-out') { zoomGraph(1 / 1.2); return; }
      if (target.dataset.graphAction === 'reset') { resetGraph(); return; }
      if (target.dataset.detailTab) { state.detailTab = target.dataset.detailTab; renderDetail(); persistState(); return; }
      if (target.id === 'themeToggle') { state.theme = state.theme === 'dark' ? 'light' : 'dark'; applyPersistentChromeState(); persistState(); return; }
      if (target.id === 'navToggle') { state.navOpen = !state.navOpen; if (isMobileDrawerLayout() && state.navOpen) state.inspectorOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.navOpen ? 'nav' : 'graph'); persistState(); requestAnimationFrame(() => fitGraph({ readable: state.mode === 'current' })); return; }
      if (target.id === 'inspectorToggle') { state.inspectorOpen = !state.inspectorOpen; if (isMobileDrawerLayout() && state.inspectorOpen) state.navOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.inspectorOpen ? 'inspector' : 'graph'); persistState(); if (!isInspectorOverlayLayout()) requestAnimationFrame(() => fitGraph({ readable: state.mode === 'current' })); return; }
      if (target.dataset.closePanel === 'inspector') { state.inspectorOpen = false; applyPersistentChromeState(); updateMobileNavigation('graph'); focusPanelReturn('inspectorToggle'); persistState(); return; }
      if (target.dataset.closeMobilePanels !== undefined) { closeMobilePanels({ restoreFocus: true }); return; }
      if (target.dataset.mobilePanel) { openMobilePanel(target.dataset.mobilePanel); return; }
      if (target.classList.contains('graph-minimap')) recenterFromMinimap(event);
    });
    document.addEventListener('change', event => {
      const target = event.target;
      if (target.id === 'layoutPreset') state.layoutPreset = target.value;
      else if (target.id === 'orientationSelect') state.orientation = target.value;
      else if (target.id === 'densitySelect') state.density = target.value;
      else if (target.id === 'edgeStyleSelect') state.edgeStyle = target.value;
      else if (target.id === 'labelModeSelect') state.labelMode = target.value;
      else return;
      applyPersistentChromeState(); rerenderForControls();
    });
    element('branchSearch').addEventListener('input', event => {
      clearReplacementPreviewForFilterChange();
      state.query = event.target.value;
      rerenderForControls();
    });
    bindViewportEvents();
    bindResizeHandles();
    document.addEventListener('keydown', handleKeyboard);
    window.addEventListener('resize', debounce(() => renderGraph(), 120));
  }

  function bindViewportEvents() {
    const viewport = element('graphViewport');
    const suppressNextPointerClick = pointerId => {
      clearTimeout(suppressGraphClickTimer);
      suppressGraphClickPointerId = pointerId;
      suppressGraphClickTimer = setTimeout(() => {
        if (suppressGraphClickPointerId === pointerId) suppressGraphClickPointerId = null;
      }, 0);
    };
    const finishPan = (event, { cancelled = false } = {}) => {
      if (!panSession || panSession.pointerId !== event.pointerId) return;
      const session = panSession;
      panSession = null;
      if (session.moved) {
        if (!cancelled) {
          state.panX = session.panX + event.clientX - session.x;
          state.panY = session.panY + event.clientY - session.y;
        }
        if (panAnimationFrame) cancelAnimationFrame(panAnimationFrame);
        panAnimationFrame = 0;
        applyViewportTransform();
        if (!cancelled) suppressNextPointerClick(event.pointerId);
        event.preventDefault();
      }
      viewport.classList.remove('is-pan-ready', 'is-panning');
      if (session.captured && viewport.hasPointerCapture?.(event.pointerId)) {
        viewport.releasePointerCapture(event.pointerId);
      }
    };
    viewport.addEventListener('pointerdown', event => {
      if (event.isPrimary === false || event.button !== 0) return;
      clearTimeout(suppressGraphClickTimer);
      suppressGraphClickPointerId = null;
      if (panSession) return;
      if (event.target.closest('.graph-minimap, button, input, select, textarea, a, summary')) return;
      panSession = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        panX: state.panX,
        panY: state.panY,
        moved: false,
        captured: false
      };
      viewport.classList.add('is-pan-ready');
    });
    window.addEventListener('pointermove', event => {
      if (!panSession || panSession.pointerId !== event.pointerId) return;
      const deltaX = event.clientX - panSession.x;
      const deltaY = event.clientY - panSession.y;
      if (!panSession.moved) {
        if (Math.hypot(deltaX, deltaY) < PAN_DRAG_THRESHOLD_PX) return;
        panSession.moved = true;
        try {
          viewport.setPointerCapture(event.pointerId);
          panSession.captured = true;
        } catch (_) { /* capture can fail if the pointer ended between frames */ }
        viewport.classList.remove('is-pan-ready');
        viewport.classList.add('is-panning');
      }
      event.preventDefault();
      state.panX = panSession.panX + deltaX;
      state.panY = panSession.panY + deltaY;
      if (!panAnimationFrame) {
        panAnimationFrame = requestAnimationFrame(() => {
          panAnimationFrame = 0;
          applyPanTransform();
        });
      }
    }, { capture: true, passive: false });
    window.addEventListener('pointerup', event => finishPan(event), true);
    window.addEventListener('pointercancel', event => finishPan(event, { cancelled: true }), true);
    viewport.addEventListener('lostpointercapture', event => {
      if (event.target === viewport) finishPan(event, { cancelled: true });
    });
    window.addEventListener('blur', () => {
      if (!panSession) return;
      viewport.classList.remove('is-pan-ready', 'is-panning');
      panSession = null;
      if (panAnimationFrame) cancelAnimationFrame(panAnimationFrame);
      panAnimationFrame = 0;
      applyViewportTransform();
    });
    viewport.addEventListener('click', event => {
      if (suppressGraphClickPointerId === null || event.detail === 0) return;
      if (Number.isInteger(event.pointerId) && event.pointerId !== suppressGraphClickPointerId) return;
      clearTimeout(suppressGraphClickTimer);
      suppressGraphClickPointerId = null;
      event.preventDefault();
      event.stopPropagation();
    }, true);
    viewport.addEventListener('dragstart', event => event.preventDefault());
    viewport.addEventListener('selectstart', event => event.preventDefault());
    viewport.addEventListener('wheel', event => {
      if (event.target.closest('select, input')) return;
      event.preventDefault();
      zoomGraph(event.deltaY < 0 ? 1.1 : 1 / 1.1, event.clientX, event.clientY);
    }, { passive: false });
  }

  function handleKeyboard(event) {
    if (isTypingTarget(event.target)) {
      if (event.key === 'Escape') {
        if (event.target.id === 'branchSearch' && event.target.value) {
          event.target.value = '';
          state.query = '';
          rerenderForControls();
        } else {
          closeMobilePanels({ restoreFocus: true });
        }
      }
      return;
    }
    const branchButton = event.target.closest?.('[data-branch-id]');
    if (branchButton && ['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
      event.preventDefault(); moveBranchFocus(branchButton, event.key); return;
    }
    const graphNode = event.target.closest?.('[data-graph-node-id]');
    if (graphNode && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); selectGraphNode(graphNode.dataset.graphNodeId, { focus: true, branchId: graphNode.dataset.branchId || '', instanceId: graphNode.dataset.instanceId || '' }); return;
    }
    if (graphNode && event.key.startsWith('Arrow')) {
      event.preventDefault(); moveGraphFocus(graphNode, event.key); return;
    }
    if (event.target.id === 'graphViewport' && (event.key === 'Enter' || event.key.startsWith('Arrow'))) {
      event.preventDefault(); document.querySelector('.graph-node[tabindex="0"], .graph-node')?.focus(); return;
    }
    const tab = event.target.closest?.('[data-detail-tab]');
    if (tab && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
      event.preventDefault(); moveTabFocus(tab, event.key); return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '/') { event.preventDefault(); element('branchSearch').focus(); return; }
    if (event.key.toLowerCase() === 'f') { event.preventDefault(); fitGraph(); return; }
    if (event.key === '0') { event.preventDefault(); resetGraph(); return; }
    if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomGraph(1.2); return; }
    if (event.key === '-') { event.preventDefault(); zoomGraph(1 / 1.2); return; }
    if (event.key === 'Escape') { closeMobilePanels({ restoreFocus: true }); }
  }

  function moveBranchFocus(current, key) {
    const rows = [...document.querySelectorAll('.branch-card[data-branch-id]')];
    let index = rows.indexOf(current);
    if (key === 'Home') index = 0;
    else if (key === 'End') index = rows.length - 1;
    else index = clamp(index + (key === 'ArrowDown' ? 1 : -1), 0, rows.length - 1);
    rows[index]?.focus();
  }
  function moveTabFocus(current, key) {
    const rows = [...document.querySelectorAll('[data-detail-tab]')];
    let index = rows.indexOf(current);
    if (key === 'Home') index = 0;
    else if (key === 'End') index = rows.length - 1;
    else index = (index + (key === 'ArrowRight' ? 1 : -1) + rows.length) % rows.length;
    state.detailTab = rows[index].dataset.detailTab; renderDetail(); rows[index].focus(); persistState();
  }
  function moveGraphFocus(current, key) {
    if (!renderModel) return;
    const currentPos = renderModel.positions.get(current.dataset.instanceId);
    if (!currentPos) return;
    const center = pos => ({ x: pos.x + pos.w / 2, y: pos.y + pos.h / 2 });
    const origin = center(currentPos);
    const candidates = renderModel.instances.map(instance => ({ instance, pos: renderModel.positions.get(instance.instanceId) }))
      .filter(row => row.pos && row.instance.instanceId !== current.dataset.instanceId)
      .map(row => ({ ...row, point: center(row.pos) }))
      .filter(row => key === 'ArrowRight' ? row.point.x > origin.x : key === 'ArrowLeft' ? row.point.x < origin.x : key === 'ArrowDown' ? row.point.y > origin.y : row.point.y < origin.y)
      .sort((a, b) => Math.hypot(a.point.x-origin.x, a.point.y-origin.y) - Math.hypot(b.point.x-origin.x, b.point.y-origin.y));
    const next = candidates[0]?.instance;
    if (next) [...document.querySelectorAll('[data-instance-id]')].find(row => row.dataset.instanceId === next.instanceId)?.focus();
  }

  function bindResizeHandles() {
    for (const [id, side] of [['navResizeHandle','nav'], ['inspectorResizeHandle','inspector']]) {
      const handle = element(id); if (!handle) continue;
      handle.addEventListener('pointerdown', event => {
        resizeSession = { side, pointerId: event.pointerId, startX: event.clientX, width: side === 'nav' ? state.navWidth : state.inspectorWidth };
        handle.setPointerCapture(event.pointerId);
      });
      handle.addEventListener('pointermove', event => {
        if (!resizeSession || resizeSession.pointerId !== event.pointerId) return;
        const delta = event.clientX - resizeSession.startX;
        if (side === 'nav') state.navWidth = clamp(resizeSession.width + delta, 240, 460);
        else state.inspectorWidth = clamp(resizeSession.width - delta, 320, 560);
        applyPersistentChromeState(); handle.setAttribute('aria-valuenow', String(side === 'nav' ? state.navWidth : state.inspectorWidth));
      });
      handle.addEventListener('pointerup', () => { resizeSession = null; persistState(); requestAnimationFrame(() => fitGraph({ readable: state.mode === 'current' })); });
      handle.addEventListener('keydown', event => {
        if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
        event.preventDefault();
        const delta = event.key === 'ArrowLeft' ? -16 : event.key === 'ArrowRight' ? 16 : 0;
        if (side === 'nav') state.navWidth = event.key === 'Home' ? 240 : event.key === 'End' ? 460 : clamp(state.navWidth + delta, 240, 460);
        else state.inspectorWidth = event.key === 'Home' ? 320 : event.key === 'End' ? 560 : clamp(state.inspectorWidth - delta, 320, 560);
        applyPersistentChromeState();
        handle.setAttribute('aria-valuenow', String(side === 'nav' ? state.navWidth : state.inspectorWidth));
        persistState();
      });
    }
  }

  function openMobilePanel(panel) {
    state.navOpen = panel === 'nav';
    state.inspectorOpen = panel === 'inspector';
    applyPersistentChromeState();
    updateMobileNavigation(panel);
    persistState();
    requestAnimationFrame(() => {
      if (panel === 'nav') element('branchSearch')?.focus();
      else if (panel === 'inspector') element('detailTabs')?.querySelector('[aria-selected="true"]')?.focus();
      else element('graphViewport')?.focus();
    });
  }
  function closeMobilePanels({ restoreFocus = false } = {}) {
    if (!isMobileDrawerLayout()) return;
    state.navOpen = false;
    state.inspectorOpen = false;
    applyPersistentChromeState();
    updateMobileNavigation('graph');
    persistState();
    if (restoreFocus) element('graphViewport')?.focus();
  }
  function updateMobileNavigation(activePanel) {
    document.querySelectorAll('[data-mobile-panel]').forEach(button => {
      const active = button.dataset.mobilePanel === activePanel;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-expanded', String(active && activePanel !== 'graph'));
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
  }
  function isInspectorOverlayLayout() { return matchMedia('(max-width: 1699px)').matches; }
  function isMobileDrawerLayout() { return matchMedia('(max-width: 1023px)').matches; }
  function recenterFromMinimap(event) {
    if (!renderModel) return;
    const rect = element('graphMinimap').getBoundingClientRect();
    const worldX = ((event.clientX - rect.left) / rect.width) * renderModel.bounds.w;
    const worldY = ((event.clientY - rect.top) / rect.height) * renderModel.bounds.h;
    const viewport = element('graphViewport');
    state.panX = viewport.clientWidth / 2 - worldX * state.zoom;
    state.panY = viewport.clientHeight / 2 - worldY * state.zoom;
    applyViewportTransform();
  }
  function debounce(fn, delay) {
    let timer = 0;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
  }

  function init() {
    ensureSelectedBranchMatchesFilters();
    persistState();
    renderChrome();
    const integrityCheck = verifyEmbeddedDeliveryDigest();
    updateMobileNavigation(state.navOpen ? 'nav' : state.inspectorOpen ? 'inspector' : 'graph');
    renderFilters();
    renderBranchGroups();
    renderDetail();
    bindEvents();
    requestAnimationFrame(() => {
      renderGraph();
      integrityCheck.finally(() => requestAnimationFrame(notifyParentReady));
    });
  }

  function notifyParentReady() {
    if (window.parent === window) return;
    const token = new URLSearchParams(location.search).get('parent_token') || '';
    window.parent.postMessage({
      type: 'autoplanner.route_forest.ready.v1',
      token,
      schema_version: forest.schema_version || '',
      delivery_sha256: forest.delivery_sha256 || '',
      source_forest_sha256: forest.source_forest_sha256 || '',
      integrity_status: deliveryIntegrityStatus,
      byte_integrity_status: deliveryIntegrityStatus,
      external_closeout_authority: false,
      counts: forest.counts || {}
    }, '*');
  }
  init();
})();
