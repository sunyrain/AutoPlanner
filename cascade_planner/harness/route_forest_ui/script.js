(() => {
  'use strict';

  const forestDataText = document.getElementById('forest-data')?.textContent || '{}';
  const forest = JSON.parse(forestDataText);
  const STORAGE_KEY = 'autoplanner.route-forest-ui.v2';
  const COPY = Object.freeze({
    consensus: 'Multi-source consensus audit',
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
  const graphNodes = new Map((graph.nodes || []).map(row => [row.graph_node_id, row]));
  const moleculeNodes = new Map((forest.nodes || []).map(row => [row.node_id, row]));
  const steps = new Map((forest.steps || []).map(row => [row.step_id, row]));
  const branches = new Map((forest.branches || []).map(row => [row.branch_id, row]));
  const laneByBranch = new Map((lanesProjection.lanes || []).map(row => [row.branch_id, row]));
  const edgeById = new Map((graph.edges || []).map(row => [row.edge_id, row]));
  const layoutByNode = new Map((layout.nodes || []).map(row => [row.graph_node_id, row]));
  const persisted = loadState();
  const allProofTiers = unique((lanesProjection.lanes || []).map(row => row.proof_tier).filter(Boolean));
  const allKinds = unique((lanesProjection.lanes || []).map(row => row.kind).filter(Boolean));
  const defaultBranchId = forest.primary_branch_id
    || (lanesProjection.lanes || []).find(row => row.is_primary)?.branch_id
    || (lanesProjection.lanes || [])[0]?.branch_id || '';
  const initialBranchId = persisted.selectedBranchId && branches.has(persisted.selectedBranchId)
    ? persisted.selectedBranchId : defaultBranchId;

  const state = {
    mode: oneOf(persisted.mode, ['clusters', 'shared', 'current'], 'clusters'),
    selectedBranchId: initialBranchId,
    selectedGraphNodeId: '',
    selectedInstanceId: '',
    selectedStepId: laneByBranch.get(initialBranchId)?.step_ids?.[0] || '',
    detailTab: oneOf(persisted.detailTab, ['step', 'evidence', 'alternatives'], 'step'),
    query: '',
    branchFilter: oneOf(persisted.branchFilter, ['all', 'verified', 'evidence', 'advisory', 'diagnostic'], 'all'),
    proofFilters: new Set(Array.isArray(persisted.proofFilters) ? persisted.proofFilters.filter(tier => allProofTiers.includes(tier)) : allProofTiers),
    kindFilters: new Set(Array.isArray(persisted.kindFilters) ? persisted.kindFilters.filter(kind => allKinds.includes(kind)) : allKinds),
    edgeFilter: oneOf(persisted.edgeFilter, ['all', 'selected'], 'all'),
    orientation: oneOf(persisted.orientation, ['horizontal', 'vertical'], 'horizontal'),
    density: oneOf(persisted.density, ['comfortable', 'compact', 'overview'], 'comfortable'),
    edgeStyle: oneOf(persisted.edgeStyle, ['trust', 'simple', 'contrast'], 'trust'),
    labelMode: oneOf(persisted.labelMode, ['semantic', 'full', 'minimal'], 'semantic'),
    layoutPreset: oneOf(persisted.layoutPreset, ['explore', 'focus', 'review'], 'explore'),
    theme: oneOf(persisted.theme, ['light', 'dark'], preferredTheme()),
    navOpen: Object.hasOwn(persisted, 'navOpen') ? persisted.navOpen !== false : !matchMedia('(max-width: 1023px)').matches,
    inspectorOpen: Object.hasOwn(persisted, 'inspectorOpen') ? persisted.inspectorOpen !== false : !matchMedia('(max-width: 1439px)').matches,
    navWidth: clamp(Number(persisted.navWidth) || 300, 240, 460),
    inspectorWidth: clamp(Number(persisted.inspectorWidth) || 380, 320, 560),
    zoom: 1,
    panX: 0,
    panY: 0,
    activeReplacement: null
  };
  let renderModel = null;
  let panSession = null;
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
  function safeStorageGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function safeStorageSet(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* sandboxed embed */ } }
  function loadState() {
    try { return JSON.parse(safeStorageGet(STORAGE_KEY) || '{}'); } catch (_) { return {}; }
  }
  function persistState() {
    safeStorageSet(STORAGE_KEY, JSON.stringify({
      mode: state.mode, selectedBranchId: state.selectedBranchId, detailTab: state.detailTab,
      branchFilter: state.branchFilter, proofFilters: [...state.proofFilters],
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
  function targetName() { return forest.target?.name || forest.case_id || '目标分子'; }
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

  function renderChrome() {
    const primary = forest.primary_selection || {};
    const primaryBranch = branches.get(primary.primary_branch_id || forest.primary_branch_id) || {};
    const verified = primary.status === 'deterministically_verified'
      && primary.proof_level === 'parent_route_proof'
      && primary.advisory_only === false
      && primaryBranch.solved === true
      && primaryBranch.executable === true
      && primaryBranch.advisory_only === false
      && primaryBranch.not_parent_route_proof === false;
    element('pageTitle').textContent = `${targetName()} · 路线工作台`;
    const verdict = element('verdictBadge');
    verdict.textContent = verified ? '父路线已验证' : '父路线未闭合';
    verdict.dataset.status = verified ? 'verified' : 'unresolved';
    verdict.classList.toggle('status-badge--verified', verified);
    verdict.classList.toggle('status-badge--unresolved', !verified);
    const counts = forest.counts || {};
    element('overviewMetrics').innerHTML = [
      ['分支', counts.branches ?? branches.size], ['反应', counts.reaction_nodes ?? steps.size],
      ['分子', counts.nodes ?? moleculeNodes.size], ['显式边', counts.dependency_edges ?? (graph.edges || []).length]
    ].map(([label, value]) => `<span class="metric-chip"><strong>${esc(value)}</strong>${esc(label)}</span>`).join('');
    renderIntegrityStatus('pending');
    renderEvidenceStats();
    applyPersistentChromeState();
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
      return deliveryIntegrityStatus;
    }
    if (!globalThis.crypto?.subtle || typeof TextEncoder === 'undefined') {
      deliveryIntegrityStatus = 'unavailable';
      renderIntegrityStatus(deliveryIntegrityStatus);
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
    return deliveryIntegrityStatus;
  }

  function renderEvidenceStats() {
    const counts = forest.counts || {};
    const literature = forest.run_trace?.literature_counts || {};
    const rows = [
      ['信源候选', literature.source_candidates ?? 0], ['文献/图像链', counts.visual_chains ?? 0],
      ['共识路线', counts.route_consensus_proposals ?? 0], ['后端替换', forest.replacement_validation?.validated_count ?? 0]
    ];
    element('evidenceStats').innerHTML = rows.map(([label, value]) =>
      `<div class="evidence-stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join('');
  }

  function renderFilters() {
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
    document.querySelectorAll('[data-graph-mode]').forEach(button => {
      const active = button.dataset.graphMode === state.mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
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
      return `<section class="branch-group" data-branch-kind="${esc(group.kind)}">
        <div class="branch-group-heading"><strong>${esc(group.label)}</strong><span>${rows.length}</span></div>
        <div class="branch-group-list">${rows.map(lane => branchCard(lane)).join('')}</div>
      </section>`;
    }).join('') || `<div class="empty-state"><strong>没有匹配路线</strong><span>清除搜索或放宽证明层级筛选。</span></div>`;
    updateBranchRovingTabindex();
    if (restoreFocusId) {
      [...host.querySelectorAll('.branch-card[data-branch-id]')].find(row => row.dataset.branchId === restoreFocusId)?.focus();
    }
  }

  function branchCard(lane) {
    const selected = lane.branch_id === state.selectedBranchId;
    const branch = branches.get(lane.branch_id) || {};
    const stateLabel = lane.solved && lane.executable && !lane.advisory_only ? '已验证' : '建议';
    return `<button class="branch-card ${tierClass(lane.proof_tier)}${selected ? ' is-selected' : ''}" type="button"
      data-branch-id="${esc(lane.branch_id)}" aria-current="${selected ? 'true' : 'false'}" tabindex="-1">
      <span class="branch-card-title">${esc(lane.title || lane.branch_id)}</span>
      <span class="branch-card-meta">${esc((lane.step_ids || []).length)} 步 · ${esc(tierLabel(lane.proof_tier))}</span>
      <span class="branch-card-badges">${lane.is_primary ? '<span class="branch-badge">主分支</span>' : ''}<span class="branch-badge">${esc(stateLabel)}</span><span class="branch-badge">${esc(synthesisLabel(branch.synthesis_class))}</span></span>
    </button>`;
  }

  function updateBranchRovingTabindex() {
    const rows = [...document.querySelectorAll('.branch-card[data-branch-id]')];
    const active = rows.find(row => row.dataset.branchId === state.selectedBranchId) || rows[0];
    rows.forEach(row => row.tabIndex = row === active ? 0 : -1);
  }

  function densityMetrics() {
    if (state.density === 'compact') return { layerGap: 188, rowGap: 78, laneGap: 18, componentGap: 18, nodeScale: .88 };
    if (state.density === 'overview') return { layerGap: 150, rowGap: 58, laneGap: 12, componentGap: 12, nodeScale: .72 };
    return { layerGap: 238, rowGap: 108, laneGap: 28, componentGap: 26, nodeScale: 1 };
  }

  function buildGraphModel() {
    const lanes = filteredLanes({ includeReplacementPreview: true });
    if (state.mode === 'shared') return buildSharedModel(lanes);
    const selected = lanes.find(lane => lane.branch_id === state.selectedBranchId);
    const activeLanes = state.mode === 'current' ? (selected ? [selected] : []) : lanes;
    return buildLaneModel(activeLanes);
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
      const wrap = state.orientation === 'vertical'
        ? Math.min(verticalWrap, bucket.length)
        : Math.min(horizontalWrap, bucket.length);
      const crossCount = Math.max(1, Math.ceil(bucket.length / wrap));
      bucket.forEach((row, index) => {
        const size = nodeSize(row.node, metrics.nodeScale);
        const primaryIndex = Math.floor(index / crossCount);
        const crossIndex = index % crossCount;
        const position = state.orientation === 'vertical'
          ? { x: 44 + crossIndex * cellWidth, y: layerCursor + primaryIndex * cellHeight, ...size }
          : { x: layerCursor + primaryIndex * cellWidth, y: 44 + crossIndex * cellHeight, ...size };
        positions.set(row.node.graph_node_id, position);
      });
      const bandExtent = wrap * (state.orientation === 'vertical' ? cellHeight : cellWidth);
      layerCursor += bandExtent + Math.max(72, metrics.layerGap * .55);
    }
    const instances = nodes.map(node => ({ instanceId: node.graph_node_id, graphNodeId: node.graph_node_id, branchId: '', node }));
    return finaliseModel('shared', instances, edges.map(edge => ({ ...edge, sourceInstanceId: edge.source_graph_node_id, targetInstanceId: edge.target_graph_node_id })), positions, []);
  }

  function buildLaneModel(rawLanes) {
    const lanes = effectiveLaneRows(rawLanes);
    const metrics = densityMetrics();
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
      const tileWidth = state.orientation === 'vertical'
        ? 40 + (maxLayerRows - 1) * metrics.rowGap + maximumNode.w
        : 92 + maximumLayer * metrics.layerGap + maximumNode.w;
      const tileHeight = state.orientation === 'vertical'
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
          const position = state.orientation === 'vertical'
            ? { x: 20 + rowIndex * metrics.rowGap, y: 52 + layerIndex * metrics.layerGap, ...size }
            : { x: 52 + layerIndex * metrics.layerGap, y: 52 + rowIndex * metrics.rowGap, ...size };
          relativePositions.set(instanceId, position);
        });
      }
      return { lane, byLayer, relativePositions, w: tileWidth, h: tileHeight };
    });
    const averageWidth = tiles.reduce((sum, tile) => sum + tile.w, 0) / Math.max(1, tiles.length);
    const averageHeight = tiles.reduce((sum, tile) => sum + tile.h, 0) / Math.max(1, tiles.length);
    const targetRatio = state.orientation === 'vertical' ? 1.55 : 1.68;
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
    return { w: Math.round((reaction ? 166 : 194) * scale), h: Math.round((reaction ? 70 : 78) * scale) };
  }

  function finaliseModel(mode, instances, edges, positions, decorations) {
    const boxes = [...positions.values(), ...decorations];
    const maxX = Math.max(700, ...boxes.map(row => row.x + row.w + 44));
    const maxY = Math.max(420, ...boxes.map(row => row.y + row.h + 44));
    return { mode, instances, edges, positions, decorations, bounds: { x: 0, y: 0, w: maxX, h: maxY }, packing: null };
  }

  function renderGraph({ fit = true } = {}) {
    renderModel = buildGraphModel();
    const viewport = element('graphViewport');
    viewport.dataset.graphMode = state.mode;
    const width = Math.max(1, viewport.clientWidth || 1100);
    const height = Math.max(1, viewport.clientHeight || 680);
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
    element('graphVisibleCount').textContent = state.mode === 'shared'
      ? `${visibleBranches.size || filteredLanes().length}/${(lanesProjection.lanes || []).length} 分支 · ${renderModel.instances.length} 节点 · ${renderModel.edges.length} 边`
      : `${visibleBranches.size}/${(lanesProjection.lanes || []).length} 分支 · ${renderModel.instances.length} 节点 · ${renderModel.edges.length} 边`;
    const replacementPreview = state.activeReplacement && state.mode === 'current';
    element('graphTitle').textContent = replacementPreview ? '完整替换路线预览 · 后端已重验'
      : state.mode === 'clusters' ? '全部路线 · 分支泳道'
        : state.mode === 'shared' ? '共享分子–反应超图' : '当前分支 · 完整依赖 DAG';
    element('graphSubtitle').textContent = replacementPreview
      ? '当前画布是完整的后端 AND/OR 重验分支；它不是单步拼接，也不建立父路线证明。'
      : `${COPY.explicitEdges} ${state.mode === 'clusters'
        ? '共享分子以视觉别名呈现，但 canonical ID 保持一致。'
        : '颜色仅表达 proof tier；选择轮廓不改变证明语义。'}`;
    renderMinimap();
    if (fit) fitGraph(); else applyViewportTransform();
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
    if (state.orientation === 'vertical') {
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
    const step = reaction ? steps.get(node.reaction_step_id) : null;
    const tier = reaction ? (node.proof_tier || tierOfStep(step)) : 'L0_advisory';
    const fullLabel = node.label || node.graph_node_id;
    const lines = wrapLabel(state.labelMode === 'minimal' ? (reaction ? tierLabel(tier) : '分子') : fullLabel, reaction ? 22 : 25);
    const textX = position.x + 12;
    const textY = position.y + (lines.length > 1 ? 27 : 34);
    const selected = node.graph_node_id === state.selectedGraphNodeId || (reaction && node.reaction_step_id === state.selectedStepId);
    return `<g class="graph-node dependency-${reaction ? 'reaction graph-node--reaction' : 'molecule graph-node--molecule'} ${tierClass(tier)}${selected ? ' is-selected' : ''}" data-graph-node-id="${esc(node.graph_node_id)}" data-node-type="${esc(node.node_type)}" data-node-role="${esc(node.role || '')}" data-branch-id="${esc(instance.branchId)}" data-instance-id="${esc(instance.instanceId)}" ${reaction ? `data-route-step="${esc(node.reaction_step_id)}"` : ''} tabindex="${selected ? '0' : '-1'}" role="button" aria-label="${esc(`${reaction ? '反应' : '分子'}：${fullLabel}，${tierLabel(tier)}`)}">
      <title>${esc(fullLabel)}</title><rect class="node-surface" x="${position.x}" y="${position.y}" width="${position.w}" height="${position.h}" rx="${reaction ? 12 : 24}"></rect>
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
      row.classList.toggle('is-dimmed', hasSelection && !related);
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

  function fitGraph() {
    if (!renderModel) return;
    const viewport = element('graphViewport');
    const width = Math.max(1, viewport.clientWidth || 1000);
    const height = Math.max(1, viewport.clientHeight || 620);
    const padding = 42;
    state.zoom = clamp(Math.min((width - padding * 2) / renderModel.bounds.w, (height - padding * 2) / renderModel.bounds.h), .015, 2.5);
    state.panX = (width - renderModel.bounds.w * state.zoom) / 2;
    state.panY = (height - renderModel.bounds.h * state.zoom) / 2;
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
  function applyViewportTransform() {
    const world = document.querySelector('.graph-world');
    if (world) world.setAttribute('transform', `translate(${state.panX} ${state.panY}) scale(${state.zoom})`);
    const viewport = element('graphViewport');
    viewport.dataset.labelMode = state.labelMode;
    viewport.dataset.zoomBand = state.zoom < .18 ? 'overview' : state.zoom < .5 ? 'medium' : 'detail';
    element('zoomReadout').textContent = `${Math.round(state.zoom * 100)}%`;
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
    const svg = element('graphMinimap')?.querySelector('.minimap-svg');
    const rectNode = svg?.querySelector('.minimap-viewport');
    if (!svg || !rectNode) return;
    const scale = Number(svg.dataset.minimapScale || 1);
    const viewport = element('graphViewport');
    const worldX = -state.panX / state.zoom;
    const worldY = -state.panY / state.zoom;
    rectNode.setAttribute('x', String(clamp(worldX * scale, 0, 180)));
    rectNode.setAttribute('y', String(clamp(worldY * scale, 0, 108)));
    rectNode.setAttribute('width', String(clamp((viewport.clientWidth / state.zoom) * scale, 2, 180)));
    rectNode.setAttribute('height', String(clamp((viewport.clientHeight / state.zoom) * scale, 2, 108)));
  }

  function selectBranch(branchId, { focusGraph = false } = {}) {
    if (!branches.has(branchId)) return;
    if (state.activeReplacement?.replacementBranchId !== branchId) {
      state.activeReplacement = null;
    }
    state.selectedBranchId = branchId;
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedStepId = laneByBranch.get(branchId)?.step_ids?.[0] || '';
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
    if (isDrawerLayout()) state.navOpen = false;
    applyPersistentChromeState();
    updateMobileNavigation('inspector');
    applyGraphSelection();
    renderDetail();
    persistState();
    if (isDrawerLayout()) {
      requestAnimationFrame(() => element('detailTabs')?.querySelector('[aria-selected="true"]')?.focus());
    } else {
      requestAnimationFrame(fitGraph);
    }
    announce(`已选择${node.node_type === 'reaction' ? '反应' : '分子'} ${node.label || graphNodeId}`);
    if (focus && !isDrawerLayout()) [...document.querySelectorAll('[data-graph-node-id]')].find(row =>
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
    const refs = value.source_refs || [];
    const conflicts = value.conflicts || [];
    const support = value.support_records || [];
    host.innerHTML = `<article><header><p class="detail-kind">${COPY.consensus}</p><h3 class="detail-title">证据与来源链</h3></header>
      <section class="detail-section"><h3>${COPY.support}</h3><div class="trace-list">${support.length ? support.map(row => `<div class="trace-row"><strong>${esc(row.source_group || row.source_ref || '来源')}</strong><span>${esc(row.claim || row.condition_summary || '')}</span></div>`).join('') : '<div class="empty">当前节点没有独立支持组明细。</div>'}</div></section>
      <section class="detail-section"><h3>${COPY.conflicts}</h3><div class="trace-list">${conflicts.length ? conflicts.map(row => `<div class="trace-row"><strong>${esc(row.field || '字段')}</strong><span>${esc((row.values || []).join(' / ') || row.reason || '')}</span></div>`).join('') : '<div class="empty">没有记录到条件冲突。</div>'}</div></section>
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
    setPanelExposure(element('navResizeHandle'), navVisible && !isDrawerLayout());
    setPanelExposure(element('inspectorResizeHandle'), inspectorVisible && !isDrawerLayout());
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
      if (target.dataset.branchFilter) { state.branchFilter = target.dataset.branchFilter; rerenderForControls(); return; }
      if (target.dataset.proofFilter) {
        const tier = target.dataset.proofFilter;
        state.proofFilters.has(tier) ? state.proofFilters.delete(tier) : state.proofFilters.add(tier);
        rerenderForControls({ restoreControl: { kind: 'proof', value: tier } }); return;
      }
      if (target.dataset.kindFilter) {
        const kind = target.dataset.kindFilter;
        state.kindFilters.has(kind) ? state.kindFilters.delete(kind) : state.kindFilters.add(kind);
        rerenderForControls({ restoreControl: { kind: 'kind', value: kind } }); return;
      }
      if (target.dataset.edgeFilter) { state.edgeFilter = target.dataset.edgeFilter; rerenderForControls({ restoreControl: { kind: 'edge', value: target.dataset.edgeFilter } }); return; }
      if (target.dataset.filterReset !== undefined) {
        state.branchFilter = 'all'; state.proofFilters = new Set(allProofTiers);
        state.kindFilters = new Set(allKinds); state.edgeFilter = 'all'; state.query = '';
        element('branchSearch').value = ''; rerenderForControls(); return;
      }
      if (target.dataset.graphAction === 'fit') { fitGraph(); return; }
      if (target.dataset.graphAction === 'zoom-in') { zoomGraph(1.2); return; }
      if (target.dataset.graphAction === 'zoom-out') { zoomGraph(1 / 1.2); return; }
      if (target.dataset.graphAction === 'reset') { resetGraph(); return; }
      if (target.dataset.detailTab) { state.detailTab = target.dataset.detailTab; renderDetail(); persistState(); return; }
      if (target.id === 'themeToggle') { state.theme = state.theme === 'dark' ? 'light' : 'dark'; applyPersistentChromeState(); persistState(); return; }
      if (target.id === 'navToggle') { state.navOpen = !state.navOpen; if (isDrawerLayout() && state.navOpen) state.inspectorOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.navOpen ? 'nav' : 'graph'); persistState(); requestAnimationFrame(fitGraph); return; }
      if (target.id === 'inspectorToggle') { state.inspectorOpen = !state.inspectorOpen; if (isDrawerLayout() && state.inspectorOpen) state.navOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.inspectorOpen ? 'inspector' : 'graph'); persistState(); requestAnimationFrame(fitGraph); return; }
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
    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('[data-graph-node-id]')) return;
      panSession = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-panning');
    });
    viewport.addEventListener('pointermove', event => {
      if (!panSession || panSession.pointerId !== event.pointerId) return;
      state.panX = panSession.panX + event.clientX - panSession.x;
      state.panY = panSession.panY + event.clientY - panSession.y;
      requestAnimationFrame(applyViewportTransform);
    });
    const endPan = event => {
      if (!panSession || panSession.pointerId !== event.pointerId) return;
      panSession = null; viewport.classList.remove('is-panning');
    };
    viewport.addEventListener('pointerup', endPan);
    viewport.addEventListener('pointercancel', endPan);
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
      handle.addEventListener('pointerup', () => { resizeSession = null; persistState(); requestAnimationFrame(fitGraph); });
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
    if (!isDrawerLayout()) return;
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
  function isDrawerLayout() { return matchMedia('(max-width: 1439px)').matches; }
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
      counts: forest.counts || {}
    }, '*');
  }
  init();
})();
