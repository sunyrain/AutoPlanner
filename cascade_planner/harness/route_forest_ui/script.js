(() => {
  'use strict';

  const forestDataText = document.getElementById('forest-data')?.textContent || '{}';
  const forest = JSON.parse(forestDataText);
  const STORAGE_KEY = `autoplanner.route-forest-ui.v4:${forest.case_id || forest.target?.name || 'route'}`;
  const LEGACY_STORAGE_KEY = 'autoplanner.route-forest-ui.v2';
  const PAN_DRAG_THRESHOLD_PX = 5;
  const MAX_PORTFOLIO_ROUTES = 5;
  const CULLING_OBJECT_THRESHOLD = 120;
  const CULLING_WORLD_MARGIN_PX = 180;
  const BRANCH_LANE_SCHEMA_VERSION = 'route_forest_branch_lanes.v2';
  const BRANCH_STAGE_EVIDENCE_SCHEMA_VERSION = 'route_forest_branch_stage_evidence.v3';
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
    'L2_mapping_consistent', 'L1_source_reported', 'L1_graph_stock_closed', 'L1_graph_and_stock_closed', 'L1_structural_materialized',
    'L0_materialized', 'L0_advisory', 'L0_rejected'
  ];
  const PROOF_LABEL = {
    L4_procurement_ready: 'L4 可采购',
    L3_precedent_supported: 'L3 精确先例',
    L2_reaction_validated: 'L2 反应重验',
    L2_mapping_consistent: 'L2 映射一致',
    L1_source_reported: 'L1 文献报道',
    L1_graph_stock_closed: 'L1 图与库存闭合',
    L1_graph_and_stock_closed: 'L1 图与库存闭合',
    L1_structural_materialized: 'L1 结构物化',
    L0_materialized: 'L0 结构已具象',
    L0_advisory: 'L0 探索建议',
    L0_rejected: 'L0 已拒绝'
  };
  const PROOF_AXIS_LABEL = Object.freeze({
    identity: '结构身份', reaction: '反应验证', conditions: '实验条件',
    sources: '独立来源', stock: '叶节点边界', process: '工艺可执行性'
  });
  const PROOF_VALUE_LABEL = Object.freeze({
    proposed: '仅建议', materialized: '已物化', source_exact: '精确来源结构',
    all_materialized: '全部已物化', all_source_exact: '全部精确来源', incomplete: '未闭合',
    untested: '未测试', mapped: '已映射', host_validated: '主机验证通过',
    source_reaction_exact: '来源反应精确', all_validated: '全部验证通过',
    missing: '缺失', model_predicted: '模型预测', source_recorded_unverified: '来源候选待核',
    mixed_supported: '混合支持', none: '无', single_group: '单一来源组',
    independent_2_plus: '两个以上独立来源组', conflicted: '存在冲突',
    not_applicable_to_edge: '边不适用', unknown: '未知', benchmark_hit: '搜索边界命中',
    offer_verified: '供应 offer 已验证', in_house: '厂内库存', blocked: '未就绪',
    procedure_bound_candidate: '已绑定过程候选', executable_candidate: '可执行工艺候选'
  });
  const CONDITION_LABEL = Object.freeze({
    reagents: '试剂', reagent: '试剂', agents: '辅助试剂', agent: '辅助试剂',
    catalyst: '催化剂', catalysts: '催化剂', solvent: '溶剂', solvents: '溶剂',
    temperature: '温度', 'temperature c': '温度', 'temperature program': '温度程序',
    time: '时间', 'time program': '时间程序', scale: '投料规模', equivalents: '当量',
    'addition order': '加料顺序', addition_order: '加料顺序', workup: '后处理',
    purification: '纯化', yield: '收率', 'yield percent': '收率', yield_percent: '收率'
  });
  const CORE_CONDITION_KEYS = new Set([
    'reagents', 'reagent', 'agents', 'agent', 'catalyst', 'catalysts', 'solvent', 'solvents',
    'temperature', 'temperature c', 'temperature program', 'time', 'time program', 'scale',
    'equivalents', 'yield', 'yield percent', 'yield_percent'
  ]);
  const TIER_CLASS = {
    L4_procurement_ready: 'tier-l4',
    L3_precedent_supported: 'tier-l3',
    L2_reaction_validated: 'tier-l2-validated',
    L2_mapping_consistent: 'tier-l2-mapping',
    L1_source_reported: 'tier-l1-reported',
    L1_graph_stock_closed: 'tier-l1',
    L1_graph_and_stock_closed: 'tier-l1',
    L1_structural_materialized: 'tier-l0-materialized',
    L0_materialized: 'tier-l0-materialized',
    L0_advisory: 'tier-l0-advisory',
    L0_rejected: 'tier-l0-rejected'
  };
  const TIER_COLOR = {
    L4_procurement_ready: '#15803d', L3_precedent_supported: '#0f766e',
    L2_reaction_validated: '#2563eb', L2_mapping_consistent: '#64748b',
    L1_source_reported: '#4f46e5',
    L1_graph_stock_closed: '#a16207', L1_graph_and_stock_closed: '#a16207',
    L1_structural_materialized: '#7c3aed',
    L0_materialized: '#7c3aed', L0_advisory: '#ea580c', L0_rejected: '#be123c'
  };
  const PRODUCER_CLASS = {
    chemenzy: 'producer-chemenzy', codex: 'producer-codex',
    codex_global_director: 'producer-codex', literature: 'producer-literature',
    literature_replay: 'producer-literature', self_evo_patent_template: 'producer-self-evo',
    template: 'producer-template', manual: 'producer-manual',
    host_product_grounded_repair: 'producer-host', planner: 'producer-codex'
  };
  const PRODUCER_COLOR = {
    'producer-codex': '#0369a1', 'producer-chemenzy': '#7c3aed',
    'producer-literature': '#0f766e', 'producer-self-evo': '#c2410c',
    'producer-template': '#a16207', 'producer-manual': '#475569',
    'producer-host': '#be123c', 'producer-unknown': '#64748b'
  };
  const graph = forest.dependency_graph || {};
  const layout = forest.dependency_layout || {};
  const lanesProjection = forest.branch_lanes || {};
  const frontierLedger = forest.frontier_ledger || forest.semantic_summary?.frontier_ledger || {};
  const retrosynthesisControl = forest.retrosynthesis_control || {};
  const campaignSummary = forest.campaign_summary || {};
  const routeClosure = forest.route_closure || {};
  const graphNodes = new Map((graph.nodes || []).map(row => [row.graph_node_id, row]));
  const moleculeNodes = new Map((forest.nodes || []).map(row => [row.node_id, row]));
  const steps = new Map((forest.steps || []).map(row => [row.step_id, row]));
  const branches = new Map((forest.branches || []).map(row => [row.branch_id, row]));
  const programOverlays = new Map((forest.program_overlays || []).map(row => [row.program_id, row]));
  const programOverlaysByBranch = new Map();
  for (const overlay of programOverlays.values()) {
    if (!programOverlaysByBranch.has(overlay.branch_id)) programOverlaysByBranch.set(overlay.branch_id, []);
    programOverlaysByBranch.get(overlay.branch_id).push(overlay);
  }
  const mechanismHypotheses = new Map((forest.mechanism_hypotheses || [])
    .map(row => [row.hypothesis_id, row]));
  const mechanismHypothesesByBranch = new Map();
  for (const hypothesis of mechanismHypotheses.values()) {
    if (!mechanismHypothesesByBranch.has(hypothesis.branch_id)) mechanismHypothesesByBranch.set(hypothesis.branch_id, []);
    mechanismHypothesesByBranch.get(hypothesis.branch_id).push(hypothesis);
  }
  const graphNodeIdByMolecule = new Map((graph.nodes || [])
    .filter(row => row.node_type === 'molecule' && row.molecule_node_id)
    .map(row => [row.molecule_node_id, row.graph_node_id]));
  const reactionGraphNodeIdByStep = new Map((graph.nodes || [])
    .filter(row => row.node_type === 'reaction' && row.reaction_step_id)
    .map(row => [row.reaction_step_id, row.graph_node_id]));
  const laneByBranch = new Map((lanesProjection.lanes || []).map(row => [row.branch_id, row]));
  const edgeById = new Map((graph.edges || []).map(row => [row.edge_id, row]));
  const layoutByNode = new Map((layout.nodes || []).map(row => [row.graph_node_id, row]));
  const persisted = loadState();
  const legacyChrome = loadState(LEGACY_STORAGE_KEY);
  const sourceRevisionKey = String(
    forest.source_revision_context?.revision_id
      || forest.source_revision_context?.revision
      || forest.campaign_projection?.revision
      || ''
  );
  const embeddedRoute = new URLSearchParams(location.search).get('embed') === '1';
  const allProofTiers = unique([
    ...(lanesProjection.lanes || []).map(row => row.proof_tier),
    ...(forest.steps || []).map(row => row?.trust_vector?.proof_tier || row?.proof_tier)
  ].filter(Boolean));
  const allKinds = unique((lanesProjection.lanes || []).map(row => row.kind).filter(Boolean));
  const defaultBranchId = chooseDefaultBranchId();
  const persistedSelectionIsCurrent = !sourceRevisionKey
    || persisted.sourceRevisionKey === sourceRevisionKey;
  const initialBranchId = persisted.selectedBranchId
    && persistedSelectionIsCurrent && branches.has(persisted.selectedBranchId)
    ? persisted.selectedBranchId : defaultBranchId;

  const state = {
    mode: oneOf(persisted.mode, ['clusters', 'shared', 'current'], 'current'),
    selectedBranchId: initialBranchId,
    selectedGraphNodeId: '',
    selectedInstanceId: '',
    selectedStepId: '',
    selectedProgramId: '',
    selectedMechanismId: '',
    detailTab: oneOf(persisted.detailTab, ['step', 'evidence', 'alternatives'], 'step'),
    query: '',
    stageFilter: oneOf(persisted.stageFilter, [
      'all', 'suggestion', 'expanded', 'reaction', 'literature', 'conditions',
      'stock', 'procurement', 'process'
    ], 'all'),
    branchFilter: oneOf(persisted.branchFilter, ['all', 'verified', 'evidence', 'advisory', 'diagnostic'], 'all'),
    proofFilters: new Set(Array.isArray(persisted.proofFilters) ? persisted.proofFilters.filter(tier => allProofTiers.includes(tier)) : allProofTiers),
    kindFilters: new Set(Array.isArray(persisted.kindFilters) ? persisted.kindFilters.filter(kind => allKinds.includes(kind)) : allKinds),
    edgeFilter: oneOf(persisted.edgeFilter, ['all', 'selected'], 'all'),
    orientation: oneOf(persisted.orientation, ['horizontal', 'vertical'], 'horizontal'),
    routeDirection: oneOf(
      persisted.routeDirection,
      ['synthesis', 'retrosynthesis'],
      oneOf(forest.display_policy?.default_route_direction, ['synthesis', 'retrosynthesis'], 'synthesis')
    ),
    showAuxiliary: Object.hasOwn(persisted, 'showAuxiliary')
      ? persisted.showAuxiliary === true
      : forest.display_policy?.auxiliary_inputs_collapsed !== true,
    density: oneOf(persisted.density, ['comfortable', 'compact', 'overview'], 'comfortable'),
    edgeStyle: oneOf(persisted.edgeStyle, ['trust', 'simple', 'contrast'], 'trust'),
    labelMode: oneOf(persisted.labelMode, ['semantic', 'full', 'minimal'], 'semantic'),
    layoutPreset: oneOf(persisted.layoutPreset, ['explore', 'focus', 'review'], 'explore'),
    theme: oneOf(persisted.theme || legacyChrome.theme, ['light', 'dark'], preferredTheme()),
    navOpen: Object.hasOwn(persisted, 'navOpen') ? persisted.navOpen !== false : !matchMedia('(max-width: 1023px)').matches,
    inspectorOpen: Object.hasOwn(persisted, 'inspectorOpen') ? persisted.inspectorOpen !== false : false,
    ledgerOpen: Object.hasOwn(persisted, 'ledgerOpen') ? persisted.ledgerOpen === true : false,
    navWidth: clamp(Number(persisted.navWidth || legacyChrome.navWidth) || 280, 240, 460),
    inspectorWidth: clamp(Number(persisted.inspectorWidth || legacyChrome.inspectorWidth) || 380, 320, 560),
    zoom: 1,
    panX: 0,
    panY: 0,
    cameraMode: 'fit',
    showAllOverview: false,
    expandedGroups: new Set(),
    expandedProgramIds: new Set((persisted.expandedProgramIds || []).map(String)),
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
  let lastCullCamera = null;
  let viewportResizeObserver = null;
  const depictionCache = new Map();
  const graphModelCache = new Map();
  const renderPerformance = {
    cameraFrames: 0,
    droppedFrames: 0,
    maximumFrameDelayMs: 0,
    maximumCameraFrameMs: 0,
    totalCameraFrameMs: 0,
    lastGraphUpdateMs: 0,
    renderedObjects: 0,
    culledObjects: 0
  };

  window.__AUTOPLANNER_ROUTE_PERF__ = Object.freeze({
    snapshot: () => ({
      ...renderPerformance,
      graphRevision: forest.source_revision_context?.revision || forest.campaign_projection?.revision || null,
      memoryBytes: Number(performance.memory?.usedJSHeapSize || 0),
      meanCameraFrameMs: renderPerformance.cameraFrames
        ? renderPerformance.totalCameraFrameMs / renderPerformance.cameraFrames : 0,
      zoom: state.zoom,
      panX: state.panX,
      panY: state.panY
    })
  });

  if (embeddedRoute) {
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
  function portfolioDisplayLimit() {
    return clamp(Number(forest.display_policy?.default_overview_top_k || MAX_PORTFOLIO_ROUTES), 2, MAX_PORTFOLIO_ROUTES);
  }
  function stableTextCompare(left, right) {
    const a = String(left), b = String(right);
    return a < b ? -1 : a > b ? 1 : 0;
  }
  function branchDisplayScore(lane) {
    const branch = branches.get(lane?.branch_id) || {};
    if (branch.solved === true && branch.executable === true && branch.advisory_only === false) return 10000;
    const kindScore = {
      stitched_verified_route: 900, direct_verified_route: 860,
      proof_eligible_portfolio_route: 760, reported_candidate_route: 730,
      exploratory_canonical_route: 180,
      exact_literature: 700,
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
    const structureScore = Math.min(12, (lane?.graph_node_ids || []).filter(graphNodeId => {
      const graphNode = graphNodes.get(graphNodeId) || {};
      return Boolean(moleculeNodes.get(graphNode.molecule_node_id)?.structure_svg);
    }).length) * 28;
    const failedPenalty = /failed|diagnostic|rejected/i.test(`${lane?.title || ''} ${branch.summary || ''}`) ? 80 : 0;
    return kindScore + proofScore + stockScore + primaryScore + confidenceScore
      + stepScore + structureScore - failedPenalty;
  }
  function isPriorityBranch(branchId) {
    const lane = laneByBranch.get(branchId) || {};
    const declaredPrimaryId = String(forest.primary_branch_id || '');
    return lane.is_primary === true
      && (!declaredPrimaryId || String(branchId || '') === declaredPrimaryId);
  }
  function branchDisplayRank(lane) {
    const branch = branches.get(lane?.branch_id) || {};
    const routeTrust = branch.trust_vector || {};
    return [
      branch.solved === true && branch.executable === true && branch.advisory_only === false ? 1 : 0,
      Math.max(-1, Number(lane?.proof_rank ?? -1)),
      Math.max(0, Number(routeTrust.min_trusted_source_group_count_across_steps || 0)),
      routeTrust.all_edges_corroborated === true ? 1 : 0,
      Math.max(0, Number(routeTrust.corroborated_edge_count || 0)),
      branchDisplayScore(lane)
    ];
  }
  function compareBranchDisplay(left, right) {
    const leftRank = branchDisplayRank(left);
    const rightRank = branchDisplayRank(right);
    for (let index = 0; index < leftRank.length; index += 1) {
      if (leftRank[index] !== rightRank[index]) return rightRank[index] - leftRank[index];
    }
    return stableTextCompare(left?.branch_id, right?.branch_id);
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
      .sort(compareBranchDisplay)[0];
    return featured?.branch_id || primaryId || (lanesProjection.lanes || [])[0]?.branch_id || '';
  }
  function safeStructureSvg(value) {
    const svg = String(value || '').trim();
    if (depictionCache.has(svg)) return depictionCache.get(svg);
    const safe = svg.startsWith('<svg') && !/<script\b|<foreignObject\b|\son\w+\s*=|javascript:/i.test(svg)
      ? svg : '';
    if (depictionCache.size > 512) depictionCache.clear();
    depictionCache.set(svg, safe);
    return safe;
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
      sourceRevisionKey,
      stageFilter: state.stageFilter, branchFilter: state.branchFilter,
      proofFilters: [...state.proofFilters],
      kindFilters: [...state.kindFilters], edgeFilter: state.edgeFilter,
      orientation: state.orientation, density: state.density, edgeStyle: state.edgeStyle,
      routeDirection: state.routeDirection, showAuxiliary: state.showAuxiliary,
      labelMode: state.labelMode, layoutPreset: state.layoutPreset, theme: state.theme,
      navOpen: state.navOpen, inspectorOpen: state.inspectorOpen, ledgerOpen: state.ledgerOpen,
      navWidth: state.navWidth, inspectorWidth: state.inspectorWidth,
      expandedProgramIds: [...state.expandedProgramIds]
    }));
  }
  function preferredTheme() {
    return matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function tierOfStep(step) { return step?.trust_vector?.proof_tier || 'L0_advisory'; }
  function tierClass(tier) { return TIER_CLASS[tier] || 'tier-l0-advisory'; }
  function tierLabel(tier) { return PROOF_LABEL[tier] || tier || '未分级'; }
  function routeProofMixLabel(lane) {
    const reported = Math.max(0, Number(lane?.reported_step_count || 0));
    const planner = Math.max(0, Number(lane?.planner_hypothesis_step_count || 0));
    if (reported || planner) {
      return [
        planner ? `${planner} 步 L0 规划` : '',
        reported ? `${reported} 步 L1 文献` : ''
      ].filter(Boolean).join(' · ');
    }
    const counts = lane?.proof_level_counts || {};
    const parts = Object.entries(counts)
      .filter(([, count]) => Number(count) > 0)
      .sort((left, right) => Number(left[0]) - Number(right[0]))
      .map(([level, count]) => `${Number(count)} 步 L${level}`);
    return parts.join(' · ') || tierLabel(lane?.proof_tier);
  }
  function isRetrosynthesis() { return state.routeDirection === 'retrosynthesis'; }
  function routeStepDisplayLabel(step) {
    if (!step) return '反应步骤';
    return isRetrosynthesis()
      ? (step.retrosynthesis_display_label || step.retrosynthesis_label || step.display_label || step.step_id)
      : (step.display_label || step.stage_label || step.step_id);
  }
  function inlineConditionText(step) {
    const rows = normalizedConditionRows(step);
    const priority = ['reagents', 'agents', 'catalyst', 'solvent', 'temperature', 'temperature c', 'time', 'yield', 'yield percent', 'workup', 'addition order'];
    const ranked = rows
      .sort((left, right) => {
        const a = priority.indexOf(left.key);
        const b = priority.indexOf(right.key);
        return (a < 0 ? priority.length : a) - (b < 0 ? priority.length : b);
      });
    if (ranked.length) {
      const summary = ranked.slice(0, 3).map(row => `${row.displayLabel} ${row.value}`).join(' · ');
      const prefix = step.condition_status === 'model_predicted' ? '预测' : '来源';
      return `${prefix} · ${middleEllipsis(summary, 34)}`;
    }
    if (step?.condition_status === 'source_exact') return '来源条件已绑定 · 字段待展开';
    if (step?.condition_status === 'source_recorded_unverified') return '来源条件候选 · 待核验';
    if (step?.condition_status === 'model_predicted') return '预测条件 · 非文献事实';
    return '条件待取证';
  }
  function conditionValueText(value) {
    if (Array.isArray(value)) return value.map(conditionValueText).filter(Boolean).join('、');
    if (value && typeof value === 'object') {
      return Object.entries(value).map(([key, item]) => `${key}: ${conditionValueText(item)}`).join('；');
    }
    return String(value ?? '').trim();
  }
  function normalizedConditionRows(step) {
    const candidates = [];
    for (const row of Array.isArray(step?.conditions) ? step.conditions : []) {
      if (!row || typeof row !== 'object') continue;
      candidates.push({ label: row.label || row.key || '条件', value: row.value });
    }
    for (const observation of Array.isArray(step?.source_observation_records) ? step.source_observation_records : []) {
      for (const [label, value] of Object.entries(observation?.conditions || {})) {
        candidates.push({ label, value });
      }
    }
    const seen = new Set();
    return candidates.flatMap(row => {
      const key = String(row.label || '条件').trim().toLowerCase().replaceAll('_', ' ');
      const value = conditionValueText(row.value);
      const signature = `${key}\u0000${value}`;
      if (!value || seen.has(signature)) return [];
      seen.add(signature);
      return [{ key, label: String(row.label || '条件'), displayLabel: CONDITION_LABEL[key] || String(row.label || '条件'), value }];
    });
  }
  function conditionLinesHtml(rows) {
    return rows.map(row => `<div class="condition-line"><span class="condition-label">${esc(row.displayLabel)}</span><span class="condition-value">${esc(row.value)}</span></div>`).join('');
  }
  function conditionGroupHtml(label, rows, { open = false } = {}) {
    if (!rows.length) return '';
    return `<details class="condition-group" ${open ? 'open' : ''}><summary><span>${esc(label)}</span><strong>${rows.length} 项</strong></summary><div class="condition-group-body">${conditionLinesHtml(rows)}</div></details>`;
  }
  function conditionResolutionHtml(step) {
    const resolution = step?.condition_resolution || {};
    const label = resolution.label || '条件待取证';
    const summary = resolution.summary || step?.condition_summary || '尚无可重放的来源条件。';
    const nextAction = resolution.next_action || '系统继续主动检索并解析原始来源。';
    return `<div class="condition-resolution" data-condition-stage="${esc(resolution.stage || 'unknown')}">
      <strong>${esc(label)}</strong><span>${esc(summary)}</span><small>下一步：${esc(nextAction)}</small>
    </div>`;
  }
  function modelConditionPredictionsHtml(step) {
    const predictions = Array.isArray(step?.condition_predictions) ? step.condition_predictions : [];
    if (!predictions.length) return '';
    const ignored = new Set([
      'authority scope', 'not reaction proof', 'not source evidence',
      'schema version', 'prediction producer', 'condition model', 'rank',
      'condition prediction issues', 'null1', 'null2'
    ]);
    const aliases = {
      reagent: 'reagents', reagents: 'reagents', 'reagent smiles': 'reagents',
      catalyst: 'catalyst', catalysts: 'catalyst',
      solvent: 'solvent', solvents: 'solvent', 'solvent smiles': 'solvent',
      temperature: 'temperature', 'temperature c': 'temperature', 'temp c': 'temperature',
      time: 'time', duration: 'time', score: 'prediction score'
    };
    const cards = predictions.map((prediction, index) => {
      if (!prediction || typeof prediction !== 'object') return '';
      const rows = Object.entries(prediction).flatMap(([label, rawValue]) => {
        const rawKey = String(label).trim().toLowerCase().replaceAll('_', ' ');
        if (ignored.has(rawKey) || rawValue == null || rawValue === '') return [];
        const key = aliases[rawKey] || rawKey;
        let value = conditionValueText(rawValue);
        if (key === 'temperature' && Number.isFinite(Number(rawValue))) value = `${Number(rawValue).toFixed(1)} °C`;
        if (!value) return [];
        return [{ key, label, displayLabel: CONDITION_LABEL[key] || (key === 'prediction score' ? '预测分数' : label), value }];
      });
      if (!rows.length) return '';
      const score = prediction.Score ?? prediction.score;
      return `<details class="model-condition-prediction" ${index === 0 ? 'open' : ''}><summary><span>候选 ${index + 1}</span><strong>${score == null || score === '' ? `${rows.length} 项` : `分数 ${esc(score)}`}</strong></summary><div class="condition-group-body">${conditionLinesHtml(rows)}</div></details>`;
    }).join('');
    if (!cards) return '';
    return `<section class="detail-section model-condition-predictions"><h3>模型条件候选 <span class="section-count">${predictions.length} 组</span></h3><div class="notice"><strong>非文献事实</strong><span>各候选相互独立，未合并为单一实验配方；需经实验或来源过程验证。</span></div><div class="condition-list">${cards}</div></section>`;
  }
  function producerClass(step) {
    return PRODUCER_CLASS[(step?.producer_kinds || [])[0]] || 'producer-unknown';
  }
  function producerColor(step) {
    return PRODUCER_COLOR[producerClass(step)] || PRODUCER_COLOR['producer-unknown'];
  }
  function visibleEdges(values) {
    return state.showAuxiliary ? values : values.filter(row => row.visual_role !== 'auxiliary');
  }
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
    const graphRouteClosed = deliveryBytesVerified
      && routeClosure.any_declared_route_graph_closed === true;
    const allGraphClosed = ledgerAuthoritative && closure.all_explored_benchmark_closed === true;
    const anyProcurementClosed = ledgerAuthoritative && closure.any_procurement_route_closed === true;
    const selectedRouteProof = forest.selected_route_parent_proof || {};
    const selectedProofRequired = selectedRouteProof.available === true;
    const selectedProofAuthoritative = deliveryBytesVerified
      && selectedRouteProof.accepted === true;
    const controlAuthoritative = deliveryBytesVerified
      && retrosynthesisControl.authoritative === true;
    const hardAcceptance = controlAuthoritative
      && retrosynthesisControl.acceptance?.accepted === true;
    const parentL3Solved = selectedProofRequired
      ? selectedProofAuthoritative && selectedRouteProof.benchmark_solved === true
      : deliveryBytesVerified && verified && closure.l3_parent_solved === true;
    const procurementL4Ready = selectedProofRequired
      ? parentL3Solved && selectedRouteProof.procurement_ready === true
      : parentL3Solved && anyProcurementClosed && closure.l4_procurement_ready === true;
    element('pageTitle').textContent = `${targetName()} · 路线工作台`;
    const verdict = element('verdictBadge');
    verdict.textContent = hardAcceptance ? 'V4 路线硬验收通过'
      : procurementL4Ready ? 'L4 双路线采购就绪'
      : parentL3Solved ? `${Number(selectedRouteProof.distinct_complete_route_count || 1)} 条 L3 替代路线闭合`
        : allGraphClosed ? 'Benchmark 全探索闭合'
          : anyRouteClosed ? '存在 Benchmark 闭合路线' : '父路线未闭合';
    if (!hardAcceptance && !procurementL4Ready && !parentL3Solved
        && !anyRouteClosed && graphRouteClosed) {
      verdict.textContent = '存在结构闭合路线 · 证据/库存开放';
    }
    const verdictState = hardAcceptance || procurementL4Ready || parentL3Solved
      ? 'verified' : anyRouteClosed || allGraphClosed || graphRouteClosed ? 'partial' : 'unresolved';
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
      ['L1 文献边', ledgerValue('l1_source_reported_edges')],
      ['已展开 work', ledgerValue('expanded_work_molecules')],
      ['L2 反应边', ledgerValue('l2_reaction_edges')],
      ['L3 先例边', ledgerValue('l3_precedent_edges')],
      ['库存边界叶', ledgerValue('stock_closed_leaves')],
      ['L3 完整路线', selectedProofAuthoritative
        ? Number(selectedRouteProof.benchmark_route_count || 0) : '—'],
      ['L4 采购路线', selectedProofAuthoritative
        ? Number(selectedRouteProof.procurement_route_count || 0) : '—']
    ];
    if (retrosynthesisControl.available === true) {
      overviewRows.unshift([
        '硬验收',
        controlAuthoritative
          ? (hardAcceptance ? '通过' : '未通过')
          : '—'
      ]);
    }
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
    const selectedRouteProof = forest.selected_route_parent_proof || {};
    const control = retrosynthesisControl;
    const controlAuthoritative = deliveryBytesVerified
      && control.authoritative === true;
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
      ['L0 规划建议', countValue('l0_break_suggestion_edges'), '仅由规划器提出、尚未获得来源或主机反应验证的边'],
      ['L1 文献报道', countValue('l1_source_reported_edges'), '论文已报道且结构已具象，但尚未升级为主机反应验证或精确人工绑定'],
      ['已展开 work', ratioValue('expanded_work_molecules', 'reachable_molecules'), '已成功完成 proposal expansion 的分子 frontier'],
      ['L2 反应验证', countValue('l2_reaction_edges'), '当前 host verifier 接受的 L2 反应边'],
      ['L3 精确先例', countValue('l3_precedent_edges'), '绑定精确文献先例的 L3 反应边'],
      ['搜索库存叶', ratioValue('stock_closed_leaves', 'reachable_leaves'), stockDetail]
    ];
    element('ledgerProgressMetrics').innerHTML = progress.map(([label, value, detail]) => `
      <span class="ledger-progress-chip" data-authoritative="${String(authoritative)}" title="${esc(detail)}">
        <span>${esc(label)}</span><strong>${esc(value)}</strong>
      </span>`).join('');

    const acceptance = controlAuthoritative ? (control.acceptance || {}) : {};
    const acceptanceSpec = acceptance.acceptance_spec || {};
    const costTotals = controlAuthoritative ? (control.cost_totals || {}) : {};
    const costBudget = controlAuthoritative ? (control.cost_budget || {}) : {};
    const nextDeficit = controlAuthoritative ? (control.next_deficit || {}) : {};
    const campaignAvailable = deliveryBytesVerified
      && campaignSummary.available === true;
    const gateLabels = {
      B0_blind_input: 'B0 Blind',
      B1_global_multi_route: 'B1 Global',
      B2_host_validated_routes: 'B2 Host',
      B3_exact_multi_source: 'B3 Evidence',
      B4_stock_boundary: 'B4 Stock',
      B5_configured_portfolio_acceptance: 'B5 Policy'
    };
    const campaignGates = campaignSummary.gates || {};
    const campaignCost = campaignSummary.model_cost || {};
    const campaignResource = campaignSummary.resource_envelope || {};
    const legacyControlRows = [
      [
        '硬验收',
        controlAuthoritative
          ? `${Number(acceptance.selected_route_count || 0)}/${Number(acceptanceSpec.minimum_complete_routes || 0)} 路线 · ${acceptance.accepted === true ? '通过' : '未通过'}`
          : '—',
        acceptance.accepted === true ? 'closed' : 'open'
      ],
      [
        '下一缺口',
        controlAuthoritative
          ? (nextDeficit.kind ? String(nextDeficit.kind).replaceAll('_', ' ') : '无待办')
          : '—',
        controlAuthoritative && !nextDeficit.kind ? 'closed' : 'open'
      ],
      [
        '模型调用',
        controlAuthoritative
          ? `${Number(costTotals.model_invocations || 0)}/${Number(costBudget.max_model_invocations || 0)}`
          : '—',
        'neutral'
      ],
      [
        '模型 Token',
        controlAuthoritative
          ? `${Number(costTotals.input_tokens || 0) + Number(costTotals.output_tokens || 0)}/${Number(costBudget.max_total_input_tokens || 0) + Number(costBudget.max_total_output_tokens || 0)}`
          : '—',
        'neutral'
      ]
    ];
    const controlRows = campaignAvailable
      ? Object.entries(gateLabels).map(([key, label]) => [
          label,
          campaignGates[key] === true ? '通过' : '未通过',
          campaignGates[key] === true ? 'closed' : 'open'
        ]).concat([
          [
            '资源预算',
            campaignResource.within_budget === true ? '合规' : '超限',
            campaignResource.within_budget === true ? 'closed' : 'open'
          ],
          [
            '模型调用',
            Number(campaignCost.model_invocations || 0),
            'neutral'
          ]
        ])
      : legacyControlRows;
    element('runControlMetrics').innerHTML = controlRows.map(([label, value, state]) => `
      <span class="run-control-chip" data-state="${esc(campaignAvailable || controlAuthoritative ? state : 'unknown')}">
        <span>${esc(label)}</span><strong>${esc(value)}</strong>
      </span>`).join('');

    const ledgerState = value => authoritative ? (value === true ? 'closed' : 'open') : 'unknown';
    const ledgerValue = (value, positive, negative) => authoritative
      ? (value === true ? positive : negative) : '账本缺失';
    const cards = [
      {
        label: 'DECLARED ROUTE GRAPH',
        value: !deliveryBytesVerified ? '交付字节未验证'
          : routeClosure.any_declared_route_graph_closed === true
            ? `${Number(routeClosure.graph_closed_program_count || 0)}/${Number(routeClosure.declared_program_count || 0)} 条结构闭合 · 最长 ${Number(routeClosure.longest_graph_closed_step_count || 0)} 步`
            : `0/${Number(routeClosure.declared_program_count || 0)} 条结构闭合`,
        state: !deliveryBytesVerified ? 'unknown'
          : routeClosure.any_declared_route_graph_closed === true ? 'closed' : 'open',
        detail: '仅表示目标到声明叶节点的每一步均已进入规范图；不等于反应验证、文献精确绑定或库存/采购闭合'
      },
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
        label: 'L3 SELECTED ROUTES',
        value: !deliveryBytesVerified ? '交付字节未验证'
          : parentL3Solved
            ? `${Number(selectedRouteProof.distinct_complete_route_count || 1)} 条不同 edge-set 已闭合`
            : `${Number(selectedRouteProof.distinct_complete_route_count || 0)}/${Number(selectedRouteProof.minimum_complete_routes || 2)} 条`,
        state: !deliveryBytesVerified ? 'unknown' : parentL3Solved ? 'closed' : 'open',
        detail: '至少两条完整替代路线；每条反应边均达到 L3 精确先例，且所有叶节点逐一库存闭合'
      },
      {
        label: 'L4 PROCUREMENT',
        value: !deliveryBytesVerified ? '交付字节未验证'
          : procurementL4Ready
            ? `${Number(selectedRouteProof.procurement_route_count || 0)} 条采购就绪`
            : '未证明',
        state: !deliveryBytesVerified ? 'unknown' : procurementL4Ready ? 'closed' : 'open',
        detail: `${authoritative ? Number(counts.procurement_boundary_leaves || 0) : '—'} 个采购边界 · ${authoritative ? Number(counts.l4_procurement_edges || 0) : '—'} 条 L4 边；benchmark 命中绝不冒充商业采购`
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
      ['论文过程观察', literature.source_observation_records ?? 0],
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
        button.title = `${count} 条全路径完成展开；${partialExpandedCount} 条仅部分展开，不计入本阶段；数量不代表完整路线数`;
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
    const mixedProof = Object.values(lane.proof_level_counts || {})
      .filter(count => Number(count) > 0).length > 1;
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
    const proofMix = routeProofMixLabel(lane);
    const stateLabel = lane.route_state_label || lane.completion_label || (lane.solved && lane.executable && !lane.advisory_only
      ? '完整父路线'
      : lane.kind === 'proof_eligible_portfolio_route' ? '完整 portfolio'
        : stageMembershipIsAuthoritative(lane, 'stock')
          ? (stockEvidence.closure_scope === 'procurement' ? '采购闭合' : 'Benchmark 闭合')
          : stageMembershipIsAuthoritative(lane, 'reaction') ? '反应已验证'
            : stageMembershipIsAuthoritative(lane, 'expanded') ? '全路径已展开'
              : stageEvidence.suggestion?.member === true ? '断键建议'
                : partialProgress ? '探索中' : '阶段证据未绑定');
    return `<button class="branch-card ${tierClass(lane.proof_tier)}${mixedProof ? ' is-mixed-proof' : ''}${selected ? ' is-selected' : ''}" type="button"
      data-branch-id="${esc(lane.branch_id)}" aria-current="${selected ? 'true' : 'false'}" tabindex="-1">
      <span class="branch-card-title">${esc(lane.title || lane.branch_id)}</span>
      <span class="branch-card-meta">${esc((lane.step_ids || []).length)} 步 · ${esc(proofMix)} · ${esc(lane.condition_label || '条件状态未知')}</span>
      <span class="branch-card-badges">${isPriorityBranch(lane.branch_id) ? `<span class="branch-badge">${forest.primary_selection?.display_tiebreak_only ? '展示锚点' : '重点分支'}</span>` : ''}<span class="branch-badge">${esc(stateLabel)}</span>${partialBadge}<span class="branch-badge">${esc(synthesisLabel(branch.synthesis_class))}</span></span>
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
    const cacheKey = JSON.stringify({
      mode: state.mode,
      branch: state.selectedBranchId,
      stage: state.stageFilter,
      branchFilter: state.branchFilter,
      proof: [...state.proofFilters].sort(),
      kinds: [...state.kindFilters].sort(),
      edge: state.edgeFilter,
      orientation: effectiveOrientation(),
      routeDirection: state.routeDirection,
      showAuxiliary: state.showAuxiliary,
      mobile: matchMedia('(max-width: 639px)').matches,
      density: state.density,
      showAll: state.showAllOverview,
      expandedPrograms: [...state.expandedProgramIds].sort(),
      replacement: state.activeReplacement?.replacement_id || ''
    });
    const cached = graphModelCache.get(cacheKey);
    if (cached) return cached;
    const lanes = filteredLanes({ includeReplacementPreview: true });
    let model;
    if (state.mode === 'shared') model = buildSharedModel(overviewLanes(lanes));
    else {
      const selected = lanes.find(lane => lane.branch_id === state.selectedBranchId);
      const activeLanes = state.mode === 'current' ? (selected ? [selected] : []) : overviewLanes(lanes);
      model = buildLaneModel(activeLanes);
    }
    if (graphModelCache.size >= 12) graphModelCache.clear();
    graphModelCache.set(cacheKey, model);
    return model;
  }

  function overviewLanes(lanes) {
    if (state.showAllOverview) return lanes;
    const topK = portfolioDisplayLimit();
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
    let edges = visibleEdges((graph.edges || []).filter(edge => branchIds.has(edge.branch_id)));
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
    const layerEntries = [...buckets.entries()].sort((a, b) =>
      isRetrosynthesis() ? b[0] - a[0] : a[0] - b[0]);
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

  function programIsExpanded(programId) {
    return state.expandedProgramIds.has(String(programId || ''));
  }

  function collapsedProgramProjection(lane, rawEdges, rawRows) {
    const overlays = state.mode === 'current'
      ? (programOverlaysByBranch.get(lane.branch_id) || [])
      : [];
    if (!overlays.length) return { edges: rawEdges, rows: rawRows };

    const rawLayerByNode = new Map(rawRows.map(row => [
      row.graph_node_id,
      Number(row.layer || 0)
    ]));
    const hiddenGraphNodeIds = new Set();
    const intervals = [];
    for (const overlay of overlays) {
      const boundaryMoleculeIds = new Set([
        ...(overlay.input_molecule_node_ids || []),
        ...(overlay.output_molecule_node_ids || [])
      ]);
      for (const stepId of overlay.replaced_step_ids || []) {
        const reactionGraphId = reactionGraphNodeIdByStep.get(stepId);
        if (reactionGraphId) hiddenGraphNodeIds.add(reactionGraphId);
        const step = steps.get(stepId) || {};
        const participatingMoleculeIds = unique([
          ...(step.from_node_ids || []),
          ...(step.main_from_node_ids || []),
          ...(step.auxiliary_from_node_ids || []),
          ...(step.to_node_ids || [])
        ]);
        for (const moleculeId of participatingMoleculeIds) {
          if (boundaryMoleculeIds.has(moleculeId)) continue;
          const graphNodeId = graphNodeIdByMolecule.get(moleculeId);
          if (graphNodeId) hiddenGraphNodeIds.add(graphNodeId);
        }
      }
      const inputGraphId = graphNodeIdByMolecule.get(overlay.input_molecule_node_ids?.[0]);
      const outputGraphId = graphNodeIdByMolecule.get(overlay.output_molecule_node_ids?.[0]);
      const inputLayer = rawLayerByNode.get(inputGraphId);
      const outputLayer = rawLayerByNode.get(outputGraphId);
      if (Number.isFinite(inputLayer) && Number.isFinite(outputLayer)) {
        intervals.push({
          start: Math.min(inputLayer, outputLayer),
          end: Math.max(inputLayer, outputLayer)
        });
      }
    }
    intervals.sort((left, right) => left.start - right.start || left.end - right.end);

    const compactLayer = rawLayer => {
      let removedLayers = 0;
      for (const interval of intervals) {
        const reduction = Math.max(0, interval.end - interval.start - 2);
        if (rawLayer >= interval.end) {
          removedLayers += reduction;
          continue;
        }
        if (rawLayer > interval.start) return interval.start - removedLayers + 1;
        break;
      }
      return rawLayer - removedLayers;
    };
    return {
      edges: rawEdges.filter(edge => !hiddenGraphNodeIds.has(edge.source_graph_node_id)
        && !hiddenGraphNodeIds.has(edge.target_graph_node_id)),
      rows: rawRows.filter(row => !hiddenGraphNodeIds.has(row.graph_node_id)).map(row => ({
        ...row,
        layout_layer: compactLayer(Number(row.layer || 0))
      }))
    };
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
      const lanePrograms = state.mode === 'current'
        ? (programOverlaysByBranch.get(lane.branch_id) || []) : [];
      const expandedProgram = lanePrograms.find(overlay =>
        programIsExpanded(overlay.program_id));
      const programDrawerInset = expandedProgram
        ? (orientation === 'vertical' ? 470 : 208) : 0;
      const mechanismCount = state.mode === 'current'
        ? Math.min(3, (mechanismHypothesesByBranch.get(lane.branch_id) || []).length)
        : 0;
      const programInset = (mechanismCount ? 14 : 0) + mechanismCount * 46;
      const rawEdges = visibleEdges((lane.edge_ids || []).map(edgeId => edgeById.get(edgeId)).filter(Boolean));
      const visibleNodeIds = new Set(rawEdges.flatMap(edge => [edge.source_graph_node_id, edge.target_graph_node_id]));
      const rawRows = (lane.node_layout || []).filter(row => visibleNodeIds.has(row.graph_node_id));
      const projection = collapsedProgramProjection(lane, rawEdges, rawRows);
      const localEdges = projection.edges;
      const localRows = projection.rows.slice().sort((a, b) => Number(a.layout_layer ?? a.layer) - Number(b.layout_layer ?? b.layer)
        || Number(a.order) - Number(b.order) || stableTextCompare(a.graph_node_id, b.graph_node_id));
      const byLayer = new Map();
      for (const row of localRows) {
        const layoutLayer = Number(row.layout_layer ?? row.layer ?? 0);
        if (!byLayer.has(layoutLayer)) byLayer.set(layoutLayer, []);
        byLayer.get(layoutLayer).push(row);
      }
      const maxLayerRows = Math.max(1, ...[...byLayer.values()].map(rows => rows.length));
      const maximumNode = nodeSize({ node_type: 'molecule' }, metrics.nodeScale);
      const maximumLayer = Math.max(0, ...localRows.map(row => Number(row.layout_layer ?? row.layer ?? 0)));
      const wrapsLongLinearRoute = state.mode === 'current'
        && orientation === 'horizontal'
        && maxLayerRows === 1
        && maximumLayer >= 12;
      const layerByNode = new Map(localRows.map(row => [row.graph_node_id, Number(row.layout_layer ?? row.layer ?? 0)]));
      const layerEdgeCounts = new Map();
      for (const edge of localEdges) {
        const sourceLayer = layerByNode.get(edge.source_graph_node_id);
        const targetLayer = layerByNode.get(edge.target_graph_node_id);
        if (!Number.isFinite(sourceLayer) || !Number.isFinite(targetLayer)) continue;
        const key = [sourceLayer, targetLayer].sort((left, right) => left - right).join(':');
        layerEdgeCounts.set(key, Number(layerEdgeCounts.get(key) || 0) + 1);
      }
      const maximumLayerEdgeCount = Math.max(1, ...layerEdgeCounts.values());
      const routedLayerGap = state.mode === 'current' && orientation === 'horizontal' && !wrapsLongLinearRoute
        ? Math.max(
            metrics.layerGap,
            maximumNode.w + 34 + Math.min(8, maximumLayerEdgeCount - 1) * 10
          )
        : metrics.layerGap;
      const wrapColumns = wrapsLongLinearRoute
        ? Math.min(7, maximumLayer + 1)
        : 0;
      const wrapRows = wrapsLongLinearRoute
        ? Math.ceil((maximumLayer + 1) / wrapColumns)
        : 0;
      const expandedBoundaryLayers = expandedProgram ? [
        graphNodeIdByMolecule.get(expandedProgram.input_molecule_node_ids?.[0]),
        graphNodeIdByMolecule.get(expandedProgram.output_molecule_node_ids?.[0])
      ].map(graphNodeId => layerByNode.get(graphNodeId)).filter(Number.isFinite) : [];
      const expandedBoundaryDisplayLayers = expandedBoundaryLayers.map(layer =>
        isRetrosynthesis() ? maximumLayer - layer : layer);
      const drawerAfterWrapRow = wrapsLongLinearRoute && expandedBoundaryDisplayLayers.length
        ? Math.max(...expandedBoundaryDisplayLayers.map(layer => Math.floor(layer / wrapColumns)))
        : 0;
      const drawerAfterDisplayLayer = expandedBoundaryDisplayLayers.length
        ? Math.max(...expandedBoundaryDisplayLayers) : 0;
      const wrapColumnGap = Math.max(routedLayerGap, maximumNode.w + 24);
      const wrapRowGap = Math.max(metrics.rowGap + 52, 196);
      const naturalTileWidth = wrapsLongLinearRoute
        ? 92 + (wrapColumns - 1) * wrapColumnGap + maximumNode.w
        : orientation === 'vertical'
        ? 40 + (maxLayerRows - 1) * metrics.rowGap + maximumNode.w
        : 92 + maximumLayer * routedLayerGap + maximumNode.w;
      const tileWidth = orientation === 'vertical' && expandedProgram
        ? Math.max(360, naturalTileWidth) : naturalTileWidth;
      const tileHeight = wrapsLongLinearRoute
        ? 86 + programInset + (wrapRows - 1) * wrapRowGap + maximumNode.h + programDrawerInset
        : orientation === 'vertical'
        ? 76 + programInset + maximumLayer * routedLayerGap + maximumNode.h + programDrawerInset
        : 76 + programInset + (maxLayerRows - 1) * metrics.rowGap + maximumNode.h + programDrawerInset;
      const relativePositions = new Map();
      for (const [layerIndex, rows] of byLayer) {
        rows.sort((a, b) => Number(a.order || 0) - Number(b.order || 0) || stableTextCompare(a.graph_node_id, b.graph_node_id));
        rows.forEach((logical, rowIndex) => {
          const node = graphNodes.get(logical.graph_node_id);
          if (!node) return;
          const instanceId = `${lane.branch_id}::${node.graph_node_id}`;
          const size = nodeSize(node, metrics.nodeScale);
          const displayLayer = isRetrosynthesis() ? maximumLayer - layerIndex : layerIndex;
          const wrapRow = wrapsLongLinearRoute
            ? Math.floor(displayLayer / wrapColumns)
            : 0;
          const wrapOffset = wrapsLongLinearRoute
            ? displayLayer % wrapColumns
            : 0;
          const wrapColumn = wrapsLongLinearRoute && wrapRow % 2 === 1
            ? wrapColumns - 1 - wrapOffset
            : wrapOffset;
          const drawerShift = !programDrawerInset ? 0
            : wrapsLongLinearRoute
              ? (wrapRow > drawerAfterWrapRow ? programDrawerInset : 0)
              : orientation === 'vertical' && displayLayer > drawerAfterDisplayLayer
                ? programDrawerInset : 0;
          const position = wrapsLongLinearRoute
            ? {
                x: 52 + wrapColumn * wrapColumnGap,
                y: 52 + programInset + wrapRow * wrapRowGap + drawerShift,
                ...size
              }
            : orientation === 'vertical'
            ? { x: 20 + rowIndex * metrics.rowGap, y: 52 + programInset + displayLayer * routedLayerGap + drawerShift, ...size }
            : { x: 52 + displayLayer * routedLayerGap, y: 52 + programInset + rowIndex * metrics.rowGap, ...size };
          relativePositions.set(instanceId, position);
        });
      }
      return {
        lane, localEdges, byLayer, relativePositions, w: tileWidth, h: tileHeight,
        programInset,
        packing: wrapsLongLinearRoute ? 'serpentine_long_route.v1' : 'logical_layers.v1'
      };
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
        for (const edge of tile.localEdges) {
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
    model.packing = {
      algorithm: tiles.some(tile => tile.packing === 'serpentine_long_route.v1')
        ? 'serpentine_long_route.v1'
        : 'deterministic_adaptive_shelf_grid.v1',
      columns,
      targetRatio
    };
    model.programOverlays = layoutProgramOverlays(model);
    model.mechanismHypotheses = layoutMechanismHypotheses(model);
    return model;
  }

  function layoutProgramOverlays(model) {
    if (state.mode !== 'current') return [];
    const rows = programOverlaysByBranch.get(state.selectedBranchId) || [];
    const decoration = model.decorations.find(row => row.branchId === state.selectedBranchId);
    if (!decoration) return [];
    return rows.map(overlay => {
      const inputGraphId = graphNodeIdByMolecule.get(overlay.input_molecule_node_ids?.[0]);
      const outputGraphId = graphNodeIdByMolecule.get(overlay.output_molecule_node_ids?.[0]);
      const input = model.positions.get(`${overlay.branch_id}::${inputGraphId}`);
      const output = model.positions.get(`${overlay.branch_id}::${outputGraphId}`);
      const sourceGraphId = isRetrosynthesis() ? outputGraphId : inputGraphId;
      const targetGraphId = isRetrosynthesis() ? inputGraphId : outputGraphId;
      const source = model.positions.get(`${overlay.branch_id}::${sourceGraphId}`);
      const target = model.positions.get(`${overlay.branch_id}::${targetGraphId}`);
      if (!input || !output || !source || !target) return null;
      const expanded = programIsExpanded(overlay.program_id);
      let card;
      const sourceCenter = { x: source.x + source.w / 2, y: source.y + source.h / 2 };
      const targetCenter = { x: target.x + target.w / 2, y: target.y + target.h / 2 };
      const horizontal = Math.abs(sourceCenter.y - targetCenter.y)
        <= Math.max(source.h, target.h) * .7;
      if (horizontal) {
        const left = sourceCenter.x <= targetCenter.x ? source : target;
        const right = left === source ? target : source;
        const gap = Math.max(0, right.x - (left.x + left.w));
        const cardWidth = clamp(gap - 18, 224, 292);
        card = {
          x: left.x + left.w + (gap - cardWidth) / 2,
          y: (sourceCenter.y + targetCenter.y) / 2 - 53,
          w: cardWidth,
          h: 106
        };
      } else {
        const top = sourceCenter.y <= targetCenter.y ? source : target;
        const bottom = top === source ? target : source;
        const gap = Math.max(0, bottom.y - (top.y + top.h));
        const cardWidth = Math.min(284, Math.max(224, decoration.w - 48));
        card = {
          x: clamp(
            (sourceCenter.x + targetCenter.x) / 2 - cardWidth / 2,
            decoration.x + 24,
            Math.max(decoration.x + 24, decoration.x + decoration.w - cardWidth - 24)
          ),
          y: top.y + top.h + (gap - 106) / 2,
          w: cardWidth,
          h: 106
        };
      }
      const comparisonLane = expanded ? {
        x: card.x - 12,
        y: card.y - 10,
        w: card.w + 24,
        h: card.h + 20
      } : null;
      let fallbackDrawer = null;
      let fallbackCards = [];
      if (expanded) {
        const stepIds = overlay.replaced_step_ids || [];
        const columns = effectiveOrientation() === 'vertical'
          ? 1 : Math.min(3, Math.max(1, stepIds.length));
        const drawerWidth = effectiveOrientation() === 'vertical'
          ? Math.max(214, decoration.w - 48)
          : Math.min(760, Math.max(460, decoration.w - 48));
        const drawerRows = Math.max(1, Math.ceil(stepIds.length / columns));
        const drawerHeight = 54 + drawerRows * 62 + 12;
        fallbackDrawer = {
          x: clamp(
            card.x + card.w / 2 - drawerWidth / 2,
            decoration.x + 24,
            Math.max(decoration.x + 24, decoration.x + decoration.w - drawerWidth - 24)
          ),
          y: Math.max(source.y + source.h, target.y + target.h, card.y + card.h) + 18,
          w: drawerWidth,
          h: drawerHeight
        };
        const gap = 10;
        const stepCardWidth = (fallbackDrawer.w - 32 - (columns - 1) * gap) / columns;
        fallbackCards = stepIds.map((stepId, index) => {
          const drawerRow = Math.floor(index / columns);
          const offset = index % columns;
          const drawerColumn = drawerRow % 2 === 1 ? columns - 1 - offset : offset;
          return {
            stepId,
            index,
            x: fallbackDrawer.x + 16 + drawerColumn * (stepCardWidth + gap),
            y: fallbackDrawer.y + 45 + drawerRow * 62,
            w: stepCardWidth,
            h: 52
          };
        });
      }
      return {
        overlay, input, output, source, target, card,
        comparisonLane, fallbackDrawer, fallbackCards, expanded
      };
    }).filter(Boolean);
  }

  function programOverlayPath(from, to, { vertical = false } = {}) {
    if (vertical) {
      const middle = (from.y + to.y) / 2;
      return `M ${from.x} ${from.y} C ${from.x} ${middle}, ${to.x} ${middle}, ${to.x} ${to.y}`;
    }
    const middle = (from.x + to.x) / 2;
    return `M ${from.x} ${from.y} C ${middle} ${from.y}, ${middle} ${to.y}, ${to.x} ${to.y}`;
  }

  function boxPortToward(box, toward) {
    const center = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
    const dx = toward.x - center.x;
    const dy = toward.y - center.y;
    if (Math.abs(dx) / Math.max(box.w, 1) >= Math.abs(dy) / Math.max(box.h, 1)) {
      return { x: dx >= 0 ? box.x + box.w : box.x, y: center.y };
    }
    return { x: center.x, y: dy >= 0 ? box.y + box.h : box.y };
  }

  function layoutMechanismHypotheses(model) {
    if (state.mode !== 'current') return [];
    const rows = mechanismHypothesesByBranch.get(state.selectedBranchId) || [];
    const anchorSlots = new Map();
    return rows.map(hypothesis => {
      const graphNodeId = graphNodeIdByMolecule.get(hypothesis.anchor_molecule_node_id);
      const anchor = model.positions.get(`${hypothesis.branch_id}::${graphNodeId}`);
      if (!anchor) return null;
      const slot = Number(anchorSlots.get(graphNodeId) || 0);
      anchorSlots.set(graphNodeId, slot + 1);
      const card = {
        x: anchor.x + Math.max(8, (anchor.w - Math.min(188, anchor.w - 16)) / 2),
        y: anchor.y - 46 * (slot + 1),
        w: Math.min(188, anchor.w - 16),
        h: 40
      };
      return { hypothesis, anchor, card };
    }).filter(Boolean);
  }

  function mechanismHypothesisSvg(row) {
    const { hypothesis, anchor, card } = row;
    const selected = state.selectedMechanismId === hypothesis.hypothesis_id;
    const anchorX = clamp(anchor.x + anchor.w * .72, card.x + 12, card.x + card.w - 12);
    const label = middleEllipsis(hypothesis.proposed_product?.label || 'proposed one-hop product', 27);
    return `<g class="mechanism-hypothesis-callout${selected ? ' is-selected' : ''}" data-mechanism-id="${esc(hypothesis.hypothesis_id)}" data-branch-id="${esc(hypothesis.branch_id)}" tabindex="${selected ? '0' : '-1'}" role="button" aria-label="机理假设：文献锚点后的一跳提案，尚未验证">
      <path class="mechanism-hypothesis-tether" d="M ${anchorX} ${card.y + card.h} L ${anchorX} ${anchor.y}"></path>
      <circle class="mechanism-hypothesis-pin" cx="${anchorX}" cy="${anchor.y}" r="4"></circle>
      <rect class="mechanism-hypothesis-card" x="${card.x}" y="${card.y}" width="${card.w}" height="${card.h}" rx="10"></rect>
      <rect class="mechanism-hypothesis-stripe" x="${card.x}" y="${card.y + 5}" width="4" height="${card.h - 10}" rx="2"></rect>
      <text class="mechanism-hypothesis-kicker" x="${card.x + 12}" y="${card.y + 15}">H1 · MECHANISM · ONE HOP</text>
      <text class="mechanism-hypothesis-title" x="${card.x + 12}" y="${card.y + 31}">${esc(label)} · 待验证</text>
    </g>`;
  }

  function programFallbackDrawerSvg(row) {
    const { overlay, card, fallbackDrawer, fallbackCards, expanded } = row;
    if (!expanded || !fallbackDrawer) return '';
    const connectors = fallbackCards.slice(1).map((current, index) => {
      const previous = fallbackCards[index];
      if (Math.abs(previous.y - current.y) < 1) {
        const forward = current.x > previous.x;
        const x1 = forward ? previous.x + previous.w : previous.x;
        const x2 = forward ? current.x : current.x + current.w;
        const y = previous.y + previous.h / 2;
        return `<path class="program-fallback-connector" d="M ${x1} ${y} H ${x2}" marker-end="url(#arrow-neutral)"></path>`;
      }
      const x1 = previous.x + previous.w / 2;
      const x2 = current.x + current.w / 2;
      const y1 = previous.y + previous.h;
      const y2 = current.y;
      const middle = (y1 + y2) / 2;
      return `<path class="program-fallback-connector" d="M ${x1} ${y1} V ${middle} H ${x2} V ${y2}" marker-end="url(#arrow-neutral)"></path>`;
    }).join('');
    const cards = fallbackCards.map(value => {
      const step = steps.get(value.stepId) || {};
      const selected = state.selectedStepId === value.stepId;
      const title = middleEllipsis(routeStepDisplayLabel(step), 29);
      const detail = middleEllipsis(
        step.reaction_class || step.label || step.condition_summary || '化学反应步骤',
        31
      );
      return `<g class="program-fallback-compact${selected ? ' is-selected' : ''}" data-program-fallback-step="${esc(value.stepId)}" data-program-owner-id="${esc(overlay.program_id)}" data-branch-id="${esc(overlay.branch_id)}" role="button" tabindex="${selected ? '0' : '-1'}" aria-label="化学基线第 ${value.index + 1} 步：${esc(title)}">
        <rect x="${value.x}" y="${value.y}" width="${value.w}" height="${value.h}" rx="11"></rect>
        <circle cx="${value.x + 18}" cy="${value.y + 18}" r="10"></circle>
        <text class="program-fallback-number" x="${value.x + 18}" y="${value.y + 21}" text-anchor="middle">${value.index + 1}</text>
        <text class="program-fallback-title" x="${value.x + 34}" y="${value.y + 18}">${esc(title)}</text>
        <text class="program-fallback-detail" x="${value.x + 34}" y="${value.y + 36}">${esc(detail)}</text>
        <text class="program-fallback-proof" x="${value.x + value.w - 10}" y="${value.y + 36}" text-anchor="end">${esc(tierLabel(tierOfStep(step)))}</text>
      </g>`;
    }).join('');
    const tetherX = clamp(card.x + card.w / 2, fallbackDrawer.x + 24, fallbackDrawer.x + fallbackDrawer.w - 24);
    return `<g class="program-fallback-drawer-layer" aria-label="展开的 canonical 化学基线">
      <path class="program-fallback-drawer-tether" d="M ${card.x + card.w / 2} ${card.y + card.h} V ${fallbackDrawer.y - 8} H ${tetherX} V ${fallbackDrawer.y}"></path>
      <rect class="program-fallback-drawer" x="${fallbackDrawer.x}" y="${fallbackDrawer.y}" width="${fallbackDrawer.w}" height="${fallbackDrawer.h}" rx="16"></rect>
      <text class="program-fallback-drawer-title" x="${fallbackDrawer.x + 16}" y="${fallbackDrawer.y + 22}">化学基线 · ${fallbackCards.length} 步局部对照</text>
      <text class="program-fallback-drawer-meta" x="${fallbackDrawer.x + 16}" y="${fallbackDrawer.y + 38}">CANONICAL · 条件与证据完整保留 · 点击步骤检查</text>
      <rect class="program-fallback-authority" x="${fallbackDrawer.x + fallbackDrawer.w - 82}" y="${fallbackDrawer.y + 12}" width="68" height="20" rx="10"></rect>
      <text class="program-fallback-authority-label" x="${fallbackDrawer.x + fallbackDrawer.w - 48}" y="${fallbackDrawer.y + 26}" text-anchor="middle">权威基线</text>
      ${connectors}${cards}
    </g>`;
  }

  function programOverlaySvg(row) {
    const {
      overlay, input, output, source, target, card,
      comparisonLane, expanded
    } = row;
    const cardCenter = { x: card.x + card.w / 2, y: card.y + card.h / 2 };
    const sourceCenter = { x: source.x + source.w / 2, y: source.y + source.h / 2 };
    const targetCenter = { x: target.x + target.w / 2, y: target.y + target.h / 2 };
    const sourcePoint = boxPortToward(source, cardCenter);
    const targetPoint = boxPortToward(target, cardCenter);
    const cardInput = boxPortToward(card, sourceCenter);
    const cardOutput = boxPortToward(card, targetCenter);
    const vertical = Math.abs(sourceCenter.y - targetCenter.y)
      > Math.abs(sourceCenter.x - targetCenter.x) * .6;
    const first = programOverlayPath(sourcePoint, cardInput, { vertical });
    const second = programOverlayPath(cardOutput, targetPoint, { vertical });
    const selected = state.selectedProgramId === overlay.program_id;
    const equivalent = Number(overlay.chemical_step_equivalent_count || overlay.replaced_step_ids?.length || 0);
    const enzymeLabel = (overlay.candidate_enzyme_ids || []).join(' · ')
      || (overlay.enzyme_classes || []).join(' · ') || '候选酶待筛选';
    const warning = overlay.validation_status === 'experiment_required' ? '待实验' : overlay.validation_status;
    const inputLabelX = input.x + input.w / 2;
    const outputLabelX = output.x + output.w / 2;
    const toggleLabel = expanded
      ? `化学基线 ${equivalent} 步已展开 · 收起对照`
      : `化学基线 ${equivalent} 步完整保留 · 展开对照`;
    const fallbackDrawer = programFallbackDrawerSvg(row);
    return `<g class="program-overlay${selected ? ' is-selected' : ''}${expanded ? ' is-expanded' : ' is-collapsed'}" data-program-id="${esc(overlay.program_id)}" data-branch-id="${esc(overlay.branch_id)}" data-program-view="${expanded ? 'comparison' : 'replacement'}" tabindex="${selected ? '0' : '-1'}" role="button" aria-label="候选酶 Program：${equivalent} 步化学路线建议压缩为 1 步，${esc(warning)}，化学基线${expanded ? '已展开' : '已折叠'}">
      ${expanded ? `<rect class="program-comparison-lane" x="${comparisonLane.x}" y="${comparisonLane.y}" width="${comparisonLane.w}" height="${comparisonLane.h}" rx="18"></rect>
      <text class="program-comparison-lane-label" x="${comparisonLane.x + 12}" y="${comparisonLane.y + 16}">候选酶路径 · 不授予路线权威</text>` : ''}
      <path class="program-overlay-path program-overlay-path--input" d="${first}"></path>
      <path class="program-overlay-path program-overlay-path--output" d="${second}" marker-end="url(#arrow-program)"></path>
      <circle class="program-overlay-port" cx="${sourcePoint.x}" cy="${sourcePoint.y}" r="4"></circle>
      <circle class="program-overlay-port" cx="${targetPoint.x}" cy="${targetPoint.y}" r="4"></circle>
      <g class="program-boundary-label program-boundary-label--input"><rect x="${inputLabelX - 34}" y="${input.y - 23}" width="68" height="17" rx="8.5"></rect><text x="${inputLabelX}" y="${input.y - 11}" text-anchor="middle">PROGRAM 输入</text></g>
      <g class="program-boundary-label program-boundary-label--output"><rect x="${outputLabelX - 34}" y="${output.y - 23}" width="68" height="17" rx="8.5"></rect><text x="${outputLabelX}" y="${output.y - 11}" text-anchor="middle">PROGRAM 输出</text></g>
      <rect class="program-overlay-card" x="${card.x}" y="${card.y}" width="${card.w}" height="${card.h}" rx="15"></rect>
      <rect class="program-overlay-warning-stripe" x="${card.x}" y="${card.y + 8}" width="5" height="${card.h - 16}" rx="2.5"></rect>
      <text class="program-overlay-kicker" x="${card.x + 16}" y="${card.y + 19}">候选酶替代 · 未验证</text>
      <text class="program-overlay-title" x="${card.x + 16}" y="${card.y + 42}">${equivalent} 个化学步骤 → 1 个酶操作</text>
      <text class="program-overlay-meta" x="${card.x + 16}" y="${card.y + 61}">净节省 ${Number(overlay.net_step_savings || 0)} 步 · ${esc(middleEllipsis(enzymeLabel, 25))}</text>
      <rect class="program-overlay-status" x="${card.x + card.w - 72}" y="${card.y + 9}" width="60" height="20" rx="10"></rect>
      <text class="program-overlay-status-label" x="${card.x + card.w - 42}" y="${card.y + 23}" text-anchor="middle">${esc(warning)}</text>
      <g class="program-baseline-toggle" data-program-toggle="${esc(overlay.program_id)}" role="button" tabindex="0" aria-expanded="${expanded}" aria-label="${esc(toggleLabel)}">
        <rect x="${card.x + 11}" y="${card.y + card.h - 30}" width="${card.w - 22}" height="22" rx="11"></rect>
        <text x="${card.x + card.w / 2}" y="${card.y + card.h - 15}" text-anchor="middle">${esc(toggleLabel)} ${expanded ? '⌃' : '⌄'}</text>
      </g>
      ${fallbackDrawer}
    </g>`;
  }

  function stepIsProgramFallback(stepId, branchId) {
    return (programOverlaysByBranch.get(branchId) || [])
      .some(overlay => (overlay.replaced_step_ids || []).includes(stepId));
  }

  function nodeSize(node, scale) {
    const reaction = node.node_type === 'reaction';
    const mobileCurrent = state.mode === 'current' && matchMedia('(max-width: 639px)').matches;
    const base = state.mode === 'current'
      ? (mobileCurrent
        ? (reaction ? { w: 148, h: 76 } : { w: 158, h: 118 })
        : (reaction ? { w: 176, h: 84 } : { w: 220, h: 144 }))
      : (reaction ? { w: 182, h: 86 } : { w: 194, h: 78 });
    return { w: Math.round(base.w * scale), h: Math.round(base.h * scale) };
  }

  function finaliseModel(mode, instances, edges, positions, decorations) {
    const boxes = [...positions.values(), ...decorations];
    const maxX = Math.max(mode === 'current' ? 1 : 700, ...boxes.map(row => row.x + row.w + 44));
    const maxY = Math.max(mode === 'current' ? 1 : 420, ...boxes.map(row => row.y + row.h + 44));
    return { mode, instances, edges, positions, decorations, bounds: { x: 0, y: 0, w: maxX, h: maxY }, packing: null };
  }

  function renderGraph({ fit = true } = {}) {
    const updateStartedAt = performance.now();
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
    const marker = (id, color) => `<marker id="${id}" viewBox="0 0 14 10" markerWidth="14" markerHeight="10" refX="13" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L13,5 L1,9 Z" fill="${color}"></path></marker>`;
    const markers = unique(PROOF_ORDER.map(tierClass)).map(cssClass => {
      const tier = PROOF_ORDER.find(value => tierClass(value) === cssClass);
      return marker(`arrow-${cssClass}`, TIER_COLOR[tier] || '#64748b');
    }).join('')
      + Object.entries(PRODUCER_COLOR).map(([producer, color]) =>
        marker(`arrow-${producer}`, color)
      ).join('')
      + marker('arrow-neutral', '#94a3b8')
      + marker('arrow-program', '#7c3aed');
    const decorations = renderModel.decorations.map(row => {
      const priority = state.mode === 'clusters' && isPriorityBranch(row.branchId);
      const actionLabel = priority ? '查看重点分支' : '查看路线';
      return `<g class="graph-lane-decoration${row.branchId === state.selectedBranchId ? ' is-selected' : ''}${priority ? ' is-priority' : ''}" data-lane-branch-id="${esc(row.branchId)}" role="button" tabindex="0" aria-label="${actionLabel} ${esc(row.label)}">
        <rect x="${row.x}" y="${row.y}" width="${row.w}" height="${row.h}" rx="16"></rect>
        ${priority ? `<text class="graph-lane-priority-label" x="${row.x + 16}" y="${row.y + 24}">重点分支</text>` : ''}
        <text x="${row.x + (priority ? 82 : 16)}" y="${row.y + 24}">${esc(middleEllipsis(row.label, priority ? 36 : 46))}</text>
        <text class="graph-lane-open-label" x="${row.x + row.w - 14}" y="${row.y + 24}" text-anchor="end">${actionLabel}</text></g>`;
    }).join('');
    const edgeRoutingPlan = buildEdgeRoutingPlan(
      renderModel.edges,
      renderModel.positions,
      renderModel.packing?.algorithm || 'logical_layers.v1'
    );
    const edges = renderModel.edges.map(edge => edgeSvg(
      edge,
      renderModel.positions,
      edgeRoutingPlan.byEdge.get(edge)
    )).join('');
    const nodes = renderModel.instances.map(instance => nodeSvg(
      instance,
      renderModel.positions.get(instance.instanceId),
      edgeRoutingPlan.portsByInstance.get(instance.instanceId) || []
    )).join('');
    const laneOpenControls = renderModel.decorations.map(row => {
      const priority = state.mode === 'clusters' && isPriorityBranch(row.branchId);
      const actionLabel = priority ? '查看重点分支' : '查看路线';
      const controlWidth = Math.min(138, Math.max(84, row.w - 28));
      const controlX = row.x + row.w - controlWidth - 12;
      return `<g class="graph-lane-open-control${priority ? ' is-priority' : ''}" data-lane-branch-id="${esc(row.branchId)}" role="button" tabindex="0" aria-label="${actionLabel} ${esc(row.label)}">
        <rect x="${controlX}" y="${row.y + 7}" width="${controlWidth}" height="26" rx="13"></rect>
        <text x="${controlX + controlWidth / 2}" y="${row.y + 24}" text-anchor="middle">${actionLabel}</text></g>`;
    }).join('');
    const programOverlayRows = (renderModel.programOverlays || []).map(programOverlaySvg).join('');
    const mechanismRows = (renderModel.mechanismHypotheses || []).map(mechanismHypothesisSvg).join('');
    element('mainRoute').innerHTML = `<svg class="graph-svg dependency-svg" data-layout-packing="${esc(renderModel.packing?.algorithm || 'shared_component_layers.v1')}" data-maximum-edge-tracks="${edgeRoutingPlan.maximumTrackCount}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="graphTitle graphSubtitle">
      <defs>${markers}</defs><g class="graph-world">${decorations}${edges}${programOverlayRows}${nodes}${mechanismRows}${laneOpenControls}</g></svg>`;
    indexRenderedObjects();
    renderPerformance.lastGraphUpdateMs = performance.now() - updateStartedAt;
    const visibleBranches = new Set(renderModel.instances.map(row => row.branchId).filter(Boolean));
    element('graphVisibleCount').textContent = `${visibleBranches.size || (state.mode === 'shared' ? Math.min(filteredLanes().length, portfolioDisplayLimit()) : 0)}/${filteredLanes().length} 探索视图 · ${renderModel.instances.length} 节点 · ${renderModel.edges.length} 边${renderModel.programOverlays?.length ? ` · ${renderModel.programOverlays.length} 个候选酶 Program` : ''}`;
    if (renderModel.mechanismHypotheses?.length) element('graphVisibleCount').textContent += ` · ${renderModel.mechanismHypotheses.length} 个机理假设`;
    const replacementPreview = state.activeReplacement && state.mode === 'current';
    const overviewToggle = element('overviewToggle');
    const filteredCount = filteredLanes().length;
    const topK = portfolioDisplayLimit();
    overviewToggle.hidden = !['clusters', 'shared'].includes(state.mode) || filteredCount <= topK;
    overviewToggle.textContent = state.showAllOverview
      ? `仅显示 Top ${topK}` : `显示全部 ${filteredCount} 个探索视图`;
    const currentLane = laneByBranch.get(state.selectedBranchId) || {};
    element('graphTitle').textContent = replacementPreview ? '完整替换路线预览 · 后端已重验'
      : state.mode === 'clusters' ? `路线全景 · ${visibleBranches.size}/${filteredCount} 个探索视图`
        : state.mode === 'shared' ? '共享骨架 · 规范分子–反应图'
          : `当前路线 · ${middleEllipsis(currentLane.title || state.selectedBranchId, 48)}`;
    element('graphSubtitle').textContent = replacementPreview
      ? '当前画布是完整的后端 AND/OR 重验分支；它不是单步拼接，也不建立父路线证明。'
      : state.mode === 'clusters'
        ? '默认按闭合度与可信度展示高价值 Top-K；只有主分支标为重点分支。反应框线、左色条、连接线与箭头均表示方案生产者；虚实、线宽和 L 等级文字表示证明。'
        : state.mode === 'shared'
          ? '相同规范分子合并显示；框线、左色条、连接线与箭头表示生产者，虚实、线宽和 L 等级文字表示证明。'
          : `${(currentLane.step_ids || []).length} 步 · ${routeProofMixLabel(currentLane)} · ${(currentLane.source_refs || []).length} 个来源引用；框线、左色条、连接线与箭头表示生产者，虚实、线宽和 L 等级文字表示证明。`;
    renderMinimap();
    if (fit) fitGraph({ readable: preferReadableFocus() }); else applyViewportTransform();
    applyGraphSelection();
  }

  function visualEdgeRecord(edge, positions) {
    const forwardSource = positions.get(edge.sourceInstanceId);
    const forwardTarget = positions.get(edge.targetInstanceId);
    const source = isRetrosynthesis() ? forwardTarget : forwardSource;
    const target = isRetrosynthesis() ? forwardSource : forwardTarget;
    if (!source || !target) return null;
    const sourceNode = graphNodes.get(isRetrosynthesis()
      ? edge.target_graph_node_id : edge.source_graph_node_id);
    const targetNode = graphNodes.get(isRetrosynthesis()
      ? edge.source_graph_node_id : edge.target_graph_node_id);
    return {
      edge,
      source,
      target,
      sourceNode,
      targetNode,
      sourceInstanceId: isRetrosynthesis() ? edge.targetInstanceId : edge.sourceInstanceId,
      targetInstanceId: isRetrosynthesis() ? edge.sourceInstanceId : edge.targetInstanceId,
      sourceCenterY: source.y + source.h / 2,
      targetCenterY: target.y + target.h / 2
    };
  }

  function buildEdgeRoutingPlan(edges, positions, packing) {
    const byEdge = new Map();
    const records = edges.map(edge => visualEdgeRecord(edge, positions)).filter(Boolean);
    if (state.mode !== 'current' || effectiveOrientation() !== 'horizontal'
        || packing === 'serpentine_long_route.v1') {
      return { byEdge, portsByInstance: new Map(), maximumTrackCount: 1 };
    }
    assignPhysicalPortOffsets(records, byEdge);
    const gapGroups = groupBy(records, row => {
      const reverse = row.target.x + row.target.w / 2 < row.source.x + row.source.w / 2;
      const sourceX = reverse ? row.source.x : row.source.x + row.source.w;
      const targetX = reverse ? row.target.x + row.target.w : row.target.x;
      return `${Math.round((sourceX + targetX) / 8)}`;
    });
    let maximumTrackCount = 1;
    for (const group of gapGroups.values()) {
      group.sort((left, right) => left.sourceCenterY - right.sourceCenterY
        || left.targetCenterY - right.targetCenterY
        || stableTextCompare(left.edge.edge_id, right.edge.edge_id));
      maximumTrackCount = Math.max(maximumTrackCount, group.length);
      const gapLeft = Math.max(...group.map(row => {
        const reverse = row.target.x + row.target.w / 2 < row.source.x + row.source.w / 2;
        const sourceX = reverse ? row.source.x : row.source.x + row.source.w;
        const targetX = reverse ? row.target.x + row.target.w : row.target.x;
        return Math.min(sourceX, targetX);
      }));
      const gapRight = Math.min(...group.map(row => {
        const reverse = row.target.x + row.target.w / 2 < row.source.x + row.source.w / 2;
        const sourceX = reverse ? row.source.x : row.source.x + row.source.w;
        const targetX = reverse ? row.target.x + row.target.w : row.target.x;
        return Math.max(sourceX, targetX);
      }));
      const usableWidth = Math.max(0, gapRight - gapLeft - 20);
      const trackSpan = Math.min(usableWidth, Math.max(0, group.length - 1) * 12);
      const trackStart = (gapLeft + gapRight - trackSpan) / 2;
      group.forEach((row, index) => {
        const current = byEdge.get(row.edge) || {};
        byEdge.set(row.edge, {
          ...current,
          channelX: group.length === 1
            ? (gapLeft + gapRight) / 2
            : trackStart + index * trackSpan / (group.length - 1),
          trackIndex: index + 1,
          trackCount: group.length
        });
      });
    }
    return {
      byEdge,
      portsByInstance: collectReactionSidePorts(records, byEdge),
      maximumTrackCount
    };
  }

  function groupBy(rows, keyOf) {
    const groups = new Map();
    for (const row of rows) {
      const key = keyOf(row);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    return groups;
  }

  function edgePhysicalSides(row) {
    const reverse = row.target.x + row.target.w / 2 < row.source.x + row.source.w / 2;
    return {
      sourceSide: reverse ? 'left' : 'right',
      targetSide: reverse ? 'right' : 'left'
    };
  }

  function assignPhysicalPortOffsets(records, plan) {
    const attachments = [];
    for (const row of records) {
      const { sourceSide, targetSide } = edgePhysicalSides(row);
      attachments.push({
        row,
        instanceId: row.sourceInstanceId,
        position: row.source,
        graphNode: row.sourceNode,
        side: sourceSide,
        field: 'sourcePortOffset',
        peerY: row.targetCenterY
      });
      attachments.push({
        row,
        instanceId: row.targetInstanceId,
        position: row.target,
        graphNode: row.targetNode,
        side: targetSide,
        field: 'targetPortOffset',
        peerY: row.sourceCenterY
      });
    }
    const groups = groupBy(attachments, row => `${row.instanceId}:${row.side}`);
    for (const group of groups.values()) {
      group.sort((left, right) => left.peerY - right.peerY
        || stableTextCompare(left.row.edge.edge_id, right.row.edge.edge_id));
      const maximumSpan = group[0]?.graphNode?.node_type === 'reaction'
        ? Math.min(40, Number(group[0]?.position?.h || 0) * .5)
        : Number(group[0]?.position?.h || 0) * .62;
      const span = Math.min(maximumSpan, Math.max(0, group.length - 1) * 18);
      group.forEach((attachment, index) => {
        const current = plan.get(attachment.row.edge) || {};
        plan.set(attachment.row.edge, {
          ...current,
          [attachment.field]: group.length === 1
            ? 0 : -span / 2 + index * span / (group.length - 1)
        });
      });
    }
  }

  function collectReactionSidePorts(records, plan) {
    const portsByInstance = new Map();
    const append = (instanceId, port) => {
      if (!portsByInstance.has(instanceId)) portsByInstance.set(instanceId, []);
      const ports = portsByInstance.get(instanceId);
      if (!ports.some(row => Math.abs(row.x - port.x) < .5 && Math.abs(row.y - port.y) < .5)) {
        ports.push(port);
      }
    };
    for (const row of records) {
      const edgePlan = plan.get(row.edge) || {};
      const { sourceSide, targetSide } = edgePhysicalSides(row);
      if (row.sourceNode?.node_type === 'reaction') {
        append(row.sourceInstanceId, {
          x: sourceSide === 'left' ? row.source.x : row.source.x + row.source.w,
          y: row.sourceCenterY + Number(edgePlan.sourcePortOffset || 0)
        });
      }
      if (row.targetNode?.node_type === 'reaction') {
        append(row.targetInstanceId, {
          x: targetSide === 'left' ? row.target.x : row.target.x + row.target.w,
          y: row.targetCenterY + Number(edgePlan.targetPortOffset || 0)
        });
      }
    }
    return portsByInstance;
  }

  function edgeSvg(edge, positions, routingPlan = {}) {
    const record = visualEdgeRecord(edge, positions);
    if (!record) return '';
    const { source, target } = record;
    const step = steps.get(edge.reaction_step_id);
    const tier = edge.trust_vector?.proof_tier || tierOfStep(step);
    const visual = edge.visual_encoding || step?.visual_encoding || step?.trust_vector?.visual_encoding || {};
    const originClass = producerClass(step);
    const simple = state.edgeStyle === 'simple';
    const contrast = state.edgeStyle === 'contrast';
    const color = simple ? '#94a3b8' : producerColor(step);
    const width = contrast ? Math.max(2.4, Number(visual.width || 1.5)) : (simple ? 1.25 : Number(visual.width || 1.5));
    const opacity = contrast ? .92 : (simple ? .48 : Number(visual.opacity || .62));
    const dash = simple ? '' : String(visual.dash_pattern || '');
    const packing = renderModel?.packing?.algorithm || 'logical_layers.v1';
    const forceOrthogonal = state.edgeStyle !== 'trust';
    const path = edgePath(source, target, {
      orthogonal: forceOrthogonal,
      packing,
      sourcePortOffset: routingPlan.sourcePortOffset,
      targetPortOffset: routingPlan.targetPortOffset,
      channelX: routingPlan.channelX
    });
    const markerId = simple ? 'arrow-neutral' : `arrow-${originClass}`;
    const routing = forceOrthogonal ? 'fixed-port-channels.v3' : 'side-port-curves.v4';
    return `<path class="graph-edge dependency-edge trust-edge ${tierClass(tier)} ${originClass}${edge.visual_role === 'auxiliary' ? ' is-auxiliary' : ''}" data-edge-id="${esc(edge.edge_id)}" data-edge-color="${esc(color)}" data-edge-routing="${routing}" data-edge-track="${Number(routingPlan.trackIndex || 1)}/${Number(routingPlan.trackCount || 1)}" data-branch-id="${esc(edge.branch_id)}" data-reaction-step-id="${esc(edge.reaction_step_id)}" data-source-instance-id="${esc(edge.sourceInstanceId)}" data-target-instance-id="${esc(edge.targetInstanceId)}" d="${path}" style="--edge-color:${esc(color)};--edge-width:${width}px" stroke-width="${width}" opacity="${opacity}" stroke-dasharray="${esc(dash)}" marker-end="url(#${markerId})"><title>${esc(`${step?.producer_label || '来源未标记'} · ${tierLabel(tier)} · ${edge.visual_role === 'auxiliary' ? '辅助投入' : edge.edge_type || '显式依赖'} · ${edge.branch_id || ''}`)}</title></path>`;
  }

  function edgePath(source, target, {
    orthogonal = false,
    packing = '',
    sourcePortOffset = 0,
    targetPortOffset = 0,
    channelX = null
  } = {}) {
    if (effectiveOrientation() === 'vertical') {
      const x1 = source.x + source.w / 2, y1 = source.y + source.h;
      const x2 = target.x + target.w / 2, y2 = target.y;
      const middle = (y1 + y2) / 2;
      return orthogonal ? `M ${x1} ${y1} V ${middle} H ${x2} V ${y2}`
        : `M ${x1} ${y1} C ${x1} ${middle}, ${x2} ${middle}, ${x2} ${y2}`;
    }
    const sourceCenterX = source.x + source.w / 2;
    const targetCenterX = target.x + target.w / 2;
    const sourceCenterY = source.y + source.h / 2;
    const targetCenterY = target.y + target.h / 2;
    const rowSeparated = Math.abs(sourceCenterY - targetCenterY)
      > Math.max(source.h, target.h) * .62;
    const serpentineRowTurn = packing === 'serpentine_long_route.v1' && rowSeparated;
    if (serpentineRowTurn) {
      const descending = targetCenterY > sourceCenterY;
      const x1 = sourceCenterX, y1 = descending ? source.y + source.h : source.y;
      const x2 = targetCenterX, y2 = descending ? target.y : target.y + target.h;
      const channelY = (y1 + y2) / 2;
      return `M ${x1} ${y1} V ${channelY} H ${x2} V ${y2}`;
    }
    const reverse = targetCenterX < sourceCenterX;
    const x1 = reverse ? source.x : source.x + source.w;
    const y1 = sourceCenterY + Number(sourcePortOffset || 0);
    const x2 = reverse ? target.x + target.w : target.x;
    const y2 = targetCenterY + Number(targetPortOffset || 0);
    const middle = Number.isFinite(channelX) ? channelX : (x1 + x2) / 2;
    return orthogonal ? (Math.abs(y1 - y2) < 1
      ? `M ${x1} ${y1} H ${x2}`
      : `M ${x1} ${y1} H ${middle} V ${y2} H ${x2}`)
      : `M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`;
  }

  function nodeSvg(instance, position, ports = []) {
    if (!position) return '';
    const node = instance.node;
    const reaction = node.node_type === 'reaction';
    const molecule = reaction ? null : (moleculeNodes.get(node.molecule_node_id) || node);
    const step = reaction ? steps.get(node.reaction_step_id) : null;
    const tier = reaction ? (node.proof_tier || tierOfStep(step)) : '';
    const nodeTierClass = reaction ? tierClass(tier) : 'node-tier-neutral';
    const fullLabel = reaction ? routeStepDisplayLabel(step) : (node.label || node.graph_node_id);
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
    const originClass = reaction ? producerClass(step) : '';
    const producerMeta = reaction
      ? middleEllipsis(step?.producer_label || '来源未标记', 16)
      : '';
    const proofMeta = reaction ? middleEllipsis(tierLabel(tier), 12) : '';
    const conditionMeta = reaction ? inlineConditionText(step) : '';
    const conditionClass = step?.condition_status === 'source_exact'
      ? 'is-source-exact'
      : step?.condition_status === 'model_predicted'
        ? 'is-model-predicted'
        : step?.condition_status === 'source_recorded_unverified'
          ? 'is-source-candidate'
          : 'is-missing';
    const programFallbackClass = reaction
      && stepIsProgramFallback(node.reaction_step_id, instance.branchId)
      ? ' is-program-fallback' : '';
    const titleText = reaction ? `${fullLabel}\n${conditionMeta}` : fullLabel;
    const portDots = reaction ? ports.map(port => `<circle class="graph-node-port" cx="${port.x}" cy="${port.y}" r="3.4"></circle>`).join('') : '';
    const producerKind = reaction ? String((step?.producer_kinds || [])[0] || 'unknown') : '';
    const originColor = reaction ? producerColor(step) : '';
    return `<g class="graph-node dependency-${reaction ? 'reaction graph-node--reaction' : 'molecule graph-node--molecule'} ${nodeTierClass} ${originClass}${selected ? ' is-selected' : ''}${programFallbackClass}" data-graph-node-id="${esc(node.graph_node_id)}" data-node-type="${esc(node.node_type)}" data-node-role="${esc(node.role || '')}" data-producer-kind="${esc(producerKind)}"${reaction ? ` data-producer-color="${esc(originColor)}" style="--origin-color:${esc(originColor)}"` : ''} data-branch-id="${esc(instance.branchId)}" data-instance-id="${esc(instance.instanceId)}" ${reaction ? `data-route-step="${esc(node.reaction_step_id)}"` : ''} tabindex="${selected ? '0' : '-1'}" role="button" aria-label="${esc(`${reaction ? '反应' : '分子'}：${fullLabel}，${semanticLabel}`)}">
      <title>${esc(titleText)}</title>${reaction ? `<rect class="reaction-hit-target" x="${position.x - 5}" y="${position.y - 5}" width="${position.w + 10}" height="${position.h + 10}" rx="16"></rect>` : ''}<rect class="node-surface" x="${position.x}" y="${position.y}" width="${position.w}" height="${position.h}" rx="${reaction ? 12 : 24}"></rect>
      ${reaction ? `<rect class="reaction-origin-stripe" x="${position.x}" y="${position.y + 6}" width="5" height="${position.h - 12}" rx="2.5"></rect>` : ''}
      ${reaction ? `<rect class="reaction-proof-marker" x="${position.x + position.w - 28}" y="${position.y + 7}" width="16" height="4" rx="2"></rect>` : ''}
      ${structureSvg ? `<foreignObject class="node-depiction" x="${position.x + 7}" y="${position.y + 7}" width="${position.w - 14}" height="${position.h - 39}"><div xmlns="http://www.w3.org/1999/xhtml" class="node-depiction-frame">${structureSvg}</div></foreignObject>` : ''}
      <text class="node-label" x="${textX}" y="${textY}">${svgTextLines(lines, textX, textY)}</text>
      ${reaction && state.labelMode !== 'minimal' ? `<text class="reaction-producer-meta node-meta" x="${textX}" y="${position.y + position.h - 24}">${esc(producerMeta)}</text><text class="graph-node-tier node-meta" x="${position.x + position.w - 12}" y="${position.y + position.h - 24}" text-anchor="end">${esc(proofMeta)}</text><text class="reaction-condition-meta ${conditionClass}" x="${textX}" y="${position.y + position.h - 9}">${esc(conditionMeta)}</text>` : ''}
      ${portDots}</g>`;
  }

  function applyGraphSelection() {
    if (!renderModel) return;
    const selectedBranch = state.selectedBranchId;
    const selectedNode = state.selectedGraphNodeId;
    const selectedStep = state.selectedStepId;
    const selectedProgram = programOverlays.get(state.selectedProgramId);
    const selectedMechanism = mechanismHypotheses.get(state.selectedMechanismId);
    const mechanismAnchorGraphId = graphNodeIdByMolecule.get(selectedMechanism?.anchor_molecule_node_id);
    const programFallbackSteps = new Set(selectedProgram?.replaced_step_ids || []);
    const hasSelection = Boolean(selectedNode || selectedStep || selectedBranch || selectedProgram || selectedMechanism);
    let focusAssigned = false;
    document.querySelectorAll('.graph-node').forEach(row => {
      const selected = row.dataset.graphNodeId === selectedNode || row.dataset.routeStep === selectedStep;
      const mechanismAnchor = row.dataset.graphNodeId === mechanismAnchorGraphId;
      const related = selected || mechanismAnchor || (selectedBranch && row.dataset.branchId === selectedBranch);
      const focusable = selected && (state.selectedInstanceId
        ? row.dataset.instanceId === state.selectedInstanceId
        : !focusAssigned);
      if (focusable) focusAssigned = true;
      row.classList.toggle('is-selected', selected);
      row.classList.toggle('is-program-active', programFallbackSteps.has(row.dataset.routeStep));
      row.classList.toggle('is-mechanism-anchor', mechanismAnchor);
      row.classList.toggle('is-dimmed', hasSelection && !related && state.edgeFilter === 'selected');
      row.tabIndex = focusable ? 0 : -1;
    });
    document.querySelectorAll('.graph-edge').forEach(row => {
      const related = (selectedStep && row.dataset.reactionStepId === selectedStep)
        || (selectedBranch && row.dataset.branchId === selectedBranch)
        || (!selectedStep && !selectedBranch);
      row.classList.toggle('is-selected', Boolean(selectedStep && row.dataset.reactionStepId === selectedStep));
      row.classList.toggle('is-program-active', programFallbackSteps.has(row.dataset.reactionStepId));
      row.classList.toggle('is-dimmed', state.edgeFilter === 'selected' && hasSelection && !related);
    });
    document.querySelectorAll('[data-program-id]').forEach(row => {
      const selected = row.dataset.programId === state.selectedProgramId;
      row.classList.toggle('is-selected', selected);
      row.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-program-fallback-step]').forEach(row => {
      const selected = row.dataset.programFallbackStep === state.selectedStepId;
      row.classList.toggle('is-selected', selected);
      if (row.classList.contains('program-fallback-compact')) row.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-mechanism-id]').forEach(row => {
      const selected = row.dataset.mechanismId === state.selectedMechanismId;
      row.classList.toggle('is-selected', selected);
      row.tabIndex = selected ? 0 : -1;
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

  function preferReadableFocus() {
    if (state.mode !== 'current') return false;
    const lane = laneByBranch.get(state.selectedBranchId) || {};
    // Short embedded routes remain fully visible.  A convergent route with
    // many steps becomes illegible when squeezed into the narrower workspace,
    // so open that case at a moderate readable zoom and use the minimap for
    // the global overview.
    if (embeddedRoute) return (lane.step_ids || []).length > 4;
    // Short routes benefit from a readable close view.  Longer routes must
    // open fully visible; users can zoom in without first discovering that
    // the upstream half was placed outside the viewport.
    return (lane.step_ids || []).length <= 4;
  }

  function fitGraph({ readable = false, remember = true } = {}) {
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
      const minimumReadableZoom = embeddedRoute
        ? .42
        : matchMedia('(max-width: 639px)').matches ? .85 : .82;
      state.zoom = clamp(Math.max(naturalFit, minimumReadableZoom), minimumReadableZoom, 1.15);
      const oversized = naturalFit < minimumReadableZoom;
      const vertical = effectiveOrientation() === 'vertical';
      const target = currentTargetPosition();
      const targetOnLeft = target
        && target.x + target.w / 2 < renderModel.bounds.w / 2;
      state.panX = oversized && !vertical
        ? targetOnLeft
          ? 48 - target.x * state.zoom
          : width - ((target?.x ?? renderModel.bounds.w) + (target?.w || 0)) * state.zoom - 48
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
    if (remember) state.cameraMode = 'fit';
    applyViewportTransform();
  }

  function resetGraph() { state.zoom = 1; state.panX = 24; state.panY = 24; state.cameraMode = 'manual'; applyViewportTransform(); }
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
    state.cameraMode = 'manual';
    applyViewportTransform();
  }

  function resizeGraphViewport() {
    if (!renderModel) return;
    const viewport = element('graphViewport');
    if (viewport.dataset.orientation !== effectiveOrientation()) {
      graphModelCache.clear();
      renderGraph();
      return;
    }
    const svg = viewport.querySelector('.graph-svg');
    if (!svg) return;
    const width = Math.max(1, viewport.clientWidth || 1000);
    const height = Math.max(1, viewport.clientHeight || 620);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    if (state.cameraMode === 'fit') {
      fitGraph({ readable: preferReadableFocus(), remember: false });
    } else {
      applyViewportTransform({ forceCull: true });
    }
  }
  function indexRenderedObjects() {
    if (!renderModel) return;
    renderModel.nodeElements = new Map(
      [...document.querySelectorAll('.graph-node[data-instance-id]')]
        .map(row => [row.dataset.instanceId, row])
    );
    renderModel.edgeElements = [...document.querySelectorAll('.graph-edge')].map(row => ({
      row,
      sourceInstanceId: row.dataset.sourceInstanceId,
      targetInstanceId: row.dataset.targetInstanceId
    }));
    renderPerformance.renderedObjects = renderModel.nodeElements.size + renderModel.edgeElements.length;
    renderPerformance.culledObjects = 0;
    lastCullCamera = null;
  }
  function updateViewportCulling({ force = false } = {}) {
    if (!renderModel?.nodeElements) return;
    const objectCount = renderModel.nodeElements.size + renderModel.edgeElements.length;
    if (objectCount <= CULLING_OBJECT_THRESHOLD || state.mode === 'current') {
      if (lastCullCamera !== 'disabled') {
        renderModel.nodeElements.forEach(row => row.classList.remove('is-canvas-culled'));
        renderModel.edgeElements.forEach(({ row }) => row.classList.remove('is-canvas-culled'));
        renderPerformance.culledObjects = 0;
        lastCullCamera = 'disabled';
      }
      return;
    }
    const camera = { x: state.panX, y: state.panY, zoom: state.zoom };
    if (!force && lastCullCamera && lastCullCamera !== 'disabled'
      && Math.abs(camera.x - lastCullCamera.x) < 24
      && Math.abs(camera.y - lastCullCamera.y) < 24
      && Math.abs(camera.zoom - lastCullCamera.zoom) < .015) return;
    const viewport = element('graphViewport');
    const margin = CULLING_WORLD_MARGIN_PX / Math.max(state.zoom, .015);
    const bounds = {
      left: -state.panX / state.zoom - margin,
      top: -state.panY / state.zoom - margin,
      right: (viewport.clientWidth - state.panX) / state.zoom + margin,
      bottom: (viewport.clientHeight - state.panY) / state.zoom + margin
    };
    let culled = 0;
    renderModel.nodeElements.forEach((row, instanceId) => {
      const position = renderModel.positions.get(instanceId);
      const visible = position && position.x + position.w >= bounds.left
        && position.x <= bounds.right && position.y + position.h >= bounds.top
        && position.y <= bounds.bottom;
      row.classList.toggle('is-canvas-culled', !visible);
      if (!visible) culled += 1;
    });
    renderModel.edgeElements.forEach(({ row, sourceInstanceId, targetInstanceId }) => {
      const source = renderModel.positions.get(sourceInstanceId);
      const target = renderModel.positions.get(targetInstanceId);
      const left = Math.min(source?.x ?? Infinity, target?.x ?? Infinity);
      const top = Math.min(source?.y ?? Infinity, target?.y ?? Infinity);
      const right = Math.max((source?.x || 0) + (source?.w || 0), (target?.x || 0) + (target?.w || 0));
      const bottom = Math.max((source?.y || 0) + (source?.h || 0), (target?.y || 0) + (target?.h || 0));
      const visible = left <= bounds.right && right >= bounds.left
        && top <= bounds.bottom && bottom >= bounds.top;
      row.classList.toggle('is-canvas-culled', !visible);
      if (!visible) culled += 1;
    });
    renderPerformance.culledObjects = culled;
    lastCullCamera = camera;
  }
  function applyViewportTransform({ updateMinimap = true, forceCull = false } = {}) {
    const world = document.querySelector('.graph-world');
    const cameraTransform = `translate(${state.panX} ${state.panY}) scale(${state.zoom})`;
    if (world && world.getAttribute('transform') !== cameraTransform) world.setAttribute('transform', cameraTransform);
    const viewport = element('graphViewport');
    const zoomBand = state.zoom < .18 ? 'overview' : state.zoom < .5 ? 'medium' : 'detail';
    if (viewport.dataset.labelMode !== state.labelMode) viewport.dataset.labelMode = state.labelMode;
    if (viewport.dataset.zoomBand !== zoomBand) viewport.dataset.zoomBand = zoomBand;
    const zoomLabel = `${Math.round(state.zoom * 100)}%`;
    if (element('zoomReadout').textContent !== zoomLabel) element('zoomReadout').textContent = zoomLabel;
    updateViewportCulling({ force: forceCull });
    if (updateMinimap) updateMinimapViewport();
  }

  function renderMinimap() {
    const host = element('graphMinimap');
    if (!renderModel || !renderModel.instances.length) { host.innerHTML = ''; return; }
    host.hidden = false;
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
    host.toggleAttribute('hidden', matchMedia('(max-width: 639px)').matches || visibleRatio >= .92);
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
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
    persistState();
    if (state.mode === 'current' || state.edgeFilter === 'selected') renderGraph();
    else applyGraphSelection();
    renderDetail();
    announce(`已选择路线 ${branches.get(branchId)?.title || branchId}`);
    if (focusGraph) document.querySelector('.graph-node:not(.is-dimmed)')?.focus();
  }

  function openBranchInFocus(branchId, { focusGraph = false } = {}) {
    if (!branches.has(branchId)) return;
    state.activeReplacement = null;
    state.mode = 'current';
    state.showAllOverview = false;
    state.selectedBranchId = branchId;
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedStepId = '';
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
    rerenderForControls();
    announce(`已在当前路线中打开 ${isPriorityBranch(branchId) ? '重点分支' : '路线'} ${branches.get(branchId)?.title || branchId}`);
    if (focusGraph) requestAnimationFrame(() => {
      document.querySelector('.graph-node:not(.is-dimmed)')?.focus();
    });
  }
  function selectGraphNode(graphNodeId, { focus = false, branchId = '', instanceId = '' } = {}) {
    const node = graphNodes.get(graphNodeId);
    if (!node) return;
    state.selectedGraphNodeId = graphNodeId;
    state.selectedInstanceId = instanceId;
    state.selectedStepId = node.node_type === 'reaction' ? node.reaction_step_id : '';
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
    if (node.node_type === 'reaction') {
      state.detailTab = 'step';
      if (state.layoutPreset === 'focus') state.layoutPreset = 'review';
    }
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

  function selectProgram(programId, { focus = false } = {}) {
    const program = programOverlays.get(programId);
    if (!program) return;
    state.selectedProgramId = programId;
    state.selectedMechanismId = '';
    state.selectedBranchId = program.branch_id;
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedStepId = '';
    state.detailTab = 'step';
    state.inspectorOpen = true;
    if (isMobileDrawerLayout()) state.navOpen = false;
    applyPersistentChromeState();
    updateMobileNavigation('inspector');
    applyGraphSelection();
    renderDetail();
    persistState();
    announce(`已选择酶 Program，${program.chemical_step_equivalent_count} 步化学 fallback 压缩为 1 个候选操作`);
    if (focus && !isInspectorOverlayLayout()) {
      document.querySelector(`[data-program-id="${CSS.escape(programId)}"]`)?.focus();
    }
  }

  function toggleProgramFallback(programId, { expanded = null, focus = false } = {}) {
    const program = programOverlays.get(programId);
    if (!program) return;
    const nextExpanded = expanded === null ? !programIsExpanded(programId) : Boolean(expanded);
    if (nextExpanded) state.expandedProgramIds.add(programId);
    else state.expandedProgramIds.delete(programId);
    graphModelCache.clear();
    renderGraph();
    if (state.selectedProgramId === programId) renderDetail();
    persistState();
    announce(nextExpanded
      ? `已展开 ${program.chemical_step_equivalent_count} 步 canonical 化学基线，与候选酶路径对照显示`
      : `已收起化学基线，恢复 ${program.chemical_step_equivalent_count} 步到 1 个候选酶操作的内嵌视图`);
    if (focus) requestAnimationFrame(() => {
      document.querySelector(`[data-program-toggle="${CSS.escape(programId)}"]`)?.focus();
    });
  }

  function selectMechanism(hypothesisId, { focus = false } = {}) {
    const hypothesis = mechanismHypotheses.get(hypothesisId);
    if (!hypothesis) return;
    state.selectedMechanismId = hypothesisId;
    state.selectedProgramId = '';
    state.selectedBranchId = hypothesis.branch_id;
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedStepId = '';
    state.detailTab = 'step';
    state.inspectorOpen = true;
    if (isMobileDrawerLayout()) state.navOpen = false;
    applyPersistentChromeState();
    updateMobileNavigation('inspector');
    applyGraphSelection();
    renderDetail();
    persistState();
    announce('已选择文献锚点后的一跳机理假设；该提案尚未验证。');
    if (focus && !isInspectorOverlayLayout()) {
      document.querySelector(`[data-mechanism-id="${CSS.escape(hypothesisId)}"]`)?.focus();
    }
  }

  function selectedEntity() {
    if (state.selectedMechanismId && mechanismHypotheses.has(state.selectedMechanismId)) {
      return { type: 'mechanism', value: mechanismHypotheses.get(state.selectedMechanismId) };
    }
    if (state.selectedProgramId && programOverlays.has(state.selectedProgramId)) {
      return { type: 'program', value: programOverlays.get(state.selectedProgramId) };
    }
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
    document.querySelectorAll('[data-detail-tab]').forEach(button => {
      button.hidden = ['program', 'mechanism'].includes(entity.type) && button.dataset.detailTab !== 'step';
    });
    element('inspectorTitle').textContent = entity.type === 'reaction' ? '反应检查器'
      : entity.type === 'molecule' ? '分子检查器'
        : entity.type === 'program' ? '酶 Program 检查器' : '路线检查器';
    if (entity.type === 'mechanism') element('inspectorTitle').textContent = '机理假设检查器';
    if (entity.type === 'program') {
      host.innerHTML = programOverview(entity.value);
      return;
    }
    if (entity.type === 'mechanism') {
      host.innerHTML = mechanismOverview(entity.value);
      return;
    }
    if (state.detailTab === 'alternatives') { renderAlternatives(entity, host); return; }
    if (state.detailTab === 'evidence') { renderEvidence(entity, host); return; }
    host.innerHTML = entity.type === 'reaction' ? reactionOverview(entity.value)
      : entity.type === 'molecule' ? moleculeOverview(entity.value) : branchOverview(entity.value);
  }

  function mechanismOverview(hypothesis) {
    const anchor = moleculeNodes.get(hypothesis.anchor_molecule_node_id) || {};
    const product = hypothesis.proposed_product || {};
    const structureCard = (row, label, smiles) => `<div class="program-boundary-card mechanism-boundary-card"><span>${esc(label)}</span>${row.structure_svg ? `<div class="mol-structure">${safeStructureSvg(row.structure_svg)}</div>` : ''}<strong>${esc(row.formula || row.label || '未命名结构')}</strong><code>${esc(smiles || '未记录')}</code></div>`;
    const elementarySteps = (hypothesis.elementary_steps || [])
      .map((value, index) => `<div class="mechanism-task-row"><span>${index + 1}</span><p>${esc(value)}</p></div>`).join('');
    const checks = (hypothesis.falsifiable_checks || [])
      .map((value, index) => `<div class="mechanism-task-row mechanism-check-row"><span>${index + 1}</span><p>${esc(value)}</p></div>`).join('');
    const refs = (hypothesis.anchor_source_refs || [])
      .map(value => `<div class="trace-row"><code>${esc(value)}</code></div>`).join('');
    const warnings = (hypothesis.warning_codes || [])
      .map(value => `<code class="reason-code">${esc(value)}</code>`).join('');
    return `<article class="program-detail mechanism-detail"><header><p class="detail-kind">文献外机理假设 · 一跳边界</p><h3 class="detail-title">${esc(product.label || '提案产物')}</h3></header>
      <div class="notice mechanism-warning-notice"><strong>锚点论文不证明这条新边</strong><span>论文只证明下方锚点结构与原路线步骤；本提案不会继承其证据等级，也不会改变路线闭合或 canonical 边数。</span>${warnings ? `<div class="reason-code-list">${warnings}</div>` : ''}</div>
      <section class="program-stat-grid mechanism-stat-grid"><div><strong>H${Number(hypothesis.proposal_depth || 1)}</strong><span>只允许一跳</span></div><div><strong>${Math.round(Number(hypothesis.priority_score || 0) * 100)}%</strong><span>探索优先级</span></div><div><strong>未物化</strong><span>验证状态</span></div></section>
      <section class="detail-section"><h3>结构重接提案</h3><div class="program-boundary-grid">${structureCard(anchor, '文献锚点产物', anchor.canonical_isomeric_smiles || anchor.smiles)}${structureCard(product, '假设的一跳产物', product.canonical_smiles)}</div><p class="v">提案产物只显示在影子层中，不是路线节点；完成物化、结构重接与实验验证后才可进入候选边审查。</p></section>
      <section class="detail-section"><h3>机理依据</h3><p class="mechanism-rationale">${esc(hypothesis.mechanistic_rationale || '尚未记录')}</p><div class="mechanism-task-list">${elementarySteps || '<div class="empty">尚未拆解基本步骤</div>'}</div></section>
      <section class="detail-section"><h3>可证伪验证任务</h3><div class="mechanism-task-list">${checks}</div></section>
      <section class="detail-section"><h3>锚点证据隔离</h3><div class="trace-list">${refs}</div><div class="kv"><span class="k">锚点步骤</span><span class="v"><code>${esc((hypothesis.anchor_edge_ids || []).join(' · '))}</code></span></div><div class="kv"><span class="k">权限</span><span class="v">${esc(hypothesis.authority_scope || 'proposal_only')} · 不授予路线闭合</span></div></section>
    </article>`;
  }

  function programOverview(program) {
    const equivalent = Number(program.chemical_step_equivalent_count || program.replaced_step_ids?.length || 0);
    const boundaryCard = (nodeId, label) => {
      const node = moleculeNodes.get(nodeId) || {};
      return `<div class="program-boundary-card"><span>${esc(label)}</span>${node.structure_svg ? `<div class="mol-structure">${safeStructureSvg(node.structure_svg)}</div>` : ''}<strong>${esc(node.formula || node.label || nodeId)}</strong><code>${esc(node.canonical_isomeric_smiles || node.smiles || '未记录')}</code></div>`;
    };
    const inputs = (program.input_molecule_node_ids || []).map(id => boundaryCard(id, '精确输入边界')).join('');
    const outputs = (program.output_molecule_node_ids || []).map(id => boundaryCard(id, '精确输出边界')).join('');
    const fallbackRows = (program.replaced_step_ids || []).map((stepId, index) => {
      const step = steps.get(stepId) || {};
      return `<button class="program-fallback-row" type="button" data-program-fallback-step="${esc(stepId)}" data-program-owner-id="${esc(program.program_id)}" data-branch-id="${esc(program.branch_id)}"><span>${index + 1}</span><strong>${esc(routeStepDisplayLabel(step))}</strong><small>${esc(tierLabel(tierOfStep(step)))}</small></button>`;
    }).join('');
    const enzymeIds = (program.candidate_enzyme_ids || []).map(value => `<code>${esc(value)}</code>`).join('');
    const enzymeClasses = (program.enzyme_classes || []).map(value => `<div class="trace-row"><span>${esc(value)}</span></div>`).join('');
    const assays = (program.required_assays || []).map(row => `<div class="trace-row"><strong>${esc(row.assay_id || 'required assay')}</strong><span>${esc(row.objective || '')}</span></div>`).join('');
    const warnings = (program.warning_codes || []).map(value => `<code class="reason-code">${esc(value)}</code>`).join('');
    const requirements = Object.entries(program.cofactor_and_carrier_ledger?.requirements || {})
      .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(' / ') : value}`).join(' · ');
    const regenerations = Object.entries(program.cofactor_and_carrier_ledger?.regenerations || {})
      .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(' / ') : value}`).join(' · ');
    return `<article class="program-detail"><header><p class="detail-kind">候选替代模块 · ${esc(program.status || 'proposal_only')}</p><h3 class="detail-title">${equivalent} 步化学路线 → 1 个酶 Program</h3></header>
      <div class="notice program-warning-notice"><strong>待实验，尚未准入</strong><span>这是跨越精确边界的可证伪提案，不是文献已证实路线，也不会继承下方化学步骤的证明。</span>${warnings ? `<div class="reason-code-list">${warnings}</div>` : ''}</div>
      <section class="program-stat-grid"><div><strong>${equivalent}→1</strong><span>物理步骤压缩</span></div><div><strong>+${Number(program.net_step_savings || 0)}</strong><span>净节省步骤</span></div><div><strong>${esc(program.validation_status || 'experiment_required')}</strong><span>验证状态</span></div></section>
      <section class="detail-section"><h3>精确 Program 边界</h3><div class="program-boundary-grid">${inputs}${outputs}</div><div class="kv"><span class="k">能力</span><span class="v"><code>${esc(program.source_capability_id || '未记录')}</code></span></div></section>
      <section class="detail-section"><h3>候选酶与选择性</h3><div class="reason-code-list">${enzymeIds || '<span class="empty">候选酶待筛选</span>'}</div><div class="trace-list">${enzymeClasses}</div><div class="kv"><span class="k">选择性目标</span><span class="v">${esc((program.selectivity_constraints || []).join(' · ') || '待定义')}</span></div></section>
      <section class="detail-section"><h3>辅因子闭环</h3><div class="kv"><span class="k">需求</span><span class="v">${esc(requirements || '未记录')}</span></div><div class="kv"><span class="k">再生筛选</span><span class="v">${esc(regenerations || '未记录')}</span></div></section>
      <section class="detail-section program-fallback-section"><h3>canonical 化学基线 <span class="section-count">${equivalent} 步完整保留</span></h3><p class="v">候选酶模块不会覆盖这些步骤；展开后可在画布中作双路径对照。</p><button class="detail-action program-detail-toggle" type="button" data-program-toggle="${esc(program.program_id)}" aria-expanded="${programIsExpanded(program.program_id)}">${programIsExpanded(program.program_id) ? '收起画布中的化学基线' : `在画布中展开 ${equivalent} 步化学基线`}</button><details class="program-fallback-details"><summary>查看 ${equivalent} 步条件与证据入口</summary><div class="program-fallback-list">${fallbackRows}</div></details></section>
      <section class="detail-section"><h3>可证伪验证任务</h3><div class="trace-list">${assays || '<div class="empty">尚无验证任务</div>'}</div></section>
      <section class="detail-section"><h3>先例边界</h3><div class="kv"><span class="k">依据</span><span class="v">${program.analogy_only ? '类似底物先例；非当前底物精确证据' : '已绑定精确底物证据'}</span></div><div class="trace-list">${(program.precedent_refs || []).map(ref => `<div class="trace-row"><code>${esc(ref)}</code></div>`).join('') || '<div class="empty">未记录先例</div>'}</div></section>
      <div class="notice"><strong>权威边界</strong><span>Program 仅在专项验证通过后才可能进入影子优化器；当前 canonical 路线、闭合状态与证明均保持不变。</span></div></article>`;
  }

  function innovationSectionHtml(step) {
    const options = Array.isArray(step.route_innovations) ? step.route_innovations : [];
    if (!options.length) return '';
    const rows = options.map(option => {
      const kind = String(option.kind || 'route_innovation');
      if (kind === 'biocatalytic_superstep' || kind === 'biocatalytic_step') {
        const enzyme = option.enzyme || {};
        const enzymeLabels = [...(enzyme.classes || []), ...(enzyme.ec_numbers || [])].join(' · ');
        const equivalent = Number(option.chemical_step_equivalent_count || 1);
        const savings = Number(option.step_savings || Math.max(0, equivalent - 1));
        const title = kind === 'biocatalytic_superstep' ? `酶催化超级步骤 · 1 步替代 ${equivalent} 个化学步骤` : '酶催化步骤';
        return `<div class="trace-row route-innovation" data-innovation-kind="${esc(kind)}"><strong>${esc(title)}</strong><span>净节省 ${esc(savings)} 步 · ${esc(enzymeLabels || '酶类别待筛选')}</span><span>选择性目标：${esc(option.selectivity_objective || '待定义')}</span><span>底物范围依据：${esc(option.substrate_scope_basis || '未验证')}</span><code>${esc(option.validation_status || 'proposed')}</code></div>`;
      }
      const anchor = option.anchor || {};
      const anchorRefs = [...(anchor.source_refs || []), ...(anchor.source_binding_ids || []), ...(anchor.edge_ids || [])];
      const checks = Array.isArray(option.falsifiable_checks) ? option.falsifiable_checks.join(' · ') : '';
      return `<div class="trace-row route-innovation" data-innovation-kind="mechanism_extrapolation"><strong>机理外推 · 文献锚点后一跳</strong><span>${esc(option.mechanistic_rationale || '机理说明待补')}</span><span>锚点：${esc(anchorRefs.join(' · ') || '未绑定')}</span><span>可证伪检查：${esc(checks || '未记录')}</span><code>${esc(option.evidence_grade || 'low_mechanistic_hypothesis')}</code></div>`;
    }).join('');
    const gate = step.innovation_proof_gate || {};
    const warning = gate.required === true && gate.accepted !== true
      ? '<div class="notice"><strong>尚未闭合</strong><span>酶标签、EC 预测或机理合理性不等于反应验证；需要绑定当前主机可重放的专项验证。</span></div>'
      : '';
    return `<section class="detail-section route-innovation-section"><h3>路线创新与替代</h3>${warning}<div class="trace-list">${rows}</div></section>`;
  }

  function reactionOverview(step) {
    const trust = step.trust_vector || {};
    const bindingSet = step.edge_evidence_binding_set || trust.edge_evidence_binding_set || {};
    const trustedBindings = (bindingSet.bindings || []).filter(row => row.trusted === true);
    const trustedSources = trustedBindings.map(row => `<div class="trace-row"><strong>${esc(row.independent_source_group || row.source_ref || 'trusted source')}</strong><span>${esc(row.binding_id || '')}</span></div>`).join('');
    const citations = (step.source_refs || []).map(ref => `<div class="trace-row">${esc(basename(ref))}</div>`).join('');
    const sourceCount = Number(bindingSet.independent_trusted_source_group_count || 0);
    const corroborated = bindingSet.corroborated === true;
    const conditionRows = normalizedConditionRows(step);
    const coreConditions = conditionRows.filter(row => CORE_CONDITION_KEYS.has(row.key));
    const operationConditions = conditionRows.filter(row => !CORE_CONDITION_KEYS.has(row.key));
    const conditions = [
      conditionGroupHtml('核心反应条件', coreConditions, { open: true }),
      conditionGroupHtml('加料、后处理与纯化', operationConditions)
    ].join('');
    const procedures = Array.isArray(step.procedure_records) ? step.procedure_records : [];
    const procedureRows = procedures.map(row => {
      const fragment = row.source_fragment || {};
      const locations = Array.isArray(row.location_refs) ? row.location_refs.join(' · ') : '';
      const missing = row.condition_completeness?.missing_required_groups || [];
      return `<div class="trace-row"><strong>${esc(row.procedure_status || 'source procedure')}</strong><span>${esc(locations || row.source_ref || '')}</span><code>${esc(fragment.procedure_text_sha256 || '')}</code>${missing.length ? `<span>缺失：${esc(missing.join('、'))}</span>` : ''}</div>`;
    }).join('');
    const sourceObservations = Array.isArray(step.source_observation_records) ? step.source_observation_records : [];
    const observationRows = sourceObservations.map((row, index) => {
      const locations = Array.isArray(row.location_refs) ? row.location_refs.join(' · ') : '';
      const excerpt = String(row.procedure_excerpt || '').trim();
      const source = row.source_ref || '来源过程观察';
      return `<details class="source-procedure-observation" ${index === 0 ? 'open' : ''}><summary><span>${esc(source)}</span><strong>${esc(locations || `过程 ${index + 1}`)}</strong></summary><div class="source-procedure-body">${excerpt ? `<p class="procedure-excerpt">${esc(excerpt)}</p>` : '<div class="empty">未保存过程摘录</div>'}<div class="procedure-digests"><code>${esc(row.source_artifact_sha256 || '')}</code><code>${esc(row.source_pdf_sha256 || '')}</code></div></div></details>`;
    }).join('');
    const missingGroups = Array.isArray(step.condition_missing_required_groups) ? step.condition_missing_required_groups : [];
    const validationFindings = Array.isArray(step.validation_findings) ? step.validation_findings : [];
    const rejectionReasons = Array.isArray(step.rejection_reasons) ? step.rejection_reasons : [];
    const validationRows = validationFindings.map(row => {
      const audit = row.evidence?.audit || {};
      const gains = Object.entries(audit.unexplained_element_gains || {})
        .map(([element, count]) => `${element}+${count}`).join('、');
      const detail = gains ? `未解释原子增加：${gains}` : (row.message || '验证未通过');
      return `<div class="trace-row validation-finding" data-severity="${esc(row.severity || 'warning')}"><strong>${esc(row.finding_code || 'validation_finding')}</strong><span>${esc(detail)}</span>${row.required_action ? `<span>下一步：${esc(row.required_action)}</span>` : ''}</div>`;
    }).join('');
    const reasonRows = rejectionReasons.map(reason => `<code class="reason-code">${esc(reason)}</code>`).join('');
    const lifecycleSection = lifecycleFactsHtml(step.inactive_facts);
    return `<article><header><p class="detail-kind">${esc(routeStepDisplayLabel(step))} · ${esc(tierLabel(tierOfStep(step)))}</p><h3 class="detail-title">${esc(step.reaction_class || step.label || step.step_id)}</h3></header>
      ${state.activeReplacement ? `<div class="notice replacement-preview-notice"><strong>完整替换路线预览</strong><span>该分支已由后端 AND/OR 对 connectivity、stock 与 reaction proof 整路重验；预览不等于父路线证明。</span><button class="detail-action" type="button" data-replacement-reset>恢复原路线</button></div>` : ''}
      <section class="detail-section"><h3>来源分解</h3><div class="kv"><span class="k">方案生产者</span><span class="v">${esc(step.producer_label || '来源未标记')}</span></div><div class="kv"><span class="k">证据载体</span><span class="v">${esc(step.evidence_label || '无精确证据')}</span></div><div class="kv"><span class="k">主机验证</span><span class="v">${esc(tierLabel(tierOfStep(step)))}</span></div></section>
      ${innovationSectionHtml(step)}
      <section class="detail-section"><h3>反应连接</h3><p class="v">${esc(nodeNames(step.main_from_node_ids || step.from_node_ids))} → ${esc(nodeNames(step.to_node_ids))}</p>${(step.auxiliary_from_node_ids || []).length ? `<div class="kv"><span class="k">辅助试剂/小分子</span><span class="v">${esc(nodeNames(step.auxiliary_from_node_ids))}</span></div>` : ''}</section>
      <section class="detail-section condition-section"><h3>反应条件 <span class="section-count">${conditionRows.length ? `${conditionRows.length} 个字段${procedures.length ? ` · ${procedures.length} 组来源方案` : ''}` : '待取证'}</span></h3><div class="condition-list">${conditions || conditionResolutionHtml(step)}</div></section>
      ${modelConditionPredictionsHtml(step)}
      <section class="detail-section"><h3>来源过程 <span class="section-count">${sourceObservations.length + procedures.length} 条</span></h3><div class="kv"><span class="k">哈希绑定过程</span><span class="v">${procedures.length}</span></div><div class="kv"><span class="k">来源观察</span><span class="v">${sourceObservations.length}</span></div><div class="kv"><span class="k">条件缺失组</span><span class="v">${esc(missingGroups.join('、') || '无')}</span></div><div class="source-observation-list">${observationRows}</div><div class="trace-list">${procedureRows || (!observationRows ? conditionResolutionHtml(step) : '')}</div></section>
      ${(validationRows || reasonRows) ? `<section class="detail-section evidence-gap-section"><h3>证据缺口与补证动作</h3><div class="trace-list">${validationRows || '<div class="empty">尚无结构化验证发现</div>'}</div>${reasonRows ? `<div class="reason-code-list">${reasonRows}</div>` : ''}</section>` : ''}
      ${lifecycleSection}
      <section class="detail-section"><h3>科学 Proof vector</h3>${proofVectorHtml(step.proof_vector)}</section>
      <section class="detail-section"><h3>Trust vector</h3><div class="trust-grid">${['identity','connectivity','source_independence','stock','conditions','forward_feasibility'].map(key => `<div class="trust-cell ${tierClass(tierOfStep(step))}" style="--trust-value:${clamp(Number(trust[key] || 0), 0, 1)}"><strong>${esc(key)}</strong><span>${Number(trust[key] || 0).toFixed(2)}</span></div>`).join('')}</div></section>
      <section class="detail-section"><h3>逐边可信绑定</h3><div class="kv"><span class="k">可信独立来源</span><span class="v">${esc(sourceCount)}</span></div><div class="kv"><span class="k">多信源佐证</span><span class="v">${corroborated ? '是' : '否'}</span></div><div class="trace-list">${trustedSources || '<div class="empty">尚无可信精确绑定；引用不会自动升级证明。</div>'}</div></section>
      <section class="detail-section"><h3>普通引用</h3><div class="trace-list">${citations || '<div class="empty">来源未记录</div>'}</div></section></article>`;
  }
  function moleculeOverview(node) {
    const lifecycleSection = lifecycleFactsHtml(node.inactive_facts);
    return `<article><header><p class="detail-kind">分子节点</p><h3 class="detail-title">${esc(node.label || node.node_id || '未命名分子')}</h3></header>
      ${node.structure_svg ? `<div class="mol-structure">${node.structure_svg}</div>` : ''}
      <section class="detail-section"><div class="kv"><span class="k">分子式</span><span class="v">${esc(node.formula || '未记录')}</span></div>
      <div class="kv"><span class="k">Canonical</span><span class="v"><code>${esc(node.canonical_isomeric_smiles || node.smiles || '未记录')}</code></span></div>
      <div class="kv"><span class="k">角色</span><span class="v">${esc(node.role || 'intermediate')}</span></div></section>${lifecycleSection}</article>`;
  }
  function branchOverview(branch) {
    const lane = laneByBranch.get(branch.branch_id) || {};
    const executionLabel = lane.process_ready
      ? '工艺候选可执行'
      : lane.route_state_label || lane.completion_label || (branch.solved && branch.executable ? '已验证' : '探索建议');
    const lifecycleSection = lifecycleFactsHtml(lane.inactive_facts);
    return `<article><header><p class="detail-kind">路线分支 · ${esc(tierLabel(lane.proof_tier))}</p><h3 class="detail-title">${esc(branch.title || branch.branch_id)}</h3></header>
      <div class="notice">${esc(branch.summary || branch.recommendation || '没有路线摘要。')}</div>
      <section class="detail-section"><div class="kv"><span class="k">实际执行步骤</span><span class="v">${esc(lane.physical_step_count ?? (lane.step_ids || []).length)}</span></div><div class="kv"><span class="k">化学等效步骤</span><span class="v">${esc(lane.chemical_step_equivalent_count ?? (lane.step_ids || []).length)}</span></div><div class="kv"><span class="k">酶法净省步骤</span><span class="v">${esc(lane.net_step_savings || 0)}</span></div><div class="kv"><span class="k">机理外推</span><span class="v">${esc(lane.mechanism_extrapolation_count || 0)} 个一跳假设</span></div><div class="kv"><span class="k">DAG</span><span class="v">${lane.acyclic === false ? '检测到环路' : '无环'}</span></div><div class="kv"><span class="k">路线状态</span><span class="v">${esc(executionLabel)}</span></div><div class="kv"><span class="k">条件状态</span><span class="v">${esc(lane.condition_label || '条件状态未知')}</span></div><div class="kv"><span class="k">已达档位</span><span class="v">${esc((lane.achieved_profiles || []).join(' · ') || 'unresolved')}</span></div><div class="kv"><span class="k">完整合成声明</span><span class="v">${lane.full_synthesis_claim ? '是' : '否；仍是骨架或未闭合路线'}</span></div></section>
      ${lifecycleSection}
      <section class="detail-section"><h3>路线 Proof vector</h3>${proofVectorHtml(lane.proof_vector)}</section></article>`;
  }
  function lifecycleFactsHtml(values) {
    const facts = Array.isArray(values) ? values : [];
    if (!facts.length) return '';
    const rows = facts.map(row => {
      const status = row.status === 'revoked' ? '已撤销' : row.status === 'expired' ? '已过期' : '已失效';
      const reasons = Array.isArray(row.reason_codes) ? row.reason_codes.join('、') : '';
      return `<div class="trace-row"><strong>${esc(status)} · ${esc(row.subject_kind || 'fact')}</strong><span>${esc(row.subject_id || '')}</span><span>${esc(row.effective_at || '')}</span>${reasons ? `<span>原因：${esc(reasons)}</span>` : ''}<code>${esc(row.lifecycle_event_id || '')}</code></div>`;
    }).join('');
    return `<section class="detail-section lifecycle-impact"><h3>失效事实</h3><div class="notice"><strong>路线已按当前权威降级</strong><span>原事实仍保留用于审计，但不再授予 reaction、source、condition 或 stock 权威。</span></div><div class="trace-list">${rows}</div></section>`;
  }
  function proofVectorHtml(value) {
    const vector = value && typeof value === 'object' ? value : {};
    const axes = ['identity', 'reaction', 'conditions', 'sources', 'stock', 'process'];
    if (vector.schema_version !== 'retrosynthesis_proof_vector.v1') {
      return '<div class="empty">当前投影没有规范 proof vector；不会从颜色或总数反推证明。</div>';
    }
    return `<div class="proof-vector-grid">${axes.map(axis => {
      const raw = String(vector[axis] || 'unknown');
      const label = PROOF_VALUE_LABEL[raw] || raw.replaceAll('_', ' ');
      const state = ['missing', 'none', 'unknown', 'blocked', 'incomplete', 'untested', 'invalidated'].includes(raw)
        ? 'open' : raw === 'conflicted' ? 'conflict' : 'closed';
      return `<div class="proof-axis" data-state="${esc(state)}"><span>${esc(PROOF_AXIS_LABEL[axis] || axis)}</span><strong>${esc(label)}</strong><code>${esc(raw)}</code></div>`;
    }).join('')}</div>`;
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
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
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
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
    state.selectedGraphNodeId = '';
    state.selectedInstanceId = '';
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
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
    document.body.classList.toggle('ledger-open', state.ledgerOpen);
    document.documentElement.style.setProperty('--nav-width', `${state.navWidth}px`);
    document.documentElement.style.setProperty('--inspector-width', `${state.inspectorWidth}px`);
    element('themeToggle').setAttribute('aria-pressed', String(state.theme === 'dark'));
    element('layoutPreset').value = state.layoutPreset;
    element('orientationSelect').value = state.orientation;
    element('routeDirectionSelect').value = state.routeDirection;
    element('auxiliarySelect').value = state.showAuxiliary ? 'shown' : 'collapsed';
    element('densitySelect').value = state.density;
    element('edgeStyleSelect').value = state.edgeStyle;
    element('labelModeSelect').value = state.labelMode;
    element('navResizeHandle')?.setAttribute('aria-valuenow', String(state.navWidth));
    element('inspectorResizeHandle')?.setAttribute('aria-valuenow', String(state.inspectorWidth));
    const ledgerPanel = element('closureStatusPanel');
    const ledgerToggle = element('ledgerToggle');
    ledgerPanel.hidden = !state.ledgerOpen;
    ledgerPanel.toggleAttribute('inert', !state.ledgerOpen);
    ledgerToggle.setAttribute('aria-expanded', String(state.ledgerOpen));
    const ledgerToggleLabel = ledgerToggle.querySelector('span');
    if (ledgerToggleLabel) ledgerToggleLabel.textContent = state.ledgerOpen ? '收起证明' : '闭合证明';
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
    state.selectedProgramId = '';
    state.selectedMechanismId = '';
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
      const target = event.target.closest('button, [data-program-toggle], [data-program-fallback-step], [data-program-id], [data-mechanism-id], [data-graph-node-id], [data-lane-branch-id], .graph-minimap');
      if (!target) return;
      if (target.dataset.replacementPreview !== undefined) { previewReplacement(target); return; }
      if (target.dataset.replacementReset !== undefined) { restoreReplacementPreview(); return; }
      if (target.dataset.programToggle) {
        toggleProgramFallback(target.dataset.programToggle, { focus: target.tagName.toLowerCase() !== 'button' });
        return;
      }
      if (target.dataset.programFallbackStep) {
        const owner = programOverlays.get(target.dataset.programOwnerId)
          || [...programOverlays.values()].find(row =>
            (row.replaced_step_ids || []).includes(target.dataset.programFallbackStep));
        if (owner && !programIsExpanded(owner.program_id)) {
          state.expandedProgramIds.add(owner.program_id);
          graphModelCache.clear();
          renderGraph();
        }
        const graphNode = [...graphNodes.values()].find(row => row.node_type === 'reaction'
          && row.reaction_step_id === target.dataset.programFallbackStep);
        if (graphNode) selectGraphNode(graphNode.graph_node_id, {
          branchId: target.dataset.branchId || '',
          instanceId: `${target.dataset.branchId || ''}::${graphNode.graph_node_id}`
        });
        return;
      }
      if (target.dataset.programId) { selectProgram(target.dataset.programId); return; }
      if (target.dataset.mechanismId) { selectMechanism(target.dataset.mechanismId); return; }
      if (target.dataset.graphNodeId) { selectGraphNode(target.dataset.graphNodeId, { branchId: target.dataset.branchId || '', instanceId: target.dataset.instanceId || '' }); return; }
      if (target.dataset.laneBranchId) { openBranchInFocus(target.dataset.laneBranchId, { focusGraph: true }); return; }
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
      if (target.id === 'ledgerToggle' || target.id === 'closureDismiss') {
        state.ledgerOpen = target.id === 'closureDismiss' ? false : !state.ledgerOpen;
        applyPersistentChromeState(); persistState();
        return;
      }
      if (target.id === 'pdfExport') {
        const pdfPath = location.pathname.endsWith('/workbench.html')
          ? location.pathname.replace(/\/workbench\.html$/, '/workbench.pdf')
          : '';
        if (pdfPath) location.assign(pdfPath);
        else window.print();
        return;
      }
      if (target.dataset.detailTab) { state.detailTab = target.dataset.detailTab; renderDetail(); persistState(); return; }
      if (target.id === 'themeToggle') { state.theme = state.theme === 'dark' ? 'light' : 'dark'; applyPersistentChromeState(); persistState(); return; }
      if (target.id === 'navToggle') { state.navOpen = !state.navOpen; if (isMobileDrawerLayout() && state.navOpen) state.inspectorOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.navOpen ? 'nav' : 'graph'); persistState(); requestAnimationFrame(() => fitGraph({ readable: preferReadableFocus() })); return; }
      if (target.id === 'inspectorToggle') { state.inspectorOpen = !state.inspectorOpen; if (isMobileDrawerLayout() && state.inspectorOpen) state.navOpen = false; applyPersistentChromeState(); updateMobileNavigation(state.inspectorOpen ? 'inspector' : 'graph'); persistState(); if (!isInspectorOverlayLayout()) requestAnimationFrame(() => fitGraph({ readable: preferReadableFocus() })); return; }
      if (target.dataset.closePanel === 'inspector') { state.inspectorOpen = false; applyPersistentChromeState(); updateMobileNavigation('graph'); focusPanelReturn('inspectorToggle'); persistState(); return; }
      if (target.dataset.closeMobilePanels !== undefined) { closeMobilePanels({ restoreFocus: true }); return; }
      if (target.dataset.mobilePanel) { openMobilePanel(target.dataset.mobilePanel); return; }
      if (target.classList.contains('graph-minimap')) recenterFromMinimap(event);
    });
    document.addEventListener('change', event => {
      const target = event.target;
      if (target.id === 'layoutPreset') state.layoutPreset = target.value;
      else if (target.id === 'orientationSelect') state.orientation = target.value;
      else if (target.id === 'routeDirectionSelect') state.routeDirection = target.value;
      else if (target.id === 'auxiliarySelect') state.showAuxiliary = target.value === 'shown';
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
    const handleViewportResize = debounce(resizeGraphViewport, 90);
    window.addEventListener('resize', handleViewportResize);
    if ('ResizeObserver' in window) {
      viewportResizeObserver?.disconnect();
      viewportResizeObserver = new ResizeObserver(handleViewportResize);
      viewportResizeObserver.observe(element('graphViewport'));
    }
  }

  function commitPendingPanFrame(frameTime = performance.now()) {
    panAnimationFrame = 0;
    if (!panSession) return;
    const frameStartedAt = performance.now();
    state.panX = panSession.panX + panSession.latestX - panSession.x;
    state.panY = panSession.panY + panSession.latestY - panSession.y;
    const delay = Math.max(0, frameTime - (panSession.frameRequestedAt || frameTime));
    renderPerformance.cameraFrames += 1;
    renderPerformance.maximumFrameDelayMs = Math.max(renderPerformance.maximumFrameDelayMs, delay);
    if (delay > 24) renderPerformance.droppedFrames += 1;
    applyViewportTransform({ updateMinimap: false });
    const frameDuration = performance.now() - frameStartedAt;
    renderPerformance.totalCameraFrameMs += frameDuration;
    renderPerformance.maximumCameraFrameMs = Math.max(
      renderPerformance.maximumCameraFrameMs,
      frameDuration
    );
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
        if (panAnimationFrame) cancelAnimationFrame(panAnimationFrame);
        panAnimationFrame = 0;
        const finalX = cancelled ? session.x : Number(event.clientX ?? session.latestX);
        const finalY = cancelled ? session.y : Number(event.clientY ?? session.latestY);
        state.panX = session.panX + finalX - session.x;
        state.panY = session.panY + finalY - session.y;
        applyViewportTransform({ forceCull: true });
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
        latestX: event.clientX,
        latestY: event.clientY,
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
        state.cameraMode = 'manual';
        try {
          viewport.setPointerCapture(event.pointerId);
          panSession.captured = true;
        } catch (_) { /* global listeners still finish the active drag */ }
        viewport.classList.remove('is-pan-ready');
        viewport.classList.add('is-panning');
      }
      event.preventDefault();
      panSession.latestX = event.clientX;
      panSession.latestY = event.clientY;
      if (!panAnimationFrame) {
        panSession.frameRequestedAt = performance.now();
        panAnimationFrame = requestAnimationFrame(commitPendingPanFrame);
      }
    }, { capture: true, passive: false });
    window.addEventListener('pointerup', event => finishPan(event), true);
    window.addEventListener('pointercancel', event => finishPan(event, { cancelled: true }), true);
    viewport.addEventListener('lostpointercapture', event => {
      if (event.target === viewport) finishPan(event, { cancelled: true });
    });
    window.addEventListener('blur', () => {
      if (!panSession) return;
      state.panX = panSession.panX;
      state.panY = panSession.panY;
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
    const overviewLane = event.target.closest?.('[data-lane-branch-id]');
    if (overviewLane && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      openBranchInFocus(overviewLane.dataset.laneBranchId, { focusGraph: true });
      return;
    }
    const graphNode = event.target.closest?.('[data-graph-node-id]');
    if (graphNode && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); selectGraphNode(graphNode.dataset.graphNodeId, { focus: true, branchId: graphNode.dataset.branchId || '', instanceId: graphNode.dataset.instanceId || '' }); return;
    }
    const programToggle = event.target.closest?.('[data-program-toggle]');
    if (programToggle && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      toggleProgramFallback(programToggle.dataset.programToggle, { focus: true });
      return;
    }
    const fallbackStep = event.target.closest?.('[data-program-fallback-step]');
    if (fallbackStep && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      const graphNode = [...graphNodes.values()].find(row => row.node_type === 'reaction'
        && row.reaction_step_id === fallbackStep.dataset.programFallbackStep);
      if (graphNode) selectGraphNode(graphNode.graph_node_id, {
        focus: true,
        branchId: fallbackStep.dataset.branchId || '',
        instanceId: `${fallbackStep.dataset.branchId || ''}::${graphNode.graph_node_id}`
      });
      return;
    }
    const programNode = event.target.closest?.('[data-program-id]');
    if (programNode && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); selectProgram(programNode.dataset.programId, { focus: true }); return;
    }
    const mechanismNode = event.target.closest?.('[data-mechanism-id]');
    if (mechanismNode && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); selectMechanism(mechanismNode.dataset.mechanismId, { focus: true }); return;
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
      handle.addEventListener('pointerup', () => { resizeSession = null; persistState(); requestAnimationFrame(() => fitGraph({ readable: preferReadableFocus() })); });
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
    state.cameraMode = 'manual';
    applyViewportTransform();
  }
  function debounce(fn, delay) {
    let timer = 0;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
  }

  function maybeRunBrowserSelfTest() {
    if (new URLSearchParams(location.search).get('route_ui_selftest') !== '1') return;
    const output = document.createElement('pre');
    output.id = 'routeUiSelfTest';
    output.hidden = true;
    document.body.appendChild(output);
    const checks = {};
    const finish = error => {
      const passed = !error && Object.values(checks).every(Boolean);
      output.textContent = JSON.stringify({
        status: passed ? 'passed' : 'failed',
        checks,
        error: error ? String(error.stack || error) : '',
        performance: window.__AUTOPLANNER_ROUTE_PERF__.snapshot()
      });
      document.documentElement.dataset.routeUiSelfTest = passed ? 'passed' : 'failed';
    };
    try {
      state.mode = 'clusters';
      renderGraph();
      const viewport = element('graphViewport');
      state.ledgerOpen = false;
      applyPersistentChromeState();
      const canvasHeightBeforeLedger = viewport.getBoundingClientRect().height;
      checks.proofDrawerDefaultClosed = element('closureStatusPanel').hidden
        && element('ledgerToggle').getAttribute('aria-expanded') === 'false';
      element('ledgerToggle').click();
      checks.proofDrawerOverlay = !element('closureStatusPanel').hidden
        && element('ledgerToggle').getAttribute('aria-expanded') === 'true'
        && Math.abs(viewport.getBoundingClientRect().height - canvasHeightBeforeLedger) < 1;
      element('closureDismiss').click();
      checks.proofDrawerDismiss = element('closureStatusPanel').hidden
        && element('ledgerToggle').getAttribute('aria-expanded') === 'false';
      const initialPan = { x: state.panX, y: state.panY };
      viewport.dispatchEvent(new PointerEvent('pointerdown', {
        bubbles: true, button: 0, buttons: 1, pointerId: 901,
        pointerType: 'mouse', isPrimary: true, clientX: 120, clientY: 120
      }));
      window.dispatchEvent(new PointerEvent('pointermove', {
        bubbles: true, button: 0, buttons: 1, pointerId: 901,
        pointerType: 'mouse', isPrimary: true, clientX: 178, clientY: 151
      }));
      // Synthetic pointer dispatch is synchronous.  Finishing in this frame keeps
      // --dump-dom regression runs deterministic while pointerup still exercises
      // the production camera-frame path and final pointerup commit.
      try {
          if (panAnimationFrame) cancelAnimationFrame(panAnimationFrame);
          commitPendingPanFrame(performance.now());
          window.dispatchEvent(new PointerEvent('pointerup', {
            bubbles: true, button: 0, buttons: 0, pointerId: 901,
            pointerType: 'mouse', isPrimary: true, clientX: 178, clientY: 151
          }));
          checks.drag = Math.abs(state.panX - initialPan.x - 58) < .01
            && Math.abs(state.panY - initialPan.y - 31) < .01;
          checks.singleWorldTransform = document.querySelector('.graph-svg').style.transform === ''
            && document.querySelector('.graph-world').getAttribute('transform')
              === `translate(${state.panX} ${state.panY}) scale(${state.zoom})`;
          const rect = viewport.getBoundingClientRect();
          const anchorX = Math.max(1, rect.width * .43);
          const anchorY = Math.max(1, rect.height * .47);
          const beforeWorld = {
            x: (anchorX - state.panX) / state.zoom,
            y: (anchorY - state.panY) / state.zoom
          };
          zoomGraph(1.2, rect.left + anchorX, rect.top + anchorY);
          checks.zoomAnchor = Math.abs((anchorX - state.panX) / state.zoom - beforeWorld.x) < 1e-6
            && Math.abs((anchorY - state.panY) / state.zoom - beforeWorld.y) < 1e-6;
          fitGraph();
          checks.fit = Number.isFinite(state.zoom) && state.zoom > 0
            && Number.isFinite(state.panX) && Number.isFinite(state.panY);
          const reactionNodes = [...document.querySelectorAll(
            '.graph-node[data-node-type="reaction"]'
          )];
          const firstNode = reactionNodes.find(row => {
            const step = steps.get(row.dataset.routeStep || '');
            return (step?.source_observation_records || []).length > 0;
          }) || reactionNodes[0] || document.querySelector('.graph-node[data-graph-node-id]');
          const firstHitTarget = firstNode?.querySelector('.reaction-hit-target') || firstNode;
          firstHitTarget?.dispatchEvent(new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window
          }));
          checks.selection = !firstNode || Boolean(document.querySelector('.graph-node.is-selected'));
          checks.reactionHitTarget = !firstNode || firstNode.dataset.nodeType !== 'reaction'
            || Boolean(firstNode.querySelector('.reaction-hit-target'));
          checks.reactionInspector = !firstNode || firstNode.dataset.nodeType !== 'reaction'
            || (state.inspectorOpen && state.detailTab === 'step' && Boolean(state.selectedStepId)
              && Boolean(element('detail').querySelector('.condition-section')));
          const selectedStep = steps.get(state.selectedStepId || '');
          const conditionRowCount = selectedStep ? normalizedConditionRows(selectedStep).length : 0;
          const sourceObservationCount = (selectedStep?.source_observation_records || []).length;
          checks.fullConditionGroups = conditionRowCount === 0
            || Boolean(element('detail').querySelector('.condition-group'));
      checks.sourceProcedure = sourceObservationCount === 0
        || (Boolean(element('detail').querySelector('.source-procedure-observation'))
          && Boolean(element('detail').querySelector('.procedure-excerpt')));
      state.mode = 'clusters';
      renderGraph();
      const renderedEdges = [...document.querySelectorAll('.graph-edge[data-edge-color]')];
      checks.edgeProducerColorConsistent = renderedEdges.every(edge => {
        const markerMatch = String(edge.getAttribute('marker-end') || '').match(/^url\(#([^)]+)\)$/);
        const markerPath = markerMatch
          ? document.getElementById(markerMatch[1])?.querySelector('path')
          : null;
        return edge.dataset.edgeColor === edge.style.getPropertyValue('--edge-color').trim()
          && Boolean(markerPath)
          && getComputedStyle(edge).stroke === getComputedStyle(markerPath).fill;
      });
      const renderedReactionNodes = [...document.querySelectorAll(
        '.graph-node--reaction[data-producer-color][data-route-step]'
      )];
      const reactionColorRows = renderedReactionNodes.map(node => {
        const surface = node.querySelector('.node-surface');
        const stripe = node.querySelector('.reaction-origin-stripe');
        const ports = [...node.querySelectorAll('.graph-node-port')];
        const stepEdges = renderedEdges.filter(edge => edge.dataset.reactionStepId === node.dataset.routeStep);
        const nodeStroke = surface ? getComputedStyle(surface).stroke : '';
        return {
          token: node.dataset.producerColor === node.style.getPropertyValue('--origin-color').trim(),
          surface: Boolean(stripe) && nodeStroke === getComputedStyle(stripe).fill,
          ports: ports.every(port => getComputedStyle(port).stroke === nodeStroke),
          edges: stepEdges.every(edge => getComputedStyle(edge).stroke === nodeStroke)
        };
      });
      checks.reactionProducerTokenConsistent = reactionColorRows.every(row => row.token);
      checks.reactionProducerSurfaceConsistent = reactionColorRows.every(row => row.surface);
      checks.reactionProducerPortsConsistent = reactionColorRows.every(row => row.ports);
      checks.reactionProducerEdgesConsistent = reactionColorRows.every(row => row.edges);
      const renderedPriorityBranches = renderModel.decorations
        .filter(row => isPriorityBranch(row.branchId))
        .map(row => row.branchId);
      const priorityMarkers = [...document.querySelectorAll(
        '.graph-lane-decoration.is-priority[data-lane-branch-id]'
      )].map(row => row.dataset.laneBranchId);
      checks.overviewPriorityMarkerExclusive = priorityMarkers.length === renderedPriorityBranches.length
        && priorityMarkers.length <= 1
        && priorityMarkers.every(branchId => isPriorityBranch(branchId));
      const overviewLane = document.querySelector('.graph-lane-open-control[data-lane-branch-id]');
      const overviewBranchId = overviewLane?.dataset.laneBranchId || '';
      overviewLane?.dispatchEvent(new MouseEvent('click', {
        bubbles: true,
        cancelable: true,
        view: window
      }));
      checks.overviewOpensFocusedBranch = !overviewBranchId
        || (state.mode === 'current' && state.selectedBranchId === overviewBranchId);
      // Return to the intentionally large overview before exercising the
      // culling assertion below; focused-branch mode is intentionally small.
      state.mode = 'clusters';
      renderGraph();
      const minimap = element('graphMinimap');
          minimap.hidden = false;
          const minimapRect = minimap.getBoundingClientRect();
          if (minimapRect.width && minimapRect.height) {
            recenterFromMinimap({
              clientX: minimapRect.left + minimapRect.width / 2,
              clientY: minimapRect.top + minimapRect.height / 2
            });
          }
          checks.minimap = Number.isFinite(state.panX) && Number.isFinite(state.panY);
          state.panX = -100000;
          state.panY = -100000;
          applyViewportTransform({ forceCull: true });
          checks.largeGraphCulling = renderPerformance.renderedObjects <= CULLING_OBJECT_THRESHOLD
            || renderPerformance.culledObjects > 0;
          fitGraph();
          finish();
      } catch (error) { finish(error); }
    } catch (error) { finish(error); }
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
      maybeRunBrowserSelfTest();
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
