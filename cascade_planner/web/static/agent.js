const $ = (id) => document.getElementById(id);

const samples = {
  atorvastatin: {
    name: "atorvastatin",
    hint: "statin, HMG-CoA reductase inhibitor, Paal-Knorr convergent route",
    smiles: "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    rounds: 5,
    scout: 2,
    visual: 1,
    chemenzy: 2,
    subgoal: 2,
  },
  bufotalin: {
    name: "bufotalin",
    hint: "bufadienolide steroid, C17 pyrone, visual literature chain",
    smiles: "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O",
    rounds: 7,
    scout: 3,
    visual: 3,
    chemenzy: 2,
    subgoal: 3,
  },
  paclitaxel: {
    name: "paclitaxel",
    hint: "taxane semisynthesis, baccatin III, 10-deacetylbaccatin III, C13 side-chain installation",
    smiles: "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    rounds: 7,
    scout: 3,
    visual: 2,
    chemenzy: 1,
    subgoal: 2,
  },
  ibuprofen: {
    name: "ibuprofen",
    hint: "arylpropionic acid NSAID",
    smiles: "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    rounds: 3,
    scout: 1,
    visual: 0,
    chemenzy: 1,
    subgoal: 1,
  },
};

const state = {
  currentJobId: "",
  pollTimer: null,
  loadedRoutePath: "",
  routeLoadingPath: "",
  routeFailedPath: "",
  routeFailedAt: 0,
  routeLoadToken: 0,
  routeHandshakeToken: "",
  routeHandshakeTimer: 0,
  controlsOpen: true,
  activityOpen: true,
  mobileView: "route",
};

const LAST_JOB_KEY = "autoplanner.agent.lastJobId";
const LAYOUT_KEY = "autoplanner.agent.layout.v2";
const MOBILE_BREAKPOINT = "(max-width: 900px)";
const ROUTE_READY_MESSAGE = "autoplanner.route_forest.ready.v1";
const ROUTE_HANDSHAKE_TIMEOUT_MS = 8000;

function isMobileLayout() {
  return window.matchMedia(MOBILE_BREAKPOINT).matches;
}

function readLayoutState() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "{}");
    if (typeof saved.controlsOpen === "boolean") state.controlsOpen = saved.controlsOpen;
    if (typeof saved.activityOpen === "boolean") state.activityOpen = saved.activityOpen;
    if (["route", "controls", "activity"].includes(saved.mobileView)) {
      state.mobileView = saved.mobileView;
    }
  } catch {
    // Invalid or unavailable storage falls back to the route-first defaults.
  }
}

function saveLayoutState() {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({
      controlsOpen: state.controlsOpen,
      activityOpen: state.activityOpen,
      mobileView: state.mobileView,
    }));
  } catch {
    // The workbench remains fully usable when storage is unavailable.
  }
}

function setPanelAvailability(panel, visible) {
  if (!panel) return;
  panel.setAttribute("aria-hidden", visible ? "false" : "true");
  panel.toggleAttribute("inert", !visible);
}

function applyLayoutState() {
  const mobile = isMobileLayout();
  document.body.classList.toggle("controls-collapsed", !state.controlsOpen);
  document.body.classList.toggle("activity-collapsed", !state.activityOpen);
  document.body.dataset.mobileView = state.mobileView;

  const controlsVisible = mobile ? state.mobileView === "controls" : state.controlsOpen;
  const activityVisible = mobile ? state.mobileView === "activity" : state.activityOpen;
  const routeVisible = !mobile || state.mobileView === "route";
  $("toggle-controls").setAttribute("aria-expanded", String(controlsVisible));
  $("toggle-activity").setAttribute("aria-expanded", String(activityVisible));
  setPanelAvailability($("controls-panel"), controlsVisible);
  setPanelAvailability($("activity-panel"), activityVisible);
  setPanelAvailability($("route-workspace"), routeVisible);

  document.querySelectorAll("[data-mobile-view]").forEach((button) => {
    if (!button.classList.contains("mobile-view-tab")) return;
    const selected = button.dataset.mobileView === state.mobileView;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
}

function setMobileView(view, { focusTab = false } = {}) {
  if (!["route", "controls", "activity"].includes(view)) return;
  state.mobileView = view;
  saveLayoutState();
  applyLayoutState();
  if (focusTab) $("mobile-view-" + view)?.focus();
}

function toggleControls() {
  if (isMobileLayout()) {
    setMobileView(state.mobileView === "controls" ? "route" : "controls");
    return;
  }
  state.controlsOpen = !state.controlsOpen;
  saveLayoutState();
  applyLayoutState();
}

function toggleActivity() {
  if (isMobileLayout()) {
    setMobileView(state.mobileView === "activity" ? "route" : "activity");
    return;
  }
  state.activityOpen = !state.activityOpen;
  saveLayoutState();
  applyLayoutState();
}

function focusRoute() {
  if (isMobileLayout()) setMobileView("route");
  const route = $("route-workspace");
  route.focus({ preventScroll: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  route.scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { ok: res.ok, text };
  }
  if (!res.ok) {
    throw new Error(data.error || data.text || res.statusText);
  }
  return data;
}

function normalizeResultPath(path) {
  return String(path || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

function resultFileUrl(path) {
  return `/api/result-file?path=${encodeURIComponent(normalizeResultPath(path))}`;
}

function artifactUrl(path) {
  return `/api/artifact?path=${encodeURIComponent(normalizeResultPath(path))}`;
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    $("server-status").textContent = data.ok ? "ready" : "error";
    $("server-status").className = `status-pill ${data.ok ? "good" : "bad"}`;
  } catch {
    $("server-status").textContent = "offline";
    $("server-status").className = "status-pill bad";
  }
}

function applySample(key) {
  const sample = samples[key];
  if (!sample) return;
  $("target-name").value = sample.name;
  $("family-hint").value = sample.hint;
  $("target-smiles").value = sample.smiles;
  $("max-rounds").value = String(sample.rounds);
  $("scout-calls").value = String(sample.scout);
  $("visual-calls").value = String(sample.visual);
  $("chemenzy-runs").value = String(sample.chemenzy);
  $("subgoal-runs").value = String(sample.subgoal);
}

function readPayload() {
  const tools = $("planner-tools").value;
  const timeout = Number($("timeout-s").value || 1800);
  const scoutCalls = Number($("scout-calls").value || 0);
  const chemenzyRuns = Number($("chemenzy-runs").value || 1);
  return {
    planner_backend: "codex_fullflow",
    planner_mode: "codex_fullflow",
    search_preset: "agentic_delivery",
    target_name: $("target-name").value.trim() || "target",
    target_smiles: $("target-smiles").value.trim(),
    family_hint: $("family-hint").value.trim(),
    run_prefix: "ui_agent_fullflow",
    max_rounds: Number($("max-rounds").value || 3),
    timeout_s: timeout,
    guided_chemenzy_timeout_s: Math.max(300, Math.floor(timeout / 2)),
    max_chem_enzy_runs: chemenzyRuns,
    max_guided_chemenzy_runs: chemenzyRuns,
    max_route_expansion_subgoal_runs: Number($("subgoal-runs").value || 1),
    max_scout_calls: scoutCalls,
    max_visual_calls: Number($("visual-calls").value || 0),
    max_codex_research_runs: scoutCalls > 0 ? 1 : 0,
    max_template_applications_per_round: 5,
    codex_action_planner: $("codex-action-planner").checked,
    codex_agent_team: $("codex-agent-team").checked,
    codex_agent_team_max_depth: Number($("codex-team-depth").value || 2),
    codex_agent_team_max_expansions: Number($("codex-team-expansions").value || 4),
    codex_agent_team_model: $("codex-team-model").value.trim(),
    exhaust_round_budget: $("exhaust-round-budget").checked,
    auto_local_pdf_discovery: $("auto-local-pdf").checked,
    model: $("model").value.trim() || "gpt-5.5",
    codex_action_planner_tools: tools || undefined,
    codex_worker_auth: "auto",
  };
}

function validateRunControls() {
  const controls = [...$("controls-panel").querySelectorAll("input, textarea, select")];
  const invalid = controls.find((control) => !control.checkValidity());
  if (!invalid) return true;
  if (isMobileLayout()) setMobileView("controls");
  invalid.reportValidity();
  invalid.focus();
  return false;
}

async function startRun() {
  if (!validateRunControls()) return;
  const payload = readPayload();
  $("start-run").disabled = true;
  $("cancel-run").disabled = true;
  setJobHeader({
    status: "queued",
    label: `Agent fullflow · ${payload.target_name}`,
    target_preview: shorten(payload.target_smiles, 82),
  });
  renderTimeline([]);
  renderArtifacts({});
  renderLog([]);
  clearRoute();
  try {
    const job = await api("/api/plan-jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJobId = job.job_id;
    rememberJob(job.job_id);
    $("cancel-run").disabled = false;
    await pollJob();
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollJob, 2500);
  } catch (err) {
    $("start-run").disabled = false;
    setJobHeader({ status: "failed", label: "启动失败", error: err.message });
  }
}

async function cancelRun() {
  if (!state.currentJobId) return;
  $("cancel-run").disabled = true;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    renderJob(job);
  } catch (err) {
    setJobHeader({ status: "failed", label: "取消失败", error: err.message });
  }
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}`);
    renderJob(job);
    if (isTerminalStatus(job.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("start-run").disabled = false;
      $("cancel-run").disabled = true;
    }
  } catch (err) {
    $("job-progress").textContent = err.message;
    if (String(err.message || "").includes("404")) {
      clearRememberedJob(state.currentJobId);
    }
  }
}

function rememberJob(jobId) {
  if (!jobId) return;
  try {
    localStorage.setItem(LAST_JOB_KEY, jobId);
  } catch {
    // Local storage may be disabled; the live page still works without it.
  }
}

function readRememberedJob() {
  try {
    return localStorage.getItem(LAST_JOB_KEY) || "";
  } catch {
    return "";
  }
}

function clearRememberedJob(jobId = "") {
  try {
    if (!jobId || localStorage.getItem(LAST_JOB_KEY) === jobId) {
      localStorage.removeItem(LAST_JOB_KEY);
    }
  } catch {
    // Ignore storage errors.
  }
}

function isTerminalStatus(status) {
  return ["complete", "failed", "cancelled"].includes(status);
}

function armPollingFor(job) {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  const terminal = isTerminalStatus(job.status);
  $("start-run").disabled = !terminal;
  $("cancel-run").disabled = terminal;
  if (!terminal) {
    state.pollTimer = setInterval(pollJob, 2500);
  }
}

async function restoreJobById(jobId) {
  if (!jobId) return false;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    state.currentJobId = job.job_id;
    rememberJob(job.job_id);
    renderJob(job);
    armPollingFor(job);
    return true;
  } catch {
    clearRememberedJob(jobId);
    return false;
  }
}

async function restoreActiveJob() {
  const remembered = readRememberedJob();
  if (await restoreJobById(remembered)) return true;
  try {
    const data = await api("/api/jobs");
    const jobs = data.jobs || [];
    const active = jobs.find((job) => !isTerminalStatus(job.status));
    if (!active) return false;
    return restoreJobById(active.job_id);
  } catch {
    return false;
  }
}

function setJobHeader(job) {
  const status = job.status || "idle";
  $("job-title").textContent = job.label || job.job_id || "未命名任务";
  $("job-subtitle").textContent = job.target_preview || job.target_smiles || job.error || "等待运行";
  $("job-status").textContent = status;
  $("job-progress").textContent = job.error || statusText(job);
}

function statusText(job) {
  if (job.queue_position && job.queue_position > 0) return `队列位置 ${job.queue_position}`;
  if (job.elapsed_s) return `${job.elapsed_s}s`;
  if (job.started_at) return "运行中";
  return "等待执行";
}

function renderJob(job) {
  setJobHeader(job);
  const steps = job.agent_steps || [];
  renderTimeline(steps);
  renderArtifacts(job);
  renderLog(job.log_tail || []);
  const summary = job.summary || {};
  $("route-state").textContent = summary.solved ? "solved" : (summary.status || job.status || "unknown");
  const counts = [];
  if (summary.routes != null) counts.push(`${summary.routes} 条分支`);
  if (summary.steps != null) counts.push(`${summary.steps} 步`);
  if (summary.time_s != null) counts.push(`${summary.time_s}s`);
  $("route-counts").textContent = counts.join(" · ") || "等待 route forest";
  const routePath = normalizeResultPath(job.route_forest_html);
  if (routePath && !routePath.endsWith(".html")) {
    showRouteMessage("error", "路线图路径无效", "后端返回的不是 route_forest.html，暂时不能嵌入主画布。", routePath);
  } else if (routePath && shouldLoadRoute(routePath)) {
    loadRoute(routePath);
  }
}

function renderTimeline(steps) {
  $("step-total").textContent = String(steps.length || 0);
  const last = steps[steps.length - 1];
  $("step-last").textContent = last
    ? `${stageLabel(last.stage)} · ${last.action_type || last.last_action?.action_type || "blackboard"}`
    : "暂无记录";
  if (!steps.length) {
    $("timeline").innerHTML = '<div class="empty">任务启动后会显示初始化、动作规划、工具执行、审计与最终汇总。</div>';
    return;
  }
  $("timeline").innerHTML = steps.map((step) => {
    const stage = stageLabel(step.stage);
    const tone = stepTone(step);
    const actions = step.last_planner?.action_types || [];
    const actionText = step.action_type || step.last_action?.action_type || actions.join(", ") || "blackboard update";
    const detail = step.detail || {};
    const meta = [
      step.round_index ? `round ${step.round_index}` : "",
      step.last_planner?.mode ? `planner: ${step.last_planner.mode}` : "",
      detail.validation_accepted === false ? "validation rejected" : "",
      detail.useful_artifact === true ? "useful artifact" : "",
      detail.accepted === false ? "action rejected" : "",
    ].filter(Boolean).join(" · ");
    const chips = countChips(step.counts || {});
    return `
      <article class="timeline-item ${tone}">
        <div class="step-index">${escapeHtml(step.step_index || "")}</div>
        <div class="timeline-main">
          <div class="timeline-title">
            <span>${escapeHtml(stage)}</span>
            <span class="mini-badge">${escapeHtml(actionText)}</span>
          </div>
          <div class="timeline-meta">${escapeHtml(meta || "黑板状态更新")}</div>
          ${chips}
        </div>
      </article>
    `;
  }).join("");
}

function countChips(counts) {
  const keys = [
    ["artifact_refs", "artifacts"],
    ["source_candidates", "sources"],
    ["exact_rows", "exact rows"],
    ["visual_chains", "visual"],
    ["retrosynthetic_proposals", "proposals"],
    ["route_objectives", "objectives"],
    ["route_failures", "failures"],
  ];
  const html = keys
    .filter(([key]) => Number(counts[key] || 0) > 0)
    .map(([key, label]) => `<span class="count-chip">${escapeHtml(label)} ${escapeHtml(counts[key])}</span>`)
    .join("");
  return html ? `<div class="count-strip">${html}</div>` : "";
}

function stageLabel(stage) {
  return {
    initialized: "初始化",
    action_batch: "动作规划",
    agent_action: "工具执行",
    auto_critic: "自动审计",
    round_complete: "轮次完成",
    finalized: "最终汇总",
  }[stage] || stage || "黑板更新";
}

function stepTone(step) {
  if (step.detail?.accepted === false || step.detail?.validation_accepted === false) return "bad";
  if (step.stage === "agent_action" && step.detail?.useful_artifact) return "good";
  if (step.stage === "round_complete") return "good";
  return "";
}

function renderArtifacts(job) {
  const links = [
    ["路线图", job.route_forest_html, "file"],
    ["路线 JSON", job.explored_route_forest, "json"],
    ["黑板", job.agent_blackboard, "json"],
    ["最终判定", job.final_verdict, "json"],
    ["Web 结果", job.output_json, "json"],
    ["请求", job.request_json, "json"],
  ].filter(([, path]) => path);
  $("artifact-links").innerHTML = links.length
    ? links.map(([label, path, kind]) => {
        const normalizedPath = normalizeResultPath(path);
        const href = kind === "file" ? resultFileUrl(normalizedPath) : artifactUrl(normalizedPath);
        return `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
      }).join("")
    : '<span class="mini-badge">暂无产物</span>';
}

function renderLog(lines) {
  $("log-tail").textContent = lines.length ? lines.join("\n") : "暂无日志";
}

function shouldLoadRoute(path) {
  const normalizedPath = normalizeResultPath(path);
  if (!normalizedPath || !normalizedPath.endsWith(".html")) return false;
  if (normalizedPath === state.loadedRoutePath) return false;
  if (normalizedPath === state.routeLoadingPath) return false;
  if (normalizedPath === state.routeFailedPath && Date.now() - state.routeFailedAt < 5000) return false;
  return true;
}

function showRouteMessage(kind, title, detail, path = "") {
  const panel = document.querySelector(".route-panel");
  panel.classList.toggle("loading-route", kind === "loading");
  panel.classList.toggle("route-error", kind === "error");
  panel.classList.remove("has-route");
  panel.setAttribute("aria-busy", kind === "loading" ? "true" : "false");
  $("route-empty").innerHTML = `
    <div class="route-message">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(detail)}</span>
      ${path ? `<code>${escapeHtml(path)}</code>` : ""}
      ${kind === "error" && path ? '<button class="secondary-button" type="button" data-retry-route>重试路线图</button>' : ""}
    </div>
  `;
}

function setOpenRouteLink(url = "") {
  const link = $("open-route");
  const enabled = Boolean(url);
  link.href = enabled ? url : "#";
  link.classList.toggle("disabled", !enabled);
  link.setAttribute("aria-disabled", String(!enabled));
  link.tabIndex = enabled ? 0 : -1;
}

function failRouteLoad(path, token, detail) {
  if (token !== state.routeLoadToken) return;
  clearRouteHandshake();
  state.routeLoadingPath = "";
  state.routeFailedPath = path;
  state.routeFailedAt = Date.now();
  state.loadedRoutePath = "";
  $("route-state").textContent = "载入失败";
  $("route-counts").textContent = path;
  setOpenRouteLink();
  showRouteMessage("error", "路线图暂时不可用", detail || "后端没有返回可渲染的 HTML。", path);
}

function clearRouteHandshake() {
  window.clearTimeout(state.routeHandshakeTimer);
  state.routeHandshakeTimer = 0;
  state.routeHandshakeToken = "";
}

function createRouteHandshakeToken(token) {
  const random = new Uint32Array(2);
  window.crypto.getRandomValues(random);
  return `${token}-${Date.now()}-${random[0].toString(36)}${random[1].toString(36)}`;
}

function completeRouteLoad(path, token, message) {
  if (token !== state.routeLoadToken) return;
  clearRouteHandshake();
  state.routeLoadingPath = "";
  state.routeFailedPath = "";
  state.loadedRoutePath = path;
  const panel = document.querySelector(".route-panel");
  const frame = $("route-frame");
  panel.setAttribute("aria-busy", "false");
  panel.classList.remove("loading-route", "route-error");
  panel.classList.add("has-route");
  frame.title = `逆合成路线森林：${path.split("/").slice(-2, -1)[0] || "结果"}`;
  if (!state.currentJobId) {
    $("route-state").textContent = "已载入";
    const counts = message?.counts || {};
    $("route-counts").textContent = counts.branches
      ? `${counts.branches} 分支 · ${counts.reaction_nodes || counts.steps || 0} 反应`
      : path;
  }
}

function handleRouteFrameMessage(event) {
  const frame = $("route-frame");
  const message = event.data || {};
  if (event.source !== frame.contentWindow || message.type !== ROUTE_READY_MESSAGE) return;
  if (!state.routeHandshakeToken || message.token !== state.routeHandshakeToken) return;
  if (message.schema_version !== "route_forest_delivery.v1") {
    failRouteLoad(state.routeLoadingPath, state.routeLoadToken, "路线图返回了不受支持的交付契约。");
    return;
  }
  if (message.integrity_status !== "verified") {
    failRouteLoad(state.routeLoadingPath, state.routeLoadToken, "路线图交付摘要未在浏览器中完成验证，已拒绝显示。");
    return;
  }
  completeRouteLoad(state.routeLoadingPath, state.routeLoadToken, message);
}

function loadRoute(path) {
  const normalizedPath = normalizeResultPath(path);
  if (!normalizedPath.endsWith(".html")) {
    showRouteMessage("error", "路线图路径无效", "当前产物不是可嵌入的 HTML 路线图。", normalizedPath);
    return;
  }
  const frame = $("route-frame");
  const token = ++state.routeLoadToken;
  const url = resultFileUrl(normalizedPath);
  clearRouteHandshake();
  state.routeHandshakeToken = createRouteHandshakeToken(token);
  const embedUrl = `${url}&embed=1&parent_token=${encodeURIComponent(state.routeHandshakeToken)}`;
  state.routeLoadingPath = normalizedPath;
  state.loadedRoutePath = "";
  setOpenRouteLink(url);
  showRouteMessage("loading", "正在载入路线图", "大型路线图可能需要数秒；任务运行时会在下一次轮询继续检查。", normalizedPath);
  frame.onload = () => {};
  frame.onerror = () => failRouteLoad(normalizedPath, token, "路线图网络加载失败，可重试或在新窗口检查。");
  state.routeHandshakeTimer = window.setTimeout(() => {
    failRouteLoad(normalizedPath, token, "路线图未完成交付握手；文件可能缺失、损坏或属于旧版格式。");
  }, ROUTE_HANDSHAKE_TIMEOUT_MS);
  frame.src = `${embedUrl}&_=${Date.now()}`;
}

function clearRoute() {
  state.routeLoadToken += 1;
  clearRouteHandshake();
  state.loadedRoutePath = "";
  state.routeLoadingPath = "";
  state.routeFailedPath = "";
  state.routeFailedAt = 0;
  const frame = $("route-frame");
  frame.onload = null;
  frame.onerror = null;
  frame.removeAttribute("src");
  frame.title = "逆合成路线森林";
  setOpenRouteLink();
  const panel = document.querySelector(".route-panel");
  panel.setAttribute("aria-busy", "false");
  panel.classList.remove("has-route", "loading-route", "route-error");
  $("route-empty").textContent = "尚未载入路线图";
  $("route-state").textContent = "未生成";
  $("route-counts").textContent = "等待 route forest";
}

function loadExisting() {
  const path = normalizeResultPath($("existing-route").value);
  if (!path) return;
  if (!state.currentJobId) {
    $("job-title").textContent = "已有路线";
    $("job-subtitle").textContent = path;
    $("job-status").textContent = "preview";
    $("job-progress").textContent = "只读预览，不占用 agent";
    $("route-state").textContent = "载入中";
    $("route-counts").textContent = path;
  }
  loadRoute(path);
}

function syncDemoTarget() {
  const selected = $("demo-route").selectedOptions[0];
  const sampleKey = selected?.dataset.sampleKey || "";
  if (sampleKey) applySample(sampleKey);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function shorten(value, limit = 70) {
  const text = String(value || "");
  return text.length > limit ? `${text.slice(0, limit - 1)}...` : text;
}

function bind() {
  window.addEventListener("message", handleRouteFrameMessage);
  document.querySelector(".skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    focusRoute();
  });
  $("refresh-status").addEventListener("click", refreshStatus);
  $("toggle-controls").addEventListener("click", toggleControls);
  $("toggle-activity").addEventListener("click", toggleActivity);
  $("focus-route").addEventListener("click", focusRoute);
  $("start-run").addEventListener("click", startRun);
  $("cancel-run").addEventListener("click", cancelRun);
  $("load-existing").addEventListener("click", loadExisting);
  $("route-empty").addEventListener("click", (event) => {
    if (!event.target.closest("[data-retry-route]")) return;
    const path = state.routeFailedPath || normalizeResultPath($("existing-route").value);
    if (path) loadRoute(path);
  });
  $("demo-route").addEventListener("change", () => {
    $("existing-route").value = $("demo-route").value;
    syncDemoTarget();
  });
  document.querySelectorAll(".mobile-view-tab").forEach((button) => {
    button.addEventListener("click", () => setMobileView(button.dataset.mobileView || "route"));
    button.addEventListener("keydown", (event) => {
      const tabs = [...document.querySelectorAll(".mobile-view-tab")];
      const index = tabs.indexOf(button);
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      setMobileView(tabs[next].dataset.mobileView || "route", { focusTab: true });
    });
  });
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => applySample(button.dataset.sample));
  });
  const layoutMedia = window.matchMedia(MOBILE_BREAKPOINT);
  if (layoutMedia.addEventListener) layoutMedia.addEventListener("change", applyLayoutState);
  else layoutMedia.addListener(applyLayoutState);
}

async function init() {
  readLayoutState();
  bind();
  applyLayoutState();
  refreshStatus();
  const restored = await restoreActiveJob();
  if (!restored) {
    syncDemoTarget();
    loadExisting();
  }
}

init();
