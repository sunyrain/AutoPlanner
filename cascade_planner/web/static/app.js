const state = {
  currentJob: null,
  jobTimer: null,
  currentPlanJob: null,
  planJobTimer: null,
  jobs: [],
  artifacts: [],
  currentPlan: null,
  selectedRouteIndex: 0,
  activeTool: "plan",
  activeView: "setup",
  artifactsOpen: false,
};

const $ = (id) => document.getElementById(id);

const statinSamples = [
  "CC(C)C1=NC(=NC(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
  "CCC(C)(C)C(=O)O[C@H]1C[C@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@@H]3C[C@H](CC(=O)O3)O)C",
  "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
];

const routePresets = {
  quick: {
    maxSteps: 6,
    iterations: 10,
    expansionTopk: 50,
  },
  balanced: {
    maxSteps: 6,
    iterations: 25,
    expansionTopk: 75,
  },
  thorough: {
    maxSteps: 8,
    iterations: 50,
    expansionTopk: 100,
  },
};

function applyRoutePreset(name) {
  const preset = routePresets[name] || routePresets.quick;
  $("max-steps").value = String(preset.maxSteps);
  $("chem-enzy-iterations").value = String(preset.iterations);
  $("chem-enzy-expansion-topk").value = String(preset.expansionTopk);
}

function setActiveTool(tool) {
  state.activeTool = tool;
  document.body.dataset.tool = tool;
  const nav = {
    demo: "nav-demo",
    plan: "nav-plan",
    benchmark: "nav-benchmark",
    artifacts: "nav-artifacts",
  };
  Object.entries(nav).forEach(([key, id]) => {
    const button = $(id);
    if (button) {
      button.classList.toggle("active", key === tool);
    }
  });
  if (tool === "artifacts") {
    setArtifactsOpen(true);
  } else {
    syncArtifactToggle();
  }
}

function setArtifactsOpen(isOpen) {
  state.artifactsOpen = Boolean(isOpen);
  document.body.classList.toggle("artifacts-open", state.artifactsOpen);
  syncArtifactToggle();
}

function syncArtifactToggle() {
  const button = $("toggle-artifacts");
  if (button) {
    button.textContent = state.artifactsOpen ? "收起文件" : "结果文件";
    button.classList.toggle("active", state.artifactsOpen);
  }
}

function setDetailsButtonsVisible(isVisible) {
  ["expand-details", "collapse-details"].forEach((id) => {
    const button = $(id);
    if (button) {
      button.style.display = isVisible ? "" : "none";
    }
  });
}

function setBusy(isBusy, message) {
  $("run-plan").disabled = isBusy;
  $("run-eval").disabled = isBusy && state.currentJob;
  $("route-meta").textContent = message || (isBusy ? "running" : "idle");
  $("route-meta").className = isBusy ? "pill warn" : "pill muted";
}

function setPlanBusy(isBusy, message) {
  $("run-plan").disabled = false;
  const cancelButton = $("cancel-plan");
  if (cancelButton) {
    cancelButton.disabled = !isBusy || !state.currentPlanJob;
  }
  $("route-meta").textContent = message || (isBusy ? "queued" : "idle");
  $("route-meta").className = isBusy ? "pill warn" : "pill muted";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { ok: response.ok, text };
  }
  if (!response.ok) {
    throw new Error(data.error || data.text || response.statusText);
  }
  return data;
}

function setWorkspaceMode(mode) {
  state.activeView = mode;
  document.body.dataset.view = mode;
  const routeMode = mode === "route";
  $("steps-panel").style.display = routeMode ? "" : "none";
  $("skeleton-panel").style.display = routeMode ? "" : "none";
  $("metrics-panel").style.display = routeMode ? "" : "none";
}

function setPrimaryPanel(title, metaText, status = "muted") {
  $("primary-panel-title").textContent = title;
  $("route-meta").textContent = metaText;
  $("route-meta").className = `pill ${status}`;
}

function showRouteSearchHome() {
  state.currentPlan = null;
  state.selectedRouteIndex = 0;
  setActiveTool("plan");
  setArtifactsOpen(false);
  setWorkspaceMode("setup");
  $("view-title").textContent = "Route Search";
  $("view-subtitle").textContent = "核心工作台：调用 CascadePlanner，并启用已整合的 cascade/value hooks。";
  setPrimaryPanel("路线搜索", "就绪", "muted");
  setDetailsButtonsVisible(false);
  $("summary-cards").innerHTML = "";
  $("route-view").className = "";
  $("route-view").innerHTML = `
    <div class="route-start">
      <div>
        <span class="eyebrow">核心流程</span>
        <h3>从路线搜索开始</h3>
        <p>路线搜索调用 CascadePlanner：以 ChemEnzyRetroPlanner 多步搜索为底层引擎，并在搜索内部接入 cascade cost、source policy、state/action value hook。</p>
      </div>
      <div class="route-start-grid">
        <div><b>CascadePlanner</b><span>主路径是多步搜索控制器，不再走临时外层搜索。</span></div>
        <div><b>Cascade hooks</b><span>source policy、cost model、action value trace 已在 ChemEnzy 搜索内部启用。</span></div>
        <div><b>CUDA</b><span>当 ChemEnzy 模型支持 GPU 时，搜索请求会使用本机 GPU。</span></div>
      </div>
    </div>
  `;
  clearRouteSpecificPanels();
}

function showBenchmarkHome() {
  state.currentPlan = null;
  state.selectedRouteIndex = 0;
  setActiveTool("benchmark");
  setArtifactsOpen(false);
  setWorkspaceMode("benchmark");
  $("view-title").textContent = "批量验收";
  $("view-subtitle").textContent = "批量验收入口。不会自动启动实验，点击开始后才会运行。";
  setPrimaryPanel("批量验收", "就绪", "muted");
  setDetailsButtonsVisible(false);
  $("summary-cards").innerHTML = "";
  $("route-view").className = "";
  $("route-view").innerHTML = `
    <div class="empty-state task-state">
      <b>批量验收未启动</b>
      <span>左侧选择 benchmark 配置后手动启动。当前切换不会复用上一个路线结果。</span>
    </div>
  `;
  clearRouteSpecificPanels();
}

function showArtifactsHome() {
  state.currentPlan = null;
  state.selectedRouteIndex = 0;
  setActiveTool("artifacts");
  setWorkspaceMode("raw");
  $("view-title").textContent = "结果文件";
  $("view-subtitle").textContent = "打开本地 benchmark 报告、JSON artifact 或模型报告。";
  setPrimaryPanel("结果浏览器", "就绪", "muted");
  setDetailsButtonsVisible(false);
  $("summary-cards").innerHTML = "";
  $("route-view").className = "";
  $("route-view").innerHTML = `
    <div class="empty-state task-state">
      <b>结果浏览器</b>
      <span>从右侧 Saved Results 选择文件预览；不会启动新的搜索或实验。</span>
    </div>
  `;
  clearRouteSpecificPanels();
}

function clearRouteSpecificPanels() {
  $("step-table").innerHTML = "";
  $("step-count").textContent = "0";
  $("skeleton-view").innerHTML = "";
  $("skeleton-count").textContent = "0";
  $("metrics-view").innerHTML = "";
  $("metric-state").textContent = "empty";
}

function readPlanPayload() {
  let constraints = null;
  const raw = $("constraints-json").value.trim();
  if (raw) {
    constraints = JSON.parse(raw);
  }
  const maxSteps = Number($("max-steps").value);
  const stockMode = $("stock-mode")?.value || "building-block";
  return {
    target_smiles: $("target-smiles").value.trim(),
    search_preset: $("search-preset").value,
    stock_mode: stockMode,
    stock_names: stockNamesForMode(stockMode),
    search_mode: "fixed",
    planner_backend: "chem_enzy_native",
    planner_mode: "chem_enzy_native",
    adaptive_depth: false,
    n_steps: maxSteps,
    min_steps: maxSteps,
    max_steps: maxSteps,
    stop_on_solved: true,
    domain: "chemoenzymatic",
    device: $("device").value,
    chem_enzy_iterations: Number($("chem-enzy-iterations").value),
    chem_enzy_expansion_topk: Number($("chem-enzy-expansion-topk").value),
    enable_condition_prediction: $("enable-condition-prediction")?.checked || false,
    enable_enzyme_assignment: $("enable-enzyme-assignment")?.checked || false,
    condition_model: "rcr",
    constraints,
  };
}

function stockNamesForMode(mode) {
  if (mode === "commercial") return ["Zinc_Fix-stock"];
  if (mode === "benchmark-n5") return ["PaRotes_n5-stock"];
  return ["PaRotes_n1-stock"];
}

async function runPlan() {
  try {
    setActiveTool("plan");
    setArtifactsOpen(false);
    setWorkspaceMode("setup");
    setDetailsButtonsVisible(false);
    setPlanBusy(true, "queued");
    $("route-view").className = "";
    $("route-view").innerHTML = `
      <div class="empty-state task-state">
        <b>路线搜索已提交</b>
        <span>任务会进入本机队列；左侧“路线任务”显示排队、运行和完成状态。</span>
      </div>
    `;
    const job = await api("/api/plan-jobs", {
      method: "POST",
      body: JSON.stringify(readPlanPayload()),
    });
    state.currentPlanJob = job.job_id;
    renderPlanJobs([job, ...state.jobs.filter((row) => row.job_id !== job.job_id)]);
    startPlanJobPolling(job.job_id);
  } catch (err) {
    showError(err.message);
    setPlanBusy(false, "idle");
  }
}

function startPlanJobPolling(jobId) {
  state.currentPlanJob = jobId;
  setPlanBusy(true, "queued");
  if (state.planJobTimer) {
    clearInterval(state.planJobTimer);
  }
  state.planJobTimer = setInterval(() => pollPlanJob(jobId), 1500);
  pollPlanJob(jobId);
  loadJobs();
}

async function pollPlanJob(jobId) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    upsertJob(job);
    renderActivePlanJob(job);
    if (isTerminalPlanStatus(job.status)) {
      clearInterval(state.planJobTimer);
      state.planJobTimer = null;
      state.currentPlanJob = null;
      setPlanBusy(false, job.status);
      await loadJobs();
      await loadArtifacts();
      if (job.output_json && job.status === "complete") {
        await openArtifact(job.output_json);
      }
    }
  } catch (err) {
    setPlanBusy(false, "error");
    $("route-view").innerHTML = `<div class="empty-state">${escapeHtml(err.message)}</div>`;
  }
}

function renderActivePlanJob(job) {
  const summary = job.summary || {};
  const status = jobStatusLabel(job.status);
  setPrimaryPanel("路线任务", `${status} · ${job.job_id}`, jobTone(job.status));
  $("view-title").textContent = "Route Search";
  $("view-subtitle").textContent = `${job.target_preview || ""} · ${status}`;
  renderSummaryCards([
    ["Task", status],
    ["Preset", job.search_preset || "-"],
    ["Stock", readableStockMode(job.stock_mode, job.stock_names)],
    ["Depth", job.max_depth ?? "-"],
    ["Iterations", job.iterations ?? "-"],
    ["Top-k", job.expansion_topk ?? "-"],
    ["Routes", summary.routes ?? "-"],
    ["Elapsed", job.elapsed_s ? `${fmt(job.elapsed_s)}s` : "-"],
  ]);
  $("route-view").className = "";
  $("route-view").innerHTML = planJobStatusHtml(job);
}

function planJobStatusHtml(job) {
  const summary = job.summary || {};
  const logRows = (job.log_tail || []).slice(-8).map((line) => `<div>${escapeHtml(line)}</div>`).join("");
  const done = isTerminalPlanStatus(job.status);
  const output = job.output_json ? `<button class="small-btn" type="button" data-open-job-output="${escapeHtml(job.output_json)}">打开结果</button>` : "";
  const rawOutput = job.raw_output_json ? `<button class="small-btn" type="button" data-open-job-output="${escapeHtml(job.raw_output_json)}">打开 raw</button>` : "";
  const rejectedOutput = job.rejected_output_json ? `<button class="small-btn" type="button" data-open-job-output="${escapeHtml(job.rejected_output_json)}">打开 rejected</button>` : "";
  const cancel = isCancellablePlanStatus(job.status)
    ? `<button class="danger-btn small-danger" type="button" data-cancel-job-id="${escapeHtml(job.job_id)}">${job.status === "cancelling" ? "正在终止" : "终止路线"}</button>`
    : "";
  const queueText = job.queue_position ? `队列位置 #${job.queue_position}` : (job.status === "running" || job.status === "cancelling" ? "当前运行" : "-");
  return `
    <div class="job-status-card ${job.status}">
      <div class="job-status-head">
        <div>
          <b>${escapeHtml(jobStatusLabel(job.status))}</b>
          <span>${escapeHtml(job.label || "Route search")}</span>
        </div>
        <span class="pill ${jobTone(job.status)}">${escapeHtml(job.status)}</span>
      </div>
      <div class="job-status-grid">
        ${miniMetric("preset", job.search_preset || "-")}
        ${miniMetric("stock", readableStockMode(job.stock_mode, job.stock_names))}
        ${miniMetric("max depth", job.max_depth ?? "-")}
        ${miniMetric("iterations", job.iterations ?? "-")}
        ${miniMetric("top-k", job.expansion_topk ?? "-")}
        ${miniMetric("routes", summary.routes ?? "-")}
        ${miniMetric("queue", queueText)}
        ${miniMetric("elapsed", job.elapsed_s ? `${fmt(job.elapsed_s)}s` : "-")}
      </div>
      <div class="smiles">target=${escapeHtml(job.target_smiles || "")}</div>
      ${summary.message ? `<div class="job-message">${escapeHtml(summary.message)}</div>` : ""}
      ${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ""}
      <div class="job-log">${logRows || "<div>waiting for log...</div>"}</div>
      <div class="job-actions">${output}${rawOutput}${rejectedOutput}${cancel}${done ? "" : '<span class="muted">任务会按队列顺序运行；需要停止时点“终止路线”。</span>'}</div>
    </div>
  `;
}

document.addEventListener("click", async (event) => {
  const cancelButton = event.target.closest("[data-cancel-job-id]");
  if (cancelButton) {
    await cancelPlanJob(cancelButton.getAttribute("data-cancel-job-id"));
    return;
  }
  const button = event.target.closest("[data-open-job-output]");
  if (!button) return;
  const path = button.getAttribute("data-open-job-output");
  if (path) {
    await openArtifact(path);
  }
});

function readEvalPayload() {
  return {
    bench: $("bench-select").value,
    label: $("eval-label").value || "ui_depth",
    device: $("device").value,
    depths: $("eval-depths").value.split(/\s+/).filter(Boolean).map(Number),
    n_per_depth: Number($("eval-n").value),
    ultra_depth: Number($("ultra-depth").value),
    ultra_targets: Number($("ultra-targets").value),
    skeleton_samples: 1,
    n_results: 3,
    candidate_budget: 4,
    expansion_multiplier: 4,
  };
}

async function runEval() {
  try {
    setActiveTool("benchmark");
    setArtifactsOpen(false);
    setWorkspaceMode("benchmark");
    setDetailsButtonsVisible(false);
    $("view-title").textContent = "Benchmark Run";
    $("view-subtitle").textContent = "Queued local evaluation job";
    setPrimaryPanel("Benchmark Job", "queued", "warn");
    $("summary-cards").innerHTML = "";
    $("route-view").className = "";
    $("route-view").innerHTML = '<div class="empty-state">Benchmark job queued. Progress appears in the left panel.</div>';
    const job = await api("/api/evaluate", {
      method: "POST",
      body: JSON.stringify(readEvalPayload()),
    });
    state.currentJob = job.job_id;
    $("job-box").textContent = `Job ${job.job_id}\nqueued`;
    if (state.jobTimer) {
      clearInterval(state.jobTimer);
    }
    state.jobTimer = setInterval(() => pollJob(job.job_id), 1800);
    pollJob(job.job_id);
  } catch (err) {
    showError(err.message);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    const lines = [
      `Job ${job.job_id}`,
      `status: ${job.status}`,
      `output: ${job.output_json}`,
      "",
      ...(job.log_tail || []),
    ];
    $("job-box").textContent = lines.join("\n");
    if (job.status === "complete" || job.status === "failed") {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      state.currentJob = null;
      await loadArtifacts();
      if (job.output_json) {
        openArtifact(job.output_json);
      }
    }
  } catch (err) {
    $("job-box").textContent = err.message;
  }
}

async function loadJobs() {
  try {
    const data = await api("/api/jobs");
    renderPlanJobs((data.jobs || []).filter((job) => job.kind === "plan"));
  } catch (err) {
    const list = $("plan-job-list");
    if (list) {
      list.innerHTML = `<div class="job-empty">${escapeHtml(err.message)}</div>`;
    }
  }
}

function upsertJob(job) {
  const rows = [job, ...state.jobs.filter((row) => row.job_id !== job.job_id)];
  renderPlanJobs(rows);
}

function renderPlanJobs(jobs) {
  state.jobs = jobs || [];
  const list = $("plan-job-list");
  if (!list) return;
  if (!state.jobs.length) {
    list.innerHTML = '<div class="job-empty">当前没有路线任务</div>';
    return;
  }
  list.innerHTML = state.jobs.slice(0, 12).map((job) => {
    const summary = job.summary || {};
    const active = job.job_id === state.currentPlanJob ? "active" : "";
    const status = jobStatusLabel(job.status);
    const stock = readableStockMode(job.stock_mode, job.stock_names);
    const routes = summary.routes !== undefined ? `${summary.routes} routes` : `${job.search_preset || "-"} · ${stock} · d${job.max_depth || "-"}`;
    const elapsed = job.elapsed_s ? `${fmt(job.elapsed_s)}s` : (job.queue_position ? `queue #${job.queue_position}` : (job.started_at ? "running" : "queued"));
    return `
      <button class="job-row ${active}" type="button" data-job-id="${escapeHtml(job.job_id)}">
        <span class="job-dot ${escapeHtml(job.status || "")}"></span>
        <span class="job-row-main">
          <b>${escapeHtml(status)}</b>
          <small>${escapeHtml(job.target_preview || job.job_id)}</small>
          <small>${escapeHtml(routes)} · ${escapeHtml(elapsed)}</small>
        </span>
        <span class="pill ${jobTone(job.status)}">${escapeHtml(job.status || "-")}</span>
      </button>
    `;
  }).join("");
  list.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", () => openJob(button.getAttribute("data-job-id")));
  });
}

async function openJob(jobId) {
  if (!jobId) return;
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  upsertJob(job);
  if (job.output_json && isTerminalPlanStatus(job.status)) {
    await openArtifact(job.output_json);
    return;
  }
  setActiveTool("plan");
  setArtifactsOpen(false);
  setWorkspaceMode("setup");
  renderActivePlanJob(job);
  if (isCancellablePlanStatus(job.status)) {
    startPlanJobPolling(job.job_id);
  }
}

async function cancelCurrentPlan() {
  const fallback = state.jobs.find((job) => isCancellablePlanStatus(job.status));
  const jobId = state.currentPlanJob || fallback?.job_id;
  if (!jobId) return;
  await cancelPlanJob(jobId);
}

async function cancelPlanJob(jobId) {
  if (!jobId) return;
  const cancelButton = $("cancel-plan");
  if (cancelButton) {
    cancelButton.disabled = true;
  }
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  upsertJob(job);
  renderActivePlanJob(job);
  if (!isTerminalPlanStatus(job.status)) {
    state.currentPlanJob = job.job_id;
    startPlanJobPolling(job.job_id);
  } else if (state.currentPlanJob === job.job_id) {
    state.currentPlanJob = null;
    setPlanBusy(false, job.status);
  }
  await loadJobs();
}

function isTerminalPlanStatus(status) {
  return status === "complete" || status === "failed" || status === "cancelled";
}

function isCancellablePlanStatus(status) {
  return status === "queued" || status === "running" || status === "cancelling";
}

function jobStatusLabel(status) {
  if (status === "queued") return "排队中";
  if (status === "running") return "运行中";
  if (status === "cancelling") return "终止中";
  if (status === "complete") return "已完成";
  if (status === "failed") return "失败";
  if (status === "cancelled") return "已终止";
  return status || "未知";
}

function readableStockMode(mode, names) {
  if (mode === "commercial") return "ZINC";
  if (mode === "benchmark-n5") return "n5";
  if (mode === "building-block") return "n1";
  const list = Array.isArray(names) ? names : [];
  return list.length ? list.join("+").replaceAll("-stock", "") : "-";
}

function jobTone(status) {
  if (status === "complete") return "good";
  if (status === "failed" || status === "cancelled") return "bad";
  if (status === "running" || status === "queued" || status === "cancelling") return "warn";
  return "muted";
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    const pill = $("status-pill");
    pill.textContent = data.cuda.available ? "cuda ready" : "cpu";
    pill.className = data.cuda.available ? "pill good" : "pill warn";
  } catch {
    $("status-pill").textContent = "offline";
    $("status-pill").className = "pill bad";
  }
}

async function loadArtifacts() {
  const data = await api("/api/artifacts");
  state.artifacts = data.artifacts || [];
  $("artifact-count").textContent = String(state.artifacts.length);
  const list = $("artifact-list");
  list.innerHTML = "";
  state.artifacts.slice(0, 80).forEach((item) => {
    const row = document.createElement("div");
    row.className = "artifact-row";
    row.innerHTML = `
      <div class="artifact-row-head">
        <b>${escapeHtml(artifactTitle(item.name))}</b>
        <span class="pill muted">${escapeHtml(artifactKind(item))}</span>
      </div>
      <div class="muted">${item.size_kb} KB · ${escapeHtml(item.mtime)}</div>
    `;
    row.addEventListener("click", () => openArtifact(item.path));
    list.appendChild(row);
  });
}

async function openCascadeDemo() {
  try {
    setActiveTool("demo");
    setArtifactsOpen(false);
    setPrimaryPanel("Cascade Demo", "loading", "muted");
    const data = await api("/api/cascade-demo");
    renderCascadeDemo(data);
  } catch (err) {
    showError(err.message);
  }
}

async function openArtifact(path) {
  try {
    const encoded = encodeURIComponent(path);
    const item = state.artifacts.find((x) => x.path === path) || { suffix: path.split(".").pop() };
    const response = await fetch(`/api/artifact?path=${encoded}`);
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text);
    }
    if (path.endsWith(".json")) {
      const data = JSON.parse(text);
      if (data.routes) {
        state.selectedRouteIndex = 0;
        renderPlan(data);
      } else if (data.targets) {
        renderBenchmark(data, path);
      } else {
        renderRaw(path, JSON.stringify(data, null, 2));
      }
    } else {
      renderRaw(path, text);
    }
  } catch (err) {
    showError(err.message);
  }
}

function renderCascadeDemo(data) {
  state.currentPlan = null;
  setActiveTool("demo");
  setWorkspaceMode("demo");
  setDetailsButtonsVisible(false);
  $("view-title").textContent = "AutoPlanner-Cascade";
  $("view-subtitle").textContent = data.headline?.subtitle || "Cascade-native program search";
  setPrimaryPanel("Cascade Demo Board", "current local results", "good");
  renderSummaryCards((data.cards || []).slice(0, 6).map((card) => [card.label, formatDemoValue(card.value)]));
  clearRouteSpecificPanels();
  $("route-view").className = "";
  $("route-view").innerHTML = cascadeDemoHtml(data);
}

function cascadeDemoHtml(data) {
  const cards = (data.cards || []).map((card) => `
    <div class="demo-metric-card">
      <b>${escapeHtml(formatDemoValue(card.value))}</b>
      <span>${escapeHtml(card.label)}</span>
      <small>${escapeHtml(card.note || "")}</small>
    </div>
  `).join("");
  const heroStats = (data.cards || []).slice(0, 3).map((card) => `
    <div>
      <b>${escapeHtml(formatDemoValue(card.value))}</b>
      <span>${escapeHtml(card.label)}</span>
    </div>
  `).join("");
  const models = (data.models || []).map((model) => `
    <div class="demo-model-row">
      <div>
        <b>${escapeHtml(model.name)}</b>
        <span>${escapeHtml(model.role || "")}</span>
      </div>
      <span class="pill ${escapeHtml(model.tone || (model.status?.includes("pending") ? "warn" : "good"))}">${escapeHtml(model.status || "")}</span>
      <div class="demo-model-metrics">${Object.entries(model.metrics || {}).map(([key, value]) => `
        <span><b>${escapeHtml(formatDemoValue(value))}</b>${escapeHtml(key)}</span>
      `).join("") || '<span class="muted">no numeric report</span>'}</div>
    </div>
  `).join("");
  const cases = (data.cases || []).map((item, index) => cascadeCaseHtml(item, index)).join("");
  const artifacts = Object.entries(data.artifacts || {}).map(([key, item]) => `
    <div class="artifact-chip">
      <b>${escapeHtml(key.replaceAll("_", " "))}</b>
      <span>${escapeHtml(item.exists ? item.path : "missing")}</span>
    </div>
  `).join("");
  const next = data.next_step || {};
  return `
    <div class="demo-board">
      <section class="demo-hero">
        <div class="demo-hero-copy">
          <span class="eyebrow">本地验收快照</span>
          <h3>${escapeHtml(data.headline?.title || "AutoPlanner-Cascade")}</h3>
          <p>${escapeHtml(data.headline?.message || "")}</p>
          <div class="demo-hero-points">
            <span>级联偏好与 state-action scoring 已接入搜索控制器。</span>
            <span>Full100 与 hard-gap 指标来自当前本地 artifacts，不是手写展示数。</span>
            <span>当前主要瓶颈仍是 candidate recovery，而不是结果展示。</span>
          </div>
        </div>
        <aside class="demo-hero-panel">
          <b>关键读数</b>
          <div class="demo-hero-stats">${heroStats}</div>
        </aside>
      </section>
      <section class="demo-section">
        <div class="subsection-head">
          <b>当前实测指标</b>
          <span class="muted">${escapeHtml(data.generated_at || "")}</span>
        </div>
        <div class="demo-metric-grid">${cards}</div>
      </section>
      <section class="demo-section">
        <div class="subsection-head">
          <b>模型与控制器状态</b>
          <span class="muted">已实现内容与下一步</span>
        </div>
        <div class="demo-model-list">${models}</div>
      </section>
      <section class="demo-section">
        <div class="subsection-head">
          <b>可展示案例</b>
          <span class="muted">讲证据时再展开卡片</span>
        </div>
        <div class="demo-case-list">${cases || '<div class="empty-state">No cases found.</div>'}</div>
      </section>
      <section class="demo-section">
        <div class="subsection-head">
          <b>${escapeHtml(next.title || "Next step")}</b>
          <span class="pill warn">planned</span>
        </div>
        <div class="demo-next-grid">
          <div><b>Why</b><span>${escapeHtml(next.why || "")}</span></div>
          <div><b>Training target</b><span>${escapeHtml(next.training_target || "")}</span></div>
          <div><b>Data</b><span>${escapeHtml(next.data || "")}</span></div>
        </div>
      </section>
      <details class="demo-artifact-details">
        <summary>本面板引用的本地 artifacts</summary>
        <div class="demo-artifacts">${artifacts}</div>
      </details>
    </div>
  `;
}

function cascadeCaseHtml(item, index) {
  const flags = item.flags || {};
  const recovery = item.recovery || {};
  const flagRows = [
    ["stock", flags.stock_closed],
    ["condition", flags.condition_conflict_free],
    ["cofactor", flags.cofactor_closed],
    ["best exact", flags.best_exact],
    ["best reactant", flags.best_gt_reactant],
    ["candidate exact", flags.candidate_exact],
    ["candidate reactant", flags.candidate_gt_reactant],
    ["top-k exact", flags.topk_exact],
    ["top-k reactant", flags.topk_gt_reactant],
  ].map(([label, value]) => `<span class="pill ${value ? "good" : "muted"}">${escapeHtml(label)} ${value ? "yes" : "no"}</span>`).join("");
  const routeRxns = (item.route_rxns || []).map((rxn) => `<li>${escapeHtml(rxn)}</li>`).join("");
  const gtRxns = (item.gt_rxns || []).map((rxn) => `<li>${escapeHtml(rxn)}</li>`).join("");
  const programs = (item.programs || []).map((program) => `
    <div class="demo-program-row">
      <span class="pill muted">#${escapeHtml(program.rank ?? "-")}</span>
      <span>score ${escapeHtml(fmt(program.score))}</span>
      <span>exact ${escapeHtml(program.exact_reaction_hit_count ?? 0)}</span>
      <span>reactant ${escapeHtml(program.gt_reactant_hit_count ?? 0)}</span>
    </div>
  `).join("");
  return `
    <details class="demo-case-card" ${index === 0 ? "open" : ""}>
      <summary>
        <span class="target-rank">${index + 1}</span>
        <span>
          <b>${escapeHtml(item.label || item.target_smiles)}</b>
          ${item.label ? `<small>${escapeHtml(item.target_smiles)}</small>` : ""}
          <small>${escapeHtml(item.route_domain)} · ${escapeHtml(item.source)} · steps ${escapeHtml(item.step_count ?? "-")}</small>
        </span>
        <span class="pill ${flags.best_exact || flags.best_gt_reactant ? "good" : "warn"}">${escapeHtml(caseTier(flags))}</span>
      </summary>
      <div class="demo-case-body">
        <div class="demo-flags">${flagRows}</div>
        <div class="demo-case-metrics">
          ${miniMetric("route score", item.route_score)}
          ${miniMetric("GT overlap", recovery.gt_step_overlap_fraction)}
          ${miniMetric("exact hits", recovery.exact_reaction_hit_count)}
          ${miniMetric("reactant hits", recovery.gt_reactant_hit_count)}
          ${miniMetric("candidate rxns", recovery.proposal_pool_reaction_count)}
        </div>
        <div class="demo-rxn-grid">
          <div>
            <b>Planner route reactions</b>
            <ol>${routeRxns || "<li>none exported</li>"}</ol>
          </div>
          <div>
            <b>Gold cascade reactions</b>
            <ol>${gtRxns || "<li>none exported</li>"}</ol>
          </div>
        </div>
        ${programs ? `<div class="demo-programs"><b>Top-k result pool evidence</b>${programs}</div>` : ""}
      </div>
    </details>
  `;
}

function caseTier(flags) {
  if (flags.best_exact) return "best-route exact";
  if (flags.best_gt_reactant) return "best-route reactant";
  if (flags.topk_exact) return "top-k exact";
  if (flags.topk_gt_reactant) return "top-k reactant";
  if (flags.candidate_exact) return "candidate exact";
  if (flags.candidate_gt_reactant) return "candidate reactant";
  return "stock-closed";
}

function formatDemoValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n >= 0 && n <= 1) return `${Math.round(n * 100)}%`;
  return Math.abs(n) >= 10 ? n.toFixed(0) : n.toFixed(3);
}

function renderPlan(data) {
  state.currentPlan = data;
  setActiveTool("plan");
  setArtifactsOpen(false);
  setWorkspaceMode("route");
  setDetailsButtonsVisible(true);
  const routes = data.routes || [];
  const selectedIndex = Math.max(0, Math.min(state.selectedRouteIndex || 0, Math.max(routes.length - 1, 0)));
  const route = routes[selectedIndex];
  const isChemEnzyNative = isCascadePlannerPayload(data);
  $("steps-panel").style.display = isChemEnzyNative ? "none" : "";
  $("skeleton-panel").style.display = isChemEnzyNative ? "none" : "";
  const status = normalizedSearchStatus(data);
  $("view-title").textContent = "Route Search";
  $("view-subtitle").textContent = [data.ui_metadata?.saved_at || data.target || "", status.message || ""]
    .filter(Boolean)
    .join(" - ");
  setPrimaryPanel(
    "Route Plan",
    route ? `${statusLabel(status.status)} · route ${selectedIndex + 1}/${routes.length}` : emptyRoutePanelLabel(data, status),
    pillTone(status.status),
  );
  renderSummaryCards(planCards(data, route));
  if (isChemEnzyNative) {
    $("step-table").innerHTML = "";
    $("step-count").textContent = String(route?.steps?.length || 0);
    $("skeleton-view").innerHTML = "";
    $("skeleton-count").textContent = "0";
  } else {
    renderSkeletons(data.skeletons || []);
  }
  if (!route) {
    $("route-view").className = "";
    $("route-view").innerHTML = planStatusHtml(data) + emptyRouteHtml(data);
    $("step-table").innerHTML = "";
    $("metrics-view").innerHTML = "";
    return;
  }
  renderRoute(route, data, selectedIndex);
  if (!isChemEnzyNative) {
    renderSteps(route.steps || []);
  }
  renderMetrics(route.metrics || {}, data);
}

function renderBenchmark(data, path) {
  state.currentPlan = null;
  setActiveTool("benchmark");
  setArtifactsOpen(false);
  setWorkspaceMode("benchmark");
  setDetailsButtonsVisible(false);
  $("view-title").textContent = "Benchmark";
  $("view-subtitle").textContent = path || "Saved benchmark artifact";
  setPrimaryPanel("Benchmark Results", `${data.targets?.length || 0} targets`, "muted");
  renderSummaryCards(benchmarkCards(data.summary || {}));
  const view = $("route-view");
  view.className = "";
  view.innerHTML = benchmarkHtml(data);
  view.querySelectorAll("[data-benchmark-target-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.getAttribute("data-benchmark-target-index") || 0);
      const target = (data.targets || [])[index] || {};
      const plan = target.planner_output || { routes: [] };
      plan.skeletons = target.skeletons || [];
      plan.target = plan.target || target.target_smiles;
      plan.route_recovery = target.route_recovery || {};
      plan.failure_diagnosis = [
        ...(plan.failure_diagnosis || []),
        ...recoveryDiagnosisLabels(target.route_recovery || {}),
      ];
      state.selectedRouteIndex = 0;
      renderPlan(plan);
    });
  });
  clearRouteSpecificPanels();
}

function renderRaw(title, text) {
  state.currentPlan = null;
  setActiveTool("artifacts");
  setWorkspaceMode("raw");
  setDetailsButtonsVisible(false);
  $("view-title").textContent = "Artifact";
  $("view-subtitle").textContent = title;
  setPrimaryPanel("Artifact Preview", "raw file", "muted");
  $("summary-cards").innerHTML = "";
  $("route-view").className = "";
  $("route-view").innerHTML = `<pre class="raw-json">${escapeHtml(text)}</pre>`;
  clearRouteSpecificPanels();
}

function planCards(data, route) {
  const metrics = route?.metrics || {};
  const progress = metrics.retrosynthesis_progress || {};
  const diversity = data.route_set_metrics?.diversity || {};
  const status = normalizedSearchStatus(data);
  return [
    ["Runtime", `${data.time_s ?? "-"}s`],
    ["Search status", statusLabel(status.status)],
    ["Best depth", status.best_depth ?? "-"],
    ["Routes", data.routes?.length ?? 0],
    ["Unique routes", diversity.unique_full_signatures ?? "-"],
    ["Solved", boolText(professionalSolved(route))],
    ["Diagnostic", boolText(diagnosticSolved(route))],
    ["Progressive", boolText(metrics.progressive_route)],
    ["Main reduction", fmtFraction(progress.main_chain_reduction)],
    ["Terminal stock", boolText(metrics.strict_stock_solve)],
  ];
}

function benchmarkCards(summary) {
  const rows = benchmarkSummaryGroups(summary).map(([, row]) => row);
  if (!rows.length) return [];
  const avg = (key) => {
    const nums = rows.map((r) => Number(r[key])).filter((n) => Number.isFinite(n));
    return nums.length ? nums.reduce((s, n) => s + n, 0) / nums.length : null;
  };
  const avgText = (key) => {
    const value = avg(key);
    return value === null ? "-" : value.toFixed(2);
  };
  return [
    ["Groups", rows.length],
    ["Plan rate", avgText("plan_rate")],
    ["Solved rate", avgText("solve_rate")],
    ["Progressive rate", avgText("progressive_rate")],
    ["Main reduction", avgText("avg_main_chain_reduction")],
    ["Filled rate", avgText("filled_rate")],
    ["Compatibility", avgText("compatibility_rate")],
    ["Recovery bottlenecks", topRecoveryBottleneck(summary.recovery_bottleneck_counts || {})],
  ];
}

function benchmarkHtml(data) {
  const summary = data.summary || {};
  const targets = data.targets || [];
  const summaryGroups = benchmarkSummaryGroups(summary);
  const summaryRows = summaryGroups.map(([mode, row]) => `
    <div class="benchmark-card">
      <div class="benchmark-card-head">
        <b>${escapeHtml(formatModeName(mode))}</b>
        <span class="pill muted">${escapeHtml(mode)}</span>
      </div>
      <div class="benchmark-metrics">
        ${miniMetric("Plan", row.plan_rate)}
        ${miniMetric("Exact route", row.exact_route_reaction_match_any)}
        ${miniMetric("Candidate exact", row.candidate_exact_reaction_in_pool)}
        ${miniMetric("GT reactant", row.candidate_gt_reactant_in_pool)}
        ${miniMetric("Pool diversity", row.avg_candidate_pool_diversity_score)}
        ${miniMetric("Top bottleneck", topRecoveryBottleneck(row.recovery_bottleneck_counts || {}))}
      </div>
    </div>
  `).join("");
  const targetRows = targets.map((target, index) => {
    const top = target.top_route || {};
    const outcome = routeOutcome(top);
    const bottleneck = target.route_recovery?.recovery_bottleneck || "";
    return `
      <button class="benchmark-target ${outcome.className}" type="button" data-benchmark-target-index="${index}">
        <span class="target-rank">${index + 1}</span>
        <span class="target-main">
          <b>${escapeHtml(formatModeName(target.mode || target.cascade_id || "target"))}</b>
          <span class="smiles">${escapeHtml(target.target_smiles || "")}</span>
        </span>
        <span class="target-stats">
          <span class="pill ${outcome.tone}">${escapeHtml(outcome.label)}</span>
          ${bottleneck ? `<span class="pill muted">${escapeHtml(formatRiskLabel(bottleneck))}</span>` : ""}
          <span>${escapeHtml(target.n_routes ?? 0)} routes</span>
          <span>${fmtFraction(top.main_chain_reduction)}</span>
          <span>${fmt(target.elapsed_s)}s</span>
        </span>
      </button>
    `;
  }).join("");
  return `
    <div class="benchmark-layout">
      <section class="subsection">
        <div class="subsection-head">
          <b>Summary by Search Group</b>
          <span class="muted">${Object.keys(summary).length} groups</span>
        </div>
        <div class="benchmark-grid">${summaryRows || '<div class="empty-state">No summary metrics.</div>'}</div>
      </section>
      <section class="subsection">
        <div class="subsection-head">
          <b>Targets</b>
          <span class="muted">Open a target to inspect its route plan</span>
        </div>
        <div class="benchmark-target-list">${targetRows || '<div class="empty-state">No target rows.</div>'}</div>
      </section>
    </div>
  `;
}

function benchmarkSummaryGroups(summary) {
  if (!summary || !Object.keys(summary).length) return [];
  const values = Object.values(summary);
  const nested = values.some((value) => {
    return value && typeof value === "object" && !Array.isArray(value) && (
      "plan_rate" in value
      || "exact_route_reaction_match_any" in value
      || "recovery_bottleneck_counts" in value
    );
  });
  if (nested && !("n_targets" in summary || "plan_rate" in summary)) {
    return Object.entries(summary);
  }
  return [["overall", summary]];
}

function topRecoveryBottleneck(counts) {
  const entries = Object.entries(counts || {})
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) return "-";
  const [label, count] = entries[0];
  return `${formatRiskLabel(label)} (${count})`;
}

function miniMetric(label, value) {
  return `
    <span class="mini-metric">
      <b>${escapeHtml(fmtBenchmark(value))}</b>
      <span>${escapeHtml(label)}</span>
    </span>
  `;
}

function renderSummaryCards(cards) {
  const wrap = $("summary-cards");
  wrap.innerHTML = "";
  cards.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.className = "summary-card";
    div.innerHTML = `<b>${escapeHtml(String(value))}</b><span>${escapeHtml(label)}</span>`;
    wrap.appendChild(div);
  });
}

function renderSkeletons(skeletons) {
  $("skeleton-count").textContent = String(skeletons.length);
  const wrap = $("skeleton-view");
  wrap.innerHTML = "";
  skeletons.forEach((s, index) => {
    const row = document.createElement("div");
    row.className = "skeleton-row";
    row.innerHTML = `
      <b>Skeleton ${index + 1}</b>
      <div class="smiles">${(s.types || []).map(escapeHtml).join(" -> ")}</div>
      <div class="muted">EC1 ${(s.ec1s || []).join(", ")} | ${escapeHtml(s.compatibility || "")} | ${escapeHtml(s.operation_mode || "")}</div>
    `;
    wrap.appendChild(row);
  });
  if (!skeletons.length) {
    wrap.innerHTML = '<div class="empty-state">No skeleton data.</div>';
  }
}

function renderRoute(route, data = {}, selectedIndex = 0) {
  if (isCascadePlannerPayload(data)) {
    renderChemEnzyRoute(route, data, selectedIndex);
    return;
  }
  const steps = route.steps || [];
  const graph = document.createElement("div");
  graph.className = "route-graph";
  graph.innerHTML = `
    ${planStatusHtml(data)}
    ${failureRiskHtml(data)}
    ${routeSelectorHtml(data, selectedIndex)}
    ${routeTreeHtml(route, data.target)}
    ${routeWhyHtml(route)}
  `;
  steps.forEach((step) => {
    const card = document.createElement("div");
    card.className = "route-step";
    card.innerHTML = `
      <div class="route-step-head">
        <div>
          <b>Step ${step.index}: ${escapeHtml(formatReactionName(step.reaction_type))}</b>
          <div class="muted">${escapeHtml(step.source || "candidate")} · ${escapeHtml(step.ec || "no EC")} · ${escapeHtml(step.catalyst || "catalyst not specified")}</div>
        </div>
        <span class="pill">${escapeHtml(routeStepOutcome(step))}</span>
      </div>
      ${reactionInsightHtml(step)}
      <div class="molecule-row">
        ${molCard("Precursor", step.main_reactant)}
        <div class="arrow-stack">
          <span class="arrow">-&gt;</span>
          <span>${escapeHtml(formatReactionName(step.reaction_type))}</span>
        </div>
        ${molCard("Product formed", step.product)}
      </div>
      ${stepWhyHtml(step)}
      ${candidatePoolHtml(step)}
    `;
    graph.appendChild(card);
  });
  const view = $("route-view");
  view.className = "";
  view.innerHTML = "";
  view.appendChild(graph);
  graph.querySelectorAll("[data-route-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedRouteIndex = Number(button.getAttribute("data-route-index") || 0);
      renderPlan(state.currentPlan);
    });
  });
  graph.querySelectorAll("[data-apply-retry]").forEach((button) => {
    button.addEventListener("click", () => {
      applyRetrySettings(data.failure_risk?.retry_policy?.adjusted_settings || {});
    });
  });
}

function renderChemEnzyRoute(route, data = {}, selectedIndex = 0) {
  const steps = route.steps || [];
  const graph = document.createElement("div");
  graph.className = "route-graph chem-route";
  graph.innerHTML = `
    ${planStatusHtml(data)}
    ${routeSelectorHtml(data, selectedIndex)}
    ${chemEnzyOverviewHtml(route, data)}
    ${routeStockCascadeHtml(route, data)}
    <div class="chem-step-timeline">
      ${steps.map((step) => chemEnzyStepHtml(step)).join("") || '<div class="empty-state">No steps exported.</div>'}
    </div>
  `;
  const view = $("route-view");
  view.className = "";
  view.innerHTML = "";
  view.appendChild(graph);
  graph.querySelectorAll("[data-route-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedRouteIndex = Number(button.getAttribute("data-route-index") || 0);
      renderPlan(state.currentPlan);
    });
  });
}

function chemEnzyOverviewHtml(route, data = {}) {
  const meta = data.ui_metadata || {};
  const hooks = meta.cascade_hooks || {};
  const metrics = route.metrics || {};
  const postFilter = data.post_filter || {};
  const cards = [
    ["planner", plannerPublicName(meta.backend)],
    ["engine", meta.engine || "ChemEnzyRetroPlanner"],
    ["routes", data.routes?.length ?? 0],
    ["raw routes", postFilter.original_route_count ?? "-"],
    ["filtered", postFilter.removed_route_count ?? 0],
    ["steps", route.steps?.length ?? 0],
    ["time", `${fmt(data.time_s)}s`],
    ["solved", boolText(metrics.route_solved)],
    ["audit class", route.product_audit?.route_class || "-"],
    ["condition", route.product_audit?.condition_audit?.route_risk || "-"],
    ["action value", hooks.action_value_model_path ? "loaded" : "off"],
    ["source value", hooks.source_value_model_path ? "loaded" : "off"],
  ];
  return `
    <section class="chem-route-overview">
      ${cards.map(([label, value]) => `
        <div>
          <b>${escapeHtml(value)}</b>
          <span>${escapeHtml(label)}</span>
        </div>
      `).join("")}
    </section>
    ${productAuditPostFilterHtml(data)}
    ${routeProductAuditHtml(route)}
    <div class="route-strategy-line">
      <b>strategy</b>
      <span>${escapeHtml(meta.planner_strategy || "CascadePlanner search")}</span>
    </div>
  `;
}

function productAuditPostFilterHtml(data = {}) {
  const pf = data.post_filter || {};
  if (!pf.enabled) {
    return "";
  }
  const before = pf.original_route_count ?? 0;
  const kept = pf.kept_route_count ?? (data.routes?.length ?? 0);
  const removed = pf.removed_route_count ?? 0;
  const mode = pf.mode || "hide_rejects";
  const fallback = pf.fallback_reason ? ` · fallback=${pf.fallback_reason}` : "";
  const removedIssues = Object.entries(pf.issue_counts_removed || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5)
    .map(([label, count]) => `${label}:${count}`)
    .join(", ");
  return `
    <div class="route-strategy-line product-audit-line">
      <b>initial filter</b>
      <span>${escapeHtml(mode)} · ${escapeHtml(kept)}/${escapeHtml(before)} kept · ${escapeHtml(removed)} hidden${escapeHtml(fallback)}${removedIssues ? ` · removed=${escapeHtml(removedIssues)}` : ""}</span>
    </div>
  `;
}

function routeProductAuditHtml(route = {}) {
  const audit = route.product_audit || {};
  if (!Object.keys(audit).length) {
    return "";
  }
  const issues = (audit.issues || []).slice(0, 8).join(", ") || "none";
  const tags = (audit.tags || []).slice(0, 8).join(", ") || "none";
  return `
    <div class="route-strategy-line product-audit-line">
      <b>route audit</b>
      <span>${escapeHtml(audit.route_class || "-")} · risk=${escapeHtml(audit.risk_order ?? "-")} · issues=${escapeHtml(issues)} · tags=${escapeHtml(tags)}</span>
    </div>
    ${routeConditionAuditHtml(audit.condition_audit || {})}
  `;
}

function routeConditionAuditHtml(condition = {}) {
  if (!Object.keys(condition).length) {
    return "";
  }
  const risk = condition.route_risk || "ok";
  const tone = risk === "high" ? "bad" : risk === "warn" ? "warn" : "good";
  const issues = Object.entries(condition.issue_counts || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5)
    .map(([label, count]) => `${label}:${count}`)
    .join(", ") || "none";
  const span = condition.temperature_span_c !== null && condition.temperature_span_c !== undefined
    ? ` · T span=${fmt(condition.temperature_span_c)}°C`
    : "";
  return `
    <div class="route-strategy-line product-audit-line">
      <b>condition audit</b>
      <span><span class="pill ${tone}">${escapeHtml(risk)}</span> high=${escapeHtml(condition.high_risk_step_count ?? 0)} · warn=${escapeHtml(condition.warning_step_count ?? 0)}${escapeHtml(span)} · issues=${escapeHtml(issues)}</span>
    </div>
  `;
}

function routeStockCascadeHtml(route, data = {}) {
  const stock = routeStockSummary(route);
  const cascade = routeCascadeSummary(route, data);
  const stockRows = stock.rows.map((row) => `
    <div class="stock-row ${row.ok ? "is-stock" : "not-stock"}">
      <span class="pill ${row.ok ? "good" : "warn"}">${row.ok ? "in stock" : "not in stock"}</span>
      <span class="smiles">${escapeHtml(row.smiles)}</span>
    </div>
  `).join("");
  const cascadeSignals = cascade.signals.map((item) => `
    <span class="cascade-signal ${item.tone || "muted"}">${escapeHtml(item.label)}</span>
  `).join("");
  return `
    <section class="route-evidence-grid">
      <div class="route-evidence-card">
        <div class="subsection-head">
          <b>库存闭合</b>
          <span class="pill ${stock.closed ? "good" : "warn"}">${stock.closed ? "closed" : "open"}</span>
        </div>
        <div class="route-evidence-summary">
          <span>${escapeHtml(stock.inCount)} in stock</span>
          <span>${escapeHtml(stock.outCount)} not in stock</span>
          <span>${escapeHtml(stock.total)} checked</span>
        </div>
        <div class="stock-list">${stockRows || '<div class="muted">No stock fields exported.</div>'}</div>
      </div>
      <div class="route-evidence-card">
        <div class="subsection-head">
          <b>级联信号</b>
          <span class="pill ${cascade.tone}">${escapeHtml(cascade.label)}</span>
        </div>
        <div class="route-evidence-summary">
          <span>${escapeHtml(cascade.stepText)}</span>
          <span>${escapeHtml(cascade.transitionText)}</span>
          <span>${escapeHtml(cascade.hookText)}</span>
        </div>
        <div class="cascade-signal-list">${cascadeSignals}</div>
      </div>
    </section>
  `;
}

function chemEnzyStepHtml(step) {
  const reactants = [step.main_reactant, ...(step.aux_reactants || [])].filter(Boolean);
  const stock = step.stock_status || {};
  const stockValues = Object.values(stock);
  const stockReady = stockValues.length > 0 && stockValues.every(Boolean);
  const stockBits = Object.entries(stock).map(([smi, ok]) => `${ok ? "stock" : "not stock"}: ${smi}`);
  const stepIndex = Number(step.index);
  const displayIndex = Number.isFinite(stepIndex) ? stepIndex + 1 : (step.index ?? "-");
  const displayType = readableStepType(step);
  const displaySource = readableStepSource(step);
  const displayStatus = stockReady ? "stock-ready precursor" : (step.ec ? "enzyme candidate" : displaySource);
  return `
    <article class="chem-step-card">
      <div class="chem-step-index">#${escapeHtml(displayIndex)}</div>
      <div class="chem-step-main">
        <div class="chem-step-head">
          <div>
            <b>${escapeHtml(displayType)}</b>
            <span>${escapeHtml(displaySource)}</span>
          </div>
          <span class="pill ${stockReady ? "good" : "muted"}">${escapeHtml(displayStatus)}</span>
        </div>
        <div class="chem-mol-row">
          <div class="chem-mol-group">
            ${reactants.map((smi, index) => chemMolCard(index === 0 ? "Precursor" : `Partner ${index}`, smi, stock[smi])).join("") || chemMolCard("Precursor", "")}
          </div>
          <div class="chem-arrow">-&gt;</div>
          <div class="chem-mol-group">
            ${chemMolCard("Product", step.product || "")}
          </div>
        </div>
        <div class="chem-step-meta">
          <span>${escapeHtml(displaySource)}</span>
          <span>${escapeHtml(step.ec ? `EC ${step.ec}` : "no EC")}</span>
          <span>${escapeHtml(conditionSummaryText(step))}</span>
          <span>${escapeHtml(stockBits.join(" | ") || "stock not exported")}</span>
        </div>
        ${chemEnzyStepProvenanceHtml(step)}
        ${chemEnzyConditionHtml(step)}
        <details class="chem-step-detail">
          <summary>Reaction SMILES</summary>
          <div class="smiles">${escapeHtml(step.reaction_smiles || "")}</div>
        </details>
      </div>
    </article>
  `;
}

function chemEnzyStepProvenanceHtml(step) {
  const scores = step.scores || {};
  const interp = step.reaction_interpretation || {};
  const atom = interp.atom_change || {};
  const evidence = step.evidence || {};
  const model = step.model_full_name || step.model_name || step.model || step.provider_model || "-";
  const stock = step.stock_status || {};
  const stockText = Object.entries(stock).map(([smi, ok]) => `${ok ? "stock" : "not stock"}:${smi}`).join(" | ");
  const evidenceBits = [
    evidence.uniprot_accession ? `UniProt ${evidence.uniprot_accession}` : "",
    evidence.doi ? `DOI ${evidence.doi}` : "",
    evidence.literature_precedent ? "literature precedent" : "",
  ].filter(Boolean).join(" | ");
  const atomNotes = (atom.notes || []).join(" | ");
  const transformHints = (interp.likely_added_or_removed || []).join(" | ");
  return `
    <details class="chem-step-detail provenance-detail">
      <summary>Proposal provenance</summary>
      <div class="explain-grid">
        <div><b>source</b><span>${escapeHtml(readableStepSource(step))}</span></div>
        <div><b>type / model</b><span>${escapeHtml(step.reaction_type || "-")} / ${escapeHtml(model)}</span></div>
        <div><b>scores</b><span>retro=${escapeHtml(fmt(scores.retro))} enzyme=${escapeHtml(fmt(scores.enzyme))} condition=${escapeHtml(fmt(scores.condition))} confidence=${escapeHtml(fmt(scores.confidence))}</span></div>
        <div><b>atom balance screen</b><span>reactants=${escapeHtml(fmt(atom.reactant_heavy_atoms))} product=${escapeHtml(fmt(atom.product_heavy_atoms))} delta=${escapeHtml(fmt(atom.heavy_atom_delta))}</span></div>
        <div><b>stock evidence</b><span>${escapeHtml(stockText || "-")}</span></div>
        <div><b>external evidence</b><span>${escapeHtml(evidenceBits || "-")}</span></div>
      </div>
      ${atomNotes ? `<div class="smiles">atom_notes=${escapeHtml(atomNotes)}</div>` : ""}
      ${transformHints ? `<div class="smiles">transform_hint=${escapeHtml(transformHints)}</div>` : ""}
    </details>
  `;
}

function conditionSummaryText(step) {
  const bits = [
    step.T !== null && step.T !== undefined ? `T=${fmt(step.T)} C` : "",
    step.pH !== null && step.pH !== undefined ? `pH=${fmt(step.pH)}` : "",
    step.solvent ? `solvent=${step.solvent}` : "",
    step.catalyst ? `reagent/catalyst=${step.catalyst}` : "",
  ].filter(Boolean);
  return bits.join(" | ") || "conditions not predicted";
}

function chemEnzyConditionHtml(step) {
  const conditions = step.condition_predictions || [];
  const enzymes = step.enzyme_ec_annotations || [];
  if (!conditions.length && !enzymes.length) {
    return "";
  }
  const conditionRows = conditions.slice(0, 3).map((row, index) => {
    const temp = row.Temperature ?? row.temperature ?? row.temperature_c ?? "-";
    const ph = row.pH ?? row.ph ?? "-";
    const solvent = row.Solvent ?? row.solvent ?? "-";
    const reagent = row.Reagent ?? row.reagent ?? "-";
    const catalyst = row.Catalyst ?? row.catalyst ?? "-";
    const score = row.Score ?? row.score ?? row.confidence ?? "-";
    return `
      <div class="condition-row">
        <span class="pill muted">#${index + 1}</span>
        <span>T=${escapeHtml(fmt(temp))} C</span>
        <span>pH=${escapeHtml(fmt(ph))}</span>
        <span>solvent=${escapeHtml(solvent)}</span>
        <span>reagent=${escapeHtml(reagent)}</span>
        <span>catalyst=${escapeHtml(catalyst)}</span>
        <span>score=${escapeHtml(fmt(score))}</span>
      </div>
    `;
  }).join("");
  const enzymeRows = enzymes.slice(0, 3).map((row, index) => `
    <div class="condition-row">
      <span class="pill muted">EC #${index + 1}</span>
      <span>${escapeHtml(row.ec_number || row["EC Number"] || "-")}</span>
      <span>confidence=${escapeHtml(fmt(row.confidence || row.Confidence))}</span>
    </div>
  `).join("");
  return `
    <details class="chem-step-detail condition-detail" open>
      <summary>Predicted Conditions</summary>
      <div class="condition-list">
        ${conditionRows || '<div class="muted">No condition predictions.</div>'}
        ${enzymeRows ? `<div class="condition-enzyme-list">${enzymeRows}</div>` : ""}
      </div>
    </details>
  `;
}

function chemMolCard(label, smiles, stockState) {
  const safe = escapeHtml(smiles || "");
  const src = smiles ? `/api/mol.svg?smiles=${encodeURIComponent(smiles)}&w=300&h=140` : "";
  return `
    <div class="chem-mol-card">
      <div class="chem-mol-label-row">
        <span class="chem-mol-label">${escapeHtml(label)}</span>
        ${stockState === undefined ? "" : `<span class="pill ${stockState ? "good" : "warn"}">${stockState ? "in stock" : "not stock"}</span>`}
      </div>
      ${src ? `<img src="${src}" alt="${safe}">` : '<div class="chem-mol-empty">empty</div>'}
      <div class="chem-mol-smiles">${safe || "-"}</div>
    </div>
  `;
}

function routeStockSummary(route) {
  const bySmiles = new Map();
  (route.steps || []).forEach((step) => {
    Object.entries(step.stock_status || {}).forEach(([smiles, ok]) => {
      if (smiles) {
        bySmiles.set(smiles, Boolean(ok));
      }
    });
  });
  const rows = Array.from(bySmiles.entries())
    .map(([smiles, ok]) => ({ smiles, ok }))
    .sort((a, b) => Number(b.ok) - Number(a.ok) || a.smiles.localeCompare(b.smiles));
  const inCount = rows.filter((row) => row.ok).length;
  const outCount = rows.length - inCount;
  return {
    rows,
    total: rows.length,
    inCount,
    outCount,
    closed: rows.length > 0 && outCount === 0,
  };
}

function routeCascadeSummary(route, data = {}) {
  const steps = route.steps || [];
  const hooks = (data.ui_metadata || {}).cascade_hooks || {};
  const types = steps.map((step) => readableStepType(step));
  const isEnzymatic = steps.map((step) => Boolean(step.ec || step.is_enzymatic));
  const enzymeSteps = isEnzymatic.filter(Boolean).length;
  const templateSteps = types.filter((type) => type === "Template step").length;
  const transitions = [];
  for (let i = 1; i < isEnzymatic.length; i += 1) {
    if (isEnzymatic[i] !== isEnzymatic[i - 1]) {
      transitions.push(`${i}->${i + 1}`);
    }
  }
  const hookCount = ["cost_model", "source_policy", "expansion_trace"].filter((key) => hooks[key]).length;
  const signals = [
    { label: hookCount ? `${hookCount}/3 cascade hooks enabled` : "cascade hooks not exported", tone: hookCount ? "good" : "muted" },
    { label: steps.length > 1 ? `${steps.length - 1} adjacent step pairs` : "single-step route", tone: steps.length > 1 ? "good" : "muted" },
    { label: enzymeSteps ? `${enzymeSteps} enzymatic step(s)` : "no enzymatic step in route", tone: enzymeSteps ? "good" : "muted" },
    { label: templateSteps ? `${templateSteps} template proposal(s)` : "no template proposal", tone: templateSteps ? "muted" : "warn" },
    { label: transitions.length ? `${transitions.length} chemo/enzyme transition(s)` : "no chemo/enzyme transition", tone: transitions.length ? "good" : "muted" },
  ];
  const label = steps.length > 1
    ? (enzymeSteps ? "cascade-aware mixed route" : "multi-step telescoping candidate")
    : "single-step route";
  return {
    label,
    tone: steps.length > 1 ? "good" : "muted",
    stepText: `${steps.length} step${steps.length === 1 ? "" : "s"}`,
    transitionText: transitions.length ? `${transitions.length} chemo/enzyme transition(s)` : "no domain transition",
    hookText: hookCount ? `${hookCount} cascade hooks active` : "hook metadata absent",
    signals,
  };
}

function renderSteps(steps) {
  $("step-count").textContent = String(steps.length);
  const table = $("step-table");
  table.innerHTML = "";
  steps.forEach((step) => {
    const row = document.createElement("div");
    row.className = "step-row";
    row.innerHTML = `
      <div class="step-row-grid">
        <span class="pill">#${step.index}</span>
        <div>
          <b>${escapeHtml(step.reaction_type || "unknown")}</b>
          <div class="muted">source=${escapeHtml(step.source || "")} ec=${escapeHtml(step.ec || "")} T=${fmt(step.T)} pH=${fmt(step.pH)}</div>
          <div class="smiles">${escapeHtml(step.reaction_smiles || "")}</div>
          <div class="muted">candidates=${step.candidate_pool?.n_candidates ?? 0}</div>
        </div>
      </div>
    `;
    table.appendChild(row);
  });
}

function renderMetrics(metrics, data = {}) {
  const wrap = $("metrics-view");
  wrap.innerHTML = "";
  if (!Object.keys(metrics).length) {
    $("metric-state").textContent = "empty";
    wrap.innerHTML = '<div class="empty-state">No metrics.</div>';
    return;
  }
  $("metric-state").textContent = "loaded";
  const natural = metrics.route_naturalness || {};
  const compat = metrics.cascade_compatibility || {};
  const cond = metrics.condition || {};
  const enz = metrics.enzyme_evidence || {};
  const progress = metrics.retrosynthesis_progress || {};
  const operation = metrics.operation_transitions || {};
  const candPool = metrics.candidate_pool || {};
  const rows = [
    ["solved", professionalSolved({ metrics })],
    ["diagnostic", diagnosticSolved({ metrics })],
    ["stock closed", metrics.route_solved],
    ["progressive", metrics.progressive_route],
    ["main reduction", progress.main_chain_reduction],
    ["leaf reduction", progress.largest_leaf_reduction],
    ["strict stock", metrics.strict_stock_solve],
    ["filled slots", metrics.filled_route],
    ["condition", cond.condition_window_success],
    ["compatibility", compat.cascade_compatibility_success],
    ["naturalness", natural.naturalness_score],
    ["enzyme coverage", enz.enzyme_evidence_coverage],
    ["operation score", operation.operation_score],
    ["candidate diversity", candPool.avg_pool_diversity_score],
    ["candidate coverage", candPool.candidate_pool_coverage],
    ["candidate dup rxn", candPool.avg_duplicate_reaction_fraction],
  ];
  const block = document.createElement("div");
  block.className = "metric-grid";
  block.innerHTML = rows.map(([k, v]) => metricCell(k, v)).join("");
  wrap.appendChild(block);

  const progressRow = document.createElement("div");
  progressRow.className = "metric-row";
  progressRow.innerHTML = `
    <b>Retrosynthesis Progress</b>
    <div class="smiles">target_atoms=${fmt(progress.target_heavy_atoms)} terminal_main_atoms=${fmt(progress.terminal_main_heavy_atoms)} largest_leaf_atoms=${fmt(progress.largest_leaf_heavy_atoms)} progressive_steps=${fmt(progress.progressive_steps)} / ${fmt(metrics.n_steps)} step_fraction=${fmtFraction(progress.progressive_step_fraction)} terminal_simplified=${boolText(progress.terminal_simplified)} leaf_simplified=${boolText(progress.leaf_simplified)}</div>
  `;
  wrap.appendChild(progressRow);

  if (Object.keys(operation).length) {
    const operationRow = document.createElement("div");
    operationRow.className = "metric-row";
    operationRow.innerHTML = `
      <b>Operation Transitions</b>
      <div class="smiles">classes=${(operation.step_classes || []).map(escapeHtml).join(" -> ")} chemo_bio=${fmt(operation.chemo_bio_transitions)} T_shifts=${fmt(operation.temperature_shifts)} pH_shifts=${fmt(operation.pH_shifts)} solvent_switches=${fmt(operation.solvent_switches)} operation_cost=${fmt(operation.operation_cost)} issues=${(operation.issues || []).map(escapeHtml).join(", ") || "none"}</div>
    `;
    wrap.appendChild(operationRow);
  }

  if (Object.keys(candPool).length) {
    const candidateRow = document.createElement("div");
    candidateRow.className = "metric-row";
    candidateRow.innerHTML = `
      <b>Candidate Pool</b>
      <div class="smiles">steps=${fmt(candPool.steps_with_candidates)} total=${fmt(candPool.total_candidates)} avg_per_step=${fmt(candPool.avg_candidates_per_step)} diversity=${fmt(candPool.avg_pool_diversity_score)} min_diversity=${fmt(candPool.min_pool_diversity_score)} dup_rxn=${fmt(candPool.avg_duplicate_reaction_fraction)} dup_reactants=${fmt(candPool.avg_duplicate_reactant_set_fraction)} single_reactant_set_steps=${fmt(candPool.single_reactant_set_steps)}</div>
    `;
    wrap.appendChild(candidateRow);
  }

  const issues = [
    ...(compat.issues || []),
    ...((natural.issues_by_step || []).flatMap((x) => x.issues || [])),
    ...(metrics.filled_route && !metrics.progressive_route ? ["insufficient_retrosynthesis_progress"] : []),
    ...(metrics.strict_stock_solve === false ? ["terminal_reactants_not_all_in_stock"] : []),
  ];
  const issueRow = document.createElement("div");
  issueRow.className = "metric-row";
  issueRow.innerHTML = `<b>Issues</b><div class="smiles">${issues.length ? issues.map(escapeHtml).join(", ") : "none"}</div>`;
  wrap.appendChild(issueRow);

  const diagnosis = derivedDiagnosis(data);
  if (diagnosis.length) {
    const diagnosisRow = document.createElement("div");
    diagnosisRow.className = "metric-row";
    diagnosisRow.innerHTML = `<b>Failure Diagnosis</b><div class="smiles">${diagnosis.map(escapeHtml).join(", ")}</div>`;
    wrap.appendChild(diagnosisRow);
  }

  const diversity = data.route_set_metrics?.diversity || {};
  if (Object.keys(diversity).length) {
    const diversityRow = document.createElement("div");
    diversityRow.className = "metric-row";
    diversityRow.innerHTML = `
      <b>Route Set Diversity</b>
      <div class="smiles">unique_full=${fmt(diversity.unique_full_signatures)} / ${fmt(diversity.n_routes)} unique_types=${fmt(diversity.unique_type_sequences)} unique_terminals=${fmt(diversity.unique_terminal_reactant_sets)} duplicate_fraction=${fmtFraction(diversity.duplicate_route_fraction)} type_distance=${fmt(diversity.mean_pairwise_type_distance)} terminal_distance=${fmt(diversity.mean_pairwise_terminal_jaccard_distance)}</div>
    `;
    wrap.appendChild(diversityRow);
  }
}

function planStatusHtml(data) {
  const status = normalizedSearchStatus(data);
  const attempts = data.depth_attempts || [];
  const diagnosis = derivedDiagnosis(data);
  if (!Object.keys(status).length && !attempts.length && !diagnosis.length) {
    return "";
  }
  const attemptRows = attempts.map((row) => {
    const best = row.best || {};
    const status = attemptStatus(row);
    return `
      <div class="depth-row">
        <span class="pill ${attemptClass(status)}">${escapeHtml(status || "pending")}</span>
        <b>d${escapeHtml(row.depth)}</b>
        <span>${escapeHtml(plannerDisplayName(row.planner))}</span>
        <span>${escapeHtml(row.n_routes ?? 0)} routes</span>
        <span>solved ${boolText(professionalSolved(best))}</span>
        <span>diag ${boolText(diagnosticSolved(best))}</span>
        <span>prog ${boolText(best.progressive_route)}</span>
        <span>red ${fmtFraction(best.main_chain_reduction)}</span>
        <span>leaf ${fmtFraction(best.largest_leaf_reduction)}</span>
        <span>${fmt(row.elapsed_s)}s</span>
      </div>
    `;
  }).join("");
  return `
    <div class="route-status">
      <div class="route-status-head">
        <b>${escapeHtml(status.status || "pending")}</b>
        <span class="muted">${escapeHtml(status.message || "")}</span>
      </div>
      ${attemptRows ? `<div class="depth-list">${attemptRows}</div>` : ""}
      ${diagnosis.length ? `<div class="smiles">diagnosis=${diagnosis.map(escapeHtml).join(", ")}</div>` : ""}
      ${failureAnalysisHtml(data)}
    </div>
  `;
}

function failureAnalysisHtml(data = {}) {
  const analysis = data.failure_analysis || {};
  if (!analysis.available) {
    return "";
  }
  const diagnosis = (analysis.diagnosis || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const suggestions = (analysis.retry_suggestions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const cfg = analysis.search_config || {};
  const cfgLine = [
    cfg.preset ? `preset=${cfg.preset}` : "",
    cfg.max_depth !== undefined ? `depth=${cfg.max_depth}` : "",
    cfg.iterations !== undefined ? `iter=${cfg.iterations}` : "",
    cfg.expansion_topk !== undefined ? `topk=${cfg.expansion_topk}` : "",
    cfg.condition_prediction_enabled ? "condition=on" : "condition=off",
    cfg.enzyme_assignment_enabled ? "enzyme=on" : "enzyme=off",
  ].filter(Boolean).join(" | ");
  return `
    <details class="route-explain failure-analysis" open>
      <summary>Failure Analysis</summary>
      <div class="smiles">${escapeHtml(cfgLine || "-")}</div>
      ${diagnosis ? `<ul>${diagnosis}</ul>` : ""}
      ${suggestions ? `<b>Retry suggestions</b><ul>${suggestions}</ul>` : ""}
    </details>
  `;
}

function failureRiskHtml(data) {
  const risk = data?.failure_risk || {};
  if (!risk.available) {
    return "";
  }
  const active = risk.active_labels || [];
  const suppressed = risk.suppressed_labels || [];
  const shown = active.length ? active : (risk.labels || []).slice(0, 3);
  const suggestions = risk.search_suggestions || [];
  const retry = risk.retry_policy || {};
  const changed = retry.changed_settings || {};
  if (!shown.length && !suggestions.length) {
    return "";
  }
  const labelRows = shown.map((item) => `
    <div>
      <b>${escapeHtml(formatRiskLabel(item.label))}</b>
      <span>p=${fmtFraction(item.probability)} support=${escapeHtml(item.support ?? "-")}</span>
    </div>
  `).join("");
  const actionRows = suggestions.map((item) => `
    <div>
      <b>${escapeHtml(formatRiskLabel(item.action))}</b>
      <span>${escapeHtml(item.rationale || "")}${item.budget_hint ? ` | budget ${escapeHtml(item.budget_hint)}` : ""}</span>
    </div>
  `).join("");
  const retryRows = Object.entries(changed).map(([key, value]) => `
    <div>
      <b>${escapeHtml(formatRiskLabel(key))}</b>
      <span>${escapeHtml(value)}${retry.automatic_retry_safe ? " | auto-safe" : ""}</span>
    </div>
  `).join("");
  const suppressedText = suppressed.length
    ? `<div class="smiles">suppressed=${suppressed.map((item) => escapeHtml(item.label)).join(", ")}</div>`
    : "";
  return `
    <details class="route-explain" open>
      <summary>Search Guidance</summary>
      <div class="explain-grid">
        ${labelRows || "<div><b>No active risk</b><span>Top learned risks are below threshold.</span></div>"}
        ${actionRows}
        ${retryRows}
      </div>
      ${retryRows ? '<button class="small-btn" type="button" data-apply-retry="1">Apply Retry Settings</button>' : ""}
      ${retry.note ? `<div class="smiles">retry_policy=${escapeHtml(retry.note)}</div>` : ""}
      ${suppressedText}
    </details>
  `;
}

function applyRetrySettings(settings) {
  const mapping = {
    max_steps: "max-steps",
  };
  Object.entries(mapping).forEach(([key, id]) => {
    if (settings[key] !== undefined && $(id)) {
      $(id).value = settings[key];
    }
  });
}

function routeSelectorHtml(data, selectedIndex) {
  const routes = data.routes || [];
  if (routes.length <= 1) {
    return "";
  }
  const buttons = routes.map((route, index) => {
    const metrics = route.metrics || {};
    const progress = metrics.retrosynthesis_progress || {};
    const active = index === selectedIndex ? "active" : "";
    const solved = professionalSolved(route);
    const diagnostic = diagnosticSolved(route);
    const statusClass = solved ? "solved" : diagnostic ? "diagnostic" : "";
    return `
      <button class="route-tab ${active} ${statusClass}" type="button" data-route-index="${index}">
        <b>Route ${index + 1}</b>
        <span>${boolText(solved)} solved</span>
        <span>${boolText(metrics.route_solved)} stock</span>
        <span>${boolText(metrics.progressive_route)} prog</span>
        <span>${fmtFraction(progress.main_chain_reduction)}</span>
      </button>
    `;
  }).join("");
  return `<div class="route-tabs">${buttons}</div>`;
}

function routeTreeHtml(route, target) {
  const steps = route.steps || [];
  if (!steps.length) {
    return "";
  }
  const byProduct = new Map();
  const stockBySmiles = new Map();
  steps.forEach((step) => {
    if (step.product) {
      byProduct.set(step.product, step);
    }
    Object.entries(step.stock_status || {}).forEach(([smi, value]) => {
      stockBySmiles.set(smi, value);
    });
  });
  const root = target || steps[0]?.product || "";
  const tree = renderTreeNode(root, byProduct, stockBySmiles, new Set());
  return `
    <details class="route-tree" open>
      <summary>Route Tree</summary>
      <div class="tree-body">${tree}</div>
    </details>
  `;
}

function renderTreeNode(smiles, byProduct, stockBySmiles, seen) {
  const key = smiles || "";
  const step = byProduct.get(key);
  if (!step || seen.has(key)) {
    const stock = stockBySmiles.get(key);
    const label = seen.has(key) ? "cycle" : stock === true ? "stock" : stock === false ? "not stock" : "leaf";
    return `<div class="tree-leaf"><span class="pill ${stockClass(stock)}">${label}</span><span class="smiles">${escapeHtml(key || "empty")}</span></div>`;
  }
  seen.add(key);
  const reactants = [step.main_reactant, ...(step.aux_reactants || [])].filter(Boolean);
  const children = reactants.map((smi) => renderTreeNode(smi, byProduct, stockBySmiles, new Set(seen))).join("");
  return `
    <div class="tree-node">
      <div class="tree-product"><span class="pill">step ${escapeHtml(step.index)}</span><span class="smiles">${escapeHtml(key)}</span></div>
      <div class="tree-children">${children || '<span class="muted">no reactants</span>'}</div>
    </div>
  `;
}

function routeWhyHtml(route) {
  const explanation = route.explanation || {};
  const uncertainty = explanation.uncertainty_table || {};
  const report = route.constraint_report || {};
  const lines = [
    ["why", explanation.why_selected],
    ["search", report.search_mode],
    ["score", route.score],
    ["confidence", route.confidence],
    ["expansions", uncertainty.expansions],
    ["generated", uncertainty.generated_reactions],
    ["pruned quality", uncertainty.pruned_by_route_quality],
    ["cache hits", uncertainty.candidate_cache_hits],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (!lines.length) {
    return "";
  }
  return `
    <details class="route-explain" open>
      <summary>Why This Route</summary>
      <div class="explain-grid">${lines.map(([k, v]) => `<div><b>${escapeHtml(k)}</b><span>${escapeHtml(v)}</span></div>`).join("")}</div>
    </details>
  `;
}

function reactionInsightHtml(step) {
  const info = step.reaction_interpretation || fallbackReactionInterpretation(step);
  const atomNotes = info.atom_change?.notes || [];
  const added = info.likely_added_or_removed || [];
  const catalysis = info.catalysis_and_conditions || [];
  return `
    <div class="reaction-insight">
      <div class="insight-card">
        <b>Reaction idea</b>
        <span>${escapeHtml(info.reaction_principle || info.forward_summary || "-")}</span>
      </div>
      <div class="insight-card">
        <b>What changes</b>
        <span>${escapeHtml([...atomNotes, ...added].join(" | ") || "No atom-count change detected from exported fields.")}</span>
      </div>
      <div class="insight-card">
        <b>Catalysis / conditions</b>
        <span>${escapeHtml(catalysis.join(" | ") || catalystFallback(step))}</span>
      </div>
    </div>
  `;
}

function fallbackReactionInterpretation(step) {
  const reactionType = step.reaction_type || "unknown";
  const aux = step.aux_reactants || [];
  const added = aux.length ? [`auxiliary reactant/coupling partner: ${aux.join(" . ")}`] : [];
  return {
    reaction_class: reactionType,
    forward_summary: `${reactionType} converts the displayed precursor into the product.`,
    reaction_principle: reactionPrincipleText(reactionType),
    likely_added_or_removed: added,
    catalysis_and_conditions: [catalystFallback(step)].filter(Boolean),
    atom_change: { notes: [] },
  };
}

function reactionPrincipleText(reactionType) {
  const key = String(reactionType || "").toLowerCase().replaceAll("_", " ");
  const rules = [
    ["glycosyl", "Glycosylation transfers a sugar unit from an activated donor and forms a glycosidic bond."],
    ["hydrolysis", "Hydrolysis uses water to cleave a labile bond such as an ester, amide, glycoside, or phosphate."],
    ["reduction", "Reduction adds hydride, hydrogen, or electrons and lowers oxidation state."],
    ["oxidation", "Oxidation removes hydrogen/electrons or introduces oxygen and raises oxidation state."],
    ["amination", "Amination installs or exchanges a nitrogen substituent through C-N bond formation."],
    ["acyl", "Acyl transfer moves an acyl group onto an O, N, S, or C nucleophile."],
    ["ester", "Esterification or transesterification forms or exchanges an ester linkage."],
    ["alkyl", "Alkylation installs an alkyl substituent through substitution or transfer chemistry."],
    ["methyl", "Methylation transfers a methyl group, often from SAM or a chemical methyl donor."],
    ["phosph", "Phosphorylation installs or transfers a phosphate group."],
    ["coupling", "Coupling joins two molecular fragments by forming a new bond."],
    ["c-c", "C-C bond formation joins carbon frameworks through coupling, addition, alkylation, or ligation."],
    ["deprotect", "Deprotection removes a protecting group to reveal a functional handle."],
    ["isomer", "Isomerization rearranges connectivity or stereochemistry with little atom-count change."],
  ];
  const hit = rules.find(([token]) => key.includes(token));
  return hit ? hit[1] : "General transformation; inspect the reaction SMILES and candidate evidence before assigning a precise mechanism.";
}

function catalystFallback(step) {
  const bits = [
    step.ec ? `EC ${step.ec}` : "",
    step.catalyst || "",
    step.solvent ? `solvent ${step.solvent}` : "",
    step.T !== null && step.T !== undefined ? `T=${fmt(step.T)} C` : "",
    step.pH !== null && step.pH !== undefined ? `pH=${fmt(step.pH)}` : "",
  ].filter(Boolean);
  return bits.join(" | ") || "No catalyst or condition field exported.";
}

function formatReactionName(value) {
  return String(value || "unknown reaction")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function isCascadePlannerPayload(data) {
  const backend = String((data.ui_metadata || {}).backend || "");
  return backend === "CascadePlanner" || backend === "ChemEnzyRetroPlanner";
}

function plannerPublicName(value) {
  const text = String(value || "");
  if (text === "ChemEnzyRetroPlanner" || text === "CascadePlanner") return "CascadePlanner";
  return text || "CascadePlanner";
}

function readableStepType(step) {
  if (step.ec || step.is_enzymatic) return "Enzymatic step";
  const raw = String(step.reaction_type || step.source || "");
  if (isRawTemplate(raw)) return "Template step";
  if (!raw || raw === "reaction") return "Retrosynthetic step";
  return formatReactionName(raw);
}

function readableStepSource(step) {
  const raw = String(step.source || "");
  const backend = String(step.evidence?.backend || "");
  if (step.ec || step.is_enzymatic) return "CascadePlanner enzyme module";
  if (isRawTemplate(raw) || step.reaction_type === "template") return "Template proposal";
  if (!raw || raw === "ChemEnzyRetroPlanner" || backend === "ChemEnzyRetroPlanner") return "CascadePlanner";
  return raw.replaceAll("_", " ");
}

function isRawTemplate(value) {
  const text = String(value || "");
  return text.length > 48 || text.includes(">>") || text.startsWith("[") || text.includes("[#");
}

function formatRiskLabel(value) {
  return String(value || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function routeStepOutcome(step) {
  const stockValues = step.stock_status ? Object.values(step.stock_status) : [];
  if (stockValues.length > 0 && stockValues.every(Boolean)) return "stock-ready precursor";
  if (step.ec) return "enzyme candidate";
  return step.source || "candidate";
}

function stepWhyHtml(step) {
  const scores = step.scores || {};
  const stock = step.stock_status || {};
  const stockText = Object.entries(stock).map(([smi, ok]) => `${ok ? "stock" : "not stock"}:${smi}`).join(" | ");
  const evidence = step.evidence || {};
  const evidenceBits = [
    evidence.uniprot_accession ? `UniProt ${evidence.uniprot_accession}` : "",
    evidence.doi ? `DOI ${evidence.doi}` : "",
    evidence.literature_precedent ? "literature precedent" : "",
  ].filter(Boolean).join(" | ");
  return `
    <details class="step-detail">
      <summary>Selection evidence</summary>
      <div class="explain-grid">
        <div><b>source</b><span>${escapeHtml(step.source || "-")}</span></div>
        <div><b>type / EC</b><span>${escapeHtml(step.reaction_type || "-")} / ${escapeHtml(step.ec || "-")}</span></div>
        <div><b>scores</b><span>retro=${fmt(scores.retro)} enzyme=${fmt(scores.enzyme)} condition=${fmt(scores.condition)} confidence=${fmt(scores.confidence)}</span></div>
        <div><b>conditions</b><span>T=${fmt(step.T)} pH=${fmt(step.pH)} ${escapeHtml(step.solvent || "")}</span></div>
        <div><b>stock</b><span>${escapeHtml(stockText || "-")}</span></div>
        <div><b>evidence</b><span>${escapeHtml(evidenceBits || "-")}</span></div>
      </div>
      <div class="smiles">${escapeHtml(step.reaction_smiles || "")}</div>
    </details>
  `;
}

function candidatePoolHtml(step) {
  const pool = step.candidate_pool || {};
  const candidates = pool.top_candidates || [];
  if (!candidates.length) {
    return "";
  }
  const poolMetrics = [
    ["diversity", pool.pool_diversity_score],
    ["unique rxn", pool.unique_reactions],
    ["unique main", pool.unique_main_reactants],
    ["unique reactants", pool.unique_reactant_sets],
    ["dup rxn", pool.duplicate_reaction_fraction],
    ["dup reactants", pool.duplicate_reactant_set_fraction],
  ];
  const rows = candidates.map((cand, index) => candidateRowHtml(cand, index)).join("");
  return `
    <details class="candidate-panel">
      <summary>Alternative candidates (${escapeHtml(pool.n_candidates ?? candidates.length)})</summary>
      <div class="candidate-pool-metrics">
        ${poolMetrics.map(([label, value]) => `
          <div class="mini-metric">
            <b>${escapeHtml(fmt(value))}</b>
            <span>${escapeHtml(label)}</span>
          </div>
        `).join("")}
      </div>
      <div class="candidate-list">${rows}</div>
    </details>
  `;
}

function candidateRowHtml(cand, index) {
  const evidence = cand.evidence || {};
  const reactants = [cand.main_reactant, ...(cand.aux_reactants || [])].filter(Boolean).join(" . ");
  const bits = [
    cand.source || "candidate",
    cand.reaction_type || cand.type || "",
    cand.ec || "",
    cand.score !== null && cand.score !== undefined ? `score=${fmt(cand.score)}` : "",
    cand.value_score !== null && cand.value_score !== undefined ? `value=${fmt(cand.value_score)}` : "",
    cand.value_probability !== null && cand.value_probability !== undefined ? `p=${fmtFraction(cand.value_probability)}` : "",
    cand.candidate_ranker_score !== null && cand.candidate_ranker_score !== undefined ? `ranker=${fmtFraction(cand.candidate_ranker_score)}` : "",
    cand.uniprot_accession || evidence.uniprot_accession || "",
    cand.doi || evidence.doi || "",
  ].filter(Boolean).join(" | ");
  return `
    <div class="candidate-row">
      <div class="candidate-rank">#${index + 1}</div>
      <div>
        <div class="muted">${escapeHtml(bits)}</div>
        <div class="smiles">${escapeHtml(reactants || cand.main_reactant || "")}</div>
        <div class="smiles">${escapeHtml(cand.reaction_smiles || "")}</div>
      </div>
    </div>
  `;
}

function molCard(label, smiles) {
  const safe = escapeHtml(smiles || "");
  const src = smiles ? `/api/mol.svg?smiles=${encodeURIComponent(smiles)}&w=320&h=160` : "";
  return `
    <div class="mol-card">
      <div class="mol-label">${escapeHtml(label)}</div>
      ${src ? `<img src="${src}" alt="${safe}">` : ""}
      <div class="smiles">${safe || "empty"}</div>
    </div>
  `;
}

function metricCell(label, value) {
  const numeric = Number(value);
  const hasNumber = Number.isFinite(numeric);
  const width = hasNumber ? Math.max(0, Math.min(100, numeric * 100)) : 0;
  const shown = typeof value === "boolean" || value === null || value === undefined ? boolText(value) : String(value);
  return `
    <div class="metric-row">
      <b>${escapeHtml(shown)}</b>
      <div class="muted">${escapeHtml(label)}</div>
      ${hasNumber ? `<div class="bar"><span style="width:${width}%"></span></div>` : ""}
    </div>
  `;
}

function routeOutcome(subject) {
  const metrics = subject?.metrics || subject || {};
  if (professionalSolved(subject)) {
    return { label: "solved", tone: "good", className: "is-solved" };
  }
  if (diagnosticSolved(subject)) {
    return { label: "diagnostic", tone: "warn", className: "is-diagnostic" };
  }
  if (metrics.progressive_route) {
    return { label: "partial", tone: "warn", className: "is-partial" };
  }
  if (metrics.filled_route) {
    return { label: "filled", tone: "muted", className: "is-filled" };
  }
  if (metrics.route_solved) {
    return { label: "stock-closed", tone: "warn", className: "is-diagnostic" };
  }
  return { label: "no route", tone: "bad", className: "is-failed" };
}

function plannerDisplayName(value) {
  const key = String(value || "").toLowerCase();
  const names = {
    advanced: "Advanced",
    advanced_andor: "Advanced AND-OR",
    "advanced_andor+cc_aostar": "Advanced AND-OR + AO*",
    advanced_cc_aostar: "Advanced AO*",
    "advanced_cc_aostar+stock_rescue": "Advanced AO* + stock rescue",
    stock_closed_andor: "Stock-closed AND-OR",
    cc_aostar: "AO* rescue",
    cascade_fallback: "AO* rescue",
    hybrid: "Advanced",
    and_or: "Advanced",
    cascade: "Advanced",
  };
  return names[key] || String(value || "-").replaceAll("_", " ");
}

function statusLabel(status) {
  if (status === "solved") return "Solved";
  if (status === "partial") return "Partial";
  if (status === "diagnostic") return "Diagnostic";
  if (status === "filtered") return "Filtered";
  if (status === "failed") return "Failed";
  return status || "Idle";
}

function pillTone(status) {
  if (status === "solved") return "good";
  if (status === "partial" || status === "diagnostic" || status === "filtered") return "warn";
  if (status === "failed") return "bad";
  return "muted";
}

function formatModeName(value) {
  const text = String(value || "")
    .replace(/^gt_/, "GT ")
    .replace(/_/g, " ")
    .trim();
  return text ? text.replace(/\bdepth\b/i, "depth") : "Target";
}

function artifactTitle(name) {
  return String(name || "")
    .replace(/\.json$|\.md$|\.csv$/i, "")
    .replace(/^ui_plan_/, "Route search ")
    .replace(/^ui_smoke_/, "UI smoke ")
    .replace(/^gt_direct_candidate_recall_/, "Candidate recall ")
    .replace(/_/g, " ");
}

function artifactKind(item) {
  const name = item?.name || "";
  if (name.endsWith(".md")) return "report";
  if (name.endsWith(".csv")) return "table";
  if (name.includes("benchmark") || name.includes("depth")) return "benchmark";
  if (name.includes("plan")) return "route";
  return "json";
}

function fmtBenchmark(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return n <= 1 ? `${Math.round(n * 100)}%` : n.toFixed(2);
}

function normalizedSearchStatus(data) {
  const raw = data?.search_status || {};
  const routes = data?.routes || [];
  const filter = productAuditFilteredAll(data);
  if (!routes.length && filter.filteredAll) {
    return {
      ...raw,
      status: "filtered",
      solved: false,
      native_returned_routes: true,
      post_filter_removed_all: true,
      message: raw.message || `ChemEnzy returned ${filter.original} route(s), but product-audit hid all of them`,
    };
  }
  if (!routes.length) {
    return raw;
  }
  const solved = routes.some((route) => professionalSolved(route));
  const diagnostic = routes.some((route) => diagnosticSolved(route));
  const stockClosed = routes.some((route) => Boolean((route.metrics || {}).route_solved));
  const progressive = routes.some((route) => Boolean((route.metrics || {}).progressive_route));
  const status = solved ? "solved" : progressive ? "partial" : diagnostic ? "diagnostic" : "failed";
  const bestDepth = raw.best_depth ?? routes[0]?.n_steps ?? null;
  return {
    ...raw,
    status,
    solved,
    diagnostic,
    stock_closed: stockClosed,
    progressive,
    best_depth: bestDepth,
    message: statusMessage(status, bestDepth, raw.message),
  };
}

function statusMessage(status, depth, fallback) {
  if (status === "solved") {
    return depth ? `solved at depth ${depth}` : "solved";
  }
  if (status === "partial") {
    return "progressive route found, but terminal reactants are not solved";
  }
  if (status === "diagnostic") {
    return "stock-closed diagnostic route found, but it is not a progressive retrosynthesis";
  }
  return fallback || "no solved retrosynthesis route found within the searched depth range";
}

function derivedDiagnosis(data) {
  const raw = data?.failure_diagnosis || [];
  if (raw.length) {
    return Array.from(new Set([...raw, ...recoveryDiagnosisLabels(data?.route_recovery || {})]));
  }
  const route = (data?.routes || [])[0];
  if (!route) {
    return recoveryDiagnosisLabels(data?.route_recovery || {});
  }
  const metrics = route.metrics || {};
  const progress = metrics.retrosynthesis_progress || {};
  const compat = metrics.cascade_compatibility || {};
  const reasons = [];
  const add = (reason) => {
    if (reason && !reasons.includes(reason)) reasons.push(reason);
  };
  if (diagnosticSolved(route)) add("diagnostic_stock_closed_but_not_progressive");
  if (metrics.filled_route && !metrics.progressive_route) add("insufficient_retrosynthesis_progress");
  if (Number(progress.main_chain_reduction || 0) === 0) add("main_chain_not_reduced");
  if (progress.terminal_simplified === false) add("terminal_main_reactant_still_complex");
  if (progress.leaf_simplified === false) add("largest_leaf_reactant_still_complex");
  if (metrics.strict_stock_solve === false) add("terminal_reactants_not_all_in_stock");
  (compat.issues || []).forEach(add);
  recoveryDiagnosisLabels(data?.route_recovery || {}).forEach(add);
  return reasons;
}

function recoveryDiagnosisLabels(recovery) {
  const bottleneck = recovery?.recovery_bottleneck || "";
  const labels = recovery?.recovery_bottleneck_labels || [];
  const out = [];
  const add = (value) => {
    if (value && value !== "recovered_exact_route" && !out.includes(value)) out.push(value);
  };
  add(bottleneck);
  labels.forEach(add);
  return out;
}

function attemptStatus(row) {
  const best = row?.best || {};
  if (professionalSolved(best)) return "solved";
  if (diagnosticSolved(best)) return "diagnostic";
  if (best.progressive_route) return "progressive";
  if (best.filled_route) return "filled_only";
  return row?.status || "partial";
}

function routeMetaClass(status) {
  if (status === "solved") return "pill good";
  if (status === "partial" || status === "diagnostic" || status === "filtered") return "pill warn";
  if (status === "failed") return "pill bad";
  return "pill muted";
}

function productAuditFilteredAll(data = {}) {
  const pf = data.post_filter || data.ui_metadata?.product_audit_post_filter || {};
  const original = Number(pf.original_route_count || 0);
  const kept = Number(pf.kept_route_count ?? (data.routes?.length ?? 0));
  const removed = Number(pf.removed_route_count || 0);
  return {
    filteredAll: original > 0 && kept === 0 && removed >= original,
    original,
    removed,
    mode: pf.mode || "",
    issueCounts: pf.issue_counts_removed || pf.issue_counts_before || {},
    classCounts: pf.route_class_counts_removed || pf.route_class_counts_before || {},
    rejectedPath: pf.rejected_saved_at || data.ui_metadata?.rejected_saved_at || "",
  };
}

function emptyRoutePanelLabel(data = {}, status = {}) {
  if (status.status === "filtered" || productAuditFilteredAll(data).filteredAll) {
    return "filtered by audit";
  }
  return "no route";
}

function emptyRouteHtml(data = {}) {
  const filter = productAuditFilteredAll(data);
  if (filter.filteredAll) {
    const analysis = data.failure_analysis || {};
    const target = analysis.target_complexity || {};
    const issueRows = topCounterEntries(filter.issueCounts, 6).map(([label, count]) => `
      <span class="audit-chip bad">${escapeHtml(formatRiskLabel(label))}<b>${escapeHtml(count)}</b></span>
    `).join("");
    const classRows = topCounterEntries(filter.classCounts, 4).map(([label, count]) => `
      <span class="audit-chip muted">${escapeHtml(formatRiskLabel(label))}<b>${escapeHtml(count)}</b></span>
    `).join("");
    const diagnosis = (analysis.diagnosis || []).slice(0, 4).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    const targetBits = target.available ? [
      `heavy atoms=${target.heavy_atoms}`,
      `rings=${target.rings}`,
      `chiral centers=${target.chiral_centers}`,
      target.natural_product_like ? "large natural-product-like target" : "",
    ].filter(Boolean).join(" | ") : "";
    const rejected = filter.rejectedPath
      ? `<button class="small-btn" type="button" data-open-job-output="${escapeHtml(filter.rejectedPath)}">打开 rejected artifact</button>`
      : "";
    return `
      <section class="audit-empty-state">
        <div class="audit-empty-head">
          <div>
            <b>Raw routes found, all rejected by audit</b>
            <span>ChemEnzy 返回了 ${escapeHtml(filter.original)} 条候选，但 ${escapeHtml(filter.removed)} 条都被产物物料守恒审计判为 severe artifact。</span>
          </div>
          <span class="pill warn">audit-filtered</span>
        </div>
        <div class="audit-chip-list">
          ${issueRows || '<span class="audit-chip muted">no issue counts</span>'}
        </div>
        ${classRows ? `<div class="audit-chip-list compact">${classRows}</div>` : ""}
        ${targetBits ? `<div class="smiles">target_complexity=${escapeHtml(targetBits)}</div>` : ""}
        ${diagnosis ? `<ul class="audit-diagnosis">${diagnosis}</ul>` : ""}
        <div class="audit-actions">
          ${rejected}
          <span>这些 rejected 记录用于调试，不应作为可汇报合成路线。</span>
        </div>
      </section>
    `;
  }
  return '<div class="empty-state">No route returned.</div>';
}

function topCounterEntries(counts = {}, limit = 5) {
  return Object.entries(counts || {})
    .map(([label, value]) => [label, Number(value)])
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => Number(b[1]) - Number(a[1]) || String(a[0]).localeCompare(String(b[0])))
    .slice(0, limit);
}

function attemptClass(status) {
  if (status === "solved") return "good";
  if (status === "diagnostic" || status === "progressive" || status === "filled_only") return "warn";
  if (status === "no_route") return "bad";
  return "muted";
}

function professionalSolved(subject) {
  const metrics = subject?.metrics || subject || {};
  if (metrics.professional_solved !== undefined) {
    return metrics.professional_solved === true;
  }
  return Boolean(metrics.route_solved && metrics.progressive_route);
}

function diagnosticSolved(subject) {
  const metrics = subject?.metrics || subject || {};
  if (metrics.diagnostic_solved !== undefined) {
    return metrics.diagnostic_solved === true;
  }
  return Boolean(metrics.route_solved && !professionalSolved(subject));
}

function stockClass(value) {
  if (value === true) return "good";
  if (value === false) return "bad";
  return "muted";
}

function showError(message) {
  $("route-view").className = "";
  $("route-view").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  $("route-meta").textContent = "error";
  $("route-meta").className = "pill bad";
}

function boolText(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  if (value === null || value === undefined) return "-";
  return String(value);
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(1) : String(value);
}

function fmtFraction(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n * 100)}%` : String(value);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvents() {
  $("nav-demo").addEventListener("click", openCascadeDemo);
  $("nav-plan").addEventListener("click", () => {
    showRouteSearchHome();
  });
  $("nav-benchmark").addEventListener("click", () => {
    showBenchmarkHome();
  });
  $("nav-artifacts").addEventListener("click", async () => {
    showArtifactsHome();
    await loadArtifacts();
  });
  $("toggle-artifacts").addEventListener("click", async () => {
    setArtifactsOpen(!state.artifactsOpen);
    if (state.artifactsOpen) {
      await loadArtifacts();
    }
  });
  $("run-plan").addEventListener("click", runPlan);
  $("cancel-plan").addEventListener("click", cancelCurrentPlan);
  $("run-eval").addEventListener("click", runEval);
  $("refresh-artifacts").addEventListener("click", loadArtifacts);
  $("refresh-jobs").addEventListener("click", loadJobs);
  $("open-cascade-demo").addEventListener("click", openCascadeDemo);
  $("search-preset").addEventListener("change", () => applyRoutePreset($("search-preset").value));
  $("expand-details").addEventListener("click", () => setDetailsOpen(true));
  $("collapse-details").addEventListener("click", () => setDetailsOpen(false));
  $("sample-statin").addEventListener("click", () => {
    const index = Math.floor(Math.random() * statinSamples.length);
    $("target-smiles").value = statinSamples[index];
    $("search-preset").value = "quick";
    applyRoutePreset("quick");
  });
}

function setDetailsOpen(isOpen) {
  document.querySelectorAll("#route-view details").forEach((node) => {
    node.open = isOpen;
  });
}

async function boot() {
  bindEvents();
  applyRoutePreset($("search-preset").value || "quick");
  showRouteSearchHome();
  loadStatus();
  loadJobs();
}

boot();
