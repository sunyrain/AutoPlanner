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
    rounds: 6,
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
};

const LAST_JOB_KEY = "autoplanner.agent.lastJobId";

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

async function startRun() {
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

async function verifyRouteFile(path) {
  const url = resultFileUrl(path);
  const res = await fetch(`${url}&_=${Date.now()}`, { cache: "no-store" });
  const contentType = res.headers.get("content-type") || "";
  const text = await res.text();
  const looksLikeHtml = /<html[\s>]/i.test(text) || /<!doctype html>/i.test(text);
  if (!res.ok || !looksLikeHtml || contentType.includes("application/json")) {
    let message = text.slice(0, 220).trim();
    try {
      const data = JSON.parse(text);
      message = data.error || data.text || message;
    } catch {
      // Keep the short raw preview for non-JSON responses.
    }
    throw new Error(message || `HTTP ${res.status}`);
  }
  return url;
}

function showRouteMessage(kind, title, detail, path = "") {
  const panel = document.querySelector(".route-panel");
  panel.classList.toggle("loading-route", kind === "loading");
  panel.classList.toggle("route-error", kind === "error");
  panel.classList.remove("has-route");
  $("route-empty").innerHTML = `
    <div class="route-message">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(detail)}</span>
      ${path ? `<code>${escapeHtml(path)}</code>` : ""}
    </div>
  `;
}

async function loadRoute(path) {
  const normalizedPath = normalizeResultPath(path);
  if (!normalizedPath.endsWith(".html")) {
    showRouteMessage("error", "路线图路径无效", "当前产物不是可嵌入的 HTML 路线图。", normalizedPath);
    return;
  }
  const panel = document.querySelector(".route-panel");
  const frame = $("route-frame");
  state.routeLoadingPath = normalizedPath;
  showRouteMessage("loading", "正在载入路线图", "如果 agent 仍在写入文件，这里会自动重试。", normalizedPath);
  let url;
  try {
    url = await verifyRouteFile(normalizedPath);
  } catch (err) {
    state.routeLoadingPath = "";
    state.routeFailedPath = normalizedPath;
    state.routeFailedAt = Date.now();
    frame.removeAttribute("src");
    $("open-route").href = "#";
    $("open-route").classList.add("disabled");
    showRouteMessage("error", "路线图暂时不可用", err.message || "后端没有返回可渲染的 HTML。", normalizedPath);
    return;
  }
  state.routeLoadingPath = "";
  state.routeFailedPath = "";
  state.loadedRoutePath = normalizedPath;
  frame.onload = () => {
    try {
      const doc = frame.contentDocument;
      doc?.body?.classList.add("embedded-route");
      doc?.defaultView?.scrollTo(0, 0);
      const canvas = doc?.querySelector(".route-canvas");
      if (canvas) {
        canvas.scrollTop = 0;
        canvas.scrollLeft = 0;
      }
    } catch {
      // Same-origin should hold for local result files, but keep the shell resilient.
    }
  };
  frame.src = `${url}&_=${Date.now()}`;
  $("open-route").href = url;
  $("open-route").classList.remove("disabled");
  if (!state.currentJobId) {
    $("route-state").textContent = "已载入";
    $("route-counts").textContent = normalizedPath;
  }
  panel.classList.remove("loading-route", "route-error");
  panel.classList.add("has-route");
}

function clearRoute() {
  state.loadedRoutePath = "";
  state.routeLoadingPath = "";
  state.routeFailedPath = "";
  state.routeFailedAt = 0;
  $("route-frame").removeAttribute("src");
  $("open-route").href = "#";
  $("open-route").classList.add("disabled");
  const panel = document.querySelector(".route-panel");
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
  $("refresh-status").addEventListener("click", refreshStatus);
  $("start-run").addEventListener("click", startRun);
  $("cancel-run").addEventListener("click", cancelRun);
  $("load-existing").addEventListener("click", loadExisting);
  $("demo-route").addEventListener("change", () => {
    $("existing-route").value = $("demo-route").value;
  });
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => applySample(button.dataset.sample));
  });
}

async function init() {
  bind();
  refreshStatus();
  const restored = await restoreActiveJob();
  if (!restored) {
    loadExisting();
  }
}

init();
