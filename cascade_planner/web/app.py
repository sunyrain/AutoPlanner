"""Flask web UI for AutoPlanner/CascadeBoard.

The server intentionally stays thin: it wraps the existing planner and
benchmark entry points, serves a static single-page UI, and stores generated
artifacts under results/v2 so command-line and web workflows share files.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import html
import hmac
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from cascade_planner.agent.case_blackboard import load_blackboard
from cascade_planner.agent.case_trace import load_case_bundle
from cascade_planner.agent.chem_enzy_policy import (
    apply_chem_enzy_search_policy,
    compile_chem_enzy_search_policy,
    compile_strategic_operator_from_case_bundle,
)
from cascade_planner.agent.codex_worker import run_codex_worker, worker_task_from_dict
from cascade_planner.agent.route_auditor import audit_route_package
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.baselines.proposal_gate import (
    ProposalGateConfig,
    gate_web_route,
    normalize_proposal_gate_mode,
    summarize_route_gate_reports,
)
from cascade_planner.baselines.chem_enzy_runtime import (
    chem_enzy_runtime_selection_from_request,
    diagnose_chem_enzy_runtime,
    format_chem_enzy_runtime_diagnostic,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.baselines.template_relevance_runtime import check_template_relevance
from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller
from cascade_planner.harness.tools import HarnessBudget
from cascade_planner.runtime.artifact_revision import (
    ArtifactRevisionError,
    load_latest_closeout_decision,
    load_latest_closeout_manifest,
    validate_latest_closeout_revision,
)


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = ROOT / "results" / "v2"
SHARED_RESULTS_DIR = ROOT / "results" / "shared"
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
TRUSTED_STOCK_CATALOGS_CONFIG = CONFIG_DIR / "trusted_stock_catalogs.json"
STATIN_SHOWCASE_PATH = ROOT / "results" / "shared" / "statin_panel_20260520" / "web_showcase" / "statin_showcase_routes.json"
ROUTE_EXAMPLE_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "artemisinin",
        "label": "artemisinin 多文献探索（51 分支，未闭合）",
        "path": "results/shared/full_rerun_advisory_visual_20260702/artemisinin/route_forest.html",
    },
    {
        "key": "paclitaxel",
        "label": "paclitaxel Architecture V2（96 分支，压力案例，未闭合）",
        "path": "results/shared/paclitaxel_architecture_v2_20260710/route_forest.html",
    },
)
ROUTE_EXAMPLE_MAX_DEPTH = 8
ROUTE_EXAMPLE_MAX_ENTRIES_PER_ROOT = 50_000
ROUTE_EXAMPLE_MAX_RESULTS = 256
DEFAULT_MODEL = "results/shared/skeleton_inpainter/best.pt"
MAX_SKELETON_STEPS = 8
DEFAULT_PLANNER_MODE = "advanced"
CHEMENZY_NATIVE_BACKENDS = {"chem_enzy", "chem_enzy_native", "chemenzy", "chemenzy_native"}
CODEX_FULLFLOW_BACKENDS = {"codex", "codex_fullflow", "codex_search", "bufotalin_codex_fullflow"}
CODEX_RUN_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {
        "rounds": 1, "depth": 1, "accepted_expansions": 2,
        "attempt_runs": 3, "per_invocation": 1, "attempts_per_invocation": 1,
        "chem_enzy_runs": 1, "child_target_runs": 1,
        "codex_research_runs": 0, "scout_calls": 1, "visual_calls": 0,
    },
    "standard": {
        "rounds": 4, "depth": 2, "accepted_expansions": 8,
        "attempt_runs": 12, "per_invocation": 1, "attempts_per_invocation": 1,
        "chem_enzy_runs": 1, "child_target_runs": 2,
        "codex_research_runs": 1, "scout_calls": 1, "visual_calls": 1,
    },
    "deep": {
        # Deep broadens deterministic search and tool work, but deliberately
        # keeps the same default model-backed campaign envelope as standard.
        "rounds": 6, "depth": 4, "accepted_expansions": 8,
        "attempt_runs": 12, "per_invocation": 1, "attempts_per_invocation": 1,
        "chem_enzy_runs": 2, "child_target_runs": 3,
        "codex_research_runs": 1, "scout_calls": 2, "visual_calls": 1,
    },
}

_RETRO_ENGINE: dict[str, Any] | None = None
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_JOBS: dict[str, dict[str, Any]] = {}
_CUDA_STATUS_CACHE: tuple[float, dict[str, Any]] | None = None
_ARTIFACT_SUMMARY_CACHE: tuple[float, dict[str, int]] | None = None
_STATIN_SHOWCASE_CACHE: tuple[float, dict[str, Any]] | None = None
_LOCK = threading.Lock()
_PLAN_JOB_QUEUE: deque[str] = deque()
_PLAN_WORKER_THREAD: threading.Thread | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_PLAN_CURRENT_JOB_ID: str | None = None
_PLAN_PROCESS_BY_JOB: dict[str, subprocess.Popen] = {}
_TERMINAL_JOB_STATUSES = {"complete", "failed", "cancelled"}


class _PlanJobCancelled(RuntimeError):
    pass


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.before_request
    def protect_mutating_api() -> None:
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        if not request.is_json:
            abort(415, description="mutating API requests require application/json")
        configured_token = str(os.environ.get("AUTOPLANNER_WEB_API_TOKEN") or "")
        if configured_token:
            supplied = str(request.headers.get("X-Autoplanner-Token") or "")
            if not hmac.compare_digest(configured_token, supplied):
                abort(401, description="missing or invalid API token")
        fetch_site = str(request.headers.get("Sec-Fetch-Site") or "").lower()
        if fetch_site in {"cross-site", "same-site"}:
            abort(403, description="cross-site mutation rejected")
        origin = str(request.headers.get("Origin") or "").rstrip("/")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403, description="origin does not match this AutoPlanner service")
        return None

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/agent")
    def agent_workbench():
        return send_from_directory(STATIC_DIR, "agent.html")

    @app.get("/statins")
    def statins_showcase():
        return send_from_directory(STATIC_DIR, "statins.html")

    @app.get("/api/status")
    def status():
        return jsonify({
            "ok": True,
            "root": str(ROOT),
            "model_exists": (ROOT / DEFAULT_MODEL).exists(),
            "retrochimera_model_exists": (ROOT / "data_external/retrochimera_model").exists(),
            "cuda": _cuda_status(),
            "chem_enzy_runtime": _chem_enzy_runtime_status(),
            "template_relevance": _template_relevance_status(),
            "artifacts": _artifact_summary(),
        })

    @app.get("/api/artifacts")
    def artifacts():
        return jsonify({"artifacts": _list_artifacts(filter_kind=request.args.get("filter"))})

    @app.get("/api/route-examples")
    def route_examples():
        return jsonify(_route_examples_payload())

    @app.post("/api/cases")
    def create_case_api():
        payload = request.get_json(force=True, silent=False) or {}
        return jsonify(_api_run_smiles_first_case(payload))

    @app.get("/api/blackboard")
    def blackboard_api():
        case_bundle = request.args.get("case_bundle")
        blackboard = request.args.get("blackboard")
        return jsonify(_api_inspect_case_or_blackboard(case_bundle=case_bundle, blackboard=blackboard))

    @app.post("/api/route-audit")
    def route_audit_api():
        payload = request.get_json(force=True, silent=False) or {}
        return jsonify(_api_route_audit(payload))

    @app.post("/api/worker-trace")
    def worker_trace_api():
        payload = request.get_json(force=True, silent=False) or {}
        return jsonify(_api_worker_trace(payload))

    @app.post("/api/guided-policy")
    def guided_policy_api():
        payload = request.get_json(force=True, silent=False) or {}
        return jsonify(_api_guided_policy(payload))

    @app.get("/api/final-report")
    def final_report_api():
        case_bundle = request.args.get("case_bundle")
        if not case_bundle:
            abort(400, description="case_bundle is required")
        return jsonify(_api_final_report(case_bundle))

    @app.get("/api/cascade-demo")
    def cascade_demo():
        return jsonify(_cascade_demo_payload())

    @app.get("/api/statins")
    def statins_api():
        return jsonify(_statin_showcase_public_payload())

    @app.get("/api/statins/route/<target_key>/<int:route_index>")
    def statin_route_doc(target_key: str, route_index: int):
        payload = _load_statin_showcase_payload()
        target = _find_statin_showcase_target(payload, target_key)
        route = _find_statin_showcase_route(target, route_index)
        return jsonify(
            {
                "ok": True,
                "target": _statin_showcase_target_summary(target, include_routes=False),
                "route": route,
            }
        )

    @app.get("/api/statins/route-svg/<target_key>/<int:route_index>")
    def statin_route_svg(target_key: str, route_index: int):
        payload = _load_statin_showcase_payload()
        target = _find_statin_showcase_target(payload, target_key)
        route = _find_statin_showcase_route(target, route_index)
        try:
            from scripts.render_linear_route_schemes import render_scheme_svg

            svg = render_scheme_svg(
                route,
                route_number=int(route.get("display_rank") or route.get("rank") or route_index),
                target_smiles=str(target.get("target_smiles") or ""),
                mol_width=_as_int(request.args.get("mol_w"), 230, lo=120, hi=360),
                mol_height=_as_int(request.args.get("mol_h"), 150, lo=90, hi=260),
                steps_per_row=_as_int(request.args.get("steps_per_row"), 4, lo=2, hi=6),
                aux_mode=str(request.args.get("aux_mode") or "mini"),
            )
        except Exception as exc:
            svg = _statin_showcase_error_svg(f"Route SVG render failed: {type(exc).__name__}: {exc}")
        return Response(svg, mimetype="image/svg+xml")

    @app.get("/api/artifact")
    def artifact():
        rel_path = request.args.get("path", "")
        path = _safe_path(rel_path, allowed_roots=[RESULTS_DIR, SHARED_RESULTS_DIR, DATA_DIR])
        if not path.exists() or not path.is_file():
            abort(404)
        if path.suffix.lower() == ".json":
            return jsonify(json.loads(path.read_text(encoding="utf-8")))
        return Response(path.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain")

    @app.get("/api/result-file")
    def result_file():
        rel_path = request.args.get("path", "")
        path = _safe_path(rel_path, allowed_roots=[RESULTS_DIR, SHARED_RESULTS_DIR, DATA_DIR])
        if not path.exists() or not path.is_file():
            abort(404)
        suffix = path.suffix.lower()
        mimetype = {
            ".html": "text/html; charset=utf-8",
            ".htm": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".jsonl": "application/x-ndjson; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".pdf": "application/pdf",
            ".txt": "text/plain; charset=utf-8",
            ".log": "text/plain; charset=utf-8",
            ".md": "text/markdown; charset=utf-8",
        }.get(suffix, "application/octet-stream")
        if request.method == "HEAD":
            response = Response(mimetype=mimetype)
            response.content_length = path.stat().st_size
        elif mimetype.startswith("text/") or "json" in mimetype:
            response = Response(path.read_text(encoding="utf-8", errors="replace"), mimetype=mimetype)
        else:
            response = Response(path.read_bytes(), mimetype=mimetype)
        if suffix in {".html", ".htm"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'"
            )
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/mol.svg")
    def mol_svg():
        smiles = request.args.get("smiles", "")
        width = _as_int(request.args.get("w"), 260, lo=80, hi=800)
        height = _as_int(request.args.get("h"), 180, lo=80, hi=600)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            abort(400, description="invalid SMILES")
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Response(drawer.GetDrawingText(), mimetype="image/svg+xml")

    @app.post("/api/plan")
    def plan():
        payload = request.get_json(force=True, silent=False) or {}
        result = _run_plan(payload)
        return jsonify(result)

    @app.post("/api/plan-jobs")
    def plan_job():
        payload = request.get_json(force=True, silent=False) or {}
        job = _start_plan_job(payload)
        return jsonify(job)

    @app.post("/api/evaluate")
    def evaluate():
        payload = request.get_json(force=True, silent=False) or {}
        job = _start_eval_job(payload)
        return jsonify(job)

    @app.get("/api/jobs")
    def jobs():
        with _LOCK:
            rows = [_job_response(dict(job), include_log=False) for job in _JOBS.values()]
        rows.sort(key=lambda row: row.get("created_at") or row.get("job_id") or "", reverse=True)
        return jsonify({"ok": True, "jobs": rows})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        with _LOCK:
            job = dict(_JOBS.get(job_id) or {})
        if not job:
            abort(404)
        return jsonify(_job_response(job, include_log=True))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        return jsonify(_cancel_job(job_id))

    @app.errorhandler(Exception)
    def json_error(exc):
        code = getattr(exc, "code", 500)
        message = getattr(exc, "description", None) or str(exc)
        return jsonify({"ok": False, "error": message, "type": type(exc).__name__}), code

    return app


def _run_plan(payload: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
    backend = _planner_backend(payload)
    if backend in CODEX_FULLFLOW_BACKENDS:
        return _run_codex_fullflow_plan(payload, job_id=job_id)
    return _run_chem_enzy_native_plan(payload, job_id=job_id)


def _planner_backend(payload: dict[str, Any]) -> str:
    backend = str(payload.get("planner_backend") or payload.get("planner_mode") or "chem_enzy_native").strip().lower()
    if backend not in CHEMENZY_NATIVE_BACKENDS | CODEX_FULLFLOW_BACKENDS:
        abort(400, description="planner_backend must be chem_enzy_native or codex_fullflow")
    return backend


def _run_chem_enzy_native_plan(payload: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
    target = str(payload.get("target_smiles") or "").strip()
    if Chem.MolFromSmiles(target) is None:
        abort(400, description="target_smiles is not a valid SMILES")
    missing_template_models = _missing_selected_template_relevance_models(payload)
    if missing_template_models:
        abort(
            400,
            description=(
                "missing local template_relevance .mar archive(s): "
                + ", ".join(missing_template_models)
            ),
        )
    if _plan_job_cancel_requested(job_id):
        raise _PlanJobCancelled("route search cancelled before ChemEnzy launch")
    runtime_preflight = _chem_enzy_runtime_status(
        production=True,
        request_payload=payload,
    )
    if not runtime_preflight["accepted"]:
        abort(500, description=format_chem_enzy_runtime_diagnostic(runtime_preflight))
    python_bin = Path(runtime_preflight["python_executable"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    run_id = uuid.uuid4().hex[:6]
    req_path = RESULTS_DIR / f"ui_chem_enzy_request_{stamp}_{run_id}.json"
    out_path = RESULTS_DIR / f"ui_chem_enzy_plan_{stamp}_{run_id}.json"
    req_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    gpu = 0 if _resolve_device(str(payload.get("device") or "cpu")) == "cuda" else -1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    cmd = [
        str(python_bin),
        str(ROOT / "scripts/run_chem_enzy_plan_for_web.py"),
        "--input",
        str(req_path),
        "--output",
        str(out_path),
        "--vendor-root",
        str(ROOT / "vendor/ChemEnzyRetroPlanner"),
        "--gpu",
        str(gpu),
    ]
    timeout_s = _chem_enzy_timeout(payload)
    started = time.monotonic()
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        if job_id:
            with _LOCK:
                _PLAN_PROCESS_BY_JOB[job_id] = proc
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if proc is not None:
            _terminate_process(proc)
            stdout, stderr = proc.communicate()
        else:
            stdout, stderr = "", ""
        timeout_exc = subprocess.TimeoutExpired(cmd, timeout_s, output=stdout, stderr=stderr)
        output = _chem_enzy_timeout_output(
            payload=payload,
            req_path=req_path,
            out_path=out_path,
            timeout_s=timeout_s,
            elapsed_s=time.monotonic() - started,
            exc=timeout_exc,
        )
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output
    finally:
        if job_id and proc is not None:
            with _LOCK:
                if _PLAN_PROCESS_BY_JOB.get(job_id) is proc:
                    _PLAN_PROCESS_BY_JOB.pop(job_id, None)
    if _plan_job_cancel_requested(job_id):
        raise _PlanJobCancelled("route search cancelled by user")
    if proc.returncode != 0:
        detail = "\n".join([stdout[-2000:], stderr[-4000:]]).strip()
        abort(500, description=f"ChemEnzy native search failed with code {proc.returncode}: {detail}")
    if not out_path.exists():
        abort(500, description="ChemEnzy native search did not write output")
    output = json.loads(out_path.read_text(encoding="utf-8"))
    output["time_s"] = round(time.monotonic() - started, 3)
    ui_metadata = output.setdefault("ui_metadata", {})
    ui_metadata["saved_at"] = _rel(out_path)
    ui_metadata["request_path"] = _rel(req_path)
    raw_out_path = out_path.with_name(f"{out_path.stem}_raw.json")
    _save_native_raw_output(output, raw_out_path)
    ui_metadata["raw_saved_at"] = _rel(raw_out_path)
    _apply_proposal_gate_post_filter(output, payload)
    rejected_out_path = out_path.with_name(f"{out_path.stem}_rejected.json")
    _apply_product_audit_post_filter(output, payload, rejected_out_path=rejected_out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _codex_run_profile(payload: dict[str, Any]) -> tuple[str, dict[str, int]]:
    name = str(payload.get("run_profile") or "standard").strip().lower()
    if name not in CODEX_RUN_PROFILES:
        abort(400, description="run_profile must be smoke, standard, or deep")
    return name, dict(CODEX_RUN_PROFILES[name])


def _trusted_benchmark_stock_catalog(
    config_path: str | Path = TRUSTED_STOCK_CATALOGS_CONFIG,
) -> dict[str, Any]:
    """Load one operator-owned, hash-pinned benchmark catalog definition."""
    path = Path(config_path).resolve()
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(config, dict) or config.get("schema_version") != "trusted_stock_catalogs.v1":
        return {}
    key = str(config.get("default_catalog") or "")
    row = dict((config.get("catalogs") or {}).get(key) or {})
    artifact_value = str(row.get("artifact") or "").strip()
    artifact = (ROOT / artifact_value).resolve() if artifact_value else Path()
    if (
        not artifact_value
        or (artifact != ROOT and not artifact.is_relative_to(ROOT))
        or not artifact.is_file()
    ):
        return {}
    digest = str(row.get("sha256") or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return {}
    return {
        "artifact": str(artifact),
        "sha256": digest,
        "name": str(row.get("name") or key or "benchmark-stock"),
        "boundary_type": "benchmark_stock",
        "commercial_orderability_claimed": False,
    }


def _run_codex_fullflow_plan(payload: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
    _validate_web_codex_controls(payload)
    profile_name, profile = _codex_run_profile(payload)
    benchmark_stock = _trusted_benchmark_stock_catalog()
    stock_authority_label = (
        "benchmark stock membership (not commercial procurement)"
        if benchmark_stock
        else "no trusted stock catalog loaded; terminal closure remains fail-closed"
    )
    target = str(payload.get("target_smiles") or "").strip()
    if Chem.MolFromSmiles(target) is None:
        abort(400, description="target_smiles is not a valid SMILES")
    closure_objective = str(
        payload.get("codex_agent_team_closure_objective") or "benchmark_search"
    ).strip().lower()
    if closure_objective not in {"benchmark_search", "procurement", "in_house"}:
        abort(400, description="invalid codex_agent_team_closure_objective")
    exploration_mode = str(
        payload.get("codex_agent_team_exploration_mode") or "exhaustive"
    ).strip().lower()
    if exploration_mode not in {"first_solved", "exhaustive"}:
        abort(400, description="invalid codex_agent_team_exploration_mode")
    child_acceptance_mode = str(
        payload.get("codex_agent_team_child_acceptance_mode") or "strict_all"
    ).strip().lower()
    if child_acceptance_mode not in {"strict_all", "valid_subset_l0"}:
        abort(400, description="invalid codex_agent_team_child_acceptance_mode")
    campaign_authority_lock_timeout_s = _as_float(
        payload.get("codex_agent_team_authority_lock_timeout_s"),
        3600.0,
        lo=0.1,
        hi=24 * 3600.0,
    )
    run_dir = _codex_fullflow_run_dir(payload)
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "web_request.json"
    request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if job_id:
        with _LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(
                    {
                        "run_dir": _rel(run_dir),
                        "request_json": _rel(request_path),
                        "agent_blackboard": _rel(run_dir / "agent_blackboard.json"),
                        "route_forest_html": _rel(run_dir / "route_forest.html"),
                        "explored_route_forest": _rel(run_dir / "explored_route_forest.json"),
                        "final_verdict": _rel(run_dir / "final_verdict.json"),
                    }
                )
    if _plan_job_cancel_requested(job_id):
        raise _PlanJobCancelled("agent fullflow cancelled before launch")

    timeout_s = _as_float(payload.get("timeout_s"), 1800.0, lo=30.0, hi=24 * 3600.0)
    started = time.monotonic()
    env_overrides = _codex_fullflow_env_overrides(payload)
    previous_env = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            os.environ[key] = value
        result = run_agentic_blackboard_controller(
            target_name=str(payload.get("target_name") or _safe_label(str(payload.get("family_hint") or "target"))),
            target_smiles=target,
            family_hint=str(payload.get("family_hint") or ""),
            output_dir=run_dir,
            literature_pdf_path=str(payload.get("literature_pdf_path") or ""),
            literature_pdf_source_ref=str(payload.get("literature_pdf_source_ref") or ""),
            literature_sources=[dict(row) for row in payload.get("literature_sources") or [] if isinstance(row, dict)],
            auto_discover_local_pdfs=_as_bool(payload.get("auto_local_pdf_discovery"), False),
            local_pdf_search_dirs=[Path(item) for item in payload.get("local_pdf_search_dirs") or []],
            timeout_s=timeout_s,
            # Credentials and provider endpoints are server-owned.  Accepting
            # either from an unauthenticated HTTP payload would turn a route
            # request into a credential-forwarding primitive.
            key_path=str(os.environ.get("AUTOPLANNER_CODEX_KEY_PATH") or ROOT / "key.txt"),
            base_url=str(os.environ.get("AUTOPLANNER_CODEX_BASE_URL") or "https://api.wellau.com/v1"),
            model=str(payload.get("model") or "gpt-5.5"),
            max_rounds=_as_int(payload.get("max_rounds"), profile["rounds"], lo=1, hi=30),
            exhaust_round_budget=_as_bool(payload.get("exhaust_round_budget"), False),
            enable_analogical_templates=_as_bool(payload.get("enable_analogical_templates"), True),
            max_template_applications_per_round=_as_int(payload.get("max_template_applications_per_round"), 5, lo=0, hi=50),
            template_radius_policy=str(payload.get("template_radius_policy") or "auto"),
            analog_template_confidence_threshold=str(payload.get("analog_template_confidence_threshold") or "medium"),
            use_codex_action_planner=_as_bool(payload.get("codex_action_planner"), False),
            use_codex_agent_team=_as_bool(payload.get("codex_agent_team"), True),
            codex_agent_team_max_depth=_as_int(payload.get("codex_agent_team_max_depth"), profile["depth"], lo=1, hi=12),
            codex_agent_team_max_expansions=_as_int(payload.get("codex_agent_team_max_expansions"), profile["accepted_expansions"], lo=1, hi=96),
            codex_agent_team_max_attempt_runs=_as_int(
                payload.get("codex_agent_team_max_attempt_runs"),
                profile["attempt_runs"],
                lo=1,
                hi=288,
            ),
            codex_agent_team_bootstrap_expansions=1,
            codex_agent_team_max_expansions_per_invocation=profile["per_invocation"],
            codex_agent_team_max_attempt_runs_per_invocation=profile["attempts_per_invocation"],
            codex_agent_team_frontier_batch_size=_as_int(payload.get("codex_agent_team_frontier_batch_size"), 2, lo=1, hi=8),
            codex_agent_team_closure_objective=closure_objective,
            codex_agent_team_exploration_mode=exploration_mode,
            codex_agent_team_child_acceptance_mode=child_acceptance_mode,
            codex_agent_team_authority_lock_timeout_s=(
                campaign_authority_lock_timeout_s
            ),
            codex_agent_team_model=str(
                payload.get("codex_agent_team_model") or payload.get("model") or "gpt-5.5"
            ),
            codex_agent_team_auth_mode="auto",
            codex_agent_team_benchmark_stock_catalog_artifact=benchmark_stock.get("artifact", ""),
            codex_agent_team_benchmark_stock_catalog_sha256=benchmark_stock.get("sha256", ""),
            codex_agent_team_benchmark_stock_catalog_name=benchmark_stock.get("name", ""),
            stop_on_problem=_as_bool(payload.get("stop_on_problem"), False),
            budget=_codex_fullflow_budget(
                payload,
                timeout_s=timeout_s,
                profile=profile,
            ),
            emit_blackboard_steps=True,
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        closeout_decision = load_latest_closeout_decision(run_dir)
        closeout_validation = validate_latest_closeout_revision(run_dir)
        final = dict(closeout_decision.get("final_verdict") or {})
    except ArtifactRevisionError:
        closeout_decision = {}
        closeout_validation = validate_latest_closeout_revision(run_dir)
        final = dict(result.get("final_verdict") or {})
    artifacts = dict(result.get("artifacts") or {})
    forest_counts = _route_forest_counts(artifacts.get("explored_route_forest") or run_dir / "explored_route_forest.json")
    compact = {
        "schema_version": "web_agent_fullflow_result.v1",
        "ok": bool(final.get("solved") or final.get("verdict") == "solved"),
        "target": target,
        "run_dir": str(run_dir),
        "final_verdict": final,
        "final_verdict_authority": (
            "content_addressed_closeout_objects"
            if closeout_decision
            else "compatibility_result"
        ),
        "closeout_revision": closeout_validation,
        "artifacts": artifacts,
        "preflight": result.get("preflight") or {},
        "target_input": result.get("target_input") or {},
        "blackboard_steps": _read_blackboard_step_summary(run_dir),
        "forest_counts": forest_counts,
        "time_s": round(time.monotonic() - started, 3),
        "run_profile": profile_name,
        "stock_authority": {
            **benchmark_stock,
            "available": bool(benchmark_stock),
            "label": stock_authority_label,
        },
        "routes": [],
        "search_status": {
            "status": final.get("route_status") or final.get("verdict") or "unknown",
            "solved": bool(final.get("solved") or final.get("verdict") == "solved"),
            "best_depth": forest_counts.get("steps"),
            "message": final.get("verdict") or final.get("route_status") or "agent fullflow finished",
        },
        "ui_metadata": {
            "backend": "AgenticBlackboard",
            "engine": "Codex action planner + ChemEnzy/literature/tools",
            "planner_strategy": "Agentic blackboard fullflow with per-step blackboard snapshots and final route forest.",
            "search_mode": "codex_fullflow",
            "run_profile": profile_name,
            "stock_authority_label": stock_authority_label,
            "run_dir": _rel(run_dir),
            "saved_at": "",
            "request_path": _rel(request_path),
            "agent_blackboard": _rel(Path(artifacts.get("agent_blackboard") or run_dir / "agent_blackboard.json")),
            "route_forest_html": _rel(Path(artifacts.get("route_forest_html") or run_dir / "route_forest.html")),
            "explored_route_forest": _rel(Path(artifacts.get("explored_route_forest") or run_dir / "explored_route_forest.json")),
            "final_verdict": _rel(Path(artifacts.get("final_verdict") or run_dir / "final_verdict.json")),
            "closeout_revision_manifest": _rel(
                Path(
                    artifacts.get("closeout_revision_manifest")
                    or run_dir / ".autoplanner" / "closeout" / "latest.json"
                )
            ),
        },
    }
    out_path = run_dir / "web_agent_fullflow_result.json"
    compact["ui_metadata"]["saved_at"] = _rel(out_path)
    out_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return compact


def _codex_fullflow_run_dir(payload: dict[str, Any]) -> Path:
    raw = str(payload.get("output_dir") or "").strip()
    if raw:
        return _safe_path(raw, allowed_roots=[RESULTS_DIR, SHARED_RESULTS_DIR])
    label = _safe_label(str(payload.get("target_name") or payload.get("family_hint") or "agent_target"))
    prefix = _safe_label(str(payload.get("run_prefix") or "ui_agent_fullflow"))
    return SHARED_RESULTS_DIR / "ui_agent_runs" / f"{prefix}_{label}_{_utc_stamp()}_{uuid.uuid4().hex[:6]}"


def _codex_fullflow_budget(
    payload: dict[str, Any],
    *,
    timeout_s: float,
    profile: Mapping[str, int] | None = None,
) -> HarnessBudget:
    selected = dict(profile or CODEX_RUN_PROFILES["standard"])
    budget = HarnessBudget(timeout_s=float(timeout_s))
    budget.max_chem_enzy_runs = _as_int(
        payload.get("max_chem_enzy_runs"),
        selected["chem_enzy_runs"],
        lo=0,
        hi=20,
    )
    budget.max_guided_chemenzy_runs = _as_int(
        payload.get("max_guided_chemenzy_runs"),
        budget.max_chem_enzy_runs,
        lo=0,
        hi=20,
    )
    budget.guided_chemenzy_timeout_s = _as_float(
        payload.get("guided_chemenzy_timeout_s"),
        min(max(timeout_s / 2, 300.0), timeout_s),
        lo=30.0,
        hi=24 * 3600.0,
    )
    budget.max_route_expansion_subgoal_runs = _as_int(
        payload.get("max_route_expansion_subgoal_runs"),
        selected["child_target_runs"],
        lo=0,
        hi=20,
    )
    budget.max_codex_research_runs = _as_int(
        payload.get("max_codex_research_runs"),
        selected["codex_research_runs"],
        lo=0,
        hi=20,
    )
    budget.max_scout_calls = _as_int(
        payload.get("max_scout_calls"),
        selected["scout_calls"],
        lo=0,
        hi=50,
    )
    budget.max_visual_calls = _as_int(
        payload.get("max_visual_calls"),
        selected["visual_calls"],
        lo=0,
        hi=50,
    )
    budget.max_template_applications_per_round = _as_int(payload.get("max_template_applications_per_round"), 5, lo=0, hi=50)
    return budget


def _codex_fullflow_env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if payload.get("codex_action_planner_tools") is not None:
        requested = {
            item.strip().lower()
            for item in str(payload.get("codex_action_planner_tools") or "").split(",")
            if item.strip()
        }
        allowed = requested & {"web_search", "browser", "literature_search"}
        out["AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS"] = ",".join(sorted(allowed))
    if payload.get("codex_action_planner_max_tool_calls") is not None:
        out["AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS"] = str(_as_int(payload.get("codex_action_planner_max_tool_calls"), 8, lo=0, hi=100))
    if payload.get("codex_action_planner_timeout_s") is not None:
        out["AUTOPLANNER_CODEX_ACTION_PLANNER_TIMEOUT_S"] = str(_as_float(payload.get("codex_action_planner_timeout_s"), 900.0, lo=30.0, hi=24 * 3600.0))
    if payload.get("codex_scout_timeout_s") is not None:
        out["AUTOPLANNER_CODEX_SCOUT_TIMEOUT_S"] = str(_as_float(payload.get("codex_scout_timeout_s"), 900.0, lo=30.0, hi=24 * 3600.0))
    effort = str(payload.get("codex_scout_reasoning_effort") or "").strip()
    if effort:
        out["AUTOPLANNER_CODEX_SCOUT_REASONING_EFFORT"] = effort
    # Web-launched Codex children are always read-only.  Deterministic harness
    # tools own all writes after validating typed output.
    out["AUTOPLANNER_CODEX_WORKER_SANDBOX"] = "read-only"
    local_seeded = bool(
        payload.get("literature_pdf_path")
        or payload.get("literature_sources")
        or (payload.get("auto_local_pdf_discovery") and payload.get("local_pdf_search_dirs"))
    )
    out["AUTOPLANNER_CODEX_ACTION_PLANNER_LOCAL_PDF_FALLBACK_ALLOWED"] = "1" if local_seeded else "0"
    return out


def _validate_web_codex_controls(payload: dict[str, Any]) -> None:
    """Reject process-level controls that must never cross the HTTP boundary."""
    forbidden = [
        key
        for key in ("key_path", "base_url", "codex_worker_sandbox")
        if str(payload.get(key) or "").strip()
    ]
    worker_auth = str(payload.get("codex_worker_auth") or "").strip().lower()
    if worker_auth not in {"", "auto"}:
        forbidden.append("codex_worker_auth")
    if forbidden:
        abort(400, description=f"server-controlled Codex settings: {', '.join(sorted(set(forbidden)))}")


def _validate_web_literature_inputs(payload: dict[str, Any]) -> None:
    """Constrain HTTP-selected local source files to an operator-owned root."""
    path_values: list[tuple[str, str, bool]] = []
    if str(payload.get("literature_pdf_path") or "").strip():
        path_values.append(("literature_pdf_path", str(payload["literature_pdf_path"]), True))
    for value in payload.get("local_pdf_search_dirs") or []:
        if str(value or "").strip():
            path_values.append(("local_pdf_search_dirs", str(value), False))
    for index, row in enumerate(payload.get("literature_sources") or []):
        if not isinstance(row, dict):
            continue
        for key in ("local_pdf", "path", "local_ref"):
            value = str(row.get(key) or "").strip()
            if value and (key != "local_ref" or Path(value).suffix.lower() == ".pdf"):
                path_values.append((f"literature_sources[{index}].{key}", value, True))
    if not path_values:
        return
    configured_root = str(os.environ.get("AUTOPLANNER_WEB_LITERATURE_ROOT") or "").strip()
    if not configured_root:
        abort(400, description="local literature inputs are disabled; configure AUTOPLANNER_WEB_LITERATURE_ROOT")
    root = Path(configured_root).expanduser().resolve()
    for field, raw, require_pdf in path_values:
        candidate = Path(raw).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = candidate.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            abort(400, description=f"{field} is outside AUTOPLANNER_WEB_LITERATURE_ROOT")
        if require_pdf and resolved.suffix.lower() != ".pdf":
            abort(400, description=f"{field} must reference a PDF")


def _read_blackboard_step_summary(run_dir: Path | str, *, limit: int = 120) -> list[dict[str, Any]]:
    path = Path(run_dir) / "blackboard_steps" / "summary.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _route_examples_payload() -> dict[str, Any]:
    """Return a bounded inventory of route forests present on this checkout."""
    examples: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []
    result_roots = _route_example_roots()
    known_paths: set[Path] = set()
    for spec in ROUTE_EXAMPLE_SPECS:
        key = str(spec.get("key") or "").strip()
        label = str(spec.get("label") or key).strip()
        raw_path = str(spec.get("path") or "").strip()
        candidate = (ROOT / raw_path).resolve() if raw_path else ROOT.resolve()
        in_results = any(candidate == root or candidate.is_relative_to(root) for root in result_roots)
        if (
            raw_path
            and in_results
            and candidate.is_file()
            and candidate.name.casefold() == "route_forest.html"
        ):
            examples.append({"key": key, "label": label, "path": _rel(candidate)})
            known_paths.add(candidate)
            continue
        unavailable.append(
            {
                "key": key,
                "label": label,
                "reason": "artifact_missing" if in_results else "artifact_outside_results",
                "run_target": key,
            }
        )

    discovered, scanned_entries, scan_truncated = _discover_route_forests(result_roots)
    for candidate in discovered:
        if candidate in known_paths or len(examples) >= ROUTE_EXAMPLE_MAX_RESULTS:
            continue
        examples.append(_discovered_route_example(candidate, result_roots))

    if examples:
        message = f"发现 {len(examples)} 个可预览的本地路线图。"
    else:
        message = "当前 checkout 没有本地路线图；请选择目标并启动 Agent 生成结果。"
    return {
        "schema_version": "route_example_availability.v1",
        "ok": True,
        "examples": examples,
        "unavailable_examples": unavailable,
        "available_count": len(examples),
        "scan": {
            "roots": [_rel(root) for root in result_roots],
            "scanned_entries": scanned_entries,
            "truncated": scan_truncated,
            "max_depth": ROUTE_EXAMPLE_MAX_DEPTH,
            "max_entries_per_root": ROUTE_EXAMPLE_MAX_ENTRIES_PER_ROOT,
            "max_results": ROUTE_EXAMPLE_MAX_RESULTS,
        },
        "message": message,
    }


def _route_example_roots() -> tuple[Path, ...]:
    """Resolve the two configured roots, rejecting aliases outside repository results."""
    repository_results = (ROOT / "results").resolve()
    roots: list[Path] = []
    for configured in (SHARED_RESULTS_DIR, RESULTS_DIR):
        resolved = Path(configured).resolve()
        if resolved != repository_results and not resolved.is_relative_to(repository_results):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _discover_route_forests(roots: tuple[Path, ...]) -> tuple[list[Path], int, bool]:
    """Find route_forest.html without following links or scanning without bounds."""
    discovered: set[Path] = set()
    scanned_entries = 0
    truncated = False
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        root_entries = 0
        root_limit_reached = False
        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        while queue:
            directory, depth = queue.popleft()
            child_directories: list[Path] = []
            found_route_here = False
            try:
                with os.scandir(directory) as iterator:
                    for entry in iterator:
                        root_entries += 1
                        scanned_entries += 1
                        if root_entries > ROUTE_EXAMPLE_MAX_ENTRIES_PER_ROOT:
                            truncated = True
                            root_limit_reached = True
                            break
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_file(follow_symlinks=False):
                                if entry.name.casefold() != "route_forest.html":
                                    continue
                                candidate = Path(entry.path).resolve()
                                if candidate.is_relative_to(root):
                                    discovered.add(candidate)
                                    found_route_here = True
                                continue
                            if depth < ROUTE_EXAMPLE_MAX_DEPTH and entry.is_dir(follow_symlinks=False):
                                child_directories.append(Path(entry.path))
                            elif depth >= ROUTE_EXAMPLE_MAX_DEPTH and entry.is_dir(follow_symlinks=False):
                                truncated = True
                        except OSError:
                            continue
            except OSError:
                continue
            if root_limit_reached:
                break
            if found_route_here:
                continue
            # Breadth-first traversal finds run-level artifacts before large nested traces.
            queue.extend(
                (child, depth + 1)
                for child in sorted(child_directories, key=lambda path: path.name.casefold())
            )
    ordered = sorted(discovered, key=lambda path: _rel(path).replace("\\", "/").casefold())
    if len(ordered) > ROUTE_EXAMPLE_MAX_RESULTS:
        truncated = True
        ordered = ordered[:ROUTE_EXAMPLE_MAX_RESULTS]
    return ordered, scanned_entries, truncated


def _discovered_route_example(candidate: Path, roots: tuple[Path, ...]) -> dict[str, str]:
    relative_path = _rel(candidate)
    owning_root = next(root for root in roots if candidate.is_relative_to(root))
    relative_run = candidate.relative_to(owning_root).parent
    run_name = relative_run.name or owning_root.name
    friendly_name = " ".join(part for part in run_name.replace("_", " ").split() if part)
    label = f"{friendly_name[:80] or 'local route'}（本地 {owning_root.name}）"
    return {
        "key": f"local:{relative_path.replace(os.sep, '/')}",
        "label": label,
        "path": relative_path,
    }


def _route_forest_counts(path_value: Any) -> dict[str, Any]:
    try:
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data.get("counts") or {})
    except Exception:
        return {}


def _apply_proposal_gate_post_filter(output: dict[str, Any], payload: dict[str, Any]) -> None:
    routes = output.get("routes")
    if not isinstance(routes, list):
        return

    mode = _proposal_gate_mode(payload)
    enabled = mode != "off"
    report: dict[str, Any] = {
        "schema_version": "web_proposal_gate.v1",
        "enabled": enabled,
        "mode": mode,
        "input_routes": len(routes),
        "kept_routes": len(routes),
        "dropped_routes": 0,
        "route_decision_counts": {},
        "reason_counts": {},
        "frontiers": [],
        "dropped": [],
        "description": (
            "Proposal gate applies conservative material/core-growth checks before "
            "product audit. hard_reject hides routes containing impossible one-step proposals."
        ),
    }
    output.setdefault("route_set_metrics", {})["proposal_gate"] = report
    output.setdefault("ui_metadata", {})["proposal_gate"] = report
    if not enabled or not routes:
        return

    config = ProposalGateConfig(mode=mode)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    for original_index, route in enumerate(routes):
        if not isinstance(route, dict):
            continue
        route.setdefault("native_rank", original_index)
        route.setdefault("original_route_rank", route.get("route_rank", original_index))
        gate_report = gate_web_route(route, config=config)
        route["proposal_gate"] = _compact_route_proposal_gate(gate_report)
        metrics = route.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics["proposal_gate"] = route["proposal_gate"]
        route_reports.append(gate_report)
        if mode == "hard_reject" and gate_report.get("hard_reject"):
            dropped.append({"route": route, "gate": gate_report, "original_index": original_index})
        else:
            kept.append(route)

    for new_rank, route in enumerate(kept):
        route["route_rank"] = new_rank
        route["post_proposal_gate_rank"] = new_rank

    frontiers = _proposal_gate_frontiers(dropped)
    summary = summarize_route_gate_reports(route_reports)
    report.update(
        {
            "input_routes": len(routes),
            "kept_routes": len(kept),
            "dropped_routes": len(dropped),
            "route_decision_counts": summary.get("route_decision_counts") or {},
            "reason_counts": summary.get("reason_counts") or {},
            "frontiers": frontiers,
            "dropped": [_compact_dropped_proposal_gate_row(item) for item in dropped[:50]],
        }
    )
    output["routes"] = kept
    output["n_results"] = len(kept)
    output["frontiers"] = frontiers
    output["proposal_gate"] = report
    output.setdefault("route_set_metrics", {})["proposal_gate"] = report
    output.setdefault("ui_metadata", {})["proposal_gate"] = report
    _refresh_native_route_payload_after_proposal_gate(output)
    if routes and not kept and dropped:
        _attach_proposal_gate_failure_analysis(output)


def _proposal_gate_mode(payload: dict[str, Any]) -> str:
    enabled = _as_bool(payload.get("enable_proposal_gate"), True)
    if not enabled:
        return "off"
    return normalize_proposal_gate_mode(
        payload.get("proposal_gate_mode")
        or os.environ.get("AUTOPLANNER_PROPOSAL_GATE_MODE")
        or "hard_reject"
    )


def _compact_route_proposal_gate(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "decision": report.get("decision"),
        "hard_reject": bool(report.get("hard_reject")),
        "mode": report.get("mode"),
        "step_count": report.get("step_count"),
        "rejected_step_count": report.get("rejected_step_count"),
        "route_hard_reasons": report.get("route_hard_reasons") or [],
        "reason_counts": report.get("reason_counts") or {},
        "frontier": report.get("frontier"),
    }


def _compact_dropped_proposal_gate_row(item: dict[str, Any]) -> dict[str, Any]:
    route = item.get("route") or {}
    gate = item.get("gate") or {}
    return {
        "route_rank": route.get("original_route_rank", route.get("route_rank")),
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "reason_counts": gate.get("reason_counts") or {},
        "frontier": gate.get("frontier"),
    }


def _proposal_gate_frontiers(dropped: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in dropped:
        route = item.get("route") or {}
        gate = item.get("gate") or {}
        frontier = dict(gate.get("frontier") or {})
        smiles = str(frontier.get("smiles") or "")
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        frontier.setdefault("route_rank", route.get("original_route_rank", route.get("route_rank")))
        frontier.setdefault("suggested_next_policy", "literature_or_core_provider")
        out.append(frontier)
        if len(out) >= limit:
            break
    return out


def _refresh_native_route_payload_after_proposal_gate(output: dict[str, Any]) -> None:
    routes = [route for route in output.get("routes") or [] if isinstance(route, dict)]
    output["n_results"] = len(routes)
    diversity = output.setdefault("route_set_metrics", {}).setdefault("diversity", {})
    diversity["n_routes"] = len(routes)
    diversity["unique_full_signatures"] = len({_native_route_signature(route) for route in routes})
    for attempt in output.get("depth_attempts") or []:
        if isinstance(attempt, dict):
            attempt.setdefault("raw_n_routes", attempt.get("n_routes"))
            attempt["n_routes"] = len(routes)
            attempt["best"] = _route_ui_summary(routes[0]) if routes else None
    gate = output.get("proposal_gate") or {}
    search_status = output.setdefault("search_status", {})
    solved = any(bool((route.get("metrics") or {}).get("route_solved")) for route in routes)
    search_status["solved"] = solved
    search_status["proposal_gate_removed_all"] = bool(
        gate.get("enabled") and int(gate.get("input_routes") or 0) > 0 and not routes
    )
    if routes:
        output["ok"] = True
        search_status["status"] = "solved" if solved else "partial"
        search_status["best_depth"] = (routes[0].get("metrics") or {}).get("n_steps") or routes[0].get("n_steps")
        search_status["message"] = (
            f"ChemEnzy native core search returned {gate.get('input_routes', len(routes))} route(s); "
            f"proposal gate kept {len(routes)}"
        )
    elif search_status["proposal_gate_removed_all"]:
        output["ok"] = False
        search_status["status"] = "frontier"
        search_status["native_returned_routes"] = False
        search_status["native_raw_returned_routes"] = True
        search_status["best_depth"] = None
        search_status["message"] = (
            f"ChemEnzy returned {gate.get('input_routes')} raw route(s), but proposal gate removed all "
            "because each contained impossible material/core-growth steps; unresolved frontiers are reported."
        )


def _attach_proposal_gate_failure_analysis(output: dict[str, Any]) -> None:
    gate = output.get("proposal_gate") or {}
    input_routes = int(gate.get("input_routes") or 0)
    dropped_routes = int(gate.get("dropped_routes") or 0)
    categories = [str(item) for item in output.get("failure_diagnosis") or []]
    for category in ("proposal_gate_filtered_all", "frontier_unresolved_core"):
        if category not in categories:
            categories.append(category)
    output["failure_diagnosis"] = categories

    analysis = output.setdefault("failure_analysis", {})
    existing_categories = [str(item) for item in analysis.get("failure_categories") or []]
    for category in categories:
        if category not in existing_categories:
            existing_categories.append(category)
    diagnosis = [str(item) for item in analysis.get("diagnosis") or []]
    _append_unique(
        diagnosis,
        f"ChemEnzy returned {input_routes} raw route(s), but proposal gate removed {dropped_routes} route(s) before product audit.",
    )
    if gate.get("reason_counts"):
        _append_unique(diagnosis, "Dominant proposal gate reasons: " + _format_counter_rows(gate.get("reason_counts") or {}) + ".")
    _append_unique(
        diagnosis,
        "Rejected routes contain one-step proposals where product core material is not supplied by listed reactants or accepted condition reagents.",
    )
    suggestions = [str(item) for item in analysis.get("retry_suggestions") or []]
    _append_unique(suggestions, "inspect raw_output_json for rejected debug routes; do not present them as proposed syntheses")
    _append_unique(suggestions, "add a known advanced core intermediate or literature/core provider for unresolved frontiers")
    _append_unique(suggestions, "use proposal_gate_mode=warn only for debugging the raw proposal family")
    analysis.update(
        {
            "available": True,
            "failure_categories": existing_categories,
            "diagnosis": diagnosis,
            "retry_suggestions": suggestions,
            "proposal_gate": {
                "removed_all": True,
                "input_routes": input_routes,
                "dropped_routes": dropped_routes,
                "kept_routes": int(gate.get("kept_routes") or 0),
                "mode": gate.get("mode"),
                "reason_counts": gate.get("reason_counts") or {},
                "frontier_count": len(gate.get("frontiers") or []),
            },
        }
    )


def _apply_product_audit_post_filter(
    output: dict[str, Any],
    payload: dict[str, Any],
    *,
    rejected_out_path: Path | None = None,
) -> None:
    """Conservatively rank and hide clearly impossible native routes for the UI."""
    routes = output.get("routes")
    if not isinstance(routes, list) or not routes:
        return

    mode = _product_audit_filter_mode(payload)
    if mode == "off":
        output["post_filter"] = {
            "schema_version": "web_product_audit_post_filter.v1",
            "enabled": False,
            "mode": mode,
            "original_route_count": len(routes),
            "kept_route_count": len(routes),
            "removed_route_count": 0,
        }
        return

    try:
        from cascade_planner.eval.product_route_feasibility_audit import (
            build_product_route_feasibility_audit,
            product_audit_guard_key,
            product_audit_risk_order,
        )

        target_smiles = str(output.get("target") or payload.get("target_smiles") or "").strip()
        audit_run = {
            "metadata": {"source": "web_ui", "post_filter_mode": mode},
            "targets": [
                {
                    "index": 0,
                    "target_id": str(payload.get("target_id") or "web_target"),
                    "target_smiles": target_smiles,
                    "planner_output": {"routes": routes},
                    "metrics": {
                        "strict_stock_solve_any": any(
                            bool((route.get("metrics") or {}).get("strict_stock_solve"))
                            for route in routes
                            if isinstance(route, dict)
                        )
                    },
                }
            ],
        }
        audit = build_product_route_feasibility_audit(audit_run)
        audit_target = (audit.get("targets") or [{}])[0]
        audit_by_index = {
            int(row.get("rank") or 0) - 1: row
            for row in audit_target.get("routes") or []
            if row.get("rank") is not None
        }

        ranked: list[dict[str, Any]] = []
        for original_index, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            row = audit_by_index.get(original_index)
            risk = product_audit_risk_order(row or {})
            audit_meta = _compact_product_audit_row(row, risk) if row else _missing_product_audit_row()
            route.setdefault("native_rank", original_index)
            route.setdefault("original_route_rank", route.get("route_rank", original_index))
            route["product_audit"] = audit_meta
            route["rule_post_rank_metadata"] = {
                "route_class": audit_meta.get("route_class"),
                "risk_order": audit_meta.get("risk_order"),
                "issues": audit_meta.get("issues") or [],
                "tags": audit_meta.get("tags") or [],
                "route_plausibility": audit_meta.get("route_plausibility") or {},
            }
            guard = product_audit_guard_key(row or {}) if row else (99, 99)
            ranked.append(
                {
                    "route": route,
                    "audit": row or {},
                    "risk": risk,
                    "guard": (*guard, original_index),
                    "remove": _remove_route_by_product_audit(row or {}, risk=risk, mode=mode),
                }
            )

        ranked.sort(key=lambda item: item["guard"])
        kept_items = [item for item in ranked if not item["remove"]]
        removed_items = [item for item in ranked if item["remove"]]
        fallback_reason = None

        kept_routes = [item["route"] for item in kept_items]
        for new_rank, route in enumerate(kept_routes):
            route["post_filter_rank"] = new_rank
            route["route_rank"] = new_rank

        output["routes"] = kept_routes
        output["n_results"] = len(kept_routes)
        output["post_filter"] = _product_audit_filter_summary(
            mode=mode,
            original_count=len(routes),
            kept_items=kept_items,
            removed_items=removed_items,
            all_items=ranked,
            audit=audit,
            fallback_reason=fallback_reason,
        )
        if rejected_out_path is not None and removed_items:
            _save_rejected_routes_output(output, rejected_out_path, removed_items=removed_items, audit=audit)
            rejected_saved_at = _rel(rejected_out_path)
            output["post_filter"]["rejected_saved_at"] = rejected_saved_at
            output.setdefault("ui_metadata", {})["rejected_saved_at"] = rejected_saved_at
        output.setdefault("ui_metadata", {})["product_audit_post_filter"] = output["post_filter"]
        _refresh_native_route_payload_after_filter(output)
    except Exception as exc:
        output["post_filter"] = {
            "schema_version": "web_product_audit_post_filter.v1",
            "enabled": False,
            "mode": mode,
            "original_route_count": len(routes),
            "kept_route_count": len(routes),
            "removed_route_count": 0,
            "fallback_reason": "audit_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        output.setdefault("ui_metadata", {})["product_audit_post_filter"] = output["post_filter"]


def _product_audit_filter_mode(payload: dict[str, Any]) -> str:
    enabled = _as_bool(payload.get("enable_product_audit_filter"), True)
    if not enabled:
        return "off"
    raw = str(
        payload.get("product_audit_filter_mode")
        or os.environ.get("AUTOPLANNER_PRODUCT_AUDIT_FILTER_MODE")
        or "risk_guarded"
    ).strip().lower()
    aliases = {
        "rerank": "risk_guarded",
        "rank": "risk_guarded",
        "rank_only": "risk_guarded",
        "filter": "hide_rejects",
        "strict": "hide_risky",
        "hide_artifacts": "hide_rejects",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"off", "risk_guarded", "hide_rejects", "hide_risky", "triage_only"}:
        mode = "hide_rejects"
    return mode


def _remove_route_by_product_audit(row: dict[str, Any], *, risk: int, mode: str) -> bool:
    if mode in {"off", "risk_guarded"}:
        return False
    route_class = str(row.get("route_class") or "")
    severe = route_class == "reject_artifact" or risk >= 40
    if mode == "hide_rejects":
        return severe
    if mode == "hide_risky":
        return severe or risk >= 30
    if mode == "triage_only":
        return severe or route_class not in {"triage_semisynthesis", "triage_late_stage", "triage_fragment", "needs_chemist_review"}
    return False


def _compact_product_audit_row(row: dict[str, Any] | None, risk: int) -> dict[str, Any]:
    row = row or {}
    return {
        "schema_version": "route_product_audit.v1",
        "route_class": row.get("route_class"),
        "risk_order": risk,
        "autonomous_route_candidate": bool(row.get("autonomous_route_candidate")),
        "stock_closed": bool(row.get("stock_closed")),
        "route_solved": bool(row.get("route_solved")),
        "filled_route": bool(row.get("filled_route")),
        "issues": list(row.get("issues") or []),
        "tags": list(row.get("tags") or []),
        "terminal_profile": row.get("terminal_profile") or {},
        "reaction_profile": row.get("reaction_profile") or {},
        "condition_audit": row.get("condition_audit") or {},
        "route_plausibility": row.get("route_plausibility") or {},
    }


def _missing_product_audit_row() -> dict[str, Any]:
    return {
        "schema_version": "route_product_audit.v1",
        "route_class": "audit_missing",
        "risk_order": 99,
        "autonomous_route_candidate": False,
        "stock_closed": False,
        "route_solved": False,
        "filled_route": False,
        "issues": ["audit_missing"],
        "tags": [],
        "terminal_profile": {},
        "reaction_profile": {},
        "condition_audit": {},
        "route_plausibility": {},
    }


def _product_audit_filter_summary(
    *,
    mode: str,
    original_count: int,
    kept_items: list[dict[str, Any]],
    removed_items: list[dict[str, Any]],
    all_items: list[dict[str, Any]],
    audit: dict[str, Any],
    fallback_reason: str | None,
) -> dict[str, Any]:
    def class_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(str((item.get("audit") or {}).get("route_class") or "audit_missing") for item in items).items()))

    def issue_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for item in items:
            for issue in (item.get("audit") or {}).get("issues") or []:
                counter[str(issue)] += 1
        return dict(sorted(counter.items()))

    return {
        "schema_version": "web_product_audit_post_filter.v1",
        "enabled": True,
        "mode": mode,
        "original_route_count": original_count,
        "kept_route_count": len(kept_items),
        "removed_route_count": len(removed_items),
        "would_remove_route_count": sum(1 for item in all_items if item.get("remove")),
        "fallback_reason": fallback_reason,
        "route_class_counts_before": class_counts(all_items),
        "route_class_counts_kept": class_counts(kept_items),
        "route_class_counts_removed": class_counts(removed_items),
        "issue_counts_before": issue_counts(all_items),
        "issue_counts_removed": issue_counts(removed_items),
        "target_verdict_counts": audit.get("target_verdict_counts") or {},
        "description": (
            "Product-audit post-filter: routes are first sorted by product triage class and "
            "material sanity risk; hide_rejects removes only reject_artifact/severe routes."
        ),
    }


def _refresh_native_route_payload_after_filter(output: dict[str, Any]) -> None:
    routes = [route for route in output.get("routes") or [] if isinstance(route, dict)]
    output["n_results"] = len(routes)
    diversity = output.setdefault("route_set_metrics", {}).setdefault("diversity", {})
    diversity["n_routes"] = len(routes)
    diversity["unique_full_signatures"] = len({_native_route_signature(route) for route in routes})
    for attempt in output.get("depth_attempts") or []:
        if isinstance(attempt, dict):
            attempt.setdefault("raw_n_routes", attempt.get("n_routes"))
            attempt["n_routes"] = len(routes)
            attempt["best"] = _route_ui_summary(routes[0]) if routes else None
    search_status = output.setdefault("search_status", {})
    solved = any(bool((route.get("metrics") or {}).get("route_solved")) for route in routes)
    search_status["solved"] = solved
    search_status["status"] = "solved" if solved else "partial" if routes else "failed"
    if routes:
        search_status["best_depth"] = (routes[0].get("metrics") or {}).get("n_steps") or routes[0].get("n_steps")
        pf = output.get("post_filter") or {}
        closure = "stock-closed" if solved else "open-stock"
        search_status["message"] = (
            f"ChemEnzy native core search returned {pf.get('original_route_count', len(routes))} {closure} routes; "
            f"product-audit post-filter kept {len(routes)}"
        )
    else:
        pf = output.get("post_filter") or {}
        original_count = int(pf.get("original_route_count") or 0)
        removed_count = int(pf.get("removed_route_count") or 0)
        search_status["best_depth"] = None
        if original_count > 0 and removed_count >= original_count:
            search_status["status"] = "filtered"
            search_status["native_returned_routes"] = True
            search_status["post_filter_removed_all"] = True
            rejected = pf.get("rejected_saved_at")
            suffix = f"; rejected routes saved at {rejected}" if rejected else ""
            search_status["message"] = (
                f"ChemEnzy native core search returned {original_count} route(s), "
                f"but product-audit hid all of them{suffix}"
            )
            _attach_product_audit_filtered_failure_analysis(output)
        else:
            search_status["message"] = "ChemEnzy native core search returned no route after product-audit post-filter"


def _attach_product_audit_filtered_failure_analysis(output: dict[str, Any]) -> None:
    pf = output.get("post_filter") or {}
    original_count = int(pf.get("original_route_count") or 0)
    removed_count = int(pf.get("removed_route_count") or 0)
    if original_count <= 0 or removed_count < original_count:
        return

    top_issues = _top_counter_rows(pf.get("issue_counts_removed") or pf.get("issue_counts_before") or {}, limit=6)
    class_counts = dict(pf.get("route_class_counts_removed") or pf.get("route_class_counts_before") or {})
    target_profile = _target_complexity_profile(str(output.get("target") or ""))

    categories = [str(item) for item in output.get("failure_diagnosis") or []]
    if "product_audit_filtered_all" not in categories:
        categories.append("product_audit_filtered_all")
    output["failure_diagnosis"] = categories

    analysis = output.setdefault("failure_analysis", {})
    existing_categories = [str(item) for item in analysis.get("failure_categories") or []]
    for category in categories:
        if category not in existing_categories:
            existing_categories.append(category)

    diagnosis = [str(item) for item in analysis.get("diagnosis") or []]
    _append_unique(
        diagnosis,
        f"ChemEnzy returned {original_count} candidate route(s), but product-audit removed all of them as severe material-sanity artifacts.",
    )
    if class_counts:
        _append_unique(diagnosis, "Route triage before filtering: " + _format_counter_rows(class_counts) + ".")
    if top_issues:
        _append_unique(diagnosis, "Dominant rejection issues: " + _format_counter_rows(dict(top_issues)) + ".")
    issue_names = {name for name, _ in top_issues}
    if {
        "large_unexplained_heavy_atom_gain",
        "large_unexplained_carbon_gain",
        "large_unexplained_hetero_atom_gain",
    } & issue_names:
        _append_unique(
            diagnosis,
            "The rejected routes contain steps where a product gains many atoms that are not supplied by listed reactants or accepted condition reagents.",
        )
    if target_profile.get("natural_product_like"):
        _append_unique(
            diagnosis,
            "The target is a large polycyclic, stereochemically dense molecule; de novo stock-closed template search is unlikely without advanced core intermediates.",
        )

    suggestions = [str(item) for item in analysis.get("retry_suggestions") or []]
    rejected = pf.get("rejected_saved_at")
    if rejected:
        _append_unique(suggestions, f"inspect rejected diagnostic routes at {rejected}; do not present them as proposed syntheses")
    _append_unique(suggestions, "add or select advanced core intermediates in stock/constraints for semisynthesis-style planning")
    _append_unique(suggestions, "use risk_guarded mode only for debugging raw proposals; hide_rejects is the safer presentation mode")

    analysis.update(
        {
            "available": True,
            "target_heavy_atoms": target_profile.get("heavy_atoms"),
            "failure_categories": existing_categories,
            "diagnosis": diagnosis,
            "retry_suggestions": suggestions,
            "product_audit_filter": {
                "removed_all": True,
                "original_route_count": original_count,
                "removed_route_count": removed_count,
                "kept_route_count": int(pf.get("kept_route_count") or 0),
                "mode": pf.get("mode"),
                "route_class_counts_removed": class_counts,
                "issue_counts_removed": dict(pf.get("issue_counts_removed") or {}),
                "rejected_saved_at": rejected,
            },
            "target_complexity": target_profile,
        }
    )


def _target_complexity_profile(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"available": False}
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rings = int(mol.GetRingInfo().NumRings())
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return {
        "available": True,
        "heavy_atoms": heavy_atoms,
        "rings": rings,
        "chiral_centers": chiral_centers,
        "hetero_atoms": int(hetero_atoms),
        "natural_product_like": heavy_atoms >= 45 and rings >= 4 and chiral_centers >= 5,
    }


def _top_counter_rows(counts: dict[str, Any], *, limit: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for key, value in counts.items():
        try:
            rows.append((str(key), int(value)))
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:limit]


def _format_counter_rows(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in _top_counter_rows(counts, limit=8)) or "none"


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _native_route_signature(route: dict[str, Any]) -> str:
    return "|".join(str(step.get("reaction_smiles") or "") for step in route.get("steps") or [] if isinstance(step, dict))


def _chem_enzy_timeout(payload: dict[str, Any]) -> int:
    override = os.environ.get("AUTOPLANNER_CHEMENZY_TIMEOUT_S")
    if override:
        return _as_int(override, 900, lo=30, hi=7200)

    preset = str(payload.get("search_preset") or "quick").lower()
    iterations = _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=500)
    topk = _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=500)
    depth = _as_int(payload.get("max_steps"), 6, lo=1, hi=20)
    annotation_extra = 240 if _as_bool(payload.get("enable_condition_prediction")) or _as_bool(payload.get("enable_enzyme_assignment")) else 0
    dynamic = int(120 + iterations * max(1.0, topk / 50.0) * max(1.0, depth / 10.0) * 1.4 + annotation_extra)
    if preset == "thorough":
        return min(max(900, dynamic), 2400)
    if preset == "balanced":
        return min(max(420, dynamic), 1800)
    return min(max(180, dynamic), 1200)


def _chem_enzy_timeout_output(
    *,
    payload: dict[str, Any],
    req_path: Path,
    out_path: Path,
    timeout_s: int,
    elapsed_s: float,
    exc: subprocess.TimeoutExpired,
) -> dict[str, Any]:
    target = str(payload.get("target_smiles") or "")
    stdout = _tail_text(getattr(exc, "stdout", None), 2000)
    stderr = _tail_text(getattr(exc, "stderr", None), 4000)
    output = {
        "ok": False,
        "target": target,
        "objective": "chem_enzy_native",
        "constraints": payload.get("constraints"),
        "n_results": 0,
        "time_s": round(float(elapsed_s), 3),
        "routes": [],
        "route_set_metrics": {"diversity": {"n_routes": 0, "unique_full_signatures": 0}},
        "ui_metadata": {
            "backend": "CascadePlanner",
            "engine": "ChemEnzyRetroPlanner",
            "planner_strategy": "ChemEnzy native multi-step search with AutoPlanner product audit and rule cascade verifier",
            "search_mode": "chem_enzy_native",
            "search_preset": payload.get("search_preset", "quick"),
            "max_depth": _as_int(payload.get("max_steps"), 6, lo=1, hi=20),
            "iterations": _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=500),
            "expansion_topk": _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=500),
            "timeout_s": timeout_s,
            "saved_at": _rel(out_path),
            "request_path": _rel(req_path),
        },
        "skeletons": [],
        "depth_attempts": [
            {
                "depth": _as_int(payload.get("max_steps"), 6, lo=1, hi=20),
                "elapsed_s": round(float(elapsed_s), 3),
                "n_skeletons": 0,
                "n_routes": 0,
                "planner": "CascadePlanner",
                "engine": "ChemEnzyRetroPlanner",
                "status": "timeout",
                "best": None,
            }
        ],
        "search_status": {
            "status": "timeout",
            "solved": False,
            "best_depth": _as_int(payload.get("max_steps"), 6, lo=1, hi=20),
            "message": f"ChemEnzy native search exceeded the Web timeout ({timeout_s}s)",
        },
        "failure_diagnosis": ["backend_timeout"],
        "failure_analysis": {
            "available": True,
            "failure_categories": ["backend_timeout"],
            "diagnosis": [
                f"ChemEnzy subprocess exceeded the Web timeout of {timeout_s}s.",
                "This is a runtime cutoff, not chemical proof that no retrosynthesis exists.",
                "For large statin-like targets, repaired full-depth search can take several minutes.",
            ],
            "retry_suggestions": [
                "increase AUTOPLANNER_CHEMENZY_TIMEOUT_S or reduce iterations/topk/depth",
                "use quick/balanced settings for interactive checks",
            ],
            "search_config": {
                "preset": payload.get("search_preset", "quick"),
                "max_depth": _as_int(payload.get("max_steps"), 6, lo=1, hi=20),
                "iterations": _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=500),
                "expansion_topk": _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=500),
                "timeout_s": timeout_s,
            },
        },
        "backend_failures": [
            {
                "category": "backend_timeout",
                "message": f"ChemEnzy native search timed out after {timeout_s}s",
                "target_smiles": target,
                "retryable": True,
                "raw_backend_metadata": {"stdout_tail": stdout, "stderr_tail": stderr},
            }
        ],
        "raw_backend_metadata": {"timeout_s": timeout_s, "stdout_tail": stdout, "stderr_tail": stderr},
    }
    return output


def _tail_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-int(limit):]


def _save_native_raw_output(output: dict[str, Any], raw_out_path: Path) -> None:
    raw_output = copy.deepcopy(output)
    raw_ui_metadata = raw_output.setdefault("ui_metadata", {})
    raw_ui_metadata["saved_at"] = _rel(raw_out_path)
    if output.get("ui_metadata") and output["ui_metadata"].get("saved_at"):
        raw_ui_metadata["filtered_saved_at"] = output["ui_metadata"]["saved_at"]
    raw_out_path.write_text(json.dumps(raw_output, indent=2), encoding="utf-8")


def _save_rejected_routes_output(
    output: dict[str, Any],
    rejected_out_path: Path,
    *,
    removed_items: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    ui_metadata = copy.deepcopy(output.get("ui_metadata") or {})
    filtered_saved_at = ui_metadata.get("saved_at")
    ui_metadata.update(
        {
            "artifact_type": "product_audit_rejected_routes",
            "saved_at": _rel(rejected_out_path),
            "filtered_saved_at": filtered_saved_at,
            "planner_strategy": (
                "Rejected ChemEnzy native routes hidden by AutoPlanner product-audit; "
                "inspect for debugging, not for proposed synthesis."
            ),
        }
    )
    routes: list[dict[str, Any]] = []
    for rejected_rank, item in enumerate(removed_items):
        route = copy.deepcopy(item.get("route") or {})
        audit_meta = route.get("product_audit") or _compact_product_audit_row(item.get("audit") or {}, item.get("risk") or 99)
        route["product_audit"] = audit_meta
        route["post_filter_removed"] = True
        route["rejected_rank"] = rejected_rank
        route["post_filter_remove_reason"] = _product_audit_reason_text(audit_meta)
        routes.append(route)

    artifact = {
        "ok": True,
        "target": output.get("target"),
        "objective": "chem_enzy_native_rejected_routes",
        "n_results": len(routes),
        "time_s": output.get("time_s"),
        "routes": routes,
        "route_set_metrics": {"diversity": {"n_routes": len(routes)}},
        "ui_metadata": ui_metadata,
        "post_filter": copy.deepcopy(output.get("post_filter") or {}),
        "rejection_summary": {
            "removed_route_count": len(routes),
            "target_verdict_counts": audit.get("target_verdict_counts") or {},
            "description": "Routes in this artifact were hidden from the main UI result by product-audit filtering.",
        },
        "search_status": {
            "status": "rejected",
            "solved": False,
            "best_depth": None,
            "message": f"{len(routes)} route(s) hidden by product-audit filtering; these are diagnostic records.",
        },
    }
    rejected_out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _product_audit_reason_text(audit_meta: dict[str, Any]) -> str:
    route_class = str(audit_meta.get("route_class") or "audit_missing")
    issues = [str(issue) for issue in audit_meta.get("issues") or []]
    if issues:
        return f"{route_class}: " + ", ".join(issues[:6])
    return route_class


def _load_statin_showcase_payload() -> dict[str, Any]:
    global _STATIN_SHOWCASE_CACHE
    if not STATIN_SHOWCASE_PATH.exists():
        abort(404, description=f"statin showcase data not found: {_rel(STATIN_SHOWCASE_PATH)}")
    mtime = STATIN_SHOWCASE_PATH.stat().st_mtime
    if _STATIN_SHOWCASE_CACHE and _STATIN_SHOWCASE_CACHE[0] == mtime:
        return _STATIN_SHOWCASE_CACHE[1]
    data = json.loads(STATIN_SHOWCASE_PATH.read_text(encoding="utf-8"))
    _STATIN_SHOWCASE_CACHE = (mtime, data)
    return data


def _statin_showcase_public_payload() -> dict[str, Any]:
    payload = _load_statin_showcase_payload()
    return {
        "ok": True,
        "schema_version": payload.get("schema_version"),
        "created_at": payload.get("created_at"),
        "source_native": payload.get("source_native"),
        "filters": payload.get("filters") or {},
        "aggregate": payload.get("aggregate") or {},
        "targets": [
            _statin_showcase_target_summary(target, include_routes=True)
            for target in payload.get("targets") or []
            if isinstance(target, dict)
        ],
    }


def _statin_showcase_target_summary(target: dict[str, Any], *, include_routes: bool) -> dict[str, Any]:
    summary = {
        "target_name": target.get("target_name"),
        "slug": target.get("slug"),
        "target_smiles": target.get("target_smiles"),
        "cascade_id": target.get("cascade_id"),
        "panel": target.get("panel"),
        "source_solved": bool(target.get("source_solved")),
        "raw_route_count": int(target.get("raw_route_count") or 0),
        "web_kept_route_count": int(target.get("web_kept_route_count") or 0),
        "web_removed_route_count": int(target.get("web_removed_route_count") or 0),
        "short_removed_route_count": int(target.get("short_removed_route_count") or 0),
        "showcase_route_count": int(target.get("showcase_route_count") or 0),
        "route_class_counts_showcase": target.get("route_class_counts_showcase") or {},
        "step_count_distribution_showcase": target.get("step_count_distribution_showcase") or {},
    }
    if include_routes:
        summary["routes"] = [
            _statin_showcase_route_summary(target, route)
            for route in target.get("routes") or []
            if isinstance(route, dict)
        ]
    return summary


def _statin_showcase_route_summary(target: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    audit = route.get("product_audit") or {}
    metrics = route.get("metrics") or {}
    display_rank = int(route.get("display_rank") or route.get("rank") or 0)
    slug = str(target.get("slug") or _slug_text(str(target.get("target_name") or "")))
    return {
        "id": route.get("id") or f"route-{display_rank:04d}",
        "display_rank": display_rank,
        "original_rank": route.get("original_rank"),
        "backend_route_rank": route.get("backend_route_rank"),
        "n_steps": int(route.get("n_steps") or len(route.get("steps") or [])),
        "score": _float_or_none(route.get("score")),
        "solved": bool(route.get("solved")),
        "route_class": audit.get("route_class"),
        "risk_order": audit.get("risk_order"),
        "issues": list(audit.get("issues") or [])[:8],
        "tags": list(audit.get("tags") or [])[:8],
        "metrics": {
            "condition_coverage": metrics.get("condition_coverage"),
            "enzymatic_step_count": metrics.get("enzymatic_step_count"),
            "terminal_stock_count": metrics.get("terminal_stock_count"),
            "max_terminal_heavy_atoms": metrics.get("max_terminal_heavy_atoms"),
        },
        "terminal_profile": _statin_compact_terminal_profile(audit.get("terminal_profile") or {}),
        "condition_audit": _statin_compact_condition_audit(audit.get("condition_audit") or {}),
        "svg_url": f"/api/statins/route-svg/{slug}/{display_rank}",
        "route_url": f"/api/statins/route/{slug}/{display_rank}",
    }


def _find_statin_showcase_target(payload: dict[str, Any], target_key: str) -> dict[str, Any]:
    key = _slug_text(target_key)
    for target in payload.get("targets") or []:
        if not isinstance(target, dict):
            continue
        aliases = [
            target.get("slug"),
            target.get("target_name"),
            target.get("cascade_id"),
        ]
        if key in {_slug_text(str(alias or "")) for alias in aliases}:
            return target
    abort(404, description=f"unknown statin target: {target_key}")


def _find_statin_showcase_route(target: dict[str, Any], route_index: int) -> dict[str, Any]:
    routes = [route for route in target.get("routes") or [] if isinstance(route, dict)]
    if 1 <= route_index <= len(routes):
        route = routes[route_index - 1]
        if int(route.get("display_rank") or route.get("rank") or route_index) == route_index:
            return route
    for route in routes:
        if int(route.get("display_rank") or route.get("rank") or -1) == route_index:
            return route
    abort(404, description=f"unknown statin route: {target.get('target_name')} #{route_index}")


def _statin_compact_terminal_profile(profile: dict[str, Any]) -> dict[str, Any]:
    terminals = profile.get("terminal_reactants") or []
    return {
        "terminal_count": len(terminals) if isinstance(terminals, list) else 0,
        "max_terminal_heavy_atoms": profile.get("max_terminal_heavy_atoms"),
        "effective_max_terminal_heavy_atoms": profile.get("effective_max_terminal_heavy_atoms"),
        "max_terminal_ring_count": profile.get("max_terminal_ring_count"),
        "max_terminal_similarity_to_product": profile.get("max_terminal_similarity_to_product"),
        "product_like_terminal": bool(profile.get("product_like_terminal")),
        "large_polycyclic_terminal": bool(profile.get("large_polycyclic_terminal")),
        "carrier_reagent_count": len(profile.get("carrier_reagents") or []),
        "all_terminals_small": bool(profile.get("all_terminals_small")),
    }


def _statin_compact_condition_audit(condition_audit: dict[str, Any]) -> dict[str, Any]:
    if not condition_audit:
        return {}
    return {
        "route_risk": condition_audit.get("route_risk"),
        "high_risk_step_count": condition_audit.get("high_risk_step_count"),
        "warning_step_count": condition_audit.get("warning_step_count"),
        "temperature_span_c": condition_audit.get("temperature_span_c"),
    }


def _statin_showcase_error_svg(message: str) -> str:
    text = html.escape(message[:220])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="160" viewBox="0 0 920 160">'
        '<rect width="920" height="160" fill="#fff7ed"/>'
        '<text x="28" y="74" font-family="Arial, sans-serif" font-size="18" fill="#9a3412">'
        f"{text}"
        "</text></svg>"
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug_text(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "")).strip("_") or "target"


def _start_plan_job(payload: dict[str, Any]) -> dict[str, Any]:
    target = str(payload.get("target_smiles") or "").strip()
    if Chem.MolFromSmiles(target) is None:
        abort(400, description="target_smiles is not a valid SMILES")
    backend = str(payload.get("planner_backend") or payload.get("planner_mode") or "chem_enzy_native").strip().lower()
    if backend not in CHEMENZY_NATIVE_BACKENDS | CODEX_FULLFLOW_BACKENDS:
        abort(400, description="planner_backend must be chem_enzy_native or codex_fullflow")
    if backend in CODEX_FULLFLOW_BACKENDS:
        _validate_web_codex_controls(payload)
        _validate_web_literature_inputs(payload)

    job_id = "plan_" + _utc_stamp() + "_" + uuid.uuid4().hex[:8]
    log_dir = RESULTS_DIR / "ui_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    preset = str(payload.get("search_preset") or "quick")
    job = {
        "ok": True,
        "job_id": job_id,
        "kind": "plan",
        "status": "queued",
        "planner_backend": backend,
        "label": f"Route search · {preset}",
        "target_smiles": target,
        "target_name": str(payload.get("target_name") or ""),
        "target_preview": _target_preview(target),
        "search_preset": preset,
        "stock_mode": str(payload.get("stock_mode") or "building-block"),
        "stock_names": list(payload.get("stock_names") or []),
        "max_depth": _as_int(payload.get("max_steps"), 6, lo=1, hi=20),
        "iterations": _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=500),
        "expansion_topk": _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=500),
        "device": _resolve_device(str(payload.get("device") or "cpu")),
        "payload": payload,
        "log_path": _rel(log_path),
        "output_json": None,
        "raw_output_json": None,
        "rejected_output_json": None,
        "request_json": None,
        "run_dir": None,
        "agent_blackboard": None,
        "route_forest_html": None,
        "explored_route_forest": None,
        "final_verdict": None,
        "summary": None,
        "return_code": None,
        "error": None,
        "cancel_requested": False,
        "queue_position": None,
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "elapsed_s": None,
    }
    job["label"] = f"{'Agent fullflow' if backend in CODEX_FULLFLOW_BACKENDS else 'Route search'} · {preset}"
    with _LOCK:
        max_active = _as_int(os.environ.get("AUTOPLANNER_WEB_MAX_ACTIVE_JOBS"), 2, lo=1, hi=32)
        active = sum(
            1
            for row in _JOBS.values()
            if str(row.get("status") or "") not in _TERMINAL_JOB_STATUSES
        )
        if active >= max_active:
            abort(429, description=f"active job limit reached ({max_active})")
        _JOBS[job_id] = dict(job)
        _PLAN_JOB_QUEUE.append(job_id)
        _refresh_plan_queue_positions_locked()
        _ensure_plan_worker_locked()
        response = dict(_JOBS[job_id])
    return _job_response(response, include_log=False)


def _ensure_plan_worker_locked() -> None:
    global _PLAN_WORKER_THREAD
    if _PLAN_WORKER_THREAD is not None and _PLAN_WORKER_THREAD.is_alive():
        return
    _PLAN_WORKER_THREAD = threading.Thread(target=_plan_job_worker_loop, name="autoplanner-plan-queue", daemon=True)
    _PLAN_WORKER_THREAD.start()


def _plan_job_worker_loop() -> None:
    global _PLAN_CURRENT_JOB_ID, _PLAN_WORKER_THREAD
    while True:
        job_id = None
        payload: dict[str, Any] = {}
        log_path = RESULTS_DIR / "ui_jobs" / "missing.log"
        with _LOCK:
            while _PLAN_JOB_QUEUE:
                candidate = _PLAN_JOB_QUEUE.popleft()
                job = _JOBS.get(candidate)
                if not job or job.get("kind") != "plan":
                    continue
                if job.get("cancel_requested") or job.get("status") == "cancelled":
                    _mark_plan_job_cancelled_locked(candidate, "cancelled before start")
                    continue
                if job.get("status") != "queued":
                    continue
                job_id = candidate
                _PLAN_CURRENT_JOB_ID = job_id
                payload = dict(job.get("payload") or {})
                log_path = _rooted_path(str(job.get("log_path") or ""))
                _refresh_plan_queue_positions_locked()
                break
            if job_id is None:
                _PLAN_CURRENT_JOB_ID = None
                _PLAN_WORKER_THREAD = None
                _refresh_plan_queue_positions_locked()
                return
        try:
            _run_plan_job(job_id, payload, log_path)
        finally:
            with _LOCK:
                if _PLAN_CURRENT_JOB_ID == job_id:
                    _PLAN_CURRENT_JOB_ID = None
                _PLAN_PROCESS_BY_JOB.pop(job_id, None)
                _refresh_plan_queue_positions_locked()


def _run_plan_job(job_id: str, payload: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with _LOCK:
        if job_id not in _JOBS:
            return
        if _JOBS[job_id].get("cancel_requested"):
            _mark_plan_job_cancelled_locked(job_id, "cancelled before start")
            _append_job_log(log_path, "route search cancelled before start")
            return
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["started_at"] = _utc_now_iso()
        _JOBS[job_id]["queue_position"] = 0
        _JOBS[job_id]["_started_monotonic"] = started
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[{_utc_now_iso()}] route search started\n")
            log.write(f"preset={payload.get('search_preset', 'quick')} max_depth={payload.get('max_steps')} iterations={payload.get('chem_enzy_iterations')} topk={payload.get('chem_enzy_expansion_topk')}\n")
            log.write(f"target={str(payload.get('target_smiles') or '')[:220]}\n")
            log.flush()
            output = _run_plan(payload, job_id=job_id)
            output_path = ((output.get("ui_metadata") or {}).get("saved_at"))
            request_path = ((output.get("ui_metadata") or {}).get("request_path"))
            raw_output_path = ((output.get("ui_metadata") or {}).get("raw_saved_at"))
            rejected_output_path = ((output.get("ui_metadata") or {}).get("rejected_saved_at"))
            run_dir_path = ((output.get("ui_metadata") or {}).get("run_dir"))
            agent_blackboard_path = ((output.get("ui_metadata") or {}).get("agent_blackboard"))
            route_forest_html_path = ((output.get("ui_metadata") or {}).get("route_forest_html"))
            explored_route_forest_path = ((output.get("ui_metadata") or {}).get("explored_route_forest"))
            final_verdict_path = ((output.get("ui_metadata") or {}).get("final_verdict"))
            summary = _plan_output_summary(output)
            log.write(f"[{_utc_now_iso()}] route search finished status={summary['status']} routes={summary['routes']}\n")
            if output_path:
                log.write(f"output_json={output_path}\n")
            if request_path:
                log.write(f"request_json={request_path}\n")
            if raw_output_path:
                log.write(f"raw_output_json={raw_output_path}\n")
            if rejected_output_path:
                log.write(f"rejected_output_json={rejected_output_path}\n")
            failure_analysis = output.get("failure_analysis") or {}
            if failure_analysis.get("diagnosis"):
                log.write("failure_analysis=" + "; ".join(str(row) for row in failure_analysis.get("diagnosis") or []) + "\n")
            if failure_analysis.get("retry_suggestions"):
                log.write("retry_suggestions=" + "; ".join(str(row) for row in failure_analysis.get("retry_suggestions") or []) + "\n")
            log.flush()
        status = "complete"
        error = None
        return_code = 0
    except _PlanJobCancelled as exc:
        output_path = None
        request_path = None
        raw_output_path = None
        rejected_output_path = None
        run_dir_path = None
        agent_blackboard_path = None
        route_forest_html_path = None
        explored_route_forest_path = None
        final_verdict_path = None
        summary = {"status": "cancelled", "message": str(exc), "routes": 0, "solved": False}
        status = "cancelled"
        error = None
        return_code = -15
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_utc_now_iso()}] route search cancelled: {exc}\n")
    except Exception as exc:
        output_path = None
        request_path = None
        raw_output_path = None
        rejected_output_path = None
        run_dir_path = None
        agent_blackboard_path = None
        route_forest_html_path = None
        explored_route_forest_path = None
        final_verdict_path = None
        summary = None
        status = "failed"
        error = getattr(exc, "description", None) or str(exc)
        return_code = 1
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_utc_now_iso()}] route search failed: {error}\n")
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update({
                "status": status,
                "return_code": return_code,
                "summary": summary,
                "error": error,
                "output_json": output_path,
                "raw_output_json": raw_output_path,
                "rejected_output_json": rejected_output_path,
                "request_json": request_path,
                "run_dir": run_dir_path or _JOBS[job_id].get("run_dir"),
                "agent_blackboard": agent_blackboard_path or _JOBS[job_id].get("agent_blackboard"),
                "route_forest_html": route_forest_html_path or _JOBS[job_id].get("route_forest_html"),
                "explored_route_forest": explored_route_forest_path or _JOBS[job_id].get("explored_route_forest"),
                "final_verdict": final_verdict_path or _JOBS[job_id].get("final_verdict"),
                "elapsed_s": round(time.monotonic() - started, 3),
                "finished_at": _utc_now_iso(),
                "queue_position": None,
            })
            _JOBS[job_id].pop("_started_monotonic", None)


def _cancel_job(job_id: str) -> dict[str, Any]:
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        if job.get("kind") != "plan":
            abort(400, description="only route search jobs can be cancelled")
        if job.get("status") in _TERMINAL_JOB_STATUSES:
            return _job_response(dict(job), include_log=True)
        job["cancel_requested"] = True
        job["cancel_requested_at"] = _utc_now_iso()
        log_path = _rooted_path(str(job.get("log_path") or ""))
        if job.get("status") == "queued":
            _remove_from_plan_queue_locked(job_id)
            _mark_plan_job_cancelled_locked(job_id, "cancelled before start")
            _refresh_plan_queue_positions_locked()
            response = dict(job)
            proc = None
        else:
            job["status"] = "cancelling"
            job["error"] = "cancellation requested"
            proc = _PLAN_PROCESS_BY_JOB.get(job_id)
            response = dict(job)
    if log_path is not None:
        _append_job_log(log_path, "route search cancellation requested")
    if proc is not None:
        _terminate_process(proc)
    with _LOCK:
        job = dict(_JOBS.get(job_id) or response)
    return _job_response(job, include_log=True)


def _mark_plan_job_cancelled_locked(job_id: str, message: str) -> None:
    job = _JOBS.get(job_id)
    if not job:
        return
    started = job.get("_started_monotonic")
    elapsed = round(time.monotonic() - float(started), 3) if started else 0.0
    job.update({
        "status": "cancelled",
        "return_code": None,
        "summary": {"status": "cancelled", "message": message, "routes": 0, "solved": False},
        "error": None,
        "finished_at": _utc_now_iso(),
        "elapsed_s": elapsed,
        "queue_position": None,
    })
    job.pop("_started_monotonic", None)


def _remove_from_plan_queue_locked(job_id: str) -> None:
    remaining = [candidate for candidate in _PLAN_JOB_QUEUE if candidate != job_id]
    _PLAN_JOB_QUEUE.clear()
    _PLAN_JOB_QUEUE.extend(remaining)


def _refresh_plan_queue_positions_locked() -> None:
    queued_ids = [
        job_id for job_id in _PLAN_JOB_QUEUE
        if (_JOBS.get(job_id) or {}).get("status") == "queued"
    ]
    for job in _JOBS.values():
        if job.get("kind") != "plan":
            continue
        if job.get("status") in _TERMINAL_JOB_STATUSES:
            job["queue_position"] = None
        elif job.get("status") in {"running", "cancelling"}:
            job["queue_position"] = 0
        elif job.get("status") == "queued":
            job["queue_position"] = None
    for position, job_id in enumerate(queued_ids, start=1):
        if job_id in _JOBS:
            _JOBS[job_id]["queue_position"] = position
            _JOBS[job_id]["queue_size"] = len(queued_ids)


def _plan_job_cancel_requested(job_id: str | None) -> bool:
    if not job_id:
        return False
    with _LOCK:
        return bool((_JOBS.get(job_id) or {}).get("cancel_requested"))


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)


def _append_job_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_utc_now_iso()}] {message}\n")


def _rooted_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _job_response(job: dict[str, Any], *, include_log: bool) -> dict[str, Any]:
    out = dict(job)
    if not include_log:
        out.pop("payload", None)
        out.pop("command", None)
    out.pop("_started_monotonic", None)
    raw_log_path = str(out.get("log_path") or "")
    if raw_log_path:
        log_path = Path(raw_log_path)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        if include_log and log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            out["log_tail"] = lines[-80:]
    if include_log:
        out.update(_agent_runtime_payload(out))
    return out


def _agent_runtime_payload(job: dict[str, Any]) -> dict[str, Any]:
    raw_run_dir = str(job.get("run_dir") or "")
    if not raw_run_dir:
        return {"agent_steps": []}
    run_dir = Path(raw_run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    payload: dict[str, Any] = {"agent_steps": _read_blackboard_step_summary(run_dir)}
    try:
        closeout = validate_latest_closeout_revision(run_dir)
        decision = load_latest_closeout_decision(run_dir)
        manifest = load_latest_closeout_manifest(run_dir)
    except ArtifactRevisionError:
        closeout = validate_latest_closeout_revision(run_dir)
        decision = {}
        manifest = {}
    payload["closeout_revision"] = closeout
    payload["final_verdict_authority"] = (
        "content_addressed_closeout_objects" if decision else "compatibility_path"
    )
    if decision:
        payload["authoritative_final_verdict"] = dict(decision.get("final_verdict") or {})
        payload["authoritative_parent_route_proof"] = dict(
            decision.get("parent_route_proof") or {}
        )
        semantic_drift: list[str] = []
        fixed_final = _read_json_or_empty(run_dir / "final_verdict.json")
        fixed_final.pop("artifact_refs", None)
        fixed_final.pop("artifact_digest_refs", None)
        if fixed_final != payload["authoritative_final_verdict"]:
            semantic_drift.append("final_verdict_compatibility_drift")
        fixed_board = _read_json_or_empty(run_dir / "agent_blackboard.json")
        if dict(fixed_board.get("parent_route_proof") or {}) != payload[
            "authoritative_parent_route_proof"
        ]:
            semantic_drift.append("agent_blackboard_parent_proof_drift")
        payload["closeout_compatibility_semantic_drift"] = semantic_drift
        rows = {
            str(row.get("artifact_id") or ""): dict(row)
            for row in manifest.get("artifacts") or []
            if isinstance(row, dict) and str(row.get("artifact_id") or "")
        }
        for artifact_id in (
            "parent_route_proof_snapshot",
            "final_verdict_core",
            "explored_route_forest",
            "route_forest_html",
        ):
            row = rows.get(artifact_id) or {}
            if row.get("content_path"):
                payload[f"authoritative_{artifact_id}"] = _rel(
                    run_dir / str(row["content_path"])
                )
    for key, filename in {
        "agent_blackboard": "agent_blackboard.json",
        "route_forest_html": "route_forest.html",
        "explored_route_forest": "explored_route_forest.json",
        "final_verdict": "final_verdict.json",
    }.items():
        value = str(job.get(key) or "")
        path = Path(value) if value else run_dir / filename
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            payload[key] = _rel(path)
    return payload


def _plan_output_summary(output: dict[str, Any]) -> dict[str, Any]:
    if output.get("schema_version") == "web_agent_fullflow_result.v1" or output.get("final_verdict"):
        final = dict(output.get("final_verdict") or {})
        counts = dict(output.get("forest_counts") or {})
        ui_metadata = dict(output.get("ui_metadata") or {})
        complete_routes = int(
            counts.get("selected_route_benchmark_routes")
            or counts.get("verified_parent_routes")
            or counts.get("complete_portfolio_routes")
            or 0
        )
        return {
            "status": (output.get("search_status") or {}).get("status") or final.get("route_status") or final.get("verdict"),
            "message": (output.get("search_status") or {}).get("message") or final.get("verdict") or "agent fullflow finished",
            "routes": complete_routes,
            "complete_routes": complete_routes,
            "branches": int(counts.get("explored_branch_views") or counts.get("branches") or 0),
            "l0_advisory": int(counts.get("l0_advisory_branches") or 0),
            "reaction_validated": int(counts.get("reaction_validated_branches") or 0),
            "stock_closed": int(counts.get("stock_closed_branches") or 0),
            "l3_selected_routes": int(
                counts.get("selected_route_benchmark_routes") or 0
            ),
            "l4_procurement_routes": int(
                counts.get("selected_route_procurement_routes") or 0
            ),
            "agent_tasks_completed": int(counts.get("agent_tasks_completed") or 0),
            "agent_tasks_total": int(counts.get("agent_tasks_total") or 0),
            "steps": int(counts.get("steps") or 0),
            "solved": bool(final.get("solved") or final.get("verdict") == "solved"),
            "best_depth": counts.get("steps"),
            "time_s": output.get("time_s"),
            "output_json": ui_metadata.get("saved_at"),
            "run_dir": ui_metadata.get("run_dir"),
            "route_forest_html": ui_metadata.get("route_forest_html"),
            "explored_route_forest": ui_metadata.get("explored_route_forest"),
            "agent_blackboard": ui_metadata.get("agent_blackboard"),
            "final_verdict": ui_metadata.get("final_verdict"),
        }
    routes = output.get("routes") or []
    search_status = output.get("search_status") or {}
    failure_analysis = output.get("failure_analysis") or {}
    ui_metadata = output.get("ui_metadata") or {}
    verifier_gate = (
        (output.get("route_set_metrics") or {}).get("cascade_verifier_gate")
        or ui_metadata.get("cascade_verifier_gate")
        or {}
    )
    learned_annotation = (
        (output.get("route_set_metrics") or {}).get("learned_verifier_annotation")
        or ui_metadata.get("learned_verifier_annotation")
        or {}
    )
    return {
        "status": search_status.get("status"),
        "message": search_status.get("message"),
        "routes": len(routes),
        "solved": bool(search_status.get("solved")),
        "best_depth": search_status.get("best_depth"),
        "time_s": output.get("time_s"),
        "failure_categories": list(failure_analysis.get("failure_categories") or output.get("failure_diagnosis") or []),
        "output_json": ui_metadata.get("saved_at"),
        "raw_output_json": ui_metadata.get("raw_saved_at"),
        "rejected_output_json": ui_metadata.get("rejected_saved_at"),
        "cascade_verifier_gate": {
            "enabled": bool(verifier_gate.get("enabled")),
            "input_routes": int(verifier_gate.get("input_routes") or 0),
            "kept_routes": int(verifier_gate.get("kept_routes") or len(routes)),
            "dropped_routes": int(verifier_gate.get("dropped_routes") or 0),
        },
        "learned_verifier_annotation": {
            "enabled": bool(learned_annotation.get("enabled")),
            "model_loaded": bool(learned_annotation.get("model_loaded")),
            "input_routes": int(learned_annotation.get("input_routes") or 0),
            "annotated_routes": int(learned_annotation.get("annotated_routes") or 0),
            "policy": learned_annotation.get("policy") or "annotation_only",
        },
    }


def _target_preview(smiles: str, limit: int = 64) -> str:
    return smiles if len(smiles) <= limit else smiles[:limit - 1] + "…"


def _plan_depths(payload: dict[str, Any]) -> tuple[str, list[int]]:
    raw_mode = str(payload.get("search_mode") or "").strip().lower()
    if not raw_mode:
        raw_mode = "adaptive" if _as_bool(payload.get("adaptive_depth"), True) else "fixed"
    if raw_mode not in {"adaptive", "fixed"}:
        abort(400, description="search_mode must be adaptive or fixed")

    n_steps = _as_bounded_steps(payload.get("n_steps"), 3, field="n_steps")
    if raw_mode == "fixed":
        return "fixed", [n_steps]

    min_steps = _as_bounded_steps(payload.get("min_steps"), 3, field="min_steps")
    max_steps = _as_bounded_steps(payload.get("max_steps"), MAX_SKELETON_STEPS, field="max_steps")
    if min_steps > max_steps:
        min_steps, max_steps = max_steps, min_steps
    return "adaptive", list(range(min_steps, max_steps + 1))


def _normalize_planner_mode(value: Any) -> tuple[str, str]:
    raw = str(value or DEFAULT_PLANNER_MODE).strip().lower().replace("-", "_")
    aliases = {
        "advanced": DEFAULT_PLANNER_MODE,
        "frontier": DEFAULT_PLANNER_MODE,
        "hybrid": DEFAULT_PLANNER_MODE,
        "and_or": DEFAULT_PLANNER_MODE,
        "stock_and_or": DEFAULT_PLANNER_MODE,
        "stock_andor": DEFAULT_PLANNER_MODE,
        "cascade": DEFAULT_PLANNER_MODE,
        "cascade_skeleton": DEFAULT_PLANNER_MODE,
    }
    if raw not in aliases:
        abort(400, description="planner_mode must be advanced")
    return aliases[raw], raw


def _rank_route_results(results: list[Any], stock_checker) -> list[Any]:
    from cascade_planner.cascadeboard.route_export import diversify_ranked_route_results

    ranked = sorted(
        results,
        key=lambda result: _route_result_rank_key(result, stock_checker),
        reverse=True,
    )
    return diversify_ranked_route_results(ranked)


def _route_result_rank_key(result: Any, stock_checker) -> tuple:
    from cascade_planner.cascadeboard.route_export import route_metrics

    metrics = route_metrics(result.board, stock_checker=stock_checker)
    progress = metrics.get("retrosynthesis_progress") or {}
    natural = metrics.get("route_naturalness") or {}
    strict_stock = metrics.get("strict_stock_solve")
    operation = metrics.get("operation_transitions") or {}
    professional_solved = _professional_solved_from_metrics(metrics)
    return (
        int(professional_solved),
        int(bool(metrics.get("progressive_route"))),
        int(strict_stock is True),
        int(bool(metrics.get("route_solved"))),
        float(progress.get("main_chain_reduction") or 0.0),
        int(bool(metrics.get("filled_route"))),
        float(natural.get("naturalness_score") or 0.0),
        float(operation.get("operation_score") or 0.0),
        -len(operation.get("issues") or []),
        float(result.score or 0.0),
    )


def _results_have_professional_solved(results: list[Any], stock_checker) -> bool:
    from cascade_planner.cascadeboard.route_export import route_metrics

    return any(
        _professional_solved_from_metrics(route_metrics(result.board, stock_checker=stock_checker))
        for result in results
    )


def _depth_attempt_summary(
    depth: int,
    payload: dict[str, Any],
    n_skeletons: int,
    elapsed_s: float,
    planner_used: str = "",
) -> dict[str, Any]:
    routes = payload.get("routes") or []
    best = _route_ui_summary(routes[0]) if routes else None
    return {
        "depth": depth,
        "elapsed_s": round(elapsed_s, 3),
        "n_skeletons": n_skeletons,
        "n_routes": len(routes),
        "planner": planner_used,
        "status": _attempt_status(best),
        "best": best,
    }


def _route_ui_summary(route: dict[str, Any]) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    professional_solved = _professional_solved_from_metrics(metrics)
    diagnostic_solved = _diagnostic_solved_from_metrics(metrics)
    progress = metrics.get("retrosynthesis_progress") or {}
    natural = metrics.get("route_naturalness") or {}
    compat = metrics.get("cascade_compatibility") or {}
    return {
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "filled_route": metrics.get("filled_route"),
        "progressive_route": metrics.get("progressive_route"),
        "route_solved": metrics.get("route_solved"),
        "professional_solved": professional_solved,
        "diagnostic_solved": diagnostic_solved,
        "strict_stock_solve": metrics.get("strict_stock_solve"),
        "main_chain_reduction": progress.get("main_chain_reduction"),
        "largest_leaf_reduction": progress.get("largest_leaf_reduction"),
        "progressive_step_fraction": progress.get("progressive_step_fraction"),
        "terminal_main_heavy_atoms": progress.get("terminal_main_heavy_atoms"),
        "largest_leaf_heavy_atoms": progress.get("largest_leaf_heavy_atoms"),
        "terminal_simplified": progress.get("terminal_simplified"),
        "leaf_simplified": progress.get("leaf_simplified"),
        "naturalness_score": natural.get("naturalness_score"),
        "compatibility_success": compat.get("cascade_compatibility_success"),
        "issues": list(compat.get("issues") or []),
    }


def _annotate_route_statuses(routes: list[dict[str, Any]]) -> None:
    for route in routes:
        metrics = route.get("metrics") or {}
        route["metrics"] = metrics
        metrics["professional_solved"] = _professional_solved_from_metrics(metrics)
        metrics["diagnostic_solved"] = _diagnostic_solved_from_metrics(metrics)


def _professional_solved_from_metrics(metrics: dict[str, Any]) -> bool:
    return bool(metrics.get("route_solved") and metrics.get("progressive_route"))


def _diagnostic_solved_from_metrics(metrics: dict[str, Any]) -> bool:
    return bool(metrics.get("route_solved") and not _professional_solved_from_metrics(metrics))


def _attempt_status(best: dict[str, Any] | None) -> str:
    if not best:
        return "no_route"
    if best.get("professional_solved"):
        return "solved"
    if best.get("diagnostic_solved"):
        return "diagnostic"
    if best.get("progressive_route"):
        return "progressive"
    if best.get("filled_route"):
        return "filled_only"
    return "partial"


def _payload_has_solved_route(payload: dict[str, Any]) -> bool:
    return any(
        _professional_solved_from_metrics(route.get("metrics") or {})
        for route in payload.get("routes") or []
    )


def _plan_search_status(
    payload: dict[str, Any],
    depth_attempts: list[dict[str, Any]],
    *,
    mode: str,
    stopped_on_solved: bool,
) -> dict[str, Any]:
    routes = payload.get("routes") or []
    summaries = [_route_ui_summary(route) for route in routes]
    solved = any(bool(row.get("professional_solved")) for row in summaries)
    diagnostic = any(bool(row.get("diagnostic_solved")) for row in summaries)
    stock_closed = any(bool(row.get("route_solved")) for row in summaries)
    progressive = any(bool(row.get("progressive_route")) for row in summaries)
    best = summaries[0] if summaries else None
    status = "solved" if solved else "partial" if progressive else "diagnostic" if diagnostic else "failed"
    if not routes:
        status = "failed"
    return {
        "mode": mode,
        "status": status,
        "solved": solved,
        "diagnostic": diagnostic,
        "stock_closed": stock_closed,
        "progressive": progressive,
        "best_depth": best.get("n_steps") if best else None,
        "completed_depths": [row.get("depth") for row in depth_attempts],
        "stopped_on_solved": stopped_on_solved,
        "message": _search_status_message(status, best),
    }


def _search_status_message(status: str, best: dict[str, Any] | None) -> str:
    if status == "solved":
        return f"solved at depth {best.get('n_steps')}" if best else "solved"
    if status == "partial":
        return "progressive route found, but terminal reactants are not solved"
    if status == "diagnostic":
        return "stock-closed diagnostic route found, but it is not a progressive retrosynthesis"
    return "no solved retrosynthesis route found within the searched depth range"


def _plan_failure_diagnosis(
    routes: list[dict[str, Any]],
    depth_attempts: list[dict[str, Any]],
) -> list[str]:
    if any(_professional_solved_from_metrics(route.get("metrics") or {}) for route in routes):
        return []

    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not routes:
        add("no_candidate_route_returned")
    if depth_attempts and not any((row.get("n_routes") or 0) > 0 for row in depth_attempts):
        add("candidate_generation_returned_no_routes")
    if any(_diagnostic_solved_from_metrics(route.get("metrics") or {}) for route in routes):
        add("diagnostic_stock_closed_but_not_progressive")

    best_route = routes[0] if routes else None
    metrics = (best_route or {}).get("metrics") or {}
    progress = metrics.get("retrosynthesis_progress") or {}
    natural = metrics.get("route_naturalness") or {}
    compat = metrics.get("cascade_compatibility") or {}

    if metrics:
        if not metrics.get("filled_route"):
            add("route_slots_not_filled")
        if metrics.get("filled_route") and not metrics.get("progressive_route"):
            add("insufficient_retrosynthesis_progress")
        main_reduction = progress.get("main_chain_reduction")
        step_fraction = progress.get("progressive_step_fraction")
        if main_reduction in (None, 0, 0.0):
            add("main_chain_not_reduced")
        elif float(main_reduction or 0.0) < 0.35:
            add("insufficient_main_chain_reduction")
        if step_fraction is not None and float(step_fraction or 0.0) < 0.5:
            add("insufficient_stepwise_disconnection")
        if progress.get("terminal_simplified") is False:
            add("terminal_main_reactant_still_complex")
        if progress.get("leaf_simplified") is False:
            add("largest_leaf_reactant_still_complex")
        if metrics.get("strict_stock_solve") is False:
            add("terminal_reactants_not_all_in_stock")
        if natural.get("naturalness_score") is not None and float(natural.get("naturalness_score") or 0.0) < 1.0:
            add("route_naturalness_artifacts")
        for issue in compat.get("issues") or []:
            add(str(issue))

    if depth_attempts and not any(((row.get("best") or {}).get("progressive_route")) for row in depth_attempts):
        add("no_progressive_route_within_depth_range")
    if depth_attempts and not any(((row.get("best") or {}).get("professional_solved")) for row in depth_attempts):
        add("no_solved_route_within_depth_range")
    return reasons


def _start_eval_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = _utc_stamp() + "_" + uuid.uuid4().hex[:8]
    bench = str(payload.get("bench") or "data/benchmark_v2_100.json")
    bench_path = _safe_path(bench, allowed_roots=[DATA_DIR, RESULTS_DIR, ROOT])
    if not bench_path.exists():
        abort(400, description=f"benchmark file not found: {bench}")

    label = _safe_label(str(payload.get("label") or "ui_eval"))
    out_base = RESULTS_DIR / f"{label}_{job_id}"
    device = _resolve_device(str(payload.get("device") or "cpu"))
    depths = payload.get("depths") or [3, 4]
    if not isinstance(depths, list):
        abort(400, description="depths must be a list")

    cmd = [
        sys.executable,
        "-m",
        "cascade_planner.eval.cc_aostar_depth_benchmark",
        "--bench",
        _rel(bench_path),
        "--output-json",
        _rel(out_base.with_suffix(".json")),
        "--output-csv",
        _rel(out_base.with_suffix(".csv")),
        "--output-md",
        _rel(out_base.with_suffix(".md")),
        "--device",
        device,
        "--n-per-depth",
        str(_as_int(payload.get("n_per_depth"), 3, lo=1, hi=50)),
        "--ultra-targets",
        str(_as_int(payload.get("ultra_targets"), 2, lo=0, hi=50)),
        "--ultra-depth",
        str(_as_int(payload.get("ultra_depth"), 6, lo=1, hi=12)),
        "--skeleton-samples",
        str(_as_int(payload.get("skeleton_samples"), 1, lo=1, hi=20)),
        "--n-results",
        str(_as_int(payload.get("n_results"), 1, lo=1, hi=10)),
        "--candidate-budget",
        str(_as_int(payload.get("candidate_budget"), 2, lo=1, hi=20)),
        "--expansion-multiplier",
        str(_as_int(payload.get("expansion_multiplier"), 4, lo=1, hi=20)),
        "--depths",
        *[str(_as_int(x, 3, lo=1, hi=12)) for x in depths],
    ]

    log_dir = RESULTS_DIR / "ui_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    job = {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "command": cmd,
        "log_path": _rel(log_path),
        "output_json": _rel(out_base.with_suffix(".json")),
        "output_csv": _rel(out_base.with_suffix(".csv")),
        "output_md": _rel(out_base.with_suffix(".md")),
        "summary": None,
        "return_code": None,
        "started_at": None,
        "finished_at": None,
    }
    with _LOCK:
        _JOBS[job_id] = dict(job)
    thread = threading.Thread(target=_run_eval_job, args=(job_id, cmd, log_path), daemon=True)
    thread.start()
    return job


def _run_eval_job(job_id: str, cmd: list[str], log_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    with _LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["started_at"] = _utc_now_iso()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        return_code = proc.wait()

    summary = None
    status = "failed" if return_code else "complete"
    with _LOCK:
        out_json = ROOT / _JOBS[job_id]["output_json"]
    if out_json.exists():
        try:
            summary = json.loads(out_json.read_text(encoding="utf-8")).get("summary")
        except Exception:
            summary = None
    with _LOCK:
        _JOBS[job_id].update({
            "status": status,
            "return_code": return_code,
            "summary": summary,
            "finished_at": _utc_now_iso(),
        })


def _get_retro_engine() -> dict[str, Any]:
    global _RETRO_ENGINE
    if _RETRO_ENGINE is None:
        from cascade_planner.cascadeboard.live_retro import build_live_retro_engine
        _RETRO_ENGINE = build_live_retro_engine()
    return _RETRO_ENGINE


def _get_skeleton_model(model_path: str, device: str):
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        from cascade_planner.cascadeboard.skeleton_inpainter import load_model
        _MODEL_CACHE[key] = load_model(model_path, device=device)
    return _MODEL_CACHE[key]


def _fixed_slots_from_constraints(constraints: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    fixed: dict[int, dict[str, Any]] = {}
    for item in (constraints or {}).get("fixed_steps", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        values = item.get("values") or {}
        if isinstance(values, dict):
            fixed[idx] = dict(values)
    return fixed


def _skeleton_to_dict(skel) -> dict[str, Any]:
    row = {
        "types": list(getattr(skel, "types", [])),
        "ec1s": list(getattr(skel, "ec1s", [])),
        "Ts": list(getattr(skel, "Ts", [])),
        "pHs": list(getattr(skel, "pHs", [])),
        "compatibility": getattr(skel, "compat_pred", ""),
        "operation_mode": getattr(skel, "opmode_pred", ""),
        "issues": list(getattr(skel, "issues_pred", []) or []),
        "log_prob": float(getattr(skel, "log_prob", 0.0) or 0.0),
    }
    retrieval_prior = getattr(skel, "retrieval_prior", None)
    if retrieval_prior:
        row["retrieval_prior"] = retrieval_prior
    reranker_score = getattr(skel, "skeleton_reranker_score", None)
    if reranker_score is not None:
        row["skeleton_reranker_score"] = float(reranker_score)
    return row


def _cuda_status() -> dict[str, Any]:
    global _CUDA_STATUS_CACHE
    now = time.time()
    if _CUDA_STATUS_CACHE and now - _CUDA_STATUS_CACHE[0] < 30:
        return dict(_CUDA_STATUS_CACHE[1])
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        devices = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 5:
                    devices.append({
                        "index": _as_int(parts[0], len(devices), lo=0, hi=128),
                        "name": parts[1],
                        "memory_used_mb": _as_int(parts[2], 0, lo=0, hi=10**9),
                        "memory_total_mb": _as_int(parts[3], 0, lo=0, hi=10**9),
                        "utilization_gpu": _as_int(parts[4], 0, lo=0, hi=100),
                    })
        status = {
            "available": bool(devices),
            "device_count": len(devices),
            "devices": devices,
        }
        _CUDA_STATUS_CACHE = (now, status)
        return dict(status)
    except Exception as exc:
        status = {"available": False, "error": str(exc)}
        _CUDA_STATUS_CACHE = (now, status)
        return dict(status)


def _template_relevance_status() -> dict[str, Any]:
    try:
        return check_template_relevance(ROOT / "vendor/ChemEnzyRetroPlanner")
    except Exception as exc:
        return {"available_count": 0, "models": [], "error": str(exc)}


def _chem_enzy_runtime_status(
    *,
    production: bool = False,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    one_step_models, stock_names, model_overrides = (
        chem_enzy_runtime_selection_from_request(request_payload)
    )
    return diagnose_chem_enzy_runtime(
        vendor_root=ROOT / "vendor" / "ChemEnzyRetroPlanner",
        launcher_path=ROOT / "scripts" / "run_chem_enzy_plan_for_web.py",
        capability_probe=production,
        capability_probe_timeout_s=60.0,
        one_step_models=one_step_models,
        stock_names=stock_names,
        model_overrides=model_overrides,
    )


def _missing_selected_template_relevance_models(payload: dict[str, Any]) -> list[str]:
    selected = list(payload.get("one_step_models") or [])
    if not selected:
        return []
    available = set((_template_relevance_status().get("available_model_names") or []))
    return [
        model
        for model in selected
        if str(model).startswith("template_relevance.") and str(model) not in available
    ]


def _resolve_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return "cpu"


def _artifact_summary() -> dict[str, int]:
    global _ARTIFACT_SUMMARY_CACHE
    now = time.time()
    if _ARTIFACT_SUMMARY_CACHE and now - _ARTIFACT_SUMMARY_CACHE[0] < 10:
        return dict(_ARTIFACT_SUMMARY_CACHE[1])
    files = _list_artifacts()
    counts: dict[str, int] = {}
    for row in files:
        suffix = Path(row["path"]).suffix.lower() or "unknown"
        counts[suffix] = counts.get(suffix, 0) + 1
    _ARTIFACT_SUMMARY_CACHE = (now, counts)
    return dict(counts)


def _list_artifacts(limit: int = 120, *, filter_kind: str | None = None) -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    requested_filter = str(filter_kind or "").strip().lower()
    for path in RESULTS_DIR.glob("**/*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv"}:
            continue
        stat = path.stat()
        tags = _artifact_browser_tags(path)
        if requested_filter and requested_filter not in tags:
            continue
        rows.append({
            "path": _rel(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "tags": sorted(tags),
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


def _artifact_browser_tags(path: Path) -> set[str]:
    tags: set[str] = set()
    name = path.name.lower()
    if "worker" in name:
        tags.add("worker_traces")
    if "reject" in name or "rejected" in name:
        tags.add("rejected")
    if path.suffix.lower() != ".json":
        return tags
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return tags
    if not isinstance(data, dict):
        return tags
    rows = [data]
    if isinstance(data.get("worker_trace"), dict):
        rows.append(dict(data["worker_trace"]))
    for artifact in data.get("artifacts") or []:
        if isinstance(artifact, dict):
            rows.append(artifact)
    for item in rows:
        schema = str(item.get("schema_version") or "")
        artifact_type = str(item.get("artifact_type") or "")
        status = str(item.get("status") or item.get("validation_status") or "")
        if schema.startswith("worker_run_record") or artifact_type == "WorkerRunRecord":
            tags.add("worker_traces")
        if item.get("backend") and item.get("output_validation") is not None:
            tags.add("worker_traces")
        output_validation = item.get("output_validation") or {}
        if status in {"rejected", "rejected_output", "worker_error", "timeout"}:
            tags.add("rejected")
        if isinstance(output_validation, dict) and output_validation.get("accepted") is False:
            tags.add("rejected")
    return tags


def _cascade_demo_payload() -> dict[str, Any]:
    full100 = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/full100_eval/state_action_value_e4_full100_trace_recovered_acceptance.json"
    baseline_full100 = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/full100_eval/learned_baseline_action_source_transition_full100_20260511.json"
    stage2 = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/full100_eval/gap_trace_subset_12_fragment_rank_transition_e4_expansion_minstep0_routeoutcomes.json"
    state_action_gap = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/full100_eval/gap_trace_subset_12_state_action_value_e4_expansion_minstep0_routeoutcomes.json"
    pair_report = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/reports/cascade_action_value_fragment_blend_candidate_e4.json"
    transition_report = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/reports/cascade_transition_value_fragment_rank_candidate_e4.json"
    state_action_report = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/reports/cascade_state_action_value_e4.json"
    state_action_model = ROOT / "results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/models/cascade_state_action_value_e4.pt"

    full_data = _read_json_or_empty(full100)
    stage2_data = _read_json_or_empty(stage2)
    state_action_data = _read_json_or_empty(state_action_gap)
    state_action_report_data = _read_json_or_empty(state_action_report)
    return {
        "ok": True,
        "generated_at": _utc_now_iso(),
        "headline": {
            "title": "AutoPlanner-Cascade",
            "subtitle": "Cascade-native program search around ChemEnzyRetroPlanner multi-step traces",
            "message": "Demo view uses current local artifacts and fixed showcase cases; values are measured, not synthetic.",
        },
        "artifacts": {
            "full100": _artifact_brief(full100),
            "fragment_gap_subset": _artifact_brief(stage2),
            "state_action_gap_subset": _artifact_brief(state_action_gap),
            "fragment_action_report": _artifact_brief(pair_report),
            "fragment_rank_transition_report": _artifact_brief(transition_report),
            "state_action_report": _artifact_brief(state_action_report),
            "state_action_model": _artifact_brief(state_action_model),
            "baseline_full100": _artifact_brief(baseline_full100),
        },
        "cards": _cascade_demo_cards(
            full_data,
            stage2_data,
            state_action_data,
            state_action_report_data,
        ),
        "models": _cascade_model_status(pair_report, transition_report, state_action_report),
        "cases": _select_cascade_demo_cases(full_data, state_action_data, stage2_data, limit=7),
        "next_step": {
            "title": "Next production step: improve route-level recovery",
            "why": "State-action value now runs through full100 and removes low-plausibility failures; GT route recovery is still limited by candidate coverage and route ordering.",
            "training_target": "Q(S,a) action ranking, with result-value rerank kept auxiliary rather than the main model.",
            "data": "Train/val traces from dataset_v4_release; full100 is held as presentation/evaluation evidence.",
        },
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _artifact_brief(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": _rel(path) if exists else str(path.relative_to(ROOT)),
        "exists": exists,
        "size_kb": round(stat.st_size / 1024, 1) if stat else None,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
    }


def _cascade_demo_cards(
    full_data: dict[str, Any],
    stage2_data: dict[str, Any],
    state_action_data: dict[str, Any],
    state_action_report: dict[str, Any],
) -> list[dict[str, Any]]:
    full = full_data.get("summary") or {}
    stage2 = stage2_data.get("summary") or {}
    state_action = state_action_data.get("summary") or {}
    state_action_metrics = state_action_report.get("final_metrics") or {}
    state_action_targets = state_action_data.get("targets") or []
    state_action_topk_exact = _topk_result_rate(state_action_targets, "exact_reaction_hit_count")
    state_action_topk_react = _topk_result_rate(state_action_targets, "gt_reactant_hit_count")
    return [
        _metric_card("Full100 solved", full.get("cascade_solved_rate"), "stock-closed cascade controller outputs"),
        _metric_card("Full100 candidate GT", full.get("candidate_gt_reactant_in_pool"), "proposal-pool GT reactant coverage"),
        _metric_card("Full100 best-route GT", full.get("gt_reactant_in_route_pool"), "best route GT reactant overlap"),
        _metric_card("Hard-gap candidate GT", state_action.get("candidate_gt_reactant_in_pool") or stage2.get("candidate_gt_reactant_in_pool"), "12-target gap subset"),
        _metric_card("Hard-gap top-k exact", state_action_topk_exact, "top-k result pool after state-action scoring"),
        _metric_card("Hard-gap top-k GT", state_action_topk_react, "top-k result pool after state-action scoring"),
        _metric_card("Q top1 positive", state_action_metrics.get("top1_positive_state_hit_rate"), "state-action validation"),
        _metric_card("Avg search time", _seconds_text(full.get("avg_cascade_search_time_s")), "seconds per target inside cascade search"),
    ]


def _metric_card(label: str, value: Any, note: str) -> dict[str, Any]:
    return {"label": label, "value": value, "note": note}


def _seconds_text(value: Any) -> str | None:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return None


def _topk_result_rate(targets: list[dict[str, Any]], hit_key: str) -> float | None:
    if not targets:
        return None
    count = 0
    for target in targets:
        programs = ((target.get("cascade_search") or {}).get("result_programs") or [])
        count += int(any((program.get(hit_key) or 0) > 0 for program in programs))
    return count / len(targets)


def _cascade_model_status(pair_report: Path, transition_report: Path, state_action_report: Path) -> list[dict[str, Any]]:
    action_data = _read_json_or_empty(pair_report)
    transition_data = _read_json_or_empty(transition_report)
    action_metrics = action_data.get("final_metrics") or action_data.get("metrics") or action_data
    transition_metrics = transition_data.get("final_metrics") or transition_data.get("metrics") or transition_data
    state_action_data = _read_json_or_empty(state_action_report)
    state_action_metrics = state_action_data.get("final_metrics") or {}
    return [
        {
            "name": "CascadePairScorer",
            "role": "Local adjacent-step compatibility signal",
            "status": "implemented and validated on pair/fragment packs",
            "tone": "good",
            "metrics": {},
        },
        {
            "name": "Fragment action value",
            "role": "Keeps GT-like generated actions alive before route closure",
            "status": "trained quick e4 candidate",
            "tone": "good",
            "metrics": _compact_metrics(action_metrics, ["auc", "top1_positive_rate", "pairwise_positive_rate", "exact_top5_rate"]),
        },
        {
            "name": "Fragment-rank transition value",
            "role": "Ranks one-step candidate actions within the selected leaf",
            "status": "trained quick e4 candidate",
            "tone": "good",
            "metrics": _compact_metrics(transition_metrics, ["top1_best_transition_rate", "mean_top1_regret", "value_mae", "n_val_pools"]),
        },
        {
            "name": "State-action value",
            "role": "Search value model for Q(S, action)",
            "status": "trained e4; full100 acceptance pending",
            "tone": "warn" if state_action_report.exists() else "bad",
            "metrics": _compact_metrics(
                state_action_metrics,
                [
                    "auc",
                    "top1_positive_state_hit_rate",
                    "top5_positive_state_hit_rate",
                    "pairwise_positive_state_accuracy",
                    "top1_exact_state_hit_rate",
                ],
            ),
        },
    ]


def _compact_metrics(metrics: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out = {}
    for key in keys:
        if key in metrics:
            out[key] = metrics[key]
    return out


def _select_cascade_demo_cases(
    *datasets: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    source_names = ("full100", "state_action_gap_subset", "fragment_gap_subset")
    for idx, data in enumerate(datasets):
        source = source_names[idx] if idx < len(source_names) else f"dataset_{idx + 1}"
        for target in data.get("targets") or []:
            candidates.append(_case_summary(source, target))
    candidates = [case for case in candidates if case["score"] > 0]
    candidates.sort(key=_case_sort_key)
    seen: set[str] = set()
    out = []
    for case in candidates:
        key = str(case.get("target_smiles") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
        if len(out) >= limit:
            break
    return out


_SHOWCASE_TARGET_ORDER = [
    "OC1CC(O)c2ccccc21",
    "CCCCCCC(C)OC(C)=O",
    "OC(c1ccccc1)C(Cc1ccc2c(c1)OCO2)C(O)c1ccccc1",
    "O=C(O)C(O)Cc1ccc(O)c(O)c1",
    "[O-][n+]1ccccc1",
    "O=c1[nH]c2ccccc2o1",
    "CO/N=C(\\C(=O)O)c1ccco1",
]

_SHOWCASE_LABELS = {
    "OC1CC(O)c2ccccc21": "Indanol diol cascade hit",
    "CCCCCCC(C)OC(C)=O": "Chemoenzymatic ester route",
    "OC(c1ccccc1)C(Cc1ccc2c(c1)OCO2)C(O)c1ccccc1": "Three-step alcohol route",
    "O=C(O)C(O)Cc1ccc(O)c(O)c1": "Enzymatic hydroxy-acid hit",
    "[O-][n+]1ccccc1": "Hard-gap N-oxide target",
    "O=c1[nH]c2ccccc2o1": "Hard-gap heterocycle target",
    "CO/N=C(\\C(=O)O)c1ccco1": "Hard-gap chemoenzymatic oxime acid",
}


def _case_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    target = str(case.get("target_smiles") or "")
    try:
        curated_rank = _SHOWCASE_TARGET_ORDER.index(target)
    except ValueError:
        curated_rank = 999
    flags = case.get("flags") or {}
    return (
        curated_rank,
        -int(bool(flags.get("best_exact"))),
        -int(bool(flags.get("best_gt_reactant"))),
        -int(bool(flags.get("topk_exact"))),
        -int(bool(flags.get("topk_gt_reactant"))),
        -int(case.get("step_count") or 0),
        -float(case.get("score") or 0.0),
        -float(case.get("route_score") or -999.0),
    )


def _case_summary(source: str, target: dict[str, Any]) -> dict[str, Any]:
    recovery = target.get("recovery") or target.get("route_recovery") or {}
    cascade = target.get("cascade_search") or {}
    programs = cascade.get("result_programs") or []
    topk_exact = any((program.get("exact_reaction_hit_count") or 0) > 0 for program in programs)
    topk_reactant = any((program.get("gt_reactant_hit_count") or 0) > 0 for program in programs)
    score = (
        10 * int(bool(recovery.get("exact_reaction_in_route_pool")))
        + 5 * int(bool(recovery.get("gt_reactant_in_route_pool")))
        + 3 * int(topk_exact)
        + 2 * int(topk_reactant)
        + int(bool(recovery.get("candidate_exact_reaction_in_pool")))
        + int(bool(recovery.get("candidate_gt_reactant_in_pool")))
        + int(bool(cascade.get("solved")))
    )
    route_rxns = list(cascade.get("route_rxns") or [])
    if not route_rxns and programs:
        route_rxns = list((programs[0] or {}).get("route_rxns") or [])
    gt_route = target.get("gt_route") or []
    return {
        "source": source,
        "score": score,
        "target_smiles": target.get("target_smiles") or target.get("target") or "",
        "label": _SHOWCASE_LABELS.get(target.get("target_smiles") or target.get("target") or ""),
        "route_domain": target.get("route_domain") or "unknown",
        "step_count": cascade.get("step_count"),
        "route_score": cascade.get("score"),
        "stage_count": cascade.get("stage_count"),
        "flags": {
            "stock_closed": bool(cascade.get("stock_closed")),
            "condition_conflict_free": bool(cascade.get("condition_conflict_free")),
            "cofactor_closed": bool(cascade.get("cofactor_closed")),
            "best_exact": bool(recovery.get("exact_reaction_in_route_pool")),
            "best_gt_reactant": bool(recovery.get("gt_reactant_in_route_pool")),
            "candidate_exact": bool(recovery.get("candidate_exact_reaction_in_pool")),
            "candidate_gt_reactant": bool(recovery.get("candidate_gt_reactant_in_pool")),
            "topk_exact": topk_exact,
            "topk_gt_reactant": topk_reactant,
        },
        "recovery": {
            "gt_step_overlap_fraction": recovery.get("gt_step_overlap_fraction"),
            "exact_reaction_hit_count": recovery.get("exact_reaction_hit_count"),
            "gt_reactant_hit_count": recovery.get("gt_reactant_hit_count"),
            "candidate_exact_reaction_hit_count": recovery.get("candidate_exact_reaction_hit_count"),
            "candidate_gt_reactant_hit_count": recovery.get("candidate_gt_reactant_hit_count"),
            "proposal_pool_reaction_count": recovery.get("proposal_pool_reaction_count"),
        },
        "route_rxns": route_rxns[:6],
        "gt_rxns": [step.get("rxn_smiles") for step in gt_route if step.get("rxn_smiles")][:6],
        "programs": [
            {
                "rank": program.get("rank"),
                "score": program.get("score"),
                "exact_reaction_hit_count": program.get("exact_reaction_hit_count"),
                "gt_reactant_hit_count": program.get("gt_reactant_hit_count"),
                "route_rxns": list(program.get("route_rxns") or [])[:4],
            }
            for program in programs[:5]
        ],
    }


def _write_result_artifact(prefix: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    out = RESULTS_DIR / f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _api_run_smiles_first_case(payload: dict[str, Any]) -> dict[str, Any]:
    target = str(payload.get("target_smiles") or "").strip()
    if Chem.MolFromSmiles(target) is None:
        abort(400, description="target_smiles is not a valid SMILES")
    label = _safe_label(str(payload.get("target_name") or payload.get("case_id") or "case"))
    output_dir = payload.get("output_dir")
    if output_dir:
        out_dir = _safe_path(str(output_dir), allowed_roots=[RESULTS_DIR, DATA_DIR])
    else:
        stamp = _utc_stamp()
        out_dir = RESULTS_DIR / "agent_cases" / f"{label}_{stamp}_{uuid.uuid4().hex[:6]}"
    result = run_smiles_first_workflow(
        SmilesFirstWorkflowConfig(
            target_smiles=target,
            target_name=str(payload.get("target_name") or label),
            family_hint=str(payload.get("family_hint") or ""),
            objective=str(payload.get("objective") or "route"),
            output_dir=out_dir,
            frontier_smiles=str(payload.get("frontier_smiles") or ""),
            baseline_json=payload.get("baseline_json"),
            evidence_jsonl=payload.get("evidence_jsonl"),
            db_paths=[str(item) for item in payload.get("db_paths") or []] or None,
            query_budget=_as_int(payload.get("query_budget"), 12, lo=1, hi=100),
            literature_backend=str(payload.get("literature_backend") or "api_json"),
            worker_timeout_s=float(payload.get("worker_timeout_s") or 60.0),
            worker_max_output_bytes=_as_int(payload.get("worker_max_output_bytes"), 200_000, lo=1, hi=2_000_000),
            worker_max_tool_calls=_as_int(payload.get("worker_max_tool_calls"), 8, lo=0, hi=100),
        )
    )
    return {
        "ok": bool((result.get("validation") or {}).get("accepted", False)),
        "schema_version": "web_agent_case_result.v1",
        **result,
    }


def _api_inspect_case_or_blackboard(*, case_bundle: str | None, blackboard: str | None) -> dict[str, Any]:
    if bool(case_bundle) == bool(blackboard):
        abort(400, description="provide exactly one of case_bundle or blackboard")
    if case_bundle:
        bundle = load_case_bundle(_safe_path(case_bundle, allowed_roots=[RESULTS_DIR, DATA_DIR]))
        return {
            "ok": True,
            "schema_version": "web_case_bundle_inspection.v1",
            "case_id": bundle.case_id,
            "route_status": bundle.route_status.value,
            "artifact_count": len(bundle.artifacts),
            "failure_event_count": len(bundle.failure_events),
            "artifact_types": sorted({artifact.artifact_type for artifact in bundle.artifacts}),
            "failure_reasons": [event.reason for event in bundle.failure_events],
            "case_bundle": bundle.to_dict(),
        }
    board = load_blackboard(_safe_path(str(blackboard), allowed_roots=[RESULTS_DIR, DATA_DIR]))
    return {
        "ok": True,
        "schema_version": "web_blackboard_inspection.v1",
        "summary": board.current_summary(),
        "blackboard": board.to_dict(),
    }


def _api_route_audit(payload: dict[str, Any]) -> dict[str, Any]:
    package = dict(payload.get("package") or {})
    if not package and payload.get("package_path"):
        package = _read_json_or_empty(_safe_path(str(payload["package_path"]), allowed_roots=[RESULTS_DIR, DATA_DIR]))
    if not package:
        abort(400, description="package or package_path is required")
    validation = payload.get("validation")
    if validation is None and payload.get("validation_path"):
        validation = _read_json_or_empty(_safe_path(str(payload["validation_path"]), allowed_roots=[RESULTS_DIR, DATA_DIR]))
    report = audit_route_package(
        package,
        validation=dict(validation or {}),
        stock_audit_passed=bool(payload.get("stock_audit_passed")),
        target_match=not bool(payload.get("target_mismatch")),
        condition_candidates=[dict(item) for item in payload.get("condition_candidates") or []],
        enzyme_actions=[dict(item) for item in payload.get("enzyme_actions") or []],
    )
    out = report.to_dict()
    return {
        "ok": True,
        "schema_version": "web_route_audit_result.v1",
        "route_status": report.route_status,
        "audit": out,
        "final_report": _route_audit_final_report(out),
    }


def _api_worker_trace(payload: dict[str, Any]) -> dict[str, Any]:
    task_payload = dict(payload.get("task") or {})
    if not task_payload and payload.get("task_path"):
        task_payload = _read_json_or_empty(_safe_path(str(payload["task_path"]), allowed_roots=[RESULTS_DIR, DATA_DIR]))
    if not task_payload:
        abort(400, description="task or task_path is required")
    mock_output = payload.get("mock_output")
    if mock_output is None and payload.get("mock_output_path"):
        mock_output = _read_json_or_empty(_safe_path(str(payload["mock_output_path"]), allowed_roots=[RESULTS_DIR, DATA_DIR]))
    task = worker_task_from_dict(task_payload)
    task.allowed_workdir = str(ROOT)
    backend = str(payload.get("backend") or os.environ.get("AUTOPLANNER_CODEX_WORKER_BACKEND") or "codex").lower()
    real_execution = backend in {"codex", "api_json"} and mock_output is None and not task.dry_run
    if real_execution and not _as_bool(os.environ.get("AUTOPLANNER_WEB_ENABLE_REAL_WORKER_TRACE"), False):
        abort(403, description="real worker trace execution is disabled on the unauthenticated web surface")
    mock = dict(mock_output) if isinstance(mock_output, dict) else None
    use_codex_cli = backend == "codex" and mock is None and not task.dry_run
    use_api_json = backend == "api_json" and mock is None and not task.dry_run
    record = run_codex_worker(
        task,
        mock_output=mock,
        use_codex_cli=use_codex_cli,
        use_api_json=use_api_json,
    )
    non_real_backend = record.backend in {"mock_output", "dry_run_mock", "default_mock"}
    return {
        "ok": record.status == "accepted_draft",
        "schema_version": "web_worker_trace_result.v1",
        "worker_trace": record.to_dict(),
        "non_real_backend_warning": (
            f"worker backend is {record.backend}; this is not a real Codex/API run"
            if non_real_backend
            else ""
        ),
    }


def _api_guided_policy(payload: dict[str, Any]) -> dict[str, Any]:
    case_bundle_path = str(payload.get("case_bundle") or "")
    if not case_bundle_path:
        abort(400, description="case_bundle is required")
    bundle = load_case_bundle(_safe_path(case_bundle_path, allowed_roots=[RESULTS_DIR, DATA_DIR]))
    operator = compile_strategic_operator_from_case_bundle(
        bundle,
        max_iterations=_as_int(payload.get("max_iterations"), 16, lo=1, hi=500),
        max_depth=_as_int(payload.get("max_depth"), 6, lo=1, hi=20),
        expansion_topk=_as_int(payload.get("expansion_topk"), 50, lo=1, hi=500),
    )
    policy = compile_chem_enzy_search_policy(operator)
    target = str(payload.get("target_smiles") or _target_smiles_from_case_bundle(bundle) or "CCO")
    guided_config = apply_chem_enzy_search_policy(RouteSearchConfig(target_smiles=target), policy)
    return {
        "ok": True,
        "schema_version": "web_guided_policy_result.v1",
        "case_id": bundle.case_id,
        "route_status": bundle.route_status.value,
        "operator": operator.to_dict(),
        "policy": policy.to_dict(),
        "guided_request_payload": {
            "target_smiles": guided_config.target_smiles,
            "chem_enzy_search_policy": policy.to_dict(),
            "chem_enzy_iterations": guided_config.max_iterations,
            "max_steps": guided_config.max_depth,
            "chem_enzy_expansion_topk": guided_config.expansion_topk,
        },
        "rerun_history": {
            "policy_id": policy.policy_id,
            "operator_id": operator.operator_id,
            "evidence_refs": policy.evidence_refs,
            "budget": policy.budget.to_dict(),
        },
    }


def _api_final_report(case_bundle_path: str) -> dict[str, Any]:
    bundle = load_case_bundle(_safe_path(case_bundle_path, allowed_roots=[RESULTS_DIR, DATA_DIR]))
    artifact_types = sorted({artifact.artifact_type for artifact in bundle.artifacts})
    evidence_refs = sorted({ref for artifact in bundle.artifacts for ref in artifact.evidence_refs})
    validation = _artifact_payload(bundle, "RoutePackageValidation")
    package = _artifact_payload(bundle, "HybridRoutePackage")
    audit = audit_route_package(package, validation=validation).to_dict() if package else {}
    return {
        "ok": True,
        "schema_version": "web_final_report.v1",
        "case_id": bundle.case_id,
        "route_status": audit.get("route_status") or bundle.route_status.value,
        "audit": audit,
        "evidence_refs": evidence_refs,
        "condition": audit.get("condition_status") or "unknown",
        "rerun_history": [],
        "artifact_types": artifact_types,
        "failure_events": [event.to_dict() for event in bundle.failure_events],
        "case_bundle": bundle.to_dict(),
    }


def _target_smiles_from_case_bundle(bundle: Any) -> str:
    package = _artifact_payload(bundle, "HybridRoutePackage")
    return str(((package.get("target") or {}).get("smiles")) or "")


def _artifact_payload(bundle: Any, artifact_type: str) -> dict[str, Any]:
    rows = bundle.accepted_artifacts(artifact_type)
    if not rows:
        return {}
    payload = rows[0].payload
    return dict(payload) if isinstance(payload, dict) else {}


def _route_audit_final_report(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "route_audit_final_report.v1",
        "route_status": audit.get("route_status"),
        "audit": audit,
        "target_summary": audit.get("top_route_summary") or {},
        "top_route_summary": audit.get("top_route_summary") or {},
        "stock_audit": {"passed": bool(audit.get("stock_audit_passed"))},
        "step_structural_audit": audit.get("step_structural_audit"),
        "route_mode": audit.get("route_mode"),
        "enzyme_step_status": audit.get("enzyme_step_status"),
        "evidence_status": audit.get("evidence_status"),
        "condition_status": audit.get("condition_status"),
        "fake_closure_rejected_terminals": audit.get("rejected_terminal_list") or [],
        "failure_events": audit.get("failure_events") or [],
        "rerun_history": [],
        "unresolved_core": bool(audit.get("unresolved_core")),
        "next_recommended_action": audit.get("next_action"),
    }


def _safe_path(rel_path: str, *, allowed_roots: list[Path]) -> Path:
    raw = Path(rel_path)
    path = raw if raw.is_absolute() else ROOT / raw
    resolved = path.resolve()
    roots = [p.resolve() for p in allowed_roots]
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        abort(400, description=f"path is outside allowed roots: {rel_path}")
    return resolved


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _safe_label(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip()]
    label = "".join(keep).strip("_")
    return label[:48] or "ui_eval"


def _as_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return max(lo, min(hi, out))


def _as_float(value: Any, default: float, *, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    return max(float(lo), min(float(hi), out))


def _as_bounded_steps(value: Any, default: int, *, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if out < 1 or out > MAX_SKELETON_STEPS:
        abort(400, description=f"{field} must be between 1 and {MAX_SKELETON_STEPS}; the skeleton model supports at most {MAX_SKELETON_STEPS} slots")
    return out


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoPlanner web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
