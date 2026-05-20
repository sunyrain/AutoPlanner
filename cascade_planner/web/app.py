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
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = ROOT / "results" / "v2"
DATA_DIR = ROOT / "data"
STATIN_SHOWCASE_PATH = ROOT / "results" / "shared" / "statin_panel_20260520" / "web_showcase" / "statin_showcase_routes.json"
DEFAULT_MODEL = "results/shared/skeleton_inpainter/best.pt"
MAX_SKELETON_STEPS = 8
DEFAULT_PLANNER_MODE = "advanced"

_RETRO_ENGINE: dict[str, Any] | None = None
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_JOBS: dict[str, dict[str, Any]] = {}
_CUDA_STATUS_CACHE: tuple[float, dict[str, Any]] | None = None
_ARTIFACT_SUMMARY_CACHE: tuple[float, dict[str, int]] | None = None
_STATIN_SHOWCASE_CACHE: tuple[float, dict[str, Any]] | None = None
_LOCK = threading.Lock()
_PLAN_JOB_QUEUE: deque[str] = deque()
_PLAN_WORKER_THREAD: threading.Thread | None = None
_PLAN_CURRENT_JOB_ID: str | None = None
_PLAN_PROCESS_BY_JOB: dict[str, subprocess.Popen] = {}
_TERMINAL_JOB_STATUSES = {"complete", "failed", "cancelled"}


class _PlanJobCancelled(RuntimeError):
    pass


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

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
            "artifacts": _artifact_summary(),
        })

    @app.get("/api/artifacts")
    def artifacts():
        return jsonify({"artifacts": _list_artifacts()})

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
        path = _safe_path(rel_path, allowed_roots=[RESULTS_DIR, DATA_DIR])
        if not path.exists() or not path.is_file():
            abort(404)
        if path.suffix.lower() == ".json":
            return jsonify(json.loads(path.read_text(encoding="utf-8")))
        return Response(path.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain")

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
    backend = str(payload.get("planner_backend") or payload.get("planner_mode") or "chem_enzy_native").strip().lower()
    if backend not in {"chem_enzy", "chem_enzy_native", "chemenzy", "chemenzy_native"}:
        abort(400, description="planner_backend must be chem_enzy_native")
    return _run_chem_enzy_native_plan(payload, job_id=job_id)


def _run_chem_enzy_native_plan(payload: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
    target = str(payload.get("target_smiles") or "").strip()
    if Chem.MolFromSmiles(target) is None:
        abort(400, description="target_smiles is not a valid SMILES")
    if _plan_job_cancel_requested(job_id):
        raise _PlanJobCancelled("route search cancelled before ChemEnzy launch")
    env_prefix = Path(os.environ.get("CHEMENZY_ENV_PREFIX", "/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env"))
    python_bin = env_prefix / "bin" / "python"
    if not python_bin.exists():
        abort(500, description=f"ChemEnzy runtime python not found: {python_bin}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    req_path = RESULTS_DIR / f"ui_chem_enzy_request_{stamp}_{run_id}.json"
    out_path = RESULTS_DIR / f"ui_chem_enzy_plan_{stamp}_{run_id}.json"
    req_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    gpu = 0 if _resolve_device(str(payload.get("device") or "cpu")) == "cuda" else -1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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
    except subprocess.TimeoutExpired as exc:
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
    rejected_out_path = out_path.with_name(f"{out_path.stem}_rejected.json")
    _apply_product_audit_post_filter(output, payload, rejected_out_path=rejected_out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


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
        or "hide_rejects"
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
            "planner_strategy": "CascadePlanner search with ChemEnzy RSPlanner core and AutoPlanner-Cascade hooks",
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
    if backend not in {"chem_enzy", "chem_enzy_native", "chemenzy", "chemenzy_native"}:
        abort(400, description="planner_backend must be chem_enzy_native")

    job_id = "plan_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_dir = RESULTS_DIR / "ui_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    preset = str(payload.get("search_preset") or "quick")
    job = {
        "ok": True,
        "job_id": job_id,
        "kind": "plan",
        "status": "queued",
        "label": f"Route search · {preset}",
        "target_smiles": target,
        "target_preview": _target_preview(target),
        "search_preset": preset,
        "stock_mode": str(payload.get("stock_mode") or "commercial"),
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
        "summary": None,
        "return_code": None,
        "error": None,
        "cancel_requested": False,
        "queue_position": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "finished_at": None,
        "elapsed_s": None,
    }
    with _LOCK:
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
        _JOBS[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
        _JOBS[job_id]["queue_position"] = 0
        _JOBS[job_id]["_started_monotonic"] = started
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[{datetime.utcnow().isoformat()}Z] route search started\n")
            log.write(f"preset={payload.get('search_preset', 'quick')} max_depth={payload.get('max_steps')} iterations={payload.get('chem_enzy_iterations')} topk={payload.get('chem_enzy_expansion_topk')}\n")
            log.write(f"target={str(payload.get('target_smiles') or '')[:220]}\n")
            log.flush()
            output = _run_plan(payload, job_id=job_id)
            output_path = ((output.get("ui_metadata") or {}).get("saved_at"))
            request_path = ((output.get("ui_metadata") or {}).get("request_path"))
            raw_output_path = ((output.get("ui_metadata") or {}).get("raw_saved_at"))
            rejected_output_path = ((output.get("ui_metadata") or {}).get("rejected_saved_at"))
            routes = output.get("routes") or []
            search_status = output.get("search_status") or {}
            failure_analysis = output.get("failure_analysis") or {}
            summary = {
                "status": search_status.get("status"),
                "message": search_status.get("message"),
                "routes": len(routes),
                "solved": bool(search_status.get("solved")),
                "best_depth": search_status.get("best_depth"),
                "time_s": output.get("time_s"),
                "failure_categories": list(failure_analysis.get("failure_categories") or output.get("failure_diagnosis") or []),
                "output_json": output_path,
                "raw_output_json": raw_output_path,
                "rejected_output_json": rejected_output_path,
            }
            log.write(f"[{datetime.utcnow().isoformat()}Z] route search finished status={summary['status']} routes={summary['routes']}\n")
            if output_path:
                log.write(f"output_json={output_path}\n")
            if request_path:
                log.write(f"request_json={request_path}\n")
            if raw_output_path:
                log.write(f"raw_output_json={raw_output_path}\n")
            if rejected_output_path:
                log.write(f"rejected_output_json={rejected_output_path}\n")
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
        summary = {"status": "cancelled", "message": str(exc), "routes": 0, "solved": False}
        status = "cancelled"
        error = None
        return_code = -15
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{datetime.utcnow().isoformat()}Z] route search cancelled: {exc}\n")
    except Exception as exc:
        output_path = None
        request_path = None
        raw_output_path = None
        rejected_output_path = None
        summary = None
        status = "failed"
        error = getattr(exc, "description", None) or str(exc)
        return_code = 1
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{datetime.utcnow().isoformat()}Z] route search failed: {error}\n")
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
                "elapsed_s": round(time.monotonic() - started, 3),
                "finished_at": datetime.utcnow().isoformat() + "Z",
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
        job["cancel_requested_at"] = datetime.utcnow().isoformat() + "Z"
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
        "finished_at": datetime.utcnow().isoformat() + "Z",
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
        log.write(f"[{datetime.utcnow().isoformat()}Z] {message}\n")


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
    return out


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
    job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
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
        _JOBS[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
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
            "finished_at": datetime.utcnow().isoformat() + "Z",
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


def _list_artifacts(limit: int = 120) -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in RESULTS_DIR.glob("**/*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv"}:
            continue
        stat = path.stat()
        rows.append({
            "path": _rel(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


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
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
    stage2_targets = stage2_data.get("targets") or []
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
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _safe_path(rel_path: str, *, allowed_roots: list[Path]) -> Path:
    raw = Path(rel_path)
    path = raw if raw.is_absolute() else ROOT / raw
    resolved = path.resolve()
    roots = [p.resolve() for p in allowed_roots]
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        abort(400, description=f"path is outside allowed roots: {rel_path}")
    return resolved


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


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
