"""Thin V4 HTTP and Web UI adapter over :mod:`campaign_gateway`."""
from __future__ import annotations

from pathlib import Path
from threading import RLock, Thread
from typing import Any, Callable, Mapping

from flask import Blueprint, Response, jsonify, request

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.harness.v4_route_workbench import render_v4_route_workbench_html
from cascade_planner.interfaces.campaign_gateway import (
    CampaignGateway,
    CampaignGatewayError,
)
from cascade_planner.web.v4_target_runtime import (
    historical_job as _historical_job,
    job_projection as _job_projection,
    live_job_progress as _live_job_progress,
    new_run_id as _new_run_id,
    run_target_job as _run_target_job,
    solve_target_request as _solve_target_request,
    utc_now as _utc_now,
)


GatewayFactory = Callable[[], CampaignGateway]


def create_v4_blueprint(
    gateway_factory: GatewayFactory | None = None,
) -> Blueprint:
    blueprint = Blueprint("autoplanner_v4", __name__)
    factory = gateway_factory or CampaignGateway
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = RLock()

    @blueprint.errorhandler(CampaignGatewayError)
    def campaign_error(exc: CampaignGatewayError):
        reason = str(exc)
        code = 404 if reason.startswith("run_not_found:") else 400
        return jsonify({"error": "campaign_gateway_error", "reason": reason}), code

    @blueprint.errorhandler(ValueError)
    def value_error(exc: ValueError):
        return jsonify({"error": "invalid_request", "reason": str(exc)}), 400

    @blueprint.get("/v4")
    def v4_index() -> Response:
        console_path = Path(__file__).resolve().parent / "static" / "v4.html"
        return Response(console_path.read_text(encoding="utf-8"), mimetype="text/html")

    @blueprint.get("/api/v4/runs")
    def list_runs():
        return jsonify(factory().list_runs(limit=_query_int("limit", 100)))

    @blueprint.post("/api/v4/runs")
    def create_run():
        payload = _payload()
        acceptance = RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 3),
            minimum_independent_source_groups=_int(
                payload, "minimum_independent_source_groups", 2
            ),
            stock_boundary=str(payload.get("stock_boundary") or "procurement"),
        )
        plan = payload.get("global_plan")
        if plan is not None and not isinstance(plan, dict):
            raise ValueError("global_plan_must_be_an_object")
        result = factory().create_run(
            target_name=str(payload.get("target_name") or ""),
            target_smiles=str(payload.get("target_smiles") or ""),
            run_id=str(payload.get("run_id") or "") or None,
            acceptance=acceptance,
            budget=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_visual_invocations=0,
                max_accepted_expansions=_int(
                    payload, "max_accepted_expansions", 8
                ),
                max_attempt_runs=_int(payload, "max_attempt_runs", 12),
            ),
            global_plan=plan,
            materialize=payload.get("materialize") is True,
            closeout=payload.get("closeout") is True,
        )
        return jsonify(result), 201

    @blueprint.post("/api/v4/solve-target")
    def solve_target():
        payload = _payload()
        result = _solve_target_request(factory(), payload)
        return jsonify(result), 200 if payload.get("resume") is True else 201

    @blueprint.post("/api/v4/jobs")
    def start_target_job():
        payload = _payload()
        if not str(payload.get("target_smiles") or "").strip():
            raise ValueError("target_smiles_is_required")
        run_id = str(payload.get("run_id") or "") or _new_run_id(
            str(payload.get("target_name") or "target")
        )
        job_id = f"solve:{run_id}"
        payload = {**payload, "run_id": run_id}
        now = _utc_now()
        with jobs_lock:
            existing = jobs.get(job_id)
            if existing and existing.get("status") in {"queued", "running"}:
                return jsonify(_job_projection(existing)), 200
            jobs[job_id] = {
                "job_id": job_id,
                "run_id": run_id,
                "target_name": str(payload.get("target_name") or "blind target"),
                "status": "queued",
                "phase": "queued",
                "created_at": now,
                "started_at": "",
                "finished_at": "",
                "updated_at": now,
                "elapsed_s": 0.0,
                "error": "",
                "result": {},
            }
            row = dict(jobs[job_id])
        Thread(
            target=_run_target_job,
            args=(factory, payload, job_id, jobs, jobs_lock),
            daemon=True,
            name=f"autoplanner-{run_id[:32]}",
        ).start()
        return jsonify(_job_projection(row)), 202

    @blueprint.get("/api/v4/jobs")
    def list_target_jobs():
        with jobs_lock:
            rows = [_job_projection(value) for value in jobs.values()]
        known_run_ids = {str(row.get("run_id") or "") for row in rows}
        for run in factory().list_runs(limit=30).get("runs") or []:
            if not isinstance(run, Mapping):
                continue
            run_id = str(run.get("run_id") or "")
            if run_id and run_id not in known_run_ids:
                rows.append(_historical_job(run))
        rows.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        return jsonify({"jobs": rows})

    @blueprint.get("/api/v4/jobs/<path:job_id>")
    def target_job_status(job_id: str):
        with jobs_lock:
            row = dict(jobs.get(job_id) or {})
        if not row:
            run_id = job_id.removeprefix("solve:")
            historical = next(
                (
                    value
                    for value in factory().list_runs(limit=100).get("runs") or []
                    if isinstance(value, Mapping)
                    and str(value.get("run_id") or "") == run_id
                ),
                None,
            )
            if historical is None:
                return jsonify({"error": "job_not_found", "job_id": job_id}), 404
            row = _historical_job(historical)
        return jsonify(
            {**_job_projection(row), "progress": _live_job_progress(factory, row)}
        )

    @blueprint.get("/api/v4/runs/<run_id>/status")
    def run_status(run_id: str):
        return jsonify(factory().status(run_id))

    @blueprint.post("/api/v4/runs/<run_id>/resume")
    def resume_run(run_id: str):
        payload = _payload()
        return jsonify(
            factory().resume(
                run_id,
                materialize=payload.get("materialize") is True,
                closeout=payload.get("closeout") is True,
            )
        )

    @blueprint.post("/api/v4/runs/<run_id>/plan")
    def apply_plan(run_id: str):
        payload = _payload()
        plan = payload.get("global_plan")
        if not isinstance(plan, dict):
            raise ValueError("global_plan_must_be_an_object")
        return jsonify(
            factory().apply_plan(
                run_id, plan, materialize=payload.get("materialize") is True
            )
        )

    @blueprint.get("/api/v4/runs/<run_id>/validate")
    def validate_run(run_id: str):
        return jsonify(factory().validate(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/replay")
    def replay_run(run_id: str):
        return jsonify(factory().replay(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/benchmark")
    def benchmark_run(run_id: str):
        return jsonify(factory().benchmark(run_id, iterations=_query_int("iterations", 3)))

    @blueprint.get("/api/v4/runs/<run_id>/workbench")
    def workbench(run_id: str):
        return jsonify(factory().workbench(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/workbench.html")
    def workbench_html(run_id: str) -> Response:
        snapshot = factory().workbench(run_id)["snapshot"]
        return Response(render_v4_route_workbench_html(snapshot), mimetype="text/html")

    return blueprint


def _payload() -> dict[str, Any]:
    value = request.get_json(force=False, silent=False)
    if not isinstance(value, dict):
        raise ValueError("request_body_must_be_an_object")
    return value


def _int(value: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(value.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}_must_be_an_integer") from exc


def _query_int(key: str, default: int) -> int:
    try:
        return int(request.args.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}_must_be_an_integer") from exc


__all__ = ["create_v4_blueprint"]
