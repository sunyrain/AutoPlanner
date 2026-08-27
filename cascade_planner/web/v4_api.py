"""Thin V4 HTTP and Web UI adapter over :mod:`campaign_gateway`."""

from __future__ import annotations

from html import escape
import re
from threading import RLock
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
    run_target_job as _run_target_job,
    solve_target_request as _solve_target_request,
)
from cascade_planner.interfaces.target_solve_request import _target_constraints
from cascade_planner.web.v4_target_routes import register_target_routes
from cascade_planner.web.v4_experiment_api import register_experiment_routes
from cascade_planner.web.v4_milestone_api import register_milestone_routes
from cascade_planner.web.v4_live_synthesis import register_live_synthesis_routes
from cascade_planner.web.v4_program_innovation_api import (
    register_program_innovation_routes,
)
from cascade_planner.web.v4_program_overlay_reviews import (
    collect_program_overlay_reviews,
)
from cascade_planner.web.workspace_surface import (
    compiled_mechanism_hypothesis_attachments,
    compiled_program_overlay_attachments,
    inject_workspace_return,
    register_workspace_routes,
)
from cascade_planner.web.workbench_pdf import (
    WorkbenchPdfError,
    render_workbench_pdf,
)


GatewayFactory = Callable[[], CampaignGateway]


def create_v4_blueprint(
    gateway_factory: GatewayFactory | None = None,
) -> Blueprint:
    blueprint = Blueprint("autoplanner_v4", __name__)
    factory = gateway_factory or CampaignGateway
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = RLock()
    register_experiment_routes(blueprint, factory)
    register_program_innovation_routes(blueprint, factory)
    register_workspace_routes(blueprint, factory)
    register_milestone_routes(blueprint, factory, payload_reader=_payload)
    register_target_routes(
        blueprint,
        factory,
        jobs=jobs,
        jobs_lock=jobs_lock,
        payload_reader=_payload,
        solve_target_request=_solve_target_request,
        run_target_job=_run_target_job,
    )
    register_live_synthesis_routes(
        blueprint,
        factory,
        jobs=jobs,
        jobs_lock=jobs_lock,
    )

    @blueprint.errorhandler(CampaignGatewayError)
    def campaign_error(exc: CampaignGatewayError):
        reason = str(exc)
        if reason.startswith("run_not_found:"):
            code = 404
        elif reason.startswith(
            (
                "program_store_",
                "biocatalytic_program_event_",
                "biocatalytic_program_artifact_",
                "biocatalytic_program_store_",
                "mechanism_program_event_",
                "mechanism_program_artifact_",
                "mechanism_program_store_",
                "experimental_claim_event_",
                "experimental_claim_artifact_",
                "experimental_claim_store_",
                "program_experience_library_",
            )
        ):
            code = 500
        elif reason.startswith(
            (
                "program_admission_disabled:",
                "biocatalytic_program_admission_disabled:",
                "experimental_claim_admission_disabled:",
                "mechanism_program_admission_disabled:",
                "program_experience_learning_disabled:",
            )
        ):
            code = 409
        else:
            code = 400
        return jsonify({"error": "campaign_gateway_error", "reason": reason}), code

    @blueprint.errorhandler(ValueError)
    def value_error(exc: ValueError):
        return jsonify({"error": "invalid_request", "reason": str(exc)}), 400

    @blueprint.get("/api/v4/runs")
    def list_runs():
        return jsonify(factory().list_runs(limit=_query_int("limit", 100)))

    @blueprint.get("/api/v4/program-migration")
    def audit_program_migration():
        return jsonify(
            factory().audit_programs(
                run_ids=tuple(request.args.getlist("run_id")),
                limit=_query_int("limit", 100),
            )
        )

    @blueprint.post("/api/v4/runs")
    def create_run():
        payload = _payload()
        acceptance = RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 3),
            minimum_independent_source_groups=_int(payload, "minimum_independent_source_groups", 2),
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
                max_accepted_expansions=_int(payload, "max_accepted_expansions", 8),
                max_attempt_runs=_int(payload, "max_attempt_runs", 12),
            ),
            constraints=_target_constraints(payload),
            global_plan=plan,
            materialize=payload.get("materialize") is True,
            closeout=payload.get("closeout") is True,
        )
        return jsonify(result), 201

    @blueprint.get("/api/v4/runs/<run_id>/status")
    def run_status(run_id: str):
        return jsonify(factory().status(run_id))

    @blueprint.delete("/api/v4/runs/<run_id>/history")
    def remove_run_history(run_id: str):
        with jobs_lock:
            active = any(
                str(value.get("run_id") or "") == run_id
                and str(value.get("status") or "") in {"queued", "running"}
                for value in jobs.values()
            )
            if active:
                return jsonify(
                    {
                        "error": "run_history_removal_conflict",
                        "reason": "active_run_cannot_be_removed_from_history",
                        "run_id": run_id,
                    }
                ), 409
            stale_job_ids = [
                job_id
                for job_id, value in jobs.items()
                if str(value.get("run_id") or "") == run_id
            ]
        result = factory().remove_run_from_history(run_id)
        with jobs_lock:
            for job_id in stale_job_ids:
                jobs.pop(job_id, None)
        return jsonify(result)

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
            factory().apply_plan(run_id, plan, materialize=payload.get("materialize") is True)
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

    @blueprint.get("/api/v4/runs/<run_id>/programs")
    def program_projection(run_id: str):
        return jsonify(factory().program_projection(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/programs/store")
    def program_store(run_id: str):
        return jsonify(factory().program_store(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/programs/routes")
    def route_program_dual_read(run_id: str):
        return jsonify(factory().route_program_dual_read(run_id))

    @blueprint.post("/api/v4/runs/<run_id>/programs/admit")
    def admit_programs(run_id: str):
        payload = _payload()
        result = factory().admit_programs(
            run_id,
            enable_program_admission=(payload.get("enable_program_admission") is True),
        )
        return jsonify(result), 201 if result.get("created") is True else 200

    @blueprint.get("/api/v4/runs/<run_id>/workbench.html")
    def workbench_html(run_id: str) -> Response:
        try:
            gateway = factory()
            snapshot = gateway.workbench(run_id)["snapshot"]
            reviews = collect_program_overlay_reviews(gateway, run_id, snapshot)
            attachments = compiled_program_overlay_attachments(run_id)
            mechanism_attachments = compiled_mechanism_hypothesis_attachments(run_id)
            body = render_v4_route_workbench_html(
                snapshot,
                program_innovation_reviews=reviews,
                program_overlay_attachments=attachments,
                mechanism_hypothesis_attachments=mechanism_attachments,
            )
            return Response(
                inject_workspace_return(body),
                mimetype="text/html",
            )
        except ValueError as exc:
            # An archived display projection can fail its strict integrity check
            # while the immutable campaign snapshot remains available.  Browsers
            # must receive an explainable HTML page, never a raw API JSON body.
            return Response(
                _workbench_error_html(run_id, str(exc)),
                status=422,
                mimetype="text/html",
            )

    @blueprint.get("/api/v4/runs/<run_id>/workbench.pdf")
    def workbench_pdf(run_id: str) -> Response:
        try:
            snapshot = factory().workbench(run_id)["snapshot"]
            pdf = render_workbench_pdf(snapshot)
        except ValueError as exc:
            return Response(
                _workbench_error_html(run_id, str(exc)),
                status=422,
                mimetype="text/html",
            )
        except WorkbenchPdfError as exc:
            return jsonify(
                {
                    "error": "workbench_pdf_unavailable",
                    "reason": str(exc),
                    "run_id": run_id,
                }
            ), 503
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip(".-")
        response = Response(pdf, mimetype="application/pdf")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{filename or "route-workbench"}-retrosynthesis-dossier.pdf"'
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    return blueprint


def _workbench_error_html(run_id: str, reason: str) -> str:
    """Render a bounded fallback for a rejected display projection."""

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>工作台暂不可用</title><style>
body{{margin:0;display:grid;min-height:100vh;place-items:center;background:#f4f6fa;color:#172236;font:14px/1.55 system-ui,sans-serif}}
main{{max-width:620px;margin:24px;padding:28px;border:1px solid #dce2ec;border-radius:16px;background:#fff;box-shadow:0 12px 34px rgba(28,39,75,.07)}}
h1{{margin:0 0 8px;font-size:20px}}p{{color:#59657b}}code{{display:block;overflow:auto;padding:12px;border-radius:9px;background:#f7f8fb;color:#6d7890;font-size:11px;white-space:pre-wrap}}
</style></head><body><main><h1>该运行的工作台暂不可用</h1>
<p>路线投影未通过显示完整性检查；原始运行快照没有被修改。请回到运行中心重载，或选择其他运行。</p>
<p>运行：<strong>{escape(run_id)}</strong></p><code>{escape(reason)}</code></main></body></html>"""


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
