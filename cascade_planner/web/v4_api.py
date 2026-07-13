"""Thin V4 HTTP and Web UI adapter over :mod:`campaign_gateway`."""
from __future__ import annotations

import html
from typing import Any, Callable
from urllib.parse import quote

from flask import Blueprint, Response, jsonify, request

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.harness.v4_route_workbench import (
    render_v4_route_workbench_html,
)
from cascade_planner.interfaces.campaign_gateway import (
    CampaignGateway,
    CampaignGatewayError,
)


GatewayFactory = Callable[[], CampaignGateway]


def create_v4_blueprint(
    gateway_factory: GatewayFactory | None = None,
) -> Blueprint:
    blueprint = Blueprint("autoplanner_v4", __name__)
    factory = gateway_factory or CampaignGateway

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
        rows = factory().list_runs(limit=100)["runs"]
        items = "".join(
            "<li><a href='/api/v4/runs/"
            + quote(str(row["run_id"]), safe="")
            + "/workbench.html'>"
            + html.escape(str(row.get("target_name") or row["run_id"]))
            + "</a> <small>"
            + html.escape(str(row.get("status") or ""))
            + "</small></li>"
            for row in rows
        )
        body = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoPlanner V4 runs</title>
<style>body{{font:16px/1.55 system-ui;margin:3rem auto;max-width:70rem;padding:0 1.5rem;color:#172033}}a{{color:#3659d9}}li{{margin:.7rem 0}}small{{color:#68738a}}</style>
<h1>AutoPlanner V4</h1><p>每个页面都直接投影同一 RunKernel、超图、frontier 与 proof portfolio。</p>
<ul>{items or '<li>暂无运行。请使用 python -m cascade_planner run 创建。</li>'}</ul></html>"""
        return Response(body, mimetype="text/html")

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
                run_id,
                plan,
                materialize=payload.get("materialize") is True,
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
        return jsonify(
            factory().benchmark(run_id, iterations=_query_int("iterations", 3))
        )

    @blueprint.get("/api/v4/runs/<run_id>/workbench")
    def workbench(run_id: str):
        return jsonify(factory().workbench(run_id))

    @blueprint.get("/api/v4/runs/<run_id>/workbench.html")
    def workbench_html(run_id: str) -> Response:
        snapshot = factory().workbench(run_id)["snapshot"]
        return Response(
            render_v4_route_workbench_html(snapshot),
            mimetype="text/html",
        )

    return blueprint


def _payload() -> dict[str, Any]:
    value = request.get_json(force=False, silent=False)
    if not isinstance(value, dict):
        raise ValueError("request_body_must_be_an_object")
    return value


def _int(value: dict[str, Any], key: str, default: int) -> int:
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
