"""HTTP routes for bounded experiment handoff operations."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from flask import Blueprint, jsonify, request

from cascade_planner.web.v4_program_payload import program_innovation_payload


def register_experiment_routes(
    blueprint: Blueprint, factory: Callable[[], Any]
) -> None:
    prefix = "/api/v4/runs/<run_id>/programs/innovations/experiments"

    @blueprint.post(prefix + "/audit")
    def audit_route_experiment_result(run_id: str):
        payload = _payload()
        return jsonify(factory().audit_route_experiment_result(
            run_id, **program_innovation_payload(payload),
            result=_object(payload, "result", "experiment_result_must_be_an_object"),
        ))

    @blueprint.post(prefix + "/dispatch")
    def dispatch_route_experiment(run_id: str):
        payload = _payload()
        return jsonify(factory().dispatch_route_experiment(
            run_id, **program_innovation_payload(payload),
            request_id=str(payload.get("request_id") or ""),
            policy=_object(
                payload, "provider_policy",
                "experiment_provider_policy_must_be_an_object",
            ),
            enable_experiment_dispatch=payload.get("enable_experiment_dispatch") is True,
        ))

    @blueprint.post(prefix + "/recover")
    def recover_route_experiment_dispatch(run_id: str):
        payload = _payload()
        return jsonify(factory().recover_route_experiment_dispatch(
            run_id, **program_innovation_payload(payload),
            dispatch_id=str(payload.get("dispatch_id") or ""),
            enable_experiment_dispatch_recovery=(
                payload.get("enable_experiment_dispatch_recovery") is True
            ),
        ))

    @blueprint.post(prefix + "/settle")
    def settle_route_experiment_dispatch(run_id: str):
        payload = _payload()
        return jsonify(factory().settle_route_experiment_dispatch(
            run_id, **program_innovation_payload(payload),
            dispatch_id=str(payload.get("dispatch_id") or ""),
            result=_object(payload, "result", "experiment_result_must_be_an_object"),
            enable_experiment_settlement=(
                payload.get("enable_experiment_settlement") is True
            ),
        ))

    @blueprint.post(prefix + "/artifacts/json")
    def stage_experiment_json_artifact(run_id: str):
        payload = _payload()
        result = factory().stage_experiment_json_artifact(
            run_id,
            artifact=_object(
                payload, "artifact", "experiment_artifact_must_be_an_object"
            ),
            logical_name=str(payload.get("logical_name") or ""),
            enable_experiment_artifact_staging=(
                payload.get("enable_experiment_artifact_staging") is True
            ),
        )
        return jsonify(result), 201


def _payload() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def _object(value: Mapping[str, Any], key: str, reason: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(reason)
    return item


__all__ = ["register_experiment_routes"]
