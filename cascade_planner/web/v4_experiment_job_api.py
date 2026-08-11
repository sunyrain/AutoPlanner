"""HTTP routes for external experiment job receipts and cancellation."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from flask import Blueprint, jsonify, request

from cascade_planner.web.v4_program_payload import program_innovation_payload
from cascade_planner.web.v4_experiment_transport_api import (
    register_experiment_transport_routes,
)


def register_experiment_job_routes(
    blueprint: Blueprint, factory: Callable[[], Any]
) -> None:
    prefix = "/api/v4/runs/<run_id>/programs/innovations/experiments"
    register_experiment_transport_routes(blueprint, factory)

    @blueprint.post(prefix + "/job")
    def record_route_experiment_job_receipt(run_id: str):
        payload = _payload()
        return jsonify(factory().record_route_experiment_job_receipt(
            run_id, **program_innovation_payload(payload),
            dispatch_id=str(payload.get("dispatch_id") or ""),
            job_receipt=_object(
                payload, "job_receipt", "experiment_job_receipt_must_be_an_object"
            ),
            enable_experiment_job_receipt=(
                payload.get("enable_experiment_job_receipt") is True
            ),
        ))

    @blueprint.post(prefix + "/cancel")
    def request_route_experiment_cancellation(run_id: str):
        payload = _payload()
        return jsonify(factory().request_route_experiment_cancellation(
            run_id, **program_innovation_payload(payload),
            dispatch_id=str(payload.get("dispatch_id") or ""),
            cancellation_request=_object(
                payload, "cancellation_request",
                "experiment_cancellation_request_must_be_an_object",
            ),
            enable_experiment_cancellation=(
                payload.get("enable_experiment_cancellation") is True
            ),
        ))


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


__all__ = ["register_experiment_job_routes"]
