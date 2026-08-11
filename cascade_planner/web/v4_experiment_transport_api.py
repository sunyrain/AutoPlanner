"""HTTP routes for explicit external experiment provider transport."""

from __future__ import annotations

from typing import Any, Callable

from flask import Blueprint, jsonify, request

from cascade_planner.web.v4_program_payload import program_innovation_payload


def register_experiment_transport_routes(
    blueprint: Blueprint, factory: Callable[[], Any]
) -> None:
    prefix = "/api/v4/runs/<run_id>/programs/innovations/experiments/transport"

    for operation, method_name in (
        ("submit", "submit_route_experiment_job"),
        ("poll", "poll_route_experiment_job"),
        ("cancel", "transmit_route_experiment_cancellation"),
    ):
        _register_operation(
            blueprint, factory, prefix=prefix, operation=operation,
            method_name=method_name,
        )


def _register_operation(
    blueprint: Blueprint,
    factory: Callable[[], Any],
    *,
    prefix: str,
    operation: str,
    method_name: str,
) -> None:
    endpoint = f"experiment_transport_{operation}"

    def handler(run_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("json_object_required")
        method = getattr(factory(), method_name)
        return jsonify(method(
            run_id, **program_innovation_payload(payload),
            dispatch_id=str(payload.get("dispatch_id") or ""),
            timeout_s=float(payload.get("timeout_s") or 0.0),
            enable_experiment_transport=(
                payload.get("enable_experiment_transport") is True
            ),
        ))

    blueprint.add_url_rule(
        prefix + f"/{operation}", endpoint=endpoint,
        view_func=handler, methods=["POST"],
    )


__all__ = ["register_experiment_transport_routes"]
