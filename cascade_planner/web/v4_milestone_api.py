"""Explicit product milestone routes kept outside solver/benchmark adapters."""
from __future__ import annotations

from typing import Any, Callable

from flask import Blueprint, jsonify


def register_milestone_routes(
    blueprint: Blueprint,
    factory: Callable[[], Any],
    *,
    payload_reader: Callable[[], dict[str, Any]],
) -> None:
    prefix = "/api/v4/runs/<run_id>/milestone-subscriptions"

    @blueprint.post(prefix + "/observe")
    def observe_run_milestone(run_id: str):
        payload = payload_reader()
        return jsonify(
            factory().observe_milestone(
                run_id,
                policy=str(payload.get("policy") or ""),
                milestone=str(payload.get("milestone") or "B4_stock_boundary"),
            )
        )

    @blueprint.post(prefix + "/acknowledge")
    def acknowledge_run_milestone(run_id: str):
        payload = payload_reader()
        receipt = payload.get("channel_receipt")
        if receipt is not None and not isinstance(receipt, dict):
            raise ValueError("channel_receipt_must_be_an_object")
        return jsonify(
            factory().acknowledge_milestone_notification(
                run_id,
                channel=str(payload.get("channel") or ""),
                status=str(payload.get("status") or ""),
                channel_receipt=receipt,
            )
        )


__all__ = ["register_milestone_routes"]
