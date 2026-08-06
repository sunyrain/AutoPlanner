"""Stable public projections and identifiers for V4 target jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping
from urllib.parse import quote
from uuid import uuid4


def compact_solve_result(value: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(value.get("gates") or {})
    claim = dict(value.get("claim") or {})
    return {
        "run_id": str(value.get("run_id") or ""),
        "report_path": str(value.get("report_path") or ""),
        "accepted": claim.get("accepted_under_configured_policy") is True,
        "objective_mode": str(claim.get("objective_mode") or "scientific_proof"),
        "objective_achieved": claim.get("objective_achieved") is True,
        "highest_contiguous_gate": str(gates.get("highest_contiguous_gate") or "none"),
        "gates": dict(gates.get("gates") or {}),
        "counts": dict(gates.get("counts") or {}),
        "model_cost": dict(value.get("model_cost") or {}),
        "workbench_url": (
            "/api/v4/runs/"
            + quote(str(value.get("run_id") or ""), safe="")
            + "/workbench.html"
        ),
    }


def job_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "job_id",
            "run_id",
            "target_name",
            "status",
            "phase",
            "created_at",
            "started_at",
            "finished_at",
            "updated_at",
            "elapsed_s",
            "continuation_pass_count",
            "error",
            "result",
        )
    }


def new_run_id(target_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", target_name.lower()).strip("-") or "target"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"v4-{slug[:28]}-{stamp}-{uuid4().hex[:6]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["compact_solve_result", "job_projection", "new_run_id", "utc_now"]
