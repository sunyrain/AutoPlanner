"""Independent closure audit for externally proposed strategic routes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from cascade_planner.application.strategy_experiment_closure_route import (
    compile_imported_route_closure,
)

STRATEGY_TO_EXPERIMENT_CLOSURE_SCHEMA = "strategy_to_experiment_closure.v1"


def compile_strategy_to_experiment_closure(
    *,
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    import_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit only imported routes; unrelated routes cannot close their axes."""
    route_rows = [
        compile_imported_route_closure(imported, graph=graph, portfolio=portfolio)
        for imported in import_receipt.get("routes") or []
    ]
    result = {
        "schema_version": STRATEGY_TO_EXPERIMENT_CLOSURE_SCHEMA,
        "origin_ref": str(import_receipt.get("origin_ref") or ""),
        "provider": str(import_receipt.get("provider") or ""),
        "route_count": len(route_rows),
        "routes": route_rows,
        "next_required_capabilities": _required_capabilities(route_rows),
        "publishable_claim_boundary": (
            "Imported strategy reach is reported separately from independent "
            "reaction, source, condition, stock, and experimental closure."
        ),
        "semantics": {
            "external_claims_grant_no_host_authority": True,
            "closure_is_scoped_to_imported_routes": True,
            "axes_are_independent": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _required_capabilities(routes: list[dict[str, Any]]) -> list[str]:
    checks = (
        ("canonical_materialization", "materialize_admissible_hypotheses"),
        ("host_reaction_validation", "host_reaction_validation"),
        ("exact_source_evidence", "exact_source_binding_and_procedure_extraction"),
        ("complete_exact_conditions", "condition_resolution"),
        ("stock_closure", "stock_oracle_audit"),
    )
    required = [
        capability
        for axis, capability in checks
        if any(
            dict(route.get(axis) or {}).get("status") != "complete" for route in routes
        )
    ]
    return [*required, "experimental_program_validation"]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "STRATEGY_TO_EXPERIMENT_CLOSURE_SCHEMA",
    "compile_strategy_to_experiment_closure",
]
