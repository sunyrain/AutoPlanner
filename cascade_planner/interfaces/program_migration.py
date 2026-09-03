"""Read-only cross-run audit for the TransformationProgram migration."""

from __future__ import annotations

from typing import Any, Iterable

from cascade_planner.application.canonical_hypergraph import CanonicalHypergraphError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_MIGRATION_AUDIT_SCHEMA = "transformation_program_migration_audit.v1"


def audit_program_migration(
    gateway: Any,
    *,
    run_ids: Iterable[str] = (),
    limit: int = 100,
) -> dict[str, Any]:
    """Audit selected indexed runs without creating Program store events."""

    selected_ids = {str(value).strip() for value in run_ids if str(value).strip()}
    manifests = list(gateway.list_runs(limit=max(1, min(1_000, int(limit))))["runs"])
    by_id = {str(row.get("run_id") or ""): row for row in manifests}
    missing = sorted(selected_ids - set(by_id))
    if missing:
        raise ValueError("program_migration_runs_not_found:" + ",".join(missing))
    selected = [
        row for row in manifests if not selected_ids or str(row.get("run_id") or "") in selected_ids
    ]
    selected.sort(key=lambda row: str(row.get("run_id") or ""))
    rows = [_audit_run(gateway, manifest) for manifest in selected]
    accepted_count = sum(row["accepted"] is True for row in rows)
    state_counts = {
        state: sum(row.get("migration_state") == state for row in rows)
        for state in (
            "projection_ready",
            "empty_graph",
            "canonical_replay_required",
            "error",
        )
    }
    report = {
        "schema_version": PROGRAM_MIGRATION_AUDIT_SCHEMA,
        "run_count": len(rows),
        "accepted_run_count": accepted_count,
        "rejected_run_count": len(rows) - accepted_count,
        "target_count": len({str(row.get("target_name") or "") for row in rows} - {""}),
        "migration_state_counts": state_counts,
        "runs": rows,
        "semantics": {
            "read_only": True,
            "program_admission_performed": False,
            "target_names_are_labels_not_rules": True,
            "edge_ids_remain_production_route_authority": True,
        },
    }
    report["content_sha256"] = strict_canonical_json_sha256(report)
    return report


def _audit_run(gateway: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    run_id = str(manifest.get("run_id") or "")
    try:
        projection_result = gateway.program_projection(run_id)
        store_result = gateway.program_store(run_id)
    except CanonicalHypergraphError as exc:
        return _error_row(
            manifest,
            migration_state="canonical_replay_required",
            error=f"{type(exc).__name__}:{exc}",
        )
    except Exception as exc:
        return _error_row(
            manifest,
            migration_state="error",
            error=f"{type(exc).__name__}:{exc}",
        )
    projection = projection_result["projection"]
    counts = dict(projection.get("counts") or {})
    source_counts = dict(projection.get("source_counts") or {})
    graph_counts = dict(manifest.get("graph") or {})
    store = store_result["status"]
    checks = {
        "projection_oracle_accepted": (projection_result["oracle"].get("accepted") is True),
        "projection_run_id_bound": projection.get("run_id") == run_id,
        "chemical_state_count_equal": (
            counts.get("chemical_states") == source_counts.get("molecules")
        ),
        "single_operation_per_edge": (counts.get("operation_nodes") == source_counts.get("edges")),
        "single_program_per_edge": (counts.get("programs") == source_counts.get("edges")),
        "route_family_count_equal": (counts.get("routes") == source_counts.get("route_families")),
        "store_current_or_uninitialized": (
            store.get("initialized") is False or store.get("oracle", {}).get("accepted") is True
        ),
        "production_edge_authority_declared": (
            projection.get("semantics", {}).get("edge_ids_remain_production_route_authority")
            is True
        ),
    }
    return {
        "run_id": run_id,
        "target_name": str(manifest.get("target_name") or ""),
        "run_status": str(manifest.get("status") or ""),
        "production_accepted": manifest.get("accepted"),
        "accepted": all(checks.values()),
        "migration_state": (
            "projection_ready" if int(counts.get("programs") or 0) else "empty_graph"
        ),
        "checks": checks,
        "graph_counts": graph_counts,
        "program_counts": counts,
        "source_counts": source_counts,
        "projection_sha256": str(projection.get("content_sha256") or ""),
        "store_initialized": store.get("initialized") is True,
        "store_event_count": int(store.get("event_count") or 0),
        "error": "",
    }


def _error_row(manifest: dict[str, Any], *, migration_state: str, error: str) -> dict[str, Any]:
    return {
        "run_id": str(manifest.get("run_id") or ""),
        "target_name": str(manifest.get("target_name") or ""),
        "accepted": False,
        "migration_state": migration_state,
        "checks": {},
        "error": error,
    }


__all__ = ["PROGRAM_MIGRATION_AUDIT_SCHEMA", "audit_program_migration"]
