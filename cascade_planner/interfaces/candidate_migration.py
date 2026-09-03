"""Read-only, digest-deduplicated audit of Candidate Program snapshots."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.candidate_programs import (
    candidate_program_projection_oracle,
    project_candidate_route_to_programs,
)
from cascade_planner.application.candidate_route_observations import (
    CandidateProgramError,
    candidate_route_observation_from_workbench,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CANDIDATE_MIGRATION_AUDIT_SCHEMA = "candidate_program_migration_audit.v1"


def audit_candidate_workbench_snapshots(
    snapshots: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Classify Workbench snapshots without mutating canonical or Program stores."""

    grouped: dict[str, dict[str, Any]] = {}
    snapshot_count = 0
    for source_ref, value in snapshots:
        snapshot_count += 1
        source = _json_value(value)
        fingerprint = strict_canonical_json_sha256(source)
        group = grouped.setdefault(
            fingerprint,
            {"snapshot": source, "source_refs": set()},
        )
        group["source_refs"].add(str(source_ref))

    rows = [
        _audit_snapshot(
            fingerprint=fingerprint,
            source_refs=sorted(group["source_refs"]),
            source=group["snapshot"],
        )
        for fingerprint, group in sorted(grouped.items())
    ]
    states = ("projection_ready", "empty_graph", "invalid_snapshot", "error")
    report = {
        "schema_version": CANDIDATE_MIGRATION_AUDIT_SCHEMA,
        "snapshot_count": snapshot_count,
        "unique_workbench_count": len(rows),
        "duplicate_snapshot_count": snapshot_count - len(rows),
        "target_count": len({row["target_name"] for row in rows} - {""}),
        "migration_state_counts": {
            state: sum(row["migration_state"] == state for row in rows) for state in states
        },
        "workbenches": rows,
        "semantics": {
            "read_only": True,
            "identical_snapshots_are_content_deduplicated": True,
            "target_names_are_labels_not_rules": True,
            "canonical_graph_not_modified": True,
            "program_store_admission_performed": False,
            "candidate_projection_never_grants_production_closure": True,
        },
    }
    return _with_digest(report)


def _audit_snapshot(
    *,
    fingerprint: str,
    source_refs: list[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    target = dict(source.get("target") or {})
    base = {
        "snapshot_sha256": fingerprint,
        "source_refs": source_refs,
        "source_copy_count": len(source_refs),
        "run_id": str(source.get("run_id") or ""),
        "target_name": str(target.get("name") or ""),
        "source_diagnostics": _source_diagnostics(source),
    }
    try:
        observation = candidate_route_observation_from_workbench(source)
        projection = project_candidate_route_to_programs(observation)
        oracle = candidate_program_projection_oracle(observation, projection)
        if oracle.get("accepted") is not True:
            raise CandidateProgramError("candidate_program_projection_oracle_failed")
    except CandidateProgramError as exc:
        error = str(exc)
        if (
            error == "candidate_workbench_routes_invalid"
            and source.get("routes") == {}
            and source.get("edges") == {}
        ):
            return {
                **base,
                "migration_state": "empty_graph",
                "accepted": True,
                "source_counts": _source_counts(source),
                "program_counts": {},
                "observation_sha256": "",
                "projection_sha256": "",
                "error": "",
            }
        return {
            **base,
            "migration_state": "invalid_snapshot",
            "accepted": False,
            "source_counts": _source_counts(source),
            "program_counts": {},
            "observation_sha256": "",
            "projection_sha256": "",
            "error": f"{type(exc).__name__}:{error}",
        }
    except Exception as exc:  # pragma: no cover - defensive classification
        return {
            **base,
            "migration_state": "error",
            "accepted": False,
            "source_counts": _source_counts(source),
            "program_counts": {},
            "observation_sha256": "",
            "projection_sha256": "",
            "error": f"{type(exc).__name__}:{exc}",
        }
    return {
        **base,
        "migration_state": "projection_ready",
        "accepted": True,
        "source_counts": _source_counts(source),
        "program_counts": dict(projection["counts"]),
        "observation_sha256": observation["content_sha256"],
        "projection_sha256": projection["content_sha256"],
        "error": "",
    }


def _source_counts(source: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: len(value) if isinstance(value, Mapping) else 0
        for key, value in (
            ("molecules", source.get("molecules")),
            ("edges", source.get("edges")),
            ("routes", source.get("routes")),
        )
    }


def _source_diagnostics(source: Mapping[str, Any]) -> dict[str, Any]:
    edges = _mapping_rows(source.get("edges"))
    routes = _mapping_rows(source.get("routes"))
    inspectors = _mapping_rows(dict(source.get("inspectors") or {}).get("edges"))
    proof_levels = sorted({int(row.get("proof_level") or 0) for row in edges})
    source_refs = {
        str(value)
        for route in routes
        for value in route.get("reported_source_refs") or []
        if str(value)
    }
    source_refs.update(
        str(item.get("source_ref") or "")
        for inspector in inspectors
        for item in inspector.get("sources") or []
        if isinstance(item, Mapping) and str(item.get("source_ref") or "")
    )
    portfolio = dict(source.get("portfolio") or {})
    claim = dict(dict(source.get("campaign_summary") or {}).get("claim") or {})
    return {
        "portfolio_accepted_claim": portfolio.get("accepted") is True,
        "process_ready_claim": portfolio.get("process_ready") is True,
        "campaign_claim_status": str(claim.get("status") or ""),
        "accepted_edge_count": sum(row.get("accepted") is True for row in edges),
        "condition_observation_edge_count": sum(
            bool(row.get("source_observation_records")) for row in inspectors
        ),
        "reported_source_ref_count": len(source_refs),
        "complete_route_count": sum(row.get("complete") is True for row in routes),
        "max_route_steps": max((len(row.get("edge_ids") or []) for row in routes), default=0),
        "proof_level_counts": {
            str(level): sum(int(row.get("proof_level") or 0) == level for row in edges)
            for level in proof_levels
        },
    }


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value.values()] if isinstance(value, Mapping) else []


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_value(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = ["CANDIDATE_MIGRATION_AUDIT_SCHEMA", "audit_candidate_workbench_snapshots"]
