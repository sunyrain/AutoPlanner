"""Catalog-scoped, read-only readiness projection for route assets."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


TARGET_ROUTE_READINESS_SCHEMA = "target_route_readiness_catalog.v1"
TARGET_ROUTE_ATTESTATION_SCHEMA = "target_route_authority_attestation.v1"
CURRENT_REPLAY_RECEIPT_SCHEMA = "statin_current_canonical_replay_receipt.v1"


class TargetRouteReadinessError(ValueError):
    """Raised when a readiness input is malformed or internally inconsistent."""


def current_replay_attestation_from_receipt(
    receipt: Mapping[str, Any],
    *,
    source_ref: str,
) -> dict[str, Any]:
    """Convert a verified current-replay receipt into display-only authority data."""

    value = _verified_object(receipt, "current_replay_receipt")
    if value.get("schema_version") != CURRENT_REPLAY_RECEIPT_SCHEMA:
        raise TargetRouteReadinessError("current_replay_receipt_schema_invalid")
    gates = _object(value.get("replay_gates"), "current_replay_gates")
    required_gates = (
        "run_validate_accepted",
        "run_replay_accepted",
        "event_replay_digest_equal",
        "graph_oracle_equal",
        "snapshot_reproduced",
        "program_store_current_projection_equal",
        "program_store_replay_valid",
    )
    if any(gates.get(key) is not True for key in required_gates):
        raise TargetRouteReadinessError("current_replay_gate_not_closed")
    semantics = _object(value.get("semantics"), "current_replay_semantics")
    if semantics.get("current_canonical_replay_is_not_route_acceptance") is not True:
        raise TargetRouteReadinessError("current_replay_acceptance_separation_missing")
    target_name = _text(value.get("target_name"), "current_replay_target_name")
    run_id = _text(value.get("run_id"), "current_replay_run_id")
    source_run = _object(value.get("source_run"), "current_replay_source_run")
    workbench = _object(value.get("workbench"), "current_replay_workbench")
    attestation = {
        "schema_version": TARGET_ROUTE_ATTESTATION_SCHEMA,
        "target_name": target_name,
        "run_id": run_id,
        "authority_level": "current_canonical_replay",
        "source_ref": _text(source_ref, "current_replay_source_ref"),
        "source_receipt_sha256": str(value["content_sha256"]),
        "route_accepted": _boolean(workbench.get("accepted"), "route_accepted"),
        "condition_complete": _boolean(
            source_run.get("condition_complete"), "condition_complete"
        ),
        "literature_grounded": _boolean(
            source_run.get("literature_grounded"), "literature_grounded"
        ),
        "process_ready": _boolean(source_run.get("process_ready"), "process_ready"),
        "complete_route_count": _integer(
            source_run.get("complete_route_count"), "complete_route_count"
        ),
        "selected_route_count": _integer(
            source_run.get("selected_route_count"), "selected_route_count"
        ),
        "semantics": {
            "display_attestation_only": True,
            "does_not_grant_route_acceptance": True,
            "receipt_replay_gates_were_verified": True,
        },
    }
    return _with_digest(attestation)


def compile_target_route_readiness(
    catalog: Sequence[Mapping[str, Any]],
    migration_audit: Mapping[str, Any],
    *,
    authority_attestations: Iterable[Mapping[str, Any]] = (),
    minimum_long_route_steps: int = 10,
) -> dict[str, Any]:
    """Project route availability, evidence, conditions and authority separately."""

    if type(minimum_long_route_steps) is not int or minimum_long_route_steps < 1:
        raise TargetRouteReadinessError("minimum_long_route_steps_invalid")
    targets = _catalog_rows(catalog)
    audit = _verified_object(migration_audit, "candidate_migration_audit")
    if audit.get("schema_version") != "candidate_program_migration_audit.v1":
        raise TargetRouteReadinessError("candidate_migration_audit_schema_invalid")
    workbenches = audit.get("workbenches")
    if not isinstance(workbenches, list):
        raise TargetRouteReadinessError("candidate_migration_workbenches_invalid")

    target_keys = {row["target_name"].casefold() for row in targets}
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in target_keys}
    for raw in workbenches:
        row = _object(raw, "candidate_migration_workbench")
        key = str(row.get("target_name") or "").strip().casefold()
        if key in grouped:
            grouped[key].append(row)

    attested: dict[str, list[dict[str, Any]]] = {key: [] for key in target_keys}
    seen_attestations: set[tuple[str, str]] = set()
    for raw in authority_attestations:
        row = _verified_object(raw, "target_route_attestation")
        if row.get("schema_version") != TARGET_ROUTE_ATTESTATION_SCHEMA:
            raise TargetRouteReadinessError("target_route_attestation_schema_invalid")
        key = _text(row.get("target_name"), "attestation_target_name").casefold()
        if key not in attested:
            raise TargetRouteReadinessError("attestation_target_not_in_catalog")
        identity = (key, _text(row.get("source_ref"), "attestation_source_ref"))
        if identity in seen_attestations:
            raise TargetRouteReadinessError("duplicate_target_route_attestation")
        seen_attestations.add(identity)
        if row.get("authority_level") != "current_canonical_replay":
            raise TargetRouteReadinessError("attestation_authority_level_invalid")
        for field in ("route_accepted", "condition_complete", "literature_grounded"):
            _boolean(row.get(field), f"attestation_{field}")
        attested[key].append(row)

    rows = [
        _project_target(
            target,
            grouped[target["target_name"].casefold()],
            attested[target["target_name"].casefold()],
            minimum_long_route_steps=minimum_long_route_steps,
        )
        for target in targets
    ]
    readiness_counts = Counter(row["readiness"] for row in rows)
    report = {
        "schema_version": TARGET_ROUTE_READINESS_SCHEMA,
        "policy": {"minimum_long_route_steps": minimum_long_route_steps},
        "target_count": len(rows),
        "summary": {
            "readiness_counts": dict(sorted(readiness_counts.items())),
            "route_observed": sum(row["observations"]["route_observed"] for row in rows),
            "long_route_observed": sum(
                row["observations"]["long_route_observed"] for row in rows
            ),
            "conditioned_route_observed": sum(
                row["observations"]["condition_observation_edge_count"] > 0
                for row in rows
            ),
            "reported_source_observed": sum(
                row["observations"]["reported_source_ref_count"] > 0 for row in rows
            ),
            "current_canonical_replay_attested": sum(
                row["authority"]["current_canonical_replay_attested"] for row in rows
            ),
            "route_acceptance_attested": sum(
                row["authority"]["route_acceptance_attested"] for row in rows
            ),
        },
        "targets": rows,
        "semantics": {
            "read_only_display_projection": True,
            "target_names_are_identity_not_runtime_rules": True,
            "low_confidence_and_empty_targets_remain_visible": True,
            "portfolio_acceptance_claims_are_not_recomputed_acceptance": True,
            "only_verified_attestations_can_raise_authority_tier": True,
            "canonical_graph_and_program_store_not_modified": True,
            "diagnostic_maxima_may_come_from_different_workbench_snapshots": True,
        },
    }
    return _with_digest(report)


def _project_target(
    target: Mapping[str, Any],
    workbenches: list[dict[str, Any]],
    attestations: list[dict[str, Any]],
    *,
    minimum_long_route_steps: int,
) -> dict[str, Any]:
    ready = [row for row in workbenches if row.get("migration_state") == "projection_ready"]
    empty = sum(row.get("migration_state") == "empty_graph" for row in workbenches)
    invalid = sum(row.get("migration_state") == "invalid_snapshot" for row in workbenches)
    diagnostics = [_object(row.get("source_diagnostics"), "source_diagnostics") for row in ready]
    max_steps = max((_int_value(row, "max_route_steps") for row in diagnostics), default=0)
    complete_routes = max(
        (_int_value(row, "complete_route_count") for row in diagnostics), default=0
    )
    conditions = max(
        (_int_value(row, "condition_observation_edge_count") for row in diagnostics),
        default=0,
    )
    sources = max(
        (_int_value(row, "reported_source_ref_count") for row in diagnostics), default=0
    )
    best_proof = max(
        (
            int(level)
            for row in diagnostics
            for level, count in _object(row.get("proof_level_counts"), "proof_levels").items()
            if str(level).isdigit() and type(count) is int and count > 0
        ),
        default=0,
    )
    accepted_claim = any(row.get("portfolio_accepted_claim") is True for row in diagnostics)
    workbench_source_refs = sorted(
        {
            ref
            for row in ready
            for ref in row.get("source_refs") or []
            if type(ref) is str and ref
        }
    )
    current = attestations[-1] if attestations else {}
    current_attested = bool(current)
    accepted_attested = current.get("route_accepted") is True
    long_route = max_steps >= minimum_long_route_steps
    if current_attested:
        readiness = "current_canonical_accepted" if accepted_attested else "current_canonical_unaccepted"
        confidence = "high" if accepted_attested else "medium"
    elif ready:
        readiness = "candidate_long_route" if long_route else "candidate_short_route"
        confidence = "low"
    elif empty:
        readiness = "empty_workbench_only"
        confidence = "warning"
    else:
        readiness = "not_observed"
        confidence = "warning"

    warnings: list[str] = []
    if not ready:
        warnings.append("NO_ROUTE_OBSERVED")
    if empty and not ready:
        warnings.append("EMPTY_WORKBENCH_ONLY")
    if invalid:
        warnings.append("INVALID_SNAPSHOT_PRESENT")
    if ready and not current_attested:
        warnings.append("CANDIDATE_ONLY_NO_CURRENT_REPLAY")
    if ready and not long_route:
        warnings.append("ONLY_SHORT_ROUTES_OBSERVED")
    if conditions == 0:
        warnings.append("NO_CONDITION_OBSERVATIONS")
    if sources == 0:
        warnings.append("NO_REPORTED_SOURCE_REFS")
    if accepted_claim and not current_attested:
        warnings.append("PORTFOLIO_ACCEPTED_CLAIM_WITHOUT_CURRENT_REPLAY")
    if current_attested and not accepted_attested:
        warnings.append("CURRENT_REPLAY_NOT_ROUTE_ACCEPTED")
    if current_attested and current.get("condition_complete") is not True:
        warnings.append("CURRENT_REPLAY_MISSING_CONDITIONS")
    if current_attested and current.get("literature_grounded") is not True:
        warnings.append("CURRENT_REPLAY_NOT_LITERATURE_GROUNDED")

    return {
        "target_name": target["target_name"],
        "display_name": target["display_name"],
        "scope": str(target.get("scope") or ""),
        "rerun_wave": str(target.get("rerun_wave") or ""),
        "aliases": list(target.get("aliases") or []),
        "readiness": readiness,
        "confidence": confidence,
        "observations": {
            "workbench_count": len(workbenches),
            "projection_ready_count": len(ready),
            "empty_workbench_count": empty,
            "invalid_workbench_count": invalid,
            "route_observed": bool(ready),
            "max_route_steps": max_steps,
            "long_route_observed": long_route,
            "max_complete_route_count": complete_routes,
            "best_proof_level": best_proof,
            "condition_observation_edge_count": conditions,
            "reported_source_ref_count": sources,
            "portfolio_accepted_claim_observed": accepted_claim,
            "workbench_source_refs": workbench_source_refs,
        },
        "authority": {
            "current_canonical_replay_attested": current_attested,
            "route_acceptance_attested": accepted_attested,
            "condition_complete_attested": current.get("condition_complete") is True,
            "literature_grounded_attested": current.get("literature_grounded") is True,
            "source_refs": [row["source_ref"] for row in attestations],
        },
        "warning_codes": warnings,
    }


def _catalog_rows(catalog: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Sequence):
        raise TargetRouteReadinessError("target_catalog_sequence_required")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in catalog:
        row = _object(raw, "target_catalog_row")
        name = _text(row.get("target_name"), "target_catalog_name")
        key = name.casefold()
        if key in seen:
            raise TargetRouteReadinessError("duplicate_target_catalog_name")
        seen.add(key)
        display_name = _text(row.get("display_name") or name, "target_display_name")
        aliases = row.get("aliases") or []
        if not isinstance(aliases, list) or any(type(value) is not str for value in aliases):
            raise TargetRouteReadinessError("target_aliases_invalid")
        rows.append({**row, "target_name": name, "display_name": display_name, "aliases": aliases})
    return rows


def _verified_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    row = _object(value, label)
    digest = row.pop("content_sha256", None)
    if type(digest) is not str or digest != strict_canonical_json_sha256(row):
        raise TargetRouteReadinessError(f"{label}_digest_invalid")
    row["content_sha256"] = digest
    return row


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetRouteReadinessError(f"{label}_object_required")
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TargetRouteReadinessError(f"{label}_invalid")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TargetRouteReadinessError(f"{label}_boolean_required")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TargetRouteReadinessError(f"{label}_nonnegative_integer_required")
    return value


def _int_value(value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key, 0)
    return raw if type(raw) is int and raw >= 0 else 0


__all__ = [
    "CURRENT_REPLAY_RECEIPT_SCHEMA",
    "TARGET_ROUTE_ATTESTATION_SCHEMA",
    "TARGET_ROUTE_READINESS_SCHEMA",
    "TargetRouteReadinessError",
    "compile_target_route_readiness",
    "current_replay_attestation_from_receipt",
]
