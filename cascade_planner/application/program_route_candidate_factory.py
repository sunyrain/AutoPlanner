"""Shared construction helpers for non-authoritative Program route candidates."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.program_route_candidate_contracts import (
    PROGRAM_ROUTE_CANDIDATE_SCHEMA,
    program_route_candidate_semantics,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def build_program_route_candidate(
    *,
    candidate_id: str,
    source_kind: str,
    source_route_id: str,
    program_ids: list[str],
    fallback_program_ids: list[str],
    substitution_program_ids: list[str],
    execution_domains: list[str],
    metrics: dict[str, float | int],
    shadow_optimizer: bool,
    specialized_validation_ids: list[str],
    source_refs: list[str],
    source_artifact_sha256s: list[str],
    warning_codes: list[str],
    authority_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one digest-bound row without granting any production authority."""

    experimental_ready = shadow_optimizer and all(
        metrics[key] == 0
        for key in (
            "reaction_validation_deficit_count",
            "condition_deficit_count",
            "specialized_validation_deficit_count",
            "source_deficit_count",
        )
    )
    process_ready = experimental_ready and all(
        metrics[key] == 0
        for key in ("procurement_deficit_count", "process_deficit_count")
    )
    return with_program_route_digest(
        {
            "schema_version": PROGRAM_ROUTE_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "source_kind": source_kind,
            "source_route_id": source_route_id,
            "program_ids": list(program_ids),
            "fallback_program_ids": list(fallback_program_ids),
            "substitution_program_ids": list(substitution_program_ids),
            "execution_domains": sorted(set(execution_domains)),
            "metrics": dict(metrics),
            "eligibility": {
                "exploration_visible": True,
                "shadow_optimizer": shadow_optimizer,
                "experimental_ready": experimental_ready,
                "process_ready": process_ready,
                "production_authoritative": False,
                "route_completion": False,
            },
            "evidence": {
                "source_refs": sorted(set(source_refs)),
                "source_artifact_sha256s": sorted(set(source_artifact_sha256s)),
                "minimum_proof_level": metrics["minimum_proof_level"],
                "specialized_validation_ids": sorted(
                    set(specialized_validation_ids)
                ),
                "authority_snapshot_sha256": strict_canonical_json_sha256(
                    authority_snapshot
                ),
            },
            "warning_codes": sorted(set(warning_codes)),
            "semantics": program_route_candidate_semantics(),
        }
    )


def canonical_route_metrics(
    route: Mapping[str, Any],
    *,
    physical: int,
    chemical: int,
    replaced_edges: Iterable[str] = (),
    substitution_validated: bool = False,
    specialized_validation_deficit: int = 0,
    cofactor_systems: int = 0,
) -> dict[str, float | int]:
    """Project current canonical route diagnostics onto common objective axes."""

    original_unproven = set(normalize_strings(route.get("unproven_edge_ids")))
    unproven = set(original_unproven)
    if substitution_validated:
        unproven.difference_update(str(value) for value in replaced_edges)
    if route.get("reaction_validated") is True:
        reaction_deficit = 0
    elif substitution_validated and original_unproven and not unproven:
        reaction_deficit = 0
    else:
        reaction_deficit = max(1, len(unproven))
    condition_deficit = int(route.get("condition_complete") is not True)
    procurement_deficit = int(
        route.get("procurement_closed") is not True
        and route.get("configured_boundary_closed") is not True
    )
    process_deficit = int(route.get("process_ready") is not True)
    source_deficit = int(not normalize_strings(route.get("reported_source_refs")))
    risk = route.get("risk_score")
    risk_known = isinstance(risk, (int, float)) and not isinstance(risk, bool)
    return {
        "physical_operation_count": physical,
        "chemical_step_equivalent_count": chemical,
        "net_step_savings": chemical - physical,
        "minimum_proof_level": nonnegative_integer(
            route.get("proof_level"),
            nonnegative_integer(route.get("minimum_edge_proof_level"), 0),
        ),
        "reaction_validation_deficit_count": reaction_deficit,
        "condition_deficit_count": condition_deficit,
        "specialized_validation_deficit_count": specialized_validation_deficit,
        "procurement_deficit_count": procurement_deficit,
        "process_deficit_count": process_deficit,
        "source_deficit_count": source_deficit,
        "cofactor_system_count": cofactor_systems,
        "risk_burden": max(0.0, float(risk)) if risk_known else 0.0,
        "risk_data_deficit_count": int(not risk_known),
    }


def canonical_route_authority_snapshot(route: Mapping[str, Any]) -> dict[str, Any]:
    """Return only route fields whose authority must not be changed by a candidate."""

    return {
        key: route.get(key)
        for key in (
            "route_id",
            "route_family_id",
            "edge_ids",
            "reaction_validated",
            "condition_complete",
            "procurement_closed",
            "configured_boundary_closed",
            "process_ready",
            "proof_level",
            "reported_source_refs",
            "warning_codes",
            "unproven_edge_ids",
        )
    }


def program_execution_domains(
    program_ids: Iterable[str], programs: Mapping[str, Any]
) -> list[str]:
    return sorted(
        {
            str(dict(programs[value]).get("execution_domain") or "unknown")
            for value in program_ids
            if value in programs
        }
    )


def normalize_strings(value: Any) -> list[str]:
    if value is None:
        return []
    rows = [value] if isinstance(value, str) else value
    return sorted({str(item) for item in rows if str(item)})


def nonnegative_integer(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def with_program_route_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "build_program_route_candidate",
    "canonical_route_authority_snapshot",
    "canonical_route_metrics",
    "nonnegative_integer",
    "normalize_strings",
    "program_execution_domains",
    "with_program_route_digest",
]
