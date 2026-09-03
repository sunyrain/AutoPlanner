"""Fail-closed contracts for read-only Program route candidates."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_ROUTE_CANDIDATE_SCHEMA = "program_route_candidate.v1"
PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA = "program_route_candidate_set.v1"
PROGRAM_ROUTE_SOURCE_KINDS = {
    "baseline",
    "literature",
    "chemical",
    "biocatalytic",
    "whole_cell",
    "hybrid",
    "mechanism",
}
_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "source_kind",
    "source_route_id",
    "program_ids",
    "fallback_program_ids",
    "substitution_program_ids",
    "execution_domains",
    "metrics",
    "eligibility",
    "evidence",
    "warning_codes",
    "semantics",
    "content_sha256",
}
_SET_FIELDS = {
    "schema_version",
    "run_id",
    "route_id",
    "source_graph_revision",
    "source_graph_scientific_sha256",
    "source_route_sha256",
    "source_projection_sha256",
    "source_bundle_sha256",
    "source_mechanism_bundle_sha256",
    "source_execution_bundle_sha256",
    "candidates",
    "counts",
    "unmodeled_objectives",
    "semantics",
    "content_sha256",
}
_METRIC_FIELDS = {
    "physical_operation_count",
    "chemical_step_equivalent_count",
    "net_step_savings",
    "minimum_proof_level",
    "reaction_validation_deficit_count",
    "condition_deficit_count",
    "specialized_validation_deficit_count",
    "procurement_deficit_count",
    "process_deficit_count",
    "source_deficit_count",
    "cofactor_system_count",
    "risk_burden",
    "risk_data_deficit_count",
}
_COUNT_METRICS = _METRIC_FIELDS - {"risk_burden"}
_NONNEGATIVE_METRICS = _METRIC_FIELDS - {"net_step_savings"}
_ELIGIBILITY_FIELDS = {
    "exploration_visible",
    "shadow_optimizer",
    "experimental_ready",
    "process_ready",
    "production_authoritative",
    "route_completion",
}
class ProgramRouteCandidateError(ValueError):
    """Program route candidates failed a structural or authority contract."""
def program_route_candidate_semantics() -> dict[str, bool]:
    """Return immutable-by-copy authority semantics for one candidate."""

    return {
        "candidate_is_read_only": True,
        "program_ids_are_not_production_route_authority": True,
        "candidate_cannot_grant_proof_or_completion": True,
        "source_kind_is_not_an_optimization_objective": True,
    }


def program_route_candidate_set_semantics() -> dict[str, bool]:
    """Return immutable-by-copy authority semantics for a candidate set."""

    return {
        "baseline_fallback_is_always_present": True,
        "unvalidated_candidates_remain_visible": True,
        "unvalidated_substitutions_are_not_shadow_optimizer_eligible": True,
        "source_kind_is_not_an_optimization_objective": True,
        "edge_ids_remain_production_route_authority": True,
        "target_names_are_not_candidate_inputs": True,
        "reported_candidate_routes_are_exploration_only": True,
        "mechanism_one_hop_requires_a_restitched_route": True,
        "unvalidated_mechanism_routes_are_exploration_only": True,
        "unvalidated_execution_programs_are_exploration_only": True,
        "negative_operation_savings_remain_visible": True,
    }


def validate_program_route_candidate_set(value: Mapping[str, Any]) -> list[str]:
    """Validate candidates without granting optimizer or route authority."""

    try:
        source = _strict_object(value, "candidate_set")
    except ProgramRouteCandidateError as exc:
        return [str(exc)]
    reasons: list[str] = []
    if set(source) != _SET_FIELDS:
        reasons.append("program_candidate_set_fields_invalid")
    if source.get("schema_version") != PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA:
        reasons.append("program_candidate_set_schema_invalid")
    if source.get("semantics") != program_route_candidate_set_semantics():
        reasons.append("program_candidate_set_semantics_invalid")
    if not _digest_valid(source):
        reasons.append("program_candidate_set_digest_invalid")
    candidates = source.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        reasons.append("program_candidate_set_empty")
        return sorted(set(reasons))
    baseline_count = 0
    for candidate_id, raw in candidates.items():
        if not isinstance(raw, dict):
            reasons.append(f"program_candidate_not_object:{candidate_id}")
            continue
        row = dict(raw)
        if set(row) != _CANDIDATE_FIELDS:
            reasons.append(f"program_candidate_fields_invalid:{candidate_id}")
        if row.get("schema_version") != PROGRAM_ROUTE_CANDIDATE_SCHEMA:
            reasons.append(f"program_candidate_schema_invalid:{candidate_id}")
        if row.get("candidate_id") != candidate_id:
            reasons.append(f"program_candidate_identity_invalid:{candidate_id}")
        if row.get("source_kind") not in PROGRAM_ROUTE_SOURCE_KINDS:
            reasons.append(f"program_candidate_source_kind_invalid:{candidate_id}")
        if row.get("source_kind") == "baseline":
            baseline_count += 1
        if row.get("semantics") != program_route_candidate_semantics():
            reasons.append(f"program_candidate_semantics_invalid:{candidate_id}")
        if not _digest_valid(row):
            reasons.append(f"program_candidate_digest_invalid:{candidate_id}")
        reasons.extend(_candidate_structure_reasons(candidate_id, row))
    if baseline_count != 1:
        reasons.append("program_candidate_baseline_count_invalid")
    if source.get("counts") != program_route_candidate_counts(candidates):
        reasons.append("program_candidate_set_counts_invalid")
    if not _string_list(source.get("unmodeled_objectives"), allow_empty=False):
        reasons.append("program_candidate_unmodeled_objectives_invalid")
    return sorted(set(reasons))


def _candidate_structure_reasons(candidate_id: str, row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    programs = row.get("program_ids")
    fallback = row.get("fallback_program_ids")
    substitutions = row.get("substitution_program_ids")
    domains = row.get("execution_domains")
    if not _string_list(programs, allow_empty=False):
        reasons.append(f"program_candidate_programs_invalid:{candidate_id}")
    if not _string_list(fallback, allow_empty=False):
        reasons.append(f"program_candidate_fallback_invalid:{candidate_id}")
    if not _string_list(substitutions, allow_empty=True):
        reasons.append(f"program_candidate_substitutions_invalid:{candidate_id}")
    elif isinstance(programs, list) and not set(substitutions).issubset(programs):
        reasons.append(f"program_candidate_substitutions_unselected:{candidate_id}")
    if not _string_list(domains, allow_empty=False):
        reasons.append(f"program_candidate_domains_invalid:{candidate_id}")
    if not _string_list(row.get("warning_codes"), allow_empty=True):
        reasons.append(f"program_candidate_warnings_invalid:{candidate_id}")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not _valid_metrics(metrics):
        reasons.append(f"program_candidate_metrics_invalid:{candidate_id}")
    eligibility = row.get("eligibility")
    if not isinstance(eligibility, dict) or not _valid_eligibility(eligibility):
        reasons.append(f"program_candidate_eligibility_invalid:{candidate_id}")
    evidence = row.get("evidence")
    if not isinstance(evidence, dict) or not _valid_evidence(evidence, metrics):
        reasons.append(f"program_candidate_evidence_invalid:{candidate_id}")
    return reasons


def program_route_candidate_counts(candidates: Mapping[str, Any]) -> dict[str, int]:
    """Return deterministic source and eligibility counts for one candidate map."""

    def eligible(raw: Any, key: str) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get("eligibility"), dict) and raw[
            "eligibility"
        ].get(key) is True

    return {
        "candidates": len(candidates),
        **{
            source_kind: sum(
                isinstance(row, dict) and row.get("source_kind") == source_kind
                for row in candidates.values()
            )
            for source_kind in sorted(PROGRAM_ROUTE_SOURCE_KINDS)
        },
        "shadow_optimizer_eligible": sum(
            eligible(row, "shadow_optimizer") for row in candidates.values()
        ),
        "experimental_ready": sum(
            eligible(row, "experimental_ready") for row in candidates.values()
        ),
        "process_ready": sum(eligible(row, "process_ready") for row in candidates.values()),
    }


def _valid_metrics(value: Mapping[str, Any]) -> bool:
    if set(value) != _METRIC_FIELDS:
        return False
    if any(
        not isinstance(amount, (int, float))
        or isinstance(amount, bool)
        or not math.isfinite(float(amount))
        for amount in value.values()
    ):
        return False
    if any(float(value[key]) < 0 for key in _NONNEGATIVE_METRICS):
        return False
    if any(not isinstance(value[key], int) for key in _COUNT_METRICS):
        return False
    physical = value["physical_operation_count"]
    chemical = value["chemical_step_equivalent_count"]
    return (
        physical > 0
        and chemical > 0
        and value["net_step_savings"] == chemical - physical
    )


def _valid_eligibility(value: Mapping[str, Any]) -> bool:
    if set(value) != _ELIGIBILITY_FIELDS or any(
        not isinstance(flag, bool) for flag in value.values()
    ):
        return False
    return (
        value["exploration_visible"] is True
        and value["production_authoritative"] is False
        and value["route_completion"] is False
        and (not value["experimental_ready"] or value["shadow_optimizer"])
        and (not value["process_ready"] or value["experimental_ready"])
    )


def _valid_evidence(value: Mapping[str, Any], metrics: Any) -> bool:
    return (
        set(value)
        == {
            "source_refs",
            "source_artifact_sha256s",
            "minimum_proof_level",
            "specialized_validation_ids",
            "authority_snapshot_sha256",
        }
        and _string_list(value.get("source_refs"), allow_empty=True)
        and _digest_list(value.get("source_artifact_sha256s"))
        and _string_list(value.get("specialized_validation_ids"), allow_empty=True)
        and isinstance(value.get("authority_snapshot_sha256"), str)
        and bool(value["authority_snapshot_sha256"])
        and isinstance(metrics, dict)
        and value.get("minimum_proof_level") == metrics.get("minimum_proof_level")
    )


def _digest_list(value: Any) -> bool:
    return (
        _string_list(value, allow_empty=False)
        and all(
            len(item) == 64 and all(character in "0123456789abcdef" for character in item)
            for item in value
        )
    )


def _string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    try:
        return bool(observed) and observed == strict_canonical_json_sha256(material)
    except (TypeError, ValueError):
        return False


def _strict_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ProgramRouteCandidateError(
            f"program_candidate_{label}_not_strict_json"
        ) from exc
    if not isinstance(copied, dict):
        raise ProgramRouteCandidateError(f"program_candidate_{label}_not_object")
    return copied


__all__ = [
    "PROGRAM_ROUTE_CANDIDATE_SCHEMA",
    "PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA",
    "PROGRAM_ROUTE_SOURCE_KINDS",
    "ProgramRouteCandidateError",
    "program_route_candidate_counts",
    "program_route_candidate_semantics",
    "program_route_candidate_set_semantics",
    "validate_program_route_candidate_set",
]
