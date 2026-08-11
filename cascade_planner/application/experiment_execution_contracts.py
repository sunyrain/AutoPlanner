"""Immutable executor-neutral requests for exact-boundary experiments."""

from __future__ import annotations

import math
from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_EXECUTION_REQUEST_SCHEMA = "experiment_execution_request.v1"
EXPERIMENT_DOMAINS = {"biocatalytic", "execution", "mechanism"}
REQUEST_SEMANTICS = {
    "request_is_read_only": True,
    "exact_boundary_is_immutable": True,
    "request_grants_no_scientific_authority": True,
    "request_cannot_mutate_graph_frontier_or_catalog": True,
}


class ExperimentExecutionContractError(ValueError):
    """An experiment request or result is not safely bound."""


def build_experiment_execution_request(
    *,
    run_id: str,
    route_id: str,
    work_item_id: str,
    domain: str,
    plan: Mapping[str, Any],
    canonical_frontier_sha256: str,
    resource_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable, executor-neutral request from a validation plan."""

    plan_value = _digest_object(plan, label="source_plan")
    domain_value = str(domain)
    if domain_value not in EXPERIMENT_DOMAINS:
        raise ExperimentExecutionContractError("experiment_request_domain_invalid")
    boundary = _normalized_boundary(plan_value.get("exact_boundary"))
    checks = _normalized_checks(
        plan_value.get("required_checks") or plan_value.get("required_assays") or []
    )
    output_contract = dict(plan_value.get("required_output_contract") or {})
    if not str(output_contract.get("schema_version") or ""):
        raise ExperimentExecutionContractError("experiment_request_output_schema_missing")
    source_plan = {
        "schema_version": str(plan_value.get("schema_version") or ""),
        "content_sha256": str(plan_value.get("content_sha256") or ""),
    }
    identity = {
        "run_id": str(run_id),
        "route_id": str(route_id),
        "work_item_id": str(work_item_id),
        "domain": domain_value,
        "plan_id": str(plan_value.get("plan_id") or ""),
        "program_id": str(plan_value.get("program_id") or ""),
        "source_plan": source_plan,
        "canonical_frontier_sha256": str(canonical_frontier_sha256),
    }
    if not all(identity.values()) or not _sha256(canonical_frontier_sha256):
        raise ExperimentExecutionContractError("experiment_request_identity_invalid")
    request = {
        "schema_version": EXPERIMENT_EXECUTION_REQUEST_SCHEMA,
        "request_id": "experiment-request:"
        + strict_canonical_json_sha256(identity)[:32],
        **identity,
        "exact_boundary": boundary,
        "required_checks": checks,
        "required_output_contract": output_contract,
        "plan_payload": plan_value,
        "resource_hints": _resource_hints(resource_hints),
        "authority_scope": "experiment_execution_request_only",
        "semantics": dict(REQUEST_SEMANTICS),
    }
    result = _with_digest(request)
    validate_experiment_execution_request(result)
    return result


def validate_experiment_execution_request(value: Mapping[str, Any]) -> None:
    row = dict(value)
    expected = {
        "schema_version", "request_id", "run_id", "route_id", "work_item_id",
        "domain", "plan_id", "program_id", "source_plan",
        "canonical_frontier_sha256", "exact_boundary", "required_checks",
        "required_output_contract", "plan_payload", "resource_hints",
        "authority_scope", "semantics", "content_sha256",
    }
    reasons: list[str] = []
    if set(row) != expected or row.get("schema_version") != EXPERIMENT_EXECUTION_REQUEST_SCHEMA:
        reasons.append("experiment_request_contract_invalid")
    if row.get("domain") not in EXPERIMENT_DOMAINS:
        reasons.append("experiment_request_domain_invalid")
    if row.get("authority_scope") != "experiment_execution_request_only":
        reasons.append("experiment_request_authority_invalid")
    if row.get("semantics") != REQUEST_SEMANTICS or not _digest_valid(row):
        reasons.append("experiment_request_digest_or_semantics_invalid")
    if not all(str(row.get(key) or "") for key in (
        "request_id", "run_id", "route_id", "work_item_id", "plan_id", "program_id"
    )):
        reasons.append("experiment_request_identity_missing")
    if not _sha256(row.get("canonical_frontier_sha256")):
        reasons.append("experiment_request_frontier_digest_invalid")
    try:
        plan = _digest_object(row.get("plan_payload"), label="source_plan")
        if _normalized_boundary(row.get("exact_boundary")) != row.get("exact_boundary"):
            reasons.append("experiment_request_boundary_not_canonical")
        if dict(row.get("source_plan") or {}) != {
            "schema_version": plan.get("schema_version"),
            "content_sha256": plan.get("content_sha256"),
        }:
            reasons.append("experiment_request_source_plan_mismatch")
        if row.get("plan_id") != plan.get("plan_id") or row.get("program_id") != plan.get("program_id"):
            reasons.append("experiment_request_plan_identity_mismatch")
        if row.get("required_checks") != _normalized_checks(
            plan.get("required_checks") or plan.get("required_assays") or []
        ):
            reasons.append("experiment_request_checks_mismatch")
        if row.get("required_output_contract") != plan.get("required_output_contract"):
            reasons.append("experiment_request_output_contract_mismatch")
        if row.get("resource_hints") != _resource_hints(row.get("resource_hints")):
            reasons.append("experiment_request_resource_hints_invalid")
    except (ExperimentExecutionContractError, TypeError, ValueError):
        reasons.append("experiment_request_nested_contract_invalid")
    if reasons:
        raise ExperimentExecutionContractError(";".join(sorted(set(reasons))))


def _normalized_boundary(value: Any) -> dict[str, list[dict[str, str]]]:
    row = dict(value or {})
    result = {
        "input_states": _states(row.get("input_states")),
        "output_states": _states(row.get("output_states")),
    }
    if not result["input_states"] or not result["output_states"]:
        raise ExperimentExecutionContractError("experiment_request_boundary_missing")
    return result


def _states(value: Any) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in value or []:
        row = dict(raw) if isinstance(raw, Mapping) else {}
        state = {
            "state_id": str(row.get("state_id") or ""),
            "molecule_id": str(row.get("molecule_id") or ""),
            "canonical_smiles": str(row.get("canonical_smiles") or ""),
        }
        if not all(state.values()) or not state["state_id"].startswith("state:"):
            raise ExperimentExecutionContractError("experiment_request_state_invalid")
        rows[state["state_id"]] = state
    return [rows[key] for key in sorted(rows)]


def _normalized_checks(value: Any) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in value or []:
        row = dict(raw) if isinstance(raw, Mapping) else {}
        check_id = str(row.get("check_id") or row.get("assay_id") or "")
        if not check_id or row.get("required") is not True:
            raise ExperimentExecutionContractError("experiment_request_check_invalid")
        rows[check_id] = {
            "check_id": check_id,
            "objective": str(row.get("objective") or ""),
            "required": True,
        }
    if not rows:
        raise ExperimentExecutionContractError("experiment_request_checks_missing")
    return [rows[key] for key in sorted(rows)]


def _resource_hints(value: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    raw_estimated_cost = row.get("estimated_cost_units", 0.0)
    if isinstance(raw_estimated_cost, bool) or not isinstance(
        raw_estimated_cost, (int, float)
    ):
        raise ExperimentExecutionContractError("experiment_request_resource_hints_invalid")
    result = {
        "priority_class": str(row.get("priority_class") or "experimental_validation"),
        "timeout_s": float(row.get("timeout_s", 3600.0)),
        "max_artifact_bytes": int(row.get("max_artifact_bytes", 100_000_000)),
        "estimated_cost_units": float(raw_estimated_cost),
    }
    if (
        not result["priority_class"]
        or not math.isfinite(result["timeout_s"])
        or not math.isfinite(result["estimated_cost_units"])
        or result["timeout_s"] <= 0
        or result["max_artifact_bytes"] <= 0
        or result["estimated_cost_units"] < 0
    ):
        raise ExperimentExecutionContractError("experiment_request_resource_hints_invalid")
    return result


def _digest_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentExecutionContractError(f"experiment_{label}_not_object")
    row = dict(value)
    if not _digest_valid(row):
        raise ExperimentExecutionContractError(f"experiment_{label}_digest_invalid")
    return row


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    observed = str(row.pop("content_sha256", ""))
    try:
        return bool(observed) and observed == strict_canonical_json_sha256(row)
    except (TypeError, ValueError):
        return False


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "EXPERIMENT_EXECUTION_REQUEST_SCHEMA",
    "ExperimentExecutionContractError",
    "build_experiment_execution_request",
    "validate_experiment_execution_request",
]
