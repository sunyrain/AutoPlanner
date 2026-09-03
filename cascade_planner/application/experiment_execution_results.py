"""Audit experiment result envelopes and release domain-gate candidates."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.experiment_execution_contracts import (
    ExperimentExecutionContractError,
    validate_experiment_execution_request,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_EXECUTION_RESULT_SCHEMA = "experiment_execution_result.v1"
EXPERIMENT_EXECUTION_RESULT_AUDIT_SCHEMA = "experiment_execution_result_audit.v1"
EXPERIMENT_EXECUTION_STATUSES = {"success", "failure", "inconclusive", "aborted"}
RESULT_SEMANTICS = {
    "result_is_an_execution_envelope": True,
    "raw_artifacts_are_content_addressed": True,
    "result_grants_no_domain_validation": True,
    "result_does_not_create_claim_or_canonical_reaction_proof": True,
    "domain_gate_is_required_after_release": True,
}
AUDIT_SEMANTICS = {
    "audit_is_read_only": True,
    "release_means_candidate_only": True,
    "domain_gate_still_owns_validation": True,
    "audit_cannot_create_claim_proof_completion_or_acceptance": True,
}


def with_experiment_execution_result_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address a provider result without granting or auditing it."""

    return _with_digest(value)


def build_experiment_execution_result(
    request: Mapping[str, Any],
    *,
    result_id: str,
    executor_id: str,
    executor_version: str,
    status: str,
    artifact_refs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    domain_validation_candidate: Mapping[str, Any] | None = None,
    failure_reasons: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a bound provider envelope; scientific acceptance is still absent."""

    request_value = dict(request)
    validate_experiment_execution_request(request_value)
    result = {
        "schema_version": EXPERIMENT_EXECUTION_RESULT_SCHEMA,
        "result_id": str(result_id),
        "request_id": str(request_value["request_id"]),
        "request_sha256": str(request_value["content_sha256"]),
        "run_id": str(request_value["run_id"]),
        "route_id": str(request_value["route_id"]),
        "domain": str(request_value["domain"]),
        "plan_id": str(request_value["plan_id"]),
        "program_id": str(request_value["program_id"]),
        "executor": {
            "executor_id": str(executor_id),
            "executor_version": str(executor_version),
        },
        "status": str(status),
        "artifact_refs": sorted(
            (dict(value) for value in artifact_refs),
            key=lambda row: (str(row.get("role") or ""), str(row.get("sha256") or "")),
        ),
        "domain_validation_candidate": dict(domain_validation_candidate or {}),
        "failure_reasons": sorted({str(value) for value in failure_reasons if str(value)}),
        "semantics": dict(RESULT_SEMANTICS),
    }
    return _with_digest(result)


def audit_experiment_execution_result(
    request: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit an executor envelope against its immutable request."""

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    try:
        request_value = dict(request)
        validate_experiment_execution_request(request_value)
        checks["request_valid"] = True
    except (ExperimentExecutionContractError, TypeError, ValueError):
        request_value = dict(request) if isinstance(request, Mapping) else {}
        checks["request_valid"] = False
        reasons.append("experiment_result_request_invalid")
    result_value = dict(result) if isinstance(result, Mapping) else {}
    checks["result_digest_valid"] = _digest_valid(result_value)
    checks["result_shape_valid"] = _result_shape_valid(result_value)
    checks["request_binding_equal"] = _result_request_binding_equal(
        request_value, result_value
    )
    checks["executor_identity_present"] = _executor_valid(result_value.get("executor"))
    checks["artifact_refs_valid"] = _artifacts_valid(
        result_value.get("artifact_refs"),
        required=result_value.get("status") != "aborted",
    )
    checks["status_payload_consistent"] = _status_payload_consistent(
        request_value, result_value
    )
    reasons.extend(key for key, accepted in checks.items() if accepted is not True)
    releasable = not reasons and result_value.get("status") != "aborted"
    return _with_digest(
        {
            "schema_version": EXPERIMENT_EXECUTION_RESULT_AUDIT_SCHEMA,
            "request_id": str(request_value.get("request_id") or ""),
            "result_id": str(result_value.get("result_id") or ""),
            "status": str(result_value.get("status") or ""),
            "accepted_for_domain_gate": releasable,
            "released_validation_candidate_sha256": (
                str(dict(result_value.get("domain_validation_candidate") or {}).get("content_sha256") or "")
                if releasable
                else ""
            ),
            "checks": checks,
            "reasons": sorted(set(reasons)),
            "semantics": dict(AUDIT_SEMANTICS),
        }
    )


def release_experiment_validation_candidate(
    request: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Release a bound candidate; the caller must still run its domain gate."""

    audit = audit_experiment_execution_result(request, result)
    if audit["accepted_for_domain_gate"] is not True:
        raise ExperimentExecutionContractError(
            "experiment_result_not_releasable:" + ",".join(audit["reasons"])
        )
    return dict(result["domain_validation_candidate"])


def _result_shape_valid(row: Mapping[str, Any]) -> bool:
    fields = {
        "schema_version", "result_id", "request_id", "request_sha256", "run_id",
        "route_id", "domain", "plan_id", "program_id", "executor", "status",
        "artifact_refs", "domain_validation_candidate", "failure_reasons",
        "semantics", "content_sha256",
    }
    failures = row.get("failure_reasons")
    return (
        set(row) == fields
        and row.get("schema_version") == EXPERIMENT_EXECUTION_RESULT_SCHEMA
        and row.get("status") in EXPERIMENT_EXECUTION_STATUSES
        and row.get("semantics") == RESULT_SEMANTICS
        and isinstance(failures, list)
        and all(isinstance(value, str) and value for value in failures)
        and failures == sorted(set(failures))
        and bool(str(row.get("result_id") or ""))
    )


def _result_request_binding_equal(request: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    return bool(request) and all(
        result.get(result_key) == request.get(request_key)
        for result_key, request_key in (
            ("request_id", "request_id"), ("request_sha256", "content_sha256"),
            ("run_id", "run_id"), ("route_id", "route_id"),
            ("domain", "domain"), ("plan_id", "plan_id"),
            ("program_id", "program_id"),
        )
    )


def _status_payload_consistent(request: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    status = result.get("status")
    candidate = result.get("domain_validation_candidate")
    failures = result.get("failure_reasons")
    if status == "aborted":
        return candidate == {} and isinstance(failures, list) and bool(failures)
    if not isinstance(candidate, Mapping) or not candidate or not _digest_valid(candidate):
        return False
    output = dict(request.get("required_output_contract") or {})
    boundary = dict(request.get("exact_boundary") or {})
    if (
        candidate.get("schema_version") != output.get("schema_version")
        or candidate.get("program_id") != request.get("program_id")
        or candidate.get("input_state_ids") != _state_ids(boundary.get("input_states"))
        or candidate.get("output_state_ids") != _state_ids(boundary.get("output_states"))
    ):
        return False
    domain = request.get("domain")
    if domain in {"execution", "mechanism"}:
        return candidate.get("outcome_status") == status
    return domain == "biocatalytic" and (
        (status == "success") == (candidate.get("accepted") is True)
    )


def _executor_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"executor_id", "executor_version"}
        and all(isinstance(item, str) and item for item in value.values())
    )


def _artifacts_valid(value: Any, *, required: bool) -> bool:
    if not isinstance(value, list) or (required and not value):
        return False
    if not all(
        isinstance(row, dict)
        and set(row) == {"sha256", "media_type", "role"}
        and _sha256(row.get("sha256"))
        and all(isinstance(row.get(key), str) and row.get(key) for key in ("media_type", "role"))
        for row in value
    ):
        return False
    canonical = sorted(value, key=lambda row: (row["role"], row["sha256"]))
    identities = [(row["role"], row["sha256"]) for row in value]
    return value == canonical and len(identities) == len(set(identities))


def _state_ids(value: Any) -> list[str]:
    return [str(dict(row).get("state_id") or "") for row in value or []]


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
    "AUDIT_SEMANTICS",
    "EXPERIMENT_EXECUTION_RESULT_AUDIT_SCHEMA",
    "EXPERIMENT_EXECUTION_RESULT_SCHEMA",
    "EXPERIMENT_EXECUTION_STATUSES",
    "RESULT_SEMANTICS",
    "audit_experiment_execution_result",
    "build_experiment_execution_result",
    "release_experiment_validation_candidate",
    "with_experiment_execution_result_digest",
]
