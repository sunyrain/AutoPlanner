"""Provider-neutral contracts for external experiment job transport calls."""

from __future__ import annotations

import math
from typing import Any, Mapping

from cascade_planner.application.experiment_execution_contracts import (
    validate_experiment_execution_request,
)
from cascade_planner.application.experiment_external_jobs import (
    EXTERNAL_JOB_STATUSES,
    validate_experiment_operator_identity,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA = "experiment_job_operation_request.v1"
EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA = "experiment_job_transport_result.v1"
EXPERIMENT_JOB_OPERATIONS = {"submit", "poll", "cancel"}
EXPERIMENT_TRANSPORT_OUTCOMES = {
    "success", "timeout", "http_error", "transport_error",
    "invalid_response", "authentication_unavailable",
}
_REQUEST_SEMANTICS = {
    "endpoint_and_credentials_are_host_configured": True,
    "operation_reuses_the_reserved_experiment_task": True,
    "request_grants_no_validation_claim_graph_or_completion": True,
}
_RESULT_SEMANTICS = {
    "transport_success_is_not_experiment_or_scientific_success": True,
    "response_body_is_represented_by_digest_only": True,
    "credentials_are_never_returned": True,
    "result_grants_no_validation_claim_graph_or_completion": True,
}


class ExperimentJobOperationContractError(ValueError):
    """Raised when a transport request or result violates its strict contract."""


def build_experiment_job_operation_request(
    *,
    operation: str,
    attempt_number: int,
    run_id: str,
    dispatch_id: str,
    task_id: str,
    request_id: str,
    request_sha256: str,
    provider_id: str,
    provider_version: str,
    timeout_s: float,
    external_job_id: str = "",
    current_external_job_receipt_sha256: str = "",
    cancellation_request_sha256: str = "",
    execution_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    operation_value = str(operation).strip()
    identity = {
        "operation": operation_value,
        "attempt_number": attempt_number,
        "run_id": str(run_id).strip(),
        "dispatch_id": str(dispatch_id).strip(),
        "task_id": str(task_id).strip(),
        "request_id": str(request_id).strip(),
        "request_sha256": str(request_sha256).strip(),
        "provider_id": str(provider_id).strip(),
        "provider_version": str(provider_version).strip(),
        "external_job_id": str(external_job_id).strip(),
        "current_external_job_receipt_sha256": str(
            current_external_job_receipt_sha256
        ).strip(),
        "cancellation_request_sha256": str(cancellation_request_sha256).strip(),
    }
    value = _with_digest({
        "schema_version": EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA,
        "operation_id": "experiment-job-operation:"
        + strict_canonical_json_sha256(identity)[:32],
        **identity,
        "timeout_s": timeout_s,
        "execution_request": dict(execution_request or {}),
        "semantics": dict(_REQUEST_SEMANTICS),
    })
    validate_experiment_job_operation_request(value)
    return value


def validate_experiment_job_operation_request(value: Mapping[str, Any]) -> None:
    row = dict(value) if isinstance(value, Mapping) else {}
    expected = {
        "schema_version", "operation_id", "operation", "attempt_number",
        "run_id", "dispatch_id", "task_id", "request_id", "request_sha256",
        "provider_id", "provider_version", "external_job_id",
        "current_external_job_receipt_sha256", "cancellation_request_sha256",
        "timeout_s", "execution_request", "semantics", "content_sha256",
    }
    attempt = row.get("attempt_number")
    timeout = row.get("timeout_s")
    operation = row.get("operation")
    identity = {
        key: row.get(key)
        for key in (
            "operation", "attempt_number", "run_id", "dispatch_id", "task_id",
            "request_id", "request_sha256", "provider_id", "provider_version",
            "external_job_id", "current_external_job_receipt_sha256",
            "cancellation_request_sha256",
        )
    }
    expected_operation_id = "experiment-job-operation:" + (
        strict_canonical_json_sha256(identity)[:32]
    )
    if (
        set(row) != expected
        or row.get("schema_version") != EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA
        or operation not in EXPERIMENT_JOB_OPERATIONS
        or not _nonempty_strings(row, (
            "operation_id", "run_id", "dispatch_id", "task_id", "request_id",
            "provider_id", "provider_version",
        ))
        or row.get("operation_id") != expected_operation_id
        or not _sha256(row.get("request_sha256"))
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt <= 0
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 3600
        or not _optional_sha256(row.get("current_external_job_receipt_sha256"))
        or not _optional_sha256(row.get("cancellation_request_sha256"))
        or row.get("semantics") != _REQUEST_SEMANTICS
        or not _digest_valid(row)
    ):
        raise ExperimentJobOperationContractError(
            "experiment_job_operation_request_invalid"
        )
    execution_request = row.get("execution_request")
    if not isinstance(execution_request, Mapping):
        raise ExperimentJobOperationContractError(
            "experiment_job_operation_execution_request_invalid"
        )
    if operation == "submit":
        if any(row.get(key) for key in (
            "external_job_id", "current_external_job_receipt_sha256",
            "cancellation_request_sha256",
        )):
            raise ExperimentJobOperationContractError(
                "experiment_job_submit_binding_invalid"
            )
        try:
            validate_experiment_execution_request(dict(execution_request))
        except (TypeError, ValueError) as exc:
            raise ExperimentJobOperationContractError(
                "experiment_job_submit_execution_request_invalid"
            ) from exc
        if (
            execution_request.get("run_id") != row.get("run_id")
            or execution_request.get("request_id") != row.get("request_id")
            or execution_request.get("content_sha256") != row.get("request_sha256")
        ):
            raise ExperimentJobOperationContractError(
                "experiment_job_submit_execution_request_binding_invalid"
            )
    elif execution_request != {} or not _nonempty_strings(row, ("external_job_id",)):
        raise ExperimentJobOperationContractError(
            "experiment_job_followup_binding_invalid"
        )
    elif not _sha256(row.get("current_external_job_receipt_sha256")):
        raise ExperimentJobOperationContractError(
            "experiment_job_followup_receipt_invalid"
        )
    if operation == "cancel" and not _sha256(row.get("cancellation_request_sha256")):
        raise ExperimentJobOperationContractError(
            "experiment_job_cancel_request_binding_invalid"
        )


def build_experiment_job_transport_result(
    request: Mapping[str, Any],
    *,
    outcome: str,
    endpoint_config_sha256: str,
    authentication_context_sha256: str,
    recorded_by: Mapping[str, Any],
    external_job_id: str = "",
    provider_sequence: int = 0,
    status: str = "",
    status_detail: str = "",
    http_status: int = 0,
    response_body_sha256: str = "",
    detail_code: str = "",
) -> dict[str, Any]:
    request_value = dict(request)
    validate_experiment_job_operation_request(request_value)
    value = _with_digest({
        "schema_version": EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA,
        "operation_id": request_value["operation_id"],
        "operation": request_value["operation"],
        "attempt_number": request_value["attempt_number"],
        "run_id": request_value["run_id"],
        "dispatch_id": request_value["dispatch_id"],
        "task_id": request_value["task_id"],
        "request_id": request_value["request_id"],
        "request_sha256": request_value["request_sha256"],
        "provider_id": request_value["provider_id"],
        "provider_version": request_value["provider_version"],
        "outcome": str(outcome).strip(),
        "external_job_id": str(external_job_id).strip(),
        "provider_sequence": provider_sequence,
        "status": str(status).strip(),
        "status_detail": str(status_detail),
        "http_status": http_status,
        "response_body_sha256": str(response_body_sha256).strip(),
        "endpoint_config_sha256": str(endpoint_config_sha256).strip(),
        "authentication_context_sha256": str(
            authentication_context_sha256
        ).strip(),
        "recorded_by": dict(recorded_by),
        "detail_code": str(detail_code).strip(),
        "semantics": dict(_RESULT_SEMANTICS),
    })
    validate_experiment_job_transport_result(value, request=request_value)
    return value


def validate_experiment_job_transport_result(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> None:
    row = dict(value) if isinstance(value, Mapping) else {}
    request_value = dict(request)
    validate_experiment_job_operation_request(request_value)
    expected = {
        "schema_version", "operation_id", "operation", "attempt_number", "run_id",
        "dispatch_id", "task_id", "request_id", "request_sha256", "provider_id",
        "provider_version", "outcome", "external_job_id", "provider_sequence",
        "status", "status_detail", "http_status", "response_body_sha256",
        "endpoint_config_sha256", "authentication_context_sha256", "recorded_by",
        "detail_code", "semantics", "content_sha256",
    }
    sequence = row.get("provider_sequence")
    http_status = row.get("http_status")
    bound = (
        "operation_id", "operation", "attempt_number", "run_id", "dispatch_id",
        "task_id", "request_id", "request_sha256", "provider_id", "provider_version",
    )
    if (
        set(row) != expected
        or row.get("schema_version") != EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA
        or any(row.get(key) != request_value.get(key) for key in bound)
        or row.get("outcome") not in EXPERIMENT_TRANSPORT_OUTCOMES
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or isinstance(http_status, bool)
        or not isinstance(http_status, int)
        or not (http_status == 0 or 100 <= http_status <= 599)
        or not isinstance(row.get("status_detail"), str)
        or len(row.get("status_detail") or "") > 1000
        or not _optional_sha256(row.get("response_body_sha256"))
        or not _sha256(row.get("endpoint_config_sha256"))
        or not _sha256(row.get("authentication_context_sha256"))
        or row.get("semantics") != _RESULT_SEMANTICS
        or not _digest_valid(row)
    ):
        raise ExperimentJobOperationContractError(
            "experiment_job_transport_result_invalid"
        )
    validate_experiment_operator_identity(dict(row.get("recorded_by") or {}))
    if (
        dict(row["recorded_by"]).get("authentication_context_sha256")
        != row.get("authentication_context_sha256")
    ):
        raise ExperimentJobOperationContractError(
            "experiment_job_transport_operator_binding_invalid"
        )
    if row["outcome"] == "success":
        if (
            not _nonempty_strings(row, ("external_job_id", "status"))
            or sequence <= 0
            or row["status"] not in EXTERNAL_JOB_STATUSES
            or not 200 <= http_status <= 299
            or not _sha256(row.get("response_body_sha256"))
            or row.get("detail_code")
        ):
            raise ExperimentJobOperationContractError(
                "experiment_job_transport_success_invalid"
            )
        if row["operation"] == "submit" and row["status"] not in {
            "submitted", "running"
        }:
            raise ExperimentJobOperationContractError(
                "experiment_job_transport_submit_status_invalid"
            )
    elif (
        row.get("status")
        or sequence
        or not _nonempty_strings(row, ("detail_code",))
        or len(row["detail_code"]) > 128
    ):
        raise ExperimentJobOperationContractError(
            "experiment_job_transport_failure_invalid"
        )


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    observed = row.pop("content_sha256", "")
    try:
        return (
            isinstance(observed, str)
            and bool(observed)
            and observed == strict_canonical_json_sha256(row)
        )
    except (TypeError, ValueError):
        return False


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _optional_sha256(value: Any) -> bool:
    return value == "" or _sha256(value)


def _nonempty_strings(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(
        isinstance(value.get(key), str)
        and bool(value[key])
        and value[key] == value[key].strip()
        for key in keys
    )


__all__ = [
    "EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA",
    "EXPERIMENT_JOB_OPERATIONS",
    "EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA",
    "EXPERIMENT_TRANSPORT_OUTCOMES",
    "ExperimentJobOperationContractError",
    "build_experiment_job_operation_request",
    "build_experiment_job_transport_result",
    "validate_experiment_job_operation_request",
    "validate_experiment_job_transport_result",
]
