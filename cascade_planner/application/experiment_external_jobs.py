"""Strict operational contracts for externally executed experiment jobs."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_OPERATOR_IDENTITY_SCHEMA = "experiment_operator_identity.v1"
EXPERIMENT_EXTERNAL_JOB_RECEIPT_SCHEMA = "experiment_external_job_receipt.v1"
EXPERIMENT_CANCELLATION_REQUEST_SCHEMA = "experiment_cancellation_request.v1"
EXTERNAL_JOB_STATUSES = {
    "submitted", "running", "completed", "failed", "cancelled"
}
EXTERNAL_JOB_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_OPERATOR_SEMANTICS = {
    "identity_is_an_audit_binding_only": True,
    "authentication_context_digest_contains_no_credential": True,
    "identity_grants_no_provider_or_scientific_authority": True,
}
_RECEIPT_SEMANTICS = {
    "receipt_is_an_operational_observation_only": True,
    "provider_sequence_does_not_define_scientific_revision": True,
    "cancelled_requires_a_bound_cancellation_request": True,
    "receipt_grants_no_validation_claim_graph_or_completion": True,
}
_CANCELLATION_SEMANTICS = {
    "request_does_not_acknowledge_or_complete_cancellation": True,
    "external_cancelled_receipt_is_required_to_settle_cancelled": True,
    "request_grants_no_validation_claim_graph_or_completion": True,
}


class ExperimentExternalJobContractError(ValueError):
    """Raised when an external job audit contract is malformed or inconsistent."""


def build_experiment_operator_identity(
    *, principal_id: str, principal_type: str, authentication_context_sha256: str
) -> dict[str, Any]:
    value = _with_digest({
        "schema_version": EXPERIMENT_OPERATOR_IDENTITY_SCHEMA,
        "principal_id": str(principal_id).strip(),
        "principal_type": str(principal_type).strip(),
        "authentication_context_sha256": str(authentication_context_sha256).strip(),
        "semantics": dict(_OPERATOR_SEMANTICS),
    })
    validate_experiment_operator_identity(value)
    return value


def validate_experiment_operator_identity(value: Mapping[str, Any]) -> None:
    row = dict(value) if isinstance(value, Mapping) else {}
    if (
        set(row) != {
            "schema_version", "principal_id", "principal_type",
            "authentication_context_sha256", "semantics", "content_sha256",
        }
        or row.get("schema_version") != EXPERIMENT_OPERATOR_IDENTITY_SCHEMA
        or not _nonempty_strings(row, ("principal_id",))
        or row.get("principal_type") not in {"human", "service"}
        or not _sha256(row.get("authentication_context_sha256"))
        or row.get("semantics") != _OPERATOR_SEMANTICS
        or not _digest_valid(row)
    ):
        raise ExperimentExternalJobContractError("experiment_operator_identity_invalid")


def build_experiment_external_job_receipt(
    *,
    dispatch_id: str,
    task_id: str,
    request_id: str,
    request_sha256: str,
    provider_id: str,
    provider_version: str,
    external_job_id: str,
    provider_sequence: int,
    status: str,
    recorded_by: Mapping[str, Any],
    predecessor_receipt_sha256: str = "",
    cancellation_request_sha256: str = "",
    status_detail: str = "",
) -> dict[str, Any]:
    value = _with_digest({
        "schema_version": EXPERIMENT_EXTERNAL_JOB_RECEIPT_SCHEMA,
        "dispatch_id": str(dispatch_id).strip(),
        "task_id": str(task_id).strip(),
        "request_id": str(request_id).strip(),
        "request_sha256": str(request_sha256).strip(),
        "provider_id": str(provider_id).strip(),
        "provider_version": str(provider_version).strip(),
        "external_job_id": str(external_job_id).strip(),
        "provider_sequence": provider_sequence,
        "status": str(status).strip(),
        "predecessor_receipt_sha256": str(predecessor_receipt_sha256).strip(),
        "cancellation_request_sha256": str(cancellation_request_sha256).strip(),
        "recorded_by": dict(recorded_by),
        "status_detail": str(status_detail),
        "semantics": dict(_RECEIPT_SEMANTICS),
    })
    validate_experiment_external_job_receipt(value)
    return value


def validate_experiment_external_job_receipt(value: Mapping[str, Any]) -> None:
    row = dict(value) if isinstance(value, Mapping) else {}
    expected = {
        "schema_version", "dispatch_id", "task_id", "request_id",
        "request_sha256", "provider_id", "provider_version", "external_job_id",
        "provider_sequence", "status", "predecessor_receipt_sha256",
        "cancellation_request_sha256", "recorded_by", "status_detail",
        "semantics", "content_sha256",
    }
    sequence = row.get("provider_sequence")
    if (
        set(row) != expected
        or row.get("schema_version") != EXPERIMENT_EXTERNAL_JOB_RECEIPT_SCHEMA
        or not _nonempty_strings(row, (
            "dispatch_id", "task_id", "request_id", "provider_id",
            "provider_version", "external_job_id",
        ))
        or not _sha256(row.get("request_sha256"))
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or row.get("status") not in EXTERNAL_JOB_STATUSES
        or not _optional_sha256(row.get("predecessor_receipt_sha256"))
        or not _optional_sha256(row.get("cancellation_request_sha256"))
        or not isinstance(row.get("status_detail"), str)
        or row.get("semantics") != _RECEIPT_SEMANTICS
        or not _digest_valid(row)
    ):
        raise ExperimentExternalJobContractError(
            "experiment_external_job_receipt_invalid"
        )
    validate_experiment_operator_identity(dict(row.get("recorded_by") or {}))


def build_experiment_cancellation_request(
    *,
    dispatch_id: str,
    task_id: str,
    request_id: str,
    request_sha256: str,
    provider_id: str,
    provider_version: str,
    external_job_id: str,
    current_external_job_receipt_sha256: str,
    requested_by: Mapping[str, Any],
    reason_code: str,
    reason_detail: str = "",
) -> dict[str, Any]:
    value = _with_digest({
        "schema_version": EXPERIMENT_CANCELLATION_REQUEST_SCHEMA,
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
        "requested_by": dict(requested_by),
        "reason_code": str(reason_code).strip(),
        "reason_detail": str(reason_detail),
        "semantics": dict(_CANCELLATION_SEMANTICS),
    })
    validate_experiment_cancellation_request(value)
    return value


def validate_experiment_cancellation_request(value: Mapping[str, Any]) -> None:
    row = dict(value) if isinstance(value, Mapping) else {}
    if (
        set(row) != {
            "schema_version", "dispatch_id", "task_id", "request_id",
            "request_sha256", "provider_id", "provider_version",
            "external_job_id", "current_external_job_receipt_sha256",
            "requested_by", "reason_code", "reason_detail", "semantics",
            "content_sha256",
        }
        or row.get("schema_version") != EXPERIMENT_CANCELLATION_REQUEST_SCHEMA
        or not _nonempty_strings(row, (
            "dispatch_id", "task_id", "request_id", "provider_id",
            "provider_version", "external_job_id", "reason_code",
        ))
        or not _sha256(row.get("request_sha256"))
        or not _sha256(row.get("current_external_job_receipt_sha256"))
        or not isinstance(row.get("reason_detail"), str)
        or row.get("semantics") != _CANCELLATION_SEMANTICS
        or not _digest_valid(row)
    ):
        raise ExperimentExternalJobContractError(
            "experiment_cancellation_request_invalid"
        )
    validate_experiment_operator_identity(dict(row.get("requested_by") or {}))


def validate_experiment_external_job_transition(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    cancellation_request: Mapping[str, Any] | None = None,
) -> None:
    current_value = dict(current)
    validate_experiment_external_job_receipt(current_value)
    previous_value = dict(previous or {})
    cancellation = dict(cancellation_request or {})
    if previous_value:
        validate_experiment_external_job_receipt(previous_value)
        bound_fields = (
            "dispatch_id", "task_id", "request_id", "request_sha256",
            "provider_id", "provider_version", "external_job_id",
        )
        if (
            previous_value.get("status") in EXTERNAL_JOB_TERMINAL_STATUSES
            or any(previous_value.get(key) != current_value.get(key) for key in bound_fields)
            or current_value["provider_sequence"] <= previous_value["provider_sequence"]
            or current_value.get("predecessor_receipt_sha256")
            != previous_value.get("content_sha256")
        ):
            raise ExperimentExternalJobContractError(
                "experiment_external_job_transition_invalid"
            )
    elif (
        current_value.get("status") not in {"submitted", "running"}
        or current_value.get("predecessor_receipt_sha256")
    ):
        raise ExperimentExternalJobContractError(
            "experiment_external_job_initial_receipt_invalid"
        )
    if cancellation:
        validate_experiment_cancellation_request(cancellation)
        if current_value.get("cancellation_request_sha256") != cancellation.get(
            "content_sha256"
        ):
            raise ExperimentExternalJobContractError(
                "experiment_external_job_cancellation_binding_invalid"
            )
    elif current_value.get("cancellation_request_sha256"):
        raise ExperimentExternalJobContractError(
            "experiment_external_job_cancellation_binding_invalid"
        )
    if current_value.get("status") == "cancelled" and not cancellation:
        raise ExperimentExternalJobContractError(
            "experiment_external_job_cancelled_without_request"
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
    "EXPERIMENT_CANCELLATION_REQUEST_SCHEMA",
    "EXPERIMENT_EXTERNAL_JOB_RECEIPT_SCHEMA",
    "EXPERIMENT_OPERATOR_IDENTITY_SCHEMA",
    "EXTERNAL_JOB_STATUSES",
    "EXTERNAL_JOB_TERMINAL_STATUSES",
    "ExperimentExternalJobContractError",
    "build_experiment_cancellation_request",
    "build_experiment_external_job_receipt",
    "build_experiment_operator_identity",
    "validate_experiment_cancellation_request",
    "validate_experiment_external_job_receipt",
    "validate_experiment_external_job_transition",
    "validate_experiment_operator_identity",
]
