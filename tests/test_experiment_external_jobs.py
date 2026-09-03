from __future__ import annotations

import pytest

from cascade_planner.application.experiment_external_jobs import (
    ExperimentExternalJobContractError,
    build_experiment_cancellation_request,
    build_experiment_external_job_receipt,
    build_experiment_operator_identity,
    validate_experiment_external_job_receipt,
    validate_experiment_external_job_transition,
    validate_experiment_operator_identity,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _operator() -> dict:
    return build_experiment_operator_identity(
        principal_id="operator:fixture",
        principal_type="human",
        authentication_context_sha256="a" * 64,
    )


def _receipt(
    *, sequence: int, status: str, predecessor: str = "", cancellation: str = ""
) -> dict:
    return build_experiment_external_job_receipt(
        dispatch_id="experiment-dispatch:" + "1" * 32,
        task_id="experiment-dispatch-task:" + "1" * 32,
        request_id="experiment-request:fixture",
        request_sha256="b" * 64,
        provider_id="fixture.executor",
        provider_version="1.0.0",
        external_job_id="external-job:123",
        provider_sequence=sequence,
        status=status,
        predecessor_receipt_sha256=predecessor,
        cancellation_request_sha256=cancellation,
        recorded_by=_operator(),
    )


def test_operator_identity_is_strict_digest_bound_and_non_authoritative() -> None:
    operator = _operator()
    assert operator["semantics"][
        "identity_grants_no_provider_or_scientific_authority"
    ] is True
    tampered = dict(operator)
    tampered["principal_type"] = "administrator"
    with pytest.raises(
        ExperimentExternalJobContractError, match="operator_identity_invalid"
    ):
        validate_experiment_operator_identity(tampered)
    extra = dict(operator)
    extra["credential"] = "must-not-be-accepted"
    with pytest.raises(ExperimentExternalJobContractError):
        validate_experiment_operator_identity(extra)


def test_external_job_receipts_enforce_sequence_terminal_and_cancellation_rules() -> None:
    submitted = _receipt(sequence=1, status="submitted")
    validate_experiment_external_job_transition(None, submitted)
    running = _receipt(
        sequence=3, status="running", predecessor=submitted["content_sha256"]
    )
    validate_experiment_external_job_transition(submitted, running)
    cancellation = build_experiment_cancellation_request(
        dispatch_id=running["dispatch_id"], task_id=running["task_id"],
        request_id=running["request_id"], request_sha256=running["request_sha256"],
        provider_id=running["provider_id"], provider_version=running["provider_version"],
        external_job_id=running["external_job_id"],
        current_external_job_receipt_sha256=running["content_sha256"],
        requested_by=_operator(), reason_code="operator_requested",
    )
    completed_after_cancel_race = _receipt(
        sequence=4, status="completed", predecessor=running["content_sha256"],
        cancellation=cancellation["content_sha256"],
    )
    validate_experiment_external_job_transition(
        running, completed_after_cancel_race, cancellation_request=cancellation
    )
    with pytest.raises(
        ExperimentExternalJobContractError, match="transition_invalid"
    ):
        validate_experiment_external_job_transition(
            completed_after_cancel_race,
            _receipt(
                sequence=5, status="running",
                predecessor=completed_after_cancel_race["content_sha256"],
                cancellation=cancellation["content_sha256"],
            ),
            cancellation_request=cancellation,
        )
    with pytest.raises(
        ExperimentExternalJobContractError, match="cancelled_without_request"
    ):
        validate_experiment_external_job_transition(
            running,
            _receipt(
                sequence=4, status="cancelled",
                predecessor=running["content_sha256"],
            ),
        )


def test_external_job_receipt_rejects_boolean_sequence_type_confusion() -> None:
    receipt = _receipt(sequence=1, status="submitted")
    tampered = dict(receipt)
    tampered["provider_sequence"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = strict_canonical_json_sha256(tampered)
    with pytest.raises(
        ExperimentExternalJobContractError, match="external_job_receipt_invalid"
    ):
        validate_experiment_external_job_receipt(tampered)
