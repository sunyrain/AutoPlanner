"""Durable external-job observations on the existing experiment task ledger."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.experiment_external_jobs import (
    EXTERNAL_JOB_TERMINAL_STATUSES,
    ExperimentExternalJobContractError,
    validate_experiment_cancellation_request,
    validate_experiment_external_job_receipt,
    validate_experiment_external_job_transition,
)
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.orchestration.experiment_dispatch_support import (
    EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
    ExperimentDispatchError,
    artifact_ref,
    current_reserved_descriptor,
    index_artifact,
    read_object,
    receipt_semantics,
    reservation_metadata,
    task_id_from_dispatch,
    with_digest,
)
from cascade_planner.orchestration.experiment_execution_runtime import (
    locate_current_route_experiment_request,
)
from cascade_planner.providers.registry import ProviderRegistry


EXPERIMENT_JOB_RECEIPT_CHECKPOINT = "experiment_external_job_receipt"
EXPERIMENT_CANCELLATION_CHECKPOINT = "experiment_cancellation_request"


def record_current_route_experiment_job_receipt(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    job_receipt: Mapping[str, Any],
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_job_receipt: bool = False,
) -> dict[str, Any]:
    """Record one provider-sequenced external job observation."""

    if enable_experiment_job_receipt is not True:
        raise ExperimentDispatchError("experiment_job_receipt_explicit_enable_required")
    lifecycle, metadata, descriptor = current_experiment_dispatch_context(
        kernel, graph, acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, dispatch_id=dispatch_id, registry=registry,
        mechanism_proposals=mechanism_proposals, validations=validations,
    )
    value = dict(job_receipt)
    try:
        validate_experiment_external_job_receipt(value)
    except ExperimentExternalJobContractError as exc:
        raise ExperimentDispatchError(str(exc)) from exc
    _assert_contract_binding(value, metadata, lifecycle["task_id"], descriptor)
    projection = external_job_projection(kernel, lifecycle)
    previous = projection["latest_receipt"]
    cancellation = projection["cancellation_request"]
    try:
        validate_experiment_external_job_transition(
            previous or None, value, cancellation_request=cancellation or None
        )
    except ExperimentExternalJobContractError as exc:
        if previous == value:
            return previous
        raise ExperimentDispatchError(str(exc)) from exc
    if lifecycle["status"] != "in_flight":
        if previous == value:
            return previous
        raise ExperimentDispatchError("experiment_dispatch_already_settled")
    ref = kernel.artifacts.put_json(
        value,
        logical_name=f"{dispatch_id}.external-job-{value['provider_sequence']}.json",
        producer="autoplanner.experiment_external_job",
    )
    kernel.record_task_checkpoint(
        task_id=lifecycle["task_id"],
        checkpoint_kind=EXPERIMENT_JOB_RECEIPT_CHECKPOINT,
        artifact_ref=ref,
        predecessor_checkpoint_sha256=projection["latest_checkpoint_sha256"],
        operational_status=str(value["status"]),
        idempotency_key=(
            f"experiment-job-receipt:{dispatch_id}:{value['provider_sequence']}"
        ),
        metadata={"dispatch_id": dispatch_id, "external_job_id": value["external_job_id"]},
    )
    index_artifact(
        kernel, dispatch_id, f"external-job-{value['provider_sequence']}", ref,
        "external_experiment_job_observation_only",
    )
    kernel.artifacts.write_pointer(
        str(metadata["pointer_name"]) + ".external-job", ref,
        metadata={
            "run_id": kernel.spec.run_id, "dispatch_id": dispatch_id,
            "status": value["status"],
        },
    )
    if value["status"] == "cancelled":
        _settle_cancelled_job(
            kernel, metadata=metadata, descriptor=descriptor,
            job_receipt=value, job_receipt_ref=ref,
            cancellation_request=cancellation,
            cancellation_request_ref=projection["cancellation_request_ref"],
        )
    return read_object(kernel, ref, "experiment_external_job_receipt_artifact_invalid")


def request_current_route_experiment_cancellation(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    cancellation_request: Mapping[str, Any],
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_cancellation: bool = False,
) -> dict[str, Any]:
    """Record a cancellation request without treating it as acknowledgement."""

    if enable_experiment_cancellation is not True:
        raise ExperimentDispatchError("experiment_cancellation_explicit_enable_required")
    lifecycle, metadata, descriptor = current_experiment_dispatch_context(
        kernel, graph, acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, dispatch_id=dispatch_id, registry=registry,
        mechanism_proposals=mechanism_proposals, validations=validations,
    )
    value = dict(cancellation_request)
    try:
        validate_experiment_cancellation_request(value)
    except ExperimentExternalJobContractError as exc:
        raise ExperimentDispatchError(str(exc)) from exc
    _assert_contract_binding(value, metadata, lifecycle["task_id"], descriptor)
    projection = external_job_projection(kernel, lifecycle)
    existing = projection["cancellation_request"]
    if existing:
        if existing == value:
            return existing
        raise ExperimentDispatchError("experiment_cancellation_request_conflict")
    latest = projection["latest_receipt"]
    if not latest:
        raise ExperimentDispatchError("experiment_external_job_receipt_required")
    if latest["status"] in EXTERNAL_JOB_TERMINAL_STATUSES:
        raise ExperimentDispatchError("experiment_external_job_already_terminal")
    if (
        value["external_job_id"] != latest["external_job_id"]
        or value["current_external_job_receipt_sha256"]
        != latest["content_sha256"]
    ):
        raise ExperimentDispatchError("experiment_cancellation_current_receipt_mismatch")
    if lifecycle["status"] != "in_flight":
        raise ExperimentDispatchError("experiment_dispatch_already_settled")
    ref = kernel.artifacts.put_json(
        value, logical_name=f"{dispatch_id}.cancellation-request.json",
        producer="autoplanner.experiment_external_job",
    )
    kernel.record_task_checkpoint(
        task_id=lifecycle["task_id"],
        checkpoint_kind=EXPERIMENT_CANCELLATION_CHECKPOINT,
        artifact_ref=ref,
        predecessor_checkpoint_sha256=projection["latest_checkpoint_sha256"],
        operational_status="cancellation_requested",
        idempotency_key=f"experiment-cancellation-request:{dispatch_id}",
        metadata={"dispatch_id": dispatch_id, "external_job_id": value["external_job_id"]},
    )
    index_artifact(
        kernel, dispatch_id, "cancellation-request", ref,
        "external_experiment_cancellation_request_only",
    )
    kernel.artifacts.write_pointer(
        str(metadata["pointer_name"]) + ".cancellation-request", ref,
        metadata={
            "run_id": kernel.spec.run_id, "dispatch_id": dispatch_id,
            "status": "cancellation_requested",
        },
    )
    return read_object(kernel, ref, "experiment_cancellation_request_artifact_invalid")


def external_job_projection(
    kernel: RunKernel, lifecycle: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay application contracts from CAS-bound task checkpoints."""

    receipts: list[dict[str, Any]] = []
    receipt_refs: list[dict[str, Any]] = []
    cancellation: dict[str, Any] = {}
    cancellation_ref: dict[str, Any] = {}
    latest_checkpoint_sha256 = ""
    metadata = reservation_metadata(lifecycle)
    descriptor = {
        "provider_id": metadata.get("provider_id"),
        "version": metadata.get("provider_version"),
    }
    for event in lifecycle.get("checkpoints") or []:
        payload = dict(dict(event).get("payload") or {})
        kind = str(payload.get("checkpoint_kind") or "")
        ref = artifact_ref(payload.get("artifact_ref"))
        if kind not in {
            EXPERIMENT_JOB_RECEIPT_CHECKPOINT,
            EXPERIMENT_CANCELLATION_CHECKPOINT,
        }:
            latest_checkpoint_sha256 = ref.sha256
            continue
        value = read_object(kernel, ref, "experiment_task_checkpoint_artifact_invalid")
        _assert_contract_binding(value, metadata, lifecycle["task_id"], descriptor)
        if kind == EXPERIMENT_JOB_RECEIPT_CHECKPOINT:
            try:
                validate_experiment_external_job_transition(
                    receipts[-1] if receipts else None,
                    value,
                    cancellation_request=cancellation or None,
                )
            except ExperimentExternalJobContractError as exc:
                raise ExperimentDispatchError(str(exc)) from exc
            receipts.append(value)
            receipt_refs.append(ref.to_dict())
        elif kind == EXPERIMENT_CANCELLATION_CHECKPOINT:
            try:
                validate_experiment_cancellation_request(value)
            except ExperimentExternalJobContractError as exc:
                raise ExperimentDispatchError(str(exc)) from exc
            if (
                cancellation or not receipts
                or receipts[-1]["status"] in EXTERNAL_JOB_TERMINAL_STATUSES
                or value["external_job_id"] != receipts[-1]["external_job_id"]
                or value["current_external_job_receipt_sha256"]
                != receipts[-1]["content_sha256"]
            ):
                raise ExperimentDispatchError("experiment_cancellation_checkpoint_invalid")
            cancellation = value
            cancellation_ref = ref.to_dict()
        latest_checkpoint_sha256 = ref.sha256
    return {
        "receipts": receipts,
        "receipt_refs": receipt_refs,
        "latest_receipt": receipts[-1] if receipts else {},
        "cancellation_request": cancellation,
        "cancellation_request_ref": cancellation_ref,
        "latest_checkpoint_sha256": latest_checkpoint_sha256,
    }


def assert_external_job_result_settleable(
    kernel: RunKernel, lifecycle: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    projection = external_job_projection(kernel, lifecycle)
    latest = projection["latest_receipt"]
    if not latest:
        return
    if latest["status"] not in {"completed", "failed"}:
        raise ExperimentDispatchError("experiment_external_job_not_ready_for_result")
    if result.get("status") == "aborted":
        raise ExperimentDispatchError(
            "experiment_external_job_aborted_requires_cancelled_acknowledgement"
        )


def current_experiment_dispatch_context(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]],
    validations: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_id = task_id_from_dispatch(dispatch_id)
    lifecycle = kernel.task_lifecycle(task_id)
    if lifecycle["status"] == "absent":
        raise ExperimentDispatchError("experiment_dispatch_not_found")
    metadata = reservation_metadata(lifecycle)
    descriptor = current_reserved_descriptor(registry, metadata)
    request = read_object(
        kernel, artifact_ref(metadata.get("request_ref")),
        "experiment_dispatch_request_artifact_invalid",
    )
    _, current = locate_current_route_experiment_request(
        graph, acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, request_id=str(metadata["request_id"]),
        mechanism_proposals=mechanism_proposals, validations=validations,
    )
    if current != request or request.get("content_sha256") != metadata.get(
        "request_sha256"
    ):
        raise ExperimentDispatchError("experiment_dispatch_request_not_current")
    return lifecycle, metadata, descriptor


def _assert_contract_binding(
    value: Mapping[str, Any], metadata: Mapping[str, Any],
    task_id: str, descriptor: Mapping[str, Any]
) -> None:
    expected = {
        "dispatch_id": metadata.get("dispatch_id"),
        "task_id": task_id,
        "request_id": metadata.get("request_id"),
        "request_sha256": metadata.get("request_sha256"),
        "provider_id": descriptor.get("provider_id"),
        "provider_version": descriptor.get("version"),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ExperimentDispatchError("experiment_external_job_dispatch_binding_invalid")


def _settle_cancelled_job(
    kernel: RunKernel,
    *,
    metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    job_receipt: Mapping[str, Any],
    job_receipt_ref: Any,
    cancellation_request: Mapping[str, Any],
    cancellation_request_ref: Mapping[str, Any],
) -> None:
    if not cancellation_request or not cancellation_request_ref:
        raise ExperimentDispatchError("experiment_external_job_cancelled_without_request")
    dispatch_id = str(metadata["dispatch_id"])
    receipt = with_digest({
        "schema_version": EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
        "dispatch_id": dispatch_id,
        "task_id": str(job_receipt["task_id"]),
        "status": "settled",
        "request_id": str(metadata["request_id"]),
        "request_sha256": str(metadata["request_sha256"]),
        "provider_id": descriptor["provider_id"],
        "provider_version": descriptor["version"],
        "result_status": "cancelled",
        "external_job_receipt_ref": job_receipt_ref.to_dict(),
        "cancellation_request_ref": dict(cancellation_request_ref),
        "domain_validation_candidate": {},
        "next_boundary": "retain_cancelled_operational_receipt_without_domain_validation",
        "semantics": receipt_semantics(),
    })
    ref = kernel.artifacts.put_json(
        receipt, logical_name=f"{dispatch_id}.cancelled-settlement.json",
        producer="autoplanner.experiment_external_job",
    )
    index_artifact(
        kernel, dispatch_id, "settlement", ref, "operational_settlement_only"
    )
    kernel.settle_task(
        task_id=str(job_receipt["task_id"]),
        idempotency_key=f"experiment-dispatch:cancelled:{dispatch_id}",
        status="cancelled", output_sha256=ref.sha256,
        failure_reasons=("external_job_cancelled",),
    )
    kernel.artifacts.write_pointer(
        str(metadata["pointer_name"]), ref,
        metadata={
            "run_id": kernel.spec.run_id, "dispatch_id": dispatch_id,
            "status": "settled",
        },
    )


__all__ = [
    "EXPERIMENT_CANCELLATION_CHECKPOINT",
    "EXPERIMENT_JOB_RECEIPT_CHECKPOINT",
    "assert_external_job_result_settleable",
    "current_experiment_dispatch_context",
    "external_job_projection",
    "record_current_route_experiment_job_receipt",
    "request_current_route_experiment_cancellation",
]
