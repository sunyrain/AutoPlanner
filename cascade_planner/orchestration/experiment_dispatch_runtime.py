"""Bounded, recoverable experiment dispatch on the single RunKernel ledger."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.orchestration.experiment_dispatch_handoff import (
    materialize_experiment_handoff,
)
from cascade_planner.orchestration.experiment_external_job_runtime import (
    assert_external_job_result_settleable,
)
from cascade_planner.orchestration.experiment_dispatch_support import (
    EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
    EXPERIMENT_DISPATCH_TASK_SCHEMA,
    ExperimentDispatchError,
    artifact_ref,
    current_reserved_descriptor,
    dispatch_identity,
    existing_receipt,
    index_artifact,
    read_object,
    receipt_semantics,
    reservation_metadata,
    selection_from_reservation,
    task_id_from_dispatch,
    verify_result_artifacts,
    with_digest,
)
from cascade_planner.orchestration.experiment_execution_runtime import (
    audit_current_route_experiment_result,
    locate_current_route_experiment_request,
)
from cascade_planner.providers.experiment import select_experiment_executor
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def dispatch_current_route_experiment(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    request_id: str,
    registry: ProviderRegistry,
    policy: Mapping[str, Any],
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_dispatch: bool = False,
) -> dict[str, Any]:
    """Reserve and materialize one idempotent executor handoff."""

    if enable_experiment_dispatch is not True:
        raise ExperimentDispatchError("experiment_dispatch_explicit_enable_required")
    frontier, request = locate_current_route_experiment_request(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        request_id=request_id,
        mechanism_proposals=mechanism_proposals,
        validations=validations,
    )
    selection = select_experiment_executor(registry, request, policy)
    dispatch_id, task_id, pointer_name = dispatch_identity(request)
    lifecycle = kernel.task_lifecycle(task_id)
    if lifecycle["status"] != "absent":
        metadata = reservation_metadata(lifecycle)
        if str(metadata.get("request_sha256") or "") != request["content_sha256"]:
            raise ExperimentDispatchError("experiment_dispatch_request_binding_changed")
        if str(metadata.get("provider_id") or "") not in selection["eligible_provider_ids"]:
            raise ExperimentDispatchError("experiment_dispatch_provider_no_longer_allowed")
        try:
            return existing_receipt(kernel, lifecycle, pointer_name)
        except ExperimentDispatchError as exc:
            if str(exc) != "experiment_dispatch_handoff_recovery_required":
                raise
        request_ref = artifact_ref(metadata.get("request_ref"))
        stored_request = read_object(
            kernel, request_ref, "experiment_dispatch_request_artifact_invalid"
        )
        if stored_request != request:
            raise ExperimentDispatchError("experiment_dispatch_request_binding_changed")
        return materialize_experiment_handoff(
            kernel,
            request=request,
            frontier_sha256=frontier["content_sha256"],
            registry=registry,
            selection=selection_from_reservation(kernel, registry, request, metadata),
            dispatch_id=str(metadata["dispatch_id"]),
            task_id=task_id,
            pointer_name=pointer_name,
            request_ref=request_ref,
        )

    selection_ref = kernel.artifacts.put_json(
        selection,
        logical_name=f"{dispatch_id}.selection.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(
        kernel, dispatch_id, "selection", selection_ref, "executor_selection_only"
    )
    selection = read_object(
        kernel, selection_ref, "experiment_dispatch_selection_artifact_invalid"
    )
    descriptor = dict(selection["selected"]["descriptor"])
    request_ref = kernel.artifacts.put_json(
        request,
        logical_name=f"{dispatch_id}.request.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(kernel, dispatch_id, "request", request_ref, "experiment_request_only")
    metadata = {
        "schema_version": EXPERIMENT_DISPATCH_TASK_SCHEMA,
        "dispatch_id": dispatch_id,
        "pointer_name": pointer_name,
        "request_id": request["request_id"],
        "request_sha256": request["content_sha256"],
        "request_ref": request_ref.to_dict(),
        "route_id": request["route_id"],
        "domain": request["domain"],
        "provider_id": descriptor["provider_id"],
        "provider_version": descriptor["version"],
        "provider_descriptor_sha256": strict_canonical_json_sha256(descriptor),
        "selection_sha256": selection["content_sha256"],
        "selection_ref": selection_ref.to_dict(),
    }
    kernel.reserve_task(
        task_id=task_id,
        kind="experiment",
        idempotency_key=f"experiment-dispatch:reserve:{dispatch_id}",
        input_revision=kernel.state.graph_revision,
        metadata=metadata,
    )
    return materialize_experiment_handoff(
        kernel,
        request=request,
        frontier_sha256=frontier["content_sha256"],
        registry=registry,
        selection=selection,
        dispatch_id=dispatch_id,
        task_id=task_id,
        pointer_name=pointer_name,
        request_ref=request_ref,
    )


def recover_current_route_experiment_dispatch(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_dispatch_recovery: bool = False,
) -> dict[str, Any]:
    """Rebuild a missing idempotent handoff from the durable reservation."""

    if enable_experiment_dispatch_recovery is not True:
        raise ExperimentDispatchError("experiment_dispatch_recovery_explicit_enable_required")
    task_id = task_id_from_dispatch(dispatch_id)
    lifecycle = kernel.task_lifecycle(task_id)
    if lifecycle["status"] == "absent":
        raise ExperimentDispatchError("experiment_dispatch_not_in_flight")
    metadata = reservation_metadata(lifecycle)
    if lifecycle["status"] == "settled":
        return existing_receipt(kernel, lifecycle, str(metadata["pointer_name"]))
    if lifecycle["status"] != "in_flight":
        raise ExperimentDispatchError("experiment_dispatch_not_in_flight")
    try:
        return existing_receipt(kernel, lifecycle, str(metadata["pointer_name"]))
    except ExperimentDispatchError as exc:
        if str(exc) != "experiment_dispatch_handoff_recovery_required":
            raise
    request_ref = artifact_ref(metadata.get("request_ref"))
    request = read_object(kernel, request_ref, "experiment_dispatch_request_artifact_invalid")
    frontier, current = locate_current_route_experiment_request(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        request_id=str(request.get("request_id") or ""),
        mechanism_proposals=mechanism_proposals,
        validations=validations,
    )
    if current != request or metadata.get("request_sha256") != request.get("content_sha256"):
        raise ExperimentDispatchError("experiment_dispatch_request_not_current")
    return materialize_experiment_handoff(
        kernel,
        request=request,
        frontier_sha256=frontier["content_sha256"],
        registry=registry,
        selection=selection_from_reservation(kernel, registry, request, metadata),
        dispatch_id=str(metadata["dispatch_id"]),
        task_id=task_id,
        pointer_name=str(metadata["pointer_name"]),
        request_ref=request_ref,
    )


def settle_current_route_experiment_dispatch(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    result: Mapping[str, Any],
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_settlement: bool = False,
) -> dict[str, Any]:
    """Audit and settle a dispatch without accepting its validation candidate."""

    if enable_experiment_settlement is not True:
        raise ExperimentDispatchError("experiment_settlement_explicit_enable_required")
    task_id = task_id_from_dispatch(dispatch_id)
    lifecycle = kernel.task_lifecycle(task_id)
    if lifecycle["status"] == "absent":
        raise ExperimentDispatchError("experiment_dispatch_not_found")
    metadata = reservation_metadata(lifecycle)
    pointer_name = str(metadata.get("pointer_name") or "")
    result_value = dict(result)
    result_ref = kernel.artifacts.put_json(
        result_value,
        logical_name=f"{dispatch_id}.result.json",
        producer="autoplanner.experiment_dispatch",
    )
    if lifecycle["status"] == "settled":
        receipt = existing_receipt(kernel, lifecycle, pointer_name)
        if dict(receipt.get("result_ref") or {}).get("sha256") != result_ref.sha256:
            raise ExperimentDispatchError("experiment_dispatch_already_settled_with_other_result")
        return receipt
    descriptor = current_reserved_descriptor(registry, metadata)
    if dict(result_value.get("executor") or {}) != {
        "executor_id": descriptor["provider_id"],
        "executor_version": descriptor["version"],
    }:
        raise ExperimentDispatchError("experiment_result_executor_not_dispatched_provider")
    assert_external_job_result_settleable(kernel, lifecycle, result_value)
    verify_result_artifacts(kernel, result_value)
    review = audit_current_route_experiment_result(
        graph,
        acceptance_spec=acceptance_spec,
        route_id=route_id,
        capabilities=capabilities,
        result=result_value,
        mechanism_proposals=mechanism_proposals,
        validations=validations,
    )
    if review["result_audit"]["reasons"]:
        raise ExperimentDispatchError(
            "experiment_result_audit_rejected:"
            + ",".join(review["result_audit"]["reasons"])
        )
    index_artifact(
        kernel, dispatch_id, "result", result_ref, "experiment_result_envelope_only"
    )
    review_ref = kernel.artifacts.put_json(
        review,
        logical_name=f"{dispatch_id}.review.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(kernel, dispatch_id, "review", review_ref, "domain_gate_candidate_only")
    receipt = with_digest({
        "schema_version": EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
        "dispatch_id": dispatch_id,
        "task_id": task_id,
        "status": "settled",
        "request_id": str(result_value.get("request_id") or ""),
        "request_sha256": str(result_value.get("request_sha256") or ""),
        "provider_id": descriptor["provider_id"],
        "provider_version": descriptor["version"],
        "result_status": str(result_value.get("status") or ""),
        "result_ref": result_ref.to_dict(),
        "review_ref": review_ref.to_dict(),
        "domain_validation_candidate": review["domain_validation_candidate"],
        "next_boundary": review["next_boundary"],
        "semantics": receipt_semantics(),
    })
    receipt_ref = kernel.artifacts.put_json(
        receipt,
        logical_name=f"{dispatch_id}.settlement.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(
        kernel, dispatch_id, "settlement", receipt_ref, "operational_settlement_only"
    )
    kernel.settle_task(
        task_id=task_id,
        idempotency_key=f"experiment-dispatch:settle:{dispatch_id}",
        status="cancelled" if result_value.get("status") == "aborted" else "completed",
        output_sha256=receipt_ref.sha256,
        failure_reasons=result_value.get("failure_reasons") or (),
    )
    kernel.artifacts.write_pointer(pointer_name, receipt_ref, metadata={
        "run_id": kernel.spec.run_id,
        "dispatch_id": dispatch_id,
        "status": "settled",
    })
    return read_object(kernel, receipt_ref, "experiment_settlement_artifact_invalid")


__all__ = [
    "EXPERIMENT_DISPATCH_RECEIPT_SCHEMA",
    "ExperimentDispatchError",
    "dispatch_current_route_experiment",
    "recover_current_route_experiment_dispatch",
    "settle_current_route_experiment_dispatch",
]
