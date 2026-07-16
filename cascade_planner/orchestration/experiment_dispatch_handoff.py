"""Materialize one provider handoff for a reserved experiment task."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.providers.contracts import ProviderContext
from cascade_planner.providers.experiment import (
    EXPERIMENT_DISPATCH_HANDOFF_SCHEMA,
    validate_experiment_dispatch_handoff,
)
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.runtime.artifact_store import ArtifactRef
from cascade_planner.orchestration.experiment_dispatch_support import (
    EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
    ExperimentDispatchError,
    index_artifact,
    read_object,
    receipt_semantics,
    with_digest,
)


def materialize_experiment_handoff(
    kernel: RunKernel,
    *,
    request: Mapping[str, Any],
    frontier_sha256: str,
    registry: ProviderRegistry,
    selection: Mapping[str, Any],
    dispatch_id: str,
    task_id: str,
    pointer_name: str,
    request_ref: ArtifactRef,
) -> dict[str, Any]:
    selected = dict(selection["selected"])
    descriptor = dict(selected["descriptor"])
    envelope = registry.invoke(
        descriptor["provider_id"],
        request,
        context=ProviderContext(
            run_id=kernel.spec.run_id,
            case_id=kernel.spec.run_id,
            target_smiles=kernel.spec.target_smiles,
            artifact_revision_id=str(kernel.state.graph_revision),
            config={"dispatch_id": dispatch_id},
        ),
    )
    if envelope.accepted is not True or envelope.output_schema != EXPERIMENT_DISPATCH_HANDOFF_SCHEMA:
        raise ExperimentDispatchError("experiment_executor_handoff_rejected")
    handoff = dict(envelope.payload)
    validate_experiment_dispatch_handoff(handoff, request=request)
    if (handoff.get("executor_id"), handoff.get("executor_version")) != (
        descriptor["provider_id"], descriptor["version"]
    ):
        raise ExperimentDispatchError("experiment_handoff_executor_identity_mismatch")
    envelope_ref = kernel.artifacts.put_json(
        envelope.to_dict(), logical_name=f"{dispatch_id}.handoff-envelope.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(kernel, dispatch_id, "handoff", envelope_ref, "experiment_handoff_only")
    receipt = with_digest({
        "schema_version": EXPERIMENT_DISPATCH_RECEIPT_SCHEMA,
        "dispatch_id": dispatch_id, "task_id": task_id,
        "status": "awaiting_external_result",
        "request_id": request["request_id"],
        "request_sha256": request["content_sha256"],
        "provider_id": descriptor["provider_id"],
        "provider_version": descriptor["version"],
        "frontier_sha256": frontier_sha256,
        "request_ref": request_ref.to_dict(),
        "handoff_envelope_ref": envelope_ref.to_dict(),
        "handoff": handoff, "selection": dict(selection),
        "semantics": receipt_semantics(),
    })
    receipt_ref = kernel.artifacts.put_json(
        receipt, logical_name=f"{dispatch_id}.receipt.json",
        producer="autoplanner.experiment_dispatch",
    )
    index_artifact(kernel, dispatch_id, "receipt", receipt_ref, "operational_dispatch_only")
    kernel.artifacts.write_pointer(pointer_name, receipt_ref, metadata={
        "run_id": kernel.spec.run_id, "dispatch_id": dispatch_id,
        "status": "awaiting_external_result",
    })
    return read_object(kernel, receipt_ref, "experiment_dispatch_receipt_invalid")


__all__ = ["materialize_experiment_handoff"]
