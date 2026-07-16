"""Shared integrity and artifact helpers for experiment dispatch runtime."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.providers.contracts import ProviderKind
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.runtime.artifact_store import (
    ArtifactRef,
    ArtifactReferenceError,
    ArtifactStoreError,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_DISPATCH_RECEIPT_SCHEMA = "experiment_dispatch_receipt.v1"
EXPERIMENT_DISPATCH_TASK_SCHEMA = "experiment_dispatch_task.v1"


class ExperimentDispatchError(RuntimeError):
    """Fail-closed operational dispatch or settlement error."""


def selection_from_reservation(
    kernel: RunKernel,
    registry: ProviderRegistry,
    request: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = current_reserved_descriptor(registry, metadata)
    if "experiment.dispatch.idempotent" not in descriptor["capabilities"]:
        raise ExperimentDispatchError("experiment_executor_no_longer_idempotent")
    selection = read_object(
        kernel,
        artifact_ref(metadata.get("selection_ref")),
        "experiment_dispatch_selection_artifact_invalid",
    )
    selected = dict(selection.get("selected") or {})
    selected_descriptor = dict(selected.get("descriptor") or {})
    if (
        selection.get("content_sha256") != metadata.get("selection_sha256")
        or selection.get("request_id") != request.get("request_id")
        or selection.get("request_sha256") != request.get("content_sha256")
        or selected.get("provider_id") != descriptor["provider_id"]
        or strict_canonical_json_sha256(selected_descriptor)
        != metadata.get("provider_descriptor_sha256")
        or with_digest(selection) != selection
    ):
        raise ExperimentDispatchError("experiment_dispatch_selection_binding_invalid")
    return selection


def current_reserved_descriptor(
    registry: ProviderRegistry, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    provider_id = str(metadata.get("provider_id") or "")
    descriptor = registry.descriptor(provider_id).to_dict()
    trust = registry.trust_record(provider_id)
    if (
        trust.get("trusted") is not True
        or descriptor["kind"] != ProviderKind.EXPERIMENT_EXECUTOR.value
        or descriptor["version"] != metadata.get("provider_version")
        or strict_canonical_json_sha256(descriptor)
        != metadata.get("provider_descriptor_sha256")
    ):
        raise ExperimentDispatchError("experiment_executor_registry_binding_changed")
    return descriptor


def existing_receipt(
    kernel: RunKernel, lifecycle: Mapping[str, Any], pointer_name: str
) -> dict[str, Any]:
    if lifecycle["status"] == "settled":
        digest = str(dict(lifecycle["settlement"]["payload"]).get("output_sha256") or "")
        receipt = read_object(kernel, digest, "experiment_settlement_artifact_invalid")
        validate_receipt(receipt, lifecycle=lifecycle)
        ref = artifact_ref_from_digest(kernel, digest, receipt)
        kernel.artifacts.write_pointer(pointer_name, ref, metadata={
            "run_id": kernel.spec.run_id,
            "dispatch_id": receipt.get("dispatch_id"),
            "status": "settled",
        })
        return receipt
    try:
        ref, _ = kernel.artifacts.load_pointer(pointer_name)
    except ArtifactReferenceError as exc:
        raise ExperimentDispatchError("experiment_dispatch_handoff_recovery_required") from exc
    receipt = read_object(kernel, ref, "experiment_dispatch_receipt_invalid")
    validate_receipt(receipt, lifecycle=lifecycle)
    return receipt


def validate_receipt(
    value: Mapping[str, Any], *, lifecycle: Mapping[str, Any]
) -> None:
    row = dict(value)
    metadata = reservation_metadata(lifecycle)
    observed = str(row.pop("content_sha256", ""))
    if (
        row.get("schema_version") != EXPERIMENT_DISPATCH_RECEIPT_SCHEMA
        or row.get("dispatch_id") != metadata.get("dispatch_id")
        or row.get("task_id") != lifecycle.get("task_id")
        or row.get("request_id") != metadata.get("request_id")
        or row.get("request_sha256") != metadata.get("request_sha256")
        or row.get("provider_id") != metadata.get("provider_id")
        or row.get("provider_version") != metadata.get("provider_version")
        or row.get("status") != (
            "settled" if lifecycle.get("status") == "settled"
            else "awaiting_external_result"
        )
        or not observed
        or observed != strict_canonical_json_sha256(row)
    ):
        raise ExperimentDispatchError("experiment_dispatch_receipt_binding_invalid")


def reservation_metadata(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    reservation = dict(lifecycle.get("reservation") or {})
    metadata = dict(dict(reservation.get("payload") or {}).get("metadata") or {})
    if metadata.get("schema_version") != EXPERIMENT_DISPATCH_TASK_SCHEMA:
        raise ExperimentDispatchError("experiment_dispatch_reservation_invalid")
    return metadata


def verify_result_artifacts(kernel: RunKernel, result: Mapping[str, Any]) -> None:
    for row in result.get("artifact_refs") or []:
        digest = str(dict(row).get("sha256") or "")
        try:
            kernel.artifacts.verify(digest)
        except (ArtifactStoreError, OSError, ValueError) as exc:
            raise ExperimentDispatchError(
                f"experiment_result_artifact_unavailable:{digest}"
            ) from exc


def dispatch_identity(request: Mapping[str, Any]) -> tuple[str, str, str]:
    digest = strict_canonical_json_sha256({
        "run_id": request["run_id"],
        "request_id": request["request_id"],
        "request_sha256": request["content_sha256"],
    })
    dispatch_id = f"experiment-dispatch:{digest[:32]}"
    task_id = f"experiment-dispatch-task:{digest[:32]}"
    run_digest = hashlib.sha256(str(request["run_id"]).encode("utf-8")).hexdigest()
    return dispatch_id, task_id, f"experiment_dispatch/{run_digest[:24]}/{digest}"


def task_id_from_dispatch(dispatch_id: str) -> str:
    prefix = "experiment-dispatch:"
    value = str(dispatch_id or "")
    if not value.startswith(prefix) or len(value) != len(prefix) + 32:
        raise ExperimentDispatchError("experiment_dispatch_id_invalid")
    return "experiment-dispatch-task:" + value[len(prefix):]


def artifact_ref(value: Any) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(dict(value or {}))
    except (TypeError, ValueError, ArtifactReferenceError) as exc:
        raise ExperimentDispatchError("experiment_dispatch_artifact_ref_invalid") from exc


def artifact_ref_from_digest(
    kernel: RunKernel, digest: str, value: Mapping[str, Any]
) -> ArtifactRef:
    data = kernel.artifacts.read_bytes(digest)
    return ArtifactRef(
        sha256=digest,
        size_bytes=len(data),
        media_type="application/json",
        logical_name=f"{value.get('dispatch_id', 'experiment-dispatch')}.settlement.json",
        producer="autoplanner.experiment_dispatch",
    )


def read_object(
    kernel: RunKernel, ref: ArtifactRef | str, reason: str
) -> dict[str, Any]:
    try:
        value = kernel.artifacts.read_json(ref)
    except (ArtifactStoreError, OSError, ValueError) as exc:
        raise ExperimentDispatchError(reason) from exc
    if not isinstance(value, Mapping):
        raise ExperimentDispatchError(reason)
    return dict(value)


def index_artifact(
    kernel: RunKernel, dispatch_id: str, role: str, ref: ArtifactRef, scope: str
) -> None:
    kernel.index.index_artifact(
        run_id=kernel.spec.run_id,
        artifact_id=f"{dispatch_id}:{role}",
        ref=ref,
        revision=kernel.state.graph_revision,
        authority_scope=scope,
    )


def receipt_semantics() -> dict[str, bool]:
    return {
        "run_kernel_is_single_task_ledger": True,
        "artifact_pointer_is_rebuildable_projection": True,
        "receipt_grants_no_validation_claim_graph_or_completion": True,
        "current_frontier_and_domain_gate_remain_required": True,
    }


def with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "EXPERIMENT_DISPATCH_RECEIPT_SCHEMA",
    "EXPERIMENT_DISPATCH_TASK_SCHEMA",
    "ExperimentDispatchError",
    "artifact_ref",
    "current_reserved_descriptor",
    "dispatch_identity",
    "existing_receipt",
    "index_artifact",
    "read_object",
    "receipt_semantics",
    "reservation_metadata",
    "selection_from_reservation",
    "task_id_from_dispatch",
    "verify_result_artifacts",
    "with_digest",
]
