"""Execute submit/poll/cancel calls through the reserved experiment provider."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from cascade_planner.application.experiment_external_job_operations import (
    EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA,
    build_experiment_job_operation_request,
    validate_experiment_job_operation_request,
    validate_experiment_job_transport_result,
)
from cascade_planner.application.experiment_external_jobs import (
    EXTERNAL_JOB_TERMINAL_STATUSES,
    build_experiment_external_job_receipt,
)
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.orchestration.experiment_dispatch_support import (
    ExperimentDispatchError,
    artifact_ref,
    existing_receipt,
    index_artifact,
    read_object,
    reservation_metadata,
)
from cascade_planner.orchestration.experiment_external_job_runtime import (
    current_experiment_dispatch_context,
    external_job_projection,
    record_current_route_experiment_job_receipt,
)
from cascade_planner.providers.contracts import ProviderContext, validate_provider_result
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_TRANSPORT_ATTEMPT_CHECKPOINT = "experiment_external_job_transport_attempt"
EXPERIMENT_TRANSPORT_EXECUTION_SCHEMA = "experiment_job_transport_execution.v1"
_OPERATIONS = {"submit", "poll", "cancel"}


def execute_current_route_experiment_transport(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    operation: str,
    registry: ProviderRegistry,
    timeout_s: float = 0.0,
    mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    validations: Iterable[Mapping[str, Any]] = (),
    enable_experiment_transport: bool = False,
) -> dict[str, Any]:
    """Invoke one bounded provider operation and checkpoint its audit result."""

    if enable_experiment_transport is not True:
        raise ExperimentDispatchError("experiment_transport_explicit_enable_required")
    operation_value = str(operation or "").strip()
    if operation_value not in _OPERATIONS:
        raise ExperimentDispatchError("experiment_transport_operation_invalid")
    lifecycle, metadata, descriptor = current_experiment_dispatch_context(
        kernel, graph, acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, dispatch_id=dispatch_id, registry=registry,
        mechanism_proposals=mechanism_proposals, validations=validations,
    )
    if lifecycle["status"] != "in_flight":
        raise ExperimentDispatchError("experiment_dispatch_already_settled")
    if (
        descriptor.get("network_access") is not True
        or f"experiment.transport.{operation_value}"
        not in descriptor.get("capabilities", ())
    ):
        raise ExperimentDispatchError("experiment_provider_transport_unsupported")
    handoff = existing_receipt(kernel, lifecycle, str(metadata["pointer_name"]))
    handoff_value = dict(handoff.get("handoff") or {})
    provider = registry.get(str(metadata["provider_id"]))
    configured_sha256 = str(
        getattr(getattr(provider, "config", None), "content_sha256", "")
    )
    if (
        not configured_sha256
        or configured_sha256 != handoff_value.get("endpoint_config_sha256")
    ):
        raise ExperimentDispatchError("experiment_transport_config_binding_changed")
    projection = external_job_projection(kernel, lifecycle)
    request = read_object(
        kernel, artifact_ref(metadata.get("request_ref")),
        "experiment_dispatch_request_artifact_invalid",
    )
    attempts = transport_attempt_projection(
        kernel, lifecycle, provider_descriptor=descriptor,
        expected_endpoint_config_sha256=configured_sha256,
        expected_operator_identity=dict(handoff_value.get("operator_identity") or {}),
    )
    if operation_value == "submit" and projection["latest_receipt"]:
        matching = next(
            (
                row for row in reversed(attempts)
                if row["operation_request"]["operation"] == "submit"
                and row["transport_result"]["outcome"] == "success"
                and _result_matches_receipt(
                    row["transport_result"], projection["latest_receipt"]
                )
            ),
            None,
        )
        if matching is not None:
            return _execution_result(
                matching, job_receipt=projection["latest_receipt"],
                changed=False, cached=True,
            )
    _assert_operation_allowed(operation_value, projection)
    anchor = _operation_anchor(operation_value, projection)
    prior = [
        row for row in attempts
        if _operation_anchor_from_request(row["operation_request"]) == anchor
    ]
    reusable = next(
        (
            row for row in reversed(prior)
            if row["transport_result"]["outcome"] == "success"
            and (
                not projection["latest_receipt"]
                or row["transport_result"]["provider_sequence"]
                > projection["latest_receipt"]["provider_sequence"]
            )
        ),
        None,
    )
    if reusable is not None:
        return _apply_transport_execution(
            kernel, graph, execution=reusable, projection=projection,
            acceptance_spec=acceptance_spec, route_id=route_id,
            capabilities=capabilities, dispatch_id=dispatch_id, registry=registry,
            mechanism_proposals=mechanism_proposals, validations=validations,
            cached=True,
        )
    resolved_timeout = _resolved_timeout(timeout_s, request)
    operation_request = build_experiment_job_operation_request(
        operation=operation_value,
        attempt_number=max(
            (int(row["operation_request"]["attempt_number"]) for row in prior),
            default=0,
        ) + 1,
        run_id=kernel.spec.run_id,
        dispatch_id=dispatch_id,
        task_id=lifecycle["task_id"],
        request_id=str(metadata["request_id"]),
        request_sha256=str(metadata["request_sha256"]),
        provider_id=str(metadata["provider_id"]),
        provider_version=str(metadata["provider_version"]),
        timeout_s=resolved_timeout,
        external_job_id=str(projection["latest_receipt"].get("external_job_id") or ""),
        current_external_job_receipt_sha256=str(
            projection["latest_receipt"].get("content_sha256") or ""
        ),
        cancellation_request_sha256=str(
            projection["cancellation_request"].get("content_sha256") or ""
        ),
        execution_request=request if operation_value == "submit" else {},
    )
    envelope = registry.invoke(
        str(metadata["provider_id"]), operation_request,
        context=ProviderContext(
            run_id=kernel.spec.run_id, case_id=kernel.spec.run_id,
            target_smiles=kernel.spec.target_smiles,
            artifact_revision_id=str(kernel.state.graph_revision),
            config={
                "dispatch_id": dispatch_id,
                "operation_id": operation_request["operation_id"],
            },
        ),
    )
    result = dict(envelope.payload)
    if envelope.output_schema != EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA:
        raise ExperimentDispatchError("experiment_transport_result_schema_invalid")
    try:
        validate_experiment_job_transport_result(result, request=operation_request)
    except ValueError as exc:
        raise ExperimentDispatchError(str(exc)) from exc
    if (result["outcome"] == "success") != (envelope.accepted is True):
        raise ExperimentDispatchError("experiment_transport_envelope_outcome_invalid")
    if (
        result["endpoint_config_sha256"] != configured_sha256
        or result["recorded_by"] != handoff_value.get("operator_identity")
    ):
        raise ExperimentDispatchError("experiment_transport_provider_binding_invalid")
    execution = _transport_execution(operation_request, envelope.to_dict(), result)
    ref = kernel.artifacts.put_json(
        execution,
        logical_name=(
            f"{dispatch_id}.transport-{operation_value}-"
            f"{operation_request['attempt_number']}.json"
        ),
        producer="autoplanner.experiment_transport",
    )
    latest_checkpoint = str(projection["latest_checkpoint_sha256"])
    if lifecycle.get("checkpoints"):
        latest_checkpoint = str(
            dict(lifecycle["checkpoints"][-1]["payload"]).get("artifact_sha256") or ""
        )
    kernel.record_task_checkpoint(
        task_id=lifecycle["task_id"],
        checkpoint_kind=EXPERIMENT_TRANSPORT_ATTEMPT_CHECKPOINT,
        artifact_ref=ref,
        predecessor_checkpoint_sha256=latest_checkpoint,
        operational_status=str(result["outcome"]),
        idempotency_key=f"experiment-transport:{operation_request['operation_id']}",
        metadata={
            "dispatch_id": dispatch_id, "operation": operation_value,
            "operation_id": operation_request["operation_id"],
            "attempt_number": operation_request["attempt_number"],
        },
    )
    index_artifact(
        kernel, dispatch_id,
        f"transport-{operation_value}-{operation_request['attempt_number']}", ref,
        "external_experiment_transport_audit_only",
    )
    if result["outcome"] != "success":
        return _execution_result(execution, job_receipt={}, changed=False, cached=False)
    refreshed = kernel.task_lifecycle(lifecycle["task_id"])
    return _apply_transport_execution(
        kernel, graph, execution=execution,
        projection=external_job_projection(kernel, refreshed),
        acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, dispatch_id=dispatch_id, registry=registry,
        mechanism_proposals=mechanism_proposals, validations=validations,
        cached=False,
    )


def transport_attempt_projection(
    kernel: RunKernel,
    lifecycle: Mapping[str, Any],
    *,
    provider_descriptor: Mapping[str, Any],
    expected_endpoint_config_sha256: str,
    expected_operator_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay strict transport attempts from their task checkpoints."""

    attempts: list[dict[str, Any]] = []
    metadata = reservation_metadata(lifecycle)
    for event in lifecycle.get("checkpoints") or []:
        payload = dict(dict(event).get("payload") or {})
        if payload.get("checkpoint_kind") != EXPERIMENT_TRANSPORT_ATTEMPT_CHECKPOINT:
            continue
        value = read_object(
            kernel, artifact_ref(payload.get("artifact_ref")),
            "experiment_transport_attempt_artifact_invalid",
        )
        _validate_transport_execution(
            value, provider_descriptor=provider_descriptor,
            expected_endpoint_config_sha256=expected_endpoint_config_sha256,
            expected_operator_identity=expected_operator_identity,
        )
        request = dict(value["operation_request"])
        expected = {
            "run_id": kernel.spec.run_id,
            "dispatch_id": metadata.get("dispatch_id"),
            "task_id": lifecycle["task_id"],
            "request_id": metadata.get("request_id"),
            "request_sha256": metadata.get("request_sha256"),
            "provider_id": metadata.get("provider_id"),
            "provider_version": metadata.get("provider_version"),
        }
        if any(request.get(key) != item for key, item in expected.items()):
            raise ExperimentDispatchError("experiment_transport_task_binding_invalid")
        attempts.append(value)
    return attempts


def _apply_transport_execution(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    projection: Mapping[str, Any],
    acceptance_spec: Any,
    route_id: str,
    capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    dispatch_id: str,
    registry: ProviderRegistry,
    mechanism_proposals: Iterable[Mapping[str, Any]],
    validations: Iterable[Mapping[str, Any]],
    cached: bool,
) -> dict[str, Any]:
    result = dict(execution["transport_result"])
    latest = dict(projection["latest_receipt"])
    if latest:
        if (
            result["provider_sequence"] == latest["provider_sequence"]
            and result["external_job_id"] == latest["external_job_id"]
            and result["status"] == latest["status"]
        ):
            return _execution_result(
                execution, job_receipt=latest, changed=False, cached=cached
            )
        if result["provider_sequence"] <= latest["provider_sequence"]:
            raise ExperimentDispatchError("experiment_transport_result_sequence_invalid")
    receipt = build_experiment_external_job_receipt(
        dispatch_id=result["dispatch_id"], task_id=result["task_id"],
        request_id=result["request_id"], request_sha256=result["request_sha256"],
        provider_id=result["provider_id"], provider_version=result["provider_version"],
        external_job_id=result["external_job_id"],
        provider_sequence=result["provider_sequence"], status=result["status"],
        predecessor_receipt_sha256=str(latest.get("content_sha256") or ""),
        cancellation_request_sha256=str(
            projection["cancellation_request"].get("content_sha256") or ""
        ),
        recorded_by=result["recorded_by"], status_detail=result["status_detail"],
    )
    recorded = record_current_route_experiment_job_receipt(
        kernel, graph, acceptance_spec=acceptance_spec, route_id=route_id,
        capabilities=capabilities, dispatch_id=dispatch_id, job_receipt=receipt,
        registry=registry, mechanism_proposals=mechanism_proposals,
        validations=validations, enable_experiment_job_receipt=True,
    )
    return _execution_result(execution, job_receipt=recorded, changed=True, cached=cached)


def _assert_operation_allowed(
    operation: str, projection: Mapping[str, Any]
) -> None:
    latest = dict(projection["latest_receipt"])
    cancellation = dict(projection["cancellation_request"])
    if operation == "submit":
        if latest or cancellation:
            raise ExperimentDispatchError("experiment_external_job_already_submitted")
        return
    if not latest:
        raise ExperimentDispatchError("experiment_external_job_receipt_required")
    if latest["status"] in EXTERNAL_JOB_TERMINAL_STATUSES:
        raise ExperimentDispatchError("experiment_external_job_already_terminal")
    if operation == "cancel" and not cancellation:
        raise ExperimentDispatchError("experiment_cancellation_request_required")


def _operation_anchor(
    operation: str, projection: Mapping[str, Any]
) -> tuple[str, str, str]:
    return (
        operation,
        str(projection["latest_receipt"].get("content_sha256") or ""),
        str(projection["cancellation_request"].get("content_sha256") or ""),
    )


def _operation_anchor_from_request(
    request: Mapping[str, Any]
) -> tuple[str, str, str]:
    return (
        str(request["operation"]),
        str(request["current_external_job_receipt_sha256"]),
        str(request["cancellation_request_sha256"]),
    )


def _result_matches_receipt(
    result: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    return all(
        result.get(key) == receipt.get(key)
        for key in ("external_job_id", "provider_sequence", "status")
    )


def _resolved_timeout(value: float, request: Mapping[str, Any]) -> float:
    raw = value or dict(request.get("resource_hints") or {}).get("timeout_s") or 30.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ExperimentDispatchError("experiment_transport_timeout_invalid")
    timeout = float(raw)
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ExperimentDispatchError("experiment_transport_timeout_invalid")
    return timeout


def _transport_execution(
    request: Mapping[str, Any], envelope: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    return _with_digest({
        "schema_version": EXPERIMENT_TRANSPORT_EXECUTION_SCHEMA,
        "operation_request": dict(request),
        "provider_envelope": dict(envelope),
        "transport_result": dict(result),
        "semantics": {
            "task_checkpoint_is_the_operational_authority": True,
            "transport_result_grants_no_scientific_authority": True,
            "credentials_and_raw_response_body_are_not_persisted": True,
        },
    })


def _validate_transport_execution(
    value: Mapping[str, Any],
    *,
    provider_descriptor: Mapping[str, Any],
    expected_endpoint_config_sha256: str,
    expected_operator_identity: Mapping[str, Any],
) -> None:
    row = dict(value)
    expected = {
        "schema_version", "operation_request", "provider_envelope",
        "transport_result", "semantics", "content_sha256",
    }
    observed = str(row.pop("content_sha256", ""))
    request = dict(row.get("operation_request") or {})
    result = dict(row.get("transport_result") or {})
    envelope = dict(row.get("provider_envelope") or {})
    try:
        validate_experiment_job_operation_request(request)
        validate_experiment_job_transport_result(result, request=request)
    except ValueError as exc:
        raise ExperimentDispatchError(str(exc)) from exc
    reasons = validate_provider_result(envelope)
    if (
        set(value) != expected
        or value.get("schema_version") != EXPERIMENT_TRANSPORT_EXECUTION_SCHEMA
        or observed != strict_canonical_json_sha256(row)
        or reasons
        or envelope.get("provider_id") != provider_descriptor.get("provider_id")
        or envelope.get("provider_version") != provider_descriptor.get("version")
        or envelope.get("output_schema") != EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA
        or dict(envelope.get("payload") or {}) != result
        or (result["outcome"] == "success") != (envelope.get("accepted") is True)
        or result.get("endpoint_config_sha256")
        != expected_endpoint_config_sha256
        or result.get("recorded_by") != dict(expected_operator_identity)
    ):
        raise ExperimentDispatchError("experiment_transport_execution_invalid")


def _execution_result(
    execution: Mapping[str, Any], *, job_receipt: Mapping[str, Any],
    changed: bool, cached: bool,
) -> dict[str, Any]:
    value = {
        "schema_version": "experiment_job_transport_execution_result.v1",
        "operation_request": dict(execution["operation_request"]),
        "transport_result": dict(execution["transport_result"]),
        "job_receipt": dict(job_receipt),
        "changed": changed,
        "cached": cached,
        "semantics": {
            "transport_success_is_not_scientific_success": True,
            "job_receipt_uses_the_existing_task_checkpoint_chain": True,
        },
    }
    value["content_sha256"] = strict_canonical_json_sha256(value)
    return value


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "EXPERIMENT_TRANSPORT_ATTEMPT_CHECKPOINT",
    "EXPERIMENT_TRANSPORT_EXECUTION_SCHEMA",
    "execute_current_route_experiment_transport",
    "transport_attempt_projection",
]
