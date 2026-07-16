"""Host-trusted experiment executor selection and manual handoff adapter."""

from __future__ import annotations

import math
from typing import Any, Mapping

from cascade_planner.application.experiment_execution_contracts import (
    EXPERIMENT_EXECUTION_REQUEST_SCHEMA,
    validate_experiment_execution_request,
)
from cascade_planner.application.experiment_execution_results import (
    EXPERIMENT_EXECUTION_RESULT_SCHEMA,
)
from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
)
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.providers.registry import ProviderRegistryError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


EXPERIMENT_EXECUTOR_POLICY_SCHEMA = "experiment_executor_policy.v1"
EXPERIMENT_EXECUTOR_SELECTION_SCHEMA = "experiment_executor_selection.v1"
EXPERIMENT_DISPATCH_HANDOFF_SCHEMA = "experiment_dispatch_handoff.v1"
_IDEMPOTENT_CAPABILITY = "experiment.dispatch.idempotent"
_HANDOFF_REQUIREMENTS = {
    "artifact_sha256_required": True,
    "current_frontier_reaudit_required": True,
    "domain_validation_gate_required": True,
    "request_binding_required": True,
}
_HANDOFF_SEMANTICS = {
    "handoff_is_not_experiment_completion": True,
    "handoff_grants_no_validation_claim_or_route_authority": True,
    "manual_operator_execution_is_external": True,
}


class ExperimentExecutorPolicyError(ValueError):
    """Raised when dispatch policy or provider eligibility fails closed."""


class ManualExperimentExecutorProvider:
    """Materialize a deterministic lab handoff; it performs no experiment."""

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.manual_experiment_executor",
        kind=ProviderKind.EXPERIMENT_EXECUTOR,
        version="1.0.0",
        input_schemas=(EXPERIMENT_EXECUTION_REQUEST_SCHEMA,),
        output_schemas=(EXPERIMENT_DISPATCH_HANDOFF_SCHEMA,),
        correlation_group="manual_experiment_handoff",
        capabilities=(
            _IDEMPOTENT_CAPABILITY,
            "experiment.recovery.reinvoke",
            "experiment.domain.biocatalytic",
            "experiment.domain.execution",
            "experiment.domain.mechanism",
        ),
        deterministic=True,
        network_access=False,
        estimated_cost_units=0.0,
    )

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        value = dict(request)
        validate_experiment_execution_request(value)
        dispatch_id = str(context.config.get("dispatch_id") or "").strip()
        if not dispatch_id:
            raise ValueError("experiment_dispatch_id_required")
        payload = _with_digest(
            {
                "schema_version": EXPERIMENT_DISPATCH_HANDOFF_SCHEMA,
                "dispatch_id": dispatch_id,
                "request_id": value["request_id"],
                "request_sha256": value["content_sha256"],
                "run_id": value["run_id"],
                "route_id": value["route_id"],
                "domain": value["domain"],
                "executor_id": self.descriptor.provider_id,
                "executor_version": self.descriptor.version,
                "state": "awaiting_external_result",
                "expected_result_schema": EXPERIMENT_EXECUTION_RESULT_SCHEMA,
                "submission_requirements": dict(_HANDOFF_REQUIREMENTS),
                "semantics": dict(_HANDOFF_SEMANTICS),
            }
        )
        validate_experiment_dispatch_handoff(payload, request=value)
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema=EXPERIMENT_DISPATCH_HANDOFF_SCHEMA,
            accepted=True,
            payload=payload,
            reasons=(),
        )


def select_experiment_executor(
    registry: ProviderRegistry,
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only a host-trusted, explicitly allowed, idempotent executor."""

    request_value = dict(request)
    validate_experiment_execution_request(request_value)
    policy_value = _normalize_policy(policy)
    if policy_value["enabled"] is not True:
        raise ExperimentExecutorPolicyError("experiment_executor_policy_disabled")
    if request_value["domain"] not in policy_value["allowed_domains"]:
        raise ExperimentExecutorPolicyError("experiment_executor_domain_not_allowed")
    candidates: list[dict[str, Any]] = []
    rejections: dict[str, list[str]] = {}
    for provider_id in policy_value["allowed_provider_ids"]:
        reasons: list[str] = []
        try:
            descriptor = registry.descriptor(provider_id)
            trust = registry.trust_record(provider_id)
        except ProviderRegistryError:
            rejections[provider_id] = ["provider_not_registered"]
            continue
        if trust.get("trusted") is not True:
            reasons.append("provider_not_host_trusted")
        if descriptor.kind is not ProviderKind.EXPERIMENT_EXECUTOR:
            reasons.append("provider_kind_invalid")
        if EXPERIMENT_EXECUTION_REQUEST_SCHEMA not in descriptor.input_schemas:
            reasons.append("provider_request_schema_unsupported")
        if EXPERIMENT_DISPATCH_HANDOFF_SCHEMA not in descriptor.output_schemas:
            reasons.append("provider_handoff_schema_unsupported")
        if _IDEMPOTENT_CAPABILITY not in descriptor.capabilities:
            reasons.append("provider_dispatch_not_idempotent")
        if f"experiment.domain.{request_value['domain']}" not in descriptor.capabilities:
            reasons.append("provider_domain_capability_missing")
        if descriptor.network_access and not policy_value["allow_network_access"]:
            reasons.append("provider_network_access_disallowed")
        if descriptor.estimated_cost_units > policy_value["max_estimated_cost_units"]:
            reasons.append("provider_cost_limit_exceeded")
        if reasons:
            rejections[provider_id] = sorted(set(reasons))
        else:
            candidates.append(
                {
                    "provider_id": provider_id,
                    "descriptor": descriptor.to_dict(),
                    "trust_record": trust,
                }
            )
    if not candidates:
        raise ExperimentExecutorPolicyError(
            "no_eligible_experiment_executor:" + ",".join(
                f"{key}={'|'.join(value)}" for key, value in sorted(rejections.items())
            )
        )
    priority = {value: index for index, value in enumerate(policy_value["preferred_provider_ids"])}
    candidates.sort(key=lambda row: (priority.get(row["provider_id"], len(priority)), row["provider_id"]))
    selection = {
        "schema_version": EXPERIMENT_EXECUTOR_SELECTION_SCHEMA,
        "request_id": request_value["request_id"],
        "request_sha256": request_value["content_sha256"],
        "policy": policy_value,
        "selected": candidates[0],
        "eligible_provider_ids": [row["provider_id"] for row in candidates],
        "rejections": rejections,
        "semantics": {
            "host_registry_is_trust_authority": True,
            "client_policy_cannot_elevate_provider_trust": True,
            "selection_grants_no_scientific_authority": True,
        },
    }
    return _with_digest(selection)


def validate_experiment_dispatch_handoff(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> None:
    row = dict(value)
    expected = {
        "schema_version", "dispatch_id", "request_id", "request_sha256",
        "run_id", "route_id", "domain", "executor_id", "executor_version",
        "state", "expected_result_schema", "submission_requirements",
        "semantics", "content_sha256",
    }
    bound = dict(request)
    valid = (
        set(row) == expected
        and row.get("schema_version") == EXPERIMENT_DISPATCH_HANDOFF_SCHEMA
        and row.get("state") == "awaiting_external_result"
        and row.get("expected_result_schema") == EXPERIMENT_EXECUTION_RESULT_SCHEMA
        and row.get("submission_requirements") == _HANDOFF_REQUIREMENTS
        and row.get("semantics") == _HANDOFF_SEMANTICS
        and row.get("request_id") == bound.get("request_id")
        and row.get("request_sha256") == bound.get("content_sha256")
        and row.get("run_id") == bound.get("run_id")
        and row.get("route_id") == bound.get("route_id")
        and row.get("domain") == bound.get("domain")
        and _digest_valid(row)
    )
    if not valid:
        raise ExperimentExecutorPolicyError("experiment_dispatch_handoff_invalid")


def _normalize_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value) if isinstance(value, Mapping) else {}
    expected = {
        "schema_version", "enabled", "allowed_provider_ids", "preferred_provider_ids",
        "allowed_domains", "allow_network_access", "max_estimated_cost_units",
    }
    if set(row) != expected or row.get("schema_version") != EXPERIMENT_EXECUTOR_POLICY_SCHEMA:
        raise ExperimentExecutorPolicyError("experiment_executor_policy_contract_invalid")
    if not all(
        isinstance(row.get(key), list)
        for key in ("allowed_provider_ids", "preferred_provider_ids", "allowed_domains")
    ):
        raise ExperimentExecutorPolicyError("experiment_executor_policy_values_invalid")
    if (
        not isinstance(row.get("enabled"), bool)
        or not isinstance(row.get("allow_network_access"), bool)
        or isinstance(row.get("max_estimated_cost_units"), bool)
        or not isinstance(row.get("max_estimated_cost_units"), (int, float))
    ):
        raise ExperimentExecutorPolicyError("experiment_executor_policy_values_invalid")
    allowed = sorted({str(item) for item in row.get("allowed_provider_ids") or [] if str(item)})
    preferred = [str(item) for item in row.get("preferred_provider_ids") or [] if str(item)]
    domains = sorted({str(item) for item in row.get("allowed_domains") or [] if str(item)})
    cost = float(row.get("max_estimated_cost_units"))
    if (
        not allowed or not domains or len(preferred) != len(set(preferred))
        or not set(preferred).issubset(allowed) or not math.isfinite(cost) or cost < 0
    ):
        raise ExperimentExecutorPolicyError("experiment_executor_policy_values_invalid")
    if not set(domains).issubset({"biocatalytic", "execution", "mechanism"}):
        raise ExperimentExecutorPolicyError("experiment_executor_policy_domain_invalid")
    return {
        "schema_version": EXPERIMENT_EXECUTOR_POLICY_SCHEMA,
        "enabled": row["enabled"],
        "allowed_provider_ids": allowed,
        "preferred_provider_ids": preferred,
        "allowed_domains": domains,
        "allow_network_access": row["allow_network_access"],
        "max_estimated_cost_units": cost,
    }


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    observed = str(row.pop("content_sha256", ""))
    return bool(observed) and observed == strict_canonical_json_sha256(row)


__all__ = [
    "EXPERIMENT_DISPATCH_HANDOFF_SCHEMA",
    "EXPERIMENT_EXECUTOR_POLICY_SCHEMA",
    "ExperimentExecutorPolicyError",
    "ManualExperimentExecutorProvider",
    "select_experiment_executor",
    "validate_experiment_dispatch_handoff",
]
