"""Host-configured bounded HTTP bridge for external experiment job control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import Any, Mapping
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from cascade_planner.application.experiment_execution_contracts import (
    EXPERIMENT_DOMAINS,
    EXPERIMENT_EXECUTION_REQUEST_SCHEMA,
    validate_experiment_execution_request,
)
from cascade_planner.application.experiment_execution_results import (
    EXPERIMENT_EXECUTION_RESULT_SCHEMA,
)
from cascade_planner.application.experiment_external_job_operations import (
    EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA,
    EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA,
    build_experiment_job_transport_result,
    validate_experiment_job_operation_request,
)
from cascade_planner.application.experiment_external_jobs import (
    build_experiment_operator_identity,
    validate_experiment_operator_identity,
)
from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
)
from cascade_planner.providers.experiment import (
    EXPERIMENT_HTTP_DISPATCH_HANDOFF_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


_HTTP_HANDOFF_SEMANTICS = {
    "handoff_does_not_submit_the_external_job": True,
    "submit_poll_cancel_require_explicit_host_calls": True,
    "endpoint_and_credentials_are_host_configured": True,
    "handoff_grants_no_validation_claim_or_route_authority": True,
}
_HTTP_HANDOFF_REQUIREMENTS = {
    "artifact_sha256_required": True,
    "current_frontier_reaudit_required": True,
    "domain_validation_gate_required": True,
    "request_binding_required": True,
    "explicit_transport_enable_required": True,
}


@dataclass(frozen=True, slots=True)
class HttpExperimentExecutorConfig:
    provider_id: str
    version: str
    base_url: str
    submit_path: str = "/jobs"
    poll_path_template: str = "/jobs/{external_job_id}"
    cancel_path_template: str = "/jobs/{external_job_id}/cancel"
    auth_token_env: str = ""
    operator_principal_id: str = "service:autoplanner-http-experiment"
    allowed_domains: tuple[str, ...] = tuple(sorted(EXPERIMENT_DOMAINS))
    max_timeout_s: float = 30.0
    max_response_bytes: int = 1_000_000
    estimated_cost_units: float = 0.0
    allow_loopback_http: bool = False

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id).strip()
        version = str(self.version).strip()
        principal = str(self.operator_principal_id).strip()
        auth_env = str(self.auth_token_env).strip()
        base_url = _normalized_base_url(self.base_url)
        parsed = urlsplit(base_url)
        loopback = _loopback_host(parsed.hostname)
        if not provider_id or not version or not principal:
            raise ValueError("http_experiment_provider_identity_invalid")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and loopback and self.allow_loopback_http is True
        ):
            raise ValueError("http_experiment_endpoint_requires_https")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("http_experiment_endpoint_authority_invalid")
        if not auth_env and not loopback:
            raise ValueError("http_experiment_auth_token_env_required")
        if auth_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", auth_env):
            raise ValueError("http_experiment_auth_token_env_invalid")
        for path, template in (
            (self.submit_path, False),
            (self.poll_path_template, True),
            (self.cancel_path_template, True),
        ):
            _validate_path(path, template=template)
        domains = tuple(sorted({str(item) for item in self.allowed_domains}))
        if not domains or not set(domains).issubset(EXPERIMENT_DOMAINS):
            raise ValueError("http_experiment_allowed_domains_invalid")
        if (
            not math.isfinite(float(self.max_timeout_s))
            or not 0 < float(self.max_timeout_s) <= 3600
            or isinstance(self.max_response_bytes, bool)
            or not 1 <= int(self.max_response_bytes) <= 20_000_000
            or not math.isfinite(float(self.estimated_cost_units))
            or float(self.estimated_cost_units) < 0
            or not isinstance(self.allow_loopback_http, bool)
        ):
            raise ValueError("http_experiment_transport_limits_invalid")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "operator_principal_id", principal)
        object.__setattr__(self, "allowed_domains", domains)
        object.__setattr__(self, "auth_token_env", auth_env)
        object.__setattr__(self, "max_timeout_s", float(self.max_timeout_s))
        object.__setattr__(self, "max_response_bytes", int(self.max_response_bytes))
        object.__setattr__(self, "estimated_cost_units", float(self.estimated_cost_units))

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": "http_experiment_executor_config.v1",
            "provider_id": self.provider_id,
            "version": self.version,
            "base_url": self.base_url,
            "submit_path": self.submit_path,
            "poll_path_template": self.poll_path_template,
            "cancel_path_template": self.cancel_path_template,
            "auth_token_env": self.auth_token_env,
            "operator_principal_id": self.operator_principal_id,
            "allowed_domains": list(self.allowed_domains),
            "max_timeout_s": self.max_timeout_s,
            "max_response_bytes": self.max_response_bytes,
            "estimated_cost_units": self.estimated_cost_units,
            "allow_loopback_http": self.allow_loopback_http,
            "semantics": {
                "contains_no_credential_value": True,
                "client_payload_cannot_override_endpoint": True,
            },
        }
        value["content_sha256"] = strict_canonical_json_sha256(value)
        return value

    @property
    def content_sha256(self) -> str:
        return str(self.to_dict()["content_sha256"])

    @property
    def authentication_context_sha256(self) -> str:
        return strict_canonical_json_sha256({
            "provider_id": self.provider_id,
            "base_url": self.base_url,
            "auth_mode": "bearer_env" if self.auth_token_env else "none",
            "auth_token_env": self.auth_token_env,
            "operator_principal_id": self.operator_principal_id,
        })


class HttpExperimentExecutorProvider:
    """Execute explicit JSON bridge calls without storing credentials or raw bodies."""

    def __init__(
        self,
        config: HttpExperimentExecutorConfig,
        *,
        environ: Mapping[str, str] | None = None,
        requester: Any = None,
    ) -> None:
        self.config = config
        self._environ = environ if environ is not None else os.environ
        self._requester = requester or requests.request
        self.descriptor = ProviderDescriptor(
            provider_id=config.provider_id,
            kind=ProviderKind.EXPERIMENT_EXECUTOR,
            version=config.version,
            input_schemas=(
                EXPERIMENT_EXECUTION_REQUEST_SCHEMA,
                EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA,
            ),
            output_schemas=(
                EXPERIMENT_HTTP_DISPATCH_HANDOFF_SCHEMA,
                EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA,
            ),
            correlation_group="host_configured_http_experiment_transport",
            capabilities=(
                "experiment.dispatch.idempotent",
                "experiment.recovery.reinvoke",
                "experiment.transport.submit",
                "experiment.transport.poll",
                "experiment.transport.cancel",
                *(f"experiment.domain.{domain}" for domain in config.allowed_domains),
            ),
            deterministic=False,
            network_access=True,
            estimated_cost_units=config.estimated_cost_units,
        )
        self.operator_identity = build_experiment_operator_identity(
            principal_id=config.operator_principal_id,
            principal_type="service",
            authentication_context_sha256=config.authentication_context_sha256,
        )

    def invoke(
        self, request: Mapping[str, Any], *, context: ProviderContext
    ) -> ProviderResultEnvelope:
        value = dict(request)
        schema = value.get("schema_version")
        if schema == EXPERIMENT_EXECUTION_REQUEST_SCHEMA:
            return self._handoff(value, context=context)
        if schema == EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA:
            return self._transport(value)
        raise ValueError("http_experiment_request_schema_unsupported")

    def _handoff(
        self, request: Mapping[str, Any], *, context: ProviderContext
    ) -> ProviderResultEnvelope:
        value = dict(request)
        validate_experiment_execution_request(value)
        dispatch_id = str(context.config.get("dispatch_id") or "").strip()
        if not dispatch_id:
            raise ValueError("experiment_dispatch_id_required")
        payload = _with_digest({
            "schema_version": EXPERIMENT_HTTP_DISPATCH_HANDOFF_SCHEMA,
            "dispatch_id": dispatch_id,
            "request_id": value["request_id"],
            "request_sha256": value["content_sha256"],
            "run_id": value["run_id"],
            "route_id": value["route_id"],
            "domain": value["domain"],
            "executor_id": self.descriptor.provider_id,
            "executor_version": self.descriptor.version,
            "state": "awaiting_explicit_external_submission",
            "expected_result_schema": EXPERIMENT_EXECUTION_RESULT_SCHEMA,
            "operation_request_schema": EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA,
            "transport_result_schema": EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA,
            "endpoint_config_sha256": self.config.content_sha256,
            "operator_identity": self.operator_identity,
            "submission_requirements": dict(_HTTP_HANDOFF_REQUIREMENTS),
            "semantics": dict(_HTTP_HANDOFF_SEMANTICS),
        })
        validate_http_experiment_dispatch_handoff(payload, request=value)
        return self._envelope(payload, accepted=True)

    def _transport(self, request: Mapping[str, Any]) -> ProviderResultEnvelope:
        value = dict(request)
        validate_experiment_job_operation_request(value)
        if (
            value["provider_id"] != self.descriptor.provider_id
            or value["provider_version"] != self.descriptor.version
        ):
            raise ValueError("http_experiment_operation_provider_binding_invalid")
        token = str(self._environ.get(self.config.auth_token_env) or "")
        if self.config.auth_token_env and not token:
            return self._failure(
                value, outcome="authentication_unavailable",
                detail_code="configured_bearer_token_unavailable",
            )
        method, url, body = self._request_spec(value)
        headers = {
            "Accept": "application/json",
            "User-Agent": "AutoPlanner-Experiment-Bridge/1.0",
            "Idempotency-Key": value["operation_id"],
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = self._requester(
                method, url, json=body, headers=headers,
                timeout=min(float(value["timeout_s"]), self.config.max_timeout_s),
                stream=True, allow_redirects=False,
            )
        except requests.Timeout:
            return self._failure(value, outcome="timeout", detail_code="request_timeout")
        except requests.RequestException:
            return self._failure(
                value, outcome="transport_error", detail_code="request_transport_error"
            )
        try:
            raw, too_large = _bounded_response_bytes(
                response, limit=self.config.max_response_bytes
            )
        except requests.Timeout:
            return self._failure(
                value, outcome="timeout", detail_code="response_read_timeout"
            )
        except requests.RequestException:
            return self._failure(
                value, outcome="transport_error",
                detail_code="response_read_transport_error",
            )
        response_sha256 = hashlib.sha256(raw).hexdigest() if raw else ""
        status_code = int(getattr(response, "status_code", 0) or 0)
        if too_large:
            return self._failure(
                value, outcome="invalid_response", detail_code="response_too_large",
                http_status=status_code,
            )
        if not 200 <= status_code <= 299:
            return self._failure(
                value, outcome="http_error", detail_code=f"http_status_{status_code}",
                http_status=status_code, response_body_sha256=response_sha256,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._failure(
                value, outcome="invalid_response", detail_code="response_json_invalid",
                http_status=status_code, response_body_sha256=response_sha256,
            )
        parsed = _response_fields(payload)
        if parsed is None or (
            value["operation"] != "submit"
            and parsed["external_job_id"] != value["external_job_id"]
        ):
            return self._failure(
                value, outcome="invalid_response", detail_code="response_contract_invalid",
                http_status=status_code, response_body_sha256=response_sha256,
            )
        result = build_experiment_job_transport_result(
            value, outcome="success",
            endpoint_config_sha256=self.config.content_sha256,
            authentication_context_sha256=self.config.authentication_context_sha256,
            recorded_by=self.operator_identity,
            external_job_id=parsed["external_job_id"],
            provider_sequence=parsed["provider_sequence"], status=parsed["status"],
            status_detail=_sanitized_status_detail(
                parsed["status_detail"], secret=token
            ),
            http_status=status_code,
            response_body_sha256=response_sha256,
        )
        return self._envelope(result, accepted=True)

    def _request_spec(
        self, request: Mapping[str, Any]
    ) -> tuple[str, str, Mapping[str, Any] | None]:
        operation = request["operation"]
        if operation == "submit":
            return "POST", self.config.base_url + self.config.submit_path, {
                "operation_id": request["operation_id"],
                "dispatch_id": request["dispatch_id"],
                "experiment_request": request["execution_request"],
            }
        job_id = quote(str(request["external_job_id"]), safe="")
        template = (
            self.config.poll_path_template
            if operation == "poll"
            else self.config.cancel_path_template
        )
        url = self.config.base_url + template.format(external_job_id=job_id)
        if operation == "poll":
            return "GET", url, None
        return "POST", url, {
            "operation_id": request["operation_id"],
            "dispatch_id": request["dispatch_id"],
            "cancellation_request_sha256": request["cancellation_request_sha256"],
        }

    def _failure(
        self,
        request: Mapping[str, Any],
        *,
        outcome: str,
        detail_code: str,
        http_status: int = 0,
        response_body_sha256: str = "",
    ) -> ProviderResultEnvelope:
        result = build_experiment_job_transport_result(
            request, outcome=outcome,
            endpoint_config_sha256=self.config.content_sha256,
            authentication_context_sha256=self.config.authentication_context_sha256,
            recorded_by=self.operator_identity,
            external_job_id=str(request.get("external_job_id") or ""),
            http_status=http_status, response_body_sha256=response_body_sha256,
            detail_code=detail_code,
        )
        return self._envelope(result, accepted=False)

    def _envelope(
        self, payload: Mapping[str, Any], *, accepted: bool
    ) -> ProviderResultEnvelope:
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema=str(payload["schema_version"]),
            accepted=accepted,
            payload=dict(payload),
            reasons=() if accepted else (str(payload.get("detail_code") or "rejected"),),
        )


def validate_http_experiment_dispatch_handoff(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> None:
    row = dict(value)
    expected = {
        "schema_version", "dispatch_id", "request_id", "request_sha256", "run_id",
        "route_id", "domain", "executor_id", "executor_version", "state",
        "expected_result_schema", "operation_request_schema", "transport_result_schema",
        "endpoint_config_sha256", "operator_identity", "submission_requirements",
        "semantics", "content_sha256",
    }
    bound = dict(request)
    if (
        set(row) != expected
        or row.get("schema_version") != EXPERIMENT_HTTP_DISPATCH_HANDOFF_SCHEMA
        or row.get("state") != "awaiting_explicit_external_submission"
        or row.get("expected_result_schema") != EXPERIMENT_EXECUTION_RESULT_SCHEMA
        or row.get("operation_request_schema") != EXPERIMENT_JOB_OPERATION_REQUEST_SCHEMA
        or row.get("transport_result_schema") != EXPERIMENT_JOB_TRANSPORT_RESULT_SCHEMA
        or row.get("submission_requirements") != _HTTP_HANDOFF_REQUIREMENTS
        or row.get("semantics") != _HTTP_HANDOFF_SEMANTICS
        or any(row.get(key) != bound.get(source) for key, source in (
            ("request_id", "request_id"), ("request_sha256", "content_sha256"),
            ("run_id", "run_id"), ("route_id", "route_id"), ("domain", "domain"),
        ))
        or not _sha256(row.get("endpoint_config_sha256"))
        or not _digest_valid(row)
    ):
        raise ValueError("experiment_http_dispatch_handoff_invalid")
    try:
        validate_experiment_operator_identity(dict(row.get("operator_identity") or {}))
    except ValueError as exc:
        raise ValueError("experiment_http_dispatch_operator_identity_invalid") from exc


def configured_http_experiment_executor(
    environ: Mapping[str, str] | None = None,
) -> HttpExperimentExecutorProvider | None:
    values = environ if environ is not None else os.environ
    base_url = str(values.get("AUTOPLANNER_EXPERIMENT_HTTP_BASE_URL") or "").strip()
    if not base_url:
        return None
    domains = tuple(
        item.strip()
        for item in str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_ALLOWED_DOMAINS")
            or "biocatalytic,execution,mechanism"
        ).split(",")
        if item.strip()
    )
    config = HttpExperimentExecutorConfig(
        provider_id=str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_PROVIDER_ID")
            or "autoplanner.http_experiment_executor"
        ),
        version=str(values.get("AUTOPLANNER_EXPERIMENT_HTTP_VERSION") or "1.0.0"),
        base_url=base_url,
        submit_path=str(values.get("AUTOPLANNER_EXPERIMENT_HTTP_SUBMIT_PATH") or "/jobs"),
        poll_path_template=str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_POLL_PATH")
            or "/jobs/{external_job_id}"
        ),
        cancel_path_template=str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_CANCEL_PATH")
            or "/jobs/{external_job_id}/cancel"
        ),
        auth_token_env=str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_BEARER_TOKEN_ENV") or ""
        ),
        operator_principal_id=str(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_OPERATOR_ID")
            or "service:autoplanner-http-experiment"
        ),
        allowed_domains=domains,
        max_timeout_s=float(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_TIMEOUT_S") or 30.0
        ),
        max_response_bytes=int(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_MAX_RESPONSE_BYTES") or 1_000_000
        ),
        estimated_cost_units=float(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_ESTIMATED_COST_UNITS") or 0.0
        ),
        allow_loopback_http=_truthy(
            values.get("AUTOPLANNER_EXPERIMENT_HTTP_ALLOW_LOOPBACK")
        ),
    )
    return HttpExperimentExecutorProvider(config, environ=values)


def _response_fields(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    row = dict(value)
    if set(row) != {
        "external_job_id", "provider_sequence", "status", "status_detail"
    }:
        return None
    sequence = row.get("provider_sequence")
    if (
        not isinstance(row.get("external_job_id"), str)
        or not row["external_job_id"].strip()
        or row["external_job_id"] != row["external_job_id"].strip()
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or row.get("status") not in {
            "submitted", "running", "completed", "failed", "cancelled"
        }
        or not isinstance(row.get("status_detail"), str)
    ):
        return None
    return row


def _bounded_response_bytes(response: Any, *, limit: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=65_536):
            data = bytes(chunk or b"")
            total += len(data)
            if total > limit:
                return b"".join(chunks), True
            chunks.append(data)
    finally:
        response.close()
    return b"".join(chunks), False


def _sanitized_status_detail(value: str, *, secret: str) -> str:
    detail = "".join(
        character if character >= " " or character in "\t" else " "
        for character in str(value)
    )[:1000]
    if secret:
        detail = detail.replace(secret, "[REDACTED]")
    return detail


def _normalized_base_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("http_experiment_base_url_invalid")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _validate_path(value: str, *, template: bool) -> None:
    path = str(value)
    if not path.startswith("/") or "//" in path or "?" in path or "#" in path:
        raise ValueError("http_experiment_path_invalid")
    count = path.count("{external_job_id}")
    remainder = path.replace("{external_job_id}", "")
    if count != (1 if template else 0) or "{" in remainder or "}" in remainder:
        raise ValueError("http_experiment_path_template_invalid")


def _loopback_host(value: str | None) -> bool:
    return str(value or "").lower() in {"localhost", "127.0.0.1", "::1"}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    observed = row.pop("content_sha256", "")
    return (
        isinstance(observed, str)
        and bool(observed)
        and observed == strict_canonical_json_sha256(row)
    )


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "EXPERIMENT_HTTP_DISPATCH_HANDOFF_SCHEMA",
    "HttpExperimentExecutorConfig",
    "HttpExperimentExecutorProvider",
    "configured_http_experiment_executor",
    "validate_http_experiment_dispatch_handoff",
]
