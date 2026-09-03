from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time
from typing import Iterator

import pytest

from cascade_planner.application.experiment_execution_contracts import (
    build_experiment_execution_request,
)
from cascade_planner.application.experiment_external_job_operations import (
    build_experiment_job_operation_request,
    build_experiment_job_transport_result,
    validate_experiment_job_operation_request,
    validate_experiment_job_transport_result,
)
from cascade_planner.application.experiment_external_jobs import (
    build_experiment_operator_identity,
)
from cascade_planner.providers.builtins import build_default_provider_registry
from cascade_planner.providers.contracts import ProviderContext
from cascade_planner.providers.experiment import select_experiment_executor
from cascade_planner.providers.http_experiment import (
    HttpExperimentExecutorConfig,
    HttpExperimentExecutorProvider,
    configured_http_experiment_executor,
    validate_http_experiment_dispatch_handoff,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _request() -> dict:
    plan = {
        "schema_version": "fixture_experiment_plan.v1",
        "plan_id": "plan:http-provider",
        "program_id": "program:http-provider",
        "exact_boundary": {
            "input_states": [{
                "state_id": "state:input", "molecule_id": "m:input",
                "canonical_smiles": "CC=O",
            }],
            "output_states": [{
                "state_id": "state:output", "molecule_id": "m:output",
                "canonical_smiles": "CCO",
            }],
        },
        "required_checks": [{
            "check_id": "conversion", "objective": "measure conversion",
            "required": True,
        }],
        "required_output_contract": {
            "schema_version": "biocatalysis_program_validation.v1"
        },
    }
    plan["content_sha256"] = strict_canonical_json_sha256(plan)
    return build_experiment_execution_request(
        run_id="run:http-provider", route_id="route:http-provider",
        work_item_id="work:http-provider", domain="biocatalytic", plan=plan,
        canonical_frontier_sha256="a" * 64,
    )


class _BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        del format, args

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.state["requests"].append({
            "method": "POST", "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "idempotency_key": self.headers.get("Idempotency-Key"),
            "body": json.loads(body or b"{}"),
        })
        if self.server.state.get("delay_s"):
            time.sleep(float(self.server.state["delay_s"]))
        if self.path == "/jobs":
            self._json(201, {
                "external_job_id": "lab-job:1", "provider_sequence": 1,
                "status": "submitted", "status_detail": "queued secret-token",
            })
        elif self.path.endswith("/cancel"):
            self._json(200, {
                "external_job_id": "lab-job:1", "provider_sequence": 3,
                "status": "cancelled", "status_detail": "cancelled by bridge",
            })
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):
        self.server.state["requests"].append({
            "method": "GET", "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "idempotency_key": self.headers.get("Idempotency-Key"),
        })
        self._json(200, {
            "external_job_id": "lab-job:1", "provider_sequence": 2,
            "status": "running", "status_detail": "instrument running",
        })

    def _json(self, status: int, value: dict) -> None:
        data = json.dumps(value).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextmanager
def _bridge(*, delay_s: float = 0.0) -> Iterator[tuple[str, dict]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
    server.state = {"requests": [], "delay_s": delay_s}
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", server.state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _provider(base_url: str, *, token: bool = True) -> HttpExperimentExecutorProvider:
    return HttpExperimentExecutorProvider(
        HttpExperimentExecutorConfig(
            provider_id="fixture.http-experiment",
            version="1.0.0",
            base_url=base_url,
            auth_token_env="FIXTURE_EXPERIMENT_TOKEN" if token else "",
            operator_principal_id="service:fixture-http-bridge",
            allow_loopback_http=True,
        ),
        environ={"FIXTURE_EXPERIMENT_TOKEN": "secret-token"} if token else {},
    )


def _operation(
    request: dict, *, operation: str, attempt: int,
    external_job_id: str = "", receipt_sha256: str = "",
    cancellation_sha256: str = "",
) -> dict:
    return build_experiment_job_operation_request(
        operation=operation, attempt_number=attempt,
        run_id=request["run_id"], dispatch_id="experiment-dispatch:" + "b" * 32,
        task_id="experiment-dispatch-task:" + "b" * 32,
        request_id=request["request_id"], request_sha256=request["content_sha256"],
        provider_id="fixture.http-experiment", provider_version="1.0.0",
        timeout_s=1.0, external_job_id=external_job_id,
        current_external_job_receipt_sha256=receipt_sha256,
        cancellation_request_sha256=cancellation_sha256,
        execution_request=request if operation == "submit" else {},
    )


def test_http_provider_uses_real_loopback_transport_without_persisting_secret() -> None:
    with _bridge() as (base_url, state):
        provider = _provider(base_url)
        registry = build_default_provider_registry(
            include_http_experiment_executor=provider
        )
        request = _request()
        selection = select_experiment_executor(registry, request, {
            "schema_version": "experiment_executor_policy.v1", "enabled": True,
            "allowed_provider_ids": [provider.descriptor.provider_id],
            "preferred_provider_ids": [provider.descriptor.provider_id],
            "allowed_domains": ["biocatalytic"], "allow_network_access": True,
            "max_estimated_cost_units": 0,
        })
        handoff = registry.invoke(
            selection["selected"]["provider_id"], request,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
                config={"dispatch_id": "experiment-dispatch:" + "b" * 32},
            ),
        )
        validate_http_experiment_dispatch_handoff(handoff.payload, request=request)
        assert handoff.payload["state"] == "awaiting_explicit_external_submission"
        assert state["requests"] == []

        submit_request = _operation(request, operation="submit", attempt=1)
        submitted = registry.invoke(
            provider.descriptor.provider_id, submit_request,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
            ),
        )
        validate_experiment_job_transport_result(
            submitted.payload, request=submit_request
        )
        assert submitted.accepted is True
        assert submitted.payload["status"] == "submitted"
        assert submitted.payload["status_detail"] == "queued [REDACTED]"
        assert state["requests"][0]["authorization"] == "Bearer secret-token"
        assert state["requests"][0]["idempotency_key"] == submit_request["operation_id"]
        assert "secret-token" not in json.dumps(submitted.to_dict())

        poll_request = _operation(
            request, operation="poll", attempt=1, external_job_id="lab-job:1",
            receipt_sha256="c" * 64,
        )
        polled = registry.invoke(
            provider.descriptor.provider_id, poll_request,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
            ),
        )
        assert polled.payload["status"] == "running"
        cancel_request = _operation(
            request, operation="cancel", attempt=1, external_job_id="lab-job:1",
            receipt_sha256="d" * 64, cancellation_sha256="e" * 64,
        )
        cancelled = registry.invoke(
            provider.descriptor.provider_id, cancel_request,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
            ),
        )
        assert cancelled.payload["status"] == "cancelled"


def test_http_provider_audits_missing_auth_and_timeout_without_job_success() -> None:
    with _bridge(delay_s=0.2) as (base_url, state):
        request = _request()
        missing_auth = HttpExperimentExecutorProvider(
            HttpExperimentExecutorConfig(
                provider_id="fixture.http-experiment", version="1.0.0",
                base_url=base_url, auth_token_env="MISSING_TOKEN",
                allow_loopback_http=True,
            ),
            environ={},
        )
        operation = _operation(request, operation="submit", attempt=1)
        rejected = missing_auth.invoke(
            operation,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
            ),
        )
        assert rejected.accepted is False
        assert rejected.payload["outcome"] == "authentication_unavailable"
        assert state["requests"] == []

        timeout_provider = _provider(base_url, token=False)
        timeout_operation = dict(operation)
        timeout_operation["timeout_s"] = 0.05
        timeout_operation.pop("content_sha256")
        timeout_operation["content_sha256"] = strict_canonical_json_sha256(
            timeout_operation
        )
        timed_out = timeout_provider.invoke(
            timeout_operation,
            context=ProviderContext(
                run_id=request["run_id"], case_id=request["run_id"],
                target_smiles="CCO",
            ),
        )
        assert timed_out.accepted is False
        assert timed_out.payload["outcome"] == "timeout"


def test_http_provider_rejects_client_controlled_or_insecure_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="requires_https"):
        HttpExperimentExecutorConfig(
            provider_id="fixture.http", version="1", base_url="http://example.com"
        )
    with pytest.raises(ValueError, match="authority_invalid"):
        HttpExperimentExecutorConfig(
            provider_id="fixture.http", version="1",
            base_url="https://user:secret@example.com",
            auth_token_env="TOKEN",
        )


def test_http_provider_is_enabled_only_by_explicit_host_environment() -> None:
    assert configured_http_experiment_executor({}) is None
    provider = configured_http_experiment_executor({
        "AUTOPLANNER_EXPERIMENT_HTTP_BASE_URL": "https://lab.example.test",
        "AUTOPLANNER_EXPERIMENT_HTTP_PROVIDER_ID": "fixture.env-http-experiment",
        "AUTOPLANNER_EXPERIMENT_HTTP_BEARER_TOKEN_ENV": "LAB_TOKEN",
        "AUTOPLANNER_EXPERIMENT_HTTP_OPERATOR_ID": "service:env-lab",
        "LAB_TOKEN": "host-secret",
    })
    assert provider is not None
    assert provider.descriptor.provider_id == "fixture.env-http-experiment"
    assert provider.descriptor.network_access is True
    assert "host-secret" not in json.dumps(provider.config.to_dict())


def test_transport_contracts_reject_fresh_digest_cross_binding_and_operator_drift() -> None:
    request = _request()
    operation = _operation(request, operation="submit", attempt=1)
    tampered = dict(operation)
    tampered["dispatch_id"] = "experiment-dispatch:" + "c" * 32
    tampered.pop("content_sha256")
    tampered["content_sha256"] = strict_canonical_json_sha256(tampered)
    with pytest.raises(ValueError, match="operation_request_invalid"):
        validate_experiment_job_operation_request(tampered)

    operator = build_experiment_operator_identity(
        principal_id="service:transport-contract", principal_type="service",
        authentication_context_sha256="d" * 64,
    )
    result = build_experiment_job_transport_result(
        operation, outcome="success", endpoint_config_sha256="e" * 64,
        authentication_context_sha256="d" * 64, recorded_by=operator,
        external_job_id="job:contract", provider_sequence=1,
        status="submitted", http_status=200, response_body_sha256="f" * 64,
    )
    wrong_operator = build_experiment_operator_identity(
        principal_id="service:transport-contract", principal_type="service",
        authentication_context_sha256="0" * 64,
    )
    tampered_result = dict(result)
    tampered_result["recorded_by"] = wrong_operator
    tampered_result.pop("content_sha256")
    tampered_result["content_sha256"] = strict_canonical_json_sha256(
        tampered_result
    )
    with pytest.raises(ValueError, match="operator_binding_invalid"):
        validate_experiment_job_transport_result(
            tampered_result, request=operation
        )
