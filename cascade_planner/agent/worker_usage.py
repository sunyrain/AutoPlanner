"""Local, non-authoritative measurements of worker requests and context sizes.

Codex OTel log events supply per-response usage which exec --json cannot split.
Only allowlisted scalar attributes are retained; prompts, credentials, tool
contents and provider request bodies are never stored by the collector.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Mapping
import uuid


def text_size(value: str) -> dict[str, int]:
    return {"characters": len(value), "utf8_bytes": len(value.encode("utf-8"))}


def prompt_measurements(objective: str, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Measure exact locally authored text, without pretending bytes are tokens."""
    components = {}
    # The chemistry prompts append one JSON context. raw_decode avoids
    # matching an instruction's braces unless it really ends in that object.
    for position, char in enumerate(objective):
        if char != "{":
            continue
        try:
            payload, end = json.JSONDecoder().raw_decode(objective[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and not objective[position + end:].strip():
            components = {k: text_size(json.dumps(v, ensure_ascii=False)) for k, v in payload.items()}
            break
    return {
        "objective": text_size(objective),
        "stdin_prompt": text_size(prompt),
        "worker_wrapper": text_size(prompt.replace(objective, "", 1)) if objective in prompt else None,
        "output_schema": text_size(json.dumps(schema, indent=2)),
        "context_fields_reserialized": components,
        "context_field_sizes_are_reserialized_not_additive": True,
        "local_sizes_are_not_provider_token_counts": True,
        "provider_system_instructions_and_tool_schemas_observed": False,
    }


_EVENTS = {"codex.conversation_starts", "codex.api_request", "codex.sse_event", "codex.websocket_event", "codex.websocket_request"}
_ATTRIBUTES = {
    "event.name", "event_name", "event.kind", "kind", "event_type", "model", "slug",
    "app.version", "service.version", "reasoning_effort", "model_reasoning_effort",
    "conversation.id", "conversation_id", "response_id", "response.id", "request_id",
    "input_token_count", "output_token_count", "cached_token_count", "reasoning_token_count",
    "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens",
    "total_tokens", "duration_ms", "duration", "attempt", "status", "status_code",
    "http.status_code", "http.response.status_code", "success", "wire_api", "transport",
    "event.timestamp",
}


def _scalar(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    for key in ("stringValue", "intValue", "doubleValue", "boolValue", "string_value", "int_value", "double_value", "bool_value"):
        if key in value:
            raw = value[key]
            if key in {"intValue", "int_value"}:
                try:
                    return int(raw)
                except (ValueError, TypeError):
                    return None
            return raw if isinstance(raw, (str, int, float, bool)) else None
    return None


def _attributes(rows: Any) -> dict[str, Any]:
    return {
        str(row.get("key")): _scalar(row.get("value"))
        for row in rows or []
        if isinstance(row, dict) and str(row.get("key")) in _ATTRIBUTES
    }


def otel_usage_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for resource in payload.get("resourceLogs", payload.get("resource_logs", [])):
        common = _attributes(resource.get("resource", {}).get("attributes", []))
        for scope in resource.get("scopeLogs", resource.get("scope_logs", [])):
            for record in scope.get("logRecords", scope.get("log_records", [])):
                attrs = {**common, **_attributes(record.get("attributes"))}
                name = record.get("eventName", record.get("event_name")) or attrs.get("event.name") or attrs.get("event_name")
                if not name:
                    name = _scalar(record.get("body"))
                if name not in _EVENTS:
                    continue
                # SSE token information is useful only at response completion.
                if name in {"codex.sse_event", "codex.websocket_event"} and not any(
                    attrs.get(key) == "response.completed" for key in ("event.kind", "kind", "event_type")
                ):
                    continue
                result.append({
                    "event": name,
                    "timestamp_unix_nano": record.get("timeUnixNano", record.get("time_unix_nano")),
                    "attributes": attrs,
                })
    return result


def request_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    completions = [r for r in events if r["event"] in {"codex.sse_event", "codex.websocket_event"}]
    # CLI 0.145 emits a transport completion and a separate usage-bearing
    # completion for the same response. Only the latter is a usage record.
    responses = [r for r in completions if any(
        k in r["attributes"] for k in (
            "input_tokens", "input_token_count", "output_tokens", "output_token_count",
        )
    )]
    requests = [r for r in events if r["event"] in {"codex.api_request", "codex.websocket_request"}]
    counts = []
    for event in responses:
        attrs = event["attributes"]
        usage = {}
        for key, aliases in {
            "input_tokens": ("input_tokens", "input_token_count"),
            "cached_input_tokens": ("cached_input_tokens", "cached_token_count"),
            "output_tokens": ("output_tokens", "output_token_count"),
            "reasoning_output_tokens": ("reasoning_output_tokens", "reasoning_token_count"),
        }.items():
            value = next((attrs[x] for x in aliases if attrs.get(x) is not None), None)
            try:
                usage[key] = max(0, int(value)) if value is not None else None
            except (ValueError, TypeError):
                usage[key] = None
        counts.append(usage)
    return {
        "observed_api_attempts": len(requests),
        "observed_completed_responses": len(responses),
        "completion_notifications_without_usage": len(completions) - len(responses),
        "responses": counts,
        "first_response_input_tokens": counts[0]["input_tokens"] if counts else None,
        "subsequent_response_input_tokens": (
            sum(r["input_tokens"] for r in counts[1:])
            if counts and all(r["input_tokens"] is not None for r in counts) else None
        ),
        "telemetry_observed": bool(events),
        "cli_versions": sorted({str(r["attributes"]["app.version"]) for r in events if r["attributes"].get("app.version")}),
        "models_reported": sorted({str(r["attributes"]["model"]) for r in events if r["attributes"].get("model")}),
        "telemetry_is_not_budget_authority": True,
    }


class WorkerUsageTrace:
    """Capture allowlisted Codex OTel events from stderr in one local file.

    This uses the CLI's log emitter instead of a network exporter. Restrict the
    Rust log filter to codex_otel and strip telemetry text before the ordinary
    stderr journal is saved, so tool output snippets and prompts are not copied.
    """

    def __init__(self, audit_root: Path, *, task_id: str, enabled: bool = True):
        self.path = audit_root / "codex_worker_usage" / f"{uuid.uuid4().hex}.jsonl"
        self.task_id = task_id
        self.enabled = enabled
        self.events: list[dict[str, Any]] = []
        self.error = ""

    def __enter__(self):
        return self

    def configure(self, command: list[str], env: dict[str, str]) -> None:
        if not self.enabled:
            return
        index = command.index("exec")
        command[index:index] = ["-c", "otel.log_user_prompt=false"]
        env["RUST_LOG"] = str(env.get("RUST_LOG") or "warn") + ",codex_otel=info"

    def capture(self, stderr: str) -> str:
        if not self.enabled:
            return stderr
        lines = stderr.splitlines(keepends=True)
        # CLI versions can emit both copies. Select one stream, never sum both.
        target = "codex_otel.log_only:" if any("codex_otel.log_only:" in s for s in lines) else "codex_otel.trace_safe:"
        records = []
        for line in lines:
            if target not in line:
                continue
            attrs = []
            for match in re.finditer(r'([\w.]+)=("(?:\\.|[^"\\])*"|[^\s]+)', line):
                key, value = match.groups()
                if key not in _ATTRIBUTES:
                    continue
                try:
                    parsed = json.loads(value)
                except ValueError:
                    parsed = value
                scalar = {"intValue": str(parsed)} if isinstance(parsed, int) and not isinstance(parsed, bool) else {"stringValue": str(parsed)}
                attrs.append({"key": key, "value": scalar})
            records.append({"attributes": attrs})
        normalized = otel_usage_events({"resourceLogs": [{"scopeLogs": [{"logRecords": records}]}]})
        self.events.extend(normalized)
        if normalized:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    for row in normalized:
                        handle.write(json.dumps({"task_id": self.task_id, **row}, ensure_ascii=False) + "\n")
            except OSError as exc:
                self.error = type(exc).__name__
        return "".join(line for line in lines if "codex_otel." not in line)

    def summary(self) -> dict[str, Any]:
        return {
            "event_log_path": str(self.path) if self.path.exists() else "",
            "enabled": self.enabled, "collector_error": self.error,
            "source": "codex_otel_stderr",
            **request_summary(list(self.events)),
        }

    def __exit__(self, *args):
        pass


def search_policy_diagnostics(command: list[str], tool_calls: list[dict[str, Any]], *, observation_complete: bool) -> dict[str, Any]:
    """Compare the requested CLI policy to events; do not infer search success."""
    requested = "live" if "--search" in command else "unspecified"
    for index, arg in enumerate(command):
        if index and command[index - 1] in {"-c", "--config"}:
            key, separator, value = str(arg).partition("=")
            if separator and key.strip() == "web_search":
                requested = value.strip().strip("\"'")
    observed = [row for row in tool_calls if str(row.get("tool") or row.get("name") or "") in {"web_search", "web_search_call"}]
    mismatch = requested == "disabled" and bool(observed)
    return {
        "requested_mode": requested,
        "observed_search_events": len(observed),
        "observed_actions": dict(Counter(str(row.get("action_type") or "unknown") for row in observed)),
        "observation_complete": observation_complete,
        "status": "requested_disabled_but_observed" if mismatch else (
            "no_observed_mismatch" if observation_complete else "observation_incomplete"
        ),
    }


def worker_usage_diagnostics(*, measurements: Mapping[str, Any], usage: Mapping[str, Any], trace: Mapping[str, Any], tool_calls: list[dict[str, Any]], stderr: str) -> dict[str, Any]:
    observed_input = usage.get("input_tokens")
    cached = usage.get("cached_input_tokens")
    responses = list(trace.get("responses") or [])
    measured_sum = (
        sum(r["input_tokens"] for r in responses)
        if responses and all(r.get("input_tokens") is not None for r in responses) else None
    )
    return {
        "schema_version": "worker_usage_diagnostics.v1",
        "prompt_sizes": dict(measurements),
        "observed_usage": dict(usage),
        "uncached_input_tokens": (
            max(0, int(observed_input) - int(cached)) if observed_input is not None and cached is not None else None
        ),
        "requests": dict(trace),
        "response_input_sum": measured_sum,
        "response_input_sum_matches_turn_usage": (
            measured_sum == int(observed_input)
            if measured_sum is not None and observed_input is not None else None
        ),
        "tool_counts": dict(Counter(str(r.get("tool") or "unknown") for r in tool_calls)),
        "fallback_model_metadata": "fallback model metadata" in stderr or "Defaulting to fallback metadata" in stderr,
        "unobserved_context": ["provider_system_instructions", "provider_tool_schemas", "server_added_context"],
    }
