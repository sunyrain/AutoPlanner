import json
import subprocess

from cascade_planner.agent import codex_worker as worker
from cascade_planner.agent.worker_usage import (
    WorkerUsageTrace, otel_usage_events, prompt_measurements, request_summary,
)


def _event(name, **attrs):
    return {"body": {"stringValue": name}, "attributes": [
        {"key": k, "value": {"intValue" if isinstance(v, int) else "stringValue": str(v)}}
        for k, v in attrs.items()
    ]}


def _payload(*events):
    return {"resourceLogs": [{"resource": {"attributes": []}, "scopeLogs": [{"logRecords": list(events)}]}]}


def test_request_trace_separates_first_input_from_followup_and_drops_content():
    payload = _payload(
        _event("codex.api_request", attempt=1, status=200),
        _event("codex.sse_event", **{"event.kind": "response.completed", "input_token_count": 100, "cached_token_count": 40, "output_token_count": 10, "reasoning_token_count": 3, "authorization": "SECRET", "prompt": "PRIVATE"}),
        _event("codex.sse_event", **{"event.kind": "response.output_text.delta", "input_token_count": 999}),
        _event("codex.tool_result", output="PRIVATE"),
        _event("codex.sse_event", **{"event.kind": "response.completed", "input_token_count": 160, "cached_token_count": 80, "output_token_count": 5}),
    )
    events = otel_usage_events(payload)
    summary = request_summary(events)
    assert summary["observed_completed_responses"] == 2
    assert summary["observed_api_attempts"] == 1
    assert summary["first_response_input_tokens"] == 100
    assert summary["subsequent_response_input_tokens"] == 160
    assert summary["responses"][1]["reasoning_output_tokens"] is None
    assert "PRIVATE" not in json.dumps(events)
    assert "SECRET" not in json.dumps(events)
    assert request_summary([])["first_response_input_tokens"] is None


def test_local_collector_isolated_task_binding_and_survives_closing(tmp_path):
    with WorkerUsageTrace(tmp_path, task_id="worker:one") as trace:
        argv = ["codex", "exec", "-"]
        env = {"RUST_LOG": "warn"}
        trace.configure(argv, env)
        assert "otel.log_user_prompt=false" in argv
        assert env["RUST_LOG"] == "warn,codex_otel=info"
        plain = 'INFO codex_otel.log_only: event.name="codex.sse_event" event.kind="response.completed"\n'
        measured = plain.rstrip() + ' input_token_count=100 prompt="PRIVATE"\n'
        duplicate = measured.replace("log_only", "trace_safe")
        clean = trace.capture("ordinary warning\n" + plain + measured + duplicate)
        assert clean == "ordinary warning\n"
        assert trace.summary()["observed_completed_responses"] == 1
        assert trace.summary()["completion_notifications_without_usage"] == 1
        assert trace.summary()["first_response_input_tokens"] == 100
        path = trace.path
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(row["task_id"] == "worker:one" for row in saved)
    assert "PRIVATE" not in path.read_text(encoding="utf-8")


def test_prompt_measurements_distinguish_bytes_from_tokens_and_include_schema():
    objective = 'instructions\n{"target":"乙醇","steps":[1,2]}'
    sizes = prompt_measurements(objective, "wrapper\n" + objective, {"type": "object"})
    assert sizes["stdin_prompt"]["utf8_bytes"] > sizes["stdin_prompt"]["characters"]
    assert sizes["objective"]["characters"] == len(objective)
    assert sizes["worker_wrapper"]["characters"] == 8
    assert "steps" in sizes["context_fields_reserialized"]
    assert sizes["output_schema"]["utf8_bytes"] > 0
    assert sizes["provider_system_instructions_and_tool_schemas_observed"] is False


def test_cli_timeout_preserves_diagnostics_without_claiming_zero_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_codex_executable_available", lambda _: True)
    def environment(root, workspace, task):
        home = root / "codex_home"
        home.mkdir()
        (home / "config.toml").write_text("", encoding="utf-8")
        return {"CODEX_HOME": str(home)}, {"model": task.model}
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 1)
    monkeypatch.setattr(worker, "_codex_cli_runtime_environment", environment)
    monkeypatch.setattr(worker, "_run_worker_command", timeout)
    task = worker.WorkerTask(
        task_id="timeout-trace", case_id="test", task_type="paper_matched_strategy_generator",
        required_artifact_type="StrategyCardReport", allowed_workdir=str(tmp_path), objective="short",
    )
    record = worker.run_codex_worker(task, use_codex_cli=True)
    assert record.status == "timeout"
    diagnostics = record.metadata["usage_diagnostics"]
    assert diagnostics["prompt_sizes"]["objective"]["characters"] == 5
    assert diagnostics["uncached_input_tokens"] is None
    assert diagnostics["requests"]["first_response_input_tokens"] is None
    assert record.usage == {}


def test_comparison_keeps_missing_usage_unknown_and_deduplicates_seed_aliases(tmp_path):
    from scripts.report_worker_usage import build_report

    completed = dict(task_id="one", metadata={"model": "example", "task_type": "builder"},
                     usage={"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10},
                     status="accepted_draft")
    timeout = dict(task_id="two", metadata={"model": "example", "task_type": "builder"},
                   usage={}, status="timeout")
    path = tmp_path / "sequential-director-worker-records.jsonl"
    path.write_text("\n".join(json.dumps({"record": r}) for r in (completed, completed, timeout)), encoding="utf-8")
    report = build_report([path])
    assert len(report["calls"]) == 2
    assert report["calls"][1]["input_tokens"] is None
    assert report["calls"][0]["api_attempts_observed"] is None
    assert report["groups"][0]["input_usage_known"] == 1
    assert report["groups"][0]["input_tokens"] == 100
    assert report["groups"][0]["uncached_input_tokens"] == 60


def test_comparison_reads_unicode_separators_inside_jsonl_strings(tmp_path):
    from scripts.report_worker_usage import build_report

    record = dict(task_id="one", metadata={"model": "example"},
                  usage={"input_tokens": 100, "output_tokens": 10},
                  status="accepted_draft", stdout="condition\u2028alternative\u2029detail")
    path = tmp_path / "sequential-director-worker-records.jsonl"
    path.write_text(json.dumps({"record": record}, ensure_ascii=False) + "\n", encoding="utf-8")
    prompt = "target\u2028conditions\u2029description"
    path.with_name("model-io.jsonl").write_text(json.dumps(
        dict(event="model_input", task_id="one", task_type="builder", prompt=prompt),
        ensure_ascii=False) + "\n", encoding="utf-8")
    report = build_report([path])
    assert len(report["calls"]) == 1
    assert report["calls"][0]["objective_chars"] == len(prompt)
    assert report["groups"][0]["task_type"] == "builder"
    assert report["groups"][0]["input_tokens"] == 100
def test_search_policy_reports_observed_mismatch_without_claiming_success():
    from cascade_planner.agent.worker_usage import search_policy_diagnostics

    command = ["codex", "-c", 'web_search="disabled"', "exec"]
    events = [{"tool": "web_search", "action_type": "search", "status": "completed"},
              {"tool": "web_search", "action_type": "open", "status": "failed"},
              {"tool": "inspect_mapped_smiles"}]
    diagnostic = search_policy_diagnostics(command, events, observation_complete=True)
    assert diagnostic["status"] == "requested_disabled_but_observed"
    assert diagnostic["observed_search_events"] == 2
    assert diagnostic["observed_actions"] == {"search": 1, "open": 1}
    assert len(events) == 3
    assert search_policy_diagnostics(command, [], observation_complete=False)["status"] == "observation_incomplete"
