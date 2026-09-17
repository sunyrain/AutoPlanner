"""Exercise observed-tool containment through actual worker subprocess pipes."""
import json
import os
from pathlib import Path
import time
import tomllib

import pytest

from cascade_planner.agent import codex_worker as worker
from cascade_planner.orchestration.model_call_budget import budget_exposure


def event(tool, call_id="one", *, event_type="item.completed", **extra):
    return json.dumps({"type": event_type, "item": {
        "id": call_id, "type": "mcp_tool_call", "tool": tool, **extra,
    }}, ensure_ascii=False) + "\n"


def test_stream_policy_preserves_mcp_aliases_refusals_and_unicode():
    task = worker.WorkerTask(
        task_id="policy", case_id="case", task_type="target_research",
        required_artifact_type="ResearchReport", allowed_tools=["query_planning_evidence"],
        budget=worker.WorkerBudget(max_tool_calls=1),
    )
    monitor = worker._CodexToolPolicyMonitor(task)
    # A started local command can still be refused by the sandbox. It must
    # not interrupt otherwise valid work before that outcome is known.
    monitor.observe_line(event("shell", "shell", event_type="item.started"))
    monitor.observe_line(event("shell", "shell", status="failed", exit_code=-1,
                               aggregated_output="execution error: sandbox launch rejected"))
    monitor.observe_line(event("mcp__chemistry_inspection__query_planning_evidence",
                               arguments={"query": "line\u2028separator\u2029inside JSON"}))
    monitor.observe_line(event("query_planning_evidence"))  # same ID, no double charge
    assert not monitor.stop_requested.is_set()
    assert len(monitor.calls) == 1
    monitor.observe_line(event("query_planning_evidence", "two"))
    assert monitor.violation["reasons"] == ["tool_call_budget_exceeded"]


def test_strict_worker_exposed_tools_and_developer_scope_match(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    for evidence in (None, {"endpoint": "http://127.0.0.1:1234", "token": "test-only"}):
        config_path = home / "config.toml"
        config_path.write_text("", encoding="utf-8")
        worker._configure_strict_chemistry_worker_environment(
            {"CODEX_HOME": str(home)}, model_workspace=tmp_path / "model",
            audit_root=tmp_path / "audit", enable_local_chemistry_tool=True,
            evidence_transport=evidence,
        )
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        exposed = config["mcp_servers"]["chemistry_inspection"]["enabled_tools"]
        instruction = config["developer_instructions"]
        assert "use only these tools: " + ", ".join(exposed) + "." in instruction
        assert "Do not invoke native web_search" in instruction
        task = worker.WorkerTask(
            task_id="scope", case_id="case", task_type="paper_matched_route_step",
            required_artifact_type="RetrosynthesisProposalReport", allowed_tools=exposed,
        )
        monitor = worker._CodexToolPolicyMonitor(task)
        for i, tool in enumerate(exposed):
            monitor.observe_line(event(tool, str(i)))
        assert not monitor.stop_requested.is_set()
        monitor.observe_line(json.dumps({"type": "item.started", "item": {
            "id": "native", "type": "web_search", "action": {"type": "other"},
        }}))
        assert monitor.violation["reasons"] == ["tool_not_allowed"]


@pytest.mark.parametrize("violate", [False, True])
def test_production_worker_preserves_output_or_stops_before_next_action(
    tmp_path, monkeypatch, violate,
):
    task = worker.WorkerTask(
        task_id="stream-canary", case_id="case", task_type="paper_matched_key_event_critic",
        required_artifact_type="ChemicalStrategyCritique",
        allowed_tools=["inspect_mapped_smiles", "query_planning_evidence"],
        allowed_workdir=str(tmp_path / "audit"),
        budget=worker.WorkerBudget(timeout_s=15, max_tool_calls=3),
    )
    artifact = {
        "checkpoint_match": True, "verdict": "pass", "blocking_type": "none",
        "repair_scope": "none", "required_change_kind": "none", "competing_site_maps": [],
        "reasons": [], "suggested_revision": "",
    }
    escaped = tmp_path / "next-action.txt"
    cli = tmp_path / "fake_codex.py"
    lines = [event("query_planning_evidence")]
    if violate:
        lines.append(json.dumps({"type": "item.started", "item": {
            "id": "ws_native", "type": "web_search", "action": {"type": "other"},
        }}) + "\n")
    cli.write_text(
        "import json, sys, time\nfrom pathlib import Path\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({''.join(lines)!r}); sys.stdout.flush()\n"
        f"time.sleep({6 if violate else 0})\n"
        f"Path({str(escaped)!r}).write_text('next action')\n"
        f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(artifact)!r}, encoding='utf-8')\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':3}}),flush=True)\n",
        encoding="utf-8",
    )

    def isolated_environment(root, workdir, value):
        home = root / "home"
        home.mkdir()
        (home / "config.toml").write_text("", encoding="utf-8")
        return {**os.environ, "CODEX_HOME": str(home)}, {"codex_home": "ephemeral"}

    monkeypatch.setattr(worker, "_codex_executable", lambda: str(cli))
    monkeypatch.setattr(worker, "_codex_cli_runtime_environment", isolated_environment)
    started = time.monotonic()
    record = worker.run_codex_worker(task, use_codex_cli=True, use_api_json=False)
    assert time.monotonic() - started < 5
    log = Path(record.metadata["event_log_path"]).read_text(encoding="utf-8")
    assert "query_planning_evidence" in log
    if violate:
        assert not escaped.exists()
        assert record.status == "provider_error"
        assert worker.worker_provider_failure_reason(record) == "provider_tool_policy_violation"
        assert record.metadata["tool_policy_stop"]["event_type"] == "item.started"
        assert "ws_native" in log
        assert record.output_artifact is None
        assert record.usage == {}
        exposure = budget_exposure([record])
        assert exposure["unknown_input_tokens_held"] > 0
        assert exposure["unknown_output_tokens_held"] > 0
    else:
        assert escaped.exists()
        assert record.status == "accepted_draft", record.output_validation
        assert record.metadata["tool_policy_stop"] == {}
        assert record.usage["input_tokens"] == 12
