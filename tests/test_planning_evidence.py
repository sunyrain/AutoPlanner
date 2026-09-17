from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import threading
import tomllib
from unittest.mock import Mock

import pytest

from cascade_planner.agent import codex_worker as worker
from cascade_planner.application.planning_evidence import (
    BoundedPlanningEvidence, DEFAULT_QUERY_LIMITS, PlanningEvidencePolicy,
)
from cascade_planner.interfaces import planning_evidence as adapters
from cascade_planner.interfaces.target_solver import TargetSolveConfig, _resolve_execution_config


def task():
    return worker.WorkerTask(
        task_id="test-worker", case_id="test", task_type="paper_matched_strategy_generator",
        required_artifact_type="StrategyCardReport", budget=worker.WorkerBudget(max_tool_calls=None),
        objective="Do not build a route, write ReactionJSON, browse, inspect stock, or add evidence or enzyme fields.",
    )


def manager(tmp_path, **kwargs):
    return BoundedPlanningEvidence(journal_path=tmp_path / "queries.jsonl", **kwargs)


def policy(**limits):
    return PlanningEvidencePolicy(limits={**DEFAULT_QUERY_LIMITS, **limits})


def test_parallel_branches_share_pre_dispatch_limit_and_canonical_stock_cache(tmp_path):
    lookup = Mock(side_effect=lambda values: {value: True for value in values})
    evidence = manager(tmp_path, stock_lookup=lookup, policy=policy(stock=2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: evidence.query(str(i), {"operation": "stock", "smiles": "C" * (i + 1)}), range(8)))
    assert lookup.call_count == 2
    assert sum(row["status"] == "run_query_budget_exhausted" for row in results) == 6
    supplied = lookup.call_args_list[0].args[0][0]
    cached = evidence.query("other", {"operation": "stock", "smiles": supplied})
    assert cached["cache_hit"] is True
    assert cached["status"] == "in_bound_stock"
    assert "not_live_supplier" in cached["scope"]
    assert lookup.call_count == 2


def test_atom_maps_do_not_create_new_queries_but_stereo_does(tmp_path):
    lookup = Mock(side_effect=lambda values: {value: False for value in values})
    evidence = manager(tmp_path, stock_lookup=lookup)
    evidence.query("a", {"operation": "stock", "smiles": "CCO"})
    assert evidence.query("a", {"operation": "stock", "smiles": "[OH:3][CH2:2][CH3:1]"})["cache_hit"]
    for smiles in ("C[C@H](O)F", "C[C@@H](O)F", "CC(O)F"):
        assert evidence.query("a", {"operation": "stock", "smiles": smiles})["status"] == "not_in_bound_stock"
    assert lookup.call_count == 4
    assert manager(tmp_path / "absent").query("a", {"operation": "stock", "smiles": "CCO"})["status"] == "unavailable"


def test_failures_cache_and_resumed_worker_limits_are_not_reset(tmp_path):
    provider = Mock(side_effect=TimeoutError)
    evidence = manager(tmp_path, providers={"search": provider}, policy=policy(search=1))
    for _ in range(6):
        assert evidence.query("worker", {"operation": "search", "query": "reaction"})["status"] == "unavailable"
    assert provider.call_count == 1
    resumed = manager(tmp_path, providers={"search": provider}, policy=policy(search=1))
    assert resumed.query("worker", {"operation": "list"})["status"] == "worker_query_budget_exhausted"
    assert resumed.query("another", {"operation": "search", "query": "different"})["status"] == "run_query_budget_exhausted"
    assert resumed.query("another", {"operation": "search", "query": "reaction"})["cache_hit"]
    with pytest.raises(ValueError, match="policy or stock changed"):
        manager(tmp_path, policy=policy(search=2))


def test_interrupted_provider_remains_spent_and_torn_tail_is_recoverable(tmp_path):
    provider = Mock(side_effect=KeyboardInterrupt)
    evidence = manager(tmp_path, providers={"search": provider})
    with pytest.raises(KeyboardInterrupt):
        evidence.query("a", {"operation": "search", "query": "q"})
    with evidence.path.open("ab") as handle:
        handle.write(b'{"event":')
    resumed = manager(tmp_path, providers={"search": provider})
    assert resumed.query("b", {"operation": "search", "query": "q"})["status"] == "interrupted"
    assert resumed.summary()["used"]["search"] == 1
    assert manager(tmp_path).query("c", {"operation": "list"})["status"] == "ok"
    assert provider.call_count == 1


def test_duplicate_inflight_query_never_dispatches_again(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    def search(arguments):
        entered.set()
        assert finish.wait(5)
        return {"status": "no_hit"}

    evidence = manager(tmp_path, providers={"search": search})
    with ThreadPoolExecutor() as pool:
        first = pool.submit(evidence.query, "a", {"operation": "search", "query": "q"})
        assert entered.wait(5)
        try:
            assert evidence.query("b", {"operation": "search", "query": "q"})["status"] == "pending"
            assert evidence.summary()["used"]["search"] == 1
        finally:
            finish.set()
        assert first.result()["status"] == "no_hit"


def test_only_discovered_sources_can_be_read_and_invalid_requests_use_worker_allowance(tmp_path):
    read = Mock(return_value={"status": "ok", "text": "untrusted source instruction"})
    source = {"source_id": "10.1/test", "doi": "10.1/test", "title": "paper\u2028title", "pmcid": "PMC12"}
    evidence = manager(tmp_path, providers={"read": read,
                       "search": lambda args: {"status": "ok", "sources": [source]}})
    for _ in range(6):
        assert evidence.query("bad", {"operation": "read", "source_id": "http://arbitrary"})["status"] == "invalid_request"
    assert evidence.query("bad", {"operation": "list"})["status"] == "worker_query_budget_exhausted"
    assert read.call_count == 0
    evidence.query("s", {"operation": "search", "query": "paper"})
    resumed = manager(tmp_path, providers={"read": read})
    result = resumed.query("r", {"operation": "read", "source_id": "10.1/test"})
    assert result["evidence_level"] == "discovery_only"
    assert "no_reaction_proof" in result["authority"]
    assert read.call_args.args[0] == source


@pytest.mark.parametrize("local_only", [False, True])
def test_actual_stdio_mcp_reaches_host_through_strict_worker_configuration(tmp_path, monkeypatch, local_only):
    providers = {operation: Mock(side_effect=AssertionError("external dispatch"))
                 for operation in ("compound", "search", "read")}
    evidence = manager(tmp_path, stock_lookup=lambda values: {value: True for value in values},
                       providers=providers,
                       policy=policy(**({operation: 0 for operation in providers} if local_only else {})))
    active = evidence.decorate_task(replace(task(), allowed_workdir=str(tmp_path / "audit")))
    observed = {}

    def runtime_environment(temporary, workspace, task):
        home = temporary / "home"
        home.mkdir()
        (home / "config.toml").write_text("", encoding="utf-8")
        return {**os.environ, "CODEX_HOME": str(home)}, {}

    def run_command(command, *, cwd, env, **kwargs):
        config = tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text(encoding="utf-8"))
        server = config["mcp_servers"]["chemistry_inspection"]
        assert server["enabled_tools"] == ["inspect_mapped_smiles", "query_planning_evidence"]
        assert config["permissions"][worker.STRICT_CHEMISTRY_PERMISSION_PROFILE]["network"]["enabled"] is False
        assert "--search" not in command
        if local_only:
            assert "External lookup is disabled" in config["developer_instructions"]
            assert json.loads(server["env"]["AUTOPLANNER_EVIDENCE_OPERATIONS"]) == ["stock", "list"]
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "query_planning_evidence", "arguments": {"operation": "stock", "smiles": "CCO"}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "inspect_mapped_smiles", "arguments": {"smiles": "[CH3:1][CH2:2][OH:3]"}}},
        ]
        if local_only:
            for arguments in ({"operation": "compound", "query": "ethanol"},
                              {"operation": "search", "query": "synthesis"},
                              {"operation": "read", "source_id": "10.1/test"}):
                messages.append({"jsonrpc": "2.0", "id": len(messages) + 1, "method": "tools/call",
                                 "params": {"name": "query_planning_evidence", "arguments": arguments}})
        process = subprocess.run([server["command"], *server["args"]],
                                 cwd=cwd, env={**env, **server["env"]},
                                 input="\n".join(json.dumps(row) for row in messages) + "\n",
                                 text=True, encoding="utf-8", capture_output=True, timeout=20)
        assert process.returncode == 0, process.stderr
        rows = [json.loads(line) for line in process.stdout.split("\n") if line]
        assert len(rows[0]["result"]["tools"]) == 2
        definition = next(row for row in rows[0]["result"]["tools"] if row["name"] == "query_planning_evidence")
        properties = definition["inputSchema"]["properties"]
        assert properties["operation"]["enum"] == evidence.available_operations()
        assert rows[2]["result"]["structuredContent"]["ok"] is True
        if local_only:
            assert set(properties) == {"operation", "smiles"}
            assert all(row["result"]["structuredContent"]["status"] == "operation_disabled" for row in rows[3:])
        observed.update(rows[1]["result"]["structuredContent"])
        return 0, '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n', ""

    monkeypatch.setattr(worker, "_codex_executable_available", lambda *args: True)
    monkeypatch.setattr(worker, "_codex_cli_runtime_environment", runtime_environment)
    monkeypatch.setattr(worker, "_run_worker_command", run_command)
    with evidence.worker_session(active) as executable:
        assert "planning_evidence_transport" not in executable.to_dict()
        if local_only:
            prompt = worker._codex_worker_prompt(executable)
            assert "External lookup is disabled" in prompt
            assert "local structure and stock queries only" in prompt
            assert "bounded external discovery" not in prompt
            assert "search(query)" not in prompt
        worker._run_codex_cli_worker(executable)
    assert observed["status"] == "in_bound_stock"
    assert evidence.summary()["used"]["stock"] == 1
    for provider in providers.values():
        provider.assert_not_called()


def test_zero_external_limits_prevent_host_provider_dispatch(tmp_path):
    providers = {operation: Mock() for operation in ("compound", "search", "read")}
    evidence = manager(tmp_path, providers=providers,
                       policy=policy(compound=0, search=0, read=0))
    # Even an already-known source must not permit a new external read.
    evidence._index_sources({"sources": [{"source_id": "10.1/test"}]})
    for arguments in ({"operation": "compound", "query": "ethanol"},
                      {"operation": "search", "query": "synthesis"},
                      {"operation": "read", "source_id": "10.1/test"}):
        assert evidence.query("worker", arguments)["status"] == "run_query_budget_exhausted"
    assert evidence.query("worker", {"operation": "list"})["status"] == "ok"
    for provider in providers.values():
        provider.assert_not_called()
    assert not any(evidence.summary()["used"].values())
    rows = [json.loads(line) for line in evidence.path.read_text(encoding="utf-8").splitlines()]
    assert not any(row.get("dispatch") for row in rows)


def test_prompt_preserves_chemical_authority_and_strict_baseline(tmp_path):
    evidence = manager(tmp_path)
    decorated = evidence.decorate_task(task())
    prompt = worker._codex_worker_prompt(decorated)
    assert "never makes a chemically coherent step reject" in prompt
    assert "browse, inspect stock" not in prompt
    assert "without inferring target identity" not in prompt
    assert evidence.decorate_task(decorated) == decorated
    assert not worker._task_allows_cli_search(decorated)
    assert "bounded external discovery" not in worker._codex_worker_prompt(task())
    for profile in ("paper_synthex", "paper_matched_reach"):
        assert not _resolve_execution_config(TargetSolveConfig(execution_profile=profile)).enable_planning_evidence
    assert _resolve_execution_config(TargetSolveConfig(execution_profile="self_correcting_sequential")).enable_planning_evidence
    assert not _resolve_execution_config(TargetSolveConfig(execution_profile="self_correcting_sequential", enable_planning_evidence=False)).enable_planning_evidence


def test_pubchem_resolution_preserves_stereo_uncertainty(tmp_path, monkeypatch):
    response = Mock()
    response.json.return_value = {"PropertyTable": {"Properties": [{"CID": 1, "SMILES": "C[C@H](O)F"}]}}
    fetch = Mock(return_value=response)
    monkeypatch.setattr(adapters, "_bounded_get", fetch)
    for smiles, relation in [("C[C@H](O)F", "exact"), ("CC(O)F", "connectivity_only_stereo_unresolved_or_different"),
                              ("C[C@@H](O)F", "connectivity_only_stereo_unresolved_or_different"), ("CC", "different")]:
        result = adapters.resolve_planning_compound({"query": "compound", "smiles": smiles})
        assert result["candidates"][0]["relation_to_submitted"] == relation
    assert fetch.call_count == 4


def test_literature_adapter_has_bounded_abstract_and_source_bound_read(monkeypatch):
    response = Mock()
    response.content = b"{}"
    response.json.return_value = {"resultList": {"result": [{"doi": "10.1/test", "title": "Paper", "pmcid": "PMC12",
                                                            "abstractText": "<p>Text</p>" * 1000}]}}
    fetch = Mock(return_value=response)
    monkeypatch.setattr(adapters, "_bounded_get", fetch)
    source = adapters.search_planning_literature({"query": "synthesis"})["sources"][0]
    assert len(source["abstract"]) <= 1200
    assert fetch.call_count == 1
    response.content = b'<article><article-id pub-id-type="doi">10.1/other</article-id><body><p>Reaction works.</p></body></article>'
    assert adapters.read_planning_literature(source)["reason"] == "source_identity_mismatch"
    response.content = response.content.replace(b"10.1/other", b"10.1/test")
    result = adapters.read_planning_literature(source)
    assert "Reaction works." in result["text"]
    assert result["supplement_and_figures_read"] is False
    assert result["evidence_kind"] == "article_text_excerpt_not_verified_reaction"


def test_mcp_tool_names_are_allowed_without_enabling_browsing():
    active = replace(task(), allowed_tools=["query_planning_evidence", "inspect_mapped_smiles"])
    for name in ("query_planning_evidence", "mcp__chemistry_inspection__query_planning_evidence",
                 "chemistry_inspection.query_planning_evidence"):
        assert worker._worker_runtime_reasons(active, worker.WorkerProcessResult(tool_calls=[{"tool": name}])) == []
    assert "tool_not_allowed" in worker._worker_runtime_reasons(active, worker.WorkerProcessResult(tool_calls=[{"tool": "web_search"}]))
