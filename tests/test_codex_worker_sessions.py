"""Verify per-call isolation through the CLI transport without a model provider."""
import json
import os
from pathlib import Path

import pytest

from cascade_planner.agent import codex_worker as w
from cascade_planner.orchestration import sequential_strategy_director as d
from cascade_planner.orchestration.reaction_granularity import STAGE_GUIDANCE


def test_each_role_call_has_a_fresh_home_and_complete_instructions(tmp_path, monkeypatch):
    cli = tmp_path / "fake_cli.py"
    log = tmp_path / "calls.jsonl"
    cli.write_text('''
import json, os, sys, uuid
from pathlib import Path
args = sys.argv[1:]
home = Path(os.environ["CODEX_HOME"])
assert not (home / "previous_call").exists()
(home / "previous_call").touch()
schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
prompt = sys.stdin.read()
with Path(os.environ["TEST_CALL_LOG"]).open("a", encoding="utf-8") as f:
    f.write(json.dumps({"home": str(home), "args": args, "prompt": prompt}) + "\\n")
if "strategy_cards" in schema["properties"]:
    output = {"strategy_cards": [{"strategy_query":"reduce ketone", "critical_assumption":"selectivity", "critic_checkpoint":"carbonyl reduction"}]}
else:
    output = {"checkpoint_match":True,"verdict":"uncertain","uncertainty_source":"evidence_missing","blocking_type":"stereochemistry","repair_scope":"none","required_change_kind":"none","competing_site_maps":[],"reasons":["selectivity unverified"],"suggested_revision":"retain screening hypothesis"}
Path(args[args.index("--output-last-message") + 1]).write_text(json.dumps(output))
print(json.dumps({"type":"thread.started","thread_id":str(uuid.uuid4())}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":10}}))
''', encoding="utf-8")
    monkeypatch.setattr(w, "_codex_executable", lambda: str(cli))

    def environment(root, workspace, task):
        home = root / "codex_home"
        home.mkdir()
        (home / "config.toml").write_text("", encoding="utf-8")
        return {**os.environ, "CODEX_HOME": str(home), "TEST_CALL_LOG": str(log)}, {"model": "test"}

    monkeypatch.setattr(w, "_codex_cli_runtime_environment", environment)
    results = []
    for i, (kind, artifact) in enumerate((
        ("paper_matched_strategy_generator", "StrategyPortfolioReport"),
        ("paper_matched_key_event_critic", "ChemicalStrategyCritique"),
        ("paper_matched_strategy_generator", "StrategyPortfolioReport"),
    )):
        task = w.WorkerTask(
            task_id=f"task-{i}", case_id="case", task_type=kind,
            required_artifact_type=artifact, allowed_workdir=str(tmp_path / "audit"),
            objective=STAGE_GUIDANCE + '\nContext:\n{"current_map":' + str(i) + '}',
            host_context={"target_smiles": "CCO", "focus_step_id": "focus"},
            budget=w.WorkerBudget(timeout_s=10),
        )
        result = w.run_codex_worker(task, use_codex_cli=True, use_api_json=False)
        assert result.status == "accepted_draft", result.to_dict()
        assert result.usage["input_tokens"] == 100
        results.append(result)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({r.metadata["session_id"] for r in results}) == 3
    assert len({row["home"] for row in rows}) == 3
    for i, row in enumerate(rows):
        assert "--ephemeral" in row["args"] and "resume" not in row["args"]
        assert STAGE_GUIDANCE in row["prompt"]
        assert json.dumps({"current_map": i}, separators=(",", ":")) in row["prompt"]
        assert not Path(row["home"]).exists()


@pytest.mark.parametrize("mode,independent", [("shared", False), ("shared_experiment", False),
                                             ("isolated", True), (None, True)])
def test_critic_independence_preserves_historical_session_metadata(mode, independent):
    record = w.WorkerRunRecord(
        run_id="r", task_id="t", case_id="c", status="accepted_draft",
        metadata={"session_mode": mode} if mode else {},
        output_artifact={"artifact_type": "ChemicalStrategyCritique", "payload": {
            "schema_version": "chemical_strategy_critique.v1", "overall_assessment": "uncertain",
        }},
    )
    critique = d._critique_from_record(record)
    assert critique["status"] == "uncertain"
    assert critique["semantics"]["independent_codex_critic"] is independent
