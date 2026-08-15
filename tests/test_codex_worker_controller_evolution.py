import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cascade_planner.agent.case_trace import (
    ArtifactRecord,
    CaseBundle,
    FailureEvent,
    RouteStatus,
)
from cascade_planner.agent.codex_controller import (
    ControllerAction,
    ControllerBudget,
    ControllerTrace,
    decide_next_action,
    execute_action,
    observe_case,
    run_controller_loop,
    run_controller_once,
    update_blackboard,
    validate_controller_action,
)
from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerProcessResult,
    WorkerRunRecord,
    WorkerTask,
    WorkerCancelledError,
    WorkerTimeoutError,
    _api_json_config,
    _codex_cli_runtime_environment,
    _codex_cli_worker_environment,
    _codex_cli_command,
    _codex_worker_prompt,
    _run_worker_command,
    _worker_output_json_schema,
    _task_allows_cli_search,
    _task_reasoning_effort,
    run_codex_worker,
)
from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
)


class CodexWorkerControllerEvolutionTest(unittest.TestCase):
    def test_strategy_worker_schema_does_not_require_evidence_metadata(self):
        task = WorkerTask(
            task_id="strategy",
            case_id="opaque-case",
            task_type="strategic_disconnection_mining",
            required_artifact_type="RetrosynthesisProposalReport",
            input_refs=[],
            allowed_tools=[],
            budget=WorkerBudget(max_tool_calls=0),
        )

        schema = _worker_output_json_schema(task)
        candidate = schema["properties"]["payload"]["properties"]["candidates"][
            "items"
        ]
        prompt = _codex_worker_prompt(task)

        self.assertFalse(
            {
                "source_channel",
                "source_refs",
                "evidence_refs",
                "evidence_level",
                "confidence",
            }
            & set(candidate["required"])
        )
        self.assertIn("blind strategy design", prompt)
        self.assertNotIn("Prefer traceable sources", prompt)

    def test_chemical_strategy_critic_has_closed_no_authority_schema(self):
        task = WorkerTask(
            task_id="critic",
            case_id="opaque-critic",
            task_type="route_chemistry_critique",
            required_artifact_type="ChemicalStrategyCritique",
            input_refs=[],
            allowed_tools=[],
            budget=WorkerBudget(max_tool_calls=0),
        )

        payload = _worker_output_json_schema(task)["properties"]["payload"]

        self.assertFalse(payload["additionalProperties"])
        self.assertEqual(
            payload["properties"]["no_reaction_proof"]["enum"], [True]
        )
        self.assertEqual(
            payload["properties"]["no_source_authority"]["enum"], [True]
        )

    def test_global_plan_schema_binds_full_context_ref_not_prompt_digest(self):
        task = WorkerTask(
            task_id="director",
            case_id="case",
            task_type="global_campaign_direction",
            required_artifact_type="GlobalCampaignPlan",
            input_refs=["a" * 64],
        )

        schema = _worker_output_json_schema(task)

        context_schema = schema["properties"]["payload"]["properties"][
            "context_sha256"
        ]
        self.assertEqual(context_schema["enum"], ["a" * 64])

    def test_mock_worker_returns_valid_draft_artifact(self):
        task = WorkerTask(
            task_id="target_research_1",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            allowed_tools=["local_search"],
            dry_run=True,
        )

        record = run_codex_worker(task)

        self.assertEqual(record.status, "accepted_draft")
        self.assertTrue(record.output_validation["accepted"], record.output_validation)
        self.assertEqual(record.output_artifact["artifact_type"], "ResearchReport")
        self.assertEqual(record.output_artifact["validation_status"], "draft")

    def test_worker_rejects_invalid_json_timeout_budget_and_unsafe_output(self):
        task = WorkerTask(
            task_id="bad_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            budget=WorkerBudget(max_tool_calls=1),
        )

        bad_json = run_codex_worker(task, runner=lambda _: WorkerProcessResult(stdout="{not-json", exit_code=0))
        timeout = run_codex_worker(task, runner=lambda _: (_ for _ in ()).throw(TimeoutError("worker timed out")))
        unsafe = run_codex_worker(
            task,
            mock_output={
                "schema_version": "research_report.draft.v1",
                "artifact_id": "bad",
                "artifact_type": "ResearchReport",
                "case_id": "case",
                "source": "codex_worker_mock",
                "input_refs": ["target_profile"],
                "evidence_refs": [],
                "validation_status": "draft",
                "route_status": "solved",
                "rxn_smiles": "CCO>>CC=O",
            },
        )
        tool_budget = run_codex_worker(
            task,
            runner=lambda _: WorkerProcessResult(
                stdout=json.dumps({
                    "schema_version": "research_report.draft.v1",
                    "artifact_id": "ok",
                    "artifact_type": "ResearchReport",
                    "case_id": "case",
                    "source": "codex_worker_mock",
                    "input_refs": ["target_profile"],
                    "evidence_refs": [],
                    "validation_status": "draft",
                }),
                exit_code=0,
                tool_calls=[{"tool": "local_search"}, {"tool": "web_search"}],
            ),
        )

        self.assertIn("output_not_json_object", bad_json.output_validation["reasons"])
        self.assertEqual(timeout.status, "timeout")
        self.assertTrue(timeout.timed_out)
        self.assertIn("timeout", timeout.output_validation["reasons"])
        self.assertIn("worker_direct_solved_claim", unsafe.output_validation["reasons"])
        self.assertIn("worker_raw_reaction_injection", unsafe.output_validation["reasons"])
        self.assertIn("tool_call_budget_exceeded", tool_budget.output_validation["reasons"])

    def test_worker_accepts_current_cli_wait_tool_as_wait_agent_alias(self):
        task = WorkerTask(
            task_id="coordinator_wait_alias",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            allowed_tools=["spawn_agent", "wait_agent"],
            budget=WorkerBudget(max_tool_calls=2),
        )
        artifact = {
            "schema_version": "research_report.draft.v1",
            "artifact_id": "wait_alias",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "codex_cli",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
        }

        record = run_codex_worker(
            task,
            runner=lambda _: WorkerProcessResult(
                stdout=json.dumps(artifact),
                exit_code=0,
                tool_calls=[{"tool": "wait"}],
            ),
        )

        self.assertEqual(record.status, "accepted_draft")
        self.assertNotIn("tool_not_allowed", record.output_validation["reasons"])

    def test_worker_accepts_valid_codex_output_after_recovered_transport_error(self):
        task = WorkerTask(
            task_id="recovered_codex_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
        )
        artifact = {
            "schema_version": "research_report.draft.v1",
            "artifact_id": "recovered",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "codex_cli",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
        }

        record = run_codex_worker(
            task,
            runner=lambda _: WorkerProcessResult(
                stdout=json.dumps(artifact),
                stderr="Reconnecting... 5/5 (request timed out)",
                exit_code=1,
                backend="codex_cli",
                metadata={
                    "event_summary": {
                        "turn_completed": True,
                        "last_terminal_event_type": "turn.completed",
                        "fatal_error": "",
                    }
                },
            ),
        )

        self.assertEqual(record.status, "accepted_draft")
        self.assertNotIn(
            "worker_exit_code_nonzero", record.output_validation["reasons"]
        )

    def test_worker_only_recovers_disallowed_tool_rejected_before_execution(self):
        task = WorkerTask(
            task_id="sandbox_rejected_tool_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            allowed_tools=["web_search"],
        )
        artifact = {
            "schema_version": "research_report.draft.v1",
            "artifact_id": "sandbox_recovered",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "codex_cli",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
        }
        terminal = {
            "event_summary": {
                "turn_completed": True,
                "last_terminal_event_type": "turn.completed",
                "fatal_error": "",
            }
        }

        rejected_before_launch = run_codex_worker(
            task,
            runner=lambda _: WorkerProcessResult(
                stdout=json.dumps(artifact),
                exit_code=0,
                backend="codex_cli",
                metadata=terminal,
                tool_calls=[
                    {
                        "tool": "shell",
                        "status": "failed",
                        "exit_code": -1,
                        "aggregated_output": "execution error: sandbox launch rejected",
                    }
                ],
            ),
        )
        command_ran_and_failed = run_codex_worker(
            task,
            runner=lambda _: WorkerProcessResult(
                stdout=json.dumps(artifact),
                exit_code=0,
                backend="codex_cli",
                metadata=terminal,
                tool_calls=[
                    {
                        "tool": "shell",
                        "status": "failed",
                        "exit_code": 1,
                        "aggregated_output": "command produced output before failure",
                    }
                ],
            ),
        )

        self.assertEqual(rejected_before_launch.status, "accepted_draft")
        self.assertNotIn(
            "tool_not_allowed", rejected_before_launch.output_validation["reasons"]
        )
        self.assertEqual(command_ran_and_failed.status, "rejected_output")
        self.assertIn(
            "tool_not_allowed", command_ran_and_failed.output_validation["reasons"]
        )

    def test_worker_command_timeout_does_not_hang_when_descendant_keeps_pipe_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_path = root / "descendant.pid"
            script = (
                "import pathlib, subprocess, sys, time\n"
                f"pid_path = pathlib.Path({str(pid_path)!r})\n"
                "desc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
                "pid_path.write_text(str(desc.pid), encoding='utf-8')\n"
                "time.sleep(30)\n"
            )

            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_worker_command([sys.executable, "-c", script], cwd=root, timeout_s=0.2)
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 4.0)
            if pid_path.exists():
                _kill_test_process_tree(int(pid_path.read_text(encoding="utf-8")))

    def test_worker_command_cancellation_terminates_process_tree_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_path = root / "descendant.pid"
            script = (
                "import pathlib, subprocess, sys, time\n"
                f"pid_path = pathlib.Path({str(pid_path)!r})\n"
                "desc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
                "pid_path.write_text(str(desc.pid), encoding='utf-8')\n"
                "time.sleep(30)\n"
            )
            cancel_event = threading.Event()
            timer = threading.Timer(0.2, cancel_event.set)
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaises(WorkerCancelledError) as raised:
                    _run_worker_command(
                        [sys.executable, "-c", script],
                        cwd=root,
                        timeout_s=30.0,
                        cancel_event=cancel_event,
                        cancel_backend="fixture_backend",
                    )
            finally:
                timer.cancel()
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 4.0)
            self.assertEqual(raised.exception.backend, "fixture_backend")
            if pid_path.exists():
                _kill_test_process_tree(int(pid_path.read_text(encoding="utf-8")))

    def test_worker_output_cannot_self_validate_for_consumption_layer(self):
        task = WorkerTask(
            task_id="strategic_operator_worker",
            case_id="case",
            task_type="strategic_disconnection_mining",
            required_artifact_type="StrategicOperator",
            input_refs=["evidence"],
        )

        record = run_codex_worker(
            task,
            mock_output={
                "schema_version": "strategic_operator.draft.v1",
                "artifact_id": "operator_draft",
                "artifact_type": "StrategicOperator",
                "case_id": "case",
                "source": "codex_worker_mock",
                "input_refs": ["evidence"],
                "evidence_refs": ["ev1"],
                "validation_status": "validated",
                "payload": {"terminal_blacklist": ["CCO"]},
            },
        )

        self.assertEqual(record.status, "rejected_output")
        self.assertIn("worker_output_must_be_draft", record.output_validation["reasons"])

    def test_worker_can_use_codex_cli_backend(self):
        task = WorkerTask(
            task_id="codex_cli_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        artifact = {
            "schema_version": "researchreport.draft.v1",
            "artifact_id": "codex_cli_worker:ResearchReport",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "codex_cli",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "summary": "Codex CLI draft",
        }

        with patch(
            "cascade_planner.agent.codex_worker._run_codex_cli_worker",
            return_value=WorkerProcessResult(
                stdout=json.dumps(artifact),
                exit_code=0,
                backend="codex_cli",
                command=["codex", "exec", "-"],
            ),
        ) as run_cli:
            record = run_codex_worker(task, use_codex_cli=True)

        run_cli.assert_called_once()
        self.assertEqual(record.status, "accepted_draft")
        self.assertEqual(record.backend, "codex_cli")
        self.assertEqual(record.command[:2], ["codex", "exec"])
        self.assertEqual(record.output_artifact["source"], "codex_cli")

    def test_codex_cli_command_places_top_level_options_before_exec(self):
        command = _codex_cli_command(
            executable="codex",
            workdir=Path("/tmp/autoplanner-case"),
            output_path=Path("/tmp/autoplanner-worker/last_message.json"),
            schema_path=Path("/tmp/autoplanner-worker/schema.json"),
        )
        no_search_command = _codex_cli_command(
            executable="codex",
            workdir=Path("/tmp/autoplanner-case"),
            output_path=Path("/tmp/autoplanner-worker/last_message.json"),
            schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            search_enabled=False,
        )
        search_command = _codex_cli_command(
            executable="codex",
            workdir=Path("/tmp/autoplanner-case"),
            output_path=Path("/tmp/autoplanner-worker/last_message.json"),
            schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            search_enabled=True,
        )
        effort_command = _codex_cli_command(
            executable="codex",
            workdir=Path("/tmp/autoplanner-case"),
            output_path=Path("/tmp/autoplanner-worker/last_message.json"),
            schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            runtime_metadata={"codex_home": "ambient", "model_reasoning_effort": "medium"},
            search_enabled=False,
        )

        self.assertEqual(command[0], "codex")
        self.assertIn("exec", command)
        exec_index = command.index("exec")
        approval_index = command.index("--ask-for-approval")
        self.assertLess(approval_index, exec_index)
        self.assertEqual(command[approval_index + 1], "never")
        self.assertGreater(command.index("--sandbox"), exec_index)
        self.assertNotIn("--search", no_search_command)
        self.assertIn("--search", search_command)
        self.assertIn('model_reasoning_effort="medium"', effort_command)
        self.assertLess(search_command.index("--search"), search_command.index("exec"))

    def test_codex_cli_command_allows_unsandboxed_worker_modes(self):
        with patch.dict("os.environ", {"AUTOPLANNER_CODEX_WORKER_SANDBOX": "danger-full-access"}, clear=False):
            danger_command = _codex_cli_command(
                executable="codex",
                workdir=Path("/tmp/autoplanner-case"),
                output_path=Path("/tmp/autoplanner-worker/last_message.json"),
                schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            )
        with patch.dict("os.environ", {"AUTOPLANNER_CODEX_WORKER_SANDBOX": "bypassed"}, clear=False):
            bypass_command = _codex_cli_command(
                executable="codex",
                workdir=Path("/tmp/autoplanner-case"),
                output_path=Path("/tmp/autoplanner-worker/last_message.json"),
                schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            )

        self.assertIn("--sandbox", danger_command)
        self.assertEqual(danger_command[danger_command.index("--sandbox") + 1], "danger-full-access")
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", bypass_command)
        self.assertNotIn("--sandbox", bypass_command)

    def test_agent_action_batch_schema_uses_supported_json_schema_keywords(self):
        task = WorkerTask(
            task_id="planner",
            case_id="case",
            task_type="strategic_disconnection_mining",
            required_artifact_type="AgentActionBatch",
            allowed_tools=["web_search"],
        )

        schema_text = json.dumps(_worker_output_json_schema(task), sort_keys=True)
        schema = _worker_output_json_schema(task)
        batch_payload_schema = schema["properties"]["payload"]
        action_payload_schema = (
            batch_payload_schema["properties"]["actions"]["items"]["properties"]["payload"]
        )

        self.assertNotIn("maxProperties", schema_text)
        self.assertNotIn("maxItems", schema_text)
        self.assertNotIn("maxLength", schema_text)
        self.assertFalse(action_payload_schema["additionalProperties"])
        self.assertEqual(
            set(batch_payload_schema["required"]),
            set(batch_payload_schema["properties"]),
        )
        self.assertIn("queries", action_payload_schema["properties"])
        self.assertEqual(
            set(action_payload_schema["required"]),
            set(action_payload_schema["properties"]),
        )

    def test_codex_cli_search_is_task_gated(self):
        planner_task = WorkerTask(
            task_id="planner",
            case_id="case",
            task_type="strategic_disconnection_mining",
            required_artifact_type="AgentActionBatch",
            allowed_tools=[],
            budget=WorkerBudget(max_tool_calls=0),
        )
        local_only_task = WorkerTask(
            task_id="local_only",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            allowed_tools=["local_search"],
            budget=WorkerBudget(max_tool_calls=3),
        )
        literature_task = WorkerTask(
            task_id="literature",
            case_id="case",
            task_type="target_research",
            required_artifact_type="LiteratureScoutReport",
            allowed_tools=["web_search", "browser", "local_search"],
            budget=WorkerBudget(max_tool_calls=3),
        )

        self.assertFalse(_task_allows_cli_search(planner_task))
        self.assertFalse(_task_allows_cli_search(local_only_task))
        self.assertTrue(_task_allows_cli_search(literature_task))

    def test_codex_cli_worker_uses_retrosynthesis_key_file_in_ephemeral_home(self):
        task = WorkerTask(
            task_id="codex_cli_worker_key_file",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            key_path = tmp_path / "key.txt"
            key_path.write_text("'  file-worker-key  '\n", encoding="utf-8")
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_WORKER_API_BASE_URL": "https://api.example.test/v1",
                        "AUTOPLANNER_WORKER_API_MODEL": "test-model",
                        "AUTOPLANNER_WORKER_API_PROVIDER": "test-provider",
                    },
                    clear=False,
                ):
                    config = _api_json_config(task)
                    env, metadata = _codex_cli_worker_environment(
                        tmp_path / "runtime",
                        Path("/tmp/autoplanner-case"),
                        config,
                    )

                    codex_home = Path(env["CODEX_HOME"])
                    auth = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
                    config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
                    self.assertEqual(auth["OPENAI_API_KEY"], "file-worker-key")
                    self.assertEqual(auth["auth_mode"], "apikey")
                    self.assertIn('openai_base_url = "https://api.example.test/v1"', config_text)
                    self.assertIn('model = "test-model"', config_text)
                    self.assertEqual(metadata["auth_source"], str(key_path))
                    self.assertEqual(metadata["codex_home"], "ephemeral")
                    self.assertNotIn("file-worker-key", json.dumps(metadata))

    def test_codex_cli_worker_can_use_ambient_codex_auth(self):
        task = WorkerTask(
            task_id="codex_cli_worker_ambient",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.dict(
                "os.environ",
                {
                    "AUTOPLANNER_CODEX_WORKER_AUTH": "ambient",
                    "CODEX_HOME": str(tmp_path / "should_not_leak"),
                },
                clear=False,
            ):
                env, metadata = _codex_cli_runtime_environment(
                    tmp_path / "runtime",
                    Path("/tmp/autoplanner-case"),
                    task,
                )

        self.assertNotIn("CODEX_HOME", env)
        self.assertEqual(metadata["provider"], "ambient_codex_cli")
        self.assertEqual(metadata["auth_source"], "ambient_codex_cli")
        self.assertEqual(metadata["codex_home"], "ambient")

    def test_codex_cli_worker_passes_ephemeral_home_to_subprocess(self):
        task = WorkerTask(
            task_id="codex_cli_worker_subprocess_env",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_path = root / "key.txt"
            key_path.write_text("subprocess-worker-key\n", encoding="utf-8")
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json, os, pathlib, sys",
                        "out = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])",
                        "home = pathlib.Path(os.environ.get('CODEX_HOME', ''))",
                        "capture = {",
                        "  'codex_home': str(home),",
                        "  'auth': json.loads((home / 'auth.json').read_text()) if home else {},",
                        "  'config': (home / 'config.toml').read_text() if home else '',",
                        "  'argv': sys.argv,",
                        "}",
                        "pathlib.Path(os.environ['CAPTURE_CODEX_WORKER_ENV']).write_text(json.dumps(capture), encoding='utf-8')",
                        "artifact = {",
                        "  'schema_version': 'research_report.v1',",
                        "  'artifact_id': 'codex_cli_worker_subprocess_env:ResearchReport',",
                        "  'artifact_type': 'ResearchReport',",
                        "  'case_id': 'case',",
                        "  'source': 'codex_cli',",
                        "  'input_refs': ['target_profile'],",
                        "  'evidence_refs': [],",
                        "  'validation_status': 'draft',",
                        "  'summary': 'ok',",
                        "  'payload': {'schema_version': 'research_report_payload.v1', 'no_solved_claim': True},",
                        "}",
                        "out.write_text(json.dumps(artifact), encoding='utf-8')",
                    ]
                ),
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_CODEX_CLI_BIN": str(fake_codex),
                        "AUTOPLANNER_WORKER_API_BASE_URL": "https://api.example.test/v1",
                        "AUTOPLANNER_WORKER_API_MODEL": "test-model",
                        "AUTOPLANNER_WORKER_API_PROVIDER": "test-provider",
                        "AUTOPLANNER_CODEX_WORKER_REASONING_EFFORT": "xhigh",
                        "CAPTURE_CODEX_WORKER_ENV": str(root / "captured_codex_worker_env.json"),
                    },
                    clear=False,
                ):
                    record = run_codex_worker(task, use_codex_cli=True)

            captured = json.loads((root / "captured_codex_worker_env.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted_draft", record.output_validation)
        self.assertEqual(record.backend, "codex_cli")
        self.assertEqual(record.metadata["codex_home"], "ephemeral")
        self.assertEqual(record.metadata["auth_source"], str(key_path))
        self.assertEqual(captured["auth"]["OPENAI_API_KEY"], "subprocess-worker-key")
        self.assertIn("--ignore-user-config", captured["argv"])
        self.assertIn('openai_base_url="https://api.example.test/v1"', captured["argv"])
        # The test process may carry an explicit operator override; it must
        # still take precedence over the safer implicit medium fallback.
        self.assertIn('model_reasoning_effort="xhigh"', captured["argv"])
        self.assertIn('openai_base_url = "https://api.example.test/v1"', captured["config"])
        self.assertIn('model = "test-model"', captured["config"])
        self.assertNotIn("--search", captured["argv"])
        self.assertNotIn("subprocess-worker-key", json.dumps(record.metadata))

    def test_codex_cli_worker_task_can_override_reasoning_effort(self):
        task = WorkerTask(
            task_id="codex_cli_worker_effort",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            budget=WorkerBudget(reasoning_effort="high"),
            dry_run=False,
        )
        command = _codex_cli_command(
            executable="codex",
            workdir=Path("/tmp/autoplanner-case"),
            output_path=Path("/tmp/autoplanner-worker/last_message.json"),
            schema_path=Path("/tmp/autoplanner-worker/schema.json"),
            runtime_metadata={"codex_home": "ambient", "model_reasoning_effort": _task_reasoning_effort(task)},
            search_enabled=False,
        )

        self.assertEqual(_task_reasoning_effort(task), "high")
        self.assertIn('model_reasoning_effort="high"', command)

    def test_codex_cli_worker_implicit_reasoning_effort_is_bounded(self):
        task = WorkerTask(
            task_id="codex_cli_worker_default_effort",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
        )
        with patch.dict(
            "os.environ",
            {"AUTOPLANNER_CODEX_WORKER_REASONING_EFFORT": ""},
            clear=False,
        ):
            self.assertEqual(_task_reasoning_effort(task), "medium")

    def test_worker_timeout_preserves_backend_and_command(self):
        task = WorkerTask(
            task_id="timeout_codex_cli_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )

        with patch(
            "cascade_planner.agent.codex_worker._run_codex_cli_worker",
            side_effect=WorkerTimeoutError(
                "worker timeout after 1.0s",
                backend="codex_cli",
                command=["codex", "--ask-for-approval", "never", "exec", "-"],
            ),
        ):
            record = run_codex_worker(task, use_codex_cli=True)

        self.assertEqual(record.status, "timeout")
        self.assertTrue(record.timed_out)
        self.assertEqual(record.backend, "codex_cli")
        self.assertEqual(record.command[:4], ["codex", "--ask-for-approval", "never", "exec"])
        self.assertIn("timeout", record.output_validation["reasons"])

    def test_worker_can_use_api_json_backend_with_provider_metadata(self):
        task = WorkerTask(
            task_id="api_json_worker",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        artifact = {
            "schema_version": "research_report.v1",
            "artifact_id": "api_json_worker:ResearchReport",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "api_json",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {"summary": "API JSON draft"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "key.txt"
            key_path.write_text("test-key\n", encoding="utf-8")
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_WORKER_API_KEY": "ignored-env-key",
                        "AUTOPLANNER_WORKER_API_BASE_URL": "https://api.example.test/v1",
                        "AUTOPLANNER_WORKER_API_MODEL": "test-model",
                        "AUTOPLANNER_WORKER_API_PROVIDER": "test-provider",
                        "AUTOPLANNER_WORKER_API_ENDPOINT": "responses",
                    },
                    clear=False,
                ):
                    with patch(
                        "cascade_planner.agent.codex_worker._post_api_json",
                        return_value={"output_text": json.dumps(artifact), "usage": {"input_tokens": 12}},
                    ) as post_api:
                        record = run_codex_worker(task, use_api_json=True)

        post_api.assert_called_once()
        self.assertEqual(post_api.call_args.args[0]["api_key"], "test-key")
        self.assertEqual(record.status, "accepted_draft")
        self.assertEqual(record.backend, "api_json")
        self.assertEqual(record.command, ["api_json", "POST", "/responses"])
        self.assertEqual(record.metadata["provider"], "test-provider")
        self.assertEqual(record.metadata["model"], "test-model")
        self.assertIn("base_url_fingerprint", record.metadata)
        self.assertEqual(record.usage["input_tokens"], 12)

    def test_api_json_worker_uses_retrosynthesis_key_file_without_trace_leak(self):
        task = WorkerTask(
            task_id="api_json_worker_key_file",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        artifact = {
            "schema_version": "research_report.v1",
            "artifact_id": "api_json_worker_key_file:ResearchReport",
            "artifact_type": "ResearchReport",
            "case_id": "case",
            "source": "api_json",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {"summary": "API JSON draft"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "key.txt"
            key_path.write_text("'  file-worker-key  '\n", encoding="utf-8")
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_RETROSYNTHESIS_API_KEY": "ignored-env-key",
                        "AUTOPLANNER_WORKER_API_BASE_URL": "https://api.example.test/v1",
                        "AUTOPLANNER_WORKER_API_MODEL": "test-model",
                        "AUTOPLANNER_WORKER_API_PROVIDER": "test-provider",
                    },
                    clear=True,
                ):
                    config = _api_json_config(task)
                    with patch(
                        "cascade_planner.agent.codex_worker._post_api_json",
                        return_value={"output_text": json.dumps(artifact), "usage": {"input_tokens": 12}},
                    ) as post_api:
                        record = run_codex_worker(task, use_api_json=True)

        post_api.assert_called_once()
        self.assertEqual(config["api_key"], "file-worker-key")
        self.assertEqual(post_api.call_args.args[0]["api_key"], "file-worker-key")
        self.assertEqual(record.status, "accepted_draft")
        self.assertEqual(record.backend, "api_json")
        self.assertNotIn("api_key", record.metadata)
        self.assertNotIn("file-worker-key", json.dumps(record.to_dict()))

    def test_api_json_worker_does_not_fallback_to_environment_key(self):
        task = WorkerTask(
            task_id="api_json_worker_missing_key_file",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "key.txt"
            key_path.write_text("", encoding="utf-8")
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_WORKER_API_KEY": "ignored-env-key",
                        "AUTOPLANNER_RETROSYNTHESIS_API_KEY": "ignored-retrosynthesis-key",
                        "OPENAI_API_KEY": "ignored-openai-key",
                        "DEEPSEEK_API_KEY": "ignored-deepseek-key",
                        "AUTOPLANNER_WORKER_API_BASE_URL": "https://api.example.test/v1",
                    },
                    clear=False,
                ):
                    with patch("cascade_planner.agent.codex_worker._post_api_json") as post_api:
                        record = run_codex_worker(task, use_api_json=True)

        post_api.assert_not_called()
        self.assertEqual(record.status, "worker_error")
        self.assertEqual(record.backend, "api_json")
        self.assertIn("missing API key", record.stderr)
        self.assertFalse(record.output_validation["accepted"])

    def test_api_json_config_defaults_to_wellau_chat_with_key_file(self):
        task = WorkerTask(
            task_id="api_json_worker_default_wellau",
            case_id="case",
            task_type="target_research",
            required_artifact_type="ResearchReport",
            input_refs=["target_profile"],
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "key.txt"
            key_path.write_text("test-key\n", encoding="utf-8")
            with patch("cascade_planner.agent.codex_worker.DEFAULT_RETROSYNTHESIS_KEY_FILE", key_path):
                with patch.dict(
                    "os.environ",
                    {
                        "AUTOPLANNER_WORKER_API_BASE_URL": "",
                        "AUTOPLANNER_CODEX_WORKER_BASE_URL": "",
                        "OPENAI_BASE_URL": "",
                        "DEEPSEEK_BASE_URL": "",
                        "AUTOPLANNER_WORKER_API_PROVIDER": "",
                        "AUTOPLANNER_WORKER_API_ENDPOINT": "",
                    },
                    clear=False,
                ):
                    config = _api_json_config(task)

        self.assertEqual(config["base_url"], "https://api.wellau.com/v1")
        self.assertEqual(config["provider"], "wellau")
        self.assertEqual(config["endpoint"], "chat/completions")

    def test_controller_selects_policy_compile_for_validated_partial_anchor(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.PARTIAL_ANCHOR)
        bundle.append_artifact(ArtifactRecord(
            artifact_id="validation",
            case_id="case",
            artifact_type="RoutePackageValidation",
            payload={"accepted": True, "route_status": "partial_anchor"},
        ))
        trace = ControllerTrace(case_id="case")
        observation = observe_case(bundle, trace)

        action = decide_next_action(observation, {"budget": ControllerBudget().to_dict()})

        self.assertEqual(action.action_type, "COMPILE_STRATEGIC_OPERATOR")

    def test_controller_rejects_forbidden_actions_and_raw_reactions(self):
        forbidden = validate_controller_action(ControllerAction("LLM_RERANK_CANDIDATES"))
        raw = validate_controller_action(ControllerAction(
            "RUN_GUIDED_CHEMENZY",
            payload={"rxn_smiles": "CCO>>CC=O"},
        ))
        production = validate_controller_action(ControllerAction(
            "SUBMIT_EVOLUTION_CANDIDATE",
            payload={"write_layer": "production"},
        ))

        self.assertFalse(forbidden["accepted"])
        self.assertIn("forbidden_controller_action", forbidden["reasons"])
        self.assertFalse(raw["accepted"])
        self.assertIn("raw_reaction_injection", raw["reasons"])
        self.assertFalse(production["accepted"])
        self.assertIn("controller_direct_production_write", production["reasons"])

    def test_controller_budget_exhaustion_returns_unresolved(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.UNRESOLVED)
        bundle.append_failure_event(FailureEvent(
            failure_id="frontier",
            case_id="case",
            reason="unresolved_core",
        ))

        result = run_controller_once(
            bundle,
            budget=ControllerBudget(max_tool_calls=0),
        )

        self.assertEqual(result["action"]["action_type"], "FINAL_UNRESOLVED")
        self.assertEqual(result["trace"]["final_route_status"], "unresolved")

    def test_controller_appends_worker_artifact_validation_and_rerun_history(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.UNRESOLVED)
        evidence_artifact = {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": "ev_artifact",
            "artifact_type": "EvidenceCard",
            "case_id": "case",
            "source": "unit_test",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {
                "evidence_id": "ev1",
                "case_id": "case",
                "source_type": "literature",
                "source_title": "Traceable precedent",
                "target_relation": "family_precedent",
                "claim_type": "strategic_disconnection",
                "route_role": "strategic_disconnection",
                "url": "https://example.org/route",
                "validation_status": "draft",
            },
        }
        worker_trace = {
            "run_id": "worker_run",
            "task_id": "worker_task",
            "case_id": "case",
            "status": "accepted_draft",
            "backend": "api_json",
            "command": ["api_json", "POST", "/responses"],
            "output_validation": {"accepted": True, "reasons": []},
            "output_artifact": evidence_artifact,
        }

        update_blackboard(bundle, {
            "worker_trace": worker_trace,
            "output_artifact": evidence_artifact,
            "rerun_history": {"policy_id": "policy_1", "attempts": []},
        })

        by_type = {artifact.artifact_type: artifact for artifact in bundle.artifacts}
        self.assertIn("WorkerRunRecord", by_type)
        self.assertIn("EvidenceCard", by_type)
        self.assertIn("ArtifactValidationRecord", by_type)
        self.assertIn("GuidedRerunHistory", by_type)
        self.assertEqual(by_type["WorkerRunRecord"].validation_status, "accepted")
        self.assertEqual(by_type["EvidenceCard"].validation_status, "accepted")

    def test_controller_default_flow_research_validate_compile_rerun(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.UNRESOLVED)
        evidence_artifact = {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": "ev_artifact",
            "artifact_type": "EvidenceCard",
            "case_id": "case",
            "source": "api_json",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {
                "evidence_id": "ev1",
                "case_id": "case",
                "source_type": "literature",
                "source_title": "Traceable precedent",
                "target_relation": "family_precedent",
                "claim_type": "strategic_disconnection",
                "route_role": "strategic_disconnection",
                "url": "https://example.org/route",
                "validation_status": "draft",
            },
        }
        record_payload = {
            "run_id": "worker_run",
            "task_id": "worker_task",
            "case_id": "case",
            "status": "accepted_draft",
            "backend": "api_json",
            "command": ["api_json", "POST", "/responses"],
            "output_validation": {"accepted": True, "reasons": []},
        }
        trace = ControllerTrace(case_id="case")

        update_blackboard(bundle, {
            "worker_trace": record_payload,
            "output_artifact": evidence_artifact,
        })
        validate_result = execute_action(ControllerAction("VALIDATE_EVIDENCE"), bundle, trace)
        bundle = update_blackboard(bundle, validate_result)
        compile_result = execute_action(ControllerAction("COMPILE_STRATEGIC_OPERATOR"), bundle, trace)
        bundle = update_blackboard(bundle, compile_result)
        rerun_result = execute_action(ControllerAction("RUN_GUIDED_CHEMENZY"), bundle, trace)
        bundle = update_blackboard(bundle, rerun_result)

        types = [artifact.artifact_type for artifact in bundle.artifacts]
        self.assertIn("EvidenceCard", types)
        self.assertIn("StrategicOperator", types)
        self.assertIn("GuidedRerunHistory", types)
        self.assertEqual(rerun_result["final_route_status"], "unresolved")
        self.assertNotIn("RAW_REACTION_INJECTION", [row["action"]["action_type"] for row in trace.actions])

    def test_controller_loop_research_validate_compile_rerun_audit(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.UNRESOLVED)
        evidence_artifact = {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": "ev_artifact",
            "artifact_type": "EvidenceCard",
            "case_id": "case",
            "source": "api_json",
            "input_refs": ["target_profile"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {
                "evidence_id": "ev1",
                "case_id": "case",
                "source_type": "literature",
                "source_title": "Traceable precedent",
                "target_relation": "family_precedent",
                "claim_type": "strategic_disconnection",
                "route_role": "strategic_disconnection",
                "url": "https://example.org/route",
                "validation_status": "draft",
            },
        }
        record = WorkerRunRecord(
            run_id="worker_run",
            task_id="worker_task",
            case_id="case",
            status="accepted_draft",
            backend="api_json",
            command=["api_json", "POST", "/responses"],
            output_validation={"schema_version": "worker_output_validation.v1", "accepted": True, "reasons": []},
            output_artifact=evidence_artifact,
            metadata={"provider": "test-provider", "base_url_fingerprint": "abc123", "model": "test-model"},
        )

        with patch("cascade_planner.agent.codex_controller.run_codex_worker", return_value=record) as run_worker:
            result = run_controller_loop(bundle, budget=ControllerBudget(max_tool_calls=10))

        action_types = [row["action"]["action_type"] for row in result["executed"]]
        artifact_types = {row["artifact_type"] for row in result["case_bundle"]["artifacts"]}
        worker_records = [
            row for row in result["case_bundle"]["artifacts"]
            if row["artifact_type"] == "WorkerRunRecord"
        ]

        self.assertEqual(action_types, [
            "RESEARCH_TARGET",
            "EXTRACT_EVIDENCE",
            "VALIDATE_EVIDENCE",
            "MINE_STRATEGIC_DISCONNECTION",
            "COMPILE_STRATEGIC_OPERATOR",
            "RUN_GUIDED_CHEMENZY",
            "AUDIT_ROUTE",
        ])
        self.assertTrue(run_worker.call_args.kwargs["use_codex_cli"])
        self.assertIn("WorkerRunRecord", artifact_types)
        self.assertIn("EvidenceCard", artifact_types)
        self.assertIn("ArtifactValidationRecord", artifact_types)
        self.assertIn("StrategicDisconnectionCard", artifact_types)
        self.assertIn("StrategicOperator", artifact_types)
        self.assertIn("GuidedRerunHistory", artifact_types)
        self.assertIn("RouteAuditReport", artifact_types)
        self.assertEqual(worker_records[0]["payload"]["backend"], "api_json")
        self.assertEqual(worker_records[0]["payload"]["command"], ["api_json", "POST", "/responses"])
        self.assertEqual(result["trace"]["final_route_status"], "unresolved")

    def test_evolution_kb_requires_gate_for_production_and_supports_rollback(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="anchor_candidate",
            candidate_type="AnchorCandidate",
            payload={"anchor_id": "known_anchor"},
            evidence_refs=["ev1"],
            validation_status="validated",
        )
        kb.add_candidate(candidate)
        kb.promote("anchor_candidate", from_layer="candidate", to_layer="shadow")
        kb.promote("anchor_candidate", from_layer="shadow", to_layer="staging")

        failed_gate = evaluate_benchmark_gate({"fake_closure_rate_delta": 0.1})
        passed_gate = evaluate_benchmark_gate({
            "true_solved_rate_delta": 0.0,
            "fake_closure_rate_delta": 0.0,
            "condition_quality_delta": 0.0,
        })

        with self.assertRaisesRegex(ValueError, "benchmark_gate_failed"):
            kb.promote("anchor_candidate", from_layer="staging", to_layer="production", gate_report=failed_gate)
        with self.assertRaisesRegex(ValueError, "target_run_cannot_write_production"):
            kb.promote("anchor_candidate", from_layer="staging", to_layer="production", gate_report=passed_gate, target_run=True)

        kb.promote("anchor_candidate", from_layer="staging", to_layer="production", gate_report=passed_gate)
        self.assertIn("anchor_candidate", kb.layers["production"])
        kb.rollback("anchor_candidate")
        self.assertNotIn("anchor_candidate", kb.layers["production"])


def _kill_test_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        # taskkill can return just before Windows releases the process current
        # directory and inherited pipe handles. Poll briefly so TemporaryDirectory
        # cleanup verifies the production timeout path instead of a kernel race.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if str(int(pid)) not in str(probe.stdout or ""):
                break
            time.sleep(0.02)
        return
    try:
        os.kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    unittest.main()
