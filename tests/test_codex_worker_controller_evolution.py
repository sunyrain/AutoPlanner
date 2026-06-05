import json
import tempfile
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
    WorkerTimeoutError,
    _api_json_config,
    _codex_cli_worker_environment,
    _codex_cli_command,
    run_codex_worker,
)
from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
)


class CodexWorkerControllerEvolutionTest(unittest.TestCase):
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

        self.assertEqual(command[0], "codex")
        self.assertIn("exec", command)
        exec_index = command.index("exec")
        approval_index = command.index("--ask-for-approval")
        self.assertLess(approval_index, exec_index)
        self.assertEqual(command[approval_index + 1], "never")
        self.assertGreater(command.index("--sandbox"), exec_index)

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


if __name__ == "__main__":
    unittest.main()
