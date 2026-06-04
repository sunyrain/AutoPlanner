import json
import unittest

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
    run_controller_once,
    validate_controller_action,
)
from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerProcessResult,
    WorkerTask,
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
