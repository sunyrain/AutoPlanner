import copy
import json
import unittest

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

from cascade_planner.agent.action_contracts import (
    ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS,
    contains_raw_reaction_payload,
)
from cascade_planner.agent.artifact_schemas import (
    ARTIFACT_CLASSES,
    StructureProfile,
    artifact_json_round_trip,
)
from cascade_planner.agent.artifact_validators import (
    validate_artifact_list,
    validate_typed_artifact,
)
from cascade_planner.agent.artifact_validators import (
    AGENT_ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS as VALIDATOR_AGENT_ACTIONS,
)
from cascade_planner.agent.codex_worker import WORKER_AGENT_ACTION_TYPES
from cascade_planner.agent.statin_panel import valid_statin_field_resolution_candidate_status
from cascade_planner.harness.agent_action_planner import (
    ACTION_BATCH_SCHEMA as PLANNER_ACTION_BATCH_SCHEMA,
    ALLOWED_AGENT_ACTIONS as PLANNER_AGENT_ACTIONS,
)


class AgentArtifactContractsTest(unittest.TestCase):
    def test_agent_action_contract_is_shared_by_planner_worker_and_validator(self):
        self.assertEqual(PLANNER_ACTION_BATCH_SCHEMA, ACTION_BATCH_SCHEMA)
        self.assertEqual(AGENT_ACTION_BATCH_SCHEMA, ACTION_BATCH_SCHEMA)
        self.assertEqual(PLANNER_AGENT_ACTIONS, ALLOWED_AGENT_ACTIONS)
        self.assertEqual(VALIDATOR_AGENT_ACTIONS, ALLOWED_AGENT_ACTIONS)
        self.assertEqual(WORKER_AGENT_ACTION_TYPES, ALLOWED_AGENT_ACTIONS)
        self.assertTrue(contains_raw_reaction_payload({"notes": "hidden CCO>>CC=O"}))
        self.assertTrue(contains_raw_reaction_payload({"payload": {"reaction_smiles": "CCO>>CC=O"}}))

    def test_all_typed_artifacts_construct_and_round_trip(self):
        for artifact_type, cls in ARTIFACT_CLASSES.items():
            artifact = cls(
                artifact_id=f"{artifact_type.lower()}_1",
                case_id="case",
                source="unit_test",
                input_refs=["input"],
                payload={"value": artifact_type},
            )

            data = artifact.to_dict()
            loaded = artifact_json_round_trip(artifact)

            self.assertEqual(data["artifact_type"], artifact_type)
            self.assertIn("schema_version", data)
            self.assertEqual(loaded.to_dict(), data)
            self.assertEqual(json.loads(artifact.to_json())["case_id"], "case")

    def test_agent_action_batch_artifact_is_registered(self):
        artifact = ARTIFACT_CLASSES["AgentActionBatch"](
            artifact_id="action_batch_1",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="draft",
            payload={
                "schema_version": "agent_action_batch.v1",
                "case_id": "case",
                "round_index": 1,
                "mode": "codex_blackboard_planner",
                "actions": [],
                "semantics": {
                    "planner_can_emit_solved": False,
                    "raw_reaction_output_allowed": False,
                    "deterministic_validator_required": True,
                },
            },
        )

        self.assertEqual(artifact.to_dict()["schema_version"], "agent_action_batch_artifact.v1")
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())

        validation = ARTIFACT_CLASSES["AgentActionBatchValidation"](
            artifact_id="action_batch_validation_1",
            case_id="case",
            source="unit_test",
            input_refs=["action_batch_round_1.json"],
            validation_status="accepted",
            payload={
                "schema_version": "agent_action_batch_validation.v1",
                "accepted": True,
                "reasons": [],
                "case_id": "case",
                "action_count": 0,
            },
        )
        self.assertEqual(validation.to_dict()["schema_version"], "agent_action_batch_validation_artifact.v1")
        self.assertEqual(artifact_json_round_trip(validation).to_dict(), validation.to_dict())

    def test_agent_action_batch_artifact_validator_rejects_solver_or_raw_injection(self):
        artifact = ARTIFACT_CLASSES["AgentActionBatch"](
            artifact_id="bad_action_batch",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="draft",
            payload={
                "schema_version": "agent_action_batch.v1",
                "case_id": "case",
                "round_index": 1,
                "route_status": "solved",
                "semantics": {
                    "planner_can_emit_solved": True,
                    "raw_reaction_output_allowed": True,
                    "deterministic_validator_required": True,
                },
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "bad:route",
                        "action_type": "run_guided_chemenzy",
                        "rationale": "bad",
                        "expected_artifact": "bad",
                        "success_condition": "bad",
                        "payload": {"comment": "hidden CCO>>CC=O"},
                    }
                ],
            },
        )

        result = validate_typed_artifact(artifact)

        self.assertFalse(result["accepted"])
        self.assertIn("planner_direct_solved_claim", result["reasons"])
        self.assertIn("planner_semantics_allow_solved_claim", result["reasons"])
        self.assertIn("planner_semantics_allow_raw_reaction_output", result["reasons"])
        self.assertIn("raw_reaction_injection", result["reasons"])

    def test_agent_blackboard_snapshot_artifact_validates_state_and_rejects_direct_solved_or_raw_route(self):
        payload = {
            "schema_version": "agent_blackboard.v1",
            "case_id": "case",
            "target_profile": {
                "schema_version": "agent_target_profile_summary.v1",
                "target_name": "ethanol",
                "target_smiles": "CCO",
                "valid": True,
            },
            "literature_evidence": {
                "schema_version": "agent_literature_evidence_summary.v1",
                "source_candidates": [],
                "source_refs": [],
                "confidence": "none",
            },
            "budget_state": {
                "schema_version": "agent_blackboard_budget_state.v1",
                "rounds_completed": 0,
                "max_rounds": 3,
                "scout_calls": 0,
                "max_scout_calls": 3,
                "visual_calls": 0,
                "max_visual_calls": 3,
                "chemenzy_runs": 0,
                "max_chemenzy_runs": 1,
                "child_target_runs": 0,
                "max_child_target_runs": 2,
                "template_application_actions": 0,
                "max_template_application_actions": 3,
            },
            "current_belief": {
                "schema_version": "agent_current_belief.v1",
                "template_policy": {"analogy_is_advisory_only": True},
            },
            "planner_history": [],
            "action_history": [],
            "artifact_refs": {},
            "parent_route_proof": {},
        }
        artifact = ARTIFACT_CLASSES["AgentBlackboardSnapshot"](
            artifact_id="blackboard_snapshot",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload=payload,
        )
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
        self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["AgentBlackboardSnapshot"](
            artifact_id="bad_blackboard_snapshot",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload={**payload, "route_status": "solved", "raw_reaction": "CCO>>CC=O"},
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("agent_blackboard_direct_solved_claim", bad_result["reasons"])
        self.assertIn("raw_reaction_injection", bad_result["reasons"])

    def test_agent_blackboard_snapshot_ignores_only_trusted_coordinator_prompt_syntax(self):
        payload = {
            "schema_version": "agent_blackboard.v1",
            "case_id": "case",
            "target_profile": {
                "schema_version": "agent_target_profile_summary.v1",
                "target_smiles": "CCO",
            },
            "literature_evidence": {
                "schema_version": "agent_literature_evidence_summary.v1",
            },
            "budget_state": {
                "schema_version": "agent_blackboard_budget_state.v1",
                "rounds_completed": 0,
                "max_rounds": 1,
            },
            "current_belief": {
                "schema_version": "agent_current_belief.v1",
                "template_policy": {"analogy_is_advisory_only": True},
            },
            "planner_history": [],
            "action_history": [],
            "artifact_refs": {},
            "codex_agent_team": {
                "coordinator": {
                    "observed_child_agents": [
                        {"prompt": "Never emit reaction strings containing '>>'."}
                    ]
                },
                "provider_envelope": {
                    "payload": {
                        "coordinator": {
                            "observed_child_agents": [
                                {"prompt": "The forbidden delimiter is >>."}
                            ]
                        }
                    }
                },
            },
            "agent_team_history": [
                {
                    "coordinator": {
                        "observed_child_agents": [
                            {"prompt": "Do not return reactants>>products."}
                        ]
                    }
                }
            ],
        }

        def artifact(value: dict) -> object:
            return ARTIFACT_CLASSES["AgentBlackboardSnapshot"](
                artifact_id="blackboard_snapshot",
                case_id="case",
                source="unit_test",
                input_refs=["agent_blackboard.json"],
                validation_status="accepted",
                payload=value,
            )

        self.assertTrue(validate_typed_artifact(artifact(payload))["accepted"])

        unsafe_model_payload = copy.deepcopy(payload)
        unsafe_model_payload["codex_agent_team"]["coordinator"][
            "observed_child_agents"
        ][0]["parsed_output"] = {"proposal": "CCO>>CC=O"}
        model_result = validate_typed_artifact(artifact(unsafe_model_payload))
        self.assertFalse(model_result["accepted"])
        self.assertIn("raw_reaction_injection", model_result["reasons"])

        unsafe_action_payload = copy.deepcopy(payload)
        unsafe_action_payload["action_history"] = [
            {
                "action_type": "run_guided_chemenzy",
                "payload": {"reaction_smiles": "CCO>>CC=O"},
            }
        ]
        action_result = validate_typed_artifact(artifact(unsafe_action_payload))
        self.assertFalse(action_result["accepted"])
        self.assertIn("raw_reaction_injection", action_result["reasons"])

    def test_agentic_run_audit_artifact_round_trip_and_parent_proof_guard(self):
        artifact = ARTIFACT_CLASSES["AgenticRunAudit"](
            artifact_id="run_audit",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload={
                "schema_version": "agentic_blackboard_run_audit.v1",
                "case_id": "case",
                "audit_authority": "diagnostic_only",
                "deterministic_final_verdict_required": True,
                "round_summaries": [],
                "budget_state": {},
                "parent_route_proof": {"accepted": False, "solved": False},
                "final_verdict": {"verdict": "unresolved", "solved": False},
            },
        )
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
        self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["AgenticRunAudit"](
            artifact_id="bad_run_audit",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload={
                **artifact.payload,
                "parent_route_proof": {"accepted": False, "solved": False},
                "final_verdict": {"verdict": "solved", "route_status": "solved", "solved": True},
            },
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("agentic_run_audit_solved_without_parent_proof", bad_result["reasons"])

        bad_authority = ARTIFACT_CLASSES["AgenticRunAudit"](
            artifact_id="bad_run_audit_analog_authority",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload={
                **artifact.payload,
                "analogical_template_summary": {
                    "schema_version": "agent_analogical_template_summary.v1",
                    "final_verdict_authority": "analogical_template",
                    "no_solved_claim": True,
                },
            },
        )
        bad_authority_result = validate_typed_artifact(bad_authority)
        self.assertFalse(bad_authority_result["accepted"])
        self.assertIn(
            "agentic_run_audit_analogical_template_as_final_authority",
            bad_authority_result["reasons"],
        )

    def test_agentic_capability_audit_artifact_round_trip_and_guardrails(self):
        artifact = ARTIFACT_CLASSES["AgenticCapabilityAudit"](
            artifact_id="capability_audit",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json", "final_verdict.json"],
            validation_status="accepted",
            payload={
                "schema_version": "agentic_capability_audit.v1",
                "case_id": "case",
                "accepted": True,
                "audit_authority": "diagnostic_only",
                "deterministic_final_verdict_required": True,
                "final_verdict_authority": "deterministic_parent_route_proof",
                "requirement_checks": [
                    {
                        "schema_version": "agentic_capability_requirement_check.v1",
                        "requirement_id": "blackboard_single_state_source",
                        "accepted": True,
                        "status": "checked",
                        "evidence": ["agent_blackboard.json"],
                        "reasons": [],
                        "no_solved_claim": True,
                    }
                ],
                "failed_requirements": [],
                "warning_requirements": [],
                "no_solved_claim": True,
            },
        )
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
        self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["AgenticCapabilityAudit"](
            artifact_id="bad_capability_audit",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json", "final_verdict.json"],
            validation_status="accepted",
            payload={
                **artifact.payload,
                "accepted": True,
                "final_verdict_authority": "analogical_template",
                "requirement_checks": [
                    {
                        "schema_version": "agentic_capability_requirement_check.v1",
                        "requirement_id": "final_verdict_requires_parent_route_proof",
                        "accepted": False,
                        "status": "checked",
                        "evidence": [],
                        "reasons": ["missing_parent_proof"],
                        "no_solved_claim": True,
                    }
                ],
                "failed_requirements": ["final_verdict_requires_parent_route_proof"],
            },
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("agentic_capability_audit_invalid_final_verdict_authority", bad_result["reasons"])
        self.assertIn("agentic_capability_audit_accepted_with_failed_requirements", bad_result["reasons"])

    def test_agentic_final_verdict_validation_artifact_round_trip_and_parent_proof_guard(self):
        artifact = ARTIFACT_CLASSES["AgenticFinalVerdictValidation"](
            artifact_id="final_validation",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json", "final_verdict.json"],
            validation_status="accepted",
            payload={
                "schema_version": "agentic_final_verdict_validation.v1",
                "accepted": True,
                "reasons": [],
                "case_id": "case",
                "final_verdict": {"verdict": "unresolved", "solved": False},
                "parent_route_proof_summary": {"accepted": False, "solved": False},
                "checked_invariants": ["solved_requires_parent_proof"],
            },
        )
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
        self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["AgenticFinalVerdictValidation"](
            artifact_id="bad_final_validation",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json", "final_verdict.json"],
            validation_status="accepted",
            payload={
                **artifact.payload,
                "final_verdict": {"verdict": "solved", "route_status": "solved", "solved": True},
                "parent_route_proof_summary": {"accepted": False, "solved": False},
            },
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("agentic_final_verdict_validation_solved_without_parent_proof", bad_result["reasons"])

    def test_literature_scout_report_artifact_validates_real_sources_and_rejects_placeholder_acceptance(self):
        artifact = ARTIFACT_CLASSES["LiteratureScoutReport"](
            artifact_id="scout_report",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            evidence_refs=["doi:10.1000/example"],
            validation_status="accepted",
            payload={
                "schema_version": "literature_scout_report.v1",
                "accepted": True,
                "case_id": "case",
                "source_candidates": [
                    {
                        "schema_version": "literature_source_candidate.v1",
                        "candidate_id": "src1",
                        "source_ref": "doi:10.1000/example",
                        "title": "Example source",
                        "doi": "10.1000/example",
                        "url": "https://doi.org/10.1000/example",
                        "local_pdf": "",
                        "source_type": "journal_article",
                        "access_status": "metadata_only",
                        "no_solved_claim": True,
                    }
                ],
                "source_refs": ["doi:10.1000/example"],
                "fallback_order": ["codex_online", "local_pdf", "placeholder"],
                "scout_attempts": [{"mode": "codex_online", "attempted": True, "accepted": True}],
                "no_solved_claim": True,
            },
        )
        self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
        self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["LiteratureScoutReport"](
            artifact_id="bad_scout_report",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            validation_status="accepted",
            payload={
                **artifact.payload,
                "accepted": True,
                "source_discovery_mode": "placeholder",
                "source_candidates": [
                    {
                        "schema_version": "literature_source_candidate.v1",
                        "candidate_id": "placeholder",
                        "source_ref": "query:target",
                        "title": "missing source query",
                        "doi": "",
                        "url": "",
                        "local_pdf": "",
                        "source_type": "placeholder_query",
                        "access_status": "placeholder_only",
                        "placeholder_only": True,
                        "no_solved_claim": True,
                    }
                ],
                "source_refs": ["query:target"],
            },
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("placeholder_scout_marked_accepted", bad_result["reasons"])
        self.assertIn("accepted_literature_scout_without_real_source", bad_result["reasons"])

    def test_analogical_template_artifacts_are_advisory_and_guarded(self):
        template = {
            "schema_version": "analogical_reaction_template.v1",
            "template_id": "tpl1",
            "relation_type": "analog",
            "reaction_class": "esterification",
            "mechanistic_class": "acyl_substitution",
            "reaction_center": {"product_retron_type": "aryl_ester_acyl_oxygen"},
            "template_radius": "r2",
            "source_refs": ["doi:analog"],
            "scope_gap": "analog must be verified on target",
            "confidence": "medium",
            "required_verification": ["route_verifier", "parent_route_proof"],
            "no_solved_claim": True,
            "not_raw_reaction_injection": True,
        }
        report = ARTIFACT_CLASSES["AnalogicalReactionTemplateReport"](
            artifact_id="tpl_report",
            case_id="case",
            source="unit_test",
            input_refs=["agent_blackboard.json"],
            evidence_refs=["doi:analog"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_reaction_template_report.v1",
                "accepted": True,
                "case_id": "case",
                "templates": [template],
                "source_refs": ["doi:analog"],
                "no_solved_claim": True,
            },
        )
        ranking = ARTIFACT_CLASSES["AnalogicalReactionTemplateRanking"](
            artifact_id="tpl_ranking",
            case_id="case",
            source="unit_test",
            input_refs=["tpl_report"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_reaction_template_ranking.v1",
                "accepted": True,
                "ranked_templates": [{"template_id": "tpl1", "score": 10, "no_solved_claim": True}],
                "selected_templates": [{"template_id": "tpl1", "score": 10, "no_solved_claim": True}],
                "no_solved_claim": True,
            },
        )
        application = ARTIFACT_CLASSES["AnalogicalTemplateApplicationReport"](
            artifact_id="tpl_application",
            case_id="case",
            source="unit_test",
            input_refs=["tpl_ranking"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_template_application_report.v1",
                "accepted": True,
                "applications": [
                    {
                        "schema_version": "analogical_template_application.v1",
                        "application_id": "app1",
                        "template_id": "tpl1",
                        "target_smiles": "CCO",
                        "product_retron_type": "aryl_ester_acyl_oxygen",
                        "accepted": False,
                        "allowed_use": "advisory_or_rerank_only",
                        "cut_fragments": [],
                        "reasons": ["scope_gap"],
                        "no_solved_claim": True,
                    }
                ],
                "candidate_payload_redacted": True,
                "no_solved_claim": True,
            },
        )
        validation = ARTIFACT_CLASSES["AnalogicalTemplateApplicationValidation"](
            artifact_id="tpl_validation",
            case_id="case",
            source="unit_test",
            input_refs=["tpl_application"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_template_application_validation.v1",
                "accepted": True,
                "case_id": "case",
                "target_smiles": "CCO",
                "executable_candidate_count": 1,
                "one_step_row_count": 1,
                "compiled_downstream": {
                    "schema_version": "analogical_template_guided_hints.v1",
                    "case_id": "case",
                    "target_smiles": "CCO",
                    "accepted": True,
                    "analogical_template_hints": {
                        "schema_version": "analogical_template_plugin_hints.v1",
                        "enabled": False,
                        "guided_hint_enabled": True,
                        "one_step_rows": [
                            {
                                "row_source": "analogical_template_application",
                                "evidence_class": "analogical_template_hint",
                                "allowed_use": "guided_search_hint_only",
                                "source_policy_decision": "analogical_guided_hint_only",
                                "used_as_proof": False,
                                "not_exact_literature_segment": True,
                                "not_parent_route_proof": True,
                                "requires_verifier": True,
                                "requires_parent_route_proof": True,
                                "production_write_blocked": True,
                                "no_solved_claim": True,
                                "template": {
                                    "evidence_class": "analogical_template_hint",
                                    "source_policy_decision": "analogical_guided_hint_only",
                                    "used_as_proof": False,
                                    "literature_template_trace": {
                                        "analogical_template_hint": True,
                                        "source_detail_exact_step": False,
                                        "structured_segment_step": False,
                                    },
                                },
                            }
                        ],
                        "plugin_flags": {
                            "enabled": False,
                            "guided_hint_enabled": True,
                            "one_step_rows": [],
                        },
                    },
                    "literature_template_plugin": {
                        "enabled": False,
                        "one_step_rows": [],
                        "plugin_flags": {"enabled": False, "one_step_rows": []},
                    },
                    "allowed_use": "guided_search_hint_only",
                    "analogy_is_advisory_only": True,
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "requires_parent_route_proof": True,
                    "production_write_blocked": True,
                    "no_solved_claim": True,
                },
                "compiled_downstream_refs": {"compiled_analogical_template_hints": "compiled_analogical_template_hints.json"},
                "evidence_class": "analogical_template_hint",
                "allowed_use": "guided_search_hint_only",
                "analogy_is_advisory_only": True,
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "requires_parent_route_proof": True,
                "production_write_blocked": True,
                "final_verdict_authority": "deterministic_parent_route_proof",
                "no_solved_claim": True,
            },
        )

        for artifact in (report, ranking, application, validation):
            self.assertEqual(artifact_json_round_trip(artifact).to_dict(), artifact.to_dict())
            self.assertTrue(validate_typed_artifact(artifact)["accepted"])

        bad = ARTIFACT_CLASSES["AnalogicalReactionTemplateRanking"](
            artifact_id="bad_tpl_ranking",
            case_id="case",
            source="unit_test",
            input_refs=["tpl_report"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_reaction_template_ranking.v1",
                "accepted": True,
                "selected_templates": [{"template_id": "tpl1", "route_status": "solved", "no_solved_claim": True}],
                "ranked_templates": [],
                "no_solved_claim": True,
            },
        )
        bad_result = validate_typed_artifact(bad)
        self.assertFalse(bad_result["accepted"])
        self.assertIn("analogical_template_ranking_direct_solved_claim", bad_result["reasons"])

        bad_validation = ARTIFACT_CLASSES["AnalogicalTemplateApplicationValidation"](
            artifact_id="bad_tpl_validation",
            case_id="case",
            source="unit_test",
            input_refs=["tpl_application"],
            validation_status="accepted",
            payload={
                "schema_version": "analogical_template_application_validation.v1",
                "accepted": True,
                "case_id": "case",
                "target_smiles": "CCO",
                "executable_candidate_count": 1,
                "one_step_row_count": 1,
                "compiled_downstream": {
                    "schema_version": "compiled_downstream_consumables.v1",
                    "literature_template_plugin": {
                        "enabled": True,
                        "one_step_rows": [{"source_policy_decision": "enabled_literature_template_plugin"}],
                        "plugin_flags": {
                            "enabled": True,
                            "one_step_rows": [{"source_policy_decision": "enabled_literature_template_plugin"}],
                        },
                    },
                },
                "compiled_downstream_refs": {"compiled_literature_template_plugin": "compiled_literature_template_plugin.json"},
                "evidence_class": "analogical_template_hint",
                "allowed_use": "guided_search_hint_only",
                "analogy_is_advisory_only": True,
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "requires_parent_route_proof": True,
                "production_write_blocked": True,
                "no_solved_claim": True,
            },
        )
        bad_validation_result = validate_typed_artifact(bad_validation)
        self.assertFalse(bad_validation_result["accepted"])
        self.assertIn(
            "analogical_template_validation_uses_exact_downstream_schema",
            bad_validation_result["reasons"],
        )
        self.assertIn(
            "analogical_template_guided_hints_enabled_exact_literature_plugin",
            bad_validation_result["reasons"],
        )

    def test_missing_required_artifact_fields_raise(self):
        with self.assertRaisesRegex(ValueError, "missing_artifact_id"):
            StructureProfile(
                artifact_id="",
                case_id="case",
                source="unit_test",
                input_refs=["input"],
            )

    def test_structure_validator_checks_smiles_canonical_inchikey_and_ambiguity(self):
        mol = Chem.MolFromSmiles("CCO")
        good = StructureProfile(
            artifact_id="profile",
            case_id="case",
            source="unit_test",
            input_refs=["target"],
            validation_status="accepted",
            payload={
                "target_smiles": "CCO",
                "canonical_smiles": "CCO",
                "inchi_key": MolToInchiKey(mol),
            },
        )
        bad = StructureProfile(
            artifact_id="bad_profile",
            case_id="case",
            source="unit_test",
            input_refs=["target"],
            validation_status="accepted",
            payload={
                "target_smiles": "FC(Cl)Br",
                "canonical_smiles": "CCO",
            },
        )

        self.assertTrue(validate_typed_artifact(good)["accepted"])
        result = validate_typed_artifact(bad)

        self.assertFalse(result["accepted"])
        self.assertIn("canonical_smiles_mismatch", result["reasons"])
        self.assertIn("target_ambiguity_not_marked", result["reasons"])

    def test_evidence_and_disconnection_refs_must_validate_in_order(self):
        evidence = {
            "artifact_type": "EvidenceCard",
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": "ev1",
            "case_id": "case",
            "source": "unit_test",
            "input_refs": ["literature_search_task"],
            "validation_status": "validated",
            "payload": {
                "evidence_id": "ev1",
                "case_id": "case",
                "source_type": "literature",
                "source_title": "Traceable route paper",
                "target_relation": "family_precedent",
                "claim_type": "strategic_disconnection",
                "route_role": "strategic_disconnection",
                "url": "https://example.org/route",
                "validation_status": "validated",
            },
        }
        disconnection = {
            "artifact_type": "StrategicDisconnectionCard",
            "schema_version": "strategic_disconnection_card.v1",
            "artifact_id": "sd1",
            "case_id": "case",
            "source": "unit_test",
            "evidence_refs": ["ev1"],
            "validation_status": "validated",
            "payload": {
                "evidence_refs": ["ev1"],
                "candidate_kind": "exact_fragment_retro",
                "retrosynthetic_move": {"break_bonds": ["C-C"]},
            },
        }

        summary = validate_artifact_list([evidence, disconnection])

        self.assertTrue(summary["accepted"], summary)
        self.assertEqual(summary["accepted_evidence_refs"], ["ev1"])
        self.assertEqual(summary["accepted_disconnection_refs"], ["sd1"])

    def test_route_status_validator_rejects_unproven_solved_claims(self):
        solved = {
            "artifact_type": "RouteStatus",
            "schema_version": "route_status_artifact.v1",
            "artifact_id": "status",
            "case_id": "case",
            "source": "unit_test",
            "input_refs": ["audit"],
            "validation_status": "accepted",
            "payload": {"route_status": "solved"},
        }
        semisynthesis = {
            **solved,
            "artifact_id": "semi",
            "payload": {"route_status": "semisynthesis_closed"},
        }

        solved_result = validate_typed_artifact(solved)
        semi_result = validate_typed_artifact(semisynthesis)

        self.assertFalse(solved_result["accepted"])
        self.assertIn("solved_without_stock_audit", solved_result["reasons"])
        self.assertFalse(semi_result["accepted"])
        self.assertIn("semisynthesis_closed_without_anchor_evidence", semi_result["reasons"])

    def test_statin_field_resolution_statuses_include_full_text_signal_candidates(self):
        self.assertTrue(valid_statin_field_resolution_candidate_status("full_text_signal_candidate_ready_for_curator"))
        self.assertTrue(valid_statin_field_resolution_candidate_status("full_text_signal_no_field_signal_ready_for_curator"))
        self.assertFalse(valid_statin_field_resolution_candidate_status("promotion_allowed"))


if __name__ == "__main__":
    unittest.main()
