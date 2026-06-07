import unittest

from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
    validate_evolution_candidate,
)
from cascade_planner.agent.literature_segments import LiteratureRouteSegmentCard, SegmentStepCandidate
from cascade_planner.harness.self_evo_memory import compile_self_evo_memory
from cascade_planner.harness.self_evo_replay import run_self_evo_replay_gate


class EvolutionHardeningTest(unittest.TestCase):
    def test_evolution_candidate_accepts_valid_segment_and_condition_types(self):
        segment_candidate = EvolutionCandidate(
            candidate_id="seg_candidate",
            candidate_type="LiteratureRouteSegmentCard",
            payload=_segment_payload(),
            evidence_refs=["ev_segment"],
            validation_status="validated",
        )
        condition_candidate = EvolutionCandidate(
            candidate_id="cond_candidate",
            candidate_type="ConditionCandidate",
            payload={
                "step_id": "step_1",
                "source_type": "exact",
                "condition_status": "evidence_backed",
                "solvent": "MeCN",
                "evidence_refs": ["ev_cond"],
            },
            evidence_refs=["ev_cond"],
            validation_status="validated",
        )

        self.assertTrue(validate_evolution_candidate(segment_candidate)["accepted"])
        self.assertTrue(validate_evolution_candidate(condition_candidate)["accepted"])

    def test_bad_segment_and_bad_condition_are_rejected_before_layer_write(self):
        bad_segment = EvolutionCandidate(
            candidate_id="bad_seg",
            candidate_type="LiteratureRouteSegmentCard",
            payload={**_segment_payload(), "steps": []},
            evidence_refs=["ev_segment"],
            validation_status="validated",
        )
        bad_condition = EvolutionCandidate(
            candidate_id="bad_cond",
            candidate_type="ConditionCandidate",
            payload={"step_id": "step_1", "source_type": "exact", "solvent": "MeCN"},
            evidence_refs=["ev_cond"],
            validation_status="validated",
        )

        seg_validation = validate_evolution_candidate(bad_segment)
        cond_validation = validate_evolution_candidate(bad_condition)

        self.assertFalse(seg_validation["accepted"])
        self.assertIn("segment:segment_step_count_out_of_range", seg_validation["reasons"])
        self.assertFalse(cond_validation["accepted"])
        self.assertIn("condition:exact_condition_missing_evidence", cond_validation["reasons"])

    def test_target_run_cannot_promote_segment_to_production_and_gate_covers_segment_condition(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="seg_candidate",
            candidate_type="LiteratureRouteSegmentCard",
            payload=_segment_payload(),
            evidence_refs=["ev_segment"],
            validation_status="validated",
        )
        kb.add_candidate(candidate, target_run=True)
        kb.promote("seg_candidate", from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote("seg_candidate", from_layer="shadow", to_layer="staging", target_run=True)

        bad_gate = evaluate_benchmark_gate({
            "segment_replay_passes": False,
            "condition_quality_passes": False,
        })
        good_gate = evaluate_benchmark_gate({
            "true_solved_rate_delta": 0.0,
            "fake_closure_rate_delta": 0.0,
            "condition_quality_delta": 0.0,
            "segment_replay_passes": True,
            "condition_quality_passes": True,
        })

        self.assertFalse(bad_gate.accepted)
        self.assertIn("segment_replay_failed", bad_gate.reasons)
        self.assertIn("condition_quality_failed", bad_gate.reasons)
        with self.assertRaisesRegex(ValueError, "target_run_cannot_write_production"):
            kb.promote("seg_candidate", from_layer="staging", to_layer="production", gate_report=good_gate, target_run=True)
        with self.assertRaisesRegex(ValueError, "benchmark_gate_failed"):
            kb.promote("seg_candidate", from_layer="staging", to_layer="production", gate_report=bad_gate)

        kb.promote("seg_candidate", from_layer="staging", to_layer="production", gate_report=good_gate)
        self.assertIn("seg_candidate", kb.layers["production"])
        kb.rollback("seg_candidate", layer="production")
        self.assertNotIn("seg_candidate", kb.layers["production"])
        self.assertEqual(kb.history[-1]["event"], "rollback")
        self.assertTrue(kb.history[-1]["removed"])

    def test_harness_self_evo_replay_blocks_target_run_and_promotes_cross_case_gate(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="template_candidate",
            candidate_type="TemplateCandidate",
            payload={"schema_version": "template_candidate.v1", "not_raw_reaction_injection": True},
            evidence_refs=["ev_template"],
            validation_status="validated",
        )
        kb.add_candidate(candidate, target_run=True)
        kb.promote("template_candidate", from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote("template_candidate", from_layer="shadow", to_layer="staging", target_run=True)
        staging = {"schema_version": "self_evo_staging_compile_report.v1", "kb": kb.to_dict()}
        metrics = {
            "true_solved_rate_delta": 0.0,
            "fake_closure_rate_delta": 0.0,
            "condition_quality_delta": 0.0,
            "template_replay_passes": True,
            "structure_validated": True,
            "evidence_source_credible": True,
            "role_assignment_checked": True,
        }

        target_report = run_self_evo_replay_gate(
            staging,
            replay_metrics=metrics,
            target_run=True,
            allow_production=True,
        )
        cross_case_report = run_self_evo_replay_gate(
            staging,
            replay_metrics=metrics,
            target_run=False,
            allow_production=True,
        )

        self.assertTrue(target_report["accepted"])
        self.assertTrue(target_report["production_write_blocked"])
        self.assertEqual(target_report["production_promoted_count"], 0)
        self.assertTrue(cross_case_report["accepted"])
        self.assertFalse(cross_case_report["production_write_blocked"])
        self.assertEqual(cross_case_report["production_promoted_count"], 1)
        self.assertIn("template_candidate", cross_case_report["kb"]["layers"]["production"])

    def test_self_evo_memory_compiles_replay_accepted_staging_assets_for_future_runs(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="template_candidate",
            candidate_type="TemplateCandidate",
            payload={
                "template_id": "statin_side_chain_template",
                "reaction_class": "statin_side_chain_convergence",
                "template_card": {
                    "schema_version": "literature_template_card.v1",
                    "template_id": "statin_side_chain_template",
                    "validation_status": "draft",
                    "template_level": "advisory_strategy",
                    "reaction_class": "statin_side_chain_convergence",
                    "product_retron": {"retron_type": "statin_heptenoate_side_chain"},
                    "evidence_refs": ["ev_template"],
                    "not_raw_reaction_injection": True,
                },
                "route_expansion_task": {
                    "schema_version": "compiled_route_expansion_task.v1",
                    "task_id": "statin_expand_1",
                    "evidence_refs": ["ev_template"],
                    "preferred_reaction_classes": ["statin_side_chain_convergence"],
                    "not_raw_reaction_injection": True,
                },
            },
            evidence_refs=["ev_template"],
            validation_status="validated",
        )
        kb.add_candidate(candidate, target_run=True)
        kb.promote("template_candidate", from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote("template_candidate", from_layer="shadow", to_layer="staging", target_run=True)
        replay = run_self_evo_replay_gate(
            {"schema_version": "self_evo_staging_compile_report.v1", "kb": kb.to_dict()},
            replay_metrics={
                "template_replay_passes": True,
                "structure_validated": True,
                "evidence_source_credible": True,
                "role_assignment_checked": True,
            },
            target_run=True,
            allow_production=False,
        )

        memory = compile_self_evo_memory(replay, case_id="statin_case")

        self.assertTrue(memory["accepted"], memory["reasons"])
        self.assertTrue(memory["production_write_blocked"])
        self.assertEqual(memory["production_promoted_count"], 0)
        self.assertEqual(memory["reusable_template_cards"][0]["template_id"], "statin_side_chain_template")
        self.assertEqual(memory["reusable_route_expansion_tasks"][0]["task_id"], "statin_expand_1")
        self.assertTrue(memory["future_use_policy"]["not_route_evidence_until_current_target_relation_checked"])

    def test_self_evo_memory_keeps_executable_template_extraction_tasks_from_compiled_downstream(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="template_candidate",
            candidate_type="TemplateCandidate",
            payload={
                "template_id": "statin_side_chain_template",
                "reaction_class": "statin_side_chain_convergence",
                "template_card": {
                    "schema_version": "literature_template_card.v1",
                    "template_id": "statin_side_chain_template",
                    "validation_status": "draft",
                    "template_level": "advisory_strategy",
                    "reaction_class": "statin_side_chain_convergence",
                    "product_retron": {"retron_type": "statin_heptenoate_side_chain"},
                    "evidence_refs": ["ev_template"],
                    "not_raw_reaction_injection": True,
                },
            },
            evidence_refs=["ev_template"],
            validation_status="validated",
        )
        kb.add_candidate(candidate, target_run=True)
        kb.promote("template_candidate", from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote("template_candidate", from_layer="shadow", to_layer="staging", target_run=True)
        replay = run_self_evo_replay_gate(
            {"schema_version": "self_evo_staging_compile_report.v1", "kb": kb.to_dict()},
            replay_metrics={
                "template_replay_passes": True,
                "structure_validated": True,
                "evidence_source_credible": True,
                "role_assignment_checked": True,
            },
            target_run=True,
            allow_production=False,
        )
        compiled = {
            "schema_version": "compiled_downstream_consumables.v1",
            "literature_template_plugin": {"template_cards": [], "one_step_rows": []},
            "route_expansion": {"tasks": []},
            "executable_template_maturity": {
                "schema_version": "executable_template_maturity.v1",
                "status": "needs_structured_extraction",
                "extraction_tasks": [
                    {
                        "schema_version": "compiled_executable_template_extraction_task.v1",
                        "task_id": "extract_statin_side_chain_step",
                        "source_title": "Statin side-chain process",
                        "reaction_class": "statin_side_chain_convergence",
                        "evidence_refs": ["ev_template"],
                        "required_structured_fields": ["product_smiles", "reactant_smiles"],
                        "precursor_roles": ["beta-keto ester side-chain precursor"],
                        "not_raw_reaction_injection": True,
                    }
                ],
            },
        }

        memory = compile_self_evo_memory(replay, compiled_downstream=compiled, case_id="statin_case")

        self.assertTrue(memory["accepted"], memory["reasons"])
        self.assertEqual(
            memory["reusable_executable_template_extraction_tasks"][0]["task_id"],
            "extract_statin_side_chain_step",
        )
        self.assertTrue(any(row["hint_type"] == "executable_template_extraction_task" for row in memory["query_hints"]))
        self.assertIn("executable_template_extraction_task", memory["future_use_policy"]["allowed_use"])

    def test_harness_self_evo_replay_rejects_bad_replay_metrics(self):
        kb = LayeredKnowledgeBase()
        candidate = EvolutionCandidate(
            candidate_id="template_candidate",
            candidate_type="TemplateCandidate",
            payload={"schema_version": "template_candidate.v1", "not_raw_reaction_injection": True},
            evidence_refs=["ev_template"],
            validation_status="validated",
        )
        kb.add_candidate(candidate, target_run=True)
        kb.promote("template_candidate", from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote("template_candidate", from_layer="shadow", to_layer="staging", target_run=True)

        report = run_self_evo_replay_gate(
            {"schema_version": "self_evo_staging_compile_report.v1", "kb": kb.to_dict()},
            replay_metrics={"fake_closure_rate_delta": 0.1, "template_replay_passes": False},
            target_run=False,
            allow_production=True,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(report["production_write_blocked"])
        self.assertEqual(report["production_promoted_count"], 0)
        self.assertIn("fake_closure_rate_regressed", report["reasons"])
        self.assertIn("template_replay_failed", report["reasons"])


def _segment_payload() -> dict:
    step = SegmentStepCandidate(
        step_id="seg_1",
        product_smiles="CCO",
        reactant_smiles=["CCO"],
        evidence_refs=["ev_segment"],
        source_ref="doi:10.0000/example-si",
        applicability={
            "status": "passed",
            "product_reconstruction_passed": True,
            "reconstructed_product_smiles": "CCO",
        },
        condition_candidate={
            "step_id": "seg_1",
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "solvent": "MeCN",
            "evidence_refs": ["ev_segment"],
        },
    )
    return LiteratureRouteSegmentCard(
        segment_id="seg_candidate_payload",
        case_id="case",
        target_smiles="CCO",
        steps=[step, SegmentStepCandidate(**{**step.to_dict(), "step_id": "seg_2"})],
        evidence_refs=["ev_segment"],
        source_title="Traceable SI route segment",
    ).to_dict()


if __name__ == "__main__":
    unittest.main()
