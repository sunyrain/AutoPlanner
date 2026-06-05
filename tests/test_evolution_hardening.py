import unittest

from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
    validate_evolution_candidate,
)
from cascade_planner.agent.literature_segments import LiteratureRouteSegmentCard, SegmentStepCandidate


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
