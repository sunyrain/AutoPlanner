import unittest

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.literature_segments import (
    LiteratureRouteSegmentCard,
    SegmentStepCandidate,
    unroll_literature_route_segment,
    validate_literature_route_segment,
    validate_segment_step,
)


class LiteratureSegmentsTest(unittest.TestCase):
    def test_three_step_exact_segment_recursively_unrolls(self):
        segment = _exact_segment()

        validation = validate_literature_route_segment(segment)
        trace = unroll_literature_route_segment(segment)

        self.assertTrue(validation["accepted"], validation)
        self.assertTrue(validation["allowed_for_recursive_unroll"])
        self.assertEqual(trace["final_status"], "segment_unrolled")
        self.assertEqual(len(trace["expanded_steps"]), 3)
        self.assertEqual(trace["stop_reason"], "complete")

    def test_analog_segment_is_downgraded_not_executable(self):
        segment = _exact_segment()
        segment.steps[1].relation_type = "analog"
        segment.steps[1].scope_gap = "same reaction family, different substrate oxidation state"

        validation = validate_literature_route_segment(segment)
        trace = unroll_literature_route_segment(segment)

        self.assertFalse(validation["accepted"])
        self.assertIn("analog_segment_not_executable", validation["reasons"])
        self.assertEqual(trace["final_status"], "partial")
        self.assertEqual(trace["stop_reason"], "audit_failed")

    def test_mismatch_and_high_risk_condition_stop_unroll(self):
        mismatch = _step("bad", "CCO")
        mismatch.relation_type = "mismatch"
        high_risk = _step("risk", "CCO")
        high_risk.condition_candidate["risk_flags"] = ["extreme_temperature"]
        high_risk.condition_candidate["temperature"] = "-80 C"

        mismatch_result = validate_segment_step(mismatch)
        high_risk_trace = unroll_literature_route_segment(
            LiteratureRouteSegmentCard(
                segment_id="risk_segment",
                case_id="case",
                target_smiles="CCO",
                steps=[high_risk, _step("after", "CCO")],
                evidence_refs=["ev_segment"],
            )
        )

        self.assertFalse(mismatch_result["accepted"])
        self.assertIn("segment_step_mismatch", mismatch_result["reasons"])
        self.assertEqual(high_risk_trace["final_status"], "rejected")
        self.assertEqual(high_risk_trace["stop_reason"], "high_risk_condition")

    def test_native_solved_negative_control_skips_segment_uplift(self):
        trace = unroll_literature_route_segment(_exact_segment(), native_solved_audit_passed=True)

        self.assertEqual(trace["final_status"], "skipped")
        self.assertEqual(trace["stop_reason"], "native_solved_audit_passed")
        self.assertTrue(trace["false_uplift_blocked"])

    def test_segment_typed_artifact_validator_rejects_bad_payload(self):
        artifact = {
            "schema_version": "literature_route_segment_card.v1",
            "artifact_id": "seg_artifact",
            "artifact_type": "LiteratureRouteSegmentCard",
            "case_id": "case",
            "source": "unit_test",
            "input_refs": ["ev_segment"],
            "evidence_refs": ["ev_segment"],
            "validation_status": "draft",
            "payload": _exact_segment().to_dict(),
        }
        bad = dict(artifact)
        bad["payload"] = {**artifact["payload"], "steps": []}

        self.assertTrue(validate_typed_artifact(artifact)["accepted"])
        rejected = validate_typed_artifact(bad)
        self.assertFalse(rejected["accepted"])
        self.assertIn("segment_step_count_out_of_range", rejected["reasons"])


def _exact_segment() -> LiteratureRouteSegmentCard:
    return LiteratureRouteSegmentCard(
        segment_id="exact_three_step",
        case_id="case",
        target_smiles="CCO",
        source_title="Traceable SI route segment",
        trigger_reasons=["advanced_frontier_detected"],
        evidence_refs=["ev_segment"],
        steps=[
            _step("seg_1", "CCO"),
            _step("seg_2", "CCO"),
            _step("seg_3", "CCO"),
        ],
    )


def _step(step_id: str, product_smiles: str) -> SegmentStepCandidate:
    return SegmentStepCandidate(
        step_id=step_id,
        product_smiles=product_smiles,
        reactant_smiles=[product_smiles],
        evidence_refs=["ev_segment"],
        source_ref="doi:10.0000/example-si",
        applicability={
            "status": "passed",
            "product_reconstruction_passed": True,
            "reconstructed_product_smiles": product_smiles,
        },
        condition_candidate={
            "step_id": step_id,
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "solvent": "MeCN",
            "temperature": "25 C",
            "evidence_refs": ["ev_segment"],
        },
    )


if __name__ == "__main__":
    unittest.main()
