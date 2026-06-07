import unittest

from cascade_planner.agent.condition_agent import (
    ConditionCandidate,
    audit_conditions,
    benchmark_condition_candidates,
    condition_worker_task_from_route_step,
    condition_worker_task_to_worker_task,
    validate_condition_candidate,
)
from cascade_planner.agent.route_auditor import (
    audit_route_package,
    validate_route_audit_report,
)


class AgentRouteAuditorConditionTest(unittest.TestCase):
    def test_condition_candidate_validation_and_gap_audit(self):
        exact = ConditionCandidate(
            step_id="step_1",
            source_type="exact",
            solvent="MeCN",
            temperature="25 C",
            evidence_refs=["ev_cond"],
        )
        missing = ConditionCandidate(step_id="step_2", source_type="unknown")

        self.assertTrue(validate_condition_candidate(exact)["accepted"])
        missing_result = validate_condition_candidate(missing)
        audit = audit_conditions([exact, missing])

        self.assertFalse(missing_result["accepted"])
        self.assertIn("condition_gap", missing_result["reasons"])
        self.assertEqual(audit["route_risk"], "gap")
        self.assertTrue(audit["condition_gap"])

    def test_risky_condition_flags_high_route_risk(self):
        audit = audit_conditions([
            ConditionCandidate(
                step_id="step_1",
                source_type="model-only",
                temperature="-80 C",
                risk_flags=["extreme_temperature"],
            )
        ])

        self.assertEqual(audit["route_risk"], "high")
        self.assertIn("extreme_temperature", audit["risk_flags"])

    def test_exact_condition_source_takes_priority_over_model_prediction(self):
        audit = audit_conditions([
            ConditionCandidate(
                step_id="step_1",
                source_type="model-only",
                solvent="predicted solvent",
            ),
            ConditionCandidate(
                step_id="step_1",
                source_type="exact",
                solvent="MeCN",
                evidence_refs=["ev_exact_condition"],
            ),
        ])

        self.assertEqual(audit["best_source_type"], "exact")
        self.assertFalse(audit["condition_gap"])

    def test_partial_anchor_requires_no_solved_claim_without_stock_audit(self):
        package = _package(route_status="partial_anchor")
        report = audit_route_package(package, validation={"accepted": True, "route_status": "partial_anchor"})

        self.assertEqual(report.route_status, "partial_anchor")
        self.assertEqual(report.evidence_status, "anchor_evidence_present")
        self.assertFalse(report.stock_audit_passed)
        self.assertIn("anchor_evidence_without_full_stock_closure", report.reasons)
        self.assertTrue(validate_route_audit_report(report)["accepted"])

    def test_solved_claim_without_stock_audit_is_downgraded_to_unresolved(self):
        package = _package(route_status="solved")
        package["literature_candidates"] = []
        report = audit_route_package(
            package,
            validation={"accepted": True, "route_status": "solved"},
            stock_audit_passed=False,
            condition_candidates=[{"step_id": "step_1", "source_type": "exact", "solvent": "MeOH", "evidence_refs": ["ev"]}],
        )

        self.assertEqual(report.route_status, "unresolved")
        self.assertIn("solved_claim_without_stock_audit", report.reasons)
        self.assertTrue(validate_route_audit_report(report)["accepted"])

    def test_stock_audit_plus_condition_evidence_can_mark_solved(self):
        package = _package(route_status="solved")
        package["literature_candidates"] = []
        report = audit_route_package(
            package,
            validation={"accepted": True, "route_status": "solved"},
            stock_audit_passed=True,
            condition_candidates=[{"step_id": "step_1", "source_type": "exact", "solvent": "MeOH", "evidence_refs": ["ev"]}],
        )

        self.assertEqual(report.route_status, "solved")
        self.assertTrue(report.stock_audit_passed)
        self.assertTrue(validate_route_audit_report(report)["accepted"])

    def test_condition_gap_marks_conditions_pending_without_invalidating_stock_closure(self):
        package = _package(route_status="solved")
        package["literature_candidates"] = []
        report = audit_route_package(
            package,
            validation={"accepted": True, "route_status": "solved"},
            stock_audit_passed=True,
            condition_candidates=[],
        )

        self.assertEqual(report.route_status, "solved")
        self.assertEqual(report.condition_status, "condition_gap")
        self.assertEqual(report.next_action, "attach_or_retrieve_conditions")
        self.assertFalse(report.reasons)

    def test_invalid_package_generates_fake_closed_rejected_with_terminal_list(self):
        package = _package(route_status="invalid_package")
        validation = {
            "accepted": False,
            "route_status": "invalid_package",
            "reasons": ["route_anchor_has_rxn"],
        }

        report = audit_route_package(package, validation=validation)

        self.assertEqual(report.route_status, "fake_closed_rejected")
        self.assertTrue(report.fake_closure_rejected)
        self.assertIn("route_anchor_has_rxn", report.reasons)
        self.assertTrue(report.rejected_terminal_list)
        self.assertTrue(validate_route_audit_report(report)["accepted"])

    def test_semisynthesis_closed_requires_anchor_evidence(self):
        package = _package(route_status="semisynthesis_closed")
        report = audit_route_package(
            package,
            validation={"accepted": True, "route_status": "semisynthesis_closed"},
            stock_audit_passed=True,
            condition_candidates=[{
                "step_id": "step_1",
                "source_type": "analog",
                "solvent": "EtOAc",
                "scope_gap": "analog substrate; no exact target procedure",
            }],
        )
        no_anchor = report.to_dict()
        no_anchor["evidence_status"] = "unknown"

        self.assertEqual(report.route_status, "semisynthesis_closed")
        self.assertTrue(validate_route_audit_report(report)["accepted"])
        bad = validate_route_audit_report(no_anchor)
        self.assertFalse(bad["accepted"])
        self.assertIn("semisynthesis_closed_without_anchor_evidence", bad["reasons"])

    def test_high_risk_condition_downgrades_stock_closed_route(self):
        package = _package(route_status="solved")
        package["literature_candidates"] = []
        report = audit_route_package(
            package,
            validation={"accepted": True, "route_status": "solved"},
            stock_audit_passed=True,
            condition_candidates=[{
                "step_id": "step_1",
                "source_type": "exact",
                "solvent": "THF",
                "temperature": "-80 C",
                "evidence_refs": ["ev_cond"],
                "risk_flags": ["extreme_temperature"],
            }],
        )

        self.assertEqual(report.route_status, "unresolved")
        self.assertEqual(report.condition_status, "condition_high_risk")
        self.assertIn("condition_high_risk", report.reasons)

    def test_condition_worker_task_mapping_and_benchmark_report(self):
        task = condition_worker_task_from_route_step(
            case_id="case",
            route_step={"step_id": "step_7", "product_smiles": "CCO"},
            evidence_refs=["ev_cond"],
        )
        worker_task = condition_worker_task_to_worker_task(task)
        report = benchmark_condition_candidates([
            {
                "case_id": "exact",
                "candidates": [{"step_id": "s1", "source_type": "exact", "solvent": "MeCN", "evidence_refs": ["ev"]}],
            },
            {
                "case_id": "analog",
                "candidates": [{
                    "step_id": "s2",
                    "source_type": "analog",
                    "solvent": "EtOAc",
                    "scope_gap": "analog substrate",
                }],
            },
            {
                "case_id": "model",
                "expected_gap": False,
                "expected_risky": True,
                "candidates": [{
                    "step_id": "s3",
                    "source_type": "model-only",
                    "temperature": "-80 C",
                    "risk_flags": ["extreme_temperature"],
                }],
            },
        ])

        self.assertEqual(task.schema_version, "condition_worker_task.v1")
        self.assertEqual(worker_task.task_type, "condition_research")
        self.assertEqual(worker_task.required_artifact_type, "ConditionCandidate")
        self.assertEqual(report["case_count"], 3)
        self.assertGreaterEqual(report["audit_downgrade_count"], 1)
        self.assertEqual(report["risky_flag_precision"], 1.0)


def _package(route_status: str) -> dict:
    return {
        "case_id": "case",
        "route_status": route_status,
        "target": {"smiles": "CCO"},
        "frontier": {
            "frontier_smiles": "CCO",
            "flags": ["advanced_same_scaffold", "no_complexity_drop", "unresolved_core"],
        },
        "literature_evidence_refs": ["ev_anchor"],
        "literature_candidates": [
            {
                "candidate_id": "anchor",
                "candidate_kind": "route_anchor",
                "evidence_refs": ["ev_anchor"],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
