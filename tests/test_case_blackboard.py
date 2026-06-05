import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.agent.case_trace import (
    ArtifactRecord,
    CaseBundle,
    FailureEvent,
    RouteStatus,
    failure_events_from_p0,
    load_case_bundle,
    load_p0_outputs_as_case_bundle,
    route_status_from_p0_validation,
    write_case_bundle,
)
from cascade_planner.agent.case_blackboard import (
    BlackboardArtifact,
    create_case,
    load_blackboard,
    write_blackboard,
)
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow


class CaseBlackboardTest(unittest.TestCase):
    def test_append_only_case_bundle_rejects_duplicate_artifact_id(self):
        bundle = CaseBundle(case_id="case")
        bundle.append_artifact(ArtifactRecord(
            artifact_id="target_profile",
            case_id="case",
            artifact_type="TargetProfile",
            payload={"target_smiles": "CCO"},
        ))

        with self.assertRaises(ValueError):
            bundle.append_artifact(ArtifactRecord(
                artifact_id="target_profile",
                case_id="case",
                artifact_type="TargetProfile",
                payload={"target_smiles": "CCO"},
            ))

    def test_rejected_artifact_is_not_returned_as_accepted(self):
        bundle = CaseBundle(case_id="case")
        bundle.append_artifact(ArtifactRecord(
            artifact_id="ok",
            case_id="case",
            artifact_type="EvidenceCard",
            payload={},
            validation_status="accepted",
        ))
        bundle.append_artifact(ArtifactRecord(
            artifact_id="bad",
            case_id="case",
            artifact_type="EvidenceCard",
            payload={},
            validation_status="rejected",
        ))

        accepted = bundle.accepted_artifacts("EvidenceCard")

        self.assertEqual([item.artifact_id for item in accepted], ["ok"])

    def test_json_round_trip_preserves_artifacts_and_failure_events(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.UNRESOLVED)
        bundle.append_artifact(ArtifactRecord(
            artifact_id="validation",
            case_id="case",
            artifact_type="RoutePackageValidation",
            payload={"route_status": "literature_gap"},
        ))
        bundle.append_failure_event(FailureEvent(
            failure_id="gap",
            case_id="case",
            reason="literature_gap",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case_bundle.json"
            write_case_bundle(bundle, path)
            loaded = load_case_bundle(path)

        self.assertEqual(loaded.case_id, "case")
        self.assertEqual(loaded.route_status, RouteStatus.UNRESOLVED)
        self.assertEqual(loaded.artifacts[0].artifact_id, "validation")
        self.assertEqual(loaded.failure_events[0].reason, "literature_gap")

    def test_p0_workflow_exports_importable_case_bundle(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="bufadienolide_trace_case",
                    family_hint="bufadienolide, steroid, pyrone",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=4,
                    literature_backend="local",
                )
            )
            bundle_path = Path(result["artifacts"]["case_bundle"])
            loaded = load_case_bundle(bundle_path)
            rebuilt = load_p0_outputs_as_case_bundle(tmp)

        self.assertEqual(loaded.route_status, RouteStatus.PARTIAL_ANCHOR)
        self.assertEqual(rebuilt.route_status, RouteStatus.PARTIAL_ANCHOR)
        self.assertGreaterEqual(len(loaded.accepted_artifacts()), 4)
        self.assertIn("HybridRoutePackage", {item.artifact_type for item in loaded.artifacts})
        self.assertIn("EvidenceCardList", {item.artifact_type for item in loaded.artifacts})
        self.assertTrue(any(event.reason == "unresolved_core" for event in loaded.failure_events))

    def test_route_package_audit_result_maps_to_route_status(self):
        self.assertEqual(
            route_status_from_p0_validation({"accepted": True, "route_status": "partial_anchor"}),
            RouteStatus.PARTIAL_ANCHOR,
        )
        self.assertEqual(
            route_status_from_p0_validation({"accepted": True, "route_status": "literature_gap"}),
            RouteStatus.UNRESOLVED,
        )
        self.assertEqual(
            route_status_from_p0_validation({"accepted": False, "route_status": "invalid_package"}),
            RouteStatus.FAKE_CLOSED_REJECTED,
        )

    def test_fake_closure_and_literature_gap_are_traceable_failure_events(self):
        route_package = {
            "case_id": "case",
            "route_status": "invalid_package",
            "frontier": {
                "frontier_smiles": "CCO",
                "flags": ["advanced_same_scaffold", "ordinary_decoration_only"],
            },
        }
        validation = {
            "case_id": "case",
            "accepted": False,
            "route_status": "invalid_package",
            "reasons": ["route_anchor_has_rxn"],
        }

        events = failure_events_from_p0(validation, route_package)
        reasons = {event.reason for event in events}

        self.assertIn("route_anchor_has_rxn", reasons)
        self.assertIn("advanced_same_scaffold", reasons)
        self.assertIn("ordinary_decoration_only", reasons)

    def test_case_bundle_json_contains_route_status_string(self):
        bundle = CaseBundle(case_id="case", route_status=RouteStatus.PARTIAL_ANCHOR)
        data = bundle.to_dict()

        self.assertEqual(json.loads(json.dumps(data))["route_status"], "partial_anchor")

    def test_case_blackboard_create_append_query_reject_and_round_trip(self):
        board = create_case("case", target={"smiles": "CCO"})
        artifact = BlackboardArtifact(
            artifact_id="route_1_step_1",
            case_id="case",
            artifact_type="RouteCandidate",
            payload={"route_id": "route_1", "step_id": "step_1", "molecule_id": "mol_1"},
            source="unit_test",
            trace_id="trace_1",
            run_id="run_1",
            parent_refs=["target_profile"],
            evidence_refs=["ev1"],
            route_id="route_1",
            step_id="step_1",
            molecule_id="mol_1",
        )
        board.append_artifact(artifact)

        self.assertEqual(board.current_summary()["accepted_artifact_counts"]["RouteCandidate"], 1)
        self.assertEqual(board.artifacts_by_type("RouteCandidate")[0].artifact_id, "route_1_step_1")
        self.assertEqual(board.artifacts_by_route_id("route_1")[0].step_id, "step_1")
        self.assertEqual(board.artifacts_by_step_id("step_1")[0].molecule_id, "mol_1")
        self.assertEqual(board.artifacts_by_molecule_id("mol_1")[0].route_id, "route_1")
        self.assertEqual(board.accepted_artifacts()[0].parent_refs, ["target_profile"])

        board.reject_artifact("route_1_step_1", trace_id="trace_2", reasons=["validator_reject"])
        self.assertEqual(board.accepted_artifacts(), [])
        self.assertEqual(board.rejections[0].reasons, ["validator_reject"])
        self.assertIn("trace_2", board.current_summary()["trace_ids"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blackboard.json"
            write_blackboard(board, path)
            loaded = load_blackboard(path)

        self.assertEqual(loaded.case_id, "case")
        self.assertEqual(loaded.target["smiles"], "CCO")
        self.assertEqual(loaded.artifacts[0].validation_status, "rejected")
        self.assertEqual(loaded.rejections[0].artifact_id, "route_1_step_1")


if __name__ == "__main__":
    unittest.main()
