import json
import unittest

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

from cascade_planner.agent.artifact_schemas import (
    ARTIFACT_CLASSES,
    StructureProfile,
    artifact_json_round_trip,
)
from cascade_planner.agent.artifact_validators import (
    validate_artifact_list,
    validate_typed_artifact,
)
from cascade_planner.agent.statin_panel import valid_statin_field_resolution_candidate_status


class AgentArtifactContractsTest(unittest.TestCase):
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
