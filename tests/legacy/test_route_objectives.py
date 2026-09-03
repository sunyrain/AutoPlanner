import hashlib
import json
import os
from pathlib import Path
import unittest

from cascade_planner.legacy.harness_runtime.parent_route_proof import (
    compile_stitched_parent_route_proof,
)
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.legacy.harness_runtime.route_objectives import (
    build_broad_transform_templates_from_blackboard,
    classify_route_objectives,
    compile_route_objective_proof_bundle,
)


C22_STEROID_LIKE = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
ATORVASTATIN_FREE_ACID = (
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
    "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_SOURCE_PDF = _FIXTURES / "source_evidence_stub.pdf"
_SOURCE_IMAGE = _FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST = _FIXTURES / "source_evidence_manifest.json"
_TRUSTED_REGISTRY = _FIXTURES / "trusted_literature_step_registry.json"


def _strict_evidence_fields() -> dict:
    template_id = "source_detail_exact_step:ethanol_hydration"
    return {
        "step_id": "ethanol_hydration",
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST.resolve()),
                "manifest_sha256": hashlib.sha256(_SOURCE_MANIFEST.read_bytes()).hexdigest(),
                "source_pdf_path": str(_SOURCE_PDF.resolve()),
                "source_pdf_sha256": hashlib.sha256(_SOURCE_PDF.read_bytes()).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_IMAGE.resolve()),
                "image_sha256": hashlib.sha256(_SOURCE_IMAGE.read_bytes()).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


class RouteObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prior_registry = os.environ.get("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY")
        os.environ["AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY"] = str(_TRUSTED_REGISTRY)

    @classmethod
    def tearDownClass(cls):
        if cls._prior_registry is None:
            os.environ.pop("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY", None)
        else:
            os.environ["AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY"] = cls._prior_registry

    def test_small_molecule_keeps_stock_closure_as_top_objective(self):
        summary = classify_route_objectives(
            target_smiles="CCO",
            target_name="ethanol",
            case_id="ethanol",
        )

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["selected_objectives"][0]["objective_type"], "small_molecule_stock_closure")
        self.assertFalse(summary["route_scope"]["small_molecule_stock_closure_deprioritized"])
        self.assertFalse(summary["route_scope"]["objective_evidence_validation_required"])

    def test_complex_polycyclic_failure_deprioritizes_stock_without_hardcoded_source(self):
        summary = classify_route_objectives(
            target_smiles=C22_STEROID_LIKE,
            target_name="complex polycyclic target",
            family_hint="steroid natural product like scaffold",
            failure_reasons=["large_atom_jump"],
            case_id="complex",
        )
        selected = {row["objective_type"] for row in summary["selected_objectives"]}
        payload = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertTrue(summary["route_scope"]["small_molecule_stock_closure_deprioritized"])
        self.assertIn("advanced_intermediate_anchor", selected)
        self.assertIn("semisynthesis_from_natural_product", selected)
        self.assertIn("biotransformation_endpoint", selected)
        self.assertNotIn("10.1186/s12934-021-01717-w", payload)
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_statin_process_target_prefers_literature_and_advanced_intermediate_objectives(self):
        summary = classify_route_objectives(
            target_smiles=ATORVASTATIN_FREE_ACID,
            target_name="atorvastatin",
            family_hint="statin synthetic atorvastatin Paal-Knorr",
            case_id="atorvastatin",
        )
        selected = {row["objective_type"] for row in summary["selected_objectives"]}
        flags = summary["target"]["features"]["flags"]

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertTrue(flags["statin_process_like"])
        self.assertFalse(flags["steroid_like_polycyclic_scaffold"])
        self.assertFalse(flags["natural_product_like"])
        self.assertIn("advanced_intermediate_anchor", selected)
        self.assertIn("literature_known_scaffold_anchor", selected)

    def test_broad_templates_are_advisory_not_proof(self):
        summary = classify_route_objectives(
            target_smiles=C22_STEROID_LIKE,
            target_name="complex polycyclic target",
            family_hint="steroid",
            case_id="complex",
        )
        blackboard = {
            "case_id": "complex",
            "target_profile": {"target_smiles": C22_STEROID_LIKE},
            "route_objective_summary": summary,
            "endpoint_candidates": summary["endpoint_candidates"],
            "target_side_disconnection_hypotheses": {
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "target_handle": "semisynthesis_or_biotransformation_anchor",
                        "proposed_disconnection_region": "same-core late-stage oxidation",
                        "must_preserve_substructure": ["largest_polycyclic_core"],
                        "expected_precursor_type": "same-core redox precursor",
                        "required_verification": ["same_core_identity"],
                        "risk_flags": ["hypothesis_only"],
                    }
                ]
            },
        }

        report = build_broad_transform_templates_from_blackboard(blackboard)
        payload = json.dumps(report, sort_keys=True)

        self.assertTrue(report["accepted"], report["reasons"])
        self.assertGreaterEqual(report["template_count"], 1)
        self.assertTrue(all(row["not_parent_route_proof"] for row in report["templates"]))
        self.assertTrue(all(row["no_solved_claim"] for row in report["templates"]))
        self.assertNotIn(">>", payload)

    def test_objective_proof_distinguishes_plausible_from_solved(self):
        summary = classify_route_objectives(
            target_smiles=C22_STEROID_LIKE,
            target_name="complex polycyclic target",
            family_hint="steroid",
            case_id="complex",
        )
        blackboard = {
            "case_id": "complex",
            "route_objective_summary": summary,
            "literature_evidence": {
                "source_candidates": [{"source_ref": "doi:example", "doi": "10.0000/example"}]
            },
            "broad_transform_templates": [
                {
                    "schema_version": "broad_transform_template.v1",
                    "template_id": "broad_template:test",
                    "not_parent_route_proof": True,
                    "no_solved_claim": True,
                }
            ],
        }

        plausible = compile_route_objective_proof_bundle(blackboard=blackboard)
        simple_summary = classify_route_objectives(
            target_smiles="CCO",
            target_name="ethanol",
            case_id="ethanol",
        )
        solved_blackboard = {
            "case_id": "ethanol",
            "target_profile": {"target_smiles": "CCO"},
            "route_objective_summary": simple_summary,
        }
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {
                            "terminal_reactants": ["CC", "O"],
                            "terminal_stock_status": {"CC": True, "O": True},
                        },
                        "steps": [
                            {
                                **_strict_evidence_fields(),
                                "product": "CCO",
                                "reactant_smiles": ["CC", "O"],
                                "stock_status": {"CC": True, "O": True},
                                "atom_mapped_reaction_smiles": (
                                    "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                                ),
                            },
                        ],
                    }
                ],
            },
            target_smiles="CCO",
        )
        self.assertTrue(verifier["accepted"], verifier["reasons"])
        parent_proof = compile_stitched_parent_route_proof(
            target_smiles="CCO",
            parent_verifier=verifier,
        )
        solved = compile_route_objective_proof_bundle(
            blackboard=solved_blackboard,
            parent_route_proof=parent_proof,
        )

        self.assertFalse(plausible["accepted"])
        self.assertEqual(plausible["route_status"], "plausible_hypothesis_route")
        self.assertTrue(plausible["no_solved_claim"])
        self.assertTrue(solved["accepted"])
        self.assertTrue(solved["solved"])


if __name__ == "__main__":
    unittest.main()
