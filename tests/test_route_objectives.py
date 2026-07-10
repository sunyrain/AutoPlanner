import json
import unittest

from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.route_objectives import (
    build_broad_transform_templates_from_blackboard,
    classify_route_objectives,
    compile_route_objective_proof_bundle,
)


C22_STEROID_LIKE = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
ATORVASTATIN_FREE_ACID = (
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
    "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
)


class RouteObjectiveTest(unittest.TestCase):
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
        intermediate = "C" * 12
        verifier = verify_chemenzy_raw_routes(
            {
                "target": C22_STEROID_LIKE,
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {
                            "terminal_reactants": ["CC", "O"],
                            "terminal_stock_status": {"CC": True, "O": True},
                        },
                        "steps": [
                            {
                                "product": intermediate,
                                "reactant_smiles": ["CC"] * 6,
                                "stock_status": {"CC": True},
                            },
                            {
                                "product": C22_STEROID_LIKE,
                                "reactant_smiles": [intermediate] + ["CC"] * 5 + ["O"] * 3,
                                "stock_status": {"CC": True, "O": True},
                            },
                        ],
                    }
                ],
            },
            target_smiles=C22_STEROID_LIKE,
        )
        self.assertTrue(verifier["accepted"], verifier["reasons"])
        parent_proof = compile_stitched_parent_route_proof(
            target_smiles=C22_STEROID_LIKE,
            parent_verifier=verifier,
        )
        solved = compile_route_objective_proof_bundle(
            blackboard=blackboard,
            parent_route_proof=parent_proof,
        )

        self.assertFalse(plausible["accepted"])
        self.assertEqual(plausible["route_status"], "plausible_hypothesis_route")
        self.assertTrue(plausible["no_solved_claim"])
        self.assertTrue(solved["accepted"])
        self.assertTrue(solved["solved"])


if __name__ == "__main__":
    unittest.main()
