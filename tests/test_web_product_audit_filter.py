import json
import tempfile
import unittest
from pathlib import Path

import cascade_planner.web.app as web_app
from cascade_planner.web.app import _apply_product_audit_post_filter, _apply_proposal_gate_post_filter


class WebProductAuditFilterTest(unittest.TestCase):
    def test_proposal_gate_stops_all_routes_at_frontier_before_product_audit(self):
        target = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
        output = {
            "target": target,
            "routes": [
                {
                    "score": 0.1,
                    "n_steps": 1,
                    "route_rank": 0,
                    "metrics": {"route_solved": False},
                    "steps": [
                        {
                            "index": 0,
                            "product": target,
                            "main_reactant": "CC(C)(C)OS(C)(=O)=O",
                            "aux_reactants": ["CC(C)(C)OS(=O)(=O)O"],
                            "reaction_smiles": f"CC(C)(C)OS(C)(=O)=O.CC(C)(C)OS(=O)(=O)O>>{target}",
                            "condition_predictions": [{"Reagent": "[Li]CCCC"}],
                        }
                    ],
                }
            ],
            "depth_attempts": [{}],
            "search_status": {},
            "route_set_metrics": {"diversity": {}},
            "ui_metadata": {},
        }

        _apply_proposal_gate_post_filter(output, {"proposal_gate_mode": "hard_reject"})

        self.assertEqual(output["proposal_gate"]["input_routes"], 1)
        self.assertEqual(output["proposal_gate"]["kept_routes"], 0)
        self.assertEqual(output["proposal_gate"]["dropped_routes"], 1)
        self.assertEqual(output["routes"], [])
        self.assertEqual(output["search_status"]["status"], "frontier")
        self.assertTrue(output["search_status"]["proposal_gate_removed_all"])
        self.assertEqual(output["frontiers"][0]["smiles"], target)
        self.assertIn("proposal_gate_filtered_all", output["failure_diagnosis"])

    def test_hides_reject_artifact_and_keeps_reviewable_route(self):
        target = "CC[C@@H](O)C[C@@H](O)CC=O"
        artifact = _native_route(
            target=target,
            reactants=["CC(=O)C(=O)O"],
            source="ChemEnzyRetroPlanner",
            score=0.95,
        )
        reviewable = _native_route(
            target=target,
            reactants=["CC[C@@H](O)C[C@@H](O)CC=O"],
            source="ChemEnzyRetroPlanner",
            score=0.1,
        )
        output = {"target": target, "routes": [artifact, reviewable], "depth_attempts": [{}], "search_status": {}}

        _apply_product_audit_post_filter(output, {"target_smiles": target, "product_audit_filter_mode": "hide_rejects"})

        self.assertEqual(output["post_filter"]["original_route_count"], 2)
        self.assertEqual(output["post_filter"]["kept_route_count"], 1)
        self.assertEqual(output["post_filter"]["removed_route_count"], 1)
        self.assertEqual(len(output["routes"]), 1)
        self.assertNotEqual(output["routes"][0]["product_audit"]["route_class"], "reject_artifact")
        self.assertIn("large_unexplained_carbon_gain", output["post_filter"]["issue_counts_removed"])

    def test_filter_removes_all_severe_artifacts_instead_of_showing_fake_routes(self):
        target = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        output = {
            "target": target,
            "routes": [
                _native_route(target=target, reactants=["C"], source="ChemEnzyRetroPlanner", score=0.9),
                _native_route(target=target, reactants=["CC"], source="ChemEnzyRetroPlanner", score=0.8),
            ],
            "depth_attempts": [{}],
            "search_status": {},
        }

        _apply_product_audit_post_filter(output, {"target_smiles": target, "product_audit_filter_mode": "hide_rejects"})

        self.assertEqual(output["post_filter"]["kept_route_count"], 0)
        self.assertEqual(output["post_filter"]["removed_route_count"], 2)
        self.assertEqual(output["post_filter"]["would_remove_route_count"], 2)
        self.assertIsNone(output["post_filter"]["fallback_reason"])
        self.assertEqual(len(output["routes"]), 0)
        self.assertEqual(output["search_status"]["status"], "filtered")
        self.assertTrue(output["search_status"]["native_returned_routes"])
        self.assertTrue(output["search_status"]["post_filter_removed_all"])
        self.assertIn("product-audit hid all", output["search_status"]["message"])
        self.assertIn("product_audit_filtered_all", output["failure_diagnosis"])
        self.assertIn("product_audit_filtered_all", output["failure_analysis"]["failure_categories"])
        self.assertTrue(output["failure_analysis"]["product_audit_filter"]["removed_all"])
        self.assertEqual(output["failure_analysis"]["product_audit_filter"]["original_route_count"], 2)
        self.assertEqual(output["failure_analysis"]["target_complexity"]["heavy_atoms"], 30)
        self.assertTrue(
            any("Dominant rejection issues" in row for row in output["failure_analysis"]["diagnosis"])
        )

    def test_rejected_sidecar_records_removed_routes_with_audit_reasons(self):
        target = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        output = {
            "target": target,
            "routes": [_native_route(target=target, reactants=["C"], source="ChemEnzyRetroPlanner", score=0.9)],
            "ui_metadata": {"backend": "CascadePlanner", "saved_at": "results/v2/filtered.json"},
            "depth_attempts": [{}],
            "search_status": {},
        }
        with tempfile.TemporaryDirectory(dir=web_app.ROOT) as td:
            rejected_path = Path(td) / "plan_rejected.json"

            _apply_product_audit_post_filter(
                output,
                {"target_smiles": target, "product_audit_filter_mode": "hide_rejects"},
                rejected_out_path=rejected_path,
            )

            saved = json.loads(rejected_path.read_text(encoding="utf-8"))
            self.assertEqual(output["post_filter"]["removed_route_count"], 1)
            self.assertEqual(output["ui_metadata"]["rejected_saved_at"], web_app._rel(rejected_path))
            self.assertEqual(saved["objective"], "chem_enzy_native_rejected_routes")
            self.assertEqual(len(saved["routes"]), 1)
            self.assertTrue(saved["routes"][0]["post_filter_removed"])
            self.assertEqual(saved["routes"][0]["product_audit"]["route_class"], "reject_artifact")
            self.assertIn("large_unexplained_carbon_gain", saved["routes"][0]["post_filter_remove_reason"])

    def test_risk_guarded_mode_only_reranks(self):
        target = "CC[C@@H](O)C[C@@H](O)CC=O"
        artifact = _native_route(
            target=target,
            reactants=["CC(=O)C(=O)O"],
            source="ChemEnzyRetroPlanner",
            score=0.95,
        )
        reviewable = _native_route(
            target=target,
            reactants=["CC[C@@H](O)C[C@@H](O)CC=O"],
            source="ChemEnzyRetroPlanner",
            score=0.1,
        )
        output = {"target": target, "routes": [artifact, reviewable], "depth_attempts": [{}], "search_status": {}}

        _apply_product_audit_post_filter(output, {"target_smiles": target, "product_audit_filter_mode": "risk_guarded"})

        self.assertEqual(output["post_filter"]["kept_route_count"], 2)
        self.assertEqual(output["post_filter"]["removed_route_count"], 0)
        self.assertEqual(len(output["routes"]), 2)
        self.assertNotEqual(output["routes"][0]["product_audit"]["route_class"], "reject_artifact")
        self.assertEqual(output["routes"][1]["product_audit"]["route_class"], "reject_artifact")

    def test_default_mode_shows_all_routes_with_risk_labels(self):
        target = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        output = {
            "target": target,
            "routes": [
                _native_route(target=target, reactants=["C"], source="ChemEnzyRetroPlanner", score=0.9),
                _native_route(target=target, reactants=["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"], source="ChemEnzyRetroPlanner", score=0.1),
            ],
            "depth_attempts": [{}],
            "search_status": {},
        }

        _apply_product_audit_post_filter(output, {"target_smiles": target})

        self.assertEqual(output["post_filter"]["mode"], "risk_guarded")
        self.assertEqual(output["post_filter"]["kept_route_count"], 2)
        self.assertEqual(output["post_filter"]["removed_route_count"], 0)
        self.assertEqual(len(output["routes"]), 2)
        self.assertTrue(all("product_audit" in route for route in output["routes"]))


def _native_route(*, target: str, reactants: list[str], source: str, score: float) -> dict:
    stock = {smi: True for smi in reactants}
    return {
        "score": score,
        "n_steps": 1,
        "steps": [
            {
                "index": 0,
                "product": target,
                "main_reactant": reactants[0] if reactants else "",
                "aux_reactants": reactants[1:],
                "reaction_smiles": f"{'.'.join(reactants)}>>{target}",
                "reaction_type": "unknown",
                "source": source,
                "scores": {"confidence": score},
                "stock_status": stock,
                "reaction_interpretation": {
                    "reaction_class": "unknown",
                    "atom_change": {"heavy_atom_delta": 0},
                },
            }
        ],
        "metrics": {
            "strict_stock_solve": True,
            "route_solved": True,
            "filled_route": True,
            "terminal_reactants": reactants,
            "retrosynthesis_progress": {},
            "route_naturalness": {},
            "cascade_compatibility": {"issues": []},
        },
    }


if __name__ == "__main__":
    unittest.main()
