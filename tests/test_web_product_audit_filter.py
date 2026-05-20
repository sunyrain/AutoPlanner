import json
import tempfile
import unittest
from pathlib import Path

import cascade_planner.web.app as web_app
from cascade_planner.web.app import _apply_product_audit_post_filter


class WebProductAuditFilterTest(unittest.TestCase):
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
