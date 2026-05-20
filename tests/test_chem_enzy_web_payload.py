import unittest
import tempfile
from pathlib import Path

from cascade_planner.baselines.route_contract import (
    BackendFailure,
    BaselineRunResult,
    RouteCandidate,
    RouteSearchConfig,
    RouteStepCandidate,
)
from scripts.run_chem_enzy_plan_for_web import _stock_names_from_payload, _web_payload_from_result


class ChemEnzyWebPayloadTest(unittest.TestCase):
    def test_stock_mode_maps_to_smaller_building_block_stock(self):
        self.assertEqual(_stock_names_from_payload({"stock_mode": "commercial"}), ["Zinc_Fix-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_mode": "benchmark-n5"}), ["PaRotes_n5-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_mode": "building-block"}), ["PaRotes_n1-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_names": ["RetroStar-stock"]}), ["RetroStar-stock"])

    def test_exports_condition_and_enzyme_annotations(self):
        target = "CCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CC", "O"],
            rxn_smiles="CC.O>>CCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.82,
            stock_status={"CC": True, "O": True},
            condition_predictions=[
                {
                    "Temperature": 25.0,
                    "pH": 7.4,
                    "Solvent": "water",
                    "Reagent": "buffer",
                    "Catalyst": "NADH",
                    "Score": "0.9123",
                }
            ],
            enzyme_ec_annotations=[{"ec_number": "1.1.1.1", "confidence": 0.91}],
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[
                RouteCandidate(
                    target_smiles=target,
                    steps=[step],
                    solved=True,
                    score=0.82,
                )
            ],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "enable_condition_prediction": True, "enable_enzyme_assignment": True},
            RouteSearchConfig(target_smiles=target, max_iterations=10, max_depth=6, expansion_topk=50),
            1.2,
        )

        web_step = payload["routes"][0]["steps"][0]
        self.assertEqual(web_step["T"], 25.0)
        self.assertEqual(web_step["pH"], 7.4)
        self.assertEqual(web_step["solvent"], "water")
        self.assertEqual(web_step["catalyst"], "NADH")
        self.assertEqual(web_step["ec"], "1.1.1.1")
        self.assertEqual(web_step["scores"]["condition"], 0.9123)
        self.assertEqual(web_step["scores"]["enzyme"], 0.91)
        self.assertTrue(web_step["condition_predictions"])
        self.assertTrue(web_step["enzyme_ec_annotations"])
        self.assertIn("condition_score=0.9123", web_step["reaction_interpretation"]["catalysis_and_conditions"])

    def test_no_route_failure_has_retry_diagnosis(self):
        target = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            failures=[
                BackendFailure(
                    category="no_route_found",
                    message="ChemEnzyRetroPlanner returned no successful routes",
                    target_smiles=target,
                    retryable=True,
                )
            ],
            raw_backend_metadata={"elapsed_s": 7.9},
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "thorough", "enable_condition_prediction": False, "enable_enzyme_assignment": False},
            RouteSearchConfig(target_smiles=target, max_iterations=50, max_depth=12, expansion_topk=100),
            23.2,
        )

        analysis = payload["failure_analysis"]
        self.assertTrue(analysis["available"])
        self.assertIn("no_route_found", analysis["failure_categories"])
        self.assertGreaterEqual(analysis["target_heavy_atoms"], 38)
        self.assertIn("increase chem_enzy_iterations to 100-200", analysis["retry_suggestions"])
        self.assertIn("increase chem_enzy_expansion_topk to 150-200", analysis["retry_suggestions"])
        self.assertIn("increase max_steps to 16-20", analysis["retry_suggestions"])

    def test_open_stock_native_route_is_not_reported_as_solved(self):
        target = "CCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CC"],
            rxn_smiles="CC>>CCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"CC": False},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[step], solved=True, score=0.5)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block"},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        metrics = payload["routes"][0]["metrics"]
        self.assertFalse(metrics["strict_stock_solve"])
        self.assertFalse(metrics["route_solved"])
        self.assertTrue(metrics["native_returned_route"])
        self.assertEqual(payload["search_status"]["status"], "partial")
        self.assertFalse(payload["search_status"]["solved"])

    def test_internal_intermediate_does_not_make_multistep_route_open_stock(self):
        target = "CCO"
        first = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CCOC"],
            rxn_smiles="CCOC>>CCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"CCOC": False},
        )
        second = RouteStepCandidate(
            product_smiles="CCOC",
            reactant_smiles=["CC", "CO"],
            rxn_smiles="CC.CO>>CCOC",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.4,
            stock_status={"CC": True, "CO": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[first, second], solved=True, score=0.2)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block"},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        metrics = payload["routes"][0]["metrics"]
        self.assertTrue(metrics["strict_stock_solve"])
        self.assertTrue(metrics["route_solved"])
        self.assertEqual(set(metrics["terminal_reactants"]), {"CC", "CO"})
        self.assertNotIn("CCOC", metrics["terminal_reactants"])
        self.assertEqual(payload["search_status"]["status"], "solved")

    def test_no_route_failure_reports_target_stock_hit(self):
        target = "CCO"
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            failures=[
                BackendFailure(
                    category="no_route_found",
                    message="ChemEnzyRetroPlanner returned no successful routes",
                    target_smiles=target,
                    retryable=True,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            vendor = Path(td)
            cfg = vendor / "retro_planner" / "config" / "config.yaml"
            stock = vendor / "retro_planner" / "building_block_dataset" / "stock.csv"
            cfg.parent.mkdir(parents=True)
            stock.parent.mkdir(parents=True)
            cfg.write_text('stocks:\n  Test-stock: "building_block_dataset/stock.csv"\n', encoding="utf-8")
            stock.write_text("CCO\nCCN\n", encoding="utf-8")

            payload = _web_payload_from_result(
                result,
                {"search_preset": "quick"},
                RouteSearchConfig(
                    target_smiles=target,
                    stock_names=["Test-stock"],
                    max_iterations=10,
                    max_depth=6,
                    expansion_topk=50,
                ),
                1.0,
                vendor_root=vendor,
            )

        membership = payload["failure_analysis"]["search_config"]["target_stock_membership"]
        self.assertTrue(membership["target_in_selected_stock"])
        self.assertEqual(membership["hit_stocks"], ["Test-stock"])
        self.assertTrue(
            any("Target itself is present in the selected stock" in row for row in payload["failure_analysis"]["diagnosis"])
        )


if __name__ == "__main__":
    unittest.main()
