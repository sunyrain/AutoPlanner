import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer

from cascade_planner.baselines.route_contract import (
    BackendFailure,
    BaselineRunResult,
    RouteCandidate,
    RouteSearchConfig,
    RouteStepCandidate,
)
from cascade_planner.cascade_verifier import cascade_verifier_features
from scripts.run_chem_enzy_plan_for_web import _route_config_from_payload, _stock_names_from_payload, _web_payload_from_result


class ChemEnzyWebPayloadTest(unittest.TestCase):
    def test_stock_mode_maps_to_smaller_building_block_stock(self):
        self.assertEqual(_stock_names_from_payload({"stock_mode": "commercial"}), ["Zinc_Fix-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_mode": "benchmark-n5"}), ["PaRotes_n5-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_mode": "building-block"}), ["PaRotes_n1-stock"])
        self.assertEqual(_stock_names_from_payload({"stock_names": ["RetroStar-stock"]}), ["RetroStar-stock"])

    def test_web_route_config_keeps_legacy_cascade_hooks_off_by_default(self):
        config = _route_config_from_payload({"target_smiles": "CCO", "search_preset": "quick"}, gpu=-1)

        self.assertFalse(config.search_flags["use_cascade_cost_model"])
        self.assertFalse(config.search_flags["use_cascade_source_policy"])
        self.assertFalse(config.search_flags["legacy_cascade_hooks_enabled"])
        self.assertEqual(config.search_flags["chem_enzy_onmt_tokenizer"], "char")
        self.assertIsNone((config.search_flags["cascade_cost_model"] or {}).get("action_value_model_path"))
        self.assertIsNone((config.search_flags["cascade_source_policy"] or {}).get("source_value_model_path"))

    def test_web_route_config_accepts_onmt_tokenizer_opt_in(self):
        config = _route_config_from_payload(
            {"target_smiles": "CCO", "search_preset": "quick", "chem_enzy_onmt_tokenizer": "token"},
            gpu=-1,
        )

        self.assertEqual(config.search_flags["chem_enzy_onmt_tokenizer"], "token")

    def test_web_route_config_accepts_native_enzyme_plugin_opt_in(self):
        config = _route_config_from_payload(
            {
                "target_smiles": "CCO",
                "search_preset": "quick",
                "enable_native_enzyme_plugin": True,
                "native_enzyme_topk": 3,
                "native_enzyme_max_added": 2,
            },
            gpu=-1,
        )

        plugin = config.search_flags["native_enzyme_plugin"]
        self.assertTrue(plugin["enabled"])
        self.assertEqual(plugin["top_k"], 3)
        self.assertEqual(plugin["max_added"], 2)

    def test_web_route_config_accepts_step_strengthening_opt_in(self):
        config = _route_config_from_payload(
            {
                "target_smiles": "CCO",
                "search_preset": "quick",
                "enable_chem_enzy_step_strengthening": True,
            },
            gpu=-1,
        )

        self.assertTrue(config.search_flags["chem_enzy_step_strengthening"]["enabled"])

    def test_web_route_config_consumes_guided_chem_enzy_policy(self):
        policy = {
            "schema_version": "chem_enzy_search_policy.v1",
            "policy_id": "policy_api",
            "operator_id": "operator_api",
            "case_id": "case_api",
            "mode": "guided",
            "evidence_refs": ["ev1"],
            "terminal_blacklist": ["CCO"],
            "anchor_whitelist": ["CC"],
            "preferred_subgoal": {"kind": "anchor"},
            "source_budget": {"enable_native_chemical_plugin": True, "native_chemical_max_added": 2},
            "rerun_reason": "api guided policy smoke",
            "budget": {"max_reruns": 1, "max_iterations": 7, "max_depth": 3, "expansion_topk": 9},
        }

        config = _route_config_from_payload(
            {
                "target_smiles": "CCO",
                "search_preset": "quick",
                "chem_enzy_search_policy": policy,
            },
            gpu=-1,
        )

        self.assertEqual(config.max_iterations, 7)
        self.assertEqual(config.max_depth, 3)
        self.assertEqual(config.expansion_topk, 9)
        self.assertEqual(config.search_flags["chem_enzy_search_policy"]["policy_id"], "policy_api")
        self.assertEqual(config.search_flags["cascade_search_context"]["chem_enzy_policy_id"], "policy_api")
        self.assertIn("CCO", config.search_flags["cascade_search_context"]["terminal_blacklist"])
        self.assertIn("CC", config.search_flags["cascade_search_context"]["anchor_whitelist"])

    def test_web_route_config_rejects_invalid_onmt_tokenizer(self):
        with self.assertRaises(ValueError):
            _route_config_from_payload(
                {"target_smiles": "CCO", "search_preset": "quick", "chem_enzy_onmt_tokenizer": "sentencepiece"},
                gpu=-1,
            )

    def test_web_route_config_rejects_legacy_hooks_without_opt_in_env(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            _route_config_from_payload(
                {
                    "target_smiles": "CCO",
                    "enable_legacy_cascade_hooks": True,
                },
                gpu=-1,
            )

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
            raw_backend_metadata={
                "template": {
                    "autoplanner_enzyme_quality_v1": {
                        "decision": "pass",
                        "quality_score": 0.94,
                        "flags": [],
                        "material_sanity": {"passed": True, "reasons": []},
                    }
                }
            },
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
        self.assertEqual(web_step["enzyme_quality"]["decision"], "pass")
        self.assertEqual(web_step["evidence"]["enzyme_quality_score"], 0.94)
        self.assertIn("condition_score=0.9123", web_step["reaction_interpretation"]["catalysis_and_conditions"])

    def test_exports_derived_quality_for_native_enzyme_like_step(self):
        target = "CCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CC=O"],
            rxn_smiles="CC=O>>CCO",
            source_model="onmt_models.bionav_one_step",
            enzyme_ec_annotations=[{"ec_number": "1.1.1.1", "confidence": 0.73}],
            raw_backend_metadata={
                "cascade_cost": {
                    "cascade_adjustment": 0.7,
                    "components": {"enzyme_evidence": 0.7},
                    "material_sanity": {"passed": True, "reasons": []},
                }
            },
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[step], solved=True, score=0.7)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "enable_enzyme_assignment": True},
            RouteSearchConfig(target_smiles=target, max_iterations=10, max_depth=6, expansion_topk=50),
            1.0,
        )

        quality = payload["routes"][0]["steps"][0]["enzyme_quality"]
        self.assertEqual(quality["origin"], "derived_from_selected_step")
        self.assertEqual(quality["decision"], "warn")
        self.assertIn("missing_sp_v1", quality["flags"])

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

    def test_no_route_can_display_late_stage_semisynthesis_anchor(self):
        target = "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
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

        payload = _web_payload_from_result(
            result,
            {"target_smiles": target, "search_preset": "thorough"},
            RouteSearchConfig(target_smiles=target, max_iterations=200, max_depth=20, expansion_topk=100),
            1.0,
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["n_results"], 1)
        self.assertEqual(payload["search_status"]["status"], "solved")
        self.assertTrue(payload["search_status"]["solved"])
        self.assertFalse(payload["search_status"]["native_raw_returned_routes"])
        self.assertTrue(payload["search_status"]["semisynthesis_rescue_returned_routes"])
        self.assertEqual(payload["route_set_metrics"]["semisynthesis_rescue"]["route_count"], 1)
        route = payload["routes"][0]
        self.assertTrue(route["metrics"]["semisynthesis_anchor"])
        self.assertTrue(route["metrics"]["source_supported_semisynthesis"])
        self.assertFalse(route["metrics"]["native_returned_route"])
        self.assertTrue(route["metrics"]["route_solved"])
        self.assertEqual(route["metrics"]["cascade_verifier"]["reason_counts"], {})
        step = route["steps"][0]
        self.assertEqual(step["aux_reactants"], ["CC(=O)OC(C)=O"])
        self.assertIn("semisynthesis_rescue", step["source"])

    def test_no_route_can_display_tbs_protected_known_precursor_anchor(self):
        target = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
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

        payload = _web_payload_from_result(
            result,
            {"target_smiles": target, "search_preset": "thorough"},
            RouteSearchConfig(target_smiles=target, max_iterations=200, max_depth=20, expansion_topk=100),
            1.0,
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["n_results"], 1)
        self.assertTrue(payload["search_status"]["semisynthesis_rescue_returned_routes"])
        route = payload["routes"][0]
        self.assertTrue(route["metrics"]["source_supported_semisynthesis"])
        step = route["steps"][0]
        self.assertIn("CC(C)(C)[Si](C)(C)Cl", step["aux_reactants"])
        self.assertIn("tbs_silylation", step["source"])

    def test_stitched_semisynthesis_route_can_be_stock_closed(self):
        target = "CCOC(C)=O"
        anchor = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CCO", "CC(=O)OC(C)=O"],
            rxn_smiles="CCO.CC(=O)OC(C)=O>>CCOC(C)=O",
            source_model="semisynthesis_rescue.o_acetylation",
            stock_status={"CCO": False, "CC(=O)OC(C)=O": True},
        )
        upstream = RouteStepCandidate(
            product_smiles="CCO",
            reactant_smiles=["CC", "O"],
            rxn_smiles="CC.O>>CCO",
            source_model="graphfp",
            stock_status={"CC": True, "O": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="AutoPlanner semisynthesis rescue + upstream ChemEnzy",
            routes=[
                RouteCandidate(
                    target_smiles=target,
                    steps=[anchor, upstream],
                    solved=True,
                    score=0.8,
                    raw_backend_metadata={
                        "rescue_type": "late_stage_o_acetylation",
                        "route_class_hint": "stitched_semisynthesis_upstream",
                    },
                )
            ],
        )

        payload = _web_payload_from_result(
            result,
            {"target_smiles": target, "enable_semisynthesis_rescue": False},
            RouteSearchConfig(target_smiles=target, max_iterations=10, max_depth=6),
            1.0,
        )

        metrics = payload["routes"][0]["metrics"]
        self.assertTrue(metrics["stitched_semisynthesis"])
        self.assertTrue(metrics["strict_stock_solve"])
        self.assertTrue(metrics["route_solved"])
        self.assertEqual(set(metrics["terminal_reactants"]), {"CC", "O", "CC(=O)OC(C)=O"})
        self.assertEqual(payload["search_status"]["status"], "solved")

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

    def test_web_route_metrics_include_rule_verifier_report(self):
        target = "CCCCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["C"],
            rxn_smiles="C>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"C": True},
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
        self.assertFalse(metrics["cascade_compatibility"]["cascade_compatibility_success"])
        self.assertIn("atom_balance_violation", metrics["cascade_compatibility"]["issues"])
        self.assertIn("cascade_verifier", metrics)
        self.assertEqual(metrics["cascade_verifier"]["reason_counts"]["atom_balance_violation"], 1)

    def test_web_payload_orders_source_supported_and_verifier_feasible_routes_first(self):
        target = "CCCCO"
        bad_native = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["C"],
            rxn_smiles="C>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=10.0,
            stock_status={"C": True},
        )
        good_native = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CCCC", "O"],
            rxn_smiles="CCCC.O>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=1.0,
            stock_status={"CCCC": True, "O": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[
                RouteCandidate(target_smiles=target, steps=[bad_native], solved=True, score=10.0),
                RouteCandidate(target_smiles=target, steps=[good_native], solved=True, score=1.0),
            ],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block"},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        self.assertEqual(payload["routes"][0]["steps"][0]["reaction_smiles"], "CCCC.O>>CCCCO")
        self.assertTrue(payload["routes"][0]["metrics"]["cascade_verifier"]["feasible"])
        self.assertEqual(payload["routes"][1]["steps"][0]["reaction_smiles"], "C>>CCCCO")
        self.assertFalse(payload["routes"][1]["metrics"]["cascade_verifier"]["feasible"])

    def test_web_route_metrics_include_learned_verifier_annotation_when_enabled(self):
        target = "CCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CC", "O"],
            rxn_smiles="CC.O>>CCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"CC": True, "O": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[step], solved=True, score=0.5)],
        )
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "learned.joblib"
            _constant_learned_verifier(model, target_smiles=target, probability=0.77, threshold=0.98)

            payload = _web_payload_from_result(
                result,
                {
                    "search_preset": "quick",
                    "stock_mode": "building-block",
                    "enable_learned_verifier_annotation": True,
                    "learned_verifier_model": str(model),
                },
                RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
                1.0,
            )

        report = payload["route_set_metrics"]["learned_verifier_annotation"]
        learned = payload["routes"][0]["metrics"]["learned_cascade_verifier"]
        self.assertTrue(report["enabled"])
        self.assertTrue(report["model_loaded"])
        self.assertEqual(report["input_routes"], 1)
        self.assertEqual(report["annotated_routes"], 1)
        self.assertEqual(report["policy"], "annotation_only")
        self.assertTrue(learned["available"])
        self.assertEqual(learned["policy"], "annotation_only")
        self.assertEqual(learned["feasible_probability"], 0.77)
        self.assertFalse(learned["conservative_feasible"])
        self.assertEqual(payload["n_results"], 1)
        self.assertFalse(payload["route_set_metrics"]["cascade_verifier_gate"]["enabled"])

    def test_enzyme_coverage_sidecar_can_be_attached_to_real_web_payload(self):
        target = "CC=O"
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

        with patch("scripts.run_chem_enzy_plan_for_web.build_enzyme_coverage_sidecar") as sidecar:
            sidecar.return_value = {
                "schema_version": "enzyme_coverage_sidecar.v1",
                "enabled": True,
                "source": "enzyme_precedent",
                "bridge_hit_count": 1,
                "candidate_count": 2,
                "sp_v1_accepted_count": 1,
                "error": "",
            }
            payload = _web_payload_from_result(
                result,
                {"target_smiles": target, "enable_enzyme_coverage_sidecar": True},
                RouteSearchConfig(target_smiles=target, max_iterations=10, max_depth=6, expansion_topk=50),
                1.0,
            )

        report = payload["route_set_metrics"]["enzyme_coverage_sidecar"]
        self.assertTrue(payload["ui_metadata"]["enzyme_coverage_sidecar_enabled"])
        self.assertEqual(report["source"], "enzyme_precedent")
        self.assertEqual(payload["ui_metadata"]["enzyme_coverage_sidecar"]["candidate_count"], 2)

    def test_missing_learned_verifier_annotation_model_does_not_fail_web_payload(self):
        target = "CCO"
        step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CC", "O"],
            rxn_smiles="CC.O>>CCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"CC": True, "O": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[step], solved=True, score=0.5)],
        )

        payload = _web_payload_from_result(
            result,
            {
                "search_preset": "quick",
                "stock_mode": "building-block",
                "enable_learned_verifier_annotation": True,
                "learned_verifier_model": "/tmp/autoplanner_missing_learned_verifier.joblib",
            },
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        report = payload["route_set_metrics"]["learned_verifier_annotation"]
        learned = payload["routes"][0]["metrics"]["learned_cascade_verifier"]
        self.assertTrue(report["enabled"])
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["error"], "model_not_found")
        self.assertEqual(report["annotated_routes"], 0)
        self.assertFalse(learned["available"])
        self.assertEqual(payload["n_results"], 1)

    def test_rule_verifier_gate_is_off_by_default(self):
        target = "CCCCO"
        bad_step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["C"],
            rxn_smiles="C>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"C": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[bad_step], solved=True, score=0.5)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block"},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        self.assertEqual(payload["n_results"], 1)
        self.assertFalse(payload["route_set_metrics"]["cascade_verifier_gate"]["enabled"])
        self.assertEqual(payload["route_set_metrics"]["cascade_verifier_gate"]["dropped_routes"], 0)
        self.assertFalse(payload["routes"][0]["metrics"]["cascade_verifier"]["feasible"])

    def test_rule_verifier_gate_filters_infeasible_routes_when_enabled(self):
        target = "CCCCO"
        bad_step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["C"],
            rxn_smiles="C>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.4,
            stock_status={"C": True},
        )
        good_step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CCCC", "O"],
            rxn_smiles="CCCC.O>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.8,
            stock_status={"CCCC": True, "O": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[
                RouteCandidate(target_smiles=target, steps=[bad_step], solved=True, score=0.4),
                RouteCandidate(target_smiles=target, steps=[good_step], solved=True, score=0.8),
            ],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block", "enable_rule_verifier_gate": True},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        gate = payload["route_set_metrics"]["cascade_verifier_gate"]
        self.assertTrue(gate["enabled"])
        self.assertEqual(gate["input_routes"], 2)
        self.assertEqual(gate["kept_routes"], 1)
        self.assertEqual(gate["dropped_routes"], 1)
        self.assertEqual(payload["n_results"], 1)
        self.assertEqual(payload["routes"][0]["route_rank"], 0)
        self.assertEqual(payload["routes"][0]["steps"][0]["reaction_smiles"], "CCCC.O>>CCCCO")
        self.assertTrue(payload["routes"][0]["metrics"]["cascade_verifier"]["feasible"])
        self.assertEqual(gate["dropped"][0]["reason_counts"]["atom_balance_violation"], 1)

    def test_rule_verifier_gate_reports_when_all_routes_are_filtered(self):
        target = "CCCCO"
        bad_step = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["C"],
            rxn_smiles="C>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            score=0.5,
            stock_status={"C": True},
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[bad_step], solved=True, score=0.5)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block", "cascade_verifier_gate": True},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["n_results"], 0)
        self.assertEqual(payload["search_status"]["status"], "filtered")
        self.assertFalse(payload["search_status"]["native_returned_routes"])
        self.assertTrue(payload["search_status"]["native_raw_returned_routes"])
        self.assertEqual(payload["search_status"]["native_raw_n_routes"], 1)
        self.assertIn("rule verifier gate removed all displayed candidates", payload["search_status"]["message"])
        self.assertEqual(payload["ui_metadata"]["cascade_verifier_gate"]["dropped_routes"], 1)

    def test_web_verifier_treats_unpartitioned_routes_as_stepwise_by_default(self):
        target = "CCCCO"
        first = RouteStepCandidate(
            product_smiles=target,
            reactant_smiles=["CCCC"],
            rxn_smiles="CCCC>>CCCCO",
            source_model="graphfp_models.USPTO-full_remapped",
            condition_predictions=[{"Temperature": 0, "pH": 2, "Solvent": "water"}],
        )
        second = RouteStepCandidate(
            product_smiles="CCCC",
            reactant_smiles=["CC", "CC"],
            rxn_smiles="CC.CC>>CCCC",
            source_model="graphfp_models.USPTO-full_remapped",
            condition_predictions=[{"Temperature": 100, "pH": 12, "Solvent": "toluene"}],
        )
        result = BaselineRunResult(
            target_smiles=target,
            backend="ChemEnzyRetroPlanner",
            routes=[RouteCandidate(target_smiles=target, steps=[first, second], solved=True, score=0.5)],
        )

        payload = _web_payload_from_result(
            result,
            {"search_preset": "quick", "stock_mode": "building-block"},
            RouteSearchConfig(target_smiles=target, stock_names=["PaRotes_n1-stock"], max_iterations=10, max_depth=6),
            1.0,
        )

        reasons = payload["routes"][0]["metrics"]["cascade_verifier"]["reason_counts"]
        self.assertNotIn("temperature_conflict", reasons)
        self.assertNotIn("ph_conflict", reasons)
        self.assertNotIn("solvent_conflict", reasons)

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


def _constant_learned_verifier(path: Path, *, target_smiles: str, probability: float, threshold: float) -> None:
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([cascade_verifier_features({"target_smiles": target_smiles, "cascade": {"steps": []}})])
    model = _FixedProbabilityModel(probability)
    model.fit(x, [1])
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "feasible_model": model,
            "reason_models": {},
            "reason_labels": [],
            "recommended_feasible_threshold": threshold,
        },
        path,
    )


class _FixedProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probability: float):
        self.probability = float(probability)

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return [[1.0 - self.probability, self.probability] for _ in range(x.shape[0])]


if __name__ == "__main__":
    unittest.main()
