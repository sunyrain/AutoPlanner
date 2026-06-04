import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.agent.target_profile import build_frontier_report, build_target_profile
from cascade_planner.baselines.chem_enzy_adapter import DEFAULT_ONE_STEP_MODELS
from cascade_planner.baselines.route_contract import RouteSearchConfig


class SmilesFirstWorkflowTest(unittest.TestCase):
    def test_target_profile_and_frontier_report_for_advanced_frontier(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        profile = build_target_profile(target, target_name="bufadienolide_like", family_hint="steroid")
        report = build_frontier_report(profile, frontier_smiles=target)

        self.assertTrue(profile.valid)
        self.assertGreaterEqual(profile.rings, 4)
        self.assertIn("polycyclic_or_steroid_like", profile.family_hints)
        self.assertTrue(report["advanced_frontier_found"])
        self.assertIn("advanced_same_scaffold", report["frontiers"][0]["flags"])
        self.assertIn("unresolved_core", report["frontiers"][0]["flags"])

    def test_full_workflow_writes_required_artifacts_and_guarded_status(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="bufadienolide_like",
                    family_hint="bufadienolide, steroid, pyrone",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=5,
                )
            )
            out = Path(tmp)
            required = [
                "target_profile.json",
                "baseline_routes.json",
                "frontier_report.json",
                "literature_search_report.md",
                "evidence_cards.jsonl",
                "validation.json",
                "summary.md",
            ]

            for name in required:
                self.assertTrue((out / name).exists(), name)
            candidate_path = Path(result["artifacts"]["literature_candidates"])
            package_path = Path(result["artifacts"]["hybrid_route_package"])
            route_map_path = Path(result["artifacts"]["route_map"])
            self.assertTrue(candidate_path.exists())
            self.assertTrue(package_path.exists())
            self.assertTrue(route_map_path.exists())

            validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
            package = json.loads(package_path.read_text(encoding="utf-8"))
            kinds = {row["candidate_kind"] for row in package["literature_candidates"]}

        self.assertTrue(validation["accepted"], validation)
        self.assertEqual(validation["route_status"], "partial_anchor")
        self.assertNotEqual(package["route_status"], "solved")
        self.assertIn("exact_fragment_retro", kinds)
        self.assertIn("forward_surrogate", kinds)
        self.assertIn("route_anchor", kinds)
        self.assertTrue(validation["guards"]["route_anchor_not_stock"])
        self.assertTrue(validation["guards"]["forward_surrogate_not_lab_procedure"])

    def test_invalid_target_stops_before_literature_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles="not_a_smiles",
                    target_name="bad",
                    output_dir=tmp,
                )
            )
            validation = json.loads((Path(tmp) / "validation.json").read_text(encoding="utf-8"))

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["route_status"], "invalid_package")
        self.assertIn("invalid_target_smiles", validation["reasons"])
        self.assertIn("validation", result)

    def test_baseline_json_is_preserved_and_frontier_is_extracted(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        baseline = {
            "schema_version": "baseline_routes.v1",
            "status": "provided",
            "solved": False,
            "routes": [
                {
                    "route_id": "late_decoration_only",
                    "ordinary_steps": ["O-acetylation"],
                    "unresolved_frontiers": [target],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="bufadienolide_like",
                    family_hint="bufadienolide, steroid, pyrone",
                    baseline_json=baseline_path,
                    output_dir=Path(tmp) / "out",
                    query_budget=3,
                )
            )
            saved_baseline = json.loads((Path(tmp) / "out" / "baseline_routes.json").read_text(encoding="utf-8"))
            frontier_report = json.loads((Path(tmp) / "out" / "frontier_report.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_baseline["routes"][0]["route_id"], "late_decoration_only")
        self.assertTrue(frontier_report["advanced_frontier_found"])
        self.assertIn("ordinary_decoration_only", frontier_report["frontiers"][0]["flags"])
        self.assertIn("unresolved_core", frontier_report["frontiers"][0]["flags"])

    def test_cli_direct_execution_writes_artifacts(self):
        target = "O=C1CCCCCCCCCCCCO1"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_smiles_first_literature_workflow.py",
                    "--target-smiles",
                    target,
                    "--target-name",
                    "macro_lactone_cli",
                    "--family-hint",
                    "macrocycle, macrolactonization, polyketide",
                    "--frontier-smiles",
                    target,
                    "--output-dir",
                    tmp,
                    "--query-budget",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            validation = json.loads((Path(tmp) / "validation.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["case_id"], "macro_lactone_cli")
        self.assertTrue(validation["accepted"], validation)
        self.assertEqual(validation["route_status"], "partial_anchor")

    def test_workflow_does_not_mutate_chem_enzy_default_search_config(self):
        before_models = list(DEFAULT_ONE_STEP_MODELS)
        before_config = RouteSearchConfig(target_smiles="CCO").to_dict()
        target = "Oc1ccccc1OC1COC(O)C(O)C1O"

        with tempfile.TemporaryDirectory() as tmp:
            run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="glycoside_default_guard",
                    family_hint="glycoside, sugar, glycosylation",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=3,
                )
            )
        after_config = RouteSearchConfig(target_smiles="CCO").to_dict()

        self.assertEqual(DEFAULT_ONE_STEP_MODELS, before_models)
        self.assertEqual(after_config, before_config)
        self.assertEqual(after_config["search_flags"], {})

    def test_macrocycle_case_finds_macrolactonization_template(self):
        target = "O=C1CCCCCCCCCCCCO1"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="macro_lactone_like",
                    family_hint="macrocycle, macrolactonization, polyketide",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=4,
                )
            )
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))
            templates = package["strategy_templates"]

        self.assertTrue(any(t.get("reaction_class") == "macrolactonization" for t in templates))
        self.assertTrue(all(t.get("not_raw_reaction_injection") for t in templates))

    def test_glycoside_case_finds_glycosylation_template(self):
        target = "Oc1ccccc1OC1COC(O)C(O)C1O"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="phenolic_glycoside_like",
                    family_hint="glycoside, sugar, glycosylation",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=4,
                )
            )
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))
            templates = package["strategy_templates"]

        self.assertTrue(any(t.get("reaction_class") == "glycosylation" for t in templates))
        self.assertTrue(any(t.get("candidate_kind") == "forward_surrogate" for t in templates))


if __name__ == "__main__":
    unittest.main()
