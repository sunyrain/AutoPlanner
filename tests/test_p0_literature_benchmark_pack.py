import tempfile
import unittest

from cascade_planner.agent.p0_benchmarks import (
    P0BenchmarkCase,
    frozen_p0_benchmark_cases,
    run_p0_benchmark_pack,
)


class P0LiteratureBenchmarkPackTest(unittest.TestCase):
    def test_frozen_cases_include_known_natural_product_strategies(self):
        cases = frozen_p0_benchmark_cases()
        ids = {case.case_id for case in cases}

        self.assertIn("bufotalin_like_c17_pyrone", ids)
        self.assertIn("paclitaxel_taxane_semisynthesis", ids)
        self.assertIn("artemisinin_peroxide_anchor", ids)
        self.assertIn("lovastatin_semisynthesis_core", ids)
        self.assertIn("corey_lactone_prostaglandin", ids)

    def test_benchmark_pack_replays_literature_templates_end_to_end(self):
        cases = [
            P0BenchmarkCase(
                case_id="bench_buf",
                target_name="Bufotalin-like frontier",
                target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
                family_hint="bufotalin, bufadienolide, steroid, pyrone",
                expected_reaction_classes=["C_C_coupling"],
            ),
            P0BenchmarkCase(
                case_id="bench_taxane",
                target_name="Paclitaxel taxane semisynthesis",
                target_smiles="CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12",
                family_hint="paclitaxel, taxane, baccatin, 10-deacetylbaccatin III",
                expected_reaction_classes=["taxane_side_chain_acylation"],
            ),
            P0BenchmarkCase(
                case_id="bench_artemisinin",
                target_name="Artemisinin peroxide anchor",
                target_smiles="CC(C)C1OC2OOCC1CC2=O",
                family_hint="artemisinin, sesquiterpene peroxide, dihydroartemisinic acid",
                expected_reaction_classes=["late_stage_peroxide_formation"],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_p0_benchmark_pack(output_root=tmp, cases=cases, query_budget=8)

        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["failed"], 0, report)
        self.assertTrue(report["hard_gates"]["all_cases_accept_validation"])
        self.assertTrue(report["hard_gates"]["all_cases_have_expected_templates"])
        self.assertTrue(report["hard_gates"]["no_case_claims_solved"])
        self.assertTrue(report["hard_gates"]["all_cases_have_route_map"])


if __name__ == "__main__":
    unittest.main()
