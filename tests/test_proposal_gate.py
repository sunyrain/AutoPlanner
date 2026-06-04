import unittest

from cascade_planner.baselines.proposal_gate import evaluate_step_candidate, gate_web_route


class ProposalGateTest(unittest.TestCase):
    def test_rejects_small_sulfonates_making_steroid_core(self):
        product = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
        report = evaluate_step_candidate(
            product_smiles=product,
            reactant_smiles=["CC(C)(C)OS(C)(=O)=O", "CC(C)(C)OS(=O)(=O)O"],
            rxn_smiles=f"CC(C)(C)OS(C)(=O)=O.CC(C)(C)OS(=O)(=O)O>>{product}",
            condition_predictions=[{"Reagent": "[Li]CCCC", "Temperature": -61.7}],
        )

        self.assertEqual(report["decision"], "reject")
        self.assertTrue(report["hard_reject"])
        self.assertIn("large_unexplained_heavy_atom_gain", report["hard_reasons"])
        self.assertIn("unexplained_complex_core_growth", report["hard_reasons"])
        self.assertEqual(report["frontier_reason"], "complex_core_unresolved")
        self.assertTrue(
            any(row["role"] == "sulfonate_or_sulfate_reagent" for row in report["recognized_reagent_roles"])
        )

    def test_condition_reagent_can_explain_small_protecting_group(self):
        report = evaluate_step_candidate(
            product_smiles="CO[Si](C)(C)C(C)(C)C",
            reactant_smiles=["CO"],
            rxn_smiles="CO>>CO[Si](C)(C)C(C)(C)C",
            condition_predictions=[{"Reagent": "CC(C)(C)[Si](C)(C)Cl"}],
        )

        self.assertEqual(report["decision"], "keep")
        self.assertFalse(report["hard_reject"])
        self.assertEqual(report["hard_reasons"], [])

    def test_web_route_reports_frontier_for_rejected_step(self):
        product = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
        route = {
            "route_rank": 7,
            "n_steps": 1,
            "steps": [
                {
                    "index": 0,
                    "product": product,
                    "main_reactant": "CC(C)(C)OS(C)(=O)=O",
                    "aux_reactants": ["CC(C)(C)OS(=O)(=O)O"],
                    "reaction_smiles": f"CC(C)(C)OS(C)(=O)=O.CC(C)(C)OS(=O)(=O)O>>{product}",
                    "condition_predictions": [{"Reagent": "[Li]CCCC"}],
                }
            ],
        }

        report = gate_web_route(route)

        self.assertEqual(report["decision"], "reject")
        self.assertEqual(report["frontier"]["smiles"], product)
        self.assertEqual(route["steps"][0]["proposal_gate"]["decision"], "reject")

    def test_web_route_rejects_unsupported_prenyl_terminal_without_enzyme_evidence(self):
        target = "CC(=O)O[C@H]1C[C@@]2(O)CCC3CCCCC3C2C1"
        prenyl_terminal = (
            "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
            "CC/C(C)=C/CC/C(C)=C/CO"
        )
        route = {
            "route_rank": 1,
            "n_steps": 4,
            "metrics": {"terminal_reactants": [prenyl_terminal]},
            "steps": [
                {
                    "index": idx,
                    "product": "C" * (idx + 2),
                    "main_reactant": "C" * (idx + 1),
                    "reaction_smiles": f"{'C' * (idx + 1)}>>{'C' * (idx + 2)}",
                    "condition_predictions": [{"condition_label": "RCR model prediction"}],
                }
                for idx in range(4)
            ],
        }

        report = gate_web_route(route)

        self.assertEqual(report["decision"], "reject")
        self.assertIn("unsupported_biosynthetic_prenyl_terminal", report["route_hard_reasons"])
        self.assertIn("unsupported_biosynthetic_prenyl_terminal", report["reason_counts"])
        self.assertEqual(report["frontier"]["smiles"], prenyl_terminal)

    def test_web_route_keeps_enzyme_supported_prenyl_terminal(self):
        prenyl_terminal = (
            "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/"
            "CC/C(C)=C/CC/C(C)=C/CO"
        )
        route = {
            "route_rank": 1,
            "n_steps": 4,
            "metrics": {"terminal_reactants": [prenyl_terminal]},
            "steps": [
                {
                    "index": 0,
                    "product": "CC",
                    "main_reactant": "C",
                    "reaction_smiles": "C>>CC",
                    "is_enzymatic": True,
                    "ec": "2.5.1.21",
                    "condition_predictions": [{"condition_label": "RCR model prediction"}],
                },
            ],
        }

        report = gate_web_route(route)

        self.assertEqual(report["decision"], "keep")
        self.assertEqual(report["route_hard_reasons"], [])


if __name__ == "__main__":
    unittest.main()
