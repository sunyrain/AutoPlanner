import unittest

from cascade_planner.baselines.route_contract import RouteCandidate, RouteStepCandidate
from cascade_planner.baselines.route_plausibility import (
    audit_first_step_material_gate,
    audit_route_plausibility,
    split_plausible_routes,
)


class RoutePlausibilityTest(unittest.TestCase):
    def test_rejects_large_unexplained_carbon_gain(self):
        route = _route("CC(=O)C(=O)O>>CC[C@@H](O)C[C@@H](O)CC=O")

        audit = audit_route_plausibility(route)

        self.assertFalse(audit["passed"])
        self.assertIn("large_unexplained_carbon_gain", audit["reasons"])
        self.assertIn("large_unexplained_heavy_atom_gain", audit["reasons"])
        self.assertEqual(audit["steps"][0]["carbon_gain"], 4)

    def test_accepts_multi_reactant_route_with_no_large_gain(self):
        route = _route("CCOC(=O)CC(=O)CCl.O=C(O)CC(O)C(=O)O>>CC[C@@H](O)C[C@@H](O)CC=O")

        audit = audit_route_plausibility(route)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["reasons"], [])
        self.assertLessEqual(audit["steps"][0]["carbon_gain"], 2)

    def test_condition_reagent_can_explain_missing_element_source(self):
        route = _route("CO>>COCl")
        route.steps[0].condition_predictions = [{"Reagent": "O=P(Cl)(Cl)Cl", "Score": "0.9"}]

        audit = audit_route_plausibility(route)

        self.assertTrue(audit["passed"])
        step = audit["steps"][0]
        self.assertEqual(step["raw_element_gains"], {"Cl": 1})
        self.assertEqual(step["condition_supported_element_gains"], {"Cl": 1})
        self.assertEqual(step["unexplained_element_gains"], {})

    def test_missing_element_source_still_fails_without_condition_reagent(self):
        route = _route("CO>>COCl")

        audit = audit_route_plausibility(route)

        self.assertFalse(audit["passed"])
        self.assertIn("unexplained_new_element_source", audit["reasons"])
        self.assertEqual(audit["steps"][0]["unexplained_element_gains"], {"Cl": 1})
        self.assertEqual(audit["steps"][0]["unexplained_new_elements"], ["Cl"])

    def test_split_plausible_routes_keeps_only_passing_routes(self):
        bad = _route("CC(=O)C(=O)O>>CC[C@@H](O)C[C@@H](O)CC=O")
        good = _route("CCOC(=O)CC(=O)CCl.O=C(O)CC(O)C(=O)O>>CC[C@@H](O)C[C@@H](O)CC=O")

        plausible, audits = split_plausible_routes([bad, good])

        self.assertEqual(len(audits), 2)
        self.assertEqual(len(plausible), 1)
        self.assertIs(plausible[0][0], good)

    def test_first_step_gate_warns_on_boc_transfer_reagent(self):
        step = _route("NCCO>>CC(C)(C)OC(=O)NCCO").steps[0]
        step.condition_predictions = [{"Reagent": "Boc2O", "Score": "0.9"}]

        gate = audit_first_step_material_gate(step)

        self.assertEqual(gate["decision"], "warn")
        self.assertFalse(gate["hard_reject"])
        self.assertIn("possible_condition_or_transfer_reagent_atom_source", gate["warnings"])
        self.assertEqual(gate["residual_unexplained_element_gains"], {})

    def test_first_step_gate_warns_on_fmoc_transfer_reagent(self):
        step = _route("NCCO>>O=C(NCCO)OCC1c2ccccc2-c2ccccc21").steps[0]
        step.condition_predictions = [{"Reagent": "Fmoc-Cl", "Score": "0.8"}]

        gate = audit_first_step_material_gate(step)

        self.assertEqual(gate["decision"], "warn")
        self.assertFalse(gate["hard_reject"])
        self.assertEqual(gate["residual_unexplained_element_gains"], {})

    def test_first_step_gate_warns_on_tms_transfer_reagent(self):
        step = _route("CCO>>CCO[Si](C)(C)C").steps[0]
        step.condition_predictions = [{"Reagent": "TMSCl", "Score": "0.8"}]

        gate = audit_first_step_material_gate(step)

        self.assertEqual(gate["decision"], "warn")
        self.assertFalse(gate["hard_reject"])
        self.assertEqual(gate["residual_unexplained_element_gains"], {})

    def test_first_step_gate_passes_small_condition_reagent_element_source(self):
        step = _route("CO>>COCl").steps[0]
        step.condition_predictions = [{"Reagent": "O=P(Cl)(Cl)Cl", "Score": "0.9"}]

        gate = audit_first_step_material_gate(step)

        self.assertEqual(gate["decision"], "pass")
        self.assertFalse(gate["hard_reject"])

    def test_first_step_gate_hard_rejects_unexplained_large_scaffold_growth(self):
        step = _route("CC(=O)C(=O)O>>CC[C@@H](O)C[C@@H](O)CC=O").steps[0]

        gate = audit_first_step_material_gate(step)

        self.assertEqual(gate["decision"], "hard_reject")
        self.assertTrue(gate["hard_reject"])
        self.assertIn("large_unexplained_material_growth", gate["hard_reasons"])


def _route(rxn: str) -> RouteCandidate:
    lhs, rhs = rxn.split(">>", 1)
    step = RouteStepCandidate(
        product_smiles=rhs,
        reactant_smiles=[part for part in lhs.split(".") if part],
        rxn_smiles=rxn,
        source_model="ChemEnzyRetroPlanner",
        score=1.0,
        stock_status={part: True for part in lhs.split(".") if part},
    )
    return RouteCandidate(target_smiles=rhs, steps=[step], solved=True, score=1.0)


if __name__ == "__main__":
    unittest.main()
