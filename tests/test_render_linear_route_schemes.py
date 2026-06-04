import unittest

from scripts.render_linear_route_schemes import _renderable_feasible_route, render_scheme_svg


class RenderLinearRouteSchemesTest(unittest.TestCase):
    def test_render_scheme_svg_uses_arrows_and_conditions(self):
        route = {
            "score": 0.5,
            "n_steps": 1,
            "product_audit": {
                "route_class": "triage_fragment",
                "tags": ["carrier_reagent_terminal"],
                "condition_audit": {
                    "route_risk": "warn",
                    "warning_step_count": 1,
                    "high_risk_step_count": 0,
                    "temperature_span_c": 0,
                    "steps": [{"step_index": 1, "risk": "warn", "issues": ["low_condition_score"]}],
                },
            },
            "steps": [
                {
                    "product": "CCOC(=O)/C=C/c1ccccc1",
                    "main_reactant": "O=Cc1ccccc1",
                    "aux_reactants": ["CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1"],
                    "reaction_smiles": (
                        "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1.O=Cc1ccccc1"
                        ">>CCOC(=O)/C=C/c1ccccc1"
                    ),
                    "reaction_type": "template",
                    "condition_predictions": [{"Temperature": 56.0, "Solvent": "Cc1ccccc1"}],
                }
            ],
        }

        svg = render_scheme_svg(route, route_number=1, target_smiles="CCOC(=O)/C=C/c1ccccc1")

        self.assertIn("Route 01 (1 steps)", svg)
        self.assertIn('class="arrow"', svg)
        self.assertNotIn("template", svg)
        self.assertNotIn("planner", svg.lower())
        self.assertIn("56 °C", svg)
        self.assertIn("toluene", svg)
        self.assertIn("toluene; 56 °C ?", svg)
        self.assertIn("condition-audit warnings", svg)
        self.assertNotIn('class="reagent"', svg)
        self.assertNotIn("CCOC(=O)/C=C", svg)
        self.assertNotIn("O=Cc1ccccc1", svg)
        self.assertNotIn("continued from previous row", svg)
        self.assertEqual(svg.count("<svg"), 1)

    def test_render_scheme_labels_dichloromethane_as_dcm(self):
        route = {
            "score": 0.9,
            "n_steps": 1,
            "steps": [
                {
                    "product": "CCOC(C)=O",
                    "main_reactant": "CCO",
                    "aux_reactants": ["CC(=O)OC(C)=O"],
                    "reaction_smiles": "CCO.CC(=O)OC(C)=O>>CCOC(C)=O",
                    "condition_predictions": [{"Reagent": "CC(=O)OC(C)=O", "Solvent": "ClCCl", "Temperature": 25}],
                }
            ],
        }

        svg = render_scheme_svg(route, route_number=1, target_smiles="CCOC(C)=O")

        self.assertIn("DCM; 25 °C", svg)
        self.assertNotIn("DCE; 25 °C", svg)

    def test_renderable_feasible_route_keeps_source_supported_anchor(self):
        anchor = {
            "metrics": {
                "source_supported_semisynthesis": True,
                "cascade_verifier": {"feasible": False},
            },
            "steps": [{"condition_predictions": []}],
        }
        good_native = {
            "metrics": {"cascade_verifier": {"feasible": True}},
            "steps": [{"condition_predictions": [{"Temperature": 25}]}],
        }
        no_condition_native = {
            "metrics": {"cascade_verifier": {"feasible": True}},
            "steps": [{"condition_predictions": []}],
        }
        bad_native = {
            "metrics": {"cascade_verifier": {"feasible": False}},
            "steps": [{"condition_predictions": [{"Temperature": 25}]}],
        }

        self.assertTrue(_renderable_feasible_route(anchor))
        self.assertTrue(_renderable_feasible_route(good_native))
        self.assertFalse(_renderable_feasible_route(no_condition_native))
        self.assertFalse(_renderable_feasible_route(bad_native))


if __name__ == "__main__":
    unittest.main()
