import unittest

from scripts.render_linear_route_schemes import render_scheme_svg


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


if __name__ == "__main__":
    unittest.main()
