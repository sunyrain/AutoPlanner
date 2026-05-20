import unittest

from scripts.render_route_figures import render_route_svg


class RenderRouteFiguresTest(unittest.TestCase):
    def test_render_route_svg_contains_molecules_and_metadata(self):
        route = {
            "score": 0.42,
            "n_steps": 1,
            "product_audit": {
                "route_class": "triage_fragment",
                "tags": ["carrier_reagent_terminal"],
                "terminal_profile": {"terminal_reactants": ["CCO", "O=Cc1ccccc1"]},
            },
            "steps": [
                {
                    "index": 0,
                    "product": "CCOC(=O)/C=C/c1ccccc1",
                    "main_reactant": "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
                    "aux_reactants": ["O=Cc1ccccc1"],
                    "reaction_smiles": (
                        "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1.O=Cc1ccccc1"
                        ">>CCOC(=O)/C=C/c1ccccc1"
                    ),
                    "reaction_type": "template",
                    "source": "Template proposal",
                    "condition_predictions": [{"Temperature": 25.0}],
                }
            ],
        }

        svg = render_route_svg(route, route_number=1, target_smiles="CCOC(=O)/C=C/c1ccccc1")

        self.assertIn("<svg", svg)
        self.assertIn("Route 1: 1 steps", svg)
        self.assertIn("triage_fragment", svg)
        self.assertIn("Terminal materials", svg)
        self.assertEqual(svg.count("<svg"), 1)


if __name__ == "__main__":
    unittest.main()
