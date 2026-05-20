import tempfile
import unittest
from pathlib import Path

from scripts.render_route_trees import build_route_dot


class RenderRouteTreesTest(unittest.TestCase):
    def test_build_route_dot_is_connected_tree(self):
        route = {
            "score": 0.5,
            "n_steps": 2,
            "product_audit": {
                "route_class": "triage_fragment",
                "tags": ["carrier_reagent_terminal"],
                "terminal_profile": {"terminal_reactants": ["CCO", "O"]},
            },
            "steps": [
                {
                    "product": "CCCO",
                    "main_reactant": "CCO",
                    "aux_reactants": ["C=O"],
                    "reaction_smiles": "CCO.C=O>>CCCO",
                    "reaction_type": "template",
                    "source": "Template proposal",
                },
                {
                    "product": "C=O",
                    "main_reactant": "CO",
                    "reaction_smiles": "CO>>C=O",
                    "reaction_type": "enzymatic",
                    "source": "CascadePlanner enzyme module",
                    "ec": "1.1.1.1",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dot = build_route_dot(
                route,
                route_number=1,
                target_smiles="CCCO",
                image_dir=Path(tmp),
                mol_width=120,
                mol_height=90,
            )

        self.assertIn("digraph RouteTree", dot)
        self.assertIn("rankdir=LR", dot)
        self.assertIn("Route 01", dot)
        self.assertIn("S1.1", dot)
        self.assertIn("S2.1", dot)


if __name__ == "__main__":
    unittest.main()
