import unittest

from scripts.reaudit_route_pool import refresh_route_pool_audit


class ReauditRoutePoolTest(unittest.TestCase):
    def test_refresh_route_pool_adds_condition_audit_summary(self):
        payload = {
            "target": "CCO",
            "routes": [
                {
                    "score": 0.8,
                    "steps": [
                        {
                            "product": "CCO",
                            "main_reactant": "CC=O",
                            "aux_reactants": [],
                            "reaction_smiles": "CC=O>>CCO",
                            "reaction_type": "enzymatic",
                            "source": "enzexpand",
                            "condition_predictions": [{"Temperature": -60.0, "Solvent": "Cc1ccccc1", "Score": "0.4"}],
                            "reaction_interpretation": {
                                "reaction_class": "reduction",
                                "atom_change": {"heavy_atom_delta": 0},
                            },
                        }
                    ],
                    "metrics": {
                        "strict_stock_solve": True,
                        "route_solved": True,
                        "filled_route": True,
                        "terminal_reactants": ["CC=O"],
                    },
                }
            ],
        }

        refreshed, summary = refresh_route_pool_audit(payload, target_id="test", mode="preserve")

        audit = refreshed["routes"][0]["product_audit"]
        self.assertEqual(audit["condition_audit"]["route_risk"], "warn")
        self.assertIn("condition_warning", audit["issues"])
        self.assertEqual(summary["condition"]["route_risk_counts"]["warn"], 1)
        self.assertEqual(refreshed["route_condition_audit_summary"]["route_risk_counts"]["warn"], 1)


if __name__ == "__main__":
    unittest.main()
