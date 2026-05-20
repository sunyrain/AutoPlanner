import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascadeboard.retrorules_applicator import RetroRulesApplicator
from cascade_planner.cascadeboard.skeleton_planner import _candidates_for_skeleton_slot


class RetroRulesApplicatorTest(unittest.TestCase):
    def test_apply_tiny_retrorule_template(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "templates_rhea.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "TEMPLATE_ID",
                        "SCORE",
                        "TEMPLATE",
                        "REACTIONS",
                        "RADII",
                        "REACTIONS_COUNT",
                        "ECS",
                        "ECS_COUNT",
                        "RADIUS_MIN",
                        "RADIUS_MAX",
                        "VALID",
                        "DATASETS",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "TEMPLATE_ID": "unit:cleave-co",
                    "SCORE": "1.0",
                    "TEMPLATE": "[C:1]-[O:2]>>[C:1].[O:2]",
                    "REACTIONS": "RHEA:1",
                    "RADII": "0",
                    "REACTIONS_COUNT": "3",
                    "ECS": "3.1.1.1",
                    "ECS_COUNT": "1",
                    "RADIUS_MIN": "0",
                    "RADIUS_MAX": "0",
                    "VALID": "True",
                    "DATASETS": "rhea",
                })

            applicator = RetroRulesApplicator(
                [path],
                max_templates=10,
                max_per_ec1=10,
                max_templates_per_query=10,
            )
            direct = applicator.predict("CCO", top_k=3, ec_token="3")
            via_slot = _candidates_for_skeleton_slot(
                {"retrorules": applicator},
                product_smiles="CCO",
                ec1=3,
                skel_type="hydrolysis",
                top_k=3,
            )

        self.assertTrue(direct)
        self.assertEqual(direct[0]["source"], "retrorules_rhea")
        self.assertEqual(direct[0]["main_reactant"], "CC")
        self.assertIn("O", direct[0]["aux_reactants"])
        self.assertEqual(direct[0]["ec"], "3.1.1.1")
        self.assertTrue(via_slot)
        self.assertIn("retrorules_rhea", {cand.get("source") for cand in via_slot})


if __name__ == "__main__":
    unittest.main()
