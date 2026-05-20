import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_locked_validation import build_locked_validation


class BuildLockedValidationTest(unittest.TestCase):
    def test_excludes_target_and_reaction_overlaps(self):
        source_rows = [
            {
                "doi": "d1",
                "cascade_id": "c1",
                "target_smiles": "CCO",
                "route_domain": "all_chemical",
                "depth": 2,
                "gt_route": [
                    {"rxn_smiles": "CC.O>>CCO", "transformation": "other"},
                    {"rxn_smiles": "C.C>>CC", "transformation": "other"},
                ],
            },
            {
                "doi": "d2",
                "cascade_id": "c2",
                "target_smiles": "CCN",
                "route_domain": "all_chemical",
                "depth": 2,
                "gt_route": [
                    {"rxn_smiles": "CC.N>>CCN", "transformation": "amination"},
                    {"rxn_smiles": "C.C>>CC", "transformation": "other"},
                ],
            },
            {
                "doi": "d3",
                "cascade_id": "c3",
                "target_smiles": "CCC",
                "route_domain": "all_chemical",
                "depth": 2,
                "gt_route": [
                    {"rxn_smiles": "CC.C>>CCC", "transformation": "coupling"},
                    {"rxn_smiles": "C.C>>CC", "transformation": "other"},
                ],
            },
        ]
        exclude_rows = [
            {
                "target_smiles": "CCO",
                "gt_route": [{"rxn_smiles": "CC.N>>CCN"}],
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.json"
            exclude = root / "exclude.json"
            output = root / "locked.json"
            audit_json = root / "audit.json"
            audit_md = root / "audit.md"
            source.write_text(json.dumps(source_rows), encoding="utf-8")
            exclude.write_text(json.dumps(exclude_rows), encoding="utf-8")

            audit = build_locked_validation(
                source_paths=[source],
                output_path=output,
                audit_json_path=audit_json,
                audit_md_path=audit_md,
                exclude_benchmarks=[exclude],
                limit=5,
                min_depth=2,
            )
            selected = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(audit["overlap"]["locked_safe"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target_smiles"], "CCC")


if __name__ == "__main__":
    unittest.main()
