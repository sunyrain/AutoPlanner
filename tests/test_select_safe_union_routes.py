import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.select_safe_union_routes import select_safe_union_routes


class SafeUnionSelectionTest(unittest.TestCase):
    def test_selects_union_only_when_target_level_audit_does_not_regress(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "native.json"
            union = root / "union.json"
            output = root / "selected.json"
            report = root / "report.json"
            native.write_text(json.dumps(_native_run()), encoding="utf-8")
            union.write_text(json.dumps(_union_run()), encoding="utf-8")

            result = select_safe_union_routes(native_run=native, union_run=union, output=output, report=report)
            selected = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["decision_counts"], {"native": 1, "union": 1})
        self.assertEqual(result["reason_counts"], {"artifact_regression": 1, "triage_gain": 1})
        self.assertEqual(result["selected_ranked_product_metrics"]["top3_product_usable_rate"], 0.5)
        self.assertEqual(result["delta_selected_minus_native"]["top3_triage_signal_rate"], 0.5)
        self.assertEqual(selected["targets"][0]["planner_output"]["routes"][0]["score"], 0.9)
        self.assertEqual(selected["targets"][1]["planner_output"]["routes"][0]["score"], 0.8)


def _native_run():
    return {
        "targets": [
            _target(0, "CC=O", _route("CCO>>CC=O", ["CCO"], score=0.9)),
            _target(1, "Cc1ccccc1", _route("CC>>Cc1ccccc1", ["CC"], score=0.1)),
        ]
    }


def _union_run():
    return {
        "targets": [
            _target(
                0,
                "CC=O",
                _route(
                    "CCO>>CC=O",
                    ["CCO"],
                    score=0.2,
                    naturalness={"atom_balance_violations": 1},
                ),
            ),
            _target(
                1,
                "Cc1ccccc1",
                _route(
                    "Cc1ccccc1Br.C>>Cc1ccccc1",
                    ["Cc1ccccc1Br", "C"],
                    reaction_class="C_C_coupling",
                    score=0.8,
                ),
            ),
        ]
    }


def _target(index, smiles, route):
    return {
        "index": index,
        "target_smiles": smiles,
        "metrics": {"strict_stock_solve_any": True},
        "planner_output": {"routes": [route]},
    }


def _route(rxn, terminals, *, reaction_class="unknown", score=0.0, naturalness=None):
    return {
        "score": score,
        "steps": [
            {
                "reaction_smiles": rxn,
                "source": "test",
                "reaction_interpretation": {
                    "reaction_class": reaction_class,
                    "atom_change": {"heavy_atom_delta": 0},
                },
            }
        ],
        "metrics": {
            "strict_stock_solve": True,
            "route_solved": True,
            "filled_route": True,
            "terminal_reactants": terminals,
            "route_naturalness": naturalness or {},
        },
    }


if __name__ == "__main__":
    unittest.main()
