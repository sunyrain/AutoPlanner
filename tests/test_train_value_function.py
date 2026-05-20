import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_value_function import (
    collect_training_pack_rows,
    collect_training_rows,
    train_value_models,
    train_value_models_from_training_pack,
)


def _metrics(solved):
    return {
        "filled_route": True,
        "progressive_route": bool(solved),
        "route_solved": bool(solved),
        "strict_stock_solve": bool(solved),
        "retrosynthesis_progress": {
            "main_chain_reduction": 0.7 if solved else 0.0,
            "largest_leaf_reduction": 0.7 if solved else 0.0,
        },
        "route_naturalness": {"naturalness_score": 1.0 if solved else 0.0},
        "condition": {"condition_window_success": True},
        "cascade_compatibility": {
            "cascade_compatibility_success": bool(solved),
            "issues": [] if solved else ["route_cycle"],
        },
        "enzyme_evidence": {"enzyme_evidence_score": 0.0},
    }


class TrainValueFunctionTest(unittest.TestCase):
    def test_collect_and_train_from_exported_routes(self):
        route = {
            "metrics": _metrics(True),
            "steps": [{
                "product": "CCCCCC",
                "main_reactant": "CC",
                "stock_status": {"CC": True, "CO": True, "CCCCCC": False},
                "candidate_pool": {
                    "top_candidates": [
                        {"main_reactant": "CC", "aux_reactants": ["CO"], "score": 0.9},
                        {"main_reactant": "CCCCCC", "aux_reactants": [], "score": 0.2},
                    ],
                },
            }],
        }
        bad_route = {
            "metrics": _metrics(False),
            "steps": [{
                "product": "CCCCCC",
                "main_reactant": "CCCCCC",
                "stock_status": {"CC": True, "CCCCCC": False},
                "candidate_pool": {
                    "top_candidates": [
                        {"main_reactant": "CCCCCC", "aux_reactants": [], "score": 0.8},
                        {"main_reactant": "CC", "aux_reactants": [], "score": 0.1},
                    ],
                },
            }],
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "routes.json"
            path.write_text(json.dumps({"routes": [route, bad_route]}), encoding="utf-8")
            board_rows, candidate_rows, metadata = collect_training_rows([path])
            payload = train_value_models([path], epochs=3, lr=0.01)

        self.assertEqual(metadata["board_samples"], 2)
        self.assertEqual(len(candidate_rows), 4)
        self.assertEqual(sum(int(row.label) for row in board_rows), 1)
        self.assertIn("board_weights", payload)
        self.assertIn("candidate_weights", payload)
        self.assertEqual(payload["metadata"]["candidate_training"]["positive"], 2)
        self.assertLessEqual(payload["candidate_weights"]["large_aux_penalty"], 0.0)
        self.assertLessEqual(payload["candidate_weights"]["self_loop"], 0.0)

    def test_train_from_training_pack_jsonl(self):
        route_rows = [
            {"features": {"filled_route": 1, "progressive_route": 1, "route_solved": 1, "strict_stock_solve": 1}, "label": 1.0},
            {"features": {"filled_route": 1, "progressive_route": 0, "route_solved": 0, "strict_stock_solve": -0.5, "issue_count": 2}, "label": 0.0},
        ]
        candidate_rows = [
            {"features": {"candidate_score": 0.9, "stock_fraction": 1.0, "main_reduction": 0.7}, "label": 1.0, "weight": 2.0},
            {"features": {"candidate_score": 0.1, "stock_fraction": 0.0, "main_reduction": 0.0, "self_loop": 1.0}, "label": 0.0},
        ]

        with tempfile.TemporaryDirectory() as td:
            pack = Path(td)
            (pack / "route_value.jsonl").write_text("\n".join(json.dumps(row) for row in route_rows), encoding="utf-8")
            (pack / "candidate_ranking.jsonl").write_text("\n".join(json.dumps(row) for row in candidate_rows), encoding="utf-8")
            boards, cands, metadata = collect_training_pack_rows(pack)
            payload = train_value_models_from_training_pack(pack, epochs=3, lr=0.01)

        self.assertEqual(metadata["board_samples"], 2)
        self.assertEqual(metadata["candidate_samples"], 2)
        self.assertEqual(len(boards), 2)
        self.assertEqual(len(cands), 2)
        self.assertIn("board_weights", payload)
        self.assertIn("candidate_weights", payload)
        self.assertEqual(payload["metadata"]["label_source"], "training_pack_route_and_candidate_jsonl")


if __name__ == "__main__":
    unittest.main()
