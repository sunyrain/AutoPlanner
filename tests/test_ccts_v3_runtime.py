import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from cascade_planner.eval.train_ccts_v3_runtime_pairwise_ranker import (
    FittedRuntimeModel,
    _feature_names,
    _model_specs,
)
from cascade_planner.route_tree.ccts_v3_runtime import CCTSV3Runtime, DEFAULT_CCTS_V3_MODEL_NAME
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


class CCTSV3RuntimeTest(unittest.TestCase):
    def test_scores_actions_with_runtime_safe_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_program_manifest(root)
            model_path = self._write_model(root)

            runtime = CCTSV3Runtime(
                model_path,
                program_manifest=manifest,
                model_name=DEFAULT_CCTS_V3_MODEL_NAME,
            )
            state = RouteTreeState.initial("CCO")
            actions = [
                CandidateAction(product="CCO", reactants=("CC",), main_reactant="CC", rxn_smiles="CC>>CCO", raw_score=0.9, rank=1),
                CandidateAction(product="CCO", reactants=("O",), main_reactant="O", rxn_smiles="O>>CCO", raw_score=0.1, rank=2),
            ]

            result = runtime.score_actions(state, "CCO", actions, max_depth=2)

            self.assertTrue(result.active)
            self.assertEqual(result.reason, "ccts_v3_runtime")
            self.assertEqual(len(result.scores), 2)
            self.assertEqual(len(result.normalized_scores), 2)
            self.assertEqual(len(result.rows), 2)
            self.assertIn("model", result.score_columns)
            self.assertIn("runtime_any", result.score_columns)
            self.assertIn("runtime_inferred_transform", result.rows[0])

    def test_search_hook_applies_ccts_v3_bonus(self):
        class FakeCCTS:
            def score_actions(self, state, leaf, actions, *, max_depth):
                del state, leaf, actions, max_depth

                class Result:
                    scores = [2.0, -1.0]
                    normalized_scores = [1.0, -1.0]
                    active = True
                    reason = "ccts_v3_runtime"
                    selected_score = "model"
                    rows = [{"selected_rank": 1, "chem_rank": 2}, {"selected_rank": 2, "chem_rank": 1}]

                return Result()

        planner = NeuralGuidedAOSearch(retro_engine=None, controller=None, max_depth=2)
        planner.ccts_scorer = FakeCCTS()
        state = RouteTreeState.initial("CCO")
        actions = [
            CandidateAction(product="CCO", reactants=("CC",), main_reactant="CC", rank=1),
            CandidateAction(product="CCO", reactants=("O",), main_reactant="O", rank=2),
        ]
        rows = [{"total": 0.0, "cost_model": "base"}, {"total": 0.0, "cost_model": "base"}]

        updated = planner._apply_ccts_selection_scores(state, "CCO", actions, rows)

        self.assertEqual(updated[0]["cost_model"], "base+ccts_v3")
        self.assertTrue(updated[0]["ccts_active"])
        self.assertGreater(updated[0]["total"], updated[1]["total"])
        self.assertEqual(updated[0]["ccts_rank"], 1)

    def _write_program_manifest(self, root: Path) -> Path:
        train = root / "train.jsonl"
        train.write_text(
            json.dumps(
                {
                    "program_id": "p1",
                    "steps": [
                        {
                            "transition_id": "t1",
                            "transformation_superclass": "reduction",
                            "product_smiles": "CCO",
                            "main_reactant": "CC",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        graph = root / "train_evidence_graph.json"
        graph.write_text("{}", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"outputs": {"train": str(train), "train_evidence_graph": str(graph)}}),
            encoding="utf-8",
        )
        return manifest

    def _write_model(self, root: Path) -> Path:
        feature_names = _feature_names()
        feature_indices = _model_specs(feature_names)["runtime_evidence_only"]
        x = np.vstack([
            np.ones(len(feature_indices), dtype=np.float32),
            np.zeros(len(feature_indices), dtype=np.float32),
            -np.ones(len(feature_indices), dtype=np.float32),
        ])
        y = np.asarray([1, 0, 0], dtype=np.int32)
        model = LogisticRegression(solver="liblinear", random_state=1).fit(x, y)
        fitted = FittedRuntimeModel(
            model=model,
            feature_indices=feature_indices,
            mean=np.zeros(len(feature_indices), dtype=np.float32),
            std=np.ones(len(feature_indices), dtype=np.float32),
            train_label_key="unit",
            c_value=1.0,
        )
        path = root / "model.pkl"
        with path.open("wb") as fh:
            pickle.dump({"models": {DEFAULT_CCTS_V3_MODEL_NAME: fitted}, "metadata": {}}, fh)
        return path


if __name__ == "__main__":
    unittest.main()
