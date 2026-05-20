import json
import os
import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascadeboard.candidate_ranker import PackCandidateRankerInference
from cascade_planner.eval.train_candidate_ranker_from_pack import (
    build_dataset,
    load_candidate_rows,
    train_candidate_ranker_from_pack,
)


class TrainCandidateRankerFromPackTest(unittest.TestCase):
    def test_train_small_pack_and_write_artifacts(self):
        rows = []
        for target in ["CCO", "CCN"]:
            for route_idx in range(2):
                route_id = f"{target}_{route_idx}"
                rows.append({
                    "route_id": route_id,
                    "target_smiles": target,
                    "product": target,
                    "step_index": 0,
                    "rank": 1,
                    "label": 1.0,
                    "label_type": "benchmark_exact",
                    "weight": 2.0,
                    "gt_available": True,
                    "candidate": {
                        "main_reactant": "CC",
                        "aux_reactants": ["O" if target == "CCO" else "N"],
                        "source": "retrochimera",
                        "score": 0.9,
                    },
                    "features": {
                        "candidate_score": 0.9,
                        "stock_fraction": 1.0,
                        "main_reduction": 0.5,
                        "has_ec": 0.0,
                        "has_evidence": 0.0,
                        "large_aux_penalty": 0.0,
                        "self_loop": 0.0,
                    },
                })
                rows.append({
                    "route_id": route_id,
                    "target_smiles": target,
                    "product": target,
                    "step_index": 0,
                    "rank": 2,
                    "label": 0.0,
                    "label_type": "negative",
                    "weight": 1.0,
                    "gt_available": True,
                    "candidate": {
                        "main_reactant": target,
                        "aux_reactants": [],
                        "source": "retrochimera",
                        "score": 0.1,
                    },
                    "features": {
                        "candidate_score": 0.1,
                        "stock_fraction": 0.0,
                        "main_reduction": 0.0,
                        "has_ec": 0.0,
                        "has_evidence": 0.0,
                        "large_aux_penalty": 0.0,
                        "self_loop": 1.0,
                    },
                })

        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "candidate_ranking.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            loaded = load_candidate_rows(pack)
            dataset = build_dataset(loaded, n_bits=16)
            model_path = Path(td) / "ranker.pt"
            report = train_candidate_ranker_from_pack(
                pack_dir=pack,
                model_output=model_path,
                report_output=Path(td) / "ranker.json",
                md_output=Path(td) / "ranker.md",
                epochs=1,
                batch_size=4,
                n_bits=16,
                hidden=16,
            )

            self.assertTrue((Path(td) / "ranker.pt").exists())
            self.assertTrue((Path(td) / "ranker.json").exists())
            self.assertTrue((Path(td) / "ranker.md").exists())
            infer = PackCandidateRankerInference(model_path)
            score = infer.score_candidate("CCO", {
                "main_reactant": "CC",
                "aux_reactants": ["O"],
                "source": "retrochimera",
                "score": 0.9,
            })

        self.assertEqual(len(loaded), 8)
        self.assertEqual(dataset.x.shape[0], 8)
        self.assertIn("reranking", report)
        self.assertGreaterEqual(report["reranking"]["groups_with_exact_gt"], 1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_default_ranker_can_be_disabled_by_env(self):
        from cascade_planner.cascadeboard.candidate_ranker import default_candidate_ranker

        old = os.environ.get("AUTOPLANNER_DISABLE_CANDIDATE_RANKER")
        try:
            os.environ["AUTOPLANNER_DISABLE_CANDIDATE_RANKER"] = "1"
            self.assertIsNone(default_candidate_ranker())
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_DISABLE_CANDIDATE_RANKER", None)
            else:
                os.environ["AUTOPLANNER_DISABLE_CANDIDATE_RANKER"] = old

    def test_candidate_ranker_weight_is_configurable_and_clamped(self):
        from cascade_planner.cascadeboard.candidate_ranker import candidate_ranker_weight

        old = os.environ.get("AUTOPLANNER_CANDIDATE_RANKER_WEIGHT")
        try:
            os.environ["AUTOPLANNER_CANDIDATE_RANKER_WEIGHT"] = "1.25"
            self.assertEqual(candidate_ranker_weight(), 1.25)
            os.environ["AUTOPLANNER_CANDIDATE_RANKER_WEIGHT"] = "-2"
            self.assertEqual(candidate_ranker_weight(), 0.0)
            os.environ["AUTOPLANNER_CANDIDATE_RANKER_WEIGHT"] = "5"
            self.assertEqual(candidate_ranker_weight(), 3.0)
            os.environ["AUTOPLANNER_CANDIDATE_RANKER_WEIGHT"] = "bad"
            self.assertEqual(candidate_ranker_weight(default=0.4), 0.4)
        finally:
            if old is None:
                os.environ.pop("AUTOPLANNER_CANDIDATE_RANKER_WEIGHT", None)
            else:
                os.environ["AUTOPLANNER_CANDIDATE_RANKER_WEIGHT"] = old


if __name__ == "__main__":
    unittest.main()
