import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.replay_route_block_value_model import replay_route_block_value_model
from cascade_planner.eval.train_route_block_value_model import train_route_block_value_model


class ReplayRouteBlockValueModelTest(unittest.TestCase):
    def test_replays_trained_model_as_final_reranker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split in ("train", "val", "test"):
                rows.extend(
                    [
                        _row(
                            split,
                            f"{split}_g1",
                            "native_bad",
                            no_human_positive=False,
                            no_human_negative=True,
                            native_rank=0,
                            route_signal=0.1,
                        ),
                        _row(
                            split,
                            f"{split}_g1",
                            "model_good",
                            no_human_positive=True,
                            no_human_negative=False,
                            native_rank=2,
                            route_signal=0.9,
                        ),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="no_human_consensus_positive",
                negative_task="no_human_consensus_negative",
                exclude_groups=["product_audit"],
                c_values=[0.1],
            )

            report = replay_route_block_value_model(
                pack_jsonl=pack,
                model_pickle=root / "model" / "route_block_value_model.pkl",
                output_json=root / "replay.json",
                output_md=root / "replay.md",
                positive_task="no_human_consensus_positive",
                negative_task="no_human_consensus_negative",
                min_mrr_delta_vs_retrieval=0.0,
            )

        self.assertEqual(report["metrics"]["route_block_value_model"]["mrr_covered"], 1.0)
        self.assertEqual(report["metrics"]["native_rank"]["mrr_covered"], 0.5)
        self.assertEqual(report["metrics"]["route_block_value_model"]["top_route_changed_vs_native"], 1)
        self.assertTrue(report["decision"]["fixed_pool_final_rerank_passed"])

def _row(
    split,
    group,
    route_id,
    *,
    no_human_positive=False,
    no_human_negative=False,
    native_rank=0,
    route_signal=0.0,
):
    return {
        "schema_version": "route_block_value_pack.v1",
        "split": split,
        "selector_group_id": group,
        "target_id": group,
        "target_smiles": "CCO",
        "route_id": route_id,
        "native_rank": native_rank,
        "weak_label_tasks": {
            "no_human_consensus_positive": no_human_positive,
            "no_human_consensus_negative": no_human_negative,
        },
        "product_audit": {"risk_order": 20},
        "feature_groups": {
            "native": {
                "native_rank": float(native_rank),
                "native_inv_rank": 1.0 / float(native_rank + 1),
            },
            "cascade_retrieval": {
                "ccts_v3_runtime_best_route_evidence": route_signal,
            },
            "product_audit": {
                "audit_risk_order": 20.0,
            },
            "route_context": {
                "source_model_count": 1.0,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
