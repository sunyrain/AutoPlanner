import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_route_block_value_model import train_route_block_value_model


class TrainRouteBlockValueModelTest(unittest.TestCase):
    def test_trains_with_feature_group_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split in ("train", "val", "test"):
                rows.extend(
                    [
                        _row(split, f"{split}_g1", "pos", strong=True, native_rank=3, route_signal=0.8),
                        _row(split, f"{split}_g1", "neg", strong=False, native_rank=0, route_signal=0.1),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="strong_route_evidence",
                exclude_groups=["cascade_retrieval"],
                c_values=[0.1],
            )

        self.assertEqual(report["counts"]["train"]["positive_rows"], 1)
        self.assertTrue(all(not name.startswith("cascade_retrieval.") for name in report["feature_names"]))
        self.assertIn("test", report["model"])
        self.assertEqual(report["baselines"]["retrieval_only"]["test"]["mrr_covered"], 1.0)
        self.assertEqual(report["evidence_provenance_audit"]["status"], "retrieval_baseline_unverifiable")
        self.assertFalse(report["evidence_provenance_audit"]["model_uses_cascade_retrieval_features"])

    def test_negative_task_filters_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split in ("train", "val", "test"):
                rows.extend(
                    [
                        _row(split, f"{split}_g1", "review", reviewable=True, reject=False, native_rank=0),
                        _row(split, f"{split}_g1", "reject", reviewable=False, reject=True, native_rank=1),
                        _row(split, f"{split}_g1", "neutral", reviewable=False, reject=False, native_rank=2),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="reviewable_by_audit",
                negative_task="reject_artifact",
                exclude_groups=["product_audit"],
                c_values=[0.1],
            )

        self.assertEqual(report["counts"]["train"]["rows"], 2)
        self.assertEqual(report["counts"]["train"]["negative_rows"], 1)
        self.assertTrue(all(not name.startswith("product_audit.") for name in report["feature_names"]))

    def test_reports_when_model_uses_unverified_retrieval_features(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split in ("train", "val", "test"):
                rows.extend(
                    [
                        _row(split, f"{split}_g1", "pos", strong=True, native_rank=2, route_signal=0.7),
                        _row(split, f"{split}_g1", "neg", strong=False, native_rank=0, route_signal=0.1),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="strong_route_evidence",
                c_values=[0.1],
            )

        self.assertEqual(
            report["evidence_provenance_audit"]["status"],
            "model_uses_unverified_retrieval_provenance",
        )
        self.assertTrue(report["evidence_provenance_audit"]["model_uses_cascade_retrieval_features"])

    def test_trains_from_expert_review_tasks_after_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split in ("train", "val", "test"):
                rows.extend(
                    [
                        _row(split, f"{split}_g1", "expert_pos", expert_positive=True, native_rank=2, route_signal=0.6),
                        _row(split, f"{split}_g1", "expert_neg", expert_negative=True, native_rank=0, route_signal=0.2),
                        _row(split, f"{split}_g1", "unreviewed", native_rank=1, route_signal=0.4),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="expert_review_positive",
                negative_task="expert_review_negative",
                exclude_groups=["product_audit"],
                c_values=[0.1],
            )

        self.assertEqual(report["counts"]["train"]["rows"], 2)
        self.assertEqual(report["counts"]["train"]["positive_rows"], 1)
        self.assertEqual(report["counts"]["train"]["negative_rows"], 1)
        self.assertEqual(report["metadata"]["positive_task"], "expert_review_positive")

    def test_trains_from_no_human_consensus_tasks_without_expert_labels(self):
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
                            "auto_pos",
                            no_human_positive=True,
                            no_human_negative=False,
                            native_rank=2,
                            route_signal=0.8,
                        ),
                        _row(
                            split,
                            f"{split}_g1",
                            "auto_neg",
                            no_human_positive=False,
                            no_human_negative=True,
                            native_rank=0,
                            route_signal=0.1,
                        ),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_block_value_model(
                pack_jsonl=pack,
                output_dir=root / "model",
                positive_task="no_human_consensus_positive",
                negative_task="no_human_consensus_negative",
                exclude_groups=["product_audit", "cascade_retrieval"],
                c_values=[0.1],
            )

        self.assertEqual(report["metadata"]["positive_task"], "no_human_consensus_positive")
        self.assertEqual(report["counts"]["train"]["positive_rows"], 1)
        self.assertEqual(report["counts"]["train"]["negative_rows"], 1)
        self.assertTrue(all(not name.startswith("product_audit.") for name in report["feature_names"]))
        self.assertTrue(all(not name.startswith("cascade_retrieval.") for name in report["feature_names"]))


def _row(
    split,
    group,
    route_id,
    *,
    strong=False,
    reviewable=False,
    reject=False,
    expert_positive=False,
    expert_negative=False,
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
            "strong_route_evidence": strong,
            "reviewable_by_audit": reviewable,
            "reject_artifact": reject,
            "expert_review_positive": expert_positive,
            "expert_review_negative": expert_negative,
            "no_human_consensus_positive": no_human_positive,
            "no_human_consensus_negative": no_human_negative,
        },
        "product_audit": {"risk_order": 0 if reviewable else 20},
        "feature_groups": {
            "native": {
                "native_rank": float(native_rank),
                "native_inv_rank": 1.0 / float(native_rank + 1),
            },
            "cascade_retrieval": {
                "ccts_v3_runtime_best_route_evidence": route_signal,
            },
            "product_audit": {
                "audit_risk_order": 0.0 if reviewable else 20.0,
            },
            "route_context": {
                "source_model_count": 1.0,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
