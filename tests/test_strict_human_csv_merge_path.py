import csv
import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_route_block_review_label_pack import build_route_block_review_label_pack
from cascade_planner.eval.merge_route_block_review_labels import merge_route_block_review_labels
from cascade_planner.eval.run_route_pool_evidence_review_csv_pipeline import run_route_pool_evidence_review_csv_pipeline


class StrictHumanCsvMergePathTest(unittest.TestCase):
    def test_filled_csv_preserves_route_ids_and_can_pass_merge_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "filled.csv"
            value_pack = root / "value_pack.jsonl"
            out_dir = root / "csv_pipeline"
            review_pack = out_dir / "strict_model_human_review_label_pack.jsonl"
            review_report = out_dir / "strict_model_human_review_label_pack_report.json"
            merged_pack = out_dir / "strict_model_human_merged_route_block_value_pack.jsonl"
            merge_report = out_dir / "strict_model_human_merged_route_block_value_pack_report.json"

            _write_value_pack(value_pack)
            _write_filled_csv(csv_path)

            manifest = run_route_pool_evidence_review_csv_pipeline(
                review_csv=csv_path,
                output_dir=out_dir,
                prefix="strict_model_human",
                min_rows=1,
                min_usable_positive=1,
                min_usable_negative=1,
                min_evidence_classes=1,
                min_auc=0.0,
            )
            label_pack_report = build_route_block_review_label_pack(
                inputs=[out_dir / "strict_model_human_labels.jsonl"],
                output_jsonl=review_pack,
                report_json=review_report,
                dataset="strict_model_human_review_labels",
            )
            merge = merge_route_block_review_labels(
                value_pack_jsonl=value_pack,
                review_label_pack_jsonl=review_pack,
                output_jsonl=merged_pack,
                report_json=merge_report,
                dataset="strict_model_human_merged_route_block_value",
                min_usable_positive=1,
                min_usable_negative=1,
            )

        self.assertEqual(manifest["summaries"]["labels"]["accepted_rows"], 6)
        self.assertEqual(label_pack_report["counts"]["usable_positive_rows"], 3)
        self.assertEqual(label_pack_report["counts"]["usable_negative_rows"], 3)
        self.assertTrue(label_pack_report["decision"]["ready_for_route_block_merge_evaluation"])
        self.assertEqual(merge["counts"]["matched_route_ids"], 6)
        self.assertEqual(merge["counts"]["review_rows_without_route_id"], 0)
        self.assertTrue(merge["decision"]["ready_for_expert_training"])
        self.assertEqual(
            merge["decision"]["split_ready"],
            {"train": True, "val": True, "test": True},
        )


def _write_value_pack(path: Path) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for label in ("pos", "neg"):
            rows.append(
                {
                    "schema_version": "route_block_value_pack.v1",
                    "dataset": "synthetic",
                    "route_id": f"{split}-{label}",
                    "target_id": f"target-{split}",
                    "target_smiles": f"C{split}{label}",
                    "split": split,
                    "weak_label_tasks": {},
                    "features": {},
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_filled_csv(path: Path) -> None:
    fieldnames = [
        "review_id",
        "target_id",
        "route_id",
        "value_split",
        "source_pool",
        "evidence_class",
        "target_smiles",
        "native_rank",
        "n_steps",
        "stock_closed",
        "model_rank",
        "retrieval_rank",
        "audit_rank",
        "upstream_transform",
        "downstream_transform",
        "upstream_rxn",
        "downstream_rxn",
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_risk_tags",
        "expert_comments",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            writer.writerow(_csv_row(split=split, label="pos", positive=True))
            writer.writerow(_csv_row(split=split, label="neg", positive=False))


def _csv_row(*, split: str, label: str, positive: bool) -> dict[str, str]:
    base = {
        "review_id": f"{split}-{label}-review",
        "target_id": f"target-{split}",
        "route_id": f"{split}-{label}",
        "value_split": split,
        "source_pool": "synthetic_pool",
        "evidence_class": "strict_model_control_disagreement",
        "target_smiles": f"C{split}{label}",
        "native_rank": "1" if positive else "2",
        "n_steps": "2",
        "stock_closed": "true" if positive else "false",
        "model_rank": "1" if positive else "3",
        "retrieval_rank": "3" if positive else "1",
        "audit_rank": "3" if positive else "1",
        "upstream_transform": "A",
        "downstream_transform": "B",
        "upstream_rxn": "CC>>C",
        "downstream_rxn": "C>>CO" if positive else "C>>CN",
        "expert_comments": "synthetic reviewed block",
    }
    if positive:
        base.update(
            {
                "expert_route_plausible": "yes",
                "expert_block_transform_correct": "yes",
                "expert_support_precedent_relevant": "yes",
                "expert_cascade_coherent": "yes",
                "expert_priority": "high",
            }
        )
    else:
        base.update(
            {
                "expert_route_plausible": "no",
                "expert_block_transform_correct": "no",
                "expert_support_precedent_relevant": "no",
                "expert_cascade_coherent": "no",
                "expert_priority": "reject",
                "expert_reject_reason": "wrong_transform_label",
                "expert_risk_tags": "wrong_transform_label",
            }
        )
    return base


if __name__ == "__main__":
    unittest.main()
