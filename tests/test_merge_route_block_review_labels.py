import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.merge_route_block_review_labels import merge_route_block_review_labels


class MergeRouteBlockReviewLabelsTest(unittest.TestCase):
    def test_merges_review_tasks_by_route_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            value_pack = root / "value_pack.jsonl"
            review_pack = root / "review_pack.jsonl"
            value_rows = [
                _value_row(route_id="r1", split="train"),
                _value_row(route_id="r2", split="train"),
                _value_row(route_id="r3", split="val"),
                _value_row(route_id="r4", split="val"),
                _value_row(route_id="r5", split="test"),
                _value_row(route_id="r6", split="test"),
            ]
            value_pack.write_text(
                "\n".join(json.dumps(row) for row in value_rows) + "\n",
                encoding="utf-8",
            )
            review_rows = [
                _review_row(route_id="r1", positive=True),
                _review_row(route_id="r2", negative=True),
                _review_row(route_id="r3", positive=True),
                _review_row(route_id="r4", negative=True),
                _review_row(route_id="r5", positive=True),
                _review_row(route_id="r6", negative=True),
            ]
            review_pack.write_text(
                "\n".join(json.dumps(row) for row in review_rows) + "\n",
                encoding="utf-8",
            )

            report = merge_route_block_review_labels(
                value_pack_jsonl=value_pack,
                review_label_pack_jsonl=review_pack,
                output_jsonl=root / "merged.jsonl",
                report_json=root / "report.json",
                min_usable_positive=1,
                min_usable_negative=1,
            )
            rows = [
                json.loads(line)
                for line in (root / "merged.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        by_id = {row["route_id"]: row for row in rows}
        self.assertTrue(by_id["r1"]["weak_label_tasks"]["expert_review_positive"])
        self.assertFalse(by_id["r1"]["weak_label_tasks"]["expert_review_negative"])
        self.assertTrue(by_id["r2"]["weak_label_tasks"]["expert_review_negative"])
        self.assertEqual(report["counts"]["matched_route_ids"], 6)
        self.assertTrue(all(report["decision"]["split_ready"].values()))
        self.assertTrue(report["decision"]["ready_for_expert_training"])

    def test_placeholder_reviews_do_not_become_training_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            value_pack = root / "value_pack.jsonl"
            review_pack = root / "review_pack.jsonl"
            value_pack.write_text(json.dumps(_value_row(route_id="r1")) + "\n", encoding="utf-8")
            review_pack.write_text(json.dumps(_review_row(route_id="r1", placeholder=True)) + "\n", encoding="utf-8")

            report = merge_route_block_review_labels(
                value_pack_jsonl=value_pack,
                review_label_pack_jsonl=review_pack,
                output_jsonl=root / "merged.jsonl",
                report_json=root / "report.json",
            )
            row = json.loads((root / "merged.jsonl").read_text(encoding="utf-8"))

        self.assertTrue(row["weak_label_tasks"]["expert_reviewed"])
        self.assertTrue(row["weak_label_tasks"]["expert_review_placeholder"])
        self.assertFalse(row["weak_label_tasks"]["expert_review_positive"])
        self.assertFalse(row["weak_label_tasks"]["expert_review_negative"])
        self.assertFalse(report["decision"]["ready_for_expert_training"])

    def test_conflicting_reviews_are_not_training_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            value_pack = root / "value_pack.jsonl"
            review_pack = root / "review_pack.jsonl"
            value_pack.write_text(json.dumps(_value_row(route_id="r1")) + "\n", encoding="utf-8")
            review_pack.write_text(
                json.dumps(_review_row(route_id="r1", positive=True))
                + "\n"
                + json.dumps(_review_row(route_id="r1", negative=True))
                + "\n",
                encoding="utf-8",
            )

            report = merge_route_block_review_labels(
                value_pack_jsonl=value_pack,
                review_label_pack_jsonl=review_pack,
                output_jsonl=root / "merged.jsonl",
                report_json=root / "report.json",
                min_usable_positive=1,
                min_usable_negative=1,
            )
            row = json.loads((root / "merged.jsonl").read_text(encoding="utf-8"))

        self.assertTrue(row["weak_label_tasks"]["expert_review_conflict"])
        self.assertFalse(row["weak_label_tasks"]["expert_review_positive"])
        self.assertFalse(row["weak_label_tasks"]["expert_review_negative"])
        self.assertEqual(report["counts"]["conflict_rows"], 1)
        self.assertFalse(report["decision"]["ready_for_expert_training"])


def _value_row(*, route_id, split="train"):
    return {
        "schema_version": "route_block_value_pack.v1",
        "dataset": "unit",
        "split": split,
        "target_id": "t1",
        "target_smiles": "CCO",
        "selector_group_id": "g1",
        "route_id": route_id,
        "native_rank": 0,
        "weak_label_tasks": {
            "reviewable_by_audit": True,
            "reject_artifact": False,
        },
        "feature_groups": {
            "native": {"native_score": 1.0},
        },
    }


def _review_row(*, route_id, positive=False, negative=False, placeholder=False):
    return {
        "schema_version": "route_block_review_label_pack.v1",
        "dataset": "unit",
        "review_source": "human_csv",
        "review_id": f"review-{route_id}-{positive}-{negative}-{placeholder}",
        "target_id": "t1",
        "target_smiles": "CCO",
        "route_id": route_id,
        "evidence_class": "strict_model_control_disagreement",
        "review_labels": {
            "route_plausible_yes": positive,
            "block_transform_correct_yes": positive,
            "support_precedent_relevant_yes": positive,
            "cascade_coherent_yes": positive,
            "priority_high_or_medium": positive,
            "priority_reject": negative,
            "placeholder_review": placeholder,
            "usable_positive": positive,
            "usable_negative": negative,
        },
        "review_fields": {
            "priority": "medium" if positive else "reject" if negative else "low",
            "route_plausible": "yes" if positive else "no" if negative else "unclear",
            "block_transform_correct": "yes" if positive else "no" if negative else "unclear",
            "support_precedent_relevant": "yes" if positive else "no" if negative else "unclear",
            "cascade_coherent": "yes" if positive else "no" if negative else "unclear",
            "risk_tags": ["other"],
            "rationale": "unit",
        },
    }


if __name__ == "__main__":
    unittest.main()
