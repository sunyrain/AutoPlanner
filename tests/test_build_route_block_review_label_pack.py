import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_route_block_review_label_pack import build_route_block_review_label_pack


class BuildRouteBlockReviewLabelPackTest(unittest.TestCase):
    def test_builds_calibration_pack_and_marks_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "self_review_labels.jsonl"
            source.write_text(json.dumps(_review_row(priority="medium")) + "\n", encoding="utf-8")

            report = build_route_block_review_label_pack(
                inputs=[source],
                output_jsonl=root / "review_pack.jsonl",
                report_json=root / "report.json",
                dataset="unit",
            )

            row = json.loads((root / "review_pack.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["schema_version"], "route_block_review_label_pack.v1")
        self.assertEqual(row["review_source"], "self_review")
        self.assertEqual(row["target_id"], "t1")
        self.assertEqual(row["route_id"], "route1")
        self.assertEqual(row["value_split"], "train")
        self.assertTrue(row["review_labels"]["usable_positive"])
        self.assertTrue(row["training_contract"]["not_route_preference_label"])
        self.assertFalse(report["decision"]["sufficient_for_main_training"])
        self.assertIn("self-review dominated", report["decision"]["reason"])
        self.assertEqual(report["counts"]["usable_positive_rows"], 1)
        self.assertEqual(report["value_split_counts"], {"train": 1})

    def test_reject_rows_are_usable_negatives(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "human_labels.jsonl"
            source.write_text(json.dumps(_review_row(priority="reject", route_plausible="no")) + "\n", encoding="utf-8")

            report = build_route_block_review_label_pack(
                inputs=[source],
                output_jsonl=root / "review_pack.jsonl",
                report_json=root / "report.json",
            )

            row = json.loads((root / "review_pack.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["review_source"], "human_csv")
        self.assertTrue(row["review_labels"]["usable_negative"])
        self.assertEqual(report["risk_tag_counts"]["not_cascade"], 1)
        self.assertFalse(report["decision"]["ready_for_route_block_merge_evaluation"])
        self.assertIn("both usable positives", report["decision"]["reason"])

    def test_placeholder_reviews_are_not_usable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "dryrun.jsonl"
            row = _review_row(priority="low")
            row["expert_review"] = {
                "priority": "low",
                "route_plausible": "unclear",
                "block_transform_correct": "unclear",
                "support_precedent_relevant": "unclear",
                "cascade_coherent": "unclear",
                "risk_tags": ["other"],
                "rationale": "Dry-run placeholder response; no chemistry judgment was performed.",
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = build_route_block_review_label_pack(
                inputs=[source],
                output_jsonl=root / "review_pack.jsonl",
                report_json=root / "report.json",
            )

            out = json.loads((root / "review_pack.jsonl").read_text(encoding="utf-8"))

        self.assertTrue(out["review_labels"]["placeholder_review"])
        self.assertFalse(out["review_labels"]["usable_positive"])
        self.assertFalse(out["review_labels"]["usable_negative"])
        self.assertEqual(report["decision"]["sufficient_for_main_training"], False)


def _review_row(*, priority, route_plausible="yes"):
    return {
        "review_id": "r1",
        "target_id": "t1",
        "target_smiles": "CCO",
        "route_id": "route1",
        "value_split": "train",
        "native_rank": 2,
        "stock_closed": True,
        "source_pool": "20",
        "evidence_class": "any_analog_supported",
        "diagnostic_labels": {
            "has_any_analog_block": True,
            "has_same_pair_analog_block": False,
            "has_observed_pair_block": True,
        },
        "diagnostic_scores": {
            "best_any_block_min_sim": 0.7,
            "best_same_pair_block_min_sim": 0.2,
        },
        "route_block": {
            "route_block_index": 0,
            "transform_pair": "reduction->acylation",
            "upstream_transform": "reduction",
            "downstream_transform": "acylation",
            "upstream_rxn": "A>>B",
            "downstream_rxn": "B>>C",
            "any_analog_supported": True,
            "same_pair_analog_supported": False,
        },
        "transform_sanity": {
            "block_has_label_mismatch": False,
            "block_label_mismatch_count": 0,
        },
        "expert_review": {
            "priority": priority,
            "route_plausible": route_plausible,
            "block_transform_correct": "yes",
            "support_precedent_relevant": "yes",
            "cascade_coherent": "yes",
            "risk_tags": ["not_cascade"] if priority == "reject" else [],
            "rationale": "unit",
        },
    }


if __name__ == "__main__":
    unittest.main()
