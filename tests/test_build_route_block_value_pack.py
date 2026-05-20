import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_route_block_value_pack import build_route_block_value_pack


class BuildRouteBlockValuePackTest(unittest.TestCase):
    def test_builds_feature_groups_and_weak_labels_without_single_score(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(_selector_row()) + "\n", encoding="utf-8")

            report = build_route_block_value_pack(
                input_jsonl=source,
                output_jsonl=root / "value_pack.jsonl",
                report_json=root / "report.json",
                dataset="unit",
            )

            row = json.loads((root / "value_pack.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["schema_version"], "route_block_value_pack.v1")
        self.assertEqual(row["dataset"], "unit")
        self.assertIn("native", row["feature_groups"])
        self.assertIn("product_audit", row["feature_groups"])
        self.assertIn("cascade_retrieval", row["feature_groups"])
        self.assertIn("learned_ccts", row["feature_groups"])
        self.assertNotIn("route_value_score", row)
        self.assertTrue(row["weak_label_tasks"]["retrieval_evidence_any"])
        self.assertTrue(row["weak_label_tasks"]["pair_context_evidence_any"])
        self.assertFalse(row["weak_label_tasks"]["strong_route_evidence"])
        self.assertTrue(row["weak_label_tasks"]["stock_closed_reviewable"])
        self.assertTrue(row["weak_label_tasks"]["no_human_route_positive"])
        self.assertFalse(row["weak_label_tasks"]["no_human_route_negative"])
        self.assertTrue(row["weak_label_tasks"]["no_human_consensus_positive"])
        self.assertFalse(row["weak_label_tasks"]["no_human_consensus_negative"])
        self.assertTrue(report["weak_label_positive_counts"]["retrieval_evidence_any"] == 1)
        self.assertIn("retrieval-only evidence rank", " ".join(report["training_contract"]["required_controls"]))
        self.assertIn("no_human_consensus_positive", report["training_contract"]["no_human_tasks"])
        self.assertEqual(report["evidence_provenance_audit"]["status"], "unverifiable_without_source_provenance")
        self.assertEqual(report["evidence_provenance_audit"]["missing_retrieval_provenance_rows"], 1)

    def test_flags_artifact_and_generic_template_as_separate_tasks(self):
        row = _selector_row()
        row["product_audit_class"] = "reject_artifact"
        row["labels"]["is_reject_artifact"] = True
        row["labels"]["is_reviewable"] = False
        row["generic_template_fraction"] = 1.0
        row["feature"]["generic_template_fraction"] = 1.0
        row["large_atom_gain_count"] = 2
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            build_route_block_value_pack(
                input_jsonl=source,
                output_jsonl=root / "value_pack.jsonl",
                report_json=root / "report.json",
            )

            out = json.loads((root / "value_pack.jsonl").read_text(encoding="utf-8"))

        self.assertTrue(out["weak_label_tasks"]["reject_artifact"])
        self.assertTrue(out["weak_label_tasks"]["large_atom_gain"])
        self.assertTrue(out["weak_label_tasks"]["generic_template_heavy"])
        self.assertFalse(out["weak_label_tasks"]["reviewable_by_audit"])
        self.assertFalse(out["weak_label_tasks"]["no_human_route_positive"])
        self.assertTrue(out["weak_label_tasks"]["no_human_route_negative"])
        self.assertFalse(out["weak_label_tasks"]["no_human_consensus_positive"])
        self.assertTrue(out["weak_label_tasks"]["no_human_consensus_negative"])

    def test_require_evidence_provenance_rejects_retrieval_features_without_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(_selector_row()) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train-only source provenance is incomplete"):
                build_route_block_value_pack(
                    input_jsonl=source,
                    output_jsonl=root / "value_pack.jsonl",
                    report_json=root / "report.json",
                    require_evidence_provenance=True,
                )

    def test_require_evidence_provenance_rejects_source_without_train_only_marker(self):
        row = _selector_row()
        row["evidence_source_split"] = "train"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing_train_only=1"):
                build_route_block_value_pack(
                    input_jsonl=source,
                    output_jsonl=root / "value_pack.jsonl",
                    report_json=root / "report.json",
                    require_evidence_provenance=True,
                )

    def test_accepts_explicit_evidence_provenance_when_required(self):
        row = _selector_row()
        row["evidence_source_split"] = "train"
        row["retrieval_corpus_manifest"] = "train_only_v4_manifest.json"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = build_route_block_value_pack(
                input_jsonl=source,
                output_jsonl=root / "value_pack.jsonl",
                report_json=root / "report.json",
                require_evidence_provenance=True,
            )
            out = json.loads((root / "value_pack.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(report["evidence_provenance_audit"]["status"], "verified_or_no_retrieval_features")
        self.assertEqual(report["evidence_provenance_audit"]["missing_retrieval_provenance_rows"], 0)
        self.assertEqual(out["evidence_provenance"]["status"], "present")
        self.assertTrue(out["evidence_provenance"]["has_source_split_marker"])

    def test_runtime_retrieval_only_separates_unverified_v4_step_evidence(self):
        row = _selector_row()
        row["feature"].pop("ccts_v3_runtime_best_route_evidence", None)
        row["feature"].pop("ccts_v3_runtime_step_pair_max", None)
        row["feature"]["v4_evidence_hits"] = 2.0
        row["feature"]["cascade_block_hits"] = 1.0
        row["evidence_source_split"] = "train"
        row["train_only_retrieval"] = True
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selector.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = build_route_block_value_pack(
                input_jsonl=source,
                output_jsonl=root / "value_pack.jsonl",
                report_json=root / "report.json",
                runtime_retrieval_only=True,
                require_evidence_provenance=True,
            )
            out = json.loads((root / "value_pack.jsonl").read_text(encoding="utf-8"))

        self.assertFalse(out["weak_label_tasks"]["retrieval_evidence_any"])
        self.assertNotIn("v4_evidence_hits", out["feature_groups"]["cascade_retrieval"])
        self.assertEqual(out["feature_groups"]["route_step_v4_evidence"]["v4_evidence_hits"], 2.0)
        self.assertEqual(report["evidence_provenance_audit"]["retrieval_feature_rows"], 0)
        self.assertTrue(report["training_contract"]["runtime_retrieval_only"])


def _selector_row():
    return {
        "schema_version": "route_pool_selector_pack.v1",
        "split": "train",
        "target_id": "t1",
        "target_smiles": "CCO",
        "selector_group_id": "g1",
        "route_id": "r1",
        "artifact_path": "demo.json",
        "artifact_type": "filtered",
        "native_rank": 0,
        "native_score": 0.7,
        "n_steps": 2,
        "route_diversity_signature": "A>>B|B>>C",
        "terminal_reactants": ["A"],
        "terminal_stock_status": {"A": True},
        "strict_stock_solve": True,
        "product_audit_class": "needs_chemist_review",
        "product_audit_class_order": 3,
        "product_audit_risk_order": 10,
        "product_audit_issues": [],
        "product_audit_tags": [],
        "route_plausibility_passed": True,
        "large_atom_gain_count": 0,
        "generic_template_fraction": 0.2,
        "cascade_block_hits": 1,
        "labels": {
            "is_reject_artifact": False,
            "is_reviewable": True,
            "is_stock_closed_reviewable": True,
        },
        "feature": {
            "native_score": 0.7,
            "native_rank": 0.0,
            "native_inv_rank": 1.0,
            "n_steps": 2.0,
            "stock_closed": 1.0,
            "route_solved": 1.0,
            "audit_is_reject": 0.0,
            "audit_class_order": 3.0,
            "audit_risk_order": 10.0,
            "route_plausibility_passed": 1.0,
            "large_atom_gain_count": 0.0,
            "generic_template_fraction": 0.2,
            "v4_evidence_hits": 2.0,
            "cascade_block_hits": 1.0,
            "ccts_v3_runtime_best_route_evidence": 0.8,
            "ccts_v3_runtime_step_pair_max": 0.2,
            "ccts_v3_runtime_model_mean": 1.5,
            "condition_score_mean": 0.4,
            "enzyme_confidence_mean": 0.3,
            "source_model_count": 1.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
