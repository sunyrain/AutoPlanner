import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from cascade_planner.eval.build_strict_model_review_worklist import build_strict_model_review_worklist


class BuildStrictModelReviewWorklistTest(unittest.TestCase):
    def test_builds_review_rows_from_model_control_disagreement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "routes.json"
            artifact.write_text(json.dumps(_artifact()), encoding="utf-8")
            pack = root / "value_pack.jsonl"
            rows = [
                _value_row("r0", artifact, native_rank=0, signal=0.1, retrieval=0.9, audit_risk=0),
                _value_row("r1", artifact, native_rank=1, signal=5.0, retrieval=0.1, audit_risk=20),
            ]
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            model_pkl = root / "model.pkl"
            _write_model(model_pkl)

            report = build_strict_model_review_worklist(
                value_pack=pack,
                model_pickle=model_pkl,
                output_jsonl=root / "worklist.jsonl",
                output_csv=root / "worklist.csv",
                report_json=root / "report.json",
                max_rows=1,
                min_disagreement=0,
            )

            out = json.loads((root / "worklist.jsonl").read_text(encoding="utf-8"))
            with (root / "worklist.csv").open(newline="", encoding="utf-8") as fh:
                csv_row = next(csv.DictReader(fh))

        self.assertEqual(report["summary"]["selected_rows"], 1)
        self.assertEqual(out["schema_version"], "strict_model_control_disagreement_review.v1")
        self.assertEqual(out["evidence_class"], "strict_model_control_disagreement")
        self.assertEqual(out["target_id"], "t1")
        self.assertEqual(out["value_split"], "test")
        self.assertIn("upstream_rxn", out["review_block"])
        self.assertIn("downstream_rxn", out["review_block"])
        self.assertIn("model_rank", out["diagnostic_scores"])
        self.assertEqual(csv_row["target_id"], "t1")
        self.assertEqual(csv_row["value_split"], "test")
        self.assertEqual(csv_row["source_value_pack"], str(pack))


def _write_model(path: Path) -> None:
    model = LogisticRegression(solver="liblinear", random_state=0)
    model.fit(np.asarray([[0.0], [1.0], [5.0], [6.0]], dtype=np.float32), np.asarray([0, 0, 1, 1]))
    with path.open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "mean": np.asarray([0.0], dtype=np.float32),
                "std": np.asarray([1.0], dtype=np.float32),
                "feature_names": ["route_context.signal"],
            },
            fh,
        )


def _artifact():
    steps = [
        {
            "rxn_smiles": "A>>B",
            "reactants": ["A"],
            "products": ["B"],
            "transformation_superclass": "oxidation",
            "v4_step_evidence": {"similarity": 0.5, "doi": "10.demo/up"},
        },
        {
            "rxn_smiles": "B>>C",
            "reactants": ["B"],
            "products": ["C"],
            "transformation_superclass": "amination",
            "v4_step_evidence": {"similarity": 0.6, "doi": "10.demo/down"},
        },
    ]
    return {
        "targets": [
            {
                "target_id": "t1",
                "target_smiles": "C",
                "routes": [
                    {"steps": steps},
                    {"steps": steps},
                ],
            }
        ]
    }


def _value_row(route_id, artifact, *, native_rank, signal, retrieval, audit_risk):
    return {
        "schema_version": "route_block_value_pack.v1",
        "split": "test",
        "selector_group_id": "g1",
        "target_id": "t1",
        "target_smiles": "C",
        "route_id": route_id,
        "artifact_path": str(artifact),
        "artifact_route_index": native_rank,
        "native_rank": native_rank,
        "native_score": 0.5,
        "n_steps": 2,
        "product_audit": {"risk_order": audit_risk},
        "weak_label_tasks": {"stock_closed": True, "reviewable_by_audit": audit_risk == 0, "reject_artifact": audit_risk > 0},
        "feature_groups": {
            "route_context": {"signal": signal},
            "cascade_retrieval": {"ccts_v3_runtime_best_route_evidence": retrieval},
        },
    }


if __name__ == "__main__":
    unittest.main()
