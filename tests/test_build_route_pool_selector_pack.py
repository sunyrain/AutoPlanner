import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_route_pool_selector_pack import build_route_pool_selector_pack


class BuildRoutePoolSelectorPackTest(unittest.TestCase):
    def test_builds_pack_with_product_audit_labels(self):
        target = "CC[C@@H](O)C[C@@H](O)CC=O"
        payload = {
            "target": target,
            "objective": "chem_enzy_native",
            "routes": [
                _route(target=target, reactants=["CC(=O)C(=O)O"], score=0.9),
                _route(target=target, reactants=[target], score=0.1),
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "plan.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            pack = root / "route_pool_pack.jsonl"
            report = root / "report.json"
            manifest = root / "split.json"

            result = build_route_pool_selector_pack(
                inputs=[source],
                output_jsonl=pack,
                report_json=report,
                split_manifest=manifest,
                dataset="unit",
                split_strategy="train",
            )

            rows = [json.loads(line) for line in pack.read_text(encoding="utf-8").splitlines()]
            classes = {row["product_audit_class"] for row in rows}
            self.assertEqual(result["counts"]["rows"], 2)
            self.assertEqual(len(rows), 2)
            self.assertIn("reject_artifact", classes)
            self.assertTrue(any(row["labels"]["is_reject_artifact"] for row in rows))
            self.assertTrue(any(row["labels"]["is_reviewable"] for row in rows))
            self.assertTrue(all(row["dataset"] == "unit" for row in rows))
            self.assertTrue(all("feature" in row for row in rows))
            self.assertIn("native_score", rows[0]["feature"])
            self.assertTrue(any(row["route_label"] == 0 for row in rows))
            self.assertTrue(any(row["route_label"] > 0 for row in rows))
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["split_counts"], {"train": 2})

    def test_preserves_rejected_artifact_type_and_existing_audit(self):
        target = "CCCCCCCCCCCC"
        route = _route(target=target, reactants=["C"], score=0.7)
        route["product_audit"] = {
            "route_class": "reject_artifact",
            "issues": ["large_unexplained_carbon_gain"],
            "tags": [],
            "route_plausibility": {"passed": False, "reasons": ["large_unexplained_carbon_gain"]},
        }
        payload = {
            "target": target,
            "objective": "chem_enzy_native_rejected_routes",
            "routes": [route],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "plan_rejected.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            build_route_pool_selector_pack(
                inputs=[source],
                output_jsonl=root / "pack.jsonl",
                report_json=root / "report.json",
                split_manifest=root / "split.json",
                dataset="unit",
                split_strategy="train",
            )

            row = json.loads((root / "pack.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["artifact_type"], "rejected")
            self.assertEqual(row["product_audit_class"], "reject_artifact")
            self.assertIn("large_unexplained_carbon_gain", row["product_audit_issues"])
            self.assertGreaterEqual(row["large_atom_gain_count"], 1)

    def test_deduplicates_raw_and_rejected_route_signatures(self):
        target = "CCCCCCCCCCCC"
        route = _route(target=target, reactants=["C"], score=0.7)
        rejected_route = json.loads(json.dumps(route))
        rejected_route["product_audit"] = {
            "route_class": "reject_artifact",
            "issues": ["large_unexplained_carbon_gain"],
            "tags": [],
            "route_plausibility": {"passed": False, "reasons": ["large_unexplained_carbon_gain"]},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "plan_raw.json"
            rejected = root / "plan_rejected.json"
            raw.write_text(json.dumps({"target": target, "routes": [route]}), encoding="utf-8")
            rejected.write_text(json.dumps({"target": target, "objective": "chem_enzy_native_rejected_routes", "routes": [rejected_route]}), encoding="utf-8")

            report = build_route_pool_selector_pack(
                inputs=[raw, rejected],
                output_jsonl=root / "pack.jsonl",
                report_json=root / "report.json",
                split_manifest=root / "split.json",
                dataset="unit",
                split_strategy="train",
            )

            rows = [json.loads(line) for line in (root / "pack.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["counts"]["raw_rows"], 2)
            self.assertEqual(report["counts"]["rows"], 1)
            self.assertEqual(report["counts"]["deduplicated_rows_removed"], 1)
            self.assertEqual(rows[0]["artifact_type"], "rejected")
            self.assertEqual(rows[0]["duplicate_artifact_types"], ["raw", "rejected"])

    def test_path_name_split_strategy_preserves_train_val_test_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {
                "train": root / "v4_train_pool.json",
                "val": root / "v4_val_pool.json",
                "test": root / "v4_test_pool.json",
            }
            for split, path in paths.items():
                target = f"CCCC{split}"
                path.write_text(json.dumps({"target": target, "routes": [_route(target=target, reactants=["CC"], score=0.5)]}), encoding="utf-8")

            build_route_pool_selector_pack(
                inputs=list(paths.values()),
                output_jsonl=root / "pack.jsonl",
                report_json=root / "report.json",
                split_manifest=root / "split.json",
                dataset="unit",
                split_strategy="path_name",
            )

            rows = [json.loads(line) for line in (root / "pack.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
            self.assertEqual(json.loads((root / "split.json").read_text(encoding="utf-8"))["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_extracts_ccts_runtime_route_features(self):
        target = "CCO"
        route = _route(target=target, reactants=["CC=O"], score=0.5)
        route.pop("score")
        route["native_score"] = 0.73
        route["stock_closed"] = True
        route["solved"] = True
        route["terminal_reactants"] = ["CC=O"]
        route["ccts_v3_runtime_best_route_evidence"] = 4.2
        route["ccts_v3_runtime_model_mean"] = 3.1
        route["ccts_v3_runtime_step_any_mean"] = 0.8
        route["steps"][0]["v4_step_evidence"] = {"matched": True, "similarity": 1.0}
        payload = {"target": target, "routes": [route]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "v4_train_ccts_runtime_pool.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            build_route_pool_selector_pack(
                inputs=[source],
                output_jsonl=root / "pack.jsonl",
                report_json=root / "report.json",
                split_manifest=root / "split.json",
                dataset="unit",
                split_strategy="path_name",
            )

            row = json.loads((root / "pack.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["native_score"], 0.73)
            self.assertTrue(row["strict_stock_solve"])
            self.assertEqual(row["feature"]["ccts_v3_runtime_best_route_evidence"], 4.2)
            self.assertEqual(row["feature"]["ccts_v3_runtime_model_mean"], 3.1)
            self.assertGreaterEqual(row["feature"]["v4_evidence_hits"], 1.0)

    def test_attaches_explicit_evidence_provenance(self):
        target = "CCO"
        route = _route(target=target, reactants=["CC=O"], score=0.5)
        route["ccts_v3_runtime_best_route_evidence"] = 0.9
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "v4_train_ccts_runtime_pool.json"
            source.write_text(json.dumps({"target": target, "routes": [route]}), encoding="utf-8")

            report = build_route_pool_selector_pack(
                inputs=[source],
                output_jsonl=root / "pack.jsonl",
                report_json=root / "report.json",
                split_manifest=root / "split.json",
                dataset="unit",
                split_strategy="path_name",
                evidence_provenance={
                    "evidence_source_split": "train",
                    "retrieval_corpus_manifest": "train_only_v4_manifest.json",
                    "train_only_retrieval": True,
                },
            )

            row = json.loads((root / "pack.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["evidence_provenance"]["evidence_source_split"], "train")
        self.assertTrue(row["evidence_provenance"]["train_only_retrieval"])
        self.assertEqual(report["evidence_provenance"]["retrieval_corpus_manifest"], "train_only_v4_manifest.json")


def _route(*, target: str, reactants: list[str], score: float) -> dict:
    stock = {smi: True for smi in reactants}
    return {
        "score": score,
        "n_steps": 1,
        "route_rank": 0,
        "steps": [
            {
                "index": 0,
                "product": target,
                "main_reactant": reactants[0] if reactants else "",
                "aux_reactants": reactants[1:],
                "reaction_smiles": f"{'.'.join(reactants)}>>{target}",
                "reaction_type": "unknown",
                "source": "ChemEnzyRetroPlanner",
                "scores": {"retro": score, "confidence": score},
                "stock_status": stock,
                "reaction_interpretation": {
                    "reaction_class": "unknown",
                    "atom_change": {"heavy_atom_delta": 0},
                },
            }
        ],
        "metrics": {
            "strict_stock_solve": True,
            "route_solved": True,
            "filled_route": True,
            "terminal_reactants": reactants,
            "terminal_stock_status": stock,
        },
    }


if __name__ == "__main__":
    unittest.main()
