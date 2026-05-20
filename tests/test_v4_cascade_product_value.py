import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_v4_cascade_preference_pack import build_v4_cascade_preference_pack
from cascade_planner.eval.build_routepool_preference_pack import _preference_decision
from cascade_planner.eval.build_v4_cascade_product_value_pack import build_v4_cascade_product_value_pack
from cascade_planner.eval.rerank_native_routes_with_v4_value import _ranking_key, rerank_native_routes_with_v4_value
from cascade_planner.eval.train_v4_cascade_product_value import train_v4_cascade_product_value
from cascade_planner.cascade_search.v4_product_value import build_route_feature_schema, route_feature_vector, route_numeric_features


class V4CascadeProductValueTest(unittest.TestCase):
    def test_build_train_and_rerank_native_pool(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v4 = root / "v4.jsonl"
            split_dir = root / "splits"
            split_dir.mkdir()
            rows = [
                _v4_row("10.a", "cascade_1", "gold", "oxidation", "CCO>>CC=O", "train"),
                _v4_row("10.b", "cascade_1", "silver", "oxidation", "CCO>>CC=O", "train", sparse=True),
                _v4_row("10.c", "cascade_1", "gold", "reduction", "CC=O>>CCO", "train"),
                _v4_row("10.d", "cascade_1", "silver", "reduction", "CC=O>>CCO", "train", sparse=True),
                _v4_row("10.e", "cascade_1", "gold", "amination", "CC=O.N>>CCN", "val"),
                _v4_row("10.f", "cascade_1", "silver", "amination", "CC=O.N>>CCN", "val", sparse=True),
            ]
            v4.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            for split in ["train", "val", "test"]:
                payload = [
                    {
                        "doi": row["doi"],
                        "cascade_id": row["cascade_id"],
                        "target_smiles": row["target_product_smiles"],
                        "split": split,
                        "split_group_id": f"group-{row['doi']}",
                    }
                    for row in rows
                    if row["split"] == split
                ]
                (split_dir / f"v4_trace_{split}.json").write_text(json.dumps(payload), encoding="utf-8")

            pack_dir = root / "pack"
            pack_report = build_v4_cascade_product_value_pack(
                v4_jsonl=v4,
                split_dir=split_dir,
                output_dir=pack_dir,
            )
            pref_report = build_v4_cascade_preference_pack(
                feature_pack=pack_dir / "cascade_v4_route_feature_pack_train.jsonl",
                output=pack_dir / "cascade_v4_preference_train.jsonl",
            )
            model_path = root / "v4_value.pt"
            train_report = train_v4_cascade_product_value(
                train_pack=pack_dir / "cascade_v4_route_feature_pack_train.jsonl",
                val_pack=pack_dir / "cascade_v4_route_feature_pack_val.jsonl",
                preference_pack=pack_dir / "cascade_v4_preference_train.jsonl",
                output=model_path,
                report=root / "train_report.json",
                epochs=1,
                batch_size=2,
                hidden=32,
                n_bits=16,
                device="cpu",
            )
            native_pool = root / "native.json"
            native_pool.write_text(json.dumps(_native_pool()), encoding="utf-8")
            rerank_report = rerank_native_routes_with_v4_value(
                native_pool=native_pool,
                model=model_path,
                output=root / "reranked.json",
                report=root / "rerank_report.json",
                device="cpu",
            )
            reranked = json.loads((root / "reranked.json").read_text(encoding="utf-8"))
            model_exists = model_path.exists()

        self.assertEqual(pack_report["counts"]["rows"], 6)
        self.assertGreater(pref_report["counts"]["preferences"], 0)
        self.assertTrue(model_exists)
        self.assertEqual(train_report["metadata"]["n_train"], 4)
        self.assertEqual(rerank_report["summary"]["targets"], 1)
        self.assertIn("v4_cascade_product_prediction", reranked["targets"][0]["routes"][0])

    def test_audit_guarded_ranking_uses_product_audit_class(self):
        artifact_high_value = {
            "native_rank": 0,
            "v4_cascade_product_value": 1.0,
            "v4_product_audit_features": {"route_class": "reject_artifact"},
        }
        triage_low_value = {
            "native_rank": 1,
            "v4_cascade_product_value": 0.1,
            "v4_product_audit_features": {"route_class": "triage_late_stage"},
        }

        ranked = sorted(
            [artifact_high_value, triage_low_value],
            key=lambda row: _ranking_key(row, ranking_mode="audit_guarded"),
        )

        self.assertIs(ranked[0], triage_low_value)

    def test_audit_guarded_requires_audit_features(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native_pool = root / "native.json"
            native_pool.write_text(json.dumps(_native_pool()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "audit_guarded ranking requires"):
                rerank_native_routes_with_v4_value(
                    native_pool=native_pool,
                    model=root / "missing_model.pt",
                    output=root / "reranked.json",
                    report=root / "report.json",
                    ranking_mode="audit_guarded",
                    include_audit_features=False,
                )

    def test_plausibility_features_are_exposed_to_route_model(self):
        row = {
            "target_smiles": "CCCCCCC",
            "starting_material_smiles": "",
            "terminal_reactants": ["CCC"],
            "steps": [],
            "product_audit": {
                "route_class": "reject_artifact",
                "issues": ["large_unexplained_heavy_atom_gain"],
                "tags": [],
                "route_plausibility": {
                    "passed": False,
                    "reasons": ["large_unexplained_carbon_gain"],
                    "steps": [
                        {
                            "heavy_atom_gain": 4,
                            "carbon_gain": 4,
                            "hetero_atom_gain": 0,
                        }
                    ],
                },
            },
        }

        numeric = route_numeric_features(row)
        schema = build_route_feature_schema([row], n_bits=16)
        vector = route_feature_vector(row, schema)

        self.assertEqual(numeric["audit_issue_large_heavy_atom_gain"], 1.0)
        self.assertEqual(numeric["audit_issue_large_carbon_gain"], 1.0)
        self.assertEqual(numeric["audit_plausibility_passed"], 0.0)
        self.assertGreater(numeric["audit_max_carbon_gain_scaled"], 0.0)
        self.assertEqual(len(vector), schema["feature_dim"])

    def test_routepool_preference_penalizes_new_material_artifacts(self):
        better = {
            "route_class": "triage_late_stage",
            "issues": [],
            "tags": [],
            "route_plausibility": {"passed": True, "reasons": []},
        }
        worse = {
            "route_class": "triage_late_stage",
            "issues": ["large_unexplained_carbon_gain"],
            "tags": [],
            "route_plausibility": {"passed": False, "reasons": ["large_unexplained_carbon_gain"]},
        }

        decision = _preference_decision(better, worse)

        self.assertIsNotNone(decision)
        self.assertIn(decision["reason"], {"artifact_issue_dominance", "same_class_material_plausibility_preference"})


def _v4_row(doi, cascade_id, tier, superclass, rxn, split, sparse=False):
    return {
        "doi": doi,
        "cascade_id": cascade_id,
        "split": split,
        "trainable_recommended": True,
        "quality_tier": tier,
        "is_high_quality": True,
        "is_demonstrated_success": not sparse,
        "target_product_smiles": rxn.split(">>", 1)[1],
        "starting_material_smiles": rxn.split(">>", 1)[0],
        "cascade_type": "chemoenzymatic",
        "has_outcome": not sparse,
        "has_conditions": not sparse,
        "n_substrate_scope_entries": 2 if not sparse else 0,
        "n_catalyst_components": 1,
        "n_input_species": 1,
        "n_output_species": 1,
        "overall_yield": 80 if not sparse else None,
        "overall_ee": 95 if not sparse else None,
        "total_reaction_time": 12 if not sparse else None,
        "steps": [
            {
                "step_index": 1,
                "rxn_smiles": rxn,
                "transformation_superclass": superclass,
                "step_conditions": {"temperature_c": 30, "ph": 7, "solvent": "water"} if not sparse else {},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "1.1.1.1"}],
                "step_outcome": {"step_yield_percent": 80 if not sparse else None},
            }
        ],
        "substrate_scope": [{"substrate_smiles": "CCO", "product_smiles": "CC=O"}] if not sparse else [],
    }


def _native_pool():
    return {
        "metadata": {"dataset": "tiny_native"},
        "summary": {},
        "targets": [
            {
                "target_smiles": "CC=O",
                "routes": [
                    {
                        "target_smiles": "CC=O",
                        "steps": [
                            {
                                "product_smiles": "CC=O",
                                "reactant_smiles": ["CCO"],
                                "rxn_smiles": "CCO>>CC=O",
                                "source_model": "ChemEnzyRetroPlanner",
                                "score": 0.9,
                                "stock_status": {"CCO": True},
                            }
                        ],
                        "score": 0.9,
                        "solved": True,
                        "route_rank": 0,
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
