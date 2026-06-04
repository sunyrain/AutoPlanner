import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer

from cascade_planner.cascade_verifier import load_learned_verifier, predict_learned_verifier, verify_cascade_route
from cascade_planner.cascade_search import (
    CascadeSearchState,
    ConditionEnvelope,
    LoadedLearnedVerifierValueModel,
    StepAnnotation,
)
from scripts.train_cascade_verifier_from_pack import train_from_pack
from cascade_planner.cascade_verifier import cascade_verifier_features
from scripts.build_cascade_verifier_preference_pack import build_preference_pack


class CascadeVerifierTest(unittest.TestCase):
    def test_clean_material_and_conditions_pass(self):
        report = verify_cascade_route(_clean_route(), target_smiles="CCCCO")

        self.assertTrue(report.feasible)
        self.assertEqual(report.reason_counts, {})

    def test_atom_balance_drop_reactant_fails(self):
        route = _clean_route()
        route["steps"][0]["main_reactant"] = "C"
        route["steps"][0]["aux_reactants"] = []
        route["steps"][0]["reaction_smiles"] = "C>>CCCCO"

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertFalse(report["feasible"])
        self.assertIn("atom_balance_violation", report["reason_counts"])

    def test_same_stage_condition_conflicts_fail(self):
        route = _clean_route()
        route["steps"][0]["T"] = 20
        route["steps"][1]["T"] = 90
        route["steps"][0]["pH"] = 2
        route["steps"][1]["pH"] = 11
        route["stage_partition"] = ["stage_1", "stage_1"]

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertIn("temperature_conflict", report["reason_counts"])
        self.assertIn("ph_conflict", report["reason_counts"])

    def test_condition_conflicts_from_nested_condition_payloads_fail(self):
        route = _clean_route()
        route["steps"][0].pop("T", None)
        route["steps"][0].pop("pH", None)
        route["steps"][0].pop("solvent", None)
        route["steps"][1].pop("T", None)
        route["steps"][1].pop("pH", None)
        route["steps"][1].pop("solvent", None)
        route["steps"][0]["step_conditions"] = {"temperature_c": 5, "ph": 2, "solvents": ["water"]}
        route["steps"][1]["raw_metadata"] = {
            "condition": {"Temperature": 75, "pH": 10, "Solvent": "toluene"},
        }
        route["stage_partition"] = ["stage_1", "stage_1"]

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertIn("temperature_conflict", report["reason_counts"])
        self.assertIn("ph_conflict", report["reason_counts"])
        self.assertIn("solvent_conflict", report["reason_counts"])

    def test_enzyme_toxicity_and_cofactor_gap_fail(self):
        route = _clean_route()
        route["steps"][0]["ec"] = "1.1.1.1"
        route["steps"][0]["source"] = "CascadePlanner enzyme module"
        route["steps"][0]["solvent"] = "dichloromethane"
        route["steps"][0]["catalyst"] = "LDA"
        route["steps"][0]["cofactor_requirements"] = {"NADPH": 1.0}

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertIn("enzyme_toxicity", report["reason_counts"])
        self.assertIn("cofactor_ledger_gap", report["reason_counts"])

    def test_enzyme_toxicity_from_raw_metadata_reagent_fails(self):
        route = _clean_route()
        route["steps"][0]["ec"] = "1.1.1.1"
        route["steps"][0].pop("catalyst", None)
        route["steps"][0]["raw_metadata"] = {"conditions": {"reagents": ["NaH"]}}

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertIn("enzyme_toxicity", report["reason_counts"])

    def test_route_order_shuffle_fails(self):
        route = _clean_route()
        route["steps"][0], route["steps"][1] = route["steps"][1], route["steps"][0]

        report = verify_cascade_route(route, target_smiles="CCCCO").to_dict()

        self.assertIn("route_order_mismatch", report["reason_counts"])


class CascadePerturbationScriptTest(unittest.TestCase):
    def test_build_and_evaluate_pack_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "routes.json"
            pack = root / "pack.json"
            eval_json = root / "eval.json"
            eval_md = root / "eval.md"
            source.write_text(
                json.dumps({"target": "CCCCO", "routes": [_clean_route()]}),
                encoding="utf-8",
            )
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_cascade_perturbation_pack.py",
                    "--input",
                    str(source),
                    "--output",
                    str(pack),
                    "--max-routes",
                    "1",
                    "--perturbations-per-route",
                    "6",
                ],
                cwd=repo,
                env=env,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_cascade_verifier_pack.py",
                    "--input",
                    str(pack),
                    "--output",
                    str(eval_json),
                    "--markdown",
                    str(eval_md),
                ],
                cwd=repo,
                env=env,
                check=True,
            )

            result = json.loads(eval_json.read_text(encoding="utf-8"))
            summary = result["summary"]
            self.assertEqual(summary["n_examples"], 7)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["expected_reason_coverage"], 1.0)
            self.assertTrue(eval_md.exists())

            pref_jsonl = root / "prefs.jsonl"
            pref_summary = root / "prefs.summary.json"
            pref = build_preference_pack(
                pack,
                output_jsonl=pref_jsonl,
                summary_output=pref_summary,
                max_negatives_per_positive=6,
            )
            self.assertEqual(pref["summary"]["n_pairs"], 6)
            self.assertTrue(pref_jsonl.exists())

    def test_preference_pack_skips_seed_positives_that_verifier_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.json"
            pref_jsonl = root / "prefs.jsonl"
            pref_summary = root / "summary.json"
            pack.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "example_id": "seed_bad",
                                "label": 1,
                                "source_path": "source.json",
                                "source_target_index": 0,
                                "source_route_index": 0,
                                "target_smiles": "CCCCO",
                                "seed_verifier_feasible": False,
                                "cascade": _clean_route(),
                            },
                            {
                                "example_id": "neg",
                                "label": 0,
                                "source_path": "source.json",
                                "source_target_index": 0,
                                "source_route_index": 0,
                                "target_smiles": "CCCCO",
                                "expected_failure_reasons": ["atom_balance_violation"],
                                "cascade": _clean_route(),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pref = build_preference_pack(
                pack,
                output_jsonl=pref_jsonl,
                summary_output=pref_summary,
            )

            self.assertEqual(pref["summary"]["n_pairs"], 0)
            self.assertEqual(pref["summary"]["skipped_groups_seed_verifier_fail"], 1)

    def test_learned_verifier_training_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.json"
            model = root / "model.joblib"
            report = root / "report.json"
            examples = []
            for group, split in enumerate(["train", "val", "test"]):
                route = _clean_route()
                route["metadata"] = {"split": split, "route_domain": "test"}
                examples.append(
                    {
                        "example_id": f"{split}_seed",
                        "label": 1,
                        "source_target_index": group,
                        "target_smiles": "CCCCO",
                        "expected_failure_reasons": [],
                        "cascade": route,
                    }
                )
                neg = _clean_route()
                neg["metadata"] = {"split": split, "route_domain": "test"}
                neg["steps"][0]["main_reactant"] = ""
                neg["steps"][0]["aux_reactants"] = []
                neg["steps"][0]["reactants"] = []
                neg["steps"][0]["reaction_smiles"] = ">>CCCCO"
                examples.append(
                    {
                        "example_id": f"{split}_neg",
                        "label": 0,
                        "source_target_index": group,
                        "target_smiles": "CCCCO",
                        "expected_failure_reasons": ["atom_balance_violation"],
                        "cascade": neg,
                    }
                )
            pack.write_text(json.dumps({"examples": examples}), encoding="utf-8")

            result = train_from_pack(pack, model_output=model, report_output=report)

            self.assertTrue(model.exists())
            self.assertTrue(report.exists())
            self.assertEqual(result["summary"]["n_examples"], 6)
            self.assertIn("feasibility", result["summary"])
            calibration = result["summary"]["feasibility"]["threshold_calibration"]
            self.assertIn("recommended_threshold", calibration)
            self.assertTrue(calibration["operating_points"])
            self.assertIn("target_val_precision", calibration)
            artifact = joblib.load(model)
            self.assertEqual(artifact["recommended_feasible_threshold"], calibration["recommended_threshold"])

            state = CascadeSearchState(target_smiles="CCCCO", open_leaves=[])
            state.append_step(
                StepAnnotation(
                    product_smiles="CCCCO",
                    reactant_smiles=["CCCC"],
                    rxn_smiles="CCCC>>CCCCO",
                    condition=ConditionEnvelope(
                        temperature_c_min=25,
                        temperature_c_max=30,
                        ph_min=7,
                        ph_max=8,
                        solvents=["water"],
                    ),
                )
            )
            pred = LoadedLearnedVerifierValueModel(model, learned_weight=0.5).predict(state)
            self.assertIn("learned_verifier", pred.metadata)
            self.assertGreaterEqual(pred.value, 0.0)
            self.assertLessEqual(pred.value, 1.0)

    def test_learned_verifier_condition_features_are_stage_aware(self):
        route = _clean_route()
        route["steps"][0]["T"] = 5
        route["steps"][1]["T"] = 95
        route["steps"][0]["pH"] = 2
        route["steps"][1]["pH"] = 11
        route["stage_partition"] = ["stage_1", "stage_2"]

        features = cascade_verifier_features({"target_smiles": "CCCCO", "cascade": route})

        self.assertEqual(features["route_temperature_span"], 90.0)
        self.assertEqual(features["route_ph_span"], 9.0)
        self.assertEqual(features["max_stage_temperature_span"], 0.0)
        self.assertEqual(features["max_stage_ph_span"], 0.0)
        self.assertEqual(features["temperature_span"], 0.0)
        self.assertEqual(features["ph_span"], 0.0)

    def test_learned_verifier_runtime_loader_predicts_annotation_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "learned.joblib"
            _constant_learned_verifier(model, target_smiles="CCCCO", probability=0.73, threshold=0.98)

            learned = load_learned_verifier(model)
            payload = predict_learned_verifier(learned, _clean_route(), target_smiles="CCCCO")

            self.assertEqual(payload["model_path"], str(model))
            self.assertEqual(payload["feasible_probability"], 0.73)
            self.assertEqual(payload["recommended_feasible_threshold"], 0.98)
            self.assertFalse(payload["conservative_feasible"])
            self.assertEqual(payload["reason_probabilities"], {})


def _clean_route() -> dict:
    return {
        "steps": [
            {
                "product": "CCCCO",
                "main_reactant": "CCCC",
                "aux_reactants": [],
                "reaction_smiles": "CCCC>>CCCCO",
                "T": 30,
                "pH": 7,
                "solvent": "water",
            },
            {
                "product": "CCCC",
                "main_reactant": "CC",
                "aux_reactants": ["CC"],
                "reaction_smiles": "CC.CC>>CCCC",
                "T": 32,
                "pH": 7.2,
                "solvent": "water",
            },
        ],
        "stage_partition": ["stage_1", "stage_1"],
        "metrics": {"route_solved": True},
    }


def _constant_learned_verifier(path: Path, *, target_smiles: str, probability: float, threshold: float) -> None:
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([cascade_verifier_features({"target_smiles": target_smiles, "cascade": {"steps": []}})])
    model = _FixedProbabilityModel(probability)
    model.fit(x, [1])
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "feasible_model": model,
            "reason_models": {},
            "reason_labels": [],
            "recommended_feasible_threshold": threshold,
        },
        path,
    )


class _FixedProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probability: float):
        self.probability = float(probability)

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return [[1.0 - self.probability, self.probability] for _ in range(x.shape[0])]


if __name__ == "__main__":
    unittest.main()
