import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_cascade_action_value import build_dataset, train_cascade_action_value


class CascadeActionValueTrainingTest(unittest.TestCase):
    def test_trains_action_value_model_from_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            rows = []
            for state_idx in range(6):
                split = "val" if state_idx >= 4 else "train"
                for cand_idx in range(3):
                    positive = cand_idx == 1
                    rows.append(
                        {
                            "target_smiles": f"CCO{state_idx}",
                            "split": split,
                            "route_domain": "all_chemical",
                            "state_id": f"s{state_idx}",
                            "parent_mol": "CCO",
                            "parent_depth": state_idx % 2,
                            "candidate_index": cand_idx,
                            "source_model": "graphfp_models.USPTO-full_remapped",
                            "reaction_domain": "chemical",
                            "reactants": ["CC", "O"] if positive else ["C", "CO"],
                            "base_score": 0.2 + cand_idx * 0.1,
                            "base_cost": 1.0 + cand_idx,
                            "cascade_adjustment": 0.0,
                            "total_cost": 1.0 + cand_idx,
                            "components": {"domain_preference": 0.0},
                            "context_features": {
                                "route_domain": "all_chemical",
                                "node_depth": state_idx % 2,
                                "adjacent_reaction_domain": "enzymatic" if state_idx % 2 else "unknown",
                                "preferred_reaction_domains": ["chemical"],
                                "active_failure_modes": ["EnzymeEvidenceWeak"] if state_idx % 2 else [],
                            },
                            "source_policy_decision": {
                                "topk_by_model": {
                                    "graphfp_models.USPTO-full_remapped": 50,
                                    "onmt_models.bionav_one_step": 10,
                                },
                                "source_value_scores": {
                                    "graphfp_models.USPTO-full_remapped": 0.8,
                                    "onmt_models.bionav_one_step": 0.2,
                                },
                            },
                            "labels": {
                                "exact_gt_reaction": int(positive),
                                "gt_reactant_hit": int(positive),
                                "action_value": 1.0 if positive else 0.0,
                                "route_outcome_action_value": 0.8 if positive else 0.0,
                            },
                        }
                    )
            (pack / "action_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            model_output = root / "action_value.pt"
            report_output = root / "report.json"

            report = train_cascade_action_value(
                pack_dir=pack,
                model_output=model_output,
                report_output=report_output,
                epochs=1,
                batch_size=4,
                hidden=16,
                n_bits=16,
                device="cpu",
                label_name="route_outcome_action_value",
            )

            self.assertTrue(model_output.exists())
            self.assertTrue(report_output.exists())
            self.assertEqual(report["metadata"]["label_contract"], "internal_search_action_value_not_record_gold.v1")
            self.assertEqual(report["metadata"]["feature_schema"]["schema_version"], "cascade_action_value_features.v2")
            self.assertIn("adjacent_reaction_domain", report["metadata"]["feature_schema"]["categorical_fields"])
            self.assertIn("source_policy_score_ratio", report["metadata"]["feature_schema"]["numeric_fields"])
            self.assertIn("failure_enzyme_evidence_weak", report["metadata"]["feature_schema"]["numeric_fields"])
            self.assertIn("best_checkpoint", report)
            self.assertEqual(report["metadata"]["label_name"], "route_outcome_action_value")
            self.assertIn("top1_positive_state_hit_rate", report["final_metrics"])
            self.assertIn("total_cost", report["cost_baselines"])

    def test_process_context_features_are_nonzero_when_context_available(self):
        rows = [
            {
                "target_smiles": "CCO",
                "split": "train",
                "route_domain": "chemoenzymatic",
                "state_id": "s0",
                "parent_mol": "CCO",
                "parent_depth": 2,
                "candidate_index": 0,
                "source_model": "onmt_models.bionav_one_step",
                "reaction_domain": "enzymatic",
                "reactants": ["CC", "O"],
                "base_score": 0.5,
                "base_cost": 1.0,
                "cascade_adjustment": 0.0,
                "total_cost": 1.0,
                "components": {},
                "context_features": {
                    "route_domain": "chemoenzymatic",
                    "node_depth": 2,
                    "adjacent_reaction_domain": "chemical",
                    "preferred_reaction_domains": ["enzymatic"],
                    "active_failure_modes": ["EnzymeEvidenceWeak"],
                },
                "source_policy_decision": {
                    "topk_by_model": {
                        "onmt_models.bionav_one_step": 40,
                        "graphfp_models.USPTO-full_remapped": 10,
                    },
                    "source_value_scores": {
                        "onmt_models.bionav_one_step": 0.9,
                        "graphfp_models.USPTO-full_remapped": 0.1,
                    },
                },
                "labels": {"action_value": 1.0},
            }
        ]
        dataset = build_dataset(rows, n_bits=16)
        numeric_names = dataset.schema["numeric_fields"]
        numeric_start = len(dataset.schema["categories"]["route_domain"])
        numeric_start += len(dataset.schema["categories"]["source_model"])
        numeric_start += len(dataset.schema["categories"]["reaction_domain"])
        numeric_start += len(dataset.schema["categories"]["adjacent_reaction_domain"])
        numeric_start += 2 * int(dataset.schema["n_bits"])
        values = dict(zip(numeric_names, dataset.x[0][numeric_start:]))

        self.assertEqual(dataset.schema["schema_version"], "cascade_action_value_features.v2")
        self.assertGreater(values["preferred_domain_match"], 0.0)
        self.assertGreater(values["enzymatic_after_chemical_context"], 0.0)
        self.assertGreater(values["failure_enzyme_evidence_weak"], 0.0)
        self.assertGreater(values["source_policy_score_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
