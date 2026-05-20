import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cascade_planner.eval.build_v4_transform_pair_selector_pack import build_v4_transform_pair_selector_pack
from cascade_planner.eval.build_template_outcome_supervision_pack import build_template_outcome_supervision_pack
from cascade_planner.eval.audit_route_pool_cascade_evidence import audit_route_pool_cascade_evidence
from cascade_planner.eval.audit_route_pool_review_transform_sanity import audit_route_pool_review_transform_sanity
from cascade_planner.eval.sample_template_outcome_review_batch import sample_template_outcome_review_batch
from cascade_planner.eval.sample_route_pool_evidence_review_batch import sample_route_pool_evidence_review_batch
from cascade_planner.eval.build_route_pool_evidence_review_prompts import build_route_pool_evidence_review_prompts
from cascade_planner.eval.ingest_route_pool_evidence_review_results import ingest_route_pool_evidence_review_results
from cascade_planner.eval.run_route_pool_evidence_llm_review import run_route_pool_evidence_llm_review
from cascade_planner.eval.summarize_route_pool_evidence_review_labels import summarize_route_pool_evidence_review_labels
from cascade_planner.eval.gate_route_pool_evidence_review_promotion import gate_route_pool_evidence_review_promotion
from cascade_planner.eval.run_route_pool_evidence_review_pipeline import run_route_pool_evidence_review_pipeline
from cascade_planner.eval.export_route_pool_review_worklist import export_route_pool_review_worklist
from cascade_planner.eval.calibrate_route_pool_evidence_review_signals import calibrate_route_pool_evidence_review_signals
from cascade_planner.eval.ingest_route_pool_evidence_review_csv import ingest_route_pool_evidence_review_csv
from cascade_planner.eval.select_route_pool_review_calibration_subset import select_route_pool_review_calibration_subset
from cascade_planner.eval.select_route_pool_review_prompt_subset import select_route_pool_review_prompt_subset
from cascade_planner.eval.run_route_pool_evidence_review_csv_pipeline import run_route_pool_evidence_review_csv_pipeline
from cascade_planner.eval.build_route_pool_review_calibration_packet import build_route_pool_review_calibration_packet
from cascade_planner.eval.summarize_cascadebench_phase2 import _decision_gates
from cascade_planner.eval.train_v4_template_selector import _feature_names, _feature_value
from cascade_planner.eval.verify_cascadebench_phase2_closure import verify_cascadebench_phase2_closure


class CascadeBenchV4SelectorsTest(unittest.TestCase):
    def test_transform_pair_pack_uses_train_vocabulary_for_test_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            manifest = root / "manifest.json"
            cache = root / "cache.json"
            out = root / "pack.jsonl"
            report = root / "report.json"
            train.write_text(json.dumps(_program("CCO", "oxidation", "reduction")) + "\n", encoding="utf-8")
            test.write_text(json.dumps(_program("CCN", "amination", "acylation")) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"outputs": {"train": str(train), "test": str(test), "val": str(test)}}),
                encoding="utf-8",
            )
            cache.write_text(
                json.dumps(
                    {
                        json.dumps({"product": "CCN"}): [
                            {
                                "reaction_smiles": "CC.N>>CCN",
                                "main_reactant": "CC",
                                "score": 0.9,
                                "rank": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pack_report = build_v4_transform_pair_selector_pack(
                program_manifest=manifest,
                chem_enzy_cache=cache,
                output_jsonl=out,
                report_json=report,
                split="test",
                max_targets=1,
                min_connector_heavy_atoms=1,
                connected_ref_similarity=0.0,
                connector_label_similarity=0.0,
                top_pairs=5,
                candidate_pair_split="train",
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(pack_report["metadata"]["candidate_pair_split"], "train")
        self.assertEqual(pack_report["candidate_pairs"], ["oxidation->reduction"])
        self.assertEqual({row["transform_pair"] for row in rows}, {"oxidation->reduction"})
        self.assertEqual(pack_report["top_label_split_pairs"], {"amination->acylation": 1})

    def test_template_selector_reads_app_and_reaction_center_features(self):
        payload = {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "proposals": [
                        {
                            "proposal_score": 0.4,
                            "downstream_rank": 1,
                            "template_rank": 2,
                            "outcome_rank": 1,
                            "template_count": 3,
                            "connector_heavy_atoms": 5,
                            "connector_is_main_reactant": True,
                            "template_transform_pair": "oxidation->reduction",
                            "app_connector_main_similarity": 0.7,
                            "app_template_example_best_transition_sim": 0.8,
                            "rc_new_atom_fraction": 0.2,
                            "rc_inherited_atom_fraction": 0.8,
                            "rc_template_matched_fraction": 0.25,
                        }
                    ],
                }
            ]
        }
        names = _feature_names(payload)
        row = payload["targets"][0]["proposals"][0]

        self.assertIn("app_connector_main_similarity", names)
        self.assertIn("app_template_example_best_transition_sim", names)
        self.assertIn("rc_new_atom_fraction", names)
        self.assertIn("rc_inherited_atom_fraction", names)
        self.assertIn("rc_template_matched_fraction", names)
        self.assertAlmostEqual(_feature_value(row, "app_connector_main_similarity"), 0.7)
        self.assertAlmostEqual(_feature_value(row, "app_template_example_best_transition_sim"), 0.8)
        self.assertAlmostEqual(_feature_value(row, "rc_new_atom_fraction"), 0.2)
        self.assertAlmostEqual(_feature_value(row, "rc_inherited_atom_fraction"), 0.8)
        self.assertAlmostEqual(_feature_value(row, "rc_template_matched_fraction"), 0.25)

    def test_phase2_decision_gates_ignore_tiny_metric_noise(self):
        gates = _decision_gates(
            reports={
                "transform_pair_train_vocab": {"delta": {"mrr_all_groups": 0.001, "hit_at_5": 0.009}},
                "template_twostage_freq_top3_analog": {"delta": {"mrr": 0.001}},
                "template_twostage_freq_top3_pair": {"delta": {"mrr": 0.001}},
                "template_twostage_freq_top3_rc_pair": {"delta": {"mrr": 0.001}},
            },
            template_generation={
                "frequency_top3": {"pair_and_analog_any": 3},
                "selector_top3": {"pair_and_analog_any": 3},
                "oracle_transform_upper_bound": {"pair_and_analog_any": 8},
            },
            rc_probe={
                "pair_and_analog": {
                    "proposal_score": {"mrr": 0.0410},
                    "rc_inherited_atom_fraction": {"mrr": 0.0415},
                }
            },
        )

        self.assertFalse(gates["transform_selector_top1_or_mrr_improves"])
        self.assertFalse(gates["transform_selector_top5_improves"])
        self.assertFalse(gates["template_twostage_pair_selector_improves_mrr"])
        self.assertFalse(gates["rc_pair_selector_improves_mrr"])
        self.assertFalse(gates["rc_feature_pair_probe_improves_mrr"])
        self.assertTrue(gates["oracle_pair_upper_bound_exists"])

    def test_template_outcome_supervision_pack_samples_review_classes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proposal = root / "template_bridge.json"
            out = root / "pack.jsonl"
            report_path = root / "report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "metadata": {"split": "test"},
                        "targets": [
                            {
                                "target_smiles": "CCO",
                                "proposals": [
                                    _proposal(pair=True, analog=True, score=0.9),
                                    _proposal(pair=False, analog=True, score=0.8),
                                    _proposal(pair=True, analog=False, score=0.7),
                                    _proposal(pair=False, analog=False, score=0.6),
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_template_outcome_supervision_pack(
                proposal_jsons=[proposal],
                output_jsonl=out,
                report_json=report_path,
                max_per_target_class=2,
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["rows"], 4)
        self.assertEqual(
            {row["supervision_class"] for row in rows},
            {
                "pair_and_analog_positive",
                "analog_only_positive",
                "pair_only_near_miss",
                "high_score_hard_negative",
            },
        )
        self.assertTrue(all("features" in row for row in rows))

    def test_phase2_closure_verifier_checks_core_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "template_outcome_supervision_pack").mkdir(parents=True)
            (root / "template_outcome_review_batch").mkdir(parents=True)
            for rel in [
                "phase2_decision_summary.md",
                "CCTS_V3_CANDIDATE_EVIDENCE_REPORT.zh.md",
                "PHASE2_REPRODUCIBILITY_MANIFEST.md",
                "route_pool_cascade_evidence_summary.md",
                "route_pool_cascade_evidence_20/route_pool_cascade_evidence.json",
                "route_pool_cascade_evidence_20/route_pool_cascade_evidence.md",
                "route_pool_cascade_evidence_statin/route_pool_cascade_evidence.json",
                "route_pool_cascade_evidence_statin/route_pool_cascade_evidence.md",
                "route_pool_cascade_evidence_full100/route_pool_cascade_evidence.json",
                "route_pool_cascade_evidence_full100/route_pool_cascade_evidence.md",
                "route_pool_evidence_review_batch/route_pool_evidence_review_batch.jsonl",
                "route_pool_evidence_review_batch/route_pool_evidence_review_batch.csv",
                "route_pool_evidence_review_batch/route_pool_evidence_review_batch_report.md",
                "route_pool_evidence_review_batch/route_pool_transform_sanity_audit.md",
                "route_pool_evidence_review_batch/route_pool_evidence_review_worklist.jsonl",
                "route_pool_evidence_review_batch/route_pool_evidence_review_worklist.csv",
                "route_pool_evidence_review_batch/route_pool_evidence_review_worklist_report.md",
                "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset.jsonl",
                "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset.csv",
                "route_pool_evidence_review_batch/route_pool_evidence_review_calibration_subset_report.md",
                "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels.jsonl",
                "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_report.md",
                "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_invalid.jsonl",
                "route_pool_evidence_review_batch/human_route_pool_evidence_review_labels_unreviewed.jsonl",
                "route_pool_evidence_review_prompts/route_pool_evidence_review_prompts.jsonl",
                "route_pool_evidence_review_prompts/route_pool_evidence_review_prompts_report.md",
                "route_pool_evidence_review_prompts/route_pool_evidence_review_calibration_prompts.jsonl",
                "route_pool_evidence_review_prompts/route_pool_evidence_review_calibration_prompts_report.md",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_responses.jsonl",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_run_report.md",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_labels.jsonl",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_labels_report.md",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_label_summary.md",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_signal_calibration.md",
                "route_pool_evidence_review_prompts/dryrun_route_pool_evidence_review_promotion_gate.md",
                "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_pipeline_manifest.md",
                "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_responses.jsonl",
                "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_labels.jsonl",
                "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_signal_calibration.json",
                "route_pool_evidence_review_pipeline_dryrun/dryrun_route_pool_evidence_review_promotion_gate.json",
                "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_pipeline_manifest.md",
                "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_responses.jsonl",
                "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_labels.jsonl",
                "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_signal_calibration.json",
                "route_pool_evidence_review_calibration_pipeline_dryrun/calibration_route_pool_evidence_review_promotion_gate.json",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_csv_pipeline_manifest.md",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels.jsonl",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels_report.json",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_labels_unreviewed.jsonl",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_signal_calibration.json",
                "route_pool_evidence_review_calibration_csv_pipeline_blank/calibration_human_route_pool_evidence_review_promotion_gate.json",
                "route_pool_evidence_review_calibration_packet/route_pool_evidence_review_calibration_subset_TO_FILL.csv",
                "route_pool_evidence_review_calibration_packet/README.md",
                "template_outcome_supervision_pack/template_outcome_supervision.jsonl",
                "template_outcome_supervision_pack/template_outcome_supervision_report.md",
                "template_outcome_review_batch/template_outcome_review_batch.csv",
                "template_outcome_review_batch/template_outcome_review_batch_report.md",
            ]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("ok", encoding="utf-8")
            (root / "phase2_decision_summary.json").write_text(
                json.dumps(
                    {
                        "decision": {
                            "search_integration_ready": False,
                            "recommendation": "do_not_integrate_search; test",
                        },
                        "decision_gates": {
                            "min_meaningful_delta": 0.01,
                            "two_stage_selector_pair_beats_frequency": False,
                            "template_twostage_pair_selector_improves_mrr": False,
                            "rc_pair_selector_improves_mrr": False,
                            "oracle_pair_upper_bound_exists": True,
                        },
                        "selector_reports": {
                            "transform_pair_train_vocab": {"status": "ok", "delta": {"mrr_all_groups": -0.1}},
                            "template_twostage_freq_top3_pair": {"status": "ok", "delta": {"mrr": -0.1}},
                            "template_twostage_freq_top3_rc_pair": {"status": "ok", "delta": {"mrr": -0.1}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_cascade_evidence_summary.json").write_text(
                json.dumps(
                    {
                        "interpretation": {
                            "audits_present": 3,
                            "strict_same_pair_analog_is_sparse": True,
                            "recommended_use": "diagnostic route-pool evidence only",
                        },
                        "audits": {
                            "v4_test20": {
                                "routes": 10,
                                "any_analog_route_rate": 0.2,
                                "same_pair_analog_route_rate": 0.1,
                            },
                            "statin": {
                                "routes": 10,
                                "any_analog_route_rate": 0.2,
                                "same_pair_analog_route_rate": 0.0,
                            },
                            "full100": {
                                "routes": 10,
                                "any_analog_route_rate": 0.2,
                                "same_pair_analog_route_rate": 0.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_batch_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "sampled_rows": 2,
                            "sampled_classes": {
                                "any_analog_supported": 1,
                                "multistep_without_observed_pair": 1,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_transform_sanity_audit.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "rows": 2,
                            "steps": 4,
                            "rows_with_label_mismatch": 1,
                            "row_label_mismatch_rate": 0.5,
                        },
                        "interpretation": {
                            "transform_labels_safe_for_training_without_review": False,
                            "label_noise_warning": True,
                            "recommended_use": "review_triage_only; do not use route-block transform labels as supervised training labels without review",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_worklist_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "rows": 2,
                            "rows_with_transform_label_warning": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_calibration_subset_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "selected_rows": 36,
                            "classes": {
                                "any_analog_supported": 18,
                                "multistep_without_observed_pair": 17,
                                "same_pair_analog_supported": 1,
                            },
                            "pools": {"20": 12, "full100": 12, "statin": 12},
                            "transform_label_warning": {"True": 18, "False": 18},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "human_route_pool_evidence_review_labels_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "csv_rows": 2,
                            "accepted_rows": 0,
                            "invalid_rows": 0,
                            "unreviewed_rows": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_worklist.jsonl").write_text(
                json.dumps(
                    {
                        "review_id": "abc",
                        "upstream_rxn": "C>>CC",
                        "downstream_rxn": "CC>>CCC",
                        "transform_label_warning": True,
                        "transform_label_warning_reasons": ["label_hydrolysis_without_hydrolysis_like_change"],
                        "expert_route_plausible": None,
                        "expert_block_transform_correct": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_batch" / "route_pool_evidence_review_batch.jsonl").write_text(
                json.dumps(
                    {
                        "evidence_class": "any_analog_supported",
                        "review_block": {"any_analog_supported": True},
                        "expert_route_plausible": None,
                        "expert_block_transform_correct": None,
                        "expert_support_precedent_relevant": None,
                        "expert_cascade_coherent": None,
                        "expert_priority": None,
                        "expert_reject_reason": None,
                        "expert_comments": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_prompts_report.json").write_text(
                json.dumps(
                    {
                        "summary": {"prompt_rows": 2},
                        "expected_output_schema": {
                            "route_plausible": "yes|no|unclear",
                            "cascade_coherent": "yes|no|unclear",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_calibration_prompts_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "selected_prompts": 36,
                            "missing_ids": 0,
                            "duplicate_subset_ids": 0,
                            "rows_with_transform_label_warning": 18,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_prompts.jsonl").write_text(
                json.dumps(
                    {
                        "prompt": "Return JSON only. transform_sanity is heuristic triage.",
                        "expected_output_schema": {"route_plausible": "yes|no|unclear"},
                        "transform_sanity": {"block_has_label_mismatch": True},
                        "route_block": {
                            "upstream_rxn": "C>>CC",
                            "downstream_rxn": "CC>>CCC",
                            "upstream_product": "CC",
                            "downstream_product": "CCC",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "route_pool_evidence_review_calibration_prompts.jsonl").write_text(
                json.dumps(
                    {
                        "review_id": "abc",
                        "prompt": "Return JSON only. transform_sanity",
                        "transform_sanity": {"block_has_label_mismatch": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_run_report.json").write_text(
                json.dumps({"summary": {"dry_run": True, "written_rows": 1}}),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_labels_report.json").write_text(
                json.dumps({"summary": {"accepted_rows": 1, "invalid_rows": 0}}),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_labels.jsonl").write_text(
                json.dumps({"review_id": "abc", "transform_sanity": {"block_has_label_mismatch": True}}) + "\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_label_summary.json").write_text(
                json.dumps(
                    {
                        "decision": {
                            "reviewed_enough_for_proxy_calibration": False,
                            "recommendation": "insufficient real review labels; run human/LLM review before training",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_signal_calibration.json").write_text(
                json.dumps(
                    {
                        "decision": {
                            "ready_for_proxy_training": False,
                            "recommendation": "do not train a scorer from these review labels yet",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_prompts" / "dryrun_route_pool_evidence_review_promotion_gate.json").write_text(
                json.dumps({"ready_for_training": False, "recommendation": "do not train or promote"}),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_pipeline_dryrun" / "dryrun_route_pool_evidence_review_pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "summaries": {
                            "run": {"written_rows": 1},
                            "labels": {"accepted_rows": 1},
                            "signal_calibration": {"ready_for_proxy_training": False},
                            "promotion_gate": {"ready_for_training": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_pipeline_dryrun" / "dryrun_route_pool_evidence_review_promotion_gate.json").write_text(
                json.dumps({"ready_for_training": False}),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_calibration_pipeline_dryrun" / "calibration_route_pool_evidence_review_pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "summaries": {
                            "run": {"prompt_rows_total": 36},
                            "signal_calibration": {"ready_for_proxy_training": False},
                            "promotion_gate": {"ready_for_training": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_calibration_csv_pipeline_blank" / "calibration_human_route_pool_evidence_review_csv_pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "summaries": {
                            "labels": {"csv_rows": 36, "accepted_rows": 0, "unreviewed_rows": 36},
                            "signal_calibration": {"ready_for_proxy_training": False},
                            "promotion_gate": {"ready_for_training": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_calibration_packet" / "route_pool_review_calibration_packet.json").write_text(
                json.dumps({"contract": {"does_not_create_labels": True}}),
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_calibration_packet" / "route_pool_evidence_review_calibration_subset_TO_FILL.csv").write_text(
                "review_id,expert_risk_tags\nabc,\n",
                encoding="utf-8",
            )
            (root / "route_pool_evidence_review_calibration_packet" / "README.md").write_text(
                "run_route_pool_evidence_review_csv_pipeline",
                encoding="utf-8",
            )
            (root / "template_outcome_supervision_pack" / "template_outcome_supervision_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "rows": 3,
                            "targets": 2,
                            "classes": {
                                "pair_and_analog_positive": 1,
                                "high_score_hard_negative": 1,
                                "pair_only_near_miss": 1,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "template_outcome_review_batch" / "template_outcome_review_batch.jsonl").write_text(
                json.dumps(
                    {
                        "expert_template_applicable": None,
                        "expert_outcome_plausible": None,
                        "expert_cascade_coherent": None,
                        "expert_priority": None,
                        "expert_reject_reason": None,
                        "expert_comments": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "template_outcome_review_batch" / "template_outcome_review_batch_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "sampled_rows": 3,
                            "sampled_classes": {
                                "pair_and_analog_positive": 1,
                                "high_score_hard_negative": 1,
                                "analog_only_positive": 1,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = verify_cascadebench_phase2_closure(root=root)

        self.assertTrue(report["ok"])
        self.assertEqual(report["summary"]["failed"], 0)

    def test_template_outcome_review_batch_has_expert_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            supervision = root / "supervision.jsonl"
            out = root / "review.jsonl"
            report_path = root / "review_report.json"
            rows = []
            for idx, (pair, analog) in enumerate(
                [
                    (True, True),
                    (False, True),
                    (True, False),
                    (False, False),
                ]
            ):
                row = {
                    "source_pool": "pool",
                    "split": "test",
                    "target_smiles": f"CC{idx}",
                    "supervision_class": _class_name(pair, analog),
                    "connector": "CC",
                    "template": "[C:1]>>[C:1]",
                    "reactants": ["C"],
                    "labels": {"pair_hit": pair, "analog_hit": analog, "pair_and_analog": pair and analog},
                    "similarities": {"upstream_similarity": 0.7, "downstream_similarity": 0.7},
                    "features": {"rc_inherited_atom_fraction": 1.0},
                }
                rows.append(row)
            supervision.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = sample_template_outcome_review_batch(
                supervision_jsonl=supervision,
                output_jsonl=out,
                report_json=report_path,
                per_class=1,
                seed=1,
            )
            sampled = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["sampled_rows"], 4)
        self.assertTrue(all("review_id" in row for row in sampled))
        self.assertTrue(all("expert_template_applicable" in row for row in sampled))
        self.assertTrue(all(row["expert_template_applicable"] is None for row in sampled))

    def test_route_pool_cascade_evidence_keeps_strict_pair_analog_separate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            manifest = root / "manifest.json"
            pool = root / "pool.jsonl"
            out = root / "audit.json"
            report = root / "audit.md"
            train.write_text(json.dumps(_program("CCC", "oxidation", "reduction")) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"outputs": {"train": str(train), "val": str(train), "test": str(train)}}),
                encoding="utf-8",
            )
            route = {
                "route_id": "r1",
                "target_smiles": "CCC",
                "target_id": "t1",
                "native_rank": 0,
                "stock_closed": True,
                "steps": [
                    {
                        "step_id": "s1",
                        "step_index": 1,
                        "rxn_smiles": "C>>CC",
                        "product_smiles": "CC",
                        "reactants": ["C"],
                        "main_reactant": "C",
                        "transformation_superclass": "hydrolysis",
                    },
                    {
                        "step_id": "s2",
                        "step_index": 2,
                        "rxn_smiles": "CC>>CCC",
                        "product_smiles": "CCC",
                        "reactants": ["CC"],
                        "main_reactant": "CC",
                        "transformation_superclass": "hydrolysis",
                    },
                ],
            }
            pool.write_text(json.dumps(route) + "\n", encoding="utf-8")

            result = audit_route_pool_cascade_evidence(
                route_pool=pool,
                program_manifest=manifest,
                output_json=out,
                output_md=report,
                evidence_split="train",
                analog_similarity=0.55,
            )

        summary = result["summary"]
        route_audit = result["routes"][0]
        self.assertEqual(summary["routes_with_any_analog_block"], 1)
        self.assertEqual(summary["routes_with_same_pair_analog_block"], 0)
        self.assertTrue(route_audit["has_any_analog_block"])
        self.assertFalse(route_audit["has_same_pair_analog_block"])

    def test_route_pool_evidence_review_batch_selects_class_specific_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = root / "route_pool_cascade_evidence_20" / "route_pool_cascade_evidence.json"
            out = root / "review.jsonl"
            report_path = root / "review_report.json"
            audit.parent.mkdir(parents=True)
            audit.write_text(
                json.dumps(
                    {
                        "metadata": {"route_pool": "pool", "analog_similarity": 0.55},
                        "examples": {
                            "any_analog_supported_routes": [
                                _route_evidence_example(
                                    "r_any",
                                    any_block=False,
                                    same_pair=False,
                                    include_second_any_block=True,
                                )
                            ],
                            "same_pair_analog_supported_routes": [
                                _route_evidence_example("r_same", any_block=True, same_pair=True)
                            ],
                            "multistep_without_observed_pair_examples": [
                                _route_evidence_example("r_none", any_block=False, same_pair=False)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = sample_route_pool_evidence_review_batch(
                audit_jsons=[audit],
                output_jsonl=out,
                report_json=report_path,
                per_class=2,
                seed=1,
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["sampled_rows"], 3)
        by_class = {row["evidence_class"]: row for row in rows}
        self.assertTrue(by_class["any_analog_supported"]["review_block"]["any_analog_supported"])
        self.assertEqual(by_class["any_analog_supported"]["review_block"]["upstream_rxn"], "CC>>CCC")
        self.assertEqual(by_class["any_analog_supported"]["review_block"]["downstream_product"], "CCCC")
        self.assertTrue(by_class["same_pair_analog_supported"]["review_block"]["same_pair_analog_supported"])
        self.assertFalse(by_class["multistep_without_observed_pair"]["review_block"]["pair_observed_in_evidence"])
        self.assertTrue(all("expert_route_plausible" in row for row in rows))

    def test_route_pool_review_prompt_and_ingest_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            review = root / "review.jsonl"
            sanity = root / "transform_sanity.json"
            prompts = root / "prompts.jsonl"
            prompt_report = root / "prompt_report.json"
            responses = root / "responses.jsonl"
            labels = root / "labels.jsonl"
            ingest_report_path = root / "ingest_report.json"
            review_row = {
                "review_id": "abc123",
                "source_pool": "pool",
                "source_value_pack": "value_pack.jsonl",
                "evidence_class": "any_analog_supported",
                "target_id": "target1",
                "target_smiles": "CCO",
                "route_id": "route1",
                "native_rank": 1,
                "native_score": 0.5,
                "n_steps": 2,
                "stock_closed": True,
                "review_block": {
                    "upstream_rxn": "C>>CC",
                    "downstream_rxn": "CC>>CCO",
                    "upstream_product": "CC",
                    "downstream_product": "CCO",
                    "upstream_main_reactant": "C",
                    "downstream_main_reactant": "CC",
                    "upstream_transform": "oxidation",
                    "downstream_transform": "amination",
                    "transform_pair": "oxidation->amination",
                },
                "diagnostic_labels": {"has_any_analog_block": True},
                "diagnostic_scores": {"best_any_block_min_sim": 0.7},
                "support_any": {"doi": "10.1/test", "transform_pair": "oxidation->amination"},
                "support_same_pair": {},
            }
            review.write_text(json.dumps(review_row) + "\n", encoding="utf-8")
            sanity.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "review_id": "abc123",
                                "block_has_label_mismatch": True,
                                "block_label_mismatch_count": 1,
                                "block_mismatch_reasons": ["label_amination_without_nitrogen_gain"],
                                "upstream": {
                                    "label": "oxidation",
                                    "inferred_classes": ["unclassified_change"],
                                    "mismatch_reasons": [],
                                },
                                "downstream": {
                                    "label": "amination",
                                    "inferred_classes": ["oxygenation_or_oxidation_like"],
                                    "mismatch_reasons": ["label_amination_without_nitrogen_gain"],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            prompt_report_payload = build_route_pool_evidence_review_prompts(
                review_jsonl=review,
                output_jsonl=prompts,
                report_json=prompt_report,
                transform_sanity_json=sanity,
            )
            prompt_rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
            responses.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "review_id": "abc123",
                                "route_plausible": "yes",
                                "block_transform_correct": "no",
                                "support_precedent_relevant": "unclear",
                                "cascade_coherent": "no",
                                "priority": "reject",
                                "risk_tags": ["wrong_transform_label"],
                                "rationale": "Transform labels do not match the shown chemistry.",
                            }
                        ),
                        json.dumps(
                            {
                                "review_id": "missing",
                                "route_plausible": "maybe",
                                "block_transform_correct": "yes",
                                "support_precedent_relevant": "yes",
                                "cascade_coherent": "yes",
                                "priority": "high",
                                "risk_tags": [],
                                "rationale": "bad id",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_report = ingest_route_pool_evidence_review_results(
                prompts_jsonl=prompts,
                responses_jsonl=responses,
                output_jsonl=labels,
                report_json=ingest_report_path,
            )
            labeled_rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(prompt_report_payload["summary"]["prompt_rows"], 1)
        self.assertEqual(prompt_report_payload["summary"]["rows_with_transform_label_warning"], 1)
        self.assertIn("Return JSON only", prompt_rows[0]["prompt"])
        self.assertEqual(prompt_rows[0]["target_id"], "target1")
        self.assertEqual(prompt_rows[0]["route_id"], "route1")
        self.assertIn("transform_sanity", prompt_rows[0])
        self.assertTrue(prompt_rows[0]["transform_sanity"]["block_has_label_mismatch"])
        self.assertEqual(ingest_report["summary"]["accepted_rows"], 1)
        self.assertEqual(ingest_report["summary"]["invalid_rows"], 1)
        self.assertEqual(labeled_rows[0]["target_id"], "target1")
        self.assertEqual(labeled_rows[0]["route_id"], "route1")
        self.assertEqual(labeled_rows[0]["expert_review"]["priority"], "reject")
        self.assertTrue(labeled_rows[0]["transform_sanity"]["block_has_label_mismatch"])
        self.assertIn("unknown_review_id", ingest_report["invalid_error_counts"])

    def test_route_pool_llm_review_runner_dry_run_outputs_ingestable_responses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            responses = root / "responses.jsonl"
            run_report_path = root / "run_report.json"
            labels = root / "labels.jsonl"
            ingest_report_path = root / "ingest_report.json"
            prompt_row = {
                "review_id": "abc123",
                "source_pool": "pool",
                "evidence_class": "any_analog_supported",
                "target_smiles": "CCO",
                "native_rank": 1,
                "stock_closed": True,
                "route_block": {
                    "upstream_rxn": "C>>CC",
                    "downstream_rxn": "CC>>CCO",
                    "upstream_product": "CC",
                    "downstream_product": "CCO",
                },
                "diagnostic_labels": {},
                "diagnostic_scores": {},
                "support_any": {},
                "support_same_pair": {},
                "expected_output_schema": {"route_plausible": "yes|no|unclear"},
                "prompt": "Return JSON only",
            }
            prompts.write_text(json.dumps(prompt_row) + "\n", encoding="utf-8")

            run_report = run_route_pool_evidence_llm_review(
                prompts_jsonl=prompts,
                output_jsonl=responses,
                report_json=run_report_path,
                dry_run=True,
            )
            ingest_report = ingest_route_pool_evidence_review_results(
                prompts_jsonl=prompts,
                responses_jsonl=responses,
                output_jsonl=labels,
                report_json=ingest_report_path,
            )

        self.assertTrue(run_report["summary"]["dry_run"])
        self.assertEqual(run_report["summary"]["written_rows"], 1)
        self.assertEqual(ingest_report["summary"]["accepted_rows"], 1)
        self.assertEqual(ingest_report["summary"]["invalid_rows"], 0)

    def test_route_pool_llm_review_runner_requires_key_for_real_run_and_can_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            responses = root / "responses.jsonl"
            run_report_path = root / "run_report.json"
            prompt_rows = [
                {
                    "review_id": "done",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "prompt": "Return JSON only",
                    "expected_output_schema": {"route_plausible": "yes|no|unclear"},
                },
                {
                    "review_id": "next",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "prompt": "Return JSON only",
                    "expected_output_schema": {"route_plausible": "yes|no|unclear"},
                },
            ]
            prompts.write_text("\n".join(json.dumps(row) for row in prompt_rows) + "\n", encoding="utf-8")
            responses.write_text(json.dumps({"review_id": "done", "response": {"review_id": "done"}}) + "\n", encoding="utf-8")

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False):
                with self.assertRaises(RuntimeError):
                    run_route_pool_evidence_llm_review(
                        prompts_jsonl=prompts,
                        output_jsonl=responses,
                        report_json=run_report_path,
                        dry_run=False,
                    )

            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "\"  replace_with_your_deepseek_key  \""},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "placeholder"):
                    run_route_pool_evidence_llm_review(
                        prompts_jsonl=prompts,
                        output_jsonl=responses,
                        report_json=run_report_path,
                        dry_run=False,
                    )

            report = run_route_pool_evidence_llm_review(
                prompts_jsonl=prompts,
                output_jsonl=responses,
                report_json=run_report_path,
                dry_run=True,
                resume=True,
            )
            rows = [json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["completed_rows_before_run"], 1)
        self.assertEqual(report["summary"]["written_rows"], 1)
        self.assertEqual([row["review_id"] for row in rows], ["done", "next"])

    def test_route_pool_llm_review_runner_parallel_dry_run_is_ingestable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            responses = root / "responses.jsonl"
            run_report_path = root / "run_report.json"
            labels = root / "labels.jsonl"
            ingest_report_path = root / "ingest_report.json"
            prompt_rows = [
                {
                    "review_id": f"rid{i}",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "prompt": "Return JSON only",
                    "expected_output_schema": {"route_plausible": "yes|no|unclear"},
                }
                for i in range(2)
            ]
            prompts.write_text("\n".join(json.dumps(row) for row in prompt_rows) + "\n", encoding="utf-8")

            run_report = run_route_pool_evidence_llm_review(
                prompts_jsonl=prompts,
                output_jsonl=responses,
                report_json=run_report_path,
                dry_run=True,
                workers=2,
            )
            ingest_report = ingest_route_pool_evidence_review_results(
                prompts_jsonl=prompts,
                responses_jsonl=responses,
                output_jsonl=labels,
                report_json=ingest_report_path,
            )

        self.assertEqual(run_report["metadata"]["workers"], 2)
        self.assertEqual(run_report["summary"]["written_rows"], 2)
        self.assertEqual(ingest_report["summary"]["accepted_rows"], 2)
        self.assertEqual(ingest_report["summary"]["invalid_rows"], 0)

    def test_route_pool_review_label_summary_refuses_tiny_dryrun_as_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.jsonl"
            out = root / "summary.json"
            labels.write_text(
                json.dumps(
                    {
                        "review_id": "abc123",
                        "source_pool": "pool",
                        "evidence_class": "any_analog_supported",
                        "diagnostic_scores": {"best_any_block_min_sim": 0.7},
                        "expert_review": {
                            "route_plausible": "unclear",
                            "block_transform_correct": "unclear",
                            "support_precedent_relevant": "unclear",
                            "cascade_coherent": "unclear",
                            "priority": "low",
                            "risk_tags": ["other"],
                            "rationale": "dry run",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = summarize_route_pool_evidence_review_labels(labels_jsonl=labels, output_json=out)

        self.assertFalse(report["decision"]["reviewed_enough_for_proxy_calibration"])
        self.assertEqual(report["summary"]["usable_positive_rows"], 0)
        self.assertEqual(report["summary"]["unclear_rows"], 1)
        self.assertIn("label_mismatch_warning", report["by_transform_sanity"])

    def test_route_pool_review_label_summary_groups_transform_sanity_warnings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.jsonl"
            out = root / "summary.json"
            rows = [
                {
                    "review_id": "warn",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "transform_sanity": {"block_has_label_mismatch": True},
                    "expert_review": {
                        "route_plausible": "no",
                        "block_transform_correct": "no",
                        "support_precedent_relevant": "no",
                        "cascade_coherent": "no",
                        "priority": "reject",
                        "risk_tags": ["wrong_transform_label"],
                        "rationale": "wrong label",
                    },
                },
                {
                    "review_id": "clean",
                    "source_pool": "pool",
                    "evidence_class": "multistep_without_observed_pair",
                    "transform_sanity": {"block_has_label_mismatch": False},
                    "expert_review": {
                        "route_plausible": "yes",
                        "block_transform_correct": "yes",
                        "support_precedent_relevant": "yes",
                        "cascade_coherent": "yes",
                        "priority": "high",
                        "risk_tags": [],
                        "rationale": "plausible",
                    },
                },
            ]
            labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = summarize_route_pool_evidence_review_labels(labels_jsonl=labels, output_json=out)

        self.assertEqual(report["by_transform_sanity"]["label_mismatch_warning"]["rows"], 1)
        self.assertEqual(report["by_transform_sanity"]["label_mismatch_warning"]["usable_negative_rate"], 1.0)
        self.assertEqual(report["by_transform_sanity"]["label_mismatch_warning"]["wrong_transform_label_risk_rate"], 1.0)
        self.assertEqual(report["by_transform_sanity"]["no_label_mismatch_warning"]["usable_positive_rate"], 1.0)

    def test_route_pool_review_promotion_gate_rejects_insufficient_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "summary.json"
            out = root / "gate.json"
            summary.write_text(
                json.dumps(
                    {
                        "summary": {
                            "rows": 2,
                            "reviewed_classes": {"any_analog_supported": 2},
                            "usable_positive_rows": 0,
                            "usable_negative_rows": 0,
                            "unclear_rows": 2,
                        },
                        "decision": {"review_rows": 2, "unclear_rate": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            report = gate_route_pool_evidence_review_promotion(label_summary_json=summary, output_json=out)

        self.assertFalse(report["ready_for_training"])
        self.assertTrue(all(not row["ok"] for row in report["checks"]))
        self.assertIn("do not train", report["recommendation"])

    def test_route_pool_review_pipeline_dry_run_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            review = root / "review.jsonl"
            sanity = root / "sanity.json"
            out_dir = root / "pipeline"
            review_rows = [
                {
                    "review_id": f"rid{i}",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "target_smiles": "CCO",
                    "review_block": {
                        "upstream_rxn": "C>>CC",
                        "downstream_rxn": "CC>>CCO",
                        "upstream_product": "CC",
                        "downstream_product": "CCO",
                        "upstream_main_reactant": "C",
                        "downstream_main_reactant": "CC",
                        "upstream_transform": "hydrolysis",
                        "downstream_transform": "hydrolysis",
                        "transform_pair": "hydrolysis->hydrolysis",
                    },
                }
                for i in range(2)
            ]
            review.write_text("\n".join(json.dumps(row) for row in review_rows) + "\n", encoding="utf-8")
            sanity.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "review_id": "rid0",
                                "block_has_label_mismatch": True,
                                "block_label_mismatch_count": 1,
                                "block_mismatch_reasons": ["label_hydrolysis_without_hydrolysis_like_change"],
                                "upstream": {"label": "hydrolysis", "inferred_classes": ["amination_like"], "mismatch_reasons": []},
                                "downstream": {
                                    "label": "hydrolysis",
                                    "inferred_classes": ["esterification_like"],
                                    "mismatch_reasons": ["label_hydrolysis_without_hydrolysis_like_change"],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_route_pool_evidence_review_pipeline(
                prompts_jsonl=prompts,
                output_dir=out_dir,
                review_jsonl=review,
                transform_sanity_json=sanity,
                prefix="dryrun",
                max_rows=2,
                workers=2,
            )
            manifest_exists = Path(manifest["outputs"]["pipeline_manifest_json"]).exists()

        self.assertEqual(manifest["summaries"]["prompts"]["prompt_rows"], 2)
        self.assertEqual(manifest["summaries"]["prompts"]["rows_with_transform_label_warning"], 1)
        self.assertEqual(manifest["summaries"]["run"]["written_rows"], 2)
        self.assertEqual(manifest["summaries"]["labels"]["accepted_rows"], 2)
        self.assertEqual(manifest["summaries"]["label_summary"]["rows"], 2)
        self.assertFalse(manifest["summaries"]["signal_calibration"]["ready_for_proxy_training"])
        self.assertFalse(manifest["summaries"]["promotion_gate"]["ready_for_training"])
        self.assertTrue(manifest_exists)

    def test_route_pool_transform_sanity_flags_obvious_label_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            review = root / "review.jsonl"
            out = root / "transform_sanity.json"
            row = {
                "review_id": "mismatch",
                "source_pool": "pool",
                "evidence_class": "any_analog_supported",
                "target_smiles": "CCN",
                "native_rank": 1,
                "review_block": {
                    "upstream_rxn": "CC(=O)O>>CC(N)O",
                    "downstream_rxn": "CC(N)O.CO>>CC(N)OC",
                    "upstream_product": "CC(N)O",
                    "downstream_product": "CC(N)OC",
                    "upstream_main_reactant": "CC(=O)O",
                    "downstream_main_reactant": "CC(N)O",
                    "upstream_transform": "hydrolysis",
                    "downstream_transform": "hydrolysis",
                    "transform_pair": "hydrolysis->hydrolysis",
                },
            }
            review.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = audit_route_pool_review_transform_sanity(
                review_jsonl=review,
                output_json=out,
                mismatch_warning_rate=0.01,
            )

        self.assertEqual(report["summary"]["rows"], 1)
        self.assertEqual(report["summary"]["rows_with_label_mismatch"], 1)
        self.assertTrue(report["interpretation"]["label_noise_warning"])
        self.assertFalse(report["interpretation"]["transform_labels_safe_for_training_without_review"])
        self.assertIn("review_triage_only", report["interpretation"]["recommended_use"])
        self.assertIn(
            "label_hydrolysis_without_hydrolysis_like_change",
            report["rows"][0]["block_mismatch_reasons"],
        )

    def test_route_pool_transform_sanity_leaves_unknown_labels_as_triage_not_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            review = root / "review.jsonl"
            out = root / "transform_sanity.json"
            row = {
                "review_id": "unknown",
                "source_pool": "pool",
                "evidence_class": "multistep_without_observed_pair",
                "review_block": {
                    "upstream_rxn": "C>>CC",
                    "downstream_rxn": "CC>>CCC",
                    "upstream_product": "CC",
                    "downstream_product": "CCC",
                    "upstream_main_reactant": "C",
                    "downstream_main_reactant": "CC",
                    "upstream_transform": "unknown",
                    "downstream_transform": "unknown",
                    "transform_pair": "unknown->unknown",
                },
            }
            review.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = audit_route_pool_review_transform_sanity(
                review_jsonl=review,
                output_json=out,
                mismatch_warning_rate=0.20,
            )

        self.assertEqual(report["summary"]["rows_with_label_mismatch"], 0)
        self.assertFalse(report["interpretation"]["label_noise_warning"])
        self.assertIn("heuristic review triage", report["interpretation"]["recommended_use"])

    def test_route_pool_review_worklist_flattens_sanity_for_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            review = root / "review.jsonl"
            sanity = root / "sanity.json"
            out = root / "worklist.jsonl"
            csv_out = root / "worklist.csv"
            report_json = root / "worklist_report.json"
            review.write_text(
                json.dumps(
                    {
                        "review_id": "rid",
                        "source_pool": "pool",
                        "evidence_class": "any_analog_supported",
                        "target_smiles": "CCO",
                        "route_id": "route",
                        "review_block": {
                            "route_block_index": 0,
                            "upstream_rxn": "C>>CC",
                            "downstream_rxn": "CC>>CCO",
                            "upstream_transform": "hydrolysis",
                            "downstream_transform": "hydrolysis",
                            "transform_pair": "hydrolysis->hydrolysis",
                            "any_analog_supported": True,
                        },
                        "support_any": {"doi": "10/test", "transform_pair": "hydrolysis->hydrolysis"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sanity.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "review_id": "rid",
                                "block_has_label_mismatch": True,
                                "block_label_mismatch_count": 1,
                                "block_mismatch_reasons": ["label_hydrolysis_without_hydrolysis_like_change"],
                                "upstream": {"inferred_classes": ["amination_like"], "mismatch_reasons": []},
                                "downstream": {"inferred_classes": ["esterification_like"], "mismatch_reasons": []},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = export_route_pool_review_worklist(
                review_jsonl=review,
                transform_sanity_json=sanity,
                output_jsonl=out,
                output_csv=csv_out,
                report_json=report_json,
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            csv_exists = csv_out.exists()

        self.assertEqual(report["summary"]["rows"], 1)
        self.assertEqual(report["summary"]["rows_with_transform_label_warning"], 1)
        self.assertTrue(rows[0]["transform_label_warning"])
        self.assertEqual(rows[0]["upstream_rxn"], "C>>CC")
        self.assertTrue(csv_exists)

    def test_route_pool_signal_calibration_requires_real_balanced_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.jsonl"
            out = root / "calibration.json"
            rows = [
                {
                    "review_id": "pos",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "native_rank": 1,
                    "diagnostic_scores": {"best_any_block_min_sim": 0.9},
                    "diagnostic_labels": {"has_any_analog_block": True},
                    "transform_sanity": {"block_has_label_mismatch": False},
                    "expert_review": {
                        "route_plausible": "yes",
                        "block_transform_correct": "yes",
                        "support_precedent_relevant": "yes",
                        "cascade_coherent": "yes",
                        "priority": "high",
                        "risk_tags": [],
                        "rationale": "positive",
                    },
                },
                {
                    "review_id": "neg",
                    "source_pool": "pool",
                    "evidence_class": "multistep_without_observed_pair",
                    "native_rank": 10,
                    "diagnostic_scores": {"best_any_block_min_sim": 0.1},
                    "diagnostic_labels": {"has_any_analog_block": False},
                    "transform_sanity": {"block_has_label_mismatch": True},
                    "expert_review": {
                        "route_plausible": "no",
                        "block_transform_correct": "no",
                        "support_precedent_relevant": "no",
                        "cascade_coherent": "no",
                        "priority": "reject",
                        "risk_tags": ["wrong_transform_label"],
                        "rationale": "negative",
                    },
                },
            ]
            labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = calibrate_route_pool_evidence_review_signals(labels_jsonl=labels, output_json=out, min_rows=30)

        self.assertFalse(report["decision"]["ready_for_proxy_training"])
        self.assertEqual(report["summary"]["label_counts"], {"positive": 1, "negative": 1})
        self.assertEqual(report["decision"]["best_numeric_signal"]["signal"], "native_rank")
        self.assertIn("do not train", report["decision"]["recommendation"])

    def test_route_pool_human_csv_ingest_validates_reviews(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "filled.csv"
            labels = root / "labels.jsonl"
            report_json = root / "report.json"
            invalid = root / "invalid.jsonl"
            unreviewed = root / "unreviewed.jsonl"
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
                "model_rank",
                "retrieval_rank",
                "audit_rank",
                "stock_closed",
                "route_block_index",
                "upstream_rxn",
                "downstream_rxn",
                "upstream_transform",
                "downstream_transform",
                "transform_pair",
                "pair_observed_in_evidence",
                "any_analog_supported",
                "same_pair_analog_supported",
                "best_any_block_min_sim",
                "best_same_pair_block_min_sim",
                "transform_label_warning",
                "transform_label_warning_count",
                "transform_label_warning_reasons",
                "upstream_inferred_classes",
                "downstream_inferred_classes",
                "support_any_doi",
                "support_any_transform_pair",
                "support_same_pair_doi",
                "support_same_pair_transform_pair",
                "expert_route_plausible",
                "expert_block_transform_correct",
                "expert_support_precedent_relevant",
                "expert_cascade_coherent",
                "expert_priority",
                "expert_reject_reason",
                "expert_risk_tags",
                "expert_comments",
            ]
            rows = [
                {
                    "review_id": "ok",
                    "target_id": "target-1",
                    "route_id": "route-1",
                    "value_split": "train",
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "target_smiles": "CCO",
                    "native_rank": "1",
                    "n_steps": "2",
                    "model_rank": "3",
                    "retrieval_rank": "4",
                    "audit_rank": "5",
                    "stock_closed": "true",
                    "route_block_index": "0",
                    "upstream_rxn": "C>>CC",
                    "downstream_rxn": "CC>>CCO",
                    "upstream_transform": "oxidation",
                    "downstream_transform": "amination",
                    "transform_pair": "oxidation->amination",
                    "pair_observed_in_evidence": "true",
                    "any_analog_supported": "true",
                    "same_pair_analog_supported": "false",
                    "best_any_block_min_sim": "0.8",
                    "best_same_pair_block_min_sim": "0.1",
                    "transform_label_warning": "false",
                    "transform_label_warning_count": "0",
                    "support_any_doi": "10/test",
                    "support_any_transform_pair": "oxidation->amination",
                    "expert_route_plausible": "yes",
                    "expert_block_transform_correct": "yes",
                    "expert_support_precedent_relevant": "yes",
                    "expert_cascade_coherent": "yes",
                    "expert_priority": "high",
                    "expert_comments": "plausible block",
                },
                {"review_id": "blank", "source_pool": "pool"},
                {
                    "review_id": "bad",
                    "source_pool": "pool",
                    "expert_route_plausible": "maybe",
                    "expert_block_transform_correct": "yes",
                    "expert_support_precedent_relevant": "yes",
                    "expert_cascade_coherent": "yes",
                    "expert_priority": "high",
                    "expert_comments": "bad enum",
                },
                {
                    "review_id": "missing-route",
                    "source_pool": "pool",
                    "expert_route_plausible": "yes",
                    "expert_block_transform_correct": "yes",
                    "expert_support_precedent_relevant": "yes",
                    "expert_cascade_coherent": "yes",
                    "expert_priority": "high",
                    "expert_comments": "valid review but missing route id",
                },
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = ingest_route_pool_evidence_review_csv(
                review_csv=csv_path,
                output_jsonl=labels,
                report_json=report_json,
                invalid_jsonl=invalid,
                unreviewed_jsonl=unreviewed,
            )
            accepted_rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
            invalid_rows = [json.loads(line) for line in invalid.read_text(encoding="utf-8").splitlines() if line.strip()]
            unreviewed_rows = [json.loads(line) for line in unreviewed.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["accepted_rows"], 1)
        self.assertEqual(report["summary"]["invalid_rows"], 2)
        self.assertEqual(report["summary"]["unreviewed_rows"], 1)
        self.assertEqual(report["summary"]["value_split_counts"], {"train": 1})
        self.assertEqual(accepted_rows[0]["target_id"], "target-1")
        self.assertEqual(accepted_rows[0]["route_id"], "route-1")
        self.assertEqual(accepted_rows[0]["value_split"], "train")
        self.assertEqual(accepted_rows[0]["n_steps"], 2)
        self.assertEqual(accepted_rows[0]["model_rank"], 3)
        self.assertEqual(accepted_rows[0]["retrieval_rank"], 4)
        self.assertEqual(accepted_rows[0]["audit_rank"], 5)
        self.assertEqual(accepted_rows[0]["expert_review"]["route_plausible"], "yes")
        self.assertEqual(accepted_rows[0]["diagnostic_scores"]["best_any_block_min_sim"], 0.8)
        self.assertIn("invalid_expert_route_plausible", invalid_rows[0]["errors"])
        self.assertTrue(any("missing_route_id" in row["errors"] for row in invalid_rows))
        self.assertEqual(unreviewed_rows[0]["review_id"], "blank")

    def test_route_pool_human_csv_pipeline_rejects_blank_worklist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "blank.csv"
            out_dir = root / "pipeline"
            fieldnames = [
                "review_id",
                "source_pool",
                "evidence_class",
                "expert_route_plausible",
                "expert_block_transform_correct",
                "expert_support_precedent_relevant",
                "expert_cascade_coherent",
                "expert_priority",
                "expert_reject_reason",
                "expert_comments",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"review_id": "blank", "source_pool": "pool", "evidence_class": "any_analog_supported"})

            manifest = run_route_pool_evidence_review_csv_pipeline(
                review_csv=csv_path,
                output_dir=out_dir,
                prefix="human",
            )

        self.assertEqual(manifest["summaries"]["labels"]["accepted_rows"], 0)
        self.assertEqual(manifest["summaries"]["labels"]["unreviewed_rows"], 1)
        self.assertFalse(manifest["summaries"]["signal_calibration"]["ready_for_proxy_training"])
        self.assertFalse(manifest["summaries"]["promotion_gate"]["ready_for_training"])

    def test_route_pool_review_calibration_packet_copies_csv_and_documents_rubric(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subset_csv = root / "subset.csv"
            subset_report = root / "subset_report.json"
            out_dir = root / "packet"
            subset_csv.write_text("review_id,expert_route_plausible,expert_risk_tags\nrid,,\n", encoding="utf-8")
            subset_report.write_text(
                json.dumps({"summary": {"selected_rows": 1, "transform_label_warning": {"True": 1}}}),
                encoding="utf-8",
            )

            packet = build_route_pool_review_calibration_packet(
                subset_csv=subset_csv,
                subset_report_json=subset_report,
                output_dir=out_dir,
                min_evidence_classes=1,
            )
            editable = Path(packet["editable_csv"])
            editable_exists = editable.exists()
            editable_header = editable.read_text(encoding="utf-8").splitlines()[0]
            readme = (out_dir / "README.md").read_text(encoding="utf-8")

        self.assertTrue(editable_exists)
        self.assertIn("expert_risk_tags", editable_header)
        self.assertIn("expert_route_plausible", readme)
        self.assertIn("run_route_pool_evidence_review_csv_pipeline", packet["pipeline_command"])
        self.assertIn("--min-evidence-classes 1", packet["pipeline_command"])
        self.assertIn("value_split", packet["context_columns"])
        self.assertIn("value_split", readme)
        self.assertTrue(packet["contract"]["does_not_create_labels"])
        self.assertTrue(packet["contract"]["context_columns_must_remain_unchanged"])

    def test_route_pool_calibration_subset_is_balanced_and_preserves_blanks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worklist = root / "worklist.jsonl"
            out = root / "subset.jsonl"
            out_csv = root / "subset.csv"
            report_json = root / "subset_report.json"
            rows = []
            idx = 0
            for cls in ("any_analog_supported", "multistep_without_observed_pair"):
                for pool in ("20", "full100", "statin"):
                    for warning in (True, False):
                        idx += 1
                        rows.append(_worklist_row(f"rid{idx}", cls, pool, warning))
            rows.append(_worklist_row("same", "same_pair_analog_supported", "20", True))
            worklist.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = select_route_pool_review_calibration_subset(
                worklist_jsonl=worklist,
                output_jsonl=out,
                output_csv=out_csv,
                report_json=report_json,
                size=8,
            )
            selected = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            csv_exists = out_csv.exists()

        self.assertEqual(report["summary"]["selected_rows"], 8)
        self.assertIn("same_pair_analog_supported", report["summary"]["classes"])
        self.assertIn("True", report["summary"]["transform_label_warning"])
        self.assertIn("False", report["summary"]["transform_label_warning"])
        self.assertTrue(any(row["review_id"] == "same" for row in selected))
        self.assertTrue(all(row.get("expert_route_plausible") is None for row in selected))
        self.assertTrue(csv_exists)

    def test_route_pool_prompt_subset_follows_calibration_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            subset = root / "subset.jsonl"
            out = root / "subset_prompts.jsonl"
            report_json = root / "subset_prompts_report.json"
            prompt_rows = [
                {
                    "review_id": rid,
                    "source_pool": "pool",
                    "evidence_class": "any_analog_supported",
                    "transform_sanity": {"block_has_label_mismatch": rid == "b"},
                    "prompt": "Return JSON only. transform_sanity",
                }
                for rid in ("a", "b", "c")
            ]
            prompts.write_text("\n".join(json.dumps(row) for row in prompt_rows) + "\n", encoding="utf-8")
            subset.write_text(
                "\n".join(json.dumps({"review_id": rid}) for rid in ("b", "a")) + "\n",
                encoding="utf-8",
            )

            report = select_route_pool_review_prompt_subset(
                prompts_jsonl=prompts,
                subset_jsonl=subset,
                output_jsonl=out,
                report_json=report_json,
            )
            selected = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(report["summary"]["selected_prompts"], 2)
        self.assertEqual(report["summary"]["missing_ids"], 0)
        self.assertEqual([row["review_id"] for row in selected], ["b", "a"])
        self.assertEqual(report["summary"]["rows_with_transform_label_warning"], 1)

    def test_route_pool_prompt_subset_reports_missing_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            subset = root / "subset.jsonl"
            out = root / "subset_prompts.jsonl"
            report_json = root / "subset_prompts_report.json"
            prompts.write_text(json.dumps({"review_id": "a", "prompt": "Return JSON only"}) + "\n", encoding="utf-8")
            subset.write_text(json.dumps({"review_id": "missing"}) + "\n", encoding="utf-8")

            report = select_route_pool_review_prompt_subset(
                prompts_jsonl=prompts,
                subset_jsonl=subset,
                output_jsonl=out,
                report_json=report_json,
            )

        self.assertEqual(report["summary"]["selected_prompts"], 0)
        self.assertEqual(report["summary"]["missing_ids"], 1)
        self.assertEqual(report["missing_review_ids"], ["missing"])


def _program(target, first_transform, second_transform):
    return {
        "program_id": f"{target}-{first_transform}-{second_transform}",
        "target_smiles": target,
        "steps": [
            {
                "rxn_smiles": "C>>CC",
                "product_smiles": "CC",
                "main_reactant": "C",
                "transformation_superclass": first_transform,
            },
            {
                "rxn_smiles": f"CC>>{target}",
                "product_smiles": target,
                "main_reactant": "CC",
                "transformation_superclass": second_transform,
            },
        ],
    }


def _proposal(pair, analog, score):
    return {
        "proposal_score": score,
        "proposal_rank": 1,
        "downstream_rank": 1,
        "connector": "CC",
        "template_rank": 1,
        "outcome_rank": 1,
        "template_transform_pair": "oxidation->reduction",
        "template": "[C:1]>>[C:1]",
        "reactants": ["C"],
        "main_reactant": "C",
        "pair_hit": pair,
        "analog_hit": analog,
        "pair_and_analog": pair and analog,
        "upstream_similarity": 0.7 if analog else 0.1,
        "downstream_similarity": 0.7 if analog else 0.1,
        "app_connector_main_similarity": 0.5,
        "rc_inherited_atom_fraction": 1.0,
    }


def _class_name(pair, analog):
    if pair and analog:
        return "pair_and_analog_positive"
    if analog:
        return "analog_only_positive"
    if pair:
        return "pair_only_near_miss"
    return "high_score_hard_negative"


def _route_evidence_example(route_id, any_block, same_pair, include_second_any_block=False):
    blocks = [
        {
            "route_block_index": 0,
            "upstream_rxn": "C>>CC",
            "downstream_rxn": "CC>>CCC",
            "upstream_product": "CC",
            "downstream_product": "CCC",
            "upstream_main_reactant": "C",
            "downstream_main_reactant": "CC",
            "upstream_transform": "hydrolysis",
            "downstream_transform": "hydrolysis",
            "transform_pair": "hydrolysis->hydrolysis",
            "pair_count_in_evidence": 1 if any_block or same_pair else 0,
            "pair_observed_in_evidence": bool(any_block or same_pair),
            "best_any_block_min_sim": 0.1,
            "best_any_block_mean_sim": 0.1,
            "best_same_pair_block_min_sim": 0.1,
            "best_same_pair_block_mean_sim": 0.1,
            "any_analog_supported": bool(any_block),
            "same_pair_analog_supported": bool(same_pair),
        }
    ]
    if include_second_any_block:
        blocks.append(
            {
                "route_block_index": 1,
                "upstream_rxn": "CC>>CCC",
                "downstream_rxn": "CCC>>CCCC",
                "upstream_product": "CCC",
                "downstream_product": "CCCC",
                "upstream_main_reactant": "CC",
                "downstream_main_reactant": "CCC",
                "upstream_transform": "oxidation",
                "downstream_transform": "amination",
                "transform_pair": "oxidation->amination",
                "pair_count_in_evidence": 2,
                "pair_observed_in_evidence": True,
                "best_any_block_min_sim": 0.8,
                "best_any_block_mean_sim": 0.85,
                "best_same_pair_block_min_sim": 0.2,
                "best_same_pair_block_mean_sim": 0.2,
                "any_analog_supported": True,
                "same_pair_analog_supported": False,
            }
        )
    return {
        "route_id": route_id,
        "target_id": "t",
        "target_smiles": "CCO",
        "native_rank": 1,
        "stock_closed": True,
        "n_steps": 2,
        "n_blocks": len(blocks),
        "transform_pairs": [block["transform_pair"] for block in blocks],
        "has_observed_pair_block": any(block["pair_observed_in_evidence"] for block in blocks),
        "has_any_analog_block": any(block["any_analog_supported"] for block in blocks),
        "has_same_pair_analog_block": any(block["same_pair_analog_supported"] for block in blocks),
        "observed_pair_block_count": sum(1 for block in blocks if block["pair_observed_in_evidence"]),
        "any_analog_block_count": sum(1 for block in blocks if block["any_analog_supported"]),
        "same_pair_analog_block_count": sum(1 for block in blocks if block["same_pair_analog_supported"]),
        "known_transform_step_fraction": 1.0,
        "best_any_block_min_sim": max(block["best_any_block_min_sim"] for block in blocks),
        "best_same_pair_block_min_sim": max(block["best_same_pair_block_min_sim"] for block in blocks),
        "top_supported_block": blocks[0],
        "blocks": blocks,
    }


def _worklist_row(review_id, evidence_class, source_pool, warning):
    return {
        "review_id": review_id,
        "source_pool": source_pool,
        "evidence_class": evidence_class,
        "target_smiles": "CCO",
        "route_id": review_id,
        "native_rank": 1,
        "stock_closed": True,
        "route_block_index": 0,
        "transform_pair": "hydrolysis->hydrolysis" if warning else "unknown->unknown",
        "upstream_rxn": "C>>CC",
        "downstream_rxn": "CC>>CCO",
        "transform_label_warning": bool(warning),
        "transform_label_warning_reasons": ["label_hydrolysis_without_hydrolysis_like_change"] if warning else [],
        "expert_route_plausible": None,
        "expert_block_transform_correct": None,
        "expert_support_precedent_relevant": None,
        "expert_cascade_coherent": None,
        "expert_priority": None,
        "expert_reject_reason": None,
        "expert_comments": None,
    }


if __name__ == "__main__":
    unittest.main()
