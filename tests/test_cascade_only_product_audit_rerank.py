import unittest

from cascade_planner.eval.rerank_cascade_only_features_with_product_audit import (
    _feature_row,
    _key_for_route,
    _ranking_key,
    _validate_alignment,
)


class CascadeOnlyProductAuditRerankTest(unittest.TestCase):
    def test_audit_guard_beats_higher_cascade_score_artifact(self):
        artifact = {
            "native_rank": 0,
            "cascade_only_route_pool_ranker_score": 10.0,
            "cascade_only_product_audit_features": {"route_class": "reject_artifact"},
        }
        triage = {
            "native_rank": 1,
            "cascade_only_route_pool_ranker_score": -10.0,
            "cascade_only_product_audit_features": {"route_class": "triage_late_stage"},
        }

        ordered = sorted([artifact, triage], key=lambda row: _ranking_key(row, ranking_mode="audit_guarded_cascade_ranker"))

        self.assertIs(ordered[0], triage)

    def test_issue_guard_beats_higher_cascade_score_generic_within_same_class(self):
        generic = {
            "native_rank": 0,
            "cascade_only_route_pool_ranker_score": 10.0,
            "cascade_only_product_audit_features": {
                "route_class": "triage_late_stage",
                "issues": ["generic_reaction_sequence"],
            },
        }
        clean = {
            "native_rank": 1,
            "cascade_only_route_pool_ranker_score": -10.0,
            "cascade_only_product_audit_features": {
                "route_class": "triage_late_stage",
                "issues": [],
            },
        }

        ordered = sorted([generic, clean], key=lambda row: _ranking_key(row, ranking_mode="audit_guarded_cascade_ranker"))

        self.assertIs(ordered[0], clean)

    def test_alignment_failure_is_explicit(self):
        run = {
            "targets": [
                {
                    "target_id": "target_a",
                    "routes": [{"native_rank": 0}],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "alignment failed"):
            _validate_alignment(run, enriched={}, blocks={})

    def test_feature_row_combines_runtime_block_and_v4_step_evidence(self):
        route = {
            "native_rank": 0,
            "ccts_v3_runtime_best_route_evidence": 0.3,
            "ccts_v3_runtime_model_max": 0.4,
            "ccts_v3_runtime_model_mean": 0.2,
            "ccts_v3_runtime_step_any_max": 0.5,
            "ccts_v3_runtime_step_any_mean": 0.25,
            "ccts_v3_runtime_step_pair_max": 0.1,
            "ccts_v3_runtime_step_pair_mean": 0.05,
        }
        enriched = {
            "native_rank": 0,
            "steps": [
                {"v4_step_evidence": {"matched": True, "accepted": True, "similarity": 0.7}},
                {"v4_step_evidence": {"matched": True, "accepted": False, "similarity": 0.2}},
            ],
        }
        block = {
            "native_rank": 0,
            "block_coherence": {
                "conservative_route_coherence_score": 0.11,
                "low_block_count_lt_0_25": 1,
                "max": 0.8,
                "mean": 0.6,
                "min": 0.4,
                "rerank_score": 0.62,
                "route_coherence_score": 0.6,
                "n_blocks": 2,
            },
        }

        feature = _feature_row(route, enriched, block, route_count=3)

        self.assertEqual(feature["ccts_step_any_max"], 0.5)
        self.assertEqual(feature["block_low_count_lt_0_25"], 1.0)
        self.assertEqual(feature["v4_step_matched_rate"], 1.0)
        self.assertEqual(feature["v4_step_accepted_rate"], 0.5)
        self.assertAlmostEqual(feature["v4_step_similarity_mean"], 0.45)

    def test_key_for_route_preserves_zero_index(self):
        key = _key_for_route({"index": 0, "target_smiles": "CCO"}, {}, native_rank=2)

        self.assertEqual(key, ("0", 2))

    def test_key_for_route_uses_target_index_before_smiles(self):
        key = _key_for_route({"target_smiles": "CCO"}, {}, native_rank=2, target_index=0)

        self.assertEqual(key, ("0", 2))


if __name__ == "__main__":
    unittest.main()
