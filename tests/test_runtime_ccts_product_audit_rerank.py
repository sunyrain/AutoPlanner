import unittest

from cascade_planner.eval.rerank_runtime_ccts_with_product_audit import _ranking_key


class RuntimeCCTSProductAuditRerankTest(unittest.TestCase):
    def test_audit_guard_beats_higher_ccts_artifact(self):
        artifact = {
            "native_rank": 0,
            "runtime_ccts_route_pool_ranker_score": 10.0,
            "runtime_ccts_product_audit_features": {"route_class": "reject_artifact"},
        }
        triage = {
            "native_rank": 1,
            "runtime_ccts_route_pool_ranker_score": -10.0,
            "runtime_ccts_product_audit_features": {"route_class": "triage_late_stage"},
        }

        ordered = sorted([artifact, triage], key=lambda row: _ranking_key(row, ranking_mode="audit_guarded_ccts_ranker"))

        self.assertIs(ordered[0], triage)

    def test_issue_guard_beats_higher_ccts_generic_within_same_class(self):
        generic = {
            "native_rank": 0,
            "runtime_ccts_route_pool_ranker_score": 10.0,
            "runtime_ccts_product_audit_features": {
                "route_class": "triage_late_stage",
                "issues": ["generic_reaction_sequence"],
            },
        }
        clean = {
            "native_rank": 1,
            "runtime_ccts_route_pool_ranker_score": -10.0,
            "runtime_ccts_product_audit_features": {
                "route_class": "triage_late_stage",
                "issues": [],
            },
        }

        ordered = sorted([generic, clean], key=lambda row: _ranking_key(row, ranking_mode="audit_guarded_ccts_ranker"))

        self.assertIs(ordered[0], clean)

    def test_ccts_only_can_promote_high_score(self):
        low = {
            "native_rank": 0,
            "runtime_ccts_route_pool_ranker_score": -1.0,
            "runtime_ccts_product_audit_features": {"route_class": "triage_late_stage"},
        }
        high = {
            "native_rank": 1,
            "runtime_ccts_route_pool_ranker_score": 2.0,
            "runtime_ccts_product_audit_features": {"route_class": "triage_late_stage"},
        }

        ordered = sorted([low, high], key=lambda row: _ranking_key(row, ranking_mode="ccts_ranker_only"))

        self.assertIs(ordered[0], high)


if __name__ == "__main__":
    unittest.main()
