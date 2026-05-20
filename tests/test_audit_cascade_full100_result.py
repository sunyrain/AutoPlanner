import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.audit_cascade_full100_result import audit_cascade_full100_result


class AuditCascadeFull100ResultTest(unittest.TestCase):
    def test_audit_separates_candidate_generation_and_ranking_gaps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "summary": {"n_targets": 3},
                        "targets": [
                            _row("A", candidate_reactant=False, route_reactant=False, programs=[]),
                            _row(
                                "B",
                                candidate_reactant=True,
                                route_reactant=False,
                                programs=[
                                    {"rank": 1, "gt_reactant_hit_count": 0},
                                    {"rank": 2, "gt_reactant_hit_count": 1},
                                ],
                            ),
                            _row(
                                "C",
                                candidate_reactant=True,
                                route_reactant=True,
                                programs=[{"rank": 1, "gt_reactant_hit_count": 1}],
                            ),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_cascade_full100_result(result_path=result)

        self.assertEqual(report["funnel"]["bins"]["candidate_generation_miss"], 1)
        self.assertEqual(report["funnel"]["bins"]["generated_reactant_but_not_route"], 1)
        self.assertEqual(report["funnel"]["bins"]["route_contains_gt_reactant"], 1)
        self.assertEqual(report["target_uniqueness"]["n_unique_targets"], 3)
        self.assertEqual(report["topk_route_metrics"]["top1"]["gt_reactant_count"], 1)
        self.assertEqual(report["topk_route_metrics"]["top2"]["gt_reactant_count"], 2)
        self.assertEqual(report["topk_route_metrics"]["top5_oracle_rerank_gain_over_top1"], 1)
        self.assertEqual(report["gap_analysis"]["bucket_by_depth"]["2"]["candidate_generation_miss"], 1)
        self.assertEqual(report["gap_analysis"]["early_closure"]["n_result_shorter_than_gt_depth"], 3)
        self.assertIn("next_actions", report["diagnosis"])


def _row(
    target: str,
    *,
    candidate_reactant: bool,
    route_reactant: bool,
    programs: list[dict],
) -> dict:
    return {
        "target_smiles": target,
        "route_domain": "all_chemical",
        "depth": 2,
        "gt_route": [
            {"rxn_smiles": "A>>B", "transformation": "oxidation"},
            {"rxn_smiles": "B>>C", "transformation": "reduction"},
        ],
        "recovery": {
            "candidate_exact_reaction_in_pool": False,
            "candidate_gt_reactant_in_pool": candidate_reactant,
            "exact_reaction_in_route_pool": False,
            "gt_reactant_in_route_pool": route_reactant,
        },
        "cascade_search": {
            "solved": True,
            "step_count": 1,
            "stats": {
                "expansions": 1,
                "generated_actions": 2,
                "stop_reason": "result_limit",
            },
            "failure_categories": [],
            "result_programs": programs,
        },
        "chem_enzy": {
            "route_count": 3,
            "proposal_step_count": 4,
        },
    }


if __name__ == "__main__":
    unittest.main()
