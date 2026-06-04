import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_legal_corpus_search_fusion import diagnose_search_fusion, render_markdown


class DiagnoseLegalCorpusSearchFusionTest(unittest.TestCase):
    def test_classifies_retained_and_lost_direct_hits(self):
        proposal_audit = {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "returned": 2,
                    "target_step_gt_reaction_hit": True,
                    "target_step_gt_reaction_best_rank": 1,
                    "gt_reactant_hit": True,
                    "gt_reactant_best_rank": 1,
                },
                {
                    "target_smiles": "CCN",
                    "returned": 2,
                    "target_step_gt_reaction_hit": True,
                    "target_step_gt_reaction_best_rank": 2,
                    "gt_reactant_hit": True,
                    "gt_reactant_best_rank": 2,
                },
                {
                    "target_smiles": "CCC",
                    "returned": 2,
                    "target_step_gt_reaction_hit": False,
                    "gt_reactant_hit": True,
                    "gt_reactant_best_rank": 2,
                },
                {
                    "target_smiles": "CCCC",
                    "returned": 2,
                    "target_step_gt_reaction_hit": False,
                    "gt_reactant_hit": False,
                },
            ]
        }
        route_run = {
            "metadata": {"cascade_search": {"use_legal_corpus_proposals": True}},
            "summary": {},
            "targets": [
                {
                    "target_smiles": "CCO",
                    "cascade_search": {
                        "n_results": 1,
                        "failure_categories": [],
                        "result_programs": [
                            {"rank": 1, "exact_reaction_hit_count": 1, "gt_reactant_hit_count": 1}
                        ],
                    },
                    "recovery": {
                        "exact_reaction_in_route_pool": True,
                        "gt_reactant_in_route_pool": True,
                        "result_exact_reaction_in_pool": True,
                        "result_gt_reactant_in_pool": True,
                    },
                },
                {
                    "target_smiles": "CCN",
                    "cascade_search": {
                        "n_results": 1,
                        "failure_categories": ["StockDeadEnd"],
                        "result_programs": [
                            {"rank": 1, "exact_reaction_hit_count": 0, "gt_reactant_hit_count": 0}
                        ],
                    },
                    "recovery": {},
                },
                {
                    "target_smiles": "CCC",
                    "cascade_search": {
                        "n_results": 2,
                        "failure_categories": [],
                        "result_programs": [
                            {"rank": 1, "exact_reaction_hit_count": 0, "gt_reactant_hit_count": 0},
                            {"rank": 2, "exact_reaction_hit_count": 0, "gt_reactant_hit_count": 1},
                        ],
                    },
                    "recovery": {"result_gt_reactant_in_pool": True},
                },
                {
                    "target_smiles": "CCCC",
                    "cascade_search": {"n_results": 1, "failure_categories": [], "result_programs": []},
                    "recovery": {},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proposal_path = root / "proposal.json"
            route_path = root / "route.json"
            output = root / "fusion.json"
            markdown = root / "fusion.md"
            proposal_path.write_text(json.dumps(proposal_audit), encoding="utf-8")
            route_path.write_text(json.dumps(route_run), encoding="utf-8")

            payload = diagnose_search_fusion(
                proposal_audit=proposal_path,
                route_run=route_path,
                output_json=output,
                output_md=markdown,
            )
            rendered = render_markdown(payload)
            output_exists = output.exists()
            markdown_exists = markdown.exists()

        statuses = [row["fusion_status"] for row in payload["targets"]]
        self.assertEqual(
            statuses,
            [
                "exact_candidate_top_result",
                "exact_candidate_lost_after_direct_audit",
                "reactant_candidate_retained_below_top",
                "no_direct_candidate_hit",
            ],
        )
        self.assertEqual(payload["summary"]["direct_exact_hit"], 2)
        self.assertEqual(payload["summary"]["route_any_exact_hit"], 1)
        self.assertEqual(payload["summary"]["direct_exact_retention_rate_any_result"], 0.5)
        self.assertEqual(payload["decision"]["status"], "score_fusion_or_budget_blocks_exact_candidates")
        self.assertTrue(output_exists)
        self.assertTrue(markdown_exists)
        self.assertIn("Legal Corpus Search Fusion Diagnostic", rendered)


if __name__ == "__main__":
    unittest.main()
