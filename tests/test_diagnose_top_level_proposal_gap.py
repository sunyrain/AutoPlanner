import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_top_level_proposal_gap import diagnose_gap, render_markdown


class DiagnoseTopLevelProposalGapTest(unittest.TestCase):
    def test_classifies_corpus_miss_generator_miss_and_partial_hit(self):
        coverage = {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "target_coverage_label": "exact_reaction_covered",
                    "target_step_reactions": [
                        {
                            "exact_reaction_in_corpus": True,
                            "exact_product_in_corpus": True,
                            "exact_reactant_side_any_product": True,
                            "nearest_pair": {"combined_similarity": 1.0},
                            "nearest_product": {"product_similarity": 1.0},
                        }
                    ],
                },
                {
                    "target_smiles": "CCN",
                    "target_coverage_label": "out_of_distribution",
                    "target_step_reactions": [
                        {
                            "exact_reaction_in_corpus": False,
                            "exact_product_in_corpus": False,
                            "exact_reactant_side_any_product": False,
                            "nearest_pair": {"combined_similarity": 0.2},
                            "nearest_product": {"product_similarity": 0.3},
                        }
                    ],
                },
                {
                    "target_smiles": "CCC",
                    "target_coverage_label": "near_pair_covered",
                    "target_step_reactions": [
                        {
                            "exact_reaction_in_corpus": False,
                            "exact_product_in_corpus": False,
                            "exact_reactant_side_any_product": False,
                            "nearest_pair": {"combined_similarity": 0.8},
                            "nearest_product": {"product_similarity": 0.9},
                        }
                    ],
                },
            ]
        }
        proposals = {
            "targets": [
                {
                    "target_smiles": "CCO",
                    "returned": 2,
                    "exact_gt_reaction_hit": False,
                    "target_step_gt_reaction_hit": False,
                    "gt_reactant_hit": False,
                },
                {
                    "target_smiles": "CCN",
                    "returned": 2,
                    "exact_gt_reaction_hit": False,
                    "target_step_gt_reaction_hit": False,
                    "gt_reactant_hit": False,
                },
                {
                    "target_smiles": "CCC",
                    "returned": 2,
                    "exact_gt_reaction_hit": False,
                    "target_step_gt_reaction_hit": False,
                    "gt_reactant_hit": True,
                    "gt_reactant_best_rank": 2,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            coverage_path = root / "coverage.json"
            proposal_path = root / "proposal.json"
            output = root / "gap.json"
            markdown = root / "gap.md"
            coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
            proposal_path.write_text(json.dumps(proposals), encoding="utf-8")

            payload = diagnose_gap(
                coverage_audit=coverage_path,
                proposal_audit=proposal_path,
                output_json=output,
                output_md=markdown,
            )
            rendered = render_markdown(payload)
            output_exists = output.exists()
            markdown_exists = markdown.exists()

        statuses = [row["gap_status"] for row in payload["targets"]]
        self.assertEqual(statuses, [
            "generator_missed_exact_corpus_reaction",
            "target_domain_gap",
            "proposal_partial_reactant_hit",
        ])
        self.assertEqual(payload["decision"]["status"], "partial_candidate_pool_gain")
        self.assertEqual(payload["summary"]["proposal_reactant_hit"], 1)
        self.assertTrue(output_exists)
        self.assertTrue(markdown_exists)
        self.assertIn("Top-level Proposal Gap Diagnostic", rendered)


if __name__ == "__main__":
    unittest.main()
