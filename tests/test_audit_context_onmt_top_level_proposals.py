import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascade_search import CascadeAction, CascadeActionType, StepAnnotation
from scripts.audit_context_onmt_top_level_proposals import audit_top_level_proposals, render_markdown


class AuditContextONMTTopLevelProposalsTest(unittest.TestCase):
    def test_audit_counts_exact_and_reactant_hits(self):
        benchmark = """
[
  {
    "target_smiles": "CCO",
    "route_domain": "all_chemical",
    "depth": 1,
    "gt_route": [
      {"rxn_smiles": "CC.O>>CCO"}
    ]
  },
  {
    "target_smiles": "CCN",
    "route_domain": "all_chemical",
    "depth": 1,
    "gt_route": [
      {"rxn_smiles": "CC.N>>CCN"}
    ]
  }
]
"""

        class FakeProvider:
            def propose(self, request):
                product = request.leaf_smiles
                if product == "CCO":
                    rxn = "O.CC>>CCO"
                    reactants = ["O", "CC"]
                else:
                    rxn = "CCC>>CCN"
                    reactants = ["CCC"]
                return [
                    CascadeAction(
                        CascadeActionType.RETROSYNTHETIC_STEP,
                        target_leaf=product,
                        source="fake_context",
                        step=StepAnnotation(
                            product_smiles=product,
                            reactant_smiles=reactants,
                            rxn_smiles=rxn,
                            source_model="fake_context",
                            score=0.5,
                            raw_metadata={"preference_score": 0.8, "preference_rank": 1},
                        ),
                    )
                ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(benchmark, encoding="utf-8")
            output = root / "audit.json"
            markdown = root / "audit.md"

            payload = audit_top_level_proposals(
                benchmark_path=benchmark_path,
                model_path=Path("fake.pt"),
                output_json=output,
                output_md=markdown,
                provider=FakeProvider(),
            )
            rendered = render_markdown(payload)

        self.assertEqual(payload["summary"]["n_targets"], 2)
        self.assertEqual(payload["summary"]["targets_with_proposals"], 2)
        self.assertEqual(payload["summary"]["exact_gt_reaction_hit"], 1)
        self.assertEqual(payload["summary"]["target_step_gt_reaction_hit"], 1)
        self.assertEqual(payload["summary"]["gt_reactant_hit"], 1)
        self.assertEqual(payload["targets"][0]["proposals"][0]["preference_score"], 0.8)
        self.assertEqual(payload["targets"][0]["proposals"][0]["preference_rank"], 1)
        self.assertEqual(payload["decision"]["status"], "proposal_hits_exist_check_search_fusion")
        self.assertIn("Context ONMT Top-level Proposal Audit", rendered)


if __name__ == "__main__":
    unittest.main()
