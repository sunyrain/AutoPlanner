import tempfile
import unittest
from pathlib import Path

from scripts.audit_legal_corpus_top_level_proposals import audit_legal_corpus_top_level_proposals


class AuditLegalCorpusTopLevelProposalsTest(unittest.TestCase):
    def test_audit_uses_legal_corpus_provider_and_counts_hits(self):
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

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(benchmark, encoding="utf-8")
            corpus = root / "meta.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        '{"source":"toy","source_row_id":1,"product":"CCO","reactants":["CC","O"],"reaction_smiles":"CC.O>>CCO"}',
                        '{"source":"toy","source_row_id":2,"product":"CCN","reactants":["CCC"],"reaction_smiles":"CCC>>CCN"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "audit.json"
            markdown = root / "audit.md"
            cache = root / "index.pkl"

            payload = audit_legal_corpus_top_level_proposals(
                benchmark_path=benchmark_path,
                corpus_paths=[corpus],
                output_json=output,
                output_md=markdown,
                topk=1,
                index_cache_path=cache,
            )
            self.assertTrue(output.exists())
            self.assertTrue(markdown.exists())
            self.assertTrue(cache.exists())

        self.assertEqual(payload["schema_version"], "legal_corpus_top_level_proposal_audit.v1")
        self.assertEqual(payload["summary"]["n_targets"], 2)
        self.assertEqual(payload["summary"]["targets_with_proposals"], 2)
        self.assertEqual(payload["summary"]["exact_gt_reaction_hit"], 1)
        self.assertEqual(payload["summary"]["target_step_gt_reaction_hit"], 1)
        self.assertEqual(payload["summary"]["gt_reactant_hit"], 1)
        self.assertEqual(payload["settings"]["provider"], "legal_corpus")
        self.assertEqual(payload["settings"]["index_cache_path"], str(cache))


if __name__ == "__main__":
    unittest.main()
