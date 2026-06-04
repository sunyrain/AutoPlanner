import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_context_onmt_training_coverage import audit_training_coverage, render_markdown


class AuditContextONMTTrainingCoverageTest(unittest.TestCase):
    def test_audits_exact_and_product_only_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            benchmark = root / "benchmark.json"
            output = root / "coverage.json"
            markdown = root / "coverage.md"
            _write_meta(
                corpus / "context.train.meta.jsonl",
                [
                    _row("ex1", product="CCO", reactants=["CC", "O"]),
                    _row("ex2", product="CCN", reactants=["CCC"]),
                    _row("ex3", product="c1ccccc1O", reactants=["c1ccccc1"]),
                ],
            )
            benchmark.write_text(
                json.dumps(
                    [
                        {"target_smiles": "CCO", "gt_route": [{"rxn_smiles": "O.CC>>CCO"}]},
                        {"target_smiles": "CCN", "gt_route": [{"rxn_smiles": "CC.N>>CCN"}]},
                        {"target_smiles": "CCCl", "gt_route": [{"rxn_smiles": "CC.Cl>>CCCl"}]},
                    ]
                ),
                encoding="utf-8",
            )

            payload = audit_training_coverage(
                benchmark_path=benchmark,
                corpus_dir=corpus,
                output_json=output,
                output_md=markdown,
                corpus_splits=("train",),
            )
            rendered = render_markdown(payload)
            self.assertTrue(output.exists())
            self.assertTrue(markdown.exists())

        labels = [row["target_coverage_label"] for row in payload["targets"]]
        self.assertEqual(labels[0], "exact_reaction_covered")
        self.assertEqual(labels[1], "exact_product_only")
        self.assertIn(labels[2], {"near_pair_covered", "near_product_only", "out_of_distribution"})
        self.assertEqual(payload["summary"]["targets_with_exact_reaction"], 1)
        self.assertEqual(payload["summary"]["targets_with_exact_product"], 2)
        self.assertIn("Context ONMT Training Coverage Audit", rendered)


def _write_meta(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(example_id: str, *, product: str, reactants: list[str]) -> dict:
    return {
        "source_example_id": example_id,
        "source_target_index": example_id,
        "route_index": 0,
        "step_index": 0,
        "split": "train",
        "product": product,
        "reactants": reactants,
    }


if __name__ == "__main__":
    unittest.main()
