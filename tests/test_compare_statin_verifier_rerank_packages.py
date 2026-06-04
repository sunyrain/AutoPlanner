import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_statin_verifier_rerank_packages import compare_statin_packages


class StatinVerifierRerankPackageMatrixTest(unittest.TestCase):
    def test_compares_multiple_packages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg_a = _package(root, "pkg_a", "target_a")
            pkg_b = _package(root, "pkg_b", "target_b")
            output = root / "matrix.json"
            out_dir = root / "out"
            markdown = root / "matrix.md"

            result = compare_statin_packages(
                package_dirs=[pkg_a, pkg_b],
                output=output,
                output_dir=out_dir,
                markdown=markdown,
            )

            self.assertEqual(result["summary"]["n_packages"], 2)
            self.assertEqual(result["summary"]["n_routes"], 4)
            self.assertEqual(
                result["summary"]["promotion_decision"]["rule_verifier"],
                "promote_as_default_metric_and_optional_gate",
            )
            self.assertEqual(
                result["summary"]["promotion_decision"]["learned_verifier"],
                "calibrated_no_extra_top1_effect_observed",
            )
            self.assertEqual(len(result["packages"]), 2)
            self.assertTrue(output.exists())
            self.assertTrue(markdown.exists())
            self.assertTrue((out_dir / "pkg_a" / "comparison.json").exists())
            self.assertTrue((out_dir / "pkg_b" / "comparison.md").exists())


def _package(root: Path, name: str, target: str) -> Path:
    pkg = root / name
    docs = pkg / "route_docs"
    docs.mkdir(parents=True)
    (docs / f"{target}_top2_routes.json").write_text(
        json.dumps(
            {
                "target": "CCCCO",
                "routes": [
                    _route("bad", "C", "CCCCO", risk_order=30),
                    _route("good", "CCCC", "CCCCO", risk_order=10),
                ],
            }
        ),
        encoding="utf-8",
    )
    return pkg


def _route(route_id: str, reactant: str, product: str, *, risk_order: int) -> dict:
    return {
        "id": route_id,
        "score": 0.5,
        "product_audit": {"risk_order": risk_order, "route_class": "triage_fragment"},
        "steps": [
            {
                "product": product,
                "main_reactant": reactant,
                "reaction_smiles": f"{reactant}>>{product}",
                "T": 30,
                "pH": 7,
                "solvent": "water",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
