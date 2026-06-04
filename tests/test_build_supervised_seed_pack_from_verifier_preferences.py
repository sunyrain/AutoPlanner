import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_supervised_seed_pack_from_verifier_preferences import build_seed_pack


class BuildSupervisedSeedPackFromVerifierPreferencesTest(unittest.TestCase):
    def test_extracts_unique_chosen_routes_and_never_emits_rejected_as_positive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prefs = root / "prefs.jsonl"
            output = root / "chosen_pack.json"
            markdown = root / "chosen_pack.md"
            rows = [
                _pair("pref_1", chosen_id="seed_a", rejected_id="neg_a", reason="atom_balance_violation"),
                _pair("pref_2", chosen_id="seed_a", rejected_id="neg_b", reason="ph_conflict"),
                _pair("pref_3", chosen_id="seed_b", rejected_id="neg_c", reason="solvent_conflict", source_route_index="1"),
            ]
            prefs.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = build_seed_pack(preference_jsonl=prefs, output=output)
            from scripts.build_supervised_seed_pack_from_verifier_preferences import _write_markdown

            _write_markdown(result, markdown)

            examples = result["examples"]
            self.assertEqual(result["summary"]["source_pairs_scanned"], 3)
            self.assertEqual(result["summary"]["n_examples"], 2)
            self.assertEqual(result["summary"]["skipped"]["duplicate_chosen_route"], 1)
            self.assertEqual([row["example_id"] for row in examples], ["seed_a", "seed_b"])
            self.assertTrue(all(row["label"] == 1 for row in examples))
            self.assertTrue(all(row["metadata"]["rejected_side_used"] is False for row in examples))
            self.assertFalse(any(row["example_id"].startswith("neg_") for row in examples))
            self.assertEqual(result["summary"]["split_counts"]["train"], 2)
            self.assertIn("atom_balance_violation", result["summary"]["rejected_reason_counts_observed_not_used_as_positive"])
            self.assertTrue(output.exists())
            self.assertTrue(markdown.exists())


def _pair(pair_id: str, *, chosen_id: str, rejected_id: str, reason: str, source_route_index: str = "0") -> dict:
    chosen = _cascade("CCO", "CC", split="train")
    rejected = _cascade("CCCCO", "C", split="train")
    return {
        "pair_id": pair_id,
        "schema_version": "cascade_verifier_preference_pair.v1",
        "target_smiles": "CCO",
        "source_path": "routes.json",
        "source_target_index": "0",
        "source_route_index": source_route_index,
        "chosen_example_id": chosen_id,
        "rejected_example_id": rejected_id,
        "chosen_cascade": chosen,
        "rejected_cascade": rejected,
        "preference_source": "verifier_perturbation",
        "rejected_expected_failure_reasons": [reason],
    }


def _cascade(product: str, reactant: str, *, split: str) -> dict:
    return {
        "metadata": {"split": split, "route_domain": "test", "quality_tier": "gold"},
        "stage_partition": ["stage_1"],
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
