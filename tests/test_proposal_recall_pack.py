import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_proposal_recall_pack import (
    build_proposal_recall_pack,
    label_candidate_pool_recall,
    validate_proposal_recall_row,
)


class ProposalRecallPackTest(unittest.TestCase):
    def test_label_generation_for_chemical_step(self):
        labels = label_candidate_pool_recall(
            product="CCO",
            gt_rxn="CC.O>>CCO",
            gt_reactants=["CC", "O"],
            candidate_pool=[
                {"main_reactant": "CC", "aux_reactants": ["O"], "rxn_smiles": "CC.O>>CCO"},
                {"main_reactant": "C", "aux_reactants": [], "rxn_smiles": "C>>CCO"},
            ],
        )

        self.assertEqual(labels["positive_candidate_indices"], [0])
        self.assertTrue(labels["candidate_exact_reaction_in_pool"])
        self.assertTrue(labels["candidate_gt_reactant_in_pool"])

    def test_label_generation_for_enzymatic_step_with_reactant_hit(self):
        labels = label_candidate_pool_recall(
            product="CC=O",
            gt_rxn="CCO>>CC=O",
            gt_reactants=["CCO"],
            candidate_pool=[
                {"main_reactant": "CCN", "rxn_smiles": "CCN>>CC=O", "source": "enzyformer"},
                {"main_reactant": "CCO", "rxn_smiles": "CCO>>CC=O", "source": "retrorules", "ec": "1.1.1.1"},
            ],
        )

        self.assertEqual(labels["positive_candidate_indices"], [1])
        self.assertEqual(labels["gt_reactant_indices"], [1])
        self.assertTrue(labels["candidate_gt_reactant_in_pool"])

    def test_build_pack_outputs_required_schema_and_source_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            steps = root / "steps.jsonl"
            pools = root / "pools.jsonl"
            steps.write_text(
                json.dumps(
                    {
                        "step_id": "s1",
                        "product": "CCO",
                        "reactants": ["CC", "O"],
                        "reaction_smiles": "CC.O>>CCO",
                        "reaction_type": "hydrolysis",
                        "source": "uspto50k",
                        "candidate": {"T": 25.0, "pH": 7.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pools.write_text(
                json.dumps(
                    {
                        "external_step_id": "s1",
                        "product": "CCO",
                        "external_source": "uspto50k",
                        "candidates": [
                            {
                                "rank": 1,
                                "label": 1.0,
                                "label_type": "external_positive",
                                "candidate": {
                                    "main_reactant": "CC",
                                    "aux_reactants": ["O"],
                                    "rxn_smiles": "CC.O>>CCO",
                                    "source": "uspto50k",
                                },
                            },
                            {
                                "rank": 2,
                                "label": 0.0,
                                "label_type": "external_hard_negative",
                                "candidate": {
                                    "main_reactant": "C",
                                    "rxn_smiles": "C>>CCO",
                                    "source": "uspto50k",
                                },
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_proposal_recall_pack(
                external_step_pairs=steps,
                candidate_pools=pools,
                output_dir=root / "out",
            )
            row = json.loads((root / "out" / "proposal_recall_pack.jsonl").read_text(encoding="utf-8").splitlines()[0])

        validate_proposal_recall_row(row)
        self.assertEqual(manifest["counts"]["rows"], 1)
        self.assertEqual(row["positive_candidate_indices"], [0])
        self.assertEqual(row["hard_negative_type"], "source_type_ec_hard_negative")
        self.assertEqual(manifest["audit"]["chemical"]["candidate_exact_reaction_in_pool"], 1.0)


if __name__ == "__main__":
    unittest.main()
