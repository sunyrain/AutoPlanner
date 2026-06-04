import unittest

from cascade_planner.cascade_search.proposal_validity import (
    ProposalValidityConfig,
    canonicalize_reactant_side_for_output,
    filter_reactant_predictions,
    proposal_rejection_reason,
)


class ProposalValidityTest(unittest.TestCase):
    def test_filters_invalid_self_empty_and_duplicate_sides(self):
        report = filter_reactant_predictions(
            ["C1", "CCO", "", "O.CC", "CC.O", "CCC"],
            product_smiles="CCO",
        )

        self.assertEqual(report.kept, ["O.CC", "CCC"])
        self.assertEqual(
            [row["reason"] for row in report.rejected],
            [
                "invalid_reactant_molecule",
                "self_reaction",
                "empty_reactant_side",
                "duplicate_canonical_side",
            ],
        )

    def test_atom_count_ratio_is_opt_in(self):
        self.assertIsNone(proposal_rejection_reason("CCCCCCCCCCCC", product_smiles="CCO"))
        self.assertEqual(
            proposal_rejection_reason(
                "CCCCCCCCCCCC",
                product_smiles="CCO",
                config=ProposalValidityConfig(max_reactant_to_product_heavy_ratio=2.0),
            ),
            "reactant_atom_count_exceeds_ratio",
        )

    def test_canonicalizes_reactant_side_for_output(self):
        self.assertEqual(canonicalize_reactant_side_for_output("O.C(C)"), "CC.O")


if __name__ == "__main__":
    unittest.main()
