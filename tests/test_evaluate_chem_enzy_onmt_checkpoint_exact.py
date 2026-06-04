import unittest

from scripts.evaluate_chem_enzy_onmt_checkpoint_exact import (
    _align_kept_predictions_with_scores,
    _product_smiles_from_source,
)


class EvaluateChemEnzyONMTCheckpointExactTest(unittest.TestCase):
    def test_aligns_filtered_predictions_with_original_scores(self):
        predictions = ["C1", "O.CC", "CC.O", "CCC"]
        scores = [-0.1, -0.2, -0.3, -0.4]

        kept, kept_scores = _align_kept_predictions_with_scores(predictions, scores, ["O.CC", "CCC"])

        self.assertEqual(kept, ["O.CC", "CCC"])
        self.assertEqual(kept_scores, [-0.2, -0.4])

    def test_parses_product_smiles_from_context_source(self):
        self.assertEqual(_product_smiles_from_source("<target> C C O <product> C C O"), "CCO")
        self.assertEqual(_product_smiles_from_source("<target> C C O <product> C C O <candidate> C C"), "CCO")
        self.assertEqual(_product_smiles_from_source("C C O"), "CCO")


if __name__ == "__main__":
    unittest.main()
