import unittest

from scripts.benchmark_chem_enzy_native_bionav import score_prediction_row, summarize_scored_rows


class BenchmarkChemEnzyNativeBioNavTest(unittest.TestCase):
    def test_scores_canonical_reactant_side_exact_recall(self):
        row = score_prediction_row(
            idx=0,
            product="CCO",
            target_reactants="O.CC",
            predictions=["CC.O", "CCC"],
            scores=[0.9, 0.1],
            metadata={"ec": "1.1.1.1", "source": "unit"},
            topk=2,
        )

        self.assertTrue(row["top1_exact"])
        self.assertTrue(row["top2_exact"])
        self.assertEqual(row["ec1"], "1")

    def test_summarizes_overall_and_ec1_metrics(self):
        rows = [
            score_prediction_row(
                idx=0,
                product="CCO",
                target_reactants="O.CC",
                predictions=["CC.O"],
                scores=[0.9],
                metadata={"ec": "1.1.1.1"},
                topk=2,
            ),
            score_prediction_row(
                idx=1,
                product="CCN",
                target_reactants="N.CC",
                predictions=["CCC", "CC.N"],
                scores=[0.6, 0.4],
                metadata={"ec": "2.7.1.1"},
                topk=2,
            ),
        ]

        summary = summarize_scored_rows(rows, topk=2, elapsed_s=4.0)

        self.assertEqual(summary["n_examples"], 2)
        self.assertEqual(summary["top1_exact"], 1)
        self.assertEqual(summary["top2_exact"], 2)
        self.assertEqual(summary["top1_rate"], 0.5)
        self.assertEqual(summary["top2_rate"], 1.0)
        self.assertEqual(summary["ec1"]["1"]["top1_rate"], 1.0)
        self.assertEqual(summary["ec1"]["2"]["top2_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
