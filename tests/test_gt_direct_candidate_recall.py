import unittest

from cascade_planner.eval.gt_direct_candidate_recall import (
    _candidate_reactants,
    _iter_gt_steps,
    audit_step,
)


class _FakeRetroChimera:
    def predict(self, product_smiles: str, top_k: int = 10):
        return [
            {
                "main_reactant": "A",
                "aux_reactants": ["B"],
                "rxn_smiles": "A.B>>P",
                "source": "retrochimera",
            },
            {
                "main_reactant": "C",
                "rxn_smiles": "C>>P",
                "source": "retrochimera",
            },
        ][:top_k]


class _ValidFakeRetroChimera:
    def predict(self, product_smiles: str, top_k: int = 10, **_kwargs):
        return [
            {
                "main_reactant": "CC",
                "rxn_smiles": "CC>>CCO",
                "source": "retrochimera",
            }
        ][:top_k]


class GTDirectCandidateRecallTest(unittest.TestCase):
    def test_candidate_reactants_reads_rxn_smiles(self):
        self.assertEqual(
            _candidate_reactants({"rxn_smiles": "B.A>>P"}),
            {"A", "B"},
        )

    def test_iter_gt_steps_extracts_product_and_kind(self):
        rows = _iter_gt_steps([
            {
                "target_smiles": "P",
                "route_domain": "all_chemical",
                "gt_route": [
                    {
                        "rxn_smiles": "A.B>>P",
                        "ec_number": None,
                        "transformation": "amination",
                    }
                ],
            }
        ])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gt_product"], "P")
        self.assertEqual(rows[0]["step_kind"], "chemical")
        self.assertEqual(rows[0]["gt_type"], "amination")

    def test_audit_step_detects_exact_hit_from_live_style_candidate(self):
        gt_step = {
            "target_index": 0,
            "target_smiles": "P",
            "route_domain": "all_chemical",
            "doi": None,
            "cascade_id": None,
            "gt_step_index": 1,
            "gt_rxn": "A.B>>P",
            "gt_product": "P",
            "gt_reactants": {"A", "B"},
            "gt_type": "amination",
            "gt_ec": "",
            "gt_ec1": 0,
            "step_kind": "chemical",
        }

        row = audit_step({"retrochimera": _FakeRetroChimera()}, gt_step, top_k=2)

        self.assertTrue(row["direct_exact_hit"])
        self.assertTrue(row["direct_exact_reactant_set_hit"])
        self.assertEqual(row["best_exact_rank"], 1)

    def test_audit_step_can_use_route_tree_proposal_mode(self):
        gt_step = {
            "target_index": 0,
            "target_smiles": "CCO",
            "route_domain": "all_chemical",
            "doi": None,
            "cascade_id": None,
            "gt_step_index": 1,
            "gt_rxn": "CC>>CCO",
            "gt_product": "CCO",
            "gt_reactants": {"CC"},
            "gt_type": "reduction",
            "gt_ec": "",
            "gt_ec1": 0,
            "step_kind": "chemical",
        }

        row = audit_step(
            {"retrochimera": _ValidFakeRetroChimera()},
            gt_step,
            top_k=2,
            proposal_mode="route_tree",
        )

        self.assertTrue(row["direct_exact_hit"])
        self.assertEqual(row["candidate_source_counts"], {"retrochimera": 1})


if __name__ == "__main__":
    unittest.main()
