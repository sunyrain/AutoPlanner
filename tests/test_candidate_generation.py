import unittest

from cascade_planner.cascadeboard.skeleton_planner import (
    _candidates_for_skeleton_slot,
    _dedupe_candidates,
    fill_route_from_skeleton,
    _prefer_candidates,
    _source_diverse_candidates,
    RouteSkeleton,
)


class _TerminalRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles != "CC=O":
            return []
        return [
            {
                "main_reactant": "CCC",
                "rxn_smiles": "CCC>>CC=O",
                "type": "reduction",
                "score": 10.0,
                "source": "fake_chem",
            },
            {
                "main_reactant": "CCO",
                "rxn_smiles": "CCO>>CC=O",
                "type": "reduction",
                "score": 0.1,
                "source": "fake_chem",
            },
        ][:top_k]


class _EnzymeOnly:
    def predict(self, product_smiles: str, top_k: int = 10, ec_token: str | None = None):
        return [{
            "main_reactant": "NADH",
            "rxn_smiles": "NADH>>CC=O",
            "type": "oxidation",
            "ec": f"{ec_token or '1'}.1.1.1",
            "score": 0.8,
            "source": "enzyformer",
        }]


class _ChemicalFallback:
    def predict(self, product_smiles: str, top_k: int = 10):
        return [{
            "main_reactant": "CCN",
            "rxn_smiles": "CCN>>CC=O",
            "type": "reduction",
            "score": 0.4,
            "source": "retrochimera",
        }]


class CandidateGenerationTest(unittest.TestCase):
    def test_source_diverse_candidates_round_robin_across_sources(self):
        cands = [
            {"rxn_smiles": "A>>P", "source": "enzyformer"},
            {"rxn_smiles": "B>>P", "source": "retrochimera"},
            {"rxn_smiles": "C>>P", "source": "v3_retrieval"},
            {"rxn_smiles": "D>>P", "source": "enzyformer"},
            {"rxn_smiles": "E>>P", "source": "retrochimera"},
        ]

        out = _source_diverse_candidates(
            cands,
            top_k=4,
            source_priority=("v3_retrieval", "enzyformer", "retrochimera"),
        )

        self.assertEqual([c["rxn_smiles"] for c in out], ["C>>P", "A>>P", "B>>P", "D>>P"])

    def test_prefer_candidates_keeps_fallbacks_after_matches(self):
        cands = [
            {"rxn_smiles": "A>>P", "source": "retrochimera"},
            {"rxn_smiles": "B>>P", "source": "v3_retrieval"},
            {"rxn_smiles": "C>>P", "source": "enzyformer"},
        ]

        out = _prefer_candidates(
            cands,
            lambda cand: cand["source"] == "v3_retrieval",
            top_k=3,
        )

        self.assertEqual([c["rxn_smiles"] for c in out], ["B>>P", "A>>P", "C>>P"])

    def test_dedupe_candidates_removes_duplicate_reactions(self):
        cands = [
            {"rxn_smiles": "A>>P", "source": "retrochimera"},
            {"rxn_smiles": "A>>P", "source": "v3_retrieval"},
            {"rxn_smiles": "B>>P", "source": "enzyformer"},
        ]

        out = _dedupe_candidates(cands)

        self.assertEqual([c["rxn_smiles"] for c in out], ["A>>P", "B>>P"])

    def test_fill_route_respects_starting_material_anchor_in_beam(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
        )

        boards = fill_route_from_skeleton(
            skeleton,
            "CC=O",
            retro_engine={"retrochimera": _TerminalRetro()},
            starting_material="CCO",
            n_routes=2,
        )

        self.assertTrue(boards)
        self.assertEqual(boards[0].slots[0].main_reactant, "CCO")

    def test_skeleton_slot_candidates_keep_chemical_fallback_for_enzymatic_prior(self):
        cands = _candidates_for_skeleton_slot(
            {
                "enzyformer": _EnzymeOnly(),
                "retrochimera": _ChemicalFallback(),
            },
            "CC=O",
            ec1=1,
            skel_type="oxidation",
            top_k=4,
        )

        sources = {cand["source"] for cand in cands}
        self.assertIn("enzyformer", sources)
        self.assertIn("retrochimera", sources)


if __name__ == "__main__":
    unittest.main()
