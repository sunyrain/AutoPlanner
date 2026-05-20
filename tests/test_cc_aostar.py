import unittest
from unittest.mock import patch

from cascade_planner.cascadeboard.cc_aostar import _candidate_reactant_set, _merge_candidate_lists, plan_with_cc_aostar
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton


class _FakeRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CC=O":
            return [{
                "main_reactant": "CCO",
                "rxn_smiles": "CCO>>CC=O",
                "type": "reduction",
                "score": 0.9,
                "source": "fake_chem",
            }]
        if product_smiles == "CCO":
            return [{
                "main_reactant": "CC",
                "rxn_smiles": "CC>>CCO",
                "type": "oxidation",
                "score": 0.8,
                "source": "fake_chem",
            }]
        return []


class _AnchorRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CC=O":
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
            ]
        return []


class _WaypointRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "P":
            return [
                {
                    "main_reactant": "X",
                    "rxn_smiles": "X>>P",
                    "type": "reduction",
                    "score": 10.0,
                    "source": "fake_chem",
                },
                {
                    "main_reactant": "I",
                    "rxn_smiles": "I>>P",
                    "type": "reduction",
                    "score": 0.1,
                    "source": "fake_chem",
                },
            ]
        if product_smiles == "I":
            return [{
                "main_reactant": "S",
                "rxn_smiles": "S>>I",
                "type": "oxidation",
                "score": 0.5,
                "source": "fake_chem",
            }]
        if product_smiles == "X":
            return [{
                "main_reactant": "Z",
                "rxn_smiles": "Z>>X",
                "type": "oxidation",
                "score": 9.0,
                "source": "fake_chem",
            }]
        return []


class _ControlledNodeRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "P":
            return [
                {
                    "main_reactant": "X",
                    "rxn_smiles": "X>>P",
                    "type": "reduction",
                    "score": 10.0,
                    "source": "fake_chem",
                },
                {
                    "main_reactant": "I",
                    "rxn_smiles": "I>>P",
                    "type": "reduction",
                    "score": 0.1,
                    "source": "fake_chem",
                },
            ]
        if product_smiles in {"I", "X"}:
            return [{
                "main_reactant": "S",
                "rxn_smiles": f"S>>{product_smiles}",
                "type": "oxidation",
                "score": 1.0,
                "source": "fake_chem",
            }]
        return []


class _LoopRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "P":
            return [
                {
                    "main_reactant": "P",
                    "rxn_smiles": "P>>P",
                    "type": "reduction",
                    "score": 10.0,
                    "source": "fake_chem",
                },
                {
                    "main_reactant": "S",
                    "rxn_smiles": "S>>P",
                    "type": "reduction",
                    "score": 0.1,
                    "source": "fake_chem",
                },
            ]
        return []


class CCAStarTest(unittest.TestCase):
    def test_cc_aostar_fills_linear_cascade_and_preserves_candidates(self):
        skeleton = RouteSkeleton(
            n_steps=2,
            types=["reduction", "oxidation"],
            ec1s=[0, 0],
            Ts=[30.0, 31.0],
            pHs=[7.0, 7.1],
            compatibility="empirically_compatible",
            operation_mode="one_pot_sequential_addition",
        )

        results = plan_with_cc_aostar(
            target="CC=O",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _FakeRetro()},
            n_results=1,
            candidate_budget=3,
            expansion_budget=10,
            stock_checker=lambda smi: smi == "CC",
        )

        self.assertEqual(len(results), 1)
        board = results[0].board
        self.assertEqual(board.slots[0].main_reactant, "CCO")
        self.assertEqual(board.slots[1].main_reactant, "CC")
        self.assertEqual(board.slots[0].reaction_type, "reduction")
        self.assertEqual(board.slots[1].reaction_type, "oxidation")
        self.assertTrue(board.slots[0].candidates)
        self.assertEqual(results[0].constraint_report["search_mode"], "cc_aostar")

    def test_cc_aostar_respects_fixed_main_reactant_anchor(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
        )

        results = plan_with_cc_aostar(
            target="CC=O",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _AnchorRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=5,
            constraints={"fixed_steps": [{"index": 0, "values": {"main_reactant": "CCO"}}]},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCO")

    def test_cc_aostar_respects_starting_material_terminal_constraint(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
        )

        results = plan_with_cc_aostar(
            target="CC=O",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _AnchorRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=5,
            constraints={"starting_material": "CCO"},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCO")

    def test_cc_aostar_requires_known_intermediate_waypoint(self):
        skeleton = RouteSkeleton(
            n_steps=2,
            types=["reduction", "oxidation"],
            ec1s=[0, 0],
            Ts=[30.0, 31.0],
            pHs=[7.0, 7.1],
        )

        results = plan_with_cc_aostar(
            target="P",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _WaypointRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=10,
            constraints={"required_intermediates": ["I"], "starting_material": "S"},
        )

        self.assertEqual(len(results), 1)
        board = results[0].board
        self.assertEqual(board.slots[0].main_reactant, "I")
        self.assertEqual(board.slots[1].main_reactant, "S")

    def test_cc_aostar_respects_fixed_product_node_anchor(self):
        skeleton = RouteSkeleton(
            n_steps=2,
            types=["reduction", "oxidation"],
            ec1s=[0, 0],
            Ts=[30.0, 31.0],
            pHs=[7.0, 7.1],
        )

        results = plan_with_cc_aostar(
            target="P",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _ControlledNodeRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=10,
            constraints={
                "starting_material": "S",
                "fixed_steps": [{"index": 1, "values": {"product": "I"}}],
            },
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "I")
        self.assertEqual(results[0].board.slots[1].product, "I")

    def test_cc_aostar_rejects_forbidden_intermediate_node(self):
        skeleton = RouteSkeleton(
            n_steps=2,
            types=["reduction", "oxidation"],
            ec1s=[0, 0],
            Ts=[30.0, 31.0],
            pHs=[7.0, 7.1],
        )

        results = plan_with_cc_aostar(
            target="P",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _ControlledNodeRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=10,
            constraints={"starting_material": "S", "forbidden_intermediates": ["X"]},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "I")
        self.assertNotEqual(results[0].board.slots[0].main_reactant, "X")

    def test_cc_aostar_prunes_self_loop_candidate(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["reduction"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
        )

        results = plan_with_cc_aostar(
            target="P",
            skeletons=[skeleton],
            retro_engine={"retrochimera": _LoopRetro()},
            n_results=1,
            candidate_budget=5,
            expansion_budget=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "S")

    def test_cc_aostar_uses_skeleton_as_soft_prior_with_chemical_fallback(self):
        skeleton = RouteSkeleton(
            n_steps=1,
            types=["oxidation"],
            ec1s=[1],
            Ts=[30.0],
            pHs=[7.0],
        )

        def fake_candidates(retro_engine, product_smiles, ec1=0, skel_type="", top_k=10):
            if ec1 > 0:
                return [
                    {
                        "main_reactant": f"WRONG{i}",
                        "rxn_smiles": f"WRONG{i}>>CC=O",
                        "type": "oxidation",
                        "ec": "1.1.1.1",
                        "score": 10.0 - i,
                        "source": "fake_enzyme",
                    }
                    for i in range(top_k)
                ]
            return [{
                "main_reactant": "CCO",
                "rxn_smiles": "CCO>>CC=O",
                "type": "reduction",
                "score": 0.1,
                "source": "fake_chem",
            }]

        with patch("cascade_planner.cascadeboard.cc_aostar._candidates_for_skeleton_slot", side_effect=fake_candidates):
            results = plan_with_cc_aostar(
                target="CC=O",
                skeletons=[skeleton],
                retro_engine={},
                n_results=1,
                candidate_budget=5,
                expansion_budget=5,
                constraints={"starting_material": "CCO"},
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCO")
        self.assertEqual(results[0].board.slots[0].source, "fake_chem")

    def test_merge_candidate_lists_preserves_enzymatic_source_diversity(self):
        primary = [
            {
                "main_reactant": f"R{i}",
                "rxn_smiles": f"R{i}>>P",
                "source": "retrochimera",
                "score": 10 - i,
            }
            for i in range(5)
        ]
        fallback = [{
            "main_reactant": "E",
            "rxn_smiles": "E>>P",
            "source": "v3_retrieval",
            "score": 0.1,
        }]

        merged = _merge_candidate_lists(
            primary,
            fallback,
            top_k=2,
            source_priority=("v3_retrieval", "enzyformer", "retrochimera"),
        )

        self.assertEqual([cand["source"] for cand in merged], ["v3_retrieval", "retrochimera"])

    def test_candidate_reactant_set_infers_missing_aux_from_reaction_smiles(self):
        candidate = {
            "main_reactant": "CCO",
            "aux_reactants": [],
            "rxn_smiles": "CCO.N>>CCN",
        }

        self.assertEqual(_candidate_reactant_set(candidate), {"CCO", "N"})


if __name__ == "__main__":
    unittest.main()
