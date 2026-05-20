import unittest

from cascade_planner.cascadeboard.route_export import route_metrics
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton
from cascade_planner.cascadeboard.stock_andor import _merge_candidates, plan_stock_closed_andor


class _BranchingRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCC":
            return [{
                "main_reactant": "CCCC",
                "aux_reactants": ["CO"],
                "rxn_smiles": "CCCC.CO>>CCCCCC",
                "type": "coupling",
                "score": 0.9,
                "source": "fake_chem",
            }]
        if product_smiles == "CCCC":
            return [{
                "main_reactant": "CC",
                "aux_reactants": ["CC"],
                "rxn_smiles": "CC.CC>>CCCC",
                "type": "coupling",
                "score": 0.8,
                "source": "fake_chem",
            }]
        return []


class _SkeletonFallbackRetro(_BranchingRetro):
    def predict(self, product_smiles: str, top_k: int = 10):
        rows = super().predict(product_smiles, top_k=top_k)
        if product_smiles == "CCCCCC":
            return [
                {
                    "main_reactant": "CCCCCC",
                    "rxn_smiles": "CCCCCC>>CCCCCC",
                    "type": "oxidation",
                    "score": 10.0,
                    "source": "bad_skeleton_match",
                },
                *rows,
            ]
        return rows


class _MissingAuxRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCN":
            return [{
                "main_reactant": "CCO",
                "aux_reactants": [],
                "rxn_smiles": "CCO.N>>CCN",
                "type": "amination",
                "score": 1.0,
                "source": "fake_chem",
            }]
        return []


class _AntiGrowthRetro:
    def predict(self, product_smiles: str, top_k: int = 10):
        if product_smiles == "CCCCCCCCCCCC":
            return [
                {
                    "main_reactant": "CCCCCCCCCCCCCCCCCC",
                    "rxn_smiles": "CCCCCCCCCCCCCCCCCC>>CCCCCCCCCCCC",
                    "type": "protection",
                    "score": 4.0,
                    "source": "fake_chem",
                },
                {
                    "main_reactant": "CCCCCC",
                    "aux_reactants": ["CCCCCC"],
                    "rxn_smiles": "CCCCCC.CCCCCC>>CCCCCCCCCCCC",
                    "type": "coupling",
                    "score": 0.1,
                    "source": "fake_chem",
                },
            ]
        return []


class StockAndOrTest(unittest.TestCase):
    def test_stock_closed_andor_expands_auxiliary_leaves_to_stock(self):
        results = plan_stock_closed_andor(
            target="CCCCCC",
            retro_engine={"retrochimera": _BranchingRetro()},
            stock_checker=lambda smi: smi in {"CC", "CO"},
            max_depth=3,
            n_results=1,
            branch_factor=3,
            expansion_budget=20,
        )

        self.assertEqual(len(results), 1)
        route = results[0]
        self.assertEqual(route.constraint_report["search_mode"], "stock_closed_andor")
        self.assertEqual(route.board.n_steps, 2)
        metrics = route_metrics(route.board, stock_checker=lambda smi: smi in {"CC", "CO"})
        self.assertTrue(metrics["strict_stock_solve"])
        self.assertTrue(metrics["route_solved"])
        self.assertEqual(set(metrics["terminal_reactants"]), {"CC", "CO"})
        self.assertNotIn("CCCC", metrics["terminal_reactants"])

    def test_skeleton_is_soft_prior_not_required_for_stock_closed_search(self):
        misleading = RouteSkeleton(
            n_steps=1,
            types=["oxidation"],
            ec1s=[0],
            Ts=[30.0],
            pHs=[7.0],
        )

        results = plan_stock_closed_andor(
            target="CCCCCC",
            retro_engine={"retrochimera": _SkeletonFallbackRetro()},
            stock_checker=lambda smi: smi in {"CC", "CO"},
            max_depth=3,
            n_results=1,
            branch_factor=3,
            expansion_budget=20,
            skeletons=[misleading],
        )

        self.assertEqual(len(results), 1)
        metrics = route_metrics(results[0].board, stock_checker=lambda smi: smi in {"CC", "CO"})
        self.assertTrue(metrics["route_solved"])
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCCC")

    def test_stock_andor_uses_skeleton_conditions_when_candidates_lack_conditions(self):
        skeleton = RouteSkeleton(
            n_steps=2,
            types=["coupling", "coupling"],
            ec1s=[0, 0],
            Ts=[30.0, 31.0],
            pHs=[7.0, 7.1],
        )

        results = plan_stock_closed_andor(
            target="CCCCCC",
            retro_engine={"retrochimera": _BranchingRetro()},
            stock_checker=lambda smi: smi in {"CC", "CO"},
            max_depth=3,
            n_results=1,
            branch_factor=3,
            expansion_budget=20,
            skeletons=[skeleton],
        )

        self.assertEqual(len(results), 1)
        board = results[0].board
        self.assertEqual([slot.T for slot in board.slots], [30.0, 31.0])
        self.assertEqual([slot.pH for slot in board.slots], [7.0, 7.1])
        metrics = route_metrics(board, stock_checker=lambda smi: smi in {"CC", "CO"})
        self.assertTrue(metrics["condition"]["condition_window_success"])

    def test_stock_andor_infers_missing_aux_reactants_from_reaction_smiles(self):
        results = plan_stock_closed_andor(
            target="CCN",
            retro_engine={"retrochimera": _MissingAuxRetro()},
            stock_checker=lambda smi: smi in {"CCO"},
            max_depth=1,
            n_results=1,
            branch_factor=3,
            expansion_budget=5,
        )
        self.assertEqual(results, [])

        solved = plan_stock_closed_andor(
            target="CCN",
            retro_engine={"retrochimera": _MissingAuxRetro()},
            stock_checker=lambda smi: smi in {"CCO", "N"},
            max_depth=1,
            n_results=1,
            branch_factor=3,
            expansion_budget=5,
        )
        self.assertEqual(len(solved), 1)
        self.assertEqual(solved[0].board.slots[0].aux_reactants, ["N"])

    def test_stock_andor_penalizes_antiretrosynthetic_growth(self):
        results = plan_stock_closed_andor(
            target="CCCCCCCCCCCC",
            retro_engine={"retrochimera": _AntiGrowthRetro()},
            stock_checker=lambda smi: smi in {"CCCCCCCCCCCCCCCCCC", "CCCCCC"},
            max_depth=1,
            n_results=1,
            branch_factor=3,
            expansion_budget=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].board.slots[0].main_reactant, "CCCCCC")

    def test_merge_candidates_preserves_enzymatic_source_diversity(self):
        rows = [
            {
                "main_reactant": f"R{i}",
                "rxn_smiles": f"R{i}>>P",
                "source": "retrochimera",
                "score": 10 - i,
            }
            for i in range(5)
        ]
        rows.append({
            "main_reactant": "E",
            "rxn_smiles": "E>>P",
            "source": "v3_retrieval",
            "score": 0.1,
        })

        merged = _merge_candidates(
            rows,
            top_k=2,
            source_priority=("v3_retrieval", "enzyformer", "retrochimera"),
        )

        self.assertEqual([cand["source"] for cand in merged], ["v3_retrieval", "retrochimera"])


if __name__ == "__main__":
    unittest.main()
