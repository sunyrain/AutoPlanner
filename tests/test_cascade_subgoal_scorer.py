import unittest
from pathlib import Path

from cascade_planner.cascade_search import (
    CascadeProgramSearch,
    CascadeSearchConfig,
    CascadeSearchController,
    CascadeSubgoalEvidenceProvider,
    HeuristicCascadeValueModel,
    StaticProposalProvider,
    SubgoalHintActionScorer,
)
from cascade_planner.eval.train_cascade_subgoal_scorer import _audit_query_role, _row_from_audit_match


class CascadeSubgoalScorerTest(unittest.TestCase):
    def test_runtime_sources_map_to_training_query_roles(self):
        self.assertEqual(_audit_query_role(["target"]), "program_target")
        self.assertEqual(_audit_query_role(["target_fragment"]), "target_fragment")
        self.assertEqual(_audit_query_role(["route_step_product"]), "step_product")
        self.assertEqual(_audit_query_role(["route_leaf_or_reactant"]), "step_product")
        self.assertEqual(_audit_query_role(["route_leaf_fragment"]), "step_product_fragment")

    def test_audit_match_row_preserves_step_product_evidence(self):
        row = _row_from_audit_match(
            {"target_id": "demo"},
            {
                "subgoal_id": "sg1",
                "smiles": "CC[C@@H](O)C[C@@H](O)CC=O",
                "heavy_atoms": 10,
                "ring_count": 0,
                "hetero_atoms": 3,
                "sources": ["route_leaf_or_reactant"],
            },
            {
                "evidence_id": "ev1",
                "program_id": "p1",
                "role": "step_product",
                "smiles": "C=CC(CC=O)CCCC",
                "transformation_superclass": "C_C_coupling",
                "cascade_type": "chemoenzymatic",
                "quality_tier": "gold",
                "evidence_strength": "strong_process_evidence",
                "motif_similarity": 0.8,
            },
            rank=2,
        )

        self.assertEqual(row["query_role"], "step_product")
        self.assertEqual(row["evidence_role"], "step_product")
        self.assertEqual(row["evidence_transform"], "C_C_coupling")
        self.assertEqual(row["candidate_rank"], 2)
        self.assertAlmostEqual(row["similarity"], 0.8)

    def test_subgoal_provider_emits_evidence_action_not_retro_step(self):
        model = Path("results/shared/model_focus_20260518/subgoal_scorer_v1/cascade_subgoal_scorer.pkl")
        manifest = Path("results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
        if not model.exists() or not manifest.exists():
            self.skipTest("trained subgoal scorer artifacts are not present")
        provider = CascadeSubgoalEvidenceProvider(
            manifest,
            model,
            min_score=-1.0,
            min_similarity=0.30,
            evidence_candidates=12,
            max_hints_per_leaf=1,
        )

        actions = provider.propose("CC[C@@H](O)C[C@@H](O)CC=O", top_k=1)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "evidence_retrieval")
        self.assertIsNone(actions[0].step)
        self.assertEqual(actions[0].evidence_payload["contract"], "learned subgoal evidence hint; not a retrosynthetic reaction")

    def test_search_records_subgoal_hint_without_solving_route(self):
        model = Path("results/shared/model_focus_20260518/subgoal_scorer_v1/cascade_subgoal_scorer.pkl")
        manifest = Path("results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
        if not model.exists() or not manifest.exists():
            self.skipTest("trained subgoal scorer artifacts are not present")
        leaf = "CC[C@@H](O)C[C@@H](O)CC=O"
        provider = CascadeSubgoalEvidenceProvider(
            manifest,
            model,
            min_score=-1.0,
            min_similarity=0.30,
            evidence_candidates=12,
            max_hints_per_leaf=1,
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda _smi: False,
            config=CascadeSearchConfig(max_depth=1, expansion_budget=2, branch_factor=1, allow_repair_actions=False),
        )

        results = planner.search(leaf, n_results=1)

        self.assertTrue(results)
        self.assertFalse(results[0].solved)
        self.assertFalse(results[0].state.step_annotations)
        self.assertEqual(len(results[0].state.raw_metadata.get("cascade_subgoal_hints") or []), 1)

    def test_subgoal_hint_is_sidecar_for_real_retro_proposal(self):
        model = Path("results/shared/model_focus_20260518/subgoal_scorer_v1/cascade_subgoal_scorer.pkl")
        manifest = Path("results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
        if not model.exists() or not manifest.exists():
            self.skipTest("trained subgoal scorer artifacts are not present")
        leaf = "CC[C@@H](O)C[C@@H](O)CC=O"
        evidence_provider = CascadeSubgoalEvidenceProvider(
            manifest,
            model,
            min_score=-2.0,
            min_similarity=0.30,
            evidence_candidates=12,
            max_hints_per_leaf=3,
        )
        retro_provider = StaticProposalProvider({
            leaf: [
                {
                    "product_smiles": leaf,
                    "reactant_smiles": ["CC"],
                    "rxn_smiles": f"CC>>{leaf}",
                    "score": 0.9,
                    "stock_status": {"CC": True},
                }
            ]
        })
        planner = CascadeProgramSearch(
            [evidence_provider, retro_provider],
            stock_checker=lambda smi: smi == "CC",
            config=CascadeSearchConfig(max_depth=1, expansion_budget=4, branch_factor=1, allow_repair_actions=False),
        )

        result = planner.search(leaf, n_results=1)[0]

        self.assertTrue(result.solved)
        self.assertEqual(len(result.state.step_annotations), 1)
        self.assertGreaterEqual(len(result.state.raw_metadata.get("cascade_subgoal_hints") or []), 1)
        self.assertEqual(result.state.step_annotations[0].rxn_smiles, f"CC>>{leaf}")

    def test_subgoal_hint_action_scorer_softly_prioritizes_supported_action(self):
        leaf = "CC[C@@H](O)C[C@@H](O)CC=O"
        evidence_provider = StaticProposalProvider({})
        evidence_provider.provider_name = "cascade_subgoal_evidence"
        retro_provider = StaticProposalProvider({
            leaf: [
                {
                    "product_smiles": leaf,
                    "reactant_smiles": ["NN"],
                    "rxn_smiles": f"NN>>{leaf}",
                    "score": 0.60,
                    "stock_status": {"NN": False},
                },
                {
                    "product_smiles": leaf,
                    "reactant_smiles": ["C=CC(CC=O)CCCC"],
                    "rxn_smiles": f"C=CC(CC=O)CCCC>>{leaf}",
                    "score": 0.10,
                    "stock_status": {"C=CC(CC=O)CCCC": True},
                },
            ]
        })

        class FixedHintProvider:
            provider_name = "cascade_subgoal_evidence"
            max_hints_per_leaf = 1

            def __init__(self):
                self.last_diagnostics = {}

            def propose(self, request):
                from cascade_planner.cascade_search import CascadeAction, CascadeActionType

                return [
                    CascadeAction(
                        CascadeActionType.EVIDENCE_RETRIEVAL,
                        target_leaf=request.leaf_smiles,
                        evidence_payload={
                            "contract": "learned subgoal evidence hint; not a retrosynthetic reaction",
                            "target_leaf": request.leaf_smiles,
                            "subgoal_hint_id": "hint1",
                            "subgoal_smiles": leaf,
                            "evidence_smiles": "C=CC(CC=O)CCCC",
                            "evidence_transform": "c_c_coupling",
                            "doi": "demo",
                            "learned_subgoal_score": 1.0,
                        },
                    )
                ]

        controller = CascadeSearchController(
            value_model=HeuristicCascadeValueModel(),
            action_value_model=SubgoalHintActionScorer(max_bonus=1.0, min_similarity=0.40),
        )
        planner = CascadeProgramSearch(
            [FixedHintProvider(), retro_provider],
            stock_checker=lambda smi: smi == "C=CC(CC=O)CCCC",
            config=CascadeSearchConfig(
                max_depth=1,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=4,
                allow_repair_actions=False,
            ),
            controller=controller,
        )

        result = planner.search(leaf, n_results=1)[0]

        self.assertTrue(result.solved)
        self.assertEqual(result.state.step_annotations[0].reactant_smiles, ["C=CC(CC=O)CCCC"])
        action = result.state.raw_metadata["applied_actions"][-1]
        self.assertTrue(action["metadata"]["subgoal_hint_action_score"]["applicable"])


if __name__ == "__main__":
    unittest.main()
