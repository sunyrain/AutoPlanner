import unittest

from cascade_planner.cascadeboard import CascadeBoard, RouteResult
from cascade_planner.cascadeboard.route_export import (
    candidate_pool_summary,
    cascade_compatibility_metrics,
    diversify_ranked_route_results,
    enzyme_evidence_metrics,
    reaction_interpretation,
    route_set_diversity_metrics,
    route_naturalness_metrics,
    route_metrics,
    operation_transition_metrics,
)


class RouteExportCompatibilityTest(unittest.TestCase):
    def test_candidate_pool_summary_reports_duplicate_and_reactant_set_diversity(self):
        board = CascadeBoard.from_n_steps(1, "CCOC(C)=O")
        slot = board.slots[0]
        slot.candidates = [
            {
                "rxn_smiles": "CCO.CC(=O)O>>CCOC(C)=O",
                "main_reactant": "CCO",
                "aux_reactants": ["CC(=O)O"],
                "source": "retrochimera",
            },
            {
                "rxn_smiles": "OCC.CC(=O)O>>CCOC(C)=O",
                "main_reactant": "OCC",
                "aux_reactants": ["CC(=O)O"],
                "source": "enzyformer",
            },
            {
                "reaction_smiles": "CCN.CC(=O)O>>CCNC(C)=O",
                "main_reactant": "CCN",
                "aux_reactants": ["CC(=O)O"],
                "source": "retrochimera",
            },
        ]

        summary = candidate_pool_summary(slot)

        self.assertEqual(summary["n_candidates"], 3)
        self.assertEqual(summary["source_counts"]["retrochimera"], 2)
        self.assertEqual(summary["source_counts"]["enzyformer"], 1)
        self.assertEqual(summary["unique_reactions"], 2)
        self.assertEqual(summary["unique_main_reactants"], 2)
        self.assertEqual(summary["unique_reactant_sets"], 2)
        self.assertAlmostEqual(summary["duplicate_reaction_fraction"], 1 / 3, places=4)
        self.assertAlmostEqual(summary["duplicate_main_reactant_fraction"], 1 / 3, places=4)
        self.assertAlmostEqual(summary["duplicate_reactant_set_fraction"], 1 / 3, places=4)
        self.assertAlmostEqual(summary["pool_diversity_score"], 2 / 3, places=4)
        self.assertTrue(summary["diversity_flags"]["has_duplicate_reactions"])
        self.assertTrue(summary["diversity_flags"]["has_duplicate_main_reactants"])
        self.assertTrue(summary["diversity_flags"]["has_duplicate_reactant_sets"])
        self.assertFalse(summary["diversity_flags"]["single_reactant_set_pool"])

        route = route_metrics(board)
        pool = route["candidate_pool"]
        self.assertEqual(pool["steps_with_candidates"], 1)
        self.assertEqual(pool["total_candidates"], 3)
        self.assertAlmostEqual(pool["avg_pool_diversity_score"], 2 / 3, places=4)
        self.assertAlmostEqual(pool["avg_duplicate_reactant_set_fraction"], 1 / 3, places=4)
        self.assertEqual(pool["candidate_pool_source_counts"]["retrochimera"], 2)

    def test_structured_compatibility_dimension_checks(self):
        board = CascadeBoard.from_n_steps(2, "CC=O")
        board.slots[0].reaction_smiles = "CCO.NAD+>>CC=O.NADH"
        board.slots[0].main_reactant = "CCO"
        board.slots[0].ec = "1.1.1.1"
        board.slots[0].solvent = "DMSO"
        board.slots[0].T = 30
        board.slots[0].pH = 7
        board.slots[1].reaction_smiles = "CC=O.NaBH4>>CCO"
        board.slots[1].main_reactant = "CC=O"
        board.slots[1].catalyst = "Pd/C"
        board.slots[1].T = 30
        board.slots[1].pH = 7

        metrics = cascade_compatibility_metrics(board)

        self.assertIn("solvent_incompatibility", metrics["issues"])
        self.assertIn("metal_enzyme_conflict", metrics["issues"])
        self.assertIn("cofactor_cross_talk", metrics["issues"])
        self.assertTrue(metrics["dimension_checks"]["solvent"]["has_solvent_fields"])
        self.assertTrue(metrics["dimension_checks"]["metal_enzyme"]["metal_enzyme_conflict"])

    def test_enzyme_evidence_counts_structured_fields(self):
        board = CascadeBoard.from_n_steps(1, "CC=O")
        slot = board.slots[0]
        slot.reaction_smiles = "CCO.NAD+>>CC=O.NADH"
        slot.main_reactant = "CCO"
        slot.ec = "1.1.1.1"
        slot.source = "v3_retrieval"
        slot.T = 30
        slot.pH = 7
        slot.evidence = {
            "uniprot_accession": "P12345",
            "organism": "Test organism",
            "sequence_length": 350,
            "cofactor": "NAD+",
            "doi": "10.0000/example",
            "substrate_similarity": 0.82,
            "condition_match": {"temperature_c": 30, "ph": 7},
            "literature_precedent": True,
        }

        metrics = enzyme_evidence_metrics(board)

        self.assertEqual(metrics["supported_steps"], 1)
        dims = metrics["steps"][0]["evidence_dimensions"]
        self.assertTrue(dims["uniprot"])
        self.assertTrue(dims["organism"])
        self.assertTrue(dims["sequence"])
        self.assertTrue(dims["literature_precedent"])
        self.assertGreater(metrics["enzyme_evidence_score"], 0.8)

    def test_reaction_interpretation_explains_type_ec_and_atom_change(self):
        board = CascadeBoard.from_n_steps(1, "CCOC(C)=O")
        slot = board.slots[0]
        slot.product = "CCOC(C)=O"
        slot.main_reactant = "CCO"
        slot.aux_reactants = ["CC(=O)O"]
        slot.reaction_smiles = "CCO.CC(=O)O>>CCOC(C)=O"
        slot.reaction_type = "esterification"
        slot.ec = "3.1.1.1"
        slot.T = 30
        slot.pH = 7

        interp = reaction_interpretation(slot)

        self.assertEqual(interp["reaction_class"], "esterification")
        self.assertIn("Esterification", interp["reaction_principle"])
        self.assertIn("EC 3 hydrolase", interp["ec_principle"])
        self.assertEqual(interp["atom_change"]["heavy_atom_delta"], 3)
        self.assertTrue(interp["likely_added_or_removed"])
        self.assertTrue(interp["catalysis_and_conditions"])

    def test_route_naturalness_flags_cycles_and_product_mismatch(self):
        board = CascadeBoard.from_n_steps(2, "CC=O")
        board.slots[0].product = "CC=O"
        board.slots[0].main_reactant = "CC=O"
        board.slots[0].reaction_smiles = "CC=O>>CC=O"
        board.slots[0].T = 30
        board.slots[0].pH = 7
        board.slots[1].product = "CC=O"
        board.slots[1].main_reactant = "CCO"
        board.slots[1].reaction_smiles = "CCO>>CC"
        board.slots[1].T = 31
        board.slots[1].pH = 7

        metrics = route_naturalness_metrics(board)
        compat = cascade_compatibility_metrics(board)

        self.assertEqual(metrics["self_loop_steps"], 1)
        self.assertEqual(metrics["product_mismatch_steps"], 1)
        self.assertLess(metrics["naturalness_score"], 1.0)
        self.assertIn("route_cycle", compat["issues"])
        self.assertIn("reaction_product_mismatch", compat["issues"])

    def test_route_naturalness_flags_atom_balance_violation(self):
        board = CascadeBoard.from_n_steps(1, "CCCCCCCCCCCC")
        slot = board.slots[0]
        slot.product = "CCCCCCCCCCCC"
        slot.main_reactant = "O"
        slot.reaction_smiles = "O>>CCCCCCCCCCCC"
        slot.T = 30
        slot.pH = 7

        metrics = route_naturalness_metrics(board)
        compat = cascade_compatibility_metrics(board)

        self.assertEqual(metrics["atom_balance_violations"], 1)
        self.assertLess(metrics["naturalness_score"], 1.0)
        self.assertIn("atom_balance_violation", compat["issues"])

    def test_filled_route_is_not_solved_without_real_disconnection_progress(self):
        board = CascadeBoard.from_n_steps(2, "CCCCCCCCCCCCCCCCCCCC")
        board.slots[0].product = "CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].main_reactant = "CCCCCCCCCCCCCCCCCC"
        board.slots[0].reaction_smiles = "CCCCCCCCCCCCCCCCCC>>CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].T = 30
        board.slots[0].pH = 7
        board.slots[1].product = "CCCCCCCCCCCCCCCCCC"
        board.slots[1].main_reactant = "CCCCCCCCCCCCCCCC"
        board.slots[1].reaction_smiles = "CCCCCCCCCCCCCCCC>>CCCCCCCCCCCCCCCCCC"
        board.slots[1].T = 30
        board.slots[1].pH = 7

        metrics = route_metrics(board)

        self.assertTrue(metrics["filled_route"])
        self.assertFalse(metrics["progressive_route"])
        self.assertFalse(metrics["route_solved"])
        self.assertEqual(metrics["retrosynthesis_progress"]["main_chain_reduction"], 0.2)

    def test_progressive_route_can_be_solved_without_stock_checker_when_terminal_is_simple(self):
        board = CascadeBoard.from_n_steps(2, "CCCCCCCCCCCCCCCCCCCC")
        board.slots[0].product = "CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].main_reactant = "CCCCCCCCCCCC"
        board.slots[0].reaction_smiles = "CCCCCCCCCCCC>>CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].T = 30
        board.slots[0].pH = 7
        board.slots[1].product = "CCCCCCCCCCCC"
        board.slots[1].main_reactant = "CCCCCC"
        board.slots[1].reaction_smiles = "CCCCCC>>CCCCCCCCCCCC"
        board.slots[1].T = 30
        board.slots[1].pH = 7

        metrics = route_metrics(board)

        self.assertTrue(metrics["filled_route"])
        self.assertTrue(metrics["progressive_route"])
        self.assertTrue(metrics["route_solved"])
        self.assertEqual(metrics["retrosynthesis_progress"]["main_chain_reduction"], 0.7)
        self.assertTrue(metrics["retrosynthesis_progress"]["terminal_simplified"])

    def test_stock_checker_overrides_simple_terminal_solve_heuristic(self):
        board = CascadeBoard.from_n_steps(1, "CCCCCCCCCCCC")
        board.slots[0].product = "CCCCCCCCCCCC"
        board.slots[0].main_reactant = "CCCCCC"
        board.slots[0].reaction_smiles = "CCCCCC>>CCCCCCCCCCCC"
        board.slots[0].T = 30
        board.slots[0].pH = 7

        metrics = route_metrics(board, stock_checker=lambda _smi: False)

        self.assertTrue(metrics["filled_route"])
        self.assertTrue(metrics["progressive_route"])
        self.assertFalse(metrics["strict_stock_solve"])
        self.assertFalse(metrics["route_solved"])

    def test_large_auxiliary_leaf_prevents_progressive_route(self):
        board = CascadeBoard.from_n_steps(1, "CCCCCCCCCCCCCCCCCCCC")
        board.slots[0].product = "CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].main_reactant = "CCCCCC"
        board.slots[0].aux_reactants = ["CCCCCCCCCCCCCCCCCC"]
        board.slots[0].reaction_smiles = "CCCCCC.CCCCCCCCCCCCCCCCCC>>CCCCCCCCCCCCCCCCCCCC"
        board.slots[0].T = 30
        board.slots[0].pH = 7

        metrics = route_metrics(board)
        progress = metrics["retrosynthesis_progress"]

        self.assertTrue(metrics["filled_route"])
        self.assertFalse(metrics["progressive_route"])
        self.assertFalse(metrics["route_solved"])
        self.assertFalse(progress["leaf_simplified"])
        self.assertEqual(progress["largest_leaf_heavy_atoms"], 18)

    def test_route_set_diversity_detects_duplicate_and_distinct_routes(self):
        routes = [
            {
                "steps": [
                    {"reaction_type": "oxidation", "source": "enzyformer", "ec": "1.1.1.1", "main_reactant": "CCO"},
                    {"reaction_type": "reduction", "source": "retrochimera", "ec": "", "main_reactant": "CC=O"},
                ],
                "metrics": {"terminal_reactants": ["CCO"]},
            },
            {
                "steps": [
                    {"reaction_type": "oxidation", "source": "enzyformer", "ec": "1.1.1.1", "main_reactant": "CCO"},
                    {"reaction_type": "reduction", "source": "retrochimera", "ec": "", "main_reactant": "CC=O"},
                ],
                "metrics": {"terminal_reactants": ["CCO"]},
            },
            {
                "steps": [
                    {"reaction_type": "amination", "source": "v3_retrieval", "ec": "2.6.1.1", "main_reactant": "CCN"},
                    {"reaction_type": "hydrolysis", "source": "retrochimera", "ec": "3.1.1.1", "main_reactant": "CC(=O)O"},
                ],
                "metrics": {"terminal_reactants": ["CCN", "CC(=O)O"]},
            },
        ]

        metrics = route_set_diversity_metrics(routes)

        self.assertEqual(metrics["n_routes"], 3)
        self.assertEqual(metrics["unique_type_sequences"], 2)
        self.assertEqual(metrics["unique_full_signatures"], 2)
        self.assertGreater(metrics["duplicate_route_fraction"], 0.0)
        self.assertGreater(metrics["mean_pairwise_type_distance"], 0.0)
        self.assertGreater(metrics["mean_pairwise_terminal_jaccard_distance"], 0.0)

    def test_operation_transition_metrics_counts_chemo_bio_and_condition_switches(self):
        board = CascadeBoard.from_n_steps(2, "CC=O")
        board.slots[0].reaction_smiles = "CCO>>CC=O"
        board.slots[0].main_reactant = "CCO"
        board.slots[0].reaction_type = "oxidation"
        board.slots[0].source = "enzyformer"
        board.slots[0].ec = "1.1.1.1"
        board.slots[0].T = 30
        board.slots[0].pH = 7
        board.slots[0].solvent = "buffer"
        board.slots[1].reaction_smiles = "CC=O>>CCO"
        board.slots[1].main_reactant = "CC=O"
        board.slots[1].reaction_type = "reduction"
        board.slots[1].source = "retrochimera"
        board.slots[1].T = 55
        board.slots[1].pH = 4
        board.slots[1].solvent = "THF"

        metrics = operation_transition_metrics(board)
        route = route_metrics(board)

        self.assertEqual(metrics["step_classes"], ["enzymatic", "chemical"])
        self.assertEqual(metrics["chemo_bio_transitions"], 1)
        self.assertEqual(metrics["temperature_shifts"], 1)
        self.assertEqual(metrics["pH_shifts"], 1)
        self.assertEqual(metrics["solvent_switches"], 1)
        self.assertIn("operation_transitions", route)
        self.assertLess(metrics["operation_score"], 1.0)

    def test_diversify_ranked_route_results_pushes_exact_duplicates_later(self):
        def result(rtype, reactant, score):
            board = CascadeBoard.from_n_steps(1, "CCO")
            board.slots[0].reaction_type = rtype
            board.slots[0].source = "retrochimera"
            board.slots[0].main_reactant = reactant
            return RouteResult(board=board, score=score)

        best = result("oxidation", "CCO", 3.0)
        duplicate = result("oxidation", "CCO", 2.0)
        distinct = result("amination", "CCN", 1.0)

        reranked = diversify_ranked_route_results([best, duplicate, distinct])

        self.assertIs(reranked[0], best)
        self.assertIs(reranked[1], distinct)
        self.assertIs(reranked[2], duplicate)


if __name__ == "__main__":
    unittest.main()
