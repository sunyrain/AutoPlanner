import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascade_search import (
    CascadeProgramSearch,
    CascadeSearchConfig,
    CascadeSearchController,
    ConditionEnvelope,
    HeuristicCascadeValueModel,
    LearnedCascadePairScorer,
    RuleCascadePairScorer,
    StaticProposalProvider,
    StepAnnotation,
)
from cascade_planner.eval.build_cascade_pair_pack import build_cascade_pair_pack
from cascade_planner.eval.train_cascade_pair_scorer import train_cascade_pair_scorer


class CascadePairScorerTest(unittest.TestCase):
    def test_rule_pair_scorer_prefers_same_pot_over_isolation(self):
        scorer = RuleCascadePairScorer()
        compatible = scorer.score_pair(
            {
                "rxn_smiles": "CCO>>CC=O",
                "pairwise_mode": "simultaneous",
                "step_conditions": {"temperature_c": 30, "ph": 7, "solvent": "water"},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "1.1.1.1"}],
            },
            {
                "rxn_smiles": "CC=O>>CCN",
                "pairwise_mode": "not_applicable",
                "step_conditions": {"temperature_c": 32, "ph": 7.2, "solvent": "water"},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "2.6.1.1"}],
            },
        )
        isolated = scorer.score_pair(
            {
                "rxn_smiles": "CCO>>CC=O",
                "pairwise_mode": "isolated_transfer",
                "intermediate_isolated": True,
                "step_conditions": {"temperature_c": 90, "ph": 1, "solvent": "toluene"},
                "catalyst_components": [{"catalyst_class": "metal_catalyst"}],
            },
            {
                "rxn_smiles": "CC=O>>CCN",
                "pairwise_mode": "not_applicable",
                "step_conditions": {"temperature_c": 30, "ph": 7, "solvent": "water"},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "2.6.1.1"}],
            },
        )

        self.assertGreater(compatible.compatibility_score, isolated.compatibility_score)
        self.assertLess(compatible.isolation_required_probability, isolated.isolation_required_probability)

    def test_pair_reward_controls_candidate_selection(self):
        downstream = StepAnnotation(
            product_smiles="CCN",
            reactant_smiles=["CC=O"],
            rxn_smiles="CC=O>>CCN",
            score=0.9,
            condition=ConditionEnvelope(temperature_c_min=30, temperature_c_max=35, ph_min=7, ph_max=8, solvents=["water"]),
            stock_status={"CC=O": False},
        )
        provider = StaticProposalProvider(
            {
                "CC=O": [
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["bad"],
                        "rxn_smiles": "bad>>CC=O",
                        "score": 0.99,
                        "stock_status": {"bad": True},
                        "condition": {"Temperature": 95, "pH": 1, "Solvent": "toluene"},
                    },
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["good"],
                        "rxn_smiles": "good>>CC=O",
                        "score": 0.05,
                        "stock_status": {"good": True},
                        "condition": {"Temperature": 32, "pH": 7.2, "Solvent": "water"},
                    },
                ]
            }
        )
        state_provider = StaticProposalProvider({})
        controller = CascadeSearchController(
            value_model=HeuristicCascadeValueModel(),
            pair_scorer=RuleCascadePairScorer(),
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi in {"good", "bad"},
            config=CascadeSearchConfig(
                max_depth=2,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=4,
                allow_repair_actions=False,
                pair_reward_weight=4.0,
            ),
            controller=controller,
        )
        # Seed the planner through a direct state expansion: this isolates the
        # local pair reward without depending on a multi-step provider setup.
        from cascade_planner.cascade_search.state import CascadeProgramState

        seeded = CascadeProgramState.initial("CCN")
        seeded.append_step(downstream, opened_leaves=["CC=O"])
        children = planner._expand_state(seeded, [])

        self.assertTrue(children)
        self.assertEqual(children[0].step_annotations[-1].reactant_smiles, ["good"])
        self.assertIn("cascade_pair_score", children[0].raw_metadata["applied_actions"][-1]["metadata"])
        self.assertGreater(children[0].raw_metadata["cascade_pair_summary"]["valid_pair_count"], 0)

    def test_guarded_pair_reward_preserves_stock_closed_base_top(self):
        downstream = StepAnnotation(
            product_smiles="CCN",
            reactant_smiles=["CC=O"],
            rxn_smiles="CC=O>>CCN",
            score=0.9,
            condition=ConditionEnvelope(temperature_c_min=30, temperature_c_max=35, ph_min=7, ph_max=8, solvents=["water"]),
            stock_status={"CC=O": False},
        )
        provider = StaticProposalProvider(
            {
                "CC=O": [
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["bad_stock"],
                        "rxn_smiles": "bad_stock>>CC=O",
                        "score": 0.5,
                        "stock_status": {"bad_stock": True},
                        "condition": {"Temperature": 95, "pH": 1, "Solvent": "toluene"},
                    },
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["good_open"],
                        "rxn_smiles": "good_open>>CC=O",
                        "score": 0.5,
                        "stock_status": {"good_open": False},
                        "condition": {"Temperature": 32, "pH": 7.2, "Solvent": "water"},
                    },
                ]
            }
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi == "bad_stock",
            config=CascadeSearchConfig(
                max_depth=2,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=4,
                allow_repair_actions=False,
                pair_reward_weight=4.0,
                pair_reward_mode="guarded_tie_break",
            ),
            controller=CascadeSearchController(
                value_model=HeuristicCascadeValueModel(),
                pair_scorer=RuleCascadePairScorer(),
            ),
        )
        from cascade_planner.cascade_search.state import CascadeProgramState

        seeded = CascadeProgramState.initial("CCN")
        seeded.append_step(downstream, opened_leaves=["CC=O"])
        children = planner._expand_state(seeded, [])

        self.assertTrue(children)
        applied = children[0].raw_metadata["applied_actions"][-1]["metadata"]
        self.assertEqual(children[0].step_annotations[-1].reactant_smiles, ["bad_stock"])
        self.assertEqual(applied["cascade_pair_reward_mode"], "guarded_tie_break")

    def test_guarded_pair_reward_allows_safe_tie_break(self):
        downstream = StepAnnotation(
            product_smiles="CCN",
            reactant_smiles=["CC=O"],
            rxn_smiles="CC=O>>CCN",
            score=0.9,
            condition=ConditionEnvelope(temperature_c_min=30, temperature_c_max=35, ph_min=7, ph_max=8, solvents=["water"]),
            stock_status={"CC=O": False},
        )
        provider = StaticProposalProvider(
            {
                "CC=O": [
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["bad_stock"],
                        "rxn_smiles": "bad_stock>>CC=O",
                        "score": 0.5,
                        "stock_status": {"bad_stock": True},
                        "condition": {"Temperature": 95, "pH": 1, "Solvent": "toluene"},
                    },
                    {
                        "product_smiles": "CC=O",
                        "reactant_smiles": ["good_stock"],
                        "rxn_smiles": "good_stock>>CC=O",
                        "score": 0.5,
                        "stock_status": {"good_stock": True},
                        "condition": {"Temperature": 32, "pH": 7.2, "Solvent": "water"},
                    },
                ]
            }
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi in {"bad_stock", "good_stock"},
            config=CascadeSearchConfig(
                max_depth=2,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=4,
                allow_repair_actions=False,
                pair_reward_weight=4.0,
                pair_reward_mode="guarded_tie_break",
            ),
            controller=CascadeSearchController(
                value_model=HeuristicCascadeValueModel(),
                pair_scorer=RuleCascadePairScorer(),
            ),
        )
        from cascade_planner.cascade_search.state import CascadeProgramState

        seeded = CascadeProgramState.initial("CCN")
        seeded.append_step(downstream, opened_leaves=["CC=O"])
        children = planner._expand_state(seeded, [])

        self.assertTrue(children)
        applied = children[0].raw_metadata["applied_actions"][-1]["metadata"]
        self.assertEqual(children[0].step_annotations[-1].reactant_smiles, ["good_stock"])
        self.assertEqual(applied["cascade_pair_guard_reason"], "applied")

    def test_no_adjacent_pair_does_not_create_pair_reward(self):
        provider = StaticProposalProvider(
            {
                "CCN": [
                    {
                        "product_smiles": "CCN",
                        "reactant_smiles": ["bad"],
                        "rxn_smiles": "bad>>CCN",
                        "score": 0.99,
                        "stock_status": {"bad": True},
                        "condition": {"Temperature": 95, "pH": 1, "Solvent": "toluene"},
                    },
                    {
                        "product_smiles": "CCN",
                        "reactant_smiles": ["good"],
                        "rxn_smiles": "good>>CCN",
                        "score": 0.05,
                        "stock_status": {"good": True},
                        "condition": {"Temperature": 32, "pH": 7.2, "Solvent": "water"},
                    },
                ]
            }
        )
        controller = CascadeSearchController(
            value_model=HeuristicCascadeValueModel(),
            pair_scorer=RuleCascadePairScorer(),
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi in {"good", "bad"},
            config=CascadeSearchConfig(
                max_depth=1,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=2,
                allow_repair_actions=False,
                pair_reward_weight=100.0,
            ),
            controller=controller,
        )
        from cascade_planner.cascade_search.state import CascadeProgramState

        children = planner._expand_state(
            CascadeProgramState.initial("CCN"),
            [],
        )

        self.assertTrue(children)
        self.assertEqual(children[0].step_annotations[-1].reactant_smiles, ["bad"])
        applied = children[0].raw_metadata["applied_actions"][-1]["metadata"]
        self.assertFalse(applied["cascade_pair_applicable"])
        self.assertEqual(applied["cascade_pair_reward"], 0.0)
        self.assertNotIn("cascade_pair_summary", children[0].raw_metadata)

    def test_build_train_and_load_pair_scorer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v4 = root / "v4.jsonl"
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps([]), encoding="utf-8")
            records = [
                _record("10.a", "simultaneous", False, "water", "water", 30, 32),
                _record("10.b", "isolated_transfer", True, "toluene", "water", 90, 30),
                _record("10.c", "telescoped", False, "water", "water", 28, 35),
                _record("10.d", "sequential_addition", False, "water", "water", 25, 25),
            ]
            v4.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            pack_dir = root / "pack"
            report = build_cascade_pair_pack(
                v4_jsonl=v4,
                benchmark_path=benchmark,
                output_dir=pack_dir,
                hard_negative_per_positive=1,
            )
            model_path = root / "pair.pt"
            train_report = train_cascade_pair_scorer(
                pack_dir=pack_dir,
                model_output=model_path,
                report_output=root / "report.json",
                md_output=root / "report.md",
                epochs=1,
                batch_size=4,
                n_bits=16,
                hidden=32,
                device="cpu",
            )
            scorer = LearnedCascadePairScorer(model_path)
            score = scorer.score_pair(records[0]["steps"][0], records[0]["steps"][1])
            model_exists = model_path.exists()

        self.assertGreaterEqual(report["counts"]["rows"], 4)
        self.assertTrue(model_exists)
        self.assertGreaterEqual(train_report["metadata"]["n_rows"], 4)
        self.assertGreaterEqual(score.compatibility_score, 0.0)
        self.assertLessEqual(score.compatibility_score, 1.0)


def _record(doi, mode, isolated, left_solvent, right_solvent, left_temp, right_temp):
    return {
        "doi": doi,
        "cascade_id": "cascade_1",
        "trainable_recommended": True,
        "target_product_smiles": "CCN",
        "cascade_type": "chemoenzymatic",
        "quality_tier": "gold",
        "compatibility": {
            "compatibility_label": "empirically_compatible" if not isolated else "sequential_preferred",
            "evidence_strength": "strong_process_evidence",
        },
        "steps": [
            {
                "step_index": 1,
                "rxn_smiles": "CCO>>CC=O",
                "pairwise_mode": mode,
                "intermediate_isolated": isolated,
                "step_conditions": {"temperature_c": left_temp, "ph": 7, "solvent": left_solvent},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "1.1.1.1"}],
                "transformation_superclass": "oxidation",
            },
            {
                "step_index": 2,
                "rxn_smiles": "CC=O>>CCN",
                "pairwise_mode": "not_applicable",
                "step_conditions": {"temperature_c": right_temp, "ph": 7.2, "solvent": right_solvent},
                "catalyst_components": [{"catalyst_class": "enzyme", "ec_number": "2.6.1.1"}],
                "transformation_superclass": "amination",
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
