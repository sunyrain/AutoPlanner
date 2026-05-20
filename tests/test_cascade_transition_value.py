import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascade_search import (
    CascadeAction,
    CascadeActionType,
    CascadeProgramSearch,
    CascadeProgramState,
    CascadeSearchConfig,
    CascadeSearchController,
    HeuristicCascadeValueModel,
    LoadedCascadeTransitionValueModel,
    StaticProposalProvider,
    StepAnnotation,
)
from cascade_planner.eval.build_cascade_transition_pack import build_cascade_transition_pack
from cascade_planner.eval.train_cascade_transition_value import train_cascade_transition_value


class CascadeTransitionValueTest(unittest.TestCase):
    def test_transition_value_model_controls_child_selection(self):
        class PreferSecondTransition:
            def score_transitions(self, state, actions, child_states, *, expanded_leaf=None):
                return [0.1, 0.9]

        provider = StaticProposalProvider(
            {
                "CCO": [
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": ["NN"],
                        "rxn_smiles": "NN>>CCO",
                        "score": 0.99,
                        "stock_status": {"NN": False},
                    },
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC"],
                        "rxn_smiles": "CC>>CCO",
                        "score": 0.05,
                        "stock_status": {"CC": True},
                    },
                ]
            }
        )
        controller = CascadeSearchController(
            value_model=HeuristicCascadeValueModel(),
            transition_value_model=PreferSecondTransition(),
        )
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi == "CC",
            config=CascadeSearchConfig(
                max_depth=1,
                branch_factor=1,
                proposal_top_k=2,
                expansion_budget=4,
                allow_repair_actions=False,
            ),
            controller=controller,
        )

        result = planner.search("CCO", n_results=1)[0]

        self.assertTrue(result.solved)
        self.assertEqual(result.state.step_annotations[0].reactant_smiles, ["CC"])
        self.assertEqual(result.diagnostics["controller"]["transition_value_model"], "PreferSecondTransition")

    def test_train_load_and_score_transition_value_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = root / "pack"
            pack_dir.mkdir()
            rows = []
            for target, good_reactant, bad_reactant in [
                ("CCO", "CC", "NN"),
                ("CCN", "CN", "OO"),
                ("CCC", "CC", "N#N"),
                ("CCCl", "CCl", "BrBr"),
            ]:
                parent = CascadeProgramState.initial(target).to_dict()
                for idx, (reactant, label) in enumerate([(good_reactant, 0.9), (bad_reactant, 0.1)], start=1):
                    action = CascadeAction(
                        CascadeActionType.RETROSYNTHETIC_STEP,
                        target_leaf=target,
                        step=StepAnnotation(
                            product_smiles=target,
                            reactant_smiles=[reactant],
                            rxn_smiles=f"{reactant}>>{target}",
                            score=1.0 - idx * 0.1,
                            stock_status={reactant: bool(label > 0.5)},
                        ),
                    ).to_dict()
                    rows.append(
                        {
                            "transition_id": f"{target}_{idx}",
                            "pool_id": target,
                            "target_smiles": target,
                            "expanded_leaf": target,
                            "parent_state": parent,
                            "candidate_action": action,
                            "child_summary": {
                                "step_count": 1,
                                "stage_count": 1,
                                "stock_closed": bool(label > 0.5),
                                "cofactor_closed": True,
                                "failure_categories": [] if label > 0.5 else ["StockDeadEnd"],
                                "open_leaves": [] if label > 0.5 else [reactant],
                                "cascade_cost": 0.2 if label > 0.5 else 2.0,
                            },
                            "labels": {
                                "transition_value": label,
                                "stock_closed": float(label > 0.5),
                                "condition_compatible": 1.0,
                                "cofactor_closed": 1.0,
                                "evidence_sufficient": 1.0,
                            },
                        }
                    )
            (pack_dir / "transition_value.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            model_path = root / "cascade_transition_value.pt"
            report = train_cascade_transition_value(
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

            loaded = LoadedCascadeTransitionValueModel(model_path)
            state = CascadeProgramState.initial("CCO")
            actions = [
                CascadeAction(
                    CascadeActionType.RETROSYNTHETIC_STEP,
                    target_leaf="CCO",
                    step=StepAnnotation(product_smiles="CCO", reactant_smiles=["CC"], rxn_smiles="CC>>CCO"),
                )
            ]
            scores = loaded.score_transitions(state, actions, [CascadeProgramState.initial("CC")], expanded_leaf="CCO")

            self.assertTrue(model_path.exists())
            self.assertGreaterEqual(report["metadata"]["n_rows"], 8)
            self.assertEqual(len(scores), 1)
            self.assertGreaterEqual(scores[0], 0.0)
            self.assertLessEqual(scores[0], 1.0)

    def test_transition_pack_adds_fragment_labels_when_benchmark_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            trace = root / "trace.jsonl"
            out = root / "transition_pack"
            benchmark.write_text(
                json.dumps([
                    {
                        "target_smiles": "CCO",
                        "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
                    }
                ]),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "target_smiles": "CCO",
                        "event": {
                            "state_id": "state0",
                            "state": CascadeProgramState.initial("CCO").to_dict(),
                            "depth": 0,
                            "open_leaves": ["CCO"],
                            "expanded_leaf": "CCO",
                            "candidate_scores": [0.9, 0.1],
                            "candidate_actions": [
                                CascadeAction(
                                    CascadeActionType.RETROSYNTHETIC_STEP,
                                    target_leaf="CCO",
                                    step=StepAnnotation(
                                        product_smiles="CCO",
                                        reactant_smiles=["CC", "O"],
                                        rxn_smiles="CC.O>>CCO",
                                    ),
                                ).to_dict(),
                                CascadeAction(
                                    CascadeActionType.RETROSYNTHETIC_STEP,
                                    target_leaf="CCO",
                                    step=StepAnnotation(
                                        product_smiles="CCO",
                                        reactant_smiles=["C", "CO"],
                                        rxn_smiles="C.CO>>CCO",
                                    ),
                                ).to_dict(),
                            ],
                            "child_summaries": [
                                {"step_count": 1, "stage_count": 1, "stock_closed": True, "cofactor_closed": True, "failure_categories": [], "open_leaves": []},
                                {"step_count": 1, "stage_count": 1, "stock_closed": False, "cofactor_closed": True, "failure_categories": ["StockDeadEnd"], "open_leaves": ["C"]},
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            build_cascade_transition_pack(trace_paths=[trace], benchmark_path=benchmark, output_dir=out)
            rows = [json.loads(line) for line in (out / "transition_value.jsonl").read_text().splitlines()]

        self.assertEqual(rows[0]["labels"]["exact_gt_reaction"], 1)
        self.assertEqual(rows[0]["labels"]["fragment_transition_value"], 0.85)
        self.assertGreaterEqual(rows[0]["labels"]["process_fragment_transition_value"], 0.85)
        self.assertEqual(rows[0]["labels"]["fragment_rank_transition_value"], 0.85)
        self.assertEqual(rows[1]["labels"]["fragment_transition_value"], 0.0)
        self.assertGreater(rows[1]["labels"]["fragment_rank_transition_value"], 0.0)
        self.assertLess(rows[1]["labels"]["fragment_rank_transition_value"], 0.30)


if __name__ == "__main__":
    unittest.main()
