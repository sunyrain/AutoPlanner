import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cascade_planner.eval.build_route_tree_source_policy_pack import build_route_tree_source_policy_pack
from cascade_planner.eval.train_cascade_source_policy import (
    build_cascade_source_policy_dataset,
    train_cascade_source_policy,
)
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import CascadeSourcePolicyGate, SourceGate, default_source_gate, source_policy_group


class _StaticRetroEngine:
    def predict(self, product, **kwargs):
        del product, kwargs
        return [
            {
                "main_reactant": "CC",
                "aux_reactants": ["CC"],
                "rxn_smiles": "CC.CC>>CCCC",
                "score": 0.9,
                "source": "retrochimera",
                "type": "coupling",
            }
        ]


class RouteTreeSourcePolicyTest(unittest.TestCase):
    def test_build_source_policy_pack_and_train_checkpoint(self):
        action = CandidateAction.from_candidate(
            "CCCCCCCC",
            {
                "main_reactant": "CCCC",
                "aux_reactants": ["CCCC"],
                "rxn_smiles": "CCCC.CCCC>>CCCCCCCC",
                "score": 0.9,
                "source": "retrochimera",
                "type": "coupling",
            },
        )
        trace_row = {
            "schema_version": "route_tree_trace.v1",
            "benchmark_index": 3,
            "target_smiles": "CCCCCCCC",
            "route_metrics": [{"strict_stock_solve": True}],
            "event": {
                "state_id": "s0",
                "state": {
                    "state_id": "s0",
                    "target": "CCCCCCCC",
                    "depth": 0,
                    "open_leaves": ["CCCCCCCC"],
                    "expanded": [],
                    "steps": [],
                },
                "depth": 0,
                "open_leaves": ["CCCCCCCC"],
                "expanded_leaf": "CCCCCCCC",
                "candidate_actions": [action.to_dict()],
                "selected_action_key": action.canonical_key,
                "selected_next_stock_closed": True,
                "proposal_diagnostics": [
                    {
                        "leaf": "CCCCCCCC",
                        "proposal_budget": 4,
                        "top_k": 4,
                        "ordered_sources": ["retrochimera", "v3_retrieval"],
                        "allocation": {
                            "source_budgets": {"retrochimera": 3, "v3_retrieval": 0},
                            "budget_multiplier": 1.0,
                            "budget_multiplier_label": "1x",
                            "decision": "query",
                            "policy_confidence": 0.8,
                        },
                        "sources": {
                            "retrochimera": {
                                "queried": True,
                                "allocated_budget": 3,
                                "requested_k_total": 3,
                                "raw_returned": 2,
                                "kept_returned": 1,
                                "final_returned": 1,
                                "latency_ms_total": 12.0,
                            },
                            "v3_retrieval": {
                                "queried": False,
                                "allocated_budget": 0,
                                "skip_reason": "zero_budget",
                            },
                        },
                        "raw_actions": 2,
                        "final_actions": 1,
                    }
                ],
                "outcome": {"search_status": "solved", "solved_routes": 1},
            },
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            manifest = build_route_tree_source_policy_pack(trace_paths=[trace], output_dir=root / "pack")
            pack = Path(manifest["files"]["source_policy_pack"])
            rows = [json.loads(line) for line in pack.read_text(encoding="utf-8").splitlines()]
            dataset = build_cascade_source_policy_dataset(pack, n_bits=16)
            report = train_cascade_source_policy(
                pack=pack,
                output=root / "cascade_source_policy.pt",
                report=root / "cascade_source_policy.json",
                epochs=1,
                batch_size=2,
                n_bits=16,
                device="cpu",
            )
            gate = CascadeSourcePolicyGate(root / "cascade_source_policy.pt")

        useful = [row for row in rows if row["source"] == "retrochimera"][0]
        skipped = [row for row in rows if row["source"] == "v3_retrieval"][0]
        self.assertEqual(manifest["counts"]["rows"], 2)
        self.assertTrue(useful["useful_candidate_hit"])
        self.assertTrue(useful["stock_closing_hit"])
        self.assertIn("source_not_queried", skipped["failure_labels"])
        self.assertEqual(source_policy_group("v3_retrieval"), "retrieval")
        self.assertEqual(dataset.x.shape[0], 2)
        self.assertIn("retrieval", dataset.schema["source_groups"])
        self.assertIn("best_val_loss", report)
        self.assertEqual(gate.groups, dataset.schema["source_groups"])

    def test_default_source_gate_falls_back_on_missing_checkpoint(self):
        old_policy = os.environ.get("AUTOPLANNER_CASCADE_SOURCE_POLICY")
        try:
            os.environ["AUTOPLANNER_CASCADE_SOURCE_POLICY"] = "/tmp/definitely_missing_source_policy.pt"
            gate = default_source_gate()
        finally:
            if old_policy is None:
                os.environ.pop("AUTOPLANNER_CASCADE_SOURCE_POLICY", None)
            else:
                os.environ["AUTOPLANNER_CASCADE_SOURCE_POLICY"] = old_policy
        self.assertIsInstance(gate, SourceGate)

    def test_source_gate_budget_sum_and_safety_guard(self):
        gate = SourceGate()
        context = SimpleNamespace(ec1=1, reaction_type="", route_metadata={"carbohydrate_like_route": True})
        allocation = gate.allocate(
            "CCCCCCCC",
            context=context,
            available_sources=["retrochimera", "v3_retrieval", "retrorules"],
            total_budget=5,
        )
        self.assertLessEqual(sum(allocation.source_budgets.values()), 5)
        self.assertEqual(allocation.safety_guard, "carbohydrate_ec_prefers_enzymatic_sources")
        self.assertEqual(allocation.source_budgets.get("retrochimera", 0), 0)

    def test_depth_disable_policy_records_source_skip(self):
        old_policy = os.environ.get("AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH")
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH"] = "retrochimera:0"
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            tool = RetroEngineProposalTool(
                {"retrochimera": _StaticRetroEngine()},
                source_order=("retrochimera",),
                source_gate=SourceGate(),
            )
            root_actions = tool.propose("CCCC", ProposalContext(depth=0), top_k=1)
            self.assertEqual(len(root_actions), 1)
            self.assertTrue(tool.last_diagnostics["sources"]["retrochimera"]["queried"])

            downstream_actions = tool.propose("CCCC", ProposalContext(depth=1), top_k=1)
            self.assertEqual(downstream_actions, [])
            self.assertEqual(
                tool.last_diagnostics["sources"]["retrochimera"]["skip_reason"],
                "disabled_after_depth_0",
            )
        finally:
            if old_policy is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH"] = old_policy
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = old_retrieval

    def test_stock_rescue_retry_context_applies_late_source_floors(self):
        old_retrieval = os.environ.get("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL")
        try:
            os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = "0"
            tool = RetroEngineProposalTool(
                {
                    "retrochimera": _StaticRetroEngine(),
                    "chemtemplates": _StaticRetroEngine(),
                },
                source_order=("retrochimera", "chemtemplates"),
                source_gate=SourceGate(),
            )
            tool.propose(
                "CCCC",
                ProposalContext(depth=1, route_metadata={"stock_rescue_retry": True}),
                top_k=4,
            )
        finally:
            if old_retrieval is None:
                os.environ.pop("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", None)
            else:
                os.environ["AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL"] = old_retrieval

        budgets = tool.last_diagnostics["allocation"]["source_budgets"]
        self.assertGreaterEqual(budgets["retrochimera"], 2)
        self.assertGreaterEqual(budgets["chemtemplates"], 2)


if __name__ == "__main__":
    unittest.main()
