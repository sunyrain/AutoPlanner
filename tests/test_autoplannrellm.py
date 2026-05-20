import os
import unittest
from unittest.mock import patch

from AUTOPLANNRELLM.controller import DeepSeekSelectionController
from AUTOPLANNRELLM.deepseek_client import DeepSeekJSONClient
from AUTOPLANNRELLM.proposals import append_llm_candidate
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.runtime import RouteTreeEvaluation
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request_json(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.response)


class FixedRuntime:
    def __init__(self, *, action_scores=None, node_scores=None):
        self.action_scores = list(action_scores or [])
        self.node_scores = list(node_scores or [])

    def evaluate(self, state, leaf, actions, *, stock_checker=None):
        return RouteTreeEvaluation(action_scores=list(self.action_scores), reason="fixed")

    def score_open_leaves(self, state, leaves, *, stock_checker=None):
        return RouteTreeEvaluation(action_scores=[], node_scores=list(self.node_scores), reason="fixed")


class AutoPlannrEllmTest(unittest.TestCase):
    def test_deepseek_client_normalizes_key_and_rejects_placeholder(self):
        self.assertEqual(DeepSeekJSONClient(api_key='"  test-key  "').api_key, "test-key")

        client = DeepSeekJSONClient(api_key="'  replace_with_your_deepseek_key  '")
        with patch.dict(os.environ, {"AUTOPLANNRELLM_MOCK_PLACEHOLDER_TEST_JSON": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "placeholder"):
                client.request_json(
                    task="placeholder_test",
                    system="Return JSON.",
                    user_payload={"input": "value"},
                )

    def test_action_selection_scores_prefer_deepseek_choice(self):
        state = RouteTreeState.initial("CC=O")
        actions = [
            CandidateAction.from_candidate("CC=O", {"main_reactant": "C", "rxn_smiles": "C>>CC=O", "score": 0.1}, source="base"),
            CandidateAction.from_candidate("CC=O", {"main_reactant": "CCO", "rxn_smiles": "CCO>>CC=O", "score": 0.1}, source="base"),
        ]
        controller = DeepSeekSelectionController(
            client=FakeClient({
                "action_preferences": [
                    {"index": 0, "score": 0.1},
                    {"index": 1, "score": 0.9},
                ],
                "confidence": 0.8,
            }),
            selection_weight=1.0,
        )
        result = controller.evaluate(state, "CC=O", actions)
        self.assertGreater(result.action_scores[1], result.action_scores[0])
        self.assertIn("autoplannrellm:deepseek_action_selection", result.reason)

    def test_leaf_selection_scores_prefer_deepseek_choice(self):
        state = RouteTreeState(
            target="CC=O",
            open_leaves=("CC=O", "CCO"),
            depth=1,
        )
        controller = DeepSeekSelectionController(
            client=FakeClient({
                "leaf_preferences": [
                    {"index": 0, "score": 0.2},
                    {"index": 1, "score": 0.8},
                ],
                "confidence": 0.7,
            }),
            selection_weight=1.0,
        )
        result = controller.score_open_leaves(state, ["CC=O", "CCO"])
        self.assertGreater(result.node_scores[1], result.node_scores[0])
        self.assertIn("autoplannrellm:deepseek_leaf_selection", result.reason)

    def test_action_selection_can_boost_multiple_selected_branches_without_penalizing_others(self):
        state = RouteTreeState.initial("CC=O")
        actions = [
            CandidateAction.from_candidate("CC=O", {"main_reactant": "C", "rxn_smiles": "C>>CC=O", "score": 0.1}, source="base"),
            CandidateAction.from_candidate("CC=O", {"main_reactant": "CO", "rxn_smiles": "CO>>CC=O", "score": 0.1}, source="base"),
            CandidateAction.from_candidate("CC=O", {"main_reactant": "CCO", "rxn_smiles": "CCO>>CC=O", "score": 0.1}, source="base"),
        ]
        controller = DeepSeekSelectionController(
            client=FakeClient({
                "selected_action_indices": [0, 2],
                "action_preferences": [
                    {"index": 0, "score": 0.2},
                    {"index": 1, "score": 0.99},
                    {"index": 2, "score": 0.7},
                ],
                "confidence": 0.8,
            }),
            fallback_runtime=FixedRuntime(action_scores=[0.0, 0.0, 0.0]),
            selection_weight=1.0,
        )
        with patch.dict(os.environ, {"AUTOPLANNRELLM_ACTION_TOPK": "3"}, clear=False):
            result = controller.evaluate(state, "CC=O", actions)
        self.assertGreater(result.action_scores[0], result.action_scores[1])
        self.assertGreater(result.action_scores[2], result.action_scores[1])
        self.assertEqual(result.action_scores[1], 0.0)
        self.assertIn("topk=3", result.reason)
        payload = controller.client.calls[0]["user_payload"]
        self.assertEqual(payload["selection_instructions"]["max_selected"], 3)
        self.assertIn("selected_action_indices", payload["required_schema"])

    def test_leaf_selection_can_boost_multiple_selected_leaves(self):
        state = RouteTreeState(
            target="CC=O",
            open_leaves=("CC=O", "CCO", "CO"),
            depth=1,
        )
        controller = DeepSeekSelectionController(
            client=FakeClient({
                "selected_leaf_indices": [0, 2],
                "leaf_preferences": [
                    {"index": 0, "score": 0.7},
                    {"index": 1, "score": 0.99},
                    {"index": 2, "score": 0.8},
                ],
                "confidence": 0.7,
            }),
            fallback_runtime=FixedRuntime(node_scores=[0.0, 0.0, 0.0]),
            selection_weight=1.0,
        )
        with patch.dict(os.environ, {"AUTOPLANNRELLM_LEAF_TOPK": "3"}, clear=False):
            result = controller.score_open_leaves(state, ["CC=O", "CCO", "CO"])
        self.assertGreater(result.node_scores[0], result.node_scores[1])
        self.assertGreater(result.node_scores[2], result.node_scores[1])
        self.assertEqual(result.node_scores[1], 0.0)
        self.assertIn("topk=3", result.reason)

    def test_llm_candidate_appends_one_deepseek_source(self):
        with patch.dict(
            os.environ,
            {
                "AUTOPLANNRELLM_ENABLE": "1",
                "AUTOPLANNRELLM_MOCK_REACTION_SUGGESTION_JSON": (
                    '{"reactants":["CCO"],"reaction_smiles":"CCO>>CC=O",'
                    '"reaction_type":"oxidation","ec":"","confidence":0.82,'
                    '"rationale":"test","unsupported_claims":[]}'
                ),
            },
            clear=False,
        ):
            diagnostics = {"sources": {}}
            actions = append_llm_candidate(
                product="CC=O",
                actions=[],
                context=ProposalContext(),
                diagnostics=diagnostics,
            )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "llm_deepseek")
        self.assertEqual(diagnostics["sources"]["llm_deepseek"]["final_returned"], 1)

    def test_retro_engine_tool_can_return_llm_candidate_when_base_pool_empty(self):
        with patch.dict(
            os.environ,
            {
                "AUTOPLANNRELLM_ENABLE": "1",
                "AUTOPLANNRELLM_MOCK_REACTION_SUGGESTION_JSON": (
                    '{"reactants":["CCO"],"reaction_smiles":"CCO>>CC=O",'
                    '"reaction_type":"oxidation","ec":"","confidence":0.82,'
                    '"rationale":"test","unsupported_claims":[]}'
                ),
            },
            clear=False,
        ):
            tool = RetroEngineProposalTool({}, source_order=())
            actions = tool.propose("CC=O", ProposalContext(), top_k=1)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].source, "llm_deepseek")
        self.assertEqual(tool.last_diagnostics["sources"]["llm_deepseek"]["final_returned"], 1)

    def test_default_runtime_env_can_wrap_autoplannrellm_controller(self):
        from cascade_planner.route_tree import runtime

        with patch.dict(
            os.environ,
            {
                "AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER": "1",
                "AUTOPLANNER_ROUTE_TREE_POLICY": "/missing/policy.pt",
                "AUTOPLANNRELLM_ENABLE": "1",
                "AUTOPLANNRELLM_LLM_SELECTION": "1",
            },
            clear=False,
        ):
            runtime._RUNTIME_CACHE.clear()
            controller = runtime.default_route_tree_runtime()
        self.assertIsInstance(controller, DeepSeekSelectionController)


if __name__ == "__main__":
    unittest.main()
