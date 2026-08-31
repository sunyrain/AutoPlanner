from __future__ import annotations

from threading import Event

import pytest

from cascade_planner.interfaces.aizynthfinder_strategy_sidecar import (
    run_aizynthfinder_strategy_branch_sidecar,
)


def test_strategy_sidecar_executes_real_aiz_mcts_and_backtracks() -> None:
    requests: list[dict] = []

    def handler(request):
        request = dict(request)
        requests.append(request)
        selected = request["expandable_smiles"][0]
        mapped = request["expandable_mapped_smiles"][0]
        if selected == "CCO":
            return {
                "candidates": [
                    {
                        "candidate_id": "dead-high-prior",
                        "product_smiles": "CCO",
                        "mapped_product_smiles": mapped,
                        "precursor_smiles": ["CC", "O"],
                        "mapped_precursor_smiles": [
                            "[CH3:1][CH3:2]",
                            "[OH2:3]",
                        ],
                        "route_step": {
                            "step_id": "dead-high-prior",
                            "product_smiles": "CCO",
                        },
                        "prior": 0.95,
                        "candidate_key": "dead-high-prior",
                    },
                    {
                        "candidate_id": "solvable-lower-prior",
                        "product_smiles": "CCO",
                        "mapped_product_smiles": mapped,
                        "precursor_smiles": ["C", "CO"],
                        "mapped_precursor_smiles": [
                            "[CH4:1]",
                            "[CH3:2][OH:3]",
                        ],
                        "route_step": {
                            "step_id": "solvable-lower-prior",
                            "product_smiles": "CCO",
                        },
                        "prior": 0.2,
                        "candidate_key": "solvable-lower-prior",
                    },
                ]
            }
        if selected == "CO":
            return {
                "candidates": [
                    {
                        "candidate_id": "solvable-tail",
                        "product_smiles": "CO",
                        "mapped_product_smiles": mapped,
                        "precursor_smiles": ["C", "O"],
                        "mapped_precursor_smiles": ["[CH4:2]", "[OH2:3]"],
                        "route_step": {
                            "step_id": "solvable-tail",
                            "product_smiles": "CO",
                        },
                        "prior": 1.0,
                        "candidate_key": "solvable-tail",
                    }
                ]
            }
        return {"candidates": []}

    result = run_aizynthfinder_strategy_branch_sidecar(
        target_smiles="CCO",
        strategy_id="strategy-1",
        strategy_text="prefer a convergent disconnection",
        request_handler=handler,
        inline_stock_smiles=("C", "O"),
        max_policy_calls=25,
        max_candidates_per_call=3,
        max_transforms=6,
        max_mcts_iterations=50,
        timeout_s=60,
    )

    assert result["solved"] is True
    assert [row["step_id"] for row in result["route_steps"]] == [
        "solvable-lower-prior",
        "solvable-tail",
    ]
    diagnostics = result["diagnostics"]
    assert diagnostics["engine"] == "AiZynthFinder.MctsSearchTree"
    assert diagnostics["selected_solved"] is True
    assert diagnostics["tree_nodes"] >= 4
    assert diagnostics["maximum_tree_depth"] >= diagnostics["selected_depth"]
    assert result["mcts_iterations"] > 1
    assert any(row["depth"] > 0 and row["route_steps"] for row in requests)


def test_strategy_sidecar_honors_a_pre_requested_cancellation() -> None:
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match="strategy_sidecar_cancelled"):
        run_aizynthfinder_strategy_branch_sidecar(
            target_smiles="CCO",
            strategy_id="strategy-cancelled",
            strategy_text="cancel before launch",
            request_handler=lambda _request: {"candidates": []},
            inline_stock_smiles=("C",),
            cancel_event=cancel_event,
        )


def test_strategy_sidecar_preserves_unbilled_callback_semantics() -> None:
    callbacks = 0

    def handler(request):
        nonlocal callbacks
        callbacks += 1
        if callbacks == 1:
            return {"candidates": [], "model_call_consumed": False}
        return {
            "model_call_consumed": True,
            "candidates": [
                {
                    "candidate_id": "one-paid-call",
                    "product_smiles": request["expandable_smiles"][0],
                    "mapped_product_smiles": request[
                        "expandable_mapped_smiles"
                    ][0],
                    "precursor_smiles": ["C", "O"],
                    "mapped_precursor_smiles": ["[CH4:1]", "[OH2:2]"],
                    "route_step": {
                        "step_id": "one-paid-call",
                        "product_smiles": request["expandable_smiles"][0],
                    },
                }
            ],
        }

    result = run_aizynthfinder_strategy_branch_sidecar(
        target_smiles="CO",
        strategy_id="unbilled-callback",
        strategy_text="callback and model call are distinct",
        request_handler=handler,
        inline_stock_smiles=("C", "O"),
        max_policy_calls=1,
        max_candidates_per_call=1,
        max_transforms=2,
        max_mcts_iterations=5,
        timeout_s=60,
    )

    assert result["solved"] is True
    assert result["policy_calls"] == 1
    assert result["diagnostics"]["provider_callback_count"] == 2


def test_strategy_sidecar_roundtrips_host_path_rejection_and_retries_parent() -> None:
    root_calls = 0

    def handler(request):
        nonlocal root_calls
        if request["depth"] == 0:
            root_calls += 1
            if root_calls == 1:
                return {
                    "candidates": [
                        {
                            "candidate_id": "rejected-root",
                            "product_smiles": "CCO",
                            "mapped_product_smiles": request[
                                "expandable_mapped_smiles"
                            ][0],
                            "precursor_smiles": ["CC", "O"],
                            "mapped_precursor_smiles": [
                                "[CH3:1][CH3:2]",
                                "[OH2:3]",
                            ],
                            "route_step": {
                                "step_id": "rejected-root",
                                "product_smiles": "CCO",
                            },
                        }
                    ]
                }
            return {
                "candidates": [
                    {
                        "candidate_id": "alternate-root",
                        "product_smiles": "CCO",
                        "mapped_product_smiles": request[
                            "expandable_mapped_smiles"
                        ][0],
                        "precursor_smiles": ["C", "O"],
                        "mapped_precursor_smiles": ["[CH4:1]", "[OH2:3]"],
                        "route_step": {
                            "step_id": "alternate-root",
                            "product_smiles": "CCO",
                        },
                    }
                ]
            }
        return {
            "candidates": [],
            "model_call_consumed": False,
            "rejected_path_step_ids": ["rejected-root"],
            "rejection_reason": "followup critic rejected the root action",
        }

    result = run_aizynthfinder_strategy_branch_sidecar(
        target_smiles="CCO",
        mapped_target_smiles="[CH3:1][CH2:2][OH:3]",
        strategy_id="path-rejection-roundtrip",
        strategy_text="retry the parent after rejecting one edge",
        request_handler=handler,
        inline_stock_smiles=("C", "O"),
        max_policy_calls=2,
        max_candidates_per_call=1,
        max_transforms=3,
        max_mcts_iterations=6,
        timeout_s=60,
    )

    assert result["solved"] is True
    assert result["policy_calls"] == 2
    assert result["diagnostics"]["provider_callback_count"] == 3
    assert result["diagnostics"]["path_rejection_count"] == 1
    assert [row["step_id"] for row in result["route_steps"]] == [
        "alternate-root"
    ]
