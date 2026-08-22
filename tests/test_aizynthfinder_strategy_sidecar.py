from __future__ import annotations

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
