from __future__ import annotations

from cascade_planner.eval.evaluate_paroutes_topk_proxy import _target_metrics
from cascade_planner.eval.rerank_native_routes_with_v4_value import (
    _routes_for_target,
)


def test_current_cascade_search_result_programs_are_exposed_as_routes() -> None:
    target = {
        "cascade_search": {
            "result_programs": [
                {
                    "rank": 1,
                    "route_steps": [
                        {
                            "rxn_smiles": "CC.O>>CCO",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CC", "O"],
                        }
                    ],
                }
            ]
        }
    }

    routes = _routes_for_target(target)

    assert routes[0]["steps"] == target["cascade_search"]["result_programs"][0][
        "route_steps"
    ]
    metrics = _target_metrics(
        {"target_smiles": "CCO", **target},
        {
            "target_smiles": "CCO",
            "gt_route": [{"rxn_smiles": "CC.O>>CCO"}],
        },
    )
    assert metrics["n_routes"] == 1
    assert metrics["topk"]["1"]["exact_reaction_set_hit"] is True
    assert metrics["topk"]["1"]["best_leaf_overlap"] == 1.0


def test_paroutes_proxy_ignores_reference_atom_map_annotations() -> None:
    metrics = _target_metrics(
        {
            "target_smiles": "CCO",
            "cascade_search": {
                "result_programs": [
                    {
                        "route_steps": [
                            {
                                "rxn_smiles": "CC.O>>CCO",
                                "product_smiles": "CCO",
                                "reactant_smiles": ["CC", "O"],
                            }
                        ]
                    }
                ]
            },
        },
        {
            "target_smiles": "CCO",
            "gt_route": [
                {"rxn_smiles": "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"}
            ],
        },
    )

    assert metrics["topk"]["1"]["exact_reaction_set_hit"] is True
    assert metrics["topk"]["1"]["best_leaf_overlap"] == 1.0
