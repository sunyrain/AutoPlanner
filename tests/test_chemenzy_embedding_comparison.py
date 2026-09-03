from __future__ import annotations

from copy import deepcopy

from cascade_planner.interfaces.chemenzy_probe import (
    compile_chemenzy_route_fingerprints,
)
from scripts.compare_chemenzy_embedding import compare_embedding


TARGET = "CCO"


def _standalone() -> dict:
    return {
        "status": "completed",
        "routes": [
            {
                "score": 1.0,
                "steps": [
                    {
                        "product_smiles": TARGET,
                        "reactant_smiles": ["CC"],
                    },
                    {
                        "product_smiles": "CC",
                        "reactant_smiles": ["C"],
                    },
                ],
            }
        ],
    }


def _report(final_row: dict) -> dict:
    fingerprints = compile_chemenzy_route_fingerprints(
        _standalone(),
        target_smiles=TARGET,
    )
    seed_row = {
        **fingerprints["routes"][0],
        "host_portfolio_selected": True,
        "disposition": "host_portfolio_selected",
    }
    return {
        "target": {"canonical_smiles": TARGET},
        "stages": [
            {
                "stage": "chemenzy_baseline",
                "detail": {
                    "raw_proposal_sha256": fingerprints["raw_proposal_sha256"],
                    "raw_result_sha256": fingerprints["raw_result_sha256"],
                    "route_lineage": [seed_row],
                },
            },
            {
                "stage": "chemenzy_route_lineage",
                "detail": {
                    "routes": [
                        {
                            **seed_row,
                            "canonical_route_family_id": "route-family:one",
                            "step_proposal_ids": ["step:1", "step:2"],
                            "canonical_hypothesis_ids": [
                                "hypothesis:1",
                                "hypothesis:2",
                            ],
                            **final_row,
                        }
                    ]
                },
            },
        ],
    }


def test_embedding_comparison_separates_materialization_validation_and_stock() -> None:
    partial = compare_embedding(
        _standalone(),
        _report(
            {
                "canonical_edge_ids": ["edge:1"],
                "canonical_minimum_proof_level": 1,
                "final_disposition": "canonical_hypothesis_only",
            }
        ),
    )
    rejected = compare_embedding(
        _standalone(),
        _report(
            {
                "canonical_edge_ids": ["edge:1", "edge:2"],
                "canonical_minimum_proof_level": 1,
                "final_disposition": "materialized_or_partially_materialized",
            }
        ),
    )
    validated = compare_embedding(
        _standalone(),
        _report(
            {
                "canonical_edge_ids": ["edge:1", "edge:2"],
                "canonical_minimum_proof_level": 2,
                "final_disposition": "materialized_or_partially_materialized",
            }
        ),
    )

    assert partial["first_loss_counts"] == {
        "canonical_hypotheses_not_fully_materialized": 1
    }
    assert rejected["first_loss_counts"] == {
        "materialized_not_host_validated": 1
    }
    assert validated["first_loss_counts"] == {
        "host_validated_not_stock_closed": 1
    }
    assert validated["raw_proposal_digest_equal"] is True
    assert validated["counts"]["fully_materialized_routes"] == 1
    assert validated["counts"]["host_validated_routes"] == 1


def test_embedding_comparison_does_not_mutate_input_report() -> None:
    report = _report(
        {
            "canonical_edge_ids": ["edge:1", "edge:2"],
            "canonical_minimum_proof_level": 2,
            "final_disposition": "stock_closed",
        }
    )
    before = deepcopy(report)

    result = compare_embedding(_standalone(), report)

    assert result["first_loss_counts"] == {"stock_closed": 1}
    assert report == before
