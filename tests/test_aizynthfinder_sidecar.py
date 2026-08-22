from __future__ import annotations

from typing import Any

from cascade_planner.interfaces.aizynthfinder_sidecar import (
    run_aizynthfinder_guided_frontier_stage,
)
from cascade_planner.application.route_edge_scope import guided_provider_group_ids


class _Service:
    def __init__(self) -> None:
        self.batches: list[Any] = []

    def apply_batch(self, batch: Any, *, idempotency_key: str) -> dict[str, Any]:
        self.batches.append((batch, idempotency_key))
        return {"changed": True}


def _provider(**_kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "aizynthfinder_paper_search.v1",
        "status": "completed",
        "engine": "AiZynthFinder 4.4.1",
        "mode": "short_tail",
        "solved": True,
        "search_executed": True,
        "provider_invocation_count": 1,
        "statistics": {"iterations": 2},
        "proposal_routes": [
            {
                "route_trace_id": "aizynthfinder:test",
                "steps": [
                    {
                        "product_smiles": "CCOC(=O)C",
                        "reactant_smiles": ["CCO", "CC(=O)OC(C)=O"],
                        "source_model": "AiZynthFinder:uspto",
                        "policy_probability": 0.4,
                        "template_hash": "abc",
                        "reactant_stock_status": [True, True],
                    }
                ],
            }
        ],
    }


def test_guided_frontier_ingests_aizynthfinder_provenance() -> None:
    service = _Service()
    result = run_aizynthfinder_guided_frontier_stage(
        service,
        frontier_smiles="CCOC(=O)C",
        parent_route_family_ids=("route:parent",),
        provider=_provider,
    )

    assert result["status"] == "completed"
    assert result["provider_id"] == "aizynthfinder"
    assert result["proposal_count"] == 1
    assert result["selected_proposal_route_count"] == 1
    assert result["budget_truncated_route_count"] == 0
    assert result["route_lineage"][0]["route_trace_id"] == "aizynthfinder:test"
    assert result["route_lineage"][0]["step_proposal_ids"]
    batch, idempotency_key = service.batches[0]
    hypothesis = batch.hypotheses[0]
    assert hypothesis["origin_kind"] == "aizynthfinder"
    assert hypothesis["canonical_route_family_ids"] == ["route:parent"]
    assert hypothesis["precursor_smiles"] == ["CCO", "CC(=O)OC(C)=O"]
    assert guided_provider_group_ids(
        {"origin_records": [hypothesis]}
    ) == (hypothesis["route_family_id"],)
    assert "aizynthfinder:guided-" in idempotency_key


def test_guided_frontier_selects_one_coherent_partial_route() -> None:
    service = _Service()

    def two_routes(**_kwargs: Any) -> dict[str, Any]:
        payload = _provider()
        payload["solved"] = False
        payload["proposal_routes"][0]["all_leaves_in_provider_stock"] = False
        payload["proposal_routes"].append(
            {
                "route_trace_id": "aizynthfinder:second",
                "all_leaves_in_provider_stock": False,
                "steps": [
                    {
                        "product_smiles": "CCOC(=O)C",
                        "reactant_smiles": ["CCOC(=O)Cl"],
                        "source_model": "AiZynthFinder:uspto",
                    }
                ],
            }
        )
        return payload

    result = run_aizynthfinder_guided_frontier_stage(
        service,
        frontier_smiles="CCOC(=O)C",
        parent_route_family_ids=("route:parent",),
        provider=two_routes,
    )

    assert result["accepted_route_count"] == 2
    assert result["selected_proposal_route_count"] == 1
    assert result["budget_truncated_route_count"] == 1
    assert result["proposal_count"] == 1
    assert result["route_lineage"][0]["route_trace_id"] == "aizynthfinder:test"
    assert len(service.batches[0][0].hypotheses) == 1


def test_paper_guided_frontier_does_not_ingest_partial_route() -> None:
    service = _Service()

    def partial_only(**_kwargs: Any) -> dict[str, Any]:
        payload = _provider()
        payload["solved"] = False
        payload["proposal_routes"][0]["all_leaves_in_provider_stock"] = False
        return payload

    result = run_aizynthfinder_guided_frontier_stage(
        service,
        frontier_smiles="CCOC(=O)C",
        parent_route_family_ids=("route:parent",),
        provider=partial_only,
        accept_partial_routes=False,
    )

    assert result["status"] == "unresolved"
    assert result["accepted_route_count"] == 1
    assert result["complete_provider_route_count"] == 0
    assert result["selected_proposal_route_count"] == 0
    assert result["proposal_count"] == 0
    assert result["partial_route_ingestion_allowed"] is False
    assert result["reason"] == "paper_short_tail_no_complete_stock_closed_route"
    assert result["semantics"]["paper_matched_complete_route_required"] is True
    assert service.batches == []


def test_disconnected_aizynthfinder_route_is_rejected_before_ingestion() -> None:
    service = _Service()

    def disconnected(**_kwargs: Any) -> dict[str, Any]:
        payload = _provider()
        payload["proposal_routes"][0]["steps"].append(
            {
                "product_smiles": "N#N",
                "reactant_smiles": ["N", "N"],
                "source_model": "AiZynthFinder:uspto",
            }
        )
        return payload

    result = run_aizynthfinder_guided_frontier_stage(
        service,
        frontier_smiles="CCOC(=O)C",
        parent_route_family_ids=("route:parent",),
        provider=disconnected,
    )

    assert result["status"] == "unresolved"
    assert result["proposal_count"] == 0
    assert result["rejected_route_count"] == 1
    assert result["rejected_routes"][0]["reasons"] == [
        "aizynthfinder_route_step_disconnected"
    ]
    assert service.batches == []
