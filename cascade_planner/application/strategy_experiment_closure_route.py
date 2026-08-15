"""Per-route projection for strategy-to-experiment closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cascade_planner.application.route_workbench_edge_proof_vector import (
    edge_proof_vector,
)


def compile_imported_route_closure(
    imported: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    hypotheses = dict(graph.get("hypotheses") or {})
    edges = dict(graph.get("edges") or {})
    proofs = dict(portfolio.get("edge_proofs") or {})
    families = dict(graph.get("route_families") or {})
    candidates = list(portfolio.get("route_candidates") or [])
    alias = str(imported.get("route_family_alias") or "")
    family_id = next(
        (
            str(key)
            for key, row in families.items()
            if alias in set(dict(row).get("aliases") or [])
        ),
        "",
    )
    expected_hypotheses = [str(v) for v in imported.get("hypothesis_ids") or []]
    expected_edges = [str(v) for v in imported.get("edge_ids") or []]
    admitted_hypotheses = [v for v in expected_hypotheses if v in hypotheses]
    materialized_edges = [v for v in expected_edges if v in edges]
    blockers = _materialization_blockers(
        expected_hypotheses,
        expected_edges,
        hypotheses=hypotheses,
        edges=edges,
    )
    vectors = [
        edge_proof_vector(edge=edges[e], proof=proofs.get(e, {}), graph=graph)
        for e in materialized_edges
    ]
    candidate = next(
        (
            dict(row)
            for row in candidates
            if str(row.get("route_family_id") or "") == family_id
        ),
        {},
    )
    materialized = bool(expected_edges) and len(materialized_edges) == len(
        expected_edges
    )
    return {
        "external_route_id": str(imported.get("external_route_id") or ""),
        "route_family_id": family_id,
        "strategy_structure": _axis(
            len(admitted_hypotheses) == len(expected_hypotheses),
            len(admitted_hypotheses),
            len(expected_hypotheses),
        ),
        "canonical_materialization": _axis(
            materialized,
            len(materialized_edges),
            len(expected_edges),
            blockers=blockers,
        ),
        "host_reaction_validation": _axis(
            materialized
            and all(
                proofs[e].get("reaction_validated") is True for e in expected_edges
            ),
            sum(
                proofs[e].get("reaction_validated") is True for e in materialized_edges
            ),
            len(expected_edges),
        ),
        "exact_source_evidence": _axis(
            materialized
            and all(
                proofs[e].get("exact_source_bound") is True for e in expected_edges
            ),
            sum(
                proofs[e].get("exact_source_bound") is True for e in materialized_edges
            ),
            len(expected_edges),
        ),
        "complete_exact_conditions": _axis(
            materialized
            and bool(vectors)
            and all(v.get("condition_completeness") == "complete" for v in vectors),
            sum(v.get("condition_completeness") == "complete" for v in vectors),
            len(expected_edges),
        ),
        "stock_closure": {
            "status": (
                "complete"
                if candidate.get("all_leaves_stock_closed") is True
                else "open"
            ),
            "rate": float(candidate.get("stock_closure_rate") or 0.0),
        },
        "experimental_validation": {
            "status": "not_assessed_by_route_import",
            "authority_required": "host_bound_experimental_program_result",
        },
    }


def _materialization_blockers(
    hypothesis_ids: list[str],
    edge_ids: list[str],
    *,
    hypotheses: Mapping[str, Any],
    edges: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for hypothesis_id, edge_id in zip(hypothesis_ids, edge_ids, strict=True):
        if edge_id in edges:
            continue
        hypothesis = dict(hypotheses.get(hypothesis_id) or {})
        result.append(
            {
                "hypothesis_id": hypothesis_id,
                "edge_id": edge_id,
                "status": str(hypothesis.get("status") or "hypothesis_missing"),
                "reasons": sorted(
                    {
                        str(value)
                        for value in hypothesis.get("admission_reasons") or []
                        if str(value)
                    }
                ),
            }
        )
    return result


def _axis(
    complete: bool,
    achieved: int,
    required: int,
    *,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "status": "complete" if complete else "open",
        "achieved": int(achieved),
        "required": int(required),
    }
    if blockers:
        result["blockers"] = blockers
    return result


__all__ = ["compile_imported_route_closure"]
