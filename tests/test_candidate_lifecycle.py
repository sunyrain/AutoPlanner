from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from cascade_planner.application.candidate_lifecycle import (
    CANDIDATE_LIFECYCLE_STATUSES,
    compile_candidate_lifecycle,
)


def _with_digest(value: dict) -> dict:
    row = deepcopy(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


def _graph() -> dict:
    hypotheses = {}
    edges = {}
    for suffix, admitted, materialized in (
        ("quarantine", False, False),
        ("frontier", True, False),
        ("unproved", True, True),
        ("validated", True, True),
        ("accepted", True, True),
    ):
        edge_digest = hashlib.sha256(suffix.encode("utf-8")).hexdigest()
        hypothesis_id = f"hypothesis:{edge_digest}"
        hypotheses[hypothesis_id] = {
            "hypothesis_id": hypothesis_id,
            "edge_digest": edge_digest,
            "product_smiles": "CCO",
            "precursor_smiles": ["CC", "O"],
            "status": "materialized" if materialized else "frontier_candidate",
            "admission_accepted": admitted,
            "admission_reasons": [] if admitted else ["atom_jump_exceeds_limit"],
            "admission_audit_sha256": suffix * 4,
            "route_family_ids": [f"route-family:{suffix}"],
            "origin_records": [{"origin_kind": "codex", "origin_ref": suffix}],
        }
        if materialized:
            edge_id = f"edge:{edge_digest}"
            edges[edge_id] = {
                "edge_id": edge_id,
                "edge_digest": edge_digest,
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "route_family_ids": [f"route-family:{suffix}"],
                "origin_records": [
                    {"origin_kind": "codex", "origin_ref": suffix}
                ],
                "exact_record_ids": (["exact:1"] if suffix == "accepted" else []),
                "independent_source_groups": (
                    ["source-group:1"] if suffix == "accepted" else []
                ),
                "condition_predictions": (
                    [{"temperature_c": 20}] if suffix == "validated" else []
                ),
            }
    return {
        "revision": 7,
        "scientific_sha256": "a" * 64,
        "hypotheses": hypotheses,
        "edges": edges,
    }


def _portfolio(graph: dict, *, accepted: bool = True) -> dict:
    edge_ids = {
        suffix: "edge:" + hashlib.sha256(suffix.encode("utf-8")).hexdigest()
        for suffix in ("unproved", "validated", "accepted")
    }
    return _with_digest(
        {
            "schema_version": "proof_stitched_route_portfolio.v1",
            "graph_revision": graph["revision"],
            "graph_scientific_sha256": graph["scientific_sha256"],
            "edge_proofs": {
                edge_ids["unproved"]: {
                    "accepted": False,
                    "achieved_level": 1,
                    "reasons": ["reaction_validation_missing"],
                },
                edge_ids["validated"]: {
                    "accepted": True,
                    "achieved_level": 2,
                    "reasons": [],
                },
                edge_ids["accepted"]: {
                    "accepted": True,
                    "achieved_level": 3,
                    "reasons": [],
                    "independent_source_groups": ["source-group:1"],
                },
            },
            "route_candidates": [
                {
                    "route_id": "route:unproved",
                    "edge_ids": [edge_ids["unproved"]],
                    "pareto_optimal": False,
                    "selected": False,
                    "complete": False,
                    "all_leaves_stock_closed": False,
                },
                {
                    "route_id": "route:validated",
                    "edge_ids": [edge_ids["validated"]],
                    "pareto_optimal": True,
                    "selected": True,
                    "complete": False,
                    "all_leaves_stock_closed": True,
                },
                {
                    "route_id": "route:accepted",
                    "edge_ids": [edge_ids["accepted"]],
                    "pareto_optimal": True,
                    "selected": True,
                    "complete": True,
                    "all_leaves_stock_closed": True,
                },
            ],
            "selected_routes": [],
            "accepted": accepted,
        }
    )


def _ingestion_report(graph: dict) -> dict:
    return _with_digest(
        {
            "schema_version": "canonical_hypergraph_ingestion_report.v1",
            "changed": False,
            "evidence_changed": False,
            "revision": graph["revision"],
            "scientific_sha256": graph["scientific_sha256"],
            "dirty_entity_ids": [],
            "rejected": [
                {
                    "kind": "reaction_edge",
                    "proposal_id": "proposal:invalid",
                    "reasons": ["canonical_hypergraph_cycle"],
                },
                {
                    "kind": "reaction_proof",
                    "proposal_id": "proof:ignored",
                    "reasons": ["reaction_proof_not_replayable_or_edge_missing"],
                },
                {
                    "kind": "hypothesis",
                    "proposal_id": "proposal:retained",
                    "hypothesis_id": "hypothesis:retained",
                    "retained_as_l0": True,
                    "reasons": ["atom_jump_exceeds_limit"],
                },
            ],
            "semantics": {
                "single_ingestion_path": True,
                "rejected_inputs_did_not_mutate_graph": True,
                "incremental_projection": True,
            },
        }
    )


def test_candidate_lifecycle_projects_all_five_dispositions() -> None:
    graph = _graph()
    report = _ingestion_report(graph)
    tampered = deepcopy(report)
    tampered["rejected"][0]["proposal_id"] = "proposal:tampered"

    lifecycle = compile_candidate_lifecycle(
        graph,
        _portfolio(graph),
        ingestion_observations=(
            {"detail": {"ingestion": report}},
            {"detail": {"ingestion": tampered}},
        ),
    )

    assert tuple(lifecycle["status_counts"]) == CANDIDATE_LIFECYCLE_STATUSES
    assert lifecycle["status_counts"] == {
        "rejected_invalid": 1,
        "quarantined_reviewable": 1,
        "admitted_unproved": 2,
        "validated": 1,
        "accepted": 1,
    }
    assert lifecycle["candidate_count"] == 6
    assert lifecycle["canonical_candidate_count"] == 5
    assert lifecycle["ignored_ingestion_report_count"] == 1
    by_status = {
        status: [row for row in lifecycle["records"] if row["status"] == status]
        for status in CANDIDATE_LIFECYCLE_STATUSES
    }
    assert by_status["rejected_invalid"][0]["candidate_id"] == "proposal:invalid"
    assert by_status["quarantined_reviewable"][0]["materialization"] == {
        "materialized": False
    }
    assert {row["status_reason"] for row in by_status["admitted_unproved"]} == {
        "materialization_pending",
        "reaction_proof_open",
    }
    assert by_status["validated"][0]["conditions"]["prediction_count"] == 1
    assert by_status["accepted"][0]["portfolio"]["accepted_route_ids"] == [
        "route:accepted"
    ]


def test_candidate_acceptance_requires_configured_portfolio_acceptance() -> None:
    graph = _graph()
    lifecycle = compile_candidate_lifecycle(graph, _portfolio(graph, accepted=False))

    accepted_edge = "edge:" + hashlib.sha256(b"accepted").hexdigest()
    record = next(row for row in lifecycle["records"] if row["edge_id"] == accepted_edge)

    assert record["status"] == "validated"
    assert record["portfolio"]["accepted_route_ids"] == []


def test_candidate_lifecycle_fails_closed_on_portfolio_binding_errors() -> None:
    graph = _graph()
    tampered = _portfolio(graph)
    tampered["accepted"] = False
    with pytest.raises(ValueError, match="portfolio_digest_invalid"):
        compile_candidate_lifecycle(graph, tampered)

    mismatched = _portfolio(graph)
    mismatched["graph_revision"] = 8
    mismatched = _with_digest(mismatched)
    with pytest.raises(ValueError, match="graph_portfolio_mismatch"):
        compile_candidate_lifecycle(graph, mismatched)
