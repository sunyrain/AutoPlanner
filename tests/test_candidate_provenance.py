from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from cascade_planner.application.candidate_provenance import (
    compile_candidate_provenance,
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


def _candidate_record() -> dict:
    return _with_digest(
        {
            "schema_version": "canonical_candidate_lifecycle_record.v1",
            "candidate_id": "hypothesis:one",
            "canonical_entity_ids": ["edge:one", "hypothesis:one"],
            "edge_id": "edge:one",
            "edge_digest": "1" * 64,
            "product_smiles": "CCO",
            "precursor_smiles": ["CC", "O"],
            "status": "validated",
            "status_reason": "host_reaction_validation_accepted",
            "admission": {"accepted": True, "reasons": []},
            "materialization": {"materialized": True},
            "validation": {"accepted": True, "achieved_level": 2, "reasons": []},
            "evidence": {
                "exact_record_count": 1,
                "independent_source_groups": ["source:one"],
            },
            "conditions": {"prediction_count": 1},
            "portfolio": {
                "route_ids": ["route:one"],
                "pareto_route_ids": ["route:one"],
                "selected_route_ids": ["route:one"],
                "complete_route_ids": ["route:one"],
                "stock_closed_route_ids": ["route:one"],
                "accepted_route_ids": [],
            },
            "route_family_ids": ["route-family:one"],
            "origin_records": [
                {
                    "origin_kind": "chemenzy",
                    "proposal_id": "chemenzy:seed:route:1:step:1",
                    "route_family_id": "chemenzy:seed:route:1",
                    "origin_ref": "ChemEnzyRetroPlanner",
                }
            ],
            "semantics": {"open_proof_or_stock_axes_do_not_delete_topology": True},
        }
    )


def _lifecycle() -> dict:
    record = _candidate_record()
    return _with_digest(
        {
            "schema_version": "canonical_candidate_lifecycle.v1",
            "graph_revision": 3,
            "graph_scientific_sha256": "a" * 64,
            "portfolio_sha256": "b" * 64,
            "candidate_count": 1,
            "canonical_candidate_count": 1,
            "status_counts": {
                "rejected_invalid": 0,
                "quarantined_reviewable": 0,
                "admitted_unproved": 0,
                "validated": 1,
                "accepted": 0,
            },
            "ignored_ingestion_report_count": 0,
            "records": [record],
            "semantics": {"projection_is_read_only": True},
        }
    )


def _lineage() -> dict:
    return _with_digest(
        {
            "schema_version": "chemenzy_route_lineage.v1",
            "route_count": 2,
            "disposition_counts": {
                "host_portfolio_budget_truncated": 1,
                "stock_closed": 1,
            },
            "campaign_B4_stock_boundary": True,
            "routes": [
                {
                    "route_trace_id": "chemenzy-route:one",
                    "raw_route_sha256": "1" * 64,
                    "normalized_route_sha256": "2" * 64,
                    "proposal_eligible": True,
                    "host_portfolio_selected": True,
                    "preserved_as_advisory": False,
                    "disposition": "host_portfolio_selected",
                    "reasons": [],
                    "canonical_route_family_id": "route-family:one",
                    "step_proposal_ids": ["chemenzy:seed:route:1:step:1"],
                    "canonical_hypothesis_ids": ["hypothesis:one"],
                    "canonical_edge_ids": ["edge:one"],
                    "canonical_route_ids": ["route:one"],
                    "stock_closed_route_ids": ["route:one"],
                    "final_disposition": "stock_closed",
                },
                {
                    "route_trace_id": "chemenzy-route:two",
                    "raw_route_sha256": "3" * 64,
                    "normalized_route_sha256": "4" * 64,
                    "proposal_eligible": True,
                    "host_portfolio_selected": False,
                    "preserved_as_advisory": False,
                    "disposition": "host_portfolio_budget_truncated",
                    "reasons": [],
                    "canonical_route_family_id": "",
                    "step_proposal_ids": [],
                    "canonical_hypothesis_ids": [],
                    "canonical_edge_ids": [],
                    "canonical_route_ids": [],
                    "stock_closed_route_ids": [],
                    "final_disposition": "host_portfolio_budget_truncated",
                },
            ],
            "semantics": {"raw_normalized_canonical_bound_by_digest": True},
        }
    )


def test_candidate_provenance_binds_provider_routes_and_preserves_first_loss() -> None:
    provenance = compile_candidate_provenance(
        _lifecycle(),
        lineage_observations=[{"detail": _lineage()}],
    )

    assert provenance["candidate_record_count"] == 1
    assert provenance["provider_route_count"] == 2
    assert provenance["bound_provider_route_count"] == 1
    assert provenance["provider_only_route_count"] == 1
    assert provenance["first_loss_counts"] == {
        "host_portfolio_selection": 1,
        "none": 1,
    }
    candidate = provenance["candidate_records"][0]
    assert candidate["provider_normalization"]["raw_route_sha256"] == ["1" * 64]
    assert candidate["provider_normalization"]["normalized_route_sha256"] == [
        "2" * 64
    ]
    assert candidate["reaction_validation"]["accepted"] is True
    assert candidate["stock_closure"]["route_ids"] == ["route:one"]
    routes = {
        row["route_trace_id"]: row for row in provenance["provider_route_records"]
    }
    assert routes["chemenzy-route:one"]["candidate_ids"] == ["hypothesis:one"]
    assert routes["chemenzy-route:one"]["first_loss_boundary"] == "none"
    assert routes["chemenzy-route:two"]["candidate_ids"] == []
    assert (
        routes["chemenzy-route:two"]["first_loss_boundary"]
        == "host_portfolio_selection"
    )


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        ("host_admission", "host_admission"),
        ("host_quarantine", "host_quarantine"),
        ("canonical_ingestion", "canonical_ingestion"),
        ("canonical_materialization", "canonical_materialization"),
        ("reaction_validation", "reaction_validation"),
        ("stock_closure", "stock_closure"),
        ("none", "none"),
    ],
)
def test_candidate_provenance_reports_the_first_open_boundary(
    boundary: str,
    expected: str,
) -> None:
    lifecycle = _lifecycle()
    candidate = deepcopy(lifecycle["records"][0])
    lineage = _lineage()
    route = deepcopy(lineage["routes"][0])
    lineage["routes"] = [route]
    lineage["route_count"] = 1
    lineage["disposition_counts"] = {}
    if boundary == "host_admission":
        route["proposal_eligible"] = False
        route["host_portfolio_selected"] = False
    elif boundary == "host_quarantine":
        route["proposal_eligible"] = False
        route["host_portfolio_selected"] = False
        route["preserved_as_advisory"] = True
        route["quarantined"] = True
    elif boundary == "canonical_ingestion":
        route["step_proposal_ids"] = []
        route["canonical_hypothesis_ids"] = []
        route["canonical_edge_ids"] = []
    elif boundary == "canonical_materialization":
        candidate["materialization"] = {"materialized": False}
    elif boundary == "reaction_validation":
        candidate["validation"] = {"accepted": False, "achieved_level": 1}
    elif boundary == "stock_closure":
        candidate["portfolio"]["stock_closed_route_ids"] = []
    lifecycle["records"] = [_with_digest(candidate)]
    lifecycle = _with_digest(lifecycle)
    lineage = _with_digest(lineage)

    provenance = compile_candidate_provenance(
        lifecycle,
        lineage_observations=[lineage],
    )

    assert provenance["provider_route_records"][0]["first_loss_boundary"] == expected


def test_partial_step_binding_is_not_reported_as_complete_route_conservation() -> None:
    lifecycle = _lifecycle()
    lineage = _lineage()
    route = lineage["routes"][0]
    route.update(
        {
            "provider_step_count": 3,
            "normalized_step_count": 3,
            "imported_proposal_count": 2,
            "canonical_bound_step_count": 2,
            "topology_conservation_applicable": True,
            "topology_conservation_accepted": False,
            "missing_imported_proposal_ids": ["chemenzy:seed:route:1:step:3"],
            "missing_canonical_proposal_ids": [],
        }
    )
    lineage = _with_digest(lineage)

    provenance = compile_candidate_provenance(
        lifecycle,
        lineage_observations=[lineage],
    )

    assert provenance["bound_provider_route_count"] == 1
    assert provenance["fully_conserved_provider_route_count"] == 0
    assert provenance["partially_bound_provider_route_count"] == 1
    record = provenance["provider_route_records"][0]
    assert record["candidate_ids"]
    assert record["canonical_bound_step_count"] == 2
    assert record["first_loss_boundary"] == "canonical_topology_conservation"


def test_candidate_provenance_ignores_tampered_provider_lineage() -> None:
    tampered = _lineage()
    tampered["routes"][0]["raw_route_sha256"] = "f" * 64

    provenance = compile_candidate_provenance(
        _lifecycle(),
        lineage_observations=[tampered],
    )

    assert provenance["provider_route_count"] == 0
    assert provenance["ignored_provider_lineage_count"] == 1
    assert provenance["candidate_records"][0]["provider_normalization"] == {
        "route_trace_ids": [],
        "raw_route_sha256": [],
        "normalized_route_sha256": [],
        "lineage_sha256": [],
    }


def test_rejected_candidate_preserves_verified_ingestion_provenance() -> None:
    lifecycle = _lifecycle()
    rejected = _with_digest(
        {
            "schema_version": "canonical_candidate_lifecycle_record.v1",
            "candidate_id": "proposal:invalid",
            "canonical_entity_ids": [],
            "edge_id": "",
            "status": "rejected_invalid",
            "status_reason": "canonical_ingestion_rejected",
            "rejection_kind": "reaction_edge",
            "rejection_reasons": ["canonical_hypergraph_cycle"],
            "ingestion_report_sha256": ["d" * 64],
        }
    )
    lifecycle["records"].append(rejected)
    lifecycle["candidate_count"] = 2
    lifecycle["status_counts"]["rejected_invalid"] = 1
    lifecycle = _with_digest(lifecycle)

    provenance = compile_candidate_provenance(lifecycle)

    record = next(
        row
        for row in provenance["candidate_records"]
        if row["candidate_id"] == "proposal:invalid"
    )
    assert record["canonical_entity_ids"] == []
    assert record["ingestion_rejection"] == {
        "kind": "reaction_edge",
        "reasons": ["canonical_hypergraph_cycle"],
        "report_sha256": ["d" * 64],
    }


def test_candidate_provenance_fails_closed_for_tampered_lifecycle() -> None:
    lifecycle = _lifecycle()
    lifecycle["status_counts"]["accepted"] = 1

    with pytest.raises(ValueError, match="lifecycle_digest_invalid"):
        compile_candidate_provenance(lifecycle)
