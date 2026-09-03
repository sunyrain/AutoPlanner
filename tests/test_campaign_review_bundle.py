from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from cascade_planner.application.campaign_review_bundle import (
    compile_campaign_review_bundle,
)
from cascade_planner.application.campaign_trajectory import (
    compile_campaign_snapshot,
    compile_campaign_trajectory,
)


def _with_digest(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _report() -> dict:
    snapshot = compile_campaign_snapshot(
        phase="closeout",
        observed_at="2026-08-10T00:00:00Z",
        event_sequence=2,
        graph_revision=1,
        wall_time_s=3.0,
        gates={
            "gates": {"B1_global_multi_route": True},
            "counts": {"target_rooted_distinct_skeletons": 1},
        },
        resource_usage={
            "model": {"model_invocations": 1},
            "settled_task_count": 2,
        },
        action_counts={"total": 1},
        route_counts={"target_rooted_route_count": 1},
        pareto_archive=[{"route_id": "route:1", "edge_ids": ["edge:1"]}],
    )
    return _with_digest({
        "run_id": "review-example",
        "trajectory": compile_campaign_trajectory([snapshot]),
        "candidate_lifecycle": _with_digest(
            {
                "schema_version": "canonical_candidate_lifecycle.v1",
                "graph_revision": 1,
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
                "records": [
                    {
                        "candidate_id": "edge:1",
                        "status": "validated",
                    }
                ],
                "semantics": {"projection_is_read_only": True},
            }
        ),
        "candidate_provenance": _with_digest(
            {
                "schema_version": "canonical_candidate_provenance.v1",
                "lifecycle_sha256": "c" * 64,
                "graph_revision": 1,
                "graph_scientific_sha256": "a" * 64,
                "candidate_record_count": 1,
                "provider_route_count": 1,
                "bound_provider_route_count": 1,
                "provider_only_route_count": 0,
                "ignored_provider_lineage_count": 0,
                "first_loss_counts": {"stock_closure": 1},
                "candidate_records": [
                    {"candidate_id": "edge:1", "status": "validated"}
                ],
                "provider_route_records": [
                    {
                        "route_trace_id": "chemenzy-route:1",
                        "first_loss_boundary": "stock_closure",
                    }
                ],
                "semantics": {"projection_is_read_only": True},
            }
        ),
        "stages": [
            {
                "stage": "campaign_action_unified_core_01",
                "status": "failed",
                "detail": {
                    "action": {
                        "execution_id": "action:1",
                        "kind": "reaction_validate",
                    },
                    "outcome": {
                        "action_execution_id": "action:1",
                        "status": "failed",
                        "failure_type": "host_rejected",
                        "failure_reasons": ["atom_balance_failed"],
                    },
                },
            },
            {
                "stage": "chemenzy_baseline",
                "status": "completed",
                "detail": {
                    "route_lineage": [
                        {
                            "raw_route_sha256": "1" * 64,
                            "normalized_route_sha256": "2" * 64,
                        }
                    ]
                },
            },
        ],
        "stop_decision": {
            "decision": "budget_exhausted",
            "terminal": True,
            "reasons": ["run_total_task_budget_exhausted"],
        },
    })


def test_review_bundle_exports_all_four_independently_hashed_traces() -> None:
    bundle = compile_campaign_review_bundle(_report())

    assert bundle["schema_version"] == "campaign_review_bundle.v1"
    assert set(bundle["components"]) == {
        "action_trace",
        "failure_trace",
        "route_lineage",
        "resource_curve",
    }
    assert all(len(value) == 64 for value in bundle["component_sha256"].values())
    assert bundle["components"]["action_trace"]["record_count"] == 1
    assert bundle["components"]["failure_trace"]["record_count"] == 2
    assert bundle["components"]["route_lineage"]["record_count"] == 4
    lifecycle = next(
        row
        for row in bundle["components"]["route_lineage"]["records"]
        if row["kind"] == "canonical_candidate_lifecycle"
    )
    assert lifecycle["available"] is True
    assert lifecycle["lifecycle"]["status_counts"]["validated"] == 1
    provenance = next(
        row
        for row in bundle["components"]["route_lineage"]["records"]
        if row["kind"] == "canonical_candidate_provenance"
    )
    assert provenance["available"] is True
    assert provenance["provenance"]["first_loss_counts"] == {"stock_closure": 1}
    assert bundle["components"]["resource_curve"]["available"] is True
    assert bundle["components"]["resource_curve"]["records"][0][
        "wall_time_s"
    ] == 3.0


def test_review_bundle_fails_closed_for_a_tampered_trajectory() -> None:
    report = _report()
    report["trajectory"] = deepcopy(report["trajectory"])
    report["trajectory"]["resource_curve"][0]["wall_time_s"] = 999.0
    report = _with_digest(report)

    bundle = compile_campaign_review_bundle(report)

    resource = bundle["components"]["resource_curve"]
    assert resource["available"] is False
    assert resource["records"] == []
    assert resource["unavailable_reason"] == "trajectory_digest_invalid_or_missing"


def test_review_bundle_fails_closed_for_a_tampered_report() -> None:
    report = _report()
    report["stages"][0]["detail"]["outcome"]["status"] = "completed"

    bundle = compile_campaign_review_bundle(report)

    assert bundle["available"] is False
    assert bundle["source_report_digest_valid"] is False
    assert bundle["components"]["action_trace"]["record_count"] == 0
    assert bundle["components"]["resource_curve"]["records"] == []
    assert bundle["unavailable_reason"] == "target_solve_report_digest_invalid"


def test_open_scientific_gate_is_not_exported_as_a_runtime_failure() -> None:
    report = _report()
    report["stages"].append(
        {
            "stage": "exact_evidence_binding",
            "status": "unresolved",
            "detail": {"reasons": ["exact_evidence_missing"]},
        }
    )
    report = _with_digest(report)

    bundle = compile_campaign_review_bundle(report)

    records = bundle["components"]["failure_trace"]["records"]
    assert all(record.get("stage") != "exact_evidence_binding" for record in records)


def test_review_bundle_fails_closed_for_tampered_candidate_lifecycle() -> None:
    report = _report()
    report["candidate_lifecycle"]["status_counts"]["accepted"] = 1
    report = _with_digest(report)

    bundle = compile_campaign_review_bundle(report)

    lifecycle = next(
        row
        for row in bundle["components"]["route_lineage"]["records"]
        if row["kind"] == "canonical_candidate_lifecycle"
    )
    assert lifecycle["available"] is False
    assert lifecycle["lifecycle"] == {}
    assert lifecycle["unavailable_reason"] == "candidate_lifecycle_digest_invalid"


def test_review_bundle_fails_closed_for_tampered_candidate_provenance() -> None:
    report = _report()
    report["candidate_provenance"]["first_loss_counts"] = {"none": 1}
    report = _with_digest(report)

    bundle = compile_campaign_review_bundle(report)

    provenance = next(
        row
        for row in bundle["components"]["route_lineage"]["records"]
        if row["kind"] == "canonical_candidate_provenance"
    )
    assert provenance["available"] is False
    assert provenance["provenance"] == {}
    assert provenance["unavailable_reason"] == "candidate_provenance_digest_invalid"
