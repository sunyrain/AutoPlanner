from __future__ import annotations

import copy

from cascade_planner.interfaces.candidate_migration import (
    CANDIDATE_MIGRATION_AUDIT_SCHEMA,
    audit_candidate_workbench_snapshots,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _ready_workbench() -> dict:
    value = {
        "schema_version": "retrosynthesis_route_workbench.v1",
        "run_id": "candidate-audit-ready",
        "revision": {"graph": 1, "evidence": 1},
        "target": {
            "molecule_id": "m:target",
            "canonical_smiles": "CCO",
            "name": "ready-target",
        },
        "molecules": {
            "m:target": {
                "molecule_id": "m:target",
                "canonical_smiles": "CCO",
                "label": "target",
                "role": "target",
                "stock_closed": False,
            },
            "m:leaf": {
                "molecule_id": "m:leaf",
                "canonical_smiles": "CC=O",
                "label": "leaf",
                "role": "stock_leaf",
                "stock_closed": True,
            },
        },
        "edges": {
            "edge:one": {
                "edge_id": "edge:one",
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:leaf"],
                "accepted": False,
                "proof_level": 0,
                "proof_vector": {},
            }
        },
        "routes": {
            "route:one": {
                "route_id": "route:one",
                "edge_ids": ["edge:one"],
                "root_edge_ids": ["edge:one"],
                "leaf_molecule_ids": ["m:leaf"],
                "complete": True,
                "closure_profile": "exploration_closed",
                "reported_source_refs": [],
                "warning_codes": [],
            }
        },
        "inspectors": {"edges": {"edge:one": {}}},
    }
    value["content_sha256"] = strict_canonical_json_sha256(value)
    return value


def _empty_workbench() -> dict:
    value = _ready_workbench()
    value["run_id"] = "candidate-audit-empty"
    value["target"]["name"] = "empty-target"
    value["molecules"] = {"m:target": value["molecules"]["m:target"]}
    value["edges"] = {}
    value["routes"] = {}
    value["inspectors"] = {"edges": {}}
    value.pop("content_sha256")
    value["content_sha256"] = strict_canonical_json_sha256(value)
    return value


def test_candidate_migration_audit_deduplicates_and_classifies() -> None:
    ready = _ready_workbench()
    empty = _empty_workbench()
    tampered = copy.deepcopy(ready)
    tampered["target"]["name"] = "tampered"

    report = audit_candidate_workbench_snapshots(
        [
            ("ready-a.json", ready),
            ("ready-copy.json", ready),
            ("empty.json", empty),
            ("tampered.json", tampered),
        ]
    )

    observed_digest = report.pop("content_sha256")
    assert observed_digest == strict_canonical_json_sha256(report)
    assert report["schema_version"] == CANDIDATE_MIGRATION_AUDIT_SCHEMA
    assert report["snapshot_count"] == 4
    assert report["unique_workbench_count"] == 3
    assert report["duplicate_snapshot_count"] == 1
    assert report["target_count"] == 3
    assert report["migration_state_counts"] == {
        "projection_ready": 1,
        "empty_graph": 1,
        "invalid_snapshot": 1,
        "error": 0,
    }
    ready_row = next(
        row for row in report["workbenches"] if row["migration_state"] == "projection_ready"
    )
    assert ready_row["source_refs"] == ["ready-a.json", "ready-copy.json"]
    assert ready_row["program_counts"]["programs"] == 1
    assert ready_row["source_diagnostics"]["accepted_edge_count"] == 0
    assert ready_row["source_diagnostics"]["complete_route_count"] == 1
    assert ready_row["source_diagnostics"]["max_route_steps"] == 1
    invalid_row = next(
        row for row in report["workbenches"] if row["migration_state"] == "invalid_snapshot"
    )
    assert "candidate_workbench_digest_invalid" in invalid_row["error"]


def test_candidate_migration_audit_is_empty_and_read_only() -> None:
    report = audit_candidate_workbench_snapshots([])

    assert report["snapshot_count"] == 0
    assert report["unique_workbench_count"] == 0
    assert report["migration_state_counts"] == {
        "projection_ready": 0,
        "empty_graph": 0,
        "invalid_snapshot": 0,
        "error": 0,
    }
    assert report["semantics"]["read_only"] is True
    assert report["semantics"]["program_store_admission_performed"] is False
