from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "cross_category_program_regression_set.v1.json"


def test_cross_category_program_regression_manifest_is_digest_bound() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed_digest = manifest.pop("content_sha256")
    cases = manifest["cases"]

    assert observed_digest == strict_canonical_json_sha256(manifest)
    assert len(cases) == 7
    assert len({row["target_name"] for row in cases}) == 7
    assert len({row["category"] for row in cases}) == 7
    assert dict(Counter(row["evidence_class"] for row in cases)) == {
        "canonical_replay_store": 3,
        "candidate_projection": 4,
    }
    assert manifest["coverage"] == {
        "target_categories": 7,
        "canonical_replay_store_categories": 3,
        "candidate_projection_categories": 4,
        "negative_control_categories": 1,
    }
    assert manifest["gate"]["program_ids_shadow_gate_satisfied"] is False
    assert manifest["gate"]["reasons"] == [
        "four_categories_are_candidate_projection_only"
    ]
    assert manifest["semantics"][
        "three_current_route_ui_dual_read_oracles_have_passed"
    ] is True
    assert {row["control_kind"] for row in manifest["enzyme_screen_controls"]} == {
        "positive",
        "no_applicable_enzyme",
    }


def test_cross_category_program_regression_artifacts_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for row in manifest["cases"]:
        path = ROOT / row["artifact"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["artifact_file_sha256"]
        assert payload["content_sha256"] == row["artifact_sha256"]
        if row["evidence_class"] == "candidate_projection":
            assert payload["counts"]["chemical_states"] == row["counts"]["chemical_states"]
            assert payload["counts"]["programs"] == row["counts"]["programs"]
            assert payload["counts"]["routes"] == row["counts"]["routes"]
            assert all(route["production_closed"] is False for route in payload["routes"].values())
            assert all(route["accepted"] is False for route in payload["routes"].values())
        if row["target_name"] == "fluvastatin":
            assert payload["source_run"]["accepted_under_configured_policy"] is False
            assert payload["program_store"]["counts"] == {
                "chemical_states": row["counts"]["chemical_states"],
                "operation_nodes": row["counts"]["programs"],
                "programs": row["counts"]["programs"],
                "routes": row["counts"]["routes"],
            }
            assert payload["program_store"]["store_oracle_accepted"] is True
            assert payload["route_program_dual_read"]["oracle_accepted"] is True

    for row in manifest["enzyme_screen_controls"]:
        path = ROOT / row["artifact"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["artifact_file_sha256"]
        assert payload["content_sha256"] == row["artifact_sha256"]
    controls = {row["control_kind"]: row for row in manifest["enzyme_screen_controls"]}
    assert controls["positive"]["maximum_replaced_chemical_steps"] == 6
    assert controls["no_applicable_enzyme"]["enzyme_candidate_count"] == 0
    assert controls["no_applicable_enzyme"]["screenable_route_count"] == 3
