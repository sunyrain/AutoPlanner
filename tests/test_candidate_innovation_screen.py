from __future__ import annotations

import copy
import json
from pathlib import Path

from cascade_planner.application.candidate_innovation_screen import (
    screen_candidate_route_innovations,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = json.loads(
    (ROOT / "config" / "route_innovation_capabilities.v1.json").read_text(encoding="utf-8")
)


def _load(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / name).read_text(encoding="utf-8"))


def test_bufotalin_candidate_screen_finds_six_to_one_enzyme_superstep() -> None:
    observation = _load("bufotalin_candidate_route_observation.v1.json")
    expected = _load("bufotalin_candidate_innovation_screen.v1.json")
    screen = screen_candidate_route_innovations(observation, capabilities=CAPABILITIES)

    assert screen == expected
    assert screen["counts"] == {
        "routes": 1,
        "screenable_routes": 1,
        "no_applicable_enzyme_routes": 0,
        "enzyme_candidates": 5,
        "mechanism_candidates": 0,
    }
    route_screen = next(iter(screen["route_screens"].values()))
    superstep = next(
        row
        for row in route_screen["discovery"]["candidates"]
        if len(row["boundary"]["replaced_edge_ids"]) == 6
    )
    assert superstep["route_innovation"]["step_savings"] == 5
    assert superstep["review_status"] == "ready_for_enzyme_screen"
    assert superstep["boundary"]["minimum_boundary_proof_level"] == 1
    assert superstep["warning_codes"] == ["EXACT_SUBSTRATE_UNVALIDATED"]


def test_ibrutinib_is_real_no_applicable_enzyme_negative_control() -> None:
    observation = _load("ibrutinib_candidate_route_observation.v1.json")
    expected = _load("ibrutinib_candidate_innovation_negative_control.v1.json")
    screen = screen_candidate_route_innovations(observation, capabilities=CAPABILITIES)

    assert screen == expected
    assert screen["accepted_capability_count"] == 3
    assert screen["counts"] == {
        "routes": 3,
        "screenable_routes": 3,
        "no_applicable_enzyme_routes": 3,
        "enzyme_candidates": 0,
        "mechanism_candidates": 0,
    }
    assert all(
        row["screen_status"] == "no_applicable_enzyme_capability"
        and row["negative_control_eligible"] is True
        for row in screen["route_screens"].values()
    )


def test_candidate_enzyme_screen_does_not_match_on_target_name() -> None:
    observation = _load("ibrutinib_candidate_route_observation.v1.json")
    renamed = copy.deepcopy(observation)
    renamed["target"]["name"] = "unrelated regression label"
    renamed.pop("content_sha256")
    renamed["content_sha256"] = strict_canonical_json_sha256(renamed)

    original = screen_candidate_route_innovations(observation, capabilities=CAPABILITIES)
    observed = screen_candidate_route_innovations(renamed, capabilities=CAPABILITIES)

    assert observed["counts"] == original["counts"]
    assert {
        route_id: row["screen_status"] for route_id, row in observed["route_screens"].items()
    } == {
        route_id: row["screen_status"] for route_id, row in original["route_screens"].items()
    }
    assert observed["semantics"]["target_names_are_not_matching_inputs"] is True


def test_empty_capability_catalog_cannot_create_a_negative_control() -> None:
    observation = _load("ibrutinib_candidate_route_observation.v1.json")
    screen = screen_candidate_route_innovations(
        observation,
        capabilities={"capabilities": []},
    )

    assert screen["accepted_capability_count"] == 0
    assert screen["accepted_capability_ids"] == []
    assert screen["counts"]["no_applicable_enzyme_routes"] == 0
    assert all(
        row["screen_status"] == "no_accepted_capabilities"
        and row["negative_control_eligible"] is False
        for row in screen["route_screens"].values()
    )
