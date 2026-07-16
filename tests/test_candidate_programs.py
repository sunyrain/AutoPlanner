from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path

import pytest

from cascade_planner.application.candidate_programs import (
    CandidateProgramError,
    candidate_program_projection_oracle,
    candidate_route_observation_from_workbench,
    project_candidate_route_to_programs,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
BUFOTALIN_OBSERVATION = ROOT / "benchmarks" / "bufotalin_candidate_route_observation.v1.json"
ATORVASTATIN_OBSERVATION = (
    ROOT / "benchmarks" / "atorvastatin_candidate_route_observation.v1.json"
)
ATORVASTATIN_PROJECTION = (
    ROOT / "benchmarks" / "atorvastatin_candidate_program_projection.v1.json"
)
STATIN_MIGRATION_AUDIT = ROOT / "benchmarks" / "statin_candidate_migration_audit.v1.json"
CROSS_CATEGORY_CASES = (
    (
        "Ibrutinib",
        ROOT / "benchmarks" / "ibrutinib_candidate_route_observation.v1.json",
        ROOT / "benchmarks" / "ibrutinib_candidate_program_projection.v1.json",
        17,
        12,
        3,
        5,
    ),
    (
        "Enzalutamide",
        ROOT / "benchmarks" / "enzalutamide_candidate_route_observation.v1.json",
        ROOT / "benchmarks" / "enzalutamide_candidate_program_projection.v1.json",
        16,
        10,
        3,
        4,
    ),
)


def _with_digest(value: dict) -> dict:
    row = copy.deepcopy(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _workbench() -> dict:
    value = {
        "schema_version": "retrosynthesis_route_workbench.v1",
        "run_id": "candidate-example",
        "revision": {"graph": 1, "evidence": 1},
        "target": {
            "molecule_id": "m:target",
            "canonical_smiles": "CCO",
            "name": "ethanol",
        },
        "molecules": {
            "m:target": {
                "molecule_id": "m:target",
                "canonical_smiles": "CCO",
                "label": "ethanol",
                "role": "target",
                "stock_closed": False,
            },
            "m:mid": {
                "molecule_id": "m:mid",
                "canonical_smiles": "CC=O",
                "label": "acetaldehyde",
                "role": "intermediate",
                "stock_closed": False,
            },
            "m:leaf": {
                "molecule_id": "m:leaf",
                "canonical_smiles": "C",
                "label": "incomplete precursor record",
                "role": "stock_leaf",
                "stock_closed": True,
            },
        },
        "edges": {
            "edge:reduction": {
                "edge_id": "edge:reduction",
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:mid"],
                "accepted": False,
                "proof_level": 1,
                "proof_vector": {
                    "sources": "single_group",
                    "reaction": "unvalidated",
                    "conditions": "source_recorded_unverified",
                },
            },
            "edge:gap": {
                "edge_id": "edge:gap",
                "product_molecule_id": "m:mid",
                "precursor_molecule_ids": ["m:leaf"],
                "accepted": False,
                "proof_level": 0,
                "proof_vector": {
                    "sources": "none",
                    "reaction": "incomplete",
                    "conditions": "missing",
                },
            },
        },
        "routes": {
            "route:candidate": {
                "route_id": "route:candidate",
                "edge_ids": ["edge:gap", "edge:reduction"],
                "root_edge_ids": ["edge:reduction"],
                "leaf_molecule_ids": ["m:leaf"],
                "complete": True,
                "closure_profile": "exploration_closed",
                "reported_source_refs": ["doi:10.example/source"],
                "warning_codes": ["reported_route_contains_unresolved_edges"],
            }
        },
        "inspectors": {
            "edges": {
                "edge:reduction": {
                    "condition_status": "source_recorded_unverified",
                    "rejection_reasons": ["reaction_validation_missing"],
                    "provenance": [{"origin_kind": "literature_visual_extraction"}],
                    "sources": [{"source_ref": "doi:10.example/source"}],
                    "source_observation_records": [
                        {
                            "record_id": "observation:reduction",
                            "source_ref": "doi:10.example/source",
                            "location_refs": ["Scheme 1"],
                            "conditions": {
                                "reagents": ["NaBH4"],
                                "solvent": ["MeOH"],
                                "temperature": "0 °C",
                            },
                            "condition_completeness": {"complete": False},
                        }
                    ],
                },
                "edge:gap": {
                    "condition_status": "missing",
                    "rejection_reasons": ["missing_atom_contributing_reactant"],
                },
            }
        },
    }
    value["content_sha256"] = strict_canonical_json_sha256(value)
    return value


def test_workbench_candidate_projection_preserves_gap_and_conditions() -> None:
    workbench = _workbench()
    workbench["molecules"]["m:target"]["canonical_smiles"] = "OCC"
    workbench["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in workbench.items() if key != "content_sha256"}
    )
    observation = candidate_route_observation_from_workbench(workbench)
    projection = project_candidate_route_to_programs(observation)
    oracle = candidate_program_projection_oracle(observation, projection)

    assert projection["counts"] == {
        "chemical_states": 3,
        "operation_nodes": 2,
        "programs": 2,
        "routes": 1,
        "canonical_admissible": 1,
        "inventory_gap": 1,
        "blocked_candidate": 0,
    }
    assert oracle["accepted"] is True
    assert observation["molecules"]["m:target"]["canonical_smiles"] == "CCO"
    route = projection["routes"]["route:candidate"]
    assert len(route["program_ids"]) == 2
    assert len(route["inventory_gap_program_ids"]) == 1
    assert route["source_exploration_closed"] is True
    assert route["production_closed"] is False
    assert route["accepted"] is False
    assert all(
        program["validation_vector"]["authoritative"] is False
        for program in projection["programs"].values()
    )
    reduction = next(
        row
        for row in projection["operation_nodes"].values()
        if row["source_transformation_id"] == "edge:reduction"
    )
    assert reduction["condition_observations"][0]["conditions"]["reagents"] == ["NaBH4"]


def test_candidate_projection_fails_closed_on_tamper_and_fatal_self_loop() -> None:
    workbench = _workbench()
    workbench["target"]["name"] = "tampered"
    with pytest.raises(CandidateProgramError, match="candidate_workbench_digest_invalid"):
        candidate_route_observation_from_workbench(workbench)

    observation = candidate_route_observation_from_workbench(_workbench())
    fatal = copy.deepcopy(observation)
    edge = dict(fatal["transformations"]["edge:gap"])
    edge["precursor_molecule_ids"] = ["m:mid"]
    fatal["transformations"]["edge:gap"] = _with_digest(edge)
    fatal = _with_digest(fatal)
    with pytest.raises(CandidateProgramError, match="candidate_route_edge_fatal:edge:gap"):
        project_candidate_route_to_programs(fatal)

    cycle = candidate_route_observation_from_workbench(_workbench())
    edge = dict(cycle["transformations"]["edge:gap"])
    edge["precursor_molecule_ids"] = ["m:target"]
    cycle["transformations"]["edge:gap"] = _with_digest(edge)
    cycle = _with_digest(cycle)
    with pytest.raises(CandidateProgramError, match="candidate_route_cycle:route:candidate"):
        project_candidate_route_to_programs(cycle)


def test_bufotalin_twenty_step_observation_projects_without_false_closure() -> None:
    observation = json.loads(BUFOTALIN_OBSERVATION.read_text(encoding="utf-8"))
    projection = project_candidate_route_to_programs(observation)
    oracle = candidate_program_projection_oracle(observation, projection)

    assert projection["counts"] == {
        "chemical_states": 21,
        "operation_nodes": 20,
        "programs": 20,
        "routes": 1,
        "canonical_admissible": 15,
        "inventory_gap": 5,
        "blocked_candidate": 0,
    }
    assert oracle["accepted"] is True
    route = next(iter(projection["routes"].values()))
    assert len(route["program_ids"]) == 20
    assert len(route["inventory_gap_program_ids"]) == 5
    assert route["source_exploration_closed"] is True
    assert route["production_closed"] is False
    assert route["accepted"] is False
    assert (
        sum(bool(row["condition_observations"]) for row in projection["operation_nodes"].values())
        == 15
    )


def test_atorvastatin_eleven_step_candidate_projects_from_frozen_observation() -> None:
    observation = json.loads(ATORVASTATIN_OBSERVATION.read_text(encoding="utf-8"))
    expected = json.loads(ATORVASTATIN_PROJECTION.read_text(encoding="utf-8"))
    projection = project_candidate_route_to_programs(observation)
    oracle = candidate_program_projection_oracle(observation, projection)

    assert projection == expected
    assert projection["counts"] == {
        "chemical_states": 22,
        "operation_nodes": 11,
        "programs": 11,
        "routes": 1,
        "canonical_admissible": 11,
        "inventory_gap": 0,
        "blocked_candidate": 0,
    }
    assert oracle["accepted"] is True
    route = next(iter(projection["routes"].values()))
    assert route["source_exploration_closed"] is True
    assert route["production_closed"] is False
    assert route["accepted"] is False
    assert not any(
        row["condition_observations"] for row in projection["operation_nodes"].values()
    )


def test_statin_migration_audit_keeps_all_twelve_entities_honest() -> None:
    audit = json.loads(STATIN_MIGRATION_AUDIT.read_text(encoding="utf-8"))
    observed_digest = audit.pop("content_sha256")
    records = audit["records"]

    assert observed_digest == strict_canonical_json_sha256(audit)
    assert audit["catalog_entity_count"] == 12
    assert len(records) == 12
    assert len({row["target_name"] for row in records}) == 12
    assert dict(Counter(row["classification"] for row in records)) == audit["summary"]
    ready = [row for row in records if row["classification"] == "candidate_projection_ready"]
    assert [row["target_name"] for row in ready] == ["atorvastatin"]
    assert ready[0]["candidate_projection"]["counts"]["programs"] == 11
    assert all(row["next_action"] for row in records)
    assert audit["semantics"]["empty_or_missing_assets_never_grant_route_closure"] is True


@pytest.mark.parametrize(
    ("target_name", "observation_path", "projection_path", "states", "programs", "routes", "max_steps"),
    CROSS_CATEGORY_CASES,
)
def test_cross_category_candidate_program_regressions(
    target_name: str,
    observation_path: Path,
    projection_path: Path,
    states: int,
    programs: int,
    routes: int,
    max_steps: int,
) -> None:
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    expected = json.loads(projection_path.read_text(encoding="utf-8"))
    projection = project_candidate_route_to_programs(observation)

    assert observation["target"]["name"] == target_name
    assert projection == expected
    assert projection["counts"] == {
        "chemical_states": states,
        "operation_nodes": programs,
        "programs": programs,
        "routes": routes,
        "canonical_admissible": programs,
        "inventory_gap": 0,
        "blocked_candidate": 0,
    }
    assert max(len(row["program_ids"]) for row in projection["routes"].values()) == max_steps
    assert all(row["production_closed"] is False for row in projection["routes"].values())
    assert all(row["accepted"] is False for row in projection["routes"].values())
    assert all(
        row["validation_vector"]["authoritative"] is False
        for row in projection["programs"].values()
    )
