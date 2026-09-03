from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cascade_planner.application.biocatalytic_programs import (
    compile_biocatalytic_program_bundle,
)
from cascade_planner.application.program_route_candidate_contracts import (
    ProgramRouteCandidateError,
)
from cascade_planner.application.program_route_candidates import (
    compile_program_route_candidate_set,
)
from cascade_planner.application.program_route_optimizer import (
    optimize_program_route_candidates,
    program_route_portfolio_oracle,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
BUFOTALIN_OBSERVATION = (
    ROOT / "benchmarks" / "bufotalin_candidate_route_observation.v1.json"
)
BUFOTALIN_PROJECTION = (
    ROOT / "benchmarks" / "bufotalin_candidate_program_projection.v1.json"
)
ATORVASTATIN_OBSERVATION = (
    ROOT / "benchmarks" / "atorvastatin_candidate_route_observation.v1.json"
)
ATORVASTATIN_PROJECTION = (
    ROOT / "benchmarks" / "atorvastatin_candidate_program_projection.v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack(observation_path: Path, projection_path: Path) -> dict:
    return {
        "schema_version": "reported_program_route_pack.v1",
        "observation": _load(observation_path),
        "projection": _load(projection_path),
        "route_ids": [],
    }


def _current_materials(target_smiles: str) -> tuple[dict, dict, dict, dict, dict]:
    graph = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "reported-candidate-fixture",
        "revision": 1,
        "scientific_sha256": "reported-candidate-fixture-revision-1",
        "target_molecule_id": "m:target",
        "molecules": {
            "m:target": {
                "canonical_smiles": target_smiles,
                "stock_observation_ids": [],
            },
            "m:baseline-leaf": {
                "canonical_smiles": "C",
                "stock_observation_ids": [],
            },
        },
        "edges": {
            "edge:baseline": {
                "precursor_molecule_ids": ["m:baseline-leaf"],
                "product_molecule_id": "m:target",
                "innovation_boundary_proof_level": 0,
                "procedure_record_ids": [],
                "source_binding_ids": [],
                "exact_record_ids": [],
                "reaction_proofs": [],
            }
        },
        "route_families": {
            "family:baseline": {"edge_ids": ["edge:baseline"], "closed": False}
        },
    }
    route = {
        "route_id": "route:baseline",
        "route_family_id": "family:baseline",
        "edge_ids": ["edge:baseline"],
        "reported_source_refs": [],
    }
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(graph, route, capabilities=[])
    bundle = compile_biocatalytic_program_bundle(
        graph, route, projection, discovery
    )
    return graph, route, projection, discovery, bundle


def test_reported_bufotalin_route_enters_exploration_without_false_promotion() -> None:
    pack = _pack(BUFOTALIN_OBSERVATION, BUFOTALIN_PROJECTION)
    target_state = pack["projection"]["chemical_states"][
        pack["projection"]["target_state_id"]
    ]
    graph, route, projection, discovery, bundle = _current_materials(
        target_state["canonical_smiles"]
    )

    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
        reported_candidate_packs=[pack],
    )
    portfolio = optimize_program_route_candidates(candidate_set)
    oracle = program_route_portfolio_oracle(candidate_set, portfolio)
    reported_id = next(
        candidate_id
        for candidate_id, row in candidate_set["candidates"].items()
        if row["source_kind"] == "literature"
    )
    baseline_id = next(
        candidate_id
        for candidate_id, row in candidate_set["candidates"].items()
        if row["source_kind"] == "baseline"
    )
    reported = candidate_set["candidates"][reported_id]

    assert candidate_set["counts"]["candidates"] == 2
    assert candidate_set["counts"]["literature"] == 1
    assert len(reported["program_ids"]) == 20
    assert reported["source_route_id"] == "route:bufotalin-20-step-reported-candidate"
    assert reported["evidence"]["source_refs"] == [
        "doi:10.1016/j.tet.2025.134610"
    ]
    assert len(reported["evidence"]["source_artifact_sha256s"]) == 2
    assert reported["eligibility"]["exploration_visible"] is True
    assert reported["eligibility"]["shadow_optimizer"] is False
    assert reported["eligibility"]["production_authoritative"] is False
    assert reported["eligibility"]["route_completion"] is False
    assert reported_id in portfolio["profiles"]["exploration"][
        "eligible_candidate_ids"
    ]
    assert portfolio["profiles"]["shadow_optimizer"]["pareto_front_ids"] == [
        baseline_id
    ]
    assert oracle["accepted"] is True


def test_reported_candidate_adapter_is_target_structural_and_fails_closed() -> None:
    bufotalin_pack = _pack(BUFOTALIN_OBSERVATION, BUFOTALIN_PROJECTION)
    target_state = bufotalin_pack["projection"]["chemical_states"][
        bufotalin_pack["projection"]["target_state_id"]
    ]
    graph, route, projection, discovery, bundle = _current_materials(
        target_state["canonical_smiles"]
    )
    renamed = deepcopy(graph)
    renamed["target_name"] = "opaque-structural-target"

    baseline = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
        reported_candidate_packs=[bufotalin_pack],
    )
    observed = compile_program_route_candidate_set(
        renamed,
        route,
        projection,
        discovery,
        bundle,
        reported_candidate_packs=[bufotalin_pack],
    )
    assert observed == baseline

    with pytest.raises(ProgramRouteCandidateError, match="reported_program_target_mismatch"):
        compile_program_route_candidate_set(
            graph,
            route,
            projection,
            discovery,
            bundle,
            reported_candidate_packs=[
                _pack(ATORVASTATIN_OBSERVATION, ATORVASTATIN_PROJECTION)
            ],
        )

    tampered = deepcopy(bufotalin_pack)
    tampered["projection"]["counts"]["programs"] = 999
    tampered["projection"]["content_sha256"] = strict_canonical_json_sha256(
        {
            key: value
            for key, value in tampered["projection"].items()
            if key != "content_sha256"
        }
    )
    with pytest.raises(
        ProgramRouteCandidateError,
        match="reported_program_projection_not_current",
    ):
        compile_program_route_candidate_set(
            graph,
            route,
            projection,
            discovery,
            bundle,
            reported_candidate_packs=[tampered],
        )


def test_reported_route_without_bound_sources_is_not_mislabeled_as_literature() -> None:
    pack = _pack(ATORVASTATIN_OBSERVATION, ATORVASTATIN_PROJECTION)
    target_state = pack["projection"]["chemical_states"][
        pack["projection"]["target_state_id"]
    ]
    graph, route, projection, discovery, bundle = _current_materials(
        target_state["canonical_smiles"]
    )

    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
        reported_candidate_packs=[pack],
    )
    reported = next(
        row
        for row in candidate_set["candidates"].values()
        if row["source_kind"] != "baseline"
    )

    assert reported["source_kind"] == "chemical"
    assert reported["evidence"]["source_refs"] == []
    assert "SOURCE_PROVENANCE_MISSING" in reported["warning_codes"]
    assert reported["eligibility"]["exploration_visible"] is True
    assert reported["eligibility"]["shadow_optimizer"] is False
