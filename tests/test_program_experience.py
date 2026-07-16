from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.experimental_claim_contracts import (
    CLAIM_SEMANTICS,
    CLAIM_SET_SEMANTICS,
    experimental_claim_counts,
    with_experimental_claim_digest,
)
from cascade_planner.application.program_experience import (
    apply_program_experience,
    synchronize_program_experience_library,
)
from cascade_planner.application.program_experience_store import (
    read_program_experience_library,
)
from cascade_planner.application.biocatalysis_validation_frontier import (
    compile_biocatalysis_validation_frontier,
)
from cascade_planner.application.biocatalytic_programs import (
    compile_biocatalytic_program_bundle,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)


def _graph_and_route() -> tuple[dict, dict]:
    smiles = [
        "CC(=O)C1CCCCC1",
        "CC(O)(O)C1CCCCC1",
        "CC(Cl)C1CCCCC1",
        "CC(O)C1CCCCC1",
    ]
    molecules = {f"m:{index}": {"canonical_smiles": value} for index, value in enumerate(smiles)}
    edges = {
        f"edge:{index}": {
            "precursor_molecule_ids": [f"m:{index}"],
            "product_molecule_id": f"m:{index + 1}",
            "innovation_boundary_proof_level": 1,
        }
        for index in range(3)
    }
    return (
        {"run_id": "experience-fixture", "molecules": molecules, "edges": edges},
        {
            "route_id": "route:experience-fixture",
            "route_family_id": "family:experience-fixture",
            "edge_ids": list(edges),
            "reported_source_refs": ["doi:10.1000/experience"],
        },
    )


def _capability() -> dict:
    return {
        "capability_id": "fixture:cyclic-ketone-reduction",
        "enzyme": {"classes": ["ketoreductase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.3,
            "max_abs_heavy_atom_delta": 0,
            "min_substrate_carbons": 6,
            "min_substrate_rings": 1,
            "min_window_steps": 3,
            "max_window_steps": 3,
            "reject_unlisted_motif_changes": True,
        },
        "selectivity_objective": "Reduce the cyclic ketone to the requested alcohol.",
        "substrate_scope_basis": "fixture analog",
        "precedent_refs": ["doi:10.1000/experience"],
    }


def _claim_set(discovery: dict, *, claim_id: str = "claim:positive", polarity: str = "positive") -> dict:
    candidate = discovery["candidates"][0]
    claim = with_experimental_claim_digest(
        {
            "schema_version": "experimental_observation_claim.v1",
            "claim_id": claim_id,
            "claim_kind": "program_validation_observation",
            "domain": "biocatalytic",
            "polarity": polarity,
            "outcome_status": "success" if polarity == "positive" else "failure",
            "interpretation_status": "exact_substrate_biocatalysis_supported",
            "program_id": "program:fixture:biocatalytic",
            "subject_refs": {
                "capability_id": candidate["capability_id"],
                "innovation_id": candidate["route_innovation"]["innovation_id"],
            },
            "boundary": {
                "input_state_ids": ["state:m:0"],
                "output_state_ids": ["state:m:3"],
            },
            "source_validation": {
                "schema_version": "biocatalysis_program_validation.v1",
                "validation_id": "validation:" + claim_id,
                "content_sha256": "a" * 64,
            },
            "evidence_tier": "exact_substrate_screen",
            "supporting_claim_refs": ["claim:fixture"],
            "condition_record_ids": ["condition:fixture"],
            "outcome_metrics": {"conversion_fraction": 0.82},
            "grants_domain_validation": polarity == "positive",
            "generalization_scope": "exact_boundary_only",
            "authority_scope": "experimental_observation_exact_boundary",
            "domain_context": {"selectivity_assessed": True},
            "semantics": dict(CLAIM_SEMANTICS),
        }
    )
    claims = {claim_id: claim}
    rejected: list[dict] = []
    return with_experimental_claim_digest(
        {
            "schema_version": "experimental_observation_claim_set.v1",
            "run_id": "experience-fixture",
            "route_id": "route:experience-fixture",
            "source_artifacts": {
                "biocatalytic_bundle_sha256": "b" * 64,
                "biocatalytic_oracle_sha256": "c" * 64,
                "execution_feedback_sha256": "d" * 64,
                "execution_oracle_sha256": "e" * 64,
                "mechanism_feedback_sha256": "f" * 64,
                "mechanism_oracle_sha256": "1" * 64,
                "validation_pack_sha256": "2" * 64,
            },
            "claims": claims,
            "rejected_validations": rejected,
            "counts": experimental_claim_counts(claims, rejected),
            "semantics": dict(CLAIM_SET_SEMANTICS),
        }
    )


def test_replay_validated_claims_become_bounded_ranking_memory(tmp_path) -> None:
    graph, route = _graph_and_route()
    discovery = discover_route_innovations(graph, route, capabilities=[_capability()])
    source = {"graph": graph, "discovery": discovery, "claim_set": _claim_set(discovery)}
    path = tmp_path / "program-experience.json"

    learned = synchronize_program_experience_library(path, [source])
    repeated = synchronize_program_experience_library(path, [source])
    library, error = read_program_experience_library(path)
    reranked = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
        experience_library=library,
    )

    assert error == ""
    assert learned["new_claim_count"] == 1
    assert repeated["new_claim_count"] == 0
    candidate = reranked["candidates"][0]
    assert candidate["priority_score"] > candidate["base_priority_score"]
    assert candidate["experience_memory"]["strongest_transfer_scope"] == "exact_boundary"
    assert candidate["experience_memory"]["current_candidate_still_requires_exact_validation"] is True
    assert "EXACT_SUBSTRATE_UNVALIDATED" in candidate["warning_codes"]
    assert reranked["program_experience"]["matched_candidate_count"] == 1


def test_experience_hint_reaches_validation_plan_without_granting_validation(tmp_path) -> None:
    graph, route = _graph_and_route()
    graph.update(
        {
            "schema_version": "canonical_retrosynthesis_hypergraph.v1",
            "revision": 1,
            "scientific_sha256": "experience-scientific-1",
            "target_molecule_id": "m:3",
            "route_families": {route["route_family_id"]: {"edge_ids": route["edge_ids"]}},
        }
    )
    discovery = discover_route_innovations(graph, route, capabilities=[_capability()])
    path = tmp_path / "program-experience.json"
    synchronize_program_experience_library(
        path, [{"graph": graph, "discovery": discovery, "claim_set": _claim_set(discovery)}]
    )
    library, _ = read_program_experience_library(path)
    reranked = discover_route_innovations(
        graph, route, capabilities=[_capability()], experience_library=library
    )
    projection = project_canonical_graph_to_programs(graph)
    bundle = compile_biocatalytic_program_bundle(graph, route, projection, reranked)

    frontier = compile_biocatalysis_validation_frontier(graph, reranked, bundle)
    plan = next(iter(frontier["plans"].values()))

    assert plan["experience_memory"]["disposition"] == "supported"
    assert plan["priority_score"] == reranked["candidates"][0]["priority_score"]
    assert plan["grants_validation"] is False
    assert plan["eligible_for_shadow_admission"] is False


def test_conflicting_memory_is_visible_and_never_promotes_candidate(tmp_path) -> None:
    graph, route = _graph_and_route()
    discovery = discover_route_innovations(graph, route, capabilities=[_capability()])
    sources = [
        {"graph": graph, "discovery": discovery, "claim_set": _claim_set(discovery)},
        {
            "graph": graph,
            "discovery": discovery,
            "claim_set": _claim_set(
                discovery, claim_id="claim:negative", polarity="negative"
            ),
        },
    ]
    path = tmp_path / "program-experience.json"
    synchronize_program_experience_library(path, sources)
    library, _ = read_program_experience_library(path)

    candidates, projection = apply_program_experience(discovery["candidates"], library)

    assert candidates[0]["priority_score"] == candidates[0]["base_priority_score"]
    assert candidates[0]["experience_memory"]["disposition"] == "conflicting"
    assert "SELF_EVOLUTION_CONFLICTING_PRIOR" in candidates[0]["warning_codes"]
    assert projection["semantics"]["cannot_grant_program_validation_proof_completion_or_acceptance"] is True


def test_corrupted_library_and_claim_set_fail_closed(tmp_path) -> None:
    graph, route = _graph_and_route()
    discovery = discover_route_innovations(graph, route, capabilities=[_capability()])
    invalid_claim_set = deepcopy(_claim_set(discovery))
    invalid_claim_set["route_id"] = "tampered"
    path = tmp_path / "program-experience.json"

    rejected = synchronize_program_experience_library(
        path, [{"graph": graph, "discovery": discovery, "claim_set": invalid_claim_set}]
    )
    assert rejected["rejected_source_count"] == 1
    assert rejected["experience_count"] == 0

    synchronize_program_experience_library(
        path,
        [{"graph": graph, "discovery": discovery, "claim_set": _claim_set(discovery)}],
    )
    path.write_text(path.read_text(encoding="utf-8").replace("supported", "forged"), encoding="utf-8")
    library, error = read_program_experience_library(path)
    assert library == {}
    assert error == "program_experience_library_digest_invalid"
