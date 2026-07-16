from __future__ import annotations

from copy import deepcopy

import pytest

from cascade_planner.application.biocatalytic_programs import (
    compile_biocatalytic_program_bundle,
)
from cascade_planner.application.capability_applicability_calibration import (
    compile_capability_applicability_calibration,
)
from cascade_planner.application.experimental_claim_contracts import (
    with_experimental_claim_digest,
)
from cascade_planner.application.experimental_claims import (
    experimental_claim_set_oracle,
)
from cascade_planner.application.mechanism_programs import (
    MechanismProgramError,
    compile_mechanism_program_bundle,
    mechanism_program_bundle_oracle,
)
from cascade_planner.application.mechanism_experiment_feedback import (
    compile_mechanism_experiment_feedback,
    mechanism_experiment_feedback_oracle,
)
from cascade_planner.application.mechanism_program_validations import (
    mechanism_signature_sha256,
    with_mechanism_program_validation_digest,
)
from cascade_planner.application.mechanism_validation_frontier import (
    compile_mechanism_validation_frontier,
)
from cascade_planner.application.program_route_candidates import (
    ProgramRouteCandidateError,
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
from cascade_planner.orchestration.mechanism_program_review_materials import (
    compile_mechanism_program_review_materials,
)
from cascade_planner.orchestration.program_candidate_review_materials import (
    compile_program_candidate_review_materials,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _fixture() -> tuple[dict, dict]:
    smiles = [
        "CC(=O)C1CCCCC1",
        "CC(O)(O)C1CCCCC1",
        "CC(Cl)C1CCCCC1",
        "CC(O)C1CCCCC1",
    ]
    molecules = {
        f"m:{index}": {
            "canonical_smiles": value,
            "stock_observation_ids": [],
        }
        for index, value in enumerate(smiles)
    }
    edges = {
        f"edge:{index}": {
            "precursor_molecule_ids": [f"m:{index}"],
            "product_molecule_id": f"m:{index + 1}",
            "innovation_boundary_proof_level": 1,
            "procedure_record_ids": [],
            "source_binding_ids": [],
            "exact_record_ids": [],
            "reaction_proofs": [],
        }
        for index in range(3)
    }
    route = {
        "route_id": "route:mechanism-fixture",
        "route_family_id": "family:mechanism-fixture",
        "edge_ids": list(edges),
        "reported_source_refs": ["doi:10.1000/mechanism-anchor"],
        "proof_level": 2,
        "minimum_edge_proof_level": 1,
        "reaction_validated": False,
        "condition_complete": False,
        "procurement_closed": False,
        "configured_boundary_closed": False,
        "process_ready": False,
        "unproven_edge_ids": list(edges),
    }
    graph = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "mechanism-program-fixture",
        "revision": 4,
        "scientific_sha256": "mechanism-scientific-revision-4",
        "target_molecule_id": "m:3",
        "molecules": molecules,
        "edges": edges,
        "route_families": {route["route_family_id"]: {"edge_ids": list(edges), "closed": False}},
    }
    return graph, route


def _proposal(*, product_smiles: str = "CC(O)C1CCCCC1") -> dict:
    return {
        "proposal_id": "mechanism:bypass-two-steps",
        "precursor_smiles": "CC(O)(O)C1CCCCC1",
        "product_smiles": product_smiles,
        "anchor_edge_ids": ["edge:0"],
        "anchor_source_refs": ["doi:10.1000/mechanism-anchor"],
        "mechanistic_rationale": (
            "The materialized hydrate may undergo a concerted net reduction "
            "while retaining the cyclic carbon skeleton."
        ),
        "elementary_steps": ["activation", "selective hydride transfer"],
        "falsifiable_checks": [
            "LC-MS mass balance and NMR connectivity must match the route state"
        ],
    }


def _materials(proposal: dict | None = None) -> tuple[dict, dict, dict, dict, dict]:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[],
        mechanism_proposals=[proposal or _proposal()],
    )
    bundle = compile_mechanism_program_bundle(graph, route, projection, discovery)
    return graph, route, projection, discovery, bundle


def _mechanism_validation(proposal: dict, outcome: str) -> dict:
    interpretation = {
        "success": "net_transform_observed",
        "failure": "competing_pathway_observed",
        "inconclusive": "unresolved",
    }[outcome]
    checks = {
        row["check_id"]: outcome == "success"
        for row in proposal["validation_plan"]["required_checks"]
    }
    return with_mechanism_program_validation_digest(
        {
            "schema_version": "mechanism_program_validation.v1",
            "validation_id": f"validation:mechanism:{outcome}",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "outcome_status": outcome,
            "evidence_tier": "exact_reaction_screen",
            "interpretation_status": interpretation,
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "mechanism_signature_sha256": mechanism_signature_sha256(proposal),
            "required_check_results": checks,
            "claim_refs": ["claim:mechanism:fixture"],
            "condition_record_ids": ["condition:mechanism:fixture"],
            "analytical_record_ids": ["analytical:mechanism:fixture"],
            "outcome_metrics": {"conversion_fraction": 0.8 if outcome == "success" else 0.0},
        }
    )


def test_one_hop_rejoins_route_and_compiles_full_exploration_candidate() -> None:
    graph, route, projection, discovery, mechanism_bundle = _materials()

    assert discovery["candidate_count"] == 1
    assert mechanism_bundle["counts"] == {
        "program_proposals": 1,
        "route_candidates": 1,
        "rejected_candidates": 0,
        "validated_substitutions": 0,
        "unbound_validations": 0,
    }
    proposal = next(iter(mechanism_bundle["program_proposals"].values()))
    restitched = next(iter(mechanism_bundle["route_candidates"].values()))
    assert proposal["equivalent_reference_span"] == ["edge:1", "edge:2"]
    assert proposal["validation_vector"]["reaction"] == ("unvalidated_mechanism_hypothesis")
    assert restitched["full_candidate_route_restitched"] is True
    assert restitched["physical_step_count"] == 2
    assert restitched["chemical_step_equivalent_count"] == 3
    assert restitched["eligible_for_program_optimizer"] is False
    assert (
        mechanism_program_bundle_oracle(graph, route, projection, discovery, mechanism_bundle)[
            "accepted"
        ]
        is True
    )

    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        mechanism_bundle=mechanism_bundle,
    )
    candidate = next(
        value
        for value in candidate_set["candidates"].values()
        if value["source_kind"] == "mechanism"
    )
    assert candidate_set["counts"]["mechanism"] == 1
    assert candidate["eligibility"]["exploration_visible"] is True
    assert candidate["eligibility"]["shadow_optimizer"] is False
    assert candidate["eligibility"]["route_completion"] is False
    assert candidate["metrics"]["minimum_proof_level"] == 0
    assert candidate["metrics"]["reaction_validation_deficit_count"] >= 1
    assert "FULL_CANDIDATE_ROUTE_RESTITCHED" in candidate["warning_codes"]
    portfolio = optimize_program_route_candidates(candidate_set)
    assert (
        candidate["candidate_id"] in portfolio["profiles"]["exploration"]["eligible_candidate_ids"]
    )
    assert (
        candidate["candidate_id"]
        not in portfolio["profiles"]["shadow_optimizer"]["eligible_candidate_ids"]
    )
    assert program_route_portfolio_oracle(candidate_set, portfolio)["accepted"] is True


def test_non_rejoining_one_hop_stays_visible_only_in_discovery() -> None:
    graph, route, projection, discovery, bundle = _materials(
        _proposal(product_smiles="CC(=O)NC1CCCCC1")
    )

    assert discovery["candidate_count"] == 1
    assert bundle["program_proposals"] == {}
    assert bundle["route_candidates"] == {}
    assert bundle["rejections"] == [
        {
            "candidate_id": "mechanism:bypass-two-steps",
            "reasons": ["mechanism_full_route_restitch_missing"],
        }
    ]
    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        mechanism_bundle=bundle,
    )
    assert candidate_set["counts"]["mechanism"] == 0
    assert candidate_set["counts"]["candidates"] == 1


def test_tampered_or_stale_mechanism_materials_fail_closed() -> None:
    graph, route, projection, discovery, bundle = _materials()
    tampered = deepcopy(bundle)
    restitched = next(iter(tampered["route_candidates"].values()))
    restitched["full_candidate_route_restitched"] = False
    restitched["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in restitched.items() if key != "content_sha256"}
    )
    tampered["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    assert (
        mechanism_program_bundle_oracle(graph, route, projection, discovery, tampered)["accepted"]
        is False
    )
    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    with pytest.raises(
        ProgramRouteCandidateError,
        match="program_candidate_mechanism_bundle_not_current",
    ):
        compile_program_route_candidate_set(
            graph,
            route,
            projection,
            discovery,
            enzyme_bundle,
            mechanism_bundle=tampered,
        )

    stale_graph = deepcopy(graph)
    stale_graph["revision"] = 5
    with pytest.raises(MechanismProgramError, match="program_projection_not_current"):
        compile_mechanism_program_bundle(stale_graph, route, projection, discovery)


def test_mechanism_matching_uses_structures_not_target_name() -> None:
    graph, route, projection, discovery, bundle = _materials()
    renamed_graph = deepcopy(graph)
    renamed_graph["target_name"] = "renamed display label"

    rebuilt = compile_mechanism_program_bundle(renamed_graph, route, projection, discovery)

    assert rebuilt == bundle


def test_mechanism_results_enable_only_shadow_and_retain_all_outcomes() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    draft = next(iter(draft_bundle["program_proposals"].values()))
    validations = [
        _mechanism_validation(draft, "success"),
        _mechanism_validation(draft, "failure"),
        _mechanism_validation(draft, "inconclusive"),
    ]
    bundle = compile_mechanism_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    proposal = next(iter(bundle["program_proposals"].values()))
    variant = next(iter(bundle["route_candidates"].values()))
    assert bundle["counts"]["validated_substitutions"] == 1
    assert proposal["eligible_for_shadow_optimizer"] is True
    assert proposal["validation_vector"]["reaction"] == ("exact_boundary_experiment_bound")
    assert proposal["validation_vector"]["mechanism"] == (
        "net_transform_observed_mechanism_unresolved"
    )
    assert proposal["validation_plan"]["negative_validation_ids"] == [
        "validation:mechanism:failure"
    ]
    assert proposal["validation_plan"]["inconclusive_validation_ids"] == [
        "validation:mechanism:inconclusive"
    ]
    assert variant["eligible_for_program_optimizer"] is True
    assert variant["eligible_for_route_completion"] is False
    assert (
        mechanism_program_bundle_oracle(
            graph,
            route,
            projection,
            discovery,
            bundle,
            validations=validations,
        )["accepted"]
        is True
    )

    frontier = compile_mechanism_validation_frontier(graph, discovery, bundle)
    assert frontier["counts"] == {
        "experiment_required": 0,
        "validation_granted": 0,
    }
    feedback = compile_mechanism_experiment_feedback(discovery, bundle, validations)
    assert feedback["counts"] == {
        "feedback_records": 3,
        "positive": 1,
        "negative": 1,
        "inconclusive": 1,
        "rejected_validations": 0,
        "reaction_proofs_created": 0,
        "store_mutations": 0,
    }
    assert (
        mechanism_experiment_feedback_oracle(discovery, bundle, feedback, validations)["accepted"]
        is True
    )
    rows = {row["polarity"]: row for row in feedback["feedback"].values()}
    assert rows["positive"]["interpretation_status"] == "net_transform_observed"
    assert rows["positive"]["canonical_reaction_proof_created"] is False
    assert rows["negative"]["candidate_disposition"] == "exploration_visible"
    assert rows["inconclusive"]["grants_validation"] is False

    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        mechanism_bundle=bundle,
        mechanism_validations=validations,
    )
    candidate = next(
        row for row in candidate_set["candidates"].values() if row["source_kind"] == "mechanism"
    )
    assert candidate["eligibility"]["shadow_optimizer"] is True
    assert candidate["eligibility"]["route_completion"] is False
    assert candidate["metrics"]["reaction_validation_deficit_count"] == 0
    assert candidate["metrics"]["condition_deficit_count"] == 0
    assert candidate["metrics"]["specialized_validation_deficit_count"] == 0
    assert candidate["metrics"]["source_deficit_count"] >= 1
    assert candidate["metrics"]["minimum_proof_level"] == 0
    assert (
        "MECHANISM_SUPPORT_NET_TRANSFORM_OBSERVED_MECHANISM_UNRESOLVED"
        in candidate["warning_codes"]
    )

    materials = compile_mechanism_program_review_materials(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )
    assert materials["mechanism_bundle"] == bundle
    assert materials["mechanism_validation_frontier"] == frontier
    assert materials["mechanism_experiment_feedback"] == feedback
    assert materials["mechanism_feedback_oracle"]["accepted"] is True
    candidate_materials = compile_program_candidate_review_materials(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        mechanism_validations=validations,
    )
    assert candidate_materials["mechanism_bundle"] == bundle
    assert candidate_materials["program_route_candidates"] == candidate_set
    assert candidate_materials["program_optimizer_oracle"]["accepted"] is True
    claims = candidate_materials["experimental_claims"]
    assert claims["counts"]["claims"] == 3
    assert claims["counts"]["mechanism"] == 3
    assert claims["counts"]["positive"] == 1
    assert claims["counts"]["negative"] == 1
    assert claims["counts"]["inconclusive"] == 1
    assert claims["counts"]["canonical_reaction_proofs_created"] == 0
    assert candidate_materials["experimental_claims_oracle"]["accepted"] is True
    calibration = candidate_materials["capability_calibration"]
    assert calibration["counts"]["calibrations"] == 1
    assert calibration["counts"]["conflicting"] == 1
    assert calibration["counts"]["dirty_domains"] == 1
    assert candidate_materials["capability_calibration_oracle"]["accepted"] is True
    claim = next(iter(claims["claims"].values()))
    assert claim["generalization_scope"] == "exact_boundary_only"
    assert claim["semantics"]["does_not_create_canonical_reaction_proof"] is True
    unchanged = compile_capability_applicability_calibration(claims, previous=calibration)
    assert unchanged["counts"]["dirty_domains"] == 0

    tampered = deepcopy(claims)
    claim_id = next(iter(tampered["claims"]))
    claim_row = deepcopy(tampered["claims"][claim_id])
    claim_row["outcome_metrics"]["invented_metric"] = 1
    tampered["claims"][claim_id] = with_experimental_claim_digest(claim_row)
    tampered = with_experimental_claim_digest(tampered)
    assert (
        experimental_claim_set_oracle(
            enzyme_bundle,
            candidate_materials["biocatalytic_oracle"],
            candidate_materials["execution_capability_feedback"],
            candidate_materials["execution_feedback_oracle"],
            feedback,
            candidate_materials["mechanism_feedback_oracle"],
            tampered,
            validations=validations,
        )["accepted"]
        is False
    )


def test_invalid_and_unbound_mechanism_results_are_explicitly_rejected() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposal = next(iter(draft_bundle["program_proposals"].values()))
    invalid = _mechanism_validation(proposal, "success")
    invalid["required_check_results"].pop(next(iter(invalid["required_check_results"])))
    invalid = with_mechanism_program_validation_digest(invalid)
    wrong_signature = _mechanism_validation(proposal, "success")
    wrong_signature["validation_id"] = "validation:mechanism:wrong-signature"
    wrong_signature["mechanism_signature_sha256"] = "0" * 64
    wrong_signature = with_mechanism_program_validation_digest(wrong_signature)
    wrong_interpretation = _mechanism_validation(proposal, "success")
    wrong_interpretation["validation_id"] = "validation:mechanism:wrong-interpretation"
    wrong_interpretation["interpretation_status"] = "not_supported"
    wrong_interpretation = with_mechanism_program_validation_digest(wrong_interpretation)
    missing_analytics = _mechanism_validation(proposal, "success")
    missing_analytics["validation_id"] = "validation:mechanism:no-analytics"
    missing_analytics["analytical_record_ids"] = []
    missing_analytics = with_mechanism_program_validation_digest(missing_analytics)
    unbound = deepcopy(_mechanism_validation(proposal, "inconclusive"))
    unbound["validation_id"] = "validation:mechanism:unbound"
    unbound["program_id"] = "program:mechanism:missing"
    unbound = with_mechanism_program_validation_digest(unbound)
    validations = [
        invalid,
        wrong_signature,
        wrong_interpretation,
        missing_analytics,
        unbound,
    ]
    bundle = compile_mechanism_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    assert bundle["counts"]["validated_substitutions"] == 0
    assert bundle["counts"]["unbound_validations"] == 1
    frontier = compile_mechanism_validation_frontier(graph, discovery, bundle)
    assert frontier["counts"]["experiment_required"] == 1
    feedback = compile_mechanism_experiment_feedback(discovery, bundle, validations)
    assert feedback["counts"]["feedback_records"] == 0
    assert feedback["counts"]["rejected_validations"] == 5
    reasons = {reason for row in feedback["rejected_validations"] for reason in row["reasons"]}
    assert "validation_required_checks_invalid" in reasons
    assert "validation_mechanism_signature_mismatch" in reasons
    assert "validation_outcome_interpretation_mismatch" in reasons
    assert "validation_analytical_records_missing" in reasons
    assert "validation_program_unbound" in reasons

    tampered = deepcopy(feedback)
    tampered["counts"]["negative"] = 99
    tampered["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    assert (
        mechanism_experiment_feedback_oracle(discovery, bundle, tampered, validations)["accepted"]
        is False
    )


def test_mechanism_validation_ids_are_unique_across_the_batch() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposal = next(iter(draft_bundle["program_proposals"].values()))
    success = _mechanism_validation(proposal, "success")
    failure = _mechanism_validation(proposal, "failure")
    failure["validation_id"] = success["validation_id"]
    failure = with_mechanism_program_validation_digest(failure)
    validations = [success, failure]
    bundle = compile_mechanism_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    assert bundle["counts"]["validated_substitutions"] == 0
    feedback = compile_mechanism_experiment_feedback(discovery, bundle, validations)
    assert feedback["counts"]["feedback_records"] == 0
    assert feedback["counts"]["rejected_validations"] == 2
    assert all(
        "validation_id_duplicate" in row["reasons"] for row in feedback["rejected_validations"]
    )
