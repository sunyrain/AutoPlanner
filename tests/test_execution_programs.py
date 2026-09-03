from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from cascade_planner.application.experimental_claim_store import (
    ExperimentalClaimAdmissionDisabled,
    ExperimentalClaimStore,
    ExperimentalClaimStoreCorruption,
    ExperimentalClaimStoreError,
)

from cascade_planner.application.biocatalytic_programs import (
    compile_biocatalytic_program_bundle,
)
from cascade_planner.application.execution_capability_feedback import (
    compile_execution_capability_feedback,
    execution_capability_feedback_oracle,
)
from cascade_planner.application.execution_program_validations import (
    execution_operation_sequence_sha256,
    with_execution_program_validation_digest,
)
from cascade_planner.application.execution_programs import (
    compile_execution_program_bundle,
    execution_program_bundle_oracle,
)
from cascade_planner.application.execution_validation_frontier import (
    compile_execution_validation_frontier,
)
from cascade_planner.application.deficit_frontier import compile_deficit_frontier
from cascade_planner.application.experiment_execution_results import (
    audit_experiment_execution_result,
    build_experiment_execution_result,
    release_experiment_validation_candidate,
)
from cascade_planner.application.experimental_work_frontier import (
    compile_experimental_work_frontier,
    experimental_work_frontier_oracle,
)
from cascade_planner.application.biocatalysis_validation_frontier import (
    compile_biocatalysis_validation_frontier,
)
from cascade_planner.application.program_route_candidates import (
    ProgramRouteCandidateError,
    compile_program_route_candidate_set,
)
from cascade_planner.application.program_route_optimizer import (
    optimize_program_route_candidates,
    program_route_portfolio_oracle,
)
from cascade_planner.application.route_execution_capabilities import (
    normalize_program_execution_capability,
    normalize_program_execution_catalog,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.application.program_validation_routing import (
    partition_program_validations,
)
from cascade_planner.orchestration.execution_program_review_materials import (
    compile_execution_program_review_materials,
)
from cascade_planner.orchestration.program_candidate_review_materials import (
    compile_program_candidate_review_materials,
)
from cascade_planner.runtime.artifact_store import ArtifactStore
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
        "route_id": "route:execution-fixture",
        "route_family_id": "family:execution-fixture",
        "edge_ids": list(edges),
        "reported_source_refs": ["doi:10.1000/baseline-route"],
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
        "run_id": "execution-program-fixture",
        "revision": 5,
        "scientific_sha256": "execution-scientific-revision-5",
        "target_molecule_id": "m:3",
        "molecules": molecules,
        "edges": edges,
        "route_families": {route["route_family_id"]: {"edge_ids": list(edges), "closed": False}},
    }
    return graph, route


def _match() -> dict:
    return {
        "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
        "element_delta": {"C": 0, "O": 0},
        "min_scaffold_similarity": 0.3,
        "max_abs_heavy_atom_delta": 0,
        "min_substrate_carbons": 6,
        "min_substrate_rings": 1,
        "min_window_steps": 3,
        "max_window_steps": 3,
        "reject_unlisted_motif_changes": True,
    }


def _whole_cell_capability() -> dict:
    return {
        "schema_version": "program_execution_capability.v1",
        "capability_id": "fixture:whole-cell-net-reduction",
        "execution_domain": "whole_cell",
        "actors": {
            "organism": {
                "strain_ids": ["fixture-strain-1"],
                "taxa": ["Escherichia coli"],
                "preparation_modes": ["resting_cell"],
            }
        },
        "match": _match(),
        "operation_blueprints": [
            {
                "operation_kind": "whole_cell_preparation",
                "execution_domain": "whole_cell",
                "isolated_operation": True,
                "contributes_to_net_transform": False,
                "description": "Prepare and wash resting cells.",
            },
            {
                "operation_kind": "whole_cell_biotransformation",
                "execution_domain": "whole_cell",
                "isolated_operation": True,
                "contributes_to_net_transform": True,
                "description": "Run the net reduction with the intact-cell catalyst.",
            },
            {
                "operation_kind": "workup",
                "execution_domain": "whole_cell",
                "isolated_operation": True,
                "contributes_to_net_transform": False,
                "description": "Remove biomass and extract the product.",
            },
            {
                "operation_kind": "separation",
                "execution_domain": "chemical",
                "isolated_operation": True,
                "contributes_to_net_transform": False,
                "description": "Purify the product from the extract.",
            },
        ],
        "selectivity_objective": "Deliver the specified alcohol stereoisomer.",
        "substrate_scope_basis": "whole-cell analog fixture only",
        "cofactor_requirements": {"intracellular": ["NAD(P)H"]},
        "cofactor_regenerations": {"intracellular": ["glucose metabolism"]},
        "carrier_requirements": {"cells": ["wet biomass"]},
        "precedent_refs": ["doi:10.1000/whole-cell-fixture"],
    }


def _hybrid_capability() -> dict:
    return {
        "schema_version": "program_execution_capability.v1",
        "capability_id": "fixture:chemoenzymatic-net-reduction",
        "execution_domain": "hybrid",
        "actors": {
            "enzyme": {"classes": ["ketoreductase"]},
            "catalyst_classes": ["Lewis acid"],
        },
        "match": _match(),
        "operation_blueprints": [
            {
                "operation_kind": "chemical_reaction",
                "execution_domain": "chemical",
                "isolated_operation": True,
                "contributes_to_net_transform": True,
                "description": "Chemically activate the route boundary substrate.",
            },
            {
                "operation_kind": "enzyme_reaction",
                "execution_domain": "enzymatic",
                "isolated_operation": True,
                "contributes_to_net_transform": True,
                "description": "Complete the selective reduction enzymatically.",
            },
            {
                "operation_kind": "separation",
                "execution_domain": "chemical",
                "isolated_operation": True,
                "contributes_to_net_transform": False,
                "description": "Remove catalyst and isolate the product.",
            },
        ],
        "selectivity_objective": "Preserve the scaffold and obtain one alcohol isomer.",
        "substrate_scope_basis": "chemoenzymatic analog fixture only",
        "cofactor_requirements": {"accepted": ["NADPH"]},
        "precedent_refs": ["doi:10.1000/hybrid-fixture"],
    }


def _materials() -> tuple[dict, dict, dict, dict, dict]:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_whole_cell_capability(), _hybrid_capability()],
    )
    bundle = compile_execution_program_bundle(graph, route, projection, discovery)
    return graph, route, projection, discovery, bundle


def _execution_validation(proposal: dict, outcome: str) -> dict:
    checks = {
        check_id: outcome == "success"
        for check_id in proposal["validation_plan"]["required_checks"]
    }
    return with_execution_program_validation_digest(
        {
            "schema_version": "execution_program_validation.v1",
            "validation_id": (f"validation:{proposal['execution_domain']}:{outcome}"),
            "program_id": proposal["program_id"],
            "capability_id": proposal["source_capability_id"],
            "source_capability_sha256": proposal["source_capability_sha256"],
            "execution_domain": proposal["execution_domain"],
            "outcome_status": outcome,
            "evidence_tier": "exact_execution_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "operation_sequence_sha256": execution_operation_sequence_sha256(proposal),
            "required_check_results": checks,
            "claim_refs": ["claim:fixture:exact-boundary"],
            "condition_record_ids": ["condition:fixture:execution"],
            "actor_identity_refs": ["actor:fixture:verified"],
            "cofactor_carrier_ledger_closed": outcome == "success",
            "outcome_metrics": {"conversion_fraction": 0.85 if outcome == "success" else 0.0},
        }
    )


def test_whole_cell_and_hybrid_compile_as_distinct_restitched_programs() -> None:
    graph, route, projection, discovery, bundle = _materials()

    assert discovery["candidate_count"] == 2
    assert len(discovery["execution_program_draft_candidate_ids"]) == 2
    assert discovery["program_draft_candidate_ids"] == []
    assert bundle["counts"] == {
        "program_proposals": 2,
        "route_candidates": 2,
        "whole_cell": 1,
        "hybrid": 1,
        "rejected_candidates": 0,
        "validated_substitutions": 0,
        "unbound_validations": 0,
    }
    assert (
        execution_program_bundle_oracle(graph, route, projection, discovery, bundle)["accepted"]
        is True
    )
    proposals = {row["execution_domain"]: row for row in bundle["program_proposals"].values()}
    whole_cell = proposals["whole_cell"]
    hybrid = proposals["hybrid"]
    assert whole_cell["isolated_operation_count"] == 4
    assert whole_cell["net_operation_savings"] == -1
    assert "organism_identity_and_viability" in whole_cell["validation_plan"]["required_checks"]
    assert hybrid["isolated_operation_count"] == 3
    assert "ordered_operation_compatibility" in hybrid["validation_plan"]["required_checks"]
    assert "HYBRID_INTERNAL_STATES_UNMATERIALIZED" in hybrid["warning_codes"]
    transform_ops = [
        row for row in hybrid["operation_blueprints"] if row["contributes_to_net_transform"]
    ]
    assert transform_ops[0]["input_state_ids"] == hybrid["input_state_ids"]
    assert transform_ops[-1]["output_state_ids"] == hybrid["output_state_ids"]


def test_execution_candidates_enter_exploration_even_with_negative_savings() -> None:
    graph, route, projection, discovery, execution_bundle = _materials()
    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        execution_bundle=execution_bundle,
    )

    assert candidate_set["counts"]["candidates"] == 3
    assert candidate_set["counts"]["whole_cell"] == 1
    assert candidate_set["counts"]["hybrid"] == 1
    execution = {
        row["source_kind"]: row
        for row in candidate_set["candidates"].values()
        if row["source_kind"] in {"whole_cell", "hybrid"}
    }
    assert execution["whole_cell"]["metrics"]["net_step_savings"] == -1
    for row in execution.values():
        assert row["eligibility"]["exploration_visible"] is True
        assert row["eligibility"]["shadow_optimizer"] is False
        assert row["eligibility"]["route_completion"] is False
        assert row["metrics"]["minimum_proof_level"] == 0
        assert row["metrics"]["specialized_validation_deficit_count"] == 1
        assert "SPECIALIZED_EXECUTION_VALIDATION_REQUIRED" in row["warning_codes"]
    portfolio = optimize_program_route_candidates(candidate_set)
    exploration = portfolio["profiles"]["exploration"]["eligible_candidate_ids"]
    assert {row["candidate_id"] for row in execution.values()}.issubset(exploration)
    assert portfolio["profiles"]["shadow_optimizer"]["eligible_candidate_ids"] == [
        next(
            row["candidate_id"]
            for row in candidate_set["candidates"].values()
            if row["source_kind"] == "baseline"
        )
    ]
    assert program_route_portfolio_oracle(candidate_set, portfolio)["accepted"] is True


def test_invalid_execution_capabilities_are_rejected_before_matching() -> None:
    accepted, rejected = normalize_program_execution_catalog(
        {
            "schema_version": "program_execution_capability_catalog.v0",
            "capabilities": [_whole_cell_capability()],
        }
    )
    assert accepted == []
    assert rejected[0]["reasons"] == ["program_execution_capability_catalog_schema_invalid"]

    wrong_schema = _whole_cell_capability()
    wrong_schema["schema_version"] = "program_execution_capability.v0"
    normalized, reasons = normalize_program_execution_capability(wrong_schema)
    assert normalized == {}
    assert "program_execution_capability_schema_invalid" in reasons

    invalid_flags = _whole_cell_capability()
    del invalid_flags["operation_blueprints"][0]["isolated_operation"]
    normalized, reasons = normalize_program_execution_capability(invalid_flags)
    assert normalized == {}
    assert "program_execution_operation_isolated_flag_invalid" in reasons

    invalid_match = _whole_cell_capability()
    invalid_match["match"]["net_motif_delta"]["carbonyl"] = "not-an-integer"
    normalized, reasons = normalize_program_execution_capability(invalid_match)
    assert normalized == {}
    assert "program_execution_structure_match_invalid" in reasons

    missing_actor = _whole_cell_capability()
    missing_actor["actors"] = {}
    normalized, reasons = normalize_program_execution_capability(missing_actor)
    assert normalized == {}
    assert "whole_cell_organism_or_preparation_missing" in reasons

    not_hybrid = _hybrid_capability()
    not_hybrid["operation_blueprints"] = [
        row for row in not_hybrid["operation_blueprints"] if row["execution_domain"] != "enzymatic"
    ]
    normalized, reasons = normalize_program_execution_capability(not_hybrid)
    assert normalized == {}
    assert "hybrid_requires_chemical_and_biological_transforms" in reasons

    graph, route = _fixture()
    discovery = discover_route_innovations(graph, route, capabilities=[missing_actor, not_hybrid])
    assert discovery["candidate_count"] == 0
    assert {row["kind"] for row in discovery["rejected"]} == {"execution_capability"}


def test_valid_but_inapplicable_execution_capability_is_an_empty_result() -> None:
    graph, route = _fixture()
    capability = _whole_cell_capability()
    capability["match"]["min_substrate_carbons"] = 100
    discovery = discover_route_innovations(graph, route, capabilities=[capability])
    projection = project_canonical_graph_to_programs(graph)
    bundle = compile_execution_program_bundle(graph, route, projection, discovery)

    assert discovery["candidate_count"] == 0
    assert discovery["rejected"] == []
    assert bundle["counts"] == {
        "program_proposals": 0,
        "route_candidates": 0,
        "whole_cell": 0,
        "hybrid": 0,
        "rejected_candidates": 0,
        "validated_substitutions": 0,
        "unbound_validations": 0,
    }
    assert (
        execution_program_bundle_oracle(graph, route, projection, discovery, bundle)["accepted"]
        is True
    )


def test_execution_bundle_tampering_fails_closed_and_target_name_is_ignored() -> None:
    graph, route, projection, discovery, bundle = _materials()
    renamed = deepcopy(graph)
    renamed["target_name"] = "display-only rename"
    assert compile_execution_program_bundle(renamed, route, projection, discovery) == bundle

    tampered = deepcopy(bundle)
    proposal = next(iter(tampered["program_proposals"].values()))
    proposal["eligible_for_shadow_optimizer"] = True
    proposal["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in proposal.items() if key != "content_sha256"}
    )
    tampered["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    assert (
        execution_program_bundle_oracle(graph, route, projection, discovery, tampered)["accepted"]
        is False
    )
    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    with pytest.raises(
        ProgramRouteCandidateError,
        match="program_candidate_execution_bundle_not_current",
    ):
        compile_program_route_candidate_set(
            graph,
            route,
            projection,
            discovery,
            enzyme_bundle,
            execution_bundle=tampered,
        )

    malformed_discovery = deepcopy(discovery)
    embedded = malformed_discovery["candidates"][0]["execution_capability"]
    embedded["operation_blueprints"] = []
    embedded["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in embedded.items() if key != "content_sha256"}
    )
    malformed_discovery["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in malformed_discovery.items() if key != "content_sha256"}
    )
    rejected = compile_execution_program_bundle(graph, route, projection, malformed_discovery)
    assert rejected["counts"]["rejected_candidates"] == 1
    assert rejected["counts"]["program_proposals"] == 1


def test_execution_results_enable_only_shadow_and_retain_negative_feedback() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposals = {row["execution_domain"]: row for row in draft_bundle["program_proposals"].values()}
    validations = [
        _execution_validation(proposals["whole_cell"], "success"),
        _execution_validation(proposals["hybrid"], "failure"),
        _execution_validation(proposals["hybrid"], "inconclusive"),
    ]
    bundle = compile_execution_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    validated = {row["execution_domain"]: row for row in bundle["program_proposals"].values()}
    assert bundle["counts"]["validated_substitutions"] == 1
    assert validated["whole_cell"]["eligible_for_shadow_optimizer"] is True
    assert validated["hybrid"]["eligible_for_shadow_optimizer"] is False
    assert validated["hybrid"]["validation_plan"]["negative_validation_ids"] == [
        "validation:hybrid:failure"
    ]
    assert validated["hybrid"]["validation_plan"]["inconclusive_validation_ids"] == [
        "validation:hybrid:inconclusive"
    ]
    frontier = compile_execution_validation_frontier(graph, discovery, bundle)
    assert frontier["counts"] == {
        "experiment_required": 1,
        "whole_cell": 0,
        "hybrid": 1,
        "validation_granted": 0,
    }
    feedback = compile_execution_capability_feedback(discovery, bundle, validations)
    assert feedback["counts"] == {
        "feedback_records": 3,
        "positive": 1,
        "negative": 1,
        "inconclusive": 1,
        "rejected_validations": 0,
        "catalog_mutations": 0,
    }
    assert (
        execution_capability_feedback_oracle(discovery, bundle, feedback, validations)["accepted"]
        is True
    )
    by_polarity = {row["polarity"]: row for row in feedback["feedback"].values()}
    assert by_polarity["positive"]["grants_validation"] is True
    assert by_polarity["negative"]["grants_validation"] is False
    assert by_polarity["negative"]["candidate_disposition"] == "exploration_visible"
    assert by_polarity["negative"]["capability_disabled"] is False
    assert by_polarity["inconclusive"]["grants_validation"] is False
    materials = compile_execution_program_review_materials(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )
    assert materials["execution_bundle"] == bundle
    assert materials["execution_validation_frontier"] == frontier
    assert materials["execution_capability_feedback"] == feedback
    assert materials["execution_feedback_oracle"]["accepted"] is True

    enzyme_bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        execution_bundle=bundle,
        execution_validations=validations,
    )
    execution = {
        row["source_kind"]: row
        for row in candidate_set["candidates"].values()
        if row["source_kind"] in {"whole_cell", "hybrid"}
    }
    assert execution["whole_cell"]["eligibility"]["shadow_optimizer"] is True
    assert execution["whole_cell"]["metrics"]["specialized_validation_deficit_count"] == 0
    assert execution["whole_cell"]["metrics"]["reaction_validation_deficit_count"] == 0
    assert execution["hybrid"]["eligibility"]["shadow_optimizer"] is False
    assert execution["hybrid"]["metrics"]["specialized_validation_deficit_count"] == 1
    assert all(row["eligibility"]["route_completion"] is False for row in execution.values())
    review = compile_program_candidate_review_materials(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        execution_validations=validations,
    )
    claims = review["experimental_claims"]
    assert claims["counts"]["claims"] == 3
    assert claims["counts"]["execution"] == 3
    assert claims["counts"]["positive"] == 1
    assert claims["counts"]["negative"] == 1
    assert claims["counts"]["inconclusive"] == 1
    assert claims["counts"]["canonical_reaction_proofs_created"] == 0
    assert review["experimental_claims_oracle"]["accepted"] is True
    calibration = review["capability_calibration"]
    assert calibration["counts"]["calibrations"] == 2
    assert calibration["counts"]["dirty_domains"] == 2
    assert calibration["counts"]["catalog_mutations"] == 0
    assert review["capability_calibration_oracle"]["accepted"] is True


def test_invalid_and_unbound_execution_results_are_explicitly_rejected() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposal = next(iter(draft_bundle["program_proposals"].values()))
    invalid = _execution_validation(proposal, "success")
    invalid["required_check_results"].pop(next(iter(invalid["required_check_results"])))
    invalid = with_execution_program_validation_digest(invalid)
    unbound = deepcopy(_execution_validation(proposal, "inconclusive"))
    unbound["validation_id"] = "validation:unbound:inconclusive"
    unbound["program_id"] = "program:missing"
    unbound = with_execution_program_validation_digest(unbound)
    validations = [invalid, unbound]
    bundle = compile_execution_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    assert bundle["counts"]["validated_substitutions"] == 0
    assert bundle["counts"]["unbound_validations"] == 1
    feedback = compile_execution_capability_feedback(discovery, bundle, validations)
    assert feedback["counts"]["feedback_records"] == 0
    assert feedback["counts"]["rejected_validations"] == 2
    reasons = {reason for row in feedback["rejected_validations"] for reason in row["reasons"]}
    assert "validation_required_checks_invalid" in reasons
    assert "validation_program_unbound" in reasons
    frontier = compile_execution_validation_frontier(graph, discovery, bundle)
    assert frontier["counts"]["experiment_required"] == 2

    tampered = deepcopy(feedback)
    tampered["counts"]["negative"] = 99
    tampered["content_sha256"] = strict_canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    assert (
        execution_capability_feedback_oracle(discovery, bundle, tampered, validations)["accepted"]
        is False
    )


def test_experimental_work_frontier_routes_negative_execution_result_through_domain_gate() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposals = {
        row["execution_domain"]: row
        for row in draft_bundle["program_proposals"].values()
    }
    failure = _execution_validation(proposals["whole_cell"], "failure")
    execution_bundle = compile_execution_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=[failure],
    )
    enzyme_bundle = compile_biocatalytic_program_bundle(
        graph, route, projection, discovery
    )
    materials = compile_program_candidate_review_materials(
        graph,
        route,
        projection,
        discovery,
        enzyme_bundle,
        execution_validations=[failure],
    )
    canonical = compile_deficit_frontier(graph)
    work = compile_experimental_work_frontier(
        canonical,
        compile_biocatalysis_validation_frontier(graph, discovery, enzyme_bundle),
        materials["execution_validation_frontier"],
        materials["mechanism_validation_frontier"],
        materials["capability_calibration"],
    )

    assert work["counts"]["execution"] == 2
    assert work["counts"]["dirty_recompute_hints"] == 1
    assert sum(bool(row["dirty_hint_ids"]) for row in work["work_items"].values()) == 1
    dirty_items = [row for row in work["work_items"].values() if row["dirty_hint_ids"]]
    clean_items = [row for row in work["work_items"].values() if not row["dirty_hint_ids"]]
    assert dirty_items[0]["scheduling"]["components"]["information_gain"][
        "dirty_exact_boundary_signal"
    ] == 0.18
    assert all(
        row["scheduling"]["components"]["information_gain"][
            "dirty_exact_boundary_signal"
        ]
        == 0.0
        for row in clean_items
    )
    assert experimental_work_frontier_oracle(
        canonical,
        compile_biocatalysis_validation_frontier(graph, discovery, enzyme_bundle),
        materials["execution_validation_frontier"],
        materials["mechanism_validation_frontier"],
        materials["capability_calibration"],
        work,
    )["accepted"] is True

    tampered_work = deepcopy(work)
    tampered_item = next(iter(tampered_work["work_items"].values()))
    tampered_item["scheduling"]["information_gain_score"] = 0.0
    tampered_item["scheduling"].pop("content_sha256")
    tampered_item["scheduling"]["content_sha256"] = strict_canonical_json_sha256(
        tampered_item["scheduling"]
    )
    tampered_item.pop("content_sha256")
    tampered_item["content_sha256"] = strict_canonical_json_sha256(tampered_item)
    tampered_work.pop("content_sha256")
    tampered_work["content_sha256"] = strict_canonical_json_sha256(tampered_work)
    assert experimental_work_frontier_oracle(
        canonical,
        compile_biocatalysis_validation_frontier(graph, discovery, enzyme_bundle),
        materials["execution_validation_frontier"],
        materials["mechanism_validation_frontier"],
        materials["capability_calibration"],
        tampered_work,
    )["accepted"] is False

    request = next(
        row["execution_request"]
        for row in work["work_items"].values()
        if row["program_id"] == proposals["whole_cell"]["program_id"]
    )
    result = build_experiment_execution_result(
        request,
        result_id="experiment-result:whole-cell-failure",
        executor_id="fixture-executor",
        executor_version="1",
        status="failure",
        artifact_refs=[
            {"sha256": "c" * 64, "media_type": "application/json", "role": "raw_assay"}
        ],
        domain_validation_candidate=failure,
        failure_reasons=["requested_product_not_observed"],
    )
    audit = audit_experiment_execution_result(request, result)

    assert audit["accepted_for_domain_gate"] is True
    assert release_experiment_validation_candidate(request, result) == failure
    feedback = compile_execution_capability_feedback(
        discovery, execution_bundle, [failure]
    )
    assert feedback["counts"]["negative"] == 1
    assert all(row["grants_validation"] is False for row in feedback["feedback"].values())


def test_validation_partition_routes_only_the_explicit_execution_schema() -> None:
    biocatalytic, execution, mechanism = partition_program_validations(
        [
            {"schema_version": "biocatalysis_program_validation.v1"},
            {"schema_version": "execution_program_validation.v1"},
            {"schema_version": "mechanism_program_validation.v1"},
            {"schema_version": "unknown_validation.v1"},
        ]
    )

    assert [row["schema_version"] for row in execution] == ["execution_program_validation.v1"]
    assert [row["schema_version"] for row in mechanism] == ["mechanism_program_validation.v1"]
    assert [row["schema_version"] for row in biocatalytic] == [
        "biocatalysis_program_validation.v1",
        "unknown_validation.v1",
    ]


def test_execution_validation_ids_are_unique_across_the_entire_batch() -> None:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposals = list(draft_bundle["program_proposals"].values())
    validations = [_execution_validation(proposal, "success") for proposal in proposals]
    validations[1]["validation_id"] = validations[0]["validation_id"]
    validations[1] = with_execution_program_validation_digest(validations[1])
    bundle = compile_execution_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )

    assert bundle["counts"]["validated_substitutions"] == 0
    feedback = compile_execution_capability_feedback(discovery, bundle, validations)
    assert feedback["counts"]["feedback_records"] == 0
    assert feedback["counts"]["rejected_validations"] == 2
    assert all(
        "validation_id_duplicate" in row["reasons"] for row in feedback["rejected_validations"]
    )


def _claim_store_inputs() -> tuple[dict, dict, dict, dict, list[dict]]:
    graph, route, projection, discovery, draft_bundle = _materials()
    proposals = {row["execution_domain"]: row for row in draft_bundle["program_proposals"].values()}
    validations = [
        _execution_validation(proposals["whole_cell"], "success"),
        _execution_validation(proposals["hybrid"], "failure"),
        _execution_validation(proposals["hybrid"], "inconclusive"),
    ]
    return graph, route, projection, discovery, validations


def test_experimental_claim_store_is_explicit_nonempty_and_idempotent(tmp_path) -> None:
    graph, route, projection, discovery, validations = _claim_store_inputs()
    original_graph = deepcopy(graph)
    store = ExperimentalClaimStore(
        run_id=graph["run_id"],
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )
    inputs = {
        "graph": graph,
        "route": route,
        "projection": projection,
        "discovery": discovery,
    }

    with pytest.raises(ExperimentalClaimAdmissionDisabled, match="explicit_enable_required"):
        store.admit(**inputs, validations=validations)
    with pytest.raises(ExperimentalClaimStoreError, match="requires_observation"):
        store.admit(
            **inputs,
            validations=[],
            enable_experimental_claim_admission=True,
        )

    admitted = store.admit(
        **inputs,
        validations=validations,
        enable_experimental_claim_admission=True,
    )
    repeated = store.admit(
        **inputs,
        validations=list(reversed(validations)),
        enable_experimental_claim_admission=True,
    )

    assert admitted["created"] is True
    assert repeated["created"] is False
    assert admitted["event"]["counts"]["claims"] == 3
    assert admitted["event"]["counts"]["positive"] == 1
    assert admitted["event"]["counts"]["negative"] == 1
    assert admitted["event"]["counts"]["inconclusive"] == 1
    assert admitted["event"]["semantics"]["cannot_create_canonical_reaction_proof"] is True
    assert admitted["store"]["oracle"]["accepted"] is True
    assert store.replay()["event_count"] == 1
    assert graph == original_graph


def test_experimental_claim_store_concurrent_admission_publishes_one_event(tmp_path) -> None:
    graph, route, projection, discovery, validations = _claim_store_inputs()
    store = ExperimentalClaimStore(
        run_id=graph["run_id"],
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )

    def admit(_: int) -> dict:
        return store.admit(
            graph=graph,
            route=route,
            projection=projection,
            discovery=discovery,
            validations=validations,
            enable_experimental_claim_admission=True,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(admit, range(12)))

    assert sum(result["created"] is True for result in results) == 1
    assert store.replay()["event_count"] == 1


def test_experimental_claim_store_rejects_claim_set_cas_tampering(tmp_path) -> None:
    graph, route, projection, discovery, validations = _claim_store_inputs()
    store = ExperimentalClaimStore(
        run_id=graph["run_id"],
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )
    admitted = store.admit(
        graph=graph,
        route=route,
        projection=projection,
        discovery=discovery,
        validations=validations,
        enable_experimental_claim_admission=True,
    )
    digest = admitted["event"]["claim_set_ref"]["sha256"]
    store.artifacts.object_path(digest).write_bytes(b"{}")

    with pytest.raises(ExperimentalClaimStoreCorruption, match="artifact_replay_failed"):
        store.replay()


def test_experimental_claim_store_rejects_identity_and_event_tampering(tmp_path) -> None:
    graph, route, projection, discovery, validations = _claim_store_inputs()
    mismatched = ExperimentalClaimStore(
        run_id="different-run",
        run_dir=tmp_path / "mismatched-run",
        artifacts=ArtifactStore(tmp_path / "mismatched-cas"),
    )
    with pytest.raises(ExperimentalClaimStoreError, match="identity_mismatch"):
        mismatched.admit(
            graph=graph,
            route=route,
            projection=projection,
            discovery=discovery,
            validations=validations,
            enable_experimental_claim_admission=True,
        )
    assert not mismatched.root.exists()

    store = ExperimentalClaimStore(
        run_id=graph["run_id"],
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )
    store.admit(
        graph=graph,
        route=route,
        projection=projection,
        discovery=discovery,
        validations=validations,
        enable_experimental_claim_admission=True,
    )
    event_path = next(store.event_root.glob("*/*.json"))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["claim_ids"] = ["claim:forged"]
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(ExperimentalClaimStoreCorruption, match="event_content_digest_invalid"):
        store.replay()
