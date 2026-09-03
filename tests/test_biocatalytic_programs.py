from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from cascade_planner.application.biocatalytic_program_store import (
    BiocatalyticProgramStore,
    BiocatalyticProgramStoreCorruption,
)
from cascade_planner.application.biocatalytic_programs import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    BiocatalyticProgramError,
    biocatalytic_program_bundle_oracle,
    compile_biocatalytic_program_bundle,
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.biocatalysis_validation_frontier import (
    compile_biocatalysis_validation_frontier,
)
from cascade_planner.application.route_innovation_discovery import (
    discover_route_innovations,
)
from cascade_planner.application.program_route_candidate_contracts import (
    program_route_candidate_counts,
    validate_program_route_candidate_set,
)
from cascade_planner.application.program_route_candidates import (
    compile_program_route_candidate_set,
)
from cascade_planner.application.program_route_optimizer import (
    optimize_program_route_candidates,
    program_route_portfolio_oracle,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.orchestration.program_candidate_review_materials import (
    compile_program_candidate_review_materials,
)
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _fixture(*, branch_from_internal_state: bool = False) -> tuple[dict, dict]:
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
    edge_ids = list(edges)
    if branch_from_internal_state:
        molecules["m:branch"] = {
            "canonical_smiles": "CC(C)C1CCCCC1",
            "stock_observation_ids": [],
        }
        edges["edge:branch"] = {
            "precursor_molecule_ids": ["m:1"],
            "product_molecule_id": "m:branch",
            "innovation_boundary_proof_level": 1,
            "procedure_record_ids": [],
            "source_binding_ids": [],
            "exact_record_ids": [],
            "reaction_proofs": [],
        }
        edge_ids.append("edge:branch")
    route = {
        "route_id": "route:fixture",
        "route_family_id": "family:fixture",
        "edge_ids": edge_ids,
        "reported_source_refs": ["doi:10.1000/anchor"],
    }
    graph = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "biocatalytic-program-fixture",
        "revision": 3,
        "scientific_sha256": "fixture-scientific-revision-3",
        "target_molecule_id": "m:3",
        "molecules": molecules,
        "edges": edges,
        "route_families": {
            "family:fixture": {
                "edge_ids": edge_ids,
                "closed": False,
            }
        },
    }
    return graph, route


def _capability() -> dict:
    return {
        "capability_id": "fixture:cyclic-ketone-reduction",
        "enzyme": {"classes": ["ketoreductase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "preserved_motifs": ["alkene", "ester"],
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.3,
            "max_abs_heavy_atom_delta": 0,
            "min_substrate_carbons": 6,
            "min_substrate_rings": 1,
            "min_window_steps": 2,
            "max_window_steps": 4,
            "reject_unlisted_motif_changes": True,
        },
        "selectivity_objective": "Reduce one cyclic ketone to the specified alcohol.",
        "substrate_scope_basis": "fixture analog only",
        "precedent_refs": ["doi:10.1000/fixture"],
    }


def _bundle(*, branch: bool = False, validations: list[dict] | None = None) -> dict:
    graph, route = _fixture(branch_from_internal_state=branch)
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    return compile_biocatalytic_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations or [],
    )


def _validated_materials() -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    initial = compile_biocatalytic_program_bundle(
        graph,
        route,
        projection,
        discovery,
    )
    proposal = next(iter(initial["program_proposals"].values()))
    validation = with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "validation:durable-fixture",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:durable-exact-substrate"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_fraction": 0.86},
        }
    )
    validations = [validation]
    bundle = compile_biocatalytic_program_bundle(
        graph,
        route,
        projection,
        discovery,
        validations=validations,
    )
    return graph, route, projection, discovery, bundle, validations


def test_superstep_is_a_boundary_program_with_explicit_fallback() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )

    bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    proposal = next(iter(bundle["program_proposals"].values()))
    candidate = next(iter(bundle["route_candidates"].values()))
    oracle = biocatalytic_program_bundle_oracle(
        graph,
        route,
        projection,
        discovery,
        bundle,
    )

    assert proposal["proposal_kind"] == "biocatalytic_superstep"
    assert proposal["input_state_ids"] == ["state:m:0"]
    assert proposal["output_state_ids"] == ["state:m:3"]
    assert proposal["equivalent_reference_span"] == ["edge:0", "edge:1", "edge:2"]
    assert proposal["chemical_step_equivalent_count"] == 3
    assert proposal["isolated_operation_count"] == 1
    assert proposal["status"] == "proposal_only"
    assert candidate["fallback_program_ids"] == [
        "program:edge:0",
        "program:edge:1",
        "program:edge:2",
    ]
    assert candidate["physical_step_count"] == 1
    assert candidate["chemical_step_equivalent_count"] == 3
    assert candidate["net_step_savings"] == 2
    assert candidate["eligible_for_program_optimizer"] is False
    assert candidate["eligible_for_route_completion"] is False
    assert oracle["accepted"] is True
    assert all(oracle["checks"].values())


def test_unvalidated_program_compiles_to_non_authoritative_assay_plan() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)

    frontier = compile_biocatalysis_validation_frontier(graph, discovery, bundle)
    plan = next(iter(frontier["plans"].values()))

    assert frontier["counts"] == {"experiment_required": 1, "validation_granted": 0}
    assert plan["exact_boundary"]["input_states"][0]["canonical_smiles"] == ("CC(=O)C1CCCCC1")
    assert plan["exact_boundary"]["output_states"][0]["canonical_smiles"] == ("CC(O)C1CCCCC1")
    assert plan["screen_matrix"]["enzyme_candidates"]["classes"] == ["ketoreductase"]
    assert {row["assay_id"] for row in plan["required_assays"]} >= {
        "endpoint_conversion",
        "regio_and_stereoselectivity",
        "cofactor_regeneration_closure",
    }
    assert plan["status"] == "experiment_required"
    assert plan["grants_validation"] is False
    assert plan["eligible_for_shadow_admission"] is False
    assert plan["eligible_for_route_completion"] is False

    tampered = deepcopy(bundle)
    tampered["counts"]["program_proposals"] = 999
    with pytest.raises(BiocatalyticProgramError, match="bundle_digest_invalid"):
        compile_biocatalysis_validation_frontier(graph, discovery, tampered)


def test_validated_program_leaves_no_validation_frontier_plan() -> None:
    graph, _, _, discovery, bundle, _ = _validated_materials()

    frontier = compile_biocatalysis_validation_frontier(graph, discovery, bundle)

    assert frontier["plans"] == {}
    assert frontier["counts"] == {"experiment_required": 0, "validation_granted": 0}


def test_unvalidated_superstep_remains_on_exploration_front_only() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)

    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
    )
    portfolio = optimize_program_route_candidates(candidate_set)
    candidates = candidate_set["candidates"]
    baseline_id = next(key for key, row in candidates.items() if row["source_kind"] == "baseline")
    enzyme_id = next(key for key, row in candidates.items() if row["source_kind"] == "biocatalytic")

    assert set(portfolio["profiles"]["exploration"]["pareto_front_ids"]) == {
        baseline_id,
        enzyme_id,
    }
    assert portfolio["profiles"]["shadow_optimizer"]["pareto_front_ids"] == [baseline_id]
    assert candidates[enzyme_id]["metrics"]["physical_operation_count"] == 1
    assert candidates[enzyme_id]["metrics"]["specialized_validation_deficit_count"] == 1
    assert candidates[enzyme_id]["eligibility"]["exploration_visible"] is True
    assert candidates[enzyme_id]["eligibility"]["shadow_optimizer"] is False
    assert portfolio["semantics"]["source_kind_is_not_an_objective"] is True
    assert all(row["metric"] != "source_kind" for row in portfolio["objective_definitions"])


def test_validated_superstep_can_pareto_dominate_longer_fallback() -> None:
    graph, route, projection, discovery, bundle, validations = _validated_materials()

    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
        validations=validations,
    )
    portfolio = optimize_program_route_candidates(candidate_set)
    oracle = program_route_portfolio_oracle(candidate_set, portfolio)
    candidates = candidate_set["candidates"]
    baseline_id = next(key for key, row in candidates.items() if row["source_kind"] == "baseline")
    enzyme_id = next(key for key, row in candidates.items() if row["source_kind"] == "biocatalytic")

    assert portfolio["profiles"]["exploration"]["pareto_front_ids"] == [enzyme_id]
    assert portfolio["profiles"]["shadow_optimizer"]["pareto_front_ids"] == [enzyme_id]
    assert portfolio["candidate_evaluations"][enzyme_id]["dominates_in_exploration"] == [
        baseline_id
    ]
    assert candidates[enzyme_id]["eligibility"]["route_completion"] is False
    assert oracle["accepted"] is True

    tampered = deepcopy(portfolio)
    tampered["profiles"]["exploration"]["pareto_front_ids"] = [baseline_id]
    rejected = program_route_portfolio_oracle(candidate_set, tampered)
    assert rejected["accepted"] is False
    assert "content_digest_valid" in rejected["reasons"]
    assert "portfolio_equal" in rejected["reasons"]


def test_program_source_kind_does_not_change_pareto_layer() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    candidate_set = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
    )
    baseline_id = next(
        key for key, row in candidate_set["candidates"].items() if row["source_kind"] == "baseline"
    )
    clone_id = "program-route:mechanism:fairness-control"
    clone = deepcopy(candidate_set["candidates"][baseline_id])
    clone["candidate_id"] = clone_id
    clone["source_kind"] = "mechanism"
    clone = _redigest(clone)
    expanded = deepcopy(candidate_set)
    expanded["candidates"][clone_id] = clone
    expanded["counts"] = program_route_candidate_counts(expanded["candidates"])
    expanded = _redigest(expanded)

    portfolio = optimize_program_route_candidates(expanded)
    baseline_layer = portfolio["candidate_evaluations"][baseline_id]["profile_pareto_layers"]
    clone_layer = portfolio["candidate_evaluations"][clone_id]["profile_pareto_layers"]

    assert baseline_layer == clone_layer
    assert (
        clone_id not in portfolio["candidate_evaluations"][baseline_id]["dominates_in_exploration"]
    )
    assert (
        baseline_id not in portfolio["candidate_evaluations"][clone_id]["dominates_in_exploration"]
    )


def test_program_candidate_contract_rejects_redigested_structural_tampering() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    bundle = compile_biocatalytic_program_bundle(graph, route, projection, discovery)
    source = compile_program_route_candidate_set(
        graph,
        route,
        projection,
        discovery,
        bundle,
    )
    baseline_id = next(
        key for key, row in source["candidates"].items() if row["source_kind"] == "baseline"
    )
    enzyme_id = next(
        key for key, row in source["candidates"].items() if row["source_kind"] == "biocatalytic"
    )

    duplicate_program = deepcopy(source)
    duplicate_program["candidates"][baseline_id]["program_ids"].append(
        duplicate_program["candidates"][baseline_id]["program_ids"][0]
    )
    duplicate_program["candidates"][baseline_id] = _redigest(
        duplicate_program["candidates"][baseline_id]
    )
    duplicate_program = _redigest(duplicate_program)

    injected_field = deepcopy(source)
    injected_field["candidates"][baseline_id]["hidden_weight"] = 100
    injected_field["candidates"][baseline_id] = _redigest(injected_field["candidates"][baseline_id])
    injected_field = _redigest(injected_field)

    promoted_without_gate = deepcopy(source)
    promoted_without_gate["candidates"][enzyme_id]["eligibility"]["process_ready"] = True
    promoted_without_gate["candidates"][enzyme_id] = _redigest(
        promoted_without_gate["candidates"][enzyme_id]
    )
    promoted_without_gate["counts"]["process_ready"] += 1
    promoted_without_gate = _redigest(promoted_without_gate)

    assert any(
        reason.startswith("program_candidate_programs_invalid")
        for reason in validate_program_route_candidate_set(duplicate_program)
    )
    assert any(
        reason.startswith("program_candidate_fields_invalid")
        for reason in validate_program_route_candidate_set(injected_field)
    )
    assert any(
        reason.startswith("program_candidate_eligibility_invalid")
        for reason in validate_program_route_candidate_set(promoted_without_gate)
    )


def test_specialized_validation_can_ready_substitution_but_not_complete_route() -> None:
    initial = _bundle()
    proposal = next(iter(initial["program_proposals"].values()))
    validation = with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "biovalidation:fixture:1",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:exact-substrate-screen:1"],
            "condition_record_ids": ["conditions:screen:1"],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_percent": 84.0, "desired_selectivity_percent": 97.0},
        }
    )

    bundle = _bundle(validations=[validation])
    admitted = next(iter(bundle["program_proposals"].values()))
    route = next(iter(bundle["route_candidates"].values()))

    assert admitted["validation_gate"]["accepted"] is True
    assert admitted["status"] == "admission_ready"
    assert admitted["eligible_for_shadow_admission"] is True
    assert route["substitution_validated"] is True
    assert route["eligible_for_program_optimizer"] is True
    assert route["eligible_for_route_completion"] is False


def test_validated_biocatalysis_projects_one_exact_boundary_claim() -> None:
    graph, route, projection, discovery, bundle, validations = _validated_materials()

    review = compile_program_candidate_review_materials(
        graph,
        route,
        projection,
        discovery,
        bundle,
        biocatalytic_validations=validations,
    )

    claims = review["experimental_claims"]
    assert claims["counts"]["claims"] == 1
    assert claims["counts"]["biocatalytic"] == 1
    assert claims["counts"]["positive"] == 1
    assert claims["counts"]["negative"] == 0
    assert review["experimental_claims_oracle"]["accepted"] is True
    claim = next(iter(claims["claims"].values()))
    assert claim["interpretation_status"] == ("exact_substrate_biocatalysis_supported")
    assert claim["grants_domain_validation"] is True
    assert claim["semantics"]["does_not_grant_route_completion_or_acceptance"] is True
    assert review["capability_calibration"]["counts"]["positive"] == 1
    assert review["capability_calibration_oracle"]["accepted"] is True


def test_invalid_validation_remains_visible_without_readiness() -> None:
    initial = _bundle()
    proposal = next(iter(initial["program_proposals"].values()))
    validation = with_biocatalysis_program_validation_digest(
        {
            "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
            "validation_id": "biovalidation:fixture:forged",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:screen:forged"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_percent": 99.0},
        }
    )
    validation["outcome"]["conversion_percent"] = 100.0

    bundle = _bundle(validations=[validation])
    observed = next(iter(bundle["program_proposals"].values()))

    assert observed["status"] == "proposal_only"
    assert observed["validation_gate"]["accepted"] is False
    assert "validation_digest_invalid" in observed["validation_gate"]["audits"][0]["reasons"]
    assert bundle["counts"]["unvalidated_substitutions"] == 1


def test_interval_with_externally_consumed_internal_state_fails_closed() -> None:
    bundle = _bundle(branch=True)

    assert bundle["counts"]["program_proposals"] == 0
    assert bundle["counts"]["rejected_candidates"] >= 1
    assert any(
        "replacement_span_internal_state_has_external_consumer" in row["reasons"]
        for row in bundle["rejections"]
    )


def test_target_name_is_not_a_superstep_compilation_rule() -> None:
    graph, route = _fixture()
    projection = project_canonical_graph_to_programs(graph)
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=[_capability()],
    )
    baseline = compile_biocatalytic_program_bundle(
        graph,
        route,
        projection,
        discovery,
    )
    renamed = deepcopy(graph)
    renamed["target_name"] = "opaque-target-label"
    observed = compile_biocatalytic_program_bundle(
        renamed,
        route,
        projection,
        discovery,
    )

    assert observed == baseline


def test_biocatalytic_store_concurrent_admission_publishes_one_event(tmp_path) -> None:
    graph, route, projection, discovery, bundle, validations = _validated_materials()
    store = BiocatalyticProgramStore(
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
            bundle=bundle,
            validations=validations,
            enable_biocatalytic_program_admission=True,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(admit, range(12)))

    assert sum(result["created"] is True for result in results) == 1
    assert store.replay()["event_count"] == 1


def test_biocatalytic_store_rejects_bundle_cas_tampering(tmp_path) -> None:
    graph, route, projection, discovery, bundle, validations = _validated_materials()
    store = BiocatalyticProgramStore(
        run_id=graph["run_id"],
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )
    admitted = store.admit(
        graph=graph,
        route=route,
        projection=projection,
        discovery=discovery,
        bundle=bundle,
        validations=validations,
        enable_biocatalytic_program_admission=True,
    )
    digest = admitted["event"]["bundle_ref"]["sha256"]
    store.artifacts.object_path(digest).write_bytes(b"{}")

    with pytest.raises(BiocatalyticProgramStoreCorruption, match="artifact_replay_failed"):
        store.replay()


def _redigest(value: dict) -> dict:
    row = deepcopy(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row
