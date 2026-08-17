from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
)
from cascade_planner.application.proof_portfolio import (
    PortfolioConfig,
    compile_proof_portfolio,
    publish_proof_portfolio,
    validate_module_replacement,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
    normalize_source_binding,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.worker_runtime import (
    WorkerBudget,
    WorkerCommand,
    WorkerRuntime,
)
from cascade_planner.application.route_strategy_value import (
    compile_evidence_maturity_vector,
    compile_strategic_value_vector,
)


TARGET = "CCOC(C)=O"
ROUTES = (
    {
        "product_smiles": TARGET,
        "precursor_smiles": ["CCO", "CC(=O)Cl"],
        "mapped_reaction_smiles": (
            "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>"
            "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
        ),
    },
    {
        "product_smiles": TARGET,
        "precursor_smiles": ["CCO", "CC(=O)O"],
        "mapped_reaction_smiles": (
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>"
            "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
        ),
    },
)


def test_strategic_value_and_evidence_maturity_are_independent_axes() -> None:
    graph = {
        "molecules": {
            "mol:target": {"canonical_smiles": TARGET},
            "mol:a": {"canonical_smiles": "CCO"},
            "mol:b": {"canonical_smiles": "CC(=O)Cl"},
        },
        "edges": {
            "edge:root": {
                "product_molecule_id": "mol:target",
                "precursor_molecule_ids": ["mol:a", "mol:b"],
                "source_binding_ids": [],
            }
        }
    }
    card = {
        "key_bond_changes": ["form C-O bond"],
        "skeleton_change_class": "fragment union",
        "key_forward_transformation": "convergent coupling",
        "expected_complexity_drop": "high",
        "stereochemical_plan": "substrate controlled",
        "protection_policy": "avoid protection",
    }
    first = compile_strategic_value_vector(
        graph,
        edge_ids=["edge:root"],
        root_edge_ids=["edge:root"],
        strategy_card=card,
        convergence_score=0.0,
    )
    graph["edges"]["edge:root"]["source_binding_ids"] = ["source:1", "source:2"]
    second = compile_strategic_value_vector(
        graph,
        edge_ids=["edge:root"],
        root_edge_ids=["edge:root"],
        strategy_card=card,
        convergence_score=0.0,
    )
    evidence = compile_evidence_maturity_vector(
        reaction_feasibility_rate=0.25,
        exact_evidence_rate=0.0,
        condition_completeness_rate=0.0,
        source_independence_met=False,
    )

    assert first == second
    assert first["score"] > evidence["score"]
    assert first["structure_metrics"]["known"] is True
    assert first["basis"] == "canonical_root_structures_and_strategy_edit_identity"
    assert evidence["basis"] == "host_proof_and_source_records_only"


def test_declared_complexity_label_does_not_change_structural_strategy_score() -> None:
    graph = {
        "molecules": {
            "mol:target": {"canonical_smiles": "c1ccc2ccccc2c1"},
            "mol:a": {"canonical_smiles": "c1ccccc1"},
            "mol:b": {"canonical_smiles": "C=CC=C"},
        },
        "edges": {
            "edge:root": {
                "product_molecule_id": "mol:target",
                "precursor_molecule_ids": ["mol:a", "mol:b"],
            }
        },
    }
    card = {
        "key_bond_changes": ["map 1-map 2"],
        "stereochemical_plan": "not applicable",
        "protection_policy": "avoid protection",
        "expected_complexity_drop": "low",
    }

    low = compile_strategic_value_vector(
        graph,
        edge_ids=["edge:root"],
        root_edge_ids=["edge:root"],
        strategy_card=card,
        convergence_score=0.0,
    )
    high = compile_strategic_value_vector(
        graph,
        edge_ids=["edge:root"],
        root_edge_ids=["edge:root"],
        strategy_card={**card, "expected_complexity_drop": "high"},
        convergence_score=0.0,
    )

    assert low["score"] == high["score"]
    assert low["complexity_drop"] == high["complexity_drop"]
    assert low["declared_complexity_drop"] == "low"
    assert high["declared_complexity_drop"] == "high"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash(value: dict) -> dict:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


def _command(
    kernel: RunKernel,
    worker_type: str,
    payload: dict,
    *,
    task_kind: str,
    suffix: str,
    artifact_refs: tuple[dict, ...] = (),
) -> WorkerCommand:
    return WorkerCommand(
        command_id=f"{worker_type}:{suffix}",
        run_id=kernel.spec.run_id,
        worker_type=worker_type,
        input_revision=kernel.state.graph_revision,
        idempotency_key=f"{worker_type}:{suffix}",
        payload=payload,
        budget=WorkerBudget(task_kind=task_kind),
        dependency_revisions={
            "graph_revision": kernel.state.graph_revision,
            "evidence_revision": kernel.state.evidence_revision,
        },
        artifact_refs=artifact_refs,
    )


def _build_closed_graph(
    tmp_path: Path,
    *,
    split_route_families: bool = False,
) -> tuple[RunKernel, dict]:
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=3,
        require_all_selected_leaves_stock_closed=True,
        minimum_independent_source_groups=2,
        require_distinct_edge_sets=True,
    )
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="proof-portfolio",
            target_name="ethyl acetate",
            target_smiles=TARGET,
            acceptance=acceptance,
            created_at="2026-07-13T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=16,
                    max_attempt_runs=32,
                ),
                max_total_tasks=64,
            ),
        ),
    )
    kernel.start()
    store = CanonicalHypergraphStore(kernel)
    route_family_ids = (
        ("family:acyl-chloride", "family:acetic-acid")
        if split_route_families
        else (
            "family:alternative-acyl-donors",
            "family:alternative-acyl-donors",
        )
    )
    route_families = tuple(
        {
            "route_family_id": family_id,
            "strategic_disconnection": "late ester bond formation",
        }
        for family_id in dict.fromkeys(route_family_ids)
    )
    store.apply(
        CanonicalIngestionBatch(
            route_families=route_families
        ),
        idempotency_key="route-family",
    )

    sources = (
        {
            "source_kind": "patent",
            "source_ref": "patent:US2020123456A1",
            "title": "Independent patent fixture",
        },
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/autoplanner.fixture",
            "title": "Independent paper fixture",
        },
    )
    extraction_refs: list[dict] = []
    for source_index, source in enumerate(sources, start=1):
        binding = normalize_source_binding(source)
        extraction_refs.append(
            kernel.artifacts.put_json(
                {
                    "schema_version": "structured_exact_row_extraction.v1",
                    "source_binding_id": binding["binding_id"],
                    "extractor": {
                        "producer_kind": "deterministic_structure_parser",
                        "producer_id": f"tests.proof_portfolio.{source_index}",
                        "version": "1.0.0",
                    },
                    "rows": [
                        {
                            "product_smiles": route["product_smiles"],
                            "reactant_smiles": route["precursor_smiles"],
                            "location_ref": f"Source {source_index}, example {route_index}",
                            "conditions": {"temperature_c": 20 + route_index},
                        }
                        for route_index, route in enumerate(ROUTES, start=1)
                    ],
                },
                logical_name=f"exact-{source_index}.json",
                producer="tests.proof_portfolio",
            ).to_dict()
        )
    inventory_ref = kernel.artifacts.put_json(
        {
            "schema_version": "versioned_inventory_snapshot.v1",
            "adapter_version": "tests.inventory.v1",
            "inventory_version": "2026-07-13",
            "retrieved_at": "2026-07-13T00:00:00Z",
            "offers": [
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": catalog,
                    "smiles": smiles,
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                }
                for catalog, smiles in (
                    ("ETHANOL", "CCO"),
                    ("ACETYL-CHLORIDE", "CC(=O)Cl"),
                    ("ACETIC-ACID", "CC(=O)O"),
                )
            ],
        },
        logical_name="inventory.json",
        producer="tests.inventory",
    ).to_dict()
    runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={
            **{
                value["sha256"]: "structured_exact_row_extraction"
                for value in extraction_refs
            },
            inventory_ref["sha256"]: "inventory_snapshot_set",
        },
    )

    proposals = tuple(
        {
            "product_smiles": route["product_smiles"],
            "precursor_smiles": route["precursor_smiles"],
            "origin_kind": "template",
            "proposal_id": f"proposal:{index}",
            "route_family_id": route_family_ids[index - 1],
        }
        for index, route in enumerate(ROUTES, start=1)
    )
    materialized = tuple(
        runtime.execute(command) for command in store.materialization_commands(proposals)
    )
    assert len(materialized) == 2
    validations = tuple(
        runtime.execute(
            _command(
                kernel,
                "validate_reaction",
                {
                    "candidate": result.payload,
                    "mapped_reaction_smiles": route["mapped_reaction_smiles"],
                },
                task_kind="validation",
                suffix=f"route-{index}",
            )
        )
        for index, (route, result) in enumerate(
            zip(ROUTES, materialized, strict=True),
            start=1,
        )
    )
    assert all(value.status == "completed" for value in validations)
    evidence_results = []
    for source_index, (source, extraction_ref) in enumerate(
        zip(sources, extraction_refs, strict=True),
        start=1,
    ):
        batch = runtime.execute_pipeline(
            _command(
                kernel,
                "discover_sources",
                {
                    "sources": [
                        {
                            **source,
                            "extraction_artifact_sha256": extraction_ref["sha256"],
                        }
                    ]
                },
                task_kind="evidence",
                suffix=f"source-{source_index}",
                artifact_refs=(extraction_ref,),
            )
        )
        evidence_results.extend(batch.results)
    stock = runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": TARGET,
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:ethanol", "smiles": "CCO"},
                    {"leaf_id": "leaf:acetyl-chloride", "smiles": "CC(=O)Cl"},
                    {"leaf_id": "leaf:acetic-acid", "smiles": "CC(=O)O"},
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T12:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="all-leaves",
            artifact_refs=(inventory_ref,),
        )
    )
    result = store.apply(
        CanonicalIngestionBatch(
            worker_results=(
                stock,
                *reversed(evidence_results),
                *reversed(validations),
                *reversed(materialized),
            )
        ),
        worker_runtime=runtime,
        idempotency_key="all-proof-facts",
    )
    return kernel, result["graph"]


def test_two_route_portfolio_is_proof_stitched_and_published(tmp_path: Path) -> None:
    kernel, graph = _build_closed_graph(tmp_path)

    portfolio = compile_proof_portfolio(graph, acceptance_spec=kernel.spec.acceptance)

    assert portfolio["closeout"]["decision"] == "accepted"
    assert portfolio["accepted"] is True
    assert portfolio["deficits"] == []
    assert len(portfolio["selected_routes"]) == 2
    assert all(route["complete"] is True for route in portfolio["selected_routes"])
    assert all(route["minimum_edge_proof_level"] == 3 for route in portfolio["selected_routes"])
    assert all(route["all_leaves_stock_closed"] is True for route in portfolio["selected_routes"])
    assert portfolio["metrics"]["distinct_complete_edge_set_count"] == 2
    assert portfolio["metrics"]["complete_strategic_disconnection_count"] == 2
    assert len(portfolio["route_modules"]) == 1
    assert {
        value["proof_level"]
        for value in portfolio["route_modules"][0]["alternatives"]
    } == {3}
    for proof in portfolio["edge_proofs"].values():
        assert proof["reaction_proof_digests"]
        assert len(proof["exact_record_ids"]) == 2
        assert len(proof["source_binding_ids"]) == 2
    for proof in portfolio["leaf_proofs"].values():
        assert proof["stock_observation_id"]
        assert proof["inventory_snapshot_set_id"]
    for route in portfolio["route_candidates"]:
        assert route["content_sha256"] == _digest(
            {key: value for key, value in route.items() if key != "content_sha256"}
        )
        vector = route["pareto_objective_vector"]
        assert vector["schema_version"] == "route_pareto_objective_vector.v1"
        assert set(vector["axes"]) == {
            "strategic_value",
            "evidence_maturity",
            "topology_closure",
            "stock_closure",
            "reaction_feasibility",
            "proof_evidence",
            "condition_completeness",
            "route_diversity",
            "cost_length",
            "program_readiness",
        }
        assert vector["axes"]["cost_length"]["known"] is False
        assert vector["axes"]["cost_length"]["unknown_cost_is_not_zero"] is True
        assert vector["axes"]["program_readiness"]["applicability"] == "not_applicable"
        assert vector["axes"]["program_readiness"]["score"] == 1.0
        assert vector["axes"]["strategic_value"]["evidence_independent"] is True
        assert (
            vector["axes"]["evidence_maturity"]["strategy_wording_independent"]
            is True
        )
        assert vector["semantics"]["scalar_utility_is_display_only"] is True

    published = publish_proof_portfolio(
        kernel,
        graph,
        idempotency_key="accepted-revision",
    )
    assert published["acceptance_report"]["accepted"] is True
    assert kernel.state.acceptance_report["accepted"] is True
    assert kernel.state.deficits == ()
    assert kernel.decide_stop().decision == "completed"


def test_exact_sources_do_not_replace_reaction_proof_and_removal_reopens_one_gap(
    tmp_path: Path,
) -> None:
    kernel, graph = _build_closed_graph(tmp_path)
    changed = deepcopy(graph)
    edge_id = sorted(changed["edges"])[0]
    edge = dict(changed["edges"][edge_id])
    assert len(edge["exact_record_ids"]) == 2
    edge["reaction_proofs"] = []
    changed["edges"][edge_id] = _rehash(edge)

    portfolio = compile_proof_portfolio(changed, acceptance_spec=kernel.spec.acceptance)
    proof = portfolio["edge_proofs"][edge_id]
    edge_deficits = [
        value for value in portfolio["deficits"] if value["kind"] == "validation"
    ]

    assert proof["exact_source_bound"] is True
    assert proof["reaction_validated"] is False
    assert proof["accepted"] is False
    assert proof["achieved_level"] == 1
    assert {value["object_id"] for value in edge_deficits} == {edge_id}
    assert portfolio["closeout"]["decision"] == "unresolved"
    assert portfolio["closeout"]["complete_route_count"] == 1


def test_removing_stock_reopens_only_the_missing_leaf(tmp_path: Path) -> None:
    kernel, graph = _build_closed_graph(tmp_path)
    changed = deepcopy(graph)
    ethanol_id = next(
        molecule_id
        for molecule_id, molecule in changed["molecules"].items()
        if molecule["canonical_smiles"] == "CCO"
    )
    observation_id = changed["molecules"][ethanol_id]["active_stock_observation_id"]
    changed["stock_observations"].pop(observation_id)

    portfolio = compile_proof_portfolio(changed, acceptance_spec=kernel.spec.acceptance)
    stock_deficits = [
        value for value in portfolio["deficits"] if value["kind"] == "stock"
    ]

    assert {value["object_id"] for value in stock_deficits} == {ethanol_id}
    assert len(stock_deficits[0]["metadata"]["route_ids"]) == 2
    assert portfolio["closeout"]["decision"] == "unresolved"
    assert portfolio["closeout"]["complete_route_count"] == 0


def test_conflict_budget_and_corruption_have_explicit_closeout_states(
    tmp_path: Path,
) -> None:
    kernel, graph = _build_closed_graph(tmp_path)
    conflicted = deepcopy(graph)
    edge_id = sorted(conflicted["edges"])[0]
    edge = conflicted["edges"][edge_id]
    conflict = _rehash(
        {
            "conflict_id": "conflict:fixture",
            "subject_id": edge["edge_digest"],
            "record_ids": list(edge["exact_record_ids"]),
            "status": "open",
            "reasons": ["incompatible_exact_conditions"],
        }
    )
    conflicted["conflicts"][conflict["conflict_id"]] = conflict

    unresolved = compile_proof_portfolio(
        conflicted,
        acceptance_spec=kernel.spec.acceptance,
    )
    exhausted = compile_proof_portfolio(
        conflicted,
        acceptance_spec=kernel.spec.acceptance,
        budget_exhausted=True,
    )
    invalid_graph = deepcopy(graph)
    invalid_graph["edges"][edge_id]["content_sha256"] = "0" * 64
    invalid = compile_proof_portfolio(
        invalid_graph,
        acceptance_spec=kernel.spec.acceptance,
    )

    assert unresolved["closeout"]["decision"] == "unresolved"
    assert any(value["kind"] == "conflict" for value in unresolved["deficits"])
    assert exhausted["closeout"]["decision"] == "budget_exhausted"
    assert exhausted["accepted"] is False
    assert invalid["closeout"]["decision"] == "invalid"
    assert invalid["accepted"] is False


def test_alternative_step_is_a_canonical_module_patch_not_a_route_copy(
    tmp_path: Path,
) -> None:
    kernel, graph = _build_closed_graph(tmp_path)
    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=kernel.spec.acceptance,
        config=PortfolioConfig(minimum_routes_to_show=2, maximum_routes_to_show=5),
    )
    module = portfolio["route_modules"][0]
    route = portfolio["selected_routes"][0]
    current = route["module_selections"][module["module_id"]]
    replacement = next(
        value["edge_id"]
        for value in module["alternatives"]
        if value["edge_id"] != current
    )

    patch = validate_module_replacement(
        portfolio,
        route_id=route["route_id"],
        module_id=module["module_id"],
        replacement_edge_id=replacement,
    )
    same = validate_module_replacement(
        portfolio,
        route_id=route["route_id"],
        module_id=module["module_id"],
        replacement_edge_id=current,
    )

    assert patch["accepted"] is True
    assert patch["remove_edge_id"] == current
    assert patch["add_edge_id"] == replacement
    assert patch["semantics"]["patch_does_not_duplicate_entire_route"] is True
    assert same["accepted"] is False
    assert "replacement_edge_is_already_selected" in same["reasons"]


def test_cross_family_same_product_edges_form_restitched_replacement_module(
    tmp_path: Path,
) -> None:
    kernel, graph = _build_closed_graph(tmp_path, split_route_families=True)

    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=kernel.spec.acceptance,
        config=PortfolioConfig(minimum_routes_to_show=2, maximum_routes_to_show=5),
    )

    assert len(portfolio["route_modules"]) == 1
    module = portfolio["route_modules"][0]
    assert module["route_family_id"] == ""
    candidate_family_ids = sorted(
        {route["route_family_id"] for route in portfolio["route_candidates"]}
    )
    assert len(candidate_family_ids) == 2
    assert module["route_family_ids"] == candidate_family_ids
    assert module["semantics"]["cross_family_replacement_supported"] is True
    assert module["semantics"]["full_restitched_candidate_route_required"] is True
    selections = {
        route["route_id"]: route["module_selections"][module["module_id"]]
        for route in portfolio["route_candidates"]
    }
    assert len(set(selections.values())) == 2
    assert all(route["complete"] is True for route in portfolio["route_candidates"])

    base, replacement = portfolio["selected_routes"]
    patch = validate_module_replacement(
        portfolio,
        route_id=base["route_id"],
        module_id=module["module_id"],
        replacement_edge_id=selections[replacement["route_id"]],
    )
    assert patch["accepted"] is True
