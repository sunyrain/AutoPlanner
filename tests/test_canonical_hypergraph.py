from __future__ import annotations

from pathlib import Path

from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
    canonical_scientific_projection,
    full_recompute_canonical_hypergraph,
    hypothesis_identity,
    molecule_identity,
    reaction_edge_identity,
    route_family_identity,
    source_binding_identity,
    stock_observation_identity,
)
from cascade_planner.application.deficit_frontier import (
    compile_deficit_frontier,
    frontier_scientific_projection,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
    materialization_commands_for_global_plan,
    normalize_source_binding,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.worker_runtime import (
    WorkerBudget,
    WorkerCommand,
    WorkerRuntime,
)


def _kernel(tmp_path: Path) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="canonical-graph",
            target_name="ethyl acetate",
            target_smiles="CCOC(C)=O",
            created_at="2026-07-13T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=32,
                    max_attempt_runs=64,
                ),
                max_total_tasks=64,
            ),
        ),
    )
    kernel.start()
    return kernel


def test_rejected_stock_leaf_becomes_provider_expansion_deficit() -> None:
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {
                "molecule:target": {
                    "canonical_smiles": "CCOC(C)=O",
                    "is_leaf": False,
                    "stock_closed": False,
                },
                "molecule:leaf": {
                    "canonical_smiles": "CC(=O)Cl",
                    "is_leaf": True,
                    "stock_closed": False,
                    "active_stock_observation_id": "stock:miss",
                },
            },
            "stock_observations": {
                "stock:miss": {"accepted": False, "reasons": ["catalog_miss"]}
            },
            "route_families": {
                "route:acyl": {"selected": True, "closed": False, "edge_ids": []}
            },
            "dependency_index": {
                "routes_by_entity": {"molecule:leaf": ["route:acyl"]}
            },
            "edges": {},
            "hypotheses": {},
            "conflicts": {},
        }
    )

    expansion = next(
        row for row in frontier["items"] if row["kind"] == "expansion"
    )
    assert expansion["model_allowed"] is True
    assert expansion["deterministic"] is False
    assert expansion["metadata"]["frontier_smiles"] == "CC(=O)Cl"
    assert expansion["metadata"]["provider_preferences"][0] == "chemenzy"
    assert frontier["summary"]["by_kind"]["stock"] == 0


def test_discovered_source_lifecycle_becomes_evidence_deficit() -> None:
    binding = normalize_source_binding(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/restricted",
            "title": "restricted route paper",
            "acquisition_status": "queued_for_authorized_browser",
            "proxy_request_id": "pdfreq-one",
        }
    )
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {
                "molecule:target": {
                    "canonical_smiles": "CC",
                    "is_leaf": True,
                    "stock_closed": False,
                }
            },
            "source_bindings": {binding["binding_id"]: binding},
            "exact_records": {},
            "stock_observations": {},
            "route_families": {},
            "dependency_index": {"routes_by_entity": {}},
            "edges": {},
            "hypotheses": {},
            "conflicts": {},
        }
    )

    item = next(
        value
        for value in frontier["items"]
        if value["object_id"] == binding["binding_id"]
    )
    assert item["kind"] == "evidence"
    assert item["reason"] == "source_waiting_authorized_pdf_acquisition"
    assert item["metadata"]["proxy_request_id"] == "pdfreq-one"
    assert item["model_allowed"] is False


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:acyl",
                "strategic_disconnection": "acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:acyl",
                "route_family_id": "family:acyl",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


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


def _apply_proposals(
    kernel: RunKernel,
    store: CanonicalHypergraphStore,
    runtime: WorkerRuntime,
    proposals: tuple[dict, ...],
    *,
    key: str,
) -> dict:
    commands = store.materialization_commands(proposals)
    results = tuple(runtime.execute(command) for command in commands)
    return store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key=key,
    )


def test_all_core_identities_are_canonical_and_order_independent() -> None:
    first_molecule = molecule_identity("OCC")
    second_molecule = molecule_identity("CCO")
    first_edge, first_audit = reaction_edge_identity(
        "CCOC(C)=O", ["CCO", "CC(=O)Cl"]
    )
    second_edge, second_audit = reaction_edge_identity(
        "CCOC(C)=O", ["CC(=O)Cl", "OCC"]
    )
    hypothesis, hypothesis_audit = hypothesis_identity(
        "CCOC(C)=O", ["CCO", "CC(=O)Cl"]
    )

    assert first_molecule == second_molecule
    assert first_edge == second_edge
    assert first_audit["precursor_smiles_multiset"] == second_audit[
        "precursor_smiles_multiset"
    ]
    assert hypothesis.removeprefix("hypothesis:") == first_edge.removeprefix("edge:")
    assert hypothesis_audit["accepted"] is True
    source_a = source_binding_identity(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/example",
            "independence_group": "doi:10.1000/example",
        }
    )
    source_b = source_binding_identity(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/example",
            "independence_group": "doi:10.1000/example",
            "title": "presentation-only title",
        }
    )
    stock_a = stock_observation_identity(
        {
            "leaf_id": "leaf:1",
            "canonical_smiles": "CCO",
            "inventory_snapshot_set_id": "inventory:1",
            "audited_as_of": "2026-07-13T00:00:00Z",
            "provider_result": {"content_hash": "a" * 64},
        }
    )
    target_id = first_molecule[0]
    route_a = route_family_identity(
        {"route_family_id": "family:a", "strategy": "acyl"},
        target_molecule_id=target_id,
    )
    route_b = route_family_identity(
        {"route_family_id": "family:a", "strategy": "acyl", "name": "label"},
        target_molecule_id=target_id,
    )
    assert source_a == source_b
    assert stock_a.startswith("stock:")
    assert route_a == route_b


def test_global_codex_plan_enters_real_frontier_then_materializes_once(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())

    planned = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="plan",
    )
    graph = planned["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))

    assert planned["changed"] is True
    assert hypothesis["status"] == "frontier_candidate"
    assert graph["deficit_frontier"]["summary"]["by_kind"]["materialization"] == 1
    assert kernel.state.graph_revision == 1
    assert kernel.state.accepted_expansion_count == 0

    commands = store.frontier_materialization_commands()
    results = tuple(runtime.execute(command) for command in commands)
    materialized = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialized",
    )
    graph = materialized["graph"]
    edge = next(iter(graph["edges"].values()))
    route = next(iter(graph["route_families"].values()))

    assert len(graph["edges"]) == 1
    assert edge["origin_records"][0]["origin_kind"] == "codex_global_director"
    assert next(iter(graph["hypotheses"].values()))["status"] == "materialized"
    assert route["edge_ids"] == [edge["edge_id"]]
    assert graph["deficit_frontier"]["summary"]["by_kind"]["materialization"] == 0
    assert graph["deficit_frontier"]["summary"]["by_kind"]["validation"] == 1
    assert kernel.state.graph_revision == 2


def test_codex_provider_delegation_becomes_one_canonical_expansion_deficit(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    plan["frontier_priorities"] = [
        {
            "priority_id": "priority:chemenzy:ethanol",
            "target_smiles": "OCC",
            "provider_preferences": ["chemenzy"],
            "retron_hints": ["alcohol feedstock alternatives"],
            "priority": 9,
            "rationale": "compare a local upstream module",
        }
    ]

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="plan-with-provider-frontier",
    )["graph"]

    molecule_id, _ = molecule_identity("CCO")
    molecule = graph["molecules"][molecule_id]
    expansion = next(
        item
        for item in graph["deficit_frontier"]["items"]
        if item["kind"] == "expansion" and item["object_id"] == molecule_id
    )
    assert molecule["provider_expansion_requested"] is True
    assert molecule["provider_preferences"] == ["chemenzy"]
    assert expansion["reason"] == "codex_selected_frontier_requires_local_generation"
    assert expansion["metadata"]["frontier_smiles"] == "CCO"
    assert graph["deficit_frontier"]["semantics"][
        "frontier_is_not_scientific_authority"
    ] is True


def test_codex_can_delegate_a_non_leaf_shared_intermediate(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    plan["multi_step_skeletons"][0]["steps"].append(
        {
            "step_id": "step:ethanol",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC=O"],
            "transformation_hypothesis": "carbonyl reduction",
        }
    )
    plan["frontier_priorities"] = [
        {
            "priority_id": "priority:chemenzy:acid",
            "target_smiles": "CCO",
            "provider_preferences": ["chemenzy"],
            "retron_hints": ["alternative acid construction"],
            "priority": 9,
            "rationale": "compare upstream modules around a shared node",
        }
    ]

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="plan-with-non-leaf-provider-frontier",
    )["graph"]
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    results = tuple(
        runtime.execute(command)
        for command in store.frontier_materialization_commands()
    )
    graph = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize-non-leaf-provider-frontier",
    )["graph"]

    molecule_id, _ = molecule_identity("CCO")
    molecule = graph["molecules"][molecule_id]
    assert molecule["is_leaf"] is False
    expansion = next(
        item
        for item in graph["deficit_frontier"]["items"]
        if item["kind"] == "expansion" and item["object_id"] == molecule_id
    )
    assert expansion["reason"] == "codex_selected_frontier_requires_local_generation"
    assert expansion["metadata"]["frontier_smiles"] == "CCO"


def test_one_ingestion_path_deduplicates_edges_and_preserves_all_origins(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    base = {
        "product_smiles": "CCOC(C)=O",
        "precursor_smiles": ["CCO", "CC(=O)Cl"],
    }
    result = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {**base, "origin_kind": "chemenzy", "proposal_id": "chem:1"},
            {**base, "origin_kind": "template", "proposal_id": "template:1"},
            {**base, "origin_kind": "manual", "proposal_id": "manual:1"},
        ),
        key="origins",
    )
    edge = next(iter(result["graph"]["edges"].values()))

    assert len(result["graph"]["edges"]) == 1
    assert {row["origin_kind"] for row in edge["origin_records"]} == {
        "chemenzy",
        "template",
        "manual",
    }
    assert kernel.state.graph_revision == 1
    assert kernel.state.accepted_expansion_count == 1

    repeated = _apply_proposals(
        kernel,
        store,
        runtime,
        ({**base, "origin_kind": "manual", "proposal_id": "manual:1"},),
        key="same-origin-again",
    )
    assert repeated["changed"] is False
    assert kernel.state.graph_revision == 1


def test_cycle_and_impossible_edges_are_rejected_before_graph_expansion(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    first = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["COC"],
                "origin_kind": "manual",
            },
        ),
        key="forward",
    )
    second = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "COC",
                "precursor_smiles": ["CCO"],
                "origin_kind": "manual",
            },
            {
                "product_smiles": "CCCCCCCCCCCCCCCCCCCC",
                "precursor_smiles": ["C"],
                "origin_kind": "manual",
            },
        ),
        key="rejected",
    )

    assert len(first["graph"]["edges"]) == 1
    assert second["changed"] is False
    assert len(second["graph"]["edges"]) == 1
    reasons = {reason for row in second["rejected"] for reason in row["reasons"]}
    assert "ancestor_or_target_cycle" in reasons
    assert "large_atom_jump" in reasons


def test_incremental_projection_equals_full_recompute_oracle(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    first = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
            },
            {
                "product_smiles": "CC=O",
                "precursor_smiles": ["CCO"],
                "origin_kind": "chemenzy",
            },
        ),
        key="initial",
    )
    graph = first["graph"]
    oracle = full_recompute_canonical_hypergraph(
        graph,
        acceptance_spec=kernel.spec.acceptance,
    )

    assert graph["scientific_sha256"] == oracle["scientific_sha256"]
    assert canonical_scientific_projection(graph) == canonical_scientific_projection(
        oracle
    )
    assert graph["delta"]["dirty_entity_count"] <= graph["delta"][
        "total_entity_count"
    ]


def test_worker_facts_merge_order_independently_without_false_route_closure(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="plan-for-facts",
    )
    source = {
        "source_kind": "patent",
        "source_ref": "patent:US2020123456A1",
        "title": "Fixture process",
    }
    binding = normalize_source_binding(source)
    extraction_ref = kernel.artifacts.put_json(
        {
            "schema_version": "structured_exact_row_extraction.v1",
            "source_binding_id": binding["binding_id"],
            "extractor": {
                "producer_kind": "deterministic_structure_parser",
                "producer_id": "tests.fixture",
                "version": "1.0.0",
            },
            "rows": [
                {
                    "product_smiles": "CCOC(C)=O",
                    "reactant_smiles": ["CCO", "CC(=O)Cl"],
                    "location_ref": "Example 1",
                    "conditions": {"temperature_c": 20},
                }
            ],
        },
        logical_name="exact.json",
        producer="tests.fixture",
    ).to_dict()
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
                    "catalog_number": "ETHANOL",
                    "smiles": "CCO",
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                },
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": "ACETYL-CHLORIDE",
                    "smiles": "CC(=O)Cl",
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                },
            ],
        },
        logical_name="inventory.json",
        producer="tests.inventory",
    ).to_dict()
    runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={
            extraction_ref["sha256"]: "structured_exact_row_extraction",
            inventory_ref["sha256"]: "inventory_snapshot_set",
        },
    )
    materialized = runtime.execute(
        materialization_commands_for_global_plan(
            _plan(),
            run_id=kernel.spec.run_id,
            input_revision=kernel.state.graph_revision,
            dependency_revisions={
                "graph_revision": kernel.state.graph_revision,
                "evidence_revision": kernel.state.evidence_revision,
            },
        )[0]
    )
    validated = runtime.execute(
        _command(
            kernel,
            "validate_reaction",
            {
                "candidate": materialized.payload,
                "mapped_reaction_smiles": (
                    "[CH3:1][C:2](=[O:3])[Cl:4]."
                    "[CH3:5][CH2:6][OH:7]>>"
                    "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
                ),
            },
            task_kind="validation",
            suffix="ester",
        )
    )
    discovery_batch = runtime.execute_pipeline(
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
            suffix="patent",
            artifact_refs=(extraction_ref,),
        )
    )
    stock = runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": "CCOC(C)=O",
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:ethanol", "smiles": "CCO"},
                    {"leaf_id": "leaf:acetyl-chloride", "smiles": "CC(=O)Cl"},
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T12:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="leaves",
            artifact_refs=(inventory_ref,),
        )
    )

    assert validated.status == "completed"
    all_results = (
        stock,
        validated,
        *reversed(discovery_batch.results),
        materialized,
    )
    ingested = store.apply(
        CanonicalIngestionBatch(worker_results=all_results),
        worker_runtime=runtime,
        idempotency_key="all-worker-facts",
    )
    graph = ingested["graph"]
    edge = next(iter(graph["edges"].values()))
    route = next(iter(graph["route_families"].values()))

    assert len(edge["reaction_proofs"]) == 1
    assert len(edge["exact_record_ids"]) == 1
    exact = graph["exact_records"][edge["exact_record_ids"][0]]
    assert exact["procedure_authority_scope"] == "source_exact_reaction_procedure"
    assert exact["condition_completeness"]["complete"] is False
    assert set(exact["condition_completeness"]["missing_required_groups"]) == {
        "agents",
        "solvent",
        "time",
    }
    assert len(edge["independent_source_groups"]) == 1
    assert {row["origin_kind"] for row in edge["origin_records"]} == {
        "codex_global_director",
        "literature",
    }
    assert all(
        graph["molecules"][molecule_id]["stock_closed"] is True
        for molecule_id in route["leaf_molecule_ids"]
    )
    assert route["minimum_proof_level"] == 2
    assert route["stock_closure_rate"] == 1.0
    assert route["closed"] is False
    assert graph["deficit_frontier"]["summary"]["by_kind"]["validation"] == 0
    assert graph["deficit_frontier"]["summary"]["by_kind"]["stock"] == 0
    assert graph["deficit_frontier"]["summary"]["by_kind"]["evidence"] == 1
    oracle = store.full_recompute_oracle()
    assert graph["scientific_sha256"] == oracle["scientific_sha256"]


def test_deficit_frontier_ties_and_incremental_replacement_are_deterministic(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    result = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="frontier",
    )
    graph = result["graph"]
    full = compile_deficit_frontier(
        graph,
        acceptance_spec=kernel.spec.acceptance,
    )
    hypothesis_id = next(iter(graph["hypotheses"]))
    incremental = compile_deficit_frontier(
        graph,
        acceptance_spec=kernel.spec.acceptance,
        previous_frontier=full,
        dirty_entity_ids={hypothesis_id},
    )

    assert frontier_scientific_projection(full) == frontier_scientific_projection(
        incremental
    )
    assert [row["deficit_id"] for row in full["items"]] == [
        row["deficit_id"] for row in compile_deficit_frontier(
            graph,
            acceptance_spec=kernel.spec.acceptance,
        )["items"]
    ]


def test_different_procurement_boundaries_are_not_marked_dominated(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    store.apply(
        CanonicalIngestionBatch(
            route_families=(
                {
                    "route_family_id": "route:short",
                    "strategic_disconnection": "buy ethanol",
                },
                {
                    "route_family_id": "route:long",
                    "strategic_disconnection": "make ethanol",
                },
            )
        ),
        idempotency_key="routes",
    )
    result = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:short",
            },
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
        ),
        key="route-edges",
    )
    routes = result["graph"]["route_families"]
    short = next(route for route in routes.values() if "route:short" in route["aliases"])
    long = next(route for route in routes.values() if "route:long" in route["aliases"])

    assert set(short["edge_ids"]) < set(long["edge_ids"])
    assert set(short["leaf_molecule_ids"]) != set(long["leaf_molecule_ids"])
    assert long["status"] != "dominated"
    assert "dominated_by_route_family_id" not in long
    assert {short["route_family_id"], long["route_family_id"]} <= {
        row["route_family_id"] for row in result["graph"]["portfolio_ranking"]
    }


def test_local_stock_update_recomputes_only_dirty_subgraph_and_matches_oracle(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    proposal_runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    proposals = tuple(
        {
            "product_smiles": "C" * carbon_count + "O",
            "precursor_smiles": ["C" * (carbon_count - 1) + "OC"],
            "origin_kind": "template",
            "proposal_id": f"isomer:{carbon_count}",
        }
        for carbon_count in range(2, 17)
    )
    first = _apply_proposals(
        kernel,
        store,
        proposal_runtime,
        proposals,
        key="large-fixture",
    )
    selected_smiles = "C" * 9 + "OC"
    inventory_ref = kernel.artifacts.put_json(
        {
            "schema_version": "versioned_inventory_snapshot.v1",
            "adapter_version": "tests.inventory.v1",
            "inventory_version": "local-update",
            "retrieved_at": "2026-07-13T00:00:00Z",
            "offers": [
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": "LOCAL-LEAF",
                    "smiles": selected_smiles,
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                }
            ],
        },
        logical_name="local-inventory.json",
        producer="tests.inventory",
    ).to_dict()
    stock_runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={inventory_ref["sha256"]: "inventory_snapshot_set"},
    )
    stock_result = stock_runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": kernel.spec.target_smiles,
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:local", "smiles": selected_smiles}
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T12:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="local-update",
            artifact_refs=(inventory_ref,),
        )
    )
    updated = store.apply(
        CanonicalIngestionBatch(worker_results=(stock_result,)),
        worker_runtime=stock_runtime,
        idempotency_key="local-stock",
    )
    graph = updated["graph"]
    oracle = store.full_recompute_oracle()

    assert len(first["graph"]["edges"]) == 15
    assert graph["delta"]["recomputed_fraction"] < 0.2
    assert graph["scientific_sha256"] == oracle["scientific_sha256"]
    assert canonical_scientific_projection(graph) == canonical_scientific_projection(
        oracle
    )
