from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
    detect_source_conflicts,
    materialization_commands_for_global_plan,
    normalize_source_binding,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.worker_runtime import (
    WorkerBudget,
    WorkerCommand,
    WorkerHandlerSpec,
    WorkerRuntime,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.orchestration.global_campaign_director import (
    director_trigger_reasons,
)


def _kernel(tmp_path: Path) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="v4-workers",
            target_name="fixture",
            target_smiles="CCOC(C)=O",
            created_at="2026-07-13T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=16,
                    max_attempt_runs=64,
                ),
                max_total_tasks=64,
                max_evidence_tasks=32,
                max_stock_tasks=16,
                max_validation_tasks=16,
            ),
        ),
    )
    kernel.start()
    return kernel


def _command(
    kernel: RunKernel,
    worker_type: str,
    payload: dict,
    *,
    task_kind: str,
    suffix: str,
    timeout_s: float = 60.0,
    dependency_revisions: dict | None = None,
    artifact_refs: tuple[dict, ...] = (),
) -> WorkerCommand:
    return WorkerCommand(
        command_id=f"{worker_type}:{suffix}",
        run_id=kernel.spec.run_id,
        worker_type=worker_type,
        input_revision=kernel.state.graph_revision,
        idempotency_key=f"{worker_type}:{suffix}",
        payload=payload,
        budget=WorkerBudget(task_kind=task_kind, timeout_s=timeout_s),
        dependency_revisions=dependency_revisions or {
            "graph_revision": kernel.state.graph_revision,
            "evidence_revision": kernel.state.evidence_revision,
        },
        artifact_refs=artifact_refs,
    )


def _extraction_artifact(
    kernel: RunKernel,
    binding: dict,
    rows: list[dict],
    *,
    producer_kind: str = "deterministic_structure_parser",
) -> dict:
    return kernel.artifacts.put_json(
        {
            "schema_version": "structured_exact_row_extraction.v1",
            "source_binding_id": binding["binding_id"],
            "extractor": {
                "producer_kind": producer_kind,
                "producer_id": "tests.fixture_extractor",
                "version": "1.0.0",
            },
            "rows": rows,
        },
        logical_name="structured_exact_row_extraction.json",
        producer="tests.fixture_extractor",
    ).to_dict()


def _runtime_with_extractions(
    kernel: RunKernel,
    *refs: dict,
) -> WorkerRuntime:
    return WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={
            str(ref["sha256"]): "structured_exact_row_extraction" for ref in refs
        },
    )


def _runtime_with_authorities(
    kernel: RunKernel,
    authorities: dict[str, str],
) -> WorkerRuntime:
    return WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities=authorities,
    )


def test_materialization_counts_one_expansion_and_cache_hit_counts_no_attempt(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    payload = {
        "product_smiles": "CCOC(C)=O",
        "precursor_smiles": ["CCO", "CC(=O)Cl"],
        "reagent_smiles": ["N"],
    }
    first = runtime.execute(
        _command(
            kernel,
            "materialize_candidate",
            payload,
            task_kind="proposal",
            suffix="first",
        )
    )
    second = runtime.execute(
        _command(
            kernel,
            "materialize_candidate",
            payload,
            task_kind="proposal",
            suffix="same-input-new-command",
        )
    )

    assert first.status == "completed"
    assert first.payload["proof_state"]["states"] == [
        "L0_hypothesis",
        "L1_structural_materialized",
    ]
    assert second.cache_hit is True
    assert kernel.state.attempt_count == 1
    assert kernel.state.accepted_expansion_count == 1


def test_global_multistep_skeleton_compiles_to_unique_edge_workers(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    plan = {
        "multi_step_skeletons": [
            {
                "skeleton_id": "route-a",
                "route_family_id": "family-a",
                "steps": [
                    {
                        "step_id": "a1",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    },
                    {
                        "step_id": "a2",
                        "product_smiles": "CC=O",
                        "precursor_smiles": ["CCO"],
                        "transformation_hypothesis": "oxidation",
                    },
                ],
            },
            {
                "skeleton_id": "route-b",
                "route_family_id": "family-b",
                "steps": [
                    {
                        "step_id": "b-shared",
                        "product_smiles": "CC=O",
                        "precursor_smiles": ["CCO"],
                        "transformation_hypothesis": "shared oxidation",
                    }
                ],
            },
        ]
    }
    commands = materialization_commands_for_global_plan(
        plan,
        run_id=kernel.spec.run_id,
        input_revision=kernel.state.graph_revision,
        dependency_revisions={
            "graph_revision": kernel.state.graph_revision,
            "evidence_revision": kernel.state.evidence_revision,
        },
    )
    results = [runtime.execute(command) for command in commands]

    assert len(commands) == 2
    assert all(result.status == "completed" for result in results)
    shared = next(
        result for result in results if result.payload["product_smiles"] == "CC=O"
    )
    assert {row["step_id"] for row in shared.payload["proposal_refs"]} == {
        "a2",
        "b-shared",
    }
    assert kernel.state.attempt_count == 2
    assert kernel.state.accepted_expansion_count == 2


def test_cheap_gates_reject_cycle_duplicate_and_impossible_precursor_without_expansion(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    duplicate_digest = audit_retrosynthetic_candidate(
        "CCOC(C)=O", ["CCO", "CC(=O)Cl"]
    )["edge_digest"]
    commands = (
        _command(
            kernel,
            "materialize_candidate",
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["CCO"],
                "ancestor_smiles": ["CCO"],
            },
            task_kind="proposal",
            suffix="cycle",
        ),
        _command(
            kernel,
            "materialize_candidate",
            {
                "product_smiles": "CCCCCCCCCCCCCCCCCCCC",
                "precursor_smiles": ["C"],
            },
            task_kind="proposal",
            suffix="atom-jump",
        ),
        _command(
            kernel,
            "materialize_candidate",
            {
                "product_smiles": "not-smiles",
                "precursor_smiles": ["CCO"],
            },
            task_kind="proposal",
            suffix="parse",
        ),
        _command(
            kernel,
            "materialize_candidate",
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "existing_edge_digests": [duplicate_digest],
            },
            task_kind="proposal",
            suffix="duplicate",
        ),
    )
    results = [runtime.execute(command) for command in commands]

    assert all(result.status == "rejected" for result in results)
    assert "target_or_current_node_self_loop" in results[0].failure_reasons
    assert "large_atom_jump" in results[1].failure_reasons
    assert "invalid_or_missing_material" in results[2].failure_reasons
    assert "duplicate_reaction_edge" in results[3].failure_reasons
    assert kernel.state.attempt_count == 4
    assert kernel.state.accepted_expansion_count == 0


def test_materialized_edge_can_be_independently_reaction_validated(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    materialized = runtime.execute(
        _command(
            kernel,
            "materialize_candidate",
            {"product_smiles": "CC=O", "precursor_smiles": ["CCO"]},
            task_kind="proposal",
            suffix="oxidation",
        )
    )
    validated = runtime.execute(
        _command(
            kernel,
            "validate_reaction",
            {
                "candidate": materialized.payload,
                "mapped_reaction_smiles": (
                    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
                ),
            },
            task_kind="validation",
            suffix="oxidation",
        )
    )

    assert validated.status == "completed"
    assert validated.payload["reaction_proof"]["proof_level"] == (
        "L2_reaction_validated"
    )
    assert validated.payload["proof_state"]["reaction_validated"] is True
    assert kernel.state.accepted_expansion_count == 1


def test_discovery_automatically_extracts_exact_rows_and_signals_resume(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    patent_source = {
        "source_kind": "patent",
        "source_ref": "patent:US2020123456A1",
        "title": "Fixture process",
    }
    patent_binding = normalize_source_binding(patent_source)
    extraction_ref = _extraction_artifact(
        kernel,
        patent_binding,
        [
            {
                "step_id": "patent-example-1",
                "claim_scope_id": "acetaldehyde-oxidation",
                "product_smiles": "CC=O",
                "reactant_smiles": ["CCO"],
                "location_ref": "Example 1, paragraph 42",
                "conditions": {"temperature_c": 20},
            }
        ],
    )
    runtime = _runtime_with_extractions(kernel, extraction_ref)
    discovery = _command(
        kernel,
        "discover_sources",
        {
            "sources": [
                {
                    **patent_source,
                    "extraction_artifact_sha256": extraction_ref["sha256"],
                },
                {
                    "source_kind": "codex_claim",
                    "title": "Uncited model memory",
                },
            ]
        },
        task_kind="evidence",
        suffix="patent",
        artifact_refs=(extraction_ref,),
    )
    batch = runtime.execute_pipeline(discovery)

    assert len(batch.results) == 2
    assert batch.results[0].payload["extraction_task_count"] == 1
    assert batch.results[1].payload["exact_records"][0]["relation_type"] == "exact"
    assert batch.results[1].payload["exact_records"][0]["proof_state"][
        "reaction_validated"
    ] is False
    assert "exact_rows_added" in batch.material_events
    assert batch.resume_campaign is True
    assert "exact_rows_added" in director_trigger_reasons(
        SimpleNamespace(delta=SimpleNamespace(material_events=batch.material_events)),
        mode="event_replan",
    )
    assert kernel.state.model_totals["model_invocations"] == 0
    attempts = kernel.state.attempt_count
    replayed = runtime.replay_result(batch.results[1].to_dict())
    assert replayed.payload == batch.results[1].payload
    assert kernel.state.attempt_count == attempts


def test_partial_extraction_and_tampered_source_binding_fail_closed(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    binding = normalize_source_binding(
        {
            "source_kind": "paper_si",
            "doi": "10.1000/fixture-si",
            "content_scope": "supporting_information",
        }
    )
    partial_ref = _extraction_artifact(
        kernel,
        binding,
        [{"product_smiles": "CC=O", "reactant_smiles": ["CCO"]}],
    )
    model_ref = _extraction_artifact(
        kernel,
        binding,
        [],
        producer_kind="codex_source_text_translation",
    )
    runtime = _runtime_with_extractions(kernel, partial_ref, model_ref)
    partial = runtime.execute(
        _command(
            kernel,
            "extract_exact_source",
            {
                "source_binding": binding,
                "extraction_artifact_sha256": partial_ref["sha256"],
            },
            task_kind="evidence",
            suffix="partial",
            artifact_refs=(partial_ref,),
        )
    )
    tampered_binding = {**binding, "title": "changed after digest"}
    tampered = runtime.execute(
        _command(
            kernel,
            "extract_exact_source",
            {
                "source_binding": tampered_binding,
                "extraction_artifact_sha256": partial_ref["sha256"],
            },
            task_kind="evidence",
            suffix="tampered",
            artifact_refs=(partial_ref,),
        )
    )
    model_claim = runtime.execute(
        _command(
            kernel,
            "extract_exact_source",
            {
                "source_binding": binding,
                "extraction_artifact_sha256": model_ref["sha256"],
            },
            task_kind="evidence",
            suffix="model-produced",
            artifact_refs=(model_ref,),
        )
    )

    assert partial.status == "partial"
    assert partial.payload["exact_records"] == []
    assert partial.payload["rejected_rows"][0]["reasons"] == [
        "exact_source_location_missing"
    ]
    assert tampered.status == "rejected"
    assert tampered.failure_reasons == (
        "source_binding_not_replayable_or_extractable",
    )
    assert model_claim.status == "rejected"
    assert model_claim.failure_reasons == (
        "structured_extraction_producer_untrusted",
    )


def test_independent_exact_sources_and_conflicts_are_explicit() -> None:
    first_binding = normalize_source_binding(
        {"source_kind": "patent", "source_ref": "patent:US2020123456A1"}
    )
    second_binding = normalize_source_binding(
        {"source_kind": "paper_si", "doi": "10.1000/fixture-paper"}
    )
    records = [
        {
            "record_id": "exact:patent",
            "claim_scope_id": "claim:one",
            "edge_digest": "edge-a",
            "independence_group": first_binding["independence_group"],
            "conditions": {"temperature_c": 20},
        },
        {
            "record_id": "exact:paper",
            "claim_scope_id": "claim:one",
            "edge_digest": "edge-a",
            "independence_group": second_binding["independence_group"],
            "conditions": {"temperature_c": 80},
        },
        {
            "record_id": "exact:alternative",
            "claim_scope_id": "claim:one",
            "edge_digest": "edge-b",
            "independence_group": second_binding["independence_group"],
            "conditions": {},
        },
    ]

    conflicts = detect_source_conflicts(records)

    assert {row["conflict_kind"] for row in conflicts} == {
        "incompatible_condition:temperature_c",
        "incompatible_exact_structures",
    }
    assert all(row["status"] == "unresolved" for row in conflicts)
    assert all(row["semantics"]["no_automatic_winner"] is True for row in conflicts)


def test_second_exact_source_promotes_independence_without_implying_validation(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    patent = normalize_source_binding(
        {"source_kind": "patent", "source_ref": "patent:US2020123456A1"}
    )
    paper = normalize_source_binding(
        {"source_kind": "paper_si", "doi": "10.1000/fixture-paper"}
    )
    common_row = {
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "location_ref": "explicit procedure",
    }
    patent_ref = _extraction_artifact(kernel, patent, [common_row])
    paper_ref = _extraction_artifact(
        kernel,
        paper,
        [{**common_row, "location_ref": "SI table S1"}],
    )
    runtime = _runtime_with_extractions(kernel, patent_ref, paper_ref)
    first = runtime.execute(
        _command(
            kernel,
            "extract_exact_source",
            {
                "source_binding": patent,
                "extraction_artifact_sha256": patent_ref["sha256"],
            },
            task_kind="evidence",
            suffix="independence-patent",
            artifact_refs=(patent_ref,),
        )
    )
    second = runtime.execute(
        _command(
            kernel,
            "extract_exact_source",
            {
                "source_binding": paper,
                "extraction_artifact_sha256": paper_ref["sha256"],
                "existing_exact_records": first.payload["exact_records"],
            },
            task_kind="evidence",
            suffix="independence-paper",
            artifact_refs=(paper_ref,),
        )
    )
    state = second.payload["exact_records"][0]["proof_state"]

    assert state["exact_source_bound"] is True
    assert state["independently_supported"] is True
    assert len(state["independent_source_groups"]) == 2
    assert state["reaction_validated"] is False
    assert second.payload["conflicts"] == []


def test_all_supported_source_channels_normalize_with_explicit_authority() -> None:
    sources = (
        {"source_kind": "patent", "source_ref": "patent:US2020123456A1"},
        {"source_kind": "paper_si", "doi": "10.1000/fixture"},
        {
            "source_kind": "curated_registry",
            "registry_id": "trusted-row-1",
            "artifact_sha256": "a" * 64,
        },
        {
            "source_kind": "image_extraction",
            "local_pdf": "fixtures/source.pdf",
            "provenance": "ocr_structured_table_extraction",
        },
        {"source_kind": "codex_claim", "title": "model memory only"},
    )
    bindings = [normalize_source_binding(source) for source in sources]

    assert {binding["source_kind"] for binding in bindings} == {
        "patent",
        "paper_si",
        "curated_registry",
        "image_extraction",
        "codex_claim",
    }
    assert all(binding["content_sha256"] for binding in bindings)
    assert bindings[-1]["authority_scope"] == "model_advisory_claim"
    assert bindings[-1]["usable_for_extraction"] is False


def test_stock_worker_audits_every_leaf_and_rejects_stale_authority(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    inventory = {
        "schema_version": "versioned_inventory_snapshot.v1",
        "adapter_version": "fixture.inventory_adapter.v1",
        "inventory_version": "catalog-2026-07-01",
        "retrieved_at": "2026-07-01T00:00:00Z",
        "offers": [
            {
                "schema_version": "stock_offer_snapshot.v1",
                "supplier": "fixture",
                "catalog_number": "ETHANOL-1",
                "smiles": "CCO",
                "checked_at": "2026-07-01T00:00:00Z",
                "available": True,
            }
        ],
    }
    inventory_ref = kernel.artifacts.put_json(
        inventory,
        logical_name="versioned_inventory_snapshot.json",
        producer="tests.fixture_inventory_adapter",
    ).to_dict()
    runtime = _runtime_with_authorities(
        kernel,
        {inventory_ref["sha256"]: "inventory_snapshot_set"},
    )
    result = runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": "CCOC(C)=O",
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:ethanol", "smiles": "CCO", "common": True},
                    {"leaf_id": "leaf:missing", "smiles": "CN", "common": True},
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T00:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="fresh",
            artifact_refs=(inventory_ref,),
        )
    )
    stale = runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": "CCOC(C)=O",
                "selected_deep_leaves": [{"leaf_id": "leaf:ethanol", "smiles": "CCO"}],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2027-07-13T00:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="stale",
            artifact_refs=(inventory_ref,),
        )
    )
    no_authority_runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
    )
    untrusted_inventory = no_authority_runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": "CCOC(C)=O",
                "selected_deep_leaves": [{"leaf_id": "leaf:ethanol", "smiles": "CCO"}],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T00:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="untrusted-inventory",
            artifact_refs=(inventory_ref,),
        )
    )

    assert result.status == "partial"
    assert result.payload["audited_leaf_count"] == 2
    assert result.payload["stock_closed_leaf_count"] == 1
    assert result.payload["leaf_audits"][1]["accepted"] is False
    assert stale.status == "partial"
    assert stale.payload["leaf_audits"][0]["accepted"] is False
    assert "inventory_snapshot_stale" in stale.failure_reasons
    assert "inventory_authority_stale" in stale.payload["leaf_audits"][0]["reasons"]
    assert untrusted_inventory.status == "rejected"
    assert untrusted_inventory.failure_reasons == (
        "worker_artifact_authority_scope_missing",
    )
    attempts = kernel.state.attempt_count
    replayed_stock = runtime.replay_result(result.to_dict())
    assert replayed_stock.payload["leaf_audits"] == result.payload["leaf_audits"]
    assert kernel.state.attempt_count == attempts


def test_runtime_timeout_and_stale_revision_are_deterministic(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)

    def slow_fixture(_: WorkerCommand, __) -> dict:
        return {"status": "completed", "payload": {"ignored": True}, "elapsed_s": 2.0}

    runtime = WorkerRuntime(
        kernel,
        {
            "slow_fixture": WorkerHandlerSpec(
                "slow_fixture",
                "fixture.v1",
                "other",
                slow_fixture,
            )
        },
    )
    timed_out = runtime.execute(
        _command(
            kernel,
            "slow_fixture",
            {},
            task_kind="other",
            suffix="timeout",
            timeout_s=1.0,
        )
    )
    stale = runtime.execute(
        WorkerCommand(
            command_id="slow_fixture:stale",
            run_id=kernel.spec.run_id,
            worker_type="slow_fixture",
            input_revision=1,
            idempotency_key="slow_fixture:stale",
            payload={},
            budget=WorkerBudget(task_kind="other"),
            dependency_revisions={"graph_revision": 1},
        )
    )

    assert timed_out.status == "timed_out"
    assert timed_out.failure_reasons == ("worker_timeout_exceeded",)
    assert stale.status == "stale"
    assert "worker_input_graph_revision_stale" in stale.failure_reasons
    assert kernel.state.attempt_count == 1
