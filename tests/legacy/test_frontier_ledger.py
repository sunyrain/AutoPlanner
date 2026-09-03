from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from cascade_planner.legacy.application_runtime.frontier_ledger import (
    exact_edge_signature,
    project_frontier_ledger,
    validate_frontier_ledger,
)
from cascade_planner.legacy.application_runtime.frontier_scheduler import (
    FrontierScheduler,
    PersistentFrontierQueue,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_step
from cascade_planner.providers.stock import (
    BenchmarkCatalogStockProvider,
    SnapshotStockProvider,
    stock_snapshot_sha256,
)


TARGET = "CCC(C)=O"
PRECURSORS = ("CC[C@H](C)O", "CC[C@@H](C)O")
MAPPED_REACTIONS = {
    "CC[C@H](C)O": (
        "[CH3:1][CH2:2][C@H:3]([CH3:4])[OH:5]"
        ">>[CH3:1][CH2:2][C:3]([CH3:4])=[O:5]"
    ),
    "CC[C@@H](C)O": (
        "[CH3:1][CH2:2][C@@H:3]([CH3:4])[OH:5]"
        ">>[CH3:1][CH2:2][C:3]([CH3:4])=[O:5]"
    ),
}


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


def _graph() -> dict:
    steps = []
    for index, precursor in enumerate(PRECURSORS, start=1):
        signature = f"{TARGET}<-{precursor}"
        steps.append(
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": f"step:{index}",
                "signature": signature,
                "product_smiles": TARGET,
                "precursor_smiles": [precursor],
                "product_node_id": "mol:target",
                "precursor_node_ids": [f"mol:precursor:{index}"],
                "proposal_ids": [f"proposal:{index}"],
                "source_refs": [f"source:{index}"],
                "evidence_refs": [],
            }
        )
    return {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "frontier-ledger-alternatives",
        "target_smiles": TARGET,
        "nodes": [],
        "steps": steps,
        # Intentionally false and incomplete: the ledger must never consume
        # this bounded presentation layer when deriving closure.
        "route_hypotheses": [{"solved": False, "retrosynthetic_step_ids": []}],
    }


def _stock_snapshots(smiles_values: tuple[str, ...] = PRECURSORS) -> list[dict]:
    return [
        {
            "schema_version": "stock_offer_snapshot.v1",
            "supplier": "fixture",
            "catalog_number": f"SKU-{index}",
            "smiles": smiles,
            "checked_at": "2026-07-12T00:00:00Z",
            "available": True,
        }
        for index, smiles in enumerate(smiles_values, start=1)
    ]


def _trusted_stock_providers(
    smiles_values: tuple[str, ...] = PRECURSORS,
) -> dict[str, SnapshotStockProvider]:
    provider = SnapshotStockProvider(
        trusted_snapshots=_stock_snapshots(smiles_values)
    )
    return {provider.descriptor.provider_id: provider}


def _stock_queue(tmp_path, smiles_values: tuple[str, ...] = PRECURSORS) -> dict:
    snapshots = _stock_snapshots(smiles_values)
    queue = PersistentFrontierQueue(tmp_path / "queue")
    scheduler = FrontierScheduler(
        queue,
        SnapshotStockProvider(trusted_snapshots=snapshots),
    )
    for index, snapshot in enumerate(snapshots, start=1):
        scheduler.submit(
            run_id="frontier-ledger-alternatives",
            case_id="frontier-ledger-alternatives",
            frontier_smiles=str(snapshot["smiles"]),
            frontier_node_id=f"mol:precursor:{index}",
            idempotency_key=f"stock:{index}",
            stock_request={
                "offers": [
                    {
                        **snapshot,
                        "snapshot_sha256": stock_snapshot_sha256(snapshot),
                    }
                ]
            },
            metadata={
                "campaign_root_smiles": TARGET,
                "campaign_identity_sha256": "a" * 64,
                "campaign_policy_sha256": "c" * 64,
            },
            now="2026-07-12T00:00:00.000000Z",
        )
    return queue.snapshot("frontier-ledger-alternatives")


def _proof_state(graph: dict, *, validated_step_ids: set[str]) -> dict:
    records = []
    for step in graph["steps"]:
        precursor = step["precursor_smiles"][0]
        materialized = {
            "schema_version": "materialized_reaction_candidate.v1",
            "step_id": step["step_id"],
            "product_smiles": step["product_smiles"],
            "reactant_smiles": step["precursor_smiles"],
            "atom_mapped_reaction_smiles": MAPPED_REACTIONS[precursor],
            "mapping_source": "frontier_ledger_test",
        }
        if step["step_id"] in validated_step_ids:
            proof = verify_reaction_step(
                materialized,
                graph_and_stock_closed=False,
            )
            assert proof["accepted"] is True
            record = {
                "schema_version": "codex_retrosynthesis_reaction_proof_record.v2",
                "proof_request_id": f"proof:{step['step_id']}",
                "step_id": step["step_id"],
                "signature": step["signature"],
                "product_smiles": step["product_smiles"],
                "precursor_smiles": step["precursor_smiles"],
                "required_proof_level": 2,
                "status": "validated",
                "achieved_proof_level": 2,
                "materialized_candidate": materialized,
                "materialized_candidate_sha256": _digest(materialized),
                "proof": proof,
                "proof_authority": "current_host_verifier_replay",
            }
        else:
            record = {
                "schema_version": "codex_retrosynthesis_reaction_proof_record.v2",
                "proof_request_id": f"proof:{step['step_id']}",
                "step_id": step["step_id"],
                "signature": step["signature"],
                "product_smiles": step["product_smiles"],
                "precursor_smiles": step["precursor_smiles"],
                "required_proof_level": 2,
                "status": "pending",
                "achieved_proof_level": 0,
                "materialized_candidate": {},
                "materialized_candidate_sha256": "",
                "proof": {},
                "proof_authority": "none",
            }
        records.append(record)
    identity_payload = {
        "schema_version": graph["schema_version"],
        "case_id": graph["case_id"],
        "target_smiles": TARGET,
        "steps": sorted(
            [
                {
                    "step_id": step["step_id"],
                    "signature": step["signature"],
                    "product_smiles": TARGET,
                    "precursor_smiles": sorted(step["precursor_smiles"]),
                }
                for step in graph["steps"]
            ],
            key=lambda row: (row["step_id"], row["signature"]),
        ),
    }
    state = {
        "schema_version": "codex_retrosynthesis_reaction_proof_state.v1",
        "graph_identity_sha256": _digest(identity_payload),
        "records": records,
    }
    state["content_sha256"] = _digest(state)
    return state


def test_one_closed_alternative_means_any_true_but_all_false(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert ledger["summary"]["any_route_closed"] is True
    assert ledger["summary"]["all_explored_graph_closed"] is False
    assert ledger["summary"]["any_benchmark_route_closed"] is True
    assert ledger["summary"]["all_explored_benchmark_closed"] is False
    assert ledger["summary"]["any_procurement_route_closed"] is True
    assert ledger["summary"]["all_explored_procurement_closed"] is False
    assert ledger["molecules"][TARGET]["proposal"]["alternative_count"] == 2
    assert ledger["semantics"]["route_hypotheses_are_not_consumed"] is True
    assert validate_frontier_ledger(
        ledger,
        trusted_stock_provider_instances=_trusted_stock_providers(),
    ) == []


def test_all_alternatives_closed_means_both_closure_claims_are_true(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert ledger["summary"]["any_route_closed"] is True
    assert ledger["summary"]["all_explored_graph_closed"] is True
    assert ledger["summary"]["any_procurement_route_closed"] is True
    assert ledger["summary"]["all_explored_procurement_closed"] is True
    assert ledger["summary"]["reaction_proven_edge_count"] == 2
    assert all(row["reaction_proof"]["closed"] for row in ledger["edges"].values())
    assert ledger == json.loads(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def test_serialized_stock_envelopes_cannot_self_authorize_closure(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances={},
    )

    assert ledger["summary"]["stock_closed_molecule_count"] == 0
    assert ledger["summary"]["any_route_closed"] is False
    assert all(
        row["stock"]["host_replay_verified"] is False
        for smiles, row in ledger["molecules"].items()
        if smiles != TARGET
    )
    stock_validation = ledger["input_validation"]["stock_authority"]
    assert stock_validation["host_replayed_claim_count"] == 0
    assert stock_validation["rejected_claim_count"] == 2
    assert (
        "stock_observation_provider_set_not_current_host_policy"
        in stock_validation["rejection_reasons"]
    )


def test_validator_replays_positive_stock_bindings_with_current_trust_set(
    tmp_path,
) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )
    untrusted_runtime = SnapshotStockProvider(trusted_snapshots=[])

    reasons = validate_frontier_ledger(
        ledger,
        trusted_stock_provider_instances={
            untrusted_runtime.descriptor.provider_id: untrusted_runtime
        },
    )

    assert any(reason.endswith(":current_host_replay_failed") for reason in reasons)


def test_benchmark_membership_closes_search_without_procurement_claim(
    tmp_path,
) -> None:
    graph = _graph()
    catalog = tmp_path / "benchmark.smi"
    catalog.write_text("\n".join(PRECURSORS) + "\n", encoding="utf-8")
    provider = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog_name="ledger-fixture",
    )
    queue = PersistentFrontierQueue(tmp_path / "benchmark-queue")
    scheduler = FrontierScheduler(queue, provider)
    for index, smiles in enumerate(PRECURSORS, start=1):
        job = scheduler.submit(
            run_id="frontier-ledger-alternatives",
            case_id="frontier-ledger-alternatives",
            frontier_smiles=smiles,
            frontier_node_id=f"mol:precursor:{index}",
            idempotency_key=f"benchmark:{index}",
            metadata={
                "campaign_root_smiles": TARGET,
                "campaign_identity_sha256": "b" * 64,
                "campaign_policy_sha256": "c" * 64,
            },
            now="2026-07-12T00:00:00.000000Z",
        )
        assert job.achieved_proof_level == 0
    ledger = project_frontier_ledger(
        graph,
        queue.snapshot("frontier-ledger-alternatives"),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances={provider.descriptor.provider_id: provider},
    )

    precursor_stock = [
        ledger["molecules"][smiles]["stock"] for smiles in PRECURSORS
    ]
    assert all(row["benchmark_membership_closed"] for row in precursor_stock)
    assert all(row["benchmark_only"] for row in precursor_stock)
    assert not any(row["procurement_boundary_closed"] for row in precursor_stock)
    assert ledger["summary"]["any_benchmark_route_closed"] is True
    assert ledger["summary"]["all_explored_benchmark_closed"] is True
    assert ledger["summary"]["any_procurement_route_closed"] is False
    assert ledger["summary"]["all_explored_procurement_closed"] is False
    assert ledger["summary"]["all_explored_graph_closed"] is True


def test_multi_provider_refresh_revokes_procurement_but_keeps_benchmark_history(
    tmp_path,
) -> None:
    graph = _graph()
    catalog = tmp_path / "benchmark-refresh.smi"
    catalog.write_text("\n".join(PRECURSORS) + "\n", encoding="utf-8")
    benchmark = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog_name="refresh-fixture",
    )
    available = _stock_snapshots()
    unavailable = [
        {
            **row,
            "checked_at": "2026-07-13T00:00:00Z",
            "available": False,
        }
        for row in available
    ]
    snapshots = SnapshotStockProvider(
        trusted_snapshots=[*available, *unavailable]
    )
    providers = {
        provider.descriptor.provider_id: provider
        for provider in (benchmark, snapshots)
    }
    queue = PersistentFrontierQueue(tmp_path / "provider-set-refresh")
    scheduler = FrontierScheduler(queue, providers)
    for index, row in enumerate(available, start=1):
        scheduler.submit(
            run_id="frontier-ledger-alternatives",
            case_id="frontier-ledger-alternatives",
            frontier_smiles=str(row["smiles"]),
            frontier_node_id=f"mol:precursor:{index}",
            idempotency_key=f"provider-set:{index}",
            stock_request={
                "offers": [
                    {**row, "snapshot_sha256": stock_snapshot_sha256(row)}
                ]
            },
            metadata={
                "campaign_root_smiles": TARGET,
                "campaign_identity_sha256": "d" * 64,
                "campaign_policy_sha256": "e" * 64,
            },
            now="2026-07-12T00:00:00.000000Z",
        )
    first = project_frontier_ledger(
        graph,
        queue.snapshot("frontier-ledger-alternatives"),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=providers,
    )
    assert first["summary"]["all_explored_benchmark_closed"] is True
    assert first["summary"]["all_explored_procurement_closed"] is True
    assert all(
        set(first["molecules"][smiles]["stock"]["boundary_types"])
        == {"benchmark_stock", "commercially_orderable"}
        for smiles in PRECURSORS
    )

    jobs_by_smiles = {
        job.frontier_smiles: job
        for job in queue.list_jobs("frontier-ledger-alternatives")
    }
    for row in unavailable:
        current = jobs_by_smiles[str(row["smiles"])]
        scheduler.refresh(
            current,
            case_id="frontier-ledger-alternatives",
            stock_request={
                "offers": [
                    {**row, "snapshot_sha256": stock_snapshot_sha256(row)}
                ]
            },
            now="2026-07-13T00:00:00.000000Z",
        )
    second = project_frontier_ledger(
        graph,
        queue.snapshot("frontier-ledger-alternatives"),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=providers,
    )

    assert second["summary"]["all_explored_benchmark_closed"] is True
    assert second["summary"]["any_procurement_route_closed"] is False
    assert all(
        second["molecules"][smiles]["stock"]["boundary_types"]
        == ["benchmark_stock"]
        for smiles in PRECURSORS
    )
    assert all(
        second["molecules"][smiles]["stock"]["history_observation_count"]
        == 4
        for smiles in PRECURSORS
    )
    assert validate_frontier_ledger(
        second,
        trusted_stock_provider_instances=providers,
    ) == []


@pytest.mark.parametrize("proof_kind", ["empty", "forged"])
def test_empty_or_self_claimed_proof_fails_closed(tmp_path, proof_kind: str) -> None:
    graph = _graph()
    if proof_kind == "empty":
        proof_state = {}
    else:
        proof_state = _proof_state(graph, validated_step_ids=set())
        forged = proof_state["records"][0]
        forged.update(
            {
                "status": "validated",
                "achieved_proof_level": 4,
                "proof_authority": "self_claimed",
                "proof": {"accepted": True, "proof_level": "L4_procurement_ready"},
            }
        )
        proof_state["content_sha256"] = _digest(
            {key: value for key, value in proof_state.items() if key != "content_sha256"}
        )

    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        proof_state,
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert ledger["summary"]["any_route_closed"] is False
    assert ledger["summary"]["all_explored_graph_closed"] is False
    assert ledger["summary"]["reaction_proven_edge_count"] == 0
    if proof_kind == "forged":
        assert ledger["input_validation"]["reaction_proof_state"]["rejected_record_count"] == 1


def test_projection_and_digest_are_stable_across_input_order(tmp_path) -> None:
    graph = _graph()
    queue = _stock_queue(tmp_path)
    proofs = _proof_state(graph, validated_step_ids={"step:1", "step:2"})
    first = project_frontier_ledger(
        graph,
        queue,
        proofs,
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    reordered_graph = deepcopy(graph)
    reordered_graph["steps"].reverse()
    reordered_queue = deepcopy(queue)
    reordered_queue["jobs"].reverse()
    reordered_queue["content_sha256"] = _digest(
        {key: value for key, value in reordered_queue.items() if key != "content_sha256"}
    )
    reordered_proofs = deepcopy(proofs)
    reordered_proofs["records"].reverse()
    reordered_proofs["content_sha256"] = _digest(
        {key: value for key, value in reordered_proofs.items() if key != "content_sha256"}
    )
    second = project_frontier_ledger(
        reordered_graph,
        reordered_queue,
        reordered_proofs,
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert first["summary"] == second["summary"]
    assert first["molecules"] == second["molecules"]
    assert first["edges"] == second["edges"]
    assert first["content_sha256"] != second["content_sha256"]
    assert (
        first["input_bindings"]["frontier_queue_content_sha256"]
        != second["input_bindings"]["frontier_queue_content_sha256"]
    )


def test_validator_rejects_tampered_projection(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )
    ledger["summary"]["all_explored_graph_closed"] = True

    assert "frontier_ledger_content_digest_invalid" in validate_frontier_ledger(ledger)


def test_validator_fences_rehashed_ledger_to_expected_input_bindings(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )
    expected = deepcopy(ledger["input_bindings"])
    ledger["input_bindings"]["campaign_policy_sha256"] = "f" * 64
    ledger["content_sha256"] = _digest(
        {key: value for key, value in ledger.items() if key != "content_sha256"}
    )

    reasons = validate_frontier_ledger(
        ledger,
        trusted_stock_provider_instances=_trusted_stock_providers(),
        expected_input_bindings=expected,
    )

    assert (
        "frontier_ledger_input_binding_mismatch:campaign_policy_sha256"
        in reasons
    )


def test_validator_recomputes_closure_after_attacker_rehash(tmp_path) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )
    ledger["summary"]["all_explored_graph_closed"] = True
    ledger["content_sha256"] = _digest(
        {key: value for key, value in ledger.items() if key != "content_sha256"}
    )

    reasons = validate_frontier_ledger(ledger)

    assert "frontier_ledger_content_digest_invalid" not in reasons
    assert (
        "frontier_ledger_summary_mismatch:all_explored_graph_closed" in reasons
    )


@pytest.mark.parametrize(
    "field",
    [
        "any_benchmark_route_closed",
        "all_explored_benchmark_closed",
        "any_procurement_route_closed",
        "all_explored_procurement_closed",
    ],
)
def test_validator_recomputes_each_stock_authority_fixed_point_after_rehash(
    tmp_path,
    field: str,
) -> None:
    graph = _graph()
    ledger = project_frontier_ledger(
        graph,
        _stock_queue(tmp_path),
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )
    ledger["root"]["closure"][field] = False
    ledger["summary"][field] = False
    for molecule in ledger["molecules"].values():
        molecule["closure"][field] = False
    for edge in ledger["edges"].values():
        edge["closure"][field] = False
    ledger["content_sha256"] = _digest(
        {key: value for key, value in ledger.items() if key != "content_sha256"}
    )

    reasons = validate_frontier_ledger(
        ledger,
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert any(field in reason for reason in reasons)


def test_malformed_queue_metadata_fails_closed_without_projection_crash(tmp_path) -> None:
    graph = _graph()
    queue = _stock_queue(tmp_path)
    queue["jobs"][0]["metadata"] = "not-an-object"
    queue["content_sha256"] = _digest(
        {key: value for key, value in queue.items() if key != "content_sha256"}
    )

    ledger = project_frontier_ledger(
        graph,
        queue,
        _proof_state(graph, validated_step_ids={"step:1", "step:2"}),
        trusted_stock_provider_instances=_trusted_stock_providers(),
    )

    assert ledger["summary"]["any_route_closed"] is False
    assert ledger["summary"]["all_explored_graph_closed"] is False
    assert ledger["input_validation"]["frontier_queue"]["valid"] is False
    assert any(
        reason.endswith(":metadata_not_object")
        for reason in ledger["input_validation"]["frontier_queue"]["reasons"]
    )


def test_exact_edge_signature_preserves_stoichiometry_without_dot_collisions() -> None:
    split_components = exact_edge_signature("CCO", ["CC", "O"])
    one_multicomponent_value = exact_edge_signature("CCO", ["CC.O"])
    duplicated_component = exact_edge_signature("CCO", ["CC", "O", "O"])

    assert split_components.startswith("edge:sha256:")
    assert len({split_components, one_multicomponent_value, duplicated_component}) == 3
