from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cascade_planner.application.fact_lifecycle import build_fact_lifecycle_event
from cascade_planner.application.proof_portfolio import compile_proof_portfolio
from cascade_planner.interfaces.replay_pack import (
    ReplayPackError,
    load_replay_pack,
    run_replay_pack,
    with_replay_pack_digest,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.paths import RuntimePaths


_PACK = (
    Path(__file__).resolve().parents[1] / "config" / "examples" / "nirmatrelvir_v4_replay_pack.json"
)
_ARTEMISININ_PACK = (
    Path(__file__).resolve().parents[1] / "config" / "examples" / "artemisinin_v4_replay_pack.json"
)


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def test_nirmatrelvir_pack_replays_two_complete_routes_without_models(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    result = run_replay_pack(
        _PACK,
        paths=paths,
        run_id="nirmatrelvir-golden",
    )

    assert result["accepted"] is True
    assert result["status"] == "completed"
    assert result["observed"] == {
        "accepted": True,
        "complete_route_count": 2,
        "selected_route_count": 2,
        "hyperedge_count": 12,
        "validated_edge_count": 12,
        "exact_record_count": 15,
        "active_exact_record_count": 15,
        "procedure_record_count": 15,
        "active_procedure_record_count": 15,
        "condition_complete_procedure_count": 0,
        "condition_partial_procedure_count": 7,
        "condition_unparsed_procedure_count": 8,
        "condition_complete_route_count": 0,
        "process_ready_route_count": 0,
        "stock_terminal_count": 7,
        "independent_source_groups": [
            "doi:10.1126/science.abl4784",
            "patent:WO2021250648A1",
        ],
        "fact_lifecycle_event_count": 0,
        "inactive_fact_count": 0,
        "revoked_fact_count": 0,
        "expired_fact_count": 0,
        "accepted_expansion_count": 12,
        "attempt_count": 12,
        "settled_task_count": 29,
        "model_invocations": 0,
        "visual_invocations": 0,
    }
    replay_service = RetrosynthesisCampaignService.open(
        paths.runtime_root,
        result["run_dir"],
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )
    proposal_origins = {
        origin["origin_kind"]
        for hypothesis in replay_service.graph_store.load()["hypotheses"].values()
        for origin in hypothesis["origin_records"]
    }
    assert proposal_origins == {"literature_replay"}
    kernel_sha256_before = replay_service.kernel.state.to_dict()["content_sha256"]
    graph_sha256_before = replay_service.graph_store.load()["scientific_sha256"]
    empty_program_store = replay_service.program_store()
    admitted_programs = replay_service.admit_programs(enable_program_admission=True)
    durable_program_store = replay_service.program_store()

    assert empty_program_store["status"]["event_count"] == 0
    assert admitted_programs["created"] is True
    assert admitted_programs["store"]["event_count"] == 1
    assert admitted_programs["store"]["oracle"]["accepted"] is True
    assert admitted_programs["event"]["counts"] == {
        "chemical_states": 18,
        "operation_nodes": 12,
        "programs": 12,
        "routes": 2,
    }
    assert durable_program_store["replay"]["event_count"] == 1
    assert replay_service.kernel.state.to_dict()["content_sha256"] == kernel_sha256_before
    assert replay_service.graph_store.load()["scientific_sha256"] == graph_sha256_before
    repeated = run_replay_pack(
        _PACK,
        paths=paths,
        run_id="nirmatrelvir-golden",
    )
    assert repeated["accepted"] is True
    assert repeated["stages"] == []
    assert repeated["observed"] == result["observed"]
    assert repeated["workbench_sha256"] == result["workbench_sha256"]


def test_artemisinin_pack_replays_into_shadow_program_store_without_mutation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    result = run_replay_pack(
        _ARTEMISININ_PACK,
        paths=paths,
        run_id="artemisinin-program-migration",
    )
    service = RetrosynthesisCampaignService.open(
        paths.runtime_root,
        result["run_dir"],
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )
    kernel_sha256 = service.kernel.state.to_dict()["content_sha256"]
    graph_sha256 = service.graph_store.load()["scientific_sha256"]

    admitted = service.admit_programs(enable_program_admission=True)
    replay = service.program_store()["replay"]

    assert result["accepted"] is True
    assert admitted["event"]["counts"] == {
        "chemical_states": 5,
        "operation_nodes": 2,
        "programs": 2,
        "routes": 2,
    }
    assert admitted["store"]["oracle"]["accepted"] is True
    assert replay["event_count"] == 1
    assert service.kernel.state.to_dict()["content_sha256"] == kernel_sha256
    assert service.graph_store.load()["scientific_sha256"] == graph_sha256


@pytest.mark.parametrize(
    ("stop_after", "expected_tasks"),
    (("materialization", 12), ("evidence", 16)),
)
def test_replay_resumes_and_reconstructs_same_science(
    tmp_path: Path,
    stop_after: str,
    expected_tasks: int,
) -> None:
    paths = _paths(tmp_path)
    interrupted = run_replay_pack(
        _PACK,
        paths=paths,
        run_id="resumable",
        stop_after=stop_after,
    )

    assert interrupted["interrupted"] is True
    assert interrupted["status"] == "paused"
    assert interrupted["observed"]["attempt_count"] == 12
    assert interrupted["observed"]["settled_task_count"] == expected_tasks

    resumed = run_replay_pack(_PACK, paths=paths, run_id="resumable")
    fresh = run_replay_pack(
        _PACK,
        paths=_paths(tmp_path / "fresh-runtime"),
        run_id="resumable",
    )

    assert resumed["accepted"] is fresh["accepted"] is True
    assert resumed["observed"] == fresh["observed"]
    assert resumed["graph_scientific_sha256"] == fresh["graph_scientific_sha256"]
    assert resumed["portfolio_sha256"] == fresh["portfolio_sha256"]
    assert resumed["workbench_sha256"] == fresh["workbench_sha256"]


def test_replay_pack_rejects_content_tampering() -> None:
    tampered = json.loads(_PACK.read_text(encoding="utf-8"))
    tampered["target"]["name"] = "tampered"

    with pytest.raises(ReplayPackError, match="replay_pack_digest_invalid"):
        load_replay_pack(tampered)


def test_replay_pack_applies_lifecycle_stage_idempotently(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    seed = run_replay_pack(
        _PACK,
        paths=paths,
        run_id="lifecycle-seed",
        stop_after="evidence",
    )
    seed_service = RetrosynthesisCampaignService.open(
        paths.runtime_root,
        seed["run_dir"],
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )
    seed_graph = seed_service.graph_store.load()
    source_id, source = next(
        (source_id, source)
        for source_id, source in seed_graph["source_bindings"].items()
        if source.get("source_kind") == "patent"
    )
    event = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id=source_id,
        subject_content_sha256=source["content_sha256"],
        action="revoke",
        effective_at="2026-07-15T12:00:00Z",
        reason_codes=["showcase_source_retraction"],
    )
    lifecycle_pack = deepcopy(load_replay_pack(_PACK))
    lifecycle_pack["fact_lifecycle_events"] = [event]
    lifecycle_pack = with_replay_pack_digest(lifecycle_pack)

    first = run_replay_pack(
        lifecycle_pack,
        paths=paths,
        run_id="lifecycle-replay",
        stop_after="lifecycle",
    )
    service = RetrosynthesisCampaignService.open(
        paths.runtime_root,
        first["run_dir"],
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )
    graph = service.graph_store.load()
    portfolio = compile_proof_portfolio(graph, acceptance_spec=service.kernel.spec.acceptance)

    assert first["interrupted"] is True
    assert first["stages"][-1] == {
        "stage": "lifecycle",
        "status": "executed",
        "work_count": 1,
    }
    assert event["event_id"] in graph["fact_lifecycle_events"]
    assert first["observed"]["inactive_fact_count"] == 1
    assert first["observed"]["revoked_fact_count"] == 1
    assert any(
        fact.get("subject_id") == source_id
        for proof in portfolio["edge_proofs"].values()
        for fact in proof.get("inactive_facts") or []
    )
    assert any(
        proof.get("exact_source_bound") is False and proof.get("reaction_validated") is True
        for proof in portfolio["edge_proofs"].values()
    )

    replayed = run_replay_pack(
        lifecycle_pack,
        paths=paths,
        run_id="lifecycle-replay",
        stop_after="lifecycle",
    )
    assert replayed["graph_scientific_sha256"] == first["graph_scientific_sha256"]
    assert replayed["observed"]["fact_lifecycle_event_count"] == 1
    assert replayed["stages"][-1]["status"] == "reused"


def test_replay_pack_rejects_duplicate_edges_and_unbound_source_artifacts() -> None:
    duplicate = json.loads(_PACK.read_text(encoding="utf-8"))
    duplicate["reactions"].append(dict(duplicate["reactions"][0]))
    duplicate["budget"]["max_accepted_expansions"] = 13
    duplicate = with_replay_pack_digest(duplicate)
    with pytest.raises(ReplayPackError, match="replay_pack_reaction_duplicate"):
        load_replay_pack(duplicate)

    unbound = json.loads(_PACK.read_text(encoding="utf-8"))
    unbound["sources"][0]["binding"]["artifact_sha256"] = ""
    unbound = with_replay_pack_digest(unbound)
    with pytest.raises(ReplayPackError, match="replay_pack_source_invalid"):
        load_replay_pack(unbound)


def test_replay_fails_closed_on_stale_stock_and_expected_metric_mismatch(
    tmp_path: Path,
) -> None:
    stale = json.loads(_PACK.read_text(encoding="utf-8"))
    stale["inventory"]["artifact"]["retrieved_at"] = "2025-01-01T00:00:00Z"
    for offer in stale["inventory"]["artifact"]["offers"]:
        offer["checked_at"] = "2025-01-01T00:00:00Z"
    stale = with_replay_pack_digest(stale)
    stale_result = run_replay_pack(
        stale,
        paths=_paths(tmp_path / "stale"),
        run_id="stale-stock",
    )
    assert stale_result["accepted"] is False
    assert stale_result["observed"]["stock_terminal_count"] == 0
    assert stale_result["observed"]["accepted"] is False

    mismatch = json.loads(_PACK.read_text(encoding="utf-8"))
    mismatch["expected"]["hyperedge_count"] = 13
    mismatch = with_replay_pack_digest(mismatch)
    mismatch_result = run_replay_pack(
        mismatch,
        paths=_paths(tmp_path / "mismatch"),
        run_id="metric-mismatch",
    )
    assert mismatch_result["accepted"] is False
    assert mismatch_result["checks"]["hyperedge_count"] is False
