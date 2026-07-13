from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade_planner.interfaces.replay_pack import (
    ReplayPackError,
    load_replay_pack,
    run_replay_pack,
    with_replay_pack_digest,
)
from cascade_planner.runtime.paths import RuntimePaths


_PACK = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "examples"
    / "nirmatrelvir_v4_replay_pack.json"
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
        "stock_terminal_count": 7,
        "independent_source_groups": [
            "doi:10.1126/science.abl4784",
            "patent:WO2021250648A1",
        ],
        "accepted_expansion_count": 12,
        "attempt_count": 29,
        "model_invocations": 0,
        "visual_invocations": 0,
    }
    repeated = run_replay_pack(
        _PACK,
        paths=paths,
        run_id="nirmatrelvir-golden",
    )
    assert repeated["accepted"] is True
    assert repeated["stages"] == []
    assert repeated["observed"] == result["observed"]
    assert repeated["workbench_sha256"] == result["workbench_sha256"]


@pytest.mark.parametrize(
    ("stop_after", "expected_attempts"),
    (("materialization", 12), ("evidence", 16)),
)
def test_replay_resumes_and_reconstructs_same_science(
    tmp_path: Path,
    stop_after: str,
    expected_attempts: int,
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
    assert interrupted["observed"]["attempt_count"] == expected_attempts

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
