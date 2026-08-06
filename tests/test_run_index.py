from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.run_index import (
    RUN_MANIFEST_SCHEMA,
    RunIndex,
)


def _manifest(run_id: str, *, revision: int = 1) -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "case_id": "case-1",
        "target_name": "target",
        "status": "running" if revision == 1 else "completed",
        "revision": revision,
        "updated_at": f"2026-07-13T00:00:0{revision}Z",
        "run_dir": f"runs/{run_id}",
        "state_sha256": f"state-{revision}",
        "accepted": revision > 1,
        "cost_totals": {
            "model_invocations": 0,
            "attempt_runs": revision,
            "accepted_expansions": max(0, revision - 1),
        },
        "graph": {
            "molecule_count": revision + 2,
            "hyperedge_count": revision,
            "complete_route_count": max(0, revision - 1),
        },
        "deficits": {"proof": max(0, 2 - revision), "stock": 0},
        "metrics": {"sha256": f"metrics-{revision}"},
        "semantics": {
            "manifest_is_replay_input_for_operational_index": True,
            "manifest_does_not_grant_scientific_authority": True,
        },
    }


def test_run_index_is_wal_projection_and_rejects_stale_revision(
    tmp_path: Path,
) -> None:
    index = RunIndex(tmp_path / "runs.sqlite3")
    index.upsert_run(_manifest("run-1", revision=2))
    index.upsert_run(_manifest("run-1", revision=1))

    saved = index.get_run("run-1")
    health = index.health()

    assert saved is not None
    assert saved["revision"] == 2
    assert saved["status"] == "completed"
    assert health["accepted"] is True
    assert health["journal_mode"].casefold() == "wal"
    assert health["semantics"]["index_grants_no_scientific_authority"] is True


def test_run_index_tracks_artifacts_and_idempotent_tasks(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "objects")
    ref = store.put_json({"route": "proof"}, logical_name="proof.json")
    index = RunIndex(tmp_path / "runs.sqlite3")
    index.upsert_run(_manifest("run-1"))
    index.index_artifact(
        run_id="run-1",
        artifact_id="proof",
        ref=ref,
        revision=1,
        authority_scope="scientific_artifact_reference",
    )
    task = {
        "run_id": "run-1",
        "task_id": "task-1",
        "kind": "evidence_extract",
        "status": "pending",
        "idempotency_key": "evidence:source-1:revision-1",
        "input_revision": 1,
        "updated_at": "2026-07-13T00:00:00Z",
    }
    index.upsert_task(task)
    index.upsert_task({**task, "status": "completed"})

    artifacts = index.artifacts_for_run("run-1")
    tasks = index.tasks_for_run("run-1")

    assert artifacts[0]["ref"]["sha256"] == ref.sha256
    assert artifacts[0]["authority_scope"] == "scientific_artifact_reference"
    assert tasks == [{**task, "status": "completed"}]


def test_run_history_removal_only_deletes_rebuildable_projections(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "objects")
    ref = store.put_bytes(b"preserved evidence", logical_name="evidence.json")
    index = RunIndex(tmp_path / "runs.sqlite3")
    manifest = _manifest("history-run", revision=2)
    index.upsert_run(manifest)
    index.index_artifact(
        run_id="history-run",
        artifact_id="evidence",
        ref=ref,
        revision=2,
    )
    index.upsert_task(
        {
            "run_id": "history-run",
            "task_id": "task-1",
            "kind": "validation",
            "status": "completed",
            "idempotency_key": "validation:history-run",
        }
    )

    removed = index.remove_run_projection("history-run")

    assert removed["removed"] is True
    assert removed["removed_artifact_projection_count"] == 1
    assert removed["removed_task_projection_count"] == 1
    assert removed["scientific_artifacts_preserved"] is True
    assert index.get_run("history-run") is None
    assert index.artifacts_for_run("history-run") == []
    assert index.tasks_for_run("history-run") == []
    assert store.read_bytes(ref) == b"preserved evidence"

    index.upsert_run(manifest)
    assert index.get_run("history-run") is not None


def test_run_index_rebuild_does_not_touch_scientific_objects(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "objects")
    ref = store.put_bytes(b"scientific bytes", logical_name="edge-proof.json")
    manifest = _manifest("rebuild-run", revision=2)
    manifest["indexed_artifacts"] = [
        {
            "artifact_id": "edge-proof",
            "revision": 2,
            "authority_scope": "scientific_artifact_reference",
            "ref": ref.to_dict(),
        }
    ]
    manifest["indexed_tasks"] = [
        {
            "task_id": "stock-audit",
            "kind": "stock_audit",
            "status": "completed",
            "idempotency_key": "stock:leaf-1:revision-2",
            "input_revision": 2,
            "output_sha256": ref.sha256,
            "updated_at": "2026-07-13T00:00:02Z",
        }
    ]
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    first = RunIndex(tmp_path / "first.sqlite3")
    first.rebuild_from_manifest_paths([manifest_path])
    second = RunIndex(tmp_path / "rebuilt.sqlite3")
    report = second.rebuild_from_manifest_paths([manifest_path])

    assert first.get_run("rebuild-run") == second.get_run("rebuild-run")
    assert first.artifacts_for_run("rebuild-run") == second.artifacts_for_run(
        "rebuild-run"
    )
    assert second.tasks_for_run("rebuild-run")[0]["status"] == "completed"
    assert store.read_bytes(ref) == b"scientific bytes"
    assert report["semantics"]["scientific_artifacts_were_not_modified"] is True


def test_run_index_supports_concurrent_run_updates(tmp_path: Path) -> None:
    index = RunIndex(tmp_path / "runs.sqlite3")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda number: index.upsert_run(_manifest(f"run-{number}")),
                range(32),
            )
        )

    assert len(index.list_runs(limit=100)) == 32
    assert index.health()["run_count"] == 32
