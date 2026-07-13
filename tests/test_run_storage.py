from __future__ import annotations

from pathlib import Path

from cascade_planner.runtime.run_index import RUN_MANIFEST_SCHEMA, RunIndex
from cascade_planner.runtime.run_storage import (
    publish_run_projection,
    rebuild_run_index,
    run_storage_object_stats,
)


def _manifest(run_id: str, revision: int = 1) -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "case_id": "case",
        "target_name": "target",
        "producer": "test",
        "status": "completed",
        "revision": revision,
        "updated_at": "2026-07-13T00:00:00Z",
        "run_dir": f"runs/{run_id}",
        "state_sha256": f"state-{revision}",
        "accepted": True,
        "cost_totals": {"model_invocations": 0},
        "graph": {"complete_route_count": 2},
        "deficits": {"proof": 0, "stock": 0},
        "metrics": {},
    }


def test_run_projection_deduplicates_and_rebuilds_index(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    first_dir = tmp_path / "run-1"
    second_dir = tmp_path / "run-2"
    first_dir.mkdir()
    second_dir.mkdir()
    first_artifact = first_dir / "proof.json"
    second_artifact = second_dir / "proof.json"
    first_artifact.write_text('{"accepted":true}\n', encoding="utf-8")
    second_artifact.write_bytes(first_artifact.read_bytes())

    first = publish_run_projection(
        runtime_root,
        manifest=_manifest("run-1"),
        artifacts={"proof": first_artifact},
        authority_scopes={"proof": "scientific_artifact_reference"},
    )
    second = publish_run_projection(
        runtime_root,
        manifest=_manifest("run-2"),
        artifacts={"proof": second_artifact},
        authority_scopes={"proof": "scientific_artifact_reference"},
    )
    stats = run_storage_object_stats(runtime_root)
    rebuilt_path = runtime_root / "rebuilt.sqlite3"
    rebuild = rebuild_run_index(runtime_root, index_path=rebuilt_path)
    rebuilt = RunIndex(rebuilt_path)

    assert first["index_health"]["accepted"] is True
    assert second["index_health"]["run_count"] == 2
    # One shared proof object plus one manifest object for each run.
    assert stats["object_count"] == 3
    assert stats["indexed_artifact_count"] == 2
    assert rebuild["manifest_count"] == 2
    assert rebuilt.health()["run_count"] == 2
    assert rebuilt.artifacts_for_run("run-1")[0]["ref"]["sha256"] == (
        rebuilt.artifacts_for_run("run-2")[0]["ref"]["sha256"]
    )
    assert first_artifact.read_text(encoding="utf-8") == '{"accepted":true}\n'
    assert rebuild["semantics"]["scientific_artifacts_were_not_modified"] is True
