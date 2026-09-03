from __future__ import annotations

from cascade_planner.web.workspace_visibility import WorkspaceVisibilityStore


def test_workspace_visibility_deletes_route_and_queue_independently_and_restores(
    tmp_path,
) -> None:
    store = WorkspaceVisibilityStore(tmp_path / "workspace_visibility.json")

    route = store.hide_route("run:example")
    queue = store.hide_queue_run("example")

    assert route["scientific_artifacts_preserved"] is True
    assert queue["recoverable"] is True
    snapshot = store.snapshot()
    assert set(snapshot["hidden_routes"]) == {"run:example"}
    assert set(snapshot["hidden_queue_runs"]) == {"example"}

    restored = store.restore(scope="routes")

    assert restored["restored_count"] == 1
    snapshot = store.snapshot()
    assert snapshot["hidden_routes"] == {}
    assert set(snapshot["hidden_queue_runs"]) == {"example"}


def test_workspace_visibility_persists_across_store_instances(tmp_path) -> None:
    path = tmp_path / "workspace_visibility.json"
    WorkspaceVisibilityStore(path).hide_route("showcase:one")

    snapshot = WorkspaceVisibilityStore(path).snapshot()

    assert set(snapshot["hidden_routes"]) == {"showcase:one"}
    assert snapshot["revision"] == 1
