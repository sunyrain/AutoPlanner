from __future__ import annotations

from pathlib import Path

import pytest

from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_registry_catalog import (
    RunRegistryCatalog,
    RunRegistryCatalogError,
    binding_from_paths,
    binding_from_registry_root,
)


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.discover(
        repository_root=tmp_path,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "runtime" / "run_index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
        },
    )


def test_run_registry_catalog_upserts_discovery_without_copying_run_state(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path / "primary")
    catalog = RunRegistryCatalog(tmp_path / "catalog.sqlite3")
    first = binding_from_paths(
        paths,
        registry_id="paper-case-1",
        registry_label="Paper case 1",
        project_id="paper25",
        project_label="Paper 25-step panel",
        case_id="case1",
    )

    catalog.register(first)
    catalog.register(
        binding_from_paths(
            paths,
            registry_id="paper-case-1",
            registry_label="Paper case 1 updated",
            project_id="paper25",
            project_label="Paper 25-step panel",
            case_id="case1",
        )
    )

    rows = catalog.list_registries()
    assert len(rows) == 1
    assert rows[0].registry_label == "Paper case 1 updated"
    assert rows[0].runtime_paths().run_index_path == paths.run_index_path
    assert "status" not in rows[0].to_dict()
    assert "run_id" not in rows[0].to_dict()


def test_run_registry_catalog_rejects_two_ids_for_one_run_index(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path / "primary")
    catalog = RunRegistryCatalog(tmp_path / "catalog.sqlite3")
    for identity in ("first", "second"):
        binding = binding_from_paths(
            paths,
            registry_id=identity,
            registry_label=identity,
            project_id="project",
            project_label="Project",
        )
        if identity == "first":
            catalog.register(binding)
        else:
            with pytest.raises(
                RunRegistryCatalogError,
                match="run_registry_path_already_registered",
            ):
                catalog.register(binding)


def test_disabled_registry_is_not_returned_by_the_default_catalog_view(
    tmp_path: Path,
) -> None:
    catalog = RunRegistryCatalog(tmp_path / "catalog.sqlite3")
    binding = binding_from_registry_root(
        tmp_path / "panel",
        registry_id="panel-case",
        registry_label="Panel case",
        project_id="project",
        project_label="Project",
        repository_root=tmp_path,
    )
    catalog.register(binding)

    catalog.set_enabled("panel-case", enabled=False)

    assert catalog.list_registries() == []
    assert [row.registry_id for row in catalog.list_registries(enabled_only=False)] == [
        "panel-case"
    ]


def test_binding_from_panel_root_roundtrips_all_runtime_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "panel" / "case1"
    binding = binding_from_registry_root(
        root,
        registry_id="case1",
        registry_label="Case 1",
        project_id="paper25",
        project_label="Paper 25-step panel",
        case_id="synthexfig1-001",
        repository_root=tmp_path,
    )
    paths = binding.runtime_paths()

    assert paths.runtime_root == root / "runtime"
    assert paths.runs_root == root / "runs"
    assert paths.artifact_store_root == root / "artifacts"
    assert paths.run_index_path == root / "runtime" / "run_index.sqlite3"
    assert paths.external_data_root == root / "external"
