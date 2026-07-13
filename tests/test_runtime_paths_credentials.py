from __future__ import annotations

from pathlib import Path

from cascade_planner.runtime.credentials import resolve_codex_credential
from cascade_planner.runtime.paths import RuntimePaths


def test_runtime_paths_are_configurable_and_create_only_runtime_dirs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    external = tmp_path / "external"
    paths = RuntimePaths.discover(
        repository_root=repo,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(runtime),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(external),
        },
    )

    paths.ensure_runtime_directories()

    assert paths.runtime_root == runtime.resolve()
    assert paths.external_data_root == external.resolve()
    assert paths.model_root == (external / "models").resolve()
    assert paths.artifact_store_root.is_dir()
    assert paths.run_index_path.parent.is_dir()
    assert not paths.model_root.exists()
    assert paths.to_dict()["semantics"][
        "credentials_are_not_filesystem_defaults"
    ] is True


def test_codex_credential_has_no_repository_key_fallback(tmp_path: Path) -> None:
    (tmp_path / "key.txt").write_text("must-not-be-read", encoding="utf-8")

    assert resolve_codex_credential(environ={}) is None


def test_codex_credential_prefers_environment_and_redacts_repr(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "configured.key"
    key_path.write_text("file-secret", encoding="utf-8")

    credential = resolve_codex_credential(
        environ={
            "AUTOPLANNER_CODEX_API_KEY": "env-secret",
            "AUTOPLANNER_CODEX_KEY_PATH": str(key_path),
        }
    )

    assert credential is not None
    assert credential.value == "env-secret"
    assert credential.source == "env:AUTOPLANNER_CODEX_API_KEY"
    assert "env-secret" not in repr(credential)
