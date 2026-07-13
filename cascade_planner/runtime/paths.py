"""Configured filesystem boundaries for the V4 runtime."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    repository_root: Path
    runtime_root: Path
    runs_root: Path
    artifact_store_root: Path
    run_index_path: Path
    cache_root: Path
    source_root: Path
    external_data_root: Path
    model_root: Path
    vendor_root: Path

    @classmethod
    def discover(
        cls,
        *,
        repository_root: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RuntimePaths":
        env = dict(os.environ if environ is None else environ)
        repo = Path(
            repository_root or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()
        runtime = _configured_path(
            env,
            "AUTOPLANNER_RUNTIME_ROOT",
            repo / "results" / ".autoplanner",
        )
        external = _configured_path(
            env,
            "AUTOPLANNER_EXTERNAL_DATA_ROOT",
            repo / "data_external",
        )
        return cls(
            repository_root=repo,
            runtime_root=runtime,
            runs_root=_configured_path(
                env,
                "AUTOPLANNER_RUNS_ROOT",
                runtime / "runs",
            ),
            artifact_store_root=_configured_path(
                env,
                "AUTOPLANNER_ARTIFACT_STORE_ROOT",
                runtime / "artifacts",
            ),
            run_index_path=_configured_path(
                env,
                "AUTOPLANNER_RUN_INDEX_PATH",
                runtime / "run_index.sqlite3",
            ),
            cache_root=_configured_path(
                env,
                "AUTOPLANNER_CACHE_ROOT",
                runtime / "cache",
            ),
            source_root=_configured_path(
                env,
                "AUTOPLANNER_SOURCE_ROOT",
                runtime / "sources",
            ),
            external_data_root=external,
            model_root=_configured_path(
                env,
                "AUTOPLANNER_MODEL_ROOT",
                external / "models",
            ),
            vendor_root=_configured_path(
                env,
                "AUTOPLANNER_VENDOR_ROOT",
                repo / "vendor",
            ),
        )

    def ensure_runtime_directories(self) -> None:
        for path in (
            self.runtime_root,
            self.runs_root,
            self.artifact_store_root,
            self.run_index_path.parent,
            self.cache_root,
            self.source_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": str(self.repository_root),
            "runtime_root": str(self.runtime_root),
            "runs_root": str(self.runs_root),
            "artifact_store_root": str(self.artifact_store_root),
            "run_index_path": str(self.run_index_path),
            "cache_root": str(self.cache_root),
            "source_root": str(self.source_root),
            "external_data_root": str(self.external_data_root),
            "model_root": str(self.model_root),
            "vendor_root": str(self.vendor_root),
            "semantics": {
                "runtime_data_is_not_source_code": True,
                "external_data_is_configurable": True,
                "credentials_are_not_filesystem_defaults": True,
            },
        }


def _configured_path(
    environ: Mapping[str, str],
    name: str,
    default: Path,
) -> Path:
    value = str(environ.get(name) or "").strip()
    return Path(value or default).expanduser().resolve()


__all__ = ["RuntimePaths"]
