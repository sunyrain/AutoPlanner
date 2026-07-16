"""Recover CAS pins from replayable event stores without trusting RunIndex."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from cascade_planner.interfaces.campaign_gateway_contract import CampaignGatewayError
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RunIndex


def replay_store_pinned_digests(
    paths: RuntimePaths,
    index: RunIndex,
    *,
    event_glob: str,
    store_marker: str,
    store_factory: Callable[[str, Path, ArtifactStore], Any],
    store_errors: tuple[type[BaseException], ...],
    ref_keys: Iterable[str],
    label: str,
) -> set[str]:
    """Discover stores, replay them, and collect immutable artifact refs."""

    directories: set[Path] = set()
    if paths.runs_root.is_dir():
        for event_root in paths.runs_root.glob(event_glob):
            directory = _run_dir(event_root)
            if directory is not None:
                directories.add(directory)
    marker = Path(store_marker)
    for manifest in index.list_runs(limit=10_000):
        value = str(manifest.get("run_dir") or "").strip()
        if value:
            directory = Path(value).expanduser().resolve()
            if (directory / marker).exists():
                directories.add(directory)
    artifacts = ArtifactStore(paths.artifact_store_root)
    pinned: set[str] = set()
    for directory in sorted(directories):
        run_id = _run_id(directory, label=label)
        try:
            replay = store_factory(run_id, directory, artifacts).replay()
        except store_errors as exc:
            raise CampaignGatewayError(str(exc)) from exc
        for event in replay["events"]:
            for key in ref_keys:
                digest = str(dict(event.get(key) or {}).get("sha256") or "")
                if digest:
                    pinned.add(digest)
    return pinned


def _run_dir(event_root: Path) -> Path | None:
    for parent in (event_root, *event_root.parents):
        if parent.name == ".autoplanner":
            return parent.parent.resolve()
    return None


def _run_id(directory: Path, *, label: str) -> str:
    spec_path = directory / ".autoplanner" / "kernel" / "run_spec.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_run_spec_unreadable:{directory.name}") from exc
    run_id = str(spec.get("run_id") or "") if isinstance(spec, dict) else ""
    if not run_id:
        raise ValueError(f"{label}_run_id_missing:{directory.name}")
    return run_id


__all__ = ["replay_store_pinned_digests"]
