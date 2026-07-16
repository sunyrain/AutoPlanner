"""Shared enumeration and publication primitives for replayable event stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from cascade_planner.runtime.immutable_json_events import (
    publish_immutable_json_event,
)


RecordT = TypeVar("RecordT")


def load_replayable_event_records(
    event_root: Path,
    *,
    load: Callable[[Path], RecordT],
    event_id: Callable[[RecordT], str],
    corruption: Callable[[str], Exception],
    root_not_directory: str,
    duplicate_identity: str,
) -> list[RecordT]:
    """Replay a content-addressed event set and reject duplicate identities."""

    if not event_root.exists():
        return []
    if not event_root.is_dir():
        raise corruption(root_not_directory)
    records = [load(path) for path in sorted(event_root.glob("*/*.json"))]
    identities = [event_id(record) for record in records]
    if len(identities) != len(set(identities)):
        raise corruption(duplicate_identity)
    return records


def publish_replayable_event(
    event_root: Path,
    event: Mapping[str, Any],
    *,
    load: Callable[[Path], RecordT],
) -> tuple[Path, bool]:
    """Atomically publish an event and replay it before returning success."""

    path, created = publish_immutable_json_event(
        event_root,
        event,
        content_sha256=str(event.get("content_sha256") or ""),
    )
    load(path)
    return path, created


__all__ = ["load_replayable_event_records", "publish_replayable_event"]
