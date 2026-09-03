"""Crash-safe publication for content-addressed immutable JSON event sets."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_bytes


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def publish_immutable_json_event(
    event_root: str | os.PathLike[str],
    value: Mapping[str, Any],
    *,
    content_sha256: str,
) -> tuple[Path, bool]:
    """Publish a fully written inode once; concurrent identical writers reuse it."""

    digest = str(content_sha256 or "").lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError("immutable_json_event_digest_invalid")
    root = Path(event_root).expanduser().resolve()
    path = root / digest[:2] / f"{digest}.json"
    if path.is_file():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".event.", suffix=".tmp", dir=path.parent)
    created = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(strict_canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
            created = True
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return path, created


__all__ = ["publish_immutable_json_event"]
