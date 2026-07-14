"""Digest-bound, atomically written storage for reaction template memory."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


TEMPLATE_LIBRARY_SCHEMA = "patent_reaction_template_library.v1"
TEMPLATE_RECORD_SCHEMA = "patent_reaction_template_record.v1"
DEFAULT_TEMPLATE_LIBRARY_NAME = "patent-reaction-template-library.json"


def read_template_library(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return build_template_library({}, generation=0), ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, "template_library_unreadable"
    if not isinstance(value, Mapping) or value.get("schema_version") != TEMPLATE_LIBRARY_SCHEMA:
        return {}, "template_library_schema_invalid"
    library = dict(value)
    if not valid_digest(library, "content_sha256"):
        return {}, "template_library_digest_invalid"
    templates = dict(library.get("templates") or {})
    if any(
        not isinstance(value, Mapping)
        or value.get("schema_version") != TEMPLATE_RECORD_SCHEMA
        or str(value.get("template_id") or "") != str(key)
        or not valid_digest(value, "content_sha256")
        for key, value in templates.items()
    ):
        return {}, "template_record_digest_invalid"
    return library, ""


def build_template_library(
    templates: Mapping[str, Mapping[str, Any]],
    *,
    generation: int,
) -> dict[str, Any]:
    row = {
        "schema_version": TEMPLATE_LIBRARY_SCHEMA,
        "generation": generation,
        "templates": {str(key): dict(value) for key, value in sorted(templates.items())},
        "semantics": {
            "external_cross_campaign_memory": True,
            "repository_contains_no_target_dossier": True,
            "corruption_fails_closed": True,
            "model_invocations": 0,
        },
    }
    row["content_sha256"] = template_digest(row)
    return row


@contextmanager
def template_library_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def write_template_library(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def valid_digest(value: Mapping[str, Any], field: str) -> bool:
    row = dict(value)
    supplied = str(row.pop(field, ""))
    return bool(supplied and supplied == template_digest(row))


def template_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DEFAULT_TEMPLATE_LIBRARY_NAME",
    "TEMPLATE_LIBRARY_SCHEMA",
    "TEMPLATE_RECORD_SCHEMA",
    "build_template_library",
    "read_template_library",
    "template_digest",
    "template_library_lock",
    "valid_digest",
    "write_template_library",
]
