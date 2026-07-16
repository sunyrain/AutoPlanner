"""Digest-bound external memory for validated Program observations."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_EXPERIENCE_LIBRARY_SCHEMA = "program_experience_library.v1"
PROGRAM_EXPERIENCE_RECORD_SCHEMA = "program_experience_record.v1"
DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME = "program-experience-library.json"


def build_program_experience_library(
    experiences: Mapping[str, Mapping[str, Any]], *, generation: int
) -> dict[str, Any]:
    row = {
        "schema_version": PROGRAM_EXPERIENCE_LIBRARY_SCHEMA,
        "generation": max(0, int(generation)),
        "experiences": {
            str(key): dict(value) for key, value in sorted(experiences.items())
        },
        "semantics": {
            "external_cross_campaign_memory": True,
            "learns_only_from_replay_validated_experimental_claims": True,
            "positive_negative_inconclusive_and_conflicting_results_retained": True,
            "memory_changes_ranking_and_validation_priority_only": True,
            "cannot_grant_program_validation_proof_completion_or_acceptance": True,
            "cannot_mutate_or_disable_capability_catalog": True,
            "corruption_fails_closed": True,
        },
    }
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def read_program_experience_library(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return build_program_experience_library({}, generation=0), ""
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, "program_experience_library_unreadable"
    if not isinstance(value, Mapping):
        return {}, "program_experience_library_not_object"
    library = dict(value)
    reasons = validate_program_experience_library(library)
    return (library, "") if not reasons else ({}, reasons[0])


def validate_program_experience_library(value: Mapping[str, Any]) -> list[str]:
    library = dict(value)
    reasons: list[str] = []
    if library.get("schema_version") != PROGRAM_EXPERIENCE_LIBRARY_SCHEMA:
        reasons.append("program_experience_library_schema_invalid")
    if not _digest_valid(library):
        reasons.append("program_experience_library_digest_invalid")
    experiences = library.get("experiences")
    if not isinstance(experiences, dict):
        reasons.append("program_experience_records_invalid")
        return sorted(set(reasons))
    for experience_id, record in experiences.items():
        if (
            not isinstance(record, Mapping)
            or record.get("schema_version") != PROGRAM_EXPERIENCE_RECORD_SCHEMA
            or record.get("experience_id") != experience_id
            or not _digest_valid(record)
        ):
            reasons.append("program_experience_record_invalid")
            break
    return sorted(set(reasons))


def write_program_experience_library(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


@contextmanager
def program_experience_library_lock(path: str | Path) -> Iterator[None]:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
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


def _digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    try:
        return bool(observed) and observed == strict_canonical_json_sha256(material)
    except (TypeError, ValueError):
        return False


__all__ = [
    "DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME",
    "PROGRAM_EXPERIENCE_LIBRARY_SCHEMA",
    "PROGRAM_EXPERIENCE_RECORD_SCHEMA",
    "build_program_experience_library",
    "program_experience_library_lock",
    "read_program_experience_library",
    "validate_program_experience_library",
    "write_program_experience_library",
]
