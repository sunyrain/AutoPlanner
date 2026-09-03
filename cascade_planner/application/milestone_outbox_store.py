"""Atomic local persistence helpers for milestone subscription outboxes."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping


@contextmanager
def outbox_lock(outbox: Path) -> Iterator[None]:
    lock = outbox / ".writer-lock"
    outbox.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 10.0
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 60.0:
                    lock.rmdir()
                    continue
            except (FileNotFoundError, OSError):
                continue
            if time.monotonic() >= deadline:
                raise ValueError("milestone_subscription_writer_lock_timeout")
            time.sleep(0.01)
    try:
        yield
    finally:
        lock.rmdir()


def write_receipt(outbox: Path, receipt: Mapping[str, Any]) -> Path:
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{receipt['content_sha256']}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(receipt):
            raise ValueError("milestone_subscription_receipt_collision")
        return path
    atomic_json(path, receipt)
    return path


def write_latest(outbox: Path, receipt: Mapping[str, Any]) -> None:
    atomic_json(
        outbox / "latest.json",
        {
            "schema_version": "campaign_milestone_subscription_pointer.v1",
            "subscription_id": receipt["subscription_id"],
            "receipt_sha256": receipt["content_sha256"],
        },
    )


def load_latest(
    outbox: Path,
    *,
    expected_run_id: str,
    receipt_schema: str,
    digest: Any,
) -> dict[str, Any]:
    try:
        pointer = json.loads((outbox / "latest.json").read_text(encoding="utf-8"))
        receipt_sha256 = str(pointer.get("receipt_sha256") or "")
        receipt = json.loads(
            (outbox / f"{receipt_sha256}.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("milestone_subscription_latest_invalid") from exc
    if (
        pointer.get("schema_version")
        != "campaign_milestone_subscription_pointer.v1"
        or receipt.get("content_sha256") != receipt_sha256
        or digest(receipt) != receipt_sha256
        or receipt.get("schema_version") != receipt_schema
        or str(receipt.get("run_id") or "") != expected_run_id
    ):
        raise ValueError("milestone_subscription_latest_invalid")
    return receipt


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["load_latest", "outbox_lock", "write_latest", "write_receipt"]
