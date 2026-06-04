"""Mark a bufotalin iteration root as intentionally stopped by the user."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def mark_run_stopped(root: Path | str, *, reason: str = "user_cancelled") -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    events_path = root / "runner_events.jsonl"
    manifest = _read_json(manifest_path)
    now = datetime.now(timezone.utc).isoformat()
    already_marked = manifest.get("running") is False and manifest.get("stop_reason") == reason
    manifest.update(
        {
            "running": False,
            "stopped": True,
            "stop_reason": reason,
            "stopped_at": manifest.get("stopped_at") or now,
            "updated_at": now,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    event_written = False
    if not _has_stop_event(events_path, reason):
        _append_event(
            events_path,
            {
                "time": now,
                "event": "user_stop",
                "reason": reason,
                "manifest_updated": True,
            },
        )
        event_written = True
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "events": str(events_path),
        "already_marked": already_marked,
        "event_written": event_written,
        "running": manifest.get("running"),
        "stop_reason": manifest.get("stop_reason"),
        "stopped_at": manifest.get("stopped_at"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_stop_event(path: Path, reason: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "user_stop" and row.get("reason") == reason:
            return True
    return False


def _append_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark a bufotalin run as intentionally stopped.")
    parser.add_argument("root")
    parser.add_argument("--reason", default="user_cancelled")
    args = parser.parse_args()
    report = mark_run_stopped(args.root, reason=args.reason)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
