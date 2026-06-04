"""Watch a bufotalin long run and export/audit final artifacts after it exits."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a bufotalin long-run PID, then export and audit.")
    parser.add_argument("root")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--min-hours", type=float, default=12.0)
    parser.add_argument(
        "--refresh-final-candidates",
        action="store_true",
        help="Refresh final_candidates after each heartbeat while the runner is still active.",
    )
    args = parser.parse_args()
    report = watch_and_finalize(
        Path(args.root),
        pid=int(args.pid),
        poll_s=float(args.poll_s),
        min_hours=float(args.min_hours),
        refresh_final_candidates=bool(args.refresh_final_candidates),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def watch_and_finalize(
    root: Path,
    *,
    pid: int,
    poll_s: float = 60.0,
    min_hours: float = 12.0,
    refresh_final_candidates: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "completion_watcher_events.jsonl"
    _event(
        log_path,
        "watch_start",
        {
            "pid": pid,
            "root": str(root),
            "min_hours": min_hours,
            "refresh_final_candidates": bool(refresh_final_candidates),
        },
    )
    while _pid_alive(pid):
        _event(log_path, "watch_heartbeat", {"pid": pid, "alive": True})
        if refresh_final_candidates:
            refresh = _export_final_candidates(root)
            _event(log_path, "periodic_final_export", refresh)
        time.sleep(max(1.0, poll_s))
    _event(log_path, "runner_exited", {"pid": pid})
    export = _export_final_candidates(root)
    _event(log_path, "final_export", export)
    quality = _audit_final_candidate_quality(root)
    _event(log_path, "final_candidate_quality_audit", quality)
    audit = _run_command(
        [
            sys.executable,
            "scripts/audit_bufotalin_12h_completion.py",
            str(root),
            "--min-hours",
            str(float(min_hours)),
        ]
    )
    _event(log_path, "completion_audit", audit)
    report = {
        "root": str(root),
        "pid": pid,
        "final_export_returncode": export["returncode"],
        "final_quality_returncode": quality["returncode"],
        "audit_returncode": audit["returncode"],
        "completed": audit["returncode"] == 0 and quality["returncode"] == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "completion_watcher_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _export_final_candidates(root: Path) -> dict[str, Any]:
    return _run_command(
        [
            sys.executable,
            "scripts/export_bufotalin_final_candidates.py",
            str(root),
            "--output-dir",
            str(root / "final_candidates"),
            "--top-native",
            "5",
        ]
    )


def _audit_final_candidate_quality(root: Path) -> dict[str, Any]:
    return _run_command(
        [
            sys.executable,
            "scripts/audit_bufotalin_final_candidate_quality.py",
            str(root),
            "--output",
            str(root / "final_candidate_quality_audit.json"),
        ]
    )


def _run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _event(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
