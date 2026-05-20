"""Execute selected commands from a pipeline manifest with bounded parallelism."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def run_manifest_commands(
    *,
    manifest_path: Path,
    output_log_dir: Path,
    stage: str | None = None,
    config: str | None = None,
    split: str | None = None,
    command_indices: list[int] | None = None,
    max_workers: int = 1,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = _select_commands(
        manifest.get("commands") or [],
        stage=stage,
        config=config,
        split=split,
        command_indices=command_indices,
    )
    if not commands:
        raise ValueError("no commands matched the requested filters")
    output_log_dir = Path(output_log_dir)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    if skip_existing:
        runnable = []
        for row in commands:
            if _outputs_exist(row):
                skipped.append({**_command_summary(row), "returncode": 0, "elapsed_s": 0.0, "skipped": True})
            else:
                runnable.append(row)
        commands = runnable

    if dry_run:
        return {
            "manifest": str(manifest_path),
            "dry_run": True,
            "selected": [_command_summary(row) for row in commands],
            "skipped": skipped,
        }
    if not commands:
        report = {
            "manifest": str(manifest_path),
            "stage": stage,
            "config": config,
            "split": split,
            "command_indices": command_indices or [],
            "max_workers": max_workers,
            "skip_existing": skip_existing,
            "elapsed_s": 0.0,
            "selected_count": 0,
            "skipped_count": len(skipped),
            "failed_count": 0,
            "results": skipped,
        }
        report_path = output_log_dir / "manifest_command_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    max_workers = max(1, int(max_workers or 1))
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    started = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_command = {
            executor.submit(_run_one, row, output_log_dir): row for row in commands
        }
        for future in concurrent.futures.as_completed(future_to_command):
            result = future.result()
            results.append(result)
            if result["returncode"] != 0:
                failures.append(result)

    results.extend(skipped)
    results.sort(key=lambda row: row["index"])
    report = {
        "manifest": str(manifest_path),
        "stage": stage,
        "config": config,
        "split": split,
        "command_indices": command_indices or [],
        "max_workers": max_workers,
        "skip_existing": skip_existing,
        "elapsed_s": round(time.monotonic() - started, 3),
        "selected_count": len(commands),
        "skipped_count": len(skipped),
        "failed_count": len(failures),
        "results": results,
    }
    report_path = output_log_dir / "manifest_command_run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        failed = [row["index"] for row in failures]
        raise RuntimeError(f"manifest commands failed: {failed}; report={report_path}")
    return report


def _select_commands(
    commands: list[dict[str, Any]],
    *,
    stage: str | None,
    config: str | None,
    split: str | None,
    command_indices: list[int] | None,
) -> list[dict[str, Any]]:
    selected = []
    wanted = set(command_indices or [])
    for index, command in enumerate(commands, start=1):
        if wanted and index not in wanted:
            continue
        if stage is not None and command.get("stage") != stage:
            continue
        if config is not None and command.get("config") != config:
            continue
        if split is not None and command.get("split") != split:
            continue
        selected.append({"index": index, **command})
    return selected


def _run_one(command: dict[str, Any], output_log_dir: Path) -> dict[str, Any]:
    index = int(command["index"])
    label = "_".join(
        str(part)
        for part in (index, command.get("stage"), command.get("config"), command.get("split"))
        if part
    )
    log_path = output_log_dir / f"command_{label}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            str(command["cmd"]),
            shell=True,
            cwd=Path.cwd(),
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    return {
        **_command_summary(command),
        "returncode": int(proc.returncode),
        "elapsed_s": round(elapsed, 3),
        "log": str(log_path),
        "skipped": False,
    }


def _command_summary(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(command["index"]),
        "stage": command.get("stage"),
        "config": command.get("config"),
        "split": command.get("split"),
        "shard_index": command.get("shard_index"),
        "num_shards": command.get("num_shards"),
        "outputs": command.get("outputs") or {},
        "cmd": command.get("cmd"),
    }


def _outputs_exist(command: dict[str, Any]) -> bool:
    outputs = command.get("outputs") or {}
    if not outputs:
        return False
    paths = []
    for value in outputs.values():
        if isinstance(value, str):
            paths.append(Path(value))
        elif isinstance(value, list):
            paths.extend(Path(item) for item in value if isinstance(item, str))
    return bool(paths) and all(path.exists() for path in paths)


def _parse_indices(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                out.extend(range(int(start), int(end) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run selected commands from a pipeline manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--index", action="append", default=[], help="Command index or range, e.g. 1 or 1-8")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    report = run_manifest_commands(
        manifest_path=Path(args.manifest),
        output_log_dir=Path(args.log_dir),
        stage=args.stage,
        config=args.config,
        split=args.split,
        command_indices=_parse_indices(args.index),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    print(json.dumps({
        "selected_count": report.get("selected_count", len(report.get("selected", []))),
        "skipped_count": report.get("skipped_count", len(report.get("skipped", []))),
        "failed_count": report.get("failed_count", 0),
        "elapsed_s": report.get("elapsed_s"),
        "selected": report.get("selected"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
