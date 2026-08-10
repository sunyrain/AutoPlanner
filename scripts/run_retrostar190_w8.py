#!/usr/bin/env python3
"""Run the four frozen W8 RetroStar-190 arms in isolated panel roots."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
W8_ARMS = (
    "chemenzy-only",
    "codex-only",
    "unified-round-robin",
    "unified-adaptive",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "benchmarks" / "retrostar190_v4.json",
    )
    parser.add_argument(
        "--stock-index",
        type=Path,
        default=(
            ROOT
            / "data_external"
            / "retrostar190"
            / "retrostar_emolecules_stock.sqlite3"
        ),
    )
    parser.add_argument(
        "--stock-sha256",
        default="30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c",
    )
    parser.add_argument("--stock-name", default="Retro*-190 eMolecules ~23M")
    parser.add_argument("--chemenzy-env-prefix", default="D:/conda/envs/py312")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument(
        "--execution-profile",
        choices=("fast", "standard", "proof"),
        default="standard",
    )
    parser.add_argument("--panel-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--parallel-arms", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--fixed-cutoff-wall-time-s", type=float, default=7_200.0)
    parser.add_argument("--fixed-cutoff-total-tasks", type=int, default=256)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help=(
            "Run only the first N manifest-ordered targets in every arm. "
            "The selected case IDs are frozen independently in each arm snapshot."
        ),
    )
    parser.add_argument("--arm", action="append", choices=W8_ARMS, default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.fixed_cutoff_wall_time_s <= 0:
        raise SystemExit("--fixed-cutoff-wall-time-s must be positive")
    if args.fixed_cutoff_total_tasks <= 0:
        raise SystemExit("--fixed-cutoff-total-tasks must be positive")

    output_root = args.output_root.expanduser().resolve()
    stock_index = args.stock_index.expanduser().resolve()
    actual_stock_sha256 = _file_sha256(stock_index)
    if actual_stock_sha256 != args.stock_sha256:
        raise SystemExit("frozen W8 stock SHA-256 mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "orchestrator").mkdir(parents=True, exist_ok=True)
    arms = tuple(dict.fromkeys(args.arm or W8_ARMS))
    lock = Lock()
    status_path = output_root / "w8-orchestrator-status.json"
    state: dict[str, Any] = {
        "schema_version": "retrostar190_w8_orchestrator_status.v1",
        "started_at": _utc_now(),
        "output_root": str(output_root),
        "manifest": str(args.manifest.expanduser().resolve()),
        "stock_index": str(stock_index),
        "stock_sha256": actual_stock_sha256,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "execution_profile": args.execution_profile,
        "panel_workers": args.panel_workers,
        "parallel_arms": min(args.parallel_arms, len(arms)),
        "fixed_cutoff": {
            "wall_time_s": args.fixed_cutoff_wall_time_s,
            "settled_task_count": args.fixed_cutoff_total_tasks,
        },
        "max_targets": args.max_targets,
        "arms": {arm: {"status": "queued"} for arm in arms},
        "semantics": {
            "same_target_manifest_for_every_arm": True,
            "same_stock_for_every_arm": True,
            "same_case_budget_for_every_arm": True,
            "same_target_cutoff_for_every_arm": True,
            "target_grouping_or_target_specific_tuning": False,
            "arm_roots_are_isolated": True,
        },
    }
    _write_json(status_path, state)

    def run_arm(arm: str) -> tuple[str, int]:
        arm_root = output_root / arm
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_v4_blind_panel.py"),
            "--manifest",
            str(args.manifest.expanduser().resolve()),
            "--output-root",
            str(arm_root),
            "--model",
            args.model,
            "--reasoning-effort",
            args.reasoning_effort,
            "--execution-profile",
            args.execution_profile,
            "--workers",
            str(args.panel_workers),
            "--ablation",
            arm,
            "--benchmark-stock-index",
            str(stock_index),
            "--benchmark-stock-name",
            args.stock_name,
            "--chemenzy-env-prefix",
            args.chemenzy_env_prefix,
            "--fixed-cutoff-wall-time-s",
            str(args.fixed_cutoff_wall_time_s),
            "--fixed-cutoff-total-tasks",
            str(args.fixed_cutoff_total_tasks),
        ]
        if args.max_targets is not None:
            command.extend(["--max-targets", str(args.max_targets)])
        if args.preflight_only:
            command.append("--preflight-only")
        if args.resume and (arm_root / "snapshots" / "benchmark-snapshot.json").is_file():
            command.append("--resume")
        log_path = output_root / "orchestrator" / f"{arm}.log"
        with lock:
            state["arms"][arm] = {
                "status": "running",
                "started_at": _utc_now(),
                "panel_root": str(arm_root),
                "log_path": str(log_path),
                "command": command,
            }
            _write_json(status_path, state)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        with lock:
            state["arms"][arm].update(
                {
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "finished_at": _utc_now(),
                }
            )
            _write_json(status_path, state)
        return arm, completed.returncode

    returncodes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(args.parallel_arms, len(arms))) as executor:
        futures = {executor.submit(run_arm, arm): arm for arm in arms}
        for future in as_completed(futures):
            arm, returncode = future.result()
            returncodes[arm] = returncode
    state["finished_at"] = _utc_now()
    state["complete"] = all(returncodes.get(arm) == 0 for arm in arms)
    _write_json(status_path, state)
    return 0 if state["complete"] else 2


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
