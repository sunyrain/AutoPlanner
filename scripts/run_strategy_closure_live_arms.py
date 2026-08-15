#!/usr/bin/env python3
"""Run protocol-bound clean strategy-closure live arms."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.strategy_closure_live import (  # noqa: E402
    LIVE_ARM_ABLATIONS,
    bind_live_execution,
    live_arm_command,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="benchmarks/synthatlas_strategy_closure_clean20.protocol.json",
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/synthatlas_strategy_closure_clean20.v1.json",
    )
    parser.add_argument(
        "--evaluator-pack",
        default=("data_external/synthatlas/strategy_closure_clean20_20260812/evaluator_pack.json"),
    )
    parser.add_argument(
        "--leakage-pack",
        default=(
            "data_external/synthatlas/strategy_closure_clean20_20260812/"
            "blind_leakage_audit_pack.json"
        ),
    )
    parser.add_argument(
        "--stock-index",
        default="data_external/retrostar190/retrostar_emolecules_stock.sqlite3",
    )
    parser.add_argument("--stock-name", default="Retro*-190 eMolecules ~23M")
    parser.add_argument(
        "--output-root",
        default="results/shared/synthatlas_strategy_closure_clean20_live",
    )
    parser.add_argument(
        "--receipt",
        default=("results/shared/synthatlas_strategy_closure_clean20_live/execution-receipt.json"),
    )
    parser.add_argument("--arm", action="append", choices=tuple(LIVE_ARM_ABLATIONS))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument(
        "--execution-profile", choices=("fast", "standard", "proof"), default="standard"
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--parallel-arms", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--fixed-cutoff-total-tasks", type=int, default=192)
    parser.add_argument("--chemenzy-env-prefix", default="D:/conda/envs/py312")
    parser.add_argument(
        "--host-python",
        default=shutil.which("python") or sys.executable,
        help="Host AutoPlanner Python; must remain distinct from the ChemEnzy env.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bind-only", action="store_true")
    args = parser.parse_args(argv)

    arms = args.arm or list(LIVE_ARM_ABLATIONS)
    execution = bind_live_execution(
        protocol_path=_path(args.protocol),
        manifest_path=_path(args.manifest),
        evaluator_pack_path=_path(args.evaluator_pack),
        leakage_pack_path=_path(args.leakage_pack),
        stock_index_path=_path(args.stock_index),
        output_root=_path(args.output_root),
        arms=arms,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        execution_profile=args.execution_profile,
        workers=args.workers,
        fixed_cutoff_total_tasks=args.fixed_cutoff_total_tasks,
        host_python_executable=args.host_python,
        chemenzy_env_prefix=args.chemenzy_env_prefix,
    )
    receipt = _path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_receipt(receipt, execution)
    commands = {
        arm: live_arm_command(
            execution,
            arm_id=arm,
            python_executable=args.host_python,
            runner_script=ROOT / "scripts" / "run_v4_blind_panel.py",
            chemenzy_env_prefix=args.chemenzy_env_prefix,
            benchmark_stock_name=args.stock_name,
            preflight_only=args.preflight_only,
            resume=args.resume,
        )
        for arm in arms
    }
    print(
        json.dumps(
            {
                "execution_receipt": str(receipt),
                "content_sha256": execution["content_sha256"],
                "commands": commands,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.bind_only:
        return 0

    def run(arm_id: str) -> tuple[str, int]:
        completed = subprocess.run(commands[arm_id], cwd=ROOT, check=False)
        return arm_id, completed.returncode

    codes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.parallel_arms) as executor:
        futures = {executor.submit(run, arm): arm for arm in arms}
        for future in as_completed(futures):
            arm, code = future.result()
            codes[arm] = code
            print(json.dumps({"arm": arm, "returncode": code}))
    return 0 if all(code == 0 for code in codes.values()) else 2


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _write_immutable_receipt(path: Path, execution: dict) -> None:
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("strategy_closure_existing_receipt_invalid") from exc
        if current != execution:
            raise RuntimeError("strategy_closure_existing_receipt_mismatch")
        return
    path.write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
