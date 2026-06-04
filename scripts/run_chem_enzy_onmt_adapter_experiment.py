#!/usr/bin/env python3
"""Run or dry-run a reproducible ChemEnzy OpenNMT adapter experiment.

The default mode is intentionally non-executing. It writes an auditable
manifest with the exact preprocess/train/evaluate commands needed to compare a
supervised ChemEnzy adapter checkpoint against the native checkpoint. Use
``--execute`` only when the ChemEnzy runtime is available and a real training
run is intended.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "chem_enzy_onmt_adapter_experiment.v1"
DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_CORPUS_DIR = Path("results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k")
DEFAULT_OUTPUT_DIR = Path("results/shared/chem_enzy_adapter_mainline_20260521/plain_runner_low_lr")
DEFAULT_RUNTIME_PYTHON = Path("/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env/bin/python")


def main() -> None:
    args = _parse_args()
    vendor_root = args.vendor_root
    base_checkpoint = args.base_checkpoint or (
        vendor_root / "retro_planner" / "packages" / "onmt" / "checkpoints" / "np-like" / "model_step_100000.pt"
    )
    runtime_python = args.runtime_python or _default_runtime_python()
    plan = build_experiment_plan(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        vendor_root=vendor_root,
        runtime_python=runtime_python,
        base_checkpoint=base_checkpoint,
        mode=args.mode,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        valid_steps=args.valid_steps,
        save_checkpoint_steps=args.save_checkpoint_steps,
        keep_checkpoint=args.keep_checkpoint,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        beam_size=args.beam_size,
        topk=args.topk,
        translate_tokenizer=args.translate_tokenizer,
        device=args.device,
        gpuid=args.gpuid,
        eval_splits=args.eval_split,
        eval_limit=args.eval_limit,
        save_data_name=args.save_data_name,
        save_model_name=args.save_model_name,
        src_seq_length=args.src_seq_length,
        tgt_seq_length=args.tgt_seq_length,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
    )
    if args.execute:
        execute_plan(plan, keep_going=args.keep_going)
    else:
        plan["status"] = "planned_not_executed"
    collect_outputs(plan)
    write_manifest(plan)
    print(json.dumps(plan["summary"], indent=2, ensure_ascii=False))


def build_experiment_plan(
    *,
    corpus_dir: Path,
    output_dir: Path,
    vendor_root: Path,
    runtime_python: Path,
    base_checkpoint: Path,
    mode: str = "plain",
    train_steps: int = 2,
    learning_rate: float = 0.0001,
    valid_steps: int | None = None,
    save_checkpoint_steps: int | None = None,
    keep_checkpoint: int = 1,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    beam_size: int = 5,
    topk: int = 5,
    translate_tokenizer: str = "char",
    device: int = 0,
    gpuid: int | None = None,
    eval_splits: list[str] | tuple[str, ...] = ("valid",),
    eval_limit: int | None = 20,
    save_data_name: str = "plain_onmt",
    save_model_name: str = "plain_cascade_adapter_low_lr",
    src_seq_length: int | None = None,
    tgt_seq_length: int | None = None,
    skip_preprocess: bool = False,
    skip_train: bool = False,
) -> dict[str, Any]:
    if mode not in {"plain", "context"}:
        raise ValueError("mode must be 'plain' or 'context'")
    if translate_tokenizer not in {"char", "token", "pretokenized"}:
        raise ValueError(f"unsupported translate_tokenizer: {translate_tokenizer}")
    if mode == "context" and translate_tokenizer != "pretokenized":
        raise ValueError("context mode evaluation requires --translate-tokenizer pretokenized")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if keep_checkpoint <= 0:
        raise ValueError("keep_checkpoint must be positive")
    eval_splits = _normalize_eval_splits(eval_splits)

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    vendor_root = Path(vendor_root)
    runtime_python = Path(runtime_python)
    base_checkpoint = Path(base_checkpoint)

    preprocess_py = vendor_root / "retro_planner" / "packages" / "onmt" / "onmt" / "bin" / "preprocess.py"
    train_py = vendor_root / "retro_planner" / "packages" / "onmt" / "onmt" / "bin" / "train.py"
    eval_py = REPO_ROOT / "scripts" / "evaluate_chem_enzy_onmt_checkpoint_exact.py"
    save_data = output_dir / save_data_name
    save_model_prefix = output_dir / save_model_name
    adapter_checkpoint = Path(f"{save_model_prefix}_step_{train_steps}.pt")
    logs_dir = output_dir / "logs"

    valid_steps = int(valid_steps if valid_steps is not None else train_steps)
    save_checkpoint_steps = int(save_checkpoint_steps if save_checkpoint_steps is not None else train_steps)

    commands: list[dict[str, Any]] = []
    preprocess_argv: list[Any] = [
        runtime_python,
        preprocess_py,
        "-train_src",
        corpus_dir / f"{mode}.train.src",
        "-train_tgt",
        corpus_dir / f"{mode}.train.tgt",
        "-valid_src",
        corpus_dir / f"{mode}.valid.src",
        "-valid_tgt",
        corpus_dir / f"{mode}.valid.tgt",
        "-save_data",
        save_data,
        "-overwrite",
    ]
    if src_seq_length is not None:
        preprocess_argv.extend(["-src_seq_length", int(src_seq_length)])
    if tgt_seq_length is not None:
        preprocess_argv.extend(["-tgt_seq_length", int(tgt_seq_length)])

    if not skip_preprocess:
        commands.append(_command_row(
            label="preprocess",
            argv=preprocess_argv,
            log_path=logs_dir / "preprocess.log",
            required_outputs=[
                Path(f"{save_data}.train.0.pt"),
                Path(f"{save_data}.valid.0.pt"),
                Path(f"{save_data}.vocab.pt"),
            ],
        ))

    train_argv: list[Any] = [
        runtime_python,
        train_py,
        "-data",
        save_data,
        "-train_from",
        base_checkpoint,
        "-reset_optim",
        "all",
        "-save_model",
        save_model_prefix,
        "-train_steps",
        train_steps,
        "-valid_steps",
        valid_steps,
        "-save_checkpoint_steps",
        save_checkpoint_steps,
        "-keep_checkpoint",
        keep_checkpoint,
        "-batch_size",
        batch_size,
        "-learning_rate",
        learning_rate,
        "-report_every",
        1,
    ]
    if gpuid is not None:
        train_argv.extend(["-world_size", 1, "-gpu_ranks", gpuid])
    if not skip_train:
        commands.append(_command_row(
            label="train",
            argv=train_argv,
            log_path=logs_dir / "train.log",
            required_outputs=[adapter_checkpoint],
        ))

    eval_commands = []
    for split in eval_splits:
        split = str(split)
        if split not in {"valid", "test", "train"}:
            raise ValueError(f"unsupported eval split: {split}")
        limit_suffix = f"{eval_limit}" if eval_limit is not None else "full"
        output = output_dir / f"exact_recall_{split}{limit_suffix}.json"
        summary_output = output_dir / f"exact_recall_{split}{limit_suffix}_summary.json"
        row = _command_row(
            label=f"eval_{split}",
            argv=[
                runtime_python,
                eval_py,
                "--model",
                base_checkpoint,
                "--model",
                adapter_checkpoint,
                "--src",
                corpus_dir / f"{mode}.{split}.src",
                "--tgt",
                corpus_dir / f"{mode}.{split}.tgt",
                "--vendor-root",
                vendor_root,
                "--tokenizer",
                translate_tokenizer,
                "--beam-size",
                beam_size,
                "--topk",
                topk,
                "--batch-size",
                eval_batch_size,
                "--device",
                device,
                "--output",
                output,
                "--summary-output",
                summary_output,
            ],
            log_path=logs_dir / f"eval_{split}.log",
            required_outputs=[output, summary_output],
        )
        if eval_limit is not None:
            row["argv"].extend(["--limit", str(eval_limit)])
            row["cmd"] = _shell_join(row["argv"])
        commands.append(row)
        eval_commands.append(row)

    input_paths = {
        "runtime_python": runtime_python,
        "vendor_root": vendor_root,
        "preprocess_py": preprocess_py,
        "train_py": train_py,
        "eval_py": eval_py,
        "base_checkpoint": base_checkpoint,
        "train_src": corpus_dir / f"{mode}.train.src",
        "train_tgt": corpus_dir / f"{mode}.train.tgt",
        "valid_src": corpus_dir / f"{mode}.valid.src",
        "valid_tgt": corpus_dir / f"{mode}.valid.tgt",
    }
    if "test" in set(eval_splits):
        input_paths["test_src"] = corpus_dir / f"{mode}.test.src"
        input_paths["test_tgt"] = corpus_dir / f"{mode}.test.tgt"

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "planned",
        "repo_root": str(REPO_ROOT),
        "mode": mode,
        "output_dir": str(output_dir),
        "inputs": {name: {"path": str(path), "exists": Path(path).exists()} for name, path in input_paths.items()},
        "settings": {
            "train_steps": train_steps,
            "learning_rate": learning_rate,
            "valid_steps": valid_steps,
            "save_checkpoint_steps": save_checkpoint_steps,
            "keep_checkpoint": keep_checkpoint,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "beam_size": beam_size,
            "topk": topk,
            "translate_tokenizer": translate_tokenizer,
            "device": device,
            "gpuid": gpuid,
            "eval_splits": list(eval_splits),
            "eval_limit": eval_limit,
            "src_seq_length": src_seq_length,
            "tgt_seq_length": tgt_seq_length,
            "skip_preprocess": skip_preprocess,
            "skip_train": skip_train,
        },
        "artifacts": {
            "save_data_prefix": str(save_data),
            "save_model_prefix": str(save_model_prefix),
            "adapter_checkpoint": str(adapter_checkpoint),
            "manifest_json": str(output_dir / "experiment_manifest.json"),
            "manifest_markdown": str(output_dir / "experiment_manifest.md"),
        },
        "commands": commands,
        "evaluation_commands": [row["label"] for row in eval_commands],
        "promotion_contract": (
            "This experiment runner can produce a supervised ChemEnzy ONMT adapter checkpoint and exact-recall "
            "comparisons. It does not promote the adapter. Promotion requires full valid/test comparison, "
            "route-level proposal recall checks, and no regression against the native checkpoint."
        ),
    }
    plan["summary"] = _summary(plan)
    return plan


def execute_plan(plan: dict[str, Any], *, keep_going: bool = False) -> None:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "completed"
    for command in plan["commands"]:
        log_path = Path(command["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {command['cmd']}\n\n")
            proc = subprocess.run(
                command["argv"],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        command["run"] = {
            "returncode": proc.returncode,
            "elapsed_s": round(time.monotonic() - started, 3),
            "completed_at": _utc_now(),
        }
        if proc.returncode != 0:
            status = "failed"
            if not keep_going:
                break
    plan["status"] = status


def collect_outputs(plan: dict[str, Any]) -> None:
    output_checks = []
    for command in plan["commands"]:
        for path in command.get("required_outputs") or []:
            output_checks.append({"command": command["label"], "path": path, "exists": Path(path).exists()})
    plan["output_checks"] = output_checks
    eval_summaries = {}
    for command in plan["commands"]:
        if not command["label"].startswith("eval_"):
            continue
        for path in command.get("required_outputs") or []:
            path_obj = Path(path)
            if path_obj.name.endswith("_summary.json") and path_obj.exists():
                eval_summaries[command["label"]] = json.loads(path_obj.read_text(encoding="utf-8"))
    plan["evaluation_summaries"] = eval_summaries
    plan["adapter_ab_summary"] = _adapter_ab_summary(eval_summaries)
    plan["summary"] = _summary(plan)


def write_manifest(plan: dict[str, Any]) -> None:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(plan["artifacts"]["manifest_json"])
    md_path = Path(plan["artifacts"]["manifest_markdown"])
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# ChemEnzy ONMT Adapter Experiment",
        "",
        f"生成时间：{plan['created_at']}",
        "",
        "## Status",
        "",
        f"- status: `{plan['status']}`",
        f"- output_dir: `{plan['output_dir']}`",
        f"- train_steps: {plan['settings']['train_steps']}",
        f"- learning_rate: {plan['settings']['learning_rate']}",
        f"- eval_splits: {', '.join(plan['settings']['eval_splits'])}",
        f"- eval_limit: {plan['settings']['eval_limit']}",
        "",
        "## Contract",
        "",
        plan["promotion_contract"],
        "",
        "## Input Checks",
        "",
        "| input | exists | path |",
        "| --- | ---: | --- |",
    ]
    for name, row in plan["inputs"].items():
        lines.append(f"| {name} | {row['exists']} | `{row['path']}` |")
    lines.extend([
        "",
        "## Commands",
        "",
        "| label | returncode | log | command |",
        "| --- | ---: | --- | --- |",
    ])
    for command in plan["commands"]:
        returncode = (command.get("run") or {}).get("returncode", "")
        lines.append(
            f"| {command['label']} | {returncode} | `{command['log_path']}` | `{command['cmd']}` |"
        )
    lines.extend([
        "",
        "## Output Checks",
        "",
        f"- expected_outputs: {summary['expected_outputs']}",
        f"- existing_outputs: {summary['existing_outputs']}",
        "",
    ])
    eval_summaries = plan.get("evaluation_summaries") or {}
    if eval_summaries:
        lines.extend([
            "## Evaluation Summaries",
            "",
            "| eval | model | n | nonempty | top1 | topk | top1_rate | topk_rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, rows in eval_summaries.items():
            for row in rows:
                topk_key = next((key for key in row if key.startswith("top") and key.endswith("_exact") and key != "top1_exact"), None)
                topk_rate_key = next((key for key in row if key.startswith("top") and key.endswith("_rate") and key != "top1_rate"), None)
                lines.append(
                    f"| {label} | `{row.get('model_path')}` | {row.get('n_examples')} | {row.get('nonempty')} | "
                    f"{row.get('top1_exact')} | {row.get(topk_key) if topk_key else ''} | "
                    f"{row.get('top1_rate')} | {row.get(topk_rate_key) if topk_rate_key else ''} |"
                )
        lines.append("")
    ab_summary = plan.get("adapter_ab_summary") or {}
    if ab_summary:
        lines.extend([
            "## Adapter A/B Summary",
            "",
            f"- decision: `{ab_summary.get('decision')}`",
            f"- promotion_ready: `{ab_summary.get('promotion_ready')}`",
            "",
            "| eval | n | top1 delta | topK metric | topK delta | adapter nonempty |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ])
        for row in ab_summary.get("rows") or []:
            lines.append(
                f"| {row['eval']} | {row['n_examples']} | {row['top1_delta']} | `{row.get('topk_metric')}` | {row['topk_delta']} | {row['adapter_nonempty']} |"
            )
        lines.extend(["", ab_summary.get("contract") or "", ""])
    lines.extend([
        "## Promotion Gate",
        "",
        "This is not a promotion artifact by itself. Keep any generated adapter sidecar-only until it beats the native checkpoint on full valid/test and route-level proposal smoke.",
        "",
    ])
    return "\n".join(lines)


def _command_row(*, label: str, argv: list[Any], log_path: Path, required_outputs: list[Path]) -> dict[str, Any]:
    argv_str = [str(item) for item in argv]
    return {
        "label": label,
        "argv": argv_str,
        "cmd": _shell_join(argv_str),
        "log_path": str(log_path),
        "required_outputs": [str(path) for path in required_outputs],
    }


def _summary(plan: dict[str, Any]) -> dict[str, Any]:
    outputs = plan.get("output_checks") or []
    ab_summary = plan.get("adapter_ab_summary") or {}
    return {
        "schema_version": plan["schema_version"],
        "status": plan["status"],
        "output_dir": plan["output_dir"],
        "command_count": len(plan.get("commands") or []),
        "expected_outputs": len(outputs),
        "existing_outputs": sum(1 for row in outputs if row.get("exists")),
        "adapter_checkpoint": plan["artifacts"]["adapter_checkpoint"],
        "adapter_ab_decision": ab_summary.get("decision"),
        "adapter_ab_split_count": ab_summary.get("split_count"),
        "promotion_ready": bool(ab_summary.get("promotion_ready", False)),
    }


def _adapter_ab_summary(eval_summaries: dict[str, Any]) -> dict[str, Any]:
    rows = []
    any_regression = False
    any_lift = False
    all_nonempty = True
    for label, summary_rows in sorted((eval_summaries or {}).items()):
        if not isinstance(summary_rows, list) or len(summary_rows) < 2:
            continue
        native = summary_rows[0]
        adapter = summary_rows[1]
        topk_key = _topk_exact_key(native, adapter)
        topk_rate_key = topk_key.replace("_exact", "_rate") if topk_key else None
        native_top1 = int(native.get("top1_exact") or 0)
        adapter_top1 = int(adapter.get("top1_exact") or 0)
        native_topk = int(native.get(topk_key) or 0) if topk_key else 0
        adapter_topk = int(adapter.get(topk_key) or 0) if topk_key else 0
        top1_delta = adapter_top1 - native_top1
        topk_delta = adapter_topk - native_topk
        if top1_delta < 0 or topk_delta < 0:
            any_regression = True
        if top1_delta > 0 or topk_delta > 0:
            any_lift = True
        if int(adapter.get("nonempty") or 0) <= 0:
            all_nonempty = False
        rows.append(
            {
                "eval": label,
                "n_examples": int(adapter.get("n_examples") or native.get("n_examples") or 0),
                "native_model": native.get("model_path"),
                "adapter_model": adapter.get("model_path"),
                "native_top1_exact": native_top1,
                "adapter_top1_exact": adapter_top1,
                "top1_delta": top1_delta,
                "topk_metric": topk_key,
                "native_topk_exact": native_topk,
                "adapter_topk_exact": adapter_topk,
                "topk_delta": topk_delta,
                "native_top1_rate": native.get("top1_rate"),
                "adapter_top1_rate": adapter.get("top1_rate"),
                "native_topk_rate": native.get(topk_rate_key) if topk_rate_key else None,
                "adapter_topk_rate": adapter.get(topk_rate_key) if topk_rate_key else None,
                "adapter_nonempty": int(adapter.get("nonempty") or 0),
            }
        )
    if not rows:
        decision = "not_evaluated"
    elif any_regression:
        decision = "hold_due_to_exact_recall_regression"
    elif any_lift and all_nonempty:
        decision = "candidate_for_larger_route_level_ab"
    else:
        decision = "hold_no_exact_recall_lift"
    return {
        "schema_version": "chem_enzy_onmt_adapter_ab_summary.v1",
        "split_count": len(rows),
        "promotion_ready": False,
        "decision": decision,
        "rows": rows,
        "contract": (
            "Exact-recall A/B summary only. Promotion still requires full valid/test evaluation "
            "and route-level proposal checks against the native checkpoint."
        ),
    }


def _topk_exact_key(*rows: dict[str, Any]) -> str | None:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row if str(key).startswith("top") and str(key).endswith("_exact") and key != "top1_exact")
    return sorted(keys)[-1] if keys else None


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_runtime_python() -> Path:
    env_prefix = os.environ.get("CHEMENZY_ENV_PREFIX")
    if env_prefix:
        candidate = Path(env_prefix) / "bin" / "python"
        if candidate.exists():
            return candidate
    if DEFAULT_RUNTIME_PYTHON.exists():
        return DEFAULT_RUNTIME_PYTHON
    return Path(sys.executable)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--mode", choices=["plain", "context"], default="plain")
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--valid-steps", type=int)
    parser.add_argument("--save-checkpoint-steps", type=int)
    parser.add_argument("--keep-checkpoint", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--translate-tokenizer", choices=["char", "token", "pretokenized"], default="char")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpuid", type=int)
    parser.add_argument("--eval-split", action="append", choices=["train", "valid", "test"])
    parser.add_argument("--eval-limit", type=int, default=20, help="Use -1 for full split evaluation.")
    parser.add_argument("--save-data-name", default="plain_onmt")
    parser.add_argument("--save-model-name", default="plain_cascade_adapter_low_lr")
    parser.add_argument("--src-seq-length", type=int)
    parser.add_argument("--tgt-seq-length", type=int)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run preprocess/train/eval commands.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed command when --execute is used.")
    args = parser.parse_args()
    if args.eval_limit is not None and args.eval_limit < 0:
        args.eval_limit = None
    return args


def _normalize_eval_splits(eval_splits: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    for split in eval_splits or ("valid",):
        split = str(split)
        if split not in {"valid", "test", "train"}:
            raise ValueError(f"unsupported eval split: {split}")
        if split not in normalized:
            normalized.append(split)
    return normalized


if __name__ == "__main__":
    main()
