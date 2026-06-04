#!/usr/bin/env python3
"""Run the formal BioNav-v2 EC-conditioned OpenNMT training pipeline."""
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = Path("results/shared/bionav_v2_enzyme_corpus_20260529")
DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_BASE_CHECKPOINT = (
    DEFAULT_VENDOR_ROOT / "retro_planner/packages/onmt/checkpoints/np-like/model_step_100000.pt"
)
DEFAULT_OUTPUT_DIR = Path("results/shared/bionav_v2_formal_ec_context_20260529")


def run_pipeline(
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gpu: int = 0,
    train_steps: int = 50000,
    save_checkpoint_steps: int = 5000,
    valid_steps: int = 5000,
    batch_size_tokens: int = 4096,
    valid_batch_size: int = 32,
    accum_count: int = 4,
    learning_rate: float = 5e-4,
    decay_method: str = "none",
    warmup_steps: int = 8000,
    max_grad_norm: float = 1.0,
    src_seq_length: int = 768,
    tgt_seq_length: int = 768,
    num_threads: int = 8,
    seed: int = 20260529,
    evaluate_after_train: bool = True,
    benchmark_topk: int = 10,
    benchmark_batch_size: int = 64,
    force_preprocess: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    data_prefix = output_dir / "onmt" / "ec_context"
    data_prefix.parent.mkdir(parents=True, exist_ok=True)
    extended_checkpoint = checkpoints_dir / "bionav_v2_ec_context_vocab_extended.pt"
    save_model_prefix = checkpoints_dir / "bionav_v2_ec_context"

    manifest: dict[str, Any] = {
        "schema_version": "bionav_v2_formal_training.v1",
        "created_at": _now_utc(),
        "output_dir": str(output_dir),
        "corpus_dir": str(corpus_dir),
        "vendor_root": str(vendor_root),
        "base_checkpoint": str(base_checkpoint),
        "extended_checkpoint": str(extended_checkpoint),
        "save_model_prefix": str(save_model_prefix),
        "settings": {
            "gpu": gpu,
            "train_steps": train_steps,
            "save_checkpoint_steps": save_checkpoint_steps,
            "valid_steps": valid_steps,
            "batch_size_tokens": batch_size_tokens,
            "valid_batch_size": valid_batch_size,
            "accum_count": accum_count,
            "learning_rate": learning_rate,
            "decay_method": decay_method,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "src_seq_length": src_seq_length,
            "tgt_seq_length": tgt_seq_length,
            "num_threads": num_threads,
            "seed": seed,
            "architecture": [
                "ChemEnzy BioNav Transformer checkpoint initialization",
                "shared vocabulary extended with EC/context tokens",
                "EC1 + full-EC conditioned source sequence",
                "full local enzyme corpus train/valid split",
                "locked held-out enzyme benchmark for exact reactant-side recall",
            ],
        },
        "commands": {},
        "artifacts": {},
        "steps": [],
    }

    started = time.monotonic()
    extend_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "extend_chem_enzy_onmt_context_vocab.py"),
        "--checkpoint",
        str(base_checkpoint),
        "--corpus-dir",
        str(corpus_dir),
        "--output-checkpoint",
        str(extended_checkpoint),
        "--vendor-root",
        str(vendor_root),
        "--mode",
        "ec_context",
        "--side",
        "src",
        "--side",
        "tgt",
        "--split",
        "train",
        "--split",
        "valid",
        "--split",
        "test",
        "--report",
        str(output_dir / "vocab_extension_report.json"),
        "--markdown",
        str(output_dir / "vocab_extension_report.md"),
    ]
    _run_step("extend_vocab", extend_cmd, output_dir=output_dir, manifest=manifest)

    train_pt = Path(str(data_prefix) + ".train.0.pt")
    valid_pt = Path(str(data_prefix) + ".valid.0.pt")
    if force_preprocess or not train_pt.exists() or not valid_pt.exists():
        preprocess_cmd = [
            sys.executable,
            str(vendor_root / "retro_planner/packages/onmt/onmt/bin/preprocess.py"),
            "-train_src",
            str(corpus_dir / "ec_context.train.src"),
            "-train_tgt",
            str(corpus_dir / "ec_context.train.tgt"),
            "-valid_src",
            str(corpus_dir / "ec_context.valid.src"),
            "-valid_tgt",
            str(corpus_dir / "ec_context.valid.tgt"),
            "-save_data",
            str(data_prefix),
            "-share_vocab",
            "-src_seq_length",
            str(src_seq_length),
            "-tgt_seq_length",
            str(tgt_seq_length),
            "-filter_valid",
            "-num_threads",
            str(num_threads),
            "-overwrite",
        ]
        _run_step("preprocess", preprocess_cmd, output_dir=output_dir, manifest=manifest, vendor_root=vendor_root)
    else:
        manifest["steps"].append({"name": "preprocess", "status": "skipped_existing"})

    train_cmd = [
        sys.executable,
        str(vendor_root / "retro_planner/packages/onmt/onmt/bin/train.py"),
        "-data",
        str(data_prefix),
        "-save_model",
        str(save_model_prefix),
        "-train_from",
        str(extended_checkpoint),
        "-reset_optim",
        "all",
        "-train_steps",
        str(train_steps),
        "-save_checkpoint_steps",
        str(save_checkpoint_steps),
        "-keep_checkpoint",
        "8",
        "-valid_steps",
        str(valid_steps),
        "-report_every",
        "200",
        "-gpu_ranks",
        str(gpu),
        "-world_size",
        "1",
        "-batch_type",
        "tokens",
        "-batch_size",
        str(batch_size_tokens),
        "-valid_batch_size",
        str(valid_batch_size),
        "-normalization",
        "tokens",
        "-accum_count",
        str(accum_count),
        "-optim",
        "adam",
        "-adam_beta2",
        "0.998",
        "-decay_method",
        decay_method,
        "-warmup_steps",
        str(warmup_steps),
        "-learning_rate",
        str(learning_rate),
        "-max_grad_norm",
        str(max_grad_norm),
        "-dropout",
        "0.1",
        "-attention_dropout",
        "0.1",
        "-seed",
        str(seed),
        "-log_file",
        str(output_dir / "train.log"),
    ]
    _run_step("train", train_cmd, output_dir=output_dir, manifest=manifest, vendor_root=vendor_root)

    final_checkpoint = _latest_checkpoint(save_model_prefix)
    manifest["artifacts"]["latest_checkpoint"] = str(final_checkpoint) if final_checkpoint else ""
    if evaluate_after_train and final_checkpoint is not None:
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_chem_enzy_native_bionav.py"),
            "--src",
            str(corpus_dir / "benchmark/native_bionav_benchmark.ec_context.src"),
            "--tgt",
            str(corpus_dir / "benchmark/native_bionav_benchmark.tgt"),
            "--meta",
            str(corpus_dir / "benchmark/native_bionav_benchmark.meta.jsonl"),
            "--output-dir",
            str(output_dir / "benchmark_ec_context"),
            "--gpu",
            str(gpu),
            "--topk",
            str(benchmark_topk),
            "--engine",
            "batch",
            "--batch-size",
            str(benchmark_batch_size),
            "--tokenizer",
            "pretokenized",
            "--inference-input",
            "source",
            "--model",
            str(final_checkpoint),
        ]
        _run_step("evaluate_ec_context", eval_cmd, output_dir=output_dir, manifest=manifest)

    manifest["elapsed_s"] = round(time.monotonic() - started, 3)
    _write_manifest(manifest, output_dir)
    return manifest


def _run_step(
    name: str,
    cmd: list[str],
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    vendor_root: Path | None = None,
) -> None:
    env = dict(os.environ)
    if vendor_root is not None:
        onmt_root = Path(vendor_root).resolve() / "retro_planner/packages/onmt"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(onmt_root) if not existing else f"{onmt_root}{os.pathsep}{existing}"
    log_path = output_dir / f"{name}.cmd.log"
    started = time.monotonic()
    manifest["commands"][name] = " ".join(cmd)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    row = {
        "name": name,
        "returncode": proc.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        "log": str(log_path),
    }
    manifest["steps"].append(row)
    _write_manifest(manifest, output_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {proc.returncode}; see {log_path}")


def _latest_checkpoint(prefix: Path) -> Path | None:
    candidates = sorted(prefix.parent.glob(prefix.name + "_step_*.pt"), key=_checkpoint_step)
    return candidates[-1] if candidates else None


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if "_step_" not in stem:
        return -1
    try:
        return int(stem.rsplit("_step_", 1)[1])
    except ValueError:
        return -1


def _write_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    manifest_path = output_dir / "formal_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# BioNav-v2 Formal EC-Context Training",
        "",
        f"created_at: {manifest.get('created_at')}",
        f"output_dir: `{manifest.get('output_dir')}`",
        "",
        "## Steps",
        "",
        "| step | status | elapsed_s | log |",
        "| --- | ---: | ---: | --- |",
    ]
    for step in manifest.get("steps", []):
        status = step.get("status", step.get("returncode"))
        lines.append(f"| {step.get('name')} | {status} | {step.get('elapsed_s', '')} | `{step.get('log', '')}` |")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- latest_checkpoint: `{manifest.get('artifacts', {}).get('latest_checkpoint', '')}`",
        "",
    ])
    (output_dir / "formal_training_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--save-checkpoint-steps", type=int, default=5000)
    parser.add_argument("--valid-steps", type=int, default=5000)
    parser.add_argument("--batch-size-tokens", type=int, default=4096)
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--accum-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--decay-method", choices=["none", "noam", "noamwd", "rsqrt"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--src-seq-length", type=int, default=768)
    parser.add_argument("--tgt-seq-length", type=int, default=768)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--benchmark-topk", type=int, default=10)
    parser.add_argument("--benchmark-batch-size", type=int, default=64)
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_pipeline(
        corpus_dir=args.corpus_dir,
        vendor_root=args.vendor_root,
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        gpu=args.gpu,
        train_steps=args.train_steps,
        save_checkpoint_steps=args.save_checkpoint_steps,
        valid_steps=args.valid_steps,
        batch_size_tokens=args.batch_size_tokens,
        valid_batch_size=args.valid_batch_size,
        accum_count=args.accum_count,
        learning_rate=args.learning_rate,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        src_seq_length=args.src_seq_length,
        tgt_seq_length=args.tgt_seq_length,
        num_threads=args.num_threads,
        seed=args.seed,
        evaluate_after_train=not args.skip_eval,
        benchmark_topk=args.benchmark_topk,
        benchmark_batch_size=args.benchmark_batch_size,
        force_preprocess=args.force_preprocess,
    )
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "latest_checkpoint": manifest.get("artifacts", {}).get("latest_checkpoint", ""),
        "elapsed_s": manifest.get("elapsed_s"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
