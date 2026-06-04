#!/usr/bin/env python3
"""Run locked benchmark for BioNav EC-context ablation sources."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_DIR = Path("results/shared/bionav_v2_ec_context_ablation_20260529/sources")
DEFAULT_CORPUS_DIR = Path("results/shared/bionav_v2_enzyme_corpus_20260529")
DEFAULT_CHECKPOINT = Path(
    "results/shared/bionav_v2_formal_ec_context_20260529_valid32/checkpoints/archive/"
    "bionav_v2_ec_context_step_15000_benchmarked.pt"
)
DEFAULT_OUTPUT_DIR = Path("results/shared/bionav_v2_ec_context_ablation_20260529/step15000")


VARIANT_ORDER = [
    "product_plain",
    "product_marker",
    "ec1_only",
    "ec1_shuffled",
    "full_ec_oracle",
    "full_ec_shuffled",
]


def run_ablation_benchmarks(
    *,
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gpu: int = 1,
    topk: int = 10,
    batch_size: int = 64,
    variants: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = variants or VARIANT_ORDER
    sources = _load_sources(sources_dir)
    manifest: dict[str, Any] = {
        "schema_version": "bionav_ec_ablation_benchmarks.v1",
        "created_at": _now_utc(),
        "checkpoint": str(checkpoint.resolve()),
        "sources_dir": str(sources_dir.resolve()),
        "corpus_dir": str(corpus_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "settings": {
            "gpu": gpu,
            "topk": topk,
            "batch_size": batch_size,
            "variants": variants,
        },
        "runs": [],
    }

    started = time.monotonic()
    for variant in variants:
        if variant not in sources:
            raise KeyError(f"unknown source variant: {variant}")
        variant_dir = output_dir / variant
        summary_path = variant_dir / "native_bionav_benchmark_summary.json"
        if summary_path.exists() and not force:
            manifest["runs"].append({"variant": variant, "status": "skipped_existing", "summary": str(summary_path)})
            continue
        variant_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_chem_enzy_native_bionav.py"),
            "--src",
            str(Path(sources[variant]).resolve()),
            "--tgt",
            str((corpus_dir / "benchmark/native_bionav_benchmark.tgt").resolve()),
            "--meta",
            str((corpus_dir / "benchmark/native_bionav_benchmark.meta.jsonl").resolve()),
            "--output-dir",
            str(variant_dir.resolve()),
            "--gpu",
            str(gpu),
            "--topk",
            str(topk),
            "--engine",
            "batch",
            "--batch-size",
            str(batch_size),
            "--tokenizer",
            "pretokenized",
            "--inference-input",
            "source",
            "--model",
            str(checkpoint.resolve()),
        ]
        log_path = output_dir / f"{variant}.cmd.log"
        run_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        row = {
            "variant": variant,
            "returncode": proc.returncode,
            "elapsed_s": round(time.monotonic() - run_started, 3),
            "log": str(log_path),
            "summary": str(summary_path),
            "command": " ".join(cmd),
        }
        manifest["runs"].append(row)
        _write_json(output_dir / "ablation_manifest.json", manifest)
        if proc.returncode != 0:
            raise RuntimeError(f"{variant} benchmark failed; see {log_path}")

    summary = _collect_summary(output_dir, variants)
    manifest["elapsed_s"] = round(time.monotonic() - started, 3)
    manifest["summary"] = summary
    _write_json(output_dir / "ablation_manifest.json", manifest)
    _write_summary(output_dir, summary)
    return manifest


def _load_sources(sources_dir: Path) -> dict[str, str]:
    manifest_path = sources_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {k: str((REPO_ROOT / v).resolve()) if not Path(v).is_absolute() else v for k, v in manifest["outputs"].items()}


def _collect_summary(output_dir: Path, variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        summary_path = output_dir / variant / "native_bionav_benchmark_summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        data = payload.get("summary", payload)
        rows.append(
            {
                "variant": variant,
                "n_examples": data["n_examples"],
                "nonempty_rate": data["nonempty_rate"],
                "top1_rate": data["top1_rate"],
                "top10_rate": data["top10_rate"],
                "top1_exact": data["top1_exact"],
                "top10_exact": data["top10_exact"],
                "elapsed_s": data["elapsed_s"],
                "ec1": data.get("ec1", {}),
            }
        )
    return rows


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(output_dir / "ablation_summary.json", rows)
    lines = [
        "# BioNav EC Context Ablation Benchmark",
        "",
        f"generated_at: {_now_utc()}",
        "",
        "## Overall",
        "",
        "| variant | examples | nonempty | top1 | top10 | exact top1/top10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {n_examples} | {nonempty_rate:.4f} | {top1_rate:.4f} | {top10_rate:.4f} | {top1_exact}/{top10_exact} |".format(
                **row
            )
        )
    lines.extend(["", "## EC1 top10", "", "| variant | EC1 | EC2 | EC3 | EC4 | EC5 | EC6 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        ec1 = row.get("ec1", {})
        vals = []
        for key in ["1", "2", "3", "4", "5", "6"]:
            vals.append(ec1.get(key, {}).get("top10_rate", 0.0))
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(row["variant"], *vals))
    (output_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = run_ablation_benchmarks(
        sources_dir=args.sources_dir,
        corpus_dir=args.corpus_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        gpu=args.gpu,
        topk=args.topk,
        batch_size=args.batch_size,
        variants=args.variants,
        force=args.force,
    )
    print(json.dumps(manifest.get("summary", []), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
