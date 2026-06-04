#!/usr/bin/env python3
"""Measure RetroChimera one-step retrosynthesis latency on product SMILES."""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RETROCHIMERA_ROOT = REPO_ROOT / "data_external" / "retrochimera"
if str(RETROCHIMERA_ROOT) not in sys.path:
    sys.path.insert(0, str(RETROCHIMERA_ROOT))

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from retrochimera import RetroChimeraModel  # noqa: E402
from syntheseus import Molecule  # noqa: E402


SCHEMA_VERSION = "retrochimera_latency.v1"


def main() -> None:
    args = _parse_args()
    products = _load_products(args.src, args.limit)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "src": str(args.src),
            "limit": args.limit,
            "model_dir": str(args.model_dir),
            "devices": args.device,
            "topk": args.topk,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
        },
        "n_products": len(products),
        "runs": [],
    }
    for device in args.device:
        payload["runs"].extend(
            _measure_device(
                products=products,
                model_dir=args.model_dir,
                device=device,
                topks=args.topk,
                batch_sizes=args.batch_size,
                warmup=args.warmup,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(payload), encoding="utf-8")
    print(_render_markdown(payload))


def _measure_device(
    *,
    products: list[str],
    model_dir: Path,
    device: str,
    topks: list[int],
    batch_sizes: list[int],
    warmup: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    model = RetroChimeraModel(model_dir=model_dir, device=device)
    load_elapsed_s = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for topk in topks:
        for batch_size in batch_sizes:
            warmup_batches = _batched(products[: max(0, warmup)], batch_size)
            for batch in warmup_batches:
                model([Molecule(smi) for smi in batch], num_results=topk)

            per_product_latencies: list[float] = []
            candidate_counts: list[int] = []
            errors: list[str] = []
            for batch in _batched(products, batch_size):
                t0 = time.perf_counter()
                try:
                    result = model([Molecule(smi) for smi in batch], num_results=topk)
                    elapsed = time.perf_counter() - t0
                    per_product_latencies.extend([elapsed / len(batch)] * len(batch))
                    candidate_counts.extend(len(item) for item in result)
                except Exception as exc:  # pragma: no cover - depends on external runtime
                    elapsed = time.perf_counter() - t0
                    per_product_latencies.extend([elapsed / len(batch)] * len(batch))
                    candidate_counts.extend([0] * len(batch))
                    errors.append(f"{type(exc).__name__}: {exc}")
            rows.append(
                {
                    "device": device,
                    "topk": topk,
                    "batch_size": batch_size,
                    "load_elapsed_s": round(load_elapsed_s, 3),
                    "n_products": len(products),
                    "errors": len(errors),
                    "error_preview": errors[:5],
                    "mean_latency_s": _round(statistics.fmean(per_product_latencies)),
                    "median_latency_s": _round(statistics.median(per_product_latencies)),
                    "p90_latency_s": _round(_quantile(per_product_latencies, 0.90)),
                    "p95_latency_s": _round(_quantile(per_product_latencies, 0.95)),
                    "mean_candidate_count": _round(statistics.fmean(candidate_counts)),
                }
            )
    return rows


def _load_products(src: Path, limit: int | None) -> list[str]:
    products: list[str] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            smi = line.strip().replace(" ", "")
            if smi:
                products.append(smi)
            if limit is not None and len(products) >= limit:
                break
    if not products:
        raise ValueError(f"no products loaded from {src}")
    return products


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RetroChimera Latency",
        "",
        f"generated_at: `{payload['generated_at']}`",
        f"n_products: `{payload['n_products']}`",
        "",
        "| device | topk | batch | load s | mean s/product | median | p90 | p95 | candidates | errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| `{row['device']}` | {row['topk']} | {row['batch_size']} | {row['load_elapsed_s']} | "
            f"{row['mean_latency_s']} | {row['median_latency_s']} | {row['p90_latency_s']} | "
            f"{row['p95_latency_s']} | {row['mean_candidate_count']} | {row['errors']} |"
        )
    return "\n".join(lines)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _round(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("results/shared/uspto_product_only_chem_onmt_20260530/corpus/plain.test.src"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("data_external/retrochimera_model"))
    parser.add_argument("--device", action="append")
    parser.add_argument("--topk", action="append", type=int)
    parser.add_argument("--batch-size", action="append", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    args.device = args.device or ["cuda:0"]
    args.topk = args.topk or [20]
    args.batch_size = args.batch_size or [1]
    return args


if __name__ == "__main__":
    main()
