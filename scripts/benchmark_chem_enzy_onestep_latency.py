#!/usr/bin/env python3
"""Measure loaded ChemEnzy one-step proposal latency on product SMILES."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider  # noqa: E402
from scripts.audit_chem_enzy_onestep_benchmark import _collect_transitions, _load_benchmark  # noqa: E402


SCHEMA_VERSION = "chem_enzy_onestep_latency.v1"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = _parse_args()
    products = _load_products(args)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "benchmark": str(args.benchmark) if args.benchmark else None,
            "src": str(args.src) if args.src else None,
            "limit": args.limit,
            "topk": args.topk,
            "gpu": args.gpu,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "vendor_root": str(args.vendor_root),
            "onmt_model_path": str(args.onmt_model_path) if args.onmt_model_path else None,
            "onmt_tokenizer": args.onmt_tokenizer,
        },
        "n_products": len(products),
        "runs": {},
    }
    for raw_run in args.run:
        name, models = _parse_run(raw_run)
        payload["runs"][name] = _measure_run(
            products=products,
            models=models,
            vendor_root=args.vendor_root,
            gpu=args.gpu,
            topk=args.topk,
            warmup=args.warmup,
            repeat=args.repeat,
            onmt_model_path=args.onmt_model_path,
            onmt_tokenizer=args.onmt_tokenizer,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(payload), encoding="utf-8")
    print(_render_markdown(payload))


def _measure_run(
    *,
    products: list[str],
    models: list[str],
    vendor_root: Path,
    gpu: int,
    topk: int,
    warmup: int,
    repeat: int,
    onmt_model_path: Path | None,
    onmt_tokenizer: str | None,
) -> dict[str, Any]:
    provider = ChemEnzyOneStepProposalProvider(
        vendor_root=vendor_root,
        models=tuple(models),
        expansion_topk=int(topk),
        gpu=int(gpu),
        onmt_model_path=onmt_model_path,
        onmt_tokenizer=onmt_tokenizer,
    )
    load_started = time.perf_counter()
    one_step = provider._ensure_one_step()
    load_elapsed_s = time.perf_counter() - load_started
    for product in products[: max(0, int(warmup))]:
        one_step.run(product, topk=int(topk))

    latencies: list[float] = []
    candidate_counts: list[int] = []
    errors: list[str] = []
    for _ in range(max(1, int(repeat))):
        for product in products:
            started = time.perf_counter()
            try:
                result = one_step.run(product, topk=int(topk)) or {}
                candidate_counts.append(len(result.get("reactants") or []))
            except Exception as exc:  # pragma: no cover - depends on vendor runtime
                errors.append(f"{type(exc).__name__}:{exc}")
                candidate_counts.append(0)
            latencies.append(time.perf_counter() - started)
    return {
        "models": models,
        "load_elapsed_s": round(load_elapsed_s, 3),
        "n_calls": len(latencies),
        "errors": len(errors),
        "error_preview": errors[:5],
        "mean_latency_s": _round(statistics.fmean(latencies) if latencies else None),
        "median_latency_s": _round(statistics.median(latencies) if latencies else None),
        "p90_latency_s": _round(_quantile(latencies, 0.90)),
        "p95_latency_s": _round(_quantile(latencies, 0.95)),
        "min_latency_s": _round(min(latencies) if latencies else None),
        "max_latency_s": _round(max(latencies) if latencies else None),
        "mean_candidate_count": _round(statistics.fmean(candidate_counts) if candidate_counts else None),
    }


def _load_products(args: argparse.Namespace) -> list[str]:
    products: list[str] = []
    if args.src:
        with Path(args.src).open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    products.append(text.replace(" ", ""))
    if args.benchmark:
        transitions = _collect_transitions(_load_benchmark(Path(args.benchmark)), step_scope=args.step_scope)
        products.extend(str(row.get("product_smiles") or "") for row in transitions if row.get("product_smiles"))
    deduped = list(dict.fromkeys(product for product in products if product))
    if args.limit is not None:
        deduped = deduped[: max(0, int(args.limit))]
    if not deduped:
        raise ValueError("no products loaded")
    return deduped


def _parse_run(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise ValueError("--run must be NAME=model1,model2")
    name, models_text = raw.split("=", 1)
    models = [item.strip() for item in models_text.split(",") if item.strip()]
    if not name.strip() or not models:
        raise ValueError("--run must include a nonempty name and at least one model")
    return name.strip(), models


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy One-step Latency",
        "",
        f"generated_at: `{payload['generated_at']}`",
        f"n_products: `{payload['n_products']}`",
        "",
        "| run | models | calls | load s | mean s | median s | p90 s | p95 s | candidates | errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload.get("runs", {}).items():
        lines.append(
            f"| {name} | `{','.join(row['models'])}` | {row['n_calls']} | {row['load_elapsed_s']} | "
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
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--step-scope", choices=["all", "target"], default="target")
    parser.add_argument("--run", action="append", required=True, help="NAME=model1,model2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--onmt-model-path", type=Path)
    parser.add_argument("--onmt-tokenizer", choices=["char", "token"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
