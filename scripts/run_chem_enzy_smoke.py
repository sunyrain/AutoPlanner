"""Run ChemEnzyRetroPlanner external-baseline smoke targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
    write_baseline_results,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig


def read_targets(path: Path | None, direct_targets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("targets", "items", "rows"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"unsupported benchmark format: {path}")
        rows.extend(row for row in data if isinstance(row, dict) and row.get("target_smiles"))
    rows.extend({"target_smiles": target} for target in direct_targets)
    if not rows:
        raise ValueError("no targets supplied; use --benchmark or --target")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ChemEnzyRetroPlanner core-search smoke")
    ap.add_argument("--benchmark", default="data/benchmark_cascade_gold_smoke_v1.json")
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--output", default="results/shared/chem_enzy_baseline/smoke.json")
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--stock", action="append", default=[])
    ap.add_argument("--one-step-model", action="append", default=[])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--expansion-topk", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Validate setup and emit structured requests only")
    ap.add_argument("--enable-condition-prediction", action="store_true")
    ap.add_argument("--condition-model", default="rcr", choices=["rcr", "parrot"])
    ap.add_argument("--enable-enzyme-assignment", action="store_true")
    ap.add_argument("--enable-easifa", action="store_true")
    ap.add_argument("--no-reuse-planner", action="store_true", help="Rebuild the vendor planner for every target")
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark) if args.benchmark else None
    rows = read_targets(benchmark_path, args.target)
    if args.limit is not None:
        rows = rows[: args.limit]

    adapter = ChemEnzyBackendAdapter(
        vendor_root=Path(args.vendor_root),
        gpu=args.gpu,
        enable_condition_prediction=args.enable_condition_prediction,
        enable_enzyme_assignment=args.enable_enzyme_assignment,
        enable_easifa=args.enable_easifa,
    )
    configs = []
    for row in rows:
        configs.append(RouteSearchConfig(
            target_smiles=str(row["target_smiles"]),
            stock_names=args.stock or DEFAULT_STOCKS,
            max_iterations=args.iterations,
            max_depth=args.max_depth,
            expansion_topk=args.expansion_topk,
            one_step_models=args.one_step_model or DEFAULT_ONE_STEP_MODELS,
            search_flags={"gpu": args.gpu, "condition_model": args.condition_model},
        ))
    results = adapter.run_targets(
        configs,
        dry_run=args.dry_run,
        reuse_planner=not args.no_reuse_planner,
    )
    if args.fail_on_error:
        first_failure_idx = next((idx for idx, result in enumerate(results) if result.failures), None)
        if first_failure_idx is not None:
            results = results[: first_failure_idx + 1]

    write_baseline_results(
        results,
        Path(args.output),
        metadata={
            "backend": "ChemEnzyRetroPlanner",
            "benchmark": str(benchmark_path) if benchmark_path else None,
            "vendor_root": args.vendor_root,
            "dry_run": args.dry_run,
            "n_requested": len(rows),
            "core_search_only": not (args.enable_condition_prediction or args.enable_enzyme_assignment),
            "reuse_planner": not args.no_reuse_planner,
            "condition_model": args.condition_model if args.enable_condition_prediction else None,
            "disabled_first_stage_components": {
                "llama_agent": True,
                "web_ui": True,
                "easifa": not args.enable_easifa,
            },
        },
    )
    print(json.dumps({"output": args.output, "targets": len(results)}, indent=2))

    if args.fail_on_error and any(result.failures for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
