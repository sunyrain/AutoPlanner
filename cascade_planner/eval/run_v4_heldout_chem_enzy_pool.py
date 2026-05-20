"""Run ChemEnzy on held-out v4 targets and emit a native route pool."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig


SCHEMA_VERSION = "v4_heldout_chem_enzy_route_pool.v1"
ZERO_COST_TRACE_WEIGHTS = {
    "preferred_domain_reward": 0.0,
    "discouraged_domain_penalty": 0.0,
    "preferred_model_reward": 0.0,
    "discouraged_model_penalty": 0.0,
    "condition_conflict_penalty": 0.0,
    "missing_condition_penalty": 0.0,
    "cofactor_debt_penalty": 0.0,
    "cofactor_repair_reward": 0.0,
    "weak_enzyme_evidence_penalty": 0.0,
    "stage_transition_penalty": 0.0,
    "active_failure_match_reward": 0.0,
    "learned_source_value_rank_reward": 0.0,
    "learned_source_value_rank_penalty": 0.0,
    "learned_source_value_score_reward": 0.0,
    "learned_action_value_score_reward": 0.0,
    "min_cost": 1e-6,
}


def run_v4_heldout_chem_enzy_pool(
    *,
    benchmark: Path,
    output: Path,
    vendor_root: Path = Path("vendor/ChemEnzyRetroPlanner"),
    stock: list[str] | None = None,
    one_step_model: list[str] | None = None,
    iterations: int = 10,
    max_depth: int = 6,
    expansion_topk: int = 50,
    gpu: int = -1,
    limit: int | None = None,
    dry_run: bool = False,
    reuse_planner: bool = True,
    collect_expansion_trace: bool = False,
    action_value_model: Path | None = None,
    action_value_reward: float = 0.35,
) -> dict[str, Any]:
    targets = _read_targets(benchmark)
    if limit is not None and limit > 0:
        targets = targets[: int(limit)]
    search_flags = {
        "gpu": gpu,
        "keep_search": True,
        "use_filter": False,
        "use_depth_value_fn": False,
    }
    if collect_expansion_trace or action_value_model is not None:
        weights = dict(ZERO_COST_TRACE_WEIGHTS)
        cascade_cost_model: dict[str, Any] = {
            "enabled": True,
            "weights": weights,
        }
        if action_value_model is not None:
            weights["learned_action_value_score_reward"] = float(action_value_reward)
            cascade_cost_model["action_value_model_path"] = str(action_value_model)
        search_flags.update(
            {
                "use_cascade_cost_model": True,
                "cascade_cost_model": cascade_cost_model,
                "cascade_search_context": {
                    "context_source": "chem_enzy_zero_cost_expansion_trace",
                    "context_policy": "action_value_model" if action_value_model is not None else "trace_collection_only",
                },
                "include_cascade_expansion_trace": True,
                "cascade_expansion_trace_preview": 20,
            }
        )
    configs = [
        RouteSearchConfig(
            target_smiles=str(row["target_smiles"]),
            stock_names=stock or DEFAULT_STOCKS,
            max_iterations=iterations,
            max_depth=max_depth,
            expansion_topk=expansion_topk,
            one_step_models=one_step_model or DEFAULT_ONE_STEP_MODELS,
            search_flags=dict(search_flags),
        )
        for row in targets
    ]
    adapter = ChemEnzyBackendAdapter(vendor_root=vendor_root, gpu=gpu)
    started = time.monotonic()
    results = adapter.run_targets(configs, dry_run=dry_run, reuse_planner=reuse_planner)

    output_targets = []
    for row, result in zip(targets, results):
        output_targets.append(
            {
                **{key: value for key, value in row.items() if key not in {"routes", "failures"}},
                "backend": result.backend,
                "solved": result.solved,
                "route_count": result.route_count,
                "routes": [route.to_dict() for route in result.routes],
                "failures": [failure.to_dict() for failure in result.failures],
                "raw_backend_metadata": result.raw_backend_metadata,
            }
        )

    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark": str(benchmark),
            "vendor_root": str(vendor_root),
            "dry_run": dry_run,
            "reuse_planner": reuse_planner,
            "iterations": iterations,
            "max_depth": max_depth,
            "expansion_topk": expansion_topk,
            "gpu": gpu,
            "stock": stock or DEFAULT_STOCKS,
            "one_step_model": one_step_model or DEFAULT_ONE_STEP_MODELS,
            "collect_expansion_trace": collect_expansion_trace,
            "action_value_model": str(action_value_model) if action_value_model else None,
            "action_value_reward": float(action_value_reward) if action_value_model else None,
            "trace_cost_policy": (
                "action_value_score_reward"
                if action_value_model
                else "zero_cost_no_search_score_change"
                if collect_expansion_trace
                else "off"
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "summary": _summary(output_targets),
        "targets": output_targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _read_targets(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("targets") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    out = [row for row in rows if isinstance(row, dict) and row.get("target_smiles")]
    if not out:
        raise ValueError(f"no target_smiles rows found: {path}")
    return out


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = [int(row.get("route_count") or 0) for row in targets]
    elapsed = []
    for row in targets:
        value = (row.get("raw_backend_metadata") or {}).get("elapsed_s")
        try:
            elapsed.append(float(value))
        except (TypeError, ValueError):
            pass
    return {
        "n_targets": len(targets),
        "solved": sum(1 for row in targets if row.get("solved")),
        "solved_rate": round(sum(1 for row in targets if row.get("solved")) / max(len(targets), 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(targets), 1), 6),
        "avg_search_time_s": round(sum(elapsed) / max(len(elapsed), 1), 6) if elapsed else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ChemEnzy on held-out v4 targets and save a native route pool")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--stock", action="append", default=[])
    ap.add_argument("--one-step-model", action="append", default=[])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--expansion-topk", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-reuse-planner", action="store_true")
    ap.add_argument("--collect-expansion-trace", action="store_true")
    ap.add_argument("--action-value-model")
    ap.add_argument("--action-value-reward", type=float, default=0.35)
    args = ap.parse_args()
    payload = run_v4_heldout_chem_enzy_pool(
        benchmark=Path(args.benchmark),
        output=Path(args.output),
        vendor_root=Path(args.vendor_root),
        stock=args.stock or None,
        one_step_model=args.one_step_model or None,
        iterations=args.iterations,
        max_depth=args.max_depth,
        expansion_topk=args.expansion_topk,
        gpu=args.gpu,
        limit=args.limit,
        dry_run=args.dry_run,
        reuse_planner=not args.no_reuse_planner,
        collect_expansion_trace=args.collect_expansion_trace,
        action_value_model=Path(args.action_value_model) if args.action_value_model else None,
        action_value_reward=args.action_value_reward,
    )
    print(json.dumps({"summary": payload["summary"], "output": args.output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
