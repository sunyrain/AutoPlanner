"""Compare ChemEnzy native search with and without AutoPlanner enzyme injection."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig
from scripts.run_bridge_live_policy_benchmark_v0 import load_negative_targets, load_positive_targets


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/native_enzyme_plugin_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--expansion-topk", type=int, default=75)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--plugin-top-k", type=int, default=6)
    parser.add_argument("--plugin-max-added", type=int, default=6)
    parser.add_argument("--bridge-top-k", type=int, default=8)
    parser.add_argument("--max-ec-contexts", type=int, default=2)
    parser.add_argument("--sp-v1-score-bonus", type=float, default=0.0)
    parser.add_argument("--quality-score-bonus", type=float, default=0.0)
    parser.add_argument("--min-quality-score", type=float, default=None)
    parser.add_argument("--disable-material-gate", action="store_true")
    parser.add_argument("--material-max-heavy-gain", type=int, default=3)
    parser.add_argument("--material-max-carbon-gain", type=int, default=2)
    parser.add_argument("--material-max-hetero-gain", type=int, default=3)
    parser.add_argument("--disable-sp-v1", action="store_true")
    parser.add_argument("--disable-sp-v1-hard-gate", action="store_true")
    parser.add_argument("--disable-bridge-gate", action="store_true")
    parser.add_argument("--disable-bridge-verifier", action="store_true")
    parser.add_argument("--enable-chemical-plugin", action="store_true")
    parser.add_argument("--chemical-plugin-top-k", type=int, default=8)
    parser.add_argument("--chemical-plugin-max-added", type=int, default=8)
    parser.add_argument("--chemical-plugin-dual-top-k", type=int, default=100)
    parser.add_argument("--chemical-plugin-graphfp-top-k", type=int, default=50)
    parser.add_argument("--chemical-plugin-score-scale", type=float, default=0.75)
    parser.add_argument("--chemical-plugin-fusion-mode", default="graphfp_first")
    parser.add_argument("--disable-chemical-proposal-gate", action="store_true")
    parser.add_argument("--skip-native-only", action="store_true")
    parser.add_argument("--skip-plugin", action="store_true")
    parser.add_argument("--skip-combo-plugin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    targets = load_targets(args)
    rows: list[dict[str, Any]] = []
    native_adapter = ChemEnzyBackendAdapter(gpu=int(args.gpu))
    plugin_adapter = ChemEnzyBackendAdapter(gpu=int(args.gpu))
    combo_adapter = ChemEnzyBackendAdapter(gpu=int(args.gpu))
    for idx, target in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] {target['target_smiles']}", flush=True)
        if not args.skip_native_only:
            rows.append(run_one(target, run_name="native_only", adapter=native_adapter, config=base_config(target, args)))
        if not args.skip_plugin:
            rows.append(
                run_one(
                    target,
                    run_name="native_enzyme_plugin",
                    adapter=plugin_adapter,
                    config=plugin_config(target, args),
                )
            )
        if args.enable_chemical_plugin and not args.skip_combo_plugin:
            rows.append(
                run_one(
                    target,
                    run_name="native_enzyme_chemical_plugin",
                    adapter=combo_adapter,
                    config=combo_config(target, args),
                )
            )
    report = {
        "schema_version": "native_enzyme_plugin_benchmark.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "targets": len(targets),
            "positives": sum(1 for row in targets if int(row.get("label") or 0) == 1),
            "negatives": sum(1 for row in targets if int(row.get("label") or 0) == 0),
            "iterations": int(args.iterations),
            "max_depth": int(args.max_depth),
            "expansion_topk": int(args.expansion_topk),
            "plugin": enzyme_plugin_payload(args),
            "chemical_plugin": chemical_plugin_payload(args) if args.enable_chemical_plugin else {"enabled": False},
        },
        "summaries": summarize(rows),
        "rows_jsonl": str(args.output_dir / "native_enzyme_plugin_rows.jsonl"),
        "comparability_notes": [
            "Both runs use ChemEnzy native search and stock closure.",
            "The plugin run only wraps native per-node one-step expansion; it does not replace the native search loop.",
            "The combo run wraps the same native expansion with enzyme candidates plus GraphFP-first dual-tower chemical tail candidates.",
            "A plugin-added enzyme step is counted only if it appears in a returned ChemEnzy route.",
            "Partial plugin stats count proposal injections during search, including candidates that were not selected into final routes.",
        ],
    }
    rows_path = args.output_dir / "native_enzyme_plugin_rows.jsonl"
    report_json = args.output_dir / "native_enzyme_plugin_report.json"
    report_md = args.output_dir / "native_enzyme_plugin_report.md"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "rows": str(rows_path), "summaries": report["summaries"]}, ensure_ascii=False, indent=2))


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    positives = load_positive_targets(args.probe_rows, count=max(0, int(args.positives)))
    negatives = load_negative_targets(args.pack_dir, count=max(0, int(args.negatives)), seed=int(args.seed))
    rows = [*positives, *negatives]
    if int(args.max_targets) > 0:
        rows = rows[: int(args.max_targets)]
    return rows


def base_config(target: dict[str, Any], args: argparse.Namespace) -> RouteSearchConfig:
    return RouteSearchConfig(
        target_smiles=str(target.get("target_smiles") or ""),
        stock_names=list(DEFAULT_STOCKS),
        max_iterations=max(1, int(args.iterations)),
        max_depth=max(1, int(args.max_depth)),
        expansion_topk=max(1, int(args.expansion_topk)),
        one_step_models=list(DEFAULT_ONE_STEP_MODELS),
        search_flags={
            "gpu": int(args.gpu),
            "keep_search": True,
            "use_filter": False,
            "use_depth_value_fn": False,
        },
    )


def plugin_config(target: dict[str, Any], args: argparse.Namespace) -> RouteSearchConfig:
    config = base_config(target, args)
    config.search_flags["native_enzyme_plugin"] = enzyme_plugin_payload(args)
    return config


def combo_config(target: dict[str, Any], args: argparse.Namespace) -> RouteSearchConfig:
    config = plugin_config(target, args)
    config.search_flags["native_chemical_plugin"] = chemical_plugin_payload(args)
    return config


def enzyme_plugin_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": True,
        "pack_dir": str(args.pack_dir),
        "top_k": max(1, int(args.plugin_top_k)),
        "bridge_top_k": max(1, int(args.bridge_top_k)),
        "max_ec_contexts": max(0, int(args.max_ec_contexts)),
        "require_bridge": not bool(args.disable_bridge_gate),
        "require_verifier_pass": not bool(args.disable_bridge_verifier),
        "enable_sp_v1": not bool(args.disable_sp_v1),
        "sp_v1_hard_gate": not bool(args.disable_sp_v1_hard_gate),
        "max_added": max(1, int(args.plugin_max_added)),
        "sp_v1_score_bonus": float(args.sp_v1_score_bonus),
        "quality_score_bonus": float(args.quality_score_bonus),
        "require_material_sanity": not bool(args.disable_material_gate),
        "material_max_heavy_gain": max(0, int(args.material_max_heavy_gain)),
        "material_max_carbon_gain": max(0, int(args.material_max_carbon_gain)),
        "material_max_hetero_gain": max(0, int(args.material_max_hetero_gain)),
    }
    if args.min_quality_score is not None:
        payload["min_quality_score"] = float(args.min_quality_score)
    return payload


def chemical_plugin_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "top_k": max(1, int(args.chemical_plugin_top_k)),
        "max_added": max(1, int(args.chemical_plugin_max_added)),
        "dual_top_k": max(1, int(args.chemical_plugin_dual_top_k)),
        "graphfp_top_k": max(1, int(args.chemical_plugin_graphfp_top_k)),
        "fusion_mode": str(args.chemical_plugin_fusion_mode or "graphfp_first"),
        "score_scale": float(args.chemical_plugin_score_scale),
        "require_proposal_gate": not bool(args.disable_chemical_proposal_gate),
    }


def run_one(
    target: dict[str, Any],
    *,
    run_name: str,
    adapter: ChemEnzyBackendAdapter,
    config: RouteSearchConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    result = adapter.run_target(config)
    elapsed = time.monotonic() - started
    return result_row(target, run_name=run_name, result=result, elapsed_s=elapsed)


def result_row(target: dict[str, Any], *, run_name: str, result: BaselineRunResult, elapsed_s: float) -> dict[str, Any]:
    plugin_stats = (result.raw_backend_metadata or {}).get("native_enzyme_plugin") or {}
    chemical_plugin_stats = (result.raw_backend_metadata or {}).get("native_chemical_plugin") or {}
    routes = [route.to_dict() for route in result.routes]
    enzyme_routes = [route for route in result.routes if route.enzymatic_step_present]
    enzyme_steps = [
        (route.route_rank, step)
        for route in result.routes
        for step in route.steps
        if step.has_enzyme_annotation or "enzyme" in str(step.source_model).lower()
    ]
    return {
        "run": run_name,
        "target_smiles": result.target_smiles,
        "label": int(target.get("label") or 0),
        "label_source": target.get("label_source"),
        "ok": result.solved,
        "solved": result.solved,
        "route_count": result.route_count,
        "enzyme_route_count": len(enzyme_routes),
        "enzyme_step_count": len(enzyme_steps),
        "enzyme_steps_preview": [
            {
                "route_rank": route_rank,
                "source_model": step.source_model,
                "rxn_smiles": step.rxn_smiles,
                "ec_numbers": [item.get("ec_number") for item in step.enzyme_ec_annotations],
                "score": step.score,
            }
            for route_rank, step in enzyme_steps[:8]
        ],
        "elapsed_s": round(float(elapsed_s), 3),
        "backend_elapsed_s": (result.raw_backend_metadata or {}).get("elapsed_s"),
        "failure_categories": [failure.category for failure in result.failures],
        "plugin_stats": plugin_stats,
        "chemical_plugin_stats": chemical_plugin_stats,
        "routes_preview": routes[:3],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_name in sorted({str(row.get("run")) for row in rows}):
        group = [row for row in rows if row.get("run") == run_name]
        n = len(group)
        plugin_stats = [row.get("plugin_stats") or {} for row in group]
        chemical_stats = [row.get("chemical_plugin_stats") or {} for row in group]
        out[run_name] = {
            "targets": n,
            "solved_targets": sum(1 for row in group if row.get("solved")),
            "solved_rate": _rate(sum(1 for row in group if row.get("solved")), n),
            "targets_with_routes": sum(1 for row in group if int(row.get("route_count") or 0) > 0),
            "total_routes": sum(int(row.get("route_count") or 0) for row in group),
            "targets_with_enzyme_route": sum(1 for row in group if int(row.get("enzyme_route_count") or 0) > 0),
            "total_enzyme_steps": sum(int(row.get("enzyme_step_count") or 0) for row in group),
            "mean_elapsed_s": _mean([row.get("elapsed_s") for row in group]),
            "failure_categories": _count(cat for row in group for cat in row.get("failure_categories") or []),
            "plugin_calls": sum(int(stat.get("calls") or 0) for stat in plugin_stats),
            "plugin_bridge_hit_calls": sum(int(stat.get("bridge_hit_calls") or 0) for stat in plugin_stats),
            "plugin_added_candidates": sum(int(stat.get("added_candidates") or 0) for stat in plugin_stats),
            "plugin_sp_v1_scored": sum(int(stat.get("sp_v1_scored") or 0) for stat in plugin_stats),
            "plugin_sp_v1_accepted": sum(int(stat.get("sp_v1_accepted") or 0) for stat in plugin_stats),
            "plugin_sp_v1_rejected": sum(int(stat.get("sp_v1_rejected") or 0) for stat in plugin_stats),
            "plugin_quality_passed": sum(int(stat.get("quality_passed") or 0) for stat in plugin_stats),
            "plugin_quality_warned": sum(int(stat.get("quality_warned") or 0) for stat in plugin_stats),
            "plugin_quality_rejected": sum(int(stat.get("quality_rejected") or 0) for stat in plugin_stats),
            "plugin_material_rejected": sum(int(stat.get("material_rejected") or 0) for stat in plugin_stats),
            "plugin_errors": sum(int(stat.get("error_count") or 0) for stat in plugin_stats),
            "chemical_plugin_calls": sum(int(stat.get("calls") or 0) for stat in chemical_stats),
            "chemical_plugin_dual_candidates": sum(int(stat.get("dual_candidates") or 0) for stat in chemical_stats),
            "chemical_plugin_added_candidates": sum(int(stat.get("added_candidates") or 0) for stat in chemical_stats),
            "chemical_plugin_gate_kept": sum(int(stat.get("proposal_gate_kept") or 0) for stat in chemical_stats),
            "chemical_plugin_gate_rejected": sum(int(stat.get("proposal_gate_rejected") or 0) for stat in chemical_stats),
            "chemical_plugin_errors": sum(int(stat.get("error_count") or 0) for stat in chemical_stats),
        }
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native ChemEnzy Enzyme Plugin Benchmark",
        "",
        f"Elapsed seconds: {report['elapsed_seconds']}",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(report["inputs"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| run | targets | solved | solved rate | routes | enzyme route targets | enzyme steps | enzyme added | chemical added | SP accepted | SP rejected | mean s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, row in report["summaries"].items():
        lines.append(
            "| {run} | {targets} | {solved} | {rate:.4f} | {routes} | {enzyme_targets} | {enzyme_steps} | {added} | {chem_added} | {accepted} | {rejected} | {elapsed:.3f} |".format(
                run=run_name,
                targets=row["targets"],
                solved=row["solved_targets"],
                rate=float(row["solved_rate"] or 0.0),
                routes=row["total_routes"],
                enzyme_targets=row["targets_with_enzyme_route"],
                enzyme_steps=row["total_enzyme_steps"],
                added=row["plugin_added_candidates"],
                chem_added=row["chemical_plugin_added_candidates"],
                accepted=row["plugin_sp_v1_accepted"],
                rejected=row["plugin_sp_v1_rejected"],
                elapsed=float(row["mean_elapsed_s"] or 0.0),
            )
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("comparability_notes") or [])
    lines.append("")
    return "\n".join(lines)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _mean(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return round(sum(nums) / len(nums), 6) if nums else None


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    main()
