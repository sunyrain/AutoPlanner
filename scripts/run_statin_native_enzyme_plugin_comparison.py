"""Run native ChemEnzy vs native+enzyme-plugin comparison on the statin panel."""
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

from cascade_planner.baselines.chem_enzy_adapter import ChemEnzyBackendAdapter, DEFAULT_ONE_STEP_MODELS, DEFAULT_STOCKS
from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import PLUGIN_MODEL_FULL_NAME
from cascade_planner.baselines.route_plausibility import (
    audit_route_plausibility,
    audit_step_plausibility,
    plausibility_failure_counts,
)
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteSearchConfig, RouteStepCandidate


DEFAULT_STATIN_SUMMARY = Path("docs/statins/summary.json")
DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_OUTPUT_DIR = Path("results/shared/statin_native_enzyme_plugin_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statin-summary", type=Path, default=DEFAULT_STATIN_SUMMARY)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default="", help="Comma-separated statin safe names. Empty means all.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--expansion-topk", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--plugin-top-k", type=int, default=4)
    parser.add_argument("--plugin-max-added", type=int, default=4)
    parser.add_argument("--bridge-top-k", type=int, default=8)
    parser.add_argument("--max-ec-contexts", type=int, default=2)
    parser.add_argument("--sp-v1-score-bonus", type=float, default=0.0)
    parser.add_argument("--quality-score-bonus", type=float, default=0.0)
    parser.add_argument("--min-quality-score", type=float, default=None)
    parser.add_argument("--disable-material-gate", action="store_true")
    parser.add_argument("--material-max-heavy-gain", type=int, default=3)
    parser.add_argument("--material-max-carbon-gain", type=int, default=2)
    parser.add_argument("--material-max-hetero-gain", type=int, default=3)
    parser.add_argument(
        "--enable-enzyme-assignment",
        action="store_true",
        help="Ask ChemEnzy to classify returned route steps and assign EC numbers in both runs.",
    )
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
    parser.add_argument(
        "--dump-routes",
        action="store_true",
        help="Write every returned route with a conservative material plausibility audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    targets = load_statin_targets(args.statin_summary, selected=args.targets)
    native_adapter = ChemEnzyBackendAdapter(
        gpu=int(args.gpu),
        enable_enzyme_assignment=bool(args.enable_enzyme_assignment),
    )
    plugin_adapter = ChemEnzyBackendAdapter(
        gpu=int(args.gpu),
        enable_enzyme_assignment=bool(args.enable_enzyme_assignment),
    )
    combo_adapter = ChemEnzyBackendAdapter(
        gpu=int(args.gpu),
        enable_enzyme_assignment=bool(args.enable_enzyme_assignment),
    )
    print(f"Running native_only on {len(targets)} statins", flush=True)
    native_results = native_adapter.run_targets([base_config(target, args) for target in targets], reuse_planner=True)
    print(f"Running native_enzyme_plugin on {len(targets)} statins", flush=True)
    plugin_results = plugin_adapter.run_targets([plugin_config(target, args) for target in targets], reuse_planner=True)
    combo_results: list[BaselineRunResult] = []
    if args.enable_chemical_plugin:
        print(f"Running native_enzyme_chemical_plugin on {len(targets)} statins", flush=True)
        combo_results = combo_adapter.run_targets([combo_config(target, args) for target in targets], reuse_planner=True)
    native_by_target = {result.target_smiles: result for result in native_results}
    plugin_by_target = {result.target_smiles: result for result in plugin_results}
    combo_by_target = {result.target_smiles: result for result in combo_results}
    rows: list[dict[str, Any]] = []
    for target in targets:
        native = native_by_target.get(target["target_smiles"])
        plugin = plugin_by_target.get(target["target_smiles"])
        if native is None or plugin is None:
            raise RuntimeError(f"missing benchmark result for {target['name']}")
        combo = combo_by_target.get(target["target_smiles"]) if combo_results else None
        rows.append(compare_target(target, native=native, plugin=plugin, combo=combo))
    route_dump_jsonl = None
    if args.dump_routes:
        route_dump_jsonl = args.output_dir / "statin_native_enzyme_plugin_full_routes.jsonl"
        write_route_dump_jsonl(
            route_dump_jsonl,
            targets=targets,
            native_by_target=native_by_target,
            plugin_by_target=plugin_by_target,
            combo_by_target=combo_by_target,
        )
    report = {
        "schema_version": "statin_native_enzyme_plugin_comparison.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "statin_summary": str(args.statin_summary),
            "targets": [target["name"] for target in targets],
            "iterations": int(args.iterations),
            "max_depth": int(args.max_depth),
            "expansion_topk": int(args.expansion_topk),
            "enable_enzyme_assignment": bool(args.enable_enzyme_assignment),
            "dump_routes": bool(args.dump_routes),
            "plugin": enzyme_plugin_payload(args),
            "chemical_plugin": chemical_plugin_payload(args) if args.enable_chemical_plugin else {"enabled": False},
        },
        "summary": summarize(rows),
        "targets": rows,
    }
    if route_dump_jsonl is not None:
        report["route_dump_jsonl"] = str(route_dump_jsonl)
    report_json = args.output_dir / "statin_native_enzyme_plugin_comparison.json"
    report_md = args.output_dir / "statin_native_enzyme_plugin_comparison.md"
    rows_jsonl = args.output_dir / "statin_native_enzyme_plugin_rows.jsonl"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    rows_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "markdown": str(report_md), "summary": report["summary"]}, indent=2, ensure_ascii=False))


def load_statin_targets(path: Path, *, selected: str = "") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wanted = {item.strip().lower() for item in str(selected or "").split(",") if item.strip()}
    rows = []
    for row in payload.get("targets") or []:
        safe = str(row.get("safe") or row.get("name") or "").lower()
        if wanted and safe not in wanted:
            continue
        rows.append(
            {
                "name": str(row.get("name") or safe),
                "safe": safe,
                "target_smiles": str(row.get("smiles") or ""),
                "showcase_route_count": int(row.get("showcase_route_count") or 0),
                "source_route_count": int(row.get("source_route_count") or 0),
            }
        )
    return rows


def base_config(target: dict[str, Any], args: argparse.Namespace) -> RouteSearchConfig:
    return RouteSearchConfig(
        target_smiles=str(target["target_smiles"]),
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


def compare_target(
    target: dict[str, Any],
    *,
    native: BaselineRunResult,
    plugin: BaselineRunResult,
    combo: BaselineRunResult | None = None,
) -> dict[str, Any]:
    native_rep = representative_route(native.routes, prefer_enzyme=False)
    plugin_rep = representative_route(plugin.routes, prefer_enzyme=True)
    combo_rep = representative_route(combo.routes, prefer_enzyme=True) if combo is not None else None
    plugin_stats = (plugin.raw_backend_metadata or {}).get("native_enzyme_plugin") or {}
    row = {
        "name": target["name"],
        "safe": target["safe"],
        "target_smiles": target["target_smiles"],
        "native": result_summary(native, native_rep),
        "plugin": result_summary(plugin, plugin_rep),
        "plugin_stats": plugin_stats,
        "delta": {
            "solved_changed": bool(native.solved) != bool(plugin.solved),
            "route_count_delta": int(plugin.route_count) - int(native.route_count),
            "any_enzyme_route_count_delta": any_enzyme_route_count(plugin.routes) - any_enzyme_route_count(native.routes),
            "native_classified_enzyme_route_count_delta": (
                native_classified_enzyme_route_count(plugin.routes) - native_classified_enzyme_route_count(native.routes)
            ),
            "plugin_injected_enzyme_route_count_delta": plugin_injected_enzyme_route_count(plugin.routes),
            "representative_changed": route_signature(native_rep) != route_signature(plugin_rep),
            "representative_step_delta": route_step_count(plugin_rep) - route_step_count(native_rep),
            "plugin_selected_enzyme_route": bool(plugin_rep and has_any_enzyme_step(plugin_rep)),
            "plugin_selected_injected_enzyme_route": bool(plugin_rep and has_plugin_injected_enzyme_step(plugin_rep)),
            "plugin_selected_native_classified_enzyme_route": bool(
                plugin_rep and has_native_classified_enzyme_step(plugin_rep)
            ),
        },
        "representative_difference": representative_difference(native_rep, plugin_rep),
        "plugin_representative_audit": audit_plugin_representative(plugin_rep),
    }
    if combo is not None:
        combo_enzyme_stats = (combo.raw_backend_metadata or {}).get("native_enzyme_plugin") or {}
        combo_chemical_stats = (combo.raw_backend_metadata or {}).get("native_chemical_plugin") or {}
        row["combo"] = result_summary(combo, combo_rep)
        row["combo_plugin_stats"] = combo_enzyme_stats
        row["combo_chemical_plugin_stats"] = combo_chemical_stats
        row["combo_delta"] = {
            "solved_changed_vs_native": bool(native.solved) != bool(combo.solved),
            "route_count_delta_vs_native": int(combo.route_count) - int(native.route_count),
            "route_count_delta_vs_enzyme_plugin": int(combo.route_count) - int(plugin.route_count),
            "plugin_injected_enzyme_route_count_delta": plugin_injected_enzyme_route_count(combo.routes),
            "combo_selected_enzyme_route": bool(combo_rep and has_any_enzyme_step(combo_rep)),
            "combo_selected_injected_enzyme_route": bool(combo_rep and has_plugin_injected_enzyme_step(combo_rep)),
            "combo_selected_native_chemical_route": bool(combo_rep and has_plugin_injected_chemical_step(combo_rep)),
        }
        row["combo_representative_difference"] = representative_difference(native_rep, combo_rep)
        row["combo_representative_audit"] = audit_plugin_representative(combo_rep)
    return row


def result_summary(result: BaselineRunResult, rep: RouteCandidate | None) -> dict[str, Any]:
    route_audit = route_pool_plausibility_summary(result.routes)
    return {
        "solved": result.solved,
        "route_count": result.route_count,
        "any_enzyme_route_count": any_enzyme_route_count(result.routes),
        "any_enzyme_step_count": any_enzyme_step_count(result.routes),
        "native_classified_enzyme_route_count": native_classified_enzyme_route_count(result.routes),
        "native_classified_enzyme_step_count": native_classified_enzyme_step_count(result.routes),
        "plugin_injected_enzyme_route_count": plugin_injected_enzyme_route_count(result.routes),
        "plugin_injected_enzyme_step_count": plugin_injected_enzyme_step_count(result.routes),
        "plugin_injected_chemical_route_count": plugin_injected_chemical_route_count(result.routes),
        "plugin_injected_chemical_step_count": plugin_injected_chemical_step_count(result.routes),
        "elapsed_s": (result.raw_backend_metadata or {}).get("elapsed_s"),
        "failure_categories": [failure.category for failure in result.failures],
        "route_plausibility": route_audit,
        "representative": route_payload(rep),
    }


def route_pool_plausibility_summary(routes: list[RouteCandidate]) -> dict[str, Any]:
    audits = [audit_route_plausibility(route) for route in routes]
    passed = sum(1 for audit in audits if audit.get("passed"))
    injected_enzyme_audits = [
        audit for route, audit in zip(routes, audits) if has_plugin_injected_enzyme_step(route)
    ]
    injected_chemical_audits = [
        audit for route, audit in zip(routes, audits) if has_plugin_injected_chemical_step(route)
    ]
    return {
        "routes": len(routes),
        "passed": passed,
        "failed": len(routes) - passed,
        "pass_rate": round(passed / len(routes), 6) if routes else 0.0,
        "failure_counts": plausibility_failure_counts(audits),
        "injected_enzyme_routes": {
            "routes": len(injected_enzyme_audits),
            "passed": sum(1 for audit in injected_enzyme_audits if audit.get("passed")),
            "failure_counts": plausibility_failure_counts(injected_enzyme_audits),
        },
        "injected_chemical_routes": {
            "routes": len(injected_chemical_audits),
            "passed": sum(1 for audit in injected_chemical_audits if audit.get("passed")),
            "failure_counts": plausibility_failure_counts(injected_chemical_audits),
        },
    }


def representative_route(routes: list[RouteCandidate], *, prefer_enzyme: bool) -> RouteCandidate | None:
    if prefer_enzyme:
        for route in routes:
            if has_plugin_injected_enzyme_step(route):
                return route
    return routes[0] if routes else None


def representative_difference(native: RouteCandidate | None, plugin: RouteCandidate | None) -> dict[str, Any]:
    return {
        "native_signature": route_signature(native),
        "plugin_signature": route_signature(plugin),
        "native_first_disconnection": step_payload(native.steps[0]) if native and native.steps else None,
        "plugin_first_disconnection": step_payload(plugin.steps[0]) if plugin and plugin.steps else None,
        "plugin_first_enzyme_step": step_payload(first_enzyme_step(plugin)),
        "plugin_first_injected_enzyme_step": step_payload(first_plugin_injected_enzyme_step(plugin)),
        "plugin_first_native_classified_enzyme_step": step_payload(first_native_classified_enzyme_step(plugin)),
        "native_source_sequence": source_sequence(native),
        "plugin_source_sequence": source_sequence(plugin),
    }


def route_payload(route: RouteCandidate | None) -> dict[str, Any] | None:
    if route is None:
        return None
    return {
        "route_rank": route.route_rank,
        "solved": route.solved,
        "steps": len(route.steps),
        "score": route.score,
        "enzymatic_step_present": route.enzymatic_step_present,
        "any_enzyme_step_present": has_any_enzyme_step(route),
        "native_classified_enzyme_step_present": has_native_classified_enzyme_step(route),
        "plugin_injected_enzyme_step_present": has_plugin_injected_enzyme_step(route),
        "source_sequence": source_sequence(route),
        "first_enzyme_step": step_payload(first_enzyme_step(route)),
        "first_injected_enzyme_step": step_payload(first_plugin_injected_enzyme_step(route)),
        "first_native_classified_enzyme_step": step_payload(first_native_classified_enzyme_step(route)),
        "steps_preview": [step_payload(step) for step in route.steps[:6]],
    }


def step_payload(step: RouteStepCandidate | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "product": step.product_smiles,
        "reactants": list(step.reactant_smiles),
        "rxn_smiles": step.rxn_smiles,
        "source_model": step.source_model,
        "score": step.score,
        "ec_numbers": [row.get("ec_number") for row in step.enzyme_ec_annotations],
        "stock_status": dict(step.stock_status or {}),
    }


def first_enzyme_step(route: RouteCandidate | None) -> RouteStepCandidate | None:
    if route is None:
        return None
    for step in route.steps:
        if is_plugin_injected_enzyme_step(step):
            return step
    for step in route.steps:
        if step.has_enzyme_annotation or "enzyme" in str(step.source_model).lower():
            return step
    return None


def first_plugin_injected_enzyme_step(route: RouteCandidate | None) -> RouteStepCandidate | None:
    if route is None:
        return None
    for step in route.steps:
        if is_plugin_injected_enzyme_step(step):
            return step
    return None


def first_native_classified_enzyme_step(route: RouteCandidate | None) -> RouteStepCandidate | None:
    if route is None:
        return None
    for step in route.steps:
        if is_native_classified_enzyme_step(step):
            return step
    return None


def is_plugin_injected_enzyme_step(step: RouteStepCandidate) -> bool:
    return str(step.source_model or "") == PLUGIN_MODEL_FULL_NAME


def is_plugin_injected_chemical_step(step: RouteStepCandidate) -> bool:
    template = (step.raw_backend_metadata or {}).get("template")
    return isinstance(template, dict) and bool(template.get("autoplanner_native_chemical_plugin"))


def is_native_classified_enzyme_step(step: RouteStepCandidate) -> bool:
    return bool(step.enzyme_ec_annotations) and not is_plugin_injected_enzyme_step(step)


def has_any_enzyme_step(route: RouteCandidate) -> bool:
    return any(
        is_plugin_injected_enzyme_step(step)
        or step.has_enzyme_annotation
        or "enzyme" in str(step.source_model).lower()
        for step in route.steps
    )


def has_plugin_injected_enzyme_step(route: RouteCandidate) -> bool:
    return any(is_plugin_injected_enzyme_step(step) for step in route.steps)


def has_plugin_injected_chemical_step(route: RouteCandidate) -> bool:
    return any(is_plugin_injected_chemical_step(step) for step in route.steps)


def has_native_classified_enzyme_step(route: RouteCandidate) -> bool:
    return any(is_native_classified_enzyme_step(step) for step in route.steps)


def source_sequence(route: RouteCandidate | None) -> list[str]:
    if route is None:
        return []
    return [str(step.source_model or "") for step in route.steps]


def route_signature(route: RouteCandidate | None) -> str:
    if route is None:
        return ""
    return "|".join(step.rxn_smiles for step in route.steps)


def route_step_count(route: RouteCandidate | None) -> int:
    return len(route.steps) if route is not None else 0


def any_enzyme_route_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes if has_any_enzyme_step(route))


def any_enzyme_step_count(routes: list[RouteCandidate]) -> int:
    return sum(
        1
        for route in routes
        for step in route.steps
        if is_plugin_injected_enzyme_step(step)
        or step.has_enzyme_annotation
        or "enzyme" in str(step.source_model).lower()
    )


def native_classified_enzyme_route_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes if has_native_classified_enzyme_step(route))


def native_classified_enzyme_step_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes for step in route.steps if is_native_classified_enzyme_step(step))


def plugin_injected_enzyme_route_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes if has_plugin_injected_enzyme_step(route))


def plugin_injected_enzyme_step_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes for step in route.steps if is_plugin_injected_enzyme_step(step))


def plugin_injected_chemical_route_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes if has_plugin_injected_chemical_step(route))


def plugin_injected_chemical_step_count(routes: list[RouteCandidate]) -> int:
    return sum(1 for route in routes for step in route.steps if is_plugin_injected_chemical_step(step))


def audit_plugin_representative(route: RouteCandidate | None) -> dict[str, Any]:
    if route is None:
        return {"available": False, "reason": "no_representative_route"}
    injected_steps = [step for step in route.steps if is_plugin_injected_enzyme_step(step)]
    if not injected_steps:
        return {"available": False, "reason": "representative_has_no_plugin_injected_step"}
    route_audit = audit_route_plausibility(route)
    step_audits = [audit_injected_step(step) for step in injected_steps]
    return {
        "available": True,
        "route_plausibility_passed": bool(route_audit.get("passed")),
        "route_plausibility_reasons": list(route_audit.get("reasons") or []),
        "injected_step_count": len(injected_steps),
        "injected_steps": step_audits,
        "contract": route_audit.get("contract"),
    }


def audit_injected_step(step: RouteStepCandidate) -> dict[str, Any]:
    template = (step.raw_backend_metadata or {}).get("template")
    template = template if isinstance(template, dict) else {}
    evidence = template.get("evidence") if isinstance(template.get("evidence"), dict) else {}
    sp_payload = (
        template.get("enzyme_sp_verifier_v1") if isinstance(template.get("enzyme_sp_verifier_v1"), dict) else {}
    )
    material_audit = audit_step_plausibility(step)
    return {
        "rxn_smiles": step.rxn_smiles,
        "ec_numbers": [row.get("ec_number") for row in step.enzyme_ec_annotations],
        "score": step.score,
        "sp_v1": sp_payload,
        "transition_signature": evidence.get("transition_signature"),
        "precedent_reaction_ids": compact_list(evidence.get("reaction_ids") or evidence.get("precedent_reaction_ids")),
        "precedent_similarities": compact_list(evidence.get("similarities") or evidence.get("precedent_similarities")),
        "material_plausibility": {
            "passed": bool(material_audit.get("passed")),
            "reasons": list(material_audit.get("reasons") or []),
            "heavy_atom_gain": material_audit.get("heavy_atom_gain"),
            "carbon_gain": material_audit.get("carbon_gain"),
            "hetero_atom_gain": material_audit.get("hetero_atom_gain"),
            "unexplained_element_gains": material_audit.get("unexplained_element_gains"),
        },
    }


def compact_list(value: Any, *, limit: int = 6) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [value]
    if isinstance(value, dict):
        return list(value.values())[:limit]
    try:
        return list(value)[:limit]
    except TypeError:
        return [str(value)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    summary = {
        "targets": n,
        "native_solved": sum(1 for row in rows if row["native"]["solved"]),
        "plugin_solved": sum(1 for row in rows if row["plugin"]["solved"]),
        "native_total_routes": sum(int(row["native"]["route_count"]) for row in rows),
        "plugin_total_routes": sum(int(row["plugin"]["route_count"]) for row in rows),
        "native_native_classified_enzyme_routes": sum(
            int(row["native"]["native_classified_enzyme_route_count"]) for row in rows
        ),
        "plugin_native_classified_enzyme_routes": sum(
            int(row["plugin"]["native_classified_enzyme_route_count"]) for row in rows
        ),
        "plugin_injected_enzyme_routes": sum(int(row["plugin"]["plugin_injected_enzyme_route_count"]) for row in rows),
        "native_any_enzyme_routes": sum(int(row["native"]["any_enzyme_route_count"]) for row in rows),
        "plugin_any_enzyme_routes": sum(int(row["plugin"]["any_enzyme_route_count"]) for row in rows),
        "native_targets_with_native_classified_enzyme_route": sum(
            1 for row in rows if int(row["native"]["native_classified_enzyme_route_count"]) > 0
        ),
        "plugin_targets_with_native_classified_enzyme_route": sum(
            1 for row in rows if int(row["plugin"]["native_classified_enzyme_route_count"]) > 0
        ),
        "plugin_targets_with_injected_enzyme_route": sum(
            1 for row in rows if int(row["plugin"]["plugin_injected_enzyme_route_count"]) > 0
        ),
        "plugin_targets_with_selected_enzyme_route": sum(1 for row in rows if row["delta"]["plugin_selected_enzyme_route"]),
        "plugin_targets_with_selected_injected_enzyme_route": sum(
            1 for row in rows if row["delta"]["plugin_selected_injected_enzyme_route"]
        ),
        "plugin_targets_with_added_candidates": sum(1 for row in rows if int((row.get("plugin_stats") or {}).get("added_candidates") or 0) > 0),
        "plugin_added_candidates": sum(int((row.get("plugin_stats") or {}).get("added_candidates") or 0) for row in rows),
        "plugin_sp_v1_accepted": sum(int((row.get("plugin_stats") or {}).get("sp_v1_accepted") or 0) for row in rows),
        "plugin_sp_v1_rejected": sum(int((row.get("plugin_stats") or {}).get("sp_v1_rejected") or 0) for row in rows),
        "plugin_quality_passed": sum(int((row.get("plugin_stats") or {}).get("quality_passed") or 0) for row in rows),
        "plugin_quality_warned": sum(int((row.get("plugin_stats") or {}).get("quality_warned") or 0) for row in rows),
        "plugin_quality_rejected": sum(int((row.get("plugin_stats") or {}).get("quality_rejected") or 0) for row in rows),
        "plugin_material_rejected": sum(
            int((row.get("plugin_stats") or {}).get("material_rejected") or 0) for row in rows
        ),
        "plugin_errors": sum(int((row.get("plugin_stats") or {}).get("error_count") or 0) for row in rows),
        "plugin_audit_available": sum(1 for row in rows if (row.get("plugin_representative_audit") or {}).get("available")),
        "plugin_audit_route_plausibility_passed": sum(
            1 for row in rows if (row.get("plugin_representative_audit") or {}).get("route_plausibility_passed")
        ),
        "native_plausible_routes": sum(
            int(((row.get("native") or {}).get("route_plausibility") or {}).get("passed") or 0) for row in rows
        ),
        "plugin_plausible_routes": sum(
            int(((row.get("plugin") or {}).get("route_plausibility") or {}).get("passed") or 0) for row in rows
        ),
        "plugin_plausible_injected_enzyme_routes": sum(
            int(
                ((((row.get("plugin") or {}).get("route_plausibility") or {}).get("injected_enzyme_routes") or {}).get("passed"))
                or 0
            )
            for row in rows
        ),
    }
    combo_rows = [row for row in rows if row.get("combo")]
    if combo_rows:
        summary.update(
            {
                "combo_solved": sum(1 for row in combo_rows if row["combo"]["solved"]),
                "combo_total_routes": sum(int(row["combo"]["route_count"]) for row in combo_rows),
                "combo_injected_enzyme_routes": sum(
                    int(row["combo"]["plugin_injected_enzyme_route_count"]) for row in combo_rows
                ),
                "combo_injected_chemical_routes": sum(
                    plugin_injected_chemical_route_count_from_payload(row["combo"]) for row in combo_rows
                ),
                "combo_targets_with_injected_enzyme_route": sum(
                    1 for row in combo_rows if int(row["combo"]["plugin_injected_enzyme_route_count"]) > 0
                ),
                "combo_targets_with_injected_chemical_route": sum(
                    1 for row in combo_rows if plugin_injected_chemical_route_count_from_payload(row["combo"]) > 0
                ),
                "combo_targets_with_selected_injected_enzyme_route": sum(
                    1 for row in combo_rows if (row.get("combo_delta") or {}).get("combo_selected_injected_enzyme_route")
                ),
                "combo_targets_with_selected_injected_chemical_route": sum(
                    1 for row in combo_rows if (row.get("combo_delta") or {}).get("combo_selected_native_chemical_route")
                ),
                "combo_enzyme_added_candidates": sum(
                    int((row.get("combo_plugin_stats") or {}).get("added_candidates") or 0) for row in combo_rows
                ),
                "combo_chemical_added_candidates": sum(
                    int((row.get("combo_chemical_plugin_stats") or {}).get("added_candidates") or 0)
                    for row in combo_rows
                ),
                "combo_chemical_gate_kept": sum(
                    int((row.get("combo_chemical_plugin_stats") or {}).get("proposal_gate_kept") or 0)
                    for row in combo_rows
                ),
                "combo_chemical_gate_rejected": sum(
                    int((row.get("combo_chemical_plugin_stats") or {}).get("proposal_gate_rejected") or 0)
                    for row in combo_rows
                ),
                "combo_chemical_errors": sum(
                    int((row.get("combo_chemical_plugin_stats") or {}).get("error_count") or 0) for row in combo_rows
                ),
                "combo_plausible_routes": sum(
                    int(((row.get("combo") or {}).get("route_plausibility") or {}).get("passed") or 0)
                    for row in combo_rows
                ),
                "combo_plausible_injected_enzyme_routes": sum(
                    int(
                        ((((row.get("combo") or {}).get("route_plausibility") or {}).get("injected_enzyme_routes") or {}).get("passed"))
                        or 0
                    )
                    for row in combo_rows
                ),
                "combo_plausible_injected_chemical_routes": sum(
                    int(
                        ((((row.get("combo") or {}).get("route_plausibility") or {}).get("injected_chemical_routes") or {}).get("passed"))
                        or 0
                    )
                    for row in combo_rows
                ),
            }
        )
    return summary


def plugin_injected_chemical_route_count_from_payload(summary: dict[str, Any]) -> int:
    return int(summary.get("plugin_injected_chemical_route_count") or 0)


def write_route_dump_jsonl(
    path: Path,
    *,
    targets: list[dict[str, Any]],
    native_by_target: dict[str, BaselineRunResult],
    plugin_by_target: dict[str, BaselineRunResult],
    combo_by_target: dict[str, BaselineRunResult],
) -> None:
    run_maps: list[tuple[str, dict[str, BaselineRunResult]]] = [
        ("native", native_by_target),
        ("native_enzyme_plugin", plugin_by_target),
    ]
    if combo_by_target:
        run_maps.append(("native_enzyme_chemical_plugin", combo_by_target))
    lines = []
    for target in targets:
        target_smiles = str(target["target_smiles"])
        for run_name, result_by_target in run_maps:
            result = result_by_target.get(target_smiles)
            if result is None:
                continue
            for route in result.routes:
                audit = audit_route_plausibility(route)
                lines.append(
                    json.dumps(
                        {
                            "run": run_name,
                            "target": target["safe"],
                            "target_smiles": target_smiles,
                            "route_rank": route.route_rank,
                            "solved": route.solved,
                            "score": route.score,
                            "has_injected_enzyme_step": has_plugin_injected_enzyme_step(route),
                            "has_injected_chemical_step": has_plugin_injected_chemical_step(route),
                            "route_plausibility": audit,
                            "route": route.to_dict(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    path.write_text("".join(lines), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Statin Native Enzyme Plugin Comparison",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Per Target",
        "",
        "| target | native solved/routes | plugin solved/routes | plugin enzyme routes | combo solved/routes | combo enzyme routes | combo chemical routes | enzyme added | chemical added | audit | representative difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["targets"]:
        stats = row.get("plugin_stats") or {}
        audit = row.get("plugin_representative_audit") or {}
        audit_text = audit_status(audit)
        lines.append(
            (
                "| {name} | {ns}/{nr} | {ps}/{pr} | {pinj} | {cs}/{cr} | {cinj} | {cchem} | "
                "{added} | {chem_added} | {audit} | {desc} |"
            ).format(
                name=row["name"],
                ns=int(bool(row["native"]["solved"])),
                nr=row["native"]["route_count"],
                ps=int(bool(row["plugin"]["solved"])),
                pr=row["plugin"]["route_count"],
                pinj=row["plugin"]["plugin_injected_enzyme_route_count"],
                cs=int(bool((row.get("combo") or row["plugin"])["solved"])),
                cr=(row.get("combo") or row["plugin"])["route_count"],
                cinj=(row.get("combo") or row["plugin"])["plugin_injected_enzyme_route_count"],
                cchem=(row.get("combo") or {}).get("plugin_injected_chemical_route_count", 0),
                added=int(stats.get("added_candidates") or 0),
                chem_added=int((row.get("combo_chemical_plugin_stats") or {}).get("added_candidates") or 0),
                audit=audit_text.replace("|", "/"),
                desc=short_difference(row).replace("|", "/"),
            )
        )
    lines.extend(["", "## Representative Route Details", ""])
    for row in report["targets"]:
        diff = row.get("representative_difference") or {}
        lines.extend([
            f"### {row['name']}",
            "",
            f"- native source sequence: `{', '.join(diff.get('native_source_sequence') or []) or 'none'}`",
            f"- plugin source sequence: `{', '.join(diff.get('plugin_source_sequence') or []) or 'none'}`",
        ])
        injected = diff.get("plugin_first_injected_enzyme_step")
        classified = diff.get("plugin_first_native_classified_enzyme_step")
        if injected:
            lines.append(
                f"- plugin injected enzyme step: `{injected.get('source_model')}` EC={injected.get('ec_numbers')} `{injected.get('rxn_smiles')}`"
            )
        else:
            lines.append("- plugin injected enzyme step: none selected")
        if classified:
            lines.append(
                f"- ChemEnzy-classified enzyme step: `{classified.get('source_model')}` EC={classified.get('ec_numbers')} `{classified.get('rxn_smiles')}`"
            )
        audit = row.get("plugin_representative_audit") or {}
        if audit.get("available"):
            lines.append(
                f"- injected-route audit: passed={audit.get('route_plausibility_passed')} reasons={audit.get('route_plausibility_reasons')}"
            )
            for idx, step_audit in enumerate(audit.get("injected_steps") or [], start=1):
                material = step_audit.get("material_plausibility") or {}
                lines.append(
                    f"- injected step {idx} audit: EC={step_audit.get('ec_numbers')} "
                    f"SP accepted={(step_audit.get('sp_v1') or {}).get('accepted')} "
                    f"material_passed={material.get('passed')} material_reasons={material.get('reasons')}"
                )
        lines.append("")
    return "\n".join(lines)


def short_difference(row: dict[str, Any]) -> str:
    if not row["delta"]["representative_changed"]:
        return "same representative route"
    if row["delta"]["plugin_selected_injected_enzyme_route"]:
        return "plugin representative switches to injected enzyme route"
    if row["delta"]["plugin_selected_native_classified_enzyme_route"]:
        return "plugin representative switches to ChemEnzy-classified enzyme route"
    if row["delta"]["route_count_delta"]:
        return "route pool size changed; representative remains chemical"
    return "representative chemical route differs"


def audit_status(audit: dict[str, Any]) -> str:
    if not audit.get("available"):
        return str(audit.get("reason") or "not available")
    if audit.get("route_plausibility_passed"):
        return "material audit passed"
    reasons = ",".join(str(item) for item in audit.get("route_plausibility_reasons") or [])
    return f"material audit failed: {reasons or 'unknown'}"


if __name__ == "__main__":
    main()
