"""Benchmark chemical closure of accepted enzymatic-step precursors.

This script consumes route rows from ``run_native_vs_enhanced_route_benchmark``,
extracts selected enzyme steps accepted by SP-v1, and reruns route-tree search
on their upstream precursors with chemical proposal sources only.
"""
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

from cascade_planner.cascadeboard.live_retro import build_chemical_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.baselines.chem_enzy_adapter import ChemEnzyBackendAdapter, DEFAULT_ONE_STEP_MODELS, DEFAULT_STOCKS
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import SourceGate
from scripts.run_native_vs_enhanced_route_benchmark import (
    ENZYME_SOURCES,
    build_enhanced_stock_checker,
    dict_count,
    mean,
    route_tree_payload,
    temporary_env,
)


DEFAULT_ROUTE_ROWS = Path(
    "results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_sptrace/"
    "native_vs_enhanced_route_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/shared/enzyme_precursor_expansion_benchmark_20260528")
CHEMICAL_ENGINE_SOURCES = {"retrochimera", "chemtemplates", "chem_enzy_onestep"}
TRIVIAL_AUXILIARY_SMILES = {
    "",
    "O",
    "[H]O[H]",
    "O=O",
    "[O][O]",
    "[H][H]",
    "N",
    "[NH4+]",
    "Cl",
    "[Cl-]",
    "[Na+]",
    "[K+]",
}
CARRIER_FRAGMENT_MARKERS = (
    "n2cnc3c(N)ncnc32",
    "n1cnc2c(N)ncnc12",
    "P(=O)([O-])OP(=O)",
    "OP(=O)([O-])OP(=O)",
    "NCCC(=O)NCCSC(=O)",
    "NCCSC(=O)",
    "C(C)(COP(=O)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-rows",
        type=Path,
        nargs="+",
        default=[DEFAULT_ROUTE_ROWS],
        help="One or more native-vs-enhanced route rows JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-subgoals", type=int, default=0, help="0 means all extracted subgoals.")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--branch-factor", type=int, default=6)
    parser.add_argument("--expansion-budget", type=int, default=20)
    parser.add_argument("--n-results", type=int, default=2)
    parser.add_argument("--route-tree-timeout-s", type=float, default=45.0)
    parser.add_argument("--enhanced-stock", choices=("none", "zinc"), default="zinc")
    parser.add_argument("--include-aux", action="store_true", help="Also expand nontrivial aux reactants.")
    parser.add_argument("--no-dedupe-subgoals", action="store_true")
    parser.add_argument(
        "--include-carrier-like",
        action="store_true",
        help="Also expand CoA/NAD/ATP/nucleotide/polyphosphate-like accepted enzyme precursors.",
    )
    parser.add_argument("--enable-chem-enzy-onestep", action="store_true")
    parser.add_argument("--chem-enzy-onestep-topk", type=int, default=20)
    parser.add_argument("--chem-enzy-onestep-min-budget", type=int, default=3)
    parser.add_argument(
        "--native-subplanner",
        action="store_true",
        help="Also run native ChemEnzy on each accepted enzyme precursor as a chemical subplanner.",
    )
    parser.add_argument("--native-subplanner-iterations", type=int, default=80)
    parser.add_argument("--native-subplanner-max-depth", type=int, default=0, help="0 uses --max-depth.")
    parser.add_argument("--native-subplanner-expansion-topk", type=int, default=50)
    parser.add_argument("--native-subplanner-gpu", type=int, default=-1)
    parser.add_argument(
        "--native-subplanner-models",
        default=",".join(DEFAULT_ONE_STEP_MODELS),
        help="Comma-separated ChemEnzy one-step models for native precursor subplanning.",
    )
    parser.add_argument(
        "--stock-aware-action-rerank",
        action="store_true",
        help="Prefer actions whose reactants exactly hit stock during chemical closure.",
    )
    parser.add_argument("--exact-stock-reactant-bonus", type=float, default=1.0)
    parser.add_argument("--full-stock-action-bonus", type=float, default=2.0)
    parser.add_argument(
        "--normalized-stock-reactant-bonus",
        type=float,
        default=0.0,
        help="Selection bonus for reactants whose neutralized form hits stock; does not change strict stock solve.",
    )
    parser.add_argument(
        "--normalized-stock-full-action-bonus",
        type=float,
        default=0.0,
        help="Additional selection bonus when all reactants are exact or normalized stock hits.",
    )
    parser.add_argument(
        "--no-progress-single-reactant-penalty",
        type=float,
        default=0.0,
        help="Selection penalty for one-reactant steps with no heavy-atom progress and no stock closure.",
    )
    parser.add_argument(
        "--closure-stock-rescue",
        action="store_true",
        help="Enable early stock-rescue retry for accepted enzyme precursor closure.",
    )
    parser.add_argument(
        "--closure-stock-rescue-remaining-depth",
        type=int,
        default=0,
        help="Remaining-depth threshold for stock rescue; 0 uses --max-depth so root precursor nodes can retry.",
    )
    parser.add_argument("--closure-stock-rescue-max-retries", type=int, default=12)
    parser.add_argument("--closure-stock-rescue-budget-multiplier", type=float, default=3.0)
    parser.add_argument("--closure-stock-rescue-budget-cap", type=int, default=30)
    parser.add_argument("--closure-stock-rescue-min-actions", type=int, default=12)
    parser.add_argument(
        "--closure-stock-rescue-require-stock-gain",
        action="store_true",
        help="Reject rescue retry candidates unless terminal fraction improves.",
    )
    parser.add_argument(
        "--chem-template-max-per-query",
        type=int,
        default=0,
        help="Set AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY during closure search; 0 preserves the environment/default.",
    )
    parser.add_argument(
        "--chem-template-max-templates",
        type=int,
        default=0,
        help="Set AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES during closure search; 0 preserves the environment/default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent_rows = load_route_rows(args.route_rows)
    stock_checker = build_enhanced_stock_checker(args)
    subgoals = extract_accepted_enzyme_precursors(
        parent_rows,
        include_aux=bool(args.include_aux),
        dedupe=not bool(args.no_dedupe_subgoals),
        include_carrier_like=bool(args.include_carrier_like),
    )
    if args.max_subgoals > 0:
        subgoals = subgoals[: int(args.max_subgoals)]

    rows: list[dict[str, Any]] = []
    with temporary_env(**closure_env_values(args)):
        chemical_engine = chemical_only_engine(build_chemical_retro_engine())
        for idx, subgoal in enumerate(subgoals, start=1):
            print(
                f"[{idx}/{len(subgoals)}] chemical closure for {subgoal.get('subgoal_smiles', '')}",
                flush=True,
            )
            rows.append(run_subgoal_expansion(subgoal, chemical_engine=chemical_engine, stock_checker=stock_checker, args=args))
    native_rows: list[dict[str, Any]] = []
    if bool(args.native_subplanner):
        native_rows = run_native_subplanner_for_subgoals(subgoals, args=args)

    report = {
        "schema_version": "enzyme_precursor_expansion_benchmark.v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "route_rows": [str(path) for path in args.route_rows],
            "parent_rows": len(parent_rows),
            "extracted_subgoals": len(subgoals),
            "max_depth": int(args.max_depth),
            "branch_factor": int(args.branch_factor),
            "expansion_budget": int(args.expansion_budget),
            "n_results": int(args.n_results),
            "route_tree_timeout_s": float(args.route_tree_timeout_s),
            "enhanced_stock": str(args.enhanced_stock),
            "include_aux": bool(args.include_aux),
            "dedupe_subgoals": not bool(args.no_dedupe_subgoals),
            "include_carrier_like": bool(args.include_carrier_like),
            "enable_chem_enzy_onestep": bool(args.enable_chem_enzy_onestep),
            "chem_enzy_onestep_topk": int(args.chem_enzy_onestep_topk),
            "chem_enzy_onestep_min_budget": int(args.chem_enzy_onestep_min_budget),
            "native_subplanner": bool(args.native_subplanner),
            "native_subplanner_iterations": int(args.native_subplanner_iterations),
            "native_subplanner_max_depth": int(args.native_subplanner_max_depth or args.max_depth),
            "native_subplanner_expansion_topk": int(args.native_subplanner_expansion_topk),
            "native_subplanner_models": _csv_list(args.native_subplanner_models),
            "stock_aware_action_rerank": bool(args.stock_aware_action_rerank),
            "exact_stock_reactant_bonus": float(args.exact_stock_reactant_bonus),
            "full_stock_action_bonus": float(args.full_stock_action_bonus),
            "normalized_stock_reactant_bonus": float(args.normalized_stock_reactant_bonus),
            "normalized_stock_full_action_bonus": float(args.normalized_stock_full_action_bonus),
            "no_progress_single_reactant_penalty": float(args.no_progress_single_reactant_penalty),
            "closure_stock_rescue": bool(args.closure_stock_rescue),
            "closure_stock_rescue_remaining_depth": int(args.closure_stock_rescue_remaining_depth),
            "closure_stock_rescue_max_retries": int(args.closure_stock_rescue_max_retries),
            "closure_stock_rescue_budget_multiplier": float(args.closure_stock_rescue_budget_multiplier),
            "closure_stock_rescue_budget_cap": int(args.closure_stock_rescue_budget_cap),
            "closure_stock_rescue_min_actions": int(args.closure_stock_rescue_min_actions),
            "closure_stock_rescue_require_stock_gain": bool(args.closure_stock_rescue_require_stock_gain),
            "chem_template_max_per_query": int(args.chem_template_max_per_query),
            "chem_template_max_templates": int(args.chem_template_max_templates),
            "chemical_engine_sources": sorted(chemical_engine),
        },
        "summary": summarize_subgoal_rows(rows),
        "native_subplanner_summary": summarize_native_subplanner_rows(native_rows),
        "hybrid_summary": summarize_hybrid_rows(rows, native_rows),
        "conclusion": conclusion(summarize_subgoal_rows(rows)),
        "hybrid_conclusion": hybrid_conclusion(summarize_subgoal_rows(rows), summarize_native_subplanner_rows(native_rows)),
        "subgoal_outcomes": subgoal_outcomes(rows),
        "native_subplanner_outcomes": native_subplanner_outcomes(native_rows),
        "rows_jsonl": str(args.output_dir / "enzyme_precursor_expansion_rows.jsonl"),
        "native_subplanner_rows_jsonl": str(args.output_dir / "enzyme_precursor_native_subplanner_rows.jsonl"),
        "live_engine_cache_stats": retro_engine_cache_stats(chemical_engine),
    }
    rows_path = args.output_dir / "enzyme_precursor_expansion_rows.jsonl"
    native_rows_path = args.output_dir / "enzyme_precursor_native_subplanner_rows.jsonl"
    report_json = args.output_dir / "enzyme_precursor_expansion_report.json"
    report_md = args.output_dir / "enzyme_precursor_expansion_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    native_rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in native_rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "rows": str(rows_path), "conclusion": report["conclusion"]}, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_route_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in load_jsonl(path):
            row.setdefault("_route_rows_path", str(path))
            rows.append(row)
    return rows


def closure_env_values(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS": "1" if args.enable_chem_enzy_onestep else "0",
        "AUTOPLANNER_CHEMENZY_ONESTEP_TOPK": str(max(1, int(args.chem_enzy_onestep_topk))),
        "AUTOPLANNER_CHEMENZY_ONESTEP_MIN_BUDGET": str(max(1, int(args.chem_enzy_onestep_min_budget))),
        "AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL": "0",
        "AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL": "0",
        "AUTOPLANNER_ROUTE_TREE_BRIDGE_EC_CONTEXT_PROPOSALS": "0",
        "AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS": "0",
        "AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS": "",
        "AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S": str(max(1.0, float(args.route_tree_timeout_s))),
        "AUTOPLANNER_ROUTE_TREE_SOFT_TIMEOUT_S": str(max(1.0, float(args.route_tree_timeout_s) * 0.75)),
        "AUTOPLANNER_ENZYME_SP_VERIFIER_V1_REJECT_BELOW_THRESHOLD": "0",
        "AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS": (
            str(max(0.0, float(args.exact_stock_reactant_bonus))) if args.stock_aware_action_rerank else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS": (
            str(max(0.0, float(args.full_stock_action_bonus))) if args.stock_aware_action_rerank else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_REACTANT_BONUS": (
            str(max(0.0, float(getattr(args, "normalized_stock_reactant_bonus", 0.0))))
            if args.stock_aware_action_rerank
            else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_FULL_ACTION_BONUS": (
            str(max(0.0, float(getattr(args, "normalized_stock_full_action_bonus", 0.0))))
            if args.stock_aware_action_rerank
            else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_NO_PROGRESS_SINGLE_REACTANT_PENALTY": str(
            max(0.0, float(getattr(args, "no_progress_single_reactant_penalty", 0.0)))
        ),
    }
    if bool(args.closure_stock_rescue):
        remaining_depth = int(args.closure_stock_rescue_remaining_depth or 0)
        if remaining_depth <= 0:
            remaining_depth = max(0, int(args.max_depth))
        values.update(
            {
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE": "1",
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REMAINING_DEPTH": str(max(0, remaining_depth)),
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MAX_RETRIES": str(
                    max(0, int(args.closure_stock_rescue_max_retries))
                ),
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_MULTIPLIER": str(
                    max(1.0, float(args.closure_stock_rescue_budget_multiplier))
                ),
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_BUDGET_CAP": str(
                    max(1, int(args.closure_stock_rescue_budget_cap))
                ),
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_MIN_ACTIONS": str(
                    max(0, int(args.closure_stock_rescue_min_actions))
                ),
                "AUTOPLANNER_ROUTE_TREE_LATE_STOCK_RESCUE_REQUIRE_STOCK_GAIN": (
                    "1" if args.closure_stock_rescue_require_stock_gain else "0"
                ),
            }
        )
    if int(args.chem_template_max_per_query or 0) > 0:
        values["AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY"] = str(max(1, int(args.chem_template_max_per_query)))
    if int(args.chem_template_max_templates or 0) > 0:
        values["AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES"] = str(max(1, int(args.chem_template_max_templates)))
    return values


def extract_accepted_enzyme_precursors(
    rows: list[dict[str, Any]],
    *,
    include_aux: bool = False,
    dedupe: bool = True,
    include_carrier_like: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent_idx, parent in enumerate(rows):
        routes = parent.get("routes") or []
        for route_idx, route in enumerate(routes):
            for step_idx, step in enumerate(route.get("steps") or []):
                if not _is_sp_v1_accepted_enzyme_step(step):
                    continue
                for component_role, component_smiles in _step_components(step, include_aux=include_aux):
                    canonical = canonical_smiles(component_smiles) or str(component_smiles or "")
                    if not canonical:
                        continue
                    carrier_payload = carrier_like_payload(canonical)
                    if carrier_payload["carrier_like"] and not include_carrier_like:
                        continue
                    key = canonical if dedupe else f"{parent_idx}:{route_idx}:{step_idx}:{component_role}:{canonical}"
                    if key in seen:
                        continue
                    seen.add(key)
                    sp_payload = _step_sp_payload(step)
                    out.append(
                        {
                            "parent_target_smiles": str(parent.get("target_smiles") or ""),
                            "parent_target_canonical": str(parent.get("target_canonical") or ""),
                            "parent_label": int(parent.get("label") or 0),
                            "parent_label_source": str(parent.get("label_source") or ""),
                            "parent_run": str(parent.get("run") or ""),
                            "parent_route_rows_path": str(parent.get("_route_rows_path") or ""),
                            "parent_route_index": int(route_idx),
                            "parent_step_index": int(step_idx),
                            "parent_route_status": str(route.get("route_tree_search_status") or ""),
                            "enzyme_step_source": str(step.get("source") or ""),
                            "enzyme_step_ec": str(step.get("ec") or ""),
                            "enzyme_step_product": str(step.get("product") or ""),
                            "enzyme_step_reaction_smiles": str(step.get("reaction_smiles") or ""),
                            "enzyme_sp_v1_score": _safe_float(sp_payload.get("score")),
                            "enzyme_sp_v1_threshold": _safe_float(sp_payload.get("threshold")),
                            "component_role": component_role,
                            "subgoal_smiles": str(component_smiles or ""),
                            "subgoal_canonical": canonical,
                            "carrier_like": bool(carrier_payload["carrier_like"]),
                            "carrier_like_reasons": list(carrier_payload["reasons"]),
                        }
                    )
    return out


def carrier_like_payload(smiles: str) -> dict[str, Any]:
    text = str(smiles or "")
    reasons: list[str] = []
    if any(marker in text for marker in CARRIER_FRAGMENT_MARKERS):
        reasons.append("known_carrier_fragment")
    phosphate_count = text.count("P(=O)")
    if phosphate_count >= 2:
        reasons.append("polyphosphate")
    if "ncnc" in text and phosphate_count >= 1:
        reasons.append("nucleotide_phosphate")
    if "NCCSC(=O)" in text or "NCCC(=O)NCCSC(=O)" in text:
        reasons.append("coa_thioester_motif")
    return {"carrier_like": bool(reasons), "reasons": sorted(set(reasons))}


def chemical_only_engine(engine: dict[str, Any]) -> dict[str, Any]:
    return {
        source: model
        for source, model in (engine or {}).items()
        if source in CHEMICAL_ENGINE_SOURCES and model is not None
    }


def run_native_subplanner_for_subgoals(subgoals: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not subgoals:
        return []
    configs = [
        RouteSearchConfig(
            target_smiles=str(subgoal.get("subgoal_smiles") or ""),
            stock_names=list(DEFAULT_STOCKS),
            max_iterations=max(1, int(args.native_subplanner_iterations)),
            max_depth=max(1, int(args.native_subplanner_max_depth or args.max_depth)),
            expansion_topk=max(1, int(args.native_subplanner_expansion_topk)),
            one_step_models=_csv_list(args.native_subplanner_models) or list(DEFAULT_ONE_STEP_MODELS),
            search_flags={"gpu": int(args.native_subplanner_gpu), "keep_search": False},
        )
        for subgoal in subgoals
    ]
    adapter = ChemEnzyBackendAdapter(gpu=int(args.native_subplanner_gpu))
    started = time.monotonic()
    results = adapter.run_targets(configs, reuse_planner=True)
    rows: list[dict[str, Any]] = []
    for idx, (subgoal, result) in enumerate(zip(subgoals, results), start=1):
        print(
            f"[{idx}/{len(subgoals)}] native ChemEnzy subplanner for {subgoal.get('subgoal_smiles', '')}",
            flush=True,
        )
        rows.append(native_subplanner_row(subgoal, result, elapsed_total_s=time.monotonic() - started))
    return rows


def native_subplanner_row(
    subgoal: dict[str, Any],
    result: BaselineRunResult,
    *,
    elapsed_total_s: float,
) -> dict[str, Any]:
    failures = [failure.to_dict() for failure in result.failures]
    routes = [route.to_dict() for route in result.routes]
    return {
        **subgoal,
        "native_ok": bool(result.routes),
        "native_solved": bool(result.solved),
        "native_route_count": int(result.route_count),
        "native_failures": failures,
        "native_failure_categories": [str(item.get("category") or "") for item in failures],
        "native_routes": routes,
        "native_mean_steps": mean(len(route.get("steps") or []) for route in routes),
        "native_elapsed_s": (result.raw_backend_metadata or {}).get("elapsed_s"),
        "native_total_elapsed_s": (result.raw_backend_metadata or {}).get("total_elapsed_s"),
        "native_batch_elapsed_s": round(float(elapsed_total_s), 3),
    }


def run_subgoal_expansion(
    subgoal: dict[str, Any],
    *,
    chemical_engine: dict[str, Any],
    stock_checker: Any | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subgoal_smiles = str(subgoal.get("subgoal_smiles") or "")
    started = time.monotonic()
    initially_in_stock = bool(stock_checker and stock_checker(subgoal_smiles))
    if initially_in_stock:
        return {
            **subgoal,
            "ok": True,
            "subgoal_in_stock_initial": True,
            "route_count": 0,
            "solved_routes": 0,
            "progressive_routes": 0,
            "enzyme_routes": 0,
            "mean_steps": 0.0,
            "elapsed_s": round(time.monotonic() - started, 3),
            "stats": {"search_stop_reason": "initial_stock"},
            "routes": [],
        }
    planner = NeuralGuidedAOSearch(
        retro_engine=chemical_engine,
        stock_checker=stock_checker,
        max_depth=max(1, int(args.max_depth)),
        branch_factor=max(1, int(args.branch_factor)),
        expansion_budget=max(1, int(args.expansion_budget)),
        controller=None,
        enzyme_sp_verifier=None,
    )
    planner.proposals.source_gate = SourceGate()
    results = planner.search(subgoal_smiles, n_results=max(1, int(args.n_results)))
    stats = planner.stats.to_dict()
    routes = [route_tree_payload(result, stock_checker=stock_checker) for result in results]
    return {
        **subgoal,
        "ok": bool(routes),
        "subgoal_in_stock_initial": False,
        "route_count": len(routes),
        "solved_routes": sum(1 for route in routes if bool(route.get("route_solved"))),
        "progressive_routes": sum(1 for route in routes if bool(route.get("progressive_route"))),
        "enzyme_routes": sum(1 for route in routes if bool(route.get("has_enzyme_step"))),
        "mean_steps": mean(route.get("n_steps") for route in routes),
        "elapsed_s": round(time.monotonic() - started, 3),
        "stats": stats,
        "routes": routes,
    }


def summarize_subgoal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "subgoals": len(rows),
        "subgoals_initial_stock": sum(1 for row in rows if bool(row.get("subgoal_in_stock_initial"))),
        "subgoals_with_routes": sum(1 for row in rows if int(row.get("route_count") or 0) > 0),
        "subgoals_with_solved_route": sum(1 for row in rows if int(row.get("solved_routes") or 0) > 0),
        "subgoals_closed_by_stock_or_route": sum(
            1
            for row in rows
            if bool(row.get("subgoal_in_stock_initial")) or int(row.get("solved_routes") or 0) > 0
        ),
        "subgoals_with_progressive_route": sum(1 for row in rows if int(row.get("progressive_routes") or 0) > 0),
        "subgoals_with_timeout_frontier_route": sum(
            1
            for row in rows
            if any(str(route.get("route_tree_search_status") or "") == "timeout_frontier" for route in row.get("routes") or [])
        ),
        "timeout_frontier_routes": sum(
            1
            for row in rows
            for route in row.get("routes") or []
            if str(route.get("route_tree_search_status") or "") == "timeout_frontier"
        ),
        "route_count": sum(int(row.get("route_count") or 0) for row in rows),
        "solved_routes": sum(int(row.get("solved_routes") or 0) for row in rows),
        "progressive_routes": sum(int(row.get("progressive_routes") or 0) for row in rows),
        "mean_routes": mean(row.get("route_count") for row in rows),
        "mean_steps": mean(row.get("mean_steps") for row in rows if row.get("mean_steps") is not None),
        "mean_elapsed_s": mean(row.get("elapsed_s") for row in rows),
        "search_stop_reasons": dict_count((row.get("stats") or {}).get("search_stop_reason") for row in rows),
        "proposal_source_calls": _merge_source_metric(rows, metric="calls"),
        "proposal_source_outputs": _merge_source_outputs(rows),
        "proposal_source_candidates": _merge_source_metric(rows, metric="final_returned"),
        "selected_step_source_counts": _merge_route_source_counts(rows),
        "runtime_bottlenecks": dict_count(
            label
            for row in rows
            for label in (row.get("stats") or {}).get("route_tree_runtime_bottlenecks") or []
        ),
        "stock_rescue_retries": sum(int((row.get("stats") or {}).get("stock_rescue_retries") or 0) for row in rows),
        "stock_rescue_rejected": sum(int((row.get("stats") or {}).get("stock_rescue_rejected") or 0) for row in rows),
    }


def summarize_native_subplanner_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_categories: dict[str, int] = {}
    for row in rows:
        for category in row.get("native_failure_categories") or []:
            if not category:
                continue
            failure_categories[category] = failure_categories.get(category, 0) + 1
    return {
        "subgoals": len(rows),
        "native_subgoals_with_routes": sum(1 for row in rows if int(row.get("native_route_count") or 0) > 0),
        "native_subgoals_solved": sum(1 for row in rows if bool(row.get("native_solved"))),
        "native_total_routes": sum(int(row.get("native_route_count") or 0) for row in rows),
        "native_mean_routes": mean(row.get("native_route_count") for row in rows),
        "native_mean_steps": mean(row.get("native_mean_steps") for row in rows if row.get("native_mean_steps") is not None),
        "native_mean_elapsed_s": mean(row.get("native_elapsed_s") for row in rows if row.get("native_elapsed_s") is not None),
        "native_failure_categories": failure_categories,
    }


def summarize_hybrid_rows(route_tree_rows: list[dict[str, Any]], native_rows: list[dict[str, Any]]) -> dict[str, Any]:
    native_by_key = {str(row.get("subgoal_canonical") or ""): row for row in native_rows}
    closed = 0
    solved_by_route_tree = 0
    solved_by_native = 0
    for row in route_tree_rows:
        key = str(row.get("subgoal_canonical") or "")
        route_tree_solved = bool(row.get("subgoal_in_stock_initial")) or int(row.get("solved_routes") or 0) > 0
        native_solved = bool((native_by_key.get(key) or {}).get("native_solved"))
        if route_tree_solved:
            solved_by_route_tree += 1
        if native_solved:
            solved_by_native += 1
        if route_tree_solved or native_solved:
            closed += 1
    return {
        "subgoals": len(route_tree_rows),
        "hybrid_closed_by_route_tree_or_native": closed,
        "route_tree_solved_or_initial_stock": solved_by_route_tree,
        "native_subplanner_solved": solved_by_native,
        "native_rows": len(native_rows),
    }


def subgoal_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        routes = list(row.get("routes") or [])
        outcomes.append(
            {
                "index": idx,
                "subgoal_canonical": str(row.get("subgoal_canonical") or ""),
                "component_role": str(row.get("component_role") or ""),
                "carrier_like": bool(row.get("carrier_like")),
                "carrier_like_reasons": list(row.get("carrier_like_reasons") or []),
                "parent_label": int(row.get("parent_label") or 0),
                "enzyme_step_source": str(row.get("enzyme_step_source") or ""),
                "enzyme_step_ec": str(row.get("enzyme_step_ec") or ""),
                "enzyme_sp_v1_score": row.get("enzyme_sp_v1_score"),
                "subgoal_in_stock_initial": bool(row.get("subgoal_in_stock_initial")),
                "route_count": int(row.get("route_count") or 0),
                "solved_routes": int(row.get("solved_routes") or 0),
                "progressive_routes": int(row.get("progressive_routes") or 0),
                "mean_steps": row.get("mean_steps"),
                "elapsed_s": row.get("elapsed_s"),
                "search_stop_reason": str((row.get("stats") or {}).get("search_stop_reason") or ""),
                "stock_rescue_retries": int((row.get("stats") or {}).get("stock_rescue_retries") or 0),
                "stock_rescue_rejected": int((row.get("stats") or {}).get("stock_rescue_rejected") or 0),
                "runtime_bottlenecks": list((row.get("stats") or {}).get("route_tree_runtime_bottlenecks") or []),
                "route_statuses": dict_count(str(route.get("route_tree_search_status") or "unknown") for route in routes),
                "selected_step_source_counts": dict_count(
                    source
                    for route in routes
                    for source, count in (route.get("source_counts") or {}).items()
                    for _ in range(int(count or 0))
                ),
            }
        )
    return outcomes


def native_subplanner_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        failures = list(row.get("native_failure_categories") or [])
        outcomes.append(
            {
                "index": idx,
                "subgoal_canonical": str(row.get("subgoal_canonical") or ""),
                "component_role": str(row.get("component_role") or ""),
                "carrier_like": bool(row.get("carrier_like")),
                "parent_label": int(row.get("parent_label") or 0),
                "enzyme_step_source": str(row.get("enzyme_step_source") or ""),
                "enzyme_step_ec": str(row.get("enzyme_step_ec") or ""),
                "enzyme_sp_v1_score": row.get("enzyme_sp_v1_score"),
                "native_ok": bool(row.get("native_ok")),
                "native_solved": bool(row.get("native_solved")),
                "native_route_count": int(row.get("native_route_count") or 0),
                "native_mean_steps": row.get("native_mean_steps"),
                "native_elapsed_s": row.get("native_elapsed_s"),
                "native_total_elapsed_s": row.get("native_total_elapsed_s"),
                "native_failure_categories": failures,
            }
        )
    return outcomes


def conclusion(summary: dict[str, Any]) -> str:
    total = int(summary.get("subgoals") or 0)
    if total <= 0:
        return "No SP-v1 accepted enzymatic precursor was found in the input route rows."
    closed = int(summary.get("subgoals_closed_by_stock_or_route") or 0)
    solved = int(summary.get("subgoals_with_solved_route") or 0)
    progressive = int(summary.get("subgoals_with_progressive_route") or 0)
    with_routes = int(summary.get("subgoals_with_routes") or 0)
    if closed:
        return (
            f"Chemical-only expansion closed {closed}/{total} accepted enzyme precursors "
            f"({solved} by route, {summary.get('subgoals_initial_stock', 0)} initially in stock)."
        )
    if progressive:
        return (
            f"Chemical-only expansion made progressive partial routes for {progressive}/{total} accepted enzyme precursors, "
            "but did not close them to stock under this budget."
        )
    if with_routes:
        return (
            f"Chemical-only expansion returned partial routes for {with_routes}/{total} accepted enzyme precursors, "
            "but none were solved or progressive. The current bottleneck is upstream chemical closure after enzyme-step selection."
        )
    return (
        f"Chemical-only expansion returned no routes for {total} accepted enzyme precursors. "
        "The current bottleneck is chemical proposal coverage or route-tree budget after accepted enzyme-step selection."
    )


def hybrid_conclusion(route_tree_summary: dict[str, Any], native_summary: dict[str, Any]) -> str:
    total = int(route_tree_summary.get("subgoals") or native_summary.get("subgoals") or 0)
    if total <= 0:
        return "No accepted enzyme precursor was available for hybrid continuation."
    native_solved = int(native_summary.get("native_subgoals_solved") or 0)
    route_tree_closed = int(route_tree_summary.get("subgoals_closed_by_stock_or_route") or 0)
    if native_solved > route_tree_closed:
        return (
            f"Native ChemEnzy subplanning closed {native_solved}/{total} accepted enzyme precursors, "
            f"better than route-tree chemical continuation ({route_tree_closed}/{total})."
        )
    if native_solved:
        return (
            f"Native ChemEnzy subplanning closed {native_solved}/{total} accepted enzyme precursors, "
            f"similar to route-tree chemical continuation ({route_tree_closed}/{total})."
        )
    if native_summary.get("subgoals"):
        return (
            f"Native ChemEnzy subplanning did not close any of {total} accepted enzyme precursors under this budget."
        )
    return "Native ChemEnzy subplanning was not run."


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Enzyme Precursor Expansion Benchmark",
        "",
        "Chemical-only route-tree expansion of SP-v1 accepted enzymatic-step precursors.",
        "",
        "| subgoals | initial stock | with routes | solved | progressive | timeout-frontier routes | total routes | mean steps | mean elapsed s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| {subgoals} | {subgoals_initial_stock} | {subgoals_with_routes} | "
            "{subgoals_with_solved_route} | {subgoals_with_progressive_route} | "
            "{timeout_frontier_routes} | {route_count} | {mean_steps:.3f} | {mean_elapsed_s:.3f} |"
        ).format(**summary),
        "",
        "## Conclusion",
        "",
        report.get("conclusion") or "",
        "",
        "## Stop Reasons",
        "",
    ]
    for reason, count in sorted((summary.get("search_stop_reasons") or {}).items()):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Proposal Source Outputs", ""])
    for source, count in sorted((summary.get("proposal_source_outputs") or {}).items()):
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Selected Step Sources", ""])
    for source, count in sorted((summary.get("selected_step_source_counts") or {}).items()):
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Runtime Bottlenecks", ""])
    for label, count in sorted((summary.get("runtime_bottlenecks") or {}).items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Stock Rescue",
            "",
            f"- retries: {int(summary.get('stock_rescue_retries') or 0)}",
            f"- rejected retries: {int(summary.get('stock_rescue_rejected') or 0)}",
        ]
    )
    lines.extend(
        [
            "",
            "## Subgoal Outcomes",
            "",
            "| # | closed | solved | progressive | source | EC | SP-v1 score | routes | stop | bottlenecks | subgoal |",
            "|---:|---:|---:|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in report.get("subgoal_outcomes") or []:
        closed = bool(row.get("subgoal_in_stock_initial")) or int(row.get("solved_routes") or 0) > 0
        subgoal = str(row.get("subgoal_canonical") or "")
        if len(subgoal) > 96:
            subgoal = f"{subgoal[:93]}..."
        lines.append(
            "| {index} | {closed} | {solved_routes} | {progressive_routes} | `{enzyme_step_source}` | `{enzyme_step_ec}` | {score:.3f} | {route_count} | `{stop}` | {bottlenecks} | `{subgoal}` |".format(
                index=row.get("index"),
                closed=int(closed),
                solved_routes=int(row.get("solved_routes") or 0),
                progressive_routes=int(row.get("progressive_routes") or 0),
                enzyme_step_source=str(row.get("enzyme_step_source") or ""),
                enzyme_step_ec=str(row.get("enzyme_step_ec") or ""),
                score=float(row.get("enzyme_sp_v1_score") or 0.0),
                route_count=int(row.get("route_count") or 0),
                stop=str(row.get("search_stop_reason") or ""),
                bottlenecks=",".join(str(item) for item in row.get("runtime_bottlenecks") or []),
                subgoal=subgoal,
            )
        )
    lines.append("")
    native_summary = report.get("native_subplanner_summary") or {}
    if native_summary.get("subgoals"):
        lines.extend(
            [
                "",
                "## Native ChemEnzy Subplanner",
                "",
                "| subgoals | with routes | solved | total routes | mean steps | mean elapsed s |",
                "|---:|---:|---:|---:|---:|---:|",
                (
                    "| {subgoals} | {native_subgoals_with_routes} | {native_subgoals_solved} | "
                    "{native_total_routes} | {native_mean_steps:.3f} | {native_mean_elapsed_s:.3f} |"
                ).format(**native_summary),
                "",
                "Hybrid conclusion:",
                "",
                report.get("hybrid_conclusion") or "",
                "",
            ]
        )
    return "\n".join(lines)


def _is_sp_v1_accepted_enzyme_step(step: dict[str, Any]) -> bool:
    sp_payload = _step_sp_payload(step)
    if not bool(sp_payload.get("accepted")):
        return False
    source = str(step.get("source") or "").lower()
    if source in ENZYME_SOURCES:
        return True
    return bool(step.get("ec") or step.get("is_enzymatic"))


def _step_sp_payload(step: dict[str, Any]) -> dict[str, Any]:
    payload = step.get("enzyme_sp_verifier_v1")
    if isinstance(payload, dict):
        return payload
    evidence = step.get("evidence") or {}
    payload = evidence.get("enzyme_sp_verifier_v1")
    return payload if isinstance(payload, dict) else {}


def _step_components(step: dict[str, Any], *, include_aux: bool) -> list[tuple[str, str]]:
    components = [("main_reactant", str(step.get("main_reactant") or ""))]
    if include_aux:
        for idx, aux in enumerate(step.get("aux_reactants") or [], start=1):
            aux_smiles = str(aux or "")
            canonical = canonical_smiles(aux_smiles) or aux_smiles
            if canonical in TRIVIAL_AUXILIARY_SMILES or aux_smiles in TRIVIAL_AUXILIARY_SMILES:
                continue
            components.append((f"aux_reactant_{idx}", aux_smiles))
    return components


def _merge_source_metric(rows: list[dict[str, Any]], *, metric: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        source_stats = (row.get("stats") or {}).get("proposal_source_stats") or {}
        for source, payload in source_stats.items():
            out[source] = out.get(source, 0) + int((payload or {}).get(metric) or 0)
    return out


def _merge_source_outputs(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        source_stats = (row.get("stats") or {}).get("proposal_source_stats") or {}
        for source, payload in source_stats.items():
            payload = payload or {}
            count = max(
                int(payload.get("final_returned") or 0),
                int(payload.get("kept_returned") or 0),
                int(payload.get("raw_returned") or 0),
            )
            out[source] = out.get(source, 0) + count
    return out


def _merge_route_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for route in row.get("routes") or []:
            for source, count in (route.get("source_counts") or {}).items():
                out[str(source)] = out.get(str(source), 0) + int(count or 0)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_list(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
