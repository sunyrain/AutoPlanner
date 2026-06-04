"""Export step-level enzyme audit rows from ChemEnzy native route runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_adapter import ChemEnzyBackendAdapter, DEFAULT_ONE_STEP_MODELS, DEFAULT_STOCKS
from cascade_planner.baselines.enzyme_step_audit import audit_baseline_results, summarize_enzyme_step_audit
from cascade_planner.baselines.enzyme_step_enhancement import EnzymeStepEnhancementConfig, make_default_sp_v1_scorer
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig


DEFAULT_STATIN_SUMMARY = Path("docs/statins/summary.json")
DEFAULT_OUTPUT_DIR = Path("results/shared/chem_enzy_enzyme_step_audit")
FULL_ONE_STEP_MODELS = [
    "graphfp_models.USPTO-full_remapped",
    "onmt_models.bionav_one_step",
    "onmt_models.bionav_native_one_step",
    "template_relevance.pistachio",
    "template_relevance.pistachio_ringbreaker",
    "template_relevance.reaxys",
    "template_relevance.bkms_metabolic",
    "template_relevance.reaxys_biocatalysis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statin-summary", type=Path, default=DEFAULT_STATIN_SUMMARY)
    parser.add_argument("--targets", default="", help="Comma-separated statin safe names. Empty means all.")
    parser.add_argument("--smiles", default="", help="Optional comma-separated target SMILES. Used in addition to statins.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--expansion-topk", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--model-preset", choices=["adapter_default", "full"], default="adapter_default")
    parser.add_argument("--one-step-models", default="", help="Comma-separated explicit ChemEnzy model names.")
    parser.add_argument("--enable-condition-prediction", action="store_true")
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument(
        "--disable-expansion-metadata",
        action="store_true",
        help="Do not enable the zero-adjustment cascade cost hook used to retain one-step source metadata.",
    )
    parser.add_argument(
        "--strengthen-chemenzy-steps",
        action="store_true",
        help="Enable bridge/SP-v1/material enzyme-step strengthening for the ChemEnzy run.",
    )
    parser.add_argument(
        "--compare-strengthening",
        action="store_true",
        help="Run native and strengthened settings and write a before/after comparison report.",
    )
    parser.add_argument(
        "--enable-step-enhancement-audit",
        action="store_true",
        help="Audit each selected step for missing enzyme steps, wrong enzyme-step replacements, and efficient enzyme upgrades.",
    )
    parser.add_argument("--step-enhancement-topk", type=int, default=24)
    parser.add_argument("--step-enhancement-output-topk", type=int, default=5)
    parser.add_argument("--disable-step-enhancement-sp-v1", action="store_true")
    parser.add_argument("--no-reuse-planner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    targets = load_targets(args)
    if not targets:
        raise SystemExit("no targets selected")
    one_step_models = selected_one_step_models(args)
    adapter = ChemEnzyBackendAdapter(
        gpu=int(args.gpu),
        enable_condition_prediction=bool(args.enable_condition_prediction),
        enable_enzyme_assignment=True,
    )
    configs = [
        route_config(target, args, one_step_models, strengthened=bool(args.strengthen_chemenzy_steps))
        for target in targets
    ]
    results = adapter.run_targets(configs, reuse_planner=not bool(args.no_reuse_planner))
    target_metadata = {target["target_smiles"]: target for target in targets}
    step_enhancement_config = enhancement_config_from_args(args)
    step_enhancement_scorer = enhancement_scorer_from_args(args)
    rows = audit_baseline_results(
        results,
        target_metadata=target_metadata,
        enable_step_enhancement=bool(args.enable_step_enhancement_audit),
        step_enhancement_config=step_enhancement_config,
        step_enhancement_scorer=step_enhancement_scorer,
    )
    summary = summarize_enzyme_step_audit(rows)
    report = {
        "schema_version": "chem_enzy_enzyme_step_audit_run.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "targets": [{"name": row.get("name"), "safe": row.get("safe"), "target_smiles": row["target_smiles"]} for row in targets],
            "iterations": int(args.iterations),
            "max_depth": int(args.max_depth),
            "expansion_topk": int(args.expansion_topk),
            "stock_names": list(DEFAULT_STOCKS),
            "one_step_models": one_step_models,
            "enable_enzyme_assignment": True,
            "enable_condition_prediction": bool(args.enable_condition_prediction),
            "use_filter": bool(args.use_filter),
            "capture_expansion_metadata": not bool(args.disable_expansion_metadata),
            "strengthen_chemenzy_steps": bool(args.strengthen_chemenzy_steps),
            "enable_step_enhancement_audit": bool(args.enable_step_enhancement_audit),
            "step_enhancement_config": step_enhancement_config.to_dict(),
            "step_enhancement_sp_v1_available": bool(step_enhancement_scorer is not None),
        },
        "summary": summary,
        "native_enzyme_plugin": plugin_stats_summary(results),
        "failures": failure_rows(results, target_metadata),
    }
    paths = write_outputs(args.output_dir, report=report, rows=rows, results=results)
    if args.compare_strengthening:
        paths.update(
            run_strengthening_comparison(
                adapter=adapter,
                targets=targets,
                args=args,
                one_step_models=one_step_models,
                target_metadata=target_metadata,
                step_enhancement_config=step_enhancement_config,
                step_enhancement_scorer=step_enhancement_scorer,
            )
        )
    print(json.dumps({"paths": paths, "summary": summary}, indent=2, ensure_ascii=False, sort_keys=True))


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = {item.strip().lower() for item in str(args.targets or "").split(",") if item.strip()}
    if args.statin_summary.exists():
        payload = json.loads(args.statin_summary.read_text(encoding="utf-8"))
        for row in payload.get("targets") or []:
            safe = str(row.get("safe") or row.get("name") or "").lower()
            if wanted and safe not in wanted:
                continue
            rows.append(
                {
                    "name": str(row.get("name") or safe),
                    "safe": safe,
                    "target_smiles": str(row.get("smiles") or ""),
                    "source": str(args.statin_summary),
                }
            )
    for idx, smiles in enumerate(str(args.smiles or "").split(","), start=1):
        smiles = smiles.strip()
        if not smiles:
            continue
        rows.append(
            {
                "name": f"custom_{idx}",
                "safe": f"custom_{idx}",
                "target_smiles": smiles,
                "source": "cli_smiles",
            }
        )
    seen = set()
    out = []
    for row in rows:
        smiles = row["target_smiles"]
        if smiles and smiles not in seen:
            seen.add(smiles)
            out.append(row)
    return out


def selected_one_step_models(args: argparse.Namespace) -> list[str]:
    explicit = [item.strip() for item in str(args.one_step_models or "").split(",") if item.strip()]
    if explicit:
        return explicit
    if args.model_preset == "full":
        return list(FULL_ONE_STEP_MODELS)
    return list(DEFAULT_ONE_STEP_MODELS)


def route_config(
    target: dict[str, Any],
    args: argparse.Namespace,
    one_step_models: list[str],
    *,
    strengthened: bool = False,
) -> RouteSearchConfig:
    search_flags = {
        "gpu": int(args.gpu),
        "keep_search": True,
        "use_filter": bool(args.use_filter),
        "use_depth_value_fn": False,
        "use_cascade_cost_model": not bool(args.disable_expansion_metadata),
        "cascade_cost_model": {"enabled": True, "weights": {}} if not bool(args.disable_expansion_metadata) else {},
    }
    if strengthened:
        search_flags["chem_enzy_step_strengthening"] = True
    return RouteSearchConfig(
        target_smiles=str(target["target_smiles"]),
        stock_names=list(DEFAULT_STOCKS),
        max_iterations=max(1, int(args.iterations)),
        max_depth=max(1, int(args.max_depth)),
        expansion_topk=max(1, int(args.expansion_topk)),
        one_step_models=list(one_step_models),
        search_flags=search_flags,
    )


def enhancement_config_from_args(args: argparse.Namespace) -> EnzymeStepEnhancementConfig:
    return EnzymeStepEnhancementConfig(
        retrieve_top_k=max(1, int(args.step_enhancement_topk)),
        output_top_k=max(1, int(args.step_enhancement_output_topk)),
    )


def enhancement_scorer_from_args(args: argparse.Namespace) -> Any | None:
    if args.disable_step_enhancement_sp_v1:
        return None
    if not args.enable_step_enhancement_audit:
        return None
    return make_default_sp_v1_scorer()


def run_strengthening_comparison(
    *,
    adapter: ChemEnzyBackendAdapter,
    targets: list[dict[str, Any]],
    args: argparse.Namespace,
    one_step_models: list[str],
    target_metadata: dict[str, dict[str, Any]],
    step_enhancement_config: EnzymeStepEnhancementConfig,
    step_enhancement_scorer: Any | None,
) -> dict[str, str]:
    native_configs = [route_config(target, args, one_step_models, strengthened=False) for target in targets]
    strong_configs = [route_config(target, args, one_step_models, strengthened=True) for target in targets]
    native_results = adapter.run_targets(native_configs, reuse_planner=not bool(args.no_reuse_planner))
    strong_results = adapter.run_targets(strong_configs, reuse_planner=not bool(args.no_reuse_planner))
    native_rows = audit_baseline_results(
        native_results,
        target_metadata=target_metadata,
        enable_step_enhancement=bool(args.enable_step_enhancement_audit),
        step_enhancement_config=step_enhancement_config,
        step_enhancement_scorer=step_enhancement_scorer,
    )
    strong_rows = audit_baseline_results(
        strong_results,
        target_metadata=target_metadata,
        enable_step_enhancement=bool(args.enable_step_enhancement_audit),
        step_enhancement_config=step_enhancement_config,
        step_enhancement_scorer=step_enhancement_scorer,
    )
    native_summary = summarize_enzyme_step_audit(native_rows)
    strong_summary = summarize_enzyme_step_audit(strong_rows)
    comparison = {
        "schema_version": "chem_enzy_step_strengthening_comparison.v1",
        "native": native_summary,
        "strengthened": strong_summary,
        "native_plugin": plugin_stats_summary(native_results),
        "strengthened_plugin": plugin_stats_summary(strong_results),
        "delta": comparison_delta(native_summary, strong_summary),
        "plugin_delta": comparison_delta(plugin_stats_summary(native_results), plugin_stats_summary(strong_results)),
    }
    json_path = args.output_dir / "chem_enzy_step_strengthening_comparison.json"
    md_path = args.output_dir / "chem_enzy_step_strengthening_comparison.md"
    native_rows_jsonl = args.output_dir / "chem_enzy_step_strengthening_native_rows.jsonl"
    strong_rows_jsonl = args.output_dir / "chem_enzy_step_strengthening_strengthened_rows.jsonl"
    native_rows_csv = args.output_dir / "chem_enzy_step_strengthening_native_rows.csv"
    strong_rows_csv = args.output_dir / "chem_enzy_step_strengthening_strengthened_rows.csv"
    native_results_json = args.output_dir / "chem_enzy_step_strengthening_native_results.json"
    strong_results_json = args.output_dir / "chem_enzy_step_strengthening_strengthened_results.json"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    native_rows_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in native_rows),
        encoding="utf-8",
    )
    strong_rows_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in strong_rows),
        encoding="utf-8",
    )
    write_csv(native_rows_csv, native_rows)
    write_csv(strong_rows_csv, strong_rows)
    native_results_json.write_text(
        json.dumps([result.to_dict() for result in native_results], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    strong_results_json.write_text(
        json.dumps([result.to_dict() for result in strong_results], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "comparison_json": str(json_path),
        "comparison_md": str(md_path),
        "comparison_native_rows_jsonl": str(native_rows_jsonl),
        "comparison_strengthened_rows_jsonl": str(strong_rows_jsonl),
        "comparison_native_rows_csv": str(native_rows_csv),
        "comparison_strengthened_rows_csv": str(strong_rows_csv),
        "comparison_native_results_json": str(native_results_json),
        "comparison_strengthened_results_json": str(strong_results_json),
    }


def comparison_delta(native: dict[str, Any], strengthened: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "routes",
        "steps",
        "enzyme_like_source_steps",
        "plugin_injected_steps",
        "enzyme_quality_scored_steps",
        "enzyme_quality_warned_steps",
        "derived_quality_scored_steps",
        "step_enhancement_available_steps",
        "missing_enzyme_step_opportunities",
        "wrong_enzyme_step_replacements",
        "efficient_enzyme_step_upgrades",
        "step_enhancement_viable_candidates",
        "search_time_quality_scored_steps",
        "search_time_quality_passed_steps",
        "material_failed_steps",
        "enzyme_like_material_failed_steps",
        "enzyme_source_without_ec_steps",
        "posthoc_enzymatic_on_chemical_source_steps",
    ]
    extra_keys = sorted((set(native) | set(strengthened)) - set(keys))
    return {
        key: int(strengthened.get(key) or 0) - int(native.get(key) or 0)
        for key in [*keys, *extra_keys]
        if (key in native or key in strengthened)
        and isinstance(native.get(key, 0), (int, float, bool))
        and isinstance(strengthened.get(key, 0), (int, float, bool))
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    native = comparison.get("native") or {}
    strengthened = comparison.get("strengthened") or {}
    delta = comparison.get("delta") or {}
    native_plugin = comparison.get("native_plugin") or {}
    strong_plugin = comparison.get("strengthened_plugin") or {}
    plugin_delta = comparison.get("plugin_delta") or {}
    rows = [
        ("routes", "routes"),
        ("steps", "steps"),
        ("enzyme_like_source_steps", "enzyme-like source steps"),
        ("plugin_injected_steps", "structured enzyme plugin steps"),
        ("enzyme_quality_scored_steps", "all enzyme quality scored steps"),
        ("derived_quality_scored_steps", "native/post-hoc derived quality steps"),
        ("missing_enzyme_step_opportunities", "missing enzyme-step opportunities"),
        ("wrong_enzyme_step_replacements", "wrong enzyme-step replacements"),
        ("efficient_enzyme_step_upgrades", "efficient enzyme-step upgrades"),
        ("step_enhancement_viable_candidates", "viable enhancement candidates"),
        ("search_time_quality_scored_steps", "search-time quality scored steps"),
        ("search_time_quality_passed_steps", "search-time quality passed steps"),
        ("material_failed_steps", "material failed steps"),
        ("enzyme_like_material_failed_steps", "enzyme-like material failed steps"),
        ("enzyme_source_without_ec_steps", "enzyme source without EC"),
    ]
    lines = [
        "# ChemEnzy Step Strengthening Comparison",
        "",
        "| metric | native | strengthened | delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in rows:
        lines.append(
            f"| {label} | {int(native.get(key) or 0)} | {int(strengthened.get(key) or 0)} | {int(delta.get(key) or 0):+d} |"
        )
    plugin_rows = [
        ("calls", "plugin calls"),
        ("bridge_hit_calls", "bridge-hit calls"),
        ("retrieved_candidates", "retrieved enzyme candidates"),
        ("sp_v1_scored", "SP-v1 scored candidates"),
        ("sp_v1_accepted", "SP-v1 accepted candidates"),
        ("quality_scored", "quality-scored candidates"),
        ("quality_passed", "quality-passed candidates"),
        ("material_rejected", "material-rejected candidates"),
        ("added_candidates", "added candidates"),
    ]
    lines.extend(
        [
            "",
            "## Search-Time Plugin Delta",
            "",
            "| metric | native | strengthened | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in plugin_rows:
        lines.append(
            f"| {label} | {int(native_plugin.get(key) or 0)} | {int(strong_plugin.get(key) or 0)} | {int(plugin_delta.get(key) or 0):+d} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `structured enzyme plugin steps` should increase only when bridge/SP-v1/material gates allow search-time enzyme candidates.",
            "- `search-time quality scored/passed steps` are visible evidence that enzyme support was used before route selection.",
            "- `material failed steps` should decrease or move out of selected routes when strengthening is effective.",
        ]
    )
    return "\n".join(lines)


def plugin_stats_summary(results: list[BaselineRunResult]) -> dict[str, Any]:
    keys = [
        "calls",
        "bridge_hit_calls",
        "skipped_no_bridge",
        "retrieved_candidates",
        "sp_v1_scored",
        "sp_v1_accepted",
        "sp_v1_rejected",
        "quality_scored",
        "quality_passed",
        "quality_warned",
        "quality_rejected",
        "material_rejected",
        "added_candidates",
        "duplicate_candidates",
        "invalid_candidates",
        "source_policy_skips",
        "error_count",
    ]
    out = {key: 0 for key in keys}
    enabled = 0
    targets = 0
    for result in results:
        stats = (result.raw_backend_metadata or {}).get("native_enzyme_plugin")
        if not isinstance(stats, dict):
            continue
        targets += 1
        enabled += int(bool(stats.get("enabled")))
        for key in keys:
            out[key] += int(stats.get(key) or 0)
    out["targets_with_plugin_stats"] = targets
    out["targets_with_plugin_enabled"] = enabled
    return out


def failure_rows(results: list[BaselineRunResult], target_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        meta = target_metadata.get(result.target_smiles) or {}
        for failure in result.failures:
            rows.append(
                {
                    "target_name": meta.get("name") or "",
                    "target_safe": meta.get("safe") or "",
                    "target_smiles": result.target_smiles,
                    "category": failure.category,
                    "message": failure.message,
                    "retryable": failure.retryable,
                }
            )
    return rows


def write_outputs(
    output_dir: Path,
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    results: list[BaselineRunResult],
) -> dict[str, str]:
    report_json = output_dir / "chem_enzy_enzyme_step_audit_report.json"
    report_md = output_dir / "chem_enzy_enzyme_step_audit_report.md"
    rows_jsonl = output_dir / "chem_enzy_enzyme_step_audit_rows.jsonl"
    rows_csv = output_dir / "chem_enzy_enzyme_step_audit_rows.csv"
    results_json = output_dir / "chem_enzy_native_results.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    rows_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    results_json.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(rows_csv, rows)
    report_md.write_text(render_markdown(report), encoding="utf-8")
    return {
        "report_json": str(report_json),
        "report_md": str(report_md),
        "rows_jsonl": str(rows_jsonl),
        "rows_csv": str(rows_csv),
        "results_json": str(results_json),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "target_safe",
        "route_rank",
        "step_index",
        "step_position",
        "source_model",
        "proposal_domain",
        "proposal_source_kind",
        "posthoc_reaction_type",
        "posthoc_reaction_type_confidence",
        "ec_top1",
        "ec_top1_confidence",
        "sp_v1_score",
        "sp_v1_accepted",
        "enzyme_quality_origin",
        "enzyme_quality_score",
        "enzyme_quality_decision",
        "enzyme_quality_flags",
        "enzyme_step_enhancement_kind",
        "enzyme_step_enhancement_best_score",
        "enzyme_step_enhancement_best_ec",
        "enzyme_step_enhancement_best_main_reactant",
        "enzyme_step_enhancement_viable_candidate_count",
        "cascade_cost_adjustment",
        "material_audit_passed",
        "material_audit_reasons",
        "weakness_flags",
        "rxn_smiles",
    ]
    keys = set()
    for row in rows:
        keys.update(row)
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# ChemEnzy Enzyme Step Audit",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        "- `enzyme_like_source_steps`: steps generated by enzyme-like proposal sources such as BioNav/BKMS/biocatalysis/plugin.",
        "- `posthoc_enzymatic_steps`: steps later classified as enzymatic by ChemEnzy annotation.",
        "- `posthoc_enzymatic_on_chemical_source_steps`: likely evidence that EC labels are post-hoc weak annotations, not search-time enzyme choices.",
        "- `enzyme_quality_scored_steps`: selected enzyme-like or post-hoc enzymatic steps with visible quality evidence.",
        "- `search_time_quality_scored_steps`: selected plugin enzyme candidates that carried explicit bridge/SP-v1/material quality evidence during search.",
        "- `derived_quality_scored_steps`: native ChemEnzy/post-hoc enzyme-like steps scored after selection to expose missing evidence.",
        "- `missing_enzyme_step_opportunities`: selected non-enzyme steps where an evidence-backed enzyme precedent was found.",
        "- `wrong_enzyme_step_replacements`: selected enzyme-like steps with weak evidence where a stronger precedent replacement was found.",
        "- `efficient_enzyme_step_upgrades`: selected enzyme-like steps where a higher efficiency-proxy enzyme precedent was found.",
        "- `material_failed_steps`: steps failing the conservative material sanity screen.",
        "",
        "## Failures",
        "",
    ]
    failures = report.get("failures") or []
    if not failures:
        lines.append("none")
    else:
        lines.extend(
            f"- {row.get('target_safe') or row.get('target_name')}: {row.get('category')} - {row.get('message')}"
            for row in failures
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
