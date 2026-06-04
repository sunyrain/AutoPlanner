"""Run a small live-provider smoke for bridge-aware route-tree gating.

This is the first live proposal-provider check after the controlled route-gate
benchmark. It uses the actual ``build_live_retro_engine`` providers with a very
small target set and capped search budget. The goal is to inspect source calls,
runtime, and enzyme-step pollution under native, ungated, and bridge-gated
source allocation.
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

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import BridgeAwareSourceGate, SourceGate
from scripts.run_bridge_route_gate_ablation_v0 import load_targets


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_MODEL_PATH = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
ENZYME_SOURCES = {"enzyformer", "enzexpand", "retrorules", "enzyme", "enzymatic"}
CHEMICAL_KEEP = {"retrochimera", "chemtemplates", "chem_enzy_onestep", "template_relevance"}


def engine_for_policy(full_engine: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == "native_no_enzyme":
        return {key: value for key, value in full_engine.items() if value is not None and key in CHEMICAL_KEEP}
    return {key: value for key, value in full_engine.items() if value is not None}


def gate_for_policy(policy: str, pack_dir: Path) -> SourceGate:
    if policy in {"bridge_gate_no_verifier", "bridge_gate_verifier"}:
        return BridgeAwareSourceGate(
            SourceGate(),
            retriever=BridgeRetrieverV0(pack_dir, scorer=None),
            require_verifier_pass=(policy == "bridge_gate_verifier"),
        )
    return SourceGate()


def route_sources(result: Any) -> list[str]:
    if result is None:
        return []
    return [str(getattr(slot, "source", "") or "") for slot in getattr(result.board, "slots", [])]


def run_one(target: dict[str, Any], policy: str, full_engine: dict[str, Any], pack_dir: Path) -> dict[str, Any]:
    engine = engine_for_policy(full_engine, policy)
    planner = NeuralGuidedAOSearch(
        retro_engine=engine,
        stock_checker=None,
        max_depth=2,
        branch_factor=4,
        expansion_budget=6,
        controller=None,
    )
    planner.proposals.source_gate = gate_for_policy(policy, pack_dir)
    started = time.monotonic()
    results = planner.search(str(target["target_smiles"]), n_results=1)
    elapsed = time.monotonic() - started
    result = results[0] if results else None
    sources = route_sources(result)
    source_stats = planner.stats.to_dict().get("proposal_source_stats", {})
    enzyme_calls = sum(int((source_stats.get(source) or {}).get("calls") or 0) for source in ENZYME_SOURCES)
    chemical_calls = sum(
        int((source_stats.get(source) or {}).get("calls") or 0)
        for source in source_stats
        if source not in ENZYME_SOURCES
    )
    return {
        **target,
        "policy": policy,
        "available_sources": sorted(engine),
        "result_count": len(results),
        "selected_sources": sources,
        "selected_enzyme_route": any(source in ENZYME_SOURCES for source in sources),
        "enzyme_source_calls": enzyme_calls,
        "chemical_source_calls": chemical_calls,
        "generated_actions": planner.stats.generated_actions,
        "proposal_calls": planner.stats.proposal_calls,
        "expansions": planner.stats.expansions,
        "search_stop_reason": planner.stats.search_stop_reason,
        "elapsed_s": round(float(elapsed), 3),
        "proposal_source_stats": source_stats,
    }


def summarize(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy]
    positives = [row for row in subset if int(row["label"]) == 1]
    negatives = [row for row in subset if int(row["label"]) == 0]
    selected = [row for row in subset if row["selected_enzyme_route"]]
    false_selected = [row for row in selected if int(row["label"]) == 0]
    true_selected = [row for row in selected if int(row["label"]) == 1]
    return {
        "policy": policy,
        "rows": len(subset),
        "positives": len(positives),
        "negatives": len(negatives),
        "selected_enzyme_routes": len(selected),
        "true_selected_enzyme_routes": len(true_selected),
        "false_selected_enzyme_routes": len(false_selected),
        "selected_enzyme_precision": len(true_selected) / len(selected) if selected else 0.0,
        "selected_enzyme_recall": len(true_selected) / len(positives) if positives else 0.0,
        "false_enzyme_route_rate": len(false_selected) / len(negatives) if negatives else 0.0,
        "mean_enzyme_source_calls": mean(row["enzyme_source_calls"] for row in subset),
        "mean_chemical_source_calls": mean(row["chemical_source_calls"] for row in subset),
        "mean_generated_actions": mean(row["generated_actions"] for row in subset),
        "mean_elapsed_s": mean(row["elapsed_s"] for row in subset),
    }


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Live-Provider Smoke v0",
        "",
        "Small route-tree smoke using actual live proposal providers.",
        "",
        "Scope: capped live-provider diagnostic, not a final solved-rate benchmark. Runtime is cache-order biased because providers are shared across policies; source-call counts are the primary signal.",
        "",
        f"- Targets: {report['inputs']['targets']}",
        f"- Available live sources: {', '.join(report['inputs']['live_sources'])}",
        "",
        "| policy | enzyme routes | false enzyme | precision | recall | false enzyme rate | mean enzyme calls | mean elapsed s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {selected_enzyme_routes} | {false_selected_enzyme_routes} | "
            "{selected_enzyme_precision:.4f} | {selected_enzyme_recall:.4f} | "
            "{false_enzyme_route_rate:.4f} | {mean_enzyme_source_calls:.2f} | {mean_elapsed_s:.2f} |".format(**row)
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bridge live-provider smoke")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    started = time.monotonic()
    full_engine = build_live_retro_engine()
    live_sources = sorted(key for key, value in full_engine.items() if value is not None)
    targets = load_targets(
        args.pack_dir,
        args.output_dir,
        positives=max(0, int(args.positives)),
        negatives=max(0, int(args.negatives)),
        seed=int(args.seed),
    )
    policies = [
        "native_no_enzyme",
        "ungated_default_source_gate",
        "bridge_gate_no_verifier",
        "bridge_gate_verifier",
    ]
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for target in targets:
            rows.append(run_one(target, policy, full_engine, args.pack_dir))
    summaries = [summarize(rows, policy) for policy in policies]
    ungated = next(row for row in summaries if row["policy"] == "ungated_default_source_gate")
    gated = next(row for row in summaries if row["policy"] == "bridge_gate_verifier")
    conclusion = (
        "Live-provider smoke completed. Verifier gating changed mean enzyme source calls "
        f"from {ungated['mean_enzyme_source_calls']:.2f} to {gated['mean_enzyme_source_calls']:.2f}. "
        "This is a capped diagnostic; larger live-provider route evidence is still required for P5."
    )
    report = {
        "schema_version": "bridge_live_provider_smoke_v0.route_tree.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "model_path": str(args.model_path),
            "targets": len(targets),
            "positives": sum(1 for row in targets if int(row["label"]) == 1),
            "negatives": sum(1 for row in targets if int(row["label"]) == 0),
            "live_sources": live_sources,
        },
        "policies": summaries,
        "conclusion": conclusion,
        "retro_cache_stats": retro_engine_cache_stats(full_engine),
        "scope_note": "Runtime is cache-order biased because providers are shared across policies; source-call counts are the primary signal.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "bridge_live_provider_smoke_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    json_path = args.output_dir / "bridge_live_provider_smoke_report.json"
    md_path = args.output_dir / "bridge_live_provider_smoke_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
