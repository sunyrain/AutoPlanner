"""Build live-provider route evidence for bridge-gated enzyme proposals.

This script promotes targets from ``probe_live_enzyme_bridge_targets_v0`` into
actual route-tree searches with live proposal providers. It records whether an
enzyme step is selected under normal bridge-gated search and under an
enzyme-only diagnostic policy. The latter is not a production policy; it is a
provider-capability check used to decide whether P5 evidence is limited by
provider recall or by integrated route selection.
"""
from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
ENZYME_SOURCES = {"enzyformer", "enzexpand", "retrorules", "enzyme", "enzymatic"}
CHEMICAL_SOURCES = {"retrochimera", "chemtemplates", "chem_enzy_onestep", "template_relevance"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_targets(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(path)
        if bool(row.get("has_usable_live_enzyme_candidate")) and int(row.get("usable_enzyme_candidates") or 0) > 0
    ]
    rows.sort(
        key=lambda row: (
            -int(row.get("usable_enzyme_candidates") or 0),
            int(row.get("heavy_atoms") or 10**9),
            str(row.get("target_smiles") or ""),
        )
    )
    return rows[: max(1, int(limit))]


def engine_for_policy(full_engine: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == "normal_bridge_gated":
        keep = ENZYME_SOURCES | CHEMICAL_SOURCES
    elif policy == "enzyme_only_bridge_gated":
        keep = ENZYME_SOURCES
    else:
        raise ValueError(f"unknown policy: {policy}")
    return {key: value for key, value in full_engine.items() if value is not None and key in keep}


def build_gate(retriever: BridgeRetrieverV0) -> BridgeAwareSourceGate:
    return BridgeAwareSourceGate(
        SourceGate(),
        retriever=retriever,
        require_verifier_pass=True,
    )


def route_sources(result: Any) -> list[str]:
    if result is None:
        return []
    return [str(getattr(slot, "source", "") or "") for slot in getattr(result.board, "slots", [])]


def route_steps(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    steps: list[dict[str, Any]] = []
    for slot in getattr(result.board, "slots", []):
        if not getattr(slot, "is_filled", lambda: bool(getattr(slot, "reaction_smiles", "")))():
            continue
        steps.append(
            {
                "index": int(getattr(slot, "index", len(steps))),
                "source": str(getattr(slot, "source", "") or ""),
                "product": str(getattr(slot, "product", "") or ""),
                "main_reactant": str(getattr(slot, "main_reactant", "") or ""),
                "aux_reactants": list(getattr(slot, "aux_reactants", []) or []),
                "reaction_smiles": str(getattr(slot, "reaction_smiles", "") or ""),
                "ec": str(getattr(slot, "ec", "") or ""),
                "catalyst": str(getattr(slot, "catalyst", "") or ""),
                "T": getattr(slot, "T", None),
                "pH": getattr(slot, "pH", None),
                "solvent": str(getattr(slot, "solvent", "") or ""),
            }
        )
    return steps


def bridge_evidence(target_smiles: str, retriever: BridgeRetrieverV0) -> list[dict[str, Any]]:
    hits = retriever.retrieve(target_smiles, top_k=3, require_verifier_pass=True)
    return [
        {
            "source": hit.source,
            "bridge_direction": hit.bridge_direction,
            "chemical_smiles": hit.chemical_smiles,
            "enzyme_smiles": hit.enzyme_smiles,
            "tanimoto": round(float(hit.tanimoto), 4),
            "verifier_score": round(float(hit.verifier_score or 0.0), 6),
            "verifier_pass": bool(hit.verifier_pass),
            "enzyme_ec_sample": list(hit.enzyme_ec_sample[:8]),
        }
        for hit in hits
    ]


def run_one(
    target: dict[str, Any],
    *,
    policy: str,
    full_engine: dict[str, Any],
    retriever: BridgeRetrieverV0,
    max_depth: int,
    branch_factor: int,
    expansion_budget: int,
    n_results: int,
) -> dict[str, Any]:
    engine = engine_for_policy(full_engine, policy)
    planner = NeuralGuidedAOSearch(
        retro_engine=engine,
        stock_checker=None,
        max_depth=max_depth,
        branch_factor=branch_factor,
        expansion_budget=expansion_budget,
        controller=None,
    )
    planner.proposals.source_gate = build_gate(retriever)
    started = time.monotonic()
    results = planner.search(str(target["target_smiles"]), n_results=n_results)
    elapsed = time.monotonic() - started
    result_rows = []
    for idx, result in enumerate(results, start=1):
        sources = route_sources(result)
        steps = route_steps(result)
        result_rows.append(
            {
                "rank": idx,
                "score": float(getattr(result, "score", 0.0) or 0.0),
                "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
                "quality_vector": dict(getattr(result, "quality_vector", {}) or {}),
                "selected_sources": sources,
                "selected_enzyme_route": any(source in ENZYME_SOURCES for source in sources),
                "steps": steps,
            }
        )
    source_stats = planner.stats.to_dict().get("proposal_source_stats", {})
    return {
        "target_smiles": target.get("target_smiles") or "",
        "chemical_inchikey": target.get("chemical_inchikey") or "",
        "heavy_atoms": target.get("heavy_atoms"),
        "policy": policy,
        "available_sources": sorted(engine),
        "bridge_probe": {
            "bridge_source": target.get("bridge_source") or "",
            "bridge_direction": target.get("bridge_direction") or "",
            "verifier_score": target.get("verifier_score"),
            "usable_enzyme_candidates": target.get("usable_enzyme_candidates"),
            "source_rows": target.get("source_rows") or {},
        },
        "bridge_evidence": bridge_evidence(str(target["target_smiles"]), retriever),
        "result_count": len(results),
        "selected_enzyme_routes": sum(1 for row in result_rows if row["selected_enzyme_route"]),
        "results": result_rows,
        "stats": planner.stats.to_dict(),
        "proposal_source_stats": source_stats,
        "elapsed_s": round(float(elapsed), 3),
    }


def build_cards(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in rows:
        enzyme_results = [result for result in row["results"] if result["selected_enzyme_route"]]
        if not enzyme_results:
            continue
        result = enzyme_results[0]
        cards.append(
            {
                "route_card_id": f"live_bridge_route_v0_{len(cards) + 1:03d}",
                "target_smiles": row["target_smiles"],
                "chemical_inchikey": row["chemical_inchikey"],
                "policy": row["policy"],
                "policy_scope": (
                    "integrated live search"
                    if row["policy"] == "normal_bridge_gated"
                    else "enzyme-provider capability diagnostic"
                ),
                "selected_sources": result["selected_sources"],
                "steps": result["steps"],
                "bridge_evidence": row["bridge_evidence"],
                "route_score": result["score"],
                "route_quality": result["quality_vector"],
                "search_stats": {
                    "elapsed_s": row["elapsed_s"],
                    "generated_actions": row["stats"].get("generated_actions"),
                    "expansions": row["stats"].get("expansions"),
                    "proposal_calls": row["stats"].get("proposal_calls"),
                    "search_stop_reason": row["stats"].get("search_stop_reason"),
                },
                "evidence": [
                    "route-tree search used actual live proposal providers",
                    "selected route contains at least one enzyme-source step",
                    "enzyme source was enabled only after verifier-pass bridge evidence",
                    "bridge evidence is a trigger, not a synthetic pseudo-reaction",
                ],
            }
        )
        if len(cards) >= limit:
            break
    return cards


def summarize(rows: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in sorted({row["policy"] for row in rows}):
        subset = [row for row in rows if row["policy"] == policy]
        by_policy[policy] = {
            "targets": len(subset),
            "routes_returned": sum(int(row["result_count"]) for row in subset),
            "targets_with_selected_enzyme_route": sum(1 for row in subset if int(row["selected_enzyme_routes"]) > 0),
            "selected_enzyme_routes": sum(int(row["selected_enzyme_routes"]) for row in subset),
            "mean_elapsed_s": round(sum(float(row["elapsed_s"]) for row in subset) / len(subset), 3) if subset else 0.0,
        }
    return {
        "policies": by_policy,
        "cards": len(cards),
    }


def render_markdown(report: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Live Bridge Route Evidence v0",
        "",
        "Route-tree searches using actual live providers on bridge-positive targets from the probe set.",
        "",
        "The enzyme-only policy is a diagnostic, not a production search policy.",
        "",
        "| policy | targets | routes | targets with enzyme route | enzyme routes | mean elapsed s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, row in report["summary"]["policies"].items():
        lines.append(
            f"| {policy} | {row['targets']} | {row['routes_returned']} | "
            f"{row['targets_with_selected_enzyme_route']} | {row['selected_enzyme_routes']} | {row['mean_elapsed_s']} |"
        )
    lines.extend(["", "## Evidence Cards", ""])
    for card in cards:
        steps = " -> ".join(card["selected_sources"])
        bridge = card["bridge_evidence"][0] if card["bridge_evidence"] else {}
        lines.extend(
            [
                f"### {card['route_card_id']}",
                "",
                f"- Target: `{card['target_smiles']}`",
                f"- Policy: `{card['policy']}` ({card['policy_scope']})",
                f"- Selected sources: {steps}",
                f"- Bridge source: `{bridge.get('source', '')}` / direction `{bridge.get('bridge_direction', '')}`",
                f"- Verifier score: {bridge.get('verifier_score', '')}",
                f"- EC sample: {', '.join(bridge.get('enzyme_ec_sample') or []) if bridge else 'N/A'}",
                f"- Steps: {len(card['steps'])}",
                "",
            ]
        )
        for step in card["steps"]:
            lines.append(
                f"  - Step {step['index']}: `{step['source']}` `{step['main_reactant']}` >> `{step['product']}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live bridge route evidence search")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=6)
    parser.add_argument("--expansion-budget", type=int, default=12)
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument("--card-limit", type=int, default=12)
    parser.add_argument("--bridge-enzyme-bonus", type=float, default=0.0)
    args = parser.parse_args()

    started = time.monotonic()
    old_bonus = os.environ.get("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS")
    if float(args.bridge_enzyme_bonus) > 0.0:
        os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = str(float(args.bridge_enzyme_bonus))
    targets = load_targets(args.probe_rows, limit=max(1, int(args.limit)))
    try:
        full_engine = build_live_retro_engine()
        retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
        policies = ["normal_bridge_gated", "enzyme_only_bridge_gated"]
        rows: list[dict[str, Any]] = []
        for policy in policies:
            for target in targets:
                rows.append(
                    run_one(
                        target,
                        policy=policy,
                        full_engine=full_engine,
                        retriever=retriever,
                        max_depth=max(1, int(args.max_depth)),
                        branch_factor=max(1, int(args.branch_factor)),
                        expansion_budget=max(1, int(args.expansion_budget)),
                        n_results=max(1, int(args.n_results)),
                    )
                )
    finally:
        if old_bonus is None:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS", None)
        else:
            os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = old_bonus
    cards = build_cards(rows, limit=max(0, int(args.card_limit)))
    report = {
        "schema_version": "bridge_live_route_evidence_v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "targets": len(targets),
            "max_depth": int(args.max_depth),
            "branch_factor": int(args.branch_factor),
            "expansion_budget": int(args.expansion_budget),
            "n_results": int(args.n_results),
            "bridge_enzyme_bonus": float(args.bridge_enzyme_bonus),
            "live_sources": sorted(key for key, value in full_engine.items() if value is not None),
        },
        "summary": summarize(rows, cards),
        "retro_cache_stats": retro_engine_cache_stats(full_engine),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "bridge_live_route_evidence_rows.jsonl"
    cards_json_path = args.output_dir / "bridge_live_route_evidence_cards.json"
    cards_md_path = args.output_dir / "bridge_live_route_evidence_cards.md"
    report_json_path = args.output_dir / "bridge_live_route_evidence_report.json"
    report_md_path = args.output_dir / "bridge_live_route_evidence_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    cards_json_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    cards_md_path.write_text(render_markdown(report, cards), encoding="utf-8")
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report_md_path.write_text(render_markdown(report, cards), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_json_path),
                "rows": str(rows_path),
                "cards": len(cards),
                "summary": report["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
