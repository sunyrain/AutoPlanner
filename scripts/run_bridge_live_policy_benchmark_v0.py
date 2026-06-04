"""Fair live-provider benchmark for bridge-gated enzyme proposal policies.

Unlike the early smoke, this benchmark builds a fresh live-engine wrapper per
policy so provider caches do not give later policies an artificial runtime
advantage. Model weights may still be globally cached by the provider modules,
which is intentional and reflects realistic warm-start service behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import BridgeAwareSourceGate, SourceGate
from scripts.run_bridge_route_gate_ablation_v0 import heavy_atoms, inchikey_from_smiles


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_live_policy_benchmark_v0_20260528")
ENZYME_SOURCES = {"enzyformer", "enzexpand", "retrorules", "enzyme", "enzymatic"}
CHEMICAL_SOURCES = {"retrochimera", "chemtemplates", "chem_enzy_onestep", "template_relevance"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def molecule_ok(smiles: str) -> bool:
    atoms = heavy_atoms(smiles)
    return 8 <= atoms <= 80 and "." not in str(smiles or "")


def load_positive_targets(path: Path, *, count: int) -> list[dict[str, Any]]:
    if int(count or 0) <= 0:
        return []
    rows = [
        row
        for row in read_jsonl(path)
        if bool(row.get("has_usable_live_enzyme_candidate")) and molecule_ok(str(row.get("target_smiles") or ""))
    ]
    rows.sort(
        key=lambda row: (
            -int(row.get("usable_enzyme_candidates") or 0),
            int(row.get("heavy_atoms") or 10**9),
            str(row.get("target_smiles") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        smiles = str(row.get("target_smiles") or "")
        key = str(row.get("chemical_inchikey") or inchikey_from_smiles(smiles))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "target_smiles": smiles,
                "chemical_inchikey": key,
                "label": 1,
                "label_source": "live_enzyme_bridge_probe_positive",
                "usable_enzyme_candidates": int(row.get("usable_enzyme_candidates") or 0),
            }
        )
        if len(out) >= count:
            break
    return out


def positive_chemical_keys(pack_dir: Path) -> set[str]:
    keys: set[str] = set()
    for name in ("exact_bridge_strict.parquet", "similarity_bridge_filtered.parquet"):
        path = pack_dir / name
        if not path.exists():
            continue
        schema_names = set(pq.read_schema(path).names)
        column = "chemical_inchikey" if "chemical_inchikey" in schema_names else "inchikey"
        table = pq.read_table(path, columns=[column])
        keys.update(str(row[column]) for row in table.to_pylist() if row.get(column))
    return keys


def load_negative_targets(pack_dir: Path, *, count: int, seed: int) -> list[dict[str, Any]]:
    positive_keys = positive_chemical_keys(pack_dir)
    rows: list[dict[str, Any]] = []
    for row in pq.read_table(pack_dir / "chemical_product_pool.parquet").to_pylist():
        smiles = str(row.get("canonical_smiles") or "")
        key = str(row.get("inchikey") or inchikey_from_smiles(smiles))
        if not key or key in positive_keys or not molecule_ok(smiles):
            continue
        rows.append(
            {
                "target_smiles": smiles,
                "chemical_inchikey": key,
                "label": 0,
                "label_source": "chemical_product_without_bridge_hit",
                "usable_enzyme_candidates": 0,
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["chemical_inchikey"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= count:
            break
    return out


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    positives = load_positive_targets(args.probe_rows, count=max(0, int(args.positives)))
    negatives = load_negative_targets(args.pack_dir, count=max(0, int(args.negatives)), seed=int(args.seed))
    rows = [*positives, *negatives]
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    return rows


def engine_for_policy(full_engine: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == "native_no_enzyme":
        keep = CHEMICAL_SOURCES
    else:
        keep = ENZYME_SOURCES | CHEMICAL_SOURCES
    return {key: value for key, value in full_engine.items() if value is not None and key in keep}


def gate_for_policy(policy: str, retriever: BridgeRetrieverV0) -> SourceGate:
    if policy in {"bridge_gate_verifier", "bridge_gate_verifier_bonus2"}:
        return BridgeAwareSourceGate(SourceGate(), retriever=retriever, require_verifier_pass=True)
    return SourceGate()


def route_sources(result: Any) -> list[str]:
    if result is None:
        return []
    return [str(getattr(slot, "source", "") or "") for slot in getattr(result.board, "slots", [])]


def route_steps(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    steps: list[dict[str, Any]] = []
    for slot in getattr(result.board, "slots", []):
        reaction = str(getattr(slot, "reaction_smiles", "") or "")
        if not reaction:
            continue
        steps.append(
            {
                "source": str(getattr(slot, "source", "") or ""),
                "product": str(getattr(slot, "product", "") or ""),
                "main_reactant": str(getattr(slot, "main_reactant", "") or ""),
                "reaction_smiles": reaction,
                "ec": str(getattr(slot, "ec", "") or ""),
            }
        )
    return steps


def run_one(
    target: dict[str, Any],
    *,
    policy: str,
    full_engine: dict[str, Any],
    retriever: BridgeRetrieverV0,
    enzyme_sp_verifier: EnzymeSPVerifierV1Scorer | None,
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
        enzyme_sp_verifier=enzyme_sp_verifier if policy == "bridge_gate_verifier_sp_v1" else None,
    )
    planner.proposals.source_gate = gate_for_policy(policy, retriever)
    started = time.monotonic()
    old_bonus = os.environ.get("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS")
    if policy == "bridge_gate_verifier_bonus2":
        os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = "2.0"
    else:
        os.environ.pop("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS", None)
    try:
        results = planner.search(str(target["target_smiles"]), n_results=n_results)
    finally:
        if old_bonus is None:
            os.environ.pop("AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS", None)
        else:
            os.environ["AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS"] = old_bonus
    elapsed = time.monotonic() - started
    result_rows = []
    for idx, result in enumerate(results, start=1):
        sources = route_sources(result)
        result_rows.append(
            {
                "rank": idx,
                "selected_sources": sources,
                "selected_enzyme_route": any(source in ENZYME_SOURCES for source in sources),
                "score": float(getattr(result, "score", 0.0) or 0.0),
                "quality_vector": dict(getattr(result, "quality_vector", {}) or {}),
                "steps": route_steps(result),
            }
        )
    source_stats = planner.stats.to_dict().get("proposal_source_stats", {})
    enzyme_calls = sum(int((source_stats.get(source) or {}).get("calls") or 0) for source in ENZYME_SOURCES)
    enzyme_sp_rejections = int(planner.stats.to_dict().get("enzyme_sp_verifier_rejections") or 0)
    return {
        **target,
        "target_canonical": canonical_smiles(str(target["target_smiles"])) or str(target["target_smiles"]),
        "policy": policy,
        "available_sources": sorted(engine),
        "result_count": len(results),
        "selected_enzyme_routes": sum(int(row["selected_enzyme_route"]) for row in result_rows),
        "selected_enzyme_route": any(row["selected_enzyme_route"] for row in result_rows),
        "results": result_rows,
        "enzyme_source_calls": enzyme_calls,
        "enzyme_sp_verifier_rejections": enzyme_sp_rejections,
        "stats": planner.stats.to_dict(),
        "elapsed_s": round(float(elapsed), 3),
    }


def summarize(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy]
    positives = [row for row in subset if int(row["label"]) == 1]
    negatives = [row for row in subset if int(row["label"]) == 0]
    selected = [row for row in subset if row["selected_enzyme_route"]]
    true_selected = [row for row in selected if int(row["label"]) == 1]
    false_selected = [row for row in selected if int(row["label"]) == 0]
    return {
        "policy": policy,
        "targets": len(subset),
        "positives": len(positives),
        "negatives": len(negatives),
        "targets_with_selected_enzyme_route": len(selected),
        "selected_enzyme_routes": sum(int(row["selected_enzyme_routes"]) for row in subset),
        "true_selected_targets": len(true_selected),
        "false_selected_targets": len(false_selected),
        "selected_target_precision": len(true_selected) / len(selected) if selected else 0.0,
        "selected_target_recall": len(true_selected) / len(positives) if positives else 0.0,
        "false_enzyme_target_rate": len(false_selected) / len(negatives) if negatives else 0.0,
        "mean_enzyme_source_calls": _mean(row["enzyme_source_calls"] for row in subset),
        "mean_enzyme_sp_rejections": _mean(row.get("enzyme_sp_verifier_rejections", 0) for row in subset),
        "mean_elapsed_s": _mean(row["elapsed_s"] for row in subset),
    }


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Live Policy Benchmark v0",
        "",
        "Fair live-provider policy benchmark with a fresh live-engine wrapper per policy.",
        "",
        f"- Targets: {report['inputs']['targets']} ({report['inputs']['positives']} positive, {report['inputs']['negatives']} negative)",
        f"- Search: depth={report['inputs']['max_depth']}, branch={report['inputs']['branch_factor']}, budget={report['inputs']['expansion_budget']}, n_results={report['inputs']['n_results']}",
        "",
        "| policy | selected targets | true | false | precision | recall | false rate | mean enzyme calls | mean v1 reject | mean elapsed s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {targets_with_selected_enzyme_route} | {true_selected_targets} | {false_selected_targets} | "
            "{selected_target_precision:.4f} | {selected_target_recall:.4f} | {false_enzyme_target_rate:.4f} | "
            "{mean_enzyme_source_calls:.2f} | {mean_enzyme_sp_rejections:.2f} | {mean_elapsed_s:.2f} |".format(**row)
        )
    lines.append("")
    lines.append(report["conclusion"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair bridge live policy benchmark")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=6)
    parser.add_argument("--expansion-budget", type=int, default=12)
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument(
        "--reuse-live-engine",
        action="store_true",
        help="Reuse one live-engine wrapper across policies to measure warm service behavior.",
    )
    args = parser.parse_args()

    started = time.monotonic()
    targets = load_targets(args)
    policies = [
        "native_no_enzyme",
        "ungated_default_source_gate",
        "bridge_gate_verifier",
        "bridge_gate_verifier_sp_v1",
        "bridge_gate_verifier_bonus2",
    ]
    rows: list[dict[str, Any]] = []
    enzyme_sp_verifier = EnzymeSPVerifierV1Scorer()
    shared_engine = build_live_retro_engine() if args.reuse_live_engine else None
    for policy in policies:
        full_engine = shared_engine if shared_engine is not None else build_live_retro_engine()
        retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
        for target in targets:
            rows.append(
                run_one(
                    target,
                    policy=policy,
                    full_engine=full_engine,
                    retriever=retriever,
                    enzyme_sp_verifier=enzyme_sp_verifier,
                    max_depth=max(1, int(args.max_depth)),
                    branch_factor=max(1, int(args.branch_factor)),
                    expansion_budget=max(1, int(args.expansion_budget)),
                    n_results=max(1, int(args.n_results)),
                )
            )
    summaries = [summarize(rows, policy) for policy in policies]
    gated = next(row for row in summaries if row["policy"] == "bridge_gate_verifier")
    sp_v1 = next(row for row in summaries if row["policy"] == "bridge_gate_verifier_sp_v1")
    bonus = next(row for row in summaries if row["policy"] == "bridge_gate_verifier_bonus2")
    conclusion = (
        "SP-v1 gate selected "
        f"{sp_v1['targets_with_selected_enzyme_route']} targets versus {gated['targets_with_selected_enzyme_route']} "
        f"for bridge-gated baseline, with mean v1 rejections {sp_v1['mean_enzyme_sp_rejections']:.2f}; "
        f"bonus policy selected {bonus['targets_with_selected_enzyme_route']} targets. "
        f"False target rates: gated={gated['false_enzyme_target_rate']:.4f}, "
        f"sp_v1={sp_v1['false_enzyme_target_rate']:.4f}, bonus={bonus['false_enzyme_target_rate']:.4f}."
    )
    report = {
        "schema_version": "bridge_live_policy_benchmark_v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "targets": len(targets),
            "positives": sum(1 for row in targets if int(row["label"]) == 1),
            "negatives": sum(1 for row in targets if int(row["label"]) == 0),
            "max_depth": int(args.max_depth),
            "branch_factor": int(args.branch_factor),
            "expansion_budget": int(args.expansion_budget),
            "n_results": int(args.n_results),
            "reuse_live_engine": bool(args.reuse_live_engine),
        },
        "policies": summaries,
        "conclusion": conclusion,
        "retro_cache_stats_last_policy": retro_engine_cache_stats(full_engine),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "bridge_live_policy_benchmark_rows.jsonl"
    report_json = args.output_dir / "bridge_live_policy_benchmark_report.json"
    report_md = args.output_dir / "bridge_live_policy_benchmark_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_json),
                "rows": str(rows_path),
                "conclusion": conclusion,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
