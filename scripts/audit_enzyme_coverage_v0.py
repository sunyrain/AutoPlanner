"""Audit enzyme data/proposal coverage for the next mainline stage."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents
from cascade_planner.cascadeboard.enz_retrieval import retrieve_enzymatic_reactions
from scripts.run_bridge_live_policy_benchmark_v0 import load_targets


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/enzyme_coverage_audit_v0_20260528")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit enzyme coverage and retrieval proposal gaps")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-targets", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args)
    if args.max_targets > 0:
        targets = targets[: int(args.max_targets)]
    inventory = data_inventory(args.pack_dir)
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
    rows = []
    for target in targets:
        rows.append(audit_target(target, retriever=retriever, top_k=max(1, int(args.top_k))))
    report = {
        "schema_version": "enzyme_coverage_audit_v0",
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "positives": int(args.positives),
            "negatives": int(args.negatives),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "targets": len(targets),
        },
        "inventory": inventory,
        "target_summary": summarize_rows(rows),
        "conclusion": conclusion(inventory, rows),
    }
    rows_path = args.output_dir / "enzyme_coverage_audit_rows.jsonl"
    report_json = args.output_dir / "enzyme_coverage_audit_report.json"
    report_md = args.output_dir / "enzyme_coverage_audit_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "rows": str(rows_path), "conclusion": report["conclusion"]}, ensure_ascii=False, indent=2))


def data_inventory(pack_dir: Path) -> dict[str, Any]:
    enzyme_rxn = pq.read_table(pack_dir / "enzyme_reaction_pool.parquet")
    enzyme_sp = pq.read_table(pack_dir / "enzyme_substrate_product_pool.parquet")
    bridge = pq.read_table(pack_dir / "bridge_candidates_scored.parquet")
    chemical = pq.read_table(pack_dir / "chemical_product_pool.parquet")
    ec_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    multi_ec = 0
    for row in enzyme_rxn.to_pylist():
        ecs = json_list(row.get("ec_numbers_json"))
        if len(set(ecs)) > 1:
            multi_ec += 1
        for ec in ecs:
            head = str(ec).split(".", 1)[0]
            ec_counter[head if head in {"1", "2", "3", "4", "5", "6", "7"} else "unknown"] += 1
        for source, count in json_dict(row.get("source_counts_json")).items():
            source_counter[source] += int(count or 0)
    return {
        "enzyme_reaction_rows": enzyme_rxn.num_rows,
        "enzyme_substrate_product_molecules": enzyme_sp.num_rows,
        "chemical_product_molecules": chemical.num_rows,
        "scored_bridge_candidates": bridge.num_rows,
        "enzyme_reactions_with_multiple_ec": multi_ec,
        "enzyme_reaction_ec1_counts": dict(sorted(ec_counter.items())),
        "enzyme_reaction_source_counts_top": dict(source_counter.most_common(12)),
    }


def audit_target(target: dict[str, Any], *, retriever: BridgeRetrieverV0, top_k: int) -> dict[str, Any]:
    smiles = str(target.get("target_smiles") or "")
    bridge_hits = retriever.retrieve(smiles, top_k=8, require_verifier_pass=True)
    ec1s = bridge_ec1s(bridge_hits)
    v3_no_ec = retrieve_enzymatic_reactions(smiles, top_k=top_k)
    precedent_no_ec = retrieve_enzyme_precedents(smiles, top_k=top_k)
    by_ec = []
    for ec1 in ec1s[:3]:
        by_ec.append(
            {
                "ec1": ec1,
                "v3_hits": len(retrieve_enzymatic_reactions(smiles, ec_class=str(ec1), top_k=top_k)),
                "precedent_hits": len(retrieve_enzyme_precedents(smiles, ec_class=str(ec1), top_k=top_k)),
            }
        )
    return {
        "target_smiles": smiles,
        "label": int(target.get("label") or 0),
        "label_source": target.get("label_source") or "",
        "bridge_hit_count": len(bridge_hits),
        "bridge_ec1s": ec1s,
        "v3_no_ec_hits": len(v3_no_ec),
        "precedent_no_ec_hits": len(precedent_no_ec),
        "v3_no_ec_top_score": float(v3_no_ec[0].get("score") or 0.0) if v3_no_ec else 0.0,
        "precedent_no_ec_top_score": float(precedent_no_ec[0].get("score") or 0.0) if precedent_no_ec else 0.0,
        "by_bridge_ec1": by_ec,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    return {
        "targets": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "bridge_positive_targets": sum(1 for row in rows if row["bridge_hit_count"] > 0),
        "v3_targets_with_hits": sum(1 for row in rows if row["v3_no_ec_hits"] > 0),
        "precedent_targets_with_hits": sum(1 for row in rows if row["precedent_no_ec_hits"] > 0),
        "v3_positive_hit_rate": ratio(sum(1 for row in positives if row["v3_no_ec_hits"] > 0), len(positives)),
        "precedent_positive_hit_rate": ratio(sum(1 for row in positives if row["precedent_no_ec_hits"] > 0), len(positives)),
        "v3_negative_hit_rate": ratio(sum(1 for row in negatives if row["v3_no_ec_hits"] > 0), len(negatives)),
        "precedent_negative_hit_rate": ratio(sum(1 for row in negatives if row["precedent_no_ec_hits"] > 0), len(negatives)),
        "mean_v3_hits": mean(row["v3_no_ec_hits"] for row in rows),
        "mean_precedent_hits": mean(row["precedent_no_ec_hits"] for row in rows),
    }


def conclusion(inventory: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summary = summarize_rows(rows)
    return (
        f"Enzyme pool has {inventory['enzyme_reaction_rows']} reaction precedents and "
        f"{inventory['enzyme_substrate_product_molecules']} substrate/product molecules. "
        f"On audited targets, v3 retrieval hit rate={summary['v3_targets_with_hits']}/{summary['targets']}, "
        f"large precedent retrieval hit rate={summary['precedent_targets_with_hits']}/{summary['targets']}. "
        "This supports moving effort from verifier-only work to enzyme proposal coverage."
    )


def render_markdown(report: dict[str, Any]) -> str:
    inventory = report["inventory"]
    summary = report["target_summary"]
    lines = [
        "# Enzyme Coverage Audit v0",
        "",
        "Scope: enzyme data inventory and retrieval coverage audit.",
        "",
        "## Data Inventory",
        "",
        f"- Enzyme reaction precedents: {inventory['enzyme_reaction_rows']}",
        f"- Enzyme substrate/product molecules: {inventory['enzyme_substrate_product_molecules']}",
        f"- Chemical product molecules: {inventory['chemical_product_molecules']}",
        f"- Scored bridge candidates: {inventory['scored_bridge_candidates']}",
        "",
        "## Target Coverage",
        "",
        f"- Targets: {summary['targets']} ({summary['positives']} positive, {summary['negatives']} negative)",
        f"- v3 targets with hits: {summary['v3_targets_with_hits']}",
        f"- large precedent targets with hits: {summary['precedent_targets_with_hits']}",
        f"- v3 positive hit rate: {summary['v3_positive_hit_rate']:.4f}",
        f"- large precedent positive hit rate: {summary['precedent_positive_hit_rate']:.4f}",
        f"- v3 negative hit rate: {summary['v3_negative_hit_rate']:.4f}",
        f"- large precedent negative hit rate: {summary['precedent_negative_hit_rate']:.4f}",
        "",
        report["conclusion"],
        "",
    ]
    return "\n".join(lines)


def bridge_ec1s(hits: list[Any]) -> list[int]:
    out: list[int] = []
    for hit in hits:
        for ec in getattr(hit, "enzyme_ec_sample", ()) or ():
            head = str(ec or "").split(".", 1)[0]
            if head.isdigit() and 1 <= int(head) <= 7 and int(head) not in out:
                out.append(int(head))
    return out


def json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def json_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


if __name__ == "__main__":
    main()
