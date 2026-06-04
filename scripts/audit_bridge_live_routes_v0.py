"""Audit live bridge/enzyme routes for P5 evidence quality.

The live route and policy benchmarks intentionally count whether an enzyme
source was selected. For P5 we need a stricter view: selected enzyme steps must
not be self-loops, product mismatches, or tiny-reagent artifacts, and final
cards must be clearly marked as partial route evidence unless stock-closed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascadeboard.route_recovery import canonical_smiles

RDLogger.DisableLog("rdApp.*")

DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_route_quality_audit_v0_20260528")
DEFAULT_INPUTS = (
    Path("results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_rows.jsonl"),
    Path("results/shared/bridge_gate_ablation_v0_20260527_bonus2/bridge_live_route_evidence_rows.jsonl"),
    Path("results/shared/bridge_live_policy_benchmark_v0_20260528_smoke/bridge_live_policy_benchmark_rows.jsonl"),
    Path("results/shared/bridge_live_policy_benchmark_v0_20260528_depth1_4p4n/bridge_live_policy_benchmark_rows.jsonl"),
)
ENZYME_SOURCES = {"enzyformer", "enzexpand", "retrorules", "enzyme", "enzymatic"}
PRODUCTION_POLICIES = {
    "normal_bridge_gated",
    "bridge_gate_verifier",
    "bridge_gate_verifier_bonus2",
    "ungated_default_source_gate",
}
HARD_STEP_FLAGS = {"self_loop", "product_mismatch", "no_reactants", "tiny_largest_reactant"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def split_reaction(reaction: str) -> tuple[list[str], list[str]]:
    text = str(reaction or "")
    if ">>" not in text:
        return [], []
    lhs, rhs = text.split(">>", 1)
    return [part.strip() for part in lhs.split(".") if part.strip()], [
        part.strip() for part in rhs.split(".") if part.strip()
    ]


def is_enzyme_step(step: dict[str, Any]) -> bool:
    source = str(step.get("source") or "").lower()
    return source in ENZYME_SOURCES or bool(step.get("ec"))


def bridge_hits_for_row(row: dict[str, Any], retriever: BridgeRetrieverV0) -> list[dict[str, Any]]:
    if isinstance(row.get("bridge_evidence"), list):
        return list(row.get("bridge_evidence") or [])
    target = str(row.get("target_smiles") or "")
    if not target:
        return []
    try:
        hits = retriever.retrieve(target, top_k=3, require_verifier_pass=True)
    except Exception:
        return []
    return [
        {
            "source": hit.source,
            "bridge_direction": hit.bridge_direction,
            "verifier_score": float(hit.verifier_score or 0.0),
            "tanimoto": float(hit.tanimoto),
            "enzyme_ec_sample": list(hit.enzyme_ec_sample[:8]),
        }
        for hit in hits
    ]


def audit_step(step: dict[str, Any], *, min_largest_reactant_ratio: float) -> dict[str, Any]:
    product = str(step.get("product") or "")
    reaction = str(step.get("reaction_smiles") or "")
    reactants, rhs = split_reaction(reaction)
    main = str(step.get("main_reactant") or "")
    if main and main not in reactants:
        reactants.insert(0, main)
    product_can = canonical_smiles(product) or product
    rhs_can = {canonical_smiles(smi) or smi for smi in rhs}
    reactant_can = [canonical_smiles(smi) or smi for smi in reactants if smi]
    product_heavy = heavy_atoms(product)
    reactant_heavies = [heavy_atoms(smi) for smi in reactants]
    largest = max(reactant_heavies, default=0)
    ratio = largest / product_heavy if product_heavy else 0.0
    flags: list[str] = []
    risk_flags: list[str] = []
    if not reaction or ">>" not in reaction:
        flags.append("no_reaction_smiles")
    if not reactants:
        flags.append("no_reactants")
    if product_can and rhs_can and product_can not in rhs_can:
        flags.append("product_mismatch")
    if product_can and product_can in set(reactant_can):
        flags.append("self_loop")
    if product_heavy >= 8 and largest and ratio < min_largest_reactant_ratio:
        flags.append("tiny_largest_reactant")
    if product_heavy >= 8 and largest > int(product_heavy * 1.25):
        risk_flags.append("larger_reactant_than_product")
    if not str(step.get("ec") or ""):
        risk_flags.append("missing_ec")
    elif str(step.get("ec") or "").endswith(".x"):
        risk_flags.append("generic_ec")
    return {
        "source": str(step.get("source") or ""),
        "product": product,
        "main_reactant": main,
        "reaction_smiles": reaction,
        "ec": str(step.get("ec") or ""),
        "product_heavy_atoms": product_heavy,
        "largest_reactant_heavy_atoms": largest,
        "largest_reactant_ratio": round(float(ratio), 4),
        "hard_flags": flags,
        "risk_flags": risk_flags,
    }


def route_quality_level(
    *,
    policy: str,
    hard_flags: set[str],
    bridge_hit_count: int,
    stock_closed: bool,
    route_solved: bool,
) -> str:
    if hard_flags & HARD_STEP_FLAGS:
        return "reject_artifact"
    if bridge_hit_count <= 0:
        return "unverified_no_bridge_support"
    if policy not in PRODUCTION_POLICIES:
        return "diagnostic_only"
    if stock_closed or route_solved:
        return "production_candidate_stock_closed"
    return "production_candidate_partial"


def audit_route(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    input_name: str,
    route_index: int,
    retriever: BridgeRetrieverV0,
    min_largest_reactant_ratio: float,
) -> dict[str, Any]:
    steps = list(result.get("steps") or [])
    enzyme_steps = [step for step in steps if is_enzyme_step(step)]
    audited_steps = [
        audit_step(step, min_largest_reactant_ratio=min_largest_reactant_ratio)
        for step in enzyme_steps
    ]
    hard_flags = sorted({flag for step in audited_steps for flag in step["hard_flags"]})
    risk_flags = sorted({flag for step in audited_steps for flag in step["risk_flags"]})
    bridge_hits = bridge_hits_for_row(row, retriever)
    quality = dict(result.get("quality_vector") or {})
    stock_closed = bool(float(quality.get("stock_closed") or 0.0) > 0.0)
    route_solved = bool(float(quality.get("route_solved") or 0.0) > 0.0)
    policy = str(row.get("policy") or "")
    level = route_quality_level(
        policy=policy,
        hard_flags=set(hard_flags),
        bridge_hit_count=len(bridge_hits),
        stock_closed=stock_closed,
        route_solved=route_solved,
    )
    return {
        "input_name": input_name,
        "target_smiles": row.get("target_smiles") or "",
        "chemical_inchikey": row.get("chemical_inchikey") or "",
        "label": row.get("label", 1 if bridge_hits else 0),
        "policy": policy,
        "route_rank": result.get("rank", route_index),
        "selected_sources": list(result.get("selected_sources") or []),
        "steps": steps,
        "step_count": len(steps),
        "enzyme_step_count": len(enzyme_steps),
        "bridge_hit_count": len(bridge_hits),
        "bridge_evidence": bridge_hits[:3],
        "stock_closed": stock_closed,
        "route_solved": route_solved,
        "route_quality_level": level,
        "hard_flags": hard_flags,
        "risk_flags": risk_flags,
        "audited_enzyme_steps": audited_steps,
        "quality_vector": quality,
    }


def audit_inputs(
    inputs: list[Path],
    *,
    retriever: BridgeRetrieverV0,
    min_largest_reactant_ratio: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in inputs:
        for row in read_jsonl(path):
            for idx, result in enumerate(row.get("results") or [], start=1):
                if not bool(result.get("selected_enzyme_route")):
                    continue
                out.append(
                    audit_route(
                        row,
                        result,
                        input_name=str(path),
                        route_index=idx,
                        retriever=retriever,
                        min_largest_reactant_ratio=min_largest_reactant_ratio,
                    )
                )
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_level = Counter(record["route_quality_level"] for record in records)
    by_policy: dict[str, Counter] = defaultdict(Counter)
    by_level_label: dict[str, Counter] = defaultdict(Counter)
    flag_counts = Counter()
    risk_counts = Counter()
    unique_targets_by_level: dict[str, set[str]] = defaultdict(set)
    unique_targets_by_policy: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_policy[record["policy"]][record["route_quality_level"]] += 1
        by_level_label[record["route_quality_level"]][str(record.get("label"))] += 1
        unique_targets_by_level[record["route_quality_level"]].add(str(record.get("chemical_inchikey") or record["target_smiles"]))
        unique_targets_by_policy[record["policy"]].add(str(record.get("chemical_inchikey") or record["target_smiles"]))
        flag_counts.update(record["hard_flags"])
        risk_counts.update(record["risk_flags"])
    production_records = [
        record
        for record in records
        if record["route_quality_level"] in {"production_candidate_partial", "production_candidate_stock_closed"}
    ]
    production_positive = sum(1 for record in production_records if int(record.get("label") or 0) == 1)
    production_negative = sum(1 for record in production_records if int(record.get("label") or 0) == 0)
    return {
        "routes_audited": len(records),
        "by_quality_level": dict(sorted(by_level.items())),
        "by_policy": {policy: dict(sorted(counter.items())) for policy, counter in sorted(by_policy.items())},
        "by_quality_level_label": {
            level: dict(sorted(counter.items())) for level, counter in sorted(by_level_label.items())
        },
        "unique_targets_by_quality_level": {
            level: len(values) for level, values in sorted(unique_targets_by_level.items())
        },
        "unique_targets_by_policy": {
            policy: len(values) for policy, values in sorted(unique_targets_by_policy.items())
        },
        "hard_flag_counts": dict(sorted(flag_counts.items())),
        "risk_flag_counts": dict(sorted(risk_counts.items())),
        "production_candidate_partial": by_level.get("production_candidate_partial", 0),
        "production_candidate_stock_closed": by_level.get("production_candidate_stock_closed", 0),
        "production_candidate_positive_routes": production_positive,
        "production_candidate_negative_routes": production_negative,
        "reject_artifact": by_level.get("reject_artifact", 0),
    }


def render_markdown(report: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Bridge Live Route Quality Audit v0",
        "",
        "Audits selected live enzyme routes for artifact flags and P5 evidence quality.",
        "",
        f"- Routes audited: {report['summary']['routes_audited']}",
        f"- Production partial candidates: {report['summary']['production_candidate_partial']}",
        f"- Stock-closed production candidates: {report['summary']['production_candidate_stock_closed']}",
        f"- Production positive routes: {report['summary']['production_candidate_positive_routes']}",
        f"- Production negative routes: {report['summary']['production_candidate_negative_routes']}",
        f"- Reject artifacts: {report['summary']['reject_artifact']}",
        "",
        "## Quality Levels",
        "",
        "| level | count |",
        "|---|---:|",
    ]
    for level, count in report["summary"]["by_quality_level"].items():
        lines.append(f"| {level} | {count} |")
    lines.extend(["", "## By Policy", "", "| policy | levels |", "|---|---|"])
    for policy, counts in report["summary"]["by_policy"].items():
        text = ", ".join(f"{key}:{value}" for key, value in counts.items())
        lines.append(f"| {policy} | {text} |")
    lines.extend(["", "## Flags", ""])
    lines.append(f"- Hard flags: {json.dumps(report['summary']['hard_flag_counts'], ensure_ascii=False)}")
    lines.append(f"- Risk flags: {json.dumps(report['summary']['risk_flag_counts'], ensure_ascii=False)}")
    lines.extend(["", "## Production Candidate Examples", ""])
    candidates = [
        record
        for record in records
        if record["route_quality_level"] in {"production_candidate_partial", "production_candidate_stock_closed"}
    ]
    for record in candidates[:20]:
        lines.extend(
            [
                f"### {record['policy']} rank {record['route_rank']}",
                "",
                f"- Target: `{record['target_smiles']}`",
                f"- Sources: {', '.join(record['selected_sources'])}",
                f"- Steps: {record['step_count']}; enzyme steps: {record['enzyme_step_count']}",
                f"- Bridge hits: {record['bridge_hit_count']}",
                f"- Quality: `{record['route_quality_level']}`",
                f"- Risk flags: {', '.join(record['risk_flags']) if record['risk_flags'] else 'none'}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live bridge enzyme routes")
    parser.add_argument("--input", dest="inputs", action="append", type=Path)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-largest-reactant-heavy-ratio", type=float, default=0.35)
    args = parser.parse_args()

    inputs = args.inputs or [path for path in DEFAULT_INPUTS if path.exists()]
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
    records = audit_inputs(
        inputs,
        retriever=retriever,
        min_largest_reactant_ratio=max(0.0, float(args.min_largest_reactant_heavy_ratio)),
    )
    report = {
        "schema_version": "bridge_live_route_quality_audit_v0",
        "inputs": [str(path) for path in inputs],
        "parameters": {
            "pack_dir": str(args.pack_dir),
            "min_largest_reactant_heavy_ratio": float(args.min_largest_reactant_heavy_ratio),
        },
        "summary": summarize(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "bridge_live_route_quality_audit_rows.jsonl"
    report_json = args.output_dir / "bridge_live_route_quality_audit_report.json"
    report_md = args.output_dir / "bridge_live_route_quality_audit_report.md"
    rows_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report, records), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_json),
                "rows": str(rows_path),
                "summary": report["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
