"""Derived route coverage and graph metrics; no authority to admit references."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def route_graph_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    """Count target ancestors, including convergent branches, separately from LLS."""
    compounds = {c["compound_id"]: c for c in candidate["compounds"]}
    available = {cid for cid, c in compounds.items() if c.get("role") in {"starting_material", "reagent"}}
    depth = dict.fromkeys(available, 0)
    producer: dict[str, dict[str, Any]] = {}
    for step in candidate["steps"]:
        product = step["product_compound_id"]
        if product in depth:
            raise RuntimeError(f"structured_route_product_already_available:{product}")
        precursors = step["precursor_compound_ids"]
        if not precursors or any(cid not in depth for cid in precursors):
            raise RuntimeError(f"structured_route_graph_not_topological:{step['step_id']}")
        depth[product] = 1 + max(depth[cid] for cid in precursors)
        producer[product] = step
    target = candidate["target_compound_id"]
    if target not in producer:
        raise RuntimeError("structured_route_target_not_produced")
    ancestors: set[str] = set()
    frontier = [target]
    while frontier:
        cid = frontier.pop()
        if cid in ancestors:
            continue
        ancestors.add(cid)
        if cid in producer:
            frontier.extend(producer[cid]["precursor_compound_ids"])
    unrelated = set(producer) - ancestors
    if unrelated:
        raise RuntimeError(f"structured_route_unrelated_branch:{','.join(sorted(unrelated))}")
    return {
        "total_operation_count": len(producer),
        "longest_linear_step_count": depth[target],
        "starting_compound_count": len(ancestors & available),
        "convergent_union_count": sum(sum(cid in producer for cid in step["precursor_compound_ids"]) > 1 for step in producer.values()),
    }


def source_step_key(candidate: dict[str, Any], step: dict[str, Any]) -> str:
    # Source labels identify the same shared intermediate across target routes.
    # Conditions distinguish route variants; target-specific step IDs do not.
    payload = {
        "paper_id": candidate["paper_id"],
        "precursors": sorted(step["precursor_labels"]),
        "product": step["product_label"],
        "conditions": step.get("conditions") or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def structured_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "complete_ordered_route_candidates": len(candidates),
        "structured_route_papers": len({c["paper_id"] for c in candidates}),
        "route_operation_instances": sum(len(c["steps"]) for c in candidates),
        "unique_paper_bound_source_steps": len({source_step_key(c, s) for c in candidates for s in c["steps"]}),
    }


def build_inventory(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p1 = dataset_dir / "curation_candidates/p1_scope"
    targets = [r for r in read_jsonl(dataset_dir / "target_slots.jsonl") + read_jsonl(p1 / "candidate-target-slots.jsonl") if r.get("slot_class") in {"primary", "primary_candidate"}]
    visuals = {r["target_slot_id"]: r for r in read_jsonl(dataset_dir / "visual_structure_candidates.jsonl") + read_jsonl(p1 / "visual-structure-candidates.jsonl")}
    leads = {r["target_slot_id"]: r for r in read_jsonl(dataset_dir / "route_evidence_candidates.jsonl") + read_jsonl(p1 / "route-evidence-candidates.jsonl")}
    receipts = {r["paper_id"]: r for r in read_jsonl(dataset_dir / "source_package_receipts.jsonl") + read_jsonl(dataset_dir / "p1_source_package_receipts.jsonl")}
    findings_by_target: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((dataset_dir / "curation_candidates").glob("extraction_findings_*.jsonl")):
        for finding in read_jsonl(path):
            for tid in finding.get("target_slot_ids") or []:
                findings_by_target.setdefault(tid, []).append({"finding_type": finding["finding_type"], "required_next_action": finding["required_next_action"], "source_record_path": path.relative_to(dataset_dir).as_posix()})
    structured = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((dataset_dir / "curation_candidates/structured_routes").glob("*.json"))]
    by_target = {r["target_slot_id"]: r for r in structured}
    if len(by_target) != len(structured):
        raise ValueError("duplicate structured target routes; select a preferred route explicitly")
    target_ids = {t["target_slot_id"] for t in targets}
    if len(target_ids) != len(targets):
        raise ValueError("duplicate target slots in combined cohorts")
    unknown = set(by_target) - target_ids
    if unknown:
        raise ValueError(f"structured route targets outside candidate universe: {sorted(unknown)}")
    rows = []
    for target in targets:
        tid = target["target_slot_id"]
        route = by_target.get(tid)
        v = visuals.get(tid, {})
        lead = leads.get(tid, {})
        receipt = receipts.get(target["paper_id"], {})
        source_ok = bool(receipt.get("source_package_acquired"))
        blockers = []
        if not source_ok:
            blockers.append("source_package_missing")
        coverage = lead.get("source_coverage") or {}
        if coverage.get("si_duplicates_article") and not coverage.get("si_text_inspected"):
            blockers.append("supporting_information_duplicates_article")
        if v.get("visual_status") != "exact_source_structure_candidate":
            blockers.append("target_structure_or_stereochemistry_unresolved")
        if not route:
            blockers.append("ordered_route_not_extracted")
        if not target.get("runnable"):
            blockers.append("independent_expert_review_pending")
        rows.append({
            "target_slot_id": tid, "paper_id": target["paper_id"], "doi": target["doi"],
            "target_name": target["target_name"], "cohort": "P0" if target["slot_class"] == "primary" else "P1",
            "split_group_id": target.get("article_family_id") or target["paper_id"],
            "source_package_acquired": source_ok, "visual_status": v.get("visual_status", "missing"),
            "route_evidence_status": lead.get("extraction_status", "not_extracted"),
            "route_passage_count": len(lead.get("evidence_passages") or []),
            "source_coverage": lead.get("source_coverage") or {"status": "legacy_extraction_source_coverage_unknown"},
            "ordered_route_candidate": bool(route),
            "structured_route_path": f"curation_candidates/structured_routes/{tid}.json" if route else "",
            "graph_metrics": route_graph_metrics(route) if route else {},
            "source_review_findings": findings_by_target.get(tid, []),
            "blockers": blockers, "admission_authority": False,
        })
    rows.sort(key=lambda r: (not r["ordered_route_candidate"], r["paper_id"], r["target_name"]))
    paper_ids = {r["paper_id"] for r in rows}
    summary = {
        "schema_version": "recent_total_synthesis_route_coverage.v1", "admission_authority": False,
        "candidate_papers": len(paper_ids), "candidate_targets": len(rows),
        "source_packages_acquired": sum(bool(receipts.get(pid, {}).get("source_package_acquired")) for pid in paper_ids),
        **structured_counts(structured),
        "targets_with_passage_leads": sum(r["route_passage_count"] > 0 for r in rows),
        "papers_with_duplicate_article_mislabeled_si": len({r["paper_id"] for r in rows if r["source_coverage"].get("si_duplicates_article")}),
        "visual_status_counts": dict(Counter(r["visual_status"] for r in rows)),
        "blocker_counts": dict(Counter(b for r in rows for b in r["blockers"])),
        "runnable_primary_targets": sum(bool(t.get("runnable")) for t in targets if t.get("slot_class") == "primary"),
        "counting_note": "Source-step deduplication uses paper, precursor/product labels and conditions. It is not a count of independent strategies or experiments. Split by article family and inspect cross-paper route overlap before evaluation.",
    }
    return rows, summary


def write_inventory(dataset_dir: Path) -> dict[str, Any]:
    rows, summary = build_inventory(dataset_dir)
    (dataset_dir / "route_coverage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (dataset_dir / "route_coverage.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    complete = [r for r in rows if r["ordered_route_candidate"]]
    table = ["| 目标 | DOI | 操作总数 | 最长线性步数 | 候选 |", "|---|---|---:|---:|---|"]
    table.extend(f"| {r['target_name']} | {r['doi']} | {r['graph_metrics']['total_operation_count']} | {r['graph_metrics']['longest_linear_step_count']} | [JSON]({r['structured_route_path']}) |" for r in complete)
    report = f"""# 天然产物路线提取覆盖

此表由 `python scripts/recent_total_synthesis_route_inventory.py` 从当前数据生成。所有自动提取均为待复核候选。

- 候选范围：{summary['candidate_papers']} 篇论文，{summary['candidate_targets']} 个目标；{summary['source_packages_acquired']} 篇已有来源包。
- 完整有序路线候选：{summary['complete_ordered_route_candidates']} 条，来自 {summary['structured_route_papers']} 篇论文。
- 按目标展开的操作记录：{summary['route_operation_instances']}；按论文、化合物编号和条件合并共享操作：{summary['unique_paper_bound_source_steps']}。
- 有段落线索的目标：{summary['targets_with_passage_leads']}。段落线索不能算完整路线。
- 可直接运行的正式 benchmark 目标：{summary['runnable_primary_targets']}。

## 已完成的有序候选

{chr(10).join(table)}

## 剩余缺口

{chr(10).join(f'- `{key}`：{value} 个目标。' for key, value in summary['blocker_counts'].items())}

完整逐目标清单在 [route_coverage.jsonl](route_coverage.jsonl)，包括 P0/P1、来源、结构、段落、完整路线和未完成事项。
多目标集体合成应按论文族分组，共享前缀不可跨训练/测试集泄漏；跨论文的同路线复用仍须另行检查。
最长线性步数从目标的反应 DAG 计算。合并收率保留其覆盖步骤，不推导虚假的单步收率。
"""
    (dataset_dir / "ROUTE_EXTRACTION_STATUS.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("benchmarks/recent_total_synthesis"))
    args = parser.parse_args()
    print(json.dumps(write_inventory(args.dataset_dir), ensure_ascii=False, indent=2))
