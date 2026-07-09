"""Select benchmark targets likely to exercise condition-aware candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


ENZYMATIC_SOURCES = {"enzyformer", "v3_retrieval", "enzexpand"}


def select_condition_rich_targets(
    *,
    benchmark_path: Path,
    pack_dir: Path,
    output_path: Path,
    report_path: Path,
    limit: int = 12,
) -> list[dict[str, Any]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    by_target = {canonical_smiles(row.get("target_smiles") or ""): row for row in benchmark}
    stats = candidate_target_stats(pack_dir / "candidate_ranking.jsonl")
    ranked = []
    for target, row in by_target.items():
        stat = stats.get(target) or Counter()
        score = target_score(stat)
        if score <= 0:
            continue
        item = dict(row)
        item["_condition_rich_score"] = round(score, 3)
        item["_condition_rich_stats"] = dict(stat)
        ranked.append(item)
    ranked.sort(
        key=lambda row: (
            row["_condition_rich_score"],
            row["_condition_rich_stats"].get("candidate_has_T_and_pH", 0),
            row["_condition_rich_stats"].get("v3_retrieval", 0),
        ),
        reverse=True,
    )
    selected = ranked[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown(selected, benchmark_path, pack_dir), encoding="utf-8")
    return selected


def candidate_target_stats(path: Path) -> dict[str, Counter]:
    stats: dict[str, Counter] = defaultdict(Counter)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = canonical_smiles(row.get("target_smiles") or "")
        cand = row.get("candidate") or {}
        source = cand.get("source") or "unknown"
        stat = stats[target]
        stat["candidate_total"] += 1
        stat[source] += 1
        if source in ENZYMATIC_SOURCES:
            stat["enzymatic_source"] += 1
        if cand.get("T") is not None:
            stat["candidate_has_T"] += 1
        if cand.get("pH") is not None:
            stat["candidate_has_pH"] += 1
        if cand.get("T") is not None and cand.get("pH") is not None:
            stat["candidate_has_T_and_pH"] += 1
        evidence = cand.get("evidence") or {}
        if cand.get("doi") or evidence.get("doi"):
            stat["candidate_has_doi"] += 1
        if cand.get("uniprot_accession") or evidence.get("uniprot_accession"):
            stat["candidate_has_uniprot"] += 1
        if cand.get("cofactor") or evidence.get("cofactor"):
            stat["candidate_has_cofactor"] += 1
    return stats


def target_score(stat: Counter) -> float:
    total = max(stat.get("candidate_total", 0), 1)
    return (
        5.0 * stat.get("candidate_has_T_and_pH", 0)
        + 3.0 * stat.get("v3_retrieval", 0)
        + 2.0 * stat.get("candidate_has_doi", 0)
        + 2.0 * stat.get("candidate_has_uniprot", 0)
        + 1.0 * stat.get("enzymatic_source", 0)
    ) / total


def report_markdown(rows: list[dict[str, Any]], benchmark_path: Path, pack_dir: Path) -> str:
    lines = [
        "# Condition-Rich Benchmark Slice",
        "",
        f"Benchmark: `{benchmark_path}`",
        f"Training pack: `{pack_dir}`",
        f"Selected targets: `{len(rows)}`",
        "",
        "| Rank | Score | Target | Domain | Depth | Total cands | v3 | enzymatic | T+pH | DOI | UniProt |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, 1):
        stat = row.get("_condition_rich_stats") or {}
        lines.append(
            f"| {idx} | {row.get('_condition_rich_score')} | `{row.get('target_smiles')}` | "
            f"{row.get('route_domain') or ''} | {row.get('depth') or ''} | "
            f"{stat.get('candidate_total', 0)} | {stat.get('v3_retrieval', 0)} | "
            f"{stat.get('enzymatic_source', 0)} | {stat.get('candidate_has_T_and_pH', 0)} | "
            f"{stat.get('candidate_has_doi', 0)} | {stat.get('candidate_has_uniprot', 0)} |"
        )
    lines.extend([
        "",
        "This slice is intended for route-level ablations of condition-aware candidate reranking.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select condition-rich benchmark targets")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--pack-dir", default="results/shared/training_pack/condition_20260507")
    ap.add_argument("--output", default="data/benchmark_condition_rich_20260507.json")
    ap.add_argument("--report", default="results/v2/benchmark_condition_rich_20260507.md")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    selected = select_condition_rich_targets(
        benchmark_path=Path(args.benchmark),
        pack_dir=Path(args.pack_dir),
        output_path=Path(args.output),
        report_path=Path(args.report),
        limit=args.limit,
    )
    print(json.dumps({
        "output": args.output,
        "report": args.report,
        "selected": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
