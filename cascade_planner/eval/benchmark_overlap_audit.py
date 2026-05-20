"""Audit benchmark target/reaction overlap against training packs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


VNEXT_FILES = ["route_states.jsonl", "candidate_pools.jsonl", "step_pairs.jsonl", "search_transitions.jsonl"]
TRAINING_PACK_FILES = ["route_value.jsonl", "candidate_ranking.jsonl", "skeleton_prior.jsonl", "failure_diagnosis.jsonl"]


def audit_benchmark_overlap(benchmark_path: Path, pack_paths: list[Path]) -> dict[str, Any]:
    benchmark = _read_benchmark(benchmark_path)
    bench_targets, bench_gt_rxns, bench_stats = _benchmark_sets(benchmark)
    pack_reports = []
    total_target_overlap: set[str] = set()
    total_rxn_overlap: set[str] = set()
    for pack in pack_paths:
        report = _audit_pack(pack, bench_targets, bench_gt_rxns)
        pack_reports.append(report)
        total_target_overlap.update(report["target_overlap_values"])
        total_rxn_overlap.update(report["gt_reaction_overlap_values"])
    return {
        "benchmark": {
            "path": str(benchmark_path),
            **bench_stats,
        },
        "packs": pack_reports,
        "summary": {
            "n_packs": len(pack_reports),
            "target_overlap_count": len(total_target_overlap),
            "gt_reaction_overlap_count": len(total_rxn_overlap),
            "has_target_overlap": bool(total_target_overlap),
            "has_gt_reaction_overlap": bool(total_rxn_overlap),
            "blind_safe": not total_target_overlap and not total_rxn_overlap,
        },
    }


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported benchmark format: {path}")


def _benchmark_sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], dict[str, Any]]:
    targets = [str(row.get("target_smiles") or "") for row in rows if row.get("target_smiles")]
    gt_rxns = []
    domain_counts = Counter()
    depth_counts = Counter()
    transformation_counts = Counter()
    doi_counts = Counter()
    for row in rows:
        domain_counts[str(row.get("route_domain") or "unknown")] += 1
        depth = int(row.get("depth") or len(row.get("gt_route") or []) or 0)
        depth_counts[str(depth)] += 1
        doi_counts[str(row.get("doi") or "")] += 1
        for step in row.get("gt_route") or []:
            rxn = str((step or {}).get("rxn_smiles") or "")
            if rxn:
                gt_rxns.append(rxn)
            transformation_counts[str((step or {}).get("transformation") or "unknown")] += 1
    return set(targets), set(gt_rxns), {
        "n_rows": len(rows),
        "unique_targets": len(set(targets)),
        "duplicate_targets": len(targets) - len(set(targets)),
        "gt_step_count": len(gt_rxns),
        "unique_gt_reactions": len(set(gt_rxns)),
        "duplicate_gt_reactions": len(gt_rxns) - len(set(gt_rxns)),
        "unique_doi": len([key for key in doi_counts if key]),
        "domain_counts": dict(domain_counts),
        "depth_counts": dict(sorted(depth_counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)),
        "top_transformations": transformation_counts.most_common(20),
        "duplicate_doi": {key: value for key, value in doi_counts.items() if key and value > 1},
    }


def _audit_pack(pack: Path, bench_targets: set[str], bench_gt_rxns: set[str]) -> dict[str, Any]:
    files = _pack_files(pack)
    file_reports = []
    target_overlap_values: set[str] = set()
    rxn_overlap_values: set[str] = set()
    for file_path in files:
        report, targets, rxns = _audit_jsonl(file_path, bench_targets, bench_gt_rxns)
        file_reports.append(report)
        target_overlap_values.update(targets & bench_targets)
        rxn_overlap_values.update(rxns & bench_gt_rxns)
    return {
        "path": str(pack),
        "files": file_reports,
        "target_overlap_count": len(target_overlap_values),
        "gt_reaction_overlap_count": len(rxn_overlap_values),
        "target_overlap_values": sorted(target_overlap_values),
        "gt_reaction_overlap_values": sorted(rxn_overlap_values),
        "blind_safe": not target_overlap_values and not rxn_overlap_values,
    }


def _pack_files(pack: Path) -> list[Path]:
    if pack.is_file():
        return [pack]
    out = []
    for name in [*VNEXT_FILES, *TRAINING_PACK_FILES]:
        path = pack / name
        if path.exists():
            out.append(path)
    if not out:
        out.extend(sorted(pack.glob("*.jsonl")))
    if not out:
        raise FileNotFoundError(f"no JSONL training files found in {pack}")
    return out


def _audit_jsonl(path: Path, bench_targets: set[str], bench_gt_rxns: set[str]) -> tuple[dict[str, Any], set[str], set[str]]:
    targets: set[str] = set()
    rxns: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            target = row.get("target_smiles") or row.get("product")
            if target:
                targets.add(str(target))
            _collect_reactions(row, rxns)
    target_overlap = targets & bench_targets
    rxn_overlap = rxns & bench_gt_rxns
    return {
        "path": str(path),
        "rows": rows,
        "unique_targets": len(targets),
        "unique_reactions": len(rxns),
        "target_overlap_count": len(target_overlap),
        "gt_reaction_overlap_count": len(rxn_overlap),
        "blind_safe": not target_overlap and not rxn_overlap,
    }, targets, rxns


def _collect_reactions(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key in ("rxn_smiles", "reaction_smiles"):
            rxn = value.get(key)
            if isinstance(rxn, str) and ">>" in rxn:
                out.add(rxn)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                _collect_reactions(nested, out)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _collect_reactions(item, out)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Overlap Audit",
        "",
        f"Benchmark: `{report['benchmark']['path']}`",
        "",
        "## Summary",
        "",
        f"- blind_safe: `{report['summary']['blind_safe']}`",
        f"- target_overlap_count: `{report['summary']['target_overlap_count']}`",
        f"- gt_reaction_overlap_count: `{report['summary']['gt_reaction_overlap_count']}`",
        "",
        "## Benchmark",
        "",
        f"- rows: `{report['benchmark']['n_rows']}`",
        f"- unique targets: `{report['benchmark']['unique_targets']}`",
        f"- duplicate targets: `{report['benchmark']['duplicate_targets']}`",
        f"- GT steps: `{report['benchmark']['gt_step_count']}`",
        f"- unique GT reactions: `{report['benchmark']['unique_gt_reactions']}`",
        "",
        "## Packs",
        "",
    ]
    for pack in report["packs"]:
        lines.extend([
            f"### `{pack['path']}`",
            "",
            f"- blind_safe: `{pack['blind_safe']}`",
            f"- target_overlap_count: `{pack['target_overlap_count']}`",
            f"- gt_reaction_overlap_count: `{pack['gt_reaction_overlap_count']}`",
            "",
            "| File | Rows | Unique Targets | Unique Reactions | Target Overlap | GT Reaction Overlap | Blind Safe |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for file_report in pack["files"]:
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | `{}` |".format(
                    file_report["path"],
                    file_report["rows"],
                    file_report["unique_targets"],
                    file_report["unique_reactions"],
                    file_report["target_overlap_count"],
                    file_report["gt_reaction_overlap_count"],
                    file_report["blind_safe"],
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit benchmark target/reaction overlap against training packs")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--pack", action="append", required=True, help="Training or vNext pack directory; can be repeated")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--fail-on-overlap", action="store_true")
    args = ap.parse_args()

    report = audit_benchmark_overlap(Path(args.benchmark), [Path(path) for path in args.pack])
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_markdown(report, Path(args.output_md))
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_overlap and not report["summary"]["blind_safe"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
