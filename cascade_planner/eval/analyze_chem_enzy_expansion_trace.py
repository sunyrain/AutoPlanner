"""Analyze ChemEnzy internal cascade expansion traces.

The trace is emitted by ``run_cascade_search_benchmark`` when
``--chem-enzy-expansion-trace-output`` is enabled. It captures candidates before
they are inserted into the MolStar tree, which is the right level for debugging
and training cascade-aware internal cost hooks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
    gt_reactants,
)


def analyze_expansion_trace(*, trace_path: Path, benchmark_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    benchmark_rows = _read_benchmark(benchmark_path)
    gt_by_target = {
        row["target_smiles"]: {
            "route_domain": row.get("route_domain"),
            "gt_rxns": {
                canonical_reaction(step.get("rxn_smiles") or "") or step.get("rxn_smiles")
                for step in row.get("gt_route") or []
                if step.get("rxn_smiles")
            },
            "gt_reactants": gt_reactants(row),
        }
        for row in benchmark_rows
    }
    rows = _read_jsonl(trace_path)
    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_route_domain: dict[str, Counter[str]] = defaultdict(Counter)
    adjustment_hit_values = []
    adjustment_miss_values = []
    examples = []

    for row in rows:
        target = str(row.get("target_smiles") or "")
        gt = gt_by_target.get(target)
        if not gt:
            continue
        source = str(row.get("source_model") or "unknown")
        domain = str(row.get("reaction_domain") or "unknown")
        route_domain = str(gt.get("route_domain") or "unknown")
        rxn = _candidate_reaction(row)
        exact_hit = bool(rxn and rxn in gt["gt_rxns"])
        reactant_hit = bool(_candidate_reactants(row) & gt["gt_reactants"])
        adjustment = _float_or_none(row.get("cascade_adjustment"))

        totals["candidate_rows"] += 1
        totals["exact_gt_hits"] += int(exact_hit)
        totals["gt_reactant_hits"] += int(reactant_hit)
        by_source[source]["candidate_rows"] += 1
        by_source[source]["exact_gt_hits"] += int(exact_hit)
        by_source[source]["gt_reactant_hits"] += int(reactant_hit)
        by_domain[domain]["candidate_rows"] += 1
        by_domain[domain]["exact_gt_hits"] += int(exact_hit)
        by_domain[domain]["gt_reactant_hits"] += int(reactant_hit)
        by_route_domain[route_domain]["candidate_rows"] += 1
        by_route_domain[route_domain]["exact_gt_hits"] += int(exact_hit)
        by_route_domain[route_domain]["gt_reactant_hits"] += int(reactant_hit)
        if adjustment is not None:
            if exact_hit or reactant_hit:
                adjustment_hit_values.append(adjustment)
            else:
                adjustment_miss_values.append(adjustment)
        if (exact_hit or reactant_hit) and len(examples) < 20:
            examples.append(
                {
                    "target_smiles": target,
                    "route_domain": route_domain,
                    "parent_mol": row.get("parent_mol"),
                    "candidate_index": row.get("candidate_index"),
                    "source_model": source,
                    "reaction_domain": domain,
                    "exact_gt_hit": exact_hit,
                    "gt_reactant_hit": reactant_hit,
                    "cascade_adjustment": adjustment,
                    "total_cost": _float_or_none(row.get("total_cost")),
                    "rxn_smiles": rxn,
                }
            )

    report = {
        "metadata": {
            "trace_path": str(trace_path),
            "benchmark_path": str(benchmark_path),
            "n_trace_rows": len(rows),
            "n_benchmark_targets": len(gt_by_target),
        },
        "summary": _rates(dict(totals)),
        "by_source": {key: _rates(dict(value)) for key, value in sorted(by_source.items())},
        "by_reaction_domain": {key: _rates(dict(value)) for key, value in sorted(by_domain.items())},
        "by_route_domain": {key: _rates(dict(value)) for key, value in sorted(by_route_domain.items())},
        "adjustment": {
            "mean_hit_adjustment": _mean(adjustment_hit_values),
            "mean_miss_adjustment": _mean(adjustment_miss_values),
            "n_hit_adjustments": len(adjustment_hit_values),
            "n_miss_adjustments": len(adjustment_miss_values),
        },
        "hit_examples": examples,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_markdown(report, output_path.with_suffix(".md"))
    return report


def _rates(counter: dict[str, int]) -> dict[str, Any]:
    rows = int(counter.get("candidate_rows") or 0)
    exact = int(counter.get("exact_gt_hits") or 0)
    reactant = int(counter.get("gt_reactant_hits") or 0)
    return {
        "candidate_rows": rows,
        "exact_gt_hits": exact,
        "gt_reactant_hits": reactant,
        "exact_gt_hit_rate": round(exact / rows, 6) if rows else 0.0,
        "gt_reactant_hit_rate": round(reactant / rows, 6) if rows else 0.0,
    }


def _candidate_reaction(row: dict[str, Any]) -> str:
    parent = str(row.get("parent_mol") or "")
    reactants = [str(item) for item in row.get("reactants") or [] if item]
    if not parent or not reactants:
        return ""
    rxn = ".".join(reactants) + ">>" + parent
    return canonical_reaction(rxn) or rxn


def _candidate_reactants(row: dict[str, Any]) -> set[str]:
    out = set()
    for item in row.get("reactants") or []:
        key = canonical_smiles(str(item)) or str(item)
        if key:
            out.add(key)
    return out


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict) and row.get("target_smiles")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# ChemEnzy Expansion Trace Analysis",
        "",
        "## Summary",
        "",
        _table(["metric", "value"], [[key, value] for key, value in report["summary"].items()]),
        "",
        "## By Source",
        "",
        _table_from_mapping(report["by_source"]),
        "",
        "## By Reaction Domain",
        "",
        _table_from_mapping(report["by_reaction_domain"]),
        "",
        "## By Route Domain",
        "",
        _table_from_mapping(report["by_route_domain"]),
        "",
        "## Adjustment",
        "",
        _table(["metric", "value"], [[key, value] for key, value in report["adjustment"].items()]),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_from_mapping(value: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, metrics in value.items():
        rows.append([
            name,
            metrics["candidate_rows"],
            metrics["exact_gt_hits"],
            metrics["exact_gt_hit_rate"],
            metrics["gt_reactant_hits"],
            metrics["gt_reactant_hit_rate"],
        ])
    return _table(
        ["name", "rows", "exact_hits", "exact_rate", "reactant_hits", "reactant_rate"],
        rows,
    )


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze ChemEnzy internal cascade expansion trace")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = analyze_expansion_trace(
        trace_path=Path(args.trace),
        benchmark_path=Path(args.benchmark),
        output_path=Path(args.output),
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
