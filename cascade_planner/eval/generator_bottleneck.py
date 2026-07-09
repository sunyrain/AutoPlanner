"""Diagnose generator and selector bottlenecks from live benchmark artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import recovery_bottleneck_labels


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def _bottleneck_labels(target: dict[str, Any]) -> list[str]:
    recovery = target.get("route_recovery") or {}
    metrics = target.get("metrics") or {}
    labels = []

    labels.extend(recovery.get("recovery_bottleneck_labels") or recovery_bottleneck_labels(recovery))
    if metrics.get("strict_stock_solve_any") is False:
        labels.append("stock_dead_end")
    if metrics.get("condition_window_success_any") is False:
        labels.append("condition_failure")
    if metrics.get("cascade_compatibility_success_any") is False:
        labels.append("compatibility_failure")
    if not labels:
        labels.append("no_primary_bottleneck_detected")
    return labels


def summarize_bottlenecks(data: dict[str, Any]) -> dict[str, Any]:
    targets = data.get("targets") or []
    overall = Counter()
    by_domain: dict[str, Counter] = defaultdict(Counter)
    source_counts = Counter()
    ranks = {
        "candidate_exact_reaction_best_candidate_rank": [],
        "candidate_gt_reactant_best_candidate_rank": [],
    }

    for target in targets:
        domain = target.get("route_domain") or "unknown"
        overall["n"] += 1
        by_domain[domain]["n"] += 1
        for label in _bottleneck_labels(target):
            overall[label] += 1
            by_domain[domain][label] += 1
        recovery = target.get("route_recovery") or {}
        for key in ranks:
            value = recovery.get(key)
            if isinstance(value, (int, float)):
                ranks[key].append(float(value))
        for route in (target.get("planner_output") or {}).get("routes") or []:
            source_counts.update((route.get("metrics") or {}).get("candidate_source_counts") or {})

    return {
        "n_targets": overall["n"],
        "overall": dict(overall),
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "candidate_source_counts": dict(source_counts),
        "rank_means": {
            k: (round(sum(v) / len(v), 3) if v else None)
            for k, v in ranks.items()
        },
    }


def write_markdown(summary: dict[str, Any], output_path: str, source_path: str) -> None:
    n = int(summary.get("n_targets") or 0)
    overall = Counter(summary.get("overall") or {})
    lines = [
        "# Generator Bottleneck Diagnosis",
        "",
        f"Source: `{source_path}`",
        f"Targets: `{n}`",
        "",
        "## Overall",
        "",
        "| Bottleneck | Count | Rate |",
        "|---|---:|---:|",
    ]
    for label, count in sorted(overall.items()):
        if label == "n":
            continue
        lines.append(f"| `{label}` | {count} | {_pct(int(count), n)} |")

    lines.extend([
        "",
        "## By Domain",
        "",
        "| Domain | n | Candidate reactant miss | Candidate reaction detail miss | Selector missed exact | Route composition/order miss | Selector missed GT reactant | Stock dead-end | Condition fail | Compatibility fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for domain, row in sorted((summary.get("by_domain") or {}).items()):
        d = int(row.get("n") or 0)
        lines.append(
            f"| `{domain}` | {d} | "
            f"{_pct(int(row.get('candidate_generator_reactant_miss') or 0), d)} | "
            f"{_pct(int(row.get('candidate_generator_reaction_detail_miss') or 0), d)} | "
            f"{_pct(int(row.get('selector_missed_exact_candidate') or 0), d)} | "
            f"{_pct(int(row.get('route_composition_or_order_miss') or 0), d)} | "
            f"{_pct(int(row.get('selector_missed_gt_reactant_candidate') or 0), d)} | "
            f"{_pct(int(row.get('stock_dead_end') or 0), d)} | "
            f"{_pct(int(row.get('condition_failure') or 0), d)} | "
            f"{_pct(int(row.get('compatibility_failure') or 0), d)} |"
        )

    lines.extend(["", "## Candidate Sources", "", "| Source | Count |", "|---|---:|"])
    for source, count in sorted((summary.get("candidate_source_counts") or {}).items()):
        lines.append(f"| `{source}` | {count} |")

    ranks = summary.get("rank_means") or {}
    lines.extend([
        "",
        "## Rank Means",
        "",
        f"- candidate exact reaction best candidate rank: `{ranks.get('candidate_exact_reaction_best_candidate_rank')}`",
        f"- candidate GT reactant best candidate rank: `{ranks.get('candidate_gt_reactant_best_candidate_rank')}`",
        "",
    ])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose generator bottlenecks from a live benchmark JSON")
    ap.add_argument("--input", default="results/v2/live_benchmark_beam_type_aligned_full.json")
    ap.add_argument("--output", default="results/v2/generator_bottleneck_diagnosis.md")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    summary = summarize_bottlenecks(data)
    write_markdown(summary, args.output, args.input)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
