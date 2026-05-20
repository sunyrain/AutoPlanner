"""Per-step candidate miss audit for live benchmark artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import RDLogger

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_side,
    canonical_smiles,
    candidate_reactants,
    reaction_reactants,
)

RDLogger.DisableLog("rdApp.*")


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def _rxn_product(rxn_smiles: str | None) -> str:
    if not rxn_smiles or ">>" not in rxn_smiles:
        return ""
    rhs = rxn_smiles.split(">>", 1)[1]
    return ".".join(canonical_side(rhs))


def _candidate_rxn(candidate: dict[str, Any]) -> str:
    return candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or ""


def _candidate_type(candidate: dict[str, Any]) -> str:
    return candidate.get("reaction_type") or candidate.get("type") or ""


def _ec1(ec: str | None) -> str:
    if not ec:
        return ""
    first = str(ec).split(".", 1)[0]
    return first if first.isdigit() else ""


def _slot_reactants(step: dict[str, Any]) -> set[str]:
    out = set()
    if step.get("main_reactant"):
        out.add(canonical_smiles(step.get("main_reactant")))
    for smi in step.get("aux_reactants") or []:
        out.add(canonical_smiles(smi))
    out.update(reaction_reactants(step.get("reaction_smiles")))
    return {x for x in out if x}


def _candidate_pool(step: dict[str, Any]) -> list[dict[str, Any]]:
    pool = step.get("candidate_pool") or {}
    rows = pool.get("top_candidates") or []
    return rows if isinstance(rows, list) else []


def _load_trace_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            rows.extend(_coerce_trace_event(row) for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows.append(_coerce_trace_event(value))
    return [row for row in rows if row]


def _coerce_trace_event(row: dict[str, Any]) -> dict[str, Any]:
    event = row.get("event")
    if isinstance(event, dict):
        out = dict(event)
        out["_target_smiles"] = row.get("target_smiles")
        out["_benchmark_index"] = row.get("benchmark_index")
        out["_route_domain"] = row.get("route_domain")
        out["_same_run_benchmark_trace"] = bool(row.get("same_run_benchmark_trace"))
        return out
    if event is None and "event" in row:
        return {}
    return row


def _trace_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target_product: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    all_source_counts: Counter[str] = Counter()
    for event in rows:
        leaf = canonical_smiles(event.get("expanded_leaf")) or str(event.get("expanded_leaf") or "")
        if leaf:
            by_product[leaf].append(event)
            benchmark_index = event.get("_benchmark_index")
            if benchmark_index is not None:
                try:
                    by_target_product[(int(benchmark_index), leaf)].append(event)
                except (TypeError, ValueError):
                    pass
        for action in event.get("candidate_actions") or []:
            source = str(action.get("source") or "unknown")
            all_source_counts[source] += 1
    return {
        "rows": rows,
        "by_product": by_product,
        "by_target_product": by_target_product,
        "source_counts": dict(all_source_counts),
    }


def _gt_steps(target: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, step in enumerate(target.get("gt_route") or [], 1):
        rxn = step.get("rxn_smiles") or ""
        canon_rxn = canonical_reaction(rxn)
        ec = step.get("ec_number") or ""
        rows.append({
            "gt_step_index": idx,
            "gt_rxn": canon_rxn,
            "gt_product": _rxn_product(canon_rxn),
            "gt_reactants": sorted(reaction_reactants(canon_rxn)),
            "gt_reactant_set": set(reaction_reactants(canon_rxn)),
            "gt_type": step.get("transformation") or "",
            "gt_ec": ec,
            "gt_ec1": _ec1(ec),
            "step_kind": "enzymatic" if ec else "chemical",
        })
    return rows


def _route_steps(target: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for route_rank, route in enumerate((target.get("planner_output") or {}).get("routes") or [], 1):
        for step in route.get("steps") or []:
            product = canonical_smiles(step.get("product")) or _rxn_product(step.get("reaction_smiles"))
            candidates = []
            for cand_rank, cand in enumerate(_candidate_pool(step), 1):
                cand_rxn = canonical_reaction(_candidate_rxn(cand))
                candidates.append({
                    "candidate_rank": cand_rank,
                    "candidate": cand,
                    "rxn": cand_rxn,
                    "product": _rxn_product(cand_rxn) or product,
                    "reactants": candidate_reactants(cand),
                    "source": cand.get("source") or "unknown",
                    "type": _candidate_type(cand),
                    "ec1": _ec1(cand.get("ec")),
                })
            rows.append({
                "route_rank": route_rank,
                "slot_index": step.get("index"),
                "product": product,
                "selected_rxn": canonical_reaction(step.get("reaction_smiles")),
                "selected_reactants": _slot_reactants(step),
                "selected_type": step.get("reaction_type") or "",
                "selected_ec1": _ec1(step.get("ec")),
                "source": step.get("source") or "",
                "candidates": candidates,
            })
    return rows


def _audit_gt_step(
    target: dict[str, Any],
    gt: dict[str, Any],
    route_steps: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    product_steps = [s for s in route_steps if s["product"] == gt["gt_product"]]
    candidate_rows = [c for s in product_steps for c in s["candidates"]]
    # Fallback for artifacts where product canonicalization is incomplete.
    if not candidate_rows:
        candidate_rows = [c for s in route_steps for c in s["candidates"] if c["product"] == gt["gt_product"]]

    exact_candidate_hits = [c for c in candidate_rows if c["rxn"] == gt["gt_rxn"]]
    exact_reactant_hits = [
        c for c in candidate_rows
        if c["reactants"] == gt["gt_reactant_set"] and gt["gt_reactant_set"]
    ]
    any_reactant_hits = [
        c for c in candidate_rows
        if c["reactants"] & gt["gt_reactant_set"]
    ]
    selected_exact_hits = [s for s in product_steps if s["selected_rxn"] == gt["gt_rxn"]]
    selected_reactant_hits = [
        s for s in product_steps
        if s["selected_reactants"] == gt["gt_reactant_set"] and gt["gt_reactant_set"]
    ]

    pool_ec1_match = True
    if gt["gt_ec1"]:
        pool_ec1_match = any(c["ec1"] == gt["gt_ec1"] for c in candidate_rows)
    type_candidates = [c for c in candidate_rows if c["type"]]
    pool_type_match = True
    if gt["gt_type"] and type_candidates:
        pool_type_match = any(c["type"] == gt["gt_type"] for c in type_candidates)
    selected_type_match = True
    selected_types = [s["selected_type"] for s in product_steps if s["selected_type"]]
    if gt["gt_type"] and selected_types:
        selected_type_match = any(t == gt["gt_type"] for t in selected_types)

    labels = []
    if not product_steps:
        labels.append("product_slot_missing")
    if not exact_candidate_hits:
        labels.append("generator_exact_miss")
    if exact_reactant_hits and not exact_candidate_hits:
        labels.append("reactant_set_present_exact_missing")
    elif any_reactant_hits and not exact_candidate_hits:
        labels.append("reactant_fragment_present_exact_missing")
    if exact_candidate_hits and not selected_exact_hits:
        labels.append("selector_missed_exact_candidate")
    if gt["gt_ec1"] and not pool_ec1_match:
        labels.append("ec1_miss")
    if gt["gt_type"] and not pool_type_match:
        labels.append("candidate_type_miss")
    if gt["gt_type"] and not selected_type_match:
        labels.append("selected_type_miss")

    metrics = target.get("metrics") or {}
    if metrics.get("strict_stock_solve_any") is False:
        labels.append("target_stock_dead_end")
    if metrics.get("condition_window_success_any") is False:
        labels.append("target_condition_failure")
    if metrics.get("cascade_compatibility_success_any") is False:
        labels.append("target_compatibility_failure")
    if not labels:
        labels.append("no_step_bottleneck_detected")
    bottleneck = _coverage_bottleneck(
        target,
        gt,
        product_steps,
        candidate_rows,
        selected_exact_hits,
        labels,
        trace or {},
    )
    if bottleneck not in labels:
        labels.insert(0, bottleneck)

    source_counts = Counter(c["source"] for c in candidate_rows)
    exact_sources = Counter(c["source"] for c in exact_candidate_hits)
    reactant_sources = Counter(c["source"] for c in exact_reactant_hits or any_reactant_hits)
    best_exact_rank = min((c["candidate_rank"] for c in exact_candidate_hits), default=None)
    best_reactant_rank = min((c["candidate_rank"] for c in exact_reactant_hits or any_reactant_hits), default=None)

    return {
        "target_index": target.get("index"),
        "target_smiles": target.get("target_smiles"),
        "doi": target.get("doi"),
        "cascade_id": target.get("cascade_id"),
        "route_domain": target.get("route_domain") or "unknown",
        "gt_step_index": gt["gt_step_index"],
        "step_kind": gt["step_kind"],
        "gt_type": gt["gt_type"],
        "gt_ec1": gt["gt_ec1"],
        "gt_rxn": gt["gt_rxn"],
        "gt_product": gt["gt_product"],
        "n_product_slots": len(product_steps),
        "n_candidates": len(candidate_rows),
        "product_slot_seen": bool(product_steps),
        "candidate_exact_hit": bool(exact_candidate_hits),
        "candidate_exact_reactant_set_hit": bool(exact_reactant_hits),
        "candidate_any_gt_reactant_hit": bool(any_reactant_hits),
        "selected_exact_hit": bool(selected_exact_hits),
        "selected_exact_reactant_set_hit": bool(selected_reactant_hits),
        "pool_ec1_match": pool_ec1_match,
        "pool_type_match": pool_type_match,
        "selected_type_match": selected_type_match,
        "best_exact_candidate_rank": best_exact_rank,
        "best_reactant_candidate_rank": best_reactant_rank,
        "candidate_source_counts": dict(source_counts),
        "exact_candidate_sources": dict(exact_sources),
        "reactant_hit_sources": dict(reactant_sources),
        "labels": labels,
        "coverage_bottleneck": bottleneck,
    }


def audit_candidate_misses(data: dict[str, Any], trace_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    trace = _trace_index(trace_rows or [])
    rows = []
    for target in data.get("targets") or []:
        route_steps = _route_steps(target)
        for gt in _gt_steps(target):
            rows.append(_audit_gt_step(target, gt, route_steps, trace))
    return summarize_rows(rows)


def _coverage_bottleneck(
    target: dict[str, Any],
    gt: dict[str, Any],
    product_steps: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    selected_exact_hits: list[dict[str, Any]],
    labels: list[str],
    trace: dict[str, Any],
) -> str:
    gt_product = gt.get("gt_product") or ""
    trace_events = []
    target_index = target.get("index")
    if target_index is not None:
        try:
            trace_events = (trace.get("by_target_product") or {}).get((int(target_index), gt_product)) or []
        except (TypeError, ValueError):
            trace_events = []
    if not trace_events:
        trace_events = (trace.get("by_product") or {}).get(gt_product) or []
    if not product_steps and not trace_events:
        return "intermediate_not_reached"
    exact_hit = "generator_exact_miss" not in labels
    if exact_hit and not selected_exact_hits:
        return "selector_missed_candidate"
    if exact_hit:
        return "candidate_recovered"
    if not trace_events:
        return "candidate_missing"

    queried_sources: set[str] = set()
    raw_returned = 0
    final_returned = 0
    ranker_dropped = 0
    invalid_dropped = 0
    dedupe_dropped = 0
    zero_budget_sources: set[str] = set()
    available_sources: set[str] = set()
    low_budget = False
    for event in trace_events:
        for diag in event.get("proposal_diagnostics") or []:
            for source, row in (diag.get("sources") or {}).items():
                available_sources.add(str(source))
                budget = int(row.get("allocated_budget") or 0)
                if budget <= 0:
                    zero_budget_sources.add(str(source))
                if bool(row.get("queried")) or int(row.get("calls") or 0) > 0:
                    queried_sources.add(str(source))
                raw_returned += int(row.get("raw_returned") or 0)
                final_returned += int(row.get("final_returned") or row.get("kept_returned") or 0)
                ranker_dropped += int(row.get("ranker_dropped") or 0)
                invalid_dropped += int(row.get("invalid_dropped") or 0)
                dedupe_dropped += int(row.get("dedupe_dropped") or 0)
                if 0 < budget <= 2:
                    low_budget = True

    if available_sources and not queried_sources:
        return "source_not_queried"
    if low_budget and raw_returned <= final_returned:
        return "queried_budget_too_small"
    if raw_returned > 0 and invalid_dropped > 0 and final_returned <= 0:
        return "generated_invalid_filtered"
    if raw_returned > 0 and (ranker_dropped > 0 or dedupe_dropped > 0):
        return "generated_ranked_out"
    if raw_returned <= 0:
        return "candidate_missing"
    if candidate_rows:
        return "generated_but_wrong_candidate"
    return "candidate_missing"


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    labels = Counter(label for row in rows for label in row["labels"])
    bottlenecks = Counter(row.get("coverage_bottleneck") or "unknown" for row in rows)
    source_counts = Counter()
    exact_sources = Counter()
    reactant_sources = Counter()
    for row in rows:
        source_counts.update(row["candidate_source_counts"])
        exact_sources.update(row["exact_candidate_sources"])
        reactant_sources.update(row["reactant_hit_sources"])
    return {
        "n_steps": n,
        "product_slot_seen": sum(row["product_slot_seen"] for row in rows),
        "candidate_exact_hit": sum(row["candidate_exact_hit"] for row in rows),
        "candidate_exact_reactant_set_hit": sum(row["candidate_exact_reactant_set_hit"] for row in rows),
        "candidate_any_gt_reactant_hit": sum(row["candidate_any_gt_reactant_hit"] for row in rows),
        "selected_exact_hit": sum(row["selected_exact_hit"] for row in rows),
        "selected_exact_reactant_set_hit": sum(row["selected_exact_reactant_set_hit"] for row in rows),
        "pool_ec1_match": sum(row["pool_ec1_match"] for row in rows),
        "pool_type_match": sum(row["pool_type_match"] for row in rows),
        "selected_type_match": sum(row["selected_type_match"] for row in rows),
        "avg_candidates_per_step": round(
            sum(int(row["n_candidates"]) for row in rows) / max(n, 1),
            3,
        ),
        "label_counts": dict(labels),
        "coverage_bottleneck_counts": dict(bottlenecks),
        "candidate_source_counts": dict(source_counts),
        "exact_candidate_sources": dict(exact_sources),
        "reactant_hit_sources": dict(reactant_sources),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = defaultdict(list)
    by_kind = defaultdict(list)
    by_ec1 = defaultdict(list)
    by_type = defaultdict(list)
    for row in rows:
        by_domain[row["route_domain"]].append(row)
        by_kind[row["step_kind"]].append(row)
        by_ec1[row["gt_ec1"] or "chemical"].append(row)
        by_type[row["gt_type"] or "unknown"].append(row)

    miss_examples = [
        row for row in rows
        if "generator_exact_miss" in row["labels"]
    ][:25]
    return {
        "overall": _summarize_subset(rows),
        "by_domain": {k: _summarize_subset(v) for k, v in sorted(by_domain.items())},
        "by_step_kind": {k: _summarize_subset(v) for k, v in sorted(by_kind.items())},
        "by_ec1": {k: _summarize_subset(v) for k, v in sorted(by_ec1.items())},
        "by_transformation": {k: _summarize_subset(v) for k, v in sorted(by_type.items())},
        "step_rows": rows,
        "miss_examples": miss_examples,
    }


def _rate(summary: dict[str, Any], key: str) -> str:
    return _pct(int(summary.get(key) or 0), int(summary.get("n_steps") or 0))


def _label_rate(summary: dict[str, Any], label: str) -> str:
    labels = summary.get("label_counts") or {}
    return _pct(int(labels.get(label) or 0), int(summary.get("n_steps") or 0))


def _write_group_table(lines: list[str], title: str, rows: dict[str, dict[str, Any]]) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Group | Steps | Product slot | Exact cand | Reactant set cand | Any reactant cand | Selected exact | Generator miss | Reactant-only miss | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, row in rows.items():
        labels = row.get("label_counts") or {}
        reactant_only = (
            int(labels.get("reactant_set_present_exact_missing") or 0)
            + int(labels.get("reactant_fragment_present_exact_missing") or 0)
        )
        lines.append(
            f"| `{key}` | {row.get('n_steps', 0)} | "
            f"{_rate(row, 'product_slot_seen')} | "
            f"{_rate(row, 'candidate_exact_hit')} | "
            f"{_rate(row, 'candidate_exact_reactant_set_hit')} | "
            f"{_rate(row, 'candidate_any_gt_reactant_hit')} | "
            f"{_rate(row, 'selected_exact_hit')} | "
            f"{_label_rate(row, 'generator_exact_miss')} | "
            f"{_pct(reactant_only, int(row.get('n_steps') or 0))} | "
            f"{row.get('avg_candidates_per_step')} |"
        )


def write_markdown(summary: dict[str, Any], output_path: str, source_path: str) -> None:
    overall = summary["overall"]
    labels = overall.get("label_counts") or {}
    reactant_only = (
        int(labels.get("reactant_set_present_exact_missing") or 0)
        + int(labels.get("reactant_fragment_present_exact_missing") or 0)
    )
    lines = [
        "# Candidate Miss Audit",
        "",
        f"Source: `{source_path}`",
        "",
        "## Overall",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| GT steps | {overall['n_steps']} | 100.0% |",
        f"| Product slot seen | {overall['product_slot_seen']} | {_rate(overall, 'product_slot_seen')} |",
        f"| Exact candidate hit | {overall['candidate_exact_hit']} | {_rate(overall, 'candidate_exact_hit')} |",
        f"| Exact reactant-set candidate hit | {overall['candidate_exact_reactant_set_hit']} | {_rate(overall, 'candidate_exact_reactant_set_hit')} |",
        f"| Any GT-reactant candidate hit | {overall['candidate_any_gt_reactant_hit']} | {_rate(overall, 'candidate_any_gt_reactant_hit')} |",
        f"| Selected exact hit | {overall['selected_exact_hit']} | {_rate(overall, 'selected_exact_hit')} |",
        f"| Generator exact miss | {labels.get('generator_exact_miss', 0)} | {_label_rate(overall, 'generator_exact_miss')} |",
        f"| Reactant-only exact miss | {reactant_only} | {_pct(reactant_only, overall['n_steps'])} |",
        f"| Selector missed exact candidate | {labels.get('selector_missed_exact_candidate', 0)} | {_label_rate(overall, 'selector_missed_exact_candidate')} |",
        f"| Avg candidates per GT step | {overall['avg_candidates_per_step']} | n/a |",
    ]

    lines.extend([
        "",
        "## Coverage Bottlenecks",
        "",
        "| Bottleneck | Count | Rate |",
        "|---|---:|---:|",
    ])
    for label, count in sorted((overall.get("coverage_bottleneck_counts") or {}).items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| `{label}` | {count} | {_pct(int(count), int(overall.get('n_steps') or 0))} |")

    _write_group_table(lines, "By Step Kind", summary.get("by_step_kind") or {})
    _write_group_table(lines, "By Domain", summary.get("by_domain") or {})
    _write_group_table(lines, "By EC1", summary.get("by_ec1") or {})

    lines.extend([
        "",
        "## Candidate Sources",
        "",
        "| Source | All candidates | Exact hits | Reactant hits |",
        "|---|---:|---:|---:|",
    ])
    sources = set((overall.get("candidate_source_counts") or {}).keys())
    sources.update((overall.get("exact_candidate_sources") or {}).keys())
    sources.update((overall.get("reactant_hit_sources") or {}).keys())
    for source in sorted(sources):
        lines.append(
            f"| `{source}` | "
            f"{(overall.get('candidate_source_counts') or {}).get(source, 0)} | "
            f"{(overall.get('exact_candidate_sources') or {}).get(source, 0)} | "
            f"{(overall.get('reactant_hit_sources') or {}).get(source, 0)} |"
        )

    lines.extend([
        "",
        "## First Generator Miss Examples",
        "",
        "| Target | Domain | Step | Kind | Type | EC1 | Product slot | Candidates | Labels |",
        "|---|---|---:|---|---|---|---:|---:|---|",
    ])
    for row in summary.get("miss_examples") or []:
        target = row.get("target_smiles") or ""
        if len(target) > 32:
            target = target[:29] + "..."
        lines.append(
            f"| `{target}` | `{row.get('route_domain')}` | {row.get('gt_step_index')} | "
            f"`{row.get('step_kind')}` | `{row.get('gt_type')}` | "
            f"`{row.get('gt_ec1') or '-'}` | {int(bool(row.get('product_slot_seen')))} | "
            f"{row.get('n_candidates')} | `{row.get('coverage_bottleneck')}: {', '.join(row.get('labels') or [])}` |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit per-step candidate misses in live benchmark artifacts")
    ap.add_argument("--input", default="results/v2/live_benchmark_beam_type_aligned_full.json")
    ap.add_argument("--output", default="results/v2/candidate_miss_audit.md")
    ap.add_argument(
        "--json-output",
        default=None,
        help="Summary JSON path (default: output path with .json suffix)",
    )
    ap.add_argument(
        "--trace",
        default=None,
        help="Optional route_tree trace JSONL to classify source/budget/filter bottlenecks.",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = audit_candidate_misses(data, trace_rows=_load_trace_rows(args.trace))
    write_markdown(summary, args.output, args.input)
    json_output = args.json_output or str(Path(args.output).with_suffix(".json"))
    if json_output:
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(json_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "json_output": json_output,
        "overall": summary["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
