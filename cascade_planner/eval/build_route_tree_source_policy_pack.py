"""Build source/budget supervision rows from route-tree traces."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.route_tree.source_gate import (
    SOURCE_GROUPS,
    SOURCE_POLICY_BUDGET_LABELS,
    SOURCE_POLICY_DECISIONS,
    source_policy_group,
)
from cascade_planner.vnext.features import read_jsonl, write_jsonl


SOURCE_POLICY_PACK_SCHEMA_VERSION = "route_tree_source_policy_pack.v1"
FAILURE_LABELS = (
    "source_not_queried",
    "queried_budget_too_small",
    "provider_missing",
    "invalid_filtered",
    "selector_missed_candidate",
    "generated_ranked_out",
    "stock_dead_end",
)


def build_route_tree_source_policy_pack(
    *,
    trace_paths: Iterable[Path],
    output_dir: Path,
    eval_only: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in trace_paths]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for trace_row in read_jsonl(path):
            rows.extend(source_policy_rows_from_trace(trace_row, source_path=path, eval_only=eval_only))
            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break
        if max_rows is not None and len(rows) >= max_rows:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "source_policy_pack.jsonl"
    manifest_path = output_dir / "source_policy_manifest.json"
    report_path = output_dir / "source_policy_report.md"
    write_jsonl(pack_path, rows)
    manifest = {
        "schema_version": SOURCE_POLICY_PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_paths": [str(path) for path in paths],
        "eval_only": bool(eval_only),
        "files": {
            "source_policy_pack": str(pack_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
        "counts": _pack_counts(rows),
        "source_groups": list(SOURCE_GROUPS),
        "budget_labels": list(SOURCE_POLICY_BUDGET_LABELS),
        "decision_labels": list(SOURCE_POLICY_DECISIONS),
        "failure_labels": list(FAILURE_LABELS),
        "row_contract": [
            "state_id",
            "target_id",
            "depth",
            "leaf",
            "source_group",
            "allocated_budget",
            "source_called",
            "raw_returned",
            "kept_returned",
            "final_returned",
            "latency_ms",
            "useful_candidate_hit",
            "stock_closing_hit",
            "failure_labels",
            "leaf_utility",
            "source_utility",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def source_policy_rows_from_trace(
    trace_row: dict[str, Any],
    *,
    source_path: Path | None = None,
    eval_only: bool = False,
) -> list[dict[str, Any]]:
    event = trace_row.get("event") if "event" in trace_row else trace_row
    if not isinstance(event, dict):
        return []
    proposal_rows = list(event.get("proposal_diagnostics") or [])
    if not proposal_rows:
        return []
    selected_action = _selected_action(event)
    selected_source = str((selected_action or {}).get("source") or "")
    target = str(trace_row.get("target_smiles") or (event.get("state") or {}).get("target") or "")
    target_id = _target_id(trace_row, target)
    state_id = str(event.get("state_id") or (event.get("state") or {}).get("state_id") or "")
    out: list[dict[str, Any]] = []
    for proposal in proposal_rows:
        leaf = str(proposal.get("leaf") or event.get("expanded_leaf") or "")
        sources = proposal.get("sources") or {}
        allocation = proposal.get("allocation") or {}
        source_budgets = allocation.get("source_budgets") or {}
        source_names = list(dict.fromkeys([
            *[str(source) for source in proposal.get("ordered_sources") or []],
            *[str(source) for source in source_budgets],
            *[str(source) for source in sources],
        ]))
        if not source_names:
            continue
        for source in source_names:
            diagnostics = sources.get(source) or {}
            allocated = int(diagnostics.get("allocated_budget") or source_budgets.get(source) or 0)
            useful = bool(selected_source and selected_source == source and _canonical_leaf(leaf) == _canonical_leaf(event.get("expanded_leaf")))
            stock_closing = bool(useful and event.get("selected_next_stock_closed"))
            failure_labels = _failure_labels(
                diagnostics=diagnostics,
                allocated_budget=allocated,
                useful_candidate_hit=useful,
                stock_closing_hit=stock_closing,
                event=event,
                proposal=proposal,
            )
            row = {
                "schema_version": SOURCE_POLICY_PACK_SCHEMA_VERSION,
                "eval_only": bool(eval_only),
                "source_trace_path": str(source_path) if source_path else "",
                "state_id": state_id,
                "target_id": target_id,
                "target_smiles": target,
                "benchmark_index": trace_row.get("benchmark_index"),
                "depth": int(event.get("depth") or 0),
                "leaf": leaf,
                "source": source,
                "source_name": source,
                "source_group": source_policy_group(source),
                "proposal_budget": int(proposal.get("proposal_budget") or proposal.get("top_k") or 0),
                "allocated_budget": allocated,
                "requested_k": int(diagnostics.get("requested_k_total") or 0),
                "source_called": bool(diagnostics.get("queried")),
                "raw_returned": int(diagnostics.get("raw_returned") or 0),
                "kept_returned": int(diagnostics.get("kept_returned") or diagnostics.get("ranker_kept") or 0),
                "final_returned": int(diagnostics.get("final_returned") or 0),
                "latency_ms": float(diagnostics.get("latency_ms_total") or 0.0),
                "skip_reason": diagnostics.get("skip_reason") or "",
                "useful_candidate_hit": useful,
                "stock_closing_hit": stock_closing,
                "failure_labels": failure_labels,
                "leaf_utility": _leaf_utility(trace_row, event, proposal, leaf),
                "source_utility": _source_utility(useful, stock_closing, diagnostics, failure_labels),
                "budget_multiplier": float(allocation.get("budget_multiplier") or _budget_multiplier_from_budget(allocated, proposal)),
                "budget_multiplier_label": allocation.get("budget_multiplier_label") or _budget_label_from_multiplier(_budget_multiplier_from_budget(allocated, proposal)),
                "decision": allocation.get("decision") or "query",
                "policy_confidence": float(allocation.get("policy_confidence") or 0.0),
                "policy_reason": allocation.get("policy_reason") or "",
                "reaction_type": _event_context_value(event, "reaction_type"),
                "ec1": _event_context_value(event, "ec1"),
                "T": _event_context_value(event, "T"),
                "pH": _event_context_value(event, "pH"),
                "route_metadata": _route_metadata(trace_row, event, proposal, leaf),
                "source_diagnostics": dict(diagnostics),
            }
            out.append(row)
    return out


def _selected_action(event: dict[str, Any]) -> dict[str, Any] | None:
    selected_key = str(event.get("selected_action_key") or "")
    actions = event.get("candidate_actions") or []
    if not selected_key:
        return None
    for action in actions:
        key = str((action or {}).get("canonical_key") or "")
        if not key:
            key = _candidate_action_key(action or {})
        raw_key = str((action or {}).get("rxn_smiles") or (action or {}).get("reaction_smiles") or "")
        if key and (key == selected_key or raw_key == selected_key):
            return action or {}
    return None


def _candidate_action_key(action: dict[str, Any]) -> str:
    rxn = str(action.get("rxn_smiles") or action.get("reaction_smiles") or "")
    if rxn:
        return canonical_reaction(rxn) or rxn
    reactants = [canonical_smiles(smi) or smi for smi in action.get("reactants") or [] if smi]
    product = canonical_smiles(action.get("product")) or str(action.get("product") or "")
    return f"{'.'.join(sorted(reactants))}>>{product}" if product else ""


def _failure_labels(
    *,
    diagnostics: dict[str, Any],
    allocated_budget: int,
    useful_candidate_hit: bool,
    stock_closing_hit: bool,
    event: dict[str, Any],
    proposal: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    called = bool(diagnostics.get("queried"))
    raw = int(diagnostics.get("raw_returned") or 0)
    final = int(diagnostics.get("final_returned") or 0)
    invalid = int(diagnostics.get("invalid_dropped") or 0) + int(proposal.get("invalid_filtered") or 0)
    skip = str(diagnostics.get("skip_reason") or "")
    if not called:
        labels.append("provider_missing" if skip == "missing_engine" else "source_not_queried")
    if called and raw <= 0 and int(allocated_budget or 0) <= 2:
        labels.append("queried_budget_too_small")
    if raw > 0 and final <= 0 and invalid > 0:
        labels.append("invalid_filtered")
    if final > 0 and not useful_candidate_hit:
        labels.append("selector_missed_candidate")
    if raw > 0 and final <= 0 and not invalid:
        labels.append("generated_ranked_out")
    if _stock_dead_end(event) and not stock_closing_hit:
        labels.append("stock_dead_end")
    return sorted(set(labels))


def _leaf_utility(trace_row: dict[str, Any], event: dict[str, Any], proposal: dict[str, Any], leaf: str) -> float:
    metrics = trace_row.get("route_metrics") or []
    solved = bool((event.get("outcome") or {}).get("solved_routes")) or str((event.get("outcome") or {}).get("search_status") or "") == "solved"
    stock_closed = bool(event.get("selected_next_stock_closed")) or any(bool((item or {}).get("strict_stock_solve")) for item in metrics)
    selected_leaf = _canonical_leaf(leaf) == _canonical_leaf(event.get("expanded_leaf"))
    raw = int(proposal.get("raw_actions") or 0)
    final = int(proposal.get("final_actions") or 0)
    low_yield = bool(raw > 0 and final <= 1)
    utility = 0.05
    utility += 0.20 * float(final > 0)
    utility += 0.10 * min(final / 4.0, 1.0)
    utility += 0.12 * float(selected_leaf and bool(event.get("selected_action_key")))
    utility += 0.12 * float(solved)
    utility += 0.16 * float(stock_closed)
    utility += 0.08 * float(bool(event.get("expanded_leaf_parent_adjacent")) and selected_leaf)
    utility += 0.08 * float(bool(event.get("expanded_leaf_stock_hit")) and selected_leaf)
    if low_yield:
        utility -= 0.12
    if raw <= 0 and final <= 0:
        utility -= 0.10
    return max(0.0, min(1.0, utility))


def _source_utility(
    useful_candidate_hit: bool,
    stock_closing_hit: bool,
    diagnostics: dict[str, Any],
    failure_labels: list[str],
) -> float:
    raw = int(diagnostics.get("raw_returned") or 0)
    final = int(diagnostics.get("final_returned") or 0)
    utility = 0.05
    utility += 0.55 * float(useful_candidate_hit)
    utility += 0.20 * float(stock_closing_hit)
    utility += 0.10 * float(final > 0)
    utility += 0.05 * float(raw > 0)
    penalty = {
        "source_not_queried": 0.08,
        "queried_budget_too_small": 0.08,
        "provider_missing": 0.20,
        "invalid_filtered": 0.12,
        "generated_ranked_out": 0.10,
        "stock_dead_end": 0.12,
    }
    for label in failure_labels:
        utility -= penalty.get(label, 0.0)
    return max(0.0, min(1.0, utility))


def _route_metadata(trace_row: dict[str, Any], event: dict[str, Any], proposal: dict[str, Any], leaf: str) -> dict[str, Any]:
    state = event.get("state") or {}
    open_leaves = [str(smi) for smi in event.get("open_leaves") or state.get("open_leaves") or [] if smi]
    target = str(trace_row.get("target_smiles") or state.get("target") or "")
    raw = int(proposal.get("raw_actions") or 0)
    final = int(proposal.get("final_actions") or 0)
    leaf_key = _canonical_leaf(leaf)
    parent_reactants = _parent_reactants(state)
    return {
        "state_id": str(event.get("state_id") or state.get("state_id") or ""),
        "remaining_depth": max(0, int(trace_row.get("max_depth") or 6) - int(event.get("depth") or 0)),
        "open_leaf_count": len(open_leaves),
        "nonstock_leaf_count": sum(1 for smi in open_leaves if not _small_stock_like(smi)),
        "leaf_stock_hit": bool(event.get("expanded_leaf_stock_hit")) if leaf_key == _canonical_leaf(event.get("expanded_leaf")) else _small_stock_like(leaf),
        "leaf_parent_adjacent": bool(event.get("expanded_leaf_parent_adjacent")) if leaf_key == _canonical_leaf(event.get("expanded_leaf")) else leaf_key in parent_reactants,
        "leaf_low_yield": bool(raw > 0 and final <= 1),
        "leaf_heavy_atoms": _heavy_atoms(leaf),
        "target_heavy_atoms": _heavy_atoms(target),
        "state_depth": int(event.get("depth") or 0),
    }


def _event_context_value(event: dict[str, Any], field: str) -> Any:
    actions = event.get("candidate_actions") or []
    for action in actions:
        context = ((action or {}).get("metadata") or {}).get("route_tree_context") or {}
        if field in context and context.get(field) not in (None, ""):
            return context.get(field)
    return "" if field in {"reaction_type", "ec1"} else None


def _stock_dead_end(event: dict[str, Any]) -> bool:
    outcome = event.get("outcome") or {}
    return bool(
        not event.get("selected_next_stock_closed")
        and (
            bool(event.get("expanded_leaf_low_yield"))
            or int(outcome.get("dead_ends") or 0) > 0
            or (not event.get("selected_action_key") and str(outcome.get("search_status") or "") == "failed")
        )
    )


def _budget_multiplier_from_budget(allocated: int, proposal: dict[str, Any]) -> float:
    total = max(1, int(proposal.get("proposal_budget") or proposal.get("top_k") or 1))
    ordered_sources = proposal.get("ordered_sources") or []
    expected = max(1.0, total / max(len(ordered_sources), 1))
    ratio = max(0.5, min(3.0, float(allocated or 0) / expected))
    return min((0.5, 1.0, 2.0, 3.0), key=lambda item: abs(item - ratio))


def _budget_label_from_multiplier(multiplier: float) -> str:
    value = min((0.5, 1.0, 2.0, 3.0), key=lambda item: abs(item - float(multiplier or 1.0)))
    return {0.5: "0.5x", 1.0: "1x", 2.0: "2x", 3.0: "3x"}[value]


def _target_id(trace_row: dict[str, Any], target: str) -> str:
    if trace_row.get("benchmark_index") is not None:
        return f"idx:{trace_row.get('benchmark_index')}"
    can = canonical_smiles(target) or target
    return f"target:{can}"


def _canonical_leaf(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _parent_reactants(state: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in state.get("steps") or []:
        action = (step or {}).get("action") or {}
        for smi in action.get("reactants") or []:
            can = canonical_smiles(smi) or smi
            if can:
                out.add(can)
    return out


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _small_stock_like(smiles: str | None) -> bool:
    return _heavy_atoms(smiles) <= 6


def _pack_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "states": len({row.get("state_id") for row in rows}),
        "targets": len({row.get("target_id") for row in rows}),
        "source_groups": dict(Counter(row.get("source_group") or "unknown" for row in rows)),
        "failure_labels": dict(Counter(label for row in rows for label in row.get("failure_labels") or [])),
        "useful_candidate_hit": sum(int(bool(row.get("useful_candidate_hit"))) for row in rows),
        "stock_closing_hit": sum(int(bool(row.get("stock_closing_hit"))) for row in rows),
    }


def _report_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    lines = [
        "# Route Tree Source Policy Pack",
        "",
        f"Rows: `{counts.get('rows', 0)}`",
        f"Targets: `{counts.get('targets', 0)}`",
        f"Eval only: `{manifest.get('eval_only')}`",
        "",
        "## Source Groups",
        "",
    ]
    for group, count in sorted((counts.get("source_groups") or {}).items()):
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## Failure Labels", ""])
    for label, count in sorted((counts.get("failure_labels") or {}).items()):
        lines.append(f"- `{label}`: `{count}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CascadeSourcePolicy supervision from route-tree traces")
    ap.add_argument("--trace", nargs="+", required=True, help="Route-tree trace JSONL path(s)")
    ap.add_argument("--output-dir", required=True, help="Directory for source_policy_pack.jsonl and manifest")
    ap.add_argument("--eval-only", action="store_true", help="Mark rows as diagnostic/eval-only")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    manifest = build_route_tree_source_policy_pack(
        trace_paths=[Path(path) for path in args.trace],
        output_dir=Path(args.output_dir),
        eval_only=args.eval_only,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
