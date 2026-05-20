"""Build reservoir-distillation packs from route-tree traces and teacher routes."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles, route_recovery_metrics, target_recovery_metrics
from cascade_planner.eval.chem_enzy_broad_union import _chem_route_stock_closed, _convert_chem_route, _select_chem_routes
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import source_group, source_policy_group
from cascade_planner.vnext.features import stable_bucket


RESERVOIR_SPLIT_NAMES = ("train", "val", "eval")


def build_reservoir_distill_pack(
    *,
    trace_paths: Iterable[Path],
    output_dir: Path,
    benchmark_path: Path | None = None,
    autoplanner_path: Path | Iterable[Path] | None = None,
    chem_enzy_path: Path | Iterable[Path] | None = None,
    reservoir_payload_path: Path | Iterable[Path] | None = None,
    val_trace_paths: Iterable[Path] | None = None,
    eval_trace_paths: Iterable[Path] | None = None,
    eval_benchmark_path: Path | None = None,
    native_topk: int = 5,
    native_selection: str = "rank_plus_stock",
) -> dict[str, Any]:
    trace_paths = [Path(path) for path in trace_paths]
    val_trace_paths = [Path(path) for path in (val_trace_paths or [])]
    eval_trace_paths = [Path(path) for path in (eval_trace_paths or [])]
    benchmark_rows = _load_benchmark(benchmark_path)
    eval_benchmark_rows = _load_benchmark(eval_benchmark_path)
    benchmark_index = _index_by_benchmark(benchmark_rows)
    eval_benchmark_index = _index_by_benchmark(eval_benchmark_rows)
    auto_index = _target_index(_load_payloads(autoplanner_path))
    chem_index = _target_index(_load_payloads(chem_enzy_path))
    reservoir_index = _target_index(_load_payloads(reservoir_payload_path))

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in RESERVOIR_SPLIT_NAMES}
    teacher_stats = Counter()
    source_stats = Counter()
    split_counts = Counter()
    teacher_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for trace_path in trace_paths:
        for ordinal, trace_row in enumerate(_load_trace_rows(trace_path)):
            split = _trace_split(trace_row, ordinal, eval_only=False, eval_benchmark_index=eval_benchmark_index)
            candidate_rows, stats = _rows_from_trace_row(
                trace_row,
                benchmark_index=benchmark_index,
                auto_index=auto_index,
                chem_index=chem_index,
                reservoir_index=reservoir_index,
                native_topk=native_topk,
                native_selection=native_selection,
                teacher_cache=teacher_cache,
                eval_only=split == "eval",
            )
            rows[split].extend(candidate_rows)
            teacher_stats.update(stats)
            split_counts[split] += len(candidate_rows)
            for candidate_row in candidate_rows:
                source_stats[str(candidate_row.get("source") or "")] += 1

    for trace_path in val_trace_paths:
        for trace_row in _load_trace_rows(trace_path):
            candidate_rows, stats = _rows_from_trace_row(
                trace_row,
                benchmark_index=benchmark_index,
                auto_index=auto_index,
                chem_index=chem_index,
                reservoir_index=reservoir_index,
                native_topk=native_topk,
                native_selection=native_selection,
                teacher_cache=teacher_cache,
                eval_only=False,
            )
            rows["val"].extend(candidate_rows)
            teacher_stats.update(stats)
            split_counts["val"] += len(candidate_rows)
            for candidate_row in candidate_rows:
                source_stats[str(candidate_row.get("source") or "")] += 1

    for trace_path in eval_trace_paths:
        for ordinal, trace_row in enumerate(_load_trace_rows(trace_path)):
            candidate_rows, stats = _rows_from_trace_row(
                trace_row,
                benchmark_index=benchmark_index,
                auto_index=auto_index,
                chem_index=chem_index,
                reservoir_index=reservoir_index,
                native_topk=native_topk,
                native_selection=native_selection,
                teacher_cache=teacher_cache,
                eval_only=True,
            )
            rows["eval"].extend(candidate_rows)
            teacher_stats.update(stats)
            split_counts["eval"] += len(candidate_rows)
            for candidate_row in candidate_rows:
                source_stats[str(candidate_row.get("source") or "")] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train": str(output_dir / "reservoir_distill_pack_train.jsonl"),
        "val": str(output_dir / "reservoir_distill_pack_val.jsonl"),
        "eval": str(output_dir / "reservoir_distill_pack_eval_full100.jsonl"),
    }
    for split_name, path in files.items():
        _write_jsonl(Path(path), rows[split_name])

    manifest = {
        "schema_version": "reservoir_distill_pack.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "counts": {split: len(rows[split]) for split in RESERVOIR_SPLIT_NAMES},
        "sources": {
            "trace_paths": [str(path) for path in trace_paths],
            "val_trace_paths": [str(path) for path in val_trace_paths],
            "eval_trace_paths": [str(path) for path in eval_trace_paths],
            "benchmark_path": str(benchmark_path) if benchmark_path else None,
            "eval_benchmark_path": str(eval_benchmark_path) if eval_benchmark_path else None,
            "autoplanner_path": _manifest_paths(autoplanner_path),
            "chem_enzy_path": _manifest_paths(chem_enzy_path),
            "reservoir_payload_path": _manifest_paths(reservoir_payload_path),
        },
        "native_topk": native_topk,
        "native_selection": native_selection,
        "teacher_stats": dict(teacher_stats),
        "source_stats": dict(source_stats),
        "split_counts": dict(split_counts),
    }
    manifest_path = output_dir / "reservoir_distill_manifest.json"
    report_path = output_dir / "teacher_report.json"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["teacher_report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(_teacher_report(rows), indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _rows_from_trace_row(
    trace_row: dict[str, Any],
    *,
    benchmark_index: dict[int, dict[str, Any]],
    auto_index: dict[str, list[dict[str, Any]]],
    chem_index: dict[str, list[dict[str, Any]]],
    reservoir_index: dict[str, list[dict[str, Any]]],
    native_topk: int,
    native_selection: str,
    teacher_cache: dict[tuple[Any, ...], dict[str, Any]] | None,
    eval_only: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    event = trace_row.get("event") if isinstance(trace_row.get("event"), dict) else trace_row
    if not isinstance(event, dict):
        return [], Counter()
    candidates = event.get("candidate_actions") or []
    if not candidates:
        return [], Counter()
    target = str(trace_row.get("target_smiles") or event.get("state", {}).get("target") or event.get("target_smiles") or "")
    benchmark_index_value = trace_row.get("benchmark_index")
    benchmark_entry = _benchmark_entry(benchmark_index, trace_row, target)
    auto_row = _lookup_target(auto_index, trace_row, target, benchmark_index_value)
    chem_row = _lookup_target(chem_index, trace_row, target, benchmark_index_value)
    reservoir_row = _lookup_target(reservoir_index, trace_row, target, benchmark_index_value)
    teacher_bundle = _teacher_bundle(
        target=target,
        benchmark_index=benchmark_index_value,
        benchmark_entry=benchmark_entry or trace_row,
        auto_row=auto_row,
        chem_row=chem_row,
        reservoir_row=reservoir_row,
        native_topk=native_topk,
        native_selection=native_selection,
        cache=teacher_cache,
    )
    teacher_routes = teacher_bundle["routes"]
    teacher_recovery = teacher_bundle["recovery"]
    teacher_route_infos = teacher_bundle["route_infos"]
    selected_key = str(event.get("selected_action_key") or "")
    allocation = _extract_allocation(event)
    source_diag = _extract_source_diagnostics(event)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for rank, action_dict in enumerate(candidates, start=1):
        candidate = CandidateAction.from_candidate(target, action_dict, rank=rank, source=action_dict.get("source"))
        candidate_row = _candidate_row(
            trace_row,
            event,
            benchmark_entry or trace_row,
            candidate,
            candidate_dict=action_dict,
            selected_key=selected_key,
            teacher_routes=teacher_routes,
            teacher_route_infos=teacher_route_infos,
            teacher_recovery=teacher_recovery,
            allocation=allocation,
            source_diagnostics=source_diag,
            eval_only=eval_only,
            benchmark_index=benchmark_index_value,
        )
        rows.append(candidate_row)
        stats["rows"] += 1
        stats[f"source_group:{candidate_row['source_group']}"] += 1
        if candidate_row["teacher_selected"]:
            stats["teacher_selected"] += 1
        if candidate_row["teacher_stock_closed"]:
            stats["teacher_stock_closed"] += 1
        if candidate_row["teacher_exact_hit"]:
            stats["teacher_exact_hit"] += 1
        if candidate_row["teacher_gt_reactant_hit"]:
            stats["teacher_gt_reactant_hit"] += 1
    return rows, stats


def _candidate_row(
    trace_row: dict[str, Any],
    event: dict[str, Any],
    benchmark_row: dict[str, Any],
    candidate: CandidateAction,
    *,
    candidate_dict: dict[str, Any],
    selected_key: str,
    teacher_routes: list[dict[str, Any]],
    teacher_route_infos: list[dict[str, Any]],
    teacher_recovery: dict[str, Any],
    allocation: dict[str, Any],
    source_diagnostics: dict[str, Any],
    eval_only: bool,
    benchmark_index: Any,
) -> dict[str, Any]:
    target = str(trace_row.get("target_smiles") or event.get("state", {}).get("target") or "")
    route_context = _route_context(trace_row, event)
    candidate_route = _single_step_route(candidate, candidate_dict)
    candidate_recovery = route_recovery_metrics(candidate_route, benchmark_row or trace_row)
    teacher_exact_hit = bool(
        candidate_recovery.get("candidate_exact_reaction_hit")
        or candidate_recovery.get("exact_reaction_in_route_pool")
        or (candidate_recovery.get("exact_reaction_hits") or 0) > 0
        or candidate_recovery.get("exact_route_reaction_match")
    )
    teacher_gt_reactant_hit = bool(
        candidate_recovery.get("candidate_gt_reactant_hit")
        or candidate_recovery.get("gt_reactant_in_route_pool")
        or candidate_recovery.get("gt_reactant_hit")
    )
    teacher_route_rank, teacher_route_info = _teacher_route_rank(candidate, teacher_route_infos)
    teacher_route = (teacher_route_info or {}).get("route")
    teacher_stock_closed = bool(teacher_route and (teacher_route.get("metrics") or {}).get("strict_stock_solve"))
    teacher_selected = bool(selected_key and selected_key == candidate.canonical_key)
    teacher_route_label = _teacher_route_label(
        candidate=candidate,
        teacher_route=teacher_route,
    )
    teacher_action_label = _teacher_action_label(
        candidate=candidate,
        teacher_stock_closed=teacher_stock_closed,
        teacher_route_cost=teacher_route_label["cost"],
    )
    failure_labels = list(candidate_recovery.get("recovery_bottleneck_labels") or [])
    if teacher_route and (teacher_route.get("metrics") or {}).get("strict_stock_solve") is False:
        failure_labels.append("stock_dead_end")
    if bool(event.get("expanded_leaf_low_yield")):
        failure_labels.append("repeated_low_yield_expansion")
    if len(teacher_routes) > 5:
        failure_labels.append("excessive_route_duplication")
    source = str(candidate.source or candidate_dict.get("source") or "unknown")
    row = {
        "state_id": str(event.get("state_id") or ""),
        "target_id": str(trace_row.get("cascade_id") or target or benchmark_index or ""),
        "target_smiles": target,
        "benchmark_index": benchmark_index,
        "depth": int(event.get("depth") or 0),
        "remaining_depth": max(0, int(trace_row.get("max_depth") or 6) - int(event.get("depth") or 0)),
        "leaf": str(event.get("expanded_leaf") or target or ""),
        "source": source,
        "source_group": source_group(source),
        "source_policy_group": source_policy_group(source),
        "candidate_reaction": candidate.rxn_smiles,
        "reactants": list(candidate.reactants),
        "route_context_features": route_context,
        "source_diagnostics": source_diagnostics,
        "reservoir_rank": int(candidate.rank or 0),
        "teacher_selected": teacher_selected,
        "teacher_route_rank": teacher_route_rank,
        "teacher_stock_closed": teacher_stock_closed,
        "teacher_exact_hit": teacher_exact_hit,
        "teacher_gt_reactant_hit": teacher_gt_reactant_hit,
        "teacher_route_value": round(float(teacher_route_label["value"]), 6),
        "teacher_action_value": round(float(teacher_action_label["value"]), 6),
        "teacher_route_cost": round(float(teacher_route_label["cost"]), 6),
        "teacher_action_cost": round(float(teacher_action_label["cost"]), 6),
        "teacher_value_policy": "reaction_cost_and_or.v1",
        "budget_label": _budget_label(allocation, event),
        "teacher_source_group_distribution": _source_group_distribution(allocation, source),
        "failure_labels": sorted(set(failure_labels)),
        "latency_ms": round(float(_latency_ms_for_source(source_diagnostics, source)), 3),
        "eval_only": bool(eval_only),
        "candidate_rank": int(candidate.rank or 0),
        "candidate_score": float(candidate.raw_score or 0.0),
        "candidate_exact_reaction_in_pool": teacher_exact_hit,
        "candidate_gt_reactant_in_pool": teacher_gt_reactant_hit,
        "teacher_route_count": len(teacher_routes),
    }
    row["teacher_recovery"] = {
        "exact_reaction_in_route_pool": bool(teacher_recovery.get("exact_reaction_in_route_pool")),
        "gt_reactant_in_route_pool": bool(teacher_recovery.get("gt_reactant_in_route_pool")),
        "candidate_exact_reaction_in_pool": bool(teacher_recovery.get("candidate_exact_reaction_in_pool")),
        "candidate_gt_reactant_in_pool": bool(teacher_recovery.get("candidate_gt_reactant_in_pool")),
    }
    row["candidate_recovery"] = {
        "exact_reaction_in_route_pool": bool(candidate_recovery.get("exact_reaction_in_route_pool")),
        "gt_reactant_in_route_pool": bool(candidate_recovery.get("gt_reactant_in_route_pool")),
        "candidate_exact_reaction_in_pool": teacher_exact_hit,
        "candidate_gt_reactant_in_pool": teacher_gt_reactant_hit,
    }
    return row


def _teacher_bundle(
    *,
    target: str,
    benchmark_index: Any,
    benchmark_entry: dict[str, Any],
    auto_row: dict[str, Any],
    chem_row: dict[str, Any],
    reservoir_row: dict[str, Any],
    native_topk: int,
    native_selection: str,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None,
) -> dict[str, Any]:
    cache_key = _teacher_cache_key(
        target=target,
        benchmark_index=benchmark_index,
        auto_row=auto_row,
        chem_row=chem_row,
        reservoir_row=reservoir_row,
        native_topk=native_topk,
        native_selection=native_selection,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    teacher_routes = _teacher_routes(auto_row, chem_row, reservoir_row, native_topk=native_topk, native_selection=native_selection)
    teacher_recovery = target_recovery_metrics(teacher_routes, benchmark_entry)
    bundle = {
        "routes": teacher_routes,
        "recovery": teacher_recovery,
        "route_infos": _teacher_route_infos(teacher_routes, teacher_recovery),
    }
    if cache is not None:
        cache[cache_key] = bundle
    return bundle


def _teacher_cache_key(
    *,
    target: str,
    benchmark_index: Any,
    auto_row: dict[str, Any],
    chem_row: dict[str, Any],
    reservoir_row: dict[str, Any],
    native_topk: int,
    native_selection: str,
) -> tuple[Any, ...]:
    return (
        benchmark_index,
        canonical_smiles(target) or target,
        _row_identity(auto_row),
        _row_identity(chem_row),
        _row_identity(reservoir_row),
        int(native_topk),
        native_selection,
    )


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    if not isinstance(row, dict) or not row:
        return ("missing",)
    routes = row.get("routes")
    planner_routes = (row.get("planner_output") or {}).get("routes")
    return (
        row.get("index"),
        row.get("target_smiles"),
        id(row),
        len(routes or []),
        len(planner_routes or []),
        bool((row.get("metrics") or {}).get("strict_stock_solve_any")),
    )


def _teacher_routes(
    auto_row: dict[str, Any],
    chem_row: dict[str, Any],
    reservoir_row: dict[str, Any],
    *,
    native_topk: int,
    native_selection: str,
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    auto_stock_closed = False
    if isinstance(auto_row, dict):
        routes.extend([dict(route) for route in ((auto_row.get("planner_output") or {}).get("routes") or [])])
        auto_stock_closed = bool((auto_row.get("metrics") or {}).get("strict_stock_solve_any"))
    if isinstance(reservoir_row, dict):
        routes.extend([dict(route) for route in ((reservoir_row.get("planner_output") or {}).get("routes") or [])])
    chem_routes = list((chem_row or {}).get("routes") or [])
    if chem_routes:
        selected_topk = min(native_topk, 1) if auto_stock_closed else native_topk
        selected = _select_chem_routes(chem_routes, topk=selected_topk, selection=native_selection)
        routes.extend(
            _convert_chem_route(route, native_rank=route.get("_native_rank"), stock_closed=_chem_route_stock_closed(route))
            for route in selected
        )
    return routes


def _teacher_route_infos(teacher_routes: list[dict[str, Any]], teacher_recovery: dict[str, Any]) -> list[dict[str, Any]]:
    per_route = list(teacher_recovery.get("per_route") or [])
    infos: list[dict[str, Any]] = []
    for idx, route in enumerate(teacher_routes):
        route_keys = {
            key
            for step in route.get("steps") or []
            for key in [canonical_reaction(step.get("reaction_smiles"))]
            if key
        }
        infos.append(
            {
                "route": route,
                "metrics": per_route[idx] if idx < len(per_route) and isinstance(per_route[idx], dict) else {},
                "route_keys": route_keys,
            }
        )
    return infos


def _teacher_route_rank(candidate: CandidateAction, teacher_route_infos: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None]:
    candidate_keys = {
        key
        for key in [candidate.canonical_key, canonical_reaction(candidate.rxn_smiles)]
        if key
    }
    if not candidate_keys:
        return None, None
    for idx, info in enumerate(teacher_route_infos, start=1):
        if candidate_keys & set(info.get("route_keys") or []):
            return idx, info
    return None, None


def _teacher_route_label(
    *,
    candidate: CandidateAction,
    teacher_route: dict[str, Any] | None,
) -> dict[str, float]:
    cost = _route_cost(teacher_route) if teacher_route else _candidate_action_cost(candidate)
    return {"cost": float(cost), "value": _cost_to_value(cost)}


def _teacher_action_label(
    *,
    candidate: CandidateAction,
    teacher_stock_closed: bool,
    teacher_route_cost: float,
) -> dict[str, float]:
    action_cost = _candidate_action_cost(candidate)
    if teacher_stock_closed:
        action_cost = min(action_cost, float(teacher_route_cost))
    return {"cost": float(action_cost), "value": _cost_to_value(action_cost)}


def _route_cost(route: dict[str, Any] | None) -> float:
    if not route:
        return math.log1p(1.0)
    steps = list(route.get("steps") or [])
    step_cost = sum(_step_cost(step) for step in steps)
    if not steps:
        step_cost = _negative_log_probability(_probability_from_score(route.get("score")))
    stock_closed = bool((route.get("metrics") or {}).get("strict_stock_solve"))
    terminal_gap_cost = 0.0 if stock_closed else math.log1p(float(_nonstock_terminal_count(route) or 1))
    duplicate_cost = _negative_log_probability(1.0 - min(_duplicate_reaction_fraction(steps), 0.999))
    return float(step_cost + terminal_gap_cost + duplicate_cost)


def _candidate_action_cost(candidate: CandidateAction) -> float:
    return _negative_log_probability(_probability_from_score(candidate.raw_score)) + math.log1p(float(len(candidate.reactants) or 1))


def _step_cost(step: dict[str, Any]) -> float:
    scores = step.get("scores") if isinstance(step.get("scores"), dict) else {}
    probability = 0.0
    for value in (
        step.get("score"),
        scores.get("retro") if scores else None,
        scores.get("confidence") if scores else None,
    ):
        probability = _probability_from_score(value)
        if probability > 0.0:
            break
    if probability > 0.0:
        return _negative_log_probability(probability)
    reactants = step.get("reactant_smiles") or []
    if not reactants:
        reactants = [step.get("main_reactant"), *(step.get("aux_reactants") or [])]
    return math.log1p(float(len([smi for smi in reactants if smi]) or 1))


def _nonstock_terminal_count(route: dict[str, Any]) -> int:
    count = 0
    for step in route.get("steps") or []:
        statuses = step.get("stock_status") or {}
        if isinstance(statuses, dict):
            count += sum(1 for value in statuses.values() if value is False)
    return count


def _duplicate_reaction_fraction(steps: list[dict[str, Any]]) -> float:
    keys = [canonical_reaction(str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")) for step in steps]
    keys = [key for key in keys if key]
    if not keys:
        return 0.0
    return 1.0 - len(set(keys)) / max(len(keys), 1)


def _probability_from_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 0.0:
        return 0.0
    return min(1.0, score)


def _negative_log_probability(probability: Any) -> float:
    probability = max(1e-6, _probability_from_score(probability))
    return -math.log(probability)


def _cost_to_value(cost: Any) -> float:
    try:
        value = float(cost)
    except (TypeError, ValueError):
        value = math.inf
    if not math.isfinite(value):
        return 0.0
    return 1.0 / (1.0 + max(0.0, value))


def _route_context(trace_row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    context = {
        "state_id": str(event.get("state_id") or ""),
        "depth": int(event.get("depth") or 0),
        "open_leaf_count": len(event.get("open_leaves") or []),
        "selected_next_open_leaves": event.get("selected_next_open_leaves"),
        "selected_next_stock_closed": event.get("selected_next_stock_closed"),
        "expanded_leaf_stock_hit": bool(event.get("expanded_leaf_stock_hit")),
        "expanded_leaf_parent_adjacent": bool(event.get("expanded_leaf_parent_adjacent")),
        "expanded_leaf_low_yield": bool(event.get("expanded_leaf_low_yield")),
        "elapsed_s": trace_row.get("elapsed_s"),
        "route_model_active": trace_row.get("route_model_active"),
        "proposal_budget": _proposal_budget(event),
        "route_tree_stop_reason": trace_row.get("planner_error") or "",
    }
    return context


def _proposal_budget(event: dict[str, Any]) -> int | None:
    diagnostics = event.get("proposal_diagnostics") or []
    for row in diagnostics:
        if row.get("proposal_budget") is not None:
            try:
                return int(row.get("proposal_budget"))
            except (TypeError, ValueError):
                continue
    return None


def _extract_allocation(event: dict[str, Any]) -> dict[str, Any]:
    diagnostics = event.get("proposal_diagnostics") or []
    for row in diagnostics:
        allocation = row.get("allocation")
        if isinstance(allocation, dict):
            return dict(allocation)
    return {}


def _extract_source_diagnostics(event: dict[str, Any]) -> dict[str, Any]:
    source_rows = {}
    for diag in event.get("proposal_diagnostics") or []:
        if not isinstance(diag, dict):
            continue
        for source, row in (diag.get("sources") or {}).items():
            if isinstance(row, dict):
                source_rows[str(source)] = dict(row)
    return source_rows


def _budget_label(allocation: dict[str, Any], event: dict[str, Any]) -> str:
    label = str(allocation.get("budget_multiplier_label") or "")
    if label:
        return label
    value = allocation.get("budget_multiplier")
    if value is not None:
        try:
            return {0.5: "0.5x", 1.0: "1x", 2.0: "2x", 3.0: "3x"}[min([0.5, 1.0, 2.0, 3.0], key=lambda x: abs(x - float(value)))]
        except Exception:
            pass
    budget = _proposal_budget(event) or 1
    if budget <= 2:
        return "0.5x"
    if budget <= 4:
        return "1x"
    if budget <= 8:
        return "2x"
    return "3x"


def _source_group_distribution(allocation: dict[str, Any], source: str) -> dict[str, float]:
    group_probs = allocation.get("source_group_probs")
    if isinstance(group_probs, dict) and group_probs:
        total = sum(max(0.0, float(value)) for value in group_probs.values())
        if total > 0:
            return {str(key): float(max(0.0, float(value)) / total) for key, value in group_probs.items()}
    source_weights = allocation.get("source_weights")
    if isinstance(source_weights, dict) and source_weights:
        grouped: dict[str, float] = defaultdict(float)
        for item, value in source_weights.items():
            grouped[source_policy_group(str(item))] += max(0.0, float(value))
        total = sum(grouped.values())
        if total > 0:
            return {group: value / total for group, value in grouped.items()}
    group = source_policy_group(source)
    return {group: 1.0}


def _latency_ms_for_source(source_diagnostics: dict[str, Any], source: str) -> float:
    row = source_diagnostics.get(source)
    if not isinstance(row, dict):
        return 0.0
    return float(row.get("latency_ms_total") or 0.0)


def _single_step_route(candidate: CandidateAction, candidate_dict: dict[str, Any]) -> dict[str, Any]:
    reactants = [str(smi) for smi in candidate.reactants if smi]
    return {
        "steps": [
            {
                "reaction_smiles": candidate.rxn_smiles or candidate_dict.get("rxn_smiles") or candidate_dict.get("reaction_smiles") or "",
                "reaction_type": candidate.reaction_type or candidate_dict.get("reaction_type") or candidate_dict.get("type") or "",
                "main_reactant": candidate.main_reactant or candidate_dict.get("main_reactant") or "",
                "aux_reactants": list(candidate.aux_reactants or candidate_dict.get("aux_reactants") or []),
                "source": candidate.source or candidate_dict.get("source") or "",
                "stock_status": {smi: None for smi in reactants},
            }
        ]
    }


def _load_benchmark(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    return []


def _index_by_benchmark(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for idx, row in enumerate(rows):
        try:
            key = int(row.get("index", idx))
        except (TypeError, ValueError):
            key = idx
        out[key] = row
    return out


def _benchmark_entry(index: dict[int, dict[str, Any]], trace_row: dict[str, Any], target: str) -> dict[str, Any]:
    benchmark_index = trace_row.get("benchmark_index")
    if benchmark_index is not None:
        try:
            item = index.get(int(benchmark_index))
            if item is not None:
                return item
        except (TypeError, ValueError):
            pass
    for row in index.values():
        if str(row.get("target_smiles") or "") == target:
            return row
    return trace_row if "gt_route" in trace_row else {}


def _lookup_target(index: dict[str, list[dict[str, Any]]], trace_row: dict[str, Any], target: str, benchmark_index: Any) -> dict[str, Any]:
    rows = index.get(str(target)) or []
    if not rows:
        return {}
    if benchmark_index is not None:
        for row in rows:
            try:
                if int(row.get("index")) == int(benchmark_index):
                    return row
            except (TypeError, ValueError):
                continue
    return rows[0]


def _target_index(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not payload:
        return {}
    targets = payload.get("targets") or []
    for row in targets:
        if not isinstance(row, dict):
            continue
        target = str(row.get("target_smiles") or "")
        if target:
            index[target].append(row)
    return dict(index)


def _load_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_payloads(paths: Path | Iterable[Path] | None) -> dict[str, Any]:
    out = {"targets": []}
    for path in _path_list(paths):
        payload = _load_payload(path)
        if isinstance(payload.get("targets"), list):
            out["targets"].extend(row for row in payload["targets"] if isinstance(row, dict))
    return out


def _path_list(paths: Path | Iterable[Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _manifest_paths(paths: Path | Iterable[Path] | None) -> list[str] | None:
    values = [str(path) for path in _path_list(paths)]
    return values or None


def _load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _teacher_report(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for split_rows in rows.values() for row in split_rows]
    counter = Counter()
    for row in all_rows:
        counter["rows"] += 1
        if row.get("teacher_selected"):
            counter["teacher_selected"] += 1
        if row.get("teacher_stock_closed"):
            counter["teacher_stock_closed"] += 1
        if row.get("teacher_exact_hit"):
            counter["teacher_exact_hit"] += 1
        if row.get("teacher_gt_reactant_hit"):
            counter["teacher_gt_reactant_hit"] += 1
        if row.get("eval_only"):
            counter["eval_only"] += 1
    return {
        "schema_version": "reservoir_distill_teacher_report.v1",
        "counts": dict(counter),
        "by_split": {
            split: {
                "rows": len(split_rows),
                "teacher_selected": sum(1 for row in split_rows if row.get("teacher_selected")),
                "teacher_stock_closed": sum(1 for row in split_rows if row.get("teacher_stock_closed")),
                "teacher_exact_hit": sum(1 for row in split_rows if row.get("teacher_exact_hit")),
                "teacher_gt_reactant_hit": sum(1 for row in split_rows if row.get("teacher_gt_reactant_hit")),
                "eval_only": sum(1 for row in split_rows if row.get("eval_only")),
            }
            for split, split_rows in rows.items()
        },
    }


def _trace_split(
    trace_row: dict[str, Any],
    ordinal: int,
    *,
    eval_only: bool,
    eval_benchmark_index: dict[int, dict[str, Any]],
) -> str:
    if eval_only:
        return "eval"
    benchmark_index = trace_row.get("benchmark_index")
    if benchmark_index is not None:
        try:
            if int(benchmark_index) in eval_benchmark_index:
                return "eval"
        except (TypeError, ValueError):
            pass
    return "val" if (int(trace_row.get("benchmark_index") or ordinal) % 5 == 0) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reservoir distillation pack files")
    ap.add_argument("--trace", action="append", default=[], help="Route-tree trace JSONL")
    ap.add_argument("--val-trace", action="append", default=[], help="Validation route-tree trace JSONL")
    ap.add_argument("--eval-trace", action="append", default=[], help="Eval-only route-tree trace JSONL")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--eval-benchmark", default=None)
    ap.add_argument("--autoplanner", action="append", default=[])
    ap.add_argument("--chem-enzy", action="append", default=[])
    ap.add_argument("--reservoir-payload", action="append", default=[])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--native-topk", type=int, default=5)
    ap.add_argument("--native-selection", default="rank_plus_stock", choices=["rank", "stock_first", "rank_plus_stock"])
    args = ap.parse_args()
    build_reservoir_distill_pack(
        trace_paths=[Path(path) for path in args.trace],
        val_trace_paths=[Path(path) for path in args.val_trace],
        eval_trace_paths=[Path(path) for path in args.eval_trace],
        benchmark_path=Path(args.benchmark) if args.benchmark else None,
        eval_benchmark_path=Path(args.eval_benchmark) if args.eval_benchmark else None,
        autoplanner_path=[Path(path) for path in args.autoplanner],
        chem_enzy_path=[Path(path) for path in args.chem_enzy],
        reservoir_payload_path=[Path(path) for path in args.reservoir_payload],
        output_dir=Path(args.output_dir) if args.output_dir else Path("results/shared") / f"reservoir_distill_{time.strftime('%Y%m%d')}",
        native_topk=args.native_topk,
        native_selection=args.native_selection,
    )


if __name__ == "__main__":
    main()
