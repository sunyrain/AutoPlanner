"""Evaluate ChemEnzy broad route pools combined with AutoPlanner route pools."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean
from typing import Any

from cascade_planner.cascadeboard.route_recovery import target_recovery_metrics


def build_union_report(
    *,
    benchmark_path: Path,
    chem_enzy_path: Path,
    autoplanner_path: Path,
    native_topk: int | None = None,
    autoplanner_topk: int | None = None,
    native_selection: str = "rank",
    sweep_native_topk: list[int | None] | None = None,
    synthesize_output: Path | None = None,
) -> dict[str, Any]:
    benchmark = _rows(json.loads(benchmark_path.read_text(encoding="utf-8")))
    chem_payload = json.loads(chem_enzy_path.read_text(encoding="utf-8"))
    auto_payload = json.loads(autoplanner_path.read_text(encoding="utf-8"))
    chem_index = _target_index(chem_payload.get("targets") or [])
    auto_index = _target_index(auto_payload.get("targets") or [])

    rows_out = _build_rows(
        benchmark=benchmark,
        chem_index=chem_index,
        auto_index=auto_index,
        native_topk=native_topk,
        autoplanner_topk=autoplanner_topk,
        native_selection=native_selection,
    )
    report = {
        "inputs": {
            "benchmark": str(benchmark_path),
            "chem_enzy": str(chem_enzy_path),
            "autoplanner": str(autoplanner_path),
            "native_topk": native_topk,
            "autoplanner_topk": autoplanner_topk,
            "native_selection": native_selection,
        },
        "summary": {
            "native": _summary(rows_out, "native"),
            "autoplanner": _summary(rows_out, "autoplanner"),
            "union": _summary(rows_out, "union"),
        },
        "delta_union_vs_autoplanner": _delta_summary(rows_out, base="autoplanner", candidate="union"),
        "targets": rows_out,
    }
    if sweep_native_topk:
        report["sweep"] = [
            _sweep_row(
                topk=topk,
                rows=_build_rows(
                    benchmark=benchmark,
                    chem_index=chem_index,
                    auto_index=auto_index,
                    native_topk=topk,
                    autoplanner_topk=autoplanner_topk,
                    native_selection=native_selection,
                ),
            )
            for topk in sweep_native_topk
        ]
        report["recommendation"] = _recommend_sweep(report["sweep"])
    if synthesize_output is not None:
        synthetic = build_synthetic_reservoir_payload(
            benchmark=benchmark,
            auto_payload=auto_payload,
            chem_index=chem_index,
            auto_index=auto_index,
            native_topk=native_topk,
            autoplanner_topk=autoplanner_topk,
            native_selection=native_selection,
        )
        synthesize_output.parent.mkdir(parents=True, exist_ok=True)
        synthesize_output.write_text(json.dumps(synthetic, indent=2, ensure_ascii=False), encoding="utf-8")
        report["synthetic_output"] = str(synthesize_output)
    return report


def build_synthetic_reservoir_payload(
    *,
    benchmark: list[dict[str, Any]],
    auto_payload: dict[str, Any],
    chem_index: dict[str, Any],
    auto_index: dict[str, Any],
    native_topk: int | None,
    autoplanner_topk: int | None,
    native_selection: str,
) -> dict[str, Any]:
    targets = []
    for benchmark_index, entry in enumerate(benchmark):
        target = str(entry.get("target_smiles") or "")
        if not target:
            continue
        auto_row = _lookup_target(auto_index, entry, benchmark_index)
        chem_row = _lookup_target(chem_index, entry, benchmark_index)
        target_row = copy.deepcopy(auto_row) if auto_row else {
            "index": benchmark_index,
            "target_smiles": target,
            "route_domain": entry.get("route_domain"),
            "depth": entry.get("depth"),
            "gt_route": entry.get("gt_route") or [],
            "metrics": {},
            "planner_output": {"target": target, "routes": []},
        }
        target_row.setdefault("index", benchmark_index)
        target_row.setdefault("target_smiles", target)
        target_row.setdefault("gt_route", entry.get("gt_route") or [])
        payload = target_row.setdefault("planner_output", {})
        auto_routes = list(((auto_row.get("planner_output") or {}).get("routes") or []))
        if autoplanner_topk is not None:
            auto_routes = auto_routes[: max(0, autoplanner_topk)]
        native_raw_routes = _select_payload_routes(
            _chem_payload_routes(chem_row),
            topk=native_topk,
            selection=native_selection,
        )
        native_routes = [
            _convert_payload_route(route, native_rank=route.get("_native_rank"), stock_closed=_payload_route_stock_closed(route))
            for route in native_raw_routes
        ]
        routes = [*auto_routes, *native_routes]
        payload["routes"] = routes
        payload["n_results"] = len(routes)
        payload.setdefault("target", target)
        payload["broad_reservoir"] = {
            "enabled": True,
            "native_topk": native_topk,
            "native_selection": native_selection,
            "native_route_count": len(native_routes),
            "autoplanner_route_count": len(auto_routes),
            "note": "Offline synthesized route reservoir. Runtime promotion still needs bounded online collection.",
        }
        recovery = target_recovery_metrics(routes, entry)
        metrics = target_row.setdefault("metrics", {})
        native_stock = any(_payload_route_stock_closed(route) for route in native_raw_routes)
        native_recovery = target_recovery_metrics(native_routes, entry)
        auto_recovery = target_recovery_metrics(auto_routes, entry)
        auto_stock = bool((auto_row.get("metrics") or {}).get("strict_stock_solve_any")) if auto_row else False
        metrics["plan"] = bool(routes) or bool(metrics.get("plan"))
        metrics["filled_route_any"] = bool(routes) or bool(metrics.get("filled_route_any"))
        metrics["strict_stock_solve_any"] = bool(auto_stock or native_stock)
        metrics["broad_reservoir_stock_any"] = bool(native_stock)
        metrics["broad_reservoir_route_count"] = len(native_routes)
        metrics["broad_reservoir_native_topk"] = native_topk
        target_row["route_recovery"] = {
            **recovery,
            "recovery_bottleneck_labels": [],
            "broad_reservoir_native": _selected_recovery_fields(native_recovery),
            "broad_reservoir_autoplanner": _selected_recovery_fields(auto_recovery),
        }
        targets.append(target_row)
    rows = _build_rows(
        benchmark=benchmark,
        chem_index=chem_index,
        auto_index=auto_index,
        native_topk=native_topk,
        autoplanner_topk=autoplanner_topk,
        native_selection=native_selection,
    )
    live_summary = _live_compatible_synthetic_summary(targets, auto_payload=auto_payload)
    return {
        "metadata": {
            "synthesized_from": {
                "autoplanner_metadata": auto_payload.get("metadata") or {},
                "native_topk": native_topk,
                "native_selection": native_selection,
                "autoplanner_topk": autoplanner_topk,
            },
            "metric_note": (
                "Routes are AutoPlanner outputs plus selected native ChemEnzy reservoir routes. "
                "This is an offline integration artifact, not an online runtime measurement."
            ),
        },
        "summary": {
            **live_summary,
            "autoplanner": _summary(rows, "autoplanner"),
            "broad_reservoir": _summary(rows, "native"),
            "synthesized_union": _summary(rows, "union"),
            "delta_union_vs_autoplanner": _delta_summary(rows, base="autoplanner", candidate="union"),
        },
        "targets": targets,
    }


def _build_rows(
    *,
    benchmark: list[dict[str, Any]],
    chem_index: dict[str, Any],
    auto_index: dict[str, Any],
    native_topk: int | None,
    autoplanner_topk: int | None,
    native_selection: str,
) -> list[dict[str, Any]]:
    rows_out = []
    for benchmark_index, entry in enumerate(benchmark):
        target = str(entry.get("target_smiles") or "")
        if not target:
            continue
        chem_row = _lookup_target(chem_index, entry, benchmark_index)
        auto_row = _lookup_target(auto_index, entry, benchmark_index)
        native_raw_routes = _select_payload_routes(
            _chem_payload_routes(chem_row or {}),
            topk=native_topk,
            selection=native_selection,
        )
        native_routes = [
            _convert_payload_route(route, native_rank=route.get("_native_rank"), stock_closed=_payload_route_stock_closed(route))
            for route in native_raw_routes
        ]
        auto_routes = list((((auto_row or {}).get("planner_output") or {}).get("routes") or []))
        if autoplanner_topk is not None:
            auto_routes = auto_routes[: max(0, autoplanner_topk)]
        native_stock = any(_payload_route_stock_closed(route) for route in native_raw_routes)
        auto_stock = bool(((auto_row or {}).get("metrics") or {}).get("strict_stock_solve_any"))
        native_recovery = target_recovery_metrics(native_routes, entry)
        auto_recovery = target_recovery_metrics(auto_routes, entry)
        union_recovery = target_recovery_metrics([*auto_routes, *native_routes], entry)
        rows_out.append(
            {
                "benchmark_index": benchmark_index,
                "target_smiles": target,
                "native_route_count": len(native_routes),
                "autoplanner_route_count": len(auto_routes),
                "union_route_count": len(native_routes) + len(auto_routes),
                "native_stock": native_stock,
                "autoplanner_stock": auto_stock,
                "union_stock": bool(native_stock or auto_stock),
                "native_recovery": _selected_recovery_fields(native_recovery),
                "autoplanner_recovery": _selected_recovery_fields(auto_recovery),
                "union_recovery": _selected_recovery_fields(union_recovery),
            }
        )
    return rows_out


def _sweep_row(*, topk: int | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "native": _summary(rows, "native"),
        "autoplanner": _summary(rows, "autoplanner"),
        "union": _summary(rows, "union"),
    }
    return {
        "native_topk": topk,
        "summary": summary,
        "delta_union_vs_autoplanner": _delta_summary(rows, base="autoplanner", candidate="union"),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    delta = report.get("delta_union_vs_autoplanner") or {}
    lines = [
        "# ChemEnzy Broad Union Report",
        "",
        "This is an offline route-pool upper-bound check: native ChemEnzy routes are merged with AutoPlanner routes before recovery metrics are computed.",
        "",
        "| Metric | Native ChemEnzy | AutoPlanner | Union |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "stock_rate",
        "exact_reaction_in_route_pool",
        "exact_route_reaction_match_any",
        "gt_reactant_in_route_pool",
        "avg_best_exact_reaction_fraction",
        "avg_best_reaction_edit_distance",
        "avg_route_count",
    ):
        lines.append(
            "| {key} | {native} | {auto} | {union} |".format(
                key=key,
                native=_fmt((summary.get("native") or {}).get(key)),
                auto=_fmt((summary.get("autoplanner") or {}).get(key)),
                union=_fmt((summary.get("union") or {}).get(key)),
            )
        )
    lines.extend(
        [
            "",
            "## Union Gains",
            "",
            f"- Stock gains over AutoPlanner: `{delta.get('stock_gain_targets')}` targets",
            f"- Exact-reaction gains over AutoPlanner: `{delta.get('exact_reaction_gain_targets')}` targets",
            f"- GT-reactant gains over AutoPlanner: `{delta.get('gt_reactant_gain_targets')}` targets",
            f"- Exact-reaction losses: `{delta.get('exact_reaction_loss_targets')}` targets",
            f"- GT-reactant losses: `{delta.get('gt_reactant_loss_targets')}` targets",
            "",
            "Interpretation: if union improves stock without reducing recovery, the broad collector is worth integrating as a reservoir. Runtime promotion still requires a non-oracle reranker and a bounded route budget.",
            "",
        ]
    )
    if report.get("sweep"):
        lines.extend(
            [
                "## Native Top-k Sweep",
                "",
                "| Native top-k | Union stock | Union exact | Union GT reactant | Avg routes | Stock gains | Exact gains | GT gains |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["sweep"]:
            union = ((row.get("summary") or {}).get("union") or {})
            delta_row = row.get("delta_union_vs_autoplanner") or {}
            lines.append(
                "| {topk} | {stock} | {exact} | {gt} | {routes} | {stock_gain} | {exact_gain} | {gt_gain} |".format(
                    topk=_topk_label(row.get("native_topk")),
                    stock=_fmt(union.get("stock_rate")),
                    exact=_fmt(union.get("exact_reaction_in_route_pool")),
                    gt=_fmt(union.get("gt_reactant_in_route_pool")),
                    routes=_fmt(union.get("avg_route_count")),
                    stock_gain=delta_row.get("stock_gain_targets"),
                    exact_gain=delta_row.get("exact_reaction_gain_targets"),
                    gt_gain=delta_row.get("gt_reactant_gain_targets"),
                )
            )
        recommendation = report.get("recommendation") or {}
        if recommendation:
            lines.extend(
                [
                    "",
                    "## Recommendation",
                    "",
                    f"- Native top-k: `{_topk_label(recommendation.get('native_topk'))}`",
                    f"- Reason: {recommendation.get('reason')}",
                    "",
                ]
            )
    if report.get("synthetic_output"):
        lines.extend(["## Synthetic Output", "", f"- `{report['synthetic_output']}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n = len(rows)

    def rate(field: str) -> float | None:
        if not n:
            return None
        return sum(1 for row in rows if ((row.get(f"{prefix}_recovery") or {}).get(field))) / n

    best_fracs = [
        (row.get(f"{prefix}_recovery") or {}).get("best_exact_reaction_fraction")
        for row in rows
        if (row.get(f"{prefix}_recovery") or {}).get("best_exact_reaction_fraction") is not None
    ]
    edit_distances = [
        (row.get(f"{prefix}_recovery") or {}).get("best_reaction_edit_distance")
        for row in rows
        if (row.get(f"{prefix}_recovery") or {}).get("best_reaction_edit_distance") is not None
    ]
    route_counts = [int(row.get(f"{prefix}_route_count") or 0) for row in rows]
    return {
        "n_targets": n,
        "stock_rate": sum(1 for row in rows if row.get(f"{prefix}_stock")) / n if n else None,
        "exact_reaction_in_route_pool": rate("exact_reaction_in_route_pool"),
        "exact_route_reaction_match_any": rate("exact_route_reaction_match_any"),
        "gt_reactant_in_route_pool": rate("gt_reactant_in_route_pool"),
        "avg_best_exact_reaction_fraction": mean(best_fracs) if best_fracs else None,
        "avg_best_reaction_edit_distance": mean(edit_distances) if edit_distances else None,
        "avg_route_count": mean(route_counts) if route_counts else None,
    }


def _live_compatible_synthetic_summary(targets: list[dict[str, Any]], *, auto_payload: dict[str, Any]) -> dict[str, Any]:
    n = len(targets)

    def metric_rate(field: str) -> float | None:
        if not n:
            return None
        seen = False
        hits = 0
        for row in targets:
            metrics = row.get("metrics") or {}
            if field not in metrics:
                continue
            seen = True
            hits += int(bool(metrics.get(field)))
        return hits / n if seen else None

    def recovery_rate(field: str) -> float | None:
        if not n:
            return None
        seen = False
        hits = 0
        for row in targets:
            recovery = row.get("route_recovery") or {}
            if field not in recovery:
                continue
            seen = True
            hits += int(bool(recovery.get(field)))
        return hits / n if seen else None

    route_counts = [
        len(((row.get("planner_output") or {}).get("routes") or []))
        for row in targets
    ]
    out = {
        "plan_rate": metric_rate("plan"),
        "strict_stock_solve_any": metric_rate("strict_stock_solve_any"),
        "candidate_exact_reaction_in_pool": recovery_rate("candidate_exact_reaction_in_pool"),
        "candidate_gt_reactant_in_pool": recovery_rate("candidate_gt_reactant_in_pool"),
        "exact_reaction_in_route_pool": recovery_rate("exact_reaction_in_route_pool"),
        "gt_reactant_in_route_pool": recovery_rate("gt_reactant_in_route_pool"),
        "avg_route_count": mean(route_counts) if route_counts else None,
    }
    auto_summary = auto_payload.get("summary") or {}
    if auto_summary.get("avg_time_per_target_s") is not None:
        out["avg_time_per_target_s"] = auto_summary.get("avg_time_per_target_s")
        out["avg_time_source"] = "autoplanner_reused_offline_append_only"
    return out


def _delta_summary(rows: list[dict[str, Any]], *, base: str, candidate: str) -> dict[str, Any]:
    out = {
        "stock_gain_targets": 0,
        "exact_reaction_gain_targets": 0,
        "gt_reactant_gain_targets": 0,
        "exact_reaction_loss_targets": 0,
        "gt_reactant_loss_targets": 0,
    }
    for row in rows:
        base_rec = row.get(f"{base}_recovery") or {}
        cand_rec = row.get(f"{candidate}_recovery") or {}
        if row.get(f"{candidate}_stock") and not row.get(f"{base}_stock"):
            out["stock_gain_targets"] += 1
        if cand_rec.get("exact_reaction_in_route_pool") and not base_rec.get("exact_reaction_in_route_pool"):
            out["exact_reaction_gain_targets"] += 1
        if cand_rec.get("gt_reactant_in_route_pool") and not base_rec.get("gt_reactant_in_route_pool"):
            out["gt_reactant_gain_targets"] += 1
        if base_rec.get("exact_reaction_in_route_pool") and not cand_rec.get("exact_reaction_in_route_pool"):
            out["exact_reaction_loss_targets"] += 1
        if base_rec.get("gt_reactant_in_route_pool") and not cand_rec.get("gt_reactant_in_route_pool"):
            out["gt_reactant_loss_targets"] += 1
    return out


def _selected_recovery_fields(recovery: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "exact_reaction_in_route_pool",
        "exact_route_reaction_match_any",
        "gt_reactant_in_route_pool",
        "best_exact_reaction_fraction",
        "best_reaction_edit_distance",
    )
    return {key: recovery.get(key) for key in keys}


def _convert_chem_route(
    route: dict[str, Any],
    *,
    native_rank: int | None = None,
    stock_closed: bool | None = None,
) -> dict[str, Any]:
    steps = []
    products = {step.get("product_smiles") for step in route.get("steps") or [] if step.get("product_smiles")}
    terminal_reactants = []
    for step_index, step in enumerate(route.get("steps") or []):
        reactants = [str(smi) for smi in step.get("reactant_smiles") or [] if smi]
        for smi in reactants:
            if smi and smi not in products and smi not in terminal_reactants:
                terminal_reactants.append(smi)
        steps.append(
            {
                "index": step_index,
                "product": step.get("product_smiles"),
                "reaction_smiles": step.get("rxn_smiles"),
                "main_reactant": reactants[0] if reactants else "",
                "aux_reactants": reactants[1:],
                "reaction_type": step.get("reaction_type") or "",
                "source": step.get("source_model") or "ChemEnzyRetroPlanner",
                "scores": {"retro": step.get("score"), "confidence": step.get("score")},
                "is_filled": True,
                "stock_status": step.get("stock_status") or {},
                "reaction_interpretation": {
                    "reaction_class": step.get("reaction_type") or "unknown",
                    "atom_change": {},
                    "source_model": step.get("source_model"),
                    "template": (step.get("raw_backend_metadata") or {}).get("template"),
                },
                "broad_reservoir": {
                    "source": "native_chem_enzy",
                    "native_rank": native_rank,
                    "stock_closed": stock_closed,
                },
            }
        )
    return {
        "score": route.get("score"),
        "n_steps": len(steps),
        "steps": steps,
        "metrics": {
            **({"strict_stock_solve": stock_closed, "route_solved": stock_closed} if stock_closed is not None else {}),
            "filled_route": bool(steps),
            "terminal_reactants": terminal_reactants,
            "route_naturalness": {},
        },
        "broad_reservoir": {
            "source": "native_chem_enzy",
            "native_rank": native_rank,
            "stock_closed": stock_closed,
            "original_route_count": route.get("route_count"),
        },
    }


def _chem_payload_routes(row: dict[str, Any]) -> list[dict[str, Any]]:
    routes = list((row or {}).get("routes") or [])
    if routes:
        return routes
    exported = list((((row or {}).get("planner_output") or {}).get("routes") or []))
    reservoir_routes = [route for route in exported if isinstance(route, dict) and route.get("broad_reservoir")]
    return reservoir_routes or exported


def _select_payload_routes(
    routes: list[dict[str, Any]],
    *,
    topk: int | None,
    selection: str,
) -> list[dict[str, Any]]:
    if not routes:
        return []
    if _native_route_payload_format(routes):
        return _select_chem_routes(routes, topk=topk, selection=selection)
    annotated = []
    for rank, route in enumerate(routes, 1):
        item = copy.deepcopy(route)
        broad = item.get("broad_reservoir") or {}
        item["_native_rank"] = int(broad.get("native_rank") or item.get("route_rank") or rank)
        annotated.append(item)
    if selection == "rank_plus_stock":
        if topk is None:
            selected = annotated
        else:
            k = max(0, int(topk))
            selected = list(annotated[:k])
            if k > 0 and not any(_payload_route_stock_closed(route) for route in selected):
                stock_route = next((route for route in annotated if _payload_route_stock_closed(route)), None)
                if stock_route is not None:
                    selected = [*selected[: k - 1], stock_route]
    elif selection == "stock_first":
        selected = sorted(annotated, key=lambda route: (not _payload_route_stock_closed(route), route.get("_native_rank") or 0))
    elif selection == "rank":
        selected = annotated
    else:
        raise ValueError(f"unsupported native selection mode: {selection}")
    if topk is None:
        return selected
    return selected[: max(0, int(topk))]


def _convert_payload_route(
    route: dict[str, Any],
    *,
    native_rank: int | None,
    stock_closed: bool | None,
) -> dict[str, Any]:
    if _exported_route_payload_format(route):
        out = copy.deepcopy(route)
        broad = dict(out.get("broad_reservoir") or {})
        broad.setdefault("source", "native_chem_enzy")
        broad.setdefault("native_rank", native_rank)
        broad["stock_closed"] = stock_closed
        out["broad_reservoir"] = broad
        metrics = out.setdefault("metrics", {})
        if stock_closed is not None:
            metrics["strict_stock_solve"] = bool(stock_closed)
        return out
    return _convert_chem_route(route, native_rank=native_rank, stock_closed=stock_closed)


def _payload_route_stock_closed(route: dict[str, Any]) -> bool:
    if _exported_route_payload_format(route):
        metrics = route.get("metrics") or {}
        broad = route.get("broad_reservoir") or {}
        return bool(
            metrics.get("strict_stock_solve")
            or metrics.get("strict_stock_solve_any")
            or broad.get("stock_closed")
        )
    return _chem_route_stock_closed(route)


def _native_route_payload_format(routes: list[dict[str, Any]]) -> bool:
    for route in routes:
        steps = route.get("steps") or []
        if steps and any("product_smiles" in step or "reactant_smiles" in step for step in steps):
            return True
    return False


def _exported_route_payload_format(route: dict[str, Any]) -> bool:
    if route.get("broad_reservoir") or route.get("metrics"):
        return True
    for step in route.get("steps") or []:
        if "reaction_smiles" in step or "main_reactant" in step or "product" in step:
            return True
    return False


def _chem_route_stock_closed(route: dict[str, Any]) -> bool:
    products = {step.get("product_smiles") for step in route.get("steps") or [] if step.get("product_smiles")}
    terminal_flags = []
    for step in route.get("steps") or []:
        status = step.get("stock_status") or {}
        for smi in step.get("reactant_smiles") or []:
            if smi and smi not in products:
                terminal_flags.append(bool(status.get(smi)))
    return bool(terminal_flags) and all(terminal_flags)


def _select_chem_routes(
    routes: list[dict[str, Any]],
    *,
    topk: int | None,
    selection: str,
) -> list[dict[str, Any]]:
    annotated = []
    for rank, route in enumerate(routes, 1):
        item = dict(route)
        item["_native_rank"] = rank
        annotated.append(item)
    if selection == "rank":
        selected = annotated
    elif selection == "stock_first":
        selected = sorted(annotated, key=lambda route: (not _chem_route_stock_closed(route), route["_native_rank"]))
    elif selection == "rank_plus_stock":
        if topk is None:
            selected = annotated
        else:
            k = max(0, int(topk))
            selected = list(annotated[:k])
            if k > 0 and not any(_chem_route_stock_closed(route) for route in selected):
                stock_route = next((route for route in annotated if _chem_route_stock_closed(route)), None)
                if stock_route is not None:
                    selected = [*selected[: k - 1], stock_route]
    else:
        raise ValueError(f"unsupported native selection mode: {selection}")
    if topk is None:
        return selected
    return selected[: max(0, int(topk))]


def _target_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_index = {}
    by_target: dict[str, list[dict[str, Any]]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if idx is not None:
            try:
                by_index[int(idx)] = row
            except (TypeError, ValueError):
                pass
        target = str(row.get("target_smiles") or "")
        if target:
            by_target.setdefault(target, []).append(row)
        row.setdefault("_target_index_ordinal", ordinal)
    return {"by_index": by_index, "by_target": by_target}


def _lookup_target(index: dict[str, Any], entry: dict[str, Any], benchmark_index: int) -> dict[str, Any]:
    by_index = index.get("by_index") or {}
    if benchmark_index in by_index:
        return by_index[benchmark_index]
    target = str(entry.get("target_smiles") or "")
    rows = (index.get("by_target") or {}).get(target) or []
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    for row in rows:
        try:
            if int(row.get("index")) == int(benchmark_index):
                return row
        except (TypeError, ValueError):
            continue
    return rows[0]


def _recommend_sweep(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in sweep:
        summary = row.get("summary") or {}
        auto = summary.get("autoplanner") or {}
        union = summary.get("union") or {}
        if (union.get("stock_rate") or 0.0) < 0.95:
            continue
        if (union.get("exact_reaction_in_route_pool") or 0.0) < (auto.get("exact_reaction_in_route_pool") or 0.0):
            continue
        if (union.get("gt_reactant_in_route_pool") or 0.0) < (auto.get("gt_reactant_in_route_pool") or 0.0):
            continue
        candidates.append(row)
    if not candidates:
        return {"native_topk": None, "reason": "No swept top-k met stock >= 0.95 without recovery regression."}
    best = min(
        candidates,
        key=lambda row: (
            ((row.get("summary") or {}).get("union") or {}).get("avg_route_count") or 10**9,
            10**9 if row.get("native_topk") is None else int(row.get("native_topk")),
        ),
    )
    union = ((best.get("summary") or {}).get("union") or {})
    return {
        "native_topk": best.get("native_topk"),
        "reason": (
            "Smallest swept route pool meeting stock >= 0.95 without exact/GT recovery regression "
            f"(stock={_fmt(union.get('stock_rate'))}, exact={_fmt(union.get('exact_reaction_in_route_pool'))}, "
            f"GT={_fmt(union.get('gt_reactant_in_route_pool'))})."
        ),
    }


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    raise ValueError("unsupported benchmark format")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _topk_label(value: Any) -> str:
    return "all" if value is None else str(value)


def _parse_topk_list(raw: str | None) -> list[int | None] | None:
    if not raw:
        return None
    out = []
    for token in raw.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in {"all", "none", "full"}:
            out.append(None)
        else:
            out.append(int(value))
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ChemEnzy broad route union with AutoPlanner routes")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--chem-enzy", required=True)
    ap.add_argument("--autoplanner", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--native-topk", type=int, default=None)
    ap.add_argument("--autoplanner-topk", type=int, default=None)
    ap.add_argument("--native-selection", default="rank", choices=["rank", "stock_first", "rank_plus_stock"])
    ap.add_argument("--sweep-native-topk", default=None, help="Comma list such as 5,10,20,50,all")
    ap.add_argument("--synthesize-output", default=None, help="Optional synthesized AutoPlanner+reservoir JSON output")
    args = ap.parse_args()
    report = build_union_report(
        benchmark_path=Path(args.benchmark),
        chem_enzy_path=Path(args.chem_enzy),
        autoplanner_path=Path(args.autoplanner),
        native_topk=args.native_topk,
        autoplanner_topk=args.autoplanner_topk,
        native_selection=args.native_selection,
        sweep_native_topk=_parse_topk_list(args.sweep_native_topk),
        synthesize_output=Path(args.synthesize_output) if args.synthesize_output else None,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        write_markdown(report, Path(args.markdown))
    print(json.dumps({"output": str(out), "summary": report["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
