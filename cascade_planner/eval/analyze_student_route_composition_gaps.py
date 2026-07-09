"""Audit student-only route composition gaps against teacher/hybrid runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RECOVERY_KEYS = [
    "candidate_gt_reactant_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
]


def analyze_student_route_composition_gaps(
    *,
    runs: dict[str, Path],
    teacher_label: str,
    student_label: str,
    hybrid_label: str | None,
    output_json: Path,
    output_md: Path,
    indices: list[int] | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    loaded = {label: _load_run(path) for label, path in runs.items()}
    teacher = _targets_by_index(loaded[teacher_label])
    student = _targets_by_index(loaded[student_label])
    hybrid = _targets_by_index(loaded[hybrid_label]) if hybrid_label else {}
    selected_indices = indices or _derive_priority_indices(
        teacher=teacher,
        student=student,
        hybrid=hybrid,
        top_n=top_n,
    )
    rows = []
    for index in selected_indices:
        teacher_target = teacher.get(index)
        student_target = student.get(index)
        hybrid_target = hybrid.get(index) if hybrid else None
        if not teacher_target or not student_target:
            continue
        row = _audit_row(
            index=index,
            runs=loaded,
            target_by_label={
                label: (_targets_by_index(run).get(index) if label in loaded else None)
                for label, run in loaded.items()
            },
            teacher_label=teacher_label,
            student_label=student_label,
            hybrid_label=hybrid_label,
            teacher_target=teacher_target,
            student_target=student_target,
            hybrid_target=hybrid_target,
        )
        rows.append(row)
    report = {
        "schema_version": "student_route_composition_gap_audit.v1",
        "runs": {label: str(path) for label, path in runs.items()},
        "teacher_label": teacher_label,
        "student_label": student_label,
        "hybrid_label": hybrid_label,
        "indices": selected_indices,
        "summary": _summary(rows),
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _load_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = data.get("targets") or data.get("results") or []
    if not isinstance(targets, list):
        raise ValueError(f"run has no target list: {path}")
    return data


def _targets_by_index(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for pos, target in enumerate(run.get("targets") or run.get("results") or []):
        index = target.get("benchmark_index", target.get("index", pos))
        try:
            out[int(index)] = target
        except (TypeError, ValueError):
            out[pos] = target
    return out


def _derive_priority_indices(
    *,
    teacher: dict[int, dict[str, Any]],
    student: dict[int, dict[str, Any]],
    hybrid: dict[int, dict[str, Any]],
    top_n: int,
) -> list[int]:
    scored = []
    for index, teacher_target in teacher.items():
        student_target = student.get(index)
        if not student_target:
            continue
        hybrid_target = hybrid.get(index)
        score = _miss_score(teacher_target, student_target)
        if hybrid_target:
            score += int(_metric_bool(hybrid_target, "strict_stock_solve_any") and not _metric_bool(student_target, "strict_stock_solve_any"))
            score += int(_recovery_bool(hybrid_target, "gt_reactant_in_route_pool") and not _recovery_bool(student_target, "gt_reactant_in_route_pool"))
        if score:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in scored[:top_n]]


def _miss_score(teacher_target: dict[str, Any], student_target: dict[str, Any]) -> int:
    score = int(_metric_bool(teacher_target, "strict_stock_solve_any") and not _metric_bool(student_target, "strict_stock_solve_any"))
    for key in RECOVERY_KEYS:
        score += int(_recovery_bool(teacher_target, key) and not _recovery_bool(student_target, key))
    return score


def _audit_row(
    *,
    index: int,
    runs: dict[str, dict[str, Any]],
    target_by_label: dict[str, dict[str, Any] | None],
    teacher_label: str,
    student_label: str,
    hybrid_label: str | None,
    teacher_target: dict[str, Any],
    student_target: dict[str, Any],
    hybrid_target: dict[str, Any] | None,
) -> dict[str, Any]:
    label_summaries = {}
    for label in runs:
        target = target_by_label.get(label)
        if target:
            label_summaries[label] = _target_summary(target)
    gap_class = _gap_class(
        teacher_target=teacher_target,
        student_target=student_target,
        hybrid_target=hybrid_target,
    )
    return {
        "index": index,
        "target_smiles": str(student_target.get("target_smiles") or teacher_target.get("target_smiles") or ""),
        "route_domain": str(student_target.get("route_domain") or teacher_target.get("route_domain") or ""),
        "miss_score_vs_teacher": _miss_score(teacher_target, student_target),
        "gap_class": gap_class,
        "losses_vs_teacher": _losses(teacher_target, student_target),
        "teacher_label": teacher_label,
        "student_label": student_label,
        "hybrid_label": hybrid_label,
        "configs": label_summaries,
        "student_next_action": _next_action(gap_class),
    }


def _target_summary(target: dict[str, Any]) -> dict[str, Any]:
    metrics = target.get("metrics") or {}
    recovery = target.get("route_recovery") or {}
    routes = _routes(target)
    route_summaries = [_route_summary(route, rank=idx + 1) for idx, route in enumerate(routes)]
    first_stock_rank = _first_route_rank(route_summaries, "strict_stock_solve")
    first_gt_rank = _first_recovery_route_rank(recovery, "gt_reactant_hit")
    first_exact_rank = recovery.get("exact_reaction_first_rank")
    return {
        "plan": bool(metrics.get("plan")),
        "stock": bool(metrics.get("strict_stock_solve_any")),
        "stock_first_rank": metrics.get("strict_stock_first_rank") or first_stock_rank,
        "candidate_gt": bool(recovery.get("candidate_gt_reactant_in_pool")),
        "candidate_exact": bool(recovery.get("candidate_exact_reaction_in_pool")),
        "exact_route": bool(recovery.get("exact_reaction_in_route_pool")),
        "route_gt": bool(recovery.get("gt_reactant_in_route_pool")),
        "recovery_bottleneck": recovery.get("recovery_bottleneck"),
        "n_routes": len(routes),
        "broad_reservoir": _broad_reservoir_summary(target),
        "best_routes": {
            "selected": _select_route(route_summaries, recovery),
            "first_stock": _route_by_rank(route_summaries, first_stock_rank),
            "first_gt": _route_by_rank(route_summaries, first_gt_rank),
            "first_exact": _route_by_rank(route_summaries, first_exact_rank),
        },
        "route_source_counts": dict(_route_source_counts(route_summaries)),
    }


def _routes(target: dict[str, Any]) -> list[dict[str, Any]]:
    planner = target.get("planner_output") or {}
    routes = planner.get("routes") or []
    return routes if isinstance(routes, list) else []


def _route_summary(route: dict[str, Any], *, rank: int) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    steps = route.get("steps") or []
    return {
        "rank": rank,
        "score": route.get("score"),
        "strict_stock_solve": bool(metrics.get("strict_stock_solve")),
        "terminal_reactants": list(metrics.get("terminal_reactants") or []),
        "source_counts": _normalize_source_counts(metrics.get("candidate_source_counts") or {}),
        "step_sources": [_normalize_source(step.get("source")) for step in steps if isinstance(step, dict)],
        "reaction_types": [str(step.get("reaction_type") or "") for step in steps if isinstance(step, dict)],
        "stock_closed_step_count": _stock_closed_step_count(steps),
    }


def _normalize_source_counts(counts: dict[str, Any]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for source, count in counts.items():
        try:
            value = int(count)
        except (TypeError, ValueError):
            value = 1
        out[_normalize_source(source)] += value
    return dict(out)


def _normalize_source(source: Any) -> str:
    text = str(source or "")
    if not text:
        return ""
    if text == "ChemEnzyRetroPlanner":
        return "native_chem_enzy"
    if ">>" in text or text.startswith("("):
        return "native_template"
    if text == "native_chem_enzy":
        return "native_chem_enzy"
    return text


def _stock_closed_step_count(steps: list[Any]) -> int:
    count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        stock = step.get("stock_status") or {}
        if stock and all(bool(value) for value in stock.values()):
            count += 1
    return count


def _broad_reservoir_summary(target: dict[str, Any]) -> dict[str, Any]:
    broad = (target.get("planner_output") or {}).get("broad_reservoir") or {}
    routes = broad.get("routes") or []
    if not isinstance(routes, list):
        routes = []
    stock_routes = [route for route in routes if route.get("stock_closed")]
    return {
        "enabled": bool(broad.get("enabled")),
        "native_topk": broad.get("native_topk"),
        "native_route_count": int(broad.get("native_route_count") or len(routes)),
        "stock_closed_count": len(stock_routes),
        "first_stock_route_rank": stock_routes[0].get("route_rank") if stock_routes else None,
        "first_stock_native_rank": stock_routes[0].get("native_rank") if stock_routes else None,
    }


def _route_source_counts(route_summaries: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for route in route_summaries:
        for source, count in (route.get("source_counts") or {}).items():
            counts[source] += int(count)
    return counts


def _select_route(route_summaries: list[dict[str, Any]], recovery: dict[str, Any]) -> dict[str, Any] | None:
    for rank in [
        recovery.get("exact_reaction_first_rank"),
        _first_recovery_route_rank(recovery, "gt_reactant_hit"),
        _first_route_rank(route_summaries, "strict_stock_solve"),
        1 if route_summaries else None,
    ]:
        route = _route_by_rank(route_summaries, rank)
        if route:
            return route
    return None


def _first_route_rank(route_summaries: list[dict[str, Any]], key: str) -> int | None:
    for route in route_summaries:
        if route.get(key):
            return int(route["rank"])
    return None


def _first_recovery_route_rank(recovery: dict[str, Any], key: str) -> int | None:
    for idx, item in enumerate(recovery.get("per_route") or [], start=1):
        if isinstance(item, dict) and item.get(key):
            return idx
    return None


def _route_by_rank(route_summaries: list[dict[str, Any]], rank: Any) -> dict[str, Any] | None:
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        return None
    if rank_int < 1 or rank_int > len(route_summaries):
        return None
    return route_summaries[rank_int - 1]


def _losses(teacher_target: dict[str, Any], student_target: dict[str, Any]) -> dict[str, bool]:
    losses = {
        "stock": _metric_bool(teacher_target, "strict_stock_solve_any") and not _metric_bool(student_target, "strict_stock_solve_any"),
    }
    for key in RECOVERY_KEYS:
        losses[key] = _recovery_bool(teacher_target, key) and not _recovery_bool(student_target, key)
    return losses


def _gap_class(
    *,
    teacher_target: dict[str, Any],
    student_target: dict[str, Any],
    hybrid_target: dict[str, Any] | None,
) -> str:
    teacher_route_gt = _recovery_bool(teacher_target, "gt_reactant_in_route_pool")
    hybrid_route_gt = _recovery_bool(hybrid_target or {}, "gt_reactant_in_route_pool")
    student_route_gt = _recovery_bool(student_target, "gt_reactant_in_route_pool")
    student_candidate_gt = _recovery_bool(student_target, "candidate_gt_reactant_in_pool")
    teacher_candidate_gt = _recovery_bool(teacher_target, "candidate_gt_reactant_in_pool")
    hybrid_candidate_gt = _recovery_bool(hybrid_target or {}, "candidate_gt_reactant_in_pool")
    stock_loss = _metric_bool(teacher_target, "strict_stock_solve_any") and not _metric_bool(student_target, "strict_stock_solve_any")
    broad_stock = _broad_reservoir_summary(hybrid_target or teacher_target).get("stock_closed_count", 0)
    if stock_loss and broad_stock and not student_candidate_gt and (teacher_route_gt or hybrid_route_gt):
        return "native_route_only_stock_gap"
    if not student_candidate_gt and (teacher_candidate_gt or hybrid_candidate_gt):
        return "proposal_reactant_gap"
    if student_candidate_gt and not student_route_gt and (teacher_route_gt or hybrid_route_gt):
        return "route_composition_or_order_gap"
    if stock_loss and broad_stock:
        return "native_stock_closure_gap"
    if stock_loss:
        return "stock_closure_gap"
    if _recovery_bool(teacher_target, "exact_reaction_in_route_pool") and not _recovery_bool(student_target, "exact_reaction_in_route_pool"):
        return "exact_reaction_order_gap"
    return "no_teacher_student_loss"


def _next_action(gap_class: str) -> str:
    if gap_class == "native_route_only_stock_gap":
        return "distill native route-composition steps or add a non-eval native-route replay source; search/budget knobs alone are unlikely to close it"
    if gap_class == "proposal_reactant_gap":
        return "audit proposal providers for missing GT reactant and add source/budget replay on non-eval rows"
    if gap_class == "route_composition_or_order_gap":
        return "train pairwise route/action ordering to keep the candidate sequence that reaches GT/stock"
    if gap_class == "native_stock_closure_gap":
        return "train stock-closure rerank/replay from native positives without using eval-only rows"
    if gap_class == "stock_closure_gap":
        return "increase stock-aware retention and leaf expansion around the first stock-closing branch"
    if gap_class == "exact_reaction_order_gap":
        return "train exact-reaction rerank against teacher-positive candidate positions"
    return "no targeted student-only action from this row"


def _metric_bool(target: dict[str, Any], key: str) -> bool:
    return bool((target.get("metrics") or {}).get(key))


def _recovery_bool(target: dict[str, Any], key: str) -> bool:
    return bool((target.get("route_recovery") or {}).get(key))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(row["gap_class"] for row in rows)
    loss_counts: Counter[str] = Counter()
    broad_stock_rows = 0
    for row in rows:
        for key, value in (row.get("losses_vs_teacher") or {}).items():
            if value:
                loss_counts[key] += 1
        for config in (row.get("configs") or {}).values():
            broad = config.get("broad_reservoir") or {}
            if broad.get("stock_closed_count"):
                broad_stock_rows += 1
                break
    return {
        "n": len(rows),
        "gap_class_counts": dict(classes),
        "loss_counts": dict(loss_counts),
        "rows_with_broad_stock_reservoir": broad_stock_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Student Route Composition Gap Audit",
        "",
        f"Teacher: `{report['teacher_label']}`",
        f"Student: `{report['student_label']}`",
        f"Hybrid: `{report.get('hybrid_label') or ''}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Priority Rows",
        "",
        "| idx | class | losses | B stock/exact/GT | C stock/exact/GT | D stock/exact/GT | C bottleneck | action | target |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("rows") or []:
        configs = row.get("configs") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("index")),
                    str(row.get("gap_class")),
                    _loss_text(row.get("losses_vs_teacher") or {}),
                    _metric_triplet(configs.get("B") or configs.get(report["teacher_label"]) or {}),
                    _metric_triplet(configs.get("C") or configs.get(report["student_label"]) or {}),
                    _metric_triplet(configs.get("D") or configs.get(report.get("hybrid_label") or "") or {}),
                    str((configs.get(report["student_label"]) or {}).get("recovery_bottleneck") or ""),
                    str(row.get("student_next_action") or ""),
                    str(row.get("target_smiles") or "")[:80],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Route Evidence", ""])
    for row in report.get("rows") or []:
        lines.append(f"### Index {row.get('index')} `{row.get('gap_class')}`")
        lines.append("")
        for label, config in (row.get("configs") or {}).items():
            selected = ((config.get("best_routes") or {}).get("selected") or {})
            broad = config.get("broad_reservoir") or {}
            lines.append(
                "- `{label}` stock={stock} exact={exact} routeGT={gt} routes={routes} "
                "broadStock={broad_stock} selectedRank={rank} sources={sources} terminals={terms}".format(
                    label=label,
                    stock=config.get("stock"),
                    exact=config.get("exact_route"),
                    gt=config.get("route_gt"),
                    routes=config.get("n_routes"),
                    broad_stock=broad.get("stock_closed_count"),
                    rank=selected.get("rank"),
                    sources=selected.get("source_counts"),
                    terms=selected.get("terminal_reactants"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _metric_triplet(config: dict[str, Any]) -> str:
    if not config:
        return ""
    return f"{int(bool(config.get('stock')))}/{int(bool(config.get('exact_route')))}/{int(bool(config.get('route_gt')))}"


def _loss_text(losses: dict[str, bool]) -> str:
    names = [key for key, value in losses.items() if value]
    return ",".join(names) if names else "none"


def _parse_run_specs(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        out[label] = Path(path)
    return out


def _parse_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit student-only route composition gaps")
    ap.add_argument("--run", action="append", required=True, help="LABEL=run.json; repeat for B/C/D")
    ap.add_argument("--teacher-label", default="B")
    ap.add_argument("--student-label", default="C")
    ap.add_argument("--hybrid-label", default="D")
    ap.add_argument("--indices", default=None, help="Comma-separated benchmark indices; defaults to top misses")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    report = analyze_student_route_composition_gaps(
        runs=_parse_run_specs(args.run),
        teacher_label=args.teacher_label,
        student_label=args.student_label,
        hybrid_label=args.hybrid_label or None,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        indices=_parse_indices(args.indices),
        top_n=args.top_n,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
