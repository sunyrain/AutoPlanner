"""Apply a v4-trained cascade product-value model to native route pools."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import (
    LoadedV4CascadeProductValue,
    route_record_from_native_route,
    route_record_from_planner_route,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit, product_audit_guard_key


def rerank_native_routes_with_v4_value(
    *,
    native_pool: Path,
    model: Path,
    output: Path,
    report: Path,
    benchmark: Path | None = None,
    device: str = "cpu",
    top_k: int | None = None,
    ranking_mode: str = "model_value",
    include_audit_features: bool = True,
) -> dict[str, Any]:
    if ranking_mode == "audit_guarded" and not include_audit_features:
        raise ValueError("audit_guarded ranking requires include_audit_features=True")
    run = json.loads(Path(native_pool).read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    scorer = LoadedV4CascadeProductValue(model, device=device)
    scoring_run = _cap_native_run_for_audit(run, top_k=top_k)
    scoring_audit = build_product_route_feasibility_audit(scoring_run, benchmark_rows=benchmark_rows)
    targets_out = []
    route_rows = []
    for target_index, target in enumerate(scoring_run.get("targets") or []):
        audit_target = (scoring_audit.get("targets") or [{}])[target_index] if target_index < len(scoring_audit.get("targets") or []) else {}
        audit_by_rank = {
            int(row.get("rank") or 0): row
            for row in audit_target.get("routes") or []
            if row.get("rank") is not None
        }
        target_smiles = str(target.get("target_smiles") or "")
        target_id = str(target.get("cascade_id") or target.get("target_id") or target.get("index") or target_index)
        scored_routes = []
        for native_rank, route in enumerate(_routes_for_target(target)):
            feature_row = _route_feature_row(
                route,
                target_smiles=target_smiles,
                target_id=target_id,
                native_rank=native_rank,
                dataset=str(run.get("metadata", {}).get("dataset") or "native_route_pool"),
            )
            audit_row = audit_by_rank.get(native_rank + 1, {})
            if include_audit_features:
                feature_row["product_audit"] = {
                    "route_class": audit_row.get("route_class"),
                    "issues": audit_row.get("issues") or [],
                    "tags": audit_row.get("tags") or [],
                    "route_plausibility": audit_row.get("route_plausibility") or {},
                }
            prediction = scorer.predict(feature_row)
            payload = dict(route)
            payload["native_rank"] = native_rank
            payload["v4_cascade_product_value"] = prediction.route_value
            payload["v4_cascade_product_prediction"] = prediction.to_dict()
            payload["v4_route_feature_id"] = feature_row["route_id"]
            payload["v4_product_audit_features"] = feature_row.get("product_audit") or {}
            scored_routes.append(payload)
            route_rows.append(
                {
                    "target_index": target_index,
                    "target_id": target_id,
                    "target_smiles": target_smiles,
                    "native_rank": native_rank,
                    "route_id": feature_row["route_id"],
                    "v4_cascade_product_value": prediction.route_value,
                    "confidence": prediction.confidence,
                    "stock_closed": feature_row.get("stock_closed"),
                    "n_steps": len(feature_row.get("steps") or []),
                }
            )
        reranked = sorted(scored_routes, key=lambda route: _ranking_key(route, ranking_mode=ranking_mode))
        target_out = dict(target)
        if isinstance((target.get("planner_output") or {}).get("routes"), list):
            planner = dict(target.get("planner_output") or {})
            planner["routes"] = reranked
            planner["n_results"] = len(reranked)
            target_out["planner_output"] = planner
        target_out["routes"] = reranked
        target_out["route_count"] = len(reranked)
        target_out["v4_rerank_metadata"] = {
            "model": str(model),
            "ranking_mode": ranking_mode,
            "top_k_input": top_k,
            "include_audit_features": include_audit_features,
            "native_route_count": len(_routes_for_target(target)),
            "reranked_route_count": len(reranked),
        }
        targets_out.append(target_out)

    output_run = {
        "metadata": {
            **(run.get("metadata") or {}),
            "reranker": "v4_cascade_product_value",
            "reranker_model": str(model),
            "source_native_pool": str(native_pool),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": _run_summary(targets_out),
        "targets": targets_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_run, indent=2, ensure_ascii=False), encoding="utf-8")

    native_audit_run = _cap_native_run_for_audit(run, top_k=top_k)
    native_audit = build_product_route_feasibility_audit(native_audit_run, benchmark_rows=benchmark_rows)
    learned_audit = build_product_route_feasibility_audit(output_run, benchmark_rows=benchmark_rows)
    result = {
        "schema_version": "v4_native_route_rerank_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_pool": str(native_pool),
            "model": str(model),
            "output": str(output),
            "benchmark": str(benchmark) if benchmark else None,
            "top_k": top_k,
            "device": device,
            "ranking_mode": ranking_mode,
            "include_audit_features": include_audit_features,
        },
        "summary": output_run["summary"],
        "native_audit_summary": _audit_summary(native_audit),
        "learned_audit_summary": _audit_summary(learned_audit),
        "native_ranked_product_metrics": _ranked_product_metrics(native_audit),
        "learned_ranked_product_metrics": _ranked_product_metrics(learned_audit),
        "native_gt_recovery": _gt_recovery(native_audit_run, benchmark_rows),
        "learned_gt_recovery": _gt_recovery(output_run, benchmark_rows),
        "delta_learned_minus_native": _audit_delta(native_audit, learned_audit),
        "route_value_summary": _route_value_summary(route_rows),
        "route_rows": route_rows[:200],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = report.with_suffix(".md")
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _ranking_key(route: dict[str, Any], *, ranking_mode: str) -> tuple[Any, ...]:
    value = -float(route.get("v4_cascade_product_value") or 0.0)
    native_rank = int(route.get("native_rank") or 10**9)
    if ranking_mode == "model_value":
        return (value, native_rank)
    if ranking_mode == "audit_guarded":
        audit = route.get("v4_product_audit_features") or {}
        # Keep the product-audit risk constraint as a safety guard, and let the
        # learned model decide only among routes in the same route/risk bucket.
        return (*product_audit_guard_key(audit), value, native_rank)
    raise ValueError(f"unknown ranking_mode: {ranking_mode}")


def _run_summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(targets)
    solved = sum(1 for target in targets if target.get("routes"))
    route_counts = [len(target.get("routes") or []) for target in targets]
    return {
        "targets": n,
        "plan_rate": round(solved / max(n, 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(n, 1), 6),
    }


def _routes_for_target(target: dict[str, Any]) -> list[dict[str, Any]]:
    routes = target.get("routes")
    if isinstance(routes, list):
        return [route for route in routes if isinstance(route, dict)]
    planner_routes = (target.get("planner_output") or {}).get("routes")
    if isinstance(planner_routes, list):
        return [route for route in planner_routes if isinstance(route, dict)]
    return []


def _route_feature_row(
    route: dict[str, Any],
    *,
    target_smiles: str,
    target_id: str | None,
    native_rank: int,
    dataset: str,
) -> dict[str, Any]:
    steps = route.get("steps") or []
    if steps and any(isinstance(step, dict) and ("product_smiles" in step or "reactant_smiles" in step or "rxn_smiles" in step) for step in steps):
        return route_record_from_native_route(
            route,
            target_smiles=target_smiles,
            target_id=target_id,
            native_rank=native_rank,
            dataset=dataset,
        )
    return route_record_from_planner_route(
        route,
        target_smiles=target_smiles,
        target_id=target_id,
        native_rank=native_rank,
        dataset=dataset,
    )


def _cap_native_run_for_audit(run: dict[str, Any], *, top_k: int | None) -> dict[str, Any]:
    if top_k is None or top_k <= 0:
        return run
    capped = dict(run)
    targets = []
    for target in run.get("targets") or []:
        payload = dict(target)
        if isinstance(target.get("routes"), list):
            payload["routes"] = list(target.get("routes") or [])[: int(top_k)]
            payload["route_count"] = len(payload["routes"])
        elif isinstance((target.get("planner_output") or {}).get("routes"), list):
            planner = dict(target.get("planner_output") or {})
            planner["routes"] = list(planner.get("routes") or [])[: int(top_k)]
            planner["n_results"] = len(planner["routes"])
            payload["planner_output"] = planner
        targets.append(payload)
    capped["targets"] = targets
    return capped


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_targets": audit.get("n_targets"),
        "strict_stock_solve_targets": audit.get("strict_stock_solve_targets"),
        "strict_stock_solve_rate": audit.get("strict_stock_solve_rate"),
        "triage_signal_targets": audit.get("triage_signal_targets"),
        "triage_signal_rate": audit.get("triage_signal_rate"),
        "top3_triage_signal_targets": audit.get("top3_triage_signal_targets"),
        "top3_triage_signal_rate": audit.get("top3_triage_signal_rate"),
        "autonomous_route_candidate_targets": audit.get("autonomous_route_candidate_targets"),
        "autonomous_route_candidate_rate": audit.get("autonomous_route_candidate_rate"),
        "route_class_counts": audit.get("route_class_counts"),
        "route_issue_counts": audit.get("route_issue_counts"),
    }


def _audit_delta(native: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strict_stock_solve_rate",
        "triage_signal_rate",
        "top3_triage_signal_rate",
        "autonomous_route_candidate_rate",
    ]
    out = {}
    for key in keys:
        left = native.get(key)
        right = learned.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[key] = round(float(right) - float(left), 6)
    return out


def _ranked_product_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    targets = audit.get("targets") or []
    out: dict[str, Any] = {"n_targets": len(targets)}
    triage_classes = {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}
    for k in (1, 3, 5):
        product_usable = 0
        artifact = 0
        trivial = 0
        generic = 0
        class_counts: Counter[str] = Counter()
        for target in targets:
            routes = sorted(target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))
            top = routes[:k]
            product_usable += int(any(route.get("route_class") in triage_classes for route in top))
            artifact += int(any(route.get("route_class") == "reject_artifact" for route in top))
            trivial += int(any("trivial_stock_closure" in (route.get("issues") or []) for route in top))
            generic += int(any("generic_reaction_sequence" in (route.get("issues") or []) for route in top))
            class_counts.update(str(route.get("route_class") or "unknown") for route in top)
        denom = max(len(targets), 1)
        out[f"top{k}_product_usable_rate"] = round(product_usable / denom, 6)
        out[f"top{k}_artifact_rate"] = round(artifact / denom, 6)
        out[f"top{k}_trivial_stock_closure_rate"] = round(trivial / denom, 6)
        out[f"top{k}_generic_route_rate"] = round(generic / denom, 6)
        out[f"top{k}_route_class_counts"] = dict(sorted(class_counts.items()))
    return out


def _gt_recovery(run: dict[str, Any], benchmark_rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not benchmark_rows:
        return None
    benchmark_by_target = {
        str(row.get("target_smiles") or ""): row
        for row in benchmark_rows
        if row.get("target_smiles")
    }
    exact = 0
    partial = 0
    reactant = 0
    n = 0
    for target in run.get("targets") or []:
        target_smiles = str(target.get("target_smiles") or "")
        bench = benchmark_by_target.get(target_smiles)
        if bench is None:
            canonical_target = canonical_smiles(target_smiles) or target_smiles
            bench = next(
                (
                    row
                    for row in benchmark_rows
                    if (canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")) == canonical_target
                ),
                None,
            )
        if bench is None:
            continue
        n += 1
        gt_rxns = _gt_reactions(bench)
        gt_reactants = _gt_reactants(bench)
        route_rxns = set()
        route_reactants = set()
        for route in _routes_for_target(target):
            for step in route.get("steps") or []:
                rxn = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
                if rxn:
                    route_rxns.add(canonical_reaction(rxn) or rxn)
                    route_reactants.update(_reaction_reactants(rxn))
                reactants = list(step.get("reactant_smiles") or [])
                if not reactants:
                    reactants = [
                        smi
                        for smi in [step.get("main_reactant"), *(step.get("aux_reactants") or [])]
                        if smi
                    ]
                route_reactants.update(canonical_smiles(str(smi)) or str(smi) for smi in reactants)
        exact += int(bool(gt_rxns and gt_rxns.issubset(route_rxns)))
        partial += int(bool(gt_rxns and route_rxns & gt_rxns))
        reactant += int(bool(gt_reactants and route_reactants & gt_reactants))
    return {
        "n_targets_with_gt": n,
        "exact_gt_route_recovered_rate": round(exact / max(n, 1), 6),
        "partial_gt_step_overlap_rate": round(partial / max(n, 1), 6),
        "gt_reactant_in_route_pool_rate": round(reactant / max(n, 1), 6),
    }


def _gt_reactions(row: dict[str, Any]) -> set[str]:
    out = set()
    for step in row.get("gt_route") or []:
        rxn = step.get("rxn_smiles")
        if rxn:
            out.add(canonical_reaction(rxn) or rxn)
    return out


def _gt_reactants(row: dict[str, Any]) -> set[str]:
    out = set()
    for step in row.get("gt_route") or []:
        out.update(_reaction_reactants(step.get("rxn_smiles")))
    return out


def _reaction_reactants(rxn_smiles: Any) -> set[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return set()
    left, _ = text.split(">>", 1)
    return {canonical_smiles(part.strip()) or part.strip() for part in left.split(".") if part.strip()}


def _route_value_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("v4_cascade_product_value") or 0.0) for row in rows]
    if not values:
        return {"routes": 0}
    return {
        "routes": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "top_native_rank_counts": dict(Counter(row.get("native_rank") for row in rows if row.get("native_rank") in {0, 1, 2})),
    }


def _read_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    return None


def _markdown(report: dict[str, Any]) -> str:
    native = report.get("native_audit_summary") or {}
    learned = report.get("learned_audit_summary") or {}
    delta = report.get("delta_learned_minus_native") or {}
    lines = [
        "# v4 Native Route Rerank Report",
        "",
        "## Summary",
        "",
        f"- Targets: `{report['summary']['targets']}`",
        f"- Routes: `{report['summary']['total_routes']}`",
        f"- Native top3 triage rate: `{native.get('top3_triage_signal_rate')}`",
        f"- Learned top3 triage rate: `{learned.get('top3_triage_signal_rate')}`",
        f"- Delta top3 triage rate: `{delta.get('top3_triage_signal_rate')}`",
        f"- Native stock rate: `{native.get('strict_stock_solve_rate')}`",
        f"- Learned stock rate: `{learned.get('strict_stock_solve_rate')}`",
        "",
        "## Route Value",
        "",
        "```json",
        json.dumps(report.get("route_value_summary") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Ranked Product Metrics",
        "",
        "```json",
        json.dumps(
            {
                "native": report.get("native_ranked_product_metrics"),
                "learned": report.get("learned_ranked_product_metrics"),
                "native_gt_recovery": report.get("native_gt_recovery"),
                "learned_gt_recovery": report.get("learned_gt_recovery"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerank native ChemEnzy routes with a v4 cascade product-value model")
    ap.add_argument("--native-pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--benchmark")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--ranking-mode", default="model_value", choices=["model_value", "audit_guarded"])
    ap.add_argument(
        "--include-audit-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject rule-audit features into the model input and audit-guarded ordering",
    )
    args = ap.parse_args()
    result = rerank_native_routes_with_v4_value(
        native_pool=Path(args.native_pool),
        model=Path(args.model),
        output=Path(args.output),
        report=Path(args.report),
        benchmark=Path(args.benchmark) if args.benchmark else None,
        device=args.device,
        top_k=args.top_k,
        ranking_mode=args.ranking_mode,
        include_audit_features=args.include_audit_features,
    )
    print(json.dumps(result["delta_learned_minus_native"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
