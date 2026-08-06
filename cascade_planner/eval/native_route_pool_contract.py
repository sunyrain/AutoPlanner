"""Shared route-pool parsing and product-audit reporting contracts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
)


def routes_for_target(target: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized route dictionaries from supported run projections."""
    routes = target.get("routes")
    if isinstance(routes, list):
        return [route for route in routes if isinstance(route, dict)]
    planner_routes = (target.get("planner_output") or {}).get("routes")
    if isinstance(planner_routes, list):
        return [route for route in planner_routes if isinstance(route, dict)]
    cascade_programs = (target.get("cascade_search") or {}).get("result_programs")
    if isinstance(cascade_programs, list):
        return [
            {
                **program,
                "steps": list(program.get("steps") or program.get("route_steps") or []),
            }
            for raw in cascade_programs
            if isinstance(raw, dict)
            for program in [dict(raw)]
        ]
    return []


def cap_native_run_for_audit(
    run: dict[str, Any],
    *,
    top_k: int | None,
) -> dict[str, Any]:
    """Return a shallow run projection capped to the first ``top_k`` routes."""
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


def summarize_product_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Select the stable run-level product-audit summary fields."""
    return {
        "n_targets": audit.get("n_targets"),
        "strict_stock_solve_targets": audit.get("strict_stock_solve_targets"),
        "strict_stock_solve_rate": audit.get("strict_stock_solve_rate"),
        "triage_signal_targets": audit.get("triage_signal_targets"),
        "triage_signal_rate": audit.get("triage_signal_rate"),
        "top3_triage_signal_targets": audit.get("top3_triage_signal_targets"),
        "top3_triage_signal_rate": audit.get("top3_triage_signal_rate"),
        "autonomous_route_candidate_targets": audit.get(
            "autonomous_route_candidate_targets"
        ),
        "autonomous_route_candidate_rate": audit.get(
            "autonomous_route_candidate_rate"
        ),
        "route_class_counts": audit.get("route_class_counts"),
        "route_issue_counts": audit.get("route_issue_counts"),
    }


def product_audit_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Calculate candidate-minus-baseline deltas for stable audit rates."""
    keys = [
        "strict_stock_solve_rate",
        "triage_signal_rate",
        "top3_triage_signal_rate",
        "autonomous_route_candidate_rate",
    ]
    out = {}
    for key in keys:
        left = baseline.get(key)
        right = candidate.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[key] = round(float(right) - float(left), 6)
    return out


def ranked_product_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    """Summarize risk and usability rates at stable top-k cutoffs."""
    targets = audit.get("targets") or []
    out: dict[str, Any] = {"n_targets": len(targets)}
    triage_classes = {
        "triage_semisynthesis",
        "triage_late_stage",
        "triage_fragment",
    }
    for top_k in (1, 3, 5):
        product_usable = 0
        artifact = 0
        trivial = 0
        generic = 0
        class_counts: Counter[str] = Counter()
        for target in targets:
            routes = sorted(
                target.get("routes") or [],
                key=lambda row: int(row.get("rank") or 10**9),
            )
            top = routes[:top_k]
            product_usable += int(
                any(route.get("route_class") in triage_classes for route in top)
            )
            artifact += int(
                any(route.get("route_class") == "reject_artifact" for route in top)
            )
            trivial += int(
                any(
                    "trivial_stock_closure" in (route.get("issues") or [])
                    for route in top
                )
            )
            generic += int(
                any(
                    "generic_reaction_sequence" in (route.get("issues") or [])
                    for route in top
                )
            )
            class_counts.update(
                str(route.get("route_class") or "unknown") for route in top
            )
        denom = max(len(targets), 1)
        out[f"top{top_k}_product_usable_rate"] = round(product_usable / denom, 6)
        out[f"top{top_k}_artifact_rate"] = round(artifact / denom, 6)
        out[f"top{top_k}_trivial_stock_closure_rate"] = round(trivial / denom, 6)
        out[f"top{top_k}_generic_route_rate"] = round(generic / denom, 6)
        out[f"top{top_k}_route_class_counts"] = dict(sorted(class_counts.items()))
    return out


def ground_truth_recovery(
    run: dict[str, Any],
    benchmark_rows: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Report reaction and reactant recovery against optional benchmark rows."""
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
    n_targets = 0
    for target in run.get("targets") or []:
        target_smiles = str(target.get("target_smiles") or "")
        bench = benchmark_by_target.get(target_smiles)
        if bench is None:
            canonical_target = canonical_smiles(target_smiles) or target_smiles
            bench = next(
                (
                    row
                    for row in benchmark_rows
                    if (
                        canonical_smiles(str(row.get("target_smiles") or ""))
                        or str(row.get("target_smiles") or "")
                    )
                    == canonical_target
                ),
                None,
            )
        if bench is None:
            continue
        n_targets += 1
        gt_rxns = _gt_reactions(bench)
        gt_reactants = _gt_reactants(bench)
        route_rxns = set()
        route_reactants = set()
        for route in routes_for_target(target):
            for step in route.get("steps") or []:
                rxn = str(
                    step.get("rxn_smiles") or step.get("reaction_smiles") or ""
                )
                if rxn:
                    route_rxns.add(canonical_reaction(rxn) or rxn)
                    route_reactants.update(_reaction_reactants(rxn))
                reactants = list(step.get("reactant_smiles") or [])
                if not reactants:
                    reactants = [
                        smiles
                        for smiles in [
                            step.get("main_reactant"),
                            *(step.get("aux_reactants") or []),
                        ]
                        if smiles
                    ]
                route_reactants.update(
                    canonical_smiles(str(smiles)) or str(smiles)
                    for smiles in reactants
                )
        exact += int(bool(gt_rxns and gt_rxns.issubset(route_rxns)))
        partial += int(bool(gt_rxns and route_rxns & gt_rxns))
        reactant += int(bool(gt_reactants and route_reactants & gt_reactants))
    return {
        "n_targets_with_gt": n_targets,
        "exact_gt_route_recovered_rate": round(exact / max(n_targets, 1), 6),
        "partial_gt_step_overlap_rate": round(partial / max(n_targets, 1), 6),
        "gt_reactant_in_route_pool_rate": round(
            reactant / max(n_targets, 1),
            6,
        ),
    }


def read_json_rows(path: Path | None) -> list[dict[str, Any]] | None:
    """Read rows from supported list or keyed-list JSON documents."""
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
    return {
        canonical_smiles(part.strip()) or part.strip()
        for part in left.split(".")
        if part.strip()
    }


__all__ = [
    "cap_native_run_for_audit",
    "ground_truth_recovery",
    "product_audit_delta",
    "ranked_product_metrics",
    "read_json_rows",
    "routes_for_target",
    "summarize_product_audit",
]
