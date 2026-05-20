"""Product-oriented route feasibility audit.

This audit intentionally does not treat benchmark GT recovery or strict stock
closure as sufficient product success. It reviews exported routes for practical
triage value: late-stage derivatization, semisynthesis hints, fragment coupling
hints, and obvious stock-closure artifacts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.baselines.route_contract import RouteCandidate, RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_route_plausibility

RDLogger.DisableLog("rdApp.*")


NATURAL_STATINS = {"lovastatin", "simvastatin", "pravastatin", "mevastatin"}
SYNTHETIC_STATINS = {"atorvastatin", "fluvastatin", "pitavastatin", "rosuvastatin", "cerivastatin"}

LATE_STAGE_CLASSES = {"hydrolysis", "esterification", "acylation", "deprotection"}
GENERIC_OR_RISKY_CLASSES = {"other", "unknown", "racemization", "isomerization"}

ROUTE_CLASS_ORDER = {
    "triage_semisynthesis": 0,
    "triage_late_stage": 1,
    "triage_fragment": 2,
    "needs_chemist_review": 3,
    "weak_hint": 4,
    "reject_artifact": 5,
}

TERMINAL_CARRIER_REAGENT_PATTERNS = {
    "wittig_phosphorane": "[#6]=[#15]([c])([c])[c]",
    "hwe_phosphonate": "[#6][P](=O)(O[#6])O[#6]",
}

SEVERE_PRODUCT_ISSUES = {
    "unfilled_route",
    "atom_balance_violation",
    "product_mismatch",
    "self_loop",
    "racemization_artifact",
    "large_unexplained_atom_gain",
    "large_unexplained_heavy_atom_gain",
    "large_unexplained_carbon_gain",
    "large_unexplained_hetero_atom_gain",
    "unexplained_new_element_source",
    "invalid_product_smiles",
    "invalid_or_missing_reactants",
    "trivial_stock_closure",
}
INCOMPLETE_ROUTE_ISSUES = {"open_stock", "not_route_solved"}


def product_audit_risk_order(row: dict[str, Any]) -> int:
    """Risk bucket used only as a guard before learned tie-breakers."""
    issues = {str(issue) for issue in row.get("issues") or []}
    condition_risk = str(((row.get("condition_audit") or {}).get("route_risk") or "ok")).lower()
    condition_order = {"high": 25, "warn": 10, "ok": 0}.get(condition_risk, 0)
    if issues & SEVERE_PRODUCT_ISSUES:
        return 40
    if "generic_reaction_sequence" in issues:
        return max(30, condition_order)
    if issues & INCOMPLETE_ROUTE_ISSUES:
        return max(20, condition_order)
    if issues:
        return max(10, condition_order)
    return condition_order


def product_audit_guard_key(row: dict[str, Any]) -> tuple[int, int]:
    route_class = str(row.get("route_class") or "")
    return (ROUTE_CLASS_ORDER.get(route_class, 99), product_audit_risk_order(row))


def build_product_route_feasibility_audit(
    run: dict[str, Any],
    *,
    benchmark_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    targets = _merge_benchmark_metadata(list(run.get("targets") or []), benchmark_rows or [])
    target_rows: list[dict[str, Any]] = []
    route_class_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()

    for target in targets:
        row = _audit_target(target)
        target_rows.append(row)
        verdict_counts[row["target_verdict"]] += 1
        for route in row["routes"]:
            route_class_counts[route["route_class"]] += 1
            for issue in route["issues"]:
                issue_counts[issue] += 1

    n_targets = len(target_rows)
    triage_targets = sum(1 for row in target_rows if row["triage_signal_any"])
    autonomous_targets = sum(1 for row in target_rows if row["autonomous_route_candidate_any"])
    top3_triage_targets = sum(1 for row in target_rows if row["top3_triage_signal_any"])
    stock_targets = sum(1 for row in target_rows if row["strict_stock_solve_any"])

    return {
        "schema_version": "product_route_feasibility_audit.v1",
        "n_targets": n_targets,
        "strict_stock_solve_targets": stock_targets,
        "strict_stock_solve_rate": _rate(stock_targets, n_targets),
        "triage_signal_targets": triage_targets,
        "triage_signal_rate": _rate(triage_targets, n_targets),
        "top3_triage_signal_targets": top3_triage_targets,
        "top3_triage_signal_rate": _rate(top3_triage_targets, n_targets),
        "autonomous_route_candidate_targets": autonomous_targets,
        "autonomous_route_candidate_rate": _rate(autonomous_targets, n_targets),
        "target_verdict_counts": dict(sorted(verdict_counts.items())),
        "route_class_counts": dict(sorted(route_class_counts.items())),
        "route_issue_counts": dict(sorted(issue_counts.items())),
        "targets": target_rows,
        "interpretation": {
            "triage_signal": "A route contains a potentially useful late-stage, semisynthetic, or fragment-coupling idea for chemist review.",
            "autonomous_route_candidate": "A stricter full-route candidate. For complex products this requires more than an advanced intermediate or trivial stock closure.",
            "strict_stock_solve": "Reported separately because stock closure can be chemically misleading.",
        },
    }


def write_product_route_feasibility_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Product Route Feasibility Audit",
        "",
        "## Summary",
        "",
        f"- Targets: `{audit.get('n_targets')}`",
        f"- Strict stock-solved targets: `{audit.get('strict_stock_solve_targets')}` (`{_fmt(audit.get('strict_stock_solve_rate'))}`)",
        f"- Targets with chemist-triage signal: `{audit.get('triage_signal_targets')}` (`{_fmt(audit.get('triage_signal_rate'))}`)",
        f"- Targets with top-3 triage signal: `{audit.get('top3_triage_signal_targets')}` (`{_fmt(audit.get('top3_triage_signal_rate'))}`)",
        f"- Autonomous full-route candidates: `{audit.get('autonomous_route_candidate_targets')}` (`{_fmt(audit.get('autonomous_route_candidate_rate'))}`)",
        "",
        "## Target Verdicts",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    for verdict, count in (audit.get("target_verdict_counts") or {}).items():
        lines.append(f"| `{verdict}` | {count} |")

    lines.extend(["", "## Route Classes", "", "| Class | Routes |", "| --- | ---: |"])
    for klass, count in (audit.get("route_class_counts") or {}).items():
        lines.append(f"| `{klass}` | {count} |")

    lines.extend(["", "## Issues", "", "| Issue | Routes |", "| --- | ---: |"])
    for issue, count in (audit.get("route_issue_counts") or {}).items():
        lines.append(f"| `{issue}` | {count} |")

    lines.extend(
        [
            "",
            "## Targets",
            "",
            "| idx | target | family | stock | best class | verdict | best tags | best issues |",
            "| ---: | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in audit.get("targets") or []:
        best = row.get("best_route") or {}
        tags = ", ".join(f"`{tag}`" for tag in best.get("tags") or [])
        issues = ", ".join(f"`{issue}`" for issue in best.get("issues") or [])
        lines.append(
            "| {idx} | `{target}` | `{family}` | {stock} | `{klass}` | `{verdict}` | {tags} | {issues} |".format(
                idx=row.get("index"),
                target=row.get("target_id") or "",
                family=row.get("product_family"),
                stock=int(bool(row.get("strict_stock_solve_any"))),
                klass=best.get("route_class"),
                verdict=row.get("target_verdict"),
                tags=tags,
                issues=issues,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_target(target: dict[str, Any]) -> dict[str, Any]:
    target_id = str(target.get("cascade_id") or target.get("target_id") or target.get("index") or "")
    target_smiles = str(target.get("target_smiles") or "")
    product_family = _product_family(target_id)
    product_props = _mol_props(target_smiles)
    routes = []
    for rank, route in enumerate(_routes_for_target(target), start=1):
        routes.append(_audit_route(route, rank=rank, product_family=product_family, product_props=product_props))
    routes.sort(key=_route_sort_key)
    best = routes[0] if routes else {}
    ranked_by_export = sorted(routes, key=lambda row: int(row.get("rank") or 10**9))

    triage_signal_any = any(_is_triage_signal(route) for route in routes)
    top3_triage_signal_any = any(_is_triage_signal(route) for route in ranked_by_export[:3])
    autonomous_any = any(route.get("autonomous_route_candidate") for route in routes)
    strict_stock_any = bool((target.get("metrics") or {}).get("strict_stock_solve_any"))
    if not strict_stock_any and target.get("routes") is not None:
        strict_stock_any = any(route.get("stock_closed") for route in routes)
    if autonomous_any:
        verdict = "autonomous_candidate"
    elif any(route.get("route_class") == "triage_semisynthesis" for route in routes):
        verdict = "semisynthesis_triage"
    elif triage_signal_any:
        verdict = "late_stage_or_fragment_triage"
    else:
        verdict = "not_product_ready"

    return {
        "index": target.get("index"),
        "target_id": target_id,
        "target_smiles": target_smiles,
        "product_family": product_family,
        "target_heavy_atoms": product_props.get("heavy_atoms"),
        "strict_stock_solve_any": strict_stock_any,
        "triage_signal_any": triage_signal_any,
        "top3_triage_signal_any": top3_triage_signal_any,
        "autonomous_route_candidate_any": autonomous_any,
        "target_verdict": verdict,
        "best_route": best,
        "routes": routes,
    }


def _routes_for_target(target: dict[str, Any]) -> list[dict[str, Any]]:
    planner_routes = (target.get("planner_output") or {}).get("routes")
    if isinstance(planner_routes, list):
        return [route for route in planner_routes if isinstance(route, dict)]
    native_routes = target.get("routes")
    if isinstance(native_routes, list):
        return [_convert_native_route(route) for route in native_routes if isinstance(route, dict)]
    return []


def _convert_native_route(route: dict[str, Any]) -> dict[str, Any]:
    steps = [_convert_native_step(step, idx) for idx, step in enumerate(route.get("steps") or []) if isinstance(step, dict)]
    terminal_reactants = _native_terminal_reactants(steps)
    stock_map: dict[str, Any] = {}
    for step in steps:
        stock_map.update(step.get("stock_status") or {})
    strict_stock = bool(steps) and all(bool(stock_map.get(smi)) for smi in terminal_reactants)
    solved = bool(route.get("solved"))
    return {
        "score": route.get("score"),
        "steps": steps,
        "metrics": {
            "strict_stock_solve": strict_stock,
            "route_solved": solved,
            "filled_route": bool(steps),
            "terminal_reactants": terminal_reactants,
            "route_naturalness": {},
        },
        "quality_vector": {"stock_closed": float(strict_stock), "route_solved": float(solved)},
        "raw_backend_metadata": route.get("raw_backend_metadata") or {},
    }


def _convert_native_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    reactants = [str(smi) for smi in step.get("reactant_smiles") or []]
    rxn = str(step.get("rxn_smiles") or "")
    source = str(step.get("source_model") or "")
    return {
        "index": index,
        "product": step.get("product_smiles"),
        "main_reactant": reactants[0] if reactants else "",
        "aux_reactants": reactants[1:],
        "reaction_smiles": rxn,
        "reaction_type": "unknown",
        "source": source,
        "scores": {"retro": step.get("score"), "confidence": step.get("score")},
        "stock_status": step.get("stock_status") or {},
        "reaction_interpretation": {
            "reaction_class": _infer_reaction_class({"source": source, "reaction_smiles": rxn, "reaction_type": "unknown"}),
            "atom_change": _atom_change_from_rxn(rxn),
        },
    }


def _native_terminal_reactants(steps: list[dict[str, Any]]) -> list[str]:
    reactants: list[str] = []
    products = {str(step.get("product") or "") for step in steps if step.get("product")}
    for step in steps:
        for smi in [step.get("main_reactant"), *(step.get("aux_reactants") or [])]:
            text = str(smi or "")
            if text and text not in products and text not in reactants:
                reactants.append(text)
    if reactants:
        return reactants
    for step in steps:
        for smi in (step.get("stock_status") or {}):
            text = str(smi or "")
            if text and text not in reactants:
                reactants.append(text)
    return reactants


def _merge_benchmark_metadata(targets: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not benchmark_rows:
        return targets
    by_smiles = {str(row.get("target_smiles") or ""): row for row in benchmark_rows if row.get("target_smiles")}
    out = []
    for idx, target in enumerate(targets):
        merged = dict(target)
        bench = by_smiles.get(str(target.get("target_smiles") or ""))
        if bench:
            for key in ("cascade_id", "target_id", "route_domain", "doi"):
                if key not in merged and key in bench:
                    merged[key] = bench[key]
            merged.setdefault("index", bench.get("index", idx))
        out.append(merged)
    return out


def _audit_route(
    route: dict[str, Any],
    *,
    rank: int,
    product_family: str,
    product_props: dict[str, Any],
) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    steps = list(route.get("steps") or [])
    stock_closed = bool(metrics.get("strict_stock_solve") or (route.get("quality_vector") or {}).get("stock_closed"))
    route_solved = bool(metrics.get("route_solved", stock_closed))
    filled_route = bool(metrics.get("filled_route", bool(steps)))
    target_heavy = int(product_props.get("heavy_atoms") or 0)
    terminal_reactants = [str(smi) for smi in metrics.get("terminal_reactants") or []]
    terminal_profile = _terminal_profile(terminal_reactants, product_smiles=str(product_props.get("smiles") or ""), target_heavy=target_heavy)
    reaction_profile = _reaction_profile(steps)
    condition_audit = audit_route_condition_profile(steps)
    route_plausibility = _route_plausibility_from_steps(
        steps,
        solved=route_solved,
        route_score=route.get("score"),
    )
    issues: list[str] = []
    tags: list[str] = []

    if not filled_route:
        issues.append("unfilled_route")
    if not stock_closed:
        issues.append("open_stock")
    if not route_solved:
        issues.append("not_route_solved")

    naturalness = metrics.get("route_naturalness") or {}
    if int(naturalness.get("atom_balance_violations") or 0) > 0:
        issues.append("atom_balance_violation")
    if int(naturalness.get("product_mismatch_steps") or 0) > 0:
        issues.append("product_mismatch")
    if int(naturalness.get("self_loop_steps") or 0) > 0:
        issues.append("self_loop")

    class_counts = Counter(reaction_profile["classes"])
    generic_fraction = reaction_profile["generic_fraction"]
    if generic_fraction >= 0.75 and len(steps) >= 2:
        issues.append("generic_reaction_sequence")
    if class_counts.get("racemization", 0) >= max(2, len(steps) // 2 + 1):
        issues.append("racemization_artifact")
    if reaction_profile["large_unexplained_atom_gain"]:
        issues.append("large_unexplained_atom_gain")
    if route_plausibility.get("steps") and not route_plausibility.get("passed"):
        for reason in route_plausibility.get("reasons") or []:
            issues.append(str(reason))
    if condition_audit.get("route_risk") == "high":
        issues.append("condition_high_risk")
    elif condition_audit.get("route_risk") == "warn":
        issues.append("condition_warning")

    if stock_closed and target_heavy >= 25 and terminal_profile["all_terminals_small"]:
        issues.append("trivial_stock_closure")
    if terminal_profile["product_like_terminal"]:
        tags.append("advanced_or_product_like_terminal")
    if terminal_profile["all_terminals_small"]:
        tags.append("small_terminal_set")
    if terminal_profile["carrier_reagents"]:
        tags.append("carrier_reagent_terminal")

    if product_family == "natural_statin":
        tags.append("natural_product_family")
        if terminal_profile["large_polycyclic_terminal"]:
            tags.append("natural_core_terminal")
        if "natural_core_terminal" not in tags and not any(cls in LATE_STAGE_CLASSES for cls in reaction_profile["classes"]):
            issues.append("natural_product_core_missing")

    if any(cls in LATE_STAGE_CLASSES for cls in reaction_profile["classes"]):
        tags.append("late_stage_derivatization")
    if _has_acylating_piece(terminal_reactants):
        tags.append("acylating_piece_present")
    if _has_aryl_coupling_signal(terminal_reactants, reaction_profile["classes"]):
        tags.append("aryl_coupling_hint")
    if condition_audit.get("stepwise_required"):
        tags.append("condition_stepwise_required")

    severe = _has_severe_issue(issues)
    semisynthesis = (
        product_family == "natural_statin"
        and "natural_core_terminal" in tags
        and ("late_stage_derivatization" in tags or "acylating_piece_present" in tags)
        and not severe
    )
    late_stage = (
        "advanced_or_product_like_terminal" in tags
        and "late_stage_derivatization" in tags
        and not severe
    )
    fragment = "aryl_coupling_hint" in tags and not severe

    if severe:
        route_class = "reject_artifact"
    elif semisynthesis:
        route_class = "triage_semisynthesis"
    elif late_stage:
        route_class = "triage_late_stage"
    elif fragment:
        route_class = "triage_fragment"
    elif stock_closed:
        route_class = "needs_chemist_review"
    else:
        route_class = "weak_hint"

    autonomous = (
        stock_closed
        and route_class in {"triage_semisynthesis", "triage_late_stage"}
        and "advanced_or_product_like_terminal" not in tags
        and "open_stock" not in issues
        and not terminal_profile["all_terminals_small"]
    )

    return {
        "rank": rank,
        "route_class": route_class,
        "autonomous_route_candidate": autonomous,
        "stock_closed": stock_closed,
        "route_solved": route_solved,
        "filled_route": filled_route,
        "n_steps": len(steps),
        "route_score": route.get("score"),
        "tags": sorted(set(tags)),
        "issues": sorted(set(issues)),
        "terminal_profile": terminal_profile,
        "reaction_profile": reaction_profile,
        "condition_audit": condition_audit,
        "route_plausibility": route_plausibility,
        "step_summaries": [_step_summary(step) for step in steps[:5]],
    }


def audit_route_condition_profile(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit predicted conditions as weak, unverified route metadata.

    This does not validate reaction feasibility.  It separates condition-model
    guesses from material-sanity evidence so extreme temperatures or solvents
    can be surfaced without turning a chemically useful disconnection into an
    atom-source artifact.
    """
    step_rows = [audit_step_condition_profile(step, index=idx) for idx, step in enumerate(steps, start=1)]
    temps = [
        float(row["temperature_c"])
        for row in step_rows
        if row.get("temperature_c") is not None
    ]
    score_values = [
        float(row["condition_score"])
        for row in step_rows
        if row.get("condition_score") is not None
    ]
    issue_counts: Counter[str] = Counter()
    for row in step_rows:
        issue_counts.update(str(issue) for issue in row.get("issues") or [])
    high_steps = [row for row in step_rows if row.get("risk") == "high"]
    warn_steps = [row for row in step_rows if row.get("risk") == "warn"]
    has_low_and_high = bool(temps) and min(temps) < -20 and max(temps) > 80
    temp_span = round(max(temps) - min(temps), 3) if temps else None
    route_issues: list[str] = []
    if high_steps:
        route_issues.append("high_risk_condition_steps")
    if warn_steps:
        route_issues.append("warning_condition_steps")
    if has_low_and_high or (temp_span is not None and temp_span > 100):
        route_issues.append("not_one_pot_temperature_compatible")
    if issue_counts.get("reactive_reagent_low_temperature"):
        route_issues.append("stepwise_reactive_reagent_handling")
    route_risk = "high" if high_steps else "warn" if warn_steps or route_issues else "ok"
    return {
        "schema_version": "route_condition_audit.v1",
        "contract": (
            "condition predictions are weak per-step hypotheses; they are not "
            "validated process conditions and are used only for warning/rerank"
        ),
        "route_risk": route_risk,
        "route_issues": sorted(set(route_issues)),
        "step_count": len(step_rows),
        "predicted_condition_count": sum(1 for row in step_rows if row.get("has_condition_prediction")),
        "high_risk_step_count": len(high_steps),
        "warning_step_count": len(warn_steps),
        "temperature_min_c": round(min(temps), 3) if temps else None,
        "temperature_max_c": round(max(temps), 3) if temps else None,
        "temperature_span_c": temp_span,
        "top1_score_mean": round(sum(score_values) / len(score_values), 4) if score_values else None,
        "low_score_step_count": int(issue_counts.get("low_condition_score") or 0),
        "issue_counts": dict(sorted(issue_counts.items())),
        "stepwise_required": bool(route_issues),
        "steps": step_rows,
    }


def audit_step_condition_profile(step: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    condition = _top_condition(step)
    temp = _safe_float(_condition_value(condition, "Temperature", "temperature", "temperature_c"))
    score = _safe_float(_condition_value(condition, "Score", "score", "confidence"))
    solvent = _condition_value(condition, "Solvent", "solvent")
    reagent = _condition_value(condition, "Reagent", "reagent")
    catalyst = _condition_value(condition, "Catalyst", "catalyst")
    domain = _condition_domain(step, reagent=reagent, catalyst=catalyst, solvent=solvent)
    enzyme_confidence = _top_enzyme_confidence(step)
    enzyme_weak = domain == "enzymatic" and (enzyme_confidence is None or enzyme_confidence < 0.25)
    issues: list[str] = []
    notes: list[str] = []

    if not condition:
        issues.append("missing_condition_prediction")
    if score is not None and score < 0.10:
        issues.append("low_condition_score")
    if temp is not None:
        if temp < -40:
            issues.append("extreme_low_temperature")
        elif temp < -20:
            issues.append("low_temperature")
        if temp > 120:
            issues.append("extreme_high_temperature")
        elif temp > 100:
            issues.append("high_temperature")
        if domain == "enzymatic" and (temp < 0 or temp > 70):
            issues.append("weak_enzyme_temperature_out_of_window" if enzyme_weak else "enzyme_temperature_out_of_window")
    if domain == "enzymatic" and solvent and _organic_solvent_without_water(str(solvent)):
        issues.append("weak_enzyme_organic_solvent_without_water" if enzyme_weak else "enzyme_organic_solvent_without_water")
    if _low_temperature_reactive_context(reagent):
        notes.append("reactive reagent low-temperature context")
        if temp is not None and temp < -20:
            issues.append("reactive_reagent_low_temperature")
    if _heated_coupling_context(step, reagent, catalyst):
        notes.append("heated coupling context")
        if temp is not None and temp > 80:
            issues.append("heated_coupling_temperature")

    high_issue_set = {
        "enzyme_temperature_out_of_window",
        "extreme_high_temperature",
    }
    risk = "high" if any(issue in high_issue_set for issue in issues) else "warn" if issues else "ok"
    return {
        "step_index": index,
        "domain": domain,
        "risk": risk,
        "issues": sorted(set(issues)),
        "notes": sorted(set(notes)),
        "has_condition_prediction": bool(condition),
        "temperature_c": round(float(temp), 3) if temp is not None else None,
        "condition_score": round(float(score), 4) if score is not None else None,
        "enzyme_confidence": round(float(enzyme_confidence), 4) if enzyme_confidence is not None else None,
        "solvent": str(solvent or ""),
        "reagent": str(reagent or ""),
        "catalyst": str(catalyst or ""),
    }


def _product_family(target_id: str) -> str:
    name = target_id.lower()
    if name in NATURAL_STATINS:
        return "natural_statin"
    if name in SYNTHETIC_STATINS:
        return "synthetic_statin"
    if "statin" in name:
        return "statin"
    return "general_product"


def _terminal_profile(terminal_reactants: list[str], *, product_smiles: str, target_heavy: int) -> dict[str, Any]:
    props = [_mol_props(smi) for smi in terminal_reactants]
    heavies = [int(prop.get("heavy_atoms") or 0) for prop in props]
    rings = [int(prop.get("ring_count") or 0) for prop in props]
    similarities = [_tanimoto(product_smiles, smi) for smi in terminal_reactants]
    carrier_roles = [_terminal_carrier_reagent_role(smi) for smi in terminal_reactants]
    effective_heavies = [heavy for heavy, role in zip(heavies, carrier_roles) if not role]
    effective_rings = [ring for ring, role in zip(rings, carrier_roles) if not role]
    effective_similarities = [sim for sim, role in zip(similarities, carrier_roles) if not role and sim is not None]
    max_heavy = max(heavies or [0])
    max_ring = max(rings or [0])
    max_similarity = max([sim for sim in similarities if sim is not None] or [0.0])
    effective_max_heavy = max(effective_heavies or [0])
    effective_max_ring = max(effective_rings or [0])
    effective_max_similarity = max(effective_similarities or [0.0])
    all_small = bool(terminal_reactants) and bool(effective_heavies) and effective_max_heavy <= 8
    product_like = bool(target_heavy) and (
        effective_max_heavy >= max(18, int(target_heavy * 0.65)) or effective_max_similarity >= 0.45
    )
    large_polycyclic = (
        bool(target_heavy)
        and effective_max_heavy >= max(16, int(target_heavy * 0.45))
        and effective_max_ring >= 2
    )
    carrier_reagents = [
        {"smiles": smi, "role": role, "heavy_atoms": heavy}
        for smi, role, heavy in zip(terminal_reactants, carrier_roles, heavies)
        if role
    ]
    return {
        "terminal_reactants": terminal_reactants,
        "terminal_heavy_atoms": heavies,
        "terminal_ring_counts": rings,
        "max_terminal_heavy_atoms": max_heavy,
        "max_terminal_ring_count": max_ring,
        "max_terminal_similarity_to_product": round(float(max_similarity), 4),
        "effective_terminal_heavy_atoms": effective_heavies,
        "effective_terminal_ring_counts": effective_rings,
        "effective_max_terminal_heavy_atoms": effective_max_heavy,
        "effective_max_terminal_ring_count": effective_max_ring,
        "effective_max_terminal_similarity_to_product": round(float(effective_max_similarity), 4),
        "carrier_reagents": carrier_reagents,
        "all_terminals_small": all_small,
        "product_like_terminal": product_like,
        "large_polycyclic_terminal": large_polycyclic,
    }


def _terminal_carrier_reagent_role(smiles: str) -> str | None:
    """Identify bulky terminal reagents whose scaffold is mostly a leaving carrier.

    Triphenylphosphoranes and phosphonates can have many heavy atoms, but their
    phenyl/alkoxy scaffold is a reagent carrier for olefination rather than a
    product-like terminal intermediate.  They remain visible in the audit, but
    are excluded from product-like terminal complexity scoring.
    """
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    for role, smarts in TERMINAL_CARRIER_REAGENT_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return role
    return None


def _reaction_profile(steps: list[dict[str, Any]]) -> dict[str, Any]:
    classes: list[str] = []
    sources: list[str] = []
    large_unexplained_atom_gain = False
    for step in steps:
        cls = _infer_reaction_class(step)
        source = str(step.get("source") or "unknown")
        classes.append(cls)
        sources.append(source)
        atom_change = ((step.get("reaction_interpretation") or {}).get("atom_change") or {})
        delta = _safe_float(atom_change.get("heavy_atom_delta"), 0.0) or 0.0
        if delta >= 10 and cls in {"other", "unknown"} and _source_is_model_artifact_prone(source):
            large_unexplained_atom_gain = True
    generic = sum(1 for cls in classes if cls in GENERIC_OR_RISKY_CLASSES)
    return {
        "classes": classes,
        "sources": sources,
        "class_counts": dict(Counter(classes)),
        "source_counts": dict(Counter(sources)),
        "generic_fraction": round(generic / max(len(classes), 1), 4),
        "large_unexplained_atom_gain": large_unexplained_atom_gain,
    }


def _atom_change_from_rxn(rxn: str) -> dict[str, Any]:
    if ">>" not in rxn:
        return {"heavy_atom_delta": 0}
    lhs, rhs = rxn.split(">>", 1)
    reactants = [part for part in lhs.split(".") if part]
    products = [part for part in rhs.split(".") if part]
    reactant_heavy = sum(int(_mol_props(smi).get("heavy_atoms") or 0) for smi in reactants)
    product_heavy = sum(int(_mol_props(smi).get("heavy_atoms") or 0) for smi in products)
    return {"heavy_atom_delta": product_heavy - reactant_heavy}


def _infer_reaction_class(step: dict[str, Any]) -> str:
    cls = str((step.get("reaction_interpretation") or {}).get("reaction_class") or step.get("reaction_type") or "unknown")
    if cls and cls not in {"unknown", "template"}:
        return cls
    interpretation = step.get("reaction_interpretation") or {}
    source = str(step.get("source") or interpretation.get("source_model") or interpretation.get("template") or "")
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    text = f"{source} {interpretation.get('template') or ''} {rxn}"
    if "[O;D1;H0:3]=[C:2]-[OH" in text and ("-[O;H0;D2" in text or "OC(=O)" in text or "C(=O)O" in text):
        return "hydrolysis"
    if "[C:4]-[O;H0;D2" in text and "[C;H0;D3" in text:
        return "esterification"
    if _reaction_has_esterification_signal(rxn):
        return "esterification"
    if "Br-[c" in text and "[CH2;D1" in text:
        return "C_C_coupling"
    if ".OB(O)" in text or "OB(O)" in text and ">>" in text and "Br" in text:
        return "C_C_coupling"
    return cls or "unknown"


def _reaction_has_esterification_signal(rxn: str) -> bool:
    if ">>" not in rxn:
        return False
    lhs, rhs = rxn.split(">>", 1)
    reactants = [part for part in lhs.split(".") if part]
    products = [part for part in rhs.split(".") if part]
    if not reactants or not products:
        return False
    has_acid = any(_mol_has_pattern(smi, "C(=O)[OX2H1]") for smi in reactants)
    has_alcohol = any(_mol_has_alcohol_or_phenol(smi) for smi in reactants)
    has_ester_product = any(_mol_has_pattern(smi, "[CX3](=O)[OX2][#6]") for smi in products)
    return bool(has_acid and has_alcohol and has_ester_product)


def _mol_has_alcohol_or_phenol(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    alcohol = Chem.MolFromSmarts("[OX2H][#6;!$(C=O)]")
    phenol = Chem.MolFromSmarts("[OX2H][c]")
    return bool(
        (alcohol is not None and mol.HasSubstructMatch(alcohol))
        or (phenol is not None and mol.HasSubstructMatch(phenol))
    )


def _mol_has_pattern(smiles: str, smarts: str) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    patt = Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return False
    return bool(mol.HasSubstructMatch(patt))


def _has_severe_issue(issues: list[str]) -> bool:
    severe = {
        "unfilled_route",
        "atom_balance_violation",
        "product_mismatch",
        "self_loop",
        "racemization_artifact",
        "large_unexplained_atom_gain",
        "large_unexplained_heavy_atom_gain",
        "large_unexplained_carbon_gain",
        "large_unexplained_hetero_atom_gain",
        "unexplained_new_element_source",
        "invalid_product_smiles",
        "invalid_or_missing_reactants",
        "trivial_stock_closure",
    }
    return any(issue in severe for issue in issues)


def _route_plausibility_from_steps(
    steps: list[dict[str, Any]],
    *,
    solved: bool,
    route_score: Any,
) -> dict[str, Any]:
    route_steps = []
    for idx, step in enumerate(steps):
        product, reactants, rxn = _plausibility_step_parts(step)
        if not rxn and product and reactants:
            rxn = f"{'.'.join(reactants)}>>{product}"
        route_steps.append(
            RouteStepCandidate(
                product_smiles=product,
                reactant_smiles=reactants,
                rxn_smiles=rxn,
                source_model=str(step.get("source") or step.get("source_model") or ""),
                score=_safe_float((step.get("scores") or {}).get("confidence"), _safe_float(step.get("score"))),
                stock_status=dict(step.get("stock_status") or {}),
                condition_predictions=list(step.get("condition_predictions") or []),
                enzyme_ec_annotations=list(step.get("enzyme_ec_annotations") or []),
                catalyst_annotations=list(step.get("catalyst_annotations") or []),
                raw_backend_metadata={"index": idx},
            )
        )
    if not route_steps:
        return {
            "passed": False,
            "reasons": [],
            "steps": [],
            "contract": "minimum material-sanity screen; no steps available",
        }
    route = RouteCandidate(
        target_smiles=route_steps[0].product_smiles,
        steps=route_steps,
        solved=bool(solved),
        score=_safe_float(route_score),
    )
    return audit_route_plausibility(route)


def _plausibility_step_parts(step: dict[str, Any]) -> tuple[str, list[str], str]:
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    product = str(step.get("product") or step.get("product_smiles") or "")
    reactants = []
    if step.get("main_reactant"):
        reactants.append(str(step.get("main_reactant")))
    reactants.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    reactants.extend(str(smi) for smi in step.get("reactant_smiles") or [] if smi)
    if rxn and ">>" in rxn:
        lhs, rhs = rxn.split(">>", 1)
        if not product:
            product = rhs.split(".")[0] if rhs else ""
        if not reactants:
            reactants = [part for part in lhs.split(".") if part]
    # Preserve order while dropping duplicate reactants.
    deduped = []
    seen = set()
    for smi in reactants:
        if smi and smi not in seen:
            deduped.append(smi)
            seen.add(smi)
    return product, deduped, rxn


def _is_triage_signal(route: dict[str, Any]) -> bool:
    return route.get("route_class") in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}


def _route_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return (ROUTE_CLASS_ORDER.get(str(row.get("route_class")), 99), int(row.get("rank") or 10**9))


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": step.get("source"),
        "reaction_class": _infer_reaction_class(step),
        "reaction_smiles": step.get("reaction_smiles") or step.get("rxn_smiles"),
    }


def _top_condition(step: dict[str, Any]) -> dict[str, Any]:
    for row in step.get("condition_predictions") or []:
        if isinstance(row, dict):
            return row
    conditions = step.get("step_conditions") or step.get("conditions") or {}
    return conditions if isinstance(conditions, dict) else {}


def _condition_value(row: dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row.get(key) not in {None, ""}:
            return row.get(key)
        low = str(key).lower()
        if low in lower and lower[low] not in {None, ""}:
            return lower[low]
    return None


def _condition_domain(step: dict[str, Any], *, reagent: Any = None, catalyst: Any = None, solvent: Any = None) -> str:
    if _has_strong_chemical_condition(reagent, catalyst):
        return "chemical"
    text = " ".join(
        str(value or "")
        for value in [
            step.get("reaction_type"),
            step.get("source"),
            step.get("source_model"),
            step.get("ec"),
            (step.get("reaction_interpretation") or {}).get("reaction_class"),
        ]
    ).lower()
    if step.get("enzyme_ec_annotations") or "enzym" in text:
        return "enzymatic"
    if "template" in text or "chem" in text or "uspto" in text:
        return "chemical"
    return "unknown"


def _top_enzyme_confidence(step: dict[str, Any]) -> float | None:
    for row in step.get("enzyme_ec_annotations") or []:
        if not isinstance(row, dict):
            continue
        value = row.get("confidence")
        if value is None and isinstance(row.get("raw"), dict):
            value = row["raw"].get("Confidence")
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _has_strong_chemical_condition(reagent: Any, catalyst: Any = None) -> bool:
    text = " ".join(str(value or "") for value in [reagent, catalyst])
    tokens = [
        "[Li+]",
        "[Li]",
        "[AlH4]",
        "[BH4-]",
        "[H-]",
        "[NaH]",
        "O=P(Cl)(Cl)Cl",
        "Cl[AlH3]",
        "[Al+3]",
        "O=[Mn]=O",
        "Pd",
        "c1ccncc1",
        "O=C([O-])[O-]",
        "[Mg++]",
        "O=C(N1C=CN=C1)N1C=CN=C1",
    ]
    return any(token in text for token in tokens)


def _organic_solvent_without_water(solvent: str) -> bool:
    text = str(solvent or "")
    if not text:
        return False
    water_tokens = {"O", "water", "h2o", "H2O"}
    if any(token in water_tokens for token in text.replace(";", ".").split(".")):
        return False
    mols = [Chem.MolFromSmiles(token) for token in text.replace(";", ".").split(".") if token]
    return any(mol is not None and any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()) for mol in mols)


def _low_temperature_reactive_context(reagent: Any) -> bool:
    text = str(reagent or "")
    tokens = [
        "[Li+]",
        "[Li]",
        "[AlH4]",
        "[BH4-]",
        "[H-]",
        "[NaH]",
        "C[Si](C)(C)[N-][Si]",
    ]
    return any(token in text for token in tokens)


def _heated_coupling_context(step: dict[str, Any], reagent: Any, catalyst: Any) -> bool:
    text = " ".join(str(value or "") for value in [reagent, catalyst, step.get("reaction_smiles"), step.get("rxn_smiles")])
    return "Pd" in text or "OB(O)" in text or "O=C([O-])[O-]" in text


def _has_acylating_piece(smiles_list: list[str]) -> bool:
    for smi in smiles_list:
        text = smi.upper()
        if "C(=O)CL" in text or "C(=O)O" in text or "C(=O)OC" in text:
            props = _mol_props(smi)
            if int(props.get("heavy_atoms") or 0) <= 12:
                return True
    return False


def _has_aryl_coupling_signal(smiles_list: list[str], classes: list[str]) -> bool:
    if "C_C_coupling" in classes:
        return True
    for smi in smiles_list:
        text = smi
        has_boron_coupling_handle = "B(O)" in text or "[B" in text
        has_aryl_halide = _has_aryl_halide(smi)
        if has_boron_coupling_handle or has_aryl_halide:
            props = _mol_props(smi)
            if int(props.get("ring_count") or 0) >= 1:
                return True
    return False


def _has_aryl_halide(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    for smarts in ("[c][Br]", "[c][Cl]", "[c][I]"):
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return True
    return False


def _source_is_model_artifact_prone(source: str) -> bool:
    text = source.lower()
    return "enzyformer" in text or "enzexpand" in text


def _mol_props(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {"smiles": smiles, "valid": False, "heavy_atoms": 0, "ring_count": 0}
    return {
        "smiles": smiles,
        "valid": True,
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(mol.GetRingInfo().NumRings()),
    }


def _tanimoto(a: str, b: str) -> float | None:
    mol_a = Chem.MolFromSmiles(a or "")
    mol_b = Chem.MolFromSmiles(b or "")
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _rate(num: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(float(num) / float(denom), 4)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError(f"unsupported benchmark format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit product-oriented route feasibility from benchmark run.json")
    ap.add_argument("--run", required=True)
    ap.add_argument("--benchmark", default=None, help="Optional benchmark JSON used to recover target IDs for native baseline outputs")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(Path(args.benchmark)) if args.benchmark else None
    audit = build_product_route_feasibility_audit(run, benchmark_rows=benchmark_rows)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_product_route_feasibility_markdown(audit, Path(args.output_md))
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(args.output_md),
        "triage_signal_targets": audit["triage_signal_targets"],
        "autonomous_route_candidate_targets": audit["autonomous_route_candidate_targets"],
    }, indent=2))


if __name__ == "__main__":
    main()
