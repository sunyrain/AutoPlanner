#!/usr/bin/env python3
"""Rerank existing route pools with the cascade verifier.

This is a mainline bridge between ChemEnzy proposal generation and
verifier-first cascade training/search. It does not create new proposals; it
rescoring route pools that already exist so we can measure whether rule or
learned verifier signals improve route selection before generator fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cascade_planner.cascade_verifier import load_learned_verifier, predict_learned_verifier, verify_cascade_route


SCHEMA_VERSION = "cascade_verifier_route_rerank.v1"
LEARNED_VERIFIER_POLICIES = ("annotation_only", "calibrated_conservative", "raw_score")


def main() -> None:
    args = _parse_args()
    result = rerank_routes_with_verifier(
        input_path=args.input,
        output_path=args.output,
        learned_verifier_model=args.learned_verifier_model,
        learned_verifier_policy=args.learned_verifier_policy,
        drop_infeasible=args.drop_infeasible,
        min_rule_score=args.min_rule_score,
        max_routes_per_target=args.max_routes_per_target,
        default_stage_mode=args.default_stage_mode,
    )
    if args.markdown:
        _write_markdown(result, args.markdown)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def rerank_routes_with_verifier(
    *,
    input_path: Path,
    output_path: Path,
    learned_verifier_model: Path | None = None,
    learned_verifier_policy: str = "annotation_only",
    drop_infeasible: bool = False,
    min_rule_score: float | None = None,
    max_routes_per_target: int | None = None,
    default_stage_mode: str = "stepwise",
) -> dict[str, Any]:
    if learned_verifier_policy not in LEARNED_VERIFIER_POLICIES:
        raise ValueError(f"unknown learned_verifier_policy: {learned_verifier_policy}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    learned = load_learned_verifier(learned_verifier_model)
    targets = []
    total_in = 0
    total_out = 0
    feasible_count = 0
    dropped_count = 0
    reason_counts: Counter[str] = Counter()
    score_deltas = []
    for target_index, target_row in enumerate(_iter_target_rows(payload)):
        target = target_row["target_smiles"]
        scored_routes = []
        for route_index, route in enumerate(target_row["routes"]):
            total_in += 1
            scored = _score_route(
                route,
                target_smiles=target,
                original_rank=route_index,
                learned=learned,
                learned_verifier_policy=learned_verifier_policy,
                default_stage_mode=default_stage_mode,
            )
            reason_counts.update(scored["cascade_verifier_rerank"]["reason_counts"])
            feasible = bool(scored["cascade_verifier_rerank"]["rule_feasible"])
            feasible_count += int(feasible)
            if drop_infeasible and not feasible:
                dropped_count += 1
                continue
            if min_rule_score is not None and float(scored["cascade_verifier_rerank"]["rule_score"]) < float(min_rule_score):
                dropped_count += 1
                continue
            scored_routes.append(scored)
        scored_routes.sort(key=_route_sort_key, reverse=True)
        if max_routes_per_target is not None:
            kept = max(0, int(max_routes_per_target))
            dropped_count += max(0, len(scored_routes) - kept)
            scored_routes = scored_routes[:kept]
        for rank, route in enumerate(scored_routes):
            meta = dict(route.get("cascade_verifier_rerank") or {})
            meta["rerank_rank"] = rank
            route["cascade_verifier_rerank"] = meta
            score_deltas.append(int(meta.get("original_rank", rank)) - rank)
        total_out += len(scored_routes)
        targets.append(
            {
                "target_index": target_index,
                "target_smiles": target,
                "n_routes_input": len(target_row["routes"]),
                "n_routes_output": len(scored_routes),
                "routes": scored_routes,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(input_path),
        "output": str(output_path),
        "learned_verifier_model": str(learned_verifier_model) if learned_verifier_model else None,
        "learned_verifier_policy": learned_verifier_policy if learned_verifier_model else None,
        "drop_infeasible": bool(drop_infeasible),
        "min_rule_score": min_rule_score,
        "max_routes_per_target": max_routes_per_target,
        "default_stage_mode": default_stage_mode,
        "n_targets": len(targets),
        "n_routes_input": total_in,
        "n_routes_output": total_out,
        "n_feasible_by_rule": feasible_count,
        "n_dropped": dropped_count,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "mean_rank_lift": round(sum(score_deltas) / len(score_deltas), 4) if score_deltas else None,
        "contract": (
            "Verifier rerank evaluates existing route pools. It cannot recover a route "
            "that was absent from the proposal pool."
        ),
    }
    result = {"summary": summary, "targets": targets}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _iter_target_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("targets"), list):
        for item in payload["targets"]:
            if isinstance(item, dict):
                target = str(item.get("target_smiles") or item.get("target") or "")
                routes = _route_list(item)
                if target and routes:
                    rows.append({"target_smiles": target, "routes": routes})
        return rows
    if isinstance(payload, dict) and isinstance(payload.get("routes"), list):
        target = str(payload.get("target_smiles") or payload.get("target") or "")
        if target:
            return [{"target_smiles": target, "routes": _route_list(payload)}]
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                target = str(item.get("target_smiles") or item.get("target") or "")
                routes = _route_list(item)
                if target and routes:
                    rows.append({"target_smiles": target, "routes": routes})
    return rows


def _route_list(row: dict[str, Any]) -> list[dict[str, Any]]:
    routes = (row.get("planner_output") or {}).get("routes") if isinstance(row.get("planner_output"), dict) else None
    if routes is None:
        routes = row.get("routes") or []
    return [route for route in routes if isinstance(route, dict)]


def _score_route(
    route: dict[str, Any],
    *,
    target_smiles: str,
    original_rank: int,
    learned: dict[str, Any] | None,
    learned_verifier_policy: str,
    default_stage_mode: str,
) -> dict[str, Any]:
    out = json.loads(json.dumps(route))
    cascade = {
        "target": target_smiles,
        "target_smiles": target_smiles,
        "steps": [_normalize_step(step) for step in out.get("steps") or [] if isinstance(step, dict)],
        "stage_partition": _stage_partition(out, default_stage_mode=default_stage_mode),
        "metadata": dict(out.get("metadata") or {}),
    }
    report = verify_cascade_route(cascade, target_smiles=target_smiles).to_dict()
    learned_payload = predict_learned_verifier(learned, cascade, target_smiles=target_smiles) if learned else None
    rule_score = float(report.get("score") or 0.0)
    learned_score = (
        float(learned_payload["feasible_probability"])
        if learned_payload is not None
        else None
    )
    if learned_score is None or learned_verifier_policy == "annotation_only":
        rerank_score = rule_score
    elif learned_verifier_policy == "raw_score":
        rerank_score = learned_score
    elif learned_payload.get("conservative_feasible"):
        rerank_score = 1.0 + learned_score
    else:
        rerank_score = rule_score
    out["cascade_verifier_rerank"] = {
        "schema_version": SCHEMA_VERSION,
        "original_rank": int(original_rank),
        "rerank_rank": None,
        "rule_feasible": bool(report.get("feasible")),
        "rule_score": round(rule_score, 6),
        "reason_counts": report.get("reason_counts") or {},
        "findings": report.get("findings") or [],
        "learned_verifier": learned_payload,
        "learned_verifier_policy": learned_verifier_policy if learned_payload else None,
        "rerank_score": round(float(rerank_score), 6),
        "original_score": _route_score(out),
    }
    return out


def _normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    reactants = _step_reactants(step)
    condition = _top_condition_prediction(step)
    product = str(step.get("product") or step.get("product_smiles") or "")
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    if not rxn and product:
        rxn = ".".join(reactants) + ">>" + product
    return {
        "product": product,
        "product_smiles": product,
        "main_reactant": str(step.get("main_reactant") or (reactants[0] if reactants else "")),
        "aux_reactants": list(step.get("aux_reactants") or reactants[1:]),
        "reactants": reactants,
        "reactant_smiles": reactants,
        "reaction_smiles": rxn,
        "rxn_smiles": rxn,
        "source": str(step.get("source") or step.get("source_model") or ""),
        "reaction_type": str(step.get("reaction_type") or ""),
        "ec": str(step.get("ec") or ""),
        "T": _first_present(step, condition, "T", "Temperature", "temperature", "temperature_c"),
        "pH": _first_present(step, condition, "pH", "ph", "PH"),
        "solvent": str(_first_present(step, condition, "solvent", "Solvent") or ""),
        "catalyst": str(_first_present(step, condition, "catalyst", "Catalyst", "reagent", "Reagent") or ""),
        "condition_predictions": list(step.get("condition_predictions") or []),
        "enzyme_ec_annotations": list(step.get("enzyme_ec_annotations") or []),
        "cofactor_requirements": dict(step.get("cofactor_requirements") or {}),
        "cofactor_regenerations": dict(step.get("cofactor_regenerations") or {}),
    }


def _step_reactants(step: dict[str, Any]) -> list[str]:
    values = step.get("reactants") or step.get("reactant_smiles")
    if isinstance(values, list) and values:
        return [str(value) for value in values if value]
    out = []
    if step.get("main_reactant"):
        out.append(str(step.get("main_reactant")))
    out.extend(str(value) for value in step.get("aux_reactants") or [] if value)
    if out:
        return out
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    if ">>" in rxn:
        return [part for part in rxn.split(">>", 1)[0].split(".") if part]
    return []


def _stage_partition(route: dict[str, Any], *, default_stage_mode: str) -> list[str]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    raw = route.get("stage_partition")
    if isinstance(raw, list) and len(raw) == len(steps):
        return [str(value or "stage_1") for value in raw]
    if default_stage_mode == "single":
        return ["stage_1" for _ in steps]
    return [f"stage_{idx + 1}" for idx in range(len(steps))]


def _top_condition_prediction(step: dict[str, Any]) -> dict[str, Any]:
    for row in step.get("condition_predictions") or []:
        if isinstance(row, dict):
            return row
    return {}


def _first_present(step: dict[str, Any], condition: dict[str, Any], *keys: str) -> Any:
    lower_step = {str(key).lower(): value for key, value in step.items()}
    lower_condition = {str(key).lower(): value for key, value in condition.items()}
    for key in keys:
        for mapping in (step, condition, lower_step, lower_condition):
            lookup = key if mapping in (step, condition) else key.lower()
            if lookup in mapping and mapping[lookup] not in (None, ""):
                return mapping[lookup]
    return None


def _route_sort_key(route: dict[str, Any]) -> tuple[float, float, float, float]:
    meta = route.get("cascade_verifier_rerank") or {}
    feasible = 1.0 if meta.get("rule_feasible") else 0.0
    rerank_score = float(meta.get("rerank_score") or 0.0)
    original_rank = int(meta.get("original_rank", 0))
    # Preserve the original route-pool order when verifier scores tie. This
    # keeps A/B comparisons attributable to verifier signals rather than a
    # hidden secondary score.
    return (feasible, rerank_score, -float(original_rank), 0.0)


def _route_score(route: dict[str, Any]) -> float:
    value = route.get("score")
    if value in (None, ""):
        value = (route.get("metrics") or {}).get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Cascade Verifier Route Rerank",
        "",
        f"- Input routes: `{summary['n_routes_input']}`",
        f"- Output routes: `{summary['n_routes_output']}`",
        f"- Rule-feasible routes: `{summary['n_feasible_by_rule']}`",
        f"- Dropped routes: `{summary['n_dropped']}`",
        f"- Mean rank lift: `{summary['mean_rank_lift']}`",
        "",
        "## Failure Reasons",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in summary["reason_counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Contract", "", summary["contract"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank existing routes with cascade verifier scores")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--learned-verifier-model", type=Path)
    parser.add_argument(
        "--learned-verifier-policy",
        choices=LEARNED_VERIFIER_POLICIES,
        default="annotation_only",
        help=(
            "How learned verifier scores affect reranking. annotation_only writes "
            "learned probabilities/reasons without changing route order; calibrated_conservative "
            "uses the trained recommended threshold and falls back to rule score "
            "below threshold; raw_score preserves the older experimental behavior."
        ),
    )
    parser.add_argument("--drop-infeasible", action="store_true")
    parser.add_argument("--min-rule-score", type=float)
    parser.add_argument("--max-routes-per-target", type=int)
    parser.add_argument(
        "--default-stage-mode",
        choices=["stepwise", "single"],
        default="stepwise",
        help=(
            "Stage assumption when routes do not export stage_partition. "
            "stepwise avoids treating ordinary sequential synthesis as a one-pot cascade."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
