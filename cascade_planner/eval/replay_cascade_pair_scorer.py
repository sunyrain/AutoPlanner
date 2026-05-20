"""Replay existing cascade-search traces with local pair-scorer rewards.

This is a small-data diagnostic for Stage-1 integration.  It does not call the
ChemEnzy backend; it reuses candidate pools already captured in trace JSONL and
asks how a rule or learned adjacent-step cascade scorer would reorder the same
actions.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import fields
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.pair_scorer import LearnedCascadePairScorer, RuleCascadePairScorer
from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeActionType,
    CascadeModule,
    CascadeProgramState,
    CofactorLedger,
    ConditionEnvelope,
    RedoxLedger,
    StageGraph,
    StepAnnotation,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, gt_reactants, reaction_reactants


def replay_cascade_pair_scorer(
    *,
    trace_path: Path,
    output_path: Path,
    model_path: Path | None = None,
    weights: list[float] | None = None,
    tie_epsilons: list[float] | None = None,
    limit_events: int | None = None,
    limit_targets: int | None = None,
) -> dict[str, Any]:
    weights = weights or [0.35]
    tie_epsilons = tie_epsilons or []
    rule_scorer = RuleCascadePairScorer()
    learned_scorer = LearnedCascadePairScorer(model_path) if model_path else None

    rows = _read_trace_rows(trace_path, limit_events=limit_events, limit_targets=limit_targets)
    event_reports = []
    for row_index, row in enumerate(rows):
        event = row.get("event") or {}
        actions = [_action_from_dict(item) for item in event.get("candidate_actions") or []]
        child_summaries = list(event.get("child_summaries") or [])
        if len(actions) < 2 or len(child_summaries) != len(actions):
            continue
        state = _state_from_dict(event.get("state") or {}, fallback_target=str(row.get("target_smiles") or ""))
        base_scores = _base_scores(event, actions)
        labels = [_candidate_labels(action, row) for action in actions]
        child_quality = [
            _child_quality(child, label)
            for child, label in zip(child_summaries, labels)
        ]
        base_order = _rank_indices(base_scores)
        pair_payloads: dict[str, list[float]] = {"rule": _pair_rewards(rule_scorer, state, actions, event)}
        if learned_scorer is not None:
            pair_payloads["learned"] = _pair_rewards(learned_scorer, state, actions, event)

        variants = {
            "base": _selection_report(
                name="base",
                scores=base_scores,
                rewards=[0.0 for _ in actions],
                base_order=base_order,
                labels=labels,
                child_summaries=child_summaries,
                child_quality=child_quality,
            )
        }
        for scorer_name, rewards in pair_payloads.items():
            for weight in weights:
                scores = [float(base) + float(weight) * float(reward) for base, reward in zip(base_scores, rewards)]
                variants[f"{scorer_name}_w{_weight_token(weight)}"] = _selection_report(
                    name=f"{scorer_name}_w{_weight_token(weight)}",
                    scores=scores,
                    rewards=rewards,
                    base_order=base_order,
                    labels=labels,
                    child_summaries=child_summaries,
                    child_quality=child_quality,
                )
                for tie_epsilon in tie_epsilons:
                    guarded_scores = _guarded_tie_break_scores(
                        base_scores=base_scores,
                        rewards=rewards,
                        child_summaries=child_summaries,
                        base_order=base_order,
                        weight=float(weight),
                        tie_epsilon=float(tie_epsilon),
                    )
                    variants[f"{scorer_name}_guarded_w{_weight_token(weight)}_eps{_weight_token(tie_epsilon)}"] = _selection_report(
                        name=f"{scorer_name}_guarded_w{_weight_token(weight)}_eps{_weight_token(tie_epsilon)}",
                        scores=guarded_scores,
                        rewards=rewards,
                        base_order=base_order,
                        labels=labels,
                        child_summaries=child_summaries,
                        child_quality=child_quality,
                    )
        learned_rewards = pair_payloads.get("learned") or pair_payloads["rule"]
        event_reports.append(
            {
                "row_index": row_index,
                "target_smiles": row.get("target_smiles"),
                "route_domain": row.get("route_domain"),
                "depth": int(event.get("depth") or len(state.step_annotations)),
                "expanded_leaf": event.get("expanded_leaf"),
                "n_candidates": len(actions),
                "pair_informative": (max(learned_rewards) - min(learned_rewards)) > 1e-6,
                "base_top_rxn": _rxn(actions[base_order[0]]),
                "variants": variants,
            }
        )

    report = {
        "metadata": {
            "trace_path": str(trace_path),
            "model_path": str(model_path) if model_path else None,
            "weights": weights,
            "tie_epsilons": tie_epsilons,
            "limit_events": limit_events,
            "limit_targets": limit_targets,
            "n_trace_rows_read": len(rows),
            "n_ranking_events": len(event_reports),
            "n_pair_informative_events": sum(int(item["pair_informative"]) for item in event_reports),
            "diagnostic_contract": "trace_replay_pair_scorer_search_ordering.v1",
            "note": "Replay uses captured proposal pools; it does not run ChemEnzy backend or measure live solve rate.",
        },
        "all_events": _aggregate(event_reports),
        "pair_informative_events": _aggregate([item for item in event_reports if item["pair_informative"]]),
        "changed_top1_examples": _changed_examples(event_reports, limit=8),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _guarded_tie_break_scores(
    *,
    base_scores: list[float],
    rewards: list[float],
    child_summaries: list[dict[str, Any]],
    base_order: list[int],
    weight: float,
    tie_epsilon: float,
) -> list[float]:
    if not base_scores:
        return []
    base_top_idx = base_order[0]
    base_top_score = float(base_scores[base_top_idx])
    base_top_child = child_summaries[base_top_idx] if base_top_idx < len(child_summaries) else {}
    base_top_stock_closed = bool(base_top_child.get("stock_closed"))
    base_top_no_failure = not bool(base_top_child.get("failure_categories") or [])
    scores = [float(value) for value in base_scores]
    for idx, (base, reward) in enumerate(zip(base_scores, rewards)):
        if float(base) < base_top_score - float(tie_epsilon):
            continue
        child = child_summaries[idx] if idx < len(child_summaries) else {}
        child_stock_closed = bool(child.get("stock_closed"))
        child_no_failure = not bool(child.get("failure_categories") or [])
        if base_top_stock_closed and not child_stock_closed:
            continue
        if base_top_no_failure and not child_no_failure:
            continue
        scores[idx] = float(base) + float(weight) * float(reward)
    return scores


def _selection_report(
    *,
    name: str,
    scores: list[float],
    rewards: list[float],
    base_order: list[int],
    labels: list[dict[str, Any]],
    child_summaries: list[dict[str, Any]],
    child_quality: list[float],
) -> dict[str, Any]:
    order = _rank_indices(scores)
    top_idx = order[0]
    best_quality = max(child_quality) if child_quality else float("-inf")
    child = child_summaries[top_idx] if top_idx < len(child_summaries) else {}
    failure_categories = list(child.get("failure_categories") or [])
    return {
        "name": name,
        "top_index": top_idx,
        "top_rank_change_vs_base": int(base_order.index(top_idx) + 1) if top_idx in base_order else None,
        "top_score": round(float(scores[top_idx]), 6),
        "top_pair_reward": round(float(rewards[top_idx]), 6),
        "top_exact_gt_reaction": bool(labels[top_idx].get("exact_gt_reaction")),
        "top_gt_reactant_hit": bool(labels[top_idx].get("gt_reactant_hit")),
        "top_stock_closed": bool(child.get("stock_closed")),
        "top_no_failure": not bool(failure_categories),
        "top_failure_count": len(failure_categories),
        "top_child_quality": round(float(child_quality[top_idx]), 6),
        "top_oracle_child_quality": bool(child_quality[top_idx] >= best_quality - 1e-12),
    }


def _aggregate(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"n_events": 0, "variants": {}}
    variant_names = sorted({name for event in events for name in (event.get("variants") or {})})
    out = {"n_events": len(events), "variants": {}}
    base_top = {
        idx: (event.get("variants") or {}).get("base", {}).get("top_index")
        for idx, event in enumerate(events)
    }
    for name in variant_names:
        rows = [(event.get("variants") or {}).get(name) or {} for event in events]
        changed = 0
        for idx, row in enumerate(rows):
            if name != "base" and row.get("top_index") != base_top.get(idx):
                changed += 1
        out["variants"][name] = {
            "top1_exact_gt_reaction_rate": _rate(sum(bool(row.get("top_exact_gt_reaction")) for row in rows), len(rows)),
            "top1_gt_reactant_hit_rate": _rate(sum(bool(row.get("top_gt_reactant_hit")) for row in rows), len(rows)),
            "top1_stock_closed_rate": _rate(sum(bool(row.get("top_stock_closed")) for row in rows), len(rows)),
            "top1_no_failure_rate": _rate(sum(bool(row.get("top_no_failure")) for row in rows), len(rows)),
            "top1_oracle_child_quality_rate": _rate(sum(bool(row.get("top_oracle_child_quality")) for row in rows), len(rows)),
            "mean_top1_child_quality": _mean([row.get("top_child_quality") for row in rows]),
            "mean_top1_failure_count": _mean([row.get("top_failure_count") for row in rows]),
            "mean_top1_pair_reward": _mean([row.get("top_pair_reward") for row in rows]),
            "top1_changed_vs_base_rate": _rate(changed, len(rows)) if name != "base" else 0.0,
            "top1_changed_vs_base_count": changed if name != "base" else 0,
        }
    return out


def _changed_examples(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out = []
    for event in events:
        base = (event.get("variants") or {}).get("base") or {}
        for name, variant in (event.get("variants") or {}).items():
            if name == "base" or variant.get("top_index") == base.get("top_index"):
                continue
            out.append(
                {
                    "target_smiles": event.get("target_smiles"),
                    "route_domain": event.get("route_domain"),
                    "depth": event.get("depth"),
                    "expanded_leaf": event.get("expanded_leaf"),
                    "variant": name,
                    "base_top_index": base.get("top_index"),
                    "variant_top_index": variant.get("top_index"),
                    "variant_base_rank": variant.get("top_rank_change_vs_base"),
                    "base_top_quality": base.get("top_child_quality"),
                    "variant_top_quality": variant.get("top_child_quality"),
                    "variant_pair_reward": variant.get("top_pair_reward"),
                    "variant_exact_gt_reaction": variant.get("top_exact_gt_reaction"),
                    "variant_gt_reactant_hit": variant.get("top_gt_reactant_hit"),
                }
            )
            break
        if len(out) >= limit:
            break
    return out


def _read_trace_rows(path: Path, *, limit_events: int | None, limit_targets: int | None) -> list[dict[str, Any]]:
    rows = []
    targets = []
    target_set = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get("target_smiles") or "")
            if limit_targets is not None and target not in target_set and len(targets) >= int(limit_targets):
                continue
            if target and target not in target_set:
                target_set.add(target)
                targets.append(target)
            rows.append(row)
            if limit_events is not None and len(rows) >= int(limit_events):
                break
    return rows


def _state_from_dict(payload: dict[str, Any], *, fallback_target: str) -> CascadeProgramState:
    steps = [_step_from_dict(item) for item in payload.get("step_annotations") or payload.get("steps") or []]
    state = CascadeProgramState(
        target_smiles=str(payload.get("target_smiles") or fallback_target),
        open_molecule_leaves=[str(value) for value in payload.get("open_molecule_leaves") or payload.get("open_leaves") or []],
        reaction_graph=dict(payload.get("reaction_graph") or payload.get("route_graph") or {}),
        stage_graph=StageGraph.from_partition(list(payload.get("stage_partition") or [])) if payload.get("stage_partition") else StageGraph(),
        current_stage=str(payload.get("current_stage") or "stage_1"),
        step_annotations=steps,
        cofactor_ledger=_cofactor_ledger_from_dict(payload.get("cofactor_ledger") or {}),
        redox_ledger=_redox_ledger_from_dict(payload.get("redox_ledger") or {}),
        stock_status={str(k): v for k, v in (payload.get("stock_status") or {}).items()},
        evidence_confidence=_float_or_none(payload.get("evidence_confidence")),
        cascade_cost=_float_or_none(payload.get("cascade_cost")),
        raw_metadata=dict(payload.get("raw_metadata") or {}),
    )
    state._sync_aliases()
    return state


def _action_from_dict(payload: dict[str, Any]) -> CascadeAction:
    step = _step_from_dict(payload.get("step") or {}) if payload.get("step") else None
    return CascadeAction(
        action_type=payload.get("action_type") or CascadeActionType.RETROSYNTHETIC_STEP,
        target_leaf=str(payload.get("target_leaf") or (step.product_smiles if step else "")),
        step=step,
        module=_module_from_dict(payload.get("module")) if payload.get("module") else None,
        evidence_payload=dict(payload.get("evidence_payload") or {}),
        source=str(payload.get("source") or ""),
        cost_delta=float(payload.get("cost_delta") or 0.0),
        metadata=dict(payload.get("metadata") or {}),
    )


def _step_from_dict(payload: dict[str, Any]) -> StepAnnotation:
    condition = _condition_from_dict(payload.get("condition")) if payload.get("condition") else None
    module = _module_from_dict(payload.get("enzyme_module")) if payload.get("enzyme_module") else None
    return StepAnnotation(
        product_smiles=str(payload.get("product_smiles") or payload.get("product") or ""),
        reactant_smiles=[str(value) for value in payload.get("reactant_smiles") or payload.get("reactants") or []],
        rxn_smiles=str(payload.get("rxn_smiles") or payload.get("reaction_smiles") or ""),
        source_model=str(payload.get("source_model") or ""),
        score=_float_or_none(payload.get("score")),
        reaction_type=str(payload.get("reaction_type") or ""),
        ec_numbers=[str(value) for value in payload.get("ec_numbers") or []],
        uniprot_ids=[str(value) for value in payload.get("uniprot_ids") or []],
        condition=condition,
        enzyme_module=module,
        stage_id=str(payload.get("stage_id") or "stage_1"),
        cofactor_requirements={str(k): float(v or 0.0) for k, v in (payload.get("cofactor_requirements") or {}).items()},
        cofactor_regenerations={str(k): float(v or 0.0) for k, v in (payload.get("cofactor_regenerations") or {}).items()},
        redox_change=str(payload.get("redox_change") or ""),
        evidence_confidence=_float_or_none(payload.get("evidence_confidence")),
        stock_status={str(k): v for k, v in (payload.get("stock_status") or {}).items()},
        raw_metadata=dict(payload.get("raw_metadata") or {}),
    )


def _condition_from_dict(payload: dict[str, Any]) -> ConditionEnvelope:
    names = {field.name for field in fields(ConditionEnvelope)}
    return ConditionEnvelope(**{key: value for key, value in dict(payload or {}).items() if key in names})


def _module_from_dict(payload: dict[str, Any] | None) -> CascadeModule | None:
    if not payload:
        return None
    data = dict(payload)
    if data.get("condition_envelope"):
        data["condition_envelope"] = _condition_from_dict(data["condition_envelope"])
    names = {field.name for field in fields(CascadeModule)}
    return CascadeModule(**{key: value for key, value in data.items() if key in names})


def _cofactor_ledger_from_dict(payload: dict[str, Any]) -> CofactorLedger:
    return CofactorLedger(
        required={str(k): float(v or 0.0) for k, v in (payload.get("required") or {}).items()},
        regenerated={str(k): float(v or 0.0) for k, v in (payload.get("regenerated") or {}).items()},
        consumed={str(k): float(v or 0.0) for k, v in (payload.get("consumed") or {}).items()},
        produced={str(k): float(v or 0.0) for k, v in (payload.get("produced") or {}).items()},
    )


def _redox_ledger_from_dict(payload: dict[str, Any]) -> RedoxLedger:
    return RedoxLedger(
        oxidants={str(k): float(v or 0.0) for k, v in (payload.get("oxidants") or {}).items()},
        reductants={str(k): float(v or 0.0) for k, v in (payload.get("reductants") or {}).items()},
        electron_acceptors={str(k): float(v or 0.0) for k, v in (payload.get("electron_acceptors") or {}).items()},
        electron_donors={str(k): float(v or 0.0) for k, v in (payload.get("electron_donors") or {}).items()},
        conflicts=[str(value) for value in payload.get("conflicts") or []],
    )


def _base_scores(event: dict[str, Any], actions: list[CascadeAction]) -> list[float]:
    scores = event.get("candidate_scores") or []
    if len(scores) == len(actions):
        return [float(value or 0.0) for value in scores]
    out = []
    for action in actions:
        value = action.metadata.get("transition_value_score")
        if value is None and action.step is not None:
            value = action.step.score
        out.append(float(value or 0.0))
    return out


def _pair_rewards(scorer: Any, state: CascadeProgramState, actions: list[CascadeAction], event: dict[str, Any]) -> list[float]:
    rewards = []
    for action in actions:
        try:
            score = scorer.score_action(state, action, expanded_leaf=str(event.get("expanded_leaf") or action.target_leaf))
            if bool(getattr(score, "applicable", True)):
                rewards.append(float(getattr(score, "search_reward", 0.0)))
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def _candidate_labels(action: CascadeAction, trace_row: dict[str, Any]) -> dict[str, Any]:
    step = action.step
    rxn = _rxn(action)
    gt_rxns = {
        canonical_reaction(str(item.get("rxn_smiles") or "")) or str(item.get("rxn_smiles") or "")
        for item in trace_row.get("gt_route") or []
        if item.get("rxn_smiles")
    }
    reactants = set(reaction_reactants(rxn)) if rxn else set()
    if step is not None:
        reactants.update(str(value) for value in step.reactant_smiles if value)
    gt_reactant_set = gt_reactants(trace_row)
    return {
        "exact_gt_reaction": bool(rxn and (canonical_reaction(rxn) or rxn) in gt_rxns),
        "gt_reactant_hit": bool(reactants & gt_reactant_set),
    }


def _child_quality(child: dict[str, Any], label: dict[str, Any]) -> float:
    failures = list(child.get("failure_categories") or [])
    penalties = {
        "StockDeadEnd": 0.60,
        "ConditionConflict": 0.55,
        "CofactorDebt": 0.45,
        "LowPlausibility": 0.35,
        "EnzymeEvidenceWeak": 0.25,
        "StageOverComplex": 0.25,
        "RouteOrderMismatch": 0.25,
    }
    score = 0.0
    score += 4.0 * float(bool(label.get("exact_gt_reaction")))
    score += 1.2 * float(bool(label.get("gt_reactant_hit")))
    score += 1.0 * float(bool(child.get("stock_closed")))
    score += 0.45 * float(bool(child.get("cofactor_closed")))
    score += 0.55 * float(not failures)
    score -= sum(penalties.get(str(failure), 0.20) for failure in failures)
    score -= 0.08 * len(child.get("open_leaves") or [])
    cost = _float_or_none(child.get("cascade_cost"))
    if cost is not None:
        score -= 0.04 * cost
    return float(score)


def _rank_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: (float(scores[idx]), -idx), reverse=True)


def _rxn(action: CascadeAction) -> str:
    return str(action.step.rxn_smiles if action.step is not None else "")


def _rate(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return round(float(num) / float(den), 6)


def _mean(values: list[Any]) -> float | None:
    clean = [_float_or_none(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None
    return round(float(statistics.mean(clean)), 6)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weight_token(weight: float) -> str:
    text = ("%g" % float(weight)).replace(".", "p").replace("-", "m")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay trace candidate pools with cascade pair-scorer rewards")
    ap.add_argument("--trace", required=True, help="Cascade search trace JSONL")
    ap.add_argument("--output", required=True, help="Output report JSON")
    ap.add_argument("--model", default=None, help="Learned CascadePairScorer checkpoint")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.35], help="Pair reward weights to test")
    ap.add_argument("--tie-epsilons", nargs="+", type=float, default=[], help="Optional guarded tie-break score windows.")
    ap.add_argument("--limit-events", type=int, default=None)
    ap.add_argument("--limit-targets", type=int, default=None)
    args = ap.parse_args()
    report = replay_cascade_pair_scorer(
        trace_path=Path(args.trace),
        output_path=Path(args.output),
        model_path=Path(args.model) if args.model else None,
        weights=args.weights,
        tie_epsilons=args.tie_epsilons,
        limit_events=args.limit_events,
        limit_targets=args.limit_targets,
    )
    print(
        json.dumps(
            {
                "metadata": report["metadata"],
                "all_events": report["all_events"],
                "pair_informative_events": report["pair_informative_events"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
