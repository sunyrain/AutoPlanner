"""Build state/action value supervision from ChemEnzy expansion traces.

This pack is intentionally not a cascade-record gold classifier.  It turns
internal ChemEnzy expansion events into value targets for choosing proposal
sources and ranking actions inside the multi-step search loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
    gt_reactants,
    reaction_reactants,
)


ACTION_SCHEMA = "cascade_action_value.v1"
SOURCE_SCHEMA = "cascade_source_value.v1"
ROUTE_GT_SIGNAL_FLOOR = 0.30
ROUTE_EXACT_SIGNAL_FLOOR = 0.60


def build_cascade_action_value_pack(
    *,
    trace_path: Path | None = None,
    trace_paths: list[Path] | None = None,
    benchmark_path: Path,
    output_dir: Path,
    runtime_path: Path | None = None,
    runtime_paths: list[Path] | None = None,
    val_fraction: float = 0.20,
    test_fraction: float = 0.0,
    preserve_benchmark_splits: bool = False,
) -> dict[str, Any]:
    paths = []
    if trace_path is not None:
        paths.append(trace_path)
    paths.extend(trace_paths or [])
    if not paths:
        raise ValueError("at least one trace path is required")
    benchmark_rows = _read_benchmark(benchmark_path)
    gt_by_target = _gt_by_target(benchmark_rows)
    runtime_file_paths = []
    if runtime_path is not None:
        runtime_file_paths.append(runtime_path)
    runtime_file_paths.extend(runtime_paths or [])
    route_outcome_by_target = (
        _route_outcomes_by_target(runtime_file_paths)
        if runtime_file_paths
        else {}
    )
    splits = (
        _target_splits_from_benchmark_rows(benchmark_rows)
        if preserve_benchmark_splits
        else _target_splits(sorted(gt_by_target), val_fraction=val_fraction, test_fraction=test_fraction)
    )
    if preserve_benchmark_splits and not splits:
        raise ValueError("preserve_benchmark_splits=True requires split fields in benchmark rows")

    action_rows: list[dict[str, Any]] = []
    source_aggs: dict[tuple[str, str], dict[str, Any]] = {}
    state_sources: dict[str, set[str]] = defaultdict(set)
    state_positive = Counter()
    summary = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_route_domain: dict[str, Counter[str]] = defaultdict(Counter)

    for path in paths:
        for raw in _iter_action_rows_from_path(path):
            target = str(raw.get("target_smiles") or "")
            gt = gt_by_target.get(target)
            if not gt:
                summary["skipped_unknown_target"] += 1
                continue
            state_id = _state_id(target, raw.get("parent_mol"), raw.get("parent_depth"))
            source = str(raw.get("source_model") or "unknown")
            route_domain = str(gt.get("route_domain") or "unknown")
            rxn_smiles = _candidate_reaction(raw)
            exact_hit = bool(rxn_smiles and rxn_smiles in gt["gt_rxns"])
            reactant_overlap = sorted(_candidate_reactants(raw) & gt["gt_reactants"])
            reactant_hit = bool(reactant_overlap)
            action_value = _action_value(exact_hit=exact_hit, reactant_hit=reactant_hit)
            route_outcome_labels = _route_outcome_labels(
                target=target,
                rxn_smiles=rxn_smiles,
                candidate_reactants=_candidate_reactants(raw),
                route_outcome_by_target=route_outcome_by_target,
            )
            split = splits.get(target, "train")

            action = {
                "schema_version": ACTION_SCHEMA,
                "supervision_contract": "internal_search_action_value_not_record_gold.v1",
                "trace_path": str(raw.get("trace_path") or path),
                "target_smiles": target,
                "split": split,
                "route_domain": route_domain,
                "state_id": state_id,
                "parent_mol": raw.get("parent_mol"),
                "parent_depth": raw.get("parent_depth"),
                "candidate_index": raw.get("candidate_index"),
                "source_model": source,
                "reaction_domain": raw.get("reaction_domain"),
                "reactants": list(raw.get("reactants") or []),
                "rxn_smiles": rxn_smiles,
                "base_score": _float_or_none(raw.get("base_score")),
                "base_cost": _float_or_none(raw.get("base_cost")),
                "cascade_adjustment": _float_or_none(raw.get("cascade_adjustment")),
                "total_cost": _float_or_none(raw.get("total_cost")),
                "components": raw.get("components") or {},
                "action_value_score": _float_or_none(raw.get("action_value_score")),
                "context_features": raw.get("context_features") or {},
                "source_policy_decision": raw.get("source_policy_decision") or {},
                "active_failure_modes": list(raw.get("active_failure_modes") or []),
                "labels": {
                    "exact_gt_reaction": int(exact_hit),
                    "gt_reactant_hit": int(reactant_hit),
                    "gt_reactant_overlap": reactant_overlap,
                    "action_value": action_value,
                    **route_outcome_labels,
                },
            }
            action["labels"]["cascade_fragment_action_value"] = _cascade_fragment_action_value(action["labels"])
            action["labels"]["state_action_value"] = _state_action_value(action["labels"])
            action_rows.append(action)
            state_sources[state_id].add(source)
            state_positive[state_id] += int(action_value > 0.0)

            agg = source_aggs.setdefault(
                (state_id, source),
                {
                    "schema_version": SOURCE_SCHEMA,
                    "supervision_contract": "internal_search_source_value_not_record_gold.v1",
                    "target_smiles": target,
                    "split": split,
                    "route_domain": route_domain,
                    "state_id": state_id,
                    "parent_mol": raw.get("parent_mol"),
                    "parent_depth": raw.get("parent_depth"),
                    "source_model": source,
                    "reaction_domain": raw.get("reaction_domain"),
                    "context_features": raw.get("context_features") or {},
                    "candidate_rows": 0,
                    "exact_gt_hits": 0,
                    "gt_reactant_hits": 0,
                    "best_action_value": 0.0,
                    "best_total_cost": None,
                    "labels": {},
                },
            )
            agg["candidate_rows"] += 1
            agg["exact_gt_hits"] += int(exact_hit)
            agg["gt_reactant_hits"] += int(reactant_hit)
            agg["best_action_value"] = max(float(agg["best_action_value"]), action_value)
            total_cost = _float_or_none(raw.get("total_cost"))
            if total_cost is not None:
                current = agg.get("best_total_cost")
                agg["best_total_cost"] = total_cost if current is None else min(float(current), total_cost)

            summary["action_rows"] += 1
            summary["exact_gt_hits"] += int(exact_hit)
            summary["gt_reactant_hits"] += int(reactant_hit)
            by_source[source]["action_rows"] += 1
            by_source[source]["exact_gt_hits"] += int(exact_hit)
            by_source[source]["gt_reactant_hits"] += int(reactant_hit)
            by_route_domain[route_domain]["action_rows"] += 1
            by_route_domain[route_domain]["exact_gt_hits"] += int(exact_hit)
            by_route_domain[route_domain]["gt_reactant_hits"] += int(reactant_hit)

    source_rows = []
    for row in source_aggs.values():
        source_value = _source_value(
            exact_hits=int(row["exact_gt_hits"]),
            reactant_hits=int(row["gt_reactant_hits"]),
        )
        row["state_source_count"] = len(state_sources[row["state_id"]])
        row["labels"] = {
            "source_value": source_value,
            "source_has_exact_gt_reaction": int(row["exact_gt_hits"] > 0),
            "source_has_gt_reactant": int(row["gt_reactant_hits"] > 0),
            "state_has_positive_action": int(state_positive[row["state_id"]] > 0),
        }
        source_rows.append(row)

    if not preserve_benchmark_splits:
        splits = _stratify_splits_by_positive_targets(
            action_rows,
            base_splits=splits,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    for row in action_rows:
        row["split"] = splits.get(str(row.get("target_smiles") or ""), row.get("split") or "train")
    for row in source_rows:
        row["split"] = splits.get(str(row.get("target_smiles") or ""), row.get("split") or "train")

    source_value_by_key = {
        (row["state_id"], row["source_model"]): row["labels"]["source_value"]
        for row in source_rows
    }
    state_positive_by_id = {state_id: int(count > 0) for state_id, count in state_positive.items()}
    for row in action_rows:
        row["labels"]["state_has_positive_action"] = state_positive_by_id.get(row["state_id"], 0)
        row["labels"]["source_value"] = source_value_by_key.get((row["state_id"], row["source_model"]), 0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    action_path = output_dir / "action_value.jsonl"
    source_path = output_dir / "source_value.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    _write_jsonl(action_path, action_rows)
    _write_jsonl(source_path, source_rows)

    report = {
        "metadata": {
            "trace_path": str(paths[0]),
            "trace_paths": [str(path) for path in paths],
            "benchmark_path": str(benchmark_path),
            "runtime_path": str(runtime_file_paths[0]) if runtime_file_paths else None,
            "runtime_paths": [str(path) for path in runtime_file_paths],
            "output_dir": str(output_dir),
            "action_schema": ACTION_SCHEMA,
            "source_schema": SOURCE_SCHEMA,
            "n_benchmark_targets": len(gt_by_target),
            "n_states": len(state_sources),
            "n_action_rows": len(action_rows),
            "n_source_rows": len(source_rows),
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_policy": (
                "preserve_benchmark_split_fields"
                if preserve_benchmark_splits
                else "target_stratified_by_route_quality_positive_when_available"
            ),
            "preserve_benchmark_splits": preserve_benchmark_splits,
            "training_caution": (
                "Use benchmark-derived packs for diagnosis only. Train production models on traces "
                "collected from a training split, then evaluate on held-out benchmark targets."
            ),
        },
        "summary": _rate_summary(summary),
        "by_source": {key: _rate_summary(value) for key, value in sorted(by_source.items())},
        "by_route_domain": {key: _rate_summary(value) for key, value in sorted(by_route_domain.items())},
        "splits": dict(Counter(splits.values())),
        "outputs": {
            "action_value": str(action_path),
            "source_value": str(source_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(_readme(report), encoding="utf-8")
    return report


def _gt_by_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        target = str(row.get("target_smiles") or "")
        if not target:
            continue
        gt_rxns = {
            canonical_reaction(step.get("rxn_smiles") or "") or step.get("rxn_smiles")
            for step in row.get("gt_route") or []
            if step.get("rxn_smiles")
        }
        out[target] = {
            "route_domain": row.get("route_domain"),
            "gt_rxns": gt_rxns,
            "gt_reactants": gt_reactants(row),
        }
    return out


def _target_splits(targets: list[str], *, val_fraction: float, test_fraction: float) -> dict[str, str]:
    targets = sorted(targets)
    n = len(targets)
    if n == 0:
        return {}
    n_test = max(0, int(round(n * max(0.0, test_fraction))))
    n_val = max(0, int(round(n * max(0.0, val_fraction))))
    if n > 1 and val_fraction > 0.0:
        n_val = max(1, n_val)
    if n_val + n_test >= n:
        n_val = max(0, min(n - 1, n_val))
        n_test = max(0, min(n - 1 - n_val, n_test))
    out = {}
    for idx, target in enumerate(targets):
        if idx < n_test:
            out[target] = "test"
        elif idx < n_test + n_val:
            out[target] = "val"
        else:
            out[target] = "train"
    return out


def _target_splits_from_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        target = str(row.get("target_smiles") or "")
        split = str(row.get("split") or "").strip().lower()
        if not target or not split:
            continue
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported benchmark split '{split}' for target {target}")
        conflicts[target].add(split)
    bad = {target: sorted(splits) for target, splits in conflicts.items() if len(splits) > 1}
    if bad:
        examples = dict(sorted(bad.items())[:10])
        raise ValueError(f"benchmark split conflicts for targets: {examples}")
    for target, splits in conflicts.items():
        out[target] = next(iter(splits))
    return out


def _stratify_splits_by_positive_targets(
    action_rows: list[dict[str, Any]],
    *,
    base_splits: dict[str, str],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    positive_targets = sorted(
        {
            str(row.get("target_smiles") or "")
            for row in action_rows
            if str(row.get("target_smiles") or "")
            and float((row.get("labels") or {}).get("route_quality_action_value") or 0.0) > 0.0
        }
    )
    if len(positive_targets) < 2 or (val_fraction <= 0.0 and test_fraction <= 0.0):
        return dict(base_splits)

    n_pos = len(positive_targets)
    n_test = max(0, int(round(n_pos * max(0.0, test_fraction))))
    n_val = max(0, int(round(n_pos * max(0.0, val_fraction))))
    if val_fraction > 0.0:
        n_val = max(1, n_val)
    if test_fraction > 0.0:
        n_test = max(1, n_test)
    if n_val + n_test >= n_pos:
        n_val = max(0, min(n_pos - 1, n_val))
        n_test = max(0, min(n_pos - 1 - n_val, n_test))

    ordered = _stable_target_order(positive_targets, salt="route_quality")
    test_targets = set(ordered[:n_test])
    val_targets = set(ordered[n_test:n_test + n_val])
    out = dict(base_splits)
    for target in positive_targets:
        out[target] = "train"
    for target in test_targets:
        out[target] = "test"
    for target in val_targets:
        out[target] = "val"
    return out


def _stable_target_order(targets: list[str], *, salt: str) -> list[str]:
    return sorted(targets, key=lambda target: hashlib.sha1(f"{salt}\t{target}".encode("utf-8")).hexdigest())


def _candidate_reaction(row: dict[str, Any]) -> str:
    rxn_smiles = str(row.get("rxn_smiles") or "")
    if rxn_smiles:
        return canonical_reaction(rxn_smiles) or rxn_smiles
    parent = str(row.get("parent_mol") or "")
    reactants = [str(item) for item in row.get("reactants") or [] if item]
    if not parent or not reactants:
        return ""
    rxn = ".".join(reactants) + ">>" + parent
    return canonical_reaction(rxn) or rxn


def _candidate_reactants(row: dict[str, Any]) -> set[str]:
    out = set()
    for item in row.get("reactants") or []:
        key = canonical_smiles(str(item)) or str(item)
        if key:
            out.add(key)
    return out


def _iter_action_rows_from_path(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("targets"), list):
            yield from _iter_action_rows_from_native_pool_json(data, path)
            return
    for raw in _read_jsonl(path):
        event = raw.get("event")
        if isinstance(event, dict) and isinstance(event.get("candidate_actions"), list):
            yield from _flatten_cascade_trace_event(raw, path)
        else:
            yield {**raw, "trace_path": str(path)}


def _iter_action_rows_from_native_pool_json(data: dict[str, Any], path: Path):
    for target in data.get("targets") or []:
        if not isinstance(target, dict):
            continue
        metadata = target.get("raw_backend_metadata") if isinstance(target.get("raw_backend_metadata"), dict) else {}
        trace = metadata.get("cascade_expansion_trace") if isinstance(metadata.get("cascade_expansion_trace"), dict) else {}
        rows = trace.get("rows") if isinstance(trace.get("rows"), list) else None
        if rows is None:
            rows = trace.get("preview") if isinstance(trace.get("preview"), list) else []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            yield {
                **raw,
                "trace_path": str(path),
                "target_smiles": target.get("target_smiles"),
                "route_domain": target.get("route_domain"),
                "gt_route": target.get("gt_route") or [],
                "benchmark_index": target.get("benchmark_index"),
                "cascade_id": target.get("cascade_id"),
                "doi": target.get("doi"),
            }


def _flatten_cascade_trace_event(raw: dict[str, Any], path: Path):
    event = raw.get("event") or {}
    actions = list(event.get("candidate_actions") or [])
    scores = list(event.get("candidate_scores") or [])
    children = list(event.get("child_summaries") or [])
    state = event.get("state") if isinstance(event.get("state"), dict) else {}
    route_domain = raw.get("route_domain") or state.get("route_domain")
    active_failure_modes = []
    for item in state.get("unresolved_failure_modes") or []:
        if isinstance(item, dict):
            active_failure_modes.append(str(item.get("kind") or item.get("category") or item))
        else:
            active_failure_modes.append(str(item))
    failure_categories = list(event.get("failure_categories") or [])
    if not active_failure_modes:
        active_failure_modes = [str(item) for item in failure_categories]

    for idx, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            continue
        step = action.get("step") if isinstance(action.get("step"), dict) else {}
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        child = children[idx - 1] if idx - 1 < len(children) and isinstance(children[idx - 1], dict) else {}
        raw_metadata = step.get("raw_metadata") if isinstance(step.get("raw_metadata"), dict) else {}
        base_score = step.get("score")
        if base_score is None and idx - 1 < len(scores):
            base_score = scores[idx - 1]
        yield {
            "trace_path": str(path),
            "target_smiles": raw.get("target_smiles") or state.get("target_smiles"),
            "route_domain": route_domain,
            "parent_mol": event.get("expanded_leaf") or action.get("target_leaf") or step.get("product_smiles"),
            "parent_depth": event.get("depth"),
            "candidate_index": idx,
            "source_model": step.get("source_model") or action.get("source") or "unknown",
            "reaction_domain": step.get("reaction_type") or step.get("reaction_domain"),
            "reactants": list(step.get("reactant_smiles") or step.get("reactants") or []),
            "rxn_smiles": step.get("rxn_smiles") or step.get("reaction_smiles"),
            "base_score": base_score,
            "base_cost": raw_metadata.get("cost"),
            "cascade_adjustment": raw_metadata.get("cascade_adjustment"),
            "total_cost": raw_metadata.get("total_cost"),
            "cascade_adjustment_source": "cascade_trace_candidate_action",
            "components": {
                "provider_rank": metadata.get("provider_rank"),
                "transition_value_score": metadata.get("transition_value_score"),
                "transition_value_rank": metadata.get("transition_value_rank"),
                "candidate_selection_status": metadata.get("candidate_selection_status"),
                "stock_closed_after_action": child.get("stock_closed"),
                "cofactor_closed_after_action": child.get("cofactor_closed"),
                "child_stage_count": child.get("stage_count"),
                "child_step_count": child.get("step_count"),
                "child_failure_categories": child.get("failure_categories") or [],
                "proposal_provider": action.get("source"),
            },
            "action_value_score": metadata.get("transition_value_score"),
            "context_features": {
                "depth": event.get("depth"),
                "open_leaf_count": len(event.get("open_leaves") or []),
                "candidate_count": len(actions),
                "failure_count": len(failure_categories),
                "model_active": event.get("model_active"),
                "parent_step_count": len(state.get("step_annotations") or state.get("steps") or []),
                "parent_stage_count": len((state.get("stage_graph") or {}).get("stages") or []),
            },
            "source_policy_decision": {
                "provider_rank": metadata.get("provider_rank"),
                "candidate_selection_status": metadata.get("candidate_selection_status"),
                "enqueued_from_state": metadata.get("enqueued_from_state"),
            },
            "active_failure_modes": active_failure_modes,
        }


def _action_value(*, exact_hit: bool, reactant_hit: bool) -> float:
    if exact_hit:
        return 1.0
    if reactant_hit:
        return 0.35
    return 0.0


def _source_value(*, exact_hits: int, reactant_hits: int) -> float:
    if exact_hits > 0:
        return 1.0
    if reactant_hits > 0:
        return 0.50
    return 0.0


def _cascade_fragment_action_value(labels: dict[str, Any]) -> float:
    """Blend route-outcome supervision with direct GT fragment recovery.

    ``route_quality_action_value`` says that a candidate appeared in a good
    route produced by the current search. That is useful, but by itself it can
    over-imitate the current planner. The fragment term keeps direct GT
    reaction/reactant evidence alive for candidates that were generated but not
    selected into the final route.
    """
    route_quality = _float_or_none(labels.get("route_quality_action_value")) or 0.0
    exact = 1.0 if int(labels.get("exact_gt_reaction") or 0) > 0 else 0.0
    reactant = 1.0 if int(labels.get("gt_reactant_hit") or 0) > 0 else 0.0
    route_outcome = _float_or_none(labels.get("route_outcome_action_value")) or 0.0
    value = max(
        float(route_quality),
        0.85 * exact,
        0.40 * reactant,
        0.50 * float(route_outcome),
    )
    return round(max(0.0, min(1.0, value)), 6)


def _state_action_value(labels: dict[str, Any]) -> float:
    """Q(S,a) target for search ordering.

    This target is broader than the local fragment scorer: it rewards direct GT
    reaction/reactant evidence, but also preserves route-outcome supervision
    when the current search later turns an action into a useful result.  It is
    intentionally an action value inside a state, not a cascade-record label.
    """
    route_quality = _float_or_none(labels.get("route_quality_action_value")) or 0.0
    route_outcome = _float_or_none(labels.get("route_outcome_action_value")) or 0.0
    exact = 1.0 if int(labels.get("exact_gt_reaction") or 0) > 0 else 0.0
    reactant = 1.0 if int(labels.get("gt_reactant_hit") or 0) > 0 else 0.0
    value = max(
        float(route_quality),
        0.90 * exact,
        0.45 * reactant,
        0.75 * float(route_outcome),
    )
    return round(max(0.0, min(1.0, value)), 6)


def _route_outcome_labels(
    *,
    target: str,
    rxn_smiles: str,
    candidate_reactants: set[str],
    route_outcome_by_target: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    outcomes = route_outcome_by_target.get(target)
    if not outcomes:
        return {
            "program_route_action": 0,
            "program_reactant_action": 0,
            "route_outcome_action_value": 0.0,
            "program_gt_signal_action": 0,
            "route_quality_exact_action": 0,
            "route_quality_reactant_action": 0,
            "route_quality_action_value": 0.0,
        }
    rxn_values = outcomes.get("rxn_values") or {}
    reactant_values = outcomes.get("reactant_values") or {}
    route_value = float(rxn_values.get(rxn_smiles) or 0.0)
    reactant_value = 0.0
    for reactant in candidate_reactants:
        reactant_value = max(reactant_value, float(reactant_values.get(reactant) or 0.0))
    quality_route_value = route_value if route_value >= ROUTE_GT_SIGNAL_FLOOR else 0.0
    quality_reactant_value = reactant_value if reactant_value >= ROUTE_GT_SIGNAL_FLOOR else 0.0
    quality_value = max(quality_route_value, quality_reactant_value * 0.5)
    return {
        "program_route_action": int(route_value > 0.0),
        "program_reactant_action": int(reactant_value > 0.0),
        "route_outcome_action_value": round(max(route_value, reactant_value * 0.5), 6),
        "program_gt_signal_action": int(quality_value > 0.0),
        "route_quality_exact_action": int(route_value >= ROUTE_EXACT_SIGNAL_FLOOR),
        "route_quality_reactant_action": int(quality_reactant_value > 0.0),
        "route_quality_action_value": round(quality_value, 6),
    }


def _route_outcomes_by_target(runtime_paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for runtime_path in runtime_paths:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
        targets = data.get("targets") if isinstance(data, dict) else data
        for row in targets or []:
            if not isinstance(row, dict):
                continue
            target = str(row.get("target_smiles") or "")
            if not target:
                continue
            target_out = out.setdefault(target, {"rxn_values": {}, "reactant_values": {}})
            rxn_values = target_out["rxn_values"]
            reactant_values = target_out["reactant_values"]
            programs = ((row.get("cascade_search") or {}).get("result_programs") or [])
            if not programs:
                programs = _fallback_programs_from_target(row)
            for program in programs:
                value = _float_or_none(program.get("route_outcome_value"))
                if value is None:
                    value = _fallback_program_value(program, row)
                if value <= 0.0:
                    continue
                for rxn in program.get("route_rxns") or []:
                    key = canonical_reaction(str(rxn)) or str(rxn)
                    if key:
                        rxn_values[key] = max(rxn_values.get(key, 0.0), float(value))
                        for reactant in reaction_reactants(key):
                            reactant_values[reactant] = max(
                                reactant_values.get(reactant, 0.0),
                                float(value),
                            )
                for reactant in program.get("route_reactants") or []:
                    key = canonical_smiles(str(reactant)) or str(reactant)
                    if key:
                        reactant_values[key] = max(reactant_values.get(key, 0.0), float(value))
    return out


def _fallback_programs_from_target(row: dict[str, Any]) -> list[dict[str, Any]]:
    routes = row.get("routes")
    if isinstance(routes, list) and routes:
        programs = []
        for idx, route in enumerate(routes, start=1):
            if not isinstance(route, dict):
                continue
            steps = route.get("steps") if isinstance(route.get("steps"), list) else []
            route_rxns = []
            route_reactants = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("rxn_smiles"):
                    route_rxns.append(step.get("rxn_smiles"))
                for reactant in step.get("reactant_smiles") or []:
                    route_reactants.append(reactant)
            if route_rxns or route_reactants:
                programs.append(
                    {
                        "rank": idx,
                        "solved": bool(route.get("solved") or row.get("solved")),
                        "route_rxns": route_rxns,
                        "route_reactants": route_reactants,
                        "route_outcome_value": 0.10 if bool(route.get("solved") or row.get("solved")) else 0.0,
                    }
                )
        if programs:
            return programs

    cascade = row.get("cascade_search") or {}
    recovery = row.get("recovery") or {}
    return [
        {
            "rank": 1,
            "solved": bool(cascade.get("solved")),
            "route_rxns": list(cascade.get("route_rxns") or []),
            "route_reactants": [],
            "exact_gt_route_recovered": bool(recovery.get("exact_gt_route_recovered")),
            "partial_gt_step_overlap": bool(recovery.get("partial_gt_step_overlap")),
            "gt_reactant_in_route": bool(recovery.get("gt_reactant_in_route_pool")),
        }
    ]


def _fallback_program_value(program: dict[str, Any], target_row: dict[str, Any]) -> float:
    recovery = target_row.get("recovery") or {}
    if program.get("exact_gt_route_recovered") or recovery.get("exact_gt_route_recovered"):
        return 1.0
    if program.get("partial_gt_step_overlap") or recovery.get("partial_gt_step_overlap"):
        return 0.8
    if program.get("gt_reactant_in_route") or recovery.get("gt_reactant_in_route_pool"):
        return 0.45
    if program.get("solved"):
        return 0.10
    return 0.0


def _state_id(target: str, parent_mol: Any, parent_depth: Any) -> str:
    key = f"{target}\t{parent_depth}\t{parent_mol}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _rate_summary(counter: Counter[str] | dict[str, int]) -> dict[str, Any]:
    rows = int(counter.get("action_rows") or 0)
    exact = int(counter.get("exact_gt_hits") or 0)
    reactant = int(counter.get("gt_reactant_hits") or 0)
    return {
        "action_rows": rows,
        "exact_gt_hits": exact,
        "gt_reactant_hits": reactant,
        "exact_gt_hit_rate": round(exact / rows, 6) if rows else 0.0,
        "gt_reactant_hit_rate": round(reactant / rows, 6) if rows else 0.0,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict) and row.get("target_smiles")]


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _readme(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Cascade Action Value Pack",
        "",
        "This pack supervises internal search decisions from ChemEnzy expansion traces.",
        "It is not a cascade-record gold classifier.",
        "",
        "## Summary",
        "",
        f"- action_rows: {summary['action_rows']}",
        f"- exact_gt_hits: {summary['exact_gt_hits']}",
        f"- gt_reactant_hits: {summary['gt_reactant_hits']}",
        f"- exact_gt_hit_rate: {summary['exact_gt_hit_rate']}",
        f"- gt_reactant_hit_rate: {summary['gt_reactant_hit_rate']}",
        "",
        "## Files",
        "",
        f"- action_value: {report['outputs']['action_value']}",
        f"- source_value: {report['outputs']['source_value']}",
        f"- summary: {report['outputs']['summary']}",
        "",
        "## Training Caution",
        "",
        report["metadata"]["training_caution"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cascade action/source value pack from ChemEnzy trace")
    ap.add_argument("--trace", required=True, action="append")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runtime", default=None, action="append")
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--test-fraction", type=float, default=0.0)
    ap.add_argument("--preserve-benchmark-splits", action="store_true")
    args = ap.parse_args()
    report = build_cascade_action_value_pack(
        trace_paths=[Path(path) for path in args.trace],
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        runtime_paths=[Path(path) for path in args.runtime or []],
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        preserve_benchmark_splits=args.preserve_benchmark_splits,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
