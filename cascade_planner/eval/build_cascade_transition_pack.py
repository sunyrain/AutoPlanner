"""Build transition-value training data from cascade search traces.

Labels are process-progress signals derived from parent/child state deltas.
They deliberately avoid exact-GT-candidate and document-tier supervision.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
    gt_reactants,
    reaction_reactants,
)
from cascade_planner.cascade_search.transition_value import transition_reward
from cascade_planner.vnext.features import read_jsonl, stable_id, write_jsonl


def build_cascade_transition_pack(
    *,
    trace_paths: Iterable[Path],
    output_dir: Path,
    benchmark_path: Path | None = None,
    max_candidates: int = 64,
) -> dict[str, Any]:
    paths = [Path(path) for path in trace_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    for path in paths:
        trace_rows.extend(read_jsonl(path))
    gt_by_target = _gt_by_target(_read_benchmark_rows(benchmark_path)) if benchmark_path is not None else {}

    transition_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    label_bins: Counter[str] = Counter()
    failure_delta_bins: Counter[str] = Counter()

    for row in trace_rows:
        event = row.get("event")
        if not isinstance(event, dict):
            continue
        actions = list(event.get("candidate_actions") or [])[:max_candidates]
        children = list(event.get("child_summaries") or [])[: len(actions)]
        scores = list(event.get("candidate_scores") or [])
        if not actions or not children:
            continue
        state = event.get("state") or {}
        target = str(row.get("target_smiles") or state.get("target_smiles") or "")
        gt = gt_by_target.get(target) or {}
        pool_id = stable_id(
            target,
            event.get("state_id") or "",
            event.get("expanded_leaf") or "",
            row.get("benchmark") or "",
            row.get("benchmark_index") or "",
        )
        candidate_ids: list[str] = []
        values: list[float] = []
        for idx, (action, child) in enumerate(zip(actions, children), start=1):
            labels = transition_reward(state, child)
            fragment_value = _fragment_transition_value(action, gt)
            outcome = row.get("outcome") or event.get("outcome") or {}
            rollout_bonus = 0.10 if outcome.get("solved") else 0.0
            value = max(0.0, min(1.0, float(labels["transition_value"]) + rollout_bonus))
            labels["transition_value"] = value
            labels["fragment_transition_value"] = fragment_value
            labels["process_fragment_transition_value"] = round(max(value, fragment_value), 6)
            labels["fragment_rank_transition_value"] = round(
                max(fragment_value, 0.20 * value),
                6,
            )
            labels["exact_gt_reaction"] = int(fragment_value >= 0.85)
            labels["gt_reactant_hit"] = int(0.0 < fragment_value < 0.85 or fragment_value >= 0.85)
            label_bins[_value_bin(value)] += 1
            failure_delta_bins[str(int(labels.get("failure_reduction") or 0))] += 1
            candidate_id = stable_id(pool_id, idx, json.dumps(action, sort_keys=True, default=str))
            transition_rows.append(
                {
                    "transition_id": candidate_id,
                    "pool_id": pool_id,
                    "target_smiles": target,
                    "route_domain": row.get("route_domain"),
                    "benchmark": row.get("benchmark"),
                    "benchmark_index": row.get("benchmark_index"),
                    "doi": row.get("doi"),
                    "cascade_id": row.get("cascade_id"),
                    "state_id": event.get("state_id"),
                    "depth": event.get("depth"),
                    "expanded_leaf": event.get("expanded_leaf"),
                    "candidate_index": idx,
                    "parent_state": state,
                    "candidate_action": action,
                    "child_summary": child,
                    "provider_or_model_score": float(scores[idx - 1]) if idx - 1 < len(scores) else None,
                    "labels": labels,
                    "label_contract": "process_transition_delta.v1",
                }
            )
            candidate_ids.append(candidate_id)
            values.append(value)
        if candidate_ids:
            pool_rows.append(
                {
                    "pool_id": pool_id,
                    "target_smiles": target,
                    "state_id": event.get("state_id"),
                    "expanded_leaf": event.get("expanded_leaf"),
                    "candidate_ids": candidate_ids,
                    "transition_values": values,
                    "candidate_count": len(candidate_ids),
                    "best_transition_value": max(values),
                    "route_domain": row.get("route_domain"),
                    "benchmark": row.get("benchmark"),
                    "benchmark_index": row.get("benchmark_index"),
                }
            )

    transition_path = output_dir / "transition_value.jsonl"
    pool_path = output_dir / "transition_pools.jsonl"
    write_jsonl(transition_path, transition_rows)
    write_jsonl(pool_path, pool_rows)
    manifest = {
        "schema_version": "cascade_transition_pack.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_traces": [str(path) for path in paths],
        "benchmark_path": str(benchmark_path) if benchmark_path is not None else None,
        "label_contract": "process_transition_delta.v1",
        "max_candidates": max_candidates,
        "counts": {
            "trace_rows": len(trace_rows),
            "transition_rows": len(transition_rows),
            "transition_pools": len(pool_rows),
            "targets": len({row.get("target_smiles") for row in transition_rows if row.get("target_smiles")}),
        },
        "label_bins": dict(label_bins),
        "failure_delta_bins": dict(failure_delta_bins),
        "files": {
            "transition_value": str(transition_path),
            "transition_pools": str(pool_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.md"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def _value_bin(value: float) -> str:
    if value >= 0.8:
        return "0.8-1.0"
    if value >= 0.6:
        return "0.6-0.8"
    if value >= 0.4:
        return "0.4-0.6"
    if value >= 0.2:
        return "0.2-0.4"
    return "0.0-0.2"


def _gt_by_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        target = str(row.get("target_smiles") or "")
        if not target:
            continue
        out[target] = {
            "gt_rxns": {
                canonical_reaction(step.get("rxn_smiles") or "") or step.get("rxn_smiles")
                for step in row.get("gt_route") or []
                if isinstance(step, dict) and step.get("rxn_smiles")
            },
            "gt_reactants": gt_reactants(row),
        }
    return out


def _read_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("targets", "items", "rows"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    return read_jsonl(Path(path))


def _fragment_transition_value(action: dict[str, Any], gt: dict[str, Any]) -> float:
    if not gt:
        return 0.0
    step = action.get("step") if isinstance(action.get("step"), dict) else {}
    rxn = canonical_reaction(str(step.get("rxn_smiles") or "")) or str(step.get("rxn_smiles") or "")
    if rxn and rxn in (gt.get("gt_rxns") or set()):
        return 0.85
    reactants = {
        canonical_smiles(str(value)) or str(value)
        for value in step.get("reactant_smiles") or []
        if value
    }
    if rxn:
        reactants.update(reaction_reactants(rxn))
    if reactants & (gt.get("gt_reactants") or set()):
        return 0.40
    return 0.0


def _report_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    lines = [
        "# Cascade Transition Pack",
        "",
        f"- trace rows: `{counts.get('trace_rows', 0)}`",
        f"- transition rows: `{counts.get('transition_rows', 0)}`",
        f"- transition pools: `{counts.get('transition_pools', 0)}`",
        f"- targets: `{counts.get('targets', 0)}`",
        f"- label contract: `{manifest.get('label_contract')}`",
        f"- label bins: `{manifest.get('label_bins', {})}`",
        f"- failure delta bins: `{manifest.get('failure_delta_bins', {})}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build process-transition value pack from cascade search traces")
    ap.add_argument("--trace", action="append", required=True, help="Trace JSONL path. Repeat for multiple traces.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--max-candidates", type=int, default=64)
    args = ap.parse_args()
    manifest = build_cascade_transition_pack(
        trace_paths=[Path(path) for path in args.trace],
        output_dir=Path(args.output_dir),
        benchmark_path=Path(args.benchmark) if args.benchmark else None,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
