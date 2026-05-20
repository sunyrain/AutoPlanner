"""Replay a trained CCTS-v0 ranker on a fixed ChemEnzy candidate pool."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    _baseline_scores,
    _build_candidate_dataset,
    _build_evidence_bank,
    _feature_groups,
    _read_json,
    _write_jsonl,
)


REPLAY_SCHEMA_VERSION = "ccts_v0_transition_replay.v1"


def replay_ccts_v0_transition_ranker(
    *,
    model_bundle: Path,
    train_coverage: Path,
    coverage: Path,
    cache: Path,
    output: Path,
    report: Path,
    train_report: Path | None = None,
    blend_name: str | None = None,
    max_candidates_per_transition: int = 100,
    evidence_pool_size: int = 80,
    top_n: int = 10,
) -> dict[str, Any]:
    started = time.monotonic()
    bundle = _load_bundle(model_bundle)
    trained_models = bundle.get("models") or {}
    feature_schema = dict(bundle.get("feature_schema") or {})
    feature_names = list(feature_schema.get("feature_names") or [])
    bundle_model_specs = dict(feature_schema.get("model_specs") or {})
    schema = {key: value for key, value in feature_schema.items() if key not in {"feature_names", "chem_feature_names"}}
    train_payload = _read_json(train_coverage)
    payload = _read_json(coverage)
    cache_rows = _read_json(cache)
    evidence_bank = _build_evidence_bank([row for row in train_payload.get("transitions") or [] if isinstance(row, dict)])
    dataset = _build_candidate_dataset(
        [row for row in payload.get("transitions") or [] if isinstance(row, dict)],
        cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=float((payload.get("metadata") or {}).get("similarity_threshold") or 0.7),
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=False,
    )
    if not feature_names:
        feature_names = dataset.feature_names
    feature_index = {name: idx for idx, name in enumerate(dataset.feature_names)}
    feature_groups = _feature_groups(dataset.feature_names)
    train_report_payload = _read_json(train_report) if train_report else None
    blend_spec = _select_blend(train_report_payload, blend_name=blend_name)

    score_columns: dict[str, np.ndarray] = {
        "chem_rank": _baseline_scores(dataset.rows),
    }
    for model_name, model in trained_models.items():
        indices = _indices_for_model(model_name, feature_groups, feature_index, feature_names, bundle_model_specs=bundle_model_specs)
        if not indices:
            continue
        score_columns[model_name] = model.predict(dataset.x[:, indices], num_iteration=getattr(model, "best_iteration_", None))
    if blend_spec and blend_spec["base_model"] in score_columns and blend_spec["aux_model"] in score_columns:
        score_columns[blend_spec["name"]] = _standardize(score_columns[blend_spec["base_model"]]) + float(blend_spec["alpha"]) * _standardize(
            score_columns[blend_spec["aux_model"]]
        )
    selected_score = blend_spec["name"] if blend_spec and blend_spec["name"] in score_columns else _default_score_name(score_columns)

    transition_rows = _ranked_transition_rows(
        dataset,
        score_columns=score_columns,
        selected_score=selected_score,
        top_n=top_n,
    )
    metrics = {
        score_name: _ranking_metrics_from_replay(transition_rows, score_name)
        for score_name in score_columns
    }
    result = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "model_bundle": str(model_bundle),
            "train_report": str(train_report) if train_report else None,
            "train_coverage": str(train_coverage),
            "coverage": str(coverage),
            "cache": str(cache),
            "output": str(output),
            "report": str(report),
            "selected_score": selected_score,
            "blend_spec": blend_spec,
            "max_candidates_per_transition": max_candidates_per_transition,
            "evidence_pool_size": evidence_pool_size,
            "top_n": top_n,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "transitions": len(transition_rows),
            "candidate_rows": len(dataset.rows),
            "groups": len(dataset.group_sizes),
            "score_columns": sorted(score_columns),
        },
        "metrics": metrics,
        "outputs": {
            "jsonl": str(output),
            "report": str(report),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, transition_rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _ranked_transition_rows(
    dataset,
    *,
    score_columns: dict[str, np.ndarray],
    selected_score: str,
    top_n: int,
) -> list[dict[str, Any]]:
    out = []
    offset = 0
    for group_size, group_id in zip(dataset.group_sizes, dataset.group_ids):
        rows = dataset.rows[offset : offset + group_size]
        group_scores = {name: values[offset : offset + group_size] for name, values in score_columns.items()}
        order_by_score = {
            name: sorted(range(group_size), key=lambda idx: (-float(scores[idx]), int(rows[idx].get("candidate_rank") or 10**9)))
            for name, scores in group_scores.items()
        }
        selected_order = order_by_score[selected_score]
        chem_order = order_by_score["chem_rank"]
        rank_maps = {
            name: {idx: rank + 1 for rank, idx in enumerate(order)}
            for name, order in order_by_score.items()
        }
        best_positive = _best_label_rank(rows, rank_maps, "positive_label")
        best_exact = _best_label_rank(rows, rank_maps, "exact_label")
        out.append(
            {
                "transition_id": group_id,
                "target_smiles": rows[0].get("target_smiles") if rows else None,
                "product_smiles": rows[0].get("product_smiles") if rows else None,
                "route_domain": rows[0].get("route_domain") if rows else None,
                "candidate_count": group_size,
                "selected_score": selected_score,
                "chem_top_candidate": _candidate_summary_with_idx(rows[chem_order[0]], chem_order[0], group_scores, rank_maps) if chem_order else None,
                "selected_top_candidate": (
                    _candidate_summary_with_idx(rows[selected_order[0]], selected_order[0], group_scores, rank_maps)
                    if selected_order
                    else None
                ),
                "best_positive_rank": best_positive,
                "best_exact_rank": best_exact,
                "rank_delta": {
                    "positive_selected_minus_chem": _delta(best_positive, selected_score, "chem_rank"),
                    "exact_selected_minus_chem": _delta(best_exact, selected_score, "chem_rank"),
                },
                "top_candidates": [
                    _candidate_summary_with_idx(rows[idx], idx, group_scores, rank_maps)
                    for idx in selected_order[: max(1, int(top_n))]
                ],
            }
        )
        offset += group_size
    return out


def _candidate_summary_with_idx(row: dict[str, Any], idx: int, score_columns: dict[str, np.ndarray], rank_maps: dict[str, dict[int, int]]) -> dict[str, Any]:
    return {
        "candidate_rank": row.get("candidate_rank"),
        "candidate_score": row.get("candidate_score"),
        "candidate_source": row.get("candidate_source"),
        "candidate_model": row.get("candidate_model"),
        "candidate_type": row.get("candidate_type"),
        "candidate_main_reactant": row.get("candidate_main_reactant"),
        "candidate_reaction_smiles": row.get("candidate_reaction_smiles"),
        "exact_label": row.get("exact_label"),
        "similar_label": row.get("similar_label"),
        "positive_label": row.get("positive_label"),
        "reactant_similarity": row.get("reactant_similarity"),
        "scores": {name: round(float(scores[idx]), 6) for name, scores in score_columns.items()},
        "ranks": {name: int(ranks[idx]) for name, ranks in rank_maps.items()},
    }


def _best_label_rank(rows: list[dict[str, Any]], rank_maps: dict[str, dict[int, int]], label_name: str) -> dict[str, int | None]:
    out = {}
    for score_name, ranks in rank_maps.items():
        label_ranks = [ranks[idx] for idx, row in enumerate(rows) if row.get(label_name)]
        out[score_name] = min(label_ranks) if label_ranks else None
    return out


def _delta(rank_map: dict[str, int | None], selected_score: str, base_score: str) -> int | None:
    selected = rank_map.get(selected_score)
    base = rank_map.get(base_score)
    if selected is None or base is None:
        return None
    return int(selected) - int(base)


def _ranking_metrics_from_replay(rows: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    out = {}
    for label_name, rank_key in (("positive_label", "best_positive_rank"), ("exact_label", "best_exact_rank")):
        ranks = [(row.get(rank_key) or {}).get(score_name) for row in rows]
        covered_ranks = [int(rank) for rank in ranks if rank is not None]
        out[label_name] = {
            "coverage": round(len(covered_ranks) / max(len(rows), 1), 6),
            "mrr_covered": round(sum(1.0 / rank for rank in covered_ranks) / max(len(covered_ranks), 1), 6) if covered_ranks else 0.0,
            "recall_at_k_all": {
                str(k): round(sum(1 for rank in covered_ranks if rank <= k) / max(len(rows), 1), 6)
                for k in (1, 3, 5, 10, 20, 50)
            },
            "recall_at_k_covered": {
                str(k): round(sum(1 for rank in covered_ranks if rank <= k) / max(len(covered_ranks), 1), 6)
                for k in (1, 3, 5, 10, 20, 50)
            },
        }
    return out


def _indices_for_model(
    model_name: str,
    feature_groups: dict[str, list[int]],
    feature_index: dict[str, int],
    feature_names: list[str],
    *,
    bundle_model_specs: dict[str, Any] | None = None,
) -> list[int]:
    if bundle_model_specs and model_name in bundle_model_specs:
        return [
            feature_index[name]
            for name in bundle_model_specs.get(model_name) or []
            if name in feature_index
        ]
    if model_name == "chem_scalar":
        return feature_groups["chem_scalar"]
    if model_name == "chem_only":
        return [idx for idx, name in enumerate(feature_names) if name.startswith("chem__") and name in feature_index]
    if model_name == "chem_no_context":
        no_context = {
            "chem__target_heavy_atoms",
            "chem__product_target_similarity",
            "chem__step_pos",
            "chem__remaining_steps",
            "chem__has_previous_transform",
        }
        return [
            idx
            for idx, name in enumerate(feature_names)
            if name.startswith("chem__") and name not in no_context and name in feature_index
        ]
    if model_name == "context_evidence_only":
        return feature_groups["ccts_scalar"]
    if model_name == "chem_scalar_plus_context_evidence":
        return feature_groups["chem_scalar"] + feature_groups["ccts_scalar"]
    if model_name == "chem_no_context_plus_context_evidence":
        return _indices_for_model(
            "chem_no_context",
            feature_groups,
            feature_index,
            feature_names,
            bundle_model_specs=bundle_model_specs,
        ) + feature_groups["ccts_scalar"]
    if model_name == "ccts_evidence":
        return list(range(len(feature_names)))
    if model_name == "ccts_v1_full":
        return list(range(len(feature_names)))
    return list(range(len(feature_names)))


def _select_blend(report: dict[str, Any] | None, *, blend_name: str | None) -> dict[str, Any] | None:
    blends = (report or {}).get("blends") or {}
    if not blends:
        return None
    selected_name = blend_name
    if not selected_name:
        selected_name = max(blends, key=lambda name: float(blends[name].get("val_selection_score") or 0.0))
    row = blends.get(selected_name)
    if not row:
        return None
    return {
        "name": selected_name,
        "base_model": "chem_only",
        "aux_model": str(row.get("aux_model") or "").strip(),
        "alpha": float(row.get("alpha_selected_on_val") or 0.0),
        "val_selection_score": row.get("val_selection_score"),
    }


def _default_score_name(score_columns: dict[str, np.ndarray]) -> str:
    for name in ("chem_only", "chem_scalar", "ccts_evidence", "chem_rank"):
        if name in score_columns:
            return name
    return sorted(score_columns)[0]


def _standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    std = float(scores.std())
    if std < 1e-9:
        return scores * 0.0
    return (scores - float(scores.mean())) / std


def _load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v0 Replay",
        "",
        "## Metadata",
        "",
        f"- Selected score: `{(result.get('metadata') or {}).get('selected_score')}`",
        f"- Transitions: `{(result.get('counts') or {}).get('transitions')}`",
        f"- Candidate rows: `{(result.get('counts') or {}).get('candidate_rows')}`",
        "",
        "## Metrics",
        "",
        "| Score | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for score_name, report in (result.get("metrics") or {}).items():
        for label_name in ("positive_label", "exact_label"):
            metric = report.get(label_name) or {}
            at = metric.get("recall_at_k_all") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        score_name,
                        label_name,
                        str(metric.get("coverage")),
                        str(metric.get("mrr_covered")),
                        str(at.get("1")),
                        str(at.get("3")),
                        str(at.get("5")),
                        str(at.get("10")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Replay trained CCTS-v0 transition ranker")
    ap.add_argument("--model-bundle", required=True)
    ap.add_argument("--train-report")
    ap.add_argument("--train-coverage", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--blend-name")
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--evidence-pool-size", type=int, default=80)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()
    result = replay_ccts_v0_transition_ranker(
        model_bundle=Path(args.model_bundle),
        train_report=Path(args.train_report) if args.train_report else None,
        train_coverage=Path(args.train_coverage),
        coverage=Path(args.coverage),
        cache=Path(args.cache),
        output=Path(args.output),
        report=Path(args.report),
        blend_name=args.blend_name,
        max_candidates_per_transition=args.max_candidates_per_transition,
        evidence_pool_size=args.evidence_pool_size,
        top_n=args.top_n,
    )
    print(json.dumps({"counts": result["counts"], "selected_score": result["metadata"]["selected_score"], "metrics": result["metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
