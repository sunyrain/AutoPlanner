"""Build a review worklist from strict model/control route-ranking disagreements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "strict_model_control_disagreement_review.v1"
REPORT_SCHEMA_VERSION = "strict_model_control_disagreement_review_report.v1"


def build_strict_model_review_worklist(
    *,
    value_pack: Path,
    model_pickle: Path,
    output_jsonl: Path,
    report_json: Path,
    output_csv: Path | None = None,
    max_rows: int = 120,
    max_per_target: int = 3,
    min_disagreement: int = 10,
    allow_model_rank: int = 5,
) -> dict[str, Any]:
    rows = _scored_rows(value_pack=value_pack, model_pickle=model_pickle)
    _attach_control_ranks(rows)
    candidates = _candidate_rows(rows, min_disagreement=min_disagreement, allow_model_rank=allow_model_rank)
    artifact_cache = _artifact_cache(row for _, _, _, row in candidates)
    selected = _select_review_rows(
        candidates,
        artifact_cache=artifact_cache,
        value_pack=value_pack,
        max_rows=max_rows,
        max_per_target=max_per_target,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, selected)
    if output_csv is not None:
        _write_csv(output_csv, selected)
    report = _report(
        value_pack=value_pack,
        model_pickle=model_pickle,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        report_json=report_json,
        candidate_rows=rows,
        selected=selected,
        max_rows=max_rows,
        max_per_target=max_per_target,
        min_disagreement=min_disagreement,
        allow_model_rank=allow_model_rank,
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _scored_rows(*, value_pack: Path, model_pickle: Path) -> list[dict[str, Any]]:
    with Path(model_pickle).open("rb") as fh:
        payload = pickle.load(fh)
    model = payload["model"]
    mean = payload["mean"]
    std = payload["std"]
    feature_names = payload["feature_names"]
    rows = []
    for row in _read_jsonl(value_pack):
        if int(row.get("n_steps") or 0) < 2:
            continue
        x = np.asarray([[_feature_value(row, name) for name in feature_names]], dtype=np.float32)
        copied = dict(row)
        copied["_model_score"] = float(model.decision_function((x - mean) / std)[0])
        copied["_retrieval_score"] = _retrieval_score(row)
        copied["_audit_score"] = _audit_score(row)
        copied["_native_score"] = -float(row.get("native_rank") or 0)
        rows.append(copied)
    return rows


def _attach_control_ranks(rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[_route_group(row)].append(row)
    for group_rows in by_group.values():
        for score_name, rank_name in (
            ("_model_score", "model_rank"),
            ("_retrieval_score", "retrieval_rank"),
            ("_audit_score", "audit_rank"),
            ("_native_score", "native_rank_control"),
        ):
            ordered = sorted(group_rows, key=lambda row: (-float(row[score_name]), int(row.get("native_rank") or 10**9)))
            for idx, row in enumerate(ordered, start=1):
                row[rank_name] = idx


def _candidate_rows(
    rows: list[dict[str, Any]],
    *,
    min_disagreement: int,
    allow_model_rank: int,
) -> list[tuple[int, int, int, dict[str, Any]]]:
    candidates = []
    for row in rows:
        model_rank = int(row.get("model_rank") or 9999)
        ranks = [
            int(row.get("retrieval_rank") or 9999),
            int(row.get("audit_rank") or 9999),
            int(row.get("native_rank_control") or 9999),
        ]
        disagreement = max(abs(model_rank - rank) for rank in ranks)
        if disagreement < int(min_disagreement) and model_rank > int(allow_model_rank):
            continue
        candidates.append((disagreement, -int(row.get("n_steps") or 0), model_rank, row))
    return candidates


def _select_review_rows(
    candidates: list[tuple[int, int, int, dict[str, Any]]],
    *,
    artifact_cache: dict[str, dict[str, list[dict[str, Any]]]],
    value_pack: Path,
    max_rows: int,
    max_per_target: int,
) -> list[dict[str, Any]]:
    selected = []
    seen_targets: Counter[str] = Counter()
    for _, _, _, row in sorted(candidates, key=lambda item: (-item[0], item[2], int(item[3].get("native_rank") or 9999))):
        if len(selected) >= int(max_rows):
            break
        target = str(row.get("target_smiles") or row.get("target_id") or "")
        if seen_targets[target] >= int(max_per_target):
            continue
        block = _choose_block(_artifact_route(row, artifact_cache))
        if not block:
            continue
        support_any = block.pop("_support_any", {})
        support_same_pair = block.pop("_support_same_pair", {})
        selected.append(_review_row(row, block=block, support_any=support_any, support_same_pair=support_same_pair, value_pack=value_pack))
        seen_targets[target] += 1
    return selected


def _review_row(
    row: dict[str, Any],
    *,
    block: dict[str, Any],
    support_any: dict[str, Any],
    support_same_pair: dict[str, Any],
    value_pack: Path,
) -> dict[str, Any]:
    tasks = row.get("weak_label_tasks") if isinstance(row.get("weak_label_tasks"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": _review_id(row, block.get("route_block_index")),
        "source_pool": "strict_runtime_train_provenance",
        "source_value_pack": str(value_pack),
        "evidence_class": "strict_model_control_disagreement",
        "target_id": row.get("target_id"),
        "target_smiles": row.get("target_smiles"),
        "route_id": row.get("route_id"),
        "value_split": row.get("split"),
        "native_rank": row.get("native_rank"),
        "native_score": row.get("native_score"),
        "stock_closed": bool(tasks.get("stock_closed")),
        "n_steps": row.get("n_steps"),
        "diagnostic_labels": {
            "model_top3": int(row.get("model_rank") or 9999) <= 3,
            "retrieval_top3": int(row.get("retrieval_rank") or 9999) <= 3,
            "audit_top3": int(row.get("audit_rank") or 9999) <= 3,
            "native_top3": int(row.get("native_rank_control") or 9999) <= 3,
            "reviewable_by_audit": bool(tasks.get("reviewable_by_audit")),
            "reject_artifact": bool(tasks.get("reject_artifact")),
        },
        "diagnostic_scores": {
            "model_score": row.get("_model_score"),
            "retrieval_score": row.get("_retrieval_score"),
            "audit_score": row.get("_audit_score"),
            "model_rank": row.get("model_rank"),
            "retrieval_rank": row.get("retrieval_rank"),
            "audit_rank": row.get("audit_rank"),
            "native_rank_control": row.get("native_rank_control"),
            "model_retrieval_rank_gap": abs(int(row.get("model_rank") or 9999) - int(row.get("retrieval_rank") or 9999)),
            "model_audit_rank_gap": abs(int(row.get("model_rank") or 9999) - int(row.get("audit_rank") or 9999)),
        },
        "review_block": block,
        "support_any": support_any,
        "support_same_pair": support_same_pair,
        "expert_route_plausible": None,
        "expert_block_transform_correct": None,
        "expert_support_precedent_relevant": None,
        "expert_cascade_coherent": None,
        "expert_priority": None,
        "expert_reject_reason": None,
        "expert_risk_tags": None,
        "expert_comments": None,
    }


def _artifact_cache(rows: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    paths = sorted({str(row.get("artifact_path") or "") for row in rows if row.get("artifact_path")})
    cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        target_index: dict[str, list[dict[str, Any]]] = {}
        for target in data.get("targets") or []:
            routes = target.get("routes") or ((target.get("planner_output") or {}).get("routes")) or []
            for key in (target.get("target_smiles"), target.get("target_id")):
                if key:
                    target_index[str(key)] = routes
        cache[path_text] = target_index
    return cache


def _artifact_route(row: dict[str, Any], artifact_cache: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any] | None:
    index = artifact_cache.get(str(row.get("artifact_path") or "")) or {}
    routes = index.get(str(row.get("target_smiles") or "")) or index.get(str(row.get("target_id") or ""))
    if not routes:
        return None
    idx = int(row.get("artifact_route_index") or row.get("native_rank") or 0)
    return routes[idx] if 0 <= idx < len(routes) else None


def _choose_block(route: dict[str, Any] | None) -> dict[str, Any] | None:
    steps = [step for step in (route or {}).get("steps") or [] if isinstance(step, dict)]
    if len(steps) < 2:
        return None
    best = None
    for idx in range(len(steps) - 1):
        upstream = steps[idx]
        downstream = steps[idx + 1]
        upstream_evidence = upstream.get("v4_step_evidence") if isinstance(upstream.get("v4_step_evidence"), dict) else {}
        downstream_evidence = downstream.get("v4_step_evidence") if isinstance(downstream.get("v4_step_evidence"), dict) else {}
        score = float(upstream_evidence.get("similarity") or 0.0) + float(downstream_evidence.get("similarity") or 0.0)
        candidate = (score, idx, upstream, downstream)
        if best is None or candidate > best:
            best = candidate
    _, idx, upstream, downstream = best
    return {
        "route_block_index": idx,
        "upstream_rxn": _step_rxn(upstream),
        "downstream_rxn": _step_rxn(downstream),
        "upstream_product": _first(upstream.get("products"), upstream.get("product_smiles") or upstream.get("product")),
        "downstream_product": _first(downstream.get("products"), downstream.get("product_smiles") or downstream.get("product")),
        "upstream_main_reactant": _first(upstream.get("reactants"), upstream.get("main_reactant")),
        "downstream_main_reactant": _first(downstream.get("reactants"), downstream.get("main_reactant")),
        "upstream_transform": _step_transform(upstream),
        "downstream_transform": _step_transform(downstream),
        "transform_pair": [_step_transform(upstream), _step_transform(downstream)],
        "pair_count_in_evidence": None,
        "pair_observed_in_evidence": None,
        "best_any_block_min_sim": None,
        "best_any_block_mean_sim": None,
        "best_same_pair_block_min_sim": None,
        "best_same_pair_block_mean_sim": None,
        "any_analog_supported": None,
        "same_pair_analog_supported": None,
        "_support_any": _step_support(upstream) or _step_support(downstream),
        "_support_same_pair": {},
    }


def _feature_value(row: dict[str, Any], name: str) -> float:
    group, key = name.split(".", 1)
    values = ((row.get("feature_groups") or {}).get(group) or {})
    try:
        return float(values.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retrieval_score(row: dict[str, Any]) -> float:
    group = ((row.get("feature_groups") or {}).get("cascade_retrieval") or {})
    return _first_available(
        group,
        [
            "ccts_v3_runtime_best_route_evidence",
            "ccts_v3_runtime_step_pair_max",
            "ccts_v3_runtime_step_any_max",
            "ccts_v3_runtime_step_pair_mean",
            "ccts_v3_runtime_step_any_mean",
        ],
    )


def _audit_score(row: dict[str, Any]) -> float:
    audit = row.get("product_audit") if isinstance(row.get("product_audit"), dict) else {}
    tasks = row.get("weak_label_tasks") if isinstance(row.get("weak_label_tasks"), dict) else {}
    return -float(audit.get("risk_order") or 0.0) + float(bool(tasks.get("material_sane_proxy")))


def _report(
    *,
    value_pack: Path,
    model_pickle: Path,
    output_jsonl: Path,
    output_csv: Path | None,
    report_json: Path,
    candidate_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    max_rows: int,
    max_per_target: int,
    min_disagreement: int,
    allow_model_rank: int,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"value_pack": str(value_pack), "model_pickle": str(model_pickle)},
        "outputs": {
            "jsonl": str(output_jsonl),
            "csv": str(output_csv) if output_csv else None,
            "report_json": str(report_json),
        },
        "parameters": {
            "max_rows": int(max_rows),
            "max_per_target": int(max_per_target),
            "min_disagreement": int(min_disagreement),
            "allow_model_rank": int(allow_model_rank),
        },
        "summary": {
            "candidate_multistep_rows": len(candidate_rows),
            "selected_rows": len(selected),
            "targets": len({row.get("target_smiles") for row in selected}),
            "classes": dict(Counter(row.get("evidence_class") for row in selected)),
            "model_top3": sum(1 for row in selected if (row.get("diagnostic_labels") or {}).get("model_top3")),
            "retrieval_top3": sum(1 for row in selected if (row.get("diagnostic_labels") or {}).get("retrieval_top3")),
            "audit_top3": sum(1 for row in selected if (row.get("diagnostic_labels") or {}).get("audit_top3")),
        },
        "contract": {
            "review_worklist_only": True,
            "not_training_labels": True,
            "selection_basis": "strict value-model vs retrieval/audit/native rank disagreement",
            "requires_real_review_before_training": True,
        },
        "examples": selected[:3],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_id",
        "source_pool",
        "source_value_pack",
        "evidence_class",
        "target_id",
        "target_smiles",
        "route_id",
        "value_split",
        "native_rank",
        "n_steps",
        "stock_closed",
        "model_rank",
        "retrieval_rank",
        "audit_rank",
        "native_rank_control",
        "model_score",
        "retrieval_score",
        "audit_score",
        "upstream_transform",
        "downstream_transform",
        "upstream_rxn",
        "downstream_rxn",
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_risk_tags",
        "expert_comments",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.update(row.get("diagnostic_scores") or {})
            block = row.get("review_block") or {}
            flat.update({key: block.get(key) for key in ("upstream_transform", "downstream_transform", "upstream_rxn", "downstream_rxn")})
            writer.writerow({key: flat.get(key) for key in fields})


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Strict Model-Control Disagreement Review Worklist", "", "| Metric | Value |", "|---|---:|"]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "This is a review worklist only, not training labels.", ""])
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _route_group(row: dict[str, Any]) -> str:
    return str(row.get("selector_group_id") or row.get("target_id") or "")


def _first_available(values: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        if key in values:
            try:
                return float(values.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _step_rxn(step: dict[str, Any]) -> str:
    return str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")


def _step_transform(step: dict[str, Any]) -> str:
    return str(step.get("transformation_superclass") or step.get("transformation_name") or step.get("reaction_type") or "")


def _step_support(step: dict[str, Any]) -> dict[str, Any]:
    evidence = step.get("v4_step_evidence") if isinstance(step.get("v4_step_evidence"), dict) else {}
    return {
        "program_id": evidence.get("program_id"),
        "doi": evidence.get("doi"),
        "transform_pair": None,
        "source_transform": evidence.get("source_transform"),
        "similarity": evidence.get("similarity"),
        "schema_version": evidence.get("schema_version"),
    }


def _first(values: Any, fallback: Any = "") -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return fallback or ""


def _review_id(row: dict[str, Any], block_idx: Any) -> str:
    text = f"strict_model_control_disagreement|{row.get('route_id')}|{block_idx}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a strict model-control disagreement review worklist")
    ap.add_argument("--value-pack", required=True)
    ap.add_argument("--model-pickle", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--output-csv")
    ap.add_argument("--max-rows", type=int, default=120)
    ap.add_argument("--max-per-target", type=int, default=3)
    ap.add_argument("--min-disagreement", type=int, default=10)
    ap.add_argument("--allow-model-rank", type=int, default=5)
    args = ap.parse_args()
    report = build_strict_model_review_worklist(
        value_pack=Path(args.value_pack),
        model_pickle=Path(args.model_pickle),
        output_jsonl=Path(args.output_jsonl),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        report_json=Path(args.report),
        max_rows=args.max_rows,
        max_per_target=args.max_per_target,
        min_disagreement=args.min_disagreement,
        allow_model_rank=args.allow_model_rank,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
