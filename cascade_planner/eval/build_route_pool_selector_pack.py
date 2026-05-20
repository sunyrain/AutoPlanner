"""Build route-level selector packs from ChemEnzy/Web route artifacts."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from cascade_planner.eval.product_route_feasibility_audit import (
    ROUTE_CLASS_ORDER,
    build_product_route_feasibility_audit,
    product_audit_risk_order,
)


PACK_SCHEMA_VERSION = "route_pool_selector_pack.v1"


def build_route_pool_selector_pack(
    *,
    inputs: list[Path],
    output_jsonl: Path,
    report_json: Path,
    split_manifest: Path,
    dataset: str = "route_pool_selector",
    split_strategy: str = "target_hash",
    deduplicate_routes: bool = True,
    evidence_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _expand_input_paths(inputs)
    rows: list[dict[str, Any]] = []
    target_split: dict[str, str] = {}
    artifact_summaries = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact_rows = []
        for target in _targets_from_payload(payload, path=path):
            split_key = _target_split_key(target)
            split = target_split.setdefault(split_key, _split_for_key(split_key, strategy=split_strategy, path=path))
            route_rows = _rows_for_target(
                target,
                artifact_path=path,
                dataset=dataset,
                split=split,
                evidence_provenance=evidence_provenance or {},
            )
            artifact_rows.extend(route_rows)
            rows.extend(route_rows)
        artifact_summaries.append(
            {
                "path": str(path),
                "objective": payload.get("objective"),
                "routes": len(artifact_rows),
                "targets": len({row["selector_group_id"] for row in artifact_rows}),
            }
        )
    raw_row_count = len(rows)
    if deduplicate_routes:
        rows = _deduplicate_rows(rows)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, rows)

    manifest = {
        "schema_version": "route_pool_selector_split_manifest.v1",
        "generated_at": _now(),
        "dataset": dataset,
        "split_strategy": split_strategy,
        "target_splits": dict(sorted(target_split.items())),
        "split_counts": dict(Counter(row.get("split") for row in rows)),
    }
    split_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": _now(),
        "dataset": dataset,
        "inputs": [str(path) for path in paths],
        "outputs": {
            "pack": str(output_jsonl),
            "report": str(report_json),
            "split_manifest": str(split_manifest),
        },
        "counts": {
            "artifacts": len(paths),
            "raw_rows": raw_row_count,
            "rows": len(rows),
            "deduplicated_rows_removed": raw_row_count - len(rows),
            "targets": len({row["selector_group_id"] for row in rows}),
        },
        "split_counts": dict(Counter(row.get("split") for row in rows)),
        "route_class_counts": dict(Counter(row.get("product_audit_class") for row in rows)),
        "issue_counts": _issue_counts(rows),
        "artifact_summaries": artifact_summaries,
        "deduplicate_routes": deduplicate_routes,
        "evidence_provenance": evidence_provenance or {},
        "training_contract": (
            "RouteSelector-v0 pack: use product-audit class/risk as ordinal guard labels, "
            "native ChemEnzy score/rank as features, and v4/cascade evidence as features when present."
        ),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _targets_from_payload(payload: dict[str, Any], *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload.get("targets"), list):
        out = []
        for idx, target in enumerate(payload.get("targets") or []):
            if not isinstance(target, dict):
                continue
            routes = (target.get("planner_output") or {}).get("routes")
            if not isinstance(routes, list):
                routes = target.get("routes")
            out.append(
                {
                    "target_index": target.get("index", idx),
                    "target_id": target.get("target_id") or target.get("cascade_id") or target.get("name") or idx,
                    "target_smiles": target.get("target_smiles") or target.get("target"),
                    "routes": [route for route in routes or [] if isinstance(route, dict)],
                    "artifact_objective": payload.get("objective") or (payload.get("metadata") or {}).get("objective"),
                    "artifact_type": _artifact_type(payload, path),
                    "post_filter": payload.get("post_filter") or {},
                }
            )
        return out

    routes = payload.get("routes") if isinstance(payload.get("routes"), list) else []
    target_id = (
        payload.get("target_id")
        or (payload.get("ui_metadata") or {}).get("target_id")
        or path.stem
    )
    return [
        {
            "target_index": 0,
            "target_id": target_id,
            "target_smiles": payload.get("target") or payload.get("target_smiles"),
            "routes": [route for route in routes if isinstance(route, dict)],
            "artifact_objective": payload.get("objective"),
            "artifact_type": _artifact_type(payload, path),
            "post_filter": payload.get("post_filter") or {},
        }
    ]


def _rows_for_target(
    target: dict[str, Any],
    *,
    artifact_path: Path,
    dataset: str,
    split: str,
    evidence_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    routes = list(target.get("routes") or [])
    if not routes:
        return []
    target_smiles = str(target.get("target_smiles") or "")
    target_id = str(target.get("target_id") or target.get("target_index") or "")
    audit_by_rank = _audit_by_native_rank(target_id=target_id, target_smiles=target_smiles, routes=routes)
    rows = []
    for native_rank, route in enumerate(routes):
        audit = dict(route.get("product_audit") or audit_by_rank.get(native_rank) or {})
        if not audit.get("route_class") and native_rank in audit_by_rank:
            audit = dict(audit_by_rank[native_rank])
        row = _row_from_route(
            route,
            audit=audit,
            target=target,
            artifact_path=artifact_path,
            dataset=dataset,
            split=split,
            native_rank=native_rank,
            evidence_provenance=evidence_provenance,
        )
        rows.append(row)
    return rows


def _audit_by_native_rank(*, target_id: str, target_smiles: str, routes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    run = {
        "targets": [
            {
                "index": 0,
                "target_id": target_id,
                "target_smiles": target_smiles,
                "planner_output": {"routes": routes},
                "metrics": {
                    "strict_stock_solve_any": any(
                        bool((route.get("metrics") or {}).get("strict_stock_solve"))
                        for route in routes
                        if isinstance(route, dict)
                    )
                },
            }
        ]
    }
    audit = build_product_route_feasibility_audit(run)
    audit_target = (audit.get("targets") or [{}])[0]
    return {
        int(row.get("rank") or 0) - 1: row
        for row in audit_target.get("routes") or []
        if row.get("rank") is not None
    }


def _row_from_route(
    route: dict[str, Any],
    *,
    audit: dict[str, Any],
    target: dict[str, Any],
    artifact_path: Path,
    dataset: str,
    split: str,
    native_rank: int,
    evidence_provenance: dict[str, Any],
) -> dict[str, Any]:
    target_smiles = str(target.get("target_smiles") or "")
    target_id = str(target.get("target_id") or target.get("target_index") or "")
    metrics = dict(route.get("metrics") or {})
    if "strict_stock_solve" not in metrics and route.get("stock_closed") is not None:
        metrics["strict_stock_solve"] = bool(route.get("stock_closed"))
    if "route_solved" not in metrics and route.get("solved") is not None:
        metrics["route_solved"] = bool(route.get("solved"))
    if "terminal_reactants" not in metrics and isinstance(route.get("terminal_reactants"), list):
        metrics["terminal_reactants"] = list(route.get("terminal_reactants") or [])
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    condition_scores = _condition_scores(steps)
    enzyme_scores = _enzyme_scores(steps)
    source_counts = Counter(str(step.get("source") or step.get("source_model") or "unknown") for step in steps)
    reaction_type_counts = Counter(str(step.get("reaction_type") or "unknown") for step in steps)
    plausibility = audit.get("route_plausibility") if isinstance(audit.get("route_plausibility"), dict) else {}
    reaction_profile = audit.get("reaction_profile") if isinstance(audit.get("reaction_profile"), dict) else {}
    terminal_profile = audit.get("terminal_profile") if isinstance(audit.get("terminal_profile"), dict) else {}
    route_class = str(audit.get("route_class") or "audit_missing")
    risk = product_audit_risk_order(audit)
    issues = [str(item) for item in audit.get("issues") or []]
    tags = [str(item) for item in audit.get("tags") or []]
    route_id = _stable_id(
        "selector",
        str(artifact_path),
        target_id,
        target_smiles,
        native_rank,
        _route_signature(route),
    )
    feature = _feature_dict(
        native_rank=int(route.get("native_rank", route.get("original_route_rank", route.get("route_rank", native_rank))) or 0),
        native_score=_float(route.get("score", route.get("native_score"))),
        n_steps=int(route.get("n_steps") or len(steps)),
        metrics=metrics,
        terminal_profile=terminal_profile,
        route_class=route_class,
        risk=risk,
        plausibility=plausibility,
        large_atom_gain_count=_large_atom_gain_count(plausibility, issues),
        generic_template_fraction=reaction_profile.get("generic_fraction"),
        condition_stats=_stats(condition_scores),
        enzyme_stats=_stats(enzyme_scores),
        v4_evidence_hits=_evidence_hit_count(route, steps),
        cascade_block_hits=_cascade_block_hit_count(route, steps),
        source_counts=source_counts,
        reaction_type_counts=reaction_type_counts,
    )
    _add_route_level_features(feature, route)
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "artifact_path": str(artifact_path),
        "artifact_type": target.get("artifact_type"),
        "artifact_objective": target.get("artifact_objective"),
        "target_id": target_id,
        "target_smiles": target_smiles,
        "target_index": target.get("target_index"),
        "selector_group_id": _stable_id("target", target_smiles or target_id),
        "route_id": route_id,
        "native_rank": int(route.get("native_rank", route.get("original_route_rank", route.get("route_rank", native_rank))) or 0),
        "artifact_route_index": native_rank,
        "native_score": _float(route.get("score", route.get("native_score"))),
        "n_steps": int(route.get("n_steps") or len(steps)),
        "strict_stock_solve": bool(metrics.get("strict_stock_solve")),
        "route_solved": bool(metrics.get("route_solved")),
        "terminal_reactants": list(metrics.get("terminal_reactants") or []),
        "terminal_stock_status": metrics.get("terminal_stock_status") or {},
        "terminal_max_heavy_atoms": terminal_profile.get("max_terminal_heavy_atoms"),
        "terminal_similarity_to_product": terminal_profile.get("max_terminal_similarity_to_product"),
        "product_audit_class": route_class,
        "product_audit_class_order": ROUTE_CLASS_ORDER.get(route_class, 99),
        "product_audit_risk_order": risk,
        "product_audit_issues": issues,
        "product_audit_tags": tags,
        "route_plausibility": plausibility,
        "route_plausibility_passed": plausibility.get("passed"),
        "large_atom_gain_count": _large_atom_gain_count(plausibility, issues),
        "generic_template_fraction": reaction_profile.get("generic_fraction"),
        "source_model_counts": dict(sorted(source_counts.items())),
        "reaction_type_counts": dict(sorted(reaction_type_counts.items())),
        "condition_score_stats": _stats(condition_scores),
        "enzyme_confidence_stats": _stats(enzyme_scores),
        "v4_evidence_hits": _evidence_hit_count(route, steps),
        "cascade_block_hits": _cascade_block_hit_count(route, steps),
        "route_diversity_signature": _route_signature(route),
        "evidence_provenance": _merged_evidence_provenance(evidence_provenance, target, route),
        "route_label": _route_label(route_class),
        "feature": feature,
        "labels": {
            "is_reject_artifact": route_class == "reject_artifact",
            "is_reviewable": route_class not in {"reject_artifact", "audit_missing"},
            "is_stock_closed_reviewable": bool(metrics.get("strict_stock_solve")) and route_class not in {"reject_artifact", "audit_missing"},
            "is_triage_signal": route_class in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"},
        },
    }


def _merged_evidence_provenance(
    default_provenance: dict[str, Any],
    target: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for source in (default_provenance, target.get("evidence_provenance"), route.get("evidence_provenance")):
        if isinstance(source, dict):
            provenance.update(source)
    for source_name, source in (("target", target), ("route", route)):
        for key, value in source.items():
            if _looks_like_evidence_provenance_key(key):
                provenance[f"{source_name}.{key}"] = value
    return provenance


def _looks_like_evidence_provenance_key(name: str) -> bool:
    lower = str(name).lower()
    return any(
        marker in lower
        for marker in (
            "evidence_source",
            "evidence_corpus",
            "evidence_manifest",
            "retrieval_source",
            "retrieval_corpus",
            "retrieval_manifest",
            "source_split",
            "train_only_retrieval",
        )
    )


def _feature_dict(
    *,
    native_rank: int,
    native_score: float | None,
    n_steps: int,
    metrics: dict[str, Any],
    terminal_profile: dict[str, Any],
    route_class: str,
    risk: int,
    plausibility: dict[str, Any],
    large_atom_gain_count: int,
    generic_template_fraction: Any,
    condition_stats: dict[str, Any],
    enzyme_stats: dict[str, Any],
    v4_evidence_hits: int,
    cascade_block_hits: int,
    source_counts: Counter[str],
    reaction_type_counts: Counter[str],
) -> dict[str, float]:
    rank = max(0, int(native_rank))
    return {
        "native_score": float(native_score or 0.0),
        "native_rank": float(rank),
        "native_inv_rank": 1.0 / float(rank + 1),
        "n_steps": float(n_steps),
        "stock_closed": float(bool(metrics.get("strict_stock_solve"))),
        "route_solved": float(bool(metrics.get("route_solved"))),
        "terminal_max_heavy_atoms": float(terminal_profile.get("max_terminal_heavy_atoms") or 0.0),
        "terminal_similarity_to_product": float(terminal_profile.get("max_terminal_similarity_to_product") or 0.0),
        "audit_class_order": float(ROUTE_CLASS_ORDER.get(route_class, 99)),
        "audit_risk_order": float(risk),
        "audit_is_reject": float(route_class == "reject_artifact"),
        "audit_is_triage": float(route_class in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}),
        "route_plausibility_passed": float(bool(plausibility.get("passed"))),
        "large_atom_gain_count": float(large_atom_gain_count),
        "generic_template_fraction": float(generic_template_fraction or 0.0),
        "condition_score_count": float(condition_stats.get("count") or 0.0),
        "condition_score_mean": float(condition_stats.get("mean") or 0.0),
        "condition_score_max": float(condition_stats.get("max") or 0.0),
        "enzyme_confidence_count": float(enzyme_stats.get("count") or 0.0),
        "enzyme_confidence_mean": float(enzyme_stats.get("mean") or 0.0),
        "enzyme_confidence_max": float(enzyme_stats.get("max") or 0.0),
        "v4_evidence_hits": float(v4_evidence_hits),
        "cascade_block_hits": float(cascade_block_hits),
        "source_model_count": float(len(source_counts)),
        "reaction_type_count": float(len(reaction_type_counts)),
    }


def _add_route_level_features(feature: dict[str, float], route: dict[str, Any]) -> None:
    for key in (
        "ccts_v3_runtime_best_route_evidence",
        "ccts_v3_runtime_model_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_step_any_max",
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_step_pair_mean",
        "n_input_species",
        "n_output_species",
        "n_substrate_scope_entries",
        "overall_ee",
        "overall_yield",
        "search_time_s",
        "total_reaction_time",
        "value_target",
    ):
        value = _float(route.get(key))
        if value is not None:
            feature[key] = float(value)


def _route_label(route_class: str) -> int:
    if route_class in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}:
        return 3
    if route_class == "needs_chemist_review":
        return 2
    if route_class == "weak_hint":
        return 1
    return 0


def _condition_scores(steps: list[dict[str, Any]]) -> list[float]:
    values = []
    for step in steps:
        scores = step.get("scores") if isinstance(step.get("scores"), dict) else {}
        values.append(_float(scores.get("condition")))
        for row in step.get("condition_predictions") or []:
            if isinstance(row, dict):
                values.append(_float(row.get("Score", row.get("score", row.get("confidence")))))
    return [value for value in values if value is not None]


def _enzyme_scores(steps: list[dict[str, Any]]) -> list[float]:
    values = []
    for step in steps:
        scores = step.get("scores") if isinstance(step.get("scores"), dict) else {}
        values.append(_float(scores.get("enzyme")))
        for row in step.get("enzyme_ec_annotations") or []:
            if isinstance(row, dict):
                values.append(_float(row.get("confidence", row.get("Confidence"))))
    return [value for value in values if value is not None]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "min": None}
    return {"count": len(values), "mean": mean(values), "max": max(values), "min": min(values)}


def _large_atom_gain_count(plausibility: dict[str, Any], issues: list[str]) -> int:
    count = sum(1 for issue in issues if "large_unexplained" in issue)
    for step in plausibility.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reasons = " ".join(str(item) for item in step.get("reasons") or [])
        if "large_unexplained" in reasons:
            count += 1
    return count


def _evidence_hit_count(route: dict[str, Any], steps: list[dict[str, Any]]) -> int:
    count = 0
    for key in ("v4_evidence", "cascade_evidence", "route_evidence"):
        value = route.get(key)
        if isinstance(value, list):
            count += len(value)
        elif value:
            count += 1
    route_evidence = route.get("ccts_v3_runtime_route_evidence")
    if isinstance(route_evidence, dict):
        count += len(route_evidence.get("step_scores") or [])
    for step in steps:
        evidence = step.get("evidence") if isinstance(step.get("evidence"), dict) else {}
        count += sum(1 for value in evidence.values() if value not in (None, "", False, [], {}))
        v4_step_evidence = step.get("v4_step_evidence")
        if isinstance(v4_step_evidence, dict) and v4_step_evidence.get("matched"):
            count += 1
    return count


def _cascade_block_hit_count(route: dict[str, Any], steps: list[dict[str, Any]]) -> int:
    count = 0
    for key in ("cascade_blocks", "block_evidence", "v4_block_evidence"):
        value = route.get(key)
        if isinstance(value, list):
            count += len(value)
        elif value:
            count += 1
    count += sum(1 for step in steps if (step.get("evidence") or {}).get("cascade_block"))
    return count


def _route_signature(route: dict[str, Any]) -> str:
    return "|".join(str(step.get("reaction_smiles") or step.get("rxn_smiles") or "") for step in route.get("steps") or [] if isinstance(step, dict))


def _target_split_key(target: dict[str, Any]) -> str:
    return str(target.get("target_smiles") or target.get("target_id") or target.get("target_index") or "")


def _split_for_key(key: str, *, strategy: str, path: Path | None = None) -> str:
    if strategy == "path_name":
        return _split_from_path(path or Path(""))
    if strategy == "train":
        return "train"
    if strategy != "target_hash":
        return "train"
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def _split_from_path(path: Path) -> str:
    text = str(path).lower()
    stem = path.stem.lower()
    tokens = stem.replace("-", "_").split("_")
    if "test" in tokens or "/test" in text or "_test_" in text:
        return "test"
    if "val" in tokens or "valid" in tokens or "validation" in tokens or "/val" in text or "_val_" in text:
        return "val"
    if "train" in tokens or stem.startswith("train") or "/train" in text or "_train_" in text:
        return "train"
    return "train"


def _artifact_type(payload: dict[str, Any], path: Path) -> str:
    objective = str(payload.get("objective") or "")
    name = path.stem.lower()
    if "rejected" in objective or name.endswith("_rejected"):
        return "rejected"
    if name.endswith("_raw"):
        return "raw"
    return "filtered"


def _expand_input_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        text = str(item)
        matches = sorted(Path(path) for path in glob.glob(text))
        if matches:
            paths.extend(matches)
        elif item.exists():
            paths.append(item)
    seen = set()
    out = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _issue_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for issue in row.get("product_audit_issues") or []:
            counter[str(issue)] += 1
    return dict(sorted(counter.items()))


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("selector_group_id") or ""),
            str(row.get("route_diversity_signature") or row.get("route_id") or ""),
        )
        previous = best.get(key)
        if previous is None:
            best[key] = row
            continue
        if _artifact_priority(row) > _artifact_priority(previous):
            row["duplicate_artifact_types"] = _merged_artifact_types(previous, row)
            best[key] = row
        else:
            previous["duplicate_artifact_types"] = _merged_artifact_types(previous, row)
    return sorted(best.values(), key=lambda row: (str(row.get("artifact_path") or ""), int(row.get("artifact_route_index") or 0)))


def _merged_artifact_types(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    values = set(left.get("duplicate_artifact_types") or [left.get("artifact_type")])
    values.update(right.get("duplicate_artifact_types") or [right.get("artifact_type")])
    return sorted(str(value) for value in values if value)


def _artifact_priority(row: dict[str, Any]) -> tuple[int, int]:
    priority = {"filtered": 3, "rejected": 2, "raw": 1}.get(str(row.get("artifact_type") or ""), 0)
    has_audit = int(str(row.get("product_audit_class") or "") not in {"", "audit_missing"})
    return (priority, has_audit)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_id(*parts: Any) -> str:
    text = "||".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a RouteSelector-v0 route-pool pack from ChemEnzy/Web artifacts.")
    ap.add_argument("--input", nargs="+", required=True, help="Input JSON files or glob patterns.")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--dataset", default="route_pool_selector")
    ap.add_argument("--split-strategy", default="target_hash", choices=["target_hash", "train", "path_name"])
    ap.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate route signatures across raw/rejected artifacts.")
    ap.add_argument("--evidence-source-split", help="Annotate retrieval/evidence source split, e.g. train.")
    ap.add_argument("--retrieval-corpus-manifest", help="Annotate the retrieval/evidence corpus manifest path.")
    ap.add_argument("--train-only-retrieval", action="store_true", help="Mark retrieval/evidence features as train-only.")
    args = ap.parse_args()
    evidence_provenance = {}
    if args.evidence_source_split:
        evidence_provenance["evidence_source_split"] = args.evidence_source_split
    if args.retrieval_corpus_manifest:
        evidence_provenance["retrieval_corpus_manifest"] = args.retrieval_corpus_manifest
    if args.train_only_retrieval:
        evidence_provenance["train_only_retrieval"] = True
    report = build_route_pool_selector_pack(
        inputs=[Path(item) for item in args.input],
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report),
        split_manifest=Path(args.split_manifest),
        dataset=args.dataset,
        split_strategy=args.split_strategy,
        deduplicate_routes=not args.no_deduplicate,
        evidence_provenance=evidence_provenance,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
