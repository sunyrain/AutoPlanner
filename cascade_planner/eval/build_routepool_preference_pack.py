"""Build route-pool preference packs from native ChemEnzy routes and rule audit.

This is the v2 supervision bridge: v4 records teach cascade chemistry priors,
while native route pools teach how ChemEnzy candidates should be ordered within
the same target.  Preferences are pairwise; no hand-weighted total score is used
as the final label.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import route_record_from_native_route
from cascade_planner.eval.product_route_feasibility_audit import ROUTE_CLASS_ORDER, build_product_route_feasibility_audit


PACK_SCHEMA_VERSION = "routepool_preference_pack.v1"


def build_routepool_preference_pack(
    *,
    native_pool: Path,
    output_dir: Path,
    benchmark: Path | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    top_k: int | None = 50,
    max_pairs_per_target: int = 80,
    include_native_rank_tiebreak: bool = True,
) -> dict[str, Any]:
    run = json.loads(Path(native_pool).read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    capped_run = _cap_run(run, top_k=top_k)
    audit = build_product_route_feasibility_audit(capped_run, benchmark_rows=benchmark_rows)
    route_rows = []
    preferences = []
    route_audits_by_id: dict[str, dict[str, Any]] = {}
    dataset = dataset_name or str((run.get("metadata") or {}).get("dataset") or native_pool.stem)

    for target_index, (target, audit_target) in enumerate(zip(capped_run.get("targets") or [], audit.get("targets") or [])):
        target_smiles = str(target.get("target_smiles") or "")
        target_id = str(target.get("cascade_id") or target.get("target_id") or target.get("index") or target_index)
        target_route_rows = []
        route_audits = sorted(audit_target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))
        routes = target.get("routes") or []
        for native_rank, route in enumerate(routes):
            feature_row = route_record_from_native_route(
                route,
                target_smiles=target_smiles,
                target_id=target_id,
                native_rank=native_rank,
                dataset=dataset,
            )
            feature_row["split"] = split
            feature_row["dataset"] = dataset
            feature_row["route_source"] = "ChemEnzyRetroPlanner"
            feature_row["route_pool_target_index"] = target_index
            feature_row["split_group_id"] = f"{dataset}:{target_index}:{target_id}"
            audit_row = route_audits[native_rank] if native_rank < len(route_audits) else {}
            feature_row["product_audit"] = {
                "route_class": audit_row.get("route_class"),
                "issues": audit_row.get("issues") or [],
                "tags": audit_row.get("tags") or [],
                "route_plausibility": audit_row.get("route_plausibility") or {},
            }
            feature_row["value_target"] = _audit_binary_target(audit_row)
            route_rows.append(feature_row)
            target_route_rows.append(feature_row)
            route_audits_by_id[str(feature_row["route_id"])] = audit_row
        preferences.extend(
            _target_preferences(
                target_route_rows,
                route_audits_by_id,
                max_pairs=max_pairs_per_target,
                include_native_rank_tiebreak=include_native_rank_tiebreak,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{dataset}_routepool_features.jsonl"
    pref_path = output_dir / f"{dataset}_routepool_preferences.jsonl"
    _write_jsonl(feature_path, route_rows)
    _write_jsonl(pref_path, preferences)
    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_pool": str(native_pool),
            "benchmark": str(benchmark) if benchmark else None,
            "dataset": dataset,
            "split": split,
            "top_k": top_k,
            "max_pairs_per_target": max_pairs_per_target,
            "include_native_rank_tiebreak": include_native_rank_tiebreak,
            "training_contract": "native_route_pool_rule_audit_pairwise_preferences.v1",
        },
        "counts": {
            "route_rows": len(route_rows),
            "preferences": len(preferences),
            "targets": len(capped_run.get("targets") or []),
        },
        "route_class_counts": dict(Counter((row.get("product_audit") or {}).get("route_class") for row in route_rows)),
        "preference_source_counts": dict(Counter(row.get("preference_source") for row in preferences)),
        "outputs": {
            "features": str(feature_path),
            "preferences": str(pref_path),
        },
    }
    report_path = output_dir / f"{dataset}_routepool_preference_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_combined_routepool_pack(
    *,
    feature_paths: list[Path],
    preference_paths: list[Path],
    output_features: Path,
    output_preferences: Path,
) -> dict[str, Any]:
    features = [row for path in feature_paths for row in _read_jsonl(path)]
    preferences = [row for path in preference_paths for row in _read_jsonl(path)]
    output_features.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_features, features)
    _write_jsonl(output_preferences, preferences)
    report = {
        "schema_version": "routepool_combined_pack.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "features": len(features),
            "preferences": len(preferences),
        },
        "inputs": {
            "features": [str(path) for path in feature_paths],
            "preferences": [str(path) for path in preference_paths],
        },
        "outputs": {
            "features": str(output_features),
            "preferences": str(output_preferences),
        },
    }
    output_preferences.with_suffix(".summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _target_preferences(
    route_rows: list[dict[str, Any]],
    audits_by_id: dict[str, dict[str, Any]],
    *,
    max_pairs: int,
    include_native_rank_tiebreak: bool,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for better in route_rows:
        for worse in route_rows:
            if better.get("route_id") == worse.get("route_id"):
                continue
            decision = _preference_decision(
                audits_by_id[str(better["route_id"])],
                audits_by_id[str(worse["route_id"])],
            )
            if decision is None and include_native_rank_tiebreak:
                decision = _native_rank_tiebreak_decision(
                    better,
                    worse,
                    audits_by_id[str(better["route_id"])],
                    audits_by_id[str(worse["route_id"])],
                )
            if decision is None:
                continue
            candidates.append((_preference_priority(decision), better, worse, decision))
    candidates.sort(
        key=lambda item: (
            item[0],
            int(item[1].get("native_rank") or 0),
            int(item[2].get("native_rank") or 0),
            str(item[1].get("route_id") or ""),
            str(item[2].get("route_id") or ""),
        )
    )
    out = []
    seen = set()
    for _, better, worse, decision in candidates:
        key = (better.get("route_id"), worse.get("route_id"), decision["reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append(_pref_row(better, worse, decision))
        if len(out) >= max_pairs:
            break
    return out


def _preference_decision(better_audit: dict[str, Any], worse_audit: dict[str, Any]) -> dict[str, Any] | None:
    better_class = str(better_audit.get("route_class") or "")
    worse_class = str(worse_audit.get("route_class") or "")
    better_issues = set(better_audit.get("issues") or [])
    worse_issues = set(worse_audit.get("issues") or [])
    better_tags = set(better_audit.get("tags") or [])
    worse_tags = set(worse_audit.get("tags") or [])
    better_order = ROUTE_CLASS_ORDER.get(better_class, 99)
    worse_order = ROUTE_CLASS_ORDER.get(worse_class, 99)
    if better_order + 1 <= worse_order:
        return {
            "reason": "rule_audit_class_dominance",
            "confidence_tier": "T2_rule_audit_preference",
            "evidence_summary": {"better_class": better_class, "worse_class": worse_class},
        }
    if better_class == "triage_late_stage" and worse_class == "triage_fragment":
        return {
            "reason": "late_stage_over_fragment",
            "confidence_tier": "T2_rule_audit_class_preference",
            "evidence_summary": {"better_class": better_class, "worse_class": worse_class},
        }
    bad_issues = {
        "trivial_stock_closure",
        "racemization_artifact",
        "large_unexplained_atom_gain",
        "large_unexplained_heavy_atom_gain",
        "large_unexplained_carbon_gain",
        "large_unexplained_hetero_atom_gain",
        "invalid_product_smiles",
        "invalid_or_missing_reactants",
    }
    if not (better_issues & bad_issues) and (worse_issues & bad_issues):
        return {
            "reason": "artifact_issue_dominance",
            "confidence_tier": "T2_rule_audit_issue_preference",
            "evidence_summary": {"better_issues": sorted(better_issues), "worse_issues": sorted(worse_issues)},
        }
    decision = _same_class_quality_decision(better_audit, worse_audit)
    if decision is not None:
        return decision
    return None


def _native_rank_tiebreak_decision(
    better: dict[str, Any],
    worse: dict[str, Any],
    better_audit: dict[str, Any],
    worse_audit: dict[str, Any],
) -> dict[str, Any] | None:
    if str(better_audit.get("route_class") or "") != str(worse_audit.get("route_class") or ""):
        return None
    if set(better_audit.get("issues") or []) != set(worse_audit.get("issues") or []):
        return None
    if set(better_audit.get("tags") or []) != set(worse_audit.get("tags") or []):
        return None
    try:
        better_rank = int(better.get("native_rank") or 0)
        worse_rank = int(worse.get("native_rank") or 0)
    except (TypeError, ValueError):
        return None
    if better_rank + 5 <= worse_rank:
        return {
            "reason": "same_evidence_native_rank_tiebreak",
            "confidence_tier": "T3_native_rank_tiebreak",
            "evidence_summary": {
                "route_class": str(better_audit.get("route_class") or ""),
                "better_native_rank": better_rank,
                "worse_native_rank": worse_rank,
            },
        }
    return None


def _same_class_quality_decision(better_audit: dict[str, Any], worse_audit: dict[str, Any]) -> dict[str, Any] | None:
    better_class = str(better_audit.get("route_class") or "")
    worse_class = str(worse_audit.get("route_class") or "")
    if better_class != worse_class:
        return None
    better_issues = set(better_audit.get("issues") or [])
    worse_issues = set(worse_audit.get("issues") or [])
    better_tags = set(better_audit.get("tags") or [])
    worse_tags = set(worse_audit.get("tags") or [])
    better_plausible = _plausibility_passed(better_audit)
    worse_plausible = _plausibility_passed(worse_audit)
    if better_plausible is True and worse_plausible is False:
        return {
            "reason": "same_class_material_plausibility_preference",
            "confidence_tier": "T2_rule_audit_plausibility_preference",
            "evidence_summary": {
                "route_class": better_class,
                "better_plausibility": (better_audit.get("route_plausibility") or {}),
                "worse_plausibility": (worse_audit.get("route_plausibility") or {}),
            },
        }
    if "generic_reaction_sequence" not in better_issues and "generic_reaction_sequence" in worse_issues:
        return {
            "reason": "same_class_non_generic_over_generic",
            "confidence_tier": "T2_rule_audit_issue_preference",
            "evidence_summary": {"route_class": better_class, "better_issues": sorted(better_issues), "worse_issues": sorted(worse_issues)},
        }
    if "late_stage_derivatization" in better_tags and "late_stage_derivatization" not in worse_tags:
        return {
            "reason": "same_class_late_stage_tag_preference",
            "confidence_tier": "T2_rule_audit_tag_preference",
            "evidence_summary": {"route_class": better_class, "better_tags": sorted(better_tags), "worse_tags": sorted(worse_tags)},
        }
    return None


def _pref_row(better: dict[str, Any], worse: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "preference_id": _stable_id(better.get("route_id"), worse.get("route_id"), decision["reason"]),
        "route_family": better.get("split_group_id"),
        "better_route_id": better.get("route_id"),
        "worse_route_id": worse.get("route_id"),
        "split": better.get("split") if better.get("split") == worse.get("split") else "mixed",
        "preference_source": decision["reason"],
        "confidence_tier": decision["confidence_tier"],
        "evidence_summary": decision["evidence_summary"],
    }


def _preference_priority(decision: dict[str, Any]) -> int:
    reason = str(decision.get("reason") or "")
    order = {
        "same_class_non_generic_over_generic": 0,
        "same_class_late_stage_tag_preference": 1,
        "same_class_material_plausibility_preference": 2,
        "late_stage_over_fragment": 3,
        "same_evidence_native_rank_tiebreak": 4,
        "rule_audit_class_dominance": 5,
        "artifact_issue_dominance": 6,
    }
    return order.get(reason, 99)


def _audit_binary_target(audit: dict[str, Any]) -> float:
    return 1.0 if audit.get("route_class") in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"} else 0.0


def _plausibility_passed(audit: dict[str, Any]) -> bool | None:
    plausibility = audit.get("route_plausibility")
    if not isinstance(plausibility, dict) or "passed" not in plausibility:
        return None
    return bool(plausibility.get("passed"))


def _cap_run(run: dict[str, Any], *, top_k: int | None) -> dict[str, Any]:
    if top_k is None or top_k <= 0:
        return run
    capped = dict(run)
    targets = []
    for target in run.get("targets") or []:
        payload = dict(target)
        payload["routes"] = list(target.get("routes") or [])[: int(top_k)]
        payload["route_count"] = len(payload["routes"])
        targets.append(payload)
    capped["targets"] = targets
    return capped


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_id(*parts: Any) -> str:
    import hashlib

    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ChemEnzy route-pool preferences from product-audit evidence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("single")
    one.add_argument("--native-pool", required=True)
    one.add_argument("--output-dir", required=True)
    one.add_argument("--benchmark")
    one.add_argument("--dataset-name")
    one.add_argument("--split", default="train")
    one.add_argument("--top-k", type=int, default=50)
    one.add_argument("--max-pairs-per-target", type=int, default=80)
    one.add_argument("--disable-native-rank-tiebreak", action="store_true")
    combo = sub.add_parser("combine")
    combo.add_argument("--feature", action="append", required=True)
    combo.add_argument("--preference", action="append", required=True)
    combo.add_argument("--output-features", required=True)
    combo.add_argument("--output-preferences", required=True)
    args = ap.parse_args()
    if args.cmd == "single":
        report = build_routepool_preference_pack(
            native_pool=Path(args.native_pool),
            output_dir=Path(args.output_dir),
            benchmark=Path(args.benchmark) if args.benchmark else None,
            dataset_name=args.dataset_name,
            split=args.split,
            top_k=args.top_k,
            max_pairs_per_target=args.max_pairs_per_target,
            include_native_rank_tiebreak=not args.disable_native_rank_tiebreak,
        )
    else:
        report = build_combined_routepool_pack(
            feature_paths=[Path(path) for path in args.feature],
            preference_paths=[Path(path) for path in args.preference],
            output_features=Path(args.output_features),
            output_preferences=Path(args.output_preferences),
        )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
