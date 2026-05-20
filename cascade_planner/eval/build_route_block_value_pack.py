"""Build a route/block value pack from RouteSelector rows.

The pack intentionally keeps feature groups and weak label tasks separate.  It
does not collapse product-audit, retrieval evidence, and learned CCTS signals
into a single hand-weighted score.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


PACK_SCHEMA_VERSION = "route_block_value_pack.v1"


def build_route_block_value_pack(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    dataset: str = "route_block_value",
    evidence_contract: str = "train_only_retrieval_expected",
    generic_template_threshold: float = 0.75,
    strong_route_evidence_threshold: float = 1.0,
    require_evidence_provenance: bool = False,
    runtime_retrieval_only: bool = False,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    if not rows:
        raise ValueError("empty input_jsonl")
    out_rows = [
        _convert_row(
            row,
            dataset=dataset,
            evidence_contract=evidence_contract,
            generic_template_threshold=generic_template_threshold,
            strong_route_evidence_threshold=strong_route_evidence_threshold,
            runtime_retrieval_only=runtime_retrieval_only,
        )
        for row in rows
    ]
    evidence_audit = _evidence_provenance_audit(out_rows, evidence_contract=evidence_contract)
    if require_evidence_provenance and (
        evidence_audit["missing_retrieval_provenance_rows"] > 0
        or evidence_audit["missing_train_only_marker_rows"] > 0
    ):
        raise ValueError(
            "retrieval/evidence features are present but train-only source provenance is incomplete: "
            f"missing_source={evidence_audit['missing_retrieval_provenance_rows']} "
            f"missing_train_only={evidence_audit['missing_train_only_marker_rows']}"
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, out_rows)
    report = _report(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        report_json=report_json,
        rows=out_rows,
        dataset=dataset,
        evidence_contract=evidence_contract,
        generic_template_threshold=generic_template_threshold,
        strong_route_evidence_threshold=strong_route_evidence_threshold,
        evidence_audit=evidence_audit,
        runtime_retrieval_only=runtime_retrieval_only,
    )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _convert_row(
    row: dict[str, Any],
    *,
    dataset: str,
    evidence_contract: str,
    generic_template_threshold: float,
    strong_route_evidence_threshold: float,
    runtime_retrieval_only: bool,
) -> dict[str, Any]:
    feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    audit_class = str(row.get("product_audit_class") or "audit_missing")
    route_plausibility_passed = bool(row.get("route_plausibility_passed"))
    large_atom_gain_count = int(row.get("large_atom_gain_count") or 0)
    generic_fraction = _float(row.get("generic_template_fraction"))
    retrieval_evidence_any = _feature_any(
        feature,
        _retrieval_feature_names(runtime_retrieval_only=runtime_retrieval_only),
    )
    pair_context_evidence_any = _float(feature.get("ccts_v3_runtime_step_pair_max")) > 0.0
    strong_route_evidence = _float(feature.get("ccts_v3_runtime_best_route_evidence")) >= float(strong_route_evidence_threshold)
    cascade_block_any = _float(row.get("cascade_block_hits")) > 0 or _float(feature.get("cascade_block_hits")) > 0
    stock_closed = bool(row.get("strict_stock_solve") or feature.get("stock_closed"))
    reject_artifact = bool(labels.get("is_reject_artifact") or audit_class == "reject_artifact")
    reviewable_by_audit = bool(labels.get("is_reviewable") or audit_class not in {"reject_artifact", "audit_missing"})
    large_atom_gain = large_atom_gain_count > 0
    generic_template_heavy = generic_fraction >= float(generic_template_threshold)
    route_evidence_any = bool(strong_route_evidence or pair_context_evidence_any or cascade_block_any)
    no_human_route_positive = bool(reviewable_by_audit and stock_closed and not large_atom_gain)
    no_human_route_negative = bool(reject_artifact or large_atom_gain or not stock_closed)
    no_human_consensus_positive = bool(
        no_human_route_positive and (route_evidence_any or not generic_template_heavy)
    )
    no_human_consensus_negative = no_human_route_negative
    weak_label_tasks = {
        "reject_artifact": reject_artifact,
        "reviewable_by_audit": reviewable_by_audit,
        "stock_closed": stock_closed,
        "stock_closed_reviewable": bool(labels.get("is_stock_closed_reviewable")),
        "material_sane_proxy": route_plausibility_passed and audit_class != "reject_artifact",
        "large_atom_gain": large_atom_gain,
        "generic_template_heavy": generic_template_heavy,
        "retrieval_evidence_any": retrieval_evidence_any,
        "pair_context_evidence_any": pair_context_evidence_any,
        "strong_route_evidence": strong_route_evidence,
        "block_evidence_any": cascade_block_any,
        "no_human_route_positive": no_human_route_positive,
        "no_human_route_negative": no_human_route_negative,
        "no_human_consensus_positive": no_human_consensus_positive,
        "no_human_consensus_negative": no_human_consensus_negative,
    }
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "dataset": dataset,
        "source_schema_version": row.get("schema_version"),
        "split": row.get("split") or "train",
        "target_id": row.get("target_id"),
        "target_smiles": row.get("target_smiles"),
        "selector_group_id": row.get("selector_group_id") or row.get("target_id"),
        "route_id": row.get("route_id"),
        "artifact_path": row.get("artifact_path"),
        "artifact_type": row.get("artifact_type"),
        "native_rank": int(row.get("native_rank") or 0),
        "native_score": _float(row.get("native_score")),
        "n_steps": int(row.get("n_steps") or 0),
        "route_diversity_signature": row.get("route_diversity_signature"),
        "terminal_reactants": row.get("terminal_reactants") or [],
        "terminal_stock_status": row.get("terminal_stock_status") or {},
        "product_audit": {
            "route_class": audit_class,
            "class_order": row.get("product_audit_class_order"),
            "risk_order": row.get("product_audit_risk_order"),
            "issues": row.get("product_audit_issues") or [],
            "tags": row.get("product_audit_tags") or [],
            "route_plausibility_passed": route_plausibility_passed,
            "large_atom_gain_count": large_atom_gain_count,
        },
        "weak_label_tasks": weak_label_tasks,
        "feature_groups": _feature_groups(feature, row, runtime_retrieval_only=runtime_retrieval_only),
        "evidence_provenance": _evidence_provenance(row, feature),
        "training_contract": {
            "objective": (
                "Train route/block outcome scorers with explicit ablations; do not use this pack as a "
                "single hand-weighted route score."
            ),
            "gold_silver_policy": "gold/silver may be evidence confidence metadata only, not route preference",
            "audit_policy": "audit labels are safety/control labels and must be ablated from model claims",
            "retrieval_policy": "retrieval-only evidence rank is a required baseline",
            "no_human_policy": (
                "no_human_* tasks are automatic weak-supervision targets derived from stock closure, "
                "material-sanity screens, and route/block evidence; they require no expert CSV but "
                "must be evaluated against audit and retrieval-only controls"
            ),
            "evidence_contract": evidence_contract,
        },
    }


def _feature_groups(feature: dict[str, Any], row: dict[str, Any], *, runtime_retrieval_only: bool) -> dict[str, dict[str, float]]:
    return {
        "native": _pick(
            feature,
            [
                "native_score",
                "native_rank",
                "native_inv_rank",
                "native_rank_fraction",
                "n_steps",
            ],
        ),
        "stock_route": {
            **_pick(
                feature,
                [
                    "stock_closed",
                    "route_solved",
                    "terminal_max_heavy_atoms",
                    "terminal_similarity_to_product",
                ],
            ),
            "strict_stock_solve": float(bool(row.get("strict_stock_solve"))),
        },
        "product_audit": _pick(
            feature,
            [
                "audit_class_order",
                "audit_risk_order",
                "audit_is_reject",
                "audit_is_triage",
                "route_plausibility_passed",
                "large_atom_gain_count",
                "generic_template_fraction",
            ],
        ),
        "cascade_retrieval": _pick(feature, _retrieval_feature_names(runtime_retrieval_only=runtime_retrieval_only)),
        "route_step_v4_evidence": _pick(feature, ["v4_evidence_hits", "cascade_block_hits"]),
        "learned_ccts": _pick(
            feature,
            [
                "ccts_v3_runtime_model_max",
                "ccts_v3_runtime_model_mean",
            ],
        ),
        "condition_enzyme": _pick(
            feature,
            [
                "condition_score_count",
                "condition_score_mean",
                "condition_score_max",
                "enzyme_confidence_count",
                "enzyme_confidence_mean",
                "enzyme_confidence_max",
            ],
        ),
        "route_context": _pick(
            feature,
            [
                "source_model_count",
                "reaction_type_count",
                "n_input_species",
                "n_output_species",
                "n_substrate_scope_entries",
                "overall_ee",
                "overall_yield",
                "search_time_s",
                "total_reaction_time",
            ],
        ),
    }


def _report(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    rows: list[dict[str, Any]],
    dataset: str,
    evidence_contract: str,
    generic_template_threshold: float,
    strong_route_evidence_threshold: float,
    evidence_audit: dict[str, Any],
    runtime_retrieval_only: bool,
) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split") or "train") for row in rows)
    task_counts: dict[str, int] = Counter()
    for row in rows:
        for key, value in (row.get("weak_label_tasks") or {}).items():
            if value:
                task_counts[str(key)] += 1
    feature_group_counts = {
        group: sum(1 for row in rows if (row.get("feature_groups") or {}).get(group))
        for group in _feature_group_names(rows)
    }
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "inputs": {"input_jsonl": str(input_jsonl)},
        "outputs": {"pack": str(output_jsonl), "report": str(report_json)},
        "counts": {
            "rows": len(rows),
            "targets": len({row.get("selector_group_id") for row in rows}),
            "routes": len({row.get("route_id") for row in rows}),
        },
        "split_counts": dict(sorted(split_counts.items())),
        "weak_label_positive_counts": dict(sorted(task_counts.items())),
        "feature_group_row_counts": dict(sorted(feature_group_counts.items())),
        "evidence_provenance_audit": evidence_audit,
        "training_contract": {
            "evidence_contract": evidence_contract,
            "generic_template_threshold": float(generic_template_threshold),
            "strong_route_evidence_threshold": float(strong_route_evidence_threshold),
            "runtime_retrieval_only": bool(runtime_retrieval_only),
            "required_controls": [
                "ChemEnzy native rank",
                "product-audit guard",
                "retrieval-only evidence rank",
                "learned model without audit features",
                "learned model without retrieval features",
            ],
            "no_human_tasks": [
                "no_human_route_positive",
                "no_human_route_negative",
                "no_human_consensus_positive",
                "no_human_consensus_negative",
            ],
            "promotion_gate": (
                "learned route/block scorer must beat retrieval-only on held-out split and "
                "show guarded live-search quality lift before search-time promotion"
            ),
        },
    }


def _retrieval_feature_names(*, runtime_retrieval_only: bool) -> list[str]:
    names = [
        "ccts_v3_runtime_best_route_evidence",
        "ccts_v3_runtime_step_any_max",
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_step_pair_mean",
    ]
    if not runtime_retrieval_only:
        return ["v4_evidence_hits", "cascade_block_hits", *names]
    return names


def _evidence_provenance(row: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    found = {}
    explicit = row.get("evidence_provenance")
    if isinstance(explicit, dict):
        found.update({f"evidence_provenance.{key}": value for key, value in explicit.items()})
    for name, value in row.items():
        if _looks_like_evidence_provenance_key(name):
            found[name] = value
    for name, value in feature.items():
        if _looks_like_evidence_provenance_key(name):
            found[f"feature.{name}"] = value
    keys = sorted(found)
    return {
        "status": "present" if keys else "missing",
        "keys": keys,
        "has_train_only_marker": _has_train_only_marker(found),
        "has_source_split_marker": any("split" in key.lower() for key in keys),
    }


def _has_train_only_marker(values: dict[str, Any]) -> bool:
    for key, value in values.items():
        text = f"{key} {value}".lower()
        if "train_only" in text or "train-only" in text:
            return True
    return False


def _looks_like_evidence_provenance_key(name: str) -> bool:
    lower = str(name).lower()
    markers = [
        "evidence_source",
        "evidence_corpus",
        "evidence_manifest",
        "retrieval_source",
        "retrieval_corpus",
        "retrieval_manifest",
        "source_split",
        "train_only_retrieval",
    ]
    return any(marker in lower for marker in markers)


def _evidence_provenance_audit(rows: list[dict[str, Any]], *, evidence_contract: str) -> dict[str, Any]:
    retrieval_rows = []
    missing_rows = []
    missing_train_only_rows = []
    requires_train_only = "train_only" in str(evidence_contract).lower() or "train-only" in str(evidence_contract).lower()
    for row in rows:
        if _row_has_retrieval_features(row):
            retrieval_rows.append(row)
            provenance = row.get("evidence_provenance") if isinstance(row.get("evidence_provenance"), dict) else {}
            if provenance.get("status") != "present":
                missing_rows.append(row)
            elif requires_train_only and not provenance.get("has_train_only_marker"):
                missing_train_only_rows.append(row)
    status = "verified_or_no_retrieval_features"
    warnings = []
    if missing_rows:
        status = "unverifiable_without_source_provenance"
        warnings.append(
            "retrieval/evidence features exist, but the pack does not contain source-corpus provenance; "
            "treat train-only retrieval as an expectation, not a verified property"
        )
    elif missing_train_only_rows:
        status = "unverifiable_without_train_only_marker"
        warnings.append(
            "retrieval/evidence source provenance is present, but it does not explicitly mark a train-only corpus"
        )
    return {
        "status": status,
        "retrieval_feature_rows": len(retrieval_rows),
        "missing_retrieval_provenance_rows": len(missing_rows),
        "missing_train_only_marker_rows": len(missing_train_only_rows),
        "provenance_present_rows": len(retrieval_rows) - len(missing_rows),
        "warnings": warnings,
    }


def _row_has_retrieval_features(row: dict[str, Any]) -> bool:
    groups = row.get("feature_groups") if isinstance(row.get("feature_groups"), dict) else {}
    retrieval = groups.get("cascade_retrieval") if isinstance(groups.get("cascade_retrieval"), dict) else {}
    return any(_float(value) > 0.0 for value in retrieval.values())


def _feature_group_names(rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        names.update((row.get("feature_groups") or {}).keys())
    return sorted(names)


def _pick(feature: dict[str, Any], names: list[str]) -> dict[str, float]:
    out = {}
    for name in names:
        if name in feature:
            out[name] = _float(feature.get(name))
    return out


def _feature_any(feature: dict[str, Any], names: list[str]) -> bool:
    return any(_float(feature.get(name)) > 0.0 for name in names)


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object row in {path}")
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build route_block_value_pack_v1 from a RouteSelector JSONL pack")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--dataset", default="route_block_value")
    ap.add_argument("--evidence-contract", default="train_only_retrieval_expected")
    ap.add_argument("--generic-template-threshold", type=float, default=0.75)
    ap.add_argument("--strong-route-evidence-threshold", type=float, default=1.0)
    ap.add_argument("--require-evidence-provenance", action="store_true")
    ap.add_argument("--runtime-retrieval-only", action="store_true")
    args = ap.parse_args()
    report = build_route_block_value_pack(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report),
        dataset=args.dataset,
        evidence_contract=args.evidence_contract,
        generic_template_threshold=args.generic_template_threshold,
        strong_route_evidence_threshold=args.strong_route_evidence_threshold,
        require_evidence_provenance=args.require_evidence_provenance,
        runtime_retrieval_only=args.runtime_retrieval_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
