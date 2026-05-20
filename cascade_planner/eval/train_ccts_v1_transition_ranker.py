"""Train CCTS-v1 on strict-blind CascadeBench transition splits.

CCTS-v1 keeps ChemEnzy fixed as the generator and tests whether cascade context
and train-only v4 evidence improve transition ranking beyond ChemEnzy's native
candidate order.  The script is intentionally ranker-only; search integration is
left as a downstream experiment after the strict transition/block benchmarks.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker, early_stopping
from rdkit import RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    CandidateDataset,
    _baseline_scores,
    _build_candidate_dataset,
    _build_evidence_bank,
    _build_feature_schema,
    _compact_candidate_rows,
    _evaluate_dataset,
    _feature_groups,
    _metric_for_selection,
    _read_json,
    _standardize,
    _write_jsonl,
)


CCTS_V1_SCHEMA_VERSION = "ccts_v1_transition_ranker.strict_split.v1"
NO_CONTEXT_CHEM_FEATURES = {
    "chem__target_heavy_atoms",
    "chem__product_target_similarity",
    "chem__step_pos",
    "chem__remaining_steps",
    "chem__has_previous_transform",
}


def train_ccts_v1_transition_ranker(
    *,
    train_coverage: Path,
    train_cache: Path,
    val_coverage: Path,
    val_cache: Path,
    test_coverage: Path,
    test_cache: Path,
    output_dir: Path,
    split_manifest: Path | None = None,
    similarity_threshold: float = 0.7,
    max_candidates_per_transition: int = 100,
    evidence_pool_size: int = 80,
    n_estimators: int = 320,
    learning_rate: float = 0.04,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_payload = _read_json(train_coverage)
    val_payload = _read_json(val_coverage)
    test_payload = _read_json(test_coverage)
    train_cache_rows = _read_json(train_cache)
    val_cache_rows = _read_json(val_cache)
    test_cache_rows = _read_json(test_cache)

    train_transitions = [row for row in train_payload.get("transitions") or [] if isinstance(row, dict)]
    val_transitions = [row for row in val_payload.get("transitions") or [] if isinstance(row, dict)]
    test_transitions = [row for row in test_payload.get("transitions") or [] if isinstance(row, dict)]
    evidence_bank = _build_evidence_bank(train_transitions)
    schema = _build_feature_schema(train_transitions, train_cache_rows)

    train_data = _build_candidate_dataset(
        train_transitions,
        train_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=True,
    )
    val_data = _build_candidate_dataset(
        val_transitions,
        val_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=False,
    )
    test_data = _build_candidate_dataset(
        test_transitions,
        test_cache_rows,
        evidence_bank=evidence_bank,
        schema=schema,
        similarity_threshold=similarity_threshold,
        max_candidates_per_transition=max_candidates_per_transition,
        evidence_pool_size=evidence_pool_size,
        require_trainable_group=False,
    )
    if not train_data.rows:
        raise ValueError("no trainable CCTS-v1 candidate rows")
    if not val_data.rows:
        raise ValueError("no validation candidate rows")
    if not test_data.rows:
        raise ValueError("no test candidate rows")

    feature_groups = _feature_groups(train_data.feature_names)
    model_specs = _model_specs(train_data, feature_groups)
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for model_name, feature_indices in model_specs.items():
        if not feature_indices:
            continue
        model = _fit_ranker(
            train_data,
            val_data,
            feature_indices=feature_indices,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            seed=seed,
        )
        models[model_name] = model
        reports[model_name] = _model_report(
            model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_indices=feature_indices,
        )

    baseline_reports = {
        "train": _evaluate_dataset(train_data, _baseline_scores(train_data.rows)),
        "val": _evaluate_dataset(val_data, _baseline_scores(val_data.rows)),
        "test": _evaluate_dataset(test_data, _baseline_scores(test_data.rows)),
    }
    blends = _blend_reports(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        models=models,
        model_specs=model_specs,
        aux_names=[
            "context_evidence_only",
            "chem_no_context_plus_context_evidence",
            "ccts_v1_full",
        ],
    )
    leakage = _coverage_leakage_report(
        {
            "train": train_transitions,
            "val": val_transitions,
            "test": test_transitions,
        }
    )
    split_manifest_payload = _read_json(split_manifest) if split_manifest and split_manifest.exists() else None

    result = {
        "schema_version": CCTS_V1_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_coverage": str(train_coverage),
            "val_coverage": str(val_coverage),
            "test_coverage": str(test_coverage),
            "train_cache": str(train_cache),
            "val_cache": str(val_cache),
            "test_cache": str(test_cache),
            "split_manifest": str(split_manifest) if split_manifest else None,
            "output_dir": str(output_dir),
            "similarity_threshold": similarity_threshold,
            "max_candidates_per_transition": max_candidates_per_transition,
            "evidence_pool_size": evidence_pool_size,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "train_transitions": len(train_transitions),
            "val_transitions": len(val_transitions),
            "test_transitions": len(test_transitions),
            "evidence_bank": len(evidence_bank),
            "train_candidate_rows": len(train_data.rows),
            "val_candidate_rows": len(val_data.rows),
            "test_candidate_rows": len(test_data.rows),
            "train_groups": len(train_data.group_sizes),
            "val_groups": len(val_data.group_sizes),
            "test_groups": len(test_data.group_sizes),
            "train_positive_rows": int(train_data.y.sum()),
            "val_positive_rows": int(val_data.y.sum()),
            "test_positive_rows": int(test_data.y.sum()),
        },
        "leakage_checks": leakage,
        "strict_split_manifest_summary": _strict_manifest_summary(split_manifest_payload),
        "baseline_chem_rank": baseline_reports,
        "models": reports,
        "blends": blends,
        "feature_schema": {
            "feature_names": train_data.feature_names,
            "chem_feature_names": [train_data.feature_names[idx] for idx in train_data.chem_feature_indices],
            "model_specs": {
                name: [train_data.feature_names[idx] for idx in indices]
                for name, indices in model_specs.items()
            },
            **schema,
        },
    }

    with (output_dir / "ccts_v1_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": CCTS_V1_SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "ccts_v1_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ccts_v1_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v1_test_candidates.jsonl", _compact_candidate_rows(test_data.rows))
    return result


def _model_specs(data: CandidateDataset, feature_groups: dict[str, list[int]]) -> dict[str, list[int]]:
    feature_names = data.feature_names
    chem_no_context = [
        idx
        for idx in data.chem_feature_indices
        if feature_names[idx] not in NO_CONTEXT_CHEM_FEATURES
    ]
    return {
        "chem_no_context": chem_no_context,
        "chem_only": list(data.chem_feature_indices),
        "context_evidence_only": list(feature_groups["ccts_scalar"]),
        "chem_no_context_plus_context_evidence": chem_no_context + list(feature_groups["ccts_scalar"]),
        "ccts_v1_full": list(range(data.x.shape[1])),
    }


def _fit_ranker(
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    *,
    feature_indices: list[int],
    n_estimators: int,
    learning_rate: float,
    seed: int,
) -> LGBMRanker:
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=15,
        min_child_samples=24,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=3.0,
        reg_alpha=0.1,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        train_data.x[:, feature_indices],
        train_data.y,
        group=train_data.group_sizes,
        eval_set=[(val_data.x[:, feature_indices], val_data.y)],
        eval_group=[val_data.group_sizes],
        eval_at=[1, 3, 5, 10],
        callbacks=[early_stopping(30, verbose=False)],
    )
    return model


def _model_report(
    model: Any,
    *,
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    test_data: CandidateDataset,
    feature_indices: list[int],
) -> dict[str, Any]:
    return {
        "train": _evaluate_dataset(train_data, model.predict(train_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "val": _evaluate_dataset(val_data, model.predict(val_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "test": _evaluate_dataset(test_data, model.predict(test_data.x[:, feature_indices], num_iteration=model.best_iteration_)),
        "feature_count": len(feature_indices),
        "feature_names": [train_data.feature_names[idx] for idx in feature_indices],
        "best_iteration": int(model.best_iteration_ or 0),
    }


def _blend_reports(
    *,
    train_data: CandidateDataset,
    val_data: CandidateDataset,
    test_data: CandidateDataset,
    models: dict[str, Any],
    model_specs: dict[str, list[int]],
    aux_names: list[str],
) -> dict[str, Any]:
    del train_data
    if "chem_only" not in models:
        return {}
    base_model = models["chem_only"]
    base_indices = model_specs["chem_only"]
    base_val = base_model.predict(val_data.x[:, base_indices], num_iteration=base_model.best_iteration_)
    base_test = base_model.predict(test_data.x[:, base_indices], num_iteration=base_model.best_iteration_)
    out = {}
    for aux_name in aux_names:
        if aux_name not in models:
            continue
        aux_model = models[aux_name]
        aux_indices = model_specs[aux_name]
        aux_val = aux_model.predict(val_data.x[:, aux_indices], num_iteration=aux_model.best_iteration_)
        aux_test = aux_model.predict(test_data.x[:, aux_indices], num_iteration=aux_model.best_iteration_)
        best_alpha = 0.0
        best_score = _metric_for_selection(_evaluate_dataset(val_data, base_val))
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
            score = _metric_for_selection(_evaluate_dataset(val_data, _standardize(base_val) + float(alpha) * _standardize(aux_val)))
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        blended_test = _standardize(base_test) + best_alpha * _standardize(aux_test)
        out[f"chem_only_plus_{aux_name}"] = {
            "base_model": "chem_only",
            "aux_model": aux_name,
            "alpha_selected_on_val": best_alpha,
            "val_selection_score": round(best_score, 6),
            "test": _evaluate_dataset(test_data, blended_test),
        }
    return out


def _coverage_leakage_report(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checks = {
        "doi_cross_split": _cross_split_values(split_rows, lambda row: _norm(row.get("doi"))),
        "target_cross_split": _cross_split_values(split_rows, lambda row: canonical_smiles(str(row.get("target_smiles") or ""))),
        "product_cross_split": _cross_split_values(split_rows, lambda row: canonical_smiles(str(row.get("product_smiles") or ""))),
        "reaction_cross_split": _cross_split_values(split_rows, lambda row: canonical_reaction(str(row.get("rxn_smiles") or ""))),
        "product_main_cross_split": _cross_split_values(split_rows, _product_main_key),
        "product_reactants_cross_split": _cross_split_values(split_rows, _product_reactants_key),
    }
    checks["strict_pass"] = all(int((row or {}).get("count") or 0) == 0 for row in checks.values() if isinstance(row, dict))
    return checks


def _cross_split_values(split_rows: dict[str, list[dict[str, Any]]], getter: Any) -> dict[str, Any]:
    values: dict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            value = str(getter(row) or "")
            if value:
                values[value].add(split)
    offenders = {value: sorted(splits) for value, splits in values.items() if len(splits) > 1}
    return {"count": len(offenders), "examples": dict(sorted(offenders.items())[:25])}


def _product_main_key(row: dict[str, Any]) -> str:
    product = canonical_smiles(str(row.get("product_smiles") or ""))
    main = canonical_smiles(str(row.get("main_reactant") or ""))
    return f"{product}<<{main}" if product and main else ""


def _product_reactants_key(row: dict[str, Any]) -> str:
    product = canonical_smiles(str(row.get("product_smiles") or ""))
    reactants = sorted(canonical_smiles(str(smi or "")) for smi in row.get("reactants") or [])
    reactants = [smi for smi in reactants if smi]
    return f"{product}<<{'.'.join(reactants)}" if product and reactants else ""


def _strict_manifest_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "schema_version": payload.get("schema_version"),
        "counts": payload.get("counts"),
        "splits": {
            split: {"rows": row.get("rows"), "unique_targets": row.get("unique_targets"), "unique_doi": row.get("unique_doi")}
            for split, row in (payload.get("splits") or {}).items()
            if isinstance(row, dict)
        },
        "strict_pass": (payload.get("leakage_checks") or {}).get("strict_pass"),
    }


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v1 Strict Transition Ranker",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Leakage",
        "",
        f"- strict pass: `{(result.get('leakage_checks') or {}).get('strict_pass')}`",
        "",
        "| Check | Count |",
        "|---|---:|",
    ]
    for key, value in (result.get("leakage_checks") or {}).items():
        if isinstance(value, dict):
            lines.append(f"| {key} | {value.get('count')} |")
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Model | Label | Coverage | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all | R@5 covered |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    rows = [("chem_rank", (result.get("baseline_chem_rank") or {}).get("test") or {})]
    for model_name, report in (result.get("models") or {}).items():
        rows.append((model_name, (report.get("test") or {})))
    for model_name, report in (result.get("blends") or {}).items():
        rows.append((model_name, (report.get("test") or {})))
    for model_name, report in rows:
        for label in ("exact_label", "similar_label", "positive_label"):
            metric = report.get(label) or {}
            all_k = metric.get("recall_at_k_all") or {}
            cov_k = metric.get("recall_at_k_covered") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        label,
                        str(metric.get("coverage")),
                        str(metric.get("mrr_covered")),
                        str(all_k.get("1")),
                        str(all_k.get("3")),
                        str(all_k.get("5")),
                        str(all_k.get("10")),
                        str(cov_k.get("5")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CCTS-v1 strict-blind transition ranker")
    ap.add_argument("--train-coverage", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-coverage", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--test-coverage", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--split-manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--similarity-threshold", type=float, default=0.7)
    ap.add_argument("--max-candidates-per-transition", type=int, default=100)
    ap.add_argument("--evidence-pool-size", type=int, default=80)
    ap.add_argument("--n-estimators", type=int, default=320)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report = train_ccts_v1_transition_ranker(
        train_coverage=Path(args.train_coverage),
        train_cache=Path(args.train_cache),
        val_coverage=Path(args.val_coverage),
        val_cache=Path(args.val_cache),
        test_coverage=Path(args.test_coverage),
        test_cache=Path(args.test_cache),
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
        output_dir=Path(args.output_dir),
        similarity_threshold=args.similarity_threshold,
        max_candidates_per_transition=args.max_candidates_per_transition,
        evidence_pool_size=args.evidence_pool_size,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "leakage_checks": report["leakage_checks"],
                "test": {
                    "baseline": report["baseline_chem_rank"]["test"],
                    "models": {name: row["test"] for name, row in report["models"].items()},
                    "blends": {name: row["test"] for name, row in report.get("blends", {}).items()},
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
