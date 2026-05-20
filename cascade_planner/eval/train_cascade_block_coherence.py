"""Train and evaluate a CascadeBlock coherence scorer."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import average_precision_score, roc_auc_score

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.build_ccts_v3_runtime_evidence_cache import _runtime_evidence_scores


BLOCK_COHERENCE_SCHEMA_VERSION = "cascade_block_coherence_scorer.v1"


def train_cascade_block_coherence(
    *,
    pack_manifest: Path,
    output_dir: Path,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    seed: int = 42,
    enable_runtime_evidence: bool = False,
    program_manifest: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(pack_manifest.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    train_rows = _read_jsonl(Path(outputs["train"]))
    val_rows = _read_jsonl(Path(outputs["val"]))
    test_rows = _read_jsonl(Path(outputs["test"]))
    evidence = _train_evidence(train_rows)
    if enable_runtime_evidence:
        resolved_program_manifest = program_manifest or Path((manifest.get("metadata") or {}).get("program_manifest") or "")
        if not resolved_program_manifest:
            raise ValueError("--enable-runtime-evidence requires --program-manifest or manifest metadata.program_manifest")
        evidence["runtime_train_bank"] = _train_bank(resolved_program_manifest)
        evidence["runtime_product_sim_cache"] = {}
    schema = _build_schema(train_rows)
    schema["runtime_evidence_features_enabled"] = bool(enable_runtime_evidence)
    train_data = _dataset(train_rows, schema=schema, evidence=evidence)
    val_data = _dataset(val_rows, schema=schema, evidence=evidence)
    test_data = _dataset(test_rows, schema=schema, evidence=evidence)
    model_specs = _model_specs(train_data["feature_names"])
    models = {}
    reports = {}
    for name, indices in model_specs.items():
        if not indices:
            continue
        model = _fit_classifier(
            train_data,
            val_data,
            feature_indices=indices,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            seed=seed,
        )
        models[name] = model
        reports[name] = _model_report(model, train_data=train_data, val_data=val_data, test_data=test_data, feature_indices=indices)
    result = {
        "schema_version": BLOCK_COHERENCE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "pack_manifest": str(pack_manifest),
            "output_dir": str(output_dir),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "enable_runtime_evidence": bool(enable_runtime_evidence),
            "program_manifest": str(program_manifest) if program_manifest else str((manifest.get("metadata") or {}).get("program_manifest") or ""),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_positive_rows": int(np.sum(train_data["y"])),
            "val_positive_rows": int(np.sum(val_data["y"])),
            "test_positive_rows": int(np.sum(test_data["y"])),
            "test_example_type_counts": dict(Counter(row.get("example_type") for row in test_rows)),
        },
        "models": reports,
        "feature_schema": {
            "feature_names": train_data["feature_names"],
            "model_specs": {name: [train_data["feature_names"][idx] for idx in indices] for name, indices in model_specs.items()},
            **schema,
        },
    }
    with (output_dir / "cascade_block_coherence_models.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": BLOCK_COHERENCE_SCHEMA_VERSION,
                "models": models,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "cascade_block_coherence_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "cascade_block_coherence_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _dataset(rows: list[dict[str, Any]], *, schema: dict[str, Any], evidence: dict[str, Counter]) -> dict[str, Any]:
    feature_names = _feature_names(schema)
    x = [_feature_vector(row, schema=schema, evidence=evidence) for row in rows]
    y = [int(row.get("label") or 0) for row in rows]
    return {
        "rows": rows,
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "feature_names": feature_names,
    }


def _feature_vector(row: dict[str, Any], *, schema: dict[str, Any], evidence: dict[str, Counter]) -> list[float]:
    left = row.get("left_step") or {}
    right = row.get("right_step") or {}
    left_main = str(left.get("main_reactant") or "")
    left_product = str(left.get("product_smiles") or "")
    right_main = str(right.get("main_reactant") or "")
    right_product = str(right.get("product_smiles") or "")
    left_props = _mol_props(left_product)
    right_props = _mol_props(right_product)
    left_main_props = _mol_props(left_main)
    right_main_props = _mol_props(right_main)
    pair = _transform_pair(left, right)
    cat_pair = _catalyst_pair(left, right)
    left_tokens = set(left.get("condition_tokens") or [])
    right_tokens = set(right.get("condition_tokens") or [])
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    evidence_features = [
        math.log1p(evidence["transform_pairs"].get(pair, 0)),
        math.log1p(evidence["catalyst_pairs"].get(cat_pair, 0)),
        math.log1p(evidence["left_transform"].get(str(left.get("transformation_superclass") or ""), 0)),
        math.log1p(evidence["right_transform"].get(str(right.get("transformation_superclass") or ""), 0)),
        float(pair in evidence["transform_pairs"]),
        float(cat_pair in evidence["catalyst_pairs"]),
    ]
    values = [
        float(int(right.get("step_pos") or 0) - int(left.get("step_pos") or 0)),
        float(int(left.get("remaining_steps") or 0)),
        float(int(right.get("remaining_steps") or 0)),
        float(bool(left.get("intermediate_isolated") is False)),
        float(bool(left.get("intermediate_isolated") is True)),
        float(bool(right.get("pseudo_terminal_stock"))),
        float(bool(right.get("metadata_corrupted"))),
        float(len(left.get("catalyst_classes") or [])),
        float(len(right.get("catalyst_classes") or [])),
        float(len(left_tokens)),
        float(len(right_tokens)),
        float(overlap),
        float(overlap / union) if union else 0.0,
        _tanimoto(left_product, right_main),
        _tanimoto(left_product, right_product),
        _tanimoto(left_main, right_main),
        left_props["heavy_atoms"] - right_main_props["heavy_atoms"],
        right_props["heavy_atoms"] - left_props["heavy_atoms"],
        abs(left_props["mw"] - right_props["mw"]) / 500.0,
        abs(left_main_props["mw"] - right_main_props["mw"]) / 500.0,
        *evidence_features,
    ]
    if schema.get("runtime_evidence_features_enabled"):
        values.extend(_runtime_block_features(left, right, evidence=evidence))
    for key, source in (
        ("route_domains", row.get("route_domain")),
        ("compatibility_labels", row.get("compatibility_label")),
        ("left_transforms", left.get("transformation_superclass")),
        ("right_transforms", right.get("transformation_superclass")),
        ("transform_pairs", pair),
        ("left_modes", left.get("step_mode")),
        ("right_modes", right.get("step_mode")),
        ("pairwise_modes", right.get("pairwise_mode") or left.get("pairwise_mode")),
        ("catalyst_pairs", cat_pair),
    ):
        values.extend(_one_hot(str(source or ""), schema[key]))
    values.extend(_multi_hot(left.get("catalyst_classes") or [], schema["catalyst_classes"]))
    values.extend(_multi_hot(right.get("catalyst_classes") or [], schema["catalyst_classes"]))
    values.extend(_multi_hot(left.get("condition_tokens") or [], schema["condition_tokens"]))
    values.extend(_multi_hot(right.get("condition_tokens") or [], schema["condition_tokens"]))
    values.extend(_fp_bits(left_product, int(schema["n_bits"])))
    values.extend(_fp_bits(right_main, int(schema["n_bits"])))
    return values


def _train_evidence(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    out = {
        "transform_pairs": Counter(),
        "catalyst_pairs": Counter(),
        "left_transform": Counter(),
        "right_transform": Counter(),
    }
    for row in rows:
        if int(row.get("label") or 0) != 1:
            continue
        left = row.get("left_step") or {}
        right = row.get("right_step") or {}
        out["transform_pairs"][_transform_pair(left, right)] += 1
        out["catalyst_pairs"][_catalyst_pair(left, right)] += 1
        out["left_transform"][str(left.get("transformation_superclass") or "")] += 1
        out["right_transform"][str(right.get("transformation_superclass") or "")] += 1
    return out


def _build_schema(rows: list[dict[str, Any]], *, n_bits: int = 128) -> dict[str, Any]:
    route_domains = set()
    compatibility_labels = set()
    left_transforms = set()
    right_transforms = set()
    transform_pairs = set()
    left_modes = set()
    right_modes = set()
    pairwise_modes = set()
    catalyst_pairs = set()
    catalyst_classes = set()
    condition_tokens = set()
    for row in rows:
        left = row.get("left_step") or {}
        right = row.get("right_step") or {}
        route_domains.add(str(row.get("route_domain") or ""))
        compatibility_labels.add(str(row.get("compatibility_label") or ""))
        left_transforms.add(str(left.get("transformation_superclass") or ""))
        right_transforms.add(str(right.get("transformation_superclass") or ""))
        transform_pairs.add(_transform_pair(left, right))
        left_modes.add(str(left.get("step_mode") or ""))
        right_modes.add(str(right.get("step_mode") or ""))
        pairwise_modes.add(str(right.get("pairwise_mode") or left.get("pairwise_mode") or ""))
        catalyst_pairs.add(_catalyst_pair(left, right))
        catalyst_classes.update(str(value) for value in left.get("catalyst_classes") or [])
        catalyst_classes.update(str(value) for value in right.get("catalyst_classes") or [])
        condition_tokens.update(str(value) for value in left.get("condition_tokens") or [])
        condition_tokens.update(str(value) for value in right.get("condition_tokens") or [])
    return {
        "n_bits": int(n_bits),
        "route_domains": sorted(route_domains),
        "compatibility_labels": sorted(compatibility_labels),
        "left_transforms": sorted(left_transforms),
        "right_transforms": sorted(right_transforms),
        "transform_pairs": sorted(transform_pairs),
        "left_modes": sorted(left_modes),
        "right_modes": sorted(right_modes),
        "pairwise_modes": sorted(pairwise_modes),
        "catalyst_pairs": sorted(catalyst_pairs),
        "catalyst_classes": sorted(catalyst_classes),
        "condition_tokens": sorted(condition_tokens)[:256],
    }


def _feature_names(schema: dict[str, Any]) -> list[str]:
    names = [
        "structure__step_pos_delta",
        "context__left_remaining_steps",
        "context__right_remaining_steps",
        "context__left_nonisolated",
        "context__left_isolated",
        "context__right_terminal_stock",
        "context__right_metadata_corrupted",
        "context__left_n_catalyst_classes",
        "context__right_n_catalyst_classes",
        "context__left_n_condition_tokens",
        "context__right_n_condition_tokens",
        "context__condition_overlap",
        "context__condition_jaccard",
        "structure__left_product_right_main_sim",
        "structure__left_product_right_product_sim",
        "structure__left_main_right_main_sim",
        "structure__left_product_minus_right_main_heavy",
        "structure__right_product_minus_left_product_heavy",
        "structure__left_right_product_mw_delta",
        "structure__left_right_main_mw_delta",
        "evidence__log_transform_pair_count",
        "evidence__log_catalyst_pair_count",
        "evidence__log_left_transform_count",
        "evidence__log_right_transform_count",
        "evidence__has_transform_pair",
        "evidence__has_catalyst_pair",
    ]
    if schema.get("runtime_evidence_features_enabled"):
        names.extend(_runtime_feature_names())
    for key in (
        "route_domains",
        "compatibility_labels",
        "left_transforms",
        "right_transforms",
        "transform_pairs",
        "left_modes",
        "right_modes",
        "pairwise_modes",
        "catalyst_pairs",
    ):
        prefix = "context" if key not in ("transform_pairs", "catalyst_pairs") else "evidence"
        names.extend([f"{prefix}__{key}={value}" for value in schema[key]])
    for key in ("catalyst_classes", "condition_tokens"):
        names.extend([f"context__left_{key}={value}" for value in schema[key]])
        names.extend([f"context__right_{key}={value}" for value in schema[key]])
    names.extend([f"structure__left_product_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    names.extend([f"structure__right_main_fp_{idx}" for idx in range(int(schema["n_bits"]))])
    return names


def _runtime_feature_names() -> list[str]:
    per_step = [
        "any_transition_sim",
        "any_product_sim",
        "any_main_sim",
        "pair_compatible_sim",
        "pair_bucket_log",
        "prev_pair_supported",
        "next_pair_supported",
        "inferred_transform_prior",
    ]
    names = [f"runtime__left_{name}" for name in per_step]
    names.extend(f"runtime__right_{name}" for name in per_step)
    names.extend(
        [
            "runtime__block_any_mean",
            "runtime__block_any_min",
            "runtime__block_pair_mean",
            "runtime__block_pair_min",
        ]
    )
    return names


def _runtime_block_features(left: dict[str, Any], right: dict[str, Any], *, evidence: dict[str, Any]) -> list[float]:
    left_transform = str(left.get("transformation_superclass") or "")
    right_transform = str(right.get("transformation_superclass") or "")
    left_scores = _runtime_step_features(
        left,
        previous_transform=str(left.get("previous_transformation_superclass") or ""),
        next_transform=right_transform or str(left.get("next_transformation_superclass") or ""),
        evidence=evidence,
    )
    right_scores = _runtime_step_features(
        right,
        previous_transform=left_transform or str(right.get("previous_transformation_superclass") or ""),
        next_transform=str(right.get("next_transformation_superclass") or ""),
        evidence=evidence,
    )
    left_any = left_scores[0]
    right_any = right_scores[0]
    left_pair = left_scores[3]
    right_pair = right_scores[3]
    return [
        *left_scores,
        *right_scores,
        (left_any + right_any) / 2.0,
        min(left_any, right_any),
        (left_pair + right_pair) / 2.0,
        min(left_pair, right_pair),
    ]


def _runtime_step_features(
    step: dict[str, Any],
    *,
    previous_transform: str,
    next_transform: str,
    evidence: dict[str, Any],
) -> list[float]:
    train_bank = evidence.get("runtime_train_bank")
    if not train_bank:
        return [0.0] * 8
    scores = _runtime_evidence_scores(
        product=str(step.get("product_smiles") or ""),
        candidate_main=str(step.get("main_reactant") or ""),
        previous_transform=str(previous_transform or ""),
        next_transform=str(next_transform or ""),
        train_bank=train_bank,
        product_sim_cache=evidence.setdefault("runtime_product_sim_cache", {}),
    )
    return [
        _float(scores.get("runtime_nearest_any_transition_sim")),
        _float(scores.get("runtime_nearest_any_product_sim")),
        _float(scores.get("runtime_nearest_any_main_sim")),
        _float(scores.get("runtime_nearest_pair_compatible_sim")),
        math.log1p(_float(scores.get("runtime_pair_bucket_size"))),
        float(bool(scores.get("runtime_prev_pair_supported"))),
        float(bool(scores.get("runtime_next_pair_supported"))),
        _float(scores.get("runtime_inferred_transform_prior")),
    ]


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("structure__"):
            groups["structure_only"].append(idx)
        if name.startswith("context__"):
            groups["context_only"].append(idx)
        if name.startswith("evidence__"):
            groups["evidence_only"].append(idx)
    groups["structure_plus_context"] = groups["structure_only"] + groups["context_only"]
    groups["structure_plus_evidence"] = groups["structure_only"] + groups["evidence_only"]
    groups["route_pool_compatible"] = [
        idx
        for idx, name in enumerate(feature_names)
        if name.startswith("structure__")
        or name.startswith("context__left_transforms=")
        or name.startswith("context__right_transforms=")
        or name.startswith("evidence__transform_pairs=")
        or name.startswith("context__left_remaining_steps")
        or name.startswith("context__right_remaining_steps")
    ]
    groups["full"] = list(range(len(feature_names)))
    return dict(groups)


def _fit_classifier(train_data: dict[str, Any], val_data: dict[str, Any], *, feature_indices: list[int], n_estimators: int, learning_rate: float, seed: int) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        train_data["x"][:, feature_indices],
        train_data["y"],
        eval_set=[(val_data["x"][:, feature_indices], val_data["y"])],
        callbacks=[early_stopping(40, verbose=False)],
    )
    return model


def _model_report(model: Any, *, train_data: dict[str, Any], val_data: dict[str, Any], test_data: dict[str, Any], feature_indices: list[int]) -> dict[str, Any]:
    return {
        "train": _eval_scores(train_data, _predict(model, train_data, feature_indices)),
        "val": _eval_scores(val_data, _predict(model, val_data, feature_indices)),
        "test": _eval_scores(test_data, _predict(model, test_data, feature_indices)),
        "feature_count": len(feature_indices),
        "best_iteration": int(model.best_iteration_ or 0),
    }


def _predict(model: Any, data: dict[str, Any], indices: list[int]) -> np.ndarray:
    return model.predict_proba(data["x"][:, indices], num_iteration=model.best_iteration_)[:, 1]


def _eval_scores(data: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    y = data["y"]
    out = {
        "overall": _binary_metrics(y, scores),
        "by_example_type": {},
    }
    positives = [idx for idx, row in enumerate(data["rows"]) if int(row.get("label") or 0) == 1]
    for example_type in sorted({row.get("example_type") for row in data["rows"] if row.get("example_type") != "positive_adjacent"}):
        idxs = positives + [idx for idx, row in enumerate(data["rows"]) if row.get("example_type") == example_type]
        out["by_example_type"][str(example_type)] = _binary_metrics(y[idxs], scores[idxs])
    return out


def _binary_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int32)
    preds = np.asarray(scores, dtype=np.float32)
    pred_label = preds >= 0.5
    tp = int(np.sum((pred_label == 1) & (labels == 1)))
    tn = int(np.sum((pred_label == 0) & (labels == 0)))
    fp = int(np.sum((pred_label == 1) & (labels == 0)))
    fn = int(np.sum((pred_label == 0) & (labels == 1)))
    try:
        auc = float(roc_auc_score(labels, preds)) if len(set(labels.tolist())) > 1 else None
    except ValueError:
        auc = None
    try:
        ap = float(average_precision_score(labels, preds)) if len(set(labels.tolist())) > 1 else None
    except ValueError:
        ap = None
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(np.mean(labels)) if len(labels) else 0.0, 6),
        "roc_auc": round(auc, 6) if auc is not None else None,
        "average_precision": round(ap, 6) if ap is not None else None,
        "accuracy_at_0_5": round(float((tp + tn) / max(len(labels), 1)), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _transform_pair(left: dict[str, Any], right: dict[str, Any]) -> str:
    return f"{left.get('transformation_superclass') or ''}->{right.get('transformation_superclass') or ''}"


def _catalyst_pair(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_cls = ".".join(sorted(str(value) for value in left.get("catalyst_classes") or ["unknown"]))
    right_cls = ".".join(sorted(str(value) for value in right.get("catalyst_classes") or ["unknown"]))
    return f"{left_cls}->{right_cls}"


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == item) for item in values]


def _multi_hot(values: list[str], schema_values: list[str]) -> list[float]:
    present = {str(value) for value in values}
    return [float(value in present) for value in schema_values]


def _fp_bits(smiles: str, n_bits: int) -> list[float]:
    fp = _fp(str(smiles or ""), n_bits)
    if fp is None:
        return [0.0 for _ in range(n_bits)]
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return [float(value) for value in arr]


def _tanimoto(left: str, right: str) -> float:
    fp_left = _fp(str(left or ""), 1024)
    fp_right = _fp(str(right or ""), 1024)
    if fp_left is None or fp_right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_left, fp_right))


def _fp(smiles: str, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(n_bits))


def _mol_props(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"heavy_atoms": 0.0, "mw": 0.0}
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mw": float(Descriptors.MolWt(mol)),
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CascadeBlock Coherence Scorer",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Model | AUC | AP | Accuracy | Best Iter |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in (result.get("models") or {}).items():
        metric = ((report.get("test") or {}).get("overall") or {})
        lines.append(
            f"| {name} | {metric.get('roc_auc')} | {metric.get('average_precision')} | {metric.get('accuracy_at_0_5')} | {report.get('best_iteration')} |"
        )
    lines.extend(["", "## Test By Negative Type", ""])
    for name, report in (result.get("models") or {}).items():
        lines.extend([f"### {name}", "", "| Negative Type | AUC | AP | Accuracy |", "|---|---:|---:|---:|"])
        for neg_type, metric in (((report.get("test") or {}).get("by_example_type") or {}).items()):
            lines.append(
                f"| {neg_type} | {metric.get('roc_auc')} | {metric.get('average_precision')} | {metric.get('accuracy_at_0_5')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CascadeBlock coherence scorer")
    ap.add_argument("--pack-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-runtime-evidence", action="store_true")
    ap.add_argument("--program-manifest")
    args = ap.parse_args()
    result = train_cascade_block_coherence(
        pack_manifest=Path(args.pack_manifest),
        output_dir=Path(args.output_dir),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        enable_runtime_evidence=args.enable_runtime_evidence,
        program_manifest=Path(args.program_manifest) if args.program_manifest else None,
    )
    summary = {
        "counts": result["counts"],
        "test_overall": {
            name: (report.get("test") or {}).get("overall")
            for name, report in (result.get("models") or {}).items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
