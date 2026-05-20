"""Train CBA-v0 as transform-pair classification from cascade context.

This is a cleaner first CBA formulation than candidate-wise LambdaRank:

    context -> P(train transform-pair prototype)

The held-out upstream transform is used only as the class label.  Input features
are restricted to target/downstream context and train-split vocabularies.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import LabelEncoder

from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "cba_v0_pair_classifier.v1"
TOP_KS = (1, 3, 5, 10, 20, 50)


def train_cba_v0_pair_classifier(
    *,
    program_manifest: Path,
    output_dir: Path,
    n_estimators: int = 280,
    learning_rate: float = 0.04,
    seed: int = 42,
    fp_bits: int = 128,
    condition_token_limit: int = 80,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = _load_split_rows(program_manifest)
    train_rows = split_rows["train"]
    val_rows = split_rows["val"]
    test_rows = split_rows["test"]
    encoder = LabelEncoder()
    encoder.fit([row["transform_pair"] for row in train_rows])
    schema = _build_schema(train_rows, fp_bits=fp_bits, condition_token_limit=condition_token_limit)
    train_data = _dataset(train_rows, schema=schema, encoder=encoder, require_seen_label=True)
    val_data = _dataset(val_rows, schema=schema, encoder=encoder, require_seen_label=False)
    test_data = _dataset(test_rows, schema=schema, encoder=encoder, require_seen_label=False)
    models = {}
    reports = {}
    for name, indices in _model_specs(train_data["feature_names"]).items():
        if not indices:
            continue
        model = _fit_classifier(
            train_data,
            val_data,
            indices=indices,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            seed=seed,
        )
        models[name] = model
        reports[name] = {
            "train": _eval_classifier(model, train_data, indices=indices, encoder=encoder),
            "val": _eval_classifier(model, val_data, indices=indices, encoder=encoder),
            "test": _eval_classifier(model, test_data, indices=indices, encoder=encoder),
            "feature_count": len(indices),
            "best_iteration": int(model.best_iteration_ or 0),
        }
    baselines = _baseline_reports(train_data, val_data, test_data, encoder=encoder)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "fp_bits": fp_bits,
            "condition_token_limit": condition_token_limit,
            "elapsed_s": round(time.monotonic() - started, 3),
            "leakage_guard": "classifier labels are held-out transform pairs; features exclude held-out upstream transform and upstream structures",
        },
        "counts": {
            "train_blocks": len(train_rows),
            "val_blocks": len(val_rows),
            "test_blocks": len(test_rows),
            "train_classes": len(encoder.classes_),
            "val_seen_label_blocks": int(np.sum(val_data["seen_label_mask"])),
            "test_seen_label_blocks": int(np.sum(test_data["seen_label_mask"])),
            "val_seen_label_rate": round(float(np.mean(val_data["seen_label_mask"])), 6),
            "test_seen_label_rate": round(float(np.mean(test_data["seen_label_mask"])), 6),
        },
        "baselines": baselines,
        "models": reports,
        "feature_schema": {
            "feature_names": train_data["feature_names"],
            "model_specs": {name: [train_data["feature_names"][idx] for idx in indices] for name, indices in _model_specs(train_data["feature_names"]).items()},
            "classes": list(encoder.classes_),
            **schema,
        },
    }
    with (output_dir / "cba_v0_pair_classifier.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "models": models,
                "label_encoder": encoder,
                "feature_schema": result["feature_schema"],
                "metadata": result["metadata"],
            },
            fh,
        )
    (output_dir / "cba_v0_pair_classifier_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "cba_v0_pair_classifier_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "cba_v0_pair_classifier_test_rankings.jsonl", _ranking_examples(models, test_data, encoder=encoder))
    return result


def _load_split_rows(program_manifest: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    return {
        split: _rows_from_programs(_read_jsonl(Path(outputs[split])), split=split)
        for split in ("train", "val", "test")
    }


def _rows_from_programs(programs: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    rows = []
    for program in programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        compatibility = program.get("compatibility") or {}
        for idx, (left, right) in enumerate(zip(steps, steps[1:])):
            up = _norm(left.get("transformation_superclass"))
            down = _norm(right.get("transformation_superclass"))
            rows.append(
                {
                    "block_id": f"{program.get('program_id')}::{idx}",
                    "split": split,
                    "program_id": str(program.get("program_id") or ""),
                    "doi": str(program.get("doi") or ""),
                    "target_smiles": _canon(program.get("target_smiles")),
                    "downstream_product": _canon(right.get("product_smiles")),
                    "downstream_main_reactant": _canon(right.get("main_reactant")),
                    "downstream_transform": down,
                    "transform_pair": f"{up}->{down}",
                    "cascade_type": _norm(program.get("cascade_type")),
                    "compatibility_label": _norm(compatibility.get("compatibility_label")),
                    "right_catalyst_classes": _norm_list(right.get("catalyst_classes")),
                    "right_condition_tokens": _norm_list(right.get("condition_tokens")),
                }
            )
    return rows


def _build_schema(rows: list[dict[str, Any]], *, fp_bits: int, condition_token_limit: int) -> dict[str, Any]:
    condition_counts = Counter(token for row in rows for token in row["right_condition_tokens"])
    return {
        "fp_bits": int(fp_bits),
        "downstream_transforms": sorted({row["downstream_transform"] for row in rows}),
        "cascade_types": sorted({row["cascade_type"] for row in rows}),
        "compatibility_labels": sorted({row["compatibility_label"] for row in rows}),
        "catalyst_classes": sorted({value for row in rows for value in row["right_catalyst_classes"]}),
        "condition_tokens": [token for token, _count in condition_counts.most_common(int(condition_token_limit))],
    }


def _dataset(rows: list[dict[str, Any]], *, schema: dict[str, Any], encoder: LabelEncoder, require_seen_label: bool) -> dict[str, Any]:
    feature_names = _feature_names(schema)
    x_rows = []
    labels = []
    seen_mask = []
    kept_rows = []
    known = set(str(value) for value in encoder.classes_)
    for row in rows:
        seen = row["transform_pair"] in known
        if require_seen_label and not seen:
            continue
        kept_rows.append(row)
        x_rows.append(_features(row, schema=schema))
        labels.append(int(encoder.transform([row["transform_pair"]])[0]) if seen else -1)
        seen_mask.append(seen)
    return {
        "rows": kept_rows,
        "x": np.asarray(x_rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int32),
        "seen_label_mask": np.asarray(seen_mask, dtype=bool),
        "feature_names": feature_names,
    }


def _features(row: dict[str, Any], *, schema: dict[str, Any]) -> list[float]:
    target_props = _mol_props(row["target_smiles"])
    down_props = _mol_props(row["downstream_product"])
    main_props = _mol_props(row["downstream_main_reactant"])
    values = [
        target_props["heavy_atoms"] / 100.0,
        down_props["heavy_atoms"] / 100.0,
        main_props["heavy_atoms"] / 100.0,
        target_props["rings"] / 10.0,
        down_props["rings"] / 10.0,
        main_props["rings"] / 10.0,
        target_props["hetero_atoms"] / 30.0,
        down_props["hetero_atoms"] / 30.0,
        main_props["hetero_atoms"] / 30.0,
        _tanimoto(row["target_smiles"], row["downstream_product"]),
        _tanimoto(row["downstream_product"], row["downstream_main_reactant"]),
        float(len(row["right_catalyst_classes"])),
        float(len(row["right_condition_tokens"])),
    ]
    values.extend(_one_hot(row["downstream_transform"], schema["downstream_transforms"]))
    values.extend(_one_hot(row["cascade_type"], schema["cascade_types"]))
    values.extend(_one_hot(row["compatibility_label"], schema["compatibility_labels"]))
    values.extend(_multi_hot(row["right_catalyst_classes"], schema["catalyst_classes"]))
    values.extend(_multi_hot(row["right_condition_tokens"], schema["condition_tokens"]))
    values.extend(_fp_bits(row["target_smiles"], int(schema["fp_bits"])))
    values.extend(_fp_bits(row["downstream_product"], int(schema["fp_bits"])))
    values.extend(_fp_bits(row["downstream_main_reactant"], int(schema["fp_bits"])))
    return values


def _feature_names(schema: dict[str, Any]) -> list[str]:
    names = [
        "target__heavy_atoms",
        "downstream__product_heavy_atoms",
        "downstream__main_heavy_atoms",
        "target__rings",
        "downstream__product_rings",
        "downstream__main_rings",
        "target__hetero_atoms",
        "downstream__product_hetero_atoms",
        "downstream__main_hetero_atoms",
        "target__target_downstream_product_sim",
        "downstream__product_main_sim",
        "downstream__n_catalyst_classes",
        "downstream__n_condition_tokens",
    ]
    names.extend([f"downstream__transform={value}" for value in schema["downstream_transforms"]])
    names.extend([f"context__cascade_type={value}" for value in schema["cascade_types"]])
    names.extend([f"context__compatibility_label={value}" for value in schema["compatibility_labels"]])
    names.extend([f"downstream__catalyst_class={value}" for value in schema["catalyst_classes"]])
    names.extend([f"downstream__condition_token={value}" for value in schema["condition_tokens"]])
    names.extend([f"target__fp_{idx}" for idx in range(int(schema["fp_bits"]))])
    names.extend([f"downstream__product_fp_{idx}" for idx in range(int(schema["fp_bits"]))])
    names.extend([f"downstream__main_fp_{idx}" for idx in range(int(schema["fp_bits"]))])
    return names


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    specs: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        if name.startswith("target__") or name.startswith("context__"):
            specs["target_context"].append(idx)
        if name.startswith("downstream__transform="):
            specs["downstream_transform_only"].append(idx)
        if name.startswith("downstream__") or name.startswith("context__"):
            specs["downstream_context"].append(idx)
        if name.startswith("target__") or name.startswith("downstream__") or name.startswith("context__"):
            specs["full_context"].append(idx)
    return dict(specs)


def _fit_classifier(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    *,
    indices: list[int],
    n_estimators: int,
    learning_rate: float,
    seed: int,
) -> LGBMClassifier:
    val_seen = val_data["seen_label_mask"]
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=31,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.05,
        random_state=int(seed),
        verbose=-1,
    )
    model.fit(
        train_data["x"][:, indices],
        train_data["y"],
        eval_set=[(val_data["x"][val_seen][:, indices], val_data["y"][val_seen])],
        callbacks=[early_stopping(35, verbose=False)],
    )
    return model


def _eval_classifier(model: Any, data: dict[str, Any], *, indices: list[int], encoder: LabelEncoder) -> dict[str, Any]:
    proba = model.predict_proba(data["x"][:, indices], num_iteration=model.best_iteration_)
    return _metrics_from_scores(data, proba, encoder=encoder)


def _baseline_reports(train_data: dict[str, Any], val_data: dict[str, Any], test_data: dict[str, Any], *, encoder: LabelEncoder) -> dict[str, Any]:
    train_counts = Counter(train_data["y"].tolist())
    by_downstream = defaultdict(Counter)
    for row, label in zip(train_data["rows"], train_data["y"]):
        by_downstream[row["downstream_transform"]][int(label)] += 1
    reports = {}
    for name, scorer in (
        ("global_frequency", lambda data: _global_frequency_scores(data, train_counts, len(encoder.classes_))),
        ("downstream_conditional_frequency", lambda data: _conditional_frequency_scores(data, by_downstream, train_counts, len(encoder.classes_))),
    ):
        reports[name] = {
            "train": _metrics_from_scores(train_data, scorer(train_data), encoder=encoder),
            "val": _metrics_from_scores(val_data, scorer(val_data), encoder=encoder),
            "test": _metrics_from_scores(test_data, scorer(test_data), encoder=encoder),
        }
    return reports


def _global_frequency_scores(data: dict[str, Any], counts: Counter, n_classes: int) -> np.ndarray:
    base = np.asarray([counts.get(idx, 0) for idx in range(n_classes)], dtype=np.float32)
    return np.tile(base.reshape(1, -1), (len(data["rows"]), 1))


def _conditional_frequency_scores(data: dict[str, Any], by_downstream: dict[str, Counter], global_counts: Counter, n_classes: int) -> np.ndarray:
    rows = []
    for row in data["rows"]:
        counts = by_downstream.get(row["downstream_transform"]) or global_counts
        rows.append([counts.get(idx, 0) for idx in range(n_classes)])
    return np.asarray(rows, dtype=np.float32)


def _metrics_from_scores(data: dict[str, Any], scores: np.ndarray, *, encoder: LabelEncoder) -> dict[str, Any]:
    seen = data["seen_label_mask"]
    labels = data["y"]
    ranks = []
    top1 = []
    for idx in range(len(labels)):
        order = sorted(range(scores.shape[1]), key=lambda col: (-float(scores[idx, col]), str(encoder.classes_[col])))
        top1.append(str(encoder.classes_[order[0]]) if order else "")
        if not seen[idx]:
            ranks.append(None)
            continue
        true_idx = int(labels[idx])
        ranks.append(order.index(true_idx) + 1 if true_idx in order else None)
    covered = [rank for rank in ranks if rank is not None]
    total = len(ranks)
    return {
        "blocks": total,
        "covered_blocks": len(covered),
        "coverage": round(len(covered) / max(total, 1), 6),
        "mrr_all": round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / max(total, 1), 6),
        "mrr_covered": round(sum(1.0 / rank for rank in covered) / max(len(covered), 1), 6) if covered else 0.0,
        "recall_at_k_all": {str(k): round(sum(1 for rank in covered if rank <= k) / max(total, 1), 6) for k in TOP_KS},
        "recall_at_k_covered": {str(k): round(sum(1 for rank in covered if rank <= k) / max(len(covered), 1), 6) for k in TOP_KS},
        "top1_pair_counts": dict(Counter(top1).most_common(20)),
    }


def _ranking_examples(models: dict[str, Any], data: dict[str, Any], *, encoder: LabelEncoder, top_n: int = 10) -> list[dict[str, Any]]:
    if "full_context" in models:
        model_name = "full_context"
    else:
        model_name = sorted(models)[0]
    model = models[model_name]
    indices = _model_specs(data["feature_names"])[model_name]
    proba = model.predict_proba(data["x"][:, indices], num_iteration=model.best_iteration_)
    rows = []
    for idx, row in enumerate(data["rows"][:100]):
        order = sorted(range(proba.shape[1]), key=lambda col: (-float(proba[idx, col]), str(encoder.classes_[col])))
        true_pair = row["transform_pair"]
        rank = None
        if data["seen_label_mask"][idx]:
            true_idx = int(data["y"][idx])
            rank = order.index(true_idx) + 1
        rows.append(
            {
                "block_id": row["block_id"],
                "true_transform_pair": true_pair,
                "positive_rank": rank,
                "top_pairs": [
                    {"transform_pair": str(encoder.classes_[col]), "score": round(float(proba[idx, col]), 6)}
                    for col in order[:top_n]
                ],
            }
        )
    return rows


def _fp_bits(smiles: str, n_bits: int) -> list[float]:
    fp = _fp(smiles, n_bits)
    if fp is None:
        return [0.0 for _ in range(n_bits)]
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return [float(value) for value in arr]


def _fp(smiles: str, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(n_bits))


def _tanimoto(left: str, right: str) -> float:
    left_fp = _fp(left)
    right_fp = _fp(right)
    if left_fp is None or right_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _mol_props(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"heavy_atoms": 0.0, "rings": 0.0, "hetero_atoms": 0.0, "mw": 0.0}
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "rings": float(Descriptors.RingCount(mol)),
        "hetero_atoms": float(hetero),
        "mw": float(Descriptors.MolWt(mol)),
    }


def _canon(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _norm(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _norm_list(values: Any) -> list[str]:
    return sorted({_norm(value) for value in (values or []) if _norm(value) != "unknown"})


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == item) for item in values]


def _multi_hot(values: Any, schema_values: list[str]) -> list[float]:
    present = {str(value) for value in (values or [])}
    return [float(value in present) for value in schema_values]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CBA-v0 Pair Classifier",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Model | Coverage | MRR all | R@1 | R@3 | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {}
    for name, payload in (result.get("baselines") or {}).items():
        rows[name] = payload.get("test") or {}
    for name, payload in (result.get("models") or {}).items():
        rows[name] = payload.get("test") or {}
    for name, metric in rows.items():
        at = metric.get("recall_at_k_all") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    str(metric.get("coverage")),
                    str(metric.get("mrr_all")),
                    str(at.get("1")),
                    str(at.get("3")),
                    str(at.get("5")),
                    str(at.get("10")),
                    str(at.get("20")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This classifier predicts transform-pair prototypes from target/downstream context. "
            "It is the clean CBA-v0 test for whether v4 context contains learnable block-selection signal.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Train CBA-v0 pair classifier")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-estimators", type=int, default=280)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp-bits", type=int, default=128)
    ap.add_argument("--condition-token-limit", type=int, default=80)
    args = ap.parse_args()
    result = train_cba_v0_pair_classifier(
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fp_bits=args.fp_bits,
        condition_token_limit=args.condition_token_limit,
    )
    summary = {
        "counts": result["counts"],
        "test": {
            **{name: payload["test"] for name, payload in result["baselines"].items()},
            **{name: payload["test"] for name, payload in result["models"].items()},
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
