"""Train bridge enzyme-feasibility verifier v0.

The v0 model is deliberately simple and auditable:

* Morgan fingerprint features for chemical and enzyme-side molecules.
* Shared-bit features for structural overlap.
* Numeric pair features: Tanimoto, heavy atom counts, heavy atom deltas, EC count,
  same-InChIKey, and direction one-hot features.
* LightGBM binary classifier with sample weights from the verifier pack.

Outputs include the model, metrics, per-label-type analysis, and a markdown
report suitable for project review.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
CHEM_OFFSET = 0
ENZYME_OFFSET = FP_BITS
SHARED_OFFSET = FP_BITS * 2
NUMERIC_OFFSET = FP_BITS * 3

DIRECTION_VALUES = [
    "chemical_product_to_enzyme_product",
    "chemical_product_to_enzyme_substrate",
    "chemical_product_to_similar_enzyme_product",
    "chemical_product_to_similar_enzyme_substrate",
    "chemical_product_to_similar_enzyme_molecule",
]

NUMERIC_FEATURES = [
    "tanimoto",
    "same_inchikey",
    "chem_heavy",
    "enzyme_heavy",
    "heavy_abs_delta",
    "heavy_signed_delta",
    "heavy_ratio_min_over_max",
    "ec_count",
    "ec_known",
    "chemical_ring_count",
    "enzyme_ring_count",
    "ring_abs_delta",
    "chemical_hetero_count",
    "enzyme_hetero_count",
    "hetero_abs_delta",
]

DIRECTION_FEATURES = [f"direction={value}" for value in DIRECTION_VALUES] + ["direction=other"]
FEATURE_NAMES = (
    [f"chem_morgan_{i}" for i in range(FP_BITS)]
    + [f"enzyme_morgan_{i}" for i in range(FP_BITS)]
    + [f"shared_morgan_{i}" for i in range(FP_BITS)]
    + NUMERIC_FEATURES
    + DIRECTION_FEATURES
)


class MoleculeCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.invalid = Counter()

    def get(self, smiles: str) -> dict[str, Any]:
        smiles = str(smiles or "")
        if smiles in self.cache:
            return self.cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.invalid["mol_from_smiles_failed"] += 1
            info = {
                "fp": None,
                "bits": [],
                "heavy": 0,
                "rings": 0,
                "hetero": 0,
            }
            self.cache[smiles] = info
            return info
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS)
        bits = list(fp.GetOnBits())
        info = {
            "fp": fp,
            "bits": bits,
            "heavy": int(mol.GetNumHeavyAtoms()),
            "rings": int(mol.GetRingInfo().NumRings()),
            "hetero": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
        }
        self.cache[smiles] = info
        return info


def parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    return []


def load_split(path: Path, *, limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if limit is not None and len(rows) > limit:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(rows)), limit))
        return [rows[idx] for idx in indices]
    return rows


def numeric_values(row: dict[str, Any], chem: dict[str, Any], enzyme: dict[str, Any]) -> list[float]:
    chem_heavy = float(chem["heavy"])
    enzyme_heavy = float(enzyme["heavy"])
    max_heavy = max(chem_heavy, enzyme_heavy, 1.0)
    min_heavy = min(chem_heavy, enzyme_heavy)
    ec_count = float(len(set(parse_json_list(row.get("enzyme_ec_sample_json")))))
    return [
        float(row.get("tanimoto") or 0.0),
        1.0 if row.get("chemical_inchikey") == row.get("enzyme_inchikey") else 0.0,
        chem_heavy,
        enzyme_heavy,
        abs(chem_heavy - enzyme_heavy),
        chem_heavy - enzyme_heavy,
        min_heavy / max_heavy,
        ec_count,
        1.0 if ec_count > 0 else 0.0,
        float(chem["rings"]),
        float(enzyme["rings"]),
        abs(float(chem["rings"]) - float(enzyme["rings"])),
        float(chem["hetero"]),
        float(enzyme["hetero"]),
        abs(float(chem["hetero"]) - float(enzyme["hetero"])),
    ]


def build_matrix(rows: list[dict[str, Any]], cache: MoleculeCache) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    labels = np.zeros(len(rows), dtype=np.int8)
    weights = np.ones(len(rows), dtype=np.float32)
    label_types: list[str] = []
    chemical_keys: list[str] = []
    tanimoto_scores = np.zeros(len(rows), dtype=np.float32)

    for i, row in enumerate(rows):
        chem = cache.get(row.get("chemical_smiles") or "")
        enzyme = cache.get(row.get("enzyme_smiles") or "")
        labels[i] = int(row.get("label") or 0)
        weights[i] = float(row.get("label_weight") or 1.0)
        label_types.append(str(row.get("label_type") or "unknown"))
        chemical_keys.append(str(row.get("chemical_inchikey") or ""))
        tanimoto_scores[i] = float(row.get("tanimoto") or 0.0)

        chem_bits = chem["bits"]
        enzyme_bits = enzyme["bits"]
        enzyme_bit_set = set(enzyme_bits)
        for bit in chem_bits:
            row_idx.append(i)
            col_idx.append(CHEM_OFFSET + bit)
            data.append(1.0)
        for bit in enzyme_bits:
            row_idx.append(i)
            col_idx.append(ENZYME_OFFSET + bit)
            data.append(1.0)
        for bit in chem_bits:
            if bit in enzyme_bit_set:
                row_idx.append(i)
                col_idx.append(SHARED_OFFSET + bit)
                data.append(1.0)

        for j, value in enumerate(numeric_values(row, chem, enzyme)):
            if value != 0.0:
                row_idx.append(i)
                col_idx.append(NUMERIC_OFFSET + j)
                data.append(float(value))

        direction = str(row.get("bridge_direction") or "")
        try:
            direction_idx = DIRECTION_VALUES.index(direction)
        except ValueError:
            direction_idx = len(DIRECTION_VALUES)
        row_idx.append(i)
        col_idx.append(NUMERIC_OFFSET + len(NUMERIC_FEATURES) + direction_idx)
        data.append(1.0)

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(row_idx, dtype=np.int32), np.asarray(col_idx, dtype=np.int32))),
        shape=(len(rows), len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    return matrix, labels, weights, label_types, chemical_keys, tanimoto_scores


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, score))


def binary_metrics(y_true: np.ndarray, score: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[1],
        zero_division=0,
    )
    return {
        "roc_auc": safe_auc(y_true, score),
        "pr_auc": safe_ap(y_true, score),
        "accuracy_at_0_5": float(accuracy_score(y_true, pred)),
        "precision_at_0_5": float(precision[0]) if len(precision) else 0.0,
        "recall_at_0_5": float(recall[0]) if len(recall) else 0.0,
        "f1_at_0_5": float(f1[0]) if len(f1) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_rate": float(np.mean(y_true)),
        "score_mean": float(np.mean(score)),
    }


def by_label_type(y_true: np.ndarray, score: np.ndarray, label_types: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, label_type in enumerate(label_types):
        grouped[label_type].append(idx)
    out: dict[str, dict[str, Any]] = {}
    for label_type, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int32)
        y = y_true[idx]
        s = score[idx]
        row = {
            "rows": int(len(idx)),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == 0)),
            "mean_score": float(np.mean(s)),
            "median_score": float(np.median(s)),
        }
        if row["negatives"]:
            row["rejection_rate_score_lt_0_5"] = float(np.mean(s < 0.5))
            row["rejection_rate_score_lt_0_2"] = float(np.mean(s < 0.2))
        if row["positives"]:
            row["recall_score_ge_0_5"] = float(np.mean(s >= 0.5))
            row["recall_score_ge_0_2"] = float(np.mean(s >= 0.2))
        out[label_type] = row
    return out


def ranking_metrics(y_true: np.ndarray, score: np.ndarray, chemical_keys: list[str], *, max_groups: int | None = None) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(chemical_keys):
        groups[key].append(idx)
    ranks = []
    recall_at = Counter()
    usable_groups = 0
    for indices in groups.values():
        if max_groups is not None and usable_groups >= max_groups:
            break
        labels = y_true[indices]
        if not np.any(labels == 1) or not np.any(labels == 0):
            continue
        usable_groups += 1
        order = sorted(indices, key=lambda i: float(score[i]), reverse=True)
        first_positive_rank = None
        for rank, idx in enumerate(order, 1):
            if y_true[idx] == 1:
                first_positive_rank = rank
                break
        if first_positive_rank is None:
            continue
        ranks.append(first_positive_rank)
        for k in [1, 3, 5, 10]:
            if first_positive_rank <= k:
                recall_at[k] += 1
    if not ranks:
        return {"groups": 0}
    return {
        "groups": int(len(ranks)),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_first_positive_rank": float(np.mean(ranks)),
        "recall_at_1": float(recall_at[1] / len(ranks)),
        "recall_at_3": float(recall_at[3] / len(ranks)),
        "recall_at_5": float(recall_at[5] / len(ranks)),
        "recall_at_10": float(recall_at[10] / len(ranks)),
    }


def feature_importance(model: lgb.LGBMClassifier, top_n: int = 40) -> list[dict[str, Any]]:
    importances = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature": FEATURE_NAMES[int(idx)],
            "gain": float(importances[int(idx)]),
        }
        for idx in order
        if importances[int(idx)] > 0
    ]


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Verifier v0 Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- output_dir: `{report['output_dir']}`",
        f"- train_rows: `{report['data']['train_rows']}`",
        f"- valid_rows: `{report['data']['valid_rows']}`",
        f"- test_rows: `{report['data']['test_rows']}`",
        "",
        "## Metrics",
        "",
        "| Split | Model ROC-AUC | Model PR-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Tanimoto PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["valid", "test"]:
        model = report["metrics"][split]["model"]
        baseline = report["metrics"][split]["tanimoto_baseline"]
        lines.append(
            f"| {split} | {fmt(model.get('roc_auc'))} | {fmt(model.get('pr_auc'))} | "
            f"{fmt(model.get('precision_at_0_5'))} | {fmt(model.get('recall_at_0_5'))} | "
            f"{fmt(model.get('f1_at_0_5'))} | {fmt(baseline.get('pr_auc'))} |"
        )
    lines.extend(["", "## Hard Negative Rejection On Test", "", "| Label Type | Rows | Mean Score | Reject <0.5 | Reject <0.2 |", "|---|---:|---:|---:|---:|"])
    by_type = report["metrics"]["test"]["by_label_type"]
    for label_type, row in sorted(by_type.items()):
        if row.get("negatives", 0) <= 0:
            continue
        lines.append(
            f"| `{label_type}` | {row['rows']} | {fmt(row['mean_score'])} | "
            f"{fmt(row.get('rejection_rate_score_lt_0_5'))} | {fmt(row.get('rejection_rate_score_lt_0_2'))} |"
        )
    lines.extend(["", "## Positive Recall On Test", "", "| Label Type | Rows | Mean Score | Recall >=0.5 | Recall >=0.2 |", "|---|---:|---:|---:|---:|"])
    for label_type, row in sorted(by_type.items()):
        if row.get("positives", 0) <= 0:
            continue
        lines.append(
            f"| `{label_type}` | {row['rows']} | {fmt(row['mean_score'])} | "
            f"{fmt(row.get('recall_score_ge_0_5'))} | {fmt(row.get('recall_score_ge_0_2'))} |"
        )
    lines.extend(["", "## Ranking", "", "| Split | Groups | MRR | R@1 | R@3 | R@5 | R@10 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for split in ["valid", "test"]:
        ranking = report["metrics"][split]["ranking"]
        lines.append(
            f"| {split} | {ranking.get('groups', 0)} | {fmt(ranking.get('mrr'))} | "
            f"{fmt(ranking.get('recall_at_1'))} | {fmt(ranking.get('recall_at_3'))} | "
            f"{fmt(ranking.get('recall_at_5'))} | {fmt(ranking.get('recall_at_10'))} |"
        )
    lines.extend(["", "## Top Feature Importance", "", "| Feature | Gain |", "|---|---:|"])
    for row in report["feature_importance_top"]:
        lines.append(f"| `{row['feature']}` | {row['gain']:.3f} |")
    lines.extend(["", "## Files", "", "| Artifact | Path |", "|---|---|"])
    for key, value in report["files"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bridge verifier v0")
    parser.add_argument("--pack-dir", default="data/bridge_pack_v0")
    parser.add_argument("--output-dir", default="results/shared/bridge_verifier_v0_20260527")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    started = time.time()
    pack_dir = Path(args.pack_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(pack_dir / "verifier_train.parquet", limit=args.train_limit, seed=args.seed)
    valid_rows = load_split(pack_dir / "verifier_valid.parquet", limit=args.valid_limit, seed=args.seed + 1)
    test_rows = load_split(pack_dir / "verifier_test.parquet", limit=args.test_limit, seed=args.seed + 2)

    cache = MoleculeCache()
    print(f"[verifier_v0] build train matrix rows={len(train_rows)}")
    x_train, y_train, w_train, train_types, train_keys, train_tanimoto = build_matrix(train_rows, cache)
    print(f"[verifier_v0] build valid matrix rows={len(valid_rows)}")
    x_valid, y_valid, w_valid, valid_types, valid_keys, valid_tanimoto = build_matrix(valid_rows, cache)
    print(f"[verifier_v0] build test matrix rows={len(test_rows)}")
    x_test, y_test, w_test, test_types, test_keys, test_tanimoto = build_matrix(test_rows, cache)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    print("[verifier_v0] train LightGBM")
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric=["auc", "average_precision", "binary_logloss"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(period=25)],
    )

    valid_score = model.predict_proba(x_valid)[:, 1]
    test_score = model.predict_proba(x_test)[:, 1]

    metrics = {
        "valid": {
            "model": binary_metrics(y_valid, valid_score),
            "tanimoto_baseline": binary_metrics(y_valid, valid_tanimoto),
            "by_label_type": by_label_type(y_valid, valid_score, valid_types),
            "ranking": ranking_metrics(y_valid, valid_score, valid_keys),
        },
        "test": {
            "model": binary_metrics(y_test, test_score),
            "tanimoto_baseline": binary_metrics(y_test, test_tanimoto),
            "by_label_type": by_label_type(y_test, test_score, test_types),
            "ranking": ranking_metrics(y_test, test_score, test_keys),
        },
    }

    files = {
        "model": str(out_dir / "bridge_verifier_v0_lgbm.joblib"),
        "report_json": str(out_dir / "bridge_verifier_v0_report.json"),
        "report_md": str(out_dir / "bridge_verifier_v0_report.md"),
        "feature_names": str(out_dir / "feature_names.json"),
        "test_scores": str(out_dir / "test_scores.parquet"),
        "valid_scores": str(out_dir / "valid_scores.parquet"),
    }
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "direction_values": DIRECTION_VALUES,
            "numeric_features": NUMERIC_FEATURES,
            "fp_bits": FP_BITS,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
        },
        files["model"],
    )
    Path(files["feature_names"]).write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    write_scores(Path(files["valid_scores"]), valid_rows, valid_score)
    write_scores(Path(files["test_scores"]), test_rows, test_score)

    report = {
        "schema_version": "bridge_verifier_v0.lgbm.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "pack_dir": str(pack_dir),
        "output_dir": str(out_dir),
        "data": {
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "train_labels": dict(Counter(y_train.tolist())),
            "valid_labels": dict(Counter(y_valid.tolist())),
            "test_labels": dict(Counter(y_test.tolist())),
            "feature_count": len(FEATURE_NAMES),
            "molecule_cache_size": len(cache.cache),
            "invalid_molecules": dict(cache.invalid),
        },
        "training": {
            "n_estimators_requested": args.n_estimators,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
            "train_limit": args.train_limit,
            "valid_limit": args.valid_limit,
            "test_limit": args.test_limit,
        },
        "metrics": metrics,
        "feature_importance_top": feature_importance(model, top_n=50),
        "files": files,
    }
    Path(files["report_json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(files["report_md"]).write_text(render_report(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "best_iteration": report["training"]["best_iteration"],
        "valid": report["metrics"]["valid"]["model"],
        "test": report["metrics"]["test"]["model"],
    }, indent=2, ensure_ascii=False))


def write_scores(path: Path, rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    out = []
    for row, score in zip(rows, scores):
        out.append(
            {
                "chemical_inchikey": row.get("chemical_inchikey") or "",
                "enzyme_inchikey": row.get("enzyme_inchikey") or "",
                "label": int(row.get("label") or 0),
                "label_type": row.get("label_type") or "",
                "bridge_direction": row.get("bridge_direction") or "",
                "tanimoto": float(row.get("tanimoto") or 0.0),
                "verifier_score": float(score),
            }
        )
    pq.write_table(pa.Table.from_pylist(out), path)


if __name__ == "__main__":
    main()
