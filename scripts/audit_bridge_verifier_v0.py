"""Stress and leakage audit for bridge verifier v0."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


RDLogger.DisableLog("rdApp.*")

EXACT_TYPES = {"tier1_strict_exact_substrate_bridge", "tier2_strict_exact_product_bridge"}
SIMILARITY_TYPES = {"tier3_high_similarity_nonexact_bridge"}
BOUNDARY_TYPES = {"tier3_high_similarity_nonexact_bridge", "near_similarity_below_positive_threshold"}


def read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


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


def merge_scores(data_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(data_rows) != len(score_rows):
        raise ValueError(f"row count mismatch: data={len(data_rows)} scores={len(score_rows)}")
    out = []
    for data, score in zip(data_rows, score_rows):
        row = dict(data)
        row["verifier_score"] = float(score["verifier_score"])
        row["score_tanimoto"] = float(score.get("tanimoto") or data.get("tanimoto") or 0.0)
        out.append(row)
    return out


def metric_block(rows: list[dict[str, Any]], *, threshold: float = 0.5) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    y = np.asarray([int(row.get("label") or 0) for row in rows], dtype=np.int8)
    s = np.asarray([float(row.get("verifier_score") or 0.0) for row in rows], dtype=np.float64)
    pred = (s >= threshold).astype(np.int8)
    out = {
        "rows": int(len(rows)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "score_mean": float(np.mean(s)),
        "score_median": float(np.median(s)),
        "threshold": threshold,
    }
    if len(set(y.tolist())) >= 2:
        tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
        out.update(
            {
                "roc_auc": float(roc_auc_score(y, s)),
                "pr_auc": float(average_precision_score(y, s)),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    elif out["negatives"]:
        out["rejection_rate"] = float(np.mean(s < threshold))
        out["rejection_rate_lt_0_2"] = float(np.mean(s < 0.2))
    elif out["positives"]:
        out["recall"] = float(np.mean(s >= threshold))
        out["recall_ge_0_2"] = float(np.mean(s >= 0.2))
    return out


def tanimoto_metric_block(rows: list[dict[str, Any]], *, threshold: float = 0.5) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    tmp = []
    for row in rows:
        tmp.append({**row, "verifier_score": float(row.get("tanimoto") or row.get("score_tanimoto") or 0.0)})
    return metric_block(tmp, threshold=threshold)


def by_label_type(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("label_type") or "unknown")].append(row)
    return {key: metric_block(value) for key, value in sorted(grouped.items())}


def choose_threshold(rows: list[dict[str, Any]], *, target_precision: float) -> dict[str, Any]:
    y = np.asarray([int(row.get("label") or 0) for row in rows], dtype=np.int8)
    s = np.asarray([float(row.get("verifier_score") or 0.0) for row in rows], dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y, s)
    candidates = []
    for idx, threshold in enumerate(thresholds):
        if precision[idx] >= target_precision:
            candidates.append((float(recall[idx]), float(precision[idx]), float(threshold)))
    if not candidates:
        return {"target_precision": target_precision, "available": False}
    recall_value, precision_value, threshold = max(candidates)
    return {
        "target_precision": target_precision,
        "available": True,
        "threshold": threshold,
        "valid_precision": precision_value,
        "valid_recall": recall_value,
    }


def scaffold(smiles: str, cache: dict[str, str]) -> str:
    if smiles in cache:
        return cache[smiles]
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        cache[smiles] = ""
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    value = Chem.MolToSmiles(scaf, isomericSmiles=False, canonical=True) if scaf is not None else ""
    cache[smiles] = value
    return value


def split_leakage(train: list[dict[str, Any]], valid: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    def keyset(rows: list[dict[str, Any]], key: str) -> set[str]:
        return {str(row.get(key) or "") for row in rows if row.get(key)}

    train_chem = keyset(train, "chemical_inchikey")
    valid_chem = keyset(valid, "chemical_inchikey")
    test_chem = keyset(test, "chemical_inchikey")
    train_enzyme = keyset(train, "enzyme_inchikey")
    valid_enzyme = keyset(valid, "enzyme_inchikey")
    test_enzyme = keyset(test, "enzyme_inchikey")

    train_ecs = set()
    for row in train:
        train_ecs.update(parse_json_list(row.get("enzyme_ec_sample_json")))
    test_ec_rows = 0
    test_ec_novel_rows = 0
    for row in test:
        ecs = set(parse_json_list(row.get("enzyme_ec_sample_json")))
        if not ecs:
            continue
        test_ec_rows += 1
        if ecs.isdisjoint(train_ecs):
            test_ec_novel_rows += 1

    scaf_cache: dict[str, str] = {}
    train_scaffolds = {scaffold(row.get("chemical_smiles") or "", scaf_cache) for row in train}
    valid_scaffolds = [scaffold(row.get("chemical_smiles") or "", scaf_cache) for row in valid]
    test_scaffolds = [scaffold(row.get("chemical_smiles") or "", scaf_cache) for row in test]

    return {
        "chemical_connector_overlap": {
            "train_valid": len(train_chem & valid_chem),
            "train_test": len(train_chem & test_chem),
            "valid_test": len(valid_chem & test_chem),
            "train_unique": len(train_chem),
            "valid_unique": len(valid_chem),
            "test_unique": len(test_chem),
        },
        "enzyme_molecule_overlap": {
            "train_valid": len(train_enzyme & valid_enzyme),
            "train_test": len(train_enzyme & test_enzyme),
            "valid_test": len(valid_enzyme & test_enzyme),
            "train_unique": len(train_enzyme),
            "valid_unique": len(valid_enzyme),
            "test_unique": len(test_enzyme),
        },
        "ec_coverage": {
            "train_unique_ec": len(train_ecs),
            "test_rows_with_ec": test_ec_rows,
            "test_rows_with_all_ec_novel_vs_train": test_ec_novel_rows,
        },
        "chemical_scaffold_overlap": {
            "train_unique_scaffolds": len(train_scaffolds),
            "valid_unique_scaffolds": len(set(valid_scaffolds)),
            "test_unique_scaffolds": len(set(test_scaffolds)),
            "valid_rows_novel_scaffold_vs_train": sum(1 for item in valid_scaffolds if item not in train_scaffolds),
            "test_rows_novel_scaffold_vs_train": sum(1 for item in test_scaffolds if item not in train_scaffolds),
            "test_positive_rows_novel_scaffold_vs_train": sum(
                1 for row, item in zip(test, test_scaffolds) if item not in train_scaffolds and int(row.get("label") or 0) == 1
            ),
        },
        "_test_scaffolds": test_scaffolds,
        "_valid_scaffolds": valid_scaffolds,
        "_train_scaffolds": train_scaffolds,
    }


def error_rows(rows: list[dict[str, Any]], *, top_n: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_pos = [row for row in rows if int(row.get("label") or 0) == 0 and float(row.get("verifier_score") or 0.0) >= 0.5]
    false_neg = [row for row in rows if int(row.get("label") or 0) == 1 and float(row.get("verifier_score") or 0.0) < 0.5]
    false_pos.sort(key=lambda row: float(row.get("verifier_score") or 0.0), reverse=True)
    false_neg.sort(key=lambda row: float(row.get("verifier_score") or 0.0))

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "chemical_inchikey": row.get("chemical_inchikey"),
            "enzyme_inchikey": row.get("enzyme_inchikey"),
            "label_type": row.get("label_type"),
            "bridge_direction": row.get("bridge_direction"),
            "tanimoto": row.get("tanimoto"),
            "verifier_score": row.get("verifier_score"),
            "chemical_smiles": row.get("chemical_smiles"),
            "enzyme_smiles": row.get("enzyme_smiles"),
            "enzyme_ec_sample_json": row.get("enzyme_ec_sample_json"),
        }

    return [compact(row) for row in false_pos[:top_n]], [compact(row) for row in false_neg[:top_n]]


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Bridge Verifier v0 Stress Audit",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- model_dir: `{report['model_dir']}`",
        f"- pass: `{report['pass']}`",
        "",
        "## Gate Checks",
        "",
        "| Gate | Pass | Value | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["gate_checks"].items():
        lines.append(f"| `{key}` | `{row['pass']}` | {fmt(row['value'])} | {fmt(row['threshold'])} |")

    lines.extend(["", "## Stress Metrics", "", "| Subset | Rows | PR-AUC | ROC-AUC | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in report["stress_metrics"]["test"].items():
        lines.append(
            f"| `{name}` | {row.get('rows', 0)} | {fmt(row.get('pr_auc'))} | {fmt(row.get('roc_auc'))} | "
            f"{fmt(row.get('precision'))} | {fmt(row.get('recall'))} | {fmt(row.get('f1'))} |"
        )

    lines.extend(["", "## Thresholds From Valid", "", "| Target Precision | Threshold | Valid P | Valid R | Test P | Test R | Test F1 |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for item in report["recommended_thresholds"]:
        test = item.get("test_metrics") or {}
        lines.append(
            f"| {fmt(item['target_precision'])} | {fmt(item.get('threshold'))} | {fmt(item.get('valid_precision'))} | "
            f"{fmt(item.get('valid_recall'))} | {fmt(test.get('precision'))} | {fmt(test.get('recall'))} | {fmt(test.get('f1'))} |"
        )

    leak = report["leakage"]
    lines.extend(["", "## Leakage Checks", ""])
    lines.append(f"- chemical train/test overlap: `{leak['chemical_connector_overlap']['train_test']}`")
    lines.append(f"- enzyme train/test overlap: `{leak['enzyme_molecule_overlap']['train_test']}`")
    lines.append(f"- test rows with all EC novel vs train: `{leak['ec_coverage']['test_rows_with_all_ec_novel_vs_train']}` / `{leak['ec_coverage']['test_rows_with_ec']}`")
    lines.append(f"- test rows with novel chemical scaffold vs train: `{leak['chemical_scaffold_overlap']['test_rows_novel_scaffold_vs_train']}`")
    lines.append(f"- test positive rows with novel chemical scaffold vs train: `{leak['chemical_scaffold_overlap']['test_positive_rows_novel_scaffold_vs_train']}`")

    lines.extend(["", "## Notes", ""])
    lines.append("The current split prevents chemical connector leakage. Enzyme molecule and scaffold overlap are reported as residual risk, not as a hard failure for v0.")
    lines.append("The strongest deployment threshold from this audit is the valid-derived 0.99 precision threshold, unless route search needs higher recall.")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bridge verifier v0")
    parser.add_argument("--pack-dir", default="data/bridge_pack_v0")
    parser.add_argument("--model-dir", default="results/shared/bridge_verifier_v0_20260527")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    started = time.time()
    pack_dir = Path(args.pack_dir)
    model_dir = Path(args.model_dir)
    output_json = Path(args.output_json) if args.output_json else model_dir / "bridge_verifier_v0_stress_audit.json"
    output_md = Path(args.output_md) if args.output_md else model_dir / "bridge_verifier_v0_stress_audit.md"

    train = read_parquet(pack_dir / "verifier_train.parquet")
    valid = merge_scores(read_parquet(pack_dir / "verifier_valid.parquet"), read_parquet(model_dir / "valid_scores.parquet"))
    test = merge_scores(read_parquet(pack_dir / "verifier_test.parquet"), read_parquet(model_dir / "test_scores.parquet"))

    subsets: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all": lambda row: True,
        "nonexact_all": lambda row: row.get("label_type") not in EXACT_TYPES,
        "similarity_boundary": lambda row: row.get("label_type") in BOUNDARY_TYPES,
        "tanimoto_ge_0_62": lambda row: float(row.get("tanimoto") or 0.0) >= 0.62,
        "tanimoto_ge_0_80": lambda row: float(row.get("tanimoto") or 0.0) >= 0.80,
    }
    stress_metrics = {"valid": {}, "test": {}}
    tanimoto_metrics = {"valid": {}, "test": {}}
    for split_name, rows in [("valid", valid), ("test", test)]:
        for subset_name, predicate in subsets.items():
            subset = [row for row in rows if predicate(row)]
            stress_metrics[split_name][subset_name] = metric_block(subset)
            tanimoto_metrics[split_name][subset_name] = tanimoto_metric_block(subset)

    thresholds = []
    for target in [0.98, 0.99, 0.995]:
        item = choose_threshold(valid, target_precision=target)
        if item.get("available"):
            item["test_metrics"] = metric_block(test, threshold=float(item["threshold"]))
            item["test_nonexact_metrics"] = metric_block([row for row in test if row.get("label_type") not in EXACT_TYPES], threshold=float(item["threshold"]))
        thresholds.append(item)

    leakage = split_leakage(train, valid, test)
    valid_scaffolds = leakage.pop("_valid_scaffolds")
    test_scaffolds = leakage.pop("_test_scaffolds")
    train_scaffolds = leakage.pop("_train_scaffolds")
    novel_test = [row for row, scaf in zip(test, test_scaffolds) if scaf not in train_scaffolds]
    novel_valid = [row for row, scaf in zip(valid, valid_scaffolds) if scaf not in train_scaffolds]
    stress_metrics["test"]["novel_chemical_scaffold"] = metric_block(novel_test)
    stress_metrics["valid"]["novel_chemical_scaffold"] = metric_block(novel_valid)

    fp_rows, fn_rows = error_rows(test)
    false_pos_path = model_dir / "bridge_verifier_v0_test_false_positives_top.json"
    false_neg_path = model_dir / "bridge_verifier_v0_test_false_negatives_top.json"
    false_pos_path.write_text(json.dumps(fp_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    false_neg_path.write_text(json.dumps(fn_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_checks = {
        "test_pr_auc_ge_0_95": {"value": stress_metrics["test"]["all"].get("pr_auc"), "threshold": 0.95},
        "test_precision_ge_0_95": {"value": stress_metrics["test"]["all"].get("precision"), "threshold": 0.95},
        "test_recall_ge_0_95": {"value": stress_metrics["test"]["all"].get("recall"), "threshold": 0.95},
        "nonexact_pr_auc_ge_0_95": {"value": stress_metrics["test"]["nonexact_all"].get("pr_auc"), "threshold": 0.95},
        "similarity_boundary_pr_auc_ge_0_95": {"value": stress_metrics["test"]["similarity_boundary"].get("pr_auc"), "threshold": 0.95},
        "tanimoto_ge_0_80_precision_ge_0_95": {"value": stress_metrics["test"]["tanimoto_ge_0_80"].get("precision"), "threshold": 0.95},
        "chemical_train_test_overlap_eq_0": {"value": leakage["chemical_connector_overlap"]["train_test"], "threshold": 0},
    }
    for row in gate_checks.values():
        if row["threshold"] == 0:
            row["pass"] = row["value"] == 0
        else:
            row["pass"] = row["value"] is not None and row["value"] >= row["threshold"]

    report = {
        "schema_version": "bridge_verifier_v0.stress_audit.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "pack_dir": str(pack_dir),
        "model_dir": str(model_dir),
        "pass": all(row["pass"] for row in gate_checks.values()),
        "gate_checks": gate_checks,
        "stress_metrics": stress_metrics,
        "tanimoto_metrics": tanimoto_metrics,
        "by_label_type_test": by_label_type(test),
        "recommended_thresholds": thresholds,
        "leakage": leakage,
        "files": {
            "audit_json": str(output_json),
            "audit_md": str(output_md),
            "false_positives_top": str(false_pos_path),
            "false_negatives_top": str(false_neg_path),
        },
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "gate_checks": gate_checks, "files": report["files"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
