#!/usr/bin/env python3
"""Train a lightweight learned verifier from a cascade perturbation pack."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from cascade_planner.baselines.route_plausibility import audit_step_plausibility
from cascade_planner.baselines.route_contract import RouteStepCandidate


RDLogger.DisableLog("rdApp.*")

REASON_LABELS = [
    "atom_balance_violation",
    "temperature_conflict",
    "ph_conflict",
    "solvent_conflict",
    "enzyme_toxicity",
    "cofactor_ledger_gap",
    "route_order_mismatch",
]


def main() -> None:
    args = _parse_args()
    result = train_from_pack(args.input, model_output=args.model_output, report_output=args.report_output)
    if args.markdown:
        _write_markdown(result, args.markdown)
    print(json.dumps(result["summary"], indent=2))


def train_from_pack(pack_path: Path, *, model_output: Path, report_output: Path) -> dict[str, Any]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    examples = [row for row in pack.get("examples") or [] if isinstance(row, dict)]
    if not examples:
        raise ValueError(f"no examples in {pack_path}")

    splits = _split_indices(examples)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(_features(examples[idx]) for idx in splits["train"])
    x_val = vectorizer.transform(_features(examples[idx]) for idx in splits["val"])
    x_test = vectorizer.transform(_features(examples[idx]) for idx in splits["test"])

    y_feasible_train = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["train"]])
    y_feasible_val = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["val"]])
    y_feasible_test = np.asarray([int(examples[idx].get("label") == 1) for idx in splits["test"]])
    feasible_model = _fit_binary(x_train, y_feasible_train)

    reason_models: dict[str, Any] = {}
    reason_metrics: dict[str, Any] = {}
    y_reason_test_rows: list[list[int]] = []
    y_reason_pred_rows: list[list[int]] = []
    for reason in REASON_LABELS:
        y_train = np.asarray([int(reason in (examples[idx].get("expected_failure_reasons") or [])) for idx in splits["train"]])
        y_test = np.asarray([int(reason in (examples[idx].get("expected_failure_reasons") or [])) for idx in splits["test"]])
        model = _fit_binary(x_train, y_train)
        pred = model.predict(x_test)
        reason_models[reason] = model
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            pred,
            average="binary",
            zero_division=0,
        )
        reason_metrics[reason] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(y_test.sum()),
        }
        y_reason_test_rows.append(y_test.tolist())
        y_reason_pred_rows.append(pred.tolist())

    feasible_val_pred = feasible_model.predict(x_val)
    feasible_test_pred = feasible_model.predict(x_test)
    y_reason_test = np.asarray(y_reason_test_rows, dtype=int).T if y_reason_test_rows else np.zeros((len(splits["test"]), 0))
    y_reason_pred = np.asarray(y_reason_pred_rows, dtype=int).T if y_reason_pred_rows else np.zeros((len(splits["test"]), 0))

    micro_f1 = f1_score(y_reason_test, y_reason_pred, average="micro", zero_division=0) if y_reason_test.size else 0.0
    macro_f1 = f1_score(y_reason_test, y_reason_pred, average="macro", zero_division=0) if y_reason_test.size else 0.0
    summary = {
        "schema_version": "learned_cascade_verifier_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(pack_path),
        "n_examples": len(examples),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "feature_dim": int(len(vectorizer.feature_names_)),
        "feasibility": {
            "val_accuracy": round(float(accuracy_score(y_feasible_val, feasible_val_pred)), 4) if len(y_feasible_val) else None,
            "test_accuracy": round(float(accuracy_score(y_feasible_test, feasible_test_pred)), 4) if len(y_feasible_test) else None,
            "test_report": classification_report(
                y_feasible_test,
                feasible_test_pred,
                target_names=["infeasible", "feasible"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "reasons": {
            "micro_f1": round(float(micro_f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_reason": reason_metrics,
        },
        "contract": "Learned verifier trained on rule-derived perturbation labels; not an expert feasibility model.",
    }
    artifact = {
        "vectorizer": vectorizer,
        "feasible_model": feasible_model,
        "reason_models": reason_models,
        "reason_labels": REASON_LABELS,
        "summary": summary,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)
    result = {"summary": summary}
    report_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _fit_binary(x_train: Any, y_train: np.ndarray) -> Any:
    if len(set(int(v) for v in y_train.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train[0]) if len(y_train) else 0)
        model.fit(x_train, y_train)
        return model
    model = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    model.fit(x_train, y_train)
    return model


def _split_indices(examples: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(examples):
        split = str(((row.get("cascade") or {}).get("metadata") or {}).get("split") or "").lower()
        if split in by_split:
            by_split[split].append(idx)
    if all(by_split.values()):
        return by_split

    groups = sorted({str(row.get("source_target_index")) for row in examples})
    train_groups = set(groups[: int(len(groups) * 0.7)])
    val_groups = set(groups[int(len(groups) * 0.7): int(len(groups) * 0.85)])
    out = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(examples):
        group = str(row.get("source_target_index"))
        if group in train_groups:
            out["train"].append(idx)
        elif group in val_groups:
            out["val"].append(idx)
        else:
            out["test"].append(idx)
    return out


def _features(example: dict[str, Any]) -> dict[str, float]:
    route = example.get("cascade") or {}
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    meta = route.get("metadata") or {}
    feats: dict[str, float] = {
        "bias": 1.0,
        "n_steps": float(len(steps)),
        "target_heavy": float(_heavy_atoms(example.get("target_smiles"))),
        f"route_domain={meta.get('route_domain') or 'unknown'}": 1.0,
        f"quality={meta.get('quality_tier') or 'unknown'}": 1.0,
        f"compatibility={meta.get('compatibility_label') or 'unknown'}": 1.0,
    }
    temps = [_safe_float(_condition_value(step, "temperature")) for step in steps]
    temps = [value for value in temps if value is not None]
    phs = [_safe_float(_condition_value(step, "ph")) for step in steps]
    phs = [value for value in phs if value is not None]
    feats["temperature_span"] = float(max(temps) - min(temps)) if len(temps) >= 2 else 0.0
    feats["ph_span"] = float(max(phs) - min(phs)) if len(phs) >= 2 else 0.0
    feats["enzyme_steps"] = float(sum(1 for step in steps if _is_enzymatic(step)))
    feats["toxic_enzyme_condition_hits"] = float(sum(1 for step in steps if _is_enzymatic(step) and _enzyme_toxic(step)))
    feats["cofactor_gap_count"] = float(len(_cofactor_gaps(steps)))
    feats["route_order_mismatch_count"] = float(_route_order_mismatch_count(steps, example.get("target_smiles") or ""))
    feats["solvent_conflict_pairs"] = float(_solvent_conflict_pairs(steps, route.get("stage_partition") or []))

    material_counts = []
    for step in steps:
        audit = audit_step_plausibility(
            RouteStepCandidate(
                product_smiles=str(step.get("product") or ""),
                reactant_smiles=_step_reactants(step),
                rxn_smiles=str(step.get("reaction_smiles") or ""),
                condition_predictions=[row for row in step.get("condition_predictions") or [] if isinstance(row, dict)],
            )
        )
        material_counts.append(float(len(audit.get("reasons") or [])))
        for key in ("heavy_atom_gain", "carbon_gain", "hetero_atom_gain"):
            feats[f"max_{key}"] = max(float(feats.get(f"max_{key}", 0.0)), float(audit.get(key) or 0.0))
    feats["material_issue_steps"] = float(sum(1 for value in material_counts if value > 0))

    for step in steps:
        reaction_type = str(step.get("reaction_type") or "unknown").lower()
        feats[f"rxn_type={reaction_type}"] = feats.get(f"rxn_type={reaction_type}", 0.0) + 1.0
        ec = str(step.get("ec") or "")
        if ec:
            feats[f"ec1={ec.split('.', 1)[0]}"] = feats.get(f"ec1={ec.split('.', 1)[0]}", 0.0) + 1.0
        solvent_class = _solvent_class(str(_condition_value(step, "solvent") or ""))
        if solvent_class:
            feats[f"solvent_class={solvent_class}"] = feats.get(f"solvent_class={solvent_class}", 0.0) + 1.0
    return feats


def _step_reactants(step: dict[str, Any]) -> list[str]:
    out = []
    if step.get("main_reactant"):
        out.append(str(step.get("main_reactant")))
    out.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    if not out and isinstance(step.get("reactants"), list):
        out.extend(str(smi) for smi in step.get("reactants") if smi)
    return out


def _condition_value(step: dict[str, Any], field: str) -> Any:
    keys = {
        "temperature": ("T", "Temperature", "temperature", "temperature_c"),
        "ph": ("pH", "ph", "PH"),
        "solvent": ("solvent", "Solvent"),
        "catalyst": ("catalyst", "Catalyst", "reagent", "Reagent"),
    }[field]
    for key in keys:
        if key in step and step[key] not in (None, ""):
            return step[key]
    for row in step.get("condition_predictions") or []:
        if not isinstance(row, dict):
            continue
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return row[key]
    return None


def _is_enzymatic(step: dict[str, Any]) -> bool:
    text = " ".join(str(step.get(key) or "") for key in ("source", "reaction_type", "model_name")).lower()
    return bool(step.get("ec") or step.get("enzyme_ec_annotations") or "enzyme" in text)


def _enzyme_toxic(step: dict[str, Any]) -> bool:
    solvent = str(_condition_value(step, "solvent") or "").lower()
    catalyst = str(_condition_value(step, "catalyst") or "").lower()
    return any(token in solvent for token in ("dichloromethane", "dcm", "chloroform", "dmf", "pyridine", "acetonitrile")) or any(
        token in catalyst for token in ("lda", "dibal", "pocl3", "socl2", "nah")
    )


def _cofactor_gaps(steps: list[dict[str, Any]]) -> dict[str, float]:
    required: Counter[str] = Counter()
    regenerated: Counter[str] = Counter()
    for step in steps:
        for key, amount in (step.get("cofactor_requirements") or {}).items():
            required[str(key)] += float(amount or 0.0)
        for key, amount in (step.get("cofactor_regenerations") or {}).items():
            regenerated[str(key)] += float(amount or 0.0)
    return {key: value - regenerated.get(key, 0.0) for key, value in required.items() if value > regenerated.get(key, 0.0)}


def _route_order_mismatch_count(steps: list[dict[str, Any]], target: str) -> int:
    needed = {_canonical(target) or _canonical(str(steps[0].get("product") or ""))} if steps else set()
    needed.discard("")
    count = 0
    for step in steps:
        product = _canonical(str(step.get("product") or ""))
        if product and needed and product not in needed:
            count += 1
        needed.discard(product)
        for reactant in _step_reactants(step):
            can = _canonical(reactant)
            if can:
                needed.add(can)
    return count


def _solvent_conflict_pairs(steps: list[dict[str, Any]], partition: list[str]) -> int:
    count = 0
    if not partition or len(partition) != len(steps):
        partition = [f"stage_{idx + 1}" for idx in range(len(steps))]
    for idx, (left, right) in enumerate(zip(steps, steps[1:])):
        if partition[idx] != partition[idx + 1]:
            continue
        classes = {_solvent_class(str(_condition_value(left, "solvent") or "")), _solvent_class(str(_condition_value(right, "solvent") or ""))}
        if classes == {"aqueous", "hydrophobic"}:
            count += 1
    return count


def _solvent_class(solvent: str) -> str:
    text = solvent.lower()
    if any(token in text for token in ("water", "h2o", "buffer", "pbs", "phosphate", "tris", "hepes")):
        return "aqueous"
    if any(token in text for token in ("dichloromethane", "dcm", "chloroform", "toluene", "hexane", "heptane")):
        return "hydrophobic"
    return "organic" if text else ""


def _canonical(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Learned Cascade Verifier Report",
        "",
        f"- Examples: `{summary['n_examples']}`",
        f"- Feature dim: `{summary['feature_dim']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Feasibility test accuracy: `{summary['feasibility']['test_accuracy']}`",
        f"- Reason micro F1: `{summary['reasons']['micro_f1']}`",
        f"- Reason macro F1: `{summary['reasons']['macro_f1']}`",
        "",
        "## Per Reason",
        "",
        "| Reason | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for reason, row in summary["reasons"]["per_reason"].items():
        lines.append(f"| `{reason}` | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")
    lines.extend(["", "## Contract", "", summary["contract"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned cascade verifier from perturbation pack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
