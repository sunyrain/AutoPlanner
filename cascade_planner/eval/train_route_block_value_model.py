"""Train a pairwise route/block value model from route_block_value_pack_v1."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


SCHEMA_VERSION = "route_block_value_model.v1"


def train_route_block_value_model(
    *,
    pack_jsonl: Path,
    output_dir: Path,
    positive_task: str,
    negative_task: str | None = None,
    include_groups: list[str] | None = None,
    exclude_groups: list[str] | None = None,
    c_values: list[float] | None = None,
    max_pos_per_group: int = 8,
    max_neg_per_pos: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    c_values = c_values or [0.03, 0.1, 0.3, 1.0, 3.0]
    rows = _read_jsonl(pack_jsonl)
    rows = _with_training_labels(rows, positive_task=positive_task, negative_task=negative_task)
    splits = {name: [row for row in rows if row.get("split") == name] for name in ("train", "val", "test")}
    missing = [name for name, split_rows in splits.items() if not split_rows]
    if missing:
        raise ValueError(f"pack must contain train/val/test rows after label filtering; missing {missing}")
    feature_names = _feature_names(
        splits["train"],
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    if not feature_names:
        raise ValueError("selected feature groups produced no features")
    datasets = {name: _dataset(split_rows, feature_names) for name, split_rows in splits.items()}
    evidence_audit = _evidence_provenance_audit(rows, feature_names)
    rng = np.random.default_rng(int(seed))
    pair_x, pair_y = _pairwise_matrix(
        datasets["train"],
        max_pos_per_group=max_pos_per_group,
        max_neg_per_pos=max_neg_per_pos,
        rng=rng,
    )
    mean, std = _scaler(datasets["train"]["x"])
    best_model = None
    best_payload: dict[str, Any] | None = None
    for c_value in c_values:
        model = LogisticRegression(C=float(c_value), penalty="l2", solver="liblinear", max_iter=500, random_state=int(seed))
        model.fit(pair_x / std, pair_y)
        val_scores = _model_scores(model, datasets["val"], mean=mean, std=std)
        val_report = _evaluate_rankings(datasets["val"], val_scores)
        selection_score = _selection_score(val_report)
        if best_payload is None or selection_score > best_payload["val_selection_score"]:
            best_model = model
            best_payload = {
                "c_value": float(c_value),
                "pair_rows": int(pair_x.shape[0]),
                "val_selection_score": float(selection_score),
                "val": val_report,
                "test": _evaluate_rankings(datasets["test"], _model_scores(model, datasets["test"], mean=mean, std=std)),
            }
    if best_model is None or best_payload is None:
        raise ValueError("failed to fit route/block value model")
    learned_scores = {
        name: _model_scores(best_model, dataset, mean=mean, std=std)
        for name, dataset in datasets.items()
    }
    baselines = _baseline_reports(datasets)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "pack_jsonl": str(pack_jsonl),
            "output_dir": str(output_dir),
            "positive_task": positive_task,
            "negative_task": negative_task,
            "include_groups": include_groups or [],
            "exclude_groups": exclude_groups or [],
            "c_values": [float(value) for value in c_values],
            "max_pos_per_group": int(max_pos_per_group),
            "max_neg_per_pos": int(max_neg_per_pos),
            "seed": int(seed),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            name: _counts(dataset)
            for name, dataset in datasets.items()
        },
        "feature_names": feature_names,
        "evidence_provenance_audit": evidence_audit,
        "baselines": baselines,
        "model": {
            **best_payload,
            "train": _evaluate_rankings(datasets["train"], learned_scores["train"]),
            "coef": {name: round(float(value), 6) for name, value in zip(feature_names, best_model.coef_[0])},
        },
        "selection": _select_best(baselines, best_payload),
        "interpretation": _interpretation(baselines, best_payload),
    }
    (output_dir / "route_block_value_model_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "route_block_value_model_report.md").write_text(_markdown(result), encoding="utf-8")
    with (output_dir / "route_block_value_model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model": best_model,
                "mean": mean,
                "std": std,
                "feature_names": feature_names,
                "metadata": result["metadata"],
                "selection": result["selection"],
            },
            fh,
        )
    return result


def _with_training_labels(rows: list[dict[str, Any]], *, positive_task: str, negative_task: str | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tasks = row.get("weak_label_tasks") if isinstance(row.get("weak_label_tasks"), dict) else {}
        if bool(tasks.get(positive_task)):
            label = 1
        elif negative_task is None or bool(tasks.get(negative_task)):
            label = 0
        else:
            continue
        copied = dict(row)
        copied["_training_label"] = label
        out.append(copied)
    if not any(row["_training_label"] > 0 for row in out):
        raise ValueError(f"positive task has no positive rows: {positive_task}")
    if not any(row["_training_label"] == 0 for row in out):
        raise ValueError("label filtering produced no negative rows")
    return out


def _feature_names(rows: list[dict[str, Any]], *, include_groups: list[str] | None, exclude_groups: list[str] | None) -> list[str]:
    include = {str(group) for group in include_groups or []}
    exclude = {str(group) for group in exclude_groups or []}
    names = set()
    for row in rows:
        groups = row.get("feature_groups") if isinstance(row.get("feature_groups"), dict) else {}
        for group, values in groups.items():
            group_name = str(group)
            if include and group_name not in include:
                continue
            if group_name in exclude:
                continue
            if not isinstance(values, dict):
                continue
            for key in values:
                names.add(f"{group_name}.{key}")
    return sorted(names)


def _dataset(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    return {
        "rows": rows,
        "x": np.asarray([[_feature_value(row, name) for name in feature_names] for row in rows], dtype=np.float32),
        "y": np.asarray([int(row.get("_training_label") or 0) for row in rows], dtype=np.int32),
        "group_ids": [str(row.get("selector_group_id") or row.get("target_id") or "") for row in rows],
    }


def _feature_value(row: dict[str, Any], name: str) -> float:
    group, key = name.split(".", 1)
    groups = row.get("feature_groups") if isinstance(row.get("feature_groups"), dict) else {}
    values = groups.get(group) if isinstance(groups.get(group), dict) else {}
    return _float(values.get(key))


def _pairwise_matrix(
    dataset: dict[str, Any],
    *,
    max_pos_per_group: int,
    max_neg_per_pos: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    diffs = []
    labels = []
    by_group = _group_indices(dataset)
    x = dataset["x"]
    y = dataset["y"]
    rows = dataset["rows"]
    for indices in by_group.values():
        positives = sorted(
            [idx for idx in indices if y[idx] > 0],
            key=lambda idx: int(rows[idx].get("native_rank") or 10**9),
        )[:max_pos_per_group]
        negatives = sorted(
            [idx for idx in indices if y[idx] == 0],
            key=lambda idx: int(rows[idx].get("native_rank") or 10**9),
        )[:max_neg_per_pos]
        for pos_idx in positives:
            for neg_idx in negatives:
                diff = x[pos_idx] - x[neg_idx]
                diffs.append(diff)
                labels.append(1)
                diffs.append(-diff)
                labels.append(0)
    if not diffs:
        raise ValueError("no pairwise rows; need groups with both positive and negative rows")
    order = rng.permutation(len(diffs))
    return np.asarray(diffs, dtype=np.float32)[order], np.asarray(labels, dtype=np.int32)[order]


def _baseline_reports(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    specs = {
        "native_rank": {name: _native_scores(dataset) for name, dataset in datasets.items()},
        "audit_guard": {name: _audit_scores(dataset) for name, dataset in datasets.items()},
        "retrieval_only": {name: _retrieval_scores(dataset) for name, dataset in datasets.items()},
        "learned_ccts_only": {name: _learned_ccts_scores(dataset) for name, dataset in datasets.items()},
    }
    return {
        name: {
            split: _evaluate_rankings(datasets[split], scores)
            for split, scores in split_scores.items()
        }
        for name, split_scores in specs.items()
    }


def _native_scores(dataset: dict[str, Any]) -> np.ndarray:
    return np.asarray([-float(row.get("native_rank") or 0) for row in dataset["rows"]], dtype=np.float32)


def _audit_scores(dataset: dict[str, Any]) -> np.ndarray:
    out = []
    for row in dataset["rows"]:
        audit = row.get("product_audit") if isinstance(row.get("product_audit"), dict) else {}
        out.append(-_float(audit.get("risk_order")) + float(bool((row.get("weak_label_tasks") or {}).get("material_sane_proxy"))))
    return np.asarray(out, dtype=np.float32)


def _retrieval_scores(dataset: dict[str, Any]) -> np.ndarray:
    out = []
    for row in dataset["rows"]:
        group = ((row.get("feature_groups") or {}).get("cascade_retrieval") or {})
        out.append(
            _first_available(
                group,
                [
                    "ccts_v3_runtime_best_route_evidence",
                    "ccts_v3_runtime_step_pair_max",
                    "ccts_v3_runtime_step_any_max",
                    "ccts_v3_runtime_step_pair_mean",
                    "ccts_v3_runtime_step_any_mean",
                    "cascade_block_hits",
                    "v4_evidence_hits",
                ],
            )
        )
    return np.asarray(out, dtype=np.float32)


def _learned_ccts_scores(dataset: dict[str, Any]) -> np.ndarray:
    out = []
    for row in dataset["rows"]:
        group = ((row.get("feature_groups") or {}).get("learned_ccts") or {})
        values = [_float(value) for value in group.values()]
        out.append(max(values) if values else 0.0)
    return np.asarray(out, dtype=np.float32)


def _model_scores(model: LogisticRegression, dataset: dict[str, Any], *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return model.decision_function((dataset["x"] - mean) / std)


def _evaluate_rankings(dataset: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    by_group = _group_indices(dataset)
    y = dataset["y"]
    ranks = []
    recalls = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0}
    positive_groups = 0
    for indices in by_group.values():
        positives = {idx for idx in indices if y[idx] > 0}
        if not positives:
            continue
        positive_groups += 1
        ordered = sorted(indices, key=lambda idx: (-float(scores[idx]), int(dataset["rows"][idx].get("native_rank") or 10**9)))
        first_rank = min((rank for rank, idx in enumerate(ordered, start=1) if idx in positives), default=None)
        if first_rank is None:
            continue
        ranks.append(first_rank)
        for k in recalls:
            if first_rank <= k:
                recalls[k] += 1
    total_groups = len(by_group)
    return {
        "groups": int(total_groups),
        "positive_groups": int(positive_groups),
        "mrr_covered": round(float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0, 6),
        "recall_at_k_all": {
            str(k): round(recalls[k] / total_groups, 6) if total_groups else 0.0
            for k in recalls
        },
        "recall_at_k_covered": {
            str(k): round(recalls[k] / positive_groups, 6) if positive_groups else 0.0
            for k in recalls
        },
    }


def _select_best(baselines: dict[str, Any], model_payload: dict[str, Any]) -> dict[str, Any]:
    candidates = {
        name: (report.get("test") or {}).get("mrr_covered", 0.0)
        for name, report in baselines.items()
    }
    candidates["pairwise_logistic"] = (model_payload.get("test") or {}).get("mrr_covered", 0.0)
    best = max(candidates, key=lambda key: float(candidates[key]))
    return {
        "selected_method": best,
        "selected_test_mrr_covered": float(candidates[best]),
        "model_test_mrr_covered": float(candidates["pairwise_logistic"]),
        "retrieval_only_test_mrr_covered": float(candidates.get("retrieval_only") or 0.0),
        "native_rank_test_mrr_covered": float(candidates.get("native_rank") or 0.0),
    }


def _interpretation(baselines: dict[str, Any], model_payload: dict[str, Any]) -> dict[str, Any]:
    model_mrr = _float((model_payload.get("test") or {}).get("mrr_covered"))
    retrieval_mrr = _float(((baselines.get("retrieval_only") or {}).get("test") or {}).get("mrr_covered"))
    native_mrr = _float(((baselines.get("native_rank") or {}).get("test") or {}).get("mrr_covered"))
    return {
        "model_minus_native_mrr": round(model_mrr - native_mrr, 6),
        "model_minus_retrieval_only_mrr": round(model_mrr - retrieval_mrr, 6),
        "clears_retrieval_only": model_mrr > retrieval_mrr,
    }


def _selection_score(report: dict[str, Any]) -> float:
    return float(report.get("mrr_covered") or 0.0) + float((report.get("recall_at_k_all") or {}).get("3") or 0.0)


def _counts(dataset: dict[str, Any]) -> dict[str, int]:
    return {
        "rows": int(len(dataset["rows"])),
        "groups": int(len(set(dataset["group_ids"]))),
        "positive_rows": int(np.sum(dataset["y"] > 0)),
        "negative_rows": int(np.sum(dataset["y"] == 0)),
        "positive_groups": int(sum(1 for indices in _group_indices(dataset).values() if any(dataset["y"][idx] > 0 for idx in indices))),
    }


def _evidence_provenance_audit(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    retrieval_rows = []
    missing_rows = []
    missing_train_only_rows = []
    for row in rows:
        if _row_has_retrieval_features(row):
            retrieval_rows.append(row)
            provenance = row.get("evidence_provenance") if isinstance(row.get("evidence_provenance"), dict) else {}
            if provenance.get("status") != "present":
                missing_rows.append(row)
            elif _row_requires_train_only(row) and not provenance.get("has_train_only_marker"):
                missing_train_only_rows.append(row)
    model_uses_retrieval = any(name.startswith("cascade_retrieval.") for name in feature_names)
    if missing_rows and model_uses_retrieval:
        status = "model_uses_unverified_retrieval_provenance"
    elif missing_train_only_rows and model_uses_retrieval:
        status = "model_uses_retrieval_without_train_only_marker"
    elif missing_rows:
        status = "retrieval_baseline_unverifiable"
    elif missing_train_only_rows:
        status = "retrieval_baseline_missing_train_only_marker"
    else:
        status = "verified_or_no_retrieval_features"
    warnings = []
    if missing_rows:
        warnings.append(
            "retrieval/evidence features are present without source-corpus provenance; "
            "do not treat train-only retrieval as verified"
        )
    if missing_train_only_rows:
        warnings.append(
            "retrieval/evidence source provenance is present but lacks an explicit train-only marker"
        )
    return {
        "status": status,
        "retrieval_feature_rows": len(retrieval_rows),
        "missing_retrieval_provenance_rows": len(missing_rows),
        "missing_train_only_marker_rows": len(missing_train_only_rows),
        "model_uses_cascade_retrieval_features": model_uses_retrieval,
        "warnings": warnings,
    }


def _row_requires_train_only(row: dict[str, Any]) -> bool:
    contract = row.get("training_contract") if isinstance(row.get("training_contract"), dict) else {}
    text = str(contract.get("evidence_contract") or "").lower()
    return "train_only" in text or "train-only" in text


def _row_has_retrieval_features(row: dict[str, Any]) -> bool:
    groups = row.get("feature_groups") if isinstance(row.get("feature_groups"), dict) else {}
    retrieval = groups.get("cascade_retrieval") if isinstance(groups.get("cascade_retrieval"), dict) else {}
    return any(_float(value) > 0.0 for value in retrieval.values())


def _group_indices(dataset: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(dataset["group_ids"]):
        groups[str(group_id)].append(idx)
    return groups


def _scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _markdown(result: dict[str, Any]) -> str:
    baselines = result.get("baselines") or {}
    model = result.get("model") or {}
    lines = [
        "# Route/Block Value Model",
        "",
        "## Setup",
        "",
        f"- positive task: `{result['metadata']['positive_task']}`",
        f"- negative task: `{result['metadata']['negative_task']}`",
        f"- exclude groups: `{', '.join(result['metadata']['exclude_groups'])}`",
        "",
        "## Counts",
        "",
        "| split | rows | groups | positives | negatives | positive groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, counts in (result.get("counts") or {}).items():
        lines.append(
            f"| `{split}` | {counts['rows']} | {counts['groups']} | {counts['positive_rows']} | "
            f"{counts['negative_rows']} | {counts['positive_groups']} |"
        )
    lines.extend(["", "## Test Metrics", "", "| method | MRR | R@1 all | R@3 all | R@5 all |", "|---|---:|---:|---:|---:|"])
    for name in ["native_rank", "audit_guard", "retrieval_only", "learned_ccts_only"]:
        report = ((baselines.get(name) or {}).get("test") or {})
        lines.append(_metric_row(name, report))
    lines.append(_metric_row("pairwise_logistic", model.get("test") or {}))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- model - native MRR: `{result['interpretation']['model_minus_native_mrr']}`",
            f"- model - retrieval-only MRR: `{result['interpretation']['model_minus_retrieval_only_mrr']}`",
            f"- clears retrieval-only: `{result['interpretation']['clears_retrieval_only']}`",
            f"- evidence provenance: `{(result.get('evidence_provenance_audit') or {}).get('status')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(name: str, report: dict[str, Any]) -> str:
    recall = report.get("recall_at_k_all") or {}
    return (
        f"| `{name}` | {_fmt(report.get('mrr_covered'))} | {_fmt(recall.get('1'))} | "
        f"{_fmt(recall.get('3'))} | {_fmt(recall.get('5'))} |"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_available(values: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        if key in values:
            return _float(values.get(key))
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


def _parse_csv(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend([item.strip() for item in str(value).split(",") if item.strip()])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a route/block value model from route_block_value_pack_v1")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--positive-task", required=True)
    ap.add_argument("--negative-task")
    ap.add_argument("--include-group", action="append", default=[])
    ap.add_argument("--exclude-group", action="append", default=[])
    ap.add_argument("--c-value", action="append", type=float, default=[])
    ap.add_argument("--max-pos-per-group", type=int, default=8)
    ap.add_argument("--max-neg-per-pos", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report = train_route_block_value_model(
        pack_jsonl=Path(args.pack),
        output_dir=Path(args.output_dir),
        positive_task=args.positive_task,
        negative_task=args.negative_task,
        include_groups=_parse_csv(args.include_group),
        exclude_groups=_parse_csv(args.exclude_group),
        c_values=args.c_value or None,
        max_pos_per_group=args.max_pos_per_group,
        max_neg_per_pos=args.max_neg_per_pos,
        seed=args.seed,
    )
    print(json.dumps({"selection": report["selection"], "interpretation": report["interpretation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
