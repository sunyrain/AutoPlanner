"""Train a same-target route-pool ranker from route-level labels."""
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


SCHEMA_VERSION = "route_pool_pairwise_ranker.v1"


def train_route_pool_ranker(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    output_dir: Path,
    feature_set: str = "all",
    c_values: list[float] | None = None,
    max_pos_per_group: int = 8,
    max_neg_per_pos: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    c_values = c_values or [0.03, 0.1, 0.3, 1.0, 3.0]
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    test_rows = _read_jsonl(test_jsonl)
    feature_names = _select_feature_names(sorted((train_rows[0].get("feature") or {}).keys()) if train_rows else [], feature_set)
    if not feature_names:
        raise ValueError(f"feature set {feature_set!r} selected no features")
    train = _dataset(train_rows, feature_names)
    val = _dataset(val_rows, feature_names)
    test = _dataset(test_rows, feature_names)
    rng = np.random.default_rng(int(seed))
    pair_x, pair_y = _pairwise_matrix(
        train,
        max_pos_per_group=max_pos_per_group,
        max_neg_per_pos=max_neg_per_pos,
        rng=rng,
    )
    mean, std = _scaler(train["x"])
    best_model = None
    best_payload: dict[str, Any] | None = None
    for c_value in c_values:
        model = LogisticRegression(C=float(c_value), penalty="l2", solver="liblinear", max_iter=500, random_state=int(seed))
        model.fit(pair_x / std, pair_y)
        val_scores = _model_scores(model, val, mean=mean, std=std)
        val_report = _evaluate_rankings(val, val_scores)
        score = _selection_score(val_report)
        if best_payload is None or score > best_payload["val_selection_score"]:
            best_model = model
            best_payload = {
                "c_value": float(c_value),
                "pair_rows": int(pair_x.shape[0]),
                "val_selection_score": float(score),
                "val": val_report,
                "test": _evaluate_rankings(test, _model_scores(model, test, mean=mean, std=std)),
            }
    if best_model is None or best_payload is None:
        raise ValueError("failed to fit route-pool ranker")
    learned_scores = {
        "train": _model_scores(best_model, train, mean=mean, std=std),
        "val": _model_scores(best_model, val, mean=mean, std=std),
        "test": _model_scores(best_model, test, mean=mean, std=std),
    }
    baselines = _baseline_reports(train, val, test)
    blends = {}
    blends.update(_blend_reports(val, test, base_name="native_rank", aux_name="learned", base_val=_native_scores(val), base_test=_native_scores(test), aux_val=learned_scores["val"], aux_test=learned_scores["test"]))
    blends.update(_blend_reports(val, test, base_name="audit_guard", aux_name="learned", base_val=_audit_guard_scores(val), base_test=_audit_guard_scores(test), aux_val=learned_scores["val"], aux_test=learned_scores["test"]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "output_dir": str(output_dir),
            "feature_set": str(feature_set),
            "c_values": [float(v) for v in c_values],
            "max_pos_per_group": int(max_pos_per_group),
            "max_neg_per_pos": int(max_neg_per_pos),
            "seed": int(seed),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_positive_rows": int(np.sum(train["y"] > 0)),
            "val_positive_rows": int(np.sum(val["y"] > 0)),
            "test_positive_rows": int(np.sum(test["y"] > 0)),
            "train_positive_groups": _positive_group_count(train),
            "val_positive_groups": _positive_group_count(val),
            "test_positive_groups": _positive_group_count(test),
        },
        "baselines": baselines,
        "model": {
            **best_payload,
            "train": _evaluate_rankings(train, learned_scores["train"]),
            "coef": {name: round(float(value), 6) for name, value in zip(feature_names, best_model.coef_[0])},
        },
        "blends": blends,
        "selection": _select_best(baselines, best_payload, blends),
        "feature_names": feature_names,
    }
    (output_dir / "route_pool_pairwise_ranker_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "route_pool_pairwise_ranker_report.md").write_text(_markdown(result), encoding="utf-8")
    with (output_dir / "route_pool_pairwise_ranker.pkl").open("wb") as fh:
        pickle.dump({"schema_version": SCHEMA_VERSION, "model": best_model, "mean": mean, "std": std, "feature_names": feature_names, "selection": result["selection"]}, fh)
    return result


def _select_feature_names(feature_names: list[str], feature_set: str) -> list[str]:
    feature_set = str(feature_set or "all")
    ccts = {name for name in feature_names if name.startswith("ccts_")}
    block = {name for name in feature_names if name.startswith("block_") or name in {"n_blocks", "cascade_block_hits"}}
    v4 = {name for name in feature_names if name.startswith("v4_step_") or name == "v4_evidence_hits"}
    audit = {
        name
        for name in feature_names
        if name.startswith("audit_")
        or name
        in {
            "large_atom_gain_count",
            "route_plausibility_passed",
            "generic_template_fraction",
        }
    }
    native = {
        name
        for name in feature_names
        if name
        in {
            "native_score",
            "native_rank",
            "native_inv_rank",
            "native_rank_fraction",
            "stock_closed",
            "n_steps",
        }
    }
    groups = {"ccts": ccts, "block": block, "v4": v4, "native": native}
    if feature_set == "all":
        selected = set(feature_names)
    elif feature_set == "native_only":
        selected = native
    elif feature_set == "no_audit":
        selected = set(feature_names) - audit
    elif feature_set == "no_audit_no_cascade":
        selected = set(feature_names) - audit - ccts - block - v4
    elif feature_set == "no_cascade":
        selected = set(feature_names) - ccts - block - v4
    elif feature_set == "no_ccts":
        selected = set(feature_names) - ccts
    elif feature_set == "no_block":
        selected = set(feature_names) - block
    elif feature_set == "no_v4":
        selected = set(feature_names) - v4
    elif feature_set == "no_ccts_no_v4":
        selected = set(feature_names) - ccts - v4
    elif feature_set == "no_block_no_v4":
        selected = set(feature_names) - block - v4
    elif feature_set == "ccts_only":
        selected = ccts
    elif feature_set == "block_only":
        selected = block
    elif feature_set == "v4_only":
        selected = v4
    elif feature_set == "ccts_v4_only":
        selected = ccts | v4
    elif feature_set == "cascade_only":
        selected = ccts | block | v4
    else:
        valid = ", ".join(
            [
                "all",
                "native_only",
                "no_audit",
                "no_audit_no_cascade",
                "no_cascade",
                "no_ccts",
                "no_block",
                "no_v4",
                "no_ccts_no_v4",
                "no_block_no_v4",
                "ccts_only",
                "block_only",
                "v4_only",
                "ccts_v4_only",
                "cascade_only",
            ]
        )
        raise ValueError(f"unknown feature set {feature_set!r}; valid: {valid}")
    return [name for name in feature_names if name in selected]


def _dataset(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    return {
        "rows": rows,
        "x": np.asarray([[_float((row.get("feature") or {}).get(name)) for name in feature_names] for row in rows], dtype=np.float32),
        "y": np.asarray([int(row.get("route_label") or 0) for row in rows], dtype=np.int32),
        "group_ids": [str(row.get("selector_group_id") or row.get("target_id") or "") for row in rows],
    }


def _pairwise_matrix(dataset: dict[str, Any], *, max_pos_per_group: int, max_neg_per_pos: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    diffs = []
    labels = []
    by_group = _group_indices(dataset)
    x = dataset["x"]
    y = dataset["y"]
    rows = dataset["rows"]
    for indices in by_group.values():
        positives = sorted([idx for idx in indices if y[idx] > 0], key=lambda idx: (-int(y[idx]), int(rows[idx].get("native_rank") or 10**9)))[:max_pos_per_group]
        negatives = _hard_negatives(rows, indices, y, max_neg_per_pos=max_neg_per_pos)
        for pos_idx in positives:
            for neg_idx in negatives:
                diff = x[pos_idx] - x[neg_idx]
                diffs.append(diff)
                labels.append(1)
                diffs.append(-diff)
                labels.append(0)
    if not diffs:
        raise ValueError("no pairwise rows; route-pool pack has no positive/negative target groups")
    order = rng.permutation(len(diffs))
    return np.asarray(diffs, dtype=np.float32)[order], np.asarray(labels, dtype=np.int32)[order]


def _hard_negatives(rows: list[dict[str, Any]], indices: list[int], y: np.ndarray, *, max_neg_per_pos: int) -> list[int]:
    negatives = [idx for idx in indices if y[idx] == 0]
    by_native = sorted(negatives, key=lambda idx: int(rows[idx].get("native_rank") or 10**9))[:max_neg_per_pos]
    by_evidence = sorted(
        negatives,
        key=lambda idx: (
            -_float((rows[idx].get("feature") or {}).get("ccts_best_route_evidence")),
            -_float((rows[idx].get("feature") or {}).get("block_rerank_score")),
            int(rows[idx].get("native_rank") or 10**9),
        ),
    )[:max_neg_per_pos]
    out = []
    for idx in by_native + by_evidence:
        if idx not in out:
            out.append(idx)
        if len(out) >= max_neg_per_pos:
            break
    return out


def _baseline_reports(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    specs = {
        "native_rank": (_native_scores(train), _native_scores(val), _native_scores(test)),
        "audit_guard": (_audit_guard_scores(train), _audit_guard_scores(val), _audit_guard_scores(test)),
        "ccts_best_route_evidence": (_feature_scores(train, "ccts_best_route_evidence"), _feature_scores(val, "ccts_best_route_evidence"), _feature_scores(test, "ccts_best_route_evidence")),
        "ccts_model_mean": (_feature_scores(train, "ccts_model_mean"), _feature_scores(val, "ccts_model_mean"), _feature_scores(test, "ccts_model_mean")),
        "block_rerank_score": (_feature_scores(train, "block_rerank_score"), _feature_scores(val, "block_rerank_score"), _feature_scores(test, "block_rerank_score")),
    }
    return {name: {"train": _evaluate_rankings(train, a), "val": _evaluate_rankings(val, b), "test": _evaluate_rankings(test, c)} for name, (a, b, c) in specs.items()}


def _blend_reports(
    val: dict[str, Any],
    test: dict[str, Any],
    *,
    base_name: str,
    aux_name: str,
    base_val: np.ndarray,
    base_test: np.ndarray,
    aux_val: np.ndarray,
    aux_test: np.ndarray,
) -> dict[str, Any]:
    best = None
    for alpha in [-2.0, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        val_scores = _standardize(base_val) + float(alpha) * _standardize(aux_val)
        report = _evaluate_rankings(val, val_scores)
        score = _selection_score(report)
        if best is None or score > best["val_selection_score"] or (
            abs(score - best["val_selection_score"]) < 1e-12 and abs(float(alpha)) < abs(float(best["alpha"]))
        ):
            best = {"alpha": float(alpha), "val_selection_score": float(score), "val": report}
    assert best is not None
    test_scores = _standardize(base_test) + best["alpha"] * _standardize(aux_test)
    return {
        f"{base_name}_plus_{aux_name}": {
            **best,
            "test": _evaluate_rankings(test, test_scores),
        }
    }


def _evaluate_rankings(dataset: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    by_group = _group_indices(dataset)
    y = dataset["y"]
    out = {"groups": len(by_group), "positive_groups": 0, "mrr_covered": 0.0, "recall_at_k_all": {}, "recall_at_k_covered": {}}
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0}
    rr = []
    for indices in by_group.values():
        order = sorted(indices, key=lambda idx: (-float(scores[idx]), int((dataset["rows"][idx]).get("native_rank") or 10**9), str((dataset["rows"][idx]).get("route_id") or "")))
        ranks = [rank for rank, idx in enumerate(order, start=1) if y[idx] > 0]
        if ranks:
            out["positive_groups"] += 1
            first = min(ranks)
            rr.append(1.0 / first)
            for k in hits_at:
                if first <= k:
                    hits_at[k] += 1
    out["mrr_covered"] = round(float(np.mean(rr)) if rr else 0.0, 6)
    for k, value in hits_at.items():
        out["recall_at_k_all"][str(k)] = round(value / max(1, len(by_group)), 6)
        out["recall_at_k_covered"][str(k)] = round(value / max(1, out["positive_groups"]), 6)
    return out


def _selection_score(report: dict[str, Any]) -> float:
    return float(report.get("mrr_covered") or 0.0) + float((report.get("recall_at_k_all") or {}).get("3") or 0.0)


def _select_best(baselines: dict[str, Any], model: dict[str, Any], blends: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for family, mapping in (("baseline", baselines), ("blend", blends)):
        for name, report in mapping.items():
            candidates.append((family, name, _selection_score(report["val"]), _selection_tie_priority(family, name), report))
    candidates.append(("model", "pairwise_logistic", _selection_score(model["val"]), _selection_tie_priority("model", "pairwise_logistic"), model))
    candidates.sort(key=lambda row: (-row[2], row[3]))
    family, name, score, _, report = candidates[0]
    return {
        "selected_family": family,
        "selected_method": name,
        "selected_val_score": round(float(score), 6),
        "selected_test_mrr_covered": (report.get("test") or {}).get("mrr_covered"),
        "selected_test_recall_at3_all": ((report.get("test") or {}).get("recall_at_k_all") or {}).get("3"),
        "native_test_mrr_covered": (((baselines.get("native_rank") or {}).get("test") or {}).get("mrr_covered")),
        "native_test_recall_at3_all": ((((baselines.get("native_rank") or {}).get("test") or {}).get("recall_at_k_all") or {}).get("3")),
    }


def _selection_tie_priority(family: str, name: str) -> int:
    if name == "audit_guard":
        return 0
    if name == "native_rank":
        return 1
    if name == "audit_guard_plus_learned":
        return 2
    if family == "model":
        return 3
    if name == "native_rank_plus_learned":
        return 4
    return 5


def _model_scores(model: LogisticRegression, dataset: dict[str, Any], *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return model.decision_function((dataset["x"] - mean) / std).astype(np.float32)


def _native_scores(dataset: dict[str, Any]) -> np.ndarray:
    return np.asarray([-int(row.get("native_rank") or 0) for row in dataset["rows"]], dtype=np.float32)


def _audit_guard_scores(dataset: dict[str, Any]) -> np.ndarray:
    scores = []
    for row in dataset["rows"]:
        feature = row.get("feature") or {}
        class_order = _float(feature.get("audit_class_order"))
        risk_order = _float(feature.get("audit_risk_order"))
        native_rank = int(row.get("native_rank") or 0)
        # Lexicographic guard encoded as a scalar: audit class first, risk second,
        # native rank only as a deterministic tie-breaker.
        scores.append(-(class_order * 1000.0 + risk_order * 10.0 + native_rank * 0.001))
    return np.asarray(scores, dtype=np.float32)


def _feature_scores(dataset: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([_float((row.get("feature") or {}).get(key)) for row in dataset["rows"]], dtype=np.float32)


def _group_indices(dataset: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(dataset["group_ids"]):
        out[str(group_id)].append(idx)
    return out


def _positive_group_count(dataset: dict[str, Any]) -> int:
    by_group = _group_indices(dataset)
    y = dataset["y"]
    return sum(1 for indices in by_group.values() if any(y[idx] > 0 for idx in indices))


def _scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route-Pool Pairwise Ranker",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Selection",
        "",
        "```json",
        json.dumps(result.get("selection") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| method | mrr_covered | recall@1 all | recall@3 all | recall@5 all |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = []
    for name, payload in (result.get("baselines") or {}).items():
        rows.append((name, payload.get("test") or {}))
    rows.append(("pairwise_logistic", ((result.get("model") or {}).get("test") or {})))
    for name, payload in (result.get("blends") or {}).items():
        rows.append((name, payload.get("test") or {}))
    for name, metric in rows:
        r = metric.get("recall_at_k_all") or {}
        lines.append(f"| `{name}` | {metric.get('mrr_covered')} | {r.get('1')} | {r.get('3')} | {r.get('5')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train same-target route-pool pairwise ranker")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--c-value", action="append", type=float, default=[])
    ap.add_argument("--max-pos-per-group", type=int, default=8)
    ap.add_argument("--max-neg-per-pos", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_route_pool_ranker(
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        output_dir=Path(args.output_dir),
        feature_set=args.feature_set,
        c_values=args.c_value or None,
        max_pos_per_group=args.max_pos_per_group,
        max_neg_per_pos=args.max_neg_per_pos,
        seed=args.seed,
    )
    print(json.dumps({"counts": result["counts"], "selection": result["selection"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
