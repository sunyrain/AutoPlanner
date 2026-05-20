"""Train a group-wise LambdaRank model for same-target route-pool ranking."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from cascade_planner.eval.train_route_pool_ranker import (
    _baseline_reports,
    _dataset,
    _evaluate_rankings,
    _feature_scores,
    _native_scores,
    _positive_group_count,
    _read_jsonl,
    _selection_score,
    _standardize,
)


SCHEMA_VERSION = "route_pool_lambdarank.v1"


DEFAULT_PARAM_GRID: tuple[dict[str, Any], ...] = (
    {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 1, "reg_lambda": 1.0},
    {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 3, "reg_lambda": 1.0},
    {"n_estimators": 180, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 5, "reg_lambda": 2.0},
    {"n_estimators": 120, "learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 3, "reg_lambda": 0.5},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 5, "reg_lambda": 2.0},
)


def train_route_pool_lambdarank(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    output_dir: Path,
    param_grid: list[dict[str, Any]] | None = None,
    seed: int = 42,
    eval_at: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    test_rows = _read_jsonl(test_jsonl)
    feature_names = sorted((train_rows[0].get("feature") or {}).keys()) if train_rows else []
    train = _dataset(train_rows, feature_names)
    val = _dataset(val_rows, feature_names)
    test = _dataset(test_rows, feature_names)

    train_sorted = _group_sorted_dataset(train)
    val_sorted = _group_sorted_dataset(val)
    grid = param_grid or list(DEFAULT_PARAM_GRID)
    trials = []
    best: dict[str, Any] | None = None
    best_model: lgb.LGBMRanker | None = None
    for trial_idx, params in enumerate(grid):
        model = _fit_lambdarank(
            train_sorted,
            val_sorted,
            params=params,
            seed=seed + trial_idx,
            eval_at=eval_at,
        )
        val_scores = _predict(model, val)
        test_scores = _predict(model, test)
        val_report = _evaluate_rankings(val, val_scores)
        test_report = _evaluate_rankings(test, test_scores)
        trial = {
            "trial_index": trial_idx,
            "params": _clean_params(params),
            "best_iteration": _best_iteration(model),
            "val_selection_score": round(float(_selection_score(val_report)), 6),
            "val": val_report,
            "test": test_report,
        }
        trials.append(trial)
        if best is None or float(trial["val_selection_score"]) > float(best["val_selection_score"]):
            best = trial
            best_model = model
    if best is None or best_model is None:
        raise ValueError("no LambdaRank model was fit")

    learned_scores = {
        "train": _predict(best_model, train),
        "val": _predict(best_model, val),
        "test": _predict(best_model, test),
    }
    baselines = _baseline_reports(train, val, test)
    blends = _blend_reports(
        val=val,
        test=test,
        learned_val=learned_scores["val"],
        learned_test=learned_scores["test"],
    )
    model_report = {
        **best,
        "train": _evaluate_rankings(train, learned_scores["train"]),
        "feature_importance_gain": _feature_importance(best_model, feature_names, importance_type="gain"),
        "feature_importance_split": _feature_importance(best_model, feature_names, importance_type="split"),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "output_dir": str(output_dir),
            "seed": int(seed),
            "eval_at": list(eval_at),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "group-wise LambdaRank; groups are target_id route pools; test split is never used for model or alpha selection",
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
        "model": model_report,
        "trials": trials,
        "blends": blends,
        "selection": _select_best(baselines=baselines, model=model_report, blends=blends),
        "feature_names": feature_names,
    }
    (output_dir / "route_pool_lambdarank_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "route_pool_lambdarank_report.md").write_text(_markdown(result), encoding="utf-8")
    with (output_dir / "route_pool_lambdarank.pkl").open("wb") as fh:
        pickle.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model": best_model,
                "feature_names": feature_names,
                "selection": result["selection"],
                "metadata": result["metadata"],
            },
            fh,
        )
    return result


def _fit_lambdarank(
    train: dict[str, Any],
    val: dict[str, Any],
    *,
    params: dict[str, Any],
    seed: int,
    eval_at: tuple[int, ...],
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 6],
        random_state=int(seed),
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    callbacks = [lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)]
    model.fit(
        train["x"],
        train["y"],
        group=train["group"],
        eval_set=[(val["x"], val["y"])],
        eval_group=[val["group"]],
        eval_at=list(eval_at),
        callbacks=callbacks,
    )
    return model


def _group_sorted_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    order = sorted(
        range(len(dataset["rows"])),
        key=lambda idx: (
            str(dataset["group_ids"][idx]),
            int(dataset["rows"][idx].get("native_rank") or 10**9),
            str(dataset["rows"][idx].get("route_id") or ""),
        ),
    )
    x = dataset["x"][order]
    y = dataset["y"][order]
    group_ids = [str(dataset["group_ids"][idx]) for idx in order]
    group = []
    current = None
    size = 0
    for group_id in group_ids:
        if current is None:
            current = group_id
            size = 1
        elif group_id == current:
            size += 1
        else:
            group.append(size)
            current = group_id
            size = 1
    if current is not None:
        group.append(size)
    return {"x": x, "y": y, "group": group, "group_ids": group_ids}


def _predict(model: lgb.LGBMRanker, dataset: dict[str, Any]) -> np.ndarray:
    best_iteration = _best_iteration(model)
    kwargs: dict[str, Any] = {}
    if best_iteration > 0:
        kwargs["num_iteration"] = best_iteration
    return np.asarray(model.predict(dataset["x"], **kwargs), dtype=np.float32)


def _best_iteration(model: lgb.LGBMRanker) -> int:
    value = getattr(model, "best_iteration_", None)
    try:
        return int(value or 0)
    except Exception:
        return 0


def _feature_importance(model: lgb.LGBMRanker, feature_names: list[str], *, importance_type: str) -> dict[str, float]:
    booster = model.booster_
    values = booster.feature_importance(importance_type=importance_type)
    ranked = sorted(zip(feature_names, values), key=lambda row: float(row[1]), reverse=True)
    return {name: round(float(value), 6) for name, value in ranked[:30]}


def _blend_reports(
    *,
    val: dict[str, Any],
    test: dict[str, Any],
    learned_val: np.ndarray,
    learned_test: np.ndarray,
) -> dict[str, Any]:
    bases = {
        "native_rank": (_native_scores(val), _native_scores(test)),
        "ccts_model_mean": (_feature_scores(val, "ccts_model_mean"), _feature_scores(test, "ccts_model_mean")),
        "ccts_best_route_evidence": (
            _feature_scores(val, "ccts_best_route_evidence"),
            _feature_scores(test, "ccts_best_route_evidence"),
        ),
        "block_rerank_score": (_feature_scores(val, "block_rerank_score"), _feature_scores(test, "block_rerank_score")),
    }
    out: dict[str, Any] = {}
    for base_name, (base_val, base_test) in bases.items():
        best = None
        for alpha in [-2.0, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
            val_scores = _standardize(base_val) + float(alpha) * _standardize(learned_val)
            report = _evaluate_rankings(val, val_scores)
            score = _selection_score(report)
            if best is None or score > best["val_selection_score"]:
                best = {"alpha": float(alpha), "val_selection_score": round(float(score), 6), "val": report}
        assert best is not None
        test_scores = _standardize(base_test) + float(best["alpha"]) * _standardize(learned_test)
        out[f"{base_name}_plus_lambdarank"] = {
            **best,
            "test": _evaluate_rankings(test, test_scores),
        }
    return out


def _select_best(*, baselines: dict[str, Any], model: dict[str, Any], blends: dict[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[str, str, float, dict[str, Any]]] = []
    for family, mapping in (("baseline", baselines), ("blend", blends)):
        for name, report in mapping.items():
            candidates.append((family, name, _selection_score(report["val"]), report))
    candidates.append(("model", "lambdarank", _selection_score(model["val"]), model))
    candidates.sort(key=lambda row: row[2], reverse=True)
    family, name, score, report = candidates[0]
    native = ((baselines.get("native_rank") or {}).get("test") or {})
    selected_test = report.get("test") or {}
    return {
        "selected_family": family,
        "selected_method": name,
        "selected_val_score": round(float(score), 6),
        "selected_test_mrr_covered": selected_test.get("mrr_covered"),
        "selected_test_recall_at1_all": ((selected_test.get("recall_at_k_all") or {}).get("1")),
        "selected_test_recall_at3_all": ((selected_test.get("recall_at_k_all") or {}).get("3")),
        "selected_test_recall_at5_all": ((selected_test.get("recall_at_k_all") or {}).get("5")),
        "native_test_mrr_covered": native.get("mrr_covered"),
        "native_test_recall_at3_all": ((native.get("recall_at_k_all") or {}).get("3")),
    }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route-Pool LambdaRank",
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
        "| method | mrr_covered | recall@1 all | recall@3 all | recall@5 all | recall@10 all | recall@50 all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for name, payload in (result.get("baselines") or {}).items():
        rows.append((name, payload.get("test") or {}))
    rows.append(("lambdarank", ((result.get("model") or {}).get("test") or {})))
    for name, payload in (result.get("blends") or {}).items():
        rows.append((name, payload.get("test") or {}))
    for name, metric in rows:
        r = metric.get("recall_at_k_all") or {}
        lines.append(
            f"| `{name}` | {metric.get('mrr_covered')} | {r.get('1')} | {r.get('3')} | {r.get('5')} | {r.get('10')} | {r.get('50')} |"
        )
    lines.extend(
        [
            "",
            "## Best Params",
            "",
            "```json",
            json.dumps(((result.get("model") or {}).get("params") or {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Top Gain Features",
            "",
            "```json",
            json.dumps(((result.get("model") or {}).get("feature_importance_gain") or {}), indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_param_grid(values: list[str]) -> list[dict[str, Any]] | None:
    if not values:
        return None
    out = []
    for value in values:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("--param-json must be a JSON object")
        out.append(payload)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train same-target route-pool LambdaRank model")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--param-json", action="append", default=[], help="LightGBM param JSON object; may be repeated")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = train_route_pool_lambdarank(
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        output_dir=Path(args.output_dir),
        param_grid=_parse_param_grid(list(args.param_json or [])),
        seed=args.seed,
    )
    print(json.dumps({"counts": result["counts"], "selection": result["selection"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
