"""Train lightweight route and candidate value-function weights.

The labels are weak supervision from existing planner artifacts:
route-level labels come from exported route metrics, and candidate-level labels
mark the selected step candidate as positive against alternatives in the pool.
This gives the search controllers a learned, artifact-calibrated value model
without adding a heavy training dependency.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rdkit import RDLogger

from cascade_planner.cascadeboard.value_function import (
    DEFAULT_BOARD_WEIGHTS,
    DEFAULT_CANDIDATE_WEIGHTS,
    candidate_value_features,
    canonical_or_raw,
    metric_value_features,
)


RDLogger.DisableLog("rdApp.warning")

BOARD_FEATURES = [k for k in DEFAULT_BOARD_WEIGHTS if k != "bias"]
CANDIDATE_FEATURES = [k for k in DEFAULT_CANDIDATE_WEIGHTS if k != "bias"]
BOARD_NON_NEGATIVE = {
    "filled_route",
    "progressive_route",
    "route_solved",
    "strict_stock_solve",
    "main_chain_reduction",
    "leaf_reduction",
    "naturalness",
    "condition_success",
    "compatibility_success",
    "enzyme_evidence",
}
BOARD_NON_POSITIVE = {"issue_count"}
CANDIDATE_NON_NEGATIVE = {"candidate_score", "stock_fraction", "main_reduction", "has_ec", "has_evidence"}
CANDIDATE_NON_POSITIVE = {"large_aux_penalty", "self_loop"}


@dataclass
class TrainingRow:
    features: dict[str, float]
    label: float
    source: str
    weight: float = 1.0


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def collect_training_rows(paths: Iterable[Path]) -> tuple[list[TrainingRow], list[TrainingRow], dict[str, Any]]:
    board_rows: list[TrainingRow] = []
    candidate_rows: list[TrainingRow] = []
    skipped = 0
    loaded = 0

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        loaded += 1
        for route in find_route_records(data):
            metrics = route.get("metrics") or {}
            label = route_label(route)
            if metrics and label is not None:
                board_rows.append(TrainingRow(
                    features=metric_value_features(metrics),
                    label=float(label),
                    source=str(path),
                ))
            candidate_rows.extend(candidate_rows_from_route(route, path))

    metadata = {
        "files_loaded": loaded,
        "files_skipped": skipped,
        "board_samples": len(board_rows),
        "candidate_samples": len(candidate_rows),
    }
    return board_rows, candidate_rows, metadata


def collect_training_pack_rows(pack_dir: Path) -> tuple[list[TrainingRow], list[TrainingRow], dict[str, Any]]:
    board_path = pack_dir / "route_value.jsonl"
    candidate_path = pack_dir / "candidate_ranking.jsonl"
    if not board_path.exists() or not candidate_path.exists():
        raise FileNotFoundError(f"training pack missing route_value.jsonl or candidate_ranking.jsonl: {pack_dir}")

    board_rows = []
    for row in read_jsonl(board_path):
        features = row.get("features") or {}
        if features:
            board_rows.append(TrainingRow(
                features={k: float(v or 0.0) for k, v in features.items()},
                label=float(row.get("label") or 0.0),
                source=str(board_path),
                weight=float(row.get("weight") or 1.0),
            ))

    candidate_rows = []
    for row in read_jsonl(candidate_path):
        features = row.get("features") or {}
        if features:
            candidate_rows.append(TrainingRow(
                features={k: float(v or 0.0) for k, v in features.items()},
                label=float(row.get("label") or 0.0),
                source=str(candidate_path),
                weight=float(row.get("weight") or 1.0),
            ))

    return board_rows, candidate_rows, {
        "training_pack": str(pack_dir),
        "board_samples": len(board_rows),
        "candidate_samples": len(candidate_rows),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def find_route_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("steps"), list) and isinstance(value.get("metrics"), dict):
            yield value
            return
        for key in ("routes", "targets", "planner_output", "result", "results", "outputs"):
            child = value.get(key)
            if child is not None:
                yield from find_route_records(child)
    elif isinstance(value, list):
        for item in value:
            yield from find_route_records(item)


def route_label(route: dict[str, Any]) -> float | None:
    metrics = route.get("metrics") or {}
    if metrics.get("route_solved") is True:
        return 1.0
    if metrics.get("route_solved") is False:
        return 0.0
    if metrics.get("progressive_route") is True and metrics.get("strict_stock_solve") is True:
        return 1.0
    if metrics.get("filled_route") is False:
        return 0.0
    return None


def candidate_rows_from_route(route: dict[str, Any], path: Path) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for step in route.get("steps") or []:
        product = step.get("product")
        selected = canonical_or_raw(step.get("main_reactant"))
        pool = (step.get("candidate_pool") or {}).get("top_candidates") or []
        if not product or not selected or not pool:
            continue
        stock_status = step.get("stock_status") or {}

        def stock_checker(smiles: str) -> bool:
            return bool(stock_status.get(smiles))

        for cand in pool:
            main = canonical_or_raw(cand.get("main_reactant"))
            label = 1.0 if main and main == selected else 0.0
            candidate = dict(cand)
            if "rxn_smiles" not in candidate and candidate.get("reaction_smiles"):
                candidate["rxn_smiles"] = candidate["reaction_smiles"]
            if "type" not in candidate and candidate.get("reaction_type"):
                candidate["type"] = candidate["reaction_type"]
            rows.append(TrainingRow(
                features=candidate_value_features(product, candidate, stock_checker=stock_checker),
                label=label,
                source=str(path),
            ))
    return rows


def train_from_rows(
    rows: list[TrainingRow],
    feature_names: list[str],
    prior: dict[str, float],
    *,
    epochs: int = 120,
    lr: float = 0.08,
    l2: float = 0.02,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not rows:
        return dict(prior), {"samples": 0, "positive": 0, "negative": 0, "loss": None, "accuracy": None}

    positives = sum(1 for row in rows if row.label >= 0.5)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        return dict(prior), {
            "samples": len(rows),
            "positive": positives,
            "negative": negatives,
            "loss": None,
            "accuracy": None,
            "note": "single-class labels; kept prior weights",
        }

    weights = dict(prior)
    row_weights = [
        (0.5 / positives if row.label >= 0.5 else 0.5 / negatives) * max(float(row.weight), 0.0)
        for row in rows
    ]
    norm = sum(row_weights) or 1.0

    for _ in range(max(1, epochs)):
        grad = {name: 0.0 for name in ["bias", *feature_names]}
        for row, row_weight in zip(rows, row_weights):
            score = weights.get("bias", 0.0)
            for name in feature_names:
                score += weights.get(name, 0.0) * float(row.features.get(name) or 0.0)
            pred = sigmoid(score)
            err = (pred - row.label) * row_weight
            grad["bias"] += err
            for name in feature_names:
                grad[name] += err * float(row.features.get(name) or 0.0)
        for name in feature_names:
            grad[name] += l2 * (weights.get(name, 0.0) - prior.get(name, 0.0))
        for name in ["bias", *feature_names]:
            weights[name] = weights.get(name, 0.0) - lr * grad[name] / norm

    metrics = evaluate(rows, feature_names, weights)
    metrics.update({
        "samples": len(rows),
        "positive": positives,
        "negative": negatives,
    })
    return {k: round(float(v), 6) for k, v in weights.items()}, metrics


def apply_sign_constraints(
    weights: dict[str, float],
    *,
    non_negative: set[str],
    non_positive: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Keep weakly learned weights chemically monotone for search-time use."""
    constrained = dict(weights)
    changed: dict[str, float] = {}
    for name in non_negative:
        if constrained.get(name, 0.0) < 0.0:
            changed[name] = constrained[name]
            constrained[name] = 0.0
    for name in non_positive:
        if constrained.get(name, 0.0) > 0.0:
            changed[name] = constrained[name]
            constrained[name] = 0.0
    return {k: round(float(v), 6) for k, v in constrained.items()}, changed


def evaluate(rows: list[TrainingRow], feature_names: list[str], weights: dict[str, float]) -> dict[str, Any]:
    loss = 0.0
    correct = 0
    for row in rows:
        pred = predict(row.features, feature_names, weights)
        y = row.label
        loss += -(y * math.log(max(pred, 1e-9)) + (1.0 - y) * math.log(max(1.0 - pred, 1e-9)))
        correct += int((pred >= 0.5) == (y >= 0.5))
    return {
        "loss": round(loss / max(len(rows), 1), 6),
        "accuracy": round(correct / max(len(rows), 1), 6),
    }


def predict(features: dict[str, float], feature_names: list[str], weights: dict[str, float]) -> float:
    score = weights.get("bias", 0.0)
    for name in feature_names:
        score += weights.get(name, 0.0) * float(features.get(name) or 0.0)
    return sigmoid(score)


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def train_value_models(
    paths: Iterable[Path],
    *,
    epochs: int = 120,
    lr: float = 0.08,
    l2: float = 0.02,
) -> dict[str, Any]:
    board_rows, candidate_rows, metadata = collect_training_rows(paths)
    board_weights, board_metrics = train_from_rows(
        board_rows,
        BOARD_FEATURES,
        DEFAULT_BOARD_WEIGHTS,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )
    candidate_weights, candidate_metrics = train_from_rows(
        candidate_rows,
        CANDIDATE_FEATURES,
        DEFAULT_CANDIDATE_WEIGHTS,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )
    board_weights, board_clipped = apply_sign_constraints(
        board_weights,
        non_negative=BOARD_NON_NEGATIVE,
        non_positive=BOARD_NON_POSITIVE,
    )
    candidate_weights, candidate_clipped = apply_sign_constraints(
        candidate_weights,
        non_negative=CANDIDATE_NON_NEGATIVE,
        non_positive=CANDIDATE_NON_POSITIVE,
    )
    if board_clipped:
        board_metrics["sign_constrained"] = board_clipped
    if candidate_clipped:
        candidate_metrics["sign_constrained"] = candidate_clipped
    return {
        "board_weights": board_weights,
        "candidate_weights": candidate_weights,
        "metadata": {
            **metadata,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "label_source": "weak_supervision_from_exported_route_metrics_and_selected_candidates",
            "board_training": board_metrics,
            "candidate_training": candidate_metrics,
        },
    }


def train_value_models_from_training_pack(
    pack_dir: Path,
    *,
    epochs: int = 120,
    lr: float = 0.08,
    l2: float = 0.02,
) -> dict[str, Any]:
    board_rows, candidate_rows, metadata = collect_training_pack_rows(pack_dir)
    board_weights, board_metrics = train_from_rows(
        board_rows,
        BOARD_FEATURES,
        DEFAULT_BOARD_WEIGHTS,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )
    candidate_weights, candidate_metrics = train_from_rows(
        candidate_rows,
        CANDIDATE_FEATURES,
        DEFAULT_CANDIDATE_WEIGHTS,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )
    board_weights, board_clipped = apply_sign_constraints(
        board_weights,
        non_negative=BOARD_NON_NEGATIVE,
        non_positive=BOARD_NON_POSITIVE,
    )
    candidate_weights, candidate_clipped = apply_sign_constraints(
        candidate_weights,
        non_negative=CANDIDATE_NON_NEGATIVE,
        non_positive=CANDIDATE_NON_POSITIVE,
    )
    if board_clipped:
        board_metrics["sign_constrained"] = board_clipped
    if candidate_clipped:
        candidate_metrics["sign_constrained"] = candidate_clipped
    return {
        "board_weights": board_weights,
        "candidate_weights": candidate_weights,
        "metadata": {
            **metadata,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "label_source": "training_pack_route_and_candidate_jsonl",
            "board_training": board_metrics,
            "candidate_training": candidate_metrics,
        },
    }


def write_report(payload: dict[str, Any], report_path: Path, output_path: Path, inputs: list[Path]) -> None:
    meta = payload.get("metadata") or {}
    source_line = (
        f"Training pack: `{meta.get('training_pack')}`"
        if meta.get("training_pack")
        else f"Inputs: `{len(inputs)}` file(s)"
    )
    lines = [
        "# Value Function Training",
        "",
        f"Output: `{output_path}`",
        source_line,
    ]
    if meta.get("files_loaded") is not None:
        lines.append(f"Files loaded: `{meta.get('files_loaded')}`")
    lines.extend([
        "",
        "## Samples",
        "",
        f"- board samples: `{meta.get('board_samples')}`",
        f"- candidate samples: `{meta.get('candidate_samples')}`",
        "",
        "## Board Model",
        "",
        json.dumps(meta.get("board_training") or {}, indent=2),
        "",
        "## Candidate Model",
        "",
        json.dumps(meta.get("candidate_training") or {}, indent=2),
        "",
        "## Label Caveat",
        "",
        "These are weak labels from planner artifacts, not experimentally validated reaction outcomes.",
        "Use them as a search value prior and retrain when curated success/failure route labels are available.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train artifact-calibrated route value-function weights")
    ap.add_argument("--input", action="append", default=None, help="Input JSON path or glob")
    ap.add_argument("--training-pack", default=None, help="Training pack directory with route_value/candidate_ranking JSONL")
    ap.add_argument("--output", default="results/shared/value_function/weights.json")
    ap.add_argument("--report", default="results/shared/value_function/training_report.md")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=0.02)
    args = ap.parse_args()

    if args.training_pack:
        paths = []
        payload = train_value_models_from_training_pack(
            Path(args.training_pack),
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
        )
    else:
        paths = expand_inputs(args.input or ["results/v2/*.json"])
        payload = train_value_models(paths, epochs=args.epochs, lr=args.lr, l2=args.l2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, Path(args.report), output, paths)
    print(json.dumps(payload.get("metadata") or {}, indent=2))


if __name__ == "__main__":
    main()
