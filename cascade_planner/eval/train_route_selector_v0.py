"""Train RouteSelector-v0 from a route-pool selector pack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_planner.eval.train_route_pool_ranker import train_route_pool_ranker


SCHEMA_VERSION = "route_selector_v0_training.v1"


def train_route_selector_v0(
    *,
    pack_jsonl: Path,
    output_dir: Path,
    feature_set: str = "all",
    seed: int = 42,
    auto_resplit: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(pack_jsonl)
    _validate_rows(rows)
    splits = _split_rows(rows, auto_resplit=auto_resplit)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        name: split_dir / f"route_selector_{name}.jsonl"
        for name in ("train", "val", "test")
    }
    for name, path in split_paths.items():
        _write_jsonl(path, splits[name])

    result = train_route_pool_ranker(
        train_jsonl=split_paths["train"],
        val_jsonl=split_paths["val"],
        test_jsonl=split_paths["test"],
        output_dir=output_dir,
        feature_set=feature_set,
        seed=seed,
    )
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "pack_jsonl": str(pack_jsonl),
        "output_dir": str(output_dir),
        "feature_set": feature_set,
        "seed": int(seed),
        "auto_resplit": bool(auto_resplit),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "selector_report": str(output_dir / "route_pool_pairwise_ranker_report.json"),
        "selector_model": str(output_dir / "route_pool_pairwise_ranker.pkl"),
        "selection": result.get("selection") or {},
    }
    (output_dir / "route_selector_v0_report.json").write_text(json.dumps(wrapper, indent=2, ensure_ascii=False), encoding="utf-8")
    return wrapper


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty selector pack")
    missing_feature = [idx for idx, row in enumerate(rows) if not isinstance(row.get("feature"), dict)]
    if missing_feature:
        raise ValueError(f"selector pack rows missing feature dict: first index {missing_feature[0]}")
    missing_label = [idx for idx, row in enumerate(rows) if row.get("route_label") is None]
    if missing_label:
        raise ValueError(f"selector pack rows missing route_label: first index {missing_label[0]}")


def _split_rows(rows: list[dict[str, Any]], *, auto_resplit: bool = True) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "train")
        if split not in splits:
            split = "train"
        splits[split].append(row)
    if auto_resplit and any(not items for items in splits.values()):
        splits = _round_robin_group_split(rows)
    missing = [name for name, items in splits.items() if not items]
    if missing:
        raise ValueError(f"selector pack must contain train/val/test rows; missing {missing}")
    return splits


def _round_robin_group_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("selector_group_id") or row.get("target_id") or ""), []).append(row)
    ordered_groups = sorted(groups.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    splits = {"train": [], "val": [], "test": []}
    split_names = ["train", "val", "test"]
    for idx, (_, group_rows) in enumerate(ordered_groups):
        splits[split_names[idx % len(split_names)]].extend(group_rows)
    return splits


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RouteSelector-v0 from a route-pool selector pack.")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-auto-resplit", action="store_true", help="Fail if the pack does not already contain train/val/test rows.")
    args = ap.parse_args()
    report = train_route_selector_v0(
        pack_jsonl=Path(args.pack),
        output_dir=Path(args.output_dir),
        feature_set=args.feature_set,
        seed=args.seed,
        auto_resplit=not args.no_auto_resplit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
