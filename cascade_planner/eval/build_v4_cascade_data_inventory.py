"""Inventory v4 cascade data and route-pool assets for reranker runs."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def build_v4_cascade_data_inventory(
    *,
    v4_dir: Path,
    split_dir: Path | None,
    native_pools: list[Path],
    output: Path,
) -> dict[str, Any]:
    manifest = _read_json(v4_dir / "manifest.json")
    inventory = {
        "schema_version": "v4_cascade_data_inventory.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v4_dir": str(v4_dir),
        "v4_release": {
            "manifest": str(v4_dir / "manifest.json"),
            "quality_summary": manifest.get("quality_summary") if isinstance(manifest, dict) else {},
            "files": _file_status(v4_dir),
        },
        "v4_counts": _v4_counts(v4_dir),
        "split_manifest": _split_manifest(split_dir),
        "native_pools": [_native_pool_summary(path) for path in native_pools],
        "training_policy": {
            "primary_training_data": "dataset_v4_release gold/silver high_quality",
            "forbidden_primary_training_data": "full100 eval-only route outcomes and AutoPlanner route-tree traces",
            "chem_enzy_role": "native route proposal pool and baseline, not the supervision source",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    return inventory


def _v4_counts(v4_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in [
        "cascade_v4_high_quality.csv",
        "cascade_v4_gold_reactions.csv",
        "cascade_v4_silver_reactions.csv",
        "cascade_v4_bronze_reactions.csv",
        "cascade_v4_quarantine_reactions.csv",
        "cascade_v4_steps.csv",
        "cascade_v4_catalysts.csv",
        "cascade_v4_species.csv",
        "cascade_v4_substrate_scope.csv",
    ]:
        path = v4_dir / name
        if path.exists():
            out[name] = {"rows": _csv_count(path), "bytes": path.stat().st_size}
    return out


def _file_status(v4_dir: Path) -> dict[str, Any]:
    out = {}
    for path in sorted(v4_dir.iterdir()):
        if path.is_file():
            out[path.name] = {"bytes": path.stat().st_size}
    return out


def _split_manifest(split_dir: Path | None) -> dict[str, Any] | None:
    if split_dir is None:
        return None
    path = split_dir / "v4_trace_split_manifest.json"
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = _read_json(path)
    return {
        "path": str(path),
        "exists": True,
        "counts": data.get("counts") if isinstance(data, dict) else {},
        "splits": data.get("splits") if isinstance(data, dict) else {},
        "leakage_checks": data.get("leakage_checks") if isinstance(data, dict) else {},
    }


def _native_pool_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = _read_json(path)
    targets = data.get("targets") if isinstance(data, dict) else []
    route_counts = [len(target.get("routes") or []) for target in targets if isinstance(target, dict)]
    return {
        "path": str(path),
        "exists": True,
        "targets": len(targets),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(route_counts), 1), 6),
        "summary": data.get("summary") if isinstance(data, dict) else {},
    }


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v4 cascade data inventory")
    ap.add_argument("--v4-dir", default="dataset_v4_release")
    ap.add_argument("--split-dir")
    ap.add_argument("--native-pool", action="append", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    inventory = build_v4_cascade_data_inventory(
        v4_dir=Path(args.v4_dir),
        split_dir=Path(args.split_dir) if args.split_dir else None,
        native_pools=[Path(path) for path in args.native_pool],
        output=Path(args.output),
    )
    print(json.dumps({"v4_counts": inventory["v4_counts"], "native_pools": inventory["native_pools"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
