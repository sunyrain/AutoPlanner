"""Benchmark the deterministic Nirmatrelvir V3 acceptance replay.

The benchmark never enables a model.  It runs the same hash-bound scientific
case in fresh iteration directories and checks both scientific minimums and
coarse engineering ceilings.  The first iteration is labelled cold and later
iterations warm; caches owned outside the run directory may therefore be
reused without weakening the scientific replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.runtime.run_metrics import validate_run_metrics  # noqa: E402
from cascade_planner.runtime.run_index import RUN_MANIFEST_SCHEMA  # noqa: E402
from cascade_planner.runtime.run_storage import (  # noqa: E402
    publish_run_projection,
    rebuild_run_index,
    run_storage_object_stats,
)
from cascade_planner.legacy.runtime.run_manifest_compatibility import (  # noqa: E402
    write_run_manifest_compatibility,
)
from scripts.legacy.run_nirmatrelvir_v3_golden import (  # noqa: E402
    DEFAULT_GOLDEN,
    run_golden_case,
)


BENCHMARK_SCHEMA = "retrosynthesis_performance_benchmark.v1"
DEFAULT_CONTRACT = (
    ROOT / "config/benchmarks/nirmatrelvir_v3_performance_contract.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        ROOT / "results/shared" / f"nirmatrelvir_v3_benchmark_{timestamp}"
    )
    summary = benchmark_nirmatrelvir_v3(
        golden_path=args.golden,
        contract_path=args.contract,
        output_dir=output_dir,
        iterations=max(1, int(args.iterations)),
        timeout_s=max(1.0, float(args.timeout_s)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary.get("accepted") is not True:
        raise SystemExit(1)


def benchmark_nirmatrelvir_v3(
    *,
    golden_path: Path = DEFAULT_GOLDEN,
    contract_path: Path = DEFAULT_CONTRACT,
    output_dir: Path,
    iterations: int = 2,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    contract = _read_object(contract_path)
    if contract.get("schema_version") != "retrosynthesis_performance_contract.v1":
        raise ValueError("unsupported retrosynthesis performance contract")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime_root = output / "runtime"
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for index in range(max(1, int(iterations))):
        label = "cold" if index == 0 else f"warm-{index}"
        iteration_dir = output / "iterations" / label
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        result = run_golden_case(
            golden_path=golden_path,
            output_dir=iteration_dir,
            timeout_s=timeout_s,
            resolver_cache_root=runtime_root / "artifacts",
            runtime_root=runtime_root,
        )
        wall_time_s = round(max(0.0, time.perf_counter() - wall_started), 6)
        cpu_time_s = round(max(0.0, time.process_time() - cpu_started), 6)
        metrics = dict(result.get("run_metrics") or {})
        metric_reasons = validate_run_metrics(metrics)
        counters = dict(metrics.get("counters") or {})
        portfolio = dict(result.get("portfolio") or {})
        row = {
            "iteration": index + 1,
            "mode": label,
            "accepted": result.get("accepted") is True,
            "wall_time_s": wall_time_s,
            "cpu_time_s": cpu_time_s,
            "artifact_bytes": _directory_bytes(iteration_dir),
            "model_invocations": int(result.get("model_invocations") or 0),
            "approved_source_step_count": int(
                portfolio.get("approved_source_step_count") or 0
            ),
            "unique_reaction_hyperedge_count": int(
                portfolio.get("hyperedge_count") or 0
            ),
            "complete_route_count": int(
                portfolio.get("complete_route_count") or 0
            ),
            "selected_route_count": int(
                portfolio.get("selected_route_count") or 0
            ),
            "stock_terminal_count": int(
                portfolio.get("stock_terminal_count") or 0
            ),
            "independent_support_group_count": len(
                portfolio.get("independent_support_groups") or []
            ),
            "run_metrics_sha256": str(metrics.get("content_sha256") or ""),
            "run_metrics_reasons": metric_reasons,
            "resolver_cache_hits": int(
                sum(
                    float(value or 0)
                    for key, value in counters.items()
                    if str(key).endswith(".cache_hit")
                )
            ),
            "resolver_cache_misses": int(
                sum(
                    float(value or 0)
                    for key, value in counters.items()
                    if str(key).endswith(".cache_miss")
                )
            ),
            "persistent_resolver_cache_hits": int(
                sum(
                    float(value or 0)
                    for key, value in counters.items()
                    if str(key).endswith(".persistent_cache_hit")
                )
            ),
        }
        run_id = f"{contract.get('case_id') or 'nirmatrelvir'}:{label}"
        artifact_paths = {
            path.relative_to(iteration_dir).as_posix(): path
            for path in iteration_dir.rglob("*")
            if path.is_file() and path.name != "run_manifest.json"
        }
        storage = publish_run_projection(
            runtime_root,
            manifest={
                "schema_version": RUN_MANIFEST_SCHEMA,
                "run_id": run_id,
                "case_id": str(contract.get("case_id") or ""),
                "target_name": "nirmatrelvir",
                "producer": "scripts.legacy.benchmark_nirmatrelvir_v3",
                "status": "completed" if result.get("accepted") else "failed",
                "revision": 1,
                "updated_at": str(metrics.get("observed_at") or ""),
                "run_dir": str(iteration_dir),
                "state_sha256": _digest(result),
                "accepted": result.get("accepted") is True,
                "cost_totals": {
                    "model_invocations": int(
                        result.get("model_invocations") or 0
                    ),
                    "attempt_runs": 0,
                    "accepted_expansions": 0,
                },
                "graph": {
                    "molecule_count": int(
                        portfolio.get("molecule_node_count") or 0
                    ),
                    "hyperedge_count": int(
                        portfolio.get("hyperedge_count") or 0
                    ),
                    "complete_route_count": int(
                        portfolio.get("complete_route_count") or 0
                    ),
                },
                "deficits": {
                    "proof": 0 if result.get("accepted") else 1,
                    "stock": 0 if result.get("accepted") else 1,
                },
                "metrics": {
                    "sha256": str(metrics.get("content_sha256") or ""),
                },
            },
            artifacts=artifact_paths,
            authority_scopes={
                artifact_id: _artifact_authority_scope(path)
                for artifact_id, path in artifact_paths.items()
            },
        )
        write_run_manifest_compatibility(
            iteration_dir / "run_manifest.json",
            storage["manifest"],
        )
        row["runtime_storage"] = {
            key: storage[key]
            for key in (
                "run_id",
                "revision",
                "manifest_ref",
                "artifact_count",
                "index_health",
                "semantics",
            )
        }
        row_reasons = _iteration_reasons(row, contract)
        if metric_reasons:
            row_reasons.append("run_metrics_invalid")
        row["reasons"] = sorted(set(row_reasons))
        row["accepted"] = bool(row["accepted"] and not row["reasons"])
        reasons.extend(f"{label}:{reason}" for reason in row["reasons"])
        rows.append(row)

    cold_wall = float(rows[0]["wall_time_s"])
    last_wall = float(rows[-1]["wall_time_s"])
    rebuilt_index_path = runtime_root / "run_index.rebuilt.sqlite3"
    rebuild = rebuild_run_index(runtime_root, index_path=rebuilt_index_path)
    summary: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA,
        "case_id": str(contract.get("case_id") or ""),
        "accepted": not reasons and all(row["accepted"] for row in rows),
        "reasons": sorted(set(reasons)),
        "contract_path": str(contract_path.expanduser().resolve()),
        "golden_path": str(golden_path.expanduser().resolve()),
        "output_dir": str(output),
        "iterations": rows,
        "cold_to_last_speedup": (
            round(cold_wall / last_wall, 4) if last_wall > 0.0 else None
        ),
        "runtime_storage": {
            "stats": run_storage_object_stats(runtime_root),
            "rebuild": rebuild,
        },
        "semantics": {
            "model_calls_are_forbidden": True,
            "metrics_are_observability_only": True,
            "scientific_acceptance_remains_authoritative": True,
            "warm_speedup_is_reported_not_required": True,
        },
    }
    summary["content_sha256"] = _digest(summary)
    _write_json(output / "benchmark_summary.json", summary)
    return summary


def _iteration_reasons(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    minimums = dict(contract.get("scientific_minimums") or {})
    for key, raw_minimum in minimums.items():
        if float(row.get(key) or 0) < float(raw_minimum):
            reasons.append(f"scientific_minimum_not_met:{key}")
    limits = dict(contract.get("engineering_limits") or {})
    comparisons = {
        "artifact_bytes": "max_artifact_bytes_per_iteration",
        "cpu_time_s": "max_cpu_time_s_per_iteration",
        "model_invocations": "max_model_invocations",
        "wall_time_s": "max_wall_time_s_per_iteration",
    }
    for row_key, limit_key in comparisons.items():
        if limit_key in limits and float(row.get(row_key) or 0) > float(
            limits[limit_key]
        ):
            reasons.append(f"engineering_limit_exceeded:{row_key}")
    if row.get("accepted") is not True:
        reasons.append("golden_replay_not_accepted")
    return reasons


def _directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _artifact_authority_scope(path: Path) -> str:
    name = path.name.casefold()
    if "trusted_literature_step_registry" in name:
        return "scientific_artifact_reference"
    if "audit" in name or "portfolio" in path.as_posix().casefold():
        return "scientific_validation_projection"
    return "operational_projection"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
