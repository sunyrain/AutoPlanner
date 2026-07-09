"""Summarize route-pool cascade evidence audits across fixed route pools."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_cascade_evidence_summary.v1"


DEFAULT_AUDITS = {
    "v4_test20": "route_pool_cascade_evidence_20/route_pool_cascade_evidence.json",
    "statin": "route_pool_cascade_evidence_statin/route_pool_cascade_evidence.json",
    "full100": "route_pool_cascade_evidence_full100/route_pool_cascade_evidence.json",
}


def summarize_route_pool_cascade_evidence(
    *,
    root: Path,
    output_json: Path,
    audit_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    audit_paths = audit_paths or {name: root / rel for name, rel in DEFAULT_AUDITS.items()}
    rows = {}
    for name, path in audit_paths.items():
        rows[name] = _summarize_one(path)
    interpretation = _interpret(rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "root": str(root),
            "output_json": str(output_json),
            "audit_paths": {name: str(path) for name, path in audit_paths.items()},
        },
        "audits": rows,
        "interpretation": interpretation,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _summarize_one(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    summary = payload.get("summary") or {}
    rates = summary.get("target_rates") or {}
    routes = int(summary.get("routes") or 0)
    targets = int(summary.get("targets") or rates.get("targets") or 0)
    return {
        "path": str(path),
        "exists": path.exists(),
        "routes": routes,
        "targets": targets,
        "multistep_routes": int(summary.get("multistep_routes") or 0),
        "routes_with_observed_pair_block": int(summary.get("routes_with_observed_pair_block") or 0),
        "routes_with_any_analog_block": int(summary.get("routes_with_any_analog_block") or 0),
        "routes_with_same_pair_analog_block": int(summary.get("routes_with_same_pair_analog_block") or 0),
        "observed_pair_route_rate": _rate(summary.get("routes_with_observed_pair_block"), routes),
        "any_analog_route_rate": _rate(summary.get("routes_with_any_analog_block"), routes),
        "same_pair_analog_route_rate": _rate(summary.get("routes_with_same_pair_analog_block"), routes),
        "targets_with_observed_pair_block_anywhere": int(rates.get("targets_with_observed_pair_block_anywhere") or 0),
        "targets_with_any_analog_block_anywhere": int(rates.get("targets_with_any_analog_block_anywhere") or 0),
        "targets_with_same_pair_analog_block_anywhere": int(rates.get("targets_with_same_pair_analog_block_anywhere") or 0),
        "observed_pair_block_at_10": float(rates.get("observed_pair_block_at_10") or 0.0),
        "any_analog_block_at_10": float(rates.get("any_analog_block_at_10") or 0.0),
        "same_pair_analog_block_at_10": float(rates.get("same_pair_analog_block_at_10") or 0.0),
        "best_any_block_min_sim_max": float(summary.get("best_any_block_min_sim_max") or 0.0),
        "best_same_pair_block_min_sim_max": float(summary.get("best_same_pair_block_min_sim_max") or 0.0),
    }


def _interpret(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    present = {name: row for name, row in rows.items() if row.get("exists")}
    same_pair_targets = sum(int(row.get("targets_with_same_pair_analog_block_anywhere") or 0) for row in present.values())
    total_targets = sum(int(row.get("targets") or 0) for row in present.values())
    any_analog_targets = sum(int(row.get("targets_with_any_analog_block_anywhere") or 0) for row in present.values())
    observed_pair_targets = sum(int(row.get("targets_with_observed_pair_block_anywhere") or 0) for row in present.values())
    return {
        "audits_present": len(present),
        "total_targets": total_targets,
        "observed_pair_target_rate_overall": _rate(observed_pair_targets, total_targets),
        "any_analog_target_rate_overall": _rate(any_analog_targets, total_targets),
        "same_pair_analog_target_rate_overall": _rate(same_pair_targets, total_targets),
        "strict_same_pair_analog_is_sparse": bool(total_targets and same_pair_targets / max(total_targets, 1) < 0.05),
        "recommended_use": (
            "use route-pool cascade evidence as diagnostic/review signal; "
            "do not treat broad route pools as reliable cascade-coherent generators"
        ),
        "next_step": (
            "review high-evidence route examples and improve transform/reaction-center typing "
            "before training a search-time scorer"
        ),
    }


def _markdown(result: dict[str, Any]) -> str:
    rows = result.get("audits") or {}
    interp = result.get("interpretation") or {}
    lines = [
        "# Route Pool Cascade Evidence Summary",
        "",
        "## Interpretation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in interp.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Audit Comparison",
            "",
            "| Pool | Targets | Routes | Multistep | Observed Pair Routes | Any Analog Routes | Same-Pair Analog Routes | Any Analog Targets | Same-Pair Analog Targets | Any@10 | Same-Pair@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in rows.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row.get("targets")),
                    str(row.get("routes")),
                    str(row.get("multistep_routes")),
                    f"{row.get('routes_with_observed_pair_block')} ({row.get('observed_pair_route_rate')})",
                    f"{row.get('routes_with_any_analog_block')} ({row.get('any_analog_route_rate')})",
                    f"{row.get('routes_with_same_pair_analog_block')} ({row.get('same_pair_analog_route_rate')})",
                    str(row.get("targets_with_any_analog_block_anywhere")),
                    str(row.get("targets_with_same_pair_analog_block_anywhere")),
                    str(row.get("any_analog_block_at_10")),
                    str(row.get("same_pair_analog_block_at_10")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- `observed_pair` is transform-pair support only and is broad.",
            "- `any_analog` is structural support but does not require matching transform pair.",
            "- `same-pair analog` is the strictest current proxy: structural support plus observed transform pair.",
            "- None of these fields is an expert route-feasibility label.",
            "",
        ]
    )
    return "\n".join(lines)


def _rate(num: Any, denom: Any) -> float:
    try:
        d = float(denom)
        if d == 0.0:
            return 0.0
        return round(float(num or 0.0) / d, 6)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize route-pool cascade evidence audits")
    parser.add_argument("--root", default="results/shared/cascadebench_strict_20260516")
    parser.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/route_pool_cascade_evidence_summary.json")
    args = parser.parse_args()
    result = summarize_route_pool_cascade_evidence(root=Path(args.root), output_json=Path(args.output_json))
    print(json.dumps({"interpretation": result["interpretation"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
