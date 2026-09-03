#!/usr/bin/env python3
"""Replay the frozen external-snapshot arm through the canonical host graph.

This evaluator-only process can read the reference route pack.  It performs no
route generation and makes no LLM calls.  Every route is imported separately
so one malformed public variant cannot erase valid variants for the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    BlindCase,
    load_blind_manifest,
)
from cascade_planner.application.retrosynthesis_run_contract import (  # noqa: E402
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import (  # noqa: E402
    stock_oracle_reference_from_builder,
)
from cascade_planner.eval.strategy_closure_pilot import (  # noqa: E402
    STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA,
    STRATEGY_CLOSURE_PILOT_PROTOCOL_SCHEMA,
    external_bundle_for_case,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway  # noqa: E402
from cascade_planner.interfaces.live_stock import (  # noqa: E402
    FrozenBenchmarkStockIndex,
)
from cascade_planner.runtime.paths import RuntimePaths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="benchmarks/synthatlas_strategy_closure20.v1.json"
    )
    parser.add_argument(
        "--protocol",
        default="benchmarks/synthatlas_strategy_closure20.protocol.json",
    )
    parser.add_argument(
        "--evaluator-pack",
        default=(
            "data_external/synthatlas/strategy_closure20_20260812/evaluator_pack.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results/shared/synthatlas_strategy_closure20_external_snapshot",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = _resolve(args.manifest)
    protocol_path = _resolve(args.protocol)
    evaluator_pack_path = _resolve(args.evaluator_pack)
    output_root = _resolve(args.output_root)
    manifest_cases = list(load_blind_manifest(manifest_path))
    protocol = _load_json(protocol_path)
    evaluator_pack = _load_json(evaluator_pack_path)
    _validate_bindings(
        manifest_path=manifest_path,
        protocol=protocol,
        evaluator_pack=evaluator_pack,
        manifest_cases=manifest_cases,
    )
    _validate_output_root(output_root, resume=args.resume)
    paths = RuntimePaths.discover(
        repository_root=ROOT,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(output_root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(output_root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(output_root / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(output_root / "run_index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(output_root / "external"),
        },
    )
    gateway = CampaignGateway(paths)
    stock_oracle = _frozen_stock_oracle(protocol)
    evaluator_cases = {
        str(row.get("case_id") or ""): row for row in evaluator_pack["cases"]
    }
    results: list[dict[str, Any]] = []
    for case in manifest_cases:
        result_path = output_root / "cases" / f"{case.case_id}.json"
        if result_path.is_file():
            if not args.resume:
                raise SystemExit(f"case result already exists: {case.case_id}")
            result = _load_json(result_path)
            if not _digest_valid(result):
                raise SystemExit(f"case result digest invalid: {case.case_id}")
            results.append(result)
            continue
        evaluator_case = evaluator_cases.get(case.case_id)
        if not isinstance(evaluator_case, Mapping):
            raise SystemExit(f"evaluator case missing: {case.case_id}")
        result = _run_case(
            gateway,
            case=case,
            evaluator_case=evaluator_case,
            stock_oracle=stock_oracle,
        )
        _write_json(result_path, result)
        results.append(result)
        _write_summary(output_root, protocol=protocol, results=results, complete=False)
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "routes": result["route_count"],
                    "C0": result["closure_counts"]["C0"],
                    "C1": result["closure_counts"]["C1"],
                }
            ),
            flush=True,
        )
    summary = _write_summary(
        output_root, protocol=protocol, results=results, complete=True
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_case(
    gateway: CampaignGateway,
    *,
    case: BlindCase,
    evaluator_case: Mapping[str, Any],
    stock_oracle: Any,
) -> dict[str, Any]:
    if str(evaluator_case.get("target_smiles") or "") != case.target_smiles:
        raise RuntimeError(f"external_arm_target_mismatch:{case.case_id}")
    route_results: list[dict[str, Any]] = []
    for route in evaluator_case.get("routes") or []:
        route_id = str(route.get("route_id") or "")
        run_id = f"external-snapshot:{case.case_id}:{route_id}"
        c0 = dict(route.get("host_c0_preflight") or {})
        if c0.get("accepted") is not True:
            route_results.append(
                {
                    "route_id": route_id,
                    "run_id": "",
                    "C0": "failed",
                    "C1": "not_attempted",
                    "reason": str(c0.get("reason") or "c0_preflight_failed"),
                }
            )
            continue
        try:
            bundle = external_bundle_for_case(
                {
                    "target_smiles": case.target_smiles,
                    "routes": [route],
                }
            )
            gateway.create_run(
                target_name=case.target_name,
                target_smiles=case.target_smiles,
                run_id=run_id,
                acceptance=_acceptance(case),
                budget=_zero_model_budget(case),
                stock_oracle_reference=stock_oracle,
            )
            imported = gateway.import_strategy_routes(
                run_id=run_id,
                bundle=bundle,
                materialize=True,
            )
            closure = dict(imported["strategy_to_experiment_closure"])
            closure_route = dict(closure["routes"][0])
            materialization = dict(closure_route["canonical_materialization"])
            service = gateway._open(run_id)
            state = service.kernel.state
            route_results.append(
                {
                    "route_id": route_id,
                    "run_id": run_id,
                    "C0": "complete",
                    "C1": str(materialization.get("status") or "open"),
                    "C2": str(
                        closure_route["host_reaction_validation"].get("status")
                        or "open"
                    ),
                    "C3": str(
                        closure_route["exact_source_evidence"].get("status") or "open"
                    ),
                    "C4": str(
                        closure_route["complete_exact_conditions"].get("status")
                        or "open"
                    ),
                    "C5": str(closure_route["stock_closure"].get("status") or "open"),
                    "C6": "not_assessed_by_route_import",
                    "import_receipt_sha256": str(
                        imported["external_strategy_import"].get("content_sha256") or ""
                    ),
                    "hypothesis_count": int(
                        closure_route["strategy_structure"].get("required") or 0
                    ),
                    "materialized_edge_count": int(
                        materialization.get("achieved") or 0
                    ),
                    "materialization_blockers": list(
                        materialization.get("blockers") or []
                    ),
                    "resource_usage": {
                        "settled_task_count": int(state.settled_task_count),
                        "accepted_expansion_count": int(state.accepted_expansion_count),
                        "model_totals": dict(state.model_totals),
                        "stock_oracle_binding_sha256": str(
                            service.kernel.spec.campaign_spec.stock_oracle.binding_sha256
                        ),
                    },
                    "reason": _materialization_reason(materialization),
                }
            )
        except Exception as exc:  # evaluator must retain every failure class
            route_results.append(
                {
                    "route_id": route_id,
                    "run_id": run_id,
                    "C0": "complete",
                    "C1": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    counts = {
        level: sum(row.get(level) == "complete" for row in route_results)
        for level in ("C0", "C1", "C2", "C3", "C4", "C5")
    }
    result = {
        "schema_version": "strategy_closure_external_arm_case_result.v1",
        "case_id": case.case_id,
        "run_ids": [str(row.get("run_id") or "") for row in route_results],
        "route_count": len(route_results),
        "closure_counts": counts,
        "routes": route_results,
        "resource_usage": {
            "route_generation_model_invocations": 0,
            "route_generation_input_tokens": 0,
            "route_generation_output_tokens": 0,
            "generation_cost_not_observed_for_public_snapshot": True,
            "host_settled_task_count": sum(
                int(
                    dict(row.get("resource_usage") or {}).get("settled_task_count") or 0
                )
                for row in route_results
            ),
            "host_accepted_expansion_count": sum(
                int(
                    dict(row.get("resource_usage") or {}).get(
                        "accepted_expansion_count"
                    )
                    or 0
                )
                for row in route_results
            ),
        },
        "semantics": {
            "external_route_failures_are_retained": True,
            "each_route_has_an_isolated_run_kernel": True,
            "cross_variant_ancestor_state_cannot_change_c1": True,
            "provider_claims_grant_no_host_authority": True,
        },
    }
    return _with_digest(result)


def _acceptance(case: BlindCase) -> RetrosynthesisAcceptanceSpec:
    values = dict(case.acceptance)
    values.pop("minimum_planning_route_steps", None)
    return RetrosynthesisAcceptanceSpec(**values)


def _zero_model_budget(case: BlindCase) -> RetrosynthesisRunBudget:
    values = dict(case.budget)
    values.update(
        {
            "max_model_invocations": 0,
            "max_total_input_tokens": 0,
            "max_total_output_tokens": 0,
            "max_visual_invocations": 0,
            "max_native_search_invocations": 0,
            "min_target_native_search_invocations": 0,
            "max_frontier_native_search_invocations": 0,
            "max_prompt_context_bytes": 0,
        }
    )
    return RetrosynthesisRunBudget(**values)


def _write_summary(
    output_root: Path,
    *,
    protocol: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    route_count = sum(int(row.get("route_count") or 0) for row in results)
    closure_counts = {
        level: sum(
            int(dict(row.get("closure_counts") or {}).get(level) or 0)
            for row in results
        )
        for level in ("C0", "C1", "C2", "C3", "C4", "C5")
    }
    summary = _with_digest(
        {
            "schema_version": "strategy_closure_external_arm_summary.v1",
            "status": "completed" if complete else "running",
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "protocol_content_sha256": str(protocol.get("content_sha256") or ""),
            "case_count": len(results),
            "route_count": route_count,
            "closure_counts": closure_counts,
            "closure_rates": {
                level: (closure_counts[level] / route_count if route_count else 0.0)
                for level in closure_counts
            },
            "failure_taxonomy": _failure_taxonomy(results),
            "resource_usage": {
                "route_generation_model_invocations": 0,
                "route_generation_input_tokens": 0,
                "route_generation_output_tokens": 0,
                "host_settled_task_count": sum(
                    int(
                        dict(row.get("resource_usage") or {}).get(
                            "host_settled_task_count"
                        )
                        or 0
                    )
                    for row in results
                ),
                "host_accepted_expansion_count": sum(
                    int(
                        dict(row.get("resource_usage") or {}).get(
                            "host_accepted_expansion_count"
                        )
                        or 0
                    )
                    for row in results
                ),
            },
            "claim_boundary": (
                "This is fixed public-route host replay, not a live SynthEx runtime "
                "or cost reproduction. C2-C6 remain open unless host records close them."
            ),
        }
    )
    _write_json(output_root / "summary.json", summary)
    return summary


def _failure_taxonomy(results: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for route in result.get("routes") or []:
            reason = str(route.get("reason") or "")
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _materialization_reason(materialization: Mapping[str, Any]) -> str:
    if materialization.get("status") == "complete":
        return ""
    reasons = sorted(
        {
            str(reason)
            for blocker in materialization.get("blockers") or []
            for reason in dict(blocker).get("reasons") or []
            if str(reason)
        }
    )
    return "c1_materialization:" + (",".join(reasons) if reasons else "incomplete")


def _validate_bindings(
    *,
    manifest_path: Path,
    protocol: Mapping[str, Any],
    evaluator_pack: Mapping[str, Any],
    manifest_cases: list[BlindCase],
) -> None:
    if protocol.get("schema_version") != STRATEGY_CLOSURE_PILOT_PROTOCOL_SCHEMA:
        raise SystemExit("strategy closure protocol schema mismatch")
    if evaluator_pack.get("schema_version") != STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA:
        raise SystemExit("strategy closure evaluator pack schema mismatch")
    if not _digest_valid(protocol) or not _digest_valid(evaluator_pack):
        raise SystemExit("strategy closure content digest invalid")
    bindings = dict(protocol.get("bindings") or {})
    manifest = _load_json(manifest_path)
    if bindings.get("target_manifest_content_sha256") != _json_digest(manifest):
        raise SystemExit("strategy closure target manifest binding mismatch")
    if bindings.get("evaluator_pack_content_sha256") != evaluator_pack.get(
        "content_sha256"
    ):
        raise SystemExit("strategy closure evaluator pack binding mismatch")
    evaluator_ids = [str(row.get("case_id") or "") for row in evaluator_pack["cases"]]
    if evaluator_ids != [case.case_id for case in manifest_cases]:
        raise SystemExit("strategy closure case order mismatch")


def _frozen_stock_oracle(protocol: Mapping[str, Any]) -> Any:
    stock = dict(dict(protocol.get("bindings") or {}).get("stock_oracle") or {})
    path_text = str(stock.get("index_path") or "")
    path = _resolve(path_text)
    builder = FrozenBenchmarkStockIndex(
        path,
        expected_sha256=str(stock.get("index_sha256") or ""),
        catalog_name=str(stock.get("catalog_name") or ""),
    )
    if builder.member_count != int(stock.get("member_count") or 0):
        raise SystemExit("strategy closure stock member count mismatch")
    return stock_oracle_reference_from_builder(
        builder,
        boundary="benchmark_search",
    )


def _validate_output_root(path: Path, *, resume: bool) -> None:
    if path.exists() and any(path.iterdir()) and not resume:
        raise SystemExit("external arm output root is not fresh")
    path.mkdir(parents=True, exist_ok=True)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _json_digest(result)
    return result


def _digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return len(supplied) == 64 and supplied == _json_digest(row)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
