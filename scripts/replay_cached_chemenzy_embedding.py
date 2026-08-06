#!/usr/bin/env python3
"""Replay cached ChemEnzy and Codex proposals through the current V4 runtime."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.retrosynthesis_run_contract import (  # noqa: E402
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway  # noqa: E402
from cascade_planner.interfaces.live_stock import FrozenBenchmarkStockIndex  # noqa: E402
from cascade_planner.interfaces.target_solver import TargetSolveConfig  # noqa: E402
from cascade_planner.runtime import AgentResult, AgentState  # noqa: E402
from cascade_planner.runtime.paths import RuntimePaths  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _runtime_paths(output_root: Path) -> RuntimePaths:
    return RuntimePaths.discover(
        repository_root=ROOT,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(output_root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(output_root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(output_root / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(
                output_root / "index" / "runs.sqlite3"
            ),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(ROOT / "data_external"),
            "AUTOPLANNER_MODEL_ROOT": str(ROOT / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(ROOT / "vendor"),
        },
    )


def replay_cached_embedding(
    *,
    source_run: Path,
    output_root: Path,
    run_id: str,
    stock_index: Path,
    stock_index_sha256: str,
    stock_name: str,
    provider_route_reserve: int,
    host_route_portfolio: int,
    max_live_stock_molecules: int,
) -> dict[str, Any]:
    source = source_run.expanduser().resolve()
    destination = output_root.expanduser().resolve()
    raw_path = source / "chemenzy-v4-seed-result.json"
    request_path = source / "chemenzy-v4-seed-request.json"
    report_path = source / "target-only-solve-report.json"
    raw = _read_object(raw_path)
    request = _read_object(request_path)
    source_report = _read_object(report_path)
    initial_plan = deepcopy(
        next(
            dict(row["plan"])
            for row in source_report.get("director_outcomes") or []
            if row.get("mode") == "initial_architecture"
            and isinstance(row.get("plan"), dict)
        )
    )
    target = dict(source_report.get("target") or {})
    target_smiles = str(target.get("canonical_smiles") or "")
    if not target_smiles:
        raise ValueError("source report has no canonical target SMILES")
    audit_root = destination / "blind-audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    def cached_chemenzy_provider(**kwargs: Any) -> dict[str, Any]:
        proposal_request = dict(kwargs.get("request") or {})
        if proposal_request.get("mode") != "seed":
            return {
                "status": "completed",
                "routes": [],
                "reason": "cached_seed_only_replay",
            }
        if str(kwargs.get("target_smiles") or "") != target_smiles:
            raise AssertionError("cached_chemenzy_target_mismatch")
        return deepcopy(raw)

    cached_chemenzy_provider.model_free = True  # type: ignore[attr-defined]

    def cached_director_runner(
        spec: Any,
        context: Any,
        mode: str,
        _config: Any,
    ) -> AgentResult:
        if mode != "initial_architecture":
            raise AssertionError(f"unexpected_cached_director_mode:{mode}")
        plan = deepcopy(initial_plan)
        plan.pop("content_sha256", None)
        plan["plan_id"] = (
            f"cached-{mode}-{str(context.content_sha256)[:16]}"
        )
        plan["run_id"] = spec.run_id
        plan["mode"] = mode
        plan["context_sha256"] = context.content_sha256
        plan["graph_revision"] = context.revision.revision
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage={
                "model_invocations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_s": 0.0,
            },
            metadata={"cached_source_report": str(report_path)},
        )

    cached_director_runner.model_free = True  # type: ignore[attr-defined]
    stock = FrozenBenchmarkStockIndex(
        stock_index,
        expected_sha256=stock_index_sha256,
        catalog_name=stock_name,
    )
    acceptance_source = dict(source_report.get("acceptance") or {})
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=int(
            acceptance_source.get("minimum_complete_routes") or 1
        ),
        minimum_edge_proof_level=int(
            acceptance_source.get("minimum_edge_proof_level") or 2
        ),
        minimum_independent_source_groups=int(
            acceptance_source.get("minimum_independent_source_groups") or 2
        ),
        require_all_selected_leaves_stock_closed=bool(
            acceptance_source.get("require_all_selected_leaves_stock_closed", True)
        ),
        require_distinct_edge_sets=bool(
            acceptance_source.get("require_distinct_edge_sets", True)
        ),
        stock_boundary="benchmark_search",
    )
    result = CampaignGateway(_runtime_paths(destination)).solve_target(
        target_name=str(target.get("name") or "cached target"),
        target_smiles=target_smiles,
        run_id=run_id,
        acceptance=acceptance,
        budget=RetrosynthesisRunBudget(
            # The cached Director is marked model-free, so this slot remains
            # unspent.  It must still be schedulable as an architecture action.
            max_model_invocations=1,
            max_total_input_tokens=180_000,
            max_total_output_tokens=18_000,
            max_total_wall_time_s=1_800.0,
            max_visual_invocations=0,
            max_accepted_expansions=256,
            max_attempt_runs=512,
            max_native_search_invocations=1,
            min_target_native_search_invocations=1,
            max_frontier_native_search_invocations=0,
            allow_frontier_native_search_borrowing=False,
            max_prompt_context_bytes=160_000,
        ),
        config=TargetSolveConfig(
            model="cached-director-replay",
            reasoning_effort="high",
            execution_profile="proof",
            objective_mode="benchmark_search",
            use_coordinator=False,
            enable_web_search=False,
            enable_initial_director_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=True,
            enable_builtin_patent_evidence=False,
            enable_patent_self_evolution=False,
            enable_chemenzy=True,
            enable_target_chemenzy_baseline=True,
            enable_guided_chemenzy=False,
            enable_chemenzy_condition_prediction=bool(
                request.get("enable_condition_prediction", True)
            ),
            enable_condition_enrichment=False,
            enable_chemenzy_enzyme_assignment=bool(
                request.get("enable_enzyme_assignment", True)
            ),
            enable_enzyme_coverage_sidecar=bool(
                request.get("enable_enzyme_coverage_sidecar", True)
            ),
            enable_program_review=False,
            enable_program_admission=False,
            enable_program_discovery=False,
            enable_program_validation=False,
            enable_target_identity=False,
            resolve_named_target_identity=False,
            blind_audit_root=str(audit_root),
            max_atom_mapping_reactions=48,
            max_live_stock_molecules=max_live_stock_molecules,
            provider_route_reserve=provider_route_reserve,
            host_route_portfolio=host_route_portfolio,
            display_route_limit=min(12, host_route_portfolio),
            max_chemenzy_steps=int(request.get("max_steps") or 20),
            max_chemenzy_iterations=int(
                request.get("chem_enzy_iterations") or 120
            ),
            chemenzy_expansion_topk=int(
                request.get("chem_enzy_expansion_topk") or 180
            ),
            chemenzy_timeout_s=1_200.0,
            chemenzy_search_preset=str(
                request.get("search_preset") or "thorough"
            ),
            chemenzy_pandarallel_workers=2,
            max_director_output_tokens=18_000,
            max_director_wall_time_s=1_200.0,
        ),
        director_runner=cached_director_runner,
        stock_catalog_builder=stock,
        chemenzy_provider=cached_chemenzy_provider,
    )
    return {
        "schema_version": "cached_chemenzy_embedding_replay.v1",
        "source_run": str(source),
        "source_raw_result": str(raw_path),
        "run_id": run_id,
        "run_dir": str(result.get("run_dir") or ""),
        "report_path": str(result.get("report_path") or ""),
        "model_cost": dict(result.get("model_cost") or {}),
        "attempt_count": int(result.get("attempt_count") or 0),
        "accepted_expansion_count": int(
            result.get("accepted_expansion_count") or 0
        ),
        "gates": dict(dict(result.get("gates") or {}).get("gates") or {}),
        "counts": dict(dict(result.get("gates") or {}).get("counts") or {}),
        "claim": dict(result.get("claim") or {}),
        "stop_decision": dict(result.get("stop_decision") or {}),
        "semantics": {
            "cached_provider_payload_replayed_without_new_provider_call": True,
            "cached_director_plan_rebound_to_current_context": True,
            "current_v4_runtime_used": True,
            "frozen_benchmark_stock_used": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stock-index", required=True, type=Path)
    parser.add_argument("--stock-index-sha256", required=True)
    parser.add_argument(
        "--stock-name",
        default="frozen-benchmark-stock",
    )
    parser.add_argument("--provider-route-reserve", type=int, default=2)
    parser.add_argument("--host-route-portfolio", type=int, default=2)
    parser.add_argument("--max-live-stock-molecules", type=int, default=24)
    args = parser.parse_args(argv)
    result = replay_cached_embedding(
        source_run=args.source_run,
        output_root=args.output_root,
        run_id=args.run_id,
        stock_index=args.stock_index,
        stock_index_sha256=args.stock_index_sha256,
        stock_name=args.stock_name,
        provider_route_reserve=args.provider_route_reserve,
        host_route_portfolio=args.host_route_portfolio,
        max_live_stock_molecules=args.max_live_stock_molecules,
    )
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
