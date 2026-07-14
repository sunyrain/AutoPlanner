"""CLI surface for genuine target-only retrosynthesis campaigns."""
from __future__ import annotations

import argparse
from typing import Any

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.live_evidence import (
    HttpEvidenceConnectorConfig,
    build_http_evidence_connector,
)
from cascade_planner.interfaces.live_stock import load_versioned_inventory_snapshot
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
)
from cascade_planner.interfaces.validation_fork import ValidationForkConfig


TARGET_COMMANDS = frozenset(
    {"solve-target", "fork-validation", "import-evidence"}
)


def add_target_commands(sub: argparse._SubParsersAction) -> None:
    solve = sub.add_parser(
        "solve-target",
        help="run a fresh bounded Codex campaign from only an arbitrary target SMILES",
    )
    solve.add_argument("--target-smiles", required=True)
    solve.add_argument("--target-name", default="blind target")
    solve.add_argument("--run-id")
    solve.add_argument("--run-dir")
    solve.add_argument("--manifest", help="target-only manifest allowed by blind preflight")
    solve.add_argument("--resume", action="store_true")
    solve.add_argument(
        "--full-output",
        action="store_true",
        help="emit the full report JSON; default output is a bounded summary",
    )
    solve.add_argument(
        "--model",
        default=DEFAULT_TARGET_DIRECTOR_MODEL,
        help="explicit Codex model; defaults to the locally verified CLI tier",
    )
    solve.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    agent_mode = solve.add_mutually_exclusive_group()
    agent_mode.add_argument(
        "--coordinator",
        action="store_true",
        help="allow the global director to spawn specialist children (higher cost)",
    )
    agent_mode.add_argument(
        "--single-agent",
        action="store_true",
        help="compatibility flag; one bounded global director is already the default",
    )
    solve.add_argument("--no-web-search", action="store_true")
    solve.add_argument("--no-replan", action="store_true")
    solve.add_argument("--no-live-benchmark-stock", action="store_true")
    solve.add_argument(
        "--no-patent-self-evo",
        action="store_true",
        help="disable replay-gated cross-campaign patent reaction template memory",
    )
    solve.add_argument(
        "--self-evo-library",
        default="",
        help="optional external patent template library path",
    )
    solve.add_argument(
        "--no-auto-patent-evidence",
        action="store_true",
        help="disable the bounded HTML-first built-in patent evidence connector",
    )
    solve.add_argument(
        "--evidence-endpoint",
        help="trusted structured extraction HTTPS endpoint (loopback HTTP allowed)",
    )
    solve.add_argument("--evidence-provider-id", default="")
    solve.add_argument("--evidence-provider-version", default="")
    solve.add_argument("--evidence-token-env", default="")
    solve.add_argument(
        "--inventory-snapshot",
        help="versioned trusted supplier snapshot used for procurement closure",
    )
    solve.add_argument("--minimum-complete-routes", type=int, default=2)
    solve.add_argument("--minimum-edge-proof-level", type=int, default=2)
    solve.add_argument("--minimum-source-groups", type=int, default=2)
    solve.add_argument(
        "--stock-boundary",
        choices=("benchmark_search", "procurement", "in_house"),
        default="benchmark_search",
    )
    solve.add_argument("--max-model-invocations", type=int, default=2)
    solve.add_argument("--max-input-tokens", type=int, default=50_000)
    solve.add_argument("--max-output-tokens", type=int, default=14_000)
    solve.add_argument("--max-model-wall-time-s", type=float, default=720.0)
    solve.add_argument(
        "--max-visual-invocations",
        type=int,
        choices=(0, 1),
        default=0,
        help="opt in to at most one sparse Codex page-vision call; default is zero",
    )
    solve.add_argument("--max-visual-pages", type=int, choices=range(1, 9), default=4)
    solve.add_argument("--max-accepted-expansions", type=int, default=32)
    solve.add_argument("--max-attempt-runs", type=int, default=72)
    solve.add_argument("--max-map-reactions", type=int, default=48)
    solve.add_argument("--max-stock-molecules", type=int, default=24)
    solve.add_argument("--max-patent-sources", type=int, default=3)
    solve.add_argument("--max-self-evo-candidates", type=int, default=12)
    solve.add_argument(
        "--patent-publication",
        action="append",
        default=[],
        help="optional patent publication, Google Patents URL, or direct PDF seed",
    )

    validation_fork = sub.add_parser(
        "fork-validation",
        help=(
            "replay a completed target campaign into a zero-model run and "
            "revalidate it under the current host policy"
        ),
    )
    validation_fork.add_argument("source_run_id")
    validation_fork.add_argument("--source-run-dir")
    validation_fork.add_argument("--run-id")
    validation_fork.add_argument("--run-dir")
    validation_fork.add_argument("--no-live-benchmark-stock", action="store_true")
    validation_fork.add_argument("--no-auto-patent-evidence", action="store_true")
    validation_fork.add_argument("--full-output", action="store_true")
    validation_fork.add_argument("--max-map-reactions", type=int, default=64)
    validation_fork.add_argument("--max-stock-molecules", type=int, default=32)
    validation_fork.add_argument("--max-patent-sources", type=int, default=3)
    validation_fork.add_argument(
        "--patent-publication",
        action="append",
        default=[],
        help="optional patent publication, Google Patents URL, or direct PDF seed",
    )

    evidence = sub.add_parser(
        "import-evidence",
        help="import trusted structured exact-source rows into an unresolved run",
    )
    evidence.add_argument("run_id")
    evidence.add_argument("--run-dir")
    evidence.add_argument("--input", required=True)


def dispatch_target_command(gateway: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "import-evidence":
        return gateway.import_evidence(
            run_id=args.run_id,
            run_dir=args.run_dir,
            import_path=args.input,
        )
    if args.command == "fork-validation":
        evidence_connector = None
        if not args.no_auto_patent_evidence:
            from cascade_planner.interfaces.patent_evidence import (
                BuiltinPatentEvidenceConfig,
                build_builtin_patent_evidence_connector,
            )

            evidence_connector = build_builtin_patent_evidence_connector(
                BuiltinPatentEvidenceConfig(
                    cache_dir=gateway.paths.external_data_root / "patent-evidence",
                    seed_publications=tuple(args.patent_publication),
                    max_patents=args.max_patent_sources,
                    max_validated_edges=args.max_map_reactions,
                )
            )
        result = gateway.fork_target_validation(
            source_run_id=args.source_run_id,
            source_run_dir=args.source_run_dir,
            run_id=args.run_id,
            run_dir=args.run_dir,
            evidence_connector=evidence_connector,
            config=ValidationForkConfig(
                max_atom_mapping_reactions=args.max_map_reactions,
                max_live_stock_molecules=args.max_stock_molecules,
                enable_live_benchmark_stock=not args.no_live_benchmark_stock,
            ),
        )
        return result if args.full_output else _compact_validation_fork_result(result)
    if args.command != "solve-target":
        raise ValueError(f"unsupported_target_command:{args.command}")
    evidence_connector = None
    if args.evidence_endpoint:
        evidence_connector = build_http_evidence_connector(
            HttpEvidenceConnectorConfig(
                endpoint=args.evidence_endpoint,
                provider_id=args.evidence_provider_id,
                provider_version=args.evidence_provider_version,
                token_env=args.evidence_token_env,
            )
        )
    elif not args.no_auto_patent_evidence:
        from cascade_planner.interfaces.patent_evidence import (
            BuiltinPatentEvidenceConfig,
            build_builtin_patent_evidence_connector,
        )

        evidence_connector = build_builtin_patent_evidence_connector(
            BuiltinPatentEvidenceConfig(
                cache_dir=gateway.paths.external_data_root / "patent-evidence",
                seed_publications=tuple(args.patent_publication),
                max_patents=args.max_patent_sources,
                max_validated_edges=args.max_map_reactions,
            )
        )
    inventory_snapshot_builder = None
    if args.inventory_snapshot:
        if args.stock_boundary != "procurement":
            raise ValueError("inventory_snapshot_requires_procurement_boundary")
        frozen_inventory = load_versioned_inventory_snapshot(args.inventory_snapshot)

        def inventory_snapshot_builder(_smiles: Any, **_kwargs: Any) -> Any:
            return frozen_inventory

    visual_evidence_provider = None
    if args.max_visual_invocations:
        from cascade_planner.interfaces.visual_evidence import (
            CodexVisualEvidenceConfig,
            build_codex_visual_evidence_provider,
        )

        visual_evidence_provider = build_codex_visual_evidence_provider(
            CodexVisualEvidenceConfig(
                cache_dir=gateway.paths.external_data_root / "visual-evidence",
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_pages=args.max_visual_pages,
            )
        )

    result = gateway.solve_target(
        target_name=args.target_name,
        target_smiles=args.target_smiles,
        run_id=args.run_id,
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        resume=args.resume,
        evidence_connector=evidence_connector,
        visual_evidence_provider=visual_evidence_provider,
        inventory_snapshot_builder=inventory_snapshot_builder,
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=args.minimum_complete_routes,
            minimum_edge_proof_level=args.minimum_edge_proof_level,
            minimum_independent_source_groups=args.minimum_source_groups,
            stock_boundary=args.stock_boundary,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=args.max_model_invocations,
            max_total_input_tokens=args.max_input_tokens,
            max_total_output_tokens=args.max_output_tokens,
            max_total_wall_time_s=args.max_model_wall_time_s,
            max_visual_invocations=args.max_visual_invocations,
            max_accepted_expansions=args.max_accepted_expansions,
            max_attempt_runs=args.max_attempt_runs,
            max_prompt_context_bytes=96_000,
        ),
        config=TargetSolveConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            use_coordinator=args.coordinator and not args.single_agent,
            enable_web_search=not args.no_web_search,
            enable_replan=not args.no_replan,
            enable_live_benchmark_stock=not args.no_live_benchmark_stock,
            enable_patent_self_evolution=not args.no_patent_self_evo,
            self_evo_library_path=args.self_evo_library,
            enable_builtin_patent_evidence=(
                evidence_connector is None and not args.no_auto_patent_evidence
            ),
            max_atom_mapping_reactions=args.max_map_reactions,
            max_live_stock_molecules=args.max_stock_molecules,
            max_patent_sources=args.max_patent_sources,
            max_self_evo_template_candidates=args.max_self_evo_candidates,
            max_visual_evidence_pages=args.max_visual_pages,
        ),
    )
    return result if args.full_output else _compact_target_result(result)


def _compact_target_result(result: Any) -> dict[str, Any]:
    row = dict(result)
    gates = dict(row.get("gates") or {})
    resource = dict(row.get("resource_envelope") or {})
    return {
        "schema_version": "target_solve_cli_summary.v1",
        "run_id": str(row.get("run_id") or ""),
        "target": dict(row.get("target") or {}),
        "gates": dict(gates.get("gates") or {}),
        "highest_contiguous_gate": str(
            gates.get("highest_contiguous_gate") or "none"
        ),
        "counts": dict(gates.get("counts") or {}),
        "claim": dict(row.get("claim") or {}),
        "current_disposition": dict(row.get("current_disposition") or {}),
        "self_evolution": {
            "enabled": dict(row.get("self_evolution") or {}).get("enabled") is True,
            "library_path": str(
                dict(row.get("self_evolution") or {}).get("library_path") or ""
            ),
            "learned_template_count": sum(
                len(dict(value).get("learned_template_ids") or [])
                for value in dict(row.get("self_evolution") or {}).get(
                    "learning_stages"
                )
                or []
            ),
        },
        "model_cost": dict(row.get("model_cost") or {}),
        "resource_envelope": {
            "within_budget": resource.get("within_budget") is True,
            "observed": dict(resource.get("observed") or {}),
            "violations": list(resource.get("violations") or []),
        },
        "attempt_count": int(row.get("attempt_count") or 0),
        "accepted_expansion_count": int(
            row.get("accepted_expansion_count") or 0
        ),
        "stop_decision": dict(row.get("stop_decision") or {}),
        "report_path": str(row.get("report_path") or ""),
        "report_ref": dict(row.get("report_ref") or {}),
        "report_sha256": str(row.get("content_sha256") or ""),
        "semantics": {
            "full_report_is_content_addressed_and_written_to_report_path": True,
            "summary_omits_generated_routes_and_precursors": True,
        },
    }


def _compact_validation_fork_result(result: Any) -> dict[str, Any]:
    row = dict(result)
    gates = dict(row.get("gates") or {})
    lineage = dict(row.get("lineage") or {})
    return {
        "schema_version": "target_validation_fork_cli_summary.v1",
        "run_id": str(row.get("run_id") or ""),
        "source_run_id": str(lineage.get("source_run_id") or ""),
        "target": dict(row.get("target") or {}),
        "gates": dict(gates.get("gates") or {}),
        "highest_contiguous_gate": str(gates.get("highest_contiguous_gate") or "none"),
        "counts": dict(gates.get("counts") or {}),
        "claim": dict(row.get("claim") or {}),
        "current_disposition": dict(row.get("current_disposition") or {}),
        "model_cost": dict(row.get("model_cost") or {}),
        "attempt_count": int(row.get("attempt_count") or 0),
        "accepted_expansion_count": int(row.get("accepted_expansion_count") or 0),
        "report_path": str(row.get("report_path") or ""),
        "report_ref": dict(row.get("report_ref") or {}),
        "report_sha256": str(row.get("content_sha256") or ""),
        "semantics": {
            "source_plan_replayed_without_model_calls": True,
            "full_report_is_content_addressed_and_written_to_report_path": True,
        },
    }


__all__ = [
    "TARGET_COMMANDS",
    "add_target_commands",
    "dispatch_target_command",
]
