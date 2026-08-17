"""CLI surface for genuine target-only retrosynthesis campaigns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings
from typing import Any

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import TargetConstraints
from cascade_planner.interfaces.live_evidence import (
    HttpEvidenceConnectorConfig,
    build_http_evidence_connector,
    compose_evidence_connectors,
)
from cascade_planner.interfaces.live_stock import (
    FrozenBenchmarkStockIndex,
    FrozenInventorySnapshotBuilder,
)
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
)
from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
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
    solve.add_argument(
        "--target-name",
        default="",
        help="optional public identity; omitted targets receive an opaque hash label",
    )
    solve.add_argument("--run-id")
    solve.add_argument("--run-dir")
    solve.add_argument("--manifest", help="target-only manifest allowed by blind preflight")
    solve.add_argument(
        "--blind-audit-root",
        default="",
        help=(
            "fresh isolated benchmark root scanned for answer leakage; defaults "
            "to the source repository"
        ),
    )
    solve.add_argument(
        "--blind-audit-allowed-path",
        action="append",
        default=[],
        help=(
            "exact target-only prior artifact allowed by a known-target reproduction; "
            "the path remains subject to the caller's frozen binding"
        ),
    )
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
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["reasoning_effort"],
    )
    solve.add_argument(
        "--execution-profile",
        choices=("fast", "standard", "proof"),
        default="standard",
        help=(
            "fast is the default and returns a compact two-family architecture; "
            "standard expands breadth; proof permits up to 24-step skeletons "
            "for long-route dossiers"
        ),
    )
    solve.add_argument(
        "--strategy-search-profile",
        choices=("legacy_global", "synthex_matched"),
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["strategy_search_profile"],
        help=(
            "synthex_matched runs three independent compact Codex policy "
            "branches with continuous node expansion"
        ),
    )
    solve.add_argument(
        "--strategy-branches",
        type=int,
        choices=range(1, 9),
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["strategy_branches"],
    )
    solve.add_argument(
        "--node-expansions-per-branch",
        type=int,
        choices=range(1, 65),
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_expansions_per_branch"],
    )
    solve.add_argument(
        "--route-local-repair-rounds",
        type=int,
        choices=range(1, 13),
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["route_local_repair_rounds"],
    )
    solve.add_argument(
        "--max-node-prompt-bytes",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_node_prompt_bytes"],
    )
    solve.add_argument(
        "--node-call-timeout-s",
        type=float,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_call_timeout_s"],
    )
    solve.add_argument(
        "--critic-call-timeout-s",
        type=float,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["critic_call_timeout_s"],
    )
    solve.add_argument(
        "--objective-mode",
        choices=("benchmark_search", "scientific_proof", "procurement_delivery"),
        default=None,
        help=(
            "deprecated compatibility view only; all values run the same "
            "target-blind campaign and differ only in downstream presentation"
        ),
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
    solve.add_argument(
        "--initial-director-web-search",
        action="store_true",
        help=(
            "also give the latency-critical first Codex architecture pass web "
            "tools; normally source connectors prefetch in parallel and web "
            "search is reserved for evidence-informed replanning"
        ),
    )
    solve.add_argument("--no-target-identity", action="store_true")
    solve.add_argument(
        "--no-codex",
        action="store_true",
        help="do not register Codex architecture or replan actions",
    )
    solve.add_argument("--no-replan", action="store_true")
    solve.add_argument(
        "--action-scheduler",
        choices=("adaptive", "round_robin"),
        default="adaptive",
        help="shared target-blind action ordering policy",
    )
    solve.add_argument(
        "--delivery-boundary",
        choices=("full", "stock_result"),
        default="stock_result",
        help=(
            "stock_result (default) closes after the first B4 snapshot; full "
            "continues through the lower-priority C2-C6 credibility work"
        ),
    )
    solve.add_argument("--no-live-benchmark-stock", action="store_true")
    solve.add_argument(
        "--benchmark-stock-index",
        default="",
        help=(
            "content-addressed frozen SQLite benchmark-stock index; this "
            "replaces the default PubChem benchmark-search lookup"
        ),
    )
    solve.add_argument(
        "--benchmark-stock-index-sha256",
        default="",
        help="required expected SHA-256 for --benchmark-stock-index",
    )
    solve.add_argument(
        "--benchmark-stock-name",
        default="",
        help="optional public benchmark stock label recorded in audit artifacts",
    )
    solve.add_argument(
        "--no-chemenzy",
        action="store_true",
        help="disable guided ChemEnzy local expansion",
    )
    solve.add_argument(
        "--target-chemenzy-baseline",
        action=argparse.BooleanOptionalAction,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["target_chemenzy_baseline"],
        help=(
            "run the separate whole-target ChemEnzy baseline arm; disabled "
            "for the matched Codex strategic/stitched arm"
        ),
    )
    solve.add_argument(
        "--chemenzy-env-prefix",
        default="",
        help=(
            "isolated ChemEnzy environment prefix; otherwise use "
            "CHEMENZY_ENV_PREFIX, repository default, then bounded Conda discovery"
        ),
    )
    solve.add_argument(
        "--chemenzy-stock-name",
        action="append",
        default=[],
        help=(
            "explicit ChemEnzy stock name(s) from the vendor config; useful "
            "for benchmark-aligned searches such as RetroStar-stock"
        ),
    )
    solve.add_argument(
        "--chemenzy-stock-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a selected ChemEnzy stock with an explicit CSV or SQLite path",
    )
    solve.add_argument(
        "--chemenzy-provider-route-reserve",
        type=int,
        choices=range(1, 33),
        default=16,
    )
    solve.add_argument(
        "--chemenzy-host-route-portfolio",
        type=int,
        choices=range(1, 17),
        default=16,
    )
    solve.add_argument(
        "--display-route-limit",
        type=int,
        choices=range(1, 13),
        default=4,
    )
    solve.add_argument(
        "--chemenzy-max-routes",
        type=int,
        choices=range(1, 33),
        default=None,
        help="deprecated compatibility alias for --chemenzy-provider-route-reserve",
    )
    solve.add_argument("--chemenzy-max-steps", type=int, default=6)
    solve.add_argument(
        "--chemenzy-iterations",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_iterations"],
    )
    solve.add_argument("--chemenzy-expansion-topk", type=int, default=20)
    solve.add_argument(
        "--chemenzy-timeout-s",
        type=float,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_timeout_s"],
    )
    solve.add_argument(
        "--chemenzy-seed",
        type=int,
        default=0,
        help="explicit ChemEnzy Python/NumPy/Torch seed used for replay binding",
    )
    solve.add_argument("--no-guided-chemenzy", action="store_true")
    solve.add_argument(
        "--guided-chemenzy-frontiers",
        type=int,
        default=None,
        help="optional compatibility cap; default inherits the unified native-search budget",
    )
    solve.add_argument(
        "--guided-chemenzy-iterations",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_iterations"],
    )
    solve.add_argument(
        "--guided-chemenzy-timeout-s",
        type=float,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["short_tail_timeout_s"],
    )
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
        "--no-auto-literature-evidence",
        action="store_true",
        help="disable bounded Crossref/DOI/PDF primary-paper discovery",
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
        "--minimum-planning-route-steps",
        type=int,
        choices=range(0, 25),
        default=0,
        help="require one host-contract-accepted planning skeleton of this depth",
    )
    solve.add_argument(
        "--stock-boundary",
        choices=("benchmark_search", "procurement", "in_house"),
        default="benchmark_search",
    )
    solve.add_argument(
        "--forbidden-reagent",
        action="append",
        default=[],
        help="repeatable reagent name or SMILES forbidden by campaign policy",
    )
    solve.add_argument(
        "--max-route-steps",
        type=int,
        choices=range(1, 65),
        default=None,
        help="hard route-length constraint; independent of search depth",
    )
    solve.add_argument(
        "--allowed-execution-domain",
        action="append",
        choices=(
            "chemical",
            "biocatalytic",
            "whole_cell",
            "hybrid",
            "mechanistic",
        ),
        default=[],
    )
    solve.add_argument(
        "--safety-limit",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="repeatable safety restriction, for example max_temperature_c=120",
    )
    solve.add_argument(
        "--stock-source-id",
        action="append",
        default=[],
        help="repeatable allowed source id within the configured stock oracle",
    )
    solve.add_argument(
        "--max-model-invocations",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_model_invocations"],
    )
    solve.add_argument(
        "--max-input-tokens",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_input_tokens"],
    )
    solve.add_argument(
        "--max-output-tokens",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_output_tokens"],
    )
    solve.add_argument(
        "--max-model-wall-time-s",
        type=float,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_model_wall_time_s"],
    )
    solve.add_argument(
        "--max-prompt-context-bytes",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_prompt_context_bytes"],
    )
    solve.add_argument(
        "--max-visual-invocations",
        type=int,
        choices=(0, 1),
        default=0,
        help="opt in to at most one sparse Codex page-vision call; default is zero",
    )
    solve.add_argument("--max-visual-pages", type=int, choices=range(1, 13), default=6)
    solve.add_argument(
        "--max-accepted-expansions",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_accepted_expansions"],
    )
    solve.add_argument(
        "--max-attempt-runs",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_attempt_runs"],
    )
    solve.add_argument(
        "--max-total-tasks",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_total_tasks"],
    )
    solve.add_argument("--max-evidence-tasks", type=int, default=64)
    solve.add_argument("--max-stock-tasks", type=int, default=128)
    solve.add_argument("--max-validation-tasks", type=int, default=128)
    solve.add_argument("--max-program-tasks", type=int, default=64)
    solve.add_argument("--max-experiment-tasks", type=int, default=32)
    solve.add_argument("--max-run-wall-time-s", type=float, default=7_200.0)
    solve.add_argument(
        "--max-map-reactions",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_atom_mapping_reactions"],
    )
    solve.add_argument(
        "--max-stock-molecules",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_stock_molecules"],
    )
    solve.add_argument("--max-patent-sources", type=int, default=3)
    solve.add_argument("--max-literature-sources", type=int, choices=range(1, 9), default=3)
    solve.add_argument("--max-self-evo-candidates", type=int, default=12)
    solve.add_argument(
        "--patent-publication",
        action="append",
        default=[],
        help="optional patent publication, Google Patents URL, or direct PDF seed",
    )
    solve.add_argument(
        "--literature-doi",
        action="append",
        default=[],
        help="optional DOI seed for reproducible paper-route extraction",
    )
    solve.add_argument(
        "--literature-pdf",
        action="append",
        default=[],
        help="optional local primary-paper/SI PDF seed",
    )

    validation_fork = sub.add_parser(
        "fork-validation",
        help=(
            "replay a completed target campaign into a model-free-by-default "
            "run, optionally use one sparse page-vision call, and revalidate "
            "it under the current host policy"
        ),
    )
    validation_fork.add_argument("source_run_id")
    validation_fork.add_argument("--source-run-dir")
    validation_fork.add_argument("--run-id")
    validation_fork.add_argument("--run-dir")
    validation_fork.add_argument("--no-live-benchmark-stock", action="store_true")
    validation_fork.add_argument(
        "--benchmark-stock-index",
        default="",
        help="frozen SQLite stock index reused by the model-free fork",
    )
    validation_fork.add_argument(
        "--benchmark-stock-index-sha256",
        default="",
        help="required expected SHA-256 for --benchmark-stock-index",
    )
    validation_fork.add_argument(
        "--benchmark-stock-name",
        default="",
        help="public label for the frozen benchmark stock",
    )
    validation_fork.add_argument(
        "--guided-chemenzy",
        action="store_true",
        help=(
            "after the model-free plan replay, run one bounded ChemEnzy "
            "short-tail search for each distinct stock-open leaf"
        ),
    )
    validation_fork.add_argument("--chemenzy-env-prefix", default="")
    validation_fork.add_argument(
        "--chemenzy-stock-name", action="append", default=[]
    )
    validation_fork.add_argument(
        "--chemenzy-stock-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    validation_fork.add_argument(
        "--guided-chemenzy-frontiers", type=int, default=4
    )
    validation_fork.add_argument(
        "--guided-chemenzy-routes", type=int, default=1
    )
    validation_fork.add_argument(
        "--guided-chemenzy-steps", type=int, default=6
    )
    validation_fork.add_argument(
        "--guided-chemenzy-iterations", type=int, default=500
    )
    validation_fork.add_argument(
        "--guided-chemenzy-expansion-topk", type=int, default=20
    )
    validation_fork.add_argument(
        "--guided-chemenzy-timeout-s", type=float, default=1_200.0
    )
    validation_fork.add_argument("--chemenzy-seed", type=int, default=0)
    validation_fork.add_argument("--no-auto-patent-evidence", action="store_true")
    validation_fork.add_argument("--no-auto-literature-evidence", action="store_true")
    validation_fork.add_argument("--full-output", action="store_true")
    validation_fork.add_argument("--max-map-reactions", type=int, default=64)
    validation_fork.add_argument("--max-stock-molecules", type=int, default=32)
    validation_fork.add_argument("--max-patent-sources", type=int, default=3)
    validation_fork.add_argument("--no-patent-self-evo", action="store_true")
    validation_fork.add_argument("--self-evo-library", default="")
    validation_fork.add_argument("--max-self-evo-candidates", type=int, default=12)
    validation_fork.add_argument(
        "--max-visual-invocations", type=int, choices=(0, 1), default=0
    )
    validation_fork.add_argument(
        "--max-visual-pages", type=int, choices=range(1, 13), default=2
    )
    validation_fork.add_argument(
        "--visual-model", default=DEFAULT_TARGET_DIRECTOR_MODEL
    )
    validation_fork.add_argument(
        "--visual-reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    validation_fork.add_argument(
        "--max-literature-sources", type=int, choices=range(1, 9), default=3
    )
    validation_fork.add_argument(
        "--patent-publication",
        action="append",
        default=[],
        help="optional patent publication, Google Patents URL, or direct PDF seed",
    )
    validation_fork.add_argument(
        "--literature-doi",
        action="append",
        default=[],
        help="optional DOI discovered by the source frontier for reproducible XML-first validation",
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
        evidence_connectors = []
        if not args.no_auto_patent_evidence:
            from cascade_planner.interfaces.patent_evidence import (
                BuiltinPatentEvidenceConfig,
                build_builtin_patent_evidence_connector,
            )

            evidence_connectors.append(
                build_builtin_patent_evidence_connector(
                    BuiltinPatentEvidenceConfig(
                        cache_dir=gateway.paths.external_data_root / "patent-evidence",
                        seed_publications=tuple(args.patent_publication),
                        max_patents=args.max_patent_sources,
                        max_validated_edges=args.max_map_reactions,
                    )
                )
            )
        if not args.no_auto_literature_evidence:
            from cascade_planner.interfaces.literature_evidence import (
                BuiltinLiteratureEvidenceConfig,
                build_builtin_literature_evidence_connector,
            )

            evidence_connectors.append(
                build_builtin_literature_evidence_connector(
                    BuiltinLiteratureEvidenceConfig(
                        cache_dir=(
                            gateway.paths.external_data_root / "literature-evidence"
                        ),
                        seed_dois=tuple(args.literature_doi),
                        max_sources=args.max_literature_sources,
                        auto_fetch_restricted_sources=True,
                        auto_fetch_max_items=args.max_literature_sources,
                    )
                )
            )
        evidence_connector = (
            None
            if not evidence_connectors
            else (
                evidence_connectors[0]
                if len(evidence_connectors) == 1
                else compose_evidence_connectors(*evidence_connectors)
            )
        )
        visual_evidence_provider = None
        if args.max_visual_invocations > 0:
            from cascade_planner.interfaces.visual_evidence import (
                CodexVisualEvidenceConfig,
                build_codex_visual_evidence_provider,
            )

            visual_evidence_provider = build_codex_visual_evidence_provider(
                CodexVisualEvidenceConfig(
                    cache_dir=gateway.paths.external_data_root / "visual-evidence",
                    model=args.visual_model,
                    reasoning_effort=args.visual_reasoning_effort,
                    max_pages=args.max_visual_pages,
                )
            )
        stock_catalog_builder = None
        if args.benchmark_stock_index:
            if args.no_live_benchmark_stock:
                raise ValueError(
                    "benchmark_stock_index_conflicts_with_no_live_benchmark_stock"
                )
            stock_catalog_builder = FrozenBenchmarkStockIndex(
                args.benchmark_stock_index,
                expected_sha256=args.benchmark_stock_index_sha256,
                catalog_name=args.benchmark_stock_name,
            )
        elif args.benchmark_stock_index_sha256 or args.benchmark_stock_name:
            raise ValueError("benchmark_stock_index_path_required")
        chemenzy_stock_names, chemenzy_stock_paths = _resolve_chemenzy_stock_binding(
            stock_names=tuple(
                str(value)
                for value in args.chemenzy_stock_name
                if str(value).strip()
            ),
            stock_paths=_parse_chemenzy_stock_paths(args.chemenzy_stock_path),
            benchmark_stock_index=args.benchmark_stock_index,
            benchmark_stock_name=args.benchmark_stock_name,
            chemenzy_enabled=args.guided_chemenzy,
        )
        result = gateway.fork_target_validation(
            source_run_id=args.source_run_id,
            source_run_dir=args.source_run_dir,
            run_id=args.run_id,
            run_dir=args.run_dir,
            evidence_connector=evidence_connector,
            visual_evidence_provider=visual_evidence_provider,
            stock_catalog_builder=stock_catalog_builder,
            config=ValidationForkConfig(
                max_atom_mapping_reactions=args.max_map_reactions,
                max_live_stock_molecules=args.max_stock_molecules,
                enable_live_benchmark_stock=not args.no_live_benchmark_stock,
                enable_patent_self_evolution=not args.no_patent_self_evo,
                self_evo_library_path=args.self_evo_library,
                max_self_evo_template_candidates=args.max_self_evo_candidates,
                max_visual_invocations=args.max_visual_invocations,
                max_visual_evidence_pages=args.max_visual_pages,
                enable_guided_chemenzy=args.guided_chemenzy,
                chemenzy_env_prefix=args.chemenzy_env_prefix,
                chemenzy_stock_names=chemenzy_stock_names,
                chemenzy_stock_paths=chemenzy_stock_paths,
                max_guided_chemenzy_frontiers=args.guided_chemenzy_frontiers,
                max_guided_chemenzy_routes=args.guided_chemenzy_routes,
                max_guided_chemenzy_steps=args.guided_chemenzy_steps,
                max_guided_chemenzy_iterations=args.guided_chemenzy_iterations,
                guided_chemenzy_expansion_topk=(
                    args.guided_chemenzy_expansion_topk
                ),
                guided_chemenzy_timeout_s=args.guided_chemenzy_timeout_s,
                chemenzy_seed=args.chemenzy_seed,
            ),
        )
        return result if args.full_output else _compact_validation_fork_result(result)
    if args.command != "solve-target":
        raise ValueError(f"unsupported_target_command:{args.command}")
    objective_compatibility_view = _resolve_objective_compatibility_view(
        args.objective_mode
    )
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
    else:
        builtin_connectors = []
        if not args.no_auto_patent_evidence:
            from cascade_planner.interfaces.patent_evidence import (
                BuiltinPatentEvidenceConfig,
                build_builtin_patent_evidence_connector,
            )

            builtin_connectors.append(
                build_builtin_patent_evidence_connector(
                    BuiltinPatentEvidenceConfig(
                        cache_dir=gateway.paths.external_data_root / "patent-evidence",
                        seed_publications=tuple(args.patent_publication),
                        max_patents=args.max_patent_sources,
                        max_validated_edges=args.max_map_reactions,
                    )
                )
            )
        if not args.no_auto_literature_evidence:
            from cascade_planner.interfaces.literature_evidence import (
                BuiltinLiteratureEvidenceConfig,
                build_builtin_literature_evidence_connector,
            )

            builtin_connectors.append(
                build_builtin_literature_evidence_connector(
                    BuiltinLiteratureEvidenceConfig(
                        cache_dir=gateway.paths.external_data_root / "literature-evidence",
                        seed_dois=tuple(args.literature_doi),
                        seed_pdfs=tuple(args.literature_pdf),
                        max_sources=args.max_literature_sources,
                        max_visual_pages=args.max_visual_pages,
                        auto_fetch_restricted_sources=True,
                        auto_fetch_max_items=args.max_literature_sources,
                    )
                )
            )
        if builtin_connectors:
            evidence_connector = (
                builtin_connectors[0]
                if len(builtin_connectors) == 1
                else compose_evidence_connectors(*builtin_connectors)
            )
    stock_catalog_builder = None
    if args.benchmark_stock_index:
        if args.no_live_benchmark_stock:
            raise ValueError(
                "benchmark_stock_index_conflicts_with_no_live_benchmark_stock"
            )
        if args.stock_boundary != "benchmark_search":
            raise ValueError(
                "benchmark_stock_index_requires_benchmark_search_boundary"
            )
        stock_catalog_builder = FrozenBenchmarkStockIndex(
            args.benchmark_stock_index,
            expected_sha256=args.benchmark_stock_index_sha256,
            catalog_name=args.benchmark_stock_name,
        )
    elif args.benchmark_stock_index_sha256 or args.benchmark_stock_name:
        raise ValueError("benchmark_stock_index_path_required")

    inventory_snapshot_builder = None
    if args.inventory_snapshot:
        if args.stock_boundary != "procurement":
            raise ValueError("inventory_snapshot_requires_procurement_boundary")
        inventory_snapshot_builder = FrozenInventorySnapshotBuilder(
            args.inventory_snapshot
        )

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

    chemenzy_stock_names, chemenzy_stock_paths = _resolve_chemenzy_stock_binding(
        stock_names=tuple(
            str(value) for value in args.chemenzy_stock_name if str(value).strip()
        ),
        stock_paths=_parse_chemenzy_stock_paths(args.chemenzy_stock_path),
        benchmark_stock_index=args.benchmark_stock_index,
        benchmark_stock_name=args.benchmark_stock_name,
        chemenzy_enabled=not args.no_chemenzy,
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
        stock_catalog_builder=stock_catalog_builder,
        inventory_snapshot_builder=inventory_snapshot_builder,
        constraints=TargetConstraints(
            forbidden_reagents=tuple(args.forbidden_reagent),
            max_route_steps=args.max_route_steps,
            allowed_execution_domains=(
                tuple(args.allowed_execution_domain)
                if args.allowed_execution_domain
                else TargetConstraints().allowed_execution_domains
            ),
            safety_limits=_parse_safety_limits(args.safety_limit),
            stock_source_ids=tuple(args.stock_source_id),
        ),
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
            max_prompt_context_bytes=args.max_prompt_context_bytes,
        ),
        config=TargetSolveConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            execution_profile=args.execution_profile,
            strategy_search_profile=args.strategy_search_profile,
            strategy_branch_count=args.strategy_branches,
            max_node_expansions_per_branch=args.node_expansions_per_branch,
            max_route_local_repair_rounds=args.route_local_repair_rounds,
            max_node_prompt_bytes=args.max_node_prompt_bytes,
            max_node_call_timeout_s=args.node_call_timeout_s,
            critic_call_timeout_s=args.critic_call_timeout_s,
            objective_mode=objective_compatibility_view,
            use_coordinator=args.coordinator and not args.single_agent,
            enable_web_search=not args.no_web_search,
            enable_initial_director_web_search=(
                args.initial_director_web_search and not args.no_web_search
            ),
            enable_codex=not args.no_codex,
            enable_target_identity=not args.no_target_identity,
            resolve_named_target_identity=not args.no_target_identity,
            blind_audit_root=args.blind_audit_root,
            blind_audit_allowed_paths=tuple(args.blind_audit_allowed_path),
            enable_replan=not args.no_replan,
            action_scheduler_policy=args.action_scheduler,
            delivery_boundary=args.delivery_boundary,
            enable_live_benchmark_stock=not args.no_live_benchmark_stock,
            enable_chemenzy=not args.no_chemenzy,
            enable_target_chemenzy_baseline=args.target_chemenzy_baseline,
            enable_guided_chemenzy=not args.no_guided_chemenzy,
            chemenzy_env_prefix=args.chemenzy_env_prefix,
            chemenzy_stock_names=chemenzy_stock_names,
            chemenzy_stock_paths=chemenzy_stock_paths,
            enable_patent_self_evolution=not args.no_patent_self_evo,
            self_evo_library_path=args.self_evo_library,
            enable_builtin_patent_evidence=(
                evidence_connector is None and not args.no_auto_patent_evidence
            ),
            max_atom_mapping_reactions=args.max_map_reactions,
            max_live_stock_molecules=args.max_stock_molecules,
            max_patent_sources=args.max_patent_sources,
            max_self_evo_template_candidates=args.max_self_evo_candidates,
            max_total_tasks=args.max_total_tasks,
            max_evidence_tasks=args.max_evidence_tasks,
            max_stock_tasks=args.max_stock_tasks,
            max_validation_tasks=args.max_validation_tasks,
            max_program_tasks=args.max_program_tasks,
            max_experiment_tasks=args.max_experiment_tasks,
            max_run_wall_time_s=args.max_run_wall_time_s,
            provider_route_reserve=args.chemenzy_provider_route_reserve,
            host_route_portfolio=args.chemenzy_host_route_portfolio,
            display_route_limit=args.display_route_limit,
            max_chemenzy_routes=args.chemenzy_max_routes,
            max_chemenzy_steps=args.chemenzy_max_steps,
            max_chemenzy_iterations=args.chemenzy_iterations,
            chemenzy_expansion_topk=args.chemenzy_expansion_topk,
            chemenzy_timeout_s=args.chemenzy_timeout_s,
            chemenzy_seed=args.chemenzy_seed,
            max_guided_chemenzy_frontiers=args.guided_chemenzy_frontiers,
            max_guided_chemenzy_iterations=args.guided_chemenzy_iterations,
            guided_chemenzy_timeout_s=args.guided_chemenzy_timeout_s,
            max_visual_evidence_pages=args.max_visual_pages,
            minimum_planning_route_steps=args.minimum_planning_route_steps,
            max_director_wall_time_s=args.max_model_wall_time_s,
        ),
    )
    return result if args.full_output else _compact_target_result(result)


def _resolve_objective_compatibility_view(value: str | None) -> str:
    """Resolve the legacy CLI view without granting it control-flow authority."""

    if value is None:
        return "scientific_proof"
    warnings.warn(
        "--objective-mode is deprecated and is retained only as output "
        "compatibility metadata; configure stock, acceptance and budgets "
        "directly instead",
        FutureWarning,
        stacklevel=2,
    )
    return str(value)


def _parse_chemenzy_stock_paths(values: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in values:
        name, separator, path = str(raw).partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError("chemenzy_stock_path_must_be_NAME_EQUALS_PATH")
        if name in seen:
            raise ValueError(f"duplicate_chemenzy_stock_path:{name}")
        seen.add(name)
        parsed.append((name, path))
    return tuple(parsed)


def _resolve_chemenzy_stock_binding(
    *,
    stock_names: tuple[str, ...],
    stock_paths: tuple[tuple[str, str], ...],
    benchmark_stock_index: str,
    benchmark_stock_name: str,
    chemenzy_enabled: bool,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Keep provider search and host scoring on the same benchmark stock.

    A benchmark index is a search boundary, not merely a post-hoc scoring
    oracle.  Falling back to ChemEnzy's vendor default here makes the provider
    optimize against a different terminal set than the host uses for B4.
    Explicit stock bindings remain supported, but benchmark runs must bind
    exactly the same index.
    """

    names = tuple(
        dict.fromkeys(
            str(value).strip() for value in stock_names if str(value).strip()
        )
    )
    paths = tuple(
        (str(name).strip(), str(path).strip()) for name, path in stock_paths
    )
    if not chemenzy_enabled or not str(benchmark_stock_index or "").strip():
        if paths and not names:
            names = tuple(name for name, _path in paths)
        return names, paths

    benchmark_path = Path(benchmark_stock_index).expanduser().resolve()
    if not names and not paths:
        aligned_name = str(benchmark_stock_name or "").strip() or "Benchmark-stock"
        return (aligned_name,), ((aligned_name, str(benchmark_path)),)

    if not paths:
        raise ValueError("benchmark_stock_requires_chemenzy_path_alignment")
    path_map = {name: Path(path).expanduser().resolve() for name, path in paths}
    if not names:
        names = tuple(path_map)
    if set(names) != set(path_map) or len(path_map) != 1:
        raise ValueError("benchmark_stock_requires_one_matching_chemenzy_stock")
    if next(iter(path_map.values())) != benchmark_path:
        raise ValueError("benchmark_and_chemenzy_stock_paths_differ")
    return names, tuple((name, str(path_map[name])) for name in names)


def _parse_safety_limits(values: list[str] | tuple[str, ...]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values:
        key, separator, value = str(raw).partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError("safety_limit_must_be_KEY_EQUALS_JSON_VALUE")
        if key in parsed:
            raise ValueError(f"duplicate_safety_limit:{key}")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def _compact_target_result(result: Any) -> dict[str, Any]:
    row = dict(result)
    gates = dict(row.get("gates") or {})
    resource = dict(row.get("resource_envelope") or {})
    campaign_spec = dict(row.get("campaign_spec") or {})
    stock_oracle = dict(campaign_spec.get("stock_oracle") or {})
    stock_binding = dict(stock_oracle.get("binding") or {})
    return {
        "schema_version": "target_solve_cli_summary.v1",
        "run_id": str(row.get("run_id") or ""),
        "target": dict(row.get("target") or {}),
        "campaign_contract": {
            "content_sha256": str(campaign_spec.get("content_sha256") or ""),
            "stock_oracle_id": str(stock_oracle.get("oracle_id") or ""),
            "stock_oracle_binding_sha256": str(
                stock_binding.get("content_sha256") or ""
            ),
            "constraints": dict(campaign_spec.get("constraints") or {}),
            "resource_budget": dict(
                campaign_spec.get("resource_budget") or {}
            ),
        },
        "gates": dict(gates.get("gates") or {}),
        "highest_contiguous_gate": str(
            gates.get("highest_contiguous_gate") or "none"
        ),
        "counts": dict(gates.get("counts") or {}),
        "paper_equivalent": dict(row.get("paper_equivalent") or {}),
        "quality_state": dict(row.get("quality_state") or {}),
        "claim": dict(row.get("claim") or {}),
        "planning_depth": dict(row.get("planning_depth") or {}),
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
            "task_budget": dict(resource.get("task_budget") or {}),
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
