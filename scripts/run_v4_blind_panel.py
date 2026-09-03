#!/usr/bin/env python3
"""Run a target-only V4 manifest in one fresh, isolated external workspace."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock, get_ident
import time
from typing import Any, Mapping

from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    BlindCase,
    audit_blind_preflight,
    canonical_smiles,
    load_blind_manifest,
)
from cascade_planner.application.campaign_trajectory import (  # noqa: E402
    TRAJECTORY_CUTOFF_PROJECTION_SCHEMA,
    project_campaign_trajectory_at_cutoff,
)
from cascade_planner.interfaces.target_runtime_dependencies import (  # noqa: E402
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
)
from cascade_planner.interfaces.live_stock import (  # noqa: E402
    STANDARD_STOCK_CATALOG_NAME,
    STANDARD_STOCK_INDEX_RELATIVE_PATH,
)
from cascade_planner.interfaces.target_solver import _is_paper_reach_profile  # noqa: E402
from cascade_planner.runtime.paths import RuntimePaths  # noqa: E402
from cascade_planner.runtime.run_registry_catalog import (  # noqa: E402
    RunRegistryCatalog,
    binding_from_registry_root,
    registry_catalog_path,
)
from cascade_planner.eval.synthex_protocol_preflight import (  # noqa: E402
    validate_synthex_head_to_head_protocol,
)
from scripts.compare_v4_matched_panels import (  # noqa: E402
    _markdown as _comparison_markdown,
    compare_matched_panels,
)
from scripts.summarize_v4_blind_panel import (  # noqa: E402
    _hydrate_report_diagnostics,
    _markdown as _summary_markdown,
    summarize_panel,
)

PANEL_REASONING_EFFORTS = ("low", "medium", "high")
PANEL_EXECUTION_PROFILES = (
    "fast",
    "standard",
    "proof",
    "paper_synthex",
    "paper_matched_reach",
    "self_correcting_sequential",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--publish-registry",
        action="store_true",
        help=(
            "Explicitly publish this isolated panel registry to the Web catalog; "
            "no results-directory scan is performed."
        ),
    )
    parser.add_argument(
        "--registry-catalog-path",
        default="",
        help="Optional catalog SQLite path; defaults to the main runtime catalog.",
    )
    parser.add_argument("--registry-id", default="")
    parser.add_argument("--registry-label", default="")
    parser.add_argument("--registry-project-id", default="")
    parser.add_argument("--registry-project-label", default="")
    parser.add_argument(
        "--paper-protocol",
        help=(
            "Optional frozen SynthEx head-to-head protocol. When supplied, "
            "its complete three-target, runtime-budget and exact-stock contract "
            "is validated before any provider is started."
        ),
    )
    parser.add_argument(
        "--model", default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["model"]
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=PANEL_REASONING_EFFORTS,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["reasoning_effort"],
    )
    parser.add_argument(
        "--execution-profile",
        choices=PANEL_EXECUTION_PROFILES,
        default="paper_matched_reach",
        help=(
            "execution contract; paper_matched_reach is the frozen paper-control "
            "arm, while self_correcting_sequential adds sparse online review"
        ),
    )
    parser.add_argument(
        "--strategy-portfolio-mode",
        choices=(
            "auto",
            "paper_independent",
            "autoplanner_hybrid",
            "enzyme_advantage",
            "autoplanner_strategy_v2",
        ),
        default="auto",
        help=(
            "strategy arm passed to every target; enzyme_advantage is a "
            "separate companion panel and cannot be labeled as the paper arm"
        ),
    )
    parser.add_argument(
        "--strategy-portfolio-seed",
        default="",
        help=(
            "completed Strategy-screen JSON promoted into Builder execution; "
            "labels the panel as known-strategy reproduction"
        ),
    )
    parser.add_argument(
        "--strategic-milestones-per-branch",
        type=int,
        choices=range(1, 5),
        default=1,
        help=(
            "ordered route-internal StrategyCards per branch; 1 freezes the "
            "paper single-anchor control"
        ),
    )
    parser.add_argument(
        "--node-expansions-per-branch",
        type=int,
        choices=range(1, 65),
        default=None,
        help=(
            "Diagnostic override for Route Builder calls per branch; omitted "
            "preserves the frozen paper-matched allowance."
        ),
    )
    parser.add_argument(
        "--reactionjson-candidates-per-node",
        type=int,
        choices=range(1, 9),
        default=None,
        help=(
            "ReactionJSON candidate width for a non-paper ablation; paper_synthex "
            "keeps the frozen width=1 contract"
        ),
    )
    parser.add_argument(
        "--strategy-tree-engine",
        choices=("auto", "chemenzy_best_first", "aizynthfinder_mcts"),
        default=None,
        help=(
            "Route Builder tree owner; explicitly set aizynthfinder_mcts for "
            "a matched AiZ MCTS control outside paper_synthex."
        ),
    )
    parser.add_argument(
        "--no-chemenzy",
        action="store_true",
        default=False,
        help="disable optional ChemEnzy services and target-level baselines",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("benchmark_search", "scientific_proof", "procurement_delivery"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-cutoff-wall-time-s",
        type=float,
        default=None,
        help=(
            "Fixed target wall-time cutoff. For paper_synthex the default is "
            "an operational 24 h emergency ceiling; scientific limits are the "
            "per-call, invocation, expansion, and repair budgets."
        ),
    )
    parser.add_argument(
        "--fixed-cutoff-total-tasks",
        type=int,
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_total_tasks"],
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help=(
            "Run only the first N manifest-ordered targets after --only filtering. "
            "The selected case IDs are frozen into the benchmark snapshot."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matched-baseline-summary",
        help=(
            "Optional frozen v4_blind_panel_summary JSON. After this panel "
            "finishes, compare the selected case set and publish matched-comparison."
        ),
    )
    parser.add_argument("--visual", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Freeze the panel and audit every selected target without starting providers.",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "baseline",
            "no-chemenzy",
            "no-self-evo",
            "no-replan",
            "chemenzy-only",
            "codex-only",
            "unified-round-robin",
            "unified-adaptive",
        ),
        default="baseline",
        help="Run one frozen evaluation arm without target-specific changes.",
    )
    parser.add_argument(
        "--self-evo-library-seed",
        help=(
            "Optional replay-gated template library frozen once and copied into "
            "every case-local external root."
        ),
    )
    parser.add_argument(
        "--inventory-snapshot",
        help="Optional versioned inventory snapshot for procurement-bound cases.",
    )
    parser.add_argument(
        "--benchmark-stock-index",
        help=(
            "explicit frozen SQLite stock override shared read-only by "
            "benchmark_search cases; omitted uses the standard ZINC+eMolecules "
            "full-InChIKey index"
        ),
    )
    parser.add_argument(
        "--benchmark-stock-name",
        default="",
        help="Public label recorded for --benchmark-stock-index.",
    )
    parser.add_argument(
        "--leakage-audit-pack",
        help=(
            "Evaluator-only synonyms and key intermediates used only by the "
            "supervisor preflight; never passed to the planner subprocess."
        ),
    )
    parser.add_argument(
        "--allowed-prior-target-manifest",
        action="append",
        default=[],
        help=(
            "Previously frozen target-only blind manifest allowed during a known-target "
            "reproduction. The file is schema-validated, content-bound in the panel "
            "snapshot, and never passed to the planner as route knowledge."
        ),
    )
    parser.add_argument(
        "--chemenzy-env-prefix",
        default=None,
        help=(
            "Explicit host-compatible ChemEnzy Python prefix. Passed through to "
            "every isolated target run and recorded in panel status."
        ),
    )
    parser.add_argument(
        "--chemenzy-stock-name",
        action="append",
        default=[],
        help="explicit ChemEnzy vendor stock name(s), e.g. RetroStar-stock",
    )
    parser.add_argument(
        "--chemenzy-stock-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a selected ChemEnzy stock with an explicit CSV or SQLite path",
    )
    args = parser.parse_args(argv)
    if args.objective_mode is not None:
        print(
            "warning: --objective-mode is ignored by the blind panel; "
            "scores are read-only fixed-cutoff trajectory projections",
            file=sys.stderr,
        )
    try:
        fixed_cutoff_wall_time_s = _resolve_panel_fixed_cutoff_wall_time_s(
            execution_profile=args.execution_profile,
            requested=args.fixed_cutoff_wall_time_s,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.fixed_cutoff_total_tasks <= 0:
        raise SystemExit("--fixed-cutoff-total-tasks must be positive")
    projection_policy = {
        "schema_version": "campaign_fixed_cutoff_policy.v1",
        "wall_time_s": fixed_cutoff_wall_time_s,
        "settled_task_count": int(args.fixed_cutoff_total_tasks),
        "case_budget_dimensions_are_applied": True,
    }

    manifest = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_cases = list(load_blind_manifest(manifest))
    cases = _select_cases(
        manifest_cases,
        only=args.only,
        max_targets=args.max_targets,
    )
    if not cases:
        raise SystemExit("No benchmark cases selected")
    if (
        not str(args.benchmark_stock_index or "").strip()
        and any(
            str(case.acceptance.get("stock_boundary") or "")
            == "benchmark_search"
            for case in cases
        )
    ):
        args.benchmark_stock_index = str(
            (ROOT / STANDARD_STOCK_INDEX_RELATIVE_PATH).resolve()
        )
        args.benchmark_stock_name = STANDARD_STOCK_CATALOG_NAME
    try:
        strategy_portfolio_seed = _validated_strategy_portfolio_seed(
            args.strategy_portfolio_seed,
            cases=cases,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    paper_protocol_preflight: dict[str, Any] = {}
    if args.paper_protocol:
        if (
            args.node_expansions_per_branch is not None
            and args.node_expansions_per_branch
            != int(SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_expansions_per_branch"])
        ):
            raise SystemExit(
                "paper_protocol_forbids_node_expansion_budget_override"
            )
        if args.only or args.max_targets is not None or len(cases) != len(manifest_cases):
            raise SystemExit("paper_protocol_requires_full_frozen_manifest")
        paper_protocol_preflight = validate_synthex_head_to_head_protocol(
            protocol_path=args.paper_protocol,
            manifest_path=manifest,
            repository_root=ROOT,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            execution_profile=args.execution_profile,
            strategy_portfolio_mode=args.strategy_portfolio_mode,
            benchmark_stock_index=args.benchmark_stock_index,
            benchmark_stock_name=args.benchmark_stock_name,
        )
        if paper_protocol_preflight.get("ready_for_paid_experiment") is not True:
            issue_codes = ",".join(
                str(row.get("code") or "unknown")
                for row in paper_protocol_preflight.get("issues") or []
                if isinstance(row, Mapping)
            )
            raise SystemExit(f"paper_protocol_preflight_failed:{issue_codes}")
        protocol_stock = dict(paper_protocol_preflight.get("stock") or {})
        args.benchmark_stock_index = str(protocol_stock.get("index_path") or "")
        args.benchmark_stock_name = str(protocol_stock.get("catalog_name") or "")
    matched_baseline_path = _optional_file(
        args.matched_baseline_summary,
        "matched_baseline_summary",
    )
    matched_baseline = (
        json.loads(matched_baseline_path.read_text(encoding="utf-8"))
        if matched_baseline_path is not None
        else None
    )
    if matched_baseline is not None:
        _validate_matched_baseline(matched_baseline, cases=cases)
    allowed_prior_target_manifests = _prior_target_manifest_files(
        args.allowed_prior_target_manifest
    )
    if (
        _strategy_tree_requires_benchmark_stock_index(
            execution_profile=args.execution_profile,
            strategy_tree_engine=args.strategy_tree_engine,
        )
        and not str(args.benchmark_stock_index or "").strip()
    ):
        raise SystemExit(
            "aizynthfinder_strategy_stock_index_required_before_panel_preflight"
        )

    try:
        chemenzy_stock_names, chemenzy_stock_paths = _resolve_panel_chemenzy_stock_binding(
            benchmark_stock_index=args.benchmark_stock_index,
            benchmark_stock_name=args.benchmark_stock_name,
            stock_names=tuple(args.chemenzy_stock_name),
            stock_paths=tuple(args.chemenzy_stock_path),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for name in ("audit", "runtime", "runs", "artifacts", "external", "logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    registry_publication = (
        _publish_run_registry(
            output_root=output_root,
            cases=cases,
            catalog_path=args.registry_catalog_path,
            registry_id=args.registry_id,
            registry_label=args.registry_label,
            project_id=args.registry_project_id,
            project_label=args.registry_project_label,
        )
        if args.publish_registry
        else {}
    )
    snapshot = _prepare_panel_snapshot(
        output_root=output_root,
        manifest=manifest,
        cases=cases,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        execution_profile=args.execution_profile,
        strategy_portfolio_mode=args.strategy_portfolio_mode,
        strategy_portfolio_seed=(
            str(strategy_portfolio_seed) if strategy_portfolio_seed else None
        ),
        projection_policy=projection_policy,
        worker_count=args.workers,
        visual=args.visual,
        ablation=args.ablation,
        chemenzy_env_prefix=args.chemenzy_env_prefix,
        chemenzy_stock_names=chemenzy_stock_names,
        chemenzy_stock_paths=chemenzy_stock_paths,
        self_evo_library_seed=args.self_evo_library_seed,
        inventory_snapshot=args.inventory_snapshot,
        benchmark_stock_index=args.benchmark_stock_index,
        benchmark_stock_name=args.benchmark_stock_name,
        paper_protocol=args.paper_protocol,
        leakage_audit_pack=args.leakage_audit_pack,
        allowed_prior_target_manifests=allowed_prior_target_manifests,
        resume=args.resume,
    )
    status_path = output_root / "panel-status.json"
    previous_state = _load_resume_panel_state(status_path) if args.resume else {}
    state_lock = Lock()
    state: dict[str, Any] = {
        "schema_version": "v4_blind_panel_status.v1",
        "manifest_path": str(manifest),
        "output_root": str(output_root),
        "registry_publication": registry_publication,
        "started_at": _utc_now(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "execution_profile": args.execution_profile,
        "strategy_portfolio_mode": args.strategy_portfolio_mode,
        "strategy_portfolio_seed": (
            {
                "path": str(strategy_portfolio_seed),
                "file_sha256": _file_sha256(strategy_portfolio_seed),
                "semantics": "known_strategy_reproduction_not_blind_discovery",
            }
            if strategy_portfolio_seed
            else {}
        ),
        "strategic_milestones_per_branch": (
            args.strategic_milestones_per_branch
        ),
        "node_expansions_per_branch": (
            args.node_expansions_per_branch
            if args.node_expansions_per_branch is not None
            else SYNTHEX_MATCHED_PROFILE_DEFAULTS["node_expansions_per_branch"]
        ),
        "reactionjson_candidates_per_node": (
            args.reactionjson_candidates_per_node
            if args.reactionjson_candidates_per_node is not None
            else SYNTHEX_MATCHED_PROFILE_DEFAULTS["reactionjson_candidates_per_node"]
        ),
        "no_chemenzy": bool(args.no_chemenzy),
        "leaf_continuation_engine": "same_llm_route_builder",
        "strategy_tree_engine": (
            args.strategy_tree_engine
            if args.strategy_tree_engine is not None
            else (
                "aizynthfinder_mcts"
                if _is_paper_reach_profile(args.execution_profile)
                else "auto"
            )
        ),
        "fixed_cutoff_policy": projection_policy,
        "ignored_legacy_objective_mode": str(args.objective_mode or ""),
        "ablation": args.ablation,
        "worker_count": args.workers,
        "chemenzy_env_prefix": str(args.chemenzy_env_prefix or ""),
        "chemenzy_stock_names": list(chemenzy_stock_names),
        "chemenzy_stock_paths": list(chemenzy_stock_paths),
        "chemenzy_stock_alignment": {
            "benchmark_index_bound_to_provider": bool(args.benchmark_stock_index),
            "provider_and_host_search_boundary_equal": bool(
                not _is_paper_reach_profile(args.execution_profile)
                and args.benchmark_stock_index
                and chemenzy_stock_paths
            ),
        },
        "canonical_stock_binding": {
            "owner": "host_exact_stock_oracle",
            "explicit_frozen_index": bool(args.benchmark_stock_index),
        },
        "paper_protocol_preflight": paper_protocol_preflight,
        "matched_baseline": (
            {
                "path": str(matched_baseline_path),
                "file_sha256": _file_sha256(matched_baseline_path),
                "content_sha256": str(
                    dict(matched_baseline).get("content_sha256") or ""
                ),
                "selected_case_ids": [case.case_id for case in cases],
            }
            if matched_baseline_path is not None
            and isinstance(matched_baseline, Mapping)
            else {}
        ),
        "frozen_snapshot": {
            "path": str(output_root / "snapshots" / "benchmark-snapshot.json"),
            "content_sha256": str(snapshot.get("content_sha256") or ""),
            "base_environment_sha256": str(
                snapshot.get("base_environment_sha256") or ""
            ),
            "manifest_sha256": str(snapshot.get("manifest_sha256") or ""),
            "self_evo_library_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("self_evo_library_sha256")
                or ""
            ),
            "inventory_snapshot_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("inventory_snapshot_sha256")
                or ""
            ),
            "benchmark_stock_index_sha256": str(
                dict(snapshot.get("knowledge") or {}).get(
                    "benchmark_stock_index_sha256"
                )
                or ""
            ),
            "leakage_audit_pack_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("leakage_audit_pack_sha256")
                or ""
            ),
            "allowed_prior_target_manifests": list(
                dict(snapshot.get("knowledge") or {}).get(
                    "allowed_prior_target_manifests"
                )
                or []
            ),
        },
        "target_count": len(cases),
        "selection": {
            "manifest_target_count": len(manifest_cases),
            "selected_target_count": len(cases),
            "max_targets": args.max_targets,
            "only": sorted(str(value) for value in args.only),
            "selected_case_ids": [case.case_id for case in cases],
            "manifest_order_preserved": True,
        },
        "targets": {
            case.target_name: {"status": "queued", "case_id": case.case_id}
            for case in cases
        },
        "semantics": {
            "target_name_and_smiles_only": not bool(strategy_portfolio_seed),
            "no_local_pdf_doi_patent_or_route_seed": not bool(
                strategy_portfolio_seed
            ),
            "isolated_runtime_and_external_evidence_root": True,
            "event_driven_replans_are_run_budget_bounded": True,
            "knowledge_snapshot_is_frozen_before_first_target": True,
            "every_target_receives_a_case_local_copy_of_the_same_seed_memory": True,
            "ablation_changes_exactly_one_declared_subsystem": True,
            "target_subset_is_explicit_and_frozen": True,
            "scores_are_fixed_cutoff_trajectory_projections": True,
            "legacy_objective_mode_does_not_reach_the_solver": True,
            "known_target_manifests_are_allowed_without_route_answer_authority": bool(
                allowed_prior_target_manifests
            ),
            "reviewed_strategy_portfolio_is_not_a_route_or_solved_claim": bool(
                strategy_portfolio_seed
            ),
            "known_strategy_reproduction_is_not_blind_strategy_discovery": bool(
                strategy_portfolio_seed
            ),
        },
    }
    completed_on_resume = _resume_completed_targets(
        previous_state,
        cases=cases,
        output_root=output_root,
        snapshot=snapshot,
        ablation=args.ablation,
    )
    state["targets"].update(completed_on_resume)
    state["resume"] = {
        "requested": bool(args.resume),
        "completed_targets_reused": len(completed_on_resume),
        "unfinished_targets_submitted": len(cases) - len(completed_on_resume),
    }
    _write_json(status_path, state)

    if args.preflight_only:
        passed = 0
        for case in cases:
            try:
                receipt = _prepare_case_snapshot(
                    output_root=output_root,
                    case=case,
                    snapshot=snapshot,
                    self_evo_library_seed=args.self_evo_library_seed,
                    leakage_audit_pack=args.leakage_audit_pack,
                    allowed_prior_target_manifests=allowed_prior_target_manifests,
                    strategy_portfolio_seed=strategy_portfolio_seed,
                    manifest=manifest,
                    run_dir=output_root / "runs" / case.target_name,
                    resume=False,
                )
                state["targets"][case.target_name] = {
                    "status": "preflight_passed",
                    "case_id": case.case_id,
                    "snapshot_receipt": receipt,
                }
                passed += 1
            except Exception as exc:  # bounded audit supervisor boundary
                failure_path = (
                    output_root
                    / "snapshots"
                    / "cases"
                    / f"{case.case_id}.preflight-failed.json"
                )
                state["targets"][case.target_name] = {
                    "status": "preflight_failed",
                    "case_id": case.case_id,
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "preflight_failure": (
                        json.loads(failure_path.read_text(encoding="utf-8"))
                        if failure_path.is_file()
                        else {}
                    ),
                }
        state["finished_at"] = _utc_now()
        state["preflight_only"] = True
        state["preflight_complete"] = passed == len(cases)
        state["preflight_passed_count"] = passed
        state["completed_count"] = 0
        state["complete"] = False
        _write_json(status_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if passed == len(cases) else 2

    def run(case: BlindCase) -> tuple[str, dict[str, Any]]:
        with state_lock:
            state["targets"][case.target_name] = {
                "status": "running",
                "case_id": case.case_id,
                "started_at": _utc_now(),
            }
            _write_json(status_path, state)
        return case.target_name, _run_case(
            case,
            manifest=manifest,
            output_root=output_root,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            execution_profile=args.execution_profile,
            strategy_portfolio_mode=args.strategy_portfolio_mode,
            strategy_portfolio_seed=strategy_portfolio_seed,
            strategic_milestones_per_branch=(
                args.strategic_milestones_per_branch
            ),
            node_expansions_per_branch=args.node_expansions_per_branch,
            reactionjson_candidates_per_node=args.reactionjson_candidates_per_node,
            strategy_tree_engine=args.strategy_tree_engine,
            no_chemenzy=args.no_chemenzy,
            fixed_cutoff_wall_time_s=fixed_cutoff_wall_time_s,
            fixed_cutoff_total_tasks=args.fixed_cutoff_total_tasks,
            resume=args.resume,
            visual=args.visual,
            chemenzy_env_prefix=args.chemenzy_env_prefix,
            chemenzy_stock_names=chemenzy_stock_names,
            chemenzy_stock_paths=chemenzy_stock_paths,
            snapshot=snapshot,
            ablation=args.ablation,
            self_evo_library_seed=args.self_evo_library_seed,
            inventory_snapshot=args.inventory_snapshot,
            benchmark_stock_index=args.benchmark_stock_index,
            benchmark_stock_name=args.benchmark_stock_name,
            leakage_audit_pack=args.leakage_audit_pack,
            allowed_prior_target_manifests=allowed_prior_target_manifests,
        )

    cases_to_run = [case for case in cases if case.target_name not in completed_on_resume]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, case): case for case in cases_to_run}
        for future in as_completed(futures):
            case = futures[future]
            try:
                name, result = future.result()
            except Exception as exc:  # bounded batch supervisor boundary
                name = case.target_name
                result = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "finished_at": _utc_now(),
                }
            with state_lock:
                state["targets"][name] = result
                _write_json(status_path, state)
    state["finished_at"] = _utc_now()
    state["complete"] = True
    state["completed_count"] = sum(
        row.get("status") == "completed" for row in state["targets"].values()
    )
    _write_json(status_path, state)
    summary_path = _write_result_summary(output_root, state)
    state["result_summary"] = {
        "status": "completed",
        "json_path": str(summary_path),
        "markdown_path": str(summary_path.with_suffix(".md")),
    }
    if matched_baseline is not None and matched_baseline_path is not None:
        comparison_path = _write_matched_comparison(
            output_root,
            baseline=matched_baseline,
            baseline_path=matched_baseline_path,
            candidate_path=summary_path,
        )
        state["matched_comparison"] = {
            "status": "completed",
            "json_path": str(comparison_path),
            "markdown_path": str(comparison_path.with_suffix(".md")),
        }
    _write_json(status_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["completed_count"] == len(cases) else 2


def _publish_run_registry(
    *,
    output_root: Path,
    cases: list[BlindCase],
    catalog_path: str = "",
    registry_id: str = "",
    registry_label: str = "",
    project_id: str = "",
    project_label: str = "",
) -> dict[str, Any]:
    """Publish registry location only; the owning RunIndex retains all state."""

    primary_paths = RuntimePaths.discover(repository_root=ROOT)
    resolved_catalog_path = (
        Path(catalog_path).expanduser().resolve()
        if str(catalog_path or "").strip()
        else registry_catalog_path(primary_paths)
    )
    default_project_label = output_root.parent.name or output_root.name
    resolved_project_label = str(project_label or "").strip() or default_project_label
    resolved_project_id = str(project_id or "").strip() or _catalog_slug(
        default_project_label, prefix="panel"
    )
    case_id = cases[0].case_id if len(cases) == 1 else ""
    result = RunRegistryCatalog(resolved_catalog_path).register(
        binding_from_registry_root(
            output_root,
            registry_id=str(registry_id or "").strip(),
            registry_label=(str(registry_label or "").strip() or output_root.name),
            project_id=resolved_project_id,
            project_label=resolved_project_label,
            case_id=case_id,
            repository_root=ROOT,
            source="blind_panel",
        )
    )
    return {
        "catalog_path": str(resolved_catalog_path),
        "registry": dict(result.get("registry") or {}),
        "semantics": {
            "catalog_is_discovery_only": True,
            "run_state_remains_in_owning_registry": True,
        },
    }


def _catalog_slug(value: str, *, prefix: str) -> str:
    normalized = "".join(
        char if char.isascii() and char.isalnum() else "-"
        for char in str(value or "").casefold()
    ).strip("-")
    normalized = "-".join(part for part in normalized.split("-") if part)
    if not normalized:
        normalized = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{normalized}"[:96].rstrip("-")


def _load_resume_panel_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write_result_summary(
    output_root: Path,
    state: Mapping[str, Any],
) -> Path:
    """Publish the result-first panel report immediately after the batch."""

    summary = summarize_panel(_hydrate_report_diagnostics(state))
    path = output_root / "panel-summary.json"
    _write_json(path, summary)
    path.with_suffix(".md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return path


def _validate_matched_baseline(
    baseline: Any,
    *,
    cases: list[BlindCase],
) -> None:
    if not isinstance(baseline, Mapping):
        raise ValueError("matched_baseline_summary_not_object")
    if baseline.get("schema_version") == "v4_blind_panel_summary.v3" and not (
        _json_digest_valid(baseline)
    ):
        raise ValueError("matched_baseline_summary_digest_invalid")
    case_ids = {
        str(row.get("case_id") or "")
        for row in baseline.get("per_target") or []
        if isinstance(row, Mapping)
    }
    missing = [case.case_id for case in cases if case.case_id not in case_ids]
    if missing:
        raise ValueError(
            f"selected_cases_missing_from_matched_baseline:{','.join(missing)}"
        )


def _write_matched_comparison(
    output_root: Path,
    *,
    baseline: Mapping[str, Any],
    baseline_path: Path,
    candidate_path: Path,
) -> Path:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    comparison = compare_matched_panels(
        baseline,
        candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    path = output_root / "matched-comparison.json"
    _write_json(path, comparison)
    path.with_suffix(".md").write_text(
        _comparison_markdown(comparison),
        encoding="utf-8",
    )
    return path


def _resume_completed_targets(
    previous: Mapping[str, Any],
    *,
    cases: list[BlindCase],
    output_root: Path,
    snapshot: Mapping[str, Any],
    ablation: str,
) -> dict[str, dict[str, Any]]:
    """Reuse only completed, report-backed rows from an identical panel binding."""

    if not previous:
        return {}
    if str(previous.get("schema_version") or "") != "v4_blind_panel_status.v1":
        raise RuntimeError("resume_panel_status_schema_mismatch")
    if str(previous.get("ablation") or "") != str(ablation):
        raise RuntimeError("resume_panel_ablation_mismatch")
    expected_ids = [case.case_id for case in cases]
    previous_ids = list(dict(previous.get("selection") or {}).get("selected_case_ids") or [])
    if previous_ids != expected_ids:
        raise RuntimeError("resume_panel_case_selection_mismatch")
    previous_snapshot = dict(previous.get("frozen_snapshot") or {})
    if previous_snapshot.get("content_sha256") != snapshot.get("content_sha256"):
        raise RuntimeError("resume_panel_snapshot_mismatch")
    previous_targets = dict(previous.get("targets") or {})
    completed: dict[str, dict[str, Any]] = {}
    for case in cases:
        row = dict(previous_targets.get(case.target_name) or {})
        report_path = output_root / "runs" / case.target_name / "target-only-solve-report.json"
        if (
            row.get("status") == "completed"
            and row.get("case_id") == case.case_id
            and report_path.is_file()
            and not _report_requires_operational_resume(report_path)
        ):
            completed[case.target_name] = {
                **row,
                "resume_reused_completed_report": True,
            }
    return completed


def _report_requires_operational_resume(path: Path) -> bool:
    """Reject a panel reuse when the owning run still needs local repair.

    A run may have reached a historical terminal state while a deterministic
    resume repair was only partially applied.  In that case the kernel is no
    longer paused, but reusing the report would permanently hide an admitted
    yet unmaterialized route step.  The reconciliation diagnostic is
    deliberately read-only, so using it as a resume trigger cannot upgrade a
    scientific claim.
    """

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(report, Mapping):
        return True
    reconciliation = dict(report.get("route_reconciliation") or {})
    if any(
        str(route.get("classification") or "") == "materialization_admission_gap"
        for route in reconciliation.get("routes") or []
        if isinstance(route, Mapping)
    ):
        return True
    stop = dict(report.get("stop_decision") or {})
    disposition = dict(report.get("current_disposition") or {})
    return bool(
        str(stop.get("decision") or "") == "paused"
        or stop.get("terminal") is False
        or str(disposition.get("historical_kernel_status") or "") == "paused"
    )


def _select_cases(
    cases: list[BlindCase],
    *,
    only: list[str] | tuple[str, ...] = (),
    max_targets: int | None = None,
) -> list[BlindCase]:
    selected_names = {str(value).casefold() for value in only}
    selected = [
        case
        for case in cases
        if not selected_names
        or case.target_name.casefold() in selected_names
        or case.case_id.casefold() in selected_names
    ]
    if max_targets is not None:
        if isinstance(max_targets, bool) or int(max_targets) < 1:
            raise ValueError("max_targets must be a positive integer")
        selected = selected[: int(max_targets)]
    return selected



def _resolve_panel_fixed_cutoff_wall_time_s(
    *,
    execution_profile: str,
    requested: float | None,
) -> float:
    """Resolve the panel cutoff independently of model-call budgets."""

    default_cutoff = 7_200.0
    if _is_paper_reach_profile(execution_profile):
        expected = float(SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_run_wall_time_s"])
        if requested is None:
            return expected
        value = float(requested)
        if value <= 0:
            raise ValueError("--fixed-cutoff-wall-time-s must be positive")
        if abs(value - expected) > 1e-9:
            raise ValueError(
                "paper_synthex requires the frozen operational target cutoff"
            )
        return value
    if requested is None:
        return default_cutoff
    value = float(requested)
    if value <= 0:
        raise ValueError("--fixed-cutoff-wall-time-s must be positive")
    return value
def _run_case(
    case: BlindCase,
    *,
    manifest: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    execution_profile: str,
    strategy_portfolio_mode: str = "auto",
    strategy_portfolio_seed: Path | None = None,
    strategic_milestones_per_branch: int = 1,
    node_expansions_per_branch: int | None = None,
    reactionjson_candidates_per_node: int | None = None,
    strategy_tree_engine: str | None = None,
    no_chemenzy: bool = False,
    resume: bool,
    visual: bool,
    chemenzy_env_prefix: str | None,
    chemenzy_stock_names: tuple[str, ...] = (),
    chemenzy_stock_paths: tuple[str, ...] = (),
    snapshot: Mapping[str, Any],
    ablation: str,
    self_evo_library_seed: str | None,
    inventory_snapshot: str | None,
    benchmark_stock_index: str | None,
    benchmark_stock_name: str,
    leakage_audit_pack: str | None,
    allowed_prior_target_manifests: tuple[Path, ...] = (),
    fixed_cutoff_wall_time_s: float = 7_200.0,
    fixed_cutoff_total_tasks: int = int(
        SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_total_tasks"]
    ),
) -> dict[str, Any]:
    run_id = _run_id_for_case(case)
    run_dir = output_root / "runs" / case.target_name
    report_path = run_dir / "target-only-solve-report.json"
    case_snapshot = _prepare_case_snapshot(
        output_root=output_root,
        case=case,
        snapshot=snapshot,
        self_evo_library_seed=self_evo_library_seed,
        leakage_audit_pack=leakage_audit_pack,
        allowed_prior_target_manifests=allowed_prior_target_manifests,
        strategy_portfolio_seed=strategy_portfolio_seed,
        manifest=manifest,
        run_dir=run_dir,
        resume=resume,
    )
    budget = dict(case.budget)
    proof_profile = execution_profile == "proof"
    matched = SYNTHEX_MATCHED_PROFILE_DEFAULTS
    expansion_budget = (
        int(node_expansions_per_branch)
        if node_expansions_per_branch is not None
        else int(matched["node_expansions_per_branch"])
    )
    max_model_invocations = max(
        int(matched["max_model_invocations"]),
        int(budget.get("max_model_invocations") or 0),
    )
    max_input_tokens = max(
        int(matched["max_input_tokens"]),
        int(budget.get("max_total_input_tokens") or 0),
    )
    max_output_tokens = max(
        int(matched["max_output_tokens"]),
        int(budget.get("max_total_output_tokens") or 0),
    )
    # The fixed cutoff is a hard execution limit, not just a reporting horizon.
    # A historical manifest allowance must not make a long Director task launch
    # fresh node calls after the frozen panel cutoff.
    max_wall_time_s = _resolved_model_wall_time_s(
        case_budget=budget,
        fixed_cutoff_wall_time_s=fixed_cutoff_wall_time_s,
    )
    max_accepted_expansions = max(
        int(matched["max_accepted_expansions"]),
        int(budget.get("max_accepted_expansions") or 0),
    )
    max_attempt_runs = max(
        int(matched["max_attempt_runs"]),
        int(budget.get("max_attempt_runs") or 0),
    )
    cutoff = {
        "wall_time_s": float(fixed_cutoff_wall_time_s),
        "settled_task_count": int(fixed_cutoff_total_tasks),
        "attempt_count": max_attempt_runs,
        "accepted_expansion_count": max_accepted_expansions,
        "model_invocations": max_model_invocations,
        "input_tokens": max_input_tokens,
        "output_tokens": max_output_tokens,
        "model_wall_time_s": float(max_wall_time_s),
    }
    can_resume = (
        resume and (run_dir / ".autoplanner" / "kernel" / "run_spec.json").is_file()
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not can_resume:
        if report_path.is_file():
            return _summarize_report(
                report_path,
                elapsed_s=0.0,
                reused=True,
                snapshot_receipt=case_snapshot,
                cutoff=cutoff,
            )
        raise RuntimeError("non_fresh_run_dir_requires_resume")
    candidate_width = (
        int(reactionjson_candidates_per_node)
        if reactionjson_candidates_per_node is not None
        else int(matched["reactionjson_candidates_per_node"])
    )
    if _is_paper_reach_profile(execution_profile) and candidate_width != int(
        matched["reactionjson_candidates_per_node"]
    ):
        raise ValueError(
            "paper_synthex panel cannot override the frozen ReactionJSON candidate width"
        )
    command = [
        sys.executable,
        "-m",
        "cascade_planner.cli",
        "solve-target",
        "--target-name",
        case.target_name,
        "--target-smiles",
        case.target_smiles,
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--manifest",
        str(manifest),
        "--blind-audit-root",
        str(ROOT),
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--execution-profile",
        execution_profile,
        "--strategy-search-profile",
        str(matched["strategy_search_profile"]),
        "--strategy-portfolio-mode",
        str(strategy_portfolio_mode),
        "--strategy-tree-engine",
        str(
            strategy_tree_engine
            if strategy_tree_engine is not None
            else (
                "aizynthfinder_mcts"
                if _is_paper_reach_profile(execution_profile)
                else "auto"
            )
        ),
        "--strategy-branches",
        str(matched["strategy_branches"]),
        "--strategy-branch-workers",
        str(matched["strategy_branch_workers"]),
        (
            "--stop-on-first-stock-closed-branch"
            if matched["stop_on_first_stock_closed_branch"]
            else "--no-stop-on-first-stock-closed-branch"
        ),
        "--node-expansions-per-branch",
        str(expansion_budget),
        "--strategic-milestones-per-branch",
        str(int(strategic_milestones_per_branch)),
        "--reactionjson-candidates-per-node",
        str(candidate_width),
        "--route-local-repair-rounds",
        str(matched["route_local_repair_rounds"]),
        "--max-node-prompt-bytes",
        str(matched["max_node_prompt_bytes"]),
        "--node-call-timeout-s",
        str(matched["node_call_timeout_s"]),
        "--critic-call-timeout-s",
        str(matched["critic_call_timeout_s"]),
        "--delivery-boundary",
        "stock_result",
        "--no-target-chemenzy-baseline",
        "--no-web-search",
        "--no-auto-patent-evidence",
        "--no-auto-literature-evidence",
        "--chemenzy-provider-route-reserve",
        "16",
        "--chemenzy-host-route-portfolio",
        "16",
        "--display-route-limit",
        "4",
        *_acceptance_cli_args(case),
        "--max-model-invocations",
        str(max_model_invocations),
        "--max-input-tokens",
        str(max_input_tokens),
        "--max-output-tokens",
        str(max_output_tokens),
        "--max-model-wall-time-s",
        str(max_wall_time_s),
        "--max-prompt-context-bytes",
        str(
            int(
                budget.get("max_prompt_context_bytes")
                or matched["max_prompt_context_bytes"]
            )
        ),
        "--max-accepted-expansions",
        str(max_accepted_expansions),
        "--max-attempt-runs",
        str(max_attempt_runs),
        "--max-total-tasks",
        str(int(fixed_cutoff_total_tasks)),
        "--max-run-wall-time-s",
        str(float(fixed_cutoff_wall_time_s)),
        "--max-map-reactions",
        str(matched["max_atom_mapping_reactions"]),
        "--max-stock-molecules",
        str(matched["max_stock_molecules"]),
        "--max-patent-sources",
        "3",
        "--max-literature-sources",
        "4",
        *(["--no-chemenzy"] if no_chemenzy else []),
        "--max-visual-invocations",
        "1" if visual else "0",
        "--max-visual-pages",
        "10" if visual and proof_profile else "6",
    ]
    # The paper protocol owns the matched Critic/Editor loop.  A second
    # campaign-level replan remains
    # outside this control profile.
    if _is_paper_reach_profile(execution_profile):
        command.append("--no-replan")
    if strategy_portfolio_seed:
        seed_sha256 = str(
            dict(snapshot.get("knowledge") or {}).get(
                "strategy_portfolio_seed_sha256"
            )
            or ""
        )
        command.extend(
            [
                "--strategy-portfolio-seed",
                str(strategy_portfolio_seed),
                "--strategy-portfolio-seed-sha256",
                seed_sha256,
                "--blind-audit-allowed-path",
                str(strategy_portfolio_seed),
            ]
        )
    command.extend(_ablation_cli_args(ablation))
    for path in allowed_prior_target_manifests:
        command.extend(["--blind-audit-allowed-path", str(path)])
    self_evo_path = str(case_snapshot.get("self_evo_library_path") or "")
    if self_evo_path:
        command.extend(["--self-evo-library", self_evo_path])
    if (
        inventory_snapshot
        and str(case.acceptance.get("stock_boundary") or "") == "procurement"
    ):
        command.extend(
            ["--inventory-snapshot", str(Path(inventory_snapshot).resolve())]
        )
    benchmark_index_sha256 = str(
        dict(snapshot.get("knowledge") or {}).get("benchmark_stock_index_sha256") or ""
    )
    if benchmark_stock_index:
        if str(case.acceptance.get("stock_boundary") or "") != "benchmark_search":
            raise RuntimeError(
                "frozen_benchmark_stock_index_requires_benchmark_search_case"
            )
        command.extend(
            [
                "--benchmark-stock-index",
                str(Path(benchmark_stock_index).resolve()),
                "--benchmark-stock-index-sha256",
                benchmark_index_sha256,
            ]
        )
        if benchmark_stock_name:
            command.extend(["--benchmark-stock-name", benchmark_stock_name])
    if chemenzy_env_prefix:
        command.extend(["--chemenzy-env-prefix", str(chemenzy_env_prefix)])
    for stock_name in chemenzy_stock_names:
        if str(stock_name).strip():
            command.extend(["--chemenzy-stock-name", str(stock_name)])
    for stock_path in chemenzy_stock_paths:
        if str(stock_path).strip():
            command.extend(["--chemenzy-stock-path", str(stock_path)])
    if can_resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment.update(
        {
            "AUTOPLANNER_RUNTIME_ROOT": str(output_root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(output_root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(output_root / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(
                output_root / "runtime" / "run_index.sqlite3"
            ),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(
                output_root / "external" / case.case_id
            ),
            "AUTOPLANNER_VENDOR_ROOT": str(ROOT / "vendor"),
        }
    )
    log_path = output_root / "logs" / f"{case.target_name}.log"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_s = round(time.monotonic() - started, 3)
    if not report_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_s": elapsed_s,
            "log_path": str(log_path),
            "error": tail,
            "finished_at": _utc_now(),
        }
    return _summarize_report(
        report_path,
        elapsed_s=elapsed_s,
        reused=False,
        snapshot_receipt=case_snapshot,
        cutoff=cutoff,
    )


def _run_id_for_case(case: BlindCase) -> str:
    """Use the validated manifest identity without target-family/date leakage."""

    return case.case_id


def _resolved_model_wall_time_s(
    *,
    case_budget: Mapping[str, Any],
    fixed_cutoff_wall_time_s: float,
) -> float:
    """Resolve the model allowance without crossing the frozen panel cutoff."""

    return min(
        float(fixed_cutoff_wall_time_s),
        max(
            float(SYNTHEX_MATCHED_PROFILE_DEFAULTS["max_model_wall_time_s"]),
            float(case_budget.get("max_total_wall_time_s") or 0.0),
        ),
    )


def _acceptance_cli_args(case: BlindCase) -> list[str]:
    """Translate the target-neutral manifest contract without hidden defaults."""

    acceptance = dict(case.acceptance)
    return [
        "--minimum-complete-routes",
        str(int(acceptance.get("minimum_complete_routes", 2))),
        "--minimum-edge-proof-level",
        str(int(acceptance.get("minimum_edge_proof_level", 2))),
        "--minimum-source-groups",
        str(int(acceptance.get("minimum_independent_source_groups", 2))),
        "--minimum-planning-route-steps",
        str(int(acceptance.get("minimum_planning_route_steps", 0))),
        "--stock-boundary",
        str(acceptance.get("stock_boundary") or "benchmark_search"),
    ]


def _ablation_cli_args(ablation: str) -> list[str]:
    return {
        "baseline": [],
        "no-chemenzy": ["--no-chemenzy"],
        "no-self-evo": ["--no-patent-self-evo"],
        "no-replan": ["--no-replan"],
        "chemenzy-only": ["--no-codex"],
        "codex-only": ["--no-chemenzy"],
        "unified-round-robin": ["--action-scheduler", "round_robin"],
        "unified-adaptive": ["--action-scheduler", "adaptive"],
    }[ablation]


def _prepare_panel_snapshot(
    *,
    output_root: Path,
    manifest: Path,
    cases: list[BlindCase],
    model: str,
    reasoning_effort: str,
    execution_profile: str,
    worker_count: int,
    visual: bool,
    ablation: str,
    chemenzy_env_prefix: str | None,
    chemenzy_stock_names: tuple[str, ...] = (),
    chemenzy_stock_paths: tuple[str, ...] = (),
    self_evo_library_seed: str | None,
    inventory_snapshot: str | None,
    benchmark_stock_index: str | None = None,
    benchmark_stock_name: str = "",
    paper_protocol: str | None = None,
    leakage_audit_pack: str | None,
    allowed_prior_target_manifests: tuple[Path, ...] = (),
    resume: bool,
    projection_policy: Mapping[str, Any] | None = None,
    strategy_portfolio_mode: str = "auto",
    strategy_portfolio_seed: str | None = None,
) -> dict[str, Any]:
    snapshot_path = output_root / "snapshots" / "benchmark-snapshot.json"
    seed = _optional_file(self_evo_library_seed, "self_evo_library_seed")
    inventory = _optional_file(inventory_snapshot, "inventory_snapshot")
    benchmark_index = _optional_file(benchmark_stock_index, "benchmark_stock_index")
    protocol = _optional_file(paper_protocol, "paper_protocol")
    leakage_pack = _optional_file(leakage_audit_pack, "leakage_audit_pack")
    reviewed_strategy = _optional_file(
        strategy_portfolio_seed,
        "strategy_portfolio_seed",
    )
    chemenzy_python = _chemenzy_python(chemenzy_env_prefix)
    provider_snapshot = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "execution_profile": execution_profile,
        "strategy_portfolio_mode": strategy_portfolio_mode,
        "worker_count": worker_count,
        "visual_enabled": visual,
        "codex_cli": _binary_fingerprint(shutil.which("codex")),
        "host_python": _binary_fingerprint(sys.executable),
        "chemenzy_python": _binary_fingerprint(chemenzy_python),
        "chemenzy_stock_names": [str(value) for value in chemenzy_stock_names],
        "chemenzy_stock_paths": [str(value) for value in chemenzy_stock_paths],
        "remote_model_weights_are_not_bitwise_frozen": True,
    }
    knowledge = {
        "self_evo_library_path": str(seed or ""),
        "self_evo_library_sha256": _file_sha256(seed) if seed else "",
        "inventory_snapshot_path": str(inventory or ""),
        "inventory_snapshot_sha256": _file_sha256(inventory) if inventory else "",
        "benchmark_stock_index_path": str(benchmark_index or ""),
        "benchmark_stock_index_sha256": (
            _file_sha256(benchmark_index) if benchmark_index else ""
        ),
        "benchmark_stock_name": str(benchmark_stock_name or ""),
        "paper_protocol_path": str(protocol or ""),
        "paper_protocol_sha256": _file_sha256(protocol) if protocol else "",
        "leakage_audit_pack_path": str(leakage_pack or ""),
        "leakage_audit_pack_sha256": _file_sha256(leakage_pack) if leakage_pack else "",
        "strategy_portfolio_seed_path": str(reviewed_strategy or ""),
        "strategy_portfolio_seed_sha256": (
            _file_sha256(reviewed_strategy) if reviewed_strategy else ""
        ),
        "allowed_prior_target_manifests": [
            {
                "path": str(path),
                "file_sha256": _file_sha256(path),
            }
            for path in allowed_prior_target_manifests
        ],
        "benchmark_stock_is_a_search_boundary_not_procurement": any(
            str(case.acceptance.get("stock_boundary") or "") == "benchmark_search"
            for case in cases
        ),
    }
    environment_material = {
        "manifest_sha256": _file_sha256(manifest),
        "selected_case_ids": sorted(case.case_id for case in cases),
        "provider_snapshot": provider_snapshot,
        "knowledge": knowledge,
        "projection_policy": dict(projection_policy or {}),
    }
    expected = {
        "schema_version": "v4_blind_benchmark_snapshot.v1",
        **environment_material,
        "base_environment_sha256": _json_sha256(environment_material),
        "ablation": ablation,
        "semantics": {
            "captured_before_first_target": True,
            "case_local_memory_copies_prevent_cross_target_learning_leakage": True,
            "snapshot_is_reproducibility_metadata_not_scientific_authority": True,
            "remote_provider_identity_is_recorded_but_not_bitwise_frozen": True,
            "evaluation_projection_cannot_change_solver_control_flow": True,
        },
    }
    if snapshot_path.is_file():
        current = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not _json_digest_valid(current):
            raise RuntimeError("blind_benchmark_snapshot_digest_invalid")
        comparable = {
            key: value
            for key, value in current.items()
            if key not in {"created_at", "content_sha256"}
        }
        if comparable != expected:
            raise RuntimeError("blind_benchmark_snapshot_configuration_mismatch")
        return current
    if resume:
        raise RuntimeError("blind_benchmark_resume_snapshot_missing")
    value = {**expected, "created_at": _utc_now()}
    value["content_sha256"] = _json_sha256(value)
    _write_json(snapshot_path, value)
    return value


def _prepare_case_snapshot(
    *,
    output_root: Path,
    case: BlindCase,
    snapshot: Mapping[str, Any],
    self_evo_library_seed: str | None,
    leakage_audit_pack: str | None,
    allowed_prior_target_manifests: tuple[Path, ...] = (),
    manifest: Path,
    run_dir: Path,
    resume: bool,
    strategy_portfolio_seed: Path | None = None,
) -> dict[str, Any]:
    case_external = output_root / "external" / case.case_id
    case_external.mkdir(parents=True, exist_ok=True)
    seed = _optional_file(self_evo_library_seed, "self_evo_library_seed")
    leakage_pack = _optional_file(leakage_audit_pack, "leakage_audit_pack")
    reviewed_strategy = _optional_file(
        str(strategy_portfolio_seed) if strategy_portfolio_seed else None,
        "strategy_portfolio_seed",
    )
    reviewed_strategy_sha = (
        _file_sha256(reviewed_strategy) if reviewed_strategy else ""
    )
    self_evo_path = case_external / "self-evo" / "patent-reaction-template-library.json"
    seed_sha = ""
    if seed:
        seed_sha = _file_sha256(seed)
        if self_evo_path.is_file():
            if _file_sha256(self_evo_path) != seed_sha:
                raise RuntimeError("blind_case_self_evo_snapshot_drift")
        else:
            self_evo_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed, self_evo_path)
    receipt_path = output_root / "snapshots" / "cases" / f"{case.case_id}.json"
    if receipt_path.is_file():
        current = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not _json_digest_valid(current)
            or current.get("case_id") != case.case_id
            or current.get("panel_snapshot_sha256")
            != str(snapshot.get("content_sha256") or "")
            or current.get("self_evo_library_sha256") != seed_sha
            or current.get("strategy_portfolio_seed_sha256")
            != reviewed_strategy_sha
        ):
            raise RuntimeError("blind_case_snapshot_receipt_mismatch")
        return current
    if resume:
        raise RuntimeError("blind_case_resume_snapshot_receipt_missing")
    leakage_needles, synonym_not_applicable = _case_leakage_needles(
        leakage_pack,
        case=case,
        manifest_sha256=str(snapshot.get("manifest_sha256") or ""),
    )
    supervisor_preflight = audit_blind_preflight(
        case,
        repository_root=ROOT,
        run_dir=run_dir,
        manifest_path=manifest,
        additional_allowed_paths=[
            *allowed_prior_target_manifests,
            *([leakage_pack] if leakage_pack else []),
            *([reviewed_strategy] if reviewed_strategy else []),
        ],
        additional_leakage_needles=leakage_needles,
        target_synonym_not_applicable_reason=synonym_not_applicable,
    )
    if supervisor_preflight.get("accepted") is not True:
        failure = {
            "schema_version": "v4_blind_case_preflight_failure.v1",
            "case_id": case.case_id,
            "panel_snapshot_sha256": str(snapshot.get("content_sha256") or ""),
            "supervisor_preflight": supervisor_preflight,
            "semantics": {
                "failure_is_retained": True,
                "planner_was_not_started": True,
            },
        }
        failure["content_sha256"] = _json_sha256(failure)
        _write_json(
            receipt_path.with_name(f"{case.case_id}.preflight-failed.json"),
            failure,
        )
        raise RuntimeError(
            "blind_supervisor_preflight_rejected:"
            + ",".join(supervisor_preflight.get("reasons") or [])
        )
    receipt = {
        "schema_version": "v4_blind_case_snapshot_receipt.v1",
        "case_id": case.case_id,
        "panel_snapshot_sha256": str(snapshot.get("content_sha256") or ""),
        "base_environment_sha256": str(snapshot.get("base_environment_sha256") or ""),
        "case_external_root": str(case_external),
        "self_evo_library_path": str(self_evo_path) if seed else "",
        "self_evo_library_sha256": seed_sha,
        "strategy_portfolio_seed_sha256": reviewed_strategy_sha,
        "supervisor_preflight": supervisor_preflight,
        "semantics": {
            "case_external_root_is_isolated": True,
            "seed_memory_is_identical_across_cases": True,
            "case_writes_cannot_change_another_case_seed": True,
        },
    }
    receipt["content_sha256"] = _json_sha256(receipt)
    _write_json(receipt_path, receipt)
    return receipt


def _prior_target_manifest_files(values: list[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"allowed_prior_target_manifest_missing:{path}")
        load_blind_manifest(path)
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise SystemExit("allowed_prior_target_manifest_duplicate")
    return tuple(paths)


def _validated_strategy_portfolio_seed(
    value: str,
    *,
    cases: list[BlindCase],
) -> Path | None:
    """Validate a Strategy-only screen before any provider or registry starts."""

    if not str(value or "").strip():
        return None
    if len(cases) != 1:
        raise ValueError("strategy_portfolio_seed_requires_one_selected_case")
    path = _optional_file(value, "strategy_portfolio_seed")
    if path is None:
        raise ValueError("strategy_portfolio_seed_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("strategy_portfolio_seed_json_invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("strategy_portfolio_seed_object_required")
    seed_target = str(
        payload.get("canonical_target_smiles")
        or payload.get("target_smiles")
        or ""
    ).strip()
    if not seed_target or canonical_smiles(seed_target) != cases[0].target_smiles:
        raise ValueError("strategy_portfolio_seed_target_mismatch")
    raw_cards = payload.get("reviewed_cards") or payload.get("strategy_cards") or []
    required = ("strategy_query", "critical_assumption", "critic_checkpoint")
    if (
        not isinstance(raw_cards, list)
        or len(raw_cards) != 3
        or any(
            not isinstance(card, Mapping)
            or any(not str(card.get(field) or "").strip() for field in required)
            for card in raw_cards
        )
    ):
        raise ValueError("strategy_portfolio_seed_requires_three_reviewed_cards")
    return path


def _summarize_report(
    path: Path,
    *,
    elapsed_s: float,
    reused: bool,
    cutoff: Mapping[str, int | float],
    snapshot_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    gate_report = dict(report.get("gates") or {})
    gates = dict(gate_report.get("gates") or {})
    counts = dict(gate_report.get("counts") or {})
    claim = dict(report.get("claim") or {})
    model = dict(report.get("model_cost") or {})
    try:
        projection = project_campaign_trajectory_at_cutoff(
            dict(report.get("trajectory") or {}),
            cutoff=cutoff,
        )
    except (TypeError, ValueError) as exc:
        projection = {
            "schema_version": TRAJECTORY_CUTOFF_PROJECTION_SCHEMA,
            "available": False,
            "cutoff": dict(cutoff),
            "unavailable_reason": f"{type(exc).__name__}:{exc}"[:1000],
            "gate_summary": {},
            "gate_counts": {},
            "route_counts": {},
            "resource_usage": {},
            "observed_resources": {},
            "time_to_first": {},
        }
    projection_available = projection.get("available") is True
    projected_gates = dict(projection.get("gate_summary") or {})
    projected_counts = dict(projection.get("gate_counts") or {})
    projected_route_counts = dict(projection.get("route_counts") or {})
    projected_resources = dict(projection.get("observed_resources") or {})
    projected_model = dict(
        dict(projection.get("resource_usage") or {}).get("model") or {}
    )
    final_paper_equivalent = dict(report.get("paper_equivalent") or {})
    paper_equivalent = {
        **final_paper_equivalent,
        "paper_equivalent_solved": (
            int(
                projected_route_counts.get("stock_closed_skeletons")
                or projected_counts.get("stock_closed_skeletons")
                or 0
            )
            > 0
        ),
        "metric_cutoff_bound": True,
    }
    stages = list(report.get("stages") or [])
    chemenzy_stages = [
        {"stage": str(row.get("stage") or ""), **dict(row.get("detail") or {})}
        for row in stages
        if isinstance(row, Mapping)
        and row.get("stage")
        in {
            "chemenzy_baseline",
            "chemenzy_guided_frontier",
            "chemenzy_stock_recovery",
        }
    ]
    baseline = next(
        (row for row in chemenzy_stages if row.get("stage") == "chemenzy_baseline"),
        {},
    )
    evidence_stages = [
        dict(row.get("detail") or {})
        for row in stages
        if isinstance(row, Mapping)
        and row.get("stage") in {"evidence_acquisition", "replan_evidence_acquisition"}
    ]
    source_routes = [dict(row.get("source_route") or {}) for row in evidence_stages]
    global_stages = [
        row
        for row in stages
        if isinstance(row, Mapping)
        and row.get("stage") in {"global_campaign", "global_replan"}
    ]
    failure_events = _panel_failure_events(stages)
    preflight_case = dict(dict(report.get("preflight") or {}).get("case") or {})
    stop_decision = dict(report.get("stop_decision") or {})
    current_disposition = dict(report.get("current_disposition") or {})
    claim_accepted = claim.get("accepted_under_configured_policy") is True
    terminal_completed = bool(
        stop_decision.get("terminal") is True
        and str(stop_decision.get("decision") or "") in {"completed", "accepted"}
    )
    scientifically_terminal = bool(
        claim_accepted
        or current_disposition.get("state") == "accepted"
        or terminal_completed
    )
    # ``status`` is an operational lifecycle field. A terminal solver with a
    # valid frozen projection completed its benchmark observation even when a
    # stricter scientific acceptance axis remains open. A paused kernel is
    # different: reporting it as completed makes supervisors and the UI
    # oscillate between a pending run and a frozen completed row.
    operationally_paused = bool(
        str(stop_decision.get("decision") or "") == "paused"
        or str(current_disposition.get("historical_kernel_status") or "")
        == "paused"
    )
    target_status = (
        "projection_unavailable"
        if not projection_available
        else "paused"
        if operationally_paused
        else "completed"
    )
    scientific_status = "accepted" if scientifically_terminal else "unresolved"
    return {
        "status": target_status,
        "scientific_status": scientific_status,
        "scientific_disposition": str(current_disposition.get("state") or ""),
        "case_id": str(preflight_case.get("case_id") or report.get("run_id") or ""),
        "claim": (
            "fixed_cutoff_stock_closed"
            if projected_gates.get("B4") is True
            else "fixed_cutoff_observed"
        ),
        "accepted_under_configured_policy": projected_gates.get("B5") is True,
        "elapsed_s": projected_resources.get("wall_time_s"),
        "runner_elapsed_s": elapsed_s,
        "reused": reused,
        "run_id": str(report.get("run_id") or ""),
        "run_dir": str(report.get("run_dir") or path.parent),
        "report_path": str(path),
        "snapshot_receipt": dict(snapshot_receipt or {}),
        "model_cost": projected_model,
        "within_resource_budget": projection_available,
        "false_closure_claim_count": int(
            gate_report.get("false_closure_claim_count") or 0
        ),
        "attempt_count": int(projected_resources.get("attempt_count") or 0),
        "accepted_expansion_count": int(
            projected_resources.get("accepted_expansion_count") or 0
        ),
        "gate_summary": projected_gates,
        "route_counts": {**projected_counts, **projected_route_counts},
        "paper_equivalent": paper_equivalent,
        "anytime_route_counts": projected_route_counts,
        "fixed_cutoff_projection": projection,
        "final_state": {
            "claim": claim,
            "gate_summary": {
                key.split("_", 1)[0]: bool(value)
                for key, value in gates.items()
                if key[:2] in {"B0", "B1", "B2", "B3", "B4", "B5"}
            },
            "route_counts": counts,
            "model_cost": model,
            "within_resource_budget": bool(
                dict(report.get("resource_envelope") or {}).get("within_budget")
            ),
            "attempt_count": int(report.get("attempt_count") or 0),
            "accepted_expansion_count": int(
                report.get("accepted_expansion_count") or 0
            ),
            "stop_decision": stop_decision,
            "current_disposition": current_disposition,
        },
        "planning_depth": dict(report.get("planning_depth") or {}),
        "chemenzy": {
            "status": (
                "completed"
                if any(row.get("status") == "completed" for row in chemenzy_stages)
                else str(baseline.get("status") or "")
            ),
            "frontier_count": sum(
                int(row.get("frontier_count") or 0) for row in chemenzy_stages
            ),
            "provider_invocation_count": sum(
                int(
                    row.get("provider_invocation_count")
                    or row.get("executed_frontier_count")
                    or (
                        1
                        if row.get("stage") == "chemenzy_baseline"
                        and row.get("status") not in {"disabled", "runtime_unavailable"}
                        else 0
                    )
                )
                for row in chemenzy_stages
            ),
            "proposal_count": sum(
                int(row.get("proposal_count") or 0) for row in chemenzy_stages
            ),
            "initial_delegation_status": str(baseline.get("status") or ""),
            "stock_recovery_used": any(
                row.get("stage") == "chemenzy_stock_recovery"
                and row.get("status") == "completed"
                for row in chemenzy_stages
            ),
        },
        "campaign": {
            "global_pass_count": len(global_stages),
            "evidence_replan_ran": any(
                row.get("stage") == "global_replan" for row in global_stages
            ),
        },
        "failure_events": failure_events,
        "rejection_taxonomy": dict(report.get("rejection_taxonomy") or {}),
        "evidence": {
            "pass_count": len(evidence_stages),
            "source_count": sum(
                int(row.get("source_count") or 0) for row in evidence_stages
            ),
            "exact_record_count": sum(
                int(row.get("exact_record_count") or 0) for row in evidence_stages
            ),
            "visual_invocation_count": sum(
                int(row.get("visual_invocations") or 0) for row in evidence_stages
            ),
            "source_route_proposal_count": sum(
                int(row.get("proposal_count") or 0) for row in source_routes
            ),
            "source_route_host_accepted_count": sum(
                int(
                    dict(row.get("validation") or {}).get("accepted_validation_count")
                    or 0
                )
                for row in source_routes
            ),
        },
        "stage_timings": {
            str(row.get("stage") or ""): float(row.get("elapsed_s") or 0.0)
            for row in stages
            if isinstance(row, Mapping) and row.get("stage")
        },
        "workbench_url": (f"/api/v4/runs/{report.get('run_id')}/workbench.html"),
        "finished_at": _utc_now(),
        "semantics": {
            "score_fields_are_only_from_fixed_cutoff_projection": True,
            "final_state_is_diagnostic_only": True,
            "legacy_objective_claim_does_not_control_scoring": True,
        },
    }


def _panel_failure_events(stages: list[Any]) -> list[dict[str, Any]]:
    """Retain bounded, typed stage failures instead of collapsing them to unsolved."""

    events: list[dict[str, Any]] = []
    for raw in stages:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or "")
        detail = dict(raw.get("detail") or {})
        detail_status = str(detail.get("status") or "")
        if status not in {"failed", "rejected", "timeout", "error", "partial"} and (
            detail_status
            not in {"failed", "rejected", "timeout", "error", "partial"}
        ):
            continue
        raw_reasons = detail.get("reasons")
        reasons = (
            [str(value) for value in raw_reasons if str(value)]
            if isinstance(raw_reasons, list)
            else []
        )
        reason = str(detail.get("reason") or "")
        if reason:
            reasons.append(reason)
        if not reasons:
            reasons.append(detail_status or status)
        events.append(
            {
                "stage": str(raw.get("stage") or ""),
                "status": status or detail_status,
                "reasons": sorted(set(reasons))[:8],
            }
        )
    return events


def _optional_file(value: str | None, label: str) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_missing:{path}")
    return path


def _case_leakage_needles(
    path: Path | None,
    *,
    case: BlindCase,
    manifest_sha256: str,
) -> tuple[dict[str, list[str]], str]:
    if path is None:
        return {}, ""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("blind_leakage_audit_pack_not_object")
    if value.get("schema_version") != "blind_leakage_audit_pack.v1":
        raise ValueError("blind_leakage_audit_pack_schema_invalid")
    if str(value.get("manifest_sha256") or "") != manifest_sha256:
        raise ValueError("blind_leakage_audit_pack_manifest_mismatch")
    if value.get("content_sha256") and not _json_digest_valid(value):
        raise ValueError("blind_leakage_audit_pack_digest_invalid")
    raw_cases = value.get("cases") or {}
    if not isinstance(raw_cases, Mapping):
        raise ValueError("blind_leakage_audit_pack_cases_invalid")
    row = raw_cases.get(case.case_id) or {}
    if not isinstance(row, Mapping):
        raise ValueError(f"blind_leakage_audit_case_invalid:{case.case_id}")
    synonyms = sorted(
        {
            str(item).strip()
            for item in row.get("target_synonyms") or []
            if len(str(item).strip()) >= 5
        }
    )
    synonym_not_applicable = str(
        row.get("target_synonym_not_applicable_reason") or ""
    ).strip()
    intermediate_smiles = sorted(
        {
            canonical
            for item in row.get("key_intermediate_smiles") or []
            if (canonical := canonical_smiles(item))
        }
    )
    intermediate_inchikeys = sorted(
        {
            Chem.MolToInchiKey(molecule)
            for smiles in intermediate_smiles
            if (molecule := Chem.MolFromSmiles(smiles)) is not None
        }
    )
    if not synonyms and not synonym_not_applicable:
        raise ValueError(f"blind_leakage_target_synonyms_missing:{case.case_id}")
    if not intermediate_smiles:
        raise ValueError(f"blind_leakage_key_intermediates_missing:{case.case_id}")
    return (
        {
            "target_synonym": synonyms,
            "key_intermediate_smiles": intermediate_smiles,
            "key_intermediate_inchikey": intermediate_inchikeys,
        },
        synonym_not_applicable,
    )


def _chemenzy_python(prefix: str | None) -> str | None:
    if not str(prefix or "").strip():
        return None
    root = Path(str(prefix)).expanduser().resolve()
    candidates = (
        root / "python.exe",
        root / "Scripts" / "python.exe",
        root / "bin" / "python",
    )
    return str(next((path for path in candidates if path.is_file()), "")) or None


def _resolve_panel_chemenzy_stock_binding(
    *,
    benchmark_stock_index: str | None,
    benchmark_stock_name: str,
    stock_names: tuple[str, ...],
    stock_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve one stock boundary shared by ChemEnzy search and host B4."""

    names = tuple(
        dict.fromkeys(
            str(value).strip() for value in stock_names if str(value).strip()
        )
    )
    raw_paths = tuple(str(value).strip() for value in stock_paths if str(value).strip())
    if not str(benchmark_stock_index or "").strip():
        return names, raw_paths

    benchmark_path = Path(str(benchmark_stock_index)).expanduser().resolve()
    if not names and not raw_paths:
        aligned_name = str(benchmark_stock_name or "").strip() or "Benchmark-stock"
        return (aligned_name,), (f"{aligned_name}={benchmark_path}",)

    parsed: dict[str, Path] = {}
    for raw in raw_paths:
        name, separator, path = raw.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError("chemenzy_stock_path_must_be_NAME_EQUALS_PATH")
        if name.strip() in parsed:
            raise ValueError(f"duplicate_chemenzy_stock_path:{name.strip()}")
        parsed[name.strip()] = Path(path.strip()).expanduser().resolve()
    if not parsed:
        raise ValueError("benchmark_stock_requires_chemenzy_path_alignment")
    if not names:
        names = tuple(parsed)
    if set(names) != set(parsed) or len(parsed) != 1:
        raise ValueError("benchmark_stock_requires_one_matching_chemenzy_stock")
    if next(iter(parsed.values())) != benchmark_path:
        raise ValueError("benchmark_and_chemenzy_stock_paths_differ")
    return names, tuple(f"{name}={parsed[name]}" for name in names)


def _strategy_tree_requires_benchmark_stock_index(
    *,
    execution_profile: str,
    strategy_tree_engine: str | None,
) -> bool:
    effective_engine = str(strategy_tree_engine or "").strip()
    if not effective_engine and _is_paper_reach_profile(execution_profile):
        effective_engine = "aizynthfinder_mcts"
    return effective_engine == "aizynthfinder_mcts"


def _binary_fingerprint(value: str | None) -> dict[str, Any]:
    if not value:
        return {"available": False, "path": "", "size_bytes": 0, "sha256": ""}
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return {"available": False, "path": str(path), "size_bytes": 0, "sha256": ""}
    return {
        "available": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    return bool(observed) and observed == _json_sha256(material)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
