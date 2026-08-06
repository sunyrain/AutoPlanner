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
from cascade_planner.interfaces.target_runtime_dependencies import (  # noqa: E402
    TARGET_PROFILE_DEFAULTS,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument(
        "--execution-profile",
        choices=("fast", "standard", "proof"),
        default="standard",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("benchmark_search", "scientific_proof", "procurement_delivery"),
        default="benchmark_search",
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
            "Optional frozen SQLite stock-membership index shared read-only by "
            "benchmark_search cases; no planning provider is loaded from it."
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
        help="override a selected ChemEnzy stock with an explicit CSV path",
    )
    args = parser.parse_args(argv)

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

    for name in ("audit", "runtime", "runs", "artifacts", "external", "logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    snapshot = _prepare_panel_snapshot(
        output_root=output_root,
        manifest=manifest,
        cases=cases,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        execution_profile=args.execution_profile,
        objective_mode=args.objective_mode,
        worker_count=args.workers,
        visual=args.visual,
        ablation=args.ablation,
        chemenzy_env_prefix=args.chemenzy_env_prefix,
        chemenzy_stock_names=tuple(args.chemenzy_stock_name),
        chemenzy_stock_paths=tuple(args.chemenzy_stock_path),
        self_evo_library_seed=args.self_evo_library_seed,
        inventory_snapshot=args.inventory_snapshot,
        benchmark_stock_index=args.benchmark_stock_index,
        benchmark_stock_name=args.benchmark_stock_name,
        leakage_audit_pack=args.leakage_audit_pack,
        resume=args.resume,
    )
    status_path = output_root / "panel-status.json"
    state_lock = Lock()
    state: dict[str, Any] = {
        "schema_version": "v4_blind_panel_status.v1",
        "manifest_path": str(manifest),
        "output_root": str(output_root),
        "started_at": _utc_now(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "execution_profile": args.execution_profile,
        "objective_mode": args.objective_mode,
        "ablation": args.ablation,
        "worker_count": args.workers,
        "chemenzy_env_prefix": str(args.chemenzy_env_prefix or ""),
        "chemenzy_stock_names": [str(value) for value in args.chemenzy_stock_name],
        "chemenzy_stock_paths": [str(value) for value in args.chemenzy_stock_path],
        "frozen_snapshot": {
            "path": str(output_root / "snapshots" / "benchmark-snapshot.json"),
            "content_sha256": str(snapshot.get("content_sha256") or ""),
            "base_environment_sha256": str(snapshot.get("base_environment_sha256") or ""),
            "manifest_sha256": str(snapshot.get("manifest_sha256") or ""),
            "self_evo_library_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("self_evo_library_sha256") or ""
            ),
            "inventory_snapshot_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("inventory_snapshot_sha256") or ""
            ),
            "benchmark_stock_index_sha256": str(
                dict(snapshot.get("knowledge") or {}).get(
                    "benchmark_stock_index_sha256"
                )
                or ""
            ),
            "leakage_audit_pack_sha256": str(
                dict(snapshot.get("knowledge") or {}).get("leakage_audit_pack_sha256") or ""
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
            case.target_name: {"status": "queued", "case_id": case.case_id} for case in cases
        },
        "semantics": {
            "target_name_and_smiles_only": True,
            "no_local_pdf_doi_patent_or_route_seed": True,
            "isolated_runtime_and_external_evidence_root": True,
            "event_driven_replans_are_run_budget_bounded": True,
            "knowledge_snapshot_is_frozen_before_first_target": True,
            "every_target_receives_a_case_local_copy_of_the_same_seed_memory": True,
            "ablation_changes_exactly_one_declared_subsystem": True,
            "target_subset_is_explicit_and_frozen": True,
        },
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
                    output_root / "snapshots" / "cases" / f"{case.case_id}.preflight-failed.json"
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
            objective_mode=args.objective_mode,
            resume=args.resume,
            visual=args.visual,
            chemenzy_env_prefix=args.chemenzy_env_prefix,
            chemenzy_stock_names=tuple(args.chemenzy_stock_name),
            chemenzy_stock_paths=tuple(args.chemenzy_stock_path),
            snapshot=snapshot,
            ablation=args.ablation,
            self_evo_library_seed=args.self_evo_library_seed,
            inventory_snapshot=args.inventory_snapshot,
            benchmark_stock_index=args.benchmark_stock_index,
            benchmark_stock_name=args.benchmark_stock_name,
            leakage_audit_pack=args.leakage_audit_pack,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, case): case for case in cases}
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
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["completed_count"] == len(cases) else 2


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
        if not selected_names or case.target_name.casefold() in selected_names
    ]
    if max_targets is not None:
        if isinstance(max_targets, bool) or int(max_targets) < 1:
            raise ValueError("max_targets must be a positive integer")
        selected = selected[: int(max_targets)]
    return selected


def _run_case(
    case: BlindCase,
    *,
    manifest: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    execution_profile: str,
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
    objective_mode: str = "benchmark_search",
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
        manifest=manifest,
        run_dir=run_dir,
        resume=resume,
    )
    can_resume = resume and (run_dir / ".autoplanner" / "kernel" / "run_spec.json").is_file()
    if run_dir.exists() and any(run_dir.iterdir()) and not can_resume:
        if report_path.is_file():
            return _summarize_report(
                report_path,
                elapsed_s=0.0,
                reused=True,
                snapshot_receipt=case_snapshot,
            )
        raise RuntimeError("non_fresh_run_dir_requires_resume")
    budget = dict(case.budget)
    proof_profile = execution_profile == "proof"
    profile_defaults = TARGET_PROFILE_DEFAULTS[execution_profile]
    max_model_invocations = max(
        3 if visual or proof_profile else 2,
        int(budget.get("max_model_invocations") or 0),
    )
    max_input_tokens = max(
        int(profile_defaults["max_input_tokens"]),
        int(budget.get("max_total_input_tokens") or 0),
    )
    # Keep the profile-level cumulative envelope available for the initial
    # architecture, evidence-aware replan, and final portfolio synthesis.
    max_output_tokens = max(
        int(profile_defaults["max_output_tokens"]),
        int(budget.get("max_total_output_tokens") or 0),
    )
    max_wall_time_s = max(
        int(profile_defaults["max_model_wall_time_s"]),
        int(budget.get("max_total_wall_time_s") or 0),
    )
    max_accepted_expansions = max(
        64,
        int(budget.get("max_accepted_expansions") or 0),
    )
    max_attempt_runs = max(128, int(budget.get("max_attempt_runs") or 0))
    generous_search = max_accepted_expansions >= 96 or max_model_invocations >= 5
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
        "--objective-mode",
        objective_mode,
        "--target-chemenzy-baseline",
        "--chemenzy-provider-route-reserve",
        "16",
        "--chemenzy-host-route-portfolio",
        "8",
        "--display-route-limit",
        "4",
        "--initial-director-web-search",
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
        str(int(budget.get("max_prompt_context_bytes") or 96_000)),
        "--max-accepted-expansions",
        str(max_accepted_expansions),
        "--max-attempt-runs",
        str(max_attempt_runs),
        "--max-map-reactions",
        "64",
        "--max-stock-molecules",
        "32",
        "--chemenzy-max-steps",
        str(profile_defaults["steps"]),
        "--chemenzy-iterations",
        str(profile_defaults["iterations"]),
        "--chemenzy-expansion-topk",
        str(profile_defaults["topk"]),
        "--chemenzy-timeout-s",
        str(profile_defaults["timeout"]),
        "--max-patent-sources",
        "3",
        "--max-literature-sources",
        "4",
        "--guided-chemenzy-frontiers",
        "5" if generous_search else "3",
        "--guided-chemenzy-iterations",
        "8" if generous_search else "6",
        "--guided-chemenzy-timeout-s",
        "90" if generous_search else "60",
        "--max-visual-invocations",
        "1" if visual else "0",
        "--max-visual-pages",
        "10" if visual and generous_search else "6",
    ]
    command.extend(_ablation_cli_args(ablation))
    self_evo_path = str(case_snapshot.get("self_evo_library_path") or "")
    if self_evo_path:
        command.extend(["--self-evo-library", self_evo_path])
    if inventory_snapshot and str(case.acceptance.get("stock_boundary") or "") == "procurement":
        command.extend(["--inventory-snapshot", str(Path(inventory_snapshot).resolve())])
    benchmark_index_sha256 = str(
        dict(snapshot.get("knowledge") or {}).get(
            "benchmark_stock_index_sha256"
        )
        or ""
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
            "AUTOPLANNER_RUN_INDEX_PATH": str(output_root / "runtime" / "run_index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(output_root / "external" / case.case_id),
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
    )


def _run_id_for_case(case: BlindCase) -> str:
    """Use the validated manifest identity without target-family/date leakage."""

    return case.case_id


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
        "no-chemenzy": ["--no-chemenzy", "--no-guided-chemenzy"],
        "no-self-evo": ["--no-patent-self-evo"],
        "no-replan": ["--no-replan"],
        "chemenzy-only": ["--no-codex"],
        "codex-only": ["--no-chemenzy", "--no-guided-chemenzy"],
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
    leakage_audit_pack: str | None,
    resume: bool,
    objective_mode: str = "benchmark_search",
) -> dict[str, Any]:
    snapshot_path = output_root / "snapshots" / "benchmark-snapshot.json"
    seed = _optional_file(self_evo_library_seed, "self_evo_library_seed")
    inventory = _optional_file(inventory_snapshot, "inventory_snapshot")
    benchmark_index = _optional_file(
        benchmark_stock_index, "benchmark_stock_index"
    )
    leakage_pack = _optional_file(leakage_audit_pack, "leakage_audit_pack")
    chemenzy_python = _chemenzy_python(chemenzy_env_prefix)
    provider_snapshot = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "execution_profile": execution_profile,
        "objective_mode": objective_mode,
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
        "leakage_audit_pack_path": str(leakage_pack or ""),
        "leakage_audit_pack_sha256": _file_sha256(leakage_pack) if leakage_pack else "",
        "benchmark_stock_is_a_search_boundary_not_procurement": any(
            str(case.acceptance.get("stock_boundary") or "") == "benchmark_search" for case in cases
        ),
    }
    environment_material = {
        "manifest_sha256": _file_sha256(manifest),
        "selected_case_ids": sorted(case.case_id for case in cases),
        "provider_snapshot": provider_snapshot,
        "knowledge": knowledge,
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
    manifest: Path,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    case_external = output_root / "external" / case.case_id
    case_external.mkdir(parents=True, exist_ok=True)
    seed = _optional_file(self_evo_library_seed, "self_evo_library_seed")
    leakage_pack = _optional_file(leakage_audit_pack, "leakage_audit_pack")
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
            or current.get("panel_snapshot_sha256") != str(snapshot.get("content_sha256") or "")
            or current.get("self_evo_library_sha256") != seed_sha
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
        additional_allowed_paths=([leakage_pack] if leakage_pack else []),
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


def _summarize_report(
    path: Path,
    *,
    elapsed_s: float,
    reused: bool,
    snapshot_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    gate_report = dict(report.get("gates") or {})
    gates = dict(gate_report.get("gates") or {})
    counts = dict(gate_report.get("counts") or {})
    claim = dict(report.get("claim") or {})
    model = dict(report.get("model_cost") or {})
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
        if isinstance(row, Mapping) and row.get("stage") in {"global_campaign", "global_replan"}
    ]
    preflight_case = dict(dict(report.get("preflight") or {}).get("case") or {})
    return {
        "status": "completed",
        "case_id": str(preflight_case.get("case_id") or report.get("run_id") or ""),
        "claim": (
            "benchmark_search_completed"
            if claim.get("benchmark_search_completed") is True
            else str(claim.get("achieved_profile") or "unresolved")
        ),
        "objective_mode": str(claim.get("objective_mode") or "scientific_proof"),
        "objective_achieved": claim.get("objective_achieved") is True,
        "accepted_under_configured_policy": (claim.get("accepted_under_configured_policy") is True),
        "elapsed_s": elapsed_s,
        "reused": reused,
        "run_id": str(report.get("run_id") or ""),
        "run_dir": str(report.get("run_dir") or path.parent),
        "report_path": str(path),
        "snapshot_receipt": dict(snapshot_receipt or {}),
        "model_cost": model,
        "within_resource_budget": bool(
            dict(report.get("resource_envelope") or {}).get("within_budget")
        ),
        "false_closure_claim_count": int(gate_report.get("false_closure_claim_count") or 0),
        "attempt_count": int(report.get("attempt_count") or 0),
        "accepted_expansion_count": int(report.get("accepted_expansion_count") or 0),
        "gate_summary": {
            key.split("_", 1)[0]: bool(value)
            for key, value in gates.items()
            if key[:2] in {"B0", "B1", "B2", "B3", "B4", "B5"}
        },
        "route_counts": counts,
        "planning_depth": dict(report.get("planning_depth") or {}),
        "chemenzy": {
            "status": (
                "completed"
                if any(row.get("status") == "completed" for row in chemenzy_stages)
                else str(baseline.get("status") or "")
            ),
            "frontier_count": sum(int(row.get("frontier_count") or 0) for row in chemenzy_stages),
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
            "proposal_count": sum(int(row.get("proposal_count") or 0) for row in chemenzy_stages),
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
        "evidence": {
            "pass_count": len(evidence_stages),
            "source_count": sum(int(row.get("source_count") or 0) for row in evidence_stages),
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
                int(dict(row.get("validation") or {}).get("accepted_validation_count") or 0)
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
    }


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
    synonym_not_applicable = str(row.get("target_synonym_not_applicable_reason") or "").strip()
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
