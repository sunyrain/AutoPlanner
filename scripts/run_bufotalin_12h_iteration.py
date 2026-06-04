"""Run a long bufotalin-focused ChemEnzy iteration with semisynthesis rescue."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_adapter import ChemEnzyBackendAdapter
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.baselines.proposal_gate import (
    ProposalGateConfig,
    gate_web_route,
    normalize_proposal_gate_mode,
    summarize_route_gate_reports,
)
from cascade_planner.baselines.route_contract import BackendFailure, BaselineRunResult, RouteSearchConfig
from cascade_planner.baselines.semisynthesis_rescue import (
    semisynthesis_open_precursors,
    semisynthesis_rescue_routes,
    semisynthesis_upstream_candidate_precursors,
    stitch_semisynthesis_routes,
)
from cascade_planner.baselines.template_relevance_runtime import DEFAULT_TEMPLATE_RELEVANCE_MODELS
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.cascade_verifier import verify_cascade_route
from scripts.run_chem_enzy_plan_for_web import _web_payload_from_result


BUFOTALIN_TARGET = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
DEFAULT_CYCLE_ROUTE_LIMIT = 100
ENABLE_CONDITION_PREDICTION_FLAG = "enable_condition_prediction.flag"
DEFAULT_FAST_ONE_STEP_MODELS = (
    "graphfp_models.USPTO-full_remapped",
    "onmt_models.bionav_one_step",
    "onmt_models.bionav_native_one_step",
)
BUFOTALIN_MAINLINE_ONE_STEP_MODELS = (
    *DEFAULT_FAST_ONE_STEP_MODELS,
    *DEFAULT_TEMPLATE_RELEVANCE_MODELS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 12-hour bufotalin retrosynthesis iteration.")
    parser.add_argument("--target", default=BUFOTALIN_TARGET)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--max-cycles", type=int, default=-1, help="-1 means no explicit cycle cap; 0 writes only the semisynthesis anchor.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--disable-condition-prediction",
        action="store_true",
        help="Do not enable post-search RCR condition backfill for completed cycles.",
    )
    parser.add_argument("--cycle-config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cycle-output", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cycle-timeout-s", type=int, default=1800)
    args = parser.parse_args()
    if args.cycle_config:
        _run_cycle_worker(args)
        return

    output_root = Path(args.output_root or _default_output_root())
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_runtime_root(output_root, enable_condition_prediction=not args.disable_condition_prediction)
    log_path = output_root / "runner_events.jsonl"
    manifest_path = output_root / "manifest.json"
    heartbeat_path = output_root / "heartbeat.json"
    deadline = time.monotonic() + max(0.01, float(args.hours)) * 3600.0
    started_wall = datetime.now(timezone.utc).isoformat()

    cycles = _cycle_configs(args.target)
    best: list[dict[str, Any]] = []
    event(
        log_path,
        "start",
        {
            "started_at": started_wall,
            "target": args.target,
            "hours": args.hours,
            "output_root": str(output_root),
            "cycles_available": len(cycles),
        },
    )
    _write_heartbeat(heartbeat_path, event="start", cycle="", completed_cycles=0)

    anchor_result = BaselineRunResult(target_smiles=args.target, backend="semisynthesis_anchor")
    anchor_config = cycles[0]["config"]
    anchor_payload = _web_payload_from_result(
        anchor_result,
        _request_payload(args.target, anchor_config, cycles[0]),
        anchor_config,
        0.0,
        vendor_root=Path(args.vendor_root),
    )
    _attach_template_relevance_probe(
        anchor_payload,
        target=args.target,
        anchor_routes=semisynthesis_rescue_routes(args.target),
        vendor_root=Path(args.vendor_root),
        gpu=int(args.gpu),
    )
    _apply_cycle_proposal_gate(anchor_payload)
    _write_cycle_outputs(output_root, "anchor", anchor_payload, None, render=args.render)
    best = _merge_best(best, anchor_payload, cycle_name="anchor")
    event(log_path, "anchor", _payload_summary(anchor_payload))
    if int(args.max_cycles) == 0:
        _write_manifest(
            manifest_path,
            target=args.target,
            started_at=started_wall,
            output_root=output_root,
            completed_cycles=0,
            best=best,
            running=False,
        )
        event(log_path, "finish", {"completed_cycles": 0, "manifest": str(manifest_path)})
        finalize_report = _finalize_result_package(
            output_root,
            vendor_root=Path(args.vendor_root),
            gpu=int(args.gpu),
        )
        event(log_path, "final_candidates_export", finalize_report.get("final_candidates_export") or {})
        event(log_path, "result_package_finalize", finalize_report)
        _write_heartbeat(heartbeat_path, event="finish", cycle="", completed_cycles=0)
        return

    cycle_index = 0
    completed = 0
    while time.monotonic() < deadline:
        if int(args.max_cycles) > 0 and completed >= int(args.max_cycles):
            break
        cycle = cycles[cycle_index % len(cycles)]
        cycle_index += 1
        completed += 1
        cycle_name = f"cycle_{completed:03d}_{cycle['name']}"
        if time.monotonic() >= deadline:
            break
        event(log_path, "cycle_start", {"cycle": cycle_name, "config": cycle["config"].to_dict()})
        _write_heartbeat(heartbeat_path, event="cycle_start", cycle=cycle_name, completed_cycles=completed - 1)
        cycle_started = time.monotonic()
        payload, worker_report = _run_cycle_subprocess(
            output_root,
            cycle_name,
            cycle,
            args,
            deadline=deadline,
            elapsed_s=lambda: time.monotonic() - cycle_started,
        )
        best = _merge_best(best, payload, cycle_name=cycle_name)
        event(log_path, "cycle_finish", {"cycle": cycle_name, **worker_report, **_payload_summary(payload)})
        _write_heartbeat(heartbeat_path, event="cycle_finish", cycle=cycle_name, completed_cycles=completed)
        _write_manifest(
            manifest_path,
            target=args.target,
            started_at=started_wall,
            output_root=output_root,
            completed_cycles=completed,
            best=best,
            running=time.monotonic() < deadline and not (int(args.max_cycles) > 0 and completed >= int(args.max_cycles)),
        )

    _write_manifest(
        manifest_path,
        target=args.target,
        started_at=started_wall,
        output_root=output_root,
        completed_cycles=completed,
        best=best,
        running=False,
    )
    event(log_path, "finish", {"completed_cycles": completed, "manifest": str(manifest_path)})
    finalize_report = _finalize_result_package(
        output_root,
        vendor_root=Path(args.vendor_root),
        gpu=int(args.gpu),
    )
    event(log_path, "final_candidates_export", finalize_report.get("final_candidates_export") or {})
    event(log_path, "result_package_finalize", finalize_report)
    _write_heartbeat(heartbeat_path, event="finish", cycle="", completed_cycles=completed)


def _run_cycle_worker(args: argparse.Namespace) -> None:
    cycle_doc = json.loads(Path(args.cycle_config).read_text(encoding="utf-8"))
    target = str(cycle_doc["target"])
    cycle = dict(cycle_doc["cycle"])
    config = RouteSearchConfig(**cycle_doc["config"])
    config = _adaptive_cycle_config(config, cycle)
    output_dir = Path(args.cycle_output)
    started = time.monotonic()
    condition_prediction_enabled = _condition_prediction_enabled_for_worker(output_dir)
    adapter = ChemEnzyBackendAdapter(
        vendor_root=Path(args.vendor_root),
        gpu=int(args.gpu),
    )
    anchor_routes = semisynthesis_rescue_routes(target)
    open_precursors = semisynthesis_open_precursors(anchor_routes)
    upstream_precursors = semisynthesis_upstream_candidate_precursors(anchor_routes)
    result = BaselineRunResult(target_smiles=target, backend="ChemEnzyRetroPlanner")
    upstream_report: dict[str, Any] = {"enabled": False}
    if cycle.get("upstream_first") and upstream_precursors:
        upstream_config = _upstream_config(upstream_precursors[0], config)
        upstream_started = time.monotonic()
        upstream_result = adapter.run_target(upstream_config)
        stitched = stitch_semisynthesis_routes(anchor_routes, upstream_result)
        if stitched:
            result.routes = [*stitched]
        else:
            result.failures = [*upstream_result.failures]
        upstream_report = {
            "enabled": True,
            "target_smiles": upstream_precursors[0],
            "required_open_precursor": upstream_precursors[0] in open_precursors,
            "elapsed_s": round(time.monotonic() - upstream_started, 3),
            "solved": bool(upstream_result.solved),
            "route_count": upstream_result.route_count,
            "failures": [failure.category for failure in upstream_result.failures],
            "stitched_routes": len(stitched),
            "mode": "upstream_first",
        }
    else:
        result = adapter.run_target(config)
    if upstream_precursors and not result.solved and not cycle.get("upstream_first"):
        upstream_config = _upstream_config(upstream_precursors[0], config)
        upstream_started = time.monotonic()
        upstream_result = adapter.run_target(upstream_config)
        stitched = stitch_semisynthesis_routes(anchor_routes, upstream_result)
        if stitched:
            result.routes = [*stitched, *result.routes]
            result.failures = [failure for failure in result.failures if failure.category != "no_route_found"]
        upstream_report = {
            "enabled": True,
            "target_smiles": upstream_precursors[0],
            "required_open_precursor": upstream_precursors[0] in open_precursors,
            "elapsed_s": round(time.monotonic() - upstream_started, 3),
            "solved": bool(upstream_result.solved),
            "route_count": upstream_result.route_count,
            "failures": [failure.category for failure in upstream_result.failures],
            "stitched_routes": len(stitched),
            "mode": "fallback_after_target",
        }
    _limit_result_routes(result, max_routes=DEFAULT_CYCLE_ROUTE_LIMIT)
    elapsed = time.monotonic() - started
    payload = _web_payload_from_result(
        result,
        _request_payload(target, config, cycle),
        config,
        elapsed,
        vendor_root=Path(args.vendor_root),
    )
    condition_report = _backfill_display_route_conditions(
        payload,
        vendor_root=Path(args.vendor_root),
        enabled=condition_prediction_enabled,
    )
    payload.setdefault("route_set_metrics", {})["semisynthesis_upstream_search"] = upstream_report
    payload.setdefault("ui_metadata", {})["semisynthesis_upstream_search"] = upstream_report
    payload.setdefault("route_set_metrics", {})["condition_prediction"] = condition_report
    payload.setdefault("ui_metadata", {})["condition_prediction"] = condition_report
    _apply_cycle_proposal_gate(payload)
    adaptive_budget = (config.search_flags or {}).get("adaptive_budget")
    if adaptive_budget:
        payload.setdefault("route_set_metrics", {})["adaptive_budget"] = adaptive_budget
        payload.setdefault("ui_metadata", {})["adaptive_budget"] = adaptive_budget
    payload.setdefault("route_set_metrics", {})["cycle_route_limit"] = result.raw_backend_metadata.get("cycle_route_limit")
    payload.setdefault("ui_metadata", {})["cycle_route_limit"] = result.raw_backend_metadata.get("cycle_route_limit")
    _attach_template_relevance_probe(
        payload,
        target=target,
        anchor_routes=anchor_routes,
        vendor_root=Path(args.vendor_root),
        gpu=int(args.gpu),
    )
    _write_payload_files(output_dir, payload, result, render=args.render)
    print(json.dumps(_payload_summary(payload), ensure_ascii=False), flush=True)


def _prepare_runtime_root(output_root: Path, *, enable_condition_prediction: bool = True) -> None:
    tmp_dir = Path(output_root) / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir.resolve()))
    if enable_condition_prediction:
        (Path(output_root) / ENABLE_CONDITION_PREDICTION_FLAG).write_text("1\n", encoding="utf-8")


def _apply_cycle_proposal_gate(payload: dict[str, Any], *, mode: str = "hard_reject") -> dict[str, Any]:
    routes = payload.get("routes")
    normalized_mode = normalize_proposal_gate_mode(mode)
    report: dict[str, Any] = {
        "schema_version": "bufotalin_cycle_proposal_gate.v1",
        "enabled": normalized_mode != "off",
        "mode": normalized_mode,
        "input_routes": len(routes) if isinstance(routes, list) else 0,
        "kept_routes": len(routes) if isinstance(routes, list) else 0,
        "dropped_routes": 0,
        "route_decision_counts": {},
        "reason_counts": {},
        "frontiers": [],
        "dropped": [],
        "description": (
            "12h iteration proposal gate applies the same conservative route-level "
            "material, condition, and unsupported-prenyl screens used by the web UI."
        ),
    }
    payload.setdefault("route_set_metrics", {})["proposal_gate"] = report
    payload.setdefault("ui_metadata", {})["proposal_gate"] = report
    if normalized_mode == "off" or not isinstance(routes, list) or not routes:
        payload["proposal_gate"] = report
        return report

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    repaired_count = 0
    repair_reason_counts: dict[str, int] = {}
    config = ProposalGateConfig(mode=normalized_mode)
    for original_index, route in enumerate(routes):
        if not isinstance(route, dict):
            continue
        route.setdefault("native_rank", original_index)
        route.setdefault("original_route_rank", route.get("route_rank", original_index))
        gate_report = gate_web_route(route, config=config)
        if normalized_mode == "hard_reject" and gate_report.get("hard_reject"):
            repaired_route = _repair_route_frontier_with_semisynthesis_rescue(route, gate_report)
            if repaired_route is not None:
                repaired_report = gate_web_route(repaired_route, config=config)
                if not repaired_report.get("hard_reject"):
                    route = repaired_route
                    gate_report = repaired_report
                    repaired_count += 1
                    reason = str((route.get("frontier_repair") or {}).get("rescue_type") or "semisynthesis_rescue")
                    repair_reason_counts[reason] = repair_reason_counts.get(reason, 0) + 1
        compact = _compact_cycle_proposal_gate(gate_report)
        route["proposal_gate"] = compact
        metrics = route.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics["proposal_gate"] = compact
        route_reports.append(gate_report)
        if normalized_mode == "hard_reject" and gate_report.get("hard_reject"):
            dropped.append({"route": route, "gate": gate_report, "original_index": original_index})
        else:
            kept.append(route)

    for new_rank, route in enumerate(kept):
        route["route_rank"] = new_rank
        route["post_proposal_gate_rank"] = new_rank

    summary = summarize_route_gate_reports(route_reports)
    frontiers = _cycle_proposal_gate_frontiers(dropped)
    report.update(
        {
            "input_routes": len(routes),
            "kept_routes": len(kept),
            "dropped_routes": len(dropped),
            "route_decision_counts": summary.get("route_decision_counts") or {},
            "reason_counts": summary.get("reason_counts") or {},
            "frontiers": frontiers,
            "dropped": [_compact_cycle_dropped_proposal_gate_row(row) for row in dropped[:50]],
            "repaired_routes": repaired_count,
            "repair_reason_counts": dict(sorted(repair_reason_counts.items())),
        }
    )
    payload["routes"] = kept
    payload["n_results"] = len(kept)
    payload["proposal_gate"] = report
    payload.setdefault("route_set_metrics", {})["proposal_gate"] = report
    payload.setdefault("ui_metadata", {})["proposal_gate"] = report
    if routes and not kept and dropped:
        search_status = payload.setdefault("search_status", {})
        search_status["status"] = "frontier"
        search_status["proposal_gate_removed_all"] = True
        search_status["message"] = "Proposal gate removed all displayed candidates before cycle payload export."
        payload["frontiers"] = frontiers
        _append_unique(payload.setdefault("failure_diagnosis", []), "proposal_gate_filtered_all")
        analysis = payload.setdefault("failure_analysis", {})
        categories = analysis.setdefault("failure_categories", [])
        _append_unique(categories, "proposal_gate_filtered_all")
    return report


def _repair_route_frontier_with_semisynthesis_rescue(
    route: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any] | None:
    frontier = gate_report.get("frontier") or {}
    product = str(frontier.get("smiles") or "")
    rejected_step = frontier.get("rejected_step")
    if not product or rejected_step is None:
        return None
    try:
        step_index = int(rejected_step)
    except (TypeError, ValueError):
        return None
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    if step_index < 0 or step_index >= len(steps):
        return None
    for rescue_route in semisynthesis_rescue_routes(product):
        metadata = rescue_route.raw_backend_metadata or {}
        if not metadata.get("advanced_precursor_source_supported"):
            continue
        if len(rescue_route.steps) != 1:
            continue
        rescue_step = rescue_route.steps[0]
        if canonical_smiles(rescue_step.product_smiles) != canonical_smiles(product):
            continue
        repaired = copy.deepcopy(route)
        repaired_steps = [step for step in repaired.get("steps") or [] if isinstance(step, dict)]
        old_step = copy.deepcopy(repaired_steps[step_index])
        repaired_steps[step_index] = _rescue_step_to_web_step(rescue_step, index=step_index, old_step=old_step)
        repaired["steps"] = repaired_steps
        repaired["n_steps"] = len(repaired_steps)
        repaired["frontier_repair"] = {
            "schema_version": "bufotalin_frontier_repair.v1",
            "strategy": "replace_rejected_frontier_step_with_source_supported_semisynthesis",
            "rejected_step": step_index,
            "frontier_smiles": product,
            "old_reaction_smiles": old_step.get("reaction_smiles") or old_step.get("rxn_smiles") or "",
            "new_reaction_smiles": rescue_step.rxn_smiles,
            "rescue_type": metadata.get("rescue_type"),
            "source_record": metadata.get("advanced_precursor_record") or {},
        }
        _refresh_repaired_route_metrics(repaired)
        return repaired
    return None


def _rescue_step_to_web_step(
    step: Any,
    *,
    index: int,
    old_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reactants = [str(item) for item in step.reactant_smiles or [] if str(item or "")]
    main = reactants[0] if reactants else ""
    aux = reactants[1:]
    condition_predictions = [dict(row) for row in step.condition_predictions or [] if isinstance(row, dict)]
    first_condition = condition_predictions[0] if condition_predictions else {}
    return {
        "index": index,
        "product": step.product_smiles,
        "main_reactant": main,
        "aux_reactants": aux,
        "reaction_smiles": step.rxn_smiles,
        "reaction_type": step.source_model,
        "ec": "",
        "enzyme_uid": "",
        "catalyst": first_condition.get("Catalyst") or first_condition.get("Base") or "",
        "T": first_condition.get("Temperature"),
        "pH": first_condition.get("pH"),
        "solvent": first_condition.get("Solvent") or "",
        "condition_predictions": condition_predictions,
        "enzyme_ec_annotations": [],
        "evidence": {},
        "source": step.source_model,
        "scores": {"step": step.score, "condition": first_condition.get("Score")},
        "fixed_fields": (old_step or {}).get("fixed_fields") or {},
        "is_filled": bool((old_step or {}).get("is_filled", True)),
        "is_enzymatic": False,
        "stock_status": dict(step.stock_status or {}),
        "reaction_interpretation": {
            "role": "frontier repair from source-supported semisynthesis rescue",
            "catalysis_and_conditions": first_condition.get("condition_label") or "",
        },
        "candidate_pool": (old_step or {}).get("candidate_pool") or {},
        "raw_backend_metadata": dict(step.raw_backend_metadata or {}),
    }


def _refresh_repaired_route_metrics(route: dict[str, Any]) -> None:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    produced = {canonical_smiles(str(step.get("product") or "")) for step in steps if step.get("product")}
    produced.discard("")
    terminal_reactants: list[str] = []
    terminal_stock: dict[str, bool] = {}
    seen: set[str] = set()
    for step in steps:
        stock = step.get("stock_status") or {}
        reactants = []
        main = str(step.get("main_reactant") or "")
        if main:
            reactants.append(main)
        reactants.extend(str(item) for item in step.get("aux_reactants") or [] if str(item or ""))
        for reactant in reactants:
            canonical = canonical_smiles(reactant)
            if not canonical or canonical in produced or canonical in seen:
                continue
            seen.add(canonical)
            terminal_reactants.append(reactant)
            terminal_stock[reactant] = bool(stock.get(reactant, False))
    verifier = verify_cascade_route(route).to_dict()
    metrics = route.setdefault("metrics", {})
    metrics["terminal_reactants"] = terminal_reactants
    metrics["terminal_stock_status"] = terminal_stock
    metrics["strict_stock_solve"] = bool(terminal_reactants and all(terminal_stock.values()))
    metrics["cascade_verifier"] = verifier
    metrics["cascade_compatibility"] = {
        "cascade_compatibility_success": bool(verifier.get("feasible")),
        "score": verifier.get("score"),
        "issues": sorted((verifier.get("reason_counts") or {}).keys()),
        "reason_counts": verifier.get("reason_counts") or {},
        "verifier_contract": "rule-checkable cascade verifier; high precision for perturbation labels, not an expert feasibility oracle",
    }
    metrics["frontier_repaired_semisynthesis"] = True
    metrics["route_solved"] = bool(metrics["strict_stock_solve"] and verifier.get("feasible"))


def _compact_cycle_proposal_gate(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "decision": report.get("decision"),
        "hard_reject": bool(report.get("hard_reject")),
        "mode": report.get("mode"),
        "step_count": report.get("step_count"),
        "rejected_step_count": report.get("rejected_step_count"),
        "route_hard_reasons": report.get("route_hard_reasons") or [],
        "reason_counts": report.get("reason_counts") or {},
        "frontier": report.get("frontier"),
    }


def _compact_cycle_dropped_proposal_gate_row(item: dict[str, Any]) -> dict[str, Any]:
    route = item.get("route") or {}
    gate = item.get("gate") or {}
    return {
        "route_rank": route.get("original_route_rank", route.get("route_rank")),
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "reason_counts": gate.get("reason_counts") or {},
        "frontier": gate.get("frontier"),
    }


def _cycle_proposal_gate_frontiers(dropped: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    frontiers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in dropped:
        gate = item.get("gate") or {}
        frontier = gate.get("frontier") or {}
        smiles = str(frontier.get("smiles") or "")
        reason = str(frontier.get("reason") or "")
        key = f"{smiles}|{reason}"
        if not smiles or key in seen:
            continue
        seen.add(key)
        frontiers.append(dict(frontier))
        if len(frontiers) >= int(limit):
            break
    return frontiers


def _append_unique(rows: list[Any], value: Any) -> None:
    if value not in rows:
        rows.append(value)


def _run_cycle_subprocess(
    output_root: Path,
    cycle_name: str,
    cycle: dict[str, Any],
    args: argparse.Namespace,
    *,
    deadline: float,
    elapsed_s,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle_dir = output_root / cycle_name
    cycle_dir.mkdir(parents=True, exist_ok=True)
    config_path = cycle_dir / "cycle_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target": args.target,
                "cycle": {key: value for key, value in cycle.items() if key != "config"},
                "config": cycle["config"].to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    worker_log = cycle_dir / "worker.log"
    timeout_s = min(max(1, int(args.cycle_timeout_s)), max(1, int(deadline - time.monotonic())))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cycle-config",
        str(config_path),
        "--cycle-output",
        str(cycle_dir),
        "--vendor-root",
        str(args.vendor_root),
        "--gpu",
        str(int(args.gpu)),
        "--cycle-timeout-s",
        str(timeout_s),
    ]
    if args.render:
        cmd.append("--render")
    with worker_log.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = -9
            timed_out = True
            log.write(f"\nworker timed out after {timeout_s} s\n")

    payload_path = cycle_dir / "web_payload.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        failure = BackendFailure(
            category="cycle_worker_timeout" if timed_out else "cycle_worker_failed",
            message=f"worker did not produce web_payload.json; returncode={returncode}",
            target_smiles=args.target,
            retryable=True,
            raw_backend_metadata={
                "cycle": cycle_name,
                "worker_log": str(worker_log),
                "returncode": returncode,
                "timed_out": timed_out,
            },
        )
        result = BaselineRunResult(
            target_smiles=args.target,
            backend="ChemEnzyRetroPlanner",
            failures=[failure],
        )
        payload = _web_payload_from_result(
            result,
            _request_payload(args.target, cycle["config"], cycle),
            cycle["config"],
            float(elapsed_s()),
            vendor_root=Path(args.vendor_root),
        )
        _apply_cycle_proposal_gate(payload)
        _write_payload_files(cycle_dir, payload, result, render=args.render)
    return payload, {
        "worker_returncode": returncode,
        "worker_timed_out": timed_out,
        "worker_log": str(worker_log),
    }


def _condition_prediction_enabled_for_worker(output_dir: Path | str) -> bool:
    """Allow a running long-loop master to enable condition prediction for new workers."""
    cycle_dir = Path(output_dir)
    root = cycle_dir.parent
    return (root / ENABLE_CONDITION_PREDICTION_FLAG).exists()


def _backfill_display_route_conditions(
    payload: dict[str, Any],
    *,
    vendor_root: Path,
    enabled: bool,
    max_routes: int = 10,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "model": "rcr" if enabled else None,
        "activation": ENABLE_CONDITION_PREDICTION_FLAG if enabled else None,
        "mode": "post_payload_renderable_routes",
        "routes_considered": 0,
        "steps_attempted": 0,
        "steps_filled": 0,
        "failures": [],
    }
    if not enabled:
        return report
    predictor = None
    for route in payload.get("routes") or []:
        if report["routes_considered"] >= max(1, int(max_routes)):
            break
        if not _condition_backfill_candidate(route):
            continue
        report["routes_considered"] += 1
        for step in route.get("steps") or []:
            if not isinstance(step, dict) or step.get("condition_predictions"):
                continue
            rxn = str(step.get("reaction_smiles") or "")
            if ">>" not in rxn:
                continue
            report["steps_attempted"] += 1
            try:
                if predictor is None:
                    predictor = _load_rcr_condition_predictor(vendor_root)
                condition = _predict_rcr_condition(predictor, rxn)
            except Exception as exc:
                report["failures"].append(f"{type(exc).__name__}: {exc}")
                continue
            if condition:
                step["condition_predictions"] = [condition]
                report["steps_filled"] += 1
    return report


def _condition_backfill_candidate(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    if metrics.get("source_supported_semisynthesis"):
        return True
    verifier = metrics.get("cascade_verifier") or {}
    return bool(verifier.get("feasible"))


def _load_rcr_condition_predictor(vendor_root: Path | str):
    from cascade_planner.baselines.chem_enzy_adapter import (
        _patch_dgl_graphbolt_optional_import,
        _patch_numpy_legacy_aliases,
        _patch_optional_easifa_import,
        _patch_optional_graphviz_import,
        _patch_torchdata_legacy_aliases,
        _vendor_pythonpath,
    )

    vendor_root = Path(vendor_root)
    if not vendor_root.is_absolute():
        vendor_root = (ROOT / vendor_root).resolve()
    info_path = (
        vendor_root
        / "retro_planner"
        / "packages"
        / "condition_predictor"
        / "condition_predictor"
        / "data"
    )
    with _vendor_pythonpath(vendor_root):
        _patch_numpy_legacy_aliases()
        _patch_torchdata_legacy_aliases()
        _patch_dgl_graphbolt_optional_import()
        _patch_optional_easifa_import(False)
        _patch_optional_graphviz_import(False)
        from condition_predictor.condition_model import NeuralNetContextRecommender
        predictor = NeuralNetContextRecommender()
        predictor.load_nn_model(
            info_path=str(info_path),
            weights_path=str(info_path / "dict_weights.npy"),
        )
        return predictor


def _predict_rcr_condition(predictor: Any, rxn_smiles: str) -> dict[str, Any]:
    contexts, scores = predictor.get_n_conditions(rxn_smiles, n=1, return_scores=True)
    if not contexts:
        return {}
    row = list(contexts[0] or [])
    if len(row) < 4:
        return {}
    return {
        "Temperature": _json_safe_float(row[0]),
        "Solvent": "" if row[1] != row[1] else str(row[1] or ""),
        "Reagent": "" if row[2] != row[2] else str(row[2] or ""),
        "Catalyst": "" if row[3] != row[3] else str(row[3] or ""),
        "Score": _json_safe_float(scores[0] if len(scores) else None),
        "condition_label": "RCR model prediction",
        "note": "Post-search RCR condition prediction for display-gated route.",
    }


def _json_safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _limit_result_routes(result: BaselineRunResult, *, max_routes: int) -> None:
    limit = max(1, int(max_routes))
    original_count = len(result.routes or [])
    if original_count <= limit:
        result.raw_backend_metadata = {
            **(result.raw_backend_metadata or {}),
            "cycle_route_limit": {
                "applied": False,
                "max_routes": limit,
                "original_route_count": original_count,
                "kept_route_count": original_count,
            },
        }
        return
    result.routes = list(result.routes or [])[:limit]
    result.raw_backend_metadata = {
        **(result.raw_backend_metadata or {}),
        "cycle_route_limit": {
            "applied": True,
            "max_routes": limit,
            "original_route_count": original_count,
            "kept_route_count": len(result.routes),
            "reason": "keep long iteration payloads compact; full native route count is preserved in metadata",
        },
    }


def _attach_template_relevance_probe(
    payload: dict[str, Any],
    *,
    target: str,
    anchor_routes: list[Any],
    vendor_root: Path,
    gpu: int,
    top_k: int = 10,
) -> dict[str, Any]:
    expected_precursors = _semisynthesis_anchor_precursors(anchor_routes)
    expected_precursor = expected_precursors[0] if expected_precursors else ""
    report: dict[str, Any] = {
        "enabled": True,
        "models": list(DEFAULT_TEMPLATE_RELEVANCE_MODELS),
        "target": target,
        "top_k": int(top_k),
        "expected_precursor": expected_precursor,
        "hit_expected_precursor": False,
        "rows": [],
    }
    try:
        provider = ChemEnzyOneStepProposalProvider(
            vendor_root=vendor_root,
            models=tuple(DEFAULT_TEMPLATE_RELEVANCE_MODELS),
            expansion_topk=max(1, int(top_k)),
            gpu=int(gpu),
        )
        rows = provider.predict(target, top_k=max(1, int(top_k)))
    except Exception as exc:
        rows = []
        report["error"] = f"{type(exc).__name__}: {exc}"
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        main = canonical_smiles(str(row.get("main_reactant") or ""))
        normalized = {
            "rank": row.get("rank"),
            "score": row.get("score"),
            "main_reactant": row.get("main_reactant"),
            "main_reactant_canonical": main,
            "source": row.get("source"),
            "model_full_name": row.get("model_full_name"),
            "aux_reactants": row.get("aux_reactants") or [],
            "hits_expected_precursor": bool(expected_precursor and main == expected_precursor),
        }
        if normalized["hits_expected_precursor"]:
            report["hit_expected_precursor"] = True
        normalized_rows.append(normalized)
    report["rows"] = normalized_rows
    report["returned"] = len(normalized_rows)
    payload.setdefault("route_set_metrics", {})["template_relevance_top_level_probe"] = report
    payload.setdefault("ui_metadata", {})["template_relevance_top_level_probe"] = report
    return report


def _semisynthesis_anchor_precursors(anchor_routes: list[Any]) -> list[str]:
    """Return advanced semisynthesis precursors, including source-supported ones."""
    precursors = [canonical_smiles(item) for item in semisynthesis_open_precursors(anchor_routes)]
    seen = {item for item in precursors if item}
    for route in anchor_routes or []:
        if not (getattr(route, "raw_backend_metadata", {}) or {}).get("rescue_type"):
            continue
        for step in getattr(route, "steps", []) or []:
            metadata = getattr(step, "raw_backend_metadata", {}) or {}
            rescue = metadata.get("semisynthesis_rescue") or {}
            forward_reagent = canonical_smiles(str(rescue.get("forward_reagent") or ""))
            for reactant in getattr(step, "reactant_smiles", []) or []:
                canonical = canonical_smiles(str(reactant or ""))
                if not canonical or canonical == forward_reagent or canonical in seen:
                    continue
                seen.add(canonical)
                precursors.append(canonical)
    return [item for item in precursors if item]


def _cycle_configs(target: str) -> list[dict[str, Any]]:
    base_flags = {"keep_search": True, "include_cascade_expansion_trace": True}
    mainline_models = list(BUFOTALIN_MAINLINE_ONE_STEP_MODELS)
    stock_sets = [
        ("zinc", ["Zinc_Fix-stock"]),
        ("n5", ["PaRotes_n5-stock"]),
        ("default_stock", []),
    ]
    schedules = [
        ("d20_i200_k100", 20, 200, 100),
        ("d24_i300_k150", 24, 300, 150),
        ("d30_i500_k200", 30, 500, 200),
    ]
    cycles: list[dict[str, Any]] = []
    upstream_models = [("upstream_mainline", mainline_models)]
    for model_name, models in upstream_models:
        for stock_name, stocks in stock_sets:
            cycles.append(
                {
                    "name": f"upstream_first_d16_i200_k100_{model_name}_{stock_name}",
                    "model_mode": model_name,
                    "stock_mode": stock_name,
                    "upstream_first": True,
                    "config": RouteSearchConfig(
                        target_smiles=target,
                        stock_names=stocks,
                        max_iterations=200,
                        max_depth=16,
                        expansion_topk=100,
                        one_step_models=list(models),
                        search_flags=dict(base_flags),
                    ),
                }
            )
    for schedule_name, depth, iterations, topk in schedules:
        for stock_name, stocks in stock_sets:
            name = f"{schedule_name}_default_{stock_name}"
            cycles.append(
                {
                    "name": name,
                    "model_mode": "mainline",
                    "stock_mode": stock_name,
                    "config": RouteSearchConfig(
                        target_smiles=target,
                        stock_names=stocks,
                        max_iterations=iterations,
                        max_depth=depth,
                        expansion_topk=topk,
                        one_step_models=list(mainline_models),
                        search_flags=dict(base_flags),
                    ),
                }
            )
    return cycles


def _upstream_config(target: str, config: RouteSearchConfig) -> RouteSearchConfig:
    return RouteSearchConfig(
        target_smiles=target,
        stock_names=list(config.stock_names or []),
        max_iterations=min(max(50, int(config.max_iterations)), 200),
        max_depth=min(max(8, int(config.max_depth)), 16),
        expansion_topk=min(max(50, int(config.expansion_topk)), 100),
        one_step_models=list(config.one_step_models or []),
        search_flags=dict(config.search_flags or {}),
    )


def _adaptive_cycle_config(config: RouteSearchConfig, cycle: dict[str, Any]) -> RouteSearchConfig:
    """Keep CPU-only template ensemble cycles inside the worker timeout envelope."""
    if bool(cycle.get("upstream_first")):
        return config
    stock_mode = str(cycle.get("stock_mode") or "")
    target_iterations = 160 if stock_mode == "n5" and int(config.max_depth or 0) >= 20 else 200
    if (
        int(config.max_iterations or 0) <= target_iterations
        and int(config.max_depth or 0) <= 20
        and int(config.expansion_topk or 0) <= 100
    ):
        return config
    flags = dict(config.search_flags or {})
    original = {
        "max_iterations": int(config.max_iterations or 0),
        "max_depth": int(config.max_depth or 0),
        "expansion_topk": int(config.expansion_topk or 0),
    }
    flags["adaptive_budget"] = {
        "enabled": True,
        "reason": (
            "large CPU template-ensemble cycles approached or exceeded the 20 min worker envelope; "
            "keep long-run iterations productive."
        ),
        "original": original,
        "applied": {
            "max_iterations": target_iterations,
            "max_depth": 20,
            "expansion_topk": 100,
        },
    }
    return RouteSearchConfig(
        target_smiles=config.target_smiles,
        stock_names=list(config.stock_names or []),
        max_iterations=target_iterations,
        max_depth=20,
        expansion_topk=100,
        one_step_models=list(config.one_step_models or []),
        search_flags=flags,
    )


def _request_payload(target: str, config: RouteSearchConfig, cycle: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_smiles": target,
        "search_preset": "thorough",
        "max_steps": config.max_depth,
        "chem_enzy_iterations": config.max_iterations,
        "chem_enzy_expansion_topk": config.expansion_topk,
        "stock_names": config.stock_names,
        "stock_mode": cycle.get("stock_mode"),
        "one_step_models": config.one_step_models,
        "enable_semisynthesis_rescue": True,
        "enable_condition_prediction": False,
        "enable_enzyme_assignment": False,
    }


def _write_cycle_outputs(
    output_root: Path,
    cycle_name: str,
    payload: dict[str, Any],
    result: BaselineRunResult | None,
    *,
    render: bool,
) -> None:
    cycle_dir = output_root / cycle_name
    _write_payload_files(cycle_dir, payload, result, render=render)


def _write_payload_files(
    cycle_dir: Path,
    payload: dict[str, Any],
    result: BaselineRunResult | None,
    *,
    render: bool,
) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    (cycle_dir / "web_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if result is not None:
        (cycle_dir / "backend_result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if render and payload.get("routes"):
        fig_dir = cycle_dir / "figures"
        subprocess.run(
            [
                sys.executable,
                "scripts/render_linear_route_schemes.py",
                "--input",
                str(cycle_dir / "web_payload.json"),
                "--output-dir",
                str(fig_dir),
                "--top-k",
                "5",
                "--formats",
                "svg,pdf",
                "--steps-per-row",
                "3",
                "--aux-mode",
                "mini",
                "--only-feasible",
            ],
            check=False,
        )


def _merge_best(best: list[dict[str, Any]], payload: dict[str, Any], *, cycle_name: str) -> list[dict[str, Any]]:
    rows = list(best)
    for route in payload.get("routes") or []:
        if not isinstance(route, dict):
            continue
        metrics = route.get("metrics") or {}
        signature = "|".join(str(step.get("reaction_smiles") or "") for step in route.get("steps") or [])
        if not signature:
            continue
        rows.append(
            {
                "cycle": cycle_name,
                "signature": signature,
                "n_steps": route.get("n_steps"),
                "score": route.get("score"),
                "route_solved": bool(metrics.get("route_solved")),
                "semisynthesis_anchor": bool(metrics.get("semisynthesis_anchor")),
                "cascade_feasible": bool((metrics.get("cascade_verifier") or {}).get("feasible")),
                "route_rank": route.get("route_rank"),
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        old = deduped.get(row["signature"])
        if old is None or _best_key(row) > _best_key(old):
            deduped[row["signature"]] = row
    return sorted(deduped.values(), key=_best_key, reverse=True)[:20]


def _best_key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        1 if row.get("route_solved") else 0,
        1 if row.get("cascade_feasible") else 0,
        1 if row.get("semisynthesis_anchor") else 0,
        float(row.get("score") or 0.0),
    )


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(payload.get("ok")),
        "n_results": int(payload.get("n_results") or 0),
        "status": (payload.get("search_status") or {}).get("status"),
        "native_raw_n_routes": (payload.get("search_status") or {}).get("native_raw_n_routes"),
        "semisynthesis_rescue_n_routes": (payload.get("search_status") or {}).get("semisynthesis_rescue_n_routes"),
        "backend_failures": [row.get("category") for row in payload.get("backend_failures") or [] if isinstance(row, dict)],
    }


def _write_manifest(
    path: Path,
    *,
    target: str,
    started_at: str,
    output_root: Path,
    completed_cycles: int,
    best: list[dict[str, Any]],
    running: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "target": target,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "output_root": str(output_root),
                "completed_cycles": completed_cycles,
                "running": running,
                "best": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def event(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(), "event": event_type, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _write_heartbeat(path: Path, *, event: str, cycle: str, completed_cycles: int) -> None:
    path.write_text(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "cycle": cycle,
                "completed_cycles": int(completed_cycles),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _export_final_candidates(output_root: Path) -> dict[str, Any]:
    output_dir = output_root / "final_candidates"
    cmd = [
        sys.executable,
        "scripts/export_bufotalin_final_candidates.py",
        str(output_root),
        "--output-dir",
        str(output_dir),
        "--top-native",
        "5",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    report: dict[str, Any] = {
        "enabled": True,
        "returncode": completed.returncode,
        "output_dir": str(output_dir),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    summary_path = output_dir / "final_candidates.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report.update(
                {
                    "high_confidence_count": int(summary.get("high_confidence_count") or 0),
                    "native_review_only_count": int(summary.get("native_review_only_count") or 0),
                    "selected_count": int(summary.get("selected_count") or 0),
                    "excluded_route_count": int(summary.get("excluded_route_count") or 0),
                }
            )
        except Exception as exc:
            report["summary_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _finalize_result_package(output_root: Path, *, vendor_root: Path, gpu: int) -> dict[str, Any]:
    """Write the review/audit package expected from a completed long run."""
    report: dict[str, Any] = {
        "schema_version": "bufotalin_result_package_finalize.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "steps": [],
    }
    report["final_candidates_export"] = _export_final_candidates(output_root)
    commands = [
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/summarize_bufotalin_proposal_gate.py",
                str(output_root),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/analyze_bufotalin_proposal_frontiers.py",
                str(output_root),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/probe_bufotalin_frontier_proposals.py",
                str(output_root),
                "--top-frontiers",
                "1",
                "--top-k",
                "10",
                "--vendor-root",
                str(vendor_root),
                "--gpu",
                str(int(gpu)),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/audit_bufotalin_final_candidate_quality.py",
                str(output_root),
                "--output",
                str(output_root / "final_candidate_quality_audit.json"),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/write_bufotalin_status_snapshot.py",
                str(output_root),
            ],
        },
        {
            "required": False,
            "command": [
                sys.executable,
                "scripts/audit_bufotalin_early_stop_review.py",
                str(output_root),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/write_bufotalin_package_manifest.py",
                str(output_root),
            ],
        },
        {
            "required": True,
            "command": [
                sys.executable,
                "scripts/write_bufotalin_completion_gap_report.py",
                str(output_root),
            ],
        },
    ]
    for item in commands:
        command = item["command"]
        step = _run_finalize_command(command, output_root=output_root)
        step["required"] = bool(item.get("required"))
        report["steps"].append(step)
        if command[1].endswith("audit_bufotalin_early_stop_review.py") and step.get("stdout"):
            (output_root / "early_stop_review_audit.json").write_text(
                str(step.get("stdout") or ""),
                encoding="utf-8",
            )
    report["ok"] = all(
        int(step.get("returncode") or 0) == 0 or not bool(step.get("required"))
        for step in report["steps"]
    )
    report_path = output_root / "result_package_finalize.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["path"] = str(report_path)
    return report


def _run_finalize_command(command: list[str], *, output_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    rel_command = [
        str(item).replace(str(ROOT) + "/", "")
        for item in command
    ]
    return {
        "command": rel_command,
        "returncode": completed.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _default_output_root() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"results/shared/bufotalin_12h_{stamp}"


if __name__ == "__main__":
    main()
