#!/usr/bin/env python3
"""Run the formal nine-statin native-first ChemEnzy -> Codex literature path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.route_package import write_json
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.agent.statin_panel import DEFAULT_STATIN_SUMMARY, load_statin_panel_targets
from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit


RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_STATIN_SUMMARY))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--targets", default="", help="Comma-separated target names/safe ids. Empty means all nine.")
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--expansion-topk", type=int, default=100)
    parser.add_argument("--query-budget", type=int, default=10)
    parser.add_argument(
        "--literature-backend",
        default="codex",
        choices=["codex", "api_json", "pubmed", "local_pubmed", "local", "manual"],
        help="Backend used only after native ChemEnzy/audit triggers literature mode.",
    )
    parser.add_argument("--worker-timeout-s", type=float, default=180.0)
    parser.add_argument("--worker-max-output-bytes", type=int, default=200_000)
    parser.add_argument("--worker-max-tool-calls", type=int, default=8)
    args = parser.parse_args()

    run_root = Path(args.output_dir) if args.output_dir else _default_output_dir(args.literature_backend)
    run_root.mkdir(parents=True, exist_ok=True)
    selected = _selected_targets(args.targets)
    targets = [
        target for target in load_statin_panel_targets(args.summary)
        if not selected or target.safe in selected or target.name.lower() in selected
    ]
    run_input = {
        "schema_version": "statin_panel_formal_native_codex_input.v1",
        "created_at_utc": _utc_now(),
        "summary": str(args.summary),
        "output_dir": str(run_root),
        "target_count": len(targets),
        "target_order": [target.safe for target in targets],
        "native_first_policy": (
            "Run ChemEnzy native retrosynthesis first. Enter Codex literature mode only after "
            "native failure, unclosed route, fake closure risk, route audit failure, or advanced frontier."
        ),
        "chem_enzy": {
            "vendor_root": args.vendor_root,
            "gpu": int(args.gpu),
            "iterations": int(args.iterations),
            "max_depth": int(args.max_depth),
            "expansion_topk": int(args.expansion_topk),
            "stock_names": DEFAULT_STOCKS,
            "one_step_models": DEFAULT_ONE_STEP_MODELS,
        },
        "worker": {
            "literature_backend": args.literature_backend,
            "timeout_s": float(args.worker_timeout_s),
            "max_output_bytes": int(args.worker_max_output_bytes),
            "max_tool_calls": int(args.worker_max_tool_calls),
        },
    }
    write_json(run_root / "run_input.json", run_input)

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        row = _run_target(target, args=args, run_root=run_root, panel_index=index)
        rows.append(row)
        write_json(run_root / "panel_progress.json", _panel_manifest(run_root, run_input, rows, complete=False))

    manifest = _panel_manifest(run_root, run_input, rows, complete=True)
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


def _run_target(target: Any, *, args: argparse.Namespace, run_root: Path, panel_index: int) -> dict[str, Any]:
    started = time.monotonic()
    target_dir = run_root / target.safe
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "target_input.json", {
        "schema_version": "statin_panel_formal_target_input.v1",
        "panel_index": panel_index,
        "target": target.to_dict(),
        "started_at_utc": _utc_now(),
    })

    native_result = _run_native_chemenzy(target, args=args)
    write_json(target_dir / "chemenzy_native_raw_result.json", native_result.to_dict())
    baseline_routes = _baseline_routes_from_result(native_result)
    baseline_routes = _apply_product_feasibility_audit(target, baseline_routes)
    write_json(target_dir / "chemenzy_baseline_routes.json", baseline_routes)

    stuck_frontier = "" if baseline_routes.get("solved") else (_first_stuck_frontier(baseline_routes) or target.target_smiles)
    workflow_dir = target_dir / "smiles_first_after_native"
    workflow_result = run_smiles_first_workflow(
        SmilesFirstWorkflowConfig(
            target_smiles=target.target_smiles,
            target_name=f"{target.safe}_formal_native_codex",
            family_hint=_family_hint(target),
            objective="route closure after native ChemEnzy audit; external research only after trigger gates",
            output_dir=workflow_dir,
            frontier_smiles=stuck_frontier,
            baseline_json=target_dir / "chemenzy_baseline_routes.json",
            query_budget=int(args.query_budget),
            literature_backend=str(args.literature_backend),
            worker_timeout_s=float(args.worker_timeout_s),
            worker_max_output_bytes=int(args.worker_max_output_bytes),
            worker_max_tool_calls=int(args.worker_max_tool_calls),
        )
    )
    write_json(target_dir / "smiles_first_workflow_result.json", workflow_result)

    summary = _target_summary(
        target=target,
        target_dir=target_dir,
        workflow_dir=workflow_dir,
        baseline_routes=baseline_routes,
        stuck_frontier=stuck_frontier,
        workflow_result=workflow_result,
        elapsed_s=time.monotonic() - started,
    )
    write_json(target_dir / "target_manifest.json", summary)
    return summary


def _run_native_chemenzy(target: Any, *, args: argparse.Namespace) -> Any:
    config = RouteSearchConfig(
        target_smiles=target.target_smiles,
        stock_names=DEFAULT_STOCKS,
        max_iterations=int(args.iterations),
        max_depth=int(args.max_depth),
        expansion_topk=int(args.expansion_topk),
        one_step_models=DEFAULT_ONE_STEP_MODELS,
        search_flags={
            "gpu": int(args.gpu),
            "keep_search": True,
            "include_cascade_expansion_trace": True,
            "cascade_expansion_trace_preview": 50,
            "condition_model": "rcr",
            "chem_enzy_onmt_tokenizer": "char",
            "cascade_search_context": {
                "enabled": True,
                "target_smiles": target.target_smiles,
                "target_name": target.name,
                "target_safe": target.safe,
                "domain": "chemoenzymatic",
                "search_preset": "formal_statin_panel_native_first",
            },
        },
    )
    adapter = ChemEnzyBackendAdapter(vendor_root=Path(args.vendor_root), gpu=int(args.gpu))
    return adapter.run_target(config, dry_run=False)


def _baseline_routes_from_result(result: Any) -> dict[str, Any]:
    raw = result.to_dict()
    routes = []
    ordinary_steps = []
    raw_solved_routes = 0
    accepted_solved_routes = 0
    for route in raw.get("routes") or []:
        route = dict(route)
        unresolved = _unresolved_frontiers_from_route(route)
        route["unresolved_frontiers"] = unresolved
        route_audit = _audit_native_route(route)
        route["route_audit"] = route_audit
        if route.get("solved"):
            raw_solved_routes += 1
        if route_audit["accepted_solved_route"]:
            accepted_solved_routes += 1
        routes.append(route)
        if not ordinary_steps:
            ordinary_steps = list(route.get("steps") or [])
    failures = list(raw.get("failures") or [])
    raw_solved = bool(raw.get("solved"))
    solved = accepted_solved_routes > 0
    reasons = []
    if failures:
        reasons.extend(str(item.get("category") or "backend_failure") for item in failures if isinstance(item, dict))
    if not solved:
        reasons.append("native_chemenzy_no_audited_stock_closed_route")
    if not routes:
        reasons.append("native_chemenzy_no_route")
    if raw_solved and not solved:
        reasons.append("native_chemenzy_fake_closure_rejected")
    return {
        "schema_version": "baseline_routes.v1",
        "status": "completed" if not failures else "completed_with_failures",
        "backend": raw.get("backend") or "ChemEnzyRetroPlanner",
        "solved": solved,
        "stock_audit_passed": solved,
        "raw_backend_solved": raw_solved,
        "raw_solved_route_count": raw_solved_routes,
        "accepted_solved_route_count": accepted_solved_routes,
        "route_count": len(routes),
        "routes": routes,
        "ordinary_steps": ordinary_steps,
        "failures": failures,
        "route_audit": {
            "schema_version": "route_audit_report.v1",
            "route_status": "solved" if solved else "fake_closed_rejected" if raw_solved else "unresolved",
            "stock_audit_passed": solved,
            "fake_closure_rejected": raw_solved and not solved,
            "reasons": sorted(set(reasons)),
        },
        "raw_backend_metadata": raw.get("raw_backend_metadata") or {},
    }


def _apply_product_feasibility_audit(target: Any, baseline_routes: dict[str, Any]) -> dict[str, Any]:
    audit = build_product_route_feasibility_audit({
        "targets": [
            {
                "target_id": target.safe,
                "target_smiles": target.target_smiles,
                "routes": baseline_routes.get("routes") or [],
            }
        ]
    })
    target_audit = (audit.get("targets") or [{}])[0]
    routes_by_rank = {
        int(route.get("rank") or 0): route
        for route in target_audit.get("routes") or []
        if isinstance(route, dict)
    }
    product_like_frontiers: list[str] = []
    routes = []
    for rank, route in enumerate(baseline_routes.get("routes") or [], start=1):
        route = dict(route)
        route_product_audit = dict(routes_by_rank.get(rank) or {})
        route["product_route_feasibility_audit"] = route_product_audit
        terminal_profile = route_product_audit.get("terminal_profile") or {}
        tags = {str(item) for item in route_product_audit.get("tags") or []}
        route_class = str(route_product_audit.get("route_class") or "")
        is_advanced_terminal = (
            "advanced_or_product_like_terminal" in tags
            or bool(terminal_profile.get("product_like_terminal"))
            or bool(terminal_profile.get("large_polycyclic_terminal"))
        )
        if is_advanced_terminal and not route_product_audit.get("autonomous_route_candidate"):
            terminals = [
                str(smi)
                for smi in terminal_profile.get("terminal_reactants") or []
                if _valid_smiles(str(smi))
            ]
            if terminals:
                product_like_frontiers.extend(terminals)
                route["unresolved_frontiers"] = _dedupe([*route.get("unresolved_frontiers", []), *terminals])
            route_audit = dict(route.get("route_audit") or {})
            reasons = list(route_audit.get("reasons") or [])
            reasons.extend([
                "product_route_feasibility_non_autonomous",
                "advanced_or_product_like_terminal",
                f"product_audit_route_class:{route_class or 'unknown'}",
            ])
            route_audit["accepted_solved_route"] = False
            route_audit["reasons"] = sorted(set(reasons))
            route["route_audit"] = route_audit
        routes.append(route)

    downgraded = bool(
        baseline_routes.get("solved")
        and not bool(target_audit.get("autonomous_route_candidate_any"))
        and product_like_frontiers
    )
    out = dict(baseline_routes)
    out["routes"] = routes
    out["product_route_feasibility_audit"] = audit
    if downgraded:
        reasons = list((out.get("route_audit") or {}).get("reasons") or [])
        reasons.extend([
            "product_route_feasibility_non_autonomous",
            "advanced_or_product_like_terminal",
            f"target_verdict:{target_audit.get('target_verdict') or 'unknown'}",
        ])
        out["solved"] = False
        out["stock_audit_passed"] = False
        out["product_audit_downgraded_native_solved"] = True
        out["route_audit"] = {
            "schema_version": "route_audit_report.v1",
            "route_status": "fake_closed_rejected",
            "stock_audit_passed": False,
            "fake_closure_rejected": True,
            "reasons": sorted(set(reasons)),
            "product_route_feasibility": {
                "target_verdict": target_audit.get("target_verdict"),
                "triage_signal_any": bool(target_audit.get("triage_signal_any")),
                "autonomous_route_candidate_any": bool(target_audit.get("autonomous_route_candidate_any")),
                "product_like_frontiers": _dedupe(product_like_frontiers),
            },
        }
    else:
        out["product_audit_downgraded_native_solved"] = False
    return out


def _target_summary(
    *,
    target: Any,
    target_dir: Path,
    workflow_dir: Path,
    baseline_routes: dict[str, Any],
    stuck_frontier: str,
    workflow_result: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    trigger_report = _read_json(workflow_dir / "literature_trigger_report.json")
    literature_report = _read_json(workflow_dir / "literature_search_report.json")
    worker_record = _read_json(workflow_dir / "literature_worker_run_record.json")
    package = _read_json((workflow_result.get("artifacts") or {}).get("hybrid_route_package"))
    validation = dict(workflow_result.get("validation") or {})
    candidates = list(package.get("literature_candidates") or [])
    templates = list(package.get("strategy_templates") or [])
    generated_template_payload = {
        "schema_version": "statin_panel_generated_template_inventory.v1",
        "target": target.safe,
        "template_count": len(templates),
        "templates": templates,
        "source": "smiles_first_after_native.strategy_templates",
        "direct_one_step_consumption_allowed": False,
        "note": "Generated templates are advisory/planning templates unless separately promoted by applicability and reconstruction gates.",
    }
    write_json(target_dir / "generated_template_inventory.json", generated_template_payload)
    return {
        "schema_version": "statin_panel_formal_target_manifest.v1",
        "target": target.safe,
        "target_name": target.name,
        "family_bucket": target.family_bucket,
        "target_smiles": target.target_smiles,
        "elapsed_s": round(float(elapsed_s), 3),
        "native_chemenzy": {
            "status": baseline_routes.get("status"),
            "solved": bool(baseline_routes.get("solved")),
            "route_count": int(baseline_routes.get("route_count") or 0),
            "raw_backend_solved": bool(baseline_routes.get("raw_backend_solved")),
            "raw_solved_route_count": int(baseline_routes.get("raw_solved_route_count") or 0),
            "accepted_solved_route_count": int(baseline_routes.get("accepted_solved_route_count") or 0),
            "audit": dict(baseline_routes.get("route_audit") or {}),
        },
        "stuck_frontier": stuck_frontier,
        "literature_trigger": {
            "should_trigger": bool(trigger_report.get("should_trigger")),
            "trigger_reasons": list(trigger_report.get("trigger_reasons") or []),
            "native_audit_passed": bool(trigger_report.get("native_audit_passed")),
        },
        "codex_literature_worker": {
            "backend_requested": literature_report.get("backend_requested") or "",
            "backend_resolved": literature_report.get("backend_resolved") or literature_report.get("backend") or "",
            "worker_backend": worker_record.get("backend") or "",
            "worker_status": worker_record.get("status") or ("skipped" if not trigger_report.get("should_trigger") else ""),
            "worker_metadata": worker_record.get("metadata") or {},
            "hit_count": int(literature_report.get("hit_count") or 0),
            "unresolved_literature_gap": bool(literature_report.get("unresolved_literature_gap")),
            "limitations": list(literature_report.get("limitations") or []),
        },
        "generated_material": {
            "evidence_card_count": _jsonl_count(workflow_dir / "evidence_cards.jsonl"),
            "candidate_count": len(candidates),
            "strategy_template_count": len(templates),
            "candidate_kinds": _count_by_key(candidates, "candidate_kind"),
            "reaction_classes": _count_by_key(candidates, "reaction_class"),
            "generated_template_inventory": str(target_dir / "generated_template_inventory.json"),
        },
        "route_package": {
            "accepted": bool(validation.get("accepted")),
            "route_status": validation.get("route_status") or package.get("route_status") or "",
            "reasons": list(validation.get("reasons") or []),
        },
        "artifacts": {
            "target_input": str(target_dir / "target_input.json"),
            "chemenzy_native_raw_result": str(target_dir / "chemenzy_native_raw_result.json"),
            "chemenzy_baseline_routes": str(target_dir / "chemenzy_baseline_routes.json"),
            "smiles_first_workflow_result": str(target_dir / "smiles_first_workflow_result.json"),
            "workflow_dir": str(workflow_dir),
            "literature_trigger_report": str(workflow_dir / "literature_trigger_report.json"),
            "literature_worker_run_record": str(workflow_dir / "literature_worker_run_record.json"),
            "evidence_cards": str(workflow_dir / "evidence_cards.jsonl"),
            "literature_candidates": str((workflow_result.get("artifacts") or {}).get("literature_candidates") or ""),
            "hybrid_route_package": str((workflow_result.get("artifacts") or {}).get("hybrid_route_package") or ""),
            "validation": str((workflow_result.get("artifacts") or {}).get("validation") or ""),
            "summary": str((workflow_result.get("artifacts") or {}).get("summary") or ""),
            "target_manifest": str(target_dir / "target_manifest.json"),
        },
    }


def _panel_manifest(
    run_root: Path,
    run_input: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    triggered = [row for row in rows if (row.get("literature_trigger") or {}).get("should_trigger")]
    worker_rows = [
        row for row in rows
        if (row.get("codex_literature_worker") or {}).get("worker_status")
        and (row.get("codex_literature_worker") or {}).get("worker_status") != "skipped"
    ]
    return {
        "schema_version": "statin_panel_formal_native_codex_manifest.v1",
        "complete": complete,
        "output_dir": str(run_root),
        "run_input": run_input,
        "target_count": len(rows),
        "summary": {
            "native_solved_count": sum(1 for row in rows if (row.get("native_chemenzy") or {}).get("solved")),
            "native_route_count_total": sum(int((row.get("native_chemenzy") or {}).get("route_count") or 0) for row in rows),
            "literature_triggered_count": len(triggered),
            "worker_attempted_count": len(worker_rows),
            "worker_accepted_count": sum(
                1 for row in worker_rows
                if (row.get("codex_literature_worker") or {}).get("worker_status") == "accepted_draft"
            ),
            "evidence_card_count_total": sum(
                int((row.get("generated_material") or {}).get("evidence_card_count") or 0) for row in rows
            ),
            "candidate_count_total": sum(
                int((row.get("generated_material") or {}).get("candidate_count") or 0) for row in rows
            ),
            "strategy_template_count_total": sum(
                int((row.get("generated_material") or {}).get("strategy_template_count") or 0) for row in rows
            ),
        },
        "targets": rows,
    }


def _unresolved_frontiers_from_route(route: dict[str, Any]) -> list[str]:
    unresolved: list[str] = []
    stock = dict(route.get("stock_status") or {})
    for smiles, in_stock in stock.items():
        if in_stock is False and _valid_smiles(smiles):
            unresolved.append(smiles)
    for step in route.get("steps") or []:
        for smiles, in_stock in dict(step.get("stock_status") or {}).items():
            if in_stock is False and _valid_smiles(smiles):
                unresolved.append(smiles)
    return _dedupe(unresolved)


def _audit_native_route(route: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if route.get("unresolved_frontiers"):
        reasons.append("unresolved_frontiers_present")
    steps = list(route.get("steps") or [])
    if not steps:
        reasons.append("route_has_no_steps")
    for idx, step in enumerate(steps, start=1):
        for reason in _audit_step_atom_accounting(step):
            reasons.append(f"step_{idx}:{reason}")
    return {
        "schema_version": "native_route_audit.v1",
        "accepted_solved_route": bool(route.get("solved")) and not reasons,
        "raw_backend_solved": bool(route.get("solved")),
        "reasons": sorted(set(reasons)),
    }


def _audit_step_atom_accounting(step: dict[str, Any]) -> list[str]:
    product = str(step.get("product_smiles") or "")
    reactants = [str(item) for item in step.get("reactant_smiles") or []]
    if not _valid_smiles(product):
        return ["invalid_product_smiles"]
    if not reactants:
        return ["missing_reactants"]
    product_counts = _element_counts(product)
    reactant_counts: dict[int, int] = {}
    invalid_reactant = False
    for smiles in reactants:
        if not _valid_smiles(smiles):
            invalid_reactant = True
            continue
        for atomic_num, count in _element_counts(smiles).items():
            reactant_counts[atomic_num] = reactant_counts.get(atomic_num, 0) + count
    reasons = []
    if invalid_reactant:
        reasons.append("invalid_reactant_smiles")
    for atomic_num, count in product_counts.items():
        if reactant_counts.get(atomic_num, 0) < count:
            reasons.append("atom_accounting_failed")
            break
    return reasons


def _first_stuck_frontier(baseline_routes: dict[str, Any]) -> str:
    for route in baseline_routes.get("routes") or []:
        for smiles in route.get("unresolved_frontiers") or []:
            if _valid_smiles(smiles):
                return str(smiles)
    return ""


def _family_hint(target: Any) -> str:
    base = [
        target.name,
        target.safe,
        "statin",
        str(target.family_bucket or ""),
        str(target.expected_family_id or ""),
        str(target.expected_reaction_class or ""),
    ]
    if target.family_bucket == "natural_statin":
        base.extend(["semisynthesis", "fermentation core", "lactone acid interconversion"])
    else:
        base.extend(["synthetic statin", "syn-3,5-dihydroxy acid side chain", "HWE", "Wittig"])
    return ", ".join(item for item in base if item)


def _element_counts(smiles: str) -> dict[int, int]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {}
    counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num <= 1:
            continue
        counts[atomic_num] = counts.get(atomic_num, 0) + 1
    return counts


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _selected_targets(raw: str) -> set[str]:
    return {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}


def _default_output_dir(backend: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "shared" / f"statin_panel_formal_native_{backend}_{stamp}"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _valid_smiles(smiles: str) -> bool:
    return bool(smiles) and Chem.MolFromSmiles(str(smiles)) is not None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
