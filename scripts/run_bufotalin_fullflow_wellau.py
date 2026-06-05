#!/usr/bin/env python3
"""Run a fresh Bufotalin full-flow showcase with ChemEnzy + literature workers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.codex_worker import WorkerBudget, WorkerTask, run_codex_worker
from cascade_planner.agent.literature_escalation import decide_literature_escalation
from cascade_planner.agent.literature_segments import (
    unroll_literature_route_segment,
    validate_literature_route_segment,
)
from cascade_planner.agent.route_package import write_json
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.agent.target_profile import build_frontier_report, build_target_profile
from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig


RDLogger.DisableLog("rdApp.*")

BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_FAMILY = "bufotalin, bufadienolide, steroid, C17 2-pyrone, bufalin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-smiles", default=BUFOTALIN_SMILES)
    parser.add_argument("--target-name", default="bufotalin_fullflow")
    parser.add_argument("--family-hint", default=BUFOTALIN_FAMILY)
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--expansion-topk", type=int, default=100)
    parser.add_argument("--worker-timeout-s", type=float, default=120.0)
    parser.add_argument("--query-budget", type=int, default=10)
    parser.add_argument("--literature-backend", default="api_json", choices=["api_json", "codex"])
    parser.add_argument("--segment-backend", default="", choices=["", "api_json", "codex"])
    args = parser.parse_args()
    segment_backend = args.segment_backend or args.literature_backend

    run_root = Path(args.output_dir) if args.output_dir else _default_output_dir(args.literature_backend)
    run_root.mkdir(parents=True, exist_ok=True)
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_input = {
        "schema_version": "bufotalin_fullflow_input.v1",
        "target_name": args.target_name,
        "target_smiles": args.target_smiles,
        "family_hint": args.family_hint,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "reuse_previous_results": False,
        "api_json_key_source": str(ROOT / "key.txt"),
        "literature_backend": args.literature_backend,
        "segment_backend": segment_backend,
        "chem_enzy": {
            "vendor_root": args.vendor_root,
            "gpu": args.gpu,
            "iterations": args.iterations,
            "max_depth": args.max_depth,
            "expansion_topk": args.expansion_topk,
            "stock_names": DEFAULT_STOCKS,
            "one_step_models": DEFAULT_ONE_STEP_MODELS,
        },
    }
    write_json(run_root / "run_input.json", run_input)

    profile = build_target_profile(args.target_smiles, target_name=args.target_name, family_hint=args.family_hint)
    write_json(run_root / "pre_target_profile.json", profile.to_dict())
    pre_frontier = build_frontier_report(profile, frontier_smiles=args.target_smiles, baseline_routes={})
    pre_decision = decide_literature_escalation(
        native_result={"schema_version": "baseline_routes.v1", "status": "pre_native_not_run", "solved": False, "routes": []},
        route_audit={"route_status": "unresolved", "reasons": ["native_not_run_yet"], "stock_audit_passed": False},
        frontier_report=pre_frontier,
        user_objective="triage before native ChemEnzy full run; defer external research until native stuck point",
    )
    write_json(run_root / "pre_literature_triage.json", {
        "schema_version": "bufotalin_pre_literature_triage.v1",
        "decision": pre_decision.to_dict(),
        "execution_policy": "run native ChemEnzy first, then research unresolved/stuck frontier",
    })

    baseline_result = _run_native_chemenzy(args)
    write_json(run_root / "chemenzy_native_raw_result.json", baseline_result.to_dict())
    baseline_routes = _baseline_routes_from_result(baseline_result)
    write_json(run_root / "chemenzy_baseline_routes.json", baseline_routes)

    stuck_frontier = _first_stuck_frontier(baseline_routes) or args.target_smiles
    workflow_dir = run_root / "smiles_first_after_chemenzy_stuck"
    workflow_result = run_smiles_first_workflow(
        SmilesFirstWorkflowConfig(
            target_smiles=args.target_smiles,
            target_name=args.target_name,
            family_hint=args.family_hint,
            objective="route after native ChemEnzy audit; use external research only for stuck or fake-closed frontiers",
            output_dir=workflow_dir,
            frontier_smiles=stuck_frontier,
            baseline_json=run_root / "chemenzy_baseline_routes.json",
            query_budget=args.query_budget,
            literature_backend=args.literature_backend,
            worker_timeout_s=args.worker_timeout_s,
            worker_max_output_bytes=200_000,
            worker_max_tool_calls=8,
        )
    )
    write_json(run_root / "smiles_first_workflow_result.json", workflow_result)

    segment_result = _run_segment_worker(
        run_root=run_root,
        workflow_dir=workflow_dir,
        target_smiles=args.target_smiles,
        frontier_smiles=stuck_frontier,
        case_id=str(workflow_result.get("case_id") or profile.case_id),
        worker_timeout_s=args.worker_timeout_s,
        backend=segment_backend,
    )
    write_json(run_root / "literature_segment_fullflow_result.json", segment_result)

    flow_svg = _render_fullflow_svg(
        target_smiles=args.target_smiles,
        frontier_smiles=stuck_frontier,
        baseline_routes=baseline_routes,
        workflow_dir=workflow_dir,
        workflow_result=workflow_result,
        segment_result=segment_result,
    )
    flow_path = figures_dir / "bufotalin_fullflow_reaction_flow.svg"
    flow_path.write_text(flow_svg, encoding="utf-8")

    manifest = {
        "schema_version": "bufotalin_fullflow_manifest.v1",
        "output_dir": str(run_root),
        "artifacts": {
            "run_input": str(run_root / "run_input.json"),
            "pre_literature_triage": str(run_root / "pre_literature_triage.json"),
            "chemenzy_native_raw_result": str(run_root / "chemenzy_native_raw_result.json"),
            "chemenzy_baseline_routes": str(run_root / "chemenzy_baseline_routes.json"),
            "smiles_first_workflow_result": str(run_root / "smiles_first_workflow_result.json"),
            "literature_segment_fullflow_result": str(run_root / "literature_segment_fullflow_result.json"),
            "reaction_flow_svg": str(flow_path),
            "workflow_route_map_svg": str((workflow_result.get("artifacts") or {}).get("route_map") or ""),
            "case_bundle": str((workflow_result.get("artifacts") or {}).get("case_bundle") or ""),
        },
        "summary": _run_summary(baseline_routes, workflow_dir, workflow_result, segment_result),
    }
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


def _default_output_dir(backend: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    label = "codex_search" if backend == "codex" else "wellau"
    return ROOT / "results" / "shared" / f"bufotalin_fullflow_{label}_{stamp}"


def _run_native_chemenzy(args: argparse.Namespace):
    config = RouteSearchConfig(
        target_smiles=args.target_smiles,
        stock_names=DEFAULT_STOCKS,
        max_iterations=int(args.iterations),
        max_depth=int(args.max_depth),
        expansion_topk=int(args.expansion_topk),
        one_step_models=DEFAULT_ONE_STEP_MODELS,
        search_flags={
            "gpu": int(args.gpu),
            "keep_search": True,
            "include_cascade_expansion_trace": True,
            "cascade_expansion_trace_preview": 30,
            "condition_model": "rcr",
            "chem_enzy_onmt_tokenizer": "char",
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


def _first_stuck_frontier(baseline_routes: dict[str, Any]) -> str:
    for route in baseline_routes.get("routes") or []:
        for smiles in route.get("unresolved_frontiers") or []:
            if _valid_smiles(smiles):
                return str(smiles)
    return ""


def _run_segment_worker(
    *,
    run_root: Path,
    workflow_dir: Path,
    target_smiles: str,
    frontier_smiles: str,
    case_id: str,
    worker_timeout_s: float,
    backend: str,
) -> dict[str, Any]:
    evidence_path = workflow_dir / "evidence_cards.jsonl"
    evidence_preview = evidence_path.read_text(encoding="utf-8")[:8000] if evidence_path.exists() else ""
    task = WorkerTask(
        task_id=f"{case_id}:literature_segment:{backend}",
        case_id=case_id,
        task_type="target_research",
        required_artifact_type="LiteratureRouteSegmentCard",
        input_refs=[
            str(workflow_dir / "target_profile.json"),
            str(workflow_dir / "frontier_report.json"),
            str(evidence_path),
        ],
        allowed_tools=["literature_search", "web_search"],
        budget=WorkerBudget(
            timeout_s=float(worker_timeout_s),
            max_output_bytes=200_000,
            max_tool_calls=8,
            max_worker_runs=1,
        ),
        objective=(
            "For Bufotalin/bufadienolide retrosynthesis, extract one 2-5 step literature route segment "
            "for the target or stuck frontier. Use exact relation only for the exact target/intermediate; "
            "use analog relation with scope_gap when the source is a family precedent. Do not include reaction SMILES. "
            "The payload target_smiles must be exactly the Target SMILES supplied below; do not rewrite it. "
            "reactant_smiles must be an array of complete valid SMILES strings, never an array of characters. "
            "Every product_smiles, reactant_smiles item, and reconstructed_product_smiles must be valid SMILES. "
            "Do not hand-reconstruct steroid intermediates unless you can provide a syntactically valid complete SMILES; "
            "if the paper only provides a drawing/scheme label or the structure is uncertain, set relation_type mismatch "
            "or applicability.status rejected and explain the scope_gap instead of pretending the segment is executable. "
            "Each step must include valid product_smiles/reactant_smiles, evidence refs, source_ref, applicability "
            "with status either passed/exact/rejected, product_reconstruction_passed, and reconstructed_product_smiles. "
            "Each condition_candidate must include step_id, condition_source_type, condition_status, evidence_refs, "
            "confidence, and risk flags when available. "
            f"Target SMILES: {target_smiles}. Stuck frontier SMILES: {frontier_smiles}. "
            f"Evidence preview from this run: {evidence_preview}"
        ),
        allowed_workdir=str(run_root),
        dry_run=False,
    )
    write_json(run_root / "literature_segment_worker_task.json", task.to_dict())
    record = run_codex_worker(task, use_codex_cli=backend == "codex", use_api_json=backend == "api_json")
    write_json(run_root / "literature_segment_worker_run_record.json", record.to_dict())
    payload = dict((record.output_artifact or {}).get("payload") or {}) if isinstance(record.output_artifact, dict) else {}
    if payload:
        validation = validate_literature_route_segment(payload)
        unroll = unroll_literature_route_segment(payload, max_steps=5, native_solved_audit_passed=False)
    else:
        validation = {
            "schema_version": "literature_route_segment_validation.v1",
            "accepted": False,
            "reasons": ["missing_segment_payload"],
            "allowed_for_recursive_unroll": False,
        }
        unroll = {
            "schema_version": "literature_route_segment_unroll_trace.v1",
            "final_status": "rejected",
            "stop_reason": "missing_segment_payload",
        }
    write_json(run_root / "literature_segment_validation.json", validation)
    write_json(run_root / "literature_segment_unroll_trace.json", unroll)
    return {
        "schema_version": "bufotalin_segment_fullflow_result.v1",
        "worker_record_path": str(run_root / "literature_segment_worker_run_record.json"),
        "validation_path": str(run_root / "literature_segment_validation.json"),
        "unroll_trace_path": str(run_root / "literature_segment_unroll_trace.json"),
        "worker_status": record.status,
        "worker_backend": record.backend,
        "worker_metadata": record.metadata,
        "segment_validation": validation,
        "segment_unroll": unroll,
    }


def _run_summary(
    baseline_routes: dict[str, Any],
    workflow_dir: Path,
    workflow_result: dict[str, Any],
    segment_result: dict[str, Any],
) -> dict[str, Any]:
    literature_report = _read_json(workflow_dir / "literature_search_report.json")
    worker_record = _read_json(workflow_dir / "literature_worker_run_record.json")
    validation = workflow_result.get("validation") or {}
    return {
        "native_chemenzy": {
            "status": baseline_routes.get("status"),
            "solved": baseline_routes.get("solved"),
            "route_count": baseline_routes.get("route_count"),
            "audit": baseline_routes.get("route_audit"),
        },
        "literature_worker": {
            "backend": worker_record.get("backend"),
            "status": worker_record.get("status"),
            "metadata": worker_record.get("metadata"),
            "evidence_hits": literature_report.get("hit_count"),
            "unresolved_literature_gap": literature_report.get("unresolved_literature_gap"),
        },
        "route_package": {
            "accepted": validation.get("accepted"),
            "route_status": validation.get("route_status"),
            "reasons": validation.get("reasons"),
        },
        "literature_segment": {
            "worker_status": segment_result.get("worker_status"),
            "validation": segment_result.get("segment_validation"),
            "unroll": segment_result.get("segment_unroll"),
        },
    }


def _render_fullflow_svg(
    *,
    target_smiles: str,
    frontier_smiles: str,
    baseline_routes: dict[str, Any],
    workflow_dir: Path,
    workflow_result: dict[str, Any],
    segment_result: dict[str, Any],
) -> str:
    literature_report = _read_json(workflow_dir / "literature_search_report.json")
    worker_record = _read_json(workflow_dir / "literature_worker_run_record.json")
    validation = workflow_result.get("validation") or {}
    segment_validation = segment_result.get("segment_validation") or {}
    unroll = segment_result.get("segment_unroll") or {}

    width = 1440
    height = 1160
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"><polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#1f2937\"/></marker>",
        "<style>.title{font:700 24px Arial;fill:#111827}.h{font:700 15px Arial;fill:#111827}.b{font:12px Arial;fill:#374151}.s{font:11px Arial;fill:#4b5563}.box{fill:#fff;stroke:#cbd5e1;stroke-width:1.3;rx:6}.ok{fill:#ecfdf5;stroke:#34d399}.warn{fill:#fffbeb;stroke:#fbbf24}.bad{fill:#fef2f2;stroke:#f87171}.arrow{stroke:#1f2937;stroke-width:1.8;marker-end:url(#arrow)}</style></defs>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="36" y="38" class="title">{_esc("Bufotalin Fresh Fullflow: ChemEnzy -> " + str(worker_record.get("backend") or "Literature") + " Literature -> Segment Gate")}</text>',
    ]
    chunks.append(_mol_panel("Target Bufotalin", target_smiles, 36, 66, 430, 260))
    chunks.append(_mol_panel("Stuck frontier used for literature", frontier_smiles, 502, 66, 430, 260))
    chunks.append(_stage_box(
        968,
        66,
        420,
        260,
        "Run Summary",
        [
            f"ChemEnzy solved: {baseline_routes.get('solved')} routes={baseline_routes.get('route_count')}",
            f"Literature worker: {worker_record.get('backend')} / {worker_record.get('status')}",
            f"Provider: {(worker_record.get('metadata') or {}).get('provider')} endpoint={(worker_record.get('metadata') or {}).get('endpoint')}",
            f"Evidence hits: {literature_report.get('hit_count')}",
            f"Route package: {validation.get('route_status')} accepted={validation.get('accepted')}",
            f"Segment gate: accepted={segment_validation.get('accepted')} status={unroll.get('final_status')}",
        ],
        "ok" if validation.get("accepted") else "warn",
    ))
    stages = [
        ("1. Pre-triage", "Complex steroid/bufadienolide; policy records literature need but runs native ChemEnzy first."),
        ("2. Native ChemEnzy full run", f"Backend ChemEnzyRetroPlanner; stock closed={baseline_routes.get('solved')}; audit={(baseline_routes.get('route_audit') or {}).get('route_status')}."),
        ("3. Stuck-point research", f"{worker_record.get('backend')} worker creates traceable EvidenceCard for frontier; status={worker_record.get('status')}."),
        ("4. Evidence -> candidates", f"SMILES-first package mines strategic disconnections/templates; validation={validation.get('route_status')}."),
        ("5. Literature multi-step segment", f"Segment worker status={segment_result.get('worker_status')}; unroll={unroll.get('final_status')} stop={unroll.get('stop_reason')}."),
        ("6. Final flow diagram", "This SVG and route-map SVG are generated from this run directory only."),
    ]
    y = 380
    for idx, (label, body) in enumerate(stages):
        x = 70 + (idx % 2) * 680
        if idx and idx % 2 == 0:
            y += 150
        chunks.append(_stage_box(x, y, 590, 112, label, [body], "ok" if idx in {0, 2, 3, 5} else "warn"))
        if idx < len(stages) - 1:
            nx = 70 + ((idx + 1) % 2) * 680
            ny = y + (150 if (idx + 1) % 2 == 0 else 0)
            chunks.append(f'<line class="arrow" x1="{x + 590}" y1="{y + 56}" x2="{nx}" y2="{ny + 56}"/>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def _mol_panel(title: str, smiles: str, x: int, y: int, width: int, height: int) -> str:
    return "\n".join([
        f'<rect class="box" x="{x}" y="{y}" width="{width}" height="{height}"/>',
        f'<text x="{x + 16}" y="{y + 24}" class="h">{_esc(title)}</text>',
        _mol_svg(smiles, x + 16, y + 40, width - 32, height - 92),
        f'<text x="{x + 16}" y="{y + height - 28}" class="s">{_esc(_short(smiles, 76))}</text>',
    ])


def _mol_svg(smiles: str, x: int, y: int, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return f'<text x="{x}" y="{y + 30}" class="b">Invalid SMILES</text>'
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    raw = drawer.GetDrawingText()
    svg_start = raw.find("<svg")
    start = raw.find(">", svg_start)
    end = raw.rfind("</svg>")
    inner = raw[start + 1 : end] if svg_start >= 0 and start >= 0 and end >= 0 else raw
    return f'<svg x="{x}" y="{y}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{inner}</svg>'


def _stage_box(x: int, y: int, width: int, height: int, title: str, lines: list[str], cls: str = "") -> str:
    box_cls = f"box {cls}".strip()
    chunks = [f'<rect class="{box_cls}" x="{x}" y="{y}" width="{width}" height="{height}"/>']
    chunks.append(f'<text x="{x + 16}" y="{y + 26}" class="h">{_esc(title)}</text>')
    ty = y + 52
    for line in lines[:7]:
        chunks.append(f'<text x="{x + 16}" y="{ty}" class="b">{_esc(_short(line, 120))}</text>')
        ty += 18
    return "\n".join(chunks)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _short(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[: max(0, limit - 3)] + "..."


def _esc(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
