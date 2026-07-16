#!/usr/bin/env python3
"""Build the curated closed-route presentation hub artifacts.

The hub is a display inventory, not a new proof authority.  Current V4
workbenches are re-rendered with the latest route-forest UI.  The one legacy
Atorvastatin route is explicitly migrated as a route-closed, L0 advisory
candidate so its historical ``solved`` label cannot leak into the new panel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.route_workbench import compile_route_workbench  # noqa: E402
from cascade_planner.harness.v4_route_workbench import (  # noqa: E402
    compile_v4_route_forest,
    render_v4_route_workbench_html,
)


DEFAULT_OUTPUT = ROOT / "results" / "shared" / "presentation_showcase_20260715"
BUFOTALIN_WORKBENCH = (
    ROOT
    / "results"
    / "shared"
    / "bufotalin_v4_reported_candidate_20260715"
    / "route_workbench.json"
)
BUFOTALIN_DOSSIER = BUFOTALIN_WORKBENCH.with_name("bufotalin_20_step_dossier.json")
NIRMATRELVIR_WORKBENCH = (
    ROOT / "results" / ".autoplanner" / "nirmatrelvir-showcase-v4" / "route_workbench.json"
)
ARTEMISININ_WORKBENCH = (
    ROOT / "results" / ".autoplanner" / "artemisinin-final-showcase" / "route_workbench.json"
)
ATORVASTATIN_RUN = (
    ROOT
    / "results"
    / "shared"
    / "atorvastatin_blackboard_correct_identity_runtime_fixed_parent_proof_20260706"
)
STATIN_RERUN_ROOTS = (
    ROOT / "results" / "shared" / "presentation_v4_statins_a_20260715",
    ROOT / "results" / "shared" / "presentation_v4_statins_b_20260715",
    ROOT / "results" / "shared" / "presentation_v4_statins_c_20260715",
)
STATIN_RERUN_SHOWCASE = (
    ROOT / "results" / "shared" / "presentation_v4_statins_combined_20260715" / "index.html"
)
STATIN_CATALOG_PATH = ROOT / "benchmarks" / "statin_target_catalog.v1.json"
STATIN_READINESS_PATH = ROOT / "benchmarks" / "statin_target_route_readiness.v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-short-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the closed Artemisinin short-route control.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    statin_catalog = _load_verified_json(
        STATIN_CATALOG_PATH, schema_version="target_route_catalog.v1"
    )
    statin_readiness = _load_verified_json(
        STATIN_READINESS_PATH, schema_version="target_route_readiness_catalog.v1"
    )
    readiness_by_target = {
        str(row["target_name"]): dict(row) for row in statin_readiness["targets"]
    }
    statin_targets = [
        {**dict(row), "route_readiness": readiness_by_target[str(row["target_name"])]}
        for row in statin_catalog["targets"]
    ]
    if {row["target_name"] for row in statin_targets} != set(readiness_by_target):
        raise ValueError("statin_catalog_readiness_target_mismatch")

    cases: list[dict[str, Any]] = []
    bufotalin = _load_json(BUFOTALIN_WORKBENCH)
    cases.append(
        _write_case(
            output,
            case_id="bufotalin-v4-20-step",
            display_name="蟾毒灵 · 20 步闭合基准",
            category="天然产物全合成",
            workbench=bufotalin,
            presentation_note="15 步论文报道 + 5 步规划补全；路线闭合，证明仍待补齐。",
            proof_distribution={
                "L0_rejected": 2,
                "L0_advisory": 3,
                "L1_source_reported": 15,
            },
            claim_status="route_closed_proof_unresolved",
            strategy_override="蟾毒灵 20 步 · 15 文献报道 + 5 规划补全",
            standard=True,
        )
    )

    nirmatrelvir = _load_json(NIRMATRELVIR_WORKBENCH)
    cases.append(
        _write_case(
            output,
            case_id="nirmatrelvir-v4-closed",
            display_name="奈玛特韦 · 双信源闭合",
            category="抗病毒药物",
            workbench=nirmatrelvir,
            presentation_note="两条 7–8 步独立来源路线；反应、来源与库存边界均闭合。",
            claim_status="portfolio_accepted",
        )
    )

    atorvastatin = _migrate_atorvastatin_candidate(ATORVASTATIN_RUN)
    cases.append(
        _write_case(
            output,
            case_id="atorvastatin-v4-migrated-closed",
            display_name="阿托伐他汀 · 11 步闭合候选",
            category="他汀类药物",
            workbench=atorvastatin,
            presentation_note="旧路线经新语义迁移：搜索边界闭合；11 步均为 L0 规划候选，实验条件待取证。",
            proof_distribution={"L0_advisory": 11},
            claim_status="route_closed_proof_unresolved",
        )
    )

    if args.include_short_control:
        artemisinin = _load_json(ARTEMISININ_WORKBENCH)
        cases.append(
            _write_case(
                output,
                case_id="artemisinin-v4-closed-control",
                display_name="青蒿素 · 闭合对照",
                category="半合成对照",
                workbench=artemisinin,
                presentation_note="1–2 步采购边界替代对照；用于说明长短路线与边界语义。",
                claim_status="portfolio_accepted",
            )
        )

    manifest = {
        "schema_version": "autoplanner.presentation_showcase.v1",
        "generated_at": _utc_now(),
        "standard_case_id": "bufotalin-v4-20-step",
        "standard": {
            "summary": "蟾毒灵标准：长路线闭合；信源逐边分色；低可信保留并警示；条件与来源过程可展开。",
            "requirements": [
                "route_closure_is_separate_from_proof_acceptance",
                "low_confidence_edges_remain_visible_with_warning_encoding",
                "proof_and_source_tiers_are_colored_per_edge",
                "reaction_conditions_are_clickable_and_fully_expandable",
            ],
        },
        "cases": cases,
        "statin_catalog": {
            "entity_count": len(statin_targets),
            "first_wave_count": sum(
                row["rerun_wave"] == "v4_first_9" for row in statin_targets
            ),
            "extended_wave_count": sum(
                row["rerun_wave"] == "v4_extended_3" for row in statin_targets
            ),
            "normalization_note": statin_catalog["normalization_note"],
            "catalog_sha256": statin_catalog["content_sha256"],
            "readiness_sha256": statin_readiness["content_sha256"],
            "readiness_summary": statin_readiness["summary"],
            "targets": statin_targets,
        },
        "audits": [
            {
                "audit_id": "statin-v4-live-rerun",
                "label": "他汀 V4 实跑审计",
                "artifact_path": _relative(STATIN_RERUN_SHOWCASE),
                "note": "真实新架构 blind-run；即使模型、视觉或 ChemEnzy 未产出路线，也保留失败阶段和资源计数。",
            },
            {
                "audit_id": "statin-target-route-readiness",
                "label": "他汀路线就绪度矩阵",
                "artifact_path": _relative(STATIN_READINESS_PATH),
                "note": "按路线长度、来源、条件、current replay 与 acceptance attestation 分轴；低可信和空图不隐藏。",
            },
        ],
        "reruns": [_rerun_status(path, index) for index, path in enumerate(STATIN_RERUN_ROOTS, 1)],
        "excluded": [
            {
                "target_name": "paclitaxel",
                "reason": "route_portfolio_complete_route_count_zero",
                "presentation_policy": "research_candidate_only_not_default",
            },
            {
                "target_name": "strychnine",
                "reason": "hypothesis_routes_have_open_frontier",
                "presentation_policy": "research_candidate_only_not_default",
            },
        ],
        "message": "默认展示 3 个闭合复杂案例与 1 个短路线对照；他汀按 12 个独立母体实体编目，其中首批 9 个 V4 高预算重跑、3 个历史研发品种进入扩展队列。",
    }
    _write_json(output / "manifest.json", manifest)
    _write_notes(output / "PRESENTATION_NOTES.md", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "notes": str(output / "PRESENTATION_NOTES.md"),
                "case_count": len(cases),
                "available_case_count": sum(
                    (ROOT / str(row["artifact_path"])).is_file() for row in cases
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _write_case(
    output: Path,
    *,
    case_id: str,
    display_name: str,
    category: str,
    workbench: Mapping[str, Any],
    presentation_note: str,
    claim_status: str,
    proof_distribution: Mapping[str, int] | None = None,
    strategy_override: str = "",
    standard: bool = False,
) -> dict[str, Any]:
    case_dir = output / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(json.dumps(workbench, ensure_ascii=False))
    if strategy_override:
        for route in dict(snapshot.get("routes") or {}).values():
            if isinstance(route, dict):
                route["strategy"] = strategy_override
        snapshot.pop("content_sha256", None)
        snapshot["content_sha256"] = _digest(snapshot)
    forest = compile_v4_route_forest(snapshot)
    _write_json(case_dir / "route_workbench.json", snapshot)
    _write_json(case_dir / "explored_route_forest.json", forest)
    (case_dir / "route_forest.html").write_text(
        render_v4_route_workbench_html(snapshot), encoding="utf-8"
    )
    routes = [
        dict(row)
        for row in dict(snapshot.get("routes") or {}).values()
        if isinstance(row, Mapping)
    ]
    max_route = max(routes, key=lambda row: len(row.get("edge_ids") or []), default={})
    inferred_distribution = _proof_distribution(snapshot, max_route)
    route_closed = any(row.get("complete") is True for row in routes)
    source_groups = {
        str(value)
        for route in routes
        for value in route.get("independent_source_groups") or []
        if str(value)
    }
    source_references = _source_references(snapshot, max_route)
    return {
        "case_id": case_id,
        "target_name": str(dict(snapshot.get("target") or {}).get("name") or case_id),
        "display_name": display_name,
        "category": category,
        "standard": standard,
        "artifact_path": _relative(case_dir / "route_forest.html"),
        "workbench_path": _relative(case_dir / "route_workbench.json"),
        "forest_path": _relative(case_dir / "explored_route_forest.json"),
        "artifact_sha256": _file_sha256(case_dir / "route_forest.html"),
        "route_count": len(routes),
        "complete_route_count": sum(row.get("complete") is True for row in routes),
        "max_step_count": len(max_route.get("edge_ids") or []),
        "route_closed": route_closed,
        "process_ready": bool(dict(snapshot.get("portfolio") or {}).get("process_ready")),
        "claim_status": claim_status,
        "source_group_count": len(source_groups),
        "source_reference_count": len(source_references),
        "proof_distribution": dict(proof_distribution or inferred_distribution),
        "presentation_note": presentation_note,
    }


def _migrate_atorvastatin_candidate(run_dir: Path) -> dict[str, Any]:
    forest = _load_json(run_dir / "explored_route_forest.json")
    verdict = _load_json(run_dir / "final_verdict.json")
    branch = next(
        (
            dict(row)
            for row in forest.get("branches") or []
            if str(row.get("branch_id") or "") == "branch:direct_verified_chemenzy_route"
        ),
        None,
    )
    if branch is None:
        raise ValueError("atorvastatin_direct_route_missing")
    step_order = [str(value) for value in branch.get("step_ids") or []]
    step_by_id = {
        str(row.get("step_id") or ""): dict(row)
        for row in forest.get("steps") or []
        if isinstance(row, Mapping)
    }
    steps = [step_by_id[step_id] for step_id in step_order if step_id in step_by_id]
    if len(steps) != 11:
        raise ValueError(f"atorvastatin_step_count_invalid:{len(steps)}")
    node_by_id = {
        str(row.get("node_id") or ""): dict(row)
        for row in forest.get("nodes") or []
        if isinstance(row, Mapping)
    }
    used_ids = {
        str(node_id)
        for step in steps
        for node_id in [*(step.get("from_node_ids") or []), *(step.get("to_node_ids") or [])]
    }
    target_id = next(
        (
            node_id
            for node_id in used_ids
            if str(node_by_id.get(node_id, {}).get("role") or "") == "target"
        ),
        str((steps[-1].get("to_node_ids") or [""])[0]),
    )
    product_ids = {
        str(node_id) for step in steps for node_id in step.get("to_node_ids") or []
    }
    precursor_ids = {
        str(node_id) for step in steps for node_id in step.get("from_node_ids") or []
    }
    leaf_ids = sorted(precursor_ids - product_ids)
    if verdict.get("stock_audit_passed") is not True or not leaf_ids:
        raise ValueError("atorvastatin_legacy_stock_audit_not_closed")

    molecules: dict[str, dict[str, Any]] = {}
    stock_observations: dict[str, dict[str, Any]] = {}
    for node_id in sorted(used_ids):
        source = node_by_id.get(node_id) or {}
        smiles = str(source.get("smiles") or "")
        if not smiles:
            label_candidate = str(source.get("label") or "")
            if Chem.MolFromSmiles(label_candidate) is not None:
                smiles = Chem.MolToSmiles(
                    Chem.MolFromSmiles(label_candidate),
                    canonical=True,
                    isomericSmiles=True,
                )
        if not smiles:
            raise ValueError(f"atorvastatin_selected_molecule_missing_smiles:{node_id}")
        observation_id = f"stock:legacy:{_short_digest(node_id)}" if node_id in leaf_ids else ""
        molecules[node_id] = {
            "label": str(source.get("label") or node_id),
            "canonical_smiles": smiles,
            "formula": str(source.get("formula") or ""),
            "is_leaf": node_id in leaf_ids,
            "stock_closed": node_id in leaf_ids,
            "stock_observation_ids": [observation_id] if observation_id else [],
            "active_stock_observation_id": observation_id,
        }
        if observation_id:
            stock_observations[observation_id] = {
                "stock_observation_id": observation_id,
                "canonical_smiles": smiles,
                "supplier": "legacy controller benchmark stock audit",
                "catalog_number": "legacy-benchmark-boundary",
                "authority_scope": "legacy_benchmark_stock_snapshot_not_procurement",
                "accepted": True,
            }

    edges: dict[str, dict[str, Any]] = {}
    edge_proofs: dict[str, dict[str, Any]] = {}
    edge_ids: list[str] = []
    for index, step in enumerate(steps, 1):
        edge_id = f"edge:atorvastatin:migrated:{index:02d}"
        edge_ids.append(edge_id)
        findings = [
            {
                "finding_id": f"finding:atorvastatin:{index:02d}:{item_index:02d}",
                "severity": "warning",
                "message": str(message),
                "required_action": "Bind an exact literature procedure or rerun current-host atom mapping and condition validation.",
                "evidence": {"source": "legacy_blackboard_step_missing"},
            }
            for item_index, message in enumerate(step.get("missing") or [], 1)
        ]
        edges[edge_id] = {
            "product_molecule_id": str((step.get("to_node_ids") or [""])[0]),
            "precursor_molecule_ids": [str(value) for value in step.get("from_node_ids") or []],
            "origin_records": [
                {
                    "origin_kind": "chemenzy",
                    "proposal_id": str(step.get("step_id") or edge_id),
                }
            ],
            "reaction_proofs": [
                {
                    "accepted": False,
                    "authority": "legacy_route_verifier_migration_only",
                    "reason_codes": ["current_host_proof_missing", "exact_conditions_missing"],
                }
            ],
            "source_observation_record_ids": [],
            "validation_findings": findings,
        }
        edge_proofs[edge_id] = {
            "edge_id": edge_id,
            "achieved_level": 0,
            "accepted": False,
            "reaction_validated": False,
            "exact_source_bound": False,
            "source_binding_ids": [],
            "exact_record_ids": [],
            "source_observation_record_ids": [],
            "independent_source_groups": [],
            "conflict_ids": [],
            "reasons": ["current_host_proof_missing", "exact_conditions_missing"],
        }
    graph: dict[str, Any] = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "atorvastatin-v4-migrated-presentation-20260715",
        "target_name": "atorvastatin",
        "target_molecule_id": target_id,
        "revision": 1,
        "molecules": molecules,
        "edges": edges,
        "source_bindings": {},
        "exact_records": {},
        "source_observation_records": {},
        "procedure_records": {},
        "stock_observations": stock_observations,
        "conflicts": {},
        "hypotheses": {},
        "delta": {
            "rejected": [
                {
                    "kind": "route_acceptance",
                    "reasons": ["legacy_blackboard_route_requires_current_v4_revalidation"],
                    "route_preserved_for_display": True,
                }
            ]
        },
    }
    graph["scientific_sha256"] = _digest(graph)
    route_id = "route:atorvastatin:11-step-migrated-candidate"
    route = {
        "schema_version": "proof_stitched_route.v1",
        "route_id": route_id,
        "route_family_id": "family:atorvastatin:legacy-chemenzy-migration",
        "strategy": "阿托伐他汀 11 步 · 规划路线闭合候选",
        "edge_ids": edge_ids,
        "leaf_molecule_ids": leaf_ids,
        "root_edge_ids": [edge_ids[-1]],
        "module_selections": {},
        "minimum_edge_proof_level": 0,
        "all_edges_proven": False,
        "unproven_edge_ids": edge_ids,
        "stock_closure_rate": 1.0,
        "all_leaves_stock_closed": True,
        "open_leaf_molecule_ids": [],
        "independent_source_groups": [],
        "source_independence_met": False,
        "source_independence_required": True,
        "conflict_ids": [],
        "length": len(edge_ids),
        "convergence_score": 0.5,
        "risk_score": 1.0,
        "complete": True,
        "selected": True,
        "pareto_optimal": True,
        "reported_in_source": False,
        "reported_source_refs": [],
        "reported_step_count": 0,
        "planner_hypothesis_step_count": len(edge_ids),
        "semantics": {
            "configured_boundary_closure_is_independent_of_edge_proof": True,
            "legacy_solved_label_not_reused": True,
            "full_synthesis_claim": False,
        },
    }
    portfolio: dict[str, Any] = {
        "schema_version": "proof_stitched_route_portfolio.v1",
        "graph_revision": 1,
        "graph_scientific_sha256": graph["scientific_sha256"],
        "evidence_revision": 1,
        "proof_policy": {"stock_boundary": "benchmark_search", "minimum_edge_proof_level": 2},
        "edge_proofs": edge_proofs,
        "leaf_proofs": {
            leaf_id: {
                "accepted": True,
                "active_stock_observation_id": molecules[leaf_id]["active_stock_observation_id"],
            }
            for leaf_id in leaf_ids
        },
        "route_candidates": [route],
        "selected_routes": [route],
        "route_modules": [],
        "deficits": [
            {
                "deficit_id": "deficit:atorvastatin-current-v4-proof",
                "route_id": route_id,
                "kind": "reaction_validation",
                "edge_ids": edge_ids,
            },
            {
                "deficit_id": "deficit:atorvastatin-exact-conditions",
                "route_id": route_id,
                "kind": "evidence",
                "edge_ids": edge_ids,
            },
        ],
        "metrics": {
            "selected_route_count": 1,
            "complete_route_count": 1,
            "mean_length": float(len(edge_ids)),
            "paper_reported_step_count": 0,
            "planner_hypothesis_step_count": len(edge_ids),
        },
        "closeout": {
            "schema_version": "retrosynthesis_closeout.v1",
            "decision": "route_closed_proof_unresolved",
            "accepted": False,
            "complete_route_count": 1,
            "selected_route_count": 1,
            "deficit_count": 2,
            "reasons": ["current_v4_reaction_proof_missing", "exact_conditions_missing"],
        },
        "accepted": False,
        "semantics": {
            "configured_boundary_route_is_closed": True,
            "legacy_solved_label_is_not_current_authority": True,
            "display_does_not_grant_solved_status": True,
        },
    }
    portfolio["content_sha256"] = _digest(portfolio)
    return compile_route_workbench(
        graph,
        portfolio,
        campaign_summary={
            "gates": {
                "B0_blind_input": True,
                "B1_global_multi_route": True,
                "B2_host_validated_routes": False,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": False,
                "B5_configured_portfolio_acceptance": False,
            },
            "highest_contiguous_gate": "B1_global_multi_route",
            "model_cost": {"model_invocations": 0, "migration_replay": True},
            "resource_envelope": {"within_budget": True},
            "counts": {
                "displayed_route_count": 1,
                "displayed_step_count": len(edge_ids),
                "paper_reported_step_count": 0,
                "planner_hypothesis_step_count": len(edge_ids),
                "source_procedure_count": 0,
            },
            "claim": {
                "status": "route_closed_proof_unresolved",
                "solved": False,
                "configured_boundary_closed": True,
                "closure_profile": "exploration_closed",
            },
            "current_disposition": {
                "route_is_visible": True,
                "low_confidence_edges_are_warning_encoded": True,
                "full_synthesis_claim": False,
            },
        },
    )


def _proof_distribution(workbench: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, int]:
    edges = dict(workbench.get("edges") or {})
    counts: Counter[str] = Counter()
    for edge_id in route.get("edge_ids") or []:
        edge = dict(edges.get(str(edge_id)) or {})
        tier = str(edge.get("proof_name") or edge.get("proof_tier") or "unresolved")
        counts[tier] += 1
    return dict(sorted(counts.items()))


def _source_references(
    workbench: Mapping[str, Any], route: Mapping[str, Any]
) -> set[str]:
    edge_inspectors = dict(dict(workbench.get("inspectors") or {}).get("edges") or {})
    references: set[str] = set()
    for edge_id in route.get("edge_ids") or []:
        inspector = dict(edge_inspectors.get(str(edge_id)) or {})
        for field in ("sources", "exact_records", "source_observation_records"):
            for row in inspector.get(field) or []:
                if not isinstance(row, Mapping):
                    continue
                reference = str(
                    row.get("source_ref")
                    or row.get("document_ref")
                    or row.get("doi")
                    or ""
                ).strip()
                if reference:
                    references.add(reference)
    if not references:
        references.update(
            str(value)
            for value in route.get("independent_source_groups") or []
            if str(value)
        )
    return references


def _rerun_status(root: Path, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "panel_id": f"statin-rerun-{index}",
        "label": f"他汀 V4 重跑组 {index}",
        "root": _relative(root),
        "status": "running" if root.exists() else "queued",
        "target_count": 3,
        "complete_target_count": 0,
    }
    candidates = sorted(
        [
            *root.glob("*panel*status*.json"),
            *root.glob("**/*panel*status*.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if root.is_dir() else []
    if not candidates:
        return row
    try:
        value = _load_json(candidates[0])
    except (OSError, ValueError, json.JSONDecodeError):
        return row
    targets = value.get("targets") or {}
    target_rows = list(targets.values()) if isinstance(targets, Mapping) else list(targets)
    row["status"] = str(value.get("status") or "completed")
    row["target_count"] = len(target_rows) or int(value.get("target_count") or 3)
    row["complete_target_count"] = sum(
        dict(target).get("portfolio", {}).get("metrics", {}).get("complete_route_count", 0) > 0
        or dict(target).get("complete_route_count", 0) > 0
        for target in target_rows
        if isinstance(target, Mapping)
    )
    row["panel_status_path"] = _relative(candidates[0])
    return row


def _write_notes(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# AutoPlanner 今晚展示提要",
        "",
        "## 入口与演示顺序",
        "",
        "- 启动：`python -m cascade_planner.web.app --host 127.0.0.1 --port 8879`",
        "- 打开：`http://127.0.0.1:8879/showcase`",
        "- 先展示蟾毒灵 20 步闭合基准，再点击文献 R 节点展开完整来源条件。",
        "- 切换奈玛特韦说明 L3 双信源闭合；切换阿托伐他汀说明低可信闭合路线仍保留但不冒充验证。",
        "- 点击顶部“他汀谱系 12”说明母体实体、别名去重和扩展重跑队列；“V4 实跑审计”展示真实新架构运行状态。",
        "",
        "## 主线说法",
        "",
        "- 蟾毒灵提供 20 步闭合路线；其中 15 步为论文报道，5 步为规划补全。",
        "- 闭合、反应证明、条件、采购与工艺就绪是彼此独立的状态，不互相冒充。",
        "- 蓝绿/靛蓝表示较强来源或验证，橙色表示规划候选，红色表示结构化拒绝或强警示。",
        "- 点击任意 R 节点可在检查器中展开全部条件、来源过程、缺口和 proof vector。",
        "",
        "## 可展示案例",
        "",
    ]
    for case in manifest.get("cases") or []:
        lines.append(
            f"- **{case['display_name']}**：{case['max_step_count']} 步；"
            f"`{case['claim_status']}`；{case['presentation_note']}"
        )
    lines.extend(
        [
            "",
            "## 他汀范围与实跑状态",
            "",
            f"- 目录按 {manifest.get('statin_catalog', {}).get('entity_count', 0)} 个独立母体实体计数；盐型、活性酸和开发代号不重复计数。",
            "- 当前 readiness 来自 Workbench 全量去重扫描；路线长度、条件、来源、current replay 与 acceptance 分轴显示。",
            "- 低可信候选、短路线和空图继续保留并以警示呈现，不会被目录成员身份升级为闭合路线。",
            "",
            "## 不应声称",
            "",
            "- 不把搜索边界闭合称为全部反应已验证。",
            "- 不把论文中出现过的路线称为已完成采购或工艺放大。",
            "- 不把旧 Blackboard 的 `solved` 标签当作当前 V4 proof authority。",
            "- 紫杉醇和马钱子碱当前只属于研究候选，不进入默认闭合展台。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"json_object_required:{path}")
    return dict(value)


def _load_verified_json(path: Path, *, schema_version: str) -> dict[str, Any]:
    value = _load_json(path)
    digest = value.pop("content_sha256", None)
    if type(digest) is not str or digest != _digest(value):
        raise ValueError(f"json_digest_invalid:{path}")
    if value.get("schema_version") != schema_version:
        raise ValueError(f"json_schema_invalid:{path}")
    value["content_sha256"] = digest
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
