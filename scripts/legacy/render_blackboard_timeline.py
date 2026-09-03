#!/usr/bin/env python3
"""Render agentic blackboard snapshots as a focused static HTML dashboard."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdDepictor, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception:  # pragma: no cover - renderer still works without RDKit.
    Chem = None
    Descriptors = None
    rdDepictor = None
    rdMolDescriptors = None
    rdMolDraw2D = None


STAGE_COLORS = {
    "initialized": "#6b7280",
    "action_batch": "#315f8f",
    "agent_action": "#14806f",
    "auto_critic": "#a05a1a",
    "round_complete": "#3d7a3d",
    "final": "#7b4d8d",
}

_MOLECULE_CACHE: dict[str, dict[str, Any]] = {}

ACTION_LABELS = {
    "run_guided_chemenzy": "运行 ChemEnzy",
    "stitch_parent_route": "编译父路线证明",
    "search_literature": "检索文献",
    "generate_disconnection_hypotheses": "生成断键假设",
    "build_failure_critic_report": "整理失败原因",
    "extract_pdf_literature_structures": "读取 PDF 结构",
    "extract_visual_literature_chain": "提取图文路线",
    "compile_exact_literature_rows": "编译精确文献步骤",
    "expand_child_target": "展开子目标",
    "compile_objective_route_proof": "编译目标证明",
    "stop_unresolved": "停止为未解决",
}

COUNT_LABELS = {
    "retrosynthetic_proposals": "逆合成候选",
    "recursive_hypothesis_tasks": "递归子任务",
    "reaction_idea_cards": "反应想法",
    "bridge_tasks": "桥接任务",
    "broad_transform_templates": "宽泛模板",
    "endpoint_candidates": "终点候选",
    "source_candidates": "文献来源",
    "source_refs": "来源引用",
    "source_lifecycle": "来源状态",
    "exact_rows": "精确文献步骤",
    "visual_chains": "图文路线",
    "pdf_structure_evidence": "PDF 结构证据",
    "structure_resolution_tasks": "结构解析任务",
    "route_failures": "路线问题",
    "proposal_failure_feedback": "候选反馈",
    "blocked_directions": "受阻方向",
    "planner_history": "计划记录",
    "action_history": "动作记录",
    "artifact_refs": "产物文件",
    "parent_route_proof_present": "父路线证明",
    "parent_route_proof_accepted": "父路线已通过",
    "plugin_runtime_diagnostics": "插件诊断",
    "next_action_bias": "下一步建议",
}

MODE_LABELS = {
    "deterministic_policy": "本地规则",
    "deterministic_policy_budget_exhaustive": "本地规则：耗尽预算",
    "deterministic_policy_fast_path_before_codex_planner": "本地快车道",
    "deterministic_policy_fallback_after_codex_planner": "大模型失败后本地接管",
    "codex_blackboard_planner": "规划 worker",
    "codex_blackboard_planner_repaired": "规划 worker：已本地修复",
    "codex_xhigh_blackboard_planner": "规划 worker（旧名）",
    "codex_xhigh_blackboard_planner_repaired": "规划 worker：已本地修复（旧名）",
}

REASON_LABELS = {
    "codex_action_planner_disabled": "跳过规划 worker",
    "deterministic_direct_parent_route_proof_ready": "父路线已验证，可直接证明",
    "direct_parent_route_verifier_ready": "父路线 verifier 已通过",
    "advanced_same_scaffold_terminal": "终点偏高级，但路线 verifier 已接受",
    "worker_error": "规划 worker 出错",
    "timeout": "超时",
    "solved": "已解决",
    "unresolved": "未解决",
}

IMPORTANT_COUNT_KEYS = [
    "retrosynthetic_proposals",
    "recursive_hypothesis_tasks",
    "reaction_idea_cards",
    "bridge_tasks",
    "broad_transform_templates",
    "endpoint_candidates",
    "source_candidates",
    "source_refs",
    "source_lifecycle",
    "exact_rows",
    "visual_chains",
    "pdf_structure_evidence",
    "structure_resolution_tasks",
    "route_failures",
    "proposal_failure_feedback",
    "blocked_directions",
    "planner_history",
    "action_history",
    "artifact_refs",
]

NOISE_COUNT_KEYS = {
    "artifact_refs",
    "planner_history",
    "action_history",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Agentic blackboard run directory.")
    parser.add_argument("--output", default="", help="Output HTML path. Defaults to RUN_DIR/blackboard_timeline.html.")
    parser.add_argument("--item-limit", type=int, default=4, help="Preview limit per signal section.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    step_dir = run_dir / "blackboard_steps"
    summary_path = step_dir / "summary.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"summary.jsonl not found: {summary_path}")
    rows = _read_jsonl(summary_path)
    snapshots = _read_snapshots(step_dir)
    final_verdict = _read_json(run_dir / "final_verdict.json")
    payload = _build_payload(
        run_dir,
        rows,
        snapshots,
        final_verdict=final_verdict,
        item_limit=int(args.item_limit or 4),
    )
    out_path = Path(args.output).expanduser() if args.output else run_dir / "blackboard_timeline.html"
    out_path.write_text(_html(payload), encoding="utf-8")
    print(out_path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_snapshots(step_dir: Path) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for path in sorted(step_dir.glob("*.json")):
        if path.name == "summary.jsonl":
            continue
        data = _read_json(path)
        if not data:
            continue
        step_index = _int(data.get("step_index"))
        if step_index is None:
            step_index = _int(path.stem.split("_", 1)[0])
        if step_index is None:
            continue
        data["_snapshot_file"] = path.name
        snapshots[step_index] = data
    return snapshots


def _build_payload(
    run_dir: Path,
    rows: list[dict[str, Any]],
    snapshots: dict[int, dict[str, Any]],
    *,
    final_verdict: dict[str, Any],
    item_limit: int,
) -> dict[str, Any]:
    previous_counts: dict[str, int] = {}
    steps: list[dict[str, Any]] = []
    all_count_keys: set[str] = set()
    last_blackboard: dict[str, Any] = {}
    for row in rows:
        step_index = int(row.get("step_index") or len(steps) + 1)
        counts = {str(k): int(v or 0) for k, v in dict(row.get("counts") or {}).items()}
        snapshot = snapshots.get(step_index) or {}
        blackboard = _blackboard_payload(snapshot)
        if blackboard:
            counts.update(_counts_from_blackboard(blackboard))
            last_blackboard = blackboard
        all_count_keys.update(counts)
        basic_deltas = {
            key: counts.get(key, 0) - previous_counts.get(key, 0)
            for key in set(counts) | set(previous_counts)
        }
        action_delta, action_after = _rich_action_delta(blackboard, str(row.get("action_id") or ""))
        effective_deltas = action_delta or basic_deltas
        after_counts = action_after or counts
        sections = _signal_sections(blackboard, item_limit=item_limit)
        step = {
            "step_index": step_index,
            "stage": str(row.get("stage") or ""),
            "stage_label": _stage_label(row.get("stage") or ""),
            "round_index": row.get("round_index"),
            "action_id": str(row.get("action_id") or ""),
            "action_type": str(row.get("action_type") or ""),
            "label": _step_label(row),
            "headline": _step_headline(row),
            "impact": _impact_sentence(effective_deltas),
            "status": _step_status(row, blackboard),
            "status_label": _step_status_label(row, blackboard),
            "counts": counts,
            "focus_counts": _focus_counts(counts),
            "top_changes": _top_change_entries(effective_deltas, after_counts),
            "budget_state": row.get("budget_state") or {},
            "budget_bars": _budget_bars(row.get("budget_state") or {}),
            "detail": row.get("detail") or {},
            "last_action": row.get("last_action") or {},
            "last_planner": row.get("last_planner") or {},
            "source_lifecycle_stage_counts": row.get("source_lifecycle_stage_counts") or {},
            "snapshot_file": str(snapshot.get("_snapshot_file") or ""),
            "chemistry": _chemistry_payload(run_dir, blackboard),
            "sections": sections,
            "bottlenecks": _step_bottlenecks(row, blackboard, final_verdict),
            "debug_summary": _debug_summary(row, counts, basic_deltas),
            "filter_tags": _step_filter_tags(row, blackboard),
        }
        steps.append(step)
        previous_counts = counts
    if rows and final_verdict and (not steps or steps[-1].get("stage") != "final"):
        steps.append(_final_step(run_dir, rows[-1], final_verdict, last_blackboard, len(steps) + 1, item_limit=item_limit))

    target_profile = _target_profile(last_blackboard)
    final_counts = dict(steps[-1].get("counts") or {}) if steps else {}
    return {
        "schema_version": "blackboard_dashboard_payload.v2",
        "run_dir": str(run_dir),
        "case_id": str((rows[0].get("case_id") if rows else "") or target_profile.get("target_name") or run_dir.name),
        "target_profile": target_profile,
        "overview": _overview(final_verdict, final_counts, steps),
        "final_verdict": final_verdict,
        "count_keys": _ordered_count_keys(all_count_keys),
        "steps": steps,
        "stage_colors": STAGE_COLORS,
    }


def _final_step(
    run_dir: Path,
    last_row: dict[str, Any],
    final_verdict: dict[str, Any],
    blackboard: dict[str, Any],
    step_index: int,
    *,
    item_limit: int,
) -> dict[str, Any]:
    counts = {str(k): int(v or 0) for k, v in dict(last_row.get("counts") or {}).items()}
    counts.update(_counts_from_blackboard(blackboard))
    solved = bool(final_verdict.get("solved"))
    return {
        "step_index": step_index,
        "stage": "final",
        "stage_label": "最终结论",
        "round_index": last_row.get("round_index"),
        "action_id": "",
        "action_type": "",
        "label": "最终结论",
        "headline": "已解决" if solved else "未解决",
        "impact": _short("；".join(_human_reason(item) for item in final_verdict.get("reasons") or []) or _human_reason(final_verdict.get("verdict") or "")),
        "status": "good" if solved else "bad",
        "status_label": "已解决" if solved else "未解决",
        "counts": counts,
        "focus_counts": _focus_counts(counts),
        "top_changes": [],
        "budget_state": last_row.get("budget_state") or {},
        "budget_bars": _budget_bars(last_row.get("budget_state") or {}),
        "detail": final_verdict,
        "last_action": last_row.get("last_action") or {},
        "last_planner": last_row.get("last_planner") or {},
        "source_lifecycle_stage_counts": last_row.get("source_lifecycle_stage_counts") or {},
        "snapshot_file": "",
        "chemistry": _chemistry_payload(run_dir, blackboard),
        "sections": _signal_sections(blackboard, item_limit=item_limit),
        "bottlenecks": _final_bottlenecks(final_verdict),
        "debug_summary": {"final_verdict": final_verdict, "counts": counts},
        "filter_tags": ["problem"] if not solved else [],
    }


def _counts_from_blackboard(blackboard: dict[str, Any]) -> dict[str, int]:
    if not isinstance(blackboard, dict) or not blackboard:
        return {}
    evidence = dict(blackboard.get("literature_evidence") or {})
    belief = dict(blackboard.get("current_belief") or {})
    route_objective_summary = dict(blackboard.get("route_objective_summary") or {})
    return {
        "action_history": len(blackboard.get("action_history") or []),
        "planner_history": len(blackboard.get("planner_history") or []),
        "source_candidates": len(evidence.get("source_candidates") or []),
        "source_refs": len(evidence.get("source_refs") or []),
        "source_lifecycle": len(evidence.get("source_lifecycle") or []),
        "exact_rows": len(evidence.get("exact_rows") or []),
        "pdf_structure_evidence": len(evidence.get("pdf_structure_evidence") or []),
        "visual_chains": len(evidence.get("visual_chains") or []),
        "structure_resolution_tasks": len(evidence.get("structure_resolution_tasks") or []),
        "route_failures": len(blackboard.get("route_failures") or []),
        "blocked_directions": len(belief.get("blocked_directions") or blackboard.get("blocked_directions") or []),
        "next_action_bias": len(belief.get("next_action_bias") or blackboard.get("next_action_bias") or []),
        "bridge_tasks": len(blackboard.get("bridge_tasks") or []),
        "route_objectives": len(route_objective_summary.get("objectives") or blackboard.get("route_objectives") or []),
        "endpoint_candidates": len(blackboard.get("endpoint_candidates") or []),
        "semisynthesis_anchors": len(blackboard.get("semisynthesis_anchors") or []),
        "reaction_idea_cards": len(blackboard.get("reaction_idea_cards") or []),
        "retrosynthetic_proposals": len(blackboard.get("retrosynthetic_proposals") or []),
        "recursive_hypothesis_tasks": len(blackboard.get("recursive_hypothesis_tasks") or []),
        "proposal_failure_feedback": len(blackboard.get("proposal_failure_feedback") or []),
        "artifact_refs": len(blackboard.get("artifact_refs") or {}),
    }


def _blackboard_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    for key in ("blackboard", "agent_blackboard", "snapshot"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            return value
    return snapshot if isinstance(snapshot, dict) else {}


def _target_profile(blackboard: dict[str, Any]) -> dict[str, Any]:
    profile = dict(blackboard.get("target_profile") or {})
    if not profile:
        return {}
    return {
        "target_name": str(profile.get("target_name") or ""),
        "target_smiles": str(profile.get("target_smiles") or profile.get("isomeric_smiles") or ""),
        "canonical_smiles": str(profile.get("canonical_smiles") or ""),
        "family_hint": str(profile.get("family_hint") or ""),
        "heavy_atoms": profile.get("heavy_atoms"),
        "rings": profile.get("rings"),
        "functional_handles": [str(item) for item in profile.get("functional_handles") or [] if str(item).strip()][:6],
    }


def _chemistry_payload(run_dir: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    target = _target_profile(blackboard)
    target_smiles = str(target.get("target_smiles") or target.get("canonical_smiles") or "")
    payload = {
        "target": _molecule_payload(target_smiles, label=str(target.get("target_name") or "目标分子")),
        "target_profile": target,
        "route": _best_route_payload(run_dir, blackboard),
        "literature": _literature_route_payload(blackboard),
    }
    payload["has_route"] = bool((payload["route"] or {}).get("steps"))
    return payload


def _molecule_payload(smiles: Any, *, label: str = "", stock: bool | None = None) -> dict[str, Any]:
    text = str(smiles or "").strip()
    if not text:
        return {}
    cached = _MOLECULE_CACHE.get(text)
    if cached is None:
        cached = {
            "smiles": text,
            "formula": "",
            "mol_weight": "",
            "heavy_atoms": None,
            "svg": "",
            "valid": False,
        }
        if Chem is not None:
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                try:
                    rdDepictor.Compute2DCoords(mol)
                except Exception:
                    pass
                cached.update(
                    {
                        "formula": rdMolDescriptors.CalcMolFormula(mol) if rdMolDescriptors is not None else "",
                        "mol_weight": f"{Descriptors.MolWt(mol):.1f}" if Descriptors is not None else "",
                        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                        "svg": _mol_svg(mol),
                        "valid": True,
                    }
                )
        _MOLECULE_CACHE[text] = cached
    out = dict(cached)
    out["label"] = label
    if stock is not None:
        out["stock"] = bool(stock)
    return out


def _mol_svg(mol: Any, *, width: int = 260, height: int = 180) -> str:
    if rdMolDraw2D is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg
    except Exception:
        return ""


def _best_route_payload(run_dir: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    if not _blackboard_has_chemenzy_route(blackboard):
        return {}
    raw = _load_chemenzy_raw_result(run_dir, blackboard)
    routes = raw.get("routes") if isinstance(raw, dict) else None
    if not isinstance(routes, list) or not routes:
        return {}
    verifier = dict((blackboard.get("current_belief") or {}).get("parent_route_verifier") or {})
    rank = _int(verifier.get("best_route_rank"))
    selected = _select_route(routes, rank)
    if not selected:
        return {}
    metrics = dict(selected.get("metrics") or {})
    terminal_stock = dict(metrics.get("terminal_stock_status") or {})
    terminals = [
        _molecule_payload(smiles, label="库存原料", stock=terminal_stock.get(str(smiles)))
        for smiles in metrics.get("terminal_reactants") or []
        if str(smiles or "").strip()
    ]
    steps = []
    for row in sorted([item for item in selected.get("steps") or [] if isinstance(item, dict)], key=lambda item: int(item.get("index") or 0)):
        stock_status = dict(row.get("stock_status") or {})
        reactants = []
        for smiles in _unique_smiles([row.get("main_reactant"), *(row.get("aux_reactants") or [])]):
            reactants.append(_molecule_payload(smiles, label="库存原料" if stock_status.get(smiles) else "前体", stock=stock_status.get(smiles)))
        scores = dict(row.get("scores") or {})
        steps.append(
            {
                "index": int(row.get("index") or len(steps)),
                "product": _molecule_payload(row.get("product"), label="产物"),
                "reactants": reactants,
                "reaction_type": str(row.get("reaction_type") or ""),
                "source": str(row.get("source") or ""),
                "confidence": scores.get("confidence"),
                "retro_score": scores.get("retro"),
                "summary": str((row.get("reaction_interpretation") or {}).get("forward_summary") or ""),
            }
        )
    return {
        "rank": selected.get("route_rank"),
        "score": selected.get("score"),
        "n_steps": selected.get("n_steps") or len(steps),
        "status": "solved" if metrics.get("route_solved") else "",
        "strict_stock_solve": bool(metrics.get("strict_stock_solve")),
        "terminal_reactants": terminals,
        "steps": steps,
        "verifier": {
            "accepted": bool(verifier.get("accepted")),
            "target_match": bool(verifier.get("target_match")),
            "accepted_route_count": verifier.get("accepted_route_count"),
            "rejected_route_count": verifier.get("rejected_route_count"),
        },
    }


def _literature_route_payload(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    steps: list[dict[str, Any]] = []
    sources: list[str] = []

    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        source_ref = str(chain.get("source_ref") or chain.get("source_title") or "")
        if source_ref and source_ref not in sources:
            sources.append(source_ref)
        chain_steps = chain.get("steps") or chain.get("candidate_steps") or (chain.get("parsed_output") or {}).get("steps") or []
        for row in chain_steps:
            if not isinstance(row, dict):
                continue
            item = _literature_step_from_row(row, source_ref=source_ref, origin="visual")
            if item:
                steps.append(item)

    for row in evidence.get("exact_rows") or []:
        if not isinstance(row, dict):
            continue
        source_ref = str(row.get("source_ref") or row.get("source_title") or "")
        if source_ref and source_ref not in sources:
            sources.append(source_ref)
        item = _literature_step_from_row(row, source_ref=source_ref, origin="exact")
        if item:
            steps.append(item)

    return {
        "visual_chain_count": len(evidence.get("visual_chains") or []),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "source_refs": sources[:6],
        "steps": steps[:8],
    }


def _literature_step_from_row(row: dict[str, Any], *, source_ref: str, origin: str) -> dict[str, Any]:
    product_smiles = str(
        row.get("product_smiles")
        or row.get("visible_product_smiles")
        or row.get("target_smiles")
        or ""
    ).strip()
    reactant_smiles = _smiles_values(
        row.get("reactant_smiles")
        or row.get("reactants_smiles")
        or row.get("reactants")
        or row.get("precursor_smiles")
        or row.get("substrate_smiles")
        or []
    )
    if not product_smiles and not reactant_smiles:
        return {}
    label = str(
        row.get("product_label")
        or row.get("visible_product_label")
        or row.get("reaction_label")
        or row.get("step_id")
        or origin
    )
    return {
        "label": label,
        "origin": origin,
        "source_ref": source_ref,
        "product": _molecule_payload(product_smiles, label="文献产物") if product_smiles else {},
        "reactants": [_molecule_payload(smiles, label="文献前体") for smiles in reactant_smiles[:4]],
        "condition": _short(
            row.get("condition_text")
            or row.get("reaction_conditions")
            or row.get("conditions")
            or row.get("summary")
            or "",
            220,
        ),
    }


def _smiles_values(value: Any) -> list[str]:
    values: list[Any]
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    elif isinstance(value, dict):
        values = list(value.values())
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, dict):
            text = str(item.get("smiles") or item.get("reactant_smiles") or item.get("value") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _blackboard_has_chemenzy_route(blackboard: dict[str, Any]) -> bool:
    for record in blackboard.get("action_history") or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("action_type") or "") == "run_guided_chemenzy" and bool(record.get("useful_artifact")):
            return True
    return any("chemenzy" in str(path).lower() for path in _artifact_paths(blackboard))


def _load_chemenzy_raw_result(run_dir: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Path] = []
    for path in _artifact_paths(blackboard):
        resolved = _resolve_artifact_path(path, run_dir)
        candidates.append(resolved)
        candidates.append(resolved.parent / "guided_chemenzy_raw_result.json")
    candidates.extend(sorted(run_dir.glob("*chemenzy_raw_result.json")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        data = _read_json(path)
        if _has_route_list(data):
            return data
        nested = data
        for _ in range(4):
            if not isinstance(nested, dict):
                break
            nested = nested.get("result")
            if _has_route_list(nested):
                return nested
    return {}


def _artifact_paths(value: Any) -> list[str]:
    paths: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in {"artifact_ref", "artifact_path"} and isinstance(child, str):
                    paths.append(child)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, str) and (node.endswith(".json") or "\\" in node or "/" in node):
            paths.append(node)

    visit(value.get("artifact_refs") if isinstance(value, dict) else value)
    if isinstance(value, dict):
        visit(value.get("action_history") or [])
    return paths


def _resolve_artifact_path(path: Any, run_dir: Path) -> Path:
    resolved = Path(str(path or ""))
    return resolved if resolved.is_absolute() else run_dir / resolved


def _has_route_list(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("routes"), list) and bool(value.get("routes"))


def _select_route(routes: list[Any], rank: int | None) -> dict[str, Any]:
    dict_routes = [route for route in routes if isinstance(route, dict)]
    if not dict_routes:
        return {}
    if rank is not None:
        for route in dict_routes:
            if _int(route.get("route_rank")) == rank:
                return route
        if 0 <= rank < len(dict_routes):
            return dict_routes[rank]
    return dict_routes[0]


def _unique_smiles(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _rich_action_delta(blackboard: dict[str, Any], action_id: str) -> tuple[dict[str, int], dict[str, int]]:
    if not action_id:
        return {}, {}
    for record in reversed([row for row in blackboard.get("action_history") or [] if isinstance(row, dict)]):
        if str(record.get("action_id") or "") != action_id:
            continue
        delta = {str(k): int(v or 0) for k, v in dict(record.get("blackboard_delta") or {}).items()}
        after = {str(k): int(v or 0) for k, v in dict(record.get("blackboard_counts_after") or {}).items()}
        return delta, after
    return {}, {}


def _overview(final_verdict: dict[str, Any], counts: dict[str, int], steps: list[dict[str, Any]]) -> dict[str, Any]:
    worker_error_rounds = {
        step.get("round_index") or step.get("step_index")
        for step in steps
        if step.get("stage") == "action_batch"
        and str((step.get("last_planner") or {}).get("status") or "") == "worker_error"
    }
    worker_errors = len(worker_error_rounds)
    useful_actions = sum(
        1
        for step in steps
        if step.get("stage") == "agent_action"
        and (
            bool((step.get("detail") or {}).get("useful_artifact"))
            or bool((step.get("last_action") or {}).get("useful_artifact"))
        )
    )
    solved = bool(final_verdict.get("solved"))
    proof_value = counts.get("parent_route_proof_accepted", 0) or (1 if solved else 0)
    reason_text = "；".join(_human_reason(item) for item in final_verdict.get("reasons") or [])
    return {
        "verdict": "已解决" if solved else _human_reason(final_verdict.get("verdict") or "unresolved"),
        "tone": "good" if solved else "bad",
        "primary_bottleneck": reason_text or ("路线已通过确定性证明" if solved else "暂无最终结论"),
        "metrics": [
            {"label": "动作", "value": counts.get("action_history", 0), "sub": f"{useful_actions} 个有效", "tone": "neutral"},
            {"label": "路线候选", "value": counts.get("retrosynthetic_proposals", 0), "sub": "黑板中的候选", "tone": "good" if counts.get("retrosynthetic_proposals", 0) else "warn"},
            {"label": "证据", "value": counts.get("source_candidates", 0), "sub": f"{counts.get('exact_rows', 0)} 条精确步骤", "tone": "good" if counts.get("exact_rows", 0) else "warn"},
            {"label": "证明", "value": proof_value, "sub": "父路线证明通过", "tone": "good" if proof_value else "bad"},
            {"label": "Worker 问题", "value": worker_errors, "sub": "规划 worker 错误", "tone": "bad" if worker_errors else "good"},
        ],
    }


def _focus_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    groups = [
        ("路线", ["retrosynthetic_proposals", "recursive_hypothesis_tasks", "bridge_tasks", "broad_transform_templates"]),
        ("证据", ["source_candidates", "exact_rows", "visual_chains", "pdf_structure_evidence"]),
        ("证明", ["parent_route_proof_present", "parent_route_proof_accepted", "route_failures"]),
        ("过程", ["action_history", "planner_history"]),
    ]
    out: list[dict[str, Any]] = []
    for label, keys in groups:
        out.append({"label": label, "items": [{"key": key, "label": _count_label(key), "value": counts.get(key, 0)} for key in keys]})
    return out


def _budget_bars(budget: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = [
        ("轮次", "rounds_completed", "max_rounds"),
        ("ChemEnzy", "chemenzy_runs", "max_chemenzy_runs"),
        ("Codex 检索", "codex_research_runs", "max_codex_research_runs"),
        ("文献检索", "scout_calls", "max_scout_calls"),
        ("视觉提取", "visual_calls", "max_visual_calls"),
        ("模板尝试", "template_application_actions", "max_template_application_actions"),
    ]
    bars: list[dict[str, Any]] = []
    for label, used_key, max_key in pairs:
        used = _int(budget.get(used_key)) or 0
        max_value = _int(budget.get(max_key)) or 0
        pct = min(100, max(0, int(round((used / max_value) * 100)))) if max_value > 0 else 0
        tone = "warn" if max_value > 0 and used >= max_value else "neutral"
        bars.append({"label": label, "used": used, "max": max_value, "pct": pct, "tone": tone})
    return bars


def _signal_sections(blackboard: dict[str, Any], *, item_limit: int) -> list[dict[str, Any]]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    current_belief = dict(blackboard.get("current_belief") or {})
    parent_verifier = dict(current_belief.get("parent_route_verifier") or {})
    parent_proof = dict(blackboard.get("parent_route_proof") or {})
    decision_items = [
        *_summarize_items(blackboard.get("planner_history") or [], _summarize_planner, item_limit=2),
        _summary_item(
            "下一步建议",
            _action_list(current_belief.get("next_action_bias") or []) or "暂无",
            tone="neutral",
        ),
        *_summarize_items(blackboard.get("action_history") or [], _summarize_action, item_limit=3),
    ]
    proof_items = [
        _summarize_parent_verifier(parent_verifier) if parent_verifier else {},
        _summarize_parent_proof(parent_proof) if parent_proof else {},
    ]
    route_items = [
        *_summarize_items(blackboard.get("retrosynthetic_proposals") or [], _summarize_proposal, item_limit=min(item_limit, 3)),
        *_summarize_items(blackboard.get("reaction_idea_cards") or [], _summarize_reaction_idea, item_limit=1),
        *_summarize_items(blackboard.get("bridge_tasks") or [], _summarize_bridge, item_limit=1),
        *_summarize_items(blackboard.get("broad_transform_templates") or [], _summarize_template, item_limit=1),
    ]
    evidence_items = [
        *_summarize_items(evidence.get("source_candidates") or [], _summarize_source, item_limit=min(item_limit, 3)),
        *_summarize_items(evidence.get("exact_rows") or [], _summarize_exact_row, item_limit=2),
        *_summarize_items(evidence.get("visual_chains") or [], _summarize_visual_chain, item_limit=2),
        *_summarize_items(evidence.get("scout_attempts") or [], _summarize_scout_attempt, item_limit=2),
    ]
    risk_items = [
        *_summarize_items(current_belief.get("blocked_directions") or [], _summarize_blocked, item_limit=item_limit),
        *_summarize_items(blackboard.get("route_failures") or [], _summarize_failure, item_limit=item_limit),
        *_summarize_items(blackboard.get("proposal_failure_feedback") or [], _summarize_failure, item_limit=item_limit),
        *_summarize_items(blackboard.get("terminal_blacklist") or [], _summarize_blocked, item_limit=2),
    ]
    return [
        {"id": "decision", "title": "决策", "items": _clean_items(decision_items), "empty": "还没有计划或动作记录。"},
        {"id": "proof", "title": "结论依据", "items": _clean_items(proof_items), "empty": "还没有父路线 verifier 或证明。"},
        {"id": "route", "title": "路线", "items": _clean_items(route_items), "empty": "黑板上还没有额外路线候选。"},
        {"id": "evidence", "title": "证据", "items": _clean_items(evidence_items), "empty": "还没有来源或精确文献步骤。"},
        {"id": "risk", "title": "问题", "items": _clean_items(risk_items), "empty": "当前没有明确问题记录。"},
    ]


def _summarize_items(rows: Any, summarizer: Any, *, item_limit: int) -> list[dict[str, Any]]:
    if isinstance(rows, dict):
        values = [{"key": key, "value": value} for key, value in rows.items()]
    elif isinstance(rows, list):
        values = rows
    else:
        values = []
    out: list[dict[str, Any]] = []
    for row in values[-item_limit:]:
        if isinstance(row, dict):
            out.append(summarizer(row))
        elif row not in (None, "", [], {}):
            out.append(_summary_item(_short(row), "", tone="neutral"))
    return out


def _summary_item(
    title: Any,
    note: Any = "",
    *,
    meta: list[str] | None = None,
    badges: list[str] | None = None,
    tone: str = "neutral",
) -> dict[str, Any]:
    return {
        "title": _short(title, 120),
        "note": _short(note, 260),
        "meta": [_short(item, 90) for item in (meta or []) if str(item).strip()],
        "badges": [_short(item, 64) for item in (badges or []) if str(item).strip()],
        "tone": tone,
    }


def _clean_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("title") or item.get("note") or item.get("badges")]


def _summarize_parent_verifier(row: dict[str, Any]) -> dict[str, Any]:
    accepted = bool(row.get("accepted"))
    solved = bool(row.get("solved")) or str(row.get("route_status") or "") == "solved"
    count = _int(row.get("accepted_route_count")) or 0
    rejected = _int(row.get("rejected_route_count")) or 0
    reasons = [_human_reason(item) for item in row.get("reasons") or []]
    return _summary_item(
        "父路线 verifier",
        "已通过，目标匹配，可进入父路线证明。" if accepted and solved else "尚未通过，需要继续解释失败原因。",
        meta=[
            f"通过路线 {count} 条",
            f"拒绝路线 {rejected} 条" if rejected else "",
            "目标匹配" if row.get("target_match") else "目标未确认",
        ],
        badges=reasons[:2],
        tone="good" if accepted and solved else "warn",
    )


def _summarize_parent_proof(row: dict[str, Any]) -> dict[str, Any]:
    accepted = bool(row.get("accepted") and row.get("solved"))
    mode = str(row.get("proof_mode") or "")
    mode_text = "直接父路线证明" if mode == "direct_parent_route" else "拼接父路线证明"
    clauses = dict(row.get("proof_clauses") or {})
    passed = sum(1 for value in clauses.values() if value is True)
    total = len(clauses)
    reasons = [_human_reason(item) for item in row.get("reasons") or []]
    return _summary_item(
        "父路线证明",
        f"{mode_text}已通过。" if accepted else "父路线证明尚未通过。",
        meta=[f"证明条款 {passed}/{total}", _human_reason(row.get("route_status") or "")],
        badges=reasons[:2],
        tone="good" if accepted else "bad",
    )


def _summarize_planner(row: dict[str, Any]) -> dict[str, Any]:
    codex = dict(row.get("codex_action_planner") or {})
    status = str(codex.get("status") or row.get("status") or "")
    fallback = bool(codex.get("fallback_used"))
    mode = str(row.get("mode") or "")
    fallback_reason = str(codex.get("fallback_reason") or "")
    fast_path = bool(codex.get("fast_path_used")) or mode == "deterministic_policy_fast_path_before_codex_planner" or bool((codex.get("tool_policy") or {}).get("codex_worker_bypassed"))
    disabled = bool(codex.get("planner_disabled")) or fallback_reason == "codex_action_planner_disabled"
    actions = _action_list(row.get("action_types") or [])
    if disabled:
        note = "本轮直接使用本地规则。"
    elif fast_path:
        note = _human_reason(codex.get("fast_path_reason") or "") or "满足确定性快车道。"
    elif fallback:
        note = _human_reason(fallback_reason) or "规划 worker 出错，本地规则接管。"
    else:
        note = actions or "计划已记录。"
    return _summary_item(
        f"第 {row.get('round_index') or '?'} 轮计划",
        note,
        meta=[_mode_label(mode), actions],
        badges=[
            "本地快车道" if fast_path else ("本地规则" if disabled else ("本地接管" if fallback else "规划 worker")),
            _human_reason(status) if status else "",
        ],
        tone="bad" if status == "worker_error" else ("warn" if fallback else "good"),
    )


def _summarize_action(row: dict[str, Any]) -> dict[str, Any]:
    delta = dict(row.get("blackboard_delta") or {})
    changes = _impact_sentence({str(k): int(v or 0) for k, v in delta.items()})
    useful = bool(row.get("useful_artifact"))
    action_type = str(row.get("action_type") or "")
    return _summary_item(
        _action_label(action_type) or row.get("action_id") or "动作",
        changes or str(row.get("status") or ""),
        meta=[str(row.get("action_id") or ""), _human_reason(row.get("status") or "")],
        badges=["有效" if useful else "未产生新内容"],
        tone="good" if useful else "warn",
    )


def _summarize_proposal(row: dict[str, Any]) -> dict[str, Any]:
    precursor = str(row.get("precursor_smiles") or "")
    score = row.get("score")
    return _summary_item(
        row.get("proposal_label") or row.get("proposal_id") or "路线候选",
        row.get("transformation_idea") or row.get("route_objective_type") or "",
        meta=[
            f"前体：{precursor}" if precursor else "",
            f"评分：{score}" if score not in (None, "") else "",
            str(row.get("proposal_granularity") or ""),
        ],
        badges=[str(row.get("confidence") or ""), *[str(item) for item in row.get("risk_flags") or []][:2]],
        tone="good" if str(row.get("confidence") or "").lower() in {"high", "medium_high"} else "neutral",
    )


def _summarize_reaction_idea(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("target_handle") or row.get("card_id") or "反应想法",
        row.get("transformation_idea") or "",
        meta=[str(row.get("source_type") or ""), str(row.get("expected_precursor_type") or "")],
        badges=[str(row.get("confidence") or ""), *[str(item) for item in row.get("risk_flags") or []][:2]],
        tone="neutral",
    )


def _summarize_bridge(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("task_id") or "桥接任务",
        row.get("required_bridge") or "",
        meta=[str(row.get("task_type") or ""), str(row.get("status") or "")],
        badges=[str(item) for item in row.get("required_verification") or []][:3],
        tone="warn" if str(row.get("status") or "").lower() in {"open", "pending"} else "good",
    )


def _summarize_template(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("template_id") or "宽泛模板",
        row.get("transform_logic") or "",
        meta=[str(row.get("reaction_center") or ""), str(row.get("objective_type") or "")],
        badges=[str(row.get("allowed_use") or ""), *[str(item) for item in row.get("risk_flags") or []][:2]],
        tone="neutral",
    )


def _summarize_source(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("access_status") or "")
    role = str(row.get("source_role") or row.get("source_type") or "")
    return _summary_item(
        row.get("title") or row.get("source_ref") or row.get("doi") or "来源",
        row.get("relevance_rationale") or row.get("route_sequence_hint") or "",
        meta=[str(row.get("doi") or ""), status, role],
        badges=[str(row.get("source_discovery_mode") or ""), "用户提供" if row.get("user_provided_source_seed") else ""],
        tone="good" if "available" in status else ("warn" if "placeholder" in status else "neutral"),
    )


def _summarize_exact_row(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("row_id") or row.get("reaction_label") or "精确文献步骤",
        row.get("route_sequence_hint") or row.get("summary") or "",
        meta=[str(row.get("source_ref") or ""), str(row.get("validation_status") or "")],
        badges=[str(row.get("allowed_use") or "")],
        tone="good",
    )


def _summarize_visual_chain(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("chain_id") or row.get("source_ref") or "图文路线",
        row.get("summary") or row.get("route_sequence_hint") or "",
        meta=[str(row.get("source_ref") or ""), str(row.get("validation_status") or "")],
        tone="good",
    )


def _summarize_scout_attempt(row: dict[str, Any]) -> dict[str, Any]:
    attempted = bool(row.get("attempted"))
    accepted = bool(row.get("accepted"))
    return _summary_item(
        row.get("mode") or "文献检索",
        row.get("reason") or "",
        meta=[f"已尝试：{'是' if attempted else '否'}", f"已接受：{'是' if accepted else '否'}"],
        badges=[f"候选：{row.get('candidate_count')}" if row.get("candidate_count") is not None else ""],
        tone="good" if accepted else ("warn" if attempted else "neutral"),
    )


def _summarize_failure(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("failure_id") or row.get("proposal_id") or row.get("category") or "问题",
        _human_reason(row.get("reason") or row.get("message") or row.get("summary") or ""),
        meta=[str(row.get("action_id") or ""), _human_reason(row.get("status") or "")],
        badges=[str(item) for item in row.get("risk_flags") or []][:3],
        tone="bad",
    )


def _summarize_blocked(row: dict[str, Any]) -> dict[str, Any]:
    return _summary_item(
        row.get("direction") or row.get("blocked_direction") or row.get("key") or row.get("value") or "受阻方向",
        _human_reason(row.get("reason") or row.get("message") or ""),
        meta=[_human_reason(row.get("status") or "")],
        tone="warn",
    )


def _step_label(row: dict[str, Any]) -> str:
    action = str(row.get("action_type") or "").strip()
    if action:
        return _action_label(action)
    stage = str(row.get("stage") or "").strip()
    if stage == "action_batch":
        actions = (row.get("last_planner") or {}).get("action_types") or []
        return "计划：" + _action_list(actions[:3])
    return _stage_label(stage)


def _step_headline(row: dict[str, Any]) -> str:
    stage = str(row.get("stage") or "")
    action = str(row.get("action_type") or "")
    if stage == "initialized":
        return "黑板已初始化"
    if stage == "action_batch":
        detail = dict(row.get("detail") or {})
        return f"计划了 {detail.get('action_count') or 0} 个动作"
    if stage == "agent_action":
        return _action_label(action)
    if stage == "round_complete":
        return f"第 {row.get('round_index') or ''} 轮完成".strip()
    if stage == "auto_critic":
        return "自动检查"
    return _stage_label(stage)


def _step_status(row: dict[str, Any], blackboard: dict[str, Any]) -> str:
    stage = str(row.get("stage") or "")
    detail = dict(row.get("detail") or {})
    planner = dict(row.get("last_planner") or {})
    last_action = dict(row.get("last_action") or {})
    if stage == "final":
        return "good" if bool(row.get("detail", {}).get("solved")) else "bad"
    if str(planner.get("status") or "") == "worker_error" and stage == "action_batch":
        return "warn"
    if detail.get("accepted") is False:
        return "bad"
    if last_action.get("useful_artifact") is False and stage == "agent_action":
        return "warn"
    if detail.get("useful_artifact") is True or last_action.get("useful_artifact") is True:
        return "good"
    if blackboard.get("route_failures") or blackboard.get("proposal_failure_feedback"):
        return "warn"
    return "neutral"


def _step_status_label(row: dict[str, Any], blackboard: dict[str, Any]) -> str:
    stage = str(row.get("stage") or "")
    detail = dict(row.get("detail") or {})
    planner = dict(row.get("last_planner") or {})
    if stage == "action_batch" and str(planner.get("status") or "") == "worker_error":
        return "规划异常"
    if detail.get("accepted") is False:
        return "已拒绝"
    if detail.get("useful_artifact") is True:
        return "有效"
    if detail.get("useful_artifact") is False:
        return "无新增"
    if blackboard.get("route_failures") or blackboard.get("proposal_failure_feedback"):
        return "有问题"
    return _stage_label(stage)


def _step_filter_tags(row: dict[str, Any], blackboard: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    if str(row.get("stage") or "") == "action_batch":
        tags.add("decision")
    if str(row.get("stage") or "") == "agent_action":
        tags.add("action")
    if _step_status(row, blackboard) in {"warn", "bad"}:
        tags.add("problem")
    return sorted(tags)


def _step_bottlenecks(row: dict[str, Any], blackboard: dict[str, Any], final_verdict: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    planner = dict(row.get("last_planner") or {})
    if str(planner.get("status") or "") == "worker_error":
        items.append({"label": "规划 worker", "text": "规划 worker 出错，已由本地规则接管。", "tone": "warn"})
    evidence = dict(blackboard.get("literature_evidence") or {})
    if evidence.get("source_candidates") and not evidence.get("exact_rows"):
        items.append({"label": "证据", "text": "已有来源，但还没有编译成精确文献步骤。", "tone": "warn"})
    if blackboard.get("retrosynthetic_proposals") and not blackboard.get("parent_route_proof"):
        items.append({"label": "路线证明", "text": "候选路线只是建议，必须等父路线证明通过才算解决。", "tone": "bad"})
    if final_verdict and not final_verdict.get("solved"):
        reason = "；".join(_human_reason(item) for item in final_verdict.get("reasons") or [])
        if reason:
            items.append({"label": "最终结论", "text": reason, "tone": "bad"})
    return items[:4]


def _final_bottlenecks(final_verdict: dict[str, Any]) -> list[dict[str, str]]:
    if bool(final_verdict.get("solved")):
        return [{"label": "最终结论", "text": "确定性 solved 结论已通过。", "tone": "good"}]
    reasons = [_human_reason(item) for item in final_verdict.get("reasons") or [] if str(item).strip()]
    return [{"label": "最终结论", "text": reason, "tone": "bad"} for reason in reasons[:4]]


def _top_change_entries(deltas: dict[str, int], counts: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in _ordered_count_keys(set(deltas) | set(counts)):
        delta = int(deltas.get(key, 0) or 0)
        if delta == 0:
            continue
        rows.append(
            {
                "key": key,
                "label": _count_label(key),
                "delta": delta,
                "value": int(counts.get(key, 0) or 0),
                "tone": "muted" if key in NOISE_COUNT_KEYS else ("good" if delta > 0 else "bad"),
            }
        )
    rows.sort(key=lambda row: (row["key"] in NOISE_COUNT_KEYS, -abs(int(row["delta"]))))
    return rows[:8]


def _impact_sentence(deltas: dict[str, int]) -> str:
    entries = _top_change_entries(deltas, {})
    entries = [row for row in entries if row["key"] not in NOISE_COUNT_KEYS][:5]
    if not entries:
        return "黑板没有实质变化。"
    parts = []
    for row in entries:
        sign = "+" if int(row["delta"]) > 0 else ""
        parts.append(f"{sign}{row['delta']} {row['label']}")
    return "，".join(parts)


def _debug_summary(row: dict[str, Any], counts: dict[str, int], deltas: dict[str, int]) -> dict[str, Any]:
    return {
        "step_index": row.get("step_index"),
        "stage": row.get("stage"),
        "round_index": row.get("round_index"),
        "action_id": row.get("action_id"),
        "action_type": row.get("action_type"),
        "detail": row.get("detail") or {},
        "last_planner": row.get("last_planner") or {},
        "last_action": row.get("last_action") or {},
        "counts": counts,
        "deltas": {key: value for key, value in deltas.items() if value},
    }


def _ordered_count_keys(keys: set[str]) -> list[str]:
    ordered = [key for key in IMPORTANT_COUNT_KEYS if key in keys]
    rest = sorted(key for key in keys if key not in set(ordered))
    return ordered + rest


def _action_label(value: Any) -> str:
    text = str(value or "").strip()
    return ACTION_LABELS.get(text, text.replace("_", " ") if text else "")


def _action_list(values: Any) -> str:
    if not isinstance(values, list):
        return _action_label(values)
    return "，".join(_action_label(item) for item in values if str(item or "").strip())


def _mode_label(value: Any) -> str:
    text = str(value or "").strip()
    return MODE_LABELS.get(text, text.replace("_", " ") if text else "")


def _human_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return REASON_LABELS.get(text, text.replace("_", " "))


def _count_label(key: Any) -> str:
    text = str(key or "").strip()
    return COUNT_LABELS.get(text, text.replace("_", " "))


def _stage_label(value: Any) -> str:
    stage = str(value or "").strip()
    mapping = {
        "initialized": "初始化",
        "action_batch": "计划",
        "agent_action": "动作",
        "auto_critic": "自动检查",
        "round_complete": "轮次完成",
        "final": "最终结论",
    }
    return mapping.get(stage, stage.replace("_", " "))


def _short(value: Any, limit: int = 180) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(str(payload.get("case_id") or "黑板"))
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>黑板流程 - __TITLE__</title>
  <style>
    :root {
      --bg: #f4f6f7;
      --surface: #ffffff;
      --surface-soft: #f9faf8;
      --ink: #17202a;
      --muted: #69727d;
      --line: #d9dfdf;
      --line-strong: #bac4c4;
      --blue: #315f8f;
      --green: #14806f;
      --amber: #a05a1a;
      --red: #b33b35;
      --violet: #7b4d8d;
      --radius: 8px;
      --shadow: 0 12px 28px rgba(27, 39, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button { font: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .topbar-inner {
      max-width: 1640px;
      margin: 0 auto;
      padding: 14px 22px 12px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 18px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 760;
      letter-spacing: 0;
    }
    .target-line {
      color: var(--muted);
      margin-top: 4px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      min-width: 0;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      min-height: 24px;
      max-width: 100%;
    }
    .pill.good { color: var(--green); border-color: rgba(20,128,111,.35); background: #f3fbf8; }
    .pill.warn { color: var(--amber); border-color: rgba(160,90,26,.35); background: #fff8ec; }
    .pill.bad { color: var(--red); border-color: rgba(179,59,53,.35); background: #fff2f0; }
    .main {
      max-width: 1640px;
      margin: 0 auto;
      padding: 18px 22px 30px;
      display: grid;
      gap: 16px;
    }
    .overview {
      display: grid;
      grid-template-columns: minmax(280px, 1.35fr) repeat(5, minmax(120px, .7fr));
      gap: 10px;
    }
    .verdict-band,
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .verdict-band {
      padding: 14px;
      display: grid;
      gap: 8px;
      border-left: 6px solid var(--red);
    }
    .verdict-band.good { border-left-color: var(--green); }
    .metric {
      padding: 12px;
      min-width: 0;
    }
    .metric-label,
    .section-kicker {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .metric-value {
      margin-top: 5px;
      font-size: 25px;
      font-weight: 780;
      line-height: 1;
    }
    .metric-sub {
      margin-top: 5px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(260px, 320px) minmax(560px, 1.45fr) minmax(320px, .82fr);
      gap: 16px;
      align-items: start;
    }
    .timeline-panel { grid-column: 1; }
    .signals-panel {
      grid-column: 2;
      grid-row: 1;
      border-color: rgba(49,95,143,.28);
      min-height: calc(100vh - 210px);
    }
    .step-panel {
      grid-column: 3;
      grid-row: 1;
    }
    .timeline-panel,
    .step-panel {
      position: sticky;
      top: 86px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-head {
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .panel-title {
      margin: 0;
      font-size: 15px;
      font-weight: 760;
    }
    .flow-tools {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }
    .seg {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      border-radius: 6px;
      padding: 7px 8px;
      cursor: pointer;
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .seg.active {
      border-color: var(--blue);
      color: var(--blue);
      background: #eef6ff;
    }
    .step-list {
      max-height: calc(100vh - 250px);
      overflow: auto;
      padding: 8px;
      display: grid;
      gap: 6px;
    }
    .step-btn {
      width: 100%;
      text-align: left;
      background: #fff;
      border: 1px solid var(--line);
      border-left: 5px solid var(--stage, var(--line-strong));
      border-radius: 7px;
      padding: 9px 10px;
      cursor: pointer;
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .step-btn:hover,
    .step-btn.active {
      border-color: var(--line-strong);
      background: var(--surface-soft);
    }
    .step-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .step-name {
      font-weight: 720;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .step-impact {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .focus {
      display: grid;
      gap: 12px;
      padding: 12px;
    }
    .headline {
      display: grid;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 13px;
    }
    .headline h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .headline-note {
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .change-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .change {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      background: #fff;
      min-width: 0;
    }
    .change strong {
      display: block;
      font-size: 18px;
      line-height: 1;
      margin-bottom: 5px;
    }
    .change span {
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .bar-list {
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 130px 1fr 54px;
      gap: 8px;
      align-items: center;
    }
    .bar-name,
    .bar-value {
      color: var(--muted);
      font-size: 12px;
    }
    .bar-value { text-align: right; }
    .bar-track {
      height: 8px;
      background: #e8eded;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: var(--pct);
      background: var(--blue);
    }
    .bar-fill.warn { background: var(--amber); }
    .bottlenecks {
      display: grid;
      gap: 7px;
    }
    .notice {
      border: 1px solid var(--line);
      border-left: 5px solid var(--amber);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
    }
    .notice.bad { border-left-color: var(--red); }
    .notice.good { border-left-color: var(--green); }
    .notice-label {
      font-weight: 740;
      margin-bottom: 3px;
    }
    .notice-text {
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .chemistry {
      display: grid;
      gap: 14px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #f8fbfb 100%);
    }
    .chemistry-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      min-width: 0;
    }
    .chemistry-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .target-body {
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      align-items: center;
    }
    .target-meta {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .formula {
      font-size: 28px;
      font-weight: 780;
      line-height: 1;
    }
    .route-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .route-steps {
      display: grid;
      gap: 0;
    }
    .route-step {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .route-step:last-child { border-bottom: 0; }
    .route-step-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .route-equation {
      display: grid;
      grid-template-columns: minmax(170px, .9fr) 34px minmax(220px, 1.1fr);
      gap: 10px;
      align-items: center;
    }
    .route-arrow {
      color: var(--blue);
      font-size: 24px;
      font-weight: 780;
      text-align: center;
    }
    .route-reactants {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      min-width: 0;
    }
    .mol-tile {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .mol-svg {
      min-height: 118px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .mol-svg svg {
      width: 100%;
      height: auto;
      max-height: 180px;
      display: block;
    }
    .mol-placeholder {
      color: var(--muted);
      padding: 12px;
      text-align: center;
    }
    .mol-caption {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-width: 0;
    }
    .mol-name {
      font-weight: 720;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .mol-smiles {
      color: var(--muted);
      font-size: 11px;
    }
    .stock-chip {
      color: var(--green);
      border-color: rgba(20,128,111,.35);
      background: #f3fbf8;
    }
    .blackboard-summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .summary-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      display: grid;
      gap: 5px;
      min-width: 0;
      background: var(--surface-soft);
    }
    .summary-card.good { border-left: 5px solid var(--green); }
    .summary-card.warn { border-left: 5px solid var(--amber); }
    .summary-card.bad { border-left: 5px solid var(--red); }
    .summary-value {
      font-size: 22px;
      line-height: 1;
      font-weight: 780;
    }
    details.signal-details {
      border-top: 1px solid var(--line);
    }
    details.signal-details summary {
      cursor: pointer;
      padding: 12px 14px;
      color: var(--muted);
      font-weight: 700;
      list-style-position: inside;
    }
    .signals {
      display: grid;
      gap: 12px;
      padding: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: start;
    }
    .signal-section {
      display: grid;
      gap: 8px;
      align-content: start;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px;
      background: var(--surface-soft);
      min-width: 0;
    }
    .signal-section.risk {
      grid-column: 1 / -1;
    }
    .signal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--line);
    }
    .item-row {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      background: #fff;
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .item-row.good { border-left: 5px solid var(--green); }
    .item-row.warn { border-left: 5px solid var(--amber); }
    .item-row.bad { border-left: 5px solid var(--red); }
    .item-title {
      font-weight: 720;
      overflow-wrap: anywhere;
    }
    .item-note {
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .meta-line,
    .badge-line {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      min-width: 0;
    }
    .tag {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line-strong);
      border-radius: 7px;
      padding: 10px;
      background: var(--surface-soft);
    }
    details.debug {
      border-top: 1px solid var(--line);
      padding: 0;
    }
    details.debug summary {
      cursor: pointer;
      padding: 11px 14px;
      color: var(--muted);
      font-weight: 650;
    }
    pre {
      margin: 0;
      padding: 0 14px 14px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      color: #2f3b45;
    }
    @media (max-width: 1220px) {
      .overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .verdict-band { grid-column: 1 / -1; }
      .workspace { grid-template-columns: minmax(230px, 300px) minmax(0, 1fr); }
      .timeline-panel { grid-column: 1; grid-row: 1 / span 2; }
      .signals-panel { grid-column: 2; grid-row: 1; }
      .step-panel { grid-column: 2; grid-row: 2; }
      .target-body { grid-template-columns: 1fr; }
      .route-equation { grid-template-columns: 1fr; }
      .route-arrow { transform: rotate(90deg); }
      .blackboard-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 820px) {
      .topbar-inner { grid-template-columns: 1fr; }
      .main { padding: 12px; }
      .overview { grid-template-columns: 1fr; }
      .workspace {
        display: flex;
        flex-direction: column;
      }
      .signals-panel { order: 1; min-height: 0; }
      .step-panel { order: 2; }
      .timeline-panel { order: 3; }
      .timeline-panel,
      .step-panel { position: static; }
      .target-body { grid-template-columns: 1fr; }
      .route-equation { grid-template-columns: 1fr; }
      .route-arrow { transform: rotate(90deg); }
      .blackboard-summary { grid-template-columns: 1fr; }
      .signals { grid-template-columns: 1fr; }
      .signal-section.risk { grid-column: auto; }
      .step-list { max-height: 360px; }
      .change-grid { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 108px 1fr 48px; }
      .flow-tools { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <h1 id="pageTitle"></h1>
        <div class="target-line" id="targetLine"></div>
      </div>
      <div id="verdictPill"></div>
    </div>
  </div>
  <main class="main">
    <div class="overview" id="overview"></div>
    <div class="workspace">
      <section class="panel timeline-panel">
        <div class="panel-head">
          <h2 class="panel-title">步骤索引</h2>
          <span class="pill" id="stepCount"></span>
        </div>
        <div class="flow-tools" id="flowTools"></div>
        <div class="step-list" id="stepList"></div>
      </section>
      <section class="panel step-panel">
        <div class="panel-head">
          <h2 class="panel-title">步骤说明</h2>
          <span class="pill" id="snapshotName"></span>
        </div>
        <div class="focus">
          <div class="headline">
            <div class="section-kicker" id="stepKicker"></div>
            <h2 id="stepHeadline"></h2>
            <div class="headline-note" id="stepImpact"></div>
            <div class="meta-line" id="stepMeta"></div>
          </div>
          <div>
            <div class="section-kicker">本步变化</div>
            <div class="change-grid" id="changes"></div>
          </div>
          <div>
            <div class="section-kicker">资源使用</div>
            <div class="bar-list" id="budgetBars"></div>
          </div>
          <div>
            <div class="section-kicker">提示</div>
            <div class="bottlenecks" id="bottlenecks"></div>
          </div>
        </div>
        <details class="debug">
          <summary>调试摘要</summary>
          <pre id="debugJson"></pre>
        </details>
      </section>
      <section class="panel signals-panel">
        <div class="panel-head">
          <h2 class="panel-title">核心黑板</h2>
          <span class="pill" id="signalScope"></span>
        </div>
        <div class="chemistry" id="chemistry"></div>
        <div class="blackboard-summary" id="blackboardSummary"></div>
        <details class="signal-details">
          <summary>展开过程细节</summary>
          <div class="signals" id="signals"></div>
        </details>
      </section>
    </div>
  </main>
  <script>
    const DATA = __DATA__;
    const stageColors = DATA.stage_colors || {};
    let selected = Math.max(0, (DATA.steps || []).length - 1);
    let filter = "all";
    const filters = [
      ["all", "全部"],
      ["action", "动作"],
      ["decision", "计划"],
      ["problem", "问题"],
    ];

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch]));
    }
    function short(value, limit=120) {
      const text = String(value ?? "");
      return text.length <= limit ? text : text.slice(0, limit - 3) + "...";
    }
    function toneClass(tone) {
      return ["good", "warn", "bad"].includes(tone) ? tone : "";
    }
    function stepVisible(step) {
      if (filter === "all") return true;
      return (step.filter_tags || []).includes(filter);
    }
    function currentStep() {
      return (DATA.steps || [])[selected] || {};
    }
    function renderShell() {
      const target = DATA.target_profile || {};
      document.getElementById("pageTitle").textContent = `黑板流程：${DATA.case_id || target.target_name || "case"}`;
      const targetBits = [
        target.family_hint,
        target.heavy_atoms ? `${target.heavy_atoms} 个重原子` : "",
        target.rings !== undefined && target.rings !== null ? `${target.rings} 个环` : "",
        target.target_smiles ? `SMILES ${target.target_smiles}` : "",
      ].filter(Boolean);
      document.getElementById("targetLine").innerHTML = targetBits.map(bit => `<span>${esc(bit)}</span>`).join("");
      const overview = DATA.overview || {};
      document.getElementById("verdictPill").innerHTML = `<span class="pill ${toneClass(overview.tone)}">${esc(overview.verdict || "unknown")}</span>`;
      document.getElementById("stepCount").textContent = `${(DATA.steps || []).length} 步`;
      document.getElementById("flowTools").innerHTML = filters.map(([id, label]) =>
        `<button class="seg ${filter === id ? "active" : ""}" data-filter="${esc(id)}">${esc(label)}</button>`
      ).join("");
      document.querySelectorAll("[data-filter]").forEach(btn => {
        btn.addEventListener("click", () => {
          filter = btn.getAttribute("data-filter") || "all";
          const visible = (DATA.steps || []).map((step, index) => [step, index]).filter(([step]) => stepVisible(step));
          if (!visible.some(([, index]) => index === selected) && visible.length) selected = visible[visible.length - 1][1];
          renderAll();
        });
      });
      renderOverview();
    }
    function renderOverview() {
      const overview = DATA.overview || {};
      const metrics = overview.metrics || [];
      const verdict = `<div class="verdict-band ${toneClass(overview.tone)}">
        <div class="metric-label">运行结论</div>
        <div class="metric-value">${esc(overview.verdict || "未知")}</div>
        <div class="metric-sub">${esc(overview.primary_bottleneck || "")}</div>
        <div class="metric-sub mono">${esc(DATA.run_dir || "")}</div>
      </div>`;
      const metricHtml = metrics.map(metric => `<div class="metric">
        <div class="metric-label">${esc(metric.label)}</div>
        <div class="metric-value">${esc(metric.value)}</div>
        <div class="metric-sub">${esc(metric.sub || "")}</div>
      </div>`).join("");
      document.getElementById("overview").innerHTML = verdict + metricHtml;
    }
    function renderStepList() {
      const steps = DATA.steps || [];
      const html = steps.map((step, index) => {
        if (!stepVisible(step)) return "";
        const color = stageColors[step.stage] || "#6b7280";
        return `<button class="step-btn ${index === selected ? "active" : ""}" style="--stage:${color}" data-step="${index}">
          <div class="step-top">
            <span class="step-name">#${esc(step.step_index)} ${esc(short(step.label || step.stage, 54))}</span>
            <span class="pill ${toneClass(step.status)}">${esc(step.status_label || step.stage)}</span>
          </div>
          <div class="step-impact">${esc(short(step.impact || "", 150))}</div>
        </button>`;
      }).join("");
      document.getElementById("stepList").innerHTML = html || `<div class="empty">没有匹配的步骤。</div>`;
      document.querySelectorAll("[data-step]").forEach(btn => {
        btn.addEventListener("click", () => {
          selected = Number(btn.getAttribute("data-step") || 0);
          renderAll();
        });
      });
    }
    function renderFocus() {
      const step = currentStep();
      document.getElementById("snapshotName").textContent = step.snapshot_file || "摘要";
      document.getElementById("stepKicker").textContent = [`第 ${step.step_index || ""} 步`, step.stage_label || step.stage, step.round_index ? `第 ${step.round_index} 轮` : ""].filter(Boolean).join(" / ");
      document.getElementById("stepHeadline").textContent = step.headline || step.label || "";
      document.getElementById("stepImpact").textContent = step.impact || "";
      const plannerMeta = step.stage === "action_batch" || step.stage === "final";
      const meta = [
        step.action_id ? `动作：${step.action_id}` : "",
        plannerMeta && (step.last_planner || {}).mode ? `计划器：${(step.last_planner || {}).mode}` : "",
        plannerMeta && (step.last_planner || {}).status ? `状态：${(step.last_planner || {}).status}` : "",
      ].filter(Boolean);
      document.getElementById("stepMeta").innerHTML = meta.map(item => `<span class="tag">${esc(item)}</span>`).join("");
      renderChanges(step);
      renderBudget(step);
      renderBottlenecks(step);
      document.getElementById("debugJson").textContent = JSON.stringify(step.debug_summary || {}, null, 2);
    }
    function renderChanges(step) {
      const changes = step.top_changes || [];
      document.getElementById("changes").innerHTML = changes.length ? changes.map(change => {
        const sign = Number(change.delta) > 0 ? "+" : "";
        return `<div class="change">
          <strong>${sign}${esc(change.delta)}</strong>
          <span>${esc(change.label)}${change.value ? ` / total ${esc(change.value)}` : ""}</span>
        </div>`;
      }).join("") : `<div class="empty">没有关键计数变化。</div>`;
    }
    function renderBudget(step) {
      const bars = step.budget_bars || [];
      document.getElementById("budgetBars").innerHTML = bars.map(bar => `<div class="bar-row">
        <div class="bar-name">${esc(bar.label)}</div>
        <div class="bar-track"><div class="bar-fill ${bar.tone === "warn" ? "warn" : ""}" style="--pct:${Number(bar.pct || 0)}%"></div></div>
        <div class="bar-value">${esc(bar.used)}/${esc(bar.max || "?")}</div>
      </div>`).join("");
    }
    function renderBottlenecks(step) {
      const items = step.bottlenecks || [];
      document.getElementById("bottlenecks").innerHTML = items.length ? items.map(item => `<div class="notice ${toneClass(item.tone)}">
        <div class="notice-label">${esc(item.label)}</div>
        <div class="notice-text">${esc(item.text)}</div>
      </div>`).join("") : `<div class="empty">这一步没有明显卡点。</div>`;
    }
    function renderBlackboardSummary() {
      const step = currentStep();
      const counts = step.counts || {};
      const chemistry = step.chemistry || {};
      const route = (chemistry.route || {});
      const pdfSignal = Number(counts.pdf_structure_evidence || 0) + Number(counts.visual_chains || 0);
      const problemCount = Number(counts.route_failures || 0) + Number(counts.blocked_directions || 0);
      const cards = [
        {
          label: "路线",
          value: (route.steps || []).length ? `${route.steps.length} 步` : "未成图",
          sub: (route.steps || []).length ? (route.strict_stock_solve ? "终端库存通过" : "有待审计") : "等待 ChemEnzy 或文献路线",
          tone: (route.steps || []).length ? "good" : "warn",
        },
        {
          label: "文献",
          value: Number(counts.source_candidates || 0),
          sub: `${Number(counts.source_refs || 0)} 个来源引用`,
          tone: Number(counts.source_candidates || 0) ? "good" : "warn",
        },
        {
          label: "PDF / 图文",
          value: pdfSignal,
          sub: `${Number(counts.exact_rows || 0)} 条精确步骤`,
          tone: pdfSignal ? "good" : "warn",
        },
        {
          label: "结论",
          value: step.status_label || "",
          sub: problemCount ? `${problemCount} 个问题` : "无显式阻塞",
          tone: step.status || "neutral",
        },
      ];
      document.getElementById("blackboardSummary").innerHTML = cards.map(card => `<div class="summary-card ${toneClass(card.tone)}">
        <div class="section-kicker">${esc(card.label)}</div>
        <div class="summary-value">${esc(card.value)}</div>
        <div class="metric-sub">${esc(card.sub)}</div>
      </div>`).join("");
    }
    function renderChemistry() {
      const step = currentStep();
      const chemistry = step.chemistry || {};
      const target = chemistry.target || {};
      const route = chemistry.route || {};
      const literature = chemistry.literature || {};
      const targetProfile = chemistry.target_profile || {};
      const targetHtml = target.smiles ? `<section class="chemistry-card">
        <div class="chemistry-head">
          <h3 class="panel-title">目标分子</h3>
          <span class="pill">${esc(targetProfile.heavy_atoms || target.heavy_atoms || "?")} 个重原子</span>
        </div>
        <div class="target-body">
          ${renderMolecule(target, {large: true})}
          <div class="target-meta">
            <div class="formula">${esc(target.formula || "分子式未知")}</div>
            <div class="meta-line">
              ${target.mol_weight ? `<span class="tag">分子量 ${esc(target.mol_weight)}</span>` : ""}
              ${targetProfile.rings !== undefined && targetProfile.rings !== null ? `<span class="tag">${esc(targetProfile.rings)} 个环</span>` : ""}
              ${target.valid ? `<span class="tag stock-chip">RDKit 可解析</span>` : `<span class="tag">仅 SMILES</span>`}
            </div>
            <div class="mol-smiles mono">${esc(target.smiles || "")}</div>
            ${targetProfile.family_hint ? `<div class="item-note">${esc(targetProfile.family_hint)}</div>` : ""}
          </div>
        </div>
      </section>` : `<div class="empty">黑板里还没有目标分子结构。</div>`;
      const literatureHtml = renderLiteratureRoute(literature);
      const routeHtml = (route.steps || []).length ? renderRoute(route) : `<section class="chemistry-card">
        <div class="chemistry-head">
          <h3 class="panel-title">最佳路线</h3>
          <span class="pill">未生成</span>
        </div>
        <div class="empty">这一步还没有可展示的 ChemEnzy 路线。路线生成后会在这里显示每一步产物、前体和库存状态。</div>
      </section>`;
      document.getElementById("chemistry").innerHTML = targetHtml + literatureHtml + routeHtml;
    }
    function renderLiteratureRoute(literature) {
      const steps = literature.steps || [];
      const summary = [
        `${Number(literature.visual_chain_count || 0)} 条视觉链`,
        `${Number(literature.exact_row_count || 0)} 条精确步骤`,
      ];
      const sourceHtml = (literature.source_refs || []).length ? `<div class="route-summary">
        <span class="section-kicker">来源</span>
        ${(literature.source_refs || []).map(ref => `<span class="tag">${esc(short(ref, 44))}</span>`).join("")}
      </div>` : "";
      const body = steps.length ? `<div class="route-steps">${steps.map(renderLiteratureStep).join("")}</div>` :
        `<div class="empty">黑板里还没有能画成分子路线的文献/视觉步骤。只有来源、PDF 或过程证据时，会先显示在下方过程细节里。</div>`;
      return `<section class="chemistry-card">
        <div class="chemistry-head">
          <h3 class="panel-title">文献 / 视觉路线片段</h3>
          <span class="pill">${steps.length ? `${steps.length} 个片段` : "待提取"}</span>
        </div>
        <div class="route-summary">${summary.map(item => `<span class="tag">${esc(item)}</span>`).join("")}</div>
        ${sourceHtml}
        ${body}
      </section>`;
    }
    function renderLiteratureStep(step) {
      const reactants = (step.reactants || []).length ? (step.reactants || []).map(mol => renderMolecule(mol, {compact: true})).join("") :
        `<div class="empty">未解析出前体 SMILES</div>`;
      return `<div class="route-step">
        <div class="route-step-head">
          <span>${esc(step.label || "文献步骤")} / ${esc(step.origin || "")}</span>
          <span>${esc(short(step.source_ref || "", 48))}</span>
        </div>
        <div class="route-equation">
          ${renderMolecule(step.product || {}, {compact: true})}
          <div class="route-arrow">&lArr;</div>
          <div class="route-reactants">${reactants}</div>
        </div>
        ${step.condition ? `<div class="item-note">${esc(step.condition)}</div>` : ""}
      </div>`;
    }
    function renderRoute(route) {
      const summary = [
        route.n_steps ? `${route.n_steps} 步` : "",
        route.rank !== undefined && route.rank !== null ? `rank ${route.rank}` : "",
        route.strict_stock_solve ? "终端原料均在库存" : "",
        (route.verifier || {}).target_match ? "目标匹配" : "",
      ].filter(Boolean);
      const terminals = (route.terminal_reactants || []).length ? `<div class="route-summary">
        <span class="section-kicker">终端原料</span>
        ${(route.terminal_reactants || []).map(mol => `<span class="tag ${mol.stock ? "stock-chip" : ""}">${esc(mol.formula || short(mol.smiles, 24))}</span>`).join("")}
      </div>` : "";
      const visibleSteps = (route.steps || []).slice(0, 4);
      const hiddenSteps = (route.steps || []).slice(4);
      const hiddenHtml = hiddenSteps.length ? `<details class="signal-details">
        <summary>展开剩余 ${hiddenSteps.length} 步路线</summary>
        <div class="route-steps">${hiddenSteps.map(renderRouteStep).join("")}</div>
      </details>` : "";
      return `<section class="chemistry-card">
        <div class="chemistry-head">
          <h3 class="panel-title">最佳路线图</h3>
          <span class="pill good">${esc(route.status || "route")}</span>
        </div>
        <div class="route-summary">${summary.map(item => `<span class="tag">${esc(item)}</span>`).join("")}</div>
        ${terminals}
        <div class="route-steps">${visibleSteps.map(renderRouteStep).join("")}</div>
        ${hiddenHtml}
      </section>`;
    }
    function renderRouteStep(step) {
      const score = step.confidence !== undefined && step.confidence !== null ? `置信度 ${Number(step.confidence).toFixed(3)}` : "";
      return `<div class="route-step">
        <div class="route-step-head">
          <span>Step ${Number(step.index || 0) + 1} / ${esc(step.reaction_type || "reaction")}</span>
          <span>${esc(score)}</span>
        </div>
        <div class="route-equation">
          ${renderMolecule(step.product || {}, {compact: true})}
          <div class="route-arrow">&lArr;</div>
          <div class="route-reactants">${(step.reactants || []).map(mol => renderMolecule(mol, {compact: true})).join("")}</div>
        </div>
      </div>`;
    }
    function renderMolecule(mol, options={}) {
      const valid = mol && mol.smiles;
      if (!valid) return `<div class="mol-tile"><div class="mol-svg"><div class="mol-placeholder">无结构</div></div></div>`;
      const svg = mol.svg || `<div class="mol-placeholder">${esc(short(mol.smiles, 54))}</div>`;
      const stock = mol.stock ? `<span class="tag stock-chip">stock</span>` : "";
      const caption = [mol.label, mol.formula].filter(Boolean).join(" / ");
      return `<div class="mol-tile">
        <div class="mol-svg">${svg}</div>
        <div class="mol-caption">
          <span class="mol-name">${esc(caption || "分子")}</span>
          ${stock}
        </div>
        <div class="mol-smiles mono">${esc(short(mol.smiles || "", options.compact ? 58 : 96))}</div>
      </div>`;
    }
    function renderSignals() {
      const step = currentStep();
      document.getElementById("signalScope").textContent = `第 ${step.step_index || ""} 步快照`;
      document.getElementById("signals").innerHTML = (step.sections || []).map(section => {
        const sectionClass = String(section.id || "").replace(/[^a-z0-9_-]/gi, "");
        const rows = (section.items || []).length ? section.items.map(renderSignalItem).join("") : `<div class="empty">${esc(section.empty || "No items.")}</div>`;
        return `<div class="signal-section ${esc(sectionClass)}">
          <div class="signal-head">
            <div class="section-kicker">${esc(section.title)}</div>
            <span class="pill">${(section.items || []).length}</span>
          </div>
          ${rows}
        </div>`;
      }).join("");
    }
    function renderSignalItem(item) {
      const meta = (item.meta || []).map(bit => `<span class="tag">${esc(bit)}</span>`).join("");
      const badges = (item.badges || []).map(bit => `<span class="tag">${esc(bit)}</span>`).join("");
      return `<div class="item-row ${toneClass(item.tone)}">
        <div class="item-title">${esc(item.title)}</div>
        ${item.note ? `<div class="item-note">${esc(item.note)}</div>` : ""}
        ${meta ? `<div class="meta-line">${meta}</div>` : ""}
        ${badges ? `<div class="badge-line">${badges}</div>` : ""}
      </div>`;
    }
    function renderAll() {
      renderShell();
      renderStepList();
      renderFocus();
      renderChemistry();
      renderBlackboardSummary();
      renderSignals();
    }
    renderAll();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", title).replace("__DATA__", data)


if __name__ == "__main__":
    main()
