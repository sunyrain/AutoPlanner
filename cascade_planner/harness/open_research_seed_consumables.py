"""Harness-owned downstream seeds from local literature context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_local_downstream_seed(
    *,
    manifest: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build conservative downstream drafts from validated local context.

    These drafts are intentionally advisory. They let guided ChemEnzy and
    selfEVO use validated local disconnection cards and failure feedback even
    if the open Codex agent times out before enriching artifacts.
    """
    out = Path(output_dir)
    target = dict(manifest.get("target") or {})
    case_id = _case_id(target)
    case_manifest = dict(manifest.get("case_manifest") or {})
    evidence_cards = _load_jsonl_path(case_manifest.get("evidence_cards"))
    feedback = dict((manifest.get("prior_experience") or {}).get("route_failure_feedback") or {})
    validated_cards = [
        row for row in evidence_cards
        if isinstance(row, dict)
        and str(row.get("validation_status") or "") in {"validated", "accepted", ""}
        and str(row.get("evidence_id") or "")
    ]
    evidence_refs = _dedupe([str(row.get("evidence_id")) for row in validated_cards if row.get("evidence_id")])
    if not evidence_refs and not feedback:
        return {
            "schema_version": "open_research_local_downstream_seed.v1",
            "accepted": False,
            "status": "not_applicable",
            "reasons": ["no_local_evidence_or_route_failure_feedback"],
            "downstream_consumables": {},
        }

    preferred_subgoals = _dedupe([
        *[
            str(item)
            for item in (feedback.get("next_guided_policy_patch") or {}).get("preferred_subgoals") or []
            if str(item)
        ],
        *[
            str(row.get("canonical_smiles") or row.get("smiles") or "")
            for row in feedback.get("frontier_research_targets") or []
            if isinstance(row, dict) and str(row.get("canonical_smiles") or row.get("smiles") or "")
        ],
        *[
            role
            for card in validated_cards
            for role in _precursor_roles(card)
        ],
    ])[:12]
    terminal_blacklist = _dedupe([
        *[
            str(item)
            for item in (feedback.get("next_guided_policy_patch") or {}).get("terminal_blacklist") or []
            if str(item)
        ],
        *[
            str(row.get("canonical_smiles") or row.get("smiles") or "")
            for row in feedback.get("terminal_blacklist") or []
            if isinstance(row, dict) and str(row.get("canonical_smiles") or row.get("smiles") or "")
        ],
    ])[:20]
    source_budget = dict((feedback.get("next_guided_policy_patch") or {}).get("source_budget") or {})
    if not source_budget:
        source_budget = {
            "active_failure_modes": [str(item) for item in feedback.get("source_reasons") or []],
            "terminal_blacklist_roles": ["hidden_nonstock_advanced_intermediate", "advanced_same_scaffold_terminal"],
        }
    guided_refs = evidence_refs or ["harness_route_failure_feedback"]
    guided_request = {
        "request_id": f"{case_id}_local_seed_guided_1",
        "request_type": "local_literature_seed_guided_chemenzy_rerun",
        "target": "stuck_node_or_route_failure_frontier",
        "frontier_smiles": str(target.get("frontier_smiles") or ""),
        "evidence_refs": guided_refs,
        "preferred_subgoals": preferred_subgoals,
        "preferred_reaction_classes": _reaction_classes(validated_cards),
        "terminal_blacklist": terminal_blacklist,
        "terminal_blacklist_roles": [str(item) for item in source_budget.get("terminal_blacklist_roles") or []],
        "reason": "validated local literature cards and route failure feedback should guide deep ChemEnzy search",
        "max_depth": 20,
        "max_iterations": 50,
        "expansion_topk": 100,
    }
    route_task = {
        "task_id": f"{case_id}_local_seed_route_expansion_1",
        "task_type": "stuck_node_rerun",
        "frontier_smiles": str(target.get("frontier_smiles") or ""),
        "target": "route_failure_frontier_or_advanced_intermediate",
        "preferred_subgoals": preferred_subgoals,
        "preferred_reaction_classes": _reaction_classes(validated_cards),
        "terminal_blacklist": terminal_blacklist,
        "terminal_blacklist_roles": [str(item) for item in source_budget.get("terminal_blacklist_roles") or []],
        "anchor_whitelist": _valid_smiles([str(row.get("canonical_smiles") or row.get("smiles") or "") for row in feedback.get("frontier_research_targets") or [] if isinstance(row, dict)]),
        "evidence_refs": guided_refs,
        "reason": "expand stuck frontier using local literature disconnection priors without treating advanced terminals as stock closure",
        "max_depth": 20,
        "max_iterations": 50,
        "expansion_topk": 100,
    }
    template_cards = [
        template
        for card in validated_cards
        for template in [_template_card_from_evidence(card)]
        if template
    ]
    extraction_tasks = [
        task
        for card in validated_cards
        for task in [_executable_extraction_task_from_evidence(card, target=target, case_id=case_id)]
        if task
    ]
    source_detail_patch = _source_detail_resolution_patch(manifest)
    source_detail_steps = [
        dict(item)
        for item in source_detail_patch.get("source_detail_route_steps") or []
        if isinstance(item, dict)
    ]
    source_detail_rejections = [
        dict(item)
        for item in source_detail_patch.get("rejected_consumables") or []
        if isinstance(item, dict)
    ]
    evolution_candidates = [
        {
            "candidate_id": f"{card['template_id']}_self_evo",
            "candidate_type": "TemplateCandidate",
            "validation_status": "draft",
            "target_layer": "candidate",
            "evidence_refs": list(card.get("evidence_refs") or []),
            "payload": card,
        }
        for card in template_cards
    ]
    downstream = {
        "schema_version": "open_downstream_consumables.v1",
        "case_id": case_id,
        "planner_handoff": {
            "next_action": "guided_chemenzy_rerun",
            "solved": False,
            "production_kb_promotion": False,
            "reason": "harness local evidence seed produced conservative guided rerun and selfEVO candidates",
            "generated_by": "harness_local_downstream_seed",
            "template_maturity": {
                "status": "needs_structured_extraction" if extraction_tasks else "advisory_only",
                "advisory_template_count": len(template_cards),
                "source_detail_route_step_count": len(source_detail_steps),
                "executable_one_step_row_count": 0,
                "extraction_task_count": len(extraction_tasks),
            },
            "next_extraction_tasks": [
                _extraction_task_handoff(task)
                for task in extraction_tasks[:8]
            ],
        },
        "guided_rerun_requests": [guided_request],
        "literature_template_cards": template_cards,
        "literature_route_segments": [],
        "executable_template_candidates": [],
        "executable_template_extraction_tasks": extraction_tasks,
        "source_detail_route_steps": source_detail_steps,
        "route_expansion_tasks": [route_task],
        "evolution_candidates": evolution_candidates,
        "rejected_consumables": source_detail_rejections,
    }
    return {
        "schema_version": "open_research_local_downstream_seed.v1",
        "accepted": True,
        "status": "written",
        "generated_by": "harness_local_downstream_seed",
        "evidence_card_count": len(validated_cards),
        "guided_request_count": 1,
        "template_card_count": len(template_cards),
        "executable_template_extraction_task_count": len(extraction_tasks),
        "source_detail_route_step_count": len(source_detail_steps),
        "source_detail_resolution_gap_count": len(source_detail_rejections),
        "route_expansion_task_count": 1,
        "evolution_candidate_count": len(evolution_candidates),
        "downstream_consumables": downstream,
        "artifact_refs": {
            "downstream_consumables": str((out / "downstream_consumables.json").resolve()),
        },
    }


def _source_detail_resolution_patch(manifest: dict[str, Any]) -> dict[str, Any]:
    entry = dict(manifest.get("source_detail_resolution") or {})
    path_value = str(entry.get("path") or "")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    patch = dict(payload.get("downstream_patch") or {}) if isinstance(payload, dict) else {}
    if not patch:
        return {}
    return {
        "source_detail_route_steps": [
            dict(item)
            for item in patch.get("source_detail_route_steps") or []
            if isinstance(item, dict)
        ],
        "rejected_consumables": [
            {
                **dict(item),
                "source": str(item.get("source") or "source_detail_resolution_pack"),
            }
            for item in patch.get("rejected_consumables") or []
            if isinstance(item, dict)
        ],
    }


def write_local_downstream_seed_artifacts(
    *,
    output_dir: str | Path,
    seed: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    downstream = dict(seed.get("downstream_consumables") or {})
    if not seed.get("accepted") or not downstream:
        return dict(seed)
    path = out / "downstream_consumables.json"
    if overwrite or not path.exists() or _is_seed_only_downstream(path):
        path.write_text(json.dumps(downstream, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    record_path = out / "harness_local_downstream_seed.json"
    record_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    result = dict(seed)
    refs = dict(result.get("artifact_refs") or {})
    refs["harness_local_downstream_seed"] = str(record_path.resolve())
    result["artifact_refs"] = refs
    return result


def _template_card_from_evidence(card: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(card.get("source_metadata") or {})
    record = dict(metadata.get("record") or {})
    move = dict(record.get("retrosynthetic_move") or {})
    evidence_id = str(card.get("evidence_id") or "")
    if not evidence_id or not move:
        return {}
    reaction_class = str(move.get("reaction_class") or card.get("family_id") or "local_literature_guided_disconnection")
    return {
        "schema_version": "literature_template_card.v1",
        "template_id": f"{_safe_id(evidence_id)}_advisory_template",
        "validation_status": "draft",
        "evidence_refs": [evidence_id],
        "reaction_class": reaction_class,
        "template_level": "advisory_strategy",
        "product_retron": {
            "retron_type": str(card.get("route_role") or "strategic_disconnection"),
            "description": str(card.get("source_title") or ""),
        },
        "break_bonds": [
            {"label": str(item), "source": "local_evidence_card"}
            for item in move.get("break_bonds") or []
        ],
        "precursor_roles": [str(item) for item in move.get("suggested_precursor_roles") or []],
        "applicability": {
            "direct_one_step_consumption": False,
            "status": "advisory_only",
            "source_relation": str(card.get("target_relation") or ""),
        },
        "scope_limits": [str(item) for item in card.get("limitations") or []],
        "safety_flags": ["not_raw_reaction_injection", "requires_current_target_audit"],
        "promotion_status": "advisory_only",
        "source_family": str(card.get("family_id") or ""),
        "condition_source": str(card.get("source_type") or "local_curated"),
        "not_raw_reaction_injection": True,
    }


def _executable_extraction_task_from_evidence(
    card: dict[str, Any],
    *,
    target: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    metadata = dict(card.get("source_metadata") or {})
    record = dict(metadata.get("record") or {})
    move = dict(record.get("retrosynthetic_move") or {})
    evidence_id = str(card.get("evidence_id") or "")
    if not evidence_id or not move:
        return {}
    reaction_class = str(move.get("reaction_class") or card.get("family_id") or "local_literature_guided_disconnection")
    return {
        "schema_version": "executable_template_extraction_task.v1",
        "task_id": f"{_safe_id(evidence_id)}_structured_step_extraction",
        "case_id": case_id,
        "task_type": "extract_structured_literature_route_segment",
        "status": "needs_source_grounded_product_reactant_smiles",
        "evidence_refs": [evidence_id],
        "source_ref": _source_ref(card),
        "source_title": str(card.get("source_title") or ""),
        "source_type": str(card.get("source_type") or "local_curated"),
        "source_relation": str(card.get("target_relation") or ""),
        "reaction_class": reaction_class,
        "frontier_smiles": str(target.get("frontier_smiles") or ""),
        "target_smiles": str(target.get("smiles") or ""),
        "required_artifact_type": "LiteratureRouteSegmentCard or SegmentStepCandidate",
        "required_structured_fields": [
            "product_smiles",
            "reactant_smiles",
            "source_ref",
            "evidence_refs",
            "relation_type=exact",
            "applicability.product_reconstruction_passed",
            "condition_candidate",
        ],
        "precursor_roles": [str(item) for item in move.get("suggested_precursor_roles") or []],
        "break_bonds": [str(item) for item in move.get("break_bonds") or []],
        "extraction_policy": {
            "do_not_fabricate_smiles": True,
            "no_raw_reaction_injection": True,
            "require_rdkit_valid_product_and_reactants": True,
            "only_exact_relation_can_compile_to_one_step": True,
            "record_as_gap_if_product_or_reactant_smiles_missing": True,
        },
        "downstream_use_if_completed": "literature_route_segments to one_step_rows to ChemEnzy literature_template_plugin",
        "not_raw_reaction_injection": True,
    }


def _extraction_task_handoff(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "task_type": str(task.get("task_type") or ""),
        "source_title": str(task.get("source_title") or ""),
        "reaction_class": str(task.get("reaction_class") or ""),
        "evidence_refs": [str(item) for item in task.get("evidence_refs") or []],
        "required_structured_fields": [str(item) for item in task.get("required_structured_fields") or []],
    }


def _source_ref(card: dict[str, Any]) -> str:
    for key in ("doi", "url", "local_ref", "source_record_id"):
        value = str(card.get(key) or "").strip()
        if value:
            return value
    return str(card.get("evidence_id") or "")


def _precursor_roles(card: dict[str, Any]) -> list[str]:
    metadata = dict(card.get("source_metadata") or {})
    move = dict((metadata.get("record") or {}).get("retrosynthetic_move") or {})
    return [str(item) for item in move.get("suggested_precursor_roles") or [] if str(item)]


def _reaction_classes(cards: list[dict[str, Any]]) -> list[str]:
    values = [
        str(card.get("family_id") or "")
        for card in cards
        if str(card.get("family_id") or "")
    ]
    values.extend(["statin_side_chain_convergence", "stereocontrolled_syn_diol_construction"])
    return _dedupe(values)[:8]


def _is_seed_only_downstream(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    handoff = dict(payload.get("planner_handoff") or {}) if isinstance(payload, dict) else {}
    return str(handoff.get("generated_by") or "") == "harness_prefetch_checkpoint_seed"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_jsonl_path(value: Any) -> list[dict[str, Any]]:
    path_value = str(value or "").strip()
    if not path_value:
        return []
    return _load_jsonl(Path(path_value))


def _case_id(target: dict[str, Any]) -> str:
    value = str(target.get("name") or "target").strip().lower()
    value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return value or "target"


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "template"


def _valid_smiles(values: list[str]) -> list[str]:
    return [value for value in values if any(ch in value for ch in "=#[]()/\\") or any(ch.isdigit() for ch in value)]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
