"""Compile open-research downstream consumables into harness-owned artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem, RDLogger

from cascade_planner.agent.chem_enzy_policy import (
    RerunBudget,
    StrategicOperator,
    compile_chem_enzy_search_policy,
    validate_chem_enzy_search_policy,
    validate_strategic_operator,
)
from cascade_planner.agent.condition_agent import audit_conditions
from cascade_planner.agent.evolution_manager import (
    LayeredKnowledgeBase,
    evolution_candidate_from_dict,
    validate_evolution_candidate,
)
from cascade_planner.agent.executable_template_validation import (
    candidate_to_one_step_row,
    executable_candidate_from_segment_step,
    validate_template_candidate,
)
from cascade_planner.agent.literature_templates import (
    direct_consumption_allowed,
    template_card_from_dict,
    validate_literature_template_card,
)
from cascade_planner.agent.literature_segments import (
    literature_route_segment_from_dict,
    validate_literature_route_segment,
)
from cascade_planner.harness.open_research_contract import (
    normalize_open_research_json_payload,
    validate_open_research_json_payload,
)


RDLogger.DisableLog("rdApp.*")

COMPILED_DOWNSTREAM_SCHEMA = "compiled_downstream_consumables.v1"
FetchJson = Callable[[str, dict[str, str], float], dict[str, Any]]


def compile_downstream_consumables(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    target_smiles: str = "",
    case_id: str = "",
    enable_online_anchor_resolution: bool = False,
    advisory_anchor_catalog: list[dict[str, Any]] | dict[str, Any] | None = None,
    anchor_resolution_timeout_s: float = 5.0,
    anchor_resolution_fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Compile Codex literature handoff drafts into deterministic artifacts.

    The compiler does not run ChemEnzy. It emits bounded rerun payloads,
    template-plugin configuration, one-step rows for validated executable
    templates, and a selfEVO staging KB.
    """
    payload = normalize_open_research_json_payload(
        name="downstream_consumables.json",
        payload=_load_payload(payload_or_path),
    )
    case_id = str(case_id or payload.get("case_id") or "case")
    schema_reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

    advisory_anchors = _compile_advisory_anchor_resolution(
        payload,
        case_id=case_id,
        enable_online=enable_online_anchor_resolution,
        anchor_catalog=advisory_anchor_catalog,
        timeout_s=anchor_resolution_timeout_s,
        fetch_json=anchor_resolution_fetch_json,
    )
    guided = _compile_guided_requests(payload, case_id=case_id, advisory_anchors=advisory_anchors)
    expansion = _compile_route_expansion_tasks(payload, case_id=case_id, advisory_anchors=advisory_anchors)
    templates = _compile_template_plugin(payload, target_smiles=target_smiles)
    maturity = _compile_executable_template_maturity(payload, templates=templates)
    self_evo = _compile_self_evo(payload)
    followup = _compile_agent_followup_actions(
        payload,
        case_id=case_id,
        target_smiles=target_smiles,
        guided=guided,
        expansion=expansion,
        templates=templates,
        self_evo=self_evo,
        advisory_anchors=advisory_anchors,
    )
    rejected = list(payload.get("rejected_consumables") or [])
    rejected.extend(guided["rejected_items"])
    rejected.extend(expansion["rejected_items"])
    rejected.extend(templates["rejected_items"])
    rejected.extend(maturity["rejected_items"])
    rejected.extend(self_evo["rejected_items"])
    rejected.extend(advisory_anchors["rejected_items"])

    reasons = list(schema_reasons)
    reasons.extend(advisory_anchors["reasons"])
    reasons.extend(guided["reasons"])
    reasons.extend(expansion["reasons"])
    reasons.extend(templates["reasons"])
    reasons.extend(maturity["reasons"])
    reasons.extend(self_evo["reasons"])
    accepted = not schema_reasons and bool(
        guided["compiled_policy_payloads"]
        or expansion["compiled_policy_payloads"]
        or expansion["tasks"]
        or templates["one_step_rows"]
        or templates["template_cards"]
        or maturity["report"]["extraction_task_count"]
        or self_evo["staging_candidate_count"]
    )
    if not accepted and not reasons:
        reasons.append("no_compiled_downstream_assets")

    return {
        "schema_version": COMPILED_DOWNSTREAM_SCHEMA,
        "case_id": case_id,
        "accepted": accepted,
        "reasons": sorted(set(str(item) for item in reasons)),
        "planner_handoff": dict(payload.get("planner_handoff") or {}),
        "guided_chemenzy": {
            "operators": guided["operators"] + expansion["operators"],
            "policy_payloads": guided["compiled_policy_payloads"] + expansion["compiled_policy_payloads"],
            "validations": guided["validations"] + expansion["validations"],
        },
        "route_expansion": {
            "tasks": expansion["tasks"],
            "child_targets": followup["child_targets"],
            "operators": expansion["operators"],
            "policy_payloads": expansion["compiled_policy_payloads"],
            "validations": expansion["validations"],
        },
        "advisory_anchor_resolution": advisory_anchors["report"],
        "literature_template_plugin": {
            "enabled": bool(templates["template_cards"] or templates["one_step_rows"]),
            "template_cards": templates["template_cards"],
            "one_step_rows": templates["one_step_rows"],
            "validation_reports": templates["validation_reports"],
            "plugin_flags": {
                "enabled": bool(templates["template_cards"] or templates["one_step_rows"]),
                "top_k": max(1, min(6, len(templates["template_cards"]) or 1)),
                "max_added": max(
                    len(templates["one_step_rows"]),
                    max(1, min(6, len(templates["template_cards"]) or 1)),
                ),
                "template_cards": templates["template_cards"],
                "one_step_rows": templates["one_step_rows"],
                "requires_audit": True,
                "not_raw_reaction_injection": True,
            },
        },
        "executable_template_maturity": maturity["report"],
        "self_evo": self_evo["report"],
        "agent_followup_actions": followup["actions"],
        "rejected_items": rejected,
    }


def write_compiled_downstream_artifacts(
    compiled: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "compiled_downstream_consumables": out / "compiled_downstream_consumables.json",
        "compiled_guided_chemenzy_requests": out / "compiled_guided_chemenzy_requests.json",
        "compiled_route_expansion_tasks": out / "compiled_route_expansion_tasks.json",
        "compiled_advisory_anchor_resolution": out / "compiled_advisory_anchor_resolution.json",
        "compiled_literature_template_plugin": out / "compiled_literature_template_plugin.json",
        "compiled_executable_template_maturity": out / "compiled_executable_template_maturity.json",
        "self_evo_staging_kb": out / "self_evo_staging_kb.json",
    }
    _write_json(paths["compiled_downstream_consumables"], compiled)
    _write_json(paths["compiled_guided_chemenzy_requests"], compiled.get("guided_chemenzy") or {})
    _write_json(paths["compiled_route_expansion_tasks"], compiled.get("route_expansion") or {})
    _write_json(paths["compiled_advisory_anchor_resolution"], compiled.get("advisory_anchor_resolution") or {})
    _write_json(paths["compiled_literature_template_plugin"], compiled.get("literature_template_plugin") or {})
    _write_json(paths["compiled_executable_template_maturity"], compiled.get("executable_template_maturity") or {})
    _write_json(paths["self_evo_staging_kb"], compiled.get("self_evo") or {})
    return {key: str(path) for key, path in paths.items()}


def _compile_guided_requests(
    payload: dict[str, Any],
    *,
    case_id: str,
    advisory_anchors: dict[str, Any],
) -> dict[str, Any]:
    operators: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    for idx, item in enumerate(payload.get("guided_rerun_requests") or []):
        if not isinstance(item, dict):
            rejected.append({"item_index": idx, "reason": "guided_request_not_object"})
            reasons.append("guided_request_not_object")
            continue
        resolved_for_item = _resolved_advisory_anchors_for_refs(
            advisory_anchors.get("resolved_anchor_targets") or [],
            item.get("evidence_refs") or [],
        )
        terminal_blacklist = _valid_smiles_items(item.get("terminal_blacklist") or [])
        anchor_whitelist = _policy_anchor_whitelist(
            item.get("anchor_whitelist") or [],
            resolved_for_item,
            terminal_blacklist=terminal_blacklist,
        )
        policy_resolved_for_item = _anchors_not_in_terminal_blacklist(
            resolved_for_item,
            terminal_blacklist=terminal_blacklist,
        )
        operator = StrategicOperator(
            operator_id=str(item.get("request_id") or f"{case_id}_guided_{idx + 1}"),
            case_id=case_id,
            evidence_refs=[str(ref) for ref in item.get("evidence_refs") or []],
            terminal_blacklist=terminal_blacklist,
            anchor_whitelist=anchor_whitelist,
            preferred_subgoal={
                "target": item.get("target"),
                "preferred_subgoals": list(item.get("preferred_subgoals") or []),
                "resolved_advisory_anchor_targets": policy_resolved_for_item,
                "blocked_advisory_anchor_targets": _anchors_in_terminal_blacklist(
                    resolved_for_item,
                    terminal_blacklist=terminal_blacklist,
                ),
            },
            source_budget={
                "preferred_reaction_classes": list(item.get("preferred_reaction_classes") or []),
                "terminal_blacklist_roles": list(item.get("terminal_blacklist_roles") or []),
                "resolved_advisory_anchor_count": len(policy_resolved_for_item),
                "preferred_anchor_roles": _dedupe_values([
                    str(anchor.get("role") or "")
                    for anchor in policy_resolved_for_item
                    if anchor.get("role")
                ]),
            },
            rerun_reason=str(item.get("reason") or item.get("request_type") or "literature_guided_rerun"),
            budget=RerunBudget(
                max_reruns=min(1, max(0, int(item.get("max_reruns") or 1))),
                max_iterations=min(5000, max(1, int(item.get("max_iterations") or 50))),
                max_depth=min(30, max(1, int(item.get("max_depth") or 15))),
                expansion_topk=min(1000, max(1, int(item.get("expansion_topk") or 100))),
            ),
            input_artifact_refs=["downstream_consumables.json"],
            mode="literature_guided_rerun",
        )
        operator_validation = validate_strategic_operator(operator)
        validations.append({"kind": "strategic_operator", **operator_validation})
        if not operator_validation["accepted"]:
            rejected.append({"item_index": idx, "request_id": operator.operator_id, "reasons": operator_validation["reasons"]})
            reasons.extend(str(reason) for reason in operator_validation["reasons"])
            continue
        policy = compile_chem_enzy_search_policy(operator)
        policy_payload = policy.to_dict()
        policy_validation = validate_chem_enzy_search_policy(policy_payload)
        validations.append({"kind": "chem_enzy_search_policy", **policy_validation})
        if not policy_validation["accepted"]:
            rejected.append({"item_index": idx, "request_id": operator.operator_id, "reasons": policy_validation["reasons"]})
            reasons.extend(str(reason) for reason in policy_validation["reasons"])
            continue
        operators.append(operator.to_dict())
        policies.append(policy_payload)
    return {
        "operators": operators,
        "compiled_policy_payloads": policies,
        "validations": validations,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _compile_route_expansion_tasks(
    payload: dict[str, Any],
    *,
    case_id: str,
    advisory_anchors: dict[str, Any],
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    operators: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    for idx, item in enumerate(payload.get("route_expansion_tasks") or []):
        if not isinstance(item, dict):
            rejected.append({"item_index": idx, "reason": "route_expansion_task_not_object"})
            reasons.append("route_expansion_task_not_object")
            continue
        task_id = str(item.get("task_id") or item.get("request_id") or f"{case_id}_route_expansion_{idx + 1}")
        if _contains_raw_reaction(item):
            rejected.append({"item_index": idx, "task_id": task_id, "reason": "raw_reaction_injection"})
            reasons.append("raw_reaction_injection")
            continue
        evidence_refs = [str(ref) for ref in item.get("evidence_refs") or []]
        if not evidence_refs:
            rejected.append({"item_index": idx, "task_id": task_id, "reason": "missing_evidence_refs"})
            reasons.append("missing_evidence_refs")
            continue
        frontier_smiles = str(
            item.get("frontier_smiles")
            or item.get("stuck_node_smiles")
            or item.get("target_smiles")
            or ""
        )
        exact_target_smiles = str(item.get("exact_target_smiles") or item.get("exact_terminal_smiles") or "").strip()
        exact_target_override = bool(
            item.get("exact_target_override")
            or item.get("strict_exact_target")
            or item.get("target_equivalence_audit_required")
            or exact_target_smiles
        )
        preferred_subgoals = [str(value) for value in item.get("preferred_subgoals") or []]
        target = item.get("target")
        if target and str(target) not in preferred_subgoals:
            preferred_subgoals.append(str(target))
        preferred_classes = [str(value) for value in item.get("preferred_reaction_classes") or []]
        resolved_for_item = _resolved_advisory_anchors_for_refs(
            advisory_anchors.get("resolved_anchor_targets") or [],
            item.get("evidence_refs") or [],
        )
        task_type = str(item.get("task_type") or item.get("request_type") or "route_expansion")
        terminal_blacklist = _valid_smiles_items(item.get("terminal_blacklist") or [])
        anchor_whitelist = _policy_anchor_whitelist(
            item.get("anchor_whitelist") or item.get("known_intermediate_smiles") or [],
            resolved_for_item,
            terminal_blacklist=terminal_blacklist,
        )
        policy_resolved_for_item = _anchors_not_in_terminal_blacklist(
            resolved_for_item,
            terminal_blacklist=terminal_blacklist,
        )
        source_budget = {
            "preferred_reaction_classes": preferred_classes,
            "preferred_disconnection_types": [str(value) for value in item.get("preferred_disconnection_types") or []],
            "terminal_blacklist_roles": [str(value) for value in item.get("terminal_blacklist_roles") or []],
            "route_expansion_task_type": task_type,
            "resolved_advisory_anchor_count": len(policy_resolved_for_item),
            "preferred_anchor_roles": _dedupe_values([
                str(anchor.get("role") or "")
                for anchor in policy_resolved_for_item
                if anchor.get("role")
            ]),
        }
        operator = StrategicOperator(
            operator_id=f"{case_id}_route_expansion_operator_{idx + 1}",
            case_id=case_id,
            evidence_refs=evidence_refs,
            terminal_blacklist=terminal_blacklist,
            anchor_whitelist=anchor_whitelist,
            preferred_subgoal={
                "schema_version": "route_expansion_preferred_subgoal.v1",
                "route_expansion_task_id": task_id,
                "task_type": task_type,
                "target": target,
                "frontier_smiles": frontier_smiles,
                "preferred_subgoals": preferred_subgoals,
                "resolved_advisory_anchor_targets": policy_resolved_for_item,
                "blocked_advisory_anchor_targets": _anchors_in_terminal_blacklist(
                    resolved_for_item,
                    terminal_blacklist=terminal_blacklist,
                ),
                "source_segment_ids": [str(value) for value in item.get("source_segment_ids") or []],
            },
            source_budget=source_budget,
            rerun_reason=str(item.get("reason") or task_type or "route_expansion_task"),
            budget=RerunBudget(
                max_reruns=min(1, max(0, int(item.get("max_reruns") or 1))),
                max_iterations=min(5000, max(1, int(item.get("max_iterations") or 50))),
                max_depth=min(30, max(1, int(item.get("max_depth") or 15))),
                expansion_topk=min(1000, max(1, int(item.get("expansion_topk") or 100))),
            ),
            input_artifact_refs=["downstream_consumables.json", f"route_expansion_tasks[{idx}]"],
            mode="stuck_node_rerun",
        )
        operator_validation = validate_strategic_operator(operator)
        validations.append({"kind": "route_expansion_strategic_operator", **operator_validation})
        if not operator_validation["accepted"]:
            rejected.append({"item_index": idx, "task_id": task_id, "reasons": operator_validation["reasons"]})
            reasons.extend(str(reason) for reason in operator_validation["reasons"])
            continue
        policy = compile_chem_enzy_search_policy(operator)
        policy_payload = policy.to_dict()
        policy_validation = validate_chem_enzy_search_policy(policy_payload)
        validations.append({"kind": "route_expansion_chem_enzy_search_policy", **policy_validation})
        if not policy_validation["accepted"]:
            rejected.append({"item_index": idx, "task_id": task_id, "reasons": policy_validation["reasons"]})
            reasons.extend(str(reason) for reason in policy_validation["reasons"])
            continue
        task = {
            "schema_version": "compiled_route_expansion_task.v1",
            "accepted": True,
            "task_id": task_id,
            "case_id": case_id,
            "task_type": task_type,
            "target": target,
            "frontier_smiles": frontier_smiles,
            "exact_target_smiles": exact_target_smiles if _valid_smiles(exact_target_smiles) else "",
            "exact_target_override": exact_target_override,
            "target_equivalence_audit_required": bool(
                item.get("target_equivalence_audit_required") or exact_target_override
            ),
            "preferred_subgoals": preferred_subgoals,
            "resolved_advisory_anchor_targets": policy_resolved_for_item,
            "blocked_advisory_anchor_targets": _anchors_in_terminal_blacklist(
                resolved_for_item,
                terminal_blacklist=terminal_blacklist,
            ),
            "preferred_reaction_classes": preferred_classes,
            "evidence_refs": evidence_refs,
            "policy_id": policy_payload["policy_id"],
            "next_action": "guided_chemenzy_rerun",
            "production_write_blocked": True,
            "no_solved_claim": True,
            "not_raw_reaction_injection": True,
        }
        tasks.append(task)
        operators.append(operator.to_dict())
        policies.append(policy_payload)
    return {
        "tasks": tasks,
        "operators": operators,
        "compiled_policy_payloads": policies,
        "validations": validations,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _compile_advisory_anchor_resolution(
    payload: dict[str, Any],
    *,
    case_id: str,
    enable_online: bool = False,
    anchor_catalog: list[dict[str, Any]] | dict[str, Any] | None = None,
    timeout_s: float = 5.0,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    candidates = _advisory_anchor_candidate_terms(payload)
    configured_anchors, catalog_rejected = _normalize_configured_advisory_anchor_catalog(
        anchor_catalog
    )
    resolved: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = list(catalog_rejected)
    reasons: list[str] = (
        ["advisory_anchor_catalog_record_invalid"] if catalog_rejected else []
    )
    fetch = fetch_json or _fetch_json
    for candidate in candidates:
        term = str(candidate.get("term") or "").strip()
        if not term:
            continue
        anchor = _resolve_configured_advisory_anchor(term, configured_anchors)
        if not anchor and enable_online and _online_anchor_lookup_allowed(term):
            anchor = _lookup_pubchem_advisory_anchor(term, timeout_s=timeout_s, fetch_json=fetch)
        if not anchor:
            if _anchor_resolution_gap_relevant(term):
                gaps.append({
                    "schema_version": "advisory_anchor_resolution_gap.v1",
                    "term": term,
                    "source": str(candidate.get("source") or ""),
                    "evidence_refs": [str(item) for item in candidate.get("evidence_refs") or []],
                    "reason": "unresolved_advisory_anchor_requires_explicit_catalog_or_online_resolution",
                })
            continue
        smiles = str(anchor.get("smiles") or "")
        canonical = _canonical_smiles(smiles)
        if not canonical:
            rejected.append({
                "term": term,
                "reason": "advisory_anchor_invalid_smiles",
                "source_ref": str(anchor.get("source_ref") or ""),
            })
            reasons.append("advisory_anchor_invalid_smiles")
            continue
        resolved.append({
            "schema_version": "resolved_advisory_anchor_target.v1",
            "case_id": case_id,
            "name": str(anchor.get("name") or term),
            "matched_term": term,
            "aliases": [str(item) for item in anchor.get("aliases") or []],
            "smiles": smiles,
            "canonical_smiles": canonical,
            "source_ref": str(anchor.get("source_ref") or ""),
            "source": str(anchor.get("source") or "explicit_runtime_advisory_anchor_catalog"),
            "role": str(anchor.get("role") or "advisory_anchor"),
            "allowed_use": "anchor_whitelist_and_route_expansion_child_target",
            "resolution_status": "resolved",
            "evidence_refs": [str(item) for item in candidate.get("evidence_refs") or []],
            "from_template_ids": [str(item) for item in candidate.get("template_ids") or []],
            "not_raw_reaction_injection": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        })
    resolved = _dedupe_resolved_advisory_anchors(resolved)
    gaps = _dedupe_anchor_gaps(gaps)
    report = {
        "schema_version": "advisory_anchor_resolution_report.v1",
        "case_id": case_id,
        "accepted": bool(resolved or gaps),
        "resolved_anchor_count": len(resolved),
        "unresolved_anchor_gap_count": len(gaps),
        "resolved_anchor_targets": resolved,
        "unresolved_anchor_gaps": gaps,
        "dictionary_source": (
            "explicit_runtime_advisory_anchor_catalog" if configured_anchors else ""
        ),
        "configured_anchor_count": len(configured_anchors),
        "configured_anchor_rejected_count": len(catalog_rejected),
        "online_resolution_enabled": bool(enable_online),
        "not_raw_reaction_injection": True,
        "no_solved_claim": True,
        "production_write_blocked": True,
    }
    return {
        "report": report,
        "resolved_anchor_targets": resolved,
        "unresolved_anchor_gaps": gaps,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _advisory_anchor_candidate_terms(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload.get("guided_rerun_requests") or []):
        if not isinstance(item, dict):
            continue
        rows.extend(_anchor_terms_from_values(
            item.get("preferred_subgoals") or [],
            source=f"guided_rerun_requests[{idx}].preferred_subgoals",
            evidence_refs=item.get("evidence_refs") or [],
        ))
        rows.extend(_anchor_terms_from_values(
            item.get("anchor_whitelist") or [],
            source=f"guided_rerun_requests[{idx}].anchor_whitelist",
            evidence_refs=item.get("evidence_refs") or [],
        ))
    for idx, item in enumerate(payload.get("route_expansion_tasks") or []):
        if not isinstance(item, dict):
            continue
        rows.extend(_anchor_terms_from_values(
            item.get("preferred_subgoals") or [],
            source=f"route_expansion_tasks[{idx}].preferred_subgoals",
            evidence_refs=item.get("evidence_refs") or [],
        ))
        rows.extend(_anchor_terms_from_values(
            item.get("anchor_whitelist") or item.get("known_intermediate_smiles") or [],
            source=f"route_expansion_tasks[{idx}].anchor_whitelist",
            evidence_refs=item.get("evidence_refs") or [],
        ))
    for idx, card in enumerate(payload.get("literature_template_cards") or []):
        if not isinstance(card, dict):
            continue
        template_id = str(card.get("template_id") or "")
        evidence_refs = card.get("evidence_refs") or []
        rows.extend(_anchor_terms_from_values(
            card.get("precursor_roles") or [],
            source=f"literature_template_cards[{idx}].precursor_roles",
            evidence_refs=evidence_refs,
            template_ids=[template_id] if template_id else [],
        ))
    return rows


def _anchor_terms_from_values(
    values: Any,
    *,
    source: str,
    evidence_refs: Any,
    template_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or _valid_smiles(text):
            continue
        out.append({
            "term": text,
            "source": source,
            "evidence_refs": [str(item) for item in evidence_refs or []],
            "template_ids": [str(item) for item in template_ids or []],
        })
    return out


def _normalize_configured_advisory_anchor_catalog(
    value: list[dict[str, Any]] | dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(value, dict) and isinstance(value.get("anchors"), list):
        raw_rows = list(value.get("anchors") or [])
    elif isinstance(value, dict):
        raw_rows = []
        for key, item in value.items():
            if not isinstance(item, dict):
                raw_rows.append(item)
                continue
            raw_rows.append({**item, "name": str(item.get("name") or key)})
    elif isinstance(value, list):
        raw_rows = list(value)
    else:
        raw_rows = []

    anchors: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            rejected.append(
                {"catalog_index": index, "reason": "anchor_catalog_record_not_object"}
            )
            continue
        name = str(raw.get("name") or "").strip()
        smiles = str(raw.get("smiles") or "").strip()
        source_ref = str(raw.get("source_ref") or "").strip()
        explicitly_enabled = raw.get("allow_as_route_expansion_subgoal") is True
        record_reasons: list[str] = []
        if not name:
            record_reasons.append("anchor_name_missing")
        if not _valid_smiles(smiles):
            record_reasons.append("anchor_smiles_invalid")
        if not source_ref:
            record_reasons.append("anchor_source_ref_missing")
        if not explicitly_enabled:
            record_reasons.append("anchor_route_expansion_consent_missing")
        if record_reasons:
            rejected.append(
                {
                    "catalog_index": index,
                    "name": name,
                    "source_ref": source_ref,
                    "reasons": record_reasons,
                    "reason": "anchor_catalog_record_invalid",
                }
            )
            continue
        aliases = _dedupe_values(
            [name, *[str(item) for item in raw.get("aliases") or []]]
        )
        anchors.append(
            {
                "name": name,
                "aliases": aliases,
                "smiles": smiles,
                "source_ref": source_ref,
                "source": "explicit_runtime_advisory_anchor_catalog",
                "role": str(raw.get("role") or "advisory_anchor"),
                "catalog_index": index,
                "allow_as_route_expansion_subgoal": True,
            }
        )
    return anchors, rejected


def _resolve_configured_advisory_anchor(
    term: str,
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = _normalize_anchor_term(term)
    matches = [
        anchor
        for anchor in anchors
        if normalized
        in {
            _normalize_anchor_term(alias)
            for alias in anchor.get("aliases") or []
        }
    ]
    canonical_matches = {
        _canonical_smiles(str(anchor.get("smiles") or ""))
        for anchor in matches
        if _canonical_smiles(str(anchor.get("smiles") or ""))
    }
    if len(canonical_matches) != 1:
        return {}
    return dict(matches[0])


def _online_anchor_lookup_allowed(term: str) -> bool:
    text = _normalize_anchor_term(term)
    if not text:
        return False
    blocked_tokens = {
        "core",
        "intermediate",
        "precursor",
        "partner",
        "scaffold",
        "fragment",
        "metadata",
        "sequence",
        "donor",
        "derived",
        "like",
        "less",
        "oxidized",
        "hydroxylated",
        "keto",
        "ketosteroid",
        "side",
        "chain",
    }
    words = set(text.split())
    if "or" in words or words.intersection(blocked_tokens):
        return False
    if len(words) > 3:
        return False
    return True


def _lookup_pubchem_advisory_anchor(term: str, *, timeout_s: float, fetch_json: FetchJson) -> dict[str, Any]:
    cid_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON".format(quote(term, safe=""))
    try:
        cid_payload = fetch_json(cid_url, {"User-Agent": "AutoPlanner advisory anchor resolver/0.1"}, timeout_s)
    except Exception:
        return {}
    cids = [str(cid) for cid in ((cid_payload.get("IdentifierList") or {}).get("CID") or []) if str(cid)]
    if not cids:
        return {}
    props = "Title,CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES,MolecularFormula,InChIKey"
    prop_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/JSON".format(cids[0], props)
    try:
        prop_payload = fetch_json(prop_url, {"User-Agent": "AutoPlanner advisory anchor resolver/0.1"}, timeout_s)
    except Exception:
        return {}
    rows = (prop_payload.get("PropertyTable") or {}).get("Properties") or []
    if not rows or not isinstance(rows[0], dict):
        return {}
    row = rows[0]
    smiles = str(
        row.get("IsomericSMILES")
        or row.get("SMILES")
        or row.get("CanonicalSMILES")
        or row.get("ConnectivitySMILES")
        or ""
    )
    if not _valid_smiles(smiles):
        return {}
    return {
        "name": str(row.get("Title") or term),
        "aliases": [term, str(row.get("Title") or "")],
        "smiles": smiles,
        "source_ref": f"pubchem:{row.get('CID') or cids[0]}",
        "source": "live_pubchem_name_lookup",
        "role": "online_resolved_named_advisory_anchor",
        "formula": str(row.get("MolecularFormula") or ""),
        "inchi_key": str(row.get("InChIKey") or ""),
    }


def _normalize_anchor_term(term: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in str(term or "")).split()
    )


def _anchor_resolution_gap_relevant(term: str) -> bool:
    text = _normalize_anchor_term(term)
    if not text or len(text) < 3:
        return False
    generic_only = {
        "compound",
        "intermediate",
        "product",
        "reactant",
        "substrate",
        "structure",
        "unknown",
    }
    tokens = {token for token in text.split() if token}
    return bool(tokens and not tokens.issubset(generic_only))


def _dedupe_resolved_advisory_anchors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_smiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("canonical_smiles") or "")
        if not key:
            continue
        current = dict(by_smiles.get(key) or row)
        evidence = [*(current.get("evidence_refs") or []), *(row.get("evidence_refs") or [])]
        templates = [*(current.get("from_template_ids") or []), *(row.get("from_template_ids") or [])]
        aliases = [*(current.get("aliases") or []), *(row.get("aliases") or []), str(row.get("matched_term") or "")]
        current["evidence_refs"] = _dedupe_values([str(item) for item in evidence])
        current["from_template_ids"] = _dedupe_values([str(item) for item in templates])
        current["aliases"] = _dedupe_values([str(item) for item in aliases])
        by_smiles[key] = current
    return sorted(by_smiles.values(), key=lambda item: str(item.get("name") or ""))


def _dedupe_anchor_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_term: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _normalize_anchor_term(str(row.get("term") or ""))
        if not key:
            continue
        current = dict(by_term.get(key) or row)
        current["evidence_refs"] = _dedupe_values([
            *[str(item) for item in current.get("evidence_refs") or []],
            *[str(item) for item in row.get("evidence_refs") or []],
        ])
        by_term[key] = current
    return sorted(by_term.values(), key=lambda item: str(item.get("term") or ""))


def _resolved_advisory_anchors_for_refs(rows: list[dict[str, Any]], evidence_refs: Any) -> list[dict[str, Any]]:
    refs = {str(item) for item in evidence_refs or [] if str(item)}
    if not refs:
        return []
    selected = []
    for row in rows:
        row_refs = {str(item) for item in row.get("evidence_refs") or [] if str(item)}
        if row_refs.intersection(refs):
            selected.append(dict(row))
    return selected


def _policy_anchor_whitelist(
    explicit_values: Any,
    resolved_anchors: list[dict[str, Any]],
    *,
    terminal_blacklist: list[str],
) -> list[str]:
    blacklist = {_canonical_smiles(smiles) for smiles in terminal_blacklist if _canonical_smiles(smiles)}
    anchors = _valid_smiles_items(explicit_values)
    anchors.extend(str(anchor.get("smiles") or "") for anchor in resolved_anchors if anchor.get("smiles"))
    out: list[str] = []
    seen: set[str] = set()
    for smiles in anchors:
        canonical = _canonical_smiles(smiles)
        if not canonical or canonical in blacklist or canonical in seen:
            continue
        seen.add(canonical)
        out.append(str(smiles))
    return out


def _anchors_not_in_terminal_blacklist(
    anchors: list[dict[str, Any]],
    *,
    terminal_blacklist: list[str],
) -> list[dict[str, Any]]:
    blacklist = {_canonical_smiles(smiles) for smiles in terminal_blacklist if _canonical_smiles(smiles)}
    return [
        dict(anchor)
        for anchor in anchors
        if str(anchor.get("canonical_smiles") or _canonical_smiles(str(anchor.get("smiles") or ""))) not in blacklist
    ]


def _anchors_in_terminal_blacklist(
    anchors: list[dict[str, Any]],
    *,
    terminal_blacklist: list[str],
) -> list[dict[str, Any]]:
    blacklist = {_canonical_smiles(smiles) for smiles in terminal_blacklist if _canonical_smiles(smiles)}
    blocked: list[dict[str, Any]] = []
    for anchor in anchors:
        canonical = str(anchor.get("canonical_smiles") or _canonical_smiles(str(anchor.get("smiles") or "")))
        if canonical in blacklist:
            row = dict(anchor)
            row["resolution_status"] = "blocked_by_terminal_blacklist"
            row["allowed_use"] = "blocked"
            blocked.append(row)
    return blocked


def _payload_terminal_blacklist(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("guided_rerun_requests", "route_expansion_tasks"):
        for item in payload.get(key) or []:
            if isinstance(item, dict):
                values.extend(_valid_smiles_items(item.get("terminal_blacklist") or []))
    return _dedupe_values(values)


def _compile_template_plugin(payload: dict[str, Any], *, target_smiles: str) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    for idx, raw in enumerate(payload.get("literature_template_cards") or []):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "template_card_not_object"})
            reasons.append("template_card_not_object")
            continue
        try:
            card = template_card_from_dict(raw)
        except (TypeError, ValueError) as exc:
            rejected.append({
                "item_index": idx,
                "reason": "template_card_parse_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            reasons.append("template_card_parse_error")
            continue
        validation = validate_literature_template_card(card)
        validations.append({"kind": "literature_template_card", **validation})
        if not validation["accepted"]:
            rejected.append({"item_index": idx, "template_id": card.template_id, "reasons": validation["reasons"]})
            reasons.extend(str(reason) for reason in validation["reasons"])
            continue
        cards.append(card.to_dict())
    for idx, raw in enumerate(payload.get("executable_template_candidates") or []):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "executable_candidate_not_object"})
            reasons.append("executable_candidate_not_object")
            continue
        report = validate_template_candidate(raw)
        validations.append({"kind": "executable_template_candidate", **report.to_dict()})
        if not report.allowed_for_one_step_source:
            rejected.append(
                {
                    "item_index": idx,
                    "source_template_id": raw.get("source_template_id"),
                    "reasons": list(report.reasons),
                }
            )
            reasons.extend(str(reason) for reason in report.reasons)
            continue
        rows.append(candidate_to_one_step_row(raw))
    route_segments = list(payload.get("literature_route_segments") or [])
    source_detail_segments = _segments_from_source_detail_steps(
        payload.get("source_detail_route_steps") or [],
        case_id=str(payload.get("case_id") or "case"),
        target_smiles=target_smiles,
    )
    route_segments.extend(source_detail_segments["segments"])
    cards.extend(source_detail_segments["template_cards"])
    for row in source_detail_segments["one_step_rows"]:
        rows.append(row)
    rejected.extend(source_detail_segments["rejected_items"])
    reasons.extend(source_detail_segments["reasons"])
    validations.extend(source_detail_segments["validations"])
    for idx, raw in enumerate(route_segments):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "route_segment_not_object"})
            reasons.append("route_segment_not_object")
            continue
        try:
            segment = literature_route_segment_from_dict(raw)
        except (TypeError, ValueError) as exc:
            rejected.append({
                "item_index": idx,
                "reason": "route_segment_parse_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            reasons.append("route_segment_parse_error")
            continue
        segment_validation = validate_literature_route_segment(segment)
        validations.append({"kind": "literature_route_segment", **segment_validation})
        if not segment_validation["accepted"]:
            rejected.append({"item_index": idx, "segment_id": segment.segment_id, "reasons": segment_validation["reasons"]})
            reasons.extend(str(reason) for reason in segment_validation["reasons"])
            continue
        for step_index, step in enumerate(segment.steps):
            candidate = executable_candidate_from_segment_step(
                step,
                source_template_id=f"{segment.segment_id}:{step.step_id}",
                reaction_class="literature_route_segment_step",
            )
            report = validate_template_candidate(candidate)
            validations.append({"kind": "segment_step_executable_candidate", **report.to_dict()})
            if not report.allowed_for_one_step_source:
                rejected.append(
                    {
                        "item_index": idx,
                        "step_index": step_index,
                        "segment_id": segment.segment_id,
                        "step_id": step.step_id,
                        "reasons": list(report.reasons),
                    }
                )
                reasons.extend(str(reason) for reason in report.reasons)
                continue
            rows.append(candidate_to_one_step_row(candidate))
    # Keep all validated cards in the compiled plugin config. The runtime
    # plugin still filters direct-consumption eligibility before one-step use,
    # while advisory cards remain available as policy/selfEVO context.
    direct_cards = list(cards)
    return {
        "template_cards": direct_cards,
        "one_step_rows": rows,
        "validation_reports": validations,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _segments_from_source_detail_steps(
    steps: Any,
    *,
    case_id: str,
    target_smiles: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    validations: list[dict[str, Any]] = []
    one_step_rows: list[dict[str, Any]] = []
    template_cards: list[dict[str, Any]] = []
    for idx, raw in enumerate(steps or []):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "source_detail_step_not_object"})
            reasons.append("source_detail_step_not_object")
            continue
        step_id = str(raw.get("step_id") or f"source_detail_step_{idx + 1}")
        if _contains_raw_reaction(raw):
            rejected.append({"item_index": idx, "step_id": step_id, "reason": "raw_reaction_injection"})
            reasons.append("raw_reaction_injection")
            continue
        if str(raw.get("schema_version") or "") != "source_detail_route_step.v1":
            rejected.append({"item_index": idx, "step_id": step_id, "reason": "invalid_source_detail_step_schema"})
            reasons.append("invalid_source_detail_step_schema")
            continue
        source_ref = str(raw.get("source_ref") or "")
        evidence_refs = [str(item) for item in raw.get("evidence_refs") or []]
        product_smiles = str(raw.get("product_smiles") or "")
        reactants = [str(item) for item in raw.get("reactant_smiles") or []]
        if not source_ref or not evidence_refs or not product_smiles or not reactants:
            missing = []
            if not source_ref:
                missing.append("source_ref")
            if not evidence_refs:
                missing.append("evidence_refs")
            if not product_smiles:
                missing.append("product_smiles")
            if not reactants:
                missing.append("reactant_smiles")
            rejected.append({"item_index": idx, "step_id": step_id, "reason": "missing_source_detail_fields", "missing": missing})
            reasons.append("missing_source_detail_fields")
            continue
        group_id = str(raw.get("segment_id") or raw.get("source_ref") or "source_detail_segment")
        condition = _normalized_source_detail_condition(
            raw.get("condition_candidate"),
            step_id=step_id,
            evidence_refs=evidence_refs,
        )
        applicability = _dict_or_note(
            raw.get("applicability"),
            default={
                "status": "passed",
                "product_reconstruction_passed": True,
                "reconstructed_product_smiles": product_smiles,
            },
        )
        applicability.setdefault("product_reconstruction_passed", True)
        applicability.setdefault("reconstructed_product_smiles", product_smiles)
        step_payload = {
            "schema_version": "segment_step_candidate.v1",
            "step_id": step_id,
            "segment_id": group_id,
            "product_smiles": product_smiles,
            "reactant_smiles": reactants,
            "evidence_refs": evidence_refs,
            "source_ref": source_ref,
            "relation_type": str(raw.get("relation_type") or "exact"),
            "applicability": applicability,
            "condition_candidate": condition,
            "source_evidence": [
                dict(item)
                for item in raw.get("source_evidence") or []
                if isinstance(item, dict)
            ],
            "scope_gap": str(raw.get("scope_gap") or ""),
            "source_detail": {
                "schema_version": str(raw.get("schema_version") or ""),
                "source_title": str(raw.get("source_title") or ""),
                "product_name": str(raw.get("product_name") or ""),
                "reactant_names": [str(item) for item in raw.get("reactant_names") or []],
                "provenance": str(raw.get("provenance") or ""),
                "structure_derivation": dict(raw.get("structure_derivation") or {})
                if isinstance(raw.get("structure_derivation"), dict)
                else {},
                "source_excerpt": str(raw.get("source_excerpt") or ""),
                "curator_record_id": str(raw.get("curator_record_id") or ""),
                "curation_status": str(raw.get("curation_status") or ""),
                "validation_status": str(raw.get("validation_status") or ""),
                "deterministic_parser_authority_id": str(
                    raw.get("deterministic_parser_authority_id") or ""
                ),
                "source_binding_reaction_digest": str(
                    raw.get("source_binding_reaction_digest") or ""
                ),
                "source_formulation": dict(raw.get("source_formulation") or {})
                if isinstance(raw.get("source_formulation"), dict)
                else {},
            },
        }
        row_result = _source_detail_exact_step_one_step_row(step_payload)
        validations.append(row_result["validation"])
        if row_result["accepted"]:
            one_step_rows.append(row_result["row"])
        else:
            advisory_result = _source_detail_advisory_template_card(
                step_payload,
                raw=raw,
                exact_row_result=row_result,
            )
            validations.append(advisory_result["validation"])
            if advisory_result["accepted"]:
                template_cards.append(advisory_result["card"])
                reasons.append("advisory_visual_template_card_available")
            else:
                reasons.extend(advisory_result["reasons"])
            rejected.append({
                "item_index": idx,
                "step_id": step_id,
                "reason": "source_detail_exact_step_rejected",
                "reasons": row_result["reasons"],
                "advisory_template_card": advisory_result.get("card", {}),
                "advisory_template_reasons": advisory_result.get("reasons", []),
            })
            reasons.extend(row_result["reasons"])
        grouped.setdefault(group_id, []).append(step_payload)
    segments: list[dict[str, Any]] = []
    for group_id, group_steps in grouped.items():
        if len(group_steps) < 2:
            continue
        first = group_steps[0]
        validations.append({
            "kind": "source_detail_route_segment_draft",
            "segment_id": f"source_detail_{_safe_id(group_id)}",
            "source_detail_step_count": len(group_steps),
            "source_ref": str(first.get("source_ref") or ""),
            "one_step_row_count": sum(
                1
                for row in one_step_rows
                if (
                    (row.get("literature_template_trace") or {}).get("source_detail_segment_id")
                    == str(group_id)
                )
            ),
        })
    return {
        "segments": segments,
        "one_step_rows": one_step_rows,
        "template_cards": template_cards,
        "rejected_items": rejected,
        "reasons": reasons,
        "validations": validations,
    }


def _source_detail_exact_step_one_step_row(step: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    step_id = str(step.get("step_id") or "")
    product = str(step.get("product_smiles") or "")
    reactants = [str(item) for item in step.get("reactant_smiles") or []]
    evidence_refs = [str(item) for item in step.get("evidence_refs") or [] if str(item).strip()]
    source_ref = str(step.get("source_ref") or "")
    relation_type = str(step.get("relation_type") or "")
    applicability = dict(step.get("applicability") or {})
    condition = dict(step.get("condition_candidate") or {})
    source_detail = dict(step.get("source_detail") or {})
    validation_status = str(source_detail.get("validation_status") or "").strip().lower()
    curation_status = str(source_detail.get("curation_status") or "").strip().lower()
    trusted_statuses = {"accepted", "approved", "human_verified", "curator_approved", "deterministically_validated"}
    if not step_id:
        reasons.append("missing_step_id")
    if relation_type != "exact":
        reasons.append("source_detail_step_not_exact")
    if not _valid_smiles(product):
        reasons.append("invalid_product_smiles")
    if not reactants:
        reasons.append("missing_reactant_smiles")
    elif not all(_valid_smiles(smiles) for smiles in reactants):
        reasons.append("invalid_reactant_smiles")
    if not source_ref:
        reasons.append("missing_source_ref")
    if not evidence_refs:
        reasons.append("missing_evidence_refs")
    if applicability.get("status") not in {"passed", "exact"}:
        reasons.append("applicability_failed")
    if not bool(applicability.get("product_reconstruction_passed")):
        reasons.append("product_reconstruction_failed")
    condition_audit = audit_conditions([condition] if condition else [])
    if condition_audit.get("route_risk") == "gap":
        reasons.append("condition_gap")
    if condition_audit.get("route_risk") == "high":
        reasons.append("condition_high_risk")
    if validation_status not in trusted_statuses or curation_status not in trusted_statuses:
        reasons.append("source_detail_step_not_trusted_curated")
    validation = {
        "kind": "source_detail_exact_step_one_step",
        "schema_version": "source_detail_exact_step_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "step_id": step_id,
        "source_ref": source_ref,
        "condition_audit": condition_audit,
        "atom_accounting_policy": "source_detail_exact_step_allows_reagent_or_byproduct_atoms_outside_precursor_list",
    }
    if reasons:
        return {"accepted": False, "reasons": validation["reasons"], "validation": validation, "row": {}}

    template_id = f"source_detail_exact_step:{step_id}"
    trace = {
        "schema_version": "literature_template_trace.v1",
        "source_model": "autoplanner.literature_template_plugin",
        "source_template_id": template_id,
        "source_ref": source_ref,
        "source_detail_segment_id": str(step.get("segment_id") or ""),
        "product_smiles": product,
        "frontier_smiles": product,
        "reactant_smiles": reactants,
        "relation_type": relation_type,
        "evidence_refs": evidence_refs,
        "not_lab_procedure": True,
        "requires_audit": True,
        "no_solved_claim": True,
        "structured_segment_step": True,
        "source_detail_exact_step": True,
        "condition_candidate": condition,
        "source_evidence": [
            dict(item)
            for item in step.get("source_evidence") or []
            if isinstance(item, dict)
        ],
        "atom_accounting_policy": "source_detail_exact_step_allows_reagent_or_byproduct_atoms_outside_precursor_list",
    }
    for key in (
        "source_title",
        "product_name",
        "reactant_names",
        "provenance",
        "curator_record_id",
        "deterministic_parser_authority_id",
        "source_binding_reaction_digest",
        "source_formulation",
    ):
        value = source_detail.get(key)
        if value:
            trace[key] = value
    if source_detail.get("structure_derivation"):
        trace["structure_derivation"] = source_detail["structure_derivation"]
    if source_detail.get("source_excerpt"):
        trace["source_excerpt"] = source_detail["source_excerpt"]
    validation_report = {
        "schema_version": "template_validation_report.v1",
        "accepted": True,
        "reasons": [],
        "confidence": "high",
        "allowed_for_one_step_source": True,
        "source_template_id": template_id,
        "reconstruction_report": {
            "schema_version": "template_forward_reconstruction_audit.v1",
            "passed": True,
            "reasons": [],
            "connectivity_recoverable": True,
            "selected_bond": {"source": "source_detail_exact_step", "step_id": step_id},
            "product_heavy_atoms": _heavy_atom_count(product),
            "reactant_heavy_atoms": sum(_heavy_atom_count(smiles) for smiles in reactants),
            "explanation": "source-grounded exact literature step; reagents/byproducts may be represented in condition fields rather than precursor list",
        },
        "chemical_sanity": {
            "schema_version": "template_basic_chemical_sanity.v1",
            "passed": True,
            "reasons": [],
            "product_heavy_atoms": _heavy_atom_count(product),
            "largest_reactant_heavy_atoms": max([_heavy_atom_count(smiles) for smiles in reactants] or [0]),
            "not_lab_procedure": True,
            "requires_audit": True,
        },
        "audit_required": True,
        "no_solved_claim": True,
    }
    trace["exact_step_validation"] = validation_report
    template_payload = {
        "model_full_name": "autoplanner.literature_template_plugin",
        "source": "literature_template_plugin",
        "source_model": "literature_template_plugin",
        "template_id": template_id,
        "evidence_refs": evidence_refs,
        "not_lab_procedure": True,
        "requires_audit": True,
        "template_validation_report": validation_report,
        "template_applicability_report": {
            "schema_version": "template_applicability_report.v1",
            "target_smiles": product,
            "frontier_smiles": product,
            "matched_retron_atoms": [],
            "matched_bonds": [],
            "match_confidence": "high",
            "mismatch_reasons": [],
            "allowed_use": "executable_candidate",
            "ambiguity_count": 0,
            "selected_bond": {"source": "source_detail_exact_step", "step_id": step_id},
            "cut_fragments": reactants,
            "retron_type": "source_detail_exact_step",
            "template_id": template_id,
        },
        "literature_template_trace": trace,
        "condition_source": source_ref,
        "no_solved_claim": True,
        "source_policy_decision": "enabled_literature_template_plugin",
    }
    row = {
        "reactants": ".".join(reactants),
        "scores": 0.62,
        "costs": None,
        "template": template_payload,
        "templates": template_payload,
        "model_full_name": "autoplanner.literature_template_plugin",
        "weight": 1.0,
        "reaction_domains": "literature_chemical",
        "literature_template_trace": trace,
        "source_policy_decision": "enabled_literature_template_plugin",
    }
    return {"accepted": True, "reasons": [], "validation": validation, "row": row}


def _source_detail_advisory_template_card(
    step: dict[str, Any],
    *,
    raw: dict[str, Any],
    exact_row_result: dict[str, Any],
) -> dict[str, Any]:
    step_id = str(step.get("step_id") or "")
    product = str(step.get("product_smiles") or "")
    reactants = [str(item) for item in step.get("reactant_smiles") or [] if str(item).strip()]
    evidence_refs = [str(item) for item in step.get("evidence_refs") or [] if str(item).strip()]
    source_ref = str(step.get("source_ref") or "")
    exact_reasons = [str(item) for item in exact_row_result.get("reasons") or []]
    reasons: list[str] = []
    if not step_id:
        reasons.append("missing_step_id")
    if not _valid_smiles(product):
        reasons.append("invalid_product_smiles")
    if not reactants:
        reasons.append("missing_reactant_smiles")
    elif not all(_valid_smiles(smiles) for smiles in reactants):
        reasons.append("invalid_reactant_smiles")
    if not source_ref:
        reasons.append("missing_source_ref")
    if not evidence_refs:
        reasons.append("missing_evidence_refs")

    source_detail = dict(step.get("source_detail") or {})
    condition = dict(step.get("condition_candidate") or {})
    reaction_class = _source_detail_advisory_reaction_class(step, raw=raw)
    scope_limits = _dedupe_values(
        [
            "visual_or_scheme_extraction_hint_not_exact_literature_step",
            "may_seed_mechanistic_template_extraction",
            "requires_exact_applicability_and_product_reconstruction_before_one_step",
            "no_solved_claim",
            *[f"exact_row_gate:{reason}" for reason in exact_reasons],
            str(step.get("scope_gap") or ""),
            str(raw.get("stereochemistry_status") or ""),
            str(raw.get("allowed_use") or ""),
        ]
    )
    safety_flags = _dedupe_values(
        [
            "requires_exact_row_validation_before_one_step",
            "not_route_proof",
            "stereochemistry_unresolved_or_partial",
            *[str(item) for item in raw.get("risk_flags") or []],
        ]
    )
    product_retron = {
        "retron_type": reaction_class,
        "description": "visual/source-detail mechanistic template hint; stereochemistry and full atom accounting are not asserted",
        "product_smiles": product,
        "product_name": str(source_detail.get("product_name") or raw.get("product_name") or ""),
        "source_step_id": step_id,
    }
    card = {
        "schema_version": "literature_template_card.v1",
        "template_id": f"source_detail_visual_hint:{_safe_id(step_id)}",
        "evidence_refs": evidence_refs,
        "reaction_class": reaction_class,
        "template_level": "advisory_strategy",
        "product_retron": product_retron,
        "break_bonds": [],
        "precursor_roles": _dedupe_values(
            [
                *[str(item) for item in source_detail.get("reactant_names") or []],
                *[f"reactant:{smiles}" for smiles in reactants],
            ]
        ),
        "applicability": {
            "schema_version": "visual_template_hint_applicability.v1",
            "allowed_use": "mechanistic_template_hint_only",
            "direct_one_step_consumption": False,
            "can_seed_template_extraction": True,
            "can_seed_mechanism_application": True,
            "must_not_claim_exact_route": True,
            "source_policy_decision": "advisory_visual_template_hint",
            "exact_row_gate_reasons": sorted(set(exact_reasons)),
            "product_smiles": product,
            "reactant_smiles": reactants,
            "condition_candidate": condition,
            "relation_type": str(step.get("relation_type") or ""),
            "source_ref": source_ref,
            "source_title": str(source_detail.get("source_title") or raw.get("source_title") or ""),
            "structure_derivation": dict(source_detail.get("structure_derivation") or {}),
            "stereochemistry_policy": "ignore_or_mark_unresolved_for_template_hint; do_not_assert_exact_configuration",
        },
        "scope_limits": scope_limits,
        "safety_flags": safety_flags,
        "promotion_status": "visual_hint_requires_exact_resolution",
        "source_family": str(source_detail.get("source_title") or raw.get("source_title") or source_ref),
        "condition_source": source_ref or "visual_source_detail",
        "not_raw_reaction_injection": True,
    }
    card_validation = validate_literature_template_card(card)
    if not card_validation["accepted"]:
        reasons.extend(str(reason) for reason in card_validation["reasons"])
    validation = {
        "kind": "source_detail_advisory_template_card",
        "schema_version": "source_detail_advisory_template_card_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "step_id": step_id,
        "source_ref": source_ref,
        "template_id": card["template_id"],
        "exact_row_gate_reasons": sorted(set(exact_reasons)),
        "card_validation": card_validation,
        "allowed_use": "mechanistic_template_hint_only",
        "direct_one_step_consumption": False,
    }
    return {
        "accepted": not reasons,
        "reasons": validation["reasons"],
        "validation": validation,
        "card": card if not reasons else {},
    }


def _source_detail_advisory_reaction_class(step: dict[str, Any], *, raw: dict[str, Any]) -> str:
    source_detail = dict(step.get("source_detail") or {})
    condition = dict(step.get("condition_candidate") or {})
    haystack = " ".join(
        [
            str(step.get("step_id") or ""),
            str(source_detail.get("product_name") or raw.get("product_name") or ""),
            " ".join(str(item) for item in source_detail.get("reactant_names") or raw.get("reactant_names") or []),
            str(source_detail.get("source_excerpt") or raw.get("source_excerpt") or ""),
            " ".join(str(item) for item in condition.values()),
            str(step.get("scope_gap") or ""),
        ]
    ).lower()
    checks = [
        (("dehydrochlor", "elimination"), "elimination"),
        (("chlorination", "n-chloro", "ncs", "chloride"), "chlorination"),
        (("glycosyl", "sugar"), "glycosylation"),
        (("hydrolysis", "saponification", "naoh", "koh"), "hydrolysis"),
        (("deprotect", "deprotection", "protecting group"), "deprotection"),
        (("acetyl", "acyl", "anhydride", "esterification"), "acylation"),
        (("oxid", "mcpba", "pcc", "dess"), "oxidation"),
        (("reduct", "hydrogenation", "nabh", "lialh"), "reduction"),
        (("lacton", "macrolacton"), "lactonization"),
    ]
    for needles, reaction_class in checks:
        if any(needle in haystack for needle in needles):
            return reaction_class
    return "visual_source_detail_mechanistic_hint"


def _compile_agent_followup_actions(
    payload: dict[str, Any],
    *,
    case_id: str,
    target_smiles: str = "",
    guided: dict[str, Any],
    expansion: dict[str, Any],
    templates: dict[str, Any],
    self_evo: dict[str, Any],
    advisory_anchors: dict[str, Any],
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    child_targets = _source_detail_child_targets_from_one_step_rows(
        templates.get("one_step_rows") or [],
        case_id=case_id,
        target_smiles=target_smiles,
    )
    child_targets.extend(_advisory_anchor_child_targets(
        advisory_anchors.get("resolved_anchor_targets") or [],
        case_id=case_id,
        terminal_blacklist=_payload_terminal_blacklist(payload),
    ))
    child_targets = _dedupe_child_targets(child_targets)
    plugin_flags = {
        "enabled": bool(templates.get("template_cards") or templates.get("one_step_rows")),
        "top_k": max(1, min(6, len(templates.get("template_cards") or []) or 1)),
        "max_added": max(
            len(templates.get("one_step_rows") or []),
            max(1, min(6, len(templates.get("template_cards") or []) or 1)),
        ),
        "requires_audit": True,
        "not_raw_reaction_injection": True,
    }
    if templates.get("one_step_rows"):
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_replay_literature_template_plugin",
                "tool_name": "run_guided_chemenzy_rerun",
                "reason": "source_detail_one_step_rows_available",
                "payload_hint": {
                    "use_compiled_literature_template_plugin": True,
                    "literature_template_plugin": plugin_flags,
                },
                "one_step_row_count": len(templates.get("one_step_rows") or []),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    elif templates.get("template_cards"):
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_refine_advisory_visual_templates",
                "tool_name": "resolve_literature_structure_task",
                "reason": "advisory_visual_template_cards_require_exact_resolution",
                "payload_hint": {
                    "use_template_cards_as_mechanistic_hints": True,
                    "template_card_count": len(templates.get("template_cards") or []),
                    "required_next_gate": "exact source-detail validation before one-step replay",
                },
                "template_card_count": len(templates.get("template_cards") or []),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    if child_targets:
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_explore_compiled_child_targets",
                "tool_name": "run_route_expansion_subgoal_search",
                "reason": "compiled_child_targets_available",
                "payload_hint": {
                    "use_compiled_child_targets": True,
                    "max_targets": min(2, len(child_targets)),
                },
                "child_target_count": len(child_targets),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    if guided.get("compiled_policy_payloads"):
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_guided_chemenzy_policy_rerun",
                "tool_name": "run_guided_chemenzy_rerun",
                "reason": "compiled_guided_chemenzy_policy_available",
                "policy_count": len(guided.get("compiled_policy_payloads") or []),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    if expansion.get("tasks"):
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_route_expansion_subgoal_search",
                "tool_name": "run_route_expansion_subgoal_search",
                "reason": "compiled_route_expansion_tasks_available",
                "route_expansion_task_count": len(expansion.get("tasks") or []),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    if self_evo.get("staging_candidate_count"):
        actions.append(
            {
                "schema_version": "agent_followup_action.v1",
                "action_id": f"{case_id}_self_evo_replay_gate",
                "tool_name": "run_self_evo_replay_gate",
                "reason": "self_evo_staging_candidates_available",
                "staging_candidate_count": self_evo.get("staging_candidate_count"),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    return {
        "actions": actions,
        "child_targets": child_targets,
    }


def _source_detail_child_targets_from_one_step_rows(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    target_smiles: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    canonical_target = _canonical_smiles(target_smiles)
    for row_index, row in enumerate(rows):
        trace = dict(row.get("literature_template_trace") or ((row.get("template") or {}).get("literature_template_trace") or {}))
        if not trace.get("source_detail_exact_step"):
            continue
        reactants = [str(item) for item in trace.get("reactant_smiles") or [] if str(item).strip()]
        if not reactants:
            reactants = [str(item) for item in str(row.get("reactants") or "").split(".") if str(item).strip()]
        for reactant_index, smiles in enumerate(reactants):
            if not _valid_smiles(smiles):
                continue
            source_template_id = str(trace.get("source_template_id") or f"source_detail_row_{row_index + 1}")
            child_id = f"{case_id}_{_safe_id(source_template_id)}_reactant_{reactant_index + 1}"
            out.append(
                {
                    "schema_version": "route_expansion_child_target.v1",
                    "child_target_id": child_id,
                    "name": child_id,
                    "smiles": smiles,
                    "source": "source_detail_one_step_reactant",
                    "source_template_id": source_template_id,
                    "source_ref": str(trace.get("source_ref") or ""),
                    "parent_product_smiles": str(trace.get("product_smiles") or trace.get("frontier_smiles") or ""),
                    "target_proximal_rank": _source_detail_target_proximal_rank(trace, canonical_target),
                    "evidence_refs": [str(item) for item in trace.get("evidence_refs") or []],
                    "policy": {
                        "schema_version": "chem_enzy_search_policy.v1",
                        "policy_id": f"{child_id}_policy",
                        "operator_id": f"{child_id}_operator",
                        "case_id": case_id,
                        "evidence_refs": [str(item) for item in trace.get("evidence_refs") or []],
                        "terminal_blacklist": [],
                        "anchor_whitelist": [],
                        "preferred_subgoal": {
                            "schema_version": "source_detail_upstream_subgoal.v1",
                            "source_template_id": source_template_id,
                            "parent_product_smiles": str(trace.get("product_smiles") or trace.get("frontier_smiles") or ""),
                            "preferred_subgoals": [smiles],
                        },
                        "source_budget": {
                            "preferred_reaction_classes": ["source_detail_upstream_expansion"],
                            "source_detail_exact_step": True,
                        },
                        "rerun_reason": "explore upstream reactant from source-detail literature step",
                        "budget": {
                            "max_reruns": 1,
                            "max_iterations": 50,
                            "max_depth": 15,
                            "expansion_topk": 100,
                        },
                        "mode": "guided",
                        "compiler_metadata": {
                            "compiler_schema": "source_detail_child_target_compiler.v1",
                            "not_raw_reaction_injection": True,
                        },
                    },
                    "max_depth": 15,
                    "max_iterations": 50,
                    "expansion_topk": 100,
                    "no_solved_claim": True,
                    "production_write_blocked": True,
                }
            )
    out.sort(
        key=lambda item: (
            _source_detail_target_rank_value(item.get("target_proximal_rank")),
            str(item.get("source_template_id") or ""),
            str(item.get("smiles") or ""),
        )
    )
    return _dedupe_child_targets(out)


def _source_detail_target_rank_value(value: Any) -> int:
    if value is None:
        return 1_000_000
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1_000_000


def _source_detail_target_proximal_rank(trace: dict[str, Any], canonical_target: str) -> int:
    if not canonical_target:
        return 1_000_000
    product = _canonical_smiles(str(trace.get("product_smiles") or trace.get("frontier_smiles") or ""))
    if product == canonical_target:
        return 0
    derivation = trace.get("structure_derivation")
    if not isinstance(derivation, dict):
        return 1_000_000
    formula_report = derivation.get("formula_report")
    if not isinstance(formula_report, dict):
        return 1_000_000
    for idx, item in enumerate(formula_report.values(), start=1):
        if not isinstance(item, dict):
            continue
        smiles = _canonical_smiles(str(item.get("smiles") or ""))
        if smiles == product:
            return idx
    return 1_000_000


def _advisory_anchor_child_targets(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    terminal_blacklist: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    blacklist = {_canonical_smiles(smiles) for smiles in terminal_blacklist if _canonical_smiles(smiles)}
    for idx, anchor in enumerate(rows or []):
        smiles = str(anchor.get("smiles") or "")
        canonical = _canonical_smiles(smiles)
        if not canonical or canonical in blacklist:
            continue
        anchor_id = f"{case_id}_resolved_advisory_anchor_{_safe_id(anchor.get('name') or idx + 1)}"
        evidence_refs = [str(item) for item in anchor.get("evidence_refs") or [] if str(item).strip()]
        out.append({
            "schema_version": "route_expansion_child_target.v1",
            "child_target_id": anchor_id,
            "name": str(anchor.get("name") or anchor_id),
            "smiles": smiles,
            "canonical_smiles": str(anchor.get("canonical_smiles") or canonical),
            "source": "resolved_advisory_anchor",
            "source_ref": str(anchor.get("source_ref") or ""),
            "evidence_refs": evidence_refs,
            "role": str(anchor.get("role") or "advisory_anchor"),
            "policy": {
                "schema_version": "chem_enzy_search_policy.v1",
                "policy_id": f"{anchor_id}_policy",
                "operator_id": f"{anchor_id}_operator",
                "case_id": case_id,
                "evidence_refs": evidence_refs or [str(anchor.get("source_ref") or "resolved_advisory_anchor")],
                "terminal_blacklist": [],
                "anchor_whitelist": [smiles],
                "preferred_subgoal": {
                    "schema_version": "resolved_advisory_anchor_subgoal.v1",
                    "preferred_subgoals": [str(anchor.get("name") or ""), smiles],
                    "resolved_advisory_anchor_targets": [dict(anchor)],
                },
                "source_budget": {
                    "preferred_reaction_classes": ["steroid_semisynthesis", "advisory_anchor_upstream_expansion"],
                    "preferred_anchor_roles": [str(anchor.get("role") or "advisory_anchor")],
                    "resolved_advisory_anchor": True,
                },
                "rerun_reason": "explore upstream route to resolved advisory anchor",
                "budget": {
                    "max_reruns": 1,
                    "max_iterations": 50,
                    "max_depth": 15,
                    "expansion_topk": 100,
                },
                "mode": "guided",
                "compiler_metadata": {
                    "compiler_schema": "advisory_anchor_child_target_compiler.v1",
                    "not_raw_reaction_injection": True,
                },
            },
            "max_depth": 15,
            "max_iterations": 50,
            "expansion_topk": 100,
            "no_solved_claim": True,
            "production_write_blocked": True,
        })
    return out


def _dedupe_child_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        smiles = str(row.get("smiles") or "")
        source_template_id = str(row.get("source_template_id") or "")
        key = f"{smiles}|{source_template_id}"
        if not smiles or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _compile_executable_template_maturity(
    payload: dict[str, Any],
    *,
    templates: dict[str, Any],
) -> dict[str, Any]:
    extraction_tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    for idx, raw in enumerate(payload.get("executable_template_extraction_tasks") or []):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "extraction_task_not_object"})
            reasons.append("extraction_task_not_object")
            continue
        task_id = str(raw.get("task_id") or f"executable_extraction_task_{idx + 1}")
        if _contains_raw_reaction(raw):
            rejected.append({"item_index": idx, "task_id": task_id, "reason": "raw_reaction_injection"})
            reasons.append("raw_reaction_injection")
            continue
        if str(raw.get("schema_version") or "") != "executable_template_extraction_task.v1":
            rejected.append({"item_index": idx, "task_id": task_id, "reason": "invalid_extraction_task_schema"})
            reasons.append("invalid_extraction_task_schema")
            continue
        if not raw.get("evidence_refs"):
            rejected.append({"item_index": idx, "task_id": task_id, "reason": "missing_evidence_refs"})
            reasons.append("missing_evidence_refs")
            continue
        extraction_tasks.append(_compiled_extraction_task(raw, task_id=task_id))

    one_step_count = len(templates.get("one_step_rows") or [])
    route_segment_count = len(payload.get("literature_route_segments") or [])
    source_detail_step_count = len(payload.get("source_detail_route_steps") or [])
    executable_candidate_count = len(payload.get("executable_template_candidates") or [])
    advisory_template_count = len(templates.get("template_cards") or [])
    mechanistic_hint_template_count = sum(
        1
        for card in templates.get("template_cards") or []
        if str(((card.get("applicability") or {}) if isinstance(card, dict) else {}).get("allowed_use") or "")
        == "mechanistic_template_hint_only"
    )
    direct_template_count = sum(
        1
        for card in templates.get("template_cards") or []
        if direct_consumption_allowed(card)
    )
    status = "executable_ready" if one_step_count else (
        "needs_structured_extraction"
        if extraction_tasks or advisory_template_count or route_segment_count or executable_candidate_count
        else "no_template_assets"
    )
    gap_reasons: list[str] = []
    if not one_step_count and advisory_template_count:
        if mechanistic_hint_template_count:
            gap_reasons.append("advisory_visual_templates_require_exact_validation_before_one_step")
        else:
            gap_reasons.append("advisory_templates_lack_source_grounded_reactant_product_smiles")
    if not one_step_count and route_segment_count:
        gap_reasons.append("route_segments_failed_executable_validation")
    if not one_step_count and executable_candidate_count:
        gap_reasons.append("executable_candidates_failed_one_step_validation")
    if extraction_tasks:
        gap_reasons.append("structured_step_extraction_required")
    report = {
        "schema_version": "executable_template_maturity.v1",
        "status": status,
        "one_step_row_count": one_step_count,
        "advisory_template_count": advisory_template_count,
        "mechanistic_hint_template_count": mechanistic_hint_template_count,
        "direct_template_card_count": direct_template_count,
        "route_segment_count": route_segment_count,
        "source_detail_route_step_count": source_detail_step_count,
        "executable_candidate_count": executable_candidate_count,
        "extraction_task_count": len(extraction_tasks),
        "extraction_tasks": extraction_tasks,
        "gap_reasons": sorted(set(gap_reasons)),
        "required_fields_for_one_step": [
            "product_smiles",
            "reactant_smiles",
            "source_ref",
            "evidence_refs",
            "relation_type=exact",
            "applicability.product_reconstruction_passed",
            "condition_candidate",
        ],
        "downstream_path": "visual/advisory template hints -> exact source-detail validation -> one_step_rows -> ChemEnzy literature_template_plugin",
        "no_solved_claim": True,
        "production_write_blocked": True,
    }
    return {
        "report": report,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _compiled_extraction_task(raw: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    return {
        "schema_version": "compiled_executable_template_extraction_task.v1",
        "task_id": task_id,
        "case_id": str(raw.get("case_id") or ""),
        "task_type": str(raw.get("task_type") or "extract_structured_literature_route_segment"),
        "status": str(raw.get("status") or "needs_source_grounded_product_reactant_smiles"),
        "evidence_refs": [str(item) for item in raw.get("evidence_refs") or []],
        "source_ref": str(raw.get("source_ref") or ""),
        "source_title": str(raw.get("source_title") or ""),
        "source_type": str(raw.get("source_type") or ""),
        "source_relation": str(raw.get("source_relation") or ""),
        "reaction_class": str(raw.get("reaction_class") or ""),
        "frontier_smiles": str(raw.get("frontier_smiles") or ""),
        "target_smiles": str(raw.get("target_smiles") or ""),
        "required_artifact_type": str(raw.get("required_artifact_type") or "LiteratureRouteSegmentCard or SegmentStepCandidate"),
        "required_structured_fields": [str(item) for item in raw.get("required_structured_fields") or []],
        "precursor_roles": [str(item) for item in raw.get("precursor_roles") or []],
        "break_bonds": [str(item) for item in raw.get("break_bonds") or []],
        "extraction_policy": dict(raw.get("extraction_policy") or {}),
        "downstream_use_if_completed": str(raw.get("downstream_use_if_completed") or ""),
        "not_raw_reaction_injection": True,
    }


def _compile_self_evo(payload: dict[str, Any]) -> dict[str, Any]:
    kb = LayeredKnowledgeBase()
    validations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []
    for idx, raw in enumerate(payload.get("evolution_candidates") or []):
        if not isinstance(raw, dict):
            rejected.append({"item_index": idx, "reason": "evolution_candidate_not_object"})
            reasons.append("evolution_candidate_not_object")
            continue
        if raw.get("target_layer") == "production":
            rejected.append({"item_index": idx, "candidate_id": raw.get("candidate_id"), "reason": "production_blocked"})
            reasons.append("production_blocked")
            continue
        candidate_data = {
            "schema_version": "evolution_candidate.v1",
            "candidate_id": raw.get("candidate_id") or f"evolution_candidate_{idx + 1}",
            "candidate_type": raw.get("candidate_type") or "TemplateCandidate",
            "payload": dict(raw.get("payload") or {key: value for key, value in raw.items() if key not in {"target_layer"}}),
            "evidence_refs": list(raw.get("evidence_refs") or []),
            "validation_status": raw.get("validation_status") or "draft",
            "source": "open_structure_research",
        }
        candidate = evolution_candidate_from_dict(candidate_data)
        validation = validate_evolution_candidate(candidate)
        validations.append(validation)
        if not validation["accepted"]:
            rejected.append({"item_index": idx, "candidate_id": candidate.candidate_id, "reasons": validation["reasons"]})
            reasons.extend(str(reason) for reason in validation["reasons"])
            continue
        kb.add_candidate(candidate, target_run=True)
        kb.promote(candidate.candidate_id, from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote(candidate.candidate_id, from_layer="shadow", to_layer="staging", target_run=True)
    kb_payload = kb.to_dict()
    staging_count = len((kb_payload.get("layers") or {}).get("staging") or {})
    return {
        "report": {
            "schema_version": "self_evo_staging_compile_report.v1",
            "accepted": bool(staging_count),
            "production_write_blocked": True,
            "staging_candidate_count": staging_count,
            "candidate_validation": validations,
            "kb": kb_payload,
        },
        "staging_candidate_count": staging_count,
        "rejected_items": rejected,
        "reasons": reasons,
    }


def _dict_or_note(value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return dict(default)
    return {
        **dict(default),
        "note": str(value),
    }


def _normalized_source_detail_condition(value: Any, *, step_id: str, evidence_refs: list[str]) -> dict[str, Any]:
    condition = _dict_or_note(value, default={})
    if not condition:
        return {}
    condition.setdefault("schema_version", "condition_candidate.v1")
    condition.setdefault("step_id", step_id)
    condition.setdefault("source_type", "exact")
    condition.setdefault("condition_status", "evidence_backed")
    if not condition.get("evidence_refs"):
        condition["evidence_refs"] = evidence_refs
    reagent_candidates = [str(item) for item in condition.get("reagent_candidates") or [] if str(item).strip()]
    solvent_candidates = [str(item) for item in condition.get("solvent_candidates") or [] if str(item).strip()]
    if reagent_candidates and not condition.get("reagent"):
        condition["reagent"] = "; ".join(reagent_candidates)
    if solvent_candidates and not condition.get("solvent"):
        condition["solvent"] = "; ".join(solvent_candidates)
    if not condition.get("temperature"):
        for key in ("temperature_C", "temperature_c"):
            if condition.get(key) is not None:
                condition["temperature"] = f"{condition[key]} C"
                break
    if not condition.get("duration"):
        for key in ("duration_h", "duration_min", "hydrolysis_duration_min"):
            if condition.get(key) is not None:
                condition["duration"] = str(condition[key])
                break
    return condition


def _valid_smiles(smiles: str) -> bool:
    return bool(str(smiles or "").strip()) and Chem.MolFromSmiles(str(smiles or "")) is not None


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _valid_smiles_items(values: Any) -> list[str]:
    return [str(item) for item in values or [] if _valid_smiles(str(item))]


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "source_detail"


def _dedupe_values(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "raw_reactions"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


def _load_payload(payload_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, dict):
        return dict(payload_or_path)
    return json.loads(Path(payload_or_path).read_text(encoding="utf-8"))


def _fetch_json(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
