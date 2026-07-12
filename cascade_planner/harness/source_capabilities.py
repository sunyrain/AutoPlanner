"""Derived, fail-closed capabilities for the literature evidence lifecycle.

The blackboard remains the only mutable state.  This module projects the work
that is executable from the *current* source/evidence records so the
deterministic planner, Codex prompt, and validator cannot maintain different
notions of pending PDF or visual work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    canonical_traceable_source_refs,
    independent_source_group,
    source_content_scope,
    source_document_identity,
)


SOURCE_CAPABILITY_QUEUE_SCHEMA = "source_capability_queue.v1"
SOURCE_CAPABILITY_SCHEMA = "source_capability.v1"
SOURCE_SENSITIVE_ACTIONS = frozenset(
    {
        "extract_pdf_literature_structures",
        "extract_visual_literature_chain",
        "resolve_literature_structure_task",
        "compile_exact_literature_rows",
    }
)
LITERATURE_ACTIONS = SOURCE_SENSITIVE_ACTIONS | {"search_literature"}


def pdf_evidence_has_materialized_render(row: Mapping[str, Any]) -> bool:
    """Return true only for accepted evidence with a real rendered image."""

    value = dict(row or {})
    summary = dict(value.get("summary") or {})
    try:
        rendered_count = int(
            value.get("rendered_page_count")
            or summary.get("rendered_page_count")
            or 0
        )
    except (TypeError, ValueError):
        return False
    return bool(
        value.get("accepted") is True
        and rendered_count > 0
        and not [
            str(item)
            for item in value.get("reasons") or []
            if str(item or "").strip()
        ]
        and pdf_evidence_render_paths(value)
    )


def pdf_evidence_render_paths(row: Mapping[str, Any]) -> list[str]:
    """Resolve current on-disk render paths, including an artifact wrapper."""

    value = dict(row or {})
    candidates: list[str] = []
    for field in ("rendered_pages", "scheme_crops", "indexed_images"):
        for item in value.get(field) or []:
            if isinstance(item, Mapping):
                candidates.extend(
                    str(item.get(key) or "")
                    for key in ("image_path", "source_image_path", "path")
                )
    candidates.extend(str(item or "") for item in value.get("image_paths") or [])
    artifact_ref = str(value.get("artifact_ref") or "").strip()
    if not candidates and artifact_ref:
        path = Path(artifact_ref).expanduser()
        if path.is_file() and path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = {}
            nested = (
                dict(payload.get("result") or payload)
                if isinstance(payload, Mapping)
                else {}
            )
            if nested and nested != value:
                candidates.extend(pdf_evidence_render_paths(nested))
    return sorted(
        {
            str(Path(path).expanduser().resolve())
            for path in candidates
            if str(path or "").strip() and Path(str(path)).expanduser().is_file()
        }
    )


def action_resource_cost(
    action: Mapping[str, Any] | str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Return the shared deterministic cost projection for one action."""

    if isinstance(action, Mapping):
        action_type = str(action.get("action_type") or "")
        body = dict(action.get("payload") or {})
    else:
        action_type = str(action or "")
        body = dict(payload or {})
    literature_units = 0
    if action_type == "search_literature":
        literature_units = _bounded_int(body.get("max_sources"), default=3, low=1, high=3)
    elif action_type in SOURCE_SENSITIVE_ACTIONS:
        literature_units = 1
    visual_calls = int(
        action_type == "extract_visual_literature_chain"
        or (
            action_type == "resolve_literature_structure_task"
            and bool(body.get("run_visual", True))
        )
    )
    child_runs = 0
    if action_type == "expand_child_target":
        targets = body.get("subgoal_targets") or body.get("child_targets") or []
        child_runs = max(1, len(targets)) if isinstance(targets, list) else 1
    return {
        "action_slots": 1,
        "literature_source_units": literature_units,
        "scout_calls": int(action_type == "search_literature"),
        "visual_calls": visual_calls,
        "chemenzy_runs": int(action_type == "run_guided_chemenzy"),
        "child_target_runs": child_runs,
        "template_application_actions": int(
            action_type
            in {"apply_analogical_template_to_target", "validate_template_application"}
        ),
    }


def build_source_capability_queue(
    blackboard: Mapping[str, Any],
    *,
    round_index: int = 0,
    max_literature_sources_per_round: int = 3,
) -> dict[str, Any]:
    """Build the current executable source-work queue without mutating state."""

    board = dict(blackboard or {})
    evidence = dict(board.get("literature_evidence") or {})
    cap = max(0, int(max_literature_sources_per_round or 0))
    budget = dict(board.get("budget_state") or {})
    documents: dict[str, dict[str, Any]] = {}

    candidate_rows = [
        dict(row)
        for row in evidence.get("source_candidates") or []
        if isinstance(row, Mapping)
    ]
    for row in sorted(candidate_rows, key=_stable_row_key):
        _register_document(documents, row, role="source_candidate")

    pdf_rows = [
        dict(row)
        for row in evidence.get("pdf_structure_evidence") or []
        if isinstance(row, Mapping)
    ]
    for row in sorted(pdf_rows, key=_stable_row_key):
        identity = _resolve_or_register_document(
            documents,
            row,
            role="pdf_structure_evidence",
        )
        if identity and pdf_evidence_has_materialized_render(row):
            documents[identity]["rendered"] = True
            documents[identity]["render_evidence_refs"].extend(
                _evidence_refs(row, extra=pdf_evidence_render_paths(row))
            )

    visual_rows = [
        dict(row)
        for row in evidence.get("visual_chains") or []
        if isinstance(row, Mapping)
    ]
    for row in sorted(visual_rows, key=_stable_row_key):
        identity = _resolve_or_register_document(
            documents,
            row,
            role="visual_chain",
        )
        if identity and _visual_chain_is_materialized(row):
            documents[identity]["visualized"] = True
            documents[identity]["visual_chain_ids"].extend(_visual_chain_ids(row))

    target = dict(board.get("target_profile") or {})
    target_terms = _priority_terms(
        [
            str(target.get("target_name") or ""),
            str(target.get("family_hint") or ""),
            *[str(item) for item in target.get("functional_handles") or []],
        ]
    )

    capabilities: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    visual_remaining = _budget_remaining(budget, "visual_calls", "max_visual_calls")
    scout_remaining = _budget_remaining(budget, "scout_calls", "max_scout_calls")

    for identity, document in sorted(documents.items()):
        source = _merged_source_record(identity, document["rows"])
        local_pdf = str(
            source.get("local_pdf")
            or source.get("source_pdf_path")
            or source.get("pdf_path")
            or ""
        ).strip()
        if not local_pdf:
            continue
        if not document["rendered"]:
            capability = _source_capability(
                action_type="extract_pdf_literature_structures",
                document_identity=identity,
                source=source,
                stage_from="local_pdf_available",
                stage_to="pdf_rendered",
                priority=300 + _source_priority(source, target_terms=target_terms),
                prerequisites=[],
            )
            _append_eligible_or_blocked(
                capabilities,
                blocked,
                capability,
                reason="literature_source_round_budget_exhausted" if cap < 1 else "",
            )
            continue
        if not document["visualized"]:
            capability = _source_capability(
                action_type="extract_visual_literature_chain",
                document_identity=identity,
                source=source,
                stage_from="pdf_rendered",
                stage_to="visual_extracted",
                priority=400 + _source_priority(source, target_terms=target_terms),
                prerequisites=sorted(set(document["render_evidence_refs"])),
            )
            reason = ""
            if cap < 1:
                reason = "literature_source_round_budget_exhausted"
            elif visual_remaining <= 0:
                reason = "visual_total_budget_exhausted"
            _append_eligible_or_blocked(
                capabilities,
                blocked,
                capability,
                reason=reason,
            )

    for task in sorted(_open_structure_tasks(evidence), key=_stable_row_key):
        source = _source_for_task(task, documents)
        capability = _task_capability(task, source=source)
        reason = ""
        if cap < 1:
            reason = "literature_source_round_budget_exhausted"
        elif visual_remaining <= 0 and bool((capability.get("payload_binding") or {}).get("run_visual", True)):
            reason = "visual_total_budget_exhausted"
        _append_eligible_or_blocked(
            capabilities,
            blocked,
            capability,
            reason=reason,
        )

    for chain in sorted(visual_rows, key=_stable_row_key):
        if not _visual_chain_has_uncompiled_steps(chain, evidence.get("exact_rows") or []):
            continue
        capability = _compile_capability(chain)
        _append_eligible_or_blocked(
            capabilities,
            blocked,
            capability,
            reason="literature_source_round_budget_exhausted" if cap < 1 else "",
        )

    search_cost = min(3, cap)
    if search_cost > 0:
        search_capability = _generic_search_capability(search_cost)
        _append_eligible_or_blocked(
            capabilities,
            blocked,
            search_capability,
            reason="scout_total_budget_exhausted" if scout_remaining <= 0 else "",
        )

    capabilities.sort(key=_capability_sort_key)
    blocked.sort(key=_capability_sort_key)
    payload = {
        "schema_version": SOURCE_CAPABILITY_QUEUE_SCHEMA,
        "case_id": str(board.get("case_id") or ""),
        "round_index": max(0, int(round_index or 0)),
        "budget": {
            "literature_source_units_max_this_round": cap,
            "literature_source_units_remaining_this_round": cap,
            "visual_calls_remaining": max(0, visual_remaining),
            "scout_calls_remaining": max(0, scout_remaining),
        },
        "capabilities": capabilities,
        "blocked": blocked,
        "pending_pdf_extraction_sources": _pending_sources(
            capabilities,
            "extract_pdf_literature_structures",
        ),
        "pending_visual_extraction_sources": _pending_sources(
            capabilities,
            "extract_visual_literature_chain",
        ),
        "semantics": {
            "derived_from_current_blackboard": True,
            "queue_is_not_mutable_state": True,
            "only_eligible_capabilities_are_model_selectable": True,
            "costs_are_host_derived": True,
            "no_solved_claim": True,
        },
        "no_solved_claim": True,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def eligible_source_capabilities(
    queue: Mapping[str, Any],
    action_type: str,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in queue.get("capabilities") or []
        if isinstance(row, Mapping)
        and str(row.get("action_type") or "") == str(action_type or "")
        and row.get("eligible") is True
    ]


def matching_source_capabilities(
    queue: Mapping[str, Any],
    *,
    action_type: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Match one action payload to current eligible work, failing ambiguous."""

    body = dict(payload or {})
    candidates = eligible_source_capabilities(queue, action_type)
    capability_id = str(
        body.get("source_capability_id") or body.get("capability_id") or ""
    ).strip()
    if capability_id:
        return [
            row
            for row in candidates
            if str(row.get("capability_id") or "") == capability_id
        ]
    if not _source_aliases(body) and len(candidates) == 1:
        return candidates
    matches: list[dict[str, Any]] = []
    body_aliases = _source_aliases(body)
    body_document = source_document_identity(body)
    task_id = str(body.get("task_id") or "").strip()
    chain_id = str(body.get("chain_id") or body.get("visual_chain_id") or "").strip()
    for row in candidates:
        binding = dict(row.get("payload_binding") or {})
        if task_id and task_id == str(binding.get("task_id") or ""):
            matches.append(row)
            continue
        if chain_id and chain_id in {
            str(binding.get("chain_id") or ""),
            str(binding.get("visual_chain_id") or ""),
        }:
            matches.append(row)
            continue
        if body_document and body_document == str(row.get("document_identity") or ""):
            matches.append(row)
            continue
        row_aliases = _source_aliases(
            {**dict(row.get("source") or {}), **binding}
        )
        if body_aliases and row_aliases and body_aliases & row_aliases:
            matches.append(row)
    unique = {
        str(row.get("capability_id") or ""): row
        for row in matches
        if str(row.get("capability_id") or "")
    }
    return [unique[key] for key in sorted(unique)]


def source_action_is_eligible(
    queue: Mapping[str, Any],
    *,
    action_type: str,
    payload: Mapping[str, Any],
) -> bool:
    return len(
        matching_source_capabilities(
            queue,
            action_type=action_type,
            payload=payload,
        )
    ) == 1


def _register_document(
    documents: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    role: str,
) -> str:
    identity = _document_identity(row)
    if not identity:
        return ""
    document = documents.setdefault(
        identity,
        {
            "rows": [],
            "aliases": set(),
            "rendered": False,
            "visualized": False,
            "render_evidence_refs": [],
            "visual_chain_ids": [],
        },
    )
    tagged = dict(row)
    tagged["_capability_record_role"] = role
    document["rows"].append(tagged)
    document["aliases"].update(_source_aliases(row))
    return identity


def _resolve_or_register_document(
    documents: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    role: str,
) -> str:
    identity = _document_identity(row)
    if identity in documents:
        _register_document(documents, row, role=role)
        return identity
    aliases = _source_aliases(row)
    matches = [
        key
        for key, value in documents.items()
        if aliases and aliases & set(value.get("aliases") or set())
    ]
    if len(matches) == 1:
        tagged = dict(row)
        tagged["_capability_record_role"] = role
        documents[matches[0]]["rows"].append(tagged)
        documents[matches[0]]["aliases"].update(aliases)
        return matches[0]
    if len(documents) == 1 and not _has_document_locator(row):
        only_identity = next(iter(documents))
        tagged = dict(row)
        tagged["_capability_record_role"] = role
        documents[only_identity]["rows"].append(tagged)
        return only_identity
    return _register_document(documents, row, role=role)


def _has_document_locator(row: Mapping[str, Any]) -> bool:
    return any(
        str(row.get(field) or "").strip()
        for field in (
            "source_ref",
            "doi",
            "pii",
            "url",
            "patent",
            "patent_publication",
            "document_id",
            "local_pdf",
            "source_pdf_path",
            "pdf_path",
            "title",
            "source_title",
        )
    )


def _document_identity(row: Mapping[str, Any]) -> str:
    identity = source_document_identity(row)
    if identity:
        return identity
    aliases = sorted(_source_aliases(row))
    if not aliases:
        return ""
    return f"document:fallback:{hashlib.sha256('|'.join(aliases).encode('utf-8')).hexdigest()[:20]}"


def _source_aliases(row: Mapping[str, Any]) -> set[str]:
    value = dict(row or {})
    aliases = set(
        canonical_traceable_source_refs(
            [
                value.get("source_ref"),
                value.get("doi"),
                f"doi:{value.get('doi')}" if str(value.get("doi") or "").strip() else "",
                value.get("pii"),
                f"pii:{value.get('pii')}" if str(value.get("pii") or "").strip() else "",
                value.get("url"),
                value.get("patent"),
                value.get("patent_publication"),
            ]
        )
    )
    for field in ("document_id", "candidate_id", "source_id", "task_id", "chain_id"):
        text = str(value.get(field) or "").strip().casefold()
        if text:
            aliases.add(f"{field}:{text}")
    for field in ("local_pdf", "source_pdf_path", "pdf_path"):
        text = str(value.get(field) or "").strip()
        if text:
            aliases.add(f"path:{os.path.normcase(os.path.abspath(os.path.expanduser(text)))}")
    source_ref = str(value.get("source_ref") or "").strip().casefold()
    if source_ref and not canonical_traceable_source_ref(source_ref):
        aliases.add(f"legacy_ref:{source_ref}")
    title = " ".join(
        str(value.get("title") or value.get("source_title") or "")
        .strip()
        .casefold()
        .split()
    )
    if title:
        aliases.add(f"title:{title}")
    return aliases


def _merged_source_record(
    identity: str,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=_stable_row_key)
    fields = (
        "candidate_id",
        "source_id",
        "document_id",
        "content_scope",
        "source_ref",
        "title",
        "source_title",
        "doi",
        "pii",
        "url",
        "patent",
        "patent_publication",
        "local_pdf",
        "source_pdf_path",
        "pdf_path",
        "source_role",
        "access_status",
        "route_sequence_hint",
        "relevance_rationale",
        "visual_extraction_profile",
    )
    merged: dict[str, Any] = {}
    for field in fields:
        values = [row.get(field) for row in ordered if row.get(field) not in (None, "", [], {})]
        if values:
            merged[field] = values[0]
    labels = sorted(
        {
            str(label)
            for row in ordered
            for label in row.get("expected_scheme_or_compound_labels") or []
            if str(label or "").strip()
        }
    )
    if labels:
        merged["expected_scheme_or_compound_labels"] = labels
    declared_source_refs = sorted(
        {
            str(row.get("source_ref") or "").strip()
            for row in ordered
            if str(row.get("source_ref") or "").strip()
        }
    )
    canonical_declared_refs = canonical_traceable_source_refs(declared_source_refs)
    canonical_refs = canonical_traceable_source_refs(
        [
            *[row.get("source_ref") for row in ordered],
            *[row.get("doi") for row in ordered],
            *[row.get("url") for row in ordered],
            *[row.get("patent") for row in ordered],
            *[row.get("patent_publication") for row in ordered],
        ]
    )
    if canonical_declared_refs:
        merged["source_ref"] = canonical_declared_refs[0]
    elif declared_source_refs:
        # Preserve opaque legacy/scout aliases for display and matching.  They
        # remain non-authoritative and document identity still uses a strict
        # DOI/patent/URL locator when one is available on the same record.
        merged["source_ref"] = declared_source_refs[0]
    elif canonical_refs:
        merged["source_ref"] = canonical_refs[0]
    local_pdf = str(
        merged.get("local_pdf")
        or merged.get("source_pdf_path")
        or merged.get("pdf_path")
        or ""
    ).strip()
    if local_pdf:
        merged["local_pdf"] = local_pdf
    merged["document_identity"] = identity
    merged["independent_source_group"] = independent_source_group(merged)
    merged["content_scope"] = str(merged.get("content_scope") or source_content_scope(merged))
    merged["no_solved_claim"] = True
    return merged


def _source_capability(
    *,
    action_type: str,
    document_identity: str,
    source: dict[str, Any],
    stage_from: str,
    stage_to: str,
    priority: int,
    prerequisites: list[str],
) -> dict[str, Any]:
    payload = _source_payload_binding(source)
    cost = action_resource_cost(action_type, payload)
    identity_payload = {
        "action_type": action_type,
        "document_identity": document_identity,
        "payload_binding": payload,
        "stage_from": stage_from,
        "stage_to": stage_to,
    }
    capability_id = f"source-capability:sha256:{_digest(identity_payload)}"
    return {
        "schema_version": SOURCE_CAPABILITY_SCHEMA,
        "capability_id": capability_id,
        "action_type": action_type,
        "document_identity": document_identity,
        "independent_source_group": str(source.get("independent_source_group") or ""),
        "source_ref": str(source.get("source_ref") or ""),
        "source_title": str(source.get("title") or source.get("source_title") or ""),
        "stage_from": stage_from,
        "stage_to": stage_to,
        "payload_binding": payload,
        "source": source,
        "cost": cost,
        "priority": int(priority),
        "prerequisite_evidence_refs": prerequisites,
        "eligible": True,
        "no_solved_claim": True,
    }


def _source_payload_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(source or {})
    payload: dict[str, Any] = {}
    source_ref = canonical_traceable_source_ref(value.get("source_ref")) or str(
        value.get("source_ref") or ""
    ).strip()
    if source_ref:
        payload["source_ref"] = source_ref
    title = str(value.get("title") or value.get("source_title") or "").strip()
    if title:
        payload["source_title"] = title
    local_pdf = str(
        value.get("local_pdf")
        or value.get("source_pdf_path")
        or value.get("pdf_path")
        or ""
    ).strip()
    if local_pdf:
        payload["pdf_path"] = local_pdf
    for field in ("document_id", "content_scope", "route_sequence_hint"):
        text = str(value.get(field) or "").strip()
        if text:
            payload[field] = text
    labels = [
        str(item)
        for item in value.get("expected_scheme_or_compound_labels") or []
        if str(item or "").strip()
    ]
    if labels:
        payload["expected_labels"] = labels[:16]
    return payload


def _task_capability(task: dict[str, Any], *, source: dict[str, Any]) -> dict[str, Any]:
    payload = _source_payload_binding(source)
    payload.update(
        {
            "task_id": str(task.get("task_id") or ""),
            "label": str(task.get("label") or task.get("compound_label") or ""),
            "source_ref": str(task.get("source_ref") or payload.get("source_ref") or ""),
            "artifact_ref": str(task.get("artifact_ref") or ""),
            "run_visual": bool(task.get("run_visual", True)),
            "no_solved_claim": True,
        }
    )
    identity_payload = {
        "action_type": "resolve_literature_structure_task",
        "task_id": payload["task_id"],
        "payload_binding": payload,
    }
    return {
        "schema_version": SOURCE_CAPABILITY_SCHEMA,
        "capability_id": f"source-capability:sha256:{_digest(identity_payload)}",
        "action_type": "resolve_literature_structure_task",
        "document_identity": str(source.get("document_identity") or ""),
        "independent_source_group": str(source.get("independent_source_group") or ""),
        "source_ref": str(payload.get("source_ref") or ""),
        "source_title": str(payload.get("source_title") or ""),
        "stage_from": "structure_resolution_task_open",
        "stage_to": "structure_resolved_or_rejected",
        "payload_binding": payload,
        "source": source,
        "cost": action_resource_cost("resolve_literature_structure_task", payload),
        "priority": 220,
        "prerequisite_evidence_refs": [str(payload.get("artifact_ref") or "")],
        "eligible": True,
        "no_solved_claim": True,
    }


def _compile_capability(chain: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "chain_id": str(chain.get("chain_id") or chain.get("artifact_ref") or ""),
        "source_ref": str(chain.get("source_ref") or ""),
        "artifact_ref": str(chain.get("artifact_ref") or ""),
        "no_solved_claim": True,
    }
    payload = {key: value for key, value in payload.items() if value not in (None, "")}
    identity_payload = {
        "action_type": "compile_exact_literature_rows",
        "payload_binding": payload,
    }
    return {
        "schema_version": SOURCE_CAPABILITY_SCHEMA,
        "capability_id": f"source-capability:sha256:{_digest(identity_payload)}",
        "action_type": "compile_exact_literature_rows",
        "document_identity": _document_identity(chain),
        "independent_source_group": independent_source_group(chain),
        "source_ref": str(chain.get("source_ref") or ""),
        "source_title": str(chain.get("source_title") or chain.get("title") or ""),
        "stage_from": "visual_extracted",
        "stage_to": "exact_rows_compiled_or_rejected",
        "payload_binding": payload,
        "source": {},
        "cost": action_resource_cost("compile_exact_literature_rows", payload),
        "priority": 260,
        "prerequisite_evidence_refs": _visual_chain_ids(chain),
        "eligible": True,
        "no_solved_claim": True,
    }


def _generic_search_capability(max_sources: int) -> dict[str, Any]:
    payload = {"max_sources": max_sources, "no_solved_claim": True}
    identity_payload = {
        "action_type": "search_literature",
        "max_sources": max_sources,
    }
    return {
        "schema_version": SOURCE_CAPABILITY_SCHEMA,
        "capability_id": f"source-capability:sha256:{_digest(identity_payload)}",
        "action_type": "search_literature",
        "document_identity": "",
        "independent_source_group": "",
        "source_ref": "",
        "source_title": "",
        "stage_from": "source_gap",
        "stage_to": "source_candidates_discovered",
        "payload_binding": payload,
        "source": {},
        "cost": action_resource_cost("search_literature", payload),
        "priority": 50,
        "prerequisite_evidence_refs": [],
        "eligible": True,
        "no_solved_claim": True,
    }


def _append_eligible_or_blocked(
    capabilities: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    capability: dict[str, Any],
    *,
    reason: str,
) -> None:
    if not reason:
        capabilities.append(capability)
        return
    value = dict(capability)
    value["eligible"] = False
    value["blocked_reasons"] = [reason]
    blocked.append(value)


def _pending_sources(
    capabilities: list[dict[str, Any]],
    action_type: str,
) -> list[dict[str, Any]]:
    return [
        dict(row.get("source") or {})
        for row in capabilities
        if str(row.get("action_type") or "") == action_type
        and row.get("eligible") is True
        and isinstance(row.get("source"), Mapping)
    ]


def _source_for_task(
    task: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aliases = _source_aliases(task)
    matches = [
        (identity, document)
        for identity, document in documents.items()
        if aliases and aliases & set(document.get("aliases") or set())
    ]
    if len(matches) == 1:
        return _merged_source_record(matches[0][0], matches[0][1]["rows"])
    source = dict(task)
    source["document_identity"] = _document_identity(task)
    source["independent_source_group"] = independent_source_group(task)
    return source


def _open_structure_tasks(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in evidence.get("structure_resolution_tasks") or []
        if isinstance(row, Mapping)
        and str(row.get("status") or "open").strip().lower()
        in {"", "open", "pending", "ready"}
    ]


def _visual_chain_is_materialized(row: Mapping[str, Any]) -> bool:
    value = dict(row or {})
    if value.get("accepted") is True or _visual_step_count(value) > 0:
        return True
    reasons = {str(item) for item in value.get("reasons") or []}
    return "visual_input_images_missing" not in reasons


def _visual_chain_has_uncompiled_steps(
    chain: Mapping[str, Any],
    exact_rows: Iterable[Any],
) -> bool:
    count = _visual_step_count(chain)
    if count <= 0:
        return False
    chain_id = str(chain.get("chain_id") or chain.get("artifact_ref") or "").strip()
    identity = _document_identity(chain)
    compiled = 0
    for raw in exact_rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row_chain = str(
            row.get("chain_id") or row.get("visual_chain_id") or ""
        ).strip()
        if chain_id and row_chain == chain_id:
            compiled += 1
            continue
        if identity and _document_identity(row) == identity:
            compiled += 1
    return compiled < count


def _visual_step_count(row: Mapping[str, Any]) -> int:
    try:
        declared = int(row.get("candidate_step_count") or row.get("step_count") or 0)
    except (TypeError, ValueError):
        declared = 0
    steps = row.get("steps") or []
    return max(declared, len(steps) if isinstance(steps, list) else 0)


def _visual_chain_ids(row: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(row.get(field) or "").strip()
            for field in ("chain_id", "artifact_ref")
            if str(row.get(field) or "").strip()
        }
    )


def _evidence_refs(row: Mapping[str, Any], *, extra: Iterable[str] = ()) -> list[str]:
    return sorted(
        {
            *[str(item) for item in row.get("evidence_refs") or [] if str(item)],
            str(row.get("artifact_ref") or ""),
            *[str(item) for item in extra if str(item)],
        }
        - {""}
    )


def _source_priority(
    source: Mapping[str, Any],
    *,
    target_terms: Iterable[str] = (),
) -> int:
    local_pdf = str(
        source.get("local_pdf")
        or source.get("source_pdf_path")
        or source.get("pdf_path")
        or ""
    ).strip()
    score = 60 if local_pdf and Path(local_pdf).expanduser().is_file() else 0
    if str(source.get("access_status") or "").lower() == "local_pdf_available":
        score += 15
    if str(source.get("source_role") or "").lower() == "local_pdf_proxy_download":
        score += 10
    source_ref = str(source.get("source_ref") or "")
    if str(source.get("doi") or "").strip() or source_ref.lower().startswith("doi:"):
        score += 18
        title = str(source.get("title") or source.get("source_title") or "").strip().lower()
        if not title or title.startswith("pdfreq"):
            score += 35
    text = " ".join(
        str(source.get(field) or "")
        for field in (
            "title",
            "source_title",
            "route_sequence_hint",
            "relevance_rationale",
        )
    ).lower()
    score += 10 * sum(term and term in text for term in target_terms)
    score += 8 * sum(
        token in text
        for token in (
            "synthesis",
            "preparation",
            "process",
            "route",
            "scheme",
            "intermediate",
            "kilogram",
            "kg",
            "scale",
        )
    )
    if "improved kilogram-scale preparation" in text:
        score += 40
    if "discovery" in text and not any(
        token in text for token in ("synthesis", "preparation", "process", "scheme")
    ):
        score -= 10
    return score


def _priority_terms(values: Iterable[str]) -> list[str]:
    terms: set[str] = set()
    for value in values:
        for token in (
            str(value or "")
            .lower()
            .replace(";", " ")
            .replace(",", " ")
            .split()
        ):
            normalized = token.strip("()[]{}:._-/")
            if len(normalized) >= 5 and normalized not in {
                "online",
                "local",
                "cache",
                "source",
                "target",
            }:
                terms.add(normalized)
    return sorted(terms)[:10]


def _budget_remaining(budget: Mapping[str, Any], used_key: str, max_key: str) -> int:
    if max_key not in budget:
        return 1_000_000_000
    try:
        used = int(budget.get(used_key) or 0)
        maximum = int(budget.get(max_key) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, maximum - used)


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _capability_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row.get("priority") or 0),
        str(row.get("action_type") or ""),
        str(row.get("document_identity") or ""),
        str(row.get("capability_id") or ""),
    )


def _stable_row_key(row: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
