"""Compile ranked, hash-bound visual evidence requests without executing providers."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from cascade_planner.interfaces.visual_evidence_contract import (
    VISUAL_EVIDENCE_REQUEST_SCHEMA,
    digest as _digest,
    is_sha256 as _is_sha256,
    sha256 as _sha256,
    source_kind as _source_kind,
    source_ref as _source_ref,
)

def compile_visual_evidence_request(
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    max_pages: int,
) -> dict[str, Any]:
    request, _diagnostics = _compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=max_pages,
    )
    return request


def _compile_visual_evidence_request(
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    max_pages: int,
    excluded_source_refs: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 1 <= max_pages <= 12:
        raise ValueError("visual_evidence_page_limit_invalid")
    if str(discovery.get("request_sha256") or "") != str(
        evidence_request.get("content_sha256") or ""
    ):
        return {}, [
            {
                "source_ref": "",
                "status": "rejected",
                "reasons": ["evidence_discovery_request_digest_mismatch"],
            }
        ]
    candidates = []
    excluded = {str(value) for value in (excluded_source_refs or set()) if str(value)}
    diagnostics: list[dict[str, Any]] = []
    for source in discovery.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        publication_number = str(source.get("publication_number") or "").strip()
        source_kind = str(source.get("source_kind") or "").strip().lower()
        source_ref = _source_ref(source)
        diagnostic: dict[str, Any] = {
            "source_ref": source_ref,
            "source_kind": source_kind or _source_kind(source_ref),
            "status": "rejected",
            "reasons": [],
            "declared_page_count": len(source.get("visual_candidate_pages") or []),
            "valid_page_count": 0,
        }
        if source_ref in excluded:
            diagnostic["reasons"] = ["excluded_after_visual_source_attempt"]
            diagnostics.append(diagnostic)
            continue
        source_pdf_sha256 = str(
            source.get("source_pdf_sha256") or source.get("pdf_sha256") or ""
        ).strip().lower()
        source_fulltext_sha256 = str(
            source.get("source_fulltext_sha256")
            or source.get("fulltext_xml_sha256")
            or ""
        ).strip().lower()
        source_artifact_sha256 = source_fulltext_sha256 or source_pdf_sha256
        if not source_ref or not _is_sha256(source_artifact_sha256):
            diagnostic["reasons"] = ["hash_bound_source_artifact_missing"]
            diagnostics.append(diagnostic)
            continue
        exact_row_count = int(source.get("exact_row_count") or 0)
        unresolved_edge_count = int(source.get("unresolved_edge_count") or 0)
        if unresolved_edge_count <= 0 and exact_row_count > 0:
            diagnostic["reasons"] = ["source_exact_rows_already_close_frontier"]
            diagnostics.append(diagnostic)
            continue
        target_relevance = _visual_source_target_relevance(
            source,
            evidence_request=evidence_request,
        )
        if target_relevance["accepted"] is not True:
            diagnostic["reasons"] = list(target_relevance.get("reasons") or [])
            diagnostics.append(diagnostic)
            continue
        pages = []
        for page in source.get("visual_candidate_pages") or []:
            if not isinstance(page, Mapping):
                continue
            row = dict(page)
            path = Path(str(row.get("image_path") or "")).expanduser().resolve()
            digest = str(row.get("image_sha256") or "")
            page_number = int(row.get("page_number") or 0)
            if (
                page_number <= 0
                or not path.is_file()
                or not _is_sha256(digest)
                or _sha256(path) != digest
            ):
                continue
            pages.append(
                {
                    "page_number": page_number,
                    "image_path": str(path),
                    "image_sha256": digest,
                }
            )
            if len(pages) >= max_pages:
                break
        diagnostic["valid_page_count"] = len(pages)
        if not pages:
            diagnostic["reasons"] = ["valid_visual_candidate_pages_missing"]
            diagnostics.append(diagnostic)
            continue
        labels = [
            str(row.get("label") or "")
            for row in source.get("procedure_inventory") or []
            if isinstance(row, Mapping)
            and row.get("visual_expected") is not False
            and str(row.get("label") or "").strip()
        ]
        selected_page_numbers = {
            int(row.get("page_number") or 0) for row in pages
        }
        text_snippets = []
        for raw in source.get("procedure_inventory") or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            page_number = int(row.get("page_number") or 0)
            excerpt = " ".join(
                str(
                    row.get("procedure_excerpt")
                    or row.get("procedure")
                    or row.get("text")
                    or ""
                ).split()
            )
            if (
                not excerpt
                or (selected_page_numbers and page_number not in selected_page_numbers)
            ):
                continue
            text_snippets.append(
                {
                    "compound_label": str(row.get("label") or row.get("name") or "")[:200],
                    "page_number": page_number,
                    "snippet": excerpt[:1_200],
                }
            )
            if len(text_snippets) >= 12:
                break
        route_rows = [
            dict(row)
            for row in dict(source.get("source_route_observation") or {}).get(
                "proposals"
            )
            or []
            if isinstance(row, Mapping)
        ]
        route_sequence_hint = " -> ".join(
            str(
                row.get("product_name")
                or dict(row.get("source_location") or {}).get("label")
                or row.get("proposal_id")
                or ""
            )[:160]
            for row in route_rows[:16]
            if str(
                row.get("product_name")
                or dict(row.get("source_location") or {}).get("label")
                or row.get("proposal_id")
                or ""
            ).strip()
        )
        candidates.append(
            {
                "source_ref": source_ref,
                "source_kind": source_kind or _source_kind(source_ref),
                "publication_number": publication_number,
                "doi": str(source.get("doi") or "")[:500],
                "pmid": str(source.get("pmid") or "")[:100],
                "family_id": str(source.get("family_id") or ""),
                "title": str(source.get("title") or "")[:1000],
                "source_pdf_sha256": source_pdf_sha256,
                "source_fulltext_sha256": source_fulltext_sha256,
                "source_artifact_sha256": source_artifact_sha256,
                "source_artifact_kind": (
                    "europe_pmc_fulltext_xml"
                    if source_fulltext_sha256
                    else "pdf"
                ),
                "expected_labels": list(dict.fromkeys(labels))[:24],
                "text_snippets": text_snippets,
                "route_sequence_hint": route_sequence_hint[:2_000],
                "pages": pages,
                "exact_row_count": exact_row_count,
                "unresolved_edge_count": unresolved_edge_count,
                "source_route_proposal_count": int(
                    source.get("source_route_proposal_count") or len(route_rows)
                ),
                "procedure_count": len(source.get("procedure_inventory") or []),
                "target_relevance": target_relevance,
                "target_relevance_priority": int(
                    target_relevance.get("priority") or 0
                ),
            }
        )
        diagnostic["status"] = "eligible"
        diagnostic["reasons"] = list(target_relevance.get("reasons") or [])
        diagnostics.append(diagnostic)
    if not candidates:
        return {}, diagnostics
    candidates.sort(
        key=lambda row: (
            -int(row["target_relevance_priority"]),
            -int(row["source_route_proposal_count"]),
            -int(row["procedure_count"]),
            -int(row["unresolved_edge_count"]),
            int(row["exact_row_count"] > 0),
            str(row["source_ref"]),
        )
    )
    request = {
        "schema_version": VISUAL_EVIDENCE_REQUEST_SCHEMA,
        "evidence_request_sha256": str(evidence_request.get("content_sha256") or ""),
        "run_id": str(evidence_request.get("run_id") or ""),
        "target_name": str(evidence_request.get("target_name") or ""),
        "target_smiles": str(evidence_request.get("target_smiles") or ""),
        "target_identity": dict(evidence_request.get("target_identity") or {}),
        "edges": [dict(row) for row in evidence_request.get("edges") or []],
        "source": candidates[0],
        "selection_diagnostics": diagnostics,
        "limits": {"max_pages": max_pages, "max_model_invocations": 1},
        "semantics": {
            "visual_output_is_hypothesis_only": True,
            "visual_output_cannot_grant_L2_L3_or_stock": True,
            "host_smiles_and_reaction_normalization_required": True,
        },
    }
    request["content_sha256"] = _digest(request)
    return request, diagnostics


def _visual_source_target_relevance(
    source: Mapping[str, Any],
    *,
    evidence_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank strong bridges first while allowing one bounded structure-first pass.

    Search results mentioning a therapeutic class are common patent noise.
    They may remain frozen source observations, but vision is reserved for a
    named/structural bridge, reaction-bearing procedure context, or a
    materialized target-ranked paper whose missing bridge is exactly what
    visual structure extraction is expected to recover.
    """

    target_name = " ".join(
        str(evidence_request.get("target_name") or "").split()
    ).casefold()
    generic_name = (
        not target_name
        or target_name in {"target", "blind target"}
        or "blind" in target_name
        or bool(re.fullmatch(r"target-[0-9a-f]{8,64}", target_name))
    )
    identity = dict(evidence_request.get("target_identity") or {})
    name_terms = {
        target_name,
        " ".join(str(identity.get("preferred_name") or "").split()).casefold(),
        *(
            " ".join(str(value).split()).casefold()
            for value in identity.get("synonyms") or []
        ),
    } - {""}
    searchable = " ".join(
        [
            str(source.get("title") or ""),
            *[
                str(item.get("name") or item.get("label") or "")
                for item in source.get("procedure_inventory") or []
                if isinstance(item, Mapping)
            ],
        ]
    ).casefold()
    named_match = any(term in searchable for term in name_terms)
    family_terms = _target_chemical_family_terms(name_terms)
    matched_family_terms = sorted(
        term for term in family_terms if term in searchable
    )
    family_match = bool(matched_family_terms)
    target_alias_pdf_match = bool(
        int(source.get("target_alias_hit_page_count") or 0) > 0
        or int(dict(source.get("target_focus") or {}).get("target_alias_hit_page_count") or 0)
        > 0
    )
    exact_match = bool(
        int(source.get("exact_row_count") or 0)
        or source.get("exact_edge_ids")
        or int(source.get("source_route_exact_row_count") or 0)
    )
    target_smiles = str(evidence_request.get("target_smiles") or "")
    frontier_products = {
        target_smiles,
        *(
            str(edge.get("product_smiles") or "")
            for edge in evidence_request.get("edges") or []
            if isinstance(edge, Mapping)
        ),
    } - {""}
    proposals = [
        dict(value)
        for value in dict(source.get("source_route_observation") or {}).get(
            "proposals"
        )
        or []
        if isinstance(value, Mapping)
    ]
    connected_route = any(
        str(proposal.get("product_smiles") or "") in frontier_products
        or str(proposal.get("root_anchor") or "").strip()
        for proposal in proposals
    )
    procedure_rows = [
        dict(item)
        for item in source.get("procedure_inventory") or []
        if isinstance(item, Mapping)
    ]
    procedure_context = any(
        len(excerpt) >= 60
        and sum(
            signal in excerpt.casefold()
            for signal in (
                " was added",
                " were added",
                "stirred",
                "reaction mixture",
                "afforded",
                "yield",
                "purified",
                "synthesis",
            )
        )
        >= 2
        for excerpt in (
            " ".join(
                str(
                    item.get("procedure_excerpt")
                    or item.get("procedure")
                    or item.get("text")
                    or ""
                ).split()
            )
            for item in procedure_rows
        )
    )
    materialized_unbound_paper = bool(
        str(source.get("acquisition_status") or "").lower() == "materialized"
        and _source_kind(_source_ref(source)) in {"paper_si", "doi", "pmid", "pmc"}
        and source.get("visual_candidate_pages")
        and int(source.get("unresolved_edge_count") or 0) > 0
    )
    accepted = bool(
        generic_name
        or named_match
        or target_alias_pdf_match
        or family_match
        or exact_match
        or connected_route
        or procedure_context
        or materialized_unbound_paper
    )
    reasons = [
        reason
        for condition, reason in (
            (generic_name, "generic_target_name_cannot_support_text_filter"),
            (named_match, "named_target_mentioned_in_source"),
            (
                target_alias_pdf_match,
                "target_identity_alias_mentioned_in_native_pdf_text",
            ),
            (family_match, "target_chemical_family_mentioned_in_source"),
            (exact_match, "source_matches_current_exact_edge"),
            (connected_route, "source_route_connects_to_target_frontier"),
            (procedure_context, "reaction_procedure_context_requires_visual_binding"),
            (
                materialized_unbound_paper,
                "materialized_target_ranked_paper_requires_visual_structure_binding",
            ),
        )
        if condition
    ]
    if not accepted:
        reasons.append("source_has_no_target_or_frontier_bridge")
    return {
        "schema_version": "visual_source_target_relevance.v1",
        "accepted": accepted,
        "priority": (
            100
            if exact_match or connected_route
            else 95
            if target_alias_pdf_match
            else 90
            if named_match
            else 80
            if family_match
            else 60
            if procedure_context
            else 30
            if materialized_unbound_paper
            else 10
            if generic_name
            else 0
        ),
        "reasons": reasons,
        "matched_family_terms": matched_family_terms,
        "semantics": {
            "search_result_presence_is_not_relevance": True,
            "rejected_source_bytes_remain_frozen_for_audit": True,
            "bounded_structure_first_visual_pass_breaks_binding_deadlock": True,
        },
    }


def _target_chemical_family_terms(name_terms: set[str]) -> set[str]:
    """Derive conservative suffix roots from structure-resolved target names.

    Chemical identifiers frequently prepend substituents to the actual family
    name (for example, ``pentamethylenefulvene``).  Exact phrase matching then
    misses a paper headed simply "Fulvenes".  Long suffix roots recover that
    relationship without treating generic words such as "target" as chemistry.
    """

    ignored = {
        "compound",
        "research",
        "target",
        "synthesis",
        "preparation",
        "product",
        "unknown",
    }
    roots: set[str] = set()
    for phrase in name_terms:
        for token in re.findall(r"[a-z][a-z0-9]{5,}", phrase.casefold()):
            if token in ignored or token.isdigit():
                continue
            for width in range(7, min(14, len(token)) + 1):
                suffix = token[-width:]
                if suffix not in ignored:
                    roots.add(suffix)
            if len(token) <= 18:
                roots.add(token)
    return roots


def _visual_no_candidate_reason(
    diagnostics: list[dict[str, Any]],
) -> str:
    if not diagnostics:
        return "visual_candidate_sources_missing"
    reasons = {
        str(reason)
        for row in diagnostics
        for reason in row.get("reasons") or []
        if str(reason)
    }
    if "evidence_discovery_request_digest_mismatch" in reasons:
        return "visual_discovery_request_mismatch"
    if any(int(row.get("declared_page_count") or 0) > 0 for row in diagnostics):
        if "source_has_no_target_or_frontier_bridge" in reasons:
            return "visual_candidate_sources_rejected"
        return "visual_candidate_pages_invalid_or_filtered"
    if "hash_bound_source_artifact_missing" in reasons:
        return "visual_source_artifacts_missing"
    return "visual_candidate_pages_missing"

__all__ = ["compile_visual_evidence_request"]
