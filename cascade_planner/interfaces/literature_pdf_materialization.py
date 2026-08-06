"""Finalize validated literature PDF bytes into bounded replay assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from cascade_planner.harness.literature_page_selection import (
    select_pdf_page_numbers,
    select_pdf_visual_paths,
)
from cascade_planner.harness.literature_pdf_extraction import (
    extract_literature_pdf_assets,
    rebuild_literature_pdf_page_focus,
)
from cascade_planner.interfaces.literature_candidates import (
    pdf_page_count,
    request_queries,
)
from cascade_planner.interfaces.literature_html_parser import html_procedure_inventory


def finalize_pdf_materialization(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    config: Any,
    source_dir: Path,
    raw_cache_dir: Path,
    source_ref: str,
    source_doi: str,
    content: bytes,
    acquisition_method: str,
    acquisition_receipt: Mapping[str, Any],
    structured_failure: str,
    pdf_page_counter: Callable[[Path], int] = pdf_page_count,
    focus_builder: Callable[..., Mapping[str, Any]] = rebuild_literature_pdf_page_focus,
    asset_extractor: Callable[..., dict[str, Any]] = extract_literature_pdf_assets,
) -> dict[str, Any]:
    if len(content) > config.max_pdf_bytes or not content.startswith(b"%PDF-"):
        raise ValueError("paper_pdf_invalid_or_too_large")
    pdf_sha = hashlib.sha256(content).hexdigest()
    pdf_path = raw_cache_dir / f"source-{pdf_sha[:16]}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(content)
    page_count = pdf_page_counter(pdf_path)
    if page_count < 1 or page_count > config.max_pdf_pages:
        raise ValueError(f"paper_pdf_page_limit:{page_count}")
    route_hint = "; ".join(request_queries(request))
    target_aliases = _target_aliases(request)
    focus = focus_builder(
        pdf_path,
        target_name=str(request.get("target_name") or ""),
        target_aliases=target_aliases,
        route_sequence_hint=route_hint,
    )
    page_numbers = select_pdf_page_numbers(
        focus, page_count=page_count, max_pages=config.max_visual_pages
    )
    materialized_dir = source_dir / "materialized"
    manifest = asset_extractor(
        pdf_path=pdf_path,
        output_dir=materialized_dir,
        page_numbers=page_numbers,
        target_name=str(request.get("target_name") or ""),
        target_aliases=target_aliases,
        route_sequence_hint=route_hint,
        render_zoom=config.render_zoom,
    )
    document_id = (
        f"paper:{source_doi.casefold()}"
        if source_doi
        else f"paper-sha256:{pdf_sha[:24]}"
    )
    manifest.update(
        {
            "source_ref": source_ref,
            "source_binding_audit": {
                "schema_version": "local_pdf_source_binding_audit.v1",
                "accepted": bool(source_ref),
                "source_ref": source_ref,
                "matched_source_count": 1 if source_ref else 0,
                "matched_document_ids": [document_id] if source_ref else [],
                "binding_method": (
                    "builtin_literature_identity_and_pdf_hash"
                ),
            },
        }
    )
    manifest_path = materialized_dir / "literature_pdf_structure_evidence.json"
    _write_manifest(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    selected_paths = select_pdf_visual_paths(
        manifest, max_images=config.max_visual_pages
    )
    asset_rows = [
        dict(row)
        for row in [
            *(manifest.get("scheme_crops") or []),
            *(manifest.get("rendered_pages") or []),
        ]
        if isinstance(row, Mapping)
    ]
    by_path = {
        str(row.get("image_path") or ""): row
        for row in asset_rows
        if str(row.get("image_path") or "")
    }
    selected = [by_path[path] for path in selected_paths if path in by_path]
    pages = [
        {
            "page_number": int(row.get("page_number") or 0),
            "image_path": str(row.get("image_path") or ""),
            "image_sha256": str(row.get("sha256") or row.get("image_sha256") or ""),
        }
        for row in selected
        if int(row.get("page_number") or 0) > 0
        and str(row.get("image_path") or "")
        and str(row.get("sha256") or "")
    ]
    if not pages:
        raise ValueError("paper_pdf_rendered_pages_missing")
    source_evidence = [
        {
            "schema_version": "materialized_source_evidence.v1",
            "document_id": document_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "source_pdf_path": str(pdf_path),
            "source_pdf_sha256": pdf_sha,
            "page_number": int(page["page_number"]),
            "image_path": str(page["image_path"]),
            "image_sha256": str(page["image_sha256"]),
            "source_ref": source_ref,
        }
        for page in pages
    ]
    target_focus = _target_focus_summary(manifest)
    fulltext_path = Path(str(manifest.get("fulltext_path") or ""))
    fulltext_sha = ""
    procedures: list[dict[str, object]] = []
    if fulltext_path.is_file():
        fulltext_bytes = fulltext_path.read_bytes()
        fulltext_sha = hashlib.sha256(fulltext_bytes).hexdigest()
        manifest_fulltext_sha = str(manifest.get("fulltext_sha256") or "").lower()
        if manifest_fulltext_sha and manifest_fulltext_sha != fulltext_sha:
            raise ValueError("paper_pdf_fulltext_hash_mismatch")
        try:
            fulltext = fulltext_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("paper_pdf_fulltext_not_utf8") from exc
        procedures = html_procedure_inventory(
            [("PDF extracted full text", fulltext)],
            target_terms=[
                *target_aliases,
                *[str(value) for value in request_queries(request)],
            ],
            source_artifact_sha256=fulltext_sha,
            limit=config.max_fulltext_sections,
            source_artifact_kind="pdf_text_layer",
        )
    receipt = dict(acquisition_receipt)
    if structured_failure:
        receipt["structured_fulltext_attempt"] = {
            "status": "failed",
            "reason": structured_failure,
        }
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": source_doi,
        "pmid": str(candidate.get("pmid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "source_pdf_sha256": pdf_sha,
        "pdf_sha256": pdf_sha,
        "source_fulltext_sha256": fulltext_sha,
        "fulltext_text_sha256": fulltext_sha,
        "fulltext_text_path": str(fulltext_path) if fulltext_path.is_file() else "",
        "page_count": page_count,
        "visual_candidate_pages": pages,
        "procedure_inventory": procedures,
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": len(request.get("edges") or []) or 1,
        "focus_page_numbers": [int(row["page_number"]) for row in pages],
        "target_focus": target_focus,
        "target_alias_hit_page_count": int(
            target_focus.get("target_alias_hit_page_count") or 0
        ),
        "source_pdf_path": str(pdf_path),
        "document_id": document_id,
        "materialization_manifest_path": str(manifest_path),
        "materialization_manifest_sha256": manifest_sha256,
        "source_evidence": source_evidence,
        "acquisition_status": "materialized",
        "acquisition_method": acquisition_method,
        "acquisition_receipt": receipt,
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish the source binding and its digest as one replayable snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _target_aliases(request: Mapping[str, Any]) -> list[str]:
    """Return bounded structure-resolved names for PDF text/page ranking."""

    identity = dict(request.get("target_identity") or {})
    values = [
        str(request.get("target_name") or ""),
        str(identity.get("preferred_name") or ""),
        *[str(value) for value in identity.get("synonyms") or []],
    ]
    aliases: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(raw.split())[:500]
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        aliases.append(value)
        if len(aliases) >= 16:
            break
    return aliases


def _target_focus_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    hits = [
        dict(row)
        for row in manifest.get("focus_hit_audit") or []
        if isinstance(row, Mapping)
        and str(row.get("source") or "") in {"target_name", "target_alias"}
        and row.get("matched_page_numbers")
    ]
    pages = sorted(
        {
            int(page)
            for row in hits
            for page in row.get("matched_page_numbers") or []
            if int(page) > 0
        }
    )
    return {
        "schema_version": "literature_target_focus.v1",
        "target_alias_hit_page_count": len(pages),
        "target_alias_hit_page_numbers": pages,
        "matched_target_terms": sorted(
            {str(row.get("term") or "") for row in hits if str(row.get("term") or "")}
        )[:16],
        "native_pdf_text_only": True,
        "grants_no_structure_or_reaction_authority": True,
    }


__all__ = ["finalize_pdf_materialization"]
