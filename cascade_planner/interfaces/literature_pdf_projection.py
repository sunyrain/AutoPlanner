"""Compile hash-bound replay projections from extracted literature PDF assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.harness.literature_page_selection import (
    select_pdf_visual_paths,
)


def target_aliases(request: Mapping[str, Any]) -> list[str]:
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


def compile_pdf_replay_assets(
    manifest: dict[str, Any],
    *,
    materialized_dir: Path,
    max_visual_pages: int,
    document_id: str,
    source_ref: str,
    pdf_path: Path,
    pdf_sha256: str,
) -> dict[str, Any]:
    manifest.update(
        {
            "source_ref": source_ref,
            "source_binding_audit": {
                "schema_version": "local_pdf_source_binding_audit.v1",
                "accepted": bool(source_ref),
                "source_ref": source_ref,
                "matched_source_count": 1 if source_ref else 0,
                "matched_document_ids": [document_id] if source_ref else [],
                "binding_method": "builtin_literature_identity_and_pdf_hash",
            },
        }
    )
    manifest_path = materialized_dir / "literature_pdf_structure_evidence.json"
    _write_manifest(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    selected_paths = select_pdf_visual_paths(
        manifest,
        max_images=max_visual_pages,
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
            "source_pdf_sha256": pdf_sha256,
            "page_number": int(page["page_number"]),
            "image_path": str(page["image_path"]),
            "image_sha256": str(page["image_sha256"]),
            "source_ref": source_ref,
        }
        for page in pages
    ]
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "pages": pages,
        "source_evidence": source_evidence,
        "target_focus": _target_focus_summary(manifest),
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


__all__ = ["compile_pdf_replay_assets", "target_aliases"]
