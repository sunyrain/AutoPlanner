"""Attach authorized supplementary-PDF assets to structured literature sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


PdfMaterializer = Callable[..., dict[str, Any]]


def attach_authorized_pdf_assets(
    source: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
    config: Any,
    source_dir: Path,
    raw_cache_dir: Path,
    source_ref: str,
    source_doi: str,
    artifact: Mapping[str, Any],
    pdf_materializer: PdfMaterializer,
) -> dict[str, Any]:
    """Keep structured text authority while retaining downloaded SI figures."""

    pdf_path = Path(str(artifact.get("pdf_path") or "")).expanduser().resolve()
    if not pdf_path.is_file():
        return dict(source)
    try:
        pdf_source = pdf_materializer(
            candidate,
            request=request,
            config=config,
            source_dir=source_dir / "supplementary-pdf",
            raw_cache_dir=raw_cache_dir,
            source_ref=source_ref,
            source_doi=source_doi,
            content=pdf_path.read_bytes(),
            acquisition_method="authorized_publisher_supplementary_pdf",
            acquisition_receipt={
                "provider": str(artifact.get("provider") or "authorized_local_browser"),
                "artifact_kind": str(artifact.get("artifact_kind") or ""),
                "authorized_pdf_path": str(pdf_path),
            },
            structured_failure="",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        row = dict(source)
        receipt = dict(row.get("acquisition_receipt") or {})
        receipt["supplementary_pdf_materialization"] = {
            "status": "failed",
            "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
        }
        row["acquisition_receipt"] = receipt
        return row
    row = dict(source)
    row.update(
        source_pdf_sha256=str(pdf_source.get("source_pdf_sha256") or ""),
        pdf_sha256=str(pdf_source.get("pdf_sha256") or ""),
        source_pdf_path=str(pdf_source.get("source_pdf_path") or ""),
        page_count=int(pdf_source.get("page_count") or 0),
        visual_candidate_pages=list(pdf_source.get("visual_candidate_pages") or []),
        focus_page_numbers=list(pdf_source.get("focus_page_numbers") or []),
        target_focus=dict(pdf_source.get("target_focus") or {}),
        target_alias_hit_page_count=int(
            pdf_source.get("target_alias_hit_page_count") or 0
        ),
        acquisition_method=(
            f"{str(source.get('acquisition_method') or 'authorized_publisher_source')}"
            "_with_supplementary_pdf"
        ),
    )
    # The structured source already owns text extraction. Retain the SI only
    # for hash-bound visual pages; duplicating its full text snippets here can
    # overflow the bounded discovery observation before vision runs.
    row["procedure_inventory"] = [
        dict(value)
        for value in source.get("procedure_inventory") or []
        if isinstance(value, Mapping)
    ]
    receipt = dict(row.get("acquisition_receipt") or {})
    receipt["supplementary_pdf_materialization"] = {
        "status": "materialized",
        "pdf_sha256": row["source_pdf_sha256"],
        "visual_page_count": len(row["visual_candidate_pages"]),
    }
    row["acquisition_receipt"] = receipt
    row["semantics"] = {
        **dict(row.get("semantics") or {}),
        "structured_text_and_supplementary_pdf_are_jointly_retained": True,
        "supplementary_pdf_enables_visual_structure_binding": True,
    }
    return row


__all__ = ["attach_authorized_pdf_assets"]
