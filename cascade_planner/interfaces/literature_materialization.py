"""Structured-fulltext-first materialization with bounded PDF fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from cascade_planner.harness.literature_pdf_extraction import (
    extract_literature_pdf_assets,
    rebuild_literature_pdf_page_focus,
)
from cascade_planner.interfaces.literature_access import (
    authorized_proxy_artifact,
    authorized_proxy_pdf,
)
from cascade_planner.interfaces.literature_candidates import (
    candidate_source_ref,
    doi,
    pdf_page_count,
)
from cascade_planner.interfaces.literature_fulltext import (
    materialize_europe_pmc_fulltext,
)
from cascade_planner.interfaces.literature_figshare import (
    acs_figshare_supplementary_pdf,
)
from cascade_planner.interfaces.literature_authorized_source import (
    materialize_authorized_publisher_html,
    materialize_authorized_publisher_json,
)
from cascade_planner.interfaces.literature_html import (
    materialize_pmc_repository_html,
)
from cascade_planner.interfaces.literature_pdf_materialization import (
    finalize_pdf_materialization,
)
from cascade_planner.interfaces.literature_search import (
    citation_pdf_url,
    europe_pmc_open_access_pdf,
)


BytesFetcher = Callable[[str, float, int], bytes]


def materialize_candidate(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    output_dir: Path,
    config: Any,
    fetch: BytesFetcher,
    proxy_root: Path,
) -> dict[str, Any]:
    source_ref = candidate_source_ref(candidate)
    if not source_ref:
        raise ValueError("paper_source_ref_missing")
    slug = hashlib.sha256(source_ref.encode("utf-8")).hexdigest()[:20]
    source_dir = output_dir / slug
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = output_dir.parent / "_source_pdf_cache" / slug
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    source_doi = doi(candidate)
    structured_failure = ""
    if config.enable_structured_fulltext_first and source_doi:
        try:
            return materialize_europe_pmc_fulltext(
                candidate,
                request=request,
                source_ref=source_ref,
                source_dir=source_dir,
                fulltext_cache_dir=output_dir.parent / "_source_fulltext_cache" / slug,
                config=config,
                fetch=fetch,
            )
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            structured_failure = f"{type(exc).__name__}:{str(exc)[:300]}"
        try:
            return materialize_pmc_repository_html(
                candidate,
                request=request,
                source_ref=source_ref,
                source_dir=source_dir,
                fulltext_cache_dir=output_dir.parent / "_source_fulltext_cache" / slug,
                config=config,
                fetch=fetch,
                allow_browser_fallback=config.enable_repository_browser_fallback,
            )
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            structured_failure += f"|pmc_html:{type(exc).__name__}:{str(exc)[:300]}"
    authorized_artifact = authorized_proxy_artifact(
        candidate,
        proxy_root=proxy_root,
        source_ref=source_ref,
        doi=source_doi,
    )
    if authorized_artifact.get("structured_path"):
        try:
            structured_source = materialize_authorized_publisher_json(
                candidate,
                request=request,
                source_ref=source_ref,
                source_dir=source_dir,
                fulltext_cache_dir=output_dir.parent / "_source_fulltext_cache" / slug,
                config=config,
                artifact=authorized_artifact,
            )
            return _attach_authorized_pdf_assets(
                structured_source,
                candidate=candidate,
                request=request,
                config=config,
                source_dir=source_dir,
                raw_cache_dir=raw_cache_dir,
                source_ref=source_ref,
                source_doi=source_doi,
                artifact=authorized_artifact,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            structured_failure += f"|authorized_json:{type(exc).__name__}:{str(exc)[:300]}"
    if authorized_artifact.get("html_path"):
        try:
            html_source = materialize_authorized_publisher_html(
                candidate,
                request=request,
                source_ref=source_ref,
                source_dir=source_dir,
                fulltext_cache_dir=output_dir.parent / "_source_fulltext_cache" / slug,
                config=config,
                artifact=authorized_artifact,
            )
            return _attach_authorized_pdf_assets(
                html_source,
                candidate=candidate,
                request=request,
                config=config,
                source_dir=source_dir,
                raw_cache_dir=raw_cache_dir,
                source_ref=source_ref,
                source_doi=source_doi,
                artifact=authorized_artifact,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            structured_failure += f"|authorized_html:{type(exc).__name__}:{str(exc)[:300]}"
    content = b""
    acquisition_method = ""
    acquisition_receipt: dict[str, Any] = {}
    for cached_pdf in sorted(raw_cache_dir.glob("source-*.pdf")):
        try:
            cached_content = cached_pdf.read_bytes()
        except OSError:
            continue
        if cached_content.startswith(b"%PDF-") and len(cached_content) <= config.max_pdf_bytes:
            content = cached_content
            acquisition_method = "content_addressed_pdf_cache"
            acquisition_receipt = {
                "cache_hit": True,
                "pdf_sha256": hashlib.sha256(content).hexdigest(),
            }
            break
    local_pdf = str(candidate.get("local_pdf") or "").strip()
    if not local_pdf and authorized_artifact.get("pdf_path"):
        local_pdf = str(authorized_artifact.get("pdf_path") or "")
    if not local_pdf:
        local_pdf = authorized_proxy_pdf(
            candidate,
            proxy_root=proxy_root,
            source_ref=source_ref,
            doi=source_doi,
        )
    if not content and local_pdf:
        path = Path(local_pdf).expanduser().resolve()
        if not path.is_file():
            raise ValueError("paper_seed_pdf_missing")
        content = path.read_bytes()
        acquisition_method = "authorized_proxy_or_seed_pdf"
    elif not content:
        pdf_url = str(candidate.get("pdf_url") or "").strip()
        if pdf_url:
            try:
                content = fetch(pdf_url, config.timeout_s, config.max_pdf_bytes)
                acquisition_method = "crossref_direct_pdf"
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                content = b""
        if not content and source_doi:
            try:
                content, acquisition_receipt = acs_figshare_supplementary_pdf(
                    source_doi,
                    timeout_s=config.timeout_s,
                    max_bytes=config.max_pdf_bytes,
                    fetch=fetch,
                )
                acquisition_method = "acs_figshare_public_supplement"
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                content = b""
        if not content and source_doi:
            try:
                content, acquisition_receipt = europe_pmc_open_access_pdf(
                    source_doi,
                    timeout_s=config.timeout_s,
                    max_bytes=config.max_pdf_bytes,
                    fetch=fetch,
                )
                acquisition_method = "europe_pmc_open_access_archive"
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                content = b""
        if not content and source_doi:
            try:
                pdf_url = citation_pdf_url(source_doi, timeout_s=config.timeout_s, fetch=fetch)
                if pdf_url:
                    content = fetch(pdf_url, config.timeout_s, config.max_pdf_bytes)
                    acquisition_method = "doi_landing_citation_pdf"
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                content = b""
        if not content:
            raise ValueError("paper_pdf_url_missing_or_unreachable")
    return finalize_pdf_materialization(
        candidate,
        request=request,
        config=config,
        source_dir=source_dir,
        raw_cache_dir=raw_cache_dir,
        source_ref=source_ref,
        source_doi=source_doi,
        content=content,
        acquisition_method=acquisition_method,
        acquisition_receipt=acquisition_receipt,
        structured_failure=structured_failure,
        pdf_page_counter=pdf_page_count,
        focus_builder=rebuild_literature_pdf_page_focus,
        asset_extractor=extract_literature_pdf_assets,
    )


def _attach_authorized_pdf_assets(
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
) -> dict[str, Any]:
    """Keep structured text authority while retaining downloaded SI figures."""

    pdf_path = Path(str(artifact.get("pdf_path") or "")).expanduser().resolve()
    if not pdf_path.is_file():
        return dict(source)
    try:
        pdf_source = finalize_pdf_materialization(
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
    # The structured source already owns text extraction.  Retain the SI only
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


__all__ = ["materialize_candidate"]
