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
from cascade_planner.interfaces.literature_authorized_pdf_assets import (
    attach_authorized_pdf_assets,
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
            return attach_authorized_pdf_assets(
                structured_source,
                candidate=candidate,
                request=request,
                config=config,
                source_dir=source_dir,
                raw_cache_dir=raw_cache_dir,
                source_ref=source_ref,
                source_doi=source_doi,
                artifact=authorized_artifact,
                pdf_materializer=finalize_pdf_materialization,
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
            return attach_authorized_pdf_assets(
                html_source,
                candidate=candidate,
                request=request,
                config=config,
                source_dir=source_dir,
                raw_cache_dir=raw_cache_dir,
                source_ref=source_ref,
                source_doi=source_doi,
                artifact=authorized_artifact,
                pdf_materializer=finalize_pdf_materialization,
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
__all__ = ["materialize_candidate"]
