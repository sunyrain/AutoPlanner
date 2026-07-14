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
from cascade_planner.harness.literature_page_selection import (
    select_pdf_page_numbers,
    select_pdf_visual_paths,
)
from cascade_planner.interfaces.literature_access import authorized_proxy_pdf
from cascade_planner.interfaces.literature_candidates import (
    candidate_source_ref,
    doi,
    pdf_page_count,
    request_queries,
)
from cascade_planner.interfaces.literature_fulltext import (
    materialize_europe_pmc_fulltext,
)
from cascade_planner.interfaces.literature_html import materialize_pmc_repository_html
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
            )
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            structured_failure += (
                f"|pmc_html:{type(exc).__name__}:{str(exc)[:300]}"
            )
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
                pdf_url = citation_pdf_url(
                    source_doi, timeout_s=config.timeout_s, fetch=fetch
                )
                if pdf_url:
                    content = fetch(pdf_url, config.timeout_s, config.max_pdf_bytes)
                    acquisition_method = "doi_landing_citation_pdf"
            except (OSError, RuntimeError, ValueError, requests.RequestException):
                content = b""
        if not content:
            raise ValueError("paper_pdf_url_missing_or_unreachable")
    if len(content) > config.max_pdf_bytes or not content.startswith(b"%PDF-"):
        raise ValueError("paper_pdf_invalid_or_too_large")
    pdf_sha = hashlib.sha256(content).hexdigest()
    pdf_path = raw_cache_dir / f"source-{pdf_sha[:16]}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(content)
    page_count = pdf_page_count(pdf_path)
    if page_count < 1 or page_count > config.max_pdf_pages:
        raise ValueError(f"paper_pdf_page_limit:{page_count}")
    route_hint = "; ".join(request_queries(request))
    focus = rebuild_literature_pdf_page_focus(
        pdf_path,
        target_name=str(request.get("target_name") or ""),
        target_aliases=[str(request.get("target_name") or "")],
        route_sequence_hint=route_hint,
    )
    page_numbers = select_pdf_page_numbers(
        focus, page_count=page_count, max_pages=config.max_visual_pages
    )
    manifest = extract_literature_pdf_assets(
        pdf_path=pdf_path,
        output_dir=source_dir / "materialized",
        page_numbers=page_numbers,
        target_name=str(request.get("target_name") or ""),
        target_aliases=[str(request.get("target_name") or "")],
        route_sequence_hint=route_hint,
        render_zoom=config.render_zoom,
    )
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
    if structured_failure:
        acquisition_receipt = {
            **acquisition_receipt,
            "structured_fulltext_attempt": {
                "status": "failed",
                "reason": structured_failure,
            },
        }
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": source_doi,
        "pmid": str(candidate.get("pmid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "source_pdf_sha256": pdf_sha,
        "pdf_sha256": pdf_sha,
        "page_count": page_count,
        "visual_candidate_pages": pages,
        "procedure_inventory": [],
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": len(request.get("edges") or []) or 1,
        "focus_page_numbers": [int(row["page_number"]) for row in pages],
        "source_pdf_path": str(pdf_path),
        "acquisition_status": "materialized",
        "acquisition_method": acquisition_method,
        "acquisition_receipt": acquisition_receipt,
    }


__all__ = ["materialize_candidate"]
