"""Hash-bound PMC HTML fallback between structured XML and PDF access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from cascade_planner.interfaces.literature_candidates import doi, request_queries
from cascade_planner.interfaces.literature_browser import (
    fetch_repository_html_with_browser,
)
from cascade_planner.interfaces.literature_html_parser import (
    PmcArticleParser,
    html_procedure_inventory,
    parse_pmc_html,
)
from cascade_planner.interfaces.literature_search import europe_pmc_repository_html


BytesFetcher = Callable[[str, float, int], bytes]


def materialize_pmc_repository_html(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_ref: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
    fetch: BytesFetcher,
    allow_browser_fallback: bool = True,
) -> dict[str, Any]:
    """Freeze full PMC HTML and extract bounded reaction-relevant sections."""

    source_doi = doi(candidate)
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    html_bytes = b""
    receipt: dict[str, Any] = {}
    parser: PmcArticleParser | None = None
    for cached_path in sorted(fulltext_cache_dir.glob("fulltext-*.html")):
        try:
            cached = cached_path.read_bytes()
        except OSError:
            continue
        if not 200 <= len(cached) <= config.max_fulltext_bytes:
            continue
        cached_parser = parse_pmc_html(cached)
        if cached_parser.citation_doi.casefold() != source_doi.casefold():
            continue
        cached_sha = hashlib.sha256(cached).hexdigest()
        cached_receipt = _cached_receipt(cached_path, html_sha256=cached_sha)
        pmcid = str(
            cached_parser.pmcid
            or cached_receipt.get("pmcid")
            or candidate.get("pmcid")
            or ""
        ).strip().upper()
        html_bytes = cached
        parser = cached_parser
        receipt = {
            "provider": "content_addressed_pmc_html_cache",
            "pmcid": pmcid,
            "doi": source_doi,
            "html_sha256": cached_sha,
            "html_url": str(cached_receipt.get("html_url") or "")
            or (
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                if pmcid
                else ""
            ),
            "repository_fulltext": True,
            "has_repository_fulltext": True,
            "access_class": "free_repository_fulltext",
            "cache_hit": True,
        }
        break
    if not html_bytes:
        html_bytes, receipt = europe_pmc_repository_html(
            source_doi,
            timeout_s=config.timeout_s,
            max_bytes=config.max_fulltext_bytes,
            fetch=fetch,
        )
        parser = parse_pmc_html(html_bytes)
        receipt = {**receipt, "cache_hit": False}
        if (
            allow_browser_fallback
            and parser.citation_doi.casefold() != source_doi.casefold()
            and _is_repository_browser_challenge(html_bytes)
        ):
            challenged_sha = hashlib.sha256(html_bytes).hexdigest()
            html_bytes = fetch_repository_html_with_browser(
                str(receipt.get("html_url") or ""),
                config.timeout_s,
                config.max_fulltext_bytes,
            )
            parser = parse_pmc_html(html_bytes)
            receipt = {
                **receipt,
                "transport": "isolated_playwright_repository_fallback",
                "http_challenge_sha256": challenged_sha,
                "browser_html_sha256": hashlib.sha256(html_bytes).hexdigest(),
            }
    assert parser is not None
    if parser.citation_doi.casefold() != source_doi.casefold():
        raise ValueError("pmc_repository_html_doi_mismatch")
    return materialize_parsed_html(
        parser=parser,
        html_bytes=html_bytes,
        receipt=receipt,
        candidate=candidate,
        request=request,
        source_ref=source_ref,
        source_doi=source_doi,
        source_dir=source_dir,
        fulltext_cache_dir=fulltext_cache_dir,
        config=config,
    )


def _is_repository_browser_challenge(content: bytes) -> bool:
    prefix = content[:64_000].lower()
    return any(
        marker in prefix
        for marker in (
            b"google.com/recaptcha/challenge",
            b"recaptcha/api.js",
            b"g-recaptcha",
        )
    )


def materialize_parsed_html(
    *,
    parser: PmcArticleParser,
    html_bytes: bytes,
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
    source_ref: str,
    source_doi: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
    acquisition_method: str = "pmc_repository_fulltext_html",
    artifact_kind: str = "pmc_fulltext_html",
    repository_semantics: bool = True,
) -> dict[str, Any]:
    html_sha = hashlib.sha256(html_bytes).hexdigest()
    cache_path = fulltext_cache_dir / f"fulltext-{html_sha[:16]}.html"
    _write_bytes_once(cache_path, html_bytes)
    _write_receipt_once(
        cache_path.with_suffix(".receipt.json"),
        {
            "schema_version": "pmc_html_cache_receipt.v1",
            "html_sha256": html_sha,
            "doi": source_doi,
            "pmcid": str(receipt.get("pmcid") or ""),
            "html_url": str(receipt.get("html_url") or ""),
        },
    )
    materialized = source_dir / "materialized-fulltext"
    materialized.mkdir(parents=True, exist_ok=True)
    html_path = materialized / f"fulltext-{html_sha[:16]}.html"
    _write_bytes_once(html_path, html_bytes)
    procedures = html_procedure_inventory(
        parser.sections,
        target_terms=[
            str(request.get("target_name") or ""),
            *[str(value) for value in request_queries(request)],
        ],
        source_artifact_sha256=html_sha,
        limit=config.max_fulltext_sections,
        source_artifact_kind=artifact_kind,
    )
    if not procedures:
        raise ValueError("pmc_repository_html_relevant_material_missing")
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": source_doi,
        "pmid": str(candidate.get("pmid") or ""),
        "pmcid": str(receipt.get("pmcid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "source_fulltext_sha256": html_sha,
        "fulltext_html_sha256": html_sha,
        "fulltext_html_path": str(html_path),
        "source_pdf_sha256": "",
        "pdf_sha256": "",
        "page_count": 0,
        "visual_candidate_pages": [],
        "procedure_inventory": procedures,
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": len(request.get("edges") or []) or 1,
        "focus_page_numbers": [],
        "acquisition_status": "materialized",
        "acquisition_method": acquisition_method,
        "acquisition_receipt": {
            **receipt,
            "cached_fulltext_path": str(cache_path),
        },
        "semantics": {
            "html_used_before_pdf": True,
            "html_used_after_xml_before_pdf": repository_semantics,
            "repository_access_is_distinct_from_open_access_licence": repository_semantics,
            "institutionally_authorized_source": not repository_semantics,
            "source_material_grants_no_exact_reaction_authority": True,
        },
    }


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(content).digest():
            raise ValueError("literature_content_address_collision")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _cached_receipt(path: Path, *, html_sha256: str) -> dict[str, Any]:
    receipt_path = path.with_suffix(".receipt.json")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(value, Mapping):
        return {}
    receipt = dict(value)
    if str(receipt.get("html_sha256") or "") != html_sha256:
        return {}
    return receipt


def _write_receipt_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            current = {}
        if (
            isinstance(current, Mapping)
            and str(current.get("html_sha256") or "")
            == str(value.get("html_sha256") or "")
        ):
            return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


__all__ = ["materialize_parsed_html", "materialize_pmc_repository_html"]
