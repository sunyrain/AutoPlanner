"""Hash-bound PMC HTML fallback between structured XML and PDF access."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.interfaces.literature_candidates import doi, request_queries
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
) -> dict[str, Any]:
    """Freeze full PMC HTML and extract bounded reaction-relevant sections."""

    source_doi = doi(candidate)
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    html_bytes = b""
    receipt: dict[str, Any] = {}
    parser: _PmcArticleParser | None = None
    for cached_path in sorted(fulltext_cache_dir.glob("fulltext-*.html")):
        try:
            cached = cached_path.read_bytes()
        except OSError:
            continue
        if not 200 <= len(cached) <= config.max_fulltext_bytes:
            continue
        cached_parser = _parse_pmc_html(cached)
        if cached_parser.citation_doi.casefold() != source_doi.casefold():
            continue
        html_bytes = cached
        parser = cached_parser
        receipt = {
            "provider": "content_addressed_pmc_html_cache",
            "pmcid": cached_parser.pmcid,
            "doi": source_doi,
            "html_sha256": hashlib.sha256(cached).hexdigest(),
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
        parser = _parse_pmc_html(html_bytes)
        receipt = {**receipt, "cache_hit": False}
    assert parser is not None
    if parser.citation_doi.casefold() != source_doi.casefold():
        raise ValueError("pmc_repository_html_doi_mismatch")
    return _materialize_parsed_pmc_html(
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


def _parse_pmc_html(html_bytes: bytes) -> "_PmcArticleParser":
    parser = _PmcArticleParser()
    try:
        parser.feed(html_bytes.decode("utf-8", errors="strict"))
        parser.close()
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("pmc_repository_html_parse_failed") from exc
    return parser


def _materialize_parsed_pmc_html(
    *,
    parser: "_PmcArticleParser",
    html_bytes: bytes,
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
    source_ref: str,
    source_doi: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
) -> dict[str, Any]:
    html_sha = hashlib.sha256(html_bytes).hexdigest()
    cache_path = fulltext_cache_dir / f"fulltext-{html_sha[:16]}.html"
    _write_bytes_once(cache_path, html_bytes)
    materialized = source_dir / "materialized-fulltext"
    materialized.mkdir(parents=True, exist_ok=True)
    html_path = materialized / f"fulltext-{html_sha[:16]}.html"
    _write_bytes_once(html_path, html_bytes)
    procedures = _html_procedure_inventory(
        parser.sections,
        target_terms=[
            str(request.get("target_name") or ""),
            *[str(value) for value in request_queries(request)],
        ],
        source_artifact_sha256=html_sha,
        limit=config.max_fulltext_sections,
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
        "acquisition_method": "pmc_repository_fulltext_html",
        "acquisition_receipt": {
            **receipt,
            "cached_fulltext_path": str(cache_path),
        },
        "semantics": {
            "html_used_after_xml_before_pdf": True,
            "repository_access_is_distinct_from_open_access_licence": True,
            "source_material_grants_no_exact_reaction_authority": True,
        },
    }


class _PmcArticleParser(HTMLParser):
    """Small dependency-free parser for PMC's semantic article HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.citation_doi = ""
        self.pmcid = ""
        self.sections: list[tuple[str, str]] = []
        self._ignored_depth = 0
        self._capture = ""
        self._parts: list[str] = []
        self._title = ""

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lowered = tag.casefold()
        attributes = {str(key).casefold(): str(value or "") for key, value in attrs}
        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return
        if lowered == "meta" and attributes.get("name", "").casefold() in {
            "citation_doi",
            "dc.identifier",
        }:
            value = attributes.get("content", "").strip()
            if value.casefold().startswith("doi:"):
                value = value[4:].strip()
            if value.startswith("10."):
                self.citation_doi = value
        if lowered == "meta" and attributes.get("name", "").casefold() == (
            "citation_pmcid"
        ):
            self.pmcid = attributes.get("content", "").strip().upper()
        if self._ignored_depth:
            return
        if lowered in {"h2", "h3", "h4", "p"}:
            self._capture = lowered
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth or lowered != self._capture:
            return
        text = " ".join("".join(self._parts).split())
        if lowered in {"h2", "h3", "h4"}:
            self._title = text[:1_000]
        elif lowered == "p" and len(text) >= 40:
            self.sections.append((self._title, text[:8_000]))
        self._capture = ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture and not self._ignored_depth:
            self._parts.append(data)


def _html_procedure_inventory(
    sections: Iterable[tuple[str, str]],
    *,
    target_terms: Iterable[str],
    source_artifact_sha256: str,
    limit: int,
) -> list[dict[str, Any]]:
    terms = [
        " ".join(str(value).casefold().split())
        for value in target_terms
        if len(" ".join(str(value).split())) >= 3
    ][:64]
    title_signals = (
        "experimental",
        "materials and methods",
        "synthesis",
        "preparation",
        "production",
        "biotransformation",
    )
    process_signals = (
        "was added",
        "was stirred",
        "reaction mixture",
        "yield",
        "purified",
        "incubated",
        "catalyzed",
        "conversion",
    )
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, (title, body) in enumerate(sections, start=1):
        normalized = f"{title} {body}".casefold()
        score = 40 * sum(signal in title.casefold() for signal in title_signals)
        score += 20 * sum(term in normalized for term in terms)
        score += 4 * sum(signal in normalized for signal in process_signals)
        if score < 8:
            continue
        ranked.append(
            (
                -score,
                index,
                {
                    "label": f"html-section-{index}",
                    "name": title or f"PMC full-text paragraph {index}",
                    "visual_expected": False,
                    "page_number": index,
                    "procedure_excerpt": body[:4_000],
                    "source_artifact_kind": "pmc_fulltext_html",
                    "source_artifact_sha256": source_artifact_sha256,
                },
            )
        )
    return [row for _score, _index, row in sorted(ranked)[:limit]]


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(content).digest():
            raise ValueError("literature_content_address_collision")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


__all__ = ["materialize_pmc_repository_html"]
