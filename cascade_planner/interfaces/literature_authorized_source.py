"""Hash-bound materialization for locally authorized publisher artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.interfaces.literature_candidates import doi, request_queries
from cascade_planner.interfaces.literature_html import materialize_parsed_html
from cascade_planner.interfaces.literature_html_parser import (
    html_procedure_inventory,
    parse_pmc_html,
)


def materialize_authorized_publisher_html(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_ref: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and freeze publisher HTML obtained by an authorized provider."""

    source_doi = doi(candidate)
    html_path = Path(str(artifact.get("html_path") or "")).expanduser().resolve()
    if not html_path.is_file():
        raise ValueError("authorized_publisher_html_missing")
    html_bytes = html_path.read_bytes()
    if not 200 <= len(html_bytes) <= config.max_fulltext_bytes:
        raise ValueError("authorized_publisher_html_size_invalid")
    html_sha = hashlib.sha256(html_bytes).hexdigest()
    expected_sha = str(artifact.get("html_sha256") or "").strip().lower()
    if expected_sha and expected_sha != html_sha:
        raise ValueError("authorized_publisher_html_hash_mismatch")
    parser = parse_pmc_html(html_bytes)
    parsed_doi = parser.citation_doi.casefold()
    if parsed_doi and parsed_doi != source_doi.casefold():
        raise ValueError("authorized_publisher_html_doi_mismatch")
    if source_doi and not parsed_doi:
        normalized_html = html_bytes.decode("utf-8", errors="ignore").casefold()
        if source_doi.casefold() not in normalized_html:
            raise ValueError("authorized_publisher_html_doi_missing")
    return materialize_parsed_html(
        parser=parser,
        html_bytes=html_bytes,
        receipt={
            "provider": str(artifact.get("provider") or "authorized_local_browser"),
            "doi": source_doi,
            "html_sha256": html_sha,
            "html_url": str(artifact.get("final_url") or artifact.get("url") or ""),
            "access_class": "institutionally_authorized_fulltext",
            "cache_hit": False,
        },
        candidate=candidate,
        request=request,
        source_ref=source_ref,
        source_doi=source_doi,
        source_dir=source_dir,
        fulltext_cache_dir=fulltext_cache_dir,
        config=config,
        acquisition_method="authorized_publisher_fulltext_html",
        artifact_kind="publisher_fulltext_html",
        repository_semantics=False,
    )


def materialize_authorized_publisher_json(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_ref: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify structured output produced by the legacy publisher spiders."""

    source_doi = doi(candidate)
    json_path = Path(str(artifact.get("structured_path") or "")).expanduser().resolve()
    if not json_path.is_file():
        raise ValueError("authorized_publisher_json_missing")
    content = json_path.read_bytes()
    if not 200 <= len(content) <= config.max_fulltext_bytes:
        raise ValueError("authorized_publisher_json_size_invalid")
    content_sha = hashlib.sha256(content).hexdigest()
    expected_sha = str(artifact.get("structured_sha256") or "").strip().lower()
    if expected_sha and expected_sha != content_sha:
        raise ValueError("authorized_publisher_json_hash_mismatch")
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError("authorized_publisher_json_parse_failed") from exc
    if not isinstance(document, Mapping):
        raise ValueError("authorized_publisher_json_document_invalid")
    metadata = document.get("metadata")
    metadata_doi = str(
        (metadata.get("doi") if isinstance(metadata, Mapping) else "")
        or document.get("doi")
        or ""
    ).strip()
    if metadata_doi and metadata_doi.casefold() != source_doi.casefold():
        raise ValueError("authorized_publisher_json_doi_mismatch")
    if source_doi and not metadata_doi and source_doi.casefold() not in content.decode(
        "utf-8", errors="ignore"
    ).casefold():
        raise ValueError("authorized_publisher_json_doi_missing")
    sections = _publisher_json_sections(document.get("full_text"), limit=256)
    procedures = html_procedure_inventory(
        sections,
        target_terms=[
            str(request.get("target_name") or ""),
            *[str(value) for value in request_queries(request)],
        ],
        source_artifact_sha256=content_sha,
        limit=config.max_fulltext_sections,
        source_artifact_kind="publisher_structured_json",
    )
    if not procedures:
        raise ValueError("authorized_publisher_json_relevant_material_missing")
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = fulltext_cache_dir / f"fulltext-{content_sha[:16]}.json"
    _write_bytes_once(cache_path, content)
    materialized = source_dir / "materialized-fulltext"
    materialized.mkdir(parents=True, exist_ok=True)
    frozen_path = materialized / f"fulltext-{content_sha[:16]}.json"
    _write_bytes_once(frozen_path, content)
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": source_doi,
        "pmid": str(candidate.get("pmid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "source_fulltext_sha256": content_sha,
        "fulltext_json_sha256": content_sha,
        "fulltext_json_path": str(frozen_path),
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
        "acquisition_method": "authorized_publisher_structured_json",
        "acquisition_receipt": {
            "provider": str(artifact.get("provider") or "legacy_publisher_spider"),
            "doi": source_doi,
            "structured_sha256": content_sha,
            "cached_fulltext_path": str(cache_path),
            "access_class": "institutionally_authorized_fulltext",
        },
        "semantics": {
            "structured_publisher_text_used_before_html_or_pdf": True,
            "institutionally_authorized_source": True,
            "source_material_grants_no_exact_reaction_authority": True,
        },
    }


def _publisher_json_sections(value: Any, *, limit: int) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []

    def visit(row: Any, inherited_title: str = "") -> None:
        if len(sections) >= limit:
            return
        if isinstance(row, str):
            text = " ".join(row.split())
            if len(text) >= 40:
                sections.append((inherited_title, text[:256_000]))
            return
        if isinstance(row, Mapping):
            title = " ".join(
                str(row.get("title") or row.get("heading") or inherited_title).split()
            )[:1_000]
            text = " ".join(str(row.get("text") or row.get("content") or "").split())
            if len(text) >= 40:
                sections.append((title, text[:256_000]))
            for key in ("subsections", "sections", "paragraphs"):
                visit(row.get(key), title)
            return
        if isinstance(row, list):
            for item in row:
                visit(item, inherited_title)
                if len(sections) >= limit:
                    break

    visit(value)
    return sections


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError("authorized_publisher_cached_artifact_digest_conflict")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


__all__ = [
    "materialize_authorized_publisher_html",
    "materialize_authorized_publisher_json",
]
