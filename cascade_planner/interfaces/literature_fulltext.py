"""Europe PMC XML-first materialization and original-figure extraction."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import xml.etree.ElementTree as ET
import zipfile

from cascade_planner.interfaces.literature_candidates import doi, request_queries
from cascade_planner.interfaces.literature_search import (
    europe_pmc_open_access_fulltext,
)


BytesFetcher = Callable[[str, float, int], bytes]


def materialize_europe_pmc_fulltext(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_ref: str,
    source_dir: Path,
    fulltext_cache_dir: Path,
    config: Any,
    fetch: BytesFetcher,
) -> dict[str, Any]:
    source_doi = doi(candidate)
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = fulltext_cache_dir / "fulltext-receipt.json"
    receipt: dict[str, Any] = {}
    xml_bytes = b""
    archive = b""
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            xml_path = Path(str(receipt.get("cached_fulltext_path") or ""))
            archive_path = Path(str(receipt.get("cached_archive_path") or ""))
            if (
                str(receipt.get("doi") or "").casefold() == source_doi.casefold()
                and xml_path.is_file()
            ):
                xml_bytes = xml_path.read_bytes()
                if hashlib.sha256(xml_bytes).hexdigest() != str(
                    receipt.get("fulltext_sha256") or ""
                ):
                    xml_bytes = b""
                if archive_path.is_file():
                    archive = archive_path.read_bytes()
                    if hashlib.sha256(archive).hexdigest() != str(
                        receipt.get("archive_sha256") or ""
                    ):
                        archive = b""
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            receipt = {}
            xml_bytes = b""
            archive = b""
    if not xml_bytes:
        xml_bytes, archive, receipt = europe_pmc_open_access_fulltext(
            source_doi,
            timeout_s=config.timeout_s,
            max_bytes=max(config.max_pdf_bytes, config.max_fulltext_bytes),
            fetch=fetch,
        )
        cached_xml = fulltext_cache_dir / (
            f"fulltext-{hashlib.sha256(xml_bytes).hexdigest()[:16]}.xml"
        )
        _write_bytes_once(cached_xml, xml_bytes)
        cached_archive = fulltext_cache_dir / (
            f"figures-{hashlib.sha256(archive).hexdigest()[:16]}.zip"
        )
        if archive:
            _write_bytes_once(cached_archive, archive)
        receipt = {
            **receipt,
            "cached_fulltext_path": str(cached_xml),
            "cached_archive_path": str(cached_archive) if archive else "",
            "cache_hit": False,
        }
        _write_json_atomic(receipt_path, receipt)
    else:
        receipt = {**receipt, "cache_hit": True}
    if len(xml_bytes) > config.max_fulltext_bytes:
        raise ValueError("paper_fulltext_xml_too_large")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError("paper_fulltext_xml_parse_failed") from exc
    source_dois = {
        "".join(node.itertext()).strip().lower()
        for node in root.iter()
        if _xml_name(node.tag) == "article-id"
        and str(node.attrib.get("pub-id-type") or "").lower() == "doi"
    }
    if source_doi.lower() not in source_dois:
        raise ValueError("paper_fulltext_doi_mismatch")

    xml_sha = hashlib.sha256(xml_bytes).hexdigest()
    materialized = source_dir / "materialized-fulltext"
    materialized.mkdir(parents=True, exist_ok=True)
    xml_path = materialized / f"fulltext-{xml_sha[:16]}.xml"
    _write_bytes_once(xml_path, xml_bytes)
    target_terms = [
        str(request.get("target_name") or ""),
        *[str(value) for value in request_queries(request)],
    ]
    procedures = _fulltext_procedure_inventory(
        root,
        target_terms=target_terms,
        source_artifact_sha256=xml_sha,
        limit=config.max_fulltext_sections,
    )
    figures = _fulltext_visual_figures(
        root,
        archive=archive,
        output_dir=materialized / "figures",
        target_terms=target_terms,
        max_images=config.max_visual_pages,
    )
    if not procedures and not figures:
        raise ValueError("paper_fulltext_relevant_material_missing")
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": source_doi,
        "pmid": str(candidate.get("pmid") or ""),
        "pmcid": str(receipt.get("pmcid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "source_fulltext_sha256": xml_sha,
        "fulltext_xml_sha256": xml_sha,
        "fulltext_xml_path": str(xml_path),
        "source_pdf_sha256": "",
        "pdf_sha256": "",
        "page_count": 0,
        "visual_candidate_pages": figures,
        "procedure_inventory": procedures,
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": len(request.get("edges") or []) or 1,
        "focus_page_numbers": [int(row["page_number"]) for row in figures],
        "acquisition_status": "materialized",
        "acquisition_method": "europe_pmc_structured_fulltext_xml",
        "acquisition_receipt": receipt,
        "semantics": {
            "structured_fulltext_used_before_pdf": True,
            "original_figure_assets_used_without_pdf_rendering": True,
            "source_material_grants_no_exact_reaction_authority": True,
            "repository_access_is_distinct_from_open_access_licence": True,
        },
    }


def _fulltext_procedure_inventory(
    root: ET.Element,
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
        "reaction conditions",
        "biotransformation",
    )
    process_signals = (
        "was added",
        "was stirred",
        "was treated",
        "reaction mixture",
        "yield",
        "purified",
        "incubated",
        "catalyzed",
    )
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, section in enumerate(
        (node for node in root.iter() if _xml_name(node.tag) == "sec"), start=1
    ):
        title_node = next(
            (child for child in section if _xml_name(child.tag) == "title"), None
        )
        title = _xml_text(title_node) if title_node is not None else ""
        paragraphs = [
            _xml_text(node)
            for node in section.iter()
            if _xml_name(node.tag) == "p" and _xml_text(node)
        ]
        body = " ".join(paragraphs)
        normalized = f"{title} {body}".casefold()
        score = 40 * sum(signal in title.casefold() for signal in title_signals)
        score += 20 * sum(term in normalized for term in terms)
        score += 4 * sum(signal in normalized for signal in process_signals)
        if score < 8 or len(body) < 80:
            continue
        rows.append(
            (
                -score,
                index,
                {
                    "label": f"xml-sec-{index}",
                    "name": title or f"Structured full-text section {index}",
                    "visual_expected": False,
                    "page_number": index,
                    "procedure_excerpt": body[:4_000],
                    "source_artifact_kind": "europe_pmc_fulltext_xml",
                    "source_artifact_sha256": source_artifact_sha256,
                },
            )
        )
    return [row for _score, _index, row in sorted(rows)[:limit]]


def _fulltext_visual_figures(
    root: ET.Element,
    *,
    archive: bytes,
    output_dir: Path,
    target_terms: Iterable[str],
    max_images: int,
) -> list[dict[str, Any]]:
    if not archive:
        return []
    try:
        package = zipfile.ZipFile(BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ValueError("paper_fulltext_figure_archive_invalid") from exc
    infos = [row for row in package.infolist() if not row.is_dir()]
    if len(infos) > 512 or sum(max(0, row.file_size) for row in infos) > 80_000_000:
        raise ValueError("paper_fulltext_figure_archive_limit_exceeded")
    by_name = {
        Path(row.filename).name.casefold(): row
        for row in infos
        if 0 < row.file_size <= 12_000_000
        and Path(row.filename).suffix.casefold() in {".jpg", ".jpeg", ".png"}
    }
    terms = [
        " ".join(str(value).casefold().split())
        for value in target_terms
        if len(" ".join(str(value).split())) >= 3
    ][:64]
    signals = ("chemical structure", "reaction", "scheme", "synthesis", "pathway")
    ranked: list[tuple[int, int, str, str, str]] = []
    for index, figure in enumerate(
        (node for node in root.iter() if _xml_name(node.tag) == "fig"), start=1
    ):
        caption = " ".join(
            _xml_text(node)
            for node in figure.iter()
            if _xml_name(node.tag) in {"label", "caption"}
        ).strip()
        normalized = caption.casefold()
        score = 20 * sum(term in normalized for term in terms)
        score += 8 * sum(signal in normalized for signal in signals)
        graphics = [
            _xlink_href(node)
            for node in figure.iter()
            if _xml_name(node.tag) == "graphic" and _xlink_href(node)
        ]
        for href in graphics:
            info = by_name.get(Path(href).name.casefold())
            if info is not None and score > 0:
                ranked.append((-score, index, href, caption[:1_000], info.filename))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, index, href, caption, member_name in sorted(ranked):
        if member_name in seen:
            continue
        seen.add(member_name)
        content = package.read(member_name)
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(href).suffix.casefold()
        path = output_dir / f"figure-{index:03d}-{digest[:16]}{suffix}"
        _write_bytes_once(path, content)
        rows.append(
            {
                "page_number": index,
                "image_path": str(path),
                "image_sha256": digest,
                "figure_href": href,
                "caption": caption,
            }
        )
        if len(rows) >= max_images:
            break
    return rows


def _xml_name(tag: Any) -> str:
    return str(tag or "").rsplit("}", 1)[-1].casefold()


def _xml_text(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def _xlink_href(node: ET.Element) -> str:
    return str(
        node.attrib.get("{http://www.w3.org/1999/xlink}href")
        or node.attrib.get("href")
        or ""
    ).strip()


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(
            content
        ).hexdigest():
            raise ValueError("literature_content_address_collision")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "materialize_europe_pmc_fulltext",
]
