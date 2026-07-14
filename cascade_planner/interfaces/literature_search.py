"""Bounded metadata search and byte acquisition for literature evidence."""
from __future__ import annotations

import hashlib
import html
from io import BytesIO
import json
import re
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, urljoin, urlsplit
import zipfile

import requests


BytesFetcher = Callable[[str, float, int], bytes]


def crossref_search(query: str, limit: int) -> Iterable[Mapping[str, Any]]:
    response = requests.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": max(1, min(limit, 20))},
        headers={"User-Agent": "AutoPlanner/1.0 literature-evidence"},
        timeout=20,
    )
    response.raise_for_status()
    if len(response.content) > 4_000_000:
        raise ValueError("crossref_response_too_large")
    items = dict(response.json().get("message") or {}).get("items") or []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        links = [row for row in item.get("link") or [] if isinstance(row, Mapping)]
        pdf = next(
            (
                str(row.get("URL") or "")
                for row in links
                if "pdf" in str(row.get("content-type") or "").lower()
            ),
            "",
        )
        titles = item.get("title") or []
        yield {
            "doi": str(item.get("DOI") or ""),
            "title": str(titles[0] if titles else ""),
            "pdf_url": pdf,
        }


def citation_pdf_url(
    doi: str,
    *,
    timeout_s: float,
    fetch: BytesFetcher,
) -> str:
    content = fetch(f"https://doi.org/{quote(doi, safe='/')}", timeout_s, 2_000_000)
    text = content.decode("utf-8", errors="ignore")
    match = re.search(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
        text,
        flags=re.IGNORECASE,
    ) or re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        text,
        flags=re.IGNORECASE,
    )
    return html.unescape(urljoin(f"https://doi.org/{doi}", match.group(1))) if match else ""


def europe_pmc_open_access_pdf(
    doi: str,
    *,
    timeout_s: float,
    max_bytes: int,
    fetch: BytesFetcher,
) -> tuple[bytes, dict[str, Any]]:
    """Resolve an exact DOI to a bounded OA PDF/SI through Europe PMC.

    Europe PMC's supplementary archive is useful when publisher pages are
    protected by bot challenges: it commonly contains the supporting PDF and
    scheme assets under a permissive open-access record.  The archive is read
    in memory with explicit entry and decompression limits.
    """

    normalized, record, search_url, search_bytes = _europe_pmc_open_access_record(
        doi,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        fetch=fetch,
    )
    pmcid = str(record["pmcid"]).strip().upper()
    archive_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        f"{quote(pmcid, safe='')}/supplementaryFiles"
    )
    archive = fetch(archive_url, timeout_s, max_bytes)
    pdf, member = _pdf_from_bounded_zip(archive, max_bytes=max_bytes)
    return pdf, {
        "provider": "europe_pmc",
        "pmcid": pmcid,
        "doi": normalized,
        "search_url": search_url,
        "archive_url": archive_url,
        "search_sha256": hashlib.sha256(search_bytes).hexdigest(),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_member": member,
        "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        "license": str(record.get("license") or ""),
        "open_access": True,
    }


def europe_pmc_open_access_fulltext(
    doi: str,
    *,
    timeout_s: float,
    max_bytes: int,
    fetch: BytesFetcher,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Resolve exact DOI to structured OA XML and its original figure archive."""

    normalized, record, search_url, search_bytes = _europe_pmc_open_access_record(
        doi,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        fetch=fetch,
    )
    pmcid = str(record["pmcid"]).strip().upper()
    fulltext_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        f"{quote(pmcid, safe='')}/fullTextXML"
    )
    xml = fetch(fulltext_url, timeout_s, min(max_bytes, 12_000_000))
    if len(xml) < 100 or not xml.lstrip().startswith(b"<"):
        raise ValueError("europe_pmc_fulltext_xml_invalid")
    archive_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        f"{quote(pmcid, safe='')}/supplementaryFiles"
    )
    archive = b""
    archive_error = ""
    try:
        archive = fetch(archive_url, timeout_s, max_bytes)
        if not archive.startswith(b"PK"):
            archive_error = "europe_pmc_figure_archive_invalid"
            archive = b""
    except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
        archive_error = f"{type(exc).__name__}:{str(exc)[:300]}"
    return xml, archive, {
        "provider": "europe_pmc",
        "pmcid": pmcid,
        "doi": normalized,
        "search_url": search_url,
        "fulltext_url": fulltext_url,
        "archive_url": archive_url,
        "search_sha256": hashlib.sha256(search_bytes).hexdigest(),
        "fulltext_sha256": hashlib.sha256(xml).hexdigest(),
        "archive_sha256": hashlib.sha256(archive).hexdigest() if archive else "",
        "archive_error": archive_error,
        "license": str(record.get("license") or ""),
        "open_access": True,
    }


def _europe_pmc_open_access_record(
    doi: str,
    *,
    timeout_s: float,
    max_bytes: int,
    fetch: BytesFetcher,
) -> tuple[str, dict[str, Any], str, bytes]:
    normalized = str(doi or "").strip().lower()
    if not normalized:
        raise ValueError("europe_pmc_doi_missing")
    search_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI:%22{quote(normalized, safe='')}%22&format=json"
    )
    search_bytes = fetch(search_url, timeout_s, min(max_bytes, 2_000_000))
    try:
        payload = json.loads(search_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("europe_pmc_search_invalid") from exc
    records = [
        dict(row)
        for row in dict(payload.get("resultList") or {}).get("result") or []
        if isinstance(row, Mapping)
        and str(row.get("doi") or "").strip().lower() == normalized
        and str(row.get("pmcid") or "").strip()
        and str(row.get("isOpenAccess") or "").upper() == "Y"
    ]
    if not records:
        raise ValueError("europe_pmc_exact_open_access_record_missing")
    return normalized, records[0], search_url, search_bytes


def _pdf_from_bounded_zip(
    content: bytes,
    *,
    max_bytes: int,
    depth: int = 0,
) -> tuple[bytes, str]:
    if depth > 1 or len(content) > max_bytes:
        raise ValueError("europe_pmc_archive_limit_exceeded")
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("europe_pmc_archive_invalid") from exc
    infos = archive.infolist()
    if len(infos) > 256 or sum(max(0, row.file_size) for row in infos) > max_bytes * 2:
        raise ValueError("europe_pmc_archive_expansion_limit_exceeded")
    pdf_infos = sorted(
        (
            row
            for row in infos
            if not row.is_dir()
            and row.filename.lower().endswith(".pdf")
            and 0 < row.file_size <= max_bytes
        ),
        key=lambda row: (
            "supp" not in row.filename.lower(),
            -row.file_size,
            row.filename.lower(),
        ),
    )
    for row in pdf_infos:
        value = archive.read(row)
        if value.startswith(b"%PDF-") and len(value) <= max_bytes:
            return value, row.filename
    nested_infos = sorted(
        (
            row
            for row in infos
            if not row.is_dir()
            and row.filename.lower().endswith(".zip")
            and 0 < row.file_size <= max_bytes
        ),
        key=lambda row: (-row.file_size, row.filename.lower()),
    )
    for row in nested_infos[:16]:
        try:
            value, member = _pdf_from_bounded_zip(
                archive.read(row),
                max_bytes=max_bytes,
                depth=depth + 1,
            )
        except ValueError:
            continue
        return value, f"{row.filename}!/{member}"
    raise ValueError("europe_pmc_pdf_missing")


def fetch_bytes(url: str, timeout_s: float, max_bytes: int) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("literature_url_invalid")
    response = requests.get(
        url,
        headers={"User-Agent": "AutoPlanner/1.0 literature-evidence"},
        timeout=max(1.0, timeout_s),
        stream=True,
    )
    response.raise_for_status()
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("literature_response_too_large")
            chunks.append(bytes(chunk))
    finally:
        response.close()
    return b"".join(chunks)


__all__ = [
    "BytesFetcher",
    "citation_pdf_url",
    "crossref_search",
    "europe_pmc_open_access_fulltext",
    "europe_pmc_open_access_pdf",
    "fetch_bytes",
]
