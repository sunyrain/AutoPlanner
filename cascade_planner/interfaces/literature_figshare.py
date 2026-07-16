"""Identity-checked access to public ACS supplementary files on Figshare."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import requests

from cascade_planner.interfaces.literature_search import BytesFetcher


HttpRequester = Callable[..., Any]
_ACS_DOI = re.compile(r"^10\.1021/[a-z0-9._()/-]+$", re.IGNORECASE)
_SI_SUFFIX = re.compile(r"\.s\d{3,}$", re.IGNORECASE)
_DOWNLOAD_HOSTS = {"ndownloader.figshare.com", "api.figshare.com"}


def acs_figshare_supplementary_pdf(
    doi: str,
    *,
    timeout_s: float,
    max_bytes: int,
    fetch: BytesFetcher,
    requester: HttpRequester = requests.post,
) -> tuple[bytes, dict[str, Any]]:
    """Resolve an ACS article/SI DOI through Figshare's public metadata API."""

    normalized = _normalize_doi(doi)
    if not _ACS_DOI.fullmatch(normalized):
        raise ValueError("acs_figshare_doi_unsupported")
    identities = [normalized] if _SI_SUFFIX.search(normalized) else [f"{normalized}.s001"]
    identities.append(normalized)
    for identity in dict.fromkeys(identities):
        rows, search_bytes = _search(identity, timeout_s=timeout_s, requester=requester)
        for row in rows[:20]:
            article_id = str(row.get("id") or "").strip()
            if not article_id.isdigit():
                continue
            detail_url = f"https://api.figshare.com/v2/articles/{article_id}"
            detail_bytes = fetch(detail_url, timeout_s, min(max_bytes, 2_000_000))
            detail = _json_object(detail_bytes, "acs_figshare_article_invalid")
            resolved_doi = _normalize_doi(str(detail.get("doi") or ""))
            if resolved_doi != identity:
                continue
            file_row = _select_pdf(detail.get("files"), max_bytes=max_bytes)
            if not file_row:
                continue
            download_url = str(file_row.get("download_url") or "").strip()
            parsed = urlsplit(download_url)
            if parsed.scheme != "https" or parsed.hostname not in _DOWNLOAD_HOSTS:
                continue
            content = fetch(download_url, timeout_s, max_bytes)
            if not content.startswith(b"%PDF-") or len(content) > max_bytes:
                raise ValueError("acs_figshare_pdf_invalid")
            return content, {
                "provider": "acs_figshare",
                "doi": resolved_doi,
                "article_id": int(article_id),
                "file_id": int(file_row.get("id") or 0),
                "file_name": str(file_row.get("name") or ""),
                "detail_url": detail_url,
                "download_url": download_url,
                "search_sha256": hashlib.sha256(search_bytes).hexdigest(),
                "detail_sha256": hashlib.sha256(detail_bytes).hexdigest(),
                "pdf_sha256": hashlib.sha256(content).hexdigest(),
                "public_repository": True,
                "identity_checked": True,
            }
    raise ValueError("acs_figshare_supplementary_pdf_missing")


def _search(
    identity: str, *, timeout_s: float, requester: HttpRequester
) -> tuple[list[dict[str, Any]], bytes]:
    response = requester(
        "https://api.figshare.com/v2/articles/search",
        json={"search_for": identity, "limit": 20},
        headers={"User-Agent": "AutoPlanner/1.0 literature-evidence"},
        timeout=max(1.0, timeout_s),
    )
    response.raise_for_status()
    content = bytes(response.content)
    if len(content) > 2_000_000:
        raise ValueError("acs_figshare_search_response_too_large")
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, list):
        raise ValueError("acs_figshare_search_response_invalid")
    return [dict(row) for row in value if isinstance(row, Mapping)], content


def _select_pdf(value: Any, *, max_bytes: int) -> dict[str, Any]:
    rows = [dict(row) for row in value or [] if isinstance(row, Mapping)]
    rows = [
        row
        for row in rows
        if str(row.get("name") or "").casefold().endswith(".pdf")
        and 0 < int(row.get("size") or 0) <= max_bytes
    ]
    rows.sort(
        key=lambda row: (
            "_si_" not in str(row.get("name") or "").casefold(),
            "supp" not in str(row.get("name") or "").casefold(),
            int(row.get("size") or 0),
        )
    )
    return rows[0] if rows else {}


def _normalize_doi(value: str) -> str:
    return re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I
    ).casefold()


def _json_object(content: bytes, reason: str) -> dict[str, Any]:
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(reason)
    return dict(value)


__all__ = ["acs_figshare_supplementary_pdf"]
