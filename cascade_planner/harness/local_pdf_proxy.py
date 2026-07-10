"""Local-machine PDF proxy queue for institutionally authorized downloads.

The remote harness can write URL/DOI requests here, but the actual PDF fetch is
expected to run on the user's local machine under their own library/VPN access.
No username, password, cookies, or browser profiles are stored by this module.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


LOCAL_PDF_PROXY_MANIFEST_ENTRY_SCHEMA = "local_pdf_proxy_manifest_entry.v1"
LOCAL_PDF_PROXY_REQUEST_SCHEMA = "local_pdf_proxy_request.v1"
LOCAL_PDF_PROXY_RESULT_SCHEMA = "local_pdf_proxy_result.v1"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.7778.96 Safari/537.36"
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

FetchUrl = Callable[[str, dict[str, str], float, int], dict[str, Any]]


def local_pdf_proxy_work_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "local_pdf_proxy"


def local_pdf_proxy_request_queue_path(output_dir: str | Path) -> Path:
    return local_pdf_proxy_work_dir(output_dir) / "pdf_requests.jsonl"


def local_pdf_proxy_download_manifest_path(output_dir: str | Path) -> Path:
    return local_pdf_proxy_work_dir(output_dir) / "pdf_download_manifest.jsonl"


def local_pdf_proxy_pdfs_dir(output_dir: str | Path) -> Path:
    return local_pdf_proxy_work_dir(output_dir) / "pdfs"


def local_pdf_proxy_manifest_entry(
    payload: dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    work_dir = local_pdf_proxy_work_dir(output_dir)
    request_queue = local_pdf_proxy_request_queue_path(output_dir)
    download_manifest = local_pdf_proxy_download_manifest_path(output_dir)
    pdf_dir = local_pdf_proxy_pdfs_dir(output_dir)
    payload = dict(payload or {})
    request_count = int(payload.get("request_count") or _count_jsonl(request_queue))
    result_summary = payload.get("result_summary") or _result_summary(download_manifest)
    downloaded_count = int(result_summary.get("downloaded_count") or 0)
    if payload.get("status"):
        status = str(payload["status"])
    elif downloaded_count:
        status = "downloaded"
    elif request_count:
        status = "queued"
    else:
        status = "planned"
    return {
        "schema_version": LOCAL_PDF_PROXY_MANIFEST_ENTRY_SCHEMA,
        "status": status,
        "request_queue_path": str(request_queue.resolve()),
        "download_manifest_path": str(download_manifest.resolve()),
        "pdf_dir": str(pdf_dir.resolve()),
        "schema": LOCAL_PDF_PROXY_REQUEST_SCHEMA,
        "result_schema": LOCAL_PDF_PROXY_RESULT_SCHEMA,
        "request_count": request_count,
        "result_summary": result_summary,
        "source_policy": _source_policy(),
        "use_policy": (
            "Use this only after the open-research agent has attempted native "
            "web/source access and recorded that full text or PDF content is not "
            "available from the remote agent context. Then write DOI/URL requests "
            "to request_queue_path. The user's local proxy fetches authorized PDFs "
            "and returns pdf paths in download_manifest_path. Do not store or ask "
            "for institutional credentials, cookies, or browser profiles."
        ),
        "sync_hint": {
            "work_dir": str(work_dir.resolve()),
            "server_to_local": "sync request_queue_path to the local machine",
            "local_to_server": "sync pdf_dir and download_manifest_path back to the server",
        },
    }


def requests_from_source_material_locator_pack(
    pack: dict[str, Any] | str | Path,
    *,
    case_id: str = "",
    reason: str = "source_material_locator_followup",
    max_items: int = 20,
    priority_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = _load_json(pack) if isinstance(pack, (str, Path)) else dict(pack)
    target = dict(data.get("target") or {})
    resolved_case_id = case_id or str(target.get("case_id") or target.get("name") or "")
    terms = priority_terms or _priority_terms(" ".join([
        resolved_case_id,
        str(target.get("name") or ""),
        str(target.get("search_name") or ""),
        str(target.get("family_hint") or ""),
    ]))
    requests: list[dict[str, Any]] = []
    for record in data.get("material_records") or []:
        if not isinstance(record, dict):
            continue
        if len(requests) >= max(0, int(max_items)):
            break
        try:
            requests.append(
                build_pdf_request(
                    record,
                    case_id=resolved_case_id,
                    source_ref=str(record.get("record_id") or record.get("source_ref") or ""),
                    reason=reason,
                    requested_by="source_material_locator",
                )
            )
        except ValueError:
            continue
    return _sort_requests_by_priority(_dedupe_requests(requests), terms)


def build_pdf_request(
    record: dict[str, Any],
    *,
    case_id: str = "",
    source_ref: str = "",
    reason: str = "",
    requested_by: str = "harness",
) -> dict[str, Any]:
    doi = normalize_doi(str(record.get("doi") or record.get("DOI") or ""))
    url = _target_url(record, doi=doi)
    if not url and doi:
        url = f"https://doi.org/{doi}"
    if not url:
        raise ValueError("PDF proxy request requires a URL or DOI")
    title = str(record.get("title") or record.get("source_title") or "").strip()
    evidence_refs = [str(item) for item in record.get("evidence_refs") or [] if str(item).strip()]
    resolved_source_ref = source_ref or str(record.get("source_ref") or record.get("record_id") or "")
    request_id = str(record.get("request_id") or "").strip()
    if not request_id:
        request_id = _request_id(
            case_id=case_id,
            source_ref=resolved_source_ref,
            doi=doi or "",
            url=url,
            title=title,
        )
    return {
        "schema_version": LOCAL_PDF_PROXY_REQUEST_SCHEMA,
        "request_id": sanitize_id(request_id),
        "created_at_utc": _utc_now(),
        "case_id": str(case_id or record.get("case_id") or ""),
        "requested_by": requested_by,
        "reason": reason,
        "url": url,
        "doi": doi or "",
        "title": title,
        "source_ref": resolved_source_ref,
        "material_type": str(record.get("material_type") or record.get("source_type") or ""),
        "content_scope": str(record.get("content_scope") or record.get("requested_content_scope") or ""),
        "content_type_hint": str(record.get("content_type") or record.get("content-type") or ""),
        "evidence_refs": evidence_refs,
        "access_hint": str(record.get("access_hint") or "local_institutional_access"),
        "status": "queued",
        "source_policy": _request_source_policy(),
    }


def write_pdf_request_queue(
    requests: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = True,
    dedupe: bool = True,
) -> dict[str, Any]:
    queue_path = Path(path)
    _ensure_private_dir(queue_path.parent)
    rows = load_pdf_requests(queue_path) if append and queue_path.exists() else []
    rows.extend(_normalize_request(row) for row in requests)
    if dedupe:
        rows = _dedupe_requests(rows)
    queue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "schema_version": "local_pdf_proxy_queue_write.v1",
        "accepted": True,
        "path": str(queue_path),
        "request_count": len(rows),
        "source_policy": _source_policy(),
    }


def load_pdf_requests(path: str | Path) -> list[dict[str, Any]]:
    queue_path = Path(path)
    if not queue_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            rows.append(_normalize_request(json.loads(text)))
        except Exception as exc:
            raise ValueError(f"{queue_path}:{line_no}: invalid PDF proxy request: {exc}") from exc
    return rows


def download_pdf_requests(
    *,
    queue_path: str | Path,
    pdf_dir: str | Path,
    manifest_path: str | Path,
    timeout_s: float = 30.0,
    max_items: int | None = None,
    max_bytes: int = 80 * 1024 * 1024,
    overwrite: bool = False,
    delay_s: float = 0.0,
    fetch_url: FetchUrl | None = None,
) -> dict[str, Any]:
    requests = load_pdf_requests(queue_path)
    out_dir = Path(pdf_dir)
    manifest = Path(manifest_path)
    _ensure_private_dir(out_dir)
    _ensure_private_dir(manifest.parent)
    existing_downloads = _accepted_request_ids(manifest)
    fetch = fetch_url or _fetch_url
    written: list[dict[str, Any]] = []
    processed = 0
    for request in requests:
        if max_items is not None and processed >= max(0, int(max_items)):
            break
        request_id = str(request.get("request_id") or "")
        if request_id in existing_downloads and not overwrite:
            continue
        processed += 1
        result = _download_one_request(
            request,
            pdf_dir=out_dir,
            timeout_s=float(timeout_s),
            max_bytes=int(max_bytes),
            fetch_url=fetch,
        )
        _append_jsonl(manifest, result)
        written.append(result)
        if delay_s and processed < len(requests):
            time.sleep(max(0.0, float(delay_s)))
    return {
        "schema_version": "local_pdf_proxy_download_run.v1",
        "accepted": True,
        "queue_path": str(Path(queue_path)),
        "pdf_dir": str(out_dir),
        "manifest_path": str(manifest),
        "processed_count": processed,
        "downloaded_count": sum(1 for item in written if item.get("status") == "downloaded"),
        "needs_manual_access_count": sum(1 for item in written if item.get("status") == "needs_manual_access"),
        "failed_count": sum(1 for item in written if item.get("status") == "failed"),
        "results": written,
        "source_policy": _source_policy(),
    }


def summarize_pdf_download_manifest(path: str | Path) -> dict[str, Any]:
    return _result_summary(Path(path))


def normalize_doi(value: str) -> str:
    match = DOI_RE.search(str(value or "").strip())
    if not match:
        return ""
    return match.group(0).rstrip(".,);]").lower()


def sanitize_id(value: str, fallback: str = "pdf_request") -> str:
    text = unquote(str(value or ""))
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9._:-]+", "_", text)
    text = text.strip("._:-")
    return (text or fallback)[:140]


def sanitize_filename(value: str, fallback: str = "paper") -> str:
    text = unquote(str(value or ""))
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return (text or fallback)[:120]


def is_pdf_payload(headers: dict[str, str], body: bytes) -> bool:
    del headers
    return bytes(body or b"")[:1024].lstrip().startswith(b"%PDF-")


def _download_one_request(
    request: dict[str, Any],
    *,
    pdf_dir: Path,
    timeout_s: float,
    max_bytes: int,
    fetch_url: FetchUrl,
) -> dict[str, Any]:
    url = _target_url(request, doi=str(request.get("doi") or ""))
    request_id = str(request.get("request_id") or _request_id(url=url))
    base = {
        "schema_version": LOCAL_PDF_PROXY_RESULT_SCHEMA,
        "request_id": request_id,
        "case_id": str(request.get("case_id") or ""),
        "source_ref": str(request.get("source_ref") or ""),
        "doi": str(request.get("doi") or ""),
        "url": url,
        "title": str(request.get("title") or ""),
        "fetched_at_utc": _utc_now(),
        "source_policy": _result_source_policy(),
    }
    if not url:
        return {**base, "accepted": False, "status": "failed", "reason": "missing_url_or_doi"}
    headers = {
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
        "User-Agent": USER_AGENT,
    }
    response = fetch_url(url, headers, timeout_s, max_bytes)
    status = int(response.get("status") or 0)
    final_url = str(response.get("final_url") or url)
    response_headers = {str(k).lower(): str(v) for k, v in dict(response.get("headers") or {}).items()}
    body = response.get("body") or b""
    if isinstance(body, str):
        body = body.encode("utf-8")
    if response.get("error"):
        return {
            **base,
            "accepted": False,
            "status": "failed",
            "http_status": status,
            "final_url": final_url,
            "reason": str(response.get("error")),
        }
    if is_pdf_payload(response_headers, body):
        path = _unique_pdf_path(pdf_dir, _pdf_stem(request, final_url))
        path.write_bytes(body)
        return {
            **base,
            "accepted": True,
            "status": "downloaded",
            "http_status": status,
            "final_url": final_url,
            "content_type": response_headers.get("content-type", ""),
            "pdf_path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "reason": "",
        }
    return {
        **base,
        "accepted": False,
        "status": "needs_manual_access" if _looks_like_landing_or_login(response_headers, body) else "failed",
        "http_status": status,
        "final_url": final_url,
        "content_type": response_headers.get("content-type", ""),
        "reason": response.get("reason") or "response_not_pdf",
    }


def _fetch_url(url: str, headers: dict[str, str], timeout_s: float, max_bytes: int) -> dict[str, Any]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return {
                    "status": response.status,
                    "final_url": response.geturl(),
                    "headers": dict(response.headers.items()),
                    "body": b"",
                    "reason": "content_length_exceeds_max_bytes",
                }
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                return {
                    "status": response.status,
                    "final_url": response.geturl(),
                    "headers": dict(response.headers.items()),
                    "body": b"",
                    "reason": "response_exceeds_max_bytes",
                }
            return {
                "status": response.status,
                "final_url": response.geturl(),
                "headers": dict(response.headers.items()),
                "body": body,
            }
    except HTTPError as exc:
        return {
            "status": exc.code,
            "final_url": exc.geturl(),
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "body": b"",
            "error": f"HTTP {exc.code}",
        }
    except URLError as exc:
        return {"status": 0, "final_url": url, "headers": {}, "body": b"", "error": str(exc.reason)}
    except Exception as exc:
        return {"status": 0, "final_url": url, "headers": {}, "body": b"", "error": str(exc)}


def _normalize_request(row: dict[str, Any]) -> dict[str, Any]:
    request = dict(row)
    request.setdefault("schema_version", LOCAL_PDF_PROXY_REQUEST_SCHEMA)
    request["request_id"] = sanitize_id(str(request.get("request_id") or _request_id(url=str(request.get("url") or ""))))
    request["doi"] = normalize_doi(str(request.get("doi") or request.get("url") or ""))
    request["url"] = _target_url(request, doi=str(request.get("doi") or ""))
    request.setdefault("status", "queued")
    request.setdefault("source_policy", _request_source_policy())
    return request


def _target_url(record: dict[str, Any], *, doi: str = "") -> str:
    for key in ("url", "pdf_url", "landing_url", "source_url"):
        value = str(record.get(key) or "").strip()
        if URL_RE.search(value):
            return value
        normalized = normalize_doi(value)
        if normalized:
            return f"https://doi.org/{normalized}"
    normalized_doi = normalize_doi(doi or str(record.get("doi") or ""))
    return f"https://doi.org/{normalized_doi}" if normalized_doi else ""


def _request_id(
    *,
    case_id: str = "",
    source_ref: str = "",
    doi: str = "",
    url: str = "",
    title: str = "",
) -> str:
    material = "\n".join([case_id, source_ref, doi.lower(), url, title])
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
    prefix = sanitize_id(case_id or source_ref or doi.replace("/", "_") or "pdf")
    return f"pdfreq_{prefix}_{digest}"


def _pdf_stem(request: dict[str, Any], final_url: str) -> str:
    doi = str(request.get("doi") or "")
    if doi:
        return sanitize_filename(doi.replace("/", "_"))
    title = str(request.get("title") or "")
    if title:
        return sanitize_filename(title)
    parsed = urlparse(final_url)
    name = Path(parsed.path).name
    if name and name.lower() != "pdf":
        return sanitize_filename(name.removesuffix(".pdf"))
    return sanitize_filename(str(request.get("request_id") or "paper"))


def _unique_pdf_path(out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}.pdf"
    if not base.exists():
        return base
    for i in range(2, 10000):
        candidate = out_dir / f"{stem}_{i}.pdf"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many existing PDFs for {stem}")


def _looks_like_landing_or_login(headers: dict[str, str], body: bytes) -> bool:
    content_type = str(headers.get("content-type") or "").lower()
    sample = body[:4096].decode("utf-8", "ignore").lower()
    markers = ("login", "signin", "sign in", "shibboleth", "saml", "cas", "ezproxy", "institution")
    return "html" in content_type or any(marker in sample for marker in markers)


def _source_policy() -> dict[str, Any]:
    return {
        "local_authorized_access_required": True,
        "credentials_stored": False,
        "cookies_stored": False,
        "browser_profile_stored": False,
        "no_bulk_download": True,
        "not_route_evidence_until_structured_extraction": True,
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


def _request_source_policy() -> dict[str, Any]:
    return {
        "metadata_pointer_only": True,
        "full_text_content_stored": False,
        "credentials_stored": False,
        "not_route_evidence": True,
    }


def _result_source_policy() -> dict[str, Any]:
    return {
        "fetched_by_local_authorized_proxy": True,
        "credentials_stored": False,
        "cookies_stored": False,
        "manifest_stores_full_text": False,
        "not_route_evidence_until_structured_extraction": True,
        "production_write_blocked": True,
    }


def _dedupe_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        request_id = str(row.get("request_id") or "")
        key = request_id or str(row.get("url") or row.get("doi") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _priority_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.split(r"[^A-Za-z0-9]+", str(text or "").lower()):
        if len(token) < 5:
            continue
        if token in {"target", "fullflow", "synthesis", "intermediate"}:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:8]


def _sort_requests_by_priority(rows: list[dict[str, Any]], terms: list[str]) -> list[dict[str, Any]]:
    if not terms:
        return rows
    indexed = list(enumerate(rows))
    return [
        row
        for _, row in sorted(
            indexed,
            key=lambda item: (-_request_priority_score(item[1], terms), item[0]),
        )
    ]


def _request_priority_score(row: dict[str, Any], terms: list[str]) -> int:
    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in ("title", "url", "doi", "source_ref", "material_type")
    )
    score = 0
    for term in terms:
        if term and term in haystack:
            score += 10
    if "publisher_landing" in haystack:
        score += 2
    if "pdf" in haystack:
        score += 1
    return score


def _accepted_request_ids(manifest_path: Path) -> set[str]:
    out: set[str] = set()
    for row in _iter_jsonl(manifest_path):
        if row.get("accepted") and row.get("status") == "downloaded":
            out.add(str(row.get("request_id") or ""))
    return out


def _result_summary(path: Path) -> dict[str, Any]:
    rows = list(_iter_jsonl(path))
    return {
        "result_count": len(rows),
        "downloaded_count": sum(1 for row in rows if row.get("status") == "downloaded"),
        "needs_manual_access_count": sum(1 for row in rows if row.get("status") == "needs_manual_access"),
        "failed_count": sum(1 for row in rows if row.get("status") == "failed"),
    }


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(path))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
