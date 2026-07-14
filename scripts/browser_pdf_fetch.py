#!/usr/bin/env python3
"""Fetch queued PDFs through a visible Chrome/Edge browser session."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.local_pdf_proxy import (  # noqa: E402
    filter_pdf_requests,
    load_pdf_requests,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_pdfs_dir,
    local_pdf_proxy_request_queue_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--debug-port", type=int, default=9223)
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--source-ref", action="append", default=[])
    parser.add_argument("--title-contains", action="append", default=[])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--timeout-ms", type=int, default=90000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    queue_path = local_pdf_proxy_request_queue_path(output_dir)
    manifest_path = local_pdf_proxy_download_manifest_path(output_dir)
    pdf_dir = local_pdf_proxy_pdfs_dir(output_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    requests = filter_pdf_requests(
        load_pdf_requests(queue_path),
        case_ids=tuple(args.case_id),
        source_refs=tuple(args.source_ref),
        title_terms=tuple(args.title_contains),
    )
    if args.max_items is not None:
        requests = requests[: max(0, int(args.max_items))]
    if not requests:
        print(json.dumps({"accepted": True, "downloaded_count": 0, "reason": "queue_empty"}, ensure_ascii=False))
        return 0

    chrome = args.chrome_path or _find_browser()
    if not chrome:
        raise SystemExit("Chrome/Edge executable not found")
    _ensure_browser(chrome, debug_port=args.debug_port, profile_dir=args.profile_dir, headless=bool(args.headless))

    from playwright.sync_api import sync_playwright  # noqa: WPS433

    existing = _existing_successes(manifest_path)
    written: list[dict[str, Any]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{int(args.debug_port)}")
        context = browser.contexts[0] if browser.contexts else browser.new_context(accept_downloads=True)
        for request in requests:
            request_id = str(request.get("request_id") or "")
            if request_id in existing and not args.overwrite:
                continue
            # DOI redirects and publisher viewers can keep navigating after
            # ``domcontentloaded``.  Reusing that page lets one slow redirect
            # interrupt the next source, so every queued item owns a fresh tab.
            page = context.new_page()
            try:
                result = _fetch_one(
                    page,
                    request,
                    pdf_dir=pdf_dir,
                    timeout_ms=int(args.timeout_ms),
                )
            finally:
                page.close()
            written.append(result)
            _append_manifest(manifest_path, result)
        browser.close()

    downloaded = sum(1 for row in written if row.get("status") == "downloaded")
    failed = sum(1 for row in written if row.get("status") != "downloaded")
    print(
        json.dumps(
            {
                "schema_version": "browser_pdf_fetch_result.v1",
                "accepted": failed == 0,
                "processed_count": len(written),
                "downloaded_count": downloaded,
                "failed_count": failed,
                "manifest_path": str(manifest_path),
                "pdf_dir": str(pdf_dir),
                "results": [
                    {
                        "request_id": row.get("request_id"),
                        "status": row.get("status"),
                        "pdf_path": row.get("pdf_path", ""),
                        "reason": row.get("reason", ""),
                        "final_url": row.get("final_url", ""),
                    }
                    for row in written
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


def _find_browser() -> str:
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LocalAppData", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return ""


def _ensure_browser(chrome: str, *, debug_port: int, profile_dir: str, headless: bool) -> None:
    if _cdp_ready(debug_port):
        return
    profile = Path(profile_dir).expanduser().resolve() if profile_dir else ROOT / "results" / "shared" / ".browser_pdf_profile"
    profile.mkdir(parents=True, exist_ok=True)
    args = [
        chrome,
        f"--remote-debugging-port={int(debug_port)}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-proxy-server",
        "--proxy-server=direct://",
        "--proxy-bypass-list=*",
    ]
    if headless:
        args.append("--headless=new")
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if _cdp_ready(debug_port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"browser_cdp_not_ready:{debug_port}")


def _cdp_ready(debug_port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{int(debug_port)}/json/version", timeout=1.0) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _fetch_one(page: Any, request: dict[str, Any], *, pdf_dir: Path, timeout_ms: int) -> dict[str, Any]:
    url = str(request.get("url") or "").strip()
    doi = str(request.get("doi") or "").strip()
    if not url and doi:
        url = f"https://doi.org/{doi}"
    request_id = str(request.get("request_id") or _safe_id(doi or url or "request"))
    started = _now()
    candidates = _candidate_pdf_urls(url, doi)
    errors: list[str] = []
    landing_url = ""
    if url:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
            except Exception:
                pass
            page.wait_for_timeout(2000)
            landing_url = str(page.url or "")
            landing_text = _page_body_text(page)
            if _looks_like_robot_challenge(landing_text):
                page.wait_for_timeout(min(timeout_ms, 30000))
                landing_url = str(page.url or "")
                landing_text = _page_body_text(page)
            landing_access_block = _landing_access_block_reason(landing_text)
            if landing_access_block:
                return _manifest_row(
                    request,
                    status="needs_manual_access",
                    started_at_utc=started,
                    final_url=landing_url,
                    reason=landing_access_block,
                )
            landing_pii = _pii_from_url(landing_url)
            if landing_pii:
                candidates = _dedupe(
                    [
                        f"https://www.sciencedirect.com/science/article/pii/{landing_pii}/pdfft?download=true",
                        f"https://www.sciencedirect.com/science/article/pii/{landing_pii}/pdfft?isDTMRedir=true&download=true",
                        f"https://www.sciencedirect.com/science/article/pii/{landing_pii}/pdfft?crasolve=1&download=true",
                        f"https://www.sciencedirect.com/science/article/pii/{landing_pii}/pdfft",
                        f"https://www.sciencedirect.com/science/article/pii/{landing_pii}/pdf",
                        *candidates,
                    ]
                )
            page_pdf_links = _pdf_links_from_page(page, landing_url or url)
            if page_pdf_links:
                candidates = _dedupe([*page_pdf_links, *candidates])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: landing_warm_failed {type(exc).__name__}: {str(exc)[:300]}")
    for candidate_url in candidates:
        try:
            nav_response = page.goto(candidate_url, wait_until="domcontentloaded", timeout=timeout_ms)
            if nav_response is not None:
                content_type = str(nav_response.headers.get("content-type") or "").lower()
                body = nav_response.body()
                if _is_pdf_bytes(body):
                    filename = _safe_id(request_id) + ".pdf"
                    path = pdf_dir / filename
                    path.write_bytes(body)
                    return _manifest_row(
                        request,
                        status="downloaded",
                        started_at_utc=started,
                        pdf_path=str(path.resolve()),
                        final_url=str(nav_response.url or candidate_url),
                        content_type=content_type,
                        byte_count=len(body),
                    )
            result = page.evaluate(
                """async ({url}) => {
                    const response = await fetch(url, { credentials: "include", cache: "no-store" });
                    const buffer = await response.arrayBuffer();
                    const bytes = Array.from(new Uint8Array(buffer));
                    return {
                        ok: response.ok,
                        status: response.status,
                        finalUrl: response.url,
                        contentType: response.headers.get("content-type") || "",
                        bytes
                    };
                }""",
                {"url": candidate_url},
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate_url}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        data = bytes(int(value) & 0xFF for value in result.get("bytes") or [])
        content_type = str(result.get("contentType") or "").lower()
        if _is_pdf_bytes(data):
            filename = _safe_id(request_id) + ".pdf"
            path = pdf_dir / filename
            path.write_bytes(data)
            return _manifest_row(
                request,
                status="downloaded",
                started_at_utc=started,
                pdf_path=str(path.resolve()),
                final_url=str(result.get("finalUrl") or candidate_url),
                content_type=content_type,
                byte_count=len(data),
            )
        errors.append(
            f"{candidate_url}: not_pdf status={result.get('status')} content_type={content_type} bytes={len(data)}"
        )
    return _manifest_row(
        request,
        status="failed",
        started_at_utc=started,
        reason="; ".join(errors)[:2000] or "no_candidate_pdf_url_succeeded",
    )


def _page_body_text(page: Any) -> str:
    try:
        return str(page.locator("body").inner_text(timeout=5000) or "")
    except Exception:  # noqa: BLE001
        return ""


def _pdf_links_from_page(page: Any, base_url: str) -> list[str]:
    try:
        raw_links = page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href]')).map((node) => {
                const label = [
                    node.textContent || '',
                    node.getAttribute('aria-label') || '',
                    node.getAttribute('title') || '',
                    node.getAttribute('href') || ''
                ].join(' ');
                return { href: node.getAttribute('href') || '', label };
            })"""
        )
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for row in raw_links or []:
        if not isinstance(row, dict):
            continue
        href = str(row.get("href") or "").strip()
        label = str(row.get("label") or "").lower()
        if not href:
            continue
        if ".pdf" not in href.lower() and "pdf" not in label:
            continue
        out.append(urljoin(base_url or "", href))
    return _dedupe(out)


def _looks_like_robot_challenge(text: str) -> bool:
    lowered = str(text or "").lower()
    return "are you a robot" in lowered or "captcha challenge" in lowered


def _landing_access_block_reason(text: str) -> str:
    lowered = str(text or "").lower()
    if "does not subscribe to this content" in lowered or "not subscribe" in lowered:
        return "institution_does_not_subscribe"
    if _looks_like_robot_challenge(text):
        return "publisher_robot_challenge"
    if "access through your institution" in lowered or "purchase access" in lowered:
        return "publisher_access_required"
    return ""


def _candidate_pdf_urls(url: str, doi: str) -> list[str]:
    urls: list[str] = []
    doi_lower = doi.lower()
    if "10.1021/" in doi_lower:
        urls.extend(
            [
                f"https://pubs.acs.org/doi/pdf/{doi}",
                f"https://pubs.acs.org/doi/epdf/{doi}",
                f"https://pubs.acs.org/doi/suppl/{doi}/suppl_file/",
            ]
        )
    pii = _pii_from_doi(doi)
    if pii:
        urls.extend(
            [
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?download=true",
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true",
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?crasolve=1&download=true",
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft",
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdf",
            ]
        )
    if url:
        urls.append(url)
    if doi:
        urls.append(f"https://doi.org/{doi}")
    return _dedupe(urls)


def _is_pdf_bytes(data: bytes) -> bool:
    return bytes(data or b"")[:1024].lstrip().startswith(b"%PDF-")


def _pii_from_doi(doi: str) -> str:
    text = str(doi or "")
    match = re.search(r"10\.1016/([sS][0-9][a-z0-9()_.-]+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    raw = match.group(1)
    return re.sub(r"[^a-zA-Z0-9]", "", raw).upper()


def _pii_from_url(url: str) -> str:
    match = re.search(r"/pii/([A-Za-z0-9]+)", str(url or ""))
    return match.group(1).upper() if match else ""


def _manifest_row(
    request: dict[str, Any],
    *,
    status: str,
    started_at_utc: str,
    pdf_path: str = "",
    final_url: str = "",
    content_type: str = "",
    byte_count: int = 0,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "local_pdf_proxy_result.v1",
        "request_id": str(request.get("request_id") or ""),
        "case_id": str(request.get("case_id") or ""),
        "source_ref": str(request.get("source_ref") or ""),
        "doi": str(request.get("doi") or ""),
        "url": str(request.get("url") or ""),
        "title": str(request.get("title") or ""),
        "status": status,
        "accepted": status == "downloaded",
        "pdf_path": pdf_path,
        "final_url": final_url,
        "content_type": content_type,
        "byte_count": int(byte_count),
        "reason": reason,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _now(),
        "fetch_mode": "visible_browser_cdp_fetch",
    }


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _existing_successes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "downloaded" and row.get("request_id"):
            out.add(str(row["request_id"]))
    return out


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return safe[:120] or "pdf"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
