#!/usr/bin/env python3
"""Fetch queued PDFs through a visible Chrome/Edge browser session."""
from __future__ import annotations

import argparse
import hashlib
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
    local_pdf_proxy_work_dir,
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
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=180000,
        help=(
            "One source-wide browser deadline. The 180 s default allows first-use "
            "institutional authentication and dynamically rendered full text."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-publisher-provider", action="store_true")
    parser.add_argument("--publisher-provider-python", default="")
    parser.add_argument("--publisher-provider-timeout-s", type=int, default=300)
    parser.add_argument("--publisher-package-root", default="")
    parser.add_argument("--chrome-major", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    queue_path = local_pdf_proxy_request_queue_path(output_dir)
    manifest_path = local_pdf_proxy_download_manifest_path(output_dir)
    pdf_dir = local_pdf_proxy_pdfs_dir(output_dir)
    html_dir = local_pdf_proxy_work_dir(output_dir) / "html"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
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
    provider_python = "" if args.no_publisher_provider else (
        args.publisher_provider_python or _find_publisher_provider_python()
    )
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
                    html_dir=html_dir,
                    timeout_ms=int(args.timeout_ms),
                )
            finally:
                page.close()
            if result.get("status") != "downloaded" and provider_python:
                result = _publisher_provider_fallback(
                    request,
                    prior_result=result,
                    provider_python=provider_python,
                    provider_root=(
                        local_pdf_proxy_work_dir(output_dir) / "publisher-provider"
                    ),
                    manifest_root=output_dir,
                    manifest_path=manifest_path,
                    timeout_s=max(30, int(args.publisher_provider_timeout_s)),
                    package_root=str(args.publisher_package_root or ""),
                    chrome_major=max(0, int(args.chrome_major)),
                )
            written.append(result)
            if not result.pop("_manifest_already_written", False):
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
                        "html_path": row.get("html_path", ""),
                        "artifact_kind": row.get("artifact_kind", ""),
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


def _find_publisher_provider_python() -> str:
    configured = os.getenv("AUTOPLANNER_PUBLISHER_PROVIDER_PYTHON", "").strip()
    candidates = [
        configured,
        r"D:\conda\envs\py312\python.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), "")


def _publisher_provider_fallback(
    request: dict[str, Any],
    *,
    prior_result: dict[str, Any],
    provider_python: str,
    provider_root: Path,
    manifest_root: Path,
    manifest_path: Path,
    timeout_s: int,
    package_root: str,
    chrome_major: int,
) -> dict[str, Any]:
    doi = str(request.get("doi") or "").strip()
    if not doi:
        return prior_result
    request_id = str(request.get("request_id") or hashlib.sha256(doi.encode()).hexdigest()[:16])
    output_dir = provider_root / _safe_id(request_id)
    command = [
        provider_python,
        str(ROOT / "scripts" / "authorized_literature_fetch.py"),
        "--doi",
        doi,
        "--output-dir",
        str(output_dir),
        "--manifest-root",
        str(manifest_root),
        "--request-id",
        request_id,
        "--case-id",
        str(request.get("case_id") or ""),
        "--source-ref",
        str(request.get("source_ref") or f"doi:{doi}"),
        "--url",
        str(request.get("url") or f"https://doi.org/{doi}"),
        "--title",
        str(request.get("title") or ""),
    ]
    if package_root:
        command.extend(["--package-root", package_root])
    if chrome_major:
        command.extend(["--chrome-major", str(chrome_major)])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return {
            **prior_result,
            "reason": (
                f"{str(prior_result.get('reason') or '')}; "
                f"publisher_provider_timeout:{timeout_s}s"
            )[:2000],
        }
    matched = _manifest_result_for_request(manifest_path, request_id)
    if completed.returncode == 0 and matched:
        return {**matched, "_manifest_already_written": True}
    provider_reason = " ".join(
        (completed.stderr or completed.stdout or "publisher_provider_failed").split()
    )[:600]
    return {
        **prior_result,
        "reason": (
            f"{str(prior_result.get('reason') or '')}; "
            f"publisher_provider_failed:{completed.returncode}:{provider_reason}"
        )[:2000],
    }


def _manifest_result_for_request(path: Path, request_id: str) -> dict[str, Any]:
    matched: dict[str, Any] = {}
    if not path.is_file():
        return matched
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            str(row.get("request_id") or "") == request_id
            and row.get("accepted") is True
            and row.get("status") == "downloaded"
        ):
            matched = dict(row)
    return matched


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


def _fetch_one(
    page: Any,
    request: dict[str, Any],
    *,
    pdf_dir: Path,
    html_dir: Path,
    timeout_ms: int,
) -> dict[str, Any]:
    url = str(request.get("url") or "").strip()
    doi = str(request.get("doi") or "").strip()
    if not url and doi:
        url = f"https://doi.org/{doi}"
    started = _now()
    candidates = _candidate_pdf_urls(url, doi)
    errors: list[str] = []
    landing_url = ""
    deadline = time.monotonic() + max(5.0, timeout_ms / 1000.0)
    if url:
        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_remaining_timeout_ms(deadline),
            )
            if response is not None:
                body = response.body()
                if _is_pdf_bytes(body):
                    return _write_pdf_result(
                        request,
                        data=body,
                        pdf_dir=pdf_dir,
                        started_at_utc=started,
                        final_url=str(response.url or url),
                        content_type=str(response.headers.get("content-type") or ""),
                    )
            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=min(_remaining_timeout_ms(deadline), 10000),
                )
            except Exception:
                pass
            page.wait_for_timeout(min(_remaining_timeout_ms(deadline), 1500))
            landing_url = str(page.url or "")
            landing_text = _page_body_text(page)
            if _looks_like_robot_challenge(landing_text):
                page.wait_for_timeout(min(_remaining_timeout_ms(deadline), 15000))
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
            landing_html = _page_html_bytes(page)
            if _is_usable_article_html(landing_html, doi=doi):
                return _write_html_result(
                    request,
                    data=landing_html,
                    html_dir=html_dir,
                    started_at_utc=started,
                    final_url=landing_url or url,
                    content_type="text/html",
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
        if time.monotonic() >= deadline:
            errors.append("source_fetch_deadline_exhausted")
            break
        api_response = None
        try:
            # BrowserContext.request shares the persistent browser's cookies
            # but is not constrained by page JavaScript CORS.  The previous
            # page.evaluate(fetch(...)) path failed even when the browser was
            # institutionally authorized.
            api_response = page.context.request.get(
                candidate_url,
                timeout=_remaining_timeout_ms(deadline),
                fail_on_status_code=False,
            )
            data = api_response.body()
            content_type = str(api_response.headers.get("content-type") or "").lower()
            if _is_pdf_bytes(data):
                return _write_pdf_result(
                    request,
                    data=data,
                    pdf_dir=pdf_dir,
                    started_at_utc=started,
                    final_url=str(api_response.url or candidate_url),
                    content_type=content_type,
                )
            errors.append(
                f"{candidate_url}: context_request_not_pdf "
                f"status={api_response.status} content_type={content_type} bytes={len(data)}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate_url}: context_request {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            if api_response is not None:
                api_response.dispose()
        if time.monotonic() >= deadline:
            break
        try:
            nav_response = page.goto(
                candidate_url,
                wait_until="domcontentloaded",
                timeout=_remaining_timeout_ms(deadline),
            )
            if nav_response is not None:
                content_type = str(nav_response.headers.get("content-type") or "").lower()
                body = nav_response.body()
                if _is_pdf_bytes(body):
                    return _write_pdf_result(
                        request,
                        data=body,
                        pdf_dir=pdf_dir,
                        started_at_utc=started,
                        final_url=str(nav_response.url or candidate_url),
                        content_type=content_type,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate_url}: navigation {type(exc).__name__}: {str(exc)[:300]}")
            continue
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


def _page_html_bytes(page: Any) -> bytes:
    try:
        return str(page.content() or "").encode("utf-8")
    except Exception:  # noqa: BLE001
        return b""


def _is_usable_article_html(data: bytes, *, doi: str) -> bool:
    if len(data) < 2_000:
        return False
    text = data.decode("utf-8", errors="ignore").casefold()
    normalized_doi = str(doi or "").strip().casefold()
    if normalized_doi and normalized_doi not in text:
        return False
    fulltext_signals = (
        "experimental",
        "materials and methods",
        "general procedure",
        "reaction mixture",
        "supplementary information",
        "supporting information",
        "was stirred",
        "was added",
        "purified by",
    )
    return sum(signal in text for signal in fulltext_signals) >= 2


def _pdf_links_from_page(page: Any, base_url: str) -> list[str]:
    try:
        raw_links = page.evaluate(
            """() => {
              const anchors = Array.from(document.querySelectorAll('a[href]')).map((node) => {
                const label = [
                    node.textContent || '',
                    node.getAttribute('aria-label') || '',
                    node.getAttribute('title') || '',
                    node.getAttribute('href') || ''
                ].join(' ');
                return { href: node.getAttribute('href') || '', label };
              });
              const meta = Array.from(document.querySelectorAll('meta[content]'))
                .filter((node) => /pdf/i.test([
                  node.getAttribute('name') || '',
                  node.getAttribute('property') || '',
                  node.getAttribute('content') || ''
                ].join(' ')))
                .map((node) => ({
                  href: node.getAttribute('content') || '',
                  label: [node.getAttribute('name') || '', node.getAttribute('property') || ''].join(' ')
                }));
              return anchors.concat(meta);
            }"""
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


def _remaining_timeout_ms(deadline: float) -> int:
    return max(1000, int((deadline - time.monotonic()) * 1000))


def _write_pdf_result(
    request: dict[str, Any],
    *,
    data: bytes,
    pdf_dir: Path,
    started_at_utc: str,
    final_url: str,
    content_type: str,
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "request")
    path = pdf_dir / (_safe_id(request_id) + ".pdf")
    path.write_bytes(data)
    return _manifest_row(
        request,
        status="downloaded",
        started_at_utc=started_at_utc,
        pdf_path=str(path.resolve()),
        final_url=final_url,
        content_type=content_type,
        byte_count=len(data),
        artifact_kind="publisher_or_supplementary_pdf",
    )


def _write_html_result(
    request: dict[str, Any],
    *,
    data: bytes,
    html_dir: Path,
    started_at_utc: str,
    final_url: str,
    content_type: str,
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "request")
    digest = hashlib.sha256(data).hexdigest()
    path = html_dir / f"{_safe_id(request_id)}-{digest[:16]}.html"
    path.write_bytes(data)
    return _manifest_row(
        request,
        status="downloaded",
        started_at_utc=started_at_utc,
        html_path=str(path.resolve()),
        html_sha256=digest,
        final_url=final_url,
        content_type=content_type,
        byte_count=len(data),
        artifact_kind="publisher_fulltext_html",
    )


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
    if "10.1002/" in doi_lower:
        urls.extend(
            [
                f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
                f"https://onlinelibrary.wiley.com/doi/pdf/{doi}",
                f"https://onlinelibrary.wiley.com/doi/epdf/{doi}",
            ]
        )
    if "10.1007/" in doi_lower:
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    if "10.1038/" in doi_lower:
        suffix = doi.split("/", 1)[-1]
        urls.append(f"https://www.nature.com/articles/{suffix}.pdf")
    if "10.1039/" in doi_lower:
        suffix = doi.split("/", 1)[-1].casefold()
        if len(suffix) >= 4:
            year_code = suffix[:2]
            journal_code = suffix[2:4]
            urls.append(
                "https://www.rsc.org/suppdata/"
                f"{journal_code}/{year_code}/{suffix}/{suffix}1.pdf"
            )
    if "10.1080/" in doi_lower:
        urls.append(f"https://www.tandfonline.com/doi/pdf/{doi}?download=1")
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
    html_path: str = "",
    html_sha256: str = "",
    final_url: str = "",
    content_type: str = "",
    byte_count: int = 0,
    reason: str = "",
    artifact_kind: str = "",
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
        "html_path": html_path,
        "html_sha256": html_sha256,
        "artifact_kind": artifact_kind,
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
