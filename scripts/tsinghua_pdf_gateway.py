#!/usr/bin/env python3
"""Use a persistent browser session for Tsinghua Library PDF access.

This helper is intentionally conservative:

* it never asks for or stores a username/password;
* it stores only a local Chromium profile under results/shared by default;
* it downloads one requested paper at a time through the saved browser session.

The practical flow is:

    python scripts/tsinghua_pdf_gateway.py check
    python scripts/tsinghua_pdf_gateway.py doctor
    python -m pip install playwright
    python -m playwright install chromium
    python scripts/tsinghua_pdf_gateway.py login
    python scripts/tsinghua_pdf_gateway.py download --doi 10.1021/...

For Tsinghua resources, the most reliable input is usually the URL reached from
the library database navigation or a database detail page "access" link. Bare
DOI URLs are supported, but they may not trigger the smart gateway by themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = ROOT / "results" / "shared" / "tsinghua_pdf_gateway"
DEFAULT_PROFILE_DIR = DEFAULT_WORK_DIR / "browser-profile"
DEFAULT_DOWNLOAD_DIR = DEFAULT_WORK_DIR / "pdfs"
DEFAULT_MANIFEST = DEFAULT_WORK_DIR / "download_manifest.jsonl"
DEFAULT_DOCTOR_PROFILE_DIR = DEFAULT_WORK_DIR / "doctor-profile"

TSINGHUA_DATABASE_NAV_URL = "https://ecollection.lib.tsinghua.edu.cn/databasenav/"
TSINGHUA_EPROXY_HOME_URL = "https://eproxy.lib.tsinghua.edu.cn/reader/home"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.7778.96 Safari/537.36"
)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class Target:
    source: str
    url: str
    doi: str | None = None
    title: str | None = None
    record_id: str | None = None


@dataclass(frozen=True)
class PdfCandidate:
    url: str
    text: str = ""
    source: str = ""
    score: int = 0


@dataclass(frozen=True)
class DownloadOutcome:
    ok: bool
    target: Target
    final_url: str | None = None
    pdf_path: str | None = None
    status: int | None = None
    reason: str | None = None
    candidates: list[str] | None = None


def require_playwright() -> Any:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        print(
            "Missing optional dependency: playwright\n\n"
            "Install it with:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium\n\n"
            "If Chromium starts but reports missing Linux libraries, run:\n"
            "  python -m playwright install-deps chromium\n",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return sync_playwright, PlaywrightError, PlaywrightTimeoutError


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def normalize_doi(value: str) -> str | None:
    match = DOI_RE.search(value.strip())
    if not match:
        return None
    return match.group(0).rstrip(".,);]")


def target_from_text(text: str) -> Target:
    value = text.strip()
    if not value:
        raise ValueError("empty target line")
    if URL_RE.search(value):
        doi = normalize_doi(value)
        return Target(source=value, url=value, doi=doi)
    doi = normalize_doi(value)
    if doi:
        return Target(source=value, url=f"https://doi.org/{doi}", doi=doi)
    raise ValueError(f"cannot recognize target as URL or DOI: {value!r}")


def target_from_json(record: dict[str, Any]) -> Target:
    url = (
        record.get("url")
        or record.get("pdf_url")
        or record.get("landing_url")
        or record.get("source_url")
    )
    doi = record.get("doi")
    title = record.get("title")
    record_id = record.get("id") or record.get("record_id") or record.get("paper_id")
    if url:
        normalized_doi = str(doi).strip() if doi else normalize_doi(str(url))
        return Target(
            source=json.dumps(record, ensure_ascii=False),
            url=str(url),
            doi=normalized_doi or None,
            title=str(title) if title else None,
            record_id=str(record_id) if record_id else None,
        )
    if doi:
        doi_str = normalize_doi(str(doi)) or str(doi).strip()
        return Target(
            source=json.dumps(record, ensure_ascii=False),
            url=f"https://doi.org/{doi_str}",
            doi=doi_str,
            title=str(title) if title else None,
            record_id=str(record_id) if record_id else None,
        )
    raise ValueError("JSON record must include url, pdf_url, landing_url, source_url, or doi")


def load_targets(args: argparse.Namespace) -> list[Target]:
    targets: list[Target] = []
    for url in args.url or []:
        targets.append(target_from_text(url))
    for doi in args.doi or []:
        doi_str = normalize_doi(doi) or doi.strip()
        targets.append(Target(source=doi, url=f"https://doi.org/{doi_str}", doi=doi_str))
    if args.input:
        input_path = Path(args.input)
        for line_no, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                if line.startswith("{"):
                    targets.append(target_from_json(json.loads(line)))
                else:
                    targets.append(target_from_text(line))
            except Exception as exc:
                raise SystemExit(f"{input_path}:{line_no}: {exc}") from exc
    if not targets:
        raise SystemExit("Provide at least one --url, --doi, or --input target.")
    return targets


def sanitize_stem(value: str, fallback: str = "paper") -> str:
    value = unquote(value)
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return (value or fallback)[:120]


def stem_for_target(target: Target, url: str | None = None) -> str:
    if target.record_id:
        return sanitize_stem(target.record_id)
    if target.doi:
        return sanitize_stem(target.doi.replace("/", "_"))
    if target.title:
        return sanitize_stem(target.title)
    parsed = urlparse(url or target.url)
    basename = Path(parsed.path).name
    if basename and basename.lower() != "pdf":
        return sanitize_stem(basename.removesuffix(".pdf"))
    digest = hashlib.sha1(target.url.encode("utf-8")).hexdigest()[:12]
    return f"paper_{digest}"


def unique_pdf_path(out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}.pdf"
    if not base.exists():
        return base
    for i in range(2, 1000):
        candidate = out_dir / f"{stem}_{i}.pdf"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many existing files for {stem}")


def is_pdf_payload(headers: dict[str, str], body: bytes) -> bool:
    content_type = headers.get("content-type", "").lower()
    disposition = headers.get("content-disposition", "").lower()
    return (
        "application/pdf" in content_type
        or "filename=" in disposition
        and ".pdf" in disposition
        or body[:5] == b"%PDF-"
    )


def response_headers(response: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in response.headers.items()}


def save_pdf_body(out_dir: Path, target: Target, url: str, body: bytes) -> Path:
    stem = stem_for_target(target, url)
    path = unique_pdf_path(out_dir, stem)
    path.write_bytes(body)
    return path


def cookie_header_for_url(context: Any, url: str) -> str:
    try:
        cookies = context.cookies([url])
    except Exception:
        return ""
    pairs = []
    for cookie in cookies:
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if name:
            pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def try_urllib_pdf(
    context: Any,
    url: str,
    target: Target,
    out_dir: Path,
    timeout_ms: int,
) -> DownloadOutcome | None:
    headers = {
        "Accept": "application/pdf,*/*;q=0.8",
        "User-Agent": USER_AGENT,
    }
    cookie_header = cookie_header_for_url(context, url)
    if cookie_header:
        headers["Cookie"] = cookie_header
    if target.url != url:
        headers["Referer"] = target.url

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=max(1, timeout_ms / 1000)) as response:
            body = response.read()
            headers_lower = {str(k).lower(): str(v) for k, v in response.headers.items()}
            final_url = response.geturl()
            status = response.status
    except HTTPError as exc:
        return DownloadOutcome(
            ok=False,
            target=target,
            final_url=exc.geturl(),
            status=exc.code,
            reason=f"HTTP {exc.code}",
        )
    except URLError as exc:
        return DownloadOutcome(ok=False, target=target, final_url=url, reason=str(exc.reason))
    except Exception as exc:
        return DownloadOutcome(ok=False, target=target, final_url=url, reason=str(exc))

    if status >= 400:
        return DownloadOutcome(
            ok=False,
            target=target,
            final_url=final_url,
            status=status,
            reason=f"HTTP {status}",
        )
    if is_pdf_payload(headers_lower, body):
        path = save_pdf_body(out_dir, target, final_url, body)
        return DownloadOutcome(
            ok=True,
            target=target,
            final_url=final_url,
            pdf_path=str(path),
            status=status,
        )
    return None


def try_request_pdf(
    context: Any,
    url: str,
    target: Target,
    out_dir: Path,
    timeout_ms: int,
) -> DownloadOutcome | None:
    try:
        response = context.request.get(
            url,
            fail_on_status_code=False,
            max_redirects=12,
            timeout=timeout_ms,
            headers={
                "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
                "User-Agent": USER_AGENT,
            },
        )
    except Exception as exc:
        fallback = try_urllib_pdf(context, url, target, out_dir, timeout_ms)
        return fallback or DownloadOutcome(ok=False, target=target, final_url=url, reason=str(exc))

    headers = response_headers(response)
    body = response.body()
    final_url = response.url
    if response.status >= 400:
        fallback = try_urllib_pdf(context, url, target, out_dir, timeout_ms)
        if fallback and fallback.ok:
            return fallback
        return fallback or DownloadOutcome(
            ok=False,
            target=target,
            final_url=final_url,
            status=response.status,
            reason=f"HTTP {response.status}",
        )
    if is_pdf_payload(headers, body):
        path = save_pdf_body(out_dir, target, final_url, body)
        return DownloadOutcome(
            ok=True,
            target=target,
            final_url=final_url,
            pdf_path=str(path),
            status=response.status,
        )
    return try_urllib_pdf(context, final_url, target, out_dir, timeout_ms)


def score_candidate(url: str, text: str, source: str) -> int:
    haystack = f"{url} {text} {source}".lower()
    score = 0
    if ".pdf" in url.lower():
        score += 100
    if "pdf" in haystack:
        score += 70
    if "download" in haystack:
        score += 25
    if "full text" in haystack or "fulltext" in haystack:
        score += 20
    if "epdf" in haystack:
        score += 20
    if "article-pdf" in haystack:
        score += 30
    if "citation_pdf_url" in source:
        score += 60
    if any(skip in haystack for skip in ("facebook", "twitter", "linkedin", "mailto:")):
        score -= 100
    return score


def collect_pdf_candidates(page: Any, limit: int) -> list[PdfCandidate]:
    raw_candidates = page.evaluate(
        """() => {
          const out = [];
          const add = (url, text, source) => {
            if (!url) return;
            try {
              const absolute = new URL(url, window.location.href).href;
              out.push({url: absolute, text: text || "", source: source || ""});
            } catch (_) {}
          };
          for (const el of document.querySelectorAll('a[href], area[href]')) {
            const text = [
              el.innerText || "",
              el.getAttribute('aria-label') || "",
              el.getAttribute('title') || "",
              el.getAttribute('data-track-label') || ""
            ].join(" ");
            add(el.getAttribute('href'), text, 'link');
          }
          for (const el of document.querySelectorAll('iframe[src], embed[src], object[data]')) {
            add(el.getAttribute('src') || el.getAttribute('data'), el.getAttribute('title') || "", el.tagName);
          }
          for (const el of document.querySelectorAll('meta[name], meta[property]')) {
            const key = (el.getAttribute('name') || el.getAttribute('property') || '').toLowerCase();
            if (key.includes('pdf')) {
              add(el.getAttribute('content'), key, 'meta:' + key);
            }
          }
          return out;
        }"""
    )
    dedup: dict[str, PdfCandidate] = {}
    for item in raw_candidates:
        url = str(item.get("url") or "")
        text = str(item.get("text") or "")
        source = str(item.get("source") or "")
        score = score_candidate(url, text, source)
        if score <= 0:
            continue
        previous = dedup.get(url)
        if previous is None or score > previous.score:
            dedup[url] = PdfCandidate(url=url, text=text, source=source, score=score)
    return sorted(dedup.values(), key=lambda item: item.score, reverse=True)[:limit]


def click_pdf_candidate(
    page: Any,
    candidate: PdfCandidate,
    target: Target,
    out_dir: Path,
    timeout_ms: int,
) -> DownloadOutcome | None:
    handle = page.evaluate_handle(
        """url => {
          for (const el of document.querySelectorAll('a[href], area[href]')) {
            try {
              if (new URL(el.getAttribute('href'), window.location.href).href === url) {
                return el;
              }
            } catch (_) {}
          }
          return null;
        }""",
        candidate.url,
    )
    element = handle.as_element()
    if element is None:
        return None
    try:
        with page.expect_download(timeout=timeout_ms) as download_info:
            element.click(timeout=timeout_ms)
        download = download_info.value
        path = unique_pdf_path(out_dir, stem_for_target(target, candidate.url))
        download.save_as(str(path))
        return DownloadOutcome(
            ok=True,
            target=target,
            final_url=page.url,
            pdf_path=str(path),
            status=200,
        )
    except Exception:
        return None


def download_one(
    context: Any,
    target: Target,
    out_dir: Path,
    timeout_ms: int,
    candidate_limit: int,
) -> DownloadOutcome:
    direct = try_request_pdf(context, target.url, target, out_dir, timeout_ms)
    if direct and direct.ok:
        return direct

    page = context.new_page()
    try:
        if score_candidate(target.url, "", "") > 0:
            try:
                with page.expect_download(timeout=timeout_ms) as download_info:
                    page.goto(target.url, wait_until="domcontentloaded", timeout=timeout_ms)
                download = download_info.value
                path = unique_pdf_path(out_dir, stem_for_target(target, target.url))
                download.save_as(str(path))
                return DownloadOutcome(
                    ok=True,
                    target=target,
                    final_url=download.url or target.url,
                    pdf_path=str(path),
                    status=200,
                )
            except Exception:
                pass
        response = page.goto(target.url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
        except Exception:
            pass

        final_url = page.url
        if response is not None:
            headers = response_headers(response)
            if "application/pdf" in headers.get("content-type", "").lower():
                body = response.body()
                path = save_pdf_body(out_dir, target, response.url, body)
                return DownloadOutcome(
                    ok=True,
                    target=target,
                    final_url=response.url,
                    pdf_path=str(path),
                    status=response.status,
                )

        if final_url != target.url:
            redirected = try_request_pdf(context, final_url, target, out_dir, timeout_ms)
            if redirected and redirected.ok:
                return redirected

        candidates = collect_pdf_candidates(page, candidate_limit)
        for candidate in candidates:
            outcome = try_request_pdf(context, candidate.url, target, out_dir, timeout_ms)
            if outcome and outcome.ok:
                return outcome
            clicked = click_pdf_candidate(page, candidate, target, out_dir, timeout_ms)
            if clicked and clicked.ok:
                return clicked

        return DownloadOutcome(
            ok=False,
            target=target,
            final_url=final_url,
            reason="no PDF response found",
            candidates=[candidate.url for candidate in candidates],
            status=response.status if response is not None else None,
        )
    except Exception as exc:
        return DownloadOutcome(ok=False, target=target, final_url=page.url, reason=str(exc))
    finally:
        page.close()


def launch_context(
    playwright: Any,
    profile_dir: Path,
    download_dir: Path,
    headless: bool,
    cdp_port: int | None,
) -> Any:
    ensure_private_dir(profile_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    browser_args = ["--no-sandbox", "--disable-dev-shm-usage"]
    if cdp_port is not None:
        browser_args.extend(
            [
                f"--remote-debugging-port={cdp_port}",
                "--remote-debugging-address=127.0.0.1",
                "--remote-allow-origins=*",
            ]
        )
    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=headless,
        accept_downloads=True,
        downloads_path=str(download_dir),
        args=browser_args,
        locale="zh-CN",
        user_agent=USER_AGENT,
        viewport={"width": 1440, "height": 1000},
    )


def explain_launch_error(exc: Exception) -> None:
    message = str(exc)
    print(f"Could not start Chromium: {message}", file=sys.stderr)
    if "Executable doesn't exist" in message or "playwright install" in message:
        print(
            "\nInstall the browser runtime with:\n"
            "  python -m playwright install chromium\n",
            file=sys.stderr,
        )
    if "Missing X server" in message or "no DISPLAY" in message or "Target page" in message:
        print(
            "\nThis command needs an interactive browser. On a headless remote server,\n"
            "use SSH X forwarding/noVNC if available, or run with --headless --cdp-port\n"
            "for an experimental DevTools-based login.\n",
            file=sys.stderr,
        )


def cmd_check(args: argparse.Namespace) -> int:
    urls = args.url or [TSINGHUA_DATABASE_NAV_URL, TSINGHUA_EPROXY_HOME_URL]
    ok = True
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            started = time.time()
            with urlopen(request, timeout=args.timeout) as response:
                elapsed = time.time() - started
                content = response.read(300)
                result = {
                    "url": url,
                    "ok": True,
                    "status": response.status,
                    "content_type": response.headers.get("content-type"),
                    "final_url": response.geturl(),
                    "elapsed_s": round(elapsed, 2),
                    "sample": content.decode("utf-8", "replace").replace("\n", " ")[:160],
                }
        except HTTPError as exc:
            ok = False
            result = {
                "url": url,
                "ok": False,
                "status": exc.code,
                "reason": exc.reason,
                "final_url": exc.geturl(),
            }
        except URLError as exc:
            ok = False
            result = {"url": url, "ok": False, "reason": str(exc.reason)}
        except Exception as exc:
            ok = False
            result = {"url": url, "ok": False, "reason": str(exc)}

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            status = result.get("status", "-")
            label = "OK" if result["ok"] else "FAIL"
            print(f"[{label}] {url} status={status} final={result.get('final_url', '-')}")
            if result.get("reason"):
                print(f"  reason: {result['reason']}")
    return 0 if ok else 2


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"DISPLAY: {os.environ.get('DISPLAY') or '<not set>'}")
    network_args = argparse.Namespace(
        url=[TSINGHUA_DATABASE_NAV_URL, TSINGHUA_EPROXY_HOME_URL],
        timeout=args.timeout_s,
        json=False,
    )
    network_code = cmd_check(network_args)

    try:
        sync_playwright, PlaywrightError, _ = require_playwright()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    profile_dir = Path(args.profile_dir)
    download_dir = Path(args.download_dir)
    try:
        with sync_playwright() as playwright:
            context = launch_context(
                playwright,
                profile_dir=profile_dir,
                download_dir=download_dir,
                headless=True,
                cdp_port=None,
            )
            page = context.new_page()
            page.goto(TSINGHUA_EPROXY_HOME_URL, wait_until="domcontentloaded", timeout=args.timeout_ms)
            title = page.title()
            print(f"[OK] headless Chromium opened Eproxy final={page.url} title={title!r}")
            context.close()
    except PlaywrightError as exc:
        print("[FAIL] headless Chromium launch/navigation failed", file=sys.stderr)
        explain_launch_error(exc)
        return 2
    except Exception as exc:
        print(f"[FAIL] browser probe failed: {exc}", file=sys.stderr)
        return 2
    return network_code


def cmd_login(args: argparse.Namespace) -> int:
    sync_playwright, PlaywrightError, _ = require_playwright()
    profile_dir = Path(args.profile_dir)
    download_dir = Path(args.download_dir)
    headless = bool(args.headless)
    if not headless and not os.environ.get("DISPLAY"):
        print(
            "No DISPLAY is set, so a headed browser may not open in this container.\n"
            "Use a remote desktop/noVNC/X11 session if your provider offers one, or try:\n"
            f"  python {rel(Path(__file__))} login --headless --cdp-port 9222\n"
            "Then forward/open the DevTools port manually.",
            file=sys.stderr,
        )

    with sync_playwright() as playwright:
        try:
            context = launch_context(
                playwright,
                profile_dir=profile_dir,
                download_dir=download_dir,
                headless=headless,
                cdp_port=args.cdp_port,
            )
        except PlaywrightError as exc:
            explain_launch_error(exc)
            return 2
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        print(f"Opened: {page.url}")
        if args.cdp_port is not None:
            print(f"DevTools endpoint: http://127.0.0.1:{args.cdp_port}")
            print("DevTools listens on 127.0.0.1; use SSH port forwarding from your laptop.")
        print(f"Browser profile: {rel(profile_dir)}")
        print("Complete Tsinghua authentication in the browser, then press Enter here.")
        try:
            input()
        finally:
            print(f"Final page: {page.url}")
            context.close()
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    sync_playwright, PlaywrightError, _ = require_playwright()
    profile_dir = Path(args.profile_dir)
    download_dir = Path(args.download_dir)
    with sync_playwright() as playwright:
        try:
            context = launch_context(
                playwright,
                profile_dir=profile_dir,
                download_dir=download_dir,
                headless=bool(args.headless),
                cdp_port=args.cdp_port,
            )
        except PlaywrightError as exc:
            explain_launch_error(exc)
            return 2
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        print(f"Opened: {page.url}")
        print("Press Enter to close the browser.")
        try:
            input()
        finally:
            context.close()
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    targets = load_targets(args)
    sync_playwright, PlaywrightError, _ = require_playwright()
    profile_dir = Path(args.profile_dir)
    download_dir = Path(args.download_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0

    with sync_playwright() as playwright:
        try:
            context = launch_context(
                playwright,
                profile_dir=profile_dir,
                download_dir=download_dir,
                headless=not args.headed,
                cdp_port=args.cdp_port,
            )
        except PlaywrightError as exc:
            explain_launch_error(exc)
            return 2

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for index, target in enumerate(targets, 1):
                if index > 1:
                    time.sleep(args.delay_s)
                print(f"[{index}/{len(targets)}] {target.url}")
                outcome = download_one(
                    context=context,
                    target=target,
                    out_dir=download_dir,
                    timeout_ms=args.timeout_ms,
                    candidate_limit=args.candidate_limit,
                )
                record = {
                    "ok": outcome.ok,
                    "source": target.source,
                    "url": target.url,
                    "doi": target.doi,
                    "title": target.title,
                    "record_id": target.record_id,
                    "final_url": outcome.final_url,
                    "pdf_path": outcome.pdf_path,
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "candidates": outcome.candidates,
                    "ts": int(time.time()),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                if outcome.ok:
                    print(f"  saved: {rel(Path(outcome.pdf_path or ''))}")
                else:
                    exit_code = 1
                    print(f"  failed: {outcome.reason or 'unknown'}")
                    if outcome.candidates:
                        print(f"  candidates: {len(outcome.candidates)}")
        context.close()
    print(f"Manifest: {rel(manifest_path)}")
    return exit_code


def add_common_browser_args(parser: argparse.ArgumentParser, *, default_headless: bool) -> None:
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR), help="Persistent Chromium profile directory.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="PDF output/download directory.")
    parser.add_argument("--timeout-ms", type=int, default=45000, help="Navigation/request timeout in milliseconds.")
    parser.add_argument("--cdp-port", type=int, default=None, help="Expose Chromium DevTools on this port.")
    if default_headless:
        parser.add_argument("--headed", action="store_true", help="Show a headed browser instead of headless mode.")
    else:
        parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check network access to Tsinghua library entry pages.")
    check.add_argument("--url", action="append", help="URL to check. Repeatable.")
    check.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds.")
    check.add_argument("--json", action="store_true", help="Emit JSON lines.")
    check.set_defaults(func=cmd_check)

    doctor = sub.add_parser("doctor", help="Check network access and headless Chromium launch.")
    doctor.add_argument("--timeout-s", type=int, default=15, help="HTTP timeout in seconds.")
    doctor.add_argument("--timeout-ms", type=int, default=45000, help="Browser navigation timeout in milliseconds.")
    doctor.add_argument("--profile-dir", default=str(DEFAULT_DOCTOR_PROFILE_DIR), help="Temporary browser probe profile.")
    doctor.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Download directory for browser probe.")
    doctor.set_defaults(func=cmd_doctor)

    login = sub.add_parser("login", help="Open a persistent browser session and save the login state.")
    login.add_argument("--url", default=TSINGHUA_EPROXY_HOME_URL, help="Login/start URL.")
    add_common_browser_args(login, default_headless=False)
    login.set_defaults(func=cmd_login)

    open_cmd = sub.add_parser("open", help="Open a URL with the saved browser profile.")
    open_cmd.add_argument("--url", default=TSINGHUA_DATABASE_NAV_URL, help="URL to open.")
    add_common_browser_args(open_cmd, default_headless=False)
    open_cmd.set_defaults(func=cmd_open)

    download = sub.add_parser("download", help="Download PDF(s) using the saved browser profile.")
    download.add_argument("--url", action="append", help="Landing/PDF URL. Repeatable.")
    download.add_argument("--doi", action="append", help="DOI to resolve through https://doi.org/. Repeatable.")
    download.add_argument("--input", help="Text or JSONL file with URL/DOI targets.")
    download.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Append JSONL download outcomes here.")
    download.add_argument("--delay-s", type=float, default=8.0, help="Delay between multiple targets.")
    download.add_argument("--candidate-limit", type=int, default=20, help="Maximum PDF-like links to try per page.")
    add_common_browser_args(download, default_headless=True)
    download.set_defaults(func=cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
