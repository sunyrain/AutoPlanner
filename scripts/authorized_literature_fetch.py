#!/usr/bin/env python3
"""Fetch one DOI through the existing publisher-specific local spider stack.

Run this helper with the isolated Python environment that owns Selenium and
undetected-chromedriver.  All mutable state is redirected into ``--output-dir``;
the literature_datamining source tree is imported as code only, never as a
source-document cache.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any
from urllib.parse import unquote, urljoin
from urllib.request import Request, urlopen


PUBLISHER_BY_PREFIX = {
    "10.1002/": "Wiley",
    "10.1007/": "Springer",
    "10.1016/": "Elsevier",
    "10.1021/": "ACS",
    "10.1038/": "Nature",
    "10.1039/": "RSC",
    "10.1055/": "Thieme",
    "10.1093/": "OUP",
    "10.1080/": "TnF",
    "10.2174/": "Bentham",
    "10.31635/": "Wiley",
    "10.3390/": "MDPI",
    "10.5059/": "JStage",
}

GENERIC_BROWSER_PUBLISHERS = {"Bentham", "JStage", "OUP", "Thieme"}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value}
        href = values.get("href")
        if href:
            self.links.append(href)
        if (
            tag.casefold() == "meta"
            and values.get("name", "").casefold() == "citation_pdf_url"
            and values.get("content")
        ):
            self.links.append(str(values["content"]))


def _extract_links(content: str, *, base_url: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(content)
    links: list[str] = []
    for href in parser.links:
        url = urljoin(base_url, href)
        if url.startswith(("http://", "https://")) and url not in links:
            links.append(url)
    return links


def _looks_like_supplement_link(url: str) -> bool:
    lowered = url.casefold()
    name = lowered.rsplit("/", 1)[-1]
    return any(
        token in lowered
        for token in (
            "article-supplement",
            "supplement",
            "/suppl/",
            "/supmat/",
            "supporting-information",
            "supinfo",
            "suppdata",
        )
    ) or name.startswith(("si_", "esi_"))


def _looks_like_main_article_link(url: str, *, doi: str) -> bool:
    """Reject generic publisher PDFs while retaining article-bound endpoints."""

    lowered = unquote(url).casefold()
    path = lowered.split("?", 1)[0]
    name = path.rsplit("/", 1)[-1]
    return not _looks_like_supplement_link(url) and (
        doi.casefold() in lowered
        or "article-pdf" in lowered
        or "/doi/pdf/" in lowered
        or "/doi/epdf/" in lowered
        or ("/article/" in lowered and "/_pdf" in lowered)
        or name in {"article.pdf", "main-article.pdf"}
    )


def _verified_publisher_landing(content: str, *, doi: str) -> bool:
    lowered = content.casefold()
    return (
        len(content) >= 2_000
        and doi.casefold() in lowered
        and "challenges.cloudflare.com" not in lowered
    )


def _publisher(doi: str, explicit: str) -> str:
    if explicit:
        return explicit
    lowered = doi.casefold()
    return next(
        (
            publisher
            for prefix, publisher in PUBLISHER_BY_PREFIX.items()
            if lowered.startswith(prefix)
        ),
        "",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_chrome_major() -> int:
    roots = [
        Path(os.getenv("ProgramFiles", "")) / "Google" / "Chrome" / "Application",
        Path(os.getenv("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application",
        Path(os.getenv("LocalAppData", "")) / "Google" / "Chrome" / "Application",
    ]
    versions: list[tuple[tuple[int, ...], int]] = []
    for root in roots:
        if not (root / "chrome.exe").is_file():
            continue
        for child in root.iterdir():
            if not child.is_dir() or not child.name[:1].isdigit():
                continue
            try:
                parts = tuple(int(part) for part in child.name.split("."))
            except ValueError:
                continue
            versions.append((parts, parts[0]))
    return max(versions, default=((0,), 0))[1]


def _prepare_isolated_chromedriver(output_dir: Path, chrome_major: int) -> Path | None:
    """Prepare a version-matched UC driver without trusting the global cache.

    The legacy spider enables ``user_multi_procs``.  Undetected-chromedriver's
    corresponding fast path reuses the newest global cached binary without
    checking that its major version still matches Chrome.  Keep the cache
    inside this fetch transaction and populate it explicitly before the
    legacy initializer runs.
    """

    if chrome_major <= 0:
        return None
    import undetected_chromedriver as uc

    driver_cache = output_dir / "uc-driver-cache"
    driver_cache.mkdir(parents=True, exist_ok=True)
    uc.Patcher.data_path = str(driver_cache)
    patcher = uc.Patcher(version_main=chrome_major, user_multi_procs=False)
    patcher.auto()
    driver_path = Path(str(patcher.executable_path)).resolve()
    if not driver_path.is_file():
        raise RuntimeError("version_matched_chromedriver_missing")
    return driver_path


def _initialize_pinned_webdriver(
    *,
    profile: Path,
    downloads: Path,
    chrome_binary: Path,
    chromedriver: Path | None,
    chrome_major: int,
) -> Any:
    """Launch the legacy spiders with an explicit, reproducible Chrome binary."""

    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.binary_location = str(chrome_binary)
    options.add_argument("--no-proxy-server")
    options.add_argument("--proxy-server=direct://")
    options.add_argument("--proxy-bypass-list=*")
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(downloads),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
            "safebrowsing.enabled": True,
            "download.directory_upgrade": True,
            "download.useDownloadDir": True,
            "profile.default_content_setting_values.automatic_downloads": 1,
        },
    )
    return uc.Chrome(
        data_path=str(profile),
        use_subprocess=True,
        options=options,
        version_main=chrome_major,
        driver_executable_path=str(chromedriver) if chromedriver else None,
        headless=False,
        user_multi_procs=False,
    )


def _artifacts(root: Path) -> list[dict[str, Any]]:
    allowed = {".html", ".json", ".pdf", ".zip", ".doc", ".docx", ".txt", ".xml"}
    return [
        {
            "path": str(path.resolve()),
            "relative_path": path.relative_to(root).as_posix(),
            "suffix": path.suffix.casefold(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.casefold() in allowed
        and path.name != "authorized-literature-fetch.json"
    ]


def _usable_fulltext_content(content: str, *, doi: str) -> bool:
    signals = (
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
    lowered = content.casefold()
    return (
        len(content) >= 2_000
        and doi.casefold() in lowered
        and sum(signal in lowered for signal in signals) >= 2
        and "challenges.cloudflare.com" not in lowered
    )


def _usable_fulltext_html(artifacts: list[dict[str, Any]], *, doi: str) -> bool:
    for artifact in artifacts:
        if artifact.get("suffix") != ".html":
            continue
        path = Path(str(artifact.get("path") or ""))
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _usable_fulltext_content(content, doi=doi):
            return True
    return False


def _legacy_main_pdf_required(
    artifacts: list[dict[str, Any]],
    *,
    doi: str,
) -> bool:
    """Identify legacy ACS landing pages whose scientific body is PDF-only."""

    if any(row.get("suffix") == ".pdf" for row in artifacts):
        return False
    for artifact in artifacts:
        if artifact.get("suffix") != ".html":
            continue
        try:
            content = (
                Path(str(artifact.get("path") or ""))
                .read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                .casefold()
            )
        except OSError:
            continue
        if "article_header-acslegacyarchive" in content and f"/doi/pdf/{doi}".casefold() in content:
            return True
    return False


def _authenticated_fetch(driver: Any, url: str) -> tuple[bytes, dict[str, Any]]:
    script = r"""
        const url = arguments[0];
        const done = arguments[arguments.length - 1];
        fetch(url, {credentials: 'include', redirect: 'follow'})
          .then(async (response) => {
            const bytes = new Uint8Array(await response.arrayBuffer());
            let binary = '';
            const chunk = 0x8000;
            for (let i = 0; i < bytes.length; i += chunk) {
              binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
            }
            done({
              ok: response.ok,
              status: response.status,
              contentType: response.headers.get('content-type') || '',
              bodyBase64: btoa(binary)
            });
          })
          .catch((error) => done({ok: false, status: 0, error: String(error)}));
    """
    result = dict(driver.execute_async_script(script, url) or {})
    payload = base64.b64decode(str(result.get("bodyBase64") or ""))
    return payload, result


def _declared_link_fetch(url: str, *, referer: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch a publisher-declared link when browser fetch is blocked by CORS.

    This is not a discovery bypass: callers may use it only for URLs extracted
    from the publisher page currently loaded in the authorized browser session.
    """

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/152.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/zip,application/octet-stream,*/*;q=0.8",
            "Referer": referer,
        },
    )
    with urlopen(request, timeout=180) as response:
        payload = response.read(100_000_001)
        if len(payload) > 100_000_000:
            raise ValueError("declared_link_exceeds_100mb")
        return payload, {
            "ok": 200 <= int(response.status) < 300,
            "status": int(response.status),
            "contentType": str(response.headers.get("Content-Type") or ""),
        }


def _download_authenticated_main_pdf(
    driver: Any,
    *,
    doi: str,
    publisher: str,
    output_dir: Path,
    page_source: str = "",
    current_url: str = "",
) -> dict[str, Any]:
    """Fetch a main PDF inside the authenticated Selenium page origin."""

    urls: list[str] = []
    declared_links = _extract_links(page_source, base_url=current_url)
    if publisher == "ACS":
        urls.extend(
            [
                f"https://pubs.acs.org/doi/pdf/{doi}?ref=article_openPDF",
                f"https://pubs.acs.org/doi/epdf/{doi}",
            ]
        )
    if publisher == "Wiley" and doi.casefold().startswith("10.31635/"):
        urls.extend(
            [
                f"https://www.chinesechemsoc.org/doi/pdf/{doi}",
                f"https://www.chinesechemsoc.org/doi/epdf/{doi}",
            ]
        )
    if publisher == "Thieme":
        urls.extend(
            [
                f"https://www.thieme-connect.com/products/ejournals/pdf/{doi}.pdf",
                f"https://www.thieme-connect.de/products/ejournals/pdf/{doi}.pdf",
            ]
        )
    if publisher == "Bentham":
        urls.append(f"https://www.eurekaselect.com/article/download?doi={doi}")
    for url in declared_links:
        if _looks_like_main_article_link(url, doi=doi) and url not in urls:
            urls.append(url)
    if not urls:
        return {"status": "not_supported", "reason": "main_pdf_endpoint_unknown"}
    failures: list[str] = []
    for url in urls:
        try:
            payload, result = _authenticated_fetch(driver, url)
        except Exception as exc:
            failures.append(f"{url}:{type(exc).__name__}:{str(exc)[:300]}")
            continue
        if not payload[:1024].lstrip().startswith(b"%PDF-") and url in declared_links:
            try:
                payload, result = _declared_link_fetch(url, referer=current_url)
            except Exception as exc:
                failures.append(f"{url}:declared_link_fetch:{type(exc).__name__}:{str(exc)[:300]}")
        if payload[:1024].lstrip().startswith(b"%PDF-") and len(payload) <= 30_000_000:
            digest = hashlib.sha256(payload).hexdigest()
            path = output_dir / f"main-article-{digest[:16]}.pdf"
            if not path.exists():
                path.write_bytes(payload)
            return {
                "status": "downloaded",
                "pdf_path": str(path.resolve()),
                "pdf_sha256": digest,
                "byte_count": len(payload),
                "url": url,
            }
        failures.append(
            f"{url}:status={int(result.get('status') or 0)}:"
            f"content_type={str(result.get('contentType') or '')}:bytes={len(payload)}"
        )
    return {
        "status": "failed",
        "reason": "; ".join(failures)[:2000] or "main_pdf_fetch_failed",
    }


def _download_authenticated_supplements(
    driver: Any,
    *,
    publisher: str,
    page_source: str,
    current_url: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Download publisher-declared SI links after the article page is verified."""

    del publisher

    def looks_like_file(url: str) -> bool:
        path = url.casefold().split("?", 1)[0]
        return path.endswith((".pdf", ".zip", ".doc", ".docx", ".xlsx", ".cif"))

    queue: list[tuple[str, int]] = [
        (url, 0)
        for url in _extract_links(page_source, base_url=current_url)
        if _looks_like_supplement_link(url)
    ]

    receipts: list[dict[str, Any]] = []
    supplement_dir = output_dir / "supplementary_materials"
    visited: set[str] = set()
    while queue and len(visited) < 30:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            payload, result = _authenticated_fetch(driver, url)
        except Exception as exc:
            receipts.append(
                {
                    "status": "failed",
                    "url": url,
                    "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
            )
            continue
        if not (payload[:1024].lstrip().startswith(b"%PDF-") or payload.startswith(b"PK\x03\x04")):
            try:
                payload, result = _declared_link_fetch(url, referer=current_url)
            except Exception:
                pass
        if payload[:1024].lstrip().startswith(b"%PDF-") and len(payload) <= 100_000_000:
            supplement_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            path = supplement_dir / f"supplement-{len(receipts) + 1:02d}-{digest[:16]}.pdf"
            if not path.exists():
                path.write_bytes(payload)
            receipts.append(
                {
                    "status": "downloaded",
                    "url": url,
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "byte_count": len(payload),
                }
            )
            continue
        if payload.startswith(b"PK\x03\x04") and len(payload) <= 100_000_000:
            supplement_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            lowered_path = url.casefold().split("?", 1)[0]
            suffix = next(
                (
                    extension
                    for extension in (".docx", ".xlsx", ".zip")
                    if lowered_path.endswith(extension)
                ),
                ".zip",
            )
            path = supplement_dir / (f"supplement-{len(receipts) + 1:02d}-{digest[:16]}{suffix}")
            if not path.exists():
                path.write_bytes(payload)
            receipts.append(
                {
                    "status": "downloaded",
                    "url": url,
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "byte_count": len(payload),
                }
            )
            continue
        content_type = str(result.get("contentType") or "").casefold()
        if depth == 0 and "html" in content_type and payload:
            landing = payload.decode("utf-8", errors="ignore")
            nested = [
                link
                for link in _extract_links(landing, base_url=url)
                if looks_like_file(link) or _looks_like_supplement_link(link)
            ]
            queue.extend((link, 1) for link in nested if link not in visited)
            receipts.append(
                {
                    "status": "expanded_landing",
                    "url": url,
                    "discovered_links": len(nested),
                }
            )
            continue
        receipts.append(
            {
                "status": "failed",
                "url": url,
                "reason": (
                    f"status={int(result.get('status') or 0)}:"
                    f"content_type={str(result.get('contentType') or '')}:"
                    f"bytes={len(payload)}"
                ),
            }
        )
    return receipts


def _install_tolerant_navigation(driver: Any, *, doi: str) -> None:
    """Continue after a load-event timeout when the DOI page is already usable."""

    executor = getattr(driver, "command_executor", None)
    set_timeout = getattr(executor, "set_timeout", None)
    if callable(set_timeout):
        set_timeout(180)
    original_get = driver.get

    def wait_for_managed_challenge() -> None:
        for _ in range(30):
            try:
                page_source = str(driver.page_source or "").casefold()
            except Exception:
                return
            challenge = "challenges.cloudflare.com" in page_source and (
                "cf-turnstile-response" in page_source or "challenge-platform" in page_source
            )
            if not challenge:
                return
            time.sleep(1)

    def tolerant_get(url: str) -> Any:
        try:
            result = original_get(url)
            wait_for_managed_challenge()
            return result
        except Exception:
            try:
                page_source = str(driver.page_source or "")
            except Exception:
                page_source = ""
            if len(page_source) >= 2_000 and doi.casefold() in page_source.casefold():
                return None
            raise

    driver.get = tolerant_get


def _generic_publisher_landing_url(doi: str, publisher: str) -> str:
    if publisher == "Thieme":
        return f"https://www.thieme-connect.com/products/ejournals/abstract/{doi}"
    return f"https://doi.org/{doi}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--publisher", default="")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(
            os.getenv("AUTOPLANNER_LITERATURE_DATAMINING_ROOT", r"D:\Autoplanner\shared\src")
        ),
    )
    parser.add_argument("--chrome-major", type=int, default=0)
    parser.add_argument("--chrome-binary", type=Path)
    parser.add_argument(
        "--chromedriver",
        type=Path,
        help="Prevalidated version-matched driver prepared by the batch coordinator.",
    )
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--force-refetch", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_root = args.package_root.expanduser().resolve()
    receipt_path = output_dir / "authorized-literature-fetch.json"
    publisher = _publisher(args.doi, args.publisher)
    if not publisher:
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "authorized_literature_fetch.v1",
                    "accepted": False,
                    "doi": args.doi,
                    "reason": "publisher_adapter_unavailable",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    existing_artifacts = _artifacts(output_dir)
    if (
        not args.force_refetch
        and _usable_fulltext_html(existing_artifacts, doi=args.doi)
        and not _legacy_main_pdf_required(existing_artifacts, doi=args.doi)
    ):
        receipt = {
            "schema_version": "authorized_literature_fetch.v1",
            "accepted": True,
            "doi": args.doi,
            "publisher": publisher,
            "page_status": "content_verified_after_legacy_status_mismatch",
            "reason": "",
            "artifact_count": len(existing_artifacts),
            "artifacts": existing_artifacts,
            "article_summary": {},
            "semantics": {
                "publisher_spider_code_reused": True,
                "prior_source_documents_not_reused": True,
                "isolated_current_run_artifact_resumed": True,
                "content_identity_and_procedure_signals_verified": True,
            },
        }
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.manifest_root:
            _append_proxy_manifest(
                args.manifest_root.expanduser().resolve(),
                request={
                    "request_id": args.request_id,
                    "case_id": args.case_id,
                    "source_ref": args.source_ref or f"doi:{args.doi}",
                    "doi": args.doi,
                    "url": args.url or f"https://doi.org/{args.doi}",
                    "title": args.title,
                },
                publisher=publisher,
                artifacts=existing_artifacts,
            )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    isolated_home = output_dir / "ldm-home"
    os.environ["LDM_HOME"] = str(isolated_home)
    os.environ["LDM_CONFIG_FILE"] = str(isolated_home / "no-external-config.yaml")
    os.environ["SPIDER_DOWNLOAD_PDF"] = "true"
    os.environ["SPIDER_DOWNLOAD_FIGURES"] = "true"
    os.environ["SPIDER_DOWNLOAD_SUPPLEMENTARY"] = "true"
    os.environ["SPIDER_HTML_ONLY"] = "false"
    chrome_major = int(args.chrome_major or _detect_chrome_major())
    chrome_binary = args.chrome_binary.expanduser().resolve() if args.chrome_binary else None
    if chrome_binary is not None and not chrome_binary.is_file():
        raise FileNotFoundError(f"chrome binary not found: {chrome_binary}")
    shared_driver = args.chromedriver.expanduser().resolve() if args.chromedriver else None
    if shared_driver is not None and not shared_driver.is_file():
        raise FileNotFoundError(f"chromedriver not found: {shared_driver}")
    if shared_driver is not None:
        # The legacy initializer enables ``user_multi_procs``.  Its fast path
        # consults Patcher.data_path even when an explicit executable is
        # supplied, so bind that process-local cache authority to the shared,
        # version-checked driver instead of a stale global UC cache.
        import undetected_chromedriver as uc

        uc.Patcher.data_path = str(shared_driver.parent)
    if chrome_major:
        os.environ["SPIDER_CHROME_VERSION"] = str(chrome_major)
    sys.path.insert(0, str(package_root))

    # The legacy publisher package imports html_table_extractor at module load
    # even though its spider path never uses it.  Keep the provider usable in
    # the isolated runtime when that optional table helper is absent.
    try:
        __import__("html_table_extractor.extractor")
    except ModuleNotFoundError:
        package = types.ModuleType("html_table_extractor")
        extractor = types.ModuleType("html_table_extractor.extractor")
        extractor.Extractor = object
        package.extractor = extractor
        sys.modules["html_table_extractor"] = package
        sys.modules["html_table_extractor.extractor"] = extractor
    try:
        __import__("markdownify")
    except ModuleNotFoundError:
        markdownify = types.ModuleType("markdownify")
        markdownify.markdownify = lambda value, **_kwargs: str(value)
        sys.modules["markdownify"] = markdownify

    driver = None
    article_data: dict[str, Any] = {}
    page_source = ""
    success = False
    page_status = "not_started"
    reason = ""
    failure_diagnostic: dict[str, Any] = {}
    main_pdf_receipt: dict[str, Any] = {
        "status": "not_attempted",
        "reason": "publisher_spider_not_started",
    }
    supplement_receipts: list[dict[str, Any]] = []
    try:
        from literature_datamining import config as ldm_config
        from literature_datamining.core.utils import initialize_webdriver

        # Do not let a stale bundled or global cached executable override the
        # version-matched driver prepared for this isolated fetch transaction.
        isolated_driver = shared_driver or _prepare_isolated_chromedriver(
            output_dir, chrome_major
        )
        ldm_config.WORKSPACE_ROOT = isolated_home
        ldm_config.CHROMEDRIVER_PATH = isolated_driver
        if chrome_major:
            ldm_config.CHROME_VERSION = chrome_major

        spiders = ldm_config.get_spider_classes()
        spider_class = spiders.get(publisher)
        if spider_class is None and publisher not in GENERIC_BROWSER_PUBLISHERS:
            raise RuntimeError(f"publisher_adapter_unavailable:{publisher}")
        profile = output_dir / "browser-profile"
        downloads = output_dir / "browser-downloads"
        profile.mkdir(parents=True, exist_ok=True)
        downloads.mkdir(parents=True, exist_ok=True)
        driver = (
            _initialize_pinned_webdriver(
                profile=profile,
                downloads=downloads,
                chrome_binary=chrome_binary,
                chromedriver=isolated_driver,
                chrome_major=chrome_major,
            )
            if chrome_binary is not None
            else initialize_webdriver(str(profile), str(downloads), extension_path=None)
        )
        _install_tolerant_navigation(driver, doi=args.doi)
        article_dir = output_dir / "article"
        article_dir.mkdir(parents=True, exist_ok=True)
        if spider_class is None:
            driver.get(_generic_publisher_landing_url(args.doi, publisher))
            page_source = str(driver.page_source or "")
            success = _usable_fulltext_content(page_source, doi=args.doi)
            page_status = (
                "content_verified_generic_browser" if success else "generic_publisher_landing_only"
            )
            article_data = {
                "title": str(driver.title or ""),
                "publisher": publisher,
                "doi": args.doi,
            }
            if success:
                (article_dir / "article.html").write_text(
                    page_source, encoding="utf-8", errors="ignore"
                )
        else:
            spider = spider_class(
                f"https://doi.org/{args.doi}",
                args.doi,
                str(article_dir),
                driver,
                str(downloads),
                "html",
            )
            raw_data, success, page_status = spider.run()
            article_data = dict(raw_data or {})
            page_source = str(driver.page_source or "")
            if not success and _usable_fulltext_content(page_source, doi=args.doi):
                spider.process_url()
                success = bool(spider.success)
                article_data = dict(spider.article_data or {})
                if success:
                    page_status = "content_verified_after_legacy_status_mismatch"
        if not success:
            diagnostic_html = output_dir / "publisher-failure-page.diagnostic"
            diagnostic_html.write_text(page_source, encoding="utf-8", errors="ignore")
            screenshot_path = output_dir / "publisher-failure-page.png"
            screenshot_saved = bool(driver.save_screenshot(str(screenshot_path)))
            failure_diagnostic = {
                "current_url": str(driver.current_url or ""),
                "document_title": str(driver.title or ""),
                "page_source_bytes": diagnostic_html.stat().st_size,
                "diagnostic_html": str(diagnostic_html),
                "screenshot": str(screenshot_path) if screenshot_saved else "",
            }
            lowered_source = page_source.casefold()
            if (
                "challenges.cloudflare.com" in lowered_source
                and "challenge-platform" in lowered_source
            ):
                page_status = "publisher_security_challenge"
        main_pdf_receipt = _download_authenticated_main_pdf(
            driver,
            doi=args.doi,
            publisher=publisher,
            output_dir=article_dir,
            page_source=page_source,
            current_url=str(driver.current_url or ""),
        )
        supplement_receipts = _download_authenticated_supplements(
            driver,
            publisher=publisher,
            page_source=page_source,
            current_url=str(driver.current_url or ""),
            output_dir=article_dir,
        )
        if article_data:
            (article_dir / "article-data.json").write_text(
                json.dumps(article_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:  # provider isolation boundary
        reason = f"{type(exc).__name__}:{str(exc)[:1000]}"
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    artifacts = _artifacts(output_dir)
    content_verified = _usable_fulltext_html(artifacts, doi=args.doi)
    landing_identity_verified = _verified_publisher_landing(page_source, doi=args.doi)
    supplement_downloaded = any(row.get("status") == "downloaded" for row in supplement_receipts)
    accepted = bool(
        (success and artifacts)
        or content_verified
        or main_pdf_receipt.get("status") == "downloaded"
        or (landing_identity_verified and supplement_downloaded)
    )
    receipt = {
        "schema_version": "authorized_literature_fetch.v1",
        "accepted": accepted,
        "doi": args.doi,
        "publisher": publisher,
        "chrome_major": chrome_major,
        "chrome_binary": str(chrome_binary) if chrome_binary is not None else "system",
        "page_status": page_status,
        "reason": reason if not accepted else "",
        "failure_diagnostic": failure_diagnostic if not accepted else {},
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "article_summary": {
            "title": article_data.get("title"),
            "full_text_section_count": len(article_data.get("full_text") or []),
            "figure_count": len(article_data.get("figures") or []),
            "supplementary_count": len(article_data.get("supplementary_materials") or []),
            "main_pdf": main_pdf_receipt,
            "supplementary_downloads": supplement_receipts,
        },
        "semantics": {
            "publisher_spider_code_reused": True,
            "prior_source_documents_not_reused": True,
            "all_mutable_state_isolated_under_output_dir": True,
            "source_artifacts_require_host_hash_binding_and_extraction": True,
            "authenticated_main_pdf_attempted_after_spider": True,
            "content_verified_after_legacy_status_mismatch": bool(content_verified and not success),
            "publisher_landing_identity_verified": landing_identity_verified,
            "supplement_only_acceptance": bool(
                landing_identity_verified
                and supplement_downloaded
                and not content_verified
                and main_pdf_receipt.get("status") != "downloaded"
            ),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if accepted and args.manifest_root:
        _append_proxy_manifest(
            args.manifest_root.expanduser().resolve(),
            request={
                "request_id": args.request_id,
                "case_id": args.case_id,
                "source_ref": args.source_ref or f"doi:{args.doi}",
                "doi": args.doi,
                "url": args.url or f"https://doi.org/{args.doi}",
                "title": args.title,
            },
            publisher=publisher,
            artifacts=artifacts,
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if accepted else 2


def _append_proxy_manifest(
    root: Path,
    *,
    request: dict[str, str],
    publisher: str,
    artifacts: list[dict[str, Any]],
) -> None:
    def preferred(suffix: str, *, exclude_debug: bool = False) -> dict[str, Any]:
        rows = [row for row in artifacts if row.get("suffix") == suffix]
        if exclude_debug:
            preferred_rows = [
                row for row in rows if "debug_" not in str(row.get("relative_path") or "")
            ]
            rows = preferred_rows or rows
        return rows[0] if rows else {}

    html = preferred(".html", exclude_debug=True)
    structured = next(
        (
            row
            for row in artifacts
            if str(row.get("relative_path") or "").endswith("article-data.json")
        ),
        preferred(".json"),
    )
    pdf = preferred(".pdf")
    row = {
        "schema_version": "local_pdf_proxy_result.v1",
        **request,
        "status": "downloaded",
        "accepted": True,
        "provider": "legacy_publisher_spider",
        "publisher": publisher,
        "artifact_kind": "publisher_source_bundle",
        "html_path": str(html.get("path") or ""),
        "html_sha256": str(html.get("sha256") or ""),
        "structured_path": str(structured.get("path") or ""),
        "structured_sha256": str(structured.get("sha256") or ""),
        "pdf_path": str(pdf.get("path") or ""),
        "pdf_sha256": str(pdf.get("sha256") or ""),
        "artifact_paths": [str(value.get("path") or "") for value in artifacts],
        "fetch_mode": "isolated_legacy_publisher_spider",
    }
    manifest = root / "evidence" / "local_pdf_proxy" / "pdf_download_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
