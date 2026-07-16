#!/usr/bin/env python3
"""Fetch one DOI through the existing publisher-specific local spider stack.

Run this helper with the isolated Python environment that owns Selenium and
undetected-chromedriver.  All mutable state is redirected into ``--output-dir``;
the literature_datamining source tree is imported as code only, never as a
source-document cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any


PUBLISHER_BY_PREFIX = {
    "10.1002/": "Wiley",
    "10.1007/": "Springer",
    "10.1016/": "Elsevier",
    "10.1021/": "ACS",
    "10.1038/": "Nature",
    "10.1039/": "RSC",
    "10.1080/": "TnF",
    "10.3390/": "MDPI",
}


def _publisher(doi: str, explicit: str) -> str:
    if explicit:
        return explicit
    lowered = doi.casefold()
    return next(
        (publisher for prefix, publisher in PUBLISHER_BY_PREFIX.items() if lowered.startswith(prefix)),
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


def _usable_fulltext_html(artifacts: list[dict[str, Any]], *, doi: str) -> bool:
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
    for artifact in artifacts:
        if artifact.get("suffix") != ".html":
            continue
        path = Path(str(artifact.get("path") or ""))
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError:
            continue
        if (
            len(content) >= 2_000
            and doi.casefold() in content
            and sum(signal in content for signal in signals) >= 2
        ):
            return True
    return False


def _install_tolerant_navigation(driver: Any, *, doi: str) -> None:
    """Continue after a load-event timeout when the DOI page is already usable."""

    executor = getattr(driver, "command_executor", None)
    set_timeout = getattr(executor, "set_timeout", None)
    if callable(set_timeout):
        set_timeout(180)
    original_get = driver.get

    def tolerant_get(url: str) -> Any:
        try:
            return original_get(url)
        except Exception:
            try:
                page_source = str(driver.page_source or "")
            except Exception:
                page_source = ""
            if len(page_source) >= 2_000 and doi.casefold() in page_source.casefold():
                return None
            raise

    driver.get = tolerant_get


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--publisher", default="")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(os.getenv("AUTOPLANNER_LITERATURE_DATAMINING_ROOT", r"D:\Autoplanner\shared\src")),
    )
    parser.add_argument("--chrome-major", type=int, default=0)
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
    if not args.force_refetch and _usable_fulltext_html(
        existing_artifacts,
        doi=args.doi,
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
    success = False
    page_status = "not_started"
    reason = ""
    try:
        from literature_datamining import config as ldm_config
        from literature_datamining.core.utils import initialize_webdriver

        # The shared package still carries a ChromeDriver 148 binary while the
        # workstation currently runs Chrome 150.  Do not let that stale bundled
        # executable override undetected-chromedriver's version-matched driver.
        ldm_config.WORKSPACE_ROOT = isolated_home
        ldm_config.CHROMEDRIVER_PATH = None
        if chrome_major:
            ldm_config.CHROME_VERSION = chrome_major

        spiders = ldm_config.get_spider_classes()
        spider_class = spiders.get(publisher)
        if spider_class is None:
            raise RuntimeError(f"publisher_adapter_unavailable:{publisher}")
        profile = output_dir / "browser-profile"
        downloads = output_dir / "browser-downloads"
        profile.mkdir(parents=True, exist_ok=True)
        downloads.mkdir(parents=True, exist_ok=True)
        driver = initialize_webdriver(str(profile), str(downloads), extension_path=None)
        _install_tolerant_navigation(driver, doi=args.doi)
        article_dir = output_dir / "article"
        article_dir.mkdir(parents=True, exist_ok=True)
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
        if article_data:
            (article_dir / "article-data.json").write_text(
                json.dumps(article_data, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
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
    accepted = bool((success and artifacts) or content_verified)
    receipt = {
        "schema_version": "authorized_literature_fetch.v1",
        "accepted": accepted,
        "doi": args.doi,
        "publisher": publisher,
        "chrome_major": chrome_major,
        "page_status": page_status,
        "reason": reason if not accepted else "",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "article_summary": {
            "title": article_data.get("title"),
            "full_text_section_count": len(article_data.get("full_text") or []),
            "figure_count": len(article_data.get("figures") or []),
            "supplementary_count": len(article_data.get("supplementary_materials") or []),
        },
        "semantics": {
            "publisher_spider_code_reused": True,
            "prior_source_documents_not_reused": True,
            "all_mutable_state_isolated_under_output_dir": True,
            "source_artifacts_require_host_hash_binding_and_extraction": True,
            "content_verified_after_legacy_status_mismatch": bool(
                content_verified and not success
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
