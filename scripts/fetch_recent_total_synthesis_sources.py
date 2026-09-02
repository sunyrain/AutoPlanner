#!/usr/bin/env python3
"""Acquire openly accessible source packages for admitted benchmark papers.

The script is deliberately conservative: it fetches Europe PMC full-text XML and
direct OpenAlex OA PDF URLs only. Publisher pages, authenticated access, and supporting
information that lacks a public direct URL remain explicit follow-up work items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


USER_AGENT = "AutoPlanner-literature-benchmark-source-acquisition/0.1"
XLINK = "{http://www.w3.org/1999/xlink}href"
MIN_PUBLISHER_FULLTEXT_CHARACTERS = 2_000
ARTICLE_ARTIFACT_KINDS = {
    "repository_fulltext_xml",
    "repository_main_pdf",
    "open_access_main_pdf",
    "publisher_text_mining_fulltext",
    "authorized_publisher_fulltext_html",
    "authorized_publisher_fulltext_xml",
    "authorized_publisher_main_pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/paper_review_queue.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp/recent-total-synthesis-source-cache"),
    )
    parser.add_argument(
        "--authorized-cache-dir",
        type=Path,
        default=Path("tmp/authorized-literature-source-cache"),
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/source_package_receipts.jsonl"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--review-tier",
        action="append",
        default=[],
        help=(
            "Queue review tier to fetch; repeat for multiple tiers. "
            "Defaults to P0_source_extraction."
        ),
    )
    parser.add_argument(
        "--doi",
        action="append",
        default=[],
        help="Fetch only the selected DOI; repeat for more than one paper.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def portable_diagnostic(value: Any, repo_root: Path) -> str:
    """Remove machine-local repository prefixes from durable diagnostics."""

    text = str(value)
    root = repo_root.resolve()
    for prefix in {
        str(root).rstrip("\\/") + "\\",
        str(root).rstrip("\\/") + "/",
        root.as_posix().rstrip("/") + "/",
    }:
        text = text.replace(prefix, "")
    return text.replace("\\", "/")


def fetch(url: str, path: Path, *, offline: bool) -> tuple[bytes, bool]:
    if path.exists():
        return path.read_bytes(), True
    if offline:
        raise FileNotFoundError(f"offline source cache missing: {path}")
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=120) as response:
                payload = response.read(100_000_001)
            if len(payload) > 100_000_000:
                raise ValueError("source artifact exceeds 100 MB limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return payload, False
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    assert error is not None
    raise error


def xml_metadata(payload: bytes, expected_doi: str) -> dict[str, Any]:
    root = ET.fromstring(payload)
    article_dois = {
        "".join(node.itertext()).strip().casefold()
        for node in root.iter()
        if node.tag.split("}")[-1] == "article-id" and node.attrib.get("pub-id-type") == "doi"
    }
    if expected_doi.casefold() not in article_dois:
        raise ValueError(
            f"full-text DOI mismatch: expected {expected_doi}, found {sorted(article_dois)}"
        )
    licenses = []
    supplementary_links = set()
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag in {"license", "license-p"}:
            text = " ".join("".join(node.itertext()).split())
            if text:
                licenses.append(text)
        href = node.attrib.get(XLINK) or node.attrib.get("href") or ""
        if href and (
            tag in {"media", "supplementary-material"}
            or "suppl" in href.casefold()
            or href.casefold().endswith((".pdf", ".zip", ".doc", ".docx"))
        ):
            supplementary_links.add(href)
    return {
        "article_dois": sorted(article_dois),
        "licenses": sorted(set(licenses)),
        "supplementary_links": sorted(supplementary_links),
    }


def publisher_fulltext_metadata(
    payload: bytes, content_type: str, expected_doi: str
) -> dict[str, Any]:
    """Validate that a publisher endpoint returned article text, not metadata only.

    Crossref text-mining links can resolve successfully while an unauthenticated
    request receives an Elsevier ``coredata`` wrapper.  A DOI match therefore proves
    identity, but not acquisition of a source package.  Admission requires an
    explicit article-body element and enough body text to support route extraction.
    """

    decoded = payload.decode("utf-8", errors="ignore")
    if expected_doi.casefold() not in decoded.casefold():
        raise ValueError("text-mining artifact DOI mismatch")
    if "xml" not in content_type:
        text = " ".join(decoded.split())
        if len(text) < MIN_PUBLISHER_FULLTEXT_CHARACTERS:
            raise ValueError("text-mining endpoint returned metadata without article body")
        return {
            "article_body_elements": 0,
            "article_body_characters": len(text),
            "validation_basis": "plain_text_length",
        }

    root = ET.fromstring(payload)
    body_nodes = [
        node
        for node in root.iter()
        if node.tag.split("}")[-1].casefold() in {"body", "article-body", "originaltext"}
    ]
    body_texts = [" ".join("".join(node.itertext()).split()) for node in body_nodes]
    body_characters = max((len(text) for text in body_texts), default=0)
    if not body_nodes or body_characters < MIN_PUBLISHER_FULLTEXT_CHARACTERS:
        raise ValueError("text-mining endpoint returned metadata without article body")
    section_count = sum(
        1 for node in root.iter() if node.tag.split("}")[-1].casefold() in {"sec", "section"}
    )
    return {
        "article_body_elements": len(body_nodes),
        "article_body_characters": body_characters,
        "section_elements": section_count,
        "validation_basis": "xml_article_body",
    }


def publisher_structured_text_metadata(payload: bytes, expected_doi: str) -> dict[str, Any]:
    """Validate spider JSON as article text rather than a metadata wrapper."""

    document = json.loads(payload.decode("utf-8"))
    metadata = document.get("metadata") or {}
    observed_doi = str(metadata.get("doi") or "").strip().casefold()
    if observed_doi and observed_doi != expected_doi.casefold():
        raise ValueError("structured-text artifact DOI mismatch")
    sections = document.get("full_text") or []
    if not isinstance(sections, list):
        raise ValueError("structured-text artifact has invalid full_text")
    section_texts = [
        " ".join(str(section.get("text") or "").split())
        for section in sections
        if isinstance(section, dict)
    ]
    body_characters = sum(len(text) for text in section_texts)
    if body_characters < MIN_PUBLISHER_FULLTEXT_CHARACTERS:
        raise ValueError("structured-text artifact contains metadata without article body")
    return {
        "article_body_characters": body_characters,
        "section_elements": sum(bool(text) for text in section_texts),
        "validation_basis": "structured_full_text_sections",
    }


def _authorized_artifact_relative_path_allowed(relative: str) -> bool:
    """Accept canonical or immutable versioned article roots only."""

    parts = Path(relative.replace("\\", "/")).parts
    return bool(parts) and (
        parts[0] == "article"
        or (len(parts) >= 3 and parts[0] == "versions" and parts[2] == "article")
    )


def europe_pmc_doi_resolution(payload: bytes, expected_doi: str) -> dict[str, Any]:
    document = json.loads(payload.decode("utf-8"))
    results = list((document.get("resultList") or {}).get("result") or [])
    exact = [
        row for row in results if str(row.get("doi") or "").casefold() == expected_doi.casefold()
    ]
    if len(exact) > 1:
        raise ValueError("Europe PMC DOI lookup returned multiple exact records")
    row = exact[0] if exact else {}
    return {
        "doi_exact_match": bool(row),
        "pmid": str(row.get("pmid") or ""),
        "pmcid": str(row.get("pmcid") or ""),
        "is_open_access": str(row.get("isOpenAccess") or "") == "Y",
        "has_supplementary": str(row.get("hasSuppl") or "") == "Y",
    }


def supplementary_download_urls(metadata: dict[str, Any], doi: str) -> list[tuple[str, str]]:
    downloads: list[tuple[str, str]] = []
    for raw_link in metadata.get("supplementary_links") or []:
        link = str(raw_link or "").strip()
        lowered = link.casefold()
        if not link or not any(signal in lowered for signal in ("supp", "_si_", "moesm")):
            continue
        filename = link.rsplit("/", 1)[-1]
        if link.startswith(("https://", "http://")):
            downloads.append((link, filename))
        elif doi.casefold().startswith("10.1038/"):
            downloads.append(
                (
                    "https://static-content.springer.com/esm/"
                    f"art%3A{quote(doi, safe='')}/MediaObjects/{quote(filename)}",
                    filename,
                )
            )
    return list(dict.fromkeys(downloads))


def authorized_source_artifacts(
    repo_root: Path, authorized_root: Path, paper: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    paper_root = (authorized_root / paper["paper_id"]).resolve()
    receipt_path = paper_root / "authorized-literature-fetch.json"
    if not receipt_path.is_file():
        return [], []
    errors: list[str] = []
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"authorized_receipt:{type(exc).__name__}:{exc}"]
    if not receipt.get("accepted"):
        return [], [f"authorized_fetch:{receipt.get('reason') or 'not_accepted'}"]
    if str(receipt.get("doi") or "").casefold() != str(paper["doi"]).casefold():
        return [], ["authorized_receipt:DOI mismatch"]

    artifacts: list[dict[str, Any]] = []
    for raw in receipt.get("artifacts") or []:
        relative = str(raw.get("relative_path") or "")
        if not _authorized_artifact_relative_path_allowed(relative):
            continue
        path = (paper_root / relative).resolve()
        try:
            path.relative_to(paper_root)
        except ValueError:
            errors.append(f"authorized_artifact:path_outside_cache:{relative}")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != int(raw.get("size_bytes") or -1)
            or sha256_bytes(path.read_bytes()) != raw.get("sha256")
        ):
            errors.append(f"authorized_artifact:hash_or_size_mismatch:{relative}")
            continue

        lowered = relative.casefold()
        suffix = path.suffix.casefold()
        if "/supplementary_materials/" in lowered or "supplement" in path.name.casefold():
            kind = "supporting_information"
        elif suffix == ".html" and path.name.casefold() == "article.html":
            content = path.read_text(encoding="utf-8", errors="ignore").casefold()
            if len(content) < 2_000 or paper["doi"].casefold() not in content:
                errors.append(f"authorized_artifact:unverified_fulltext_html:{relative}")
                continue
            kind = "authorized_publisher_fulltext_html"
        elif suffix == ".xml" and "elsevier_api" in path.name.casefold():
            try:
                validation = publisher_fulltext_metadata(
                    path.read_bytes(), "text/xml", paper["doi"]
                )
            except ValueError as exc:
                errors.append(f"authorized_artifact:unverified_fulltext_xml:{relative}:{exc}")
                continue
            kind = "authorized_publisher_fulltext_xml"
        elif suffix == ".pdf":
            if not path.read_bytes()[:1024].lstrip().startswith(b"%PDF-"):
                errors.append(f"authorized_artifact:invalid_pdf:{relative}")
                continue
            kind = "authorized_publisher_main_pdf"
        elif suffix == ".json" and path.name.casefold() in {
            "article.json",
            "article-data.json",
        }:
            try:
                validation = publisher_structured_text_metadata(path.read_bytes(), paper["doi"])
            except ValueError as exc:
                errors.append(f"authorized_artifact:unverified_structured_text:{relative}:{exc}")
                continue
            kind = "authorized_publisher_structured_text"
        else:
            continue
        artifact = {
            "artifact_kind": kind,
            "cache_path": str(path.relative_to(repo_root)).replace("\\", "/"),
            "source_url": str(paper.get("source_url") or ""),
            "size_bytes": path.stat().st_size,
            "sha256": raw["sha256"],
            "cache_reused": True,
            "authorized_fetch_receipt": str(receipt_path.relative_to(repo_root)).replace("\\", "/"),
        }
        if "validation" in locals():
            artifact["fulltext_validation"] = validation
            del validation
        artifacts.append(artifact)
    return artifacts, errors


def load_queue(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    queue_path = (repo_root / args.queue).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    authorized_cache_dir = (repo_root / args.authorized_cache_dir).resolve()
    receipt_path = (repo_root / args.receipt_out).resolve()
    review_tiers = set(args.review_tier or ["P0_source_extraction"])
    default_receipt = Path(
        "benchmarks/recent_total_synthesis/source_package_receipts.jsonl"
    )
    if review_tiers != {"P0_source_extraction"} and args.receipt_out == default_receipt:
        raise ValueError(
            "non-P0 fetches require a separate --receipt-out so the P0 source ledger "
            "remains exact"
        )
    papers = [
        row for row in load_queue(queue_path) if row.get("review_tier") in review_tiers
    ]
    selected_dois = {str(value).casefold() for value in args.doi}
    if selected_dois:
        active_ids = {
            row["paper_id"]
            for row in papers
            if str(row.get("doi") or "").casefold() in selected_dois
        }
    elif args.limit > 0:
        active_ids = {row["paper_id"] for row in papers[: args.limit]}
    else:
        active_ids = {row["paper_id"] for row in papers}
    previous_receipts = {
        row["paper_id"]: row for row in (load_queue(receipt_path) if receipt_path.exists() else [])
    }

    receipts: list[dict[str, Any]] = []
    for row in papers:
        if row["paper_id"] not in active_ids and row["paper_id"] in previous_receipts:
            receipts.append(previous_receipts[row["paper_id"]])
            continue
        paper_dir = cache_dir / row["paper_id"]
        artifacts, errors = authorized_source_artifacts(repo_root, authorized_cache_dir, row)
        artifact_rejections = [
            error for error in errors if error.startswith("authorized_artifact:unverified_")
        ]
        errors = [error for error in errors if error not in artifact_rejections]
        metadata: dict[str, Any] = {}
        attempted = row["paper_id"] in active_ids
        pmcid = str(row.get("pmcid") or "")
        resolution_path = paper_dir / "europe-pmc-doi-lookup.json"
        if (
            attempted
            and not pmcid
            and row.get("doi")
            and (not args.offline or resolution_path.exists())
        ):
            query = urlencode(
                {
                    "query": f'DOI:"{row["doi"]}"',
                    "format": "json",
                    "resultType": "core",
                }
            )
            lookup_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query
            try:
                payload, reused = fetch(lookup_url, resolution_path, offline=args.offline)
                resolution = europe_pmc_doi_resolution(payload, row["doi"])
                metadata["europe_pmc_doi_resolution"] = {
                    **resolution,
                    "source_url": lookup_url,
                    "cache_path": str(resolution_path.relative_to(repo_root)).replace("\\", "/"),
                    "sha256": sha256_bytes(payload),
                    "cache_reused": reused,
                }
                pmcid = resolution["pmcid"]
            except Exception as exc:
                errors.append(f"europe_pmc_doi_lookup:{type(exc).__name__}:{exc}")
        if attempted and pmcid:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            try:
                payload, reused = fetch(
                    url, paper_dir / "europe-pmc-fulltext.xml", offline=args.offline
                )
                metadata.update(xml_metadata(payload, row["doi"]))
                artifacts.append(
                    {
                        "artifact_kind": "repository_fulltext_xml",
                        "cache_path": str(
                            (paper_dir / "europe-pmc-fulltext.xml").relative_to(repo_root)
                        ).replace("\\", "/"),
                        "source_url": url,
                        "size_bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                        "cache_reused": reused,
                    }
                )
            except Exception as exc:
                errors.append(f"europe_pmc_fulltext:{type(exc).__name__}:{exc}")

            pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            pdf_path = paper_dir / "europe-pmc-main.pdf"
            try:
                payload, reused = fetch(pdf_url, pdf_path, offline=args.offline)
                if not payload[:1024].lstrip().startswith(b"%PDF-"):
                    raise ValueError("Europe PMC PDF endpoint did not return a PDF")
                artifacts.append(
                    {
                        "artifact_kind": "repository_main_pdf",
                        "cache_path": str(pdf_path.relative_to(repo_root)).replace("\\", "/"),
                        "source_url": pdf_url,
                        "size_bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                        "cache_reused": reused,
                    }
                )
            except Exception as exc:
                if pdf_path.exists() and not pdf_path.read_bytes()[:1024].lstrip().startswith(
                    b"%PDF-"
                ):
                    pdf_path.unlink()
                errors.append(f"europe_pmc_pdf:{type(exc).__name__}:{exc}")

            for index, (supp_url, filename) in enumerate(
                supplementary_download_urls(metadata, row["doi"]), start=1
            ):
                suffix = Path(filename).suffix.casefold() or ".bin"
                supp_path = paper_dir / f"supporting-information-{index}{suffix}"
                try:
                    payload, reused = fetch(supp_url, supp_path, offline=args.offline)
                    if suffix == ".pdf" and not payload[:1024].lstrip().startswith(b"%PDF-"):
                        raise ValueError("supporting-information URL did not return PDF")
                    if suffix == ".zip" and not payload.startswith(b"PK"):
                        raise ValueError("supporting-information URL did not return ZIP")
                    artifacts.append(
                        {
                            "artifact_kind": "supporting_information",
                            "source_filename": filename,
                            "cache_path": str(supp_path.relative_to(repo_root)).replace("\\", "/"),
                            "source_url": supp_url,
                            "size_bytes": len(payload),
                            "sha256": sha256_bytes(payload),
                            "cache_reused": reused,
                        }
                    )
                except Exception as exc:
                    if supp_path.exists():
                        supp_path.unlink()
                    errors.append(f"supporting_information:{filename}:{type(exc).__name__}:{exc}")

        pdf_url = str(row.get("oa_pdf_url") or "")
        if (
            attempted
            and pdf_url
            and not any(
                artifact["artifact_kind"] in ARTICLE_ARTIFACT_KINDS for artifact in artifacts
            )
        ):
            try:
                payload, reused = fetch(
                    pdf_url, paper_dir / "open-access-main.pdf", offline=args.offline
                )
                if not payload[:1024].lstrip().startswith(b"%PDF-"):
                    raise ValueError("OA PDF endpoint did not return a PDF")
                artifacts.append(
                    {
                        "artifact_kind": "open_access_main_pdf",
                        "cache_path": str(
                            (paper_dir / "open-access-main.pdf").relative_to(repo_root)
                        ).replace("\\", "/"),
                        "source_url": pdf_url,
                        "size_bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                        "cache_reused": reused,
                    }
                )
            except Exception as exc:
                invalid = paper_dir / "open-access-main.pdf"
                if invalid.exists() and not invalid.read_bytes()[:1024].lstrip().startswith(
                    b"%PDF-"
                ):
                    invalid.unlink()
                errors.append(f"oa_pdf:{type(exc).__name__}:{exc}")

        if attempted and not artifacts:
            text_mining_links = [
                link
                for link in row.get("fulltext_links") or []
                if str(link.get("intended_application") or "").casefold() == "text-mining"
                and str(link.get("content_type") or "").casefold()
                in {"text/xml", "application/xml", "text/plain"}
            ]
            for index, link in enumerate(text_mining_links, start=1):
                url = str(link.get("url") or "")
                content_type = str(link.get("content_type") or "").casefold()
                suffix = ".xml" if "xml" in content_type else ".txt"
                path = paper_dir / f"publisher-text-mining-{index}{suffix}"
                try:
                    payload, reused = fetch(url, path, offline=args.offline)
                    prefix = payload[:2048].lstrip().lower()
                    if b"<html" in prefix or b"<!doctype html" in prefix:
                        raise ValueError("text-mining endpoint returned HTML")
                    fulltext_metadata = publisher_fulltext_metadata(
                        payload, content_type, row["doi"]
                    )
                    artifacts.append(
                        {
                            "artifact_kind": "publisher_text_mining_fulltext",
                            "cache_path": str(path.relative_to(repo_root)).replace("\\", "/"),
                            "source_url": url,
                            "content_type": content_type,
                            "size_bytes": len(payload),
                            "sha256": sha256_bytes(payload),
                            "cache_reused": reused,
                            "fulltext_validation": fulltext_metadata,
                        }
                    )
                    break
                except Exception as exc:
                    if path.exists():
                        path.unlink()
                    errors.append(f"text_mining_fulltext:{type(exc).__name__}:{exc}")

        errors = [portable_diagnostic(error, repo_root) for error in errors]
        artifact_rejections = [
            portable_diagnostic(error, repo_root) for error in artifact_rejections
        ]
        acquired = bool(artifacts)
        has_article = any(
            artifact["artifact_kind"] in ARTICLE_ARTIFACT_KINDS for artifact in artifacts
        )
        has_supporting_information = any(
            artifact["artifact_kind"] == "supporting_information" for artifact in artifacts
        )
        optional_source_misses: list[str] = []
        if has_supporting_information:
            optional_source_misses = [
                error
                for error in errors
                if error.startswith("supporting_information:")
                and ":FileNotFoundError:offline source cache missing:" in error
            ]
            errors = [error for error in errors if error not in optional_source_misses]
        if not attempted:
            status = "not_attempted_due_to_limit"
        elif acquired:
            status = "source_package_partially_acquired"
        elif args.offline:
            status = "not_in_offline_cache"
        else:
            status = "pending_authorized_or_manual_acquisition"
        receipts.append(
            {
                "schema_version": "recent_total_synthesis_source_package.v1",
                "paper_id": row["paper_id"],
                "doi": row["doi"],
                "source_access_class": row["source_access_class"],
                "status": status,
                "source_package_acquired": acquired,
                "source_package_completeness": (
                    "article_and_supporting_information"
                    if has_article and has_supporting_information
                    else "article_only"
                    if has_article
                    else "supporting_information_only"
                    if has_supporting_information
                    else "none"
                ),
                "route_evidence_admitted": False,
                "artifacts": artifacts,
                "repository_metadata": metadata,
                "artifact_rejections": artifact_rejections,
                "optional_source_misses": optional_source_misses,
                "errors": errors,
                "next_action": (
                    "extract_and_verify_article_and_supporting-information route evidence"
                    if acquired
                    else "use authorized publisher/OA fetch and obtain supporting information"
                ),
            }
        )
    write_jsonl(receipt_path, receipts)
    print(
        json.dumps(
            {
                "papers": len(papers),
                "review_tiers": sorted(review_tiers),
                "source_packages_acquired": sum(
                    bool(row["source_package_acquired"]) for row in receipts
                ),
                "receipt_path": str(receipt_path),
                "errors": sum(bool(row["errors"]) for row in receipts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
