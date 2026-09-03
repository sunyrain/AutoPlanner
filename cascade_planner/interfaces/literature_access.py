"""Authorized PDF access lifecycle shared by literature connectors."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from cascade_planner.harness.local_pdf_proxy import (
    build_pdf_request,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
    write_pdf_request_queue,
)


def queue_authorized_pdf_request(
    candidate: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    proxy_root: Path,
    reason: str,
    source_ref: str,
) -> dict[str, Any]:
    try:
        row = build_pdf_request(
            {
                **dict(candidate),
                "source_ref": source_ref,
                "material_type": "paper_si",
            },
            case_id=str(request.get("run_id") or request.get("target_name") or ""),
            source_ref=source_ref,
            reason=reason,
            requested_by="v4_literature_evidence",
        )
        receipt = write_pdf_request_queue(
            [row],
            local_pdf_proxy_request_queue_path(proxy_root),
            append=True,
            dedupe=True,
        )
    except (OSError, ValueError):
        return {}
    return {
        "request_id": str(row.get("request_id") or ""),
        "queue_path": str(receipt.get("path") or ""),
        "status": "queued",
        "credentials_stored": False,
    }


def run_authorized_browser_fetch(
    *,
    proxy_root: Path,
    case_id: str,
    source_refs: tuple[str, ...],
    max_items: int,
    timeout_s: float,
    headless: bool = True,
) -> dict[str, Any]:
    """Consume queued source requests with the existing authorized browser tool.

    This is deliberately a subprocess boundary: the downloader owns its
    persistent browser/CDP lifecycle while the evidence connector remains a
    deterministic consumer of the resulting hash-bound manifest.
    """

    queue_path = local_pdf_proxy_request_queue_path(proxy_root)
    if not queue_path.is_file() or max_items <= 0 or not source_refs:
        return {
            "schema_version": "authorized_browser_autofetch.v1",
            "status": "not_needed",
            "reason": "authorized_browser_queue_empty",
            "processed_count": 0,
            "downloaded_count": 0,
        }
    script = Path(__file__).resolve().parents[2] / "scripts" / "browser_pdf_fetch.py"
    if not script.is_file():
        return {
            "schema_version": "authorized_browser_autofetch.v1",
            "status": "unavailable",
            "reason": "authorized_browser_fetch_script_missing",
            "processed_count": 0,
            "downloaded_count": 0,
        }
    command = [
        sys.executable,
        str(script),
        "--output-dir",
        str(proxy_root),
        "--case-id",
        case_id,
        "--max-items",
        str(max_items),
        "--timeout-ms",
        str(max(5_000, int(timeout_s * 1_000))),
    ]
    for source_ref in source_refs:
        command.extend(["--source-ref", source_ref])
    if headless:
        command.append("--headless")
    process_timeout_s = max(30.0, timeout_s * max_items + 45.0)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=process_timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return {
            "schema_version": "authorized_browser_autofetch.v1",
            "status": "timed_out",
            "reason": f"authorized_browser_fetch_timeout:{process_timeout_s:g}s",
            "processed_count": 0,
            "downloaded_count": 0,
        }
    try:
        payload = json.loads((completed.stdout or "").strip())
    except (TypeError, ValueError):
        payload = {}
    downloaded_count = int(payload.get("downloaded_count") or 0)
    return {
        "schema_version": "authorized_browser_autofetch.v1",
        "status": (
            "completed"
            if downloaded_count
            else "unresolved"
            if completed.returncode in {0, 2}
            else "failed"
        ),
        "reason": (
            ""
            if payload
            else " ".join(
                (completed.stderr or completed.stdout or "browser_fetch_failed").split()
            )[:1_000]
        ),
        "exit_code": int(completed.returncode),
        "processed_count": int(payload.get("processed_count") or 0),
        "downloaded_count": downloaded_count,
        "failed_count": int(payload.get("failed_count") or 0),
        "results": [
            dict(row)
            for row in payload.get("results") or []
            if isinstance(row, Mapping)
        ],
        "manifest_path": str(local_pdf_proxy_download_manifest_path(proxy_root)),
        "credentials_stored": False,
        "headless": bool(headless),
    }


def authorized_proxy_artifact(
    candidate: Mapping[str, Any],
    *,
    proxy_root: Path,
    source_ref: str,
    doi: str,
) -> dict[str, Any]:
    """Return the newest identity-matched, locally authorized source artifact.

    The local provider may freeze publisher HTML, supplementary material, or a
    PDF.  Keeping the receipt row intact lets the materializer verify both the
    source identity and the content hash before extracting evidence.
    """

    manifest = local_pdf_proxy_download_manifest_path(proxy_root)
    if not manifest.is_file():
        return {}
    identities = {
        doi.lower(),
        source_ref.lower(),
        str(candidate.get("pdf_url") or candidate.get("url") or "").strip().lower(),
    } - {""}
    matched: dict[str, Any] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if row.get("accepted") is not True or row.get("status") != "downloaded":
            continue
        row_ids = {
            str(row.get("doi") or "").lower(),
            str(row.get("source_ref") or "").lower(),
            str(row.get("url") or "").lower(),
        }
        if not identities & row_ids:
            continue
        paths = [
            str(row.get("html_path") or ""),
            str(row.get("structured_path") or ""),
            str(row.get("pdf_path") or ""),
        ]
        if any(
            path and Path(path).expanduser().resolve().is_file()
            for path in paths
        ):
            matched = dict(row)
    return matched


def authorized_proxy_pdf(
    candidate: Mapping[str, Any],
    *,
    proxy_root: Path,
    source_ref: str,
    doi: str,
) -> str:
    row = authorized_proxy_artifact(
        candidate,
        proxy_root=proxy_root,
        source_ref=source_ref,
        doi=doi,
    )
    path = Path(str(row.get("pdf_path") or "")).expanduser().resolve()
    return str(path) if path.is_file() else ""


def pending_source(
    candidate: Mapping[str, Any],
    *,
    proxy_request: Mapping[str, Any],
    source_ref: str,
    doi: str,
) -> dict[str, Any]:
    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": doi,
        "pmid": str(candidate.get("pmid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "acquisition_status": "queued_for_authorized_browser",
        "proxy_request_id": str(proxy_request.get("request_id") or ""),
        "visual_candidate_pages": [],
        "procedure_inventory": [],
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": 1,
        "semantics": {
            "metadata_only": True,
            "not_route_evidence": True,
            "resume_after_browser_download": True,
        },
    }


def downloaded_unextractable_source(
    candidate: Mapping[str, Any],
    *,
    source_ref: str,
    doi: str,
    reason: str,
) -> dict[str, Any]:
    """Represent downloaded bytes that failed deterministic extraction."""

    return {
        "source_kind": "paper_si",
        "source_ref": source_ref,
        "doi": doi,
        "pmid": str(candidate.get("pmid") or ""),
        "title": " ".join(str(candidate.get("title") or source_ref).split())[:1000],
        "acquisition_status": "downloaded_unextractable",
        "extraction_failure_reason": reason[:1_000],
        "visual_candidate_pages": [],
        "procedure_inventory": [],
        "exact_edge_ids": [],
        "exact_row_count": 0,
        "unresolved_edge_count": 1,
        "semantics": {
            "metadata_only": False,
            "not_route_evidence": True,
            "resume_after_browser_download": False,
            "download_completed_but_extraction_failed": True,
        },
    }


__all__ = [
    "authorized_proxy_artifact",
    "authorized_proxy_pdf",
    "downloaded_unextractable_source",
    "pending_source",
    "queue_authorized_pdf_request",
    "run_authorized_browser_fetch",
]
