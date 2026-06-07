#!/usr/bin/env python3
"""Queue and fetch PDFs through a local authorized machine.

Typical split workflow:

1. On the server, write requests:
   python scripts/local_pdf_proxy.py request --doi 10.1021/example

2. Sync the work directory to your laptop, connect school VPN/library access,
   then fetch:
   python scripts/local_pdf_proxy.py fetch

3. Sync the work directory back to the server. The harness reads the returned
   pdf_download_manifest.jsonl and local pdf paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.local_pdf_proxy import (  # noqa: E402
    build_pdf_request,
    download_pdf_requests,
    load_pdf_requests,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_manifest_entry,
    local_pdf_proxy_pdfs_dir,
    local_pdf_proxy_request_queue_path,
    local_pdf_proxy_work_dir,
    requests_from_source_material_locator_pack,
    summarize_pdf_download_manifest,
    write_pdf_request_queue,
)


DEFAULT_OUTPUT_DIR = ROOT / "results" / "shared" / "local_pdf_proxy_run"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Run/output root. The proxy work dir is output-dir/evidence/local_pdf_proxy.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create the local PDF proxy work directory.")
    init.set_defaults(func=cmd_init)

    request = sub.add_parser("request", help="Append DOI/URL requests to the queue.")
    request.add_argument("--doi", action="append", default=[], help="DOI to request. Repeatable.")
    request.add_argument("--url", action="append", default=[], help="Landing/PDF URL to request. Repeatable.")
    request.add_argument("--input", help="Optional JSONL/text input with one DOI/URL/request per line.")
    request.add_argument("--source-material-locator", help="source_material_locator_pack.json to convert into requests.")
    request.add_argument("--case-id", default="", help="Case id to attach to generated requests.")
    request.add_argument("--reason", default="local_pdf_proxy_request", help="Reason/provenance label.")
    request.add_argument(
        "--content-scope",
        default="",
        choices=["", "article", "si", "pdf", "landing_page", "unknown"],
        help="Requested content scope for access-audit matching.",
    )
    request.add_argument("--replace", action="store_true", help="Replace the queue instead of appending.")
    request.set_defaults(func=cmd_request)

    fetch = sub.add_parser("fetch", help="Run on your local authorized machine and download queued PDFs.")
    fetch.add_argument("--timeout-s", type=float, default=30.0)
    fetch.add_argument("--max-items", type=int)
    fetch.add_argument("--max-bytes", type=int, default=80 * 1024 * 1024)
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument("--delay-s", type=float, default=1.0)
    fetch.add_argument("--proxy", help="Optional proxy URL; sets HTTPS_PROXY and HTTP_PROXY for this run.")
    fetch.set_defaults(func=cmd_fetch)

    status = sub.add_parser("status", help="Print queue/result summary.")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    return int(args.func(args))


def paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    return {
        "output_dir": output_dir,
        "work_dir": local_pdf_proxy_work_dir(output_dir),
        "queue": local_pdf_proxy_request_queue_path(output_dir),
        "manifest": local_pdf_proxy_download_manifest_path(output_dir),
        "pdf_dir": local_pdf_proxy_pdfs_dir(output_dir),
    }


def cmd_init(args: argparse.Namespace) -> int:
    p = paths(args)
    p["pdf_dir"].mkdir(parents=True, exist_ok=True)
    p["queue"].parent.mkdir(parents=True, exist_ok=True)
    if not p["queue"].exists():
        p["queue"].write_text("", encoding="utf-8")
    print(json.dumps(local_pdf_proxy_manifest_entry(None, output_dir=p["output_dir"]), indent=2, ensure_ascii=False))
    return 0


def cmd_request(args: argparse.Namespace) -> int:
    p = paths(args)
    requests: list[dict[str, Any]] = []
    for doi in args.doi:
        requests.append(
            build_pdf_request(
                {"doi": doi, "content_scope": args.content_scope},
                case_id=args.case_id,
                reason=args.reason,
                requested_by="local_pdf_proxy_cli",
            )
        )
    for url in args.url:
        requests.append(
            build_pdf_request(
                {"url": url, "content_scope": args.content_scope},
                case_id=args.case_id,
                reason=args.reason,
                requested_by="local_pdf_proxy_cli",
            )
        )
    if args.input:
        requests.extend(_requests_from_input(Path(args.input), case_id=args.case_id, reason=args.reason))
    if args.source_material_locator:
        requests.extend(
            requests_from_source_material_locator_pack(
                args.source_material_locator,
                case_id=args.case_id,
                reason=args.reason,
            )
        )
    if not requests:
        raise SystemExit("No requests supplied. Use --doi, --url, --input, or --source-material-locator.")
    result = write_pdf_request_queue(requests, p["queue"], append=not args.replace, dedupe=True)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if args.proxy:
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["HTTP_PROXY"] = args.proxy
    p = paths(args)
    result = download_pdf_requests(
        queue_path=p["queue"],
        pdf_dir=p["pdf_dir"],
        manifest_path=p["manifest"],
        timeout_s=args.timeout_s,
        max_items=args.max_items,
        max_bytes=args.max_bytes,
        overwrite=args.overwrite,
        delay_s=args.delay_s,
    )
    print(json.dumps(_compact_download_result(result), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["failed_count"] == 0 else 2


def cmd_status(args: argparse.Namespace) -> int:
    p = paths(args)
    data = local_pdf_proxy_manifest_entry(
        {
            "request_count": len(load_pdf_requests(p["queue"])),
            "result_summary": summarize_pdf_download_manifest(p["manifest"]),
        },
        output_dir=p["output_dir"],
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, sort_keys=True))
    else:
        print(f"work_dir: {data['sync_hint']['work_dir']}")
        print(f"queue: {data['request_queue_path']}")
        print(f"manifest: {data['download_manifest_path']}")
        print(f"pdf_dir: {data['pdf_dir']}")
        print(f"status: {data['status']}")
        print(f"requests: {data['request_count']}")
        print(f"results: {json.dumps(data['result_summary'], ensure_ascii=False, sort_keys=True)}")
    return 0


def _requests_from_input(path: Path, *, case_id: str, reason: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text) if text.startswith("{") else {"url": text}
            rows.append(
                build_pdf_request(
                    record,
                    case_id=case_id or str(record.get("case_id") or ""),
                    reason=reason,
                    requested_by="local_pdf_proxy_cli",
                )
            )
        except Exception as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return rows


def _compact_download_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact["results"] = [
        {
            "request_id": row.get("request_id"),
            "status": row.get("status"),
            "pdf_path": row.get("pdf_path", ""),
            "reason": row.get("reason", ""),
            "final_url": row.get("final_url", ""),
        }
        for row in result.get("results") or []
    ]
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
