#!/usr/bin/env python3
"""Resume publisher-authorized source acquisition for selected review tiers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from authorized_literature_fetch import (
    PUBLISHER_BY_PREFIX,
    _prepare_isolated_chromedriver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/paper_review_queue.jsonl"),
    )
    parser.add_argument(
        "--source-receipts",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/source_package_receipts.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp/authorized-literature-source-cache"),
    )
    parser.add_argument(
        "--batch-receipt",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/authorized_source_fetch_batch.jsonl"),
    )
    parser.add_argument(
        "--review-tier",
        action="append",
        default=[],
        help=(
            "Queue review tier to fetch; repeat for multiple tiers. "
            "Defaults to P0_source_extraction."
        ),
    )
    parser.add_argument("--chrome-major", type=int, default=0)
    parser.add_argument("--chrome-binary", type=Path)
    parser.add_argument(
        "--chromedriver",
        type=Path,
        help=(
            "Prevalidated driver to share across workers. When omitted with a positive "
            "--chrome-major, the coordinator prepares one shared driver once."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent isolated publisher fetches (default: 3).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--doi", action="append", default=[])
    parser.add_argument("--force-refetch", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_batch_rows(
    queue: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    updated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge resumable updates without dropping audit rows for other selected papers."""

    by_paper = {str(row.get("paper_id") or ""): row for row in existing_rows if row.get("paper_id")}
    by_paper.update(
        {str(row.get("paper_id") or ""): row for row in updated_rows if row.get("paper_id")}
    )
    merged: list[dict[str, Any]] = []
    for index, paper in enumerate(queue, start=1):
        row = by_paper.get(str(paper.get("paper_id") or ""))
        if row:
            merged.append({**row, "batch_index": index})
    return merged


def publisher_for(doi: str) -> str:
    lowered = doi.casefold()
    return next(
        (
            publisher
            for prefix, publisher in PUBLISHER_BY_PREFIX.items()
            if lowered.startswith(prefix)
        ),
        "",
    )


def load_authorized_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def portable_path(path: Path, root: Path) -> str:
    try:
        displayed = path.relative_to(root)
    except ValueError:
        displayed = path
    return str(displayed).replace("\\", "/")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    repo_root = Path(__file__).resolve().parents[1]
    review_tiers = set(args.review_tier or ["P0_source_extraction"])
    default_batch_receipt = Path(
        "benchmarks/recent_total_synthesis/authorized_source_fetch_batch.jsonl"
    )
    if review_tiers != {"P0_source_extraction"} and args.batch_receipt == default_batch_receipt:
        raise ValueError(
            "non-P0 fetches require a separate --batch-receipt so the P0 audit ledger "
            "remains exact"
        )
    all_queue = [
        row
        for row in read_jsonl((repo_root / args.queue).resolve())
        if row.get("review_tier") in review_tiers
    ]
    queue = list(all_queue)
    source_receipts = {
        row["paper_id"]: row for row in read_jsonl((repo_root / args.source_receipts).resolve())
    }
    selected = {str(doi).casefold() for doi in args.doi}
    if selected:
        queue = [row for row in queue if row["doi"].casefold() in selected]
    if args.limit > 0:
        queue = queue[: args.limit]

    cache_root = (repo_root / args.cache_dir).resolve()
    helper = repo_root / "scripts" / "authorized_literature_fetch.py"
    output = (repo_root / args.batch_receipt).resolve()
    existing_batch_rows = read_jsonl(output)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], dict[str, Any], Path, Path, str, bool]] = []
    for index, paper in enumerate(queue, start=1):
        doi = paper["doi"]
        paper_root = cache_root / paper["paper_id"]
        receipt_path = paper_root / "authorized-literature-fetch.json"
        existing = load_authorized_receipt(receipt_path)
        source = source_receipts.get(paper["paper_id"], {})
        publisher = publisher_for(doi)
        base = {
            "schema_version": "recent_total_synthesis_authorized_fetch_batch.v1",
            "batch_index": index,
            "paper_id": paper["paper_id"],
            "doi": doi,
            "publisher": publisher,
            "authorized_receipt_path": portable_path(receipt_path, repo_root),
        }
        if source.get("source_package_completeness") == "article_and_supporting_information":
            rows.append({**base, "status": "already_complete", "accepted": True})
            continue
        if existing.get("accepted") and not args.force_refetch:
            rows.append(
                {
                    **base,
                    "status": "accepted_cached",
                    "accepted": True,
                    "artifact_count": int(existing.get("artifact_count") or 0),
                }
            )
            continue
        if not publisher:
            rows.append(
                {
                    **base,
                    "status": "publisher_adapter_unavailable",
                    "accepted": False,
                }
            )
            continue

        pending.append(
            (
                index,
                paper,
                base,
                paper_root,
                receipt_path,
                publisher,
                bool(existing.get("accepted")),
            )
        )

    shared_driver: Path | None = None
    if pending and args.chromedriver:
        shared_driver = (repo_root / args.chromedriver).resolve()
        if not shared_driver.is_file():
            raise FileNotFoundError(f"chromedriver not found: {shared_driver}")
    elif pending and args.chrome_major > 0:
        shared_driver = _prepare_isolated_chromedriver(
            cache_root / ".shared-drivers" / f"chrome-{args.chrome_major}",
            args.chrome_major,
        )

    def fetch_one(
        item: tuple[int, dict[str, Any], dict[str, Any], Path, Path, str, bool],
    ) -> tuple[int, dict[str, Any]]:
        index, paper, base, paper_root, receipt_path, publisher, existing_accepted = item
        doi = str(paper["doi"])
        attempt_root: Path | None = None
        output_root = paper_root
        output_receipt_path = receipt_path
        if args.force_refetch and existing_accepted:
            attempts_root = paper_root / ".attempts"
            attempts_root.mkdir(parents=True, exist_ok=True)
            attempt_root = Path(tempfile.mkdtemp(prefix="refetch-", dir=str(attempts_root)))
            output_root = attempt_root
            output_receipt_path = attempt_root / "authorized-literature-fetch.json"
        command = [
            sys.executable,
            str(helper),
            "--doi",
            doi,
            "--output-dir",
            str(output_root),
            "--publisher",
            publisher,
        ]
        if args.chrome_major > 0:
            command.extend(["--chrome-major", str(args.chrome_major)])
        if args.chrome_binary:
            chrome_binary = (repo_root / args.chrome_binary).resolve()
            command.extend(["--chrome-binary", str(chrome_binary)])
        if shared_driver is not None:
            command.extend(["--chromedriver", str(shared_driver)])
        if args.force_refetch:
            command.append("--force-refetch")
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout_seconds,
                check=False,
            )
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "batch-fetch.stdout.log").write_text(result.stdout, encoding="utf-8")
            (output_root / "batch-fetch.stderr.log").write_text(result.stderr, encoding="utf-8")
            receipt = load_authorized_receipt(output_receipt_path)
            attempt_accepted = bool(receipt.get("accepted"))
            reported_attempt_receipt_path = output_receipt_path
            if attempt_root is not None and attempt_accepted:
                versions_root = paper_root / "versions"
                versions_root.mkdir(parents=True, exist_ok=True)
                version_root = versions_root / attempt_root.name
                attempt_root.replace(version_root)
                reported_attempt_receipt_path = version_root / "authorized-literature-fetch.json"
                prefix = version_root.relative_to(paper_root).as_posix()
                for artifact in receipt.get("artifacts") or []:
                    relative = str(artifact.get("relative_path") or "")
                    if relative:
                        artifact["relative_path"] = f"{prefix}/{relative}"
                receipt_tmp = receipt_path.with_suffix(".json.tmp")
                receipt_tmp.write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                receipt_tmp.replace(receipt_path)
            preserved = bool(attempt_root is not None and not attempt_accepted)
            accepted = attempt_accepted or preserved
            return index, {
                **base,
                "status": (
                    "accepted"
                    if attempt_accepted
                    else "refetch_failed_preserved"
                    if preserved
                    else "fetch_failed"
                ),
                "accepted": accepted,
                "latest_attempt_accepted": attempt_accepted,
                "preserved_previous_accepted_source": preserved,
                "return_code": result.returncode,
                "artifact_count": int(receipt.get("artifact_count") or 0),
                "page_status": str(receipt.get("page_status") or ""),
                "reason": str(receipt.get("reason") or ""),
                "attempt_receipt_path": (
                    portable_path(reported_attempt_receipt_path, repo_root)
                    if attempt_root is not None
                    else ""
                ),
            }
        except subprocess.TimeoutExpired as exc:
            output_root.mkdir(parents=True, exist_ok=True)
            if exc.stdout:
                (output_root / "batch-fetch.stdout.log").write_text(
                    str(exc.stdout), encoding="utf-8"
                )
            if exc.stderr:
                (output_root / "batch-fetch.stderr.log").write_text(
                    str(exc.stderr), encoding="utf-8"
                )
            preserved = bool(attempt_root is not None)
            return index, {
                **base,
                "status": ("refetch_timeout_preserved" if preserved else "fetch_timeout"),
                "accepted": preserved,
                "latest_attempt_accepted": False,
                "preserved_previous_accepted_source": preserved,
                "timeout_seconds": args.timeout_seconds,
                "attempt_receipt_path": (
                    portable_path(output_receipt_path, repo_root)
                    if attempt_root is not None
                    else ""
                ),
            }

    if rows:
        write_jsonl(output, merge_batch_rows(all_queue, existing_batch_rows, rows))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, item): item for item in pending}
        completed = 0
        for future in as_completed(futures):
            index, row = future.result()
            rows.append(row)
            completed += 1
            write_jsonl(
                output,
                merge_batch_rows(all_queue, existing_batch_rows, rows),
            )
            print(
                json.dumps(
                    {
                        "progress": f"{completed}/{len(pending)} fetches",
                        "queue_index": f"{index}/{len(queue)}",
                        "doi": row["doi"],
                        "status": row["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    persisted_rows = merge_batch_rows(all_queue, existing_batch_rows, rows)
    write_jsonl(output, persisted_rows)
    print(
        json.dumps(
            {
                "papers_processed": len(rows),
                "review_tiers": sorted(review_tiers),
                "audit_rows": len(persisted_rows),
                "accepted": sum(bool(row.get("accepted")) for row in rows),
                "failed": sum(row["status"] in {"fetch_failed", "fetch_timeout"} for row in rows),
                "unsupported": sum(
                    row["status"] == "publisher_adapter_unavailable" for row in rows
                ),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
