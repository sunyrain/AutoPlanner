#!/usr/bin/env python3
"""Create resumable, non-admitting scope annotations for a review tier.

The output accelerates paper review but never changes the formal benchmark
denominator.  Each model call receives only literature metadata and writes to
an evaluator-side workspace that is never used as planner input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "recent_total_synthesis_scope_screen.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/paper_review_queue.jsonl"),
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/papers.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recent-total-synthesis-p1-scope-screen"),
    )
    parser.add_argument("--review-tier", default="P1_scope_review")
    parser.add_argument("--reviewer-id", default="reviewer-a")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def response_schema() -> dict[str, Any]:
    record = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "paper_id": {"type": "string"},
            "disposition": {
                "type": "string",
                "enum": [
                    "likely_primary",
                    "conditional_formal",
                    "conditional_noncore",
                    "control",
                    "exclude",
                    "needs_source",
                ],
            },
            "completed_synthesis": {"type": "boolean"},
            "target_identity": {
                "type": "string",
                "enum": ["exact_complete", "partial", "absent", "ambiguous"],
            },
            "target_names": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence_basis": {
                "type": "string",
                "enum": ["title", "title_and_abstract"],
            },
            "reason": {"type": "string"},
        },
        "required": [
            "paper_id",
            "disposition",
            "completed_synthesis",
            "target_identity",
            "target_names",
            "evidence_basis",
            "reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
            "records": {"type": "array", "items": record},
        },
        "required": ["schema_version", "records"],
    }


def screening_prompt(*, reviewer_id: str, records: list[dict[str, Any]]) -> str:
    evidence = [
        {
            "paper_id": row["paper_id"],
            "doi": row["doi"],
            "title": row["title"],
            "abstract": row["abstract"],
            "journal": row["journal"],
            "publication_date": row["publication_date"],
        }
        for row in records
    ]
    return "\n".join(
        [
            f"You are independent literature scope screener {reviewer_id}.",
            "Classify every supplied record for a benchmark of completed syntheses of discrete small-molecule natural products.",
            "likely_primary: the metadata affirmatively reports a completed synthesis of one or more discrete natural products.",
            "conditional_formal: only a formal synthesis or route improvement is established.",
            "conditional_noncore: peptide, glycan, polymer, material, or other excluded modality despite a completed synthesis.",
            "control: a drug, endogenous metabolite, analogue, or method-scope demonstration rather than a primary natural-product total-synthesis paper.",
            "exclude: review, correction, secondary report, unfinished studies, or unrelated work.",
            "needs_source: metadata do not establish completion, scope, or exact target identity.",
            "List only completed benchmark target names explicitly supported by title or abstract. Do not invent members hidden behind phrases such as 'alkaloids', 'derivatives', or letter ranges. If a letter range is explicit but its individual names are not enumerated, retain the literal range as one unresolved name and set target_identity=partial.",
            "Do not count intermediates, analogues, epimers made only for assignment, failed targets, or targets mentioned only as future/formal endpoints.",
            "Keep reason to one short evidence-based sentence. Return exactly one record for every input paper_id, in input order.",
            "These annotations are preliminary and have no admission authority. Do not browse or use outside knowledge.",
            "LiteratureMetadataBatch:",
            json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def valid_payload(payload: dict[str, Any], expected_ids: list[str]) -> bool:
    rows = list(payload.get("records") or [])
    return bool(
        payload.get("schema_version") == SCHEMA_VERSION
        and [str(row.get("paper_id") or "") for row in rows] == expected_ids
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("--batch-size and --workers must be positive")
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex executable not found")

    queue = [
        row
        for row in read_jsonl((REPO_ROOT / args.queue).resolve())
        if row.get("review_tier") == args.review_tier
    ]
    if args.limit > 0:
        queue = queue[: args.limit]
    papers = {
        str(row["paper_id"]): row
        for row in read_jsonl((REPO_ROOT / args.papers).resolve())
    }
    inputs = [
        {
            "paper_id": str(row["paper_id"]),
            "doi": str(row.get("doi") or ""),
            "title": str(row.get("title") or ""),
            "abstract": str(papers.get(str(row["paper_id"]), {}).get("abstract") or ""),
            "journal": str(row.get("journal") or ""),
            "publication_date": str(row.get("publication_date") or ""),
        }
        for row in queue
    ]
    batches = [
        inputs[index : index + args.batch_size]
        for index in range(0, len(inputs), args.batch_size)
    ]
    output_root = (REPO_ROOT / args.output_dir / args.reviewer_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    schema_path = output_root / "response-schema.json"
    write_json(schema_path, response_schema())

    def run_batch(index: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        batch_root = output_root / f"batch-{index + 1:03d}"
        batch_root.mkdir(parents=True, exist_ok=True)
        prompt = screening_prompt(reviewer_id=args.reviewer_id, records=records)
        digest = sha256_json(
            {
                "prompt": prompt,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "schema": response_schema(),
            }
        )
        result_path = batch_root / "screening-result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_sha256") == digest and existing.get("terminal"):
                return existing

        output_path = batch_root / "model-output.json"
        event_path = batch_root / "model-events.jsonl"
        command = [
            codex,
            "exec",
            "--cd",
            str(batch_root),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            args.model,
            "-c",
            f'model_reasoning_effort="{args.reasoning_effort}"',
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout_s,
                check=False,
            )
            event_path.write_text(completed.stdout, encoding="utf-8")
            (batch_root / "model-stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            payload = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.is_file()
                else {}
            )
            expected_ids = [str(row["paper_id"]) for row in records]
            status = (
                "accepted"
                if completed.returncode == 0 and valid_payload(payload, expected_ids)
                else "invalid"
            )
            result = {
                "schema_version": "recent_total_synthesis_scope_screen_batch.v1",
                "batch_index": index + 1,
                "input_sha256": digest,
                "input_count": len(records),
                "status": status,
                "terminal": status == "accepted",
                "return_code": completed.returncode,
                "records": list(payload.get("records") or []) if status == "accepted" else [],
            }
        except subprocess.TimeoutExpired as exc:
            event_path.write_text(str(exc.stdout or ""), encoding="utf-8")
            (batch_root / "model-stderr.log").write_text(
                str(exc.stderr or ""), encoding="utf-8"
            )
            result = {
                "schema_version": "recent_total_synthesis_scope_screen_batch.v1",
                "batch_index": index + 1,
                "input_sha256": digest,
                "input_count": len(records),
                "status": "timeout",
                "terminal": False,
                "return_code": None,
                "records": [],
            }
        write_json(result_path, result)
        return result

    batch_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_batch, index, records): index
            for index, records in enumerate(batches)
        }
        for future in as_completed(futures):
            result = future.result()
            batch_results.append(result)
            print(
                json.dumps(
                    {
                        "batch": result["batch_index"],
                        "batches": len(batches),
                        "status": result["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    batch_results.sort(key=lambda row: int(row["batch_index"]))
    annotations = [
        record
        for batch in batch_results
        if batch.get("status") == "accepted"
        for record in batch.get("records") or []
    ]
    write_jsonl(output_root / "screening-annotations.jsonl", annotations)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "reviewer_id": args.reviewer_id,
        "review_tier": args.review_tier,
        "paper_count": len(inputs),
        "annotated_paper_count": len(annotations),
        "batch_count": len(batches),
        "accepted_batch_count": sum(
            row.get("status") == "accepted" for row in batch_results
        ),
        "status_counts": dict(
            sorted(Counter(str(row.get("disposition") or "") for row in annotations).items())
        ),
        "completed_synthesis_count": sum(
            bool(row.get("completed_synthesis")) for row in annotations
        ),
        "enumerated_target_name_count": sum(
            len(row.get("target_names") or []) for row in annotations
        ),
        "admission_authority": False,
        "claim_boundary": (
            "AI-assisted title/abstract screen only; source review and independent "
            "adjudication remain required."
        ),
    }
    write_json(output_root / "screening-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if len(annotations) == len(inputs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
