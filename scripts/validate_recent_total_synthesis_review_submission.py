#!/usr/bin/env python3
"""Validate expert review submissions and optionally merge them into the ledger.

Validation is read-only by default.  ``--merge`` is intended for the dataset
administrator after an expert returns a completed JSON file.  It appends reviews
atomically; it does not itself admit a paper, structure, route, or planner target.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any


SUBMISSION_SCHEMA = "recent_total_synthesis_review_submission.v1"
LEDGER_SCHEMA = "recent_total_synthesis_review_decisions.v1"
REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
PLACEHOLDER_RE = re.compile(r"(?:TO_FILL|NOT_REVIEWED|REPLACE_ME)", re.IGNORECASE)


class SubmissionError(ValueError):
    """A concise, field-oriented error suitable for expert handoff."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Completed submission JSON.")
    parser.add_argument(
        "--dataset-dir",
        default="benchmarks/recent_total_synthesis",
        help="Repository-relative benchmark directory.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Atomically append validated rows to the authoritative review ledger.",
    )
    return parser.parse_args()


def load_builder(repo_root: Path) -> Any:
    path = repo_root / "scripts" / "build_recent_total_synthesis_benchmark.py"
    spec = importlib.util.spec_from_file_location("recent_total_synthesis_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot_load_benchmark_builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_no_placeholder(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require_no_placeholder(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_no_placeholder(child, f"{field}[{index}]")
    elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
        raise SubmissionError(f"{field}: placeholder must be replaced")


def reviewer_fields(submission: dict[str, Any]) -> dict[str, Any]:
    reviewer = dict(submission.get("reviewer") or {})
    reviewer_id = str(reviewer.get("reviewer_id") or "")
    reviewed_at = str(reviewer.get("reviewed_at") or "")
    if not REVIEWER_ID_RE.fullmatch(reviewer_id):
        raise SubmissionError("reviewer.reviewer_id: use 2-64 letters, digits, dot, dash, or underscore")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SubmissionError("reviewer.reviewed_at: use ISO-8601 with timezone") from exc
    if timestamp.tzinfo is None:
        raise SubmissionError("reviewer.reviewed_at: timezone is required")
    if reviewer.get("attestation") is not True:
        raise SubmissionError("reviewer.attestation: must be true after independent source review")
    return {
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "reviewer_attestation": True,
    }


def validate_source_bindings(
    bindings: list[Any],
    *,
    builder: Any,
    repo_root: Path,
    field: str,
) -> list[dict[str, Any]]:
    if not bindings:
        raise SubmissionError(f"{field}: at least one source binding is required")
    normalized = []
    for index, value in enumerate(bindings):
        if not isinstance(value, dict):
            raise SubmissionError(f"{field}[{index}]: expected an object")
        require_no_placeholder(value, f"{field}[{index}]")
        try:
            normalized.append(
                builder.normalize_human_review_source_artifact(
                    dict(value), repo_root=repo_root
                )
            )
        except RuntimeError as exc:
            raise SubmissionError(f"{field}[{index}]: {exc}") from exc
    return normalized


def validate_submission(
    submission: dict[str, Any],
    *,
    repo_root: Path,
    dataset_dir: Path,
    ledger: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if submission.get("schema_version") != SUBMISSION_SCHEMA:
        raise SubmissionError(f"schema_version: expected {SUBMISSION_SCHEMA}")
    packet_type = str(submission.get("packet_type") or "")
    if packet_type not in {"paper_scope", "target_truth"}:
        raise SubmissionError("packet_type: expected paper_scope or target_truth")
    builder = load_builder(repo_root)
    reviewer = reviewer_fields(submission)
    papers = {str(row["paper_id"]): row for row in read_jsonl(dataset_dir / "papers.jsonl")}
    targets = {
        str(row["target_slot_id"]): row
        for row in read_jsonl(dataset_dir / "target_slots.jsonl")
    }

    rows: list[tuple[str, dict[str, Any]]] = []
    if packet_type == "paper_scope":
        paper_id = str(submission.get("paper_id") or "")
        paper = papers.get(paper_id)
        if paper is None:
            raise SubmissionError("paper_id: unknown canonical paper")
        expected_packet_id = f"paper-scope--{paper_id}--v1"
        if submission.get("packet_id") != expected_packet_id:
            raise SubmissionError(f"packet_id: expected {expected_packet_id}")
        if builder.normalize_doi(submission.get("doi")) != builder.normalize_doi(paper.get("doi")):
            raise SubmissionError("doi: does not match paper_id")
        decision = str(submission.get("decision") or "")
        allowed = {"primary", "conditional", "control", "exclude", "needs_revision"}
        if decision == "not_reviewed":
            raise SubmissionError("decision: no review selected")
        if decision not in allowed:
            raise SubmissionError(f"decision: expected one of {sorted(allowed)}")
        paper_target_ids = {
            target_id
            for target_id, target in targets.items()
            if str(target.get("paper_id") or "") == paper_id
        }
        target_ids = sorted({str(value) for value in submission.get("target_slot_ids") or []})
        if not set(target_ids).issubset(paper_target_ids):
            raise SubmissionError("target_slot_ids: contains a target not bound to this paper")
        if decision in {"primary", "conditional", "control"} and not target_ids:
            raise SubmissionError("target_slot_ids: admitted paper decision requires at least one target")
        evidence = validate_source_bindings(
            list(submission.get("evidence_locators") or []),
            builder=builder,
            repo_root=repo_root,
            field="evidence_locators",
        )
        notes = str(submission.get("reviewer_notes") or "").strip()
        if decision == "needs_revision" and not notes:
            raise SubmissionError("reviewer_notes: explain what must be revised")
        rows.append(
            (
                "paper_reviews",
                {
                    "paper_id": paper_id,
                    **reviewer,
                    "decision": decision,
                    "target_slot_ids": target_ids,
                    "evidence_locators": evidence,
                    "reviewer_notes": notes,
                    "packet_id": expected_packet_id,
                },
            )
        )
    else:
        target_id = str(submission.get("target_slot_id") or "")
        target = targets.get(target_id)
        if target is None:
            raise SubmissionError("target_slot_id: unknown canonical P0 target")
        expected_packet_id = f"target-truth--{target_id}--v1"
        if submission.get("packet_id") != expected_packet_id:
            raise SubmissionError(f"packet_id: expected {expected_packet_id}")
        if str(submission.get("paper_id") or "") != str(target.get("paper_id") or ""):
            raise SubmissionError("paper_id: does not match target_slot_id")
        if builder.normalize_doi(submission.get("doi")) != builder.normalize_doi(target.get("doi")):
            raise SubmissionError("doi: does not match target_slot_id")
        for subject, ledger_field, normalizer in (
            ("structure_review", "structure_reviews", builder.normalize_human_structure_review_record),
            ("route_review", "route_reviews", builder.normalize_human_route_review_record),
        ):
            review = dict(submission.get(subject) or {})
            decision = str(review.get("decision") or "not_reviewed")
            if decision == "not_reviewed":
                continue
            if decision not in {"accept", "reject", "needs_revision"}:
                raise SubmissionError(f"{subject}.decision: expected accept, reject, needs_revision, or not_reviewed")
            notes = str(review.get("reviewer_notes") or "").strip()
            row = {
                "target_slot_id": target_id,
                **reviewer,
                "decision": decision,
                "reviewer_notes": notes,
                "packet_id": expected_packet_id,
            }
            if decision == "accept":
                raw_record = dict(review.get("record") or {})
                require_no_placeholder(raw_record, f"{subject}.record")
                try:
                    row["record"] = normalizer(
                        raw_record,
                        target=target,
                        repo_root=repo_root,
                    )
                except RuntimeError as exc:
                    raise SubmissionError(f"{subject}.record: {exc}") from exc
            elif not notes:
                raise SubmissionError(f"{subject}.reviewer_notes: explain {decision}")
            rows.append((ledger_field, row))
        if not rows:
            raise SubmissionError("target_truth: structure and route are both not_reviewed")

    if ledger is not None:
        for field, row in rows:
            identity = (
                str(row.get("paper_id") or row.get("target_slot_id") or ""),
                str(row["reviewer_id"]),
            )
            for existing in ledger.get(field) or []:
                existing_identity = (
                    str(existing.get("paper_id") or existing.get("target_slot_id") or ""),
                    str(existing.get("reviewer_id") or ""),
                )
                if identity == existing_identity and existing != row:
                    raise SubmissionError(
                        f"{field}: reviewer already submitted a different review for this subject"
                    )
    return rows


def merge_rows(
    rows: list[tuple[str, dict[str, Any]]],
    *,
    ledger_path: Path,
    repo_root: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise SubmissionError(f"ledger schema: expected {LEDGER_SCHEMA}")
    appended = 0
    already_present = 0
    for field, row in rows:
        values = ledger.setdefault(field, [])
        if row in values:
            already_present += 1
        else:
            values.append(row)
            appended += 1

    builder = load_builder(repo_root)
    papers = read_jsonl(dataset_dir / "papers.jsonl")
    targets = read_jsonl(dataset_dir / "target_slots.jsonl")
    try:
        paper_states = builder.materialize_paper_review_states(papers, targets, ledger)
        builder.materialize_human_admissions(
            targets,
            ledger,
            repo_root=repo_root,
            paper_review_states=paper_states,
        )
    except RuntimeError as exc:
        raise SubmissionError(f"merged ledger would violate admission contract: {exc}") from exc

    temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, ledger_path)
    return {"appended": appended, "already_present": already_present}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    submission_path = Path(args.submission)
    if not submission_path.is_absolute():
        submission_path = repo_root / submission_path
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    ledger_path = dataset_dir / "curation_inputs" / "review_decisions.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    try:
        rows = validate_submission(
            submission,
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            ledger=ledger,
        )
        result: dict[str, Any] = {
            "valid": True,
            "submission": str(submission_path),
            "review_rows": len(rows),
            "ledger_fields": [field for field, _ in rows],
            "merged": False,
        }
        if args.merge:
            result.update(
                merge_rows(
                    rows,
                    ledger_path=ledger_path,
                    repo_root=repo_root,
                    dataset_dir=dataset_dir,
                )
            )
            result["merged"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except SubmissionError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
