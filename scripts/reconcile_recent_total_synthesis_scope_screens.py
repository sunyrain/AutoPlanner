#!/usr/bin/env python3
"""Reconcile two non-admitting metadata screens and project candidate slots."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewer-a",
        type=Path,
        default=Path(
            "results/recent-total-synthesis-p1-scope-screen-20260902/"
            "reviewer-a/screening-annotations.jsonl"
        ),
    )
    parser.add_argument(
        "--reviewer-b",
        type=Path,
        default=Path(
            "results/recent-total-synthesis-p1-scope-screen-20260902/"
            "reviewer-b/screening-annotations.jsonl"
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/paper_review_queue.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/curation_candidates/p1_scope"),
    )
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


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    return " ".join(re.findall(r"[a-z0-9+()α-ωΑ-Ω'-]+", value.casefold()))


def normalized_name_map(values: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        displayed = " ".join(str(value or "").split()).strip(" .;,")
        normalized = normalize_name(displayed)
        if displayed and normalized:
            result.setdefault(normalized, displayed)
    return result


def stable_slot_id(paper_id: str, target_name: str) -> str:
    digest = hashlib.sha256(
        f"{paper_id}\x1f{normalize_name(target_name)}".encode("utf-8")
    ).hexdigest()
    return f"candidate-target-slot-{digest[:16]}"


def reconcile(
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    queue: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    a_by_id = {str(row.get("paper_id") or ""): row for row in reviewer_a}
    b_by_id = {str(row.get("paper_id") or ""): row for row in reviewer_b}
    queue_by_id = {str(row.get("paper_id") or ""): row for row in queue}
    paper_ids = [
        str(row.get("paper_id") or "")
        for row in queue
        if str(row.get("paper_id") or "") in a_by_id
        and str(row.get("paper_id") or "") in b_by_id
    ]
    consensus: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        first = a_by_id[paper_id]
        second = b_by_id[paper_id]
        first_names = normalized_name_map(list(first.get("target_names") or []))
        second_names = normalized_name_map(list(second.get("target_names") or []))
        disposition_agrees = first.get("disposition") == second.get("disposition")
        completion_agrees = bool(first.get("completed_synthesis")) == bool(
            second.get("completed_synthesis")
        )
        names_agree = set(first_names) == set(second_names)
        identity_agrees = first.get("target_identity") == second.get("target_identity")
        fully_agreed = bool(
            disposition_agrees and completion_agrees and names_agree and identity_agrees
        )
        metadata = queue_by_id[paper_id]
        row = {
            "schema_version": "recent_total_synthesis_scope_consensus.v1",
            "paper_id": paper_id,
            "doi": str(metadata.get("doi") or ""),
            "title": str(metadata.get("title") or ""),
            "fully_agreed": fully_agreed,
            "disposition_agrees": disposition_agrees,
            "completion_agrees": completion_agrees,
            "target_names_agree": names_agree,
            "target_identity_agrees": identity_agrees,
            "consensus_disposition": (
                str(first.get("disposition") or "") if disposition_agrees else ""
            ),
            "consensus_completed_synthesis": (
                bool(first.get("completed_synthesis")) if completion_agrees else None
            ),
            "consensus_target_identity": (
                str(first.get("target_identity") or "") if identity_agrees else ""
            ),
            "consensus_target_names": (
                [first_names[key] for key in sorted(first_names)] if names_agree else []
            ),
            "reviewer_a": first,
            "reviewer_b": second,
            "admission_authority": False,
        }
        consensus.append(row)
        if not fully_agreed:
            disagreements.append(row)
            continue
        if (
            row["consensus_disposition"] != "likely_primary"
            or row["consensus_completed_synthesis"] is not True
            or row["consensus_target_identity"] != "exact_complete"
        ):
            continue
        for target_name in row["consensus_target_names"]:
            slots.append(
                {
                    "schema_version": "recent_total_synthesis_candidate_target_slot.v1",
                    "target_slot_id": stable_slot_id(paper_id, target_name),
                    "paper_id": paper_id,
                    "doi": row["doi"],
                    "publication_title": row["title"],
                    "slot_class": "primary_candidate",
                    "target_name": target_name,
                    "target_identity_status": "dual_ai_metadata_consensus_pending_source",
                    "target_smiles": "",
                    "input_status": "dual_ai_metadata_consensus_nonadmitting",
                    "formal_benchmark_eligible": False,
                    "required_next_action": (
                        "verify exact target identity and stereochemistry against article/SI"
                    ),
                }
            )
    return consensus, disagreements, slots


def main() -> int:
    args = parse_args()
    reviewer_a = read_jsonl((REPO_ROOT / args.reviewer_a).resolve())
    reviewer_b = read_jsonl((REPO_ROOT / args.reviewer_b).resolve())
    queue = read_jsonl((REPO_ROOT / args.queue).resolve())
    if not reviewer_a or not reviewer_b:
        raise RuntimeError("both reviewer annotation files are required")
    consensus, disagreements, slots = reconcile(reviewer_a, reviewer_b, queue)
    output = (REPO_ROOT / args.output_dir).resolve()
    write_jsonl(output / "scope-consensus.jsonl", consensus)
    write_jsonl(output / "scope-disagreements.jsonl", disagreements)
    write_jsonl(output / "candidate-target-slots.jsonl", slots)
    disposition_counts = Counter(
        str(row.get("consensus_disposition") or "disagreement") for row in consensus
    )
    summary = {
        "schema_version": "recent_total_synthesis_scope_reconciliation.v1",
        "paper_count": len(consensus),
        "fully_agreed_paper_count": sum(bool(row["fully_agreed"]) for row in consensus),
        "disagreement_paper_count": len(disagreements),
        "candidate_target_slot_count": len(slots),
        "consensus_disposition_counts": dict(sorted(disposition_counts.items())),
        "admission_authority": False,
        "claim_boundary": (
            "Dual AI metadata agreement creates review candidates only; it does not "
            "replace source-concordant chemical review."
        ),
    }
    write_json(output / "reconciliation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
