#!/usr/bin/env python3
"""Provisionally compare blind Strategy portfolios with paper-only evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerTask,
    preflight_worker_response_schemas,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
)
from cascade_planner.runtime.contracts import AgentSpec


SCHEMA_VERSION = "recent_total_synthesis_strategy_evaluation.v2"
STRATEGY_CUES = re.compile(
    r"\b(?:retrosynth|strategy|strategic|envision|hinge|feature|key step|"
    r"core|scaffold|skeleton|reorganiz|cascade|cycliz|annulat|diels|radical|"
    r"biomim|bioinspir|chemoenzym|total synthesis|in summary)\b",
    re.IGNORECASE,
)
MATCH_RANK = {"none": 0, "partial": 1, "exact": 2}
PRIMARY_TARGET_SLOT_CLASSES = {"primary", "primary_candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy-dir",
        type=Path,
        default=Path("results/recent-total-synthesis-strategy-coverage-20260902"),
    )
    parser.add_argument(
        "--route-evidence",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/route_evidence_candidates.jsonl"),
    )
    parser.add_argument(
        "--target-slots",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/target_slots.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/recent-total-synthesis-strategy-coverage-20260902/evaluation"
        ),
    )
    parser.add_argument("--target-slot-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _strategy_result_paths(strategy_dir: Path) -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in strategy_dir.glob("*/strategy-result.json")
        if path.is_file()
    }


def _passage_score(passage: dict[str, Any]) -> int:
    title = str(passage.get("section_title") or "")
    text = str(passage.get("verbatim_text") or "")
    score = min(10, int(passage.get("transformation_signal_count") or 0))
    score += min(8, len(STRATEGY_CUES.findall(f"{title} {text}")))
    if passage.get("target_mentioned_in_paragraph") is True:
        score += 4
    if re.search(r"retrosynth|discussion|conclusion|strateg", title, re.IGNORECASE):
        score += 6
    if len(text) < 120:
        score -= 4
    return score


def _compact_text(value: str, *, max_length: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= max_length:
        return compact
    boundary = compact.rfind(" ", 0, max_length - 1)
    return compact[: boundary if boundary > 0 else max_length - 1].rstrip() + "…"


def _evaluator_passages(
    route_row: dict[str, Any],
    *,
    max_passages: int = 8,
    max_total_chars: int = 24_000,
) -> list[dict[str, Any]]:
    ranked = sorted(
        enumerate(route_row.get("evidence_passages") or []),
        key=lambda item: (-_passage_score(dict(item[1])), item[0]),
    )
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for original_index, raw in ranked:
        if not isinstance(raw, dict) or len(selected) >= max_passages:
            continue
        remaining = max_total_chars - used_chars
        if remaining < 300:
            break
        text = _compact_text(
            str(raw.get("verbatim_text") or ""),
            max_length=min(3_600, remaining),
        )
        if not text:
            continue
        locator = dict(raw.get("source_locator") or {})
        identity = {
            "source_artifact_sha256": str(raw.get("source_artifact_sha256") or ""),
            "source_locator": locator,
            "verbatim_text": str(raw.get("verbatim_text") or ""),
        }
        selected.append(
            {
                "evidence_ref": f"passage-{_sha256_json(identity)[:16]}",
                "section_title": str(raw.get("section_title") or ""),
                "source_locator": locator,
                "verbatim_text": text,
                "automatically_extracted_unverified": True,
                "original_passage_index": original_index,
            }
        )
        used_chars += len(text)
    return selected


def _strategy_cards(result: dict[str, Any]) -> list[dict[str, str]]:
    cards = []
    for raw in result.get("final_strategy_cards") or []:
        if not isinstance(raw, dict):
            continue
        cards.append(
            {
                "strategy_query": str(raw.get("strategy_query") or ""),
                "critical_assumption": str(raw.get("critical_assumption") or ""),
                "critic_checkpoint": str(raw.get("critic_checkpoint") or ""),
            }
        )
    return cards


def _evaluation_prompt(
    *,
    target_slot: dict[str, Any],
    strategy_result: dict[str, Any],
    passages: list[dict[str, Any]],
) -> str:
    context = {
        "schema_version": "literature_strategy_match_input.v1",
        "target": {
            "target_slot_id": str(target_slot["target_slot_id"]),
            "target_name": str(target_slot.get("target_name") or ""),
            "target_smiles": str(strategy_result.get("target_smiles") or ""),
            "doi": str(target_slot.get("doi") or ""),
            "publication_title": str(target_slot.get("publication_title") or ""),
        },
        "blind_strategy_cards": _strategy_cards(strategy_result),
        "paper_evidence_passages": passages,
        "evidence_status": (
            "automatically extracted, source-bound leads; not human-admitted route truth"
        ),
    }
    return "\n".join(
        [
            "Act as an evaluator only. The planner saw only the target structure; it did not see the target name, DOI, title, or paper passages below.",
            "Compare each of the three blind Strategy cards against the successful target-specific strategy actually supported by the supplied paper passages. Do not reward a card merely because it is chemically plausible, shares a generic reaction word, or resembles an approach that the paper says failed.",
            "Use match_level=exact only when a card independently recovers the same route-defining principal-scaffold construction or skeletal reorganization and its key transformation family/control logic. Exact does not require identical protecting groups, catalysts, or peripheral finishing operations.",
            "Use match_level=partial only for a substantive shared strategic disconnection, scaffold source, cascade logic, or key transformation that misses, substitutes, or mis-sequences the paper's route-defining event. Use none when overlap is generic or peripheral.",
            "Set comparability=insufficient_route_evidence when these passages do not establish the successful strategy for this target. Set target_identity_ambiguous when a collective paper does not bind the described route to this target. In either case, still return three card assessments conservatively as none.",
            "Summarize only supported successful-route claims, assign up to three paper_strategy_classes, and cite only supplied evidence_ref values. Do not browse, infer missing schemes, or claim human verification.",
            "LiteratureStrategyMatchInput:",
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def _record_projection(record: Any) -> dict[str, Any]:
    return {
        "task_id": str(record.task_id or ""),
        "status": str(record.status or ""),
        "backend": str(record.backend or ""),
        "usage": dict(record.usage or {}),
        "elapsed_s": float(record.elapsed_s or 0.0),
        "output_validation": dict(record.output_validation or {}),
    }


def _derived_match(payload: dict[str, Any]) -> tuple[str, int]:
    if str(payload.get("comparability") or "") != "comparable":
        return "non_comparable", 0
    assessments = [
        dict(value)
        for value in payload.get("card_assessments") or []
        if isinstance(value, dict)
    ]
    best = max(
        assessments,
        key=lambda row: MATCH_RANK.get(str(row.get("match_level") or "none"), -1),
        default={},
    )
    return (
        str(best.get("match_level") or "none"),
        int(best.get("card_index") or 0),
    )


def _valid_evaluation_payload(
    payload: dict[str, Any],
    *,
    target_slot_id: str,
    allowed_evidence_refs: set[str],
) -> bool:
    assessments = [
        dict(value)
        for value in payload.get("card_assessments") or []
        if isinstance(value, dict)
    ]
    cited_refs = [str(value) for value in payload.get("evidence_locator_refs") or []]
    return bool(
        payload.get("schema_version") == "literature_strategy_match_report.v1"
        and payload.get("target_slot_id") == target_slot_id
        and sorted(int(row.get("card_index") or 0) for row in assessments)
        == [1, 2, 3]
        and len(cited_refs) == len(set(cited_refs))
        and set(cited_refs) <= allowed_evidence_refs
        and (
            payload.get("comparability") != "comparable"
            or bool(cited_refs)
        )
    )


def _non_comparable_payload(
    *, target_slot_id: str, reason: str
) -> dict[str, Any]:
    return {
        "schema_version": "literature_strategy_match_report.v1",
        "case_id": f"literature-eval-{target_slot_id}",
        "target_slot_id": target_slot_id,
        "comparability": "insufficient_route_evidence",
        "paper_strategy_summary": "",
        "paper_key_transformations": [],
        "paper_strategy_classes": [],
        "card_assessments": [
            {
                "card_index": index,
                "match_level": "none",
                "matched_elements": [],
                "missing_or_conflicting_elements": [reason],
                "rationale": reason,
            }
            for index in (1, 2, 3)
        ],
        "confidence": "high",
        "evidence_locator_refs": [],
        "limitations": [reason],
        "provisional_automated_evaluation": True,
    }


def _aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    match_counts = Counter(str(row.get("overall_match") or "") for row in rows)
    comparable = sum(row.get("overall_match") != "non_comparable" for row in rows)
    exact = match_counts["exact"]
    partial = match_counts["partial"]
    gap_free_exact = sum(
        row.get("overall_match") == "exact"
        and not list(
            next(
                (
                    assessment.get("missing_or_conflicting_elements") or []
                    for assessment in dict(row.get("evaluation") or {}).get(
                        "card_assessments"
                    )
                    or []
                    if isinstance(assessment, dict)
                    and int(assessment.get("card_index") or 0)
                    == int(row.get("best_card_index") or 0)
                ),
                [],
            )
        )
        for row in rows
    )
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for strategy_class in set(row.get("paper_strategy_classes") or []):
            strata[str(strategy_class)][str(row.get("overall_match") or "")] += 1
    rows_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            rows_by_paper[paper_id].append(row)
    paper_matches: list[dict[str, Any]] = []
    for paper_id, paper_rows in sorted(rows_by_paper.items()):
        comparable_rows = [
            row
            for row in paper_rows
            if row.get("overall_match") != "non_comparable"
        ]
        paper_match = (
            max(
                (str(row.get("overall_match") or "none") for row in comparable_rows),
                key=lambda value: MATCH_RANK.get(value, -1),
            )
            if comparable_rows
            else "non_comparable"
        )
        paper_matches.append(
            {
                "paper_id": paper_id,
                "target_count": len(paper_rows),
                "comparable_target_count": len(comparable_rows),
                "overall_match": paper_match,
            }
        )
    paper_match_counts = Counter(row["overall_match"] for row in paper_matches)
    comparable_papers = len(paper_matches) - paper_match_counts["non_comparable"]
    return {
        "schema_version": SCHEMA_VERSION,
        "target_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "comparable_target_count": comparable,
        "non_comparable_target_count": match_counts["non_comparable"],
        "match_counts": dict(sorted(match_counts.items())),
        "exact_match_rate_among_comparable": exact / comparable if comparable else None,
        "gap_free_exact_match_count": gap_free_exact,
        "gap_free_exact_match_rate_among_comparable": (
            gap_free_exact / comparable if comparable else None
        ),
        "at_least_partial_match_rate_among_comparable": (
            (exact + partial) / comparable if comparable else None
        ),
        "strategy_class_strata": {
            key: {
                "target_count": sum(counts.values()),
                "comparable_target_count": sum(
                    count
                    for match_level, count in counts.items()
                    if match_level != "non_comparable"
                ),
                "non_comparable_target_count": counts["non_comparable"],
                "match_counts": dict(sorted(counts.items())),
                "at_least_partial_rate": (
                    (counts["exact"] + counts["partial"])
                    / sum(
                        count
                        for match_level, count in counts.items()
                        if match_level != "non_comparable"
                    )
                    if any(
                        count
                        for match_level, count in counts.items()
                        if match_level != "non_comparable"
                    )
                    else None
                ),
            }
            for key, counts in sorted(strata.items())
        },
        "strategy_class_strata_semantics": (
            "multi_label; one target may contribute to up to three classes"
        ),
        "paper_cluster_sensitivity": {
            "aggregation": (
                "best comparable target per paper; exploratory any-target sensitivity"
            ),
            "paper_count": len(paper_matches),
            "comparable_paper_count": comparable_papers,
            "non_comparable_paper_count": paper_match_counts["non_comparable"],
            "match_counts": dict(sorted(paper_match_counts.items())),
            "exact_match_rate_among_comparable": (
                paper_match_counts["exact"] / comparable_papers
                if comparable_papers
                else None
            ),
            "at_least_partial_match_rate_among_comparable": (
                (paper_match_counts["exact"] + paper_match_counts["partial"])
                / comparable_papers
                if comparable_papers
                else None
            ),
            "papers": paper_matches,
        },
        "claim_boundary": {
            "cohort": "unverified_unique_structure_candidates",
            "formal_benchmark_denominator": 0,
            "automated_route_evidence_is_human_admitted_truth": False,
            "automated_match_is_human_dual_review": False,
            "rates_are_provisional": True,
        },
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    strategy_dir = (REPO_ROOT / args.strategy_dir).resolve()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target_slots = _read_jsonl((REPO_ROOT / args.target_slots).resolve())
    slot_by_id = {str(row["target_slot_id"]): row for row in target_slots}
    route_by_id = {
        str(row["target_slot_id"]): row
        for row in _read_jsonl((REPO_ROOT / args.route_evidence).resolve())
    }
    result_paths = _strategy_result_paths(strategy_dir)
    selected = {str(value) for value in args.target_slot_id}
    target_ids = [
        str(row["target_slot_id"])
        for row in target_slots
        if str(row["target_slot_id"]) in result_paths
        and (not selected or str(row["target_slot_id"]) in selected)
    ]
    if args.limit > 0:
        target_ids = target_ids[: args.limit]

    def run_one(target_slot_id: str) -> dict[str, Any]:
        target_slot = slot_by_id[target_slot_id]
        strategy_result = json.loads(
            result_paths[target_slot_id].read_text(encoding="utf-8")
        )
        cards = _strategy_cards(strategy_result)
        if len(cards) != 3:
            return {
                "target_slot_id": target_slot_id,
                "status": "strategy_portfolio_invalid",
                "overall_match": "non_comparable",
                "best_card_index": 0,
                "paper_strategy_classes": [],
                "evaluation": {},
                "evaluator_record": {},
            }
        passages = _evaluator_passages(route_by_id.get(target_slot_id, {}))
        prompt = _evaluation_prompt(
            target_slot=target_slot,
            strategy_result=strategy_result,
            passages=passages,
        )
        digest = _sha256_json(
            {
                "target_slot_id": target_slot_id,
                "strategy_input_sha256": strategy_result.get("input_sha256"),
                "strategy_cards": cards,
                "evaluator_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "evaluation_contract": SCHEMA_VERSION,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
            }
        )
        workdir = output_root / target_slot_id
        workdir.mkdir(parents=True, exist_ok=True)
        result_path = workdir / "evaluation-result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_sha256") == digest and existing.get("terminal"):
                return existing

        if not passages:
            payload = _non_comparable_payload(
                target_slot_id=target_slot_id,
                reason="No target-linked route passage was available.",
            )
            status = "non_comparable_no_passages"
            record_projection: dict[str, Any] = {}
        else:
            case_id = f"literature-eval-{hashlib.sha256(target_slot_id.encode()).hexdigest()[:16]}"
            task = WorkerTask(
                task_id=f"literature-strategy-evaluator:{target_slot_id}:1",
                case_id=case_id,
                task_type="literature_strategy_match_evaluator",
                required_artifact_type="LiteratureStrategyMatchReport",
                input_refs=[],
                allowed_tools=[],
                budget=WorkerBudget(
                    timeout_s=args.timeout_s,
                    max_output_bytes=12_000,
                    max_tool_calls=None,
                    max_worker_runs=1,
                    reasoning_effort=args.reasoning_effort,
                ),
                objective=prompt,
                allowed_workdir=str(workdir),
                agent_mode="single",
                codex_auth_mode="ambient_codex_cli",
                model=args.model,
                host_context={"target_slot_id": target_slot_id},
            )
            preflight_worker_response_schemas([task])
            spec = AgentSpec(
                run_id=f"recent-literature-eval-{target_slot_id}",
                agent_id=f"literature-evaluator:{target_slot_id}",
                role="literature_strategy_match_evaluator",
                objective="Evaluate blind Strategy rediscovery against paper-only evidence.",
                idempotency_key=f"literature-evaluator:{target_slot_id}:v1",
                context_hash=digest,
                metadata={
                    "allowed_workdir": str(workdir),
                    "durable_worker_journal": True,
                },
            )
            runner = SequentialStrategyDirectorRunner()
            runner._prepare_worker_record_journal(spec)
            record = runner._run_journaled_worker(runner.critic_executor, task)
            record_projection = _record_projection(record)
            artifact = dict(record.output_artifact or {})
            payload = dict(artifact.get("payload") or {})
            status = (
                "evaluated"
                if record.status == "accepted_draft"
                and _valid_evaluation_payload(
                    payload,
                    target_slot_id=target_slot_id,
                    allowed_evidence_refs={
                        str(value["evidence_ref"]) for value in passages
                    },
                )
                else "evaluator_invalid"
            )

        overall_match, best_card_index = (
            _derived_match(payload)
            if status in {"evaluated", "non_comparable_no_passages"}
            else ("non_comparable", 0)
        )
        result = {
            "schema_version": "recent_total_synthesis_strategy_evaluation_result.v1",
            "target_slot_id": target_slot_id,
            "paper_id": str(target_slot.get("paper_id") or ""),
            "doi": str(target_slot.get("doi") or ""),
            "target_name": str(target_slot.get("target_name") or ""),
            "input_sha256": digest,
            "status": status,
            "terminal": status
            in {"evaluated", "non_comparable_no_passages", "evaluator_invalid"},
            "overall_match": overall_match,
            "best_card_index": best_card_index,
            "paper_strategy_classes": sorted(
                set(str(value) for value in payload.get("paper_strategy_classes") or [])
            ),
            "evidence_passage_count": len(passages),
            "evaluation": payload,
            "evaluator_record": record_projection,
            "formal_benchmark_eligible": False,
            "provisional_automated_evaluation": True,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, target_id): target_id for target_id in target_ids}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "target_slot_id": result["target_slot_id"],
                        "status": result["status"],
                        "overall_match": result["overall_match"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    by_id = {str(row["target_slot_id"]): row for row in results}
    ordered = [by_id[target_id] for target_id in target_ids]
    primary_target_slot_count = sum(
        str(row.get("slot_class") or "") in PRIMARY_TARGET_SLOT_CLASSES
        for row in target_slots
    )
    summary = {
        **_aggregate(ordered),
        "primary_target_slot_count": primary_target_slot_count,
        "candidate_strategy_coverage_count": len(ordered),
        "candidate_strategy_coverage_rate": (
            len(ordered) / primary_target_slot_count
            if primary_target_slot_count
            else None
        ),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "model_invocations": sum(bool(row.get("evaluator_record")) for row in ordered),
        "results": [
            {
                "target_slot_id": row["target_slot_id"],
                "status": row["status"],
                "overall_match": row["overall_match"],
                "result_path": str(
                    Path(row["target_slot_id"]) / "evaluation-result.json"
                ).replace("\\", "/"),
            }
            for row in ordered
        ],
    }
    with (output_root / "evaluation-results.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in ordered:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    (output_root / "evaluation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
