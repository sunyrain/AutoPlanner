#!/usr/bin/env python3
"""Blind-run the current Strategy Generator and Strategy Critic over a target set."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
    _paper_strategy_portfolio_critic_prompt,
    _paper_strategy_portfolio_prompt,
    _strategy_cards_from_portfolio_record,
    _strategy_portfolio_critic_task,
    _strategy_portfolio_task,
)
from cascade_planner.runtime.contracts import AgentSpec


SCHEMA_VERSION = "recent_total_synthesis_strategy_batch.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner-targets",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/planner_targets.jsonl"),
    )
    parser.add_argument(
        "--structure-candidates",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/structure_resolution_candidates.jsonl"),
    )
    parser.add_argument(
        "--candidate-coverage",
        action="store_true",
        help="Use unique RDKit-valid, conflict-free PubChem candidates as a non-formal coverage set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recent-total-synthesis-strategy-batch"),
    )
    parser.add_argument("--target-slot-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--generator-only", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _candidate_coverage_targets(path: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        candidates = list(row.get("candidates") or [])
        if len(candidates) != 1 or list(row.get("review_flags") or []):
            continue
        validation = dict(candidates[0].get("rdkit_validation") or {})
        smiles = str(validation.get("canonical_isomeric_smiles") or "")
        if validation.get("status") != "roundtrip_valid" or not smiles:
            continue
        targets.append(
            {
                "target_slot_id": str(row["target_slot_id"]),
                "target_smiles": smiles,
                "input_status": "unverified_unique_structure_candidate",
                "formal_benchmark_eligible": False,
            }
        )
    return targets


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_digest(
    target: dict[str, Any],
    *,
    model: str,
    effort: str,
    generator_prompt: str,
    critic_template_prompt: str,
) -> str:
    payload = {
        "target_slot_id": target["target_slot_id"],
        "target_smiles": target["target_smiles"],
        "model": model,
        "reasoning_effort": effort,
        "generator_prompt_sha256": _sha256_text(generator_prompt),
        "critic_template_prompt_sha256": _sha256_text(critic_template_prompt),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record_projection(record: Any) -> dict[str, Any]:
    return {
        "task_id": str(record.task_id or ""),
        "status": str(record.status or ""),
        "backend": str(record.backend or ""),
        "usage": dict(record.usage or {}),
        "elapsed_s": float(record.elapsed_s or 0.0),
        "output_validation": dict(record.output_validation or {}),
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.candidate_coverage:
        targets = _candidate_coverage_targets((REPO_ROOT / args.structure_candidates).resolve())
        cohort = "unverified_candidate_coverage"
    else:
        targets = _read_jsonl((REPO_ROOT / args.planner_targets).resolve())
        cohort = "formally_admitted_targets"
    selected = {str(value) for value in args.target_slot_id}
    if selected:
        targets = [row for row in targets if str(row["target_slot_id"]) in selected]
    if args.limit > 0:
        targets = targets[: args.limit]
    output_root = (REPO_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    def run_one(target: dict[str, Any]) -> dict[str, Any]:
        target_slot_id = str(target["target_slot_id"])
        target_smiles = str(target["target_smiles"])
        workdir = output_root / target_slot_id
        workdir.mkdir(parents=True, exist_ok=True)
        generator_prompt = _paper_strategy_portfolio_prompt(
            target=target_smiles,
            enhanced=True,
        )
        critic_template_prompt = _paper_strategy_portfolio_critic_prompt(
            target=target_smiles,
            strategy_cards=(),
        )
        digest = _target_digest(
            target,
            model=args.model,
            effort=args.reasoning_effort,
            generator_prompt=generator_prompt,
            critic_template_prompt=critic_template_prompt,
        )
        result_path = workdir / "strategy-result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_sha256") == digest and existing.get("terminal"):
                return existing
        spec = AgentSpec(
            run_id=f"recent-strategy-{target_slot_id}",
            agent_id=f"strategy:{target_slot_id}",
            role="paper_matched_strategy_generator",
            objective="Generate and review one blind three-card strategy portfolio.",
            idempotency_key=f"strategy:{target_slot_id}:v1",
            context_hash=digest,
            metadata={
                "allowed_workdir": str(workdir),
                "durable_worker_journal": True,
                "model": args.model,
                "strategy_reasoning_effort": args.reasoning_effort,
                "critic_reasoning_effort": args.reasoning_effort,
            },
        )
        runner = SequentialStrategyDirectorRunner()
        runner._prepare_worker_record_journal(spec)
        generator_task = _strategy_portfolio_task(
            spec,
            prompt=generator_prompt,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_s=args.timeout_s,
            target_smiles=target_smiles,
        )
        generator_record = runner._run_journaled_worker(runner.node_executor, generator_task)
        generated_cards = _strategy_cards_from_portfolio_record(
            generator_record,
            expected_target=target_smiles,
        )
        reviewed_cards = None
        critic_record = None
        if generated_cards is not None and not args.generator_only:
            critic_prompt = _paper_strategy_portfolio_critic_prompt(
                target=target_smiles,
                strategy_cards=generated_cards,
            )
            critic_task = _strategy_portfolio_critic_task(
                spec,
                prompt=critic_prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=args.timeout_s,
                target_smiles=target_smiles,
            )
            critic_record = runner._run_journaled_worker(runner.critic_executor, critic_task)
            reviewed_cards = _strategy_cards_from_portfolio_record(
                critic_record,
                expected_target=target_smiles,
            )
        final_cards = reviewed_cards or generated_cards or []
        status = (
            "critic_reviewed"
            if reviewed_cards is not None
            else "generator_completed"
            if generated_cards is not None
            else "generator_invalid"
        )
        result = {
            "schema_version": "recent_total_synthesis_strategy_result.v1",
            "target_slot_id": target_slot_id,
            "target_smiles": target_smiles,
            "input_sha256": digest,
            "cohort": cohort,
            "formal_benchmark_eligible": bool(
                target.get("formal_benchmark_eligible", not args.candidate_coverage)
            ),
            "input_status": str(target.get("input_status") or "admitted"),
            "status": status,
            "terminal": status
            in {
                "critic_reviewed",
                "generator_completed",
                "generator_invalid",
            },
            "generated_strategy_cards": generated_cards or [],
            "reviewed_strategy_cards": reviewed_cards or [],
            "final_strategy_cards": final_cards,
            "generator_record": _record_projection(generator_record),
            "critic_record": (
                _record_projection(critic_record) if critic_record is not None else {}
            ),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "planner_input_fields": ["target_slot_id", "target_smiles"],
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, target): target for target in targets}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "target_slot_id": result["target_slot_id"],
                        "status": result["status"],
                        "strategy_count": len(result["final_strategy_cards"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    result_by_id = {row["target_slot_id"]: row for row in results}
    ordered = [result_by_id[str(target["target_slot_id"])] for target in targets]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "target_count": len(ordered),
        "formal_benchmark_target_count": sum(
            bool(row["formal_benchmark_eligible"]) for row in ordered
        ),
        "critic_reviewed": sum(row["status"] == "critic_reviewed" for row in ordered),
        "generator_completed_without_critic": sum(
            row["status"] == "generator_completed" for row in ordered
        ),
        "generator_invalid": sum(row["status"] == "generator_invalid" for row in ordered),
        "model_invocations": sum(
            int(row["generator_record"].get("usage", {}).get("model_invocations") or 1)
            + (
                int(row["critic_record"].get("usage", {}).get("model_invocations") or 1)
                if row["critic_record"]
                else 0
            )
            for row in ordered
        ),
        "results": [
            {
                "target_slot_id": row["target_slot_id"],
                "status": row["status"],
                "result_path": str(Path(row["target_slot_id"]) / "strategy-result.json").replace(
                    "\\", "/"
                ),
            }
            for row in ordered
        ],
    }
    (output_root / "batch-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
