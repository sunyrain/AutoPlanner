#!/usr/bin/env python3
"""Run the real-source independent critic ablation on frozen procedure cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.codex_worker import (
    WorkerBudget,
    WorkerTask,
    run_codex_worker,
)
from cascade_planner.application.campaign_contract_json import bound_row
from cascade_planner.eval.independent_critic_ablation import (
    CONDITION_FIELDS,
    compile_blind_procedure_cases,
    compile_independent_critic_ablation,
)


DEFAULT_CONFIG = ROOT / "benchmarks" / "real_patent_procedure_gate_cases.v1.json"
DEFAULT_EVIDENCE = (
    ROOT
    / "results"
    / "shared"
    / "patent_procedure_gate_20260717"
    / "patent-procedure-gate"
    / "summary.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "shared" / "independent_critic_ablation_20260813"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a blind same-backbone critique loop with a digest-bound "
            "host evidence-triggered repair on real official procedures."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--parallel-cases", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    evidence_path = args.evidence.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    evidence = _read_json(evidence_path)
    blind_cases = compile_blind_procedure_cases(config)
    receipt = _receipt(
        config_path=config_path,
        evidence_path=evidence_path,
        config=config,
        evidence=evidence,
        model=str(args.model),
        reasoning_effort=str(args.reasoning_effort),
        timeout_s=float(args.timeout_s),
    )
    _write_json(output_root / "execution-receipt.json", receipt)

    results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel_cases))) as executor:
        futures = {
            executor.submit(
                _run_case,
                blind,
                output_root=output_root,
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                timeout_s=float(args.timeout_s),
                resume=bool(args.resume),
            ): str(blind["opaque_case_id"])
            for blind in blind_cases
        }
        for future in as_completed(futures):
            opaque_id = futures[future]
            initial, critique = future.result()
            results[opaque_id] = (initial, critique)
            print(
                f"{opaque_id}: initial={initial.get('status')} "
                f"self_critique={critique.get('status')}",
                flush=True,
            )

    initial_drafts = {case_id: value[0] for case_id, value in results.items()}
    self_critique_drafts = {case_id: value[1] for case_id, value in results.items()}
    report = compile_independent_critic_ablation(
        config=config,
        evidence_suite=evidence,
        initial_drafts=initial_drafts,
        self_critique_drafts=self_critique_drafts,
    )
    report["execution_receipt_sha256"] = receipt["content_sha256"]
    report = bound_row(report)
    _write_json(output_root / "summary.json", report)
    (output_root / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["arms"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_case(
    blind: Mapping[str, Any],
    *,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(blind["opaque_case_id"])
    case_root = output_root / "cases" / case_id
    workspace = case_root / "blind-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / "blind-case.json", blind)
    initial_path = case_root / "initial-model-draft.json"
    critique_path = case_root / "same-backbone-self-critique.json"
    initial = _load_resumable(initial_path) if resume else None
    if initial is None:
        initial = _run_stage(
            blind,
            workspace=workspace,
            stage="initial",
            prior=None,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
        )
        _write_json(initial_path, bound_row(initial))
    prior = dict(dict(initial.get("output_artifact") or {}).get("payload") or {})
    _write_json(workspace / "prior-model-draft.json", prior)
    critique = _load_resumable(critique_path) if resume else None
    if critique is None:
        critique = _run_stage(
            blind,
            workspace=workspace,
            stage="self_critique",
            prior=prior,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
        )
        _write_json(critique_path, bound_row(critique))
    return initial, critique


def _run_stage(
    blind: Mapping[str, Any],
    *,
    workspace: Path,
    stage: str,
    prior: Mapping[str, Any] | None,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
) -> dict[str, Any]:
    case_id = str(blind["opaque_case_id"])
    blind_json = json.dumps(blind, ensure_ascii=False, sort_keys=True)
    if stage == "initial":
        objective = (
            "Blindly audit this one reaction step and propose a conservative procedure repair. "
            "You have no source, patent, target name, reference conditions, or experimental result. "
            "Infer only from the structures; leave unsupported fields empty and enumerate risks. "
            f"Use step_id={case_id}-S1. Blind case: {blind_json}"
        )
        input_refs = ["blind-case.json"]
    else:
        prior_json = json.dumps(prior or {}, ensure_ascii=False, sort_keys=True)
        objective = (
            "Act as the same-backbone critic/editor. Re-audit your prior model-only procedure "
            "against the identical blind structures. Correct internal inconsistencies and unsafe "
            "assumptions, but do not browse, cite, invent exact evidence, or claim experimental "
            "validation. Leave information unknowable from structure empty. "
            f"Use step_id={case_id}-S1. Blind case: {blind_json}. Prior draft: {prior_json}"
        )
        input_refs = ["blind-case.json", "prior-model-draft.json"]
    task = WorkerTask(
        task_id=f"{case_id}:{stage}",
        case_id=case_id,
        task_type="condition_research",
        required_artifact_type="ProcedureRepairDraft",
        input_refs=input_refs,
        allowed_tools=[],
        budget=WorkerBudget(
            timeout_s=timeout_s,
            max_output_bytes=100_000,
            max_tool_calls=0,
            max_worker_runs=1,
            reasoning_effort=reasoning_effort,
        ),
        objective=objective,
        allowed_workdir=str(workspace),
        codex_auth_mode="ambient_codex_cli",
        model=model,
    )
    return run_codex_worker(task, use_codex_cli=True).to_dict()


def _receipt(
    *,
    config_path: Path,
    evidence_path: Path,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    timeout_s: float,
) -> dict[str, Any]:
    return bound_row(
        {
            "schema_version": "independent_critic_ablation_execution.v1",
            "config": {
                "path": str(config_path),
                "file_sha256": _file_sha256(config_path),
                "content_sha256": str(config.get("content_sha256") or ""),
            },
            "evidence": {
                "path": str(evidence_path),
                "file_sha256": _file_sha256(evidence_path),
                "content_sha256": str(evidence.get("content_sha256") or ""),
            },
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timeout_s_per_model_call": timeout_s,
            "case_count": len(config.get("cases") or []),
            "model_call_count_planned": 2 * len(config.get("cases") or []),
            "python": sys.executable,
            "platform": platform.platform(),
            "condition_fields": list(CONDITION_FIELDS),
            "semantics": {
                "same_model_is_used_for_initial_and_self_critique": True,
                "model_workspaces_contain_blind_case_and_prior_draft_only": True,
                "web_and_other_tools_are_disabled": True,
                "official_evidence_is_loaded_only_by_host_scoring": True,
                "model_output_has_proposal_authority_only": True,
            },
        }
    )


def _load_resumable(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = _read_json(path)
    supplied = str(row.pop("content_sha256", ""))
    rebound = bound_row(row)
    return row if supplied == rebound["content_sha256"] else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"json_object_required:{path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(report: Mapping[str, Any]) -> str:
    arms = dict(report.get("arms") or {})
    lines = [
        "# Independent Critic / Evidence-Triggered Repair Ablation",
        "",
        f"Cases: {report.get('case_count')} real, official, digest-bound procedures.",
        "",
        "| Arm | Assessed | Complete by presence | Frozen-oracle recall | Source-text exact recall | Unsupported field rate | Exact source closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("initial_model_draft", "same_backbone_self_critique", "evidence_triggered_repair"):
        row = dict(arms.get(arm) or {})
        lines.append(
            "| {arm} | {assessed}/{cases} | {complete}/{cases} | {oracle:.3f} | {recall:.3f} | {unsupported:.3f} | {closed}/{cases} |".format(
                arm=arm,
                assessed=row.get("assessed_count", 0),
                cases=row.get("case_count", 0),
                complete=row.get("condition_complete_count", 0),
                oracle=float(row.get("mean_frozen_oracle_criterion_recall") or 0),
                recall=float(row.get("mean_exact_field_recall") or 0),
                unsupported=float(row.get("mean_unsupported_field_rate") or 0),
                closed=row.get("exact_condition_closed_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "The evidence arm is triggered only by a new host observation bound to the same canonical reaction identity. Model-only drafts never receive source or experimental authority.",
            "",
            "This ablation evaluates exact-source procedure closure; it does not evaluate experimental success, complete-route feasibility, or stock closure.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
