#!/usr/bin/env python3
"""Run production-style PDF visual source-detail follow-up for a target."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.research.downstream_compiler import (
    compile_downstream_consumables,
    write_compiled_downstream_artifacts,
)
from cascade_planner.legacy.harness_runtime.runner import emit_final_verdict
from cascade_planner.legacy.harness_runtime.preflight import run_preflight
from cascade_planner.legacy.harness_runtime.schemas import (
    CANONICAL_RUN_SEMANTICS,
    TargetInput,
    append_jsonl,
    write_json,
)
from cascade_planner.legacy.harness_runtime.tools import (
    HarnessBudget,
    ToolExecutionState,
    artifact_bundle_from_state,
    execute_local_tool,
)


DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "shared"


@dataclass(frozen=True)
class LiteraturePdfSource:
    key: str
    pdf_path: Path
    source_ref: str
    source_title: str = ""
    page_numbers: tuple[int, ...] = ()
    compound_labels: tuple[str, ...] = ()
    expected_labels: tuple[str, ...] = ()
    route_sequence_hint: str = ""
    condition_repairs: tuple[dict[str, Any], ...] = ()
    allow_partial_chain_without_target_match: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--family-hint", default="")
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        help="JSON file containing a list of literature PDF source objects.",
    )
    parser.add_argument(
        "--literature-source",
        action="append",
        default=[],
        help=(
            "One source as JSON object, or path|source_ref|title|pages_csv|labels_csv|hint. "
            "Use --source-manifest for complex condition repairs."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-prefix", default="codex_entry_pdf_visual_followup")
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--visual-timeout-s", type=float, default=900.0)
    parser.add_argument("--visual-retries", type=int, default=1)
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=1200.0)
    parser.add_argument("--route-expansion-timeout-s", type=float, default=600.0)
    parser.add_argument("--chem-enzy-iterations", type=int, default=50)
    parser.add_argument("--chem-enzy-expansion-topk", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-route-expansion-subgoal-runs", type=int, default=2)
    parser.add_argument("--route-failure-feedback-path", default="")
    parser.add_argument("--route-audit-path", default="")
    parser.add_argument("--skip-analogical-retrosynthesis", action="store_true")
    parser.add_argument("--skip-guided-rerun", action="store_true")
    parser.add_argument("--skip-route-expansion", action="store_true")
    args = parser.parse_args()

    sources = _load_sources(args)
    if not sources:
        raise SystemExit("Provide at least one PDF source via --source-manifest or --literature-source.")

    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = TargetInput(
        target_name=args.target_name,
        target_smiles=args.target_smiles,
        family_hint=args.family_hint,
    )
    preflight = run_preflight(target)
    target.case_id = str(preflight.get("case_id") or target.case_id)
    target_data = target.to_dict()
    budget = HarnessBudget(
        timeout_s=max(float(args.guided_chemenzy_timeout_s) + float(args.route_expansion_timeout_s) + 900.0, 1800.0),
        guided_chemenzy_timeout_s=float(args.guided_chemenzy_timeout_s),
        max_route_expansion_subgoal_runs=int(args.max_route_expansion_subgoal_runs),
    )
    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_data,
        preflight=preflight,
        budget=budget,
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
        run_semantics=CANONICAL_RUN_SEMANTICS,
    )
    _load_optional_artifact(state, "route_failure_feedback", args.route_failure_feedback_path)
    _load_optional_artifact(state, "route_audit", args.route_audit_path)

    write_json(run_dir / "target_input.json", target_data)
    write_json(run_dir / "preflight.json", preflight)
    write_json(run_dir / "budget.json", budget.to_dict())
    write_json(run_dir / "literature_pdf_sources.json", {"sources": [_source_to_dict(src) for src in sources]})
    append_jsonl(run_dir / "decision_trace.jsonl", {"stage": "start_pdf_visual_followup", "source_count": len(sources)})

    tool_calls: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    source_detail_steps: list[dict[str, Any]] = []
    rejected_consumables: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        summary = _run_source_pipeline(
            state=state,
            source=source,
            source_index=index,
            args=args,
            tool_calls=tool_calls,
        )
        source_summaries.append(summary)
        source_detail_steps.extend(summary.pop("_source_detail_steps", []))
        rejected_consumables.extend(summary.pop("_rejected_consumables", []))

    compiled = _compile_combined_downstream(
        state=state,
        source_detail_steps=source_detail_steps,
        rejected_consumables=rejected_consumables,
    )
    if not args.skip_analogical_retrosynthesis and compiled.get("accepted"):
        record = execute_local_tool(
            "build_analogical_retrosynthesis_hypotheses",
            {"max_hypotheses": 12},
            state,
        )
        tool_calls.append(record.to_dict())
    if not args.skip_guided_rerun and compiled.get("accepted"):
        record = execute_local_tool(
            "run_guided_chemenzy_rerun",
            {
                "timeout_s": float(args.guided_chemenzy_timeout_s),
                "search_preset": "thorough",
                "max_steps": int(args.max_steps),
                "chem_enzy_iterations": int(args.chem_enzy_iterations),
                "chem_enzy_expansion_topk": int(args.chem_enzy_expansion_topk),
                "stock_mode": "building-block",
                "device": "cpu",
                "policy_id": f"{state.preflight.get('case_id')}_pdf_visual_literature_plugin_only",
            },
            state,
        )
        tool_calls.append(record.to_dict())
    if not args.skip_route_expansion:
        record = execute_local_tool(
            "run_route_expansion_subgoal_search",
            {
                "timeout_s": float(args.route_expansion_timeout_s),
                "max_targets": int(args.max_route_expansion_subgoal_runs),
                "search_preset": "thorough",
                "device": "cpu",
            },
            state,
        )
        tool_calls.append(record.to_dict())
    validation_record = execute_local_tool("validate_artifact_bundle", {}, state)
    tool_calls.append(validation_record.to_dict())

    workflow_plan = {
        "schema_version": "codex_entry_pdf_visual_followup_plan.v1",
        "run_semantics": CANONICAL_RUN_SEMANTICS,
        "planned_tools": [call.get("tool_name") for call in tool_calls],
        "source": "scripts/legacy/run_codex_entry_pdf_visual_followup.py",
    }
    bundle = artifact_bundle_from_state(state=state, workflow_plan=workflow_plan, tool_calls=tool_calls)
    verdict = emit_final_verdict(bundle)
    write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(run_dir / "final_verdict.json", verdict.to_dict())

    summary = {
        "schema_version": "codex_entry_pdf_visual_followup_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_count": len(sources),
        "source_summaries": source_summaries,
        "combined_source_detail_step_count": len(source_detail_steps),
        "compiled_downstream": {
            "accepted": bool(compiled.get("accepted")),
            "reasons": [str(item) for item in compiled.get("reasons") or []],
            "one_step_row_count": len(((compiled.get("literature_template_plugin") or {}).get("one_step_rows") or [])),
            "guided_policy_count": len(((compiled.get("guided_chemenzy") or {}).get("policy_payloads") or [])),
            "route_expansion_task_count": len(((compiled.get("route_expansion") or {}).get("tasks") or [])),
        },
        "analogical_retrosynthesis": _analogical_summary(state),
        "guided_rerun": _guided_summary(state),
        "route_expansion": _route_expansion_summary(state),
        "final_verdict": verdict.to_dict(),
    }
    write_json(run_dir / "run_summary.json", summary)
    print(json.dumps({"schema_version": "codex_entry_pdf_visual_followup_cli_result.v1", **summary}, indent=2, ensure_ascii=False, sort_keys=True))


def _run_source_pipeline(
    *,
    state: ToolExecutionState,
    source: LiteraturePdfSource,
    source_index: int,
    args: argparse.Namespace,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    source_dir = state.run_dir / "pdf_sources" / f"{source_index:02d}_{source.key}"
    evidence_refs = [f"pdf:{source.pdf_path}", source.source_ref]
    pdf_record = execute_local_tool(
        "extract_pdf_literature_structures",
        {
            "pdf_path": str(source.pdf_path),
            "page_numbers": list(source.page_numbers),
            "compound_labels": list(source.compound_labels),
            "render_zoom": 2.0,
            "output_dir": str(source_dir / "pdf_assets"),
        },
        state,
    )
    tool_calls.append(pdf_record.to_dict())
    pdf_evidence = dict((pdf_record.output or {}).get("result") or {})

    visual_records = []
    visual_record = None
    max_attempts = max(1, int(args.visual_retries) + 1)
    for attempt in range(1, max_attempts + 1):
        hint = source.route_sequence_hint
        if attempt > 1:
            hint = (
                f"{hint} Previous automatic extraction attempt produced no usable source-detail steps. "
                "Re-inspect the current PDF images and extract fewer, high-confidence RDKit-valid visible steps if needed."
            )
        visual_record = execute_local_tool(
            "extract_visual_literature_chain",
            {
                "pdf_evidence": pdf_evidence,
                "source_ref": source.source_ref,
                "source_title": source.source_title,
                "expected_labels": list(source.expected_labels),
                "route_sequence_hint": hint,
                "allow_partial_chain_without_target_match": source.allow_partial_chain_without_target_match,
                "timeout_s": float(args.visual_timeout_s),
                "output_dir": str(source_dir / f"visual_chain_attempt_{attempt:02d}"),
            },
            state,
        )
        visual_records.append(visual_record)
        tool_calls.append(visual_record.to_dict())
        if int(((visual_record.output or {}).get("result") or {}).get("candidate_step_count") or 0) > 0:
            break
    if visual_record is None:
        raise RuntimeError("visual extraction did not run")

    candidate_path = _candidate_path_from_record_or_state(visual_record.output, state)
    repair_record = None
    if candidate_path and source.condition_repairs:
        repair_record = execute_local_tool(
            "apply_source_text_condition_repairs",
            {
                "candidate_chain_path": str(candidate_path),
                "condition_repairs": [dict(item) for item in source.condition_repairs],
                "source_ref": source.source_ref,
                "output_dir": str(source_dir / "condition_repairs"),
            },
            state,
        )
        tool_calls.append(repair_record.to_dict())
        candidate_path = _candidate_path_from_record_or_state(repair_record.output, state) or candidate_path

    validation_record = None
    curator_record = None
    source_detail_steps: list[dict[str, Any]] = []
    rejected_consumables: list[dict[str, Any]] = []
    if candidate_path:
        validation_record = execute_local_tool(
            "validate_literature_intermediate_chain",
            {
                "candidate_chain_path": str(candidate_path),
                "allow_partial_chain_without_target_match": source.allow_partial_chain_without_target_match,
                "require_contiguous": False,
                "output_dir": str(source_dir / "validation"),
            },
            state,
        )
        tool_calls.append(validation_record.to_dict())
        validation = dict((validation_record.output or {}).get("result") or {})
        if int((validation.get("summary") or {}).get("accepted_step_count") or 0) > 0:
            curator_record = execute_local_tool(
                "build_source_detail_curator_records",
                {
                    "validation": validation,
                    "source_ref": source.source_ref,
                    "source_title": source.source_title,
                    "evidence_refs": evidence_refs,
                    "record_id": f"{source.key}_visual_chain_curator_records",
                    "main_reactant_only": True,
                    "output_dir": str(source_dir / "source_detail"),
                },
                state,
            )
            tool_calls.append(curator_record.to_dict())
            resolution = dict((curator_record.output or {}).get("source_detail_resolution") or {})
            source_detail_steps = [dict(item) for item in resolution.get("source_detail_route_steps") or [] if isinstance(item, dict)]
            rejected_consumables = [
                {
                    "reason": gap.get("reason") or "source_detail_gap",
                    "source_ref": gap.get("source_ref") or source.source_ref,
                    "queue_id": gap.get("queue_id") or "",
                    "next_action": gap.get("next_action") or "",
                    "source": "pdf_visual_followup",
                }
                for gap in resolution.get("extraction_gaps") or []
                if isinstance(gap, dict)
            ]

    return {
        "key": source.key,
        "source_ref": source.source_ref,
        "source_title": source.source_title,
        "pdf_status": pdf_record.status,
        "visual_status": visual_record.status,
        "visual_attempt_count": len(visual_records),
        "visual_reasons": list(visual_record.reasons or []),
        "visual_candidate_step_count": int(((visual_record.output or {}).get("result") or {}).get("candidate_step_count") or 0),
        "condition_repair_status": getattr(repair_record, "status", "not_requested") if repair_record else "not_requested",
        "validation_status": getattr(validation_record, "status", "not_run") if validation_record else "not_run",
        "validation_reasons": list(getattr(validation_record, "reasons", []) or []) if validation_record else [],
        "curator_status": getattr(curator_record, "status", "not_run") if curator_record else "not_run",
        "source_detail_step_count": len(source_detail_steps),
        "artifact_dir": str(source_dir),
        "_source_detail_steps": source_detail_steps,
        "_rejected_consumables": rejected_consumables,
    }


def _compile_combined_downstream(
    *,
    state: ToolExecutionState,
    source_detail_steps: list[dict[str, Any]],
    rejected_consumables: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": "open_downstream_consumables.v1",
        "case_id": str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        "planner_handoff": {
            "next_action": "template_plugin_rerun",
            "solved": False,
            "production_kb_promotion": False,
            "source": "pdf_visual_followup",
        },
        "guided_rerun_requests": [],
        "literature_template_cards": [],
        "literature_route_segments": [],
        "executable_template_candidates": [],
        "source_detail_route_steps": source_detail_steps,
        "route_expansion_tasks": [],
        "evolution_candidates": [],
        "rejected_consumables": rejected_consumables,
    }
    out = state.run_dir / "compiled_pdf_visual_downstream"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "downstream_consumables.json", payload)
    compiled = compile_downstream_consumables(
        payload,
        target_smiles=str(state.target_input.get("target_smiles") or ""),
        case_id=str(state.preflight.get("case_id") or state.target_input.get("target_name") or "case"),
        enable_online_anchor_resolution=False,
    )
    refs = write_compiled_downstream_artifacts(compiled, output_dir=out)
    state.artifacts["compiled_downstream"] = compiled
    state.artifacts["compiled_downstream_payload"] = compiled
    state.artifacts["compiled_downstream_pdf_visual_followup"] = {
        "schema_version": "compiled_downstream_pdf_visual_followup_ref.v1",
        "accepted": bool(compiled.get("accepted")),
        "artifact_refs": refs,
        "source_detail_step_count": len(source_detail_steps),
    }
    return compiled


def _load_sources(args: argparse.Namespace) -> list[LiteraturePdfSource]:
    rows: list[dict[str, Any]] = []
    for manifest in args.source_manifest:
        path = Path(manifest).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows.extend(dict(item) for item in data.get("sources") or [] if isinstance(item, dict))
        elif isinstance(data, list):
            rows.extend(dict(item) for item in data if isinstance(item, dict))
    for raw in args.literature_source:
        rows.append(_parse_source_arg(raw))
    return [_source_from_row(row, index=index) for index, row in enumerate(rows, start=1)]


def _parse_source_arg(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise SystemExit("--literature-source JSON must be an object.")
        return data
    parts = text.split("|")
    if len(parts) < 2:
        raise SystemExit("--literature-source must be JSON or path|source_ref|title|pages_csv|labels_csv|hint.")
    return {
        "pdf_path": parts[0],
        "source_ref": parts[1],
        "source_title": parts[2] if len(parts) > 2 else "",
        "page_numbers": _csv_ints(parts[3]) if len(parts) > 3 else [],
        "compound_labels": _csv_strings(parts[4]) if len(parts) > 4 else [],
        "route_sequence_hint": parts[5] if len(parts) > 5 else "",
    }


def _source_from_row(row: dict[str, Any], *, index: int) -> LiteraturePdfSource:
    pdf_path = Path(str(row.get("pdf_path") or row.get("path") or "")).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF source {index} does not exist: {pdf_path}")
    source_ref = str(row.get("source_ref") or row.get("doi") or "").strip()
    if source_ref and source_ref.startswith("10."):
        source_ref = f"doi:{source_ref}"
    if not source_ref:
        raise SystemExit(f"PDF source {index} is missing source_ref/doi.")
    key = _safe_id(str(row.get("key") or row.get("source_key") or source_ref or f"source_{index}"))
    return LiteraturePdfSource(
        key=key,
        pdf_path=pdf_path,
        source_ref=source_ref,
        source_title=str(row.get("source_title") or row.get("title") or ""),
        page_numbers=tuple(int(item) for item in row.get("page_numbers") or row.get("pages") or []),
        compound_labels=tuple(str(item) for item in row.get("compound_labels") or [] if str(item).strip()),
        expected_labels=tuple(str(item) for item in row.get("expected_labels") or [] if str(item).strip()),
        route_sequence_hint=str(row.get("route_sequence_hint") or row.get("hint") or ""),
        condition_repairs=tuple(dict(item) for item in row.get("condition_repairs") or [] if isinstance(item, dict)),
        allow_partial_chain_without_target_match=bool(row.get("allow_partial_chain_without_target_match", True)),
    )


def _candidate_path_from_record_or_state(output: dict[str, Any], state: ToolExecutionState) -> Path | None:
    refs = dict((output or {}).get("artifact_refs") or {})
    value = refs.get("visual_structure_candidate_chain") or ((output or {}).get("result") or {}).get("candidate_chain_path")
    if not value:
        value = state.artifacts.get("visual_structure_candidate_chain_path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _load_optional_artifact(state: ToolExecutionState, key: str, path_value: str) -> None:
    if not str(path_value or "").strip():
        return
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        state.artifacts[key] = json.loads(path.read_text(encoding="utf-8"))


def _analogical_summary(state: ToolExecutionState) -> dict[str, Any]:
    report = dict(state.artifacts.get("analogical_retrosynthesis_hypotheses") or {})
    return {
        "present": bool(report),
        "accepted": bool(report.get("accepted")),
        "hypothesis_count": report.get("hypothesis_count"),
        "source_row_count": report.get("source_row_count"),
        "mode": str(report.get("mode") or ""),
        "reasons": [str(item) for item in report.get("reasons") or []],
    }


def _guided_summary(state: ToolExecutionState) -> dict[str, Any]:
    guided = dict(state.artifacts.get("guided_chemenzy") or {})
    verifier = dict(guided.get("raw_route_verifier") or {})
    return {
        "present": bool(guided),
        "accepted": bool(guided.get("accepted")),
        "route_status": str(guided.get("route_status") or ""),
        "solved": bool(guided.get("solved")),
        "reasons": [str(item) for item in guided.get("reasons") or []],
        "verifier_accepted": bool(verifier.get("accepted")),
        "accepted_route_count": verifier.get("accepted_route_count"),
        "rejected_route_count": verifier.get("rejected_route_count"),
    }


def _route_expansion_summary(state: ToolExecutionState) -> dict[str, Any]:
    expansion = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    return {
        "present": bool(expansion),
        "accepted": bool(expansion.get("accepted")),
        "solved": bool(expansion.get("solved")),
        "subgoal_count": expansion.get("subgoal_count"),
        "accepted_subgoal_count": expansion.get("accepted_subgoal_count"),
        "rejected_subgoal_count": expansion.get("rejected_subgoal_count"),
        "reasons": [str(item) for item in expansion.get("reasons") or []],
    }


def _source_to_dict(source: LiteraturePdfSource) -> dict[str, Any]:
    return {
        "key": source.key,
        "pdf_path": str(source.pdf_path),
        "source_ref": source.source_ref,
        "source_title": source.source_title,
        "page_numbers": list(source.page_numbers),
        "compound_labels": list(source.compound_labels),
        "expected_labels": list(source.expected_labels),
        "route_sequence_hint": source.route_sequence_hint,
        "condition_repair_count": len(source.condition_repairs),
        "allow_partial_chain_without_target_match": source.allow_partial_chain_without_target_match,
    }


def _run_dir(args: argparse.Namespace) -> Path:
    if str(args.output_dir or "").strip():
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(args.output_root).expanduser().resolve() / f"{_safe_id(args.run_prefix)}_{_safe_id(args.target_name)}_{timestamp}"


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in str(value or "").split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _safe_id(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part) or "source"


if __name__ == "__main__":
    main()
