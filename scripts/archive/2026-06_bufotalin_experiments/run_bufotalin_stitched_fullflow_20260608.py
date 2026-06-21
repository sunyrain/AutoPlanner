#!/usr/bin/env python3
"""Run the bufotalin stitched full-flow replay with local PDF evidence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.codex_plan import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_PATH,
    DEFAULT_WIRE_API,
    _read_key,
    _write_codex_home,
    plan_workflow_with_codex,
)
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.schemas import ArtifactBundle, append_jsonl, write_json
from cascade_planner.harness.tools import HarnessBudget, ToolExecutionState, execute_local_tool


BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_FAMILY = "bufotalin, bufadienolide, steroid, C17 2-pyrone"
PDF_PATH = ROOT / "1-s2.0-S0040402025001668-main.pdf"
DOI = "doi:10.1016/j.tet.2025.134610"
SOURCE_TITLE = "Construction of advanced intermediate sharing C14-beta-OH for the synthesis of bufotalin"
COMPOUND_11_SMILES = "C[C@]12CCC(=O)C=C1CC[C@H]1[C@H]3CCC(=O)[C@@]3(C)CC[C@@H]12"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--skip-live-planner", action="store_true")
    parser.add_argument("--skip-native-chemenzy", action="store_true")
    parser.add_argument("--skip-smiles-first", action="store_true")
    parser.add_argument("--skip-open-research", action="store_true")
    parser.add_argument("--skip-visual-audit", action="store_true")
    parser.add_argument("--skip-guided-chemenzy", action="store_true")
    parser.add_argument("--skip-subgoal-search", action="store_true")
    parser.add_argument("--native-timeout-s", type=float, default=1200.0)
    parser.add_argument("--guided-timeout-s", type=float, default=1800.0)
    parser.add_argument("--open-research-timeout-s", type=float, default=900.0)
    parser.add_argument("--visual-audit-timeout-s", type=float, default=300.0)
    parser.add_argument("--visual-chain-timeout-s", type=float, default=1200.0)
    parser.add_argument("--subgoal-timeout-s", type=float, default=1800.0)
    parser.add_argument("--model", default="gpt-5.5")
    args = parser.parse_args()

    run_dir = Path(args.output_dir).resolve() if args.output_dir else _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_calls.jsonl").touch()

    target_input = {
        "schema_version": "codex_entry_target_input.v1",
        "target_name": "bufotalin",
        "target_smiles": BUFOTALIN_SMILES,
        "family_hint": BUFOTALIN_FAMILY,
        "case_id": "",
        "enable_online_anchor_resolution": False,
    }
    preflight = run_preflight(type("Target", (), target_input)())
    target_input["case_id"] = str(preflight.get("case_id") or "bufotalin")
    budget = HarnessBudget(
        max_chem_enzy_runs=0 if args.skip_native_chemenzy else 1,
        max_guided_chemenzy_runs=0 if args.skip_guided_chemenzy else 1,
        max_route_expansion_subgoal_runs=0 if args.skip_subgoal_search else 1,
        max_codex_research_runs=0 if args.skip_open_research else 1,
        timeout_s=max(args.native_timeout_s, args.guided_timeout_s, args.subgoal_timeout_s),
        chem_enzy_timeout_s=args.native_timeout_s,
        guided_chemenzy_timeout_s=max(args.guided_timeout_s, args.subgoal_timeout_s),
        open_research_timeout_s=args.open_research_timeout_s,
    )
    write_json(run_dir / "target_input.json", target_input)
    write_json(run_dir / "preflight.json", preflight)
    write_json(run_dir / "budget.json", budget.to_dict())

    planner_record = _run_live_planner(args=args, run_dir=run_dir, target_input=target_input, preflight=preflight)
    workflow_plan = _execution_plan(args=args, case_id=target_input["case_id"])
    write_json(run_dir / "codex_planner_run_record.json", planner_record)
    write_json(run_dir / "codex_workflow_plan.json", workflow_plan)

    state = ToolExecutionState(
        run_dir=run_dir,
        target_input=target_input,
        preflight=preflight,
        budget=budget,
        model=args.model,
    )
    tool_records: list[dict[str, Any]] = []

    _run_tool_sequence(args=args, state=state, tool_records=tool_records)
    final_verdict = _emit_bundle_and_verdict(
        run_dir=run_dir,
        state=state,
        workflow_plan=workflow_plan,
        tool_records=tool_records,
    )
    summary = _summary(
        args=args,
        run_dir=run_dir,
        state=state,
        planner_record=planner_record,
        tool_records=tool_records,
        final_verdict=final_verdict,
    )
    write_json(run_dir / "bufotalin_stitched_fullflow_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _run_live_planner(*, args: argparse.Namespace, run_dir: Path, target_input: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    if args.skip_live_planner:
        return {
            "schema_version": "codex_entry_planner_run.v1",
            "accepted": False,
            "mode": "skipped",
            "reasons": ["skip_live_planner_requested"],
        }
    return plan_workflow_with_codex(
        target_input=target_input,
        preflight=preflight,
        run_dir=run_dir,
        timeout_s=300.0,
        model=args.model,
    )


def _execution_plan(*, args: argparse.Namespace, case_id: str) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    if not args.skip_native_chemenzy:
        tools.extend(
            [
                {
                    "tool_name": "run_chemenzy",
                    "payload": {
                        "search_preset": "thorough",
                        "max_steps": 20,
                        "chem_enzy_iterations": 50,
                        "chem_enzy_expansion_topk": 100,
                        "stock_mode": "building-block",
                        "timeout_s": args.native_timeout_s,
                    },
                },
                {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
            ]
        )
    if not args.skip_smiles_first:
        tools.append({"tool_name": "run_smiles_first_literature_workflow", "payload": {"frontier_smiles": BUFOTALIN_SMILES}})
    if not args.skip_open_research:
        tools.append({"tool_name": "run_open_structure_research_agent", "payload": {"timeout_s": args.open_research_timeout_s}})
    tools.extend(
        [
            {
                "tool_name": "extract_pdf_literature_structures",
                "payload": {
                    "pdf_path": str(PDF_PATH),
                    "output_dir": "literature_pdf_structure_extraction",
                    "page_numbers": [3, 4, 5, 6],
                    "render_zoom": 5.0,
                    "compound_labels": [
                        "11",
                        "24",
                        "25",
                        "23",
                        "26",
                        "27",
                        "28",
                        "19",
                        "20",
                        "14",
                        "22",
                        "30",
                        "31",
                        "32",
                        "33",
                        "bufotalin",
                    ],
                    "scheme_crops": _scheme_crops(),
                },
            },
            {
                "tool_name": "extract_visual_literature_chain",
                "payload": {
                    "output_dir": "visual_literature_chain_extraction",
                    "source_ref": DOI,
                    "source_title": SOURCE_TITLE,
                    "target_name": "bufotalin",
                    "target_smiles": BUFOTALIN_SMILES,
                    "timeout_s": args.visual_chain_timeout_s,
                    "image_paths": [
                        "literature_pdf_structure_extraction/crops/scheme3_full_to_20.png",
                        "literature_pdf_structure_extraction/crops/scheme4_total_synthesis.png",
                        "literature_pdf_structure_extraction/crops/table1_allylic_oxidation.png",
                    ],
                    "expected_labels": [
                        "11",
                        "24",
                        "25",
                        "23",
                        "26",
                        "27",
                        "28",
                        "19",
                        "20",
                        "14",
                        "22",
                        "30",
                        "31",
                        "32",
                        "33",
                        "bufotalin",
                    ],
                },
            },
            {
                "tool_name": "validate_literature_intermediate_chain",
                "payload": {
                    "target_smiles": BUFOTALIN_SMILES,
                    "require_contiguous": True,
                },
            },
            {
                "tool_name": "build_source_detail_curator_records",
                "payload": {
                    "output_dir": "open_structure_research",
                    "source_ref": DOI,
                    "source_title": SOURCE_TITLE,
                    "record_id": "tet2025_bufotalin_pdf_visual_chain_rerun_20260608",
                    "provenance": "codex_source_text_translation",
                    "main_reactant_only": True,
                },
            },
            {
                "tool_name": "compile_source_detail_chain_route",
                "payload": {
                    "output_dir": "source_detail_chain_route",
                    "terminal_smiles": COMPOUND_11_SMILES,
                    "terminal_name": "compound 11 exact tet2025 terminal",
                },
            },
        ]
    )
    if not args.skip_guided_chemenzy:
        tools.append({"tool_name": "run_guided_chemenzy_rerun", "payload": {"timeout_s": args.guided_timeout_s}})
    if not args.skip_subgoal_search:
        tools.append(
            {
                "tool_name": "run_route_expansion_subgoal_search",
                "payload": {
                    "subgoal_targets": [
                        {
                            "name": "compound_11_exact_tet2025_terminal",
                            "smiles": COMPOUND_11_SMILES,
                            "exact_target_override": True,
                            "target_equivalence_audit_required": True,
                            "max_depth": 20,
                            "max_iterations": 50,
                            "expansion_topk": 100,
                        }
                    ],
                    "max_targets": 1,
                    "search_preset": "thorough",
                    "max_steps": 20,
                    "chem_enzy_iterations": 50,
                    "chem_enzy_expansion_topk": 100,
                    "stock_mode": "building-block",
                    "timeout_s": args.subgoal_timeout_s,
                },
            }
        )
    tools.extend(
        [
            {"tool_name": "stitch_literature_chain_with_subgoal_route", "payload": {"subgoal_name": "compound 11 exact tet2025 terminal"}},
            {"tool_name": "validate_artifact_bundle", "payload": {}},
            {"tool_name": "emit_final_verdict", "payload": {}},
        ]
    )
    return {
        "schema_version": "codex_entry_workflow_plan.v1",
        "case_id": case_id,
        "recommended_strategy": "hybrid",
        "planned_tools": tools,
        "rationale": "bufotalin stitched full-flow replay with live planner record and explicit local PDF evidence payloads",
        "risk_flags": ["steroid_or_polycyclic_core", "natural_product_like"],
        "expected_verdict_floor": "needs_audit",
        "planner_decision_reason": "steroid_or_polycyclic_core",
        "run_semantics": "canonical_agent_controller",
    }


def _run_tool_sequence(*, args: argparse.Namespace, state: ToolExecutionState, tool_records: list[dict[str, Any]]) -> None:
    for row in (_execution_plan(args=args, case_id=str(state.preflight.get("case_id") or "bufotalin")).get("planned_tools") or []):
        tool_name = str(row.get("tool_name") or "")
        if tool_name == "emit_final_verdict":
            append_jsonl(state.run_dir / "decision_trace.jsonl", {"stage": "emit_final_verdict_deferred"})
            continue
        started = time.monotonic()
        record = execute_local_tool(tool_name, dict(row.get("payload") or {}), state)
        item = record.to_dict()
        item["started_elapsed_marker_s"] = round(started, 3)
        tool_records.append(item)
        if tool_name == "extract_pdf_literature_structures" and record.status == "accepted" and not args.skip_visual_audit:
            state.artifacts["codex_visual_scheme_audit"] = _run_codex_visual_scheme_audit(
                run_dir=state.run_dir,
                model=state.model,
                timeout_s=float(args.visual_audit_timeout_s),
            )
        if record.status in {"rejected", "error"} and tool_name in {
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "validate_literature_intermediate_chain",
            "build_source_detail_curator_records",
            "compile_source_detail_chain_route",
            "stitch_literature_chain_with_subgoal_route",
        }:
            break


def _emit_bundle_and_verdict(
    *,
    run_dir: Path,
    state: ToolExecutionState,
    workflow_plan: dict[str, Any],
    tool_records: list[dict[str, Any]],
) -> dict[str, Any]:
    bundle = ArtifactBundle(
        case_id=str(state.preflight.get("case_id") or "bufotalin"),
        target_input=dict(state.target_input),
        preflight=dict(state.preflight),
        workflow_plan=workflow_plan,
        tool_calls=tool_records,
        artifacts=dict(state.artifacts),
        validations=list(state.validations),
        safety_flags=sorted(set(state.safety_flags)),
        run_semantics="canonical_agent_controller",
    )
    verdict = emit_final_verdict(bundle)
    write_json(run_dir / "artifact_bundle.json", bundle.to_dict())
    write_json(run_dir / "final_verdict.json", verdict.to_dict())
    return verdict.to_dict()


def _run_codex_visual_scheme_audit(*, run_dir: Path, model: str, timeout_s: float) -> dict[str, Any]:
    out_dir = run_dir / "visual_scheme_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = run_dir / "literature_pdf_structure_extraction" / "crops"
    images = [
        crop_dir / "scheme3_full_to_20.png",
        crop_dir / "scheme4_total_synthesis.png",
        crop_dir / "table1_allylic_oxidation.png",
    ]
    existing_images = [path for path in images if path.exists()]
    if not existing_images:
        result = {
            "schema_version": "codex_visual_scheme_audit.v1",
            "accepted": False,
            "reasons": ["visual_audit_images_missing"],
        }
        write_json(out_dir / "visual_scheme_audit.json", result)
        return result
    executable = shutil.which("codex")
    api_key = _read_key(Path(DEFAULT_KEY_PATH))
    if not executable or not api_key:
        result = {
            "schema_version": "codex_visual_scheme_audit.v1",
            "accepted": False,
            "reasons": ["codex_executable_or_api_key_missing"],
            "image_paths": [str(path) for path in existing_images],
        }
        write_json(out_dir / "visual_scheme_audit.json", result)
        return result
    prompt = (
        "Inspect the attached newly rendered/cropped PDF scheme images for bufotalin. "
        "Return one compact JSON object only. Do not output reaction SMILES and do not claim solved. "
        "Extract visible compound-label order, visible condition/yield text, and whether the image supports "
        "a continuous Scheme 3/Scheme 4 sequence from androstenedione (11) through 24,25,23,26,27,28,19,20,14,22,30,31,32,33 to bufotalin. "
        "Use keys schema_version, accepted, visible_sequence, visible_conditions, limitations, no_solved_claim."
    )
    prompt_path = out_dir / "visual_scheme_audit_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    event_log = out_dir / "codex_visual_events.jsonl"
    stderr_log = out_dir / "codex_visual_stderr.log"
    last_message = out_dir / "codex_visual_last_message.txt"
    command = [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "--cd",
        str(run_dir),
        "--dangerously-bypass-approvals-and-sandbox",
        "--color",
        "never",
        "--output-last-message",
        str(last_message),
    ]
    for image in existing_images:
        command.extend(["--image", str(image)])
    command.append("-")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="autoplanner_bufotalin_visual_") as tmp:
        codex_home = Path(tmp) / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        _write_codex_home(
            codex_home=codex_home,
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
            model=model,
            run_dir=run_dir,
        )
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)
        env["OPENAI_API_KEY"] = api_key
        env.pop("OPENAI_BASE_URL", None)
        try:
            with event_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    command,
                    cwd=str(run_dir),
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                    text=True,
                    env=env,
                )
                try:
                    proc.communicate(input=prompt, timeout=float(timeout_s))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    result = {
                        "schema_version": "codex_visual_scheme_audit.v1",
                        "accepted": False,
                        "status": "timeout",
                        "reasons": ["codex_visual_scheme_audit_timeout"],
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "image_paths": [str(path) for path in existing_images],
                    }
                    write_json(out_dir / "visual_scheme_audit.json", result)
                    return result
        except OSError as exc:
            result = {
                "schema_version": "codex_visual_scheme_audit.v1",
                "accepted": False,
                "status": "error",
                "reasons": [f"codex_visual_scheme_audit_os_error:{type(exc).__name__}"],
                "image_paths": [str(path) for path in existing_images],
            }
            write_json(out_dir / "visual_scheme_audit.json", result)
            return result
    raw_text = last_message.read_text(encoding="utf-8", errors="replace") if last_message.exists() else ""
    parsed: dict[str, Any]
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {}
    result = {
        "schema_version": "codex_visual_scheme_audit.v1",
        "accepted": bool(parsed.get("accepted")) and int(getattr(proc, "returncode", 1)) == 0,
        "status": "completed" if int(getattr(proc, "returncode", 1)) == 0 else "failed",
        "elapsed_s": round(time.monotonic() - started, 3),
        "provider_wire_api": DEFAULT_WIRE_API,
        "image_paths": [str(path) for path in existing_images],
        "parsed_output": parsed,
        "raw_last_message": raw_text[:4000],
        "event_log_path": str(event_log),
        "stderr_log_path": str(stderr_log),
        "reasons": [] if parsed else ["codex_visual_scheme_audit_json_parse_failed"],
        "no_solved_claim": True,
    }
    write_json(out_dir / "visual_scheme_audit.json", result)
    return result


def _scheme_crops() -> list[dict[str, Any]]:
    return [
        {"crop_id": "scheme3_full_to_20", "page_number": 3, "bbox_px": [0, 0, 2977, 2500]},
        {"crop_id": "scheme4_total_synthesis", "page_number": 3, "bbox_px": [0, 1550, 2977, 3050]},
        {"crop_id": "table1_allylic_oxidation", "page_number": 3, "bbox_px": [0, 2800, 1500, 3969]},
    ]


def _summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    state: ToolExecutionState,
    planner_record: dict[str, Any],
    tool_records: list[dict[str, Any]],
    final_verdict: dict[str, Any],
) -> dict[str, Any]:
    stitched = dict(state.artifacts.get("stitched_semisynthesis_route") or {})
    chain = dict((state.artifacts.get("source_detail_chain_route") or {}).get("chain_audit") or {})
    route_expansion = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    pdf_evidence = dict(state.artifacts.get("literature_pdf_structure_evidence") or {})
    visual_extraction = dict(state.artifacts.get("visual_literature_chain_extraction") or {})
    validation = dict(state.artifacts.get("visual_structure_chain_validation") or {})
    return {
        "schema_version": "bufotalin_stitched_fullflow_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "live_planner": {
            "skipped": bool(args.skip_live_planner),
            "accepted": bool(planner_record.get("accepted")),
            "mode": planner_record.get("mode", "live"),
            "normalization_audit": planner_record.get("normalization_audit") or {},
            "planned_tools": [
                row.get("tool_name")
                for row in ((planner_record.get("workflow_plan") or {}).get("planned_tools") or [])
                if isinstance(row, dict)
            ],
        },
        "tool_status": [
            {
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "reasons": row.get("reasons") or [],
            }
            for row in tool_records
        ],
        "pdf_evidence_summary": pdf_evidence.get("summary") or {},
        "visual_literature_chain_extraction": {
            "accepted": bool(visual_extraction.get("accepted")),
            "status": visual_extraction.get("status"),
            "elapsed_s": visual_extraction.get("elapsed_s"),
            "candidate_step_count": visual_extraction.get("candidate_step_count"),
            "candidate_chain_path": visual_extraction.get("candidate_chain_path"),
            "reasons": visual_extraction.get("reasons") or [],
            "extraction_policy": visual_extraction.get("extraction_policy") or {},
        },
        "visual_chain_summary": validation.get("summary") or {},
        "source_detail_chain": {
            "accepted": bool(chain.get("accepted")),
            "step_count": int(chain.get("step_count") or 0),
            "terminal_reached": bool(chain.get("terminal_reached")),
            "terminal_smiles": chain.get("terminal_smiles"),
            "reasons": chain.get("reasons") or [],
        },
        "route_expansion_subgoal": {
            "accepted": bool(route_expansion.get("accepted")),
            "status": route_expansion.get("status"),
            "accepted_subgoal_count": route_expansion.get("accepted_subgoal_count"),
            "subgoal_count": route_expansion.get("subgoal_count"),
        },
        "stitched_route": {
            "accepted": bool(stitched.get("accepted")),
            "solved": bool(stitched.get("solved")),
            "route_status": stitched.get("route_status"),
            "stock_audit_passed": bool(stitched.get("stock_audit_passed")),
            "combined_route": stitched.get("combined_route") or {},
            "reasons": stitched.get("reasons") or [],
            "warnings": stitched.get("warnings") or [],
        },
        "final_verdict": final_verdict,
        "artifact_refs": {
            "planner_record": str(run_dir / "codex_planner_run_record.json"),
            "pdf_evidence": str(run_dir / "literature_pdf_structure_extraction" / "literature_pdf_structure_evidence.json"),
            "visual_literature_chain_extraction": str(run_dir / "visual_literature_chain_extraction" / "visual_literature_chain_extraction_result.json"),
            "visual_candidate_chain": str(run_dir / "visual_literature_chain_extraction" / "visual_structure_candidate_chain.json"),
            "visual_chain_validation": str(run_dir / "literature_intermediate_chain_validation" / "visual_structure_chain_validation.json"),
            "source_detail_curator_records": str(run_dir / "open_structure_research" / "evidence" / "source_detail_curator_records.json"),
            "source_detail_chain_route": str(run_dir / "source_detail_chain_route" / "source_detail_route_chain_audit.json"),
            "route_expansion_subgoals": str(run_dir / "route_expansion_subgoals"),
            "stitched_route": str(run_dir / "stitched_semisynthesis_route" / "stitched_semisynthesis_route.json"),
            "final_verdict": str(run_dir / "final_verdict.json"),
        },
    }


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "shared" / f"bufotalin_stitched_fullflow_existing_pdf_{stamp}"


if __name__ == "__main__":
    main()
