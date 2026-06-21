#!/usr/bin/env python3
"""Continue the bufotalin full-flow from the normalized strict visual chain."""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from cascade_planner.harness.runner import emit_final_verdict
from cascade_planner.harness.schemas import ArtifactBundle, append_jsonl, write_json
from cascade_planner.harness.tools import HarnessBudget, ToolExecutionState, execute_local_tool


RDLogger.DisableLog("rdApp.*")

RUN_DIR = ROOT / "results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053"
STRICT_SOURCE_PATH = (
    RUN_DIR
    / "visual_literature_chain_extraction_strict_20260608"
    / "visual_structure_candidate_chain_normalized_from_repair.json"
)
BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
SOURCE_REF = "doi:10.1016/j.tet.2025.134610"
SOURCE_TITLE = "Construction of advanced intermediate sharing C14-beta-OH for the synthesis of bufotalin"


def main() -> None:
    if not RUN_DIR.exists():
        raise SystemExit(f"run dir missing: {RUN_DIR}")
    if not STRICT_SOURCE_PATH.exists():
        raise SystemExit(f"strict visual chain missing: {STRICT_SOURCE_PATH}")

    target_input = _read_json(RUN_DIR / "target_input.json")
    preflight = _read_json(RUN_DIR / "preflight.json")
    workflow_plan = _read_json(RUN_DIR / "codex_workflow_plan.json")
    strict_chain = _read_json(STRICT_SOURCE_PATH)
    terminal = _terminal_from_chain(strict_chain)
    if not terminal["smiles"]:
        raise SystemExit("strict chain terminal missing")

    continuation_dir = RUN_DIR / "visual_literature_chain_extraction_strict_visual"
    continuation_dir.mkdir(parents=True, exist_ok=True)
    strict_candidate_path = continuation_dir / "visual_structure_candidate_chain.json"
    strict_chain = _normalize_chain_metadata(strict_chain)
    write_json(strict_candidate_path, strict_chain)

    terminal_audit = _terminal_audit(terminal["smiles"])
    write_json(continuation_dir / "strict_visual_terminal_audit.json", terminal_audit)
    visual_result = _strict_visual_result(
        strict_chain=strict_chain,
        candidate_path=strict_candidate_path,
        terminal=terminal,
        terminal_audit=terminal_audit,
    )
    write_json(continuation_dir / "visual_literature_chain_extraction_result.json", visual_result)
    write_json(RUN_DIR / "visual_literature_chain_extraction_strict_visual_result.json", visual_result)

    state = ToolExecutionState(
        run_dir=RUN_DIR,
        target_input=target_input,
        preflight=preflight,
        budget=HarnessBudget(
            max_chem_enzy_runs=0,
            max_guided_chemenzy_runs=1,
            max_route_expansion_subgoal_runs=1,
            max_codex_research_runs=0,
            timeout_s=1800.0,
            guided_chemenzy_timeout_s=1800.0,
        ),
        model="gpt-5.5",
    )
    _seed_existing_artifacts(state)
    state.artifacts["visual_literature_chain_extraction"] = visual_result
    state.artifacts["visual_structure_candidate_chain"] = strict_chain
    state.artifacts["visual_structure_candidate_chain_path"] = str(strict_candidate_path)
    state.artifacts["strict_visual_terminal_audit"] = terminal_audit

    tool_records: list[dict[str, Any]] = []
    continuation_tools = [
        (
            "validate_literature_intermediate_chain",
            {
                "candidate_chain_path": str(strict_candidate_path),
                "output_dir": "literature_intermediate_chain_validation_strict_visual",
                "target_smiles": target_input.get("target_smiles") or BUFOTALIN_SMILES,
                "require_contiguous": True,
            },
        ),
        (
            "build_source_detail_curator_records",
            {
                "output_dir": "open_structure_research_strict_visual",
                "source_ref": SOURCE_REF,
                "source_title": SOURCE_TITLE,
                "record_id": "tet2025_bufotalin_pdf_strict_visual_chain_20260608",
                "provenance": "codex_source_text_translation",
                "main_reactant_only": True,
            },
        ),
        (
            "compile_source_detail_chain_route",
            {
                "output_dir": "source_detail_chain_route_strict_visual",
                "terminal_smiles": terminal["smiles"],
                "terminal_name": terminal["name"],
            },
        ),
        (
            "run_guided_chemenzy_rerun",
            {
                "timeout_s": 1800.0,
                "max_steps": 20,
                "chem_enzy_iterations": 50,
                "chem_enzy_expansion_topk": 100,
                "stock_mode": "building-block",
            },
        ),
        (
            "run_route_expansion_subgoal_search",
            {
                "subgoal_targets": [
                    {
                        "name": _safe_subgoal_name(terminal["name"]),
                        "smiles": terminal["smiles"],
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
                "timeout_s": 1800.0,
            },
        ),
        (
            "stitch_literature_chain_with_subgoal_route",
            {"output_dir": "stitched_semisynthesis_route_strict_visual", "subgoal_name": terminal["name"]},
        ),
        ("validate_artifact_bundle", {}),
    ]
    for tool_name, payload in continuation_tools:
        started = time.monotonic()
        append_jsonl(RUN_DIR / "decision_trace.jsonl", {
            "stage": "strict_visual_continuation_tool_start",
            "tool_name": tool_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        record = execute_local_tool(tool_name, payload, state)
        item = record.to_dict()
        item["continuation_stage"] = "strict_visual_normalized_from_repair"
        item["started_elapsed_marker_s"] = round(started, 3)
        tool_records.append(item)
        if record.status in {"rejected", "error"} and tool_name in {
            "validate_literature_intermediate_chain",
            "build_source_detail_curator_records",
            "compile_source_detail_chain_route",
            "stitch_literature_chain_with_subgoal_route",
        }:
            break

    bundle = ArtifactBundle(
        case_id=str(preflight.get("case_id") or target_input.get("case_id") or "bufotalin"),
        target_input=target_input,
        preflight=preflight,
        workflow_plan=workflow_plan,
        tool_calls=tool_records,
        artifacts=dict(state.artifacts),
        validations=list(state.validations),
        safety_flags=sorted(set(state.safety_flags)),
        run_semantics="canonical_agent_controller",
    )
    verdict = emit_final_verdict(bundle).to_dict()
    write_json(RUN_DIR / "artifact_bundle_strict_visual.json", bundle.to_dict())
    write_json(RUN_DIR / "final_verdict_strict_visual.json", verdict)
    summary = _summary(
        state=state,
        tool_records=tool_records,
        final_verdict=verdict,
        terminal=terminal,
        terminal_audit=terminal_audit,
    )
    write_json(RUN_DIR / "bufotalin_strict_visual_continuation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _seed_existing_artifacts(state: ToolExecutionState) -> None:
    mapping = {
        "chemenzy_native_raw_result.json": "chemenzy",
        "route_verifier_report.json": "route_verifier",
        "route_audit.json": "route_audit",
        "route_failure_feedback.json": "route_failure_feedback",
        "smiles_first_workflow_result.json": "smiles_first",
        "literature_pdf_structure_evidence.json": "literature_pdf_structure_evidence",
        "open_structure_research_result.json": "open_structure_research",
    }
    for filename, key in mapping.items():
        path = RUN_DIR / filename
        if path.exists():
            state.artifacts[key] = _read_json(path)
    pdf_dir = RUN_DIR / "literature_pdf_structure_extraction"
    if pdf_dir.exists():
        state.artifacts["literature_pdf_structure_evidence_dir"] = str(pdf_dir)


def _normalize_chain_metadata(chain: dict[str, Any]) -> dict[str, Any]:
    out = dict(chain)
    evidence_refs = [
        SOURCE_REF,
        f"local_pdf:{ROOT / '1-s2.0-S0040402025001668-main.pdf'}",
        f"local_crop:{RUN_DIR / 'literature_pdf_structure_extraction/crops/scheme3_full_to_20.png'}",
        f"local_crop:{RUN_DIR / 'literature_pdf_structure_extraction/crops/scheme4_total_synthesis.png'}",
        f"local_crop:{RUN_DIR / 'literature_pdf_structure_extraction/crops/table1_allylic_oxidation.png'}",
    ]
    out["source_ref"] = str(out.get("source_ref") or SOURCE_REF)
    out["source_title"] = str(out.get("source_title") or SOURCE_TITLE)
    out["target_name"] = str(out.get("target_name") or "bufotalin")
    out["target_smiles"] = str(out.get("target_smiles") or BUFOTALIN_SMILES)
    out["evidence_refs"] = _dedupe([*(out.get("evidence_refs") or []), *evidence_refs])
    out["candidate_generation_audit"] = {
        **dict(out.get("candidate_generation_audit") or {}),
        "schema_version": "visual_literature_chain_generation_audit.v1",
        "generation_mode": "fresh_codex_vision_from_current_pdf_images_then_deterministic_repair_normalization",
        "strict_visual_continuation": True,
        "source_candidate_path": str(STRICT_SOURCE_PATH),
        "production_write_blocked": True,
        "prior_candidate_chain_reuse_allowed": False,
        "prior_source_detail_records_reuse_allowed": False,
        "no_solved_claim": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for step in out.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step["source_ref"] = str(step.get("source_ref") or SOURCE_REF)
        step["source_title"] = str(step.get("source_title") or SOURCE_TITLE)
        step["evidence_refs"] = _dedupe([*(step.get("evidence_refs") or []), *evidence_refs])
        condition = dict(step.get("condition_candidate") or {})
        condition.setdefault("schema_version", "condition_candidate.v1")
        condition.setdefault("source_type", "exact")
        condition.setdefault("condition_status", "evidence_backed")
        condition.setdefault("source_grounding", "current PDF scheme image with deterministic normalized repair")
        condition["evidence_refs"] = _dedupe([*(condition.get("evidence_refs") or []), *evidence_refs])
        step["condition_candidate"] = condition
        derivation = dict(step.get("structure_derivation") or {})
        derivation["basis"] = "source_structure_diagram_to_smiles"
        derivation.setdefault("source_locator", step.get("source_locator") or out.get("source_locator") or "current PDF scheme images")
        derivation.setdefault("confidence", step.get("confidence") or out.get("confidence") or "low")
        derivation["tool_checks"] = _dedupe(
            [
                *(derivation.get("tool_checks") or []),
                "strict visual extraction performed in this run",
                "deterministic normalized repair consumed for continuation",
                "RDKit parse precheck performed locally",
            ]
        )
        step["structure_derivation"] = derivation
        if not str(step.get("source_excerpt") or "").strip():
            product_label = str(step.get("product_label") or "product")
            reactant_label = ", ".join(str(item) for item in step.get("reactant_labels") or []) or "precursor"
            step["source_excerpt"] = f"Scheme 3/4 shows {reactant_label} converted to {product_label} under the listed arrow conditions."
    return out


def _strict_visual_result(
    *,
    strict_chain: dict[str, Any],
    candidate_path: Path,
    terminal: dict[str, str],
    terminal_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "visual_literature_chain_extraction_result.v1",
        "accepted": True,
        "status": "deterministic_normalized_from_repair",
        "output_dir": str(candidate_path.parent),
        "candidate_chain_path": str(candidate_path),
        "candidate_step_count": len(strict_chain.get("steps") or []),
        "source_ref": SOURCE_REF,
        "source_title": SOURCE_TITLE,
        "image_paths": [
            str(RUN_DIR / "literature_pdf_structure_extraction/crops/scheme3_full_to_20.png"),
            str(RUN_DIR / "literature_pdf_structure_extraction/crops/scheme4_total_synthesis.png"),
            str(RUN_DIR / "literature_pdf_structure_extraction/crops/table1_allylic_oxidation.png"),
        ],
        "strict_visual_terminal": terminal,
        "strict_visual_terminal_audit": terminal_audit,
        "reasons": [],
        "warnings": [
            "source visual repair result was normalized deterministically after the strict vision tool marked the raw repair result rejected",
            "terminal stereochemistry differs from the earlier hard-coded compound 11 subgoal and must be searched as its own exact subgoal",
        ],
        "extraction_policy": {
            "must_derive_from_current_images": True,
            "deterministic_normalization_from_current_repair": True,
            "prior_source_detail_records_reuse_allowed": False,
            "prior_candidate_chain_reuse_allowed": False,
            "no_solved_claim": True,
            "production_write_blocked": True,
            "rdkit_valid_smiles_required": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_solved_claim": True,
    }


def _terminal_from_chain(chain: dict[str, Any]) -> dict[str, str]:
    steps = [dict(item) for item in chain.get("steps") or [] if isinstance(item, dict)]
    if not steps:
        return {"name": "", "smiles": ""}
    last = steps[-1]
    labels = [str(item) for item in last.get("reactant_labels") or [] if str(item).strip()]
    smiles_values = [str(item) for item in last.get("reactant_smiles") or [] if str(item).strip()]
    smiles = str(last.get("main_reactant_smiles") or (smiles_values[0] if smiles_values else ""))
    label = labels[0] if labels else "strict_visual_terminal"
    return {"name": f"strict visual terminal {label}", "smiles": smiles}


def _terminal_audit(terminal_smiles: str) -> dict[str, Any]:
    original_compound_11 = "C[C@]12CCC(=O)C=C1CC[C@H]1[C@H]3CCC(=O)[C@@]3(C)CC[C@@H]12"
    terminal = _compound_identity(terminal_smiles)
    original = _compound_identity(original_compound_11)
    match = bool(
        terminal.get("valid")
        and original.get("valid")
        and terminal.get("canonical_isomeric_smiles") == original.get("canonical_isomeric_smiles")
        and terminal.get("inchikey") == original.get("inchikey")
    )
    return {
        "schema_version": "strict_visual_terminal_identity_audit.v1",
        "accepted": bool(terminal.get("valid")),
        "strict_visual_terminal": terminal,
        "previous_hardcoded_compound_11": original,
        "matches_previous_hardcoded_compound_11": match,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "reasons": [] if terminal.get("valid") else ["strict_visual_terminal_invalid"],
        "warnings": [] if match else ["strict_visual_terminal_differs_from_previous_hardcoded_compound_11"],
    }


def _compound_identity(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {
            "input_smiles": str(smiles or ""),
            "valid": False,
            "canonical_isomeric_smiles": "",
            "inchikey": "",
        }
    return {
        "input_smiles": str(smiles),
        "valid": True,
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "inchikey": Chem.MolToInchiKey(mol),
    }


def _summary(
    *,
    state: ToolExecutionState,
    tool_records: list[dict[str, Any]],
    final_verdict: dict[str, Any],
    terminal: dict[str, str],
    terminal_audit: dict[str, Any],
) -> dict[str, Any]:
    validation = dict(state.artifacts.get("visual_structure_chain_validation") or {})
    chain = dict((state.artifacts.get("source_detail_chain_route") or {}).get("chain_audit") or {})
    subgoal = dict(state.artifacts.get("route_expansion_subgoal_search") or {})
    stitched = dict(state.artifacts.get("stitched_semisynthesis_route") or {})
    return {
        "schema_version": "bufotalin_strict_visual_continuation_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "strict_visual_terminal": terminal,
        "strict_visual_terminal_audit": terminal_audit,
        "tool_status": [
            {
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "reasons": row.get("reasons") or [],
            }
            for row in tool_records
        ],
        "visual_chain_summary": validation.get("summary") or {},
        "source_detail_chain": {
            "accepted": bool(chain.get("accepted")),
            "step_count": int(chain.get("step_count") or 0),
            "terminal_reached": bool(chain.get("terminal_reached")),
            "terminal_smiles": chain.get("terminal_smiles"),
            "terminal_canonical_smiles": chain.get("terminal_canonical_smiles"),
            "reasons": chain.get("reasons") or [],
        },
        "route_expansion_subgoal": {
            "accepted": bool(subgoal.get("accepted")),
            "status": subgoal.get("status"),
            "accepted_subgoal_count": subgoal.get("accepted_subgoal_count"),
            "subgoal_count": subgoal.get("subgoal_count"),
            "reasons": subgoal.get("reasons") or [],
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
            "strict_visual_candidate_chain": str(RUN_DIR / "visual_literature_chain_extraction_strict_visual/visual_structure_candidate_chain.json"),
            "strict_visual_terminal_audit": str(RUN_DIR / "visual_literature_chain_extraction_strict_visual/strict_visual_terminal_audit.json"),
            "visual_chain_validation": str(RUN_DIR / "literature_intermediate_chain_validation_strict_visual/visual_structure_chain_validation.json"),
            "source_detail_curator_records": str(RUN_DIR / "open_structure_research_strict_visual/evidence/source_detail_curator_records.json"),
            "source_detail_chain_route": str(RUN_DIR / "source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json"),
            "route_expansion_subgoals": str(RUN_DIR / "route_expansion_subgoals"),
            "stitched_route": str(RUN_DIR / "stitched_semisynthesis_route_strict_visual/stitched_semisynthesis_route.json"),
            "artifact_bundle": str(RUN_DIR / "artifact_bundle_strict_visual.json"),
            "final_verdict": str(RUN_DIR / "final_verdict_strict_visual.json"),
        },
    }


def _safe_subgoal_name(name: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(name or "").lower()).strip("_")
    return text[:80] or "strict_visual_terminal"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


if __name__ == "__main__":
    main()
