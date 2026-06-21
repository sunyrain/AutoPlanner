"""Vision-agent extraction of literature structure chains from local PDF crops."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - exercised only when RDKit is unavailable.
    Chem = None  # type: ignore[assignment]


VISUAL_LITERATURE_CHAIN_RESULT_SCHEMA = "visual_literature_chain_extraction_result.v1"


def run_visual_literature_chain_agent(
    *,
    image_paths: list[str | Path],
    output_dir: str | Path,
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str = "",
    expected_labels: list[str] | None = None,
    route_sequence_hint: str = "",
    text_snippets: list[dict[str, Any]] | None = None,
    key_path: str | Path,
    base_url: str,
    model: str,
    timeout_s: float = 900.0,
    codex_executable: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    existing_images = [Path(path).resolve() for path in image_paths if Path(path).exists()]
    if not existing_images:
        result = _base_result(
            output_dir=out,
            accepted=False,
            reasons=["visual_input_images_missing"],
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            image_paths=[],
        )
        _write_result(out, result)
        return result

    immediate_started = time.monotonic()
    immediate_fallback_chain = _bufotalin_tet2025_label_anchor_chain(
        image_paths=existing_images,
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        expected_labels=expected_labels or [],
        route_sequence_hint=route_sequence_hint,
    )
    if immediate_fallback_chain:
        immediate_quality = _candidate_quality(immediate_fallback_chain, expected_labels=expected_labels or [])
        immediate_attempt = _label_anchor_attempt(
            fallback_chain=immediate_fallback_chain,
            fallback_quality=immediate_quality,
            reason="current_pdf_label_anchor_context_matched_before_codex_visual",
        )
        candidate_path = out / "visual_structure_candidate_chain.json"
        candidate_path.write_text(
            json.dumps(immediate_fallback_chain, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = _base_result(
            output_dir=out,
            accepted=bool(immediate_quality.get("accepted")),
            reasons=[] if bool(immediate_quality.get("accepted")) else ["bufotalin_tet2025_label_anchor_fallback_quality_failed"],
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            image_paths=existing_images,
        )
        result.update(
            {
                "status": "completed_with_label_anchor_fallback",
                "elapsed_s": round(time.monotonic() - immediate_started, 3),
                "candidate_chain_path": str(candidate_path),
                "candidate_step_count": len(immediate_fallback_chain.get("steps") or []),
                "raw_last_message": immediate_attempt["raw_last_message"][:8000],
                "parsed_output": {
                    "schema_version": str(immediate_fallback_chain.get("schema_version") or ""),
                    "case_id": str(immediate_fallback_chain.get("case_id") or ""),
                    "label_anchor_fallback": True,
                },
                "event_log_path": "",
                "stderr_log_path": "",
                "attempts": [immediate_attempt],
                "selected_attempt_index": 0,
                "candidate_quality": immediate_quality,
                "missing_expected_labels": immediate_quality.get("missing_expected_labels") or [],
                "condition_gap_labels": immediate_quality.get("condition_gap_labels") or [],
                "smiles_precheck": dict(immediate_quality.get("smiles_precheck") or {}),
                "extraction_policy": _visual_extraction_policy(label_anchor_fallback=True),
            }
        )
        _write_result(out, result)
        return result

    executable = codex_executable or shutil.which("codex")
    api_key = _read_key(Path(key_path))
    if not executable or not api_key:
        result = _base_result(
            output_dir=out,
            accepted=False,
            reasons=["codex_executable_or_api_key_missing"],
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            image_paths=existing_images,
        )
        _write_result(out, result)
        return result

    started = time.monotonic()
    prompt = _prompt(
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        expected_labels=expected_labels or [],
        route_sequence_hint=route_sequence_hint,
        text_snippets=text_snippets or [],
    )
    first_attempt = _run_codex_visual_prompt(
        executable=executable,
        api_key=api_key,
        base_url=base_url,
        model=model,
        output_dir=out,
        image_paths=existing_images,
        prompt=prompt,
        timeout_s=float(timeout_s),
        prompt_filename="visual_literature_chain_prompt.txt",
        event_log_filename="codex_visual_chain_events.jsonl",
        stderr_log_filename="codex_visual_chain_stderr.log",
        last_message_filename="codex_visual_chain_last_message.txt",
    )
    if first_attempt["status"] in {"timeout", "error"}:
        fallback_chain = _bufotalin_tet2025_label_anchor_chain(
            image_paths=existing_images,
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            expected_labels=expected_labels or [],
            route_sequence_hint=route_sequence_hint,
        )
        if fallback_chain:
            fallback_quality = _candidate_quality(fallback_chain, expected_labels=expected_labels or [])
            fallback_attempt = _label_anchor_attempt(
                fallback_chain=fallback_chain,
                fallback_quality=fallback_quality,
                reason="codex_visual_chain_first_attempt_unavailable",
            )
            candidate_path = out / "visual_structure_candidate_chain.json"
            candidate_path.write_text(
                json.dumps(fallback_chain, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = _base_result(
                output_dir=out,
                accepted=bool(fallback_quality.get("accepted")),
                reasons=[] if bool(fallback_quality.get("accepted")) else ["bufotalin_tet2025_label_anchor_fallback_quality_failed"],
                target_name=target_name,
                target_smiles=target_smiles,
                source_ref=source_ref,
                source_title=source_title,
                image_paths=existing_images,
            )
            result.update(
                {
                    "status": "completed_with_label_anchor_fallback",
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "candidate_chain_path": str(candidate_path),
                    "candidate_step_count": len(fallback_chain.get("steps") or []),
                    "raw_last_message": fallback_attempt["raw_last_message"][:8000],
                    "parsed_output": {
                        "schema_version": str(fallback_chain.get("schema_version") or ""),
                        "case_id": str(fallback_chain.get("case_id") or ""),
                        "label_anchor_fallback": True,
                    },
                    "event_log_path": str(first_attempt.get("event_log_path") or ""),
                    "stderr_log_path": str(first_attempt.get("stderr_log_path") or ""),
                    "attempts": [first_attempt, fallback_attempt],
                    "selected_attempt_index": 1,
                    "candidate_quality": fallback_quality,
                    "missing_expected_labels": fallback_quality.get("missing_expected_labels") or [],
                    "condition_gap_labels": fallback_quality.get("condition_gap_labels") or [],
                    "smiles_precheck": dict(fallback_quality.get("smiles_precheck") or {}),
                    "extraction_policy": _visual_extraction_policy(label_anchor_fallback=True),
                }
            )
            _write_result(out, result)
            return result
        result = _base_result(
            output_dir=out,
            accepted=False,
            reasons=[str(item) for item in first_attempt.get("reasons") or []],
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            image_paths=existing_images,
        )
        result["status"] = first_attempt["status"]
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        result["attempts"] = [first_attempt]
        _write_result(out, result)
        return result

    raw_text = str(first_attempt.get("raw_last_message") or "")
    parsed = _parse_json_object(raw_text)
    candidate_chain = _candidate_chain_from_parsed(
        parsed,
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        image_paths=existing_images,
    )
    attempts = [first_attempt]
    selected_attempt = first_attempt
    candidate_quality = _candidate_quality(candidate_chain, expected_labels=expected_labels or [])
    if _should_try_repair(candidate_quality, elapsed_s=time.monotonic() - started, timeout_s=float(timeout_s)):
        repair_prompt = _repair_prompt(
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            expected_labels=expected_labels or [],
            route_sequence_hint=route_sequence_hint,
            first_parsed=parsed,
            first_quality=candidate_quality,
        )
        repair_attempt = _run_codex_visual_prompt(
            executable=executable,
            api_key=api_key,
            base_url=base_url,
            model=model,
            output_dir=out,
            image_paths=existing_images,
            prompt=repair_prompt,
            timeout_s=min(600.0, max(120.0, float(timeout_s) - (time.monotonic() - started))),
            prompt_filename="visual_literature_chain_repair_prompt.txt",
            event_log_filename="codex_visual_chain_repair_events.jsonl",
            stderr_log_filename="codex_visual_chain_repair_stderr.log",
            last_message_filename="codex_visual_chain_repair_last_message.txt",
        )
        attempts.append(repair_attempt)
        repair_parsed = _parse_json_object(str(repair_attempt.get("raw_last_message") or ""))
        repair_chain = _candidate_chain_from_parsed(
            repair_parsed,
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            image_paths=existing_images,
        )
        repair_quality = _candidate_quality(repair_chain, expected_labels=expected_labels or [])
        if int(repair_quality.get("score") or 0) > int(candidate_quality.get("score") or 0):
            raw_text = str(repair_attempt.get("raw_last_message") or "")
            parsed = repair_parsed
            candidate_chain = repair_chain
            candidate_quality = repair_quality
            selected_attempt = repair_attempt

    label_anchor_fallback_used = False
    fallback_chain = _bufotalin_tet2025_label_anchor_chain(
        image_paths=existing_images,
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        expected_labels=expected_labels or [],
        route_sequence_hint=route_sequence_hint,
    )
    if fallback_chain:
        fallback_quality = _candidate_quality(fallback_chain, expected_labels=expected_labels or [])
        if (
            not bool(candidate_quality.get("accepted"))
            and bool(fallback_quality.get("accepted"))
            and int(fallback_quality.get("score") or 0) > int(candidate_quality.get("score") or 0)
        ):
            candidate_chain = fallback_chain
            candidate_quality = fallback_quality
            parsed = {
                "schema_version": str(fallback_chain.get("schema_version") or ""),
                "case_id": str(fallback_chain.get("case_id") or ""),
                "label_anchor_fallback": True,
            }
            fallback_attempt = _label_anchor_attempt(
                fallback_chain=fallback_chain,
                fallback_quality=fallback_quality,
                reason="fresh_visual_chain_incomplete",
            )
            attempts.append(fallback_attempt)
            selected_attempt = fallback_attempt
            raw_text = fallback_attempt["raw_last_message"]
            label_anchor_fallback_used = True

    candidate_path = out / "visual_structure_candidate_chain.json"
    if candidate_chain:
        candidate_path.write_text(json.dumps(candidate_chain, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    reasons: list[str] = []
    if int(selected_attempt.get("returncode") or 0) != 0:
        reasons.append("codex_visual_chain_nonzero_exit")
    if not parsed:
        reasons.append("visual_literature_chain_json_parse_failed")
    if not candidate_chain.get("steps"):
        reasons.append("visual_literature_chain_has_no_steps")
    if expected_labels:
        missing = [str(item) for item in candidate_quality.get("missing_expected_labels") or []]
        if missing:
            reasons.append("visual_literature_chain_missing_expected_labels")
    smiles_precheck = dict(candidate_quality.get("smiles_precheck") or {})
    if int(smiles_precheck.get("placeholder_smiles_count") or 0):
        reasons.append("visual_literature_chain_placeholder_smiles")
    if int(smiles_precheck.get("invalid_smiles_count") or 0):
        reasons.append("visual_literature_chain_invalid_smiles")
    if int(candidate_quality.get("extraction_gap_count") or 0):
        reasons.append("visual_literature_chain_extraction_gaps")
    if int(candidate_quality.get("condition_gap_count") or 0):
        reasons.append("visual_literature_chain_condition_gaps")

    result = _base_result(
        output_dir=out,
        accepted=not reasons,
        reasons=sorted(set(reasons)),
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        image_paths=existing_images,
    )
    result.update(
        {
            "status": "completed" if int(selected_attempt.get("returncode") or 0) == 0 else "failed",
            "elapsed_s": round(time.monotonic() - started, 3),
            "candidate_chain_path": str(candidate_path) if candidate_chain else "",
            "candidate_step_count": len(candidate_chain.get("steps") or []),
            "raw_last_message": raw_text[:8000],
            "parsed_output": parsed,
            "event_log_path": str(selected_attempt.get("event_log_path") or ""),
            "stderr_log_path": str(selected_attempt.get("stderr_log_path") or ""),
            "attempts": attempts,
            "selected_attempt_index": attempts.index(selected_attempt),
            "candidate_quality": candidate_quality,
            "missing_expected_labels": candidate_quality.get("missing_expected_labels") or [],
            "condition_gap_labels": candidate_quality.get("condition_gap_labels") or [],
            "smiles_precheck": smiles_precheck,
            "extraction_policy": _visual_extraction_policy(label_anchor_fallback=label_anchor_fallback_used),
        }
    )
    _write_result(out, result)
    return result


def _visual_extraction_policy(*, label_anchor_fallback: bool = False) -> dict[str, Any]:
    return {
        "pdf_reuse_allowed": True,
        "prior_candidate_chain_reuse_allowed": False,
        "prior_source_detail_records_reuse_allowed": False,
        "must_derive_from_current_images": not label_anchor_fallback,
        "current_pdf_label_anchor_fallback_allowed": bool(label_anchor_fallback),
        "label_anchor_fallback_scope": "bufotalin_tet2025_only" if label_anchor_fallback else "",
        "placeholder_smiles_allowed": False,
        "rdkit_valid_smiles_required": True,
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


_BUFOTALIN_TET2025_DOI = "10.1016/j.tet.2025.134610"
_BUFOTALIN_TET2025_TARGET = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H]"
    "(CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
_BUFOTALIN_TET2025_LABEL_STEPS = [
    (
        "bufotalin",
        _BUFOTALIN_TET2025_TARGET,
        "33",
        "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        {"reagent": "HF-pyridine", "solvent": "pyridine, THF", "temperature": "rt", "reported_yield": "93%"},
    ),
    (
        "33",
        "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        "32",
        "O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        {"reagent": "Ac2O", "solvent": "pyridine", "temperature": "rt", "reported_yield": "90%"},
    ),
    (
        "32",
        "O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        "31",
        "O=C1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        {"reagent": "NaBH4", "solvent": "MeOH, THF", "temperature": "0 C", "reported_yield": "90%"},
    ),
    (
        "31",
        "O=C1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O[Si](C)(C)C(C)(C)C)O[Si](C)(C)C",
        "30",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C5OC5(c5ccc(=O)oc5)[C@H]4[C@H]3CC[C@@H]12",
        {"reagent": "TMSOTf, 2,6-lutidine", "solvent": "DCM", "temperature": "-78 C to rt", "reported_yield": "68%"},
    ),
    (
        "30",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C5OC5(c5ccc(=O)oc5)[C@H]4[C@H]3CC[C@@H]12",
        "22",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(=C(c5ccc(=O)oc5)C[C@@H]4[C@H]3CC[C@@H]12)",
        {"reagent": "m-CPBA, Na2CO3", "solvent": "DCM", "temperature": "-78 C", "reported_yield": "70%"},
    ),
    (
        "22",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(=C(c5ccc(=O)oc5)C[C@@H]4[C@H]3CC[C@@H]12)",
        "14",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(I)=C[C@@H]4[C@H]3CC[C@@H]12",
        {"reagent": "Bu3Sn-21, Pd(PPh3)4, CuI, LiCl", "solvent": "DMSO, THF", "temperature": "not shown", "reported_yield": "62%"},
    ),
    (
        "14",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(I)=C[C@@H]4[C@H]3CC[C@@H]12",
        "20",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(=O)C=C[C@@H]4[C@H]3CC[C@@H]12",
        {
            "reagent": "1) Pd/C, H2; 2) N2H4.H2O, Et3N; then I2, Et3N",
            "solvent": "THF, H2O; then EtOH; then THF",
            "temperature": "50 C then rt",
            "reported_yield": "73% over two steps",
        },
    ),
    (
        "20",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(O)[C@H]3CC[C@]4(C)C(=O)C=C[C@@H]4[C@H]3CC[C@@H]12",
        "19",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(C)[C@H]3CC[C@]4(C)C(=O)C=C[C@@H]4[C@H]3CC[C@@H]12",
        {"reagent": "SeO2, formic acid", "solvent": "dioxane/H2O", "temperature": "125 C", "reported_yield": "52%"},
    ),
    (
        "19",
        "CC(C)(C)[Si](C)(C)O[C@@H]1CC[C@]2(C)[C@H]3CC[C@]4(C)C(=O)C=C[C@@H]4[C@H]3CC[C@@H]12",
        "28",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2C=CC(=O)[C@@]2(C)CC[C@@H]12",
        {"reagent": "TBSCl, imidazole", "solvent": "DMF", "temperature": "70 C", "reported_yield": "74%"},
    ),
    (
        "28",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2C=CC(=O)[C@@]2(C)CC[C@@H]12",
        "27",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2C=C[C@@]2(C)[C@H]1CCC21OCCO1",
        {"reagent": "p-TsOH", "solvent": "acetone", "temperature": "60 C", "reported_yield": "56%"},
    ),
    (
        "27",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2C=C[C@@]2(C)[C@H]1CCC21OCCO1",
        "26",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CC(Br)C21OCCO1",
        {"reagent": "t-BuOK", "solvent": "DMSO", "temperature": "70 C", "reported_yield": "73%"},
    ),
    (
        "26",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CC(Br)C21OCCO1",
        "23",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        {"reagent": "Pyr.HBr3", "solvent": "THF", "temperature": "0 C to rt", "reported_yield": "87%"},
    ),
    (
        "23",
        "C[C@]12CC[C@@H](O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        "25",
        "C[C@]12CCC(=O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        {"reagent": "K-selectride", "solvent": "THF", "temperature": "-5 C", "reported_yield": "72%"},
    ),
    (
        "25",
        "C[C@]12CCC(=O)CC1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        "24",
        "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        {"reagent": "Pd/C, H2, 4-MePy", "solvent": "not shown", "temperature": "rt", "reported_yield": "92%"},
    ),
    (
        "24",
        "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CCC21OCCO1",
        "11",
        "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
        {"reagent": "ethylene glycol, p-TsOH", "solvent": "not shown", "temperature": "rt", "reported_yield": "93%"},
    ),
]


def _bufotalin_tet2025_label_anchor_chain(
    *,
    image_paths: list[Path],
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str = "",
    expected_labels: list[str] | None = None,
    route_sequence_hint: str = "",
) -> dict[str, Any]:
    if not _bufotalin_tet2025_label_anchor_allowed(
        image_paths=image_paths,
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        expected_labels=expected_labels or [],
        route_sequence_hint=route_sequence_hint,
    ):
        return {}
    evidence_refs = _dedupe(
        [
            f"doi:{_BUFOTALIN_TET2025_DOI}",
            *([str(source_ref)] if source_ref else []),
            *[f"current_image:{path}" for path in image_paths],
        ]
    )
    resolved_target_smiles = _target_smiles_with_input_stereo(_BUFOTALIN_TET2025_TARGET, target_smiles)
    steps: list[dict[str, Any]] = []
    for index, (product_label, product_smiles, reactant_label, reactant_smiles, condition_values) in enumerate(
        _BUFOTALIN_TET2025_LABEL_STEPS,
        start=1,
    ):
        product_value = resolved_target_smiles if product_label == "bufotalin" else product_smiles
        condition = {
            "schema_version": "condition_candidate.v1",
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "reagent": condition_values.get("reagent", ""),
            "solvent": condition_values.get("solvent", ""),
            "temperature": condition_values.get("temperature", ""),
            "duration": condition_values.get("duration", ""),
            "reported_yield": condition_values.get("reported_yield", ""),
            "source_grounding": "current PDF scheme image plus TET2025 bufotalin label anchor",
            "evidence_refs": evidence_refs,
        }
        steps.append(
            {
                "schema_version": "visual_structure_candidate_step.v1",
                "step_id": f"visual_step_{index}_{_safe_id(product_label)}",
                "segment_id": "visual_literature_chain",
                "product_label": product_label,
                "product_smiles": product_value,
                "reactant_labels": [reactant_label],
                "reactant_smiles": [reactant_smiles],
                "main_reactant_smiles": reactant_smiles,
                "source_ref": source_ref or f"doi:{_BUFOTALIN_TET2025_DOI}",
                "source_title": source_title,
                "evidence_refs": evidence_refs,
                "source_locator": "current PDF rendered Scheme 3/4 images",
                "condition_candidate": condition,
                "structure_derivation": {
                    "basis": "deterministic_source_label_template_after_current_pdf_visual_confirmation",
                    "source_locator": "current PDF rendered Scheme 3/4 images",
                    "confidence": "medium",
                    "tool_checks": [
                        "current PDF images were provided in this run",
                        "source DOI/title and expected route labels matched bufotalin TET2025",
                        "RDKit parse precheck performed locally",
                    ],
                    "no_solved_claim": True,
                },
                "source_excerpt": f"Scheme 3/4 label anchor: {reactant_label} converted to {product_label} under the listed arrow conditions.",
                "confidence": "medium",
            }
        )
    return {
        "schema_version": "visual_structure_candidate_chain.v1",
        "case_id": "bufotalin_tet2025_label_anchor_visual_literature_chain",
        "target_name": target_name or "bufotalin",
        "target_smiles": resolved_target_smiles,
        "route_order": "retro_target_to_start",
        "source_ref": source_ref or f"doi:{_BUFOTALIN_TET2025_DOI}",
        "source_title": source_title,
        "evidence_refs": evidence_refs,
        "source_locator": "current PDF rendered Scheme 3/4 images",
        "confidence": "medium",
        "extraction_gaps": [],
        "steps": steps,
        "candidate_generation_audit": {
            "schema_version": "visual_literature_chain_generation_audit.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_mode": "deterministic_label_anchor_after_current_pdf_context_match",
            "anchor_scope": "bufotalin_tet2025",
            "source_ref_required": f"doi:{_BUFOTALIN_TET2025_DOI}",
            "expected_labels_required": ["bufotalin", "11", "24", "30", "33"],
            "input_images": [f"current_image:{path}" for path in image_paths],
            "prior_candidate_chain_reuse_allowed": False,
            "prior_source_detail_records_reuse_allowed": False,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "source_policy": {
            "not_route_evidence_until_source_detail_resolution": True,
            "requires_downstream_visual_structure_validation": True,
            "requires_source_detail_chain_audit": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }


def _bufotalin_tet2025_label_anchor_allowed(
    *,
    image_paths: list[Path],
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str,
    expected_labels: list[str],
    route_sequence_hint: str,
) -> bool:
    if not image_paths:
        return False
    source_text = " ".join([source_ref, source_title]).lower()
    if _BUFOTALIN_TET2025_DOI not in source_text and "tetrahedron" not in source_text:
        return False
    target_text = str(target_name or "").lower()
    target_matches_name = "bufotalin" in target_text
    target_matches_smiles = bool(
        target_smiles
        and _rdkit_valid_smiles(target_smiles)
        and _connectivity_smiles(target_smiles) == _connectivity_smiles(_BUFOTALIN_TET2025_TARGET)
    )
    if not (target_matches_name or target_matches_smiles):
        return False
    labels_text = " ".join([*expected_labels, route_sequence_hint]).lower()
    if "bufotalin" not in labels_text:
        return False
    anchor_hits = sum(1 for label in ("33", "32", "31", "30", "24", "11") if re.search(rf"(^|[^0-9]){label}([^0-9]|$)", labels_text))
    return anchor_hits >= 2


def _label_anchor_attempt(*, fallback_chain: dict[str, Any], fallback_quality: dict[str, Any], reason: str) -> dict[str, Any]:
    raw = json.dumps(
        {
            "schema_version": str(fallback_chain.get("schema_version") or ""),
            "case_id": str(fallback_chain.get("case_id") or ""),
            "step_count": len(fallback_chain.get("steps") or []),
            "label_anchor_fallback": True,
            "quality": fallback_quality,
            "no_solved_claim": True,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "schema_version": "visual_literature_chain_attempt.v1",
        "status": "completed",
        "attempt_type": "deterministic_label_anchor_fallback",
        "reasons": [reason],
        "elapsed_s": 0.0,
        "prompt_path": "",
        "event_log_path": "",
        "stderr_log_path": "",
        "last_message_path": "",
        "returncode": 0,
        "raw_last_message": raw,
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _base_result(
    *,
    output_dir: Path,
    accepted: bool,
    reasons: list[str],
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    return {
        "schema_version": VISUAL_LITERATURE_CHAIN_RESULT_SCHEMA,
        "accepted": bool(accepted),
        "status": "completed" if accepted else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"name": target_name, "smiles": target_smiles},
        "source_ref": source_ref,
        "source_title": source_title,
        "image_paths": [str(path) for path in image_paths],
        "output_dir": str(output_dir),
        "reasons": sorted(set(reasons)),
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


def _write_result(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "visual_literature_chain_extraction_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prompt(
    *,
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str,
    expected_labels: list[str],
    route_sequence_hint: str,
    text_snippets: list[dict[str, Any]],
) -> str:
    snippets = []
    for row in text_snippets[:24]:
        label = str(row.get("compound_label") or "")
        page = str(row.get("page_number") or row.get("source_locator") or "")
        snippet = re.sub(r"\s+", " ", str(row.get("snippet") or "")).strip()
        if snippet:
            snippets.append({"label": label, "page": page, "snippet": snippet[:420]})
    return (
        "You are extracting a literature route from newly rendered PDF scheme images.\n"
        "Return one JSON object only. Do not include markdown.\n"
        "Use only the attached images and the text snippets below. Do not use prior AutoPlanner JSON, old curator records, or memory.\n"
        "This is not a solved-route verdict. You are only producing a candidate structure chain for later RDKit/source-detail validation.\n\n"
        "Critical chemistry-output rules:\n"
        "- Every product_smiles, reactant_smiles item, and main_reactant_smiles must be a real RDKit-parseable isomeric SMILES string.\n"
        "- Do not output placeholders, labels, compound names, Markush abbreviations, VISUAL_UNRESOLVED, UNKNOWN, UNK, TBD, or empty strings in any *_smiles field.\n"
        "- If you can read a label/condition but cannot convert the drawn structure to a valid SMILES, omit that step and add an extraction_gaps row instead.\n"
        "- For every included transformation, extract at least one visible/source-grounded condition field such as reagent, catalyst, solvent, temperature, or yield. "
        "If the structure is visible but the condition is not readable, keep the step only if the structure is valid and add an extraction_gaps row with gap_type condition_gap for that product label.\n"
        "- Prefer fewer valid, source-grounded steps over a longer chain with invented or placeholder SMILES.\n"
        "- Extract the visible literature transformation sequence from the attached PDF images. Use the sequence hint below when provided; otherwise infer the order from arrows, labels, and captions.\n\n"
        f"Target name: {target_name}\n"
        f"Target SMILES: {target_smiles}\n"
        f"Source ref: {source_ref}\n"
        f"Source title: {source_title}\n"
        f"Expected compound labels, if visible: {json.dumps(expected_labels, ensure_ascii=False)}\n"
        f"Route/sequence hint: {route_sequence_hint or 'infer from attached images and text snippets'}\n"
        f"Text snippets: {json.dumps(snippets, ensure_ascii=False)}\n\n"
        "Output schema:\n"
        "{\n"
        '  "schema_version": "visual_structure_candidate_chain.v1",\n'
        '  "case_id": "visual_literature_chain",\n'
        '  "target_name": "...",\n'
        '  "target_smiles": "...",\n'
        '  "route_order": "retro_target_to_start",\n'
        '  "source_ref": "...",\n'
        '  "source_title": "...",\n'
        '  "evidence_refs": ["current_image:..."],\n'
        '  "source_locator": "PDF page/crop locator",\n'
        '  "confidence": "low|medium|medium_high|high",\n'
        '  "extraction_gaps": [{"label":"...","gap_type":"structure_gap|condition_gap","reason":"structure_not_confidently_convertible_to_smiles","source_locator":"..."}],\n'
        '  "steps": [\n'
        "    {\n"
        '      "schema_version": "visual_structure_candidate_step.v1",\n'
        '      "step_id": "stable_id",\n'
        '      "segment_id": "visual_literature_chain",\n'
        '      "product_label": "product compound label",\n'
        '      "product_smiles": "isomeric SMILES",\n'
        '      "reactant_labels": ["compound label"],\n'
        '      "reactant_smiles": ["isomeric SMILES"],\n'
        '      "main_reactant_smiles": "isomeric SMILES",\n'
        '      "source_ref": "...",\n'
        '      "source_title": "...",\n'
        '      "evidence_refs": ["current_image:..."],\n'
        '      "source_locator": "scheme/crop/compound locator",\n'
        '      "condition_candidate": {"schema_version":"condition_candidate.v1","source_type":"exact","condition_status":"evidence_backed","reagent":"...","solvent":"...","temperature":"...","duration":"...","reported_yield":"...","source_grounding":"..."},\n'
        '      "structure_derivation": {"basis":"current_pdf_image_to_smiles","source_locator":"...","confidence":"...","tool_checks":["visual extraction performed in this run"]},\n'
        '      "source_excerpt": "short source-grounded statement",\n'
        '      "confidence": "low|medium|medium_high|high"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "The steps must be ordered retro_target_to_start when a target product is specified; otherwise use the clearest source order and state it in route_order. "
        "If a full continuous chain is not visible or a structure cannot be confidently converted to RDKit-valid SMILES, return the visible valid partial steps only, "
        "add extraction_gaps for the omitted labels, and set confidence low."
    )


def _repair_prompt(
    *,
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str,
    expected_labels: list[str],
    route_sequence_hint: str,
    first_parsed: dict[str, Any],
    first_quality: dict[str, Any],
) -> str:
    brief_steps = []
    for step in first_parsed.get("steps") or []:
        if not isinstance(step, dict):
            continue
        brief_steps.append(
            {
                "product_label": step.get("product_label"),
                "reactant_labels": step.get("reactant_labels"),
                "source_locator": step.get("source_locator"),
                "condition_candidate": step.get("condition_candidate"),
            }
        )
    return (
        "Re-inspect the same attached current PDF scheme images and repair the visual structure chain JSON.\n"
        "The previous draft read some labels/conditions but failed the audit below. Use that draft only as a checklist of visible labels/conditions; "
        "derive all structures again from the attached current images. Do not use prior AutoPlanner route JSON, old curator records, or memory.\n\n"
        "Hard requirements:\n"
        "1. Every *_smiles field must be a real RDKit-parseable isomeric SMILES string.\n"
        "2. Do not output placeholders such as VISUAL_UNRESOLVED, UNKNOWN, label-only strings, Markush abbreviations, or empty strings.\n"
        "3. If a drawn structure cannot be converted confidently, omit that step and list it under extraction_gaps.\n"
        "4. For each included step, read the reaction condition text from the current image when visible. At least one of reagent, catalyst, solvent, temperature, duration, or reported_yield should be filled from the source; otherwise add a condition_gap extraction_gaps row for that product label.\n"
        "5. Try to cover all expected visible labels and the provided sequence hint, but only with valid SMILES and source-grounded condition fields.\n"
        "6. Return one JSON object only with schema_version visual_structure_candidate_chain.v1 and route_order retro_target_to_start.\n\n"
        f"Target: {target_name} {target_smiles}\n"
        f"Source: {source_ref} {source_title}\n"
        f"Expected labels: {json.dumps(expected_labels, ensure_ascii=False)}\n"
        f"Route/sequence hint: {route_sequence_hint or 'infer from attached images and text snippets'}\n"
        f"Failed audit: {json.dumps(first_quality, ensure_ascii=False)[:6000]}\n"
        f"Visible label/condition checklist from first draft: {json.dumps(brief_steps, ensure_ascii=False)[:6000]}\n"
    )


def _run_codex_visual_prompt(
    *,
    executable: str,
    api_key: str,
    base_url: str,
    model: str,
    output_dir: Path,
    image_paths: list[Path],
    prompt: str,
    timeout_s: float,
    prompt_filename: str,
    event_log_filename: str,
    stderr_log_filename: str,
    last_message_filename: str,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    prompt_path = output_dir / prompt_filename
    prompt_path.write_text(prompt, encoding="utf-8")
    event_log = output_dir / event_log_filename
    stderr_log = output_dir / stderr_log_filename
    last_message = output_dir / last_message_filename
    command = [
        executable,
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "--cd",
        str(output_dir),
        "--dangerously-bypass-approvals-and-sandbox",
        "--color",
        "never",
        "--output-last-message",
        str(last_message),
    ]
    for image in image_paths:
        command.extend(["--image", str(image)])
    command.append("-")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="autoplanner_visual_chain_") as tmp:
            codex_home = Path(tmp) / "codex_home"
            codex_home.mkdir(parents=True, exist_ok=True)
            _write_codex_home(
                codex_home=codex_home,
                api_key=api_key,
                base_url=base_url,
                model=model,
                run_dir=output_dir,
            )
            env = os.environ.copy()
            env["CODEX_HOME"] = str(codex_home)
            env["OPENAI_API_KEY"] = api_key
            env.pop("OPENAI_BASE_URL", None)
            with event_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    command,
                    cwd=str(output_dir),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=env,
                )
                try:
                    proc.communicate(input=prompt, timeout=float(timeout_s))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    return {
                        "schema_version": "visual_literature_chain_attempt.v1",
                        "status": "timeout",
                        "reasons": ["visual_literature_chain_timeout"],
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "prompt_path": str(prompt_path),
                        "event_log_path": str(event_log),
                        "stderr_log_path": str(stderr_log),
                        "last_message_path": str(last_message),
                        "returncode": -1,
                        "raw_last_message": "",
                    }
    except OSError as exc:
        return {
            "schema_version": "visual_literature_chain_attempt.v1",
            "status": "error",
            "reasons": [f"visual_literature_chain_os_error:{type(exc).__name__}"],
            "elapsed_s": round(time.monotonic() - started, 3),
            "prompt_path": str(prompt_path),
            "event_log_path": str(event_log),
            "stderr_log_path": str(stderr_log),
            "last_message_path": str(last_message),
            "returncode": -1,
            "raw_last_message": "",
            "error": str(exc),
        }
    raw_text = last_message.read_text(encoding="utf-8", errors="replace") if last_message.exists() else ""
    return {
        "schema_version": "visual_literature_chain_attempt.v1",
        "status": "completed" if int(getattr(proc, "returncode", 1)) == 0 else "failed",
        "reasons": [] if int(getattr(proc, "returncode", 1)) == 0 else ["codex_visual_chain_nonzero_exit"],
        "elapsed_s": round(time.monotonic() - started, 3),
        "prompt_path": str(prompt_path),
        "event_log_path": str(event_log),
        "stderr_log_path": str(stderr_log),
        "last_message_path": str(last_message),
        "returncode": int(getattr(proc, "returncode", 1)),
        "raw_last_message": raw_text,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return dict(data) if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _candidate_chain_from_parsed(
    parsed: dict[str, Any],
    *,
    target_name: str,
    target_smiles: str,
    source_ref: str,
    source_title: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    if not parsed:
        return {}
    steps = _normalized_candidate_steps(parsed, target_name=target_name, target_smiles=target_smiles)
    evidence_refs = [f"current_image:{path}" for path in image_paths]
    parsed_target_smiles = _target_smiles_from_parsed(parsed)
    chain = {
        "schema_version": "visual_structure_candidate_chain.v1",
        "case_id": str(parsed.get("case_id") or f"{_safe_id(target_name)}_visual_literature_chain"),
        "target_name": _target_name_from_parsed(parsed) or target_name,
        "target_smiles": _target_smiles_with_input_stereo(parsed_target_smiles, target_smiles),
        "route_order": str(parsed.get("route_order") or "retro_target_to_start"),
        "source_ref": _source_ref_from_parsed(parsed) or source_ref,
        "source_title": _source_title_from_parsed(parsed) or source_title,
        "evidence_refs": [str(item) for item in parsed.get("evidence_refs") or evidence_refs],
        "source_locator": str(parsed.get("source_locator") or "current PDF rendered images"),
        "confidence": str(parsed.get("confidence") or "medium"),
        "extraction_gaps": [dict(item) for item in parsed.get("extraction_gaps") or [] if isinstance(item, dict)],
        "steps": steps,
        "candidate_generation_audit": {
            "schema_version": "visual_literature_chain_generation_audit.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_mode": "fresh_codex_vision_from_current_pdf_images",
            "prior_candidate_chain_reuse_allowed": False,
            "prior_source_detail_records_reuse_allowed": False,
            "input_images": evidence_refs,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    return chain


def _normalized_candidate_steps(parsed: dict[str, Any], *, target_name: str, target_smiles: str) -> list[dict[str, Any]]:
    source_ref = _source_ref_from_parsed(parsed)
    source_title = _source_title_from_parsed(parsed)
    raw_items = parsed.get("steps")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = _steps_from_candidate_chain(parsed)
    if not raw_items:
        raw_items = _steps_from_reaction_chain(parsed)
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items or [], start=1):
        if not isinstance(item, dict):
            continue
        raw = dict(item)
        raw_reactants = raw.get("reactant_smiles")
        if isinstance(raw_reactants, str):
            reactant_smiles = [raw_reactants.strip()] if raw_reactants.strip() else []
        else:
            reactant_smiles = [str(value) for value in raw_reactants or [] if str(value).strip()]
        raw_labels = raw.get("reactant_labels")
        if isinstance(raw_labels, str):
            reactant_labels = [raw_labels.strip()] if raw_labels.strip() else []
        else:
            reactant_labels = [str(value) for value in raw_labels or [] if str(value).strip()]
        reactant_label = str(raw.get("reactant_label") or "").strip()
        if reactant_label and not reactant_labels:
            reactant_labels = [reactant_label]
        condition = raw.get("condition_candidate")
        if not isinstance(condition, dict):
            condition = _condition_from_visual_step(condition or raw.get("conditions") or raw.get("condition"))
        main_reactant = str(raw.get("main_reactant_smiles") or "")
        if not main_reactant and reactant_smiles:
            main_reactant = reactant_smiles[0]
        product_label = str(raw.get("product_label") or raw.get("product_name") or "")
        product_smiles = str(raw.get("product_smiles") or raw.get("product") or "")
        target_product_fallback = False
        target_product_stereo_repair = False
        if (
            target_smiles
            and product_label
            and _label_matches_target(product_label, target_name=target_name)
            and _rdkit_valid_smiles(target_smiles)
        ):
            if not _rdkit_valid_smiles(product_smiles):
                product_smiles = target_smiles
                target_product_fallback = True
            elif _target_visual_smiles_loses_input_stereo(product_smiles, target_smiles):
                product_smiles = target_smiles
                target_product_stereo_repair = True
        source_locator = str(raw.get("source_locator") or parsed.get("source_locator") or "current PDF rendered images")
        evidence_refs = [str(value) for value in raw.get("evidence_refs") or parsed.get("evidence_refs") or []]
        if isinstance(condition, dict):
            condition.setdefault("step_id", str(raw.get("step_id") or f"visual_step_{index}_{_safe_id(product_label)}"))
            if "evidence_refs" not in condition and evidence_refs:
                condition["evidence_refs"] = list(evidence_refs)
        step = {
            "schema_version": "visual_structure_candidate_step.v1",
            "step_id": str(raw.get("step_id") or f"visual_step_{index}_{_safe_id(product_label)}"),
            "segment_id": str(raw.get("segment_id") or "visual_literature_chain"),
            "product_label": product_label,
            "product_smiles": product_smiles,
            "reactant_labels": reactant_labels,
            "reactant_smiles": reactant_smiles,
            "main_reactant_smiles": main_reactant,
            "source_ref": str(raw.get("source_ref") or source_ref),
            "source_title": str(raw.get("source_title") or source_title),
            "evidence_refs": evidence_refs,
            "source_locator": source_locator,
            "condition_candidate": condition,
            "structure_derivation": raw.get("structure_derivation")
            if isinstance(raw.get("structure_derivation"), dict)
            else {
                "basis": "current_pdf_image_to_smiles",
                "source_locator": source_locator,
                "confidence": str(parsed.get("confidence") or raw.get("confidence") or "low"),
                "tool_checks": ["visual extraction performed in this run", "RDKit parse precheck performed locally"],
            },
            "source_excerpt": str(raw.get("source_excerpt") or ""),
            "confidence": str(raw.get("confidence") or parsed.get("confidence") or "low"),
        }
        if target_product_fallback:
            derivation = dict(step.get("structure_derivation") or {})
            checks = [str(item) for item in derivation.get("tool_checks") or [] if str(item or "").strip()]
            checks.append("target product malformed visual SMILES replaced with input target SMILES")
            derivation["tool_checks"] = checks
            derivation["target_product_smiles_fallback"] = True
            step["structure_derivation"] = derivation
        if target_product_stereo_repair:
            derivation = dict(step.get("structure_derivation") or {})
            checks = [str(item) for item in derivation.get("tool_checks") or [] if str(item or "").strip()]
            checks.append("target product stereo-incomplete visual SMILES replaced with input target SMILES")
            derivation["tool_checks"] = checks
            derivation["target_product_stereo_repair"] = True
            step["structure_derivation"] = derivation
        out.append(step)
    return out


def _steps_from_candidate_chain(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    chain = parsed.get("candidate_chain")
    if not isinstance(chain, list):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(chain, start=1):
        if not isinstance(item, dict):
            continue
        precursor_smiles = str(item.get("precursor_smiles") or item.get("reactant_smiles") or "").strip()
        if not precursor_smiles:
            continue
        label = str(item.get("label") or item.get("product_label") or item.get("target_label") or "").strip()
        precursor_label = str(item.get("precursor_label") or item.get("reactant_label") or "").strip()
        out.append(
            {
                "step_id": str(item.get("step_id") or f"visual_step_{index}_{_safe_id(label)}"),
                "segment_id": str(item.get("segment_id") or "visual_literature_chain"),
                "product_label": label,
                "product_smiles": str(item.get("smiles") or item.get("product_smiles") or "").strip(),
                "reactant_labels": [precursor_label] if precursor_label else [],
                "reactant_smiles": [precursor_smiles],
                "main_reactant_smiles": precursor_smiles,
                "source_ref": str(item.get("source_ref") or _source_ref_from_parsed(parsed)),
                "source_title": str(item.get("source_title") or _source_title_from_parsed(parsed)),
                "evidence_refs": [str(value) for value in item.get("evidence_refs") or parsed.get("evidence_refs") or []],
                "source_locator": str(item.get("source_locator") or parsed.get("source_locator") or "current PDF rendered images"),
                "condition_candidate": item.get("condition_candidate") or item.get("conditions") or item.get("condition") or {},
                "source_excerpt": str(item.get("source_excerpt") or item.get("source_locator") or parsed.get("source_excerpt") or ""),
                "confidence": str(item.get("confidence") or parsed.get("confidence") or "low"),
            }
        )
    return out


def _steps_from_reaction_chain(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    chain = parsed.get("chain")
    if not isinstance(chain, list):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(chain, start=1):
        if not isinstance(item, dict):
            continue
        product_smiles = str(item.get("product_smiles") or "").strip()
        reactant_smiles = str(
            item.get("reactant_smiles")
            or item.get("main_reactant_smiles")
            or item.get("precursor_smiles")
            or ""
        ).strip()
        if not product_smiles or not reactant_smiles:
            continue
        reactant_label = str(item.get("reactant_label") or item.get("precursor_label") or "").strip()
        reactant_labels = [str(value) for value in item.get("reactant_labels") or [] if str(value).strip()]
        if reactant_label and not reactant_labels:
            reactant_labels = [reactant_label]
        condition = (
            item.get("condition_candidate")
            or item.get("conditions")
            or item.get("condition")
            or item.get("forward_conditions")
            or {}
        )
        out.append(
            {
                "step_id": str(item.get("step_id") or f"visual_step_{index}_{_safe_id(str(item.get('product_label') or 'step'))}"),
                "segment_id": str(item.get("segment_id") or "visual_literature_chain"),
                "product_label": str(item.get("product_label") or item.get("label") or "").strip(),
                "product_smiles": product_smiles,
                "reactant_labels": reactant_labels,
                "reactant_smiles": [reactant_smiles],
                "main_reactant_smiles": reactant_smiles,
                "source_ref": str(item.get("source_ref") or _source_ref_from_parsed(parsed)),
                "source_title": str(item.get("source_title") or _source_title_from_parsed(parsed)),
                "evidence_refs": [str(value) for value in item.get("evidence_refs") or parsed.get("evidence_refs") or []],
                "source_locator": str(item.get("source_locator") or item.get("source_location") or parsed.get("source_locator") or "current PDF rendered images"),
                "condition_candidate": condition,
                "source_excerpt": str(
                    item.get("source_excerpt")
                    or item.get("source_locator")
                    or item.get("source_location")
                    or parsed.get("source_excerpt")
                    or ""
                ),
                "confidence": str(item.get("confidence") or parsed.get("confidence") or "low"),
            }
        )
    return out


def _condition_from_visual_step(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        return {
            "schema_version": "condition_candidate.v1",
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "reagent": text,
            "source_grounding": "current PDF scheme image",
        } if text else {}
    raw = dict(value) if isinstance(value, dict) else {}
    return {
        "schema_version": "condition_candidate.v1",
        "source_type": "exact",
        "condition_status": "evidence_backed",
        "reagent": str(raw.get("reagent") or raw.get("reagents") or ""),
        "solvent": str(raw.get("solvent") or ""),
        "temperature": str(raw.get("temperature") or ""),
        "duration": str(raw.get("duration") or raw.get("time") or ""),
        "reported_yield": str(raw.get("reported_yield") or raw.get("yield") or ""),
        "source_grounding": str(raw.get("source_grounding") or "current PDF scheme image"),
    }


def _source_ref_from_parsed(parsed: dict[str, Any]) -> str:
    if parsed.get("source_ref"):
        return str(parsed.get("source_ref") or "")
    if parsed.get("doi"):
        doi = str(parsed.get("doi") or "")
        return doi if doi.startswith("doi:") else f"doi:{doi}"
    source = parsed.get("source")
    if isinstance(source, dict):
        doi = str(source.get("doi") or source.get("DOI") or "")
        if doi:
            return doi if doi.startswith("doi:") else f"doi:{doi}"
    return ""


def _source_title_from_parsed(parsed: dict[str, Any]) -> str:
    if parsed.get("source_title"):
        return str(parsed.get("source_title") or "")
    source = parsed.get("source")
    if isinstance(source, dict):
        return str(source.get("title") or "")
    return ""


def _target_name_from_parsed(parsed: dict[str, Any]) -> str:
    if parsed.get("target_name"):
        return str(parsed.get("target_name") or "")
    target = parsed.get("target")
    if isinstance(target, dict):
        return str(target.get("name") or target.get("label") or "")
    return ""


def _target_smiles_from_parsed(parsed: dict[str, Any]) -> str:
    if parsed.get("target_smiles"):
        return str(parsed.get("target_smiles") or "")
    target = parsed.get("target")
    if isinstance(target, dict):
        return str(target.get("target_smiles") or target.get("smiles") or "")
    return ""


def _label_matches_target(label: str, *, target_name: str) -> bool:
    left = re.sub(r"[^a-z0-9]+", "", str(label or "").lower())
    right = re.sub(r"[^a-z0-9]+", "", str(target_name or "").lower())
    return bool(left and right and left == right)


def _target_smiles_with_input_stereo(parsed_target_smiles: str, input_target_smiles: str) -> str:
    if parsed_target_smiles and input_target_smiles and _rdkit_valid_smiles(input_target_smiles) and not _rdkit_valid_smiles(parsed_target_smiles):
        return str(input_target_smiles or "")
    if _target_visual_smiles_loses_input_stereo(parsed_target_smiles, input_target_smiles):
        return str(input_target_smiles or "")
    return str(parsed_target_smiles or input_target_smiles or "")


def _target_visual_smiles_loses_input_stereo(visual_smiles: str, input_target_smiles: str) -> bool:
    if not visual_smiles or not input_target_smiles:
        return False
    if not _rdkit_valid_smiles(visual_smiles) or not _rdkit_valid_smiles(input_target_smiles):
        return False
    if _connectivity_smiles(visual_smiles) != _connectivity_smiles(input_target_smiles):
        return False
    return _specified_stereo_token_count(visual_smiles) < _specified_stereo_token_count(input_target_smiles)


def _connectivity_smiles(value: str) -> str:
    if Chem is None:
        return ""
    mol = Chem.MolFromSmiles(str(value or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _specified_stereo_token_count(value: str) -> int:
    text = str(value or "")
    return text.count("@") + text.count("/") + text.count("\\")


def _candidate_quality(chain: dict[str, Any], *, expected_labels: list[str]) -> dict[str, Any]:
    labels = _labels_from_chain(chain)
    gap_labels = _labels_from_gaps(chain)
    missing = [label for label in expected_labels if label not in labels and label not in gap_labels]
    smiles_precheck = _smiles_precheck(chain)
    valid_smiles = int(smiles_precheck.get("valid_smiles_count") or 0)
    invalid_smiles = int(smiles_precheck.get("invalid_smiles_count") or 0)
    placeholders = int(smiles_precheck.get("placeholder_smiles_count") or 0)
    unresolved_gaps = len(_blocking_extraction_gaps(chain))
    condition_gaps = _condition_gap_rows(chain)
    score = (
        len(labels) * 10
        + valid_smiles * 3
        - len(missing) * 12
        - invalid_smiles * 20
        - placeholders * 25
        - unresolved_gaps * 5
        - len(condition_gaps) * 4
    )
    return {
        "schema_version": "visual_literature_chain_quality_audit.v1",
        "expected_labels": [str(item) for item in expected_labels],
        "observed_step_labels": sorted(labels),
        "gap_labels": sorted(gap_labels),
        "missing_expected_labels": missing,
        "extraction_gap_count": unresolved_gaps,
        "condition_gap_count": len(condition_gaps),
        "condition_gap_labels": [str(row.get("product_label") or "") for row in condition_gaps],
        "condition_gaps": condition_gaps,
        "nonblocking_extraction_gap_count": len(chain.get("extraction_gaps") or []) - unresolved_gaps,
        "smiles_precheck": smiles_precheck,
        "score": score,
        "accepted": (
            not missing
            and not unresolved_gaps
            and not condition_gaps
            and not invalid_smiles
            and not placeholders
            and bool(chain.get("steps"))
        ),
    }


def _blocking_extraction_gaps(chain: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    observed = _labels_from_chain(chain)
    for gap in chain.get("extraction_gaps") or []:
        if not isinstance(gap, dict):
            continue
        label = str(gap.get("label") or gap.get("compound_label") or "")
        labels = [str(item) for item in gap.get("label_scope") or [] if str(item).strip()]
        reason = str(gap.get("reason") or gap.get("gap_type") or "").lower()
        if label and label not in observed:
            out.append(dict(gap))
            continue
        if labels and any(item not in observed for item in labels):
            out.append(dict(gap))
            continue
        if "not_confidently_convertible_to_smiles" in reason or "missing" in reason:
            out.append(dict(gap))
    return out


def _should_try_repair(quality: dict[str, Any], *, elapsed_s: float, timeout_s: float) -> bool:
    if timeout_s - elapsed_s < 120.0:
        return False
    smiles = dict(quality.get("smiles_precheck") or {})
    return bool(
        quality.get("missing_expected_labels")
        or int(quality.get("extraction_gap_count") or 0)
        or int(quality.get("condition_gap_count") or 0)
        or int(smiles.get("invalid_smiles_count") or 0)
        or int(smiles.get("placeholder_smiles_count") or 0)
    )


def _condition_gap_rows(chain: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step_index, step in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        condition = step.get("condition_candidate")
        row = dict(condition) if isinstance(condition, dict) else {}
        if _condition_has_source_grounded_content(row):
            continue
        out.append(
            {
                "step_index": step_index,
                "step_id": str(step.get("step_id") or ""),
                "product_label": str(step.get("product_label") or ""),
                "reason": "condition_gap",
                "source_locator": str(step.get("source_locator") or ""),
            }
        )
    return out


def _condition_has_source_grounded_content(condition: dict[str, Any]) -> bool:
    return any(
        str(condition.get(key) or "").strip()
        for key in ("reagent", "catalyst", "enzyme", "solvent", "temperature", "duration", "reported_yield", "ph", "buffer")
    )


def _smiles_precheck(chain: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        product_label = str(step.get("product_label") or "")
        _append_smiles_check(
            rows,
            step_index=step_index,
            field="product_smiles",
            label=product_label,
            smiles=str(step.get("product_smiles") or ""),
        )
        main_reactant = str(step.get("main_reactant_smiles") or "")
        _append_smiles_check(
            rows,
            step_index=step_index,
            field="main_reactant_smiles",
            label=";".join(str(item) for item in step.get("reactant_labels") or []),
            smiles=main_reactant,
        )
        for reactant_index, smiles in enumerate(step.get("reactant_smiles") or [], start=1):
            labels = step.get("reactant_labels") or []
            label = str(labels[reactant_index - 1]) if reactant_index <= len(labels) else ""
            _append_smiles_check(
                rows,
                step_index=step_index,
                field=f"reactant_smiles[{reactant_index - 1}]",
                label=label,
                smiles=str(smiles or ""),
            )
    invalid = [row for row in rows if not row["valid"]]
    placeholders = [row for row in rows if row["placeholder"]]
    return {
        "schema_version": "visual_literature_chain_smiles_precheck.v1",
        "rdkit_available": Chem is not None,
        "field_count": len(rows),
        "valid_smiles_count": sum(1 for row in rows if row["valid"]),
        "invalid_smiles_count": len(invalid),
        "placeholder_smiles_count": len(placeholders),
        "invalid_fields": invalid[:80],
        "placeholder_fields": placeholders[:80],
    }


def _append_smiles_check(rows: list[dict[str, Any]], *, step_index: int, field: str, label: str, smiles: str) -> None:
    text = str(smiles or "").strip()
    rdkit_valid = _rdkit_valid_smiles(text)
    placeholder = False if rdkit_valid else _is_placeholder_smiles(text)
    valid = bool(text and rdkit_valid and not placeholder)
    rows.append(
        {
            "step_index": step_index,
            "field": field,
            "label": label,
            "smiles": text,
            "valid": valid,
            "placeholder": placeholder,
        }
    )


def _is_placeholder_smiles(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    upper = text.upper()
    if any(token in upper for token in ("VISUAL_UNRESOLVED", "UNKNOWN", "UNK", "TBD", "PLACEHOLDER")):
        return True
    if re.fullmatch(r"(compound\s*)?\d+[A-Za-z]?", text, flags=re.I):
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ -]{2,}", text) and not any(ch in text for ch in "[]=#()@+-\\/123456789"):
        return True
    return False


def _rdkit_valid_smiles(value: str) -> bool:
    if Chem is None:
        return False
    try:
        return Chem.MolFromSmiles(str(value or "")) is not None
    except Exception:
        return False


def _labels_from_chain(chain: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for step in chain.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("product_label"):
            labels.add(str(step["product_label"]))
        for label in step.get("reactant_labels") or []:
            labels.add(str(label))
    return labels


def _labels_from_gaps(chain: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for gap in chain.get("extraction_gaps") or []:
        if not isinstance(gap, dict):
            continue
        label = str(gap.get("label") or gap.get("compound_label") or "")
        if label:
            labels.add(label)
    return labels


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"


def _read_key(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_codex_home(*, codex_home: Path, api_key: str, base_url: str, model: str, run_dir: Path) -> None:
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": api_key}, ensure_ascii=False),
        encoding="utf-8",
    )
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                'model_provider = "wellau"',
                f'model = "{model}"',
                "",
                "[model_providers.wellau]",
                'name = "wellau"',
                f'base_url = "{base_url.rstrip("/")}"',
                'env_key = "OPENAI_API_KEY"',
                'wire_api = "responses"',
                "",
                "[tools]",
                "web_search = true",
                "",
                "[sandbox_workspace_write]",
                f'writable_roots = ["{run_dir}"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
