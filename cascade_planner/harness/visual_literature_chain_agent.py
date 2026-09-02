"""Vision-agent extraction of literature structure chains from local PDF crops."""
from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cascade_planner.harness.source_capabilities import meaningful_compound_labels

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
    allow_repair: bool = True,
    ambient_auth: bool | None = None,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    expected_labels = meaningful_compound_labels(expected_labels or [])
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

    api_key = _read_key(Path(key_path))
    executable = codex_executable or shutil.which("codex")
    use_ambient_auth = (
        _visual_use_ambient_codex_cli_auth()
        if ambient_auth is None
        else bool(ambient_auth)
    )
    if (
        (use_ambient_auth and not executable)
        or (
            not use_ambient_auth
            and (
                not api_key
                or (not _visual_direct_api_enabled() and not executable)
            )
        )
    ):
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
        expected_labels=expected_labels,
        route_sequence_hint=route_sequence_hint,
        text_snippets=text_snippets or [],
    )
    first_attempt = _run_visual_json_prompt(
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
        ambient_auth=use_ambient_auth,
        reasoning_effort=reasoning_effort,
    )
    if first_attempt["status"] != "completed" and not str(
        first_attempt.get("raw_last_message") or ""
    ).strip():
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
        result["usage"] = _aggregate_visual_usage([first_attempt])
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
    candidate_chain = _salvage_valid_visual_subchain(candidate_chain)
    attempts = [first_attempt]
    selected_attempt = first_attempt
    candidate_quality = _candidate_quality(candidate_chain, expected_labels=expected_labels)
    if allow_repair and _should_try_repair(candidate_quality, elapsed_s=time.monotonic() - started, timeout_s=float(timeout_s)):
        repair_prompt = _repair_prompt(
            target_name=target_name,
            target_smiles=target_smiles,
            source_ref=source_ref,
            source_title=source_title,
            expected_labels=expected_labels,
            route_sequence_hint=route_sequence_hint,
            first_parsed=parsed,
            first_quality=candidate_quality,
        )
        repair_attempt = _run_visual_json_prompt(
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
            ambient_auth=use_ambient_auth,
            reasoning_effort=reasoning_effort,
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
        repair_chain = _salvage_valid_visual_subchain(repair_chain)
        repair_quality = _candidate_quality(repair_chain, expected_labels=expected_labels)
        if _should_select_repair_candidate(
            current_chain=candidate_chain,
            current_quality=candidate_quality,
            repair_chain=repair_chain,
            repair_quality=repair_quality,
        ):
            raw_text = str(repair_attempt.get("raw_last_message") or "")
            parsed = repair_parsed
            candidate_chain = repair_chain
            candidate_quality = repair_quality
            selected_attempt = repair_attempt

    candidate_path = out / "visual_structure_candidate_chain.json"
    if candidate_chain:
        candidate_path.write_text(json.dumps(candidate_chain, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    reasons: list[str] = []
    if int(selected_attempt.get("returncode") or 0) != 0:
        reasons.extend(str(item) for item in selected_attempt.get("reasons") or ["visual_literature_chain_attempt_failed"])
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
    if int(candidate_quality.get("structure_gap_count") or 0):
        reasons.append("visual_literature_chain_structure_gaps")
    accepted_for_exploration = bool(candidate_quality.get("accepted"))
    exact_ready = bool(candidate_quality.get("exact_ready"))

    result = _base_result(
        output_dir=out,
        accepted=accepted_for_exploration,
        reasons=sorted(set(reasons)),
        target_name=target_name,
        target_smiles=target_smiles,
        source_ref=source_ref,
        source_title=source_title,
        image_paths=existing_images,
    )
    selected_attempt_status = str(selected_attempt.get("status") or "")
    selected_attempt_returncode = int(selected_attempt.get("returncode") or 0)
    result_status = "completed" if selected_attempt_returncode == 0 else "failed"
    if accepted_for_exploration and selected_attempt_status == "error" and raw_text.strip():
        result_status = "completed_with_attempt_cleanup_warning"
    result.update(
        {
            "status": result_status,
            "elapsed_s": round(time.monotonic() - started, 3),
            "candidate_chain_path": str(candidate_path) if candidate_chain else "",
            "candidate_chain": candidate_chain,
            "candidate_step_count": len(candidate_chain.get("steps") or []),
            "raw_last_message": raw_text[:8000],
            "parsed_output": parsed,
            "event_log_path": str(selected_attempt.get("event_log_path") or ""),
            "stderr_log_path": str(selected_attempt.get("stderr_log_path") or ""),
            "attempts": attempts,
            "usage": _aggregate_visual_usage(attempts),
            "selected_attempt_index": attempts.index(selected_attempt),
            "candidate_quality": candidate_quality,
            "acceptance_level": str(candidate_quality.get("acceptance_level") or ("exact_source_detail_candidate" if exact_ready else "exploratory_connectivity_candidate" if accepted_for_exploration else "rejected")),
            "exact_ready": exact_ready,
            "exploratory_accepted": bool(candidate_quality.get("exploratory_accepted")),
            "missing_expected_labels": candidate_quality.get("missing_expected_labels") or [],
            "condition_gap_labels": candidate_quality.get("condition_gap_labels") or [],
            "structure_gaps": candidate_quality.get("structure_gaps") or [],
            "smiles_precheck": smiles_precheck,
            "extraction_policy": _visual_extraction_policy(),
        }
    )
    _write_result(out, result)
    return result


def _visual_extraction_policy() -> dict[str, Any]:
    return {
        "pdf_reuse_allowed": True,
        "tool_execution_allowed": False,
        "network_browsing_allowed": False,
        "direct_visual_api_preferred": True,
        "codex_subprocess_fallback_default": False,
        "prior_candidate_chain_reuse_allowed": False,
        "prior_source_detail_records_reuse_allowed": False,
        "must_derive_from_current_images": True,
        "current_pdf_label_anchor_fallback_allowed": False,
        "label_anchor_fallback_scope": "",
        "placeholder_smiles_allowed": False,
        "rdkit_valid_smiles_required": True,
        "achiral_or_connectivity_only_smiles_allowed_for_exploration": True,
        "exploratory_candidates_are_not_exact_literature_rows": True,
        "approximate_visual_candidates_cannot_satisfy_parent_proof": True,
        "no_solved_claim": True,
        "production_write_blocked": True,
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
        "Do not call shell commands, tools, Python, RDKit, file listing, web search, or any external validator. "
        "Inspect the attached images directly and return the JSON immediately; the host will validate every SMILES after your response.\n"
        "Use only the attached images and the text snippets below. Do not use prior AutoPlanner JSON, old curator records, or memory.\n"
        "This is not a solved-route verdict. You are only producing a candidate structure chain for later RDKit/source-detail validation.\n\n"
        "Critical chemistry-output rules:\n"
        "- Every product_smiles, reactant_smiles item, and main_reactant_smiles must be a real RDKit-parseable SMILES string.\n"
        "- Do not output placeholders, labels, compound names, Markush abbreviations, VISUAL_UNRESOLVED, UNKNOWN, UNK, TBD, or empty strings in any *_smiles field.\n"
        "- If exact stereochemistry is not legible but atom connectivity/protecting groups are visible, output a connectivity-only or achiral SMILES instead of omitting the structure. "
        "In that case set confidence low, set structure_derivation.basis to current_pdf_image_to_achiral_or_approximate_smiles, set stereochemistry_status to unspecified_or_partial, "
        "set not_exact_literature_segment true, and add risk_flags including stereochemistry_unspecified and exploratory_visual_candidate.\n"
        "- For polycyclic steroid/terpenoid schemes, do not reject a visible molecule merely because many stereocenters are unclear. Use an achiral connectivity SMILES that preserves the visible ring system, carbonyls, alcohols, alkenes, halides, and protecting groups as far as visible.\n"
        "- If an arrow connects two visible molecules and both have legible connectivity at the skeleton/functional-group level, return at least one exploratory step with achiral/connectivity-only SMILES rather than an empty steps array.\n"
        "- Omit a step only when atom connectivity or protecting-group identity is not visible even at connectivity-only level; then add an extraction_gaps row.\n"
        "- For every included transformation, extract visible/source-grounded conditions into condition_candidate only. "
        "Do not emit parallel condition aliases such as condition_text, reaction_conditions, visible_conditions, conditions, condition, or forward_conditions in new outputs. "
        "condition_candidate may contain reagent, catalyst, base, oxidant, solvent, temperature, duration, reported_yield, condition_text_transcribed, and source_grounding. "
        "If the structure is visible but the condition is not readable, keep the step only if the structure is valid and add an extraction_gaps row with gap_type condition_gap for that product label.\n"
        "- Prefer useful RDKit-valid exploratory connectivity over an empty chain. Never invent atoms, protecting groups, or labels that are not visible/source-grounded.\n"
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
        '      "stereochemistry_status": "specified|unspecified_or_partial",\n'
        '      "not_exact_literature_segment": false,\n'
        '      "allowed_use": "exact_candidate|exploratory_template_and_guided_hint_only",\n'
        '      "risk_flags": ["..."],\n'
        '      "source_excerpt": "short source-grounded statement",\n'
        '      "confidence": "low|medium|medium_high|high"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "The steps must be ordered retro_target_to_start when a target product is specified; otherwise use the clearest source order and state it in route_order. "
        "If a full continuous chain is not visible, return the visible valid partial steps only. If stereochemistry is uncertain, return achiral/connectivity-only SMILES with the exploratory flags above. "
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
        "Do not call shell commands, tools, Python, RDKit, file listing, web search, or any external validator. "
        "Return the repaired JSON directly; the host performs validation.\n"
        "The previous draft read some labels/conditions but failed the audit below. Use that draft only as a checklist of visible labels/conditions; "
        "derive all structures again from the attached current images. Do not use prior AutoPlanner route JSON, old curator records, or memory.\n\n"
        "Hard requirements:\n"
        "1. Every *_smiles field must be a real RDKit-parseable SMILES string.\n"
        "2. Do not output placeholders such as VISUAL_UNRESOLVED, UNKNOWN, label-only strings, Markush abbreviations, or empty strings.\n"
        "3. If stereochemistry is uncertain but atom connectivity/protecting groups are visible, output an achiral/connectivity-only SMILES. Mark it with confidence low, "
        "stereochemistry_status unspecified_or_partial, not_exact_literature_segment true, allowed_use exploratory_template_and_guided_hint_only, and risk_flags including stereochemistry_unspecified.\n"
        "4. For polycyclic steroid/terpenoid schemes, a connectivity-only SMILES is preferred over no structure when the fused skeleton and key functional groups are visible.\n"
        "5. Omit a step only when atom connectivity or protecting groups are not legible even at connectivity-only level; list omitted labels under extraction_gaps.\n"
        "6. For each included step, read the reaction condition text from the current image when visible and put it only under condition_candidate. "
        "Do not emit condition_text, reaction_conditions, visible_conditions, conditions, condition, or forward_conditions as sibling fields. "
        "At least one of reagent, catalyst, solvent, temperature, duration, reported_yield, or condition_text_transcribed should be filled from the source; otherwise add a condition_gap extraction_gaps row for that product label.\n"
        "7. Try to cover all expected visible labels and the provided sequence hint, but only with valid SMILES and source-grounded condition fields.\n"
        "8. Return one JSON object only with schema_version visual_structure_candidate_chain.v1 and route_order retro_target_to_start.\n\n"
        f"Target: {target_name} {target_smiles}\n"
        f"Source: {source_ref} {source_title}\n"
        f"Expected labels: {json.dumps(expected_labels, ensure_ascii=False)}\n"
        f"Route/sequence hint: {route_sequence_hint or 'infer from attached images and text snippets'}\n"
        f"Failed audit: {json.dumps(first_quality, ensure_ascii=False)[:6000]}\n"
        f"Visible label/condition checklist from first draft: {json.dumps(brief_steps, ensure_ascii=False)[:6000]}\n"
    )


def _run_visual_json_prompt(
    *,
    executable: str | None,
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
    ambient_auth: bool = False,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    if _visual_direct_api_enabled() and not ambient_auth:
        direct_attempt = _run_direct_visual_prompt(
            api_key=api_key,
            base_url=base_url,
            model=model,
            output_dir=output_dir,
            image_paths=image_paths,
            prompt=prompt,
            timeout_s=timeout_s,
            prompt_filename=prompt_filename,
            event_log_filename=event_log_filename,
            stderr_log_filename=stderr_log_filename,
            last_message_filename=last_message_filename,
        )
        if direct_attempt.get("status") == "completed" or not _visual_codex_fallback_enabled() or not executable:
            return direct_attempt
    if not executable:
        return {
            "schema_version": "visual_literature_chain_attempt.v1",
            "status": "error",
            "reasons": ["codex_executable_missing"],
            "elapsed_s": 0.0,
            "prompt_path": str((output_dir / prompt_filename).resolve()),
            "event_log_path": str((output_dir / event_log_filename).resolve()),
            "stderr_log_path": str((output_dir / stderr_log_filename).resolve()),
            "last_message_path": str((output_dir / last_message_filename).resolve()),
            "returncode": -1,
            "raw_last_message": "",
        }
    return _run_codex_visual_prompt(
        executable=executable,
        api_key=api_key,
        base_url=base_url,
        model=model,
        output_dir=output_dir,
        image_paths=image_paths,
        prompt=prompt,
        timeout_s=timeout_s,
        prompt_filename=prompt_filename,
        event_log_filename=event_log_filename,
        stderr_log_filename=stderr_log_filename,
        last_message_filename=last_message_filename,
        ambient_auth=ambient_auth,
        reasoning_effort=reasoning_effort,
    )


def _run_direct_visual_prompt(
    *,
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
    last_message.unlink(missing_ok=True)
    stderr_log.write_text("", encoding="utf-8")
    started = time.monotonic()
    errors: list[dict[str, Any]] = []
    requested_timeout_s = max(1.0, float(timeout_s or 0.0))
    for endpoint in _visual_api_endpoint_order(base_url):
        elapsed_before_endpoint = time.monotonic() - started
        endpoint_timeout_s = max(30.0, requested_timeout_s - elapsed_before_endpoint)
        payload = _visual_api_payload(model=model, endpoint=endpoint, prompt=prompt, image_paths=image_paths)
        try:
            response = _post_visual_api_json(
                api_key=api_key,
                base_url=base_url,
                endpoint=endpoint,
                payload=payload,
                timeout_s=endpoint_timeout_s,
            )
            raw_text = _extract_visual_api_text(response, endpoint=endpoint)
            last_message.write_text(raw_text, encoding="utf-8")
            event_log.write_text(
                json.dumps(
                    {
                        "schema_version": "visual_direct_api_event.v1",
                        "status": "completed",
                        "endpoint": endpoint,
                        "requested_timeout_s": round(requested_timeout_s, 3),
                        "endpoint_timeout_s": round(endpoint_timeout_s, 3),
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "response_keys": sorted(str(key) for key in response.keys()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "schema_version": "visual_literature_chain_attempt.v1",
                "status": "completed",
                "reasons": [],
                "elapsed_s": round(time.monotonic() - started, 3),
                "prompt_path": str(prompt_path),
                "event_log_path": str(event_log),
                "stderr_log_path": str(stderr_log),
                "last_message_path": str(last_message),
                "returncode": 0,
                "raw_last_message": raw_text,
                "execution_mode": "direct_visual_api",
                "api_endpoint": endpoint,
                "usage": _normalized_visual_usage(response.get("usage"), invoked=True),
            }
        except socket.timeout as exc:
            errors.append(
                {
                    "endpoint": endpoint,
                    "error_type": "timeout",
                    "message": str(exc),
                    "requested_timeout_s": round(requested_timeout_s, 3),
                    "endpoint_timeout_s": round(endpoint_timeout_s, 3),
                }
            )
            continue
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            errors.append({"endpoint": endpoint, "error_type": "http", "status": int(exc.code), "message": body})
            if int(exc.code) not in {400, 404, 405, 415, 422}:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            errors.append(
                {
                    "endpoint": endpoint,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "requested_timeout_s": round(requested_timeout_s, 3),
                    "endpoint_timeout_s": round(endpoint_timeout_s, 3),
                }
            )
            if isinstance(exc, TimeoutError):
                continue
    elapsed = round(time.monotonic() - started, 3)
    event_log.write_text(
        json.dumps(
            {
                "schema_version": "visual_direct_api_event.v1",
                "status": "failed",
                "elapsed_s": elapsed,
                "requested_timeout_s": round(requested_timeout_s, 3),
                "errors": errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_log.write_text(json.dumps(errors, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": "visual_literature_chain_attempt.v1",
        "status": "error",
        "reasons": ["visual_direct_api_failed"],
        "elapsed_s": elapsed,
        "prompt_path": str(prompt_path),
        "event_log_path": str(event_log),
        "stderr_log_path": str(stderr_log),
        "last_message_path": str(last_message),
        "returncode": -1,
        "raw_last_message": "",
        "execution_mode": "direct_visual_api",
        "api_errors": errors,
    }


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
    ambient_auth: bool = False,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    prompt_path = output_dir / prompt_filename
    prompt_path.write_text(prompt, encoding="utf-8")
    event_log = output_dir / event_log_filename
    stderr_log = output_dir / stderr_log_filename
    last_message = output_dir / last_message_filename
    executable_path = Path(executable)
    executable_command = [sys.executable, str(executable_path)] if executable_path.suffix.lower() == ".py" else [executable]
    command = [
        *executable_command,
        "--ask-for-approval",
        "never",
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--disable",
        "multi_agent",
        "--disable",
        "shell_tool",
        "--disable",
        "code_mode_host",
        "--disable",
        "browser_use",
        "--disable",
        "computer_use",
        "--disable",
        "in_app_browser",
        "exec",
        "--ignore-rules",
        "--json",
        "--cd",
        str(output_dir),
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--ephemeral",
        "--output-last-message",
        str(last_message),
        "--model",
        str(model),
    ]
    if reasoning_effort in {"low", "medium", "high"}:
        command.extend(
            ["-c", f"model_reasoning_effort={_toml_string(reasoning_effort)}"]
        )
    for image in image_paths:
        command.extend(["--image", str(image)])
    command.append("-")
    started = time.monotonic()
    try:
        auth_context = (
            nullcontext(None)
            if ambient_auth
            else tempfile.TemporaryDirectory(
                prefix="autoplanner_visual_chain_"
            )
        )
        with auth_context as tmp:
            env = os.environ.copy()
            if ambient_auth:
                # Reuse the operator's authenticated Codex CLI session.  API
                # key/base URL variables are removed so a stale or depleted
                # third-party key cannot shadow the ambient login.
                env.pop("OPENAI_API_KEY", None)
                env.pop("OPENAI_BASE_URL", None)
                env.pop("CODEX_HOME", None)
            else:
                codex_home = Path(str(tmp)) / "codex_home"
                codex_home.mkdir(parents=True, exist_ok=True)
                _write_codex_home(
                    codex_home=codex_home,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    run_dir=output_dir,
                )
                env["CODEX_HOME"] = str(codex_home)
                env["OPENAI_API_KEY"] = api_key
                env.pop("OPENAI_BASE_URL", None)
            # Windows inherits the active ANSI code page for text-mode pipes.
            # Literature prompts routinely contain symbols such as µ, ° and
            # stereochemical Unicode; encoding stdin with GBK used to abort a
            # valid visual extraction before Codex received it.
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            with event_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    command,
                    cwd=str(output_dir),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
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
        raw_text = last_message.read_text(encoding="utf-8", errors="replace") if last_message.exists() else ""
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
            "raw_last_message": raw_text,
            "error": str(exc),
        }
    raw_text = last_message.read_text(encoding="utf-8", errors="replace") if last_message.exists() else ""
    diagnostic_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (event_log, stderr_log)
        if path.is_file()
    )
    infrastructure_failure = _codex_visual_infrastructure_failure(
        diagnostic_text
    )
    returncode = int(getattr(proc, "returncode", 1))
    status = "completed" if returncode == 0 else "failed"
    reasons = [] if returncode == 0 else ["codex_visual_chain_nonzero_exit"]
    retryable = False
    retry_after_hint = ""
    if infrastructure_failure:
        status = "error"
        reasons = [str(infrastructure_failure["reason"])]
        retryable = True
        retry_after_hint = str(
            infrastructure_failure.get("retry_after_hint") or ""
        )
    return {
        "schema_version": "visual_literature_chain_attempt.v1",
        "status": status,
        "reasons": reasons,
        "elapsed_s": round(time.monotonic() - started, 3),
        "prompt_path": str(prompt_path),
        "event_log_path": str(event_log),
        "stderr_log_path": str(stderr_log),
        "last_message_path": str(last_message),
        "returncode": returncode,
        "raw_last_message": raw_text,
        "retryable_infrastructure_failure": retryable,
        "retry_after_hint": retry_after_hint,
        "execution_mode": (
            "ambient_codex_cli"
            if ambient_auth
            else "isolated_codex_cli_api_key"
        ),
        "usage": _codex_visual_event_usage(event_log, invoked=True),
    }


def _codex_visual_infrastructure_failure(
    diagnostic_text: str,
) -> dict[str, str]:
    text = str(diagnostic_text or "")
    lowered = text.casefold()
    if any(
        token in lowered
        for token in (
            "you've hit your usage limit",
            "you have hit your usage limit",
            "purchase more credits",
            "usage_limit_reached",
        )
    ):
        match = re.search(
            r"try again at\s+([^\"\r\n}]+)",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "reason": "codex_visual_usage_limit",
            "retry_after_hint": str(match.group(1)).strip() if match else "",
        }
    if "rate limit" in lowered or "rate_limit" in lowered:
        return {
            "reason": "codex_visual_rate_limited",
            "retry_after_hint": "",
        }
    if "selected model is at capacity" in lowered or "model is at capacity" in lowered:
        return {
            "reason": "codex_visual_model_capacity",
            "retry_after_hint": "",
        }
    return {}


def _visual_use_ambient_codex_cli_auth() -> bool:
    value = str(
        os.environ.get("AUTOPLANNER_CODEX_WORKER_AUTH") or ""
    ).strip().lower()
    return value in {
        "ambient",
        "ambient_codex",
        "ambient_codex_cli",
        "codex_login",
    }


def _visual_direct_api_enabled() -> bool:
    return _env_flag("AUTOPLANNER_VISUAL_DIRECT_API", default=True)


def _visual_codex_fallback_enabled() -> bool:
    return _env_flag("AUTOPLANNER_VISUAL_CODEX_FALLBACK", default=True)


def _codex_visual_event_usage(path: Path, *, invoked: bool) -> dict[str, Any]:
    latest: Mapping[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            latest = dict(event["usage"])
    return _normalized_visual_usage(latest, invoked=invoked)


def _normalized_visual_usage(
    value: Mapping[str, Any] | None,
    *,
    invoked: bool,
) -> dict[str, Any]:
    row = dict(value or {})
    input_details = (
        dict(row.get("input_tokens_details") or {})
        if isinstance(row.get("input_tokens_details"), Mapping)
        else {}
    )
    output_details = (
        dict(row.get("output_tokens_details") or {})
        if isinstance(row.get("output_tokens_details"), Mapping)
        else {}
    )
    input_tokens = int(
        row.get("input_tokens")
        or row.get("prompt_tokens")
        or input_details.get("total")
        or 0
    )
    output_tokens = int(
        row.get("output_tokens")
        or row.get("completion_tokens")
        or output_details.get("total")
        or 0
    )
    return {
        "model_invocations": int(bool(invoked)),
        "visual_invocations": int(bool(invoked)),
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
    }


def _aggregate_visual_usage(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        _normalized_visual_usage(
            attempt.get("usage") if isinstance(attempt.get("usage"), Mapping) else {},
            invoked=bool(dict(attempt.get("usage") or {}).get("model_invocations")),
        )
        for attempt in attempts
    ]
    return {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in (
            "model_invocations",
            "visual_invocations",
            "input_tokens",
            "output_tokens",
        )
    } | {
        "wall_time_s": round(
            sum(max(0.0, float(attempt.get("elapsed_s") or 0.0)) for attempt in attempts),
            3,
        )
    }


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _visual_api_endpoint_order(base_url: str) -> list[str]:
    host = str(base_url or "").split("//")[-1].split("/")[0].lower()
    if "wellau" in host:
        return ["chat/completions", "responses"]
    return ["responses", "chat/completions"]


def _post_visual_api_json(
    *,
    api_key: str,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{str(base_url).rstrip('/')}/{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AutoPlanner-direct-visual-json/1",
        },
        method="POST",
    )
    if _env_flag("AUTOPLANNER_VISUAL_USE_ENV_PROXY", default=False):
        opener = urllib.request.build_opener()
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=float(timeout_s)) as response:
        raw = response.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("visual_api_response_not_object")
    return data


def _visual_api_payload(*, model: str, endpoint: str, prompt: str, image_paths: list[Path]) -> dict[str, Any]:
    schema = _visual_candidate_chain_json_schema()
    if endpoint == "chat/completions":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object. You cannot browse, call tools, execute code, "
                        "or use external chemistry databases. Use only the provided images and text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {"type": "image_url", "image_url": {"url": _image_data_url(path), "detail": "high"}}
                            for path in image_paths
                        ],
                    ],
                },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "visual_structure_candidate_chain",
                    "schema": schema,
                    "strict": False,
                },
            },
        }
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Return exactly one JSON object. You cannot browse, call tools, execute code, "
                            "or use external chemistry databases.\n\n"
                            + prompt
                        ),
                    },
                    *[
                        {"type": "input_image", "image_url": _image_data_url(path), "detail": "high"}
                        for path in image_paths
                    ],
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "visual_structure_candidate_chain",
                "schema": schema,
                "strict": False,
            }
        },
    }


def _extract_visual_api_text(response: dict[str, Any], *, endpoint: str) -> str:
    if endpoint == "chat/completions":
        choices = response.get("choices") or []
        if choices:
            message = dict((choices[0] or {}).get("message") or {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict):
                        chunks.append(str(item.get("text") or item.get("content") or ""))
                return "".join(chunks)
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output") or []
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise ValueError("visual_api_missing_output_text")


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _visual_candidate_chain_json_schema() -> dict[str, Any]:
    string_value = {"type": "string"}
    condition = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "schema_version": string_value,
            "source_type": string_value,
            "condition_status": string_value,
            "reagent": string_value,
            "catalyst": string_value,
            "solvent": string_value,
            "temperature": string_value,
            "duration": string_value,
            "reported_yield": string_value,
            "source_grounding": string_value,
        },
    }
    step = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "schema_version": string_value,
            "step_id": string_value,
            "segment_id": string_value,
            "product_label": string_value,
            "product_smiles": string_value,
            "reactant_labels": {"type": "array", "items": string_value},
            "reactant_smiles": {"type": "array", "items": string_value},
            "main_reactant_smiles": string_value,
            "source_ref": string_value,
            "source_title": string_value,
            "evidence_refs": {"type": "array", "items": string_value},
            "source_locator": string_value,
            "condition_candidate": condition,
            "structure_derivation": {"type": "object", "additionalProperties": True},
            "source_excerpt": string_value,
            "confidence": string_value,
        },
    }
    gap = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "label": string_value,
            "gap_type": string_value,
            "reason": string_value,
            "source_locator": string_value,
        },
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "schema_version": {"type": "string"},
            "case_id": string_value,
            "target_name": string_value,
            "target_smiles": string_value,
            "route_order": string_value,
            "source_ref": string_value,
            "source_title": string_value,
            "evidence_refs": {"type": "array", "items": string_value},
            "source_locator": string_value,
            "confidence": string_value,
            "extraction_gaps": {"type": "array", "items": gap},
            "steps": {"type": "array", "items": step},
        },
        "required": ["schema_version", "steps"],
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
            "generation_mode": "fresh_visual_model_from_current_pdf_images",
            "prior_candidate_chain_reuse_allowed": False,
            "prior_source_detail_records_reuse_allowed": False,
            "input_images": evidence_refs,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    return chain


def _salvage_valid_visual_subchain(chain: dict[str, Any]) -> dict[str, Any]:
    if not chain:
        return {}
    precheck = _smiles_precheck(chain)
    invalid_rows = [
        dict(row)
        for row in [
            *(precheck.get("invalid_fields") or []),
            *(precheck.get("placeholder_fields") or []),
        ]
        if isinstance(row, dict)
    ]
    invalid_step_indexes: set[int] = set()
    for row in invalid_rows:
        try:
            index = int(row.get("step_index") or 0)
        except (TypeError, ValueError):
            index = 0
        if index > 0:
            invalid_step_indexes.add(index)
    if not invalid_step_indexes:
        return chain
    steps = [dict(step) for step in chain.get("steps") or [] if isinstance(step, dict)]
    kept_steps: list[dict[str, Any]] = []
    removed_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        summary = {
            "step_index": index,
            "step_id": str(step.get("step_id") or ""),
            "product_label": str(step.get("product_label") or ""),
        }
        if index in invalid_step_indexes:
            removed_steps.append(summary)
        else:
            kept_steps.append(step)
    if not kept_steps:
        return chain
    salvaged = dict(chain)
    salvaged["steps"] = kept_steps
    salvaged["visual_sanitization_audit"] = {
        "schema_version": "visual_chain_smiles_sanitization_audit.v1",
        "mode": "drop_steps_with_invalid_or_placeholder_smiles",
        "original_step_count": len(steps),
        "kept_step_count": len(kept_steps),
        "removed_steps": removed_steps,
        "invalid_fields": invalid_rows[:80],
        "no_solved_claim": True,
    }
    gaps = [dict(row) for row in chain.get("extraction_gaps") or [] if isinstance(row, dict)]
    for row in removed_steps:
        label = str(row.get("product_label") or row.get("step_id") or "").strip()
        gaps.append(
            {
                "label": label,
                "gap_type": "invalid_smiles_step_dropped",
                "reason": "visual step contained at least one invalid or placeholder SMILES and was withheld from downstream use",
                "source_locator": "visual_sanitization_audit",
            }
        )
    salvaged["extraction_gaps"] = gaps
    return salvaged


def _first_nonempty_string(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, tuple)):
            text = "; ".join(str(item).strip() for item in value if str(item or "").strip())
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalized_candidate_steps(parsed: dict[str, Any], *, target_name: str, target_smiles: str) -> list[dict[str, Any]]:
    source_ref = _source_ref_from_parsed(parsed)
    source_title = _source_title_from_parsed(parsed)
    raw_items = parsed.get("steps")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = parsed.get("route_steps")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = parsed.get("candidate_steps")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = _steps_from_candidate_chain(parsed)
    if not raw_items:
        raw_items = _steps_from_reaction_chain(parsed)
    label_smiles = _product_label_smiles_index(raw_items if isinstance(raw_items, list) else [])
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
        if not reactant_smiles:
            raw_precursors = raw.get("precursor_smiles")
            if isinstance(raw_precursors, str):
                reactant_smiles = [raw_precursors.strip()] if raw_precursors.strip() else []
            else:
                reactant_smiles = [str(value) for value in raw_precursors or [] if str(value).strip()]
        raw_reactant_objects = raw.get("reactants")
        if not reactant_smiles and isinstance(raw_reactant_objects, list):
            reactant_smiles = [
                str(value.get("smiles") or value.get("reactant_smiles") or "").strip()
                for value in raw_reactant_objects
                if isinstance(value, dict) and str(value.get("smiles") or value.get("reactant_smiles") or "").strip()
            ]
        raw_labels = raw.get("reactant_labels")
        if isinstance(raw_labels, str):
            reactant_labels = [raw_labels.strip()] if raw_labels.strip() else []
        else:
            reactant_labels = [str(value) for value in raw_labels or [] if str(value).strip()]
        if not reactant_labels:
            raw_precursor_labels = raw.get("precursor_labels")
            if isinstance(raw_precursor_labels, str):
                reactant_labels = [raw_precursor_labels.strip()] if raw_precursor_labels.strip() else []
            else:
                reactant_labels = [str(value) for value in raw_precursor_labels or [] if str(value).strip()]
        if not reactant_labels and isinstance(raw_reactant_objects, list):
            reactant_labels = [
                str(value.get("label") or value.get("name") or "").strip()
                for value in raw_reactant_objects
                if isinstance(value, dict) and str(value.get("label") or value.get("name") or "").strip()
            ]
        reactant_label = str(raw.get("reactant_label") or "").strip()
        if reactant_label and not reactant_labels:
            reactant_labels = [reactant_label]
        if not reactant_smiles and reactant_labels:
            reactant_smiles = [
                label_smiles[label]
                for label in reactant_labels
                if label in label_smiles and str(label_smiles[label]).strip()
            ]
        condition_raw = (
            raw.get("condition_candidate")
            or raw.get("visible_conditions")
            or raw.get("reaction_conditions")
            or raw.get("reaction_condition")
            or raw.get("conditions_from_source")
            or raw.get("conditions")
            or raw.get("condition")
            or raw.get("forward_conditions")
        )
        condition = _condition_from_visual_step(condition_raw)
        main_reactant = str(raw.get("main_reactant_smiles") or "")
        if not main_reactant and reactant_smiles:
            main_reactant = reactant_smiles[0]
        product_label = _first_nonempty_string(
            raw,
            (
                "product_label",
                "visible_label",
                "visible_product_label",
                "mapped_candidate_label",
                "candidate_product_label",
                "target_label",
                "label",
                "product_name",
            ),
        )
        product_smiles = str(raw.get("product_smiles") or raw.get("visible_product_smiles") or raw.get("product") or "")
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
        source_locator = _first_nonempty_string(
            raw,
            ("source_locator", "source_location", "source_scheme", "scheme", "figure", "page"),
        ) or str(parsed.get("source_locator") or "current PDF rendered images")
        evidence_refs = [str(value) for value in raw.get("evidence_refs") or parsed.get("evidence_refs") or []]
        if isinstance(condition, dict):
            condition.setdefault("step_id", str(raw.get("step_id") or f"visual_step_{index}_{_safe_id(product_label)}"))
            if "evidence_refs" not in condition and evidence_refs:
                condition["evidence_refs"] = list(evidence_refs)
            if not str(condition.get("source_grounding") or "").strip():
                condition["source_grounding"] = _first_nonempty_string(
                    raw,
                    ("source_grounding", "source_scheme", "source_locator", "source_location"),
                ) or "current PDF scheme image"
        source_excerpt = _first_nonempty_string(
            raw,
            (
                "source_excerpt",
                "source_grounding",
                "source_scheme",
                "visible_text",
                "condition_text_transcribed",
                "visual_evidence",
            ),
        )
        if not source_excerpt and isinstance(condition, dict):
            source_excerpt = str(
                condition.get("condition_text_transcribed")
                or condition.get("condition_text")
                or condition.get("source_text")
                or condition.get("source_excerpt")
                or condition.get("source_grounding")
                or ""
            )
        if isinstance(raw.get("structure_derivation"), dict):
            structure_derivation = dict(raw.get("structure_derivation") or {})
        else:
            structure_derivation = {
                "basis": "current_pdf_image_to_smiles",
                "source_locator": source_locator,
                "confidence": str(parsed.get("confidence") or raw.get("confidence") or "low"),
                "tool_checks": ["visual extraction performed in this run", "RDKit parse precheck performed locally"],
            }
        confidence = str(raw.get("confidence") or parsed.get("confidence") or "low")
        stereochemistry_status = str(
            raw.get("stereochemistry_status")
            or raw.get("stereo_status")
            or structure_derivation.get("stereochemistry_status")
            or ""
        ).strip()
        derivation_basis = str(structure_derivation.get("basis") or "").lower()
        risk_flags = _dedupe(
            [
                *[str(item) for item in raw.get("risk_flags") or [] if str(item or "").strip()],
                *[str(item) for item in structure_derivation.get("risk_flags") or [] if str(item or "").strip()],
            ]
        )
        approximate = bool(
            raw.get("not_exact_literature_segment")
            or raw.get("approximate_structure")
            or raw.get("connectivity_only")
            or "approx" in derivation_basis
            or "achiral" in derivation_basis
            or "connectivity" in derivation_basis
            or stereochemistry_status in {"unspecified", "unspecified_or_partial", "partial", "unknown"}
        )
        if approximate:
            stereochemistry_status = stereochemistry_status or "unspecified_or_partial"
            structure_derivation["not_exact_literature_segment"] = True
            structure_derivation["approximate_structure"] = True
            structure_derivation["allowed_use"] = "exploratory_template_and_guided_hint_only"
            structure_derivation["stereochemistry_status"] = stereochemistry_status
            risk_flags = _dedupe([*risk_flags, "stereochemistry_unspecified", "exploratory_visual_candidate"])
        if not reactant_smiles and not main_reactant and not _rdkit_valid_smiles(product_smiles):
            checks = [str(item) for item in structure_derivation.get("tool_checks") or [] if str(item or "").strip()]
            checks.append("reactant structure not visible or not confidently mapped in visual extraction")
            structure_derivation["tool_checks"] = checks
            structure_derivation["structure_gap"] = True
            structure_derivation["advisory_condition_only_step"] = True
        elif not reactant_smiles and not main_reactant:
            checks = [str(item) for item in structure_derivation.get("tool_checks") or [] if str(item or "").strip()]
            checks.append("product/anchor structure visible; precursor not visible or not confidently mapped")
            structure_derivation["tool_checks"] = checks
            structure_derivation["visual_structure_anchor_only"] = True
            structure_derivation["not_exact_literature_segment"] = True
            structure_derivation["allowed_use"] = "exploratory_template_and_guided_hint_only"
            risk_flags = _dedupe([*risk_flags, "visual_structure_anchor_only", "precursor_not_visible"])
            approximate = True
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
            "structure_derivation": structure_derivation,
            "stereochemistry_status": stereochemistry_status or "specified",
            "not_exact_literature_segment": bool(approximate),
            "allowed_use": "exploratory_template_and_guided_hint_only" if approximate else str(raw.get("allowed_use") or "exact_candidate"),
            "risk_flags": risk_flags,
            "source_excerpt": source_excerpt,
            "confidence": confidence,
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


def _product_label_smiles_index(raw_items: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        label = _first_nonempty_string(
            row,
            (
                "product_label",
                "visible_label",
                "visible_product_label",
                "mapped_candidate_label",
                "candidate_product_label",
                "target_label",
                "label",
                "product_name",
            ),
        )
        smiles = str(row.get("product_smiles") or row.get("product") or row.get("smiles") or "").strip()
        if label and smiles and label not in out:
            out[label] = smiles
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
    condition_text = str(
        raw.get("condition_text_transcribed")
        or raw.get("condition_text")
        or raw.get("source_text")
        or raw.get("source_excerpt")
        or ""
    )
    other_visible_process_text = raw.get("other_visible_process_text")
    if isinstance(other_visible_process_text, (list, tuple)):
        other_visible_process_text_value = "; ".join(
            str(item).strip() for item in other_visible_process_text if str(item or "").strip()
        )
    else:
        other_visible_process_text_value = str(other_visible_process_text or "").strip()
    return {
        "schema_version": "condition_candidate.v1",
        "source_type": "exact",
        "condition_status": "evidence_backed",
        "reagent": str(raw.get("reagent") or raw.get("reagents") or ""),
        "catalyst": str(raw.get("catalyst") or raw.get("catalysts") or ""),
        "base": str(raw.get("base") or ""),
        "oxidant": str(raw.get("oxidant") or ""),
        "solvent": str(raw.get("solvent") or ""),
        "temperature": str(raw.get("temperature") or ""),
        "duration": str(raw.get("duration") or raw.get("time") or ""),
        "reported_yield": str(raw.get("reported_yield") or raw.get("yield") or ""),
        "condition_text_transcribed": condition_text,
        "source_excerpt": condition_text,
        "other_visible_process_text": other_visible_process_text_value,
        "source_grounding": str(raw.get("source_grounding") or raw.get("source_scheme") or "current PDF scheme image"),
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
    expected_labels = meaningful_compound_labels(expected_labels)
    labels = _labels_from_chain(chain)
    gap_labels = _labels_from_gaps(chain)
    missing = [label for label in expected_labels if label not in labels and label not in gap_labels]
    smiles_precheck = _smiles_precheck(chain)
    valid_smiles = int(smiles_precheck.get("valid_smiles_count") or 0)
    invalid_smiles = int(smiles_precheck.get("invalid_smiles_count") or 0)
    placeholders = int(smiles_precheck.get("placeholder_smiles_count") or 0)
    unresolved_gaps = len(_blocking_extraction_gaps(chain))
    condition_gaps = _condition_gap_rows(chain)
    structure_gaps = _structure_gap_rows(chain)
    steps = [row for row in chain.get("steps") or [] if isinstance(row, dict)]
    exploratory_steps = [
        row
        for row in steps
        if _step_is_exploratory_visual_candidate(row)
    ]
    rdkit_route_step_count = _rdkit_valid_route_step_count(steps)
    rdkit_structure_anchor_count = _rdkit_valid_structure_anchor_count(steps)
    grounded_conditions = sum(
        1 for row in steps if _condition_has_source_grounded_content(dict(row.get("condition_candidate") or {}))
    )
    exact_ready = (
        not missing
        and not unresolved_gaps
        and not condition_gaps
        and not structure_gaps
        and not exploratory_steps
        and not invalid_smiles
        and not placeholders
        and bool(steps)
    )
    exploratory_ready = (
        (rdkit_route_step_count > 0 or rdkit_structure_anchor_count > 0)
        and valid_smiles > 0
        and not invalid_smiles
        and not placeholders
        and not structure_gaps
    )
    score = (
        len(labels) * 10
        + valid_smiles * 3
        + len(steps) * 4
        + grounded_conditions * 3
        - len(missing) * 12
        - invalid_smiles * 20
        - placeholders * 25
        - unresolved_gaps * 5
        - len(condition_gaps) * 4
        - len(structure_gaps) * 8
    )
    return {
        "schema_version": "visual_literature_chain_quality_audit.v1",
        "expected_labels": [str(item) for item in expected_labels],
        "observed_step_labels": sorted(labels),
        "gap_labels": sorted(gap_labels),
        "missing_expected_labels": missing,
        "extraction_gap_count": unresolved_gaps,
        "condition_gap_count": len(condition_gaps),
        "structure_gap_count": len(structure_gaps),
        "step_count": len(steps),
        "rdkit_route_step_count": rdkit_route_step_count,
        "rdkit_structure_anchor_count": rdkit_structure_anchor_count,
        "source_grounded_condition_count": grounded_conditions,
        "exploratory_step_count": len(exploratory_steps),
        "condition_gap_labels": [str(row.get("product_label") or "") for row in condition_gaps],
        "condition_gaps": condition_gaps,
        "structure_gaps": structure_gaps,
        "nonblocking_extraction_gap_count": len(chain.get("extraction_gaps") or []) - unresolved_gaps,
        "smiles_precheck": smiles_precheck,
        "score": score,
        "exact_ready": exact_ready,
        "exploratory_accepted": exploratory_ready,
        "acceptance_level": "exact_source_detail_candidate" if exact_ready else (
            "exploratory_connectivity_candidate" if exploratory_ready else "rejected"
        ),
        "accepted": bool(exact_ready or exploratory_ready),
    }


def _step_is_exploratory_visual_candidate(step: dict[str, Any]) -> bool:
    derivation = dict(step.get("structure_derivation") or {})
    text = " ".join(
        [
            str(step.get("allowed_use") or ""),
            str(step.get("stereochemistry_status") or ""),
            str(derivation.get("basis") or ""),
            str(derivation.get("allowed_use") or ""),
            str(derivation.get("stereochemistry_status") or ""),
            " ".join(str(item) for item in step.get("risk_flags") or []),
            " ".join(str(item) for item in derivation.get("risk_flags") or []),
        ]
    ).lower()
    return bool(
        step.get("not_exact_literature_segment")
        or derivation.get("not_exact_literature_segment")
        or derivation.get("approximate_structure")
        or "exploratory" in text
        or "achiral" in text
        or "connectivity" in text
        or "unspecified" in text
        or "partial" in text
    )


def _rdkit_valid_route_step_count(steps: list[dict[str, Any]]) -> int:
    count = 0
    for step in steps:
        product = str(step.get("product_smiles") or "")
        reactants = [str(item) for item in step.get("reactant_smiles") or [] if str(item or "").strip()]
        if not reactants and str(step.get("main_reactant_smiles") or "").strip():
            reactants = [str(step.get("main_reactant_smiles") or "").strip()]
        if _rdkit_valid_smiles(product) and any(_rdkit_valid_smiles(item) for item in reactants):
            count += 1
    return count


def _rdkit_valid_structure_anchor_count(steps: list[dict[str, Any]]) -> int:
    count = 0
    for step in steps:
        product = str(step.get("product_smiles") or "")
        derivation = dict(step.get("structure_derivation") or {})
        if _rdkit_valid_smiles(product) and (
            derivation.get("visual_structure_anchor_only")
            or str(step.get("allowed_use") or "") == "exploratory_template_and_guided_hint_only"
            or bool(step.get("not_exact_literature_segment"))
        ):
            count += 1
    return count


def _should_select_repair_candidate(
    *,
    current_chain: dict[str, Any],
    current_quality: dict[str, Any],
    repair_chain: dict[str, Any],
    repair_quality: dict[str, Any],
) -> bool:
    if not _chain_has_extracted_evidence(repair_chain):
        return False
    if not _chain_has_extracted_evidence(current_chain):
        return True
    if int(repair_quality.get("score") or 0) > int(current_quality.get("score") or 0):
        return True
    current_steps = len(current_chain.get("steps") or [])
    repair_steps = len(repair_chain.get("steps") or [])
    if current_steps == 0 and repair_steps > 0:
        return True
    if (
        repair_steps > current_steps
        and int(repair_quality.get("source_grounded_condition_count") or 0)
        > int(current_quality.get("source_grounded_condition_count") or 0)
        and int(repair_quality.get("score") or 0) >= int(current_quality.get("score") or 0) - 20
    ):
        return True
    return False


def _chain_has_extracted_evidence(chain: dict[str, Any]) -> bool:
    if not isinstance(chain, dict):
        return False
    if any(isinstance(row, dict) for row in chain.get("steps") or []):
        return True
    if any(isinstance(row, dict) for row in chain.get("extraction_gaps") or []):
        return True
    return False


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
    if quality.get("exploratory_accepted") and not int(smiles.get("invalid_smiles_count") or 0) and not int(smiles.get("placeholder_smiles_count") or 0):
        return False
    return bool(
        quality.get("missing_expected_labels")
        or int(quality.get("extraction_gap_count") or 0)
        or int(quality.get("condition_gap_count") or 0)
        or int(quality.get("structure_gap_count") or 0)
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


def _structure_gap_rows(chain: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step_index, step in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        derivation = dict(step.get("structure_derivation") or {})
        if not derivation.get("structure_gap"):
            continue
        out.append(
            {
                "step_index": step_index,
                "step_id": str(step.get("step_id") or ""),
                "product_label": str(step.get("product_label") or ""),
                "reason": "structure_gap",
                "source_locator": str(step.get("source_locator") or ""),
            }
        )
    return out


def _condition_has_source_grounded_content(condition: dict[str, Any]) -> bool:
    return any(
        str(condition.get(key) or "").strip()
        for key in (
            "reagent",
            "catalyst",
            "enzyme",
            "base",
            "oxidant",
            "solvent",
            "temperature",
            "duration",
            "reported_yield",
            "ph",
            "buffer",
            "condition_text_transcribed",
            "condition_text",
            "source_excerpt",
        )
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
        if main_reactant or step.get("reactant_smiles"):
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
                f"model = {_toml_string(model)}",
                "",
                "[model_providers.wellau]",
                'name = "wellau"',
                f"base_url = {_toml_string(base_url.rstrip('/'))}",
                'env_key = "OPENAI_API_KEY"',
                'wire_api = "responses"',
                "",
                "[tools]",
                "web_search = false",
                "",
                "[sandbox_workspace_write]",
                f"writable_roots = [{_toml_string(str(run_dir))}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)
