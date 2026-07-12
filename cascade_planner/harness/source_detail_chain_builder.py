"""Build and audit source-detail route chains from validated literature structures."""
from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cascade_planner.baselines.literature_one_step_plugin import (
    LiteratureOneStepPlugin,
    LiteratureOneStepPluginConfig,
)
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.harness.downstream_compiler import (
    compile_downstream_consumables,
    write_compiled_downstream_artifacts,
)
from cascade_planner.harness.reaction_step_verifier import canonical_reaction_digest
from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    independent_source_group,
)
from cascade_planner.harness.source_detail_resolution import (
    SOURCE_DETAIL_CURATOR_RECORDS_SCHEMA,
    resolve_source_detail_extraction_pack,
    source_detail_curator_records_path,
)


SOURCE_DETAIL_CHAIN_AUDIT_SCHEMA = "source_detail_route_chain_audit.v1"
HYBRID_ROUTE_SET_SCHEMA = "hybrid_route_set.v1"


def build_source_detail_curator_records_from_chain(
    validation: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    source_ref: str = "",
    source_title: str = "",
    evidence_refs: list[str] | None = None,
    record_id: str = "",
    provenance: str = "codex_source_text_translation",
    main_reactant_only: bool = False,
    write_file: bool = True,
) -> dict[str, Any]:
    payload = _load_jsonish(validation)
    out = Path(output_dir)
    steps = [dict(item) for item in payload.get("steps") or [] if isinstance(item, dict)]
    global_evidence = _dedupe([*(_string_list(payload.get("evidence_refs"))), *(_string_list(evidence_refs))])
    resolved_source_ref = source_ref or str(payload.get("source_ref") or "")
    resolved_source_title = source_title or str(payload.get("source_title") or "")
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not step.get("accepted"):
            skipped.append({"step_id": str(step.get("step_id") or f"step_{index}"), "reasons": step.get("reasons") or []})
            continue
        condition = dict(step.get("condition_candidate") or {})
        if "evidence_refs" not in condition and (step.get("evidence_refs") or global_evidence):
            condition["evidence_refs"] = _dedupe([*_string_list(step.get("evidence_refs")), *global_evidence])
        source_excerpt = str(step.get("source_excerpt") or "").strip()
        records.append(
            {
                "schema_version": "source_detail_route_step.v1",
                "step_id": str(step.get("step_id") or f"source_detail_chain_step_{index}"),
                "segment_id": str(step.get("segment_id") or "source_detail_literature_chain"),
                "source_ref": str(step.get("source_ref") or resolved_source_ref),
                "source_title": str(step.get("source_title") or resolved_source_title),
                "evidence_refs": _dedupe([*_string_list(step.get("evidence_refs")), *global_evidence]),
                "product_name": str(step.get("product_label") or ""),
                "reactant_names": [str(item) for item in step.get("reactant_labels") or [] if str(item).strip()],
                "product_smiles": str((step.get("product") or {}).get("canonical_smiles") or step.get("product_smiles") or ""),
                "reactant_smiles": _reactants_for_curator_step(step, main_only=main_reactant_only),
                "relation_type": "exact",
                "condition_candidate": condition,
                "provenance": provenance,
                "source_excerpt": source_excerpt,
                "structure_derivation": dict(step.get("structure_derivation") or {}),
                "validation_status": "draft_validated_by_rdkit_chain",
                "curation_status": "codex_visual_source_translation_draft",
                "full_text_content_stored": False,
                "procedure_text_stored": False,
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )

    result = {
        "schema_version": SOURCE_DETAIL_CURATOR_RECORDS_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": record_id or f"{_safe_id(str(payload.get('case_id') or 'literature'))}_visual_chain_curator_records",
        "records": records,
        "summary": {
            "input_step_count": len(steps),
            "record_count": len(records),
            "skipped_step_count": len(skipped),
            "main_reactant_only": bool(main_reactant_only),
        },
        "skipped_steps": skipped,
        "source_policy": {
            "from_visual_structure_chain_validation": True,
            "not_route_evidence_until_source_detail_resolution": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    if write_file:
        path = source_detail_curator_records_path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def resolve_curator_records_to_source_detail_steps(
    curator_records: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    target_name: str = "",
    target_smiles: str = "",
    source_ref: str = "",
) -> dict[str, Any]:
    pack = {
        "schema_version": "source_detail_extraction_pack.v1",
        "target": {"name": target_name, "smiles": target_smiles},
        "queue": [],
    }
    return resolve_source_detail_extraction_pack(
        pack,
        output_dir=output_dir,
        curator_records=curator_records,
        fetch_json=lambda url, headers, timeout_s: {},
        fetch_text=lambda url, headers, timeout_s: "",
    )


def compile_source_detail_chain_route(
    *,
    source_detail_steps: list[dict[str, Any]] | None = None,
    compiled_downstream: dict[str, Any] | str | Path | None = None,
    output_dir: str | Path,
    target_smiles: str,
    case_id: str,
    terminal_smiles: str = "",
    terminal_name: str = "",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiled = _compiled_payload(compiled_downstream)
    if not compiled:
        payload = _downstream_payload_from_steps(source_detail_steps or [], case_id=case_id)
        compiled = compile_downstream_consumables(payload, target_smiles=target_smiles, case_id=case_id)
        write_compiled_downstream_artifacts(compiled, output_dir=out)
    rows = [dict(item) for item in ((compiled.get("literature_template_plugin") or {}).get("one_step_rows") or []) if isinstance(item, dict)]
    audit = audit_source_detail_route_chain(
        rows,
        target_smiles=target_smiles,
        case_id=case_id,
        terminal_smiles=terminal_smiles,
        terminal_name=terminal_name,
    )
    path = out / "source_detail_route_chain_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    curation_outbox = build_trusted_precedent_curation_outbox(
        audit.get("chain") or [],
        case_id=case_id,
    )
    curation_outbox_path = out / "trusted_precedent_curation_outbox.json"
    curation_outbox_path.write_text(
        json.dumps(
            curation_outbox,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "compiled_source_detail_chain_route.v1",
        "accepted": bool(compiled.get("accepted")) and bool(audit.get("accepted")),
        "compiled_downstream": compiled,
        "chain_audit": audit,
        "trusted_precedent_curation_outbox": curation_outbox,
        "artifact_refs": {
            "source_detail_route_chain_audit": str(path),
            "trusted_precedent_curation_outbox": str(curation_outbox_path),
            **{
                key: str(out / name)
                for key, name in {
                    "compiled_downstream_consumables": "compiled_downstream_consumables.json",
                    "compiled_literature_template_plugin": "compiled_literature_template_plugin.json",
                }.items()
                if (out / name).exists()
            },
        },
        "reasons": [*([str(item) for item in compiled.get("reasons") or []]), *([str(item) for item in audit.get("reasons") or []])],
    }


def build_trusted_precedent_curation_outbox(
    chain_steps: list[dict[str, Any]],
    *,
    case_id: str,
) -> dict[str, Any]:
    """Create immutable, non-promoting L3 curation candidates.

    Visual/model extraction may prepare this queue, but only a human curator
    or deterministic structure parser can write a separate approved registry
    binding.  The outbox itself is never consumed as proof authority.
    """

    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(chain_steps, start=1):
        if not isinstance(raw, dict):
            continue
        step = dict(raw)
        product = canonical_smiles(str(step.get("product_smiles") or ""))
        reactants = [
            canonical_smiles(str(item))
            for item in step.get("reactant_smiles") or []
            if canonical_smiles(str(item))
        ]
        source_ref = canonical_traceable_source_ref(step.get("source_ref"))
        reaction_digest = canonical_reaction_digest(product, reactants)
        evidence_rows = [
            dict(item)
            for item in step.get("source_evidence") or []
            if isinstance(item, dict)
        ]
        exact_validation = dict(step.get("exact_step_validation") or {})
        if not product or not reactants or not source_ref or not reaction_digest:
            continue
        candidate_id = "curation:" + hashlib.sha256(
            json.dumps(
                {
                    "reaction_digest": reaction_digest,
                    "source_ref": source_ref,
                    "evidence": evidence_rows,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        candidates.append(
            {
                "schema_version": "trusted_precedent_curation_candidate.v1",
                "candidate_id": candidate_id,
                "case_id": str(case_id),
                "step_index": int(step.get("step_index") or index),
                "step_id": str(step.get("step_id") or ""),
                "reaction_digest": reaction_digest,
                "product_smiles": product,
                "reactant_smiles": sorted(reactants),
                "source_ref": source_ref,
                "independent_source_group": independent_source_group(
                    {"source_ref": source_ref}
                ),
                "evidence_refs": [
                    str(item) for item in step.get("evidence_refs") or []
                ],
                "materialized_source_evidence": evidence_rows,
                "materialized_source_evidence_sha256": hashlib.sha256(
                    json.dumps(
                        evidence_rows,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "exact_step_validation": exact_validation,
                "status": "pending_curator_or_deterministic_parser",
                "promotion_allowed": False,
                "production_write_blocked": True,
                "required_authority": [
                    "human_curator",
                    "deterministic_structure_parser",
                ],
                "required_registry_binding_fields": [
                    "binding_id",
                    "reaction_digest",
                    "source_ref",
                    "document_id",
                    "source_pdf_sha256",
                    "page_number",
                    "image_sha256",
                    "status=approved",
                    "authority.type",
                    "authority.id",
                ],
                "no_solved_claim": True,
            }
        )
    candidates.sort(
        key=lambda row: (
            str(row.get("reaction_digest") or ""),
            str(row.get("source_ref") or ""),
            str(row.get("candidate_id") or ""),
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "trusted_precedent_curation_outbox.v1",
        "case_id": str(case_id),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "production_write_blocked": True,
        "auto_promotion_allowed": False,
        "consumer": "out_of_band_trusted_literature_step_registry_curator",
        "semantics": {
            "model_cannot_self_sign": True,
            "outbox_is_not_precedent_authority": True,
            "approved_binding_must_be_replayed_by_current_host": True,
        },
        "no_solved_claim": True,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def audit_source_detail_route_chain(
    one_step_rows: list[dict[str, Any]],
    *,
    target_smiles: str,
    case_id: str,
    terminal_smiles: str = "",
    terminal_name: str = "",
) -> dict[str, Any]:
    product_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in one_step_rows:
        trace = _row_trace(row)
        product = canonical_smiles(str(trace.get("product_smiles") or trace.get("frontier_smiles") or ""))
        if product:
            product_to_rows.setdefault(product, []).append(row)
    reasons: list[str] = []
    target_key = canonical_smiles(target_smiles)
    terminal_key = canonical_smiles(terminal_smiles)
    chain: list[dict[str, Any]] = []
    seen_products: set[str] = set()
    current = target_key
    while current:
        if current in seen_products:
            reasons.append("cycle_in_source_detail_chain")
            break
        seen_products.add(current)
        candidates = product_to_rows.get(current) or []
        if not candidates:
            reasons.append("missing_one_step_row_for_product")
            break
        row = candidates[0]
        trace = _row_trace(row)
        exact_validation = _row_exact_validation(row)
        source_evidence = _materialized_source_evidence(trace)
        reactants = [str(item) for item in trace.get("reactant_smiles") or [] if str(item).strip()]
        main = _main_reactant(reactants)
        chain_step = {
                "step_index": len(chain) + 1,
                "source_template_id": str(trace.get("source_template_id") or ""),
                "step_id": str(trace.get("source_template_id") or "").replace("source_detail_exact_step:", ""),
                "product_smiles": current,
                "reactant_smiles": reactants,
                "main_reactant_smiles": main,
                "source_ref": str(trace.get("source_ref") or ""),
                "evidence_refs": [str(item) for item in trace.get("evidence_refs") or []],
                "condition_candidate": dict(trace.get("condition_candidate") or {}),
                "relation_type": str(trace.get("relation_type") or ""),
                "source_detail_exact_step": trace.get("source_detail_exact_step") is True,
                "exact_step_validation": exact_validation,
                "source_evidence": source_evidence,
            }
        chain.append(chain_step)
        if terminal_key and main == terminal_key:
            break
        if not main or main not in product_to_rows:
            break
        current = main
    observed_terminal = _observed_chain_terminal(chain)
    effective_terminal_smiles = terminal_smiles or str(observed_terminal.get("smiles") or "")
    effective_terminal_name = terminal_name or str(observed_terminal.get("name") or "")
    effective_terminal_key = canonical_smiles(effective_terminal_smiles)
    requested_terminal_reached = bool(chain and terminal_key and chain[-1].get("main_reactant_smiles") == terminal_key)
    observed_terminal_reached = bool(chain and not terminal_key and effective_terminal_key)
    terminal_reached = requested_terminal_reached or observed_terminal_reached
    if terminal_key and not terminal_reached:
        reasons.append("terminal_not_reached")
    if not chain:
        reasons.append("no_chain_unrolled")
    return {
        "schema_version": SOURCE_DETAIL_CHAIN_AUDIT_SCHEMA,
        "accepted": bool(chain) and not reasons,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "target_smiles": target_smiles,
        "target_canonical_smiles": target_key or "",
        "terminal_name": effective_terminal_name,
        "terminal_smiles": effective_terminal_smiles,
        "terminal_canonical_smiles": effective_terminal_key or "",
        "terminal_requested": bool(terminal_key),
        "terminal_reached": terminal_reached,
        "observed_terminal_smiles": observed_terminal.get("smiles") or "",
        "observed_terminal_canonical_smiles": observed_terminal.get("canonical_smiles") or "",
        "terminal_stereo_repair": {},
        "step_count": len(chain),
        "chain": chain,
        "summary": {
            "one_step_row_count": len(one_step_rows),
            "chain_step_count": len(chain),
            "terminal_reached": terminal_reached,
            "terminal_requested": bool(terminal_key),
            "observed_terminal_reached": observed_terminal_reached,
        },
        "source_policy": {
            "no_solved_claim": True,
            "production_write_blocked": True,
            "literature_chain_is_baseline_not_mandatory_replacement": True,
            "automatic_terminal_identity_repair_allowed": False,
        },
        "reasons": sorted(set(reasons)),
    }


def _observed_chain_terminal(chain: list[dict[str, Any]]) -> dict[str, str]:
    if not chain:
        return {}
    last = dict(chain[-1])
    smiles = str(last.get("main_reactant_smiles") or "")
    return {
        "name": str(last.get("main_reactant_name") or last.get("main_reactant_label") or ""),
        "smiles": smiles,
        "canonical_smiles": canonical_smiles(smiles),
    }


def probe_literature_plugin_chain(
    *,
    plugin_payload: dict[str, Any] | str | Path,
    expected_steps: list[dict[str, Any]] | None = None,
    validation: dict[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    plugin_data = _plugin_payload(plugin_payload)
    config = LiteratureOneStepPluginConfig.from_raw(plugin_data)
    plugin = LiteratureOneStepPlugin(config=config)
    steps = expected_steps if expected_steps is not None else _expected_steps_from_validation(validation)
    probes: list[dict[str, Any]] = []
    reasons: list[str] = []
    for index, step in enumerate(steps, start=1):
        product = str((step.get("product") or {}).get("canonical_smiles") or step.get("product_smiles") or "")
        expected_reactants = _expected_reactants(step)
        rows = plugin.one_step_rows(product, top_k=max(1, config.max_added or 1))
        observed = [str(row.get("reactants") or "") for row in rows]
        matched = any(_reactant_side_matches(item, expected_reactants) for item in observed)
        if not matched:
            reasons.append("plugin_probe_missing_expected_row")
        probes.append(
            {
                "probe_index": index,
                "step_id": str(step.get("step_id") or ""),
                "product_smiles": product,
                "expected_reactant_smiles": expected_reactants,
                "observed_reactant_sides": observed,
                "matched": matched,
            }
        )
    result = {
        "schema_version": "literature_plugin_chain_probe.v1",
        "accepted": not reasons,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_count": len(probes),
        "matched_count": sum(1 for probe in probes if probe.get("matched")),
        "plugin_config": config.to_dict(),
        "probes": probes,
        "reasons": sorted(set(reasons)),
        "source_policy": {
            "probe_only": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "literature_plugin_chain_probe.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def compile_hybrid_route_set(
    *,
    output_dir: str | Path,
    case_id: str,
    target_smiles: str,
    literature_chain_audit: dict[str, Any] | str | Path | None = None,
    chemenzy_result: dict[str, Any] | str | Path | None = None,
    verifier_report: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chain = _load_jsonish(literature_chain_audit) if literature_chain_audit else {}
    chem = _load_jsonish(chemenzy_result) if chemenzy_result else {}
    verifier = _load_jsonish(verifier_report) if verifier_report else {}
    routes: list[dict[str, Any]] = []
    if chain:
        chain_accepted = chain.get("accepted") is True
        routes.append(
            {
                "route_id": (
                    f"{case_id}_literature_exact_chain"
                    if chain_accepted
                    else f"{case_id}_literature_advisory_chain"
                ),
                "route_type": (
                    "literature_exact_chain"
                    if chain_accepted
                    else "literature_advisory_chain"
                ),
                "status": "baseline" if chain_accepted else "needs_exact_curation",
                "score": 0.9 if chain_accepted else 0.35,
                "step_count": int(chain.get("step_count") or len(chain.get("chain") or [])),
                "target_smiles": target_smiles,
                "chain": chain.get("chain") or [],
                "not_mandatory_replacement": True,
                "no_solved_claim": True,
            }
        )
    chem_routes = (chem.get("routes") or (chem.get("result") or {}).get("routes") or []) if isinstance(chem, dict) else []
    for index, route in enumerate(chem_routes[:20], start=1):
        routes.append(
            {
                "route_id": f"{case_id}_chemenzy_exploratory_{index}",
                "route_type": "chemenzy_exploratory",
                "status": "accepted_by_verifier" if verifier.get("accepted") else "verifier_rejected_or_unverified",
                "score": 0.65 if verifier.get("accepted") else 0.35,
                "route": route,
                "verifier_reasons": [str(item) for item in verifier.get("reasons") or []],
                "no_solved_claim": True,
            }
        )
    routes.sort(key=lambda row: (float(row.get("score") or 0.0), -int(row.get("step_count") or 999)), reverse=True)
    result = {
        "schema_version": HYBRID_ROUTE_SET_SCHEMA,
        "accepted": bool(routes),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "target_smiles": target_smiles,
        "routes": routes,
        "summary": {
            "route_count": len(routes),
            "literature_route_count": sum(1 for row in routes if row.get("route_type") == "literature_exact_chain"),
            "literature_advisory_route_count": sum(
                1
                for row in routes
                if row.get("route_type") == "literature_advisory_chain"
            ),
            "chemenzy_route_count": sum(1 for row in routes if row.get("route_type") == "chemenzy_exploratory"),
            "verifier_accepted": bool(verifier.get("accepted")),
        },
        "policy": {
            "literature_path_high_weight_baseline": True,
            "literature_path_not_required": True,
            "chemenzy_exploration_retained": True,
            "raw_solved_not_equivalent_to_verified_solved": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
    }
    (out / "hybrid_route_set.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _downstream_payload_from_steps(steps: list[dict[str, Any]], *, case_id: str) -> dict[str, Any]:
    return {
        "schema_version": "open_downstream_consumables.v1",
        "case_id": case_id,
        "planner_handoff": {
            "next_action": "template_plugin_rerun",
            "solved": False,
            "production_kb_promotion": False,
        },
        "guided_rerun_requests": [],
        "literature_template_cards": [],
        "literature_route_segments": [],
        "executable_template_candidates": [],
        "source_detail_route_steps": steps,
        "route_expansion_tasks": [],
        "evolution_candidates": [],
        "rejected_consumables": [],
    }


def _reactants_for_curator_step(step: dict[str, Any], *, main_only: bool) -> list[str]:
    main = str(step.get("main_reactant_smiles") or "")
    if main_only and main:
        return [main]
    reactants = [
        str(report.get("canonical_smiles") or report.get("input_smiles") or "")
        for report in step.get("reactants") or []
        if isinstance(report, dict)
    ]
    if main and main not in reactants:
        reactants.insert(0, main)
    return [item for item in reactants if item]


def _compiled_payload(value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if not value:
        return {}
    data = _load_jsonish(value)
    if data.get("literature_template_plugin"):
        return data
    return {}


def _plugin_payload(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    data = _load_jsonish(value)
    if data.get("literature_template_plugin"):
        return dict(data.get("literature_template_plugin") or {})
    return data


def _expected_steps_from_validation(value: dict[str, Any] | str | Path | None) -> list[dict[str, Any]]:
    if not value:
        return []
    data = _load_jsonish(value)
    return [dict(item) for item in data.get("steps") or [] if isinstance(item, dict)]


def _expected_reactants(step: dict[str, Any]) -> list[str]:
    if step.get("reactants"):
        return [
            str(report.get("canonical_smiles") or report.get("input_smiles") or "")
            for report in step.get("reactants") or []
            if isinstance(report, dict)
        ]
    return [str(item) for item in step.get("reactant_smiles") or [] if str(item).strip()]


def _reactant_side_matches(observed_side: str, expected: list[str]) -> bool:
    observed = {canonical_smiles(item) or item for item in str(observed_side or "").split(".") if item}
    expected_set = {canonical_smiles(item) or item for item in expected if item}
    return bool(expected_set) and expected_set.issubset(observed)


def _row_trace(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("literature_template_trace") or ((row.get("template") or {}).get("literature_template_trace") or {}))


def _row_exact_validation(row: dict[str, Any]) -> dict[str, Any]:
    template = dict(row.get("template") or {})
    report = dict(
        row.get("template_validation_report")
        or template.get("template_validation_report")
        or {}
    )
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "accepted": report.get("accepted") is True,
        "allowed_for_one_step_source": report.get("allowed_for_one_step_source") is True,
        "source_template_id": str(report.get("source_template_id") or ""),
        "reasons": [str(item) for item in report.get("reasons") or []],
    }


def _materialized_source_evidence(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_source_ref = str(trace.get("source_ref") or "").strip().lower()
    for raw_ref in trace.get("evidence_refs") or []:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        candidate = re.split(r"::(?:doi|page|pages?|crop|image):|#page=", ref, maxsplit=1, flags=re.I)[0]
        path = Path(candidate).expanduser()
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            manifest_bytes = path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "literature_pdf_structure_evidence.v1":
            continue
        binding = dict(manifest.get("source_binding_audit") or {})
        manifest_source_ref = str(manifest.get("source_ref") or binding.get("source_ref") or "").strip().lower()
        if (
            manifest.get("accepted") is not True
            or binding.get("schema_version") != "local_pdf_source_binding_audit.v1"
            or binding.get("accepted") is not True
            or int(binding.get("matched_source_count") or 0) <= 0
            or not expected_source_ref
            or manifest_source_ref != expected_source_ref
        ):
            continue
        pdf_path = Path(str(manifest.get("source_pdf_path") or "")).expanduser()
        pdf_sha = str(manifest.get("source_pdf_sha256") or "").lower()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf" or len(pdf_sha) != 64:
            continue
        page_filter = re.search(r"(?:#page=|::pages?:)(\d+)", ref, flags=re.I)
        for page in manifest.get("rendered_pages") or []:
            if not isinstance(page, dict):
                continue
            page_number = int(page.get("page_number") or 0)
            if page_number <= 0 or (page_filter and page_number != int(page_filter.group(1))):
                continue
            image_path = Path(str(page.get("image_path") or "")).expanduser()
            image_sha = str(page.get("sha256") or "").lower()
            if not image_path.is_file() or len(image_sha) != 64:
                continue
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            document_id = str((binding.get("matched_document_ids") or [""])[0] or f"sha256:{pdf_sha}")
            rows.append(
                {
                    "schema_version": "materialized_source_evidence.v1",
                    "document_id": document_id,
                    "manifest_path": str(path.resolve()),
                    "manifest_sha256": manifest_digest,
                    "source_pdf_path": str(pdf_path.resolve()),
                    "source_pdf_sha256": pdf_sha,
                    "page_number": page_number,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": image_sha,
                    "source_ref": manifest_source_ref,
                }
            )
    return rows


def _source_detail_chain_step_provenance_valid(step: dict[str, Any]) -> bool:
    validation = dict(step.get("exact_step_validation") or {})
    template_id = str(step.get("source_template_id") or "")
    evidence = [dict(item) for item in step.get("source_evidence") or [] if isinstance(item, dict)]
    return bool(
        template_id.startswith("source_detail_exact_step:")
        and step.get("source_detail_exact_step") is True
        and step.get("relation_type") == "exact"
        and str(step.get("source_ref") or "").strip()
        and validation.get("schema_version") == "template_validation_report.v1"
        and validation.get("accepted") is True
        and validation.get("allowed_for_one_step_source") is True
        and str(validation.get("source_template_id") or "") == template_id
        and not validation.get("reasons")
        and evidence
        and all(Path(str(row.get("path") or "")).is_file() for row in evidence)
    )


def _main_reactant(reactants: list[str]) -> str:
    valid = [canonical_smiles(item) or item for item in reactants if item]
    return max(valid, key=len) if valid else ""


def _load_jsonish(value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    data = json.loads(Path(value).read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"
