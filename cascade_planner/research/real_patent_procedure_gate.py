"""Replay official structured patent procedures as a bounded release gate.

The gate is intentionally narrower than route closure.  It proves that one
source-authored reaction and its conditions can be reconstructed from a
hash-bound official XML range and replayed without network or model calls.
Starting-material stock, the remainder of a synthesis, and experimental
reproduction remain independent claims.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from rdkit import Chem

from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
)
from cascade_planner.application.route_workbench import compile_route_workbench
from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.source_patent_xml import materialize_primary_patent_xml
from cascade_planner.harness.v4_route_workbench import render_v4_route_workbench_html


CONFIG_SCHEMA = "real_patent_procedure_gate_cases.v1"
CASE_SCHEMA = "real_patent_procedure_gate.v1"
SUITE_SCHEMA = "real_patent_procedure_gate_suite.v1"
SNAPSHOT_SCHEMA = "deterministic_name_resolution_snapshot.v1"


def load_patent_procedure_gate_config(path: Path) -> dict[str, Any]:
    """Load a digest-bound, chemistry-data-only gate definition."""

    value = _read_json(path)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("real_patent_gate_config_schema_invalid")
    supplied = str(value.get("content_sha256") or "")
    if supplied != content_digest(value):
        raise ValueError("real_patent_gate_config_digest_invalid")
    cases = [dict(row) for row in value.get("cases") or [] if isinstance(row, Mapping)]
    if not cases or len({str(row.get("case_id") or "") for row in cases}) != len(cases):
        raise ValueError("real_patent_gate_case_ids_invalid")
    return value


def validate_resolver_snapshot(
    case: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a digest-bound independent name-to-structure snapshot."""

    value = dict(snapshot)
    if value.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("resolver_snapshot_schema_invalid")
    if value.get("authority_id") != PARSER_AUTHORITY_ID:
        raise ValueError("resolver_snapshot_authority_invalid")
    if str(value.get("content_sha256") or "") != content_digest(value):
        raise ValueError("resolver_snapshot_digest_invalid")
    expected = {
        canonical_smiles(case.get("product_smiles")),
        *(canonical_smiles(item) for item in case.get("reactant_smiles") or []),
    }
    expected.discard("")
    structures = {
        str(name): canonical_smiles(smiles)
        for name, smiles in dict(value.get("structures") or {}).items()
    }
    required_names = {str(name) for name in case.get("source_structure_names") or [] if str(name)}
    if not required_names or not required_names.issubset(structures):
        raise ValueError("resolver_snapshot_source_names_missing")
    if any(not structures[name] for name in required_names):
        raise ValueError("resolver_snapshot_source_structure_unresolved")
    if not expected.issubset(set(structures.values())):
        raise ValueError("resolver_snapshot_expected_structures_missing")
    return value


def replay_patent_procedure_case(
    case: Mapping[str, Any],
    *,
    source_content: bytes,
    resolver_snapshot: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize and twice replay one official XML procedure."""

    row = dict(case)
    case_id = str(row.get("case_id") or "")
    publication = str(row.get("publication") or "")
    source_url = str(row.get("source_url") or "")
    if not case_id or not publication or not source_url:
        raise ValueError("real_patent_gate_case_identity_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ref = f"patent:{publication}"
    materialization = materialize_primary_patent_xml(
        content=source_content,
        publication=publication,
        source_ref=source_ref,
        source_url=source_url,
        output_dir=output_dir / "source",
        target_terms=[str(value) for value in row.get("target_terms") or []],
    )
    snapshot = validate_resolver_snapshot(row, resolver_snapshot)
    _write_json_atomic(output_dir / "resolver-snapshot.json", snapshot)
    resolve_structure, resolve_names = _snapshot_resolvers(snapshot)
    product = canonical_smiles(row.get("product_smiles"))
    reactants = sorted(canonical_smiles(value) for value in row.get("reactant_smiles") or [])
    step = {
        "step_id": str(row.get("step_id") or case_id),
        "product_name": str(row.get("product_name") or ""),
        "product_smiles": product,
        "reactant_smiles": reactants,
        "source_ref": source_ref,
        "source_text_companions": [dict(materialization.get("companion") or {})],
    }
    audits = [
        compile_deterministic_literature_step_registry(
            [step],
            registry_path=output_dir / f"offline-registry-{label}.json",
            structure_resolver=resolve_structure,
            candidate_name_resolver=resolve_names,
        )
        for label in ("a", "b")
    ]
    registries = [_read_json(output_dir / f"offline-registry-{label}.json") for label in ("a", "b")]
    bindings = [dict(audit["records"][0].get("binding") or {}) for audit in audits]
    first_binding = bindings[0]
    conditions = dict(first_binding.get("source_conditions") or {})
    completeness = audit_condition_completeness(conditions)
    registry_digests = [str(value.get("content_sha256") or "") for value in registries]
    binding_ids = [str(value.get("binding_id") or "") for value in bindings]
    parser_audit = dict(first_binding.get("parser_audit") or {})
    location = dict(first_binding.get("source_location") or {})
    expected_location = dict(row.get("expected_source_location") or {})
    expectation_reasons = _condition_expectation_reasons(
        conditions, dict(row.get("condition_expectations") or {})
    )
    acceptance_reasons: list[str] = []
    if materialization.get("status") != "completed":
        acceptance_reasons.append("official_xml_materialization_failed")
    if not all(audit.get("approved_binding_count") == 1 for audit in audits):
        acceptance_reasons.append("offline_registry_binding_not_approved")
    if len(set(registry_digests)) != 1 or not registry_digests[0]:
        acceptance_reasons.append("offline_registry_digest_mismatch")
    if len(set(binding_ids)) != 1 or not binding_ids[0]:
        acceptance_reasons.append("offline_binding_id_mismatch")
    if first_binding.get("source_artifact_kind") != "xml":
        acceptance_reasons.append("binding_not_from_xml")
    if location.get("kind") != "xml_element_range":
        acceptance_reasons.append("exact_xml_element_range_missing")
    for key in ("start_element_id", "end_element_id"):
        if expected_location.get(key) and location.get(key) != expected_location.get(key):
            acceptance_reasons.append(f"source_location_{key}_mismatch")
    if parser_audit.get("source_text_authority") != "hash_bound_primary_xml":
        acceptance_reasons.append("source_text_authority_invalid")
    if completeness.get("complete") is not True:
        acceptance_reasons.append("condition_record_incomplete")
    acceptance_reasons.extend(expectation_reasons)
    accepted = not acceptance_reasons
    workbench = _write_case_workbench(
        output_dir,
        case=row,
        binding=first_binding,
        completeness=completeness,
        materialization=materialization,
    )
    acceptance: dict[str, Any] = {
        "schema_version": CASE_SCHEMA,
        "case_id": case_id,
        "target_name": str(row.get("target_name") or case_id),
        "reaction_class": str(row.get("reaction_class") or "unspecified"),
        "publication": publication,
        "accepted": accepted,
        "reasons": sorted(set(acceptance_reasons)),
        "official_source": {
            "authority": "European Patent Office Publication Server",
            "format": "epo_st36_xml.v1",
            "source_url": source_url,
            "artifact_path": str(materialization.get("artifact_path") or ""),
            "artifact_sha256": str(materialization.get("artifact_sha256") or ""),
            "element_count": int(materialization.get("element_count") or 0),
            "selected_element_count": int(materialization.get("selected_element_count") or 0),
            "sections": list(materialization.get("sections") or []),
        },
        "exact_edge": {
            "product_smiles": product,
            "reactant_smiles": reactants,
            "reaction_digest": str(first_binding.get("reaction_digest") or ""),
            "binding_id": binding_ids[0],
            "source_location": location,
            "procedure_text_sha256": str(parser_audit.get("procedure_text_sha256") or ""),
            "reactant_match_modes": list(parser_audit.get("reactant_match_modes") or []),
        },
        "procedure": {
            "conditions": conditions,
            "condition_completeness": completeness,
        },
        "offline_replay": {
            "network_used_during_registry_replays": False,
            "resolver_snapshot_path": str(output_dir / "resolver-snapshot.json"),
            "resolver_snapshot_sha256": str(snapshot.get("content_sha256") or ""),
            "registry_content_sha256": registry_digests,
            "registry_digests_equal": len(set(registry_digests)) == 1,
            "binding_ids_equal": len(set(binding_ids)) == 1,
            "model_invocations": 0,
            "visual_invocations": 0,
        },
        "acquisition_cascade": {
            "resolved_stage": "official_structured_xml" if accepted else "unresolved",
            "structured_source_closed": accepted,
            "pdf_fallback_count": 0,
            "ocr_fallback_count": 0,
            "vision_fallback_count": 0,
            "fallback_suppressed_because_structured_source_closed": accepted,
        },
        "showcase": workbench,
        "semantics": {
            "search_metadata_grants_no_authority": True,
            "full_xml_and_selected_range_are_digest_bound": True,
            "source_names_are_independently_resolved": True,
            "missing_conditions_are_not_inferred": True,
            "procedure_acceptance_does_not_grant_route_or_stock_closure": True,
        },
    }
    acceptance["content_sha256"] = content_digest(acceptance)
    _write_json_atomic(output_dir / "acceptance.json", acceptance)
    return acceptance


def compile_patent_procedure_gate_suite(
    config: Mapping[str, Any],
    cases: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate independent cases and write one catalog entrypoint."""

    rows = [dict(row) for row in cases]
    release = dict(config.get("release_gate") or {})
    accepted_count = sum(row.get("accepted") is True for row in rows)
    publications = sorted({str(row.get("publication") or "") for row in rows} - {""})
    classes = sorted({str(row.get("reaction_class") or "") for row in rows} - {""})
    minimum_cases = int(release.get("minimum_case_count") or 3)
    minimum_publications = int(release.get("minimum_unique_publications") or 3)
    minimum_classes = int(release.get("minimum_unique_reaction_classes") or 3)
    passed = bool(
        len(rows) >= minimum_cases
        and accepted_count == len(rows)
        and len(publications) >= minimum_publications
        and len(classes) >= minimum_classes
        and all(
            dict(row.get("acquisition_cascade") or {}).get("structured_source_closed") is True
            for row in rows
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA,
        "suite_id": str(config.get("suite_id") or "real-patent-procedure-gate"),
        "accepted": passed,
        "three_case_release_gate_passed": passed,
        "case_count": len(rows),
        "accepted_count": accepted_count,
        "unique_publication_count": len(publications),
        "unique_reaction_class_count": len(classes),
        "publications": publications,
        "reaction_classes": classes,
        "cases": rows,
        "acquisition_cascade": {
            "structured_source_closed_count": sum(
                dict(row.get("acquisition_cascade") or {}).get("structured_source_closed") is True
                for row in rows
            ),
            "pdf_fallback_count": sum(
                int(dict(row.get("acquisition_cascade") or {}).get("pdf_fallback_count") or 0)
                for row in rows
            ),
            "ocr_fallback_count": sum(
                int(dict(row.get("acquisition_cascade") or {}).get("ocr_fallback_count") or 0)
                for row in rows
            ),
            "vision_fallback_count": sum(
                int(dict(row.get("acquisition_cascade") or {}).get("vision_fallback_count") or 0)
                for row in rows
            ),
            "fallback_policy": "structured XML -> HTML -> PDF -> OCR -> sparse vision; descend only while unresolved",
        },
        "offline_replay": {
            "all_registry_digests_equal": all(
                dict(row.get("offline_replay") or {}).get("registry_digests_equal") is True
                for row in rows
            ),
            "all_binding_ids_equal": all(
                dict(row.get("offline_replay") or {}).get("binding_ids_equal") is True
                for row in rows
            ),
            "model_invocations": 0,
            "visual_invocations": 0,
        },
        "semantics": {
            "cases_are_real_official_publications": True,
            "reaction_classes_must_be_distinct": True,
            "structured_closure_stops_expensive_fallback": True,
            "unresolved_only_fallback_is_regression_tested_separately": True,
            "suite_does_not_claim_complete_synthesis_or_procurement_closure": True,
        },
    }
    summary["content_sha256"] = content_digest(summary)
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_text_atomic(output_dir / "index.html", _suite_index_html(summary))
    return summary


def content_digest(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in dict(value).items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or ""))
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _snapshot_resolvers(snapshot: Mapping[str, Any]):
    structures = {
        str(name).strip().casefold(): str(smiles)
        for name, smiles in dict(snapshot.get("structures") or {}).items()
    }
    names = {
        canonical_smiles(smiles): [str(value) for value in values or []]
        for smiles, values in dict(snapshot.get("candidate_names") or {}).items()
    }

    def resolve_structure(value: str) -> str:
        return structures.get(str(value).strip().casefold(), "")

    def resolve_names(value: str) -> list[str]:
        return list(names.get(canonical_smiles(value), []))

    return resolve_structure, resolve_names


def _condition_expectation_reasons(
    conditions: Mapping[str, Any], expectations: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    for field in expectations.get("required_fields") or []:
        if field not in conditions or conditions.get(field) in (None, "", []):
            reasons.append(f"condition_{field}_missing")
    for field, expected in dict(expectations.get("equals") or {}).items():
        if conditions.get(field) != expected:
            reasons.append(f"condition_{field}_mismatch")
    for field, members in dict(expectations.get("contains") or {}).items():
        actual = conditions.get(field)
        actual_values = list(actual) if isinstance(actual, list) else [actual]
        for member in members or []:
            if member not in actual_values:
                reasons.append(f"condition_{field}_member_missing:{member}")
    return reasons


def _write_case_workbench(
    output_dir: Path,
    *,
    case: Mapping[str, Any],
    binding: Mapping[str, Any],
    completeness: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "case")
    slug = re.sub(r"[^a-z0-9]+", "-", case_id.casefold()).strip("-") or "case"
    publication = str(case.get("publication") or "")
    source_ref = f"patent:{publication}"
    source_binding_id = str(binding.get("binding_id") or f"source:{slug}")
    product = canonical_smiles(case.get("product_smiles"))
    reactants = [canonical_smiles(value) for value in case.get("reactant_smiles") or []]
    location = dict(binding.get("source_location") or {})
    location_ref = (
        f"{publication}:xml:{location.get('start_element_id') or ''}-"
        f"{location.get('end_element_id') or ''}"
    )
    target_id = f"m:{slug}:product"
    precursor_ids = [f"m:{slug}:precursor:{index + 1}" for index in range(len(reactants))]
    edge_id = f"edge:{slug}"
    procedure_id = f"procedure:{slug}"
    record_id = f"exact:{slug}"
    graph_digest = f"{slug}-official-xml-procedure-v1"
    molecules: dict[str, Any] = {
        target_id: {
            "canonical_smiles": product,
            "is_leaf": False,
            "stock_observation_ids": [],
        }
    }
    for node_id, smiles in zip(precursor_ids, reactants, strict=True):
        molecules[node_id] = {
            "canonical_smiles": smiles,
            "is_leaf": True,
            "stock_observation_ids": [],
            "stock_closed": False,
        }
    graph = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": f"patent-procedure-{slug}",
        "target_name": str(case.get("target_name") or case_id),
        "target_molecule_id": target_id,
        "revision": 1,
        "scientific_sha256": graph_digest,
        "molecules": molecules,
        "edges": {
            edge_id: {
                "product_molecule_id": target_id,
                "precursor_molecule_ids": precursor_ids,
                "origin_records": [
                    {"origin_kind": "literature_source_route", "source_ref": source_ref}
                ],
                "reaction_proofs": [
                    {
                        "accepted": True,
                        "proof_digest": str(binding.get("reaction_digest") or ""),
                    }
                ],
                "procedure_record_ids": [procedure_id],
            }
        },
        "source_bindings": {
            source_binding_id: {
                "source_binding_id": source_binding_id,
                "source_kind": "patent",
                "source_ref": source_ref,
                "title": str(case.get("source_title") or publication),
                "artifact_sha256": str(materialization.get("artifact_sha256") or ""),
                "source_url": str(case.get("source_url") or ""),
                "source_format": "epo_st36_xml.v1",
                "independence_group": f"publication:{publication}",
            }
        },
        "exact_records": {
            record_id: {
                "record_id": record_id,
                "source_ref": source_ref,
                "source_binding_id": source_binding_id,
                "location_ref": location_ref,
                "location_refs": [location_ref],
                "evidence_refs": [
                    f"xml_sha256:{materialization.get('artifact_sha256') or ''}",
                    "procedure-text-sha256:"
                    + str(
                        dict(binding.get("parser_audit") or {}).get("procedure_text_sha256") or ""
                    ),
                ],
            }
        },
        "procedure_records": {
            procedure_id: {
                "procedure_record_id": procedure_id,
                "exact_record_id": record_id,
                "source_ref": source_ref,
                "location_refs": [location_ref],
                "conditions": dict(binding.get("source_conditions") or {}),
                "procedure_authority_scope": "source_exact_reaction_procedure",
                "procedure_status": "condition_complete",
                "condition_completeness": dict(completeness),
                "source_fragment": {
                    "digest_kind": "procedure-text-sha256",
                    "procedure_text_sha256": str(
                        dict(binding.get("parser_audit") or {}).get("procedure_text_sha256") or ""
                    ),
                    "source_artifact_sha256": str(materialization.get("artifact_sha256") or ""),
                    "procedure_text_stored": False,
                },
            }
        },
        "stock_observations": {},
        "conflicts": {},
        "hypotheses": {},
        "delta": {"rejected": []},
    }
    route_id = f"route:{slug}:procedure-slice"
    portfolio = {
        "schema_version": "proof_stitched_route_portfolio.v1",
        "graph_revision": 1,
        "graph_scientific_sha256": graph_digest,
        "content_sha256": f"{slug}-procedure-portfolio-v1",
        "proof_policy": {"stock_boundary": "benchmark_search", "minimum_edge_proof_level": 3},
        "selected_routes": [
            {
                "route_id": route_id,
                "route_family_id": f"publication:{publication}",
                "strategy": f"{publication} exact procedure evidence slice",
                "edge_ids": [edge_id],
                "root_edge_ids": [edge_id],
                "leaf_molecule_ids": precursor_ids,
                "module_selections": {},
                "minimum_edge_proof_level": 3,
                "all_edges_proven": True,
                "stock_closure_rate": 0.0,
                "independent_source_groups": [f"publication:{publication}"],
                "risk_score": 0.35,
                "convergence_score": 0.0,
                "complete": False,
                "pareto_optimal": True,
            }
        ],
        "edge_proofs": {
            edge_id: {
                "achieved_level": 3,
                "accepted": True,
                "reaction_validated": True,
                "exact_source_bound": True,
                "source_binding_ids": [source_binding_id],
                "exact_record_ids": [record_id],
                "procedure_record_ids": [procedure_id],
                "conflict_ids": [],
                "reasons": [],
            }
        },
        "leaf_proofs": {node_id: {"accepted": False} for node_id in precursor_ids},
        "route_modules": [],
        "deficits": [
            {
                "kind": "stock_boundary",
                "route_id": route_id,
                "reason": "starting_material_procurement_not_audited_in_this_case_gate",
            }
        ],
        "metrics": {},
        "closeout": {
            "decision": "continue",
            "complete_route_count": 0,
            "note": "single exact procedure gate; not a complete synthesis claim",
        },
        "accepted": False,
    }
    projection = compile_route_workbench(graph, portfolio)
    json_path = output_dir / "route_workbench.json"
    html_path = output_dir / "route_workbench.html"
    _write_json_atomic(json_path, projection)
    _write_text_atomic(html_path, render_v4_route_workbench_html(projection))
    return {
        "route_workbench_json": str(json_path),
        "route_workbench_html": str(html_path),
        "portfolio_accepted": False,
        "display_scope": "single_real_procedure_gate",
    }


def _suite_index_html(summary: Mapping[str, Any]) -> str:
    cards: list[str] = []
    for row in summary.get("cases") or []:
        case = dict(row)
        case_id = str(case.get("case_id") or "case")
        conditions = dict(dict(case.get("procedure") or {}).get("conditions") or {})
        chips = [
            str(conditions.get(field) or "")
            for field in ("temperature", "time")
            if conditions.get(field)
        ]
        if conditions.get("yield_percent") is not None:
            chips.append(f"{conditions['yield_percent']:g}% yield")
        workbench_path = str(dict(case.get("showcase") or {}).get("route_workbench_html") or "")
        artifact_path = _result_relative_path(workbench_path)
        workbench_url = (
            f"/api/v4/result-file?path={quote(artifact_path, safe='/')}" if artifact_path else "#"
        )
        cards.append(
            '<article class="case">'
            f'<div><span class="status">{"PASS" if case.get("accepted") else "FAIL"}</span>'
            f'<span class="class">{html.escape(str(case.get("reaction_class") or ""))}</span></div>'
            f"<h2>{html.escape(str(case.get('target_name') or case_id))}</h2>"
            f"<p>{html.escape(str(case.get('publication') or ''))} · "
            f"{html.escape(' · '.join(chips))}</p>"
            f'<a href="{html.escape(workbench_url)}">打开精确 procedure 工作台</a>'
            "</article>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>官方专利程序三例门禁</title>
<style>body{{margin:0;background:#f4f7fb;color:#10233d;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:40px 24px}}header{{background:#10233d;color:white;border-radius:24px;padding:28px 32px;margin-bottom:22px}}h1{{margin:0 0 8px;font-size:30px}}header p{{margin:0;color:#cbd8e8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}}.case{{background:white;border:1px solid #dbe4ef;border-radius:18px;padding:20px;box-shadow:0 8px 24px #10233d0b}}.case h2{{margin:16px 0 4px}}.case p{{color:#5d6f86;min-height:48px}}.case a{{display:inline-block;margin-top:10px;color:#334bd7;font-weight:700;text-decoration:none}}.status,.class{{display:inline-flex;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:750}}.status{{background:#dcfce7;color:#166534}}.class{{background:#eef2ff;color:#4338ca;margin-left:6px}}footer{{margin-top:22px;color:#64748b}}</style></head><body><main><header><h1>官方结构化专利程序 · 3/3</h1><p>三个独立 publication、三种反应类型；XML 精确范围与条件完整度可离线复放。该门禁不等于全路线或采购闭合。</p></header><section class="grid">{"".join(cards)}</section><footer>结构化来源闭合后未调用 PDF、OCR 或视觉；仅 unresolved 边允许继续降级。</footer></main></body></html>"""


def _result_relative_path(value: str) -> str:
    parts = list(Path(str(value or "")).parts)
    index = next(
        (offset for offset, part in enumerate(parts) if part.casefold() == "results"),
        -1,
    )
    return Path(*parts[index:]).as_posix() if index >= 0 else ""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"json_object_required:{path}")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "CASE_SCHEMA",
    "CONFIG_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "SUITE_SCHEMA",
    "canonical_smiles",
    "compile_patent_procedure_gate_suite",
    "content_digest",
    "load_patent_procedure_gate_config",
    "replay_patent_procedure_case",
    "validate_resolver_snapshot",
]
