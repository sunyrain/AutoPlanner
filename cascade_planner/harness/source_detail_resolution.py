"""Harness-owned source-detail resolution for literature extraction queues."""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SOURCE_DETAIL_RESOLUTION_PACK_SCHEMA = "source_detail_resolution_pack.v1"
SOURCE_DETAIL_CURATOR_RECORDS_SCHEMA = "source_detail_curator_records.v1"

FetchJson = Callable[[str, dict[str, str], float], dict[str, Any]]
FetchText = Callable[[str, dict[str, str], float], str]


def resolve_source_detail_extraction_pack(
    pack: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    timeout_s: float = 10.0,
    max_items: int = 5,
    curator_records: dict[str, Any] | str | Path | None = None,
    fetch_json: FetchJson | None = None,
    fetch_text: FetchText | None = None,
) -> dict[str, Any]:
    """Resolve a source-detail extraction queue into exact steps or explicit gaps.

    This resolver is intentionally conservative: it may probe DOI/PMID/PMC
    metadata and scan PMC XML for explicit product/reactant SMILES markers, but
    it never stores full text and never fabricates missing structures.
    """
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    cache_dir = evidence_dir / "source_detail_cache"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_pack(pack)
    queue = [dict(row) for row in payload.get("queue") or [] if isinstance(row, dict)]
    selected = queue[:max(0, int(max_items))]
    json_fetch = fetch_json or _fetch_json
    text_fetch = fetch_text or _fetch_text

    access_probes: list[dict[str, Any]] = []
    signal_audits: list[dict[str, Any]] = []
    source_detail_steps: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        resolution = _resolve_queue_item(
            item,
            index=index,
            cache_dir=cache_dir,
            timeout_s=float(timeout_s),
            fetch_json=json_fetch,
            fetch_text=text_fetch,
        )
        access_probes.extend(resolution["access_probes"])
        signal_audits.extend(resolution["signal_audits"])
        source_detail_steps.extend(resolution["source_detail_route_steps"])
        gaps.extend(resolution["extraction_gaps"])
    curator_result = _resolve_curator_records(
        _load_curator_records(curator_records, output_dir=out),
        pack_payload=payload,
    )
    source_detail_steps.extend(curator_result["source_detail_route_steps"])
    gaps.extend(curator_result["extraction_gaps"])

    result = {
        "schema_version": SOURCE_DETAIL_RESOLUTION_PACK_SCHEMA,
        "accepted": True,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": dict(payload.get("target") or {}),
        "source_pack_ref": str(pack) if isinstance(pack, (str, Path)) else "",
        "source_policy": {
            "harness_owned_source_detail_resolution": True,
            "do_not_fabricate_smiles": True,
            "explicit_smiles_markers_required": True,
            "structured_curator_records_allowed": True,
            "pmc_xml_signal_scan_only": True,
            "full_text_content_stored": False,
            "procedure_text_stored": False,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "queue_count": len(queue),
        "processed_queue_count": len(selected),
        "access_probes": access_probes,
        "signal_audits": signal_audits,
        "source_detail_route_steps": source_detail_steps,
        "extraction_gaps": gaps,
        "curator_record_audit": curator_result["curator_record_audit"],
        "downstream_patch": {
            "schema_version": "source_detail_resolution_downstream_patch.v1",
            "source_detail_route_steps": source_detail_steps,
            "rejected_consumables": [
                {
                    "reason": gap.get("reason") or "source_detail_not_resolved",
                    "queue_id": gap.get("queue_id") or "",
                    "source_ref": gap.get("source_ref") or "",
                    "next_action": gap.get("next_action") or "",
                    "source": "source_detail_resolution_pack",
                }
                for gap in gaps
            ],
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "summary": {
            "source_detail_route_step_count": len(source_detail_steps),
            "access_probe_count": len(access_probes),
            "signal_audit_count": len(signal_audits),
            "curator_record_count": curator_result["curator_record_count"],
            "curator_step_count": curator_result["curator_step_count"],
            "gap_count": len(gaps),
            "resolved_queue_count": len({
                str(step.get("queue_id") or "")
                for step in source_detail_steps
                if step.get("queue_id")
            }),
        },
    }
    source_detail_resolution_pack_path(out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def write_source_detail_resolution_error(
    *,
    output_dir: str | Path,
    pack: dict[str, Any] | None = None,
    error: Exception | str,
) -> dict[str, Any]:
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SOURCE_DETAIL_RESOLUTION_PACK_SCHEMA,
        "accepted": False,
        "status": "error",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": dict((pack or {}).get("target") or {}),
        "source_policy": {
            "harness_owned_source_detail_resolution": True,
            "do_not_fabricate_smiles": True,
            "full_text_content_stored": False,
            "procedure_text_stored": False,
            "no_solved_claim": True,
            "production_write_blocked": True,
            "error_is_nonblocking": True,
        },
        "queue_count": len((pack or {}).get("queue") or []),
        "processed_queue_count": 0,
        "access_probes": [],
        "signal_audits": [],
        "curator_record_audit": [],
        "source_detail_route_steps": [],
        "extraction_gaps": [],
        "downstream_patch": {
            "schema_version": "source_detail_resolution_downstream_patch.v1",
            "source_detail_route_steps": [],
            "rejected_consumables": [],
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "summary": {
            "source_detail_route_step_count": 0,
            "access_probe_count": 0,
            "signal_audit_count": 0,
            "curator_record_count": 0,
            "curator_step_count": 0,
            "gap_count": 0,
            "resolved_queue_count": 0,
        },
        "error": str(error),
        "reasons": ["source_detail_resolution_error"],
    }
    source_detail_resolution_pack_path(out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def source_detail_resolution_pack_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "source_detail_resolution_pack.json"


def source_detail_resolution_manifest_entry(
    resolution: dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    path = source_detail_resolution_pack_path(output_dir)
    payload = dict(resolution or {})
    return {
        "schema_version": "source_detail_resolution_manifest_entry.v1",
        "path": str(path.resolve()),
        "schema": SOURCE_DETAIL_RESOLUTION_PACK_SCHEMA,
        "status": str(payload.get("status") or ("available" if path.exists() else "planned")),
        "accepted": bool(payload.get("accepted", path.exists())),
        "summary": dict(payload.get("summary") or {}),
        "source_policy": dict(payload.get("source_policy") or {}),
        "use_policy": (
            "Read this source-detail resolution pack after the extraction pack. "
            "Copy source_detail_route_steps only when present; otherwise preserve "
            "extraction_gaps as unresolved source-detail tasks."
        ),
    }


def source_detail_curator_records_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "source_detail_curator_records.json"


def _resolve_queue_item(
    item: dict[str, Any],
    *,
    index: int,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
    fetch_text: FetchText,
) -> dict[str, list[dict[str, Any]]]:
    queue_id = str(item.get("queue_id") or _stable_id("queue", index, item.get("doi"), item.get("title"), item.get("query")))
    source_ref = _source_ref(item)
    if bool(item.get("metadata_only")):
        return _gap_result(
            item,
            queue_id=queue_id,
            source_ref=source_ref,
            reason="metadata_only_source_requires_followup",
            next_action="Use typed patent/web connector or curator review before extracting route structures.",
        )

    doi = str(item.get("doi") or "").strip()
    pmid = str(item.get("pmid") or "").strip()
    if not doi and not pmid:
        return _gap_result(
            item,
            queue_id=queue_id,
            source_ref=source_ref,
            reason="source_detail_requires_doi_or_pmid",
            next_action="Resolve this source to DOI/PMID/PMC before structure extraction.",
        )

    access_probe = _access_probe(
        item,
        queue_id=queue_id,
        doi=doi,
        pmid=pmid,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    probes = [access_probe]
    pmcids = [str(pmcid) for pmcid in access_probe.get("pmcids") or [] if str(pmcid).strip()]
    if not pmcids:
        reason = (
            "doi_not_linked_to_pubmed"
            if doi and not access_probe.get("pmid")
            else "no_open_pmc_xml_for_source"
        )
        return {
            "access_probes": probes,
            "signal_audits": [],
            "source_detail_route_steps": [],
            "extraction_gaps": [
                _gap(
                    item,
                    queue_id=queue_id,
                    source_ref=source_ref,
                    reason=reason,
                    next_action="Use DOI landing page, publisher SI, patent connector, or curator record for exact structures.",
                )
            ],
        }

    signal_audits: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for pmcid in pmcids[:2]:
        xml_text = _cached_text(
            source="pmc_xml",
            url=_pmc_efetch_url(pmcid),
            cache_dir=cache_dir,
            timeout_s=timeout_s,
            fetch_text=fetch_text,
        )
        audit = _pmc_xml_signal_audit(xml_text, pmcid=pmcid, queue_item=item, queue_id=queue_id)
        signal_audits.append(audit)
        steps.extend(_source_detail_steps_from_xml_text(xml_text, item=item, queue_id=queue_id, pmcid=pmcid))
    if not steps:
        gaps.append(
            _gap(
                item,
                queue_id=queue_id,
                source_ref=source_ref,
                reason="no_explicit_smiles_fields_detected",
                next_action="Keep as extraction task until source tables/SI or curator records expose product/reactant SMILES.",
            )
        )
    return {
        "access_probes": probes,
        "signal_audits": signal_audits,
        "source_detail_route_steps": steps,
        "extraction_gaps": gaps,
    }


def _access_probe(
    item: dict[str, Any],
    *,
    queue_id: str,
    doi: str,
    pmid: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> dict[str, Any]:
    resolved_pmid = pmid or (_pubmed_id_for_doi(doi, cache_dir=cache_dir, timeout_s=timeout_s, fetch_json=fetch_json) if doi else "")
    pmcids = _pubmed_pmc_links_for_pmid(
        resolved_pmid,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    ) if resolved_pmid else []
    status = "pmc_open_access_link_available" if pmcids else (
        "doi_or_pubmed_access_metadata_available" if doi or resolved_pmid else "source_metadata_only_no_access_link"
    )
    return {
        "schema_version": "source_detail_access_probe.v1",
        "queue_id": queue_id,
        "source_ref": _source_ref(item),
        "doi": doi,
        "pmid": resolved_pmid,
        "pmcids": pmcids,
        "pmc_urls": [f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" for pmcid in pmcids],
        "full_text_access_status": status,
        "backend_resolved": "ncbi_esearch_elink",
        "full_text_content_stored": False,
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _pubmed_id_for_doi(
    doi: str,
    *,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> str:
    clean = str(doi or "").strip()
    if not clean:
        return ""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode({
        "db": "pubmed",
        "term": f"{clean}[doi]",
        "retmode": "json",
        "retmax": 1,
        "tool": "AutoPlanner",
    })
    payload = _cached_json(source="pubmed_doi", url=url, cache_dir=cache_dir, timeout_s=timeout_s, fetch_json=fetch_json)
    ids = ((payload.get("esearchresult") or {}).get("idlist") or [])
    return str(ids[0]) if ids else ""


def _pubmed_pmc_links_for_pmid(
    pmid: str,
    *,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> list[str]:
    clean = str(pmid or "").strip()
    if not clean:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?" + urlencode({
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": clean,
        "retmode": "json",
        "tool": "AutoPlanner",
    })
    payload = _cached_json(source="pubmed_pmc", url=url, cache_dir=cache_dir, timeout_s=timeout_s, fetch_json=fetch_json)
    links: list[str] = []
    for linkset in payload.get("linksets") or []:
        if not isinstance(linkset, dict):
            continue
        for linkset_db in linkset.get("linksetdbs") or []:
            if not isinstance(linkset_db, dict) or linkset_db.get("linkname") != "pubmed_pmc":
                continue
            for link in linkset_db.get("links") or []:
                clean_link = str(link or "").removeprefix("PMC").strip()
                if clean_link and clean_link not in links:
                    links.append(clean_link)
    return links


def _pmc_efetch_url(pmcid: str) -> str:
    clean = str(pmcid or "").removeprefix("PMC").strip()
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode({
        "db": "pmc",
        "id": clean,
        "retmode": "xml",
        "tool": "AutoPlanner",
    })


def _pmc_xml_signal_audit(xml_text: str, *, pmcid: str, queue_item: dict[str, Any], queue_id: str) -> dict[str, Any]:
    text = _xml_visible_text(xml_text)
    lower = text.lower()
    route_terms = _route_signal_terms()
    field_terms = _structure_field_terms(queue_item)
    route_hits = sorted({term for term in route_terms if term in lower})
    field_hits = sorted({term for term in field_terms if term in lower})
    return {
        "schema_version": "source_detail_signal_audit.v1",
        "queue_id": queue_id,
        "source_ref": _source_ref(queue_item, pmcid=pmcid),
        "pmcid": str(pmcid).removeprefix("PMC"),
        "signal_status": (
            "explicit_structure_signal_detected"
            if field_hits and route_hits
            else "route_signal_without_structure_fields"
            if route_hits
            else "no_route_or_structure_signal"
        ),
        "route_signal_terms": route_hits,
        "structure_signal_terms": field_hits,
        "route_signal_count": len(route_hits),
        "structure_signal_count": len(field_hits),
        "source_xml_char_count": len(xml_text),
        "full_text_content_stored": False,
        "not_template_support": True,
    }


def _source_detail_steps_from_xml_text(
    xml_text: str,
    *,
    item: dict[str, Any],
    queue_id: str,
    pmcid: str,
) -> list[dict[str, Any]]:
    text = _xml_visible_text(xml_text)
    source_ref = _source_ref(item, pmcid=pmcid)
    segment_id = _safe_id(str(item.get("doi") or item.get("record_id") or queue_id))
    rows: list[dict[str, Any]] = []
    for idx, match in enumerate(_iter_explicit_smiles_matches(text), start=1):
        product = match["product_smiles"]
        reactants = _split_reactant_smiles(match["reactant_smiles"])
        if not _valid_smiles(product) or not reactants or not all(_valid_smiles(smi) for smi in reactants):
            continue
        step_id = f"{segment_id}_step_{idx}"
        evidence_refs = _dedupe([str(ref) for ref in item.get("evidence_refs") or [] if str(ref).strip()])
        if not evidence_refs:
            evidence_refs = [str(item.get("record_id") or item.get("doi") or queue_id)]
        rows.append(
            {
                "schema_version": "source_detail_route_step.v1",
                "step_id": step_id,
                "segment_id": f"source_detail_{segment_id}",
                "queue_id": queue_id,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "source_ref": source_ref,
                "evidence_refs": evidence_refs,
                "relation_type": "exact",
                "applicability": {
                    "status": "passed",
                    "product_reconstruction_passed": True,
                    "reconstructed_product_smiles": product,
                    "source_detail_resolution": "explicit_smiles_marker",
                },
                "condition_candidate": {
                    "schema_version": "condition_candidate.v1",
                    "step_id": step_id,
                    "source_type": "exact" if match.get("condition_fields") else "unknown",
                    "condition_status": "evidence_backed" if match.get("condition_fields") else "gap",
                    **dict(match.get("condition_fields") or {}),
                    "evidence_refs": evidence_refs,
                },
                "not_raw_reaction_injection": True,
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    return rows


def _resolve_curator_records(curator_payload: dict[str, Any], *, pack_payload: dict[str, Any]) -> dict[str, Any]:
    records = _curator_record_rows(curator_payload)
    target = dict(pack_payload.get("target") or {})
    queue_by_id = {
        str(item.get("queue_id") or ""): dict(item)
        for item in pack_payload.get("queue") or []
        if isinstance(item, dict) and str(item.get("queue_id") or "")
    }
    audit_rows: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        record_id = str(record.get("record_id") or record.get("curator_record_id") or f"curator_record_{record_index}")
        base_reasons = _curator_record_base_reasons(record)
        record_steps: list[dict[str, Any]] = []
        record_gaps: list[dict[str, Any]] = []
        raw_steps = record.get("steps") if isinstance(record.get("steps"), list) else [record]
        for step_index, raw_step in enumerate(raw_steps, start=1):
            step = dict(raw_step) if isinstance(raw_step, dict) else {}
            step_reasons = list(base_reasons)
            step_reasons.extend(_codex_translation_reasons(record, step))
            if _contains_raw_reaction(step) or _contains_raw_reaction(record):
                step_reasons.append("raw_reaction_in_curator_record")
            product = str(step.get("product_smiles") or record.get("product_smiles") or "").strip()
            reactants = _curator_reactant_smiles(step.get("reactant_smiles") or record.get("reactant_smiles"))
            if not product:
                step_reasons.append("missing_product_smiles")
            elif not _valid_smiles(product):
                step_reasons.append("invalid_product_smiles")
            if not reactants:
                step_reasons.append("missing_reactant_smiles")
            elif not all(_valid_smiles(smiles) for smiles in reactants):
                step_reasons.append("invalid_reactant_smiles")
            condition = _curator_condition_candidate(
                step.get("condition_candidate") or record.get("condition_candidate") or {},
                step_id=str(step.get("step_id") or f"{_safe_id(record_id)}_step_{step_index}"),
                evidence_refs=_dedupe([str(ref) for ref in (step.get("evidence_refs") or record.get("evidence_refs") or []) if str(ref).strip()]),
            )
            if not _condition_has_fields(condition):
                step_reasons.append("condition_candidate_missing_source_grounded_fields")
            source_ref = str(step.get("source_ref") or record.get("source_ref") or "").strip()
            evidence_refs = _dedupe([str(ref) for ref in (step.get("evidence_refs") or record.get("evidence_refs") or []) if str(ref).strip()])
            if not source_ref:
                step_reasons.append("missing_source_ref")
            if not evidence_refs:
                step_reasons.append("missing_evidence_refs")
            queue_id = str(step.get("queue_id") or record.get("queue_id") or "")
            queue_item = queue_by_id.get(queue_id, {})
            if step_reasons:
                record_gaps.append(
                    _gap(
                        {
                            **queue_item,
                            "queue_id": queue_id,
                            "source": str(record.get("source_type") or record.get("source") or "curator_record"),
                            "source_ref": source_ref,
                            "title": str(record.get("source_title") or record.get("title") or ""),
                            "evidence_refs": evidence_refs,
                        },
                        queue_id=queue_id or record_id,
                        source_ref=source_ref,
                        reason="curator_record_rejected:" + ",".join(sorted(set(step_reasons))),
                        next_action="Fix structured curator record fields before compiling source-detail route steps.",
                    )
                )
                continue
            step_id = str(step.get("step_id") or f"{_safe_id(record_id)}_step_{step_index}")
            segment_id = str(step.get("segment_id") or record.get("segment_id") or f"curator_{_safe_id(record_id)}")
            provenance = str(step.get("provenance") or record.get("provenance") or "")
            source_detail_resolution = (
                "codex_source_text_translation"
                if provenance == "codex_source_text_translation"
                else "structured_curator_record"
            )
            structure_derivation = step.get("structure_derivation") or record.get("structure_derivation")
            source_excerpt = str(step.get("source_excerpt") or record.get("source_excerpt") or "").strip()
            record_steps.append(
                {
                    "schema_version": "source_detail_route_step.v1",
                    "step_id": step_id,
                    "segment_id": segment_id,
                    "queue_id": queue_id,
                    "product_smiles": product,
                    "reactant_smiles": reactants,
                    "source_ref": source_ref,
                    "evidence_refs": evidence_refs,
                    "relation_type": str(step.get("relation_type") or record.get("relation_type") or "exact"),
                    "applicability": {
                        "status": "passed",
                        "product_reconstruction_passed": True,
                        "reconstructed_product_smiles": product,
                        "source_detail_resolution": source_detail_resolution,
                        "target_name": str(target.get("name") or ""),
                    },
                    "condition_candidate": condition,
                    "curator_record_id": record_id,
                    "provenance": provenance or str(record.get("source_extraction_method") or ""),
                    **({"structure_derivation": dict(structure_derivation)} if isinstance(structure_derivation, dict) else {}),
                    **({"source_excerpt": source_excerpt} if source_excerpt else {}),
                    "curation_status": str(
                        record.get("curation_status")
                        or (
                            "codex_source_text_translation_draft"
                            if provenance == "codex_source_text_translation"
                            else "structured_source_extraction"
                        )
                    ),
                    "validation_status": str(step.get("validation_status") or record.get("validation_status") or "draft"),
                    "not_raw_reaction_injection": True,
                    "no_solved_claim": True,
                    "production_write_blocked": True,
                }
            )
        steps.extend(record_steps)
        gaps.extend(record_gaps)
        audit_rows.append(
            {
                "schema_version": "source_detail_curator_record_audit.v1",
                "record_id": record_id,
                "accepted_step_count": len(record_steps),
                "rejected_step_count": len(record_gaps),
                "source_ref": str(record.get("source_ref") or ""),
                "source_title": str(record.get("source_title") or record.get("title") or ""),
                "full_text_content_stored": False,
                "procedure_text_stored": False,
                "not_template_support_until_validated": True,
            }
        )
    return {
        "curator_record_count": len(records),
        "curator_step_count": len(steps),
        "curator_record_audit": audit_rows,
        "source_detail_route_steps": steps,
        "extraction_gaps": gaps,
    }


def _load_curator_records(
    curator_records: dict[str, Any] | str | Path | None,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    if isinstance(curator_records, dict):
        return dict(curator_records)
    path = Path(curator_records) if curator_records else source_detail_curator_records_path(output_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "schema_version": SOURCE_DETAIL_CURATOR_RECORDS_SCHEMA,
            "records": [],
            "load_error": "invalid_json",
        }
    return dict(payload) if isinstance(payload, dict) else {}


def _curator_record_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows = payload.get("records")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    if isinstance(payload.get("steps"), list) or payload.get("product_smiles"):
        return [dict(payload)]
    return []


def _curator_record_base_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    schema = str(record.get("schema_version") or "")
    if schema and schema not in {
        "source_detail_curator_record.v1",
        "source_detail_route_step.v1",
        SOURCE_DETAIL_CURATOR_RECORDS_SCHEMA,
    }:
        reasons.append("invalid_curator_record_schema")
    if bool(record.get("full_text_content_stored")) or bool(record.get("procedure_text_stored")):
        reasons.append("curator_record_stores_full_text_or_procedure")
    if bool(record.get("production_kb_promotion")) or record.get("write_layer") == "production":
        reasons.append("curator_record_requests_production_write")
    provenance = str(record.get("provenance") or record.get("source_extraction_method") or "")
    if provenance and provenance not in {
        "curator_structured_from_source",
        "curator_structured_from_SI",
        "curator_structured_from_patent",
        "typed_connector_structured_extraction",
        "ocr_structured_table_extraction",
        "manual_structured_extraction",
        "codex_source_text_translation",
    }:
        reasons.append("unsupported_curator_record_provenance")
    return reasons


def _codex_translation_reasons(record: dict[str, Any], step: dict[str, Any]) -> list[str]:
    provenance = str(step.get("provenance") or record.get("provenance") or "")
    if provenance != "codex_source_text_translation":
        return []
    reasons: list[str] = []
    derivation = step.get("structure_derivation") or record.get("structure_derivation")
    if not isinstance(derivation, dict):
        reasons.append("codex_translation_missing_structure_derivation")
    else:
        basis = str(derivation.get("basis") or "").strip()
        if basis not in {
            "explicit_smiles",
            "source_name_to_smiles",
            "source_iupac_to_smiles",
            "source_structure_diagram_to_smiles",
            "source_compound_number_to_smiles",
            "source_table_to_smiles",
            "tool_assisted_source_text_translation",
            "codex_source_text_translation",
            "current_pdf_image_to_smiles",
            "current_image_to_smiles",
            "visual_pdf_image_to_smiles",
            "visual_structure_chain_to_smiles",
        }:
            reasons.append("codex_translation_invalid_structure_basis")
        source_locator = derivation.get("source_locator")
        if isinstance(source_locator, dict):
            locator_present = bool(
                str(source_locator.get("source_ref") or "").strip()
                or str(source_locator.get("url") or "").strip()
                or str(source_locator.get("source_title") or "").strip()
            )
        else:
            locator_present = bool(str(source_locator or "").strip())
        if not locator_present:
            reasons.append("codex_translation_missing_source_locator")
        confidence = str(derivation.get("confidence") or "").strip()
        confidence_prefix = confidence.split("_for_", 1)[0]
        if confidence not in {"high", "medium_high", "medium", "low"} and confidence_prefix not in {
            "high",
            "medium_high",
            "medium",
            "low",
        }:
            reasons.append("codex_translation_missing_confidence")
        tool_checks = derivation.get("tool_checks")
        if isinstance(tool_checks, dict):
            has_tool_checks = bool(tool_checks)
        else:
            has_tool_checks = isinstance(tool_checks, list) and any(str(item).strip() for item in tool_checks)
        if not has_tool_checks:
            reasons.append("codex_translation_missing_tool_checks")
    excerpt = str(step.get("source_excerpt") or record.get("source_excerpt") or "").strip()
    if not excerpt:
        reasons.append("codex_translation_missing_source_excerpt")
    if len(excerpt.split()) > 40:
        reasons.append("codex_translation_source_excerpt_too_long")
    if bool(step.get("full_source_text_stored")) or bool(record.get("full_source_text_stored")):
        reasons.append("codex_translation_stores_full_source_text")
    return reasons


def _curator_reactant_smiles(value: Any) -> list[str]:
    if isinstance(value, list):
        return _dedupe([_clean_smiles_token(str(item)) for item in value if _clean_smiles_token(str(item))])
    if isinstance(value, str):
        return _split_reactant_smiles(value)
    return []


def _curator_condition_candidate(data: Any, *, step_id: str, evidence_refs: list[str]) -> dict[str, Any]:
    row = dict(data) if isinstance(data, dict) else {}
    reagent_candidates = [str(item) for item in row.get("reagent_candidates") or [] if str(item)]
    solvent_candidates = [str(item) for item in row.get("solvent_candidates") or [] if str(item)]
    reagent = str(row.get("reagent") or row.get("reagent_or_method") or row.get("method") or "; ".join(reagent_candidates))
    duration = str(row.get("duration") or row.get("time") or row.get("duration_h") or row.get("duration_min") or row.get("hydrolysis_duration_min") or "")
    reported_yield = str(row.get("reported_yield") or row.get("yield") or "")
    condition = {
        "schema_version": "condition_candidate.v1",
        "step_id": str(row.get("step_id") or step_id),
        "source_type": str(row.get("source_type") or "exact"),
        "condition_status": str(row.get("condition_status") or "evidence_backed"),
        "reagent": reagent,
        "reagent_candidates": reagent_candidates,
        "catalyst": str(row.get("catalyst") or ""),
        "enzyme": str(row.get("enzyme") or ""),
        "solvent": str(row.get("solvent") or "; ".join(solvent_candidates)),
        "solvent_candidates": solvent_candidates,
        "temperature": str(row.get("temperature") or row.get("temperature_c") or row.get("temperature_C") or ""),
        "duration": duration,
        "isolation": str(row.get("isolation") or row.get("workup_or_isolation") or ""),
        "reported_yield": reported_yield,
        "reported_purity": str(row.get("reported_purity") or ""),
        "source_grounding": str(row.get("source_grounding") or ""),
        "ph": str(row.get("ph") or row.get("pH") or ""),
        "buffer": str(row.get("buffer") or ""),
        "atmosphere": str(row.get("atmosphere") or ""),
        "evidence_refs": _dedupe([str(ref) for ref in (row.get("evidence_refs") or evidence_refs) if str(ref).strip()]),
        "risk_flags": [str(flag) for flag in row.get("risk_flags") or []],
    }
    return {key: value for key, value in condition.items() if value not in ("", [])}


def _condition_has_fields(condition: dict[str, Any]) -> bool:
    return any(
        str(condition.get(key) or "").strip()
        for key in (
            "reagent",
            "catalyst",
            "enzyme",
            "solvent",
            "temperature",
            "duration",
            "isolation",
            "reported_yield",
            "ph",
            "buffer",
        )
    ) or bool(condition.get("reagent_candidates") or condition.get("solvent_candidates"))


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "raw_reactions"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


def _iter_explicit_smiles_matches(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?:step\s*(?P<step>[A-Za-z0-9_.-]+)[^\n\r]{0,80})?"
        r"product[_\s-]*smiles\s*[:=]\s*(?P<product>[A-Za-z0-9@+\-\[\]\(\)\\/#%=.$]+)"
        r"[^\n\r]{0,240}?"
        r"reactant[_\s-]*smiles\s*[:=]\s*(?P<reactants>[A-Za-z0-9@+\-\[\]\(\)\\/#%=.$,; ]+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        product = _clean_smiles_token(match.group("product"))
        reactants = _clean_reactant_field(match.group("reactants"))
        context = text[match.start(): min(len(text), match.end() + 320)]
        condition_fields = _condition_fields_from_context(context)
        if product and reactants:
            rows.append(
                {
                    "step": str(match.group("step") or ""),
                    "product_smiles": product,
                    "reactant_smiles": reactants,
                    "condition_signal": "condition" if condition_fields else "",
                    "condition_fields": condition_fields,
                }
            )
    return rows


def _condition_fields_from_context(context: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    value_pattern = r"([^.;,\n<]+?)(?=\s+(?:reagent|catalyst|enzyme|solvent|temperature|pH|buffer)\s*[:=]|[.;,\n<]|$)"
    patterns = {
        "reagent": rf"\breagent\s*[:=]\s*{value_pattern}",
        "catalyst": rf"\bcatalyst\s*[:=]\s*{value_pattern}",
        "enzyme": rf"\benzyme\s*[:=]\s*{value_pattern}",
        "solvent": rf"\bsolvent\s*[:=]\s*{value_pattern}",
        "temperature": rf"\btemperature\s*[:=]\s*{value_pattern}",
        "ph": rf"\bpH\s*[:=]\s*{value_pattern}",
        "buffer": rf"\bbuffer\s*[:=]\s*{value_pattern}",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            value = str(match.group(1) or "").strip()
            if value:
                fields[key] = value
    return fields


def _split_reactant_smiles(value: str) -> list[str]:
    text = _clean_reactant_field(value)
    chunks: list[str] = []
    for item in re.split(r"\s*(?:,|;|\+)\s*", text):
        for sub in item.split("."):
            clean = _clean_smiles_token(sub)
            if clean:
                chunks.append(clean)
    return _dedupe(chunks)


def _clean_reactant_field(value: str) -> str:
    text = str(value or "").strip()
    text = re.split(
        r"\s+(?:condition|conditions|solvent|yield|source|product[_\s-]*name)\s*[:=]?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text.strip(" .;,\t\r\n")


def _clean_smiles_token(value: str) -> str:
    return str(value or "").strip().strip(".,;")


def _valid_smiles(smiles: str) -> bool:
    text = str(smiles or "").strip()
    if not text:
        return False
    try:
        from rdkit import Chem
    except Exception:
        return bool(re.match(r"^[A-Za-z0-9@+\-\[\]\(\)\\/#%=.$]+$", text))
    return Chem.MolFromSmiles(text) is not None


def _xml_visible_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return str(xml_text or "")
    return "\n".join(str(text or "") for text in root.itertext())


def _route_signal_terms() -> set[str]:
    return {
        "synthesis",
        "synthetic",
        "preparation",
        "process",
        "intermediate",
        "side-chain",
        "side chain",
        "olefination",
        "reduction",
        "lactone",
        "statin",
    }


def _structure_field_terms(item: dict[str, Any]) -> set[str]:
    terms = {
        "smiles",
        "product smiles",
        "reactant smiles",
        "structure",
        "scheme",
        "compound",
        "intermediate",
    }
    for field in item.get("required_structured_fields") or []:
        text = str(field or "").lower().replace("_", " ")
        if text and "=" not in text:
            terms.add(text)
    return terms


def _gap_result(
    item: dict[str, Any],
    *,
    queue_id: str,
    source_ref: str,
    reason: str,
    next_action: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "access_probes": [],
        "signal_audits": [],
        "source_detail_route_steps": [],
        "extraction_gaps": [_gap(item, queue_id=queue_id, source_ref=source_ref, reason=reason, next_action=next_action)],
    }


def _gap(
    item: dict[str, Any],
    *,
    queue_id: str,
    source_ref: str,
    reason: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "schema_version": "source_detail_resolution_gap.v1",
        "queue_id": queue_id,
        "source": str(item.get("source") or ""),
        "source_ref": source_ref,
        "doi": str(item.get("doi") or ""),
        "title": str(item.get("title") or ""),
        "query": str(item.get("query") or ""),
        "extraction_task_ids": [str(task) for task in item.get("extraction_task_ids") or []],
        "evidence_refs": [str(ref) for ref in item.get("evidence_refs") or []],
        "reason": reason,
        "next_action": next_action,
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


def _source_ref(item: dict[str, Any], *, pmcid: str = "") -> str:
    if pmcid:
        return f"pmc:{str(pmcid).removeprefix('PMC')}"
    doi = str(item.get("doi") or "").strip()
    if doi:
        return f"doi:{doi.lower()}"
    pmid = str(item.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    return str(item.get("record_id") or item.get("url") or item.get("query") or "")


def _load_pack(pack: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(pack, dict):
        return dict(pack)
    path = Path(pack)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _cached_json(
    *,
    source: str,
    url: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> dict[str, Any]:
    cache = cache_dir / f"{source}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:20]}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    payload = fetch_json(url, {"Accept": "application/json", "User-Agent": "AutoPlanner/1.0"}, timeout_s)
    cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _cached_text(
    *,
    source: str,
    url: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_text: FetchText,
) -> str:
    cache = cache_dir / f"{source}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:20]}.xml"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    text = fetch_text(url, {"Accept": "application/xml", "User-Agent": "AutoPlanner/1.0"}, timeout_s)
    cache.write_text(text, encoding="utf-8")
    return text


def _fetch_json(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str, headers: dict[str, str], timeout_s: float) -> str:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8", "replace")


def _stable_id(*parts: Any) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").lower()).strip("_")
    return text[:80] or "source_detail"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
