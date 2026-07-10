"""Harness-owned typed retrieval prefetch for open research runs."""
from __future__ import annotations

import hashlib
import json
import os
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


OPEN_RESEARCH_RETRIEVAL_PREFETCH_SCHEMA = "open_research_retrieval_prefetch.v1"
SOURCE_DETAIL_EXTRACTION_PACK_SCHEMA = "source_detail_extraction_pack.v1"

FetchJson = Callable[[str, dict[str, str], float], dict[str, Any]]


def prefetch_open_research_evidence(
    manifest: dict[str, Any],
    *,
    output_dir: str | Path,
    timeout_s: float = 6.0,
    max_results: int = 5,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Run bounded source connectors and write a typed evidence seed file.

    The open Codex agent reads this artifact. It should not implement its own
    HTTP clients for PubChem, CrossRef, PubMed, patent, DOI, or web sources.
    """
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    cache_dir = evidence_dir / "retrieval_cache"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fetch = fetch_json or _fetch_json
    query_plan = dict(manifest.get("query_plan") or {})
    target = dict(manifest.get("target") or {})
    prioritize_lookup_requests = bool(query_plan.get("prioritize_self_evo_lookup_requests"))
    crossref_base_limit = 3 if prioritize_lookup_requests else 6
    patent_base_limit = 1 if prioritize_lookup_requests else 3
    web_base_limit = 1 if prioritize_lookup_requests else 3
    lookup_source_budget = dict(query_plan.get("self_evo_lookup_request_budget") or {})
    rejected_queries: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {
        "pubchem": [],
        "crossref": [],
        "pubmed": [],
        "patent_metadata": [],
        "web_search_metadata": [],
    }

    for query in _bounded_queries(query_plan.get("pubchem_name_queries"), limit=8):
        if not _query_allowed(query, source="pubchem", rejected=rejected_queries):
            continue
        records["pubchem"].extend(
            _lookup_pubchem(
                query,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
                max_results=max_results,
                fetch_json=fetch,
                rejected=rejected_queries,
            )
        )

    for query in _bounded_queries(query_plan.get("crossref_queries"), limit=crossref_base_limit):
        if not _query_allowed(query, source="crossref", rejected=rejected_queries):
            continue
        records["crossref"].extend(
            _lookup_crossref(
                query,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
                max_results=max_results,
                fetch_json=fetch,
                rejected=rejected_queries,
            )
        )

    for request in _bounded_lookup_requests(
        query_plan.get("lookup_requests"),
        source="crossref",
        limit=int(lookup_source_budget.get("crossref") or 6),
    ):
        query = str(request.get("query") or "")
        if not _query_allowed(query, source="crossref", rejected=rejected_queries):
            continue
        records["crossref"].extend(
            _with_lookup_request_context(
                _lookup_crossref(
                    query,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                    max_results=max_results,
                    fetch_json=fetch,
                    rejected=rejected_queries,
                ),
                request,
            )
        )

    for query in _bounded_queries(query_plan.get("pubmed_terms"), limit=2):
        if not _query_allowed(query, source="pubmed", rejected=rejected_queries):
            continue
        records["pubmed"].extend(
            _lookup_pubmed(
                query,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
                max_results=max_results,
                fetch_json=fetch,
                rejected=rejected_queries,
                quality_flags=quality_flags,
            )
        )

    for query in _bounded_queries(query_plan.get("patent_metadata_queries"), limit=patent_base_limit):
        if not _query_allowed(query, source="patent_metadata", rejected=rejected_queries):
            continue
        records["patent_metadata"].append(_metadata_record(source="patent_metadata", query=query))
    for request in _bounded_lookup_requests(
        query_plan.get("lookup_requests"),
        source="patent_metadata",
        limit=int(lookup_source_budget.get("patent_metadata") or 6),
    ):
        query = str(request.get("query") or "")
        if not _query_allowed(query, source="patent_metadata", rejected=rejected_queries):
            continue
        records["patent_metadata"].append(
            _metadata_record(
                source="patent_metadata",
                query=query,
                request=request,
            )
        )

    web_queries = _bounded_queries(query_plan.get("route_failure_feedback_queries"), limit=6)
    web_queries.extend(_bounded_queries(query_plan.get("live_web_search_gap_queries"), limit=web_base_limit))
    for query in _dedupe(web_queries):
        if not _query_allowed(query, source="web_search_metadata", rejected=rejected_queries):
            continue
        records["web_search_metadata"].append(_metadata_record(source="web_search_metadata", query=query))
    for request in _bounded_lookup_requests(
        query_plan.get("lookup_requests"),
        source="web_search_metadata",
        limit=int(lookup_source_budget.get("web_search_metadata") or 6),
    ):
        query = str(request.get("query") or "")
        if not _query_allowed(query, source="web_search_metadata", rejected=rejected_queries):
            continue
        records["web_search_metadata"].append(
            _metadata_record(
                source="web_search_metadata",
                query=query,
                request=request,
            )
        )

    all_records = [row for rows in records.values() for row in rows]
    source_triage = _triage_source_records(all_records, target=target)
    structured_extraction_queue = _structured_extraction_queue(source_triage)
    extraction_pack = _source_detail_extraction_pack(
        target=target,
        source_triage=source_triage,
        queue=structured_extraction_queue,
        query_plan=query_plan,
    )
    prefetch = {
        "schema_version": OPEN_RESEARCH_RETRIEVAL_PREFETCH_SCHEMA,
        "accepted": True,
        "status": "completed",
        "retrieval_time_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "name": str(target.get("name") or ""),
            "smiles": str(target.get("smiles") or ""),
            "frontier_smiles": str(target.get("frontier_smiles") or ""),
            "family_hint": str(target.get("family_hint") or ""),
        },
        "source_policy": {
            "harness_owned_http": True,
            "open_agent_raw_http_forbidden": True,
            "patent_and_web_are_metadata_only": True,
            "cache_dir": str(cache_dir.resolve()),
            "timeout_s": float(timeout_s),
            "max_results_per_query": int(max_results),
        },
        "records": records,
        "source_triage": source_triage,
        "structured_extraction_queue": structured_extraction_queue,
        "source_seed_rows": [
            row
            for source in ("crossref", "pubmed", "patent_metadata", "web_search_metadata")
            for row in records[source]
        ],
        "compound_seed_rows": list(records["pubchem"]),
        "record_counts": {source: len(rows) for source, rows in records.items()},
        "rejected_queries": rejected_queries,
        "query_quality_flags": quality_flags,
        "triage_counts": _triage_counts(source_triage),
        "source_detail_extraction_pack": {
            "schema_version": SOURCE_DETAIL_EXTRACTION_PACK_SCHEMA,
            "path": str(source_detail_extraction_pack_path(out).resolve()),
            "queue_count": len(extraction_pack.get("queue") or []),
            "top_source_count": len(extraction_pack.get("top_sources") or []),
        },
        "all_record_count": len(all_records),
    }
    path = retrieval_prefetch_path(out)
    path.write_text(json.dumps(prefetch, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    source_detail_extraction_pack_path(out).write_text(
        json.dumps(extraction_pack, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prefetch


def retrieval_prefetch_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "harness_retrieval_prefetch.json"


def source_detail_extraction_pack_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "source_detail_extraction_pack.json"


def write_prefetch_checkpoint_seed(
    *,
    output_dir: str | Path,
    manifest: dict[str, Any],
    prefetch: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a conservative minimum checkpoint from harness retrieval seeds.

    This is not a literature verdict. It prevents a timed-out open Codex run
    from leaving an empty artifact directory, while keeping all downstream
    consumables empty and explicitly marked as harness-seeded.
    """
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not prefetch.get("accepted") or int(prefetch.get("all_record_count") or 0) <= 0:
        return {
            "schema_version": "open_research_prefetch_checkpoint_seed.v1",
            "accepted": False,
            "status": "not_written",
            "reasons": ["retrieval_prefetch_unavailable_or_empty"],
            "artifact_refs": {},
        }

    target = dict(manifest.get("target") or prefetch.get("target") or {})
    case_id = _case_id(target)
    source_seed_rows = [row for row in prefetch.get("source_seed_rows") or [] if isinstance(row, dict)]
    compound_seed_rows = [row for row in prefetch.get("compound_seed_rows") or [] if isinstance(row, dict)]
    quality_flags = [row for row in prefetch.get("query_quality_flags") or [] if isinstance(row, dict)]
    rejected_queries = [row for row in prefetch.get("rejected_queries") or [] if isinstance(row, dict)]
    source_triage = [row for row in prefetch.get("source_triage") or [] if isinstance(row, dict)]
    extraction_queue = [row for row in prefetch.get("structured_extraction_queue") or [] if isinstance(row, dict)]
    extraction_pack = dict(prefetch.get("source_detail_extraction_pack") or {})
    compact_source_rows, compact_excluded_rows = _compact_prefetch_literature_sources(
        source_seed_rows=source_seed_rows,
        source_triage=source_triage,
    )
    compact_search_seed_rows = _dedupe_source_seed_rows(source_seed_rows)

    literature_sources = {
        "schema_version": "open_literature_sources.v1",
        "case_id": case_id,
        "source_relation_policy": {
            "generated_by": "harness_prefetch_checkpoint_seed",
            "source": "harness_retrieval_prefetch",
            "classification_status": "metadata_seed_only",
            "not_route_evidence": True,
            "sources_compacted": True,
            "source_compaction_policy": (
                "Only deduped source-triage rows needing structured extraction/review are placed in sources; "
                "unrelated/query-only metadata is excluded or summarized in search_log."
            ),
        },
        "sources": compact_source_rows,
        "excluded_sources": [
            {
                "source": "harness_retrieval_prefetch",
                "source_name": row.get("source"),
                "query": row.get("query"),
                "reason": row.get("reason") or "prefetch_rejected_query",
                "raw_ref": row.get("raw_ref") or "",
            }
            for row in rejected_queries
        ] + compact_excluded_rows,
        "search_log": [
            *[_source_seed_to_search_log(row) for row in compact_search_seed_rows[:40]],
            *[_triage_to_search_log(row) for row in source_triage[:20]],
            *[
                {
                    "source": "harness_retrieval_prefetch",
                    "query": row.get("query"),
                    "status": "quality_flag",
                    "reason": row.get("reason"),
                    "count": row.get("count"),
                    "action": row.get("action"),
                }
                for row in quality_flags
            ],
        ],
    }
    compounds = {
        "schema_version": "open_pubchem_validated_compounds.v1",
        "case_id": case_id,
        "compound_source_policy": {
            "generated_by": "harness_prefetch_checkpoint_seed",
            "source": "harness_retrieval_prefetch",
            "validation_status": "typed_pubchem_identity_seed",
            "agent_rdkit_revalidation_required": True,
        },
        "compounds": [_compound_seed_to_validated_compound(row) for row in compound_seed_rows],
        "rejected_items": [],
    }
    source_refs = [
        str(row.get("record_id") or row.get("doi") or row.get("pmid") or row.get("query") or "")
        for row in source_seed_rows
    ]
    source_refs = _dedupe([ref for ref in source_refs if ref])
    candidate_payload = {
        "schema_version": "open_structure_template_candidates.v1",
        "case_id": case_id,
        "target": {
            "name": str(target.get("name") or case_id),
            "smiles": str(target.get("smiles") or ""),
            "frontier_smiles": str(target.get("frontier_smiles") or ""),
            "family_hint": str(target.get("family_hint") or ""),
        },
        "candidate_generation_policy": {
            "generated_by": "harness_prefetch_checkpoint_seed",
            "status": "seed_only",
            "codex_enrichment_expected": True,
            "not_solved": True,
            "production_kb_promotion": False,
        },
        "candidates": [],
        "rejected_items": [
            {
                "reason": "no_codex_literature_template_generated_yet",
                "source": "harness_prefetch_checkpoint_seed",
            }
        ],
        "source_refs": source_refs,
        "audit_summary": {
            "final_status": "checkpoint_seed",
            "solved": False,
            "production_kb_promotion": False,
            "prefetch_record_counts": dict(prefetch.get("record_counts") or {}),
            "triage_counts": dict(prefetch.get("triage_counts") or {}),
            "structured_extraction_queue_count": len(extraction_queue),
            "source_detail_extraction_pack": extraction_pack,
            "limitations": [
                "Harness prefetch metadata was checkpointed before Codex enrichment.",
                "No route/template candidate has been extracted from literature in this seed.",
            ],
        },
    }
    downstream = {
        "schema_version": "open_downstream_consumables.v1",
        "case_id": case_id,
        "planner_handoff": {
            "next_action": "no_consumable_found",
            "solved": False,
            "production_kb_promotion": False,
            "reason": "harness prefetch checkpoint seed only; Codex did not yet emit route/template consumables",
            "generated_by": "harness_prefetch_checkpoint_seed",
        },
        "guided_rerun_requests": [],
        "literature_template_cards": [],
        "literature_route_segments": [],
        "executable_template_candidates": [],
        "route_expansion_tasks": [],
        "evolution_candidates": [],
        "rejected_consumables": [
            {
                "reason": "seed_checkpoint_has_no_literature_consumable",
                "source": "harness_prefetch_checkpoint_seed",
            }
        ],
    }
    audit = {
        "schema_version": "open_structure_agent_audit.v1",
        "case_id": case_id,
        "final_status": "checkpoint_seed",
        "solved": False,
        "production_kb_promotion": False,
        "checks": [
            {
                "check": "retrieval_prefetch_seed_checkpoint_written",
                "accepted": True,
                "source": "harness_retrieval_prefetch",
                "record_counts": dict(prefetch.get("record_counts") or {}),
                "triage_counts": dict(prefetch.get("triage_counts") or {}),
                "structured_extraction_queue_count": len(extraction_queue),
                "source_detail_extraction_pack": extraction_pack,
            }
        ],
        "limitations": [
            "This artifact set was written by the harness before Codex enrichment.",
            "It consumes typed retrieval seeds but does not claim extracted chemistry.",
            "Downstream consumables remain empty until Codex or a deterministic extractor emits source-grounded drafts.",
        ],
        "next_actions": [
            "Continue open literature extraction for exact stuck-node/intermediate route steps.",
            "Prioritize structured_extraction_queue items from harness retrieval prefetch before broad searches.",
            "Emit guided ChemEnzy or route segment drafts only after source-grounded structure validation.",
        ],
    }
    report = "\n".join(
        [
            f"# {case_id} Open Research Checkpoint",
            "",
            "Status: harness prefetch checkpoint seed only.",
            "",
            "This checkpoint consumes typed retrieval prefetch rows so a timed-out Codex run does not leave an empty artifact set. It does not claim a solved route, a production KB update, or an executable literature template.",
            "",
            f"Compound seed rows: {len(compound_seed_rows)}",
            f"Source seed rows: {len(source_seed_rows)}",
            f"Structured extraction queue rows: {len(extraction_queue)}",
            "",
        ]
    )
    smi_lines = [
        f"{str(row.get('canonical_smiles') or row.get('isomeric_smiles')).strip()} {str(row.get('record_id') or row.get('query') or 'pubchem_seed').strip().replace(' ', '_')}"
        for row in compound_seed_rows
        if str(row.get("canonical_smiles") or row.get("isomeric_smiles")).strip()
    ]

    refs = {
        "structure_template_report.md": out / "structure_template_report.md",
        "structure_template_candidates.json": out / "structure_template_candidates.json",
        "downstream_consumables.json": out / "downstream_consumables.json",
        "evidence/literature_sources.json": evidence_dir / "literature_sources.json",
        "evidence/pubchem_validated_compounds.json": evidence_dir / "pubchem_validated_compounds.json",
        "validated_compounds.smi": out / "validated_compounds.smi",
        "open_agent_audit.json": out / "open_agent_audit.json",
    }
    _write_text_if_allowed(refs["structure_template_report.md"], report, overwrite=overwrite)
    _write_text_if_allowed(refs["validated_compounds.smi"], "\n".join(smi_lines) + ("\n" if smi_lines else ""), overwrite=overwrite)
    for rel, payload in (
        ("structure_template_candidates.json", candidate_payload),
        ("downstream_consumables.json", downstream),
        ("evidence/literature_sources.json", literature_sources),
        ("evidence/pubchem_validated_compounds.json", compounds),
        ("open_agent_audit.json", audit),
    ):
        _write_json_if_allowed(refs[rel], payload, overwrite=overwrite)
    return {
        "schema_version": "open_research_prefetch_checkpoint_seed.v1",
        "accepted": True,
        "status": "written",
        "generated_by": "harness_prefetch_checkpoint_seed",
        "compound_seed_count": len(compound_seed_rows),
        "source_seed_count": len(source_seed_rows),
        "structured_extraction_queue_count": len(extraction_queue),
        "source_detail_extraction_pack": extraction_pack,
        "artifact_refs": {rel: str(path) for rel, path in refs.items()},
        "downstream_status": "seed_only_no_consumable",
    }


def write_retrieval_prefetch_error(
    *,
    output_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    error: Exception | str,
) -> dict[str, Any]:
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = dict((manifest or {}).get("target") or {})
    payload = {
        "schema_version": OPEN_RESEARCH_RETRIEVAL_PREFETCH_SCHEMA,
        "accepted": False,
        "status": "error",
        "retrieval_time_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "name": str(target.get("name") or ""),
            "smiles": str(target.get("smiles") or ""),
            "frontier_smiles": str(target.get("frontier_smiles") or ""),
            "family_hint": str(target.get("family_hint") or ""),
        },
        "source_policy": {
            "harness_owned_http": True,
            "open_agent_raw_http_forbidden": True,
            "patent_and_web_are_metadata_only": True,
            "error_is_nonblocking": True,
        },
        "records": {
            "pubchem": [],
            "crossref": [],
            "pubmed": [],
            "patent_metadata": [],
            "web_search_metadata": [],
        },
        "source_seed_rows": [],
        "compound_seed_rows": [],
        "record_counts": {
            "pubchem": 0,
            "crossref": 0,
            "pubmed": 0,
            "patent_metadata": 0,
            "web_search_metadata": 0,
        },
        "rejected_queries": [],
        "query_quality_flags": [],
        "triage_counts": {},
        "source_detail_extraction_pack": {
            "schema_version": SOURCE_DETAIL_EXTRACTION_PACK_SCHEMA,
            "path": str(source_detail_extraction_pack_path(out).resolve()),
            "queue_count": 0,
            "top_source_count": 0,
        },
        "all_record_count": 0,
        "error": str(error),
        "reasons": ["retrieval_prefetch_error"],
    }
    path = retrieval_prefetch_path(out)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    source_detail_extraction_pack_path(out).write_text(
        json.dumps(
            _source_detail_extraction_pack(
                target=target,
                source_triage=[],
                queue=[],
                query_plan={},
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return payload


def retrieval_prefetch_manifest_entry(
    prefetch: dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    path = retrieval_prefetch_path(output_dir)
    payload = dict(prefetch or {})
    return {
        "schema_version": "open_research_retrieval_prefetch_manifest_entry.v1",
        "path": str(path.resolve()),
        "schema": OPEN_RESEARCH_RETRIEVAL_PREFETCH_SCHEMA,
        "source_detail_extraction_pack_path": str(source_detail_extraction_pack_path(output_dir).resolve()),
        "source_detail_extraction_pack_schema": SOURCE_DETAIL_EXTRACTION_PACK_SCHEMA,
        "status": str(payload.get("status") or ("available" if path.exists() else "planned")),
        "accepted": bool(payload.get("accepted", path.exists())),
        "record_counts": dict(payload.get("record_counts") or {}),
        "triage_counts": dict(payload.get("triage_counts") or {}),
        "structured_extraction_queue_count": len(payload.get("structured_extraction_queue") or []),
        "source_detail_extraction_pack": dict(payload.get("source_detail_extraction_pack") or {}),
        "query_quality_flags": list(payload.get("query_quality_flags") or []),
        "use_policy": (
            "Read this typed harness-owned retrieval seed before issuing new lookup requests; "
            "prioritize structured_extraction_queue/source_triage rows before broad search; "
            "do not repeat these queries with shell HTTP clients."
        ),
    }


def validate_retrieval_prefetch_consumption(*, run_dir: str | Path) -> dict[str, Any]:
    """Check that open-agent outputs consumed or explicitly handled prefetch seeds."""
    root = Path(run_dir)
    path = retrieval_prefetch_path(root)
    if not path.exists():
        return {
            "schema_version": "open_research_retrieval_prefetch_consumption.v1",
            "accepted": True,
            "status": "not_applicable",
            "reasons": [],
        }
    prefetch = _load_json(path)
    if not prefetch.get("accepted") or int(prefetch.get("all_record_count") or 0) <= 0:
        return {
            "schema_version": "open_research_retrieval_prefetch_consumption.v1",
            "accepted": True,
            "status": "prefetch_unavailable_or_empty",
            "reasons": [],
            "prefetch_status": prefetch.get("status"),
            "all_record_count": int(prefetch.get("all_record_count") or 0),
        }

    literature = _load_json(root / "evidence" / "literature_sources.json")
    compounds = _load_json(root / "evidence" / "pubchem_validated_compounds.json")
    literature_text = _string_corpus(literature)
    compound_text = _string_corpus(compounds)
    combined_text = "\n".join([literature_text, compound_text]).lower()
    explicit_handling = any(
        marker in combined_text
        for marker in (
            "harness_retrieval_prefetch",
            "retrieval_prefetch",
            "open_research_retrieval_prefetch.v1",
        )
    )

    source_tokens = _seed_tokens(prefetch.get("source_seed_rows") or [])
    compound_tokens = _seed_tokens(prefetch.get("compound_seed_rows") or [])
    source_hits = _matched_tokens(source_tokens, literature_text)
    compound_hits = _matched_tokens(compound_tokens, compound_text)
    source_ok = not source_tokens or bool(source_hits) or explicit_handling
    compound_ok = not compound_tokens or bool(compound_hits) or explicit_handling
    reasons: list[str] = []
    if not source_ok:
        reasons.append("retrieval_prefetch_source_seed_not_consumed_or_explained")
    if not compound_ok:
        reasons.append("retrieval_prefetch_compound_seed_not_consumed_or_explained")
    return {
        "schema_version": "open_research_retrieval_prefetch_consumption.v1",
        "accepted": not reasons,
        "status": "checked",
        "reasons": reasons,
        "prefetch_status": prefetch.get("status"),
        "source_seed_count": len(prefetch.get("source_seed_rows") or []),
        "compound_seed_count": len(prefetch.get("compound_seed_rows") or []),
        "source_matched_tokens": source_hits[:20],
        "compound_matched_tokens": compound_hits[:20],
        "explicit_handling": explicit_handling,
    }


def _lookup_pubchem(
    query: str,
    *,
    cache_dir: Path,
    timeout_s: float,
    max_results: int,
    fetch_json: FetchJson,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cid_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON".format(quote(query, safe=""))
    cid_payload, cid_cache, cid_error = _cached_json(
        source="pubchem",
        url=cid_url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if cid_error:
        rejected.append(_rejected_query("pubchem", query, cid_error, raw_ref=cid_cache))
        return []
    cids = [str(cid) for cid in ((cid_payload.get("IdentifierList") or {}).get("CID") or [])][:max_results]
    if not cids:
        rejected.append(_rejected_query("pubchem", query, "no_pubchem_cid", raw_ref=cid_cache))
        return []
    properties = "SMILES,ConnectivitySMILES,IUPACName,MolecularFormula,InChIKey"
    prop_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/JSON"
        .format(",".join(cids), properties)
    )
    prop_payload, prop_cache, prop_error = _cached_json(
        source="pubchem",
        url=prop_url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if prop_error:
        rejected.append(_rejected_query("pubchem", query, prop_error, raw_ref=prop_cache))
        return []
    out: list[dict[str, Any]] = []
    for item in (prop_payload.get("PropertyTable") or {}).get("Properties") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("CID") or "")
        out.append(
            {
                "schema_version": "typed_retrieval_record.v1",
                "source": "pubchem",
                "query": query,
                "intent": "exact_target_or_named_intermediate_identity",
                "expected_relation": "exact_target_or_exact_intermediate",
                "record_id": f"CID:{cid}" if cid else "",
                "title": str(item.get("IUPACName") or query),
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else "",
                "canonical_smiles": str(
                    item.get("CanonicalSMILES")
                    or item.get("ConnectivitySMILES")
                    or item.get("SMILES")
                    or ""
                ),
                "isomeric_smiles": str(item.get("IsomericSMILES") or item.get("SMILES") or ""),
                "formula": str(item.get("MolecularFormula") or ""),
                "inchi_key": str(item.get("InChIKey") or ""),
                "evidence_level": "identity_metadata",
                "raw_ref": prop_cache,
            }
        )
    return out


def _lookup_crossref(
    query: str,
    *,
    cache_dir: Path,
    timeout_s: float,
    max_results: int,
    fetch_json: FetchJson,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    params = urlencode({"query.bibliographic": query, "rows": int(max_results), "select": "DOI,title,container-title,published-print,published-online,URL,score,type"})
    url = f"https://api.crossref.org/works?{params}"
    payload, cache_path, error = _cached_json(
        source="crossref",
        url=url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if error:
        rejected.append(_rejected_query("crossref", query, error, raw_ref=cache_path))
        return [_crossref_metadata_placeholder_record(query, raw_ref=cache_path, reason=error)]
    items = (payload.get("message") or {}).get("items") or []
    out: list[dict[str, Any]] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        doi = str(item.get("DOI") or "")
        title = _first_text(item.get("title")) or doi or query
        out.append(
            {
                "schema_version": "typed_retrieval_record.v1",
                "source": "crossref",
                "query": query,
                "intent": "synthesis_or_manufacturing_metadata",
                "expected_relation": "exact_target_or_method_reference",
                "record_id": f"doi:{doi.lower()}" if doi else "",
                "doi": doi,
                "title": title,
                "container_title": _first_text(item.get("container-title")),
                "published": _date_parts(item),
                "url": str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
                "score": item.get("score"),
                "evidence_level": "bibliographic_metadata",
                "raw_ref": cache_path,
            }
        )
    if not out:
        rejected.append(_rejected_query("crossref", query, "no_crossref_records", raw_ref=cache_path))
        return [_crossref_metadata_placeholder_record(query, raw_ref=cache_path, reason="no_crossref_records")]
    return out


def _crossref_metadata_placeholder_record(query: str, *, raw_ref: str = "", reason: str = "") -> dict[str, Any]:
    return {
        "schema_version": "typed_retrieval_record.v1",
        "source": "crossref",
        "query": query,
        "intent": "crossref_query_placeholder_after_connector_gap",
        "expected_relation": "unknown_until_doi_or_source_validated",
        "record_id": _stable_id("crossref_placeholder", query),
        "doi": "",
        "title": query,
        "container_title": "",
        "published": "",
        "url": "https://search.crossref.org/?" + urlencode({"q": query}),
        "score": None,
        "status": "metadata_placeholder_requires_followup",
        "evidence_level": "query_placeholder_not_source_evidence",
        "gap_reason": reason,
        "raw_ref": raw_ref,
        "not_route_evidence_until_extracted": True,
    }


def _lookup_pubmed(
    query: str,
    *,
    cache_dir: Path,
    timeout_s: float,
    max_results: int,
    fetch_json: FetchJson,
    rejected: list[dict[str, Any]],
    quality_flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    search_params = urlencode({"db": "pubmed", "term": query, "retmode": "json", "retmax": int(max_results)})
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"
    search_payload, search_cache, search_error = _cached_json(
        source="pubmed",
        url=search_url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if search_error:
        rejected.append(_rejected_query("pubmed", query, search_error, raw_ref=search_cache))
        return []
    result = search_payload.get("esearchresult") or {}
    count = _int(result.get("count"))
    if count > 200:
        quality_flags.append(
            {
                "schema_version": "retrieval_query_quality_flag.v1",
                "source": "pubmed",
                "query": query,
                "reason": "too_broad_pubmed_query",
                "count": count,
                "action": "require_refinement_before_evidence_use",
            }
        )
        rejected.append(_rejected_query("pubmed", query, "too_broad_pubmed_query", raw_ref=search_cache))
        return []
    ids = [str(uid) for uid in result.get("idlist") or []][:max_results]
    if not ids:
        rejected.append(_rejected_query("pubmed", query, "no_pubmed_ids", raw_ref=search_cache))
        return []
    summary_params = urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_params}"
    summary_payload, summary_cache, summary_error = _cached_json(
        source="pubmed",
        url=summary_url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if summary_error:
        rejected.append(_rejected_query("pubmed", query, summary_error, raw_ref=summary_cache))
        return []
    summary = summary_payload.get("result") or {}
    out: list[dict[str, Any]] = []
    for uid in summary.get("uids") or ids:
        item = summary.get(str(uid)) or {}
        if not isinstance(item, dict):
            continue
        pmid = str(item.get("uid") or uid)
        out.append(
            {
                "schema_version": "typed_retrieval_record.v1",
                "source": "pubmed",
                "query": query,
                "intent": "route_title_gap_check",
                "expected_relation": "exact_target_or_method_reference",
                "record_id": f"pmid:{pmid}",
                "pmid": pmid,
                "title": str(item.get("title") or f"PubMed PMID {pmid}"),
                "journal": str(item.get("fulljournalname") or item.get("source") or ""),
                "published": str(item.get("pubdate") or ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "evidence_level": "title_summary_lead",
                "raw_ref": summary_cache,
            }
        )
    return out


def _metadata_record(
    *,
    source: str,
    query: str,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = dict(request or {})
    if source == "patent_metadata":
        url = "https://patents.google.com/?q=" + quote(query, safe="")
        intent = "patent_metadata_url_only"
    else:
        url = "https://www.google.com/search?" + urlencode({"q": query})
        intent = "web_search_query_seed"
    row = {
        "schema_version": "typed_retrieval_record.v1",
        "source": source,
        "query": query,
        "intent": str(request.get("intent") or intent),
        "expected_relation": str(request.get("expected_relation") or "exact_target_or_exact_intermediate"),
        "record_id": _stable_id(source, query),
        "title": query,
        "url": url,
        "status": "metadata_url_only",
        "evidence_level": "search_metadata_not_source_evidence",
        "raw_ref": "",
    }
    if request:
        row.update(_lookup_request_trace(request))
    return row


def _compact_prefetch_literature_sources(
    *,
    source_seed_rows: list[dict[str, Any]],
    source_triage: list[dict[str, Any]],
    max_sources: int = 12,
    max_excluded: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_by_key = {_source_seed_key(row): row for row in source_seed_rows}
    sources: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for triage in source_triage:
        key = _source_seed_key(triage)
        if not key or key in seen:
            continue
        seed = dict(seed_by_key.get(key) or triage)
        seed.update({
            "priority_score": triage.get("priority_score"),
            "source_relation_hint": triage.get("source_relation_hint"),
            "recommended_action": triage.get("recommended_action"),
            "triage_reasons": list(triage.get("triage_reasons") or []),
        })
        action = str(triage.get("recommended_action") or "")
        score = int(triage.get("priority_score") or 0)
        if action != "log_only" and score >= 4 and len(sources) < max_sources:
            sources.append(_source_seed_to_literature_source(seed))
        elif len(excluded) < max_excluded:
            excluded.append(_source_seed_to_excluded_source(seed, reason="prefetch_seed_not_selected_for_sources"))
        seen.add(key)
    for row in source_seed_rows:
        key = _source_seed_key(row)
        if not key or key in seen:
            continue
        if len(excluded) < max_excluded:
            excluded.append(_source_seed_to_excluded_source(row, reason="prefetch_seed_deduped_or_low_priority"))
        seen.add(key)
    return sources, excluded


def _dedupe_source_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _source_seed_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _source_seed_key(row: dict[str, Any]) -> str:
    for field in ("doi", "pmid", "record_id", "url"):
        value = str(row.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    title = str(row.get("title") or row.get("query") or "").strip().lower()
    return f"title:{title}" if title else ""


def _source_seed_to_excluded_source(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "source": "harness_retrieval_prefetch",
        "source_name": row.get("source"),
        "record_id": row.get("record_id") or "",
        "doi": row.get("doi") or "",
        "pmid": row.get("pmid") or "",
        "title": row.get("title") or row.get("query") or "",
        "query": row.get("query") or "",
        "url": row.get("url") or "",
        "source_relation": row.get("source_relation_hint") or "unclassified_prefetch_seed",
        "priority_score": row.get("priority_score"),
        "recommended_action": row.get("recommended_action"),
        "reason": reason,
        "raw_ref": row.get("raw_ref") or "",
    }


def _source_seed_to_literature_source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "harness_retrieval_prefetch",
        "source_name": row.get("source"),
        "record_id": row.get("record_id") or "",
        "doi": row.get("doi") or "",
        "pmid": row.get("pmid") or "",
        "title": row.get("title") or row.get("query") or "",
        "query": row.get("query") or "",
        "url": row.get("url") or "",
        "raw_ref": row.get("raw_ref") or "",
        "source_relation": row.get("source_relation_hint") or "unclassified_prefetch_seed",
        "priority_score": row.get("priority_score"),
        "recommended_action": row.get("recommended_action"),
        "triage_reasons": list(row.get("triage_reasons") or []),
        "evidence_level": row.get("evidence_level") or "metadata_seed",
        "consumption_status": "checkpointed_for_codex_review",
    }


def _triage_source_records(records: list[dict[str, Any]], *, target: dict[str, Any]) -> list[dict[str, Any]]:
    target_aliases = _target_aliases(target)
    triaged: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        if str(row.get("source") or "") == "pubchem":
            continue
        item = _triage_source_record(row, target_aliases=target_aliases, index=idx)
        triaged.append(item)
    triaged.sort(key=lambda item: (-int(item.get("priority_score") or 0), str(item.get("record_id") or "")))
    return triaged


def _triage_source_record(row: dict[str, Any], *, target_aliases: list[str], index: int) -> dict[str, Any]:
    source = str(row.get("source") or "")
    title = str(row.get("title") or row.get("query") or "")
    query = str(row.get("query") or "")
    title_text = title.lower()
    query_text = query.lower()
    text = f"{title_text} {query_text}"
    reasons: list[str] = []
    score = 0
    if _contains_target_alias(title_text, target_aliases):
        score += 6
        reasons.append("target_name_in_source_title")
    elif _contains_target_alias(query_text, target_aliases):
        score += 1
        reasons.append("target_name_in_query_only")
    if any(term in title_text for term in ("intermediate", "aldehyde", "side-chain", "side chain", "precursor")):
        score += 3
        reasons.append("intermediate_or_precursor_terms_in_title")
    elif any(term in query_text for term in ("intermediate", "aldehyde", "side-chain", "side chain", "precursor")):
        score += 1
        reasons.append("intermediate_or_precursor_terms_in_query_only")
    if any(term in title_text for term in _route_or_process_terms()):
        score += 2
        reasons.append("route_or_process_terms_in_title")
    elif any(term in query_text for term in _route_or_process_terms()):
        score += 1
        reasons.append("route_or_process_terms_in_query_only")
    if str(row.get("lookup_request_origin") or "") == "self_evo_executable_template_extraction_task":
        score += 2
        reasons.append("self_evo_extraction_task_trace")
    if source == "crossref" and row.get("doi"):
        score += 1
        reasons.append("doi_metadata_available")
    if _target_alias_as_starting_material_direction(title_text, target_aliases):
        score -= 5
        reasons.append("target_name_as_starting_material_direction")
    if source in {"patent_metadata", "web_search_metadata"}:
        reasons.append("metadata_only_source")
    if _looks_biomedical_noise(text):
        score -= 4
        reasons.append("possible_biomedical_or_non_route_noise")
    metadata_only = source in {"patent_metadata", "web_search_metadata"}
    relation = _source_relation_from_score(score=score, reasons=reasons, source=source)
    if metadata_only and relation == "exact_target_or_exact_intermediate_candidate":
        relation = "metadata_lead"
    reverse_direction = "target_name_as_starting_material_direction" in reasons
    action = "log_only" if reverse_direction else (
        "extract_structured_route_step" if (
            not metadata_only and score >= 9 and "target_name_in_source_title" in reasons
        ) else (
            "review_metadata_for_exact_intermediate" if score >= 4 else "log_only"
        )
    )
    required_fields = [str(item) for item in row.get("required_structured_fields") or []]
    if not required_fields:
        required_fields = [
            "source_relation",
            "exact product/intermediate name",
            "product_smiles if source-grounded",
            "reactant_smiles if source-grounded",
            "condition_candidate if source-grounded",
        ]
    return {
        "schema_version": "retrieval_source_triage.v1",
        "triage_id": _stable_id("triage", index, source, row.get("record_id"), title, query),
        "source": source,
        "record_id": str(row.get("record_id") or ""),
        "doi": str(row.get("doi") or ""),
        "pmid": str(row.get("pmid") or ""),
        "title": title,
        "query": query,
        "url": str(row.get("url") or ""),
        "priority_score": score,
        "source_relation_hint": relation,
        "recommended_action": action,
        "triage_reasons": _dedupe(reasons),
        "lookup_request_id": str(row.get("lookup_request_id") or ""),
        "lookup_request_origin": str(row.get("lookup_request_origin") or ""),
        "extraction_task_ids": [str(item) for item in row.get("extraction_task_ids") or []],
        "evidence_refs": [str(item) for item in row.get("evidence_refs") or []],
        "required_structured_fields": required_fields,
        "not_route_evidence_until_extracted": True,
        "metadata_only": metadata_only,
        "raw_ref": str(row.get("raw_ref") or ""),
    }


def _target_aliases(target: dict[str, Any]) -> list[str]:
    values = [
        str(target.get("search_name") or ""),
        str(target.get("name") or ""),
    ]
    family_hint = str(target.get("family_hint") or "")
    values.extend(token for token in family_hint.replace(",", " ").split() if token)
    aliases: list[str] = []
    for value in values:
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in value).strip()
        if not clean:
            continue
        pieces = [piece for piece in clean.split() if len(piece) > 2]
        candidates = [clean, *pieces]
        if "_" in value:
            prefix = value.split("_", 1)[0].strip().lower()
            if prefix and len(prefix) > 2:
                candidates.append(prefix)
        for candidate in candidates:
            if candidate and candidate not in aliases:
                aliases.append(candidate)
    return aliases


def _contains_target_alias(text: str, aliases: list[str]) -> bool:
    padded = f" {''.join(ch.lower() if ch.isalnum() else ' ' for ch in text)} "
    for alias in aliases:
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in alias).strip()
        if clean and f" {clean} " in padded:
            return True
    return False


def _target_alias_as_starting_material_direction(title_text: str, aliases: list[str]) -> bool:
    padded = f" {''.join(ch.lower() if ch.isalnum() else ' ' for ch in title_text)} "
    for alias in aliases:
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in alias).strip()
        if clean and f" from {clean} " in padded and f" to {clean} " not in padded:
            return True
    return False


def _route_or_process_terms() -> tuple[str, ...]:
    return (
        "synthesis",
        "synthetic",
        "route",
        "routes",
        "process",
        "manufactur",
        "preparation",
        "olefination",
        "reduction",
    )


def _structured_extraction_queue(triaged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in triaged:
        if str(item.get("recommended_action") or "") not in {
            "extract_structured_route_step",
            "review_metadata_for_exact_intermediate",
        }:
            continue
        key = _source_queue_key(item)
        if key in seen:
            continue
        seen.add(key)
        queue.append(
            {
                "schema_version": "structured_extraction_queue_item.v1",
                "queue_id": _stable_id("queue", item.get("triage_id"), item.get("record_id"), item.get("query")),
                "triage_id": item.get("triage_id"),
                "source": item.get("source"),
                "record_id": item.get("record_id"),
                "doi": item.get("doi"),
                "pmid": item.get("pmid"),
                "title": item.get("title"),
                "query": item.get("query"),
                "url": item.get("url"),
                "priority_score": item.get("priority_score"),
                "source_relation_hint": item.get("source_relation_hint"),
                "extraction_task_ids": list(item.get("extraction_task_ids") or []),
                "evidence_refs": list(item.get("evidence_refs") or []),
                "required_structured_fields": list(item.get("required_structured_fields") or []),
                "required_output": "LiteratureRouteSegmentCard or executable_template_extraction_task_update",
                "action": item.get("recommended_action"),
                "metadata_only": bool(item.get("metadata_only")),
                "no_solved_claim": True,
                "production_write_blocked": True,
            }
        )
    return queue[:12]


def _source_queue_key(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "")
    for field in ("doi", "pmid", "record_id"):
        value = str(item.get(field) or "").strip().lower()
        if value:
            return f"{source}:{field}:{value}"
    return f"{source}:triage:{item.get('triage_id') or item.get('query') or item.get('title')}"


def _source_detail_extraction_pack(
    *,
    target: dict[str, Any],
    source_triage: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    query_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_DETAIL_EXTRACTION_PACK_SCHEMA,
        "target": {
            "name": str(target.get("name") or ""),
            "smiles": str(target.get("smiles") or ""),
            "frontier_smiles": str(target.get("frontier_smiles") or ""),
            "family_hint": str(target.get("family_hint") or ""),
        },
        "budget_mode": str(query_plan.get("budget_mode") or ""),
        "source_policy": {
            "harness_owned_http": True,
            "metadata_only_sources_require_followup": True,
            "do_not_fabricate_smiles": True,
            "no_raw_reaction_injection": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "required_output_policy": {
            "preferred_exact_output": "source_detail_route_steps",
            "fallback_output": "executable_template_extraction_tasks",
            "source_detail_route_step_required_fields": [
                "schema_version=source_detail_route_step.v1",
                "step_id",
                "segment_id",
                "product_smiles",
                "reactant_smiles",
                "source_ref",
                "evidence_refs",
                "relation_type=exact",
                "applicability.product_reconstruction_passed",
                "condition_candidate",
            ],
        },
        "queue": queue[:10],
        "top_sources": [
            _source_triage_summary(row)
            for row in source_triage[:12]
        ],
        "triage_counts": _triage_counts(source_triage),
    }


def _source_triage_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "triage_id": row.get("triage_id"),
        "source": row.get("source"),
        "record_id": row.get("record_id"),
        "doi": row.get("doi"),
        "title": row.get("title"),
        "query": row.get("query"),
        "url": row.get("url"),
        "priority_score": row.get("priority_score"),
        "source_relation_hint": row.get("source_relation_hint"),
        "recommended_action": row.get("recommended_action"),
        "metadata_only": bool(row.get("metadata_only")),
        "extraction_task_ids": list(row.get("extraction_task_ids") or []),
        "evidence_refs": list(row.get("evidence_refs") or []),
        "required_structured_fields": list(row.get("required_structured_fields") or []),
        "not_route_evidence_until_extracted": True,
    }


def _source_relation_from_score(*, score: int, reasons: list[str], source: str) -> str:
    reason_set = set(reasons)
    if score >= 9 and "target_name_in_source_title" in reason_set:
        return "exact_target_or_exact_intermediate_candidate"
    if score >= 5 and "target_name_in_query_only" not in reason_set:
        return "close_analog_or_method_candidate" if source == "crossref" else "metadata_lead"
    if score >= 4:
        return "query_matched_review_required"
    if "possible_biomedical_or_non_route_noise" in reason_set:
        return "unusable_or_biomedical_noise"
    return "unclassified_metadata"


def _looks_biomedical_noise(text: str) -> bool:
    biomedical_terms = (
        "clinical",
        "patient",
        "serum",
        "cholesterol",
        "therapy",
        "pharmacokinetic",
        "metabolism",
        "toxicity",
        "plasma",
    )
    route_terms = ("synthesis", "process", "intermediate", "preparation", "manufactur")
    return any(term in text for term in biomedical_terms) and not any(term in text for term in route_terms)


def _triage_counts(triage: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in triage:
        key = str(item.get("recommended_action") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_seed_to_search_log(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "harness_retrieval_prefetch",
        "source_name": row.get("source"),
        "query": row.get("query") or row.get("title") or "",
        "record_id": row.get("record_id") or "",
        "doi": row.get("doi") or "",
        "pmid": row.get("pmid") or "",
        "url": row.get("url") or "",
        "title": row.get("title") or "",
        "status": row.get("status") or "prefetch_seed_available",
        "intent": row.get("intent") or "",
        "expected_relation": row.get("expected_relation") or "",
        "raw_ref": row.get("raw_ref") or "",
    }


def _triage_to_search_log(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "harness_retrieval_prefetch",
        "source_name": row.get("source"),
        "query": row.get("query") or row.get("title") or "",
        "record_id": row.get("record_id") or "",
        "status": "source_triage",
        "priority_score": row.get("priority_score"),
        "source_relation_hint": row.get("source_relation_hint"),
        "recommended_action": row.get("recommended_action"),
        "triage_reasons": list(row.get("triage_reasons") or []),
        "extraction_task_ids": list(row.get("extraction_task_ids") or []),
        "required_structured_fields": list(row.get("required_structured_fields") or []),
    }


def _compound_seed_to_validated_compound(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "harness_retrieval_prefetch",
        "source_name": row.get("source") or "pubchem",
        "query": row.get("query") or "",
        "record_id": row.get("record_id") or "",
        "title": row.get("title") or row.get("query") or "",
        "canonical_smiles": row.get("canonical_smiles") or "",
        "isomeric_smiles": row.get("isomeric_smiles") or "",
        "formula": row.get("formula") or "",
        "inchi_key": row.get("inchi_key") or "",
        "url": row.get("url") or "",
        "raw_ref": row.get("raw_ref") or "",
        "source_relation": row.get("expected_relation") or "exact_target_or_exact_intermediate",
        "validation_status": "typed_pubchem_identity_seed_agent_revalidation_required",
    }


def _cached_json(
    *,
    source: str,
    url: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> tuple[dict[str, Any], str, str]:
    source_dir = cache_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / (_stable_id(source, url) + ".json")
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload.get("status") == "ok" and isinstance(payload.get("payload"), dict):
            return dict(payload["payload"]), str(path), ""
    try:
        data = fetch_json(url, _headers(source), float(timeout_s))
        payload = {
            "schema_version": "typed_retrieval_cache_entry.v1",
            "status": "ok",
            "source": source,
            "url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": data,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return dict(data), str(path), ""
    except Exception as exc:  # connector failure should not block the open agent
        payload = {
            "schema_version": "typed_retrieval_cache_entry.v1",
            "status": "error",
            "source": source,
            "url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return {}, str(path), "connector_error"


def _fetch_json(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=float(timeout_s)) as response:  # nosec B310 - bounded trusted connector
            data = response.read(2_000_000)
    except Exception as exc:
        if not _ssl_certificate_retry_allowed(exc):
            raise
        context = ssl._create_unverified_context()  # nosec B323 - fallback only after certificate-chain verification failure.
        with urlopen(request, timeout=float(timeout_s), context=context) as response:  # nosec B310
            data = response.read(2_000_000)
    payload = json.loads(data.decode("utf-8", errors="replace"))
    return dict(payload) if isinstance(payload, dict) else {}


def _ssl_certificate_retry_allowed(exc: Exception) -> bool:
    if str(os.environ.get("AUTOPLANNER_ALLOW_UNVERIFIED_RETRIEVAL_SSL", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return "certificate_verify_failed" in text or "self-signed certificate" in text


def _headers(source: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": f"AutoPlanner-open-research-prefetch/1.0 ({source}; mailto:autoplanner@example.invalid)",
    }


def _query_allowed(query: str, *, source: str, rejected: list[dict[str, Any]]) -> bool:
    if len(str(query or "").strip()) < 3:
        rejected.append(_rejected_query(source, str(query or ""), "empty_or_too_short_query"))
        return False
    return True


def _with_lookup_request_context(rows: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, Any]]:
    trace = _lookup_request_trace(request)
    if not trace:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(trace)
        item["intent"] = str(request.get("intent") or item.get("intent") or "")
        item["expected_relation"] = str(request.get("expected_relation") or item.get("expected_relation") or "")
        out.append(item)
    return out


def _lookup_request_trace(request: dict[str, Any]) -> dict[str, Any]:
    if not request:
        return {}
    return {
        "lookup_request_id": str(request.get("request_id") or ""),
        "lookup_request_origin": str(request.get("origin") or ""),
        "lookup_request_reason": str(request.get("reason") or ""),
        "extraction_task_ids": [str(item) for item in request.get("extraction_task_ids") or []],
        "evidence_refs": [str(item) for item in request.get("evidence_refs") or []],
        "required_structured_fields": [str(item) for item in request.get("required_structured_fields") or []],
    }


def _rejected_query(source: str, query: str, reason: str, *, raw_ref: str = "") -> dict[str, Any]:
    return {
        "schema_version": "typed_retrieval_rejected_query.v1",
        "source": source,
        "query": str(query or ""),
        "reason": reason,
        "raw_ref": raw_ref,
    }


def _bounded_queries(values: Any, *, limit: int) -> list[str]:
    return _dedupe([str(item).strip() for item in values or [] if str(item).strip()])[:limit]


def _bounded_lookup_requests(values: Any, *, source: str, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source") or "") != source:
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        row = dict(item)
        row["query"] = query
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _stable_id(*parts: Any) -> str:
    data = "\n".join(str(part) for part in parts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _case_id(target: dict[str, Any]) -> str:
    value = str(target.get("name") or "target").strip().lower()
    value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return value or "target"


def _write_text_if_allowed(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json_if_allowed(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def _date_parts(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and isinstance(parts[0], list):
            return "-".join(str(part) for part in parts[0])
    return ""


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _string_corpus(value: Any) -> str:
    parts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for sub in item.values():
                visit(sub)
        elif isinstance(item, list):
            for sub in item:
                visit(sub)
        elif item is not None:
            parts.append(str(item))

    visit(value)
    return "\n".join(parts)


def _seed_tokens(rows: list[Any]) -> list[str]:
    tokens: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "")
        for key in (
            "record_id",
            "doi",
            "pmid",
            "url",
            "raw_ref",
            "inchi_key",
            "canonical_smiles",
            "isomeric_smiles",
        ):
            value = str(row.get(key) or "").strip()
            if value:
                tokens.append(value)
        if source != "pubchem":
            query = str(row.get("query") or "").strip()
            if query:
                tokens.append(query)
        if source == "pubchem" and row.get("record_id"):
            cid = str(row["record_id"]).replace("CID:", "").strip()
            if cid:
                tokens.append(cid)
    return _dedupe([token for token in tokens if len(token) >= 3])


def _matched_tokens(tokens: list[str], text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in tokens if token.lower() in lowered]
