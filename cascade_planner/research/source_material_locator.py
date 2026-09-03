"""Harness-owned DOI/source material metadata locator.

This module records where source-detail extraction should look next. It stores
metadata and URLs only, never full text, supplementary files, or procedures.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCE_MATERIAL_LOCATOR_PACK_SCHEMA = "source_material_locator_pack.v1"

FetchJson = Callable[[str, dict[str, str], float], dict[str, Any]]


def locate_source_materials(
    extraction_pack: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    timeout_s: float = 10.0,
    max_items: int = 8,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Resolve queued DOI sources to metadata-only material candidates."""
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    cache_dir = evidence_dir / "source_material_cache"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pack = _load_pack(extraction_pack)
    queue = [dict(item) for item in pack.get("queue") or [] if isinstance(item, dict)]
    fetch = fetch_json or _fetch_json
    records: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in queue:
        doi = str(item.get("doi") or "").strip().lower()
        if not doi or doi in seen:
            continue
        seen.add(doi)
        if len(seen) > max(0, int(max_items)):
            break
        located = _locate_doi_materials(
            item,
            doi=doi,
            cache_dir=cache_dir,
            timeout_s=float(timeout_s),
            fetch_json=fetch,
        )
        records.extend(located["records"])
        gaps.extend(located["gaps"])
    payload = {
        "schema_version": SOURCE_MATERIAL_LOCATOR_PACK_SCHEMA,
        "accepted": True,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": dict(pack.get("target") or {}),
        "source_pack_ref": str(extraction_pack) if isinstance(extraction_pack, (str, Path)) else "",
        "source_policy": {
            "harness_owned_source_material_location": True,
            "metadata_only": True,
            "full_text_content_stored": False,
            "supplementary_file_content_stored": False,
            "procedure_text_stored": False,
            "not_route_evidence_until_structured_extraction": True,
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "processed_doi_count": len(seen),
        "material_records": records,
        "material_gaps": gaps,
        "downstream_hints": {
            "schema_version": "source_material_downstream_hints.v1",
            "structured_curator_record_output": "evidence/source_detail_curator_records.json",
            "required_curator_record_schema": "source_detail_curator_records.v1",
            "next_actions": _next_actions(records, gaps),
        },
        "summary": {
            "material_record_count": len(records),
            "material_gap_count": len(gaps),
            "supplementary_candidate_count": sum(1 for row in records if row.get("material_type") == "supplementary"),
            "publisher_landing_count": sum(1 for row in records if row.get("material_type") == "publisher_landing"),
        },
    }
    source_material_locator_pack_path(out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_source_material_locator_error(
    *,
    output_dir: str | Path,
    extraction_pack: dict[str, Any] | None = None,
    error: Exception | str,
) -> dict[str, Any]:
    out = Path(output_dir)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SOURCE_MATERIAL_LOCATOR_PACK_SCHEMA,
        "accepted": False,
        "status": "error",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": dict((extraction_pack or {}).get("target") or {}),
        "source_policy": {
            "harness_owned_source_material_location": True,
            "metadata_only": True,
            "full_text_content_stored": False,
            "supplementary_file_content_stored": False,
            "procedure_text_stored": False,
            "no_solved_claim": True,
            "production_write_blocked": True,
            "error_is_nonblocking": True,
        },
        "processed_doi_count": 0,
        "material_records": [],
        "material_gaps": [],
        "downstream_hints": {
            "schema_version": "source_material_downstream_hints.v1",
            "structured_curator_record_output": "evidence/source_detail_curator_records.json",
            "required_curator_record_schema": "source_detail_curator_records.v1",
            "next_actions": [],
        },
        "summary": {
            "material_record_count": 0,
            "material_gap_count": 0,
            "supplementary_candidate_count": 0,
            "publisher_landing_count": 0,
        },
        "error": str(error),
        "reasons": ["source_material_locator_error"],
    }
    source_material_locator_pack_path(out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def source_material_locator_pack_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "evidence" / "source_material_locator_pack.json"


def source_material_locator_manifest_entry(
    payload: dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    path = source_material_locator_pack_path(output_dir)
    data = dict(payload or {})
    return {
        "schema_version": "source_material_locator_manifest_entry.v1",
        "path": str(path.resolve()),
        "schema": SOURCE_MATERIAL_LOCATOR_PACK_SCHEMA,
        "status": str(data.get("status") or ("available" if path.exists() else "planned")),
        "accepted": bool(data.get("accepted", path.exists())),
        "summary": dict(data.get("summary") or {}),
        "source_policy": dict(data.get("source_policy") or {}),
        "use_policy": (
            "Read after source_detail_resolution when DOI/PMC resolution has gaps. "
            "Use material_records as metadata-only pointers for SI/patent/curator extraction; "
            "do not treat URLs as route evidence until source_detail_curator_records are produced."
        ),
    }


def _locate_doi_materials(
    item: dict[str, Any],
    *,
    doi: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> dict[str, list[dict[str, Any]]]:
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    payload, cache_path, error = _cached_json(
        source="crossref_work",
        url=url,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        fetch_json=fetch_json,
    )
    if error:
        return {
            "records": [],
            "gaps": [_gap(item, doi=doi, reason=error, next_action="Retry DOI material metadata locator or use curator-provided source URL.")],
        }
    message = dict(payload.get("message") or {})
    title = _first_text(message.get("title")) or str(item.get("title") or "")
    records: list[dict[str, Any]] = []
    landing_url = str(message.get("URL") or item.get("url") or (f"https://doi.org/{doi}" if doi else ""))
    if landing_url:
        records.append(_material_record(item, doi=doi, title=title, material_type="publisher_landing", url=landing_url, raw_ref=cache_path))
    for link in message.get("link") or []:
        if not isinstance(link, dict):
            continue
        link_url = str(link.get("URL") or "").strip()
        if not link_url:
            continue
        records.append(
            _material_record(
                item,
                doi=doi,
                title=title,
                material_type=_material_type_from_link(link),
                url=link_url,
                content_type=str(link.get("content-type") or ""),
                content_version=str(link.get("content-version") or ""),
                intended_application=str(link.get("intended-application") or ""),
                raw_ref=cache_path,
            )
        )
    relation = message.get("relation") if isinstance(message.get("relation"), dict) else {}
    for relation_type, relation_items in relation.items():
        for rel_item in relation_items if isinstance(relation_items, list) else []:
            if not isinstance(rel_item, dict):
                continue
            rel_id = str(rel_item.get("id") or "").strip()
            if not rel_id:
                continue
            records.append(
                _material_record(
                    item,
                    doi=doi,
                    title=title,
                    material_type="related_material",
                    url=rel_id if rel_id.startswith(("http://", "https://")) else f"https://doi.org/{rel_id}",
                    relation_type=str(relation_type),
                    raw_ref=cache_path,
                )
            )
    records = _dedupe_records(records)
    gaps = []
    if not records:
        gaps.append(_gap(item, doi=doi, reason="no_crossref_material_metadata", next_action="Use publisher landing page, patent connector, or curator source record."))
    elif not any(row.get("material_type") == "supplementary" for row in records):
        gaps.append(_gap(item, doi=doi, reason="no_supplementary_material_link_in_crossref", next_action="Use publisher landing page or patent connector to locate SI before structure extraction."))
    return {"records": records, "gaps": gaps}


def _material_record(
    item: dict[str, Any],
    *,
    doi: str,
    title: str,
    material_type: str,
    url: str,
    raw_ref: str,
    content_type: str = "",
    content_version: str = "",
    intended_application: str = "",
    relation_type: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "source_material_record.v1",
        "record_id": _stable_id("material", doi, material_type, url),
        "source": "crossref_work_metadata",
        "source_ref": f"doi:{doi}",
        "doi": doi,
        "title": title,
        "queue_id": str(item.get("queue_id") or ""),
        "evidence_refs": [str(ref) for ref in item.get("evidence_refs") or []],
        "material_type": material_type,
        "url": url,
        "content_type": content_type,
        "content_version": content_version,
        "intended_application": intended_application,
        "relation_type": relation_type,
        "raw_ref": raw_ref,
        "metadata_only": True,
        "full_text_content_stored": False,
        "supplementary_file_content_stored": False,
        "procedure_text_stored": False,
        "not_route_evidence_until_structured_extraction": True,
        "next_structured_output": "evidence/source_detail_curator_records.json",
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


def _material_type_from_link(link: dict[str, Any]) -> str:
    text = " ".join(str(link.get(key) or "") for key in ("content-type", "content-version", "intended-application", "URL")).lower()
    if "support" in text or "supplement" in text or "suppl" in text:
        return "supplementary"
    if "pdf" in text:
        return "article_pdf"
    if "xml" in text:
        return "article_xml"
    return "publisher_material"


def _next_actions(records: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in records:
        actions.append(
            {
                "schema_version": "source_material_next_action.v1",
                "source_ref": row.get("source_ref"),
                "material_record_id": row.get("record_id"),
                "material_type": row.get("material_type"),
                "next_action": "extract structured product/reactant SMILES into source_detail_curator_records.v1",
                "metadata_only": True,
            }
        )
    for gap in gaps:
        actions.append(
            {
                "schema_version": "source_material_next_action.v1",
                "source_ref": gap.get("source_ref"),
                "next_action": gap.get("next_action"),
                "metadata_only": True,
            }
        )
    return actions


def _gap(item: dict[str, Any], *, doi: str, reason: str, next_action: str) -> dict[str, Any]:
    return {
        "schema_version": "source_material_gap.v1",
        "source_ref": f"doi:{doi}" if doi else str(item.get("source_ref") or ""),
        "doi": doi,
        "queue_id": str(item.get("queue_id") or ""),
        "title": str(item.get("title") or ""),
        "reason": reason,
        "next_action": next_action,
        "metadata_only": True,
        "full_text_content_stored": False,
        "no_solved_claim": True,
        "production_write_blocked": True,
    }


def _load_pack(pack: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(pack, dict):
        return dict(pack)
    path = Path(pack)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _cached_json(
    *,
    source: str,
    url: str,
    cache_dir: Path,
    timeout_s: float,
    fetch_json: FetchJson,
) -> tuple[dict[str, Any], str, str]:
    cache = cache_dir / f"{source}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:20]}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")), str(cache), ""
        except json.JSONDecodeError:
            pass
    try:
        payload = fetch_json(url, {"Accept": "application/json", "User-Agent": "AutoPlanner/1.0"}, timeout_s)
    except Exception as exc:
        return {}, str(cache), f"{type(exc).__name__}: {exc}"
    cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload, str(cache), ""


def _fetch_json(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("url") or record.get("record_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _stable_id(*parts: Any) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
