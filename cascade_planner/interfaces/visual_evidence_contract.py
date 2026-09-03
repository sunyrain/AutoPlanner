"""Shared validation and projection helpers for sparse visual evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


VISUAL_EVIDENCE_REQUEST_SCHEMA = "visual_source_candidate_request.v1"


class VisualEvidenceError(RuntimeError):
    """A visual provider or its output violated the bounded draft contract."""


def validate_request_digest(request: Mapping[str, Any]) -> None:
    body = {key: value for key, value in request.items() if key != "content_sha256"}
    if (
        request.get("schema_version") != VISUAL_EVIDENCE_REQUEST_SCHEMA
        or str(request.get("content_sha256") or "") != digest(body)
    ):
        raise VisualEvidenceError("visual_evidence_request_invalid")


def source_ref(source: Mapping[str, Any]) -> str:
    explicit = str(source.get("source_ref") or "").strip()
    if explicit:
        return explicit[:500]
    doi = str(source.get("doi") or "").strip()
    if doi:
        return f"doi:{doi.removeprefix('https://doi.org/').removeprefix('http://doi.org/')}"[
            :500
        ]
    pmid = str(source.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"[:500]
    publication = str(source.get("publication_number") or "").strip()
    return f"patent:{publication}" if publication else ""


def source_kind(source_reference: str) -> str:
    prefix = str(source_reference).split(":", 1)[0].lower()
    return "paper_si" if prefix in {"doi", "pmid", "pmc"} else prefix


def materialization_stage(
    status: str, *, reason: str = "", **values: Any
) -> dict[str, Any]:
    return {
        "stage": "visual_chain_materialization",
        "status": status,
        "reason": reason,
        **values,
    }


def normalized_usage(value: Any) -> dict[str, Any]:
    row = dict(value) if isinstance(value, Mapping) else {}
    invocations = max(0, int(row.get("model_invocations") or 0))
    visual = max(0, int(row.get("visual_invocations") or 0))
    if invocations > 1 or visual > 1 or visual > invocations:
        raise VisualEvidenceError("visual_provider_usage_invalid")
    return {
        "model_invocations": invocations,
        "visual_invocations": visual,
        "input_tokens": max(0, int(row.get("input_tokens") or 0)),
        "output_tokens": max(0, int(row.get("output_tokens") or 0)),
        "wall_time_s": max(0.0, float(row.get("wall_time_s") or 0.0)),
    }


def stage(status: str, *, reason: str = "", **values: Any) -> dict[str, Any]:
    usage = dict(values.get("model_usage") or {})
    return {
        "stage": "visual_evidence",
        "status": status,
        "reason": str(reason),
        "model_invocations": int(usage.get("model_invocations") or 0),
        "visual_invocations": int(usage.get("visual_invocations") or 0),
        **values,
    }


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


__all__ = [
    "VISUAL_EVIDENCE_REQUEST_SCHEMA",
    "VisualEvidenceError",
    "digest",
    "is_sha256",
    "materialization_stage",
    "normalized_usage",
    "sha256",
    "source_kind",
    "source_ref",
    "stage",
    "validate_request_digest",
]
