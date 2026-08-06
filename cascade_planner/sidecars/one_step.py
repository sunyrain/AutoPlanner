"""Versioned subprocess contract for one-step retrosynthesis sidecars.

The contract deliberately carries no route-admission operation.  A successful
response is a set of proposals for evaluation or later independent validation;
it is never evidence and never mutates the canonical route graph by itself.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


ONE_STEP_REQUEST_SCHEMA = "autoplanner.one_step_sidecar_request.v1"
ONE_STEP_RESPONSE_SCHEMA = "autoplanner.one_step_sidecar_response.v1"


class OneStepSidecarError(RuntimeError):
    """Raised when a sidecar violates the transport or semantic contract."""


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_one_step_request(
    queries: Sequence[Mapping[str, Any]],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        product_smiles = str(query.get("product_smiles") or "").strip()
        if not product_smiles:
            raise ValueError(f"query {index} has no product_smiles")
        top_k = int(query.get("top_k") or 10)
        if top_k < 1 or top_k > 100:
            raise ValueError(f"query {index} top_k must be between 1 and 100")
        normalized.append(
            {
                "query_id": str(query.get("query_id") or f"q{index + 1}"),
                "product_smiles": product_smiles,
                "top_k": top_k,
            }
        )
    if not normalized:
        raise ValueError("at least one query is required")
    return {
        "schema_version": ONE_STEP_REQUEST_SCHEMA,
        "request_id": str(request_id or uuid.uuid4()),
        "queries": normalized,
        "semantics": {
            "shadow_only": True,
            "canonical_route_write_authority": False,
            "candidate_is_not_evidence": True,
        },
    }


def validate_one_step_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if response.get("schema_version") != ONE_STEP_RESPONSE_SCHEMA:
        raise OneStepSidecarError("unexpected one-step sidecar response schema")
    if response.get("request_id") != request.get("request_id"):
        raise OneStepSidecarError("one-step sidecar request_id mismatch")
    expected_digest = canonical_json_sha256(request)
    if response.get("request_sha256") != expected_digest:
        raise OneStepSidecarError("one-step sidecar request digest mismatch")
    semantics = response.get("semantics")
    if not isinstance(semantics, Mapping):
        raise OneStepSidecarError("one-step sidecar omitted semantic safety flags")
    if semantics.get("shadow_only") is not True:
        raise OneStepSidecarError("one-step sidecar response is not shadow-only")
    if semantics.get("canonical_route_write_authority") is not False:
        raise OneStepSidecarError("one-step sidecar claims canonical write authority")
    if semantics.get("candidate_is_not_evidence") is not True:
        raise OneStepSidecarError("one-step sidecar conflates proposals with evidence")
    results = response.get("results")
    if not isinstance(results, list):
        raise OneStepSidecarError("one-step sidecar results must be a list")
    expected_queries = {
        str(item["query_id"]): item for item in list(request.get("queries") or [])
    }
    if len(results) != len(expected_queries):
        raise OneStepSidecarError("one-step sidecar returned the wrong result count")
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise OneStepSidecarError("one-step sidecar result is not an object")
        query_id = str(result.get("query_id") or "")
        if query_id not in expected_queries or query_id in seen:
            raise OneStepSidecarError("one-step sidecar returned an unknown or duplicate query_id")
        seen.add(query_id)
        candidates = result.get("candidates")
        if not isinstance(candidates, list):
            raise OneStepSidecarError(f"query {query_id} candidates must be a list")
        if len(candidates) > int(expected_queries[query_id]["top_k"]):
            raise OneStepSidecarError(f"query {query_id} exceeded requested top_k")
    return dict(response)


def run_one_step_sidecar(
    command: Sequence[str],
    request: Mapping[str, Any],
    *,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Run one isolated batch sidecar and verify its response envelope."""

    if not command:
        raise ValueError("sidecar command is required")
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OneStepSidecarError(f"one-step sidecar timed out after {timeout_s:g}s") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise OneStepSidecarError(
            f"one-step sidecar exited with code {completed.returncode}: {detail}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        preview = completed.stdout.strip()[:500]
        raise OneStepSidecarError(f"one-step sidecar returned invalid JSON: {preview}") from exc
    if not isinstance(response, Mapping):
        raise OneStepSidecarError("one-step sidecar response must be a JSON object")
    return validate_one_step_response(response, request=request)
