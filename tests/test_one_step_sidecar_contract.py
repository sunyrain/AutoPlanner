from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cascade_planner.sidecars.one_step import (
    ONE_STEP_RESPONSE_SCHEMA,
    OneStepSidecarError,
    build_one_step_request,
    canonical_json_sha256,
    run_one_step_sidecar,
    validate_one_step_response,
)


def _response(request):
    return {
        "schema_version": ONE_STEP_RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": canonical_json_sha256(request),
        "status": "ok",
        "results": [
            {
                "query_id": request["queries"][0]["query_id"],
                "status": "ok",
                "candidates": [],
            }
        ],
        "semantics": {
            "shadow_only": True,
            "canonical_route_write_authority": False,
            "candidate_is_not_evidence": True,
        },
    }


def test_request_normalizes_queries_and_declares_no_authority():
    request = build_one_step_request([{"product_smiles": "CCO", "top_k": 3}], request_id="r1")
    assert request["queries"] == [{"query_id": "q1", "product_smiles": "CCO", "top_k": 3}]
    assert request["semantics"]["shadow_only"] is True
    assert request["semantics"]["canonical_route_write_authority"] is False


def test_response_rejects_canonical_write_authority():
    request = build_one_step_request([{"product_smiles": "CCO"}], request_id="r1")
    response = _response(request)
    response["semantics"]["canonical_route_write_authority"] = True
    with pytest.raises(OneStepSidecarError, match="canonical write authority"):
        validate_one_step_response(response, request=request)


def test_subprocess_transport_verifies_digest_and_schema():
    request = build_one_step_request([{"product_smiles": "CCO"}], request_id="r1")
    completed = SimpleNamespace(returncode=0, stdout=json.dumps(_response(request)), stderr="")
    with patch("cascade_planner.sidecars.one_step.subprocess.run", return_value=completed) as runner:
        response = run_one_step_sidecar(["sidecar-python", "worker.py"], request, timeout_s=12)
    assert response["status"] == "ok"
    assert json.loads(runner.call_args.kwargs["input"])["request_id"] == "r1"
    assert runner.call_args.kwargs["timeout"] == 12


def test_response_rejects_more_candidates_than_requested():
    request = build_one_step_request([{"product_smiles": "CCO", "top_k": 1}], request_id="r1")
    response = _response(request)
    response["results"][0]["candidates"] = [{"rank": 1}, {"rank": 2}]
    with pytest.raises(OneStepSidecarError, match="exceeded requested top_k"):
        validate_one_step_response(response, request=request)
