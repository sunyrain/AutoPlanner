from __future__ import annotations

import hashlib
import json

import pytest

from scripts.compile_chemenzy_native_parity_panel import compile_native_parity_panel


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _report(*, target_smiles: str, accepted: bool, nonempty: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "chemenzy_native_parity_probe.v1",
        "request": {"target_smiles": target_smiles},
        "model_content_binding_sha256": "model-sha",
        "model_content_identity_complete": True,
        "parameter_binding_identity_complete": True,
        "parameter_binding_accepted": True,
        "parameter_binding_sha256": "parameter-sha",
        "stock_content_binding_sha256": "stock-sha",
        "stock_content_identity_complete": True,
        "embedded": {
            "route_count": 1 if nonempty else 0,
            "quarantined_route_count": 0,
            "search_trace_count": 2,
            "raw_proposal_sha256": "proposal-sha",
            "search_trace_sha256": "trace-sha",
        },
        "standalone": {
            "route_count": 1 if nonempty else 0,
            "quarantined_route_count": 0,
            "search_trace_count": 2,
            "raw_proposal_sha256": "proposal-sha",
            "search_trace_sha256": "trace-sha",
        },
        "backend_failure_free": True,
        "nonempty_route_set_observed": nonempty,
        "search_trace_identity_complete": True,
        "search_trace_digest_equal": True,
        "raw_proposal_digest_equal": True,
        "route_fingerprint_rows_equal": True,
        "parity_accepted": accepted,
    }
    report["content_sha256"] = _digest(report)
    return report


def test_panel_distinguishes_strict_and_vacuous_parity() -> None:
    first = _report(target_smiles="CCO", accepted=True, nonempty=True)
    second = _report(target_smiles="CCN", accepted=False, nonempty=False)
    panel = compile_native_parity_panel(
        [first, second],
        benchmark_cases=[
            {"case_id": "case-1", "target_name": "target 1", "target_smiles": "CCO"},
            {"case_id": "case-2", "target_name": "target 2", "target_smiles": "CCN"},
        ],
    )
    assert panel["summary"] == {
        "panel_size": 2,
        "strict_nonvacuous_parity_count": 1,
        "raw_proposal_parity_count": 2,
        "search_trace_parity_count": 2,
        "strict_nonvacuous_parity_rate": 0.5,
        "raw_proposal_parity_rate": 1.0,
        "all_selected_raw_proposals_equal": True,
        "all_selected_strictly_accepted": False,
    }
    assert panel["rows"][1]["disposition"] == "deterministic_but_empty_route_set"


def test_panel_rejects_tampered_report_digest() -> None:
    report = _report(target_smiles="CCO", accepted=True, nonempty=True)
    report["embedded"]["route_count"] = 99  # type: ignore[index]
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        compile_native_parity_panel([report])


def test_panel_retains_v2_normalization_rejection() -> None:
    report = _report(target_smiles="CCO", accepted=False, nonempty=True)
    report["schema_version"] = "chemenzy_native_parity_probe.v2"
    report["normalization_invariants_complete"] = True
    report["normalization_invariants_accepted"] = False
    report["content_sha256"] = _digest(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )

    panel = compile_native_parity_panel([report])

    assert panel["rows"][0]["parity_accepted"] is False
    assert panel["rows"][0]["normalization_invariants_required"] is True
    assert panel["rows"][0]["normalization_invariants_accepted"] is False
    assert panel["rows"][0]["disposition"] == "parity_rejected"
