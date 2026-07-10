from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_legacy_example_runs import (
    JsonCheck,
    LegacyExpectation,
    validate_legacy_example,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_validate_legacy_example_accepts_expected_json_contract(tmp_path) -> None:
    _write_json(
        tmp_path / "final_verdict.json",
        {
            "schema_version": "codex_entry_final_verdict.v1",
            "case_id": "fluvastatin",
            "verdict": "partial_anchor_only_not_solved",
            "route_status": "unresolved",
            "solved": False,
            "reasons": ["literature_anchor_without_executable_stock_closure"],
        },
    )
    _write_json(
        tmp_path / "artifact_bundle.json",
        {
            "schema_version": "codex_entry_artifact_bundle.v1",
            "artifacts": {"chemenzy": {"n_results": 3}},
        },
    )
    expectation = LegacyExpectation(
        label="legacy_ok",
        run_dir=tmp_path,
        required_json_files=("final_verdict.json", "artifact_bundle.json"),
        checks=(
            JsonCheck("final_verdict.json", "schema_version", equals="codex_entry_final_verdict.v1"),
            JsonCheck("final_verdict.json", "solved", equals=False),
            JsonCheck("artifact_bundle.json", "artifacts.chemenzy.n_results", min_value=1),
        ),
        required_text=("fluvastatin", "literature_anchor_without_executable_stock_closure"),
    )

    row = validate_legacy_example(expectation)

    assert row["accepted"], row["reasons"]
    assert row["checked_json_files"] == ["artifact_bundle.json", "final_verdict.json"]


def test_validate_legacy_example_reports_missing_and_mismatched_fields(tmp_path) -> None:
    _write_json(
        tmp_path / "final_verdict.json",
        {
            "schema_version": "codex_entry_final_verdict.v1",
            "case_id": "fluvastatin",
            "verdict": "solved",
            "solved": True,
        },
    )
    expectation = LegacyExpectation(
        label="legacy_bad",
        run_dir=tmp_path,
        required_json_files=("final_verdict.json", "artifact_bundle.json"),
        checks=(
            JsonCheck("final_verdict.json", "verdict", equals="partial_anchor_only_not_solved"),
            JsonCheck("final_verdict.json", "route_status", equals="unresolved"),
            JsonCheck("final_verdict.json", "solved", equals=False),
            JsonCheck("artifact_bundle.json", "artifacts.chemenzy.n_results", min_value=1),
        ),
        required_text=("literature_anchor_without_executable_stock_closure",),
    )

    row = validate_legacy_example(expectation)

    assert not row["accepted"]
    assert "required_json_missing:artifact_bundle.json" in row["reasons"]
    assert any(reason.startswith("json_path_mismatch:final_verdict.json:verdict") for reason in row["reasons"])
    assert "json_path_missing:final_verdict.json:route_status" in row["reasons"]
    assert "missing_required_text:literature_anchor_without_executable_stock_closure" in row["reasons"]
