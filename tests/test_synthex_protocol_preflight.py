from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3

from cascade_planner.eval.synthex_protocol_preflight import (
    validate_synthex_head_to_head_protocol,
)
from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_synthex_protocol_preflight_binds_runtime_manifest_and_exact_stock(
    tmp_path: Path,
) -> None:
    protocol, manifest, stock, defaults = _fixture(tmp_path)
    result = validate_synthex_head_to_head_protocol(
        protocol_path=protocol,
        manifest_path=manifest,
        repository_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        execution_profile="paper_matched_reach",
        benchmark_stock_index=stock,
        benchmark_stock_name="ZINC+eMolecules",
        matched_defaults=defaults,
    )

    assert result["ready_for_paid_experiment"] is True
    assert result["issues"] == []
    assert result["stock"]["actual_member_count"] == 2
    assert result["stock"]["index_sha256"] == _sha256(stock)


def test_synthex_protocol_preflight_rejects_manifest_budget_drift(
    tmp_path: Path,
) -> None:
    protocol, manifest, stock, defaults = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][1]["budget"]["max_attempt_runs"] = 192
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_synthex_head_to_head_protocol(
        protocol_path=protocol,
        manifest_path=manifest,
        repository_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        execution_profile="paper_matched_reach",
        benchmark_stock_index=stock,
        benchmark_stock_name="ZINC+eMolecules",
        matched_defaults=defaults,
    )

    assert result["ready_for_paid_experiment"] is False
    assert any(
        row.get("code") == "manifest_case_budget_mismatch"
        and row.get("case_id") == "synthexfig1-002-5cac6c0d9928"
        and row.get("field") == "max_attempt_runs"
        for row in result["issues"]
    )


def test_synthex_protocol_preflight_rejects_enzyme_bias_in_paper_portfolio(
    tmp_path: Path,
) -> None:
    protocol, manifest, stock, defaults = _fixture(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["execution_contract"]["strategy_portfolio_mode"] = (
        "autoplanner_hybrid"
    )
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_synthex_head_to_head_protocol(
        protocol_path=protocol,
        manifest_path=manifest,
        repository_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        execution_profile="paper_matched_reach",
        benchmark_stock_index=stock,
        benchmark_stock_name="ZINC+eMolecules",
        matched_defaults=defaults,
    )

    assert result["ready_for_paid_experiment"] is False
    assert any(
        row.get("code") == "protocol_contract_mismatch"
        and row.get("field") == "execution_contract.strategy_portfolio_mode"
        and row.get("expected") == "paper_independent"
        for row in result["issues"]
    )


def test_synthex_protocol_preflight_rejects_wrong_candidate_width_or_tail_engine(
    tmp_path: Path,
) -> None:
    protocol, manifest, stock, defaults = _fixture(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["execution_contract"]["reactionjson_candidates_per_node"] = 3
    payload["execution_contract"]["short_tail"]["engine"] = "ChemEnzy"
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_synthex_head_to_head_protocol(
        protocol_path=protocol,
        manifest_path=manifest,
        repository_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        execution_profile="paper_matched_reach",
        benchmark_stock_index=stock,
        benchmark_stock_name="ZINC+eMolecules",
        matched_defaults=defaults,
    )

    fields = {row.get("field") for row in result["issues"]}
    assert result["ready_for_paid_experiment"] is False
    assert "reactionjson_candidates_per_node" in fields
    assert "short_tail.engine" in fields


def test_synthex_protocol_preflight_rejects_enzyme_companion_in_isolated_arm(
    tmp_path: Path,
) -> None:
    protocol, manifest, stock, defaults = _fixture(tmp_path)
    result = validate_synthex_head_to_head_protocol(
        protocol_path=protocol,
        manifest_path=manifest,
        repository_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        execution_profile="paper_matched_reach",
        strategy_portfolio_mode="enzyme_advantage",
        benchmark_stock_index=stock,
        benchmark_stock_name="ZINC+eMolecules",
        matched_defaults=defaults,
    )

    assert result["ready_for_paid_experiment"] is False
    assert result["strategy_portfolio_mode"] == "enzyme_advantage"
    assert any(
        row.get("field") == "strategy_portfolio_mode"
        and row.get("expected") == "paper_independent"
        for row in result["issues"]
    )


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    stock = tmp_path / "paper-stock.sqlite3"
    with sqlite3.connect(stock) as connection:
        connection.execute(
            "CREATE TABLE stock (full_inchikey TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO stock(full_inchikey) VALUES (?)",
            [
                ("LFQSCWFLJHTTHZ-UHFFFAOYSA-N",),
                ("QUSNBJAOOMFDIB-UHFFFAOYSA-N",),
            ],
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(
                {
                    "schema_version": "frozen_benchmark_stock_index.v1",
                    "catalog_name": "ZINC+eMolecules",
                    "identity_key": "full_inchikey",
                    "complete": "true",
                    "member_count": "2",
                    "source_sha256": "1" * 64,
                }.items()
            ),
        )
    protocol_payload = json.loads(
        (
            ROOT / "benchmarks" / "synthex_figure1_head_to_head_3.protocol.json"
        ).read_text(encoding="utf-8")
    )
    protocol_payload = copy.deepcopy(protocol_payload)
    protocol_payload["stock_binding"].update(
        {
            "index_path": str(stock),
            "index_sha256": _sha256(stock),
            "unique_member_count": 2,
            "paper_declared_entry_count": 2,
            "count_reconciliation": {
                "zinc_unique_full_inchikeys": 1,
                "emolecules_input_rows": 1,
                "emolecules_valid_full_inchikey_rows": 1,
                "emolecules_unique_full_inchikeys": 1,
                "cross_source_overlap_full_inchikeys": 0,
                "redundant_or_invalid_emolecules_rows": 0,
            },
        }
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_payload), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        (
            ROOT / "benchmarks" / "synthex_figure1_head_to_head_3.v1.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    defaults = dict(SYNTHEX_MATCHED_PROFILE_DEFAULTS)
    defaults["paper_reference_stock_member_count"] = 2
    defaults["paper_reference_stock_unique_member_count"] = 2
    defaults["paper_reference_stock_declared_entry_count"] = 2
    defaults["paper_reference_stock_zinc_unique_count"] = 1
    defaults["paper_reference_stock_emolecules_input_rows"] = 1
    defaults["paper_reference_stock_emolecules_valid_rows"] = 1
    defaults["paper_reference_stock_emolecules_unique_count"] = 1
    defaults["paper_reference_stock_cross_source_overlap_count"] = 0
    defaults["paper_reference_stock_redundant_or_invalid_rows"] = 0
    return protocol, manifest, stock, defaults


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
