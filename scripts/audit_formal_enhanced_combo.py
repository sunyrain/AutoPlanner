#!/usr/bin/env python
"""Audit the formally assembled enhanced route-tree combo.

The audit is intentionally artifact-driven: it reads the p16n16 and p100
benchmark outputs and checks the invariants that define the formal enhanced
configuration, including default-off behavior for optional rescue sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_P16_REPORT = ROOT / (
    "results/shared/chemenzy_assembly_ab_20260531/"
    "route_final_clean_fastclosure_material_gate_semisynthesis_chemical_anchor_p16n16_20260601/"
    "native_vs_enhanced_route_report.json"
)
DEFAULT_P100_REPORT = ROOT / (
    "results/shared/native_vs_enhanced_route_benchmark_20260601_p30n70_"
    "fastclosure_material_gate_semisynthesis_chemical_anchor_enhanced/"
    "native_vs_enhanced_route_report.json"
)
DEFAULT_OUTPUT_JSON = ROOT / (
    "results/shared/chemenzy_assembly_ab_20260531/"
    "formal_chemical_anchor_combo_audit_20260601.json"
)
DEFAULT_OUTPUT_MD = ROOT / (
    "results/shared/chemenzy_assembly_ab_20260531/"
    "formal_chemical_anchor_combo_audit_20260601.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p16-report", type=Path, default=DEFAULT_P16_REPORT)
    parser.add_argument("--p100-report", type=Path, default=DEFAULT_P100_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p16_report, p16_rows = _load_report_and_rows(args.p16_report)
    p100_report, p100_rows = _load_report_and_rows(args.p100_report)
    checks: list[dict[str, Any]] = []
    checks.extend(_default_and_preset_checks())
    checks.extend(_p16_checks(p16_report, p16_rows))
    checks.extend(_p100_checks(p100_report, p100_rows))
    passed = all(check["status"] == "pass" for check in checks)
    audit = {
        "schema_version": "formal_enhanced_combo_audit.v1",
        "passed": passed,
        "p16_report": str(args.p16_report),
        "p100_report": str(args.p100_report),
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check["status"] == "pass"),
            "checks_failed": sum(1 for check in checks if check["status"] != "pass"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps({"passed": passed, "output_json": str(args.output_json), "output_md": str(args.output_md)}))
    if not passed:
        raise SystemExit(1)


def _load_report_and_rows(report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_path = Path(report.get("rows_jsonl") or report_path.with_name("native_vs_enhanced_route_rows.jsonl"))
    if not rows_path.is_absolute():
        rows_path = ROOT / rows_path
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return report, rows


def _default_and_preset_checks() -> list[dict[str, Any]]:
    import scripts.run_native_vs_enhanced_route_benchmark as bench

    old_argv = list(sys.argv)
    try:
        sys.argv = ["run_native_vs_enhanced_route_benchmark.py"]
        default_args = bench.parse_args()
        sys.argv = [
            "run_native_vs_enhanced_route_benchmark.py",
            "--enhancement-preset",
            bench.FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET,
        ]
        preset_args = bench.parse_args()
        preset_env = bench.enhanced_env_values(preset_args)
    finally:
        sys.argv = old_argv
    return [
        _check(
            "default_chemical_anchor_disabled",
            not default_args.enable_chemical_anchor_stock
            and not default_args.enable_chemical_anchor_rescue_source,
            {
                "enable_chemical_anchor_stock": default_args.enable_chemical_anchor_stock,
                "enable_chemical_anchor_rescue_source": default_args.enable_chemical_anchor_rescue_source,
            },
        ),
        _check(
            "preset_enables_chemical_anchor_explicitly",
            preset_args.enable_chemical_anchor_stock
            and preset_args.enable_chemical_anchor_rescue_source
            and preset_env.get("AUTOPLANNER_ENABLE_CHEMICAL_ANCHOR_RESCUE_PROPOSALS") == "1",
            {
                "preset": preset_args.enhancement_preset,
                "enable_chemical_anchor_stock": preset_args.enable_chemical_anchor_stock,
                "enable_chemical_anchor_rescue_source": preset_args.enable_chemical_anchor_rescue_source,
                "env_enabled": preset_env.get("AUTOPLANNER_ENABLE_CHEMICAL_ANCHOR_RESCUE_PROPOSALS"),
            },
        ),
    ]


def _p16_checks(report: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = _summary(report)
    return [
        _check("p16_solved_32_of_32", summary.get("targets_with_solved_route") == 32, summary),
        _check("p16_positive_enzyme_16", summary.get("targets_with_enzyme_route") == 16, summary),
        _check("p16_sp_v1_accepted_16", summary.get("targets_with_sp_v1_accepted_enzyme_route") == 16, summary),
        _check("p16_no_negative_enzyme", _negative_enzyme_rows(rows) == [], {"rows": _negative_enzyme_rows(rows)}),
        _check("p16_no_quality_reject", _quality_reject_count(rows) == 0, {"rejects": _quality_reject_count(rows)}),
        _check(
            "p16_no_anchor_calls",
            _source_calls(summary, "chemical_anchor_rescue") == 0
            and _source_calls(summary, "semisynthesis_rescue") == 0,
            {
                "chemical_anchor_calls": _source_calls(summary, "chemical_anchor_rescue"),
                "semisynthesis_calls": _source_calls(summary, "semisynthesis_rescue"),
            },
        ),
    ]


def _p100_checks(report: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = _summary(report)
    inputs = report.get("inputs") or {}
    return [
        _check("p100_solved_100_of_100", summary.get("targets_with_solved_route") == 100, summary),
        _check("p100_positive_enzyme_30", summary.get("targets_with_enzyme_route") == 30, summary),
        _check("p100_sp_v1_accepted_30", summary.get("targets_with_sp_v1_accepted_enzyme_route") == 30, summary),
        _check("p100_no_negative_enzyme", _negative_enzyme_rows(rows) == [], {"rows": _negative_enzyme_rows(rows)}),
        _check("p100_no_quality_reject", _quality_reject_count(rows) == 0, {"rejects": _quality_reject_count(rows)}),
        _check("p100_no_timeout_frontier", summary.get("timeout_frontier_routes") == 0, summary),
        _check(
            "formal_run_excludes_retrochimera_and_native_rescue",
            bool(inputs.get("skip_native")) is True
            and bool(inputs.get("enable_native_chemical_rescue")) is False
            and bool(inputs.get("retrochimera_source_enabled")) is False,
            {
                "skip_native": inputs.get("skip_native"),
                "enable_native_chemical_rescue": inputs.get("enable_native_chemical_rescue"),
                "retrochimera_source_enabled": inputs.get("retrochimera_source_enabled"),
            },
        ),
        _check(
            "p100_source_calls_are_product_filtered",
            _source_calls(summary, "chemical_anchor_rescue") == 2
            and _source_calls(summary, "semisynthesis_rescue") == 1,
            {
                "chemical_anchor_calls": _source_calls(summary, "chemical_anchor_rescue"),
                "semisynthesis_calls": _source_calls(summary, "semisynthesis_rescue"),
            },
        ),
        _row_check(rows, 1, "chemical_anchor_rescue", "benzothiazine_c2_amination", "20143033"),
        _row_check(rows, 62, "semisynthesis_rescue", "taxane_10dab_side_chain_acetylation", "10-Deacetylbaccatin III"),
        _row_check(rows, 84, "chemical_anchor_rescue", "benzothiazole_dibenzylation", "67480579"),
        _check("row76_has_no_rescue_source_stats", _rescue_source_calls(rows, 76) == {}, _rescue_source_calls(rows, 76)),
        _check("row89_has_no_rescue_source_stats", _rescue_source_calls(rows, 89) == {}, _rescue_source_calls(rows, 89)),
    ]


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return dict(((report.get("summaries") or {}).get("enhanced_route_tree") or {}))


def _source_calls(summary: dict[str, Any], source: str) -> int:
    return int((summary.get("proposal_source_calls") or {}).get(source) or 0)


def _negative_enzyme_rows(rows: list[dict[str, Any]]) -> list[int]:
    return [
        idx
        for idx, row in enumerate(rows, start=1)
        if int(row.get("label") or 0) == 0 and int(row.get("enzyme_routes") or 0) > 0
    ]


def _quality_reject_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        for route in row.get("routes") or []
        for step in route.get("steps") or []
        if ((step.get("enzyme_step_quality_v1") or {}).get("decision") == "reject")
    )


def _row_check(
    rows: list[dict[str, Any]],
    row_number: int,
    expected_source: str,
    expected_type: str,
    expected_record: str,
) -> dict[str, Any]:
    row = rows[row_number - 1]
    route = (row.get("routes") or [{}])[0]
    source_counts = route.get("source_counts") or {}
    evidence_hits = []
    for step in route.get("steps") or []:
        evidence = step.get("evidence") or {}
        payload = evidence.get(expected_source) or {}
        record = payload.get("precursor_source_record") or {}
        evidence_hits.append(
            {
                "source": step.get("source"),
                "type": payload.get("type"),
                "pubchem_cid": record.get("pubchem_cid"),
                "name": record.get("name"),
                "stock_status": step.get("stock_status"),
            }
        )
    record_ok = any(
        hit.get("pubchem_cid") == expected_record or hit.get("name") == expected_record
        for hit in evidence_hits
    )
    type_ok = any(hit.get("type") == expected_type for hit in evidence_hits)
    return _check(
        f"row{row_number}_{expected_source}_evidence",
        bool(row.get("solved_routes"))
        and source_counts.get(expected_source) == 1
        and type_ok
        and record_ok,
        {
            "solved_routes": row.get("solved_routes"),
            "source_counts": source_counts,
            "evidence_hits": evidence_hits,
        },
    )


def _rescue_source_calls(rows: list[dict[str, Any]], row_number: int) -> dict[str, int]:
    stats = (rows[row_number - 1].get("stats") or {}).get("proposal_source_stats") or {}
    return {
        source: int((stats.get(source) or {}).get("calls") or 0)
        for source in ("chemical_anchor_rescue", "semisynthesis_rescue")
        if int((stats.get(source) or {}).get("calls") or 0) > 0
    }


def _check(check_id: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    }


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Formal Enhanced Combo Audit",
        "",
        f"Passed: `{audit['passed']}`",
        "",
        f"- p16 report: `{audit['p16_report']}`",
        f"- p100 report: `{audit['p100_report']}`",
        "",
        "| check | status |",
        "| --- | --- |",
    ]
    for check in audit["checks"]:
        lines.append(f"| `{check['id']}` | `{check['status']}` |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
