"""Compile a bounded, machine-readable ChemEnzy native parity panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def compile_native_parity_panel(
    reports: Sequence[Mapping[str, Any]],
    *,
    benchmark_cases: Sequence[Mapping[str, Any]] = (),
    evidence_date: str | None = None,
    source_commit: str = "",
    benchmark_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Compile summaries without weakening the single-request probe gates.

    The panel is intentionally bounded.  A row can have equal raw proposals and
    traces while still being rejected when both arms return an empty route set;
    that distinction prevents vacuous parity from being reported as success.
    """
    if not reports:
        raise ValueError("at least one parity report is required")
    cases_by_smiles = {
        str(case.get("target_smiles") or ""): case
        for case in benchmark_cases
        if str(case.get("target_smiles") or "")
    }
    rows: list[dict[str, Any]] = []
    request_contracts: set[str] = set()
    model_bindings: set[str] = set()
    stock_bindings: set[str] = set()
    for report in reports:
        _require_report_digest(report)
        request = dict(report.get("request") or {})
        target_smiles = str(request.get("target_smiles") or "")
        if not target_smiles:
            raise ValueError("parity report request.target_smiles is required")
        case = dict(cases_by_smiles.get(target_smiles) or {})
        embedded = dict(report.get("embedded") or {})
        standalone = dict(report.get("standalone") or {})
        request_contracts.add(
            _digest(
                {
                    key: request.get(key)
                    for key in (
                        "search_preset",
                        "max_routes",
                        "max_steps",
                        "iterations",
                        "expansion_topk",
                        "random_seed",
                        "stock_names",
                        "stock_paths",
                        "condition_prediction",
                        "enzyme_assignment",
                    )
                }
            )
        )
        model_bindings.add(str(report.get("model_content_binding_sha256") or ""))
        stock_bindings.add(str(report.get("stock_content_binding_sha256") or ""))
        raw_equal = bool(report.get("raw_proposal_digest_equal"))
        trace_equal = bool(report.get("search_trace_digest_equal"))
        fingerprints_equal = bool(report.get("route_fingerprint_rows_equal"))
        backend_failure_free = bool(report.get("backend_failure_free"))
        nonempty = bool(report.get("nonempty_route_set_observed"))
        accepted = bool(
            report.get("model_content_identity_complete")
            and report.get("stock_content_identity_complete")
            and backend_failure_free
            and nonempty
            and report.get("search_trace_identity_complete")
            and trace_equal
            and raw_equal
            and fingerprints_equal
        )
        if accepted != bool(report.get("parity_accepted")):
            raise ValueError("parity report acceptance is inconsistent with its gates")
        rows.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "target_name": str(
                    case.get("target_name")
                    or request.get("target_name")
                    or "opaque parity target"
                ),
                "target_smiles": target_smiles,
                "parity_accepted": accepted,
                "raw_proposal_digest_equal": raw_equal,
                "search_trace_digest_equal": trace_equal,
                "route_fingerprint_rows_equal": fingerprints_equal,
                "backend_failure_free": backend_failure_free,
                "nonempty_route_set_observed": nonempty,
                "embedded_route_count": int(embedded.get("route_count") or 0),
                "standalone_route_count": int(standalone.get("route_count") or 0),
                "embedded_quarantined_route_count": int(
                    embedded.get("quarantined_route_count") or 0
                ),
                "standalone_quarantined_route_count": int(
                    standalone.get("quarantined_route_count") or 0
                ),
                "embedded_search_trace_count": int(
                    embedded.get("search_trace_count") or 0
                ),
                "standalone_search_trace_count": int(
                    standalone.get("search_trace_count") or 0
                ),
                "embedded_raw_proposal_sha256": str(
                    embedded.get("raw_proposal_sha256") or ""
                ),
                "standalone_raw_proposal_sha256": str(
                    standalone.get("raw_proposal_sha256") or ""
                ),
                "embedded_search_trace_sha256": str(
                    embedded.get("search_trace_sha256") or ""
                ),
                "standalone_search_trace_sha256": str(
                    standalone.get("search_trace_sha256") or ""
                ),
                "report_content_sha256": str(report.get("content_sha256") or ""),
                "report_generated_at": str(report.get("generated_at") or ""),
                "report_source_commit": str(
                    report.get("source_commit") or source_commit
                ),
                "disposition": (
                    "accepted_nonvacuous"
                    if accepted
                    else (
                        "deterministic_but_empty_route_set"
                        if raw_equal and trace_equal and fingerprints_equal and backend_failure_free
                        else "parity_rejected"
                    )
                ),
            }
        )

    accepted = sum(bool(row["parity_accepted"]) for row in rows)
    raw_equal_count = sum(bool(row["raw_proposal_digest_equal"]) for row in rows)
    trace_equal_count = sum(bool(row["search_trace_digest_equal"]) for row in rows)
    panel = {
        "schema_version": "chemenzy_native_parity_panel.v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_date": evidence_date or datetime.now(timezone.utc).date().isoformat(),
        "source_commit": source_commit,
        "scope": {
            "benchmark": "Retro*-190",
            "benchmark_manifest": "benchmarks/retrostar190_v4.json",
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "selection_rule": "manifest prefix; bounded engineering evidence only",
            "selected_target_count": len(rows),
            "full_benchmark_target_count": 190,
            "not_full_benchmark_result": True,
        },
        "contract": {
            "uniform_request_contract": len(request_contracts) == 1,
            "uniform_model_content_binding": len(model_bindings) == 1,
            "uniform_stock_content_binding": len(stock_bindings) == 1,
            "request_contract_sha256": sorted(request_contracts),
            "model_content_binding_sha256": sorted(model_bindings),
            "stock_content_binding_sha256": sorted(stock_bindings),
        },
        "summary": {
            "panel_size": len(rows),
            "strict_nonvacuous_parity_count": accepted,
            "raw_proposal_parity_count": raw_equal_count,
            "search_trace_parity_count": trace_equal_count,
            "strict_nonvacuous_parity_rate": round(accepted / len(rows), 6),
            "raw_proposal_parity_rate": round(raw_equal_count / len(rows), 6),
            "all_selected_raw_proposals_equal": raw_equal_count == len(rows),
            "all_selected_strictly_accepted": accepted == len(rows),
        },
        "rows": rows,
        "limitations": [
            "This panel is bounded evidence for the selected manifest prefix, not a 190-target result.",
            "An empty route set is reported as deterministic-but-empty and does not satisfy strict parity acceptance.",
            "Fresh/uncached CSV preparation remains a separate Windows torchtext compatibility boundary.",
        ],
    }
    panel["content_sha256"] = _digest(panel)
    return panel


def _require_report_digest(report: Mapping[str, Any]) -> None:
    supplied = str(report.get("content_sha256") or "")
    if not supplied:
        raise ValueError("parity report is missing content_sha256")
    unsigned = dict(report)
    unsigned.pop("content_sha256", None)
    if _digest(unsigned) != supplied:
        raise ValueError("parity report content_sha256 mismatch")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/retrostar190_v4.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-date")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--benchmark-manifest-sha256", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    benchmark = _read_json(args.benchmark)
    panel = compile_native_parity_panel(
        [_read_json(path) for path in args.report],
        benchmark_cases=list(benchmark.get("cases") or []),
        evidence_date=args.evidence_date,
        source_commit=args.source_commit,
        benchmark_manifest_sha256=args.benchmark_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(panel, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
