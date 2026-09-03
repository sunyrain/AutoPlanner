#!/usr/bin/env python3
"""Compare standalone ChemEnzy proposals with one embedded V4 trajectory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.interfaces.chemenzy_probe import (  # noqa: E402
    compile_chemenzy_route_fingerprints,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_detail(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    for value in reversed(list(report.get("stages") or [])):
        if isinstance(value, Mapping) and value.get("stage") == name:
            return dict(value.get("detail") or {})
    return {}


def _route_counter(rows: list[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(row.get("normalized_route_sha256") or "")
        for row in rows
        if str(row.get("normalized_route_sha256") or "")
    )


def _expanded_delta(left: Counter[str], right: Counter[str]) -> list[str]:
    return sorted((left - right).elements())


def compare_embedding(
    standalone: Mapping[str, Any],
    embedded_report: Mapping[str, Any],
) -> dict[str, Any]:
    target_smiles = str(
        dict(embedded_report.get("target") or {}).get("canonical_smiles") or ""
    )
    if not target_smiles:
        raise ValueError("embedded report has no canonical target SMILES")
    standalone_set = compile_chemenzy_route_fingerprints(
        standalone,
        target_smiles=target_smiles,
    )
    seed = _stage_detail(embedded_report, "chemenzy_baseline")
    embedded_rows = [
        dict(value)
        for value in seed.get("route_lineage") or []
        if isinstance(value, Mapping)
    ]
    if not embedded_rows:
        raise ValueError(
            "embedded report has no digest-bound route_lineage; rerun with current V4"
        )
    final = _stage_detail(embedded_report, "chemenzy_route_lineage")
    final_rows = [
        dict(value)
        for value in final.get("routes") or []
        if isinstance(value, Mapping)
    ]
    standalone_rows = [
        dict(value)
        for value in standalone_set.get("routes") or []
        if isinstance(value, Mapping)
    ]
    standalone_counter = _route_counter(standalone_rows)
    embedded_counter = _route_counter(embedded_rows)
    selected = {
        str(row.get("normalized_route_sha256") or "")
        for row in embedded_rows
        if row.get("host_portfolio_selected") is True
    } - {""}
    canonical = {
        str(row.get("normalized_route_sha256") or "")
        for row in final_rows
        if row.get("host_portfolio_selected") is True
        and str(row.get("canonical_route_family_id") or "")
    } - {""}
    partially_materialized = {
        str(row.get("normalized_route_sha256") or "")
        for row in final_rows
        if row.get("host_portfolio_selected") is True
        and row.get("canonical_edge_ids")
    } - {""}
    fully_materialized = {
        str(row.get("normalized_route_sha256") or "")
        for row in final_rows
        if row.get("host_portfolio_selected") is True
        and len(row.get("step_proposal_ids") or []) > 0
        and len(row.get("canonical_edge_ids") or [])
        >= len(row.get("step_proposal_ids") or [])
    } - {""}
    host_validated = {
        str(row.get("normalized_route_sha256") or "")
        for row in final_rows
        if str(row.get("normalized_route_sha256") or "") in fully_materialized
        and int(row.get("canonical_minimum_proof_level") or 0) >= 2
    } - {""}
    stock_closed = {
        str(row.get("normalized_route_sha256") or "")
        for row in final_rows
        if row.get("final_disposition") == "stock_closed"
    } - {""}
    first_loss_counts: Counter[str] = Counter()
    route_audit = []
    final_by_digest = {
        str(row.get("normalized_route_sha256") or ""): row
        for row in final_rows
        if str(row.get("normalized_route_sha256") or "")
    }
    embedded_by_digest = {
        str(row.get("normalized_route_sha256") or ""): row
        for row in embedded_rows
        if str(row.get("normalized_route_sha256") or "")
    }
    for row in standalone_rows:
        digest = str(row.get("normalized_route_sha256") or "")
        embedded = embedded_by_digest.get(digest, {})
        projected = final_by_digest.get(digest, {})
        if not embedded:
            first_loss = "missing_from_embedded_normalization"
        elif digest not in selected:
            first_loss = str(embedded.get("disposition") or "not_host_selected")
        elif digest not in canonical:
            first_loss = "canonical_route_family_missing"
        elif digest not in partially_materialized:
            first_loss = "canonical_hypothesis_not_materialized"
        elif digest not in fully_materialized:
            first_loss = "canonical_hypotheses_not_fully_materialized"
        elif digest not in host_validated:
            first_loss = "materialized_not_host_validated"
        elif digest not in stock_closed:
            first_loss = "host_validated_not_stock_closed"
        else:
            first_loss = "stock_closed"
        first_loss_counts[first_loss] += 1
        route_audit.append(
            {
                **row,
                "embedded_disposition": str(embedded.get("disposition") or ""),
                "canonical_route_family_id": str(
                    projected.get("canonical_route_family_id") or ""
                ),
                "canonical_edge_count": len(
                    projected.get("canonical_edge_ids") or []
                ),
                "expected_step_count": len(
                    projected.get("step_proposal_ids") or []
                ),
                "canonical_hypothesis_count": len(
                    projected.get("canonical_hypothesis_ids") or []
                ),
                "canonical_minimum_proof_level": int(
                    projected.get("canonical_minimum_proof_level") or 0
                ),
                "canonical_stock_closure_rate": float(
                    projected.get("canonical_stock_closure_rate") or 0.0
                ),
                "first_loss_boundary": first_loss,
            }
        )
    raw_result_digest_equal = bool(
        seed.get("raw_result_sha256")
        and seed.get("raw_result_sha256") == standalone_set["raw_result_sha256"]
    )
    raw_proposal_digest_equal = bool(
        seed.get("raw_proposal_sha256")
        and seed.get("raw_proposal_sha256") == standalone_set["raw_proposal_sha256"]
    )
    normalized_multiset_equal = standalone_counter == embedded_counter
    return {
        "schema_version": "chemenzy_embedding_comparison.v3",
        "target_smiles": target_smiles,
        "standalone_raw_proposal_sha256": standalone_set["raw_proposal_sha256"],
        "embedded_raw_proposal_sha256": str(seed.get("raw_proposal_sha256") or ""),
        "raw_proposal_digest_equal": raw_proposal_digest_equal,
        "standalone_raw_result_sha256": standalone_set["raw_result_sha256"],
        "embedded_raw_result_sha256": str(seed.get("raw_result_sha256") or ""),
        "raw_result_digest_equal": raw_result_digest_equal,
        "normalized_route_multiset_equal": normalized_multiset_equal,
        "counts": {
            "standalone_routes": sum(standalone_counter.values()),
            "embedded_routes": sum(embedded_counter.values()),
            "host_selected_routes": len(selected),
            "canonical_route_families": len(canonical),
            "partially_materialized_routes": len(partially_materialized),
            "fully_materialized_routes": len(fully_materialized),
            "host_validated_routes": len(host_validated),
            "stock_closed_routes": len(stock_closed),
        },
        "standalone_only_normalized_route_sha256": _expanded_delta(
            standalone_counter,
            embedded_counter,
        ),
        "embedded_only_normalized_route_sha256": _expanded_delta(
            embedded_counter,
            standalone_counter,
        ),
        "first_loss_counts": {
            key: first_loss_counts[key] for key in sorted(first_loss_counts)
        },
        "routes": route_audit,
        "semantics": {
            "comparison_is_route_digest_bound": True,
            "raw_parity_requires_identical_provider_payload": True,
            "normalization_selection_materialization_validation_and_stock_are_separate": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", required=True, type=Path)
    parser.add_argument("--embedded-report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args(argv)

    standalone = _read_object(args.standalone)
    embedded = _read_object(args.embedded_report)
    result = compare_embedding(standalone, embedded)
    result["inputs"] = {
        "standalone_path": str(args.standalone.resolve()),
        "standalone_file_sha256": _sha256(args.standalone),
        "embedded_report_path": str(args.embedded_report.resolve()),
        "embedded_report_file_sha256": _sha256(args.embedded_report),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    if args.require_parity and not result["normalized_route_multiset_equal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
