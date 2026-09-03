#!/usr/bin/env python3
"""Compare one matched V4 result panel against a frozen baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    result = compare_matched_panels(
        _read_object(baseline_path),
        _read_object(candidate_path),
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".md").write_text(
        _markdown(result),
        encoding="utf-8",
    )
    return 0


def compare_matched_panels(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    baseline_rows = _case_index(baseline)
    candidate_rows = _case_index(candidate)
    if not candidate_rows:
        raise ValueError("candidate_panel_has_no_cases")
    missing = sorted(set(candidate_rows) - set(baseline_rows))
    if missing:
        raise ValueError(f"candidate_cases_missing_from_baseline:{','.join(missing)}")

    transition_counts = {
        "newly_stock_closed": 0,
        "stock_closed_preserved": 0,
        "still_stock_open": 0,
        "stock_closed_regressed": 0,
    }
    rows: list[dict[str, Any]] = []
    for case_id in sorted(candidate_rows):
        old = baseline_rows[case_id]
        new = candidate_rows[case_id]
        old_b4 = dict(old.get("gate_summary") or {}).get("B4") is True
        new_b4 = dict(new.get("gate_summary") or {}).get("B4") is True
        transition = _b4_transition(old_b4, new_b4)
        transition_counts[transition] += 1
        old_attempts = _provider_attempt_totals(old)
        new_attempts = _provider_attempt_totals(new)
        old_resources = _numeric_map(old.get("resource_observed"))
        new_resources = _numeric_map(new.get("resource_observed"))
        rows.append(
            {
                "case_id": case_id,
                "target_name": str(new.get("target_name") or ""),
                "baseline": _result_view(old, old_attempts, old_resources),
                "candidate": _result_view(new, new_attempts, new_resources),
                "b4_transition": transition,
                "provider_attempt_delta": _numeric_delta(
                    old_attempts, new_attempts
                ),
                "resource_delta": _numeric_delta(old_resources, new_resources),
            }
        )

    candidate_terminal = all(
        not str(row["candidate"].get("terminal_disposition") or "").startswith(
            "pending_"
        )
        and not str(
            row["candidate"].get("terminal_disposition") or ""
        ).startswith("terminal_incomplete_")
        for row in rows
    )
    baseline_b4 = sum(row["baseline"]["B4"] is True for row in rows)
    candidate_b4 = sum(row["candidate"]["B4"] is True for row in rows)
    body = {
        "schema_version": "v4_matched_panel_comparison.v1",
        "baseline": {
            "path": str(baseline_path or ""),
            "content_sha256": str(baseline.get("content_sha256") or ""),
        },
        "candidate": {
            "path": str(candidate_path or ""),
            "content_sha256": str(candidate.get("content_sha256") or ""),
        },
        "case_count": len(rows),
        "all_candidate_cases_matched": True,
        "all_candidate_cases_terminal": candidate_terminal,
        "performance_claim_eligible": candidate_terminal,
        "b4": {
            "baseline_count": baseline_b4,
            "candidate_count": candidate_b4,
            "count_delta": candidate_b4 - baseline_b4,
            "transition_counts": transition_counts,
        },
        "per_target": rows,
        "semantics": {
            "candidate_case_set_defines_matched_subset": True,
            "baseline_only_cases_are_not_compared": True,
            "pending_or_incomplete_candidate_disables_performance_claim": True,
            "stock_closed_regression_is_never_hidden_by_aggregate_gain": True,
        },
    }
    body["content_sha256"] = _digest(body)
    return body


def _case_index(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in summary.get("per_target") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("panel_row_missing_case_id")
        if case_id in result:
            raise ValueError(f"duplicate_case_id:{case_id}")
        result[case_id] = row
    return result


def _b4_transition(old: bool, new: bool) -> str:
    if not old and new:
        return "newly_stock_closed"
    if old and new:
        return "stock_closed_preserved"
    if old and not new:
        return "stock_closed_regressed"
    return "still_stock_open"


def _provider_attempt_totals(row: Mapping[str, Any]) -> dict[str, int]:
    totals = {
        "attempt_count": 0,
        "raw_nonempty_attempt_count": 0,
        "host_admitted_attempt_count": 0,
        "raw_route_count": 0,
        "output_route_count": 0,
    }
    for attempt in row.get("provider_search_attempts") or []:
        if not isinstance(attempt, Mapping):
            continue
        totals["attempt_count"] += 1
        totals["raw_nonempty_attempt_count"] += attempt.get("raw_solved") is True
        totals["host_admitted_attempt_count"] += (
            attempt.get("host_admitted_solved") is True
        )
        totals["raw_route_count"] += int(
            attempt.get("native_raw_route_count") or 0
        )
        totals["output_route_count"] += int(attempt.get("output_route_count") or 0)
    return totals


def _result_view(
    row: Mapping[str, Any],
    provider_attempts: Mapping[str, int],
    resources: Mapping[str, int | float],
) -> dict[str, Any]:
    gates = dict(row.get("gate_summary") or {})
    trace = dict(row.get("result_action_trace") or {})
    return {
        "status": str(row.get("status") or ""),
        "terminal_disposition": str(row.get("terminal_disposition") or ""),
        "B4": gates.get("B4") is True,
        "recompute_route_closure_count": int(
            trace.get("recompute_route_closure_count") or 0
        ),
        "guided_before_first_route_closure": int(
            trace.get("guided_before_first_route_closure") or 0
        ),
        "guided_after_first_route_closure": int(
            trace.get("guided_after_first_route_closure") or 0
        ),
        "campaign_termination": str(trace.get("campaign_termination") or ""),
        "provider_attempts": dict(provider_attempts),
        "resources": dict(resources),
        "recovery": dict(row.get("runtime_recovery") or {}),
        "report_path": str(row.get("report_path") or ""),
    }


def _numeric_map(value: Any) -> dict[str, int | float]:
    return {
        str(key): item
        for key, item in dict(value or {}).items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _numeric_delta(
    old: Mapping[str, int | float],
    new: Mapping[str, int | float],
) -> dict[str, int | float]:
    return {
        key: new.get(key, 0) - old.get(key, 0)
        for key in sorted(set(old) | set(new))
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"summary_not_object:{path}")
    result = dict(value)
    if result.get("schema_version") == "v4_blind_panel_summary.v3":
        material = dict(result)
        observed = str(material.pop("content_sha256", ""))
        if not observed or observed != _digest(material):
            raise ValueError(f"summary_digest_invalid:{path}")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _markdown(result: Mapping[str, Any]) -> str:
    b4 = dict(result.get("b4") or {})
    lines = [
        "# V4 Matched Panel Comparison",
        "",
        f"- Cases: {result.get('case_count', 0)}",
        f"- All candidate cases terminal: "
        f"{result.get('all_candidate_cases_terminal') is True}",
        f"- Performance claim eligible: "
        f"{result.get('performance_claim_eligible') is True}",
        f"- Root B4: {b4.get('baseline_count', 0)} -> "
        f"{b4.get('candidate_count', 0)} "
        f"(delta {int(b4.get('count_delta') or 0):+d})",
        "",
        "| B4 transition | Targets |",
        "| --- | ---: |",
    ]
    for transition, count in dict(b4.get("transition_counts") or {}).items():
        lines.append(f"| {transition} | {count} |")
    lines.extend(
        [
            "",
            "| Case | B4 transition | Old B4 | New B4 | Old/New closure actions | "
            "Old/New guided after closure | Old/New termination |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for raw in result.get("per_target") or []:
        row = dict(raw)
        old = dict(row.get("baseline") or {})
        new = dict(row.get("candidate") or {})
        lines.append(
            f"| {row.get('case_id', '')} | {row.get('b4_transition', '')} | "
            f"{int(old.get('B4') is True)} | {int(new.get('B4') is True)} | "
            f"{old.get('recompute_route_closure_count', 0)} / "
            f"{new.get('recompute_route_closure_count', 0)} | "
            f"{old.get('guided_after_first_route_closure', 0)} / "
            f"{new.get('guided_after_first_route_closure', 0)} | "
            f"{old.get('campaign_termination', '')} / "
            f"{new.get('campaign_termination', '')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
