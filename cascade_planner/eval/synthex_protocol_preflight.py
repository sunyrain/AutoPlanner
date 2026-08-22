"""Launch-bound validation for a paper-comparable SynthEx head-to-head run.

This is deliberately a direct read of the frozen protocol, target-only
manifest, canonical runtime defaults and stock index.  It does not create a
second readiness artifact and it grants no route or reaction authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
)


PREFLIGHT_SCHEMA = "synthex_head_to_head_preflight.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_synthex_head_to_head_protocol(
    *,
    protocol_path: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path,
    model: str,
    reasoning_effort: str,
    execution_profile: str,
    strategy_portfolio_mode: str = "paper_independent",
    benchmark_stock_index: str | Path | None = None,
    benchmark_stock_name: str = "",
    matched_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact execution contract immediately before launch."""

    defaults = dict(matched_defaults or SYNTHEX_MATCHED_PROFILE_DEFAULTS)
    root = Path(repository_root).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    issues: list[dict[str, Any]] = []

    protocol = _load_object(protocol_file, "protocol", issues)
    manifest = _load_object(manifest_file, "manifest", issues)
    execution = dict(protocol.get("execution_contract") or {})
    planner_budget = dict(protocol.get("planner_budget") or {})
    benchmark = dict(protocol.get("benchmark") or {})
    source_snapshot = dict(protocol.get("source_snapshot") or {})
    stock_binding = dict(protocol.get("stock_binding") or {})
    enzyme_extension = dict(protocol.get("enzyme_extension") or {})
    requested_portfolio = str(strategy_portfolio_mode or "paper_independent")
    if requested_portfolio == "auto":
        requested_portfolio = "paper_independent"

    # This frozen protocol is the isolated reach arm.  ``paper_synthex`` stays
    # available only for replaying historical runs and must not launch a new
    # experiment under this stricter protocol.
    actual_profile = str(execution.get("execution_profile") or "")
    _expect(issues, "execution_profile", actual_profile, "paper_matched_reach")
    _expect(
        issues,
        "launch.execution_profile",
        str(execution_profile or ""),
        "paper_matched_reach",
    )
    _expect(issues, "model", execution.get("model"), model)
    _expect(issues, "reasoning_effort", execution.get("reasoning_effort"), reasoning_effort)
    _expect(
        issues,
        "strategy_branches",
        execution.get("strategy_branches"),
        defaults.get("strategy_branches"),
    )
    _expect(
        issues,
        "strategy_tree_engine",
        execution.get("strategy_tree_engine"),
        defaults.get("strategy_tree_engine"),
    )
    _expect(
        issues,
        "strategy_node_selection",
        execution.get("strategy_node_selection"),
        "AiZynthFinder MCTS/UCB",
    )
    _expect(issues, "enzyme_extension.enabled", enzyme_extension.get("enabled"), False)
    _expect(
        issues,
        "strategy_portfolio_mode",
        requested_portfolio,
        defaults.get("strategy_portfolio_mode"),
    )
    _expect(
        issues,
        "execution_contract.strategy_portfolio_mode",
        execution.get("strategy_portfolio_mode"),
        defaults.get("strategy_portfolio_mode"),
    )
    _expect(
        issues,
        "strategy_branch_workers",
        execution.get("strategy_branch_workers"),
        defaults.get("strategy_branch_workers"),
    )
    _expect(
        issues,
        "route_builder_calls_per_branch",
        execution.get("route_builder_calls_per_branch"),
        defaults.get("node_expansions_per_branch"),
    )
    _expect(
        issues,
        "reactionjson_candidates_per_node",
        execution.get("reactionjson_candidates_per_node"),
        defaults.get("reactionjson_candidates_per_node"),
    )
    expected_route_calls = int(defaults.get("strategy_branches") or 0) * int(
        defaults.get("node_expansions_per_branch") or 0
    )
    _expect(
        issues,
        "maximum_route_builder_calls",
        execution.get("maximum_route_builder_calls"),
        expected_route_calls,
    )
    _expect(
        issues,
        "maximum_node_prompt_bytes",
        execution.get("maximum_node_prompt_bytes"),
        defaults.get("max_node_prompt_bytes"),
    )
    _expect(
        issues,
        "stop_on_first_stock_closed_branch",
        execution.get("stop_on_first_stock_closed_branch"),
        defaults.get("stop_on_first_stock_closed_branch"),
    )
    _expect(issues, "route_builder_is_one_open_node_per_call", execution.get("route_builder_is_one_open_node_per_call"), True)
    _expect(issues, "route_builder_node_prompt_is_not_complete_route_json", execution.get("route_builder_node_prompt_is_not_complete_route_json"), True)
    _expect(issues, "host_compiles_reactionjson_into_routejson", execution.get("host_compiles_reactionjson_into_routejson"), True)
    _expect(issues, "require_complete_route_json_admission", execution.get("require_complete_route_json_admission"), True)
    _expect(issues, "partial_route_ingestion_allowed", execution.get("partial_route_ingestion_allowed"), False)
    _expect(issues, "global_initial_architecture_generation_allowed", execution.get("global_initial_architecture_generation_allowed"), False)
    _expect(issues, "complete_routejson_before_critic_editor", execution.get("complete_routejson_before_critic_editor"), True)
    _expect(issues, "critic_receives_host_mapped_route_and_conditions", execution.get("critic_receives_host_mapped_route_and_conditions"), True)
    _expect(issues, "critic_forward_simulates_every_step", execution.get("critic_forward_simulates_every_step"), True)
    _expect(
        issues,
        "editor_capabilities",
        execution.get("editor_capabilities"),
        [
            "reorder",
            "insert",
            "delete",
            "change_conditions",
            "change_functional_groups",
            "replace_dependency_closed_subroute",
        ],
    )
    _expect(issues, "editor_complete_routejson_output_allowed", execution.get("editor_complete_routejson_output_allowed"), True)
    _expect(issues, "edited_route_recompiled_as_target_rooted_dag", execution.get("edited_route_recompiled_as_target_rooted_dag"), True)
    _expect(issues, "actual_policy_call_authority", execution.get("actual_policy_call_authority"), "worker_journal")
    _expect(
        issues,
        "primary_report_fields",
        execution.get("primary_report_fields"),
        ["paper_reach", "paper_equivalent_solved"],
    )
    _expect(
        issues,
        "critic_editor_repair_rounds",
        execution.get("critic_editor_repair_rounds"),
        defaults.get("route_local_repair_rounds"),
    )
    short_tail = dict(execution.get("short_tail") or {})
    _expect(
        issues,
        "short_tail.engine",
        short_tail.get("engine"),
        defaults.get("short_tail_engine"),
    )
    _expect(issues, "short_tail.depth", short_tail.get("depth"), defaults.get("short_tail_steps"))
    _expect(issues, "short_tail.iterations", short_tail.get("iterations"), defaults.get("short_tail_iterations"))
    _expect(issues, "short_tail.timeout_s", short_tail.get("timeout_s"), defaults.get("short_tail_timeout_s"))
    _expect(issues, "short_tail.target_reachable", short_tail.get("applied_only_to_distinct_target_reachable_open_leaves"), True)
    _expect(issues, "short_tail.materialize_first", short_tail.get("materialize_validate_stock_before_more_frontier_search"), True)

    budget_expectations = {
        "max_accepted_expansions": "max_accepted_expansions",
        "max_attempt_runs": "max_attempt_runs",
        "max_model_invocations": "max_model_invocations",
        "max_model_wall_time_s": "max_model_wall_time_s",
        "max_run_wall_time_s": "max_run_wall_time_s",
        "max_total_input_tokens": "max_input_tokens",
        "max_total_output_tokens": "max_output_tokens",
        "max_total_tasks": "max_total_tasks",
        "node_call_timeout_s": "node_call_timeout_s",
        "critic_call_timeout_s": "critic_call_timeout_s",
    }
    for protocol_key, default_key in budget_expectations.items():
        _expect(
            issues,
            f"planner_budget.{protocol_key}",
            planner_budget.get(protocol_key),
            defaults.get(default_key),
        )
    _expect(
        issues,
        "planner_budget.maximum_route_builder_calls",
        planner_budget.get("maximum_route_builder_calls"),
        expected_route_calls,
    )

    cases = manifest.get("cases") if isinstance(manifest.get("cases"), list) else []
    _expect(issues, "benchmark.target_count", benchmark.get("target_count"), len(cases))
    if len(cases) != 3:
        _issue(issues, "manifest_target_count_not_three", actual=len(cases), expected=3)
    case_ids = [str(row.get("case_id") or "") for row in cases if isinstance(row, Mapping)]
    if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        _issue(issues, "manifest_case_ids_invalid", actual=case_ids)
    target_names = [str(row.get("target_name") or "") for row in cases if isinstance(row, Mapping)]
    if benchmark.get("target_names_are_hidden_from_live_planner") is True and any(
        "opaque" not in name.casefold() for name in target_names
    ):
        _issue(issues, "manifest_target_name_not_opaque", actual=target_names)

    bindings = [
        dict(row)
        for row in source_snapshot.get("target_bindings") or []
        if isinstance(row, Mapping)
    ]
    bound_case_ids = [str(row.get("case_id") or "") for row in bindings]
    if sorted(bound_case_ids) != sorted(case_ids):
        _issue(
            issues,
            "source_target_binding_case_mismatch",
            actual=sorted(bound_case_ids),
            expected=sorted(case_ids),
        )
    bound_route_ids: list[str] = []
    for row in bindings:
        source_target_id = str(row.get("source_target_id") or "")
        route_ids = [str(value) for value in row.get("route_ids") or [] if str(value)]
        if not source_target_id or len(route_ids) != 3:
            _issue(
                issues,
                "source_target_binding_invalid",
                case_id=str(row.get("case_id") or ""),
                source_target_id=source_target_id,
                route_ids=route_ids,
            )
        if any(source_target_id.casefold() in name.casefold() for name in target_names):
            _issue(
                issues,
                "source_target_identity_leaked_to_live_name",
                source_target_id=source_target_id,
            )
        bound_route_ids.extend(route_ids)
    declared_route_ids = [
        str(value) for value in source_snapshot.get("route_ids") or [] if str(value)
    ]
    if sorted(bound_route_ids) != sorted(declared_route_ids):
        _issue(
            issues,
            "source_route_binding_mismatch",
            actual=sorted(bound_route_ids),
            expected=sorted(declared_route_ids),
        )

    manifest_budget_expected = {
        "max_accepted_expansions": defaults.get("max_accepted_expansions"),
        "max_attempt_runs": defaults.get("max_attempt_runs"),
        "max_model_invocations": defaults.get("max_model_invocations"),
        "max_prompt_context_bytes": defaults.get("max_prompt_context_bytes"),
        "max_total_input_tokens": defaults.get("max_input_tokens"),
        "max_total_output_tokens": defaults.get("max_output_tokens"),
        "max_total_wall_time_s": defaults.get("max_model_wall_time_s"),
    }
    for case in cases:
        if not isinstance(case, Mapping):
            _issue(issues, "manifest_case_not_object")
            continue
        case_id = str(case.get("case_id") or "")
        case_budget = dict(case.get("budget") or {})
        for key, expected in manifest_budget_expected.items():
            if case_budget.get(key) != expected:
                _issue(
                    issues,
                    "manifest_case_budget_mismatch",
                    case_id=case_id,
                    field=key,
                    actual=case_budget.get(key),
                    expected=expected,
                )
        acceptance = dict(case.get("acceptance") or {})
        if acceptance.get("stock_boundary") != "benchmark_search":
            _issue(
                issues,
                "manifest_stock_boundary_mismatch",
                case_id=case_id,
                actual=acceptance.get("stock_boundary"),
                expected="benchmark_search",
            )

    stock = _validate_stock(
        root=root,
        binding=stock_binding,
        explicit_index=benchmark_stock_index,
        explicit_name=benchmark_stock_name,
        defaults=defaults,
        issues=issues,
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "ready_for_paid_experiment": not issues,
        "protocol_path": str(protocol_file),
        "manifest_path": str(manifest_file),
        "target_count": len(cases),
        "strategy_portfolio_mode": requested_portfolio,
        "stock": stock,
        "issues": issues,
        "semantics": {
            "launch_boundary_only": True,
            "creates_no_readiness_artifact": True,
            "grants_no_reaction_or_route_authority": True,
            "historical_reports_are_not_execution_authority": True,
        },
    }


def _validate_stock(
    *,
    root: Path,
    binding: Mapping[str, Any],
    explicit_index: str | Path | None,
    explicit_name: str,
    defaults: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_path = str(binding.get("index_path") or "").strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    catalog_name = str(binding.get("catalog_name") or "")
    expected_identity = str(defaults.get("paper_reference_stock_identity_key") or "")
    expected_count = int(
        defaults.get("paper_reference_stock_unique_member_count")
        or defaults.get("paper_reference_stock_member_count")
        or 0
    )
    paper_declared_count = int(
        defaults.get("paper_reference_stock_declared_entry_count")
        or expected_count
    )
    expected_catalog = str(defaults.get("paper_reference_stock_catalog_name") or "")
    _expect(issues, "stock.catalog_name", catalog_name, expected_catalog)
    _expect(issues, "stock.identity_key", binding.get("identity_key"), expected_identity)
    _expect(
        issues,
        "stock.unique_member_count",
        binding.get("unique_member_count"),
        expected_count,
    )
    _expect(
        issues,
        "stock.paper_declared_entry_count",
        binding.get("paper_declared_entry_count"),
        paper_declared_count,
    )
    reconciliation = dict(binding.get("count_reconciliation") or {})
    expected_reconciliation = {
        "zinc_unique_full_inchikeys": int(
            defaults.get("paper_reference_stock_zinc_unique_count") or 0
        ),
        "emolecules_input_rows": int(
            defaults.get("paper_reference_stock_emolecules_input_rows") or 0
        ),
        "emolecules_valid_full_inchikey_rows": int(
            defaults.get("paper_reference_stock_emolecules_valid_rows") or 0
        ),
        "emolecules_unique_full_inchikeys": int(
            defaults.get("paper_reference_stock_emolecules_unique_count") or 0
        ),
        "cross_source_overlap_full_inchikeys": int(
            defaults.get("paper_reference_stock_cross_source_overlap_count") or 0
        ),
        "redundant_or_invalid_emolecules_rows": int(
            defaults.get("paper_reference_stock_redundant_or_invalid_rows") or 0
        ),
    }
    for field, expected in expected_reconciliation.items():
        _expect(
            issues,
            f"stock.count_reconciliation.{field}",
            reconciliation.get(field),
            expected,
        )
    reconciled_unique = (
        expected_reconciliation["zinc_unique_full_inchikeys"]
        + expected_reconciliation["emolecules_unique_full_inchikeys"]
        - expected_reconciliation["cross_source_overlap_full_inchikeys"]
    )
    reconciled_declared = (
        reconciled_unique
        + expected_reconciliation["redundant_or_invalid_emolecules_rows"]
    )
    if reconciled_unique != expected_count or reconciled_declared != paper_declared_count:
        _issue(
            issues,
            "stock_count_reconciliation_mismatch",
            actual_unique=reconciled_unique,
            expected_unique=expected_count,
            actual_declared=reconciled_declared,
            expected_declared=paper_declared_count,
        )
    _expect(issues, "stock.required_for_paper_comparison", binding.get("required_for_paper_comparison"), True)
    _expect(issues, "stock.fallback_stock_allowed", binding.get("fallback_stock_allowed"), False)
    if explicit_index is not None and Path(explicit_index).expanduser().resolve() != path:
        _issue(
            issues,
            "explicit_benchmark_stock_path_mismatch",
            actual=str(Path(explicit_index).expanduser().resolve()),
            expected=str(path),
        )
    if str(explicit_name or "").strip() and str(explicit_name).strip() != catalog_name:
        _issue(
            issues,
            "explicit_benchmark_stock_name_mismatch",
            actual=str(explicit_name).strip(),
            expected=catalog_name,
        )
    result: dict[str, Any] = {
        "index_path": str(path),
        "catalog_name": catalog_name,
        "identity_key": str(binding.get("identity_key") or ""),
        "expected_unique_member_count": expected_count,
        "paper_declared_entry_count": paper_declared_count,
        "count_reconciliation": reconciliation,
        "index_sha256": "",
        "metadata": {},
    }
    if not path.is_file():
        _issue(issues, "stock_index_missing", actual=str(path))
        return result
    try:
        with sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30.0
        ) as connection:
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key,value FROM metadata ORDER BY key"
                ).fetchall()
            }
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(stock)").fetchall()
            }
            actual_count = int(
                connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
            )
    except sqlite3.Error as exc:
        _issue(issues, "stock_index_unreadable", actual=f"{type(exc).__name__}:{exc}")
        return result
    result["metadata"] = metadata
    result["actual_member_count"] = actual_count
    _expect(issues, "stock.metadata.schema_version", metadata.get("schema_version"), "frozen_benchmark_stock_index.v1")
    _expect(issues, "stock.metadata.catalog_name", metadata.get("catalog_name"), catalog_name)
    _expect(issues, "stock.metadata.identity_key", metadata.get("identity_key"), expected_identity)
    _expect(issues, "stock.metadata.complete", metadata.get("complete"), "true")
    _expect(
        issues,
        "stock.metadata.member_count",
        _as_int(metadata.get("member_count")),
        expected_count,
    )
    _expect(issues, "stock.actual_unique_member_count", actual_count, expected_count)
    if expected_identity not in columns:
        _issue(
            issues,
            "stock_identity_column_missing",
            actual=sorted(columns),
            expected=expected_identity,
        )
    actual_sha256 = _file_sha256(path)
    result["index_sha256"] = actual_sha256
    declared_sha256 = str(binding.get("index_sha256") or "").strip().lower()
    if not _SHA256.fullmatch(declared_sha256):
        _issue(
            issues,
            "stock_index_sha256_not_frozen",
            actual=declared_sha256,
            expected=actual_sha256,
        )
    elif declared_sha256 != actual_sha256:
        _issue(
            issues,
            "stock_index_sha256_mismatch",
            actual=actual_sha256,
            expected=declared_sha256,
        )
    return result


def _load_object(path: Path, kind: str, issues: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _issue(issues, f"{kind}_unreadable", actual=f"{type(exc).__name__}:{exc}")
        return {}
    if not isinstance(value, Mapping):
        _issue(issues, f"{kind}_not_object")
        return {}
    return dict(value)


def _expect(
    issues: list[dict[str, Any]], field: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        _issue(
            issues,
            "protocol_contract_mismatch",
            field=field,
            actual=actual,
            expected=expected,
        )


def _issue(issues: list[dict[str, Any]], code: str, **detail: Any) -> None:
    issues.append({"code": code, **detail})


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--model", default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["model"])
    parser.add_argument(
        "--reasoning-effort",
        default=SYNTHEX_MATCHED_PROFILE_DEFAULTS["reasoning_effort"],
    )
    parser.add_argument("--execution-profile", default="paper_synthex")
    parser.add_argument(
        "--strategy-portfolio-mode",
        choices=("paper_independent", "enzyme_advantage", "autoplanner_strategy_v2"),
        default="paper_independent",
    )
    parser.add_argument("--benchmark-stock-index")
    parser.add_argument("--benchmark-stock-name", default="")
    args = parser.parse_args(argv)
    result = validate_synthex_head_to_head_protocol(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        execution_profile=args.execution_profile,
        strategy_portfolio_mode=args.strategy_portfolio_mode,
        benchmark_stock_index=args.benchmark_stock_index,
        benchmark_stock_name=args.benchmark_stock_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_paid_experiment"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
