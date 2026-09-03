"""Freeze a blind, matched strategy-to-experiment closure pilot.

The planner-facing manifest contains targets only.  Provider route documents,
target names, synonyms, and route-derived leakage needles live in a separate
evaluator pack and must never be mounted into a live planning process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from rdkit import Chem

from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BLIND_MANIFEST_SCHEMA,
    BlindCase,
    canonical_smiles,
)
from cascade_planner.application.external_strategy_routes import (
    ExternalStrategyRouteError,
    compile_external_strategy_route_bundle,
)

STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA = "strategy_closure_evaluator_pack.v1"
STRATEGY_CLOSURE_PILOT_PROTOCOL_SCHEMA = "strategy_closure_pilot_protocol.v1"
BLIND_LEAKAGE_AUDIT_PACK_SCHEMA = "blind_leakage_audit_pack.v1"
SELECTION_ALGORITHM = "first_unique_canonical_target_in_public_index_order.v1"
DEFAULT_ACCEPTANCE = {
    "minimum_complete_routes": 1,
    "minimum_edge_proof_level": 2,
    "minimum_independent_source_groups": 1,
    "minimum_planning_route_steps": 0,
    "stock_boundary": "benchmark_search",
}
DEFAULT_BUDGET = {
    "max_accepted_expansions": 96,
    "max_attempt_runs": 192,
    "max_model_invocations": 3,
    "max_prompt_context_bytes": 160_000,
    "max_total_input_tokens": 1_200_000,
    "max_total_output_tokens": 200_000,
    "max_total_wall_time_s": 1_800,
}


class StrategyClosurePilotError(ValueError):
    """The requested pilot cannot be frozen without selection or leakage drift."""


def compile_strategy_closure_pilot(
    *,
    index_rows: Sequence[Mapping[str, Any]],
    route_documents: Mapping[str, Mapping[str, Any]],
    source_snapshot: Mapping[str, Any],
    target_count: int = 20,
    frozen_at: str,
    stock_binding: Mapping[str, Any] | None = None,
    selected_target_smiles: Sequence[str] | None = None,
    selection_algorithm: str = SELECTION_ALGORITHM,
    selection_audit: Sequence[Mapping[str, Any]] = (),
    case_id_prefix: str = "synthatlas",
) -> dict[str, Any]:
    """Compile target-only and evaluator-only artifacts from one frozen index."""

    if isinstance(target_count, bool) or not 1 <= int(target_count) <= 1_098:
        raise StrategyClosurePilotError("strategy_pilot_target_count_invalid")
    if not str(frozen_at or "").strip():
        raise StrategyClosurePilotError("strategy_pilot_frozen_at_required")
    rows = [
        _json_mapping(row, "strategy_pilot_index_row_invalid") for row in index_rows
    ]
    if not rows:
        raise StrategyClosurePilotError("strategy_pilot_index_empty")
    selected_targets = _select_targets(
        rows,
        target_count=int(target_count),
        selected_target_smiles=selected_target_smiles,
    )
    selection_algorithm = str(selection_algorithm or "").strip()
    if not selection_algorithm:
        raise StrategyClosurePilotError("strategy_pilot_selection_algorithm_required")
    case_id_prefix = str(case_id_prefix or "").strip()
    if not case_id_prefix or len(case_id_prefix) > 48:
        raise StrategyClosurePilotError("strategy_pilot_case_id_prefix_invalid")
    audit_rows = [
        _json_mapping(row, "strategy_pilot_selection_audit_invalid")
        for row in selection_audit
    ]
    excluded_count = sum(row.get("accepted") is False for row in audit_rows)
    selected_set = {row["target_smiles"] for row in selected_targets}
    selected_rows = [
        _normalize_index_row(row)
        for row in rows
        if canonical_smiles(row.get("target_smiles")) in selected_set
    ]
    route_ids = [row["route_id"] for row in selected_rows]
    if len(route_ids) != len(set(route_ids)):
        raise StrategyClosurePilotError("strategy_pilot_route_id_duplicate")
    missing = sorted(set(route_ids) - set(route_documents))
    extra = sorted(set(route_documents) - set(route_ids))
    if missing:
        raise StrategyClosurePilotError(
            "strategy_pilot_route_documents_missing:" + ",".join(missing)
        )
    if extra:
        raise StrategyClosurePilotError(
            "strategy_pilot_route_documents_extra:" + ",".join(extra)
        )

    cases: list[dict[str, Any]] = []
    blind_cases: list[dict[str, Any]] = []
    c0_passed = 0
    c0_failures: dict[str, int] = {}
    for index, target in enumerate(selected_targets, start=1):
        case_id = (
            f"{case_id_prefix}-{index:03d}-{_digest(target['target_smiles'])[:10]}"
        )
        blind_case = BlindCase(
            case_id=case_id,
            target_name=f"opaque benchmark target {index:03d}",
            target_smiles=target["target_smiles"],
            acceptance=DEFAULT_ACCEPTANCE,
            budget=DEFAULT_BUDGET,
            schema_version=BLIND_CASE_SCHEMA,
        ).to_dict()
        blind_cases.append(blind_case)
        case_rows = [
            row
            for row in selected_rows
            if row["target_smiles"] == target["target_smiles"]
        ]
        routes: list[dict[str, Any]] = []
        needles: set[str] = set()
        for row in case_rows:
            route_id = row["route_id"]
            document = _json_mapping(
                route_documents[route_id], "strategy_pilot_route_document_invalid"
            )
            if str(document.get("id") or "") != route_id:
                raise StrategyClosurePilotError(
                    f"strategy_pilot_route_document_id_mismatch:{route_id}"
                )
            if (
                canonical_smiles(document.get("target_smiles"))
                != target["target_smiles"]
            ):
                raise StrategyClosurePilotError(
                    f"strategy_pilot_route_document_target_mismatch:{route_id}"
                )
            c0 = _preflight_route(document, target_smiles=target["target_smiles"])
            if c0["accepted"]:
                c0_passed += 1
            else:
                reason = str(c0["reason"])
                c0_failures[reason] = c0_failures.get(reason, 0) + 1
            needles.update(_route_structure_needles(document))
            routes.append(
                {
                    "route_id": route_id,
                    "index_metadata": row["metadata"],
                    "document_sha256": _digest(document),
                    "host_c0_preflight": c0,
                    "document": document,
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "source_target_name": target["source_target_name"],
                "target_smiles": target["target_smiles"],
                "target_synonyms": [target["source_target_name"]],
                "key_intermediate_smiles": sorted(needles - {target["target_smiles"]}),
                "routes": routes,
            }
        )

    target_manifest = {
        "schema_version": BLIND_MANIFEST_SCHEMA,
        "cases": blind_cases,
    }
    evaluator_core = {
        "schema_version": STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA,
        "frozen_at": str(frozen_at).strip(),
        "source_snapshot": _json_mapping(
            source_snapshot, "strategy_pilot_source_snapshot_invalid"
        ),
        "selection": {
            "algorithm": selection_algorithm,
            "target_count": len(cases),
            "route_count": len(selected_rows),
            "manual_exclusions": 0,
            "pre_run_knowledge_exclusions": excluded_count,
            "audit": audit_rows,
            "all_public_variants_for_selected_targets_retained": True,
        },
        "cases": cases,
        "authority": {
            "planner_may_read_this_pack": False,
            "external_self_reported_solved_is_proof": False,
            "external_conditions_are_exact_procedure_evidence": False,
            "external_routes_are_reference_answers": True,
        },
    }
    evaluator_pack = _with_digest(evaluator_core)
    route_count = len(selected_rows)
    protocol_core = {
        "schema_version": STRATEGY_CLOSURE_PILOT_PROTOCOL_SCHEMA,
        "status": "frozen_not_executed",
        "frozen_at": str(frozen_at).strip(),
        "scope": {
            "kind": "exploratory_matched_pilot",
            "target_count": len(cases),
            "route_variant_count": route_count,
            "population_claim_allowed": False,
            "selection_algorithm": selection_algorithm,
            "manual_exclusions": 0,
            "pre_run_knowledge_exclusions": excluded_count,
            "selection_audit_content_sha256": _digest(audit_rows),
        },
        "bindings": {
            "target_manifest_content_sha256": _digest(target_manifest),
            "evaluator_pack_content_sha256": evaluator_pack["content_sha256"],
            "stock_oracle": _json_mapping(
                stock_binding or {}, "strategy_pilot_stock_binding_invalid"
            ),
            "source_snapshot": _public_source_binding(source_snapshot),
        },
        "arms": [
            {
                "arm_id": "external_snapshot_only",
                "input": "frozen public strategic routes",
                "generation_cost_comparable": False,
                "host_closure_pipeline_identical": True,
            },
            {
                "arm_id": "codex_only",
                "input": "target-only manifest",
                "generation_cost_comparable": True,
                "host_closure_pipeline_identical": True,
            },
            {
                "arm_id": "chemenzy_only",
                "input": "target-only manifest",
                "generation_cost_comparable": True,
                "host_closure_pipeline_identical": True,
            },
            {
                "arm_id": "unified_adaptive",
                "input": "target-only manifest",
                "generation_cost_comparable": True,
                "host_closure_pipeline_identical": True,
            },
        ],
        "budget": {
            "planner_facing_per_target": DEFAULT_BUDGET,
            "external_snapshot_generation_cost": "not_observed_and_not_imputed",
            "external_arm_valid_comparison": "route_quality_and_host_closure_only",
            "live_arm_compute_comparison_requires_full_resource_ledger": True,
        },
        "closure_levels": {
            "C0": "connected provider route structure",
            "C1": "canonical host materialization",
            "C2": "host reaction validation",
            "C3": "exact source and procedure binding",
            "C4": "complete exact conditions",
            "C5": "frozen stock or procurement closure",
            "C6": "bound experimental program result",
        },
        "preflight": {
            "external_route_count": route_count,
            "host_c0_passed": c0_passed,
            "host_c0_failed": route_count - c0_passed,
            "failure_reasons": dict(sorted(c0_failures.items())),
            "execution_blocked_if_c0_failure_rate_exceeds": 0.20,
            "c0_gate_passed": (route_count - c0_passed) / route_count <= 0.20,
        },
        "blindness": {
            "planner_receives_target_manifest_only": True,
            "evaluator_pack_must_be_outside_tracked_repository": True,
            "target_names_are_opaque": True,
            "synonyms_and_route_intermediates_are_evaluator_only_leakage_needles": True,
            "old_run_cache_forbidden": True,
        },
        "stopping_rule": {
            "complete_all_declared_arms_for_all_frozen_targets": True,
            "frozen_target_count": len(cases),
            "no_result_conditioned_target_removal": True,
            "no_target_specific_rules_after_freeze": True,
            "do_not_scale_before_pilot_failure_taxonomy": True,
        },
        "claim_boundary": (
            "The public snapshot arm is a fixed route-input baseline, not a live "
            "SynthEx efficiency reproduction. No C3/C4/C6 advantage may be claimed "
            "before all frozen arms are executed and audited."
        ),
    }
    protocol = _with_digest(protocol_core)
    return {
        "target_manifest": target_manifest,
        "evaluator_pack": evaluator_pack,
        "protocol": protocol,
    }


def external_bundle_for_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build an import bundle inside the evaluator process, never the planner."""

    value = _json_mapping(case, "strategy_pilot_evaluator_case_invalid")
    routes = value.get("routes")
    if not isinstance(routes, list) or not routes:
        raise StrategyClosurePilotError("strategy_pilot_evaluator_routes_missing")
    return {
        "schema_version": "external_strategy_route_bundle.v1",
        "provider": "synthatlas-public-snapshot",
        "target_smiles": str(value.get("target_smiles") or ""),
        "routes": [dict(row["document"]) for row in routes],
    }


def compile_strategy_closure_leakage_pack(
    *, manifest_file_sha256: str, evaluator_pack: Mapping[str, Any]
) -> dict[str, Any]:
    """Project evaluator answers into the blind supervisor's needle contract."""

    manifest_digest = str(manifest_file_sha256 or "").strip().lower()
    if len(manifest_digest) != 64 or any(
        value not in "0123456789abcdef" for value in manifest_digest
    ):
        raise StrategyClosurePilotError("strategy_pilot_manifest_digest_invalid")
    evaluator = _json_mapping(evaluator_pack, "strategy_pilot_evaluator_pack_invalid")
    if evaluator.get("schema_version") != STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA:
        raise StrategyClosurePilotError("strategy_pilot_evaluator_pack_schema_invalid")
    cases: dict[str, Any] = {}
    for raw in evaluator.get("cases") or []:
        case = _json_mapping(raw, "strategy_pilot_evaluator_case_invalid")
        case_id = str(case.get("case_id") or "")
        name = str(case.get("source_target_name") or "").strip()
        intermediates = sorted(
            {
                canonical
                for value in case.get("key_intermediate_smiles") or []
                if (canonical := canonical_smiles(value))
                and _heavy_atom_count(canonical) >= 8
                and canonical != str(case.get("target_smiles") or "")
            }
        )
        if not case_id or not intermediates:
            raise StrategyClosurePilotError(
                f"strategy_pilot_leakage_case_incomplete:{case_id}"
            )
        unnamed = name.casefold() in {"", "not named", "unnamed", "unknown"}
        cases[case_id] = {
            "target_synonyms": [] if unnamed else [name],
            "target_synonym_not_applicable_reason": (
                "public source provides no usable target name" if unnamed else ""
            ),
            "key_intermediate_smiles": intermediates,
        }
    result = {
        "schema_version": BLIND_LEAKAGE_AUDIT_PACK_SCHEMA,
        "manifest_sha256": manifest_digest,
        "cases": cases,
        "semantics": {
            "evaluator_only": True,
            "never_passed_to_planner_subprocess": True,
            "contains_public_route_derived_intermediates": True,
            "common_fragments_below_eight_heavy_atoms_excluded": True,
            "must_remain_outside_the_tracked_repository": True,
        },
    }
    return _with_digest(result)


def _select_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
    selected_target_smiles: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    requested = (
        [canonical_smiles(value) for value in selected_target_smiles]
        if selected_target_smiles is not None
        else []
    )
    if requested and (
        len(requested) != target_count
        or any(not value for value in requested)
        or len(requested) != len(set(requested))
    ):
        raise StrategyClosurePilotError("strategy_pilot_selected_targets_invalid")
    requested_set = set(requested)
    names_by_target: dict[str, str] = {}
    for row in rows:
        target = canonical_smiles(row.get("target_smiles"))
        if target and target not in names_by_target:
            names_by_target[target] = str(row.get("name") or "unnamed target").strip()
    if requested:
        if not requested_set.issubset(names_by_target):
            raise StrategyClosurePilotError("strategy_pilot_selected_target_missing")
        return [
            {
                "target_smiles": target,
                "source_target_name": names_by_target[target],
            }
            for target in requested
        ]
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        target = canonical_smiles(row.get("target_smiles"))
        if not target:
            raise StrategyClosurePilotError("strategy_pilot_index_target_invalid")
        if target in seen:
            continue
        seen.add(target)
        selected.append(
            {
                "target_smiles": target,
                "source_target_name": str(row.get("name") or "unnamed target").strip(),
            }
        )
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise StrategyClosurePilotError("strategy_pilot_unique_targets_insufficient")
    return selected


def _normalize_index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    route_id = str(row.get("id") or "").strip()
    target = canonical_smiles(row.get("target_smiles"))
    if not route_id or not target:
        raise StrategyClosurePilotError("strategy_pilot_index_identity_invalid")
    metadata = {
        key: row[key]
        for key in (
            "topology",
            "total_steps",
            "longest_linear_path",
            "solved",
            "productive",
            "feasibility",
            "stereo_risk",
            "variants",
        )
        if key in row
    }
    return {"route_id": route_id, "target_smiles": target, "metadata": metadata}


def _preflight_route(
    document: Mapping[str, Any], *, target_smiles: str
) -> dict[str, Any]:
    try:
        result = compile_external_strategy_route_bundle(
            {
                "schema_version": "external_strategy_route_bundle.v1",
                "provider": "synthatlas-public-snapshot",
                "target_smiles": target_smiles,
                "routes": [dict(document)],
            },
            expected_target_smiles=target_smiles,
        )
    except ExternalStrategyRouteError as exc:
        return {"accepted": False, "reason": str(exc)}
    receipt = result["receipt"]
    return {
        "accepted": True,
        "reason": "",
        "step_count": int(receipt["step_count"]),
        "source_payload_sha256": str(receipt["source_payload_sha256"]),
    }


def _route_structure_needles(document: Mapping[str, Any]) -> set[str]:
    needles: set[str] = set()
    for step in document.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        reaction = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
        parts = reaction.split(">")
        if len(parts) != 3:
            continue
        for value in (parts[0] + "." + parts[2]).split("."):
            canonical = canonical_smiles(value)
            if canonical and _heavy_atom_count(canonical) >= 8:
                needles.add(canonical)
    return needles


def _heavy_atom_count(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    return int(molecule.GetNumHeavyAtoms()) if molecule is not None else 0


def _public_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _json_mapping(value, "strategy_pilot_source_snapshot_invalid")
    allowed = {
        "data_base_url",
        "data_version",
        "manifest_sha256",
        "index_sha256",
        "official_repository_commit",
        "paper_version",
        "manifest_counts",
    }
    return {key: source[key] for key in sorted(set(source) & allowed)}


def _json_mapping(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyClosurePilotError(reason)
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StrategyClosurePilotError(reason) from exc


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _digest(result)
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


__all__ = [
    "DEFAULT_ACCEPTANCE",
    "DEFAULT_BUDGET",
    "SELECTION_ALGORITHM",
    "STRATEGY_CLOSURE_EVALUATOR_PACK_SCHEMA",
    "STRATEGY_CLOSURE_PILOT_PROTOCOL_SCHEMA",
    "StrategyClosurePilotError",
    "compile_strategy_closure_pilot",
    "compile_strategy_closure_leakage_pack",
    "external_bundle_for_case",
]
