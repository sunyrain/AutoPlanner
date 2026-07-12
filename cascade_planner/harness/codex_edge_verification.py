"""Materialize and independently verify Codex consensus hyperedges.

Codex child agents produce advisory product/precursor hypotheses.  This module
is the explicit bridge from those hypotheses to the deterministic reaction
proof ladder.  Missing atom mapping, source authority or stock evidence is
recorded as work to do; it is never silently converted into a solved claim.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.harness.reaction_step_verifier import (
    REACTION_STEP_VERIFIER_VERSION,
    canonical_reaction_digest,
    verify_reaction_route,
    verify_reaction_step,
)
from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    independent_source_group,
)


RDLogger.DisableLog("rdApp.*")

CODEX_EDGE_VERIFICATION_SCHEMA = "codex_edge_verification_report.v1"
REACTION_CANDIDATE_SCHEMA = "materialized_reaction_candidate.v1"
EDGE_EVIDENCE_BINDING_SET_SCHEMA = "edge_evidence_binding_set.v1"
AtomMapper = Callable[[list[str]], list[str | None]]
_RXNMAPPER_INFERENCE_LOCK = threading.RLock()
_RXNMAPPER_RESULT_CACHE_MAXSIZE = 2048
_RXNMAPPER_RESULT_CACHE: OrderedDict[str, str] = OrderedDict()
_EDGE_WORK_CACHE_SCHEMA = "codex_edge_verification_work_cache.v1"
_EDGE_WORK_INPUT_SCHEMA = "codex_edge_verification_work_input.v1"
_DEFAULT_MAPPER_CONTRACT_VERSION = "optional_rxnmapper_attention_guided.v1"


def verify_codex_consensus_graph(
    graph: Mapping[str, Any],
    *,
    exact_rows: Iterable[Mapping[str, Any]] = (),
    stock_closed_smiles: Iterable[str] = (),
    atom_mapper: AtomMapper | None = None,
    enable_optional_rxnmapper: bool = True,
    work_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Materialize every graph edge and run the deterministic proof ladder.

    When ``work_dir`` is supplied, default-mapper work is cached per exact
    edge.  Injected mappers are intentionally opaque and bypass this cache.
    """

    steps = [dict(row) for row in graph.get("steps") or [] if isinstance(row, Mapping)]
    exact_by_signature = _exact_rows_by_signature(exact_rows)
    stock_closed = {
        value for value in (_canonical_smiles(item) for item in stock_closed_smiles) if value
    }
    candidates = [
        _materialize_candidate(step, exact_by_signature=exact_by_signature)
        for step in steps
    ]
    unmapped_indices = [
        index for index, candidate in enumerate(candidates) if not _mapped_reaction(candidate)
    ]

    mapper_status = {
        "attempted": False,
        "backend": "none",
        "request_count": 0,
        "mapped_count": 0,
        "reasons": [],
    }
    selected_mapper = atom_mapper
    injected_mapper = selected_mapper is not None
    if selected_mapper is not None:
        mapper_status["backend"] = "injected_atom_mapper"
    if selected_mapper is None and enable_optional_rxnmapper and unmapped_indices:
        # Probe only the stable backend contract before cache lookup.  Loading
        # RXNMapper also loads transformer weights and is required only when at
        # least one exact edge misses the durable cache.
        mapper_status = _optional_rxnmapper_contract_status()

    cache_root = (
        Path(work_dir).expanduser().resolve() / "codex_edge_work_cache"
        if work_dir is not None
        else None
    )
    cache_eligible = bool(cache_root is not None and not injected_mapper)
    edge_results: list[tuple[dict[str, Any], dict[str, Any]] | None] = [
        None for _ in candidates
    ]
    cache_records: list[dict[str, Any]] = []
    cache_bindings: list[dict[str, Any]] = []
    cache_paths: list[Path | None] = []
    miss_indices: list[int] = []
    cache_hit_count = 0
    cache_invalid_count = 0
    for index, candidate in enumerate(candidates):
        input_binding = _edge_work_input_binding(
            candidate,
            stock_closed=stock_closed,
            mapper_status=mapper_status,
            enable_optional_rxnmapper=enable_optional_rxnmapper,
            injected_mapper=injected_mapper,
        )
        input_digest = _digest(input_binding)
        cache_path = (
            cache_root / "sha256" / input_digest[:2] / f"{input_digest}.json"
            if cache_root is not None
            else None
        )
        cache_bindings.append(input_binding)
        cache_paths.append(cache_path)
        cache_record = {
            "step_id": str(candidate.get("step_id") or ""),
            "input_sha256": input_digest,
            "ref": str(cache_path) if cache_path is not None else "",
            "status": "bypass",
            "reasons": [],
        }
        if cache_eligible and cache_path is not None:
            cached, reasons = _load_edge_work_cache(
                cache_path,
                expected_input=input_binding,
                expected_candidate=candidate,
            )
            if cached is not None:
                edge_results[index] = cached
                cache_record["status"] = "hit"
                cache_hit_count += 1
            else:
                cache_record["status"] = "miss"
                cache_record["reasons"] = reasons
                cache_invalid_count += int(cache_path.exists())
                miss_indices.append(index)
        else:
            miss_indices.append(index)
        cache_records.append(cache_record)

    mapper_indices = [
        index
        for index in miss_indices
        if not _mapped_reaction(candidates[index])
    ]
    unmapped_reactions = [
        str(candidates[index].get("reaction_smiles") or "")
        for index in mapper_indices
    ]
    mapper_execution_cacheable = True
    if (
        not injected_mapper
        and enable_optional_rxnmapper
        and mapper_indices
        and mapper_status.get("backend") == "rxnmapper"
        and not mapper_status.get("reasons")
    ):
        selected_mapper, mapper_status = _optional_rxnmapper()
        if selected_mapper is None:
            mapper_execution_cacheable = False
    mapper_status["request_count"] = len(unmapped_reactions)
    if selected_mapper is not None and unmapped_reactions:
        mapper_status["attempted"] = True
        mapper_status.setdefault("backend", "injected_atom_mapper")
        try:
            mapped_rows = list(selected_mapper(unmapped_reactions))
        except Exception as exc:  # optional model/backend boundary
            mapped_rows = []
            mapper_execution_cacheable = False
            mapper_status.setdefault("reasons", []).append(
                f"atom_mapper_error:{type(exc).__name__}:{exc}"
            )
        if len(mapped_rows) != len(unmapped_reactions):
            mapper_execution_cacheable = False
            mapper_status.setdefault("reasons", []).append("atom_mapper_result_count_mismatch")
        else:
            for candidate_index, mapped in zip(mapper_indices, mapped_rows):
                text = str(mapped or "").strip()
                if text and ":" in text and ">>" in text:
                    candidates[candidate_index]["atom_mapped_reaction_smiles"] = text
                    candidates[candidate_index]["mapping_source"] = str(
                        mapper_status.get("backend") or "injected_atom_mapper"
                    )
                    mapper_status["mapped_count"] = int(mapper_status.get("mapped_count") or 0) + 1

    for index in miss_indices:
        result = _verify_materialized_edge(candidates[index], stock_closed=stock_closed)
        edge_results[index] = result
        can_persist = bool(
            cache_eligible
            and cache_paths[index] is not None
            and (
                index not in mapper_indices
                or selected_mapper is None
                or mapper_execution_cacheable
            )
        )
        if can_persist:
            _write_edge_work_cache(
                cache_paths[index],
                input_binding=cache_bindings[index],
                result=result,
            )
            cache_records[index]["ref"] = str(cache_paths[index])
        elif cache_eligible and index in mapper_indices:
            cache_records[index]["reasons"] = sorted(
                {
                    *cache_records[index]["reasons"],
                    "mapper_execution_not_cacheable",
                }
            )

    verified_results = [result for result in edge_results if result is not None]
    rows = [dict(result[0]) for result in verified_results]
    reaction_validations = [dict(result[1]) for result in verified_results]

    payload = {
        "schema_version": CODEX_EDGE_VERIFICATION_SCHEMA,
        "graph_schema_version": str(graph.get("schema_version") or ""),
        "target_smiles": str(graph.get("target_smiles") or ""),
        "edge_count": len(rows),
        "materialized_edge_count": sum(
            bool(row["step_proof"].get("checks", {}).get("structures_materialized"))
            for row in rows
        ),
        "mapped_edge_count": sum(
            bool(row["step_proof"].get("checks", {}).get("mapped_reaction_present"))
            for row in rows
        ),
        "reaction_validated_edge_count": sum(row["reaction_validated"] for row in rows),
        "proof_closed_edge_count": sum(row["proof_closed"] for row in rows),
        "trusted_precedent_binding_count": sum(
            int((row.get("edge_evidence_binding_set") or {}).get("trusted_binding_count") or 0)
            for row in rows
        ),
        "corroborated_edge_count": sum(
            (row.get("edge_evidence_binding_set") or {}).get("corroborated") is True
            for row in rows
        ),
        "edge_verifications": rows,
        "reaction_validations": reaction_validations,
        "atom_mapper": mapper_status,
        "work_cache": {
            "schema_version": "codex_edge_work_cache_summary.v1",
            "enabled": cache_root is not None,
            "eligible": cache_eligible,
            "root": str(cache_root) if cache_root is not None else "",
            "hit_count": cache_hit_count,
            "miss_count": len(miss_indices) if cache_eligible else 0,
            "bypass_count": len(miss_indices) if not cache_eligible else 0,
            "invalid_entry_count": cache_invalid_count,
            "result_refs": cache_records,
            "injected_mapper_cache_reuse_disabled": injected_mapper,
        },
        "no_solved_claim": True,
        "semantics": {
            "agent_proposals_are_not_authority": True,
            "mapping_and_transform_reapply_required_for_l2": True,
            "stock_and_reaction_proof_are_separate": True,
            "multisource_corroboration_is_edge_local": True,
            "corroboration_does_not_upgrade_proof_tier": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def project_edge_evidence_binding_sets(
    verification_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a digest-checked sidecar keyed by chemistry and graph step.

    The route hypergraph remains structurally content addressed.  Evidence
    attachments are a separate, current-host-derived projection so adding a
    second precedent cannot duplicate or mutate the chemical edge.
    """

    by_reaction_digest: dict[str, dict[str, Any]] = {}
    by_step_id: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    for raw in verification_report.get("edge_verifications") or []:
        if not isinstance(raw, Mapping):
            continue
        edge = dict(raw)
        binding_set = dict(edge.get("edge_evidence_binding_set") or {})
        supplied_digest = str(binding_set.get("content_sha256") or "")
        digest_payload = {
            key: value for key, value in binding_set.items() if key != "content_sha256"
        }
        reaction_digest = str(binding_set.get("reaction_digest") or "")
        step_id = str(edge.get("step_id") or "")
        reasons: list[str] = []
        if binding_set.get("schema_version") != EDGE_EVIDENCE_BINDING_SET_SCHEMA:
            reasons.append("invalid_edge_evidence_binding_set_schema")
        if not reaction_digest:
            reasons.append("edge_evidence_reaction_digest_missing")
        if not supplied_digest or supplied_digest != _digest(digest_payload):
            reasons.append("edge_evidence_binding_set_digest_invalid")
        if reaction_digest in by_reaction_digest:
            reasons.append("duplicate_edge_evidence_reaction_digest")
        if reasons:
            rejected.append({"step_id": step_id, "reasons": sorted(set(reasons))})
            continue
        by_reaction_digest[reaction_digest] = binding_set
        if step_id:
            by_step_id[step_id] = reaction_digest

    payload: dict[str, Any] = {
        "schema_version": "edge_evidence_binding_projection.v1",
        "by_reaction_digest": dict(sorted(by_reaction_digest.items())),
        "by_step_id": dict(sorted(by_step_id.items())),
        "edge_count": len(by_reaction_digest),
        "corroborated_edge_count": sum(
            row.get("corroborated") is True for row in by_reaction_digest.values()
        ),
        "rejected": rejected,
        "semantics": {
            "sidecar_does_not_mutate_hypergraph_identity": True,
            "current_host_projection": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _edge_work_input_binding(
    candidate: Mapping[str, Any],
    *,
    stock_closed: set[str],
    mapper_status: Mapping[str, Any],
    enable_optional_rxnmapper: bool,
    injected_mapper: bool,
) -> dict[str, Any]:
    materialized = _json_value(dict(candidate))
    precursors = [
        value
        for value in (
            _canonical_smiles(item) for item in materialized.get("reactant_smiles") or []
        )
        if value
    ]
    if _mapped_reaction(materialized):
        mapper_contract = {
            "mode": "materialized_mapping_passthrough",
            "contract_version": "materialized_mapping_passthrough.v1",
            "available": True,
            "backend": str(materialized.get("mapping_source") or "materialized_input"),
            "reasons": [],
        }
    elif injected_mapper:
        mapper_contract = {
            "mode": "opaque_injected_mapper",
            "contract_version": "opaque_injected_mapper.not_cacheable",
            "available": True,
            "backend": "injected_atom_mapper",
            "reasons": [],
        }
    elif not enable_optional_rxnmapper:
        mapper_contract = {
            "mode": "mapping_disabled",
            "contract_version": "mapping_disabled.v1",
            "available": False,
            "backend": "none",
            "reasons": [],
        }
    else:
        mapper_contract = {
            "mode": "default_optional_mapper",
            "contract_version": _DEFAULT_MAPPER_CONTRACT_VERSION,
            "available": mapper_status.get("backend") == "rxnmapper"
            and not mapper_status.get("reasons"),
            "backend": str(mapper_status.get("backend") or "none"),
            "reasons": sorted(str(item) for item in mapper_status.get("reasons") or []),
        }
    return {
        "schema_version": _EDGE_WORK_INPUT_SCHEMA,
        "step_id": str(materialized.get("step_id") or ""),
        "edge_identity": {
            "product_smiles": str(materialized.get("product_smiles") or ""),
            "precursor_smiles": sorted(precursors),
        },
        "materialized_candidate": materialized,
        "materialized_candidate_sha256": _digest(materialized),
        "precursor_stock_state": [
            {"smiles": value, "closed": value in stock_closed}
            for value in sorted(precursors)
        ],
        "graph_and_stock_closed": bool(
            precursors and all(value in stock_closed for value in precursors)
        ),
        "reaction_step_verifier_version": REACTION_STEP_VERIFIER_VERSION,
        "mapper_contract": mapper_contract,
    }


def _verify_materialized_edge(
    candidate: Mapping[str, Any],
    *,
    stock_closed: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialized = _json_value(dict(candidate))
    precursors = [
        value
        for value in (
            _canonical_smiles(item) for item in materialized.get("reactant_smiles") or []
        )
        if value
    ]
    graph_and_stock_closed = bool(
        precursors and all(value in stock_closed for value in precursors)
    )
    proof = verify_reaction_step(
        materialized,
        graph_and_stock_closed=graph_and_stock_closed,
    )
    route_validation = verify_reaction_route(
        [materialized],
        graph_and_stock_closed=graph_and_stock_closed,
    )
    # The portfolio consumer must not trust a detached proof object whose
    # booleans and digest can both be rewritten by its producer.  Carry the
    # materialized candidate and mapper output alongside a comparison copy
    # generated with stock authority deliberately disabled.  The consumer
    # re-runs the current host verifier before granting any edge level.
    replay_step_proof = verify_reaction_step(
        materialized,
        graph_and_stock_closed=False,
    )
    replay_route_validation = verify_reaction_route(
        [materialized],
        graph_and_stock_closed=False,
    )
    supplemental_validation = {
        "schema_version": "supplemental_reaction_validation.v2",
        "materialized_candidate": dict(materialized),
        "mapper_output": {
            "schema_version": "atom_mapper_output.v1",
            "input_reaction_smiles": str(materialized.get("reaction_smiles") or ""),
            "mapped_reaction_smiles": str(
                materialized.get("atom_mapped_reaction_smiles") or ""
            ),
            "mapping_source": str(materialized.get("mapping_source") or ""),
        },
        "claimed_step_proof": replay_step_proof,
        "claimed_route_validation": replay_route_validation,
        "stock_authority_disabled_for_replay": True,
        "no_solved_claim": True,
    }
    supplemental_validation["content_sha256"] = _digest(supplemental_validation)
    tasks = _proof_tasks(
        materialized,
        proof,
        graph_and_stock_closed=graph_and_stock_closed,
    )
    edge = {
        "schema_version": "codex_edge_verification.v1",
        "step_id": str(materialized.get("step_id") or ""),
        "product_smiles": str(materialized.get("product_smiles") or ""),
        "reactant_smiles": list(materialized.get("reactant_smiles") or []),
        "materialized_candidate": materialized,
        "reaction_validation": route_validation,
        "step_proof": proof,
        "proof_level": str(proof.get("proof_level") or "L0_materialized"),
        "reaction_validated": proof.get("accepted") is True,
        "stock_closed": graph_and_stock_closed,
        "proof_closed": bool(proof.get("accepted") is True and graph_and_stock_closed),
        "edge_evidence_binding_set": dict(
            materialized.get("edge_evidence_binding_set") or {}
        ),
        "required_tasks": tasks,
    }
    return edge, supplemental_validation


def _load_edge_work_cache(
    path: Path,
    *,
    expected_input: Mapping[str, Any],
    expected_candidate: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], dict[str, Any]] | None, list[str]]:
    if not path.is_file():
        return None, ["cache_entry_missing"]
    reasons: list[str] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        row = dict(raw) if isinstance(raw, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, ["cache_entry_unreadable"]
    digest_payload = dict(row)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    expected_input_json = _json_value(dict(expected_input))
    expected_input_digest = _digest(expected_input_json)
    if row.get("schema_version") != _EDGE_WORK_CACHE_SCHEMA:
        reasons.append("cache_entry_schema_mismatch")
    if row.get("input_sha256") != expected_input_digest:
        reasons.append("cache_entry_input_digest_mismatch")
    if row.get("input_binding") != expected_input_json:
        reasons.append("cache_entry_input_binding_mismatch")
    if path.stem != expected_input_digest:
        reasons.append("cache_entry_path_digest_mismatch")
    if (
        not recorded_digest
        or recorded_digest != _digest(digest_payload)
    ):
        reasons.append("cache_entry_content_digest_invalid")
    result = row.get("result")
    if not isinstance(result, dict):
        reasons.append("cache_entry_result_missing")
        result = {}
    edge = result.get("edge_verification")
    supplemental = result.get("supplemental_validation")
    if not isinstance(edge, dict) or not isinstance(supplemental, dict):
        reasons.append("cache_entry_result_shape_invalid")
        return None, sorted(set(reasons))
    cached_candidate = edge.get("materialized_candidate")
    if not isinstance(cached_candidate, dict) or not _cached_candidate_matches_input(
        cached_candidate,
        expected_candidate=expected_candidate,
    ):
        reasons.append("cache_entry_materialized_candidate_mismatch")
    if reasons:
        return None, sorted(set(reasons))
    stock_closed = {
        str(item.get("smiles") or "")
        for item in expected_input_json.get("precursor_stock_state") or []
        if isinstance(item, dict) and item.get("closed") is True
    }
    replayed_edge, replayed_supplemental = _verify_materialized_edge(
        cached_candidate,
        stock_closed=stock_closed,
    )
    if (
        _json_value(edge) != _json_value(replayed_edge)
        or _json_value(supplemental) != _json_value(replayed_supplemental)
    ):
        return None, ["cache_entry_not_equal_to_current_host_replay"]
    if row.get("result_sha256") != _digest(result):
        return None, ["cache_entry_result_digest_invalid"]
    return (replayed_edge, replayed_supplemental), []


def _cached_candidate_matches_input(
    candidate: Mapping[str, Any],
    *,
    expected_candidate: Mapping[str, Any],
) -> bool:
    cached = _json_value(dict(candidate))
    expected = _json_value(dict(expected_candidate))
    if _mapped_reaction(expected):
        return cached == expected
    cached.pop("atom_mapped_reaction_smiles", None)
    cached.pop("mapping_source", None)
    return cached == expected


def _write_edge_work_cache(
    path: Path | None,
    *,
    input_binding: Mapping[str, Any],
    result: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    if path is None:
        return
    normalized_input = _json_value(dict(input_binding))
    result_payload = {
        "edge_verification": _json_value(result[0]),
        "supplemental_validation": _json_value(result[1]),
    }
    payload = {
        "schema_version": _EDGE_WORK_CACHE_SCHEMA,
        "input_sha256": _digest(normalized_input),
        "input_binding": normalized_input,
        "result_sha256": _digest(result_payload),
        "result": result_payload,
        "semantics": {
            "current_host_replay_required_on_read": True,
            "injected_mapper_outputs_are_not_cacheable": True,
            "cache_entry_is_not_parent_route_proof": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    )


def _materialize_candidate(
    step: Mapping[str, Any],
    *,
    exact_by_signature: Mapping[
        tuple[str, tuple[str, ...]], list[Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    product = _canonical_smiles(step.get("product_smiles"))
    reactants = sorted(
        value
        for value in (
            _canonical_smiles(item) for item in step.get("precursor_smiles") or []
        )
        if value
    )
    signature = (product, tuple(reactants))
    exact_rows = [
        dict(row)
        for row in exact_by_signature.get(signature) or []
        if isinstance(row, Mapping)
    ]
    binding_set = _edge_evidence_binding_set(
        product=product,
        reactants=reactants,
        exact_rows=exact_rows,
    )
    primary_row_sha256 = str(binding_set.get("primary_row_sha256") or "")
    exact = next(
        (
            row
            for row in exact_rows
            if _digest(_json_value(row)) == primary_row_sha256
        ),
        exact_rows[0] if exact_rows else {},
    )
    candidate = {
        "schema_version": REACTION_CANDIDATE_SCHEMA,
        "step_id": str(step.get("step_id") or ""),
        "product_smiles": product,
        "reactant_smiles": reactants,
        "reaction_smiles": f"{'.'.join(reactants)}>>{product}" if product and reactants else "",
        "reaction_family": str(step.get("reaction_family") or ""),
        "conditions": list(step.get("conditions") or []),
        "source_refs": [str(value) for value in step.get("source_refs") or []],
        "evidence_refs": [str(value) for value in step.get("evidence_refs") or []],
        "source_detail_exact_step": False,
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "advisory_input": True,
        "edge_evidence_binding_set": binding_set,
    }
    if exact:
        candidate.update(
            {
                key: exact[key]
                for key in (
                    "source_template_id",
                    "source_detail_exact_step",
                    "relation_type",
                    "source_ref",
                    "exact_step_validation",
                    "source_evidence",
                )
                if key in exact
            }
        )
        mapped = _mapped_reaction(exact)
        if mapped:
            candidate["atom_mapped_reaction_smiles"] = mapped
            candidate["mapping_source"] = "exact_literature_row"
        exact_conditions = exact.get("conditions")
        if exact_conditions:
            candidate["conditions"] = exact_conditions
        candidate["exact_row_id"] = str(
            exact.get("row_id") or exact.get("source_template_id") or ""
        )
    mapped_step = _mapped_reaction(step)
    if mapped_step and not candidate.get("atom_mapped_reaction_smiles"):
        candidate["atom_mapped_reaction_smiles"] = mapped_step
        candidate["mapping_source"] = "consensus_graph_step"
    return candidate


def _exact_rows_by_signature(
    values: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]]:
    rows: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        product = _canonical_smiles(
            row.get("product_smiles")
            or row.get("product")
            or row.get("target_smiles")
        )
        reactants = _row_reactants(row)
        signature = (product, tuple(sorted(reactants)))
        if product and reactants:
            rows.setdefault(signature, []).append(row)
    for signature, matches in rows.items():
        rows[signature] = sorted(matches, key=_exact_row_sort_key)
    return rows


def _exact_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source_ref = _canonical_source_ref_from_row(row)
    row_id = str(row.get("row_id") or row.get("source_template_id") or "")
    return source_ref, row_id, _digest(_json_value(dict(row)))


def _edge_evidence_binding_set(
    *,
    product: str,
    reactants: list[str],
    exact_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate every exact-row attachment without conflating it with proof.

    One chemical edge may have several document bindings.  The set is derived
    from current-host verifier results on every refresh; producer-authored
    source counts and binding flags are never trusted.
    """

    reaction_digest = canonical_reaction_digest(product, reactants)
    bindings: list[dict[str, Any]] = []
    row_by_sha256: dict[str, dict[str, Any]] = {}
    for raw in exact_rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row_sha256 = _digest(_json_value(row))
        row_by_sha256[row_sha256] = row
        verification_row = dict(row)
        verification_row["product_smiles"] = product
        verification_row["reactant_smiles"] = list(reactants)
        proof = verify_reaction_step(
            verification_row,
            graph_and_stock_closed=False,
        )
        checks = dict(proof.get("checks") or {})
        precedent = dict(proof.get("trusted_precedent_binding") or {})
        trusted = bool(
            checks.get("trusted_precedent_bound")
            and precedent.get("accepted") is True
            and str(precedent.get("reaction_digest") or "") == reaction_digest
        )
        source_ref = _canonical_source_ref_from_row(row)
        source_group = (
            independent_source_group({**row, "source_ref": source_ref})
            if trusted and source_ref
            else ""
        )
        evidence_rows = [
            _compact_materialized_source_evidence(item)
            for item in row.get("source_evidence") or []
            if isinstance(item, Mapping)
        ]
        evidence_rows = [item for item in evidence_rows if item]
        candidate_binding_id = "candidate:" + _digest(
            {
                "reaction_digest": reaction_digest,
                "row_sha256": row_sha256,
                "source_ref": source_ref,
            }
        )[:24]
        bindings.append(
            {
                "schema_version": "edge_evidence_binding.v1",
                "binding_id": str(precedent.get("binding_id") or candidate_binding_id),
                "row_id": str(row.get("row_id") or row.get("source_template_id") or ""),
                "row_sha256": row_sha256,
                "source_ref": source_ref,
                "independent_source_group": source_group,
                "status": "trusted" if trusted else "candidate_untrusted",
                "trusted": trusted,
                "authority": str(precedent.get("authority") or ""),
                "authority_id": str(precedent.get("authority_id") or ""),
                "reaction_digest": reaction_digest,
                "materialized_evidence": evidence_rows,
                "materialized_evidence_sha256": _digest(evidence_rows),
                "condition_sha256": _digest(row.get("conditions") or []),
                "proof_level": str(proof.get("proof_level") or "L0_materialized"),
                "reasons": [] if trusted else ["trusted_precedent_binding_not_revalidated"],
            }
        )

    bindings.sort(
        key=lambda row: (
            0 if row.get("trusted") is True else 1,
            str(row.get("independent_source_group") or ""),
            str(row.get("source_ref") or ""),
            str(row.get("binding_id") or ""),
            str(row.get("row_sha256") or ""),
        )
    )
    trusted_bindings = [row for row in bindings if row.get("trusted") is True]
    groups = sorted(
        {
            str(row.get("independent_source_group") or "")
            for row in trusted_bindings
            if str(row.get("independent_source_group") or "")
        }
    )
    trusted_condition_variants = sorted(
        {
            str(row.get("condition_sha256") or "")
            for row in trusted_bindings
            if str(row.get("condition_sha256") or "")
        }
    )
    payload: dict[str, Any] = {
        "schema_version": EDGE_EVIDENCE_BINDING_SET_SCHEMA,
        "reaction_digest": reaction_digest,
        "product_smiles": product,
        "reactant_smiles": list(reactants),
        "binding_count": len(bindings),
        "trusted_binding_count": len(trusted_bindings),
        "independent_trusted_source_groups": groups,
        "independent_trusted_source_group_count": len(groups),
        "corroborated": len(groups) >= 2,
        "primary_binding_id": str(
            (trusted_bindings or bindings or [{}])[0].get("binding_id") or ""
        ),
        "primary_row_sha256": str(
            (trusted_bindings or bindings or [{}])[0].get("row_sha256") or ""
        ),
        "trusted_condition_variant_count": len(trusted_condition_variants),
        "condition_variants_require_review": len(trusted_condition_variants) > 1,
        "bindings": bindings,
        "semantics": {
            "proof_tier_is_orthogonal": True,
            "corroboration_requires_two_independent_trusted_source_groups": True,
            "article_and_supporting_information_are_one_source_group": True,
            "current_host_revalidation_required": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _canonical_source_ref_from_row(row: Mapping[str, Any]) -> str:
    for value in (
        row.get("source_ref"),
        row.get("doi"),
        f"doi:{row.get('doi')}" if str(row.get("doi") or "").strip() else "",
        row.get("patent_publication"),
        row.get("url"),
    ):
        canonical = canonical_traceable_source_ref(value)
        if canonical:
            return canonical
    return ""


def _compact_materialized_source_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return {
        key: row.get(key)
        for key in (
            "source_ref",
            "document_id",
            "source_pdf_sha256",
            "page_number",
            "image_sha256",
            "artifact_ref",
        )
        if row.get(key) not in (None, "", [])
    }


def _row_reactants(row: Mapping[str, Any]) -> list[str]:
    raw = (
        row.get("reactant_smiles")
        or row.get("precursor_smiles")
        or row.get("reactants")
        or []
    )
    if isinstance(raw, str):
        values = raw.split(".")
    else:
        values = list(raw) if isinstance(raw, (list, tuple)) else []
    if not values:
        main = row.get("main_reactant") or row.get("main_reactant_smiles")
        aux = row.get("aux_reactants") or []
        values = [main] if main else []
        values.extend(aux if isinstance(aux, list) else str(aux).split("."))
    return [
        value for value in (_canonical_smiles(item) for item in values) if value
    ]


def _mapped_reaction(row: Mapping[str, Any]) -> str:
    for field in (
        "atom_mapped_reaction_smiles",
        "mapped_reaction_smiles",
        "reaction_smiles",
    ):
        value = str(row.get(field) or "").strip()
        if value and ":" in value and ">>" in value:
            return value
    return ""


def _proof_tasks(
    candidate: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    graph_and_stock_closed: bool,
) -> list[str]:
    checks = dict(proof.get("checks") or {})
    tasks: list[str] = []
    if not checks.get("mapped_reaction_present"):
        tasks.append("atom_map_materialized_reaction")
    elif not checks.get("deterministic_transform_reapplied") and not checks.get(
        "trusted_precedent_bound"
    ):
        tasks.append("bind_deterministic_transform_or_exact_precedent")
    if not candidate.get("source_detail_exact_step"):
        tasks.append("acquire_and_bind_exact_source_detail")
    if not graph_and_stock_closed:
        tasks.append("audit_all_precursors_against_trusted_stock_provider")
    return tasks


def _optional_rxnmapper() -> tuple[AtomMapper | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "attempted": False,
        "backend": "rxnmapper",
        "request_count": 0,
        "mapped_count": 0,
        "reasons": [],
    }
    try:
        from rxnmapper import RXNMapper  # type: ignore
    except Exception as exc:  # optional binary/import boundary
        if isinstance(exc, ModuleNotFoundError) and exc.name == "rxnmapper":
            status["reasons"] = ["rxnmapper_not_installed"]
        else:
            status["reasons"] = [
                f"rxnmapper_import_error:{type(exc).__name__}:{exc}"
            ]
        return None, status

    # Importing the package does not necessarily prove that its transformer
    # runtime and binary dependencies can be loaded.  Initialize the cached
    # model at this optional boundary so DLL/checkpoint/device failures become
    # an explicit audit reason instead of aborting consensus refresh.
    try:
        with _RXNMAPPER_INFERENCE_LOCK:
            model = _cached_mapper_instance(RXNMapper)
    except Exception as exc:  # optional model/binary initialization boundary
        status["reasons"] = [
            f"rxnmapper_initialization_error:{type(exc).__name__}:{exc}"
        ]
        return None, status

    def mapper(reactions: list[str]) -> list[str | None]:
        # Loading RXNMapper also loads its transformer weights.  Consensus is
        # refreshed after useful blackboard rounds, so constructing the model
        # here on every refresh would dominate latency and cause visible UI
        # stalls.  Cache one process-local instance while keeping every batch
        # result independently revalidated by the deterministic host verifier.
        with _RXNMAPPER_INFERENCE_LOCK:
            missing: list[str] = []
            for reaction in dict.fromkeys(reactions):
                found, _ = _rxnmapper_cache_get(reaction)
                if not found:
                    missing.append(reaction)
            if missing:
                results = model.get_attention_guided_atom_maps(missing)
                if len(results) != len(missing):
                    raise ValueError("rxnmapper_result_count_mismatch")
                for reaction, row in zip(missing, results):
                    _rxnmapper_cache_put(
                        reaction,
                        str(row.get("mapped_rxn") or "")
                        if isinstance(row, Mapping)
                        else "",
                    )
            mapped_rows: list[str | None] = []
            for reaction in reactions:
                _, mapped = _rxnmapper_cache_get(reaction)
                mapped_rows.append(mapped or None)
            return mapped_rows

    return mapper, status


def _optional_rxnmapper_contract_status() -> dict[str, Any]:
    """Probe the default mapper contract without importing model weights."""

    status: dict[str, Any] = {
        "attempted": False,
        "backend": "rxnmapper",
        "request_count": 0,
        "mapped_count": 0,
        "reasons": [],
    }
    try:
        available = "rxnmapper" in sys.modules or find_spec("rxnmapper") is not None
    except (ImportError, AttributeError, ValueError) as exc:
        status["reasons"] = [
            f"rxnmapper_probe_error:{type(exc).__name__}:{exc}"
        ]
        return status
    if not available:
        status["reasons"] = ["rxnmapper_not_installed"]
    return status


@lru_cache(maxsize=1)
def _cached_mapper_instance(factory: Any) -> Any:
    return factory()


def _rxnmapper_cache_get(reaction: str) -> tuple[bool, str]:
    """Return and touch an entry; caller must hold the inference lock."""

    if reaction not in _RXNMAPPER_RESULT_CACHE:
        return False, ""
    value = _RXNMAPPER_RESULT_CACHE.pop(reaction)
    _RXNMAPPER_RESULT_CACHE[reaction] = value
    return True, value


def _rxnmapper_cache_put(reaction: str, mapped: str) -> None:
    """Insert into the bounded process cache; caller holds the lock."""

    if _RXNMAPPER_RESULT_CACHE_MAXSIZE <= 0:
        return
    _RXNMAPPER_RESULT_CACHE.pop(reaction, None)
    _RXNMAPPER_RESULT_CACHE[reaction] = mapped
    while len(_RXNMAPPER_RESULT_CACHE) > _RXNMAPPER_RESULT_CACHE_MAXSIZE:
        _RXNMAPPER_RESULT_CACHE.popitem(last=False)


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
