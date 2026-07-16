"""Normalize baseline and innovation routes into one Program candidate contract."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_programs import (
    biocatalytic_program_bundle_oracle,
)
from cascade_planner.application.mechanism_programs import (
    compile_mechanism_program_bundle,
    mechanism_program_bundle_oracle,
)
from cascade_planner.application.mechanism_program_route_candidates import (
    compile_mechanism_program_route_candidates,
)
from cascade_planner.application.execution_programs import (
    compile_execution_program_bundle,
    execution_program_bundle_oracle,
)
from cascade_planner.application.execution_program_route_candidates import (
    compile_execution_program_route_candidates,
)
from cascade_planner.application.program_route_candidate_contracts import (
    PROGRAM_ROUTE_CANDIDATE_SCHEMA,
    PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA,
    PROGRAM_ROUTE_SOURCE_KINDS,
    ProgramRouteCandidateError,
    program_route_candidate_counts,
    program_route_candidate_set_semantics,
    validate_program_route_candidate_set,
)
from cascade_planner.application.program_route_candidate_factory import (
    build_program_route_candidate,
    canonical_route_authority_snapshot,
    canonical_route_metrics,
    normalize_strings,
    program_execution_domains,
    with_program_route_digest,
)
from cascade_planner.application.reported_program_route_candidates import (
    compile_reported_program_route_candidates,
)
from cascade_planner.application.transformation_programs import (
    program_id,
    program_projection_oracle,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def compile_program_route_candidate_set(
    graph: Mapping[str, Any],
    route: Mapping[str, Any],
    projection: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
    reported_candidate_packs: Iterable[Mapping[str, Any]] = (),
    mechanism_bundle: Mapping[str, Any] | None = None,
    mechanism_validations: Iterable[Mapping[str, Any]] = (),
    execution_bundle: Mapping[str, Any] | None = None,
    execution_validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile fallback, enzyme, mechanism, and reported routes into common rows."""

    rows = [dict(value) for value in validations]
    mechanism_rows = [dict(value) for value in mechanism_validations]
    execution_rows = [dict(value) for value in execution_validations]
    graph_value = _object(graph, "graph")
    route_value = _object(route, "route")
    projection_value = _object(projection, "projection")
    discovery_value = _object(discovery, "discovery")
    bundle_value = _object(bundle, "bundle")
    if program_projection_oracle(graph_value, projection_value).get("accepted") is not True:
        raise ProgramRouteCandidateError("program_candidate_projection_not_current")
    bundle_oracle = biocatalytic_program_bundle_oracle(
        graph_value,
        route_value,
        projection_value,
        discovery_value,
        bundle_value,
        validations=rows,
    )
    if bundle_oracle.get("accepted") is not True:
        raise ProgramRouteCandidateError("program_candidate_bundle_not_current")
    mechanism_value = (
        _object(mechanism_bundle, "mechanism_bundle")
        if mechanism_bundle is not None
        else compile_mechanism_program_bundle(
            graph_value,
            route_value,
            projection_value,
            discovery_value,
            validations=mechanism_rows,
        )
    )
    if mechanism_program_bundle_oracle(
        graph_value,
        route_value,
        projection_value,
        discovery_value,
        mechanism_value,
        validations=mechanism_rows,
    ).get("accepted") is not True:
        raise ProgramRouteCandidateError("program_candidate_mechanism_bundle_not_current")
    execution_value = (
        _object(execution_bundle, "execution_bundle")
        if execution_bundle is not None
        else compile_execution_program_bundle(
            graph_value,
            route_value,
            projection_value,
            discovery_value,
            validations=execution_rows,
        )
    )
    if execution_program_bundle_oracle(
        graph_value,
        route_value,
        projection_value,
        discovery_value,
        execution_value,
        validations=execution_rows,
    ).get("accepted") is not True:
        raise ProgramRouteCandidateError("program_candidate_execution_bundle_not_current")

    programs = dict(projection_value.get("programs") or {})
    source_route_sha256 = strict_canonical_json_sha256(route_value)
    baseline = _baseline_candidate(
        route_value,
        programs,
        source_route_sha256=source_route_sha256,
        source_projection_sha256=str(projection_value["content_sha256"]),
    )
    candidates = {baseline["candidate_id"]: baseline}
    proposals = dict(bundle_value.get("program_proposals") or {})
    for route_candidate_id, variant in sorted(
        dict(bundle_value.get("route_candidates") or {}).items()
    ):
        proposal_id = str(variant.get("superstep_program_id") or "")
        proposal = dict(proposals.get(proposal_id) or {})
        if not proposal:
            raise ProgramRouteCandidateError(
                f"program_candidate_superstep_missing:{route_candidate_id}"
            )
        candidate = _biocatalytic_candidate(
            route_value,
            programs,
            route_candidate_id=str(route_candidate_id),
            variant=dict(variant),
            proposal=proposal,
            source_route_sha256=source_route_sha256,
            source_projection_sha256=str(projection_value["content_sha256"]),
            source_bundle_sha256=str(bundle_value["content_sha256"]),
        )
        candidates[candidate["candidate_id"]] = candidate
    candidates.update(
        compile_mechanism_program_route_candidates(
            route_value,
            programs,
            mechanism_value,
            source_route_sha256=source_route_sha256,
            source_projection_sha256=str(projection_value["content_sha256"]),
            source_discovery_sha256=str(discovery_value["content_sha256"]),
        )
    )
    execution = compile_execution_program_route_candidates(
        route_value,
        programs,
        execution_value,
        source_route_sha256=source_route_sha256,
        source_projection_sha256=str(projection_value["content_sha256"]),
        source_discovery_sha256=str(discovery_value["content_sha256"]),
    )
    overlap = set(candidates).intersection(execution)
    if overlap:
        raise ProgramRouteCandidateError(
            "program_candidate_identity_collision:" + ",".join(sorted(overlap))
        )
    candidates.update(execution)
    reported = compile_reported_program_route_candidates(
        graph_value,
        route_value,
        baseline_program_ids=list(baseline["program_ids"]),
        packs=reported_candidate_packs,
    )
    overlap = set(candidates).intersection(reported)
    if overlap:
        raise ProgramRouteCandidateError(
            "program_candidate_identity_collision:" + ",".join(sorted(overlap))
        )
    candidates.update(reported)

    counts = program_route_candidate_counts(candidates)
    result = with_program_route_digest(
        {
            "schema_version": PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA,
            "run_id": str(graph_value.get("run_id") or ""),
            "route_id": str(route_value.get("route_id") or ""),
            "source_graph_revision": int(graph_value.get("revision") or 0),
            "source_graph_scientific_sha256": str(
                graph_value.get("scientific_sha256") or ""
            ),
            "source_route_sha256": source_route_sha256,
            "source_projection_sha256": str(projection_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "source_mechanism_bundle_sha256": str(
                mechanism_value["content_sha256"]
            ),
            "source_execution_bundle_sha256": str(
                execution_value["content_sha256"]
            ),
            "candidates": candidates,
            "counts": counts,
            "unmodeled_objectives": [
                "expected_success_probability",
                "purification_count",
                "feedstock_and_cofactor_cost",
                "process_mass_intensity",
            ],
            "semantics": program_route_candidate_set_semantics(),
        }
    )
    reasons = validate_program_route_candidate_set(result)
    if reasons:
        raise ProgramRouteCandidateError(";".join(reasons))
    return result


def _baseline_candidate(
    route: dict[str, Any],
    programs: dict[str, Any],
    *,
    source_route_sha256: str,
    source_projection_sha256: str,
) -> dict[str, Any]:
    edge_ids = [str(value) for value in route.get("edge_ids") or []]
    program_ids = [program_id(value) for value in edge_ids]
    if not edge_ids or any(value not in programs for value in program_ids):
        raise ProgramRouteCandidateError("program_candidate_baseline_mapping_invalid")
    metrics = canonical_route_metrics(
        route, physical=len(program_ids), chemical=len(program_ids)
    )
    identity = {
        "source_kind": "baseline",
        "route_id": route.get("route_id"),
        "program_ids": program_ids,
    }
    return build_program_route_candidate(
        candidate_id=f"program-route:baseline:{strict_canonical_json_sha256(identity)[:24]}",
        source_kind="baseline",
        source_route_id=str(route.get("route_id") or ""),
        program_ids=program_ids,
        fallback_program_ids=program_ids,
        substitution_program_ids=[],
        execution_domains=program_execution_domains(program_ids, programs),
        metrics=metrics,
        shadow_optimizer=True,
        specialized_validation_ids=[],
        source_refs=normalize_strings(route.get("reported_source_refs")),
        source_artifact_sha256s=[source_route_sha256, source_projection_sha256],
        warning_codes=normalize_strings(route.get("warning_codes")),
        authority_snapshot=canonical_route_authority_snapshot(route),
    )


def _biocatalytic_candidate(
    route: dict[str, Any],
    programs: dict[str, Any],
    *,
    route_candidate_id: str,
    variant: dict[str, Any],
    proposal: dict[str, Any],
    source_route_sha256: str,
    source_projection_sha256: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    selected = [str(value) for value in variant.get("selected_program_ids") or []]
    fallback = [str(value) for value in variant.get("fallback_program_ids") or []]
    proposal_id = str(proposal.get("program_id") or "")
    if not selected or proposal_id not in selected or any(
        value not in programs and value != proposal_id for value in selected
    ):
        raise ProgramRouteCandidateError(
            f"program_candidate_variant_mapping_invalid:{route_candidate_id}"
        )
    validated = variant.get("substitution_validated") is True
    metrics = canonical_route_metrics(
        route,
        physical=int(variant.get("physical_step_count") or 0),
        chemical=int(variant.get("chemical_step_equivalent_count") or 0),
        replaced_edges=[str(value) for value in variant.get("replaced_edge_ids") or []],
        substitution_validated=validated,
        specialized_validation_deficit=0 if validated else 1,
        cofactor_systems=int(
            bool(dict(proposal.get("cofactor_and_carrier_ledger") or {}).get("requirements"))
        ),
    )
    validation_gate = dict(proposal.get("validation_gate") or {})
    refs = dict(proposal.get("claim_refs") or {})
    return build_program_route_candidate(
        candidate_id=f"program-route:option:{strict_canonical_json_sha256({'route_candidate_id': route_candidate_id, 'proposal_id': proposal_id})[:24]}",
        source_kind="biocatalytic",
        source_route_id=str(route.get("route_id") or ""),
        program_ids=selected,
        fallback_program_ids=fallback,
        substitution_program_ids=[proposal_id],
        execution_domains=sorted(
            {
                *program_execution_domains(
                    [value for value in selected if value in programs], programs
                ),
                "enzymatic",
            }
        ),
        metrics=metrics,
        shadow_optimizer=validated,
        specialized_validation_ids=[
            str(value) for value in validation_gate.get("accepted_validation_ids") or []
        ],
        source_refs=sorted(
            {
                *normalize_strings(route.get("reported_source_refs")),
                *normalize_strings(refs.get("precedent_refs")),
                *normalize_strings(refs.get("exact_substrate_claim_refs")),
            }
        ),
        source_artifact_sha256s=[
            source_route_sha256,
            source_projection_sha256,
            source_bundle_sha256,
        ],
        warning_codes=normalize_strings(proposal.get("warning_codes")),
        authority_snapshot=canonical_route_authority_snapshot(route),
    )


def _object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ProgramRouteCandidateError(f"program_candidate_{label}_not_strict_json") from exc
    if not isinstance(copied, dict):
        raise ProgramRouteCandidateError(f"program_candidate_{label}_not_object")
    return copied


__all__ = [
    "PROGRAM_ROUTE_CANDIDATE_SCHEMA",
    "PROGRAM_ROUTE_CANDIDATE_SET_SCHEMA",
    "PROGRAM_ROUTE_SOURCE_KINDS",
    "ProgramRouteCandidateError",
    "compile_program_route_candidate_set",
    "validate_program_route_candidate_set",
]
