"""Learn and reuse non-authoritative experience from durable Program Claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cascade_planner.application.experimental_claim_contracts import (
    validate_experimental_claim_set,
)
from cascade_planner.application.program_experience_store import (
    PROGRAM_EXPERIENCE_RECORD_SCHEMA,
    build_program_experience_library,
    program_experience_library_lock,
    read_program_experience_library,
    validate_program_experience_library,
    write_program_experience_library,
)
from cascade_planner.application.route_structure_matching import (
    molecule_similarity,
    structure_transition,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_EXPERIENCE_PROJECTION_SCHEMA = "program_experience_projection.v1"


def synchronize_program_experience_library(
    path: str | Path, sources: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge replay-validated Claims into cross-campaign proposal memory."""

    destination = Path(path).expanduser().resolve()
    with program_experience_library_lock(destination):
        library, error = read_program_experience_library(destination)
        if error:
            return _sync_report(destination, "blocked_library_integrity", reason=error)
        experiences = {
            str(key): dict(value)
            for key, value in dict(library.get("experiences") or {}).items()
        }
        learned: set[str] = set()
        updated: set[str] = set()
        claim_ids_before = _claim_ids(experiences.values())
        accepted_sources = 0
        rejected_sources = 0
        for raw_source in sources:
            source = dict(raw_source)
            graph = dict(source.get("graph") or {})
            discovery = dict(source.get("discovery") or {})
            claim_set = dict(source.get("claim_set") or {})
            if validate_experimental_claim_set(claim_set):
                rejected_sources += 1
                continue
            accepted_sources += 1
            for claim in dict(claim_set.get("claims") or {}).values():
                row = _experience_observation(graph, discovery, claim_set, dict(claim))
                if not row:
                    continue
                experience_id = str(row["experience_id"])
                existing = dict(experiences.get(experience_id) or {})
                merged = _merge_experience(existing, row)
                if existing == merged:
                    continue
                experiences[experience_id] = merged
                (updated if existing else learned).add(experience_id)
        claim_ids_after = _claim_ids(experiences.values())
        changed = bool(learned or updated)
        if changed:
            library = build_program_experience_library(
                experiences, generation=int(library.get("generation") or 0) + 1
            )
            write_program_experience_library(destination, library)
        return {
            **_sync_report(destination, "completed" if changed else "reused_or_empty"),
            "library_sha256": str(library.get("content_sha256") or ""),
            "generation": int(library.get("generation") or 0),
            "experience_count": len(experiences),
            "new_claim_count": len(claim_ids_after - claim_ids_before),
            "accepted_source_count": accepted_sources,
            "rejected_source_count": rejected_sources,
            "learned_experience_ids": sorted(learned),
            "updated_experience_ids": sorted(updated),
        }


def apply_program_experience(
    candidates: Sequence[Mapping[str, Any]], library: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply bounded, auditable ranking hints without granting validation."""

    reasons = validate_program_experience_library(library)
    if reasons:
        raise ValueError("program_experience_library_invalid:" + ",".join(reasons))
    records = [dict(value) for value in dict(library.get("experiences") or {}).values()]
    annotated: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        matches = [
            match
            for record in records
            if (match := _candidate_match(candidate, record)) is not None
        ]
        if not matches:
            annotated.append(candidate)
            continue
        matched_ids.update(str(value["experience_id"]) for value in matches)
        positive = sum(int(value["counts"].get("positive") or 0) for value in matches)
        negative = sum(int(value["counts"].get("negative") or 0) for value in matches)
        exact = any(value["transfer_scope"] == "exact_boundary" for value in matches)
        adjustment = _priority_adjustment(matches, positive=positive, negative=negative)
        base = float(candidate.get("priority_score") or 0.0)
        candidate["base_priority_score"] = round(base, 6)
        candidate["priority_score"] = round(min(1.0, max(0.0, base + adjustment)), 6)
        candidate["experience_memory"] = {
            "match_count": len(matches),
            "matched_experience_ids": sorted(
                str(value["experience_id"]) for value in matches
            ),
            "positive_observation_count": positive,
            "negative_observation_count": negative,
            "inconclusive_observation_count": sum(
                int(value["counts"].get("inconclusive") or 0) for value in matches
            ),
            "strongest_transfer_scope": "exact_boundary" if exact else "structural_analog",
            "priority_adjustment": adjustment,
            "disposition": (
                "conflicting"
                if positive and negative
                else "supported"
                if positive
                else "contraindicated"
                if negative
                else "inconclusive"
            ),
            "authority_scope": "proposal_ranking_and_validation_priority_only",
            "current_candidate_still_requires_exact_validation": True,
        }
        warnings = {str(value) for value in candidate.get("warning_codes") or []}
        warnings.add(
            "SELF_EVOLUTION_CONFLICTING_PRIOR"
            if positive and negative
            else "SELF_EVOLUTION_NEGATIVE_PRIOR"
            if negative
            else "SELF_EVOLUTION_POSITIVE_PRIOR"
            if positive
            else "SELF_EVOLUTION_INCONCLUSIVE_PRIOR"
        )
        candidate["warning_codes"] = sorted(warnings)
        annotated.append(candidate)
    projection = {
        "schema_version": PROGRAM_EXPERIENCE_PROJECTION_SCHEMA,
        "library_sha256": str(library.get("content_sha256") or ""),
        "library_generation": int(library.get("generation") or 0),
        "candidate_count": len(candidates),
        "matched_candidate_count": sum("experience_memory" in row for row in annotated),
        "matched_experience_ids": sorted(matched_ids),
        "semantics": {
            "ranking_adjustments_are_bounded": True,
            "similarity_is_not_validation": True,
            "negative_and_conflicting_results_remain_visible": True,
            "cannot_grant_program_validation_proof_completion_or_acceptance": True,
            "cannot_mutate_or_disable_capability_catalog": True,
        },
    }
    projection["content_sha256"] = strict_canonical_json_sha256(projection)
    return annotated, projection


def _experience_observation(
    graph: Mapping[str, Any],
    discovery: Mapping[str, Any],
    claim_set: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(claim.get("boundary") or {})
    input_states = [str(value) for value in boundary.get("input_state_ids") or []]
    output_states = [str(value) for value in boundary.get("output_state_ids") or []]
    input_smiles = _state_smiles(graph, input_states)
    output_smiles = _state_smiles(graph, output_states)
    if not input_smiles or not output_smiles:
        return {}
    domain = str(claim.get("domain") or "")
    subject_refs = dict(claim.get("subject_refs") or {})
    strategy = _claim_strategy_signature(discovery, domain, subject_refs)
    identity = {
        "domain": domain,
        "subject_key": _subject_key(domain, subject_refs, strategy),
        "input_smiles": input_smiles,
        "output_smiles": output_smiles,
    }
    experience_id = "program-experience:" + strict_canonical_json_sha256(identity)[:32]
    claim_id = str(claim.get("claim_id") or "")
    observation = {
        "claim_id": claim_id,
        "claim_sha256": str(claim.get("content_sha256") or ""),
        "run_id": str(claim_set.get("run_id") or ""),
        "route_id": str(claim_set.get("route_id") or ""),
        "program_id": str(claim.get("program_id") or ""),
        "polarity": str(claim.get("polarity") or ""),
        "outcome_status": str(claim.get("outcome_status") or ""),
        "interpretation_status": str(claim.get("interpretation_status") or ""),
        "source_validation": dict(claim.get("source_validation") or {}),
        "condition_record_ids": list(claim.get("condition_record_ids") or []),
    }
    return {
        "schema_version": PROGRAM_EXPERIENCE_RECORD_SCHEMA,
        "experience_id": experience_id,
        "domain": domain,
        "subject_refs": subject_refs,
        "strategy_signature_sha256": strategy,
        "exact_boundary": {
            "input_state_ids": input_states,
            "output_state_ids": output_states,
            "input_smiles": input_smiles,
            "output_smiles": output_smiles,
        },
        "structural_transition": structure_transition(input_smiles[0], output_smiles[0]),
        "observations": {claim_id: observation},
    }


def _merge_experience(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    observations = {
        str(key): dict(value) for key, value in dict(existing.get("observations") or {}).items()
    }
    observations.update(
        {str(key): dict(value) for key, value in dict(incoming.get("observations") or {}).items()}
    )
    counts = {
        polarity: sum(row.get("polarity") == polarity for row in observations.values())
        for polarity in ("positive", "negative", "inconclusive")
    }
    row = {
        key: incoming[key]
        for key in (
            "schema_version",
            "experience_id",
            "domain",
            "subject_refs",
            "strategy_signature_sha256",
            "exact_boundary",
            "structural_transition",
        )
    }
    row.update(
        {
            "observations": {key: observations[key] for key in sorted(observations)},
            "counts": counts,
            "disposition": (
                "conflicting"
                if counts["positive"] and counts["negative"]
                else "supported"
                if counts["positive"]
                else "contraindicated"
                if counts["negative"]
                else "inconclusive"
            ),
            "authority_scope": "proposal_memory_only",
            "semantics": {
                "every_observation_is_exact_boundary_bound": True,
                "structural_analog_transfer_is_ranking_only": True,
                "current_candidate_requires_its_own_exact_validation": True,
                "cannot_grant_proof_completion_acceptance_or_catalog_mutation": True,
            },
        }
    )
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _candidate_match(
    candidate: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any] | None:
    domain = _candidate_domain(candidate)
    if domain != record.get("domain"):
        return None
    candidate_strategy = _candidate_strategy_signature(candidate, domain)
    if _subject_key(
        domain,
        _candidate_subject_refs(candidate, domain),
        candidate_strategy,
    ) != _subject_key(
        domain,
        dict(record.get("subject_refs") or {}),
        str(record.get("strategy_signature_sha256") or ""),
    ):
        return None
    boundary = dict(candidate.get("boundary") or {})
    precursor = str(boundary.get("precursor_smiles") or "")
    product = str(boundary.get("product_smiles") or "")
    observed = dict(record.get("exact_boundary") or {})
    prior_inputs = list(observed.get("input_smiles") or [])
    prior_outputs = list(observed.get("output_smiles") or [])
    if not precursor or not product or not prior_inputs or not prior_outputs:
        return None
    exact = precursor == prior_inputs[0] and product == prior_outputs[0]
    similarity = 1.0 if exact else min(
        molecule_similarity(precursor, prior_inputs[0]),
        molecule_similarity(product, prior_outputs[0]),
    )
    current_transition = structure_transition(precursor, product)
    prior_transition = dict(record.get("structural_transition") or {})
    transition_equal = (
        current_transition.get("valid") is True
        and prior_transition.get("valid") is True
        and current_transition.get("motif_delta") == prior_transition.get("motif_delta")
        and current_transition.get("element_delta") == prior_transition.get("element_delta")
    )
    if not exact and (similarity < 0.72 or not transition_equal):
        return None
    return {
        "experience_id": str(record.get("experience_id") or ""),
        "transfer_scope": "exact_boundary" if exact else "structural_analog",
        "similarity": round(similarity, 6),
        "counts": dict(record.get("counts") or {}),
    }


def _priority_adjustment(
    matches: Sequence[Mapping[str, Any]], *, positive: int, negative: int
) -> float:
    if positive and negative:
        return 0.0
    strongest = max(float(value.get("similarity") or 0.0) for value in matches)
    exact = any(value.get("transfer_scope") == "exact_boundary" for value in matches)
    if positive:
        return round((0.12 if exact else 0.08 * strongest), 6)
    if negative:
        return round(-(0.18 if exact else 0.12 * strongest), 6)
    return 0.0


def _state_smiles(graph: Mapping[str, Any], state_ids: Sequence[str]) -> list[str]:
    molecules = dict(graph.get("molecules") or {})
    values: list[str] = []
    for state_id in state_ids:
        molecule_id = state_id[6:] if state_id.startswith("state:") else ""
        smiles = str(dict(molecules.get(molecule_id) or {}).get("canonical_smiles") or "")
        if smiles:
            values.append(smiles)
    return values


def _claim_strategy_signature(
    discovery: Mapping[str, Any], domain: str, subject_refs: Mapping[str, Any]
) -> str:
    if domain != "mechanism":
        return ""
    innovation_id = str(subject_refs.get("innovation_id") or "")
    for candidate in discovery.get("candidates") or []:
        innovation = dict(dict(candidate).get("route_innovation") or {})
        if innovation.get("innovation_id") == innovation_id:
            return _mechanism_strategy_signature(innovation)
    return ""


def _candidate_strategy_signature(candidate: Mapping[str, Any], domain: str) -> str:
    if domain != "mechanism":
        return ""
    return _mechanism_strategy_signature(dict(candidate.get("route_innovation") or {}))


def _mechanism_strategy_signature(innovation: Mapping[str, Any]) -> str:
    anchor = dict(innovation.get("anchor") or {})
    return strict_canonical_json_sha256(
        {
            "anchor_source_refs": list(anchor.get("source_refs") or []),
            "mechanistic_rationale": str(innovation.get("mechanistic_rationale") or ""),
            "elementary_steps": list(innovation.get("elementary_steps") or []),
            "falsifiable_checks": list(innovation.get("falsifiable_checks") or []),
        }
    )


def _candidate_domain(candidate: Mapping[str, Any]) -> str:
    return {
        "enzyme_window": "biocatalytic",
        "program_execution_window": "execution",
        "mechanism_one_hop": "mechanism",
    }.get(str(candidate.get("candidate_kind") or ""), "")


def _candidate_subject_refs(candidate: Mapping[str, Any], domain: str) -> dict[str, str]:
    if domain in {"biocatalytic", "execution"}:
        return {"capability_id": str(candidate.get("capability_id") or "")}
    return {}


def _subject_key(
    domain: str, subject_refs: Mapping[str, Any], strategy_signature: str
) -> str:
    if domain in {"biocatalytic", "execution"}:
        return str(subject_refs.get("capability_id") or "")
    return strategy_signature


def _claim_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(claim_id)
        for record in records
        for claim_id in dict(record.get("observations") or {})
    }


def _sync_report(path: Path, status: str, *, reason: str = "") -> dict[str, Any]:
    return {
        "schema_version": "program_experience_library_sync.v1",
        "stage": "program_experience_learning",
        "status": status,
        "library_path": str(path),
        "reason": reason,
        "semantics": {
            "learning_requires_replay_validated_claim_store_source": True,
            "memory_is_not_scientific_authority": True,
            "no_model_calls": True,
        },
    }


__all__ = [
    "PROGRAM_EXPERIENCE_PROJECTION_SCHEMA",
    "apply_program_experience",
    "synchronize_program_experience_library",
]
