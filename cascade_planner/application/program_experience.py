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
from cascade_planner.application.program_applicability import (
    claim_program_strategy_signature,
    compile_program_applicability_model,
    program_experience_subject_key,
)
from cascade_planner.application.route_structure_matching import (
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
    applicability_models: dict[str, dict[str, Any]] = {}
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        model = compile_program_applicability_model(candidate, records)
        if not model:
            annotated.append(candidate)
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        applicability_models[candidate_id] = model
        matched_ids.update(str(value) for value in model["matched_experience_ids"])
        counts = dict(model["evidence_counts"])
        adjustment = float(model["priority_adjustment"])
        base = float(candidate.get("priority_score") or 0.0)
        candidate["base_priority_score"] = round(base, 6)
        candidate["priority_score"] = round(min(1.0, max(0.0, base + adjustment)), 6)
        candidate["experience_memory"] = {
            "match_count": int(model["match_count"]),
            "matched_experience_ids": list(model["matched_experience_ids"]),
            "positive_observation_count": int(counts["positive"]),
            "negative_observation_count": int(counts["negative"]),
            "inconclusive_observation_count": int(counts["inconclusive"]),
            "exact_observation_count": int(model["exact_observation_count"]),
            "structural_analog_observation_count": int(
                model["structural_analog_observation_count"]
            ),
            "strongest_transfer_scope": str(model["strongest_transfer_scope"]),
            "weighted_evidence": dict(model["weighted_evidence"]),
            "applicability_score": float(model["applicability_score"]),
            "evidence_strength": float(model["evidence_strength"]),
            "confidence_score": float(model["confidence_score"]),
            "uncertainty_score": float(model["uncertainty_score"]),
            "risk_score": float(model["risk_score"]),
            "priority_adjustment": adjustment,
            "disposition": str(model["disposition"]),
            "applicability_model": {
                "schema_version": str(model["schema_version"]),
                "content_sha256": str(model["content_sha256"]),
            },
            "authority_scope": "proposal_ranking_and_validation_priority_only",
            "current_candidate_still_requires_exact_validation": True,
        }
        warnings = {str(value) for value in candidate.get("warning_codes") or []}
        warnings.add(
            "SELF_EVOLUTION_CONFLICTING_PRIOR"
            if model["disposition"] == "conflicting"
            else "SELF_EVOLUTION_NEGATIVE_PRIOR"
            if model["disposition"] == "contraindicated"
            else "SELF_EVOLUTION_POSITIVE_PRIOR"
            if model["disposition"] == "supported"
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
        "candidate_applicability_models": {
            key: applicability_models[key] for key in sorted(applicability_models)
        },
        "counts": {
            "exact_boundary_models": sum(
                row["strongest_transfer_scope"] == "exact_boundary"
                for row in applicability_models.values()
            ),
            "structural_analog_models": sum(
                row["strongest_transfer_scope"] == "structural_analog"
                for row in applicability_models.values()
            ),
            "conflicting_models": sum(
                row["disposition"] == "conflicting"
                for row in applicability_models.values()
            ),
        },
        "semantics": {
            "ranking_adjustments_are_bounded": True,
            "similarity_is_not_validation": True,
            "cross_boundary_applicability_is_weighted_and_explainable": True,
            "execution_domains_do_not_share_applicability_evidence": True,
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
    strategy = claim_program_strategy_signature(discovery, domain, subject_refs)
    identity = {
        "domain": domain,
        "subject_key": program_experience_subject_key(domain, subject_refs, strategy),
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


def _state_smiles(graph: Mapping[str, Any], state_ids: Sequence[str]) -> list[str]:
    molecules = dict(graph.get("molecules") or {})
    values: list[str] = []
    for state_id in state_ids:
        molecule_id = state_id[6:] if state_id.startswith("state:") else ""
        smiles = str(dict(molecules.get(molecule_id) or {}).get("canonical_smiles") or "")
        if smiles:
            values.append(smiles)
    return values


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
